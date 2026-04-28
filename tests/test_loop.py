from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from affine.chain import Miner
from affine.config import Config, EnvSpec
from affine.evidence import EvidenceStore, Row
from affine.irt import Priors
from affine.loop import (
    SLOT_DEAD, Chain, LoopState, MinerStates, Reign, _apply_skip, _cancellable,
    art_key, _drop_next_chal, dwell, _fit, _load_envs, _provision, _provision_pair,
    _respondents, _seed, _slot_from_task, _start_prefetch, _take_prefetched, static_chain,
)
from affine.verdict import DuelStatus
from affine.vllm import SlotProvisionFailed


def _miner(uid, model=None, rev="r"):
    return Miner(uid=uid, hotkey=f"hk{uid}", model=model or f"m{uid}",
                 revision=rev, block=uid)


def _row(**kw):
    d = dict(m=0, r="r", e="E", c=0, p=1, t=0, l=1.0)
    d.update(kw)
    return Row(**d)


def test_states_filters_by_excluded_model_and_per_uid_skip():
    s = MinerStates({"banned"})
    s.mark_durable(1, ("okmodel", "badrev"), "crashloop")
    kept = s.filter([
        _miner(0, model="banned"),
        _miner(1, model="okmodel", rev="badrev"),
        _miner(2, model="okmodel", rev="goodrev"),
    ])
    assert [m.uid for m in kept] == [2]


def test_states_durable_persists_and_reloads(tmp_path):
    path = tmp_path / "skip.jsonl"
    s = MinerStates(path=path)
    s.mark_durable(7, ("modelA", "rev1"), "crashloop")
    s.mark_attempted(("modelB", "rev2"))

    s2 = MinerStates(path=path)
    assert s2.filter([_miner(7, model="modelA", rev="rev1")]) == []
    keep = _miner(8, model="modelB", rev="rev2")
    assert s2.filter([keep]) == [keep]   # ATTEMPTED is in-memory, doesn't persist
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert lines == [{"uid": 7, "model": "modelA", "revision": "rev1", "reason": "crashloop"}]


def test_states_durable_mark_is_idempotent(tmp_path):
    path = tmp_path / "skip.jsonl"
    s = MinerStates(path=path)
    s.mark_durable(1, ("m", "r"), "crashloop")
    s.mark_durable(1, ("m", "r"), "crashloop")
    assert len(path.read_text().splitlines()) == 1


def test_reign_state_roundtrip(tmp_path):
    p = tmp_path / "reign.json"
    assert Reign.load(p) is None
    Reign((7, "rev-abc"), 12345, 7).save(p)
    assert Reign.load(p) == Reign((7, "rev-abc"), 12345, 7)


def test_reign_state_legacy_missing_published_uid(tmp_path):
    """Old reign.json files predate last_published_uid. Loader must default
    last_published_uid=None when the field is missing — first iteration after
    upgrade then re-publishes once, becomes idempotent thereafter."""
    p = tmp_path / "reign.json"
    p.write_text('{"uid": 3, "revision": "r", "reign_start": 100}')
    assert Reign.load(p) == Reign((3, "r"), 100, None)


def test_reign_state_corrupt_file_resets(tmp_path):
    """A truncated/garbled reign.json must not crash startup. Cold path =
    re-elect from chain state, lose the in-flight reign — strictly better
    than refusing to boot."""
    p = tmp_path / "reign.json"
    p.write_text("not-json")
    assert Reign.load(p) is None
    p.write_text('{"uid": "not-int", "revision": "r", "reign_start": 1}')
    assert Reign.load(p) is None
    # Non-dict JSON: array, scalar, string. d.get(...) on non-dict raises
    # AttributeError; without explicit isinstance(d, dict) check, the load
    # crashes startup instead of falling back to cold-elect.
    p.write_text("[1, 2, 3]")
    assert Reign.load(p) is None
    p.write_text("42")
    assert Reign.load(p) is None
    p.write_text('"a string"')
    assert Reign.load(p) is None


def test_reign_state_atomic_replace(tmp_path):
    """Reign.save must not leave a half-written file. A SIGKILL between
    open() and write() of a direct overwrite would zero the file; we use a
    temp + os.replace, which is atomic on POSIX."""
    p = tmp_path / "reign.json"
    Reign((1, "r1"), 100).save(p)
    Reign((2, "r2"), 200).save(p)
    assert not (tmp_path / "reign.json.tmp").exists()
    assert Reign.load(p) == Reign((2, "r2"), 200, None)


def test_states_durable_write_fsyncs(tmp_path, monkeypatch):
    """SIGKILL between buffered f.write and OS flush would lose the durable mark
    and re-admit a known-crashloop artifact on restart. Single os.write + fsync."""
    import os
    s = MinerStates(path=tmp_path / "skip.jsonl")
    written, fsynced = [], []
    real_write, real_fsync = os.write, os.fsync
    monkeypatch.setattr(os, "write", lambda fd, data: (written.append(data), real_write(fd, data))[1])
    monkeypatch.setattr(os, "fsync", lambda fd: (fsynced.append(fd), real_fsync(fd))[1])
    s.mark_durable(7, ("m", "r"), "crashloop")
    assert len(written) == 1
    assert written[0] == (json.dumps({"uid": 7, "model": "m", "revision": "r", "reason": "crashloop"}) + "\n").encode()
    assert len(fsynced) == 1


def test_states_durable_short_write_truncates_to_pre_size(tmp_path, monkeypatch):
    """A short os.write must roll the file back to its pre-write size. Without
    ftruncate-back, the next load parses the orphan as JSON, drops it as malformed,
    and the artifact is silently re-admitted."""
    import os
    s = MinerStates(path=tmp_path / "skip.jsonl")
    s.mark_durable(1, ("first", "rev-a"), "crashloop")
    pre_size = (tmp_path / "skip.jsonl").stat().st_size

    real_write, real_open = os.write, os.open
    target_fd = [None]
    def open_capture(path, *a, **kw):
        fd = real_open(path, *a, **kw)
        if str(path) == str(s.path): target_fd[0] = fd
        return fd
    def short_write(fd, data):
        if fd == target_fd[0]:
            return real_write(fd, data[: len(data) // 2])
        return real_write(fd, data)
    monkeypatch.setattr(os, "open", open_capture)
    monkeypatch.setattr(os, "write", short_write)
    with pytest.raises(OSError, match="short write"):
        s.mark_durable(2, ("second", "rev-b"), "crashloop")
    assert (tmp_path / "skip.jsonl").stat().st_size == pre_size
    reload = MinerStates(path=tmp_path / "skip.jsonl")
    assert reload.is_durable(1, ("first", "rev-a"))
    assert not reload.is_attempted(("second", "rev-b")) and not reload.is_durable(2, ("second", "rev-b"))


def test_states_durable_disk_failure_does_not_mutate_memory(tmp_path):
    """In-memory mutation must not precede the disk write. If the write raises,
    in-memory must stay UNTRIED so a retry can succeed cleanly."""
    path = tmp_path / "subdir" / "skip.jsonl"
    path.parent.write_text("not-a-dir")
    s = MinerStates(path=path)
    with pytest.raises((FileExistsError, NotADirectoryError, OSError)):
        s.mark_durable(7, ("m", "r"), "crashloop")
    assert not s.is_attempted(("m", "r")) and not s.is_durable(7, ("m", "r"))


@pytest.mark.asyncio
async def test_cancellable_returns_worker_result_when_stop_idle():
    stop = asyncio.Event()
    async def work(): return 42
    assert await _cancellable(work(), stop) == 42


@pytest.mark.asyncio
async def test_cancellable_cancels_worker_when_stop_fires():
    stop = asyncio.Event()
    cancelled = asyncio.Event()
    async def work():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set(); raise
    async def trigger():
        await asyncio.sleep(0.01); stop.set()
    trigger_task = asyncio.create_task(trigger())
    with pytest.raises(asyncio.CancelledError):
        await _cancellable(work(), stop)
    assert cancelled.is_set()
    await trigger_task


@pytest.mark.asyncio
async def test_cancellable_orphan_teardown_when_outer_cancel_races_completion(monkeypatch):
    """Outer task cancellation racing worker completion: worker produces a Slot,
    asyncio.wait gets cancelled before delivering it to the caller. on_orphan
    must run, otherwise the rental survives until the next reconcile().

    The race itself is timing-sensitive, so we deterministically simulate it by
    patching asyncio.wait to raise CancelledError *after* the worker has
    completed — exactly the state the production race lands in."""
    torn_down: list = []
    real_wait = asyncio.wait

    async def fake_wait(aws, *a, **kw):
        # Let worker run to completion, then pretend the outer task got cancelled
        # before the real wait could deliver `done`.
        for _ in range(3): await asyncio.sleep(0)
        raise asyncio.CancelledError("simulated outer cancel")
    monkeypatch.setattr("affine.loop.asyncio.wait", fake_wait)

    async def work(): return SimpleNamespace(slot_id="leaked")
    async def cleanup(slot): torn_down.append(slot.slot_id)

    with pytest.raises(asyncio.CancelledError):
        await _cancellable(work(), asyncio.Event(), on_orphan=cleanup)
    assert torn_down == ["leaked"]


class _FakeSlotsProvision:
    def __init__(self, exc: BaseException):
        self._exc = exc
        self.teardown_calls = 0
    async def provision(self, model, revision):
        raise self._exc
    async def teardown(self, slot):
        self.teardown_calls += 1


@pytest.mark.asyncio
async def test_provision_crashloop_returns_status():
    stop = asyncio.Event()
    slots = _FakeSlotsProvision(SlotProvisionFailed("crashloop"))
    slot, status = await _provision(slots, _miner(7, model="m", rev="r"), stop)
    assert slot is None and status == "crashloop"


@pytest.mark.asyncio
async def test_provision_timeout_returns_status():
    stop = asyncio.Event()
    slots = _FakeSlotsProvision(TimeoutError("not ready within 1200s"))
    slot, status = await _provision(slots, _miner(7, model="m", rev="r"), stop)
    assert slot is None and status == "timeout"


@pytest.mark.asyncio
async def test_provision_generic_exception_returns_error_status():
    stop = asyncio.Event()
    slots = _FakeSlotsProvision(ConnectionError("targon down"))
    slot, status = await _provision(slots, _miner(7, model="m", rev="r"), stop)
    assert slot is None and status == "error"


@pytest.mark.asyncio
async def test_provision_httpx_error_returns_transient_status():
    """Targon API blips surface as httpx.HTTPError (ConnectError, ReadTimeout,
    5xx). _provision must classify these as transient — session-skipping every
    chal that races a 30s Targon outage stalls the validator until queue refresh."""
    import httpx
    stop = asyncio.Event()
    slots = _FakeSlotsProvision(httpx.ConnectError("targon api unreachable"))
    slot, status = await _provision(slots, _miner(7, model="m", rev="r"), stop)
    assert slot is None and status == "transient"


@pytest.mark.asyncio
async def test_provision_cancelled_propagates():
    stop = asyncio.Event(); stop.set()
    class _HangSlots:
        async def provision(self, model, revision):
            await asyncio.sleep(60)
        async def teardown(self, slot): pass
    with pytest.raises(asyncio.CancelledError):
        await _provision(_HangSlots(), _miner(7, model="m", rev="r"), stop)


def test_apply_skip_crashloop_durable(tmp_path):
    """Crashloop = artifact is broken; persists across restarts."""
    s = MinerStates(path=tmp_path / "skip.jsonl")
    m = _miner(7, model="m", rev="r")
    _apply_skip(s, m, "crashloop", is_king=False)
    assert s.is_durable(7, ("m", "r"))
    assert (tmp_path / "skip.jsonl").read_text().strip() != ""


def test_apply_skip_timeout_attempted_for_non_king(tmp_path):
    """A single Targon hiccup must NOT lock the chal out for the process
    lifetime — mark ATTEMPTED so they're re-tried on reign change or queue
    exhaustion. Disk write is for DURABLE only."""
    skip_path = tmp_path / "skip.jsonl"
    s = MinerStates(path=skip_path)
    _apply_skip(s, _miner(7, model="m", rev="r"), "timeout", is_king=False)
    assert s.is_attempted(("m", "r"))
    assert not skip_path.exists() or skip_path.read_text().strip() == ""


def test_apply_skip_timeout_noop_for_king(tmp_path):
    """The current king cannot be marked on timeout; that would force
    re-election on every Targon hiccup and burn the weight quota."""
    s = MinerStates(path=tmp_path / "skip.jsonl")
    _apply_skip(s, _miner(7, model="m", rev="r"), "timeout", is_king=True)
    assert not s.is_attempted(("m", "r")) and not s.is_durable(7, ("m", "r"))


def test_apply_skip_error_attempted_for_non_king(tmp_path):
    skip_path = tmp_path / "skip.jsonl"
    s = MinerStates(path=skip_path)
    _apply_skip(s, _miner(7, model="m", rev="r"), "error", is_king=False)
    assert s.is_attempted(("m", "r"))
    assert not skip_path.exists() or skip_path.read_text().strip() == ""


def test_apply_skip_transient_attempts_chal(tmp_path):
    """Any non-king failure (including transient) marks ATTEMPTED so the queue
    advances. Targon outage is handled by the queue-exhaustion 120s sleep, not
    by a per-chal special case."""
    skip_path = tmp_path / "skip.jsonl"
    s = MinerStates(path=skip_path)
    _apply_skip(s, _miner(7, model="m", rev="r"), "transient", is_king=False)
    assert s.is_attempted(("m", "r"))
    assert not skip_path.exists() or skip_path.read_text().strip() == ""


def test_apply_skip_error_noop_for_king(tmp_path):
    s = MinerStates(path=tmp_path / "skip.jsonl")
    _apply_skip(s, _miner(7, model="m", rev="r"), "error", is_king=True)
    assert not s.is_attempted(("m", "r")) and not s.is_durable(7, ("m", "r"))


def test_apply_skip_attempted_dedupes_across_uids_sharing_artifact(tmp_path):
    """Two uids sharing (model, revision) must not each pay a separate Targon
    provision. ATTEMPTED is keyed by art_key so a successful chal duel locks
    out all sibling uids on the same artifact this reign — IRT pools their
    evidence anyway, so one duel is sufficient."""
    s = MinerStates(path=tmp_path / "skip.jsonl")
    uid1 = _miner(1, model="shared", rev="r")
    uid2 = _miner(2, model="shared", rev="r")
    _apply_skip(s, uid1, "ok", is_king=False)
    assert s.is_attempted(("shared", "r"))
    # uid2 with the same art_key inherits ATTEMPTED — would otherwise be queued
    # again for a redundant duel against the same provisioned model.


def test_states_durable_filters_per_uid_not_per_artifact(tmp_path):
    """Two miners share an artifact; chal crashloops, king is UNTRIED. Filter
    durable-skips chal but keeps king — durability is per-(uid, art_key), not
    per-art_key. A new uid committing the same broken artifact gets re-tested
    once before being marked durable itself."""
    s = MinerStates(path=tmp_path / "skip.jsonl")
    king = _miner(7, model="shared", rev="r1")
    chal = _miner(8, model="shared", rev="r1")
    _apply_skip(s, chal, "crashloop", is_king=False)
    assert s.filter([king, chal]) == [king]


def test_states_king_crashloop_still_durable(tmp_path):
    """A king itself crashlooping is recorded — `_apply_skip` writes DURABLE for
    `crashloop` regardless of is_king; a broken king must still be removable."""
    s = MinerStates(path=tmp_path / "skip.jsonl")
    king = _miner(7, model="shared", rev="r1")
    _apply_skip(s, king, "crashloop", is_king=True)
    assert s.is_durable(7, ("shared", "r1"))
    assert s.filter([king]) == []


def test_states_clear_attempted_only_clears_attempted(tmp_path):
    s = MinerStates(path=tmp_path / "skip.jsonl")
    s.mark_attempted(("a", "r"))
    s.mark_durable(3, ("c", "r"), "crashloop")
    s.clear_attempted()
    assert not s.is_attempted(("a", "r")) and not s.is_durable(1, ("a", "r"))
    assert s.is_durable(3, ("c", "r"))


def test_states_filter_keeps_attempted_so_irt_still_sees_them(tmp_path):
    """ATTEMPTED is queue-only filtering; the IRT and election must still see
    these miners. A chal locked out of THIS reign's queue must still contribute
    historical evidence to the global fit and remain an election candidate."""
    s = MinerStates(path=tmp_path / "skip.jsonl")
    m_attempted = _miner(1, model="m1", rev="r")
    m_durable = _miner(2, model="m2", rev="r")
    m_clean = _miner(3, model="m3", rev="r")
    s.mark_attempted(("m1", "r"))
    s.mark_durable(2, ("m2", "r"), "crashloop")
    kept = s.filter([m_attempted, m_durable, m_clean])
    assert [m.uid for m in kept] == [1, 3], "ATTEMPTED stays in miners; only DURABLE_SKIP filters out"


@pytest.mark.asyncio
async def test_provision_pair_skips_shared_artifact_on_chal_crashloop(tmp_path):
    """Regression: when king and chal share (model, revision), the same-artifact
    exemption used to suppress the skip on chal — but king's task was cancelled by
    fail-fast and got no skip either, so neither side was marked. Next iteration
    re-elected the same pair and crashlooped again forever. Fix: exemption only
    applies when caller holds a cached, proven-healthy king (not in this path)."""
    states = MinerStates(path=tmp_path / "skip.jsonl")
    stop = asyncio.Event()

    class _Slots:
        async def provision(self, model, revision):
            from affine.vllm import SlotProvisionFailed
            # Whoever runs first crashes: chal happens to win the race below.
            await asyncio.sleep(0.01)
            raise SlotProvisionFailed("crashloop")
        async def teardown(self, slot):
            pass

    await _provision_pair(_Slots(),
                          _miner(0, model="shared", rev="r1"),
                          _miner(1, model="shared", rev="r1"),
                          states, stop)

    # Whichever side completes its provision call writes DURABLE for its (uid, art).
    # The other may be cancelled by fail-fast and stay UNTRIED — but next iter will
    # exercise it. Either way, at least one side must record DURABLE so progress
    # is made instead of infinite-retrying the same pair (the prior bug).
    assert states.is_durable(0, ("shared", "r1")) or states.is_durable(1, ("shared", "r1"))


@pytest.mark.asyncio
async def test_provision_pair_fail_fast_cancels_sibling(tmp_path):
    """If king crashloops, abort challenger provisioning immediately rather than
    burning Targon time on a doomed pair."""
    states = MinerStates(path=tmp_path / "skip.jsonl")
    stop = asyncio.Event()
    chal_started = asyncio.Event()
    chal_finished = asyncio.Event()
    teardowns: list[str] = []

    class _Slots:
        async def provision(self, model, revision):
            if model == "king":
                from affine.vllm import SlotProvisionFailed
                raise SlotProvisionFailed("crashloop")
            chal_started.set()
            try:
                await asyncio.sleep(60)
            finally:
                chal_finished.set()
            return SimpleNamespace(model=model, base_url="http://c", revision=revision)
        async def teardown(self, slot):
            teardowns.append(slot.model)

    k_slot, c_slot, king_failed = await _provision_pair(_Slots(),
                                           _miner(0, model="king"), _miner(1, model="chal"),
                                           states, stop)

    assert k_slot is None and c_slot is None
    assert king_failed, "king's task ran to completion with crashloop status"
    assert chal_finished.is_set(), "challenger task should have terminated (cancelled)"
    assert teardowns == [], "no slot was produced, nothing to tear down"
    assert states.is_durable(0, ("king", "r"))


@pytest.mark.asyncio
async def test_provision_pair_chal_fails_first_does_not_blame_king(tmp_path):
    """Regression: chal raises a generic Exception (status='error'), fail-fast
    cancels king mid-download. Caller must distinguish this from a real king
    failure: blaming king triggers a 60→600s sleep that delays advancing to the
    next challenger. Returning king_attempt_failed=False lets the caller skip the
    backoff and rely on chal's ATTEMPTED mark (status='error' now writes one) to
    advance to the next chal in the queue."""
    states = MinerStates(path=tmp_path / "skip.jsonl")
    stop = asyncio.Event()

    class _Slots:
        async def provision(self, model, revision):
            if model == "chal":
                raise RuntimeError("rental register returned no uid")
            # King takes longer — chal fails first.
            await asyncio.sleep(60)
            return SimpleNamespace(model=model, base_url="http://k", revision=revision)
        async def teardown(self, slot):
            pass

    k_slot, c_slot, king_failed = await _provision_pair(_Slots(),
                                                        _miner(0, model="king"), _miner(1, model="chal"),
                                                        states, stop)

    assert k_slot is None and c_slot is None
    assert king_failed is False, "king was cancelled by chal-fail-fast, not a real king failure"
    # chal status='error' marks ATTEMPTED so the loop advances to the next chal.
    assert states.is_attempted(("chal", "r"))


@pytest.mark.asyncio
async def test_provision_pair_king_succeeds_chal_fails_retains_king(tmp_path):
    """Regression: when king completes successfully BEFORE chal returns its
    failure (e.g. king cached on Targon, chal cold-starts and 5xxs), the older
    code tore the king slot down at the post-loop fail-fast check. Re-provisioning
    a 600GB king for every chal-only failure was the dominant cost on a churn
    of bad challengers. _provision_pair must retain king and only return chal=None."""
    states = MinerStates(path=tmp_path / "skip.jsonl")
    stop = asyncio.Event()
    teardowns: list[str] = []

    class _Slots:
        async def provision(self, model, revision):
            if model == "king":
                return SimpleNamespace(model=model, base_url="http://k", revision=revision)
            # chal returns AFTER king has completed and failed with a generic exception
            # (status="error"), so this is NOT the fail-fast path — it's the fix's
            # actual target case.
            await asyncio.sleep(0.05)
            raise RuntimeError("chal provision blew up")
        async def teardown(self, slot):
            teardowns.append(slot.model)

    k_slot, c_slot, king_failed = await _provision_pair(_Slots(),
                                                         _miner(0, model="king"), _miner(1, model="chal"),
                                                         states, stop)

    assert k_slot is not None and k_slot.model == "king"
    assert c_slot is None
    assert king_failed is False
    assert teardowns == [], "king slot must NOT be torn down — it caches across chal failures"


@pytest.mark.asyncio
async def test_provision_pair_king_fails_chal_succeeds_tears_chal(tmp_path):
    """Mirror of the cache-retention test: when chal succeeds but king fails
    genuinely, chal must be torn down (it's useless without the king)."""
    states = MinerStates(path=tmp_path / "skip.jsonl")
    stop = asyncio.Event()
    teardowns: list[str] = []

    class _Slots:
        async def provision(self, model, revision):
            if model == "chal":
                return SimpleNamespace(model=model, base_url="http://c", revision=revision)
            await asyncio.sleep(0.05)
            raise RuntimeError("king provision blew up")
        async def teardown(self, slot):
            teardowns.append(slot.model)

    k_slot, c_slot, king_failed = await _provision_pair(_Slots(),
                                                         _miner(0, model="king"), _miner(1, model="chal"),
                                                         states, stop)

    assert k_slot is None and c_slot is None
    assert king_failed is True
    assert teardowns == ["chal"], "chal slot must be torn down when king fails — chal alone can't duel"


@pytest.mark.asyncio
async def test_provision_pair_partial_failure_tears_down_survivor(tmp_path):
    """If gather sees one task cancelled and the other returned a slot, that slot
    must be torn down — otherwise the loop leaks a 600GB rented vLLM on every
    SIGTERM-during-provision."""
    states = MinerStates(path=tmp_path / "skip.jsonl")
    stop = asyncio.Event()
    teardowns: list[str] = []
    class _MixedSlots:
        async def provision(self, model, revision):
            if model == "king":
                return SimpleNamespace(model=model, base_url="http://k", revision=revision)
            await asyncio.sleep(60)
        async def teardown(self, slot):
            teardowns.append(slot.model)
    async def fire():
        await asyncio.sleep(0.05); stop.set()
    fire_task = asyncio.create_task(fire())
    with pytest.raises(asyncio.CancelledError):
        await _provision_pair(_MixedSlots(),
                              _miner(0, model="king"), _miner(1, model="chal"),
                              states, stop)
    await fire_task
    assert teardowns == ["king"]


@pytest.mark.asyncio
async def test_slot_from_task_handles_running_task():
    """Regression: t.exception() raises InvalidStateError on a running task.
    The not-done guard prevents _cleanup's drain teardown loop from crashing
    if a drain await is interrupted before the task completes."""
    async def hang(): await asyncio.sleep(60)
    t = asyncio.create_task(hang())
    await asyncio.sleep(0)
    try:
        assert _slot_from_task(t) is None
    finally:
        t.cancel()
        try: await t
        except asyncio.CancelledError: pass


@pytest.mark.asyncio
async def test_slot_from_task_returns_slot_for_clean_completion():
    async def ok(): return (SimpleNamespace(model="m"), "ok")
    t = asyncio.create_task(ok())
    await t
    slot = _slot_from_task(t)
    assert slot is not None and slot.model == "m"


@pytest.mark.asyncio
async def test_slot_from_task_returns_none_for_cancelled():
    async def hang(): await asyncio.sleep(60)
    t = asyncio.create_task(hang())
    await asyncio.sleep(0)
    t.cancel()
    try: await t
    except asyncio.CancelledError: pass
    assert _slot_from_task(t) is None


@pytest.mark.asyncio
async def test_slot_from_task_returns_none_for_exception():
    async def boom(): raise RuntimeError("boom")
    t = asyncio.create_task(boom())
    try: await t
    except RuntimeError: pass
    assert _slot_from_task(t) is None


class _StubSlots:
    """In-memory slots stub for prefetch tests. `provision_calls` records every
    (model, rev) provisioned; `teardowns` records every slot torn down. `delay`
    holds provision until set; `failures` maps (model, rev) → exception."""
    def __init__(self):
        self.provision_calls: list[tuple[str, str]] = []
        self.teardowns: list[tuple[str, str]] = []
        self.delay = asyncio.Event(); self.delay.set()
        self.failures: dict[tuple[str, str], BaseException] = {}

    async def provision(self, model, revision):
        self.provision_calls.append((model, revision))
        await self.delay.wait()
        if (model, revision) in self.failures:
            raise self.failures[(model, revision)]
        return SimpleNamespace(model=model, revision=revision, base_url=f"http://{model}", slot_id=f"{model}-{revision}")

    async def teardown(self, slot):
        self.teardowns.append((slot.model, slot.revision))


@pytest.mark.asyncio
async def test_start_prefetch_kicks_off_next_in_queue(tmp_path):
    states = MinerStates(path=tmp_path / "skip.jsonl")
    state = LoopState()
    slots = _StubSlots()
    queue = [_miner(1, model="cur"), _miner(2, model="nxt"), _miner(3, model="other")]
    _start_prefetch(state, slots, queue, queue[0], states, asyncio.Event())
    assert state.next_chal is not None
    task, miner = state.next_chal
    assert miner.model == "nxt"
    await task
    assert ("nxt", "r") in slots.provision_calls


@pytest.mark.asyncio
async def test_start_prefetch_noop_when_queue_only_has_current(tmp_path):
    states = MinerStates(path=tmp_path / "skip.jsonl")
    state = LoopState()
    slots = _StubSlots()
    cur = _miner(1, model="cur")
    _start_prefetch(state, slots, [cur], cur, states, asyncio.Event())
    assert state.next_chal is None
    assert slots.provision_calls == []


@pytest.mark.asyncio
async def test_start_prefetch_noop_when_already_pending(tmp_path):
    states = MinerStates(path=tmp_path / "skip.jsonl")
    state = LoopState()
    slots = _StubSlots(); slots.delay.clear()
    queue = [_miner(1, model="cur"), _miner(2, model="nxt"), _miner(3, model="other")]
    _start_prefetch(state, slots, queue, queue[0], states, asyncio.Event())
    first_task, first_miner = state.next_chal
    _start_prefetch(state, slots, queue, queue[0], states, asyncio.Event())  # 2nd call no-op
    assert state.next_chal[0] is first_task
    assert state.next_chal[1] is first_miner
    slots.delay.set()
    await first_task


@pytest.mark.asyncio
async def test_start_prefetch_does_not_mark_attempted(tmp_path):
    """Mark-attempted defers to `_take_prefetched`. A Dethrone between
    prefetch-start and use clears `attempted`; if we'd marked at provision time
    the cleared mark would re-permit the same chal next iter, but the slot is
    already torn down. Defer-to-use keeps both states aligned."""
    states = MinerStates(path=tmp_path / "skip.jsonl")
    state = LoopState()
    slots = _StubSlots()
    queue = [_miner(1, model="cur"), _miner(2, model="nxt")]
    _start_prefetch(state, slots, queue, queue[0], states, asyncio.Event())
    await state.next_chal[0]
    assert not states.is_attempted(("nxt", "r"))


@pytest.mark.asyncio
async def test_take_prefetched_returns_slot_on_match(tmp_path):
    states = MinerStates(path=tmp_path / "skip.jsonl")
    state = LoopState()
    slots = _StubSlots()
    queue = [_miner(1, model="cur"), _miner(2, model="nxt")]
    _start_prefetch(state, slots, queue, queue[0], states, asyncio.Event())
    slot = await _take_prefetched(state, slots, states, ("nxt", "r"))
    assert slot is not None and slot.model == "nxt"
    assert states.is_attempted(("nxt", "r"))   # marked at use
    assert state.next_chal is None
    assert slots.teardowns == []


@pytest.mark.asyncio
async def test_take_prefetched_tears_down_on_mismatch(tmp_path):
    """Dethrone between prefetch and next iter rebuilds the queue with a
    different head; the prefetched slot is now stale and must be torn down."""
    states = MinerStates(path=tmp_path / "skip.jsonl")
    state = LoopState()
    slots = _StubSlots()
    queue = [_miner(1, model="cur"), _miner(2, model="nxt")]
    _start_prefetch(state, slots, queue, queue[0], states, asyncio.Event())
    slot = await _take_prefetched(state, slots, states, ("other", "v1"))
    assert slot is None
    assert slots.teardowns == [("nxt", "r")]
    assert not states.is_attempted(("nxt", "r"))   # mismatch ≠ attempt
    assert state.next_chal is None


@pytest.mark.asyncio
async def test_take_prefetched_records_crashloop_as_durable_even_on_mismatch(tmp_path):
    """A crashlooping artifact is miner fault regardless of which iter saw it.
    Must persist `durable` so the next iter doesn't re-prefetch the same
    crashing artifact."""
    states = MinerStates(path=tmp_path / "skip.jsonl")
    state = LoopState()
    slots = _StubSlots()
    slots.failures[("nxt", "r")] = SlotProvisionFailed("crash")
    queue = [_miner(1, model="cur"), _miner(2, model="nxt")]
    _start_prefetch(state, slots, queue, queue[0], states, asyncio.Event())
    slot = await _take_prefetched(state, slots, states, ("other", "v1"))
    assert slot is None
    assert states.is_durable(2, ("nxt", "r"))


@pytest.mark.asyncio
async def test_take_prefetched_returns_none_when_no_prefetch(tmp_path):
    states = MinerStates(path=tmp_path / "skip.jsonl")
    state = LoopState()
    slots = _StubSlots()
    slot = await _take_prefetched(state, slots, states, ("any", "r"))
    assert slot is None


@pytest.mark.asyncio
async def test_drop_next_chal_cancels_in_flight_and_tears_down_if_landed(tmp_path):
    states = MinerStates(path=tmp_path / "skip.jsonl")
    state = LoopState()
    slots = _StubSlots()
    queue = [_miner(1, model="cur"), _miner(2, model="nxt")]
    _start_prefetch(state, slots, queue, queue[0], states, asyncio.Event())
    await state.next_chal[0]                       # let it land
    await _drop_next_chal(state, slots)
    assert state.next_chal is None
    assert slots.teardowns == [("nxt", "r")]


@pytest.mark.asyncio
async def test_drop_next_chal_cancels_running_provision(tmp_path):
    states = MinerStates(path=tmp_path / "skip.jsonl")
    state = LoopState()
    slots = _StubSlots(); slots.delay.clear()      # provision will hang
    queue = [_miner(1, model="cur"), _miner(2, model="nxt")]
    _start_prefetch(state, slots, queue, queue[0], states, asyncio.Event())
    await asyncio.sleep(0)                          # let task start
    await _drop_next_chal(state, slots)
    assert state.next_chal is None
    assert slots.teardowns == []                    # slot never landed


@pytest.mark.asyncio
async def test_drop_next_chal_idempotent_on_none(tmp_path):
    state = LoopState()
    slots = _StubSlots()
    await _drop_next_chal(state, slots)              # no-op
    assert state.next_chal is None
    assert slots.teardowns == []


def test_respondents_registered_first_then_ghosts():
    miners = [_miner(1, rev="v1"), _miner(2, rev="v1")]
    rows = [
        _row(m=2, r="v1"),       # registered (matches uid 2's art)
        _row(m=99, r="v1"),      # ghost (no live miner at (99, v1))
        _row(m=1, r="v0"),       # ghost (old rev of registered uid)
    ]
    keys = _respondents(miners, rows)
    assert keys[:2] == [("m1", "v1"), ("m2", "v1")]
    assert set(keys[2:]) == {("?ghost:99:v1", "v1"), ("?ghost:1:v0", "v0")}


def test_fit_ignores_unknown_envs():
    miners = [_miner(0)]
    rows = [_row(m=0, e="E1"), _row(m=0, e="EX")]
    fit = _fit(rows, miners, env_names=["E1"], priors=Priors())
    assert fit.n_m == 1 and fit.n_e == 1


def test_fit_pools_uids_on_shared_artifact():
    """Two uids committing the same (model, revision) share a single θ — their
    evidence pools. A failure on one uid must not penalize a third uid on a
    different artifact: the shared art_key is the unit of θ."""
    miners = [_miner(1, model="shared", rev="v"),
              _miner(2, model="shared", rev="v"),
              _miner(3, model="solo",   rev="v")]
    rows  = [_row(m=1, r="v", k="shared", e="E", c=i, p=1) for i in range(10)]
    rows += [_row(m=2, r="v", k="shared", e="E", c=i, p=0) for i in range(10)]   # mixed under one art
    rows += [_row(m=3, r="v", k="solo",   e="E", c=i, p=1) for i in range(10)]
    fit = _fit(rows, miners, ["E"], Priors())
    assert fit.n_m == 2                                                          # shared, solo
    keys = _respondents(miners, rows)
    i_shared = keys.index(("shared", "v"))
    i_solo = keys.index(("solo", "v"))
    assert abs(fit.theta[i_shared]) < abs(fit.theta[i_solo])                     # shared sees mixed; solo sees all-pass


def test_fit_theta_tracks_pass_rate():
    miners = [_miner(0), _miner(1)]
    env_names = ["E"]
    strong = [_row(m=0, c=i, p=1) for i in range(20)]
    weak = [_row(m=1, c=i, p=0) for i in range(20)]
    fit = _fit(strong + weak, miners, env_names, Priors())
    assert fit.theta[0] > fit.theta[1]


def test_fit_drops_zero_variance_envs():
    """Envs where every observation is identical (all-pass or all-fail) are
    structurally uninformative — their α likelihood is monotone in α with no
    interior optimum. They must be dropped from the fit's data; otherwise
    α saturates (or hits a hard bound) and the Hessian at the constrained MAP
    becomes an invalid posterior precision.

    Verify the structural property: degenerate envs' α stays at prior, the
    contrast SE for healthy miners is finite, and healthy θ ordering is
    preserved."""
    miners = [_miner(0), _miner(1)]
    rows  = [_row(m=0, e="H", c=i, p=1) for i in range(12)]
    rows += [_row(m=0, e="H", c=12+i, p=0) for i in range(3)]
    rows += [_row(m=1, e="H", c=i, p=0) for i in range(12)]
    rows += [_row(m=1, e="H", c=12+i, p=1) for i in range(3)]
    rows += [_row(m=0, e="D1", c=i, p=1) for i in range(40)]
    rows += [_row(m=1, e="D1", c=i, p=1) for i in range(40)]   # all-pass
    rows += [_row(m=0, e="D2", c=i, p=0) for i in range(40)]
    rows += [_row(m=1, e="D2", c=i, p=0) for i in range(40)]   # all-fail
    fit = _fit(rows, miners, ["H", "D1", "D2"], Priors())
    assert abs(fit.alpha[1]) < 1.0   # D1 stayed at prior
    assert abs(fit.alpha[2]) < 1.0   # D2 stayed at prior
    assert fit.theta[0] > fit.theta[1]
    _, se = fit.contrast(0, 1)
    assert np.isfinite(se) and se > 0

    # Acquisition is intentionally NOT told to mask zero-variance envs. Those
    # envs sit at prior (wide posterior); a Thompson draw can favor them, and
    # that's correct exploration — a single fail there un-degenerates the env.
    # The fit's job is to refuse to tighten SE on no-info envs (verified above).
    # Acquisition's job is to find the env that could change the verdict, which
    # may well be one we have no data on yet.
    from affine.irt import fisher_env
    rng = np.random.default_rng(0)
    picks = [fisher_env(fit, 0, 1, rng) for _ in range(500)]
    assert any(p in (1, 2) for p in picks), "degenerate envs should remain pickable"


def test_seed_depends_on_salt():
    """An adversary watching the chain knows uid/rev/env/c. Mixing the validator's
    hotkey in (still public, but per-validator) forces precomputation per
    validator instead of one universal answer set. Verify that changing only the
    salt changes the seed."""
    a = _seed(7, "rev", "ded", 3, salt="hkA")
    b = _seed(7, "rev", "ded", 3, salt="hkB")
    assert a != b


def test_task_id_depends_on_salt():
    from affine.loop import _task_id
    a = _task_id(3, 7, "ded", 0, 0, 999, salt="hkA")
    b = _task_id(3, 7, "ded", 0, 0, 999, salt="hkB")
    assert a != b


@pytest.mark.asyncio
async def test_load_envs_injects_openai_api_key(monkeypatch):
    """Env containers need OPENAI_API_KEY set (vLLM ignores the value but the
    openai SDK refuses to instantiate without it). Regression: without this,
    every evaluate() failed instantly with OpenAIError and every duel aborted."""
    calls: list[dict] = []
    def fake_load_env(image, mode, env_vars, mem_limit, pull, cleanup):
        calls.append(env_vars or {})
        return SimpleNamespace(_backend=None)
    monkeypatch.setattr("affinetes.load_env", fake_load_env)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = Config(environments=(
        EnvSpec(name="a", image="img-a"),
        EnvSpec(name="b", image="img-b", env_vars={"OPENAI_API_KEY": "override"}),
    ))
    await _load_envs(cfg)
    assert calls[0]["OPENAI_API_KEY"] == "vllm"
    assert calls[1]["OPENAI_API_KEY"] == "override"


@pytest.mark.asyncio
async def test_load_envs_shares_wrapper_for_duplicate_images(monkeypatch):
    """Two envs with the same image (eg affine:ded + affine:abd on affine-env:v4)
    must share one container wrapper but keep per-env params.

    Regression: spec._replace() was called on a frozen dataclass — crashed on
    every production startup with AttributeError before this test existed."""
    built: list[str] = []
    def fake_load_env(image, mode, env_vars, mem_limit, pull, cleanup):
        built.append(image)
        return SimpleNamespace(_backend=None, image=image)
    monkeypatch.setattr("affinetes.load_env", fake_load_env)
    cfg = Config(environments=(
        EnvSpec(name="ded", image="shared", params={"task_type": "ded"}),
        EnvSpec(name="abd", image="shared", params={"task_type": "abd"}),
        EnvSpec(name="other", image="distinct"),
    ))
    envs = await _load_envs(cfg)
    assert built == ["shared", "distinct"]                 # one wrapper per unique image
    assert envs["ded"][0] is envs["abd"][0]                # shared wrapper
    assert envs["ded"][1].params["task_type"] == "ded"     # but per-env params kept
    assert envs["abd"][1].params["task_type"] == "abd"
    assert envs["other"][0].image == "distinct"


@pytest.mark.asyncio
async def test_load_envs_rejects_shared_image_with_different_env_vars(monkeypatch):
    """Same image but different env_vars/mem_limit must NOT silently share a
    wrapper — the second env's container-init config would be lost."""
    monkeypatch.setattr("affinetes.load_env",
        lambda image, mode, env_vars, mem_limit, pull, cleanup: SimpleNamespace(_backend=None))
    cfg = Config(environments=(
        EnvSpec(name="a", image="shared", env_vars={"K": "1"}),
        EnvSpec(name="b", image="shared", env_vars={"K": "2"}),
    ))
    with pytest.raises(ValueError, match="env_vars/mem_limit differ"):
        await _load_envs(cfg)


@pytest.mark.asyncio
async def test_load_envs_rejects_shared_image_with_different_mem_limit(monkeypatch):
    monkeypatch.setattr("affinetes.load_env",
        lambda image, mode, env_vars, mem_limit, pull, cleanup: SimpleNamespace(_backend=None))
    cfg = Config(environments=(
        EnvSpec(name="a", image="shared", mem_limit="4g"),
        EnvSpec(name="b", image="shared", mem_limit="8g"),
    ))
    with pytest.raises(ValueError, match="env_vars/mem_limit differ"):
        await _load_envs(cfg)


@pytest.mark.asyncio
async def test_load_envs_cleans_up_wrappers_on_partial_init_failure(monkeypatch):
    """If af.load_env raises mid-loop, already-loaded wrappers must be cleaned
    up — otherwise their Docker containers leak."""
    cleaned: list[str] = []
    def make_wrapper(image):
        async def cleanup(): cleaned.append(image)
        return SimpleNamespace(_backend=None, image=image, cleanup=cleanup)
    call_count = [0]
    def fake_load_env(image, mode, env_vars, mem_limit, pull, cleanup):
        call_count[0] += 1
        if call_count[0] == 3:
            raise RuntimeError("docker daemon vanished")
        return make_wrapper(image)
    monkeypatch.setattr("affinetes.load_env", fake_load_env)
    cfg = Config(environments=(
        EnvSpec(name="a", image="img-a"),
        EnvSpec(name="b", image="img-b"),
        EnvSpec(name="c", image="img-c"),
    ))
    with pytest.raises(RuntimeError, match="docker daemon"):
        await _load_envs(cfg)
    assert sorted(cleaned) == ["img-a", "img-b"]


@pytest.mark.asyncio
async def test_elect_cold_start_falls_back_to_lowest_block_when_no_baseline():
    """No evidence + no baseline registered: argmax(θ̂) on a prior-only fit
    deterministically returns uid 0 by tie-break, which is misleading. Pick
    lowest-(block, uid) explicitly so the choice matches the queue ordering
    the loop will use for everyone else."""
    from affine.loop import _elect
    miners = [
        Miner(uid=5, hotkey="h5", model="off-list-A", revision="r5", block=200),
        Miner(uid=2, hotkey="h2", model="off-list-B", revision="r2", block=100),
        Miner(uid=9, hotkey="h9", model="off-list-C", revision="r9", block=300),
    ]
    uid, rev = await _elect([], miners, ["E"], Priors())
    assert (uid, rev) == (2, "r2")


@pytest.mark.asyncio
async def test_elect_falls_back_to_baseline_when_fit_is_degenerate(monkeypatch):
    """argmax(θ̂) on a non-MAP (degenerate) fit can publish a fabricated winner.
    Same risk the verdict path refuses for. With evidence + a degenerate fit,
    election must fall back to the baseline (or lowest-block when no baseline)."""
    from affine.loop import _elect
    import affine.irt as irt

    real_fit_2pl = irt.fit_2pl
    def degen(*a, **kw):
        f = real_fit_2pl(*a, **kw)
        f.degenerate = True
        return f
    monkeypatch.setattr("affine.loop.fit_2pl", degen)

    miners = [
        Miner(uid=5, hotkey="h5", model="off-A", revision="r5", block=200),
        Miner(uid=2, hotkey="h2", model="off-B", revision="r2", block=100),
        Miner(uid=9, hotkey="h9", model="off-C", revision="r9", block=300),
    ]
    rows = [Row(m=5, r="r5", e="E", c=0, p=1, t=0, l=0.1, i=0)]
    uid, rev = await _elect(rows, miners, ["E"], Priors())
    assert (uid, rev) == (2, "r2")


@pytest.mark.asyncio
async def test_elect_falls_back_when_rows_are_uninformative():
    """Regression: rows can be non-empty while every env is all-pass or all-fail.
    _fit drops uninformative envs from the fit data, so the optimizer runs on
    zero observations, theta stays at the prior mean (all zeros), and argmax
    deterministically picks index 0 — the lowest-uid miner — regardless of who
    actually got the (uniform) outcomes. Behaviorally identical to cold start;
    must route to the baseline/lowest-block fallback."""
    from affine.loop import _elect
    miners = [
        Miner(uid=5, hotkey="h5", model="off-A", revision="r5", block=200),
        Miner(uid=2, hotkey="h2", model="off-B", revision="r2", block=100),
        Miner(uid=9, hotkey="h9", model="off-C", revision="r9", block=300),
    ]
    # All-pass on every env — no env has both 0 and 1 outcomes, so _fit drops
    # all of them and the fit collapses to the prior. Without the fallback,
    # argmax(theta) would publish uid 5 (index 0).
    rows = [Row(m=m.uid, r=m.revision, e=e, c=c, p=1, t=0, l=0.1, i=c)
            for m in miners for e in ("E", "F") for c in range(5)]
    uid, rev = await _elect(rows, miners, ["E", "F"], Priors())
    assert (uid, rev) == (2, "r2")   # lowest-block fallback, not argmax(theta)=uid 5


@pytest.mark.asyncio
async def test_elect_uses_argmax_theta_when_fit_is_healthy():
    """Sanity: with non-degenerate evidence, election still picks argmax(θ̂)."""
    from affine.loop import _elect
    miners = [
        Miner(uid=5, hotkey="h5", model="off-A", revision="r5", block=200),
        Miner(uid=2, hotkey="h2", model="off-B", revision="r2", block=100),
    ]
    rows = []
    # Push uid 5 to high θ̂ via wins on env E, uid 2 losses.
    for c in range(20):
        rows.append(Row(m=5, r="r5", e="E", c=c, p=1, t=0, l=0.1, i=c))
        rows.append(Row(m=2, r="r2", e="E", c=c, p=0, t=0, l=0.1, i=c))
        rows.append(Row(m=5, r="r5", e="F", c=c, p=1, t=0, l=0.1, i=c))
        rows.append(Row(m=2, r="r2", e="F", c=c, p=0, t=0, l=0.1, i=c))
    uid, _ = await _elect(rows, miners, ["E", "F"], Priors())
    assert uid == 5


def test_static_chain_returns_fixed_miners():
    miners = [_miner(0), _miner(1)]
    chain = static_chain(miners, hotkey="hk")
    assert chain.hotkey == "hk"
    assert asyncio.run(chain.list_miners()) == miners
    assert asyncio.run(chain.current_block()) == 0


def _env(err=False):
    async def evaluate(*args, **kwargs):
        if err: raise RuntimeError("boom")
        return {"success": True}
    wrapper = SimpleNamespace(evaluate=AsyncMock(side_effect=evaluate))
    return {"E": (wrapper, EnvSpec(name="E", image="img", params={"timeout": 5}))}


def _stop_after_pairs(stop, n, base):
    """Wrap a `run_one` mock so that after `n` pair-completions (= 2*n single
    samples), the dwell `stop` event fires. Used in tests where the data is
    inherently uninformative (all-pass / forced-degenerate / both-fail) so
    neither z>k nor z<-k can fire — without an external stop the loop would
    never terminate now that the iter cap is gone."""
    counter = [0]
    target = 2 * n
    async def wrapped(*args, **kwargs):
        result = await base(*args, **kwargs)
        counter[0] += 1
        if counter[0] >= target:
            stop.set()
        return result
    return wrapped


@pytest.mark.asyncio
async def test_dwell_appends_two_rows_per_env_pick(tmp_path, monkeypatch):
    """Each fisher_env pick must append exactly two rows (king + challenger)."""
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=42), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(evidence_path=str(tmp_path / "ev.jsonl"))
    stop = asyncio.Event()
    # Balanced data per pair (both win or both lose, alternating) → z ≈ 0,
    # neither z>k nor z<-k fires, runtime bounded by injected stop.
    n = [0]
    async def _run_one(wrapper, params, timeout, slot, seed, task_id=0):
        n[0] += 1
        return bool((n[0] // 2) % 2), 0.01, 0
    monkeypatch.setattr("affine.loop.run_one", _stop_after_pairs(stop, 5, _run_one))
    out = await dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], _env(), ["E"], store, cfg, Priors(),
        np.random.default_rng(0), stop, reign_start_block=0,
    )
    rows, fit, _abort = out.rows, out.fit, out.status
    # Each pick appends exactly 2 rows (king + chal), uids balanced. Exact count
    # depends on race between stop and pair completion in the synchronous mock.
    assert len(rows) > 0 and len(rows) % 2 == 0
    uids = [r.m for r in store.read()]
    assert uids.count(0) == uids.count(1) > 0
    assert fit.n_m == 2


@pytest.mark.asyncio
async def test_dwell_batch_dispatches_in_parallel_with_unique_counters(tmp_path, monkeypatch):
    """dwell_batch=B dispatches B*2 samples concurrently per refit. Same-env
    samples within a batch must use sequential counters (else they collide on
    Row identity and seed). Each sample's task_id is keyed by a unique iter_idx
    so even same-env duplicates in the batch see distinct tasks."""
    inflight = [0]; max_inflight = [0]
    seen_seeds: set[int] = set()
    seen_task_ids: set[tuple[int, str]] = set()
    async def _run_one(wrapper, params, timeout, slot, seed, task_id=0):
        inflight[0] += 1; max_inflight[0] = max(max_inflight[0], inflight[0])
        seen_seeds.add(seed); seen_task_ids.add((task_id, slot.model))
        await asyncio.sleep(0.01)
        inflight[0] -= 1
        return True, 0.01, 0
    envs = {n: (SimpleNamespace(evaluate=AsyncMock()),
                EnvSpec(name=n, image="i", params={"timeout": 5}))
            for n in ("A", "B")}
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=42), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(dwell_batch=4, evidence_path=str(tmp_path / "ev.jsonl"))
    stop = asyncio.Event()
    monkeypatch.setattr("affine.loop.run_one", _stop_after_pairs(stop, 8, _run_one))
    out = await dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], envs, ["A", "B"], store, cfg, Priors(),
        np.random.default_rng(0), stop, reign_start_block=0,
    )
    assert max_inflight[0] >= 2 * cfg.dwell_batch, \
        f"expected ≥{2*cfg.dwell_batch} concurrent samples, observed {max_inflight[0]}"
    rows = store.read()
    assert len(rows) > 0, "no rows accumulated"
    by_id = {(r.m, r.r, r.e, r.c) for r in rows}
    assert len(by_id) == len(rows), "duplicate Row identity — counter collision in batched dispatch"


@pytest.mark.asyncio
async def test_dwell_queue_keeps_slot_saturated_under_mixed_latency(tmp_path, monkeypatch):
    """Queue model: a fast pair finishing triggers an immediate refill rather
    than waiting for the slow tail of a batch. Verified by time-weighting the
    in-flight sample count under mixed latencies — average stays close to
    `dwell_batch * 2` (king + chal per pair). A batch-then-wait model would
    drop to ~0 between batches as the slow tail finishes."""
    import time as _time
    inflight = [0]; samples: list[tuple[float, int]] = []
    async def _run_one(wrapper, params, timeout, slot, seed, task_id=0):
        inflight[0] += 1; samples.append((_time.monotonic(), inflight[0]))
        # task_id is shared by king+chal of a pair, so latency is per-pair, not
        # per-side. Without this, both sides of every pair span the slow band
        # → pair latencies all collapse to max → effectively batched behavior.
        await asyncio.sleep(0.05 if task_id % 3 == 0 else 0.01)
        inflight[0] -= 1; samples.append((_time.monotonic(), inflight[0]))
        return True, 0.0, 0
    envs = {n: (SimpleNamespace(evaluate=AsyncMock()),
                EnvSpec(name=n, image="i", params={"timeout": 5}))
            for n in ("A", "B")}
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    B = 8
    cfg = Config(dwell_batch=B, evidence_path=str(tmp_path / "ev.jsonl"))
    stop = asyncio.Event()
    monkeypatch.setattr("affine.loop.run_one", _stop_after_pairs(stop, 64, _run_one))
    await dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], envs, ["A", "B"], store, cfg, Priors(),
        np.random.default_rng(0), stop, reign_start_block=0,
    )
    weighted = total = 0.0
    for (t1, n1), (t2, _n2) in zip(samples, samples[1:]):
        dt = t2 - t1
        weighted += n1 * dt; total += dt
    avg = weighted / total if total else 0.0
    assert avg >= 1.4 * B, f"avg in-flight {avg:.1f} < 1.4*B={1.4 * B}; queue isn't saturating slot"


@pytest.mark.asyncio
async def test_dwell_keeps_sampling_through_zero_variance(tmp_path, monkeypatch):
    """Regression: with all-pass outcomes on every env, the old _fit returned a
    drop_idx that the dwell loop OR-ed into `excluded`. After one cycle every
    env was both all-pass AND in drop_idx → all envs excluded → dwell aborts at
    i=0 with no chance to sample a fail that would un-degenerate the env. The
    fix: degeneracy is the fit's concern (don't tighten SE on no-info envs);
    acquisition keeps every env eligible."""
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=42), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(evidence_path=str(tmp_path / "ev.jsonl"))
    stop = asyncio.Event()
    async def _all_pass(wrapper, params, timeout, slot, seed, task_id=0):
        return True, 0.01, 0
    # All-pass on every env: fit stays degenerate, neither z>k nor z<-k can fire,
    # dwell would otherwise loop forever. Cap with stop after 10 pairs.
    monkeypatch.setattr("affine.loop.run_one", _stop_after_pairs(stop, 10, _all_pass))
    envs = {
        "A": (SimpleNamespace(evaluate=AsyncMock()), EnvSpec(name="A", image="i", params={"timeout": 5})),
        "B": (SimpleNamespace(evaluate=AsyncMock()), EnvSpec(name="B", image="i", params={"timeout": 5})),
    }
    out = await dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], envs, ["A", "B"], store, cfg, Priors(),
        np.random.default_rng(0), stop, reign_start_block=0,
    )
    rows, _, abort = out.rows, out.fit, out.status
    # Pre-fix: dwell aborts at i=0 with no rows when all envs are all-pass.
    # Post-fix: rows accumulate continuously until our injected stop fires.
    assert len(rows) > 0, f"dwell must keep sampling under all-pass degeneracy; got {len(rows)} rows"
    assert abort is DuelStatus.CANCELLED


@pytest.mark.asyncio
async def test_dwell_routes_around_fisher_env_when_fit_is_degenerate(tmp_path, monkeypatch):
    """Degenerate fit → cov has 1e12-scale entries → Fit.sample produces garbage
    draws → log_info collapses, fisher_env is meaningless. dwell must fall back
    to uniform random env selection so coverage is preserved over the dwell budget."""
    import affine.loop as loop_mod
    real_fit_2pl = loop_mod.fit_2pl
    def degen(*a, **kw):
        f = real_fit_2pl(*a, **kw)
        f.degenerate = True
        return f
    monkeypatch.setattr(loop_mod, "fit_2pl", degen)
    fisher_calls = [0]
    def boom(*a, **kw):
        fisher_calls[0] += 1
        raise AssertionError("fisher_env must not be called on degenerate fit")
    monkeypatch.setattr(loop_mod, "fisher_env", boom)

    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=42), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(evidence_path=str(tmp_path / "ev.jsonl"))
    stop = asyncio.Event()
    async def _ok(wrapper, params, timeout, slot, seed, task_id=0):
        return True, 0.01, 0
    monkeypatch.setattr("affine.loop.run_one", _stop_after_pairs(stop, 5, _ok))
    envs = {
        "A": (SimpleNamespace(evaluate=AsyncMock()), EnvSpec(name="A", image="i", params={"timeout": 5})),
        "B": (SimpleNamespace(evaluate=AsyncMock()), EnvSpec(name="B", image="i", params={"timeout": 5})),
    }
    out = await dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], envs, ["A", "B"], store, cfg, Priors(),
        np.random.default_rng(0), stop, reign_start_block=0,
    )
    rows, fit, _abort = out.rows, out.fit, out.status
    assert len(rows) > 0
    assert fisher_calls[0] == 0
    # Coverage: with degenerate-fit uniform fallback over 5 picks across 2 envs,
    # both envs should appear (RNG seed 0 is deterministic; this is verifying
    # the fallback path runs, not its randomness).
    envs_seen = {r.e for r in store.read()}
    assert envs_seen == {"A", "B"}


@pytest.mark.asyncio
async def test_dwell_early_stops_when_z_exceeds_k(tmp_path, monkeypatch):
    """Mid-dwell early-stop (plan §6): once z = Δθ̂/SE > k(reign), dwell returns
    COMPLETED immediately. The caller's decide() then produces Dethrone from the
    same fit. Verified by clamping k to a permissive value via compute_k patch —
    any non-degenerate refit after the first iter trips the threshold."""
    monkeypatch.setattr("affine.loop.compute_k", lambda *a, **kw: -1.0)
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(evidence_path=str(tmp_path / "ev.jsonl"))
    n = [0]
    async def _run_one(wrapper, params, timeout, slot, seed, task_id=0):
        n[0] += 1
        return n[0] % 4 != 0, 0.01, 0  # mixed outcomes — escape the degenerate fit
    monkeypatch.setattr("affine.loop.run_one", _run_one)
    out = await dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], _env(), ["E"], store, cfg, Priors(),
        np.random.default_rng(0), asyncio.Event(), reign_start_block=0,
    )
    assert out.status is DuelStatus.COMPLETED
    delta, se = out.fit.contrast(1, 0)
    z = delta / se if se > 0 else 0.0
    assert z > -1.0, f"early-stop fires on z>k=-1; expected z>-1, got z={z:+.2f}"


@pytest.mark.asyncio
async def test_dwell_z_below_neg_k_fires_for_lopsided_chal(tmp_path, monkeypatch):
    """Mirror of z>k early-stop: with chal failing every pick and king passing
    every pick, the contrast z plunges below −k. dwell returns COMPLETED on
    the symmetric stop and decide() produces Hold (z_below_k). The asymptotic
    mirror of z>k under unbounded info — both directions share the one knob k(reign)."""
    async def split(wrapper, params, timeout, slot, seed, task_id=0):
        return (True if slot.model == "mk" else False), 0.01, 0
    monkeypatch.setattr("affine.loop.run_one", split)
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(evidence_path=str(tmp_path / "ev.jsonl"),
                 k_init=1.5, k_final=1.5, k_halflife=1)
    out = await dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], _env(), ["E"], store, cfg, Priors(),
        np.random.default_rng(0), asyncio.Event(), reign_start_block=0,
    )
    assert out.status is DuelStatus.COMPLETED
    delta, se = out.fit.contrast(1, 0)
    z = delta / se if se > 0 else 0.0
    assert z < -1.5, f"chal failing every pick should trip z<-k; got z={z:+.2f}"


@pytest.mark.asyncio
async def test_dwell_aborts_when_both_slots_dead(tmp_path):
    """Every wrapper call raises → run_one returns None for both sides every
    pair → both synthetic p=0,l=0 rows appended; both consec counters cross
    SLOT_DEAD → abort with KING_SLOT_DEAD (both-dead branch)."""
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(evidence_path=str(tmp_path / "ev.jsonl"))
    out = await dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], _env(err=True), ["E"], store, cfg, Priors(),
        np.random.default_rng(0), asyncio.Event(), reign_start_block=0,
    )
    assert out.status is DuelStatus.KING_SLOT_DEAD
    assert all(r.p == 0 and r.l == 0.0 for r in out.rows)
    assert out.rows == store.read()
    assert out.fit.n_m == 2


@pytest.mark.asyncio
async def test_dwell_aborts_on_king_slot_dead_only(tmp_path, monkeypatch):
    """King-side never delivers, chal-side delivers every pair → king consec
    grows past SLOT_DEAD, chal consec stays at 0 → CHAL_SLOT_DEAD."""
    async def split(wrapper, params, timeout, slot, seed, task_id=0):
        return (None if slot.model == "mk" else False), 0.01, 0
    monkeypatch.setattr("affine.loop.run_one", split)
    envs = {"E": (SimpleNamespace(evaluate=AsyncMock()), EnvSpec(name="E", image="i", params={"timeout": 5}))}
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(evidence_path=str(tmp_path / "ev.jsonl"))
    out = await dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], envs, list(envs), store, cfg, Priors(),
        np.random.default_rng(0), asyncio.Event(), reign_start_block=0,
    )
    assert out.status is DuelStatus.KING_SLOT_DEAD


@pytest.mark.asyncio
async def test_dwell_aborts_on_chal_slot_dead_only(tmp_path, monkeypatch):
    """Chal-side never delivers, king-side always does → chal consec grows past
    SLOT_DEAD while king consec stays at 0 → CHAL_SLOT_DEAD."""
    async def split(wrapper, params, timeout, slot, seed, task_id=0):
        return (None if slot.model == "mc" else False), 0.01, 0
    monkeypatch.setattr("affine.loop.run_one", split)
    envs = {"E": (SimpleNamespace(evaluate=AsyncMock()), EnvSpec(name="E", image="i", params={"timeout": 5}))}
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(evidence_path=str(tmp_path / "ev.jsonl"))
    out = await dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], envs, list(envs), store, cfg, Priors(),
        np.random.default_rng(0), asyncio.Event(), reign_start_block=0,
    )
    assert out.status is DuelStatus.CHAL_SLOT_DEAD


@pytest.mark.asyncio
async def test_dwell_persistent_chal_infra_failure_holds_via_synthetic_loss(tmp_path, monkeypatch):
    """Chal endpoint failing every pick (validator sees `run_one`-None) is
    matched against a delivering king — asymmetry produces a synthetic chal-loss
    row paired with a real king-pass row. The IRT contrast plunges below −k and
    dwell exits with COMPLETED via z<−k. Closes the selective-failure gaming
    attack: an attacker that 5xxs hard tasks no longer evades the loss."""
    monkeypatch.setattr("affine.loop.compute_k", lambda *a, **kw: 1.0)
    async def split(wrapper, params, timeout, slot, seed, task_id=0):
        return (True if slot.model == "mk" else None), 0.01, 0
    monkeypatch.setattr("affine.loop.run_one", split)
    envs = {n: (SimpleNamespace(evaluate=AsyncMock()), EnvSpec(name=n, image="i", params={"timeout": 5}))
            for n in ("A", "B", "C")}
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(evidence_path=str(tmp_path / "ev.jsonl"))
    out = await dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], envs, list(envs), store, cfg, Priors(),
        np.random.default_rng(0), asyncio.Event(), reign_start_block=0,
    )
    assert out.status is DuelStatus.COMPLETED
    delta, se = out.fit.contrast(1, 0)
    assert delta / se < -1.0, f"expected z<−k=−1; got z={delta/se:+.2f}"
    king_rows = [r for r in out.rows if r.m == king.uid]
    chal_rows = [r for r in out.rows if r.m == chal.uid]
    assert king_rows and all(r.p == 1 and r.l > 0 for r in king_rows), "king delivered → real p=1"
    assert chal_rows and all(r.p == 0 and r.l == 0.0 for r in chal_rows), "chal failed paired with king-pass → synthetic p=0,l=0"
    assert len(king_rows) == len(chal_rows), "matched-pair count: every kept pair appends both rows"
    assert out.rows == store.read()


@pytest.mark.asyncio
async def test_dwell_broken_king_dethrones_via_synthetic_loss(tmp_path, monkeypatch):
    """plan.md L47 'first functioning challenger wins by default': a king
    failing to deliver every pick paired with a delivering chal contributes a
    synthetic king-loss row + real chal-pass row per pair. The contrast crosses
    z>k and dwell exits with COMPLETED. The verdict path then produces Dethrone
    on those rows. Without conditional synthetic-loss, the broken-king duel
    would loop forever (slot-dead → reprovision → same broken king again)."""
    monkeypatch.setattr("affine.loop.compute_k", lambda *a, **kw: 1.0)
    async def split(wrapper, params, timeout, slot, seed, task_id=0):
        return (None if slot.model == "mk" else True), 0.01, 0
    monkeypatch.setattr("affine.loop.run_one", split)
    envs = {n: (SimpleNamespace(evaluate=AsyncMock()), EnvSpec(name=n, image="i", params={"timeout": 5}))
            for n in ("A", "B", "C")}
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(evidence_path=str(tmp_path / "ev.jsonl"))
    out = await dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], envs, list(envs), store, cfg, Priors(),
        np.random.default_rng(0), asyncio.Event(), reign_start_block=0,
    )
    assert out.status is DuelStatus.COMPLETED
    delta, se = out.fit.contrast(1, 0)
    assert delta / se > 1.0, f"expected z>k=1; got z={delta/se:+.2f}"
    king_rows = [r for r in out.rows if r.m == king.uid]
    chal_rows = [r for r in out.rows if r.m == chal.uid]
    assert king_rows and all(r.p == 0 and r.l == 0.0 for r in king_rows), "king failed paired with chal-pass → synthetic p=0,l=0"
    assert chal_rows and all(r.p == 1 and r.l > 0 for r in chal_rows), "chal delivered → real p=1"
    assert len(king_rows) == len(chal_rows), "matched-pair count: every kept pair appends both rows"


@pytest.mark.asyncio
async def test_dwell_both_sides_fail_appends_two_synthetic_zero_rows(tmp_path, monkeypatch):
    """Both-side simultaneous failure (env outage or both endpoints
    unreachable) appends BOTH synthetic p=0,l=0 rows. Matched-pair contrast
    contribution is zero (Δp=0) either way — drop and append-both produce
    identical contrast effects — but appending honestly logs the loss rate.
    Drop would let an attacker suppress their loss rate by crashing on tasks
    that happen to coincide with king failures (rate ≈ P(king_fail) in hard
    envs); appending closes that loophole. Slot-dead consec counters still
    climb on both sides → eventual KING_SLOT_DEAD (both-dead branch)."""
    async def both_fail(wrapper, params, timeout, slot, seed, task_id=0):
        return None, 0.01, 0
    monkeypatch.setattr("affine.loop.run_one", both_fail)
    envs = {"E": (SimpleNamespace(evaluate=AsyncMock()), EnvSpec(name="E", image="i", params={"timeout": 5}))}
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(evidence_path=str(tmp_path / "ev.jsonl"))
    out = await dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], envs, list(envs), store, cfg, Priors(),
        np.random.default_rng(0), asyncio.Event(), reign_start_block=0,
    )
    assert out.status is DuelStatus.KING_SLOT_DEAD  # both-dead branch
    king_rows = [r for r in out.rows if r.m == king.uid]
    chal_rows = [r for r in out.rows if r.m == chal.uid]
    assert len(king_rows) == len(chal_rows) >= SLOT_DEAD
    assert all(r.p == 0 and r.l == 0.0 for r in out.rows), "all synthetic on both-fail"
    assert out.rows == store.read()




@pytest.mark.asyncio
async def test_cross_duel_real_row_evidence_drives_dethrone(tmp_path, monkeypatch):
    """Rows persist across duels and the joint fit on accumulated real-row
    evidence drives dethrone-grade z for a chal that consistently beats the
    king. King answers wrong (real-loss row, p=0, l>0), chal answers right
    (real-pass row); no infra failures, no synthetic rows. After 8 duels each
    contributing real rows, the latest chal's contrast must exceed k_init=3."""
    async def king_loses(wrapper, params, timeout, slot, seed, task_id=0):
        return (slot.model != "mk"), 0.01, 0
    monkeypatch.setattr("affine.loop.run_one", king_loses)
    envs = {n: (SimpleNamespace(evaluate=AsyncMock()), EnvSpec(name=n, image="i", params={"timeout": 5}))
            for n in ("A", "B", "C", "D")}
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())
    king = _miner(0, model="mk")
    cfg = Config(evidence_path=str(tmp_path / "ev.jsonl"))

    rows: list[Row] = []
    chal_uids = list(range(1, 9))
    for chal_uid in chal_uids:
        chal = _miner(chal_uid, model=f"mc{chal_uid}")
        out = await dwell(
            chain, king, SimpleNamespace(model="mk", base_url="uk"),
            chal, SimpleNamespace(model=chal.model, base_url=f"uc{chal_uid}"),
            [king, chal], rows, envs, list(envs), store, cfg, Priors(),
            np.random.default_rng(chal_uid), asyncio.Event(), reign_start_block=0,
        )
        rows = out.rows

    from affine.loop import _fit, _respondents
    all_miners = [king] + [_miner(u, model=f"mc{u}") for u in chal_uids]
    fit = _fit(rows, all_miners, list(envs), Priors())
    art_keys = _respondents(all_miners, rows)
    chal_idx = art_keys.index((f"mc{chal_uids[-1]}", "r"))
    king_idx = art_keys.index(("mk", "r"))
    delta, se = fit.contrast(chal_idx, king_idx)
    assert delta > 0
    assert delta / se > 3.0, f"consistently-better chal must yield dethrone-grade z; got {delta/se:.2f}"
    assert all(r.l > 0.0 for r in rows), "no synthetic rows: every appended row has real latency"


@pytest.mark.asyncio
async def test_dwell_intermittent_one_side_does_not_abort(tmp_path, monkeypatch):
    """Dwell must complete normally with intermittent chal failures — no
    slot-dead abort. The infra-failure pairs (every 4th sample) append a
    synthetic chal p=0 + real king p=1 since king delivers, but consec_fails
    resets on the next delivered chal pair so SLOT_DEAD never trips. The
    chal's real-loss rows + synthetic-loss rows together drive z<-k."""
    counter = {"n": 0}
    async def alternating(wrapper, params, timeout, slot, seed, task_id=0):
        if slot.model == "mk":
            return True, 0.01, 0
        # chal: fail every 4th sample but otherwise succeed
        counter["n"] += 1
        return (None if counter["n"] % 4 == 0 else False), 0.01, 0
    monkeypatch.setattr("affine.loop.run_one", alternating)
    envs = {"A": (SimpleNamespace(evaluate=AsyncMock()), EnvSpec(name="A", image="i", params={"timeout": 5}))}
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(evidence_path=str(tmp_path / "ev.jsonl"))
    out = await dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], envs, list(envs), store, cfg, Priors(),
        np.random.default_rng(0), asyncio.Event(), reign_start_block=0,
    )
    rows, _, abort = out.rows, out.fit, out.status
    assert abort is DuelStatus.COMPLETED
    assert len(rows) > 0, "successful pairs must accumulate"


class _FakeSlots:
    """Fake slot pool: provisions instantly, records provision + teardown calls."""
    def __init__(self):
        self.torn_down = 0
        self.provisions: list[tuple[str, str]] = []

    async def provision(self, model, revision):
        self.provisions.append((model, revision))
        return SimpleNamespace(model=model, revision=revision,
                               base_url=f"http://fake/{model}", slot_id=f"s-{model}")

    async def teardown(self, slot):
        self.torn_down += 1


@pytest.mark.asyncio
async def test_run_cold_starts_and_dethrones(tmp_path, monkeypatch):
    """End-to-end: two miners, fake envs where uid=1 always passes and uid=0
    always fails. Cold start picks uid 0 (block 0), challenger uid 1 dethrones
    after enough samples, loop exits when no further challengers remain."""
    from affine.loop import run

    monkeypatch.setattr("affine.loop._load_envs", AsyncMock(return_value={
        "E": (SimpleNamespace(), EnvSpec(name="E", image="img", params={"timeout": 5})),
    }))


    async def _run_one(wrapper, params, timeout, slot, seed, task_id=0):
        return (slot.model == "mc"), 0.01, 0   # mc always wins, mk always loses
    monkeypatch.setattr("affine.loop.run_one", _run_one)

    miners = [_miner(0, model="mk"), _miner(1, model="mc")]
    blocks = iter(range(10**6))
    published: list[int] = []

    async def list_miners(): return miners
    async def current_block(): return next(blocks)
    async def publish(uid, hk=""):
        published.append(uid)
        # Cold-start publish + dethrone → after second call, stop the loop.
        if len(published) >= 2:
            import os, signal
            os.kill(os.getpid(), signal.SIGINT)
        return True

    chain = Chain(hotkey="V", list_miners=list_miners,
                  current_block=current_block, publish_winner=publish)

    cfg = Config(evidence_path=str(tmp_path / "ev.jsonl"),
                 environments=(EnvSpec(name="E", image="img", params={"timeout": 5}),),
                 k_init=0.5, k_final=0.5, k_halflife=1)   # easy to dethrone for the test

    await run(cfg, chain, slots=_FakeSlots())

    assert published[0] == 0        # cold start: first by block
    assert published[-1] == 1       # dethroned by uid 1
    # Evidence file has 2 rows per dwell iteration, balanced across miners.
    store_rows = EvidenceStore(tmp_path / "ev.jsonl").read()
    uids = [r.m for r in store_rows]
    assert uids.count(0) == uids.count(1) > 0


@pytest.mark.asyncio
async def test_dethrone_skipped_when_chal_deregisters_mid_dwell(tmp_path, monkeypatch):
    """If the challenger deregisters between duel start and verdict, the
    re-fetched `fresh` set must reject the dethrone — without this guard the
    validator publishes weights to a uid whose hotkey may have been recycled
    to a different operator. The cold-start publish for uid 0 must still
    happen; only the (would-be) dethrone publish for uid 1 must be suppressed."""
    from affine.loop import run

    monkeypatch.setattr("affine.loop._load_envs", AsyncMock(return_value={
        "E": (SimpleNamespace(), EnvSpec(name="E", image="img", params={"timeout": 5})),
    }))

    async def _run_one(wrapper, params, timeout, slot, seed, task_id=0):
        return (slot.model == "mc"), 0.01, 0
    monkeypatch.setattr("affine.loop.run_one", _run_one)

    full = [_miner(0, model="mk"), _miner(1, model="mc")]
    after_dereg = [_miner(0, model="mk")]
    list_calls = {"n": 0}
    async def list_miners():
        list_calls["n"] += 1
        # First call (top of iter) returns both — duel starts. Second call (fresh
        # check at verdict) returns only king — chal deregistered mid-duel.
        return full if list_calls["n"] <= 1 else after_dereg

    blocks = iter(range(10**6))
    async def current_block(): return next(blocks)
    published: list[int] = []
    async def publish(uid, hk=""):
        published.append(uid)
        # SIGINT after we've seen the cold-start publish — that's enough to verify
        # behavior. If a spurious dethrone publish lands, it lands BEFORE this.
        if len(published) >= 1:
            import os, signal
            os.kill(os.getpid(), signal.SIGINT)
        return True

    chain = Chain(hotkey="V", list_miners=list_miners,
                  current_block=current_block, publish_winner=publish)
    cfg = Config(evidence_path=str(tmp_path / "ev.jsonl"),
                 environments=(EnvSpec(name="E", image="img", params={"timeout": 5}),),
                 k_init=0.5, k_final=0.5, k_halflife=1)
    try:
        await asyncio.wait_for(run(cfg, chain, slots=_FakeSlots()), timeout=2.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass

    # Cold start published uid 0; the dethrone publish for uid 1 was suppressed.
    assert 1 not in published, f"deregistered chal must not be published; got {published}"


@pytest.mark.asyncio
async def test_dethrone_resilient_to_king_teardown_error(tmp_path, monkeypatch):
    """Regression: if slots.teardown(king_slot) raises during dethrone, the loop
    must still promote the challenger and publish the new champion. Without this
    the loop gets stuck — the slot pointer is never cleared, every iteration
    retries the same broken teardown, and the dethrone never publishes."""
    from affine.loop import run

    monkeypatch.setattr("affine.loop._load_envs", AsyncMock(return_value={
        "E": (SimpleNamespace(), EnvSpec(name="E", image="img", params={"timeout": 5})),
    }))


    async def _run_one(wrapper, params, timeout, slot, seed, task_id=0):
        return slot.model == "mc", 0.01, 0
    monkeypatch.setattr("affine.loop.run_one", _run_one)

    miners = [_miner(0, model="mk"), _miner(1, model="mc")]
    blocks = iter(range(10**6))
    published: list[int] = []

    async def list_miners(): return miners
    async def current_block(): return next(blocks)
    async def publish(uid, hk=""):
        published.append(uid)
        if len(published) >= 2:
            import os, signal
            os.kill(os.getpid(), signal.SIGINT)
        return True

    chain = Chain(hotkey="V", list_miners=list_miners,
                  current_block=current_block, publish_winner=publish)
    cfg = Config(evidence_path=str(tmp_path / "ev.jsonl"),
                 environments=(EnvSpec(name="E", image="img", params={"timeout": 5}),),
                 k_init=0.5, k_final=0.5, k_halflife=1)

    class _FlakyTeardown(_FakeSlots):
        def __init__(self):
            super().__init__()
            self.king_teardown_errors = 0
        async def teardown(self, slot):
            self.torn_down += 1
            if slot.model == "mk":
                self.king_teardown_errors += 1
                raise RuntimeError("targon delete failed")

    slots = _FlakyTeardown()
    # Iteration 2 may also surface the same flaky teardown via the chal_slot
    # path (challenger=mk after dethrone) — that is *not* the path under test,
    # so we tolerate cancellation/timeout and only assert the first two
    # publishes occurred (cold start + dethrone).
    try:
        await asyncio.wait_for(run(cfg, chain, slots=slots), timeout=2.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass

    assert published[:2] == [0, 1]
    assert slots.king_teardown_errors >= 1


@pytest.mark.asyncio
async def test_run_caches_king_across_duels_in_reign(tmp_path, monkeypatch):
    """King slot must be provisioned once per reign, not once per duel. Re-provisioning
    a 600GB model download between every challenger was the dominant live-mode cost."""
    from affine.loop import run

    monkeypatch.setattr("affine.loop._load_envs", AsyncMock(return_value={
        "E": (SimpleNamespace(), EnvSpec(name="E", image="img", params={"timeout": 5})),
    }))


    async def _run_one(wrapper, params, timeout, slot, seed, task_id=0):
        return slot.model == "mk", 0.01, 0   # king always wins → no dethrone
    monkeypatch.setattr("affine.loop.run_one", _run_one)

    # One strong king (uid 0), two weak challengers (uid 1, 2). After uid 0 is
    # seated, the queue should iterate 1 → 2 with king reused both times.
    miners = [_miner(0, model="mk"), _miner(1, model="mc1"), _miner(2, model="mc2")]
    blocks = iter(range(10**6))
    published: list[int] = []

    async def list_miners(): return miners
    async def current_block(): return next(blocks)
    async def publish(uid, hk=""):
        published.append(uid)
        return True

    async def sleep_killer(*a, **kw):
        # queue_exhausted or no-miners paths hit a 120s sleep; short-circuit and stop.
        import os, signal
        os.kill(os.getpid(), signal.SIGINT)
    monkeypatch.setattr("affine.loop.asyncio.sleep", sleep_killer)

    chain = Chain(hotkey="V", list_miners=list_miners,
                  current_block=current_block, publish_winner=publish)
    cfg = Config(evidence_path=str(tmp_path / "ev.jsonl"),
                 environments=(EnvSpec(name="E", image="img", params={"timeout": 5}),),
                 k_init=0.5, k_final=0.5, k_halflife=1)   # chal failing → z<-k Hold

    slots = _FakeSlots()
    await run(cfg, chain, slots=slots)

    # Exactly one king provision ('mk') across two duels. Challengers (mc1, mc2)
    # each provisioned once.
    king_provisions = sum(1 for m, _ in slots.provisions if m == "mk")
    chal_provisions = sum(1 for m, _ in slots.provisions if m.startswith("mc"))
    assert king_provisions == 1, f"king re-provisioned within reign: {slots.provisions}"
    assert chal_provisions >= 2, f"expected 2+ challenger provisions, got {slots.provisions}"


@pytest.mark.asyncio
async def test_chal_transient_advances_queue_then_exhaustion_sleeps(tmp_path, monkeypatch):
    """Targon-side chal failure: chal marked ATTEMPTED → queue empties →
    queue-exhaustion 120s sleep. No per-chal exponential backoff, no tight loop
    on a single chal. The same chal is re-tried after clear_attempted."""
    from affine.loop import run
    import httpx

    monkeypatch.setattr("affine.loop._load_envs", AsyncMock(return_value={
        "E": (SimpleNamespace(), EnvSpec(name="E", image="img", params={"timeout": 5})),
    }))

    async def _run_one(*a, **kw): return True, 0.01, 0
    monkeypatch.setattr("affine.loop.run_one", _run_one)

    sleep_durations: list[float] = []
    async def capture_sleep(d, *a, **kw):
        sleep_durations.append(d)
        if len(sleep_durations) >= 3:
            import os, signal
            os.kill(os.getpid(), signal.SIGINT)
    monkeypatch.setattr("affine.loop.asyncio.sleep", capture_sleep)

    miners = [_miner(0, model="mk"), _miner(1, model="mc")]
    async def list_miners(): return miners
    async def current_block(): return 0
    async def publish(uid, hk=""): return True
    chain = Chain(hotkey="V", list_miners=list_miners,
                  current_block=current_block, publish_winner=publish)
    cfg = Config(evidence_path=str(tmp_path / "ev.jsonl"),
                 environments=(EnvSpec(name="E", image="img", params={"timeout": 5}),),
                 k_init=10.0, k_final=10.0, k_halflife=1)

    class _ChalTransientSlots(_FakeSlots):
        async def provision(self, model, revision):
            if model.startswith("mc"):
                raise httpx.ConnectError("targon api unreachable")
            return await super().provision(model, revision)

    await run(cfg, chain, slots=_ChalTransientSlots())
    assert all(d == 120 for d in sleep_durations[:3]), (
        f"expected queue-exhaustion 120s cycles, got {sleep_durations[:3]}"
    )


@pytest.mark.asyncio
async def test_publish_failure_retries_next_iteration(tmp_path, monkeypatch):
    """Regression: if publish_winner returns False (set_weights rejected), the loop
    must NOT mark the publish as done. Otherwise validators silently believe they
    set weights when the chain never accepted them."""
    from affine.loop import run

    monkeypatch.setattr("affine.loop._load_envs", AsyncMock(return_value={
        "E": (SimpleNamespace(), EnvSpec(name="E", image="img", params={"timeout": 5})),
    }))

    async def _run_one(wrapper, params, timeout, slot, seed, task_id=0):
        return slot.model == "mk", 0.01, 0
    monkeypatch.setattr("affine.loop.run_one", _run_one)
    # Skip the queue-exhausted backoff so the test runs in <1s instead of 240s.
    monkeypatch.setattr("affine.loop.asyncio.sleep", AsyncMock())

    miners = [_miner(0, model="mk"), _miner(1, model="mc")]
    blocks = iter(range(10**6))
    calls: list[int] = []

    async def list_miners(): return miners
    async def current_block(): return next(blocks)

    async def publish(uid, hk=""):
        calls.append(uid)
        if len(calls) >= 3:
            import os, signal
            os.kill(os.getpid(), signal.SIGINT)
            return True
        return False

    chain = Chain(hotkey="V", list_miners=list_miners,
                  current_block=current_block, publish_winner=publish)
    cfg = Config(evidence_path=str(tmp_path / "ev.jsonl"),
                 environments=(EnvSpec(name="E", image="img", params={"timeout": 5}),),
                 k_init=0.5, k_final=0.5, k_halflife=1)   # chal fails every pick → z<-k Hold

    await run(cfg, chain, slots=_FakeSlots())

    # Failures 1+2 must not update last_published_uid; call 3 re-publishes the
    # same uid. All three calls target the cold-start champion (uid 0).
    assert calls == [0, 0, 0]


@pytest.mark.asyncio
async def test_dry_run_does_not_persist_last_published_uid(tmp_path, monkeypatch):
    """AFFINE_DRY_RUN=1 means set_weights returns True without mutating chain
    state. Persisting last_published_uid in dry-run would silently break a
    later real run sharing the same reign-state file: _maybe_publish would
    short-circuit on uid==last_published_uid and skip the actual on-chain write."""
    from affine.loop import Reign, run

    monkeypatch.setenv("AFFINE_DRY_RUN", "1")
    monkeypatch.setattr("affine.loop._load_envs", AsyncMock(return_value={
        "E": (SimpleNamespace(), EnvSpec(name="E", image="img", params={"timeout": 5})),
    }))

    async def _run_one(wrapper, params, timeout, slot, seed, task_id=0):
        return slot.model == "mk", 0.01, 0
    monkeypatch.setattr("affine.loop.run_one", _run_one)
    monkeypatch.setattr("affine.loop.asyncio.sleep", AsyncMock())

    miners = [_miner(0, model="mk"), _miner(1, model="mc")]
    blocks = iter(range(10**6))
    published: list[int] = []

    async def list_miners(): return miners
    async def current_block(): return next(blocks)
    async def publish(uid, hk=""):
        published.append(uid)
        if len(published) >= 1:
            import os, signal
            os.kill(os.getpid(), signal.SIGINT)
        return True   # dry-run mode in chain.set_weights also returns True

    chain = Chain(hotkey="V", list_miners=list_miners,
                  current_block=current_block, publish_winner=publish)
    cfg = Config(evidence_path=str(tmp_path / "ev.jsonl"),
                 environments=(EnvSpec(name="E", image="img", params={"timeout": 5}),),
                 k_init=0.5, k_final=0.5, k_halflife=1)

    await run(cfg, chain, slots=_FakeSlots())
    assert published, "dry-run should still call publish (audit/log path)"

    reign = Reign.load(tmp_path / "reign-V.json")
    assert reign is not None
    assert reign.last_published_uid is None, (
        "dry-run must NOT persist last_published_uid — a subsequent real run "
        "would short-circuit and silently skip the on-chain write"
    )


@pytest.mark.asyncio
async def test_dwell_interrupts_inflight_sample_on_stop(tmp_path):
    """Stop during a blocking sample must cancel the HTTP call rather than waiting
    up to env timeout. Regression: without _cancellable around the gather, SIGINT
    during a 600s task timeout would hang for up to 600s."""
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(evidence_path=str(tmp_path / "ev.jsonl"))
    stop = asyncio.Event()

    sample_started = asyncio.Event()
    async def slow_evaluate(*args, **kwargs):
        sample_started.set()
        await asyncio.sleep(60)
        return {"success": True}
    wrapper = SimpleNamespace(evaluate=AsyncMock(side_effect=slow_evaluate))
    envs = {"E": (wrapper, EnvSpec(name="E", image="img", params={"timeout": 600}))}

    async def trigger():
        await sample_started.wait(); stop.set()

    dwell_task = asyncio.create_task(dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], envs, ["E"], store, cfg, Priors(),
        np.random.default_rng(0), stop, reign_start_block=0,
    ))
    trigger_task = asyncio.create_task(trigger())
    out = await asyncio.wait_for(dwell_task, timeout=2.0)
    rows, _, status = out.rows, out.fit, out.status
    await trigger_task
    assert rows == []
    assert status is DuelStatus.CANCELLED


def test_seed_fits_signed_int32():
    """game's openspiel env feeds seed into np.random.RandomState which only
    accepts [0, 2^32-1], and internally computes seed+1..seed+100, so we mask
    to 2^31-1 (universal int32 ceiling) for portability across all envs."""
    maxv = (1 << 31) - 1
    for uid in range(16):
        for c in range(16):
            for rev in ("a", "rev-long-hex-" + "f" * 40):
                assert 0 <= _seed(uid, rev, "E", c) <= maxv


def test_task_id_matched_pair_and_varies_with_iter():
    from affine.loop import _task_id
    # King and challenger see the same task within an iteration regardless of order.
    a = _task_id(king_uid=3, chal_uid=7, env="E", iter_idx=0, lo=0, hi=99)
    b = _task_id(king_uid=7, chal_uid=3, env="E", iter_idx=0, lo=0, hi=99)
    assert a == b
    assert 0 <= a <= 99
    # Iteration index changes the task.
    seq = {_task_id(3, 7, "E", i, 0, 999_999) for i in range(8)}
    assert len(seq) == 8
    # Env changes the task.
    assert _task_id(3, 7, "A", 0, 0, 99) != _task_id(3, 7, "B", 0, 0, 99)


@pytest.mark.asyncio
async def test_dwell_persists_matched_task_id_per_iter(tmp_path, monkeypatch):
    """design.md §"Challenge-centric evidence": both miners face the same
    instance per iteration. End-to-end check that the rows actually persisted
    to evidence carry equal `i` for each (king, chal) pair within a dwell iter,
    while distinct iters draw distinct task_ids. Catches regressions that
    decouple king/chal task selection (e.g. per-uid task_id)."""
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="VAL", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(evidence_path=str(tmp_path / "ev.jsonl"))
    stop = asyncio.Event()
    async def _ok(wrapper, params, timeout, slot, seed, task_id=0):
        return True, 0.01, 0
    # All-pass → degenerate fit → no z-stop. Cap with stop after 12 pairs.
    monkeypatch.setattr("affine.loop.run_one", _stop_after_pairs(stop, 12, _ok))
    out = await dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], _env(), ["E"], store, cfg, Priors(),
        np.random.default_rng(0), stop, reign_start_block=0,
    )
    rows = store.read()
    assert len(rows) > 0 and len(rows) % 2 == 0
    # Group by (env, t, c-pair-index) — within a single iter, k_row immediately
    # precedes c_row (same store.append pair). The persistence order is what the
    # validator's matched-task contract relies on.
    pairs = list(zip(rows[::2], rows[1::2]))
    assert all(r0.m == king.uid and r1.m == chal.uid for r0, r1 in pairs)
    assert all(r0.i == r1.i for r0, r1 in pairs), \
        f"matched-task broken: pair task_ids: {[(p[0].i, p[1].i) for p in pairs]}"
    # And the iter-to-iter draws are diverse — a stuck task_id would also pass
    # the per-pair equality check, so this is the second axis.
    assert len({r.i for r, _ in pairs}) >= max(2, len(pairs) // 2)


@pytest.mark.asyncio
async def test_dwell_honors_stop_event(tmp_path):
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(evidence_path=str(tmp_path / "ev.jsonl"))
    stop = asyncio.Event(); stop.set()
    out = await dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], _env(), ["E"], store, cfg, Priors(),
        np.random.default_rng(0), stop, reign_start_block=0,
    )
    rows, fit, status = out.rows, out.fit, out.status
    assert rows == []
    assert status is DuelStatus.CANCELLED
