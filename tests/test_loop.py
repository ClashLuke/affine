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
    ENV_FAIL_QUARANTINE, Chain, Skiplist, _apply_skip, _cancellable, _dwell, _fit,
    _load_envs, _load_reign_state, _provision, _provision_pair, _respondents,
    _save_reign_state, _seed, static_chain,
)
from affine.vllm import SlotProvisionFailed


def _miner(uid, model=None, rev="r"):
    return Miner(uid=uid, hotkey=f"hk{uid}", model=model or f"m{uid}",
                 revision=rev, block=uid)


def _row(**kw):
    d = dict(m=0, r="r", e="E", c=0, p=1, t=0, l=0.0)
    d.update(kw)
    return Row(**d)


def test_skiplist_filters_by_model_and_pair():
    s = Skiplist({"banned"})
    s.add("okmodel", "badrev")
    kept = s.filter([
        _miner(0, model="banned"),
        _miner(1, model="okmodel", rev="badrev"),
        _miner(2, model="okmodel", rev="goodrev"),
    ])
    assert [m.uid for m in kept] == [2]


def test_skiplist_durable_persists_and_reloads(tmp_path):
    path = tmp_path / "skip.jsonl"
    s = Skiplist(path=path)
    s.add("modelA", "rev1", durable=True)
    s.add("modelB", "rev2", durable=False)

    s2 = Skiplist(path=path)
    assert s2.filter([_miner(0, model="modelA", rev="rev1")]) == []
    assert s2.filter([_miner(1, model="modelB", rev="rev2")]) == [_miner(1, model="modelB", rev="rev2")]
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert lines == [{"model": "modelA", "revision": "rev1"}]


def test_skiplist_add_is_idempotent(tmp_path):
    path = tmp_path / "skip.jsonl"
    s = Skiplist(path=path)
    s.add("m", "r", durable=True)
    s.add("m", "r", durable=True)
    assert len(path.read_text().splitlines()) == 1


def test_reign_state_roundtrip(tmp_path):
    p = tmp_path / "reign.json"
    assert _load_reign_state(p) == (None, 0, None)
    _save_reign_state(p, (7, "rev-abc"), 12345, 7)
    champ, rs, lp = _load_reign_state(p)
    assert champ == (7, "rev-abc") and rs == 12345 and lp == 7


def test_reign_state_legacy_missing_published_uid(tmp_path):
    """Old reign.json files predate last_published_uid. Loader must default
    last_published_uid=None when the field is missing — first iteration after
    upgrade then re-publishes once, becomes idempotent thereafter."""
    p = tmp_path / "reign.json"
    p.write_text('{"uid": 3, "revision": "r", "reign_start": 100}')
    assert _load_reign_state(p) == ((3, "r"), 100, None)


def test_reign_state_corrupt_file_resets(tmp_path):
    """A truncated/garbled reign.json must not crash startup. Cold path =
    re-elect from chain state, lose the in-flight reign — strictly better
    than refusing to boot."""
    p = tmp_path / "reign.json"
    p.write_text("not-json")
    assert _load_reign_state(p) == (None, 0, None)
    p.write_text('{"uid": "not-int", "revision": "r", "reign_start": 1}')
    assert _load_reign_state(p) == (None, 0, None)


def test_reign_state_atomic_replace(tmp_path):
    """_save_reign_state must not leave a half-written file. A SIGKILL between
    open() and write() of a direct overwrite would zero the file; we use a
    temp + os.replace, which is atomic on POSIX."""
    p = tmp_path / "reign.json"
    _save_reign_state(p, (1, "r1"), 100)
    _save_reign_state(p, (2, "r2"), 200)
    assert not (tmp_path / "reign.json.tmp").exists()
    assert _load_reign_state(p) == ((2, "r2"), 200, None)


def test_skiplist_durable_write_fsyncs(tmp_path, monkeypatch):
    """Regression: a SIGKILL between buffered f.write and OS flush would lose
    the durable mark, re-admitting a known-crashloop model on restart and burning
    Targon credits. The write must be a single os.write + fsync, mirroring
    EvidenceStore.append_pair."""
    import os
    s = Skiplist(path=tmp_path / "skip.jsonl")
    written = []
    fsynced = []
    real_write, real_fsync = os.write, os.fsync
    monkeypatch.setattr(os, "write", lambda fd, data: (written.append(data), real_write(fd, data))[1])
    monkeypatch.setattr(os, "fsync", lambda fd: (fsynced.append(fd), real_fsync(fd))[1])
    s.add("m", "r", durable=True)
    assert len(written) == 1
    assert written[0] == (json.dumps({"model": "m", "revision": "r"}) + "\n").encode()
    assert len(fsynced) == 1


def test_skiplist_durable_disk_failure_does_not_mutate_memory(tmp_path):
    """Regression: in-memory mutation must not precede the disk write. If the
    write raises (full disk, parent path is a file, perms), the in-memory state
    would say "skipped" while disk knows nothing — restart silently re-admits."""
    path = tmp_path / "subdir" / "skip.jsonl"
    path.parent.write_text("not-a-dir")    # parent path occupied by a regular file
    s = Skiplist(path=path)
    miner = _miner(0, model="m", rev="r")
    with pytest.raises((FileExistsError, NotADirectoryError, OSError)):
        s.add("m", "r", durable=True)
    assert miner not in s   # in-memory not mutated


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
    skip = Skiplist(path=tmp_path / "skip.jsonl")
    _apply_skip(skip, _miner(7, model="m", rev="r"), "crashloop", is_king=False)
    assert not skip.filter([_miner(7, model="m", rev="r")])
    assert (tmp_path / "skip.jsonl").read_text().strip() != ""


def test_apply_skip_timeout_session_for_non_king(tmp_path):
    """Provision timeout could be Targon infra; skip this session only — a
    restart after a Targon outage should recover."""
    skip_path = tmp_path / "skip.jsonl"
    skip = Skiplist(path=skip_path)
    _apply_skip(skip, _miner(7, model="m", rev="r"), "timeout", is_king=False)
    assert not skip.filter([_miner(7, model="m", rev="r")])
    assert not skip_path.exists() or skip_path.read_text().strip() == ""


def test_apply_skip_timeout_noop_for_king(tmp_path):
    """The current king cannot be session-skipped on timeout; that would force
    re-election on every Targon hiccup and burn the weight quota."""
    skip = Skiplist(path=tmp_path / "skip.jsonl")
    _apply_skip(skip, _miner(7, model="m", rev="r"), "timeout", is_king=True)
    assert skip.filter([_miner(7, model="m", rev="r")]) == [_miner(7, model="m", rev="r")]


def test_apply_skip_error_session_for_non_king(tmp_path):
    """Generic provision exception (e.g. rental register returning no uid) must
    session-skip the chal — without this the same chal artifact is re-picked
    every iter, _provision_pair fail-fasts king, and the loop never advances."""
    skip_path = tmp_path / "skip.jsonl"
    skip = Skiplist(path=skip_path)
    _apply_skip(skip, _miner(7, model="m", rev="r"), "error", is_king=False)
    assert not skip.filter([_miner(7, model="m", rev="r")])
    assert not skip_path.exists() or skip_path.read_text().strip() == ""


def test_apply_skip_transient_noop_for_chal(tmp_path):
    """Transient httpx errors (Targon API blip) must NOT session-skip the chal —
    a 30s outage would otherwise empty the queue of every challenger that races
    the outage. They retry next iter once Targon recovers."""
    skip_path = tmp_path / "skip.jsonl"
    skip = Skiplist(path=skip_path)
    m = _miner(7, model="m", rev="r")
    _apply_skip(skip, m, "transient", is_king=False)
    assert skip.filter([m]) == [m]
    assert not skip_path.exists() or skip_path.read_text().strip() == ""


def test_apply_skip_error_noop_for_king(tmp_path):
    """King with unclassified error must not session-skip — could be transient
    Targon network blip; re-provision rather than disqualify the champion."""
    skip = Skiplist(path=tmp_path / "skip.jsonl")
    _apply_skip(skip, _miner(7, model="m", rev="r"), "error", is_king=True)
    assert skip.filter([_miner(7, model="m", rev="r")]) == [_miner(7, model="m", rev="r")]


def test_apply_skip_unhealthy_session_for_non_king(tmp_path):
    """Post-provision probe failed after 3x retries — session-skip to avoid
    tight retry on a model whose inference path is dead."""
    skip_path = tmp_path / "skip.jsonl"
    skip = Skiplist(path=skip_path)
    _apply_skip(skip, _miner(7, model="m", rev="r"), "unhealthy", is_king=False)
    assert not skip.filter([_miner(7, model="m", rev="r")])
    assert not skip_path.exists() or skip_path.read_text().strip() == ""


def test_apply_skip_unhealthy_noop_for_king(tmp_path):
    """Cached king failing inference probe must not session-skip — re-provision
    the artifact rather than disqualify the champion mid-reign."""
    skip = Skiplist(path=tmp_path / "skip.jsonl")
    _apply_skip(skip, _miner(7, model="m", rev="r"), "unhealthy", is_king=True)
    assert skip.filter([_miner(7, model="m", rev="r")]) == [_miner(7, model="m", rev="r")]


def test_apply_skip_chal_sharing_king_artifact_exempt(tmp_path):
    """Two miners committing the same (model, revision): a session/durable skip on
    the chal would filter king out on the next iter, costing the cached king slot.
    The cached slot's actual behavior is the authoritative signal for that artifact."""
    skip = Skiplist(path=tmp_path / "skip.jsonl")
    chal = _miner(8, model="shared", rev="r1")
    _apply_skip(skip, chal, "crashloop", is_king=False, king_artifact=("shared", "r1"))
    assert skip.filter([chal]) == [chal]
    _apply_skip(skip, chal, "timeout", is_king=False, king_artifact=("shared", "r1"))
    assert skip.filter([chal]) == [chal]


def test_apply_skip_king_crashloop_still_durable(tmp_path):
    """The king itself crashlooping must still durable-skip — exempting on
    artifact-match would let a broken king poison reign forever."""
    skip = Skiplist(path=tmp_path / "skip.jsonl")
    king = _miner(7, model="shared", rev="r1")
    _apply_skip(skip, king, "crashloop", is_king=True, king_artifact=("shared", "r1"))
    assert skip.filter([king]) == []


@pytest.mark.asyncio
async def test_provision_pair_skips_shared_artifact_on_chal_crashloop(tmp_path):
    """Regression: when king and chal share (model, revision), the same-artifact
    exemption used to suppress the skip on chal — but king's task was cancelled by
    fail-fast and got no skip either, so neither side was marked. Next iteration
    re-elected the same pair and crashlooped again forever. Fix: exemption only
    applies when caller holds a cached, proven-healthy king (not in this path)."""
    skip = Skiplist(path=tmp_path / "skip.jsonl")
    stop = asyncio.Event()

    class _Slots:
        async def provision(self, model, revision):
            from affine.vllm import SlotProvisionFailed
            # Whoever runs first crashes: chal happens to win the race below.
            await asyncio.sleep(0.01)
            raise SlotProvisionFailed("crashloop")
        async def teardown(self, slot):
            pass

    from affine import loop as loop_mod
    async def _ok(*a, **kw): return True
    orig_h, orig_i = loop_mod.health_ping, loop_mod.inference_ping
    loop_mod.health_ping = _ok; loop_mod.inference_ping = _ok
    try:
        await _provision_pair(_Slots(),
                              _miner(0, model="shared", rev="r1"),
                              _miner(1, model="shared", rev="r1"),
                              skip, stop)
    finally:
        loop_mod.health_ping = orig_h; loop_mod.inference_ping = orig_i

    # The crashlooping artifact must be durable-skipped from at least one side.
    # The previous behavior exempted both → empty skiplist → infinite retry.
    assert skip.filter([_miner(0, model="shared", rev="r1")]) == []


@pytest.mark.asyncio
async def test_provision_pair_fail_fast_cancels_sibling(tmp_path):
    """If king crashloops, abort challenger provisioning immediately rather than
    burning Targon time on a doomed pair."""
    skip = Skiplist(path=tmp_path / "skip.jsonl")
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

    from affine import loop as loop_mod
    async def _ok(*a, **kw): return True
    orig_h, orig_i = loop_mod.health_ping, loop_mod.inference_ping
    loop_mod.health_ping = _ok; loop_mod.inference_ping = _ok
    try:
        k_slot, c_slot, king_failed = await _provision_pair(_Slots(),
                                               _miner(0, model="king"), _miner(1, model="chal"),
                                               skip, stop)
    finally:
        loop_mod.health_ping = orig_h; loop_mod.inference_ping = orig_i

    assert k_slot is None and c_slot is None
    assert king_failed, "king's task ran to completion with crashloop status"
    assert chal_finished.is_set(), "challenger task should have terminated (cancelled)"
    assert teardowns == [], "no slot was produced, nothing to tear down"
    # king crashloop should durable-skip
    assert not skip.filter([_miner(0, model="king")])


@pytest.mark.asyncio
async def test_provision_pair_chal_fails_first_does_not_blame_king(tmp_path):
    """Regression: chal raises a generic Exception (status='error'), fail-fast
    cancels king mid-download. Caller must distinguish this from a real king
    failure: blaming king triggers a 60→600s sleep that delays advancing to the
    next challenger and `attempted.discard`s the broken chal so it gets re-picked
    forever. Returning king_attempt_failed=False lets the caller skip the backoff
    and rely on chal's session-skip (status='error' now writes one) to advance."""
    skip = Skiplist(path=tmp_path / "skip.jsonl")
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

    from affine import loop as loop_mod
    async def _ok(*a, **kw): return True
    orig_h, orig_i = loop_mod.health_ping, loop_mod.inference_ping
    loop_mod.health_ping = _ok; loop_mod.inference_ping = _ok
    try:
        k_slot, c_slot, king_failed = await _provision_pair(_Slots(),
                                                            _miner(0, model="king"), _miner(1, model="chal"),
                                                            skip, stop)
    finally:
        loop_mod.health_ping = orig_h; loop_mod.inference_ping = orig_i

    assert k_slot is None and c_slot is None
    assert king_failed is False, "king was cancelled by chal-fail-fast, not a real king failure"
    # chal status='error' now session-skips so the loop can advance.
    assert not skip.filter([_miner(1, model="chal")])


@pytest.mark.asyncio
async def test_provision_pair_king_succeeds_chal_fails_retains_king(tmp_path):
    """Regression: when king completes successfully BEFORE chal returns its
    failure (e.g. king cached on Targon, chal cold-starts and 5xxs), the older
    code tore the king slot down at the post-loop fail-fast check. Re-provisioning
    a 600GB king for every chal-only failure was the dominant cost on a churn
    of bad challengers. _provision_pair must retain king and only return chal=None."""
    skip = Skiplist(path=tmp_path / "skip.jsonl")
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

    from affine import loop as loop_mod
    async def _ok(*a, **kw): return True
    orig_h, orig_i = loop_mod.health_ping, loop_mod.inference_ping
    loop_mod.health_ping = _ok; loop_mod.inference_ping = _ok
    try:
        k_slot, c_slot, king_failed = await _provision_pair(_Slots(),
                                                             _miner(0, model="king"), _miner(1, model="chal"),
                                                             skip, stop)
    finally:
        loop_mod.health_ping = orig_h; loop_mod.inference_ping = orig_i

    assert k_slot is not None and k_slot.model == "king"
    assert c_slot is None
    assert king_failed is False
    assert teardowns == [], "king slot must NOT be torn down — it caches across chal failures"


@pytest.mark.asyncio
async def test_provision_pair_king_fails_chal_succeeds_tears_chal(tmp_path):
    """Mirror of the cache-retention test: when chal succeeds but king fails
    genuinely, chal must be torn down (it's useless without the king)."""
    skip = Skiplist(path=tmp_path / "skip.jsonl")
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

    from affine import loop as loop_mod
    async def _ok(*a, **kw): return True
    orig_h, orig_i = loop_mod.health_ping, loop_mod.inference_ping
    loop_mod.health_ping = _ok; loop_mod.inference_ping = _ok
    try:
        k_slot, c_slot, king_failed = await _provision_pair(_Slots(),
                                                             _miner(0, model="king"), _miner(1, model="chal"),
                                                             skip, stop)
    finally:
        loop_mod.health_ping = orig_h; loop_mod.inference_ping = orig_i

    assert k_slot is None and c_slot is None
    assert king_failed is True
    assert teardowns == ["chal"], "chal slot must be torn down when king fails — chal alone can't duel"


@pytest.mark.asyncio
async def test_provision_pair_partial_failure_tears_down_survivor(tmp_path):
    """If gather sees one task cancelled and the other returned a slot, that slot
    must be torn down — otherwise the loop leaks a 600GB rented vLLM on every
    SIGTERM-during-provision."""
    skip = Skiplist(path=tmp_path / "skip.jsonl")
    stop = asyncio.Event()
    teardowns: list[str] = []
    class _MixedSlots:
        async def provision(self, model, revision):
            if model == "king":
                return SimpleNamespace(model=model, base_url="http://k", revision=revision)
            await asyncio.sleep(60)
        async def teardown(self, slot):
            teardowns.append(slot.model)
    # Patch _cancellable's view: we want one to succeed, the other to be cancelled
    # on stop. health_ping/inference_ping would otherwise block — patch them too.
    from affine import loop as loop_mod
    async def _ok(*a, **kw): return True
    orig_h, orig_i = loop_mod.health_ping, loop_mod.inference_ping
    loop_mod.health_ping = _ok; loop_mod.inference_ping = _ok
    try:
        async def fire():
            await asyncio.sleep(0.05); stop.set()
        fire_task = asyncio.create_task(fire())
        with pytest.raises(asyncio.CancelledError):
            await _provision_pair(_MixedSlots(),
                                  _miner(0, model="king"), _miner(1, model="chal"),
                                  skip, stop)
        await fire_task
    finally:
        loop_mod.health_ping = orig_h; loop_mod.inference_ping = orig_i
    assert teardowns == ["king"]


def test_respondents_registered_first_then_ghosts():
    miners = [_miner(1, rev="v1"), _miner(2, rev="v1")]
    rows = [
        _row(m=2, r="v1"),       # registered
        _row(m=99, r="v1"),      # ghost
        _row(m=1, r="v0"),       # ghost (old rev of registered uid)
    ]
    keys = _respondents(miners, rows)
    assert keys[:2] == [(1, "v1"), (2, "v1")]
    assert set(keys[2:]) == {(99, "v1"), (1, "v0")}


def test_fit_ignores_unknown_envs():
    miners = [_miner(0)]
    rows = [_row(m=0, e="E1"), _row(m=0, e="EX")]
    fit = _fit(rows, miners, env_names=["E1"], priors=Priors())
    assert fit.n_m == 1 and fit.n_e == 1


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


@pytest.mark.asyncio
async def test_dwell_appends_two_rows_per_env_pick(tmp_path, monkeypatch):
    """Each fisher_env pick must append exactly two rows (king + challenger)."""
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=42), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(dwell=5, evidence_path=str(tmp_path / "ev.jsonl"))
    n = [0]
    async def _run_one(wrapper, params, timeout, slot, seed, task_id=0):
        n[0] += 1
        return bool(n[0] % 2), 0.01
    monkeypatch.setattr("affine.loop.run_one", _run_one)
    rows, fit, _abort = await _dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], _env(), ["E"], store, cfg, Priors(),
        np.random.default_rng(0), asyncio.Event(),
    )
    assert len(rows) == 2 * cfg.dwell
    uids = [r.m for r in store.read()]
    assert uids.count(0) == cfg.dwell and uids.count(1) == cfg.dwell
    assert fit.n_m == 2


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
    cfg = Config(dwell=10, evidence_path=str(tmp_path / "ev.jsonl"))
    async def _all_pass(wrapper, params, timeout, slot, seed, task_id=0):
        return True, 0.01
    monkeypatch.setattr("affine.loop.run_one", _all_pass)
    envs = {
        "A": (SimpleNamespace(evaluate=AsyncMock()), EnvSpec(name="A", image="i", params={"timeout": 5})),
        "B": (SimpleNamespace(evaluate=AsyncMock()), EnvSpec(name="B", image="i", params={"timeout": 5})),
    }
    rows, _, _abort = await _dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], envs, ["A", "B"], store, cfg, Priors(),
        np.random.default_rng(0), asyncio.Event(),
    )
    assert len(rows) == 2 * cfg.dwell, "dwell must use full budget despite all-pass degeneracy"


@pytest.mark.asyncio
async def test_dwell_aborts_on_infra_streak(tmp_path):
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(dwell=100, evidence_path=str(tmp_path / "ev.jsonl"))
    rows, fit, _abort = await _dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], _env(err=True), ["E"], store, cfg, Priors(),
        np.random.default_rng(0), asyncio.Event(),
    )
    assert rows == []
    assert store.read() == []
    assert fit.n_m == 2


@pytest.mark.asyncio
async def test_dwell_quarantines_broken_env_keeps_sampling_healthy(tmp_path):
    """One broken env, one healthy env: dwell must quarantine the broken one
    after ENV_FAIL_QUARANTINE fails and keep collecting rows from the healthy
    one for the remaining budget."""
    broken_calls = healthy_calls = 0
    async def evaluate_broken(*a, **kw):
        nonlocal broken_calls; broken_calls += 1
        raise RuntimeError("env_broken")
    async def evaluate_healthy(*a, **kw):
        nonlocal healthy_calls; healthy_calls += 1
        return {"success": True}
    envs = {
        "bad":  (SimpleNamespace(evaluate=AsyncMock(side_effect=evaluate_broken)),
                 EnvSpec(name="bad",  image="i1", params={"timeout": 5})),
        "good": (SimpleNamespace(evaluate=AsyncMock(side_effect=evaluate_healthy)),
                 EnvSpec(name="good", image="i2", params={"timeout": 5})),
    }
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(dwell=30, evidence_path=str(tmp_path / "ev.jsonl"))
    rows, _, _abort = await _dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], envs, ["bad", "good"], store, cfg, Priors(),
        np.random.default_rng(0), asyncio.Event(),
    )
    by_env = {r.e: 0 for r in rows}
    for r in rows: by_env[r.e] = by_env.get(r.e, 0) + 1
    assert by_env.get("bad", 0) == 0                                  # broken env contributed zero rows
    assert by_env.get("good", 0) > 0                                  # healthy env kept producing
    # Bad env gets ≤ ENV_FAIL_QUARANTINE × 2 evaluate calls (2 for king+chal concurrent per pick)
    assert broken_calls <= 2 * ENV_FAIL_QUARANTINE


@pytest.mark.asyncio
async def test_dwell_aborts_with_chal_broken_when_only_chal_returns_none(tmp_path, monkeypatch):
    """Regression: when chal is bad (every sample → None) but king is fine, env_fails
    used to count toward quarantine. After ENV_FAIL_QUARANTINE × n_envs picks, the
    dwell aborts with no rows and the broken chal stays on the queue. Now it
    attributes failure to chal-side and aborts after SIDE_FAIL_THRESHOLD picks
    across diverse envs."""
    async def split(wrapper, params, timeout, slot, seed, task_id=0):
        return (True if slot.model == "mk" else None), 0.01
    monkeypatch.setattr("affine.loop.run_one", split)
    envs = {n: (SimpleNamespace(evaluate=AsyncMock()), EnvSpec(name=n, image="i", params={"timeout": 5}))
            for n in ("A", "B", "C")}
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(dwell=30, evidence_path=str(tmp_path / "ev.jsonl"))
    rows, _, abort = await _dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], envs, list(envs), store, cfg, Priors(),
        np.random.default_rng(0), asyncio.Event(),
    )
    assert abort == "chal_broken"
    assert rows == []   # no successful pairs


@pytest.mark.asyncio
async def test_dwell_aborts_with_king_broken_when_only_king_returns_none(tmp_path, monkeypatch):
    """Mirror: king-only-fails (vLLM memory leak surviving 1-token health probe)
    must surface as 'king_broken' so the loop drops the slot. Without this the
    king reigns forever — the 1-token cached-king probe doesn't catch it."""
    async def split(wrapper, params, timeout, slot, seed, task_id=0):
        return (None if slot.model == "mk" else True), 0.01
    monkeypatch.setattr("affine.loop.run_one", split)
    envs = {n: (SimpleNamespace(evaluate=AsyncMock()), EnvSpec(name=n, image="i", params={"timeout": 5}))
            for n in ("A", "B", "C")}
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(dwell=30, evidence_path=str(tmp_path / "ev.jsonl"))
    rows, _, abort = await _dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], envs, list(envs), store, cfg, Priors(),
        np.random.default_rng(0), asyncio.Event(),
    )
    assert abort == "king_broken"
    assert rows == []


@pytest.mark.asyncio
async def test_dwell_streak_resets_on_ambiguous_both_none(tmp_path, monkeypatch):
    """Both-None is *ambiguous* evidence: the env failed, can't attribute to a side.
    Without resetting side streaks on both-None, a chal-None / both-None alternation
    spuriously trips chal_broken even though the side signal is interleaved with
    env-side noise. Fix: reset streaks on ambiguous evidence; let env_fails carry
    the ambiguous signal."""
    counter = {"n": 0}
    async def alternating(wrapper, params, timeout, slot, seed, task_id=0):
        counter["n"] += 1
        # Per call (king and chal each call run_one): odd picks chal-None / king-ok,
        # even picks both-None. We tag by slot.model so each side decides on its own.
        # Pick index from king's call count: floor((n+1)/2).
        pick = (counter["n"] + 1) // 2
        if pick % 2 == 1:                                                 # odd: chal-only fail
            return (True if slot.model == "mk" else None), 0.01
        return None, 0.01                                                 # even: both-None
    monkeypatch.setattr("affine.loop.run_one", alternating)
    envs = {"X": (SimpleNamespace(evaluate=AsyncMock()), EnvSpec(name="X", image="i", params={"timeout": 5}))}
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(dwell=20, evidence_path=str(tmp_path / "ev.jsonl"))
    rows, _, abort = await _dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], envs, list(envs), store, cfg, Priors(),
        np.random.default_rng(0), asyncio.Event(),
    )
    # Before fix: chal_streak grows 1,1,2,2,3 → "chal_broken". After fix: streak
    # resets on both-None, env_fails carries the ambiguous signal → "envs_quarantined".
    assert abort != "chal_broken", "ambiguous both-None must not feed side-broken signal"


@pytest.mark.asyncio
async def test_dwell_env_fails_resets_on_single_side_success(tmp_path, monkeypatch):
    """env_fails[e] tracks consecutive *both-None* failures on env e — the env-side
    signal. A single-side success means the env produced a valid result for at least
    one model: the env is fine. Without resetting env_fails on single-side success,
    a long-running session accumulates spurious quarantine pressure across picks."""
    seq: list[tuple[bool | None, bool | None]] = [
        (None, None),                                                     # pick 1: both-None → env_fails=1
        (None, None),                                                     # pick 2: both-None → env_fails=2
        (True, None),                                                     # pick 3: chal-None, king-ok → env_fails should reset
        (None, None),                                                     # pick 4: both-None → env_fails=1 (fix) vs 3=quarantine (bug)
        (None, None),                                                     # pick 5: both-None → env_fails=2 (fix)
    ]
    pick = {"i": 0}
    last_pick = {"slot_count": 0}
    async def scripted(wrapper, params, timeout, slot, seed, task_id=0):
        last_pick["slot_count"] += 1
        idx = (last_pick["slot_count"] + 1) // 2 - 1
        if idx >= len(seq):
            return False, 0.01                                            # productive failures keep dwell going
        k, c = seq[idx]
        return (k if slot.model == "mk" else c), 0.01
    monkeypatch.setattr("affine.loop.run_one", scripted)
    envs = {"X": (SimpleNamespace(evaluate=AsyncMock()), EnvSpec(name="X", image="i", params={"timeout": 5}))}
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(dwell=10, evidence_path=str(tmp_path / "ev.jsonl"))
    rows, _, abort = await _dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], envs, list(envs), store, cfg, Priors(),
        np.random.default_rng(0), asyncio.Event(),
    )
    # After fix, picks 6+ produce both-False (productive failures) → rows accumulate.
    # Before fix, abort at pick 4 (envs_quarantined) → no rows from later picks.
    assert abort != "envs_quarantined", "env_fails must reset after single-side success"
    assert len(rows) > 0, "dwell must reach productive picks after fix"


@pytest.mark.asyncio
async def test_dwell_intermittent_one_side_does_not_abort(tmp_path, monkeypatch):
    """Verify the streak resets on success: alternating None / pass on chal
    must NOT trigger chal_broken because the streak never reaches threshold."""
    counter = {"n": 0}
    async def alternating(wrapper, params, timeout, slot, seed, task_id=0):
        if slot.model == "mk":
            return True, 0.01
        # chal: fail every 4th sample but otherwise succeed
        counter["n"] += 1
        return (None if counter["n"] % 4 == 0 else False), 0.01
    monkeypatch.setattr("affine.loop.run_one", alternating)
    envs = {"A": (SimpleNamespace(evaluate=AsyncMock()), EnvSpec(name="A", image="i", params={"timeout": 5}))}
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(dwell=20, evidence_path=str(tmp_path / "ev.jsonl"))
    rows, _, abort = await _dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], envs, list(envs), store, cfg, Priors(),
        np.random.default_rng(0), asyncio.Event(),
    )
    assert abort is None, "intermittent chal failures must not flip to chal_broken"
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

    # Patch env loader and health_ping (neither exercise real containers).
    monkeypatch.setattr("affine.loop._load_envs", AsyncMock(return_value={
        "E": (SimpleNamespace(), EnvSpec(name="E", image="img", params={"timeout": 5})),
    }))
    monkeypatch.setattr("affine.loop.health_ping", AsyncMock(return_value=True))
    monkeypatch.setattr("affine.loop.inference_ping", AsyncMock(return_value=True))

    async def _run_one(wrapper, params, timeout, slot, seed, task_id=0):
        return (slot.model == "mc"), 0.01   # mc always wins, mk always loses
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

    cfg = Config(dwell=8, evidence_path=str(tmp_path / "ev.jsonl"),
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
async def test_dethrone_resilient_to_king_teardown_error(tmp_path, monkeypatch):
    """Regression: if slots.teardown(king_slot) raises during dethrone, the loop
    must still promote the challenger and publish the new champion. Without this
    the loop gets stuck — the slot pointer is never cleared, every iteration
    retries the same broken teardown, and the dethrone never publishes."""
    from affine.loop import run

    monkeypatch.setattr("affine.loop._load_envs", AsyncMock(return_value={
        "E": (SimpleNamespace(), EnvSpec(name="E", image="img", params={"timeout": 5})),
    }))
    monkeypatch.setattr("affine.loop.health_ping", AsyncMock(return_value=True))
    monkeypatch.setattr("affine.loop.inference_ping", AsyncMock(return_value=True))

    async def _run_one(wrapper, params, timeout, slot, seed, task_id=0):
        return slot.model == "mc", 0.01
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
    cfg = Config(dwell=8, evidence_path=str(tmp_path / "ev.jsonl"),
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
    monkeypatch.setattr("affine.loop.health_ping", AsyncMock(return_value=True))
    monkeypatch.setattr("affine.loop.inference_ping", AsyncMock(return_value=True))

    async def _run_one(wrapper, params, timeout, slot, seed, task_id=0):
        return slot.model == "mk", 0.01   # king always wins → no dethrone
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
    cfg = Config(dwell=3, evidence_path=str(tmp_path / "ev.jsonl"),
                 environments=(EnvSpec(name="E", image="img", params={"timeout": 5}),),
                 k_init=10.0, k_final=10.0, k_halflife=1)   # unreachable → no dethrone

    slots = _FakeSlots()
    await run(cfg, chain, slots=slots)

    # Exactly one king provision ('mk') across two duels. Challengers (mc1, mc2)
    # each provisioned once.
    king_provisions = sum(1 for m, _ in slots.provisions if m == "mk")
    chal_provisions = sum(1 for m, _ in slots.provisions if m.startswith("mc"))
    assert king_provisions == 1, f"king re-provisioned within reign: {slots.provisions}"
    assert chal_provisions >= 2, f"expected 2+ challenger provisions, got {slots.provisions}"


@pytest.mark.asyncio
async def test_publish_failure_retries_next_iteration(tmp_path, monkeypatch):
    """Regression: if publish_winner returns False (set_weights rejected), the loop
    must NOT mark the publish as done. Otherwise validators silently believe they
    set weights when the chain never accepted them."""
    from affine.loop import run

    monkeypatch.setattr("affine.loop._load_envs", AsyncMock(return_value={
        "E": (SimpleNamespace(), EnvSpec(name="E", image="img", params={"timeout": 5})),
    }))
    monkeypatch.setattr("affine.loop.health_ping", AsyncMock(return_value=True))
    monkeypatch.setattr("affine.loop.inference_ping", AsyncMock(return_value=True))
    async def _run_one(wrapper, params, timeout, slot, seed, task_id=0):
        return slot.model == "mk", 0.01
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
    cfg = Config(dwell=2, evidence_path=str(tmp_path / "ev.jsonl"),
                 environments=(EnvSpec(name="E", image="img", params={"timeout": 5}),),
                 k_init=10.0, k_final=10.0, k_halflife=1)

    await run(cfg, chain, slots=_FakeSlots())

    # Failures 1+2 must not update last_published_uid; call 3 re-publishes the
    # same uid. All three calls target the cold-start champion (uid 0).
    assert calls == [0, 0, 0]


@pytest.mark.asyncio
async def test_dwell_interrupts_inflight_sample_on_stop(tmp_path):
    """Stop during a blocking sample must cancel the HTTP call rather than waiting
    up to env timeout. Regression: without _cancellable around the gather, SIGINT
    during a 600s task timeout would hang for up to 600s."""
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(dwell=5, evidence_path=str(tmp_path / "ev.jsonl"))
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

    dwell_task = asyncio.create_task(_dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], envs, ["E"], store, cfg, Priors(),
        np.random.default_rng(0), stop,
    ))
    trigger_task = asyncio.create_task(trigger())
    rows, _, _abort = await asyncio.wait_for(dwell_task, timeout=2.0)
    await trigger_task
    assert rows == []


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
async def test_dwell_honors_stop_event(tmp_path):
    store = EvidenceStore(tmp_path / "ev.jsonl")
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())
    king, chal = _miner(0, model="mk"), _miner(1, model="mc")
    cfg = Config(dwell=100, evidence_path=str(tmp_path / "ev.jsonl"))
    stop = asyncio.Event(); stop.set()
    rows, fit, _abort = await _dwell(
        chain, king, SimpleNamespace(model="mk", base_url="uk"),
        chal, SimpleNamespace(model="mc", base_url="uc"),
        [king, chal], [], _env(), ["E"], store, cfg, Priors(),
        np.random.default_rng(0), stop,
    )
    assert rows == []
