from __future__ import annotations

import asyncio
import json
import signal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from affine.chain import Miner
from affine.config import Config, EnvSpec
from affine.loop import (
    SLOT_DEAD, Chain, _cancellable, _champion_registration_status,
    _install_signal_handlers,
    _is_current_champion_registration, _load_envs,
    _provision, _publish_champion, _run_fixed_duel, _seed, _shutdown_signals,
    static_chain,
)
from affine.paired import PairCounts
from affine.store import BackupRecord, Champion, Store, artifact_id
from affine.vllm import SlotProvisionFailed


def _miner(uid, model=None, rev="r"):
    return Miner(uid=uid, hotkey=f"hk{uid}", model=model or f"m{uid}",
                 revision=rev, block=uid)


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
    """Outer cancel racing worker completion must invoke on_orphan, otherwise
    the rental survives until reconcile."""
    torn_down: list = []

    async def fake_wait(aws, *a, **kw):
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
    async def provision(self, model, revision, **kwargs):
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
    import httpx
    stop = asyncio.Event()
    slots = _FakeSlotsProvision(httpx.ConnectError("targon api unreachable"))
    slot, status = await _provision(slots, _miner(7, model="m", rev="r"), stop)
    assert slot is None and status == "transient"


@pytest.mark.asyncio
async def test_provision_cancelled_propagates():
    stop = asyncio.Event(); stop.set()
    class _HangSlots:
        async def provision(self, model, revision, **kwargs):
            await asyncio.sleep(60)
        async def teardown(self, slot): pass
    with pytest.raises(asyncio.CancelledError):
        await _provision(_HangSlots(), _miner(7, model="m", rev="r"), stop)


def test_seed_fits_signed_int32():
    """np.random.RandomState only accepts [0, 2^32-1] and adds small offsets,
    so we mask to 2^31-1 for portability."""
    maxv = (1 << 31) - 1
    for uid in range(16):
        for c in range(16):
            for rev in ("a", "rev-long-hex-" + "f" * 40):
                assert 0 <= _seed(uid, rev, "E", c) <= maxv


def test_seed_depends_on_salt():
    a = _seed(7, "rev", "ded", 3, salt="hkA")
    b = _seed(7, "rev", "ded", 3, salt="hkB")
    assert a != b


def test_task_id_matched_pair_and_varies_with_iter():
    from affine.loop import _task_id
    a = _task_id(king_uid=3, chal_uid=7, env="E", iter_idx=0, lo=0, hi=99)
    b = _task_id(king_uid=7, chal_uid=3, env="E", iter_idx=0, lo=0, hi=99)
    assert a == b
    assert 0 <= a <= 99
    seq = {_task_id(3, 7, "E", i, 0, 999_999) for i in range(8)}
    assert len(seq) == 8
    assert _task_id(3, 7, "A", 0, 0, 99) != _task_id(3, 7, "B", 0, 0, 99)


def test_task_id_depends_on_salt():
    from affine.loop import _task_id
    a = _task_id(3, 7, "ded", 0, 0, 999, salt="hkA")
    b = _task_id(3, 7, "ded", 0, 0, 999, salt="hkB")
    assert a != b


@pytest.mark.asyncio
async def test_load_envs_builds_local_factories():
    entrypoint = "affine.envs.python_interpreter:PythonInterpreterEnv"
    cfg = Config(environments=(
        EnvSpec(name="a", entrypoint=entrypoint, params={"lines": 1}),
        EnvSpec(name="b", entrypoint=entrypoint, params={"lines": 2}),
    ))
    envs = await _load_envs(cfg)
    assert envs["a"][0] is envs["b"][0]
    assert envs["a"][1].params["lines"] == 1
    assert envs["b"][1].params["lines"] == 2
    assert hasattr(envs["a"][0].make(), "reset")


@pytest.mark.asyncio
async def test_load_envs_rejects_bad_entrypoint():
    cfg = Config(environments=(
        EnvSpec(name="bad", entrypoint="not_a_real_module:Env"),
    ))
    with pytest.raises(ModuleNotFoundError):
        await _load_envs(cfg)


def test_static_chain_returns_fixed_miners():
    miners = [_miner(0), _miner(1)]
    chain = static_chain(miners, hotkey="hk")
    assert chain.hotkey == "hk"
    assert asyncio.run(chain.list_miners()) == miners
    assert asyncio.run(chain.current_block()) == 0


@pytest.mark.asyncio
async def test_shutdown_signal_handlers_include_hup(monkeypatch):
    if not hasattr(signal, "SIGHUP"):
        pytest.skip("SIGHUP is POSIX-only")
    callbacks = {}
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", lambda sig, cb: callbacks.setdefault(sig, cb))
    stop = asyncio.Event()
    _install_signal_handlers(stop)
    assert set(_shutdown_signals()) >= {signal.SIGTERM, signal.SIGINT, signal.SIGHUP}
    callbacks[signal.SIGHUP]()
    assert stop.is_set()



@pytest.mark.asyncio
async def test_fixed_duel_does_not_spend_budget_on_infra_non_delivery(tmp_path, monkeypatch):
    champ_calls = 0

    async def _run_one(wrapper, params, timeout, slot, seed, task_id=0):
        nonlocal champ_calls
        if slot.model == "champ":
            champ_calls += 1
            if champ_calls <= 3:
                return None, 0.0, 0
            return False, 0.01, 1
        return True, 0.01, 1

    monkeypatch.setattr("affine.loop.run_one", _run_one)
    store = Store(tmp_path / "affine.sqlite3")
    champ_art = artifact_id("champ", "r")
    backup = BackupRecord(champ_art, "champ", "r", "p", "p/manifest.json", "sha", "verified")
    champ = Champion(champ_art, "champ", "r", 0, "hk0", 0, backup.manifest_key, backup.prefix, True)
    store.set_champion(champ, backup)
    duel = store.create_duel(
        champion=champ,
        challenger_uid=1,
        challenger_hotkey="hk1",
        challenger_model="chal",
        challenger_revision="r",
        schedule_seed="seed",
        pairs_per_env=4,
        min_discordant=1,
        alpha=1.0,
        started_block=0,
    )
    envs = {"E": (SimpleNamespace(), EnvSpec(name="E", entrypoint="img", params={"timeout": 5}))}
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())

    result = await _run_fixed_duel(
        store, chain,
        _miner(0, model="champ"), SimpleNamespace(model="champ", revision="r", base_url="champ"),
        _miner(1, model="chal"), SimpleNamespace(model="chal", revision="r", base_url="chal"),
        envs, Config(dwell_batch=2, environments=(EnvSpec(name="E", entrypoint="img", params={"timeout": 5}),)),
        duel, asyncio.Event(),
    )

    expected = PairCounts(challenger_only=4)
    assert result.status == "dethrone"
    assert result.counts == expected
    assert store.counts(duel.id) == expected
    rows = store.db.execute("SELECT champion_delivered, challenger_delivered FROM samples").fetchall()
    assert len(rows) == 7
    assert sum(1 for r in rows if r["champion_delivered"] and r["challenger_delivered"]) == 4
    assert sum(1 for r in rows if not r["champion_delivered"] and r["challenger_delivered"]) == 3
    store.close()


@pytest.mark.asyncio
async def test_fixed_duel_aborts_when_paired_delivery_never_progresses(tmp_path, monkeypatch):
    calls = 0

    async def _run_one(wrapper, params, timeout, slot, seed, task_id=0):
        nonlocal calls
        calls += 1
        champion_turn = calls % 4 in {1, 2}
        if slot.model == "champ":
            return (True, 0.01, 1) if champion_turn else (None, 0.0, 0)
        return (None, 0.0, 0) if champion_turn else (True, 0.01, 1)

    monkeypatch.setattr("affine.loop.run_one", _run_one)
    store = Store(tmp_path / "affine.sqlite3")
    champ_art = artifact_id("champ", "r")
    backup = BackupRecord(champ_art, "champ", "r", "p", "p/manifest.json", "sha", "verified")
    champ = Champion(champ_art, "champ", "r", 0, "hk0", 0, backup.manifest_key, backup.prefix, True)
    store.set_champion(champ, backup)
    duel = store.create_duel(
        champion=champ,
        challenger_uid=1,
        challenger_hotkey="hk1",
        challenger_model="chal",
        challenger_revision="r",
        schedule_seed="seed",
        pairs_per_env=4,
        min_discordant=1,
        alpha=1.0,
        started_block=0,
    )
    envs = {"E": (SimpleNamespace(), EnvSpec(name="E", entrypoint="img", params={"timeout": 5}))}
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())

    result = await _run_fixed_duel(
        store, chain,
        _miner(0, model="champ"), SimpleNamespace(model="champ", revision="r", base_url="champ"),
        _miner(1, model="chal"), SimpleNamespace(model="chal", revision="r", base_url="chal"),
        envs, Config(dwell_batch=1, environments=(EnvSpec(name="E", entrypoint="img", params={"timeout": 5}),)),
        duel, asyncio.Event(),
    )

    assert result.status == "delivery_stalled"
    assert result.counts == PairCounts()
    assert store.counts(duel.id) == PairCounts()
    store.close()


@pytest.mark.asyncio
async def test_fixed_duel_both_sides_dead_reports_delivery_stalled(tmp_path, monkeypatch):
    """Symmetric env failure (a 40k-token prompt rejected as 400 by both
    models) trips both side counters at once. Without the both-dead check
    this would teardown the king on shared infra noise."""
    async def _run_one(wrapper, params, timeout, slot, seed, task_id=0):
        return None, 0.0, 0
    monkeypatch.setattr("affine.loop.run_one", _run_one)
    store = Store(tmp_path / "affine.sqlite3")
    champ_art = artifact_id("champ", "r")
    backup = BackupRecord(champ_art, "champ", "r", "p", "p/manifest.json", "sha", "verified")
    champ = Champion(champ_art, "champ", "r", 0, "hk0", 0, backup.manifest_key, backup.prefix, True)
    store.set_champion(champ, backup)
    duel = store.create_duel(
        champion=champ, challenger_uid=1, challenger_hotkey="hk1",
        challenger_model="chal", challenger_revision="r", schedule_seed="seed",
        pairs_per_env=SLOT_DEAD * 2, min_discordant=1, alpha=1.0, started_block=0,
    )
    envs = {"E": (SimpleNamespace(), EnvSpec(name="E", entrypoint="img", params={"timeout": 5}))}
    chain = Chain(hotkey="V", list_miners=AsyncMock(),
                  current_block=AsyncMock(return_value=0), publish_winner=AsyncMock())
    result = await _run_fixed_duel(
        store, chain,
        _miner(0, model="champ"), SimpleNamespace(model="champ", revision="r", base_url="champ"),
        _miner(1, model="chal"), SimpleNamespace(model="chal", revision="r", base_url="chal"),
        envs, Config(dwell_batch=1, environments=(EnvSpec(name="E", entrypoint="img", params={"timeout": 5}),)),
        duel, asyncio.Event(),
    )
    assert result.status == "delivery_stalled"
    store.close()


@pytest.mark.asyncio
async def test_publish_champion_demotes_deregistered_sqlite_champion(tmp_path):
    store = Store(tmp_path / "affine.sqlite3")
    art = artifact_id("champ", "r")
    backup = BackupRecord(art, "champ", "r", "p", "p/manifest.json", "sha", "verified")
    champ = Champion(art, "champ", "r", 3, "hk3", 0, backup.manifest_key, backup.prefix, True)
    store.set_champion(champ, backup)
    burns = []

    async def burn():
        burns.append(True)
        return True

    chain = Chain(
        hotkey="V",
        list_miners=AsyncMock(),
        current_block=AsyncMock(return_value=0),
        publish_winner=AsyncMock(),
        burn_weights=burn,
        uid_matches_hotkey=AsyncMock(return_value=False),
    )
    assert await _publish_champion(store, chain, Config(), champ) is False

    demoted = store.champion()
    assert burns == [True]
    assert demoted.uid is None
    assert demoted.hotkey is None
    assert demoted.payable is False
    store.close()


@pytest.mark.asyncio
async def test_publish_champion_dry_run_does_not_call_chain(tmp_path, monkeypatch):
    monkeypatch.setenv("AFFINE_DRY_RUN", "1")
    store = Store(tmp_path / "affine.sqlite3")
    art = artifact_id("champ", "r")
    backup = BackupRecord(art, "champ", "r", "p", "p/manifest.json", "sha", "verified")
    champ = Champion(art, "champ", "r", 3, "hk3", 0, backup.manifest_key, backup.prefix, True)
    store.set_champion(champ, backup)
    chain = Chain(
        hotkey="V",
        list_miners=AsyncMock(),
        current_block=AsyncMock(return_value=0),
        publish_winner=AsyncMock(side_effect=AssertionError("published")),
        burn_weights=AsyncMock(side_effect=AssertionError("burned")),
        uid_matches_hotkey=AsyncMock(return_value=True),
    )

    assert await _publish_champion(store, chain, Config(dry_run=True), champ) is True
    store.close()


@pytest.mark.asyncio
async def test_champion_registration_requires_same_artifact(monkeypatch):
    champ = Champion(
        artifact_id=artifact_id("m", "sha-old"),
        model="m",
        revision="sha-old",
        uid=4,
        hotkey="hk4",
        reign_start=0,
        backup_manifest="manifest",
        backup_prefix="prefix",
        payable=True,
    )
    current = _miner(4, model="m", rev="main")
    monkeypatch.setattr("affine.loop._pin_hf_revision", lambda model, rev: "sha-new")

    status = await _champion_registration_status(object(), [current], champ, missing_ref_alive=True)

    assert status != "alive"
    assert _is_current_champion_registration(current, champ, artifact_alive=False) is False


@pytest.mark.asyncio
async def test_champion_registration_uses_backup_when_hf_ref_is_gone(monkeypatch):
    import httpx
    from huggingface_hub.errors import RevisionNotFoundError

    champ = Champion(
        artifact_id=artifact_id("m", "sha-old"),
        model="m",
        revision="sha-old",
        uid=4,
        hotkey="hk4",
        reign_start=0,
        backup_manifest="manifest",
        backup_prefix="prefix",
        payable=True,
    )
    current = _miner(4, model="m", rev="main")

    def missing(_model, _rev):
        req = httpx.Request("GET", "https://huggingface.co/m/resolve/main")
        raise RevisionNotFoundError("missing", response=httpx.Response(404, request=req))

    monkeypatch.setattr("affine.loop._pin_hf_revision", missing)

    assert await _champion_registration_status(
        object(), [current], champ, missing_ref_alive=True,
    ) == "alive"
    assert await _champion_registration_status(
        object(), [current], champ, missing_ref_alive=False,
    ) == "dead"


@pytest.mark.asyncio
async def test_bootstrap_uses_registered_baseline_when_available(tmp_path, monkeypatch):
    from affine.loop import _LocalBackupManager, _bootstrap_champion

    monkeypatch.setattr("affine.loop.BASELINE_MODELS", ("mk",))
    miner = _miner(7, model="mk", rev="r")
    slots_seen = []

    class Slots:
        async def provision(self, model, revision, **kwargs):
            slots_seen.append((model, revision, kwargs.get("source", "hf")))
            return SimpleNamespace(model=model, revision=revision, base_url="http://slot", slot_id="slot")

    chain = Chain(hotkey="V", list_miners=AsyncMock(return_value=[miner]),
                  current_block=AsyncMock(return_value=10), publish_winner=AsyncMock())
    store = Store(tmp_path / "affine.sqlite3")

    champ, _slot = await _bootstrap_champion(
        store, _LocalBackupManager(tmp_path / "backups"), Slots(), chain,
        Config(db_path=str(tmp_path / "affine.sqlite3")), asyncio.Event(),
    )

    assert champ.uid == 7
    assert champ.hotkey == "hk7"
    assert champ.payable is True
    assert slots_seen == [("mk", "r", "hf")]
    store.close()


@pytest.mark.asyncio
async def test_sqlite_run_bootstraps_via_hf_and_promotes_backup_source(tmp_path, monkeypatch):
    from affine.loop import run

    monkeypatch.setattr("affine.loop.BASELINE_MODELS", ("mk",))
    monkeypatch.setattr("affine.loop._load_envs", AsyncMock(return_value={
        "E": (SimpleNamespace(), EnvSpec(name="E", entrypoint="img", params={"timeout": 5})),
    }))

    async def _run_one(wrapper, params, timeout, slot, seed, task_id=0):
        return slot.model == "mc", 0.01, 1
    monkeypatch.setattr("affine.loop.run_one", _run_one)

    class Slots:
        def __init__(self):
            self.provisions = []
            self.teardowns = 0

        async def provision(self, model, revision, **kwargs):
            self.provisions.append((model, revision, kwargs.get("source", "hf")))
            return SimpleNamespace(model=model, revision=revision,
                                   base_url=f"http://fake/{model}", slot_id=f"s-{model}")

        async def teardown(self, slot):
            self.teardowns += 1

    blocks = iter(range(10**6))
    published = []

    async def list_miners():
        return [_miner(1, model="mc", rev="r")]

    async def current_block():
        return next(blocks)

    async def publish(uid, hk=""):
        published.append(uid)
        import os, signal as _signal
        os.kill(os.getpid(), _signal.SIGINT)
        return True

    async def burn():
        return True

    cfg = Config(
        db_path=str(tmp_path / "affine.sqlite3"),
        environments=(EnvSpec(name="E", entrypoint="img", params={"timeout": 5}),),
        dwell_batch=1,
        duel_pairs_per_env=1,
        duel_min_discordant=1,
        alpha_start=0.9,
        alpha_final=0.9,
    )
    slots = Slots()
    await run(cfg, Chain(hotkey="V", list_miners=list_miners,
                         current_block=current_block, publish_winner=publish,
                         burn_weights=burn), slots=slots)

    store = Store(cfg.db_path)
    champ = store.champion()
    assert champ.uid == 1
    assert champ.model == "mc"
    assert published == [1]
    assert ("mk", "main", "hf") in slots.provisions
    assert ("mc", "r", "s3") in slots.provisions
    store.close()


@pytest.mark.asyncio
async def test_sqlite_run_keeps_champion_when_promotion_backup_restore_fails(tmp_path, monkeypatch):
    from affine.loop import run

    monkeypatch.setattr("affine.loop.BASELINE_MODELS", ("mk",))
    monkeypatch.setattr("affine.loop._load_envs", AsyncMock(return_value={
        "E": (SimpleNamespace(), EnvSpec(name="E", entrypoint="img", params={"timeout": 5})),
    }))

    async def _run_one(wrapper, params, timeout, slot, seed, task_id=0):
        return slot.model == "mc", 0.01, 1
    monkeypatch.setattr("affine.loop.run_one", _run_one)

    async def sleep_killer(*_args, **_kwargs):
        import os, signal as _signal
        os.kill(os.getpid(), _signal.SIGINT)
    monkeypatch.setattr("affine.loop.asyncio.sleep", sleep_killer)

    class Slots:
        async def provision(self, model, revision, **kwargs):
            if kwargs.get("source") == "s3":
                raise RuntimeError("no restore image")
            return SimpleNamespace(model=model, revision=revision, base_url=f"http://fake/{model}", slot_id=model)

        async def teardown(self, slot):
            pass

    miners = [_miner(0, model="mk"), _miner(1, model="mc")]
    chain = Chain(hotkey="V", list_miners=AsyncMock(return_value=miners),
                  current_block=AsyncMock(return_value=0),
                  publish_winner=AsyncMock(return_value=True))
    cfg = Config(
        db_path=str(tmp_path / "affine.sqlite3"),
        environments=(EnvSpec(name="E", entrypoint="img", params={"timeout": 5}),),
        dwell_batch=1,
        duel_pairs_per_env=1,
        duel_min_discordant=1,
        alpha_start=0.9,
        alpha_final=0.9,
    )

    await run(cfg, chain, slots=Slots())

    store = Store(cfg.db_path)
    champ = store.champion()
    assert champ.model == "mk"
    assert champ.uid == 0
    assert store.pending_champion() is None
    assert chain.publish_winner.await_args.args == (0, "hk0")
    store.close()


@pytest.mark.asyncio
async def test_sqlite_run_rechecks_challenger_identity_after_backup_restore(tmp_path, monkeypatch):
    from affine.loop import run

    monkeypatch.setattr("affine.loop._load_envs", AsyncMock(return_value={
        "E": (SimpleNamespace(), EnvSpec(name="E", entrypoint="img", params={"timeout": 5})),
    }))

    async def _run_one(wrapper, params, timeout, slot, seed, task_id=0):
        return slot.model == "chal", 0.01, 1
    monkeypatch.setattr("affine.loop.run_one", _run_one)

    async def sleep_killer(*_args, **_kwargs):
        import os, signal as _signal
        os.kill(os.getpid(), _signal.SIGINT)
    monkeypatch.setattr("affine.loop.asyncio.sleep", sleep_killer)

    db = tmp_path / "affine.sqlite3"
    store = Store(db)
    champ_art = artifact_id("champ", "r")
    backup = BackupRecord(champ_art, "champ", "r", "p", "p/manifest.json", "sha", "verified")
    champ = Champion(champ_art, "champ", "r", 0, "hk0", 0, backup.manifest_key, backup.prefix, True)
    store.set_champion(champ, backup)
    store.close()

    class Slots:
        async def provision(self, model, revision, **kwargs):
            return SimpleNamespace(model=model, revision=revision, base_url=f"http://fake/{model}", slot_id=model)

        async def teardown(self, slot):
            pass

    old = _miner(0, model="champ", rev="r")
    challenger = _miner(1, model="chal", rev="r")
    calls = 0

    async def list_miners():
        nonlocal calls
        calls += 1
        return [old, challenger] if calls <= 2 else [old]

    chain = Chain(hotkey="V", list_miners=list_miners,
                  current_block=AsyncMock(return_value=0),
                  publish_winner=AsyncMock(return_value=True),
                  uid_matches_hotkey=AsyncMock(return_value=True))
    cfg = Config(
        db_path=str(db),
        environments=(EnvSpec(name="E", entrypoint="img", params={"timeout": 5}),),
        dwell_batch=1,
        duel_pairs_per_env=1,
        duel_min_discordant=1,
        alpha_start=0.9,
        alpha_final=0.9,
    )

    await run(cfg, chain, slots=Slots())

    reopened = Store(db)
    assert reopened.champion() == champ
    assert reopened.pending_champion() is None
    assert chain.publish_winner.await_args.args == (0, "hk0")
    reopened.close()


@pytest.mark.asyncio
async def test_sqlite_run_backs_up_challenger_only_after_dethrone(tmp_path, monkeypatch):
    from affine.loop import run

    monkeypatch.setattr("affine.loop.BASELINE_MODELS", ("mk",))
    monkeypatch.setattr("affine.loop._load_envs", AsyncMock(return_value={
        "E": (SimpleNamespace(), EnvSpec(name="E", entrypoint="img", params={"timeout": 5})),
    }))

    backups = []

    async def _backup(_backup_mgr, model, revision):
        backups.append((model, revision))
        art = artifact_id(model, revision)
        return BackupRecord(art, model, revision, f"p-{model}", f"{model}/manifest.json", "sha", "verified")
    monkeypatch.setattr("affine.loop._backup_artifact", _backup)

    async def _run_one(wrapper, params, timeout, slot, seed, task_id=0):
        return slot.model == "mk", 0.01, 1
    monkeypatch.setattr("affine.loop.run_one", _run_one)

    async def sleep_killer(*_args, **_kwargs):
        import os, signal as _signal
        os.kill(os.getpid(), _signal.SIGINT)
    monkeypatch.setattr("affine.loop.asyncio.sleep", sleep_killer)

    class Slots:
        async def provision(self, model, revision, **kwargs):
            return SimpleNamespace(model=model, revision=revision, base_url=f"http://fake/{model}", slot_id=model)

        async def teardown(self, slot):
            pass

    miners = [_miner(0, model="mk"), _miner(1, model="mc")]
    chain = Chain(hotkey="V", list_miners=AsyncMock(return_value=miners),
                  current_block=AsyncMock(return_value=0),
                  publish_winner=AsyncMock(return_value=True))
    cfg = Config(
        db_path=str(tmp_path / "affine.sqlite3"),
        environments=(EnvSpec(name="E", entrypoint="img", params={"timeout": 5}),),
        dwell_batch=1,
        duel_pairs_per_env=1,
        duel_min_discordant=1,
        alpha_start=0.9,
        alpha_final=0.9,
    )

    await run(cfg, chain, slots=Slots())

    assert backups == [("mk", "r")]


@pytest.mark.asyncio
async def test_run_discards_dead_pending_champion_before_publication(tmp_path, monkeypatch):
    from affine.loop import run

    monkeypatch.setattr("affine.loop._load_envs", AsyncMock(return_value={}))

    async def sleep_killer(*_args, **_kwargs):
        import os, signal as _signal
        os.kill(os.getpid(), _signal.SIGINT)
    monkeypatch.setattr("affine.loop.asyncio.sleep", sleep_killer)

    db = tmp_path / "affine.sqlite3"
    store = Store(db)
    current_art = artifact_id("champ", "r")
    pending_art = artifact_id("chal", "r")
    current_backup = BackupRecord(current_art, "champ", "r", "p1", "p1/manifest.json", "sha1", "verified")
    pending_backup = BackupRecord(pending_art, "chal", "r", "p2", "p2/manifest.json", "sha2", "verified")
    current = Champion(current_art, "champ", "r", 1, "hk1", 0, current_backup.manifest_key, current_backup.prefix, True)
    pending = Champion(pending_art, "chal", "r", 2, "hk2", 1, pending_backup.manifest_key, pending_backup.prefix, True)
    store.set_champion(current, current_backup)
    store.set_pending_champion(pending, pending_backup)
    store.close()

    class Slots:
        async def provision(self, model, revision, **kwargs):
            return SimpleNamespace(model=model, revision=revision, base_url=f"http://fake/{model}", slot_id=model)

        async def teardown(self, slot):
            pass

    chain = Chain(
        hotkey="V",
        list_miners=AsyncMock(return_value=[_miner(1, model="champ", rev="r")]),
        current_block=AsyncMock(return_value=0),
        publish_winner=AsyncMock(return_value=True),
        uid_matches_hotkey=AsyncMock(return_value=True),
    )
    cfg = Config(db_path=str(db), environments=())

    await run(cfg, chain, slots=Slots())

    reopened = Store(db)
    assert reopened.champion() == current
    assert reopened.pending_champion() is None
    assert reopened.db.execute("SELECT 1 FROM backups WHERE manifest_key=?", (pending_backup.manifest_key,)).fetchone() is None
    assert chain.publish_winner.await_args.args == (1, "hk1")
    reopened.close()


@pytest.mark.asyncio
async def test_run_restores_pending_backup_before_publication(tmp_path, monkeypatch):
    from affine.loop import run

    monkeypatch.setattr("affine.loop._load_envs", AsyncMock(return_value={}))

    async def sleep_killer(*_args, **_kwargs):
        import os, signal as _signal
        os.kill(os.getpid(), _signal.SIGINT)
    monkeypatch.setattr("affine.loop.asyncio.sleep", sleep_killer)

    db = tmp_path / "affine.sqlite3"
    store = Store(db)
    old_art = artifact_id("old", "r")
    new_art = artifact_id("new", "r")
    old_backup = BackupRecord(old_art, "old", "r", "p-old", "old/manifest.json", "sha-old", "verified")
    new_backup = BackupRecord(new_art, "new", "r", "p-new", "new/manifest.json", "sha-new", "verified")
    old = Champion(old_art, "old", "r", 1, "hk1", 0, old_backup.manifest_key, old_backup.prefix, True)
    pending = Champion(new_art, "new", "r", 2, "hk2", 1, new_backup.manifest_key, new_backup.prefix, True)
    store.set_champion(old, old_backup)
    store.set_pending_champion(pending, new_backup)
    store.close()

    events = []

    class Slots:
        async def provision(self, model, revision, **kwargs):
            events.append(("provision", kwargs.get("source", "hf"), model))
            return SimpleNamespace(model=model, revision=revision, base_url=f"http://fake/{model}", slot_id=model)

        async def teardown(self, slot):
            events.append(("teardown", slot.model))

    async def publish(uid, hk):
        events.append(("publish", uid))
        return True

    chain = Chain(
        hotkey="V",
        list_miners=AsyncMock(return_value=[_miner(2, model="new", rev="r")]),
        current_block=AsyncMock(return_value=0),
        publish_winner=publish,
        uid_matches_hotkey=AsyncMock(return_value=True),
    )
    cfg = Config(db_path=str(db), environments=())

    await run(cfg, chain, slots=Slots())

    reopened = Store(db)
    assert reopened.champion() == pending
    assert reopened.pending_champion() is None
    assert events[:2] == [("provision", "s3", "new"), ("publish", 2)]
    assert reopened.db.execute("SELECT 1 FROM backups WHERE manifest_key=?", (old_backup.manifest_key,)).fetchone() is None
    reopened.close()
