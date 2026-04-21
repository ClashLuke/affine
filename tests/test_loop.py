from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from affine.chain import Challenger
from affine.config import Config, EnvSpec
from affine.loop import BASELINES, Skiplist, _cold_start, _load_envs, _maybe_set_weights, _run_duel_with_retry, _try_provision
from affine.scoring import Verdict
from affine.vllm import Slot, SlotProvisionFailed


def _config(*, envs=()):
    return Config(
        netuid=120,
        wallet_name="cold",
        hotkey_name="hot",
        max_tasks_per_env=12,
        tasks_per_batch=2,
        health_check_timeout=5,
        environments=tuple(envs),
    )


def _chall(uid: int, block: int, model: str, revision: str | None = None):
    return Challenger(uid=uid, hotkey=f"hk{uid}", model=model, revision=revision or f"r{uid}", block=block)


def test_load_envs_merges_runtime_tokens_without_overriding_existing(monkeypatch):
    env_a = EnvSpec(name="a", image="img:a", env_vars={"HF_TOKEN": "from-spec"}, mem_limit="2g")
    env_b = EnvSpec(name="b", image="img:b", env_vars={"X": "1"}, mem_limit="3g")
    cfg = _config(envs=(env_a, env_b))

    monkeypatch.setenv("HF_TOKEN", "from-env")
    monkeypatch.setenv("CHUTES_API_KEY", "chute")

    wrappers = [object(), object()]
    with patch("affine.loop.af.load_env", side_effect=wrappers) as load_env:
        out = _load_envs(cfg)

    assert out["a"] == (wrappers[0], env_a)
    assert out["b"] == (wrappers[1], env_b)

    kwargs_a = load_env.call_args_list[0].kwargs
    kwargs_b = load_env.call_args_list[1].kwargs

    assert kwargs_a["env_vars"]["HF_TOKEN"] == "from-spec"
    assert kwargs_a["env_vars"]["CHUTES_API_KEY"] == "chute"

    assert kwargs_b["env_vars"]["HF_TOKEN"] == "from-env"
    assert kwargs_b["env_vars"]["CHUTES_API_KEY"] == "chute"


def test_skiplist_env_wildcard_matches_all_revisions(monkeypatch):
    monkeypatch.setenv("AFFINE_MODEL_SKIPLIST", "bad/model, other/m")
    sl = Skiplist.from_env()
    assert _chall(1, 10, "bad/model", "r1") in sl
    assert _chall(2, 10, "bad/model", "r2") in sl
    assert _chall(3, 10, "good/model", "r1") not in sl


def test_skiplist_pair_add_is_revision_specific():
    sl = Skiplist()
    sl.add("m", "r1")
    assert _chall(1, 10, "m", "r1") in sl
    assert _chall(2, 10, "m", "r2") not in sl


def test_skiplist_filter_drops_matching_entries():
    sl = Skiplist()
    sl.add("m", "r1")
    q = [_chall(1, 10, "m", "r1"), _chall(2, 11, "m", "r2"), _chall(3, 12, "other", "r1")]
    assert [c.uid for c in sl.filter(q)] == [2, 3]


def test_baselines_have_synthetic_uid():
    assert len(BASELINES) >= 1
    for b in BASELINES:
        assert b.uid < 0
        assert b.hotkey == ""
        assert b.model and b.revision


@pytest.mark.asyncio
async def test_cold_start_tries_baselines_first():
    baseline = BASELINES[0]
    baseline_slot = Slot(model=baseline.model, revision=baseline.revision, base_url="http://b")
    sub = SimpleNamespace(get_current_block=AsyncMock(return_value=500))
    slots = SimpleNamespace(provision=AsyncMock(return_value=baseline_slot), teardown=AsyncMock())
    queue_mock = AsyncMock()

    with patch("affine.loop.get_challenger_queue", queue_mock):
        chosen, slot, block = await _cold_start(sub, _config(), slots, Skiplist())

    assert chosen == baseline
    assert slot == baseline_slot
    assert block == 500
    queue_mock.assert_not_called()
    slots.provision.assert_awaited_once_with(baseline.model, baseline.revision)


@pytest.mark.asyncio
async def test_cold_start_falls_back_to_chain_queue_when_baselines_fail():
    miner = _chall(10, 50, "miner/x")
    miner_slot = Slot(model=miner.model, revision=miner.revision, base_url="http://m")
    sub = SimpleNamespace(get_current_block=AsyncMock(return_value=500))

    side_effects = [SlotProvisionFailed("baseline crashed")] * len(BASELINES) + [miner_slot]
    slots = SimpleNamespace(provision=AsyncMock(side_effect=side_effects), teardown=AsyncMock())

    with patch("affine.loop.get_challenger_queue", AsyncMock(return_value=[miner])):
        chosen, slot, _ = await _cold_start(sub, _config(), slots, Skiplist())

    assert chosen == miner
    assert slot == miner_slot


@pytest.mark.asyncio
async def test_cold_start_skips_skiplisted_baselines():
    baseline = BASELINES[0]
    skip = Skiplist()
    skip.add(baseline.model, baseline.revision)
    miner = _chall(10, 50, "miner/x")
    miner_slot = Slot(model=miner.model, revision=miner.revision, base_url="http://m")
    sub = SimpleNamespace(get_current_block=AsyncMock(return_value=500))
    slots = SimpleNamespace(provision=AsyncMock(return_value=miner_slot), teardown=AsyncMock())

    with patch("affine.loop.get_challenger_queue", AsyncMock(return_value=[miner])):
        chosen, _, _ = await _cold_start(sub, _config(), slots, skip)

    assert chosen == miner
    slots.provision.assert_awaited_once_with(miner.model, miner.revision)


@pytest.mark.asyncio
async def test_maybe_set_weights_skips_baseline():
    set_w = AsyncMock(return_value=True)
    with patch("affine.loop.set_weights", set_w):
        ok = await _maybe_set_weights(None, None, 120, BASELINES[0])
    assert ok is True
    set_w.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_set_weights_calls_for_real_miner():
    miner = _chall(7, 100, "m")
    set_w = AsyncMock(return_value=True)
    with patch("affine.loop.set_weights", set_w):
        ok = await _maybe_set_weights("sub", "wallet", 120, miner, retries=2)
    assert ok is True
    set_w.assert_awaited_once_with("sub", "wallet", 120, 7, retries=2)


@pytest.mark.asyncio
async def test_cold_start_skips_failed_candidate_and_uses_next():
    old = _chall(1, 10, "old")
    new = _chall(2, 20, "new")
    new_slot = Slot(model=new.model, revision=new.revision, base_url="http://new")
    sl = Skiplist()
    for b in BASELINES:
        sl.add(b.model, b.revision)

    sub = SimpleNamespace(get_current_block=AsyncMock(return_value=777))
    slots = SimpleNamespace(
        provision=AsyncMock(side_effect=[SlotProvisionFailed("bad"), new_slot]),
        teardown=AsyncMock(),
    )

    with patch("affine.loop.get_challenger_queue", AsyncMock(side_effect=[[], [old, new]])), patch(
        "affine.loop.asyncio.sleep", new_callable=AsyncMock
    ) as sleep:
        chosen, chosen_slot, crown_block = await _cold_start(sub, _config(), slots, sl)

    assert chosen == new
    assert chosen_slot == new_slot
    assert crown_block == 777
    assert old in sl
    assert new not in sl
    sleep.assert_awaited_once_with(120)


@pytest.mark.asyncio
async def test_try_provision_returns_none_on_generic_error_without_skiplisting():
    chall = _chall(7, 100, "model-x")
    slots = SimpleNamespace(provision=AsyncMock(side_effect=RuntimeError("transient")))
    sl = Skiplist()

    slot = await _try_provision(slots, chall, sl)

    assert slot is None
    assert chall not in sl


@pytest.mark.asyncio
async def test_try_provision_returns_none_and_skiplists_on_provision_failed():
    chall = _chall(7, 100, "model-x")
    slots = SimpleNamespace(provision=AsyncMock(side_effect=SlotProvisionFailed("crashloop")))
    sl = Skiplist()

    slot = await _try_provision(slots, chall, sl)

    assert slot is None
    assert chall in sl


@pytest.mark.asyncio
async def test_try_provision_success_returns_slot():
    chall = _chall(7, 100, "model-x")
    slot = Slot(model=chall.model, revision=chall.revision, base_url="http://x")
    slots = SimpleNamespace(provision=AsyncMock(return_value=slot))

    out = await _try_provision(slots, chall, Skiplist())

    assert out == slot


@pytest.mark.asyncio
async def test_run_duel_with_retry_recovers_after_first_failure_and_returns_winner():
    cfg = _config()
    chall = _chall(5, 321, "chall")
    champion_slot = Slot(model="champ", revision="rev", base_url="http://champ")
    slot1 = Slot(model="chall", revision="r1", base_url="http://c1")
    slot2 = Slot(model="chall", revision="r2", base_url="http://c2")
    slots = SimpleNamespace(teardown=AsyncMock())

    with patch("affine.loop._try_provision", AsyncMock(side_effect=[slot1, slot2])), patch(
        "affine.loop.run_duel", AsyncMock(side_effect=[RuntimeError("infra"), Verdict.CHALLENGER_WINS])
    ) as run_duel, patch("affine.loop.health_ping", AsyncMock(return_value=True)):
        verdict, winner_slot = await _run_duel_with_retry(
            slots,
            envs={"e": (object(), EnvSpec(name="e", image="img"))},
            champion_slot=champion_slot,
            chall=chall,
            config=cfg,
            k=2.0,
            hotkey="",
        )

    assert verdict is Verdict.CHALLENGER_WINS
    assert winner_slot == slot2
    assert run_duel.await_count == 2
    slots.teardown.assert_awaited_once_with(slot1)


@pytest.mark.e2e_fault
@pytest.mark.asyncio
async def test_run_duel_with_retry_aborts_if_champion_down_between_retries():
    cfg = _config()
    chall = _chall(5, 321, "chall")
    champion_slot = Slot(model="champ", revision="rev", base_url="http://champ")
    slot1 = Slot(model="chall", revision="r1", base_url="http://c1")
    slots = SimpleNamespace(teardown=AsyncMock())

    with patch("affine.loop._try_provision", AsyncMock(return_value=slot1)), patch(
        "affine.loop.run_duel", AsyncMock(side_effect=RuntimeError("infra"))
    ), patch("affine.loop.health_ping", AsyncMock(return_value=False)):
        verdict, winner_slot = await _run_duel_with_retry(
            slots,
            envs={"e": (object(), EnvSpec(name="e", image="img"))},
            champion_slot=champion_slot,
            chall=chall,
            config=cfg,
            k=2.0,
            hotkey="",
        )

    assert verdict is None
    assert winner_slot is None
    slots.teardown.assert_awaited_once_with(slot1)


@pytest.mark.asyncio
async def test_run_duel_with_retry_tears_down_losing_challenger_slot():
    cfg = _config()
    chall = _chall(5, 321, "chall")
    champion_slot = Slot(model="champ", revision="rev", base_url="http://champ")
    slot1 = Slot(model="chall", revision="r1", base_url="http://c1")
    slots = SimpleNamespace(teardown=AsyncMock())

    with patch("affine.loop._try_provision", AsyncMock(return_value=slot1)), patch(
        "affine.loop.run_duel", AsyncMock(return_value=Verdict.CHAMPION_HOLDS)
    ), patch("affine.loop.health_ping", AsyncMock(return_value=True)):
        verdict, winner_slot = await _run_duel_with_retry(
            slots,
            envs={"e": (object(), EnvSpec(name="e", image="img"))},
            champion_slot=champion_slot,
            chall=chall,
            config=cfg,
            k=2.0,
            hotkey="",
        )

    assert verdict is Verdict.CHAMPION_HOLDS
    assert winner_slot is None
    slots.teardown.assert_awaited_once_with(slot1)


@pytest.mark.asyncio
async def test_run_duel_with_retry_returns_none_when_provision_fails():
    cfg = _config()
    chall = _chall(5, 321, "chall")
    champion_slot = Slot(model="champ", revision="rev", base_url="http://champ")
    slots = SimpleNamespace(teardown=AsyncMock())

    with patch("affine.loop._try_provision", AsyncMock(return_value=None)), patch(
        "affine.loop.run_duel", AsyncMock()
    ) as run_duel, patch("affine.loop.health_ping", AsyncMock(return_value=True)):
        verdict, winner_slot = await _run_duel_with_retry(
            slots,
            envs={"e": (object(), EnvSpec(name="e", image="img"))},
            champion_slot=champion_slot,
            chall=chall,
            config=cfg,
            k=2.0,
            hotkey="",
        )

    assert verdict is None
    assert winner_slot is None
    run_duel.assert_not_called()


@pytest.mark.asyncio
async def test_run_duel_with_retry_does_not_skiplist_on_second_attempt_failure():
    cfg = _config()
    chall = _chall(5, 321, "chall")
    champion_slot = Slot(model="champ", revision="rev", base_url="http://champ")
    slot1 = Slot(model="chall", revision="r5", base_url="http://c1")
    slots = SimpleNamespace(
        provision=AsyncMock(side_effect=[slot1, SlotProvisionFailed("flaky")]),
        teardown=AsyncMock(),
    )
    sl = Skiplist()

    with patch("affine.loop.run_duel", AsyncMock(side_effect=RuntimeError("infra"))), patch(
        "affine.loop.health_ping", AsyncMock(return_value=True)
    ):
        verdict, winner = await _run_duel_with_retry(
            slots,
            envs={"e": (object(), EnvSpec(name="e", image="img"))},
            champion_slot=champion_slot,
            chall=chall,
            config=cfg,
            k=2.0,
            hotkey="",
            skip=sl,
        )

    assert verdict is None and winner is None
    assert chall not in sl, "miner that provisioned successfully once must not be skiplisted on retry"


@pytest.mark.asyncio
async def test_run_duel_with_retry_skiplists_on_first_attempt_failure():
    cfg = _config()
    chall = _chall(5, 321, "chall")
    champion_slot = Slot(model="champ", revision="rev", base_url="http://champ")
    slots = SimpleNamespace(
        provision=AsyncMock(side_effect=SlotProvisionFailed("crashloop")),
        teardown=AsyncMock(),
    )
    sl = Skiplist()

    with patch("affine.loop.run_duel", AsyncMock()), patch(
        "affine.loop.health_ping", AsyncMock(return_value=True)
    ):
        verdict, winner = await _run_duel_with_retry(
            slots,
            envs={"e": (object(), EnvSpec(name="e", image="img"))},
            champion_slot=champion_slot,
            chall=chall,
            config=cfg,
            k=2.0,
            hotkey="",
            skip=sl,
        )

    assert verdict is None and winner is None
    assert chall in sl


@pytest.mark.asyncio
async def test_try_provision_timeout_error_does_not_skiplist():
    chall = _chall(7, 100, "model-x")
    slots = SimpleNamespace(provision=AsyncMock(side_effect=TimeoutError("slow")))
    sl = Skiplist()

    slot = await _try_provision(slots, chall, sl)

    assert slot is None
    assert chall not in sl


@pytest.mark.asyncio
async def test_run_duel_with_retry_exhausts_attempts_after_repeated_failures():
    cfg = _config()
    chall = _chall(5, 321, "chall")
    champion_slot = Slot(model="champ", revision="rev", base_url="http://champ")
    slots_created = [
        Slot(model="chall", revision="r1", base_url="http://c1"),
        Slot(model="chall", revision="r2", base_url="http://c2"),
        Slot(model="chall", revision="r3", base_url="http://c3"),
    ]
    slots = SimpleNamespace(teardown=AsyncMock())

    with patch("affine.loop._try_provision", AsyncMock(side_effect=slots_created)), patch(
        "affine.loop.run_duel", AsyncMock(side_effect=[RuntimeError("x"), RuntimeError("y"), RuntimeError("z")])
    ) as run_duel, patch("affine.loop.health_ping", AsyncMock(return_value=True)):
        verdict, winner_slot = await _run_duel_with_retry(
            slots,
            envs={"e": (object(), EnvSpec(name="e", image="img"))},
            champion_slot=champion_slot,
            chall=chall,
            config=cfg,
            k=2.0,
            hotkey="",
        )

    assert verdict is None
    assert winner_slot is None
    assert run_duel.await_count == 3
    assert slots.teardown.await_count == 3
