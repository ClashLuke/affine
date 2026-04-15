from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from affine.chain import Challenger
from affine.config import Config, EnvSpec
from affine.loop import _cold_start, _load_envs, _run_duel_with_retry, _try_provision
from affine.scoring import Verdict
from affine.vllm import Slot


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


def _chall(uid: int, block: int, model: str):
    return Challenger(uid=uid, hotkey=f"hk{uid}", model=model, revision=f"r{uid}", block=block)


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


@pytest.mark.asyncio
async def test_cold_start_waits_then_uses_earliest_viable_candidate():
    old = _chall(1, 10, "old")
    new = _chall(2, 20, "new")
    old_slot = Slot(model=old.model, revision=old.revision, base_url="http://old")
    new_slot = Slot(model=new.model, revision=new.revision, base_url="http://new")

    sub = SimpleNamespace(get_current_block=AsyncMock(return_value=777))
    slots = SimpleNamespace(
        provision=AsyncMock(side_effect=[old_slot, new_slot]),
        teardown=AsyncMock(),
    )

    with patch("affine.loop.get_challenger_queue", AsyncMock(side_effect=[[], [old, new]])), patch(
        "affine.loop.health_check", AsyncMock(side_effect=[False, True])
    ), patch("affine.loop.asyncio.sleep", new_callable=AsyncMock) as sleep:
        chosen, chosen_slot, crown_block = await _cold_start(sub, _config(), slots)

    assert chosen == new
    assert chosen_slot == new_slot
    assert crown_block == 777
    assert slots.provision.await_args_list[0].args == (old.model, old.revision)
    assert slots.provision.await_args_list[1].args == (new.model, new.revision)
    slots.teardown.assert_awaited_once_with(old_slot)
    sleep.assert_awaited_once_with(120)


@pytest.mark.asyncio
async def test_try_provision_returns_none_on_provision_error():
    chall = _chall(7, 100, "model-x")
    slots = SimpleNamespace(provision=AsyncMock(side_effect=RuntimeError("no capacity")), teardown=AsyncMock())

    with patch("affine.loop.health_check", AsyncMock(return_value=True)) as hc:
        slot = await _try_provision(slots, chall, _config())

    assert slot is None
    hc.assert_not_called()
    slots.teardown.assert_not_called()


@pytest.mark.asyncio
async def test_try_provision_tears_down_unhealthy_slot():
    chall = _chall(7, 100, "model-x")
    slot = Slot(model=chall.model, revision=chall.revision, base_url="http://x")
    slots = SimpleNamespace(provision=AsyncMock(return_value=slot), teardown=AsyncMock())

    with patch("affine.loop.health_check", AsyncMock(return_value=False)):
        out = await _try_provision(slots, chall, _config())

    assert out is None
    slots.teardown.assert_awaited_once_with(slot)


@pytest.mark.asyncio
async def test_try_provision_success_returns_slot():
    chall = _chall(7, 100, "model-x")
    slot = Slot(model=chall.model, revision=chall.revision, base_url="http://x")
    slots = SimpleNamespace(provision=AsyncMock(return_value=slot), teardown=AsyncMock())

    with patch("affine.loop.health_check", AsyncMock(return_value=True)):
        out = await _try_provision(slots, chall, _config())

    assert out == slot
    slots.teardown.assert_not_called()


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
