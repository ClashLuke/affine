import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from affine.config import EnvSpec
from affine.duel import _batch_rng, _eval, _master_seed, run_duel
from affine.scoring import Verdict
from affine.vllm import Slot


class RecorderEnv:
    def __init__(self, success=True):
        self.success = success
        self.calls = []

    async def evaluate(self, model, base_url, seed, task_id, **params):
        self.calls.append((model, seed, task_id, params))
        return {"success": self.success}


class ScriptEnv:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def evaluate(self, model, base_url, seed, task_id, **params):
        self.calls.append((model, seed, task_id, params))
        if not self.script:
            raise AssertionError("script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return {"success": item}


class ModelMapEnv:
    def __init__(self, outcomes: dict[str, bool]):
        self.outcomes = outcomes
        self.calls = []

    async def evaluate(self, model, base_url, seed, task_id, **params):
        self.calls.append((model, seed, task_id, params))
        return {"success": self.outcomes[model]}


class SlowEnv:
    async def evaluate(self, model, base_url, seed, task_id, **params):
        await asyncio.sleep(0.05)
        return {"success": True}


class ErrorEnv:
    async def evaluate(self, model, base_url, seed, task_id, **params):
        raise RuntimeError("bad eval")


def _envs(wrapper, timeout=1):
    spec = EnvSpec(name="env", image="img", params={"timeout": timeout})
    return {"env": (wrapper, spec)}


def test_master_seed_deterministic():
    champ = Slot("model-a", "rev-1", "http://a")
    chall = Slot("model-b", "rev-2", "http://b")
    assert _master_seed(champ, chall, nonce=7) == _master_seed(champ, chall, nonce=7)
    assert _master_seed(champ, chall, nonce=7) != _master_seed(champ, chall, nonce=8)


def test_batch_rng_deterministic():
    r1 = _batch_rng(123, 4, "game")
    r2 = _batch_rng(123, 4, "game")
    seq1 = [r1.randint(0, 1000) for _ in range(5)]
    seq2 = [r2.randint(0, 1000) for _ in range(5)]
    assert seq1 == seq2


@pytest.mark.asyncio
async def test_eval_success_field():
    env = RecorderEnv(success=True)
    out = await _eval(env, "m", "u", 1, 2, {}, timeout=1)
    assert out is True


@pytest.mark.asyncio
async def test_eval_falls_back_to_score_field():
    class ScoreEnv:
        async def evaluate(self, **kwargs):
            return {"score": 0.1}

    out = await _eval(ScoreEnv(), "m", "u", 1, 2, {}, timeout=1)
    assert out is True


@pytest.mark.asyncio
async def test_eval_timeout_returns_false():
    out = await _eval(SlowEnv(), "m", "u", 1, 2, {}, timeout=0.01)
    assert out is False


@pytest.mark.asyncio
async def test_eval_error_returns_none():
    out = await _eval(ErrorEnv(), "m", "u", 1, 2, {}, timeout=1)
    assert out is None


@pytest.mark.asyncio
async def test_run_duel_deterministic_task_stream():
    champ = Slot("champ", "rev-a", "http://champ")
    chall = Slot("chall", "rev-b", "http://chall")

    env1 = RecorderEnv(success=True)
    env2 = RecorderEnv(success=True)

    verdict1 = await run_duel(_envs(env1), champ, chall, max_tasks=6, tasks_per_batch=2, k=99.0, nonce=123)
    verdict2 = await run_duel(_envs(env2), champ, chall, max_tasks=6, tasks_per_batch=2, k=99.0, nonce=123)

    assert verdict1 is Verdict.CHAMPION_HOLDS
    assert verdict2 is Verdict.CHAMPION_HOLDS
    calls1 = sorted((m, s, t) for m, s, t, _ in env1.calls)
    calls2 = sorted((m, s, t) for m, s, t, _ in env2.calls)
    assert calls1 == calls2


@pytest.mark.asyncio
async def test_run_duel_counts_only_decisive_outcomes():
    champ = Slot("champ", "rev-a", "http://champ")
    chall = Slot("chall", "rev-b", "http://chall")
    env = ScriptEnv([
        True,
        True,
        True,
        False,
        False,
        True,
    ])

    observed = []

    def _capture(wins, losses, tasks, max_tasks, k):
        observed.append((wins.copy(), losses.copy(), tasks.copy()))
        return Verdict.UNDECIDED, 0.0

    with patch("affine.duel.check_duel", side_effect=_capture), patch(
        "affine.duel.health_check", AsyncMock(return_value=True)
    ):
        verdict = await run_duel(_envs(env), champ, chall, max_tasks=3, tasks_per_batch=1, k=99.0)

    assert verdict is Verdict.CHAMPION_HOLDS
    wins, losses, tasks = observed[-1]
    assert wins["env"] == 1
    assert losses["env"] == 1
    assert tasks["env"] == 3


@pytest.mark.asyncio
async def test_run_duel_early_challenger_dethrone():
    champ = Slot("champ", "rev-a", "http://champ")
    chall = Slot("chall", "rev-b", "http://chall")
    env = ModelMapEnv({champ.model: False, chall.model: True})

    with patch("affine.duel.health_check", AsyncMock(return_value=True)):
        verdict = await run_duel(_envs(env), champ, chall, max_tasks=20, tasks_per_batch=1, k=0.5)

    assert verdict is Verdict.CHALLENGER_WINS
    assert len(env.calls) == 2


@pytest.mark.asyncio
async def test_run_duel_early_hopeless_champion_hold():
    champ = Slot("champ", "rev-a", "http://champ")
    chall = Slot("chall", "rev-b", "http://chall")
    env = ModelMapEnv({champ.model: True, chall.model: False})

    with patch("affine.duel.health_check", AsyncMock(return_value=True)):
        verdict = await run_duel(_envs(env), champ, chall, max_tasks=20, tasks_per_batch=1, k=2.0)

    assert verdict is Verdict.CHAMPION_HOLDS
    assert len(env.calls) < 40


@pytest.mark.asyncio
async def test_run_duel_budget_exhaustion_defaults_to_champion_hold():
    champ = Slot("champ", "rev-a", "http://champ")
    chall = Slot("chall", "rev-b", "http://chall")
    env = RecorderEnv(success=True)

    verdict = await run_duel(_envs(env), champ, chall, max_tasks=5, tasks_per_batch=2, k=5.0)

    assert verdict is Verdict.CHAMPION_HOLDS
    assert len(env.calls) == 10


@pytest.mark.asyncio
async def test_run_duel_raises_on_sustained_infra_failure():
    champ = Slot("champ", "rev-a", "http://champ")
    chall = Slot("chall", "rev-b", "http://chall")

    async def _fake_eval(env, model, url, seed, task_id, params, timeout=600):
        return None if model == champ.model else True

    with patch("affine.duel._eval", side_effect=_fake_eval), patch(
        "affine.duel.health_check", AsyncMock(return_value=True)
    ):
        with pytest.raises(RuntimeError, match="sustained infra failure"):
            await run_duel(_envs(object()), champ, chall, max_tasks=20, tasks_per_batch=1, k=99.0)


@pytest.mark.asyncio
async def test_run_duel_raises_when_champion_slot_down_mid_duel():
    champ = Slot("champ", "rev-a", "http://champ")
    chall = Slot("chall", "rev-b", "http://chall")

    async def _fake_eval(env, model, url, seed, task_id, params, timeout=600):
        return False if model == champ.model else True

    with patch("affine.duel._eval", side_effect=_fake_eval), patch(
        "affine.duel.health_check", AsyncMock(return_value=False)
    ):
        with pytest.raises(RuntimeError, match="champion slot down mid-duel"):
            await run_duel(_envs(object()), champ, chall, max_tasks=5, tasks_per_batch=1, k=99.0)


@pytest.mark.e2e_fault
@pytest.mark.asyncio
async def test_run_duel_raises_when_challenger_slot_down_mid_duel():
    champ = Slot("champ", "rev-a", "http://champ")
    chall = Slot("chall", "rev-b", "http://chall")

    async def _fake_eval(env, model, url, seed, task_id, params, timeout=600):
        return True if model == champ.model else False

    with patch("affine.duel._eval", side_effect=_fake_eval), patch(
        "affine.duel.health_check", AsyncMock(return_value=False)
    ):
        with pytest.raises(RuntimeError, match="challenger slot down mid-duel"):
            await run_duel(_envs(object()), champ, chall, max_tasks=5, tasks_per_batch=1, k=99.0)
