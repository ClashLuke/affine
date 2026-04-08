import asyncio
import pytest
from unittest.mock import AsyncMock

from affine.wilson import Verdict
from affine.duel import (
    _check_env, _check_majority, _master_seed, _batch_rng, run_duel,
)
from affine.vllm import Slot
from affine.config import EnvSpec


# --- pure scoring functions ---

def test_check_env_challenger_wins():
    # 80/100 decisive wins, Wilson lower bound well above 0.5
    assert _check_env(80, 20, 100, 200, 1.96) is Verdict.CHALLENGER_WINS


def test_check_env_champion_holds_budget():
    assert _check_env(5, 10, 200, 200, 1.96) is Verdict.CHAMPION_HOLDS


def test_check_env_hopeless():
    # 1 win, 50 losses, 9 remaining — can't catch up even winning all remaining
    assert _check_env(1, 50, 51, 60, 1.96) is Verdict.CHAMPION_HOLDS


def test_check_env_undecided():
    assert _check_env(5, 5, 20, 200, 1.96) is Verdict.UNDECIDED


def test_check_majority_challenger_wins():
    verdicts = {"a": Verdict.CHALLENGER_WINS, "b": Verdict.CHALLENGER_WINS, "c": Verdict.CHAMPION_HOLDS}
    assert _check_majority(verdicts) is Verdict.CHALLENGER_WINS


def test_check_majority_champion_holds():
    verdicts = {"a": Verdict.CHAMPION_HOLDS, "b": Verdict.CHAMPION_HOLDS, "c": Verdict.UNDECIDED}
    assert _check_majority(verdicts) is Verdict.CHAMPION_HOLDS


def test_check_majority_undecided():
    verdicts = {"a": Verdict.CHALLENGER_WINS, "b": Verdict.CHAMPION_HOLDS, "c": Verdict.UNDECIDED, "d": Verdict.UNDECIDED}
    assert _check_majority(verdicts) is Verdict.UNDECIDED


def test_check_majority_single_env():
    assert _check_majority({"a": Verdict.CHALLENGER_WINS}) is Verdict.CHALLENGER_WINS
    assert _check_majority({"a": Verdict.CHAMPION_HOLDS}) is Verdict.CHAMPION_HOLDS


# --- seed determinism ---

def test_master_seed_deterministic():
    c = Slot(model="a", revision="1", base_url="")
    d = Slot(model="b", revision="2", base_url="")
    assert _master_seed(c, d) == _master_seed(c, d)


def test_master_seed_changes_with_nonce():
    c = Slot(model="a", revision="1", base_url="")
    d = Slot(model="b", revision="2", base_url="")
    assert _master_seed(c, d, nonce=0) != _master_seed(c, d, nonce=1)


def test_master_seed_changes_with_model():
    a = Slot(model="a", revision="1", base_url="")
    b = Slot(model="b", revision="1", base_url="")
    x = Slot(model="x", revision="1", base_url="")
    assert _master_seed(a, x) != _master_seed(b, x)


def test_batch_rng_deterministic():
    r1 = _batch_rng(42, 0, "env")
    r2 = _batch_rng(42, 0, "env")
    assert r1.randint(0, 1000) == r2.randint(0, 1000)


def test_batch_rng_differs_across_envs():
    r1 = _batch_rng(42, 0, "env_a")
    r2 = _batch_rng(42, 0, "env_b")
    assert r1.randint(0, 2**32) != r2.randint(0, 2**32)


# --- duel integration: dead endpoint detection ---

def _make_env(results_champ, results_chall):
    """Mock environment where champion returns results_champ[i] and challenger results_chall[i]."""
    call_idx = {"n": 0}

    async def evaluate(*, model, base_url, seed, task_id, **kw):
        i = call_idx["n"] // 2  # two calls per task (champ + chall)
        is_champ = call_idx["n"] % 2 == 0
        call_idx["n"] += 1
        seq = results_champ if is_champ else results_chall
        val = seq[min(i, len(seq) - 1)]
        if val is None:
            raise ConnectionError("endpoint down")
        return {"success": val}

    wrapper = AsyncMock()
    wrapper.evaluate = evaluate
    return wrapper


@pytest.mark.asyncio
async def test_dead_champion_challenger_succeeds(monkeypatch):
    """Dead champion + successful challenger → challenger wins via health check."""
    async def mock_health_ping(url, timeout=5):
        return "challenger" in url

    import affine.duel
    monkeypatch.setattr(affine.duel, "health_ping", mock_health_ping)

    champ = Slot(model="champ", revision="1", base_url="http://champion:8000/v1")
    chall = Slot(model="chall", revision="1", base_url="http://challenger:8000/v1")
    spec = EnvSpec(name="test", image="test:latest", params={"timeout": 5})

    wrapper = AsyncMock()

    async def evaluate(*, model, base_url, seed, task_id, **kw):
        if "champion" in base_url:
            return {"success": False}
        return {"success": True}

    wrapper.evaluate = evaluate

    envs = {"test": (wrapper, spec)}
    verdict = await run_duel(envs, champ, chall, max_tasks=50, tasks_per_batch=4, z=1.96)
    assert verdict is Verdict.CHALLENGER_WINS


@pytest.mark.asyncio
async def test_dead_champion_challenger_always_wrong(monkeypatch):
    """Dead champion + alive but always-wrong challenger → challenger still wins.

    This is the exact scenario from finding 1: champion is down (timeouts become False),
    challenger is alive but answers every task incorrectly (also False). All tasks are ties.
    Without the health check, budget exhaustion would default to CHAMPION_HOLDS.
    The health ping detects champion is dead and awards the win to the functioning challenger.
    """
    async def mock_health_ping(url, timeout=5):
        return "challenger" in url  # champion is down, challenger is up

    import affine.duel
    monkeypatch.setattr(affine.duel, "health_ping", mock_health_ping)

    champ = Slot(model="champ", revision="1", base_url="http://champion:8000/v1")
    chall = Slot(model="chall", revision="1", base_url="http://challenger:8000/v1")
    spec = EnvSpec(name="test", image="test:latest", params={"timeout": 5})

    wrapper = AsyncMock()

    async def evaluate(*, model, base_url, seed, task_id, **kw):
        # Both return False — champion because it's dead (timeout→False),
        # challenger because it's alive but wrong
        return {"success": False}

    wrapper.evaluate = evaluate

    envs = {"test": (wrapper, spec)}
    verdict = await run_duel(envs, champ, chall, max_tasks=50, tasks_per_batch=4, z=1.96)
    assert verdict is Verdict.CHALLENGER_WINS


@pytest.mark.asyncio
async def test_healthy_loser_not_dethroned(monkeypatch):
    """If champion is alive but losing, it should lose through normal Wilson scoring, not health check."""
    async def mock_health_ping(url, timeout=5):
        return True  # both endpoints healthy

    import affine.duel
    monkeypatch.setattr(affine.duel, "health_ping", mock_health_ping)

    champ = Slot(model="champ", revision="1", base_url="http://champion:8000/v1")
    chall = Slot(model="chall", revision="1", base_url="http://challenger:8000/v1")
    spec = EnvSpec(name="test", image="test:latest", params={"timeout": 5})

    wrapper = AsyncMock()

    async def evaluate(*, model, base_url, seed, task_id, **kw):
        if "champion" in base_url:
            return {"success": False}
        return {"success": True}

    wrapper.evaluate = evaluate

    envs = {"test": (wrapper, spec)}
    verdict = await run_duel(envs, champ, chall, max_tasks=50, tasks_per_batch=4, z=1.96)
    # Challenger should win through Wilson scoring
    assert verdict is Verdict.CHALLENGER_WINS


@pytest.mark.asyncio
async def test_infra_errors_trigger_raise(monkeypatch):
    """Sustained None results from _eval should raise, triggering retry."""
    async def mock_health_ping(url, timeout=5):
        return True  # health ping passes (so mid-duel check doesn't short-circuit)

    import affine.duel
    monkeypatch.setattr(affine.duel, "health_ping", mock_health_ping)

    champ = Slot(model="champ", revision="1", base_url="http://champion:8000/v1")
    chall = Slot(model="chall", revision="1", base_url="http://challenger:8000/v1")
    spec = EnvSpec(name="test", image="test:latest", params={"timeout": 5})

    wrapper = AsyncMock()

    async def evaluate(*, model, base_url, seed, task_id, **kw):
        raise ConnectionError("node disconnected")

    wrapper.evaluate = evaluate

    envs = {"test": (wrapper, spec)}
    with pytest.raises(RuntimeError, match="sustained infra failure"):
        await run_duel(envs, champ, chall, max_tasks=50, tasks_per_batch=4, z=1.96)
