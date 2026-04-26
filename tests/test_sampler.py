from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from affine.sampler import run_one


def _slot(model="m", url="http://s/v1"):
    return SimpleNamespace(model=model, base_url=url)


@pytest.mark.asyncio
async def test_run_one_success_bool():
    env = SimpleNamespace(evaluate=AsyncMock(return_value={"success": True}))
    passed, dt = await run_one(env, {}, 10, _slot(), seed=1)
    assert passed is True
    assert dt >= 0


@pytest.mark.asyncio
async def test_run_one_falsy_score_is_loss():
    """A success=False / score<=0 with no infra signal is a miner loss regardless
    of latency. Latency-based gating let an adversary instafail to dodge evidence."""
    env = SimpleNamespace(evaluate=AsyncMock(return_value={"score": 0}))
    passed, _ = await run_one(env, {}, 10, _slot(), seed=1)
    assert passed is False


@pytest.mark.asyncio
async def test_run_one_fast_success_false_is_loss_not_infra():
    """Regression: a fast `{"success": False}` used to be reclassified as infra
    (None) on the assumption it was a 5xx bouncing. That gating let an adversary
    silently dodge evidence by instafailing. With clear infra signals (status,
    error_type), a bare success=False is a verdict — count it."""
    env = SimpleNamespace(evaluate=AsyncMock(return_value={"success": False}))
    passed, _ = await run_one(env, {}, 10, _slot(), seed=1)
    assert passed is False


@pytest.mark.asyncio
async def test_run_one_positive_score():
    env = SimpleNamespace(evaluate=AsyncMock(return_value={"score": 0.7}))
    passed, _ = await run_one(env, {}, 10, _slot(), seed=1)
    assert passed is True


@pytest.mark.asyncio
async def test_run_one_timeout_is_false():
    async def slow(**kwargs):
        import asyncio
        await asyncio.sleep(1)
    env = SimpleNamespace(evaluate=slow)
    passed, _ = await run_one(env, {}, 0.01, _slot(), seed=1)
    assert passed is False


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [
    "llm_failure",      # affine env
    "timeout",          # distill env: student LLM didn't respond in time
    "connect_error",    # distill env: couldn't reach student
    "empty_response",   # distill env: 200 OK with zero choices
    "no_logprobs",      # distill env: structured response but missing logprobs
    "unexpected",       # distill env: any other HTTPError/exception
])
async def test_run_one_any_error_type_is_infra(error_type):
    """success=False paired with any error_type is infra, not a miner loss.

    Envs use inconsistent strings ('llm_failure' in affine, 'timeout'/'connect_error'/...
    in distill). The sampler must not couple itself to one env's vocabulary — real
    miner losses have no error_type set, so the presence of any error_type is the signal.
    """
    env = SimpleNamespace(evaluate=AsyncMock(return_value={
        "success": False, "error_type": error_type, "error": f"simulated {error_type}",
    }))
    passed, _ = await run_one(env, {}, 10, _slot(), seed=1)
    assert passed is None


@pytest.mark.asyncio
async def test_run_one_inner_timeout_is_infra_not_loss():
    """Regression: env_wrapper.evaluate's own client timeout (raised as
    asyncio.TimeoutError from inside) was conflated with the outer wait_for
    deadline. asyncio.wait_for catches both → both look like miner-loss. With
    explicit task + asyncio.wait, only the OUTER deadline maps to False; an
    inner asyncio.TimeoutError comes back as a task exception → infra (None)."""
    import asyncio
    async def _inner_timeout(**kwargs):
        raise asyncio.TimeoutError("env's own client deadline")
    env = SimpleNamespace(evaluate=_inner_timeout)
    passed, _ = await run_one(env, {}, 10, _slot(), seed=1)
    assert passed is None


@pytest.mark.asyncio
async def test_run_one_exception_is_none():
    env = SimpleNamespace(evaluate=AsyncMock(side_effect=RuntimeError("boom")))
    passed, _ = await run_one(env, {}, 10, _slot(), seed=1)
    assert passed is None


@pytest.mark.asyncio
async def test_run_one_passes_seed_and_params():
    env = SimpleNamespace(evaluate=AsyncMock(return_value={"success": True}))
    await run_one(env, {"temperature": 0.2}, 10, _slot(), seed=42, task_id=7)
    kwargs = env.evaluate.call_args.kwargs
    assert kwargs["seed"] == 42
    assert kwargs["task_id"] == 7
    assert kwargs["temperature"] == 0.2
    assert kwargs["model"] == "m"
    assert kwargs["base_url"] == "http://s/v1"


@pytest.mark.asyncio
async def test_run_one_success_with_error_type_is_infra():
    """Regression: success=True paired with error_type was being scored as a pass.
    The error_type field is authoritative — its presence means infra, regardless
    of any other field."""
    env = SimpleNamespace(evaluate=AsyncMock(return_value={"success": True, "error_type": "llm_failure"}))
    passed, _ = await run_one(env, {}, 10, _slot(), seed=1)
    assert passed is None


@pytest.mark.asyncio
async def test_run_one_score_none_is_infra_not_crash():
    """Regression: `r.get('score', 0) > 0` raised on score=None."""
    env = SimpleNamespace(evaluate=AsyncMock(return_value={"score": None}))
    passed, _ = await run_one(env, {}, 10, _slot(), seed=1)
    assert passed is None


@pytest.mark.asyncio
async def test_run_one_string_success_is_infra_not_truthy():
    """Regression: bool('false') == True, so success='false' was being passed."""
    env = SimpleNamespace(evaluate=AsyncMock(return_value={"success": "false"}))
    passed, _ = await run_one(env, {}, 10, _slot(), seed=1)
    assert passed is None


@pytest.mark.asyncio
async def test_run_one_sync_evaluator_returning_dict_is_infra_not_crash():
    """Regression: env_wrapper.evaluate that returns a dict directly (sync, not
    a coroutine) used to make asyncio.create_task raise TypeError outside the
    timeout try-block, crashing run_one. Now wrapped so it surfaces as task
    exception → infra (None)."""
    env = SimpleNamespace(evaluate=lambda **kw: {"success": True})
    passed, _ = await run_one(env, {}, 10, _slot(), seed=1)
    assert passed is None


@pytest.mark.asyncio
async def test_run_one_drain_bounded_when_evaluator_swallows_cancel(monkeypatch):
    """An evaluator that suppresses CancelledError (e.g. blocking C extension
    in a Docker exec wrapper) used to make `await task` block indefinitely
    after timeout/cancel. _drain bounds it; the leak is preferable to deadlock."""
    import asyncio
    import time
    from affine import sampler
    monkeypatch.setattr(sampler, "CLEANUP_TIMEOUT_S", 0.1)
    async def uncooperative(**kwargs):
        try: await asyncio.sleep(60)
        except asyncio.CancelledError:
            try: await asyncio.sleep(60)  # swallow first cancel
            except asyncio.CancelledError: pass
    env = SimpleNamespace(evaluate=uncooperative)
    t0 = time.monotonic()
    passed, _ = await run_one(env, {}, 0.05, _slot(), seed=1)
    elapsed = time.monotonic() - t0
    assert passed is False  # outer timeout = miner loss
    assert elapsed < 1.0    # bounded — would be 60s+ without _drain


@pytest.mark.asyncio
async def test_run_one_outer_cancel_during_drain_propagates(monkeypatch):
    """Regression: _drain previously caught BaseException, swallowing outer
    CancelledError that arrived during the bounded drain wait. SIGTERM during a
    timed-out sample would then return (False, dt) instead of unwinding shutdown
    — the loop would press on, attempt the next dwell iteration, and hold up
    teardown for the dwell duration. CancelledError from outer scope must
    propagate; only the task's own cancellation (post task.cancel()) is suppressed."""
    import asyncio
    from affine import sampler
    monkeypatch.setattr(sampler, "CLEANUP_TIMEOUT_S", 5.0)
    drain_entered = asyncio.Event()
    async def uncooperative(**kwargs):
        try: await asyncio.sleep(60)
        except asyncio.CancelledError:
            drain_entered.set()
            await asyncio.sleep(60)  # ignore the cancel — forces _drain into wait_for
    env = SimpleNamespace(evaluate=uncooperative)
    task = asyncio.create_task(run_one(env, {}, 0.01, _slot(), seed=1))
    await drain_entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
