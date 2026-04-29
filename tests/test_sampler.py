from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from affine.sampler import run_one


def _slot(model="m", url="http://s/v1"):
    return SimpleNamespace(model=model, base_url=url)


def _gym_env(*, step_fn=None, reset_max_turns=None):
    """Minimal gym env. step_fn(obs, action) -> (obs, reward, terminated, truncated, info)."""
    class Env:
        def reset(self, *, seed=None, options=None):
            r = {"challenge_id": str(seed)}
            if reset_max_turns is not None:
                r["max_turns"] = reset_max_turns
            return "prompt", r

        def step(self, action):
            if step_fn is not None:
                return step_fn(action)
            return None, 1.0, True, False, {"success": True}

    return Env()


@pytest.mark.asyncio
async def test_run_one_uses_gym_reset_step(monkeypatch):
    from affine import sampler
    calls = {}

    class Env:
        def reset(self, *, seed=None, options=None):
            calls["reset"] = (seed, options)
            return "prompt", {"challenge_id": str(seed)}

        def step(self, action):
            calls["action"] = action
            return None, 1.0, True, False, {"score": 1.0}

    async def fake_chat(slot, obs, params, seed):
        calls["chat"] = (slot.model, obs, params, seed)
        return {
            "choices": [{"message": {"content": "<ANSWER>ok</ANSWER>"}}],
            "usage": {"completion_tokens": 5},
        }

    monkeypatch.setattr(sampler, "_chat", fake_chat)
    passed, _, tok = await run_one(Env(), {"temperature": 0.2}, 10, _slot(), seed=42, task_id=7)
    assert passed is True
    assert tok == 5
    assert calls["reset"] == (7, {})  # temperature is gen param, not env option
    assert calls["chat"] == ("m", "prompt", {"temperature": 0.2}, 42)
    assert calls["action"] == "<ANSWER>ok</ANSWER>"


@pytest.mark.asyncio
async def test_run_one_gym_multistep_keeps_conversation(monkeypatch):
    from affine import sampler
    calls = []

    class Env:
        def reset(self, *, seed=None, options=None):
            return "prompt", {"challenge_id": str(seed), "max_turns": 2}

        def step(self, action):
            calls.append(("step", action))
            if len([c for c in calls if c[0] == "step"]) == 1:
                return "query result", 0.0, False, False, {}
            return None, 1.0, True, False, {"success": True}

    async def fake_chat(slot, obs, params, seed):
        calls.append(("chat", obs))
        content = "QUERY CHILDREN 0" if len([c for c in calls if c[0] == "chat"]) == 1 else "SUBMIT 0"
        return {"choices": [{"message": {"content": content}}], "usage": {"completion_tokens": 2}}

    monkeypatch.setattr(sampler, "_chat", fake_chat)
    passed, _, tok = await run_one(Env(), {}, 10, _slot(), seed=42, task_id=7)
    assert passed is True
    assert tok == 4
    assert calls[0] == ("chat", "prompt")
    assert calls[2] == ("chat", [
        {"role": "user", "content": "prompt"},
        {"role": "assistant", "content": "QUERY CHILDREN 0"},
        {"role": "user", "content": "query result"},
    ])


@pytest.mark.asyncio
async def test_run_one_gym_max_steps_caps_reset_max_turns(monkeypatch):
    from affine import sampler
    calls = []

    class Env:
        def reset(self, *, seed=None, options=None):
            return "prompt", {"challenge_id": str(seed), "max_turns": 5}

        def step(self, action):
            return "keep going", 0.0, False, False, {}

    async def fake_chat(slot, obs, params, seed):
        calls.append(obs)
        return {"choices": [{"message": {"content": "QUERY DEPTH 1"}}], "usage": {"completion_tokens": 1}}

    monkeypatch.setattr(sampler, "_chat", fake_chat)
    passed, _, tok = await run_one(Env(), {"gym_max_steps": 1}, 10, _slot(), seed=42, task_id=7)
    assert passed is False
    assert tok == 1
    assert calls == ["prompt"]


@pytest.mark.asyncio
async def test_run_one_timeout_is_false(monkeypatch):
    from affine import sampler
    async def slow_chat(slot, obs, params, seed):
        await asyncio.sleep(60)
    monkeypatch.setattr(sampler, "_chat", slow_chat)
    passed, _, _ = await run_one(_gym_env(), {}, 0.01, _slot(), seed=1)
    assert passed is False


@pytest.mark.asyncio
async def test_run_one_exception_is_none(monkeypatch):
    from affine import sampler
    async def bad(slot, obs, params, seed):
        raise RuntimeError("boom")
    monkeypatch.setattr(sampler, "_chat", bad)
    passed, _, _ = await run_one(_gym_env(), {}, 10, _slot(), seed=1)
    assert passed is None


@pytest.mark.asyncio
async def test_run_one_success_bool(monkeypatch):
    from affine import sampler

    async def fake_chat(slot, obs, params, seed):
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(sampler, "_chat", fake_chat)
    env = _gym_env(step_fn=lambda a: (None, 1.0, True, False, {"success": True}))
    passed, _, _ = await run_one(env, {}, 10, _slot(), seed=1)
    assert passed is True

    env2 = _gym_env(step_fn=lambda a: (None, 0.0, True, False, {"success": False}))
    passed2, _, _ = await run_one(env2, {}, 10, _slot(), seed=1)
    assert passed2 is False


@pytest.mark.asyncio
async def test_run_one_score_numeric(monkeypatch):
    from affine import sampler

    async def fake_chat(slot, obs, params, seed):
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(sampler, "_chat", fake_chat)
    env = _gym_env(step_fn=lambda a: (None, 0.7, True, False, {}))
    passed, _, _ = await run_one(env, {}, 10, _slot(), seed=1)
    assert passed is True

    env2 = _gym_env(step_fn=lambda a: (None, 0.0, True, False, {}))
    passed2, _, _ = await run_one(env2, {}, 10, _slot(), seed=1)
    assert passed2 is False


@pytest.mark.asyncio
async def test_run_one_drain_bounded_when_chat_swallows_cancel(monkeypatch):
    import asyncio as _asyncio
    from affine import sampler
    monkeypatch.setattr(sampler, "CLEANUP_TIMEOUT_S", 0.1)

    async def uncooperative(slot, obs, params, seed):
        try:
            await _asyncio.sleep(60)
        except _asyncio.CancelledError:
            try:
                await _asyncio.sleep(60)
            except _asyncio.CancelledError:
                pass

    monkeypatch.setattr(sampler, "_chat", uncooperative)
    t0 = time.monotonic()
    passed, _, _ = await run_one(_gym_env(), {}, 0.01, _slot(), seed=1)
    elapsed = time.monotonic() - t0
    assert passed is False
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_run_one_outer_cancel_during_drain_propagates(monkeypatch):
    import asyncio as _asyncio
    from affine import sampler
    monkeypatch.setattr(sampler, "CLEANUP_TIMEOUT_S", 5.0)
    drain_entered = _asyncio.Event()

    async def uncooperative(slot, obs, params, seed):
        try:
            await _asyncio.sleep(60)
        except _asyncio.CancelledError:
            drain_entered.set()
            await _asyncio.sleep(60)

    monkeypatch.setattr(sampler, "_chat", uncooperative)
    task = _asyncio.create_task(run_one(_gym_env(), {}, 0.01, _slot(), seed=1))
    await drain_entered.wait()
    task.cancel()
    with pytest.raises(_asyncio.CancelledError):
        await task
