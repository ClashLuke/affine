from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import time

import httpx

log = logging.getLogger(__name__)

CLEANUP_TIMEOUT_S = 30.0
GENERATION_KEYS = {
    "frequency_penalty", "logit_bias", "max_tokens", "min_p", "presence_penalty",
    "repetition_penalty", "stop", "temperature", "top_k", "top_p",
}


async def _drain(task: asyncio.Task, slot) -> None:
    """Cancel + bounded await. An evaluator that suppresses CancelledError (e.g.
    a sync block in a Docker exec wrapper) would otherwise deadlock the loop on
    SIGTERM/timeout. Bounding it leaks a single hung task — preferable to
    blocking the entire validator until k8s SIGKILL.

    CancelledError must propagate: a BaseException catch here would silently
    convert outer-task cancellation (SIGTERM during drain) into normal return,
    and run_one would then report the timed-out sample as a miner loss instead
    of unwinding shutdown."""
    if task.done():
        return
    task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=CLEANUP_TIMEOUT_S)
    except asyncio.TimeoutError:
        log.warning(f"sample task did not release within {CLEANUP_TIMEOUT_S}s ({slot.model}); leaking")
    except asyncio.CancelledError:
        # Distinguish "task's own cancellation finished" (expected — we just called
        # task.cancel()) from "outer scope cancelled us" (must propagate). The
        # task.done() check has a same-tick race where both are true; current_task
        # ().cancelling() is the authoritative signal: nonzero iff the current
        # task itself was cancelled by an outer scope (3.11+). Without this, a
        # SIGTERM during drain on a timed-out sample silently returns False and
        # delays shutdown by the dwell duration.
        if (cur := asyncio.current_task()) is not None and cur.cancelling() > 0:
            raise
    except Exception:
        pass


def _tokens(r) -> int:
    """Best-effort completion-token count from the env response. envs that don't
    surface usage return 0, which the dwell logger treats as 'unknown' rather
    than zero. Generation tok/s is the primary throughput metric, so a
    decentralized env API that doesn't ship usage will miss the stat — not the
    end of the world."""
    try:
        u = (r.get("extra") or {}).get("usage") or r.get("usage")
        if isinstance(u, dict):
            for k in ("completion_tokens", "output_tokens", "generated_tokens"):
                v = u.get(k)
                if isinstance(v, (int, float)) and v > 0: return int(v)
    except Exception: pass
    return 0


async def _maybe_await(x):
    return await x if inspect.isawaitable(x) else x


def _messages(obs) -> list[dict]:
    if isinstance(obs, str):
        return [{"role": "user", "content": obs}]
    if isinstance(obs, list) and all(isinstance(m, dict) for m in obs):
        return obs
    if isinstance(obs, dict):
        if isinstance(obs.get("messages"), list):
            return obs["messages"]
        if isinstance(obs.get("prompt"), str):
            return [{"role": "user", "content": obs["prompt"]}]
    raise TypeError(f"unsupported env observation: {type(obs).__name__}")


def _usage_add(total: dict, usage) -> dict:
    if not isinstance(usage, dict):
        return total
    for k, v in usage.items():
        if isinstance(v, (int, float)) and math.isfinite(v):
            total[k] = total.get(k, 0) + v
    return total


def _content(r: dict) -> str:
    choices = r.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenAI response has no choices")
    ch = choices[0]
    if not isinstance(ch, dict):
        raise ValueError("OpenAI choice is not an object")
    msg = ch.get("message")
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        return msg["content"]
    if isinstance(ch.get("text"), str):
        return ch["text"]
    raise ValueError("OpenAI choice has no text content")


async def _chat(slot, obs, params: dict, seed: int) -> dict:
    payload = {
        "model": slot.model,
        "messages": _messages(obs),
        "temperature": 0.0,
        "seed": seed,
    }
    payload.update({k: v for k, v in params.items() if k in GENERATION_KEYS})
    headers = {}
    api_key = params.get("api_key") or os.getenv("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(f"{slot.base_url.rstrip('/')}/chat/completions",
                              json=payload, headers=headers)
    if r.status_code != 200:
        raise RuntimeError(f"chat/completions returned {r.status_code}: {r.text[:400]}")
    data = r.json()
    if not isinstance(data, dict):
        raise TypeError(f"chat/completions returned {type(data).__name__}")
    return data


async def _legacy_eval(env_wrapper, params: dict, slot, seed: int, task_id: int):
    return await env_wrapper.evaluate(
        model=slot.model, base_url=slot.base_url,
        seed=seed, task_id=task_id, **params,
    )


async def _gym_eval(env_wrapper, params: dict, slot, seed: int, task_id: int):
    env = env_wrapper.make() if hasattr(env_wrapper, "make") else env_wrapper
    try:
        obs, reset_info = await _maybe_await(env.reset(seed=task_id, options=params))
        reset_max = reset_info.get("max_turns") if isinstance(reset_info, dict) else None
        param_max = params.get("gym_max_steps")
        if reset_max is None:
            max_steps = int(param_max if param_max is not None else 64)
        elif param_max is None:
            max_steps = int(reset_max)
        else:
            max_steps = min(int(reset_max), int(param_max))
        usage: dict = {}
        last_info = {}
        for step_idx in range(max_steps):
            raw = await _chat(slot, obs, params, seed)
            _usage_add(usage, raw.get("usage"))
            answer = _content(raw)
            next_obs, reward, terminated, truncated, info = await _maybe_await(env.step(answer))
            last_info = info
            if terminated or truncated:
                success = info.get("success") if isinstance(info, dict) else None
                return {
                    "score": float(reward),
                    "success": success if isinstance(success, bool) else float(reward) > 0.0,
                    "usage": usage or None,
                    "extra": {"reset": reset_info, "step": info, "turns": step_idx + 1},
                }
            obs = [*_messages(obs), {"role": "assistant", "content": answer}, *_messages(next_obs)]
        return {
            "score": 0.0,
            "success": False,
            "usage": usage or None,
            "extra": {
                "reset": reset_info,
                "step": last_info,
                "turns": max_steps,
                "error": f"env exceeded gym_max_steps={max_steps}",
            },
        }
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            await _maybe_await(close())


async def run_one(
    env_wrapper,
    params: dict,
    timeout: float,
    slot,
    seed: int,
    task_id: int = 0,
) -> tuple[bool | None, float, int]:
    """Run one inference sample. Returns (outcome, latency_seconds, tokens).

    outcome is True/False for decisive pass/fail, None for validator-side
    infrastructure error (vLLM 5xx, connection refused, env BackendError,
    wrapper-side exception). Per `feedback_we_own_vllm` and notes/plan.md
    "Targon node disconnects mid-duel → do not resume partial evidence",
    callers must drop None-outcome rows: vLLM is our config; failures there
    are not the miner's fault.

    False is a legitimate miner loss: a wrong answer, our outer-deadline
    timeout, or wrapper-reported `error_type=timeout` (the model didn't
    respond inside the env's per-task budget — plan.md:46 "Challenger
    times out on a task → that task is a loss").

    tokens is best-effort completion-token count parsed from env usage stats; 0
    if the env doesn't surface it. Throughput logging in the dwell uses this to
    compute generation tok/s.

    The outer deadline is enforced by an explicit task + wait so we can tell our
    timeout (miner-loss = False) from inner asyncio.TimeoutError surfaced by
    the wrapper (its own client timeout = infra = None). asyncio.wait_for
    collapses both into the same exception class, mis-attributing infra timeouts
    as miner losses.
    """
    async def _call():
        if hasattr(env_wrapper, "evaluate"):
            return await _legacy_eval(env_wrapper, params, slot, seed, task_id)
        return await _gym_eval(env_wrapper, params, slot, seed, task_id)
    t0 = time.monotonic()
    task = asyncio.create_task(_call())
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
    except BaseException:
        await _drain(task, slot)
        raise
    dt = time.monotonic() - t0
    if task not in done:
        await _drain(task, slot)
        log.warning(f"sample timed out after {timeout}s: {slot.model}")
        return False, time.monotonic() - t0, 0
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        if (cur := asyncio.current_task()) is not None and cur.cancelling() > 0:
            raise
        log.warning(f"infra failure ({slot.model}) in {dt:.2f}s — sample task cancelled itself")
        return None, dt, 0
    if exc is not None:
        log.warning(f"sample error ({slot.model}) after {dt:.2f}s: {type(exc).__name__}: {exc}")
        return None, dt, 0
    try:
        r = task.result()
    except asyncio.CancelledError:
        if (cur := asyncio.current_task()) is not None and cur.cancelling() > 0:
            raise
        log.warning(f"infra failure ({slot.model}) in {dt:.2f}s — sample task cancelled itself")
        return None, dt, 0
    if not isinstance(r, dict):
        log.warning(f"infra failure ({slot.model}) in {dt:.2f}s — non-dict response: {str(r)[:200]}")
        return None, dt, 0
    if r.get("status") == "failed":
        log.warning(f"infra failure ({slot.model}) in {dt:.2f}s: wrapper status=failed: {str(r.get('error', ''))[:200]}")
        return None, dt, 0
    if (err := r.get("error_type")) is not None:
        if err == "timeout":
            log.warning(f"sample timeout ({slot.model}) in {dt:.2f}s: {str(r.get('error', ''))[:200]}")
            return False, dt, 0
        log.warning(f"infra failure ({slot.model}) in {dt:.2f}s, error_type={err}: {str(r.get('error', ''))[:200]}")
        return None, dt, 0
    tok = _tokens(r)
    if isinstance(r.get("success"), bool):
        return r["success"], dt, tok
    if isinstance(r.get("score"), (int, float)) and math.isfinite(r["score"]):
        return r["score"] > 0, dt, tok
    log.warning(f"infra failure ({slot.model}) in {dt:.2f}s — response missing/invalid success/score: {str(r)[:200]}")
    return None, dt, 0
