from __future__ import annotations

import asyncio
import logging
import math
import time

log = logging.getLogger(__name__)

CLEANUP_TIMEOUT_S = 30.0


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


async def run_one(
    env_wrapper,
    params: dict,
    timeout: float,
    slot,
    seed: int,
    task_id: int = 0,
) -> tuple[bool | None, float]:
    """Run one inference sample. Returns (outcome, latency_seconds).

    outcome is True/False for decisive pass/fail, None for infrastructure error
    (connection refused, wrapper-reported failure, or exception in env wrapper).
    Callers use None to detect slot/infra problems and discard the row; False is
    a legitimate loss (real inference that produced a wrong answer or ran to the
    sample timeout).

    The outer deadline is enforced by an explicit task + wait so we can tell our
    timeout (miner-loss = False) from any inner asyncio.TimeoutError that the
    wrapper might surface (its own client timeout = infra = None). asyncio.wait_for
    collapses both into the same exception class, mis-attributing infra timeouts
    as miner losses.
    """
    async def _call():
        # Wrap evaluate so a sync evaluate, a non-coroutine return, or an
        # immediate TypeError from create_task surfaces as a task exception
        # (→ infra=None) rather than crashing run_one before the timeout
        # try/except is in scope.
        return await env_wrapper.evaluate(
            model=slot.model, base_url=slot.base_url,
            seed=seed, task_id=task_id, **params,
        )
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
        return False, time.monotonic() - t0
    if (exc := task.exception()) is not None:
        log.warning(f"sample error ({slot.model}) after {dt:.2f}s: {type(exc).__name__}: {exc}")
        return None, dt
    r = task.result()
    if not isinstance(r, dict):
        log.warning(f"infra failure ({slot.model}) in {dt:.2f}s — non-dict response: {str(r)[:200]}")
        return None, dt
    if r.get("status") == "failed":
        log.warning(f"infra failure ({slot.model}) in {dt:.2f}s — affinetes wrapper status=failed: {str(r.get('error', ''))[:200]}")
        return None, dt
    if (err := r.get("error_type")) is not None:
        log.warning(f"infra failure ({slot.model}) in {dt:.2f}s, error_type={err}: {str(r.get('error', ''))[:200]}")
        return None, dt
    # success and score are mutually authoritative; accept exactly one shape.
    if isinstance(r.get("success"), bool):
        return r["success"], dt
    if isinstance(r.get("score"), (int, float)) and math.isfinite(r["score"]):
        return r["score"] > 0, dt
    log.warning(f"infra failure ({slot.model}) in {dt:.2f}s — response missing/invalid success/score: {str(r)[:200]}")
    return None, dt
