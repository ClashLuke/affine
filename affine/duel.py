from __future__ import annotations
import asyncio
import hashlib
import logging
import random
from math import sqrt
from typing import Any

from .config import EnvSpec
from .scoring import Verdict, bt_mle, aggregate, check_duel
from .vllm import Slot, health_check

log = logging.getLogger(__name__)

INFRA_STREAK_LIMIT = 8


def _master_seed(champion: Slot, challenger: Slot, nonce: int = 0) -> int:
    h = hashlib.sha256(
        f"{champion.model}\0{champion.revision}\0"
        f"{challenger.model}\0{challenger.revision}\0{nonce}".encode()
    )
    return int.from_bytes(h.digest()[:8], "big")


def _batch_rng(master: int, batch: int, env_name: str) -> random.Random:
    h = hashlib.sha256(f"{master}\0{batch}\0{env_name}".encode())
    return random.Random(int.from_bytes(h.digest()[:8], "big"))


async def _eval(env, model: str, url: str, seed: int, task_id: int, params: dict, timeout: float = 600):
    try:
        r = await asyncio.wait_for(
            env.evaluate(model=model, base_url=url, seed=seed, task_id=task_id, **params),
            timeout=timeout,
        )
        return bool(r.get("success", r.get("score", 0) > 0))
    except asyncio.TimeoutError:
        log.warning(f"eval timed out after {timeout}s: {model}")
        return False
    except Exception as e:
        log.warning(f"eval error ({model}): {e}")
        return None


async def run_duel(
    envs: dict[str, tuple[Any, EnvSpec]],
    champion: Slot,
    challenger: Slot,
    *,
    max_tasks: int = 200,
    tasks_per_batch: int = 4,
    k: float = 2.0,
    nonce: int = 0,
    progress_interval: int = 0,
) -> Verdict:
    env_data = {}
    for name, (wrapper, spec) in envs.items():
        params = dict(spec.params)
        timeout = params.pop("timeout", 600)
        env_data[name] = (wrapper, params, timeout)

    wins = {name: 0 for name in env_data}
    losses = {name: 0 for name in env_data}
    tasks = {name: 0 for name in env_data}
    master = _master_seed(champion, challenger, nonce)
    champ_err_streak = 0
    chall_err_streak = 0

    next_progress = progress_interval if progress_interval > 0 else None
    for batch in range(max_tasks):
        coros, keys = [], []

        for name, (wrapper, params, task_timeout) in env_data.items():
            if tasks[name] >= max_tasks:
                continue

            rng = _batch_rng(master, batch, name)
            room = max_tasks - tasks[name]

            for _ in range(min(tasks_per_batch, room)):
                tid = rng.randint(0, max_tasks - 1) if max_tasks > 0 else 0
                seed = rng.randint(0, 2**32 - 1)
                coros.append(asyncio.gather(
                    _eval(wrapper, champion.model, champion.base_url, seed, tid, params, task_timeout),
                    _eval(wrapper, challenger.model, challenger.base_url, seed, tid, params, task_timeout),
                ))
                keys.append(name)

        if not coros:
            break

        results = await asyncio.gather(*coros, return_exceptions=True)

        batch_champ_any_ok = False
        batch_chall_any_ok = False

        for env_name, result in zip(keys, results):
            tasks[env_name] += 1
            if isinstance(result, Exception):
                log.warning(f"pair exception in {env_name}: {result}")
                champ_err_streak += 1
                chall_err_streak += 1
                continue
            champ_ok, chall_ok = result

            if champ_ok is None:
                champ_err_streak += 1
            else:
                champ_err_streak = 0
                if champ_ok:
                    batch_champ_any_ok = True
            if chall_ok is None:
                chall_err_streak += 1
            else:
                chall_err_streak = 0
                if chall_ok:
                    batch_chall_any_ok = True

            if champ_ok is not None and chall_ok is not None and champ_ok != chall_ok:
                (wins if chall_ok else losses)[env_name] += 1

        if champ_err_streak >= INFRA_STREAK_LIMIT or chall_err_streak >= INFRA_STREAK_LIMIT:
            raise RuntimeError(
                f"sustained infra failure (champion={champ_err_streak}, "
                f"challenger={chall_err_streak} consecutive errors)"
            )

        if not batch_champ_any_ok:
            if not await health_check(champion.base_url, timeout=30):
                raise RuntimeError("champion slot down mid-duel")
        if not batch_chall_any_ok:
            if not await health_check(challenger.base_url, timeout=30):
                raise RuntimeError("challenger slot down mid-duel")

        verdict, z = check_duel(wins, losses, tasks, max_tasks, k)
        if next_progress is not None:
            total_tasks = sum(tasks.values())
            while total_tasks >= next_progress:
                _log_progress(wins, losses, tasks, total_tasks, k, z)
                next_progress += progress_interval
        if verdict is not Verdict.UNDECIDED:
            _log_summary(wins, losses, tasks, verdict, champion, challenger, k, z)
            return verdict

    verdict, z = check_duel(wins, losses, tasks, max_tasks, k)
    if verdict is Verdict.UNDECIDED:
        verdict = Verdict.CHAMPION_HOLDS
    _log_summary(wins, losses, tasks, verdict, champion, challenger, k, z)
    return verdict


def _log_summary(wins, losses, tasks, verdict, champion, challenger, k, z):
    total_tasks = sum(tasks.values())
    decisive = sum(wins[n] + losses[n] for n in wins)

    if decisive:
        log.info(f"duel: {verdict.name} | {champion.model} vs {challenger.model} | "
                 f"z={z:.2f} k={k:.2f} | {total_tasks} tasks, {decisive} decisive")
    else:
        log.info(f"duel: {verdict.name} | {champion.model} vs {challenger.model} | "
                 f"no decisive outcomes | {total_tasks} tasks")

    for name in wins:
        w, l = wins[name], losses[name]
        if w + l > 0:
            d, v = bt_mle(w, l)
            log.info(f"  {name}: W={w} L={l} n={tasks[name]} delta={d:.3f} se={sqrt(v):.3f}")
        else:
            log.info(f"  {name}: W=0 L=0 n={tasks[name]}")


def _log_progress(wins, losses, tasks, total_tasks, k, z):
    decisive = sum(wins[n] + losses[n] for n in wins)
    if decisive:
        total_wins = sum(wins.values())
        pairs = [bt_mle(wins[n], losses[n]) for n in wins if wins[n] + losses[n] > 0]
        delta, _ = aggregate([d for d, _ in pairs], [v for _, v in pairs])
        log.info(f"progress: tasks={total_tasks} decisive={decisive} winrate={total_wins / decisive:.3f} "
                 f"delta={delta:.3f} z={z:.2f} k={k:.2f}")
    else:
        log.info(f"progress: tasks={total_tasks} decisive=0 z=0.00 k={k:.2f}")
