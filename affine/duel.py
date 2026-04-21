from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import os
import random
from datetime import datetime, timezone
from math import sqrt
from typing import Any

from .config import EnvSpec
from .scoring import Verdict, bt_mle, aggregate, check_duel
from .vllm import Slot

log = logging.getLogger(__name__)

INFRA_STREAK_LIMIT = 8


def _inflight_cap(tasks_per_batch: int, n_envs: int) -> int:
    override = os.getenv("AFFINE_INFLIGHT_CAP")
    if override:
        return max(int(override), 1)
    return max(tasks_per_batch * n_envs, 1)


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(int(p * (len(s) - 1) + 0.5), len(s) - 1)
    return s[i]


def _log_latency(lat_champ, lat_chall, t_start, inflight_cap):
    n = len(lat_champ)
    if n == 0:
        return
    wall = asyncio.get_event_loop().time() - t_start
    c50, c95 = _pct(lat_champ, 0.50), _pct(lat_champ, 0.95)
    h50, h95 = _pct(lat_chall, 0.50), _pct(lat_chall, 0.95)
    sum_eval_s = sum(lat_champ) + sum(lat_chall)
    throughput = n / wall if wall > 0 else 0
    saturation = sum_eval_s / (wall * 2 * inflight_cap) if wall > 0 and inflight_cap > 0 else 0
    log.info(
        f"latency: n={n} wall={wall:.1f}s rate={throughput:.2f}pair/s "
        f"champ p50={c50:.2f}s p95={c95:.2f}s "
        f"chall p50={h50:.2f}s p95={h95:.2f}s "
        f"saturation={saturation:.2%} (cap={inflight_cap})"
    )


def _master_seed(champion: Slot, challenger: Slot, nonce: int, hotkey: str) -> int:
    h = hashlib.sha256(
        f"{champion.model}\0{champion.revision}\0"
        f"{challenger.model}\0{challenger.revision}\0{nonce}\0{hotkey}".encode()
    )
    return int.from_bytes(h.digest()[:8], "big")


def _batch_rng(master: int, batch: int, env_name: str) -> random.Random:
    h = hashlib.sha256(f"{master}\0{batch}\0{env_name}".encode())
    return random.Random(int.from_bytes(h.digest()[:8], "big"))


async def _eval(env, model: str, url: str, seed: int, task_id: int, params: dict, timeout: float = 600):
    t0 = asyncio.get_event_loop().time()
    try:
        r = await asyncio.wait_for(
            env.evaluate(model=model, base_url=url, seed=seed, task_id=task_id, **params),
            timeout=timeout,
        )
        dt = asyncio.get_event_loop().time() - t0
        log.debug(f"eval ok {dt:.2f}s model={model}")
        return bool(r.get("success", r.get("score", 0) > 0)), dt
    except asyncio.TimeoutError:
        dt = asyncio.get_event_loop().time() - t0
        log.warning(f"eval timed out after {timeout}s: {model}")
        return False, dt
    except Exception as e:
        dt = asyncio.get_event_loop().time() - t0
        log.warning(f"eval error ({model}) after {dt:.2f}s: {e}")
        return None, dt


def _enumerate_work(env_data, master, max_tasks, tasks_per_batch):
    scheduled = {n: 0 for n in env_data}
    for batch in range(max_tasks):
        if all(scheduled[n] >= max_tasks for n in env_data):
            return
        for name, (wrapper, params, task_timeout) in env_data.items():
            if scheduled[name] >= max_tasks:
                continue
            rng = _batch_rng(master, batch, name)
            room = max_tasks - scheduled[name]
            for _ in range(min(tasks_per_batch, room)):
                tid = rng.randint(0, max_tasks - 1) if max_tasks > 0 else 0
                seed = rng.randint(0, 2**32 - 1)
                yield name, tid, seed, wrapper, params, task_timeout
                scheduled[name] += 1


async def run_duel(
    envs: dict[str, tuple[Any, EnvSpec]],
    champion: Slot,
    challenger: Slot,
    *,
    max_tasks: int = 200,
    tasks_per_batch: int = 4,
    k: float = 2.0,
    nonce: int = 0,
    hotkey: str,
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
    master = _master_seed(champion, challenger, nonce, hotkey)
    lat_champ: list[float] = []
    lat_chall: list[float] = []
    t_start = asyncio.get_event_loop().time()

    async def run_one(name, tid, seed, wrapper, params, task_timeout):
        (champ, champ_dt), (chall, chall_dt) = await asyncio.gather(
            _eval(wrapper, champion.model, champion.base_url, seed, tid, params, task_timeout),
            _eval(wrapper, challenger.model, challenger.base_url, seed, tid, params, task_timeout),
        )
        return name, champ, chall, champ_dt, chall_dt

    work = _enumerate_work(env_data, master, max_tasks, tasks_per_batch)
    inflight_cap = _inflight_cap(tasks_per_batch, len(env_data))
    pending: set[asyncio.Task] = set()
    champ_err = chall_err = 0
    next_progress = progress_interval if progress_interval > 0 else None
    verdict = Verdict.UNDECIDED
    z = 0.0
    exhausted = False
    log.info(f"duel start: inflight_cap={inflight_cap} tasks_per_batch={tasks_per_batch} envs={len(env_data)}")

    try:
        while pending or not exhausted:
            while not exhausted and len(pending) < inflight_cap:
                try:
                    pending.add(asyncio.create_task(run_one(*next(work))))
                except StopIteration:
                    exhausted = True
            if not pending:
                break

            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                name, champ_ok, chall_ok, champ_dt, chall_dt = t.result()
                tasks[name] += 1
                lat_champ.append(champ_dt)
                lat_chall.append(chall_dt)

                champ_err = champ_err + 1 if champ_ok is None else 0
                chall_err = chall_err + 1 if chall_ok is None else 0

                if champ_err >= INFRA_STREAK_LIMIT or chall_err >= INFRA_STREAK_LIMIT:
                    raise RuntimeError(
                        f"sustained infra failure (champion={champ_err}, "
                        f"challenger={chall_err} consecutive errors)"
                    )

                if champ_ok is not None and chall_ok is not None and champ_ok != chall_ok:
                    (wins if chall_ok else losses)[name] += 1

            verdict, z = check_duel(wins, losses, tasks, max_tasks, k)
            if next_progress is not None:
                total_tasks = sum(tasks.values())
                while total_tasks >= next_progress:
                    _log_progress(wins, losses, tasks, total_tasks, k, z)
                    _log_latency(lat_champ, lat_chall, t_start, inflight_cap)
                    next_progress += progress_interval
            if verdict is not Verdict.UNDECIDED:
                break
    finally:
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    _log_latency(lat_champ, lat_chall, t_start, inflight_cap)

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

    per_env = {}
    for name in wins:
        w, l = wins[name], losses[name]
        if w + l > 0:
            d, v = bt_mle(w, l)
            log.info(f"  {name}: W={w} L={l} n={tasks[name]} delta={d:.3f} se={sqrt(v):.3f}")
            per_env[name] = {"W": w, "L": l, "n": tasks[name], "delta": d, "se": sqrt(v)}
        else:
            log.info(f"  {name}: W=0 L=0 n={tasks[name]}")
            per_env[name] = {"W": 0, "L": 0, "n": tasks[name], "delta": None, "se": None}

    path = os.getenv("AFFINE_SHADOW_LOG")
    if path:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": "duel",
            "champion": {"model": champion.model, "revision": champion.revision},
            "challenger": {"model": challenger.model, "revision": challenger.revision},
            "verdict": verdict.name,
            "z": z, "k": k,
            "total_tasks": total_tasks, "decisive": decisive,
            "per_env": per_env,
        }
        try:
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            log.warning(f"shadow log write failed: {e}")


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
