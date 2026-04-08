from __future__ import annotations
import asyncio
import hashlib
import logging
import random
from typing import Any

from .config import EnvSpec
from .vllm import Slot, health_ping
from .wilson import Verdict, wilson_lower

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
    """Returns True (success), False (task failure/timeout), or None (infra error)."""
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


def _check_env(wins: int, losses: int, tasks: int, max_tasks: int, z: float) -> Verdict:
    n = wins + losses
    if n > 0 and wilson_lower(wins, n, z) > 0.5:
        return Verdict.CHALLENGER_WINS
    remaining = max_tasks - tasks
    if remaining <= 0:
        return Verdict.CHAMPION_HOLDS
    if wilson_lower(wins + remaining, n + remaining, z) <= 0.5:
        return Verdict.CHAMPION_HOLDS
    return Verdict.UNDECIDED


def _check_majority(verdicts: dict[str, Verdict]) -> Verdict:
    majority = len(verdicts) // 2 + 1
    won = sum(v is Verdict.CHALLENGER_WINS for v in verdicts.values())
    pending = sum(v is Verdict.UNDECIDED for v in verdicts.values())
    if won >= majority:
        return Verdict.CHALLENGER_WINS
    if won + pending < majority:
        return Verdict.CHAMPION_HOLDS
    return Verdict.UNDECIDED


async def run_duel(
    envs: dict[str, tuple[Any, EnvSpec]],
    champion: Slot,
    challenger: Slot,
    *,
    max_tasks: int = 200,
    tasks_per_batch: int = 4,
    z: float = 1.96,
    nonce: int = 0,
) -> Verdict:
    state = {
        name: {"wins": 0, "losses": 0, "tasks": 0, "verdict": Verdict.UNDECIDED}
        for name in envs
    }
    master = _master_seed(champion, challenger, nonce)
    champ_err_streak = 0
    chall_err_streak = 0

    for batch in range(max_tasks):
        coros, keys = [], []

        for name, (wrapper, spec) in envs.items():
            s = state[name]
            if s["verdict"] is not Verdict.UNDECIDED:
                continue
            if s["tasks"] >= max_tasks:
                s["verdict"] = Verdict.CHAMPION_HOLDS
                continue

            params = dict(spec.params)
            task_timeout = params.pop("timeout", 600)
            rng = _batch_rng(master, batch, name)
            room = max_tasks - s["tasks"]

            for _ in range(min(tasks_per_batch, room)):
                tid = rng.randint(0, 2**31 - 1)
                seed = rng.randint(0, 2**32 - 1)

                async def _pair(w=wrapper, s=seed, t=tid, p=params, to=task_timeout):
                    return await asyncio.gather(
                        _eval(w, champion.model, champion.base_url, s, t, p, to),
                        _eval(w, challenger.model, challenger.base_url, s, t, p, to),
                    )

                coros.append(_pair())
                keys.append(name)

        if not coros:
            break

        results = await asyncio.gather(*coros, return_exceptions=True)

        batch_champ_any_ok = False
        batch_chall_any_ok = False

        for env_name, result in zip(keys, results):
            s = state[env_name]
            s["tasks"] += 1
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

            if champ_ok is None or chall_ok is None:
                continue

            if champ_ok != chall_ok:
                s["wins" if chall_ok else "losses"] += 1

        if champ_err_streak >= INFRA_STREAK_LIMIT or chall_err_streak >= INFRA_STREAK_LIMIT:
            raise RuntimeError(
                f"sustained infra failure (champion={champ_err_streak}, "
                f"challenger={chall_err_streak} consecutive errors)"
            )

        # Mid-duel liveness: if a slot had zero successes in this batch,
        # confirm it's actually reachable. Catches dead endpoints whose
        # timeouts/failures look like task losses or produce ties.
        if not batch_champ_any_ok:
            if not await health_ping(champion.base_url):
                log.info("champion confirmed down mid-duel — challenger wins by default")
                _log_summary(state, Verdict.CHALLENGER_WINS, champion, challenger)
                return Verdict.CHALLENGER_WINS
        if not batch_chall_any_ok:
            if not await health_ping(challenger.base_url):
                log.info("challenger confirmed down mid-duel — champion holds")
                _log_summary(state, Verdict.CHAMPION_HOLDS, champion, challenger)
                return Verdict.CHAMPION_HOLDS

        for name, s in state.items():
            if s["verdict"] is Verdict.UNDECIDED:
                s["verdict"] = _check_env(s["wins"], s["losses"], s["tasks"], max_tasks, z)
                if s["verdict"] is not Verdict.UNDECIDED:
                    log.info(f"  {name}: {s['verdict'].name} W={s['wins']} L={s['losses']} n={s['tasks']}")

        overall = _check_majority({n: s["verdict"] for n, s in state.items()})
        if overall is not Verdict.UNDECIDED:
            _log_summary(state, overall, champion, challenger)
            return overall

    overall = _check_majority({n: s["verdict"] for n, s in state.items()})
    _log_summary(state, overall, champion, challenger)
    return overall


def _log_summary(state: dict, verdict: Verdict, champion: Slot, challenger: Slot):
    total = sum(s["tasks"] for s in state.values())
    decisive = sum(s["wins"] + s["losses"] for s in state.values())
    log.info(f"duel: {verdict.name} | {champion.model} vs {challenger.model} | {total} tasks, {decisive} decisive")
    for name, s in state.items():
        log.info(f"  {name}: {s['verdict'].name} W={s['wins']} L={s['losses']} n={s['tasks']}")
