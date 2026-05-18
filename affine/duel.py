from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
from typing import TYPE_CHECKING

from .decide import BettingCS, DuelOutcome, Verdict, decide
from .sampler import run_one

if TYPE_CHECKING:
    from .chain import Miner
    from .config import Config
    from .loop import Chain
    from .store import Champion, Store
    from .vllm import Slot

log = logging.getLogger(__name__)


def task_keys(validator_hotkey: str, schedule_seed: str, env_id: str, round_idx: int) -> tuple[int, int]:
    key = f"{validator_hotkey}\0{schedule_seed}\0{env_id}\0{round_idx}".encode()
    task_digest = hashlib.sha256(key).digest()
    seed_digest = hashlib.sha256(key + b"\0seed").digest()
    task_id = int.from_bytes(task_digest[:8], "big") & ((1 << 63) - 1)
    seed = int.from_bytes(seed_digest[:4], "big") & ((1 << 31) - 1)
    return task_id, seed


def _terminal_verdict(status: str, cs: BettingCS) -> Verdict:
    lo, hi = cs.ci()
    return Verdict(DuelOutcome.NO_DETHRONE, status, cs.mean, lo, hi, cs.n, cs._log_capital(0.0, +1))


async def _block(chain: "Chain") -> int | None:
    try:
        return await chain.current_block()
    except Exception as exc:
        log.warning(f"current_block failed: {exc}; leaving finished_block empty")
        return None


def _finish(store: "Store", duel_id: int, status: str, rounds_collected: int, verdict: Verdict, block: int | None) -> None:
    store.finish_duel(
        duel_id,
        status,
        rounds_collected,
        verdict.delta_hat,
        verdict.ci_low,
        verdict.ci_hi,
        block,
    )


async def run_duel(
    store: "Store",
    chain: "Chain",
    cfg: "Config",
    envs: dict[str, tuple],
    pi: dict[str, float],
    versions_hash: str,
    champ: "Champion",
    king_slot: "Slot",
    challenger: "Miner",
    chal_slot: "Slot",
    stop: asyncio.Event,
) -> Verdict:
    duel = store.create_duel(
        champion=champ,
        challenger_uid=challenger.uid,
        challenger_hotkey=challenger.hotkey,
        challenger_model=challenger.model,
        challenger_revision=challenger.revision,
        validator_hotkey=chain.hotkey,
        schedule_seed=secrets.token_hex(16),
        alpha=cfg.alpha,
        delta_dethrone=cfg.delta_dethrone,
        delta_hold=cfg.delta_hold,
        pi=pi,
        versions_hash=versions_hash,
        started_block=await chain.current_block(),
    )
    env_ids = sorted(envs)
    if not env_ids:
        raise ValueError("no environments loaded")

    cs = BettingCS(alpha=cfg.alpha)
    consec_dead = 0

    for round_idx in range(cfg.rounds_max):
        if stop.is_set():
            verdict = _terminal_verdict("cancelled", cs)
            _finish(store, duel.id, "cancelled", cs.n, verdict, None)
            return verdict

        round_samples = []
        round_y = 0.0
        round_complete = True

        for env_id in env_ids:
            if stop.is_set():
                verdict = _terminal_verdict("cancelled", cs)
                _finish(store, duel.id, "cancelled", cs.n, verdict, None)
                return verdict

            wrapper, spec = envs[env_id]
            params = {k: v for k, v in spec.params.items() if k != "timeout"}
            timeout = float(spec.params.get("timeout", 600))
            task_id, seed = task_keys(chain.hotkey, duel.schedule_seed, env_id, round_idx)

            champ_result, chal_result = await asyncio.gather(
                run_one(wrapper, params, timeout, king_slot, seed=seed, task_id=task_id),
                run_one(wrapper, params, timeout, chal_slot, seed=seed, task_id=task_id),
            )
            oc, lc, tc = champ_result
            oh, lh, th = chal_result
            if oc is None or oh is None:
                consec_dead += 1
                round_complete = False
                if consec_dead >= cfg.slot_dead_run:
                    verdict = _terminal_verdict("challenger_slot_dead", cs)
                    _finish(store, duel.id, "challenger_slot_dead", cs.n, verdict, await _block(chain))
                    return verdict
                break

            champ_correct = int(bool(oc))
            chal_correct = int(bool(oh))
            d = chal_correct - champ_correct
            round_y += float(pi[env_id]) * d
            round_samples.append(
                (
                    duel.id,
                    round_idx,
                    env_id,
                    task_id,
                    seed,
                    champ_correct,
                    chal_correct,
                    lc,
                    lh,
                    tc,
                    th,
                )
            )

        if not round_complete:
            continue

        consec_dead = 0
        for sample in round_samples:
            store.record_sample(*sample)
        cs.update(round_y)

        verdict = decide(cs, delta_dethrone=cfg.delta_dethrone, delta_hold=cfg.delta_hold)
        log.info(
            f"verdict: {verdict.reason} Δ̂={verdict.delta_hat:+.4f} "
            f"CI=[{verdict.ci_low:+.4f}, {verdict.ci_hi:+.4f}] "
            f"logK={verdict.log_capital_at_zero:+.4f}"
        )
        if verdict.reason != "continue":
            _finish(store, duel.id, verdict.reason, cs.n, verdict, await _block(chain))
            return verdict

    final = decide(cs, delta_dethrone=cfg.delta_dethrone, delta_hold=cfg.delta_hold)
    reason = "inconclusive" if final.reason == "continue" else final.reason
    outcome = DuelOutcome.DETHRONE if reason == "dethrone" else DuelOutcome.NO_DETHRONE
    verdict = Verdict(outcome, reason, final.delta_hat, final.ci_low, final.ci_hi, cs.n, final.log_capital_at_zero)
    _finish(store, duel.id, reason, cs.n, verdict, await _block(chain))
    return verdict
