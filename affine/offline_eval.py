"""Offline (shadow) evaluation harness.

Production `_task_id` (`affine/loop.py:188`) hashes `miner_uid` into the per-cell
task_id so two miners never see the same prompt — required against memorization
and replay across many duels. For offline correlation / order-stability studies
we want the opposite: every miner runs the same N prompts per env so the IRT
fit and its decision rule can be replayed offline against a comparable matrix.

This module exposes:
  * `shared_task_id`  — env+iter only, miner-independent
  * `eval_one_miner`  — provision → fixed env-blocked sweep → teardown, writes
                        CellObservation rows via `Store`

Failure semantics:
  * provision returns None → drop miner ("skipped")
  * SLOT_DEAD_RUN consecutive infra-failure (None) outcomes on one env → abort
    that miner; partial cells already written stay

The live `_task_id` is untouched.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from .chain import Miner
from .config import EnvSpec
from .envs._base import EnvFactory
from .loop import _hf_ref_missing, _pin_hf_revision, _provision, _safe_teardown
from .sampler import run_one
from .store import CellObservation, Store, artifact_id, make_observation_id
from .vllm import Slot, VllmSlots

log = logging.getLogger(__name__)

OFFLINE_SERVING_HASH = "offline_eval_v1"
OFFLINE_SALT = "offline_eval_v1"
OFFLINE_SAMPLER_POLICY_HASH = "offline_fixed_env_blocked_v1"
SLOT_DEAD_RUN = 5


def shared_task_id(env: str, iter_idx: int, lo: int, hi: int,
                   salt: str = OFFLINE_SALT) -> int:
    """sha256(salt, env, iter_idx) → uniform in [lo, hi]. Drops `miner_uid` so
    every miner draws the same task on the same (env, iter_idx)."""
    h = hashlib.sha256(f"{salt}\0{env}\0{iter_idx}".encode()).digest()
    return lo + (int.from_bytes(h[:8], "big") % (hi - lo + 1))


def shared_seed(env: str, iter_idx: int, salt: str = OFFLINE_SALT) -> int:
    h = hashlib.sha256(f"{salt}\0seed\0{env}\0{iter_idx}".encode()).digest()
    return int.from_bytes(h[:4], "big") & ((1 << 31) - 1)


@dataclass
class EvalResult:
    artifact_id: str
    miner_uid: int | None
    model: str
    revision: str
    status: str            # ok | skipped_provision | partial_slot_dead | error
    cells_written: int
    elapsed_s: float
    error: str | None = None


async def _shared_cell_sample(
    slot: Slot,
    env_name: str,
    spec: EnvSpec,
    wrapper: EnvFactory,
    iter_idx: int,
    *,
    miner_artifact_id: str,
    env_version: str,
    task_spec_hash: str,
    grader_hash: str,
    collection_context: str,
) -> CellObservation | None:
    """Mirrors `affine/loop.py::_cell_sample` but with miner-independent task_id
    and a fixed offline serving_hash. Returns None on infra failure."""
    params = {k: v for k, v in spec.params.items() if k != "timeout"}
    timeout = float(spec.params.get("timeout", 600))
    lo, hi = spec.task_range
    task_id = shared_task_id(env_name, iter_idx, lo, hi)
    seed = shared_seed(env_name, iter_idx)
    outcome, latency, tokens = await run_one(
        wrapper, params, timeout, slot, seed=seed, task_id=task_id,
    )
    if outcome is None:
        return None
    raw = int(bool(outcome))
    return CellObservation(
        observation_id=make_observation_id(),
        miner_artifact_id=miner_artifact_id,
        env_id=env_name,
        env_version=env_version,
        task_id=int(task_id),
        task_spec_hash=task_spec_hash,
        grader_hash=grader_hash,
        serving_hash=OFFLINE_SERVING_HASH,
        raw_outcome=raw,
        outcome=raw,
        gated=0,
        latency_s=float(latency),
        tokens=int(tokens),
        observed_at=int(time.time()),
        collection_context=collection_context,
        sampler_policy_hash=OFFLINE_SAMPLER_POLICY_HASH,
    )


def _existing_iter_idxs(store: Store, miner_artifact_id: str, env: str,
                        env_version: str, task_spec_hash: str, grader_hash: str,
                        ) -> set[int]:
    rows = store.db.execute(
        """SELECT task_id FROM cell_observations
           WHERE miner_artifact_id=? AND env_id=? AND env_version=?
             AND task_spec_hash=? AND grader_hash=? AND serving_hash=?""",
        (miner_artifact_id, env, env_version, task_spec_hash, grader_hash,
         OFFLINE_SERVING_HASH),
    ).fetchall()
    return {int(r["task_id"]) for r in rows}


async def _sweep_affine_envs(
    slot: Slot,
    miner_artifact_id: str,
    env_specs: dict[str, tuple[EnvFactory, EnvSpec]],
    env_states: dict[str, dict],
    store: Store,
    tasks_per_env: int,
    collection_context: str,
) -> tuple[int, str]:
    """Run `tasks_per_env` cells for each env, in env-blocked order. Resumable:
    skips iter_idx whose task_id is already in the DB for this miner+env."""
    written = 0
    for env_name, (wrapper, spec) in env_specs.items():
        st = env_states[env_name]
        already = _existing_iter_idxs(
            store, miner_artifact_id, env_name,
            st["env_version"], st["task_spec_hash"], st["grader_hash"],
        )
        consec_dead = 0
        for i in range(tasks_per_env):
            lo, hi = spec.task_range
            tid = shared_task_id(env_name, i, lo, hi)
            if tid in already:
                continue
            obs = await _shared_cell_sample(
                slot, env_name, spec, wrapper, i,
                miner_artifact_id=miner_artifact_id,
                env_version=st["env_version"],
                task_spec_hash=st["task_spec_hash"],
                grader_hash=st["grader_hash"],
                collection_context=collection_context,
            )
            if obs is None:
                consec_dead += 1
                if consec_dead >= SLOT_DEAD_RUN:
                    log.warning(
                        f"slot dead after {SLOT_DEAD_RUN} consecutive None on "
                        f"{env_name}; aborting miner {miner_artifact_id[:12]}"
                    )
                    return written, "partial_slot_dead"
                continue
            consec_dead = 0
            store.add_observation(obs)
            written += 1
    return written, "ok"


async def eval_one_miner(
    slots_chain: list[VllmSlots],
    miner: Miner,
    store: Store,
    env_specs: dict[str, tuple[EnvFactory, EnvSpec]],
    env_states: dict[str, dict],
    *,
    tasks_per_env: int,
    collection_context: str,
    semaphore: asyncio.Semaphore,
    stop: asyncio.Event,
    skipped_log: Path,
) -> EvalResult:
    art = artifact_id(miner.model, miner.revision)
    t0 = time.monotonic()
    async with semaphore:
        # Preflight HF ref check: a 30-min wait_ready on a 404'd repo is the most
        # common cause of "skipped_provision" on subnet 120 (miners' HF artifacts
        # rot). One model_info() call here is ~0.2s vs 30 min downstream.
        try:
            await asyncio.to_thread(_pin_hf_revision, miner.model, miner.revision)
        except Exception as e:
            reason = "hf_ref_missing" if _hf_ref_missing(e) else f"hf_lookup_error:{type(e).__name__}"
            with skipped_log.open("a") as f:
                f.write(json.dumps({
                    "artifact_id": art, "uid": miner.uid,
                    "model": miner.model, "revision": miner.revision,
                    "reason": reason, "error": str(e)[:200],
                    "at": int(time.time()),
                }) + "\n")
            return EvalResult(art, miner.uid, miner.model, miner.revision,
                              "skipped_hf_missing" if _hf_ref_missing(e) else "skipped_hf_error",
                              0, time.monotonic() - t0)
        slot = await _provision(slots_chain, miner, stop)
        if slot is None:
            with skipped_log.open("a") as f:
                f.write(json.dumps({
                    "artifact_id": art, "uid": miner.uid,
                    "model": miner.model, "revision": miner.revision,
                    "reason": "provision_failed",
                    "at": int(time.time()),
                }) + "\n")
            return EvalResult(art, miner.uid, miner.model, miner.revision,
                              "skipped_provision", 0, time.monotonic() - t0)
        try:
            written, status = await _sweep_affine_envs(
                slot, art, env_specs, env_states, store,
                tasks_per_env, collection_context,
            )
            return EvalResult(art, miner.uid, miner.model, miner.revision,
                              status, written, time.monotonic() - t0)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning(f"eval_one_miner failed for {miner.model}@{miner.revision}: "
                        f"{type(e).__name__}: {e}")
            return EvalResult(art, miner.uid, miner.model, miner.revision,
                              "error", 0, time.monotonic() - t0, error=str(e))
        finally:
            await _safe_teardown(slot, "offline_eval")
