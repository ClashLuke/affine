"""Colocated-backup king-of-the-hill validator loop with IRT eval.

Per-duel evaluation is via a 2PL IRT scalar log-odds rating contrast
(`affine.irt`); the data plane is `cell_observations` not pairs
(`affine.store`); cummax queue containment lives at the loop layer
(`notes/cummax.md`). See `notes/eval-target.md` for the design rationale
and `notes/finish.md` for the rest of the architectural contract.
"""

from __future__ import annotations
import asyncio
import hashlib
import logging
import math
import os
import secrets
import signal
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import bittensor as bt
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

from .backup import (
    ManifestRef,
    S3Config,
    delete_refs,
    encode_refs,
)
from .chain import Miner, Subtensor, _tiebreak, _truthy_env, clear_weights, get_miners, set_weights
from .config import BASELINE_MODELS, Config, EnvSpec
from .envs import EnvFactory
from .irt import (
    CalibrationSnapshot,
    CellTriple,
    DecisionState,
    alpha_spent_z,
    archive_fit,
    calibration_snapshot,
    champion_se_theta,
    decide_theta,
    decision_fit,
    theta_diff_se,
)
from .sampler import run_one
from .score import (
    EnvCost,
    env_versioning,
    evaluation_version_hash,
    normalize_pi,
    normalize_rho,
    sampler_pick_env,
    serving_hash_for,
)
from .store import (
    BackupRecord,
    Cell,
    CellObservation,
    Champion,
    Store,
    artifact_id,
    make_observation_id,
)
from .vllm import (
    Slot,
    SlotProvisionFailed,
    VllmSlots,
    make_slots,
    poll_backup,
)

log = logging.getLogger(__name__)

KING_REPROVISION_LIMIT = 10
BACKUP_RETIREMENT_GRACE_S = 600
SLOT_DEAD_RUN = 30   # consecutive infra-fault cells from challenger → abort duel


@dataclass
class Chain:
    """Everything the loop needs from the world besides slots, envs, and evidence."""
    hotkey: str
    list_miners: Callable[[], Awaitable[list[Miner]]]
    current_block: Callable[[], Awaitable[int]]
    publish_winner: Callable[[int, str], Awaitable[bool]]
    burn_weights: Callable[[], Awaitable[bool]]
    uid_matches_hotkey: Callable[[int, str], Awaitable[bool]]


# ---------------------------------------------------------------------------
# Async cancellation primitives (preserved from previous design)
# ---------------------------------------------------------------------------

async def _load_envs(cfg: Config) -> dict[str, tuple]:
    loaded: dict[str, EnvFactory] = {}
    out = {}
    for spec in cfg.environments:
        if spec.entrypoint not in loaded:
            loaded[spec.entrypoint] = EnvFactory(spec.entrypoint)
            log.info(f"env: {spec.name} ({spec.entrypoint})")
        else:
            log.info(f"env: {spec.name} ({spec.entrypoint}) [shared]")
        out[spec.name] = (loaded[spec.entrypoint], spec)
    return out


async def _cancellable(coro, stop: asyncio.Event, on_orphan=None):
    """Await `coro` unless `stop` fires, in which case cancel and raise CancelledError."""
    worker = asyncio.ensure_future(coro)
    stopper = asyncio.ensure_future(stop.wait())
    delivered = False
    try:
        done, _ = await asyncio.wait({worker, stopper}, return_when=asyncio.FIRST_COMPLETED)
        if worker in done:
            delivered = True
            return worker.result()
        raise asyncio.CancelledError("stop event set")
    finally:
        stopper.cancel()
        if not worker.done():
            worker.cancel()
            try: await asyncio.wait_for(asyncio.shield(worker), timeout=60)
            except asyncio.TimeoutError:
                log.warning("_cancellable: worker cleanup exceeded 60s; rental may leak until reconcile")
            except asyncio.CancelledError:
                pass
            except BaseException as e:
                log.warning(f"_cancellable: worker cleanup failed ({type(e).__name__}); rental may leak until reconcile: {e}")
        if not delivered and worker.done() and not worker.cancelled() and worker.exception() is None:
            orphan = worker.result()
            if on_orphan is not None and orphan is not None:
                try: await asyncio.shield(on_orphan(orphan))
                except BaseException as e:
                    log.warning(f"_cancellable: on_orphan failed: {type(e).__name__}: {e}")


async def _safe_teardown(slot: Slot, ctx: str) -> None:
    try: await asyncio.shield(slot.teardown())
    except Exception as e: log.warning(f"teardown error ({ctx}): {e}")


def _shutdown_signals() -> tuple[signal.Signals, ...]:
    sigs = [signal.SIGTERM, signal.SIGINT]
    if hasattr(signal, "SIGHUP"):
        sigs.append(signal.SIGHUP)
    return tuple(sigs)


def _install_signal_handlers(stop: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in _shutdown_signals():
        def request_stop(sig=sig):
            log.warning(f"shutdown signal received: {sig.name}")
            stop.set()
        try:
            loop.add_signal_handler(sig, request_stop)
        except (NotImplementedError, ValueError):
            signal.signal(sig, lambda *_args, request_stop=request_stop: request_stop())


async def _provision(chain: list[VllmSlots], miner: Miner,
                     stop: asyncio.Event, **kwargs) -> Slot | None:
    for i, slots in enumerate(chain):
        try:
            return await _cancellable(
                slots.provision(miner.model, miner.revision, **kwargs), stop,
                on_orphan=lambda s: _safe_teardown(s, "provision-orphan"),
            )
        except SlotProvisionFailed as e:
            log.warning(f"provision crashloop {miner.model}@{miner.revision}: {e}")
            return None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            tail = "; falling through" if i + 1 < len(chain) else " (no fallback left)"
            log.warning(f"{slots.NAME} provision failed{tail} uid{miner.uid} "
                        f"{miner.model}@{miner.revision}: {type(e).__name__}: {e}")
    return None


def _seed(uid: int, rev: str, env: str, c: int, salt: str = "") -> int:
    h = hashlib.sha256(f"{salt}\0{uid}\0{rev}\0{env}\0{c}".encode()).digest()
    return int.from_bytes(h[:4], "big") & ((1 << 31) - 1)


def _task_id(miner_uid: int, env: str, iter_idx: int,
             lo: int, hi: int, salt: str = "") -> int:
    """Per-cell task_id. Single-miner version (no pairing). Salt + miner uid +
    env + iter index → uniform draw in [lo, hi]. Adversarial cost: 2^63 task
    space defeats memorization."""
    h = hashlib.sha256(f"{salt}\0{miner_uid}\0{env}\0{iter_idx}".encode()).digest()
    n = hi - lo + 1
    return lo + (int.from_bytes(h[:8], "big") % n)


# ---------------------------------------------------------------------------
# Cell sampling
# ---------------------------------------------------------------------------

async def _cell_sample(
    chain: Chain,
    slot: Slot,
    miner: Miner,
    env_name: str,
    spec,
    wrapper,
    iter_idx: int,
    *,
    miner_artifact_id: str,
    env_version: str,
    task_spec_hash: str,
    grader_hash: str,
    serving_hash: str,
    sampler_policy_hash: str,
    collection_context: str,
) -> CellObservation | None:
    """Run one task on `slot` for `miner` on `env_name`, return a
    CellObservation. Returns None if the inference failed at infra level
    (None outcome from `run_one`); the caller treats this as a SLOT_DEAD-style
    consecutive-failure signal but does not insert the cell."""
    params = {k: v for k, v in spec.params.items() if k != "timeout"}
    timeout = float(spec.params.get("timeout", 600))
    lo, hi = spec.task_range
    miner_uid = miner.uid if miner.uid is not None else -1
    task_id = _task_id(miner_uid, env_name, iter_idx, lo, hi,
                       salt=f"{chain.hotkey}:cell")
    seed = _seed(miner_uid, miner.revision, env_name, iter_idx, salt=chain.hotkey)
    outcome, latency, tokens = await run_one(wrapper, params, timeout, slot,
                                             seed=seed, task_id=task_id)
    if outcome is None:
        return None
    raw = int(bool(outcome))
    # D11 hard gates: future work; for MVP latency cap is the timeout itself
    # (already enforced inside run_one as miner-loss). Tokens cap not yet
    # configured per env; placeholder.
    gated = 0
    return CellObservation(
        observation_id=make_observation_id(),
        miner_artifact_id=miner_artifact_id,
        env_id=env_name,
        env_version=env_version,
        task_id=int(task_id),
        task_spec_hash=task_spec_hash,
        grader_hash=grader_hash,
        serving_hash=serving_hash,
        raw_outcome=raw,
        outcome=raw if not gated else 0,
        gated=gated,
        latency_s=float(latency),
        tokens=int(tokens),
        observed_at=int(time.time()),
        collection_context=collection_context,
        sampler_policy_hash=sampler_policy_hash,
    )


# ---------------------------------------------------------------------------
# Run state (cummax: durable _attempted)
# ---------------------------------------------------------------------------

@dataclass
class _RunState:
    king_slot: Slot | None = None
    attempted: set[str] = field(default_factory=set)   # miner_artifact_id values
    next_challenger: Miner | None = None
    prefetch_task: asyncio.Task | None = None


@dataclass(frozen=True)
class _DuelVerdict:
    status: str            # dethrone | statistical_hold | budget_hold_inconclusive | challenger_slot_dead | cancelled | calibration_needed
    cells_collected: int
    delta_theta: float | None
    se_theta: float | None
    rating_diff_diagnostic: float | None
    calibration_snapshot_hash: str | None


# ---------------------------------------------------------------------------
# Backup configs / champion miner / pinning (preserved)
# ---------------------------------------------------------------------------

def _backup_configs(cfg: Config, hotkey: str) -> list[S3Config]:
    if _truthy_env("AFFINE_LOCAL"):
        return []
    hot_hash = hashlib.sha256(hotkey.encode()).hexdigest()[:16]
    namespace = os.getenv("AFFINE_NAMESPACE", "prod").strip().strip("/") or "prod"
    configs = S3Config.from_env(default_prefix=f"{cfg.netuid}/{namespace}/{hot_hash}")
    if not configs:
        raise RuntimeError("Hippius or R2 S3 credentials are required for production runs")
    return list(configs.values())


def _champion_miner(champ: Champion) -> Miner:
    return Miner(
        uid=champ.uid,
        hotkey=champ.hotkey or "",
        model=champ.model,
        revision=champ.revision,
        block=champ.reign_start,
    )


def _pin_hf_revision(model: str, revision: str) -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    return HfApi(token=token).model_info(model, revision=revision).sha


def _hf_ref_missing(exc: Exception) -> bool:
    return isinstance(exc, (RepositoryNotFoundError, RevisionNotFoundError))


def _with_revision(miner: Miner, revision: str) -> Miner:
    return Miner(uid=miner.uid, hotkey=miner.hotkey, model=miner.model,
                 revision=revision, block=miner.block)


async def _pin_artifact(s3_configs: list[S3Config], miner: Miner) -> Miner:
    if not s3_configs:
        return miner
    revision = await asyncio.to_thread(_pin_hf_revision, miner.model, miner.revision)
    return _with_revision(miner, revision)


async def _registered_artifact_alive(s3_configs: list[S3Config], miners: list[Miner],
                                     registered: Miner, pinned_revision: str) -> bool:
    for m in miners:
        if m.uid != registered.uid or m.hotkey != registered.hotkey or m.model != registered.model:
            continue
        if not s3_configs:
            return m.revision == registered.revision
        try:
            return await asyncio.to_thread(_pin_hf_revision, m.model, m.revision) == pinned_revision
        except Exception as exc:
            log.warning(f"fresh pin failed for uid{m.uid} {m.model}@{m.revision}: {exc}")
            return False
    return False


async def _champion_registration_status(
    s3_configs: list[S3Config],
    miners: list[Miner],
    champ: Champion,
    *,
    missing_ref_alive: bool = False,
) -> str:
    if not champ.payable or champ.uid is None or not champ.hotkey:
        return "dead"
    for m in miners:
        if m.uid != champ.uid or m.hotkey != champ.hotkey or m.model != champ.model:
            continue
        if not s3_configs:
            return "alive" if m.revision == champ.revision else "dead"
        if m.revision == champ.revision:
            return "alive"
        try:
            return "alive" if await asyncio.to_thread(_pin_hf_revision, m.model, m.revision) == champ.revision else "dead"
        except Exception as exc:
            if _hf_ref_missing(exc):
                if missing_ref_alive:
                    log.warning(f"champion HF ref missing for uid{m.uid} {m.model}@{m.revision}; continuing from backup")
                    return "alive"
                log.warning(f"champion HF ref missing for uid{m.uid} {m.model}@{m.revision}")
                return "dead"
            log.warning(f"champion pin check failed for uid{m.uid} {m.model}@{m.revision}: {exc}")
            return "unknown"
    return "dead"


def _is_current_champion_registration(m: Miner, champ: Champion, artifact_alive: bool = False) -> bool:
    if not champ.payable:
        return False
    if champ.uid is not None and m.uid == champ.uid and m.hotkey == champ.hotkey:
        return (m.model, m.revision) == (champ.model, champ.revision) or (artifact_alive and m.model == champ.model)
    return (m.model, m.revision) == (champ.model, champ.revision)


def _baseline_model(cfg: Config) -> str:
    explicit = os.getenv("AFFINE_BASELINE_MODEL", "").strip()
    if explicit:
        if explicit in cfg.model_skiplist:
            raise RuntimeError(f"AFFINE_BASELINE_MODEL={explicit!r} is skiplisted")
        return explicit
    for model in BASELINE_MODELS:
        if model not in cfg.model_skiplist:
            return model
    raise RuntimeError("all baseline models are skiplisted")


async def _bootstrap_champion(
    store: Store,
    s3_configs: list[S3Config],
    slots: list[VllmSlots],
    chain: Chain,
    cfg: Config,
    stop: asyncio.Event,
) -> tuple[Champion, Slot]:
    baseline_model = _baseline_model(cfg)
    baseline_revision = os.getenv("AFFINE_BASELINE_REVISION", "").strip()
    miners = [m for m in await chain.list_miners() if m.model not in cfg.model_skiplist]
    registered = next(
        (m for m in miners
         if m.model == baseline_model and (not baseline_revision or m.revision == baseline_revision)),
        None,
    )
    model = registered.model if registered else baseline_model
    revision = registered.revision if registered else baseline_revision
    if not revision:
        revision = "main" if not s3_configs else await asyncio.to_thread(_pin_hf_revision, model, "main")
    if registered and s3_configs:
        revision = await asyncio.to_thread(_pin_hf_revision, model, revision)
        registered = _with_revision(registered, revision)
    art = artifact_id(model, revision)
    block = await chain.current_block()
    champ = Champion(
        artifact_id=art,
        model=model,
        revision=revision,
        uid=registered.uid if registered else None,
        hotkey=registered.hotkey if registered else None,
        reign_start=block,
        payable=registered is not None,
    )
    slot = await _provision(slots, _champion_miner(champ), stop, source="hf")
    if slot is None:
        raise RuntimeError(f"baseline provision failed for {model}@{revision}")
    store.set_champion(champ)
    kind = f"uid{registered.uid}" if registered else "unregistered"
    log.info(f"bootstrap champion: {kind} baseline {model}@{revision}")
    return champ, slot


async def _ensure_king_slot(
    state: _RunState,
    champ: Champion,
    slots: list[VllmSlots],
    stop: asyncio.Event,
    store: Store | None = None,
) -> Slot | None:
    if state.king_slot is not None:
        return state.king_slot
    miner = _champion_miner(champ)
    slot = await _provision(slots, miner, stop, source="hf")
    if slot is None:
        recovered = store.latest_backup_for(champ.artifact_id) if store is not None else None
        if recovered is not None:
            log.warning("champion HF reprovision failed; using backup")
            slot = await _provision(
                slots, miner, stop, source="s3", backup_manifest_key=recovered.manifest_key,
            )
            if slot is not None and store is not None:
                store.update_backup_manifest(
                    artifact_id=champ.artifact_id,
                    manifest_key=recovered.manifest_key,
                    model=recovered.model,
                    revision=recovered.revision,
                )
    if slot is None:
        log.error("champion reprovision failed")
        return None
    state.king_slot = slot
    return slot


async def _publish_champion(store: Store, chain: Chain, cfg: Config, champ: Champion) -> bool:
    if not champ.payable or champ.uid is None or not champ.hotkey:
        return await _publish_burn(store, chain, cfg, champ.artifact_id)
    if not await chain.uid_matches_hotkey(champ.uid, champ.hotkey):
        log.warning(f"champion uid={champ.uid} hotkey mismatch; burning weights")
        store.demote_champion(champ.artifact_id)
        await _publish_burn(store, chain, cfg, champ.artifact_id)
        return False
    dry = cfg.dry_run
    pub_id = store.publication_intent(champ.artifact_id, "set_weights", champ.uid, champ.hotkey, dry)
    if store.publication_status(pub_id) in {"confirmed", "dry_run"}:
        return True
    if dry:
        store.mark_publication(pub_id, "dry_run")
        return True
    ok = await chain.publish_winner(champ.uid, champ.hotkey)
    if ok and not await chain.uid_matches_hotkey(champ.uid, champ.hotkey):
        log.warning(f"champion uid={champ.uid} hotkey changed after publish; burning weights")
        store.demote_champion(champ.artifact_id)
        await _publish_burn(store, chain, cfg, champ.artifact_id)
        return False
    store.mark_publication(pub_id, "confirmed" if ok else "failed")
    return ok


async def _publish_burn(store: Store, chain: Chain, cfg: Config, artifact: str) -> bool:
    dry = cfg.dry_run
    pub_id = store.publication_intent(artifact, "burn", None, None, dry)
    if store.publication_status(pub_id) in {"confirmed", "dry_run"}:
        return True
    ok = True if dry else bool(await chain.burn_weights())
    if dry:
        store.mark_publication(pub_id, "dry_run")
    else:
        store.mark_publication(pub_id, "confirmed" if ok else "failed")
    return ok


def _refs_to_objs(refs: list[dict]) -> list[ManifestRef]:
    return [ManifestRef(
        provider=str(r["provider"]),
        bucket=str(r["bucket"]),
        key=str(r["key"]),
        prefix=str(r["prefix"]),
        sha256=str(r.get("sha256", "")),
    ) for r in refs]


async def _reconcile_backup_manifest(
    store: Store,
    champ: Champion,
    slot: Slot,
    s3_configs: list[S3Config],
) -> None:
    if not slot.sidecar_url or not s3_configs:
        return
    state = await poll_backup(slot)
    if state is None or state.get("state") != "done":
        return
    if state.get("artifact_id") != champ.artifact_id:
        log.warning(f"sidecar artifact mismatch: slot={state.get('artifact_id')} "
                    f"champ={champ.artifact_id}; skipping reconcile")
        return
    new_refs = state.get("refs") or []
    if not new_refs:
        return
    new_key = encode_refs(_refs_to_objs(new_refs))
    cur = store.latest_backup_for(champ.artifact_id)
    if cur is None or cur.manifest_key != new_key:
        store.update_backup_manifest(
            artifact_id=champ.artifact_id, manifest_key=new_key,
            model=champ.model, revision=champ.revision,
        )


# ---------------------------------------------------------------------------
# Env state lifecycle (D10, simplified): mark all envs `score_active` at MVP
# ---------------------------------------------------------------------------

def _ensure_env_states(store: Store, cfg: Config) -> dict[str, dict]:
    """Make sure every config env has a row in `env_state`. MVP: all envs are
    `score_active` with versioning derived from the spec hash. Lifecycle
    promotion (shadow → calibration_only → score_active) is future work."""
    env_states: dict[str, dict] = {}
    for spec in cfg.environments:
        ev, ts_hash, gh = env_versioning(spec)
        existing = store.env_state(spec.name)
        if existing is None or existing["env_version"] != ev:
            store.upsert_env_state(spec.name, "score_active", ev, ts_hash, gh,
                                   serving_hash="default")
        env_states[spec.name] = {
            "state": "score_active",
            "env_version": ev,
            "task_spec_hash": ts_hash,
            "grader_hash": gh,
            "serving_hash": "default",
        }
    return env_states


def _measurement_envs(env_states: dict[str, dict]) -> set[str]:
    return {e for e, s in env_states.items()
            if s["state"] in ("calibration_only", "score_active")}


def _score_active_envs(env_states: dict[str, dict]) -> set[str]:
    return {e for e, s in env_states.items() if s["state"] == "score_active"}


def _build_pi(cfg: Config, env_states: dict[str, dict]) -> dict[str, float]:
    score_envs = _score_active_envs(env_states)
    overrides = dict(cfg.pi_overrides)
    if overrides:
        proj = {e: float(overrides.get(e, 0.0)) for e in score_envs}
        if sum(proj.values()) > 0:
            return normalize_pi(proj)
    return {e: 1.0 / len(score_envs) for e in score_envs} if score_envs else {}


def _build_rho(cfg: Config, env_ids: list[str]) -> list[float]:
    overrides = {k: v for k, v in cfg.rho_overrides}
    return normalize_rho(overrides, env_ids)


def _evaluation_version(env_states: dict[str, dict], pi: dict[str, float],
                       cfg: Config) -> str:
    measurement_set = {e: env_states[e]["env_version"]
                       for e in _measurement_envs(env_states)}
    rho_dict = dict(cfg.rho_overrides)
    hyperparams = {
        "delta_theta": cfg.delta_theta,
        "irt_K": cfg.irt_K,
        "irt_alpha_dethrone": cfg.irt_alpha_dethrone,
        "irt_beta_hold": cfg.irt_beta_hold,
        "irt_look_max": cfg.irt_look_max,
        "irt_look_interval_cells": cfg.irt_look_interval_cells,
    }
    return evaluation_version_hash(
        measurement_env_set=measurement_set,
        pi=pi,
        rho=rho_dict,
        hyperparams=hyperparams,
    )


# ---------------------------------------------------------------------------
# Duel
# ---------------------------------------------------------------------------

async def _build_archive_snapshot(
    store: Store,
    env_states: dict[str, dict],
    rho: list[float],
):
    """Materialize cells from the canonical view, fit the IRT archive."""
    measurement = _measurement_envs(env_states)
    if not measurement:
        return None, [], []
    cells = store.cells_view(
        env_set=measurement,
        env_version={e: env_states[e]["env_version"] for e in measurement},
        task_spec_hash={e: env_states[e]["task_spec_hash"] for e in measurement},
        grader_hash={e: env_states[e]["grader_hash"] for e in measurement},
        serving_hash={e: env_states[e]["serving_hash"] for e in measurement},
        rule="first",
    )
    miner_ids = sorted({c.miner_artifact_id for c in cells})
    env_ids = sorted(measurement)
    mi = {m: i for i, m in enumerate(miner_ids)}
    ei = {e: i for i, e in enumerate(env_ids)}
    triples = [CellTriple(mi[c.miner_artifact_id], ei[c.env_id], int(c.outcome)) for c in cells]
    snap = archive_fit(miner_ids, env_ids, triples, rho)
    return snap, miner_ids, env_ids


async def _calibration(
    store: Store,
    env_states: dict[str, dict],
    rho: list[float],
    exclude_artifacts: set[str],
) -> CalibrationSnapshot | None:
    """Build a leave-contestants-out calibration snapshot."""
    measurement = _measurement_envs(env_states)
    if not measurement:
        return None
    cells = store.cells_view(
        env_set=measurement,
        env_version={e: env_states[e]["env_version"] for e in measurement},
        task_spec_hash={e: env_states[e]["task_spec_hash"] for e in measurement},
        grader_hash={e: env_states[e]["grader_hash"] for e in measurement},
        serving_hash={e: env_states[e]["serving_hash"] for e in measurement},
        rule="first",
    )
    miner_ids = sorted({c.miner_artifact_id for c in cells})
    env_ids = sorted(measurement)
    mi = {m: i for i, m in enumerate(miner_ids)}
    ei = {e: i for i, e in enumerate(env_ids)}
    triples = [CellTriple(mi[c.miner_artifact_id], ei[c.env_id], int(c.outcome)) for c in cells]
    return calibration_snapshot(miner_ids, env_ids, triples, rho,
                                exclude_artifacts=exclude_artifacts)


async def _run_duel(
    store: Store,
    chain: Chain,
    king: Miner,
    king_slot: Slot,
    challenger: Miner,
    chal_slot: Slot,
    envs: dict[str, tuple],
    cfg: Config,
    duel,
    env_states: dict[str, dict],
    pi: dict[str, float],
    rho: list[float],
    evaluation_version: str,
    stop: asyncio.Event,
) -> _DuelVerdict:
    """One-slot challenger eval against a frozen calibration snapshot.

    Decision statistic is `Δθ = θ_j − θ_i`. SE is conditional on the
    frozen calibration snapshot ψ̂ = (μ̂, β̂, log_â) — see `theta_diff_se`
    docstring for the honest caveat.

    Decision points every `cfg.irt_look_interval_cells` until:
      - dethrone (Δθ lower bound > δ_theta)
      - statistical_hold (Δθ upper bound ≤ δ_theta)
      - budget_hold_inconclusive (cell budget exhausted)
      - challenger_slot_dead (consecutive infra failures)
      - calibration_needed (calibration sufficiency gate not met)
    """
    challenger_art = artifact_id(challenger.model, challenger.revision)
    champion_art = king.artifact_id if hasattr(king, "artifact_id") else artifact_id(king.model, king.revision)
    cal = await _calibration(store, env_states, rho,
                             exclude_artifacts={challenger_art, champion_art})
    if cal is None:
        log.warning("no measurement envs; aborting duel")
        return _DuelVerdict("budget_hold_inconclusive", 0, None, None, None, None)

    cal_hash = cal.fingerprint()

    # Defense-in-depth calibration sufficiency gate. The main loop checks
    # this before provisioning; this re-check guards against drift between
    # check-time and duel-start.
    measurement = _measurement_envs(env_states)
    cells_per_env = _calibration_cells_per_env(store, env_states, measurement,
                                                exclude={challenger_art, champion_art})
    n_calib_miners = len(cal.miner_ids)
    enough_miners = n_calib_miners >= cfg.calibration_min_miners
    enough_cells = all(cells_per_env.get(e, 0) >= cfg.calibration_min_cells_per_env
                       for e in measurement) if measurement else False
    if not (enough_miners and enough_cells):
        log.warning(
            f"calibration_needed (in-duel): miners={n_calib_miners}/"
            f"{cfg.calibration_min_miners}, cells/env={dict(cells_per_env)}"
        )
        _persist_verdict(store, duel.id, "calibration_needed", 0, None, None, None,
                         cal_hash, await _block(chain))
        return _DuelVerdict("calibration_needed", 0, None, None, None, cal_hash)

    z_dethrone = alpha_spent_z(cfg.irt_alpha_dethrone, cfg.irt_look_max)
    z_hold = alpha_spent_z(cfg.irt_beta_hold, cfg.irt_look_max)
    cells_max = cfg.irt_look_max * cfg.irt_look_interval_cells

    # Pull historical contestant cells from the canonical view: champion's
    # θ_i is data-fit against the frozen ψ̂ from accumulated reign cells,
    # not prior-only.
    historical = store.cells_view(
        env_set=measurement,
        env_version={e: env_states[e]["env_version"] for e in measurement},
        task_spec_hash={e: env_states[e]["task_spec_hash"] for e in measurement},
        grader_hash={e: env_states[e]["grader_hash"] for e in measurement},
        serving_hash={e: env_states[e]["serving_hash"] for e in measurement},
        rule="first",
    )
    env_ids_sorted = sorted(measurement)
    ei = {e: i for i, e in enumerate(env_ids_sorted)}
    contestant_triples: list[CellTriple] = []
    for c in historical:
        if c.miner_artifact_id == challenger_art:
            contestant_triples.append(CellTriple(0, ei[c.env_id], int(c.outcome)))
        elif c.miner_artifact_id == champion_art:
            contestant_triples.append(CellTriple(1, ei[c.env_id], int(c.outcome)))

    sampler_policy_hash = hashlib.sha256(
        f"v1:{cfg.irt_K}:{cfg.irt_n_min_per_env}".encode()
    ).hexdigest()[:16]

    n_per_env: dict[str, int] = {e: 0 for e in env_ids_sorted}
    costs: dict[str, EnvCost] = {e: EnvCost() for e in env_ids_sorted}
    cells_collected = 0
    consec_dead = 0
    iter_idx = 0
    delta_theta_last: float | None = None
    se_theta_last: float | None = None
    rating_diff_last: float | None = None
    decision = decision_fit(cal, contestant_triples, challenger_art, champion_art)

    while not stop.is_set() and cells_collected < cells_max:
        env_name = sampler_pick_env(decision, pi, costs, n_per_env,
                                    n_min_per_env=cfg.irt_n_min_per_env)
        wrapper, spec = envs[env_name]
        cell = await _cell_sample(
            chain, chal_slot, challenger, env_name, spec, wrapper, iter_idx,
            miner_artifact_id=challenger_art,
            env_version=env_states[env_name]["env_version"],
            task_spec_hash=env_states[env_name]["task_spec_hash"],
            grader_hash=env_states[env_name]["grader_hash"],
            serving_hash=env_states[env_name]["serving_hash"],
            sampler_policy_hash=sampler_policy_hash,
            collection_context=f"duel:{duel.id}",
        )
        iter_idx += 1
        if cell is None:
            consec_dead += 1
            if consec_dead >= SLOT_DEAD_RUN:
                _persist_verdict(store, duel.id, "challenger_slot_dead",
                                 cells_collected, delta_theta_last, se_theta_last,
                                 rating_diff_last, cal_hash, await _block(chain))
                return _DuelVerdict("challenger_slot_dead", cells_collected,
                                    delta_theta_last, se_theta_last,
                                    rating_diff_last, cal_hash)
            continue
        consec_dead = 0
        store.add_observation(cell)
        contestant_triples.append(CellTriple(0, ei[env_name], int(cell.outcome)))
        n_per_env[env_name] += 1
        costs[env_name] = costs[env_name].update(cell.latency_s)
        cells_collected += 1
        decision = decision_fit(cal, contestant_triples, challenger_art, champion_art)

        if cells_collected % cfg.irt_look_interval_cells != 0:
            continue
        delta_theta, se_theta, r_diff = theta_diff_se(decision, pi)
        delta_theta_last, se_theta_last, rating_diff_last = delta_theta, se_theta, r_diff
        verdict = decide_theta(delta_theta, se_theta,
                               delta_theta_threshold=cfg.delta_theta,
                               z_dethrone=z_dethrone, z_hold=z_hold)
        log.info(f"duel {duel.id} look @ {cells_collected} cells: "
                 f"Δθ={delta_theta:+.4f} SE_θ={se_theta:.4f} "
                 f"R_diff(diag)={r_diff:+.4f} → {verdict}")
        if verdict in ("dethrone", "statistical_hold"):
            _persist_verdict(store, duel.id, verdict, cells_collected,
                             delta_theta, se_theta, r_diff, cal_hash, await _block(chain))
            return _DuelVerdict(verdict, cells_collected, delta_theta,
                                se_theta, r_diff, cal_hash)

    if stop.is_set():
        _persist_verdict(store, duel.id, "cancelled", cells_collected,
                         delta_theta_last, se_theta_last, rating_diff_last,
                         cal_hash, None)
        return _DuelVerdict("cancelled", cells_collected, delta_theta_last,
                            se_theta_last, rating_diff_last, cal_hash)
    _persist_verdict(store, duel.id, "budget_hold_inconclusive", cells_collected,
                     delta_theta_last, se_theta_last, rating_diff_last,
                     cal_hash, await _block(chain))
    return _DuelVerdict("budget_hold_inconclusive", cells_collected,
                        delta_theta_last, se_theta_last, rating_diff_last,
                        cal_hash)


def _calibration_sufficient(
    cfg: Config,
    store: Store,
    env_states: dict[str, dict],
    *,
    exclude: set[str],
) -> tuple[bool, dict]:
    """Check whether the calibration cohort (cells excluding the contestants)
    has enough miners and per-env cells to support a statistical decision.

    Hoisted out of `_run_duel` so the main loop can check *before* provisioning
    a challenger slot. A failing check should NOT reprovision the same
    challenger immediately; the loop sleeps and re-checks once calibration
    grows (currently only via future backfill — placeholder).

    Returns `(ok, diagnostics)` where diagnostics is a dict for logging.
    """
    measurement = _measurement_envs(env_states)
    if not measurement:
        return False, {"reason": "no measurement envs"}
    cells_per_env = _calibration_cells_per_env(store, env_states, measurement,
                                                exclude=exclude)
    # Distinct miners in the calibration cohort
    cells = store.cells_view(
        env_set=measurement,
        env_version={e: env_states[e]["env_version"] for e in measurement},
        task_spec_hash={e: env_states[e]["task_spec_hash"] for e in measurement},
        grader_hash={e: env_states[e]["grader_hash"] for e in measurement},
        serving_hash={e: env_states[e]["serving_hash"] for e in measurement},
        rule="first",
        exclude_artifacts=exclude,
    )
    n_miners = len({c.miner_artifact_id for c in cells})
    enough_miners = n_miners >= cfg.calibration_min_miners
    enough_cells = all(cells_per_env.get(e, 0) >= cfg.calibration_min_cells_per_env
                       for e in measurement)
    diagnostics = {
        "n_miners": n_miners,
        "min_miners": cfg.calibration_min_miners,
        "cells_per_env": dict(cells_per_env),
        "min_cells_per_env": cfg.calibration_min_cells_per_env,
    }
    return (enough_miners and enough_cells), diagnostics


def _calibration_cells_per_env(
    store: Store,
    env_states: dict[str, dict],
    measurement: set[str],
    exclude: set[str],
) -> dict[str, int]:
    """Count cells per env in the canonical view, excluding the contestants.
    Used by the calibration sufficiency gate."""
    if not measurement:
        return {}
    cells = store.cells_view(
        env_set=measurement,
        env_version={e: env_states[e]["env_version"] for e in measurement},
        task_spec_hash={e: env_states[e]["task_spec_hash"] for e in measurement},
        grader_hash={e: env_states[e]["grader_hash"] for e in measurement},
        serving_hash={e: env_states[e]["serving_hash"] for e in measurement},
        rule="first",
        exclude_artifacts=exclude,
    )
    counts: dict[str, int] = {e: 0 for e in measurement}
    for c in cells:
        counts[c.env_id] = counts.get(c.env_id, 0) + 1
    return counts


async def _block(chain: Chain) -> int:
    try:
        return await chain.current_block()
    except Exception as ex:
        log.warning(f"current_block failed: {ex}; using 0")
        return 0


def _persist_verdict(
    store: Store,
    duel_id: int,
    status: str,
    cells_collected: int,
    delta_theta: float | None,
    se_theta: float | None,
    rating_diff_diagnostic: float | None,
    calibration_snapshot_hash: str | None,
    finished_block: int | None,
) -> None:
    store.finish_duel(duel_id, status, cells_collected, delta_theta, se_theta,
                      rating_diff_diagnostic, calibration_snapshot_hash,
                      finished_block)


# ---------------------------------------------------------------------------
# Champion preflight (D14)
# ---------------------------------------------------------------------------

async def _champion_preflight(
    store: Store,
    chain: Chain,
    king: Miner,
    king_slot: Slot,
    envs: dict[str, tuple],
    cfg: Config,
    env_states: dict[str, dict],
    pi: dict[str, float],
    rho: list[float],
    stop: asyncio.Event,
    *,
    max_cells: int = 200,
) -> bool:
    """Ensure √Var(θ_i) ≤ cfg.champion_se_theta_max before accepting a
    challenger. Runs champion cells against the king slot until the gate
    clears or `max_cells` exhausted. Returns True if SE_θ cleared.
    """
    champion_art = artifact_id(king.model, king.revision)
    iter_idx = int(time.time())
    sampler_policy_hash = hashlib.sha256(b"preflight").hexdigest()[:16]
    cells_added = 0
    # Cold-start bypass: if there are no other miners in the calibration set,
    # nuisance is prior-only and SE_R cannot shrink below the prior's reach
    # regardless of how many champion cells we sample. Skip the gate in that
    # case — the duel itself absorbs the uncertainty via SE_R.
    initial_cal = await _calibration(store, env_states, rho,
                                     exclude_artifacts={champion_art})
    if initial_cal is None or initial_cal.n_calibration_cells == 0:
        log.info("preflight bypassed: calibration has no other miners")
        return True
    while not stop.is_set() and cells_added < max_cells:
        cal = await _calibration(store, env_states, rho,
                                 exclude_artifacts={champion_art})
        if cal is None:
            return False
        # θ_i from current canonical cells of the champion against frozen nuisance
        measurement = _measurement_envs(env_states)
        cells = store.cells_view(
            env_set=measurement,
            env_version={e: env_states[e]["env_version"] for e in measurement},
            task_spec_hash={e: env_states[e]["task_spec_hash"] for e in measurement},
            grader_hash={e: env_states[e]["grader_hash"] for e in measurement},
            serving_hash={e: env_states[e]["serving_hash"] for e in measurement},
            rule="first",
        )
        env_ids_sorted = sorted(measurement)
        ei = {e: i for i, e in enumerate(env_ids_sorted)}
        triples = [CellTriple(0, ei[c.env_id], int(c.outcome))
                   for c in cells if c.miner_artifact_id == champion_art]
        dec = decision_fit(cal, triples, champion_art, champion_art)
        # Under K=0 θ-contrast: gate on √Var(θ_i) directly. Active-block [0,0]
        # is Var(θ_j) in our parameterization where the king is fit as
        # miner_idx=0 in the single-contestant decision_fit.
        se = champion_se_theta(cov_theta_i=float(dec.cov_active[0, 0]))
        if se <= cfg.champion_se_theta_max:
            log.info(f"preflight clear: SE_θ(king)={se:.4f} ≤ {cfg.champion_se_theta_max}")
            return True
        # Sample a champion cell on the most informative env
        n_per_env = {e: 0 for e in env_ids_sorted}
        costs = {e: EnvCost() for e in env_ids_sorted}
        env_name = sampler_pick_env(dec, pi, costs, n_per_env,
                                    n_min_per_env=cfg.irt_n_min_per_env)
        wrapper, spec = envs[env_name]
        cell = await _cell_sample(
            chain, king_slot, king, env_name, spec, wrapper, iter_idx + cells_added,
            miner_artifact_id=champion_art,
            env_version=env_states[env_name]["env_version"],
            task_spec_hash=env_states[env_name]["task_spec_hash"],
            grader_hash=env_states[env_name]["grader_hash"],
            serving_hash=env_states[env_name]["serving_hash"],
            sampler_policy_hash=sampler_policy_hash,
            collection_context="preflight",
        )
        if cell is None:
            log.warning("preflight: champion infra failure")
            return False
        store.add_observation(cell)
        cells_added += 1
    log.warning(f"preflight gave up after {cells_added} cells; SE_θ still > {cfg.champion_se_theta_max}")
    return False


# ---------------------------------------------------------------------------
# Retirement / GC (preserved)
# ---------------------------------------------------------------------------

async def _gc_retiring(store: Store, s3_configs: list[S3Config]) -> None:
    if not s3_configs:
        return
    by_name = {c.name: c for c in s3_configs}
    for old in store.retiring_backups():
        if await asyncio.to_thread(delete_refs, old.manifest_key, by_name):
            store.mark_backup_deleted(old.manifest_key)


async def _retirement_task(store: Store, s3_configs: list[S3Config], stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await _gc_retiring(store, s3_configs)
        except Exception as e:
            log.warning(f"retirement sweep error: {e}")
        try:
            await _cancellable(asyncio.sleep(BACKUP_RETIREMENT_GRACE_S), stop)
        except asyncio.CancelledError:
            return


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run(cfg: Config, chain: Chain, slots: list[VllmSlots] | None = None):
    if slots is None:
        s3_configs = _backup_configs(cfg, chain.hotkey)
        slots = make_slots(cfg=cfg, hotkey=chain.hotkey, s3_configs=s3_configs)
    else:
        s3_configs = []
    await asyncio.gather(*(s.reconcile() for s in slots), return_exceptions=True)
    envs = await _load_envs(cfg)
    store = Store(cfg.db_path)
    store.abort_running_duels()
    env_states = _ensure_env_states(store, cfg)
    pi = _build_pi(cfg, env_states)
    rho = _build_rho(cfg, sorted(_measurement_envs(env_states)))
    eval_version = _evaluation_version(env_states, pi, cfg)
    state = _RunState()
    state.attempted = store.attempted_artifacts(eval_version)
    log.info(f"evaluation_version={eval_version}; loaded {len(state.attempted)} attempted artifacts")
    king_provision_fails = 0
    stop = asyncio.Event()
    _install_signal_handlers(stop)
    retirement = asyncio.create_task(_retirement_task(store, s3_configs, stop))
    try:
        champ = store.champion()
        if champ is None:
            champ, state.king_slot = await _bootstrap_champion(store, s3_configs, slots, chain, cfg, stop)
        while not stop.is_set():
            champ = store.champion()
            if champ is None:
                champ, state.king_slot = await _bootstrap_champion(store, s3_configs, slots, chain, cfg, stop)
            all_miners = await chain.list_miners()

            registration = await _champion_registration_status(s3_configs, all_miners, champ, missing_ref_alive=True)
            champion_artifact_alive = registration == "alive"
            if champ.payable and registration == "unknown":
                log.warning("champion artifact identity unknown; pausing publication and duels")
                await _cancellable(asyncio.sleep(120), stop)
                continue
            if champ.payable and registration == "dead":
                log.warning(f"champion uid={champ.uid} artifact changed or deregistered; burning weights")
                store.demote_champion(champ.artifact_id)
                await _publish_burn(store, chain, cfg, champ.artifact_id)
            else:
                await _publish_champion(store, chain, cfg, champ)
            champ = store.champion() or champ
            if not champ.payable:
                champion_artifact_alive = False

            king_slot = await _ensure_king_slot(state, champ, slots, stop, store)
            if king_slot is None:
                king_provision_fails += 1
                if king_provision_fails >= KING_REPROVISION_LIMIT:
                    raise RuntimeError(f"champion reprovision failed {KING_REPROVISION_LIMIT} consecutive times; aborting")
                await _cancellable(asyncio.sleep(120), stop)
                continue
            king_provision_fails = 0

            await _reconcile_backup_manifest(store, champ, king_slot, s3_configs)
            champ = store.champion() or champ

            # Cummax queue: filter by miner_artifact_id (full content hash)
            # so artifact-level changes (tokenizer, decoding config, etc.)
            # produce a fresh entry. (model, revision) is too-coarse identity.
            def _attempted(m: Miner) -> bool:
                return artifact_id(m.model, m.revision) in state.attempted

            queue = sorted(
                (m for m in all_miners
                 if m.model not in cfg.model_skiplist
                 and not _is_current_champion_registration(m, champ, champion_artifact_alive)
                 and not _attempted(m)),
                key=lambda m: (m.block, _tiebreak(m)),
            )
            if not queue:
                log.info("queue exhausted this reign; sleeping 120s before re-probe")
                await _cancellable(asyncio.sleep(120), stop)
                continue

            registered = queue[0]

            def _attempt(model: str, revision: str,
                         archive_snapshot_hash: str | None = None) -> None:
                aid = artifact_id(model, revision)
                state.attempted.add(aid)
                store.mark_attempted(aid, eval_version, model=model, revision=revision,
                                     archive_snapshot_hash=archive_snapshot_hash)

            chal_slot: Slot | None = None
            promoted = False
            try:
                try:
                    challenger = await _pin_artifact(s3_configs, registered)
                except Exception as exc:
                    log.warning(f"challenger pin failed uid{registered.uid}: {exc}")
                    _attempt(registered.model, registered.revision)
                    continue

                # Calibration sufficiency check BEFORE provisioning. If the
                # cohort can't support a statistical decision yet, do not
                # provision the challenger slot — that would tight-loop
                # 5–15min provisions on every queue cycle. Sleep on a longer
                # backoff and don't mark attempted (challenger remains
                # eligible once calibration grows).
                challenger_art_now = artifact_id(challenger.model, challenger.revision)
                champion_art_now = champ.artifact_id
                ok_cal, cal_diag = _calibration_sufficient(
                    cfg, store, env_states,
                    exclude={challenger_art_now, champion_art_now},
                )
                if not ok_cal:
                    log.warning(
                        f"calibration_needed (pre-provision): {cal_diag}; "
                        f"sleeping 300s without provisioning. challenger "
                        f"uid{registered.uid} {challenger.model}@{challenger.revision} "
                        f"remains eligible (not marked attempted)."
                    )
                    await _cancellable(asyncio.sleep(300), stop)
                    continue

                # Champion preflight
                king_miner = _champion_miner(champ)
                king_miner = Miner(uid=king_miner.uid, hotkey=king_miner.hotkey,
                                   model=king_miner.model, revision=king_miner.revision,
                                   block=king_miner.block)
                ok = await _champion_preflight(store, chain, king_miner, king_slot,
                                               envs, cfg, env_states, pi, rho, stop)
                if not ok:
                    log.warning("champion preflight failed; skipping challenger this pass")
                    await _cancellable(asyncio.sleep(60), stop)
                    continue

                chal_slot = await _provision(slots, challenger, stop, source="hf")
                if chal_slot is None:
                    _attempt(registered.model, registered.revision)
                    continue

                block_no = await chain.current_block()
                duel = store.create_duel(
                    champion=champ,
                    challenger_uid=registered.uid,
                    challenger_hotkey=registered.hotkey,
                    challenger_model=challenger.model,
                    challenger_revision=challenger.revision,
                    schedule_seed=secrets.token_hex(16),
                    alpha=cfg.irt_alpha_dethrone,
                    delta_theta=cfg.delta_theta,
                    pi=pi,
                    evaluation_version=eval_version,
                    started_block=block_no,
                )
                log.info(f"duel: champion {champ.model}@{champ.revision} vs uid{registered.uid} "
                         f"{challenger.model}@{challenger.revision} "
                         f"δ_θ={cfg.delta_theta:.3g} α={cfg.irt_alpha_dethrone:.3g} "
                         f"β={cfg.irt_beta_hold:.3g}")
                # Build the king miner with artifact_id attribute for the duel function.
                _Krecord = type("KingMiner", (), {})
                k = _Krecord()
                k.uid = champ.uid; k.hotkey = champ.hotkey or ""
                k.model = champ.model; k.revision = champ.revision; k.block = champ.reign_start
                k.artifact_id = champ.artifact_id
                verdict = await _run_duel(store, chain, k, king_slot, challenger,
                                          chal_slot, envs, cfg, duel, env_states,
                                          pi, rho, eval_version, stop)
                if verdict.status == "challenger_slot_dead":
                    _attempt(registered.model, registered.revision,
                             archive_snapshot_hash=verdict.calibration_snapshot_hash)
                    continue
                if verdict.status == "calibration_needed":
                    # Defense-in-depth: should be caught pre-provision above,
                    # but if it slips through, sleep on a long backoff so we
                    # don't reprovision the same challenger every minute.
                    log.warning("calibration_needed (post-provision): not enough cohort "
                                "evidence to decide; sleeping 300s, challenger remains "
                                "un-attempted")
                    await _cancellable(asyncio.sleep(300), stop)
                    continue
                if verdict.status != "dethrone":
                    _attempt(registered.model, registered.revision,
                             archive_snapshot_hash=verdict.calibration_snapshot_hash)
                    log.info(f"verdict: champion holds ({verdict.status}) "
                             f"Δθ={verdict.delta_theta} R_diff={verdict.rating_diff_diagnostic}")
                    continue

                fresh = await chain.list_miners()
                if not await _registered_artifact_alive(s3_configs, fresh, registered, challenger.revision):
                    log.warning(f"verdict skipped: challenger uid{registered.uid}@{registered.revision} identity changed")
                    _attempt(registered.model, registered.revision)
                    continue

                snapshot = await poll_backup(chal_slot) if s3_configs else None
                refs = (snapshot or {}).get("refs") or []
                fresh2 = await chain.list_miners()
                if not await _registered_artifact_alive(s3_configs, fresh2, registered, challenger.revision):
                    log.warning(f"verdict skipped post-poll: challenger uid{registered.uid}@{registered.revision} identity changed")
                    _attempt(registered.model, registered.revision)
                    continue
                new_art = artifact_id(challenger.model, challenger.revision)
                new_champ = Champion(
                    artifact_id=new_art,
                    model=challenger.model,
                    revision=challenger.revision,
                    uid=registered.uid,
                    hotkey=registered.hotkey,
                    reign_start=await chain.current_block(),
                    payable=True,
                )
                backup = BackupRecord(
                    artifact_id=new_art,
                    model=new_champ.model,
                    revision=new_champ.revision,
                    manifest_key=encode_refs(_refs_to_objs(refs)),
                    status="current",
                ) if refs else None
                # Cummax #2: dethroned king joins _attempted with the
                # calibration snapshot hash that produced the decision.
                _attempt(champ.model, champ.revision,
                         archive_snapshot_hash=verdict.calibration_snapshot_hash)
                store.set_champion(new_champ, backup=backup)
                old_slot, state.king_slot = state.king_slot, chal_slot
                chal_slot = None
                promoted = True
                # Cummax #1: do NOT clear _attempted on dethrone.
                if old_slot is not None:
                    await _safe_teardown(old_slot, "old-champion-promoted")
                ok = await _publish_champion(store, chain, cfg, new_champ)
                log.info(f"DETHRONE: {champ.model}@{champ.revision} -> uid {registered.uid}"
                         + ("" if ok else " (publish deferred)"))
            finally:
                if chal_slot is not None and not promoted:
                    await _safe_teardown(chal_slot, "challenger-end")
    finally:
        log.info("shutdown")
        retirement.cancel()
        try: await retirement
        except (asyncio.CancelledError, Exception): pass
        if state.king_slot is not None:
            await _safe_teardown(state.king_slot, "shutdown-king")
        for s in slots:
            try: await s.aclose()
            except Exception as e: log.warning(f"aclose {s.NAME}: {e}")
        store.close()


# ---------------------------------------------------------------------------
# Chain wiring (preserved)
# ---------------------------------------------------------------------------

def bittensor_chain(cfg: Config) -> tuple[Chain, Subtensor]:
    sub = Subtensor(cfg.subtensor_endpoint, cfg.subtensor_fallback)
    wallet = bt.Wallet(name=cfg.wallet_name, hotkey=cfg.hotkey_name)
    hotkey = wallet.hotkey.ss58_address

    async def publish(uid: int, expected_hotkey: str) -> bool:
        return await set_weights(sub, wallet, cfg.netuid, uid, expected_hotkey)

    async def burn() -> bool:
        return await clear_weights(sub, wallet, cfg.netuid)

    async def uid_matches(uid: int, expected_hotkey: str) -> bool:
        meta = await sub.metagraph(cfg.netuid)
        return 0 <= uid < len(meta.hotkeys) and meta.hotkeys[uid] == expected_hotkey

    return Chain(
        hotkey=hotkey,
        list_miners=lambda: get_miners(sub, cfg.netuid, hotkey),
        current_block=sub.get_current_block,
        publish_winner=publish,
        burn_weights=burn,
        uid_matches_hotkey=uid_matches,
    ), sub


def static_chain(miners: list[Miner], hotkey: str = "local-validator") -> Chain:
    async def _miners(): return miners
    async def _block(): return 0
    async def _publish(uid: int, expected_hotkey: str) -> bool:
        log.info(f"local winner: uid {uid} (hk {expected_hotkey[:8]})"); return True
    async def _burn() -> bool:
        log.info("local winner: none"); return True
    async def _uid_matches(uid: int, expected_hotkey: str) -> bool:
        return any(m.uid == uid and m.hotkey == expected_hotkey for m in miners)
    return Chain(hotkey=hotkey, list_miners=_miners, current_block=_block,
                 publish_winner=_publish, burn_weights=_burn,
                 uid_matches_hotkey=_uid_matches)
