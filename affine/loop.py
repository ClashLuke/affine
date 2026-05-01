"""Colocated-backup king-of-the-hill validator loop.

The validator is no longer a data plane. It is a control plane: each loop
iteration reconciles four pieces of state (chain weights, king slot liveness,
backup manifest durability, challenger duel) against the champion in SQLite.
Backups are produced by the slot's sidecar, never by the validator process.

There is no pending_champion table, no _finalize_pending, no pre-promotion
restore-verify. The dethrone path is: commit champion in DB → swap king slot
pointer → broadcast set_weights. A crash anywhere in that sequence is
recoverable by the next reconcile pass.
"""

from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import os
import secrets
import signal
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import bittensor as bt
import httpx
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

from .backup import (
    BackupManager,
    ManifestRef,
    S3Config,
    encode_refs,
    refs_digest,
)
from .chain import Miner, Subtensor, _tiebreak, _truthy_env, clear_weights, get_miners, set_weights
from .config import BASELINE_MODELS, Config
from .envs import EnvFactory
from .paired import PairCounts, PairDecision, alpha_for_reign, decide_paired, pair_p_value
from .sampler import run_one
from .store import Champion, PairSample, Store, artifact_id
from .vllm import (
    Slot,
    SlotProvisionFailed,
    TargonSlots,
    poll_backup,
    setup_backup,
    teardown_backup,
)

log = logging.getLogger(__name__)

ArtKey = tuple[str, str]
SLOT_DEAD = 30
KING_REPROVISION_LIMIT = 10
BACKUP_RETIREMENT_GRACE_S = 600


@dataclass
class Chain:
    """Everything the loop needs from the world besides slots, envs, and evidence."""
    hotkey: str
    list_miners: Callable[[], Awaitable[list[Miner]]]
    current_block: Callable[[], Awaitable[int]]
    publish_winner: Callable[[int, str], Awaitable[bool]]
    burn_weights: Callable[[], Awaitable[bool]]
    uid_matches_hotkey: Callable[[int, str], Awaitable[bool]]


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
    """Await `coro` unless `stop` fires, in which case cancel and raise CancelledError.

    Used to make long-running provisions interruptible by SIGTERM/SIGINT without
    threading the stop event through every slot backend. Worker is cancelled and
    awaited in the finally so cleanup (provision's BaseException handler) runs
    even when our outer task is cancelled.

    `on_orphan(result)`: optional async cleanup for the case where the worker
    produced a result that the caller never received (outer cancel raced with
    worker completion).
    """
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


async def _safe_teardown(slots, slot: Slot, ctx: str) -> None:
    """Shielded so an outer cancel during teardown doesn't interrupt the Targon
    DELETE — the request keeps flying as a fire-and-forget task; reconcile() at
    next startup is the backstop."""
    try: await asyncio.shield(slots.teardown(slot))
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


async def _provision(slots, miner: Miner, stop: asyncio.Event, **kwargs) -> tuple[Slot | None, str]:
    """Returns (slot|None, status). `crashloop` is the only miner-fault signal."""
    try:
        slot = await _cancellable(
            slots.provision(miner.model, miner.revision, **kwargs), stop,
            on_orphan=lambda s: _safe_teardown(slots, s, "provision-orphan"),
        )
    except SlotProvisionFailed as e:
        log.warning(f"provision crashloop {miner.model}@{miner.revision}: {e}")
        return None, "crashloop"
    except TimeoutError as e:
        log.warning(f"provision timeout uid {miner.uid} {miner.model}@{miner.revision}: {e}")
        return None, "timeout"
    except asyncio.CancelledError:
        raise
    except httpx.HTTPError as e:
        log.warning(f"provision transient uid {miner.uid}: {type(e).__name__}: {e}")
        return None, "transient"
    except Exception as e:
        log.error(f"provision error uid {miner.uid}: {e}")
        return None, "error"
    return slot, "ok"


def _seed(uid: int, rev: str, env: str, c: int, salt: str = "") -> int:
    h = hashlib.sha256(f"{salt}\0{uid}\0{rev}\0{env}\0{c}".encode()).digest()
    return int.from_bytes(h[:4], "big") & ((1 << 31) - 1)


def _task_id(king_uid: int, chal_uid: int, env: str, iter_idx: int,
             lo: int, hi: int, salt: str = "") -> int:
    a, b = sorted((king_uid, chal_uid))
    h = hashlib.sha256(f"{salt}\0{a}\0{b}\0{env}\0{iter_idx}".encode()).digest()
    n = hi - lo + 1
    return lo + (int.from_bytes(h[:8], "big") % n)


async def _sample(chain: Chain, wrapper, params: dict, timeout: float, slot: Slot,
                  miner: Miner, env_name: str, c: int,
                  task_id: int) -> tuple[int, float, bool, int]:
    outcome, latency, tokens = await run_one(
        wrapper, params, timeout, slot,
        seed=_seed(miner.uid if miner.uid is not None else -1, miner.revision, env_name, c, salt=chain.hotkey),
        task_id=task_id)
    if outcome is None:
        return 0, 0.0, False, 0
    return int(bool(outcome)), float(latency), True, int(tokens)


async def _pair_sample(chain: Chain, wrapper, params: dict, timeout: float,
                       king_slot: Slot, king: Miner, ck: int,
                       chal_slot: Slot, challenger: Miner, cc: int,
                       env_name: str, task_id: int, e_idx: int):
    rk, rc = await asyncio.gather(
        _sample(chain, wrapper, params, timeout, king_slot, king, env_name, ck, task_id),
        _sample(chain, wrapper, params, timeout, chal_slot, challenger, env_name, cc, task_id),
        return_exceptions=True,
    )
    return e_idx, env_name, rk, rc


@dataclass
class _RunState:
    king_slot: Slot | None = None
    attempted: set[ArtKey] = field(default_factory=set)


@dataclass(frozen=True)
class _DuelRun:
    decision: PairDecision | None
    status: str
    counts: PairCounts


def _make_backup_manager(cfg: Config, chain: Chain, slots) -> tuple[BackupManager | None, list[S3Config]]:
    if slots is not None or _truthy_env("AFFINE_LOCAL"):
        return None, []
    if ((os.getenv("HIPPIUS_S3_ACCESS_KEY") and os.getenv("HIPPIUS_S3_SECRET_KEY"))
            or (os.getenv("R2_S3_ACCESS_KEY_ID") and os.getenv("R2_S3_SECRET_ACCESS_KEY"))):
        providers = S3Config.from_envs(hotkey=chain.hotkey, netuid=cfg.netuid)
        return BackupManager(providers), providers
    raise RuntimeError("Hippius S3 credentials are required for production runs")


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
    if champ.uid is not None and m.uid == champ.uid and m.hotkey == champ.hotkey:
        return (m.model, m.revision) == (champ.model, champ.revision) or (artifact_alive and m.model == champ.model)
    return (m.model, m.revision) == (champ.model, champ.revision)


def _derive_prefix(s3_configs: list[S3Config], art: str) -> str:
    """Per-artifact prefix root. The slot adds `{provider_name}-{ts}` inside,
    so concurrent attempts on the same artifact don't collide."""
    cfg = s3_configs[0]
    return f"{cfg.prefix}/artifacts/{art}"


async def _bootstrap_champion(
    store: Store,
    s3_configs: list[S3Config],
    slots,
    chain: Chain,
    cfg: Config,
    stop: asyncio.Event,
) -> tuple[Champion, Slot]:
    """Bootstrap a fresh champion. The slot's sidecar takes over backup
    durability; this function only commits the in-DB champion and provisions
    the king slot. The first reconcile pass calls setup_backup."""
    baseline_model = os.getenv("AFFINE_BASELINE_MODEL", BASELINE_MODELS[0]).strip() or BASELINE_MODELS[0]
    baseline_revision = os.getenv("AFFINE_BASELINE_REVISION", "").strip()
    miners = await chain.list_miners()
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
        backup_manifest="",
        backup_prefix="",
        payable=registered is not None,
    )
    slot, status = await _provision(slots, _champion_miner(champ), stop, source="hf")
    if slot is None:
        raise RuntimeError(f"baseline provision failed: {status}")
    store.set_champion(champ)
    kind = f"uid{registered.uid}" if registered else "unregistered"
    log.info(f"bootstrap champion: {kind} baseline {model}@{revision}")
    return champ, slot


async def _ensure_king_slot(
    state: _RunState,
    champ: Champion,
    slots,
    stop: asyncio.Event,
) -> Slot | None:
    if state.king_slot is not None:
        return state.king_slot
    miner = _champion_miner(champ)
    slot, status = await _provision(slots, miner, stop, source="hf")
    if slot is None and champ.backup_manifest:
        log.warning(f"champion HF reprovision failed ({status}); using backup")
        slot, status = await _provision(
            slots, miner, stop, source="s3", backup_manifest_key=champ.backup_manifest,
        )
    if slot is None:
        log.error(f"champion reprovision failed: {status}")
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


async def _reconcile_backup_manifest(
    store: Store,
    champ: Champion,
    slot: Slot,
    s3_configs: list[S3Config],
) -> None:
    """If the champion has no manifest yet, drive the slot's sidecar to produce
    one. Idempotent: if upload is already in flight this is a poll; if not yet
    started this initiates it; if done this records refs onto the champion."""
    if champ.backup_manifest or not slot.sidecar_url or not s3_configs:
        return
    state = await poll_backup(slot)
    if state is None:
        return
    if state.get("state") == "idle":
        prefix = _derive_prefix(s3_configs, champ.artifact_id)
        await setup_backup(slot, s3_configs, prefix=prefix, model=champ.model,
                           revision=champ.revision, artifact_id=champ.artifact_id)
        return
    if state.get("state") == "done" and state.get("refs"):
        _persist_refs(store, champ, state["refs"])


def _persist_refs(store: Store, champ: Champion, refs: list[dict]) -> None:
    objs = [ManifestRef(
        provider=str(r["provider"]),
        bucket=str(r["bucket"]),
        key=str(r["key"]),
        prefix=str(r["prefix"]),
        sha256=str(r.get("sha256", "")),
    ) for r in refs]
    encoded = encode_refs(objs)
    store.update_backup_manifest(
        artifact_id=champ.artifact_id,
        manifest_key=encoded,
        prefix=encoded,
        manifest_sha256=refs_digest(objs),
        model=champ.model,
        revision=champ.revision,
    )


def _pair_sample_from_rows(duel_id: int, env: str, task_id: int, iter_idx: int,
                           block: int, king: Miner, chal: Miner, rk, rc) -> PairSample:
    def unpack(result, miner: Miner) -> tuple[int, float, int, int]:
        if isinstance(result, BaseException):
            log.warning(f"sample raised uid{miner.uid}: {type(result).__name__}: {result}")
            return 0, 0.0, 0, 0
        p, l, delivered, tokens = result
        return p, l, int(delivered), tokens
    kp, kl, kd, kt = unpack(rk, king)
    cp, cl, cd, ct = unpack(rc, chal)
    return PairSample(duel_id, env, task_id, iter_idx, block, kp, cp, kl, cl, kd, cd, kt, ct)


async def _run_fixed_duel(
    store: Store,
    chain: Chain,
    king: Miner,
    king_slot: Slot,
    challenger: Miner,
    chal_slot: Slot,
    envs: dict[str, tuple],
    cfg: Config,
    duel,
    stop: asyncio.Event,
) -> _DuelRun:
    env_names = [spec.name for spec in cfg.environments]
    target_per_env = duel.pairs_per_env
    target_total = target_per_env * len(env_names)
    delivered_by_env = {env: 0 for env in env_names}
    inflight_by_env = {env: 0 for env in env_names}
    launched_by_env = {env: 0 for env in env_names}
    env_rank = {env: i for i, env in enumerate(env_names)}
    counters: dict[tuple[int, str, str], int] = {}
    inflight: set[asyncio.Task] = set()
    counts = PairCounts()
    king_dead = chal_dead = 0
    pair_dead = 0

    def alloc(miner: Miner, env: str) -> int:
        key = (miner.uid, miner.revision, env)
        c = counters.get(key, 0)
        counters[key] = c + 1
        return c

    async def drain(reason: str) -> None:
        for task in inflight:
            if not task.done():
                task.cancel()
        for task in list(inflight):
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                pass
            except BaseException as exc:
                log.warning(f"duel drain ({reason}) interrupted: {type(exc).__name__}: {exc}")
        inflight.clear()

    async def block() -> int:
        try:
            return await chain.current_block()
        except Exception as ex:
            log.warning(f"current_block failed during duel: {ex}; using 0")
            return 0

    async def _abort_duel(reason: str, status: str) -> _DuelRun:
        await drain(reason)
        store.finish_duel(duel.id, status, counts, pair_p_value(counts.challenger_only, counts.discordant), await block())
        return _DuelRun(None, status, counts)

    def next_env() -> str | None:
        eligible = [
            env for env in env_names
            if delivered_by_env[env] + inflight_by_env[env] < target_per_env
        ]
        if not eligible:
            return None
        return min(eligible, key=lambda env: (delivered_by_env[env] + inflight_by_env[env], env_rank[env]))

    async def launch(env_name: str) -> asyncio.Task:
        iter_idx = launched_by_env[env_name]
        wrapper, spec = envs[env_name]
        params = {k: v for k, v in spec.params.items() if k != "timeout"}
        timeout = float(spec.params.get("timeout", 600))
        lo, hi = spec.task_range
        task_id = _task_id(
            king.uid if king.uid is not None else -1, challenger.uid if challenger.uid is not None else -1,
            env_name, iter_idx, lo, hi,
            salt=f"{chain.hotkey}:{duel.schedule_seed}")
        task = asyncio.create_task(_pair_sample(
            chain, wrapper, params, timeout,
            king_slot, king, alloc(king, env_name),
            chal_slot, challenger, alloc(challenger, env_name),
            env_name, task_id, iter_idx))
        launched_by_env[env_name] = iter_idx + 1
        inflight_by_env[env_name] += 1
        return task

    async def fill() -> None:
        while len(inflight) < cfg.dwell_batch:
            env_name = next_env()
            if env_name is None:
                return
            inflight.add(await launch(env_name))

    await fill()

    while inflight:
        if stop.is_set():
            await drain("stop")
            store.finish_duel(duel.id, "cancelled", counts, pair_p_value(counts.challenger_only, counts.discordant), None)
            return _DuelRun(None, "cancelled", counts)
        try:
            done, _ = await _cancellable(asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED), stop)
        except asyncio.CancelledError:
            await drain("stop")
            store.finish_duel(duel.id, "cancelled", counts, pair_p_value(counts.challenger_only, counts.discordant), None)
            if stop.is_set():
                return _DuelRun(None, "cancelled", counts)
            raise
        samples = []
        for task in done:
            inflight.discard(task)
            e_idx, env_name, rk, rc = task.result()
            inflight_by_env[env_name] -= 1
            task_id = _task_id(
                king.uid if king.uid is not None else -1, challenger.uid if challenger.uid is not None else -1,
                env_name, e_idx,
                envs[env_name][1].task_range[0], envs[env_name][1].task_range[1],
                salt=f"{chain.hotkey}:{duel.schedule_seed}")
            b = await block()
            sample = _pair_sample_from_rows(duel.id, env_name, task_id, e_idx, b, king, challenger, rk, rc)
            samples.append(sample)
            if sample.champion_delivered and sample.challenger_delivered:
                counts = counts.add(sample.champion_pass, sample.challenger_pass)
                delivered_by_env[env_name] += 1
                pair_dead = 0
            else:
                pair_dead += 1
            king_dead = 0 if sample.champion_delivered else king_dead + 1
            chal_dead = 0 if sample.challenger_delivered else chal_dead + 1
        store.add_samples(samples)
        if (king_dead >= SLOT_DEAD and chal_dead >= SLOT_DEAD) or pair_dead >= SLOT_DEAD:
            return await _abort_duel("delivery-stalled", "delivery_stalled")
        if king_dead >= SLOT_DEAD:
            return await _abort_duel("champion-slot-dead", "champion_slot_dead")
        if chal_dead >= SLOT_DEAD:
            return await _abort_duel("challenger-slot-dead", "challenger_slot_dead")
        remaining = target_total - counts.total
        best_possible = PairCounts(
            counts.challenger_only + remaining,
            counts.champion_only,
            counts.both_pass,
            counts.both_fail,
        )
        if not decide_paired(best_possible, alpha=duel.alpha,
                             min_discordant=duel.min_discordant).dethrone:
            await drain("best-possible-hold")
            break
        await fill()

    decision = decide_paired(counts, alpha=duel.alpha, min_discordant=duel.min_discordant)
    status = "dethrone" if decision.dethrone else "hold"
    store.finish_duel(duel.id, status, counts, decision.p_value, await block())
    return _DuelRun(decision, status, counts)


async def _gc_retiring(store: Store, backup: BackupManager | None) -> None:
    if backup is None:
        return
    for old in store.retiring_backups():
        if await asyncio.to_thread(backup.delete_prefix, old.prefix):
            store.mark_backup_deleted(old.manifest_key)


async def _gc_staging(store: Store, backup: BackupManager | None) -> None:
    if backup is None:
        return
    for old in store.staging_backups():
        if await asyncio.to_thread(backup.delete_prefix, old.prefix):
            store.mark_backup_deleted(old.manifest_key)


async def _retirement_task(store: Store, backup: BackupManager | None, stop: asyncio.Event) -> None:
    """Background sweep: delete retiring prefixes off the dethrone critical path."""
    while not stop.is_set():
        try:
            await _gc_retiring(store, backup)
        except Exception as e:
            log.warning(f"retirement sweep error: {e}")
        try:
            await _cancellable(asyncio.sleep(BACKUP_RETIREMENT_GRACE_S), stop)
        except asyncio.CancelledError:
            return


async def run(cfg: Config, chain: Chain, slots=None):
    owned_slots = slots is None
    backup, s3_configs = _make_backup_manager(cfg, chain, slots)
    if owned_slots:
        slots = TargonSlots(cfg, hotkey=chain.hotkey, s3_configs=s3_configs)
    if hasattr(slots, "reconcile"):
        await slots.reconcile()
    envs = await _load_envs(cfg)
    store = Store(cfg.db_path)
    store.abort_running_duels()
    await _gc_staging(store, backup)
    state = _RunState()
    king_provision_fails = 0
    stop = asyncio.Event()
    _install_signal_handlers(stop)
    retirement = asyncio.create_task(_retirement_task(store, backup, stop))
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

            king_slot = await _ensure_king_slot(state, champ, slots, stop)
            if king_slot is None:
                king_provision_fails += 1
                if king_provision_fails >= KING_REPROVISION_LIMIT:
                    raise RuntimeError(f"champion reprovision failed {KING_REPROVISION_LIMIT} consecutive times; aborting")
                await _cancellable(asyncio.sleep(120), stop)
                continue
            king_provision_fails = 0

            await _reconcile_backup_manifest(store, champ, king_slot, s3_configs)
            champ = store.champion() or champ

            queue = sorted(
                (m for m in all_miners
                 if not _is_current_champion_registration(m, champ, champion_artifact_alive)
                 and (m.model, m.revision) not in state.attempted),
                key=lambda m: (m.block, _tiebreak(m)),
            )
            if not queue:
                state.attempted.clear()
                log.info("queue exhausted this reign; sleeping 120s before re-probe")
                await _cancellable(asyncio.sleep(120), stop)
                continue

            registered = queue[0]
            chal_slot = None
            promoted = False
            try:
                try:
                    challenger = await _pin_artifact(s3_configs, registered)
                except Exception as exc:
                    log.warning(f"challenger pin failed uid{registered.uid}: {exc}")
                    state.attempted.add((registered.model, registered.revision))
                    continue
                chal_slot, status = await _provision(slots, challenger, stop, source="hf")
                if chal_slot is None:
                    state.attempted.add((registered.model, registered.revision))
                    log.warning(f"challenger provision failed uid{registered.uid}: {status}")
                    continue

                # Kick off slot-side upload immediately. Refs come back via
                # poll_backup; if the duel ends in dethrone we read the latest
                # snapshot then. If hold, teardown_backup aborts.
                if s3_configs:
                    chal_art = artifact_id(challenger.model, challenger.revision)
                    await setup_backup(
                        chal_slot, s3_configs,
                        prefix=_derive_prefix(s3_configs, chal_art),
                        model=challenger.model, revision=challenger.revision,
                        artifact_id=chal_art,
                    )

                block = await chain.current_block()
                alpha = alpha_for_reign(block - champ.reign_start,
                                        cfg.alpha_start, cfg.alpha_final, cfg.alpha_halflife)
                duel = store.create_duel(
                    champion=champ,
                    challenger_uid=registered.uid,
                    challenger_hotkey=registered.hotkey,
                    challenger_model=challenger.model,
                    challenger_revision=challenger.revision,
                    schedule_seed=secrets.token_hex(16),
                    pairs_per_env=cfg.duel_pairs_per_env,
                    min_discordant=cfg.duel_min_discordant,
                    alpha=alpha,
                    started_block=block,
                )
                log.info(f"duel: champion {champ.model}@{champ.revision} vs uid{registered.uid} "
                         f"{challenger.model}@{challenger.revision} alpha={alpha:.4g}")
                result = await _run_fixed_duel(store, chain, _champion_miner(champ), king_slot,
                                               challenger, chal_slot, envs, cfg, duel, stop)
                if result.status == "champion_slot_dead":
                    await _safe_teardown(slots, king_slot, "champion-slot-dead")
                    state.king_slot = None
                    await teardown_backup(chal_slot)
                    continue
                if result.status == "delivery_stalled":
                    state.attempted.add((registered.model, registered.revision))
                    await teardown_backup(chal_slot)
                    log.warning("delivery_stalled; keeping champion slot, skipping challenger this pass")
                    continue
                if result.status != "dethrone" or result.decision is None or not result.decision.dethrone:
                    state.attempted.add((registered.model, registered.revision))
                    await teardown_backup(chal_slot)
                    log.info(f"verdict: champion holds ({result.status}) counts={result.counts}")
                    continue

                fresh = await chain.list_miners()
                if not await _registered_artifact_alive(s3_configs, fresh, registered, challenger.revision):
                    log.warning(f"verdict skipped: challenger uid{registered.uid}@{registered.revision} identity changed")
                    state.attempted.add((registered.model, registered.revision))
                    await teardown_backup(chal_slot)
                    continue

                # Atomic promotion: db first, then king pointer, then chain.
                snapshot = await poll_backup(chal_slot) if s3_configs else None
                refs = (snapshot or {}).get("refs") or []
                new_art = artifact_id(challenger.model, challenger.revision)
                manifest_str = ""
                manifest_prefix = ""
                if refs:
                    objs = [ManifestRef(
                        provider=str(r["provider"]), bucket=str(r["bucket"]),
                        key=str(r["key"]), prefix=str(r["prefix"]),
                        sha256=str(r.get("sha256", "")),
                    ) for r in refs]
                    manifest_str = encode_refs(objs)
                    manifest_prefix = manifest_str
                new_champ = Champion(
                    artifact_id=new_art,
                    model=challenger.model,
                    revision=challenger.revision,
                    uid=registered.uid,
                    hotkey=registered.hotkey,
                    reign_start=await chain.current_block(),
                    backup_manifest=manifest_str,
                    backup_prefix=manifest_prefix,
                    payable=True,
                )
                store.set_champion(new_champ)
                if refs:
                    _persist_refs(store, new_champ, refs)

                old_slot = state.king_slot
                state.king_slot = chal_slot
                chal_slot = None
                promoted = True
                state.attempted.clear()
                await _publish_champion(store, chain, cfg, new_champ)
                if old_slot is not None:
                    await _safe_teardown(slots, old_slot, "old-champion-promoted")
                log.info(f"DETHRONE: {champ.model}@{champ.revision} -> uid {registered.uid}")
            finally:
                if chal_slot is not None and not promoted:
                    await _safe_teardown(slots, chal_slot, "challenger-end")
    finally:
        log.info("shutdown")
        retirement.cancel()
        try: await retirement
        except (asyncio.CancelledError, BaseException): pass
        if state.king_slot is not None:
            await _safe_teardown(slots, state.king_slot, "shutdown-king")
        store.close()


def bittensor_chain(cfg: Config) -> tuple[Chain, Subtensor]:
    """Wire a Chain from Bittensor subtensor + wallet. Caller owns sub.close()."""
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
    """A Chain wired to a fixed miner list. For local dev / tests."""
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
