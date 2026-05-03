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
import logging
import os
import secrets
import signal
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
from .config import BASELINE_MODELS, Config
from .envs import EnvFactory
from .paired import (
    EnvCS,
    MeanDecision,
    PairCounts,
    decide_dethrone,
    env_lower_cs,
    env_upper_cs,
    select_env,
)
from .sampler import run_one
from .store import BackupRecord, Champion, PairSample, Store, artifact_id
from .vllm import (
    Slot,
    SlotProvisionFailed,
    VllmSlots,
    make_slots,
    poll_backup,
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


def _start_prefetch(state: "_RunState", slots: list[VllmSlots], miner: Miner,
                    stop: asyncio.Event) -> None:
    """Kick off a background provision of `miner` so it overlaps the current duel.

    Caller must already have ensured `state.prefetch_task is None`. The task is
    cancellable via the same `stop` event used by foreground provisions; on
    shutdown, `run()`'s outer `finally` cancels and tears down its result."""
    state.next_challenger = miner
    state.prefetch_task = asyncio.create_task(_provision(slots, miner, stop, source="hf"))


async def _consume_prefetch(state: "_RunState", expected: Miner | None) -> Slot | None:
    """Await the in-flight prefetch task. Return the slot iff (a) a task exists
    and (b) the queued challenger matches `expected` and (c) provision succeeded.
    Otherwise tear down any successfully-provisioned slot (stale prefetch) and
    return None. Always clears `state.next_challenger` and `state.prefetch_task`."""
    task, queued = state.prefetch_task, state.next_challenger
    state.prefetch_task = None
    state.next_challenger = None
    if task is None:
        return None
    try:
        slot = await task
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.warning(f"prefetch failed: {type(e).__name__}: {e}")
        return None
    if expected is None or queued != expected or slot is None:
        if slot is not None:
            await _safe_teardown(slot, "prefetch-stale")
        return None
    return slot


async def _safe_teardown(slot: Slot, ctx: str) -> None:
    """Shielded so an outer cancel during teardown doesn't interrupt the platform
    DELETE — the request keeps flying as a fire-and-forget task; reconcile() at
    next startup is the backstop. Slot owns its teardown closure (captures the
    provider instance + handle), so dispatch is self-contained."""
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
    """Iterate the provider chain on infra-class failure. `SlotProvisionFailed`
    (miner-fault crashloop) does NOT fall through — the same artifact would
    crashloop on every provider. `CancelledError` propagates."""
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


def _delivered(r) -> bool:
    if isinstance(r, BaseException) or not isinstance(r, tuple) or len(r) != 4:
        return False
    return bool(r[2])


async def _pair_sample(chain: Chain, wrapper, params: dict, timeout: float,
                       king_slot: Slot, king: Miner, ck: int,
                       chal_slot: Slot, challenger: Miner, cc: int,
                       env_name: str, task_id: int, e_idx: int):
    """One bounded retry per side on infra-fault. Same task_id, deterministic
    prompt and (at temperature=0) deterministic model output, so retry only
    changes outcome when transport recovers. Bounds asymmetric-censoring bias
    when king and chal slots have different infra-fault rates: pair retain
    rate becomes ~(1 - r^2) instead of (1 - r) per side."""
    async def king_call():
        return await _sample(chain, wrapper, params, timeout, king_slot, king, env_name, ck, task_id)

    async def chal_call():
        return await _sample(chain, wrapper, params, timeout, chal_slot, challenger, env_name, cc, task_id)

    async def maybe_retry(retry_fn, original):
        if _delivered(original):
            return original
        try:
            result = await retry_fn()
        except BaseException as exc:
            return exc
        return result if _delivered(result) else original

    rk, rc = await asyncio.gather(king_call(), chal_call(), return_exceptions=True)
    if not _delivered(rk) or not _delivered(rc):
        rk, rc = await asyncio.gather(
            maybe_retry(king_call, rk),
            maybe_retry(chal_call, rc),
        )
    return e_idx, env_name, rk, rc


@dataclass
class _RunState:
    king_slot: Slot | None = None
    attempted: set[ArtKey] = field(default_factory=set)
    # Prefetch the next challenger so its provisioning (5–15 min) overlaps the
    # current duel. `next_challenger` is the (model, revision) the in-flight
    # task is provisioning. On dethrone the prefetch is discarded (queue may
    # not be valid under the new champion); on hold the prefetched slot is
    # consumed if the next iteration's queue[0] still matches.
    next_challenger: Miner | None = None
    prefetch_task: asyncio.Task | None = None


@dataclass(frozen=True)
class _DuelRun:
    decision: MeanDecision | None
    status: str
    per_env_counts: dict[str, PairCounts]

    @property
    def counts(self) -> PairCounts:
        out = PairCounts()
        for c in self.per_env_counts.values():
            out = PairCounts(
                challenger_only=out.challenger_only + c.challenger_only,
                champion_only=out.champion_only + c.champion_only,
                both_pass=out.both_pass + c.both_pass,
                both_fail=out.both_fail + c.both_fail,
            )
        return out


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
    # Demoted champion: payable=0, uid=NULL after demote_champion. The (model,
    # revision) arm below would otherwise blacklist every miner sharing the dead
    # artifact and starve the queue, deadlocking recovery. Live-case second arm
    # stays: a different miner re-registering the same (model, revision) as the
    # live champion still gets filtered (identical-artifact duels add no evidence).
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
    """Bootstrap a fresh champion. The slot's sidecar takes over backup
    durability; this function only commits the in-DB champion and provisions
    the king slot. provision() POSTs /setup to kick off the upload."""
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
    # Content compare, not length: a reprovision can produce a different set of
    # refs with the same count (different prefixes, same provider count). Length
    # would skip persisting and leave the DB pointing at stale prefixes.
    new_key = encode_refs(_refs_to_objs(new_refs))
    cur = store.latest_backup_for(champ.artifact_id)
    if cur is None or cur.manifest_key != new_key:
        store.update_backup_manifest(
            artifact_id=champ.artifact_id, manifest_key=new_key,
            model=champ.model, revision=champ.revision,
        )


def _pair_sample_from_rows(duel_id: int, env: str, task_id: int, iter_idx: int,
                           launch_seq: int, block: int, king: Miner, chal: Miner,
                           rk, rc) -> PairSample:
    def unpack(result, miner: Miner) -> tuple[int, float, int, int]:
        if isinstance(result, BaseException):
            log.warning(f"sample raised uid{miner.uid}: {type(result).__name__}: {result}")
            return 0, 0.0, 0, 0
        p, latency, delivered, tokens = result
        return p, latency, int(delivered), tokens
    kp, kl, kd, kt = unpack(rk, king)
    cp, cl, cd, ct = unpack(rc, chal)
    return PairSample(duel_id, env, task_id, iter_idx, launch_seq, block,
                      kp, cp, kl, cl, kd, cd, kt, ct)


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
    stop: asyncio.Event,
) -> _DuelRun:
    """Stratified anytime-valid duel.

    Per-env counts feed independent always-valid CSs at level (alpha/2)/E,
    aggregated at the parameter level: L_mu = sum pi_e L_e, U_mu = sum pi_e U_e.
    Dethrone iff L_mu > p_star = 0.5 + delta_p. Futility iff U_mu <= p_star.

    Concurrency: launch-order ledger. Each task is tagged with a launch_seq;
    completed results are buffered keyed by seq; the analysis cursor advances
    only through the completed launch-order prefix; verdict checks happen only
    after the cursor moves. dwell_batch tasks are kept in flight at all times
    via continuous refill (FIRST_COMPLETED). This keeps the e-process update
    order predictable from the past filtration even though completions can
    arrive out of order.

    Sampling: cold-start round-robin until each env has n_min total samples,
    then argmax of env_score. Selection depends only on past cursor data.

    SLOT_DEAD checks run before the statistical decision so infra failures
    never cost a reign on the decision side."""
    env_names = [spec.name for spec in cfg.environments]
    e_count = len(env_names)
    weights = duel.env_weights
    p_star = 0.5 + duel.delta_p
    alpha_d = duel.alpha / 2.0
    alpha_f = duel.alpha / 2.0
    am = alpha_d / e_count
    ap = alpha_f / e_count

    per_env_counts: dict[str, PairCounts] = {e: PairCounts() for e in env_names}
    counters: dict[tuple[int, str, str], int] = {}
    in_flight: dict[int, asyncio.Task] = {}
    pending: dict[int, tuple] = {}
    env_latency_total: dict[str, float] = {e: 0.0 for e in env_names}
    env_latency_count: dict[str, int] = {e: 0 for e in env_names}
    cursor = 0
    next_seq = 0
    king_dead = chal_dead = 0
    pair_dead = 0

    def alloc(miner: Miner, env: str) -> int:
        key = (miner.uid, miner.revision, env)
        c = counters.get(key, 0)
        counters[key] = c + 1
        return c

    def env_cs_now() -> dict[str, EnvCS]:
        out: dict[str, EnvCS] = {}
        for e in env_names:
            c = per_env_counts[e]
            out[e] = EnvCS(
                k=c.challenger_only,
                n=c.discordant,
                L=env_lower_cs(c.challenger_only, c.discordant, am),
                U=env_upper_cs(c.challenger_only, c.discordant, ap),
            )
        return out

    def env_cs_payload(env_cs: dict[str, EnvCS]) -> dict[str, dict]:
        return {e: {"k": cs.k, "n": cs.n, "L": cs.L, "U": cs.U} for e, cs in env_cs.items()}

    async def drain(reason: str) -> None:
        for task in in_flight.values():
            if not task.done():
                task.cancel()
        for task in list(in_flight.values()):
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                pass
            except BaseException as exc:
                log.warning(f"duel drain ({reason}) interrupted: {type(exc).__name__}: {exc}")
        in_flight.clear()

    async def block() -> int:
        try:
            return await chain.current_block()
        except Exception as ex:
            log.warning(f"current_block failed during duel: {ex}; using 0")
            return 0

    async def _abort_duel(reason: str, status: str) -> _DuelRun:
        await drain(reason)
        env_cs = env_cs_now()
        l_mu = sum(weights[e] * env_cs[e].L for e in env_names)
        u_mu = sum(weights[e] * env_cs[e].U for e in env_names)
        store.finish_duel(duel.id, status, _DuelRun(None, status, per_env_counts).counts,
                          l_mu, u_mu, env_cs_payload(env_cs), await block())
        return _DuelRun(None, status, per_env_counts)

    async def launch_one(env_name: str, seq: int) -> asyncio.Task:
        wrapper, spec = envs[env_name]
        params = {k: v for k, v in spec.params.items() if k != "timeout"}
        timeout = float(spec.params.get("timeout", 600))
        lo, hi = spec.task_range
        task_id = _task_id(
            king.uid if king.uid is not None else -1,
            challenger.uid if challenger.uid is not None else -1,
            env_name, seq, lo, hi,
            salt=f"{chain.hotkey}:{duel.schedule_seed}")
        return asyncio.create_task(_pair_sample(
            chain, wrapper, params, timeout,
            king_slot, king, alloc(king, env_name),
            chal_slot, challenger, alloc(challenger, env_name),
            env_name, task_id, seq))

    def env_costs() -> dict[str, float]:
        # Pair latency = slower side. Floor at 1.0 so cost stays meaningful and
        # never zero (avoid division-by-zero in env_score).
        return {
            e: max(env_latency_total[e] / env_latency_count[e], 1.0)
            if env_latency_count[e] > 0 else 1.0
            for e in env_names
        }

    async def fill() -> None:
        nonlocal next_seq
        while len(in_flight) < cfg.dwell_batch:
            env_cs = env_cs_now()
            env_name = select_env(per_env_counts, weights, env_cs,
                                  p_star, cfg.n_min_per_env, cfg.score_lambda,
                                  costs=env_costs())
            seq = next_seq
            in_flight[seq] = await launch_one(env_name, seq)
            next_seq += 1

    while not stop.is_set():
        await fill()
        try:
            done, _ = await _cancellable(
                asyncio.wait(list(in_flight.values()), return_when=asyncio.FIRST_COMPLETED),
                stop,
            )
        except asyncio.CancelledError:
            await drain("stop")
            env_cs = env_cs_now()
            l_mu = sum(weights[e] * env_cs[e].L for e in env_names)
            u_mu = sum(weights[e] * env_cs[e].U for e in env_names)
            run = _DuelRun(None, "cancelled", per_env_counts)
            store.finish_duel(duel.id, "cancelled", run.counts, l_mu, u_mu,
                              env_cs_payload(env_cs), None)
            if stop.is_set():
                return run
            raise

        # Move completed tasks into the pending buffer keyed by launch_seq.
        for task in done:
            seq = next(s for s, t in in_flight.items() if t is task)
            del in_flight[seq]
            pending[seq] = task.result()

        # Advance cursor through the completed launch-order prefix.
        new_samples: list[PairSample] = []
        progressed = False
        while cursor in pending:
            e_idx, env_name, rk, rc = pending.pop(cursor)
            wrapper, spec = envs[env_name]
            lo, hi = spec.task_range
            task_id = _task_id(
                king.uid if king.uid is not None else -1,
                challenger.uid if challenger.uid is not None else -1,
                env_name, e_idx, lo, hi,
                salt=f"{chain.hotkey}:{duel.schedule_seed}")
            b = await block()
            sample = _pair_sample_from_rows(duel.id, env_name, task_id, e_idx, cursor,
                                            b, king, challenger, rk, rc)
            new_samples.append(sample)
            if sample.champion_delivered and sample.challenger_delivered:
                per_env_counts[env_name] = per_env_counts[env_name].add(
                    sample.champion_pass, sample.challenger_pass)
                env_latency_total[env_name] += max(sample.champion_latency, sample.challenger_latency)
                env_latency_count[env_name] += 1
                pair_dead = 0
            else:
                pair_dead += 1
            king_dead = 0 if sample.champion_delivered else king_dead + 1
            chal_dead = 0 if sample.challenger_delivered else chal_dead + 1
            cursor += 1
            progressed = True
        if new_samples:
            store.add_samples(new_samples)

        # SLOT_DEAD takes precedence over the statistical decision.
        if king_dead >= SLOT_DEAD and chal_dead >= SLOT_DEAD:
            return await _abort_duel("delivery-stalled", "delivery_stalled")
        if king_dead >= SLOT_DEAD:
            return await _abort_duel("champion-slot-dead", "champion_slot_dead")
        if chal_dead >= SLOT_DEAD:
            return await _abort_duel("challenger-slot-dead", "challenger_slot_dead")
        if pair_dead >= SLOT_DEAD:
            return await _abort_duel("delivery-stalled", "delivery_stalled")

        # Verdict check only after cursor advancement (launch-order ledger).
        if not progressed:
            continue
        decision = decide_dethrone(per_env_counts, weights, p_star, alpha_d, alpha_f)
        if decision.dethrone or decision.futility:
            status = "dethrone" if decision.dethrone else "hold_with_evidence"
            await drain(status)
            run = _DuelRun(decision, status, per_env_counts)
            cs_payload = {e: {"k": cs.k, "n": cs.n, "L": cs.L, "U": cs.U}
                          for e, cs in decision.env_cs}
            store.finish_duel(duel.id, status, run.counts,
                              decision.L_mu, decision.U_mu, cs_payload, await block())
            return run
        # Anytime-valid stopping budget: hold_inconclusive once cursor reaches the cap.
        if cursor >= cfg.max_pairs_per_duel:
            return await _abort_duel("budget-exhausted", "hold_inconclusive")

    await drain("stop")
    env_cs = env_cs_now()
    l_mu = sum(weights[e] * env_cs[e].L for e in env_names)
    u_mu = sum(weights[e] * env_cs[e].U for e in env_names)
    run = _DuelRun(None, "cancelled", per_env_counts)
    store.finish_duel(duel.id, "cancelled", run.counts, l_mu, u_mu,
                      env_cs_payload(env_cs), None)
    return run


async def _gc_retiring(store: Store, s3_configs: list[S3Config]) -> None:
    if not s3_configs:
        return
    by_name = {c.name: c for c in s3_configs}
    for old in store.retiring_backups():
        if await asyncio.to_thread(delete_refs, old.manifest_key, by_name):
            store.mark_backup_deleted(old.manifest_key)


async def _retirement_task(store: Store, s3_configs: list[S3Config], stop: asyncio.Event) -> None:
    """Background sweep: delete retiring prefixes off the dethrone critical path."""
    while not stop.is_set():
        try:
            await _gc_retiring(store, s3_configs)
        except Exception as e:
            log.warning(f"retirement sweep error: {e}")
        try:
            await _cancellable(asyncio.sleep(BACKUP_RETIREMENT_GRACE_S), stop)
        except asyncio.CancelledError:
            return


async def run(cfg: Config, chain: Chain, slots: list[VllmSlots] | None = None):
    if slots is None:
        s3_configs = _backup_configs(cfg, chain.hotkey)
        slots = make_slots(cfg=cfg, hotkey=chain.hotkey, s3_configs=s3_configs)
    else:
        # Test/external-injection mode: caller owns the slot list and S3 is
        # disabled (sidecar-driven backups need real S3 endpoints).
        s3_configs = []
    await asyncio.gather(*(s.reconcile() for s in slots), return_exceptions=True)
    envs = await _load_envs(cfg)
    store = Store(cfg.db_path)
    store.abort_running_duels()
    state = _RunState()
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

            queue = sorted(
                (m for m in all_miners
                 if m.model not in cfg.model_skiplist
                 and not _is_current_champion_registration(m, champ, champion_artifact_alive)
                 and (m.model, m.revision) not in state.attempted),
                key=lambda m: (m.block, _tiebreak(m)),
            )
            if not queue:
                # No retries: a challenger that already held stays in `_attempted`
                # until dethrone (which clears it). Sleep, then re-list miners; if
                # new commits appear they'll show up; old held artifacts stay out.
                log.info("queue exhausted this reign; sleeping 120s before re-probe (no retry of attempted)")
                await _cancellable(asyncio.sleep(120), stop)
                continue

            registered = queue[0]
            # Consume the prefetch if it targeted this challenger; otherwise
            # discard (queue may have shifted under a new champion / new commit).
            chal_slot = await _consume_prefetch(state, expected=registered)
            promoted = False
            try:
                try:
                    challenger = await _pin_artifact(s3_configs, registered)
                except Exception as exc:
                    log.warning(f"challenger pin failed uid{registered.uid}: {exc}")
                    if chal_slot is not None:
                        await _safe_teardown(chal_slot, "pin-failed")
                        chal_slot = None
                    state.attempted.add((registered.model, registered.revision))
                    continue
                if chal_slot is None:
                    chal_slot = await _provision(slots, challenger, stop, source="hf")
                if chal_slot is None:
                    state.attempted.add((registered.model, registered.revision))
                    continue
                # Kick off prefetch of queue[1] so its provisioning overlaps the duel.
                if state.prefetch_task is None and len(queue) > 1:
                    _start_prefetch(state, slots, queue[1], stop)

                block = await chain.current_block()
                env_names = [spec.name for spec in cfg.environments]
                env_weights = {e: 1.0 / len(env_names) for e in env_names}
                duel = store.create_duel(
                    champion=champ,
                    challenger_uid=registered.uid,
                    challenger_hotkey=registered.hotkey,
                    challenger_model=challenger.model,
                    challenger_revision=challenger.revision,
                    schedule_seed=secrets.token_hex(16),
                    alpha=cfg.alpha,
                    delta_p=cfg.delta_p,
                    env_weights=env_weights,
                    started_block=block,
                )
                log.info(f"duel: champion {champ.model}@{champ.revision} vs uid{registered.uid} "
                         f"{challenger.model}@{challenger.revision} "
                         f"alpha={cfg.alpha:.3g} delta_p={cfg.delta_p:.3g} "
                         f"p_star={0.5 + cfg.delta_p:.3g}")
                result = await _run_duel(store, chain, _champion_miner(champ), king_slot,
                                         challenger, chal_slot, envs, cfg, duel, stop)
                if result.status == "champion_slot_dead":
                    await _safe_teardown(king_slot, "champion-slot-dead")
                    state.king_slot = None
                    continue
                if result.status == "delivery_stalled":
                    state.attempted.add((registered.model, registered.revision))
                    log.warning("delivery_stalled; keeping champion slot, skipping challenger this pass")
                    continue
                if result.status != "dethrone" or result.decision is None or not result.decision.dethrone:
                    state.attempted.add((registered.model, registered.revision))
                    log.info(f"verdict: champion holds ({result.status}) counts={result.counts}")
                    continue

                fresh = await chain.list_miners()
                if not await _registered_artifact_alive(s3_configs, fresh, registered, challenger.revision):
                    log.warning(f"verdict skipped: challenger uid{registered.uid}@{registered.revision} identity changed")
                    state.attempted.add((registered.model, registered.revision))
                    continue

                # Atomic commit: champion + backup land in the same transaction.
                # Order is set_champion → pointer flip → teardown → publish.
                # set_champion raising leaves state.king_slot untouched; the
                # `chal_slot is not None and not promoted` finally tears down the
                # would-be challenger. After the commit succeeds, the in-memory
                # tuple unpack is atomic (Python doesn't raise on assignment), so
                # state.king_slot is the new slot for any subsequent failure.
                snapshot = await poll_backup(chal_slot) if s3_configs else None
                refs = (snapshot or {}).get("refs") or []
                # Re-list-and-re-pin again immediately before commit. poll_backup
                # is a 1-5s HTTP call; an on-chain rotation in that window would
                # otherwise let us commit a stale artifact identity. Repeating the
                # check here closes the window. Identity rotations take ~36s, so
                # the residual window between this check and `set_champion` is
                # negligible.
                fresh2 = await chain.list_miners()
                if not await _registered_artifact_alive(s3_configs, fresh2, registered, challenger.revision):
                    log.warning(f"verdict skipped post-poll: challenger uid{registered.uid}@{registered.revision} identity changed")
                    state.attempted.add((registered.model, registered.revision))
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
                store.set_champion(new_champ, backup=backup)
                old_slot, state.king_slot = state.king_slot, chal_slot
                chal_slot = None
                promoted = True
                state.attempted.clear()
                if old_slot is not None:
                    await _safe_teardown(old_slot, "old-champion-promoted")
                # The prefetch was for the previous reign's queue[1]; under the
                # new champion the queue is recomputed and that miner may be
                # filtered out. Discard.
                await _consume_prefetch(state, expected=None)
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
        if state.prefetch_task is not None:
            state.prefetch_task.cancel()
            try: prefetched = await state.prefetch_task
            except (asyncio.CancelledError, Exception): prefetched = None
            if prefetched is not None:
                await _safe_teardown(prefetched, "shutdown-prefetch")
            state.prefetch_task = None
            state.next_challenger = None
        if state.king_slot is not None:
            await _safe_teardown(state.king_slot, "shutdown-king")
        for s in slots:
            try: await s.aclose()
            except Exception as e: log.warning(f"aclose {s.NAME}: {e}")
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
