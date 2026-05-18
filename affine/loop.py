"""Paired king-of-the-hill validator loop."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import signal
from dataclasses import dataclass
from typing import Awaitable, Callable

import bittensor as bt
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

from .chain import Miner, Subtensor, _tiebreak, _truthy_env, clear_weights, get_miners, set_weights
from .config import BASELINE_MODELS, Config, IdleStrategy
from .duel import run_duel
from .envs import EnvFactory
from .store import BackupRecord, Champion, Store, artifact_id
from .vllm import Slot, SlotProvisionFailed, VllmSlots, make_slots, poll_backup

log = logging.getLogger(__name__)

KING_REPROVISION_LIMIT = 10
BACKUP_RETIREMENT_GRACE_S = 600
_PUBLISHED_INTENTS: set[tuple] = set()


@dataclass
class Chain:
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
            try:
                await asyncio.wait_for(asyncio.shield(worker), timeout=60)
            except asyncio.TimeoutError:
                log.warning("_cancellable: worker cleanup exceeded 60s; rental may leak until reconcile")
            except asyncio.CancelledError:
                pass
            except BaseException as exc:
                log.warning(f"_cancellable: worker cleanup failed ({type(exc).__name__}): {exc}")
        if not delivered and worker.done() and not worker.cancelled() and worker.exception() is None:
            orphan = worker.result()
            if on_orphan is not None and orphan is not None:
                try:
                    await asyncio.shield(on_orphan(orphan))
                except BaseException as exc:
                    log.warning(f"_cancellable: on_orphan failed: {type(exc).__name__}: {exc}")


async def _safe_teardown(slot: Slot, ctx: str = "") -> None:
    try:
        await asyncio.shield(slot.teardown())
    except Exception as exc:
        log.warning(f"teardown error ({ctx or slot.model}): {exc}")


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


async def _sleep(seconds: float, stop: asyncio.Event) -> None:
    await _cancellable(asyncio.sleep(seconds), stop)


async def _provision(chain: list[VllmSlots], miner: Miner, stop: asyncio.Event, **kwargs) -> Slot | None:
    for i, slots in enumerate(chain):
        try:
            return await _cancellable(
                slots.provision(miner.model, miner.revision, **kwargs),
                stop,
                on_orphan=lambda s: _safe_teardown(s, "provision-orphan"),
            )
        except SlotProvisionFailed as exc:
            log.warning(f"provision crashloop {miner.model}@{miner.revision}: {exc}")
            return None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            tail = "; falling through" if i + 1 < len(chain) else " (no fallback left)"
            log.warning(
                f"{slots.NAME} provision failed{tail} uid{miner.uid} "
                f"{miner.model}@{miner.revision}: {type(exc).__name__}: {exc}"
            )
    return None


def _backup_configs(cfg: Config, hotkey: str) -> list[S3Config]:
    from .backup import S3Config

    if _truthy_env("AFFINE_LOCAL"):
        return []
    hot_hash = hashlib.sha256(hotkey.encode()).hexdigest()[:16]
    namespace = os.getenv("AFFINE_NAMESPACE", "prod").strip().strip("/") or "prod"
    configs = S3Config.from_env(default_prefix=f"{cfg.netuid}/{namespace}/{hot_hash}")
    if not configs:
        raise RuntimeError("Hippius or R2 S3 credentials are required for production runs")
    return list(configs.values())


def _champion_miner(champ: Champion) -> Miner:
    return Miner(champ.uid, champ.hotkey or "", champ.model, champ.revision, champ.reign_start)


def _pin_hf_revision(model: str, revision: str) -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    return HfApi(token=token).model_info(model, revision=revision).sha


def _hf_ref_missing(exc: Exception) -> bool:
    return isinstance(exc, (RepositoryNotFoundError, RevisionNotFoundError))


def _with_revision(miner: Miner, revision: str) -> Miner:
    return Miner(uid=miner.uid, hotkey=miner.hotkey, model=miner.model, revision=revision, block=miner.block)


async def _pin_artifact(s3_configs: list[S3Config], miner: Miner) -> Miner:
    if not s3_configs:
        return miner
    revision = await asyncio.to_thread(_pin_hf_revision, miner.model, miner.revision)
    return _with_revision(miner, revision)


async def _registered_artifact_alive(
    s3_configs: list[S3Config],
    miners: list[Miner],
    registered: Miner,
    pinned_revision: str,
) -> bool:
    for miner in miners:
        if miner.uid != registered.uid or miner.hotkey != registered.hotkey or miner.model != registered.model:
            continue
        if not s3_configs:
            return miner.revision == registered.revision
        try:
            return await asyncio.to_thread(_pin_hf_revision, miner.model, miner.revision) == pinned_revision
        except Exception as exc:
            log.warning(f"fresh pin failed for uid{miner.uid} {miner.model}@{miner.revision}: {exc}")
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
    for miner in miners:
        if miner.uid != champ.uid or miner.hotkey != champ.hotkey or miner.model != champ.model:
            continue
        if not s3_configs:
            return "alive" if miner.revision == champ.revision else "dead"
        if miner.revision == champ.revision:
            return "alive"
        try:
            return "alive" if await asyncio.to_thread(_pin_hf_revision, miner.model, miner.revision) == champ.revision else "dead"
        except Exception as exc:
            if _hf_ref_missing(exc):
                if missing_ref_alive:
                    log.warning(f"champion HF ref missing for uid{miner.uid} {miner.model}@{miner.revision}; continuing from backup")
                    return "alive"
                log.warning(f"champion HF ref missing for uid{miner.uid} {miner.model}@{miner.revision}")
                return "dead"
            log.warning(f"champion pin check failed for uid{miner.uid} {miner.model}@{miner.revision}: {exc}")
            return "unknown"
    return "dead"


def _is_current_champion_registration(miner: Miner, champ: Champion, artifact_alive: bool = False) -> bool:
    if not champ.payable:
        return False
    if champ.uid is not None and miner.uid == champ.uid and miner.hotkey == champ.hotkey:
        return (miner.model, miner.revision) == (champ.model, champ.revision) or (
            artifact_alive and miner.model == champ.model
        )
    return (miner.model, miner.revision) == (champ.model, champ.revision)


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
        (m for m in miners if m.model == baseline_model and (not baseline_revision or m.revision == baseline_revision)),
        None,
    )
    model = registered.model if registered else baseline_model
    revision = registered.revision if registered else baseline_revision
    if not revision:
        revision = "main" if not s3_configs else await asyncio.to_thread(_pin_hf_revision, model, "main")
    if registered and s3_configs:
        revision = await asyncio.to_thread(_pin_hf_revision, model, revision)
        registered = _with_revision(registered, revision)
    champ = Champion(
        artifact_id=artifact_id(model, revision),
        model=model,
        revision=revision,
        uid=registered.uid if registered else None,
        hotkey=registered.hotkey if registered else None,
        reign_start=await chain.current_block(),
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
    king_slot: Slot | None,
    champ: Champion,
    slots: list[VllmSlots],
    s3_configs: list[S3Config],
    store: Store,
    stop: asyncio.Event,
) -> Slot | None:
    if king_slot is not None:
        return king_slot
    miner = _champion_miner(champ)
    slot = await _provision(slots, miner, stop, source="hf")
    if slot is None:
        recovered = store.latest_backup_for(champ.artifact_id)
        if recovered is not None:
            log.warning("champion HF reprovision failed; using backup")
            slot = await _provision(slots, miner, stop, source="s3", backup_manifest_key=recovered.manifest_key)
            if slot is not None:
                store.update_backup_manifest(champ.artifact_id, recovered.manifest_key, recovered.model, recovered.revision)
    if slot is None:
        log.error("champion reprovision failed")
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
    key = ("set_weights", champ.artifact_id, champ.uid, champ.hotkey, dry)
    if key in _PUBLISHED_INTENTS:
        return True
    if dry:
        log.info(f"dry-run weights set: uid {champ.uid}")
        _PUBLISHED_INTENTS.add(key)
        return True
    ok = await chain.publish_winner(champ.uid, champ.hotkey)
    if ok and not await chain.uid_matches_hotkey(champ.uid, champ.hotkey):
        log.warning(f"champion uid={champ.uid} hotkey changed after publish; burning weights")
        store.demote_champion(champ.artifact_id)
        await _publish_burn(store, chain, cfg, champ.artifact_id)
        return False
    if ok:
        _PUBLISHED_INTENTS.add(key)
    return ok


async def _publish_burn(store: Store, chain: Chain, cfg: Config, artifact: str) -> bool:
    dry = cfg.dry_run
    key = ("burn", artifact, None, None, dry)
    if key in _PUBLISHED_INTENTS:
        return True
    if dry:
        log.info("dry-run weights burn")
        _PUBLISHED_INTENTS.add(key)
        return True
    ok = bool(await chain.burn_weights())
    if ok:
        _PUBLISHED_INTENTS.add(key)
    return ok


def _refs_to_objs(refs: list[dict]) -> list[ManifestRef]:
    from .backup import ManifestRef

    return [
        ManifestRef(
            provider=str(r["provider"]),
            bucket=str(r["bucket"]),
            key=str(r["key"]),
            prefix=str(r["prefix"]),
            sha256=str(r.get("sha256", "")),
        )
        for r in refs
    ]


async def _reconcile_backup_manifest(store: Store, champ: Champion, slot: Slot, s3_configs: list[S3Config]) -> None:
    if not slot.sidecar_url or not s3_configs:
        return
    from .backup import encode_refs

    state = await poll_backup(slot)
    if state is None or state.get("state") != "done" or state.get("artifact_id") != champ.artifact_id:
        return
    refs = state.get("refs") or []
    if not refs:
        return
    manifest_key = encode_refs(_refs_to_objs(refs))
    current = store.latest_backup_for(champ.artifact_id)
    if current is None or current.manifest_key != manifest_key:
        store.update_backup_manifest(champ.artifact_id, manifest_key, champ.model, champ.revision)


async def _gc_retiring(store: Store, s3_configs: list[S3Config]) -> None:
    if not s3_configs:
        return
    from .backup import delete_refs

    by_name = {c.name: c for c in s3_configs}
    for old in store.retiring_backups():
        if await asyncio.to_thread(delete_refs, old.manifest_key, by_name):
            store.mark_backup_deleted(old.manifest_key)


async def _retirement_task(store: Store, s3_configs: list[S3Config], stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await _gc_retiring(store, s3_configs)
        except Exception as exc:
            log.warning(f"retirement sweep error: {exc}")
        try:
            await _sleep(BACKUP_RETIREMENT_GRACE_S, stop)
        except asyncio.CancelledError:
            return


def _normalize_pi(cfg: Config, envs: dict[str, tuple]) -> dict[str, float]:
    overrides = {env: float(v) for env, v in cfg.pi_overrides}
    if overrides:
        weights = {env: max(overrides.get(env, 0.0), 0.0) for env in envs}
        if sum(weights.values()) > 0.0:
            total = sum(weights.values())
            return {env: value / total for env, value in weights.items()}
    return {env: 1.0 / len(envs) for env in envs}


def _versions_hash(env_specs, pi: dict[str, float]) -> str:
    payload = {
        "envs": [
            {"name": spec.name, "entrypoint": spec.entrypoint, "params": dict(sorted(spec.params.items()))}
            for spec in env_specs
        ],
        "pi": sorted((k, float(v)) for k, v in pi.items()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:32]


def _pick_challenger(
    miners: list[Miner],
    champ: Champion,
    attempted: set[str],
    model_skiplist: tuple[str, ...],
    champion_artifact_alive: bool,
) -> Miner | None:
    queue = sorted(
        (
            miner
            for miner in miners
            if miner.model not in model_skiplist
            and not _is_current_champion_registration(miner, champ, champion_artifact_alive)
            and artifact_id(miner.model, miner.revision) not in attempted
        ),
        key=lambda m: (m.block, _tiebreak(m)),
    )
    return queue[0] if queue else None


async def _crown(
    store: Store,
    chain: Chain,
    cfg: Config,
    s3_configs: list[S3Config],
    old_champ: Champion,
    registered: Miner,
    challenger: Miner,
    chal_slot: Slot,
    attempted: set[str],
) -> Champion:
    snapshot = await poll_backup(chal_slot) if s3_configs else None
    refs = (snapshot or {}).get("refs") or []
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
    backup = None
    if refs:
        from .backup import encode_refs

        backup = BackupRecord(new_art, new_champ.model, new_champ.revision, encode_refs(_refs_to_objs(refs)), "current")
    old_art = old_champ.artifact_id
    attempted.add(old_art)
    store.set_champion(new_champ, backup=backup)
    ok = await _publish_champion(store, chain, cfg, new_champ)
    log.info(f"DETHRONE: {old_champ.model}@{old_champ.revision} -> uid {registered.uid}" + ("" if ok else " (publish deferred)"))
    return new_champ


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
    pi = _normalize_pi(cfg, envs)
    versions_hash = _versions_hash(cfg.environments, pi)
    attempted = store.attempted_artifact_ids(versions_hash)
    log.info(f"versions_hash={versions_hash}; loaded {len(attempted)} attempted artifacts")
    stop = asyncio.Event()
    _install_signal_handlers(stop)
    retirement = asyncio.create_task(_retirement_task(store, s3_configs, stop))
    king_slot: Slot | None = None
    king_provision_fails = 0

    def mark_attempted(miner: Miner) -> None:
        attempted.add(artifact_id(miner.model, miner.revision))

    try:
        champ = store.champion()
        if champ is None:
            champ, king_slot = await _bootstrap_champion(store, s3_configs, slots, chain, cfg, stop)
        while not stop.is_set():
            champ = store.champion() or champ
            all_miners = await chain.list_miners()
            registration = await _champion_registration_status(s3_configs, all_miners, champ, missing_ref_alive=True)
            champion_artifact_alive = registration == "alive"
            if champ.payable and registration == "unknown":
                log.warning("champion artifact identity unknown; pausing publication and duels")
                await _sleep(120, stop)
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

            if cfg.idle_strategy == IdleStrategy.WARM_KING:
                king_slot = await _ensure_king_slot(king_slot, champ, slots, s3_configs, store, stop)
                if king_slot is None:
                    king_provision_fails += 1
                    if king_provision_fails >= KING_REPROVISION_LIMIT:
                        raise RuntimeError(f"champion reprovision failed {KING_REPROVISION_LIMIT} consecutive times; aborting")
                    await _sleep(120, stop)
                    continue
                king_provision_fails = 0
                await _reconcile_backup_manifest(store, champ, king_slot, s3_configs)

            registered = _pick_challenger(all_miners, champ, attempted, cfg.model_skiplist, champion_artifact_alive)
            if registered is None:
                log.info("queue exhausted this reign; sleeping 120s before re-probe")
                await _sleep(120, stop)
                continue

            chal_slot: Slot | None = None
            promoted = False
            try:
                try:
                    challenger = await _pin_artifact(s3_configs, registered)
                except Exception as exc:
                    log.warning(f"challenger pin failed uid{registered.uid}: {exc}")
                    mark_attempted(registered)
                    continue

                if cfg.idle_strategy == IdleStrategy.COLD_BOTH:
                    king_slot = await _ensure_king_slot(None, champ, slots, s3_configs, store, stop)
                    if king_slot is None:
                        king_provision_fails += 1
                        if king_provision_fails >= KING_REPROVISION_LIMIT:
                            raise RuntimeError(f"champion reprovision failed {KING_REPROVISION_LIMIT} consecutive times; aborting")
                        await _sleep(120, stop)
                        continue
                    king_provision_fails = 0
                    await _reconcile_backup_manifest(store, champ, king_slot, s3_configs)

                chal_slot = await _provision(slots, challenger, stop, source="hf")
                if chal_slot is None:
                    mark_attempted(challenger)
                    continue

                log.info(
                    f"duel: champion {champ.model}@{champ.revision} vs uid{registered.uid} "
                    f"{challenger.model}@{challenger.revision} "
                    f"δ_dethrone={cfg.delta_dethrone:.3g} δ_hold={cfg.delta_hold:.3g} α={cfg.alpha:.3g}"
                )
                verdict = await run_duel(
                    store,
                    chain,
                    cfg,
                    envs,
                    pi,
                    versions_hash,
                    champ,
                    king_slot,
                    challenger,
                    chal_slot,
                    stop,
                )
                if verdict.reason == "cancelled":
                    continue
                if verdict.outcome.value != "dethrone":
                    mark_attempted(challenger)
                    continue

                fresh = await chain.list_miners()
                if not await _registered_artifact_alive(s3_configs, fresh, registered, challenger.revision):
                    log.warning(f"verdict skipped: challenger uid{registered.uid}@{registered.revision} identity changed")
                    mark_attempted(challenger)
                    continue

                old_slot = king_slot
                champ = await _crown(
                    store,
                    chain,
                    cfg,
                    s3_configs,
                    champ,
                    registered,
                    challenger,
                    chal_slot,
                    attempted,
                )
                king_slot = chal_slot
                chal_slot = None
                promoted = True
                if old_slot is not None:
                    await _safe_teardown(old_slot, "old-champion-promoted")
            finally:
                if chal_slot is not None and not promoted:
                    await _safe_teardown(chal_slot, "challenger-end")
                if cfg.idle_strategy == IdleStrategy.COLD_BOTH and king_slot is not None:
                    await _safe_teardown(king_slot, "cold-king-end")
                    king_slot = None
    finally:
        log.info("shutdown")
        retirement.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await retirement
        if king_slot is not None:
            await _safe_teardown(king_slot, "shutdown-king")
        for slot_provider in slots:
            try:
                await slot_provider.aclose()
            except Exception as exc:
                log.warning(f"aclose {slot_provider.NAME}: {exc}")
        store.close()


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
    async def _miners():
        return miners

    async def _block():
        return 0

    async def _publish(uid: int, expected_hotkey: str) -> bool:
        log.info(f"local winner: uid {uid} (hk {expected_hotkey[:8]})")
        return True

    async def _burn() -> bool:
        log.info("local winner: none")
        return True

    async def _uid_matches(uid: int, expected_hotkey: str) -> bool:
        return any(m.uid == uid and m.hotkey == expected_hotkey for m in miners)

    return Chain(
        hotkey=hotkey,
        list_miners=_miners,
        current_block=_block,
        publish_winner=_publish,
        burn_weights=_burn,
        uid_matches_hotkey=_uid_matches,
    )
