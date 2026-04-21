from __future__ import annotations
import asyncio
import logging
import os
import signal

import affinetes as af

from .chain import Subtensor, Challenger, get_challenger_queue, set_weights
from .config import Config
from .duel import run_duel
from .scoring import Verdict, compute_k
from .vllm import Slot, SlotProvisionFailed, TargonSlots, health_ping

log = logging.getLogger(__name__)

MAX_DUEL_RETRIES = 2

# Cold-start KOTH baselines. Synthetic `uid=-1` entries used as the initial
# champion when the validator has no sitting winner — real miners must beat
# the baseline on its own duel terms to claim the throne. Tried in order.
BASELINES: tuple[Challenger, ...] = (
    Challenger(uid=-1, hotkey="", model="Qwen/Qwen3-32B", revision="main", block=0),
)


class Skiplist:
    """Tracks miner artifacts that should not be provisioned.

    Two tiers: model-wildcard (from env config) and (model, revision) pairs
    (added at runtime when provisioning fails). A miner re-committing a new
    revision gets another chance; a known-bad baseline stays excluded forever.
    """

    def __init__(self):
        self._models: set[str] = set()
        self._pairs: set[tuple[str, str]] = set()

    @classmethod
    def from_env(cls) -> "Skiplist":
        sl = cls()
        raw = os.getenv("AFFINE_MODEL_SKIPLIST", "")
        for s in raw.split(","):
            s = s.strip()
            if s:
                sl._models.add(s)
        return sl

    def add(self, model: str, revision: str) -> None:
        self._pairs.add((model, revision))
        log.info(f"skiplist: added {model}@{revision}")

    def __contains__(self, c: Challenger) -> bool:
        return c.model in self._models or (c.model, c.revision) in self._pairs

    def filter(self, queue: list[Challenger]) -> list[Challenger]:
        out = [c for c in queue if c not in self]
        dropped = len(queue) - len(out)
        if dropped:
            log.info(f"skiplist filtered {dropped}/{len(queue)} candidates")
        return out


def _load_envs(config: Config) -> dict[str, tuple]:
    out = {}
    for spec in config.environments:
        env_vars = dict(spec.env_vars)
        for k in ("CHUTES_API_KEY", "HF_TOKEN"):
            v = os.environ.get(k)
            if v:
                env_vars.setdefault(k, v)

        wrapper = af.load_env(
            image=spec.image,
            mode="docker",
            env_vars=env_vars or None,
            mem_limit=spec.mem_limit,
            pull=True,
            cleanup=True,
        )
        backend = getattr(wrapper, "_backend", None)
        if backend is not None and hasattr(backend, "_check_container_restart"):
            backend._check_container_restart = lambda: False
        out[spec.name] = (wrapper, spec)
        log.info(f"loaded env: {spec.name} ({spec.image})")
    return out


async def _cold_start(sub, config, slots, skip: Skiplist) -> tuple[Challenger, Slot, int]:
    for baseline in BASELINES:
        if baseline in skip:
            continue
        log.info(f"cold start: trying baseline {baseline.model}@{baseline.revision}")
        slot = await _try_provision(slots, baseline, skip)
        if slot is not None:
            crown_block = await sub.get_current_block()
            log.info(f"cold start: baseline {baseline.model} seated at block {crown_block}")
            return baseline, slot, crown_block

    while True:
        queue = skip.filter(await get_challenger_queue(sub, config.netuid))
        if not queue:
            log.info("no miners registered, waiting 120s")
            await asyncio.sleep(120)
            continue
        for candidate in queue:
            log.info(f"cold start: trying uid {candidate.uid} ({candidate.model})")
            slot = await _try_provision(slots, candidate, skip)
            if slot is not None:
                crown_block = await sub.get_current_block()
                return candidate, slot, crown_block
        log.warning("no viable champion in queue, retrying in 120s")
        await asyncio.sleep(120)


async def run(config: Config, slots=None):
    import bittensor as bt
    for _name, _lg in logging.root.manager.loggerDict.items():
        if isinstance(_lg, logging.Logger) and _name.startswith("affine"):
            _lg.setLevel(logging.NOTSET)
    sub = Subtensor(config.subtensor_endpoint, config.subtensor_fallback)
    wallet = bt.Wallet(name=config.wallet_name, hotkey=config.hotkey_name)
    slots = slots or TargonSlots(config)
    envs = _load_envs(config)
    skip = Skiplist.from_env()

    champion, champion_slot, crown_block = await _cold_start(sub, config, slots, skip)
    weights_ok = await _maybe_set_weights(sub, wallet, config.netuid, champion)
    if not weights_ok:
        log.error("failed to set initial weights — will retry")

    running = True

    def _stop(sig, _):
        nonlocal running
        log.info(f"signal {sig}, shutting down")
        running = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    while running:
        try:
            if not weights_ok:
                weights_ok = await _maybe_set_weights(sub, wallet, config.netuid, champion, retries=1)

            queue = skip.filter(await get_challenger_queue(sub, config.netuid, champion.hotkey))

            if not queue:
                log.info("no challengers, sleeping 120s")
                await asyncio.sleep(120)
                continue

            for chall in queue:
                if not running:
                    break

                if not await health_ping(champion_slot.base_url):
                    log.warning("champion slot unhealthy, reprovisioning")
                    await slots.teardown(champion_slot)
                    new_slot = await _try_provision(slots, champion, skip=None)
                    if new_slot is None:
                        log.error("champion reprovision failed, retrying in 60s")
                        await asyncio.sleep(60)
                        break
                    champion_slot = new_slot
                    log.info(f"champion reprovisioned: {champion_slot.base_url}")

                current_block = await sub.get_current_block()
                k = compute_k(current_block - crown_block, config.k_init, config.k_final, config.k_halflife)
                log.info(f"--- duel: uid {champion.uid} vs uid {chall.uid} ({chall.model}) k={k:.2f} ---")

                verdict, chall_slot = await _run_duel_with_retry(
                    slots, envs, champion_slot, chall, config, k,
                    hotkey=wallet.hotkey.ss58_address, skip=skip,
                )

                if verdict is Verdict.CHALLENGER_WINS:
                    log.info(f"DETHRONED: uid {chall.uid} ({chall.model}) beats uid {champion.uid}")
                    await slots.teardown(champion_slot)
                    champion, champion_slot = chall, chall_slot
                    crown_block = await sub.get_current_block()
                    weights_ok = await _maybe_set_weights(sub, wallet, config.netuid, champion)
                    if not weights_ok:
                        log.error("failed to set weights after dethrone — will retry")
                    break
            else:
                log.info("queue exhausted, sleeping 120s")
                await asyncio.sleep(120)

        except (ConnectionError, OSError) as e:
            log.error(f"connection error in main loop: {e}", exc_info=True)
            await asyncio.sleep(60)
        except Exception as e:
            log.critical(f"unexpected error in main loop: {e}", exc_info=True)
            await asyncio.sleep(60)

    log.info("shutting down")
    await slots.teardown(champion_slot)
    for wrapper, _ in envs.values():
        try:
            await wrapper.cleanup()
        except Exception:
            pass
    await sub.close()


async def _maybe_set_weights(sub, wallet, netuid: int, champion: Challenger, retries: int = 3) -> bool:
    if champion.uid < 0:
        log.info(f"baseline champion {champion.model} — skipping set_weights")
        return True
    return await set_weights(sub, wallet, netuid, champion.uid, retries=retries)


async def _try_provision(slots, chall: Challenger, skip: Skiplist | None = None) -> Slot | None:
    try:
        return await slots.provision(chall.model, chall.revision)
    except SlotProvisionFailed as e:
        log.warning(f"provision failed uid {chall.uid} ({chall.model}@{chall.revision}): {e}")
        if skip is not None:
            skip.add(chall.model, chall.revision)
        return None
    except Exception as e:
        log.error(f"provision error uid {chall.uid}: {e}")
        return None


async def _run_duel_with_retry(
    slots, envs, champion_slot, chall, config, k, *, hotkey: str, skip: Skiplist | None = None,
) -> tuple[Verdict | None, Slot | None]:
    for attempt in range(MAX_DUEL_RETRIES + 1):
        if attempt > 0:
            if not await health_ping(champion_slot.base_url):
                log.warning("champion down between retries, aborting duel")
                return None, None

        # Only skiplist on the first attempt — if we provisioned once, the
        # artifact is runnable, and a later failure is probably transient infra.
        slot = await _try_provision(slots, chall, skip if attempt == 0 else None)
        if slot is None:
            return None, None

        try:
            verdict = await run_duel(
                envs, champion_slot, slot,
                max_tasks=config.max_tasks_per_env,
                tasks_per_batch=config.tasks_per_batch,
                k=k,
                nonce=chall.block,
                hotkey=hotkey,
            )
        except Exception as e:
            log.error(f"duel failed (attempt {attempt + 1}/{MAX_DUEL_RETRIES + 1}): {e}", exc_info=True)
            await slots.teardown(slot)
            if attempt < MAX_DUEL_RETRIES:
                log.info("retrying duel from scratch")
                continue
            return None, None

        if verdict is Verdict.CHALLENGER_WINS:
            return verdict, slot
        log.info(f"champion holds vs uid {chall.uid}")
        await slots.teardown(slot)
        return verdict, None

    return None, None
