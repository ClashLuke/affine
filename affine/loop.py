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
from .vllm import Slot, TargonSlots, health_check, health_ping

log = logging.getLogger(__name__)

MAX_DUEL_RETRIES = 2


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
        out[spec.name] = (wrapper, spec)
        log.info(f"loaded env: {spec.name} ({spec.image})")
    return out


async def _cold_start(sub, config, slots) -> tuple[Challenger, Slot, int]:
    while True:
        queue = await get_challenger_queue(sub, config.netuid)
        if not queue:
            log.info("no miners registered, waiting 120s")
            await asyncio.sleep(120)
            continue
        for candidate in queue:
            log.info(f"cold start: trying uid {candidate.uid} ({candidate.model})")
            try:
                slot = await slots.provision(candidate.model, candidate.revision)
            except Exception as e:
                log.warning(f"cold start provision failed uid {candidate.uid}: {e}")
                continue
            if await health_check(slot.base_url, config.health_check_timeout):
                crown_block = await sub.get_current_block()
                return candidate, slot, crown_block
            await slots.teardown(slot)
            log.warning(f"cold start health check failed uid {candidate.uid}")
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

    champion, champion_slot, crown_block = await _cold_start(sub, config, slots)
    weights_ok = await set_weights(sub, wallet, config.netuid, champion.uid)
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
                weights_ok = await set_weights(sub, wallet, config.netuid, champion.uid, retries=1)

            queue = await get_challenger_queue(sub, config.netuid, champion.hotkey)

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
                    new_slot = await _try_provision(slots, champion, config)
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
                    hotkey=wallet.hotkey.ss58_address,
                )

                if verdict is Verdict.CHALLENGER_WINS:
                    log.info(f"DETHRONED: uid {chall.uid} ({chall.model}) beats uid {champion.uid}")
                    await slots.teardown(champion_slot)
                    champion, champion_slot = chall, chall_slot
                    crown_block = await sub.get_current_block()
                    weights_ok = await set_weights(sub, wallet, config.netuid, champion.uid)
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


async def _try_provision(slots, chall: Challenger, config: Config) -> Slot | None:
    try:
        slot = await slots.provision(chall.model, chall.revision)
    except Exception as e:
        log.error(f"provision failed uid {chall.uid}: {e}")
        return None
    if not await health_check(slot.base_url, config.health_check_timeout):
        log.warning(f"health check failed uid {chall.uid}")
        await slots.teardown(slot)
        return None
    return slot


async def _run_duel_with_retry(
    slots, envs, champion_slot, chall, config, k, *, hotkey: str,
) -> tuple[Verdict | None, Slot | None]:
    for attempt in range(MAX_DUEL_RETRIES + 1):
        if attempt > 0:
            if not await health_ping(champion_slot.base_url):
                log.warning("champion down between retries, aborting duel")
                return None, None

        slot = await _try_provision(slots, chall, config)
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
