from __future__ import annotations
import asyncio
import logging
import os
import signal

import affinetes as af
import bittensor as bt

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
        if queue:
            champion = min(queue, key=lambda c: c.block)
            log.info(f"cold start: uid {champion.uid} ({champion.model})")
            slot = await slots.provision(champion.model, champion.revision)
            if await health_check(slot.base_url, config.health_check_timeout):
                crown_block = await sub.get_current_block()
                return champion, slot, crown_block
            await slots.teardown(slot)
            log.warning("cold start champion failed health check, retrying")
        else:
            log.info("no miners registered, waiting 120s")
        await asyncio.sleep(120)


async def run(config: Config, slots=None):
    sub = Subtensor(config.subtensor_endpoint, config.subtensor_fallback)
    wallet = bt.Wallet(name=config.wallet_name, hotkey=config.hotkey_name)
    slots = slots or TargonSlots(config)
    envs = _load_envs(config)

    champion, champion_slot, crown_block = await _cold_start(sub, config, slots)
    if not await set_weights(sub, wallet, config.netuid, champion.uid):
        log.error("failed to set initial weights — continuing, will retry on next dethrone")

    running = True

    def _stop(sig, _):
        nonlocal running
        log.info(f"signal {sig}, shutting down")
        running = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    while running:
        try:
            queue = await get_challenger_queue(sub, config.netuid, champion.hotkey)

            if not queue:
                log.info("no challengers, sleeping 120s")
                await asyncio.sleep(120)
                continue

            for chall in queue:
                if not running:
                    break

                if not await health_ping(champion_slot.base_url):
                    log.warning(f"champion endpoint down — attempting auto-dethrone to uid {chall.uid}")
                    new_slot = await _try_provision(slots, chall, config)
                    if new_slot is None:
                        continue
                    log.info(f"DETHRONED (default): uid {chall.uid} — champion was down")
                    await slots.teardown(champion_slot)
                    champion, champion_slot = chall, new_slot
                    crown_block = await sub.get_current_block()
                    if not await set_weights(sub, wallet, config.netuid, champion.uid):
                        log.error("failed to set weights after auto-dethrone")
                    break

                current_block = await sub.get_current_block()
                k = compute_k(current_block - crown_block, config.k_init, config.k_final, config.k_halflife)
                log.info(f"--- duel: uid {champion.uid} vs uid {chall.uid} ({chall.model}) k={k:.2f} ---")

                verdict, chall_slot = await _run_duel_with_retry(
                    slots, envs, champion_slot, chall, config, k,
                )

                if verdict is Verdict.CHALLENGER_WINS:
                    log.info(f"DETHRONED: uid {chall.uid} ({chall.model}) beats uid {champion.uid}")
                    await slots.teardown(champion_slot)
                    champion, champion_slot = chall, chall_slot
                    crown_block = await sub.get_current_block()
                    if not await set_weights(sub, wallet, config.netuid, champion.uid):
                        log.error("failed to set weights after dethrone — will retry next cycle")
                    break
            else:
                log.info("queue exhausted, sleeping 120s")
                await asyncio.sleep(120)

        except Exception as e:
            log.error(f"main loop error: {e}", exc_info=True)
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
    slots, envs, champion_slot, chall, config, k,
) -> tuple[Verdict | None, Slot | None]:
    for attempt in range(MAX_DUEL_RETRIES + 1):
        if attempt > 0 and not await health_ping(champion_slot.base_url):
            log.warning("champion down between retries — auto-dethrone")
            slot = await _try_provision(slots, chall, config)
            if slot is None:
                return None, None
            return Verdict.CHALLENGER_WINS, slot

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
