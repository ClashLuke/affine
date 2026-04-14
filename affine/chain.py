from __future__ import annotations
import asyncio
import inspect
import json
import logging
import os
from dataclasses import dataclass

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subtensor wrapper — auto-reconnect on any failure, primary + fallback
# ---------------------------------------------------------------------------

class Subtensor:
    def __init__(self, endpoint: str = "finney", fallback: str | None = None):
        self._endpoint = endpoint
        self._fallback = fallback
        self._sub: bt.AsyncSubtensor | None = None
        self._lock = asyncio.Lock()

    async def _connect(self):
        import bittensor as bt
        for url in (self._endpoint, self._fallback):
            if url is None:
                continue
            try:
                s = bt.AsyncSubtensor(url)
                await s.initialize()
                log.info(f"subtensor connected: {url}")
                return s
            except Exception:
                log.warning(f"subtensor connect failed: {url}")
        raise ConnectionError("all subtensor endpoints unreachable")

    async def _ensure(self) -> bt.AsyncSubtensor:
        async with self._lock:
            if self._sub is None:
                self._sub = await self._connect()
            return self._sub

    async def _reconnect(self) -> bt.AsyncSubtensor:
        async with self._lock:
            if self._sub:
                try:
                    await self._sub.close()
                except Exception:
                    pass
            self._sub = await self._connect()
            return self._sub

    def __getattr__(self, name: str):
        async def _call(*args, **kwargs):
            sub = await self._ensure()
            try:
                result = getattr(sub, name)(*args, **kwargs)
                return await result if inspect.isawaitable(result) else result
            except Exception:
                log.debug(f"{name} failed, reconnecting")
                sub = await self._reconnect()
                result = getattr(sub, name)(*args, **kwargs)
                return await result if inspect.isawaitable(result) else result
        return _call

    async def close(self):
        async with self._lock:
            if self._sub:
                try:
                    await self._sub.close()
                except Exception:
                    pass
                self._sub = None


# ---------------------------------------------------------------------------
# Challenger queue from on-chain commitments
# ---------------------------------------------------------------------------

@dataclass
class Challenger:
    uid: int
    hotkey: str
    model: str
    revision: str
    block: int


async def get_challenger_queue(
    sub: Subtensor, netuid: int, exclude_hotkey: str | None = None,
) -> list[Challenger]:
    meta = await sub.metagraph(netuid)
    revealed = await sub.get_all_revealed_commitments(netuid)

    out: list[Challenger] = []

    if revealed:
        for uid in range(len(meta.hotkeys)):
            hotkey = meta.hotkeys[uid]
            if hotkey == exclude_hotkey or hotkey not in revealed:
                continue
            block, raw = max(revealed[hotkey], key=lambda x: x[0])
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            model, rev = data.get("model"), data.get("revision")
            if not model or not rev:
                continue
            out.append(Challenger(uid=uid, hotkey=hotkey, model=model, revision=rev, block=int(block)))
    else:
        raw_commits = await sub.get_all_commitments(netuid)
        current_block = await sub.get_current_block()
        for uid in range(len(meta.hotkeys)):
            hotkey = meta.hotkeys[uid]
            if hotkey == exclude_hotkey or hotkey not in raw_commits:
                continue
            raw = raw_commits[hotkey]
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            model, rev = data.get("model"), data.get("revision")
            if not model or not rev:
                continue
            out.append(Challenger(uid=uid, hotkey=hotkey, model=model, revision=rev, block=current_block))

    out.sort(key=lambda c: c.block)
    log.info(f"challenger queue: {len(out)} candidates")
    return out


# ---------------------------------------------------------------------------
# Winner-takes-all weight setting
# ---------------------------------------------------------------------------

async def set_weights(
    sub: Subtensor, wallet: bt.Wallet, netuid: int, champion_uid: int, retries: int = 3,
) -> bool:
    if os.getenv("AFFINE_DRY_RUN"):
        log.info(f"DRY RUN — would set weights: uid {champion_uid} = 1.0")
        return True
    meta = await sub.metagraph(netuid)
    if champion_uid >= len(meta.hotkeys) or not meta.hotkeys[champion_uid]:
        log.warning(f"champion uid {champion_uid} not in metagraph, skipping weight set")
        return False

    for attempt in range(retries):
        try:
            ok = await sub.set_weights(
                wallet=wallet,
                netuid=netuid,
                uids=[champion_uid],
                weights=[1.0],
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )
            log.debug(f"set_weights response: success={ok.success} message={ok.message}")
            if ok.success:
                log.info(f"weights set: uid {champion_uid} = 1.0")
                return True
            log.warning(f"set_weights rejected (attempt {attempt + 1}/{retries})")
        except Exception as e:
            log.error(f"set_weights error (attempt {attempt + 1}/{retries}): {e}")
        if attempt < retries - 1:
            await asyncio.sleep(60)
    return False
