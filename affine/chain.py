from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass

import bittensor as bt

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subtensor wrapper — auto-reconnect on any failure, primary + fallback
# ---------------------------------------------------------------------------

def _truthy_env(name: str) -> bool:
    """Strict bool from env var — '0', 'false', 'no', '' all disable. Without
    this, AFFINE_DRY_RUN=0 silently leaves the validator in dry-run mode."""
    v = os.getenv(name, "").strip().lower()
    return v not in ("", "0", "false", "no", "off")


class Subtensor:
    def __init__(self, endpoint: str = "finney", fallback: str | None = None):
        self._endpoint = endpoint
        self._fallback = fallback
        self._sub = None
        self._lock = asyncio.Lock()

    async def _connect(self):
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

    async def _ensure(self):
        async with self._lock:
            if self._sub is None:
                self._sub = await self._connect()
            return self._sub

    async def _reconnect(self, stale):
        """Reconnect ONLY if the caller's failed instance is still active. Two coros
        on a transient blip would otherwise both reconnect, the second tearing down
        the first's freshly-opened connection."""
        async with self._lock:
            if self._sub is stale:
                await self._close_sub()
                self._sub = await self._connect()
            return self._sub

    async def _close_sub(self):
        # Detach first, close second: if `close()` is cancelled (CancelledError is a
        # BaseException, not caught below), self._sub stays None either way, so a
        # subsequent call doesn't try to operate on a half-closed handle.
        sub, self._sub = self._sub, None
        if sub:
            try: await sub.close()
            except Exception: pass

    async def _call_retry(self, name: str, *args, **kwargs):
        """Auto-reconnect on failure and retry once. Use for read-only RPCs only —
        a transport drop after a successful mutation broadcast would silently
        double-submit on reconnect."""
        sub = await self._ensure()
        try:
            return await getattr(sub, name)(*args, **kwargs)
        except Exception as e:
            log.warning(f"subtensor.{name} failed ({type(e).__name__}: {e}); reconnecting")
            sub = await self._reconnect(stale=sub)
            return await getattr(sub, name)(*args, **kwargs)

    async def _call_no_retry(self, name: str, *args, **kwargs):
        """Raise on failure. Use for mutations: callers own their own retry loop
        with idempotency (nonce / version_key)."""
        sub = await self._ensure()
        return await getattr(sub, name)(*args, **kwargs)

    async def get_current_block(self):
        return await self._call_retry("get_current_block")

    async def metagraph(self, *args, **kwargs):
        return await self._call_retry("metagraph", *args, **kwargs)

    async def get_all_commitments(self, *args, **kwargs):
        return await self._call_retry("get_all_commitments", *args, **kwargs)

    async def get_all_revealed_commitments(self, *args, **kwargs):
        return await self._call_retry("get_all_revealed_commitments", *args, **kwargs)

    async def set_weights(self, **kwargs):
        return await self._call_no_retry("set_weights", **kwargs)

    async def close(self):
        async with self._lock:
            await self._close_sub()


# ---------------------------------------------------------------------------
# Miner registry from on-chain commitments
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Miner:
    uid: int | None
    hotkey: str
    model: str
    revision: str
    block: int


def _parse_commitments(
    meta, commits: dict[str, tuple[int, str]], exclude_hotkey: str | None,
) -> list[Miner]:
    """Parse miner-controlled commitment blobs. Adversarial: a malformed payload
    (non-dict JSON like `[]`, list-typed model/revision) must be discarded, not
    propagated as a Miner — downstream sets keyed on (model, revision) raise
    TypeError on unhashable list and crash the loop."""
    out = []
    for uid in range(len(meta.hotkeys)):
        hotkey = meta.hotkeys[uid]
        if hotkey == exclude_hotkey or hotkey not in commits:
            continue
        block, raw = commits[hotkey]
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        model, rev = data.get("model"), data.get("revision")
        if not (isinstance(model, str) and isinstance(rev, str) and model and rev):
            continue
        if len(model) > 256 or len(rev) > 256:
            continue
        out.append(Miner(uid=uid, hotkey=hotkey, model=model, revision=rev, block=int(block)))
    return out


def _tiebreak(m: Miner) -> bytes:
    """Deterministic, hotkey-keyed tiebreak for miners sharing a block. Without it,
    multiple raw commits share the same synthetic block and the stable sort falls
    back to insertion order (uid ascending) → low-UID miners systematically jump
    the queue. Hashing (hotkey, revision) gives every validator the same order
    while removing the UID bias."""
    return hashlib.sha256(f"{m.hotkey}|{m.revision}".encode()).digest()


async def get_miners(
    sub: Subtensor, netuid: int, exclude_hotkey: str | None = None,
) -> list[Miner]:
    """Return miners from on-chain commitments, all pinned to one block.

    `raw` is the most-recent commitment per hotkey (could be unrevealed);
    `revealed` is the disclosed history. A miner who reveals `v1` then commits
    `v2` has `v2` in raw and `v1` in revealed — raw is canonical. Falling back
    to revealed only when raw is silent for a hotkey ensures a re-commit is seen
    immediately, otherwise the validator duels a stale identity until the new
    commit reveals (ours-credit window the miner controls).

    All three reads pin to one block. Without pinning, a registration or
    reveal/commit rotation between calls produces an inconsistent registry —
    e.g. a hotkey present in `raw` but missing from `meta`, silently dropped.
    """
    block = await sub.get_current_block()
    meta = await sub.metagraph(netuid, block=block)
    raw = await sub.get_all_commitments(netuid, block=block)
    revealed = await sub.get_all_revealed_commitments(netuid, block=block)

    commits: dict[str, tuple[int, str]] = {}
    if raw:
        # Raw lacks a per-entry block; use the pinned block as monotonic sort key
        # (just-committed sorts after long-revealed, which is the right priority).
        for hk, data in raw.items():
            commits[hk] = (block, data)
    for hk, entries in (revealed or {}).items():
        if hk not in commits and entries:
            commits[hk] = max(entries, key=lambda x: x[0])

    out = _parse_commitments(meta, commits, exclude_hotkey)
    out.sort(key=lambda m: (m.block, _tiebreak(m)))
    log.info(f"miners: {len(out)} registered (pinned block {block})")
    return out


# ---------------------------------------------------------------------------
# Winner-takes-all weight setting
# ---------------------------------------------------------------------------

async def set_weights(
    sub: Subtensor, wallet, netuid: int, winner_uid: int,
    expected_hotkey: str | None = None, retries: int = 3,
) -> bool:
    if _truthy_env("AFFINE_DRY_RUN"):
        log.info(f"DRY RUN — would set weights: uid {winner_uid} = 1.0")
        return True
    meta = await sub.metagraph(netuid)
    if not 0 <= winner_uid < len(meta.hotkeys) or not meta.hotkeys[winner_uid]:
        log.warning(f"winner uid {winner_uid} not in metagraph, skipping weight set")
        return False
    # Uid recycling: a deregistered hotkey's slot can be taken over by a fresh
    # registration with a different hotkey. Without this guard we'd weight the
    # new occupant — a different miner — instead of who we elected. Caller
    # passes the hotkey we selected; mismatch aborts so the loop can burn.
    if expected_hotkey is not None and meta.hotkeys[winner_uid] != expected_hotkey:
        log.warning(
            f"winner uid {winner_uid} hotkey mismatch: expected {expected_hotkey[:8]} "
            f"got {meta.hotkeys[winner_uid][:8]}; uid was recycled"
        )
        return False

    # Retry only chain-level rejections (success=False — broadcast confirmed not
    # landing). Exceptions can fire AFTER broadcast (transport drop during
    # wait_for_inclusion finalization wait): the extrinsic IS on chain but the
    # client timed out waiting. Returning False makes the outer loop call us
    # again next tick → double-submit, burning weight quota. Verify on chain
    # before deciding: read our weight row; if it already targets winner_uid,
    # the broadcast landed.
    for attempt in range(retries):
        try:
            ok = await sub.set_weights(
                wallet=wallet, netuid=netuid,
                uids=[winner_uid], weights=[1.0],
                wait_for_inclusion=True, wait_for_finalization=True,
            )
        except Exception as e:
            log.error(f"set_weights raised (attempt {attempt + 1}/{retries}): {e}; verifying on chain")
            if await _confirm_on_chain(sub, netuid, wallet.hotkey.ss58_address, winner_uid):
                log.info(f"on-chain weight already targets uid {winner_uid}; treating as success")
                return True
            return False
        log.debug(f"set_weights response: success={ok.success} message={ok.message}")
        if ok.success:
            log.info(f"weights set: uid {winner_uid} = 1.0")
            return True
        log.warning(f"set_weights rejected (attempt {attempt + 1}/{retries}): {ok.message}")
        if attempt < retries - 1:
            await asyncio.sleep(60)
    return False


async def clear_weights(sub: Subtensor, wallet, netuid: int, retries: int = 3) -> bool:
    if _truthy_env("AFFINE_DRY_RUN"):
        log.info("DRY RUN — would clear weights (baseline/no payable winner)")
        return True
    for attempt in range(retries):
        try:
            ok = await sub.set_weights(
                wallet=wallet, netuid=netuid,
                # Bittensor converts an all-zero vector to empty dests/weights.
                # Passing a dummy uid avoids min([]) in the client-side converter.
                uids=[0], weights=[0.0],
                wait_for_inclusion=True, wait_for_finalization=True,
            )
        except Exception as e:
            log.error(f"clear_weights raised (attempt {attempt + 1}/{retries}): {e}; verifying on chain")
            if await _confirm_no_weights(sub, netuid, wallet.hotkey.ss58_address):
                log.info("on-chain weights already clear; treating as success")
                return True
            return False
        log.debug(f"clear_weights response: success={ok.success} message={ok.message}")
        if ok.success:
            if await _confirm_no_weights(sub, netuid, wallet.hotkey.ss58_address):
                log.info("weights cleared: no payable winner")
                return True
            log.warning("clear_weights accepted but on-chain row is not clear yet")
            return False
        log.warning(f"clear_weights rejected (attempt {attempt + 1}/{retries}): {ok.message}")
        if attempt < retries - 1:
            await asyncio.sleep(60)
    return False


async def _confirm_on_chain(sub: Subtensor, netuid: int, my_hotkey: str,
                             winner_uid: int, settle_s: float = 12.0) -> bool:
    """Best-effort: did our last set_weights actually land? Re-read metagraph and
    inspect our own weight row. Returns True iff the row exclusively concentrates
    on `winner_uid`. Sleep one block first so a just-broadcast extrinsic has time
    to be included before we read.
    """
    try:
        await asyncio.sleep(settle_s)
        meta = await sub.metagraph(netuid)
        my_uid = next((i for i, h in enumerate(meta.hotkeys) if h == my_hotkey), None)
        if my_uid is None:
            return False
        W = getattr(meta, "W", None)
        if W is None or my_uid >= len(W):
            return False
        row = W[my_uid]
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not row or winner_uid >= len(row):
            return False
        # Winner-takes-all: row[winner_uid] dominates. Threshold 0.99 since on-chain
        # encoding is u16-quantized so 1.0 round-trips to ~0.999985.
        return float(row[winner_uid]) >= 0.99
    except Exception as e:
        log.warning(f"on-chain weight verification failed: {e}")
        return False


async def _confirm_no_weights(sub: Subtensor, netuid: int, my_hotkey: str,
                              settle_s: float = 12.0) -> bool:
    try:
        await asyncio.sleep(settle_s)
        meta = await sub.metagraph(netuid)
        my_uid = next((i for i, h in enumerate(meta.hotkeys) if h == my_hotkey), None)
        if my_uid is None:
            return False
        W = getattr(meta, "W", None)
        if W is None or my_uid >= len(W):
            return False
        row = W[my_uid]
        if hasattr(row, "tolist"):
            row = row.tolist()
        return not row or max(float(w) for w in row) <= 1e-6
    except Exception as e:
        log.warning(f"on-chain clear-weight verification failed: {e}")
        return False
