from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from affine.chain import Miner, Subtensor, get_miners, set_weights


@dataclass
class FakeExtrinsicResponse:
    success: bool = True
    message: str | None = None

    def __len__(self):
        return 2

    def __iter__(self):
        yield self.success
        yield self.message


@dataclass
class FakeMeta:
    hotkeys: list[str]


def _mock_sub(**overrides) -> AsyncMock:
    sub = AsyncMock()
    sub.metagraph = AsyncMock(return_value=overrides.get("meta", FakeMeta(["hk0", "hk1", "hk2"])))
    sub.get_all_revealed_commitments = AsyncMock(return_value=overrides.get("revealed", {}))
    sub.get_all_commitments = AsyncMock(return_value=overrides.get("raw", {}))
    sub.set_weights = AsyncMock(return_value=overrides.get("set_weights_result", FakeExtrinsicResponse(True)))
    sub.get_current_block = AsyncMock(return_value=overrides.get("current_block", 100))
    return sub


@pytest.mark.asyncio
async def test_set_weights_checks_success_field():
    sub = _mock_sub(set_weights_result=FakeExtrinsicResponse(success=False, message="rate limited"))
    wallet = AsyncMock()
    result = await set_weights(sub, wallet, netuid=1, winner_uid=0, retries=1)
    assert result is False


@pytest.mark.asyncio
async def test_set_weights_happy_path():
    sub = _mock_sub(set_weights_result=FakeExtrinsicResponse(success=True))
    wallet = AsyncMock()
    result = await set_weights(sub, wallet, netuid=1, winner_uid=0)
    assert result is True


@pytest.mark.asyncio
async def test_set_weights_retries_then_succeeds():
    fail = FakeExtrinsicResponse(success=False)
    ok = FakeExtrinsicResponse(success=True)
    sub = _mock_sub()
    sub.set_weights = AsyncMock(side_effect=[fail, fail, ok])
    wallet = AsyncMock()
    with patch("affine.chain.asyncio.sleep", new_callable=AsyncMock):
        result = await set_weights(sub, wallet, netuid=1, winner_uid=0, retries=3)
    assert result is True
    assert sub.set_weights.call_count == 3


@pytest.mark.asyncio
async def test_set_weights_does_not_retry_on_exception():
    """An exception can fire AFTER broadcast but before the result returns
    (transport drop during inclusion-wait). Retrying double-submits the
    extrinsic and burns weight quota. The outer loop owns retry cadence."""
    sub = _mock_sub()
    sub.set_weights = AsyncMock(side_effect=RuntimeError("transport drop after broadcast"))
    wallet = AsyncMock()
    with patch("affine.chain.asyncio.sleep", new_callable=AsyncMock):
        result = await set_weights(sub, wallet, netuid=1, winner_uid=0, retries=3)
    assert result is False
    assert sub.set_weights.call_count == 1


@pytest.mark.asyncio
async def test_set_weights_exception_returns_true_when_chain_already_landed():
    """Transport-drop AFTER broadcast: the extrinsic is on chain but the client
    didn't get the response. Re-broadcasting on next tick burns weight quota.
    Verify on chain — if our row already targets winner_uid, treat as success."""
    meta = FakeMeta(["hk_me", "hk1", "hk_winner"])
    meta.W = [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    sub = _mock_sub(meta=meta)
    sub.set_weights = AsyncMock(side_effect=RuntimeError("transport drop"))
    wallet = AsyncMock()
    wallet.hotkey.ss58_address = "hk_me"
    with patch("affine.chain.asyncio.sleep", new_callable=AsyncMock):
        result = await set_weights(sub, wallet, netuid=1, winner_uid=2, retries=3)
    assert result is True
    assert sub.set_weights.call_count == 1


@pytest.mark.asyncio
async def test_set_weights_exception_returns_false_when_chain_did_not_land():
    """Same exception path, but on-chain weights show the extrinsic didn't land.
    Outer loop's next iteration must retry."""
    meta = FakeMeta(["hk_me", "hk1", "hk_winner"])
    meta.W = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    sub = _mock_sub(meta=meta)
    sub.set_weights = AsyncMock(side_effect=RuntimeError("pre-broadcast network error"))
    wallet = AsyncMock()
    wallet.hotkey.ss58_address = "hk_me"
    with patch("affine.chain.asyncio.sleep", new_callable=AsyncMock):
        result = await set_weights(sub, wallet, netuid=1, winner_uid=2, retries=3)
    assert result is False


@pytest.mark.asyncio
async def test_set_weights_invalid_uid():
    sub = _mock_sub(meta=FakeMeta(["hk0"]))
    wallet = AsyncMock()
    result = await set_weights(sub, wallet, netuid=1, winner_uid=5)
    assert result is False


@pytest.mark.asyncio
async def test_set_weights_rejects_negative_uid():
    sub = _mock_sub(meta=FakeMeta(["hk0", "hk1"]))
    wallet = AsyncMock()
    result = await set_weights(sub, wallet, netuid=1, winner_uid=-1)
    assert result is False


@pytest.mark.asyncio
async def test_set_weights_dry_run_skips_chain_call(monkeypatch):
    monkeypatch.setenv("AFFINE_DRY_RUN", "1")
    sub = _mock_sub()
    wallet = AsyncMock()
    result = await set_weights(sub, wallet, netuid=1, winner_uid=0)
    assert result is True
    sub.set_weights.assert_not_called()


@pytest.mark.asyncio
async def test_set_weights_aborts_on_hotkey_mismatch():
    """Uid recycling: deregistered hotkey's slot reused by a fresh registration.
    set_weights with expected_hotkey != on-chain hotkey must abort, not weight
    the new occupant."""
    sub = _mock_sub(meta=FakeMeta(["hk_new", "hk1"]))
    wallet = AsyncMock()
    result = await set_weights(sub, wallet, netuid=1, winner_uid=0,
                               expected_hotkey="hk_old")
    assert result is False
    sub.set_weights.assert_not_called()


@pytest.mark.asyncio
async def test_set_weights_passes_when_hotkey_matches():
    sub = _mock_sub(meta=FakeMeta(["hk0", "hk1"]),
                    set_weights_result=FakeExtrinsicResponse(success=True))
    wallet = AsyncMock()
    result = await set_weights(sub, wallet, netuid=1, winner_uid=0,
                               expected_hotkey="hk0")
    assert result is True
    sub.set_weights.assert_awaited_once()


@pytest.mark.asyncio
async def test_challenger_queue_parses_revealed_commitments():
    revealed = {
        "hk0": ((10, '{"model":"m0","revision":"r0"}'),),
        "hk2": ((20, '{"model":"m2","revision":"r2"}'),),
    }
    sub = _mock_sub(revealed=revealed)
    queue = await get_miners(sub, netuid=1)
    assert queue == [
        Miner(uid=0, hotkey="hk0", model="m0", revision="r0", block=10),
        Miner(uid=2, hotkey="hk2", model="m2", revision="r2", block=20),
    ]


@pytest.mark.asyncio
async def test_challenger_queue_uses_latest_revealed_block_per_hotkey():
    revealed = {
        "hk0": (
            (20, '{"model":"old","revision":"rev-old"}'),
            (30, '{"model":"new","revision":"rev-new"}'),
        ),
    }
    sub = _mock_sub(meta=FakeMeta(["hk0"]), revealed=revealed)
    queue = await get_miners(sub, netuid=1)
    assert queue == [
        Miner(uid=0, hotkey="hk0", model="new", revision="rev-new", block=30),
    ]


@pytest.mark.asyncio
async def test_challenger_queue_falls_back_to_raw_commitments_when_revealed_empty():
    raw = {
        "hk0": '{"model":"m0","revision":"r0"}',
        "hk2": '{"model":"m2","revision":"r2"}',
    }
    sub = _mock_sub(raw=raw, current_block=777)
    queue = await get_miners(sub, netuid=1)
    assert set(queue) == {
        Miner(uid=0, hotkey="hk0", model="m0", revision="r0", block=777),
        Miner(uid=2, hotkey="hk2", model="m2", revision="r2", block=777),
    }
    sub.get_all_commitments.assert_awaited_once_with(1, block=777)
    sub.get_current_block.assert_awaited_once()


@pytest.mark.asyncio
async def test_challenger_queue_sorts_candidates_by_block_ascending():
    revealed = {
        "hk0": ((30, '{"model":"m0","revision":"r0"}'),),
        "hk1": ((10, '{"model":"m1","revision":"r1"}'),),
        "hk2": ((20, '{"model":"m2","revision":"r2"}'),),
    }
    sub = _mock_sub(revealed=revealed)
    queue = await get_miners(sub, netuid=1)
    assert [c.uid for c in queue] == [1, 2, 0]


@pytest.mark.asyncio
async def test_challenger_queue_skips_malformed_entries():
    revealed = {
        "hk0": ((10, "not json"),),
        "hk1": ((20, '{"model":"m1"}'),),
        "hk2": ((30, '{"model":"m2","revision":"r2"}'),),
    }
    sub = _mock_sub(revealed=revealed)
    queue = await get_miners(sub, netuid=1)
    assert len(queue) == 1
    assert queue[0].hotkey == "hk2"


@pytest.mark.asyncio
async def test_challenger_queue_raw_supersedes_revealed():
    """Raw is the most recent commitment per hotkey (could be unrevealed); revealed
    is the disclosed history. A miner who reveals v1 then commits v2 has v2 in raw
    and v1 in revealed — raw is canonical, otherwise the validator duels a stale
    identity until the new commit reveals."""
    revealed = {"hk0": ((10, '{"model":"m0","revision":"r0"}'),)}
    raw = {
        "hk0": '{"model":"m0_raw","revision":"r0_raw"}',  # supersedes revealed
        "hk1": '{"model":"m1","revision":"r1"}',          # only in raw
    }
    sub = _mock_sub(revealed=revealed, raw=raw, current_block=777)
    queue = await get_miners(sub, netuid=1)
    assert queue == [
        Miner(uid=0, hotkey="hk0", model="m0_raw", revision="r0_raw", block=777),
        Miner(uid=1, hotkey="hk1", model="m1", revision="r1", block=777),
    ]


@pytest.mark.asyncio
async def test_challenger_queue_rejects_adversarial_commitment_payloads():
    """Miner-controlled JSON. Non-dict, list-typed model/revision, or oversized
    strings must be discarded — propagating them as Miners crashes the loop on
    the first set/dict keyed by (model, revision)."""
    revealed = {
        "hk0": ((10, "[]"),),                                                # not a dict
        "hk1": ((20, '{"model":["x"],"revision":"r"}'),),                    # list-typed model
        "hk2": ((30, '{"model":"m","revision":42}'),),                       # int revision
        "hk3": ((40, '{"model":"' + "x" * 300 + '","revision":"r"}'),),      # oversized
        "hk4": ((50, '{"model":"good","revision":"r"}'),),                   # valid: must survive
    }
    sub = _mock_sub(meta=FakeMeta(["hk0", "hk1", "hk2", "hk3", "hk4"]),
                    revealed=revealed)
    queue = await get_miners(sub, netuid=1)
    assert [m.hotkey for m in queue] == ["hk4"]


@pytest.mark.asyncio
async def test_challenger_queue_skips_empty_revealed_entries():
    """Bittensor occasionally returns hotkeys with empty entry lists. max([])
    used to crash the whole loop; now we just skip the hotkey."""
    revealed = {
        "hk0": (),  # empty
        "hk1": ((20, '{"model":"m1","revision":"r1"}'),),
    }
    sub = _mock_sub(revealed=revealed)
    queue = await get_miners(sub, netuid=1)
    assert [m.hotkey for m in queue] == ["hk1"]


@pytest.mark.asyncio
async def test_challenger_queue_excludes_hotkey_in_raw_path():
    raw = {
        "hk0": '{"model":"m0","revision":"r0"}',
        "hk1": '{"model":"m1","revision":"r1"}',
    }
    sub = _mock_sub(meta=FakeMeta(["hk0", "hk1"]), revealed={}, raw=raw)
    queue = await get_miners(sub, netuid=1, exclude_hotkey="hk0")
    assert queue == [
        Miner(uid=1, hotkey="hk1", model="m1", revision="r1", block=100),
    ]


@pytest.mark.asyncio
async def test_get_miners_pins_all_reads_to_one_block():
    """Without block-pinning, metagraph/raw/revealed return inconsistent snapshots:
    a hotkey newly registered between calls appears in raw but not metagraph (silently
    dropped). Pinning to one block makes the registry self-consistent."""
    sub = _mock_sub(current_block=12345, raw={"hk0": '{"model":"m","revision":"r"}'})
    await get_miners(sub, netuid=7)
    sub.get_current_block.assert_awaited_once()
    sub.metagraph.assert_awaited_once_with(7, block=12345)
    sub.get_all_commitments.assert_awaited_once_with(7, block=12345)
    sub.get_all_revealed_commitments.assert_awaited_once_with(7, block=12345)


@pytest.mark.asyncio
async def test_get_miners_tiebreak_is_hotkey_hash_not_uid():
    """Multiple raw commits share one synthetic block; sorting by uid gives low-UID
    miners a structural advantage in the challenger queue. Tiebreak by hash of
    (hotkey, revision) removes the bias deterministically across all validators."""
    import hashlib
    raw = {f"hk{i}": '{"model":"m","revision":"r"}' for i in range(8)}
    sub = _mock_sub(meta=FakeMeta([f"hk{i}" for i in range(8)]), raw=raw, current_block=500)
    queue = await get_miners(sub, netuid=1)
    expected = sorted(range(8), key=lambda i: hashlib.sha256(f"hk{i}|r".encode()).digest())
    assert [m.uid for m in queue] == expected
    # Order is NOT uid-ascending (we'd be unlucky for hash to match uid order on 8 items).
    assert [m.uid for m in queue] != list(range(8))


@pytest.mark.asyncio
async def test_subtensor_proxy_awaits_coroutine_result():
    conn = AsyncMock()
    conn.initialize = AsyncMock()
    conn.echo = AsyncMock(return_value="ok")

    with patch("bittensor.AsyncSubtensor", return_value=conn):
        sub = Subtensor("primary")
        out = await sub.echo("x")

    assert out == "ok"
    conn.echo.assert_awaited_once_with("x")


@pytest.mark.asyncio
async def test_subtensor_proxy_reconnects_and_retries_safe_method():
    """Reads on the allow-list auto-reconnect on transport error — they are
    idempotent so a duplicate call is safe."""
    first = AsyncMock()
    first.initialize = AsyncMock()
    first.close = AsyncMock()
    first.get_current_block = AsyncMock(side_effect=RuntimeError("boom"))

    second = AsyncMock()
    second.initialize = AsyncMock()
    second.get_current_block = AsyncMock(return_value=42)

    with patch("bittensor.AsyncSubtensor", side_effect=[first, second]):
        sub = Subtensor("primary")
        out = await sub.get_current_block()

    assert out == 42
    first.close.assert_awaited_once()
    assert first.get_current_block.await_count == 1
    assert second.get_current_block.await_count == 1


@pytest.mark.asyncio
async def test_subtensor_proxy_does_not_retry_writes():
    """Mutations (e.g. set_weights) must NOT auto-retry: a transport error after
    a successful broadcast would silently double-submit on reconnect."""
    first = AsyncMock()
    first.initialize = AsyncMock()
    first.close = AsyncMock()
    first.set_weights = AsyncMock(side_effect=RuntimeError("transport-error-after-broadcast"))

    second = AsyncMock()
    second.initialize = AsyncMock()
    second.set_weights = AsyncMock(return_value="should not be called")

    with patch("bittensor.AsyncSubtensor", side_effect=[first, second]):
        sub = Subtensor("primary")
        with pytest.raises(RuntimeError, match="transport-error"):
            await sub.set_weights(uids=[0], weights=[1.0])

    assert first.set_weights.await_count == 1
    assert second.set_weights.await_count == 0
    first.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_subtensor_connect_uses_fallback_when_primary_fails():
    primary = AsyncMock()
    primary.initialize = AsyncMock(side_effect=RuntimeError("down"))

    fallback = AsyncMock()
    fallback.initialize = AsyncMock(return_value=None)

    with patch("bittensor.AsyncSubtensor", side_effect=[primary, fallback]):
        sub = Subtensor("primary", "fallback")
        conn = await sub._connect()

    assert conn is fallback
    primary.initialize.assert_awaited_once()
    fallback.initialize.assert_awaited_once()


@pytest.mark.asyncio
async def test_subtensor_proxy_concurrent_reconnect_only_opens_one_new_connection():
    """Two concurrent calls failing on the same transport must reconnect exactly
    once: the second caller sees self._sub already replaced and reuses it."""
    import asyncio
    boom = RuntimeError("boom")
    first = AsyncMock()
    first.initialize = AsyncMock()
    first.close = AsyncMock()
    first.get_current_block = AsyncMock(side_effect=boom)
    second = AsyncMock()
    second.initialize = AsyncMock()
    second.get_current_block = AsyncMock(return_value=42)

    with patch("bittensor.AsyncSubtensor", side_effect=[first, second]):
        sub = Subtensor("primary")
        a, b = await asyncio.gather(sub.get_current_block(), sub.get_current_block())

    assert a == 42 and b == 42
    first.close.assert_awaited_once()
    assert second.get_current_block.await_count == 2


@pytest.mark.asyncio
async def test_subtensor_connect_raises_when_all_endpoints_fail():
    primary = AsyncMock()
    primary.initialize = AsyncMock(side_effect=RuntimeError("down"))
    fallback = AsyncMock()
    fallback.initialize = AsyncMock(side_effect=RuntimeError("down"))

    with patch("bittensor.AsyncSubtensor", side_effect=[primary, fallback]):
        sub = Subtensor("primary", "fallback")
        with pytest.raises(ConnectionError, match="all subtensor endpoints unreachable"):
            await sub._connect()
