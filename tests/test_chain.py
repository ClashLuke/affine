from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from affine.chain import Challenger, Subtensor, get_challenger_queue, set_weights


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
    result = await set_weights(sub, wallet, netuid=1, champion_uid=0, retries=1)
    assert result is False


@pytest.mark.asyncio
async def test_set_weights_happy_path():
    sub = _mock_sub(set_weights_result=FakeExtrinsicResponse(success=True))
    wallet = AsyncMock()
    result = await set_weights(sub, wallet, netuid=1, champion_uid=0)
    assert result is True


@pytest.mark.asyncio
async def test_set_weights_retries_then_succeeds():
    fail = FakeExtrinsicResponse(success=False)
    ok = FakeExtrinsicResponse(success=True)
    sub = _mock_sub()
    sub.set_weights = AsyncMock(side_effect=[fail, fail, ok])
    wallet = AsyncMock()
    with patch("affine.chain.asyncio.sleep", new_callable=AsyncMock):
        result = await set_weights(sub, wallet, netuid=1, champion_uid=0, retries=3)
    assert result is True
    assert sub.set_weights.call_count == 3


@pytest.mark.asyncio
async def test_set_weights_invalid_uid():
    sub = _mock_sub(meta=FakeMeta(["hk0"]))
    wallet = AsyncMock()
    result = await set_weights(sub, wallet, netuid=1, champion_uid=5)
    assert result is False


@pytest.mark.diagnostic
@pytest.mark.xfail(reason="negative champion_uid currently indexes from end of hotkeys", strict=False)
@pytest.mark.asyncio
async def test_set_weights_rejects_negative_uid():
    sub = _mock_sub(meta=FakeMeta(["hk0", "hk1"]))
    wallet = AsyncMock()
    result = await set_weights(sub, wallet, netuid=1, champion_uid=-1)
    assert result is False


@pytest.mark.asyncio
async def test_set_weights_dry_run_skips_chain_call(monkeypatch):
    monkeypatch.setenv("AFFINE_DRY_RUN", "1")
    sub = _mock_sub()
    wallet = AsyncMock()
    result = await set_weights(sub, wallet, netuid=1, champion_uid=0)
    assert result is True
    sub.set_weights.assert_not_called()


@pytest.mark.asyncio
async def test_challenger_queue_parses_revealed_commitments():
    revealed = {
        "hk0": ((10, '{"model":"m0","revision":"r0"}'),),
        "hk2": ((20, '{"model":"m2","revision":"r2"}'),),
    }
    sub = _mock_sub(revealed=revealed)
    queue = await get_challenger_queue(sub, netuid=1)
    assert queue == [
        Challenger(uid=0, hotkey="hk0", model="m0", revision="r0", block=10),
        Challenger(uid=2, hotkey="hk2", model="m2", revision="r2", block=20),
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
    queue = await get_challenger_queue(sub, netuid=1)
    assert queue == [
        Challenger(uid=0, hotkey="hk0", model="new", revision="rev-new", block=30),
    ]


@pytest.mark.asyncio
async def test_challenger_queue_falls_back_to_raw_commitments_when_revealed_empty():
    raw = {
        "hk0": '{"model":"m0","revision":"r0"}',
        "hk2": '{"model":"m2","revision":"r2"}',
    }
    sub = _mock_sub(raw=raw, current_block=777)
    queue = await get_challenger_queue(sub, netuid=1)
    assert queue == [
        Challenger(uid=0, hotkey="hk0", model="m0", revision="r0", block=777),
        Challenger(uid=2, hotkey="hk2", model="m2", revision="r2", block=777),
    ]
    sub.get_all_commitments.assert_awaited_once_with(1)
    sub.get_current_block.assert_awaited_once()


@pytest.mark.asyncio
async def test_challenger_queue_sorts_candidates_by_block_ascending():
    revealed = {
        "hk0": ((30, '{"model":"m0","revision":"r0"}'),),
        "hk1": ((10, '{"model":"m1","revision":"r1"}'),),
        "hk2": ((20, '{"model":"m2","revision":"r2"}'),),
    }
    sub = _mock_sub(revealed=revealed)
    queue = await get_challenger_queue(sub, netuid=1)
    assert [c.uid for c in queue] == [1, 2, 0]


@pytest.mark.asyncio
async def test_challenger_queue_skips_malformed_entries():
    revealed = {
        "hk0": ((10, "not json"),),
        "hk1": ((20, '{"model":"m1"}'),),
        "hk2": ((30, '{"model":"m2","revision":"r2"}'),),
    }
    sub = _mock_sub(revealed=revealed)
    queue = await get_challenger_queue(sub, netuid=1)
    assert len(queue) == 1
    assert queue[0].hotkey == "hk2"


@pytest.mark.asyncio
async def test_challenger_queue_excludes_hotkey_in_raw_path():
    raw = {
        "hk0": '{"model":"m0","revision":"r0"}',
        "hk1": '{"model":"m1","revision":"r1"}',
    }
    sub = _mock_sub(meta=FakeMeta(["hk0", "hk1"]), revealed={}, raw=raw)
    queue = await get_challenger_queue(sub, netuid=1, exclude_hotkey="hk0")
    assert queue == [
        Challenger(uid=1, hotkey="hk1", model="m1", revision="r1", block=100),
    ]


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
async def test_subtensor_proxy_reconnects_and_retries_method():
    first = AsyncMock()
    first.initialize = AsyncMock()
    first.close = AsyncMock()
    first.ping = AsyncMock(side_effect=RuntimeError("boom"))

    second = AsyncMock()
    second.initialize = AsyncMock()
    second.ping = AsyncMock(return_value=42)

    with patch("bittensor.AsyncSubtensor", side_effect=[first, second]):
        sub = Subtensor("primary")
        out = await sub.ping()

    assert out == 42
    first.close.assert_awaited_once()
    assert first.ping.await_count == 1
    assert second.ping.await_count == 1


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
async def test_subtensor_connect_raises_when_all_endpoints_fail():
    primary = AsyncMock()
    primary.initialize = AsyncMock(side_effect=RuntimeError("down"))
    fallback = AsyncMock()
    fallback.initialize = AsyncMock(side_effect=RuntimeError("down"))

    with patch("bittensor.AsyncSubtensor", side_effect=[primary, fallback]):
        sub = Subtensor("primary", "fallback")
        with pytest.raises(ConnectionError, match="all subtensor endpoints unreachable"):
            await sub._connect()
