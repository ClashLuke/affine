from __future__ import annotations
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from affine.chain import Challenger, get_challenger_queue, set_weights


@dataclass
class FakeExtrinsicResponse:
    """Mirrors bittensor's ExtrinsicResponse: __len__ returns 2, no __bool__.
    This means bool(FakeExtrinsicResponse(success=False)) is True via __len__,
    which is exactly the bug we're regression-testing against."""
    success: bool = True
    message: str | None = None
    def __len__(self): return 2
    def __iter__(self):
        yield self.success
        yield self.message


@dataclass
class FakeMeta:
    hotkeys: list[str]


def _mock_sub(**overrides) -> AsyncMock:
    sub = AsyncMock()
    sub.metagraph = AsyncMock(return_value=overrides.get("meta", FakeMeta(["hk0", "hk1", "hk2"])))
    sub.get_all_revealed_commitments = AsyncMock(return_value=overrides.get("commits", {}))
    sub.set_weights = AsyncMock(return_value=overrides.get("set_weights_result", FakeExtrinsicResponse(True)))
    sub.get_current_block = AsyncMock(return_value=100)
    return sub


# --- set_weights ---

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


# --- get_challenger_queue ---

@pytest.mark.asyncio
async def test_challenger_queue_parses_commitments():
    commits = {
        "hk0": ((10, '{"model":"m0","revision":"r0"}'),),
        "hk2": ((20, '{"model":"m2","revision":"r2"}'),),
    }
    sub = _mock_sub(commits=commits)
    queue = await get_challenger_queue(sub, netuid=1)
    assert len(queue) == 2
    assert queue[0] == Challenger(uid=0, hotkey="hk0", model="m0", revision="r0", block=10)
    assert queue[1] == Challenger(uid=2, hotkey="hk2", model="m2", revision="r2", block=20)


@pytest.mark.asyncio
async def test_challenger_queue_skips_malformed():
    commits = {
        "hk0": ((10, "not json"),),
        "hk1": ((20, '{"model":"m1"}'),),  # missing revision
        "hk2": ((30, '{"model":"m2","revision":"r2"}'),),
    }
    sub = _mock_sub(commits=commits)
    queue = await get_challenger_queue(sub, netuid=1)
    assert len(queue) == 1
    assert queue[0].hotkey == "hk2"


@pytest.mark.asyncio
async def test_challenger_queue_excludes_hotkey():
    commits = {
        "hk0": ((10, '{"model":"m0","revision":"r0"}'),),
        "hk1": ((20, '{"model":"m1","revision":"r1"}'),),
    }
    sub = _mock_sub(commits=commits)
    queue = await get_challenger_queue(sub, netuid=1, exclude_hotkey="hk0")
    assert len(queue) == 1
    assert queue[0].hotkey == "hk1"
