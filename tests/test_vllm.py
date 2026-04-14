from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from affine.vllm import LocalSlots, Slot, TargonSlots, _get_url, _parse_field, health_check, health_ping


class _ClientFactory:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = 0

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url):
        self.calls += 1
        if self.responses:
            item = self.responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return SimpleNamespace(status_code=200)


@dataclass
class _Proc:
    returncode: int = 0
    stdout: bytes = b""
    stderr: bytes = b""
    delay: float = 0.0

    def __post_init__(self):
        self.killed = False

    async def communicate(self):
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.stdout, self.stderr

    def kill(self):
        self.killed = True


def test_parse_field_basic():
    assert _parse_field("App: abc123\nUrl: http://host:8000", "app") == "abc123"


def test_parse_field_case_insensitive():
    assert _parse_field("APP: abc123", "app") == "abc123"
    assert _parse_field("app: abc123", "App") == "abc123"


def test_parse_field_colon_in_value():
    assert _parse_field("Url: http://host:8000/v1", "url") == "http://host:8000/v1"


def test_parse_field_missing_raises():
    with pytest.raises(RuntimeError, match="field 'app' not found"):
        _parse_field("Url: http://host:8000", "app")


@pytest.mark.asyncio
async def test_health_ping_success():
    factory = _ClientFactory([SimpleNamespace(status_code=200)])
    with patch("affine.vllm.httpx.AsyncClient", factory):
        assert await health_ping("http://svc", timeout=1) is True


@pytest.mark.asyncio
async def test_health_ping_non_200():
    factory = _ClientFactory([SimpleNamespace(status_code=503)])
    with patch("affine.vllm.httpx.AsyncClient", factory):
        assert await health_ping("http://svc", timeout=1) is False


@pytest.mark.asyncio
async def test_health_ping_exception_returns_false():
    factory = _ClientFactory([RuntimeError("boom")])
    with patch("affine.vllm.httpx.AsyncClient", factory):
        assert await health_ping("http://svc", timeout=1) is False


@pytest.mark.asyncio
async def test_health_check_polls_until_healthy():
    factory = _ClientFactory([
        SimpleNamespace(status_code=503),
        SimpleNamespace(status_code=503),
        SimpleNamespace(status_code=200),
    ])
    with patch("affine.vllm.httpx.AsyncClient", factory):
        ok = await health_check("http://svc", timeout=5, interval=0)
    assert ok is True
    assert factory.calls == 3


@pytest.mark.asyncio
async def test_health_check_times_out():
    factory = _ClientFactory([SimpleNamespace(status_code=503), SimpleNamespace(status_code=503)])
    with patch("affine.vllm.httpx.AsyncClient", factory), patch(
        "affine.vllm.time.monotonic", side_effect=[0.0, 0.5, 1.0, 2.1]
    ):
        ok = await health_check("http://svc", timeout=2, interval=0)
    assert ok is False


@pytest.mark.asyncio
async def test_local_slots_lifecycle():
    slots = LocalSlots("http://a/v1", "http://b/v1")
    s1 = await slots.provision("m1", "r1")
    assert s1.base_url == "http://a/v1"
    s2 = await slots.provision("m2", "r2")
    assert s2.base_url == "http://b/v1"
    await slots.teardown(s1)
    s3 = await slots.provision("m3", "r3")
    assert s3.base_url == "http://a/v1"


@pytest.mark.asyncio
async def test_local_slots_exhaustion():
    slots = LocalSlots("http://a/v1", "http://b/v1")
    await slots.provision("m1", "r1")
    await slots.provision("m2", "r2")
    with pytest.raises(RuntimeError, match="no free local slots"):
        await slots.provision("m3", "r3")


@pytest.mark.asyncio
async def test_get_url_appends_v1_when_missing():
    proc = _Proc(returncode=0, stdout=b"Url: http://host:8000\n")
    with patch("affine.vllm.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        url = await _get_url("app-1", timeout=1)
    assert url == "http://host:8000/v1"


@pytest.mark.asyncio
async def test_get_url_keeps_existing_v1():
    proc = _Proc(returncode=0, stdout=b"Url: http://host:8000/v1\n")
    with patch("affine.vllm.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        url = await _get_url("app-1", timeout=1)
    assert url == "http://host:8000/v1"


@pytest.mark.asyncio
async def test_get_url_nonzero_exit_raises():
    proc = _Proc(returncode=1, stderr=b"bad")
    with patch("affine.vllm.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        with pytest.raises(RuntimeError, match="targon app failed"):
            await _get_url("app-1", timeout=1)


@pytest.mark.asyncio
async def test_get_url_timeout_on_spawn():
    async def _slow_spawn(*args, **kwargs):
        await asyncio.sleep(0.05)
        return _Proc()

    with patch("affine.vllm.asyncio.create_subprocess_exec", _slow_spawn):
        with pytest.raises(RuntimeError, match="timed out"):
            await _get_url("app-1", timeout=0.01)


@pytest.mark.asyncio
async def test_get_url_timeout_on_communicate_kills_process():
    proc = _Proc(delay=0.05)
    with patch("affine.vllm.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        with pytest.raises(RuntimeError, match="hung"):
            await _get_url("app-1", timeout=0.01)
    assert proc.killed is True


@pytest.mark.asyncio
async def test_targon_provision_success():
    slots = TargonSlots(config=SimpleNamespace())
    proc = _Proc(returncode=0, stdout=b"App: abc123\n")

    with patch("affine.vllm.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)), patch(
        "affine.vllm._get_url", AsyncMock(return_value="http://svc/v1")
    ):
        slot = await slots.provision("model/a", "rev1", timeout=1)

    assert slot == Slot(model="model/a", revision="rev1", base_url="http://svc/v1", slot_id="abc123")


@pytest.mark.asyncio
async def test_targon_provision_nonzero_exit_raises():
    slots = TargonSlots(config=SimpleNamespace())
    proc = _Proc(returncode=1, stderr=b"deploy failed")
    with patch("affine.vllm.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        with pytest.raises(RuntimeError, match="targon deploy failed"):
            await slots.provision("model/a", "rev1", timeout=1)


@pytest.mark.asyncio
async def test_targon_provision_hung_kills_process():
    slots = TargonSlots(config=SimpleNamespace())
    proc = _Proc(returncode=0, stdout=b"App: abc123\n", delay=0.05)

    with patch("affine.vllm.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        with pytest.raises(RuntimeError, match="targon deploy hung"):
            await slots.provision("model/a", "rev1", timeout=0.01)

    assert proc.killed is True


@pytest.mark.asyncio
async def test_targon_teardown_ignores_local_slots():
    slots = TargonSlots(config=SimpleNamespace())
    with patch("affine.vllm.asyncio.create_subprocess_exec", AsyncMock()) as spawn:
        await slots.teardown(Slot(model="m", revision="r", base_url="http://x", slot_id="local-http://x"))
    spawn.assert_not_called()


@pytest.mark.asyncio
async def test_targon_teardown_timeout_kills_process():
    slots = TargonSlots(config=SimpleNamespace())
    proc = _Proc(returncode=0)

    async def _timeout(coro, timeout):
        coro.close()
        raise asyncio.TimeoutError

    with patch("affine.vllm.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)), patch(
        "affine.vllm.asyncio.wait_for", side_effect=_timeout
    ):
        await slots.teardown(Slot(model="m", revision="r", base_url="http://x", slot_id="abc123"))

    assert proc.killed is True
