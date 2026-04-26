from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from affine.vllm import LocalSlots, Slot, SlotProvisionFailed, TargonSlots, health_ping, inference_ping

HOTKEY = "test-hotkey"


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

    async def post(self, url, json=None, headers=None):
        return await self.get(url)


@dataclass
class _FakeHttp:
    register_resp: dict | None = None
    deploy_resp: dict | None = None
    state_responses: list | None = None
    events_responses: list | None = None
    delete_resp: dict | None = None
    list_resp: dict | Exception | None = None

    def __post_init__(self):
        self.posts: list[tuple[str, dict | None]] = []
        self.gets: list[str] = []
        self.deletes: list[str] = []

    async def _async_post(self, path, json=None):
        self.posts.append((path, json))
        if path.endswith("/deploy"):
            return self.deploy_resp or {}
        return self.register_resp or {}

    async def _async_get(self, path):
        self.gets.append(path)
        if path.endswith("/events"):
            if not self.events_responses:
                return {"items": []}
            item = self.events_responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if path == "/tha/v2/workloads":
            if isinstance(self.list_resp, Exception):
                raise self.list_resp
            return self.list_resp or {"items": []}
        if not self.state_responses:
            return {"urls": []}
        item = self.state_responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def _async_delete(self, path):
        self.deletes.append(path)
        return self.delete_resp or {}


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
async def test_inference_ping_success():
    factory = _ClientFactory([SimpleNamespace(status_code=200)])
    with patch("affine.vllm.httpx.AsyncClient", factory):
        assert await inference_ping("http://svc", "m") is True


@pytest.mark.asyncio
async def test_inference_ping_gateway_5xx_is_false():
    """Targon's failure mode: /models returns 200 but /chat/completions returns 502.
    With retries enabled, three consecutive 502s confirm the slot is dead."""
    factory = _ClientFactory([SimpleNamespace(status_code=502)] * 3)
    with patch("affine.vllm.httpx.AsyncClient", factory), \
         patch("affine.vllm.asyncio.sleep", AsyncMock()):
        assert await inference_ping("http://svc", "m") is False


@pytest.mark.asyncio
async def test_inference_ping_exception_returns_false():
    factory = _ClientFactory([RuntimeError("net down")] * 3)
    with patch("affine.vllm.httpx.AsyncClient", factory), \
         patch("affine.vllm.asyncio.sleep", AsyncMock()):
        assert await inference_ping("http://svc", "m") is False


@pytest.mark.asyncio
async def test_inference_ping_429_then_success():
    """Transient Targon throttle should not tear down a healthy slot."""
    factory = _ClientFactory([
        SimpleNamespace(status_code=429),
        SimpleNamespace(status_code=200),
    ])
    with patch("affine.vllm.httpx.AsyncClient", factory), \
         patch("affine.vllm.asyncio.sleep", AsyncMock()):
        assert await inference_ping("http://svc", "m") is True


@pytest.mark.asyncio
async def test_inference_ping_4xx_no_retry():
    """Non-throttle 4xx (e.g. bad model name) is fatal — no point retrying."""
    factory = _ClientFactory([SimpleNamespace(status_code=400)])
    with patch("affine.vllm.httpx.AsyncClient", factory), \
         patch("affine.vllm.asyncio.sleep", AsyncMock()):
        assert await inference_ping("http://svc", "m") is False
    assert factory.calls == 1


@pytest.mark.asyncio
async def test_local_slots_lifecycle():
    with patch("affine.vllm._docker_host_ip", return_value=None), patch(
        "affine.vllm.health_ping", AsyncMock(return_value=True)
    ):
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
    with patch("affine.vllm._docker_host_ip", return_value=None), patch(
        "affine.vllm.health_ping", AsyncMock(return_value=True)
    ):
        slots = LocalSlots("http://a/v1", "http://b/v1")
        await slots.provision("m1", "r1")
        await slots.provision("m2", "r2")
        with pytest.raises(RuntimeError, match="no free local slots"):
            await slots.provision("m3", "r3")


@pytest.mark.asyncio
async def test_local_slots_unhealthy_raises_and_returns_slot_to_pool():
    with patch("affine.vllm._docker_host_ip", return_value=None), patch(
        "affine.vllm.health_ping", AsyncMock(return_value=False)
    ):
        slots = LocalSlots("http://a/v1", "http://b/v1")
        with pytest.raises(SlotProvisionFailed):
            await slots.provision("m1", "r1")
        assert len(slots._free) == 2


@pytest.mark.asyncio
async def test_targon_provision_registers_deploys_and_returns_slot_when_ready():
    http = _FakeHttp(
        register_resp={"uid": "wl-123"},
        state_responses=[{"urls": [{"port": 8000, "url": "http://rental.host"}]}],
    )
    factory = _ClientFactory([SimpleNamespace(status_code=200)])
    with patch("affine.vllm._http", return_value=http), patch(
        "affine.vllm.httpx.AsyncClient", factory
    ), patch("affine.vllm.asyncio.sleep", AsyncMock()):
        slot = await TargonSlots(SimpleNamespace(), hotkey=HOTKEY).provision("model/a", "rev1", timeout=10)

    assert slot.slot_id == "wl-123"
    assert slot.base_url == "http://rental.host/v1"
    register_path, register_body = http.posts[0]
    deploy_path, _ = http.posts[1]
    assert register_path == "/tha/v2/workloads"
    assert register_body["type"] == "RENTAL"
    assert register_body["image"].startswith("vllm/")
    assert register_body["resource_name"] == "h200-small"
    assert register_body["ports"] == [{"port": 8000, "protocol": "TCP"}]
    assert "--enable-prefix-caching" in register_body["args"]
    assert deploy_path == "/tha/v2/workloads/wl-123/deploy"


@pytest.mark.asyncio
async def test_targon_provision_missing_uid_raises_runtime_error():
    http = _FakeHttp(register_resp={})
    with patch("affine.vllm._http", return_value=http):
        with pytest.raises(RuntimeError, match="no uid"):
            await TargonSlots(SimpleNamespace(), hotkey=HOTKEY).provision("model/a", "rev1", timeout=1)


@pytest.mark.asyncio
async def test_targon_provision_crashloop_raises_slot_provision_failed_and_deletes():
    http = _FakeHttp(
        register_resp={"uid": "wl-123"},
        events_responses=[{"items": [{"event_type": "POD_CRASHLOOP_BACKOFF"}]}],
    )
    with patch("affine.vllm._http", return_value=http), patch(
        "affine.vllm.asyncio.sleep", AsyncMock()
    ):
        with pytest.raises(SlotProvisionFailed, match="crashlooping"):
            await TargonSlots(SimpleNamespace(), hotkey=HOTKEY).provision("model/a", "rev1", timeout=10)
    assert http.deletes == ["/tha/v2/workloads/wl-123"]


@pytest.mark.asyncio
async def test_targon_provision_timeout_raises_timeout_error_and_deletes():
    http = _FakeHttp(register_resp={"uid": "wl-123"}, state_responses=[{"urls": []}] * 10)
    with patch("affine.vllm._http", return_value=http), patch(
        "affine.vllm.asyncio.sleep", AsyncMock()
    ), patch("affine.vllm.time.monotonic", side_effect=[0.0, 0.1, 999.0] + [999.0] * 20):
        with pytest.raises(TimeoutError, match="not ready"):
            await TargonSlots(SimpleNamespace(), hotkey=HOTKEY).provision("model/a", "rev1", timeout=1)
    assert http.deletes == ["/tha/v2/workloads/wl-123"]


@pytest.mark.asyncio
async def test_targon_provision_deploy_failure_deletes_workload():
    http = _FakeHttp(register_resp={"uid": "wl-123"})

    async def _boom(path, json=None):
        http.posts.append((path, json))
        if path.endswith("/deploy"):
            raise RuntimeError("deploy failed")
        return http.register_resp or {}

    http._async_post = _boom
    with patch("affine.vllm._http", return_value=http):
        with pytest.raises(RuntimeError, match="deploy failed"):
            await TargonSlots(SimpleNamespace(), hotkey=HOTKEY).provision("model/a", "rev1", timeout=1)
    assert http.deletes == ["/tha/v2/workloads/wl-123"]


@pytest.mark.asyncio
async def test_wait_for_ready_detects_crashloop_after_models_returns_ok_first():
    from affine.vllm import _wait_for_ready

    http = _FakeHttp(
        state_responses=[{"urls": [{"port": 8000, "url": "http://rental.host"}]}],
        events_responses=[
            {"items": []},
            {"items": [{"event_type": "POD_OOM_KILLED"}]},
        ],
    )
    factory = _ClientFactory([
        SimpleNamespace(status_code=503),
        SimpleNamespace(status_code=503),
    ])
    with patch("affine.vllm._http", return_value=http), patch(
        "affine.vllm.httpx.AsyncClient", factory
    ), patch("affine.vllm.asyncio.sleep", AsyncMock()):
        with pytest.raises(SlotProvisionFailed, match="crashlooping"):
            await _wait_for_ready("wl-1", timeout=10)


@pytest.mark.asyncio
async def test_targon_teardown_deletes_workload():
    http = _FakeHttp()
    with patch("affine.vllm._http", return_value=http):
        await TargonSlots(SimpleNamespace(), hotkey=HOTKEY).teardown(
            Slot(model="m", revision="r", base_url="http://x/v1", slot_id="wl-abc")
        )
    assert http.deletes == ["/tha/v2/workloads/wl-abc"]


@pytest.mark.asyncio
async def test_targon_teardown_ignores_local_slots():
    http = _FakeHttp()
    with patch("affine.vllm._http", return_value=http):
        await TargonSlots(SimpleNamespace(), hotkey=HOTKEY).teardown(
            Slot(model="m", revision="r", base_url="http://x", slot_id="local-http://x")
        )
    assert http.deletes == []


@pytest.mark.asyncio
async def test_targon_reconcile_scoped_to_hotkey_prefix():
    """Two validators on a shared Targon API key must not delete each other's
    workloads. Hotkey-derived prefix scopes reconcile."""
    me = TargonSlots(SimpleNamespace(), hotkey="5HotkeyA")
    them = TargonSlots(SimpleNamespace(), hotkey="5HotkeyB")
    assert me._prefix != them._prefix
    http = _FakeHttp(list_resp={"items": [
        {"uid": "mine-1",  "name": f"{me._prefix}-aaa-1"},
        {"uid": "mine-2",  "name": f"{me._prefix}-bbb-2"},
        {"uid": "their-1", "name": f"{them._prefix}-ccc-3"},
        {"uid": "old-af",  "name": "af-legacy-4"},
    ]})
    with patch("affine.vllm._http", return_value=http):
        n = await me.reconcile()
    assert n == 2
    assert sorted(http.deletes) == ["/tha/v2/workloads/mine-1", "/tha/v2/workloads/mine-2"]


def test_targon_prefix_targon_safe():
    """Targon names: lowercase alphanumeric + hyphens only. Substrate hotkeys
    contain uppercase, so the prefix must hash to lowercase hex."""
    import re
    s = TargonSlots(SimpleNamespace(), hotkey="5C4iP8XYZ")
    assert re.fullmatch(r"af[0-9a-f]{6}", s._prefix), s._prefix


@pytest.mark.asyncio
async def test_targon_reconcile_noop_when_nothing_stale():
    http = _FakeHttp(list_resp={"items": []})
    with patch("affine.vllm._http", return_value=http):
        assert await TargonSlots(SimpleNamespace(), hotkey=HOTKEY).reconcile() == 0
    assert http.deletes == []


@pytest.mark.asyncio
async def test_targon_reconcile_tolerates_list_failure():
    http = _FakeHttp(list_resp=RuntimeError("API down"))
    with patch("affine.vllm._http", return_value=http):
        assert await TargonSlots(SimpleNamespace(), hotkey=HOTKEY).reconcile() == 0
    assert http.deletes == []


@pytest.mark.asyncio
async def test_targon_teardown_swallows_errors():
    http = _FakeHttp()

    async def _boom(path):
        raise RuntimeError("API down")

    http._async_delete = _boom
    with patch("affine.vllm._http", return_value=http):
        await TargonSlots(SimpleNamespace(), hotkey=HOTKEY).teardown(
            Slot(model="m", revision="r", base_url="http://x/v1", slot_id="wl-xyz")
        )
