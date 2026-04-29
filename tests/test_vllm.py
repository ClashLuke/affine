from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from affine.vllm import LocalSlots, Slot, SlotProvisionFailed, TargonSlots, health_ping

HOTKEY = "test-hotkey"


class _ExecCalled(Exception):
    pass


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


def test_vllm_entrypoint_forwards_hf_revision(monkeypatch):
    import sys
    from affine import vllm_entrypoint

    captured = {}
    monkeypatch.setattr(sys, "argv", [
        "entrypoint", "--source", "hf", "--model", "model/a",
        "--revision", "sha", "--served-model-name", "model/a",
        "--", "--host", "0.0.0.0",
    ])
    def fake_execvp(prog, argv):
        captured.update(prog=prog, argv=argv)
        raise _ExecCalled
    monkeypatch.setattr(vllm_entrypoint.os, "execvp", fake_execvp)

    with pytest.raises(_ExecCalled):
        vllm_entrypoint.main()

    argv = captured["argv"]
    assert argv[:5] == ["python", "-m", "vllm.entrypoints.openai.api_server", "--model", "model/a"]
    assert argv[argv.index("--revision") + 1] == "sha"
    assert "--host" in argv


def test_vllm_entrypoint_restores_s3_without_revision_arg(monkeypatch, tmp_path):
    import sys
    from affine import vllm_entrypoint

    captured = {}
    monkeypatch.setenv("AFFINE_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", [
        "entrypoint", "--source", "s3", "--model", "model/a",
        "--revision", "sha", "--served-model-name", "model/a",
        "--manifest-key", "manifest.json", "--", "--host", "0.0.0.0",
    ])
    monkeypatch.setattr(vllm_entrypoint, "restore_from_env",
                        lambda manifest, dest: captured.update(manifest=manifest, dest=dest))
    def fake_execvp(prog, argv):
        captured.update(prog=prog, argv=argv)
        raise _ExecCalled
    monkeypatch.setattr(vllm_entrypoint.os, "execvp", fake_execvp)

    with pytest.raises(_ExecCalled):
        vllm_entrypoint.main()

    argv = captured["argv"]
    assert captured["manifest"] == "manifest.json"
    assert captured["dest"] == tmp_path / "model__a"
    assert argv[argv.index("--model") + 1] == str(tmp_path / "model__a")
    assert "--revision" not in argv


def test_targon_client_import_path():
    from targon.client.client import Client
    assert Client is not None


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
async def test_local_slots_cancel_during_health_returns_slot_to_pool():
    """Regression: a cancellation mid-await on health_ping must still return the
    URL to the free pool. Without try/except, the URL is popped but never
    re-appended → repeated cancels exhaust the pool and the validator stalls
    in 'no free local slots' on every subsequent provision."""
    import asyncio
    started = asyncio.Event()
    async def hang(_url, *a, **kw):
        started.set()
        await asyncio.sleep(60)
    with patch("affine.vllm._docker_host_ip", return_value=None), patch(
        "affine.vllm.health_ping", side_effect=hang
    ):
        slots = LocalSlots("http://a/v1", "http://b/v1")
        task = asyncio.create_task(slots.provision("m1", "r1"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
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
async def test_targon_hf_source_with_custom_image_runs_vllm_directly(monkeypatch):
    monkeypatch.setenv("AFFINE_VLLM_IMAGE", "registry/affine-vllm:sha")
    http = _FakeHttp(
        register_resp={"uid": "wl-hf"},
        state_responses=[{"urls": [{"port": 8000, "url": "http://rental.host"}]}],
    )
    factory = _ClientFactory([SimpleNamespace(status_code=200)])
    with patch("affine.vllm._http", return_value=http), patch(
        "affine.vllm.httpx.AsyncClient", factory,
    ), patch("affine.vllm.asyncio.sleep", AsyncMock()):
        await TargonSlots(SimpleNamespace(), hotkey=HOTKEY).provision(
            "model/a", "sha", timeout=10, source="hf",
        )

    body = http.posts[0][1]
    assert body["image"] == "registry/affine-vllm:sha"
    assert body["commands"] == ["python", "-m", "vllm.entrypoints.openai.api_server"]
    assert body["args"][:6] == ["--model", "model/a", "--revision", "sha", "--served-model-name", "model/a"]
    assert "--host" in body["args"]


@pytest.mark.asyncio
async def test_targon_payload_includes_private_registry_auth(monkeypatch):
    monkeypatch.setenv("AFFINE_VLLM_IMAGE", "registry.example.com/affine-vllm:sha")
    monkeypatch.setenv("AFFINE_VLLM_REGISTRY_SERVER", "registry.example.com")
    monkeypatch.setenv("AFFINE_VLLM_REGISTRY_USERNAME", "user")
    monkeypatch.setenv("AFFINE_VLLM_REGISTRY_PASSWORD", "pass")
    http = _FakeHttp(
        register_resp={"uid": "wl-auth"},
        state_responses=[{"urls": [{"port": 8000, "url": "http://rental.host"}]}],
    )
    factory = _ClientFactory([SimpleNamespace(status_code=200)])
    with patch("affine.vllm._http", return_value=http), patch(
        "affine.vllm.httpx.AsyncClient", factory,
    ), patch("affine.vllm.asyncio.sleep", AsyncMock()):
        await TargonSlots(SimpleNamespace(), hotkey=HOTKEY).provision(
            "model/a", "sha", timeout=10,
        )

    assert http.posts[0][1]["registry_auth"] == {
        "server": "registry.example.com",
        "username": "user",
        "password": "pass",
    }


@pytest.mark.asyncio
async def test_targon_registry_auth_requires_complete_config(monkeypatch):
    monkeypatch.setenv("AFFINE_VLLM_IMAGE", "registry.example.com/affine-vllm:sha")
    monkeypatch.setenv("AFFINE_VLLM_REGISTRY_SERVER", "registry.example.com")
    monkeypatch.delenv("AFFINE_VLLM_REGISTRY_USERNAME", raising=False)
    monkeypatch.delenv("AFFINE_VLLM_REGISTRY_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="AFFINE_VLLM_REGISTRY"):
        await TargonSlots(SimpleNamespace(), hotkey=HOTKEY).provision(
            "model/a", "sha", timeout=10,
        )


@pytest.mark.asyncio
async def test_targon_s3_source_requires_affine_vllm_image(monkeypatch):
    monkeypatch.delenv("AFFINE_VLLM_IMAGE", raising=False)
    with pytest.raises(RuntimeError, match="AFFINE_VLLM_IMAGE"):
        await TargonSlots(SimpleNamespace(), hotkey=HOTKEY).provision(
            "model/a", "rev1", timeout=1, source="s3", backup_manifest_key="[]",
        )


@pytest.mark.asyncio
async def test_targon_s3_source_uses_affine_image_args(monkeypatch):
    monkeypatch.setenv("AFFINE_VLLM_IMAGE", "registry/affine-vllm:sha")
    monkeypatch.setenv("R2_S3_ACCESS_KEY_ID", "ak")
    monkeypatch.setenv("R2_S3_SECRET_ACCESS_KEY", "sk")
    monkeypatch.setenv("R2_S3_BUCKET", "bucket")
    monkeypatch.setenv("R2_S3_ENDPOINT_URL", "https://acct.r2.cloudflarestorage.com")
    http = _FakeHttp(
        register_resp={"uid": "wl-s3"},
        state_responses=[{"urls": [{"port": 8000, "url": "http://rental.host"}]}],
    )
    factory = _ClientFactory([SimpleNamespace(status_code=200)])
    with patch("affine.vllm._http", return_value=http), patch(
        "affine.vllm.httpx.AsyncClient", factory,
    ), patch("affine.vllm.asyncio.sleep", AsyncMock()):
        await TargonSlots(SimpleNamespace(), hotkey=HOTKEY).provision(
            "model/a", "sha", timeout=10, source="s3", backup_manifest_key="[]",
        )

    body = http.posts[0][1]
    assert body["image"] == "registry/affine-vllm:sha"
    assert "commands" not in body
    assert body["args"][:8] == [
        "--source", "s3", "--model", "model/a", "--revision", "sha", "--served-model-name", "model/a",
    ]


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
async def test_targon_provision_cancel_during_deploy_deletes_workload():
    """Outer cancel between register-success and ready: rental allocated on Targon
    but caller never sees a Slot. Original code's `try:` opened *after* register —
    a cancel hitting the deploy await would raise out before any cleanup. With
    register inside the try, the except shield-deletes whichever uid was captured."""
    http = _FakeHttp(register_resp={"uid": "wl-cancel"})

    async def _post(path, json=None):
        http.posts.append((path, json))
        if path.endswith("/deploy"):
            raise asyncio.CancelledError("outer cancel")
        return http.register_resp or {}

    http._async_post = _post
    with patch("affine.vllm._http", return_value=http):
        with pytest.raises(asyncio.CancelledError):
            await TargonSlots(SimpleNamespace(), hotkey=HOTKEY).provision("model/a", "rev1", timeout=1)
    assert http.deletes == ["/tha/v2/workloads/wl-cancel"]


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


def test_targon_prefix_includes_namespace(monkeypatch):
    monkeypatch.setenv("AFFINE_NAMESPACE", "prod")
    prod = TargonSlots(SimpleNamespace(), hotkey="5Hotkey")._prefix
    monkeypatch.setenv("AFFINE_NAMESPACE", "shadow")
    shadow = TargonSlots(SimpleNamespace(), hotkey="5Hotkey")._prefix
    assert prod != shadow


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
