"""Compute-provider abstraction for vLLM slots.

A `Slot` is a handle to a running vLLM (and sidecar) endpoint. The lifecycle —
allocate the platform handle, wait for readiness, optionally hand creds to the
sidecar, warm-hit vLLM — lives in `VllmSlots.provision`. Each subclass implements
four methods: `_allocate`, `_wait_ready`, `_release`, `reconcile`.

Each `Slot` carries its own `teardown` closure (captures the provider instance
and the platform handle), so dispatch is self-contained: callers just await
`slot.teardown()`. No `slot.provider` parsing, no per-provider routing table.

The `make_slots()` factory composes the chain from environment: Targon always,
Lium appended when `LIUM_API_KEY` is set. `loop._provision` iterates the chain on
infra-class failures; `SlotProvisionFailed` (miner-fault crashloop) propagates
without falling through.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, ClassVar

import httpx
from targon.client.client import Client

from .chain import _truthy_env
from .store import artifact_id as _artifact_id

if TYPE_CHECKING:
    from .backup import S3Config
    from .config import Config

log = logging.getLogger(__name__)


class SlotProvisionFailed(Exception):
    """Raised when a miner's artifact cannot be provisioned (crashloop, bad image).

    Distinct from infra-class failures: callers blacklist the (model, revision)
    that caused this, and the chain DOES NOT fall through — the same artifact
    would crashloop on every provider.
    """


async def _noop_teardown() -> None:
    return None


@dataclass(frozen=True)
class Slot:
    model: str
    revision: str
    base_url: str
    sidecar_url: str = ""
    slot_token: str = ""
    slot_id: str = ""
    name: str = ""
    provider: str = ""
    teardown: Callable[[], Awaitable[None]] = field(default=_noop_teardown)


async def health_ping(base_url: str, timeout: float = 5) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{base_url}/models")
            return r.status_code == 200
    except Exception:
        return False


def _docker_host_ip() -> str | None:
    """Override host IP for localhost rewrites in LocalSlots, e.g. when affine
    runs inside a container that needs to reach a vLLM on the host."""
    return os.getenv("AFFINE_DOCKER_GATEWAY", "").strip() or None


def _vllm_image() -> str:
    image = os.getenv("AFFINE_VLLM_IMAGE", "").strip()
    if not image:
        raise RuntimeError("AFFINE_VLLM_IMAGE must be set when provisioning vLLM slots")
    return image


def _registry_auth() -> dict[str, str] | None:
    auth = {
        "server": os.getenv("AFFINE_VLLM_REGISTRY_SERVER", "").strip(),
        "username": os.getenv("AFFINE_VLLM_REGISTRY_USERNAME", "").strip(),
        "password": os.getenv("AFFINE_VLLM_REGISTRY_PASSWORD", "").strip(),
    }
    if not any(auth.values()):
        return None
    if not all(auth.values()):
        raise RuntimeError(
            "AFFINE_VLLM_REGISTRY_SERVER, AFFINE_VLLM_REGISTRY_USERNAME, "
            "and AFFINE_VLLM_REGISTRY_PASSWORD must be set together"
        )
    return auth


def _resource() -> str:
    return os.getenv("AFFINE_TARGON_RESOURCE", "h200-small")


def _scope_prefix(hotkey: str) -> str:
    """Lowercase-hex prefix scoping reconcile to a single validator's resources
    when an API key is shared across validators. Targon names are lowercase
    [a-z0-9-]; Substrate hotkeys aren't, so we hash."""
    if not hotkey:
        return "aflocal"
    namespace = os.getenv("AFFINE_NAMESPACE", "prod").strip() or "prod"
    return f"af{hashlib.sha256((namespace + chr(0) + hotkey).encode()).hexdigest()[:6]}"


def _vllm_launch_spec(*, model: str, revision: str, source: str,
                      backup_manifest_key: str | None,
                      slot_token: str) -> dict:
    """Render the vllm_entrypoint command line + env. Provider-agnostic.

    Returns {"commands", "args", "env"}. Providers wrap this into platform-shaped
    payloads (Targon RENTAL workload; Lium template).
    """
    if source not in ("hf", "s3"):
        raise ValueError(f"unknown vLLM source: {source}")
    if source == "s3" and not backup_manifest_key:
        raise ValueError("backup_manifest_key is required for source='s3'")
    vllm_args = [
        "--host", "0.0.0.0",
        "--port", "8000",
        # Belt-and-suspenders code-execution guards. Asserted in
        # vllm_entrypoint.py before launch; both must survive any future
        # refactor untouched.
        "--trust-remote-code=False",
        "--load-format=safetensors",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        # 0.85 (was 0.90/0.95): leaves ~7 GiB on H200 for lazy first-burst
        # allocations the startup profiler underestimates — FlashInfer
        # workspace (~394 MiB), per-shape kernel JIT, cudagraph slop. PR
        # #30515 documents the profiling gap.
        "--gpu-memory-utilization", "0.85",
        # Steady-state concurrency ceiling for dwell_batch=512 streams.
        # Defaults (256/8192) cap us at half the dispatch.
        "--max-num-seqs", "1024",
        "--max-num-batched-tokens", "65536",
    ]
    env: dict[str, str] = {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # PR #30515: fold cudagraph cost into startup memory profiling so the
        # KV pool is sized smaller and real headroom remains for FlashInfer
        # workspace + JIT arenas + transient prefill activations.
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
        # Sidecar reads this once at startup, pops it from os.environ. The
        # validator presents the same token in the Authorization header on
        # /setup. Provider never sees S3 creds.
        "AFFINE_SLOT_TOKEN": slot_token,
    }
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if v := os.environ.get(k):
            env[k] = v
    commands = ["python", "-m", "affine.vllm_entrypoint"]
    args = [
        "--source", source,
        "--model", model,
        "--revision", revision,
        "--served-model-name", model,
    ]
    if source == "s3":
        args += ["--manifest-key", backup_manifest_key or ""]
    args += ["--", *vllm_args]
    return {"commands": commands, "args": args, "env": env}


def _derive_prefix(s3_configs: list["S3Config"], art: str) -> str:
    """Per-artifact prefix root. The slot adds `{provider_name}-{ts}` inside,
    so concurrent attempts on the same artifact don't collide.

    Validator passes one prefix to the slot for every provider; if operators set
    HIPPIUS_S3_PREFIX and R2_S3_PREFIX inconsistently, R2 uploads would land in
    Hippius's path and the manifest would be unreachable. Default config gives
    both providers the same prefix, so this only fires on a deliberate misconfig.
    """
    prefixes = {c.prefix for c in s3_configs}
    if len(prefixes) != 1:
        raise ValueError(f"backup providers must share a prefix, got {sorted(prefixes)}")
    return f"{next(iter(prefixes))}/artifacts/{art}"


async def _post_setup(sidecar_url: str, slot_token: str,
                      s3_configs: list["S3Config"], *,
                      model: str, revision: str, artifact_id: str,
                      manifest_key: str | None,
                      timeout: float = 3600.0) -> None:
    payload = {
        "providers": {c.name: c.payload() for c in s3_configs},
        "prefix": _derive_prefix(s3_configs, artifact_id),
        "model": model,
        "revision": revision,
        "artifact_id": artifact_id,
        "manifest_key": manifest_key,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            f"{sidecar_url}/setup",
            json=payload,
            headers={"Authorization": f"Bearer {slot_token}"},
        )
    if r.status_code != 204:
        raise RuntimeError(f"setup failed: {r.status_code} {r.text[:200]}")


async def _warm_hit(base_url: str, model: str, timeout: float = 120.0) -> None:
    """Fire a single 1-token completion before flipping the slot into rotation.

    `/v1/models` returning 200 means vLLM's busy loop is up — it does NOT mean
    the engine has done any inference. FlashInfer workspace (~394 MiB,
    vllm/v1/attention/backends/flashinfer.py:745) and per-shape kernel JIT both
    happen lazily on the first request. A 256-wide cold burst against an
    un-warmed pod has been seen to OOM-kill the engine on the first scheduler
    step (PR #30515 documents the profiling gap; PR #40383 documents the
    workspace overflow). One serialized warm hit forces those allocations on
    a non-load-bearing call.

    Treated as miner-fault on failure: a pod that can't serve a 1-token
    completion can't serve dwell traffic — surface as SlotProvisionFailed so
    the (model, revision) gets skiplisted instead of bouncing the validator.
    """
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                f"{base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                    "temperature": 0.0,
                },
            )
    except Exception as e:
        raise SlotProvisionFailed(f"warm hit failed for {base_url}: {type(e).__name__}: {e}") from e
    if r.status_code != 200:
        raise SlotProvisionFailed(f"warm hit returned {r.status_code} for {base_url}: {r.text[:200]}")
    log.info(f"warm hit ok in {time.monotonic() - t0:.1f}s: {base_url}")


# ---------------------------------------------------------------------------
# VllmSlots base
# ---------------------------------------------------------------------------


class VllmSlots:
    """Shared vLLM provision lifecycle. Subclass implements `_allocate`,
    `_wait_ready`, `_release`, and (optionally) `reconcile`/`aclose`. The
    sequence (build spec → allocate handle → wait for sidecar → /setup → wait
    for vllm → warm hit) lives here so a third provider only adds platform glue.
    """

    NAME: ClassVar[str] = ""

    def __init__(self, *, hotkey: str, s3_configs: list["S3Config"],
                 provision_timeout: float):
        self._s3_configs = list(s3_configs)
        self._scope = _scope_prefix(hotkey)
        self._provision_timeout = float(provision_timeout)

    async def provision(self, model: str, revision: str, *,
                        source: str = "hf",
                        backup_manifest_key: str | None = None) -> Slot:
        slot_token = secrets.token_urlsafe(32)
        spec = _vllm_launch_spec(
            model=model, revision=revision, source=source,
            backup_manifest_key=backup_manifest_key, slot_token=slot_token,
        )
        name = f"{self._scope}-{secrets.token_hex(4)}"
        handle: Any = None
        try:
            handle = await self._allocate(name, spec)
            base_url, sidecar_url = await self._wait_ready(handle, require_vllm=(source != "s3"))
            if self._s3_configs:
                if not sidecar_url:
                    raise SlotProvisionFailed(f"{self.NAME} slot has no sidecar; cannot run /setup")
                art = _artifact_id(model, revision)
                await _post_setup(
                    sidecar_url, slot_token, self._s3_configs,
                    model=model, revision=revision, artifact_id=art,
                    manifest_key=backup_manifest_key if source == "s3" else None,
                )
                if source == "s3":
                    base_url, sidecar_url = await self._wait_ready(handle, require_vllm=True)
            await _warm_hit(base_url, model)
            return Slot(
                model=model, revision=revision,
                base_url=base_url, sidecar_url=sidecar_url,
                slot_token=slot_token, slot_id=str(handle),
                name=name, provider=self.NAME,
                teardown=functools.partial(self._release, handle),
            )
        except BaseException:
            if handle is not None:
                # Shield: outer cancellation must not interrupt the platform
                # DELETE — the rental would otherwise leak until reconcile.
                await asyncio.shield(self._release(handle))
            raise

    # Subclass contract -----------------------------------------------------

    async def _allocate(self, name: str, spec: dict) -> Any:
        raise NotImplementedError

    async def _wait_ready(self, handle: Any, *, require_vllm: bool) -> tuple[str, str]:
        raise NotImplementedError

    async def _release(self, handle: Any) -> None:
        raise NotImplementedError

    async def reconcile(self) -> int:
        return 0

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------------------
# LocalSlots — fixed-URL stub provider (tests, AFFINE_LOCAL=1)
# ---------------------------------------------------------------------------


class LocalSlots(VllmSlots):
    NAME = "local"

    def __init__(self, champion_url: str, challenger_url: str):
        super().__init__(hotkey="", s3_configs=[], provision_timeout=10.0)
        gw = _docker_host_ip()
        if gw:
            champion_url = champion_url.replace("localhost", gw).replace("127.0.0.1", gw)
            challenger_url = challenger_url.replace("localhost", gw).replace("127.0.0.1", gw)
        self._urls = [champion_url, challenger_url]
        self._free: list[str] = list(self._urls)
        self._lock = asyncio.Lock()

    async def _allocate(self, name: str, spec: dict) -> str:
        async with self._lock:
            if not self._free:
                raise RuntimeError("no free local slots")
            return self._free.pop(0)

    async def _wait_ready(self, handle: str, *, require_vllm: bool) -> tuple[str, str]:
        if not await health_ping(handle):
            raise SlotProvisionFailed(f"local slot not healthy: {handle}")
        return handle, ""

    async def _release(self, handle: str) -> None:
        async with self._lock:
            if handle in self._urls and handle not in self._free:
                self._free.append(handle)


# ---------------------------------------------------------------------------
# TargonSlots
# ---------------------------------------------------------------------------


_WORKLOADS = "/tha/v2/workloads"
_CRASH_EVENTS = {"POD_CRASHLOOP_BACKOFF", "POD_BACK_OFF", "POD_FAILED", "POD_OOM_KILLED"}
_EVENTS_404_WARNED: set[str] = set()


def _http():
    return Client.async_serverless


async def _targon_crashed(uid: str) -> bool:
    try:
        ev = await _http()._async_get(f"{_WORKLOADS}/{uid}/events")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404 and uid not in _EVENTS_404_WARNED:
            _EVENTS_404_WARNED.add(uid)
            log.warning(f"events endpoint 404 for uid={uid}; crashloop detection disabled")
        return False
    except Exception:
        return False
    items = ev.get("items", []) if isinstance(ev, dict) else ev
    return any(e.get("event_type") in _CRASH_EVENTS for e in (items or []))


async def _extract_targon_urls(uid: str) -> dict[int, str]:
    """Map exposed-port → URL. Targon returns one entry per declared port; we
    pick by matching `port` (or `internal_port`) on each entry. Order-only
    fallback: when no port field is present, assume Targon listed entries in
    declaration order (vllm 8000, sidecar 8001)."""
    try:
        state = await _http()._async_get(f"{_WORKLOADS}/{uid}/state")
    except Exception as e:
        log.debug(f"get_state {uid} failed: {e}")
        return {}
    urls = state.get("urls") if isinstance(state, dict) else None
    if not isinstance(urls, list):
        return {}
    out: dict[int, str] = {}
    fallback_order = [8000, 8001]
    for i, u in enumerate(urls):
        if not isinstance(u, dict) or not u.get("url"):
            continue
        port = u.get("port") or u.get("internal_port") or u.get("target_port")
        if isinstance(port, int):
            out[port] = u["url"]
        elif i < len(fallback_order) and fallback_order[i] not in out:
            out[fallback_order[i]] = u["url"]
    return out


async def _wait_targon_ready(uid: str, timeout: int, interval: float = 5.0,
                             *, require_vllm: bool = True) -> tuple[str | None, str | None]:
    """Poll state+events until both vLLM `/v1/models` and sidecar `/healthz`
    return 200 (or just sidecar when require_vllm=False, for the s3-restore path).

    Raises `SlotProvisionFailed` ONLY on observed crashloop (miner-caused).
    Raises `TimeoutError` on deadline (could be infra — caller should not skiplist).
    """
    t0 = time.monotonic()
    base_url: str | None = None
    sidecar_url: str | None = None
    vllm_ok = False
    sidecar_ok = False
    async with httpx.AsyncClient(timeout=10) as client:
        while time.monotonic() - t0 < timeout:
            if await _targon_crashed(uid):
                raise SlotProvisionFailed(f"pod crashlooping after {time.monotonic() - t0:.0f}s: uid={uid}")
            if base_url is None or sidecar_url is None:
                urls = await _extract_targon_urls(uid)
                if base_url is None and 8000 in urls:
                    base_url = urls[8000].rstrip("/") + "/v1"
                if sidecar_url is None and 8001 in urls:
                    sidecar_url = urls[8001].rstrip("/")
            if base_url and not vllm_ok and require_vllm:
                try:
                    r = await client.get(f"{base_url}/models")
                    vllm_ok = r.status_code == 200
                except Exception:
                    pass
            if sidecar_url and not sidecar_ok:
                try:
                    r = await client.get(f"{sidecar_url}/healthz")
                    sidecar_ok = r.status_code == 200
                except Exception:
                    pass
            if sidecar_ok and (vllm_ok or not require_vllm):
                log.info(f"targon ready after {time.monotonic() - t0:.0f}s: vllm={base_url} sidecar={sidecar_url} require_vllm={require_vllm}")
                return base_url, sidecar_url
            await asyncio.sleep(interval)
    raise TimeoutError(f"uid={uid} not ready within {timeout}s "
                       f"(vllm_ok={vllm_ok}, sidecar_ok={sidecar_ok})")


class TargonSlots(VllmSlots):
    NAME = "targon"

    def __init__(self, *, hotkey: str, s3_configs: list["S3Config"], provision_timeout: float):
        super().__init__(hotkey=hotkey, s3_configs=s3_configs, provision_timeout=provision_timeout)
        self._image = _vllm_image()
        self._registry_auth = _registry_auth()

    async def reconcile(self) -> int:
        try:
            resp = await _http()._async_get(_WORKLOADS)
        except Exception as e:
            log.warning(f"targon reconcile: list workloads failed: {e}")
            return 0
        items = resp.get("items", []) if isinstance(resp, dict) else resp
        tag = f"{self._scope}-"
        victims = [w["uid"] for w in (items or [])
                   if isinstance(w, dict) and str(w.get("name", "")).startswith(tag)
                   and w.get("uid")]
        if not victims:
            log.info(f"targon reconcile: no stale workloads ({tag}*)")
            return 0
        log.warning(f"targon reconcile: tearing down {len(victims)} stale workloads ({tag}*): {victims}")
        await asyncio.gather(*(self._delete(uid) for uid in victims))
        return len(victims)

    async def _allocate(self, name: str, spec: dict) -> str:
        payload = {
            "type": "RENTAL",
            "name": name,
            "image": self._image,
            "resource_name": _resource(),
            "commands": spec["commands"],
            "args": spec["args"],
            "envs": [{"name": k, "value": v} for k, v in spec["env"].items()],
            "ports": [
                {"port": 8000, "protocol": "TCP"},
                {"port": 8001, "protocol": "TCP"},
            ],
        }
        if self._registry_auth is not None:
            payload["registry_auth"] = self._registry_auth
        log.info(f"targon rental register: name={name} image={self._image} resource={payload['resource_name']}")
        # Single try wrapping register+deploy. The register call itself creates
        # the workload on Targon's side; if the post returns and we then fail to
        # extract uid, we cannot delete it. So we capture uid as soon as it's
        # available and let the BaseException handler shielded-delete it.
        uid: str | None = None
        try:
            reg = await _http()._async_post(_WORKLOADS, json=payload)
            if isinstance(reg, dict):
                uid = reg.get("uid") or None
            if not uid:
                raise RuntimeError(f"rental register returned no uid: {reg!r}")
            log.info(f"targon rental deploy: uid={uid}")
            await _http()._async_post(f"{_WORKLOADS}/{uid}/deploy")
            return uid
        except BaseException:
            if uid is not None:
                await asyncio.shield(self._delete(uid))
            raise

    async def _wait_ready(self, handle: str, *, require_vllm: bool) -> tuple[str, str]:
        base_url, sidecar_url = await _wait_targon_ready(
            handle, timeout=int(self._provision_timeout), require_vllm=require_vllm,
        )
        return base_url or "", sidecar_url or ""

    async def _release(self, handle: str) -> None:
        await self._delete(handle)

    async def _delete(self, uid: str) -> None:
        try:
            await asyncio.wait_for(
                _http()._async_delete(f"{_WORKLOADS}/{uid}"),
                timeout=30,
            )
            log.info(f"targon teardown: uid={uid}")
        except asyncio.TimeoutError:
            log.warning(f"targon delete hung for {uid}")
        except Exception as e:
            log.warning(f"targon delete failed for {uid}: {e}")


# ---------------------------------------------------------------------------
# LiumSlots
# ---------------------------------------------------------------------------


def _resolve_ssh_pubkey() -> str:
    """`AFFINE_LIUM_SSH_PUBLIC_KEY` literal → ~/.ssh/id_ed25519.pub →
    id_rsa.pub → first ~/.ssh/*.pub."""
    explicit = os.getenv("AFFINE_LIUM_SSH_PUBLIC_KEY", "").strip()
    if explicit:
        return explicit
    home = Path.home() / ".ssh"
    for name in ("id_ed25519.pub", "id_rsa.pub"):
        p = home / name
        if p.is_file():
            return p.read_text().strip()
    if home.is_dir():
        for p in sorted(home.glob("*.pub")):
            return p.read_text().strip()
    raise RuntimeError(
        "no SSH public key found; set AFFINE_LIUM_SSH_PUBLIC_KEY or place a *.pub in ~/.ssh/"
    )


_LIUM_FAILED_WARNED: set[str] = set()
_LIUM_TERMINAL_FAIL = {"FAILED", "CRASHED", "ERROR", "TERMINATED", "STOPPED"}


class LiumSlots(VllmSlots):
    """Lium compute provider.

    Per-rent template strategy (per notes/lium-api-contract.md): `POST /rent` is
    not verified to honor env overrides, so we bake `AFFINE_SLOT_TOKEN` into a
    fresh template per rent and tear the template down with the pod. Cost: ~1s
    per provision; below the 5-15min provision floor.

    Crashloop detection is poll-only on `pod.status` (no events stream); the
    duel-side SLOT_DEAD mechanism is the backstop.
    """

    NAME = "lium"

    def __init__(self, *, hotkey: str, s3_configs: list["S3Config"],
                 provision_timeout: float, api_key: str,
                 image: str | None = None,
                 registry_auth: dict | None = None,
                 gpu_type: str | None = None,
                 base_url: str | None = None,
                 ssh_public_key: str | None = None):
        super().__init__(hotkey=hotkey, s3_configs=s3_configs, provision_timeout=provision_timeout)
        self._gpu_type = (gpu_type or os.getenv("AFFINE_LIUM_GPU_TYPE", "H200")).strip()
        self._image = image or _vllm_image()
        # registry_auth=None means env-derived; pass `{}` to explicitly disable.
        self._registry_auth = _registry_auth() if registry_auth is None else (registry_auth or None)
        self._ssh_pubkey = (ssh_public_key or _resolve_ssh_pubkey()).strip()
        self._http_client = httpx.AsyncClient(
            base_url=(base_url or os.getenv("AFFINE_LIUM_BASE_URL", "https://lium.io/api")).rstrip("/"),
            headers={"X-API-KEY": api_key, "User-Agent": "affine"},
            timeout=httpx.Timeout(60.0, connect=15.0, pool=10.0),
        )
        self._ssh_lock = asyncio.Lock()
        self._ssh_key_id: str | None = None
        self._templates: dict[str, str] = {}

    async def aclose(self) -> None:
        await self._http_client.aclose()

    # HTTP plumbing --------------------------------------------------------

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Lium rate-limits POST /templates and POST /rent at 5/min; one 429
        retry with the suggested backoff is enough to absorb a single burst.
        Other 4xx/5xx raise."""
        backoff = 5.0
        for attempt in range(3):
            r = await self._http_client.request(method, path, **kwargs)
            if r.status_code == 429:
                log.warning(f"lium 429 on {method} {path}; backing off {backoff:.0f}s (attempt {attempt+1})")
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            r.raise_for_status()
            if r.status_code == 204 or not r.content:
                return None
            return r.json()
        raise RuntimeError(f"lium {method} {path} rate-limited after retries")

    async def _list(self, path: str) -> list[dict]:
        data = await self._request("GET", path)
        return data if isinstance(data, list) else []

    async def _delete_pod(self, pod_id: str) -> None:
        try:
            r = await self._http_client.delete(f"/pods/{pod_id}")
        except Exception as e:
            log.warning(f"lium delete pod {pod_id}: {type(e).__name__}: {e}")
            return
        if r.status_code not in (200, 204, 404):
            log.warning(f"lium delete pod {pod_id}: {r.status_code} {r.text[:200]}")
        else:
            log.info(f"lium pod deleted: {pod_id} ({r.status_code})")

    async def _delete_template(self, template_id: str) -> None:
        try:
            r = await self._http_client.delete(f"/templates/{template_id}")
        except Exception as e:
            log.warning(f"lium delete template {template_id}: {type(e).__name__}: {e}")
            return
        if r.status_code not in (200, 204, 404):
            log.warning(f"lium delete template {template_id}: {r.status_code} {r.text[:200]}")

    async def _delete_ssh_key(self, key_id: str) -> None:
        try:
            r = await self._http_client.delete(f"/ssh-keys/{key_id}")
        except Exception as e:
            log.warning(f"lium delete ssh-key {key_id}: {type(e).__name__}: {e}")
            return
        if r.status_code not in (200, 204, 404):
            log.warning(f"lium delete ssh-key {key_id}: {r.status_code} {r.text[:200]}")

    # Provisioning ---------------------------------------------------------

    async def _ensure_ssh_key(self) -> None:
        """`POST /ssh-keys` is non-idempotent. List first, match by exact pubkey;
        register if missing. asyncio.Lock prevents same-process double-register
        when concurrent provisions race."""
        if self._ssh_key_id is not None:
            return
        async with self._ssh_lock:
            if self._ssh_key_id is not None:
                return
            keys = await self._list("/ssh-keys")
            for k in keys:
                if str(k.get("public_key", "")).strip() == self._ssh_pubkey:
                    self._ssh_key_id = str(k["id"])
                    log.info(f"lium ssh-key reused: {self._ssh_key_id}")
                    return
            data = await self._request(
                "POST", "/ssh-keys",
                json={"name": self._scope, "public_key": self._ssh_pubkey},
            )
            self._ssh_key_id = str(data["id"])
            log.info(f"lium ssh-key registered: {self._ssh_key_id}")

    async def _pick_executor(self) -> dict:
        """Filter by `\\b{gpu_type}\\b` against `machine_name`. The word-boundary
        regex matters: substring `H200` would match `GH200` (Grace Hopper)."""
        executors = await self._list("/executors")
        pattern = re.compile(rf"\b{re.escape(self._gpu_type)}\b")
        candidates = [
            e for e in executors
            if int(e.get("available_gpu_count") or 0) >= 1
            and pattern.search(str(e.get("machine_name", "")))
        ]
        if not candidates:
            raise RuntimeError(
                f"no Lium executors available for gpu_type={self._gpu_type!r} "
                f"(saw {len(executors)} total)"
            )
        candidates.sort(key=lambda e: float(e.get("price_per_gpu") or 0))
        pick = candidates[0]
        log.info(
            f"lium executor: {pick.get('id')} machine={pick.get('machine_name')} "
            f"price={pick.get('price_per_gpu')}/gpu avail={pick.get('available_gpu_count')}"
        )
        return pick

    def _template_payload(self, name: str, spec: dict) -> dict:
        """Build a Lium template body. Image, ports, env, and the entrypoint
        command line all bake in here; per-rent template means each provision
        gets a fresh AFFINE_SLOT_TOKEN."""
        image, _, suffix = self._image.rpartition(":")
        # rpartition gives ("", "", "...") if no ":". Suffix is a tag only when
        # the part after the last ":" has no "/" (`host:port/repo` has no tag).
        if image and "/" not in suffix:
            tag = suffix
        else:
            image, tag = self._image, "latest"
        body: dict[str, Any] = {
            "name": name,
            "docker_image": image,
            "docker_image_tag": tag,
            "category": "UBUNTU",
            "internal_ports": [8000, 8001],
            "environment": dict(spec["env"]),
            "entrypoint": list(spec["commands"]) + list(spec["args"]),
            "is_temporary": True,
            "container_start_immediately": True,
            "verify_ssh_connection": False,
        }
        if self._registry_auth and self._registry_auth.get("docker_credential_id"):
            body["docker_credential_id"] = self._registry_auth["docker_credential_id"]
        return body

    async def _create_template(self, name: str, spec: dict) -> str:
        data = await self._request("POST", "/templates", json=self._template_payload(name, spec))
        template_id = str(data["id"])
        log.info(f"lium template created: {template_id} ({name})")
        return template_id

    async def _rent(self, executor_id: str, template_id: str, name: str) -> str:
        data = await self._request(
            "POST", f"/executors/{executor_id}/rent",
            json={
                "pod_name": name,
                "template_id": template_id,
                "user_public_key": self._ssh_pubkey,
            },
        )
        # Rent returns the pod (or {pod: {...}} or {id: ...}); we just need an id.
        if isinstance(data, dict):
            for key in ("id", "pod_id"):
                if key in data:
                    return str(data[key])
            if isinstance(data.get("pod"), dict) and data["pod"].get("id"):
                return str(data["pod"]["id"])
        raise RuntimeError(f"lium rent returned no id: {data!r}")

    async def _allocate(self, name: str, spec: dict) -> str:
        await self._ensure_ssh_key()
        executor = await self._pick_executor()
        template_id = await self._create_template(name, spec)
        try:
            pod_id = await self._rent(str(executor["id"]), template_id, name)
        except BaseException:
            await asyncio.shield(self._delete_template(template_id))
            raise
        self._templates[pod_id] = template_id
        log.info(f"lium pod renting: {pod_id} on executor {executor.get('id')}")
        return pod_id

    async def _get_pod(self, pod_id: str) -> dict:
        data = await self._request("GET", f"/pods/{pod_id}")
        return data if isinstance(data, dict) else {}

    async def _wait_ready(self, handle: str, *, require_vllm: bool) -> tuple[str, str]:
        t0 = time.monotonic()
        base_url: str = ""
        sidecar_url: str = ""
        status: str = ""
        vllm_ok = False
        sidecar_ok = False
        async with httpx.AsyncClient(timeout=10) as client:
            while time.monotonic() - t0 < self._provision_timeout:
                pod = await self._get_pod(handle)
                status = str(pod.get("status", "")).upper()
                if status in _LIUM_TERMINAL_FAIL:
                    if handle not in _LIUM_FAILED_WARNED:
                        _LIUM_FAILED_WARNED.add(handle)
                        log.warning(f"lium pod {handle} status={status}")
                    raise SlotProvisionFailed(f"lium pod {handle} status={status}")
                if status == "RUNNING":
                    ports = pod.get("ports_mapping") or {}
                    ip = ((pod.get("executor") or {}).get("executor_ip_address")
                          or pod.get("executor_ip_address"))
                    p_vllm = ports.get("8000")
                    p_sidecar = ports.get("8001")
                    if not base_url and ip and p_vllm:
                        base_url = f"http://{ip}:{int(p_vllm)}/v1"
                    if not sidecar_url and ip and p_sidecar:
                        sidecar_url = f"http://{ip}:{int(p_sidecar)}"
                if base_url and require_vllm and not vllm_ok:
                    try:
                        r = await client.get(f"{base_url}/models")
                        vllm_ok = r.status_code == 200
                    except Exception:
                        pass
                if sidecar_url and not sidecar_ok:
                    try:
                        r = await client.get(f"{sidecar_url}/healthz")
                        sidecar_ok = r.status_code == 200
                    except Exception:
                        pass
                if sidecar_ok and (vllm_ok or not require_vllm):
                    log.info(f"lium ready after {time.monotonic() - t0:.0f}s: vllm={base_url} sidecar={sidecar_url} require_vllm={require_vllm}")
                    return base_url, sidecar_url
                await asyncio.sleep(5.0)
        raise TimeoutError(
            f"lium pod {handle} not ready within {self._provision_timeout:.0f}s "
            f"(status_seen={status!r}, vllm_ok={vllm_ok}, sidecar_ok={sidecar_ok})"
        )

    async def _release(self, handle: str) -> None:
        await self._delete_pod(handle)
        if (template_id := self._templates.pop(handle, None)) is not None:
            await self._delete_template(template_id)

    async def reconcile(self) -> int:
        """Sweep stray pods, templates, and ssh-key duplicates left by prior
        crashes. Pods + templates matched by name prefix `{scope}-`; ssh-keys
        deduped by public key (POST is non-idempotent — accumulates otherwise).
        """
        n = 0
        tag = f"{self._scope}-"
        for path, key_field in (("/pods", "pod_name"), ("/templates", "name")):
            try:
                items = await self._list(path)
            except Exception as e:
                log.warning(f"lium reconcile {path}: list failed: {e}")
                continue
            victims = [str(it["id"]) for it in items
                       if str(it.get(key_field, "")).startswith(tag) and it.get("id")]
            for vid in victims:
                if path == "/pods":
                    await self._delete_pod(vid)
                else:
                    await self._delete_template(vid)
            n += len(victims)
            if victims:
                log.warning(f"lium reconcile {path}: removed {len(victims)} ({tag}*)")
        try:
            keys = await self._list("/ssh-keys")
            ours = [k for k in keys if str(k.get("public_key", "")).strip() == self._ssh_pubkey]
            for k in ours[1:]:
                await self._delete_ssh_key(str(k["id"]))
                n += 1
            if len(ours) > 1:
                log.warning(f"lium reconcile /ssh-keys: removed {len(ours)-1} duplicate(s)")
        except Exception as e:
            log.warning(f"lium reconcile /ssh-keys failed: {e}")
        if n == 0:
            log.info(f"lium reconcile: no stale resources ({tag}*)")
        return n


# ---------------------------------------------------------------------------
# Slot helpers
# ---------------------------------------------------------------------------


async def poll_backup(slot: Slot) -> dict | None:
    """Returns the sidecar's current backup-state dict, or None on transport error."""
    if not slot.sidecar_url:
        return None
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{slot.sidecar_url}/backup")
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Chain factory
# ---------------------------------------------------------------------------


def make_slots(*, cfg: "Config", hotkey: str, s3_configs: list["S3Config"]) -> list[VllmSlots]:
    """Compose the provider chain. AFFINE_LOCAL=1 short-circuits to a single
    LocalSlots; otherwise Targon is the head and Lium is appended when
    LIUM_API_KEY is set. Adding a 3rd provider is one more `chain.append(...)`.
    """
    if _truthy_env("AFFINE_LOCAL"):
        champion = os.environ.get("CHAMPION_URL", "http://localhost:8000/v1")
        challenger = os.environ.get("CHALLENGER_URL", "http://localhost:8001/v1")
        return [LocalSlots(champion, challenger)]
    chain: list[VllmSlots] = [TargonSlots(
        hotkey=hotkey, s3_configs=s3_configs,
        provision_timeout=float(cfg.provision_timeout),
    )]
    if api_key := os.getenv("LIUM_API_KEY", "").strip():
        chain.append(LiumSlots(
            hotkey=hotkey, s3_configs=s3_configs,
            provision_timeout=float(os.getenv("AFFINE_LIUM_PROVISION_TIMEOUT", "1800")),
            api_key=api_key,
        ))
    log.info(f"compute providers: {' → '.join(s.NAME for s in chain)}")
    return chain
