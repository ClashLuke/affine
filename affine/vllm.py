from __future__ import annotations
import asyncio
import hashlib
import logging
import os
import secrets
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx
from targon.client.client import Client

from .store import artifact_id as _artifact_id

if TYPE_CHECKING:
    from .backup import S3Config

log = logging.getLogger(__name__)


class SlotProvisionFailed(Exception):
    """Raised when a miner's artifact cannot be provisioned (crashloop, health timeout).

    Distinct from transient errors: callers should blacklist the (model, revision)
    that caused this rather than retrying.
    """


@dataclass(frozen=True)
class Slot:
    model: str
    revision: str
    base_url: str
    sidecar_url: str = ""
    slot_token: str = ""
    slot_id: str = ""
    name: str = ""


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


class LocalSlots:
    def __init__(self, champion_url: str, challenger_url: str):
        gw = _docker_host_ip()
        if gw:
            champion_url = champion_url.replace("localhost", gw).replace("127.0.0.1", gw)
            challenger_url = challenger_url.replace("localhost", gw).replace("127.0.0.1", gw)
        self._urls = [champion_url, challenger_url]
        self._free: list[str] = list(self._urls)

    async def provision(self, model: str, revision: str, **_kwargs) -> Slot:
        if not self._free:
            raise RuntimeError("no free local slots")
        url = self._free.pop(0)
        try:
            if not await health_ping(url):
                raise SlotProvisionFailed(f"local slot not healthy: {url}")
            log.info(f"slot ready after 0s: {url}")
            return Slot(model=model, revision=revision, base_url=url, slot_id=f"local-{url}", name=f"local-{url}")
        except BaseException:
            self._free.append(url)
            raise

    async def teardown(self, slot: Slot) -> None:
        if slot.base_url in self._urls and slot.base_url not in self._free:
            self._free.append(slot.base_url)


_WORKLOADS = "/tha/v2/workloads"
_CRASH_EVENTS = {"POD_CRASHLOOP_BACKOFF", "POD_BACK_OFF", "POD_FAILED", "POD_OOM_KILLED"}
_EVENTS_404_WARNED: set[str] = set()  # warn once per uid — _wait_for_ready hits this every 5s for up to 900s


def _vllm_image() -> str:
    image = os.getenv("AFFINE_VLLM_IMAGE", "").strip()
    if not image:
        raise RuntimeError("AFFINE_VLLM_IMAGE must be set when provisioning Targon vLLM slots")
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


async def _extract_urls(uid: str) -> dict[int, str]:
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


async def _wait_for_ready(uid: str, timeout: int, interval: float = 5.0,
                          *, require_vllm: bool = True) -> tuple[str | None, str | None]:
    """Poll state+events until the desired endpoints are reachable.

    `require_vllm=True` (default): both vLLM `/v1/models` and sidecar `/healthz`
    must return 200. Returns (base_url, sidecar_url).

    `require_vllm=False`: returns as soon as `/healthz` is reachable, even if
    vLLM hasn't started. Used for the s3-restore path where vLLM intentionally
    blocks on `/setup` until the validator hands over creds. Returns
    (base_url-or-None, sidecar_url).

    Crashloop detection runs throughout.

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
                urls = await _extract_urls(uid)
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
                log.info(f"slot ready after {time.monotonic() - t0:.0f}s: vllm={base_url} sidecar={sidecar_url} require_vllm={require_vllm}")
                return base_url, sidecar_url
            await asyncio.sleep(interval)
    raise TimeoutError(f"uid={uid} not ready within {timeout}s "
                       f"(vllm_ok={vllm_ok}, sidecar_ok={sidecar_ok})")


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


class TargonSlots:
    def __init__(self, config, hotkey: str, s3_configs: list["S3Config"] | None = None):
        self._config = config
        # S3 configs are kept for the in-container restore path: when a slot is
        # reprovisioned from a champion's existing backup, the slot needs creds
        # to read from S3 at startup. Upload-side creds are NOT injected into
        # env; they ride the /setup POST instead.
        self._s3_configs = list(s3_configs) if s3_configs else []
        # Lowercase-hex prefix scopes reconcile() to our own workloads when
        # multiple validators share a Targon API key. (Targon names are
        # lowercase-alphanumeric-plus-hyphens only; Substrate hotkeys aren't.)
        namespace = os.getenv("AFFINE_NAMESPACE", "prod").strip() or "prod"
        scope = namespace + "\0" + hotkey
        self._prefix = f"af{hashlib.sha256(scope.encode()).hexdigest()[:6]}"

    async def reconcile(self) -> int:
        """Delete stale workloads we own — matched by `{prefix}-` so a second
        validator on the same API key is untouched. Defends against leaks from
        prior SIGKILL/OOM/crash: the surviving validator cleans up its predecessor.
        """
        try:
            resp = await _http()._async_get(_WORKLOADS)
        except Exception as e:
            log.warning(f"reconcile: list workloads failed: {e}")
            return 0
        items = resp.get("items", []) if isinstance(resp, dict) else resp
        tag = f"{self._prefix}-"
        victims = [w["uid"] for w in (items or [])
                   if isinstance(w, dict) and str(w.get("name", "")).startswith(tag)
                   and w.get("uid")]
        if not victims:
            log.info(f"reconcile: no stale workloads ({tag}*)")
            return 0
        log.warning(f"reconcile: tearing down {len(victims)} stale workloads ({tag}*): {victims}")
        await asyncio.gather(*(self._delete(uid) for uid in victims))
        return len(victims)

    async def provision(
        self,
        model: str,
        revision: str,
        timeout: int | None = None,
        *,
        source: str = "hf",
        backup_manifest_key: str | None = None,
    ) -> Slot:
        if timeout is None:
            timeout = int(getattr(self._config, "provision_timeout", 900))
        # Random suffix so concurrent provisions of the same (model, revision) —
        # king & challenger sharing a popular base — don't collide on the workload name.
        h = hashlib.sha256(f"{model}\0{revision}".encode()).hexdigest()[:8]
        name = f"{self._prefix}-{h}-{secrets.token_hex(4)}"
        slot_token = secrets.token_urlsafe(32)
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
            # workspace (~394 MiB, allocated on first call), per-shape kernel
            # JIT, cudagraph slop. PR #30515 documents the profiling gap; the
            # 150 MiB built-in cushion is not enough under 256-wide cold bursts.
            "--gpu-memory-utilization", "0.85",
            # Steady-state concurrency ceiling for dwell_batch=512 streams.
            # Defaults (256/8192) cap us at half the dispatch. batched-tokens
            # is a per-step compute budget, not a memory knob — scheduler
            # preempts on KV pressure (vllm scheduler.py:504), it doesn't OOM.
            "--max-num-seqs", "1024",
            "--max-num-batched-tokens", "65536",
        ]
        image = _vllm_image()
        env = {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # PR #30515: fold cudagraph cost into startup memory profiling so the
            # KV pool is sized smaller and real headroom remains for FlashInfer
            # workspace + JIT arenas + transient prefill activations.
            "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
            # Sidecar reads this once at startup, pops it from os.environ. The
            # validator presents the same token in the Authorization header on
            # /setup. Targon never sees S3 creds.
            "AFFINE_SLOT_TOKEN": slot_token,
        }
        for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            v = os.environ.get(k)
            if v:
                env[k] = v
        if source not in ("hf", "s3"):
            raise ValueError(f"unknown vLLM source: {source}")
        if source == "s3" and not backup_manifest_key:
            raise ValueError("backup_manifest_key is required for source='s3'")
        commands = ["python", "-m", "affine.vllm_entrypoint"]
        args = [
            "--source", source,
            "--model", model,
            "--revision", revision,
            "--served-model-name", model,
        ]
        if source == "s3":
            args += ["--manifest-key", backup_manifest_key]
        args += ["--", *vllm_args]

        payload = {
            "type": "RENTAL",
            "name": name,
            "image": image,
            "resource_name": _resource(),
            "commands": commands,
            "args": args,
            "envs": [{"name": k, "value": v} for k, v in env.items()],
            "ports": [
                {"port": 8000, "protocol": "TCP"},
                {"port": 8001, "protocol": "TCP"},
            ],
        }
        registry_auth = _registry_auth()
        if registry_auth is not None:
            payload["registry_auth"] = registry_auth

        http = _http()
        log.info(f"targon rental register: name={name} image={image} resource={payload['resource_name']} model={model} source={source}")
        # Single try wrapping register+deploy+wait. The register call itself
        # creates the workload on Targon's side: a CancelledError delivered as
        # the post() returns, or a malformed response that triggers RuntimeError
        # below, would leave the rental allocated with no handle to delete it.
        # We pull uid out of `reg` inside the try so any exception path runs the
        # shielded delete with whatever uid we managed to capture.
        uid: str | None = None
        try:
            reg = await http._async_post(_WORKLOADS, json=payload)
            if isinstance(reg, dict):
                uid = reg.get("uid") or None
            if not uid:
                raise RuntimeError(f"rental register returned no uid: {reg!r}")
            log.info(f"targon rental deploy: uid={uid}")
            await http._async_post(f"{_WORKLOADS}/{uid}/deploy")
            log.info(f"targon rental uid={uid}; waiting for ready (timeout {timeout}s)")
            art = _artifact_id(model, revision)
            if source == "s3":
                _, sidecar_url = await _wait_for_ready(uid, timeout=timeout, require_vllm=False)
                log.info(f"targon rental uid={uid}; posting setup (restore)")
                await _post_setup(sidecar_url, slot_token, self._s3_configs,
                                  model=model, revision=revision, artifact_id=art,
                                  manifest_key=backup_manifest_key)
                base_url, sidecar_url = await _wait_for_ready(uid, timeout=timeout)
            else:
                base_url, sidecar_url = await _wait_for_ready(uid, timeout=timeout)
                if self._s3_configs:
                    log.info(f"targon rental uid={uid}; posting setup (backup-only)")
                    await _post_setup(sidecar_url, slot_token, self._s3_configs,
                                      model=model, revision=revision, artifact_id=art,
                                      manifest_key=None)
            await _warm_hit(base_url, model)
        except BaseException:
            # Shield: if the outer task is being cancelled (SIGTERM during a
            # parallel provision, _cancellable timeout), still finish deletion
            # so the Targon rental doesn't leak. _delete has its own 30s budget.
            if uid is not None:
                await asyncio.shield(self._delete(uid))
            raise
        log.info(f"targon slot ready: uid={uid} base_url={base_url} sidecar={sidecar_url}")
        return Slot(model=model, revision=revision, base_url=base_url,
                    sidecar_url=sidecar_url, slot_token=slot_token,
                    slot_id=uid, name=name)

    async def teardown(self, slot: Slot) -> None:
        if not slot.slot_id or slot.slot_id.startswith("local-"):
            return
        await self._delete(slot.slot_id)

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


async def poll_backup(slot: Slot) -> dict | None:
    """Returns the sidecar's current state dict, or None on transport error."""
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
