from __future__ import annotations
import asyncio
import hashlib
import logging
import os
import secrets
import time
from dataclasses import dataclass

import httpx

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
    slot_id: str = ""


async def health_ping(base_url: str, timeout: float = 5) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{base_url}/models")
            return r.status_code == 200
    except Exception:
        return False


def _docker_host_ip() -> str | None:
    """Docker bridge gateway IP — only meaningful when affine runs inside a container
    that needs to reach a vLLM on the host. On the host itself, localhost works and
    the bridge IP would break a 127.0.0.1-bound vLLM."""
    if not os.path.exists("/.dockerenv"):
        return None
    try:
        import docker as _docker
        gw = _docker.from_env().networks.get("bridge").attrs["IPAM"]["Config"][0]["Gateway"]
        return gw if gw else None
    except Exception:
        return None


class LocalSlots:
    def __init__(self, champion_url: str, challenger_url: str):
        gw = _docker_host_ip()
        if gw:
            champion_url = champion_url.replace("localhost", gw).replace("127.0.0.1", gw)
            challenger_url = challenger_url.replace("localhost", gw).replace("127.0.0.1", gw)
        self._urls = [champion_url, challenger_url]
        self._free: list[str] = list(self._urls)

    async def provision(self, model: str, revision: str) -> Slot:
        if not self._free:
            raise RuntimeError("no free local slots")
        url = self._free.pop(0)
        try:
            if not await health_ping(url):
                raise SlotProvisionFailed(f"local slot not healthy: {url}")
            return Slot(model=model, revision=revision, base_url=url, slot_id=f"local-{url}")
        except BaseException:
            self._free.append(url)
            raise

    async def teardown(self, slot: Slot) -> None:
        if slot.base_url in self._urls and slot.base_url not in self._free:
            self._free.append(slot.base_url)


class FixedSlots:
    """Map (model, revision) → pre-existing vLLM URL. Local dev."""
    def __init__(self, urls: dict[tuple[str, str], str]):
        self._urls = urls

    async def provision(self, model: str, revision: str) -> Slot:
        url = self._urls.get((model, revision))
        if url is None:
            raise SlotProvisionFailed(f"no url for {model}@{revision}")
        if not await health_ping(url):
            raise SlotProvisionFailed(f"unhealthy: {url}")
        return Slot(model=model, revision=revision, base_url=url, slot_id=f"fixed-{url}")

    async def teardown(self, slot: Slot) -> None:
        pass


_VLLM_IMAGE = os.getenv("AFFINE_VLLM_IMAGE", "vllm/vllm-openai:latest-cu130-ubuntu2404")
_RESOURCE = os.getenv("AFFINE_TARGON_RESOURCE", "h200-small")
_WORKLOADS = "/tha/v2/workloads"
_CRASH_EVENTS = {"POD_CRASHLOOP_BACKOFF", "POD_BACK_OFF", "POD_FAILED", "POD_OOM_KILLED"}


def _http():
    from targon import Client
    return Client.async_serverless


async def _targon_crashed(uid: str) -> bool:
    try:
        ev = await _http()._async_get(f"{_WORKLOADS}/{uid}/events")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            log.warning(f"events endpoint 404 for uid={uid}; crashloop detection disabled")
        return False
    except Exception:
        return False
    items = ev.get("items", []) if isinstance(ev, dict) else ev
    return any(e.get("event_type") in _CRASH_EVENTS for e in (items or []))


async def _extract_url(uid: str) -> str | None:
    try:
        state = await _http()._async_get(f"{_WORKLOADS}/{uid}/state")
    except Exception as e:
        log.debug(f"get_state {uid} failed: {e}")
        return None
    urls = state.get("urls") if isinstance(state, dict) else None
    if not isinstance(urls, list):
        return None
    for u in urls:
        if isinstance(u, dict) and u.get("url"):
            return u["url"]
    return None


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


async def _wait_for_ready(uid: str, timeout: int, interval: float = 5.0) -> str:
    """Poll state+events+/models until the slot is ready. One unified loop so
    crashloops are detected before a URL is exposed and after /models starts
    responding (vLLM can 200 on /models while still initializing).

    Raises `SlotProvisionFailed` ONLY on observed crashloop (miner-caused).
    Raises `TimeoutError` on deadline (could be infra — caller should not skiplist).
    """
    t0 = time.monotonic()
    base_url: str | None = None
    async with httpx.AsyncClient(timeout=10) as client:
        while time.monotonic() - t0 < timeout:
            if await _targon_crashed(uid):
                raise SlotProvisionFailed(f"pod crashlooping after {time.monotonic() - t0:.0f}s: uid={uid}")
            if base_url is None:
                raw = await _extract_url(uid)
                if raw:
                    base_url = raw.rstrip("/") + "/v1"
            if base_url:
                try:
                    r = await client.get(f"{base_url}/models")
                    if r.status_code == 200:
                        log.info(f"slot ready after {time.monotonic() - t0:.0f}s: {base_url}")
                        return base_url
                except Exception:
                    pass
            await asyncio.sleep(interval)
    raise TimeoutError(f"uid={uid} not ready within {timeout}s")


class TargonSlots:
    def __init__(self, config, hotkey: str):
        self._config = config
        # Lowercase-hex prefix scopes reconcile() to our own workloads when
        # multiple validators share a Targon API key. (Targon names are
        # lowercase-alphanumeric-plus-hyphens only; Substrate hotkeys aren't.)
        self._prefix = f"af{hashlib.sha256(hotkey.encode()).hexdigest()[:6]}"

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

    async def provision(self, model: str, revision: str, timeout: int | None = None) -> Slot:
        if timeout is None:
            timeout = int(getattr(self._config, "provision_timeout", 900))
        # Random suffix so concurrent provisions of the same (model, revision) —
        # king & challenger sharing a popular base — don't collide on the workload name.
        h = hashlib.sha256(f"{model}\0{revision}".encode()).hexdigest()[:8]
        name = f"{self._prefix}-{h}-{secrets.token_hex(4)}"
        args = [
            "--model", model,
            "--revision", revision,
            "--served-model-name", model,
            "--host", "0.0.0.0",
            "--port", "8000",
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
        env = {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # PR #30515: fold cudagraph cost into startup memory profiling so the
            # KV pool is sized smaller and real headroom remains for FlashInfer
            # workspace + JIT arenas + transient prefill activations.
            "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
        }
        for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            v = os.environ.get(k)
            if v:
                env[k] = v

        payload = {
            "type": "RENTAL",
            "name": name,
            "image": _VLLM_IMAGE,
            "resource_name": _RESOURCE,
            "args": args,
            "envs": [{"name": k, "value": v} for k, v in env.items()],
            "ports": [{"port": 8000, "protocol": "TCP"}],
        }

        http = _http()
        log.info(f"targon rental register: name={name} image={_VLLM_IMAGE} resource={_RESOURCE} model={model}")
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
            base_url = await _wait_for_ready(uid, timeout=timeout)
            await _warm_hit(base_url, model)
        except BaseException:
            # Shield: if the outer task is being cancelled (SIGTERM during a
            # parallel provision, _cancellable timeout), still finish deletion
            # so the Targon rental doesn't leak. _delete has its own 30s budget.
            if uid is not None:
                await asyncio.shield(self._delete(uid))
            raise
        log.info(f"targon slot ready: uid={uid} base_url={base_url}")
        return Slot(model=model, revision=revision, base_url=base_url, slot_id=uid)

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
