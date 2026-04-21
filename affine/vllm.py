from __future__ import annotations
import asyncio
import logging
import os
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
    """Docker bridge gateway IP, allowing containers to reach the host."""
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
        if not await health_ping(url):
            self._free.append(url)
            raise SlotProvisionFailed(f"local slot not healthy: {url}")
        return Slot(model=model, revision=revision, base_url=url, slot_id=f"local-{url}")

    async def teardown(self, slot: Slot) -> None:
        if slot.base_url in self._urls and slot.base_url not in self._free:
            self._free.append(slot.base_url)


_VLLM_IMAGE = os.getenv("AFFINE_VLLM_IMAGE", "vllm/vllm-openai:latest-cu130-ubuntu2404")
_RESOURCE = os.getenv("AFFINE_TARGON_RESOURCE", "h200-small")
_WORKLOADS = "/tha/v2/workloads"
_CRASH_EVENTS = {"POD_CRASHLOOP_BACKOFF", "POD_BACK_OFF", "POD_FAILED", "POD_OOM_KILLED"}


def _slot_name(model: str, revision: str) -> str:
    import hashlib
    h = hashlib.sha256(f"{model}\0{revision}".encode()).hexdigest()[:8]
    return f"af-{h}-{int(time.time()) % 1_000_000}"


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
    def __init__(self, config):
        self._config = config

    async def provision(self, model: str, revision: str, timeout: int = 900) -> Slot:
        name = _slot_name(model, revision)
        args = [
            "--model", model,
            "--revision", revision,
            "--served-model-name", model,
            "--host", "0.0.0.0",
            "--port", "8000",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--gpu-memory-utilization", "0.95",
        ]
        env = {"HF_HUB_ENABLE_HF_TRANSFER": "1"}
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
        reg = await http._async_post(_WORKLOADS, json=payload)
        if not isinstance(reg, dict) or not reg.get("uid"):
            raise RuntimeError(f"rental register returned no uid: {reg!r}")
        uid = reg["uid"]

        try:
            log.info(f"targon rental deploy: uid={uid}")
            await http._async_post(f"{_WORKLOADS}/{uid}/deploy")
            log.info(f"targon rental uid={uid}; waiting for ready (timeout {timeout}s)")
            base_url = await _wait_for_ready(uid, timeout=timeout)
        except BaseException:
            await self._delete(uid)
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
