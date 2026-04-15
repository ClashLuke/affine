from __future__ import annotations
import asyncio
import logging
import os
import time
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)


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


async def health_check(base_url: str, timeout: int = 300, interval: int = 5) -> bool:
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=10) as client:
        while time.monotonic() - t0 < timeout:
            try:
                r = await client.get(f"{base_url}/models")
                if r.status_code == 200:
                    log.info(f"slot healthy after {time.monotonic() - t0:.0f}s: {base_url}")
                    return True
            except Exception:
                pass
            await asyncio.sleep(interval)
    log.error(f"slot not healthy after {timeout}s: {base_url}")
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
        # Environment containers (Docker bridge network) can't reach localhost
        # on the host. Replace with the Docker bridge gateway IP.
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
        return Slot(model=model, revision=revision, base_url=url, slot_id=f"local-{url}")

    async def teardown(self, slot: Slot) -> None:
        if slot.base_url in self._urls and slot.base_url not in self._free:
            self._free.append(slot.base_url)


_SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "_targon_server.py")


async def _run_cmd(*args, timeout: int = 60, env=None) -> str:
    cmd = " ".join(str(a) for a in args[:2])
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *args, env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        raise RuntimeError(f"{cmd} timed out after {timeout}s")
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError(f"{cmd} hung after {timeout}s")
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd} failed: {stderr.decode().strip()}")
    return stdout.decode()


class TargonSlots:
    def __init__(self, config):
        self._config = config

    async def provision(self, model: str, revision: str, timeout: int = 600) -> Slot:
        env = {**os.environ, "MODEL_NAME": model, "MODEL_REVISION": revision}
        output = await _run_cmd("targon", "deploy", _SERVER_SCRIPT, timeout=timeout, env=env)
        app_id = _parse_field(output, "app")
        base_url = await _get_url(app_id)
        return Slot(model=model, revision=revision, base_url=base_url, slot_id=app_id)

    async def teardown(self, slot: Slot) -> None:
        if not slot.slot_id or slot.slot_id.startswith("local-"):
            return
        proc = await asyncio.create_subprocess_exec(
            "targon", "app", "stop", slot.slot_id,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            log.warning(f"targon stop hung for {slot.slot_id}")
            return
        if proc.returncode != 0:
            log.warning(f"targon stop failed for {slot.slot_id}")


async def _get_url(app_id: str, timeout: int = 60) -> str:
    output = await _run_cmd("targon", "app", "get", app_id, timeout=timeout)
    url = _parse_field(output, "url")
    return url if url.endswith("/v1") else url.rstrip("/") + "/v1"


def _parse_field(output: str, field: str) -> str:
    for line in output.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            if k.strip().lower() == field.lower() and v.strip():
                return v.strip()
    raise RuntimeError(f"field '{field}' not found in targon output:\n{output}")
