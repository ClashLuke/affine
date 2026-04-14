from __future__ import annotations
import asyncio
import logging
import os
import time
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)


@dataclass
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


class LocalSlots:
    def __init__(self, champion_url: str, challenger_url: str):
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


class TargonSlots:
    def __init__(self, config):
        self._config = config

    async def provision(self, model: str, revision: str, timeout: int = 600) -> Slot:
        env = {**os.environ, "MODEL_NAME": model, "MODEL_REVISION": revision}
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    "targon", "deploy", _SERVER_SCRIPT,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(f"targon deploy timed out after {timeout}s")

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(f"targon deploy hung after {timeout}s")
        if proc.returncode != 0:
            raise RuntimeError(f"targon deploy failed: {stderr.decode().strip()}")
        app_id = _parse_field(stdout.decode(), "app")
        base_url = await _get_url(app_id, timeout=60)
        return Slot(model=model, revision=revision, base_url=base_url, slot_id=app_id)

    async def teardown(self, slot: Slot) -> None:
        if not slot.slot_id or slot.slot_id.startswith("local-"):
            return
        proc = await asyncio.create_subprocess_exec(
            "targon", "app", "stop", slot.slot_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
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
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                "targon", "app", "get", app_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        raise RuntimeError(f"targon app get timed out after {timeout}s for {app_id}")

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError(f"targon app get hung after {timeout}s for {app_id}")
    if proc.returncode != 0:
        raise RuntimeError(f"targon app get failed for {app_id}: {stderr.decode().strip()}")
    url = _parse_field(stdout.decode(), "url")
    return url if url.endswith("/v1") else url.rstrip("/") + "/v1"


def _parse_field(output: str, field: str) -> str:
    field_lower = field.lower()
    for line in output.splitlines():
        parts = line.strip().split(":", 1)
        if len(parts) == 2:
            key = parts[0].lower().replace("_", " ").split()
            if key and key[0] == field_lower:
                val = parts[1].strip()
                if val:
                    return val
    raise RuntimeError(f"field '{field}' not found in targon output:\n{output}")
