from __future__ import annotations
import asyncio
import logging
import os
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
    elapsed = 0
    async with httpx.AsyncClient(timeout=10) as client:
        while elapsed < timeout:
            try:
                r = await client.get(f"{base_url}/models")
                if r.status_code == 200:
                    log.info(f"slot healthy after {elapsed}s: {base_url}")
                    return True
            except Exception:
                pass
            await asyncio.sleep(interval)
            elapsed += interval
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

    async def provision(self, model: str, revision: str) -> Slot:
        env = {**os.environ, "MODEL_NAME": model, "MODEL_REVISION": revision}
        proc = await asyncio.create_subprocess_exec(
            "targon", "deploy", _SERVER_SCRIPT,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"targon deploy failed: {stderr.decode().strip()}")
        app_id = _parse_field(stdout.decode(), "app")
        base_url = await _get_url(app_id)
        return Slot(model=model, revision=revision, base_url=base_url, slot_id=app_id)

    async def teardown(self, slot: Slot) -> None:
        if not slot.slot_id or slot.slot_id.startswith("local-"):
            return
        proc = await asyncio.create_subprocess_exec(
            "targon", "app", "stop", slot.slot_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        if proc.returncode != 0:
            log.warning(f"targon stop failed for {slot.slot_id}")


async def _get_url(app_id: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        "targon", "app", "get", app_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
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
