#!/usr/bin/env python3
"""Local development and testing CLI for affine."""

from __future__ import annotations
import asyncio
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import httpx
import typer

from affine.config import Config, ENV_REGISTRY, EnvSpec
from affine.duel import run_duel
from affine.loop import _load_envs, run
from affine.scoring import Verdict, compute_k
from affine.vllm import LocalSlots, Slot, health_ping

app = typer.Typer(help="Affine local development CLI.")
log = logging.getLogger(__name__)


def _detect_model_id(url: str) -> str:
    try:
        r = httpx.get(f"{url}/models", timeout=10)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"can't reach {url}/models — pass --model-id") from e
    for item in r.json().get("data", []):
        if isinstance(item, dict) and item.get("id"):
            return item["id"]
    raise RuntimeError(f"no model id in /models response from {url}")


def _url(raw: str) -> str:
    raw = raw.rstrip("/")
    return raw if raw.endswith("/v1") else f"{raw}/v1"


def _slot(model: str, revision: str, url: str, label: str) -> Slot:
    return Slot(model=model, revision=revision, base_url=url, slot_id=f"local-{label}")


def _resolve_envs(names: list[str], timeout: int = 60) -> tuple[EnvSpec, ...]:
    names = names or ["ded"]
    for n in names:
        if n not in ENV_REGISTRY:
            raise typer.BadParameter(f"unknown environment: {n}")
    return tuple(replace(ENV_REGISTRY[n], params={**ENV_REGISTRY[n].params, "timeout": timeout}) for n in names)


def _init_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("affine.duel").setLevel(logging.INFO)


async def _run_duel(config: Config, champion: Slot, challenger: Slot, progress_interval: int, hotkey: str = "") -> Verdict:
    envs = _load_envs(config)
    try:
        for s in [champion, challenger]:
            if not await health_ping(s.base_url):
                raise RuntimeError(f"not healthy: {s.base_url}")
        k = compute_k(0, config.k_init, config.k_final, config.k_halflife)
        verdict = await run_duel(
            envs, champion, challenger,
            max_tasks=config.max_tasks_per_env, tasks_per_batch=config.tasks_per_batch,
            k=k, hotkey=hotkey, progress_interval=progress_interval,
        )
        log.info(f"verdict: {verdict.name} | {champion.model} vs {challenger.model}")
        return verdict
    finally:
        for w, _ in envs.values():
            try:
                await w.cleanup()
            except Exception:
                pass


async def _run_queue(config: Config, champion: Slot, challengers: list[Slot], progress_interval: int, hotkey: str = "") -> Slot:
    envs = _load_envs(config)
    try:
        if not await health_ping(champion.base_url):
            raise RuntimeError(f"not healthy: {champion.base_url}")
        k = compute_k(0, config.k_init, config.k_final, config.k_halflife)
        for idx, chall in enumerate(challengers, 1):
            if not await health_ping(chall.base_url):
                log.warning(f"challenger {idx} unhealthy, skipping: {chall.base_url}")
                continue
            verdict = await run_duel(
                envs, champion, chall,
                max_tasks=config.max_tasks_per_env, tasks_per_batch=config.tasks_per_batch,
                k=k, nonce=idx, hotkey=hotkey, progress_interval=progress_interval,
            )
            log.info(f"queue duel {idx}: {verdict.name} | {champion.model} vs {chall.model}")
            if verdict is Verdict.CHALLENGER_WINS:
                champion = chall
        log.info(f"final champion: {champion.model} ({champion.base_url})")
        return champion
    finally:
        for w, _ in envs.values():
            try:
                await w.cleanup()
            except Exception:
                pass


def _load_queue_file(path: str, default_model: str | None, default_revision: str, default_url: str) -> list[Slot]:
    text = Path(path).read_text().strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        items = [data] if isinstance(data, dict) else data
    except json.JSONDecodeError:
        items = [json.loads(line) for line in text.splitlines() if line.strip()]

    slots = []
    for i, d in enumerate(items, 1):
        model = d.get("model") or d.get("model_id") or d.get("id") or default_model
        if not model:
            raise ValueError(f"queue entry {i} missing model")
        url = d.get("url") or d.get("base_url") or default_url
        if not url:
            raise ValueError(f"queue entry {i} missing url")
        rev = d.get("revision") or default_revision
        slots.append(_slot(model, rev, _url(url), f"chall-{i}"))
    return slots


@app.command()
def duel(
    model_url: Annotated[str, typer.Option(envvar="MODEL_URL")] = "http://localhost:8000/v1",
    model_id: Annotated[str | None, typer.Option(envvar=["MODEL_ID", "MODEL_NAME"])] = None,
    model_revision: Annotated[str, typer.Option(envvar="MODEL_REVISION")] = "local",
    challenger_url: Annotated[str | None, typer.Option(envvar="CHALLENGER_URL")] = None,
    challenger_model_id: Annotated[str | None, typer.Option(
        envvar=["CHALLENGER_MODEL_ID", "CHALLENGER_MODEL_NAME"],
    )] = None,
    challenger_revision: Annotated[str | None, typer.Option(envvar="CHALLENGER_MODEL_REVISION")] = None,
    max_tasks: int = 50,
    tasks_per_batch: int = 4,
    env: Annotated[list[str], typer.Option()] = [],
    progress_interval: Annotated[int, typer.Option(envvar="PROGRESS_INTERVAL")] = 0,
    log_level: str = "INFO",
):
    """Run a single duel between champion and challenger models."""
    _init_logging(log_level)
    url = _url(model_url)
    c_url = _url(challenger_url or url)
    mid = model_id or _detect_model_id(url)
    cmid = challenger_model_id or (mid if c_url == url else _detect_model_id(c_url))
    crev = challenger_revision or model_revision

    config = Config(
        max_tasks_per_env=max_tasks, tasks_per_batch=tasks_per_batch,
        k_init=1.0, k_final=1.0, environments=_resolve_envs(env),
    )
    asyncio.run(_run_duel(
        config, _slot(mid, model_revision, url, "champion"),
        _slot(cmid, crev, c_url, "challenger"), progress_interval, hotkey="",
    ))


@app.command()
def queue(
    queue_file: Annotated[str, typer.Argument(help="JSON/JSONL file with challenger entries")],
    model_url: Annotated[str, typer.Option(envvar="MODEL_URL")] = "http://localhost:8000/v1",
    model_id: Annotated[str | None, typer.Option(envvar=["MODEL_ID", "MODEL_NAME"])] = None,
    model_revision: Annotated[str, typer.Option(envvar="MODEL_REVISION")] = "local",
    challenger_model_id: Annotated[str | None, typer.Option(
        envvar=["CHALLENGER_MODEL_ID", "CHALLENGER_MODEL_NAME"],
    )] = None,
    challenger_revision: Annotated[str | None, typer.Option(envvar="CHALLENGER_MODEL_REVISION")] = None,
    challenger_url: Annotated[str | None, typer.Option(envvar="CHALLENGER_URL")] = None,
    max_tasks: int = 50,
    tasks_per_batch: int = 4,
    env: Annotated[list[str], typer.Option()] = [],
    progress_interval: Annotated[int, typer.Option(envvar="PROGRESS_INTERVAL")] = 0,
    log_level: str = "INFO",
):
    """Run a queue of challengers from a file against the champion."""
    _init_logging(log_level)
    url = _url(model_url)
    mid = model_id or _detect_model_id(url)
    cmid = challenger_model_id or mid
    crev = challenger_revision or model_revision
    c_url = _url(challenger_url or url)

    challengers = _load_queue_file(queue_file, cmid, crev, c_url)
    if not challengers:
        typer.echo("empty queue file", err=True)
        raise typer.Exit(1)

    config = Config(
        max_tasks_per_env=max_tasks, tasks_per_batch=tasks_per_batch,
        k_init=1.0, k_final=1.0, environments=_resolve_envs(env),
    )
    asyncio.run(_run_queue(config, _slot(mid, model_revision, url, "champion"), challengers, progress_interval, hotkey=""))


@app.command()
def chain(
    model_url: Annotated[str, typer.Option(envvar="MODEL_URL")] = "http://localhost:8000/v1",
    challenger_url: Annotated[str | None, typer.Option(envvar="CHALLENGER_URL")] = None,
    netuid: int = 120,
    wallet_cold: Annotated[str, typer.Option(envvar="BT_WALLET_COLD")] = "default",
    wallet_hot: Annotated[str, typer.Option(envvar="BT_WALLET_HOT")] = "default",
    max_tasks: int = 50,
    tasks_per_batch: int = 4,
    env: Annotated[list[str], typer.Option()] = [],
    log_level: str = "INFO",
):
    """Run the full chain loop with local vLLM slots."""
    _init_logging(log_level)
    url = _url(model_url)
    c_url = _url(challenger_url or url)

    config = Config(
        netuid=netuid, wallet_name=wallet_cold, hotkey_name=wallet_hot,
        max_tasks_per_env=max_tasks, tasks_per_batch=tasks_per_batch,
        k_init=1.0, k_final=1.0, environments=_resolve_envs(env),
    )
    asyncio.run(run(config, LocalSlots(url, c_url)))


if __name__ == "__main__":
    app()
