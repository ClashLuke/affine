#!/usr/bin/env python3
"""Local development CLI: run the active-sampling loop against fixed vLLM URLs.

Miners are declared in a JSON file ([{uid, model, revision, url}, ...]) or via
--model-url + --model-id for a single-miner smoke test.
"""

from __future__ import annotations
import asyncio
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from affine.chain import Miner
from affine.config import Config, ENV_REGISTRY
from affine.loop import run, static_chain
from affine.vllm import FixedSlots

app = typer.Typer(help="Affine local development CLI.")
log = logging.getLogger(__name__)


def _url(raw: str) -> str:
    raw = raw.rstrip("/")
    return raw if raw.endswith("/v1") else f"{raw}/v1"


def _load_miners(path: str | None, model: str, url: str) -> list[tuple[Miner, str]]:
    if not path:
        return [(Miner(uid=0, hotkey="local", model=model, revision="local", block=0), _url(url))]
    data = json.loads(Path(path).read_text())
    items = [data] if isinstance(data, dict) else data
    return [
        (Miner(uid=int(d.get("uid", i)), hotkey=d.get("hotkey", f"hk{i}"),
               model=d["model"], revision=d.get("revision", "local"),
               block=int(d.get("block", i))),
         _url(d["url"]))
        for i, d in enumerate(items)
    ]


@app.command()
def sample(
    model_url: Annotated[str, typer.Option(envvar="MODEL_URL")] = "http://localhost:8000/v1",
    model_id: Annotated[str, typer.Option(envvar="MODEL_ID")] = "local",
    miners_file: Annotated[str | None, typer.Option(help="JSON [{uid,model,revision,url}]")] = None,
    env: Annotated[list[str], typer.Option()] = [],
    env_timeout: int = 120,
    dwell: int = 10,
    evidence_path: str = "./.affine/local-evidence.jsonl",
    log_level: str = "INFO",
):
    """Active-sample a pool of local vLLM slots. Stops on SIGINT."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    env_names = env or ["python"]
    for n in env_names:
        if n not in ENV_REGISTRY:
            raise typer.BadParameter(f"unknown environment: {n}")
    environments = tuple(
        replace(ENV_REGISTRY[n], params={**ENV_REGISTRY[n].params, "timeout": env_timeout})
        for n in env_names
    )

    pairs = _load_miners(miners_file, model_id, model_url)
    miners = [m for m, _ in pairs]
    url_map = {(m.model, m.revision): url for m, url in pairs}

    cfg = Config(environments=environments, evidence_path=evidence_path, dwell_batch=dwell)
    asyncio.run(run(cfg, static_chain(miners), FixedSlots(url_map)))


if __name__ == "__main__":
    app()
