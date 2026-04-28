"""Probe configured Gym environments without Targon slots.

Runs a local OpenAI-compatible stub by default, then samples every environment
in `ENV_REGISTRY` through the same sampler path used by the validator.

Usage:
    python scripts/probe_envs.py
    python scripts/probe_envs.py --base-url https://api.example/v1 --api-key $KEY --model model-id
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from affine.config import ENV_REGISTRY, EnvSpec
from affine.envs import EnvFactory
from affine.sampler import run_one
from affine.vllm import Slot

STUB = REPO / "scripts/e2e/stub_vllm.py"


def fingerprint(text: str | None) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="replace")).hexdigest()[:8]


async def probe_one(spec: EnvSpec, task_id: int, slot: Slot, api_key: str | None) -> dict:
    kwargs = {k: v for k, v in spec.params.items() if k != "timeout"}
    if api_key:
        kwargs["api_key"] = api_key
    factory = EnvFactory(spec.entrypoint)
    obs, _ = factory.make().reset(seed=task_id, options=kwargs)
    prompt = obs if isinstance(obs, str) else json.dumps(obs, sort_keys=True, default=str)
    passed, dt, tokens = await run_one(factory, kwargs, float(spec.params.get("timeout", 120)),
                                       slot, seed=task_id * 1_000_003, task_id=task_id)
    return {
        "env": spec.name, "task_id": task_id, "latency": round(dt, 2),
        "success": passed, "tokens": tokens,
        "prompt_fp": fingerprint(prompt),
    }


def start_stub(port: int) -> subprocess.Popen:
    p = subprocess.Popen(
        [sys.executable, str(STUB), "--port", str(port), "--model", "stub"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return p
        except OSError:
            time.sleep(0.1)
    p.terminate()
    raise RuntimeError(f"stub failed to bind on {port}")


async def main_async(args):
    if args.base_url:
        base_url = args.base_url
        api_key = args.api_key or os.environ.get("CHUTES_API_KEY") or os.environ.get("TARGON_API_KEY")
        model = args.model
        stub: subprocess.Popen | None = None
        print(f"# probing against {base_url} model={model}")
    else:
        port = args.stub_port
        stub = start_stub(port)
        base_url = f"http://127.0.0.1:{port}/v1"
        api_key = "stub"
        model = "stub"
        print(f"# probing against stub at {base_url}")

    task_ids = [int(x) for x in args.task_ids.split(",") if x.strip()]
    rows: list[dict] = []
    slot = Slot(model=model, revision="probe", base_url=base_url)

    try:
        for spec in ENV_REGISTRY.values():
            print(f"# {spec.name} ({spec.entrypoint})")
            for tid in task_ids:
                r = await probe_one(spec, tid, slot, api_key)
                rows.append(r)
                print(f"  task_id={tid:>10} success={str(r['success']):>5} "
                      f"tokens={r['tokens']:>5} prompt_fp={r['prompt_fp']} dt={r['latency']:.2f}s")
    finally:
        if stub is not None:
            stub.send_signal(signal.SIGTERM)
            try: stub.wait(timeout=2)
            except subprocess.TimeoutExpired: stub.kill()

    print()
    print("# summary")
    by_env: dict[str, list[dict]] = {}
    for r in rows:
        by_env.setdefault(r["env"], []).append(r)
    for env, rs in by_env.items():
        passes = sum(1 for r in rs if r["success"] is True)
        prompts = {r["prompt_fp"] for r in rs} - {"<none>"}
        print(f"  {env}: {passes}/{len(rs)} success, {len(prompts)} distinct prompts")

    if args.dump:
        Path(args.dump).write_text(json.dumps(rows, indent=2, default=str))
        print(f"# raw responses -> {args.dump}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-ids", default="0,1,2,100",
                    help="comma-separated task_ids to probe per env")
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible model URL; if unset, runs a local stub")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V3")
    ap.add_argument("--stub-port", type=int, default=18901)
    ap.add_argument("--dump", default=None,
                    help="write raw response JSON to this path")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
