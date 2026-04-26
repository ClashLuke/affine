"""Probe live env containers without spending Targon credits.

Spins up a local OpenAI-compatible stub on 127.0.0.1:18901 (returns "I don't
know" for any prompt, no logprobs), then POSTs `evaluate` to every running
affine-env / distill / game container with several task_ids. Prints a per-env
table of (task_id, success, score, error_type, latency, prompt-fingerprint)
and per-env summaries.

What this catches that pytest mocks can't:
  - ded/abd defaulting task_id=0 (every miner gets dataset row 0).
  - distill 1-indexed bucket (task_00000000000.json is 404).
  - sampler classifying envelope-level failures (status=failed, missing keys)
    as real losses instead of infra.
  - prompt actually varying with task_id (same uid+revision should still see
    different prompts across iterations).

Usage:
    python scripts/probe_envs.py
    python scripts/probe_envs.py --base-url https://api.targon.com/v1 \
        --api-key $TARGON_API_KEY --model deepseek-ai/DeepSeek-V3
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
from dataclasses import dataclass
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
STUB = REPO / "scripts/e2e/stub_vllm.py"


@dataclass
class EnvTarget:
    name: str
    container: str
    method: str
    base_kwargs: dict


TARGETS = [
    EnvTarget("affine:ded",  "affine-env-v4",  "evaluate", {"task_type": "ded", "temperature": 0.0}),
    EnvTarget("affine:abd",  "affine-env-v4",  "evaluate", {"task_type": "abd", "temperature": 0.0}),
    EnvTarget("distill",     "distill-latest", "evaluate", {"temperature": 0.0}),
    EnvTarget("game",        "game-openspiel", "evaluate", {"temperature": 0.0, "opponent": "mcts"}),
]


def container_ip(name: str) -> str | None:
    out = subprocess.run(
        ["docker", "inspect", "-f",
         "{{range $k,$v := .NetworkSettings.Networks}}{{$v.IPAddress}}{{end}}", name],
        capture_output=True, text=True,
    )
    ip = (out.stdout or "").strip()
    return ip or None


def host_lan_ip() -> str:
    """Address reachable from inside containers on the docker bridge."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("172.17.0.1", 1))
        return s.getsockname()[0]
    finally:
        s.close()


def fingerprint(text: str | None) -> str:
    if not text:
        return "<none>"
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:8]


async def call(client: httpx.AsyncClient, base: str, method: str, kwargs: dict) -> tuple[dict, dict | None, float]:
    """Returns (inner_result, outer_envelope_or_none, latency).

    Function-based envs respond `{"status": "...", "result": <inner>}`. HTTPExecutor
    unwraps to the inner before the sampler sees it, so we match that here. HTTP-based
    envs return the inner dict directly, so envelope is None."""
    t0 = time.monotonic()
    try:
        r = await client.post(f"{base}/call", json={"method": method, "kwargs": kwargs}, timeout=120)
        dt = time.monotonic() - t0
        try:
            data = r.json()
        except Exception:
            return {"_status": r.status_code, "_text": r.text[:400]}, None, dt
        if isinstance(data, dict) and "status" in data and "result" in data:
            return data.get("result") or {}, data, dt
        return data, None, dt
    except Exception as e:
        return {"_exception": f"{type(e).__name__}: {e}"}, None, time.monotonic() - t0


async def probe_one(client: httpx.AsyncClient, t: EnvTarget, ip: str, task_id: int,
                    model: str, base_url: str, api_key: str | None) -> dict:
    kwargs = dict(t.base_kwargs)
    kwargs.update(model=model, base_url=base_url, task_id=task_id, seed=task_id * 1000003)
    if api_key:
        kwargs["api_key"] = api_key
    body, envelope, dt = await call(client, f"http://{ip}:8000", t.method, kwargs)
    extra = body.get("extra") if isinstance(body, dict) else None
    convo = (extra or {}).get("conversation")
    prompt = "\n".join(m.get("content") or "" for m in convo) if isinstance(convo, list) else None
    return {
        "env": t.name, "task_id": task_id, "latency": round(dt, 2),
        "success": body.get("success") if isinstance(body, dict) else None,
        "score": body.get("score") if isinstance(body, dict) else None,
        "error_type": body.get("error_type") if isinstance(body, dict) else None,
        "envelope_status": envelope.get("status") if envelope else None,
        "prompt_fp": fingerprint(prompt),
        "raw": body, "envelope": envelope,
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
        host = host_lan_ip()
        base_url = f"http://{host}:{port}/v1"
        api_key = "stub"
        model = "stub"
        print(f"# probing against stub at {base_url}")

    task_ids = [int(x) for x in args.task_ids.split(",") if x.strip()]
    rows: list[dict] = []

    try:
        async with httpx.AsyncClient() as client:
            for t in TARGETS:
                ip = container_ip(t.container)
                if not ip:
                    print(f"# {t.name}: container {t.container} not running, skipped")
                    continue
                print(f"# {t.name} @ {ip}")
                for tid in task_ids:
                    r = await probe_one(client, t, ip, tid, model, base_url, api_key)
                    rows.append(r)
                    score_repr = f"{r['score']:.3f}" if isinstance(r["score"], (int, float)) else str(r["score"])
                    print(f"  task_id={tid:>5} success={str(r['success']):>5} "
                          f"score={score_repr:>6} error={r['error_type'] or '-':<24} "
                          f"envelope={r['envelope_status'] or '-':<8} prompt_fp={r['prompt_fp']:<8} "
                          f"dt={r['latency']:.2f}s")
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
        infras = sum(1 for r in rs if r["error_type"])
        prompts = {r["prompt_fp"] for r in rs} - {"<none>"}
        print(f"  {env}: {passes}/{len(rs)} success, {infras} error_type set, "
              f"{len(prompts)} distinct prompts")

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
