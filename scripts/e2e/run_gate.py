import argparse
import json
import os
import re
import select
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


RE_CONNECTED = re.compile(r"subtensor connected:")
RE_MINERS = re.compile(r"miners:\s*(\d+)\s+registered")
RE_HEALTH = re.compile(r"slot ready after")
RE_WEIGHTS = re.compile(r"weights set:\s*uid")
RE_WEIGHTS_UID = re.compile(r"weights set:\s*uid\s+(\d+)")
RE_DUEL_START = re.compile(r"duel:\s*champion\s+\S+@\S+\s+vs\s+uid(\d+)")
RE_VERDICT = re.compile(r"verdict:\s*champion\s+holds")
RE_DETHRONE = re.compile(r"DETHRONE:\s*\S+@\S+\s+->\s+uid\s+(\d+)")
RE_QUEUE_EXHAUSTED = re.compile(r"queue exhausted")
RE_SHUTDOWN = re.compile(r"^shutdown$|\bshutdown\s*$")


EXPECTED_ENVS = {"python", "nfa", "graph", "modular", "sudoku", "boolean", "tree"}


@dataclass(frozen=True)
class StageConfig:
    name: str
    spec: str
    timeout: int


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run affine pre-deploy E2E gate with hard assertions")
    p.add_argument("--stage", choices=["smoke", "full", "both", "dethrone"], default="both")
    p.add_argument("--smoke-spec", default="smoke")
    p.add_argument("--full-spec", default="full")
    p.add_argument("--dethrone-spec", default="smoke")
    p.add_argument("--smoke-timeout", type=int, default=2400)
    p.add_argument("--full-timeout", type=int, default=7200)
    p.add_argument("--dethrone-timeout", type=int, default=900)
    p.add_argument("--cmd", default="affine")
    p.add_argument("--log-dir", default=".e2e/logs")
    p.add_argument("--result-dir", default=".e2e/results")
    p.add_argument("--shutdown-grace", type=int, default=180)
    p.add_argument("--local", action="store_true", help="Keep AFFINE_LOCAL set (use LocalSlots)")
    return p.parse_args()


def _stages(args: argparse.Namespace) -> list[StageConfig]:
    if args.stage == "smoke":
        return [StageConfig("smoke", args.smoke_spec, args.smoke_timeout)]
    if args.stage == "full":
        return [StageConfig("full", args.full_spec, args.full_timeout)]
    if args.stage == "dethrone":
        return [StageConfig("dethrone", args.dethrone_spec, args.dethrone_timeout)]
    return [
        StageConfig("smoke", args.smoke_spec, args.smoke_timeout),
        StageConfig("full", args.full_spec, args.full_timeout),
    ]


def _validate_env() -> None:
    required = ["SUBTENSOR_ENDPOINT", "NETUID", "BT_WALLET_COLD", "BT_WALLET_HOT"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"missing required env vars: {', '.join(missing)}")


def _run_stage(stage: StageConfig, args: argparse.Namespace) -> dict:
    cmd = shlex.split(args.cmd)
    env = dict(os.environ)
    env["AFFINE_CONFIG_SPEC"] = stage.spec
    if not args.local:
        env.pop("AFFINE_LOCAL", None)

    log_dir = Path(args.log_dir)
    result_dir = Path(args.result_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage.name}.log"
    result_path = result_dir / f"{stage.name}.json"

    start = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    if proc.stdout is None:
        raise RuntimeError("failed to capture process stdout")

    connected = False
    miners_seen = False
    weights_set = False
    duel_start_seen = False
    verdict_seen = False
    shutdown_seen = False
    health_count = 0
    envs_loaded: set[str] = set()
    requested_shutdown = False
    deadline = start + stage.timeout

    # dethrone-stage tracking
    dethrone_seen = False
    duel_start_count = 0
    queue_exhausted_seen = False
    weights_uids: list[int] = []

    with log_path.open("w") as log_fp:
        while True:
            now = time.time()
            if now >= deadline:
                proc.send_signal(signal.SIGTERM)
                raise TimeoutError(f"{stage.name} exceeded timeout={stage.timeout}s")

            if proc.poll() is not None and proc.stdout.closed:
                break

            ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            if not ready:
                if proc.poll() is not None:
                    break
                continue

            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue

            log_fp.write(line)
            log_fp.flush()

            if RE_CONNECTED.search(line):
                connected = True
            m_miners = RE_MINERS.search(line)
            if m_miners and int(m_miners.group(1)) > 0:
                miners_seen = True
            if RE_HEALTH.search(line):
                health_count += 1
            if RE_WEIGHTS.search(line):
                weights_set = True
            m_wuid = RE_WEIGHTS_UID.search(line)
            if m_wuid:
                weights_uids.append(int(m_wuid.group(1)))
            if RE_DUEL_START.search(line):
                duel_start_seen = True
                duel_start_count += 1
            if RE_VERDICT.search(line):
                verdict_seen = True
            m_env = RE_ENV_LOADED.search(line)
            if m_env:
                envs_loaded.add(m_env.group(1))
            if RE_DETHRONE.search(line):
                dethrone_seen = True
            if RE_QUEUE_EXHAUSTED.search(line):
                queue_exhausted_seen = True
            if RE_SHUTDOWN.search(line):
                shutdown_seen = True

            if stage.name == "dethrone":
                should_stop = (
                    connected
                    and miners_seen
                    and health_count >= 2
                    and dethrone_seen
                    and duel_start_count >= 2
                    and verdict_seen
                    and len(weights_uids) >= 2
                    and queue_exhausted_seen
                    and envs_loaded == EXPECTED_ENVS
                )
            else:
                should_stop = (
                    connected
                    and miners_seen
                    and health_count >= 2
                    and weights_set
                    and duel_start_seen
                    and verdict_seen
                    and envs_loaded == EXPECTED_ENVS
                )
            if should_stop and not requested_shutdown:
                proc.send_signal(signal.SIGINT)
                requested_shutdown = True
                deadline = min(deadline, time.time() + args.shutdown_grace)

        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=30)

    base_ok = (
        connected
        and miners_seen
        and health_count >= 2
        and shutdown_seen
        and proc.returncode in (0, -2, 130)
    )
    if stage.name == "dethrone":
        stage_ok = (
            base_ok
            and dethrone_seen
            and duel_start_count >= 2
            and verdict_seen
            and len(weights_uids) >= 2
            and weights_uids[0] != weights_uids[-1]
            and queue_exhausted_seen
            and envs_loaded == EXPECTED_ENVS
        )
    else:
        stage_ok = (
            base_ok
            and weights_set
            and duel_start_seen
            and verdict_seen
            and envs_loaded == EXPECTED_ENVS
        )

    result = {
        "stage": stage.name,
        "config_spec": stage.spec,
        "connected": connected,
        "miners_seen": miners_seen,
        "health_count": health_count,
        "weights_set": weights_set,
        "weights_uids": weights_uids,
        "duel_start_seen": duel_start_seen,
        "duel_start_count": duel_start_count,
        "verdict_seen": verdict_seen,
        "dethrone_seen": dethrone_seen,
        "queue_exhausted_seen": queue_exhausted_seen,
        "envs_loaded": sorted(envs_loaded),
        "shutdown_seen": shutdown_seen,
        "exit_code": proc.returncode,
        "duration_seconds": round(time.time() - start, 2),
        "log_path": str(log_path),
        "ok": stage_ok,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise RuntimeError(f"{stage.name} gate failed; see {log_path} and {result_path}")
    print(f"[{stage.name}] ok in {result['duration_seconds']}s -> {result_path}")
    return result


def main() -> int:
    args = _parse_args()
    _validate_env()
    summaries = []
    for stage in _stages(args):
        summaries.append(_run_stage(stage, args))
    print(json.dumps({"ok": True, "stages": summaries}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
