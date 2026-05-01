"""Slot lifecycle: restore artifact, start sidecar HTTP control plane, run vLLM.

The validator no longer ships bytes through itself. The slot pulls the artifact
(HF or restore-from-S3), then the sidecar accepts S3 credentials over a bearer-
authenticated POST and uploads the local artifact dir back to the configured
providers in parallel.

Two security pins, asserted before vLLM starts:
  --trust-remote-code=False       blocks miner-controlled Python execution
  --load-format=safetensors       blocks pickle-format weight loading

The sidecar runs in a thread alongside vLLM (a subprocess). Signals are
forwarded to vLLM; the entrypoint exits with vLLM's status.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import slot_backup
from .backup import restore_from_env

log = logging.getLogger(__name__)

REQUIRED_FLAGS = ("--trust-remote-code=False", "--load-format=safetensors")
DEFAULT_SIDECAR_PORT = 8001


_token: str | None = None
_model_dir: Path | None = None


def _assert_flags(argv: list[str]) -> None:
    for flag in REQUIRED_FLAGS:
        if flag not in argv:
            raise SystemExit(f"vllm_entrypoint: missing required flag {flag!r}; refusing to start")


def _scrub_token() -> str:
    token = os.environ.pop("AFFINE_SLOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("vllm_entrypoint: AFFINE_SLOT_TOKEN missing or empty")
    return token


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_a, **_kw): pass

    def _json(self, code: int, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _empty(self, code: int):
        self.send_response(code)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _auth_ok(self) -> bool:
        h = self.headers.get("Authorization", "")
        prefix = "Bearer "
        return _token is not None and h.startswith(prefix) and h[len(prefix):] == _token

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n > 0 else b""
        return json.loads(raw or b"{}")

    def do_GET(self):
        path = self.path.rstrip("/")
        if path == "/healthz":
            return self._json(200, {"ok": True})
        if path == "/backup":
            return self._json(200, slot_backup.snapshot())
        return self._empty(404)

    def do_POST(self):
        path = self.path.rstrip("/")
        if path == "/setup-backup":
            if not self._auth_ok():
                return self._empty(401)
            try:
                payload = self._read_json()
                providers = [
                    slot_backup.ProviderCreds(
                        name=name,
                        endpoint_url=p["endpoint_url"],
                        region=p.get("region", "auto"),
                        bucket=p["bucket"],
                        access_key=p["access_key"],
                        secret_key=p["secret_key"],
                    )
                    for name, p in payload["providers"].items()
                ]
                slot_backup.start(
                    _model_dir,
                    providers,
                    prefix=payload["prefix"],
                    model=payload["model"],
                    revision=payload["revision"],
                    artifact_id=payload["artifact_id"],
                )
            except Exception as e:
                log.warning(f"setup-backup failed: {e}")
                return self._json(400, {"error": str(e)})
            return self._empty(204)
        if path == "/teardown-backup":
            if not self._auth_ok():
                return self._empty(401)
            try:
                slot_backup.abort()
            except Exception as e:
                log.warning(f"teardown-backup failed: {e}")
                return self._json(500, {"error": str(e)})
            return self._empty(204)
        return self._empty(404)


def _start_sidecar(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True, name="sidecar").start()
    log.info(f"sidecar listening on :{port}")
    return server


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        stream=sys.stderr)
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hf", "s3"], default="hf")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--served-model-name", required=True)
    ap.add_argument("--manifest-key")
    ap.add_argument("--sidecar-port", type=int,
                    default=int(os.environ.get("AFFINE_SIDECAR_PORT", DEFAULT_SIDECAR_PORT)))
    ap.add_argument("vllm_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    global _token, _model_dir
    _token = _scrub_token()

    model_path = args.model
    if args.source == "s3":
        if not args.manifest_key:
            raise SystemExit("--manifest-key is required for --source=s3")
        dest = Path(os.getenv("AFFINE_MODEL_DIR", "/models")) / args.served_model_name.replace("/", "__")
        tmp = Path(f"{dest}.tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        restore_from_env(args.manifest_key, tmp)
        if dest.exists():
            shutil.rmtree(dest)
        tmp.rename(dest)
        model_path = str(dest)
    _model_dir = Path(model_path)

    extra = args.vllm_args[1:] if args.vllm_args[:1] == ["--"] else args.vllm_args
    argv = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", args.served_model_name,
    ]
    if args.source == "hf":
        argv.extend(["--revision", args.revision])
    argv.extend(extra)
    _assert_flags(argv)

    _start_sidecar(args.sidecar_port)

    proc = subprocess.Popen(argv)

    def forward(sig, _frame):
        log.info(f"forwarding {signal.Signals(sig).name} to vllm pid={proc.pid}")
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, forward)
        except (ValueError, OSError):
            pass

    rc = proc.wait()
    log.info(f"vllm exited rc={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
