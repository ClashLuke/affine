"""Slot lifecycle: restore artifact (HF or S3), run a sidecar that accepts
upload creds over bearer auth, exec vLLM with safetensors + no-remote-code."""

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

from huggingface_hub import snapshot_download

from . import slot_backup
from .backup import S3Config, restore_refs

log = logging.getLogger(__name__)

REQUIRED_FLAGS = ("--no-trust-remote-code", "--load-format=safetensors")
DEFAULT_SIDECAR_PORT = 8001
# Read once at module load — re-reading per request would let an env-var rotation
# silently change the timeout in flight.
_RESTORE_TIMEOUT_S = int(os.getenv("AFFINE_RESTORE_TIMEOUT", "3600"))


_token: str | None = None
_model_dir: Path | None = None
_setup_lock = threading.Lock()
_setup_event = threading.Event()
_setup_payload: dict | None = None
_restore_done = threading.Event()
_restore_error: str | None = None


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
        return _token is not None and self.headers.get("Authorization") == f"Bearer {_token}"

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
        global _setup_payload
        path = self.path.rstrip("/")
        if path != "/setup":
            return self._empty(404)
        if not self._auth_ok():
            return self._empty(401)
        try:
            payload = self._read_json()
            if not isinstance(payload.get("providers"), dict):
                return self._json(400, {"error": "providers required"})
            for field in ("prefix", "model", "revision", "artifact_id"):
                if field not in payload:
                    return self._json(400, {"error": f"{field} required"})
            with _setup_lock:
                # Guard against contradictory setups, but otherwise do not dedupe:
                # slot_backup.start() and _restore_done.wait() are already idempotent,
                # and the previous dedupe shortcut returned 204 without ever calling
                # start() on the retry path — so a 504 retry never recovered.
                if _setup_payload is not None and \
                        payload["artifact_id"] != _setup_payload.get("artifact_id"):
                    return self._json(409, {"error": "different artifact in flight"})
                _setup_payload = payload
                _setup_event.set()
            if payload.get("manifest_key"):
                if not _restore_done.wait(timeout=_RESTORE_TIMEOUT_S):
                    return self._json(504, {"error": "restore did not complete"})
                if _restore_error:
                    return self._json(500, {"error": _restore_error})
            slot_backup.start(
                _model_dir,
                [slot_backup.ProviderCreds(
                    name=name, endpoint_url=p["endpoint_url"], region=p.get("region", "auto"),
                    bucket=p["bucket"], access_key=p["access_key"], secret_key=p["secret_key"],
                ) for name, p in payload["providers"].items()],
                prefix=payload["prefix"], model=payload["model"],
                revision=payload["revision"], artifact_id=payload["artifact_id"],
            )
        except Exception as e:
            log.warning(f"setup failed: {e}")
            return self._json(400, {"error": str(e)})
        return self._empty(204)


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

    global _token, _model_dir, _restore_error
    _token = os.environ.pop("AFFINE_SLOT_TOKEN", "").strip()
    if not _token:
        raise SystemExit("vllm_entrypoint: AFFINE_SLOT_TOKEN missing or empty")

    server = ThreadingHTTPServer(("0.0.0.0", args.sidecar_port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True, name="sidecar").start()
    log.info(f"sidecar listening on :{args.sidecar_port}")

    dest = Path(os.getenv("AFFINE_MODEL_DIR", "/models")) / args.served_model_name.replace("/", "__")
    if args.source == "s3":
        if not args.manifest_key:
            raise SystemExit("--manifest-key is required for --source=s3")
        log.info("waiting for /setup credentials")
        if not _setup_event.wait(timeout=_RESTORE_TIMEOUT_S):
            raise SystemExit("vllm_entrypoint: /setup not received within timeout")
        with _setup_lock:
            payload = _setup_payload
        try:
            configs = {
                name: S3Config(
                    name=name,
                    endpoint_url=p["endpoint_url"],
                    region=p.get("region", "auto"),
                    bucket=p["bucket"],
                    prefix=p.get("prefix", ""),
                    access_key=p["access_key"],
                    secret_key=p["secret_key"],
                )
                for name, p in payload["providers"].items()
            }
            tmp = Path(f"{dest}.tmp")
            if tmp.exists():
                shutil.rmtree(tmp)
            restore_refs(args.manifest_key, configs, tmp)
            if dest.exists():
                shutil.rmtree(dest)
            tmp.rename(dest)
        except Exception as e:
            _restore_error = str(e)
            _restore_done.set()
            raise
        _model_dir = dest
        _restore_done.set()
    else:
        ignore_patterns = [
            p.strip() for p in os.getenv(
                "AFFINE_HF_IGNORE_PATTERNS", "original/*,metal/*",
            ).split(",")
            if p.strip()
        ]
        log.info(f"snapshot_download: {args.model}@{args.revision} -> {dest}")
        snapshot_download(
            repo_id=args.model, revision=args.revision, local_dir=str(dest),
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
            ignore_patterns=ignore_patterns or None,
        )
        _model_dir = dest
        # No restore happened, but the wait flag must still be set so a /setup
        # POST that arrives after HF download (rare) doesn't block forever.
        _restore_done.set()

    extra = args.vllm_args[1:] if args.vllm_args[:1] == ["--"] else args.vllm_args
    argv = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(dest),
        "--served-model-name", args.served_model_name,
    ]
    argv.extend(extra)
    for flag in REQUIRED_FLAGS:
        if flag not in argv:
            raise SystemExit(f"vllm_entrypoint: missing required flag {flag!r}; refusing to start")

    proc = subprocess.Popen(argv)

    def forward(sig, _frame):
        # One syscall, no Python locks. Python signal handlers run on the main
        # thread, which may already hold the logging RLock or slot_backup._lock;
        # touching either here would deadlock. Cleanup runs after proc.wait()
        # returns, off the signal-handler stack.
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
    try:
        slot_backup.abort()
    except Exception as e:
        log.warning(f"slot_backup.abort failed: {e}")
    log.info(f"vllm exited rc={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
