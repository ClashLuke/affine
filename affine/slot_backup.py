"""Slot-side backup uploader.

Runs inside the slot container alongside vLLM. Reads the artifact from local
disk and pushes it to each configured S3 provider in parallel. First-success
semantics: state flips to `done` the moment one provider's manifest lands;
the other keeps uploading and its ref is appended when it finishes.

Credentials are passed through `start()` and held in module RAM only — never
written to disk, never copied into os.environ. `abort()` clears the dict and
deletes any partially uploaded prefixes.
"""

from __future__ import annotations

import concurrent.futures as cf
import contextlib
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig

CHUNK_SIZE = 1024 * 1024
UPLOAD_PART_SIZE = 8 << 20
UPLOAD_PART_CONCURRENCY = 16

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderCreds:
    name: str
    endpoint_url: str
    region: str
    bucket: str
    access_key: str
    secret_key: str


@dataclass
class _ProviderResult:
    name: str
    bucket: str
    key: str
    prefix: str
    sha256: str


@dataclass
class _UploadState:
    state: str = "idle"            # idle | running | done | failed
    refs: list[dict] = field(default_factory=list)
    error: str | None = None
    model: str = ""
    revision: str = ""
    artifact_id: str = ""
    started_at: int = 0
    completed_at: int = 0


_lock = threading.Lock()
_state = _UploadState()
_executor: cf.ThreadPoolExecutor | None = None
_provider_threads: list[threading.Thread] = []
_provider_prefixes: dict[str, tuple[ProviderCreds, str]] = {}
_aborted = threading.Event()


def snapshot() -> dict:
    with _lock:
        out = {"state": _state.state, "refs": list(_state.refs)}
        if _state.error:
            out["error"] = _state.error
        return out


def start(
    model_dir: str | Path,
    providers: list[ProviderCreds],
    *,
    prefix: str,
    model: str,
    revision: str,
    artifact_id: str,
) -> None:
    """Start a fresh upload. Idempotent re-trigger: if state is `done` or
    `failed`, restart with a new attempt; if `running`, no-op."""
    global _executor
    with _lock:
        if _state.state == "running":
            log.info("slot_backup: upload already running; ignoring start")
            return
        _aborted.clear()
        _state.state = "running"
        _state.refs = []
        _state.error = None
        _state.model = model
        _state.revision = revision
        _state.artifact_id = artifact_id
        _state.started_at = int(time.time())
        _state.completed_at = 0
        _provider_prefixes.clear()
        _provider_threads.clear()

    paths = _collect_files(Path(model_dir))
    log.info(f"slot_backup: starting upload of {len(paths)} files to {len(providers)} providers")

    for cfg in providers:
        sub_prefix = f"{prefix.strip('/')}/{cfg.name}-{_state.started_at}"
        _provider_prefixes[cfg.name] = (cfg, sub_prefix)
        t = threading.Thread(
            target=_upload_provider,
            args=(cfg, sub_prefix, paths, Path(model_dir), model, revision, artifact_id, _state.started_at),
            daemon=True,
            name=f"slot-backup-{cfg.name}",
        )
        _provider_threads.append(t)
        t.start()


def abort() -> None:
    """Cancel in-flight uploads, delete partial prefixes, scrub creds."""
    _aborted.set()
    threads = list(_provider_threads)
    for t in threads:
        t.join(timeout=30)
    for name, (cfg, sub_prefix) in list(_provider_prefixes.items()):
        with contextlib.suppress(Exception):
            _delete_prefix(cfg, sub_prefix)
    with _lock:
        _state.state = "idle"
        _state.refs = []
        _state.error = None
        _state.model = ""
        _state.revision = ""
        _state.artifact_id = ""
        _state.started_at = 0
        _state.completed_at = 0
        _provider_prefixes.clear()
        _provider_threads.clear()


def _collect_files(root: Path) -> list[str]:
    out: list[str] = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out.append(str(p.relative_to(root)))
    return out


def _client(cfg: ProviderCreds):
    return boto3.client(
        "s3",
        endpoint_url=cfg.endpoint_url,
        region_name=cfg.region,
        aws_access_key_id=cfg.access_key,
        aws_secret_access_key=cfg.secret_key,
        config=BotoConfig(
            signature_version="s3v4",
            retries={"total_max_attempts": 3, "mode": "standard"},
            s3={"addressing_style": "path"},
            max_pool_connections=UPLOAD_PART_CONCURRENCY * 2,
        ),
    )


def _upload_provider(
    cfg: ProviderCreds,
    prefix: str,
    paths: list[str],
    root: Path,
    model: str,
    revision: str,
    artifact_id: str,
    ts: int,
) -> None:
    s3 = _client(cfg)
    files_meta: list[dict] = []
    try:
        for rel in paths:
            if _aborted.is_set():
                raise RuntimeError("aborted")
            key = f"{prefix}/files/{rel}"
            size, sha = _upload_file(s3, cfg.bucket, key, root / rel)
            files_meta.append({"path": rel, "object_key": key, "size": size, "sha256": sha})
        manifest = {
            "schema": 1,
            "source": "slot",
            "provider": cfg.name,
            "model": model,
            "revision": revision,
            "artifact_id": artifact_id,
            "created_at": ts,
            "files": files_meta,
        }
        body = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        manifest_key = f"{prefix}/manifest.json"
        s3.put_object(Bucket=cfg.bucket, Key=manifest_key, Body=body)
        ref = {
            "provider": cfg.name,
            "bucket": cfg.bucket,
            "key": manifest_key,
            "prefix": prefix,
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        with _lock:
            _state.refs.append(ref)
            if _state.state != "done":
                _state.state = "done"
                _state.completed_at = int(time.time())
        log.info(f"slot_backup: provider {cfg.name} completed ({len(paths)} files)")
    except Exception as exc:
        log.warning(f"slot_backup: provider {cfg.name} failed: {exc}")
        with contextlib.suppress(Exception):
            _delete_prefix(cfg, prefix)
        with _lock:
            if _state.state != "done" and not any(r["provider"] == cfg.name for r in _state.refs):
                done_count = sum(1 for n in _provider_prefixes if any(r["provider"] == n for r in _state.refs))
                running = sum(1 for t in _provider_threads if t.is_alive() and t.name != threading.current_thread().name)
                if done_count == 0 and running == 0:
                    _state.state = "failed"
                    _state.error = f"{cfg.name}: {exc}"
                    _state.completed_at = int(time.time())


def _upload_file(s3, bucket: str, key: str, src: Path) -> tuple[int, str]:
    size = src.stat().st_size
    digest = hashlib.sha256()
    if size <= UPLOAD_PART_SIZE:
        body = src.read_bytes()
        digest.update(body)
        s3.put_object(Bucket=bucket, Key=key, Body=body)
        return size, digest.hexdigest()

    upload_id = s3.create_multipart_upload(Bucket=bucket, Key=key)["UploadId"]
    parts: list[dict] = []
    parts_lock = threading.Lock()
    errors: list[Exception] = []
    err_lock = threading.Lock()

    def upload_part(num: int, body: bytes) -> None:
        try:
            r = s3.upload_part(Bucket=bucket, Key=key, PartNumber=num, UploadId=upload_id, Body=body)
            with parts_lock:
                parts.append({"PartNumber": num, "ETag": r["ETag"]})
        except Exception as e:
            with err_lock:
                errors.append(e)

    try:
        with cf.ThreadPoolExecutor(max_workers=UPLOAD_PART_CONCURRENCY) as ex, src.open("rb") as f:
            futures: list[cf.Future] = []
            n = 1
            while True:
                if _aborted.is_set():
                    raise RuntimeError("aborted")
                with err_lock:
                    if errors:
                        raise errors[0]
                chunk = f.read(UPLOAD_PART_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
                futures.append(ex.submit(upload_part, n, chunk))
                n += 1
            for fut in futures:
                fut.result()
        if errors:
            raise errors[0]
    except Exception:
        with contextlib.suppress(Exception):
            s3.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        raise

    parts.sort(key=lambda p: p["PartNumber"])
    s3.complete_multipart_upload(
        Bucket=bucket, Key=key, UploadId=upload_id,
        MultipartUpload={"Parts": parts},
    )
    return size, digest.hexdigest()


def _delete_prefix(cfg: ProviderCreds, prefix: str) -> None:
    client = _client(cfg)
    token = None
    while True:
        kwargs = {"Bucket": cfg.bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs)
        objects = [{"Key": x["Key"]} for x in resp.get("Contents", [])]
        if objects:
            client.delete_objects(Bucket=cfg.bucket, Delete={"Objects": objects})
        if not resp.get("IsTruncated"):
            return
        token = resp.get("NextContinuationToken")
