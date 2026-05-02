"""Slot-side S3 uploader. First manifest to land flips state to done; the
other provider keeps uploading and appends its ref when it finishes."""

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

from .backup import delete_prefix, s3_client

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
class _UploadState:
    state: str = "idle"
    refs: list[dict] = field(default_factory=list)
    error: str | None = None
    artifact_id: str = ""


_lock = threading.Lock()
_state = _UploadState()
_provider_threads: list[threading.Thread] = []
_provider_prefixes: dict[str, tuple[ProviderCreds, str]] = {}
_aborted = threading.Event()


def snapshot() -> dict:
    with _lock:
        out = {"state": _state.state, "refs": list(_state.refs),
               "artifact_id": _state.artifact_id}
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
    start_id: int | None = None,
) -> None:
    root = Path(model_dir)
    paths = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    with _lock:
        if _state.state == "running":
            log.info("slot_backup: upload already running; ignoring start")
            return
        _aborted.clear()
        _state.state = "running"
        _state.refs = []
        _state.error = None
        _state.artifact_id = artifact_id
        ts = start_id if start_id is not None else int(time.time())
        _provider_prefixes.clear()
        _provider_threads.clear()
        log.info(f"slot_backup: starting upload of {len(paths)} files to {len(providers)} providers")
        for cfg in providers:
            sub_prefix = f"{prefix.strip('/')}/{cfg.name}-{ts}"
            _provider_prefixes[cfg.name] = (cfg, sub_prefix)
            t = threading.Thread(
                target=_upload_provider,
                args=(cfg, sub_prefix, paths, root, model, revision, artifact_id, ts),
                daemon=True,
                name=f"slot-backup-{cfg.name}",
            )
            _provider_threads.append(t)
            t.start()


def abort() -> None:
    _aborted.set()
    for t in list(_provider_threads):
        t.join(timeout=30)
    for cfg, sub_prefix in list(_provider_prefixes.values()):
        with contextlib.suppress(Exception):
            delete_prefix(_client(cfg), cfg.bucket, sub_prefix)
    with _lock:
        _state.state = "idle"
        _state.refs = []
        _state.error = None
        _state.artifact_id = ""
        _provider_prefixes.clear()
        _provider_threads.clear()


def _client(cfg: ProviderCreds):
    return s3_client(
        endpoint_url=cfg.endpoint_url, region=cfg.region,
        access_key=cfg.access_key, secret_key=cfg.secret_key,
        max_attempts=3,
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
            _state.state = "done"
        log.info(f"slot_backup: provider {cfg.name} completed ({len(paths)} files)")
    except Exception as exc:
        log.warning(f"slot_backup: provider {cfg.name} failed: {exc}")
        with contextlib.suppress(Exception):
            delete_prefix(s3, cfg.bucket, prefix)
        with _lock:
            if _state.state == "done":
                return
            others_alive = any(t.is_alive() and t.name != threading.current_thread().name
                               for t in _provider_threads)
            if not others_alive:
                _state.state = "failed"
                _state.error = f"{cfg.name}: {exc}"


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
    sem = threading.BoundedSemaphore(UPLOAD_PART_CONCURRENCY * 2)

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
                sem.acquire()
                chunk = f.read(UPLOAD_PART_SIZE)
                if not chunk:
                    sem.release()
                    break
                digest.update(chunk)
                fut = ex.submit(upload_part, n, chunk)
                fut.add_done_callback(lambda _f: sem.release())
                futures.append(fut)
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
