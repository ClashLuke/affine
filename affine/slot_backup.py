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
class _Session:
    providers: list[ProviderCreds]
    paths: list[str]
    root: Path
    prefix: str
    model: str
    revision: str
    artifact_id: str
    ts: int
    lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    aborted: threading.Event = field(default_factory=threading.Event, init=False)
    state: str = field(default="running", init=False)
    refs: list[dict] = field(default_factory=list, init=False)
    error: str | None = field(default=None, init=False)
    threads: list[threading.Thread] = field(default_factory=list, init=False)

    def __post_init__(self):
        for cfg in self.providers:
            sub_prefix = f"{self.prefix.strip('/')}/{cfg.name}-{self.ts}"
            t = threading.Thread(
                target=self._upload, args=(cfg, sub_prefix),
                daemon=True, name=f"slot-backup-{cfg.name}",
            )
            self.threads.append(t)

    def start(self) -> None:
        for t in self.threads:
            t.start()

    def abort(self) -> None:
        # Signal-only: workers see _aborted and clean up their own per-thread
        # prefix in their except branch. Then ensure the state transitions out
        # of "running" so a subsequent start() doesn't see stale state and skip.
        self.aborted.set()
        for t in self.threads:
            t.join(timeout=30)
        with self.lock:
            if self.state == "running":
                self.state = "aborted"

    def snapshot(self) -> dict:
        with self.lock:
            out = {"state": self.state, "refs": list(self.refs),
                   "artifact_id": self.artifact_id}
            if self.error:
                out["error"] = self.error
            return out

    def _upload(self, cfg: ProviderCreds, prefix: str) -> None:
        s3 = _client(cfg)
        files_meta: list[dict] = []
        try:
            for rel in self.paths:
                if self.aborted.is_set():
                    raise RuntimeError("aborted")
                key = f"{prefix}/files/{rel}"
                size, sha = _upload_file(s3, cfg.bucket, key, self.root / rel, self.aborted)
                files_meta.append({"path": rel, "object_key": key, "size": size, "sha256": sha})
            manifest = {
                "schema": 1,
                "source": "slot",
                "provider": cfg.name,
                "model": self.model,
                "revision": self.revision,
                "artifact_id": self.artifact_id,
                "created_at": self.ts,
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
            with self.lock:
                # Late writes (a worker that finished after abort() has already
                # transitioned) must not clobber the post-abort state.
                if self.state in ("aborted", "failed"):
                    return
                self.refs.append(ref)
                self.state = "done"
            log.info(f"slot_backup: provider {cfg.name} completed ({len(self.paths)} files)")
        except Exception as exc:
            log.warning(f"slot_backup: provider {cfg.name} failed: {exc}")
            with contextlib.suppress(Exception):
                delete_prefix(s3, cfg.bucket, prefix)
            with self.lock:
                if self.state == "done":
                    return
                # Use thread identity (`ident`), not name — names collide across
                # consecutive sessions; ident is unique per thread for the
                # process lifetime.
                others_alive = any(
                    t.is_alive() and t.ident != threading.current_thread().ident
                    for t in self.threads
                )
                if not others_alive and self.state == "running":
                    self.state = "failed"
                    self.error = f"{cfg.name}: {exc}"


_lock = threading.Lock()
_session: _Session | None = None


def snapshot() -> dict:
    s = _session
    return s.snapshot() if s is not None else {"state": "idle", "refs": [], "artifact_id": ""}


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
    global _session
    root = Path(model_dir)
    paths = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    ts = start_id if start_id is not None else int(time.time())
    with _lock:
        if _session is not None and _session.state == "running":
            log.info("slot_backup: upload already running; ignoring start")
            return
        log.info(f"slot_backup: starting upload of {len(paths)} files to {len(providers)} providers")
        _session = _Session(
            providers=list(providers), paths=paths, root=root,
            prefix=prefix, model=model, revision=revision,
            artifact_id=artifact_id, ts=ts,
        )
    _session.start()


def abort() -> None:
    s = _session
    if s is not None:
        s.abort()


def _client(cfg: ProviderCreds):
    return s3_client(
        endpoint_url=cfg.endpoint_url, region=cfg.region,
        access_key=cfg.access_key, secret_key=cfg.secret_key,
        max_attempts=3,
    )


def _upload_file(s3, bucket: str, key: str, src: Path,
                 aborted: threading.Event) -> tuple[int, str]:
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
                if aborted.is_set():
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
        # complete_multipart_upload now lives INSIDE the try, so a failure here
        # also runs abort_multipart_upload (issue 034). Without this, a failed
        # complete leaves the upload_id allocated forever — S3 charges for it.
        parts.sort(key=lambda p: p["PartNumber"])
        s3.complete_multipart_upload(
            Bucket=bucket, Key=key, UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
        return size, digest.hexdigest()
    except Exception:
        with contextlib.suppress(Exception):
            s3.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        raise
