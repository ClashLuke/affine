from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import boto3
import smart_open
from botocore.config import Config as BotoConfig

from .store import BackupRecord


log = logging.getLogger(__name__)

HIPPIUS_ENDPOINT = "https://s3.hippius.com"
HIPPIUS_REGION = "decentralized"
_MAX_ATTEMPTS = 5         # per verify/restore operation
_FIRST_BACKOFF_S = 2.0
_MAX_BACKOFF_S = 60.0
CHUNK_SIZE = 1024 * 1024
UPLOAD_PART_CONCURRENCY = 16


@dataclass(frozen=True)
class S3Config:
    name: str
    endpoint_url: str
    region: str
    bucket: str
    prefix: str
    access_key: str
    secret_key: str

    @functools.cached_property
    def transport_params(self) -> dict:
        client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            region_name=self.region,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=BotoConfig(
                signature_version="s3v4",
                retries={"total_max_attempts": 1, "mode": "standard"},
                s3={"addressing_style": "path"},
                max_pool_connections=UPLOAD_PART_CONCURRENCY * 2,
            ),
        )
        return {"client": client}

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"

    @classmethod
    def from_envs(cls, *, hotkey: str, netuid: int) -> list["S3Config"]:
        hot_hash = hashlib.sha256(hotkey.encode()).hexdigest()[:16]
        namespace = os.getenv("AFFINE_NAMESPACE", "prod").strip().strip("/") or "prod"
        default_prefix = f"{netuid}/{namespace}/{hot_hash}"
        out: list[S3Config] = []

        hippius_access = os.getenv("HIPPIUS_S3_ACCESS_KEY", "").strip()
        hippius_secret = os.getenv("HIPPIUS_S3_SECRET_KEY", "").strip()
        if hippius_access and hippius_secret:
            out.append(cls(
                name="hippius",
                endpoint_url=os.getenv("HIPPIUS_S3_ENDPOINT", HIPPIUS_ENDPOINT),
                region=os.getenv("HIPPIUS_S3_REGION", HIPPIUS_REGION),
                bucket=os.getenv("HIPPIUS_S3_BUCKET", f"affine-{hot_hash}").strip(),
                prefix=os.getenv("HIPPIUS_S3_PREFIX", default_prefix).strip().strip("/"),
                access_key=hippius_access,
                secret_key=hippius_secret,
            ))

        r2_access = os.getenv("R2_S3_ACCESS_KEY_ID", "").strip()
        r2_secret = os.getenv("R2_S3_SECRET_ACCESS_KEY", "").strip()
        r2_bucket = os.getenv("R2_S3_BUCKET", "").strip()
        r2_endpoint = os.getenv("R2_S3_ENDPOINT_URL", "").strip()
        if r2_access and r2_secret and r2_bucket and r2_endpoint:
            out.append(cls(
                name="r2",
                endpoint_url=r2_endpoint,
                region=os.getenv("R2_S3_REGION", "auto").strip() or "auto",
                bucket=r2_bucket,
                prefix=os.getenv("R2_S3_PREFIX", default_prefix).strip().strip("/"),
                access_key=r2_access,
                secret_key=r2_secret,
            ))

        if not out:
            raise RuntimeError("no S3 backup providers configured")
        return out

    @classmethod
    def from_env_refs(cls) -> dict[str, "S3Config"]:
        out: dict[str, S3Config] = {}
        hippius_access = os.getenv("HIPPIUS_S3_ACCESS_KEY", "").strip()
        hippius_secret = os.getenv("HIPPIUS_S3_SECRET_KEY", "").strip()
        if hippius_access and hippius_secret and os.getenv("HIPPIUS_S3_BUCKET"):
            out["hippius"] = cls(
                name="hippius",
                endpoint_url=os.getenv("HIPPIUS_S3_ENDPOINT", HIPPIUS_ENDPOINT),
                region=os.getenv("HIPPIUS_S3_REGION", HIPPIUS_REGION),
                bucket=os.environ["HIPPIUS_S3_BUCKET"],
                prefix=os.getenv("HIPPIUS_S3_PREFIX", "").strip("/"),
                access_key=hippius_access,
                secret_key=hippius_secret,
            )
        r2_access = os.getenv("R2_S3_ACCESS_KEY_ID", "").strip()
        r2_secret = os.getenv("R2_S3_SECRET_ACCESS_KEY", "").strip()
        if r2_access and r2_secret and os.getenv("R2_S3_BUCKET") and os.getenv("R2_S3_ENDPOINT_URL"):
            out["r2"] = cls(
                name="r2",
                endpoint_url=os.environ["R2_S3_ENDPOINT_URL"],
                region=os.getenv("R2_S3_REGION", "auto").strip() or "auto",
                bucket=os.environ["R2_S3_BUCKET"],
                prefix=os.getenv("R2_S3_PREFIX", "").strip("/"),
                access_key=r2_access,
                secret_key=r2_secret,
            )
        return out


@dataclass(frozen=True)
class ManifestRef:
    provider: str
    bucket: str
    key: str
    prefix: str
    sha256: str


def encode_refs(refs: list[ManifestRef]) -> str:
    payload = [
        {"provider": r.provider, "bucket": r.bucket, "key": r.key,
         "prefix": r.prefix, "sha256": r.sha256}
        for r in refs
    ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def decode_refs(raw: str) -> list[ManifestRef]:
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("backup reference must be a list")
    if not data:
        raise ValueError("backup reference must not be empty")
    return [ManifestRef(
        provider=str(x["provider"]),
        bucket=str(x["bucket"]),
        key=str(x["key"]),
        prefix=str(x["prefix"]),
        sha256=str(x.get("sha256", "")),
    ) for x in data]


def refs_digest(refs: list[ManifestRef]) -> str:
    return hashlib.sha256(encode_refs(refs).encode()).hexdigest()


def _backoff(attempt: int) -> float:
    return min(_FIRST_BACKOFF_S * (2 ** attempt), _MAX_BACKOFF_S)


class BackupManager:
    def __init__(self, configs: list[S3Config]):
        self.configs = configs
        self.by_name = {c.name: c for c in configs}
        self.transport = {c.name: c.transport_params for c in configs}

    def verify(self, manifest_key: str, expected_manifest_sha: str | None = None) -> BackupRecord:
        refs = decode_refs(manifest_key)
        if expected_manifest_sha is not None and expected_manifest_sha != refs_digest(refs):
            raise ValueError("backup reference digest mismatch")
        for attempt in range(_MAX_ATTEMPTS):
            for ref in refs:
                try:
                    manifest, cfg = self._manifest_from_ref(ref)
                    self._verify_manifest_objects(manifest, cfg)
                    return BackupRecord(
                        artifact_id=manifest["artifact_id"],
                        model=manifest["model"],
                        revision=manifest["revision"],
                        prefix=encode_refs(refs),
                        manifest_key=encode_refs(refs),
                        manifest_sha256=refs_digest(refs),
                        status="verified",
                    )
                except Exception as exc:
                    log.warning("backup verify failed on provider %s (attempt %d/%d): %s", ref.provider, attempt + 1, _MAX_ATTEMPTS, exc)
            log.warning("all backup providers failed verification (attempt %d/%d)", attempt + 1, _MAX_ATTEMPTS)
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(_backoff(attempt))
        raise RuntimeError(f"backup verify failed after {_MAX_ATTEMPTS} attempts")

    def restore(self, manifest_key: str, dest: str | Path) -> BackupRecord:
        refs = decode_refs(manifest_key)
        for attempt in range(_MAX_ATTEMPTS):
            for ref in refs:
                try:
                    return self._restore_ref(ref, refs, dest)
                except Exception as exc:
                    log.warning("backup restore failed on provider %s (attempt %d/%d): %s", ref.provider, attempt + 1, _MAX_ATTEMPTS, exc)
            if attempt < _MAX_ATTEMPTS - 1:
                log.warning("all backup providers failed restore (attempt %d/%d)", attempt + 1, _MAX_ATTEMPTS)
                time.sleep(_backoff(attempt))
        raise RuntimeError(f"backup restore failed after {_MAX_ATTEMPTS} attempts")

    def delete_prefix(self, prefix: str) -> bool:
        refs = decode_refs(prefix)
        ok = True
        for ref in refs:
            cfg = self.by_name.get(ref.provider)
            if cfg is None:
                ok = False
                continue
            try:
                self._delete_prefix(cfg, ref.prefix)
            except Exception as exc:
                ok = False
                log.warning("backup cleanup failed on provider %s prefix=%s: %s", ref.provider, ref.prefix, exc)
        return ok

    def _manifest_from_ref(self, ref: ManifestRef) -> tuple[dict, S3Config]:
        cfg = self.by_name.get(ref.provider)
        if cfg is None:
            raise RuntimeError(f"provider not configured: {ref.provider}")
        if ref.bucket and ref.bucket != cfg.bucket:
            raise RuntimeError(f"provider {ref.provider} bucket mismatch: {ref.bucket} != {cfg.bucket}")
        with smart_open.open(cfg.uri(ref.key), "rb", transport_params=self.transport[cfg.name]) as f:
            raw = f.read()
        if ref.sha256 and hashlib.sha256(raw).hexdigest() != ref.sha256:
            raise ValueError(f"manifest sha mismatch for {ref.provider}")
        data = json.loads(raw)
        if data.get("schema") != 1 or not isinstance(data.get("files"), list):
            raise ValueError(f"unsupported backup manifest: {ref.key}")
        return data, cfg

    def _verify_manifest_objects(self, manifest: dict, cfg: S3Config) -> None:
        for f in manifest["files"]:
            digest = hashlib.sha256()
            size = 0
            with smart_open.open(cfg.uri(f["object_key"]), "rb",
                                 transport_params=self.transport[cfg.name]) as inp:
                while chunk := inp.read(CHUNK_SIZE):
                    digest.update(chunk)
                    size += len(chunk)
            if digest.hexdigest() != f["sha256"] or size != int(f["size"]):
                raise ValueError(f"backup object mismatch: {f['path']}")

    def _restore_ref(self, ref: ManifestRef, refs: list[ManifestRef], dest: str | Path) -> BackupRecord:
        manifest, cfg = self._manifest_from_ref(ref)
        base = Path(dest)
        base.mkdir(parents=True, exist_ok=True)
        for f in manifest["files"]:
            target = base / f["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with smart_open.open(cfg.uri(f["object_key"]), "rb",
                                 transport_params=self.transport[cfg.name]) as inp, target.open("wb") as out:
                while chunk := inp.read(CHUNK_SIZE):
                    digest.update(chunk)
                    size += len(chunk)
                    out.write(chunk)
            if digest.hexdigest() != f["sha256"] or size != int(f["size"]):
                target.unlink(missing_ok=True)
                raise ValueError(f"restored object mismatch: {f['path']}")
        return BackupRecord(
            artifact_id=manifest["artifact_id"],
            model=manifest["model"],
            revision=manifest["revision"],
            prefix=encode_refs(refs),
            manifest_key=encode_refs(refs),
            manifest_sha256=refs_digest(refs),
            status="restored",
        )

    def _delete_prefix(self, cfg: S3Config, prefix: str) -> None:
        client = cfg.transport_params["client"]
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


def restore_from_env(manifest_key: str, dest: str | Path) -> BackupRecord:
    configs = S3Config.from_env_refs()
    if not configs:
        raise RuntimeError("no S3 backup providers configured for restore")
    refs = decode_refs(manifest_key)
    needed = {r.provider for r in refs}
    available = [configs[name] for name in sorted(needed) if name in configs]
    if not available:
        raise RuntimeError(f"none of the backup providers are configured: {sorted(needed)}")
    return BackupManager(available).restore(manifest_key, dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["restore"])
    ap.add_argument("--manifest-key", required=True)
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()
    if args.command == "restore":
        restore_from_env(args.manifest_key, args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
