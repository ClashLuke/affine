from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import boto3
import smart_open
from botocore.config import Config as BotoConfig

from .store import BackupRecord


log = logging.getLogger(__name__)

HIPPIUS_ENDPOINT = "https://s3.hippius.com"
HIPPIUS_REGION = "decentralized"
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

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"

    def payload(self) -> dict:
        return {
            "endpoint_url": self.endpoint_url,
            "region": self.region,
            "bucket": self.bucket,
            "access_key": self.access_key,
            "secret_key": self.secret_key,
        }

    @classmethod
    def from_env(cls, *, default_prefix: str | None = None) -> dict[str, "S3Config"]:
        out: dict[str, S3Config] = {}

        hippius_access = os.getenv("HIPPIUS_S3_ACCESS_KEY", "").strip()
        hippius_secret = os.getenv("HIPPIUS_S3_SECRET_KEY", "").strip()
        hippius_bucket = os.getenv("HIPPIUS_S3_BUCKET", "").strip()
        if hippius_access and hippius_secret and hippius_bucket:
            out["hippius"] = cls(
                name="hippius",
                endpoint_url=os.getenv("HIPPIUS_S3_ENDPOINT", HIPPIUS_ENDPOINT),
                region=os.getenv("HIPPIUS_S3_REGION", HIPPIUS_REGION),
                bucket=hippius_bucket,
                prefix=os.getenv("HIPPIUS_S3_PREFIX", default_prefix or "").strip().strip("/"),
                access_key=hippius_access,
                secret_key=hippius_secret,
            )

        r2_access = os.getenv("R2_S3_ACCESS_KEY_ID", "").strip()
        r2_secret = os.getenv("R2_S3_SECRET_ACCESS_KEY", "").strip()
        r2_bucket = os.getenv("R2_S3_BUCKET", "").strip()
        r2_endpoint = os.getenv("R2_S3_ENDPOINT_URL", "").strip()
        if r2_access and r2_secret and r2_bucket and r2_endpoint:
            out["r2"] = cls(
                name="r2",
                endpoint_url=r2_endpoint,
                region=os.getenv("R2_S3_REGION", "auto").strip() or "auto",
                bucket=r2_bucket,
                prefix=os.getenv("R2_S3_PREFIX", default_prefix or "").strip().strip("/"),
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
        for r in sorted(refs, key=lambda r: r.provider)
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


def s3_client(*, endpoint_url: str, region: str, access_key: str, secret_key: str,
              max_attempts: int = 1):
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url, region_name=region,
        aws_access_key_id=access_key, aws_secret_access_key=secret_key,
        config=BotoConfig(
            signature_version="s3v4",
            retries={"total_max_attempts": max_attempts, "mode": "standard"},
            s3={"addressing_style": "path"},
            max_pool_connections=UPLOAD_PART_CONCURRENCY * 2,
        ),
    )


def delete_prefix(client, bucket: str, prefix: str) -> None:
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs)
        objects = [{"Key": x["Key"]} for x in resp.get("Contents", [])]
        if objects:
            d = client.delete_objects(Bucket=bucket, Delete={"Objects": objects})
            errs = d.get("Errors") or []
            if errs:
                e = errs[0]
                raise RuntimeError(
                    f"delete_objects partial failure ({len(errs)}/{len(objects)}): "
                    f"{e.get('Key')} {e.get('Code')}"
                )
        if not resp.get("IsTruncated"):
            return
        token = resp.get("NextContinuationToken")


def restore_refs(manifest_key: str, configs: dict[str, S3Config], dest: str | Path) -> BackupRecord:
    refs = decode_refs(manifest_key)
    available = [r for r in refs if r.provider in configs]
    if not available:
        raise RuntimeError(f"none of the backup providers are configured: {sorted({r.provider for r in refs})}")
    base = Path(dest)
    base.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for ref in available:
        cfg = configs[ref.provider]
        if ref.bucket and ref.bucket != cfg.bucket:
            failures.append(f"{ref.provider}: bucket mismatch {ref.bucket} != {cfg.bucket}")
            continue
        tp = {"client": s3_client(
            endpoint_url=cfg.endpoint_url, region=cfg.region,
            access_key=cfg.access_key, secret_key=cfg.secret_key,
            max_attempts=3,
        )}
        try:
            with smart_open.open(cfg.uri(ref.key), "rb", transport_params=tp) as f:
                raw = f.read()
            if ref.sha256 and hashlib.sha256(raw).hexdigest() != ref.sha256:
                raise ValueError(f"manifest sha mismatch for {ref.provider}")
            manifest = json.loads(raw)
            if manifest.get("schema") != 1 or not isinstance(manifest.get("files"), list):
                raise ValueError(f"unsupported backup manifest: {ref.key}")
            for f in manifest["files"]:
                target = base / f["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size = 0
                with smart_open.open(cfg.uri(f["object_key"]), "rb",
                                     transport_params=tp) as inp, target.open("wb") as out:
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
                manifest_key=manifest_key,
                status="restored",
            )
        except Exception as exc:
            failures.append(f"{ref.provider}: {type(exc).__name__}: {exc}")
            log.warning("restore failover from %s: %s", ref.provider, exc)
    raise RuntimeError("restore failed on all providers — " + " | ".join(failures))


def delete_refs(manifest_key: str, configs: dict[str, S3Config]) -> bool:
    refs = decode_refs(manifest_key)
    ok = True
    for ref in refs:
        cfg = configs.get(ref.provider)
        if cfg is None:
            ok = False
            continue
        try:
            client = s3_client(
                endpoint_url=cfg.endpoint_url, region=cfg.region,
                access_key=cfg.access_key, secret_key=cfg.secret_key,
                max_attempts=3,
            )
            delete_prefix(client, cfg.bucket, ref.prefix)
        except Exception as exc:
            ok = False
            log.warning("backup cleanup failed on provider %s prefix=%s: %s",
                        ref.provider, ref.prefix, exc)
    return ok
