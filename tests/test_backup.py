from __future__ import annotations

from affine.backup import (
    BackupManager,
    BackupRecord,
    ManifestRef,
    S3Config,
    decode_refs,
    encode_refs,
    restore_from_env,
)


def test_s3_configs_include_hippius_and_r2(monkeypatch):
    monkeypatch.setenv("HIPPIUS_S3_ACCESS_KEY", "hip")
    monkeypatch.setenv("HIPPIUS_S3_SECRET_KEY", "sec")
    monkeypatch.setenv("R2_S3_ACCESS_KEY_ID", "r2")
    monkeypatch.setenv("R2_S3_SECRET_ACCESS_KEY", "r2sec")
    monkeypatch.setenv("R2_S3_BUCKET", "affine")
    monkeypatch.setenv("R2_S3_ENDPOINT_URL", "https://example.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_S3_REGION", "auto")

    providers = S3Config.from_envs(hotkey="hk", netuid=120)

    assert [p.name for p in providers] == ["hippius", "r2"]
    assert providers[1].bucket == "affine"
    assert providers[1].region == "auto"


def test_s3_default_prefix_is_namespaced(monkeypatch):
    monkeypatch.setenv("AFFINE_NAMESPACE", "shadow")
    monkeypatch.setenv("R2_S3_ACCESS_KEY_ID", "r2")
    monkeypatch.setenv("R2_S3_SECRET_ACCESS_KEY", "r2sec")
    monkeypatch.setenv("R2_S3_BUCKET", "affine")
    monkeypatch.setenv("R2_S3_ENDPOINT_URL", "https://example.r2.cloudflarestorage.com")

    providers = S3Config.from_envs(hotkey="hk", netuid=120)

    assert providers[0].prefix.startswith("120/shadow/")


def test_manifest_refs_round_trip():
    refs = [
        ManifestRef("hippius", "b1", "p1/manifest.json", "p1", "sha1"),
        ManifestRef("r2", "b2", "p2/manifest.json", "p2", "sha2"),
    ]

    assert decode_refs(encode_refs(refs)) == refs


def test_restore_from_env_loads_local_dotenv(monkeypatch, tmp_path):
    for name in (
        "HIPPIUS_S3_ACCESS_KEY",
        "HIPPIUS_S3_SECRET_KEY",
        "R2_S3_ACCESS_KEY_ID",
        "R2_S3_SECRET_ACCESS_KEY",
        "R2_S3_BUCKET",
        "R2_S3_ENDPOINT_URL",
        "R2_S3_REGION",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "\n".join((
            "R2_S3_ACCESS_KEY_ID=r2",
            "R2_S3_SECRET_ACCESS_KEY=r2sec",
            "R2_S3_BUCKET=affine",
            "R2_S3_ENDPOINT_URL=https://example.r2.cloudflarestorage.com",
            "R2_S3_REGION=auto",
            "",
        ))
    )
    seen = {}

    class FakeBackupManager:
        def __init__(self, configs):
            seen["configs"] = configs

        def restore(self, manifest_key, dest):
            seen["manifest_key"] = manifest_key
            seen["dest"] = dest
            return BackupRecord("art", "m", "r", "p", manifest_key, "sha", "restored")

    monkeypatch.setattr("affine.backup.BackupManager", FakeBackupManager)
    refs = encode_refs([ManifestRef("r2", "affine", "p/manifest.json", "p", "sha")])

    record = restore_from_env(refs, tmp_path / "out")

    assert record.status == "restored"
    assert seen["configs"][0].name == "r2"
    assert seen["configs"][0].bucket == "affine"
    assert seen["configs"][0].access_key == "r2"


def test_verify_cycles_providers_and_restarts_after_all_fail(monkeypatch):
    refs = [
        ManifestRef("hippius", "b1", "p1/manifest.json", "p1", "sha1"),
        ManifestRef("r2", "b2", "p2/manifest.json", "p2", "sha2"),
    ]
    manager = BackupManager.__new__(BackupManager)
    manager.by_name = {}
    calls = []

    def manifest(ref):
        calls.append(ref.provider)
        if calls == ["hippius", "r2", "hippius"]:
            return {
                "schema": 1,
                "files": [],
                "artifact_id": "a",
                "model": "m",
                "revision": "r",
            }, object()
        raise RuntimeError("provider down")

    manager._manifest_from_ref = manifest
    manager._verify_manifest_objects = lambda manifest, cfg: None

    record = manager.verify(encode_refs(refs))

    assert calls == ["hippius", "r2", "hippius"]
    assert record.model == "m"
