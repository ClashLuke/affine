import json

import pytest

from affine.config import Config


def _env_timeout(cfg: Config, name: str) -> int:
    for spec in cfg.environments:
        if spec.name == name:
            return int(spec.params["timeout"])
    raise AssertionError(f"missing env {name}")


def test_config_smoke_profile(monkeypatch):
    monkeypatch.setenv("AFFINE_CONFIG_SPEC", "smoke")
    cfg = Config.from_env()
    assert cfg.max_tasks_per_env == 8
    assert cfg.tasks_per_batch == 2
    assert cfg.k_init == 1.0
    assert cfg.k_final == 1.0
    assert _env_timeout(cfg, "affine:ded") <= 90
    assert _env_timeout(cfg, "affine:abd") <= 90
    assert _env_timeout(cfg, "distill") <= 90
    assert _env_timeout(cfg, "game") <= 420


def test_config_full_profile(monkeypatch):
    monkeypatch.setenv("AFFINE_CONFIG_SPEC", "full")
    cfg = Config.from_env()
    assert cfg.max_tasks_per_env == 64
    assert cfg.tasks_per_batch == 4
    assert _env_timeout(cfg, "affine:ded") <= 300
    assert _env_timeout(cfg, "game") <= 1800


def test_config_json_override(monkeypatch, tmp_path):
    spec_path = tmp_path / "affine-e2e.json"
    spec_path.write_text(
        json.dumps(
            {
                "max_tasks_per_env": 12,
                "tasks_per_batch": 3,
                "env_overrides": {
                    "affine:ded": {"params": {"timeout": 45}},
                    "distill": {"mem_limit": "3g"},
                },
            }
        )
    )
    monkeypatch.setenv("AFFINE_CONFIG_SPEC", str(spec_path))
    cfg = Config.from_env()
    assert cfg.max_tasks_per_env == 12
    assert cfg.tasks_per_batch == 3
    assert _env_timeout(cfg, "affine:ded") == 45
    distill = [spec for spec in cfg.environments if spec.name == "distill"][0]
    assert distill.mem_limit == "3g"


def test_config_missing_spec_path(monkeypatch):
    monkeypatch.setenv("AFFINE_CONFIG_SPEC", "/tmp/does-not-exist-affine-e2e.json")
    with pytest.raises(FileNotFoundError):
        Config.from_env()
