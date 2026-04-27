import json

import pytest

from affine.config import Config


def _env_timeout(cfg: Config, name: str) -> int:
    for spec in cfg.environments:
        if spec.name == name:
            return int(spec.params["timeout"])
    raise AssertionError(f"missing env {name}")


def test_config_defaults():
    cfg = Config.from_env()
    assert cfg.dwell_batch == 1
    assert cfg.k_init == 3.0
    assert cfg.k_final == 1.0
    assert cfg.k_halflife == 7200
    assert cfg.sigma_beta == 1.0
    assert cfg.sigma_alpha == 0.5
    assert cfg.evidence_path.endswith("evidence.jsonl")
    assert {spec.name for spec in cfg.environments} >= {"affine:ded", "affine:abd", "game", "distill"}


def test_config_env_overrides(monkeypatch):
    monkeypatch.setenv("AFFINE_DWELL_BATCH", "7")
    monkeypatch.setenv("AFFINE_K_INIT", "2.5")
    monkeypatch.setenv("AFFINE_K_FINAL", "0.8")
    monkeypatch.setenv("AFFINE_K_HALFLIFE", "3600")
    monkeypatch.setenv("AFFINE_SIGMA_BETA", "2.0")
    monkeypatch.setenv("AFFINE_EVIDENCE_PATH", "/tmp/ev.jsonl")
    cfg = Config.from_env()
    assert cfg.dwell_batch == 7
    assert cfg.k_init == 2.5
    assert cfg.k_final == 0.8
    assert cfg.k_halflife == 3600
    assert cfg.sigma_beta == 2.0
    assert cfg.evidence_path == "/tmp/ev.jsonl"


def test_config_json_override(monkeypatch, tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({
        "dwell_batch": 12,
        "k_init": 2.0,
        "env_overrides": {
            "affine:ded": {"params": {"timeout": 45}},
            "distill": {"mem_limit": "3g"},
        },
    }))
    monkeypatch.setenv("AFFINE_CONFIG_SPEC", str(spec_path))
    cfg = Config.from_env()
    assert cfg.dwell_batch == 12
    assert cfg.k_init == 2.0
    assert _env_timeout(cfg, "affine:ded") == 45
    distill = [spec for spec in cfg.environments if spec.name == "distill"][0]
    assert distill.mem_limit == "3g"


def test_config_env_overrides_as_list_rejected(monkeypatch, tmp_path):
    """Regression: a list of {name, ...} entries was silently dropped (the
    isinstance(dict) check failed, no error). Config then ran with default
    timeouts despite the user thinking they'd overridden them."""
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({
        "env_overrides": [
            {"name": "affine:ded", "params": {"timeout": 45}},
        ],
    }))
    monkeypatch.setenv("AFFINE_CONFIG_SPEC", str(spec_path))
    with pytest.raises(TypeError, match="env_overrides"):
        Config.from_env()


def test_config_missing_spec_path(monkeypatch):
    monkeypatch.setenv("AFFINE_CONFIG_SPEC", "/tmp/does-not-exist-affine-e2e.json")
    with pytest.raises(FileNotFoundError):
        Config.from_env()


def test_config_smoke_profile(monkeypatch):
    monkeypatch.setenv("AFFINE_CONFIG_SPEC", "smoke")
    cfg = Config.from_env()
    assert cfg.k_init == 1.0
    assert _env_timeout(cfg, "affine:ded") == 90
    assert _env_timeout(cfg, "game") == 420


def test_config_full_profile(monkeypatch):
    monkeypatch.setenv("AFFINE_CONFIG_SPEC", "full")
    cfg = Config.from_env()
    assert cfg.k_init == 3.0
    assert _env_timeout(cfg, "affine:abd") == 300
    assert _env_timeout(cfg, "game") == 1800


def test_config_default_profile(monkeypatch):
    monkeypatch.setenv("AFFINE_CONFIG_SPEC", "default")
    cfg = Config.from_env()
    # default profile is a no-op overlay on Config.from_env() defaults
    assert cfg.k_init == 3.0
    assert _env_timeout(cfg, "affine:ded") == 600


@pytest.mark.parametrize("var,val,msg", [
    ("AFFINE_SIGMA_ALPHA", "nan", "sigma_alpha"),
    ("AFFINE_SIGMA_ALPHA", "inf", "sigma_alpha"),
    ("AFFINE_SIGMA_BETA", "-1", "sigma_beta"),
    ("AFFINE_K_INIT", "nan", "k_init"),
    ("AFFINE_K_FINAL", "inf", "k_final"),
    ("AFFINE_K_FINAL", "0", "k_final must be > 0"),
    ("AFFINE_K_FINAL", "-0.5", "k_final must be > 0"),
])
def test_config_rejects_non_finite_or_nonpositive(monkeypatch, var, val, msg):
    monkeypatch.setenv(var, val)
    with pytest.raises(ValueError, match=msg):
        Config.from_env()


def test_config_rejects_empty_environments():
    import dataclasses
    from affine.config import _validate
    cfg = dataclasses.replace(Config.from_env(), environments=())
    with pytest.raises(ValueError, match="environments must not be empty"):
        _validate(cfg)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 0, -0.5])
def test_config_rejects_nonfinite_or_nonpositive_env_timeout(monkeypatch, tmp_path, bad):
    """Per-env timeout flows into asyncio.wait. NaN/inf/0/negative all turn every
    sample into a fake miner-loss. Reject at config load."""
    spec = tmp_path / "cfg.json"
    # JSON has no NaN/Infinity literals — Python's json.loads accepts them only
    # because parse_constant defaults to permissive. We force the issue by
    # rendering them as JSON literals here; the loader must reject the file.
    if bad != bad:  # NaN
        spec.write_text('{"env_overrides":{"affine:ded":{"params":{"timeout":NaN}}}}')
    elif bad in (float("inf"), float("-inf")):
        spec.write_text('{"env_overrides":{"affine:ded":{"params":{"timeout":Infinity}}}}')
    else:
        spec.write_text(json.dumps({"env_overrides":{"affine:ded":{"params":{"timeout": bad}}}}))
    monkeypatch.setenv("AFFINE_CONFIG_SPEC", str(spec))
    with pytest.raises(ValueError):
        Config.from_env()
