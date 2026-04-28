import json

import pytest

from affine.config import Config, EnvSpec, _validate
from affine.envs import EnvFactory


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
    assert [spec.name for spec in cfg.environments] == [
        "python", "nfa", "graph", "modular", "sudoku", "boolean", "tree",
    ]


def test_default_env_entrypoints_import():
    for spec in Config.from_env().environments:
        assert EnvFactory(spec.entrypoint).make() is not None


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
            "python": {"params": {"timeout": 45, "lines": 8}},
        },
    }))
    monkeypatch.setenv("AFFINE_CONFIG_SPEC", str(spec_path))
    cfg = Config.from_env()
    assert cfg.dwell_batch == 12
    assert cfg.k_init == 2.0
    assert _env_timeout(cfg, "python") == 45
    py = [spec for spec in cfg.environments if spec.name == "python"][0]
    assert py.params["lines"] == 8


def test_config_env_overrides_as_list_rejected(monkeypatch, tmp_path):
    """Regression: a list of {name, ...} entries was silently dropped (the
    isinstance(dict) check failed, no error). Config then ran with default
    timeouts despite the user thinking they'd overridden them."""
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({
        "env_overrides": [
            {"name": "python", "params": {"timeout": 45}},
        ],
    }))
    monkeypatch.setenv("AFFINE_CONFIG_SPEC", str(spec_path))
    with pytest.raises(TypeError, match="env_overrides"):
        Config.from_env()


def test_config_environments_must_be_list(monkeypatch, tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"environments": {"name": "python"}}))
    monkeypatch.setenv("AFFINE_CONFIG_SPEC", str(spec_path))
    with pytest.raises(TypeError, match="environments must be a list"):
        Config.from_env()


def test_config_spec_must_be_object(monkeypatch, tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps([{"name": "python"}]))
    monkeypatch.setenv("AFFINE_CONFIG_SPEC", str(spec_path))
    with pytest.raises(TypeError, match="must decode to an object"):
        Config.from_env()


def test_config_env_override_rejects_bad_params_shape():
    from affine.config import _apply_env_overrides
    current = (EnvSpec(name="python", entrypoint="affine.envs.python_interpreter:PythonInterpreterEnv"),)
    with pytest.raises(TypeError, match="params must be an object"):
        _apply_env_overrides(current, {"env_overrides": {"python": {"params": []}}})


def test_config_missing_spec_path(monkeypatch):
    monkeypatch.setenv("AFFINE_CONFIG_SPEC", "/tmp/does-not-exist-affine-e2e.json")
    with pytest.raises(FileNotFoundError):
        Config.from_env()


def test_config_smoke_profile(monkeypatch):
    monkeypatch.setenv("AFFINE_CONFIG_SPEC", "smoke")
    cfg = Config.from_env()
    assert cfg.k_init == 1.0
    assert _env_timeout(cfg, "python") == 90
    assert cfg.environments[0].params["lines"] == 16
    assert _env_timeout(cfg, "nfa") == 90
    nfa = [spec for spec in cfg.environments if spec.name == "nfa"][0]
    assert nfa.params["length"] == 8
    boolean = [spec for spec in cfg.environments if spec.name == "boolean"][0]
    assert boolean.params["min_influence"] == 4
    tree = [spec for spec in cfg.environments if spec.name == "tree"][0]
    assert tree.params["n"] == 10
    assert tree.params["max_turns"] == 16


def test_config_full_profile(monkeypatch):
    monkeypatch.setenv("AFFINE_CONFIG_SPEC", "full")
    cfg = Config.from_env()
    assert cfg.k_init == 3.0
    assert _env_timeout(cfg, "python") == 300
    assert _env_timeout(cfg, "boolean") == 300
    assert _env_timeout(cfg, "tree") == 300


def test_config_default_profile(monkeypatch):
    monkeypatch.setenv("AFFINE_CONFIG_SPEC", "default")
    cfg = Config.from_env()
    # default profile is a no-op overlay on Config.from_env() defaults
    assert cfg.k_init == 3.0
    assert _env_timeout(cfg, "python") == 600


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
        spec.write_text('{"env_overrides":{"python":{"params":{"timeout":NaN}}}}')
    elif bad in (float("inf"), float("-inf")):
        spec.write_text('{"env_overrides":{"python":{"params":{"timeout":Infinity}}}}')
    else:
        spec.write_text(json.dumps({"env_overrides":{"python":{"params":{"timeout": bad}}}}))
    monkeypatch.setenv("AFFINE_CONFIG_SPEC", str(spec))
    with pytest.raises(ValueError):
        Config.from_env()


@pytest.mark.parametrize("params,msg", [
    ({"n": 1}, "n must be at least 2"),
    ({"method": "bad"}, "method must be"),
    ({"max_turns": 0}, "max_turns must be > 0"),
    ({"max_queries": -1}, "max_queries must be >= 0"),
    ({"allowed_queries": ["BAD"]}, "unknown allowed_queries"),
    ({"allowed_queries": "DEPTH"}, "allowed_queries"),
])
def test_config_rejects_invalid_tree_params(params, msg):
    cfg = Config(environments=(EnvSpec(
        name="tree",
        entrypoint="affine.envs.tree_reconstruction:TreeReconstructionEnv",
        params={"timeout": 1, **params},
    ),))
    with pytest.raises(ValueError, match=msg):
        _validate(cfg)


@pytest.mark.parametrize("name,entrypoint,params,msg", [
    ("python", "affine.envs.python_interpreter:PythonInterpreterEnv", {"ops": ["NO_SUCH"]}, "unknown ops"),
    ("python", "affine.envs.python_interpreter:PythonInterpreterEnv", {"lines": True}, "lines must be an integer"),
    ("nfa", "affine.envs.nfa_trace:NFATraceEnv", {"states": 2}, "states must be"),
    ("nfa", "affine.envs.nfa_trace:NFATraceEnv", {"alphabet": ""}, "alphabet"),
    ("graph", "affine.envs.graph_path:GraphPathEnv", {"nodes": 3, "min_path_len": 5}, "min_path_len"),
    ("graph", "affine.envs.graph_path:GraphPathEnv", {"edges": 0}, "edges"),
    ("modular", "affine.envs.modular_crt:ModularCRTEnv", {"moduli": 0}, "moduli"),
    ("modular", "affine.envs.modular_crt:ModularCRTEnv", {"steps": -1}, "steps"),
    ("sudoku", "affine.envs.sudoku:SudokuEnv", {"clues": 81}, "clues"),
    ("sudoku", "affine.envs.sudoku:SudokuEnv", {"clues": 60, "min_branch_points": 1}, "clues"),
    ("sudoku", "affine.envs.sudoku:SudokuEnv", {"min_branch_points": 99}, "min_branch_points"),
    ("boolean", "affine.envs.boolean_circuit:BooleanCircuitEnv", {"variables": 2}, "variables"),
    ("boolean", "affine.envs.boolean_circuit:BooleanCircuitEnv", {"gates": 0}, "gates"),
    ("boolean", "affine.envs.boolean_circuit:BooleanCircuitEnv", {"variables": 6}, "min_influence"),
])
def test_config_rejects_invalid_registered_env_params(name, entrypoint, params, msg):
    cfg = Config(environments=(EnvSpec(name=name, entrypoint=entrypoint, params={"timeout": 1, **params}),))
    with pytest.raises(ValueError, match=msg):
        _validate(cfg)


def test_config_rejects_unknown_registered_env_param():
    cfg = Config(environments=(EnvSpec(
        name="nfa",
        entrypoint="affine.envs.nfa_trace:NFATraceEnv",
        params={"timeout": 1, "states": 10, "unknown": 1},
    ),))
    with pytest.raises(KeyError, match="unknown params"):
        _validate(cfg)


@pytest.mark.parametrize("params,msg", [
    ({"temperature": float("nan")}, "temperature"),
    ({"temperature": 3.0}, "temperature"),
    ({"top_p": 0.0}, "top_p"),
    ({"min_p": 2.0}, "min_p"),
    ({"frequency_penalty": 3.0}, "frequency_penalty"),
    ({"presence_penalty": -3.0}, "presence_penalty"),
    ({"repetition_penalty": 0.0}, "repetition_penalty"),
    ({"max_tokens": 0}, "max_tokens"),
    ({"top_k": True}, "top_k"),
    ({"gym_max_steps": 0}, "gym_max_steps"),
    ({"stop": [1]}, "stop"),
    ({"logit_bias": []}, "logit_bias"),
    ({"logit_bias": {"": 1}}, "logit_bias"),
    ({"logit_bias": {"42": 101}}, "logit_bias"),
])
def test_config_rejects_invalid_generation_params(params, msg):
    cfg = Config(environments=(EnvSpec(
        name="python",
        entrypoint="affine.envs.python_interpreter:PythonInterpreterEnv",
        params={"timeout": 1, **params},
    ),))
    with pytest.raises(ValueError, match=msg):
        _validate(cfg)


@pytest.mark.parametrize("task_range", [
    (0.2, 1.9),
    (True, 2),
    (5, 4),
    (-1, 2),
    (0, (1 << 31)),
    [0, "10"],
])
def test_config_rejects_invalid_task_range(task_range):
    cfg = Config(environments=(EnvSpec(
        name="python",
        entrypoint="affine.envs.python_interpreter:PythonInterpreterEnv",
        params={"timeout": 1},
        task_range=task_range,
    ),))
    with pytest.raises(ValueError, match="task_range"):
        _validate(cfg)
