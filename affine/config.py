from __future__ import annotations
from collections.abc import Iterable
import json
import math
import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from .chain import _truthy_env
from .envs._base import load_env_class


@dataclass(frozen=True)
class EnvSpec:
    name: str
    entrypoint: str
    params: dict = field(default_factory=dict)
    # Inclusive challenge-id range. The validator draws task_ids uniformly per
    # duel iteration so both miners see the same task.
    task_range: tuple[int, int] = (0, (1 << 31) - 1)


@dataclass
class Config:
    netuid: int = 120
    wallet_name: str = "default"
    hotkey_name: str = "default"
    subtensor_endpoint: str = "finney"
    subtensor_fallback: str = "wss://lite.sub.latent.to:443"
    dwell_batch: int = 1                  # matched-task pairs kept in flight at all times; parallelism ceiling.
    db_path: str = "./.affine/affine.sqlite3"
    duel_pairs_per_env: int = 32
    duel_min_discordant: int = 16
    alpha_start: float = 0.005
    alpha_final: float = 0.05
    alpha_halflife: int = 7200
    provision_timeout: int = 900          # seconds to wait for a vLLM slot to become /v1/models ready
    dry_run: bool = False
    log_level: str = "INFO"
    model_skiplist: tuple[str, ...] = ()
    environments: tuple[EnvSpec, ...] = ()

    @classmethod
    def from_env(cls) -> Config:
        endpoint = os.getenv("SUBTENSOR_ENDPOINT", "finney")
        cfg = cls(
            netuid=int(os.getenv("NETUID", "120")),
            wallet_name=os.getenv("BT_WALLET_COLD", "default"),
            hotkey_name=os.getenv("BT_WALLET_HOT", "default"),
            subtensor_endpoint=endpoint,
            subtensor_fallback=os.getenv("SUBTENSOR_FALLBACK", "wss://lite.sub.latent.to:443"),
            dwell_batch=int(os.getenv("AFFINE_DWELL_BATCH", "1")),
            db_path=os.getenv("AFFINE_DB_PATH", "./.affine/affine.sqlite3"),
            duel_pairs_per_env=int(os.getenv("AFFINE_DUEL_PAIRS_PER_ENV", "32")),
            duel_min_discordant=int(os.getenv("AFFINE_DUEL_MIN_DISCORDANT", "16")),
            alpha_start=float(os.getenv("AFFINE_ALPHA_START", "0.005")),
            alpha_final=float(os.getenv("AFFINE_ALPHA_FINAL", "0.05")),
            alpha_halflife=int(os.getenv("AFFINE_ALPHA_HALFLIFE", "7200")),
            provision_timeout=int(os.getenv("AFFINE_PROVISION_TIMEOUT", "900")),
            dry_run=_truthy_env("AFFINE_DRY_RUN"),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
            model_skiplist=parse_model_skiplist(os.getenv("AFFINE_MODEL_SKIPLIST", "")),
            environments=_default_environments(),
        )
        spec = os.getenv("AFFINE_CONFIG_SPEC", "").strip()
        cfg = _apply_config_spec(cfg, spec) if spec else cfg
        _validate(cfg)
        return cfg


def _reject_nonfinite(c: str):
    raise ValueError(f"AFFINE_CONFIG_SPEC: non-finite JSON constant {c!r} is not allowed")


def _reject_inf(s: str) -> float:
    # parse_constant only catches literal NaN/Infinity tokens. A numeric literal
    # like 1e999 parses through float() and overflows to inf, slipping past
    # parse_constant. Reject explicitly here.
    f = float(s)
    if not math.isfinite(f):
        raise ValueError(f"AFFINE_CONFIG_SPEC: non-finite JSON number {s!r} is not allowed")
    return f


def _apply_config_spec(cfg: Config, spec: str) -> Config:
    if spec in _PROFILES:
        return _apply_json_overrides(cfg, _PROFILES[spec])
    path = Path(spec).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"AFFINE_CONFIG_SPEC={spec!r} is not a known profile "
            f"({', '.join(_PROFILES)}) and not a file path"
        )
    return _apply_json_overrides(cfg, json.loads(path.read_text(),
                                                 parse_float=_reject_inf,
                                                 parse_constant=_reject_nonfinite))


# Named profiles. default = shipping config; full = mid-budget gate; smoke = fast CI gate.
# Per-env timeouts under env_overrides so they ride on top of ENV_REGISTRY.
_PROFILES: dict[str, dict] = {
    "default": {},
    "full": {
        "env_overrides": {
            "python": {"params": {"timeout": 300}},
            "nfa": {"params": {"timeout": 300}},
            "graph": {"params": {"timeout": 300}},
            "modular": {"params": {"timeout": 300}},
            "sudoku": {"params": {"timeout": 300}},
            "boolean": {"params": {"timeout": 300}},
            "tree": {"params": {"timeout": 300}},
        },
    },
    "smoke": {
        "env_overrides": {
            "python": {"params": {"timeout": 90, "lines": 16}},
            "nfa": {"params": {"timeout": 90, "states": 7, "length": 8, "accept_count": 2}},
            "graph": {"params": {"timeout": 90, "nodes": 9, "edges": 18, "min_path_len": 3}},
            "modular": {"params": {"timeout": 90, "moduli": 2, "steps": 3}},
            "sudoku": {"params": {"timeout": 90, "clues": 40, "min_branch_points": 0}},
            "boolean": {"params": {"timeout": 90, "variables": 6, "gates": 10, "min_influence": 4}},
            "tree": {"params": {"timeout": 120, "n": 10, "max_queries": 32, "max_turns": 16}},
        },
    },
}


_TOP_LEVEL_KEYS = frozenset(f.name for f in fields(Config)) | {"env_overrides", "environments"}
_JSON_DENIED = frozenset({"dry_run"})  # env-only kill switch, never from config spec


def _apply_json_overrides(cfg: Config, raw: dict) -> Config:
    if not isinstance(raw, dict):
        raise TypeError(f"AFFINE_CONFIG_SPEC must decode to an object, got {type(raw).__name__}")
    denied = set(raw) & _JSON_DENIED
    if denied:
        raise KeyError(f"env-only config keys cannot be set in AFFINE_CONFIG_SPEC: {sorted(denied)}")
    unknown = set(raw) - _TOP_LEVEL_KEYS
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}; "
                       f"did you mean one of {sorted(_TOP_LEVEL_KEYS)}?")
    overrides = {}
    for f in fields(Config):
        if f.name == "environments" or f.name not in raw:
            continue
        v = raw[f.name]
        # Reject silent narrowing: 120.9 → int → 120 is a footgun.
        if f.type == "int" and (isinstance(v, bool) or not isinstance(v, int)):
            raise TypeError(f"config key {f.name!r}: expected int, got {type(v).__name__}={v!r}")
        if f.type == "float" and (isinstance(v, bool) or not isinstance(v, (int, float))):
            raise TypeError(f"config key {f.name!r}: expected float, got {type(v).__name__}={v!r}")
        if f.type == "str" and not isinstance(v, str):
            raise TypeError(f"config key {f.name!r}: expected str, got {type(v).__name__}={v!r}")
        if f.name == "model_skiplist":
            overrides[f.name] = normalize_model_skiplist(v)
            continue
        overrides[f.name] = type(getattr(cfg, f.name))(v)
    return replace(cfg, environments=_apply_env_overrides(cfg.environments, raw), **overrides)


def _apply_env_overrides(current: tuple[EnvSpec, ...], raw: dict) -> tuple[EnvSpec, ...]:
    by_name = {spec.name: spec for spec in current}
    ov = raw.get("env_overrides")
    if ov is not None and not isinstance(ov, dict):
        # A list/array shape is the natural mistake here (the sibling `environments`
        # key DOES accept a list). Silently dropping it lets a tightened-timeout
        # smoke config quietly run with default 600s timeouts — burns Targon credits
        # on a flaky test that "passed" CI.
        raise TypeError(f"env_overrides must be an object {{name: override}}, got {type(ov).__name__}")
    if isinstance(ov, dict):
        for name, override in ov.items():
            if name not in by_name:
                raise KeyError(f"unknown environment in env_overrides: {name}")
            by_name[name] = _merge_env(by_name[name], override)

    if "environments" in raw and not isinstance(raw["environments"], list):
        raise TypeError(f"environments must be a list, got {type(raw['environments']).__name__}")

    if isinstance(raw.get("environments"), list):
        rebuilt: list[EnvSpec] = []
        seen: set[str] = set()
        for item in raw["environments"]:
            if not isinstance(item, dict) or "name" not in item:
                raise ValueError("each environments item must be an object with a name")
            name = str(item["name"])
            if name in seen:
                raise ValueError(f"duplicate environment name in 'environments': {name!r}")
            seen.add(name)
            base = by_name.get(name, EnvSpec(name=name, entrypoint=str(item.get("entrypoint", ""))))
            rebuilt.append(_merge_env(base, item))
        return tuple(rebuilt)

    return tuple(by_name[spec.name] for spec in current)


def _validate(cfg: Config) -> None:
    """Reject configs that would crash deeper in the stack with confusing errors.
    NaN/Inf passes through `<=`/`>=` (all comparisons against NaN are False), so
    we use math.isfinite explicitly."""
    if cfg.dwell_batch <= 0:
        raise ValueError(f"dwell_batch must be > 0, got {cfg.dwell_batch}")
    if cfg.provision_timeout <= 0:
        raise ValueError(f"provision_timeout must be > 0, got {cfg.provision_timeout}")
    if cfg.duel_pairs_per_env <= 0:
        raise ValueError(f"duel_pairs_per_env must be > 0, got {cfg.duel_pairs_per_env}")
    if cfg.duel_min_discordant < 0:
        raise ValueError(f"duel_min_discordant must be >= 0, got {cfg.duel_min_discordant}")
    if cfg.alpha_halflife <= 0:
        raise ValueError(f"alpha_halflife must be > 0, got {cfg.alpha_halflife}")
    for n in ("alpha_start", "alpha_final"):
        v = getattr(cfg, n)
        if not (math.isfinite(v) and 0.0 < v < 1.0):
            raise ValueError(f"{n} must be in (0, 1), got {v}")
    if cfg.alpha_start > cfg.alpha_final:
        raise ValueError(f"alpha_start ({cfg.alpha_start}) must be <= alpha_final ({cfg.alpha_final})")
    _validate_log_level(cfg.log_level)
    _validate_model_skiplist(cfg.model_skiplist)
    if not cfg.environments:
        raise ValueError("environments must not be empty")
    seen: set[str] = set()
    for spec in cfg.environments:
        if spec.name in seen:
            raise ValueError(f"duplicate environment name: {spec.name!r}")
        seen.add(spec.name)
        if not spec.entrypoint:
            raise ValueError(f"environment '{spec.name}' has empty entrypoint")
        _validate_task_range(spec.name, spec.task_range)
        # Per-env timeout drives asyncio.wait deadlines. NaN/Infinity make
        # asyncio.wait return immediately with empty done set → sample looks
        # timed out → False (miner-loss) for every sample. Negative is
        # equivalent. Explicit-null in JSON would slip past `t is not None`
        # and crash float(None) deeper; default-substitution form rejects null
        # at the boundary instead.
        t = spec.params.get("timeout", 600)
        if not (isinstance(t, (int, float))
                and not isinstance(t, bool)
                and math.isfinite(t) and t > 0):
            raise ValueError(f"env '{spec.name}': params['timeout'] must be finite > 0, got {t!r}")
        _validate_env_params(spec.name, spec.entrypoint, spec.params)


_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def _validate_log_level(level: str) -> None:
    if level not in _LOG_LEVELS:
        raise ValueError(f"LOG_LEVEL must be one of {sorted(_LOG_LEVELS)}, got {level!r}")


def parse_model_skiplist(raw: str) -> tuple[str, ...]:
    return normalize_model_skiplist(raw.replace("\n", ",").split(","))


def normalize_model_skiplist(raw: Iterable[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        raise TypeError("model_skiplist must be a list of model ids, not a string")
    if not isinstance(raw, Iterable):
        raise TypeError(f"model_skiplist must be iterable, got {type(raw).__name__}")
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            raise TypeError(f"model_skiplist entries must be strings, got {type(item).__name__}")
        model = item.strip()
        if model and model not in seen:
            out.append(model)
            seen.add(model)
    return tuple(out)


def _validate_model_skiplist(skiplist: tuple[str, ...]) -> None:
    for model in skiplist:
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"model_skiplist entries must be non-empty strings, got {model!r}")


_GENERATION_KEYS = frozenset({
    "api_key", "frequency_penalty", "logit_bias", "max_tokens", "min_p", "presence_penalty",
    "repetition_penalty", "stop", "temperature", "top_k", "top_p", "gym_max_steps",
})


def _validate_env_params(name: str, entrypoint: str, params: dict) -> None:
    if not isinstance(params, dict):
        raise TypeError(f"env '{name}': params must be an object, got {type(params).__name__}")
    cls = load_env_class(entrypoint)
    unknown = set(params) - cls.option_keys - _GENERATION_KEYS - {"timeout"}
    if unknown:
        raise KeyError(f"env '{name}': unknown params: {sorted(unknown)}")
    _validate_generation_params(name, {k: v for k, v in params.items() if k in _GENERATION_KEYS})
    task_params = {k: v for k, v in params.items() if k in cls.option_keys}
    try:
        cls.validate_options(task_params)
    except ValueError as exc:
        raise ValueError(f"env '{name}': {exc}") from exc


def _validate_generation_params(name: str, params: dict) -> None:
    for key, value in params.items():
        if key == "api_key":
            if not isinstance(value, str):
                raise ValueError(f"env '{name}': api_key must be a string")
        elif key in {"temperature"}:
            _finite_number(name, key, value, min_value=0.0, max_value=2.0)
        elif key in {"top_p", "min_p"}:
            _finite_number(name, key, value, min_value=0.0, max_value=1.0, inclusive_min=False)
        elif key in {"frequency_penalty", "presence_penalty"}:
            _finite_number(name, key, value, min_value=-2.0, max_value=2.0)
        elif key == "repetition_penalty":
            _finite_number(name, key, value, min_value=0.0, max_value=4.0, inclusive_min=False)
        elif key == "max_tokens":
            _bounded_int(name, key, value, min_value=1, max_value=65536)
        elif key == "top_k":
            _bounded_int(name, key, value, min_value=0, max_value=100000)
        elif key == "gym_max_steps":
            _bounded_int(name, key, value, min_value=1, max_value=1024)
        elif key == "stop":
            if isinstance(value, str):
                continue
            if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
                raise ValueError(f"env '{name}': stop must be a string or list of strings")
        elif key == "logit_bias":
            if not isinstance(value, dict):
                raise ValueError(f"env '{name}': logit_bias must be an object")
            for token, bias in value.items():
                if not isinstance(token, str) or not token:
                    raise ValueError(f"env '{name}': logit_bias keys must be non-empty strings")
                _finite_number(name, f"logit_bias[{token!r}]", bias, min_value=-100.0, max_value=100.0)


def _bounded_int(name: str, key: str, value, *, min_value: int, max_value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"env '{name}': {key} must be an integer, got {value!r}")
    if not min_value <= value <= max_value:
        raise ValueError(f"env '{name}': {key} must be in [{min_value}, {max_value}], got {value}")
    return value


def _finite_number(
    name: str,
    key: str,
    value,
    *,
    min_value: float,
    max_value: float,
    inclusive_min: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"env '{name}': {key} must be finite, got {value!r}")
    lo_ok = value >= min_value if inclusive_min else value > min_value
    if not (lo_ok and value <= max_value):
        bracket = "[" if inclusive_min else "("
        raise ValueError(f"env '{name}': {key} must be in {bracket}{min_value}, {max_value}], got {value}")
    return float(value)


def _merge_env(base: EnvSpec, override: dict) -> EnvSpec:
    if not isinstance(override, dict):
        raise TypeError(f"environment override for {base.name!r} must be an object, got {type(override).__name__}")
    unknown = set(override) - {"name", "entrypoint", "params", "task_range"}
    if unknown:
        raise KeyError(f"environment override for {base.name!r} has unknown keys: {sorted(unknown)}")
    params = override.get("params", {})
    if not isinstance(params, dict):
        raise TypeError(f"environment override for {base.name!r}: params must be an object")
    entrypoint = str(override.get("entrypoint", base.entrypoint))
    if not entrypoint:
        raise ValueError(f"environment '{base.name}' has empty entrypoint")
    tr = _validate_task_range(base.name, override.get("task_range", base.task_range))
    return replace(base,
        name=str(override.get("name", base.name)),
        entrypoint=entrypoint,
        params={**base.params, **params},
        task_range=tr,
    )


def _validate_task_range(name: str, raw) -> tuple[int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"environment '{name}' has invalid task_range: {raw!r}")
    lo, hi = raw
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (lo, hi)):
        raise ValueError(f"environment '{name}' task_range endpoints must be integers, got {raw!r}")
    if not (0 <= lo <= hi <= (1 << 31) - 1):
        raise ValueError(f"environment '{name}' task_range must be within [0, {((1 << 31) - 1)}], got {raw!r}")
    if lo > hi:
        raise ValueError(f"environment '{name}' has invalid task_range: {raw!r}")
    return lo, hi


# First-ever bootstrap baseline. Recovery never uses aggregate scores; once a
# reign exists, its saved artifact remains the baseline until direct dethrone.
BASELINE_MODELS: tuple[str, ...] = ("Qwen/Qwen3-32B", "openai/gpt-oss-120b")


_BASE_PARAMS = {"temperature": 0.0, "timeout": 600}


def _spec(name: str, entrypoint: str, **params) -> EnvSpec:
    return EnvSpec(name=name, entrypoint=entrypoint, params={**_BASE_PARAMS, **params})


ENV_REGISTRY: dict[str, EnvSpec] = {
    "python": _spec("python", "affine.envs.python_interpreter:PythonInterpreterEnv",
                    lines=64, max_tokens=4096),
    "nfa": _spec("nfa", "affine.envs.nfa_trace:NFATraceEnv",
                 states=10, length=16, accept_count=3, max_tokens=1024),
    "graph": _spec("graph", "affine.envs.graph_path:GraphPathEnv",
                   nodes=16, edges=46, min_path_len=5, max_tokens=2048),
    "modular": _spec("modular", "affine.envs.modular_crt:ModularCRTEnv",
                     moduli=3, steps=5, max_tokens=2048),
    "sudoku": _spec("sudoku", "affine.envs.sudoku:SudokuEnv",
                    clues=36, min_branch_points=2, max_tokens=4096),
    "boolean": _spec("boolean", "affine.envs.boolean_circuit:BooleanCircuitEnv",
                     variables=9, gates=18, min_influence=7, max_tokens=2048),
    "tree": _spec("tree", "affine.envs.tree_reconstruction:TreeReconstructionEnv",
                  n=20, method="prufer", max_queries=64, max_turns=32, max_tokens=4096),
}


def _default_environments() -> tuple[EnvSpec, ...]:
    return tuple(ENV_REGISTRY.values())
