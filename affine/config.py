from __future__ import annotations

import math
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

from .chain import _truthy_env
from .envs._base import load_env_class


@dataclass(frozen=True)
class EnvSpec:
    name: str
    entrypoint: str
    params: dict = field(default_factory=dict)


class IdleStrategy(str, Enum):
    WARM_KING = "warm_king"
    COLD_BOTH = "cold_both"


@dataclass
class Config:
    netuid: int = 120
    wallet_name: str = "default"
    hotkey_name: str = "default"
    subtensor_endpoint: str = "finney"
    subtensor_fallback: str = "wss://lite.sub.latent.to:443"
    db_path: str = "./.affine/affine.sqlite3"
    provision_timeout: int = 900
    slot_dead_run: int = 30
    dry_run: bool = False
    log_level: str = "INFO"
    model_skiplist: tuple[str, ...] = ()
    alpha: float = 0.025
    delta_dethrone: float = 0.02
    delta_hold: float = 0.0
    rounds_max: int = 200
    idle_strategy: IdleStrategy = IdleStrategy.WARM_KING
    pi_overrides: tuple[tuple[str, float], ...] = ()
    environments: tuple[EnvSpec, ...] = ()

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls(
            netuid=int(os.getenv("NETUID", "120")),
            wallet_name=os.getenv("BT_WALLET_COLD", "default"),
            hotkey_name=os.getenv("BT_WALLET_HOT", "default"),
            subtensor_endpoint=os.getenv("SUBTENSOR_ENDPOINT", "finney"),
            subtensor_fallback=os.getenv("SUBTENSOR_FALLBACK", "wss://lite.sub.latent.to:443"),
            db_path=os.getenv("AFFINE_DB_PATH", "./.affine/affine.sqlite3"),
            provision_timeout=int(os.getenv("AFFINE_PROVISION_TIMEOUT", "900")),
            slot_dead_run=int(os.getenv("AFFINE_SLOT_DEAD_RUN", "30")),
            dry_run=_truthy_env("AFFINE_DRY_RUN"),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
            model_skiplist=parse_model_skiplist(os.getenv("AFFINE_MODEL_SKIPLIST", "")),
            alpha=float(os.getenv("AFFINE_ALPHA", "0.025")),
            delta_dethrone=float(os.getenv("AFFINE_DELTA_DETHRONE", "0.02")),
            delta_hold=float(os.getenv("AFFINE_DELTA_HOLD", "0.0")),
            rounds_max=int(os.getenv("AFFINE_ROUNDS_MAX", "200")),
            idle_strategy=IdleStrategy(os.getenv("AFFINE_IDLE_STRATEGY", IdleStrategy.WARM_KING.value).strip()),
            pi_overrides=parse_pi_overrides(os.getenv("AFFINE_PI_OVERRIDES", "")),
            environments=_default_environments(),
        )
        _validate(cfg)
        return cfg


def parse_pi_overrides(raw: str) -> tuple[tuple[str, float], ...]:
    if not raw.strip():
        return ()
    out: list[tuple[str, float]] = []
    for part in raw.replace("\n", ",").split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"AFFINE_PI_OVERRIDES entries must be env=value, got {item!r}")
        env, value = item.split("=", 1)
        out.append((env.strip(), float(value)))
    return tuple(out)


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


def _validate(cfg: Config) -> None:
    if cfg.provision_timeout <= 0:
        raise ValueError(f"provision_timeout must be > 0, got {cfg.provision_timeout}")
    if cfg.slot_dead_run <= 0:
        raise ValueError(f"slot_dead_run must be > 0, got {cfg.slot_dead_run}")
    if not (math.isfinite(cfg.alpha) and 0.0 < cfg.alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {cfg.alpha}")
    if not (math.isfinite(cfg.delta_dethrone) and math.isfinite(cfg.delta_hold)):
        raise ValueError("delta thresholds must be finite")
    if cfg.delta_dethrone < cfg.delta_hold:
        raise ValueError("delta_dethrone must be >= delta_hold")
    if cfg.rounds_max <= 0:
        raise ValueError(f"rounds_max must be > 0, got {cfg.rounds_max}")
    if not isinstance(cfg.idle_strategy, IdleStrategy):
        cfg.idle_strategy = IdleStrategy(str(cfg.idle_strategy))
    _validate_log_level(cfg.log_level)
    for model in cfg.model_skiplist:
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"model_skiplist entries must be non-empty strings, got {model!r}")
    if not cfg.environments:
        raise ValueError("environments must not be empty")
    seen: set[str] = set()
    for spec in cfg.environments:
        if spec.name in seen:
            raise ValueError(f"duplicate environment name: {spec.name!r}")
        seen.add(spec.name)
        if not spec.entrypoint:
            raise ValueError(f"environment '{spec.name}' has empty entrypoint")
        if not isinstance(spec.params, dict):
            raise TypeError(f"env '{spec.name}': params must be an object")
        t = spec.params.get("timeout", 600)
        if not (isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) and t > 0):
            raise ValueError(f"env '{spec.name}': params['timeout'] must be finite > 0, got {t!r}")
        cls = load_env_class(spec.entrypoint)
        task_params = {k: v for k, v in spec.params.items() if k in cls.option_keys}
        try:
            cls.validate_options(task_params)
        except ValueError as exc:
            raise ValueError(f"env '{spec.name}': {exc}") from exc


_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def _validate_log_level(level: str) -> None:
    if level not in _LOG_LEVELS:
        raise ValueError(f"LOG_LEVEL must be one of {sorted(_LOG_LEVELS)}, got {level!r}")


BASELINE_MODELS: tuple[str, ...] = (
    "Qwen/Qwen3-32B",
    "nvidia/Qwen3-Nemotron-32B-RLBFF",
    "OpenBuddy/OpenBuddy-R1-0528-Distill-Qwen3-32B-Preview7-QAT-200Kbett",
)

_BASE_PARAMS = {"temperature": 0.0, "timeout": 600}


def _spec(name: str, entrypoint: str, **params) -> EnvSpec:
    return EnvSpec(name=name, entrypoint=entrypoint, params={**_BASE_PARAMS, **params})


ENV_REGISTRY: dict[str, EnvSpec] = {
    "python": _spec("python", "affine.envs.python_interpreter:PythonInterpreterEnv", lines=64, max_tokens=4096),
    "nfa": _spec("nfa", "affine.envs.nfa_trace:NFATraceEnv", states=10, length=16, accept_count=3, max_tokens=1024),
    "graph": _spec("graph", "affine.envs.graph_path:GraphPathEnv", nodes=16, edges=46, min_path_len=5, max_tokens=2048),
    "modular": _spec("modular", "affine.envs.modular_crt:ModularCRTEnv", moduli=3, steps=5, max_tokens=2048),
    "sudoku": _spec("sudoku", "affine.envs.sudoku:SudokuEnv", clues=36, min_branch_points=2, max_tokens=4096),
    "boolean": _spec(
        "boolean",
        "affine.envs.boolean_circuit:BooleanCircuitEnv",
        variables=9,
        gates=18,
        min_influence=7,
        max_tokens=2048,
    ),
    "tree": _spec(
        "tree",
        "affine.envs.tree_reconstruction:TreeReconstructionEnv",
        n=20,
        method="prufer",
        max_queries=64,
        max_turns=32,
        max_tokens=4096,
    ),
}


def _default_environments() -> tuple[EnvSpec, ...]:
    return tuple(ENV_REGISTRY.values())
