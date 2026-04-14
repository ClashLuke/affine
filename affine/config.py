from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path


@dataclass(frozen=True)
class EnvSpec:
    name: str
    image: str
    params: dict = field(default_factory=dict)
    env_vars: dict = field(default_factory=dict)
    mem_limit: str = "8g"


@dataclass
class Config:
    netuid: int = 120
    wallet_name: str = "default"
    hotkey_name: str = "default"
    network: str = "finney"
    subtensor_endpoint: str = "finney"
    subtensor_fallback: str = "wss://lite.sub.latent.to:443"
    max_tasks_per_env: int = 200
    tasks_per_batch: int = 4
    k_init: float = 3.0
    k_final: float = 1.0
    k_halflife: int = 7200
    health_check_timeout: int = 300
    log_level: str = "INFO"
    environments: tuple[EnvSpec, ...] = ()

    @classmethod
    def from_env(cls) -> Config:
        cfg = cls(
            netuid=int(os.getenv("NETUID", "120")),
            wallet_name=os.getenv("BT_WALLET_COLD", "default"),
            hotkey_name=os.getenv("BT_WALLET_HOT", "default"),
            network=os.getenv("BT_NETWORK", "finney"),
            subtensor_endpoint=os.getenv("SUBTENSOR_ENDPOINT", "finney"),
            subtensor_fallback=os.getenv("SUBTENSOR_FALLBACK", "wss://lite.sub.latent.to:443"),
            max_tasks_per_env=int(os.getenv("MAX_TASKS_PER_ENV", "200")),
            tasks_per_batch=int(os.getenv("TASKS_PER_BATCH", "4")),
            k_init=float(os.getenv("K_INIT", "3.0")),
            k_final=float(os.getenv("K_FINAL", "1.0")),
            k_halflife=int(os.getenv("K_HALFLIFE", "7200")),
            health_check_timeout=int(os.getenv("HEALTH_CHECK_TIMEOUT", "300")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            environments=_default_environments(),
        )
        spec = os.getenv("AFFINE_CONFIG_SPEC", "").strip()
        return _apply_config_spec(cfg, spec) if spec else cfg


def _apply_config_spec(cfg: Config, spec: str) -> Config:
    normalized = spec.lower()
    if normalized in {"default", "full", "smoke"}:
        return _apply_profile(cfg, normalized)

    path = Path(spec).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"AFFINE_CONFIG_SPEC path not found: {path}")
    return _apply_json_overrides(cfg, json.loads(path.read_text()))


def _apply_profile(cfg: Config, profile: str) -> Config:
    if profile == "default":
        return cfg
    if profile == "full":
        return replace(cfg,
            max_tasks_per_env=64,
            environments=_with_timeouts(cfg.environments, default_timeout=300, game_timeout=1800),
        )
    if profile == "smoke":
        return replace(cfg,
            max_tasks_per_env=8, tasks_per_batch=2,
            k_init=1.0, k_final=1.0,
            health_check_timeout=min(cfg.health_check_timeout, 180),
            environments=_with_timeouts(cfg.environments, default_timeout=90, game_timeout=420),
        )
    raise ValueError(f"unsupported profile: {profile}")


def _with_timeouts(
    environments: tuple[EnvSpec, ...], *, default_timeout: int, game_timeout: int,
) -> tuple[EnvSpec, ...]:
    out = []
    for spec in environments:
        t = game_timeout if spec.name == "game" else default_timeout
        out.append(replace(spec, params={**spec.params, "timeout": min(int(spec.params.get("timeout", t)), t)}))
    return tuple(out)


def _apply_json_overrides(cfg: Config, raw: dict) -> Config:
    overrides = {}
    for f in fields(Config):
        if f.name != "environments" and f.name in raw:
            overrides[f.name] = type(getattr(cfg, f.name))(raw[f.name])
    return replace(cfg, environments=_apply_env_overrides(cfg.environments, raw), **overrides)


def _apply_env_overrides(current: tuple[EnvSpec, ...], raw: dict) -> tuple[EnvSpec, ...]:
    by_name = {spec.name: spec for spec in current}
    if isinstance(raw.get("env_overrides"), dict):
        for name, override in raw["env_overrides"].items():
            if name not in by_name:
                raise KeyError(f"unknown environment in env_overrides: {name}")
            by_name[name] = _merge_env(by_name[name], override)

    if isinstance(raw.get("environments"), list):
        rebuilt = []
        for item in raw["environments"]:
            if not isinstance(item, dict) or "name" not in item:
                raise ValueError("each environments item must be an object with a name")
            name = str(item["name"])
            base = by_name.get(name, EnvSpec(name=name, image=str(item.get("image", ""))))
            rebuilt.append(_merge_env(base, item))
        return tuple(rebuilt)

    return tuple(by_name[spec.name] for spec in current)


def _merge_env(base: EnvSpec, override: dict) -> EnvSpec:
    image = str(override.get("image", base.image))
    if not image:
        raise ValueError(f"environment '{base.name}' has empty image")
    return replace(base,
        name=str(override.get("name", base.name)),
        image=image,
        params={**base.params, **override.get("params", {})},
        env_vars={**base.env_vars, **override.get("env_vars", {})},
        mem_limit=str(override.get("mem_limit", base.mem_limit)),
    )


ENV_REGISTRY: dict[str, EnvSpec] = {
    "ded": EnvSpec(
        name="affine:ded", image="affinefoundation/affine-env:v4",
        params={"task_type": "ded", "temperature": 0.0, "timeout": 600},
    ),
    "abd": EnvSpec(
        name="affine:abd", image="affinefoundation/affine-env:v4",
        params={"task_type": "abd", "temperature": 0.0, "timeout": 600},
    ),
    "game": EnvSpec(
        name="game", image="affinefoundation/game:openspiel",
        params={"temperature": 0.0, "timeout": 7200},
    ),
    "distill": EnvSpec(
        name="distill", image="affinefoundation/distill:latest",
        params={"temperature": 0.0, "timeout": 600}, mem_limit="2g",
    ),
}


def _default_environments() -> tuple[EnvSpec, ...]:
    return tuple(ENV_REGISTRY.values())
