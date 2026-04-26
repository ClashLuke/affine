from __future__ import annotations
import json
import math
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
    # Inclusive task-id range. Distill's R2 bucket is 1-indexed and ~2000 entries;
    # affine-env's HF dataset has ~23k rows; game encodes config in task_id (no upper
    # bound). The validator draws task_ids uniformly per dwell iteration so both miners
    # see the same task. Default chosen to be safe for all known envs.
    task_range: tuple[int, int] = (1, 2000)


@dataclass
class Config:
    netuid: int = 120
    wallet_name: str = "default"
    hotkey_name: str = "default"
    subtensor_endpoint: str = "finney"
    subtensor_fallback: str = "wss://lite.sub.latent.to:443"
    dwell: int = 50                       # env picks per duel (one king sample + one challenger sample per pick)
    k_init: float = 3.0                   # starting dethronement threshold
    k_final: float = 1.0                  # asymptotic threshold
    k_halflife: int = 7200                # blocks for k decay half-life (~24h on 12s blocks)
    sigma_beta: float = 1.0               # std of β prior (env difficulty); ±2 logits at 1.96σ
    sigma_alpha: float = 0.5              # std of log α prior; a ∈ [0.38, 2.66] at 1.96σ
    evidence_path: str = "./.affine/evidence.jsonl"
    provision_timeout: int = 900          # seconds to wait for a vLLM slot to become /v1/models ready
    log_level: str = "INFO"
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
            dwell=int(os.getenv("AFFINE_DWELL", "50")),
            k_init=float(os.getenv("AFFINE_K_INIT", "3.0")),
            k_final=float(os.getenv("AFFINE_K_FINAL", "1.0")),
            k_halflife=int(os.getenv("AFFINE_K_HALFLIFE", "7200")),
            sigma_beta=float(os.getenv("AFFINE_SIGMA_BETA", "1.0")),
            sigma_alpha=float(os.getenv("AFFINE_SIGMA_ALPHA", "0.5")),
            evidence_path=os.getenv("AFFINE_EVIDENCE_PATH", "./.affine/evidence.jsonl"),
            provision_timeout=int(os.getenv("AFFINE_PROVISION_TIMEOUT", "900")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            environments=_default_environments(),
        )
        spec = os.getenv("AFFINE_CONFIG_SPEC", "").strip()
        cfg = _apply_config_spec(cfg, spec) if spec else cfg
        _validate(cfg)
        return cfg


def _reject_nonfinite(c: str):
    raise ValueError(f"AFFINE_CONFIG_SPEC: non-finite JSON constant {c!r} is not allowed")


def _apply_config_spec(cfg: Config, spec: str) -> Config:
    if spec in _PROFILES:
        return _apply_json_overrides(cfg, _PROFILES[spec])
    path = Path(spec).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"AFFINE_CONFIG_SPEC={spec!r} is not a known profile "
            f"({', '.join(_PROFILES)}) and not a file path"
        )
    # parse_constant rejects NaN/Infinity. Without this, a NaN timeout in the
    # spec would silently flow into asyncio.wait, which returns immediately for
    # NaN → every sample looks like a timeout → every miner appears to lose.
    return _apply_json_overrides(cfg, json.loads(path.read_text(), parse_constant=_reject_nonfinite))


# Named profiles. default = shipping config; full = mid-budget gate; smoke = fast CI gate.
# Per-env timeouts under env_overrides so they ride on top of ENV_REGISTRY.
_PROFILES: dict[str, dict] = {
    "default": {},
    "full": {
        "dwell": 32,
        "k_init": 3.0,
        "env_overrides": {
            "affine:ded": {"params": {"timeout": 300}},
            "affine:abd": {"params": {"timeout": 300}},
            "game": {"params": {"timeout": 1800}},
            "distill": {"params": {"timeout": 300}},
        },
    },
    "smoke": {
        "dwell": 8,
        "k_init": 1.0,
        "env_overrides": {
            "affine:ded": {"params": {"timeout": 90}},
            "affine:abd": {"params": {"timeout": 90}},
            "game": {"params": {"timeout": 420}},
            "distill": {"params": {"timeout": 90}},
        },
    },
}


_TOP_LEVEL_KEYS = frozenset(f.name for f in fields(Config)) | {"env_overrides", "environments"}


def _apply_json_overrides(cfg: Config, raw: dict) -> Config:
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
        if f.type is int or f.type == "int":
            if isinstance(v, bool) or not isinstance(v, int):
                raise TypeError(f"config key {f.name!r}: expected int, got {type(v).__name__}={v!r}")
        elif f.type is float or f.type == "float":
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"config key {f.name!r}: expected float, got {type(v).__name__}={v!r}")
        elif f.type is str or f.type == "str":
            if not isinstance(v, str):
                raise TypeError(f"config key {f.name!r}: expected str, got {type(v).__name__}={v!r}")
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
            base = by_name.get(name, EnvSpec(name=name, image=str(item.get("image", ""))))
            rebuilt.append(_merge_env(base, item))
        return tuple(rebuilt)

    return tuple(by_name[spec.name] for spec in current)


def _validate(cfg: Config) -> None:
    """Reject configs that would crash deeper in the stack with confusing errors.
    NaN/Inf gets through `<=`/`>=` because all comparisons against NaN are False;
    use math.isfinite explicitly so Priors(σ=nan) doesn't poison the IRT fit."""
    if cfg.dwell <= 0:
        raise ValueError(f"dwell must be > 0, got {cfg.dwell}")
    if cfg.k_halflife <= 0:
        raise ValueError(f"k_halflife must be > 0, got {cfg.k_halflife}")
    for n in ("k_init", "k_final"):
        v = getattr(cfg, n)
        if not math.isfinite(v):
            raise ValueError(f"{n} must be finite, got {v}")
    # k_final must be strictly positive: with k_final=0, k decays to 0 over a long
    # reign and any positive z (even noise-driven) would dethrone. Negative would
    # dethrone strictly worse challengers. The minimum useful threshold is 1σ.
    if cfg.k_final <= 0:
        raise ValueError(f"k_final must be > 0, got {cfg.k_final}")
    if not (cfg.k_final <= cfg.k_init):
        raise ValueError(f"k_final ({cfg.k_final}) must be <= k_init ({cfg.k_init})")
    for n in ("sigma_beta", "sigma_alpha"):
        v = getattr(cfg, n)
        if not (math.isfinite(v) and v > 0):
            raise ValueError(f"{n} must be finite and > 0, got {v}")
    if cfg.provision_timeout <= 0:
        raise ValueError(f"provision_timeout must be > 0, got {cfg.provision_timeout}")
    if not cfg.environments:
        raise ValueError("environments must not be empty")
    seen: set[str] = set()
    for spec in cfg.environments:
        if spec.name in seen:
            raise ValueError(f"duplicate environment name: {spec.name!r}")
        seen.add(spec.name)
        # Per-env timeout drives asyncio.wait deadlines. NaN/Infinity make
        # asyncio.wait return immediately with empty done set → sample looks
        # timed out → False (miner-loss) for every sample. Negative is
        # equivalent. Validate at config load to fail fast.
        t = spec.params.get("timeout")
        if t is not None and not (isinstance(t, (int, float))
                                  and not isinstance(t, bool)
                                  and math.isfinite(t) and t > 0):
            raise ValueError(f"env '{spec.name}': params['timeout'] must be finite > 0, got {t!r}")


def _merge_env(base: EnvSpec, override: dict) -> EnvSpec:
    image = str(override.get("image", base.image))
    if not image:
        raise ValueError(f"environment '{base.name}' has empty image")
    tr = override.get("task_range", base.task_range)
    if not (isinstance(tr, (list, tuple)) and len(tr) == 2 and int(tr[0]) <= int(tr[1])):
        raise ValueError(f"environment '{base.name}' has invalid task_range: {tr!r}")
    return replace(base,
        name=str(override.get("name", base.name)),
        image=image,
        params={**base.params, **override.get("params", {})},
        env_vars={**base.env_vars, **override.get("env_vars", {})},
        mem_limit=str(override.get("mem_limit", base.mem_limit)),
        task_range=(int(tr[0]), int(tr[1])),
    )


# Cold-start baseline: on empty evidence, elect the first registered miner whose
# model string matches one of these. Avoids the degenerate "argmax on all-zero
# θ̂ returns uid at index 0" cold-start. Falls through to argmax if none match.
BASELINE_MODELS: tuple[str, ...] = ("Qwen/Qwen3-32B", "openai/gpt-oss-120b")


ENV_REGISTRY: dict[str, EnvSpec] = {
    # Ranges verified by scripts/probe_envs.py against the live container images:
    #   ded/abd: HF dataset AffineFoundation/rl-python has 23303 rows, 0-indexed.
    #   distill: R2 bucket has rollouts task_00000000001..task_00000000002399 only;
    #            task_id=0 is 404 (file uses 1-indexed naming).
    #   game:    task_id encodes game_idx*1e8 + config_id; game_idx=0 (goofspiel)
    #            with 1e8 configs is plenty of variety and keeps the IRT "game"
    #            env homogeneous (mixing 22 games under one β_game inflates noise).
    "ded": EnvSpec(
        name="affine:ded", image="affinefoundation/affine-env:v4",
        params={"task_type": "ded", "temperature": 0.0, "timeout": 600},
        task_range=(0, 23302),
    ),
    "abd": EnvSpec(
        name="affine:abd", image="affinefoundation/affine-env:v4",
        params={"task_type": "abd", "temperature": 0.0, "timeout": 600},
        task_range=(0, 23302),
    ),
    "game": EnvSpec(
        name="game", image="affinefoundation/game:openspiel",
        params={"temperature": 0.0, "timeout": 7200},
        task_range=(0, 99_999_999),
    ),
    "distill": EnvSpec(
        name="distill", image="affinefoundation/distill:latest",
        params={"temperature": 0.0, "timeout": 600}, mem_limit="2g",
        task_range=(1, 2399),
    ),
}


def _default_environments() -> tuple[EnvSpec, ...]:
    return tuple(ENV_REGISTRY.values())
