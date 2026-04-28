from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path


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
    dwell_batch: int = 1                  # matched-task pairs kept in flight at all times; sets the parallelism ceiling. Dwell exits only on principled stops (z>k, z<-k, shutdown, slot-dead) — there is no iter cap.
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
            dwell_batch=int(os.getenv("AFFINE_DWELL_BATCH", "1")),
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
        "k_init": 3.0,
        "env_overrides": {
            "python": {"params": {"timeout": 300}},
        },
    },
    "smoke": {
        "k_init": 1.0,
        "env_overrides": {
            "python": {"params": {"timeout": 90, "lines": 16}},
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
        if f.type == "int" and (isinstance(v, bool) or not isinstance(v, int)):
            raise TypeError(f"config key {f.name!r}: expected int, got {type(v).__name__}={v!r}")
        if f.type == "float" and (isinstance(v, bool) or not isinstance(v, (int, float))):
            raise TypeError(f"config key {f.name!r}: expected float, got {type(v).__name__}={v!r}")
        if f.type == "str" and not isinstance(v, str):
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
            base = by_name.get(name, EnvSpec(name=name, entrypoint=str(item.get("entrypoint", ""))))
            rebuilt.append(_merge_env(base, item))
        return tuple(rebuilt)

    return tuple(by_name[spec.name] for spec in current)


def _validate(cfg: Config) -> None:
    """Reject configs that would crash deeper in the stack with confusing errors.
    NaN/Inf gets through `<=`/`>=` because all comparisons against NaN are False;
    use math.isfinite explicitly so Priors(σ=nan) doesn't poison the IRT fit."""
    if cfg.dwell_batch <= 0:
        raise ValueError(f"dwell_batch must be > 0, got {cfg.dwell_batch}")
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
    if cfg.k_final > cfg.k_init:
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
        if not spec.entrypoint:
            raise ValueError(f"environment '{spec.name}' has empty entrypoint")
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
    entrypoint = str(override.get("entrypoint", base.entrypoint))
    if not entrypoint:
        raise ValueError(f"environment '{base.name}' has empty entrypoint")
    tr = override.get("task_range", base.task_range)
    if not (isinstance(tr, (list, tuple)) and len(tr) == 2 and int(tr[0]) <= int(tr[1])):
        raise ValueError(f"environment '{base.name}' has invalid task_range: {tr!r}")
    return replace(base,
        name=str(override.get("name", base.name)),
        entrypoint=entrypoint,
        params={**base.params, **override.get("params", {})},
        task_range=(int(tr[0]), int(tr[1])),
    )


# Cold-start baseline: on empty evidence, elect the first registered miner whose
# model string matches one of these. Avoids the degenerate "argmax on all-zero
# θ̂ returns uid at index 0" cold-start. Falls through to argmax if none match.
BASELINE_MODELS: tuple[str, ...] = ("Qwen/Qwen3-32B", "openai/gpt-oss-120b")


ENV_REGISTRY: dict[str, EnvSpec] = {
    "python": EnvSpec(
        name="python",
        entrypoint="affine.envs.python_interpreter:PythonInterpreterEnv",
        params={"lines": 64, "temperature": 0.0, "max_tokens": 4096, "timeout": 600},
    ),
}


def _default_environments() -> tuple[EnvSpec, ...]:
    return tuple(ENV_REGISTRY.values())
