"""Policy versioning, evaluation_version hash, sampler info-gain.

D0 of notes/eval-target.md: the eval has a versioned policy `π_e` and a
scalar log-odds rating `R_c = logit(S_c)`. Decisions live on `R_j - R_i`.
This module is the place where π, ρ, score-collapse, and adaptive-sampler
information-gain live, decoupled from the IRT mechanics in `irt.py`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from .irt import CalibrationSnapshot, DecisionState


SCORE_DEFINITION_VERSION = "v1"
DEFAULT_VIEW_POLICY = "first_observation"


def normalize_pi(pi: dict[str, float]) -> dict[str, float]:
    """Normalize π to sum 1 over its keys; reject negative or all-zero."""
    if not pi:
        raise ValueError("pi must not be empty")
    if any(v < 0 for v in pi.values()):
        raise ValueError(f"pi values must be non-negative; got {pi}")
    total = float(sum(pi.values()))
    if total <= 0:
        raise ValueError(f"pi sum must be > 0; got {total}")
    return {k: float(v) / total for k, v in pi.items()}


def normalize_rho(rho: dict[str, float], env_ids: list[str]) -> list[float]:
    """Project ρ onto `env_ids`, normalize. Default uniform over env_ids."""
    if not env_ids:
        return []
    if not rho:
        return [1.0 / len(env_ids)] * len(env_ids)
    proj = [float(rho.get(e, 0.0)) for e in env_ids]
    total = sum(proj)
    if total <= 0:
        return [1.0 / len(env_ids)] * len(env_ids)
    return [v / total for v in proj]


def evaluation_version_hash(
    *,
    measurement_env_set: dict[str, str],   # env_id -> env_version
    pi: dict[str, float],
    score_definition_version: str = SCORE_DEFINITION_VERSION,
    rho: dict[str, float] | None = None,
    view_policy: str = DEFAULT_VIEW_POLICY,
    hyperparams: dict | None = None,
) -> str:
    """Hash that bumps when any element of "what is being measured" changes.

    `measurement_env_set` maps env_id → env_version for envs whose cells are
    in archive_fit (calibration_only + score_active).
    `pi` is the score policy over score_active envs.
    Other arguments versioned per D10/D2.
    """
    payload = {
        "measurement_env_set": sorted(measurement_env_set.items()),
        "pi": sorted((k, float(v)) for k, v in normalize_pi(pi).items()),
        "score_definition_version": score_definition_version,
        "rho": sorted((k, float(v)) for k, v in (rho or {}).items()),
        "view_policy": view_policy,
        "hyperparams": dict(sorted((hyperparams or {}).items())),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:32]


@dataclass(frozen=True)
class EnvCost:
    """Running average pair latency per env, used in the sampler's
    `gain_e / cost_e`. Defaults to 1.0 when no observations yet."""
    total_latency: float = 0.0
    count: int = 0

    def update(self, latency_s: float) -> "EnvCost":
        return EnvCost(self.total_latency + max(latency_s, 0.0), self.count + 1)

    def cost(self) -> float:
        if self.count == 0:
            return 1.0
        return max(self.total_latency / self.count, 1e-3)


def sampler_pick_env(
    decision: DecisionState,
    pi: dict[str, float],
    costs: dict[str, EnvCost],
    n_per_env: dict[str, int],
    *,
    n_min_per_env: int = 1,
) -> str:
    """Pick the next env to sample for the challenger.

    Cold-start: any env with `n_per_env[e] < n_min_per_env` is preferred,
    lowest count wins, ties by env_id. Steady-state: argmax of expected
    info-gain on `R_j - R_i` per cost.

    Score (D6):
        gain_e = w_e · (∂R/∂θ_j · σ_θ_j_now)^2 / cost_e
              ≈ a_e^2 · σ_je(1-σ_je) · Var(θ_j) · (collapsed gradient terms) / cost_e
    Simplified (frozen nuisance, decision_fit Var(θ_j) is in cov_active[0,0]):
        gain_e ∝ a_e^2 · σ(η_je)(1 - σ(η_je)) · π_e · ... / cost_e
    We use a faithful-but-fast approximation: weight per-env Bernoulli info
    by a_e^2 (slope) and π_e (policy weight).
    """
    snap = decision.snapshot
    env_ids = list(snap.env_ids)
    if not env_ids:
        raise ValueError("no envs in calibration snapshot")
    cold = [(e, n_per_env.get(e, 0)) for e in env_ids if n_per_env.get(e, 0) < n_min_per_env]
    if cold:
        cold.sort(key=lambda kv: (kv[1], kv[0]))
        return cold[0][0]
    pi_norm = normalize_pi({e: pi.get(e, 0.0) for e in env_ids}) if any(pi.get(e, 0.0) > 0 for e in env_ids) else {e: 1.0 / len(env_ids) for e in env_ids}

    mu = snap.mu
    beta = np.asarray(snap.beta, dtype=float)
    log_a = np.asarray(snap.log_a, dtype=float)
    a = np.exp(np.clip(log_a, -10.0, 10.0))
    theta_j = decision.theta_j
    eta_j = mu + beta + a * theta_j
    p_j = 1.0 / (1.0 + np.exp(-np.clip(eta_j, -50.0, 50.0)))
    w_j = p_j * (1.0 - p_j)
    var_theta_j = max(float(decision.cov_active[0, 0]), 1e-9)

    best_env = env_ids[0]
    best_score = -math.inf
    for idx, e in enumerate(env_ids):
        cost = costs.get(e, EnvCost()).cost()
        # Information gain: ∂R/∂θ_j per env contributes ~ π_e · w_je · a_e
        # ; total gain in Var(R_j) when sampling env e is ~ a_e^2 · w_je / (1 / Var(θ_j) + ...).
        # Use simplified `a^2 · w · π_e · Var(θ_j) / cost`.
        gain = (a[idx] ** 2) * w_j[idx] * pi_norm.get(e, 0.0) * var_theta_j / cost
        # UCB-style exploration: add a small bonus for under-sampled envs
        n_e = n_per_env.get(e, 0)
        explore = 0.5 * math.sqrt(math.log(max(sum(n_per_env.values()), 1) + 1.0) / (n_e + 1))
        score = gain + 1e-6 * explore
        if score > best_score:
            best_score = score
            best_env = e
    return best_env


def serving_hash_for(slot) -> str:
    """Per-slot serving identity hash. A best-effort placeholder until
    we wire real serving-config introspection. Includes model + revision +
    base_url so two slots running different artifacts at the same chain
    (model, revision) get distinct serving_hashes."""
    parts = [
        f"model\0{getattr(slot, 'model', '')}\0",
        f"revision\0{getattr(slot, 'revision', '')}\0",
        f"base_url\0{getattr(slot, 'base_url', '')}\0",
        f"provider\0{getattr(slot, 'provider', '')}\0",
    ]
    return hashlib.sha256("".join(parts).encode()).hexdigest()[:16]


def env_versioning(spec) -> tuple[str, str, str]:
    """Best-effort (env_version, task_spec_hash, grader_hash) from an env spec.

    For MVP these are hashed from the spec's params + entrypoint; bumping any
    of them constitutes a measurement change per D10.
    """
    blob = json.dumps({
        "name": spec.name,
        "entrypoint": spec.entrypoint,
        "params": dict(sorted(spec.params.items())),
        "task_range": list(spec.task_range),
    }, sort_keys=True, separators=(",", ":")).encode()
    full = hashlib.sha256(blob).hexdigest()
    return full[:16], full[16:32], full[32:48]
