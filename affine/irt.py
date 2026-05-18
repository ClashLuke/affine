"""2PL IRT for the validator's eval core.

Model
-----
For miner c on env e, with `θ_c ~ Normal(0, 1)`:

    η_ce = μ + β_e + a_e · θ_c
    P(Y_ce = 1 | c, e) = σ(η_ce)

with:

    a_e > 0                      env discrimination
    β_e ∈ ℝ, Σ_m ρ_m β_e = 0    env easiness, centered with predeclared ρ
    μ ∈ ℝ                       global intercept
    θ_c ∈ ℝ                     latent miner skill, anchored by the prior

The β centering removes the (μ, β) location alias. The θ ~ N(0,1) prior
identifies the θ scale and location at the prior level (no post-hoc
rescaling, no panel needed).

Identifiability is enforced inside the parameterization: optimizer sees
`(μ, β̃, log_a, θ)`, model uses `β = β̃ - Σ_m ρ_m β̃_m`, log-likelihood
runs against identified `(μ, β, a, θ)`. MAP-invariant under the prior on
log_a.

Decision-time: nuisance is frozen from a calibration snapshot that
excludes both contestants (D5 of notes/eval-target.md). Decision SE
propagates archive nuisance covariance into the rating contrast.

The test estimand is the policy log-odds rating contrast:

    R_c = logit(Σ_e π_e · σ(μ + β_e + a_e θ_c))
    Δ_R = R_j - R_i

with `π_e ≥ 0`, `Σ π_e = 1` predeclared.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class CellTriple:
    """Minimal cell input to the IRT fit. Indices into the model's miner/env
    arrays, plus a 0/1 outcome. Decoupled from `store.Cell` so the IRT module
    has no dependency on SQLite types."""
    miner_idx: int
    env_idx: int
    outcome: int


@dataclass(frozen=True)
class ArchiveSnapshot:
    """Result of `archive_fit`. The full IRT fit over historical cells."""
    miner_ids: tuple[str, ...]
    env_ids: tuple[str, ...]
    rho: tuple[float, ...]              # centering distribution over envs
    mu: float                            # global intercept
    beta: tuple[float, ...]              # per-env easiness (centered)
    log_a: tuple[float, ...]             # per-env log discrimination
    theta: tuple[float, ...]             # per-miner skill
    cov: np.ndarray                      # full Laplace covariance, parameter order:
                                         #   [μ, β̃_0..β̃_{E-1}, log_a_0..log_a_{E-1}, θ_0..θ_{C-1}]
    sigma_log_a: float                   # EB hyperparameter
    n_cells: int

    def fingerprint(self) -> str:
        """16-hex content hash of the archive's identifying parameters.
        Stable across replays of the same data; sensitive to (μ, β, log_a)
        drift. Used for D15 archive-drift replay and decision-time pinning."""
        return _snapshot_fingerprint(
            self.miner_ids, self.env_ids, self.rho,
            self.mu, self.beta, self.log_a,
            self.sigma_log_a, self.n_cells,
        )


@dataclass(frozen=True)
class CalibrationSnapshot:
    """Frozen ψ_archive = (μ, β, log_a) plus archive θ for the champion side.

    Built by `calibration_snapshot(archive, exclude_artifacts={i, j})`.
    Includes per-miner θ for *non-excluded* miners — the champion's θ_i lives
    here when the champion isn't in the exclude set. When both contestants
    are excluded (the standard duel case), champion θ comes from the
    decision_fit alongside challenger θ.
    """
    miner_ids: tuple[str, ...]   # miners INCLUDED in the calibration fit
    env_ids: tuple[str, ...]
    rho: tuple[float, ...]
    mu: float
    beta: tuple[float, ...]
    log_a: tuple[float, ...]
    theta: tuple[float, ...]     # for included miners
    cov_nuisance: np.ndarray     # covariance over (μ, β̃, log_a) only; shape (1+2E, 1+2E)
    sigma_log_a: float
    n_calibration_cells: int

    def fingerprint(self) -> str:
        """16-hex content hash of the calibration's identifying parameters.
        Records the frozen ruler at decision time so a future replay can
        detect material archive drift (D15)."""
        return _snapshot_fingerprint(
            self.miner_ids, self.env_ids, self.rho,
            self.mu, self.beta, self.log_a,
            self.sigma_log_a, self.n_calibration_cells,
        )


def _snapshot_fingerprint(
    miner_ids: tuple[str, ...],
    env_ids: tuple[str, ...],
    rho: tuple[float, ...],
    mu: float,
    beta: tuple[float, ...],
    log_a: tuple[float, ...],
    sigma_log_a: float,
    n_cells: int,
) -> str:
    """Canonical content hash for archive/calibration snapshots."""
    import hashlib
    h = hashlib.sha256()
    for label, value in (
        ("miners", "\0".join(miner_ids)),
        ("envs", "\0".join(env_ids)),
        ("rho", ",".join(f"{x:.12g}" for x in rho)),
        ("mu", f"{mu:.12g}"),
        ("beta", ",".join(f"{x:.12g}" for x in beta)),
        ("log_a", ",".join(f"{x:.12g}" for x in log_a)),
        ("sigma_log_a", f"{sigma_log_a:.12g}"),
        ("n_cells", str(n_cells)),
    ):
        h.update(label.encode() + b"\0" + value.encode() + b"\0")
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class DecisionState:
    """Per-decision-point state: frozen nuisance + fitted contestant θ.

    `theta_jc` is `(θ_j, θ_i)` (challenger, champion). `cov_active` is the
    2x2 inverse Fisher of the active block conditional on frozen nuisance.
    """
    snapshot: CalibrationSnapshot
    challenger_id: str
    champion_id: str
    theta_j: float
    theta_i: float
    cov_active: np.ndarray       # 2x2; row/col order [θ_j, θ_i]
    n_challenger_cells: int
    n_champion_cells_used: int


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _identified_beta(beta_tilde: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """β_e = β̃_e - Σ_m ρ_m β̃_m. Removes the (μ, β) location alias."""
    return beta_tilde - float(np.dot(rho, beta_tilde))


def _identified_a(log_a: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(log_a, -10.0, 10.0))


def _eta(mu: float, beta: np.ndarray, a: np.ndarray, theta: np.ndarray,
        miner_idx: np.ndarray, env_idx: np.ndarray) -> np.ndarray:
    return mu + beta[env_idx] + a[env_idx] * theta[miner_idx]


# -----------------------------------------------------------------------------
# Archive fit
# -----------------------------------------------------------------------------

def archive_fit(
    miner_ids: list[str],
    env_ids: list[str],
    triples: list[CellTriple],
    rho: list[float],
    *,
    sigma_mu: float = 5.0,
    sigma_beta: float = 5.0,
    sigma_log_a: float = 1.0,
    sigma_theta: float = 1.0,
    eb_rounds: int = 5,
    n_restarts: int = 3,
    max_iter: int = 500,
    seed: int = 0,
) -> ArchiveSnapshot:
    """Fit 2PL IRT MAP over all cells.

    `rho` is the env-centering distribution (Σ ρ_e = 1, ρ_e ≥ 0); typically
    uniform 1/E. `sigma_log_a` is the EB hyperparameter for the prior on
    `log a_e` and is updated via empirical Bayes inside the fit. `sigma_theta`
    is fixed at 1 (D2: θ ~ Normal(0, 1) anchors the scale).
    """
    n_miners = len(miner_ids)
    n_envs = len(env_ids)
    if n_miners == 0 or n_envs == 0:
        return _empty_archive(miner_ids, env_ids, rho, sigma_log_a)

    rho_np = np.asarray(rho, dtype=float)
    if rho_np.shape != (n_envs,) or not math.isclose(float(rho_np.sum()), 1.0, abs_tol=1e-8):
        raise ValueError(f"rho must have shape ({n_envs},) and sum to 1; got shape={rho_np.shape}, sum={rho_np.sum()}")
    if (rho_np < 0).any():
        raise ValueError(f"rho must be non-negative; got {rho}")

    miner_idx = np.array([t.miner_idx for t in triples], dtype=np.int64)
    env_idx = np.array([t.env_idx for t in triples], dtype=np.int64)
    outcome = np.array([t.outcome for t in triples], dtype=np.int64)
    if len(triples) == 0:
        return _empty_archive(miner_ids, env_ids, rho, sigma_log_a)

    n_params = 1 + 2 * n_envs + n_miners
    sigmas = {"mu": sigma_mu, "beta": sigma_beta, "log_a": sigma_log_a, "theta": sigma_theta}

    def unpack(x: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mu = float(x[0])
        beta_tilde = x[1:1 + n_envs]
        log_a = x[1 + n_envs:1 + 2 * n_envs]
        theta = x[1 + 2 * n_envs:]
        beta = _identified_beta(beta_tilde, rho_np)
        a = _identified_a(log_a)
        return mu, beta_tilde, beta, log_a, a, theta

    def neg_log_post(x: np.ndarray, sla: float) -> float:
        mu, beta_tilde, beta, log_a, a, theta = unpack(x)
        eta = _eta(mu, beta, a, theta, miner_idx, env_idx)
        nll = float(np.sum(np.logaddexp(0.0, eta) - outcome * eta))
        # Priors
        prior = 0.5 * (mu / sigmas["mu"]) ** 2
        prior += 0.5 * float(np.sum((beta_tilde / sigmas["beta"]) ** 2))
        prior += 0.5 * float(np.sum((log_a / sla) ** 2))
        prior += 0.5 * float(np.sum((theta / sigmas["theta"]) ** 2))
        return nll + prior

    def init(rng: np.random.Generator) -> np.ndarray:
        x0 = np.zeros(n_params)
        x0[0] = 0.0  # μ
        x0[1:1 + n_envs] = 0.0  # β̃
        x0[1 + n_envs:1 + 2 * n_envs] = 0.0  # log a (a = 1)
        x0[1 + 2 * n_envs:] = 0.0  # θ
        # Smart init: per-env empirical pass-rate → β̃; per-miner empirical → θ
        eps = 0.02
        if n_envs > 0:
            env_pass = np.bincount(env_idx, weights=outcome, minlength=n_envs)
            env_total = np.bincount(env_idx, minlength=n_envs)
            env_rate = np.clip(env_pass / np.maximum(env_total, 1), eps, 1 - eps)
            x0[1:1 + n_envs] = np.log(env_rate / (1 - env_rate))
        if n_miners > 0:
            miner_pass = np.bincount(miner_idx, weights=outcome, minlength=n_miners)
            miner_total = np.bincount(miner_idx, minlength=n_miners)
            miner_rate = np.clip(miner_pass / np.maximum(miner_total, 1), eps, 1 - eps)
            x0[1 + 2 * n_envs:] = np.log(miner_rate / (1 - miner_rate)) - np.log(env_rate.mean() / (1 - env_rate.mean()))
        # Random perturbation
        x0 = x0 + rng.standard_normal(n_params) * 0.05
        return x0

    rng = np.random.default_rng(seed)
    sla = sigmas["log_a"]
    best_x: np.ndarray | None = None
    best_fun = math.inf
    for _ in range(eb_rounds):
        for r in range(n_restarts):
            x0 = init(rng)
            res = minimize(neg_log_post, x0, args=(sla,), method="L-BFGS-B",
                           options={"maxiter": max_iter})
            if res.fun < best_fun:
                best_fun = float(res.fun)
                best_x = res.x
        # EB update on σ_log_a, lower-bounded at 0.1 to keep the prior weakly
        # informative. No upper cap: σ_log_a is data-driven by the spread of
        # fitted log_a across envs. Under the θ-contrast rating (D3), the
        # decision SE has no `cov(log_a)` channel, so EB inflation no longer
        # contaminates dethrone decisions.
        if best_x is None:
            break
        _, _, _, log_a_hat, _, _ = unpack(best_x)
        new_sla = max(float(np.sqrt(np.mean(log_a_hat ** 2))), 0.1)
        if abs(new_sla - sla) < 0.01 * sla:
            sla = new_sla
            break
        sla = new_sla

    if best_x is None:
        raise RuntimeError("IRT MAP fit produced no valid solution")

    mu, beta_tilde, beta, log_a, a, theta = unpack(best_x)
    cov = _laplace_covariance(best_x, miner_idx, env_idx, outcome, rho_np,
                               sigmas["mu"], sigmas["beta"], sla, sigmas["theta"],
                               n_envs, n_miners)

    return ArchiveSnapshot(
        miner_ids=tuple(miner_ids),
        env_ids=tuple(env_ids),
        rho=tuple(float(x) for x in rho_np),
        mu=float(mu),
        beta=tuple(float(x) for x in beta),
        log_a=tuple(float(x) for x in log_a),
        theta=tuple(float(x) for x in theta),
        cov=cov,
        sigma_log_a=float(sla),
        n_cells=len(triples),
    )


def _empty_archive(
    miner_ids: list[str], env_ids: list[str], rho: list[float], sigma_log_a: float
) -> ArchiveSnapshot:
    """Cold-start snapshot: all priors, no data."""
    n_envs = len(env_ids)
    n_miners = len(miner_ids)
    n_params = 1 + 2 * n_envs + n_miners
    return ArchiveSnapshot(
        miner_ids=tuple(miner_ids),
        env_ids=tuple(env_ids),
        rho=tuple(rho),
        mu=0.0,
        beta=tuple(0.0 for _ in range(n_envs)),
        log_a=tuple(0.0 for _ in range(n_envs)),
        theta=tuple(0.0 for _ in range(n_miners)),
        cov=np.eye(n_params),
        sigma_log_a=float(sigma_log_a),
        n_cells=0,
    )


def _laplace_covariance(
    x: np.ndarray,
    miner_idx: np.ndarray,
    env_idx: np.ndarray,
    outcome: np.ndarray,
    rho: np.ndarray,
    sigma_mu: float,
    sigma_beta: float,
    sigma_log_a: float,
    sigma_theta: float,
    n_envs: int,
    n_miners: int,
) -> np.ndarray:
    """Observed information matrix at MAP, then invert.

    Parameter ordering: [μ, β̃_0..β̃_{E-1}, log_a_0..log_a_{E-1}, θ_0..θ_{C-1}].

    The β̃ parameter contributes to the model through β = β̃ - Σ_m ρ_m β̃_m.
    Per-cell ∂η/∂β̃_e = (1{e_obs == e} - ρ_e) (since ∂β_e/∂β̃_f = δ_ef - ρ_f
    and we evaluate at observed env e_obs).

    Per-cell ∂η/∂log_a_e = (1{e_obs == e}) · a_e · θ_c.
    Per-cell ∂η/∂θ_c = (1{c_obs == c}) · a_e (where e is the observed env).
    Per-cell ∂η/∂μ = 1.
    """
    n_params = 1 + 2 * n_envs + n_miners
    mu = float(x[0])
    beta_tilde = x[1:1 + n_envs]
    log_a = x[1 + n_envs:1 + 2 * n_envs]
    theta = x[1 + 2 * n_envs:]
    beta = _identified_beta(beta_tilde, rho)
    a = _identified_a(log_a)
    eta = _eta(mu, beta, a, theta, miner_idx, env_idx)
    p = _sigmoid(eta)
    w = p * (1.0 - p)  # per-cell Fisher info weight

    n_obs = len(miner_idx)
    obs = np.arange(n_obs)
    J = np.zeros((n_obs, n_params))
    # μ
    J[obs, 0] = 1.0
    # β̃: per-cell row gets (1 - ρ_e_obs) in column for e_obs and -ρ_f for other f
    # That's a dense block; do it with broadcasting.
    # J[i, 1+f] = (1 if env_idx[i] == f else 0) - ρ_f
    onehot = np.zeros((n_obs, n_envs))
    onehot[obs, env_idx] = 1.0
    J[:, 1:1 + n_envs] = onehot - rho[None, :]
    # log_a: only column for observed env, value a_e * θ_c
    a_obs = a[env_idx]
    theta_obs = theta[miner_idx]
    J[obs, 1 + n_envs + env_idx] = a_obs * theta_obs
    # θ: only column for observed miner, value a_e
    J[obs, 1 + 2 * n_envs + miner_idx] = a_obs

    # H = J^T diag(w) J  + prior Hessian
    H = (J * w[:, None]).T @ J
    # Prior Hessians
    H[0, 0] += 1.0 / sigma_mu ** 2
    for f in range(n_envs):
        H[1 + f, 1 + f] += 1.0 / sigma_beta ** 2
        H[1 + n_envs + f, 1 + n_envs + f] += 1.0 / sigma_log_a ** 2
    for c in range(n_miners):
        H[1 + 2 * n_envs + c, 1 + 2 * n_envs + c] += 1.0 / sigma_theta ** 2

    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    cov = 0.5 * (cov + cov.T)
    return cov


# -----------------------------------------------------------------------------
# Calibration snapshot (D5: leave-contestants-out)
# -----------------------------------------------------------------------------

def calibration_snapshot(
    miner_ids: list[str],
    env_ids: list[str],
    triples: list[CellTriple],
    rho: list[float],
    *,
    exclude_artifacts: set[str],
    sigma_log_a: float = 1.0,
    sigma_mu: float = 5.0,
    sigma_beta: float = 5.0,
    seed: int = 0,
) -> CalibrationSnapshot:
    """Fit IRT MAP over cells excluding `exclude_artifacts`.

    Returns the frozen ψ_archive = (μ, β, log_a) and Laplace covariance over
    nuisance only (the (μ, β̃, log_a) block — θ for non-excluded miners is
    fitted but not part of the frozen-nuisance covariance the decision_fit
    propagates).
    """
    # Filter out excluded miners
    keep_indices = [c for c, mid in enumerate(miner_ids) if mid not in exclude_artifacts]
    if not keep_indices:
        # No miners left; return prior-only snapshot
        n_envs = len(env_ids)
        return CalibrationSnapshot(
            miner_ids=tuple(),
            env_ids=tuple(env_ids),
            rho=tuple(rho),
            mu=0.0,
            beta=tuple(0.0 for _ in range(n_envs)),
            log_a=tuple(0.0 for _ in range(n_envs)),
            theta=tuple(),
            cov_nuisance=np.eye(1 + 2 * n_envs),
            sigma_log_a=float(sigma_log_a),
            n_calibration_cells=0,
        )

    keep_set = set(keep_indices)
    new_miner_ids = [miner_ids[c] for c in keep_indices]
    old_to_new = {old: new for new, old in enumerate(keep_indices)}
    filtered = [
        CellTriple(miner_idx=old_to_new[t.miner_idx], env_idx=t.env_idx, outcome=t.outcome)
        for t in triples
        if t.miner_idx in keep_set
    ]

    archive = archive_fit(
        new_miner_ids, env_ids, filtered, rho,
        sigma_mu=sigma_mu, sigma_beta=sigma_beta, sigma_log_a=sigma_log_a,
        seed=seed,
    )
    n_envs = len(env_ids)
    nuisance_dim = 1 + 2 * n_envs
    cov_nuisance = archive.cov[:nuisance_dim, :nuisance_dim].copy()
    return CalibrationSnapshot(
        miner_ids=archive.miner_ids,
        env_ids=archive.env_ids,
        rho=archive.rho,
        mu=archive.mu,
        beta=archive.beta,
        log_a=archive.log_a,
        theta=archive.theta,
        cov_nuisance=cov_nuisance,
        sigma_log_a=archive.sigma_log_a,
        n_calibration_cells=archive.n_cells,
    )


# -----------------------------------------------------------------------------
# Decision fit: frozen nuisance, fit θ_j and θ_i only
# -----------------------------------------------------------------------------

def decision_fit(
    snapshot: CalibrationSnapshot,
    contestant_triples: list[CellTriple],
    challenger_id: str,
    champion_id: str,
    *,
    sigma_theta: float = 1.0,
) -> DecisionState:
    """Fit only `θ_j, θ_i` against `contestant_triples`, with all nuisance
    `(μ, β, log_a)` frozen from `snapshot`.

    Triples here index miners as 0=challenger, 1=champion. Env indices match
    `snapshot.env_ids`.
    """
    n_envs = len(snapshot.env_ids)
    if any(t.miner_idx not in (0, 1) for t in contestant_triples):
        raise ValueError("decision_fit triples must use miner_idx in {0, 1}")
    mu = snapshot.mu
    beta = np.asarray(snapshot.beta, dtype=float)
    a = np.exp(np.clip(np.asarray(snapshot.log_a, dtype=float), -10.0, 10.0))

    n_chal = sum(1 for t in contestant_triples if t.miner_idx == 0)
    n_champ = sum(1 for t in contestant_triples if t.miner_idx == 1)

    if not contestant_triples:
        # Pure prior: θ_j = θ_i = 0, with prior covariance.
        cov = np.array([[sigma_theta ** 2, 0.0], [0.0, sigma_theta ** 2]])
        return DecisionState(
            snapshot=snapshot,
            challenger_id=challenger_id,
            champion_id=champion_id,
            theta_j=0.0,
            theta_i=0.0,
            cov_active=cov,
            n_challenger_cells=0,
            n_champion_cells_used=0,
        )

    miner_idx = np.array([t.miner_idx for t in contestant_triples], dtype=np.int64)
    env_idx = np.array([t.env_idx for t in contestant_triples], dtype=np.int64)
    outcome = np.array([t.outcome for t in contestant_triples], dtype=np.int64)

    def neg_log_post(theta_pair: np.ndarray) -> float:
        theta = theta_pair  # [θ_j, θ_i]
        eta = mu + beta[env_idx] + a[env_idx] * theta[miner_idx]
        nll = float(np.sum(np.logaddexp(0.0, eta) - outcome * eta))
        prior = 0.5 * float(np.sum((theta_pair / sigma_theta) ** 2))
        return nll + prior

    res = minimize(neg_log_post, np.zeros(2), method="L-BFGS-B",
                   options={"maxiter": 200})
    theta_j, theta_i = float(res.x[0]), float(res.x[1])

    # Active-block covariance: H_active = J^T diag(w) J + diag(1/σ²) where
    # ∂η/∂θ_j = a[e_obs] · 1{miner==0}, similarly for θ_i with miner==1.
    eta = mu + beta[env_idx] + a[env_idx] * res.x[miner_idx]
    p = _sigmoid(eta)
    w = p * (1.0 - p)
    a_obs = a[env_idx]
    is_j = (miner_idx == 0).astype(float)
    is_i = (miner_idx == 1).astype(float)
    H = np.zeros((2, 2))
    H[0, 0] = float(np.sum(w * (a_obs * is_j) ** 2)) + 1.0 / sigma_theta ** 2
    H[1, 1] = float(np.sum(w * (a_obs * is_i) ** 2)) + 1.0 / sigma_theta ** 2
    H[0, 1] = H[1, 0] = float(np.sum(w * (a_obs * is_j) * (a_obs * is_i)))
    try:
        cov_active = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov_active = np.linalg.pinv(H)
    cov_active = 0.5 * (cov_active + cov_active.T)

    return DecisionState(
        snapshot=snapshot,
        challenger_id=challenger_id,
        champion_id=champion_id,
        theta_j=theta_j,
        theta_i=theta_i,
        cov_active=cov_active,
        n_challenger_cells=n_chal,
        n_champion_cells_used=n_champ,
    )


# -----------------------------------------------------------------------------
# Rating contrast (R_c = logit(S_c))
# -----------------------------------------------------------------------------

_R_S_CLAMP = 1e-3  # bound rating gradient at σ-boundary; logit(0.001) ≈ ±6.9


def policy_rating(
    mu: float,
    beta: np.ndarray,
    a: np.ndarray,
    theta_c: float,
    pi: np.ndarray,
) -> tuple[float, float]:
    """Return (S_c, R_c) where S_c = Σ π_e σ(μ + β_e + a_e θ_c), R_c = logit(S_c).

    S is soft-clamped to [_R_S_CLAMP, 1 - _R_S_CLAMP] before logit. This bounds
    R ∈ [-6.9, +6.9] and bounds dR/dS at ≈ 1000. Necessary because when one
    miner saturates the policy distribution (every env's σ(η) ≈ 0 or ≈ 1)
    the rating diverges and the SE blows up. The clamp is conservative for
    realistic miners (which sit well within (0.05, 0.95)) and only matters
    in pathological cold-start / boundary cases.
    """
    eta = mu + beta + a * theta_c
    p = _sigmoid(eta)
    s = float(np.dot(pi, p))
    s_clamp = min(max(s, _R_S_CLAMP), 1.0 - _R_S_CLAMP)
    r = math.log(s_clamp / (1.0 - s_clamp))
    return s, r


def theta_diff_se(
    decision: DecisionState,
    pi: dict[str, float],
) -> tuple[float, float, float]:
    """Compute (Δθ, SE_Δθ, R_diff_diagnostic) at the current decision_fit MAP.

    Under K=0, the decision statistic is the latent BT skill difference:
        Δθ = θ_j − θ_i
    with active-block SE *conditional on the frozen calibration snapshot*:
        SE_Δθ | ψ̂ = √(g^T Σ_active g)        g = (+1, −1)

    **Honest note on conditional SE.** This is the variance of the fitted
    Δθ̂ *conditional on the calibration snapshot* `ψ̂ = (μ̂, β̂, log_â)`. It
    does not propagate the archive's posterior uncertainty in ψ̂ into the
    decision interval. The implicit-function relationship
        dθ̂/dψ̂ = − H_θθ⁻¹ · H_θψ
    means the fitted contestant θs would shift if ψ̂ changed; under
    Stage-1 we treat ψ̂ as a deterministic ruler and accept the conditional
    contract. Production runs with mature calibration cohorts make this
    near-equivalent to the unconditional bound; cold-start runs may
    under-report SE and rely on the calibration sufficiency gate (D-cold)
    to avoid premature decisions. Archive-nuisance propagation is a
    target-doc gap (`notes/eval-target.md` D5) tracked for Stage-2.

    Direction-preserving for ranking: under K=0 with `a_e > 0` and
    `π_e ≥ 0`, the score-space rating `R = logit(Σ π_e σ(η_ce))` is
    monotone in θ_c, so testing on Δθ preserves dethrone direction. The
    threshold is on the prior-SD scale (θ ~ Normal(0,1)).

    Returns `(Δθ, SE_Δθ, R_diff)`. R_diff is computed for replay and
    diagnostic logs; it is **not** the decision statistic.

    For K ≥ 1 (MIRT), the decision must move to a scalar collapse like
    `R_c = logit(S_c)` because there is no canonical θ contrast — that's
    a future migration, not an extension of this rule.
    """
    snap = decision.snapshot
    env_ids = snap.env_ids
    if not all(e in pi for e in env_ids):
        missing = [e for e in env_ids if e not in pi]
        raise ValueError(f"pi missing weights for envs: {missing}")
    pi_arr = np.array([pi[e] for e in env_ids], dtype=float)
    pi_arr = pi_arr / pi_arr.sum() if pi_arr.sum() > 0 else pi_arr

    mu = snap.mu
    beta = np.asarray(snap.beta, dtype=float)
    a = np.exp(np.clip(np.asarray(snap.log_a, dtype=float), -10.0, 10.0))

    # θ-contrast: rating = θ_c. Direction-preserving for ranking; threshold
    # in latent skill SDs anchored by the Normal(0,1) prior.
    delta_theta = float(decision.theta_j - decision.theta_i)
    g_active = np.array([+1.0, -1.0])
    var_theta = float(g_active @ decision.cov_active @ g_active)
    se_theta = math.sqrt(max(var_theta, 0.0))

    # Score-space rating for replay/logging only (not used in the decision).
    _, r_j = policy_rating(mu, beta, a, decision.theta_j, pi_arr)
    _, r_i = policy_rating(mu, beta, a, decision.theta_i, pi_arr)
    return delta_theta, se_theta, float(r_j - r_i)


# -----------------------------------------------------------------------------
# Decision rule (alpha-spent group-sequential approximation, D4 Stage 1)
# -----------------------------------------------------------------------------

def alpha_spent_z(alpha_total: float, n_looks: int) -> float:
    """Per-look one-sided z-score under equal alpha-spending."""
    from scipy.stats import norm
    if not (0.0 < alpha_total < 1.0):
        raise ValueError(f"alpha_total must be in (0, 1), got {alpha_total}")
    if n_looks < 1:
        raise ValueError(f"n_looks must be >= 1, got {n_looks}")
    return float(norm.ppf(1.0 - alpha_total / n_looks))


def decide_theta(
    delta_theta: float,
    se_theta: float,
    *,
    delta_theta_threshold: float,
    z_dethrone: float,
    z_hold: float,
) -> str:
    """Stage-1 alpha-spent group-sequential decision on the latent skill
    contrast Δθ. Threshold is in θ-units (latent skill SDs anchored by the
    Normal(0,1) prior). Stage-1 is an engineering approximation of an
    anytime-valid test under a frozen calibration snapshot; the SE is
    *conditional* on that snapshot (see notes/architecture.md and D5).

    Returns one of: "dethrone", "statistical_hold", "continue".
    `budget_hold_inconclusive` is a separate state emitted by the caller
    when the cell budget is exhausted without a verdict.

    Note on `statistical_hold`: under K=0 with frozen nuisance, β_hold is
    a *nominal futility spending budget*, not a formal false-hold rate.
    Formal false-hold control requires an indifference margin κ_hold —
    deferred to a Stage-2 design pass.
    """
    if not (math.isfinite(delta_theta) and math.isfinite(se_theta) and se_theta >= 0):
        return "continue"
    lower = delta_theta - z_dethrone * se_theta
    upper = delta_theta + z_hold * se_theta
    if lower > delta_theta_threshold:
        return "dethrone"
    if upper <= delta_theta_threshold:
        return "statistical_hold"
    return "continue"


def champion_se_theta(
    cov_theta_i: float = 0.0,
) -> float:
    """Champion preflight: √Var(θ_i). Under K=0 with frozen ψ̂, this is
    `√cov_theta_i` from the active block. Used by the preflight gate to
    decide whether the king's latent skill is sufficiently pinned to
    accept a challenger."""
    return math.sqrt(max(cov_theta_i, 0.0))
