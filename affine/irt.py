"""2PL IRT: joint MAP + Laplace posterior + active-sampling primitives.

Model:   P(pass | m, e) = σ(a_e · (θ_m − β_e))
Prior:   θ_m ~ N(0, 1)             (σ_θ pinned for IRT identification)
         β_e ~ N(0, σ_β²)          σ_β fixed
         log a_e ~ N(0, σ_α²)      σ_α fixed

Fixed-σ rather than hierarchical Half-Cauchy: MAP estimation for hierarchical
hyperpriors hits Neal's funnel (σ → 0 with β → 0 jointly diverges the joint
density). On real evidence the hierarchical fit drove log σ_β to its numerical
floor and α to ~10 (a ≈ 5e3, unphysical) with 37 negative Hessian eigenvalues.
Fixed σ converges identically across random starts to ||grad|| < 1e-3.

Pure math. No I/O, no chain, no slots.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit

log = logging.getLogger(__name__)

# Bound |α| in BOTH the optimizer and the posterior draws. exp(50) ≈ 5e21 is
# already past any meaningful discrimination, and σ_α=0.5 puts |α|>50 at >100σ.
# A non-degenerate fit cannot land here; saturating the bound surfaces as
# success=False → Fit.degenerate. Without a draw cap, wide-posterior samples
# produce 0·∞ NaNs in the harmonic-mean Fisher info.
_ALPHA_BOUND = 50.0


@dataclass(frozen=True)
class Priors:
    sigma_beta: float = 1.0    # std of β prior; ±2 logits at 1.96σ covers env-difficulty range
    sigma_alpha: float = 0.5   # std of log α prior; a ∈ [0.38, 2.66] at 1.96σ — typical IRT discrimination


@dataclass
class Fit:
    theta: np.ndarray
    beta: np.ndarray
    alpha: np.ndarray
    cov: np.ndarray         # (n_m + 2 n_e)² joint Laplace posterior
    degenerate: bool = False  # True iff optimizer failed to reach a MAP (non-PD Hessian)

    @property
    def a(self) -> np.ndarray: return np.exp(self.alpha)
    @property
    def n_m(self) -> int: return self.theta.shape[0]
    @property
    def n_e(self) -> int: return self.beta.shape[0]
    @property
    def theta_cov(self) -> np.ndarray: return self.cov[: self.n_m, : self.n_m]
    @property
    def theta_se(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.theta_cov), 0.0, None))

    def contrast(self, i: int, j: int) -> tuple[float, float]:
        delta = float(self.theta[i] - self.theta[j])
        var = float(self.cov[i, i] + self.cov[j, j] - 2.0 * self.cov[i, j])
        # fit_2pl floors eigvals at 1e-12 → cov is PSD by construction, so var<0
        # only via round-off (~1e-15 for typical cov scale). Warn only outside
        # that band; below the band, clamp silently. -1e-9 is 6 orders above
        # plausible round-off, well below any genuinely-corrupt cov.
        if var < -1e-9:
            log.warning("non-PD contrast covariance (i=%d j=%d var=%.3e)", i, j, var)
        return delta, math.sqrt(max(var, 0.0))

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One draw from the joint Laplace posterior over (θ, β, α). fit_2pl floors
        Hessian eigenvalues, so the cov is PSD modulo roundoff; Cholesky succeeds
        on the jittered matrix in the common case. Caller-constructed Fits (tests,
        offline analysis) can be non-PD — fall back to symmetric eigh sqrt, which
        accepts any symmetric matrix (negative eigenvalues clamped to 0)."""
        mean = np.concatenate([self.theta, self.beta, self.alpha])
        cov = self.cov + 1e-12 * np.eye(mean.size)
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(0.5 * (cov + cov.T))
            L = V * np.sqrt(np.clip(w, 0.0, None))
        x = mean + L @ rng.standard_normal(mean.size)
        n_m, n_e = self.n_m, self.n_e
        return x[:n_m], x[n_m:n_m + n_e], x[n_m + n_e:]


def _unpack(x, n_m, n_e):
    return x[:n_m], x[n_m:n_m + n_e], x[n_m + n_e:n_m + 2 * n_e]


def _data_terms(theta, beta, alpha, m_idx, e_idx, y):
    """Per-sample Bernoulli + linear-predictor terms shared by obj/grad/Hessian."""
    a = np.exp(alpha[e_idx])
    L = a * (theta[m_idx] - beta[e_idx])
    p = expit(L)
    return a, L, p * (1.0 - p), y - p   # a, L, w=p(1-p), r=y-p


def _obj_and_grad(x, m_idx, e_idx, y, n_m, n_e, priors):
    theta, beta, alpha = _unpack(x, n_m, n_e)
    a, L, _, r = _data_terms(theta, beta, alpha, m_idx, e_idx, y)
    inv_sb2 = 1.0 / (priors.sigma_beta ** 2)
    inv_sa2 = 1.0 / (priors.sigma_alpha ** 2)

    # log p(y|θ,β,α) = y·L − softplus(L), via stable log_expit so |L|≫1 is safe.
    ll = (y * L + log_expit(-L)).sum()
    lp = -0.5 * ((theta * theta).sum()
                 + (beta * beta).sum() * inv_sb2
                 + (alpha * alpha).sum() * inv_sa2)

    g_theta = np.zeros(n_m); np.add.at(g_theta, m_idx,  a * r); g_theta -= theta
    g_beta  = np.zeros(n_e); np.add.at(g_beta,  e_idx, -a * r); g_beta  -= beta * inv_sb2
    g_alpha = np.zeros(n_e); np.add.at(g_alpha, e_idx,  L * r); g_alpha -= alpha * inv_sa2

    return -(ll + lp), -np.concatenate([g_theta, g_beta, g_alpha])


def _hessian(x, m_idx, e_idx, y, n_m, n_e, priors):
    """Observed Hessian of the negative log posterior at x.

    NLL = −y·L + softplus(L), L = exp(α)·(θ−β). Each block follows
    H_pq = w·L_p·L_q − r·L_pq, with (w, r) = (p(1−p), y−p). Only α-blocks see
    nonzero L_pq: L_θα = a, L_βα = −a, L_αα = L.
    """
    theta, beta, alpha = _unpack(x, n_m, n_e)
    a, L, w, r = _data_terms(theta, beta, alpha, m_idx, e_idx, y)
    inv_sb2 = 1.0 / (priors.sigma_beta ** 2)
    inv_sa2 = 1.0 / (priors.sigma_alpha ** 2)

    h_tt = w * a * a                         # H_θθ = w·a² (L_θθ = 0)
    h_tb = -h_tt                             # H_θβ = -w·a²
    h_ta = w * a * L - r * a                 # H_θα = w·a·L − r·a
    h_ba = -h_ta                             # H_βα = -w·a·L + r·a
    h_aa = w * L * L - r * L                 # H_αα = w·L² − r·L

    N = n_m + 2 * n_e
    H = np.zeros((N, N))
    bj, cj = e_idx + n_m, e_idx + n_m + n_e
    np.add.at(H, (m_idx, m_idx), h_tt)
    np.add.at(H, (bj, bj),       h_tt)
    np.add.at(H, (cj, cj),       h_aa)
    np.add.at(H, (m_idx, bj),    h_tb); np.add.at(H, (bj, m_idx), h_tb)
    np.add.at(H, (m_idx, cj),    h_ta); np.add.at(H, (cj, m_idx), h_ta)
    np.add.at(H, (bj, cj),       h_ba); np.add.at(H, (cj, bj),    h_ba)

    diag = np.concatenate([np.ones(n_m), np.full(n_e, inv_sb2), np.full(n_e, inv_sa2)])
    H[np.arange(N), np.arange(N)] += diag
    return H


def fit_2pl(m_idx, e_idx, y, n_m, n_e, priors: Priors = Priors(),
            init_x: np.ndarray | None = None) -> Fit:
    """`init_x` warm-starts L-BFGS from a previous MAP. Identical posterior to
    cold-start (verified empirically) at ~5x speed when data only grew by a few
    rows. None → cold-start at the prior mean."""
    m_idx = np.asarray(m_idx, dtype=np.intp)
    e_idx = np.asarray(e_idx, dtype=np.intp)
    y = np.asarray(y, dtype=np.float64)
    N = n_m + 2 * n_e
    x = np.zeros(N) if init_x is None else np.asarray(init_x, dtype=np.float64).copy()
    if x.shape != (N,):
        x = np.zeros(N)
    nonfinite = False
    alpha_saturated = False
    if m_idx.size:
        bounds = [(None, None)] * (n_m + n_e) + [(-_ALPHA_BOUND, _ALPHA_BOUND)] * n_e
        result = minimize(_obj_and_grad, x, args=(m_idx, e_idx, y, n_m, n_e, priors),
                          method="L-BFGS-B", jac=True, bounds=bounds,
                          options={"ftol": 1e-12, "gtol": 1e-7, "maxiter": 10000})
        if np.all(np.isfinite(result.x)) and np.isfinite(result.fun):
            x = result.x
            # An α landing at the bound is an active-set KKT solution, not an
            # unconstrained MAP — the unconstrained Hessian computed below
            # ignores the active bound, so the Laplace cov on α (and any θ/β
            # entries coupled to it) is wrong. σ_α=0.5 puts |α|=50 at >100σ
            # under the prior; saturation indicates the data is forcing a
            # value the model doesn't believe. Flag degenerate so callers
            # refuse the fit. Tolerance 1e-3 = ~1e-5 of the bound, well
            # outside any optimizer-convergence wobble.
            alpha = x[n_m + n_e:]
            if np.any(np.abs(alpha) >= _ALPHA_BOUND - 1e-3):
                log.warning("fit_2pl: α saturated bound (max |α|=%.3f, bound=%g) — fit flagged degenerate",
                            float(np.abs(alpha).max()), _ALPHA_BOUND)
                alpha_saturated = True
        else:
            log.warning("fit_2pl: non-finite optimizer state (%s); using prior", result.message)
            nonfinite = True
    H = _hessian(x, m_idx, e_idx, y, n_m, n_e, priors)
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)
    # Floor at numerical-PSD only. The earlier prior-precision floor was unsafe in
    # the wrong direction: at a true MAP the data Hessian's −r·L_pq α-block term
    # is sign-indefinite, so eigenvalues of (P+D) are NOT bounded below by the
    # smallest prior precision (Weyl applies only when D is PSD). Flooring upward
    # *shrinks* cov, which *shrinks* SE, which *grows* z = Δθ̂/SE — overconfident
    # in the dethrone direction. The right policy is: keep the true small
    # eigenvalues (cov correspondingly large → SE large → z low → conservative).
    # Tolerance: round-off can yield ~−1e-12 eigvals on a true PD matrix. Flag
    # as non-MAP only when the most-negative eigenvalue exceeds −1e-8·max(|w|).
    eig_scale = float(np.abs(w).max()) if w.size else 1.0
    eig_tol = -1e-8 * max(eig_scale, 1.0)
    structurally_negative = bool((w < eig_tol).any())
    degenerate = nonfinite or structurally_negative or alpha_saturated
    if structurally_negative:
        log.warning("fit_2pl: %d negative Hessian eigenvalues (min=%.3e, tol=%.3e) — not at MAP; fit flagged degenerate",
                    int((w < eig_tol).sum()), float(w.min()), eig_tol)
    w = np.maximum(w, 1e-12)
    cov = (V / w) @ V.T
    return Fit(theta=x[:n_m].copy(), beta=x[n_m:n_m + n_e].copy(),
               alpha=x[n_m + n_e:].copy(), cov=cov, degenerate=degenerate)


def fisher_env(fit: Fit, i: int, j: int, rng: np.random.Generator,
               excluded: frozenset[int] = frozenset()) -> int:
    """Thompson-sampled env for the contrast θ_i − θ_j.

    Draw (θ, β, α) once from the joint Laplace posterior and pick the env that
    maximizes harmonic-mean Fisher info on that draw: var(θ_i − θ_j) = 1/f_i + 1/f_j,
    so info = f_i f_j / (f_i + f_j).

    Exploration emerges from posterior uncertainty — envs with few samples have
    wide draws and get selected whenever the draw beats the incumbent. No ε, no
    UCB constant. Collapses to greedy as the posterior tightens.

    `excluded`: env indices quarantined by the caller (e.g., repeated infra
    failures). They are never picked. Raises ValueError if all envs are excluded.
    """
    if len(excluded) >= fit.n_e:
        raise ValueError("all envs excluded")
    theta, beta, alpha = fit.sample(rng)
    alpha = np.clip(alpha, -_ALPHA_BOUND, _ALPHA_BOUND)
    a = np.exp(alpha)
    L_i = a * (theta[i] - beta)
    L_j = a * (theta[j] - beta)
    # log f(L) = 2α + log p + log(1−p) = 2α − softplus(L) − softplus(−L); stable for |L|≫1.
    log_f_i = 2.0 * alpha + log_expit(L_i) + log_expit(-L_i)
    log_f_j = 2.0 * alpha + log_expit(L_j) + log_expit(-L_j)
    log_info = log_f_i + log_f_j - np.logaddexp(log_f_i, log_f_j)
    log_info = np.where(np.isnan(log_info), -np.inf, log_info)
    if excluded:
        log_info[list(excluded)] = -np.inf
    if not np.any(np.isfinite(log_info)):
        choices = [e for e in range(fit.n_e) if e not in excluded]
        return int(rng.choice(choices))
    return int(np.argmax(log_info))


def compute_k(reign_blocks: int, k_init: float, k_final: float, halflife: int) -> float:
    """Dethronement threshold: decays from k_init → k_final with half-life in blocks.
    Higher k = harder to dethrone. Resets on champion change. Negative reign (chain
    reconnect/fork reporting an older block) clamps to 0 so k never exceeds k_init.
    halflife<=0 (would divide by zero) collapses to the final floor."""
    if halflife <= 0:
        return k_final
    reign_blocks = max(reign_blocks, 0)
    return k_final + (k_init - k_final) * math.exp(-math.log(2) * reign_blocks / halflife)
