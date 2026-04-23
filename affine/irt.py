"""2PL IRT: MAP fit + Laplace posterior + active-sampling primitives.

Model:   P(pass | m, e) = σ(a_e · (θ_m − β_e))
Priors:  θ ~ N(0, σ_θ²),  β ~ N(0, σ_β²),  log a ~ N(0, σ_α²)

Pure math. No I/O, no chain, no slots.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Priors:
    sigma_theta: float = 1.0
    sigma_beta: float = 1.0
    sigma_alpha: float = 0.5


@dataclass
class Fit:
    theta: np.ndarray
    beta: np.ndarray
    alpha: np.ndarray
    cov: np.ndarray

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
        """Posterior mean and std of θ_i − θ_j from the joint Laplace covariance."""
        delta = float(self.theta[i] - self.theta[j])
        var = float(self.cov[i, i] + self.cov[j, j] - 2.0 * self.cov[i, j])
        if var < 0.0:
            log.warning("non-PD contrast covariance (i=%d j=%d var=%.3e); Hessian likely singular", i, j, var)
            var = 0.0
        return delta, math.sqrt(var)

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One draw from the joint Laplace posterior. Returns (θ, β, α) samples.
        Cholesky when PD; eigh with non-negative clipping when only PSD (singular
        Hessian, fit_2pl's pinv fallback)."""
        mean = np.concatenate([self.theta, self.beta, self.alpha])
        try:
            L = np.linalg.cholesky(self.cov)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(self.cov)
            L = V * np.sqrt(np.clip(w, 0.0, None))
        x = mean + L @ rng.standard_normal(mean.size)
        n_m, n_e = self.n_m, self.n_e
        return x[:n_m], x[n_m:n_m + n_e], x[n_m + n_e:]


def _obj_and_grad(x, m_idx, e_idx, y, n_m, n_e, priors):
    theta, beta, alpha = x[:n_m], x[n_m:n_m + n_e], x[n_m + n_e:]
    a_e = np.exp(alpha[e_idx])
    L = a_e * (theta[m_idx] - beta[e_idx])

    ll = (y * L - np.logaddexp(0.0, L)).sum()
    lp = -0.5 * ((theta / priors.sigma_theta) ** 2).sum() \
         - 0.5 * ((beta / priors.sigma_beta) ** 2).sum() \
         - 0.5 * ((alpha / priors.sigma_alpha) ** 2).sum()

    r = y - 1.0 / (1.0 + np.exp(-L))
    g_theta = np.zeros(n_m); np.add.at(g_theta, m_idx, a_e * r)
    g_beta = np.zeros(n_e);  np.add.at(g_beta,  e_idx, -a_e * r)
    g_alpha = np.zeros(n_e); np.add.at(g_alpha, e_idx, L * r)
    g_theta -= theta / priors.sigma_theta ** 2
    g_beta  -= beta  / priors.sigma_beta ** 2
    g_alpha -= alpha / priors.sigma_alpha ** 2

    return -(ll + lp), -np.concatenate([g_theta, g_beta, g_alpha])


def _hessian(x, m_idx, e_idx, n_m, n_e, priors):
    theta, beta, alpha = x[:n_m], x[n_m:n_m + n_e], x[n_m + n_e:]
    a_e = np.exp(alpha[e_idx])
    L = a_e * (theta[m_idx] - beta[e_idx])
    w = 1.0 / (2.0 + np.exp(L) + np.exp(-L))

    N = n_m + 2 * n_e
    H = np.zeros((N, N))
    b, c = e_idx + n_m, e_idx + n_m + n_e
    w_a2, w_aL, w_L2 = w * a_e ** 2, w * a_e * L, w * L ** 2

    np.add.at(H, (m_idx, m_idx), w_a2)
    np.add.at(H, (b, b), w_a2)
    np.add.at(H, (c, c), w_L2)
    np.add.at(H, (m_idx, b), -w_a2); np.add.at(H, (b, m_idx), -w_a2)
    np.add.at(H, (m_idx, c),  w_aL); np.add.at(H, (c, m_idx),  w_aL)
    np.add.at(H, (b, c), -w_aL);     np.add.at(H, (c, b), -w_aL)

    prior = np.concatenate([
        np.full(n_m, priors.sigma_theta ** -2),
        np.full(n_e, priors.sigma_beta ** -2),
        np.full(n_e, priors.sigma_alpha ** -2),
    ])
    H[np.arange(N), np.arange(N)] += prior
    return H


def fit_2pl(m_idx, e_idx, y, n_m, n_e, priors: Priors = Priors()) -> Fit:
    m_idx = np.asarray(m_idx, dtype=np.intp)
    e_idx = np.asarray(e_idx, dtype=np.intp)
    y = np.asarray(y, dtype=np.float64)
    x = np.zeros(n_m + 2 * n_e)
    if m_idx.size:
        x = minimize(_obj_and_grad, x, args=(m_idx, e_idx, y, n_m, n_e, priors),
                     method="L-BFGS-B", jac=True).x
    H = _hessian(x, m_idx, e_idx, n_m, n_e, priors)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        log.warning("fit_2pl: Hessian singular; falling back to pseudoinverse")
        cov = np.linalg.pinv(H)
    return Fit(theta=x[:n_m].copy(), beta=x[n_m:n_m + n_e].copy(),
               alpha=x[n_m + n_e:].copy(), cov=cov)


def fisher_env(fit: Fit, i: int, j: int, rng: np.random.Generator) -> int:
    """Thompson-sampled env for the contrast θ_i − θ_j.

    Draw (θ, β, α) once from the joint Laplace posterior and pick the env that
    maximizes harmonic-mean Fisher info on that draw. Harmonic mean is the
    contrast-variance objective for one Bernoulli per player: var(θ_i − θ_j)
    = 1/f_i + 1/f_j, so info = f_i·f_j / (f_i + f_j).

    Thompson sampling makes exploration emerge from posterior uncertainty —
    envs with few samples have wide draws and get selected whenever the draw
    beats the incumbent. No ε, no UCB constant, no schedule. Collapses to
    greedy as the posterior tightens.
    """
    theta, beta, alpha = fit.sample(rng)
    a = np.exp(alpha)
    p_i = 1.0 / (1.0 + np.exp(-a * (theta[i] - beta)))
    p_j = 1.0 / (1.0 + np.exp(-a * (theta[j] - beta)))
    f_i = a * a * p_i * (1.0 - p_i)
    f_j = a * a * p_j * (1.0 - p_j)
    return int(np.argmax((f_i * f_j) / (f_i + f_j + 1e-18)))


def compute_k(reign_blocks: int, k_init: float, k_final: float, halflife: int) -> float:
    """Dethronement threshold: decays from k_init → k_final with half-life in blocks.
    Higher k = harder to dethrone. Resets on champion change. Negative reign (chain
    reconnect/fork reporting an older block) clamps to 0 so k never exceeds k_init."""
    reign_blocks = max(reign_blocks, 0)
    return k_final + (k_init - k_final) * math.exp(-math.log(2) * reign_blocks / halflife)


