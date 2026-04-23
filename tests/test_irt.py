from __future__ import annotations

import numpy as np

from affine.irt import Fit, Priors, compute_k, fisher_env, fit_2pl


def _synth(n_m: int, n_e: int, n_per: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 1.0, n_m)
    beta = rng.normal(0, 0.5, n_e)
    a = np.exp(rng.normal(0, 0.3, n_e))
    m_idx, e_idx, y = [], [], []
    for m in range(n_m):
        for e in range(n_e):
            p = 1.0 / (1.0 + np.exp(-a[e] * (theta[m] - beta[e])))
            for _ in range(n_per):
                m_idx.append(m); e_idx.append(e); y.append(float(rng.random() < p))
    return np.array(m_idx), np.array(e_idx), np.array(y), theta, beta, a


def test_fit_recovers_true_parameters():
    m_idx, e_idx, y, theta, beta, a = _synth(20, 4, 200)
    fit = fit_2pl(m_idx, e_idx, y, 20, 4)
    assert np.corrcoef(fit.theta, theta)[0, 1] > 0.95
    assert np.corrcoef(fit.beta, beta)[0, 1] > 0.95
    assert np.corrcoef(fit.a, a)[0, 1] > 0.9


def test_fit_posterior_narrows_with_more_data():
    small = fit_2pl(*_synth(10, 3, 10, 1)[:3], 10, 3)
    large = fit_2pl(*_synth(10, 3, 500, 1)[:3], 10, 3)
    assert large.theta_se.mean() < small.theta_se.mean()


def test_fit_on_empty_data_returns_prior():
    fit = fit_2pl([], [], [], 5, 3, Priors(sigma_theta=1.0))
    assert np.allclose(fit.theta, 0.0)
    assert np.allclose(fit.beta, 0.0)
    assert np.allclose(fit.alpha, 0.0)
    assert np.allclose(fit.theta_se, 1.0)


def test_fit_respects_priors():
    m_idx = np.zeros(1000, dtype=np.intp)
    e_idx = np.zeros(1000, dtype=np.intp)
    y = np.ones(1000)
    tight = fit_2pl(m_idx, e_idx, y, 1, 1, Priors(sigma_theta=0.1))
    loose = fit_2pl(m_idx, e_idx, y, 1, 1, Priors(sigma_theta=10.0))
    assert abs(tight.theta[0]) < abs(loose.theta[0])


def test_contrast_matches_laplace_covariance():
    m_idx, e_idx, y, *_ = _synth(5, 3, 100, 7)
    fit = fit_2pl(m_idx, e_idx, y, 5, 3)
    delta, se = fit.contrast(0, 1)
    want = float(fit.theta[0] - fit.theta[1])
    var = fit.cov[0, 0] + fit.cov[1, 1] - 2 * fit.cov[0, 1]
    assert delta == want
    assert abs(se - np.sqrt(max(var, 0.0))) < 1e-12


def _fit_with(theta, beta, alpha, cov_scale: float = 1e-12) -> Fit:
    """Fit object with specified MAP means and a diagonal covariance. cov_scale=1e-12
    collapses Thompson sampling to greedy (posterior draws ≈ means)."""
    theta = np.asarray(theta, float)
    beta = np.asarray(beta, float)
    alpha = np.asarray(alpha, float)
    n = theta.size + 2 * beta.size
    return Fit(theta=theta, beta=beta, alpha=alpha, cov=np.eye(n) * cov_scale)


def test_fisher_env_prefers_env_where_both_near_threshold():
    # Zero-variance limit: Thompson ≡ greedy. Env 0: both players at threshold;
    # env 1: saturated for both (|θ − β| ≫ 0) → f ≈ 0 → harmonic mean collapses.
    fit = _fit_with(theta=[0.0, 0.0], beta=[0.0, -5.0], alpha=[0.0, 0.0])
    rng = np.random.default_rng(0)
    assert fisher_env(fit, 0, 1, rng) == 0


def test_fisher_env_prefers_high_discrimination_at_threshold():
    fit = _fit_with(theta=[0.0, 0.0], beta=[0.0, 0.0], alpha=[1.0, -1.0])
    rng = np.random.default_rng(0)
    assert fisher_env(fit, 0, 1, rng) == 0


def test_fisher_env_punishes_one_sided_saturation():
    # env 0: θ_a=0 at threshold (f_a=0.25), θ_b=5 saturated (f_b≈0.007); sum=0.257 harm=0.006
    # env 1: both moderate (f≈0.07 each); sum=0.14 harm=0.035
    # Harmonic mean correctly picks env 1 (lower contrast variance).
    fit = _fit_with(theta=[0.0, 5.0], beta=[0.0, 2.5], alpha=[0.0, 0.0])
    rng = np.random.default_rng(0)
    assert fisher_env(fit, 0, 1, rng) == 1


def test_fisher_env_cold_start_spreads_picks_uniformly():
    # Prior-only fit: all envs identical in mean, isotropic Laplace cov.
    # Thompson draws independently → argmax is uniform over envs.
    n_e = 4
    fit = _fit_with(theta=[0.0, 0.0], beta=np.zeros(n_e), alpha=np.zeros(n_e), cov_scale=0.25)
    rng = np.random.default_rng(0)
    picks = [fisher_env(fit, 0, 1, rng) for _ in range(4000)]
    counts = np.bincount(picks, minlength=n_e)
    assert counts.min() > 0.15 * 4000


def test_fisher_env_explores_high_uncertainty_env():
    # Both envs equally informative at MAP. Env 1 has much wider posterior on α
    # (under-sampled in reality); Thompson draws high a_1 ~50% of the time and
    # prefers env 1 at those draws. Exploration frequency should be non-trivial.
    n_m, n_e = 2, 2
    theta = np.zeros(n_m); beta = np.zeros(n_e); alpha = np.zeros(n_e)
    cov = np.eye(n_m + 2 * n_e) * 1e-8
    alpha1_idx = n_m + n_e + 1
    cov[alpha1_idx, alpha1_idx] = 1.0  # wide posterior on α₁
    fit = Fit(theta=theta, beta=beta, alpha=alpha, cov=cov)
    rng = np.random.default_rng(0)
    picks = [fisher_env(fit, 0, 1, rng) for _ in range(2000)]
    # Env 1 chosen meaningfully often — not stuck on env 0 like greedy would be
    # (greedy would split 50/50 here anyway by symmetry, but the test locks in
    # that high-variance envs remain in the running).
    frac_1 = np.mean([p == 1 for p in picks])
    assert 0.25 < frac_1 < 0.75


def test_fisher_env_collapses_to_greedy_as_posterior_tightens():
    # Informative MAP, negligible posterior variance → picks are stable across seeds.
    fit = _fit_with(theta=[0.4, -0.1], beta=[0.0, 0.2, -0.3, 0.1],
                    alpha=[0.0, np.log(2.0), np.log(0.5), np.log(1.5)])
    picks = {fisher_env(fit, 0, 1, np.random.default_rng(s)) for s in range(200)}
    assert len(picks) == 1


def test_fit_sample_reproduces_posterior_covariance():
    # Empirical cov of many draws should match Fit.cov.
    n_m, n_e = 3, 2
    rng = np.random.default_rng(0)
    mean = rng.normal(size=n_m + 2 * n_e)
    A = rng.normal(size=(n_m + 2 * n_e, n_m + 2 * n_e))
    cov = A @ A.T + 0.1 * np.eye(n_m + 2 * n_e)
    fit = Fit(theta=mean[:n_m], beta=mean[n_m:n_m + n_e], alpha=mean[n_m + n_e:], cov=cov)
    draws = np.stack([np.concatenate(fit.sample(rng)) for _ in range(20000)])
    assert np.allclose(draws.mean(0), mean, atol=0.05)
    assert np.allclose(np.cov(draws, rowvar=False), cov, atol=0.15)


def test_fit_sample_handles_singular_covariance():
    # Rank-deficient cov (e.g. exact collinearity) — eigh fallback should not crash,
    # and draws should span only the positive-eigenvalue subspace.
    n_m, n_e = 2, 1
    mean = np.zeros(n_m + 2 * n_e)
    cov = np.zeros((n_m + 2 * n_e, n_m + 2 * n_e))
    cov[0, 0] = 1.0  # only θ₀ has variance
    fit = Fit(theta=mean[:n_m], beta=mean[n_m:n_m + n_e], alpha=mean[n_m + n_e:], cov=cov)
    rng = np.random.default_rng(0)
    draws = np.stack([np.concatenate(fit.sample(rng)) for _ in range(500)])
    assert draws[:, 0].std() > 0.5
    assert np.allclose(draws[:, 1:], 0.0, atol=1e-10)


def test_compute_k_decays_from_init_to_final():
    assert abs(compute_k(0, 3.0, 1.0, 7200) - 3.0) < 1e-9
    assert compute_k(7200, 3.0, 1.0, 7200) == 1.0 + (3.0 - 1.0) * 0.5
    assert abs(compute_k(10**9, 3.0, 1.0, 7200) - 1.0) < 1e-9


def test_compute_k_monotone():
    prev = float("inf")
    for r in range(0, 30000, 500):
        k = compute_k(r, 3.0, 1.0, 7200)
        assert k <= prev
        prev = k


def test_compute_k_clamps_negative_reign():
    # Chain reconnect/fork can make current_block - reign_start negative.
    # k must not exceed k_init, which means exp(+x) must not be computed.
    assert compute_k(-1, 3.0, 1.0, 7200) == compute_k(0, 3.0, 1.0, 7200)
    assert compute_k(-10**6, 3.0, 1.0, 7200) == 3.0


def test_contrast_warns_on_non_pd_covariance(caplog):
    import logging
    fit = Fit(
        theta=np.array([0.3, -0.2]),
        beta=np.zeros(1),
        alpha=np.zeros(1),
        # cov chosen so var_contrast = c[0,0]+c[1,1]-2c[0,1] < 0 (non-PD)
        cov=np.array([[1.0, 5.0, 0.0], [5.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )
    with caplog.at_level(logging.WARNING, logger="affine.irt"):
        delta, se = fit.contrast(0, 1)
    assert delta == 0.5
    assert se == 0.0
    assert any("non-PD" in r.message for r in caplog.records)




def test_fit_gradient_matches_numerical():
    from affine.irt import _obj_and_grad
    m_idx, e_idx, y, *_ = _synth(3, 2, 20)
    priors = Priors()
    x = np.random.default_rng(0).normal(size=3 + 2 * 2) * 0.1
    _, g = _obj_and_grad(x, m_idx, e_idx, y, 3, 2, priors)
    eps = 1e-5
    for i in range(len(x)):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        num = (_obj_and_grad(xp, m_idx, e_idx, y, 3, 2, priors)[0]
               - _obj_and_grad(xm, m_idx, e_idx, y, 3, 2, priors)[0]) / (2 * eps)
        assert abs(num - g[i]) < 1e-4
