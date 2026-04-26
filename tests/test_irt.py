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
    fit = fit_2pl([], [], [], 5, 3, Priors(sigma_beta=1.0, sigma_alpha=1.0))
    assert np.allclose(fit.theta, 0.0)
    assert np.allclose(fit.beta, 0.0)
    assert np.allclose(fit.alpha, 0.0)
    # σ_θ is fixed at 1 for IRT identification, so θ_se = 1 with no data.
    assert np.allclose(fit.theta_se, 1.0)
    assert fit.degenerate is False  # prior-only fit is the unique MAP, Hessian PSD


def test_fit_marks_degenerate_when_grad_far_from_zero(monkeypatch):
    """Real non-convergence: x stuck far from MAP, ||grad||_∞ large. The verdict
    path must skip this fit — the Laplace cov has no posterior interpretation
    when x isn't a stationary point. Force success=False AND a high gradient
    (mock returns the initial x0, where the data-driven gradient is nonzero)."""
    from scipy.optimize import OptimizeResult
    import affine.irt as irt
    real_min = irt.minimize
    def fake(fn, x0, *args, **kwargs):
        f0, g0 = fn(x0, *kwargs.get("args", ()))
        return OptimizeResult(x=x0, fun=f0, success=False, message="forced",
                              jac=g0, nit=0, status=1)
    monkeypatch.setattr(irt, "minimize", fake)
    m_idx, e_idx, y, *_ = _synth(5, 2, 20)
    fit = fit_2pl(m_idx, e_idx, y, 5, 2)
    assert fit.degenerate is True


def test_fit_accepts_success_false_when_grad_small(monkeypatch):
    """Regression for over-broad nonconverged: L-BFGS-B reports success=False on
    ABNORMAL_TERMINATION_IN_LNSRCH at a true MAP (line search hits machine-eps
    progress with gradient already small). Flagging that as degenerate forces
    _elect into baseline-fallback unnecessarily. The principled test is the
    gradient norm, not the success flag."""
    from scipy.optimize import OptimizeResult
    import affine.irt as irt
    real_min = irt.minimize
    def fake(fn, x0, *args, **kwargs):
        res = real_min(fn, x0, *args, **kwargs)
        return OptimizeResult(x=res.x, fun=res.fun, success=False,
                              message="ABNORMAL_TERMINATION_IN_LNSRCH",
                              jac=res.jac, nit=res.nit, status=2)
    monkeypatch.setattr(irt, "minimize", fake)
    m_idx, e_idx, y, *_ = _synth(5, 2, 20)
    fit = fit_2pl(m_idx, e_idx, y, 5, 2)
    assert fit.degenerate is False


def test_fit_respects_priors():
    # Tight prior on β shrinks β toward 0; loose prior lets it follow the likelihood.
    m_idx = np.zeros(1000, dtype=np.intp)
    e_idx = np.zeros(1000, dtype=np.intp)
    y = np.ones(1000)
    tight = fit_2pl(m_idx, e_idx, y, 1, 1, Priors(sigma_beta=0.01))
    loose = fit_2pl(m_idx, e_idx, y, 1, 1, Priors(sigma_beta=10.0))
    assert abs(tight.beta[0]) < abs(loose.beta[0])


def test_hessian_floor_independent_of_block_dynamic_range():
    """Regression: an absolute floor on Hessian eigenvalues (vs a relative
    `eps*w.max()`) preserves block scale when one block has a huge prior precision.
    With sigma_beta=1e-9 the β-prior diagonal is 1e18; a relative floor would clip
    θ-block info (= 1) and collapse contrast SE 1.414 → 0.09 — overconfident."""
    fit = fit_2pl([], [], [], 2, 1, Priors(sigma_beta=1e-9, sigma_alpha=0.5))
    _, se = fit.contrast(0, 1)
    assert abs(se - np.sqrt(2.0)) < 1e-6




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
    collapses Thompson sampling to greedy."""
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
    # Empirical cov of many draws should match Fit.cov (which is over (θ, β, α)).
    n_m, n_e = 3, 2
    D = n_m + 2 * n_e
    rng = np.random.default_rng(0)
    mean = rng.normal(size=D)
    A = rng.normal(size=(D, D))
    cov = A @ A.T + 0.1 * np.eye(D)
    fit = Fit(theta=mean[:n_m], beta=mean[n_m:n_m + n_e],
              alpha=mean[n_m + n_e:], cov=cov)
    draws = np.stack([np.concatenate(fit.sample(rng)) for _ in range(40000)])
    assert np.allclose(draws.mean(0), mean, atol=0.05)
    assert np.allclose(np.cov(draws, rowvar=False), cov, atol=0.2)


def test_fit_sample_handles_singular_covariance():
    # Rank-deficient cov (e.g. exact collinearity) — sample adds 1e-12 jitter
    # so Cholesky succeeds. Draws on zero-variance directions stay bounded by
    # sqrt(jitter) = 1e-6, while the variance direction shows full spread.
    n_m, n_e = 2, 1
    D = n_m + 2 * n_e
    mean = np.zeros(D)
    cov = np.zeros((D, D))
    cov[0, 0] = 1.0
    fit = Fit(theta=mean[:n_m], beta=mean[n_m:n_m + n_e],
              alpha=mean[n_m + n_e:], cov=cov)
    rng = np.random.default_rng(0)
    draws = np.stack([np.concatenate(fit.sample(rng)) for _ in range(500)])
    assert draws[:, 0].std() > 0.5
    assert np.all(np.abs(draws[:, 1:]) < 1e-4)


def test_fit_sample_eigh_fallback_on_non_pd_covariance():
    """Caller-constructed Fit with a slightly non-PD cov (e.g. from offline
    analysis where the user assembled the matrix by hand): Cholesky raises;
    eigh fallback must produce finite draws with second-moments matching
    the PSD-projected covariance."""
    n_m, n_e = 2, 1
    cov = np.array([
        [1.0, 0.5, 0.0, 0.0],
        [0.5, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, -1e-3],   # negative eigenvalue → non-PD
    ])
    fit = Fit(theta=np.zeros(n_m), beta=np.zeros(n_e),
              alpha=np.zeros(n_e), cov=cov)
    rng = np.random.default_rng(0)
    draws = np.stack([np.concatenate(fit.sample(rng)) for _ in range(2000)])
    assert np.all(np.isfinite(draws))
    # Negative-eigenvalue direction is clamped to zero variance in the fallback.
    assert draws[:, 3].std() < 1e-3


def test_fit_2pl_floor_caps_posterior_draw_magnitude():
    """Regression: the 1e-12 floor produced 1e12 cov entries when an env's
    Hessian eigenvalue hit the floor → posterior draws of magnitude 1e6 →
    fisher_env picked random envs. New floor at the smallest prior precision
    keeps draws bounded for Thompson sampling."""
    # Force a low-info regime: 1 miner, 1 env, no observations. Pure prior fit.
    fit = fit_2pl([], [], [], 1, 1, Priors(sigma_beta=1.0, sigma_alpha=0.5))
    rng = np.random.default_rng(0)
    draws = np.stack([np.concatenate(fit.sample(rng)) for _ in range(2000)])
    # With unit prior on θ/β and σ_α=0.5 prior on log α: stddev ≤ 1.0 in θ/β,
    # ≤ 0.5 in log α. No direction should exceed 5σ across 2000 draws.
    assert np.all(np.abs(draws).max(axis=0) < 5.0)


def test_fit_alpha_bound_matches_draw_cap():
    """Bound and draw cap must match: a healthy MAP cannot have |α|>50 under
    σ_α=0.5 (>100σ), so saturating the optimizer bound IS a degenerate signal.
    Mismatched values would let a non-degenerate fit produce α=80 (extreme but
    in-bounds) and then have draws clipped to 50 — Fisher info computed under
    an α the model doesn't believe."""
    from affine.irt import _ALPHA_BOUND
    assert _ALPHA_BOUND == 50.0


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


def test_compute_k_handles_zero_halflife():
    # config validates halflife>0, but bypassing config (tests, offline analysis)
    # used to ZeroDivision. Collapse to k_final — the no-decay equivalent.
    assert compute_k(0, 3.0, 1.0, 0) == 1.0
    assert compute_k(1000, 3.0, 1.0, 0) == 1.0
    assert compute_k(-5, 3.0, 1.0, -1) == 1.0


def test_contrast_warns_on_non_pd_covariance(caplog):
    import logging
    D = 2 + 2 * 1
    cov = np.eye(D)
    cov[0, 1] = cov[1, 0] = 5.0
    fit = Fit(
        theta=np.array([0.3, -0.2]),
        beta=np.zeros(1),
        alpha=np.zeros(1),
        cov=cov,
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


def test_hessian_matches_numerical_gradient_jacobian():
    """The Laplace covariance is `H⁻¹` where H is the OBSERVED Hessian of the
    negative log-posterior at the MAP. Earlier versions returned the Gauss-Newton
    approximation (dropping `−r·∇²L` for the α blocks); since L = exp(α)·(θ−β)
    is nonlinear in α, the difference is real and corrupts contrast SE and
    Thompson draws of α. Verify analytical H matches central differences of g."""
    from affine.irt import _obj_and_grad, _hessian
    m_idx, e_idx, y, *_ = _synth(4, 3, 30, seed=11)
    priors = Priors()
    x = np.random.default_rng(2).normal(size=4 + 2 * 3) * 0.4
    H = _hessian(x, m_idx, e_idx, y, 4, 3, priors)
    eps = 1e-5
    H_num = np.zeros_like(H)
    for k in range(x.size):
        xp = x.copy(); xp[k] += eps
        xm = x.copy(); xm[k] -= eps
        _, gp = _obj_and_grad(xp, m_idx, e_idx, y, 4, 3, priors)
        _, gm = _obj_and_grad(xm, m_idx, e_idx, y, 4, 3, priors)
        H_num[:, k] = (gp - gm) / (2 * eps)
    H_num = 0.5 * (H_num + H_num.T)
    assert np.max(np.abs(H - H_num)) < 1e-4


def test_fisher_env_robust_to_extreme_alpha_draw():
    """Wide posterior on log_a can draw α in the hundreds. The earlier fisher_env
    computed a²·p(1−p) directly, producing 0·∞ NaN when exp(α) overflowed. The
    log-domain version must remain finite and pick a real env index."""
    n_m, n_e = 2, 3
    D = n_m + 2 * n_e
    cov = np.eye(D) * 1e-8
    cov[n_m + n_e, n_m + n_e] = 25.0          # σ²(α₀) = 25 → draws ±15
    fit = Fit(theta=np.zeros(n_m), beta=np.zeros(n_e), alpha=np.zeros(n_e), cov=cov)
    rng = np.random.default_rng(0)
    picks = [fisher_env(fit, 0, 1, rng) for _ in range(500)]
    assert all(0 <= p < n_e for p in picks)
