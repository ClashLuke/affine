import math
import random

from affine.paired import (
    EnvCS,
    PairCounts,
    _bernoulli_kl,
    decide_dethrone,
    env_lower_cs,
    env_score,
    env_upper_cs,
    log_e_minus,
    log_e_plus,
    pair_log_e,
    select_env,
    stabilized_p,
)


def test_log_e_monotone_in_challenger_wins():
    n = 20
    log_es = [pair_log_e(k, n) for k in range(n + 1)]
    for a, b in zip(log_es, log_es[1:]):
        assert a <= b + 1e-12


# ---------------------------------------------------------------------------
# Per-env stratified design (mean-rule)
# ---------------------------------------------------------------------------

def test_log_e_at_zero_data_is_unity():
    for p0 in (0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95):
        assert log_e_plus(0, 0, p0) == 0.0
        assert log_e_minus(0, 0, p0) == 0.0


def test_log_e_plus_matches_pair_log_e_at_half():
    for k, n in [(0, 7), (3, 7), (7, 7), (10, 100), (50, 100), (60, 100), (100, 200)]:
        a = log_e_plus(k, n, 0.5)
        b = pair_log_e(k, n)
        assert math.isclose(a, b, abs_tol=1e-9), f"mismatch at ({k},{n}): {a} vs {b}"


def test_log_e_plus_decreasing_in_p0():
    for k, n in [(5, 10), (10, 20), (50, 100), (60, 100), (200, 300)]:
        prev = math.inf
        for p0 in (0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95):
            cur = log_e_plus(k, n, p0)
            assert cur <= prev + 1e-9, f"non-monotone at ({k},{n}): p0={p0} got {cur} after {prev}"
            prev = cur


def test_log_e_minus_increasing_in_p0():
    for k, n in [(0, 10), (5, 20), (40, 100), (10, 100), (50, 100)]:
        prev = -math.inf
        for p0 in (0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95):
            cur = log_e_minus(k, n, p0)
            assert cur >= prev - 1e-9, f"non-monotone at ({k},{n}): p0={p0} got {cur} after {prev}"
            prev = cur


def test_log_e_minus_symmetry_with_log_e_plus():
    # E^-(k, n, p) = E^+(n-k, n, 1-p) by reflection p <-> 1-p, k <-> n-k.
    for k, n, p0 in [(3, 10, 0.5), (5, 10, 0.4), (40, 100, 0.55), (60, 100, 0.7)]:
        a = log_e_minus(k, n, p0)
        b = log_e_plus(n - k, n, 1.0 - p0)
        assert math.isclose(a, b, abs_tol=1e-9), f"asymmetry at ({k},{n},{p0}): {a} vs {b}"


def test_env_cs_at_no_data_is_unit_interval():
    assert env_lower_cs(0, 0, 0.025) == 0.0
    assert env_upper_cs(0, 0, 0.025) == 1.0


def test_env_lower_cs_zero_when_k_zero():
    for n in (1, 10, 100, 1000):
        assert env_lower_cs(0, n, 0.025) == 0.0


def test_env_upper_cs_one_when_k_equals_n():
    for n in (1, 10, 100, 1000):
        assert env_upper_cs(n, n, 0.025) == 1.0


def test_env_cs_in_unit_interval():
    for k, n in [(0, 5), (5, 5), (3, 10), (50, 100), (0, 100), (100, 100), (60, 100), (1, 1000), (999, 1000)]:
        L = env_lower_cs(k, n, 0.025)
        U = env_upper_cs(k, n, 0.025)
        assert 0.0 <= L <= 1.0, f"L={L} out of [0,1] at ({k},{n})"
        assert 0.0 <= U <= 1.0, f"U={U} out of [0,1] at ({k},{n})"
        assert L <= U + 1e-9, f"L={L} > U={U} at ({k},{n})"


def test_env_cs_finite_at_extremes():
    for k, n in [(0, 1000), (1000, 1000), (1, 1000), (999, 1000), (1, 10000), (5000, 10000)]:
        L = env_lower_cs(k, n, 0.001)
        U = env_upper_cs(k, n, 0.001)
        assert math.isfinite(L) and math.isfinite(U), f"non-finite at ({k},{n}): L={L} U={U}"


def test_env_cs_threshold_crossing():
    # By definition, env_lower_cs is the supremum of {p_0 : log_e_plus >= -log alpha}.
    # Verify the bisection converges to the threshold.
    for k, n in [(60, 100), (200, 300), (700, 1000)]:
        alpha = 0.005
        threshold = -math.log(alpha)
        L = env_lower_cs(k, n, alpha)
        if 0.0 < L < 1.0:
            assert math.isclose(log_e_plus(k, n, L), threshold, abs_tol=1e-3), (
                f"L={L} at ({k},{n}): log_e_plus={log_e_plus(k, n, L)} threshold={threshold}"
            )


def test_env_cs_contains_p_hat():
    # p_hat = k/n should be in [L, U] for non-degenerate cases.
    for k, n in [(50, 100), (60, 100), (40, 100), (300, 1000), (700, 1000)]:
        L = env_lower_cs(k, n, 0.025)
        U = env_upper_cs(k, n, 0.025)
        p_hat = k / n
        assert L <= p_hat + 1e-9, f"({k},{n}): L={L} > p_hat={p_hat}"
        assert p_hat <= U + 1e-9, f"({k},{n}): p_hat={p_hat} > U={U}"


def test_env_cs_shrinks_with_n():
    widths = []
    for n in (100, 400, 1600, 6400):
        L = env_lower_cs(n // 2, n, 0.025)
        U = env_upper_cs(n // 2, n, 0.025)
        widths.append(U - L)
    for w_a, w_b in zip(widths, widths[1:]):
        assert w_b < w_a, f"CS not shrinking: {widths}"


def _uniform_weights(env_count: int) -> dict[str, float]:
    return {f"env_{i}": 1.0 / env_count for i in range(env_count)}


def test_decide_dethrone_clone_does_not_dethrone():
    # p_e = 0.5 for every env; even with 200 samples per env, mean-rule must hold.
    weights = _uniform_weights(7)
    counts = {f"env_{i}": PairCounts(challenger_only=100, champion_only=100) for i in range(7)}
    decision = decide_dethrone(counts, weights, p_star=0.55,
                               alpha_dethrone=0.025, alpha_futility=0.025)
    assert not decision.dethrone
    assert decision.L_mu < 0.55


def test_decide_dethrone_uniform_strong_advantage_dethrones():
    # p_e = 0.85 for every env with 500 samples each: clear mean > 0.55.
    weights = _uniform_weights(7)
    counts = {f"env_{i}": PairCounts(challenger_only=425, champion_only=75) for i in range(7)}
    decision = decide_dethrone(counts, weights, p_star=0.55,
                               alpha_dethrone=0.025, alpha_futility=0.025)
    assert decision.dethrone
    assert decision.L_mu > 0.55


def test_decide_dethrone_futility_at_zero_advantage():
    # p_e = 0.5 with very tight CSs: U_mu falls below p_star = 0.55 -> futility.
    weights = _uniform_weights(7)
    counts = {f"env_{i}": PairCounts(challenger_only=2500, champion_only=2500) for i in range(7)}
    decision = decide_dethrone(counts, weights, p_star=0.55,
                               alpha_dethrone=0.025, alpha_futility=0.025)
    assert decision.futility
    assert not decision.dethrone


def test_decide_dethrone_rejects_invalid_inputs():
    weights = _uniform_weights(2)
    counts = {f"env_{i}": PairCounts() for i in range(2)}
    # weights don't sum to 1
    bad_weights = {"env_0": 0.4, "env_1": 0.4}
    try:
        decide_dethrone(counts, bad_weights, p_star=0.55,
                        alpha_dethrone=0.025, alpha_futility=0.025)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for weights not summing to 1")
    # mismatched keys
    try:
        decide_dethrone(counts, {"x": 0.5, "y": 0.5}, p_star=0.55,
                        alpha_dethrone=0.025, alpha_futility=0.025)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for key mismatch")
    # invalid p_star
    try:
        decide_dethrone(counts, weights, p_star=1.5,
                        alpha_dethrone=0.025, alpha_futility=0.025)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for p_star out of (0,1)")


def _simulate_decision_rate(
    p_e_list: list[float],
    weights: dict[str, float],
    p_star: float,
    alpha: float,
    m_trials: int,
    max_steps: int,
    seed: int,
    decision: str = "dethrone",
) -> float:
    """Simulate m_trials independent runs. Each step draws one informative
    sample from a round-robin-selected env. Returns rate at which the named
    decision (dethrone or futility) ever fires within max_steps.

    Round-robin allocation makes pi_e match per-env sample fraction
    asymptotically; the per-env CSs are valid regardless.
    """
    rng = random.Random(seed)
    env_names = sorted(weights.keys())
    e_count = len(env_names)
    fired = 0
    check_every = 100
    for _ in range(m_trials):
        counts = {e: PairCounts() for e in env_names}
        decided = False
        for step in range(max_steps):
            env_idx = step % e_count
            env_name = env_names[env_idx]
            p_e = p_e_list[env_idx]
            outcome = 1 if rng.random() < p_e else 0
            c = counts[env_name]
            counts[env_name] = PairCounts(
                challenger_only=c.challenger_only + outcome,
                champion_only=c.champion_only + (1 - outcome),
                both_pass=c.both_pass,
                both_fail=c.both_fail,
            )
            if step % check_every == check_every - 1:
                d = decide_dethrone(counts, weights, p_star,
                                    alpha_dethrone=alpha / 2,
                                    alpha_futility=alpha / 2)
                if (decision == "dethrone" and d.dethrone) or (decision == "futility" and d.futility):
                    decided = True
                    break
        if decided:
            fired += 1
    return fired / m_trials


def test_simulate_clone_false_dethrone_under_alpha():
    # Clone: p_e = 0.5 for all envs. With p_star = 0.55, every env is strictly
    # under H_0_e: p_e <= p_star. Per-Bonferroni-Ville: dethrone-error <= alpha/2.
    weights = _uniform_weights(7)
    rate = _simulate_decision_rate(
        p_e_list=[0.5] * 7, weights=weights,
        p_star=0.55, alpha=0.05,
        m_trials=120, max_steps=1500, seed=0xC10E,
    )
    # Upper 95% binomial CI at 120 trials with realized rate 0.025: ~0.025 + 1.96 * sqrt(0.025 * 0.975 / 120) ~= 0.05.
    assert rate <= 0.025 + 0.030, f"realized clone-dethrone rate {rate} > alpha/2 + tol"


def test_simulate_heterogeneous_null_at_boundary():
    # Mean exactly at p_star = 0.55: p_e in {1,1,1,0,0,0,0.85}; pi uniform.
    # mean = (3 + 0.85) / 7 = 0.55. By Ville, dethrone-error <= alpha/2.
    weights = _uniform_weights(7)
    p_e_list = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.85]
    rate = _simulate_decision_rate(
        p_e_list=p_e_list, weights=weights,
        p_star=0.55, alpha=0.05,
        m_trials=120, max_steps=2000, seed=0xB0DE,
    )
    assert rate <= 0.025 + 0.030, f"realized boundary-dethrone rate {rate} > alpha/2 + tol"


def test_simulate_uniform_better_eventually_dethrones():
    # p_e = 0.85 for every env: strong uniform improvement; dethrone power should be high.
    weights = _uniform_weights(7)
    rate = _simulate_decision_rate(
        p_e_list=[0.85] * 7, weights=weights,
        p_star=0.55, alpha=0.05,
        m_trials=80, max_steps=3000, seed=0xD03E,
    )
    assert rate > 0.8, f"power against p_e=0.85 too low: {rate}"


# ---------------------------------------------------------------------------
# Adaptive sampler
# ---------------------------------------------------------------------------

def test_stabilized_p_matches_beta_posterior_mean():
    assert stabilized_p(0, 0) == 0.5
    assert stabilized_p(0, 10) == 0.5 / 11
    assert stabilized_p(10, 10) == 10.5 / 11
    assert math.isclose(stabilized_p(7, 10), 7.5 / 11)


def test_bernoulli_kl_basic():
    assert _bernoulli_kl(0.5, 0.5) == 0.0
    assert _bernoulli_kl(0.0, 0.5) > 0.0
    assert _bernoulli_kl(1.0, 0.5) > 0.0
    # KL(0.7 || 0.5) = 0.7 * log(1.4) + 0.3 * log(0.6)
    expected = 0.7 * math.log(0.7 / 0.5) + 0.3 * math.log(0.3 / 0.5)
    assert math.isclose(_bernoulli_kl(0.7, 0.5), expected, rel_tol=1e-12)


def test_env_score_zero_when_p_tilde_at_or_below_threshold():
    # Stabilized p̃ <= p_star and the L=U=0 (no uncertainty) => both terms zero.
    s = env_score(
        k=4, n_disc=10, n_total=20, weight=1.0,
        p_star=0.55, L=0.0, U=0.0, score_lambda=1.0,
    )
    assert s == 0.0


def test_env_score_positive_when_dethrone_signal():
    # p̃ > p_star with KL contribution.
    s = env_score(
        k=8, n_disc=10, n_total=20, weight=1.0,
        p_star=0.55, L=0.0, U=0.0, score_lambda=1.0,
    )
    assert s > 0.0


def test_env_score_uncertainty_term_kicks_in():
    # p̃ <= p_star but wide CS: uncertainty term contributes.
    s_with_uncertainty = env_score(
        k=4, n_disc=10, n_total=20, weight=1.0,
        p_star=0.55, L=0.0, U=0.5, score_lambda=0.0,
    )
    s_no_uncertainty = env_score(
        k=4, n_disc=10, n_total=20, weight=1.0,
        p_star=0.55, L=0.25, U=0.25, score_lambda=0.0,
    )
    assert s_with_uncertainty > s_no_uncertainty


def test_select_env_cold_start_picks_least_sampled():
    # Two envs at n_total < n_min: pick the smaller-n_total one.
    per_env_counts = {
        "a": PairCounts(challenger_only=0, champion_only=2, both_pass=3, both_fail=0),  # total=5
        "b": PairCounts(challenger_only=1, champion_only=1, both_pass=0, both_fail=1),  # total=3
    }
    weights = {"a": 0.5, "b": 0.5}
    env_cs = {"a": EnvCS(k=0, n=2, L=0.0, U=1.0), "b": EnvCS(k=1, n=2, L=0.0, U=1.0)}
    chosen = select_env(per_env_counts, weights, env_cs,
                        p_star=0.55, n_min=10, score_lambda=0.5)
    assert chosen == "b"  # smaller total


def test_select_env_steady_state_argmax():
    # Both past cold-start; pick env with highest score.
    per_env_counts = {
        "low_p": PairCounts(challenger_only=10, champion_only=10, both_pass=80, both_fail=0),
        "high_p": PairCounts(challenger_only=15, champion_only=5, both_pass=80, both_fail=0),
    }
    weights = {"low_p": 0.5, "high_p": 0.5}
    env_cs = {
        "low_p": EnvCS(k=10, n=20, L=0.3, U=0.7),
        "high_p": EnvCS(k=15, n=20, L=0.5, U=0.9),
    }
    chosen = select_env(per_env_counts, weights, env_cs,
                        p_star=0.55, n_min=20, score_lambda=1.0)
    assert chosen == "high_p"  # higher p̃, KL term dominates


def test_select_env_cold_start_round_robin_by_total():
    # All below n_min: deterministic, picks lex-smallest at minimum total.
    per_env_counts = {
        "a": PairCounts(both_pass=5),  # total = 5
        "b": PairCounts(both_pass=5),
        "c": PairCounts(both_pass=10),
    }
    weights = {e: 1/3 for e in per_env_counts}
    env_cs = {e: EnvCS(k=0, n=0, L=0.0, U=1.0) for e in per_env_counts}
    chosen = select_env(per_env_counts, weights, env_cs,
                        p_star=0.55, n_min=20, score_lambda=0.5)
    assert chosen == "a"  # tied at 5 with b; lex first


def test_select_env_handles_zero_q_gracefully():
    # All concordant (no informative samples): q_hat = (0+1)/(20+2) > 0; doesn't crash.
    per_env_counts = {
        "a": PairCounts(both_pass=20),  # n_disc = 0, n_total = 20
        "b": PairCounts(both_pass=20),
    }
    weights = {"a": 0.5, "b": 0.5}
    env_cs = {"a": EnvCS(k=0, n=0, L=0.0, U=1.0),
              "b": EnvCS(k=0, n=0, L=0.0, U=1.0)}
    chosen = select_env(per_env_counts, weights, env_cs,
                        p_star=0.55, n_min=10, score_lambda=0.5)
    assert chosen in {"a", "b"}


def test_select_env_ucb_prefers_undersampled_after_coldstart():
    # n_min=1 cold-start passed; one env has 1 sample, the other has 50.
    # UCB exploration bonus drives selection toward the under-sampled env.
    per_env_counts = {
        "fresh": PairCounts(both_pass=1),    # n_total = 1
        "well_sampled": PairCounts(both_pass=50),  # n_total = 50
    }
    weights = {"fresh": 0.5, "well_sampled": 0.5}
    env_cs = {
        "fresh": EnvCS(k=0, n=0, L=0.0, U=1.0),
        "well_sampled": EnvCS(k=0, n=0, L=0.0, U=1.0),
    }
    chosen = select_env(per_env_counts, weights, env_cs,
                        p_star=0.55, n_min=1, score_lambda=0.0)
    assert chosen == "fresh"


def test_env_score_ucb_decays_with_n():
    # Same configuration except n_total: more samples ⇒ smaller UCB term.
    s_few = env_score(k=2, n_disc=5, n_total=5, weight=1.0,
                      p_star=0.55, L=0.0, U=0.0, score_lambda=1.0,
                      cost=1.0, t_total=100)
    s_many = env_score(k=20, n_disc=50, n_total=50, weight=1.0,
                       p_star=0.55, L=0.0, U=0.0, score_lambda=1.0,
                       cost=1.0, t_total=100)
    # With score_lambda=1.0 and p_tilde <= p_star, only UCB term contributes;
    # smaller n_total ⇒ larger bonus.
    assert s_few > s_many


def test_select_env_cost_weighting_prefers_cheap():
    # Two envs identical except cost; the cheaper one wins.
    per_env_counts = {
        "cheap": PairCounts(challenger_only=10, champion_only=10),
        "expensive": PairCounts(challenger_only=10, champion_only=10),
    }
    weights = {"cheap": 0.5, "expensive": 0.5}
    env_cs = {
        "cheap": EnvCS(k=10, n=20, L=0.3, U=0.7),
        "expensive": EnvCS(k=10, n=20, L=0.3, U=0.7),
    }
    chosen = select_env(per_env_counts, weights, env_cs,
                        p_star=0.55, n_min=1, score_lambda=0.0,
                        costs={"cheap": 1.0, "expensive": 10.0})
    assert chosen == "cheap"
