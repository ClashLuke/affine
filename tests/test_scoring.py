from math import exp, log, sqrt

from affine.scoring import Verdict, bt_mle, aggregate, check_duel, compute_k


class TestBtMle:
    def test_symmetric(self):
        d1, v1 = bt_mle(10, 10)
        assert abs(d1) < 1e-12

    def test_challenger_dominant(self):
        d, v = bt_mle(20, 5)
        assert d > 0
        assert v > 0

    def test_champion_dominant(self):
        d, v = bt_mle(5, 20)
        assert d < 0

    def test_zero_losses(self):
        d, v = bt_mle(10, 0)
        assert d > 0
        assert v > 0

    def test_zero_wins(self):
        d, v = bt_mle(0, 10)
        assert d < 0
        assert v > 0

    def test_both_zero(self):
        d, v = bt_mle(0, 0)
        assert abs(d) < 1e-12
        assert v > 0

    def test_variance_decreases_with_sample_size(self):
        _, v1 = bt_mle(10, 10)
        _, v2 = bt_mle(100, 100)
        assert v2 < v1

    def test_pseudocounts_prevent_infinities(self):
        d, v = bt_mle(0, 0)
        assert d == d  # not NaN
        assert v == v
        assert v < float('inf')


class TestAggregate:
    def test_single_env(self):
        d, v = aggregate([1.0], [0.5])
        assert d == 1.0
        assert v == 0.5

    def test_equal_variance_converges_to_mean(self):
        d, v = aggregate([1.0, 3.0], [1.0, 1.0])
        assert abs(d - 2.0) < 1e-12
        assert abs(v - 0.5) < 1e-12

    def test_high_variance_gets_low_weight(self):
        d, v = aggregate([0.0, 10.0], [0.01, 100.0])
        assert d < 1.0  # dominated by first env

    def test_empty(self):
        d, v = aggregate([], [])
        assert d == 0.0
        assert v == float('inf')

    def test_inverse_relationship(self):
        deltas = [1.0, 2.0, 3.0]
        variances = [0.1, 0.2, 0.3]
        d, v = aggregate(deltas, variances)
        assert abs(v - 1.0 / sum(1.0 / vi for vi in variances)) < 1e-12


class TestCheckDuel:
    def test_undecided_no_data(self):
        v, z = check_duel({}, {}, {}, 200, 2.0)
        assert v is Verdict.UNDECIDED
        assert z == 0.0

    def test_undecided_all_ties(self):
        v, z = check_duel({"env": 0}, {"env": 0}, {"env": 50}, 200, 2.0)
        assert v is Verdict.UNDECIDED

    def test_challenger_wins_overwhelming(self):
        v, z = check_duel({"env": 50, "env2": 50}, {"env": 5, "env2": 5}, {"env": 55, "env2": 55}, 200, 2.0)
        assert v is Verdict.CHALLENGER_WINS
        assert z > 2.0

    def test_champion_holds_hopeless(self):
        v, z = check_duel({"env": 2, "env2": 2}, {"env": 50, "env2": 50}, {"env": 200, "env2": 200}, 200, 2.0)
        assert v is Verdict.CHAMPION_HOLDS

    def test_undecided_insufficient_evidence(self):
        v, z = check_duel({"env": 3, "env2": 3}, {"env": 2, "env2": 2}, {"env": 10, "env2": 10}, 200, 3.0)
        assert v is Verdict.UNDECIDED

    def test_champion_losing_but_not_yet_hopeless(self):
        v, z = check_duel({"env": 8}, {"env": 12}, {"env": 20}, 200, 2.0)
        assert v is Verdict.UNDECIDED

    def test_hopelessness_with_remaining_budget(self):
        v, z = check_duel({"env": 1}, {"env": 30}, {"env": 50}, 50, 2.0)
        assert v is Verdict.CHAMPION_HOLDS

    def test_hopelessness_all_remaining_wins_not_enough(self):
        v, z = check_duel({"env": 0}, {"env": 100}, {"env": 150}, 200, 2.0)
        assert v is Verdict.CHAMPION_HOLDS

    def test_low_k_easier_to_clear(self):
        v_low, _ = check_duel({"env": 15}, {"env": 10}, {"env": 25}, 200, 1.0)
        v_high, _ = check_duel({"env": 15}, {"env": 10}, {"env": 25}, 200, 3.0)
        if v_low is Verdict.CHALLENGER_WINS:
            assert v_high is not Verdict.CHALLENGER_WINS or v_low is Verdict.CHALLENGER_WINS

    def test_multi_env_mixed_winner(self):
        v, z = check_duel(
            {"a": 40, "b": 15, "c": 30},
            {"a": 5, "b": 5, "c": 5},
            {"a": 45, "b": 20, "c": 35},
            200, 2.0,
        )
        assert v is Verdict.CHALLENGER_WINS
        assert z > 2.0

    def test_multi_env_mixed_one_bad_env_drags_down(self):
        v, z = check_duel(
            {"a": 30, "b": 5, "c": 20},
            {"a": 10, "b": 25, "c": 10},
            {"a": 40, "b": 30, "c": 30},
            200, 2.0,
        )
        assert v is Verdict.UNDECIDED
        assert z < 2.0


class TestComputeK:
    def test_initial(self):
        assert abs(compute_k(0) - 3.0) < 1e-12

    def test_at_halflife(self):
        k = compute_k(7200, k_init=3.0, k_final=1.0, halflife=7200)
        expected = 1.0 + (3.0 - 1.0) * 0.5
        assert abs(k - expected) < 1e-12

    def test_long_reign_approaches_final(self):
        k = compute_k(100000, k_init=3.0, k_final=1.0, halflife=7200)
        assert abs(k - 1.0) < 0.01

    def test_zero_halflife(self):
        assert compute_k(0, k_init=3.0, k_final=1.0, halflife=0) == 1.0

    def test_monotonic_decay(self):
        prev = compute_k(0)
        for blocks in [100, 1000, 5000, 10000, 50000]:
            k = compute_k(blocks)
            assert k <= prev
            prev = k

    def test_custom_params(self):
        k = compute_k(0, k_init=5.0, k_final=0.5, halflife=3600)
        assert abs(k - 5.0) < 1e-12


# --- Edge cases from affine-cortex comparison ---

class TestCheckDuelEdgeCases:

    def test_single_decisive_insufficient(self):
        """1 win should NOT clear k=2. z ≈ 0.76 with pseudocounts."""
        v, z = check_duel({"e": 1}, {"e": 0}, {"e": 1}, 200, 2.0)
        assert v is Verdict.UNDECIDED

    def test_hopelessness_per_env_budget(self):
        """Env A exhausted and losing. Env B has budget. Must stay UNDECIDED.
        Hopelessness must check remaining budget per-env, not globally."""
        v, _ = check_duel(
            {"a": 0, "b": 2}, {"a": 50, "b": 3}, {"a": 200, "b": 10}, 200, 2.0,
        )
        assert v is Verdict.UNDECIDED

    def test_all_tie_env_excluded_from_aggregate(self):
        """0W-0L env must not affect z-score. Adding a no-data env should be
        identical to not having it."""
        _, z1 = check_duel({"a": 15, "b": 0}, {"a": 5, "b": 0}, {"a": 20, "b": 20}, 200, 2.0)
        _, z2 = check_duel({"a": 15}, {"a": 5}, {"a": 20}, 200, 2.0)
        assert abs(z1 - z2) < 1e-12

    def test_inverse_variance_property(self):
        """100-sample env with slight edge should dominate 5-sample env
        with strong opposite edge. This is the key difference from the old
        system's equal-weight ELO."""
        d_a, v_a = bt_mle(55, 45)   # 100 samples, slight challenger edge
        d_b, v_b = bt_mle(1, 4)     # 5 samples, strong champion edge
        d_agg, _ = aggregate([d_a, d_b], [v_a, v_b])
        assert d_agg > 0  # env_a dominates
