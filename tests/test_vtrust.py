from math import log, sqrt

from affine.scoring import Verdict, bt_mle, aggregate
from affine.vtrust import (
    ValidatorEvidence,
    fisher_info,
    per_trial_info,
    validator_info,
    update_trust,
    merge_evidence,
    merged_check_duel,
    reward_shares,
)


def _ev(hotkey, wins, losses, tasks=None):
    """Shorthand for building ValidatorEvidence."""
    if tasks is None:
        tasks = {k: wins[k] + losses[k] for k in wins}
    return ValidatorEvidence(hotkey=hotkey, wins=wins, losses=losses, tasks=tasks)


# ---------------------------------------------------------------------------
# Fisher information
# ---------------------------------------------------------------------------

class TestFisherInfo:
    def test_equals_inverse_variance(self):
        for w, l in [(10, 10), (50, 5), (0, 20), (1, 1), (100, 100)]:
            _, var = bt_mle(w, l)
            assert abs(fisher_info(w, l) - 1.0 / var) < 1e-12

    def test_more_data_more_info(self):
        assert fisher_info(100, 100) > fisher_info(10, 10)

    def test_balanced_more_info_per_trial_than_lopsided(self):
        assert per_trial_info(50, 50) > per_trial_info(90, 10)

    def test_per_trial_max_at_balance(self):
        info = per_trial_info(1000, 1000)
        assert abs(info - 0.25) < 0.001

    def test_per_trial_approaches_zero(self):
        info = per_trial_info(1000, 1)
        assert info < 0.01


class TestValidatorInfo:
    def test_single_env(self):
        ev = _ev("v1", {"a": 20, }, {"a": 10})
        assert abs(validator_info(ev) - fisher_info(20, 10)) < 1e-12

    def test_multi_env_additive(self):
        ev = _ev("v1", {"a": 20, "b": 30}, {"a": 10, "b": 15})
        expected = fisher_info(20, 10) + fisher_info(30, 15)
        assert abs(validator_info(ev) - expected) < 1e-12

    def test_zero_env_excluded(self):
        ev = _ev("v1", {"a": 10, "b": 0}, {"a": 5, "b": 0})
        assert abs(validator_info(ev) - fisher_info(10, 5)) < 1e-12


# ---------------------------------------------------------------------------
# Bayesian trust
# ---------------------------------------------------------------------------

class TestUpdateTrust:
    def test_no_observations(self):
        t, a, b = update_trust(10.0, 1.0, 0, 0)
        assert abs(t - 10.0 / 11.0) < 1e-12

    def test_all_verified(self):
        t, _, _ = update_trust(10.0, 1.0, 50, 0)
        assert t > 10.0 / 11.0  # trust increases

    def test_one_failure_hurts(self):
        t_good, _, _ = update_trust(10.0, 1.0, 50, 0)
        t_bad, _, _ = update_trust(10.0, 1.0, 49, 1)
        assert t_bad < t_good

    def test_many_failures_tank_trust(self):
        t, _, _ = update_trust(10.0, 1.0, 0, 50)
        assert t < 0.2

    def test_posterior_parameters(self):
        t, a, b = update_trust(5.0, 2.0, 10, 3)
        assert abs(a - 15.0) < 1e-12
        assert abs(b - 5.0) < 1e-12
        assert abs(t - 15.0 / 20.0) < 1e-12

    def test_sequential_update(self):
        """Two sequential updates equal one batch update."""
        _, a1, b1 = update_trust(10.0, 1.0, 20, 2)
        t_seq, _, _ = update_trust(a1, b1, 30, 1)
        t_batch, _, _ = update_trust(10.0, 1.0, 50, 3)
        assert abs(t_seq - t_batch) < 1e-12


# ---------------------------------------------------------------------------
# Evidence merge
# ---------------------------------------------------------------------------

class TestMergeEvidence:
    def test_single_validator_identity(self):
        """One validator with trust=1 should reproduce bt_mle exactly."""
        ev = _ev("v1", {"a": 20, "b": 30}, {"a": 10, "b": 5})
        d_merged, v_merged = merge_evidence([ev], {"v1": 1.0})

        d_a, v_a = bt_mle(20, 10)
        d_b, v_b = bt_mle(30, 5)
        assert abs(d_merged["a"] - d_a) < 1e-12
        assert abs(d_merged["b"] - d_b) < 1e-12
        assert abs(v_merged["a"] - v_a) < 1e-12
        assert abs(v_merged["b"] - v_b) < 1e-12

    def test_two_validators_tighter(self):
        """Two honest validators should produce lower variance than either alone."""
        ev1 = _ev("v1", {"a": 20}, {"a": 10})
        ev2 = _ev("v2", {"a": 15}, {"a": 8})
        trust = {"v1": 1.0, "v2": 1.0}

        _, v_merged = merge_evidence([ev1, ev2], trust)
        _, v1_alone = bt_mle(20, 10)
        _, v2_alone = bt_mle(15, 8)

        assert v_merged["a"] < v1_alone
        assert v_merged["a"] < v2_alone

    def test_zero_trust_excluded(self):
        """Trust=0 validator contributes nothing."""
        ev1 = _ev("v1", {"a": 20}, {"a": 10})
        ev2 = _ev("v2", {"a": 0}, {"a": 50})  # would bias heavily if included
        trust = {"v1": 1.0, "v2": 0.0}

        d_merged, v_merged = merge_evidence([ev1, ev2], trust)
        d_solo, v_solo = bt_mle(20, 10)

        assert abs(d_merged["a"] - d_solo) < 1e-12
        assert abs(v_merged["a"] - v_solo) < 1e-12

    def test_low_trust_reduces_influence(self):
        """A low-trust validator's estimate is down-weighted."""
        ev1 = _ev("v1", {"a": 10}, {"a": 10})  # delta ≈ 0
        ev2 = _ev("v2", {"a": 30}, {"a": 5})   # delta >> 0
        d_full, _ = merge_evidence([ev1, ev2], {"v1": 1.0, "v2": 1.0})
        d_disc, _ = merge_evidence([ev1, ev2], {"v1": 1.0, "v2": 0.3})
        # With full trust, v2 pulls delta up.  Discounted, less so.
        assert d_disc["a"] < d_full["a"]

    def test_missing_env_in_one_validator(self):
        """Validators can cover different environment subsets."""
        ev1 = _ev("v1", {"a": 20, "b": 10}, {"a": 5, "b": 5})
        ev2 = _ev("v2", {"a": 15}, {"a": 8})  # no env b
        trust = {"v1": 1.0, "v2": 1.0}
        d, v = merge_evidence([ev1, ev2], trust)
        assert "a" in d and "b" in d
        # env b should be purely from v1
        d_b, v_b = bt_mle(10, 5)
        assert abs(d["b"] - d_b) < 1e-12
        assert abs(v["b"] - v_b) < 1e-12

    def test_empty_evidence(self):
        d, v = merge_evidence([], {})
        assert d == {}
        assert v == {}

    def test_all_zero_trust(self):
        ev = _ev("v1", {"a": 10}, {"a": 5})
        d, v = merge_evidence([ev], {"v1": 0.0})
        assert d == {}
        assert v == {}

    def test_inverse_variance_property(self):
        """Merged variance should equal 1/sum(trust_i/var_i)."""
        ev1 = _ev("v1", {"a": 30}, {"a": 20})
        ev2 = _ev("v2", {"a": 10}, {"a": 40})
        trust = {"v1": 0.8, "v2": 0.6}
        _, v_merged = merge_evidence([ev1, ev2], trust)

        _, var1 = bt_mle(30, 20)
        _, var2 = bt_mle(10, 40)
        expected = 1.0 / (0.8 / var1 + 0.6 / var2)
        assert abs(v_merged["a"] - expected) < 1e-12


# ---------------------------------------------------------------------------
# Merged duel verdict
# ---------------------------------------------------------------------------

class TestMergedCheckDuel:
    def test_single_validator_matches_check_duel(self):
        """merged_check_duel with one validator should match check_duel."""
        from affine.scoring import check_duel
        wins = {"a": 40, "b": 30}
        losses = {"a": 5, "b": 8}
        tasks = {"a": 45, "b": 38}
        ev = _ev("v1", wins, losses, tasks)

        v1, z1 = check_duel(wins, losses, tasks, 200, 2.0)
        v2, z2 = merged_check_duel([ev], {"v1": 1.0}, 200, 2.0)

        assert v1 == v2
        assert abs(z1 - z2) < 1e-10

    def test_challenger_wins_from_merged(self):
        ev1 = _ev("v1", {"a": 30}, {"a": 5}, {"a": 35})
        ev2 = _ev("v2", {"a": 25}, {"a": 3}, {"a": 28})
        trust = {"v1": 1.0, "v2": 1.0}
        v, z = merged_check_duel([ev1, ev2], trust, 200, 2.0)
        assert v is Verdict.CHALLENGER_WINS
        assert z > 2.0

    def test_undecided_no_data(self):
        v, z = merged_check_duel([], {}, 200, 2.0)
        assert v is Verdict.UNDECIDED
        assert z == 0.0

    def test_hopelessness_from_merged(self):
        """Challenger losing badly across all validators — hopeless."""
        ev1 = _ev("v1", {"a": 2}, {"a": 40}, {"a": 190})
        ev2 = _ev("v2", {"a": 1}, {"a": 35}, {"a": 180})
        trust = {"v1": 1.0, "v2": 1.0}
        v, z = merged_check_duel([ev1, ev2], trust, 200, 2.0)
        assert v is Verdict.CHAMPION_HOLDS

    def test_fabricator_excluded_changes_verdict(self):
        """Honest validators show challenger winning.  A fabricator claiming
        champion dominance should be neutralised by trust=0."""
        honest = _ev("h1", {"a": 40}, {"a": 5}, {"a": 45})
        fabricator = _ev("fab", {"a": 0}, {"a": 100}, {"a": 100})
        # With fabricator trusted, champion might hold
        v_trusted, _ = merged_check_duel(
            [honest, fabricator], {"h1": 1.0, "fab": 1.0}, 200, 2.0,
        )
        # With fabricator excluded, challenger should win
        v_excluded, _ = merged_check_duel(
            [honest, fabricator], {"h1": 1.0, "fab": 0.0}, 200, 2.0,
        )
        assert v_excluded is Verdict.CHALLENGER_WINS


# ---------------------------------------------------------------------------
# Reward shares
# ---------------------------------------------------------------------------

class TestRewardShares:
    def test_proportional_to_info(self):
        """Validator with more decisive outcomes gets larger share."""
        ev1 = _ev("v1", {"a": 50}, {"a": 50})   # 100 decisive, balanced
        ev2 = _ev("v2", {"a": 10}, {"a": 10})    # 20 decisive, balanced
        trust = {"v1": 1.0, "v2": 1.0}
        r = reward_shares([ev1, ev2], trust)
        assert r["v1"] > r["v2"]

    def test_sums_to_one(self):
        ev1 = _ev("v1", {"a": 20, "b": 15}, {"a": 10, "b": 5})
        ev2 = _ev("v2", {"a": 30}, {"a": 12})
        ev3 = _ev("v3", {"a": 5, "b": 8}, {"a": 5, "b": 3})
        trust = {"v1": 1.0, "v2": 0.9, "v3": 0.7}
        r = reward_shares([ev1, ev2, ev3], trust)
        assert abs(sum(r.values()) - 1.0) < 1e-12

    def test_zero_trust_zero_reward(self):
        ev1 = _ev("v1", {"a": 30}, {"a": 10})
        ev2 = _ev("v2", {"a": 30}, {"a": 10})
        r = reward_shares([ev1, ev2], {"v1": 1.0, "v2": 0.0})
        assert r["v2"] == 0.0
        assert abs(r["v1"] - 1.0) < 1e-12

    def test_low_trust_reduces_reward(self):
        ev1 = _ev("v1", {"a": 30}, {"a": 10})
        ev2 = _ev("v2", {"a": 30}, {"a": 10})
        # Equal data, but v2 has lower trust
        r = reward_shares([ev1, ev2], {"v1": 1.0, "v2": 0.5})
        assert r["v1"] > r["v2"]

    def test_all_zero_trust(self):
        ev = _ev("v1", {"a": 10}, {"a": 5})
        r = reward_shares([ev], {"v1": 0.0})
        assert r["v1"] == 0.0

    def test_balanced_env_rewards_more_per_trial(self):
        """Validator in balanced env (p≈0.5) gets more reward per trial
        than one in lopsided env, because per-trial Fisher info is higher."""
        ev_balanced = _ev("v1", {"a": 25}, {"a": 25})   # 50 decisive, p≈0.5
        ev_lopsided = _ev("v2", {"a": 45}, {"a": 5})    # 50 decisive, p≈0.9
        trust = {"v1": 1.0, "v2": 1.0}
        r = reward_shares([ev_balanced, ev_lopsided], trust)
        # Same number of decisive outcomes, but balanced env is more informative
        assert r["v1"] > r["v2"]

    def test_fabrication_scenario(self):
        """Validator caught fabricating gets trust tanked, reward drops."""
        honest = _ev("h", {"a": 30}, {"a": 10})
        cheater = _ev("c", {"a": 30}, {"a": 10})  # same data, but caught lying
        # Before verification: equal
        r_before = reward_shares([honest, cheater], {"h": 1.0, "c": 1.0})
        assert abs(r_before["h"] - r_before["c"]) < 1e-12
        # After catching 5 fabricated samples out of 10 replayed:
        t_cheater, _, _ = update_trust(10.0, 1.0, 5, 5)
        r_after = reward_shares([honest, cheater], {"h": 1.0, "c": t_cheater})
        assert r_after["h"] > r_after["c"]
        assert r_after["h"] > r_before["h"]  # honest share grows


# ---------------------------------------------------------------------------
# Integration: full flow
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_vtrust_flow(self):
        """End-to-end: three validators, one partially dishonest.
        Merge evidence, compute verdict, distribute rewards."""
        v1 = _ev("val1", {"ded": 35, "abd": 25}, {"ded": 10, "abd": 8},
                 {"ded": 60, "abd": 40})
        v2 = _ev("val2", {"ded": 30, "abd": 20}, {"ded": 12, "abd": 10},
                 {"ded": 55, "abd": 35})
        v3 = _ev("val3", {"ded": 28, "abd": 22}, {"ded": 15, "abd": 6},
                 {"ded": 50, "abd": 30})

        # Trust: v3 had some samples fail replay
        t1, a1, b1 = update_trust(10.0, 1.0, 20, 0)   # clean
        t2, a2, b2 = update_trust(10.0, 1.0, 20, 0)   # clean
        t3, a3, b3 = update_trust(10.0, 1.0, 15, 5)    # 5 failures
        trust = {"val1": t1, "val2": t2, "val3": t3}

        assert t1 > t3  # honest > partially dishonest
        assert t2 > t3

        # Merged verdict
        verdict, z = merged_check_duel([v1, v2, v3], trust, 200, 2.0)
        assert verdict is Verdict.CHALLENGER_WINS
        assert z > 2.0

        # Rewards
        r = reward_shares([v1, v2, v3], trust)
        assert abs(sum(r.values()) - 1.0) < 1e-12
        assert r["val3"] < r["val1"]  # penalised for failed replay
        assert r["val3"] < r["val2"]
