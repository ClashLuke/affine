import math

import pytest
from scipy.stats import binomtest

from affine.paired import PairCounts, alpha_for_reign, decide_paired, pair_p_value


def test_pair_p_value_agrees_with_scipy():
    for trial in range(1, 20):
        for win in range(0, trial + 1):
            expected = float(binomtest(win, trial, p=0.5, alternative="greater").pvalue)
            assert math.isclose(pair_p_value(win, trial), expected, rel_tol=1e-12), \
                f"win={win} trial={trial}"


def test_pair_p_value_zero_discordant_is_one():
    assert pair_p_value(0, 0) == 1.0


def test_pair_p_value_zero_wins_is_one():
    assert pair_p_value(0, 10) == 1.0


def test_pair_p_value_all_wins_is_tiny():
    p = pair_p_value(10, 10)
    assert p == 0.5 ** 10


def test_decide_paired_rejects_challenger_behind():
    counts = PairCounts(challenger_only=5, champion_only=8)
    decision = decide_paired(counts, alpha=0.5, min_discordant=1)
    assert not decision.dethrone
    assert decision.reason == "challenger_not_ahead"


def test_decide_paired_rejects_tied():
    counts = PairCounts(challenger_only=5, champion_only=5)
    decision = decide_paired(counts, alpha=0.5, min_discordant=1)
    assert not decision.dethrone
    assert decision.reason == "challenger_not_ahead"


def test_decide_paired_p_above_alpha():
    counts = PairCounts(challenger_only=6, champion_only=4)
    decision = decide_paired(counts, alpha=0.03, min_discordant=1)
    assert not decision.dethrone
    assert decision.reason == "p_above_alpha"


def test_decide_paired_too_few_discordant():
    counts = PairCounts(challenger_only=10, champion_only=0)
    decision = decide_paired(counts, alpha=0.05, min_discordant=16)
    assert not decision.dethrone
    assert decision.reason == "too_few_discordant"


def test_decide_paired_min_discordant_exactly_met():
    counts = PairCounts(challenger_only=16, champion_only=0)
    decision = decide_paired(counts, alpha=0.05, min_discordant=16)
    assert decision.dethrone
    assert decision.reason == "exact_paired_test"


def test_decide_paired_exact_paired_test_triggers():
    counts = PairCounts(challenger_only=20, champion_only=5, both_pass=100, both_fail=100)
    decision = decide_paired(counts, alpha=0.01, min_discordant=16)
    assert decision.dethrone
    assert decision.reason == "exact_paired_test"
    assert decision.p_value == pair_p_value(20, 25)


def test_decide_paired_ties_do_not_affect_p_value():
    a = PairCounts(challenger_only=20, champion_only=5)
    b = PairCounts(challenger_only=20, champion_only=5, both_pass=300, both_fail=300)
    assert decide_paired(a, alpha=0.01, min_discordant=16).p_value == \
           decide_paired(b, alpha=0.01, min_discordant=16).p_value


def test_decide_paired_alpha_boundary_exactly_equal():
    counts = PairCounts(challenger_only=6, champion_only=4)
    p = pair_p_value(6, 10)
    decision = decide_paired(counts, alpha=p, min_discordant=1)
    assert decision.dethrone


def test_decide_paired_alpha_boundary_barely_above():
    counts = PairCounts(challenger_only=6, champion_only=4)
    p = pair_p_value(6, 10)
    decision = decide_paired(counts, alpha=p * 0.999, min_discordant=1)
    assert not decision.dethrone


def test_alpha_ratchets_from_start_to_final():
    assert alpha_for_reign(0, 0.005, 0.05, 100) == 0.005
    assert 0.005 < alpha_for_reign(100, 0.005, 0.05, 100) < 0.05


def test_alpha_for_reign_asymptotic():
    huge = alpha_for_reign(10**6, 0.005, 0.05, 100)
    assert math.isclose(huge, 0.05, rel_tol=1e-6)


def test_alpha_for_reign_exponential_shape():
    a1 = alpha_for_reign(50, 0.005, 0.05, 50)
    a2 = alpha_for_reign(100, 0.005, 0.05, 50)
    assert a1 < a2  # monotonic rise
    assert math.isclose(a1, 0.0275)


def test_add_champion_only():
    c = PairCounts().add(1, 0)
    assert c.champion_only == 1
    assert c.challenger_only == 0
    assert c.both_pass == 0
    assert c.both_fail == 0


def test_add_challenger_only():
    c = PairCounts().add(0, 1)
    assert c.champion_only == 0
    assert c.challenger_only == 1


def test_add_both_pass():
    c = PairCounts().add(1, 1)
    assert c.both_pass == 1


def test_add_both_fail():
    c = PairCounts().add(0, 0)
    assert c.both_fail == 1


def test_total_and_discordant():
    c = PairCounts(3, 1, 10, 6)
    assert c.total == 20
    assert c.discordant == 4
