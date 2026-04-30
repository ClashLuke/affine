import math

from scipy.stats import binomtest

from affine.paired import PairCounts, alpha_for_reign, decide_paired, pair_p_value


def test_p_value_matches_scipy():
    for n in range(1, 16):
        for k in range(n + 1):
            ours = pair_p_value(k, n)
            theirs = 1.0 if k == 0 else float(binomtest(k, n, p=0.5, alternative="greater").pvalue)
            assert math.isclose(ours, theirs, rel_tol=1e-12)


def test_decide_gates():
    assert decide_paired(PairCounts(8, 1), alpha=0.5, min_discordant=16).reason == "too_few_discordant"
    assert decide_paired(PairCounts(8, 8), alpha=0.5, min_discordant=1).reason == "challenger_not_ahead"
    assert decide_paired(PairCounts(6, 4), alpha=0.01, min_discordant=1).reason == "p_above_alpha"
    assert decide_paired(PairCounts(20, 0), alpha=0.05, min_discordant=16).dethrone


def test_alpha_ratchet_monotone():
    a0 = alpha_for_reign(0, 0.005, 0.05, 7200)
    a1 = alpha_for_reign(7200, 0.005, 0.05, 7200)
    a2 = alpha_for_reign(72000, 0.005, 0.05, 7200)
    assert a0 < a1 < a2 < 0.05
