from affine.decide import BettingCS, DuelOutcome, decide


def test_betting_ci_matches_bruteforce_grid():
    cs = BettingCS(alpha=0.05)
    for y in (-1.0, 0.0, 1.0, 1.0, 0.0, 1.0, -1.0, 1.0):
        cs.update(y)

    lo, hi = cs.ci()
    grid = [x / 1000 for x in range(-1000, 1001)]
    log_target = __import__("math").log(1.0 / cs.alpha)
    kept = [
        m for m in grid
        if cs._log_capital(m, -1) < log_target and cs._log_capital(m, +1) < log_target
    ]
    assert abs(lo - min(kept)) <= 1e-3
    assert abs(hi - max(kept)) <= 1e-3


def test_decide_three_way_statuses():
    cs = BettingCS(alpha=0.1)
    for _ in range(20):
        cs.update(1.0)
    verdict = decide(cs, delta_dethrone=0.0, delta_hold=0.0)
    assert verdict.outcome is DuelOutcome.DETHRONE
    assert verdict.reason == "dethrone"
    assert verdict.ci_low > 0.0
    assert verdict.log_capital_at_zero > 0.0

    cs = BettingCS(alpha=0.1)
    for _ in range(20):
        cs.update(-1.0)
    verdict = decide(cs, delta_dethrone=0.0, delta_hold=0.0)
    assert verdict.outcome is DuelOutcome.NO_DETHRONE
    assert verdict.reason == "hold"
    assert verdict.ci_hi <= 0.0


def test_capital_short_circuit_preserves_rejection():
    cs = BettingCS(alpha=0.05)
    for _ in range(50):
        cs.update(1.0)
    capped = cs._log_capital(0.0, +1)

    original_alpha = cs.alpha
    cs.alpha = 1e-100
    uncapped = cs._log_capital(0.0, +1)
    cs.alpha = original_alpha

    assert capped >= __import__("math").log(1.0 / original_alpha)
    assert uncapped >= capped
