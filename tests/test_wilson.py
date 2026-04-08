from affine.wilson import Verdict, wilson_lower, wilson_upper


def test_wilson_lower_zero_n():
    assert wilson_lower(0, 0) == 0.0


def test_wilson_upper_zero_n():
    assert wilson_upper(0, 0) == 1.0


def test_wilson_lower_all_wins():
    lo = wilson_lower(100, 100)
    assert lo > 0.95


def test_wilson_lower_no_wins():
    lo = wilson_lower(0, 100)
    assert lo < 0.05


def test_wilson_bounds_order():
    lo = wilson_lower(30, 100)
    hi = wilson_upper(30, 100)
    assert lo < 0.3 < hi


def test_wilson_lower_small_sample():
    lo = wilson_lower(3, 3)
    assert lo > 0.3
    assert lo < 1.0


def test_wilson_lower_at_half():
    lo = wilson_lower(50, 100)
    assert lo < 0.5
    hi = wilson_upper(50, 100)
    assert hi > 0.5
