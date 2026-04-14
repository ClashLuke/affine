import pytest

from affine.scoring import check_duel


pytestmark = pytest.mark.diagnostic


def _z(wins, losses, tasks, max_tasks=200, k=2.0):
    return check_duel(wins, losses, tasks, max_tasks, k)[1]


def test_permutation_invariance_of_verdict_and_z():
    wins = {"a": 33, "b": 9, "c": 11}
    losses = {"a": 8, "b": 82, "c": 5}
    tasks = {"a": 69, "b": 100, "c": 40}

    v1, z1 = check_duel(wins, losses, tasks, 116, 2.0)
    v2, z2 = check_duel(
        {"c": wins["c"], "a": wins["a"], "b": wins["b"]},
        {"c": losses["c"], "a": losses["a"], "b": losses["b"]},
        {"c": tasks["c"], "a": tasks["a"], "b": tasks["b"]},
        116,
        2.0,
    )

    assert v1 is v2
    assert z1 == pytest.approx(z2, abs=1e-12)


@pytest.mark.xfail(reason="known non-monotonic aggregate-z behavior with inverse-variance BT weighting", strict=False)
def test_z_should_not_decrease_when_adding_a_challenger_win():
    wins = {"e0": 33, "e1": 9}
    losses = {"e0": 8, "e1": 82}
    tasks = {"e0": 69, "e1": 100}

    z_before = _z(wins, losses, tasks, max_tasks=116)
    wins_after = dict(wins, e1=wins["e1"] + 1)
    tasks_after = dict(tasks, e1=tasks["e1"] + 1)
    z_after = _z(wins_after, losses, tasks_after, max_tasks=116)

    assert z_after >= z_before


@pytest.mark.xfail(reason="known non-monotonic aggregate-z behavior with inverse-variance BT weighting", strict=False)
def test_z_should_not_increase_when_adding_a_challenger_loss():
    wins = {"e0": 10, "e1": 41, "e2": 32, "e3": 36, "e4": 50}
    losses = {"e0": 42, "e1": 11, "e2": 20, "e3": 5, "e4": 2}
    tasks = {"e0": 52, "e1": 52, "e2": 52, "e3": 51, "e4": 52}

    z_before = _z(wins, losses, tasks, max_tasks=52)
    losses_after = dict(losses, e3=losses["e3"] + 1)
    tasks_after = dict(tasks, e3=tasks["e3"] + 1)
    z_after = _z(wins, losses_after, tasks_after, max_tasks=52)

    assert z_after <= z_before
