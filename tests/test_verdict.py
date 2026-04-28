from __future__ import annotations

import numpy as np
import pytest

from affine.chain import Miner
from affine.config import Config
from affine.evidence import Row
from affine.irt import Fit, compute_k
from affine.verdict import Dethrone, DuelStatus, Hold, Skip, decide


KING = Miner(uid=0, hotkey="hk0", model="king", revision="kr", block=0)
CHAL = Miner(uid=1, hotkey="hk1", model="chal", revision="cr", block=1)
ART_KEYS = [(KING.model, KING.revision), (CHAL.model, CHAL.revision)]
ROWS = [
    Row(m=0, r="kr", e="E", c=0, p=0, t=0, l=0.0, k="king"),
    Row(m=1, r="cr", e="E", c=0, p=1, t=0, l=0.0, k="chal"),
]
CFG = Config(k_init=2.0, k_final=1.0, k_halflife=10)


def _fit(delta: float = 0.0, se: float = 1.0, *, degenerate: bool = False) -> Fit:
    cov = np.zeros((4, 4))
    cov[0, 0] = cov[1, 1] = 0.5 * se * se
    return Fit(theta=np.array([0.0, delta]), beta=np.zeros(1), alpha=np.zeros(1),
               cov=cov, degenerate=degenerate)


def _decide(rows=ROWS, fit=None, status=DuelStatus.COMPLETED, reign_blocks=0):
    return decide(list(rows), fit or _fit(), status, KING, CHAL, ART_KEYS, reign_blocks, CFG)


def test_empty_rows_skip():
    assert _decide(rows=[]) == Skip("no_rows")


def test_degenerate_fit_skip():
    assert _decide(fit=_fit(degenerate=True)) == Skip("degenerate_fit")


def test_envs_quarantined_skip():
    assert _decide(status=DuelStatus.ENVS_QUARANTINED) == Skip("envs_quarantined")


def test_cancelled_skips():
    assert _decide(status=DuelStatus.CANCELLED) == Skip("cancelled")


@pytest.mark.parametrize("status,reason", [
    (DuelStatus.KING_SLOT_DEAD, "king_slot_dead"),
    (DuelStatus.CHAL_SLOT_DEAD, "chal_slot_dead"),
])
def test_slot_dead_skips_regardless_of_z(status, reason):
    """Slot-dead is a unilateral counter trip, not a duel outcome. A reign
    only changes when the challenger crosses z>k on real evidence; a partial
    fit at the moment of slot teardown must not be allowed to dethrone (or
    crown) anyone. A validator-side env bug that fails both models would
    otherwise race to KING_SLOT_DEAD vs CHAL_SLOT_DEAD with the loser losing
    its reign on infrastructure noise."""
    k = compute_k(0, CFG.k_init, CFG.k_final, CFG.k_halflife)
    assert _decide(fit=_fit(k + 1e-6), status=status) == Skip(reason)
    assert _decide(fit=_fit(-(k + 1e-6)), status=status) == Skip(reason)
    assert _decide(rows=[], status=status) == Skip(reason)


@pytest.mark.parametrize("reign_blocks", [0, 1, 10, 100])
def test_z_threshold_strictly_greater_dethrones(reign_blocks):
    k = compute_k(reign_blocks, CFG.k_init, CFG.k_final, CFG.k_halflife)

    below = _decide(fit=_fit(k - 1e-6), reign_blocks=reign_blocks)
    assert isinstance(below, Hold)
    assert below.reason == "z_below_k"
    assert below.evidence is not None
    assert below.evidence.z < below.evidence.k

    above = _decide(fit=_fit(k + 1e-6), reign_blocks=reign_blocks)
    assert isinstance(above, Dethrone)
    assert above.new_champion == (CHAL.uid, CHAL.revision)
    assert above.evidence.z > above.evidence.k
    assert above.evidence.rows_per_env == {"E": (1, 1, 0, 1)}


def test_rows_per_env_reports_totals_not_just_passes():
    """Unpaired rows: dwell appended one chal row (king failed) and two paired
    rows on a second env. rows_per_env must surface chal_n != king_n on env A so
    audit readers can tell "no king sample" from "king lost"."""
    rows = [
        Row(m=CHAL.uid, r="cr", e="A", c=0, p=1, t=0, l=0.0, k="chal"),
        Row(m=KING.uid, r="kr", e="B", c=0, p=0, t=0, l=0.0, k="king"),
        Row(m=CHAL.uid, r="cr", e="B", c=0, p=1, t=0, l=0.0, k="chal"),
    ]
    k = compute_k(0, CFG.k_init, CFG.k_final, CFG.k_halflife)
    v = _decide(rows=rows, fit=_fit(k + 1e-6))
    assert isinstance(v, Dethrone)
    assert v.evidence.rows_per_env == {
        "A": (1, 1, 0, 0),     # chal 1/1, king never sampled
        "B": (1, 1, 0, 1),     # paired: chal pass, king fail
    }


def test_zero_contrast_se_falls_back_to_zero_z():
    v = _decide(fit=_fit(999.0, se=0.0))
    assert isinstance(v, Hold)
    assert v.reason == "z_below_k"
    assert v.evidence is not None
    assert v.evidence.se == 0.0
    assert v.evidence.z == 0.0
