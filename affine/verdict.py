"""Decision layer: pure mapping from duel evidence to a Verdict.

`decide()` consumes the rows + IRT fit produced by a duel and returns one of
`Hold`, `Dethrone`, or `Skip`. No I/O, no chain calls, no audit emission, no
logging beyond `log.debug`. The orchestrator pattern-matches the returned
Verdict to render audit records and persist state.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass

from .chain import Miner
from .config import Config
from .evidence import Row
from .irt import Fit, compute_k

ArtKey = tuple[str, str]

log = logging.getLogger(__name__)


class DuelStatus(enum.Enum):
    COMPLETED = "completed"
    ENVS_QUARANTINED = "envs_quarantined"
    CANCELLED = "cancelled"
    KING_SLOT_DEAD = "king_slot_dead"
    CHAL_SLOT_DEAD = "chal_slot_dead"


@dataclass(frozen=True)
class VerdictEvidence:
    delta: float
    se: float
    z: float
    k: float
    reign_blocks: int
    # (chal_pass, chal_n, king_pass, king_n) — totals, not just passes. With
    # unpaired rows from single-side fails, chal_n and king_n diverge, and an
    # audit reader cannot reconstruct the duel from passes alone.
    rows_per_env: dict[str, tuple[int, int, int, int]]


class Verdict:
    pass


@dataclass(frozen=True)
class Hold(Verdict):
    reason: str
    evidence: VerdictEvidence


@dataclass(frozen=True)
class Dethrone(Verdict):
    new_champion: tuple[int, str]
    evidence: VerdictEvidence


@dataclass(frozen=True)
class Skip(Verdict):
    reason: str


def _rows_per_env(rows: list[Row], chal_uid: int, king_uid: int,
                  ) -> dict[str, tuple[int, int, int, int]]:
    out: dict[str, list[int]] = {}
    for r in rows:
        if r.m not in (chal_uid, king_uid):
            continue
        slot = out.setdefault(r.e, [0, 0, 0, 0])
        i = 0 if r.m == chal_uid else 2
        slot[i] += int(r.p); slot[i + 1] += 1
    return {e: tuple(s) for e, s in out.items()}


def decide(rows: list[Row], fit: Fit, status: DuelStatus,
           king: Miner, chal: Miner, art_keys: list[ArtKey],
           reign_blocks: int, cfg: Config) -> Verdict:
    """Pure decision: maps duel evidence to a Verdict.

    Non-decisions (cancelled, envs broken, no rows, degenerate fit) → Skip;
    caller audits as duel_aborted. Otherwise compute z = Δθ̂ / SE against
    k(reign): z > k → Dethrone, else Hold("z_below_k", ev).

    KING_SLOT_DEAD / CHAL_SLOT_DEAD are not short-circuited — synthetic-loss
    rows already encode the broken-slot evidence (one side fails to deliver →
    p=0 row for that miner). The contrast z over those rows expresses
    "first functioning challenger wins by default". Caller still tears down
    the dead slot regardless of the verdict."""
    if status is DuelStatus.CANCELLED:
        return Skip("cancelled")
    if status is DuelStatus.ENVS_QUARANTINED:
        return Skip("envs_quarantined")
    if not rows:
        return Skip("no_rows")
    if fit.degenerate:
        return Skip("degenerate_fit")
    delta, se = fit.contrast(art_keys.index((chal.model, chal.revision)),
                             art_keys.index((king.model, king.revision)))
    z = delta / se if se > 0 else 0.0
    k = compute_k(reign_blocks, cfg.k_init, cfg.k_final, cfg.k_halflife)
    ev = VerdictEvidence(delta=float(delta), se=float(se), z=float(z), k=float(k),
                         reign_blocks=int(reign_blocks),
                         rows_per_env=_rows_per_env(rows, chal.uid, king.uid))
    log.debug("decide: Δθ̂=%+.3f±%.3f z=%+.2f k=%.2f reign=%db",
              delta, se, z, k, reign_blocks)
    if z > k:
        return Dethrone((chal.uid, chal.revision), ev)
    return Hold("z_below_k", ev)
