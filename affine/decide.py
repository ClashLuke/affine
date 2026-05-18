from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import log, log1p, sqrt


class DuelOutcome(str, Enum):
    DETHRONE = "dethrone"
    NO_DETHRONE = "no_dethrone"


@dataclass(frozen=True)
class Verdict:
    """Duel result plus the latest CS statistics."""

    outcome: DuelOutcome
    reason: str
    delta_hat: float
    ci_low: float
    ci_hi: float
    rounds: int
    log_capital_at_zero: float

    @property
    def status(self) -> str:
        return self.reason


@dataclass
class BettingCS:
    """Anytime-valid betting CS for the mean of Y in [-1, +1]."""

    alpha: float
    clip: float = 0.75
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0
    _y: list[float] = field(default_factory=list)
    _var_prev: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if not (0.0 < self.clip < 1.0):
            raise ValueError(f"clip must be in (0, 1), got {self.clip}")

    def update(self, y: float) -> None:
        if not -1.0 <= y <= 1.0:
            raise ValueError(f"y must be in [-1, 1], got {y}")
        self._var_prev.append(max(self.M2 / max(self.n - 1, 1), 1e-4))
        self._y.append(float(y))
        self.n += 1
        delta = y - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (y - self.mean)

    def _log_capital(self, m: float, sign: int) -> float:
        if self.n == 0:
            return 0.0
        log_target = log(1.0 / self.alpha)
        log_cap = 0.0
        for s, (y, var_prev) in enumerate(zip(self._y, self._var_prev), start=1):
            lam_raw = sqrt(2.0 * log_target / (s * var_prev))
            denom = 1.0 + m if sign > 0 else 1.0 - m
            lam_cap = self.clip / max(denom, 1e-6)
            lam = min(lam_raw, lam_cap) * sign
            log_cap += log1p(lam * (y - m))
            if log_cap >= log_target:
                return log_cap
        return log_cap

    def ci(self) -> tuple[float, float]:
        if self.n == 0:
            return -1.0, 1.0
        log_target = log(1.0 / self.alpha)

        def lower() -> float:
            lo, hi = -1.0, self.mean
            if self._log_capital(lo, +1) < log_target:
                return lo
            for _ in range(48):
                mid = 0.5 * (lo + hi)
                if self._log_capital(mid, +1) >= log_target:
                    lo = mid
                else:
                    hi = mid
            return hi

        def upper() -> float:
            lo, hi = self.mean, 1.0
            if self._log_capital(hi, -1) < log_target:
                return hi
            for _ in range(48):
                mid = 0.5 * (lo + hi)
                if self._log_capital(mid, -1) >= log_target:
                    hi = mid
                else:
                    lo = mid
            return lo

        return lower(), upper()


def decide(
    cs: BettingCS,
    *,
    delta_dethrone: float,
    delta_hold: float,
) -> Verdict:
    lo, hi = cs.ci()
    outcome = DuelOutcome.DETHRONE if lo > delta_dethrone else DuelOutcome.NO_DETHRONE
    reason = "dethrone" if lo > delta_dethrone else "hold" if hi <= delta_hold else "continue"
    return Verdict(
        outcome=outcome,
        reason=reason,
        delta_hat=cs.mean,
        ci_low=lo,
        ci_hi=hi,
        rounds=cs.n,
        log_capital_at_zero=cs._log_capital(0.0, +1),
    )
