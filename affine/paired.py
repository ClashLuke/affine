from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import binomtest


@dataclass(frozen=True)
class PairCounts:
    challenger_only: int = 0
    champion_only: int = 0
    both_pass: int = 0
    both_fail: int = 0

    @property
    def total(self) -> int:
        return self.challenger_only + self.champion_only + self.both_pass + self.both_fail

    @property
    def discordant(self) -> int:
        return self.challenger_only + self.champion_only

    def add(self, champion_pass: int, challenger_pass: int) -> "PairCounts":
        if challenger_pass and not champion_pass:
            return PairCounts(self.challenger_only + 1, self.champion_only, self.both_pass, self.both_fail)
        if champion_pass and not challenger_pass:
            return PairCounts(self.challenger_only, self.champion_only + 1, self.both_pass, self.both_fail)
        if challenger_pass and champion_pass:
            return PairCounts(self.challenger_only, self.champion_only, self.both_pass + 1, self.both_fail)
        return PairCounts(self.challenger_only, self.champion_only, self.both_pass, self.both_fail + 1)


def pair_p_value(chal_only: int, discordant: int) -> float:
    """Exact one-sided p-value: P(X >= chal_only | X ~ Binomial(discordant, 0.5))."""
    if discordant == 0 or chal_only == 0:
        return 1.0
    return float(binomtest(chal_only, discordant, p=0.5, alternative="greater").pvalue)


@dataclass(frozen=True)
class PairDecision:
    dethrone: bool
    reason: str
    counts: PairCounts
    alpha: float
    min_discordant: int
    p_value: float


def decide_paired(counts: PairCounts, *, alpha: float, min_discordant: int) -> PairDecision:
    p = pair_p_value(counts.challenger_only, counts.discordant)
    if counts.discordant < min_discordant:
        return PairDecision(False, "too_few_discordant", counts, alpha, min_discordant, p)
    if counts.challenger_only <= counts.champion_only:
        return PairDecision(False, "challenger_not_ahead", counts, alpha, min_discordant, p)
    if p > alpha:
        return PairDecision(False, "p_above_alpha", counts, alpha, min_discordant, p)
    return PairDecision(True, "exact_paired_test", counts, alpha, min_discordant, p)


def alpha_for_reign(blocks: int, start: float, final: float, halflife: int) -> float:
    age = max(int(blocks), 0)
    if age == 0:
        return float(start)
    return float(final - (final - start) * (0.5 ** (age / halflife)))
