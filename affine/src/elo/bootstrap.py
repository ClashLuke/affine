"""
Paired Bootstrap Bradley-Terry Model

Implements bootstrap confidence intervals for Bradley-Terry ratings.

Key feature: Resamples PAIRED observations together to preserve the
paired structure of match+rematch data. This is critical because
treating the two games in a pair as independent would underestimate
uncertainty.

Bootstrap procedure:
1. Group matches into pairs (using pair_uuid)
2. Resample pairs with replacement
3. Fit Bradley-Terry to each bootstrap sample
4. Compute percentile confidence intervals from bootstrap distribution
"""

import random
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import numpy as np

from .config import EloConfig, DEFAULT_ELO_CONFIG
from .models import MatchResult, PairedMatchResult
from .paired_bradley_terry import PairedBradleyTerryModel, BradleyTerryResult


@dataclass
class BootstrapResult:
    """Result of bootstrap confidence interval estimation."""

    # Point estimates (from original data)
    rating_means: Dict[str, Decimal]

    # Bootstrap confidence intervals
    rating_ci_lower: Dict[str, Decimal]
    rating_ci_upper: Dict[str, Decimal]

    # Standard errors from bootstrap
    rating_se: Dict[str, Decimal]

    # First-mover advantage
    first_mover_mean: Optional[float] = None
    first_mover_ci: Optional[Tuple[float, float]] = None
    first_mover_se: Optional[float] = None

    # Bootstrap distribution (optional, for further analysis)
    rating_samples: Optional[Dict[str, np.ndarray]] = None
    first_mover_samples: Optional[np.ndarray] = None

    # Bootstrap info
    num_bootstrap_samples: int = 0
    num_pairs: int = 0
    num_matches: int = 0
    confidence_level: float = 0.95

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "rating_means": {k: float(v) for k, v in self.rating_means.items()},
            "rating_ci_lower": {k: float(v) for k, v in self.rating_ci_lower.items()},
            "rating_ci_upper": {k: float(v) for k, v in self.rating_ci_upper.items()},
            "rating_se": {k: float(v) for k, v in self.rating_se.items()},
            "first_mover_mean": self.first_mover_mean,
            "first_mover_ci": self.first_mover_ci,
            "first_mover_se": self.first_mover_se,
            "num_bootstrap_samples": self.num_bootstrap_samples,
            "num_pairs": self.num_pairs,
            "num_matches": self.num_matches,
            "confidence_level": self.confidence_level,
        }

    def ci_overlap(self, player_a: str, player_b: str) -> bool:
        """
        Check if confidence intervals overlap between two players.

        Returns:
            True if CIs overlap, False if clearly separated
        """
        a_lower = self.rating_ci_lower[player_a]
        a_upper = self.rating_ci_upper[player_a]
        b_lower = self.rating_ci_lower[player_b]
        b_upper = self.rating_ci_upper[player_b]

        return not (a_lower > b_upper or b_lower > a_upper)


class PairedBootstrapBradleyTerry:
    """
    Bootstrap confidence intervals for Bradley-Terry ratings.

    This class implements paired bootstrap resampling, which is critical
    for correctly estimating uncertainty in match+rematch data.

    The key insight is that match-rematch pairs should be resampled together,
    not independently, because:
    1. They share the same pair of players
    2. They are designed to cancel first-mover effects
    3. Treating them independently would underestimate variance

    Process:
    1. Group all matches into pairs (using pair_uuid)
    2. For each bootstrap sample:
       a. Resample pairs with replacement
       b. Fit Bradley-Terry MLE to the resampled data
       c. Store the resulting ratings
    3. Compute percentile confidence intervals from the bootstrap distribution
    """

    def __init__(self, config: Optional[EloConfig] = None):
        """
        Initialize the bootstrap model.

        Args:
            config: ELO configuration
        """
        self.config = config or DEFAULT_ELO_CONFIG

        # Match data
        self._pairs: List[PairedMatchResult] = []
        self._unpaired: List[MatchResult] = []  # Matches without pair_uuid

        # Result
        self._result: Optional[BootstrapResult] = None

    def add_match(self, match: MatchResult) -> None:
        """Add a single match. Paired matches are staged until their partner arrives."""
        if match.pair_uuid is None:
            self._unpaired.append(match)
        else:
            for pair in self._pairs:
                if pair.pair_uuid == match.pair_uuid:
                    return

            if not hasattr(self, '_pair_staging'):
                self._pair_staging: Dict[str, MatchResult] = {}

            if match.pair_uuid in self._pair_staging:
                partner = self._pair_staging.pop(match.pair_uuid)
                matches = sorted([partner, match], key=lambda m: m.pair_sequence or 0)
                game_type = "unknown"
                if matches[0].game_result and "game_type" in matches[0].game_result:
                    game_type = matches[0].game_result["game_type"]
                paired = PairedMatchResult.from_match_pair(
                    matches[0], matches[1], game_type, match.pair_uuid
                )
                self._pairs.append(paired)
            else:
                self._pair_staging[match.pair_uuid] = match

    def add_paired_result(self, paired: PairedMatchResult) -> None:
        """
        Add a paired match result directly.

        Args:
            paired: PairedMatchResult containing both games
        """
        self._pairs.append(paired)

    def add_matches(self, matches: List[MatchResult]) -> None:
        """
        Add multiple matches and automatically pair them.

        Args:
            matches: List of MatchResult objects
        """
        pair_map: Dict[str, List[MatchResult]] = {}
        unpaired = []

        for match in matches:
            if match.pair_uuid:
                if match.pair_uuid not in pair_map:
                    pair_map[match.pair_uuid] = []
                pair_map[match.pair_uuid].append(match)
            else:
                unpaired.append(match)

        for pair_uuid, pair_matches in pair_map.items():
            if len(pair_matches) == 2:
                pair_matches.sort(key=lambda m: m.pair_sequence or 0)
                match_1, match_2 = pair_matches

                game_type = "unknown"
                if match_1.game_result and "game_type" in match_1.game_result:
                    game_type = match_1.game_result["game_type"]

                paired = PairedMatchResult.from_match_pair(
                    match_1, match_2, game_type, pair_uuid
                )
                self._pairs.append(paired)
            else:
                unpaired.extend(pair_matches)

        self._unpaired.extend(unpaired)

    def _resample_data(self, rng: random.Random) -> Tuple[List[PairedMatchResult], List[MatchResult]]:
        """
        Resample data with replacement, keeping pairs together.

        Args:
            rng: Random number generator

        Returns:
            Tuple of (resampled_pairs, resampled_unpaired)
        """
        n_pairs = len(self._pairs)
        if n_pairs > 0:
            resampled_pairs = rng.choices(self._pairs, k=n_pairs)
        else:
            resampled_pairs = []

        n_unpaired = len(self._unpaired)
        if n_unpaired > 0:
            resampled_unpaired = rng.choices(self._unpaired, k=n_unpaired)
        else:
            resampled_unpaired = []

        return resampled_pairs, resampled_unpaired

    def _fit_sample(
        self,
        pairs: List[PairedMatchResult],
        unpaired: List[MatchResult],
        estimate_first_mover: bool,
    ) -> Optional[BradleyTerryResult]:
        """
        Fit Bradley-Terry to a bootstrap sample.

        Args:
            pairs: Resampled pairs
            unpaired: Resampled unpaired matches
            estimate_first_mover: Whether to estimate first-mover advantage

        Returns:
            BradleyTerryResult or None if fitting failed
        """
        model = PairedBradleyTerryModel(self.config)

        for pair in pairs:
            model.add_paired_result(pair)

        for match in unpaired:
            model.add_match_result(match)

        if len(model._players) < 2:
            return None
        try:
            return model.fit(estimate_first_mover=estimate_first_mover)
        except (ValueError, RuntimeError):
            return None

    def fit(
        self,
        num_bootstrap: Optional[int] = None,
        estimate_first_mover: Optional[bool] = None,
        confidence_level: Optional[float] = None,
        seed: int = 42,
        store_samples: bool = False,
    ) -> BootstrapResult:
        """
        Fit model and compute bootstrap confidence intervals.

        Args:
            num_bootstrap: Number of bootstrap samples (default: from config)
            estimate_first_mover: Whether to estimate first-mover advantage
            confidence_level: Confidence level (default: from config)
            seed: Random seed for reproducibility
            store_samples: Whether to store full bootstrap distribution

        Returns:
            BootstrapResult with confidence intervals
        """
        if len(self._pairs) + len(self._unpaired) == 0:
            raise ValueError("No matches to fit")

        if num_bootstrap is None:
            num_bootstrap = self.config.BOOTSTRAP_NUM_SAMPLES
        if estimate_first_mover is None:
            estimate_first_mover = self.config.ESTIMATE_FIRST_MOVER_ADVANTAGE
        if confidence_level is None:
            confidence_level = self.config.BOOTSTRAP_CONFIDENCE_LEVEL

        rng = random.Random(seed)

        original_result = self._fit_sample(
            self._pairs, self._unpaired, estimate_first_mover
        )

        if original_result is None:
            raise ValueError("Failed to fit original data")

        players = list(original_result.ratings.keys())
        rating_samples: Dict[str, List[float]] = {p: [] for p in players}
        first_mover_samples: List[float] = []

        successful_samples = 0
        for b in range(num_bootstrap):
            resampled_pairs, resampled_unpaired = self._resample_data(rng)
            result = self._fit_sample(resampled_pairs, resampled_unpaired, estimate_first_mover)

            if result is not None:
                successful_samples += 1

                for player in players:
                    if player in result.ratings:
                        rating_samples[player].append(float(result.ratings[player]))

                if estimate_first_mover:
                    first_mover_samples.append(result.first_mover_elo)

        if successful_samples < num_bootstrap * 0.8:
            raise ValueError(f"Too many bootstrap failures: {successful_samples}/{num_bootstrap}")

        ci_lower_pct = (1 - confidence_level) / 2 * 100
        ci_upper_pct = (1 + confidence_level) / 2 * 100

        rating_means: Dict[str, Decimal] = {}
        rating_ci_lower: Dict[str, Decimal] = {}
        rating_ci_upper: Dict[str, Decimal] = {}
        rating_se: Dict[str, Decimal] = {}

        for player in players:
            samples = np.array(rating_samples[player])
            rating_means[player] = original_result.ratings[player]
            rating_ci_lower[player] = Decimal(str(round(np.percentile(samples, ci_lower_pct), 2)))
            rating_ci_upper[player] = Decimal(str(round(np.percentile(samples, ci_upper_pct), 2)))
            rating_se[player] = Decimal(str(round(np.std(samples), 2)))

        first_mover_mean = None
        first_mover_ci = None
        first_mover_se = None

        if estimate_first_mover and len(first_mover_samples) > 0:
            fm_array = np.array(first_mover_samples)
            first_mover_mean = original_result.first_mover_elo
            first_mover_ci = (
                float(np.percentile(fm_array, ci_lower_pct)),
                float(np.percentile(fm_array, ci_upper_pct)),
            )
            first_mover_se = float(np.std(fm_array))

        self._result = BootstrapResult(
            rating_means=rating_means,
            rating_ci_lower=rating_ci_lower,
            rating_ci_upper=rating_ci_upper,
            rating_se=rating_se,
            first_mover_mean=first_mover_mean,
            first_mover_ci=first_mover_ci,
            first_mover_se=first_mover_se,
            rating_samples={p: np.array(s) for p, s in rating_samples.items()} if store_samples else None,
            first_mover_samples=np.array(first_mover_samples) if store_samples and estimate_first_mover else None,
            num_bootstrap_samples=successful_samples,
            num_pairs=len(self._pairs),
            num_matches=len(self._pairs) * 2 + len(self._unpaired),
            confidence_level=confidence_level,
        )

        return self._result

    def clear(self) -> None:
        """Clear all data."""
        self._pairs.clear()
        self._unpaired.clear()
        self._result = None

    @property
    def result(self) -> Optional[BootstrapResult]:
        """Get the fitted result."""
        return self._result


def bootstrap_confidence_intervals(
    matches: List[MatchResult],
    config: Optional[EloConfig] = None,
    num_bootstrap: int = 1000,
    estimate_first_mover: bool = True,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """
    Convenience function to compute bootstrap confidence intervals.

    Args:
        matches: List of MatchResult objects
        config: ELO configuration
        num_bootstrap: Number of bootstrap samples
        estimate_first_mover: Whether to estimate first-mover advantage
        confidence_level: Confidence level for intervals
        seed: Random seed

    Returns:
        BootstrapResult with confidence intervals
    """
    model = PairedBootstrapBradleyTerry(config)
    model.add_matches(matches)
    return model.fit(
        num_bootstrap=num_bootstrap,
        estimate_first_mover=estimate_first_mover,
        confidence_level=confidence_level,
        seed=seed,
    )


def bootstrap_from_pairs(
    pairs: List[PairedMatchResult],
    config: Optional[EloConfig] = None,
    num_bootstrap: int = 1000,
    estimate_first_mover: bool = True,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """
    Convenience function to compute bootstrap CIs from paired results.

    Args:
        pairs: List of PairedMatchResult objects
        config: ELO configuration
        num_bootstrap: Number of bootstrap samples
        estimate_first_mover: Whether to estimate first-mover advantage
        confidence_level: Confidence level for intervals
        seed: Random seed

    Returns:
        BootstrapResult with confidence intervals
    """
    model = PairedBootstrapBradleyTerry(config)
    for pair in pairs:
        model.add_paired_result(pair)
    return model.fit(
        num_bootstrap=num_bootstrap,
        estimate_first_mover=estimate_first_mover,
        confidence_level=confidence_level,
        seed=seed,
    )
