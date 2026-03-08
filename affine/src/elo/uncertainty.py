"""
Monte Carlo Uncertainty Propagation for Weights

Propagates uncertainty from ELO ratings through to final subnet weights.

The key insight is that point estimate weights ignore rating uncertainty:
- If two players have similar ratings with large uncertainty, their weight
  difference should be smaller than if they have similar ratings with
  small uncertainty.

This module provides:
1. Monte Carlo sampling from rating distributions
2. Weight calculation for each sample
3. Proper uncertainty quantification on final weights
4. Credible intervals that account for rating correlations
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .config import EloConfig, DEFAULT_ELO_CONFIG


@dataclass
class WeightUncertaintyResult:
    """Result of Monte Carlo weight uncertainty propagation."""

    # Point estimates (from mean ratings)
    weight_means: Dict[str, float]

    # Uncertainty from Monte Carlo
    weight_stds: Dict[str, float]
    weight_ci_lower: Dict[str, float]
    weight_ci_upper: Dict[str, float]

    # Distribution percentiles
    weight_medians: Dict[str, float]

    # Raw samples (optional)
    weight_samples: Optional[Dict[str, np.ndarray]] = None

    # Info
    num_samples: int = 0
    confidence_level: float = 0.95
    weight_method: str = "softmax"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "weight_means": self.weight_means,
            "weight_stds": self.weight_stds,
            "weight_ci_lower": self.weight_ci_lower,
            "weight_ci_upper": self.weight_ci_upper,
            "weight_medians": self.weight_medians,
            "num_samples": self.num_samples,
            "confidence_level": self.confidence_level,
            "weight_method": self.weight_method,
        }



class MonteCarloWeightCalculator:
    """
    Monte Carlo propagation of rating uncertainty to weights.

    Given posterior samples or bootstrap samples of ratings, this class
    computes the induced distribution over weights and provides proper
    credible intervals.

    Supports multiple weight calculation methods:
    - "softmax": w_i = exp(β * r_i) / Σ exp(β * r_j)
    - "linear": w_i = (r_i - r_min) / Σ (r_j - r_min)
    - "rank": w_i = 1/rank_i^α / Σ 1/rank_j^α

    The key advantage over point estimates is that this propagates
    rating uncertainty through the nonlinear weight function.
    """

    def __init__(self, config: Optional[EloConfig] = None):
        """
        Initialize the calculator.

        Args:
            config: ELO configuration
        """
        self.config = config or DEFAULT_ELO_CONFIG

    def compute_softmax_weights(
        self,
        ratings: Dict[str, float],
        temperature: float = 400.0,
    ) -> Dict[str, float]:
        """
        Compute softmax weights from ratings.

        w_i = exp(r_i / T) / Σ exp(r_j / T)

        Args:
            ratings: Player ID -> rating
            temperature: Temperature parameter (higher = more uniform)

        Returns:
            Player ID -> weight (sums to 1)
        """
        players = list(ratings.keys())

        if len(players) == 0:
            return {}

        if len(players) == 1:
            return {players[0]: 1.0}

        # Compute log-weights for numerical stability
        max_rating = max(ratings.values())
        log_weights = {p: (r - max_rating) / temperature for p, r in ratings.items()}

        # Softmax
        exp_weights = {p: math.exp(lw) for p, lw in log_weights.items()}
        total = sum(exp_weights.values())

        return {p: w / total for p, w in exp_weights.items()}

    def compute_linear_weights(
        self,
        ratings: Dict[str, float],
        min_weight: float = 0.01,
    ) -> Dict[str, float]:
        """
        Compute linear weights from ratings.

        w_i = max(r_i - r_min, ε) / Σ max(r_j - r_min, ε)

        Args:
            ratings: Player ID -> rating
            min_weight: Minimum relative weight

        Returns:
            Player ID -> weight (sums to 1)
        """
        players = list(ratings.keys())

        if len(players) == 0:
            return {}

        if len(players) == 1:
            return {players[0]: 1.0}

        min_rating = min(ratings.values())

        # Linear transformation with floor
        raw_weights = {p: max(r - min_rating, min_weight) for p, r in ratings.items()}
        total = sum(raw_weights.values())

        return {p: w / total for p, w in raw_weights.items()}

    def compute_rank_weights(
        self,
        ratings: Dict[str, float],
        alpha: float = 1.0,
    ) -> Dict[str, float]:
        """
        Compute rank-based weights.

        w_i = 1/rank_i^α / Σ 1/rank_j^α

        Args:
            ratings: Player ID -> rating
            alpha: Decay exponent (higher = more weight to top ranks)

        Returns:
            Player ID -> weight (sums to 1)
        """
        players = list(ratings.keys())

        if len(players) == 0:
            return {}

        if len(players) == 1:
            return {players[0]: 1.0}

        # Sort by rating descending
        sorted_players = sorted(players, key=lambda p: ratings[p], reverse=True)

        # Compute rank weights
        rank_weights = {}
        for rank, player in enumerate(sorted_players, 1):
            rank_weights[player] = 1.0 / (rank ** alpha)

        total = sum(rank_weights.values())
        return {p: w / total for p, w in rank_weights.items()}

    def propagate_from_samples(
        self,
        rating_samples: Dict[str, np.ndarray],
        weight_method: str = "softmax",
        temperature: float = 400.0,
        min_weight: float = 0.01,
        alpha: float = 1.0,
        confidence_level: Optional[float] = None,
        store_samples: bool = False,
    ) -> WeightUncertaintyResult:
        """
        Propagate uncertainty from rating samples to weights.

        Args:
            rating_samples: Player ID -> array of rating samples
            weight_method: "softmax", "linear", or "rank"
            temperature: Temperature for softmax
            min_weight: Minimum weight for linear
            alpha: Decay exponent for rank
            confidence_level: Confidence level for intervals
            store_samples: Whether to store full weight distributions

        Returns:
            WeightUncertaintyResult with uncertainty quantification
        """
        if confidence_level is None:
            confidence_level = self.config.BOOTSTRAP_CONFIDENCE_LEVEL

        players = list(rating_samples.keys())
        if len(players) == 0:
            raise ValueError("No rating samples provided")

        # Get number of samples (all should have same length)
        n_samples = len(list(rating_samples.values())[0])

        # Choose weight function
        if weight_method == "softmax":
            weight_fn = lambda r: self.compute_softmax_weights(r, temperature)
        elif weight_method == "linear":
            weight_fn = lambda r: self.compute_linear_weights(r, min_weight)
        elif weight_method == "rank":
            weight_fn = lambda r: self.compute_rank_weights(r, alpha)
        else:
            raise ValueError(f"Unknown weight method: {weight_method}")

        # Compute weights for each sample
        weight_samples: Dict[str, List[float]] = {p: [] for p in players}

        for i in range(n_samples):
            # Get ratings for this sample
            sample_ratings = {p: float(rating_samples[p][i]) for p in players}

            # Compute weights
            sample_weights = weight_fn(sample_ratings)

            # Store
            for p in players:
                weight_samples[p].append(sample_weights.get(p, 0.0))

        # Convert to arrays
        weight_arrays = {p: np.array(w) for p, w in weight_samples.items()}

        # Compute statistics
        ci_lower_pct = (1 - confidence_level) / 2 * 100
        ci_upper_pct = (1 + confidence_level) / 2 * 100

        weight_means = {p: float(np.mean(w)) for p, w in weight_arrays.items()}
        weight_stds = {p: float(np.std(w)) for p, w in weight_arrays.items()}
        weight_medians = {p: float(np.median(w)) for p, w in weight_arrays.items()}
        weight_ci_lower = {p: float(np.percentile(w, ci_lower_pct)) for p, w in weight_arrays.items()}
        weight_ci_upper = {p: float(np.percentile(w, ci_upper_pct)) for p, w in weight_arrays.items()}

        return WeightUncertaintyResult(
            weight_means=weight_means,
            weight_stds=weight_stds,
            weight_ci_lower=weight_ci_lower,
            weight_ci_upper=weight_ci_upper,
            weight_medians=weight_medians,
            weight_samples=weight_arrays if store_samples else None,
            num_samples=n_samples,
            confidence_level=confidence_level,
            weight_method=weight_method,
        )

    def propagate_from_mean_std(
        self,
        rating_means: Dict[str, float],
        rating_stds: Dict[str, float],
        num_samples: Optional[int] = None,
        weight_method: str = "softmax",
        temperature: float = 400.0,
        min_weight: float = 0.01,
        alpha: float = 1.0,
        confidence_level: Optional[float] = None,
        correlation: float = 0.0,
        seed: int = 42,
    ) -> WeightUncertaintyResult:
        """
        Propagate uncertainty assuming Gaussian rating distributions.

        Args:
            rating_means: Player ID -> mean rating
            rating_stds: Player ID -> rating standard deviation
            num_samples: Number of Monte Carlo samples
            weight_method: "softmax", "linear", or "rank"
            temperature: Temperature for softmax
            min_weight: Minimum weight for linear
            alpha: Decay exponent for rank
            confidence_level: Confidence level for intervals
            correlation: Correlation between player ratings (0 = independent)
            seed: Random seed

        Returns:
            WeightUncertaintyResult with uncertainty quantification
        """
        if num_samples is None:
            num_samples = self.config.WEIGHT_UNCERTAINTY_SAMPLES

        players = list(rating_means.keys())
        n_players = len(players)

        if n_players == 0:
            raise ValueError("No ratings provided")

        np.random.seed(seed)

        # Generate correlated samples if needed
        if correlation != 0.0 and n_players > 1:
            # Build correlation matrix
            corr_matrix = np.eye(n_players) * (1 - correlation) + np.ones((n_players, n_players)) * correlation

            # Cholesky decomposition
            L = np.linalg.cholesky(corr_matrix)

            # Generate standard normal samples
            z = np.random.standard_normal((num_samples, n_players))

            # Apply correlation structure
            correlated = z @ L.T

            # Scale and shift to get rating samples
            rating_samples = {}
            for i, p in enumerate(players):
                rating_samples[p] = rating_means[p] + rating_stds[p] * correlated[:, i]
        else:
            # Independent samples
            rating_samples = {
                p: np.random.normal(rating_means[p], rating_stds[p], num_samples)
                for p in players
            }

        return self.propagate_from_samples(
            rating_samples,
            weight_method=weight_method,
            temperature=temperature,
            min_weight=min_weight,
            alpha=alpha,
            confidence_level=confidence_level,
        )

    def propagate_from_bayesian(
        self,
        bayesian_result: Any,  # BayesianResult
        weight_method: str = "softmax",
        temperature: float = 400.0,
        min_weight: float = 0.01,
        alpha: float = 1.0,
        confidence_level: Optional[float] = None,
    ) -> WeightUncertaintyResult:
        """
        Propagate uncertainty from Bayesian posterior samples.

        Args:
            bayesian_result: BayesianResult from BayesianBradleyTerry
            weight_method: Weight calculation method
            temperature: Temperature for softmax
            min_weight: Minimum weight for linear
            alpha: Decay exponent for rank
            confidence_level: Confidence level for intervals

        Returns:
            WeightUncertaintyResult with uncertainty quantification
        """
        # Convert skill samples to ELO scale
        from .bayesian import BayesianResult

        if not isinstance(bayesian_result, BayesianResult):
            raise TypeError("Expected BayesianResult")

        scale_factor = self.config.SCALE / math.log(10)
        default_rating = float(self.config.DEFAULT_RATING)

        rating_samples = {}
        for player, skill_samples in bayesian_result.skill_samples.items():
            rating_samples[player] = scale_factor * skill_samples + default_rating

        return self.propagate_from_samples(
            rating_samples,
            weight_method=weight_method,
            temperature=temperature,
            min_weight=min_weight,
            alpha=alpha,
            confidence_level=confidence_level,
        )

    def propagate_from_bootstrap(
        self,
        bootstrap_result: Any,  # BootstrapResult
        weight_method: str = "softmax",
        temperature: float = 400.0,
        min_weight: float = 0.01,
        alpha: float = 1.0,
        confidence_level: Optional[float] = None,
    ) -> WeightUncertaintyResult:
        """
        Propagate uncertainty from bootstrap samples.

        Args:
            bootstrap_result: BootstrapResult from PairedBootstrapBradleyTerry
            weight_method: Weight calculation method
            temperature: Temperature for softmax
            min_weight: Minimum weight for linear
            alpha: Decay exponent for rank
            confidence_level: Confidence level for intervals

        Returns:
            WeightUncertaintyResult with uncertainty quantification
        """
        from .bootstrap import BootstrapResult

        if not isinstance(bootstrap_result, BootstrapResult):
            raise TypeError("Expected BootstrapResult")

        if bootstrap_result.rating_samples is None:
            raise ValueError("BootstrapResult does not contain samples. "
                           "Rerun fit() with store_samples=True")

        return self.propagate_from_samples(
            bootstrap_result.rating_samples,
            weight_method=weight_method,
            temperature=temperature,
            min_weight=min_weight,
            alpha=alpha,
            confidence_level=confidence_level,
        )


def compute_weight_uncertainty(
    rating_means: Dict[str, float],
    rating_stds: Dict[str, float],
    config: Optional[EloConfig] = None,
    weight_method: str = "softmax",
    temperature: float = 400.0,
    num_samples: int = 1000,
    seed: int = 42,
) -> WeightUncertaintyResult:
    """
    Convenience function to compute weight uncertainty from rating statistics.

    Args:
        rating_means: Player ID -> mean rating
        rating_stds: Player ID -> rating standard deviation
        config: ELO configuration
        weight_method: "softmax", "linear", or "rank"
        temperature: Temperature for softmax
        num_samples: Number of Monte Carlo samples
        seed: Random seed

    Returns:
        WeightUncertaintyResult with uncertainty quantification
    """
    calc = MonteCarloWeightCalculator(config)
    return calc.propagate_from_mean_std(
        rating_means,
        rating_stds,
        num_samples=num_samples,
        weight_method=weight_method,
        temperature=temperature,
        seed=seed,
    )
