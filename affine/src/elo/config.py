"""
ELO Configuration

Defines configuration parameters for the ELO rating system.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Any


@dataclass
class EloConfig:
    """Configuration for ELO rating calculations."""

    # ELO calculation parameters
    K_FACTOR: int = 32  # Base K-factor for rating changes
    K_FACTOR_NEW_PLAYER: int = 40  # Higher K for new players (< 30 matches)
    K_FACTOR_ESTABLISHED: int = 24  # Lower K for established players (> 100 matches)
    K_FACTOR_ELITE: int = 16  # Lowest K for elite players (rating > 1800)

    SCALE: int = 400  # Standard ELO scale factor for expected score calculation

    # Rating bounds
    DEFAULT_RATING: Decimal = field(default_factory=lambda: Decimal("1500"))
    RATING_FLOOR: Decimal = field(default_factory=lambda: Decimal("100"))
    RATING_CEILING: Decimal = field(default_factory=lambda: Decimal("3000"))

    # Match thresholds
    NEW_PLAYER_THRESHOLD: int = 30  # Matches before K-factor decreases
    ESTABLISHED_THRESHOLD: int = 100  # Matches for "established" status
    ELITE_RATING_THRESHOLD: Decimal = field(default_factory=lambda: Decimal("1800"))

    # Pairwise comparison settings (for backward compatibility)
    SCORE_MARGIN: Decimal = field(default_factory=lambda: Decimal("0.01"))  # Score diff for draw

    # System settings
    ELO_ENABLED: bool = False  # Master switch for ELO processing

    # Paired match settings
    PAIRED_K_FACTOR_MULTIPLIER: float = 1.41  # sqrt(2), for combining two games' worth of info

    # MLE refit settings (Hybrid ELO system)
    MLE_REFIT_MIN_GAMES: int = 50  # Minimum games between MLE refits
    MLE_REFIT_MAX_MINUTES: int = 30  # Maximum minutes between MLE refits

    # First-mover advantage estimation
    ESTIMATE_FIRST_MOVER_ADVANTAGE: bool = True  # Enable α parameter estimation

    # Sharpness-Aware Minimization (SAM) for Bradley-Terry MLE
    # SAM finds flatter minima without explicit L2 regularization
    BT_SAM_RHO: float = 0.05  # SAM perturbation radius (0 = pure MLE)

    # Bayesian Bradley-Terry settings
    BAYESIAN_NUM_SAMPLES: int = 2000  # MCMC samples for posterior
    BAYESIAN_NUM_WARMUP: int = 1000  # Warmup samples (discarded)
    BAYESIAN_NUM_CHAINS: int = 4  # Number of MCMC chains
    BAYESIAN_PRIOR_SCALE: float = 1.0  # Scale of Student-t prior on skills

    # Bootstrap settings
    BOOTSTRAP_NUM_SAMPLES: int = 1000  # Number of bootstrap resamples
    BOOTSTRAP_CONFIDENCE_LEVEL: float = 0.95  # Confidence level for intervals

    # Uncertainty propagation
    WEIGHT_UNCERTAINTY_SAMPLES: int = 1000  # Samples for weight uncertainty

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "k_factor": self.K_FACTOR,
            "k_factor_new_player": self.K_FACTOR_NEW_PLAYER,
            "k_factor_established": self.K_FACTOR_ESTABLISHED,
            "k_factor_elite": self.K_FACTOR_ELITE,
            "scale": self.SCALE,
            "default_rating": float(self.DEFAULT_RATING),
            "rating_floor": float(self.RATING_FLOOR),
            "rating_ceiling": float(self.RATING_CEILING),
            "new_player_threshold": self.NEW_PLAYER_THRESHOLD,
            "established_threshold": self.ESTABLISHED_THRESHOLD,
            "elite_rating_threshold": float(self.ELITE_RATING_THRESHOLD),
            "score_margin": float(self.SCORE_MARGIN),
            "elo_enabled": self.ELO_ENABLED,
            # Paired match settings
            "paired_k_factor_multiplier": self.PAIRED_K_FACTOR_MULTIPLIER,
            # MLE refit settings
            "mle_refit_min_games": self.MLE_REFIT_MIN_GAMES,
            "mle_refit_max_minutes": self.MLE_REFIT_MAX_MINUTES,
            # First-mover advantage
            "estimate_first_mover_advantage": self.ESTIMATE_FIRST_MOVER_ADVANTAGE,
            # SAM optimization
            "bt_sam_rho": self.BT_SAM_RHO,
            # Bayesian settings
            "bayesian_num_samples": self.BAYESIAN_NUM_SAMPLES,
            "bayesian_num_warmup": self.BAYESIAN_NUM_WARMUP,
            "bayesian_num_chains": self.BAYESIAN_NUM_CHAINS,
            "bayesian_prior_scale": self.BAYESIAN_PRIOR_SCALE,
            # Bootstrap settings
            "bootstrap_num_samples": self.BOOTSTRAP_NUM_SAMPLES,
            "bootstrap_confidence_level": self.BOOTSTRAP_CONFIDENCE_LEVEL,
            # Uncertainty propagation
            "weight_uncertainty_samples": self.WEIGHT_UNCERTAINTY_SAMPLES,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EloConfig":
        """Create config from dictionary."""
        return cls(
            K_FACTOR=data.get("k_factor", 32),
            K_FACTOR_NEW_PLAYER=data.get("k_factor_new_player", 40),
            K_FACTOR_ESTABLISHED=data.get("k_factor_established", 24),
            K_FACTOR_ELITE=data.get("k_factor_elite", 16),
            SCALE=data.get("scale", 400),
            DEFAULT_RATING=Decimal(str(data.get("default_rating", 1500))),
            RATING_FLOOR=Decimal(str(data.get("rating_floor", 100))),
            RATING_CEILING=Decimal(str(data.get("rating_ceiling", 3000))),
            NEW_PLAYER_THRESHOLD=data.get("new_player_threshold", 30),
            ESTABLISHED_THRESHOLD=data.get("established_threshold", 100),
            ELITE_RATING_THRESHOLD=Decimal(str(data.get("elite_rating_threshold", 1800))),
            SCORE_MARGIN=Decimal(str(data.get("score_margin", 0.01))),
            ELO_ENABLED=data.get("elo_enabled", False),
            # Paired match settings
            PAIRED_K_FACTOR_MULTIPLIER=data.get("paired_k_factor_multiplier", 1.41),
            # MLE refit settings
            MLE_REFIT_MIN_GAMES=data.get("mle_refit_min_games", 50),
            MLE_REFIT_MAX_MINUTES=data.get("mle_refit_max_minutes", 30),
            # First-mover advantage
            ESTIMATE_FIRST_MOVER_ADVANTAGE=data.get("estimate_first_mover_advantage", True),
            # SAM optimization
            BT_SAM_RHO=data.get("bt_sam_rho", 0.05),
            # Bayesian settings
            BAYESIAN_NUM_SAMPLES=data.get("bayesian_num_samples", 2000),
            BAYESIAN_NUM_WARMUP=data.get("bayesian_num_warmup", 1000),
            BAYESIAN_NUM_CHAINS=data.get("bayesian_num_chains", 4),
            BAYESIAN_PRIOR_SCALE=data.get("bayesian_prior_scale", 1.0),
            # Bootstrap settings
            BOOTSTRAP_NUM_SAMPLES=data.get("bootstrap_num_samples", 1000),
            BOOTSTRAP_CONFIDENCE_LEVEL=data.get("bootstrap_confidence_level", 0.95),
            # Uncertainty propagation
            WEIGHT_UNCERTAINTY_SAMPLES=data.get("weight_uncertainty_samples", 1000),
        )


# Global default config instance
DEFAULT_ELO_CONFIG = EloConfig()
