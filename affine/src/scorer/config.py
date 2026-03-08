"""
Scorer Configuration

Central configuration for the scoring algorithm.
All parameters are defined as constants for clarity and maintainability.
"""

from decimal import Decimal
from typing import Dict, Any


class ScorerConfig:
    """Configuration for the four-stage scoring algorithm."""
    
    # Stage 2: Pareto Frontier Anti-Plagiarism
    Z_SCORE: float = 1.5
    """
    Z-score for statistical confidence interval in threshold calculation.

    Uses standard error (SE) based approach to adjust threshold by sample size:
    - SE = sqrt(p * (1-p) / n)
    - gap = z * SE

    Z-score values:
    - 1.0: ~68% confidence (more aggressive, smaller gaps)
    - 1.5: ~87% confidence (balanced, recommended)
    - 1.96: 95% confidence (more conservative, larger gaps)

    Higher sample counts → smaller SE → smaller gap → easier to beat.
    Lower sample counts → larger SE → larger gap → harder to beat.

    Recommended value: 1.5
    """

    MIN_IMPROVEMENT: float = 0.02
    """
    Minimum improvement required for later miner to beat earlier miner.

    Ensures that even with very large sample sizes (small SE), there's still
    a minimum gap to prevent noise and random fluctuations from allowing
    copies to beat originals.

    Example: If SE-based gap = 0.01 but MIN_IMPROVEMENT = 0.02,
    the actual gap used will be 0.02.

    Recommended value: 0.02 (2%)
    """

    MAX_IMPROVEMENT: float = 0.10
    """
    Maximum improvement threshold cap.

    Caps the required score gap to prevent unreasonably high thresholds
    when sample size is very small (large SE).

    Example: If SE-based gap = 0.25 but MAX_IMPROVEMENT = 0.10,
    the actual gap used will be capped at 0.10.

    Recommended value: 0.10 (10%)
    """
    
    SCORE_PRECISION: int = 3
    """Number of decimal places for score comparison (avoid floating point issues)."""
    
    # Stage 3: Subset Scoring
    MAX_LAYERS: int = 6
    """Maximum number of layers to evaluate. Due to exponential growth, 6 layers provide sufficient differentiation (2^5 = 32x difference between L1 and L6)."""
    
    SUBSET_WEIGHT_BASE: int = 1
    """Base weight multiplier for subset layers (N for L1, N*2 for L2, N*4 for L3, etc.)."""
    
    SUBSET_WEIGHT_EXPONENT: int = 2
    """Exponent base for layer weights (layer_weight = N * base^(layer-1))."""
    
    DECAY_FACTOR: float = 0.5
    """
    Rank-based decay factor for score_proportional weighting.
    
    Applied as: adjusted_score = score × decay_factor^(rank - 1)
    - Rank 1: score × 1.0
    - Rank 2: score × decay_factor^1
    - Rank 3: score × decay_factor^2
    
    Set to 1.0 to disable decay (all ranks weighted equally).
    Set to 0.5 for exponential decay (each rank gets 50% of previous).
    """
    
    # Stage 4: Weight Normalization
    MIN_WEIGHT_THRESHOLD: float = 0.01
    """Minimum weight threshold (1%). Miners below this are set to 0."""
    
    # Stage 1: Data Collection
    MIN_COMPLETENESS: float = 0.9
    """Minimum sample completeness required."""
    
    # Environment Score Normalization
    # Format: env_name -> (min_score, max_score)
    # Scores will be normalized to [0, 1] range: (score - min) / (max - min)
    ENV_SCORE_RANGES: Dict[str, tuple] = {
        'agentgym:sciworld': (-100, 100.0)  # sciworld 分数范围 0-100
    }
    
    # Database & Storage
    SCORE_RECORD_TTL_DAYS: int = 30
    """TTL for score_snapshots table (in days)."""

    # ==========================================================================
    # ELO Rating System Configuration
    # ==========================================================================

    ELO_ENABLED: bool = False
    """Master switch for ELO rating processing. When True, ELO ratings are calculated."""

    ELO_K_FACTOR: int = 32
    """Base K-factor for ELO rating changes. Higher = more volatile ratings."""

    ELO_K_FACTOR_NEW_PLAYER: int = 40
    """K-factor for new players (< 30 matches). Higher for faster calibration."""

    ELO_K_FACTOR_ESTABLISHED: int = 24
    """K-factor for established players (> 100 matches). Lower for stability."""

    ELO_K_FACTOR_ELITE: int = 16
    """K-factor for elite players (rating > 1800). Lowest for maximum stability."""

    ELO_SCALE: int = 400
    """Standard ELO scale factor for expected score calculation."""

    ELO_DEFAULT_RATING: Decimal = Decimal("1500")
    """Default starting ELO rating for new miners."""

    ELO_RATING_FLOOR: Decimal = Decimal("100")
    """Minimum possible ELO rating."""

    ELO_RATING_CEILING: Decimal = Decimal("3000")
    """Maximum possible ELO rating."""

    ELO_SCORE_MARGIN: Decimal = Decimal("0.01")
    """Score difference threshold for declaring a draw in pairwise comparison."""

    ELO_NEW_PLAYER_THRESHOLD: int = 30
    """Matches before a player is no longer considered 'new'."""

    ELO_ESTABLISHED_THRESHOLD: int = 100
    """Matches before a player is considered 'established'."""

    ELO_ELITE_RATING_THRESHOLD: Decimal = Decimal("1800")
    """Rating threshold for 'elite' status (lower K-factor)."""

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Export configuration as dictionary for storage in snapshots."""
        return {
            'z_score': cls.Z_SCORE,
            'min_improvement': cls.MIN_IMPROVEMENT,
            'max_improvement': cls.MAX_IMPROVEMENT,
            'score_precision': cls.SCORE_PRECISION,
            'max_layers': cls.MAX_LAYERS,
            'subset_weight_base': cls.SUBSET_WEIGHT_BASE,
            'subset_weight_exponent': cls.SUBSET_WEIGHT_EXPONENT,
            'decay_factor': cls.DECAY_FACTOR,
            'min_weight_threshold': cls.MIN_WEIGHT_THRESHOLD,
            'min_completeness': cls.MIN_COMPLETENESS,
            # ELO config
            'elo_enabled': cls.ELO_ENABLED,
            'elo_k_factor': cls.ELO_K_FACTOR,
            'elo_scale': cls.ELO_SCALE,
            'elo_default_rating': float(cls.ELO_DEFAULT_RATING),
            'elo_score_margin': float(cls.ELO_SCORE_MARGIN),
        }

    @classmethod
    def get_elo_config(cls) -> "EloConfig":
        """Get ELO configuration as an EloConfig instance."""
        from affine.src.elo.config import EloConfig
        return EloConfig(
            K_FACTOR=cls.ELO_K_FACTOR,
            K_FACTOR_NEW_PLAYER=cls.ELO_K_FACTOR_NEW_PLAYER,
            K_FACTOR_ESTABLISHED=cls.ELO_K_FACTOR_ESTABLISHED,
            K_FACTOR_ELITE=cls.ELO_K_FACTOR_ELITE,
            SCALE=cls.ELO_SCALE,
            DEFAULT_RATING=cls.ELO_DEFAULT_RATING,
            RATING_FLOOR=cls.ELO_RATING_FLOOR,
            RATING_CEILING=cls.ELO_RATING_CEILING,
            SCORE_MARGIN=cls.ELO_SCORE_MARGIN,
            NEW_PLAYER_THRESHOLD=cls.ELO_NEW_PLAYER_THRESHOLD,
            ESTABLISHED_THRESHOLD=cls.ELO_ESTABLISHED_THRESHOLD,
            ELITE_RATING_THRESHOLD=cls.ELO_ELITE_RATING_THRESHOLD,
            ELO_ENABLED=cls.ELO_ENABLED,
        )

    @classmethod
    def validate(cls):
        """Validate configuration parameters."""
        assert cls.Z_SCORE > 0.0, "Z_SCORE must be positive"
        assert cls.MIN_IMPROVEMENT >= 0.0, "MIN_IMPROVEMENT must be non-negative"
        assert cls.MAX_IMPROVEMENT >= cls.MIN_IMPROVEMENT, "MAX_IMPROVEMENT must be >= MIN_IMPROVEMENT"
        assert cls.SCORE_PRECISION >= 0, "SCORE_PRECISION must be non-negative"
        assert cls.SUBSET_WEIGHT_BASE > 0, "SUBSET_WEIGHT_BASE must be positive"
        assert cls.SUBSET_WEIGHT_EXPONENT >= 2, "SUBSET_WEIGHT_EXPONENT must be >= 2"
        assert 0.0 <= cls.DECAY_FACTOR <= 1.0, "DECAY_FACTOR must be in [0, 1]"
        assert 0.0 <= cls.MIN_WEIGHT_THRESHOLD <= 1.0, "MIN_WEIGHT_THRESHOLD must be in [0, 1]"
        assert 0.0 <= cls.MIN_COMPLETENESS <= 1.0, "MIN_COMPLETENESS must be in [0, 1]"
        # ELO validation
        assert cls.ELO_K_FACTOR > 0, "ELO_K_FACTOR must be positive"
        assert cls.ELO_SCALE > 0, "ELO_SCALE must be positive"
        assert cls.ELO_DEFAULT_RATING > 0, "ELO_DEFAULT_RATING must be positive"
        assert cls.ELO_RATING_FLOOR < cls.ELO_RATING_CEILING, "ELO_RATING_FLOOR must be < CEILING"


# Validate configuration on import
ScorerConfig.validate()