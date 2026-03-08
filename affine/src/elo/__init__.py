"""
ELO Rating System Module

Provides ELO-based rating calculations for the affine evaluation platform.
Supports both pairwise score comparison (backward compatibility) and
multi-party direct competition games.

New features:
- Paired match support with information-weighted updates
- Bradley-Terry MLE with first-mover advantage estimation
- Bayesian inference with posterior distributions (requires NumPyro)
- Bootstrap confidence intervals for ratings
- Monte Carlo uncertainty propagation to weights
"""

from .config import EloConfig, DEFAULT_ELO_CONFIG
from .calculator import EloCalculator
from .models import (
    EloRating,
    MatchResult,
    MatchParticipant,
    MatchOutcome,
    MatchType,
    PairedOutcome,
    PairedMatchResult,
    SampleScore,
)
from .match_engine import MatchEngine, PairwiseMatchGenerator, EloSystem, HybridEloSystem
from .paired_bradley_terry import (
    PairedBradleyTerryModel,
    BradleyTerryResult,
    fit_bradley_terry_from_matches,
    fit_bradley_terry_from_pairs,
)
from .bootstrap import (
    PairedBootstrapBradleyTerry,
    BootstrapResult,
    bootstrap_confidence_intervals,
    bootstrap_from_pairs,
)
from .uncertainty import (
    MonteCarloWeightCalculator,
    WeightUncertaintyResult,
    compute_weight_uncertainty,
)

# Bayesian module requires NumPyro - import conditionally
_HAS_BAYESIAN = False
try:
    from .bayesian import HAS_NUMPYRO
    if HAS_NUMPYRO:
        from .bayesian import (
            BayesianBradleyTerry,
            BayesianResult,
            fit_bayesian_bradley_terry,
        )
        _HAS_BAYESIAN = True
except ImportError:
    pass

__all__ = [
    # Config
    "EloConfig",
    "DEFAULT_ELO_CONFIG",
    # Calculator
    "EloCalculator",
    # Models
    "EloRating",
    "MatchResult",
    "MatchParticipant",
    "MatchOutcome",
    "MatchType",
    "PairedOutcome",
    "PairedMatchResult",
    "SampleScore",
    # Match engine
    "MatchEngine",
    "PairwiseMatchGenerator",
    "EloSystem",
    "HybridEloSystem",  # backwards compatibility alias
    # Bradley-Terry MLE
    "PairedBradleyTerryModel",
    "BradleyTerryResult",
    "fit_bradley_terry_from_matches",
    "fit_bradley_terry_from_pairs",
    # Bootstrap
    "PairedBootstrapBradleyTerry",
    "BootstrapResult",
    "bootstrap_confidence_intervals",
    "bootstrap_from_pairs",
    # Uncertainty
    "MonteCarloWeightCalculator",
    "WeightUncertaintyResult",
    "compute_weight_uncertainty",
]

# Add Bayesian exports if available
if _HAS_BAYESIAN:
    __all__.extend([
        "BayesianBradleyTerry",
        "BayesianResult",
        "fit_bayesian_bradley_terry",
    ])
