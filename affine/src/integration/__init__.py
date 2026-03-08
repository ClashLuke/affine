"""
Integration Testing Module

Provides tools for end-to-end testing of the ELO scoring system
without affecting production weights.

Components:
- ShadowValidator: Runs parallel validation without setting weights
- EloScorer: Drop-in replacement for standard Scorer using ELO
- ReplayValidator: Validates ELO using historical scoring data
- EloValidatorEngine: Real validator for head-to-head game ELO
- LocalWeightSetter: Mock/testnet weight setter for local testing
- LocalValidatorEngine: Full pipeline orchestrator for local validation
"""

from .shadow_validator import ShadowValidator, ValidationReport as ShadowValidationReport, EloState
from .elo_scorer import EloScorer, create_elo_scorer
from .replay_validator import ReplayValidator, ReplayResult, ReplayRound
from .elo_validator import (
    EloValidatorEngine,
    ValidationReport,
    WeightResult,
    TournamentResult,
    create_elo_validator,
    # Local game validator
    LocalGameValidator,
    MinerInfo,
    # Runtime balancing
    MinerRuntimeStats,
    SamplerConfig,
    RuntimeBalancedSampler,
)
from .local_weight_setter import (
    LocalWeightSetter,
    WeightSetResult,
    WeightHistoryEntry,
    create_local_weight_setter,
)
from .local_validator_engine import (
    LocalValidatorEngine,
    CycleResult,
    FullValidationReport,
)

__all__ = [
    # Shadow validation
    "ShadowValidator",
    "ShadowValidationReport",
    "EloState",
    # ELO scorer
    "EloScorer",
    "create_elo_scorer",
    # Replay validation
    "ReplayValidator",
    "ReplayResult",
    "ReplayRound",
    # ELO validator (head-to-head games)
    "EloValidatorEngine",
    "ValidationReport",
    "WeightResult",
    "TournamentResult",
    "create_elo_validator",
    # Local game validator (runs real games via Chutes)
    "LocalGameValidator",
    "MinerInfo",
    # Runtime balancing
    "MinerRuntimeStats",
    "SamplerConfig",
    "RuntimeBalancedSampler",
    # Local weight setter (mock/testnet weight setting)
    "LocalWeightSetter",
    "WeightSetResult",
    "WeightHistoryEntry",
    "create_local_weight_setter",
    # Local validator engine (full pipeline orchestrator)
    "LocalValidatorEngine",
    "CycleResult",
    "FullValidationReport",
]
