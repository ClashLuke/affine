"""
Local Weight Setter

Handles weight processing for local/shadow validation testing.

This is a SHADOW validator component - it uses real chain data and real miners,
but NEVER sets weights on the actual network. It only logs/stores what weights
WOULD be set if this were the real validator.

Modes:
1. Mock mode (default): Log and store what weights would be set
2. Dry-run mode: Only validate the weight calculation logic (no storage)

Usage:
    # Mock mode (default) - logs and stores weight history
    setter = LocalWeightSetter(mode="mock")
    result = await setter.process_and_set_weights(weights, burn_percentage=0.0)

    # Dry-run mode - only validates calculations
    setter = LocalWeightSetter(mode="dry-run")
    result = await setter.process_and_set_weights(weights, burn_percentage=0.0)

Note: This component intentionally does NOT support setting weights on the
real network. Use the production WeightSetter for that.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np

from affine.core.setup import logger


@dataclass
class WeightSetResult:
    """Result of a weight setting operation."""

    timestamp: int
    mode: str  # "mock", "testnet", "dry-run"
    uids: List[int]
    weights: List[float]
    burn_percentage: float
    success: bool
    error: Optional[str] = None

    # Testnet-specific fields
    block_number: Optional[int] = None
    tx_hash: Optional[str] = None

    # Mock-specific fields
    stored_in_history: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "mode": self.mode,
            "uids": self.uids,
            "weights": self.weights,
            "burn_percentage": self.burn_percentage,
            "success": self.success,
            "error": self.error,
            "block_number": self.block_number,
            "tx_hash": self.tx_hash,
            "stored_in_history": self.stored_in_history,
        }


@dataclass
class WeightHistoryEntry:
    """Entry in the weight history."""

    timestamp: int
    cycle: int
    uids: List[int]
    weights: List[float]
    burn_percentage: float
    mode: str
    block_number: Optional[int] = None


class LocalWeightSetter:
    """
    Shadow weight setter for local validation testing.

    This component processes weights exactly as the production validator would,
    but NEVER sets them on the real network. It uses real chain data (miners
    from mainnet) but only logs/stores what weights would be set.

    Modes:
    - mock: Log weights and store in history (default)
    - dry-run: Only validate calculations, don't store
    """

    VALID_MODES = ("mock", "dry-run")

    def __init__(
        self,
        mode: str = "mock",
        weight_history_callback: Optional[callable] = None,
    ):
        """
        Initialize the local weight setter.

        Args:
            mode: Operating mode - "mock" or "dry-run"
            weight_history_callback: Optional callback to store weight history
        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {self.VALID_MODES}")

        self.mode = mode
        self._weight_history_callback = weight_history_callback

        # Internal weight history
        self._weight_history: List[WeightHistoryEntry] = []
        self._cycle_counter = 0

    def process_weights(
        self,
        raw_weights: Dict[int, float],
        burn_percentage: float = 0.0,
    ) -> tuple[List[int], List[float]]:
        """
        Process and normalize weights, applying burn if specified.

        Args:
            raw_weights: Dict mapping UID to raw weight value
            burn_percentage: Fraction (0-1) to burn to UID 0

        Returns:
            Tuple of (uids, normalized_weights)
        """
        if not raw_weights:
            return [], []

        # Extract UIDs and weights
        uids = []
        weights = []

        for uid, weight in raw_weights.items():
            try:
                uid_int = int(uid)
                weight_float = float(weight)
                if uid_int >= 0 and weight_float > 0:
                    uids.append(uid_int)
                    weights.append(weight_float)
            except (ValueError, TypeError):
                continue

        if not uids:
            return [], []

        # Normalize to sum = 1.0
        weights_array = np.array(weights, dtype=np.float64)
        weights_array = weights_array / weights_array.sum()

        # Apply burn: scale all by (1 - burn%), then UID 0 += burn%
        if burn_percentage > 0 and burn_percentage <= 1.0:
            weights_array *= (1.0 - burn_percentage)

            if 0 in uids:
                idx = uids.index(0)
                weights_array[idx] += burn_percentage
            else:
                uids = [0] + uids
                weights_array = np.concatenate([[burn_percentage], weights_array])

        return uids, weights_array.tolist()

    async def process_and_set_weights(
        self,
        weights: Dict[int, float],
        burn_percentage: float = 0.0,
    ) -> WeightSetResult:
        """
        Process weights and log/store what would be set.

        This method processes weights exactly as the production validator would,
        but NEVER sets them on the real network.

        Args:
            weights: Dict mapping UID to weight value
            burn_percentage: Fraction (0-1) to burn to UID 0

        Returns:
            WeightSetResult with operation details
        """
        timestamp = int(time.time() * 1000)

        # Process and normalize weights
        uids, normalized_weights = self.process_weights(weights, burn_percentage)

        if not uids:
            return WeightSetResult(
                timestamp=timestamp,
                mode=self.mode,
                uids=[],
                weights=[],
                burn_percentage=burn_percentage,
                success=False,
                error="No valid weights to set",
            )

        logger.info(f"Processing weights for {len(uids)} miners (mode={self.mode})")
        logger.info(f"  Burn percentage: {burn_percentage:.1%}")

        # Log weight distribution
        for uid, weight in zip(uids, normalized_weights):
            if weight > 0.001:  # Only log significant weights
                logger.info(f"  UID {uid:3d}: {weight:.6f}")

        # Validate sum
        weight_sum = sum(normalized_weights)
        if abs(weight_sum - 1.0) > 0.0001:
            logger.warning(f"  Weight sum: {weight_sum:.6f} (should be 1.0)")

        # Execute based on mode
        if self.mode == "dry-run":
            return await self._dry_run(timestamp, uids, normalized_weights, burn_percentage)
        else:  # mock mode
            return await self._mock_set(timestamp, uids, normalized_weights, burn_percentage)

    async def _dry_run(
        self,
        timestamp: int,
        uids: List[int],
        weights: List[float],
        burn_percentage: float,
    ) -> WeightSetResult:
        """Dry run - validate calculations only."""
        logger.info("DRY RUN: Validating weight calculations only")

        # Validate
        weight_sum = sum(weights)
        valid = abs(weight_sum - 1.0) < 0.0001

        if valid:
            logger.info("DRY RUN: Weights validated successfully")
        else:
            logger.error(f"DRY RUN: Weight validation failed (sum={weight_sum})")

        return WeightSetResult(
            timestamp=timestamp,
            mode="dry-run",
            uids=uids,
            weights=weights,
            burn_percentage=burn_percentage,
            success=valid,
            error=None if valid else f"Weight sum validation failed: {weight_sum}",
        )

    async def _mock_set(
        self,
        timestamp: int,
        uids: List[int],
        weights: List[float],
        burn_percentage: float,
    ) -> WeightSetResult:
        """Mock weight setting - log and store in history."""
        logger.info("MOCK: Simulating weight setting...")

        self._cycle_counter += 1

        # Store in history
        entry = WeightHistoryEntry(
            timestamp=timestamp,
            cycle=self._cycle_counter,
            uids=uids,
            weights=weights,
            burn_percentage=burn_percentage,
            mode="mock",
        )
        self._weight_history.append(entry)

        # Call callback if provided
        if self._weight_history_callback:
            try:
                await self._weight_history_callback(entry)
            except Exception as e:
                logger.warning(f"Weight history callback error: {e}")

        logger.info(f"MOCK: Weights 'set' successfully (cycle {self._cycle_counter})")

        return WeightSetResult(
            timestamp=timestamp,
            mode="mock",
            uids=uids,
            weights=weights,
            burn_percentage=burn_percentage,
            success=True,
            stored_in_history=True,
        )

    def get_weight_history(self) -> List[WeightHistoryEntry]:
        """Get the weight setting history."""
        return self._weight_history.copy()

    def get_latest_weights(self) -> Optional[WeightHistoryEntry]:
        """Get the most recent weight setting."""
        if self._weight_history:
            return self._weight_history[-1]
        return None

    def clear_history(self):
        """Clear the weight history."""
        self._weight_history.clear()
        self._cycle_counter = 0


def create_local_weight_setter(mode: str = "mock") -> LocalWeightSetter:
    """
    Factory function to create a LocalWeightSetter.

    Args:
        mode: Operating mode - "mock" or "dry-run"

    Returns:
        Configured LocalWeightSetter
    """
    return LocalWeightSetter(mode=mode)
