"""
Local Validator Engine (Shadow Validator)

Orchestrates the complete local/shadow validation pipeline:
1. Fetch miners from REAL mainnet metagraph (SN120)
2. Run REAL games via Chutes API against real miners
3. Calculate ELO ratings locally
4. Generate weights locally
5. Log what weights WOULD be set (never actually sets on chain)
6. Report results

This is a SHADOW validator - it mirrors production behavior using real chain
data and real miners, but stores everything locally and never writes to the
real network.

Usage:
    engine = LocalValidatorEngine(
        miners=[...],  # Real miners from metagraph
        game_types=["tictactoe"],
    )
    await engine.run_forever()
    report = engine.get_report()
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


from affine.core.setup import logger
from affine.src.elo.config import EloConfig, DEFAULT_ELO_CONFIG
from .local_weight_setter import LocalWeightSetter, WeightSetResult, WeightHistoryEntry
from .elo_validator import (
    LocalGameValidator, MinerInfo, WeightResult,
    calculate_weights_from_ratings,
)


@dataclass
class CycleResult:
    """Result of a single validation cycle."""

    cycle_number: int
    timestamp: int
    games_played: int
    games_completed: int
    games_errored: int

    # ELO results
    ratings: Dict[str, Dict[str, float]]  # {hotkey: {env: rating}}
    weight_results: List[WeightResult]

    # Fields with defaults must come after fields without defaults
    rate_limited: bool = False  # Whether this cycle was cut short by rate limiting
    weight_set_result: Optional[WeightSetResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cycle_number": self.cycle_number,
            "timestamp": self.timestamp,
            "games_played": self.games_played,
            "games_completed": self.games_completed,
            "games_errored": self.games_errored,
            "rate_limited": self.rate_limited,
            "ratings": self.ratings,
            "weight_results": [w.to_dict() for w in self.weight_results],
            "weight_set_result": self.weight_set_result.to_dict() if self.weight_set_result else None,
        }


@dataclass
class FullValidationReport:
    """Complete validation report across all cycles."""

    start_timestamp: int
    end_timestamp: int
    total_cycles: int
    total_games: int
    total_games_completed: int
    total_games_errored: int

    # Configuration
    miners: List[MinerInfo]
    game_types: List[str]
    elo_config: Dict[str, Any]
    weight_mode: str

    # Cycle results
    cycle_results: List[CycleResult] = field(default_factory=list)

    # Final state
    final_ratings: Dict[str, Dict[str, float]] = field(default_factory=dict)
    final_weights: List[WeightResult] = field(default_factory=list)

    # Weight history
    weight_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "total_cycles": self.total_cycles,
            "total_games": self.total_games,
            "total_games_completed": self.total_games_completed,
            "total_games_errored": self.total_games_errored,
            "miners": [{"hotkey": m.hotkey, "model": m.model, "uid": m.uid} for m in self.miners],
            "game_types": self.game_types,
            "elo_config": self.elo_config,
            "weight_mode": self.weight_mode,
            "cycle_results": [c.to_dict() for c in self.cycle_results],
            "final_ratings": self.final_ratings,
            "final_weights": [w.to_dict() for w in self.final_weights],
            "weight_history": self.weight_history,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def print_summary(self):
        """Print a summary of the validation run."""
        print("=" * 70)
        print("LOCAL ELO VALIDATOR - FULL PIPELINE REPORT")
        print("=" * 70)
        print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_timestamp))}")
        print(f"End:   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.end_timestamp))}")
        print(f"Duration: {self.end_timestamp - self.start_timestamp:.1f} seconds")
        print()
        print(f"Miners: {len(self.miners)}")
        print(f"Game Types: {', '.join(self.game_types)}")
        print(f"Weight Mode: {self.weight_mode}")
        print()
        print(f"Total Cycles: {self.total_cycles}")
        print(f"Total Games: {self.total_games}")
        print(f"  Completed: {self.total_games_completed}")
        print(f"  Errored: {self.total_games_errored}")
        print()

        if self.final_weights:
            print("FINAL WEIGHTS:")
            print("-" * 70)
            print(f"{'Rank':<6}{'Hotkey':<20}{'Weight':<12}{'Avg ELO':<10}{'Matches':<10}{'Win Rate':<10}")
            print("-" * 70)

            for i, w in enumerate(self.final_weights[:20], 1):
                hotkey_short = w.miner_hotkey[:16] + "..." if len(w.miner_hotkey) > 16 else w.miner_hotkey
                print(
                    f"{i:<6}"
                    f"{hotkey_short:<20}"
                    f"{w.normalized_weight:<12.6f}"
                    f"{w.avg_elo:<10.1f}"
                    f"{w.total_matches:<10}"
                    f"{w.win_rate * 100:<9.1f}%"
                )
            print("-" * 70)

        if self.weight_history:
            print()
            print("WEIGHT SETTING HISTORY:")
            print("-" * 70)
            for entry in self.weight_history[-5:]:  # Last 5
                ts = time.strftime('%H:%M:%S', time.localtime(entry.get("timestamp", 0) / 1000))
                mode = entry.get("mode", "?")
                n_uids = len(entry.get("uids", []))
                success = entry.get("success", False)
                block = entry.get("block_number", "-")
                print(f"  {ts} | Mode: {mode} | UIDs: {n_uids} | Success: {success} | Block: {block}")
            print("-" * 70)

        print("=" * 70)


class LocalValidatorEngine:
    """
    Shadow validator that runs the full pipeline locally.

    This is a SHADOW validator - it uses real chain data (miners from mainnet)
    and runs real games via Chutes API, but stores everything locally and
    NEVER sets weights on the real network.

    Integrates:
    - LocalGameValidator for running games and ELO calculation
    - LocalWeightSetter for weight processing (mock only)
    - LocalDynamoDBClient for persistence (optional)
    """

    def __init__(
        self,
        miners: List[MinerInfo],
        game_types: Optional[List[str]] = None,
        elo_config: Optional[EloConfig] = None,
        burn_percentage: float = 0.0,
        min_matches: int = 5,
        # Persistence
        db_client: Optional[Any] = None,
        # Game options
        chutes_api_key: Optional[str] = None,
        timeout_per_move: int = 1800,
        concurrent_games: int = 8,
        on_game_complete: Optional[callable] = None,
        # Runtime balancing
        enable_runtime_balancing: bool = True,
    ):
        """
        Initialize the shadow validator engine.

        Args:
            miners: List of miners to evaluate (from real mainnet metagraph)
            game_types: Game types to run (default: ["tictactoe"])
            elo_config: ELO configuration
            burn_percentage: Fraction (0-1) to burn to UID 0
            min_matches: Minimum matches for weight calculation
            db_client: Optional LocalDynamoDBClient for persistence
            chutes_api_key: Chutes API key
            timeout_per_move: Timeout per move in seconds
            concurrent_games: Number of concurrent double-game pairs (default: 8)
            on_game_complete: Optional callback(match_record) called after each game
            enable_runtime_balancing: Enable runtime-balanced sampling (default: True)
        """
        self.miners = miners
        self.game_types = game_types or ["tictactoe"]
        self.elo_config = elo_config or DEFAULT_ELO_CONFIG
        self.weight_mode = "mock"  # Always mock - we never set real weights
        self.burn_percentage = burn_percentage
        self.min_matches = min_matches
        self.db_client = db_client
        self.concurrent_games = concurrent_games
        self.enable_runtime_balancing = enable_runtime_balancing

        # Initialize game validator
        self.game_validator = LocalGameValidator(
            miners=miners,
            game_types=self.game_types,
            elo_config=self.elo_config,
            chutes_api_key=chutes_api_key,
            timeout_per_move=timeout_per_move,
            on_game_complete=on_game_complete,
            enable_runtime_balancing=enable_runtime_balancing,
        )

        # Initialize weight setter (always mock mode)
        self.weight_setter = LocalWeightSetter(
            mode="mock",
            weight_history_callback=self._on_weight_set if db_client else None,
        )


    def update_miners(self, new_miners: List[MinerInfo]) -> int:
        """Update miner list with fresh data.

        Merges new miners with existing ones, preserving ELO ratings.
        Also updates slugs for existing miners if they changed.
        Returns number of new miners added.

        Args:
            new_miners: Fresh miner list from metagraph

        Returns:
            Number of newly added miners
        """
        existing_by_hotkey = {m.hotkey: m for m in self.miners}
        new_by_hotkey = {m.hotkey: m for m in new_miners}

        added = 0
        updated = 0
        for hotkey, new_miner in new_by_hotkey.items():
            if hotkey in existing_by_hotkey:
                existing = existing_by_hotkey[hotkey]
                if existing.chute_slug != new_miner.chute_slug:
                    existing.chute_slug = new_miner.chute_slug
                    self.game_validator._clear_miner_failures(hotkey)
                    updated += 1
            else:
                self.miners.append(new_miner)
                added += 1

        self.game_validator.miners = self.miners

        if added > 0 or updated > 0:
            logger.info(f"Miner update: +{added} new, ~{updated} slug changes (total: {len(self.miners)})")

        return added

    async def _on_weight_set(self, entry: WeightHistoryEntry):
        """Callback when weights are set - store in DB if available."""
        if self.db_client:
            try:
                await self.db_client.put_item(
                    TableName="WeightHistory",
                    Item={
                        "pk": {"S": f"WEIGHTS#{entry.cycle}"},
                        "sk": {"S": str(entry.timestamp)},
                        "cycle": {"N": str(entry.cycle)},
                        "timestamp": {"N": str(entry.timestamp)},
                        "uids": {"S": json.dumps(entry.uids)},
                        "weights": {"S": json.dumps(entry.weights)},
                        "burn_percentage": {"N": str(entry.burn_percentage)},
                        "mode": {"S": entry.mode},
                        "block_number": {"N": str(entry.block_number or 0)},
                    }
                )
                logger.debug(f"Stored weight history for cycle {entry.cycle}")
            except Exception as e:
                logger.warning(f"Failed to store weight history: {e}")

    def _calculate_weights(self) -> Tuple[List[WeightResult], Dict[int, float]]:
        """Calculate weights from current ELO ratings."""
        results = calculate_weights_from_ratings(
            self.game_validator._ratings, self.min_matches,
        )

        miners_by_hk = {m.hotkey: m for m in self.miners}
        uid_weights: Dict[int, float] = {}
        for r in results:
            miner = miners_by_hk.get(r.miner_hotkey)
            if miner:
                r.uid = miner.uid
            if r.uid is not None:
                uid_weights[r.uid] = r.normalized_weight

        return results, uid_weights

    def _get_current_ratings(self) -> Dict[str, Dict[str, float]]:
        """Get current ratings as {hotkey: {env: rating}}."""
        ratings_dict: Dict[str, Dict[str, float]] = {}

        for key, rating in self.game_validator._ratings.items():
            if rating.miner_hotkey not in ratings_dict:
                ratings_dict[rating.miner_hotkey] = {}
            ratings_dict[rating.miner_hotkey][rating.env] = float(rating.rating)

        return ratings_dict

    async def run_forever(self) -> None:
        """
        Run games forever with continuous workers.

        Games are saved after each completion via on_game_complete callback.
        Runs until interrupted (Ctrl+C) or rate limited.
        """
        logger.info("=" * 60)
        logger.info("LOCAL ELO VALIDATOR - CONTINUOUS MODE")
        logger.info("=" * 60)
        logger.info(f"Miners: {len(self.miners)}")
        logger.info(f"Game types: {self.game_types}")
        logger.info(f"Workers: {self.concurrent_games}")
        logger.info(f"Runtime balancing: {'enabled' if self.enable_runtime_balancing else 'disabled'}")
        logger.info("=" * 60)

        # Clear any invalid miners
        self.game_validator.clear_cycle_invalid_miners()

        # Run forever
        await self.game_validator.run_forever(
            concurrent_workers=self.concurrent_games,
        )

    def get_report(self) -> FullValidationReport:
        """Get the complete validation report."""
        now = int(time.time())
        final_weights, _ = self._calculate_weights()

        weight_history = []
        for entry in self.weight_setter.get_weight_history():
            weight_history.append({
                "timestamp": entry.timestamp,
                "cycle": entry.cycle,
                "uids": entry.uids,
                "weights": entry.weights,
                "burn_percentage": entry.burn_percentage,
                "mode": entry.mode,
                "block_number": entry.block_number,
            })

        return FullValidationReport(
            start_timestamp=now,
            end_timestamp=now,
            total_cycles=0,
            total_games=0,
            total_games_completed=0,
            total_games_errored=0,
            miners=self.miners,
            game_types=self.game_types,
            elo_config=self.elo_config.to_dict(),
            weight_mode=self.weight_mode,
            final_ratings=self._get_current_ratings(),
            final_weights=final_weights,
            weight_history=weight_history,
        )
