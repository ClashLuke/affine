"""
ELO Validator Engine

Processes real head-to-head game match results and generates weights based on ELO ratings.
Does NOT set weights on-chain - read-only analysis tool.

This replaces simulation-based ELO testing with real validator data.

Modes:
- API mode: Fetches existing ELO ratings from the API
- Local mode: Runs actual games against real miners via Chutes API, stores ELO locally
"""

import asyncio
import csv
import io
import json
import os
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal

from typing import Dict, List, Any, Optional, Tuple

import httpx

from affine.core.setup import logger
from affine.src.elo.calculator import EloCalculator
from affine.src.elo.config import EloConfig, DEFAULT_ELO_CONFIG
from affine.src.elo.models import EloRating
from affine.utils.api_client import get_chute_info


def extract_final_answer(response: str) -> Optional[str]:
    """
    Extract content from the last <FINAL_ANSWER>...</FINAL_ANSWER> block.

    Returns the content inside the last FINAL_ANSWER tag, or None if not found.
    This handles verbose model outputs that include reasoning before the answer.
    """
    if not response:
        return None

    # Find all FINAL_ANSWER blocks (case-insensitive)
    pattern = r"<FINAL_ANSWER>(.*?)</FINAL_ANSWER>"
    matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)

    if matches:
        # Return the last match, stripped
        return matches[-1].strip()

    return None


class RateLimitError(Exception):
    """Raised when API rate limit (429) is hit - should fail early."""
    pass


class PersistentHTTPError(Exception):
    """Raised when API has persistent HTTP error (402) - limited retries."""
    pass


class InfrastructureError(Exception):
    """Raised for infrastructure issues (403, 503, DNS) - game skipped, not penalized."""
    pass


# Alias for backwards compatibility
ColdChuteError = InfrastructureError


class APIFailureError(Exception):
    """Raised when API call fails (503, timeout) - should retry."""
    pass


class MoveFailureError(Exception):
    """Raised when a move fails (parse error, invalid move) - should retry."""
    pass


class MinerOnCooldownError(Exception):
    """Raised when a miner is on cooldown due to repeated failures."""
    pass


@dataclass
class TournamentResult:
    """Result of a tournament run, including rate limit status."""
    results: List[Dict[str, Any]]
    rate_limited: bool = False
    games_played: int = 0
    games_completed: int = 0
    games_errored: int = 0
    games_skipped: int = 0  # Cold chute - no penalty


@dataclass
class MinerInfo:
    """Information about a miner for game execution."""
    hotkey: str
    model: str
    chute_slug: str
    uid: Optional[int] = None
    model_revision: str = "v1"
    chute_id: Optional[str] = None  # For refreshing slug on 404

    use_llm_endpoint: bool = False  # If True, use llm.chutes.ai instead of slug
    custom_base_url: Optional[str] = None  # Override base_url for local testing

    @property
    def base_url(self) -> str:
        """Get API base URL for this miner."""
        if self.custom_base_url:
            return self.custom_base_url.rstrip("/")
        if self.use_llm_endpoint:
            return "https://llm.chutes.ai/v1"
        slug = self.chute_slug.replace('.chutes.ai', '').replace('https://', '')
        return f"https://{slug}.chutes.ai/v1"


@dataclass
class MinerRuntimeStats:
    """Runtime statistics for a miner, tracking total API call time."""
    miner_hotkey: str
    model_revision: str
    total_runtime_ms: int = 0  # Cumulative runtime from all API calls
    total_games: int = 0

    def add_runtime(self, latency_ms: int) -> None:
        """Add runtime from an API call."""
        self.total_runtime_ms += latency_ms


@dataclass
class SamplerConfig:
    """Configuration for RuntimeBalancedSampler."""
    pass  # Simplified - no config needed for min-runtime selection


class RuntimeBalancedSampler:
    """
    Runtime-balanced sampler with busy miner tracking.

    Tracks total cumulative runtime per miner. When a worker needs a pair,
    it picks the two lowest-runtime miners that aren't currently busy.
    """

    def __init__(self, config: Optional[SamplerConfig] = None):
        self.config = config or SamplerConfig()
        # {miner_key: MinerRuntimeStats} where key = hotkey#revision
        self._stats: Dict[str, MinerRuntimeStats] = {}
        # Set of hotkeys currently in a game
        self._busy_miners: set = set()
        self._lock = asyncio.Lock()

    def _get_key(self, miner: MinerInfo) -> str:
        """Get unique key for a miner."""
        return f"{miner.hotkey}#{miner.model_revision}"

    def _get_or_create_stats(self, miner: MinerInfo) -> MinerRuntimeStats:
        """Get existing stats or create new ones for a miner."""
        key = self._get_key(miner)
        if key not in self._stats:
            self._stats[key] = MinerRuntimeStats(
                miner_hotkey=miner.hotkey,
                model_revision=miner.model_revision,
            )
        return self._stats[key]

    def record_move_latency(
        self,
        miner: MinerInfo,
        latency_ms: int,
        game_type: Optional[str] = None,
    ) -> None:
        """Record runtime from an API call."""
        stats = self._get_or_create_stats(miner)
        stats.add_runtime(latency_ms)

    def record_game_complete(self, miner: MinerInfo) -> None:
        """Record that a miner completed a game."""
        stats = self._get_or_create_stats(miner)
        stats.total_games += 1

    def get_total_runtime(self, miner: MinerInfo) -> int:
        """Get total runtime for a miner."""
        return self._get_or_create_stats(miner).total_runtime_ms

    async def acquire_pair(
        self,
        miners: List[MinerInfo],
        valid_miner_filter: Optional[callable] = None,
    ) -> Optional[Tuple[MinerInfo, MinerInfo]]:
        """
        Acquire the two available miners with lowest total runtime.

        Marks them as busy. Returns None if fewer than 2 miners available.

        Args:
            miners: List of all miners
            valid_miner_filter: Optional filter function (miner -> bool)

        Returns:
            Tuple of (player_1, player_2) or None if not enough available
        """
        async with self._lock:
            # Filter valid miners that aren't busy
            available = []
            for m in miners:
                if m.hotkey in self._busy_miners:
                    continue
                if valid_miner_filter and not valid_miner_filter(m):
                    continue
                available.append(m)

            if len(available) < 2:
                return None

            # Sort by total runtime ascending, pick two lowest
            available.sort(key=lambda m: self.get_total_runtime(m))
            p1, p2 = available[0], available[1]

            # Mark as busy
            self._busy_miners.add(p1.hotkey)
            self._busy_miners.add(p2.hotkey)

            return (p1, p2)

    async def release_pair(self, p1: MinerInfo, p2: MinerInfo) -> None:
        """Release miners back to the available pool."""
        async with self._lock:
            self._busy_miners.discard(p1.hotkey)
            self._busy_miners.discard(p2.hotkey)

    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary of runtime statistics for logging."""
        if not self._stats:
            return {"num_miners": 0, "total_runtime_ms": 0}

        runtimes = [s.total_runtime_ms for s in self._stats.values()]
        return {
            "num_miners": len(self._stats),
            "min_runtime_ms": min(runtimes),
            "max_runtime_ms": max(runtimes),
            "total_runtime_ms": sum(runtimes),
            "total_games": sum(s.total_games for s in self._stats.values()),
            "busy_miners": len(self._busy_miners),
        }

    def log_runtime_distribution(self, miners: List[MinerInfo]) -> None:
        """Log the current runtime distribution for debugging."""
        if not miners:
            return

        runtime_info = []
        for m in miners:
            stats = self._get_or_create_stats(m)
            runtime_info.append({
                "hotkey": m.hotkey[:12],
                "runtime_s": stats.total_runtime_ms / 1000,
                "games": stats.total_games,
                "busy": m.hotkey in self._busy_miners,
            })

        # Sort by runtime ascending
        runtime_info.sort(key=lambda x: x["runtime_s"])

        total_runtime = sum(info["runtime_s"] for info in runtime_info)
        total_games = sum(info["games"] for info in runtime_info)

        logger.info(f"Runtime distribution: {total_runtime:.0f}s total, {total_games} games, {len(self._busy_miners)} busy")
        for info in runtime_info[:10]:
            busy_str = " [BUSY]" if info["busy"] else ""
            logger.info(f"  {info['hotkey']}... {info['runtime_s']:>7.1f}s  {info['games']:>3} games{busy_str}")
        if len(runtime_info) > 10:
            logger.info(f"  ... and {len(runtime_info) - 10} more")


@dataclass
class WeightResult:
    """Weight calculation result for a miner."""

    miner_hotkey: str
    model_revision: str
    uid: Optional[int] = None
    avg_elo: float = 0.0
    raw_weight: float = 0.0
    normalized_weight: float = 0.0
    env_ratings: Dict[str, float] = field(default_factory=dict)
    total_matches: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.total_matches == 0:
            return 0.0
        return self.wins / self.total_matches

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "miner_hotkey": self.miner_hotkey,
            "model_revision": self.model_revision,
            "uid": self.uid,
            "avg_elo": self.avg_elo,
            "raw_weight": self.raw_weight,
            "normalized_weight": self.normalized_weight,
            "env_ratings": self.env_ratings,
            "total_matches": self.total_matches,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.win_rate,
        }


def calculate_weights_from_ratings(
    ratings: Dict[str, 'EloRating'],
    min_matches: int = 5,
) -> List[WeightResult]:
    """Convert ELO ratings dict to normalized weights.

    Groups ratings by miner_id (hotkey#revision), computes arithmetic mean
    across played envs, applies max(0, avg - 1000), normalizes.

    Only includes envs where matches_played > 0 (no phantom 1500s).
    """
    miner_ratings: Dict[str, Dict[str, float]] = {}
    miner_stats: Dict[str, Dict[str, Any]] = {}

    for key, rating in ratings.items():
        miner_id = f"{rating.miner_hotkey}#{rating.model_revision}"
        if miner_id not in miner_ratings:
            miner_ratings[miner_id] = {}
            miner_stats[miner_id] = {
                "matches": 0, "wins": 0, "losses": 0, "draws": 0,
                "hotkey": rating.miner_hotkey, "revision": rating.model_revision,
            }
        if rating.matches_played > 0:
            miner_ratings[miner_id][rating.env] = float(rating.rating)
        miner_stats[miner_id]["matches"] += rating.matches_played
        miner_stats[miner_id]["wins"] += rating.wins
        miner_stats[miner_id]["losses"] += rating.losses
        miner_stats[miner_id]["draws"] += rating.draws

    results: List[WeightResult] = []
    total_raw = 0.0

    for miner_id, env_ratings in miner_ratings.items():
        stats = miner_stats[miner_id]
        if stats["matches"] < min_matches or not env_ratings:
            continue
        elo_values = list(env_ratings.values())
        avg_elo = sum(elo_values) / len(elo_values)
        raw = max(0.0, avg_elo - 1000)
        total_raw += raw
        results.append(WeightResult(
            miner_hotkey=stats["hotkey"], model_revision=stats["revision"],
            avg_elo=avg_elo, raw_weight=raw, normalized_weight=0.0,
            env_ratings=env_ratings, total_matches=stats["matches"],
            wins=stats["wins"], losses=stats["losses"], draws=stats["draws"],
        ))

    if total_raw > 0:
        for r in results:
            r.normalized_weight = r.raw_weight / total_raw
    results.sort(key=lambda r: r.normalized_weight, reverse=True)
    return results


@dataclass
class ValidationReport:
    """Complete validation report."""

    timestamp: int
    environments: List[str]
    total_miners: int
    total_matches: int
    weights: List[WeightResult]
    config: Dict[str, Any]
    replay_mode: bool = False
    time_range: Optional[Tuple[int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "environments": self.environments,
            "total_miners": self.total_miners,
            "total_matches": self.total_matches,
            "weights": [w.to_dict() for w in self.weights],
            "config": self.config,
            "replay_mode": self.replay_mode,
            "time_range": self.time_range,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_csv(self) -> str:
        """Convert to CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        header = [
            "rank",
            "miner_hotkey",
            "model_revision",
            "normalized_weight",
            "avg_elo",
            "total_matches",
            "wins",
            "losses",
            "draws",
            "win_rate",
        ]
        # Add environment columns
        if self.weights and self.weights[0].env_ratings:
            header.extend(sorted(self.weights[0].env_ratings.keys()))
        writer.writerow(header)

        # Data rows
        for i, w in enumerate(self.weights, 1):
            row = [
                i,
                w.miner_hotkey,
                w.model_revision,
                f"{w.normalized_weight:.6f}",
                f"{w.avg_elo:.2f}",
                w.total_matches,
                w.wins,
                w.losses,
                w.draws,
                f"{w.win_rate:.4f}",
            ]
            # Add environment ratings
            if w.env_ratings:
                for env in sorted(w.env_ratings.keys()):
                    row.append(f"{w.env_ratings.get(env, 0):.2f}")
            writer.writerow(row)

        return output.getvalue()

    def print_console(self, top_k: int = 32):
        """Print human-readable report to console."""
        print("=" * 80)
        print("ELO VALIDATOR REPORT")
        print("=" * 80)
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}")
        print(f"Environments: {', '.join(self.environments)}")
        print(f"Total Miners: {self.total_miners}")
        print(f"Total Matches: {self.total_matches}")
        if self.replay_mode:
            print("Mode: REPLAY (ELO recalculated from match history)")
        else:
            print("Mode: CURRENT (using live ELO ratings)")
        if self.time_range:
            start_str = time.strftime('%Y-%m-%d', time.localtime(self.time_range[0] / 1000))
            end_str = time.strftime('%Y-%m-%d', time.localtime(self.time_range[1] / 1000))
            print(f"Time Range: {start_str} to {end_str}")
        print()

        print(f"GENERATED WEIGHTS (Top {min(top_k, len(self.weights))}):")
        print("-" * 80)
        print(f"{'Rank':<6}{'Hotkey':<20}{'Weight':<12}{'Avg ELO':<10}{'Matches':<10}{'Win Rate':<10}")
        print("-" * 80)

        for i, w in enumerate(self.weights[:top_k], 1):
            hotkey_short = w.miner_hotkey[:16] + "..." if len(w.miner_hotkey) > 16 else w.miner_hotkey
            print(
                f"{i:<6}"
                f"{hotkey_short:<20}"
                f"{w.normalized_weight:<12.6f}"
                f"{w.avg_elo:<10.1f}"
                f"{w.total_matches:<10}"
                f"{w.win_rate * 100:<9.1f}%"
            )

        if len(self.environments) > 1:
            print()
            print("ELO BY ENVIRONMENT:")
            print("-" * 80)
            env_header = f"{'Rank':<6}{'Hotkey':<20}"
            for env in self.environments:
                env_short = env[:15] if len(env) > 15 else env
                env_header += f"{env_short:<15}"
            print(env_header)
            print("-" * 80)

            for i, w in enumerate(self.weights[:top_k], 1):
                hotkey_short = w.miner_hotkey[:16] + "..." if len(w.miner_hotkey) > 16 else w.miner_hotkey
                row = f"{i:<6}{hotkey_short:<20}"
                for env in self.environments:
                    rating = w.env_ratings.get(env, 0)
                    row += f"{rating:<15.1f}"
                print(row)

        print("=" * 80)


class EloValidatorEngine:
    """
    Validates ELO ratings and generates weights from real match data.

    This engine fetches actual game match results and ELO ratings from the API
    or uses in-memory mock data for local testing. It does NOT set any weights on-chain.

    Modes:
    - API mode: Fetches from API endpoint
    - Local mode: Uses in-memory ratings for testing without external dependencies
    """

    # Default API base URL
    DEFAULT_API_URL = "https://api.affine.io/api/v1"

    def __init__(
        self,
        environments: Optional[List[str]] = None,
        elo_config: Optional[EloConfig] = None,
        api_url: Optional[str] = None,
        local_mode: bool = False,
    ):
        """
        Initialize the validator engine.

        Args:
            environments: List of game environments to analyze.
                         Defaults to ["game:chess", "game:tictactoe"]
            elo_config: ELO configuration for replay mode calculations.
            api_url: Base URL for the API (default: https://api.affine.io/api/v1)
            local_mode: If True, use in-memory data instead of API calls
        """
        self.environments = environments or ["game:chess", "game:tictactoe"]
        self.elo_config = elo_config or DEFAULT_ELO_CONFIG
        self.api_url = api_url or self.DEFAULT_API_URL
        self.local_mode = local_mode

        # ELO calculator for replay mode
        self.calculator = EloCalculator(self.elo_config)

        # HTTP client (for API mode)
        self._client: Optional[httpx.AsyncClient] = None

        # In-memory storage (for local mode)
        self._local_ratings: Dict[str, EloRating] = {}
        self._local_matches: List[Dict[str, Any]] = []

    def add_rating(self, rating: EloRating):
        """Add a rating to local storage (for local_mode testing)."""
        key = f"{rating.miner_hotkey}#{rating.model_revision}#{rating.env}"
        self._local_ratings[key] = rating

    def add_match(self, match: Dict[str, Any]):
        """Add a match to local storage (for local_mode testing)."""
        self._local_matches.append(match)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=1800.0)
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get_current_ratings(self) -> Dict[str, EloRating]:
        """
        Fetch current ELO ratings (from API or local storage).

        Returns:
            Dict mapping "{hotkey}#{revision}#{env}" to EloRating
        """
        # Local mode - return in-memory ratings
        if self.local_mode:
            logger.info("Using local mode - returning in-memory ratings")
            # Filter to requested environments
            return {
                k: v for k, v in self._local_ratings.items()
                if v.env in self.environments
            }

        # API mode - fetch from endpoint
        all_ratings: Dict[str, EloRating] = {}
        client = await self._get_client()

        for env in self.environments:
            logger.info(f"Fetching leaderboard for environment: {env}")
            try:
                # Fetch leaderboard from API (up to 256 entries)
                url = f"{self.api_url}/elo/leaderboard/{env}?limit=256"
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()

                entries = data.get("entries", [])
                for entry in entries:
                    rating = EloRating(
                        miner_hotkey=entry["miner_hotkey"],
                        model_revision=entry["model_revision"],
                        env=env,
                        rating=Decimal(str(entry["rating"])),
                        peak_rating=Decimal(str(entry["rating"])),  # API doesn't return peak
                        matches_played=entry.get("matches_played", 0),
                        wins=entry.get("wins", 0),
                        losses=entry.get("losses", 0),
                        draws=entry.get("draws", 0),
                    )
                    key = f"{rating.miner_hotkey}#{rating.model_revision}#{rating.env}"
                    all_ratings[key] = rating

                logger.info(f"  Found {len(entries)} ratings (total players: {data.get('total_players', 0)})")

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error fetching {env} leaderboard: {e.response.status_code}")
            except Exception as e:
                logger.error(f"Error fetching {env} leaderboard: {e}")

        return all_ratings

    async def get_match_history(
        self,
        since_timestamp: Optional[int] = None,
        until_timestamp: Optional[int] = None,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        """
        Fetch match history.

        Note: The public API doesn't expose bulk match history queries.
        Replay mode requires direct database access.

        Args:
            since_timestamp: Start timestamp (milliseconds)
            until_timestamp: End timestamp (milliseconds)
            limit: Maximum matches to fetch

        Returns:
            List of match records (empty if API-only mode)
        """
        logger.warning("Match history fetch not available via public API. Replay mode requires database access.")
        return []

    async def replay_match_history(
        self,
        matches: List[Dict[str, Any]],
    ) -> Dict[str, EloRating]:
        """
        Recalculate ELO ratings by replaying match history.

        This allows testing different K-factors or validating ELO calculations.

        Args:
            matches: List of match records sorted by timestamp

        Returns:
            Dict mapping "{hotkey}#{revision}#{env}" to recalculated EloRating
        """
        ratings: Dict[str, EloRating] = {}

        def get_or_create_rating(hotkey: str, revision: str, env: str) -> EloRating:
            """Get existing rating or create new one."""
            key = f"{hotkey}#{revision}#{env}"
            if key not in ratings:
                ratings[key] = EloRating(
                    miner_hotkey=hotkey,
                    model_revision=revision,
                    env=env,
                    rating=self.elo_config.DEFAULT_RATING,
                    peak_rating=self.elo_config.DEFAULT_RATING,
                )
            return ratings[key]

        for match in matches:
            env = match.get("env")
            participants = match.get("participants", [])
            timestamp = match.get("timestamp", 0)

            if len(participants) != 2:
                # Skip non-head-to-head matches
                continue

            p1 = participants[0]
            p2 = participants[1]

            rating_1 = get_or_create_rating(
                p1["miner_hotkey"],
                p1.get("model_revision", "unknown"),
                env,
            )
            rating_2 = get_or_create_rating(
                p2["miner_hotkey"],
                p2.get("model_revision", "unknown"),
                env,
            )

            # Determine outcome
            outcome_1 = p1.get("outcome")
            if outcome_1 == "win":
                outcome = "a_wins"
            elif outcome_1 == "loss":
                outcome = "b_wins"
            else:
                outcome = "draw"

            # Calculate new ratings
            new_rating_1, new_rating_2, delta_1, delta_2 = self.calculator.update_ratings_head_to_head(
                rating_a=rating_1.rating,
                rating_b=rating_2.rating,
                matches_a=rating_1.matches_played,
                matches_b=rating_2.matches_played,
                outcome=outcome,
            )

            # Update rating 1
            rating_1.rating = new_rating_1
            rating_1.matches_played += 1
            rating_1.last_match_at = timestamp
            if new_rating_1 > rating_1.peak_rating:
                rating_1.peak_rating = new_rating_1
            if outcome == "a_wins":
                rating_1.wins += 1
            elif outcome == "b_wins":
                rating_1.losses += 1
            else:
                rating_1.draws += 1

            # Update rating 2
            rating_2.rating = new_rating_2
            rating_2.matches_played += 1
            rating_2.last_match_at = timestamp
            if new_rating_2 > rating_2.peak_rating:
                rating_2.peak_rating = new_rating_2
            if outcome == "b_wins":
                rating_2.wins += 1
            elif outcome == "a_wins":
                rating_2.losses += 1
            else:
                rating_2.draws += 1

        return ratings

    def calculate_weights(
        self,
        ratings: Dict[str, EloRating],
        min_matches: int = 5,
    ) -> List[WeightResult]:
        """Delegate to module-level calculate_weights_from_ratings."""
        return calculate_weights_from_ratings(ratings, min_matches)

    async def validate(
        self,
        replay: bool = False,
        since_timestamp: Optional[int] = None,
        until_timestamp: Optional[int] = None,
        min_matches: int = 5,
    ) -> ValidationReport:
        """
        Run complete validation and generate report.

        Args:
            replay: If True, recalculate ELO from match history (requires DB access)
            since_timestamp: Start timestamp for time range (milliseconds)
            until_timestamp: End timestamp for time range (milliseconds)
            min_matches: Minimum matches to include in weights

        Returns:
            ValidationReport with weights and metadata
        """
        logger.info("=" * 60)
        logger.info("Starting ELO Validation")
        logger.info(f"Environments: {self.environments}")
        logger.info(f"API URL: {self.api_url}")
        logger.info(f"Mode: {'REPLAY' if replay else 'CURRENT'}")
        logger.info("=" * 60)

        if replay:
            raise NotImplementedError(
                "Replay mode requires match history access which is not available "
                "via the public API. Use PairedBradleyTerryModel directly with "
                "historical match data for replay analysis."
            )

        logger.info("Fetching current ratings from API...")
        ratings = await self.get_current_ratings()
        total_matches = sum(r.matches_played for r in ratings.values()) // 2

        logger.info(f"Total ratings: {len(ratings)}")

        # Calculate weights
        logger.info("Calculating weights...")
        weights = self.calculate_weights(ratings, min_matches=min_matches)
        logger.info(f"Generated weights for {len(weights)} miners")

        # Build report
        report = ValidationReport(
            timestamp=int(time.time()),
            environments=self.environments,
            total_miners=len(weights),
            total_matches=total_matches,
            weights=weights,
            config=self.elo_config.to_dict(),
            replay_mode=replay,
            time_range=(since_timestamp, until_timestamp) if since_timestamp else None,
        )

        return report


def create_elo_validator(
    environments: Optional[List[str]] = None,
    elo_config: Optional[EloConfig] = None,
) -> EloValidatorEngine:
    """Factory function to create an EloValidatorEngine."""
    return EloValidatorEngine(
        environments=environments,
        elo_config=elo_config,
    )


class LocalGameValidator:
    """
    Local game validator that runs real games against miners via Chutes API.

    This class executes actual games (chess, tic-tac-toe) by calling miners
    through the Chutes API, calculates ELO from game outcomes, and stores
    everything in-memory (no AWS/DynamoDB dependencies).

    Usage:
        validator = LocalGameValidator(
            miners=[MinerInfo(hotkey="...", model="...", chute_slug="...")],
            game_types=["tictactoe", "chess"],
        )
        await validator.run_tournament(num_rounds=10)
        report = validator.generate_report()
    """

    def __init__(
        self,
        miners: List[MinerInfo],
        game_types: Optional[List[str]] = None,
        elo_config: Optional[EloConfig] = None,
        chutes_api_key: Optional[str] = None,
        timeout_per_move: int = 1800,
        on_game_complete: Optional[callable] = None,
        enable_runtime_balancing: bool = True,
    ):
        """
        Initialize the local game validator.

        Args:
            miners: List of miners to evaluate
            game_types: Game types to run ["tictactoe", "chess"]
            elo_config: ELO configuration
            chutes_api_key: Chutes API key (default: from CHUTES_API_KEY env)
            timeout_per_move: Timeout in seconds per move
            on_game_complete: Optional callback(match_record) called after each game
            enable_runtime_balancing: Enable runtime-balanced sampling (default: True)
        """
        self.miners = miners
        self.game_types = game_types or ["tictactoe"]
        self.elo_config = elo_config or DEFAULT_ELO_CONFIG
        self.timeout_per_move = timeout_per_move
        self.on_game_complete = on_game_complete

        # Get Chutes API key
        self.chutes_api_key = chutes_api_key or os.getenv("CHUTES_API_KEY")
        if not self.chutes_api_key:
            raise ValueError("CHUTES_API_KEY environment variable is required")

        # ELO calculator
        self.calculator = EloCalculator(self.elo_config)

        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None

        # In-memory ELO storage: {hotkey}#{revision}#{env} -> EloRating
        self._ratings: Dict[str, EloRating] = {}

        # Match history
        self._matches: List[Dict[str, Any]] = []

        # Statistics
        self._games_played = 0
        self._games_completed = 0
        self._games_errored = 0
        self._games_skipped = 0  # Cold chute - no penalty

        # Miner failure tracking: {hotkey: {"count": N, "last_failure": timestamp, "cooldown_until": timestamp}}
        self._miner_failures: Dict[str, Dict[str, Any]] = {}
        self._failure_threshold = 3  # Failures before cooldown
        self._cooldown_duration = 300  # 5 minutes cooldown

        # Miners to skip for current cycle (404 after refresh attempt)
        self._cycle_invalid_miners: set = set()

        # Lock for thread-safe updates to shared state
        self._state_lock = asyncio.Lock()

        # Runtime-balanced sampling (selects miners with lowest total runtime)
        self.enable_runtime_balancing = enable_runtime_balancing
        self._sampler = RuntimeBalancedSampler()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=1800.0)
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_or_create_rating(self, miner: MinerInfo, env: str) -> EloRating:
        """Get existing rating or create new one for a miner."""
        key = f"{miner.hotkey}#{miner.model_revision}#{env}"
        if key not in self._ratings:
            self._ratings[key] = EloRating(
                miner_hotkey=miner.hotkey,
                model_revision=miner.model_revision,
                env=env,
                rating=self.elo_config.DEFAULT_RATING,
                peak_rating=self.elo_config.DEFAULT_RATING,
            )
        return self._ratings[key]

    def _is_miner_on_cooldown(self, miner: MinerInfo) -> bool:
        """Check if a miner is currently on cooldown due to repeated failures."""
        if miner.hotkey not in self._miner_failures:
            return False
        failure_info = self._miner_failures[miner.hotkey]
        cooldown_until = failure_info.get("cooldown_until", 0)
        if time.time() < cooldown_until:
            return True
        # Cooldown expired, reset if needed
        if failure_info.get("count", 0) >= self._failure_threshold:
            failure_info["count"] = 0  # Reset count after cooldown
        return False

    def _record_miner_failure(self, miner: MinerInfo, error_type: str):
        """Record a miner failure and potentially put them on cooldown."""
        if miner.hotkey not in self._miner_failures:
            self._miner_failures[miner.hotkey] = {"count": 0, "last_failure": 0, "cooldown_until": 0}

        failure_info = self._miner_failures[miner.hotkey]
        failure_info["count"] = failure_info.get("count", 0) + 1
        failure_info["last_failure"] = time.time()
        failure_info["last_error"] = error_type

        if failure_info["count"] >= self._failure_threshold:
            failure_info["cooldown_until"] = time.time() + self._cooldown_duration
            logger.warning(f"  {miner.hotkey[:12]}... on cooldown for {self._cooldown_duration}s ({failure_info['count']} failures, last: {error_type})")

    def _record_miner_success(self, miner: MinerInfo):
        """Record a successful miner call, reducing their failure count."""
        if miner.hotkey in self._miner_failures:
            failure_info = self._miner_failures[miner.hotkey]
            # Reduce failure count on success (but not below 0)
            failure_info["count"] = max(0, failure_info.get("count", 0) - 1)

    def _clear_miner_failures(self, hotkey: str):
        """Clear all failure state for a miner (e.g., when their slug changes)."""
        if hotkey in self._miner_failures:
            del self._miner_failures[hotkey]
            logger.debug(f"Cleared failure state for {hotkey[:12]}... (slug changed)")

    def _mark_miner_invalid_for_cycle(self, miner: MinerInfo):
        """Mark a miner as invalid for the current cycle (will be retried next cycle)."""
        self._cycle_invalid_miners.add(miner.hotkey)
        logger.info(f"Miner {miner.hotkey[:12]}... marked invalid for this cycle (will retry next cycle)")

    def clear_cycle_invalid_miners(self):
        """Clear the set of invalid miners at the start of a new cycle."""
        if self._cycle_invalid_miners:
            logger.debug(f"Clearing {len(self._cycle_invalid_miners)} invalid miners for new cycle")
            self._cycle_invalid_miners.clear()

    def is_miner_valid_for_cycle(self, miner: MinerInfo) -> bool:
        """Check if a miner is valid for the current cycle."""
        if miner.hotkey in self._cycle_invalid_miners:
            return False
        if self._is_miner_on_cooldown(miner):
            return False
        return True

    async def _refresh_miner_slug(self, miner: MinerInfo) -> bool:
        """
        Refresh a miner's chute slug by re-fetching from the API.

        Returns True if slug was updated, False otherwise.
        """
        if not miner.chute_id:
            return False

        try:
            chute = await get_chute_info(miner.chute_id)
            if not chute:
                return False

            new_slug = chute.get("slug")
            is_hot = chute.get("hot", False)

            if not new_slug or not is_hot:
                logger.debug(f"Miner {miner.hotkey[:12]}... chute still cold/missing")
                return False

            if new_slug != miner.chute_slug:
                old_slug = miner.chute_slug
                miner.chute_slug = new_slug
                self._clear_miner_failures(miner.hotkey)
                logger.info(f"Refreshed slug for {miner.hotkey[:12]}...: {old_slug} -> {new_slug}")
                return True

            return False
        except Exception as e:
            logger.debug(f"Failed to refresh slug for {miner.hotkey[:12]}...: {e}")
            return False

    def replay_matches(self, matches: List[Dict[str, Any]]) -> int:
        """
        Replay historical matches to rebuild ELO ratings.

        Args:
            matches: List of match records from JSON

        Returns:
            Number of matches successfully replayed
        """
        if not matches:
            return 0

        replayed = 0
        errors = 0

        for match in matches:
            try:
                env = match.get("env", "game:unknown")
                participants = match.get("participants", [])

                if len(participants) != 2:
                    errors += 1
                    continue

                p1 = participants[0]
                p2 = participants[1]

                # Get or create ratings
                key1 = f"{p1['miner_hotkey']}#{p1.get('model_revision', 'v1')}#{env}"
                key2 = f"{p2['miner_hotkey']}#{p2.get('model_revision', 'v1')}#{env}"

                if key1 not in self._ratings:
                    self._ratings[key1] = EloRating(
                        miner_hotkey=p1['miner_hotkey'],
                        model_revision=p1.get('model_revision', 'v1'),
                        env=env,
                        rating=self.elo_config.DEFAULT_RATING,
                        peak_rating=self.elo_config.DEFAULT_RATING,
                    )
                if key2 not in self._ratings:
                    self._ratings[key2] = EloRating(
                        miner_hotkey=p2['miner_hotkey'],
                        model_revision=p2.get('model_revision', 'v1'),
                        env=env,
                        rating=self.elo_config.DEFAULT_RATING,
                        peak_rating=self.elo_config.DEFAULT_RATING,
                    )

                rating_1 = self._ratings[key1]
                rating_2 = self._ratings[key2]

                # Determine outcome
                p1_outcome = p1.get("outcome", "draw")
                if p1_outcome == "win":
                    outcome = "a_wins"
                elif p1_outcome == "loss":
                    outcome = "b_wins"
                else:
                    outcome = "draw"

                # Calculate new ratings
                new_rating_1, new_rating_2, _, _ = self.calculator.update_ratings_head_to_head(
                    rating_a=rating_1.rating,
                    rating_b=rating_2.rating,
                    matches_a=rating_1.matches_played,
                    matches_b=rating_2.matches_played,
                    outcome=outcome,
                )

                # Update rating 1
                rating_1.rating = new_rating_1
                rating_1.matches_played += 1
                rating_1.last_match_at = match.get("timestamp", 0)
                if new_rating_1 > rating_1.peak_rating:
                    rating_1.peak_rating = new_rating_1
                if outcome == "a_wins":
                    rating_1.wins += 1
                elif outcome == "b_wins":
                    rating_1.losses += 1
                else:
                    rating_1.draws += 1

                # Update rating 2
                rating_2.rating = new_rating_2
                rating_2.matches_played += 1
                rating_2.last_match_at = match.get("timestamp", 0)
                if new_rating_2 > rating_2.peak_rating:
                    rating_2.peak_rating = new_rating_2
                if outcome == "b_wins":
                    rating_2.wins += 1
                elif outcome == "a_wins":
                    rating_2.losses += 1
                else:
                    rating_2.draws += 1

                # Add to match history
                self._matches.append(match)
                replayed += 1

            except Exception as e:
                errors += 1
                logger.debug(f"Failed to replay match: {e}")

        logger.info(f"Replayed {replayed} matches ({errors} errors), {len(self._ratings)} ratings rebuilt")
        return replayed

    def _log_api_call(self, miner: MinerInfo, prompt: str, response: str, error: str = None):
        """Log API call to debug file."""
        try:
            with open("elo_api_debug.log", "a") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"MINER: {miner.hotkey[:16]}... ({miner.model})\n")
                f.write(f"PROMPT:\n{prompt}\n")
                f.write(f"---\n")
                if error:
                    f.write(f"ERROR: {error}\n")
                else:
                    f.write(f"RESPONSE:\n{response}\n")
                f.write(f"{'='*80}\n")
        except Exception:
            pass  # Don't let logging failures break the game

    async def _call_model(
        self,
        miner: MinerInfo,
        prompt: str,
        temperature: float = 0.0,
        _already_refreshed: bool = False,
    ) -> str:
        """
        Call a miner's model via Chutes API.

        Tries OpenAI-compatible chat completions first, then falls back to
        direct completions endpoint. On 404, attempts to refresh the chute
        slug and retry once before falling back.

        Args:
            miner: Miner information with chute_slug
            prompt: Prompt to send
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            _already_refreshed: Internal flag to prevent infinite refresh loops

        Returns:
            Model response text
        """
        client = await self._get_client()

        headers = {
            "Authorization": self.chutes_api_key,  # Chutes API doesn't use Bearer prefix
            "Content-Type": "application/json",
        }

        # Try chat completions endpoint first (OpenAI-compatible)
        chat_payload = {
            "model": miner.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        try:
            response = await client.post(
                f"{miner.base_url}/chat/completions",
                headers=headers,
                json=chat_payload,
                timeout=self.timeout_per_move,
            )
            response.raise_for_status()
            data = response.json()

            # Extract response text (check both content and reasoning_content)
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                # Try content first, then reasoning_content (for reasoning models like o1/o3)
                content = message.get("content") or message.get("reasoning_content") or ""
                if content:
                    self._log_api_call(miner, prompt, content)
                    return content

            self._log_api_call(miner, prompt, "", "Empty response")
            return ""

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 429:
                # Rate limit - fail early
                self._log_api_call(miner, prompt, "", f"Rate limit (429)")
                raise RateLimitError(f"Rate limit hit for {miner.hotkey[:12]}... - stopping to avoid quota waste")
            if status == 404:
                # Try to refresh slug and retry (once)
                if not _already_refreshed:
                    refreshed = await self._refresh_miner_slug(miner)
                    if refreshed:
                        return await self._call_model(miner, prompt, temperature, _already_refreshed=True)
                # Slug didn't change or still 404 - mark invalid for this cycle
                self._mark_miner_invalid_for_cycle(miner)
                self._log_api_call(miner, prompt, "", f"HTTP 404 (chute not found)")
                raise ColdChuteError(f"Chute not found for {miner.hotkey[:12]}...")
            if status in (403, 503):
                # 403 = no access to chute, 503 = cold chute - skip game, don't penalize
                reason = "no access" if status == 403 else "cold chute"
                self._log_api_call(miner, prompt, "", f"HTTP {status} ({reason})")
                raise ColdChuteError(f"{reason} for {miner.hotkey[:12]}...")
            if status in (402, 502, 504):
                # Persistent errors - payment required, gateway errors
                self._log_api_call(miner, prompt, "", f"HTTP {status}")
                raise PersistentHTTPError(f"HTTP {status} for {miner.hotkey[:12]}...")
            self._log_api_call(miner, prompt, "", f"HTTP {status}")
            raise
        except httpx.TimeoutException:
            self._log_api_call(miner, prompt, "", "Timeout")
            raise InfrastructureError(f"Timeout for {miner.hotkey[:12]}...")
        except httpx.ConnectError as e:
            # DNS failures, connection refused, etc.
            self._log_api_call(miner, prompt, "", f"Connect error: {e}")
            raise InfrastructureError(f"Connection error for {miner.hotkey[:12]}...: {e}")
        except OSError as e:
            # Network-level errors (DNS, socket errors)
            self._log_api_call(miner, prompt, "", f"Network error: {e}")
            raise InfrastructureError(f"Network error for {miner.hotkey[:12]}...: {e}")
        except Exception as e:
            error_str = str(e).lower()
            # Check for DNS/network errors in exception message
            if 'errno' in error_str or 'dns' in error_str or 'resolve' in error_str or 'temporary failure' in error_str:
                self._log_api_call(miner, prompt, "", f"Network error: {e}")
                raise InfrastructureError(f"Network error for {miner.hotkey[:12]}...: {e}")
            self._log_api_call(miner, prompt, "", str(e))
            logger.error(f"Error calling model for {miner.hotkey[:12]}...: {e}")
            raise

    async def _get_move_with_retry(
        self,
        player: MinerInfo,
        prompt: str,
        parse_fn: callable,
        max_retries: int = 8,
        game_type: Optional[str] = None,
    ) -> Tuple[Any, int]:
        """
        Get a move from a player with retry logic for API failures, parse errors, and invalid moves.

        This is a shared wrapper that handles all retryable errors uniformly across game types.
        Temperature increases linearly from 0.0 (first attempt) to 1.0 (final attempt).

        For persistent HTTP errors (503, 402), only 2 retries are attempted since temperature
        won't help with infrastructure issues.

        Args:
            player: The miner to call
            prompt: The prompt to send
            parse_fn: Function that parses response and returns (move, error_msg).
                      Should return (move, None) on success or (None, "error message") on failure.
            max_retries: Maximum retry attempts (default: 8)
            game_type: Optional game type for runtime tracking

        Returns:
            Tuple of (parsed_move, latency_ms)

        Raises:
            RateLimitError: If rate limited (stops tournament)
            MinerOnCooldownError: If miner is on cooldown
            APIFailureError: If all retries exhausted
        """
        # Check if miner is on cooldown
        if self._is_miner_on_cooldown(player):
            raise MinerOnCooldownError(f"{player.hotkey[:12]}... is on cooldown")

        last_error = None
        http_error_count = 0  # Track consecutive HTTP errors
        max_http_retries = 2  # Only retry HTTP errors twice

        for attempt in range(max_retries):
            # Temperature: 0.0 on first attempt, linearly increase to 1.0 on final attempt
            temperature = attempt / (max_retries - 1) if max_retries > 1 else 0.0

            try:
                start_time = time.time()
                response = await asyncio.wait_for(
                    self._call_model(player, prompt, temperature=temperature),
                    timeout=self.timeout_per_move,
                )
                latency_ms = int((time.time() - start_time) * 1000)

                # Try to parse the response
                move, error_msg = parse_fn(response)
                if move is not None:
                    self._record_miner_success(player)
                    # Record latency for runtime balancing
                    if self.enable_runtime_balancing:
                        self._sampler.record_move_latency(player, latency_ms, game_type)
                    return (move, latency_ms)

                # Parse/validation failed - retry with same prompt (temperature might help)
                last_error = MoveFailureError(error_msg or "Invalid move")
                http_error_count = 0  # Reset HTTP error count on successful API call
                if attempt < max_retries - 1:
                    response_preview = response[:50].replace('\n', ' ') if response else "(empty)"
                    logger.warning(f"  Retry {attempt + 1}/{max_retries} for {player.hotkey[:12]}... (t={temperature:.2f}) ({error_msg}) Response: '{response_preview}'")
                    await asyncio.sleep(0.3)  # Brief backoff

            except RateLimitError:
                self._record_miner_failure(player, "rate_limit")
                raise
            except (ColdChuteError, InfrastructureError):
                # Infrastructure issue (403, 503, DNS) - skip game, don't penalize
                raise
            except PersistentHTTPError as e:
                http_error_count += 1
                last_error = e
                if http_error_count >= max_http_retries:
                    # Too many HTTP errors - fail fast and record failure
                    self._record_miner_failure(player, "http_error")
                    logger.error(f"  Move failed after {http_error_count} HTTP errors for {player.hotkey[:12]}...: {e}")
                    raise APIFailureError(f"Persistent HTTP error after {http_error_count} attempts: {e}")
                if attempt < max_retries - 1:
                    logger.warning(f"  Retry {attempt + 1}/{max_retries} for {player.hotkey[:12]}... (t={temperature:.2f}) (HTTPError)")
                    await asyncio.sleep(1.0)  # Longer backoff for HTTP errors
            except MoveFailureError as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(f"  Retry {attempt + 1}/{max_retries} for {player.hotkey[:12]}... (t={temperature:.2f}) ({e})")
                    await asyncio.sleep(0.3)
            except Exception as e:
                # Unknown error - check if it looks like infrastructure (but NOT timeout - that's a failure)
                error_str = str(e).lower()
                if 'timeout' in error_str:
                    # Timeout is a failure - miner took too long to respond
                    self._record_miner_failure(player, "timeout")
                    raise APIFailureError(f"Timeout for {player.hotkey[:12]}...: {e}")
                if 'errno' in error_str or 'connect' in error_str:
                    raise InfrastructureError(f"Network error for {player.hotkey[:12]}...: {e}")
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(f"  Retry {attempt + 1}/{max_retries} for {player.hotkey[:12]}... (t={temperature:.2f}) ({type(e).__name__})")
                    await asyncio.sleep(0.5)

        # All retries exhausted (format errors)
        self._record_miner_failure(player, "format_error")
        logger.error(f"  Move failed after {max_retries} retries for {player.hotkey[:12]}...: {last_error}")
        raise APIFailureError(f"Move failed after {max_retries} retries: {last_error}")

    async def _execute_tictactoe(
        self,
        player_1: MinerInfo,
        player_2: MinerInfo,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Execute a tic-tac-toe game between two miners.

        Returns:
            Tuple of (outcome, move_history)
            outcome: "a_wins", "b_wins", or "draw"
        """
        # Initialize board (0 = empty, 1 = player 1 (X), 2 = player 2 (O))
        board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        move_history = []
        players = [player_1, player_2]
        current_player_idx = 0

        def format_board(b: List[List[int]]) -> str:
            symbols = {0: ".", 1: "X", 2: "O"}
            lines = ["    0   1   2", "  +---+---+---+"]
            for row_idx, row in enumerate(b):
                cells = " | ".join(symbols[c] for c in row)
                lines.append(f"{row_idx} | {cells} |")
                lines.append("  +---+---+---+")
            return "\n".join(lines)

        def check_winner(b: List[List[int]]) -> Optional[int]:
            # Rows
            for row in b:
                if row[0] != 0 and row[0] == row[1] == row[2]:
                    return row[0]
            # Columns
            for col in range(3):
                if b[0][col] != 0 and b[0][col] == b[1][col] == b[2][col]:
                    return b[0][col]
            # Diagonals
            if b[0][0] != 0 and b[0][0] == b[1][1] == b[2][2]:
                return b[0][0]
            if b[0][2] != 0 and b[0][2] == b[1][1] == b[2][0]:
                return b[0][2]
            return None

        def is_board_full(b: List[List[int]]) -> bool:
            return all(c != 0 for row in b for c in row)

        def get_empty_cells(b: List[List[int]]) -> List[str]:
            """Get list of empty cell coordinates."""
            empty = []
            for r in range(3):
                for c in range(3):
                    if b[r][c] == 0:
                        empty.append(f"{r} {c}")
            return empty

        for turn in range(9):
            player = players[current_player_idx]
            symbol = "X" if current_player_idx == 0 else "O"
            empty_cells = get_empty_cells(board)

            # Build prompt with explicit format requirement
            prompt = (
                f"Tic-Tac-Toe. You are {symbol}.\n"
                f"{format_board(board)}\n"
                f"Empty cells: {', '.join(empty_cells)}\n"
                f"Pick one empty cell. You MUST respond with ONLY this exact format:\n"
                f"<FINAL_ANSWER>row col</FINAL_ANSWER>\n"
                f"Example: <FINAL_ANSWER>1 2</FINAL_ANSWER>\n"
                f"Your move:"
            )

            # Create parser that validates against current board state - FINAL_ANSWER required
            def parse_tictactoe_move(response: str) -> Tuple[Optional[Tuple[int, int]], Optional[str]]:
                """Parse response and validate move. Returns (move, error_msg)."""
                answer = extract_final_answer(response)
                if answer is None:
                    return (None, "No <FINAL_ANSWER> block found")

                numbers = re.findall(r"\d+", answer)
                if len(numbers) < 2:
                    return (None, f"Need two numbers (row col) in FINAL_ANSWER: {answer}")

                row, col = int(numbers[0]), int(numbers[1])

                # Validate range
                if row < 0 or row > 2 or col < 0 or col > 2:
                    return (None, f"Position ({row},{col}) is out of bounds (must be 0-2)")

                # Validate cell is empty
                if board[row][col] != 0:
                    occupied_by = "X" if board[row][col] == 1 else "O"
                    return (None, f"Position ({row},{col}) is already occupied by {occupied_by}")

                return ((row, col), None)

            # Get move with retry - if fails, current player forfeits
            try:
                (row, col), latency_ms = await self._get_move_with_retry(
                    player=player,
                    prompt=prompt,
                    parse_fn=parse_tictactoe_move,
                    game_type="tictactoe",
                )
            except (ColdChuteError, InfrastructureError) as e:
                # Infrastructure issue - skip game, but track for repeated failures
                logger.info(f"    {player.hotkey[:12]}... infrastructure issue - skipping game")
                self._record_miner_failure(player, "infrastructure")
                move_history.append({
                    "turn": turn,
                    "player": current_player_idx,
                    "skipped": True,
                    "error": str(e),
                })
                return ("skipped", move_history)
            except (APIFailureError, MinerOnCooldownError) as e:
                # Current player couldn't make a valid move - they forfeit
                reason = "on cooldown" if isinstance(e, MinerOnCooldownError) else "failed to respond"
                logger.warning(f"    {player.hotkey[:12]}... forfeits ({reason})")
                winner = 1 - current_player_idx
                move_history.append({
                    "turn": turn,
                    "player": current_player_idx,
                    "forfeit": True,
                    "error": str(e),
                })
                return ("a_wins" if winner == 0 else "b_wins", move_history)

            # Apply move
            board[row][col] = current_player_idx + 1
            move_history.append({
                "turn": turn,
                "player": current_player_idx,
                "move": {"row": row, "col": col},
                "latency_ms": latency_ms,
            })

            # Check winner
            winner = check_winner(board)
            if winner is not None:
                return ("a_wins" if winner == 1 else "b_wins", move_history)

            # Check draw
            if is_board_full(board):
                return ("draw", move_history)

            # Switch player
            current_player_idx = 1 - current_player_idx

        # Max moves reached
        return ("draw", move_history)

    async def _execute_chess(
        self,
        player_1: MinerInfo,
        player_2: MinerInfo,
        max_moves: int = 100,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Execute a chess game between two miners.

        Returns:
            Tuple of (outcome, move_history)
            outcome: "a_wins", "b_wins", or "draw"
        """
        try:
            import chess
        except ImportError:
            logger.error("python-chess not installed, skipping chess game")
            return ("draw", [])

        board = chess.Board()
        move_history = []
        players = [player_1, player_2]
        current_player_idx = 0  # 0 = White

        for turn in range(max_moves):
            player = players[current_player_idx]
            color = "White" if current_player_idx == 0 else "Black"

            # Get ALL legal moves
            legal_moves = list(board.legal_moves)
            moves_str = ", ".join(m.uci() for m in legal_moves)

            # Build prompt with explicit format requirement
            prompt = (
                f"Chess. You are {color}.\n"
                f"{board}\n"
                f"Legal moves: {moves_str}\n"
                f"Pick a legal move. You MUST respond with ONLY this exact format:\n"
                f"<FINAL_ANSWER>move</FINAL_ANSWER>\n"
                f"Example: <FINAL_ANSWER>e2e4</FINAL_ANSWER>\n"
                f"Your move:"
            )

            # Create parser that validates against legal moves (accepts UCI or SAN)
            # Use default argument to capture board reference at definition time
            # FINAL_ANSWER is required - no fallback to raw response
            def parse_chess_move(response: str, _board=board) -> Tuple[Optional[str], Optional[str]]:
                """Parse response and validate move. Returns (move_uci, error_msg)."""
                answer = extract_final_answer(response)
                if answer is None:
                    return (None, "No <FINAL_ANSWER> block found")

                # Try UCI first (e2e4, g1f3)
                uci_pattern = r"[a-h][1-8][a-h][1-8][qrbn]?"
                uci_matches = re.findall(uci_pattern, answer.lower())

                if uci_matches:
                    move_uci = uci_matches[0]
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if _board.is_legal(move):
                            return (move_uci, None)
                        else:
                            return (None, f"Move {move_uci} is not legal (turn {_board.fullmove_number}, {'W' if _board.turn else 'B'})")
                    except ValueError as e:
                        return (None, f"Invalid UCI format: {move_uci}")

                # Try SAN (Nf3, e4, Bxc6, O-O, etc.)
                san_pattern = r"[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|O-O-O|O-O"
                san_matches = re.findall(san_pattern, answer)

                for san in san_matches:
                    try:
                        move = _board.parse_san(san)
                        if _board.is_legal(move):
                            return (move.uci(), None)
                    except ValueError:
                        continue

                return (None, f"Could not parse move from FINAL_ANSWER: {answer[:30]}")

            # Get move with retry - if fails, current player forfeits
            try:
                move_uci, latency_ms = await self._get_move_with_retry(
                    player=player,
                    prompt=prompt,
                    parse_fn=parse_chess_move,
                    game_type="chess",
                )
            except (ColdChuteError, InfrastructureError) as e:
                # Infrastructure issue - skip game, but track for repeated failures
                logger.info(f"    {player.hotkey[:12]}... infrastructure issue - skipping game")
                self._record_miner_failure(player, "infrastructure")
                move_history.append({
                    "turn": turn,
                    "player": current_player_idx,
                    "skipped": True,
                    "error": str(e),
                })
                return ("skipped", move_history)
            except (APIFailureError, MinerOnCooldownError) as e:
                # Current player couldn't make a valid move - they forfeit
                reason = "on cooldown" if isinstance(e, MinerOnCooldownError) else "failed to respond"
                logger.warning(f"    {player.hotkey[:12]}... forfeits ({reason})")
                winner = 1 - current_player_idx
                move_history.append({
                    "turn": turn,
                    "player": current_player_idx,
                    "forfeit": True,
                    "error": str(e),
                })
                return ("a_wins" if winner == 0 else "b_wins", move_history)

            # Apply move
            move = chess.Move.from_uci(move_uci)
            board.push(move)

            move_history.append({
                "turn": turn,
                "player": current_player_idx,
                "move": move_uci,
                "latency_ms": latency_ms,
            })

            # Check for game end
            if board.is_checkmate():
                return ("a_wins" if current_player_idx == 0 else "b_wins", move_history)
            if board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves():
                return ("draw", move_history)

            # Switch player
            current_player_idx = 1 - current_player_idx

        # Max moves reached
        return ("draw", move_history)

    async def _execute_connect4(
        self,
        player_1: MinerInfo,
        player_2: MinerInfo,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Execute a Connect 4 game between two miners.

        Connect 4 rules:
        - 6 rows x 7 columns board
        - Players drop pieces into columns (pieces fall to lowest empty row)
        - First to get 4 in a row (horizontal, vertical, or diagonal) wins
        - Draw if board is full with no winner

        Returns:
            Tuple of (outcome, move_history)
            outcome: "a_wins", "b_wins", or "draw"
        """
        # Initialize 6x7 board (0 = empty, 1 = player 1, 2 = player 2)
        # board[row][col] where row 0 is TOP, row 5 is BOTTOM
        ROWS, COLS = 6, 7
        board = [[0] * COLS for _ in range(ROWS)]
        move_history = []
        players = [player_1, player_2]
        current_player_idx = 0

        def format_board(b: List[List[int]]) -> str:
            """Format board with column numbers and visual grid."""
            symbols = {0: ".", 1: "X", 2: "O"}
            lines = ["  " + " ".join(str(c) for c in range(COLS))]
            lines.append("  " + "-" * (COLS * 2 - 1))
            for row in b:
                lines.append("  " + " ".join(symbols[c] for c in row))
            lines.append("  " + "-" * (COLS * 2 - 1))
            lines.append("  " + " ".join(str(c) for c in range(COLS)))
            return "\n".join(lines)

        def get_valid_columns(b: List[List[int]]) -> List[int]:
            """Return list of columns that aren't full."""
            return [col for col in range(COLS) if b[0][col] == 0]

        def drop_piece(b: List[List[int]], col: int, player: int) -> int:
            """Drop piece into column, return row where it landed. Returns -1 if column full."""
            for row in range(ROWS - 1, -1, -1):  # Start from bottom
                if b[row][col] == 0:
                    b[row][col] = player
                    return row
            return -1

        def check_winner(b: List[List[int]]) -> Optional[int]:
            """Check for 4 in a row. Returns player number (1 or 2) or None."""
            # Check horizontal
            for row in range(ROWS):
                for col in range(COLS - 3):
                    if b[row][col] != 0 and b[row][col] == b[row][col+1] == b[row][col+2] == b[row][col+3]:
                        return b[row][col]

            # Check vertical
            for row in range(ROWS - 3):
                for col in range(COLS):
                    if b[row][col] != 0 and b[row][col] == b[row+1][col] == b[row+2][col] == b[row+3][col]:
                        return b[row][col]

            # Check diagonal (down-right)
            for row in range(ROWS - 3):
                for col in range(COLS - 3):
                    if b[row][col] != 0 and b[row][col] == b[row+1][col+1] == b[row+2][col+2] == b[row+3][col+3]:
                        return b[row][col]

            # Check diagonal (down-left)
            for row in range(ROWS - 3):
                for col in range(3, COLS):
                    if b[row][col] != 0 and b[row][col] == b[row+1][col-1] == b[row+2][col-2] == b[row+3][col-3]:
                        return b[row][col]

            return None

        def is_board_full(b: List[List[int]]) -> bool:
            """Check if all columns are full."""
            return all(b[0][col] != 0 for col in range(COLS))

        # Maximum 42 moves (6x7 board)
        for turn in range(ROWS * COLS):
            player = players[current_player_idx]
            symbol = "X" if current_player_idx == 0 else "O"
            valid_cols = get_valid_columns(board)

            # Build prompt with explicit format requirement
            prompt = (
                f"Connect4. You are {symbol}. Get 4 in a row to win.\n"
                f"{format_board(board)}\n"
                f"Valid columns: {valid_cols}\n"
                f"Pick a column. You MUST respond with ONLY this exact format:\n"
                f"<FINAL_ANSWER>N</FINAL_ANSWER>\n"
                f"where N is the column number. Example: <FINAL_ANSWER>3</FINAL_ANSWER>\n"
                f"Your move:"
            )

            # Create parser that validates column choice - FINAL_ANSWER required
            def parse_connect4_move(response: str) -> Tuple[Optional[int], Optional[str]]:
                """Parse response and validate column. Returns (column, error_msg)."""
                answer = extract_final_answer(response)
                if answer is None:
                    return (None, "No <FINAL_ANSWER> block found")

                numbers = re.findall(r"\d+", answer)
                if not numbers:
                    return (None, f"No number in FINAL_ANSWER: {answer}")

                col = int(numbers[0])

                if col < 0 or col >= COLS:
                    return (None, f"Column {col} is out of bounds (must be 0-6)")

                if col not in valid_cols:
                    return (None, f"Column {col} is full, choose from: {valid_cols}")

                return (col, None)

            # Get move with retry - if fails, current player forfeits
            try:
                col, latency_ms = await self._get_move_with_retry(
                    player=player,
                    prompt=prompt,
                    parse_fn=parse_connect4_move,
                    game_type="connect4",
                )
            except (ColdChuteError, InfrastructureError) as e:
                # Infrastructure issue - skip game, but track for repeated failures
                logger.info(f"    {player.hotkey[:12]}... infrastructure issue - skipping game")
                self._record_miner_failure(player, "infrastructure")
                move_history.append({
                    "turn": turn,
                    "player": current_player_idx,
                    "skipped": True,
                    "error": str(e),
                })
                return ("skipped", move_history)
            except (APIFailureError, MinerOnCooldownError) as e:
                # Current player couldn't make a valid move - they forfeit
                reason = "on cooldown" if isinstance(e, MinerOnCooldownError) else "failed to respond"
                logger.warning(f"    {player.hotkey[:12]}... forfeits ({reason})")
                winner = 1 - current_player_idx
                move_history.append({
                    "turn": turn,
                    "player": current_player_idx,
                    "forfeit": True,
                    "error": str(e),
                })
                return ("a_wins" if winner == 0 else "b_wins", move_history)

            # Apply move
            row = drop_piece(board, col, current_player_idx + 1)
            move_history.append({
                "turn": turn,
                "player": current_player_idx,
                "column": col,
                "row": row,
                "latency_ms": latency_ms,
            })

            # Check winner
            winner = check_winner(board)
            if winner is not None:
                return ("a_wins" if winner == 1 else "b_wins", move_history)

            # Check draw
            if is_board_full(board):
                return ("draw", move_history)

            # Switch player
            current_player_idx = 1 - current_player_idx

        # Max moves reached (shouldn't happen in Connect 4)
        return ("draw", move_history)

    async def _execute_nim(
        self,
        player_1: MinerInfo,
        player_2: MinerInfo,
        target: int = 21,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Execute a Nim counting game between two miners.

        Rules:
        - Count starts at 0
        - Each turn, player adds 1, 2, or 3 to the count
        - The player who reaches or exceeds the target (21) loses

        Optimal strategy: Leave opponent at positions where (target - count) % 4 == 1
        i.e., at 1, 5, 9, 13, 17 for target=21

        Returns:
            Tuple of (outcome, move_history)
            outcome: "a_wins" or "b_wins" (no draws possible)
        """
        count = 0
        move_history = []
        players = [player_1, player_2]
        current_player_idx = 0

        for turn in range(50):  # Max 50 turns (more than enough)
            player = players[current_player_idx]

            # Build prompt - be very explicit about the required format
            prompt = (
                f"Nim game. Count is {count}. First to reach {target} LOSES.\n"
                f"You must add 1, 2, or 3 to the count.\n"
                f"You MUST respond with ONLY this exact format:\n"
                f"<FINAL_ANSWER>N</FINAL_ANSWER>\n"
                f"where N is 1, 2, or 3. Example: <FINAL_ANSWER>2</FINAL_ANSWER>\n"
                f"Your move:"
            )

            # Parser for nim moves - FINAL_ANSWER preferred, but bare 1/2/3 also accepted
            # (Nim is unambiguous - only valid moves are 1, 2, or 3)
            def parse_nim_move(response: str) -> Tuple[Optional[int], Optional[str]]:
                """Parse response and validate move. Returns (add_value, error_msg)."""
                # First try FINAL_ANSWER tag
                answer = extract_final_answer(response)

                # If no FINAL_ANSWER, for Nim we can accept bare 1/2/3 since it's unambiguous
                if answer is None:
                    # Check if response is just a single digit 1, 2, or 3
                    stripped = response.strip()
                    if stripped in ("1", "2", "3"):
                        return (int(stripped), None)
                    # Also check if response contains just one number that's 1, 2, or 3
                    numbers = re.findall(r"\b([123])\b", stripped)
                    if len(numbers) == 1:
                        return (int(numbers[0]), None)
                    return (None, "No <FINAL_ANSWER> block found")

                numbers = re.findall(r"\d+", answer)
                if not numbers:
                    return (None, f"No number in FINAL_ANSWER: {answer}")

                add_val = int(numbers[0])

                if add_val not in [1, 2, 3]:
                    return (None, f"Must be 1, 2, or 3 (got {add_val})")

                return (add_val, None)

            # Get move with retry - if fails, current player forfeits
            try:
                add_val, latency_ms = await self._get_move_with_retry(
                    player=player,
                    prompt=prompt,
                    parse_fn=parse_nim_move,
                    game_type="nim",
                )
            except (ColdChuteError, InfrastructureError) as e:
                # Infrastructure issue - skip game, but track for repeated failures
                logger.info(f"    {player.hotkey[:12]}... infrastructure issue - skipping game")
                self._record_miner_failure(player, "infrastructure")
                move_history.append({
                    "turn": turn,
                    "player": current_player_idx,
                    "skipped": True,
                    "error": str(e),
                })
                return ("skipped", move_history)
            except (APIFailureError, MinerOnCooldownError) as e:
                # Current player couldn't make a valid move - they forfeit
                reason = "on cooldown" if isinstance(e, MinerOnCooldownError) else "failed to respond"
                logger.warning(f"    {player.hotkey[:12]}... forfeits ({reason})")
                winner = 1 - current_player_idx
                move_history.append({
                    "turn": turn,
                    "player": current_player_idx,
                    "forfeit": True,
                    "error": str(e),
                })
                return ("a_wins" if winner == 0 else "b_wins", move_history)

            # Apply move
            count += add_val
            move_history.append({
                "turn": turn,
                "player": current_player_idx,
                "add": add_val,
                "count_after": count,
                "latency_ms": latency_ms,
            })

            # Check if this player lost (reached or exceeded target)
            if count >= target:
                # Current player loses, other player wins
                winner = 1 - current_player_idx
                return ("a_wins" if winner == 0 else "b_wins", move_history)

            # Switch player
            current_player_idx = 1 - current_player_idx

        # Should never reach here
        return ("draw", move_history)

    async def run_game(
        self,
        player_1: MinerInfo,
        player_2: MinerInfo,
        game_type: str,
        pair_uuid: Optional[str] = None,
        pair_sequence: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run a single game between two miners.

        Args:
            player_1: First player (plays first/white)
            player_2: Second player (plays second/black)
            game_type: "tictactoe", "chess", or "connect4"
            pair_uuid: Optional UUID linking match+rematch pairs
            pair_sequence: Optional sequence number within pair (0 or 1)

        Returns:
            Match result dictionary
        """
        match_uuid = str(uuid.uuid4())
        timestamp = int(time.time() * 1000)
        env = f"game:{game_type}"

        # Increment game counter (thread-safe)
        async with self._state_lock:
            game_num = self._games_played + 1
            self._games_played += 1

        logger.info(f"  Game {game_num}: {player_1.hotkey[:12]}... vs {player_2.hotkey[:12]}... ({game_type})")

        try:
            if game_type == "tictactoe":
                outcome, move_history = await self._execute_tictactoe(player_1, player_2)
            elif game_type == "chess":
                outcome, move_history = await self._execute_chess(player_1, player_2)
            elif game_type == "connect4":
                outcome, move_history = await self._execute_connect4(player_1, player_2)
            elif game_type == "nim":
                outcome, move_history = await self._execute_nim(player_1, player_2)
            else:
                raise ValueError(f"Unknown game type: {game_type}")

            async with self._state_lock:
                self._games_completed += 1

        except RateLimitError:
            # Re-raise rate limit errors to stop tournament early
            raise
        # Note: APIFailureError is now handled within game functions as forfeit
        except Exception as e:
            logger.error(f"  Game error: {e}")
            outcome = "draw"
            move_history = []
            async with self._state_lock:
                self._games_errored += 1

        # Handle skipped games (cold chute) - no ELO update, no match record
        if outcome == "skipped":
            async with self._state_lock:
                self._games_skipped += 1
            logger.info(f"    Result: Skipped (cold chute)")
            return {
                "match_uuid": match_uuid,
                "env": env,
                "game_type": game_type,
                "timestamp": timestamp,
                "skipped": True,
                "participants": [
                    {"miner_hotkey": player_1.hotkey, "outcome": "skipped"},
                    {"miner_hotkey": player_2.hotkey, "outcome": "skipped"},
                ],
                "move_history": move_history,
                # Paired match tracking
                "pair_uuid": pair_uuid,
                "is_first_mover": True,
                "pair_sequence": pair_sequence,
            }

        # Log result
        outcome_str = "Draw" if outcome == "draw" else (
            f"{player_1.hotkey[:12]}... wins" if outcome == "a_wins" else f"{player_2.hotkey[:12]}... wins"
        )
        logger.info(f"    Result: {outcome_str} ({len(move_history)} moves)")

        # Update ELO ratings and build match record (thread-safe)
        async with self._state_lock:
            rating_1 = self._get_or_create_rating(player_1, env)
            rating_2 = self._get_or_create_rating(player_2, env)

            new_rating_1, new_rating_2, delta_1, delta_2 = self.calculator.update_ratings_head_to_head(
                rating_a=rating_1.rating,
                rating_b=rating_2.rating,
                matches_a=rating_1.matches_played,
                matches_b=rating_2.matches_played,
                outcome=outcome,
            )

            # Update rating 1
            old_rating_1 = float(rating_1.rating)
            rating_1.rating = new_rating_1
            rating_1.matches_played += 1
            rating_1.last_match_at = timestamp
            if new_rating_1 > rating_1.peak_rating:
                rating_1.peak_rating = new_rating_1
            if outcome == "a_wins":
                rating_1.wins += 1
            elif outcome == "b_wins":
                rating_1.losses += 1
            else:
                rating_1.draws += 1

            # Update rating 2
            old_rating_2 = float(rating_2.rating)
            rating_2.rating = new_rating_2
            rating_2.matches_played += 1
            rating_2.last_match_at = timestamp
            if new_rating_2 > rating_2.peak_rating:
                rating_2.peak_rating = new_rating_2
            if outcome == "b_wins":
                rating_2.wins += 1
            elif outcome == "a_wins":
                rating_2.losses += 1
            else:
                rating_2.draws += 1

            # Record game completions for runtime balancing
            if self.enable_runtime_balancing:
                self._sampler.record_game_complete(player_1)
                self._sampler.record_game_complete(player_2)

            # Build match record
            match_record = {
                "match_uuid": match_uuid,
                "env": env,
                "game_type": game_type,
                "timestamp": timestamp,
                "participants": [
                    {
                        "miner_hotkey": player_1.hotkey,
                        "model_revision": player_1.model_revision,
                        "outcome": "win" if outcome == "a_wins" else ("loss" if outcome == "b_wins" else "draw"),
                        "slot": 0,
                        "elo_before": old_rating_1,
                        "elo_after": float(new_rating_1),
                    },
                    {
                        "miner_hotkey": player_2.hotkey,
                        "model_revision": player_2.model_revision,
                        "outcome": "win" if outcome == "b_wins" else ("loss" if outcome == "a_wins" else "draw"),
                        "slot": 1,
                        "elo_before": old_rating_2,
                        "elo_after": float(new_rating_2),
                    },
                ],
                "move_history": move_history,
                "total_moves": len(move_history),
                # Paired match tracking
                "pair_uuid": pair_uuid,
                "is_first_mover": True,  # slot 0 is always the first mover
                "pair_sequence": pair_sequence,
            }

            self._matches.append(match_record)

            # Call game complete callback (for real-time saving)
            if self.on_game_complete:
                try:
                    self.on_game_complete(match_record)
                except Exception as e:
                    logger.warning(f"on_game_complete callback failed: {e}")

        logger.info(f"    ELO: {player_1.hotkey[:12]}... {old_rating_1:.0f} -> {float(new_rating_1):.0f} ({float(delta_1):+.0f})")
        logger.info(f"    ELO: {player_2.hotkey[:12]}... {old_rating_2:.0f} -> {float(new_rating_2):.0f} ({float(delta_2):+.0f})")

        return match_record

    async def run_double_game(
        self,
        player_a: MinerInfo,
        player_b: MinerInfo,
        game_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Run a double game (both directions) to cancel first-mover advantage.

        Plays A vs B, then B vs A. If a player fails to respond after retries,
        they forfeit that game (opponent wins).

        The two games are linked by a pair_uuid for paired analysis.

        Args:
            player_a: First player
            player_b: Second player
            game_type: "tictactoe", "chess", "connect4", or "nim"

        Returns:
            List of two match results with linked pair_uuid
        """
        # Generate pair UUID to link both games
        pair_uuid = str(uuid.uuid4())

        logger.info(f"  Double game: {player_a.hotkey[:12]}... <-> {player_b.hotkey[:12]}... ({game_type}) [pair: {pair_uuid[:8]}...]")

        # Game 1: A vs B (A plays first)
        result_1 = await self.run_game(player_a, player_b, game_type, pair_uuid=pair_uuid, pair_sequence=0)

        # Game 2: B vs A (B plays first)
        result_2 = await self.run_game(player_b, player_a, game_type, pair_uuid=pair_uuid, pair_sequence=1)

        return [result_1, result_2]

    async def run_forever(
        self,
        concurrent_workers: int = 8,
        log_interval: int = 10,
        rate_limit_wait: int = 60,
    ) -> None:
        """
        Run continuous tournament forever with worker threads.

        Each worker independently acquires the lowest-runtime available pair,
        runs a double game, then repeats. Runs until interrupted.
        On rate limit, waits and continues.

        Args:
            concurrent_workers: Number of concurrent worker tasks (default: 8)
            log_interval: Log runtime distribution every N completed double-games
            rate_limit_wait: Seconds to wait on rate limit (default: 60)
        """
        n_miners = len(self.miners)

        logger.info("=" * 60)
        logger.info("LOCAL GAME VALIDATOR - CONTINUOUS")
        logger.info("=" * 60)
        logger.info(f"Miners: {n_miners}")
        logger.info(f"Game types: {self.game_types}")
        logger.info(f"Workers: {concurrent_workers}")
        logger.info(f"Runtime balancing: {'enabled' if self.enable_runtime_balancing else 'disabled'}")
        logger.info("=" * 60)

        for miner in self.miners:
            logger.info(f"  - {miner.hotkey[:20]}... (slug: {miner.chute_slug})")

        logger.info("=" * 60)

        games_completed = 0

        async def worker(worker_id: int):
            """Worker that continuously runs games."""
            nonlocal games_completed

            while True:
                # Acquire a pair (lowest runtime, not busy)
                if self.enable_runtime_balancing:
                    pair = await self._sampler.acquire_pair(
                        self.miners,
                        valid_miner_filter=self.is_miner_valid_for_cycle,
                    )
                    if pair is None:
                        # Not enough available miners, wait and retry
                        await asyncio.sleep(0.5)
                        continue
                    p1, p2 = pair
                else:
                    # Random selection (fallback)
                    valid = [m for m in self.miners if self.is_miner_valid_for_cycle(m)]
                    if len(valid) < 2:
                        await asyncio.sleep(0.5)
                        continue
                    p1, p2 = random.sample(valid, 2)

                game_type = random.choice(self.game_types)

                try:
                    await self.run_double_game(p1, p2, game_type)
                    games_completed += 1

                    # Periodic logging
                    if games_completed % log_interval == 0:
                        self._sampler.log_runtime_distribution(self.miners)

                except RateLimitError as e:
                    logger.warning(f"Rate limit hit, waiting {rate_limit_wait}s: {e}")
                    await asyncio.sleep(rate_limit_wait)
                except Exception as e:
                    logger.error(f"  Worker {worker_id} game failed: {e}")
                finally:
                    # Release the pair back to available pool
                    if self.enable_runtime_balancing:
                        await self._sampler.release_pair(p1, p2)

        # Start workers and run forever
        workers = [asyncio.create_task(worker(i)) for i in range(concurrent_workers)]
        await asyncio.gather(*workers)

    def generate_report(self, min_matches: int = 1) -> ValidationReport:
        """
        Generate a validation report from local tournament results.

        Args:
            min_matches: Minimum matches to include in weights

        Returns:
            ValidationReport
        """
        # Calculate weights from local ratings
        engine = EloValidatorEngine(
            environments=list(set(f"game:{gt}" for gt in self.game_types)),
            elo_config=self.elo_config,
            local_mode=True,
        )

        # Copy ratings to engine
        for key, rating in self._ratings.items():
            engine.add_rating(rating)

        weights = engine.calculate_weights(self._ratings, min_matches=min_matches)

        # Count total unique matches
        total_matches = len(self._matches)

        return ValidationReport(
            timestamp=int(time.time()),
            environments=list(set(f"game:{gt}" for gt in self.game_types)),
            total_miners=len(self.miners),
            total_matches=total_matches,
            weights=weights,
            config=self.elo_config.to_dict(),
            replay_mode=False,
        )


