"""
MatchMaker Service

Creates balanced matches for multi-party games by pairing/grouping miners.
"""

import random
from decimal import Decimal
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from affine.core.setup import logger
from affine.database.dao.match_pool import MatchPoolDAO
from affine.database.dao.elo_ratings import EloRatingsDAO


class MatchmakingStrategy(Enum):
    """Matchmaking strategies for pairing miners."""

    RANDOM = "random"
    """Randomly pair available miners."""

    ELO_BALANCED = "elo_balanced"
    """Match miners with similar ELO ratings."""

    ROUND_ROBIN = "round_robin"
    """Ensure all miners play each other over time."""

    SKILL_BRACKET = "skill_bracket"
    """Group miners into skill brackets, match within brackets."""


@dataclass
class GameConfig:
    """Configuration for a game type."""

    game_type: str
    player_count: int
    symmetric: bool = True
    timeout_per_move: int = 30
    max_moves: int = 200
    matchmaking_strategy: MatchmakingStrategy = MatchmakingStrategy.ELO_BALANCED
    min_matches_per_miner: int = 10
    roles: Optional[List[str]] = None  # For asymmetric games


# Default game configurations
DEFAULT_GAME_CONFIGS: Dict[str, GameConfig] = {
    "tictactoe": GameConfig(
        game_type="tictactoe",
        player_count=2,
        symmetric=True,
        timeout_per_move=1800,
        max_moves=9,
        matchmaking_strategy=MatchmakingStrategy.ELO_BALANCED,
    ),
    "chess": GameConfig(
        game_type="chess",
        player_count=2,
        symmetric=True,
        timeout_per_move=1800,
        max_moves=200,
        matchmaking_strategy=MatchmakingStrategy.ELO_BALANCED,
    ),
}


class MatchMaker:
    """
    Creates balanced matches for multi-party games.

    Supports multiple matchmaking strategies:
    - Random: Simple random pairing
    - ELO Balanced: Match miners with similar ratings
    - Round Robin: Systematic pairing to ensure coverage
    - Skill Bracket: Group by skill level
    """

    def __init__(
        self,
        match_pool_dao: Optional[MatchPoolDAO] = None,
        elo_ratings_dao: Optional[EloRatingsDAO] = None,
    ):
        """
        Initialize the MatchMaker.

        Args:
            match_pool_dao: DAO for match pool (optional, creates new if not provided)
            elo_ratings_dao: DAO for ELO ratings (optional, creates new if not provided)
        """
        self.match_pool_dao = match_pool_dao or MatchPoolDAO()
        self.elo_ratings_dao = elo_ratings_dao or EloRatingsDAO()

    async def create_matches(
        self,
        game_type: str,
        valid_miners: List[Dict[str, Any]],
        task_ids: List[int],
        game_config: Optional[GameConfig] = None,
        max_matches: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Create matches for a game type.

        Args:
            game_type: Type of game (e.g., "tictactoe", "chess")
            valid_miners: List of valid miners with {hotkey, revision, model, chute_slug}
            task_ids: List of task IDs (game seeds) to use
            game_config: Game configuration (uses default if not provided)
            max_matches: Maximum number of matches to create

        Returns:
            List of created match records
        """
        if not valid_miners or len(valid_miners) < 2:
            logger.warning(f"Not enough miners for {game_type} matches: {len(valid_miners)}")
            return []

        if not task_ids:
            logger.warning(f"No task IDs provided for {game_type} matches")
            return []

        config = game_config or DEFAULT_GAME_CONFIGS.get(game_type)
        if not config:
            logger.error(f"No game config found for {game_type}")
            return []

        # Get current ELO ratings for all miners
        elo_cache = await self._get_elo_cache(valid_miners, f"game:{game_type}")

        # Apply matchmaking strategy
        strategy = config.matchmaking_strategy
        if strategy == MatchmakingStrategy.RANDOM:
            groups = await self._match_random(valid_miners, config.player_count)
        elif strategy == MatchmakingStrategy.ELO_BALANCED:
            groups = await self._match_elo_balanced(valid_miners, config.player_count, elo_cache)
        elif strategy == MatchmakingStrategy.ROUND_ROBIN:
            groups = await self._match_round_robin(valid_miners, config.player_count)
        elif strategy == MatchmakingStrategy.SKILL_BRACKET:
            groups = await self._match_skill_bracket(valid_miners, config.player_count, elo_cache)
        else:
            groups = await self._match_random(valid_miners, config.player_count)

        # Limit number of matches
        groups = groups[:max_matches]

        # Create match entries
        matches = []
        for i, group in enumerate(groups):
            task_id = task_ids[i % len(task_ids)]
            match = await self._create_match_entry(game_type, group, task_id, config, elo_cache)
            if match:
                matches.append(match)

        logger.info(f"Created {len(matches)} matches for {game_type}")
        return matches

    async def _get_elo_cache(
        self,
        miners: List[Dict[str, Any]],
        env: str,
    ) -> Dict[str, Decimal]:
        """
        Get ELO ratings for all miners.

        Args:
            miners: List of miners
            env: Environment name

        Returns:
            Dictionary mapping miner_id to ELO rating
        """
        cache = {}
        default_rating = Decimal("1500")

        for miner in miners:
            hotkey = miner["hotkey"]
            revision = miner["revision"]
            miner_id = f"{hotkey}#{revision}"

            rating = await self.elo_ratings_dao.get_rating(hotkey, revision, env)
            cache[miner_id] = Decimal(str(rating.get("rating", default_rating))) if rating else default_rating

        return cache

    async def _match_random(
        self,
        miners: List[Dict[str, Any]],
        player_count: int,
    ) -> List[List[Dict[str, Any]]]:
        """
        Randomly pair/group miners.

        Args:
            miners: List of miners
            player_count: Players per match

        Returns:
            List of miner groups
        """
        shuffled = miners.copy()
        random.shuffle(shuffled)

        groups = []
        for i in range(0, len(shuffled) - player_count + 1, player_count):
            groups.append(shuffled[i : i + player_count])

        return groups

    async def _match_elo_balanced(
        self,
        miners: List[Dict[str, Any]],
        player_count: int,
        elo_cache: Dict[str, Decimal],
    ) -> List[List[Dict[str, Any]]]:
        """
        Match miners with similar ELO ratings.

        Sorts by ELO and pairs adjacent miners for closest skill matching.

        Args:
            miners: List of miners
            player_count: Players per match
            elo_cache: ELO ratings cache

        Returns:
            List of miner groups
        """
        # Sort miners by ELO rating
        sorted_miners = sorted(
            miners,
            key=lambda m: elo_cache.get(f"{m['hotkey']}#{m['revision']}", Decimal("1500")),
        )

        # Create groups from adjacent miners (similar skill)
        groups = []
        for i in range(0, len(sorted_miners) - player_count + 1, player_count):
            groups.append(sorted_miners[i : i + player_count])

        # Shuffle groups to add variety (but keep internal pairings)
        random.shuffle(groups)

        return groups

    async def _match_round_robin(
        self,
        miners: List[Dict[str, Any]],
        player_count: int,
    ) -> List[List[Dict[str, Any]]]:
        """
        Create round-robin pairings to ensure all miners play each other.

        For 2-player games, creates all possible pairs.

        Args:
            miners: List of miners
            player_count: Players per match

        Returns:
            List of miner groups
        """
        if player_count != 2:
            # Fall back to random for non-2-player games
            return await self._match_random(miners, player_count)

        # Generate all pairs
        from itertools import combinations

        pairs = list(combinations(miners, 2))
        random.shuffle(pairs)

        return [list(pair) for pair in pairs]

    async def _match_skill_bracket(
        self,
        miners: List[Dict[str, Any]],
        player_count: int,
        elo_cache: Dict[str, Decimal],
        num_brackets: int = 4,
    ) -> List[List[Dict[str, Any]]]:
        """
        Group miners into skill brackets and match within brackets.

        Args:
            miners: List of miners
            player_count: Players per match
            elo_cache: ELO ratings cache
            num_brackets: Number of skill brackets

        Returns:
            List of miner groups
        """
        # Sort miners by ELO
        sorted_miners = sorted(
            miners,
            key=lambda m: elo_cache.get(f"{m['hotkey']}#{m['revision']}", Decimal("1500")),
        )

        # Divide into brackets
        bracket_size = max(1, len(sorted_miners) // num_brackets)
        brackets = []
        for i in range(0, len(sorted_miners), bracket_size):
            brackets.append(sorted_miners[i : i + bracket_size])

        # Match within each bracket
        groups = []
        for bracket in brackets:
            if len(bracket) >= player_count:
                bracket_groups = await self._match_random(bracket, player_count)
                groups.extend(bracket_groups)

        return groups

    async def _create_match_entry(
        self,
        game_type: str,
        miners: List[Dict[str, Any]],
        task_id: int,
        config: GameConfig,
        elo_cache: Dict[str, Decimal],
    ) -> Optional[Dict[str, Any]]:
        """
        Create a match entry in the match pool.

        Args:
            game_type: Type of game
            miners: List of miners for this match
            task_id: Task/game seed ID
            config: Game configuration
            elo_cache: ELO ratings cache

        Returns:
            Created match record or None
        """
        participants = []
        for i, miner in enumerate(miners):
            miner_id = f"{miner['hotkey']}#{miner['revision']}"
            elo = elo_cache.get(miner_id, Decimal("1500"))

            role = None
            if config.roles and i < len(config.roles):
                role = config.roles[i]

            participants.append(
                {
                    "slot": i,
                    "miner_hotkey": miner["hotkey"],
                    "model_revision": miner["revision"],
                    "model": miner.get("model", ""),
                    "chute_slug": miner.get("chute_slug", ""),
                    "role": role,
                    "elo_rating": float(elo),
                }
            )

        game_config_dict = {
            "timeout_per_move": config.timeout_per_move,
            "max_moves": config.max_moves,
            "symmetric": config.symmetric,
        }

        try:
            match = await self.match_pool_dao.create_match(
                game_type=game_type,
                player_count=config.player_count,
                participants=participants,
                task_id=task_id,
                game_config=game_config_dict,
            )
            return match
        except Exception as e:
            logger.error(f"Failed to create match: {e}")
            return None

    async def get_pool_stats(
        self,
        game_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get match pool statistics.

        Args:
            game_type: Optional filter by game type

        Returns:
            Pool statistics
        """
        return await self.match_pool_dao.get_pool_stats(game_type)

    async def cleanup_stale_matches(
        self,
        max_age_minutes: int = 60,
    ) -> int:
        """
        Clean up stale executing matches.

        Args:
            max_age_minutes: Maximum age before considering stale

        Returns:
            Number of matches cleaned up
        """
        return await self.match_pool_dao.cleanup_stale_executing(max_age_minutes)


async def create_matchmaker() -> MatchMaker:
    """Factory function to create a MatchMaker instance."""
    return MatchMaker()
