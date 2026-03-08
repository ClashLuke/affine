"""
ELO Ratings DAO

Handles storage and retrieval of ELO ratings for miners.
"""

import time
from decimal import Decimal
from typing import Dict, Any, List, Optional

from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name
from affine.database.client import get_client
from affine.core.setup import logger


class EloRatingsDAO(BaseDAO):
    """DAO for elo_ratings table.

    Stores ELO ratings per miner per environment.

    Schema:
    PK: MINER#{hotkey}#REV#{revision}
    SK: ENV#{env}
    GSI: env-rating-index (env -> rating DESC) for leaderboards
    """

    def __init__(self):
        self.table_name = get_table_name("elo_ratings")
        super().__init__()

    def _make_pk(self, miner_hotkey: str, model_revision: str) -> str:
        """Generate partition key."""
        return f"MINER#{miner_hotkey}#REV#{model_revision}"

    def _make_sk(self, env: str) -> str:
        """Generate sort key."""
        return f"ENV#{env}"

    async def get_rating(
        self,
        miner_hotkey: str,
        model_revision: str,
        env: str,
    ) -> Optional[Dict[str, Any]]:
        """Get ELO rating for a miner in an environment.

        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision hash
            env: Environment name

        Returns:
            Rating record or None if not found
        """
        pk = self._make_pk(miner_hotkey, model_revision)
        sk = self._make_sk(env)
        return await self.get(pk, sk)

    async def get_all_ratings_for_miner(
        self,
        miner_hotkey: str,
        model_revision: str,
    ) -> List[Dict[str, Any]]:
        """Get all ELO ratings for a miner across all environments.

        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision hash

        Returns:
            List of rating records
        """
        pk = self._make_pk(miner_hotkey, model_revision)
        return await self.query(pk)

    async def save_rating(
        self,
        miner_hotkey: str,
        model_revision: str,
        env: str,
        rating: Decimal,
        peak_rating: Optional[Decimal] = None,
        matches_played: int = 0,
        wins: int = 0,
        losses: int = 0,
        draws: int = 0,
        last_match_at: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Save or update an ELO rating.

        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision hash
            env: Environment name
            rating: Current ELO rating
            peak_rating: Historical peak rating (defaults to rating)
            matches_played: Total matches played
            wins: Win count
            losses: Loss count
            draws: Draw count
            last_match_at: Timestamp of last match

        Returns:
            Saved item
        """
        if peak_rating is None:
            peak_rating = rating

        timestamp = int(time.time())

        item = {
            "pk": self._make_pk(miner_hotkey, model_revision),
            "sk": self._make_sk(env),
            "miner_hotkey": miner_hotkey,
            "model_revision": model_revision,
            "env": env,
            "rating": rating,
            "peak_rating": peak_rating,
            "matches_played": matches_played,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "last_match_at": last_match_at,
            "updated_at": timestamp,
        }

        return await self.put(item)

    async def update_rating_after_match(
        self,
        miner_hotkey: str,
        model_revision: str,
        env: str,
        new_rating: Decimal,
        outcome: str,  # "win", "loss", "draw"
        match_timestamp: int,
    ) -> Dict[str, Any]:
        """Update ELO rating after a match.

        Creates the record if it doesn't exist.

        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision hash
            env: Environment name
            new_rating: New ELO rating after the match
            outcome: Match outcome ("win", "loss", "draw")
            match_timestamp: Timestamp of the match

        Returns:
            Updated item
        """
        existing = await self.get_rating(miner_hotkey, model_revision, env)

        if existing:
            matches_played = existing.get("matches_played", 0) + 1
            wins = existing.get("wins", 0) + (1 if outcome == "win" else 0)
            losses = existing.get("losses", 0) + (1 if outcome == "loss" else 0)
            draws = existing.get("draws", 0) + (1 if outcome == "draw" else 0)
            peak_rating = max(
                Decimal(str(existing.get("peak_rating", new_rating))),
                new_rating
            )
        else:
            matches_played = 1
            wins = 1 if outcome == "win" else 0
            losses = 1 if outcome == "loss" else 0
            draws = 1 if outcome == "draw" else 0
            peak_rating = new_rating

        return await self.save_rating(
            miner_hotkey=miner_hotkey,
            model_revision=model_revision,
            env=env,
            rating=new_rating,
            peak_rating=peak_rating,
            matches_played=matches_played,
            wins=wins,
            losses=losses,
            draws=draws,
            last_match_at=match_timestamp,
        )

    async def initialize_rating(
        self,
        miner_hotkey: str,
        model_revision: str,
        env: str,
        initial_rating: Decimal = Decimal("1500"),
    ) -> Dict[str, Any]:
        """Initialize ELO rating for a new miner.

        Does nothing if rating already exists.

        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision hash
            env: Environment name
            initial_rating: Starting ELO rating (default 1500)

        Returns:
            Existing or new rating record
        """
        existing = await self.get_rating(miner_hotkey, model_revision, env)
        if existing:
            return existing

        return await self.save_rating(
            miner_hotkey=miner_hotkey,
            model_revision=model_revision,
            env=env,
            rating=initial_rating,
            peak_rating=initial_rating,
            matches_played=0,
            wins=0,
            losses=0,
            draws=0,
        )

    async def get_leaderboard(
        self,
        env: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get top-rated miners for an environment.

        Uses GSI env-rating-index for efficient leaderboard queries.

        Args:
            env: Environment name
            limit: Maximum number of results (default 100)

        Returns:
            List of rating records sorted by rating DESC
        """
        client = get_client()

        params = {
            "TableName": self.table_name,
            "IndexName": "env-rating-index",
            "KeyConditionExpression": "env = :env",
            "ExpressionAttributeValues": {":env": {"S": env}},
            "ScanIndexForward": False,  # Descending order (highest first)
            "Limit": limit,
        }

        response = await client.query(**params)
        items = response.get("Items", [])

        return [self._deserialize(item) for item in items]

    async def get_all_ratings_for_env(
        self,
        env: str,
    ) -> List[Dict[str, Any]]:
        """Get all ratings for an environment.

        Args:
            env: Environment name

        Returns:
            List of all rating records for the environment
        """
        client = get_client()

        params = {
            "TableName": self.table_name,
            "IndexName": "env-rating-index",
            "KeyConditionExpression": "env = :env",
            "ExpressionAttributeValues": {":env": {"S": env}},
        }

        items = []
        while True:
            response = await client.query(**params)
            items.extend([self._deserialize(item) for item in response.get("Items", [])])

            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            params["ExclusiveStartKey"] = last_key

        return items

    async def delete_rating(
        self,
        miner_hotkey: str,
        model_revision: str,
        env: str,
    ) -> bool:
        """Delete an ELO rating record.

        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision hash
            env: Environment name

        Returns:
            True if deleted, False otherwise
        """
        pk = self._make_pk(miner_hotkey, model_revision)
        sk = self._make_sk(env)
        return await self.delete(pk, sk)
