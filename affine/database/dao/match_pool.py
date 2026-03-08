"""
Match Pool DAO

Handles storage and retrieval of pending multi-party game matches.
"""

import time
import uuid
from typing import Dict, Any, List, Optional

from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name
from affine.database.client import get_client, is_local_mode
from affine.core.setup import logger


class MatchPoolDAO(BaseDAO):
    """DAO for match_pool table.

    Stores pending and in-progress multi-party game matches.

    Schema:
    PK: GAME#{game_type}#STATUS#{status}
    SK: MATCH#{match_uuid}
    GSI: status-created-index (STATUS#{status}, created_at) for FIFO fetch
    """

    # Match statuses
    STATUS_PENDING = "pending"  # Waiting for execution
    STATUS_EXECUTING = "executing"  # Currently being executed
    STATUS_COMPLETED = "completed"  # Finished successfully
    STATUS_FAILED = "failed"  # Failed during execution

    def __init__(self):
        self.table_name = get_table_name("match_pool")
        super().__init__()

    def _make_pk(self, game_type: str, status: str) -> str:
        """Generate partition key."""
        return f"GAME#{game_type}#STATUS#{status}"

    def _make_sk(self, match_uuid: str) -> str:
        """Generate sort key."""
        return f"MATCH#{match_uuid}"

    async def create_match(
        self,
        game_type: str,
        player_count: int,
        participants: List[Dict[str, Any]],
        task_id: int,
        game_config: Optional[Dict[str, Any]] = None,
        ttl_days: int = 3,
    ) -> Dict[str, Any]:
        """Create a new match in the pool.

        Args:
            game_type: Type of game (e.g., "tictactoe", "chess")
            player_count: Required number of players
            participants: List of participant info (must have player_count entries)
            task_id: Game seed/scenario identifier
            game_config: Game-specific configuration
            ttl_days: Days until auto-deletion (default 3)

        Returns:
            Created match record
        """
        match_uuid = str(uuid.uuid4())
        created_at = int(time.time() * 1000)

        item = {
            "pk": self._make_pk(game_type, self.STATUS_PENDING),
            "sk": self._make_sk(match_uuid),
            "match_uuid": match_uuid,
            "game_type": game_type,
            "game_env": f"game:{game_type}",
            "status": self.STATUS_PENDING,
            "player_count": player_count,
            "participants": participants,
            "task_id": task_id,
            "game_config": game_config or {},
            "created_at": created_at,
            "matched_at": created_at,  # All participants present at creation
            "gsi1_pk": f"STATUS#{self.STATUS_PENDING}",
            "gsi1_sk": created_at,
            "ttl": self.get_ttl(ttl_days),
        }

        return await self.put(item)

    async def get_match(
        self,
        game_type: str,
        status: str,
        match_uuid: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a specific match.

        Args:
            game_type: Type of game
            status: Current status
            match_uuid: Match UUID

        Returns:
            Match record or None if not found
        """
        pk = self._make_pk(game_type, status)
        sk = self._make_sk(match_uuid)
        return await self.get(pk, sk)

    async def get_pending_matches(
        self,
        game_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get pending matches for execution (FIFO).

        Uses GSI for efficient FIFO ordering.

        Args:
            game_type: Optional filter by game type
            limit: Maximum number of matches to return

        Returns:
            List of pending matches, oldest first
        """
        client = get_client()

        params = {
            "TableName": self.table_name,
            "IndexName": "status-created-index",
            "KeyConditionExpression": "gsi1_pk = :status",
            "ExpressionAttributeValues": {
                ":status": {"S": f"STATUS#{self.STATUS_PENDING}"},
            },
            "ScanIndexForward": True,  # Oldest first (FIFO)
            "Limit": limit,
        }

        if game_type:
            params["FilterExpression"] = "game_type = :game_type"
            params["ExpressionAttributeValues"][":game_type"] = {"S": game_type}
            # Over-fetch due to filter
            params["Limit"] = limit * 5

        response = await client.query(**params)
        items = [self._deserialize(item) for item in response.get("Items", [])]

        return items[:limit]

    async def fetch_and_assign_match(
        self,
        executor_hotkey: str,
        game_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Fetch a pending match and assign it to an executor.

        Uses conditional delete to prevent double-assignment: the delete only
        succeeds if the item still exists with STATUS_PENDING. If two executors
        race, only one delete succeeds.

        Args:
            executor_hotkey: Executor's hotkey
            game_type: Optional filter by game type

        Returns:
            Assigned match or None if no matches available
        """
        pending_matches = await self.get_pending_matches(game_type, limit=5)
        client = get_client()

        for match in pending_matches:
            old_pk = self._make_pk(match["game_type"], self.STATUS_PENDING)
            old_sk = self._make_sk(match["match_uuid"])

            timestamp = int(time.time() * 1000)
            match["pk"] = self._make_pk(match["game_type"], self.STATUS_EXECUTING)
            match["status"] = self.STATUS_EXECUTING
            match["assigned_to"] = executor_hotkey
            match["assigned_at"] = timestamp
            match["gsi1_pk"] = f"STATUS#{self.STATUS_EXECUTING}"
            match["gsi1_sk"] = timestamp

            serialized = self._serialize(match)

            if is_local_mode():
                # Local mode: conditional delete + put under lock (single-process)
                try:
                    await client.delete_item(
                        TableName=self.table_name,
                        Key={"pk": {"S": old_pk}, "sk": {"S": old_sk}},
                        ConditionExpression="attribute_exists(pk)",
                    )
                except Exception as e:
                    if 'ConditionalCheckFailed' in str(e):
                        continue
                    raise
                await self.put(match)
            else:
                # Real DynamoDB: atomic delete + put in a single transaction
                try:
                    await client.transact_write_items(
                        TransactItems=[
                            {
                                "Delete": {
                                    "TableName": self.table_name,
                                    "Key": {"pk": {"S": old_pk}, "sk": {"S": old_sk}},
                                    "ConditionExpression": "attribute_exists(pk)",
                                }
                            },
                            {
                                "Put": {
                                    "TableName": self.table_name,
                                    "Item": serialized,
                                }
                            },
                        ]
                    )
                except Exception as e:
                    err_str = str(e)
                    if 'ConditionalCheckFailed' in err_str or 'TransactionCanceled' in err_str:
                        continue
                    raise

            return match

        return None

    async def complete_match(
        self,
        game_type: str,
        match_uuid: str,
        success: bool = True,
    ) -> bool:
        """Mark a match as completed or failed.

        Removes the match from the executing pool.

        Args:
            game_type: Type of game
            match_uuid: Match UUID
            success: Whether the match completed successfully

        Returns:
            True if match was found and updated, False otherwise
        """
        # Get the executing match
        match = await self.get_match(game_type, self.STATUS_EXECUTING, match_uuid)

        if not match:
            logger.warning(f"Match not found for completion: {match_uuid}")
            return False

        # Delete from executing
        old_pk = self._make_pk(game_type, self.STATUS_EXECUTING)
        old_sk = self._make_sk(match_uuid)
        await self.delete(old_pk, old_sk)

        # Optionally track completed/failed matches (short TTL)
        new_status = self.STATUS_COMPLETED if success else self.STATUS_FAILED
        match["pk"] = self._make_pk(game_type, new_status)
        match["status"] = new_status
        match["completed_at"] = int(time.time() * 1000)
        match["gsi1_pk"] = f"STATUS#{new_status}"
        match["gsi1_sk"] = match["completed_at"]
        match["ttl"] = self.get_ttl(1)  # 1 day retention for completed matches

        await self.put(match)

        return True

    async def get_matches_by_status(
        self,
        game_type: str,
        status: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get all matches with a specific status for a game type.

        Args:
            game_type: Type of game
            status: Match status
            limit: Maximum number of matches to return

        Returns:
            List of matches
        """
        pk = self._make_pk(game_type, status)
        return await self.query(pk, limit=limit)

    async def get_executing_matches(
        self,
        executor_hotkey: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get currently executing matches.

        Args:
            executor_hotkey: Optional filter by executor

        Returns:
            List of executing matches
        """
        client = get_client()

        params = {
            "TableName": self.table_name,
            "IndexName": "status-created-index",
            "KeyConditionExpression": "gsi1_pk = :status",
            "ExpressionAttributeValues": {
                ":status": {"S": f"STATUS#{self.STATUS_EXECUTING}"},
            },
        }

        if executor_hotkey:
            params["FilterExpression"] = "assigned_to = :executor"
            params["ExpressionAttributeValues"][":executor"] = {"S": executor_hotkey}

        response = await client.query(**params)
        return [self._deserialize(item) for item in response.get("Items", [])]

    async def get_pool_stats(
        self,
        game_type: Optional[str] = None,
    ) -> Dict[str, int]:
        """Get pool statistics.

        Args:
            game_type: Optional filter by game type

        Returns:
            Dictionary with counts by status
        """
        stats = {
            "pending": 0,
            "executing": 0,
            "completed": 0,
            "failed": 0,
        }

        client = get_client()

        for status in stats.keys():
            params = {
                "TableName": self.table_name,
                "IndexName": "status-created-index",
                "KeyConditionExpression": "gsi1_pk = :status",
                "ExpressionAttributeValues": {
                    ":status": {"S": f"STATUS#{status}"},
                },
                "Select": "COUNT",
            }

            if game_type:
                params["FilterExpression"] = "game_type = :game_type"
                params["ExpressionAttributeValues"][":game_type"] = {"S": game_type}

            response = await client.query(**params)
            stats[status] = response.get("Count", 0)

        return stats

    async def cleanup_stale_executing(
        self,
        max_age_minutes: int = 60,
    ) -> int:
        """Clean up stale executing matches (orphaned).

        Moves matches that have been executing for too long back to pending.

        Args:
            max_age_minutes: Maximum age in minutes before considering stale

        Returns:
            Number of matches cleaned up
        """
        cutoff = int(time.time() * 1000) - (max_age_minutes * 60 * 1000)

        executing = await self.get_executing_matches()
        cleaned = 0

        for match in executing:
            assigned_at = match.get("assigned_at", 0)
            if assigned_at < cutoff:
                # Move back to pending
                old_pk = self._make_pk(match["game_type"], self.STATUS_EXECUTING)
                old_sk = self._make_sk(match["match_uuid"])
                await self.delete(old_pk, old_sk)

                match["pk"] = self._make_pk(match["game_type"], self.STATUS_PENDING)
                match["status"] = self.STATUS_PENDING
                match["gsi1_pk"] = f"STATUS#{self.STATUS_PENDING}"
                match["gsi1_sk"] = int(time.time() * 1000)
                del match["assigned_to"]
                del match["assigned_at"]

                await self.put(match)
                cleaned += 1

                logger.info(f"Cleaned up stale match: {match['match_uuid']}")

        return cleaned
