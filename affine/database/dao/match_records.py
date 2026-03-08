"""
Match Records DAO

Handles storage and retrieval of match records for ELO tracking.
"""

import time
import json
from typing import Dict, Any, List, Optional

from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name
from affine.database.client import get_client
from affine.core.setup import logger


class MatchRecordsDAO(BaseDAO):
    """DAO for match_records table.

    Stores match history for ELO rating calculations and auditing.

    Schema:
    PK: ENV#{env}
    SK: MATCH#{timestamp}#{match_uuid}
    GSI: timestamp-index (gsi_partition='MATCH', timestamp) for time-range queries
    """

    def __init__(self):
        self.table_name = get_table_name("match_records")
        super().__init__()

    def _make_pk(self, env: str) -> str:
        """Generate partition key."""
        return f"ENV#{env}"

    def _make_sk(self, timestamp: int, match_uuid: str) -> str:
        """Generate sort key."""
        return f"MATCH#{timestamp}#{match_uuid}"

    async def save_match(
        self,
        match_uuid: str,
        env: str,
        match_type: str,
        task_id: int,
        participants: List[Dict[str, Any]],
        timestamp: Optional[int] = None,
        game_result: Optional[Dict[str, Any]] = None,
        validator_hotkey: Optional[str] = None,
        block_number: Optional[int] = None,
        ttl_days: int = 90,
    ) -> Dict[str, Any]:
        """Save a match record.

        Args:
            match_uuid: Unique match identifier
            env: Environment name
            match_type: Type of match ("pairwise" or "game")
            task_id: Task/game ID
            participants: List of participant records with outcomes and ELO changes
            timestamp: Match timestamp in milliseconds (defaults to now)
            game_result: Game-specific result data
            validator_hotkey: Validator that executed the match
            block_number: Block number at execution
            ttl_days: Days until auto-deletion (default 90)

        Returns:
            Saved item
        """
        if timestamp is None:
            timestamp = int(time.time() * 1000)

        # Compress game result if present
        game_result_compressed = None
        if game_result:
            game_result_json = json.dumps(game_result, separators=(",", ":"))
            game_result_compressed = self.compress_data(game_result_json)

        item = {
            "pk": self._make_pk(env),
            "sk": self._make_sk(timestamp, match_uuid),
            "match_uuid": match_uuid,
            "env": env,
            "match_type": match_type,
            "task_id": task_id,
            "timestamp": timestamp,
            "gsi_partition": "MATCH",  # Fixed value for timestamp-index GSI
            "participants": participants,
            "validator_hotkey": validator_hotkey,
            "block_number": block_number,
            "ttl": self.get_ttl(ttl_days),
        }

        if game_result_compressed:
            item["game_result_compressed"] = game_result_compressed

        return await self.put(item)

    async def get_match(
        self,
        env: str,
        timestamp: int,
        match_uuid: str,
        include_game_result: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Get a specific match by its key.

        Args:
            env: Environment name
            timestamp: Match timestamp
            match_uuid: Match UUID
            include_game_result: Whether to decompress game_result

        Returns:
            Match record or None if not found
        """
        pk = self._make_pk(env)
        sk = self._make_sk(timestamp, match_uuid)
        item = await self.get(pk, sk)

        if item and include_game_result and "game_result_compressed" in item:
            try:
                game_result_json = self.decompress_data(item["game_result_compressed"])
                item["game_result"] = json.loads(game_result_json)
                del item["game_result_compressed"]
            except Exception as e:
                logger.warning(f"Failed to decompress game_result: {e}")

        return item

    async def get_matches_for_env(
        self,
        env: str,
        limit: int = 100,
        reverse: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get recent matches for an environment.

        Args:
            env: Environment name
            limit: Maximum number of matches to return
            reverse: If True, return newest first (default)

        Returns:
            List of match records
        """
        pk = self._make_pk(env)
        return await self.query(pk, sk_prefix="MATCH#", limit=limit, reverse=reverse)

    async def get_matches_in_time_range(
        self,
        since_timestamp: int,
        until_timestamp: Optional[int] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Get matches in a time range across all environments.

        Uses GSI timestamp-index for efficient time-range queries.

        Args:
            since_timestamp: Start timestamp (milliseconds)
            until_timestamp: End timestamp (milliseconds), defaults to now
            limit: Maximum number of matches to return

        Returns:
            List of match records
        """
        if until_timestamp is None:
            until_timestamp = int(time.time() * 1000)

        client = get_client()

        params = {
            "TableName": self.table_name,
            "IndexName": "timestamp-index",
            "KeyConditionExpression": "gsi_partition = :gsi AND #ts BETWEEN :start AND :end",
            "ExpressionAttributeNames": {"#ts": "timestamp"},
            "ExpressionAttributeValues": {
                ":gsi": {"S": "MATCH"},
                ":start": {"N": str(since_timestamp)},
                ":end": {"N": str(until_timestamp)},
            },
            "Limit": limit,
        }

        items = []
        while True:
            response = await client.query(**params)
            items.extend([self._deserialize(item) for item in response.get("Items", [])])

            if len(items) >= limit:
                break

            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            params["ExclusiveStartKey"] = last_key

        return items[:limit]

    async def get_matches_for_miner(
        self,
        miner_hotkey: str,
        env: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get matches involving a specific miner.

        Note: This requires scanning with a filter, which is less efficient.
        For frequent queries, consider adding a GSI on miner_hotkey.

        Args:
            miner_hotkey: Miner's hotkey
            env: Optional environment filter
            limit: Maximum number of matches to return

        Returns:
            List of match records involving the miner
        """
        client = get_client()

        if env:
            params = {
                "TableName": self.table_name,
                "KeyConditionExpression": "pk = :pk",
                "ExpressionAttributeValues": {
                    ":pk": {"S": self._make_pk(env)},
                },
                "ScanIndexForward": False,
            }
            all_items = []
            while len(all_items) < limit * 10:
                response = await client.query(**params)
                all_items.extend(response.get("Items", []))
                last_key = response.get("LastEvaluatedKey")
                if not last_key:
                    break
                params["ExclusiveStartKey"] = last_key
        else:
            params = {
                "TableName": self.table_name,
                "Limit": limit * 10,
            }
            response = await client.scan(**params)
            all_items = response.get("Items", [])

        # Python-side filter for nested participant hotkeys
        matches = []
        for item in all_items:
            deserialized = self._deserialize(item)
            participants = deserialized.get("participants", [])
            for p in participants:
                if p.get("miner_hotkey") == miner_hotkey:
                    matches.append(deserialized)
                    break
            if len(matches) >= limit:
                break

        return matches[:limit]

    async def count_matches_for_miner(
        self,
        miner_hotkey: str,
        env: Optional[str] = None,
    ) -> int:
        """Count total matches involving a miner (unfiltered by limit)."""
        client = get_client()

        if env:
            params = {
                "TableName": self.table_name,
                "KeyConditionExpression": "pk = :pk",
                "ExpressionAttributeValues": {
                    ":pk": {"S": self._make_pk(env)},
                },
            }
            all_items = []
            while True:
                response = await client.query(**params)
                all_items.extend(response.get("Items", []))
                last_key = response.get("LastEvaluatedKey")
                if not last_key:
                    break
                params["ExclusiveStartKey"] = last_key
        else:
            all_items = []
            params: dict = {"TableName": self.table_name}
            while True:
                response = await client.scan(**params)
                all_items.extend(response.get("Items", []))
                last_key = response.get("LastEvaluatedKey")
                if not last_key:
                    break
                params["ExclusiveStartKey"] = last_key

        count = 0
        for item in all_items:
            deserialized = self._deserialize(item)
            for p in deserialized.get("participants", []):
                if p.get("miner_hotkey") == miner_hotkey:
                    count += 1
                    break

        return count

    async def get_match_count_for_env(
        self,
        env: str,
    ) -> int:
        """Get total match count for an environment.

        Args:
            env: Environment name

        Returns:
            Number of matches
        """
        client = get_client()

        params = {
            "TableName": self.table_name,
            "KeyConditionExpression": "pk = :pk",
            "ExpressionAttributeValues": {":pk": {"S": self._make_pk(env)}},
            "Select": "COUNT",
        }

        total = 0
        while True:
            response = await client.query(**params)
            total += response.get("Count", 0)

            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            params["ExclusiveStartKey"] = last_key

        return total

    async def delete_match(
        self,
        env: str,
        timestamp: int,
        match_uuid: str,
    ) -> bool:
        """Delete a match record.

        Args:
            env: Environment name
            timestamp: Match timestamp
            match_uuid: Match UUID

        Returns:
            True if deleted, False otherwise
        """
        pk = self._make_pk(env)
        sk = self._make_sk(timestamp, match_uuid)
        return await self.delete(pk, sk)
