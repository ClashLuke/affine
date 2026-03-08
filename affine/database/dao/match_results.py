"""
Match Results DAO

Handles storage and retrieval of per-participant results for multi-party games.
"""

import time
import json
from typing import Dict, Any, List, Optional

from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name
from affine.database.client import get_client
from affine.core.setup import logger


class MatchResultsDAO(BaseDAO):
    """DAO for match_results table.

    Stores per-participant results for multi-party game matches.

    Schema:
    PK: MINER#{hotkey}#REV#{revision}#GAME#{game_type}
    SK: MATCH#{match_uuid}
    GSI: match-uuid-index (MATCH#{match_uuid}, SLOT#{slot}) for fetching all participants
    """

    def __init__(self):
        self.table_name = get_table_name("match_results")
        super().__init__()

    def _make_pk(self, miner_hotkey: str, model_revision: str, game_type: str) -> str:
        """Generate partition key."""
        return f"MINER#{miner_hotkey}#REV#{model_revision}#GAME#{game_type}"

    def _make_sk(self, match_uuid: str) -> str:
        """Generate sort key."""
        return f"MATCH#{match_uuid}"

    async def save_result(
        self,
        miner_hotkey: str,
        model_revision: str,
        model: str,
        game_type: str,
        match_uuid: str,
        task_id: int,
        slot: int,
        role: Optional[str],
        outcome: str,  # "win", "loss", "draw", "timeout", "error"
        score: float,
        opponent_hotkeys: List[str],
        total_moves: int = 0,
        avg_move_latency_ms: int = 0,
        total_latency_ms: int = 0,
        move_history: Optional[List[Dict]] = None,
        elo_before: Optional[float] = None,
        elo_after: Optional[float] = None,
        validator_hotkey: Optional[str] = None,
        block_number: Optional[int] = None,
        timestamp: Optional[int] = None,
        ttl_days: int = 30,
    ) -> Dict[str, Any]:
        """Save a match result for a participant.

        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision hash
            model: Model repo/name
            game_type: Type of game
            match_uuid: Match UUID
            task_id: Task/game ID
            slot: Player slot (0, 1, 2, ...)
            role: Game-specific role
            outcome: Match outcome
            score: Normalized score (0.0-1.0)
            opponent_hotkeys: List of opponent hotkeys
            total_moves: Total moves made
            avg_move_latency_ms: Average move latency
            total_latency_ms: Total game latency
            move_history: List of moves (compressed)
            elo_before: ELO rating before match
            elo_after: ELO rating after match
            validator_hotkey: Validator that executed the match
            block_number: Block number
            timestamp: Match timestamp (defaults to now)
            ttl_days: Days until auto-deletion (default 30)

        Returns:
            Saved item
        """
        if timestamp is None:
            timestamp = int(time.time() * 1000)

        # Compress move history if present
        move_history_compressed = None
        if move_history:
            move_history_json = json.dumps(move_history, separators=(",", ":"))
            move_history_compressed = self.compress_data(move_history_json)

        item = {
            "pk": self._make_pk(miner_hotkey, model_revision, game_type),
            "sk": self._make_sk(match_uuid),
            "match_uuid": match_uuid,
            "game_type": game_type,
            "task_id": task_id,
            "miner_hotkey": miner_hotkey,
            "model_revision": model_revision,
            "model": model,
            "slot": slot,
            "role": role,
            "outcome": outcome,
            "score": score,
            "opponent_hotkeys": opponent_hotkeys,
            "total_moves": total_moves,
            "avg_move_latency_ms": avg_move_latency_ms,
            "total_latency_ms": total_latency_ms,
            "elo_before": elo_before,
            "elo_after": elo_after,
            "timestamp": timestamp,
            "validator_hotkey": validator_hotkey,
            "block_number": block_number,
            "gsi1_pk": f"MATCH#{match_uuid}",
            "gsi1_sk": f"SLOT#{slot}",
            "ttl": self.get_ttl(ttl_days),
        }

        if move_history_compressed:
            item["move_history_compressed"] = move_history_compressed

        return await self.put(item)

    async def save_match_results(
        self,
        match_uuid: str,
        game_type: str,
        task_id: int,
        participants: List[Dict[str, Any]],
        validator_hotkey: Optional[str] = None,
        block_number: Optional[int] = None,
        timestamp: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Save results for all participants in a match.

        Args:
            match_uuid: Match UUID
            game_type: Type of game
            task_id: Task/game ID
            participants: List of participant result dicts
            validator_hotkey: Validator that executed the match
            block_number: Block number
            timestamp: Match timestamp

        Returns:
            List of saved items
        """
        if timestamp is None:
            timestamp = int(time.time() * 1000)

        results = []
        for p in participants:
            result = await self.save_result(
                miner_hotkey=p["miner_hotkey"],
                model_revision=p["model_revision"],
                model=p.get("model", ""),
                game_type=game_type,
                match_uuid=match_uuid,
                task_id=task_id,
                slot=p.get("slot", 0),
                role=p.get("role"),
                outcome=p["outcome"],
                score=p.get("score", 0.0),
                opponent_hotkeys=p.get("opponent_hotkeys", []),
                total_moves=p.get("total_moves", 0),
                avg_move_latency_ms=p.get("avg_move_latency_ms", 0),
                total_latency_ms=p.get("total_latency_ms", 0),
                move_history=p.get("move_history"),
                elo_before=p.get("elo_before"),
                elo_after=p.get("elo_after"),
                validator_hotkey=validator_hotkey,
                block_number=block_number,
                timestamp=timestamp,
            )
            results.append(result)

        return results

    async def get_result(
        self,
        miner_hotkey: str,
        model_revision: str,
        game_type: str,
        match_uuid: str,
        include_move_history: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Get a specific match result.

        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision hash
            game_type: Type of game
            match_uuid: Match UUID
            include_move_history: Whether to decompress move history

        Returns:
            Result record or None if not found
        """
        pk = self._make_pk(miner_hotkey, model_revision, game_type)
        sk = self._make_sk(match_uuid)
        item = await self.get(pk, sk)

        if item and include_move_history and "move_history_compressed" in item:
            try:
                move_history_json = self.decompress_data(item["move_history_compressed"])
                item["move_history"] = json.loads(move_history_json)
                del item["move_history_compressed"]
            except Exception as e:
                logger.warning(f"Failed to decompress move_history: {e}")

        return item

    async def get_results_by_match(
        self,
        match_uuid: str,
    ) -> List[Dict[str, Any]]:
        """Get all participant results for a match.

        Uses GSI match-uuid-index for efficient lookup.

        Args:
            match_uuid: Match UUID

        Returns:
            List of result records for all participants
        """
        client = get_client()

        params = {
            "TableName": self.table_name,
            "IndexName": "match-uuid-index",
            "KeyConditionExpression": "gsi1_pk = :match",
            "ExpressionAttributeValues": {
                ":match": {"S": f"MATCH#{match_uuid}"},
            },
        }

        response = await client.query(**params)
        return [self._deserialize(item) for item in response.get("Items", [])]

    async def get_miner_match_history(
        self,
        miner_hotkey: str,
        model_revision: str,
        game_type: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get a miner's match history for a game type.

        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision hash
            game_type: Type of game
            limit: Maximum number of results

        Returns:
            List of match results, newest first
        """
        pk = self._make_pk(miner_hotkey, model_revision, game_type)
        return await self.query(pk, limit=limit, reverse=True)

    async def get_miner_stats(
        self,
        miner_hotkey: str,
        model_revision: str,
        game_type: str,
    ) -> Dict[str, Any]:
        """Get aggregated stats for a miner in a game type.

        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision hash
            game_type: Type of game

        Returns:
            Aggregated statistics
        """
        results = await self.get_miner_match_history(
            miner_hotkey, model_revision, game_type, limit=1000
        )

        if not results:
            return {
                "matches_played": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "win_rate": 0.0,
                "avg_score": 0.0,
            }

        wins = sum(1 for r in results if r.get("outcome") == "win")
        losses = sum(1 for r in results if r.get("outcome") == "loss")
        draws = sum(1 for r in results if r.get("outcome") == "draw")
        total = len(results)
        avg_score = sum(r.get("score", 0) for r in results) / total if total > 0 else 0

        return {
            "matches_played": total,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / total if total > 0 else 0.0,
            "avg_score": avg_score,
        }

    async def get_recent_results(
        self,
        game_type: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get recent results across all miners for a game type.

        Note: This requires a scan, which is less efficient.

        Args:
            game_type: Type of game
            limit: Maximum number of results

        Returns:
            List of recent results
        """
        client = get_client()

        params = {
            "TableName": self.table_name,
            "FilterExpression": "game_type = :game_type",
            "ExpressionAttributeValues": {
                ":game_type": {"S": game_type},
            },
            "Limit": limit * 5,  # Over-fetch due to filter
        }

        response = await client.scan(**params)
        items = [self._deserialize(item) for item in response.get("Items", [])]

        # Sort by timestamp descending
        items.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

        return items[:limit]

    async def delete_result(
        self,
        miner_hotkey: str,
        model_revision: str,
        game_type: str,
        match_uuid: str,
    ) -> bool:
        """Delete a match result.

        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision hash
            game_type: Type of game
            match_uuid: Match UUID

        Returns:
            True if deleted, False otherwise
        """
        pk = self._make_pk(miner_hotkey, model_revision, game_type)
        sk = self._make_sk(match_uuid)
        return await self.delete(pk, sk)
