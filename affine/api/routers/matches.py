"""
Matches Router

Endpoints for managing multi-party game matches.
"""

import time
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from affine.api.dependencies import verify_executor_auth, get_auth_service
from affine.api.services.auth import AuthService
from affine.api.config import config
from affine.database.dao.match_pool import MatchPoolDAO
from affine.database.dao.match_results import MatchResultsDAO
from affine.database.dao.match_records import MatchRecordsDAO
from affine.database.dao.elo_ratings import EloRatingsDAO
from affine.src.elo import EloCalculator, MatchEngine
from affine.src.elo.config import EloConfig

from affine.core.setup import logger

router = APIRouter(prefix="/matches", tags=["Matches"])


# =============================================================================
# Request/Response Models
# =============================================================================


class ParticipantResult(BaseModel):
    """Result for a single participant in a match."""

    slot: int = Field(..., description="Player slot (0, 1, ...)")
    miner_hotkey: str = Field(..., description="Miner's hotkey")
    model_revision: str = Field(..., description="Model revision hash")
    outcome: str = Field(..., description="Match outcome: win, loss, draw, timeout, error")
    final_rank: Optional[int] = Field(None, description="Final rank (1=best). If absent, derived from outcome.")
    score: float = Field(0.0, description="Normalized score (0.0-1.0)")
    total_moves: int = Field(0, description="Total moves made")
    avg_move_latency_ms: int = Field(0, description="Average move latency in ms")
    total_latency_ms: int = Field(0, description="Total game latency in ms")


class MatchSubmission(BaseModel):
    """Match result submission from executor."""

    match_uuid: str = Field(..., description="Match UUID")
    game_type: str = Field(..., description="Game type (e.g., tictactoe, chess)")
    task_id: int = Field(..., description="Task/game ID")
    participants: List[ParticipantResult] = Field(..., description="Results for all participants")
    game_history: Optional[List[Dict[str, Any]]] = Field(None, description="Move history")
    total_moves: int = Field(0, description="Total moves in the game")
    total_time_ms: int = Field(0, description="Total game time in ms")
    signature: str = Field(..., description="Executor's signature")
    signed_message: str = Field("", description="Message that was signed (match_uuid:task_id:timestamp)")


class MatchFetchResponse(BaseModel):
    """Response for match fetch endpoint."""

    matches: List[Dict[str, Any]] = Field(default_factory=list, description="List of matches")


class MatchSubmitResponse(BaseModel):
    """Response for match submission endpoint."""

    success: bool = Field(..., description="Whether submission was successful")
    message: str = Field("", description="Status message")
    elo_updates: Optional[List[Dict[str, Any]]] = Field(None, description="ELO rating changes")


class PoolStatsResponse(BaseModel):
    """Response for pool statistics endpoint."""

    pending: int = Field(0)
    executing: int = Field(0)
    completed: int = Field(0)
    failed: int = Field(0)


# =============================================================================
# Dependencies
# =============================================================================


def get_match_pool_dao() -> MatchPoolDAO:
    """Get MatchPoolDAO instance."""
    return MatchPoolDAO()


def get_match_results_dao() -> MatchResultsDAO:
    """Get MatchResultsDAO instance."""
    return MatchResultsDAO()


def get_match_records_dao() -> MatchRecordsDAO:
    """Get MatchRecordsDAO instance."""
    return MatchRecordsDAO()


def get_elo_ratings_dao() -> EloRatingsDAO:
    """Get EloRatingsDAO instance."""
    return EloRatingsDAO()


def get_match_engine() -> MatchEngine:
    """Get MatchEngine instance."""
    return MatchEngine(EloConfig())


# =============================================================================
# Endpoints
# =============================================================================


if config.SERVICES_ENABLED:

    @router.post("/fetch", response_model=MatchFetchResponse)
    async def fetch_match(
        game_type: Optional[str] = Query(None, description="Game type filter"),
        executor_hotkey: str = Depends(verify_executor_auth),
        match_pool: MatchPoolDAO = Depends(get_match_pool_dao),
    ):
        """
        Fetch a pending match for execution.

        The match is atomically assigned to the executor (status changes to 'executing').

        Headers:
        - X-Hotkey: Executor's SS58 hotkey
        - X-Signature: Hex-encoded signature of timestamp
        - X-Message: Unix timestamp

        Query Parameters:
        - game_type: Optional filter by game type

        Returns:
        - MatchFetchResponse with assigned match (or empty list if none available)
        """
        try:
            match = await match_pool.fetch_and_assign_match(
                executor_hotkey=executor_hotkey,
                game_type=game_type,
            )

            if not match:
                logger.debug(f"No available matches for executor {executor_hotkey[:16]}...")
                return MatchFetchResponse(matches=[])

            logger.info(
                f"Assigned match {match['match_uuid'][:8]}... to executor {executor_hotkey[:16]}..."
            )

            return MatchFetchResponse(matches=[match])

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error fetching match: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to fetch match: {str(e)}",
            )

    @router.post("/submit", response_model=MatchSubmitResponse)
    async def submit_match_result(
        submission: MatchSubmission,
        executor_hotkey: str = Depends(verify_executor_auth),
        auth_service: AuthService = Depends(get_auth_service),
        match_pool: MatchPoolDAO = Depends(get_match_pool_dao),
        match_results: MatchResultsDAO = Depends(get_match_results_dao),
        match_records: MatchRecordsDAO = Depends(get_match_records_dao),
        elo_ratings: EloRatingsDAO = Depends(get_elo_ratings_dao),
        match_engine: MatchEngine = Depends(get_match_engine),
    ):
        """
        Submit match results from executor.

        This endpoint:
        1. Validates the submission (ownership check)
        2. Marks the match as completed in the pool
        3. Saves results for all participants
        4. Updates ELO ratings for all participants
        5. Records the match in match history

        Headers:
        - X-Hotkey: Executor's SS58 hotkey
        - X-Signature: Hex-encoded signature of timestamp
        - X-Message: Unix timestamp

        Returns:
        - MatchSubmitResponse with success status and ELO updates
        """
        try:
            timestamp = int(time.time() * 1000)
            env = f"game:{submission.game_type}"

            # Validate body signature
            if submission.signed_message and submission.signature:
                # Verify signed_message format matches submission fields
                parts = submission.signed_message.split(":")
                if len(parts) != 3 or parts[0] != submission.match_uuid or parts[1] != str(submission.task_id):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="signed_message does not match submission fields",
                    )
                try:
                    sign_ts = int(parts[2])
                    if abs(int(time.time()) - sign_ts) > 300:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Submission signature expired",
                        )
                except ValueError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid timestamp in signed_message",
                    )

                if not auth_service.verify_signature(
                    message=submission.signed_message,
                    signature=submission.signature,
                    hotkey=executor_hotkey,
                ):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Invalid submission signature",
                    )
            else:
                logger.warning(
                    f"Match {submission.match_uuid[:8]}... submitted without body signature"
                )

            executing_match = await match_pool.get_match(
                game_type=submission.game_type,
                status=match_pool.STATUS_EXECUTING,
                match_uuid=submission.match_uuid,
            )
            if executing_match and executing_match.get("assigned_to") != executor_hotkey:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Match is not assigned to this executor",
                )

            completed = await match_pool.complete_match(
                game_type=submission.game_type,
                match_uuid=submission.match_uuid,
                success=True,
            )

            if not completed:
                logger.warning(f"Match not found for completion: {submission.match_uuid}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Match not found or already completed",
                )

            participants_data = []
            for p in submission.participants:
                rating = await elo_ratings.get_rating(p.miner_hotkey, p.model_revision, env)
                current_rating = rating.get("rating", 1500) if rating else 1500
                matches_played = rating.get("matches_played", 0) if rating else 0

                if hasattr(p, 'final_rank') and p.final_rank is not None:
                    final_rank = p.final_rank
                elif p.outcome == "win":
                    final_rank = 1
                elif p.outcome == "draw":
                    final_rank = 2
                elif p.outcome in ("timeout", "error"):
                    final_rank = len(submission.participants)
                else:
                    final_rank = len(submission.participants)

                participants_data.append(
                    {
                        "hotkey": p.miner_hotkey,
                        "revision": p.model_revision,
                        "rating": current_rating,
                        "matches_played": matches_played,
                        "final_rank": final_rank,
                        "slot": p.slot,
                    }
                )

            if len(participants_data) == 2:
                p1, p2 = participants_data[0], participants_data[1]
                sub_p1, sub_p2 = submission.participants[0], submission.participants[1]

                if sub_p1.outcome == "win":
                    outcome = "a_wins"
                elif sub_p2.outcome == "win":
                    outcome = "b_wins"
                else:
                    outcome = "draw"

                match_result = match_engine.process_head_to_head_game(
                    env=env,
                    task_id=submission.task_id,
                    player_a=p1,
                    player_b=p2,
                    outcome=outcome,
                    game_result={"history": submission.game_history},
                    validator_hotkey=executor_hotkey,
                )
            else:
                match_result = match_engine.process_multi_party_game(
                    env=env,
                    task_id=submission.task_id,
                    participants_data=participants_data,
                    game_result={"history": submission.game_history},
                    validator_hotkey=executor_hotkey,
                )

            elo_updates = []
            for participant in match_result.participants:
                await elo_ratings.update_rating_after_match(
                    miner_hotkey=participant.miner_hotkey,
                    model_revision=participant.model_revision,
                    env=env,
                    new_rating=participant.elo_after,
                    outcome=participant.outcome.value,
                    match_timestamp=timestamp,
                )

                elo_updates.append(
                    {
                        "miner_hotkey": participant.miner_hotkey,
                        "elo_before": float(participant.elo_before) if participant.elo_before else None,
                        "elo_after": float(participant.elo_after) if participant.elo_after else None,
                        "elo_change": float(participant.elo_change) if participant.elo_change else None,
                        "outcome": participant.outcome.value,
                    }
                )

            participants_for_save = []
            for i, p in enumerate(submission.participants):
                opponent_hotkeys = [
                    op.miner_hotkey for j, op in enumerate(submission.participants) if j != i
                ]
                participants_for_save.append(
                    {
                        "miner_hotkey": p.miner_hotkey,
                        "model_revision": p.model_revision,
                        "model": "",
                        "slot": p.slot,
                        "outcome": p.outcome,
                        "score": p.score,
                        "opponent_hotkeys": opponent_hotkeys,
                        "total_moves": p.total_moves,
                        "avg_move_latency_ms": p.avg_move_latency_ms,
                        "total_latency_ms": p.total_latency_ms,
                        "move_history": submission.game_history,
                        "elo_before": elo_updates[i]["elo_before"] if i < len(elo_updates) else None,
                        "elo_after": elo_updates[i]["elo_after"] if i < len(elo_updates) else None,
                    }
                )

            await match_results.save_match_results(
                match_uuid=submission.match_uuid,
                game_type=submission.game_type,
                task_id=submission.task_id,
                participants=participants_for_save,
                validator_hotkey=executor_hotkey,
                timestamp=timestamp,
            )

            await match_records.save_match(
                match_uuid=submission.match_uuid,
                env=env,
                match_type="game",
                task_id=submission.task_id,
                participants=[p.to_dict() for p in match_result.participants],
                timestamp=timestamp,
                game_result=match_result.game_result,
                validator_hotkey=executor_hotkey,
            )

            logger.info(
                f"Match {submission.match_uuid[:8]}... completed: "
                f"{len(elo_updates)} ELO updates"
            )

            return MatchSubmitResponse(
                success=True,
                message="Match results submitted successfully",
                elo_updates=elo_updates,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error submitting match result: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to submit match result: {str(e)}",
            )

    @router.get("/pool/stats", response_model=PoolStatsResponse)
    async def get_pool_stats(
        game_type: Optional[str] = Query(None, description="Game type filter"),
        match_pool: MatchPoolDAO = Depends(get_match_pool_dao),
    ):
        """
        Get match pool statistics.

        Query Parameters:
        - game_type: Optional filter by game type

        Returns:
        - PoolStatsResponse with counts by status
        """
        try:
            stats = await match_pool.get_pool_stats(game_type)
            return PoolStatsResponse(**stats)
        except Exception as e:
            logger.error(f"Error getting pool stats: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get pool stats: {str(e)}",
            )
