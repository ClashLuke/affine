"""
ELO Router

Endpoints for ELO ratings and leaderboards.
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from affine.api.dependencies import rate_limit_read
from affine.database.dao.elo_ratings import EloRatingsDAO
from affine.database.dao.match_records import MatchRecordsDAO

from affine.core.setup import logger

router = APIRouter(prefix="/elo", tags=["ELO"])


# =============================================================================
# Response Models
# =============================================================================


class EloRatingResponse(BaseModel):
    """ELO rating for a miner."""

    miner_hotkey: str
    model_revision: str
    env: str
    rating: float
    peak_rating: float
    matches_played: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    last_match_at: Optional[int] = None


class LeaderboardEntry(BaseModel):
    """Entry in the ELO leaderboard."""

    rank: int
    miner_hotkey: str
    model_revision: str
    rating: float
    matches_played: int
    wins: int
    losses: int
    draws: int
    win_rate: float


class LeaderboardResponse(BaseModel):
    """ELO leaderboard response."""

    env: str
    entries: List[LeaderboardEntry]
    total_players: int


class MatchHistoryEntry(BaseModel):
    """Entry in match history."""

    match_uuid: str
    env: str
    match_type: str
    task_id: int
    timestamp: int
    outcome: str
    elo_before: Optional[float] = None
    elo_after: Optional[float] = None
    elo_change: Optional[float] = None
    opponent_hotkeys: List[str] = []


class MatchHistoryResponse(BaseModel):
    """Match history response."""

    miner_hotkey: str
    model_revision: str
    matches: List[MatchHistoryEntry]
    total_matches: int


# =============================================================================
# Dependencies
# =============================================================================


def get_elo_ratings_dao() -> EloRatingsDAO:
    """Get EloRatingsDAO instance."""
    return EloRatingsDAO()


def get_match_records_dao() -> MatchRecordsDAO:
    """Get MatchRecordsDAO instance."""
    return MatchRecordsDAO()


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/ratings/{hotkey}", response_model=List[EloRatingResponse])
async def get_miner_ratings(
    hotkey: str,
    model_revision: str = Query(..., description="Model revision hash"),
    elo_ratings: EloRatingsDAO = Depends(get_elo_ratings_dao),
    _: None = Depends(rate_limit_read),
):
    """
    Get all ELO ratings for a miner across environments.

    Path Parameters:
    - hotkey: Miner's SS58 hotkey

    Query Parameters:
    - model_revision: Model revision hash

    Returns:
    - List of ELO ratings for all environments
    """
    try:
        ratings = await elo_ratings.get_all_ratings_for_miner(hotkey, model_revision)

        result = []
        for r in ratings:
            matches_played = r.get("matches_played", 0)
            wins = r.get("wins", 0)

            result.append(
                EloRatingResponse(
                    miner_hotkey=r["miner_hotkey"],
                    model_revision=r["model_revision"],
                    env=r["env"],
                    rating=float(r.get("rating", 1500)),
                    peak_rating=float(r.get("peak_rating", 1500)),
                    matches_played=matches_played,
                    wins=wins,
                    losses=r.get("losses", 0),
                    draws=r.get("draws", 0),
                    win_rate=wins / matches_played if matches_played > 0 else 0.0,
                    last_match_at=r.get("last_match_at"),
                )
            )

        return result

    except Exception as e:
        logger.error(f"Error getting miner ratings: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get miner ratings: {str(e)}",
        )


@router.get("/ratings/{hotkey}/{env}", response_model=EloRatingResponse)
async def get_miner_rating_for_env(
    hotkey: str,
    env: str,
    model_revision: str = Query(..., description="Model revision hash"),
    elo_ratings: EloRatingsDAO = Depends(get_elo_ratings_dao),
    _: None = Depends(rate_limit_read),
):
    """
    Get ELO rating for a miner in a specific environment.

    Path Parameters:
    - hotkey: Miner's SS58 hotkey
    - env: Environment name (e.g., "game:tictactoe")

    Query Parameters:
    - model_revision: Model revision hash

    Returns:
    - ELO rating for the specified environment
    """
    try:
        rating = await elo_ratings.get_rating(hotkey, model_revision, env)

        if not rating:
            # Return default rating if not found
            return EloRatingResponse(
                miner_hotkey=hotkey,
                model_revision=model_revision,
                env=env,
                rating=1500.0,
                peak_rating=1500.0,
                matches_played=0,
                wins=0,
                losses=0,
                draws=0,
                win_rate=0.0,
            )

        matches_played = rating.get("matches_played", 0)
        wins = rating.get("wins", 0)

        return EloRatingResponse(
            miner_hotkey=rating["miner_hotkey"],
            model_revision=rating["model_revision"],
            env=rating["env"],
            rating=float(rating.get("rating", 1500)),
            peak_rating=float(rating.get("peak_rating", 1500)),
            matches_played=matches_played,
            wins=wins,
            losses=rating.get("losses", 0),
            draws=rating.get("draws", 0),
            win_rate=wins / matches_played if matches_played > 0 else 0.0,
            last_match_at=rating.get("last_match_at"),
        )

    except Exception as e:
        logger.error(f"Error getting miner rating: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get miner rating: {str(e)}",
        )


@router.get("/leaderboard/{env}", response_model=LeaderboardResponse)
async def get_leaderboard(
    env: str,
    limit: int = Query(100, ge=1, le=256, description="Maximum number of entries"),
    elo_ratings: EloRatingsDAO = Depends(get_elo_ratings_dao),
    _: None = Depends(rate_limit_read),
):
    """
    Get ELO leaderboard for an environment.

    Path Parameters:
    - env: Environment name (e.g., "game:tictactoe", "game:chess")

    Query Parameters:
    - limit: Maximum number of entries (default 100, max 256)

    Returns:
    - Leaderboard with ranked entries
    """
    try:
        ratings = await elo_ratings.get_leaderboard(env, limit)

        entries = []
        for rank, r in enumerate(ratings, 1):
            matches_played = r.get("matches_played", 0)
            wins = r.get("wins", 0)

            entries.append(
                LeaderboardEntry(
                    rank=rank,
                    miner_hotkey=r["miner_hotkey"],
                    model_revision=r["model_revision"],
                    rating=float(r.get("rating", 1500)),
                    matches_played=matches_played,
                    wins=wins,
                    losses=r.get("losses", 0),
                    draws=r.get("draws", 0),
                    win_rate=wins / matches_played if matches_played > 0 else 0.0,
                )
            )

        # Get total player count
        all_ratings = await elo_ratings.get_all_ratings_for_env(env)

        return LeaderboardResponse(
            env=env,
            entries=entries,
            total_players=len(all_ratings),
        )

    except Exception as e:
        logger.error(f"Error getting leaderboard: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get leaderboard: {str(e)}",
        )


@router.get("/history/{hotkey}", response_model=MatchHistoryResponse)
async def get_match_history(
    hotkey: str,
    model_revision: str = Query(..., description="Model revision hash"),
    env: Optional[str] = Query(None, description="Environment filter"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of matches"),
    match_records: MatchRecordsDAO = Depends(get_match_records_dao),
    _: None = Depends(rate_limit_read),
):
    """
    Get match history for a miner.

    Path Parameters:
    - hotkey: Miner's SS58 hotkey

    Query Parameters:
    - model_revision: Model revision hash
    - env: Optional environment filter
    - limit: Maximum number of matches (default 50, max 200)

    Returns:
    - Match history with outcomes and ELO changes
    """
    try:
        matches = await match_records.get_matches_for_miner(hotkey, env, limit)

        entries = []
        for m in matches:
            # Find this miner's participant entry, filtering by requested revision
            participant = None
            opponent_hotkeys = []
            for p in m.get("participants", []):
                if p.get("miner_hotkey") == hotkey:
                    if p.get("model_revision") == model_revision:
                        participant = p
                else:
                    opponent_hotkeys.append(p.get("miner_hotkey", ""))

            if participant:
                entries.append(
                    MatchHistoryEntry(
                        match_uuid=m["match_uuid"],
                        env=m["env"],
                        match_type=m.get("match_type", "game"),
                        task_id=m.get("task_id", 0),
                        timestamp=m.get("timestamp", 0),
                        outcome=participant.get("outcome", "unknown"),
                        elo_before=participant.get("elo_before"),
                        elo_after=participant.get("elo_after"),
                        elo_change=participant.get("elo_change"),
                        opponent_hotkeys=opponent_hotkeys,
                    )
                )

        total = await match_records.count_matches_for_miner(hotkey, env)

        return MatchHistoryResponse(
            miner_hotkey=hotkey,
            model_revision=model_revision,
            matches=entries,
            total_matches=total,
        )

    except Exception as e:
        logger.error(f"Error getting match history: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get match history: {str(e)}",
        )
