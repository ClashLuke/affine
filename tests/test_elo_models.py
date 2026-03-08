import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "affine", "src"))

import pytest
from decimal import Decimal

from elo.models import EloRating, MatchParticipant, MatchResult, MatchOutcome, MatchType


class TestMatchOutcome:
    def test_outcome_values(self):
        assert MatchOutcome.WIN.value == "win"
        assert MatchOutcome.LOSS.value == "loss"
        assert MatchOutcome.DRAW.value == "draw"
        assert MatchOutcome.TIMEOUT.value == "timeout"
        assert MatchOutcome.ERROR.value == "error"

    def test_outcome_from_string(self):
        assert MatchOutcome("win") == MatchOutcome.WIN
        assert MatchOutcome("loss") == MatchOutcome.LOSS


class TestMatchType:
    def test_match_types(self):
        assert MatchType.PAIRWISE.value == "pairwise"
        assert MatchType.GAME.value == "game"


class TestEloRating:
    def test_create_rating(self):
        rating = EloRating(
            miner_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            model_revision="abc123", env="game:tictactoe",
            rating=Decimal("1523"), peak_rating=Decimal("1550"),
            matches_played=25, wins=15, losses=8, draws=2,
        )
        assert rating.miner_hotkey == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        assert rating.rating == Decimal("1523")
        assert rating.matches_played == 25

    def test_win_rate(self):
        rating = EloRating(
            miner_hotkey="test", model_revision="v1", env="test",
            rating=Decimal("1500"), peak_rating=Decimal("1500"),
            matches_played=10, wins=7, losses=2, draws=1,
        )
        assert rating.win_rate == 0.7

    def test_win_rate_no_matches(self):
        rating = EloRating(
            miner_hotkey="test", model_revision="v1", env="test",
            rating=Decimal("1500"), peak_rating=Decimal("1500"),
            matches_played=0, wins=0, losses=0, draws=0,
        )
        assert rating.win_rate == 0.0

    def test_to_dict(self):
        rating = EloRating(
            miner_hotkey="test_hotkey", model_revision="v1", env="game:chess",
            rating=Decimal("1600"), peak_rating=Decimal("1650"),
            matches_played=50, wins=30, losses=15, draws=5, last_match_at=1705000000,
        )
        data = rating.to_dict()
        assert data["miner_hotkey"] == "test_hotkey"
        assert data["rating"] == 1600.0
        assert data["matches_played"] == 50
        assert data["last_match_at"] == 1705000000

    def test_from_dict(self):
        data = {
            "miner_hotkey": "test_hotkey", "model_revision": "v1", "env": "game:chess",
            "rating": "1600", "peak_rating": "1650",
            "matches_played": 50, "wins": 30, "losses": 15, "draws": 5,
            "last_match_at": 1705000000,
        }
        rating = EloRating.from_dict(data)
        assert rating.miner_hotkey == "test_hotkey"
        assert rating.rating == Decimal("1600")
        assert rating.peak_rating == Decimal("1650")

    def test_roundtrip_serialization(self):
        original = EloRating(
            miner_hotkey="test", model_revision="v1", env="test",
            rating=Decimal("1523.45"), peak_rating=Decimal("1600.00"),
            matches_played=100, wins=60, losses=30, draws=10, last_match_at=1705000000,
        )
        restored = EloRating.from_dict(original.to_dict())
        assert restored.miner_hotkey == original.miner_hotkey
        assert restored.rating == original.rating
        assert restored.peak_rating == original.peak_rating
        assert restored.matches_played == original.matches_played
        assert restored.wins == original.wins
        assert restored.last_match_at == original.last_match_at


class TestMatchParticipant:
    def test_create_participant(self):
        p = MatchParticipant(miner_hotkey="test_hotkey", model_revision="v1", slot=0, role="player_x")
        assert p.miner_hotkey == "test_hotkey"
        assert p.slot == 0
        assert p.role == "player_x"
        assert p.outcome is None

    def test_participant_with_results(self):
        p = MatchParticipant(
            miner_hotkey="test_hotkey", model_revision="v1", slot=0,
            outcome=MatchOutcome.WIN, elo_before=Decimal("1500"),
            elo_after=Decimal("1520"), elo_change=Decimal("20"),
        )
        assert p.outcome == MatchOutcome.WIN
        assert p.elo_change == Decimal("20")

    def test_to_dict(self):
        p = MatchParticipant(
            miner_hotkey="test", model_revision="v1", slot=1, role="player_o",
            raw_score=Decimal("0.75"), outcome=MatchOutcome.WIN,
            elo_before=Decimal("1500"), elo_after=Decimal("1525"), elo_change=Decimal("25"),
        )
        data = p.to_dict()
        assert data["miner_hotkey"] == "test"
        assert data["slot"] == 1
        assert data["outcome"] == "win"
        assert data["elo_change"] == 25.0

    def test_from_dict(self):
        data = {
            "miner_hotkey": "test", "model_revision": "v1", "slot": 0, "role": "attacker",
            "outcome": "loss", "elo_before": "1600", "elo_after": "1575", "elo_change": "-25",
        }
        p = MatchParticipant.from_dict(data)
        assert p.outcome == MatchOutcome.LOSS
        assert p.elo_change == Decimal("-25")


class TestMatchResult:
    def test_create_match_result(self):
        result = MatchResult(
            match_uuid="test-match-001", env="game:tictactoe",
            match_type=MatchType.GAME, task_id=12345, timestamp=1705000000,
            participants=[
                MatchParticipant(miner_hotkey="player1", model_revision="v1", slot=0, outcome=MatchOutcome.WIN),
                MatchParticipant(miner_hotkey="player2", model_revision="v1", slot=1, outcome=MatchOutcome.LOSS),
            ],
        )
        assert result.match_uuid == "test-match-001"
        assert result.match_type == MatchType.GAME
        assert len(result.participants) == 2

    def test_match_result_with_game_result(self):
        result = MatchResult(
            match_uuid="test-match-002", env="game:chess",
            match_type=MatchType.GAME, task_id=12346, timestamp=1705000000,
            participants=[],
            game_result={"moves": ["e4", "e5", "Nf3"], "duration_ms": 300000, "termination": "checkmate"},
        )
        assert result.game_result["termination"] == "checkmate"

    def test_to_dict(self):
        result = MatchResult(
            match_uuid="test-uuid", env="game:test",
            match_type=MatchType.GAME, task_id=1, timestamp=1705000000,
            participants=[
                MatchParticipant(
                    miner_hotkey="p1", model_revision="v1", slot=0,
                    outcome=MatchOutcome.WIN, elo_before=Decimal("1500"),
                    elo_after=Decimal("1520"), elo_change=Decimal("20"),
                ),
            ],
            validator_hotkey="validator1",
        )
        data = result.to_dict()
        assert data["match_uuid"] == "test-uuid"
        assert data["match_type"] == "game"
        assert data["participants"][0]["outcome"] == "win"

    def test_from_dict(self):
        data = {
            "match_uuid": "test-uuid", "env": "game:test", "match_type": "pairwise",
            "task_id": 123, "timestamp": 1705000000,
            "participants": [
                {"miner_hotkey": "p1", "model_revision": "v1", "slot": 0, "outcome": "win"},
                {"miner_hotkey": "p2", "model_revision": "v1", "slot": 1, "outcome": "loss"},
            ],
        }
        result = MatchResult.from_dict(data)
        assert result.match_type == MatchType.PAIRWISE
        assert result.participants[0].outcome == MatchOutcome.WIN


class TestDefaultValues:
    def test_elo_rating_defaults(self):
        rating = EloRating(
            miner_hotkey="test", model_revision="v1", env="test",
            rating=Decimal("1500"), peak_rating=Decimal("1500"),
            matches_played=0, wins=0, losses=0, draws=0,
        )
        assert rating.last_match_at is None

    def test_match_participant_defaults(self):
        p = MatchParticipant(miner_hotkey="test", model_revision="v1", slot=0)
        assert p.role is None
        assert p.raw_score is None
        assert p.outcome is None
        assert p.elo_before is None
        assert p.elo_after is None
        assert p.elo_change is None

    def test_match_result_defaults(self):
        result = MatchResult(
            match_uuid="test", env="test", match_type=MatchType.GAME,
            task_id=1, timestamp=1705000000, participants=[],
        )
        assert result.game_result is None
        assert result.validator_hotkey is None
