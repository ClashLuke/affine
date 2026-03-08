import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "affine", "src"))

import pytest
from decimal import Decimal

from elo.match_engine import PairwiseMatchGenerator, MatchEngine
from elo.models import MatchOutcome, MatchType, SampleScore, EloRating
from elo.config import EloConfig


class TestPairwiseMatchGenerator:
    @pytest.fixture
    def generator(self, elo_config):
        return PairwiseMatchGenerator(elo_config)

    def test_generate_from_two_samples(self, generator):
        samples = [
            SampleScore(miner_hotkey="miner_a", model_revision="v1", task_id=100, env="affine:ded-v2", score=Decimal("0.85"), timestamp=1705000000),
            SampleScore(miner_hotkey="miner_b", model_revision="v1", task_id=100, env="affine:ded-v2", score=Decimal("0.75"), timestamp=1705000000),
        ]
        matches = generator.generate_matches_from_samples(samples, {})
        assert len(matches) == 1
        assert matches[0].match_type == MatchType.PAIRWISE
        assert len(matches[0].participants) == 2

    def test_generate_from_three_samples(self, generator):
        samples = [
            SampleScore(miner_hotkey="a", model_revision="v1", task_id=1, env="test", score=Decimal("0.9"), timestamp=1),
            SampleScore(miner_hotkey="b", model_revision="v1", task_id=1, env="test", score=Decimal("0.8"), timestamp=1),
            SampleScore(miner_hotkey="c", model_revision="v1", task_id=1, env="test", score=Decimal("0.7"), timestamp=1),
        ]
        assert len(generator.generate_matches_from_samples(samples, {})) == 3

    def test_determine_outcome_clear_winner(self, generator):
        assert generator.calculator.score_to_outcome(Decimal("0.90"), Decimal("0.70")) == "a_wins"
        assert generator.calculator.score_to_outcome(Decimal("0.50"), Decimal("0.80")) == "b_wins"

    def test_determine_outcome_draw(self, generator):
        assert generator.calculator.score_to_outcome(Decimal("0.80"), Decimal("0.805")) == "draw"

    def test_no_matches_for_single_sample(self, generator):
        samples = [SampleScore(miner_hotkey="a", model_revision="v1", task_id=1, env="test", score=Decimal("0.9"), timestamp=1)]
        assert len(generator.generate_matches_from_samples(samples, {})) == 0

    def test_multiple_tasks(self, generator):
        samples = [
            SampleScore(miner_hotkey="a", model_revision="v1", task_id=1, env="test", score=Decimal("0.9"), timestamp=1),
            SampleScore(miner_hotkey="b", model_revision="v1", task_id=1, env="test", score=Decimal("0.8"), timestamp=1),
            SampleScore(miner_hotkey="c", model_revision="v1", task_id=2, env="test", score=Decimal("0.7"), timestamp=1),
            SampleScore(miner_hotkey="d", model_revision="v1", task_id=2, env="test", score=Decimal("0.6"), timestamp=1),
        ]
        assert len(generator.generate_matches_from_samples(samples, {})) == 2

    def test_participant_elo_tracking(self, generator):
        samples = [
            SampleScore(miner_hotkey="a", model_revision="v1", task_id=1, env="test", score=Decimal("0.9"), timestamp=1),
            SampleScore(miner_hotkey="b", model_revision="v1", task_id=1, env="test", score=Decimal("0.7"), timestamp=1),
        ]
        ratings = {
            "a#v1#test": EloRating(miner_hotkey="a", model_revision="v1", env="test", rating=Decimal("1600")),
            "b#v1#test": EloRating(miner_hotkey="b", model_revision="v1", env="test", rating=Decimal("1400")),
        }
        matches = generator.generate_matches_from_samples(samples, ratings)
        assert len(matches) == 1
        p_a = next(p for p in matches[0].participants if p.miner_hotkey == "a")
        assert p_a.outcome == MatchOutcome.WIN


class TestMatchEngine:
    def test_process_head_to_head_win(self, match_engine, sample_players):
        result = match_engine.process_head_to_head_game(
            env="game:tictactoe", task_id=1,
            player_a=sample_players[0], player_b=sample_players[1],
            outcome="a_wins", game_result={"moves": 9}, validator_hotkey="validator1",
        )
        assert result.match_type == MatchType.GAME
        assert len(result.participants) == 2

        winner = next(p for p in result.participants if p.miner_hotkey == sample_players[0]["hotkey"])
        assert winner.outcome == MatchOutcome.WIN
        assert winner.elo_after > winner.elo_before

        loser = next(p for p in result.participants if p.miner_hotkey == sample_players[1]["hotkey"])
        assert loser.outcome == MatchOutcome.LOSS
        assert loser.elo_after < loser.elo_before

    def test_process_head_to_head_draw(self, match_engine, sample_players):
        result = match_engine.process_head_to_head_game(
            env="game:tictactoe", task_id=1,
            player_a=sample_players[0], player_b=sample_players[1],
            outcome="draw", game_result={}, validator_hotkey="validator1",
        )
        for p in result.participants:
            assert p.outcome == MatchOutcome.DRAW

    def test_process_multi_party_game(self, match_engine):
        participants_data = [
            {"hotkey": "p1", "revision": "v1", "rating": Decimal("1500"), "matches_played": 0, "final_rank": 1, "slot": 0},
            {"hotkey": "p2", "revision": "v1", "rating": Decimal("1500"), "matches_played": 0, "final_rank": 2, "slot": 1},
            {"hotkey": "p3", "revision": "v1", "rating": Decimal("1500"), "matches_played": 0, "final_rank": 3, "slot": 2},
        ]
        result = match_engine.process_multi_party_game(
            env="game:poker", task_id=1,
            participants_data=participants_data, game_result={"pot": 1000}, validator_hotkey="validator1",
        )
        assert result.match_type == MatchType.GAME
        assert len(result.participants) == 3
        assert next(p for p in result.participants if p.slot == 0).outcome == MatchOutcome.WIN
        assert next(p for p in result.participants if p.slot == 2).outcome == MatchOutcome.LOSS

    def test_match_uuid_generated(self, match_engine, sample_players):
        result = match_engine.process_head_to_head_game(
            env="game:test", task_id=1,
            player_a=sample_players[0], player_b=sample_players[1],
            outcome="a_wins", game_result={}, validator_hotkey="val1",
        )
        assert result.match_uuid and len(result.match_uuid) > 0

    def test_timestamp_set(self, match_engine, sample_players):
        result = match_engine.process_head_to_head_game(
            env="game:test", task_id=1,
            player_a=sample_players[0], player_b=sample_players[1],
            outcome="a_wins", game_result={}, validator_hotkey="val1",
        )
        assert result.timestamp > 0

    def test_game_result_stored(self, match_engine, sample_players):
        game_data = {"moves": ["e4", "e5", "Nf3"], "duration_ms": 120000, "termination": "checkmate"}
        result = match_engine.process_head_to_head_game(
            env="game:chess", task_id=1,
            player_a=sample_players[0], player_b=sample_players[1],
            outcome="a_wins", game_result=game_data, validator_hotkey="val1",
        )
        assert result.game_result == game_data


class TestMatchEngineEdgeCases:
    def test_b_wins_outcome(self, match_engine, sample_players):
        result = match_engine.process_head_to_head_game(
            env="game:test", task_id=1,
            player_a=sample_players[0], player_b=sample_players[1],
            outcome="b_wins", game_result={}, validator_hotkey="val1",
        )
        assert next(p for p in result.participants if p.miner_hotkey == sample_players[1]["hotkey"]).outcome == MatchOutcome.WIN
        assert next(p for p in result.participants if p.miner_hotkey == sample_players[0]["hotkey"]).outcome == MatchOutcome.LOSS

    def test_multi_party_ties(self, match_engine):
        participants_data = [
            {"hotkey": "p1", "revision": "v1", "rating": Decimal("1500"), "matches_played": 0, "final_rank": 1, "slot": 0},
            {"hotkey": "p2", "revision": "v1", "rating": Decimal("1500"), "matches_played": 0, "final_rank": 1, "slot": 1},
            {"hotkey": "p3", "revision": "v1", "rating": Decimal("1500"), "matches_played": 0, "final_rank": 3, "slot": 2},
        ]
        result = match_engine.process_multi_party_game(
            env="game:test", task_id=1,
            participants_data=participants_data, game_result={}, validator_hotkey="val1",
        )
        p1 = next(p for p in result.participants if p.miner_hotkey == "p1")
        p2 = next(p for p in result.participants if p.miner_hotkey == "p2")
        assert abs(p1.elo_change - p2.elo_change) < Decimal("1")
