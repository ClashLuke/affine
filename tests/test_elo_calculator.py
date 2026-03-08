import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "affine", "src"))

import pytest
from decimal import Decimal

from elo.calculator import EloCalculator
from elo.config import EloConfig


class TestExpectedScore:
    def test_equal_ratings_gives_half(self, elo_calculator):
        expected = elo_calculator.expected_score(Decimal("1500"), Decimal("1500"))
        assert abs(expected - Decimal("0.5")) < Decimal("0.001")

    def test_higher_rating_gives_higher_expected(self, elo_calculator):
        expected_a = elo_calculator.expected_score(Decimal("1600"), Decimal("1400"))
        expected_b = elo_calculator.expected_score(Decimal("1400"), Decimal("1600"))
        assert expected_a > Decimal("0.5")
        assert expected_b < Decimal("0.5")
        assert abs(expected_a + expected_b - Decimal("1.0")) < Decimal("0.001")

    def test_large_rating_difference(self, elo_calculator):
        expected = elo_calculator.expected_score(Decimal("2000"), Decimal("1200"))
        assert expected > Decimal("0.95")

    def test_symmetry(self, elo_calculator):
        ea = elo_calculator.expected_score(Decimal("1523"), Decimal("1678"))
        eb = elo_calculator.expected_score(Decimal("1678"), Decimal("1523"))
        assert abs(ea + eb - Decimal("1.0")) < Decimal("0.001")


class TestKFactor:
    def test_new_player_gets_high_k(self, elo_calculator):
        assert elo_calculator.get_k_factor(Decimal("1500"), matches_played=10) == elo_calculator.config.K_FACTOR_NEW_PLAYER

    def test_established_player_gets_medium_k(self, elo_calculator):
        assert elo_calculator.get_k_factor(Decimal("1500"), matches_played=50) == elo_calculator.config.K_FACTOR

    def test_expert_player_gets_low_k(self, elo_calculator):
        assert elo_calculator.get_k_factor(Decimal("2100"), matches_played=100) == elo_calculator.config.K_FACTOR_ELITE

    def test_high_rating_new_player(self, elo_calculator):
        assert elo_calculator.get_k_factor(Decimal("2100"), matches_played=5) == elo_calculator.config.K_FACTOR_NEW_PLAYER


class TestHeadToHeadUpdates:
    def test_winner_gains_loser_loses(self, elo_calculator):
        new_a, new_b, delta_a, delta_b = elo_calculator.update_ratings_head_to_head(
            Decimal("1500"), Decimal("1500"), 0, 0, "a_wins",
        )
        assert new_a > Decimal("1500")
        assert new_b < Decimal("1500")
        assert delta_a > 0
        assert delta_b < 0
        assert abs(delta_a + delta_b) < Decimal("0.01")

    def test_upset_gives_larger_change(self, elo_calculator):
        _, _, delta_underdog, _ = elo_calculator.update_ratings_head_to_head(
            Decimal("1400"), Decimal("1600"), 0, 0, "a_wins",
        )
        _, _, delta_favorite, _ = elo_calculator.update_ratings_head_to_head(
            Decimal("1600"), Decimal("1400"), 0, 0, "a_wins",
        )
        assert abs(delta_underdog) > abs(delta_favorite)

    def test_draw_updates_toward_center(self, elo_calculator):
        new_a, new_b, _, _ = elo_calculator.update_ratings_head_to_head(
            Decimal("1600"), Decimal("1400"), 0, 0, "draw",
        )
        assert new_a < Decimal("1600")
        assert new_b > Decimal("1400")

    def test_b_wins_outcome(self, elo_calculator):
        new_a, new_b, _, _ = elo_calculator.update_ratings_head_to_head(
            Decimal("1500"), Decimal("1500"), 0, 0, "b_wins",
        )
        assert new_a < Decimal("1500")
        assert new_b > Decimal("1500")


class TestMultiPartyUpdates:
    def test_three_player_rankings(self, elo_calculator):
        participants = [
            {"hotkey": "A", "rating": Decimal("1500"), "matches_played": 0, "final_rank": 1},
            {"hotkey": "B", "rating": Decimal("1500"), "matches_played": 0, "final_rank": 2},
            {"hotkey": "C", "rating": Decimal("1500"), "matches_played": 0, "final_rank": 3},
        ]
        updates = elo_calculator.update_ratings_multi_party(participants)
        assert updates[0][1] > 0
        assert updates[2][1] < 0

    def test_four_player_tie(self, elo_calculator):
        participants = [
            {"hotkey": "A", "rating": Decimal("1500"), "matches_played": 0, "final_rank": 1},
            {"hotkey": "B", "rating": Decimal("1500"), "matches_played": 0, "final_rank": 1},
            {"hotkey": "C", "rating": Decimal("1500"), "matches_played": 0, "final_rank": 3},
            {"hotkey": "D", "rating": Decimal("1500"), "matches_played": 0, "final_rank": 4},
        ]
        updates = elo_calculator.update_ratings_multi_party(participants)
        assert abs(updates[0][1] - updates[1][1]) < Decimal("1")

    def test_preserves_total(self, elo_calculator):
        participants = [
            {"hotkey": "A", "rating": Decimal("1600"), "matches_played": 50, "final_rank": 1},
            {"hotkey": "B", "rating": Decimal("1500"), "matches_played": 50, "final_rank": 2},
            {"hotkey": "C", "rating": Decimal("1400"), "matches_played": 50, "final_rank": 3},
        ]
        updates = elo_calculator.update_ratings_multi_party(participants)
        total_delta = sum(u[1] for u in updates)
        assert abs(total_delta) < Decimal("1")


class TestScoreToOutcome:
    def test_clear_win(self, elo_calculator):
        assert elo_calculator.score_to_outcome(Decimal("0.95"), Decimal("0.50"), margin=Decimal("0.01")) == "a_wins"

    def test_clear_loss(self, elo_calculator):
        assert elo_calculator.score_to_outcome(Decimal("0.30"), Decimal("0.80"), margin=Decimal("0.01")) == "b_wins"

    def test_draw_within_margin(self, elo_calculator):
        assert elo_calculator.score_to_outcome(Decimal("0.505"), Decimal("0.500"), margin=Decimal("0.01")) == "draw"

    def test_exactly_equal(self, elo_calculator):
        assert elo_calculator.score_to_outcome(Decimal("0.75"), Decimal("0.75"), margin=Decimal("0.01")) == "draw"


class TestEdgeCases:
    def test_zero_rating(self, elo_calculator):
        expected = elo_calculator.expected_score(Decimal("100"), Decimal("1500"))
        assert Decimal("0") < expected < Decimal("0.1")

    def test_very_high_rating(self, elo_calculator):
        assert elo_calculator.expected_score(Decimal("3000"), Decimal("1500")) > Decimal("0.99")

    def test_negative_rating_delta(self, elo_calculator):
        new_a, _, _, _ = elo_calculator.update_ratings_head_to_head(
            Decimal("100"), Decimal("2500"), 0, 0, "b_wins",
        )
        assert new_a > Decimal("-1000")

    def test_decimal_precision(self, elo_calculator):
        new_a, new_b, _, _ = elo_calculator.update_ratings_head_to_head(
            Decimal("1523.456"), Decimal("1487.789"), 50, 50, "a_wins",
        )
        assert isinstance(new_a, Decimal)
        assert isinstance(new_b, Decimal)


class TestCustomConfig:
    def test_custom_k_factor(self):
        config = EloConfig(K_FACTOR_NEW_PLAYER=40, K_FACTOR_ESTABLISHED=30, K_FACTOR_ELITE=20)
        assert EloCalculator(config).get_k_factor(Decimal("1500"), matches_played=5) == 40

    def test_custom_scale(self):
        calc_small = EloCalculator(EloConfig(SCALE=200))
        calc_large = EloCalculator(EloConfig(SCALE=800))
        exp_small = calc_small.expected_score(Decimal("1600"), Decimal("1400"))
        exp_large = calc_large.expected_score(Decimal("1600"), Decimal("1400"))
        assert exp_small > exp_large

    def test_custom_default_rating(self):
        assert EloConfig(DEFAULT_RATING=Decimal("1000")).DEFAULT_RATING == Decimal("1000")
