import pytest
from decimal import Decimal

from affine.src.elo.models import (
    MatchResult, MatchParticipant, MatchOutcome, MatchType,
    PairedOutcome, PairedMatchResult,
)
from affine.src.elo.calculator import EloCalculator
from affine.src.elo.match_engine import HybridEloSystem
from affine.src.elo.paired_bradley_terry import PairedBradleyTerryModel
from affine.src.elo.bootstrap import PairedBootstrapBradleyTerry, BootstrapResult, bootstrap_confidence_intervals
from affine.src.elo.uncertainty import MonteCarloWeightCalculator


class TestPairedOutcome:
    def test_double_win(self):
        assert PairedMatchResult.determine_paired_outcome(MatchOutcome.WIN, MatchOutcome.WIN) == PairedOutcome.DOUBLE_WIN

    def test_win_draw(self):
        assert PairedMatchResult.determine_paired_outcome(MatchOutcome.WIN, MatchOutcome.DRAW) == PairedOutcome.WIN_DRAW
        assert PairedMatchResult.determine_paired_outcome(MatchOutcome.DRAW, MatchOutcome.WIN) == PairedOutcome.WIN_DRAW

    def test_split(self):
        assert PairedMatchResult.determine_paired_outcome(MatchOutcome.WIN, MatchOutcome.LOSS) == PairedOutcome.SPLIT
        assert PairedMatchResult.determine_paired_outcome(MatchOutcome.LOSS, MatchOutcome.WIN) == PairedOutcome.SPLIT

    def test_double_draw(self):
        assert PairedMatchResult.determine_paired_outcome(MatchOutcome.DRAW, MatchOutcome.DRAW) == PairedOutcome.DOUBLE_DRAW

    def test_draw_loss(self):
        assert PairedMatchResult.determine_paired_outcome(MatchOutcome.DRAW, MatchOutcome.LOSS) == PairedOutcome.DRAW_LOSS

    def test_double_loss(self):
        assert PairedMatchResult.determine_paired_outcome(MatchOutcome.LOSS, MatchOutcome.LOSS) == PairedOutcome.DOUBLE_LOSS


class TestPairedScore:
    def test_scores(self):
        assert PairedMatchResult.get_paired_score(PairedOutcome.DOUBLE_WIN) == Decimal("2.0")
        assert PairedMatchResult.get_paired_score(PairedOutcome.WIN_DRAW) == Decimal("1.5")
        assert PairedMatchResult.get_paired_score(PairedOutcome.DOUBLE_DRAW) == Decimal("1.0")
        assert PairedMatchResult.get_paired_score(PairedOutcome.SPLIT) == Decimal("1.0")
        assert PairedMatchResult.get_paired_score(PairedOutcome.DRAW_LOSS) == Decimal("0.5")
        assert PairedMatchResult.get_paired_score(PairedOutcome.DOUBLE_LOSS) == Decimal("0.0")


class TestInformationWeight:
    def test_decisive_outcomes_high_weight(self):
        assert PairedMatchResult.get_information_weight(PairedOutcome.DOUBLE_WIN) == Decimal("1.0")
        assert PairedMatchResult.get_information_weight(PairedOutcome.DOUBLE_LOSS) == Decimal("1.0")

    def test_split_low_weight(self):
        assert PairedMatchResult.get_information_weight(PairedOutcome.SPLIT) == Decimal("0.25")

    def test_partial_outcomes_medium_weight(self):
        assert PairedMatchResult.get_information_weight(PairedOutcome.WIN_DRAW) == Decimal("0.75")
        assert PairedMatchResult.get_information_weight(PairedOutcome.DRAW_LOSS) == Decimal("0.75")
        assert PairedMatchResult.get_information_weight(PairedOutcome.DOUBLE_DRAW) == Decimal("0.5")


class TestPairedEloCalculator:
    def setup_method(self):
        self.calculator = EloCalculator()

    def test_double_win_increases_rating(self):
        new_a, new_b, delta_a, delta_b = self.calculator.update_ratings_paired(
            Decimal("1500"), Decimal("1500"), 10, 10, PairedOutcome.DOUBLE_WIN,
        )
        assert new_a > Decimal("1500") and new_b < Decimal("1500")
        assert delta_a > 0 and delta_b < 0

    def test_split_minimal_change(self):
        _, _, delta_a, delta_b = self.calculator.update_ratings_paired(
            Decimal("1500"), Decimal("1500"), 10, 10, PairedOutcome.SPLIT,
        )
        assert abs(delta_a) < Decimal("5") and abs(delta_b) < Decimal("5")

    def test_information_weight_affects_delta(self):
        _, _, delta_dw, _ = self.calculator.update_ratings_paired(
            Decimal("1500"), Decimal("1500"), 10, 10, PairedOutcome.DOUBLE_WIN,
        )
        _, _, delta_wd, _ = self.calculator.update_ratings_paired(
            Decimal("1500"), Decimal("1500"), 10, 10, PairedOutcome.WIN_DRAW,
        )
        assert abs(delta_dw) > abs(delta_wd)


class TestFirstMoverAdvantage:
    def setup_method(self):
        self.calculator = EloCalculator()

    def test_no_advantage(self):
        elo_adv, alpha = self.calculator.estimate_first_mover_advantage(50, 50)
        assert abs(elo_adv) < Decimal("5")
        assert abs(alpha) < Decimal("0.1")

    def test_first_mover_advantage(self):
        elo_adv, alpha = self.calculator.estimate_first_mover_advantage(60, 40)
        assert elo_adv > Decimal("30")
        assert alpha > Decimal("0")

    def test_second_mover_advantage(self):
        elo_adv, alpha = self.calculator.estimate_first_mover_advantage(40, 60)
        assert elo_adv < Decimal("-30")
        assert alpha < Decimal("0")


class TestSAMOptimization:
    def test_sam_vs_pure_mle_similar_results(self):
        model_sam = PairedBradleyTerryModel()
        model_mle = PairedBradleyTerryModel()

        for _ in range(10):
            model_sam.add_match("A", "B", "a_wins")
            model_mle.add_match("A", "B", "a_wins")
        for _ in range(5):
            model_sam.add_match("A", "B", "b_wins")
            model_mle.add_match("A", "B", "b_wins")
        for _ in range(8):
            model_sam.add_match("B", "C", "a_wins")
            model_mle.add_match("B", "C", "a_wins")
        for _ in range(4):
            model_sam.add_match("B", "C", "b_wins")
            model_mle.add_match("B", "C", "b_wins")

        result_sam = model_sam.fit(estimate_first_mover=False, sam_rho=0.05)
        result_mle = model_mle.fit(estimate_first_mover=False, sam_rho=0.0)

        for player in ["A", "B", "C"]:
            diff = abs(float(result_sam.ratings[player]) - float(result_mle.ratings[player]))
            assert diff < 50, f"SAM and MLE differ too much for {player}: {diff}"

    def test_no_sample_count_bias(self):
        model = PairedBradleyTerryModel()

        for _ in range(50):
            model.add_match("X", "Y", "a_wins")
            model.add_match("X", "Y", "b_wins")

        # A: 80% win rate against X, 100 games
        for _ in range(80):
            model.add_match("A", "X", "a_wins")
        for _ in range(20):
            model.add_match("A", "X", "b_wins")

        # B: 80% win rate against X, 10 games
        for _ in range(8):
            model.add_match("B", "X", "a_wins")
        for _ in range(2):
            model.add_match("B", "X", "b_wins")

        result = model.fit(estimate_first_mover=False, sam_rho=0.05)
        rating_diff = abs(float(result.ratings["A"]) - float(result.ratings["B"]))
        assert rating_diff < 150, f"Sample count bias: A={result.ratings['A']}, B={result.ratings['B']}, diff={rating_diff}"

    def test_dominant_player_gets_high_skill(self):
        model = PairedBradleyTerryModel()
        for _ in range(5):
            model.add_match("A", "B", "a_wins", is_first_mover_a=True)
            model.add_match("B", "A", "b_wins", is_first_mover_a=True)

        result = model.fit(estimate_first_mover=True, sam_rho=0.0)
        assert result.ratings["A"] > result.ratings["B"]

        skill_diff = abs(result.skills["A"] - result.skills["B"])
        assert 5 < skill_diff <= 16

    def test_mixed_results_give_bounded_skill(self):
        model = PairedBradleyTerryModel()
        for _ in range(8):
            model.add_match("A", "B", "a_wins")
        for _ in range(2):
            model.add_match("A", "B", "b_wins")

        result = model.fit(estimate_first_mover=False, sam_rho=0.0)
        assert result.ratings["A"] > result.ratings["B"]
        assert abs(result.skills["A"] - result.skills["B"]) < 5


class TestPairedBradleyTerryModel:
    def test_simple_dominance(self):
        model = PairedBradleyTerryModel()
        for _ in range(8):
            model.add_match("A", "B", "a_wins")
        for _ in range(2):
            model.add_match("A", "B", "b_wins")

        result = model.fit(estimate_first_mover=False)
        assert result.ratings["A"] > result.ratings["B"]
        assert result.converged

    def test_transitive_ranking(self):
        model = PairedBradleyTerryModel()
        for _ in range(6):
            model.add_match("A", "B", "a_wins")
        for _ in range(2):
            model.add_match("A", "B", "b_wins")
        for _ in range(6):
            model.add_match("B", "C", "a_wins")
        for _ in range(2):
            model.add_match("B", "C", "b_wins")
        for _ in range(7):
            model.add_match("A", "C", "a_wins")
        for _ in range(1):
            model.add_match("A", "C", "b_wins")

        result = model.fit(estimate_first_mover=False)
        assert result.ratings["A"] > result.ratings["B"] > result.ratings["C"]

    def test_first_mover_estimation(self):
        model = PairedBradleyTerryModel()
        for _ in range(20):
            model.add_match("A", "B", "a_wins", is_first_mover_a=True)
            model.add_match("B", "A", "a_wins", is_first_mover_a=True)

        result = model.fit(estimate_first_mover=True)
        assert abs(result.ratings["A"] - result.ratings["B"]) < Decimal("50")
        assert result.first_mover_elo > 100

    def test_win_probability_prediction(self):
        model = PairedBradleyTerryModel()
        for _ in range(10):
            model.add_match("A", "B", "a_wins")

        model.fit(estimate_first_mover=False)
        assert model.predict_win_probability("A", "B") > 0.7
        assert model.predict_win_probability("B", "A") < 0.3


class TestHybridEloSystem:
    def test_incremental_updates(self):
        system = HybridEloSystem()
        system.add_paired_match(
            env="test", task_id=1,
            player_a={"hotkey": "A", "revision": "v1"},
            player_b={"hotkey": "B", "revision": "v1"},
            outcome_1="a_wins", outcome_2="a_wins", game_type="tictactoe",
        )
        assert len(system.ratings) == 2
        assert system.ratings["A#v1#test"].rating > system.ratings["B#v1#test"].rating

    def test_paired_updates(self):
        system = HybridEloSystem()
        paired = system.add_paired_match(
            env="test", task_id=1,
            player_a={"hotkey": "A", "revision": "v1"},
            player_b={"hotkey": "B", "revision": "v1"},
            outcome_1="a_wins", outcome_2="a_wins", game_type="tictactoe",
        )
        assert paired.paired_outcome == PairedOutcome.DOUBLE_WIN
        assert system.ratings["A#v1#test"].rating > system.ratings["B#v1#test"].rating

    def test_first_mover_tracking(self):
        system = HybridEloSystem()
        system.add_paired_match(
            env="test", task_id=1,
            player_a={"hotkey": "A", "revision": "v1"},
            player_b={"hotkey": "B", "revision": "v1"},
            outcome_1="a_wins", outcome_2="b_wins", game_type="tictactoe",
        )
        elo_adv, _ = system.get_first_mover_advantage()
        assert elo_adv > Decimal("0")


class TestBootstrapConfidenceIntervals:
    def test_bootstrap_produces_intervals(self):
        model = PairedBootstrapBradleyTerry()
        for _ in range(5):
            model.add_match(_create_match("A", "B", "a_wins"))
            model.add_match(_create_match("A", "B", "b_wins"))
            model.add_match(_create_match("B", "C", "a_wins"))

        result = model.fit(num_bootstrap=100, estimate_first_mover=False)
        for player in ["A#v1", "B#v1", "C#v1"]:
            assert player in result.rating_ci_lower
            assert result.rating_ci_lower[player] < result.rating_ci_upper[player]

    def test_ci_overlap_detection(self):
        result = BootstrapResult(
            rating_means={"A": Decimal("1600"), "B": Decimal("1550")},
            rating_ci_lower={"A": Decimal("1550"), "B": Decimal("1500")},
            rating_ci_upper={"A": Decimal("1650"), "B": Decimal("1600")},
            rating_se={"A": Decimal("30"), "B": Decimal("30")},
            num_bootstrap_samples=100, num_pairs=0, num_matches=10, confidence_level=0.95,
        )
        assert result.ci_overlap("A", "B") is True


class TestMonteCarloWeights:
    def test_softmax_weights_sum_to_one(self):
        calc = MonteCarloWeightCalculator()
        weights = calc.compute_softmax_weights({"A": 1600.0, "B": 1500.0, "C": 1400.0})
        assert abs(sum(weights.values()) - 1.0) < 1e-10

    def test_higher_rating_higher_weight(self):
        calc = MonteCarloWeightCalculator()
        weights = calc.compute_softmax_weights({"A": 1700.0, "B": 1500.0, "C": 1300.0})
        assert weights["A"] > weights["B"] > weights["C"]

    def test_weight_uncertainty_propagation(self):
        calc = MonteCarloWeightCalculator()
        result = calc.propagate_from_mean_std(
            {"A": 1600.0, "B": 1500.0, "C": 1400.0},
            {"A": 50.0, "B": 50.0, "C": 50.0},
            num_samples=500,
        )
        for p in ["A", "B", "C"]:
            assert result.weight_stds[p] > 0
            assert result.weight_ci_lower[p] < result.weight_means[p] < result.weight_ci_upper[p]


def _create_match(player_a: str, player_b: str, outcome: str) -> MatchResult:
    import uuid, time
    outcome_a = MatchOutcome.WIN if outcome == "a_wins" else (MatchOutcome.LOSS if outcome == "b_wins" else MatchOutcome.DRAW)
    outcome_b = MatchOutcome.LOSS if outcome == "a_wins" else (MatchOutcome.WIN if outcome == "b_wins" else MatchOutcome.DRAW)
    return MatchResult(
        match_uuid=str(uuid.uuid4()), env="test", match_type=MatchType.GAME,
        task_id=1, timestamp=int(time.time() * 1000),
        participants=[
            MatchParticipant(miner_hotkey=player_a, model_revision="v1", slot=0, outcome=outcome_a),
            MatchParticipant(miner_hotkey=player_b, model_revision="v1", slot=1, outcome=outcome_b),
        ],
    )
