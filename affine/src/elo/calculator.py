"""
ELO Calculator

Implements the ELO rating system calculations including paired match support.
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Tuple, List, Dict, Any, Optional

from .config import EloConfig, DEFAULT_ELO_CONFIG
from .models import PairedOutcome


class EloCalculator:
    """
    ELO rating calculator.

    Implements standard ELO formulas with configurable K-factor and
    support for both head-to-head and multi-party matches.

    ELO Formula:
        Expected Score: E_A = 1 / (1 + 10^((R_B - R_A) / 400))
        New Rating: R'_A = R_A + K * (S_A - E_A)

    Where:
        R_A, R_B = ratings of players A and B
        S_A = actual score (1 for win, 0 for loss, 0.5 for draw)
        E_A = expected score
        K = K-factor (determines rating volatility)
    """

    def __init__(self, config: Optional[EloConfig] = None):
        """
        Initialize the calculator.

        Args:
            config: ELO configuration. Uses DEFAULT_ELO_CONFIG if not provided.
        """
        self.config = config or DEFAULT_ELO_CONFIG

    def expected_score(self, rating_a: Decimal, rating_b: Decimal) -> Decimal:
        """
        Calculate expected score for player A against player B.

        E_A = 1 / (1 + 10^((R_B - R_A) / scale))

        Args:
            rating_a: Rating of player A
            rating_b: Rating of player B

        Returns:
            Expected score for player A (between 0 and 1)
        """
        exponent = (rating_b - rating_a) / Decimal(self.config.SCALE)
        return Decimal("1") / (Decimal("1") + Decimal("10") ** exponent)

    def get_k_factor(self, rating: Decimal, matches_played: int) -> int:
        """
        Get dynamic K-factor based on player status.

        K-factor determines how much a single match affects rating:
        - New players (< 30 matches): Higher K for faster calibration
        - Elite players (rating > 1800): Lower K for stability
        - Established players (> 100 matches): Moderate K

        Args:
            rating: Current player rating
            matches_played: Number of matches played

        Returns:
            K-factor to use for rating calculation
        """
        if matches_played < self.config.NEW_PLAYER_THRESHOLD:
            return self.config.K_FACTOR_NEW_PLAYER
        elif rating > self.config.ELITE_RATING_THRESHOLD:
            return self.config.K_FACTOR_ELITE
        elif matches_played > self.config.ESTABLISHED_THRESHOLD:
            return self.config.K_FACTOR_ESTABLISHED
        else:
            return self.config.K_FACTOR

    def calculate_rating_change(
        self,
        rating: Decimal,
        expected: Decimal,
        actual: Decimal,
        k_factor: int,
    ) -> Decimal:
        """
        Calculate rating change for a single match.

        delta = K * (S - E)

        Args:
            rating: Current rating
            expected: Expected score (from expected_score())
            actual: Actual score (1.0 = win, 0.5 = draw, 0.0 = loss)
            k_factor: K-factor to use

        Returns:
            Rating change (can be positive or negative)
        """
        delta = Decimal(str(k_factor)) * (actual - expected)
        return delta.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def clamp_rating(self, rating: Decimal) -> Decimal:
        """
        Clamp rating to configured bounds.

        Args:
            rating: Rating to clamp

        Returns:
            Rating within [RATING_FLOOR, RATING_CEILING]
        """
        return max(self.config.RATING_FLOOR, min(self.config.RATING_CEILING, rating))

    def update_ratings_head_to_head(
        self,
        rating_a: Decimal,
        rating_b: Decimal,
        matches_a: int,
        matches_b: int,
        outcome: str,  # "a_wins", "b_wins", "draw"
    ) -> Tuple[Decimal, Decimal, Decimal, Decimal]:
        """
        Update ratings for a head-to-head match.

        Args:
            rating_a: Current rating of player A
            rating_b: Current rating of player B
            matches_a: Matches played by player A
            matches_b: Matches played by player B
            outcome: Match outcome ("a_wins", "b_wins", "draw")

        Returns:
            Tuple of (new_rating_a, new_rating_b, delta_a, delta_b)
        """
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = Decimal("1") - expected_a

        if outcome == "a_wins":
            actual_a, actual_b = Decimal("1"), Decimal("0")
        elif outcome == "b_wins":
            actual_a, actual_b = Decimal("0"), Decimal("1")
        else:  # draw
            actual_a, actual_b = Decimal("0.5"), Decimal("0.5")

        k_a = self.get_k_factor(rating_a, matches_a)
        k_b = self.get_k_factor(rating_b, matches_b)

        delta_a = self.calculate_rating_change(rating_a, expected_a, actual_a, k_a)
        delta_b = self.calculate_rating_change(rating_b, expected_b, actual_b, k_b)

        new_rating_a = self.clamp_rating(rating_a + delta_a)
        new_rating_b = self.clamp_rating(rating_b + delta_b)

        return new_rating_a, new_rating_b, delta_a, delta_b

    def update_ratings_multi_party(
        self,
        participants: List[Dict[str, Any]],
    ) -> List[Tuple[Decimal, Decimal]]:
        """
        Update ratings for a multi-party match using pairwise comparisons.

        For N-player games, each player is compared against all others.
        Rating change = sum of pairwise changes / (N-1)

        Each participant dict should have:
            - rating: Current rating (Decimal)
            - matches_played: Number of matches played (int)
            - final_rank: Final position (1 = best, lower is better)

        Args:
            participants: List of participant dictionaries

        Returns:
            List of (new_rating, delta) tuples for each participant
        """
        n = len(participants)
        results = []

        for i, player_i in enumerate(participants):
            total_delta = Decimal("0")
            rating_i = Decimal(str(player_i["rating"]))
            rank_i = player_i["final_rank"]
            k_i = self.get_k_factor(rating_i, player_i.get("matches_played", 0))

            for j, player_j in enumerate(participants):
                if i == j:
                    continue

                rating_j = Decimal(str(player_j["rating"]))
                rank_j = player_j["final_rank"]

                expected_i = self.expected_score(rating_i, rating_j)

                if rank_i < rank_j:
                    actual_i = Decimal("1")
                elif rank_i > rank_j:
                    actual_i = Decimal("0")
                else:
                    actual_i = Decimal("0.5")

                total_delta += self.calculate_rating_change(
                    rating_i, expected_i, actual_i, k_i
                )

            avg_delta = total_delta / Decimal(str(n - 1))
            avg_delta = avg_delta.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            new_rating = self.clamp_rating(rating_i + avg_delta)
            results.append((new_rating, avg_delta))

        return results

    def score_to_outcome(
        self,
        score_a: Decimal,
        score_b: Decimal,
        margin: Optional[Decimal] = None,
    ) -> str:
        """
        Convert score comparison to match outcome.

        Args:
            score_a: Score of player A
            score_b: Score of player B
            margin: Score difference threshold for draw. Uses config default if not provided.

        Returns:
            "a_wins", "b_wins", or "draw"
        """
        if margin is None:
            margin = self.config.SCORE_MARGIN

        diff = score_a - score_b

        if abs(diff) <= margin:
            return "draw"
        elif diff > 0:
            return "a_wins"
        else:
            return "b_wins"

    def update_ratings_paired(
        self,
        rating_a: Decimal,
        rating_b: Decimal,
        matches_a: int,
        matches_b: int,
        paired_outcome: PairedOutcome,
        information_weight: Optional[Decimal] = None,
    ) -> Tuple[Decimal, Decimal, Decimal, Decimal]:
        """
        Update ratings for a paired match (match + rematch).

        This method provides more statistically efficient updates for paired data by:
        1. Using information weighting based on outcome type
        2. Applying paired K-factor adjustment
        3. Reducing variance through combined analysis

        Symmetric outcomes (win-loss splits) contain minimal skill information
        because position effects cancel out, so they receive lower weight.

        Args:
            rating_a: Current rating of player A
            rating_b: Current rating of player B
            matches_a: Matches played by player A (for K-factor)
            matches_b: Matches played by player B (for K-factor)
            paired_outcome: Combined outcome of the match pair
            information_weight: Optional override for outcome weight (0.25 to 1.0)

        Returns:
            Tuple of (new_rating_a, new_rating_b, delta_a, delta_b)
        """
        if information_weight is None:
            from .models import PairedMatchResult
            information_weight = PairedMatchResult.get_information_weight(paired_outcome)

        from .models import PairedMatchResult
        paired_score = PairedMatchResult.get_paired_score(paired_outcome)
        actual_a = paired_score / Decimal("2")
        actual_b = Decimal("1") - actual_a

        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = Decimal("1") - expected_a

        k_a = self.get_k_factor(rating_a, matches_a)
        k_b = self.get_k_factor(rating_b, matches_b)

        effective_k_a = Decimal(str(k_a)) * Decimal(str(self.config.PAIRED_K_FACTOR_MULTIPLIER)) * information_weight
        effective_k_b = Decimal(str(k_b)) * Decimal(str(self.config.PAIRED_K_FACTOR_MULTIPLIER)) * information_weight

        delta_a = (actual_a - expected_a) * effective_k_a
        delta_a = delta_a.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        delta_b = (actual_b - expected_b) * effective_k_b
        delta_b = delta_b.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        new_rating_a = self.clamp_rating(rating_a + delta_a)
        new_rating_b = self.clamp_rating(rating_b + delta_b)

        return new_rating_a, new_rating_b, delta_a, delta_b

    def estimate_first_mover_advantage(
        self,
        first_mover_wins: int,
        first_mover_losses: int,
    ) -> Tuple[Decimal, Decimal]:
        """
        Estimate first-mover advantage parameter from game statistics.

        Uses the Bradley-Terry model:
            P(first mover wins) = 1 / (1 + exp(-α))

        Where α is the first-mover advantage in log-odds.

        Args:
            first_mover_wins: Number of games won by first mover
            first_mover_losses: Number of games lost by first mover (second mover won)

        Returns:
            Tuple of (advantage_elo_points, log_odds_alpha)
            - advantage_elo_points: First-mover advantage in ELO points
            - log_odds_alpha: Raw α parameter in log-odds scale
        """
        import math

        if first_mover_wins + first_mover_losses == 0:
            return Decimal("0"), Decimal("0")

        decisive_games = first_mover_wins + first_mover_losses
        win_rate = max(0.01, min(0.99, first_mover_wins / decisive_games))
        alpha = math.log(win_rate / (1 - win_rate))
        elo_advantage = alpha * self.config.SCALE / math.log(10)

        return (
            Decimal(str(round(elo_advantage, 2))),
            Decimal(str(round(alpha, 4)))
        )

