"""
Match Engine

Converts scores to matches and processes match outcomes for ELO updates.
Supports both incremental ELO updates and hybrid MLE-based systems.
"""

import time
import uuid
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from itertools import combinations

from .config import EloConfig, DEFAULT_ELO_CONFIG
from .calculator import EloCalculator
from .models import (
    MatchResult,
    MatchParticipant,
    MatchOutcome,
    MatchType,
    EloRating,
    SampleScore,
    PairedMatchResult,
)


class PairwiseMatchGenerator:
    """
    Generates pairwise matches from sample scores.

    When multiple miners complete the same task_id, their scores can be
    compared to create implicit head-to-head matches. This enables backward
    compatibility with the existing single-miner evaluation system.
    """

    def __init__(self, config: Optional[EloConfig] = None):
        """
        Initialize the generator.

        Args:
            config: ELO configuration. Uses DEFAULT_ELO_CONFIG if not provided.
        """
        self.config = config or DEFAULT_ELO_CONFIG
        self.calculator = EloCalculator(config)

    def group_samples_by_task(
        self,
        samples: List[SampleScore],
    ) -> Dict[Tuple[str, int], List[SampleScore]]:
        """
        Group samples by (env, task_id) for pairwise comparison.

        Args:
            samples: List of sample scores

        Returns:
            Dictionary mapping (env, task_id) to list of samples
        """
        groups: Dict[Tuple[str, int], List[SampleScore]] = {}

        for sample in samples:
            key = (sample.env, sample.task_id)
            if key not in groups:
                groups[key] = []
            groups[key].append(sample)

        return groups

    def generate_matches_from_samples(
        self,
        samples: List[SampleScore],
        ratings: Dict[str, EloRating],
        min_samples_per_task: int = 2,
    ) -> List[MatchResult]:
        """
        Generate pairwise matches from sample scores.

        For each task_id with 2+ miners, creates C(N,2) pairwise comparisons.
        Higher score wins.

        Args:
            samples: List of sample scores
            ratings: Current ELO ratings by miner_id+env key
            min_samples_per_task: Minimum samples needed to create matches

        Returns:
            List of match results with ELO updates
        """
        matches = []
        groups = self.group_samples_by_task(samples)

        for (env, task_id), task_samples in groups.items():
            if len(task_samples) < min_samples_per_task:
                continue

            n_participants = len(task_samples)
            scale = Decimal("1") / Decimal(str(n_participants - 1)) if n_participants > 2 else Decimal("1")

            for sample_a, sample_b in combinations(task_samples, 2):
                match = self._create_pairwise_match(
                    sample_a, sample_b, env, task_id, ratings
                )
                if match and scale != Decimal("1"):
                    for p in match.participants:
                        if p.elo_change is not None:
                            p.elo_change = (p.elo_change * scale).quantize(Decimal("0.01"))
                            p.elo_after = p.elo_before + p.elo_change
                if match:
                    matches.append(match)

        return matches

    def _create_pairwise_match(
        self,
        sample_a: SampleScore,
        sample_b: SampleScore,
        env: str,
        task_id: int,
        ratings: Dict[str, EloRating],
    ) -> Optional[MatchResult]:
        """
        Create a pairwise match from two samples.

        Args:
            sample_a: First sample
            sample_b: Second sample
            env: Environment name
            task_id: Task ID
            ratings: Current ELO ratings

        Returns:
            MatchResult with ELO updates, or None if cannot create match
        """
        rating_key_a = f"{sample_a.miner_id}#{env}"
        rating_key_b = f"{sample_b.miner_id}#{env}"

        elo_a = ratings.get(rating_key_a)
        elo_b = ratings.get(rating_key_b)

        current_rating_a = elo_a.rating if elo_a else self.config.DEFAULT_RATING
        current_rating_b = elo_b.rating if elo_b else self.config.DEFAULT_RATING
        matches_a = elo_a.matches_played if elo_a else 0
        matches_b = elo_b.matches_played if elo_b else 0

        outcome = self.calculator.score_to_outcome(sample_a.score, sample_b.score)

        new_rating_a, new_rating_b, delta_a, delta_b = self.calculator.update_ratings_head_to_head(
            current_rating_a, current_rating_b,
            matches_a, matches_b,
            outcome
        )

        if outcome == "a_wins":
            outcome_a, outcome_b = MatchOutcome.WIN, MatchOutcome.LOSS
        elif outcome == "b_wins":
            outcome_a, outcome_b = MatchOutcome.LOSS, MatchOutcome.WIN
        else:
            outcome_a, outcome_b = MatchOutcome.DRAW, MatchOutcome.DRAW

        timestamp = max(sample_a.timestamp, sample_b.timestamp)

        participants = [
            MatchParticipant(
                miner_hotkey=sample_a.miner_hotkey,
                model_revision=sample_a.model_revision,
                slot=0,
                raw_score=sample_a.score,
                outcome=outcome_a,
                elo_before=current_rating_a,
                elo_after=new_rating_a,
                elo_change=delta_a,
            ),
            MatchParticipant(
                miner_hotkey=sample_b.miner_hotkey,
                model_revision=sample_b.model_revision,
                slot=1,
                raw_score=sample_b.score,
                outcome=outcome_b,
                elo_before=current_rating_b,
                elo_after=new_rating_b,
                elo_change=delta_b,
            ),
        ]

        return MatchResult(
            match_uuid=str(uuid.uuid4()),
            env=env,
            match_type=MatchType.PAIRWISE,
            task_id=task_id,
            timestamp=timestamp,
            participants=participants,
        )


class MatchEngine:
    """
    Processes match outcomes and updates ELO ratings.

    Handles both pairwise matches (from score comparison) and
    direct competition games (tic-tac-toe, chess, etc.).
    """

    def __init__(self, config: Optional[EloConfig] = None):
        """
        Initialize the engine.

        Args:
            config: ELO configuration. Uses DEFAULT_ELO_CONFIG if not provided.
        """
        self.config = config or DEFAULT_ELO_CONFIG
        self.calculator = EloCalculator(config)

    def process_head_to_head_game(
        self,
        env: str,
        task_id: int,
        player_a: Dict[str, Any],
        player_b: Dict[str, Any],
        outcome: str,  # "a_wins", "b_wins", "draw"
        game_result: Optional[Dict[str, Any]] = None,
        validator_hotkey: Optional[str] = None,
        block_number: Optional[int] = None,
    ) -> MatchResult:
        """
        Process a head-to-head game result.

        Args:
            env: Environment name
            task_id: Task/game ID
            player_a: Player A info {hotkey, revision, rating, matches_played}
            player_b: Player B info {hotkey, revision, rating, matches_played}
            outcome: Game outcome ("a_wins", "b_wins", "draw")
            game_result: Game-specific result data
            validator_hotkey: Validator that executed the game
            block_number: Block number at execution

        Returns:
            MatchResult with ELO updates
        """
        rating_a = Decimal(str(player_a.get("rating", self.config.DEFAULT_RATING)))
        rating_b = Decimal(str(player_b.get("rating", self.config.DEFAULT_RATING)))
        matches_a = player_a.get("matches_played", 0)
        matches_b = player_b.get("matches_played", 0)

        new_rating_a, new_rating_b, delta_a, delta_b = self.calculator.update_ratings_head_to_head(
            rating_a, rating_b, matches_a, matches_b, outcome
        )

        if outcome == "a_wins":
            outcome_a, outcome_b = MatchOutcome.WIN, MatchOutcome.LOSS
        elif outcome == "b_wins":
            outcome_a, outcome_b = MatchOutcome.LOSS, MatchOutcome.WIN
        else:
            outcome_a, outcome_b = MatchOutcome.DRAW, MatchOutcome.DRAW

        timestamp = int(time.time() * 1000)

        participants = [
            MatchParticipant(
                miner_hotkey=player_a["hotkey"],
                model_revision=player_a["revision"],
                slot=0,
                role=player_a.get("role"),
                outcome=outcome_a,
                elo_before=rating_a,
                elo_after=new_rating_a,
                elo_change=delta_a,
            ),
            MatchParticipant(
                miner_hotkey=player_b["hotkey"],
                model_revision=player_b["revision"],
                slot=1,
                role=player_b.get("role"),
                outcome=outcome_b,
                elo_before=rating_b,
                elo_after=new_rating_b,
                elo_change=delta_b,
            ),
        ]

        return MatchResult(
            match_uuid=str(uuid.uuid4()),
            env=env,
            match_type=MatchType.GAME,
            task_id=task_id,
            timestamp=timestamp,
            participants=participants,
            game_result=game_result,
            validator_hotkey=validator_hotkey,
            block_number=block_number,
        )

    def process_multi_party_game(
        self,
        env: str,
        task_id: int,
        participants_data: List[Dict[str, Any]],
        game_result: Optional[Dict[str, Any]] = None,
        validator_hotkey: Optional[str] = None,
        block_number: Optional[int] = None,
    ) -> MatchResult:
        """
        Process a multi-party game result.

        Args:
            env: Environment name
            task_id: Task/game ID
            participants_data: List of participant info, each with:
                - hotkey, revision, rating, matches_played, final_rank
            game_result: Game-specific result data
            validator_hotkey: Validator that executed the game
            block_number: Block number at execution

        Returns:
            MatchResult with ELO updates for all participants
        """
        rating_updates = self.calculator.update_ratings_multi_party(participants_data)

        timestamp = int(time.time() * 1000)

        participants = []
        for i, (player_data, (new_rating, delta)) in enumerate(zip(participants_data, rating_updates)):
            old_rating = Decimal(str(player_data.get("rating", self.config.DEFAULT_RATING)))
            final_rank = player_data["final_rank"]

            num_players = len(participants_data)
            if final_rank == 1:
                outcome = MatchOutcome.WIN
            elif final_rank == num_players:
                outcome = MatchOutcome.LOSS
            else:
                outcome = MatchOutcome.DRAW

            participants.append(MatchParticipant(
                miner_hotkey=player_data["hotkey"],
                model_revision=player_data["revision"],
                slot=i,
                role=player_data.get("role"),
                outcome=outcome,
                elo_before=old_rating,
                elo_after=new_rating,
                elo_change=delta,
            ))

        return MatchResult(
            match_uuid=str(uuid.uuid4()),
            env=env,
            match_type=MatchType.GAME,
            task_id=task_id,
            timestamp=timestamp,
            participants=participants,
            game_result=game_result,
            validator_hotkey=validator_hotkey,
            block_number=block_number,
        )

    def apply_match_to_ratings(
        self,
        match: MatchResult,
        ratings: Dict[str, EloRating],
    ) -> Dict[str, EloRating]:
        """Apply match result to update ELO ratings dictionary.

        Creates missing ratings on first encounter. Only updates stats
        when elo_change is present (skips no-ops).
        """
        for participant in match.participants:
            key = f"{participant.miner_id}#{match.env}"
            if key not in ratings:
                ratings[key] = EloRating(
                    miner_hotkey=participant.miner_hotkey,
                    model_revision=participant.model_revision,
                    env=match.env,
                    rating=self.config.DEFAULT_RATING,
                    peak_rating=self.config.DEFAULT_RATING,
                )
            elo = ratings[key]

            if participant.elo_change is not None:
                elo.rating = max(
                    self.config.RATING_FLOOR,
                    min(self.config.RATING_CEILING, elo.rating + participant.elo_change),
                )
                elo.matches_played += 1
                elo.last_match_at = match.timestamp
                if elo.rating > elo.peak_rating:
                    elo.peak_rating = elo.rating
                if participant.outcome in (MatchOutcome.WIN,):
                    elo.wins += 1
                elif participant.outcome in (MatchOutcome.LOSS, MatchOutcome.TIMEOUT, MatchOutcome.ERROR):
                    elo.losses += 1
                elif participant.outcome == MatchOutcome.DRAW:
                    elo.draws += 1

        return ratings

    def process_paired_match(
        self,
        env: str,
        task_id: int,
        player_a: Dict[str, Any],
        player_b: Dict[str, Any],
        outcome_1: str,  # A's outcome when A went first
        outcome_2: str,  # A's outcome when B went first
        game_type: str,
        game_result_1: Optional[Dict[str, Any]] = None,
        game_result_2: Optional[Dict[str, Any]] = None,
        validator_hotkey: Optional[str] = None,
        block_number: Optional[int] = None,
    ) -> PairedMatchResult:
        """
        Process a paired match (match + rematch) and return combined result.

        Uses information-weighted updates based on outcome type.

        Args:
            env: Environment name
            task_id: Task/game ID
            player_a: Player A info {hotkey, revision, rating, matches_played}
            player_b: Player B info {hotkey, revision, rating, matches_played}
            outcome_1: A's outcome in game 1 ("a_wins", "b_wins", "draw")
            outcome_2: A's outcome in game 2 ("a_wins", "b_wins", "draw")
            game_type: Type of game ("tictactoe", "chess", etc.)
            game_result_1: Game-specific result data for game 1
            game_result_2: Game-specific result data for game 2
            validator_hotkey: Validator that executed the games
            block_number: Block number at execution

        Returns:
            PairedMatchResult with combined ELO updates
        """
        pair_uuid = str(uuid.uuid4())
        timestamp = int(time.time() * 1000)

        rating_a = Decimal(str(player_a.get("rating", self.config.DEFAULT_RATING)))
        rating_b = Decimal(str(player_b.get("rating", self.config.DEFAULT_RATING)))
        matches_a = player_a.get("matches_played", 0)
        matches_b = player_b.get("matches_played", 0)

        def to_match_outcome(outcome_str: str, is_player_a: bool) -> MatchOutcome:
            if outcome_str == "a_wins":
                return MatchOutcome.WIN if is_player_a else MatchOutcome.LOSS
            elif outcome_str == "b_wins":
                return MatchOutcome.LOSS if is_player_a else MatchOutcome.WIN
            elif outcome_str in ("timeout", "error"):
                return MatchOutcome.LOSS
            else:
                return MatchOutcome.DRAW

        outcome_a_1 = to_match_outcome(outcome_1, True)
        outcome_b_1 = to_match_outcome(outcome_1, False)
        outcome_a_2 = to_match_outcome(outcome_2, True)
        outcome_b_2 = to_match_outcome(outcome_2, False)

        paired_outcome = PairedMatchResult.determine_paired_outcome(outcome_a_1, outcome_a_2)
        info_weight = PairedMatchResult.get_information_weight(paired_outcome)
        paired_score = PairedMatchResult.get_paired_score(paired_outcome)

        new_rating_a, new_rating_b, delta_a, delta_b = self.calculator.update_ratings_paired(
            rating_a, rating_b,
            matches_a, matches_b,
            paired_outcome,
            info_weight,
        )

        first_mover_wins = 0
        first_mover_losses = 0

        if outcome_a_1 == MatchOutcome.WIN:
            first_mover_wins += 1
        elif outcome_a_1 == MatchOutcome.LOSS:
            first_mover_losses += 1

        if outcome_b_2 == MatchOutcome.WIN:
            first_mover_wins += 1
        elif outcome_b_2 == MatchOutcome.LOSS:
            first_mover_losses += 1

        match_1 = MatchResult(
            match_uuid=str(uuid.uuid4()),
            env=env,
            match_type=MatchType.PAIRED,
            task_id=task_id,
            timestamp=timestamp,
            participants=[
                MatchParticipant(
                    miner_hotkey=player_a["hotkey"],
                    model_revision=player_a["revision"],
                    slot=0,
                    outcome=outcome_a_1,
                    elo_before=rating_a,
                    elo_after=new_rating_a,
                    elo_change=delta_a,
                ),
                MatchParticipant(
                    miner_hotkey=player_b["hotkey"],
                    model_revision=player_b["revision"],
                    slot=1,
                    outcome=outcome_b_1,
                    elo_before=rating_b,
                    elo_after=new_rating_b,
                    elo_change=delta_b,
                ),
            ],
            game_result=game_result_1,
            validator_hotkey=validator_hotkey,
            block_number=block_number,
            pair_uuid=pair_uuid,
            is_first_mover=True,
            pair_sequence=0,
        )

        match_2 = MatchResult(
            match_uuid=str(uuid.uuid4()),
            env=env,
            match_type=MatchType.PAIRED,
            task_id=task_id,
            timestamp=timestamp + 1,  # Slightly later
            participants=[
                MatchParticipant(
                    miner_hotkey=player_b["hotkey"],
                    model_revision=player_b["revision"],
                    slot=0,
                    outcome=outcome_b_2,
                    elo_before=rating_b,
                    elo_after=new_rating_b,
                    elo_change=delta_b,
                ),
                MatchParticipant(
                    miner_hotkey=player_a["hotkey"],
                    model_revision=player_a["revision"],
                    slot=1,
                    outcome=outcome_a_2,
                    elo_before=rating_a,
                    elo_after=new_rating_a,
                    elo_change=delta_a,
                ),
            ],
            game_result=game_result_2,
            validator_hotkey=validator_hotkey,
            block_number=block_number,
            pair_uuid=pair_uuid,
            is_first_mover=True,
            pair_sequence=1,
        )

        return PairedMatchResult(
            pair_uuid=pair_uuid,
            env=env,
            task_id=task_id,
            timestamp=timestamp,
            game_type=game_type,
            player_a_hotkey=player_a["hotkey"],
            player_a_revision=player_a["revision"],
            player_b_hotkey=player_b["hotkey"],
            player_b_revision=player_b["revision"],
            match_1=match_1,
            match_2=match_2,
            paired_outcome=paired_outcome,
            paired_score=paired_score,
            information_weight=info_weight,
            player_a_elo_before=rating_a,
            player_a_elo_after=new_rating_a,
            player_a_elo_change=delta_a,
            player_b_elo_before=rating_b,
            player_b_elo_after=new_rating_b,
            player_b_elo_change=delta_b,
            first_mover_wins=first_mover_wins,
            first_mover_losses=first_mover_losses,
            validator_hotkey=validator_hotkey,
            block_number=block_number,
        )



def apply_matches_to_ratings(
    matches: List[MatchResult],
    ratings: Dict[str, EloRating],
    config: Optional[EloConfig] = None,
) -> Dict[str, EloRating]:
    """Apply a batch of match results to a ratings dict. Module-level convenience."""
    engine = MatchEngine(config or DEFAULT_ELO_CONFIG)
    for match in matches:
        engine.apply_match_to_ratings(match, ratings)
    return ratings


class EloSystem:
    """
    ELO system using MLE (Maximum Likelihood Estimation) for all rating updates.

    Runs torch.compile Newton-Raphson MLE after every paired match (~1-10ms).
    This provides optimal accuracy with minimal complexity.
    """

    def __init__(self, config: Optional[EloConfig] = None):
        self.config = config or DEFAULT_ELO_CONFIG
        self.engine = MatchEngine(config)

        self._match_history: List[MatchResult] = []
        self._paired_history: List[PairedMatchResult] = []
        self._ratings: Dict[str, EloRating] = {}
        self._last_mle_result = None
        self._matches_since_mle: int = 0
        self._last_mle_time: float = 0.0
        import threading
        self._lock = threading.Lock()

    @property
    def ratings(self) -> Dict[str, EloRating]:
        """Get current MLE-fitted ratings."""
        return self._ratings

    @property
    def match_history(self) -> List[MatchResult]:
        return self._match_history

    @property
    def paired_history(self) -> List[PairedMatchResult]:
        return self._paired_history

    @property
    def mle_result(self):
        """Get the last MLE fit result."""
        return self._last_mle_result

    def add_paired_match(
        self,
        env: str,
        task_id: int,
        player_a: Dict[str, Any],
        player_b: Dict[str, Any],
        outcome_1: str,
        outcome_2: str,
        game_type: str,
        game_result_1: Optional[Dict[str, Any]] = None,
        game_result_2: Optional[Dict[str, Any]] = None,
        validator_hotkey: Optional[str] = None,
        block_number: Optional[int] = None,
    ) -> PairedMatchResult:
        """
        Add a paired match and refit MLE.

        Args:
            env: Environment name
            task_id: Task/game ID
            player_a: Player A info {hotkey, revision}
            player_b: Player B info {hotkey, revision}
            outcome_1: A's outcome when A went first ("a_wins", "b_wins", "draw")
            outcome_2: A's outcome when B went first
            game_type: Type of game ("tictactoe", etc.)
            game_result_1: Game 1 result data
            game_result_2: Game 2 result data
            validator_hotkey: Validator
            block_number: Block number

        Returns:
            PairedMatchResult with MLE-fitted ratings
        """
        with self._lock:
            rating_key_a = f"{player_a['hotkey']}#{player_a['revision']}#{env}"
            rating_key_b = f"{player_b['hotkey']}#{player_b['revision']}#{env}"

            if rating_key_a in self._ratings:
                player_a = {**player_a, "rating": self._ratings[rating_key_a].rating,
                           "matches_played": self._ratings[rating_key_a].matches_played}
            if rating_key_b in self._ratings:
                player_b = {**player_b, "rating": self._ratings[rating_key_b].rating,
                           "matches_played": self._ratings[rating_key_b].matches_played}

            paired = self.engine.process_paired_match(
                env, task_id, player_a, player_b,
                outcome_1, outcome_2, game_type,
                game_result_1, game_result_2,
                validator_hotkey, block_number
            )

            self._paired_history.append(paired)
            self._match_history.append(paired.match_1)
            self._match_history.append(paired.match_2)
            self._matches_since_mle += 2

            now = time.time()
            elapsed_minutes = (now - self._last_mle_time) / 60.0
            should_refit = (
                self._last_mle_time == 0.0
                or self._matches_since_mle >= self.config.MLE_REFIT_MIN_GAMES
                or elapsed_minutes >= self.config.MLE_REFIT_MAX_MINUTES
            )

            if should_refit:
                self._run_mle(env)
                self._matches_since_mle = 0
                self._last_mle_time = now

            if rating_key_a in self._ratings:
                paired.player_a_elo_after = self._ratings[rating_key_a].rating
                paired.player_a_elo_change = paired.player_a_elo_after - paired.player_a_elo_before
            if rating_key_b in self._ratings:
                paired.player_b_elo_after = self._ratings[rating_key_b].rating
                paired.player_b_elo_change = paired.player_b_elo_after - paired.player_b_elo_before

            return paired

    def _run_mle(self, env: str) -> None:
        """Run MLE to refit ratings from match history for this environment."""
        from .paired_bradley_terry import PairedBradleyTerryModel

        model = PairedBradleyTerryModel(self.config)
        for match in self._match_history:
            if match.env == env:
                model.add_match_result(match)

        if len(model._players) < 2:
            return

        result = model.fit(estimate_first_mover=self.config.ESTIMATE_FIRST_MOVER_ADVANTAGE)
        self._last_mle_result = result

        for player_id, mle_rating in result.ratings.items():
            rating_key = f"{player_id}#{env}"

            if rating_key in self._ratings:
                elo = self._ratings[rating_key]
                elo.rating = mle_rating
                if mle_rating > elo.peak_rating:
                    elo.peak_rating = mle_rating
            else:
                parts = player_id.split('#')
                if len(parts) >= 2:
                    self._ratings[rating_key] = EloRating(
                        miner_hotkey=parts[0],
                        model_revision=parts[1],
                        env=env,
                        rating=mle_rating,
                        peak_rating=mle_rating,
                    )

    def get_leaderboard(self, env: Optional[str] = None) -> List[Tuple[str, EloRating]]:
        """Get sorted leaderboard."""
        filtered = self._ratings.items()
        if env:
            filtered = [(k, v) for k, v in filtered if v.env == env]
        return sorted(filtered, key=lambda x: x[1].rating, reverse=True)

    def get_first_mover_advantage(self) -> Tuple[Decimal, float]:
        """
        Get estimated first-mover advantage from MLE.

        Returns:
            Tuple of (elo_advantage, log_odds_alpha)
        """
        if self._last_mle_result is None:
            return Decimal("0"), 0.0
        return Decimal(str(round(self._last_mle_result.first_mover_elo, 2))), self._last_mle_result.first_mover_alpha

    def clear(self) -> None:
        """Clear all state."""
        self._match_history.clear()
        self._paired_history.clear()
        self._ratings.clear()
        self._last_mle_result = None


# Backwards compatibility alias
HybridEloSystem = EloSystem
