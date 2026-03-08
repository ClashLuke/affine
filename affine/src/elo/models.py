"""
ELO Data Models

Defines data structures for ELO ratings and match results.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any


class MatchOutcome(Enum):
    """Possible match outcomes."""
    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"
    TIMEOUT = "timeout"
    ERROR = "error"


class MatchType(Enum):
    """Types of matches."""
    PAIRWISE = "pairwise"  # Implicit match from score comparison
    GAME = "game"  # Direct competition game (tic-tac-toe, chess, etc.)
    PAIRED = "paired"  # Paired match (match + rematch with linked pair_uuid)


class PairedOutcome(Enum):
    """
    Combined outcome of a match+rematch pair.

    The paired outcome is determined by combining the two individual game results:
    - DOUBLE_WIN: Won both games (win, win)
    - WIN_DRAW: Won one, drew the other (win, draw)
    - DOUBLE_DRAW: Drew both games (draw, draw)
    - SPLIT: Won one, lost the other (win, loss) - symmetric, minimal skill info
    - DRAW_LOSS: Drew one, lost the other (draw, loss)
    - DOUBLE_LOSS: Lost both games (loss, loss)
    """
    DOUBLE_WIN = "double_win"  # (win, win) - decisive
    WIN_DRAW = "win_draw"      # (win, draw) - strong
    DOUBLE_DRAW = "double_draw"  # (draw, draw) - neutral
    SPLIT = "split"            # (win, loss) - symmetric, cancels
    DRAW_LOSS = "draw_loss"    # (draw, loss) - weak
    DOUBLE_LOSS = "double_loss"  # (loss, loss) - decisive


@dataclass
class EloRating:
    """ELO rating for a miner in an environment."""

    miner_hotkey: str
    model_revision: str
    env: str
    rating: Decimal = field(default_factory=lambda: Decimal("1500"))
    peak_rating: Decimal = field(default_factory=lambda: Decimal("1500"))
    matches_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    last_match_at: Optional[int] = None  # Timestamp
    updated_at: Optional[int] = None

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.matches_played == 0:
            return 0.0
        return self.wins / self.matches_played

    @property
    def miner_id(self) -> str:
        """Unique identifier for this miner."""
        return f"{self.miner_hotkey}#{self.model_revision}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "miner_hotkey": self.miner_hotkey,
            "model_revision": self.model_revision,
            "env": self.env,
            "rating": float(self.rating),
            "peak_rating": float(self.peak_rating),
            "matches_played": self.matches_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.win_rate,
            "last_match_at": self.last_match_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EloRating":
        """Create from dictionary."""
        return cls(
            miner_hotkey=data["miner_hotkey"],
            model_revision=data["model_revision"],
            env=data["env"],
            rating=Decimal(str(data.get("rating", 1500))),
            peak_rating=Decimal(str(data.get("peak_rating", 1500))),
            matches_played=data.get("matches_played", 0),
            wins=data.get("wins", 0),
            losses=data.get("losses", 0),
            draws=data.get("draws", 0),
            last_match_at=data.get("last_match_at"),
            updated_at=data.get("updated_at"),
        )


@dataclass
class MatchParticipant:
    """A participant in a match."""

    miner_hotkey: str
    model_revision: str
    slot: int = 0  # Player position (0, 1, 2, ...)
    role: Optional[str] = None  # Game-specific role (e.g., "player_x", "bug_creator")
    raw_score: Optional[Decimal] = None  # Original score from environment
    outcome: Optional[MatchOutcome] = None
    elo_before: Optional[Decimal] = None
    elo_after: Optional[Decimal] = None
    elo_change: Optional[Decimal] = None

    @property
    def miner_id(self) -> str:
        """Unique identifier for this miner."""
        return f"{self.miner_hotkey}#{self.model_revision}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "miner_hotkey": self.miner_hotkey,
            "model_revision": self.model_revision,
            "slot": self.slot,
            "role": self.role,
            "raw_score": float(self.raw_score) if self.raw_score is not None else None,
            "outcome": self.outcome.value if self.outcome else None,
            "elo_before": float(self.elo_before) if self.elo_before is not None else None,
            "elo_after": float(self.elo_after) if self.elo_after is not None else None,
            "elo_change": float(self.elo_change) if self.elo_change is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MatchParticipant":
        """Create from dictionary."""
        outcome = None
        if data.get("outcome"):
            outcome = MatchOutcome(data["outcome"])

        return cls(
            miner_hotkey=data["miner_hotkey"],
            model_revision=data["model_revision"],
            slot=data.get("slot", 0),
            role=data.get("role"),
            raw_score=Decimal(str(data["raw_score"])) if data.get("raw_score") is not None else None,
            outcome=outcome,
            elo_before=Decimal(str(data["elo_before"])) if data.get("elo_before") is not None else None,
            elo_after=Decimal(str(data["elo_after"])) if data.get("elo_after") is not None else None,
            elo_change=Decimal(str(data["elo_change"])) if data.get("elo_change") is not None else None,
        )


@dataclass
class MatchResult:
    """Result of a match between miners."""

    match_uuid: str
    env: str
    match_type: MatchType
    task_id: int
    timestamp: int  # Milliseconds since epoch
    participants: List[MatchParticipant]
    game_result: Optional[Dict[str, Any]] = None  # Game-specific result data
    validator_hotkey: Optional[str] = None
    block_number: Optional[int] = None
    # Paired match fields
    pair_uuid: Optional[str] = None  # Links match+rematch pairs
    is_first_mover: Optional[bool] = None  # True if first participant moved first
    pair_sequence: Optional[int] = None  # 0 = first game, 1 = rematch

    def get_winner(self) -> Optional[MatchParticipant]:
        """Get the winning participant, if any."""
        for p in self.participants:
            if p.outcome == MatchOutcome.WIN:
                return p
        return None

    def get_loser(self) -> Optional[MatchParticipant]:
        """Get the losing participant, if any."""
        for p in self.participants:
            if p.outcome == MatchOutcome.LOSS:
                return p
        return None

    def is_draw(self) -> bool:
        """Check if the match was a draw."""
        return all(p.outcome == MatchOutcome.DRAW for p in self.participants)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "match_uuid": self.match_uuid,
            "env": self.env,
            "match_type": self.match_type.value,
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "participants": [p.to_dict() for p in self.participants],
            "game_result": self.game_result,
            "validator_hotkey": self.validator_hotkey,
            "block_number": self.block_number,
            "pair_uuid": self.pair_uuid,
            "is_first_mover": self.is_first_mover,
            "pair_sequence": self.pair_sequence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MatchResult":
        """Create from dictionary."""
        return cls(
            match_uuid=data["match_uuid"],
            env=data["env"],
            match_type=MatchType(data["match_type"]),
            task_id=data["task_id"],
            timestamp=data["timestamp"],
            participants=[MatchParticipant.from_dict(p) for p in data["participants"]],
            game_result=data.get("game_result"),
            validator_hotkey=data.get("validator_hotkey"),
            block_number=data.get("block_number"),
            pair_uuid=data.get("pair_uuid"),
            is_first_mover=data.get("is_first_mover"),
            pair_sequence=data.get("pair_sequence"),
        )


@dataclass
class SampleScore:
    """A sample score for pairwise comparison."""

    miner_hotkey: str
    model_revision: str
    env: str
    task_id: int
    score: Decimal
    timestamp: int

    @property
    def miner_id(self) -> str:
        """Unique identifier for this miner."""
        return f"{self.miner_hotkey}#{self.model_revision}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "miner_hotkey": self.miner_hotkey,
            "model_revision": self.model_revision,
            "env": self.env,
            "task_id": self.task_id,
            "score": float(self.score),
            "timestamp": self.timestamp,
        }


@dataclass
class PairedMatchResult:
    """
    Result of a paired match (match + rematch).

    A paired match consists of two games between the same players with positions swapped:
    - Game 1: Player A goes first
    - Game 2: Player B goes first

    This structure captures additional information:
    - First-mover advantage effects cancel out
    - Symmetric outcomes (win-loss) indicate position effects dominate
    - Decisive outcomes (double win/loss) indicate true skill differences
    """

    pair_uuid: str
    env: str
    task_id: int
    timestamp: int  # Timestamp of the pair completion
    game_type: str  # "tictactoe", "chess", "connect4", "nim"

    # Player information (player_a goes first in match_1)
    player_a_hotkey: str
    player_a_revision: str
    player_b_hotkey: str
    player_b_revision: str

    # Individual match results
    match_1: MatchResult  # A goes first
    match_2: MatchResult  # B goes first

    # Combined analysis
    paired_outcome: PairedOutcome
    paired_score: Decimal  # Combined score for player A (0.0 to 2.0)
    information_weight: Decimal  # How informative this pair is (0.25 to 1.0)

    # Rating changes (applied once per pair, not per game)
    player_a_elo_before: Optional[Decimal] = None
    player_a_elo_after: Optional[Decimal] = None
    player_a_elo_change: Optional[Decimal] = None
    player_b_elo_before: Optional[Decimal] = None
    player_b_elo_after: Optional[Decimal] = None
    player_b_elo_change: Optional[Decimal] = None

    # First-mover advantage tracking
    first_mover_wins: int = 0  # Count of games won by first mover
    first_mover_losses: int = 0  # Count of games lost by first mover

    # Optional metadata
    validator_hotkey: Optional[str] = None
    block_number: Optional[int] = None

    @property
    def player_a_id(self) -> str:
        """Unique identifier for player A."""
        return f"{self.player_a_hotkey}#{self.player_a_revision}"

    @property
    def player_b_id(self) -> str:
        """Unique identifier for player B."""
        return f"{self.player_b_hotkey}#{self.player_b_revision}"

    @staticmethod
    def determine_paired_outcome(
        outcome_1: MatchOutcome,  # Player A's outcome when A went first
        outcome_2: MatchOutcome,  # Player A's outcome when B went first
    ) -> PairedOutcome:
        """
        Determine the combined paired outcome from two game results.

        Args:
            outcome_1: Player A's result in game 1 (A went first)
            outcome_2: Player A's result in game 2 (B went first)

        Returns:
            PairedOutcome representing the combined result
        """
        # Count wins and losses for player A across both games
        a_wins = sum(1 for o in [outcome_1, outcome_2] if o == MatchOutcome.WIN)
        a_losses = sum(1 for o in [outcome_1, outcome_2] if o == MatchOutcome.LOSS)

        if a_wins == 2:
            return PairedOutcome.DOUBLE_WIN
        elif a_wins == 1 and a_losses == 0:
            return PairedOutcome.WIN_DRAW
        elif a_wins == 1 and a_losses == 1:
            return PairedOutcome.SPLIT
        elif a_wins == 0 and a_losses == 0:
            return PairedOutcome.DOUBLE_DRAW
        elif a_wins == 0 and a_losses == 1:
            return PairedOutcome.DRAW_LOSS
        else:  # a_losses == 2
            return PairedOutcome.DOUBLE_LOSS

    @staticmethod
    def get_paired_score(paired_outcome: PairedOutcome) -> Decimal:
        """
        Get the combined score for a paired outcome.

        Returns a score from 0.0 to 2.0 representing player A's total score:
        - 2.0: Won both games
        - 1.5: Won one, drew one
        - 1.0: Split (won one, lost one) or drew both
        - 0.5: Drew one, lost one
        - 0.0: Lost both games
        """
        score_map = {
            PairedOutcome.DOUBLE_WIN: Decimal("2.0"),
            PairedOutcome.WIN_DRAW: Decimal("1.5"),
            PairedOutcome.DOUBLE_DRAW: Decimal("1.0"),
            PairedOutcome.SPLIT: Decimal("1.0"),
            PairedOutcome.DRAW_LOSS: Decimal("0.5"),
            PairedOutcome.DOUBLE_LOSS: Decimal("0.0"),
        }
        return score_map[paired_outcome]

    @staticmethod
    def get_information_weight(paired_outcome: PairedOutcome) -> Decimal:
        """
        Get the information weight for a paired outcome.

        Symmetric outcomes (win-loss) contain minimal skill information because
        position effects cancel out. Decisive outcomes provide more information.

        Returns:
            Weight from 0.25 to 1.0 indicating how informative this pair is
        """
        weight_map = {
            PairedOutcome.DOUBLE_WIN: Decimal("1.0"),    # Decisive, high info
            PairedOutcome.WIN_DRAW: Decimal("0.75"),     # Strong signal
            PairedOutcome.DOUBLE_DRAW: Decimal("0.5"),   # Moderate info
            PairedOutcome.SPLIT: Decimal("0.25"),        # Symmetric, low info
            PairedOutcome.DRAW_LOSS: Decimal("0.75"),    # Strong signal
            PairedOutcome.DOUBLE_LOSS: Decimal("1.0"),   # Decisive, high info
        }
        return weight_map[paired_outcome]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pair_uuid": self.pair_uuid,
            "env": self.env,
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "game_type": self.game_type,
            "player_a_hotkey": self.player_a_hotkey,
            "player_a_revision": self.player_a_revision,
            "player_b_hotkey": self.player_b_hotkey,
            "player_b_revision": self.player_b_revision,
            "match_1": self.match_1.to_dict(),
            "match_2": self.match_2.to_dict(),
            "paired_outcome": self.paired_outcome.value,
            "paired_score": float(self.paired_score),
            "information_weight": float(self.information_weight),
            "player_a_elo_before": float(self.player_a_elo_before) if self.player_a_elo_before else None,
            "player_a_elo_after": float(self.player_a_elo_after) if self.player_a_elo_after else None,
            "player_a_elo_change": float(self.player_a_elo_change) if self.player_a_elo_change else None,
            "player_b_elo_before": float(self.player_b_elo_before) if self.player_b_elo_before else None,
            "player_b_elo_after": float(self.player_b_elo_after) if self.player_b_elo_after else None,
            "player_b_elo_change": float(self.player_b_elo_change) if self.player_b_elo_change else None,
            "first_mover_wins": self.first_mover_wins,
            "first_mover_losses": self.first_mover_losses,
            "validator_hotkey": self.validator_hotkey,
            "block_number": self.block_number,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PairedMatchResult":
        """Create from dictionary."""
        return cls(
            pair_uuid=data["pair_uuid"],
            env=data["env"],
            task_id=data["task_id"],
            timestamp=data["timestamp"],
            game_type=data["game_type"],
            player_a_hotkey=data["player_a_hotkey"],
            player_a_revision=data["player_a_revision"],
            player_b_hotkey=data["player_b_hotkey"],
            player_b_revision=data["player_b_revision"],
            match_1=MatchResult.from_dict(data["match_1"]),
            match_2=MatchResult.from_dict(data["match_2"]),
            paired_outcome=PairedOutcome(data["paired_outcome"]),
            paired_score=Decimal(str(data["paired_score"])),
            information_weight=Decimal(str(data["information_weight"])),
            player_a_elo_before=Decimal(str(data["player_a_elo_before"])) if data.get("player_a_elo_before") else None,
            player_a_elo_after=Decimal(str(data["player_a_elo_after"])) if data.get("player_a_elo_after") else None,
            player_a_elo_change=Decimal(str(data["player_a_elo_change"])) if data.get("player_a_elo_change") else None,
            player_b_elo_before=Decimal(str(data["player_b_elo_before"])) if data.get("player_b_elo_before") else None,
            player_b_elo_after=Decimal(str(data["player_b_elo_after"])) if data.get("player_b_elo_after") else None,
            player_b_elo_change=Decimal(str(data["player_b_elo_change"])) if data.get("player_b_elo_change") else None,
            first_mover_wins=data.get("first_mover_wins", 0),
            first_mover_losses=data.get("first_mover_losses", 0),
            validator_hotkey=data.get("validator_hotkey"),
            block_number=data.get("block_number"),
        )

    @classmethod
    def from_match_pair(
        cls,
        match_1: MatchResult,
        match_2: MatchResult,
        game_type: str,
        pair_uuid: Optional[str] = None,
    ) -> "PairedMatchResult":
        """
        Create a PairedMatchResult from two individual MatchResults.

        Args:
            match_1: First game (player_a goes first)
            match_2: Second game (player_b goes first)
            game_type: Type of game played
            pair_uuid: Optional UUID for the pair (generated if not provided)

        Returns:
            PairedMatchResult combining the two games
        """
        import uuid as uuid_module

        # Extract player info from match_1 (A is slot 0, B is slot 1)
        p_a = match_1.participants[0]
        p_b = match_1.participants[1]

        # Determine outcomes for player A in each game
        outcome_1 = p_a.outcome  # A's result when A went first
        # In match_2, B goes first (slot 0), A is slot 1
        outcome_2 = match_2.participants[1].outcome  # A's result when B went first

        paired_outcome = cls.determine_paired_outcome(outcome_1, outcome_2)
        paired_score = cls.get_paired_score(paired_outcome)
        info_weight = cls.get_information_weight(paired_outcome)

        # Track first-mover advantage
        first_mover_wins = 0
        first_mover_losses = 0

        # Game 1: A is first mover
        if match_1.participants[0].outcome == MatchOutcome.WIN:
            first_mover_wins += 1
        elif match_1.participants[0].outcome == MatchOutcome.LOSS:
            first_mover_losses += 1

        # Game 2: B is first mover
        if match_2.participants[0].outcome == MatchOutcome.WIN:
            first_mover_wins += 1
        elif match_2.participants[0].outcome == MatchOutcome.LOSS:
            first_mover_losses += 1

        return cls(
            pair_uuid=pair_uuid or str(uuid_module.uuid4()),
            env=match_1.env,
            task_id=match_1.task_id,
            timestamp=max(match_1.timestamp, match_2.timestamp),
            game_type=game_type,
            player_a_hotkey=p_a.miner_hotkey,
            player_a_revision=p_a.model_revision,
            player_b_hotkey=p_b.miner_hotkey,
            player_b_revision=p_b.model_revision,
            match_1=match_1,
            match_2=match_2,
            paired_outcome=paired_outcome,
            paired_score=paired_score,
            information_weight=info_weight,
            first_mover_wins=first_mover_wins,
            first_mover_losses=first_mover_losses,
            validator_hotkey=match_1.validator_hotkey,
            block_number=match_1.block_number,
        )
