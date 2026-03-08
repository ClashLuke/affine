"""
Game Orchestrator - Multi-Party Game Executor

Executes multi-party games (tic-tac-toe, chess, etc.) by coordinating
turns between multiple model endpoints.
"""

import asyncio
import time
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import bittensor as bt

from affine.utils.api_client import create_api_client, APIClient
from affine.src.executor.metrics import WorkerMetrics
from affine.src.executor.logging_utils import safe_log


class GameState(Enum):
    """Game state enumeration."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class PlayerState:
    """State for a single player in a game."""

    slot: int
    miner_hotkey: str
    model_revision: str
    chute_slug: str
    role: Optional[str] = None
    moves_made: int = 0
    total_latency_ms: int = 0
    timeouts: int = 0
    errors: int = 0


@dataclass
class GameSession:
    """Session state for a game in progress."""

    match_uuid: str
    game_type: str
    task_id: int
    players: List[PlayerState]
    game_config: Dict[str, Any]
    state: GameState = GameState.PENDING
    current_turn: int = 0
    current_player_slot: int = 0
    game_board: Any = None
    move_history: List[Dict[str, Any]] = field(default_factory=list)
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    winner_slot: Optional[int] = None
    outcome: str = ""  # "win", "draw", "timeout", "error"


class GameOrchestrator:
    """
    Orchestrates multi-party game execution.

    Coordinates turn-based games between multiple model endpoints,
    handling move validation, timeouts, and result calculation.
    """

    def __init__(
        self,
        worker_id: int,
        wallet: bt.Wallet,
        game_types: Optional[List[str]] = None,
        max_concurrent_games: int = 2,
    ):
        """
        Initialize the game orchestrator.

        Args:
            worker_id: Unique worker ID
            wallet: Bittensor wallet for signing
            game_types: List of game types to handle (default: all)
            max_concurrent_games: Maximum concurrent game executions
        """
        self.worker_id = worker_id
        self.wallet = wallet
        self.hotkey = wallet.hotkey.ss58_address
        self.game_types = game_types or ["tictactoe", "chess"]
        self.max_concurrent_games = max_concurrent_games

        self.running = False
        self.metrics = WorkerMetrics(
            worker_id=worker_id,
            env="game",
        )

        self.api_client: Optional[APIClient] = None
        self.game_queue: asyncio.Queue = asyncio.Queue()
        self.execution_semaphore: Optional[asyncio.Semaphore] = None
        self.executor_tasks = []

    async def initialize(self):
        """Initialize the orchestrator (API client)."""
        safe_log(f"[game] Initializing GameOrchestrator worker {self.worker_id}...", "INFO")

        if not self.wallet or not self.hotkey:
            raise RuntimeError("Wallet not configured for orchestrator")

        self.api_client = await create_api_client()

        safe_log(
            f"[game] GameOrchestrator initialized for games: {self.game_types}",
            "INFO",
        )

    def start(self):
        """Start the orchestrator fetch and execution loops."""
        self.running = True
        self.execution_semaphore = asyncio.Semaphore(self.max_concurrent_games)

    async def stop(self):
        """Stop the orchestrator gracefully."""
        self.running = False

        # Wait for pending games to complete
        for task in self.executor_tasks:
            try:
                task.cancel()
                await task
            except asyncio.CancelledError:
                pass

        safe_log(f"[game] GameOrchestrator stopped", "INFO")

    async def fetch_match(self) -> Optional[Dict[str, Any]]:
        """
        Fetch a match from the match pool.

        Sends separate requests per game type since the DAO uses single
        equality filter (not comma-separated lists).

        Returns:
            Match data or None if no matches available
        """
        try:
            for game_type in (self.game_types or [None]):
                response = await self.api_client.post(
                    "/api/v1/matches/fetch",
                    params={"game_type": game_type},
                )
                if response and response.get("matches"):
                    return response["matches"][0]

            return None

        except Exception as e:
            safe_log(f"[game] Error fetching match: {e}", "ERROR")
            return None

    async def submit_result(
        self,
        session: GameSession,
        participants: List[Dict[str, Any]],
    ) -> bool:
        """
        Submit game result to the API.

        Args:
            session: Completed game session
            participants: List of participant results

        Returns:
            True if submission successful
        """
        try:
            sign_ts = int(time.time())
            signed_message = f"{session.match_uuid}:{session.task_id}:{sign_ts}"

            payload = {
                "match_uuid": session.match_uuid,
                "game_type": session.game_type,
                "task_id": session.task_id,
                "participants": participants,
                "game_history": session.move_history,
                "total_moves": len(session.move_history),
                "total_time_ms": (session.completed_at or 0) - (session.started_at or 0),
                "signature": self._sign_result(signed_message),
                "signed_message": signed_message,
            }

            response = await self.api_client.post("/api/v1/matches/submit", json=payload)

            return response and response.get("success", False)

        except Exception as e:
            safe_log(f"[game] Error submitting result: {e}", "ERROR")
            return False

    def _sign_result(self, message: str) -> str:
        """Sign a message with the executor's hotkey."""
        try:
            signature = self.wallet.hotkey.sign(message.encode())
            return signature.hex()
        except Exception:
            return ""

    async def execute_game(self, match: Dict[str, Any]) -> Optional[GameSession]:
        """
        Execute a complete game.

        Args:
            match: Match data from the pool

        Returns:
            Completed GameSession or None on failure
        """
        session = self._create_session(match)

        try:
            session.state = GameState.IN_PROGRESS
            session.started_at = int(time.time() * 1000)

            # Execute game based on type
            if session.game_type == "tictactoe":
                await self._execute_tictactoe(session)
            elif session.game_type == "chess":
                await self._execute_chess(session)
            else:
                safe_log(f"[game] Unknown game type: {session.game_type}", "ERROR")
                session.state = GameState.ERROR
                session.outcome = "error"

            session.completed_at = int(time.time() * 1000)

            # Build participant results
            participants = self._build_participant_results(session)

            # Submit results
            await self.submit_result(session, participants)

            return session

        except asyncio.TimeoutError:
            session.state = GameState.TIMEOUT
            session.outcome = "timeout"
            session.completed_at = int(time.time() * 1000)
            return session

        except Exception as e:
            safe_log(f"[game] Error executing game: {e}", "ERROR")
            session.state = GameState.ERROR
            session.outcome = "error"
            session.completed_at = int(time.time() * 1000)
            return session

    def _create_session(self, match: Dict[str, Any]) -> GameSession:
        """Create a game session from match data."""
        players = []
        for p in match.get("participants", []):
            players.append(
                PlayerState(
                    slot=p.get("slot", 0),
                    miner_hotkey=p.get("miner_hotkey", ""),
                    model_revision=p.get("model_revision", ""),
                    chute_slug=p.get("chute_slug", ""),
                    role=p.get("role"),
                )
            )

        return GameSession(
            match_uuid=match.get("match_uuid", ""),
            game_type=match.get("game_type", ""),
            task_id=match.get("task_id", 0),
            players=players,
            game_config=match.get("game_config", {}),
        )

    def _build_participant_results(self, session: GameSession) -> List[Dict[str, Any]]:
        """Build participant result dictionaries from session."""
        results = []

        for player in session.players:
            if session.winner_slot is not None:
                if player.slot == session.winner_slot:
                    outcome = "win"
                    score = 1.0
                elif session.outcome == "draw":
                    outcome = "draw"
                    score = 0.5
                else:
                    outcome = "loss"
                    score = 0.0
            else:
                outcome = session.outcome or "draw"
                score = 0.5 if outcome == "draw" else 0.0

            avg_latency = (
                player.total_latency_ms // player.moves_made
                if player.moves_made > 0
                else 0
            )

            results.append(
                {
                    "slot": player.slot,
                    "miner_hotkey": player.miner_hotkey,
                    "model_revision": player.model_revision,
                    "outcome": outcome,
                    "score": score,
                    "total_moves": player.moves_made,
                    "avg_move_latency_ms": avg_latency,
                    "total_latency_ms": player.total_latency_ms,
                }
            )

        return results

    async def _execute_tictactoe(self, session: GameSession):
        """
        Execute a tic-tac-toe game.

        Simple 3x3 board, players alternate, first to 3 in a row wins.
        """
        # Initialize board (0 = empty, 1 = player 0, 2 = player 1)
        board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        session.game_board = board

        timeout_per_move = session.game_config.get("timeout_per_move", 30)
        max_moves = session.game_config.get("max_moves", 9)

        current_player = 0  # Player 0 starts (X)

        for turn in range(max_moves):
            session.current_turn = turn
            session.current_player_slot = current_player

            player = session.players[current_player]

            # Get move from player
            start_time = time.time()
            try:
                move = await asyncio.wait_for(
                    self._get_player_move_tictactoe(session, player, board),
                    timeout=timeout_per_move,
                )
            except asyncio.TimeoutError:
                player.timeouts += 1
                # Player loses on timeout
                session.winner_slot = 1 - current_player
                session.outcome = "win"
                session.state = GameState.COMPLETED
                return

            latency_ms = int((time.time() - start_time) * 1000)
            player.total_latency_ms += latency_ms
            player.moves_made += 1

            # Validate and apply move
            if not self._is_valid_tictactoe_move(board, move):
                player.errors += 1
                # Invalid move = loss
                session.winner_slot = 1 - current_player
                session.outcome = "win"
                session.state = GameState.COMPLETED
                return

            row, col = move
            board[row][col] = current_player + 1

            # Record move
            session.move_history.append(
                {
                    "turn": turn,
                    "player": current_player,
                    "move": {"row": row, "col": col},
                    "latency_ms": latency_ms,
                }
            )

            # Check for winner
            winner = self._check_tictactoe_winner(board)
            if winner is not None:
                session.winner_slot = winner - 1  # Convert 1/2 to 0/1
                session.outcome = "win"
                session.state = GameState.COMPLETED
                return

            # Check for draw (board full)
            if self._is_board_full(board):
                session.outcome = "draw"
                session.state = GameState.COMPLETED
                return

            # Switch player
            current_player = 1 - current_player

        # Max moves reached without winner = draw
        session.outcome = "draw"
        session.state = GameState.COMPLETED

    async def _get_player_move_tictactoe(
        self,
        session: GameSession,
        player: PlayerState,
        board: List[List[int]],
    ) -> Tuple[int, int]:
        """
        Get a move from a player for tic-tac-toe.

        Calls the player's model endpoint with the current board state.

        Args:
            session: Game session
            player: Player state
            board: Current board state

        Returns:
            (row, col) tuple for the move
        """
        # Format board state for the model
        board_str = self._format_tictactoe_board(board)
        player_symbol = "X" if player.slot == 0 else "O"

        prompt = (
            f"You are playing Tic-Tac-Toe as {player_symbol}. "
            f"The current board state is:\n{board_str}\n"
            f"Choose your move by specifying the row (0-2) and column (0-2). "
            f"Respond with just two numbers separated by a space, e.g., '1 2' for row 1, column 2."
        )

        try:
            # Call the model endpoint via chute
            response = await self._call_model(player.chute_slug, prompt)

            # Parse response
            parts = response.strip().split()
            if len(parts) >= 2:
                row = int(parts[0])
                col = int(parts[1])
                return (row, col)

            # Try to extract numbers from response
            import re
            numbers = re.findall(r"\d+", response)
            if len(numbers) >= 2:
                return (int(numbers[0]), int(numbers[1]))

            # Default to first available move
            return self._get_first_available_move(board)

        except Exception as e:
            safe_log(f"[game] Error getting player move: {e}", "ERROR")
            return self._get_first_available_move(board)

    async def _call_model(self, chute_slug: str, prompt: str) -> str:
        """
        Call a model endpoint.

        Args:
            chute_slug: Chute slug for the model
            prompt: Prompt to send

        Returns:
            Model response text
        """
        # This would normally call the chute endpoint
        # For now, return a placeholder
        # In production, integrate with the existing chute calling infrastructure

        try:
            # TODO: Integrate with actual chute calling
            # For now, simulate a response for testing
            import random

            # Simulate thinking time
            await asyncio.sleep(0.5)

            # Return random valid move (for testing)
            return f"{random.randint(0, 2)} {random.randint(0, 2)}"

        except Exception as e:
            safe_log(f"[game] Error calling model: {e}", "ERROR")
            return "0 0"

    def _format_tictactoe_board(self, board: List[List[int]]) -> str:
        """Format board for display."""
        symbols = {0: ".", 1: "X", 2: "O"}
        lines = []
        for row in board:
            lines.append(" ".join(symbols[cell] for cell in row))
        return "\n".join(lines)

    def _is_valid_tictactoe_move(
        self,
        board: List[List[int]],
        move: Tuple[int, int],
    ) -> bool:
        """Check if a move is valid."""
        row, col = move
        if row < 0 or row > 2 or col < 0 or col > 2:
            return False
        return board[row][col] == 0

    def _get_first_available_move(self, board: List[List[int]]) -> Tuple[int, int]:
        """Get the first available move on the board."""
        for row in range(3):
            for col in range(3):
                if board[row][col] == 0:
                    return (row, col)
        return (0, 0)

    def _check_tictactoe_winner(self, board: List[List[int]]) -> Optional[int]:
        """
        Check if there's a winner.

        Returns:
            1 or 2 for the winning player, None if no winner
        """
        # Check rows
        for row in board:
            if row[0] != 0 and row[0] == row[1] == row[2]:
                return row[0]

        # Check columns
        for col in range(3):
            if board[0][col] != 0 and board[0][col] == board[1][col] == board[2][col]:
                return board[0][col]

        # Check diagonals
        if board[0][0] != 0 and board[0][0] == board[1][1] == board[2][2]:
            return board[0][0]
        if board[0][2] != 0 and board[0][2] == board[1][1] == board[2][0]:
            return board[0][2]

        return None

    def _is_board_full(self, board: List[List[int]]) -> bool:
        """Check if the board is full."""
        for row in board:
            for cell in row:
                if cell == 0:
                    return False
        return True

    async def _execute_chess(self, session: GameSession):
        """
        Execute a chess game.

        Uses python-chess for game logic.
        """
        try:
            import chess
        except ImportError:
            safe_log("[game] python-chess not installed, cannot play chess", "ERROR")
            session.state = GameState.ERROR
            session.outcome = "error"
            return

        board = chess.Board()
        session.game_board = board

        timeout_per_move = session.game_config.get("timeout_per_move", 60)
        max_moves = session.game_config.get("max_moves", 200)

        current_player = 0  # Player 0 = White

        for turn in range(max_moves):
            session.current_turn = turn
            session.current_player_slot = current_player

            player = session.players[current_player]

            # Get move from player
            start_time = time.time()
            try:
                move_uci = await asyncio.wait_for(
                    self._get_player_move_chess(session, player, board),
                    timeout=timeout_per_move,
                )
            except asyncio.TimeoutError:
                player.timeouts += 1
                session.winner_slot = 1 - current_player
                session.outcome = "win"
                session.state = GameState.COMPLETED
                return

            latency_ms = int((time.time() - start_time) * 1000)
            player.total_latency_ms += latency_ms
            player.moves_made += 1

            # Validate and apply move
            try:
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                    raise ValueError("Illegal move")
                board.push(move)
            except Exception as e:
                player.errors += 1
                session.winner_slot = 1 - current_player
                session.outcome = "win"
                session.state = GameState.COMPLETED
                return

            # Record move
            session.move_history.append(
                {
                    "turn": turn,
                    "player": current_player,
                    "move": move_uci,
                    "latency_ms": latency_ms,
                }
            )

            # Check for game end
            if board.is_checkmate():
                session.winner_slot = current_player
                session.outcome = "win"
                session.state = GameState.COMPLETED
                return

            if board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves():
                session.outcome = "draw"
                session.state = GameState.COMPLETED
                return

            # Switch player
            current_player = 1 - current_player

        # Max moves reached
        session.outcome = "draw"
        session.state = GameState.COMPLETED

    async def _get_player_move_chess(
        self,
        session: GameSession,
        player: PlayerState,
        board: Any,
    ) -> str:
        """
        Get a move from a player for chess.

        Args:
            session: Game session
            player: Player state
            board: chess.Board instance

        Returns:
            Move in UCI notation (e.g., "e2e4")
        """
        color = "White" if player.slot == 0 else "Black"

        prompt = (
            f"You are playing Chess as {color}. "
            f"The current board state in FEN is:\n{board.fen()}\n"
            f"Legal moves: {', '.join(m.uci() for m in board.legal_moves)}\n"
            f"Choose your move in UCI notation (e.g., 'e2e4'). "
            f"Respond with just the move."
        )

        try:
            response = await self._call_model(player.chute_slug, prompt)

            # Extract UCI move from response
            import re
            uci_pattern = r"[a-h][1-8][a-h][1-8][qrbn]?"
            matches = re.findall(uci_pattern, response.lower())
            if matches:
                return matches[0]

            # Return first legal move as fallback
            return list(board.legal_moves)[0].uci()

        except Exception as e:
            safe_log(f"[game] Error getting chess move: {e}", "ERROR")
            return list(board.legal_moves)[0].uci()

    async def run_loop(self):
        """Main execution loop."""
        while self.running:
            try:
                # Fetch a match
                match = await self.fetch_match()

                if match:
                    async with self.execution_semaphore:
                        await self.execute_game(match)
                else:
                    # No matches available, wait before retrying
                    await asyncio.sleep(5)

            except Exception as e:
                safe_log(f"[game] Error in run loop: {e}", "ERROR")
                await asyncio.sleep(5)


async def create_game_orchestrator(
    worker_id: int,
    wallet: bt.Wallet,
    game_types: Optional[List[str]] = None,
) -> GameOrchestrator:
    """Factory function to create a GameOrchestrator."""
    orchestrator = GameOrchestrator(worker_id, wallet, game_types)
    await orchestrator.initialize()
    return orchestrator
