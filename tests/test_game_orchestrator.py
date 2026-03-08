import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from affine.src.executor.game_orchestrator import (
    GameOrchestrator, GameState,
)


@pytest.fixture
def orchestrator():
    mock_wallet = MagicMock()
    mock_wallet.hotkey.ss58_address = "test_hotkey"
    mock_wallet.hotkey.sign.return_value = b"signature"
    return GameOrchestrator(worker_id=1, wallet=mock_wallet, game_types=["tictactoe"])


@pytest.fixture
def orchestrator_with_api(orchestrator):
    orchestrator.api_client = MagicMock()
    return orchestrator


class TestTicTacToeLogic:
    def test_valid_move_empty_cell(self, orchestrator):
        board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        assert orchestrator._is_valid_tictactoe_move(board, (0, 0)) is True
        assert orchestrator._is_valid_tictactoe_move(board, (1, 1)) is True
        assert orchestrator._is_valid_tictactoe_move(board, (2, 2)) is True

    def test_invalid_move_occupied_cell(self, orchestrator):
        board = [[1, 0, 0], [0, 2, 0], [0, 0, 0]]
        assert orchestrator._is_valid_tictactoe_move(board, (0, 0)) is False
        assert orchestrator._is_valid_tictactoe_move(board, (1, 1)) is False

    def test_invalid_move_out_of_bounds(self, orchestrator):
        board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        assert orchestrator._is_valid_tictactoe_move(board, (-1, 0)) is False
        assert orchestrator._is_valid_tictactoe_move(board, (0, 3)) is False
        assert orchestrator._is_valid_tictactoe_move(board, (5, 5)) is False

    def test_check_winner_row(self, orchestrator):
        assert orchestrator._check_tictactoe_winner([[1, 1, 1], [0, 2, 0], [2, 0, 0]]) == 1
        assert orchestrator._check_tictactoe_winner([[1, 0, 1], [2, 2, 2], [0, 1, 0]]) == 2

    def test_check_winner_column(self, orchestrator):
        assert orchestrator._check_tictactoe_winner([[1, 2, 0], [1, 2, 0], [1, 0, 0]]) == 1

    def test_check_winner_diagonal(self, orchestrator):
        assert orchestrator._check_tictactoe_winner([[1, 2, 0], [2, 1, 0], [0, 0, 1]]) == 1
        assert orchestrator._check_tictactoe_winner([[1, 1, 2], [0, 2, 0], [2, 0, 1]]) == 2

    def test_check_no_winner(self, orchestrator):
        assert orchestrator._check_tictactoe_winner([[1, 2, 1], [1, 2, 2], [0, 1, 0]]) is None

    def test_board_full(self, orchestrator):
        assert orchestrator._is_board_full([[1, 2, 1], [1, 2, 2], [2, 1, 1]]) is True
        assert orchestrator._is_board_full([[1, 2, 1], [1, 0, 2], [2, 1, 1]]) is False

    def test_format_board(self, orchestrator):
        formatted = orchestrator._format_tictactoe_board([[1, 0, 2], [0, 1, 0], [2, 0, 1]])
        assert "X" in formatted
        assert "O" in formatted
        assert "." in formatted

    def test_get_first_available_move(self, orchestrator):
        assert orchestrator._get_first_available_move([[1, 2, 0], [0, 0, 0], [0, 0, 0]]) == (0, 2)
        assert orchestrator._get_first_available_move([[1, 2, 1], [2, 1, 2], [1, 2, 1]]) == (0, 0)


class TestGameSession:
    def test_create_session_from_match(self, orchestrator, sample_match):
        session = orchestrator._create_session(sample_match)
        assert session.match_uuid == sample_match["match_uuid"]
        assert session.game_type == sample_match["game_type"]
        assert len(session.players) == 2
        assert session.state == GameState.PENDING

    def test_session_player_states(self, orchestrator, sample_match):
        session = orchestrator._create_session(sample_match)
        for i, player in enumerate(session.players):
            assert player.slot == i
            assert player.miner_hotkey is not None
            assert player.moves_made == 0
            assert player.total_latency_ms == 0

    def test_build_participant_results_win(self, orchestrator, sample_match):
        session = orchestrator._create_session(sample_match)
        session.state = GameState.COMPLETED
        session.outcome = "win"
        session.winner_slot = 0
        session.players[0].moves_made = 5
        session.players[0].total_latency_ms = 2500
        session.players[1].moves_made = 4
        session.players[1].total_latency_ms = 2000

        results = orchestrator._build_participant_results(session)
        assert len(results) == 2
        assert results[0]["outcome"] == "win"
        assert results[0]["score"] == 1.0
        assert results[0]["avg_move_latency_ms"] == 500
        assert results[1]["outcome"] == "loss"
        assert results[1]["score"] == 0.0

    def test_build_participant_results_draw(self, orchestrator, sample_match):
        session = orchestrator._create_session(sample_match)
        session.state = GameState.COMPLETED
        session.outcome = "draw"
        session.winner_slot = None
        session.players[0].moves_made = 5
        session.players[1].moves_made = 4

        results = orchestrator._build_participant_results(session)
        for result in results:
            assert result["outcome"] == "draw"
            assert result["score"] == 0.5


class TestGameExecution:
    @pytest.mark.asyncio
    async def test_execute_tictactoe_quick_win(self, orchestrator_with_api, sample_match):
        moves = iter(["0 0", "1 1", "0 1", "2 2", "0 2"])

        async def mock_call_model(chute_slug, prompt):
            return next(moves)

        orchestrator_with_api._call_model = mock_call_model
        orchestrator_with_api.submit_result = AsyncMock(return_value=True)

        session = await orchestrator_with_api.execute_game(sample_match)
        assert session is not None
        assert session.state == GameState.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_unknown_game_type(self, orchestrator_with_api):
        match = {
            "match_uuid": "test", "game_type": "unknown_game",
            "task_id": 1, "participants": [], "game_config": {},
        }
        orchestrator_with_api.submit_result = AsyncMock(return_value=True)

        session = await orchestrator_with_api.execute_game(match)
        assert session.state == GameState.ERROR
        assert session.outcome == "error"


class TestChessIntegration:
    def test_chess_module_available(self):
        try:
            import chess
        except ImportError:
            pytest.skip("python-chess not installed")

    @pytest.mark.asyncio
    async def test_chess_session_initialization(self):
        try:
            import chess
        except ImportError:
            pytest.skip("python-chess not installed")

        mock_wallet = MagicMock()
        mock_wallet.hotkey.ss58_address = "test_hotkey"
        orch = GameOrchestrator(worker_id=1, wallet=mock_wallet, game_types=["chess"])

        match = {
            "match_uuid": "chess-001", "game_type": "chess", "task_id": 1,
            "participants": [
                {"slot": 0, "miner_hotkey": "white", "model_revision": "v1", "chute_slug": "c1"},
                {"slot": 1, "miner_hotkey": "black", "model_revision": "v1", "chute_slug": "c2"},
            ],
            "game_config": {"timeout_per_move": 60, "max_moves": 200},
        }
        assert orch._create_session(match).game_type == "chess"


class TestOrchestratorLifecycle:
    @pytest.mark.asyncio
    async def test_initialize(self, orchestrator):
        with patch("affine.src.executor.game_orchestrator.create_api_client") as mock_create:
            mock_create.return_value = MagicMock()
            await orchestrator.initialize()
            mock_create.assert_called_once()
            assert orchestrator.api_client is not None

    def test_start_sets_running(self, orchestrator):
        orchestrator.start()
        assert orchestrator.running is True
        assert orchestrator.execution_semaphore is not None

    @pytest.mark.asyncio
    async def test_stop_clears_running(self, orchestrator):
        orchestrator.start()
        await orchestrator.stop()
        assert orchestrator.running is False


class TestFetchAndSubmit:
    @pytest.mark.asyncio
    async def test_fetch_match_returns_match(self, orchestrator_with_api, sample_match):
        orchestrator_with_api.api_client.post = AsyncMock(return_value={"matches": [sample_match]})
        match = await orchestrator_with_api.fetch_match()
        assert match is not None
        assert match["match_uuid"] == sample_match["match_uuid"]

    @pytest.mark.asyncio
    async def test_fetch_match_returns_none_when_empty(self, orchestrator_with_api):
        orchestrator_with_api.api_client.post = AsyncMock(return_value={"matches": []})
        assert await orchestrator_with_api.fetch_match() is None

    @pytest.mark.asyncio
    async def test_submit_result(self, orchestrator_with_api, sample_match):
        session = orchestrator_with_api._create_session(sample_match)
        session.state = GameState.COMPLETED
        session.outcome = "win"
        session.winner_slot = 0
        session.started_at = 1705000000000
        session.completed_at = 1705000030000
        session.move_history = [{"turn": 0, "move": "test"}]

        orchestrator_with_api.api_client.post = AsyncMock(return_value={"success": True})
        participants = orchestrator_with_api._build_participant_results(session)
        result = await orchestrator_with_api.submit_result(session, participants)
        assert result is True
        orchestrator_with_api.api_client.post.assert_called_once()
