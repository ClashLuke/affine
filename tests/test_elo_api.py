import pytest
from decimal import Decimal
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient


@pytest.fixture
def elo_client(mock_elo_ratings_dao, mock_match_records_dao):
    from fastapi import FastAPI
    from affine.api.routers.elo import router, get_elo_ratings_dao, get_match_records_dao
    from affine.api.dependencies import rate_limit_read

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_elo_ratings_dao] = lambda: mock_elo_ratings_dao
    app.dependency_overrides[get_match_records_dao] = lambda: mock_match_records_dao
    app.dependency_overrides[rate_limit_read] = lambda: None

    return TestClient(app), mock_elo_ratings_dao, mock_match_records_dao


class TestEloRatingsEndpoints:
    def test_get_ratings_for_miner(self, elo_client):
        client, mock_dao, _ = elo_client
        mock_dao._ratings["test_hotkey#v1#game:tictactoe"] = {
            "miner_hotkey": "test_hotkey", "model_revision": "v1",
            "env": "game:tictactoe", "rating": Decimal("1550"),
            "peak_rating": Decimal("1600"), "matches_played": 25,
            "wins": 15, "losses": 8, "draws": 2,
        }

        response = client.get("/api/v1/elo/ratings/test_hotkey", params={"model_revision": "v1"})
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_rating_for_specific_env(self, elo_client):
        client, mock_dao, _ = elo_client
        mock_dao._ratings["test_hotkey#v1#game:chess"] = {
            "miner_hotkey": "test_hotkey", "model_revision": "v1",
            "env": "game:chess", "rating": Decimal("1700"),
            "peak_rating": Decimal("1750"), "matches_played": 100,
            "wins": 60, "losses": 35, "draws": 5,
        }

        async def mock_get_rating(hotkey, revision, env):
            return mock_dao._ratings.get(f"{hotkey}#{revision}#{env}")
        mock_dao.get_rating = AsyncMock(side_effect=mock_get_rating)

        response = client.get("/api/v1/elo/ratings/test_hotkey/game:chess", params={"model_revision": "v1"})
        assert response.status_code == 200
        data = response.json()
        assert data["miner_hotkey"] == "test_hotkey"
        assert data["env"] == "game:chess"

    def test_get_rating_returns_default_when_not_found(self, elo_client):
        client, mock_dao, _ = elo_client
        mock_dao.get_rating = AsyncMock(return_value=None)

        response = client.get("/api/v1/elo/ratings/unknown_hotkey/game:tictactoe", params={"model_revision": "v1"})
        assert response.status_code == 200
        data = response.json()
        assert data["rating"] == 1500.0
        assert data["matches_played"] == 0


class TestLeaderboardEndpoints:
    def test_get_leaderboard(self, elo_client):
        client, mock_dao, _ = elo_client
        mock_dao._ratings["player1#v1#game:tictactoe"] = {
            "miner_hotkey": "player1", "model_revision": "v1",
            "env": "game:tictactoe", "rating": Decimal("1800"),
            "matches_played": 50, "wins": 40, "losses": 8, "draws": 2,
        }
        mock_dao._ratings["player2#v1#game:tictactoe"] = {
            "miner_hotkey": "player2", "model_revision": "v1",
            "env": "game:tictactoe", "rating": Decimal("1600"),
            "matches_played": 30, "wins": 18, "losses": 10, "draws": 2,
        }

        response = client.get("/api/v1/elo/leaderboard/game:tictactoe")
        assert response.status_code == 200
        data = response.json()
        assert "entries" in data
        assert "total_players" in data

    def test_leaderboard_respects_limit(self, elo_client):
        client, mock_dao, _ = elo_client
        for i in range(20):
            mock_dao._ratings[f"player{i}#v1#game:test"] = {
                "miner_hotkey": f"player{i}", "model_revision": "v1",
                "env": "game:test", "rating": Decimal(1500 + i * 10),
                "matches_played": 10, "wins": 5, "losses": 5, "draws": 0,
            }

        response = client.get("/api/v1/elo/leaderboard/game:test", params={"limit": 5})
        assert response.status_code == 200
        assert len(response.json()["entries"]) <= 5

    def test_leaderboard_sorted_by_rating(self, elo_client):
        client, mock_dao, _ = elo_client
        for name, rating in [("low", "1400"), ("high", "1800"), ("mid", "1600")]:
            mock_dao._ratings[f"{name}#v1#game:test"] = {
                "miner_hotkey": name, "model_revision": "v1", "env": "game:test",
                "rating": Decimal(rating), "matches_played": 10,
                "wins": 5, "losses": 5, "draws": 0,
            }

        response = client.get("/api/v1/elo/leaderboard/game:test")
        assert response.status_code == 200
        entries = response.json()["entries"]
        if len(entries) >= 2:
            assert entries[0]["rating"] >= entries[1]["rating"]


class TestMatchHistoryEndpoints:
    def test_get_match_history(self, elo_client):
        client, _, mock_match_dao = elo_client
        mock_match_dao._matches.append({
            "match_uuid": "match-001", "env": "game:tictactoe",
            "match_type": "game", "task_id": 1, "timestamp": 1705000000,
            "participants": [
                {"miner_hotkey": "test_hotkey", "model_revision": "v1",
                 "outcome": "win", "elo_before": 1500, "elo_after": 1520, "elo_change": 20},
                {"miner_hotkey": "opponent", "model_revision": "v1", "outcome": "loss"},
            ],
        })

        response = client.get("/api/v1/elo/history/test_hotkey", params={"model_revision": "v1"})
        assert response.status_code == 200
        data = response.json()
        assert data["miner_hotkey"] == "test_hotkey"
        assert "matches" in data
        assert "total_matches" in data

    def test_match_history_with_env_filter(self, elo_client):
        client, _, mock_match_dao = elo_client
        mock_match_dao._matches.extend([
            {"match_uuid": "match-ttt", "env": "game:tictactoe", "task_id": 1,
             "timestamp": 1705000000, "participants": [{"miner_hotkey": "test", "outcome": "win"}]},
            {"match_uuid": "match-chess", "env": "game:chess", "task_id": 2,
             "timestamp": 1705000001, "participants": [{"miner_hotkey": "test", "outcome": "loss"}]},
        ])

        response = client.get(
            "/api/v1/elo/history/test",
            params={"model_revision": "v1", "env": "game:tictactoe"},
        )
        assert response.status_code == 200


class TestValidation:
    def test_leaderboard_limit_max(self, elo_client):
        client, _, _ = elo_client
        assert client.get("/api/v1/elo/leaderboard/game:test", params={"limit": 1000}).status_code == 422

    def test_leaderboard_limit_min(self, elo_client):
        client, _, _ = elo_client
        assert client.get("/api/v1/elo/leaderboard/game:test", params={"limit": 0}).status_code == 422

    def test_history_limit_max(self, elo_client):
        client, _, _ = elo_client
        assert client.get("/api/v1/elo/history/test", params={"model_revision": "v1", "limit": 500}).status_code == 422

    def test_missing_model_revision(self, elo_client):
        client, _, _ = elo_client
        assert client.get("/api/v1/elo/ratings/test_hotkey").status_code == 422


class TestWinRateCalculation:
    def test_win_rate_calculated(self, elo_client):
        client, mock_dao, _ = elo_client
        mock_dao._ratings["player#v1#game:test"] = {
            "miner_hotkey": "player", "model_revision": "v1", "env": "game:test",
            "rating": Decimal("1600"), "peak_rating": Decimal("1650"),
            "matches_played": 100, "wins": 60, "losses": 30, "draws": 10,
        }

        async def mock_get_rating(hotkey, revision, env):
            return mock_dao._ratings.get(f"{hotkey}#{revision}#{env}")
        mock_dao.get_rating = AsyncMock(side_effect=mock_get_rating)

        response = client.get("/api/v1/elo/ratings/player/game:test", params={"model_revision": "v1"})
        assert response.status_code == 200
        assert response.json()["win_rate"] == 0.6
