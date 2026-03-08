import sys
import os
import asyncio
from decimal import Decimal
from typing import AsyncGenerator, Generator, Dict, Any
from unittest.mock import MagicMock, AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "affine", "src"))

import pytest
import pytest_asyncio

os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["DYNAMODB_TABLE_PREFIX"] = "test_"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def mock_dynamodb_client() -> AsyncGenerator[MagicMock, None]:
    mock_client = MagicMock()
    tables: Dict[str, Dict[str, Any]] = {}

    async def mock_put_item(TableName: str, Item: Dict[str, Any], **kwargs):
        if TableName not in tables:
            tables[TableName] = {}
        pk = Item.get("pk", {}).get("S", "")
        sk = Item.get("sk", {}).get("S", "")
        tables[TableName][f"{pk}#{sk}"] = Item
        return {}

    async def mock_get_item(TableName: str, Key: Dict[str, Any], **kwargs):
        if TableName not in tables:
            return {}
        pk = Key.get("pk", {}).get("S", "")
        sk = Key.get("sk", {}).get("S", "")
        item = tables[TableName].get(f"{pk}#{sk}")
        return {"Item": item} if item else {}

    async def mock_query(TableName: str, **kwargs):
        if TableName not in tables:
            return {"Items": [], "Count": 0}
        eav = kwargs.get("ExpressionAttributeValues", {})
        pk_value = eav.get(":pk", {}).get("S", "")
        items = [
            item for item in tables[TableName].values()
            if item.get("pk", {}).get("S", "").startswith(pk_value)
        ]
        return {"Items": items, "Count": len(items)}

    async def mock_delete_item(TableName: str, Key: Dict[str, Any], **kwargs):
        if TableName not in tables:
            return {}
        pk = Key.get("pk", {}).get("S", "")
        sk = Key.get("sk", {}).get("S", "")
        tables[TableName].pop(f"{pk}#{sk}", None)
        return {}

    mock_client.put_item = AsyncMock(side_effect=mock_put_item)
    mock_client.get_item = AsyncMock(side_effect=mock_get_item)
    mock_client.query = AsyncMock(side_effect=mock_query)
    mock_client.delete_item = AsyncMock(side_effect=mock_delete_item)
    mock_client.batch_write_item = AsyncMock(return_value={})
    tables.clear()
    yield mock_client


@pytest.fixture
def elo_config():
    from elo.config import EloConfig
    return EloConfig(
        K_FACTOR=32,
        K_FACTOR_NEW_PLAYER=32,
        K_FACTOR_ESTABLISHED=24,
        K_FACTOR_ELITE=16,
        NEW_PLAYER_THRESHOLD=30,
        ESTABLISHED_THRESHOLD=100,
        ELITE_RATING_THRESHOLD=Decimal("2000"),
        DEFAULT_RATING=Decimal("1500"),
        SCALE=400,
        SCORE_MARGIN=Decimal("0.01"),
    )


@pytest.fixture
def elo_calculator(elo_config):
    from elo.calculator import EloCalculator
    return EloCalculator(elo_config)


@pytest.fixture
def match_engine(elo_config):
    from elo.match_engine import MatchEngine
    return MatchEngine(elo_config)


@pytest.fixture
def sample_players():
    return [
        {"hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", "revision": "abc123", "rating": Decimal("1500"), "matches_played": 0},
        {"hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty", "revision": "def456", "rating": Decimal("1600"), "matches_played": 50},
        {"hotkey": "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy", "revision": "ghi789", "rating": Decimal("1400"), "matches_played": 100},
    ]


@pytest.fixture
def mock_elo_ratings_dao():
    mock_dao = MagicMock()
    ratings = {}

    async def mock_get_rating(hotkey, revision, env):
        return ratings.get(f"{hotkey}#{revision}#{env}")

    async def mock_save_rating(**kwargs):
        key = f"{kwargs['miner_hotkey']}#{kwargs['model_revision']}#{kwargs['env']}"
        ratings[key] = kwargs
        return kwargs

    async def mock_get_leaderboard(env, limit=100):
        env_ratings = [r for r in ratings.values() if r.get("env") == env]
        return sorted(env_ratings, key=lambda x: x.get("rating", 0), reverse=True)[:limit]

    async def mock_get_all_ratings_for_miner(hotkey, revision):
        return [r for r in ratings.values()
                if r.get("miner_hotkey") == hotkey and r.get("model_revision") == revision]

    async def mock_get_all_ratings_for_env(env):
        return [r for r in ratings.values() if r.get("env") == env]

    mock_dao.get_rating = AsyncMock(side_effect=mock_get_rating)
    mock_dao.save_rating = AsyncMock(side_effect=mock_save_rating)
    mock_dao.get_leaderboard = AsyncMock(side_effect=mock_get_leaderboard)
    mock_dao.get_all_ratings_for_miner = AsyncMock(side_effect=mock_get_all_ratings_for_miner)
    mock_dao.get_all_ratings_for_env = AsyncMock(side_effect=mock_get_all_ratings_for_env)
    mock_dao._ratings = ratings
    return mock_dao


@pytest.fixture
def mock_match_records_dao():
    mock_dao = MagicMock()
    matches = []

    async def mock_save_match(**kwargs):
        matches.append(kwargs)
        return kwargs

    async def mock_get_matches_for_miner(hotkey, env=None, limit=50):
        result = []
        for m in matches:
            for p in m.get("participants", []):
                if p.get("miner_hotkey") == hotkey:
                    if env is None or m.get("env") == env:
                        result.append(m)
                    break
        return result[:limit]

    async def mock_count_matches_for_miner(hotkey, env=None):
        count = 0
        for m in matches:
            for p in m.get("participants", []):
                if p.get("miner_hotkey") == hotkey:
                    if env is None or m.get("env") == env:
                        count += 1
                    break
        return count

    mock_dao.save_match = AsyncMock(side_effect=mock_save_match)
    mock_dao.get_matches_for_miner = AsyncMock(side_effect=mock_get_matches_for_miner)
    mock_dao.count_matches_for_miner = AsyncMock(side_effect=mock_count_matches_for_miner)
    mock_dao._matches = matches
    return mock_dao


@pytest.fixture
def sample_match():
    return {
        "match_uuid": "test-match-002",
        "game_type": "tictactoe",
        "task_id": 1,
        "participants": [
            {"slot": 0, "miner_hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", "model_revision": "abc123", "chute_slug": "test-chute-1"},
            {"slot": 1, "miner_hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty", "model_revision": "def456", "chute_slug": "test-chute-2"},
        ],
        "game_config": {"timeout_per_move": 30, "max_moves": 9},
    }
