"""
DynamoDB client management

Provides singleton client instance and connection pooling.
Supports local mode via LOCAL_DB_MODE environment variable.
"""

import os
from typing import Optional

_client = None
_session = None
_local_mode = False


def is_local_mode() -> bool:
    """Check if running in local database mode."""
    return os.getenv("LOCAL_DB_MODE", "").lower() in ("true", "1", "yes")


def get_local_db_path() -> Optional[str]:
    """Get path for local SQLite database (optional persistence)."""
    return os.getenv("LOCAL_DB_PATH")


def get_region() -> str:
    """Get AWS region from environment."""
    return os.getenv("AWS_REGION", "us-east-1")


def get_table_prefix() -> str:
    """Get table name prefix from environment."""
    return os.getenv("DYNAMODB_TABLE_PREFIX", "affine")


async def init_client():
    """Initialize DynamoDB client.

    Creates a singleton client instance with connection pooling.
    In local mode, uses LocalDynamoDBClient instead of AWS.
    """
    global _client, _session, _local_mode

    if _client is not None:
        return _client

    # Check for local mode
    if is_local_mode():
        from affine.database.local_backend import init_local_client
        _local_mode = True
        _client = await init_local_client(sqlite_path=get_local_db_path())
        return _client

    # Standard AWS DynamoDB mode
    import aiobotocore.session
    from botocore.config import Config

    _session = aiobotocore.session.get_session()

    # Create client with connection pooling
    _client = await _session.create_client(
        'dynamodb',
        region_name=get_region(),
        config=Config(
            max_pool_connections=100,
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )
    ).__aenter__()

    return _client


async def close_client():
    """Close DynamoDB client."""
    global _client, _local_mode

    if _client is not None:
        if _local_mode:
            from affine.database.local_backend import close_local_client
            await close_local_client()
        else:
            await _client.__aexit__(None, None, None)
        _client = None
        _local_mode = False


def get_client():
    """Get current DynamoDB client instance.

    Returns:
        DynamoDB client (or LocalDynamoDBClient in local mode)

    Raises:
        RuntimeError: If client not initialized
    """
    if _client is None:
        raise RuntimeError("DynamoDB client not initialized. Call init_client() first.")

    return _client