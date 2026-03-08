"""
DynamoDB table schema definitions

Defines table structures with partition keys, sort keys, and indexes.
"""

from typing import Dict, Any, List


def get_table_name(base_name: str) -> str:
    """Get full table name with prefix."""
    from affine.database.client import get_table_prefix
    return f"{get_table_prefix()}_{base_name}"


# Sample Results Table
#
# Design Philosophy:
# - PK combines the 3 most frequent query dimensions: hotkey + revision + env
# - SK uses task_id for natural ordering
# - uid removed (mutable, should query via bittensor metadata -> hotkey first)
# - GSI for efficient timestamp range queries (incremental updates)
# - block_number stored but not indexed (no block query requirement)
#
# Query Patterns:
# 1. Get samples by hotkey+revision+env -> Query by PK
# 2. Get samples by hotkey+revision (all envs) -> Query with PK prefix + filter
# 3. Get samples by hotkey (all revisions) -> Scan with hotkey prefix + filter
# 4. Get samples by timestamp range -> Use timestamp-index GSI (gsi_partition='SAMPLE' AND timestamp > :since)
# 5. Get samples by uid -> Query bittensor metadata first to get hotkey+revision, then query here
#
# GSI Design:
# - gsi_partition: Fixed value "SAMPLE" for all records (partition key)
# - timestamp: Milliseconds since epoch (range key, supports > < BETWEEN)
# - This design enables efficient Query operations for incremental updates
SAMPLE_RESULTS_SCHEMA = {
    "TableName": get_table_name("sample_results"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},   # MINER#{hotkey}#REV#{revision}#ENV#{env}
        {"AttributeName": "sk", "KeyType": "RANGE"},  # TASK#{task_id}
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
        {"AttributeName": "gsi_partition", "AttributeType": "S"},
        {"AttributeName": "timestamp", "AttributeType": "N"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "timestamp-index",
            "KeySchema": [
                {"AttributeName": "gsi_partition", "KeyType": "HASH"},   # Fixed "SAMPLE"
                {"AttributeName": "timestamp", "KeyType": "RANGE"},      # Sortable timestamp
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
    ],
    "BillingMode": "PAY_PER_REQUEST",
}

# TTL settings for sample_results (30 days retention)
SAMPLE_RESULTS_TTL = {
    "AttributeName": "ttl",
}


# Task Pool Table
#
# Design Philosophy:
# - PK: MINER#{hotkey}#REV#{revision} - partition by miner for efficient cleanup
# - SK: ENV#{env}#STATUS#{status}#TASK_ID#{task_id} - composite sort key with business semantics
# - GSI1: env-status-index for weighted random task selection
#
# Query Patterns:
# 1. Weighted random task selection (by TaskPoolManager):
#    - Query GSI1 by ENV#{env}#STATUS#pending
#    - SK sorted by MINER, enabling efficient grouping and counting
#    - Weighted random select miner (probability ∝ task count)
#    - Randomly select one task from chosen miner
# 2. Miner task cleanup (by Scheduler):
#    - Query main table by PK=MINER#{hotkey}#REV#{revision}
#    - Batch delete all tasks for invalid miners (36x faster)
# 3. Check miner pending tasks (by Scheduler):
#    - Query main table by PK with env filter
#    - Direct query, no GSI needed
# 4. Pool statistics:
#    - Query GSI1 by ENV#{env}#STATUS#{status} with Select=COUNT
#
# Design Rationale:
# - No UUID: task_id has business semantics, easier to debug
# - MINER partition: enables O(m) cleanup instead of O(n) individual deletes
# - GSI1 SK by MINER: supports efficient weighted counting
# - Fairness: new miners don't wait for old miners (weighted random, not FIFO)
TASK_POOL_SCHEMA = {
    "TableName": get_table_name("task_pool"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},   # MINER#{hotkey}#REV#{revision}
        {"AttributeName": "sk", "KeyType": "RANGE"},  # ENV#{env}#STATUS#{status}#TASK_ID#{task_id}
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
        {"AttributeName": "gsi1_pk", "AttributeType": "S"},
        {"AttributeName": "gsi1_sk", "AttributeType": "S"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "env-status-index",
            "KeySchema": [
                {"AttributeName": "gsi1_pk", "KeyType": "HASH"},   # ENV#{env}#STATUS#{status}
                {"AttributeName": "gsi1_sk", "KeyType": "RANGE"},  # MINER#{hotkey}#REV#{revision}#TASK_ID#{task_id}
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
    ],
    "BillingMode": "PAY_PER_REQUEST",
}

# Legacy name for compatibility during transition
TASK_QUEUE_SCHEMA = TASK_POOL_SCHEMA


# Execution Logs Table
EXECUTION_LOGS_SCHEMA = {
    "TableName": get_table_name("execution_logs"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},
        {"AttributeName": "sk", "KeyType": "RANGE"},
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
    ],
    "BillingMode": "PAY_PER_REQUEST",
}

# TTL settings (applied after table creation)
EXECUTION_LOGS_TTL = {
    "AttributeName": "ttl",
}


# Scores Table
SCORES_SCHEMA = {
    "TableName": get_table_name("scores"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},
        {"AttributeName": "sk", "KeyType": "RANGE"},
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
        {"AttributeName": "latest_marker", "AttributeType": "S"},
        {"AttributeName": "block_number", "AttributeType": "N"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "latest-block-index",
            "KeySchema": [
                {"AttributeName": "latest_marker", "KeyType": "HASH"},
                {"AttributeName": "block_number", "KeyType": "RANGE"},
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
    ],
    "BillingMode": "PAY_PER_REQUEST",
}

# TTL settings (applied after table creation)
SCORES_TTL = {
    "AttributeName": "ttl",
}


# System Config Table
SYSTEM_CONFIG_SCHEMA = {
    "TableName": get_table_name("system_config"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},
        {"AttributeName": "sk", "KeyType": "RANGE"},
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
    ],
    "BillingMode": "PAY_PER_REQUEST",
}


# Miners Table
# Schema design:
# - PK: UID#{uid} - unique primary key, each UID has only one record
# - No SK needed - single record per UID
# - GSI1: is-valid-index for querying valid/invalid miners
# - GSI2: hotkey-index for querying miner by hotkey
#
# Query patterns:
# 1. Get miner by UID: Direct get by PK
# 2. Get all valid miners: Query GSI1 with is_valid=true
# 3. Get miner by hotkey: Query GSI2 with hotkey
# 4. Get miners by model hash: Scan with filter (for anti-plagiarism)
MINERS_SCHEMA = {
    "TableName": get_table_name("miners"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "is_valid", "AttributeType": "S"},
        {"AttributeName": "hotkey", "AttributeType": "S"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "is-valid-index",
            "KeySchema": [
                {"AttributeName": "is_valid", "KeyType": "HASH"},
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
        {
            "IndexName": "hotkey-index",
            "KeySchema": [
                {"AttributeName": "hotkey", "KeyType": "HASH"},
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
    ],
    "BillingMode": "PAY_PER_REQUEST",
}



# Score Snapshots Table
# Stores metadata for each scoring calculation
SCORE_SNAPSHOTS_SCHEMA = {
    "TableName": get_table_name("score_snapshots"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},   # BLOCK#{block_number}
        {"AttributeName": "sk", "KeyType": "RANGE"},  # TIME#{timestamp}
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
        {"AttributeName": "latest_marker", "AttributeType": "S"},
        {"AttributeName": "timestamp", "AttributeType": "N"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "latest-index",
            "KeySchema": [
                {"AttributeName": "latest_marker", "KeyType": "HASH"},
                {"AttributeName": "timestamp", "KeyType": "RANGE"},
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
    ],
    "BillingMode": "PAY_PER_REQUEST",
}

# TTL settings for score_snapshots
SCORE_SNAPSHOTS_TTL = {
    "AttributeName": "ttl",
}


# Miner Stats Table
# Schema design:
# - PK: HOTKEY#{hotkey} - partition by hotkey
# - SK: REV#{revision} - each revision is a separate record
#
# Query patterns:
# 1. Get miner stats: Direct query by hotkey + revision
# 2. Get all revisions for a hotkey: Query by PK prefix
# 3. Get all historical miners: Full table scan
# 4. Cleanup inactive miners: Full table scan with filter
#
# Design rationale:
# - Permanent storage of all miner metadata (not just current 256)
# - Real-time sampling statistics via sliding windows
# - No GSI needed (cleanup uses full scan, which is efficient for small tables)
MINER_STATS_SCHEMA = {
    "TableName": get_table_name("miner_stats"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},   # HOTKEY#{hotkey}
        {"AttributeName": "sk", "KeyType": "RANGE"},  # REV#{revision}
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
    ],
    "BillingMode": "PAY_PER_REQUEST",
}


# =============================================================================
# ELO Rating System Tables
# =============================================================================

# ELO Ratings Table
#
# Design Philosophy:
# - PK: MINER#{hotkey}#REV#{revision} - partition by miner (same as other tables)
# - SK: ENV#{env} - one rating record per miner per environment
# - GSI: env-rating-index for leaderboard queries (get top miners by ELO in env)
#
# Query Patterns:
# 1. Get miner's rating in an env: Direct query by PK + SK
# 2. Get all ratings for a miner: Query by PK (all envs)
# 3. Get leaderboard for an env: Query GSI by env, sorted by rating DESC
# 4. Get all ratings: Full table scan (for bulk operations)
#
# Fields:
# - rating: Current ELO rating (Decimal, default 1500)
# - peak_rating: Historical peak rating
# - matches_played: Total matches played in this env
# - wins, losses, draws: Match outcome counts
# - last_match_at: Timestamp of last match
ELO_RATINGS_SCHEMA = {
    "TableName": get_table_name("elo_ratings"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},   # MINER#{hotkey}#REV#{revision}
        {"AttributeName": "sk", "KeyType": "RANGE"},  # ENV#{env}
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
        {"AttributeName": "env", "AttributeType": "S"},
        {"AttributeName": "rating", "AttributeType": "N"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "env-rating-index",
            "KeySchema": [
                {"AttributeName": "env", "KeyType": "HASH"},
                {"AttributeName": "rating", "KeyType": "RANGE"},
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
    ],
    "BillingMode": "PAY_PER_REQUEST",
}


# Match Records Table
#
# Design Philosophy:
# - PK: ENV#{env} - partition by environment for efficient env-scoped queries
# - SK: MATCH#{timestamp}#{match_uuid} - time-ordered matches within env
# - GSI: timestamp-index for cross-env time-range queries
#
# Query Patterns:
# 1. Get matches in an env: Query by PK
# 2. Get matches in time range (all envs): Query GSI by gsi_partition + timestamp
# 3. Get specific match: Query by PK + SK prefix with match_uuid
#
# Fields:
# - match_type: "pairwise" (from score comparison) or "game" (direct competition)
# - participants: List of {miner_hotkey, model_revision, outcome, elo_before, elo_after}
# - task_id: Original task_id that generated this match
# - game_result: Game-specific result data (for multi-party games)
MATCH_RECORDS_SCHEMA = {
    "TableName": get_table_name("match_records"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},   # ENV#{env}
        {"AttributeName": "sk", "KeyType": "RANGE"},  # MATCH#{timestamp}#{match_uuid}
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
        {"AttributeName": "gsi_partition", "AttributeType": "S"},
        {"AttributeName": "timestamp", "AttributeType": "N"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "timestamp-index",
            "KeySchema": [
                {"AttributeName": "gsi_partition", "KeyType": "HASH"},  # Fixed "MATCH"
                {"AttributeName": "timestamp", "KeyType": "RANGE"},
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
    ],
    "BillingMode": "PAY_PER_REQUEST",
}

# TTL settings for match_records (90 days retention)
MATCH_RECORDS_TTL = {
    "AttributeName": "ttl",
}


# Match Pool Table (for multi-party games)
#
# Design Philosophy:
# - PK: GAME#{game_type}#STATUS#{status} - partition by game type and status
# - SK: MATCH#{match_uuid} - unique match identifier
# - GSI: status-created-index for FIFO-style fetching of pending matches
#
# Query Patterns:
# 1. Get pending matches for a game: Query by PK (game_type + status=pending)
# 2. Get oldest pending matches (FIFO): Query GSI by status, sorted by created_at
# 3. Get specific match: Query any partition + SK
#
# Fields:
# - game_type: "tictactoe", "chess", etc.
# - player_count: Required number of players
# - participants: List of matched players (filled as they join)
# - task_id: Game seed/scenario identifier
# - game_config: Game-specific configuration
# - status: pending | executing | completed | failed
MATCH_POOL_SCHEMA = {
    "TableName": get_table_name("match_pool"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},   # GAME#{game_type}#STATUS#{status}
        {"AttributeName": "sk", "KeyType": "RANGE"},  # MATCH#{match_uuid}
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
        {"AttributeName": "gsi1_pk", "AttributeType": "S"},
        {"AttributeName": "gsi1_sk", "AttributeType": "N"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "status-created-index",
            "KeySchema": [
                {"AttributeName": "gsi1_pk", "KeyType": "HASH"},   # STATUS#{status}
                {"AttributeName": "gsi1_sk", "KeyType": "RANGE"},  # created_at (timestamp)
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
    ],
    "BillingMode": "PAY_PER_REQUEST",
}


# Match Results Table (per-participant results for multi-party games)
#
# Design Philosophy:
# - PK: MINER#{hotkey}#REV#{revision}#GAME#{game_type} - partition by miner+game
# - SK: MATCH#{match_uuid} - one record per match per participant
# - GSI: match-uuid-index for fetching all participants in a match
#
# Query Patterns:
# 1. Get miner's match history in a game: Query by PK
# 2. Get all participants in a match: Query GSI by match_uuid
# 3. Get miner's overall stats: Aggregate from query by PK
#
# Fields:
# - outcome: "win", "loss", "draw", "timeout", "error"
# - score: Normalized score (0.0-1.0)
# - opponent_hotkeys: List of opponent hotkeys
# - move_history: Compressed move sequence
# - latency stats: avg_move_latency_ms, total_latency_ms
MATCH_RESULTS_SCHEMA = {
    "TableName": get_table_name("match_results"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},   # MINER#{hotkey}#REV#{revision}#GAME#{game_type}
        {"AttributeName": "sk", "KeyType": "RANGE"},  # MATCH#{match_uuid}
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
        {"AttributeName": "gsi1_pk", "AttributeType": "S"},
        {"AttributeName": "gsi1_sk", "AttributeType": "S"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "match-uuid-index",
            "KeySchema": [
                {"AttributeName": "gsi1_pk", "KeyType": "HASH"},   # MATCH#{match_uuid}
                {"AttributeName": "gsi1_sk", "KeyType": "RANGE"},  # SLOT#{slot}
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
    ],
    "BillingMode": "PAY_PER_REQUEST",
}

# TTL settings for match_results (30 days retention)
MATCH_RESULTS_TTL = {
    "AttributeName": "ttl",
}


# All table schemas
ALL_SCHEMAS = [
    SAMPLE_RESULTS_SCHEMA,
    TASK_POOL_SCHEMA,
    EXECUTION_LOGS_SCHEMA,
    SCORES_SCHEMA,
    SYSTEM_CONFIG_SCHEMA,
    MINERS_SCHEMA,
    SCORE_SNAPSHOTS_SCHEMA,
    MINER_STATS_SCHEMA,
    # ELO system tables
    ELO_RATINGS_SCHEMA,
    MATCH_RECORDS_SCHEMA,
    MATCH_POOL_SCHEMA,
    MATCH_RESULTS_SCHEMA,
]