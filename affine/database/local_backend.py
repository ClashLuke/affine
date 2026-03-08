"""
Local In-Memory Database Backend

Replaces AWS DynamoDB with a local in-memory store for running the validator
locally without AWS infrastructure. Supports optional SQLite persistence.

Usage:
    Set environment variable: LOCAL_DB_MODE=true
    Optionally set: LOCAL_DB_PATH=/path/to/db.sqlite for persistence
"""

import json
import sqlite3
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional
from decimal import Decimal


class LocalDynamoDBClient:
    """
    In-memory DynamoDB-compatible client for local testing.

    Implements the core DynamoDB operations used by BaseDAO:
    - put_item
    - get_item
    - query
    - delete_item
    - batch_write_item

    Optionally persists to SQLite for data durability across restarts.
    """

    def __init__(self, sqlite_path: Optional[str] = None):
        """
        Initialize the local client.

        Args:
            sqlite_path: Optional path to SQLite database for persistence.
                        If None, uses in-memory storage only.
        """
        self.sqlite_path = sqlite_path
        self._lock = threading.RLock()

        # In-memory storage: {table_name: {composite_key: item}}
        self._tables: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

        # GSI indexes: {table_name: {index_name: {pk_value: [items]}}}
        self._gsi: Dict[str, Dict[str, Dict[str, List[Dict]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # Load from SQLite if path provided
        if sqlite_path:
            self._init_sqlite()
            self._load_from_sqlite()

    def _init_sqlite(self):
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS items (
                table_name TEXT NOT NULL,
                pk TEXT NOT NULL,
                sk TEXT,
                data TEXT NOT NULL,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                PRIMARY KEY (table_name, pk, sk)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_table_pk ON items(table_name, pk)
        """)

        conn.commit()
        conn.close()

    def _load_from_sqlite(self):
        """Load existing data from SQLite."""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()

        cursor.execute("SELECT table_name, pk, sk, data FROM items")
        for row in cursor.fetchall():
            table_name, pk, sk, data = row
            item = json.loads(data)
            key = f"{pk}#{sk}" if sk else pk
            self._tables[table_name][key] = item

        conn.close()

    def _persist_item(self, table_name: str, pk: str, sk: Optional[str], item: Dict):
        """Persist item to SQLite if enabled."""
        if not self.sqlite_path:
            return

        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO items (table_name, pk, sk, data)
            VALUES (?, ?, ?, ?)
        """, (table_name, pk, sk, json.dumps(item, default=str)))

        conn.commit()
        conn.close()

    def _delete_item_sqlite(self, table_name: str, pk: str, sk: Optional[str]):
        """Delete item from SQLite if enabled."""
        if not self.sqlite_path:
            return

        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()

        if sk:
            cursor.execute("DELETE FROM items WHERE table_name = ? AND pk = ? AND sk = ?",
                          (table_name, pk, sk))
        else:
            cursor.execute("DELETE FROM items WHERE table_name = ? AND pk = ?",
                          (table_name, pk))

        conn.commit()
        conn.close()

    @staticmethod
    def _extract_key_value(key_dict: Dict) -> str:
        """Extract string value from DynamoDB key format."""
        if 'S' in key_dict:
            return key_dict['S']
        elif 'N' in key_dict:
            return key_dict['N']
        return str(list(key_dict.values())[0])

    async def put_item(self, TableName: str, Item: Dict[str, Any], **kwargs) -> Dict:
        """Put an item into the table."""
        with self._lock:
            pk = self._extract_key_value(Item.get('pk', {}))
            sk = self._extract_key_value(Item.get('sk', {})) if 'sk' in Item else None

            key = f"{pk}#{sk}" if sk else pk
            self._tables[TableName][key] = Item

            # Persist to SQLite
            self._persist_item(TableName, pk, sk, Item)

            return {}

    async def get_item(self, TableName: str, Key: Dict[str, Any], **kwargs) -> Dict:
        """Get an item by key."""
        with self._lock:
            pk = self._extract_key_value(Key.get('pk', {}))
            sk = self._extract_key_value(Key.get('sk', {})) if 'sk' in Key else None

            key = f"{pk}#{sk}" if sk else pk

            item = self._tables[TableName].get(key)
            if item:
                return {"Item": item}
            return {}

    async def query(self, TableName: str, **kwargs) -> Dict:
        """Query items by partition key and optional sort key conditions.

        Handles GSI queries by detecting the GSI pk attribute from the
        KeyConditionExpression, and applies FilterExpression for basic
        equality/contains filters.
        """
        with self._lock:
            key_condition = kwargs.get('KeyConditionExpression', '')
            expression_values = kwargs.get('ExpressionAttributeValues', {})
            expression_names = kwargs.get('ExpressionAttributeNames', {})
            filter_expression = kwargs.get('FilterExpression', '')
            limit = kwargs.get('Limit')
            reverse = not kwargs.get('ScanIndexForward', True)
            index_name = kwargs.get('IndexName')
            select = kwargs.get('Select')

            # Parse all expression value placeholders
            resolved_values = {}
            for placeholder, val in expression_values.items():
                resolved_values[placeholder] = self._extract_key_value(val)

            # Detect GSI pk attribute from KeyConditionExpression
            pk_value = None
            sk_range = None

            import re
            sk_between_attr = None
            sk_between_start = None
            sk_between_end = None

            if index_name:
                parts = [p.strip() for p in key_condition.split(' AND ')]
                pk_match = re.match(r'(#?\w+)\s*=\s*(:[\w]+)', parts[0])
                if pk_match:
                    gsi_pk_attr = pk_match.group(1)
                    if gsi_pk_attr in expression_names:
                        gsi_pk_attr = expression_names[gsi_pk_attr]
                    pk_value = resolved_values.get(pk_match.group(2))
                else:
                    gsi_pk_attr = 'gsi_partition'
                    pk_value = resolved_values.get(':pk') or resolved_values.get(':status') or resolved_values.get(':gsi')

                # Parse BETWEEN on sort key: "#ts BETWEEN :start AND :end"
                if len(parts) > 1:
                    rest = ' AND '.join(parts[1:])
                    between_match = re.match(
                        r'(#?\w+)\s+BETWEEN\s+(:[\w]+)\s+AND\s+(:[\w]+)', rest, re.IGNORECASE
                    )
                    if between_match:
                        sk_between_attr = between_match.group(1)
                        if sk_between_attr in expression_names:
                            sk_between_attr = expression_names[sk_between_attr]
                        sk_between_start = resolved_values.get(between_match.group(2))
                        sk_between_end = resolved_values.get(between_match.group(3))
            else:
                pk_value = resolved_values.get(':pk')
                gsi_pk_attr = None

            sk_prefix = resolved_values.get(':sk')

            items = []

            if index_name:
                for key, item in self._tables[TableName].items():
                    if gsi_pk_attr and gsi_pk_attr in item:
                        item_gsi_pk = self._extract_key_value(item[gsi_pk_attr])
                    else:
                        item_gsi_pk = None
                    if item_gsi_pk != pk_value:
                        continue
                    if sk_between_attr and sk_between_attr in item:
                        val = self._extract_key_value(item[sk_between_attr])
                        try:
                            val_n = float(val) if val is not None else None
                            start_n = float(sk_between_start) if sk_between_start is not None else None
                            end_n = float(sk_between_end) if sk_between_end is not None else None
                            if val_n is not None and start_n is not None and end_n is not None:
                                if not (start_n <= val_n <= end_n):
                                    continue
                        except (TypeError, ValueError):
                            if not (sk_between_start <= str(val) <= sk_between_end):
                                continue
                    items.append(item)
            else:
                for key, item in self._tables[TableName].items():
                    item_pk = self._extract_key_value(item.get('pk', {}))
                    item_sk = self._extract_key_value(item.get('sk', {})) if 'sk' in item else ''

                    if item_pk != pk_value:
                        continue
                    if sk_prefix and not item_sk.startswith(sk_prefix):
                        continue
                    items.append(item)

            # Apply FilterExpression (basic equality and contains)
            if filter_expression:
                filtered = []
                for item in items:
                    if self._matches_filter(item, filter_expression, resolved_values, expression_names):
                        filtered.append(item)
                items = filtered

            # Sort by sk
            items.sort(
                key=lambda x: self._extract_key_value(x.get('sk', {'S': ''})),
                reverse=reverse
            )

            if limit:
                items = items[:limit]

            if select == 'COUNT':
                return {"Count": len(items)}

            return {"Items": items, "Count": len(items)}

    def _matches_filter(self, item: Dict, expression: str, values: Dict[str, str], names: Dict[str, str]) -> bool:
        """Evaluate a simple FilterExpression against an item."""
        import re
        # Resolve attribute name aliases
        resolved_expr = expression
        for alias, real_name in names.items():
            resolved_expr = resolved_expr.replace(alias, real_name)

        # Handle "attr = :val"
        eq_match = re.match(r'(\w+)\s*=\s*(:[\w]+)', resolved_expr.strip())
        if eq_match:
            attr = eq_match.group(1)
            val = values.get(eq_match.group(2))
            if attr in item:
                return self._extract_key_value(item[attr]) == val
            return False

        # Handle "contains(attr, :val)"
        contains_match = re.match(r'contains\((\w+),\s*(:[\w]+)\)', resolved_expr.strip())
        if contains_match:
            attr = contains_match.group(1)
            val = values.get(contains_match.group(2))
            if attr in item:
                item_val = item[attr]
                # If it's a DynamoDB list, search string representations
                if isinstance(item_val, dict) and 'L' in item_val:
                    return any(val in json.dumps(elem, default=str) for elem in item_val['L'])
                if isinstance(item_val, dict) and 'S' in item_val:
                    return val in item_val['S']
            return False

        # Fallback: pass everything
        return True

    async def delete_item(self, TableName: str, Key: Dict[str, Any], **kwargs) -> Dict:
        """Delete an item by key, respecting ConditionExpression."""
        with self._lock:
            pk = self._extract_key_value(Key.get('pk', {}))
            sk = self._extract_key_value(Key.get('sk', {})) if 'sk' in Key else None

            key = f"{pk}#{sk}" if sk else pk

            condition = kwargs.get('ConditionExpression', '')
            if condition and 'attribute_exists' in condition:
                if key not in self._tables[TableName]:
                    raise Exception("ConditionalCheckFailedException: condition not met")

            self._tables[TableName].pop(key, None)

            # Delete from SQLite
            self._delete_item_sqlite(TableName, pk, sk)

            return {}

    async def batch_write_item(self, RequestItems: Dict[str, List[Dict]], **kwargs) -> Dict:
        """Batch write items."""
        for table_name, requests in RequestItems.items():
            for request in requests:
                if 'PutRequest' in request:
                    item = request['PutRequest']['Item']
                    await self.put_item(table_name, item)
                elif 'DeleteRequest' in request:
                    key = request['DeleteRequest']['Key']
                    await self.delete_item(table_name, key)

        return {"UnprocessedItems": {}}

    async def scan(self, TableName: str, **kwargs) -> Dict:
        """Scan all items in a table."""
        with self._lock:
            items = list(self._tables[TableName].values())
            limit = kwargs.get('Limit')

            if limit:
                items = items[:limit]

            return {"Items": items, "Count": len(items)}

    def get_table_stats(self) -> Dict[str, int]:
        """Get item counts per table."""
        with self._lock:
            return {table: len(items) for table, items in self._tables.items()}

    def clear_all(self):
        """Clear all data (useful for testing)."""
        with self._lock:
            self._tables.clear()

            if self.sqlite_path:
                conn = sqlite3.connect(self.sqlite_path)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM items")
                conn.commit()
                conn.close()

    # Weight History Methods

    async def store_weight_history(
        self,
        cycle: int,
        timestamp: int,
        uids: List[int],
        weights: List[float],
        burn_percentage: float,
        mode: str,
        block_number: Optional[int] = None,
    ):
        """
        Store a weight setting record in history.

        Args:
            cycle: Cycle number
            timestamp: Unix timestamp in milliseconds
            uids: List of UIDs
            weights: List of weights (same order as UIDs)
            burn_percentage: Burn percentage applied
            mode: Weight setting mode (mock, testnet, dry-run)
            block_number: Chain block number (for testnet mode)
        """
        await self.put_item(
            TableName="WeightHistory",
            Item={
                "pk": {"S": "WEIGHTS"},
                "sk": {"S": f"{timestamp}#{cycle}"},
                "cycle": {"N": str(cycle)},
                "timestamp": {"N": str(timestamp)},
                "uids": {"S": json.dumps(uids)},
                "weights": {"S": json.dumps(weights)},
                "burn_percentage": {"N": str(burn_percentage)},
                "mode": {"S": mode},
                "block_number": {"N": str(block_number or 0)},
            }
        )

    async def get_weight_history(
        self,
        limit: int = 100,
        reverse: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get weight setting history.

        Args:
            limit: Maximum entries to return
            reverse: If True, return newest first

        Returns:
            List of weight history entries
        """
        result = await self.query(
            TableName="WeightHistory",
            KeyConditionExpression="pk = :pk",
            ExpressionAttributeValues={":pk": {"S": "WEIGHTS"}},
            ScanIndexForward=not reverse,
            Limit=limit,
        )

        history = []
        for item in result.get("Items", []):
            history.append({
                "cycle": int(item.get("cycle", {}).get("N", 0)),
                "timestamp": int(item.get("timestamp", {}).get("N", 0)),
                "uids": json.loads(item.get("uids", {}).get("S", "[]")),
                "weights": json.loads(item.get("weights", {}).get("S", "[]")),
                "burn_percentage": float(item.get("burn_percentage", {}).get("N", 0)),
                "mode": item.get("mode", {}).get("S", "unknown"),
                "block_number": int(item.get("block_number", {}).get("N", 0)) or None,
            })

        return history

    async def get_latest_weights(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent weight setting.

        Returns:
            Latest weight history entry or None
        """
        history = await self.get_weight_history(limit=1, reverse=True)
        return history[0] if history else None

    async def clear_weight_history(self):
        """Clear all weight history."""
        result = await self.query(
            TableName="WeightHistory",
            KeyConditionExpression="pk = :pk",
            ExpressionAttributeValues={":pk": {"S": "WEIGHTS"}},
        )

        for item in result.get("Items", []):
            await self.delete_item(
                TableName="WeightHistory",
                Key={
                    "pk": item["pk"],
                    "sk": item["sk"],
                }
            )


# Global local client instance
_local_client: Optional[LocalDynamoDBClient] = None


def get_local_client() -> LocalDynamoDBClient:
    """Get the local client singleton."""
    global _local_client
    if _local_client is None:
        raise RuntimeError("Local client not initialized. Call init_local_client() first.")
    return _local_client


async def init_local_client(sqlite_path: Optional[str] = None) -> LocalDynamoDBClient:
    """Initialize the local DynamoDB client.

    Args:
        sqlite_path: Optional path to SQLite file for persistence.
                    If None, data is stored in-memory only.

    Returns:
        LocalDynamoDBClient instance
    """
    global _local_client

    if _local_client is not None:
        return _local_client

    _local_client = LocalDynamoDBClient(sqlite_path=sqlite_path)
    return _local_client


async def close_local_client():
    """Close the local client."""
    global _local_client
    _local_client = None
