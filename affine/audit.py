"""Shadow audit log — one JSON line per duel verdict and weight intent.

Active when `AFFINE_SHADOW_LOG=/path/to/file.jsonl` is set. Otherwise noop.
"""

from __future__ import annotations
import json
import logging
import math
import os
from datetime import datetime, timezone

log = logging.getLogger(__name__)


def _sanitize(v):
    """Replace non-finite floats with None so allow_nan=False doesn't drop the
    whole record on a single bad numeric (e.g. z=delta/0=inf from a degenerate
    Laplace cov). A duel verdict with a missing field is still useful evidence
    — an empty audit log is not."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _sanitize(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_sanitize(x) for x in v]
    return v


def audit(**fields) -> None:
    path = os.getenv("AFFINE_SHADOW_LOG")
    if not path:
        return
    record = {"ts": datetime.now(timezone.utc).isoformat(), **fields}
    try:
        line = json.dumps(_sanitize(record), allow_nan=False, default=str)
        with open(path, "a") as f:
            f.write(line + "\n")
    except (OSError, TypeError, ValueError) as e:
        log.warning(f"audit log write failed: {e}")
