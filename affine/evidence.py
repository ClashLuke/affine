"""Per-sample evidence: append-only JSONL, one row per inference outcome.

Single-validator. Identity (m, r, e, c) uniquely selects a miner's sample slot;
`i` is the env-specific task id seen on that sample (shared between king and
challenger within a single dwell iteration so verdicts are matched-task).
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)


def _reject_constant(c: str):
    raise ValueError(f"non-finite JSON constant: {c}")


def _rollback(fd: int, pre: int) -> None:
    """Truncate back to pre-write size and durably commit the truncation. Without
    the trailing fsync, a power loss after rollback can journal-replay the partial
    bytes the rollback was meant to discard."""
    os.ftruncate(fd, pre)
    try:
        os.fsync(fd)
    except OSError:
        pass


@dataclass(frozen=True)
class Row:
    m: int    # miner uid
    r: str    # miner revision
    e: str    # env name
    c: int    # per-(m,r,e) counter (drives LLM seed)
    p: int    # 1 pass, 0 fail
    t: int    # block number
    l: float  # latency seconds
    i: int = 0  # env task id (env-specific)

    def __post_init__(self):
        if self.p not in (0, 1):
            raise ValueError(f"p must be 0 or 1, got {self.p}")


class EvidenceStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._heal_torn_tail()
        self._counters: dict[tuple[int, str, str], int] = {}
        if self.path.exists():
            for row in self.read():
                k = (row.m, row.r, row.e)
                self._counters[k] = max(self._counters.get(k, 0), row.c + 1)

    def _heal_torn_tail(self) -> None:
        """If a prior process died mid-append, the file ends without a `\\n`. The
        next append would glue its row onto the unterminated tail — corrupting both
        the orphan and the new row. Terminating the tail bounds the loss to one
        partial row (which read() already discards as malformed)."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        with self.path.open("rb+") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                f.write(b"\n")
                log.warning(f"evidence: torn tail healed in {self.path}")

    def next_counter(self, m: int, r: str, e: str) -> int:
        """The c to use for the next sample. Idempotent — does not advance until a
        Row carrying that c is appended. A sample whose row is never written (infra
        failure) must not consume a counter slot, otherwise after restart the disk
        rebuild regresses the counter and the same SHA-derived seed gets reused."""
        return self._counters.get((m, r, e), 0)

    def append(self, row: Row) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(row)) + "\n")
        k = (row.m, row.r, row.e)
        self._counters[k] = max(self._counters.get(k, 0), row.c + 1)

    def append_pair(self, row_a: Row, row_b: Row) -> None:
        """Single-syscall + fsync dual append for matched-task duels. Three failure
        modes the asymmetry would create:
          - Buffered text I/O can split a 'one-call' write into multiple
            syscalls. Use os.write directly to keep it one.
          - The kernel may return success on write() while only some bytes have
            reached disk. fsync forces durability *before* we update in-memory
            counters; without it, a power loss after the first row hits disk
            but the second doesn't would leave the king's c persisted while
            the challenger's c is replayed from a fresh-looking counter on
            restart — same SHA seed, deterministic-replay invariant violated.
          - ENOSPC mid-write can leave row A complete on disk and row B partial.
            On restart that's heal_torn_tail-completed, row A loads, row B is
            dropped — and the challenger's counter regresses while the king's
            advances. Truncate back to pre-write size on any short/failed write
            so the pair stays atomic at the file level.
        """
        payload = (json.dumps(asdict(row_a)) + "\n"
                   + json.dumps(asdict(row_b)) + "\n").encode()
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            pre = os.fstat(fd).st_size
            try:
                n = os.write(fd, payload)
            except OSError:
                _rollback(fd, pre); raise
            if n != len(payload):
                _rollback(fd, pre)
                raise OSError(f"short write: {n}/{len(payload)} bytes; truncated to {pre}")
            try:
                os.fsync(fd)
            except OSError:
                # fsync failure (ENOSPC late, EIO, hardware fault) means durability
                # is unconfirmed. Roll back so counters and disk stay consistent;
                # the rollback itself must be durable, otherwise journal replay
                # after power loss can resurrect the partial pair we just truncated.
                _rollback(fd, pre); raise
        finally:
            os.close(fd)
        for row in (row_a, row_b):
            k = (row.m, row.r, row.e)
            self._counters[k] = max(self._counters.get(k, 0), row.c + 1)

    def read(self) -> list[Row]:
        if not self.path.exists():
            return []
        rows: list[Row] = []
        # Binary mode + per-line decode: a SIGKILL mid-write of a multibyte char
        # (e.g. a non-ASCII model name) would otherwise raise UnicodeDecodeError
        # from the text-mode iterator, OUTSIDE the try below — read() would
        # propagate it and crash startup.
        with self.path.open("rb") as f:
            for raw in f:
                try:
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    # parse_constant rejects NaN/Infinity. json.loads accepts both
                    # by default and they'd silently flow into the IRT fit / counters.
                    d = json.loads(line, parse_constant=_reject_constant)
                    if not (isinstance(d.get("r"), str) and isinstance(d.get("e"), str)):
                        raise TypeError(f"r/e must be strings, got r={type(d.get('r')).__name__} e={type(d.get('e')).__name__}")
                    lat = float(d["l"])
                    if not math.isfinite(lat):
                        raise ValueError(f"non-finite latency: {lat}")
                    rows.append(Row(
                        m=int(d["m"]), r=d["r"], e=d["e"], c=int(d["c"]),
                        p=int(d["p"]), t=int(d["t"]), l=lat,
                        i=int(d.get("i", 0)),
                    ))
                except (json.JSONDecodeError, KeyError, ValueError, TypeError, UnicodeDecodeError) as ex:
                    log.warning(f"skipping malformed row: {ex}")
        return rows
