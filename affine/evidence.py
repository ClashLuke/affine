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

    def append(self, *rows: Row) -> None:
        """Atomic append for one or more rows. Both/all rows land in one write+fsync,
        or none do. Three failure modes the file-level atomicity defends against:
          - Buffered text I/O can split a single write into multiple syscalls.
            os.write keeps it atomic.
          - Kernel may return success on write() before bytes reach disk. fsync
            forces durability *before* in-memory counters advance — without it,
            a power loss after partial flush leaves disk inconsistent with the
            counter rebuild on restart, replaying the same SHA seed.
          - ENOSPC mid-write of a duel pair (king + challenger) can leave row A
            complete and row B partial. heal_torn_tail would then load A and
            drop B, regressing the challenger's counter while king's advanced.
            ftruncate-on-failure keeps the batch atomic at the file level.
        """
        if not rows:
            return
        payload = "".join(json.dumps(asdict(r)) + "\n" for r in rows).encode()
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            pre = os.fstat(fd).st_size
            try:
                n = os.write(fd, payload)
            except OSError:
                os.ftruncate(fd, pre); raise
            if n != len(payload):
                os.ftruncate(fd, pre)
                raise OSError(f"short write: {n}/{len(payload)} bytes; truncated to {pre}")
            try:
                os.fsync(fd)
            except OSError:
                os.ftruncate(fd, pre); raise
        finally:
            os.close(fd)
        for row in rows:
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
