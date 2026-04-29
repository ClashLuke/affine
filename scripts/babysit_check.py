#!/usr/bin/env python3
"""Validator health/correctness check + debug dashboard.

Reads `.shadow/affine.sqlite3` (or AFFINE_DB_PATH) and the most recent
`.shadow/stderr-*.log`. Detects: process down, SQLite state stale while
validator alive, recent unhandled tracebacks, no champion. Prints a one-screen
dashboard (champion, recent duels, sample counts, backup-provider failures).

Exits 0 clean, 1 if any FAIL.
"""
from __future__ import annotations

import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _prime_shadow_env() -> None:
    dotenv = REPO / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
                os.environ.setdefault(k, v.strip().strip("'\""))
    spec = REPO / ".shadow" / "config.json"
    if spec.exists():
        os.environ.setdefault("AFFINE_CONFIG_SPEC", str(spec))


_prime_shadow_env()

from affine.paired import PairCounts, pair_p_value


PROBE_WINDOW_S = 1800


def _pids() -> list[int]:
    try:
        return [int(p) for p in subprocess.check_output(["pgrep", "-f", ".shadow-venv/bin/affine"]).split()]
    except subprocess.CalledProcessError:
        return []


def _sqlite_db_path() -> Path | None:
    shadow = REPO / ".shadow" / "affine.sqlite3"
    pids = _pids()
    if pids and shadow.exists():
        return shadow
    env = os.environ.get("AFFINE_DB_PATH")
    if env and Path(env).exists():
        return Path(env)
    candidates = [REPO / ".affine" / "affine.sqlite3", shadow]
    existing = [p for p in candidates if p.exists()]
    if existing:
        return max(existing, key=_sqlite_mtime)
    return None


def _sqlite_mtime(path: Path) -> float:
    return max(
        (p.stat().st_mtime for p in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm"))
         if p.exists()),
        default=0.0,
    )


def _counts(db: sqlite3.Connection, duel_id: int) -> PairCounts:
    rows = db.execute(
        """
        SELECT champion_pass, challenger_pass
        FROM samples
        WHERE duel_id=? AND champion_delivered=1 AND challenger_delivered=1
        """,
        (duel_id,),
    ).fetchall()
    counts = PairCounts()
    for r in rows:
        counts = counts.add(int(r["champion_pass"]), int(r["challenger_pass"]))
    return counts


def main() -> int:
    path = _sqlite_db_path()
    pids = _pids()
    problems: list[str] = []
    warns: list[str] = []
    if not pids:
        problems.append("no validator process running")
    elif len(pids) > 1:
        warns.append(f"multiple validator processes: {pids}")

    logs = sorted((REPO / ".shadow").glob("stderr-*.log"))
    log_text = logs[-1].read_text(errors="replace") if logs else ""
    if logs and pids:
        age = time.time() - logs[-1].stat().st_mtime
        if age > PROBE_WINDOW_S:
            warns.append(f"stderr log unmodified {age:.0f}s while validator alive")
    for line in log_text.splitlines()[-200:]:
        if "Traceback (most recent call last):" in line:
            problems.append("unhandled Traceback in recent stderr")
            break

    if path is None:
        problems.append("no SQLite DB found")
    champ = duels = samples = None
    if path is not None:
        db = sqlite3.connect(path)
        db.row_factory = sqlite3.Row
        champ = db.execute("SELECT * FROM champion WHERE id=1").fetchone()
        duels = db.execute("SELECT * FROM duels ORDER BY id DESC").fetchall()
        samples = db.execute("SELECT * FROM samples").fetchall()
        if champ is None:
            problems.append("no SQLite champion")
        if pids:
            latest = max(
                (int(r[0] or 0) for r in (
                    db.execute("SELECT MAX(updated_at) FROM champion").fetchone(),
                    db.execute("SELECT MAX(updated_at) FROM duels").fetchone(),
                    db.execute("SELECT MAX(updated_at) FROM publications").fetchone(),
                    db.execute("SELECT MAX(verified_at) FROM backups").fetchone(),
                )),
                default=0,
            )
            age = time.time() - latest if latest else float("inf")
            if age > PROBE_WINDOW_S:
                problems.append(f"SQLite state stale {age:.0f}s while validator alive")

    status = "OK" if not problems and not warns else ("WARN" if not problems else "FAIL")
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    pid_s = pids[0] if pids else "NONE"
    print(f"[{ts}] pid={pid_s} db={path} | {status}")

    if champ:
        pay = f"uid{champ['uid']}" if champ["payable"] and champ["uid"] is not None else "burned/baseline"
        print(f"  champion: {champ['model']}@{champ['revision'][:12]}  pay={pay}  backup={champ['backup_manifest']}")

    if duels is not None:
        by_status = Counter(str(d["status"]) for d in duels)
        print(f"  duels: {len(duels)} total  statuses={dict(by_status) or '—'}")
        if duels:
            d = duels[0]
            counts = _counts(db, int(d["id"]))
            ties = counts.both_pass + counts.both_fail
            print(f"  latest: id={d['id']} {d['status']} uid{d['challenger_uid']} "
                  f"{d['challenger_model']}@{str(d['challenger_revision'])[:12]} "
                  f"W-L={counts.challenger_only}-{counts.champion_only} ties={ties} "
                  f"p={pair_p_value(counts.challenger_only, counts.discordant):.3g} alpha={float(d['alpha']):.3g}")

    if samples:
        delivered = sum(1 for s in samples if s["champion_delivered"] and s["challenger_delivered"])
        synth = len(samples) - delivered
        env_counts = Counter(str(s["env"]) for s in samples)
        top_envs = " · ".join(f"{e}={n}" for e, n in env_counts.most_common(8))
        print(f"  samples: {len(samples)} rows  delivered_pairs={delivered}  dropped_pairs={synth}  envs: {top_envs}")

    backup_failures = Counter()
    for line in log_text.splitlines():
        m = re.search(r"backup provider (\w+) .*failed", line)
        if m:
            backup_failures[m.group(1)] += 1
    if backup_failures:
        print(f"  backup failures: {dict(backup_failures)}")

    for w in warns:
        print(f"  WARN: {w}")
    for p in problems:
        print(f"  FAIL: {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
