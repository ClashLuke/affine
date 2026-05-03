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
from affine.config import parse_model_skiplist


PROBE_WINDOW_S = 1800
VERDICT_QUEUE_LIMIT = 12

VERDICT_COLS = (
    ("UID",     4,  ">"),
    ("Repo",    28, "<"),
    ("Rev",     8,  "<"),
    ("Hotkey",  6,  "<"),
    ("Verdict", 18, "<"),
    ("W-L",     7,  ">"),
    ("Ties",    5,  ">"),
    ("p",       7,  ">"),
    ("α",  7,  ">"),
    ("Age",     5,  ">"),
)


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _age(now: float, ts: float) -> str:
    d = max(int(now - ts), 0)
    if d < 60:
        return f"{d}s"
    if d < 3600:
        return f"{d // 60}m"
    if d < 86400:
        return f"{d // 3600}h"
    return f"{d // 86400}d"


def _fmt_row(cells: list[str]) -> str:
    return "  " + " ".join(f"{c:{a}{w}}" for c, (_, w, a) in zip(cells, VERDICT_COLS))


def _verdict_row(duel, counts: PairCounts, now: float) -> list[str]:
    p = pair_p_value(counts.challenger_only, counts.discordant)
    return [
        str(duel["challenger_uid"]),
        _trunc(str(duel["challenger_model"]), 28),
        str(duel["challenger_revision"])[:8],
        (str(duel["challenger_hotkey"]) or "——")[:6],
        _trunc(str(duel["status"]), 18),
        f"{counts.challenger_only}-{counts.champion_only}",
        str(counts.both_pass + counts.both_fail),
        "——" if counts.discordant == 0 else f"{p:.2g}",
        f"{float(duel['alpha']):.2g}",
        _age(now, float(duel["created_at"])),
    ]


def _pids() -> list[int]:
    try:
        return [int(p) for p in subprocess.check_output(["pgrep", "-f", ".shadow-venv/bin/affine"]).split()]
    except subprocess.CalledProcessError:
        return []


def _sqlite_db_path() -> Path | None:
    shadow = REPO / ".shadow" / "affine.sqlite3"
    env = os.environ.get("AFFINE_DB_PATH")
    if env and Path(env).exists():
        return Path(env)
    return shadow if shadow.exists() else None


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
    model_skiplist = parse_model_skiplist(os.environ.get("AFFINE_MODEL_SKIPLIST", ""))
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
        elif str(champ["model"]) in model_skiplist:
            problems.append(f"skiplisted champion active: {champ['model']}")

    status = "OK" if not problems and not warns else ("WARN" if not problems else "FAIL")
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    pid_s = pids[0] if pids else "NONE"
    print(f"[{ts}] pid={pid_s} db={path} | {status}")

    if champ:
        pay = f"uid{champ['uid']}" if champ["payable"] and champ["uid"] is not None else "burned/baseline"
        with sqlite3.connect(path) as db:
            db.row_factory = sqlite3.Row
            backup = db.execute(
                "SELECT manifest_key FROM backups WHERE artifact_id=? AND status='current'",
                (champ["artifact_id"],),
            ).fetchone()
        manifest = backup["manifest_key"] if backup else ""
        print(f"  champion: {champ['model']}@{champ['revision'][:12]}  pay={pay}  backup={manifest}")

    if duels is not None:
        by_status = Counter(str(d["status"]) for d in duels)
        print(f"  duels: {len(duels)} total  statuses={dict(by_status) or '—'}")
        if duels:
            now = time.time()
            header = [name for name, _, _ in VERDICT_COLS]
            print(_fmt_row(header))
            print(_fmt_row(["-" * w for _, w, _ in VERDICT_COLS]))
            for d in duels[:VERDICT_QUEUE_LIMIT]:
                print(_fmt_row(_verdict_row(d, _counts(db, int(d["id"])), now)))

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
