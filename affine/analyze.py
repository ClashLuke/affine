"""Offline replay of the validator SQLite DB. Pure read; safe while the loop runs."""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from .config import Config
from .paired import PairCounts, decide_paired, pair_p_value


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline replay of validator SQLite evidence.")
    ap.add_argument("path", nargs="?", default=Config.from_env().db_path)
    args = ap.parse_args()
    path = Path(args.path)
    if not path.exists():
        print(f"no evidence at {path}")
        return 1

    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    champ = db.execute("SELECT * FROM champion WHERE id=1").fetchone()
    if champ:
        pay = f"uid{champ['uid']}" if champ["uid"] is not None else "unregistered/burn"
        print(f"# champion {champ['model']}@{champ['revision']} {pay} backup={champ['backup_manifest']}")
    duels = db.execute("SELECT * FROM duels ORDER BY id DESC LIMIT 20").fetchall()
    if not duels:
        print("no duels")
        return 0
    print("id  status      challenger                      W-L  ties  p        alpha")
    for d in duels:
        rows = db.execute(
            """
            SELECT champion_pass, challenger_pass
            FROM samples
            WHERE duel_id=? AND champion_delivered=1 AND challenger_delivered=1
            """,
            (d["id"],),
        ).fetchall()
        counts = PairCounts()
        for r in rows:
            counts = counts.add(int(r["champion_pass"]), int(r["challenger_pass"]))
        decision = decide_paired(counts, alpha=float(d["alpha"]), min_discordant=int(d["min_discordant"]))
        ties = counts.both_pass + counts.both_fail
        challenger = f"uid{d['challenger_uid']} {d['challenger_model']}@{d['challenger_revision']}"
        print(f"{d['id']:<3} {d['status']:<11} {challenger[:32]:<32} "
              f"{counts.challenger_only}-{counts.champion_only:<3} {ties:<5} "
              f"{decision.p_value:.3g}  {d['alpha']:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
