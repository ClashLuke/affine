"""Offline replay of the validator SQLite DB. Pure read; safe while the loop runs."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from .config import Config
from .paired import (
    PairCounts,
    decide_dethrone,
    env_lower_cs,
    env_upper_cs,
)


def _per_env_counts(db: sqlite3.Connection, duel_id: int) -> dict[str, PairCounts]:
    rows = db.execute(
        """
        SELECT env, champion_pass, challenger_pass
        FROM samples
        WHERE duel_id=? AND champion_delivered=1 AND challenger_delivered=1
        """,
        (duel_id,),
    ).fetchall()
    out: dict[str, PairCounts] = {}
    for r in rows:
        env = str(r["env"])
        out[env] = out.get(env, PairCounts()).add(
            int(r["champion_pass"]), int(r["challenger_pass"])
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline replay of validator SQLite evidence.")
    ap.add_argument("path", nargs="?", default=Config.from_env().db_path)
    ap.add_argument("--per-env", action="store_true",
                    help="Show per-env (k,n,L,U) for the most recent duel")
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
        backup = db.execute(
            "SELECT manifest_key FROM backups WHERE artifact_id=? AND status='current'",
            (champ["artifact_id"],),
        ).fetchone()
        manifest = backup["manifest_key"] if backup else ""
        print(f"# champion {champ['model']}@{champ['revision']} {pay} backup={manifest}")
    duels = db.execute("SELECT * FROM duels ORDER BY id DESC LIMIT 20").fetchall()
    if not duels:
        print("no duels")
        return 0
    print("id  status         challenger                      W-L     ties   "
          "L_mu    U_mu    alpha   delta_p")
    for d in duels:
        per_env = _per_env_counts(db, int(d["id"]))
        try:
            weights = json.loads(d["env_weights_json"]) if d["env_weights_json"] else {}
        except (TypeError, ValueError):
            weights = {}
        alpha = float(d["alpha"])
        delta_p = float(d["delta_p"]) if d["delta_p"] is not None else 0.05
        p_star = 0.5 + delta_p
        l_mu = u_mu = None
        if weights and set(per_env) <= set(weights):
            for e in weights:
                per_env.setdefault(e, PairCounts())
            decision = decide_dethrone(
                per_env, weights, p_star=p_star,
                alpha_dethrone=alpha / 2.0, alpha_futility=alpha / 2.0,
            )
            l_mu, u_mu = decision.L_mu, decision.U_mu
        elif d["l_mu"] is not None:
            l_mu, u_mu = float(d["l_mu"]), float(d["u_mu"])

        total = PairCounts()
        for c in per_env.values():
            total = PairCounts(
                challenger_only=total.challenger_only + c.challenger_only,
                champion_only=total.champion_only + c.champion_only,
                both_pass=total.both_pass + c.both_pass,
                both_fail=total.both_fail + c.both_fail,
            )
        ties = total.both_pass + total.both_fail
        challenger = f"uid{d['challenger_uid']} {d['challenger_model']}@{d['challenger_revision']}"
        l_str = f"{l_mu:.3f}" if l_mu is not None else "—"
        u_str = f"{u_mu:.3f}" if u_mu is not None else "—"
        print(f"{d['id']:<3} {d['status']:<14} {challenger[:32]:<32} "
              f"{total.challenger_only}-{total.champion_only:<5} {ties:<6} "
              f"{l_str:<7} {u_str:<7} {alpha:.3g}  {delta_p:.3g}")

    if args.per_env and duels:
        d = duels[0]
        per_env = _per_env_counts(db, int(d["id"]))
        try:
            weights = json.loads(d["env_weights_json"]) if d["env_weights_json"] else {}
        except (TypeError, ValueError):
            weights = {}
        if not weights:
            print("\n(no env weights stored for this duel; cannot compute per-env CSs)")
            return 0
        alpha = float(d["alpha"])
        e_count = len(weights)
        am = (alpha / 2.0) / e_count
        ap_ = (alpha / 2.0) / e_count
        print(f"\n# per-env (duel {d['id']}, alpha/2/E={am:.4g})")
        print("env                      pi    k     n     L       U       p_hat")
        for e in sorted(weights):
            c = per_env.get(e, PairCounts())
            k, n = c.challenger_only, c.discordant
            L = env_lower_cs(k, n, am)
            U = env_upper_cs(k, n, ap_)
            phat = (k / n) if n > 0 else float("nan")
            print(f"{e:<24} {weights[e]:.3f}  {k:<5} {n:<5} {L:.4f}  {U:.4f}  {phat:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
