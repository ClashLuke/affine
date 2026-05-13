"""Offline replay of the validator SQLite DB. Pure read; safe while the loop runs.

Reports the most recent duels with their cell-rating verdicts plus the
current archive_fit's per-miner θ ranking and per-env (β_e, a_e). Replaces
the pair-replay analyze that depended on the deleted `paired.py`.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path

from .config import Config, EnvSpec
from .irt import archive_fit, CellTriple, policy_rating
from .score import env_versioning, normalize_pi, normalize_rho
from .store import Store

import numpy as np


def _build_archive_from_store(store: Store, cfg: Config):
    """Replay-side archive_fit. Mirrors loop._build_archive_snapshot but without
    needing async chain/env machinery."""
    env_states = {}
    for spec in cfg.environments:
        ev, ts_hash, gh = env_versioning(spec)
        existing = store.env_state(spec.name)
        env_states[spec.name] = {
            "state": existing["state"] if existing else "score_active",
            "env_version": ev,
            "task_spec_hash": ts_hash,
            "grader_hash": gh,
            "serving_hash": "default",
        }
    measurement = {e for e, s in env_states.items()
                   if s["state"] in ("calibration_only", "score_active")}
    if not measurement:
        return None, [], [], {}
    cells = store.cells_view(
        env_set=measurement,
        env_version={e: env_states[e]["env_version"] for e in measurement},
        task_spec_hash={e: env_states[e]["task_spec_hash"] for e in measurement},
        grader_hash={e: env_states[e]["grader_hash"] for e in measurement},
        serving_hash={e: env_states[e]["serving_hash"] for e in measurement},
        rule="first",
    )
    miner_ids = sorted({c.miner_artifact_id for c in cells})
    env_ids = sorted(measurement)
    mi = {m: i for i, m in enumerate(miner_ids)}
    ei = {e: i for i, e in enumerate(env_ids)}
    triples = [CellTriple(mi[c.miner_artifact_id], ei[c.env_id], int(c.outcome)) for c in cells]
    rho = normalize_rho({}, env_ids)
    snap = archive_fit(miner_ids, env_ids, triples, rho)
    return snap, miner_ids, env_ids, env_states


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline replay of validator SQLite evidence.")
    ap.add_argument("path", nargs="?", default=None)
    ap.add_argument("--ranking", action="store_true",
                    help="Print archive_fit's miner θ + R ranking")
    ap.add_argument("--envs", action="store_true",
                    help="Print per-env β_e and a_e")
    args = ap.parse_args()

    cfg = Config.from_env()
    path = Path(args.path or cfg.db_path)
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
    db.close()

    duels = sqlite3.connect(path)
    duels.row_factory = sqlite3.Row
    rows = duels.execute("SELECT * FROM duels ORDER BY id DESC LIMIT 20").fetchall()
    duels.close()
    if not rows:
        print("no duels")
        return 0
    print("id  status                    challenger                      cells   Δθ      SE_θ    R_diag   δ_θ")
    for d in rows:
        challenger = f"uid{d['challenger_uid']} {d['challenger_model']}@{d['challenger_revision']}"
        keys = d.keys()
        cells_collected = d["cells_collected"] if "cells_collected" in keys else None
        delta_theta = d["delta_theta_observed"] if "delta_theta_observed" in keys else None
        se_theta = d["se_theta"] if "se_theta" in keys else None
        rating_diff = d["rating_diff_diagnostic"] if "rating_diff_diagnostic" in keys else None
        delta_threshold = d["delta_theta"] if "delta_theta" in keys else None
        cells_str = str(cells_collected) if cells_collected is not None else "—"
        dt_str = f"{delta_theta:+.4f}" if delta_theta is not None else "—"
        se_str = f"{se_theta:.4f}" if se_theta is not None else "—"
        rd_str = f"{rating_diff:+.4f}" if rating_diff is not None else "—"
        thr_str = f"{delta_threshold:.3g}" if delta_threshold is not None else "—"
        print(f"{d['id']:<3} {d['status']:<24} {challenger[:32]:<32} "
              f"{cells_str:<7} {dt_str:<7} {se_str:<7} {rd_str:<7} {thr_str}")

    if args.ranking or args.envs:
        store = Store(path)
        snap, miner_ids, env_ids, env_states = _build_archive_from_store(store, cfg)
        if snap is None or not miner_ids:
            print("\n(no archive_fit available; need cell observations)")
            store.close()
            return 0
        if args.ranking:
            pi = {e: 1.0 / len(env_ids) for e in env_ids}
            mu = snap.mu
            beta = np.asarray(snap.beta)
            a = np.exp(np.clip(np.asarray(snap.log_a), -10, 10))
            pi_arr = np.asarray([pi[e] for e in env_ids])
            print(f"\n# archive_fit ranking ({len(miner_ids)} miners, {snap.n_cells} cells)")
            print("miner_artifact_id        θ_c       R_c       S_c")
            entries = []
            for i, mid in enumerate(miner_ids):
                s, r = policy_rating(mu, beta, a, snap.theta[i], pi_arr)
                entries.append((mid, snap.theta[i], r, s))
            entries.sort(key=lambda e: -e[2])
            for mid, theta, r, s in entries:
                print(f"{mid:<24} {theta:+.4f}  {r:+.4f}  {s:.4f}")
        if args.envs:
            print(f"\n# archive_fit env params ({len(env_ids)} envs)")
            print("env_id                   β_e       a_e       state")
            for i, e in enumerate(env_ids):
                a_e = math.exp(min(max(snap.log_a[i], -10), 10))
                state = env_states.get(e, {}).get("state", "?")
                print(f"{e:<24} {snap.beta[i]:+.4f}  {a_e:+.4f}  {state}")
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
