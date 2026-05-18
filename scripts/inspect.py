from __future__ import annotations

import argparse
import json
import math
import sys

from affine.decide import BettingCS, decide
from affine.store import Store


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect affine validator state")
    parser.add_argument("--db", default="./.affine/affine.sqlite3")
    parser.add_argument("--duel-id", type=int, default=None)
    return parser.parse_args()


def _recompute(store: Store, duel_id: int) -> dict:
    row = store.duel(duel_id)
    if row is None:
        raise KeyError(f"duel {duel_id} not found")
    pi = {str(k): float(v) for k, v in json.loads(row["pi_json"]).items()}
    cs = BettingCS(alpha=float(row["alpha"]))
    by_round: dict[int, list] = {}
    for sample in store.samples_for_duel(duel_id):
        by_round.setdefault(sample.iter_idx, []).append(sample)
    expected_envs = set(pi)
    for round_idx in sorted(by_round):
        samples = by_round[round_idx]
        if {sample.env_id for sample in samples} != expected_envs:
            continue
        y = sum(pi[sample.env_id] * (int(sample.chal_correct) - int(sample.champ_correct)) for sample in samples)
        cs.update(y)
    verdict = decide(
        cs,
        delta_dethrone=float(row["delta_dethrone"]),
        delta_hold=float(row["delta_hold"]),
    )
    return {
        "duel_id": duel_id,
        "stored_status": row["status"],
        "recomputed_status": row["status"] if row["status"] in {"inconclusive", "cancelled", "challenger_slot_dead"} else verdict.reason,
        "delta_hat": verdict.delta_hat,
        "ci_low": verdict.ci_low,
        "ci_hi": verdict.ci_hi,
        "log_capital_at_zero": verdict.log_capital_at_zero,
        "log_capital_at_dethrone": cs._log_capital(float(row["delta_dethrone"]), +1),
        "rounds": verdict.rounds,
    }


def _verify(store: Store, duel_id: int) -> bool:
    row = store.duel(duel_id)
    assert row is not None
    got = _recompute(store, duel_id)
    checks = []
    for key in ("delta_hat", "ci_low", "ci_hi"):
        stored = row[key]
        if stored is None:
            continue
        checks.append(math.isclose(float(stored), float(got[key]), rel_tol=1e-12, abs_tol=1e-12))
    return all(checks)


def main() -> int:
    args = _parse_args()
    store = Store(args.db)
    try:
        champ = store.champion()
        if champ is None:
            print("champion: none")
        else:
            print(f"champion: {champ.model}@{champ.revision} uid={champ.uid} payable={champ.payable}")

        rows = store.db.execute(
            """
            SELECT id, status, rounds_collected, delta_hat, ci_low, ci_hi
            FROM duels ORDER BY id DESC LIMIT 20
            """
        ).fetchall()
        for row in rows:
            print(
                f"duel {row['id']}: {row['status']} rounds={row['rounds_collected']} "
                f"Δ̂={row['delta_hat']} CI=[{row['ci_low']}, {row['ci_hi']}]"
            )

        if args.duel_id is not None:
            result = _recompute(store, args.duel_id)
            print(json.dumps(result, indent=2, sort_keys=True))
            if not _verify(store, args.duel_id):
                print(f"duel {args.duel_id}: stored decision fields do not match recomputation", file=sys.stderr)
                return 1
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
