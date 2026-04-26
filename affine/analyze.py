"""Offline analysis of an evidence.jsonl: fit the same 2PL IRT the live loop
uses, report per-miner θ̂, per-env β̂/â, and pairwise verdicts.

No Targon, no inference, no chain. Runs in milliseconds. Use it to:

  - answer "given today's evidence, what's the ranking?"
  - answer "what would the verdict be if we re-ran the dethrone test now?"
  - diff verdicts across scoring-logic changes (run it, save output, change
    code, run again, diff)
  - A/B filtering rules on raw rows (see `--drop-fast-fails`) to check the
    impact of a sampler fix *before* burning compute on a live re-run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import Config
from .evidence import Row
from .irt import Priors, fit_2pl


def _reject_constant(c):
    raise ValueError(f"non-finite JSON constant: {c}")


def _load(path: Path, drop_fast_fail_s: float | None) -> list[Row]:
    rows: list[Row] = []
    dropped = 0
    skipped = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line, parse_constant=_reject_constant)
            r = Row(m=int(d["m"]), r=d["r"], e=d["e"], c=int(d["c"]),
                    p=int(d["p"]), t=int(d["t"]), l=float(d["l"]),
                    i=int(d.get("i", 0)))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            skipped += 1
            continue
        if drop_fast_fail_s is not None and r.p == 0 and r.l < drop_fast_fail_s:
            dropped += 1
            continue
        rows.append(r)
    if dropped:
        print(f"# dropped {dropped} rows (p=0 AND latency<{drop_fast_fail_s}s: presumed infra failures)")
    if skipped:
        print(f"# skipped {skipped} malformed rows")
    return rows


def _fit(rows: list[Row], priors: Priors):
    miners = sorted({(r.m, r.r) for r in rows})
    envs = sorted({r.e for r in rows})
    m2i = {k: i for i, k in enumerate(miners)}
    e2i = {k: i for i, k in enumerate(envs)}
    outcomes: dict[str, set[int]] = {}
    for r in rows:
        outcomes.setdefault(r.e, set()).add(r.p)
    drop = {n for n, outs in outcomes.items() if len(outs) < 2}
    used = [r for r in rows if r.e not in drop]
    fit = fit_2pl(
        np.array([m2i[(r.m, r.r)] for r in used], dtype=np.intp),
        np.array([e2i[r.e] for r in used], dtype=np.intp),
        np.array([r.p for r in used], dtype=np.float64),
        len(miners), len(envs), priors,
    )
    if drop:
        print(f"# dropped {len(drop)} zero-variance env(s) from fit: {sorted(drop)}")
    return fit, miners, envs


def _env_table(rows: list[Row]) -> dict[tuple[int, str], dict[str, tuple[int, int]]]:
    out: dict[tuple[int, str], dict[str, tuple[int, int]]] = {}
    for r in rows:
        d = out.setdefault((r.m, r.r), {})
        f, p = d.get(r.e, (0, 0))
        d[r.e] = (f + (1 - r.p), p + r.p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline replay of evidence.jsonl through the 2PL IRT fit.")
    ap.add_argument("path", nargs="?", default=".affine/evidence.jsonl",
                    help="path to evidence.jsonl (default: .affine/evidence.jsonl)")
    ap.add_argument("--drop-fast-fails", type=float, default=None, metavar="SECS",
                    help="retroactively drop p=0 rows with latency<SECS (infra-failure filter)")
    ap.add_argument("--contrast", type=int, nargs=2, metavar=("A", "B"),
                    help="print Δθ̂±SE and verdict for uid A as challenger vs uid B as king")
    ap.add_argument("--k", type=float, default=None,
                    help="dethrone threshold for --contrast (default: cfg.k_init)")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"no evidence at {path}"); return 1
    rows = _load(path, args.drop_fast_fails)
    if not rows:
        print("no rows after filtering"); return 1

    cfg = Config.from_env()
    fit, miners, envs = _fit(rows, Priors(sigma_beta=cfg.sigma_beta, sigma_alpha=cfg.sigma_alpha))
    table = _env_table(rows)

    print(f"# {len(rows)} rows, {len(miners)} (uid,rev) pairs, {len(envs)} envs\n")
    if fit.degenerate:
        print("# WARNING: fit is DEGENERATE (optimizer non-converged or non-PD Hessian).")
        print("# θ̂, SE, and contrast verdicts below are NOT a valid Laplace posterior.")
        print("# The live loop refuses to elect or dethrone on a degenerate fit; treat")
        print("# the numbers as diagnostic only.\n")
    print("## Env parameters (sorted by discrimination)")
    print(f"{'env':>14}  {'a (disc)':>10}  {'β (diff)':>10}")
    for i in np.argsort(-fit.a):
        print(f"{envs[i]:>14}  {fit.a[i]:>10.3f}  {fit.beta[i]:>+10.3f}")

    print("\n## Miner ranking (θ̂, descending)")
    print(f"{'uid':>5}  {'rev':>10}  {'θ̂':>8}  {'SE':>6}   per-env F/P")
    for i in np.argsort(-fit.theta):
        uid, rev = miners[i]
        env_cells = "  ".join(f"{e}:{f}/{p}" for e, (f, p) in sorted(table.get((uid, rev), {}).items()))
        print(f"{uid:>5}  {rev[:10]:>10}  {fit.theta[i]:>+8.3f}  {fit.theta_se[i]:>6.3f}   {env_cells}")

    if args.contrast:
        chal_uid, king_uid = args.contrast
        # Pick the most-recently-active revision per uid: a re-committed miner
        # has multiple (uid, rev) entries, and `next()` over a sorted list
        # would silently grab the lex-first revision — usually the OLDEST,
        # whose θ̂ may differ from the live loop's current θ̂. Tie-break by
        # max-row-t (latest evidence wins).
        last_t = {(r.m, r.r): 0 for r in rows}
        for r in rows:
            k = (r.m, r.r)
            if r.t > last_t[k]: last_t[k] = r.t
        def _pick(uid: int) -> int:
            cands = [i for i, (u, _) in enumerate(miners) if u == uid]
            if not cands:
                raise StopIteration
            if len(cands) > 1:
                rev = miners[max(cands, key=lambda i: last_t[miners[i]])][1]
                print(f"# uid {uid} has {len(cands)} revisions in evidence; using most-recent: {rev}")
            return max(cands, key=lambda i: last_t[miners[i]])
        try:
            ci = _pick(chal_uid)
            ki = _pick(king_uid)
        except StopIteration:
            print(f"\nuid {chal_uid} or {king_uid} not in evidence"); return 1
        delta, se = fit.contrast(ci, ki)
        z = delta / se if se > 0 else 0.0
        k = args.k if args.k is not None else cfg.k_init
        verdict = "CHALLENGER_WINS" if z > k else "CHAMPION_HOLDS"
        print(f"\n## Contrast: challenger uid{chal_uid} vs king uid{king_uid}")
        print(f"Δθ̂ = {delta:+.3f} ± {se:.3f}    z = {z:+.2f}    k = {k:.2f}    → {verdict}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
