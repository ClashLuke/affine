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
from pathlib import Path

import numpy as np

from .config import Config
from .evidence import Row, read_rows
from .irt import Priors, fit_2pl


def _row_art(r: Row) -> tuple[str, str]:
    """Offline analyze (no live miner table): legacy rows without `k` get a
    per-uid ghost key so two old uids that happened to share `revision` don't
    pool. New rows pool by (model, rev)."""
    return ((r.k, r.r) if r.k is not None else (f"?ghost:{r.m}:{r.r}", r.r))


def _fit(rows: list[Row], priors: Priors):
    art_keys = sorted({_row_art(r) for r in rows})
    envs = sorted({r.e for r in rows})
    k2i = {k: i for i, k in enumerate(art_keys)}
    e2i = {k: i for i, k in enumerate(envs)}
    outcomes: dict[str, set[int]] = {}
    for r in rows:
        outcomes.setdefault(r.e, set()).add(r.p)
    drop = {n for n, outs in outcomes.items() if len(outs) < 2}
    used = [r for r in rows if r.e not in drop]
    fit = fit_2pl(
        np.array([k2i[_row_art(r)] for r in used], dtype=np.intp),
        np.array([e2i[r.e] for r in used], dtype=np.intp),
        np.array([r.p for r in used], dtype=np.float64),
        len(art_keys), len(envs), priors,
    )
    if drop:
        print(f"# dropped {len(drop)} zero-variance env(s) from fit: {sorted(drop)}")
    return fit, art_keys, envs


def _env_table(rows: list[Row]) -> dict[tuple[str, str], dict[str, tuple[int, int]]]:
    out: dict[tuple[str, str], dict[str, tuple[int, int]]] = {}
    for r in rows:
        d = out.setdefault(_row_art(r), {})
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
    rows = read_rows(path)
    if args.drop_fast_fails is not None:
        before = len(rows)
        rows = [r for r in rows if not (r.p == 0 and r.l < args.drop_fast_fails)]
        if before - len(rows):
            print(f"# dropped {before - len(rows)} rows (p=0 AND latency<{args.drop_fast_fails}s: presumed infra failures)")
    if not rows:
        print("no rows after filtering"); return 1

    cfg = Config.from_env()
    fit, art_keys, envs = _fit(rows, Priors(sigma_beta=cfg.sigma_beta, sigma_alpha=cfg.sigma_alpha))
    table = _env_table(rows)
    k2i = {k: i for i, k in enumerate(art_keys)}

    # uid → most-recent (model, rev) seen in rows; used for display + --contrast lookup.
    last_t: dict[int, int] = {}
    last_art: dict[int, tuple[str, str]] = {}
    for r in rows:
        if r.t >= last_t.get(r.m, -1):
            last_t[r.m] = r.t
            last_art[r.m] = _row_art(r)

    print(f"# {len(rows)} rows, {len(art_keys)} artifacts, {len(envs)} envs\n")
    if fit.degenerate:
        print("# WARNING: fit is DEGENERATE (optimizer non-converged or non-PD Hessian).")
        print("# θ̂, SE, and contrast verdicts below are NOT a valid Laplace posterior.")
        print("# The live loop refuses to elect or dethrone on a degenerate fit; treat")
        print("# the numbers as diagnostic only.\n")
    print("## Env parameters (sorted by discrimination)")
    print(f"{'env':>14}  {'a (disc)':>10}  {'β (diff)':>10}")
    for i in np.argsort(-fit.a):
        print(f"{envs[i]:>14}  {fit.a[i]:>10.3f}  {fit.beta[i]:>+10.3f}")

    art_uids: dict[tuple[str, str], set[int]] = {}
    for r in rows:
        art_uids.setdefault(_row_art(r), set()).add(r.m)

    print("\n## Artifact ranking (θ̂, descending)")
    print(f"{'uids':>10}  {'model':>20}  {'rev':>10}  {'θ̂':>8}  {'SE':>6}   per-env F/P")
    for i in np.argsort(-fit.theta):
        model, rev = art_keys[i]
        uids = ",".join(str(u) for u in sorted(art_uids.get(art_keys[i], ())))
        env_cells = "  ".join(f"{e}:{f}/{p}" for e, (f, p) in sorted(table.get(art_keys[i], {}).items()))
        print(f"{uids[:10]:>10}  {model[:20]:>20}  {rev[:10]:>10}  {fit.theta[i]:>+8.3f}  {fit.theta_se[i]:>6.3f}   {env_cells}")

    if args.contrast:
        chal_uid, king_uid = args.contrast
        if chal_uid not in last_art or king_uid not in last_art:
            print(f"\nuid {chal_uid} or {king_uid} not in evidence"); return 1
        ci, ki = k2i[last_art[chal_uid]], k2i[last_art[king_uid]]
        delta, se = fit.contrast(ci, ki)
        z = delta / se if se > 0 else 0.0
        k = args.k if args.k is not None else cfg.k_init
        verdict = "CHALLENGER_WINS" if z > k else "CHAMPION_HOLDS"
        print(f"\n## Contrast: challenger uid{chal_uid} ({last_art[chal_uid][0]}@{last_art[chal_uid][1]}) "
              f"vs king uid{king_uid} ({last_art[king_uid][0]}@{last_art[king_uid][1]})")
        print(f"Δθ̂ = {delta:+.3f} ± {se:.3f}    z = {z:+.2f}    k = {k:.2f}    → {verdict}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
