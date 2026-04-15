"""Validator trust via Fisher information.

The Bradley-Terry model is the Rasch model (1PL IRT) for pairwise comparisons.
The variance from bt_mle is the inverse Fisher information of the log-odds
estimate.  VTrust extends this to multiple validators: each contributes an
independent BT estimate (from their own committed-seed task set), weighted by
trust * precision.  The merge is statistically optimal inverse-variance
weighting with a trust discount.

All functions are pure — no I/O, no chain interaction.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from math import sqrt
from .scoring import Verdict, bt_mle, aggregate


@dataclass(frozen=True)
class ValidatorEvidence:
    """One validator's evidence for a single duel (champion vs challenger)."""
    hotkey: str
    wins: dict[str, int]    # env -> decisive wins for challenger
    losses: dict[str, int]  # env -> decisive losses for challenger
    tasks: dict[str, int]   # env -> total tasks evaluated


# ---------------------------------------------------------------------------
# Information measures
# ---------------------------------------------------------------------------

def fisher_info(w: int, l: int) -> float:
    """Fisher information of the BT log-odds estimate = 1/var.
    This is the precision: how tightly the estimate pins down delta."""
    _, var = bt_mle(w, l)
    return 1.0 / var


def per_trial_info(w: int, l: int) -> float:
    """Fisher information per decisive trial: p*(1-p) at the MLE.
    Maximum (0.25) when wins == losses.  Approaches 0 as one side dominates.
    Measures how informative the environment is for this model pair."""
    n = w + l + 1  # effective n with Jeffreys pseudocounts
    p = (w + 0.5) / n
    return p * (1.0 - p)


def validator_info(ev: ValidatorEvidence) -> float:
    """Total Fisher information contributed by one validator across all envs."""
    total = 0.0
    for name in ev.wins:
        w, l = ev.wins[name], ev.losses[name]
        if w + l == 0:
            continue
        total += fisher_info(w, l)
    return total


# ---------------------------------------------------------------------------
# Bayesian trust
# ---------------------------------------------------------------------------

def update_trust(
    alpha: float, beta: float, verified: int, failed: int,
) -> tuple[float, float, float]:
    """Bayesian trust update via Beta-Binomial conjugacy.

    Prior:     Beta(alpha, beta)
    Posterior: Beta(alpha + verified, beta + failed)

    Returns (trust_score, new_alpha, new_beta) where trust_score is the
    posterior mean.  One caught fabrication poisons the entire posterior —
    remaining un-replayed samples become suspect too.
    """
    a = alpha + verified
    b = beta + failed
    return a / (a + b), a, b


# ---------------------------------------------------------------------------
# Multi-validator evidence merge
# ---------------------------------------------------------------------------

def merge_evidence(
    evidence: list[ValidatorEvidence],
    trust: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Merge multiple validators into per-environment (delta, var).

    Each validator's per-env BT estimate is weighted by trust * precision.
    Returns ({env: delta}, {env: var}).  Reduces to single-validator bt_mle
    when len(evidence) == 1 and trust == 1.0.
    """
    env_names: set[str] = set()
    for ev in evidence:
        env_names.update(
            n for n in ev.wins if ev.wins[n] + ev.losses[n] > 0
        )

    merged_d: dict[str, float] = {}
    merged_v: dict[str, float] = {}

    for env in env_names:
        deltas, weights = [], []
        for ev in evidence:
            w = ev.wins.get(env, 0)
            l = ev.losses.get(env, 0)
            if w + l == 0:
                continue
            t = trust.get(ev.hotkey, 0.0)
            if t <= 0.0:
                continue
            d, v = bt_mle(w, l)
            deltas.append(d)
            weights.append(t / v)

        if deltas:
            tw = sum(weights)
            merged_d[env] = sum(d * wi for d, wi in zip(deltas, weights)) / tw
            merged_v[env] = 1.0 / tw

    return merged_d, merged_v


def merged_check_duel(
    evidence: list[ValidatorEvidence],
    trust: dict[str, float],
    max_tasks: int,
    k: float,
) -> tuple[Verdict, float]:
    """Full duel verdict from merged multi-validator evidence.

    Merges per-env estimates across validators, then applies the same
    z-score threshold and hopelessness check as check_duel.
    """
    merged_d, merged_v = merge_evidence(evidence, trust)

    if not merged_d:
        return Verdict.UNDECIDED, 0.0

    envs = list(merged_d.keys())
    delta, var = aggregate(
        [merged_d[e] for e in envs],
        [merged_v[e] for e in envs],
    )
    z = delta / sqrt(var)

    if z > k:
        return Verdict.CHALLENGER_WINS, z

    # Hopelessness: best-case remaining budget across all validators.
    # Each validator's remaining per-env budget contributes optimistically.
    best_d_list, best_v_list = [], []
    for env in envs:
        best_deltas, best_weights = [], []
        for ev in evidence:
            t = trust.get(ev.hotkey, 0.0)
            if t <= 0.0:
                continue
            w = ev.wins.get(env, 0)
            l = ev.losses.get(env, 0)
            done = ev.tasks.get(env, 0)
            remaining = max(0, max_tasks - done)
            bw, bl = w + remaining, l
            if bw + bl == 0:
                continue
            d, v = bt_mle(bw, bl)
            best_deltas.append(d)
            best_weights.append(t / v)

        if best_deltas:
            tw = sum(best_weights)
            best_d_list.append(
                sum(d * wi for d, wi in zip(best_deltas, best_weights)) / tw
            )
            best_v_list.append(1.0 / tw)

    if best_d_list:
        bd, bv = aggregate(best_d_list, best_v_list)
        if bd / sqrt(bv) <= k:
            return Verdict.CHAMPION_HOLDS, z

    return Verdict.UNDECIDED, z


# ---------------------------------------------------------------------------
# Reward shares
# ---------------------------------------------------------------------------

def reward_shares(
    evidence: list[ValidatorEvidence],
    trust: dict[str, float],
) -> dict[str, float]:
    """Per-validator reward as fraction of total trust-weighted information.

    reward_v = (trust_v * I_v) / sum_j(trust_j * I_j)

    Validators who contribute more precise evidence (more decisive outcomes
    in balanced environments) earn proportionally more.  Fabrication tanks
    trust, which tanks reward.
    """
    infos: dict[str, float] = {}
    for ev in evidence:
        t = trust.get(ev.hotkey, 0.0)
        infos[ev.hotkey] = max(0.0, t) * validator_info(ev)

    total = sum(infos.values())
    if total <= 0.0:
        return {hk: 0.0 for hk in infos}
    return {hk: v / total for hk, v in infos.items()}
