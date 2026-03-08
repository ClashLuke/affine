#!/usr/bin/env python3
"""
ELO crystallization and confidence analysis.

Analyzes when top-3 miners crystallize per game type and computes
Bayesian credible intervals using the Bradley-Terry model.
"""

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import gaussian_kde

from affine.src.elo.bayesian import BayesianBradleyTerry, BayesianResult, HAS_NUMPYRO
from _replay import OUTCOME_MAP


@dataclass
class MinerStats:
    hotkey: str
    uid: Optional[int] = None
    model: Optional[str] = None
    wins: int = 0
    losses: int = 0
    draws: int = 0
    final_elo: float = 1500.0

    @property
    def total_matches(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.total_matches if self.total_matches else 0.5


@dataclass
class ConfidenceInterval:
    point_estimate: float
    lower: float
    upper: float
    width: float
    mode: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None


def compute_posterior_mode(samples: np.ndarray) -> float:
    if len(samples) < 10:
        return float(np.mean(samples))
    kde = gaussian_kde(samples)
    x_grid = np.linspace(samples.min(), samples.max(), 500)
    return float(x_grid[np.argmax(kde(x_grid))])


def wilson_score_interval(successes: int, total: int, z: float = 1.96) -> ConfidenceInterval:
    if total == 0:
        return ConfidenceInterval(0.5, 0.0, 1.0, 1.0)
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) / total) + (z**2 / (4 * total**2))) / denom
    lo, hi = max(0.0, center - margin), min(1.0, center + margin)
    return ConfidenceInterval(p, lo, hi, hi - lo)


def elo_confidence_interval(stats: MinerStats, base_elo: float = 1500.0) -> ConfidenceInterval:
    effective_wins = stats.wins + 0.5 * stats.draws
    win_ci = wilson_score_interval(int(effective_wins), stats.total_matches)

    def wr_to_elo(wr):
        if wr <= 0.001: return base_elo - 800
        if wr >= 0.999: return base_elo + 800
        return base_elo - 400 * math.log10(1 / wr - 1)

    pt = wr_to_elo(win_ci.point_estimate)
    lo = wr_to_elo(win_ci.lower)
    hi = wr_to_elo(win_ci.upper)
    return ConfidenceInterval(pt, lo, hi, hi - lo)


class BayesianBradleyTerryWrapper:
    """Wraps BayesianBradleyTerry for credible intervals with posterior mode."""

    def __init__(self, base_elo: float = 1500.0):
        self.base_elo = base_elo
        self._model = BayesianBradleyTerry()
        self._result: Optional[BayesianResult] = None
        self.players: list[str] = []
        self.matches: list[tuple[str, str, float]] = []

    def add_match(self, player_a: str, player_b: str, outcome_a: str):
        for p in (player_a, player_b):
            if p not in self.players:
                self.players.append(p)

        outcome_map = {"win": ("a_wins", 1.0), "loss": ("b_wins", 0.0),
                       "timeout": ("b_wins", 0.0), "error": ("b_wins", 0.0)}
        bt_outcome, score = outcome_map.get(outcome_a, ("draw", 0.5))
        self.matches.append((player_a, player_b, score))
        self._model.add_match(player_a, player_b, bt_outcome)

    def fit(self, num_samples: int = 2000, num_warmup: int = 1000,
            confidence_level: float = 0.95) -> bool:
        if len(self.players) < 2:
            return False
        try:
            self._result = self._model.fit(
                estimate_first_mover=False, num_samples=num_samples,
                num_warmup=num_warmup, num_chains=1, confidence_level=confidence_level,
            )
            return True
        except Exception as e:
            print(f"Bayesian fit failed: {e}")
            return False

    def get_elo(self, player: str) -> float:
        if self._result is None or player not in self._result.rating_means:
            return self.base_elo
        return float(self._result.rating_means[player])

    def get_confidence_interval(self, player: str) -> ConfidenceInterval:
        if self._result is None or player not in self._result.rating_means:
            return ConfidenceInterval(self.base_elo, self.base_elo - 400, self.base_elo + 400, 800)

        mean_elo = float(self._result.rating_means[player])
        ci_lo = float(self._result.rating_ci_lower[player])
        ci_hi = float(self._result.rating_ci_upper[player])
        std_elo = float(self._result.rating_stds[player])

        mode_elo = mean_elo
        if player in self._result.skill_samples:
            scale = 400 / np.log(10)
            elo_samples = scale * self._result.skill_samples[player] + self.base_elo
            mode_elo = compute_posterior_mode(elo_samples)

        return ConfidenceInterval(mean_elo, ci_lo, ci_hi, ci_hi - ci_lo,
                                  mode=mode_elo, mean=mean_elo, std=std_elo)

    def get_all_elos(self) -> dict[str, float]:
        if self._result is None:
            return {}
        return {p: float(r) for p, r in self._result.rating_means.items()}

    def get_all_confidence_intervals(self) -> dict[str, ConfidenceInterval]:
        if self._result is None:
            return {}
        return {p: self.get_confidence_interval(p) for p in self._result.rating_means}


@dataclass
class CrystallizationResult:
    game_type: str
    total_matches: int
    final_top3_set: set
    final_top3_order: list
    set_crystallization_match: Optional[int] = None
    order_crystallization_match: Optional[int] = None
    first_place_crystallized_match: Optional[int] = None
    top3_crystallized_match: Optional[int] = None


def _ci_for_stats(wins, losses, draws):
    total = wins + losses + draws
    if total < 2:
        return None
    return elo_confidence_interval(MinerStats(hotkey="", wins=wins, losses=losses, draws=draws))


def _cis_overlap(ci1, ci2):
    if ci1 is None or ci2 is None:
        return True
    return ci1.lower <= ci2.upper and ci2.lower <= ci1.upper


def replay_matches(matches, miners_lookup):
    """Replay all matches through EloCalculator, tracking per-game ELO, stats, and top-3 history."""
    from elo.calculator import EloCalculator
    from elo.config import EloConfig
    from decimal import Decimal

    calc = EloCalculator(EloConfig())

    elo_by_game = defaultdict(lambda: defaultdict(lambda: Decimal("1500")))
    matches_by_game = defaultdict(lambda: defaultdict(int))
    stats_by_game = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "draws": 0}))
    global_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "draws": 0})

    top3_history = defaultdict(list)
    global_top3_history = []
    ci_history_by_game = defaultdict(list)
    global_ci_history = []

    bt_by_game = defaultdict(BayesianBradleyTerryWrapper)
    bt_global = BayesianBradleyTerryWrapper()

    miner_stats_by_game = defaultdict(dict)
    miner_global_stats = {}

    # Incremental replay for crystallization tracking
    inc_global_elo = defaultdict(lambda: Decimal("1500"))
    inc_global_matches = defaultdict(int)

    for match in sorted(matches, key=lambda m: m["timestamp"]):
        game_type = match["game_type"]
        participants = match.get("participants", [])
        if len(participants) != 2:
            continue

        p1, p2 = participants
        hk1, hk2 = p1["miner_hotkey"], p2["miner_hotkey"]
        outcome1 = p1["outcome"]
        outcome = OUTCOME_MAP.get(outcome1, "draw")

        for hk in (hk1, hk2):
            info = miners_lookup.get(hk, {})
            if hk not in miner_stats_by_game[game_type]:
                miner_stats_by_game[game_type][hk] = MinerStats(hk, uid=info.get("uid"), model=info.get("model"))
            if hk not in miner_global_stats:
                miner_global_stats[hk] = MinerStats(hk, uid=info.get("uid"), model=info.get("model"))

        # Update per-game ELO through calculator
        new_r1, new_r2, _, _ = calc.update_ratings_head_to_head(
            rating_a=elo_by_game[game_type][hk1], rating_b=elo_by_game[game_type][hk2],
            matches_a=matches_by_game[game_type][hk1], matches_b=matches_by_game[game_type][hk2],
            outcome=outcome,
        )
        elo_by_game[game_type][hk1] = new_r1
        elo_by_game[game_type][hk2] = new_r2
        matches_by_game[game_type][hk1] += 1
        matches_by_game[game_type][hk2] += 1

        # Global ELO (incremental)
        new_g1, new_g2, _, _ = calc.update_ratings_head_to_head(
            rating_a=inc_global_elo[hk1], rating_b=inc_global_elo[hk2],
            matches_a=inc_global_matches[hk1], matches_b=inc_global_matches[hk2],
            outcome=outcome,
        )
        inc_global_elo[hk1] = new_g1
        inc_global_elo[hk2] = new_g2
        inc_global_matches[hk1] += 1
        inc_global_matches[hk2] += 1

        # Stats — timeout/error count as losses (consistent with ELO update)
        for p in participants:
            hk, out = p["miner_hotkey"], p["outcome"]
            gs = miner_stats_by_game[game_type][hk]
            gl = miner_global_stats[hk]
            rg = stats_by_game[game_type][hk]
            rgl = global_stats[hk]
            for s, r in [(gs, rg), (gl, rgl)]:
                if out == "win": s.wins += 1; r["wins"] += 1
                elif out in ("loss", "timeout", "error"): s.losses += 1; r["losses"] += 1
                else: s.draws += 1; r["draws"] += 1
            gs.final_elo = float(elo_by_game[game_type][hk])

        bt_by_game[game_type].add_match(hk1, hk2, outcome1)
        bt_global.add_match(hk1, hk2, outcome1)

        # Top-3 tracking
        sorted_game = sorted(elo_by_game[game_type].items(), key=lambda x: -float(x[1]))
        top3_history[game_type].append([hk for hk, _ in sorted_game[:3]])

        def check_ci_crystal(sorted_miners, running_stats):
            info = {"first_crystallized": False, "top3_crystallized": False}
            if len(sorted_miners) >= 2:
                s1, s2 = running_stats[sorted_miners[0][0]], running_stats[sorted_miners[1][0]]
                info["first_crystallized"] = not _cis_overlap(
                    _ci_for_stats(s1["wins"], s1["losses"], s1["draws"]),
                    _ci_for_stats(s2["wins"], s2["losses"], s2["draws"]))
            if len(sorted_miners) >= 4:
                s3, s4 = running_stats[sorted_miners[2][0]], running_stats[sorted_miners[3][0]]
                info["top3_crystallized"] = not _cis_overlap(
                    _ci_for_stats(s3["wins"], s3["losses"], s3["draws"]),
                    _ci_for_stats(s4["wins"], s4["losses"], s4["draws"]))
            return info

        ci_history_by_game[game_type].append(check_ci_crystal(sorted_game[:4], stats_by_game[game_type]))

        sorted_global = sorted(inc_global_elo.items(), key=lambda x: -float(x[1]))
        global_top3_history.append([hk for hk, _ in sorted_global[:3]])
        global_ci_history.append(check_ci_crystal(sorted_global[:4], global_stats))

    # Use incremental global ELO for final stats
    for hk, stats in miner_global_stats.items():
        stats.final_elo = float(inc_global_elo.get(hk, Decimal("1500")))

    print("Fitting Bayesian Bradley-Terry models...")
    for gt, bt in bt_by_game.items():
        if bt.fit():
            print(f"  {gt}: {len(bt.players)} players, {len(bt.matches)} matches")
    if bt_global.fit():
        print(f"  GLOBAL: {len(bt_global.players)} players, {len(bt_global.matches)} matches")

    return {
        "top3_history": dict(top3_history), "global_top3_history": global_top3_history,
        "ci_history_by_game": dict(ci_history_by_game), "global_ci_history": global_ci_history,
        "bt_by_game": dict(bt_by_game), "bt_global": bt_global,
        "stats_by_game": dict(miner_stats_by_game), "global_stats": dict(miner_global_stats),
        "global_elo": {hk: float(r) for hk, r in inc_global_elo.items()},
    }


def find_crystallization(top3_history, global_top3_history, ci_history_by_game,
                         global_ci_history, matches):
    match_counts = defaultdict(int)
    for m in matches:
        match_counts[m["game_type"]] += 1

    results = {}

    def _find_crystal(history, ci_history, game_type, total):
        if not history:
            return None
        final = history[-1]
        final_set = set(final)

        set_at = order_at = 0
        for i in range(len(history) - 1, -1, -1):
            if set(history[i]) != final_set:
                set_at = i + 1
                break
            if history[i] != final:
                order_at = i + 1
        order_at = max(order_at, set_at)

        r = CrystallizationResult(game_type, total, final_set, list(final),
                                  set_at, order_at)

        if ci_history:
            for i, ci in enumerate(ci_history):
                if ci["first_crystallized"] and r.first_place_crystallized_match is None:
                    r.first_place_crystallized_match = i + 1
                if ci["top3_crystallized"] and r.top3_crystallized_match is None:
                    r.top3_crystallized_match = i + 1
        return r

    for gt, hist in top3_history.items():
        r = _find_crystal(hist, ci_history_by_game.get(gt, []), gt, match_counts[gt])
        if r:
            results[gt] = r

    r = _find_crystal(global_top3_history, global_ci_history, "GLOBAL", len(matches))
    if r:
        results["GLOBAL"] = r

    return results


def compute_cis(stats_by_game, global_stats, bt_by_game, bt_global):
    def _cis_for(stats_dict, bt_model):
        cis = []
        for hk, stats in stats_dict.items():
            if stats.total_matches < 2:
                continue
            if bt_model and bt_model._result and hk in bt_model._result.rating_means:
                ci = bt_model.get_confidence_interval(hk)
                stats.final_elo = bt_model.get_elo(hk)
            else:
                ci = elo_confidence_interval(stats)
            cis.append((stats, ci))
        cis.sort(key=lambda x: -x[0].final_elo)
        return cis

    game_cis = {gt: _cis_for(sd, bt_by_game.get(gt)) for gt, sd in stats_by_game.items()}
    global_cis = _cis_for(global_stats, bt_global)
    return game_cis, global_cis


def bucket_miners(cis):
    buckets = {"ELITE": [], "STRONG": [], "MID-TIER": [], "LOW-TIER": [], "UNCERTAIN": []}
    for stats, ci in cis:
        elo, width = stats.final_elo, ci.width
        if width > 200 and stats.total_matches < 10:
            buckets["UNCERTAIN"].append((stats, ci))
        elif elo > 1600 and width < 150:
            buckets["ELITE"].append((stats, ci))
        elif elo > 1550 or (elo > 1500 and width < 100):
            buckets["STRONG"].append((stats, ci))
        elif elo >= 1450:
            buckets["MID-TIER"].append((stats, ci))
        else:
            buckets["LOW-TIER"].append((stats, ci))
    return buckets


# --- Printing ---

def print_crystal_results(results, miners_lookup, bt_global, global_elo):
    print(f"\n{'=' * 80}\n{'ELO CRYSTALLIZATION ANALYSIS':^80}\n{'=' * 80}")

    if "GLOBAL" in results:
        r = results["GLOBAL"]
        print(f"\nGLOBAL ({r.total_matches} matches)")
        top3 = []
        for hk in r.final_top3_order:
            uid = miners_lookup.get(hk, {}).get("uid", "?")
            elo = bt_global.get_elo(hk) if bt_global._result and hk in bt_global._result.rating_means else global_elo.get(hk, 1500)
            top3.append(f"UID {uid} ({elo:.0f})")
        print(f"  Top 3: {' > '.join(top3)}")
        _print_crystal_detail(r)

    for gt in sorted(k for k in results if k != "GLOBAL"):
        r = results[gt]
        print(f"\n{gt} ({r.total_matches} matches)")
        top3 = [f"UID {miners_lookup.get(hk, {}).get('uid', '?')}" for hk in r.final_top3_order]
        print(f"  Top 3: {' > '.join(top3)}")
        _print_crystal_detail(r)


def _print_crystal_detail(r):
    def _status(val): return f"Match {val}" if val else "NOT YET"
    print(f"  CI-Based: 1st={_status(r.first_place_crystallized_match)}, Top3={_status(r.top3_crystallized_match)}")
    if r.total_matches > 0:
        pct = (r.total_matches - r.order_crystallization_match) / r.total_matches * 100
        print(f"  Position: stable for {pct:.1f}% of matches (since match {r.order_crystallization_match})")


def print_ci_table(label, cis, limit=20):
    if not cis:
        return
    print(f"\n{label} (95% Credible Interval)")
    print(f"{'Rank':>4} {'UID':>5} {'Mean':>7} {'Mode':>7} {'CI Low':>7} {'CI Hi':>7} {'Width':>6} {'Record':>10}")
    print("-" * 80)
    for i, (s, ci) in enumerate(cis[:limit], 1):
        uid = s.uid if s.uid is not None else "?"
        mode = ci.mode if ci.mode is not None else ci.point_estimate
        mean = ci.mean or ci.point_estimate
        print(f"{i:>4} {uid:>5} {mean:>7.1f} {mode:>7.1f} {ci.lower:>7.1f} {ci.upper:>7.1f} {ci.width:>6.1f} {s.wins}-{s.losses}-{s.draws:>3}")


def print_buckets(buckets):
    print(f"\n{'=' * 80}\n{'MINER TIERS':^80}\n{'=' * 80}")
    total = sum(len(b) for b in buckets.values())
    for tier in ["ELITE", "STRONG", "MID-TIER", "LOW-TIER", "UNCERTAIN"]:
        miners = buckets[tier]
        pct = len(miners) / total * 100 if total else 0
        print(f"\n{tier} ({len(miners)} miners, {pct:.0f}%)")
        for s, ci in miners[:10]:
            uid = s.uid if s.uid is not None else "?"
            print(f"  UID {uid:>3}: {s.final_elo:>7.1f} [{ci.lower:.0f}-{ci.upper:.0f}] "
                  f"{s.total_matches} matches ({s.wins}W-{s.losses}L-{s.draws}D)")
        if len(miners) > 10:
            print(f"  ... and {len(miners) - 10} more")


def run(data_path: str):
    if not HAS_NUMPYRO:
        raise ImportError("Requires NumPyro/JAX: pip install numpyro jax")

    print(f"ELO Crystallization Analysis — {datetime.now():%Y-%m-%d %H:%M:%S}")

    with open(data_path) as f:
        data = json.load(f)

    miners_lookup = {m["hotkey"]: m for m in data.get("miners", [])}
    matches = sorted(data.get("matches", []), key=lambda m: m["timestamp"])
    print(f"Loaded {len(miners_lookup)} miners, {len(matches)} matches")

    state = replay_matches(matches, miners_lookup)
    crystal = find_crystallization(
        state["top3_history"], state["global_top3_history"],
        state["ci_history_by_game"], state["global_ci_history"], matches)
    game_cis, global_cis = compute_cis(
        state["stats_by_game"], state["global_stats"],
        state["bt_by_game"], state["bt_global"])
    buckets = bucket_miners(global_cis)

    print_crystal_results(crystal, miners_lookup, state["bt_global"], state["global_elo"])
    for gt in sorted(game_cis):
        print_ci_table(gt.upper(), game_cis[gt])
    print_ci_table("GLOBAL", global_cis, limit=30)
    print_buckets(buckets)


def main():
    data_path = "elo_matches.json"
    if not Path(data_path).exists():
        parent = Path(__file__).parent.parent / "elo_matches.json"
        if parent.exists():
            data_path = str(parent)

    import sys
    if len(sys.argv) > 1:
        data_path = sys.argv[1]

    run(data_path)


if __name__ == "__main__":
    main()
