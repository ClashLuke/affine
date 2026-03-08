#!/usr/bin/env python3
"""
ELO simulation — validates rating convergence without a real database or API.

Modes:
    basic        Two players, 50 games
    tournament   Round-robin with N players
    pairwise     Backward-compatible score comparison
    multiparty   4+ player ranked games
    convergence  Multi-trial convergence study with rank correlation

Usage:
    python scripts/simulate_elo.py --mode basic
    python scripts/simulate_elo.py --mode tournament --players 16 --rounds 50
    python scripts/simulate_elo.py --mode convergence --game chess --trials 5
"""

import argparse
import json
import math
import random
import sys
import os
from decimal import Decimal
from dataclasses import dataclass, field
from typing import List, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "affine", "src"))

from elo.calculator import EloCalculator
from elo.config import EloConfig


@dataclass
class SimulatedPlayer:
    id: str
    true_skill: float
    rating: Decimal = Decimal("1500")
    matches_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    rating_history: List[Tuple[int, float]] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.matches_played if self.matches_played else 0.0

    def record(self, game_num: int):
        self.rating_history.append((game_num, float(self.rating)))


def simulate_game_outcome(
    player_a: SimulatedPlayer, player_b: SimulatedPlayer,
    game_type: str = "chess", skill_noise: float = 100,
) -> str:
    perf_a = player_a.true_skill + random.gauss(0, skill_noise)
    perf_b = player_b.true_skill + random.gauss(0, skill_noise)
    diff = perf_a - perf_b

    if game_type == "tictactoe":
        draw_threshold = 50 + abs(diff) * 0.1
        if abs(diff) < draw_threshold and random.random() < 0.3:
            return "draw"
    else:
        draw_prob = 0.3 * math.exp(-abs(diff) / 200)
        if random.random() < draw_prob:
            return "draw"

    return "a_wins" if diff > 0 else "b_wins"


def update_stats(player_a, player_b, outcome):
    player_a.matches_played += 1
    player_b.matches_played += 1
    if outcome == "a_wins":
        player_a.wins += 1
        player_b.losses += 1
    elif outcome == "b_wins":
        player_b.wins += 1
        player_a.losses += 1
    else:
        player_a.draws += 1
        player_b.draws += 1


def play_game(calculator, player_a, player_b, outcome):
    new_a, new_b, delta_a, delta_b = calculator.update_ratings_head_to_head(
        rating_a=player_a.rating, rating_b=player_b.rating,
        matches_a=player_a.matches_played, matches_b=player_b.matches_played,
        outcome=outcome,
    )
    player_a.rating = new_a
    player_b.rating = new_b
    update_stats(player_a, player_b, outcome)
    return delta_a, delta_b


def print_leaderboard(players, title="Final Leaderboard"):
    players_sorted = sorted(players, key=lambda p: p.rating, reverse=True)
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    print(f"{'Rank':<5}{'Player':<15}{'ELO':>8}{'True':>8}{'Error':>8}{'Record':>12}{'Win%':>7}")
    print("-" * 70)
    for i, p in enumerate(players_sorted, 1):
        error = float(p.rating) - p.true_skill
        record = f"{p.wins}-{p.losses}-{p.draws}"
        print(f"{i:<5}{p.id:<15}{float(p.rating):>8.1f}{p.true_skill:>8.0f}{error:>+8.1f}{record:>12}{p.win_rate:>6.0%}")

    true_order = sorted(players, key=lambda p: p.true_skill, reverse=True)
    correct = sum(1 for i in range(len(players)) if players_sorted[i] == true_order[i])
    print(f"\nCorrect ranking positions: {correct}/{len(players)}")


def rank_correlation(players):
    from scipy import stats
    true_ranks = {p.id: i for i, p in enumerate(sorted(players, key=lambda x: x.true_skill, reverse=True))}
    elo_ranks = {p.id: i for i, p in enumerate(sorted(players, key=lambda x: x.rating, reverse=True))}
    corr, _ = stats.spearmanr(
        [true_ranks[p.id] for p in players],
        [elo_ranks[p.id] for p in players],
    )
    return float(corr)


# --- Modes ---

def run_basic(args):
    calc = EloCalculator(EloConfig())
    a = SimulatedPlayer(id="player_a", true_skill=1600)
    b = SimulatedPlayer(id="player_b", true_skill=1400)

    print(f"Player A: true_skill=1600, Player B: true_skill=1400")
    print(f"Running 50 games...\n")

    for _ in range(50):
        outcome = simulate_game_outcome(a, b, game_type=args.game)
        play_game(calc, a, b, outcome)

    print(f"Player A: ELO={float(a.rating):.1f}, Record={a.wins}-{a.losses}-{a.draws}")
    print(f"Player B: ELO={float(b.rating):.1f}, Record={b.wins}-{b.losses}-{b.draws}")
    diff = float(a.rating - b.rating)
    print(f"Rating diff: {diff:.1f}  (true diff: 200)")
    print(f"Convergence error: {abs(diff - 200):.1f} points")


def run_tournament(args):
    calc = EloCalculator(EloConfig())
    players = [
        SimulatedPlayer(id=f"player_{i:02d}", true_skill=1200 + i * 100)
        for i in range(args.players)
    ]

    for _ in range(args.rounds):
        shuffled = players.copy()
        random.shuffle(shuffled)
        for i in range(0, len(shuffled) - 1, 2):
            outcome = simulate_game_outcome(shuffled[i], shuffled[i + 1], args.game)
            play_game(calc, shuffled[i], shuffled[i + 1], outcome)

    print_leaderboard(players, f"Tournament: {args.players} players, {args.rounds} rounds")


def run_pairwise(args):
    calc = EloCalculator(EloConfig())
    config = EloConfig()
    miners = [
        SimulatedPlayer(id="miner_elite", true_skill=1700),
        SimulatedPlayer(id="miner_good", true_skill=1550),
        SimulatedPlayer(id="miner_avg", true_skill=1450),
        SimulatedPlayer(id="miner_weak", true_skill=1300),
    ]

    for _ in range(args.samples):
        participants = random.sample(miners, random.randint(2, len(miners)))
        scores = {
            m.id: max(0, min(1, (m.true_skill - 1000) / 1000 + random.gauss(0, 0.1)))
            for m in participants
        }
        for i in range(len(participants)):
            for j in range(i + 1, len(participants)):
                a, b = participants[i], participants[j]
                outcome = calc.score_to_outcome(
                    Decimal(str(scores[a.id])), Decimal(str(scores[b.id])), config.SCORE_MARGIN,
                )
                play_game(calc, a, b, outcome)

    print_leaderboard(miners, f"Pairwise Comparison ({args.samples} tasks)")


def run_multiparty(args):
    calc = EloCalculator(EloConfig())
    players = [
        SimulatedPlayer(id=f"player_{i}", true_skill=1300 + i * 75)
        for i in range(8)
    ]

    for _ in range(100):
        game_players = random.sample(players, 4)
        rankings = sorted(game_players, key=lambda p: p.true_skill + random.gauss(0, 100), reverse=True)
        participants = [
            {"hotkey": p.id, "rating": p.rating, "matches_played": p.matches_played, "final_rank": rank}
            for rank, p in enumerate(rankings, 1)
        ]
        updates = calc.update_ratings_multi_party(participants)
        for i, player in enumerate(rankings):
            new_rating, _ = updates[i]
            player.rating = new_rating
            player.matches_played += 1
            if participants[i]["final_rank"] == 1:
                player.wins += 1
            elif participants[i]["final_rank"] == 4:
                player.losses += 1

    print_leaderboard(players, "Multi-Party (4-player games)")


def run_convergence(args):
    print(f"{'=' * 70}")
    print(f"CONVERGENCE STUDY: {args.game.upper()}")
    print(f"{args.trials} trials, {args.players} players, {args.games} games each")
    print(f"{'=' * 70}")

    all_final_corr = []
    all_converge_at = []

    for trial in range(1, args.trials + 1):
        random.seed(args.seed + trial)
        calc = EloCalculator(EloConfig())

        players = [
            SimulatedPlayer(id=f"player_{i:02d}", true_skill=1200 + i * (600 / max(args.players - 1, 1)))
            for i in range(args.players)
        ]
        random.shuffle(players)

        recent_changes = []
        metrics = []

        for game_num in range(1, args.games + 1):
            if args.matchmaking == "elo_balanced":
                by_elo = sorted(players, key=lambda p: p.rating)
                idx = random.randint(0, len(by_elo) - 2)
                a, b = by_elo[idx], by_elo[idx + 1]
            else:
                a, b = random.sample(players, 2)

            outcome = simulate_game_outcome(a, b, args.game, args.skill_noise)
            da, _ = play_game(calc, a, b, outcome)
            a.record(game_num)
            b.record(game_num)

            recent_changes.append(abs(float(da)))
            if len(recent_changes) > 100:
                recent_changes = recent_changes[-100:]

            if game_num % 50 == 0:
                corr = rank_correlation(players)
                errors = [abs(float(p.rating) - p.true_skill) for p in players]
                top_k = min(5, args.players // 2)
                true_top = set(p.id for p in sorted(players, key=lambda x: x.true_skill, reverse=True)[:top_k])
                elo_top = set(p.id for p in sorted(players, key=lambda x: x.rating, reverse=True)[:top_k])
                metrics.append((game_num, corr, sum(errors) / len(errors), len(true_top & elo_top)))

        final_corr = metrics[-1][1] if metrics else 0
        all_final_corr.append(final_corr)

        converge_at = next((m[0] for m in metrics if m[1] >= 0.9), None)
        all_converge_at.append(converge_at)

        if args.trials <= 5 or trial <= 3:
            print(f"Trial {trial}: corr={final_corr:.3f}, converged_at={converge_at or 'N/A'}")

        if trial == 1:
            print_leaderboard(players, f"Trial 1 Final Standings")

    print(f"\n{'-' * 70}\nSUMMARY:")
    print(f"  Mean final correlation: {sum(all_final_corr) / len(all_final_corr):.4f}")
    print(f"  Range: [{min(all_final_corr):.4f}, {max(all_final_corr):.4f}]")
    valid = [c for c in all_converge_at if c is not None]
    if valid:
        print(f"  Mean games to 90% corr: {sum(valid) / len(valid):.0f}")
        print(f"  Convergence rate: {len(valid)}/{args.trials} trials")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "game_type": args.game, "trials": args.trials,
                "final_correlations": all_final_corr,
                "convergence_points": all_converge_at,
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="ELO System Simulation")
    parser.add_argument("--mode", choices=["basic", "tournament", "pairwise", "multiparty", "convergence", "all"],
                        default="basic")
    parser.add_argument("--game", choices=["chess", "tictactoe"], default="chess")
    parser.add_argument("--players", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--matchmaking", choices=["random", "elo_balanced"], default="random")
    parser.add_argument("--skill-noise", type=float, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    modes = {"basic": run_basic, "tournament": run_tournament, "pairwise": run_pairwise,
             "multiparty": run_multiparty, "convergence": run_convergence}

    if args.mode == "all":
        for name, fn in modes.items():
            if name != "convergence":
                fn(args)
                print("\n")
    else:
        modes[args.mode](args)


if __name__ == "__main__":
    main()
