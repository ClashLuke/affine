#!/usr/bin/env python3
"""
ELO visualization — multiple styles from the same match data.

Styles:
    lines      Top 5 progression with smoothed lines and percentile band
    heatmap    All miners' rating history as a color matrix
    rankings   Final rankings with Bayesian 95% credible intervals

Usage:
    python scripts/visualize_elo.py --style lines
    python scripts/visualize_elo.py --style heatmap -i overnight_elo_matches.json
    python scripts/visualize_elo.py --style rankings --game chess
"""

import json
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter1d

from _replay import replay, pick_primary_hotkeys


def load_data(path):
    with open(path) as f:
        return json.load(f)


def extract_elo_progression(data):
    """Replay matches into one global rating per miner and return per-uid history."""
    state = replay(data, per_env=False)
    uid_primary, uid_display = pick_primary_hotkeys(state)

    history = defaultdict(list)
    global_hist = state["history"].get("global", {})
    for hk, uid in state["hk_to_uid"].items():
        if hk in global_hist:
            history[uid].extend(global_hist[hk])
    for uid in history:
        history[uid].sort(key=lambda x: x[0])
        if not history[uid] or history[uid][0][0] != 0:
            history[uid].insert(0, (0, 1500.0))

    return dict(history), uid_display


# --- Style: lines ---

def plot_lines(data, output_path):
    history, uid_display = extract_elo_progression(data)
    final_elos = {uid: h[-1][1] for uid, h in history.items() if len(h) > 1}
    sorted_uids = sorted(final_elos, key=lambda x: final_elos[x], reverse=True)
    num_games = len(data['matches'])

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # Percentile band
    by_game = defaultdict(list)
    for uid, h in history.items():
        for g, e in h:
            by_game[g].append(e)
    games = np.array(sorted(by_game))
    p25 = gaussian_filter1d([np.percentile(by_game[g], 25) for g in games], 60)
    p75 = gaussian_filter1d([np.percentile(by_game[g], 75) for g in games], 60)
    ax.fill_between(games, p25, p75, alpha=0.15, color='#58a6ff', linewidth=0, label='Middle 50%')

    colors = ['#3fb950', '#f0883e', '#a371f7', '#f778ba', '#79c0ff']
    widths = [3.0, 2.0, 1.8, 1.6, 1.4]

    for i, uid in enumerate(sorted_uids[:5]):
        h = history[uid]
        g = np.array([x[0] for x in h])
        e = np.array([x[1] for x in h])
        e_smooth = gaussian_filter1d(e, sigma=10) if len(e) > 10 else e
        model = uid_display.get(uid, 'unknown')
        ax.plot(g, e_smooth, color=colors[i], linewidth=widths[i], alpha=0.9,
                label=model, solid_capstyle='round')
        ax.annotate(f'{final_elos[uid]:.0f}', xy=(g[-1], e_smooth[-1]), xytext=(8, 0),
                    textcoords='offset points', fontsize=9, color=colors[i], fontweight='500', va='center')

    mn, mx = min(final_elos.values()), max(final_elos.values())
    ax.set_xlim(0, num_games + 80)
    ax.set_ylim(mn - 80, mx + 60)
    ax.set_xlabel('Games Played', fontsize=11, color='#8b949e', labelpad=10)
    ax.set_ylabel('ELO Rating', fontsize=11, color='#8b949e', labelpad=10)
    ax.set_title('ELO Rating Progression', fontsize=20, color='#f0f6fc', fontweight='600', pad=25, loc='left')
    ax.text(0.0, 1.025, f'{len(final_elos)} miners  ·  {num_games:,} games  ·  range {mn:.0f}–{mx:.0f}',
            transform=ax.transAxes, fontsize=12, color='#9198a1', va='bottom')
    ax.grid(True, alpha=0.12, linestyle='-', color='#30363d')
    ax.axhline(y=1500, color='#484f58', linestyle='--', alpha=0.5, linewidth=1)
    ax.legend(loc='upper left', fontsize=10, frameon=False, labelcolor='#c9d1d9',
              ncol=3, handlelength=1.8, columnspacing=1.2, bbox_to_anchor=(0, 0.96))
    for spine in ax.spines.values():
        spine.set_color('#30363d')
        spine.set_linewidth(0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='#8b949e', labelsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#0d1117', edgecolor='none', bbox_inches='tight', pad_inches=0.3)
    plt.close()


# --- Style: heatmap ---

def plot_heatmap(data, output_path):
    history, uid_display = extract_elo_progression(data)
    active = {uid: h for uid, h in history.items() if len(h) > 1}
    final_elos = {uid: h[-1][1] for uid, h in active.items()}
    sorted_uids = sorted(final_elos, key=lambda x: final_elos[x], reverse=True)

    num_games = len(data['matches'])
    num_miners = len(sorted_uids)
    num_cols = min(500, num_games)
    sample_pts = np.linspace(0, num_games, num_cols).astype(int)

    matrix = np.full((num_miners, num_cols), 1500.0)
    for row, uid in enumerate(sorted_uids):
        game_to_elo = dict(active[uid])
        cur = 1500.0
        series = []
        for g in range(num_games + 1):
            if g in game_to_elo:
                cur = game_to_elo[g]
            series.append(cur)
        for col, sg in enumerate(sample_pts):
            if sg < len(series):
                matrix[row, col] = series[sg]

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    cmap = mcolors.LinearSegmentedColormap.from_list('elo', ['#d73a49', '#6e7681', '#3fb950'], N=256)
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=1200, vmax=1750, interpolation='bilinear')

    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label('ELO Rating', fontsize=11, color='#c9d1d9')
    cbar.ax.yaxis.set_tick_params(color='#8b949e')
    cbar.outline.set_edgecolor('#30363d')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#8b949e')

    x_ticks = np.linspace(0, num_cols - 1, 5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{int(num_games * t / (num_cols - 1))}' for t in x_ticks])
    ax.set_xlabel('Games Played', fontsize=11, color='#8b949e', labelpad=10)

    y_idx = list(range(0, num_miners, 5))
    ax.set_yticks(y_idx)
    ax.set_yticklabels([
        f'{uid_display.get(sorted_uids[i], f"UID {sorted_uids[i]}")[:12]} ({final_elos[sorted_uids[i]]:.0f})'
        for i in y_idx
    ], fontsize=8)
    ax.set_ylabel('Miners (sorted by final ELO)', fontsize=11, color='#8b949e', labelpad=10)

    ax.set_title('Complete ELO History — All Miners', fontsize=18, color='#f0f6fc', fontweight='600', pad=15, loc='left')
    ax.text(0.0, 1.02, f'{num_miners} miners  ·  {num_games:,} games  ·  range {min(final_elos.values()):.0f}–{max(final_elos.values()):.0f}',
            transform=ax.transAxes, fontsize=11, color='#9198a1', va='bottom')
    ax.tick_params(colors='#8b949e', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#30363d')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#0d1117', edgecolor='none', bbox_inches='tight', pad_inches=0.3)
    plt.close()


# --- Style: rankings (Bayesian CI forest plot) ---

def plot_rankings(data, output_path, game_type=None, top_n=20):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.analyze_elo_crystallization import BayesianBradleyTerryWrapper, HAS_NUMPYRO

    if not HAS_NUMPYRO:
        print("Error: NumPyro required for rankings style. Install: pip install numpyro jax")
        return

    miners = {m["hotkey"]: m for m in data.get("miners", [])}
    matches = sorted(data.get("matches", []), key=lambda m: m["timestamp"])
    if game_type:
        matches = [m for m in matches if m["game_type"] == game_type]

    if not matches:
        print("No matches found")
        return

    stats = defaultdict(lambda: {"wins": 0, "losses": 0, "draws": 0})
    for match in matches:
        if len(match["participants"]) != 2:
            continue
        for p in match["participants"]:
            hk = p["miner_hotkey"]
            if p["outcome"] == "win": stats[hk]["wins"] += 1
            elif p["outcome"] == "loss": stats[hk]["losses"] += 1
            else: stats[hk]["draws"] += 1

    print("Fitting Bayesian Bradley-Terry model...")
    model = BayesianBradleyTerryWrapper()
    for match in matches:
        if len(match["participants"]) != 2:
            continue
        p1, p2 = match["participants"]
        model.add_match(p1["miner_hotkey"], p2["miner_hotkey"], p1["outcome"])
    model.fit(num_samples=2000, num_warmup=1000)

    rankings = []
    for hk, ci in model.get_all_confidence_intervals().items():
        s = stats.get(hk, {"wins": 0, "losses": 0, "draws": 0})
        rankings.append({
            "hotkey": hk, "uid": miners.get(hk, {}).get("uid"),
            "mean": ci.mean or ci.point_estimate, "mode": ci.mode or ci.point_estimate,
            "ci_lo": ci.lower, "ci_hi": ci.upper, "width": ci.width,
            "wins": s["wins"], "losses": s["losses"], "draws": s["draws"],
        })
    rankings.sort(key=lambda x: -x["mean"])
    rankings = rankings[:top_n]

    scope = game_type or "GLOBAL"
    print(f"\n{scope}: {len(matches)} matches")
    print(f"{'Rank':>4} {'UID':>5} {'Mean':>7} {'Mode':>7} {'CI':>15} {'Width':>6} {'Record':>12}")
    print("-" * 65)
    for i, r in enumerate(rankings, 1):
        uid = r["uid"] if r["uid"] is not None else "?"
        print(f"{i:>4} {uid:>5} {r['mean']:>7.1f} {r['mode']:>7.1f} "
              f"[{r['ci_lo']:>6.1f},{r['ci_hi']:>6.1f}] {r['width']:>6.1f} "
              f"{r['wins']}W-{r['losses']}L-{r['draws']}D")

    # Forest plot
    n = len(rankings)
    ys = np.arange(n)
    means = [r["mean"] for r in rankings]
    xerr_lo = [r["mean"] - r["ci_lo"] for r in rankings]
    xerr_hi = [r["ci_hi"] - r["mean"] for r in rankings]

    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
    fig.suptitle(f"ELO Rankings — {scope}\n{len(matches)} matches", fontsize=14, fontweight="bold", y=0.98)

    ax.errorbar(means, ys, xerr=[xerr_lo, xerr_hi], fmt="none",
                ecolor="steelblue", elinewidth=2, capsize=4, capthick=1.5)
    ax.scatter(means, ys, color="steelblue", s=60, zorder=5, label="Mean")
    ax.scatter([r["mode"] for r in rankings], ys, color="coral", s=40, marker="D", zorder=5, label="Mode")

    labels = [f"UID {r['uid'] or '?'} ({r['wins']}W-{r['losses']}L-{r['draws']}D)" for r in rankings]
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("ELO Rating")
    ax.set_title("Final Rankings with 95% Credible Intervals", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    ax.axvline(x=1500, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="ELO visualization")
    parser.add_argument("--style", "-s", choices=["lines", "heatmap", "rankings"], default="lines")
    parser.add_argument("--input", "-i", default="overnight_elo_matches.json")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--game", "-g", type=str, default=None, help="Filter by game type (rankings only)")
    parser.add_argument("--top", "-n", type=int, default=20, help="Top N players (rankings only)")
    args = parser.parse_args()

    if args.output is None:
        suffix = {"lines": "progression", "heatmap": "heatmap", "rankings": f"rankings_{args.game or 'global'}"}
        args.output = f"elo_{suffix[args.style]}.png"

    data = load_data(args.input)
    print(f"Loaded {len(data['matches'])} matches, {len(data.get('miners', []))} miners")

    if args.style == "lines":
        plot_lines(data, args.output)
    elif args.style == "heatmap":
        plot_heatmap(data, args.output)
    elif args.style == "rankings":
        plot_rankings(data, args.output, game_type=args.game, top_n=args.top)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
