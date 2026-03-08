#!/usr/bin/env python3
"""Animated ELO racing bar chart — two-column leaderboard video."""

import json
import argparse
import colorsys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation

from _replay import replay, pick_primary_hotkeys

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'


def distinct_colors(n):
    colors = []
    golden = 0.618033988749895
    h = 0.1
    for i in range(n):
        h = (h + golden) % 1.0
        r, g, b = colorsys.hls_to_rgb(h, 0.5 + (i % 3) * 0.1, 0.7 + (i % 2) * 0.15)
        colors.append(mcolors.to_hex([r, g, b]))
    return colors


def extract_snapshots(data):
    state = replay(data, per_env=False)
    uid_primary, uid_display_raw = pick_primary_hotkeys(state)
    uid_display = {uid: f"{name[:8]}#{uid}" for uid, name in uid_display_raw.items()}

    all_events = []
    global_hist = state["history"].get("global", {})
    for hk, uid in state["hk_to_uid"].items():
        for match_idx, rating in global_hist.get(hk, []):
            all_events.append((match_idx, uid, rating))
    all_events.sort(key=lambda x: x[0])

    current = {uid: 1500.0 for uid in uid_primary}
    snapshots = [dict(current)]

    # Group events by match_idx
    from itertools import groupby
    for match_idx, group in groupby(all_events, key=lambda x: x[0]):
        for _, uid, rating in group:
            current[uid] = rating
        snapshots.append(dict(current))

    return snapshots, uid_display


def create_video(data, output_path, fps=30, target_duration=40):
    snapshots, uid_display = extract_snapshots(data)
    num_games = len(snapshots) - 1
    total_miners = len(snapshots[0])

    interval = max(1, num_games // (fps * target_duration))
    frames = list(range(0, len(snapshots), interval))
    if frames[-1] != len(snapshots) - 1:
        frames.append(len(snapshots) - 1)

    all_uids = sorted(snapshots[0])
    uid_colors = dict(zip(all_uids, distinct_colors(len(all_uids))))
    miners_per_col = (total_miners + 1) // 2

    fig = plt.figure(figsize=(16, 9), facecolor='#fafbfc')
    ax1 = fig.add_axes([0.10, 0.10, 0.36, 0.78])
    ax2 = fig.add_axes([0.58, 0.10, 0.36, 0.78])
    title = fig.text(0.5, 0.955, 'ELO Leaderboard Race', fontsize=26, fontweight='bold', ha='center', color='#111827')
    subtitle = fig.text(0.5, 0.91, '', fontsize=12, ha='center', color='#6b7280')
    game_text = fig.text(0.94, 0.94, '', fontsize=18, fontweight='bold', ha='right', va='top', color='#1f2937',
                         bbox=dict(boxstyle='round,pad=0.4', facecolor='#e0e7ff', edgecolor='#a5b4fc', linewidth=1.5))

    def draw_col(ax, miners, rank_start=1):
        ax.clear()
        ax.set_facecolor('#fafbfc')
        if not miners:
            return
        uids = [m[0] for m in miners]
        vals = [m[1] for m in miners]
        names = [f"{rank_start + i:2d}. {uid_display[u]}" for i, u in enumerate(uids)]
        colors = [uid_colors[u] for u in uids]
        y = np.arange(len(names))
        bars = ax.barh(y, vals, color=colors, height=0.8, edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v - 1200 > 200:
                ax.text(v - 15, bar.get_y() + bar.get_height() / 2, f'{v:.0f}',
                        va='center', ha='right', fontsize=8, fontweight='bold', color='white')
            else:
                ax.text(v + 10, bar.get_y() + bar.get_height() / 2, f'{v:.0f}',
                        va='center', ha='left', fontsize=8, fontweight='bold', color='#374151')
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9, fontweight='500', color='#1e293b')
        ax.invert_yaxis()
        ax.set_xlim(1200, 1750)
        ax.set_xlabel('ELO Rating', fontsize=10, color='#475569')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axvline(x=1500, color='#6b7280', linestyle='--', linewidth=1.8, zorder=0, alpha=0.8)
        ax.xaxis.grid(True, linestyle='-', alpha=0.25, color='#d1d5db')

    def animate(fi):
        si = frames[fi]
        elos = snapshots[si]
        sorted_m = sorted(elos.items(), key=lambda x: x[1], reverse=True)
        draw_col(ax1, sorted_m[:miners_per_col], 1)
        draw_col(ax2, sorted_m[miners_per_col:], miners_per_col + 1)
        subtitle.set_text(f'{total_miners} miners  |  {si:,} games  |  Start: 1500')
        game_text.set_text(f'Game {si:,}')
        if fi % 100 == 0:
            print(f"  Frame {fi}/{len(frames)} (game {si})")
        return [ax1, ax2]

    print(f"Rendering {len(frames)} frames...")
    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=1000 / fps, blit=False)
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=4000, codec='libx264',
                                        extra_args=['-preset', 'fast', '-crf', '22'])
        anim.save(output_path, writer=writer, dpi=100)
    except Exception as e:
        print(f"FFMpeg failed ({e}), trying pillow...")
        gif_path = output_path.replace('.mp4', '.gif')
        anim.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=80)
        print(f"Saved as GIF: {gif_path}")

    print(f"Done: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ELO racing bar chart video")
    parser.add_argument("--input", "-i", default="overnight_elo_matches.json")
    parser.add_argument("--output", "-o", default="elo_race.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=int, default=40)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    print(f"Loaded {len(data['matches'])} matches, {len(data.get('miners', []))} miners")
    create_video(data, args.output, fps=args.fps, target_duration=args.duration)
