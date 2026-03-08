#!/usr/bin/env python3
"""
ELO status dashboard — check overnight run progress from matches JSON or log file.

Usage:
    python scripts/elo_status.py
    python scripts/elo_status.py --log overnight_run.log
    python scripts/elo_status.py --watch
"""

import json
import os
import re
from datetime import datetime
from collections import defaultdict

from _replay import replay


def format_duration(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h: return f"{h}h {m}m {s}s"
    if m: return f"{m}m {s}s"
    return f"{s}s"


def print_rankings(sorted_stats, title="TOP 10", header_fmt=None, n=10, start=0):
    stats_slice = sorted_stats[start:start + n] if start else sorted_stats[:n]
    print(f"\n{'-' * 70}\n{title:^70}\n{'-' * 70}")
    if header_fmt:
        print(header_fmt)
        print("-" * 70)
    for i, (key, s) in enumerate(stats_slice, start + 1):
        total = s['wins'] + s['losses'] + s.get('draws', 0)
        win_pct = s['wins'] / total * 100 if total else 0
        record = f"{s['wins']}-{s['losses']}-{s.get('draws', 0)}" if 'draws' in s else f"{s['wins']}-{s['losses']}"
        print(f"{i:<5} {str(key):<18} {s['elo']:<8.1f} {record:<12} {win_pct:<8.1f} {s.get('games', total):<6}")


def print_elo_distribution(stats):
    elos = [s['elo'] for s in stats.values()]
    print(f"\n{'-' * 70}\n{'ELO DISTRIBUTION':^70}\n{'-' * 70}")
    print(f"  Highest: {max(elos):.1f}")
    print(f"  Lowest:  {min(elos):.1f}")
    print(f"  Average: {sum(elos) / len(elos):.1f}")
    print(f"  Spread:  {max(elos) - min(elos):.1f}")


def analyze_matches(matches_file):
    if not os.path.exists(matches_file):
        print(f"No match file: {matches_file}")
        return

    with open(matches_file) as f:
        data = json.load(f)

    matches = sorted(data.get('matches', []), key=lambda m: m.get('timestamp', 0))
    miners = data.get('miners', [])
    if not matches:
        print("No matches played yet")
        return

    # Global replay for stats (one rating per miner across all envs)
    global_state = replay(data, per_env=False)
    # Per-env replay for cross-env average ELO (weight formula)
    env_state = replay(data, per_env=True)
    hk_to_uid = global_state["hk_to_uid"]

    stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0, 'elo': 1500, 'games': 0})

    # Aggregate stats across all hotkeys per UID (handles key rotations)
    global_ratings = global_state["ratings"].get("global", {})
    for hk, r in global_ratings.items():
        uid = hk_to_uid.get(hk)
        if uid is None:
            continue
        stats[uid]['wins'] += r['wins']
        stats[uid]['losses'] += r['losses']
        stats[uid]['draws'] += r['draws']
        stats[uid]['games'] += r['matches']

    # Cross-env average ELO: for each UID, collect best hotkey's per-env ratings
    # (a UID with rotated keys should use the active key's ratings per env)
    uid_env_elos = defaultdict(dict)  # uid -> {env: best_rating}
    for hk, uid in hk_to_uid.items():
        for env, env_ratings in env_state["ratings"].items():
            r = env_ratings.get(hk)
            if r and r["matches"] > 0:
                cur = uid_env_elos[uid].get(env, (0, 0))
                if r["matches"] > cur[1]:
                    uid_env_elos[uid][env] = (float(r["rating"]), r["matches"])

    for uid, env_data in uid_env_elos.items():
        elos = [rating for rating, _ in env_data.values()]
        if elos:
            stats[uid]['elo'] = sum(elos) / len(elos)

    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['elo'], reverse=True)

    first_ts = matches[0].get('timestamp', 0) / 1000
    last_ts = matches[-1].get('timestamp', 0) / 1000
    duration = last_ts - first_ts
    gpm = len(matches) / (duration / 60) if duration > 0 else 0

    print(f"\n{'=' * 70}\n{'ELO VALIDATOR STATUS DASHBOARD':^70}\n{'=' * 70}")
    print(f"Timestamp: {data.get('timestamp', 'N/A')}")
    print(f"Games: {len(matches)}  Miners: {len(stats)}  Duration: {format_duration(duration)}  Speed: {gpm:.1f}/min")

    hdr = f"{'Rank':<5} {'UID':<18} {'ELO':<8} {'W-L-D':<12} {'Win%':<8} {'Games':<6}"
    print_rankings(sorted_stats, "TOP 10", hdr, 10)
    if len(sorted_stats) > 10:
        print(f"\n... and {len(sorted_stats) - 10} more miners")
    print_rankings(sorted_stats, "BOTTOM 5", hdr, 5, max(0, len(sorted_stats) - 5))
    print_elo_distribution(stats)
    print(f"\n{'=' * 70}")


def analyze_log(log_file):
    if not os.path.exists(log_file):
        print(f"No log file: {log_file}")
        return

    elo_pat = re.compile(r'ELO: (\S+)\.\.\. (\d+) -> (\d+) \(([+-]\d+)\)')
    game_pat = re.compile(r'Game (\d+):')
    ts_pat = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')

    stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'elo': 1500, 'games': 0})
    game_count = rate_limits = 0
    first_ts = last_ts = None

    with open(log_file) as f:
        for line in f:
            ts_m = ts_pat.match(line)
            if ts_m:
                if first_ts is None: first_ts = ts_m.group(1)
                last_ts = ts_m.group(1)
            gm = game_pat.search(line)
            if gm: game_count = int(gm.group(1))
            em = elo_pat.search(line)
            if em:
                hk = em.group(1)
                stats[hk]['elo'] = int(em.group(3))
                stats[hk]['games'] += 1
                if int(em.group(4)) > 0: stats[hk]['wins'] += 1
                else: stats[hk]['losses'] += 1
            if 'RATE LIMIT HIT' in line: rate_limits += 1

    if not stats:
        print("No ELO data in log")
        return

    duration_str = "N/A"
    gpm = 0
    if first_ts and last_ts:
        try:
            d = (datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S") - datetime.strptime(first_ts, "%Y-%m-%d %H:%M:%S")).total_seconds()
            duration_str = format_duration(d)
            if d > 0: gpm = game_count / (d / 60)
        except ValueError: pass

    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['elo'], reverse=True)

    print(f"\n{'=' * 70}\n{'ELO STATUS (from log)':^70}\n{'=' * 70}")
    print(f"Start: {first_ts or 'N/A'}  Last: {last_ts or 'N/A'}  Duration: {duration_str}")
    print(f"Games: {game_count}  Rate limits: {rate_limits}  Miners: {len(stats)}  Speed: {gpm:.1f}/min")

    hdr = f"{'Rank':<5} {'Hotkey':<18} {'ELO':<8} {'W-L':<12} {'Win%':<8} {'Games':<6}"
    print_rankings(sorted_stats, "TOP 10", hdr, 10)
    if len(sorted_stats) > 10:
        print(f"\n... and {len(sorted_stats) - 10} more miners")
    print_rankings(sorted_stats, "BOTTOM 5", hdr, 5, max(0, len(sorted_stats) - 5))
    print_elo_distribution(stats)
    print(f"\n{'=' * 70}")


def check_process():
    import subprocess
    result = subprocess.run(['pgrep', '-f', 'run_local_elo_validator'], capture_output=True, text=True)
    pids = [p for p in result.stdout.strip().split('\n') if p]
    if pids:
        print(f"\n[RUNNING] Validator active (PID: {', '.join(pids)})")
    else:
        print("\n[STOPPED] No validator process found")
    return bool(pids)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check ELO validator status")
    parser.add_argument("--matches", "-m", default="overnight_elo_matches.json")
    parser.add_argument("--log", "-l", default=None)
    parser.add_argument("--watch", "-w", action="store_true")
    args = parser.parse_args()

    def run():
        if args.log:
            analyze_log(args.log)
        elif os.path.exists(args.matches):
            analyze_matches(args.matches)
        else:
            default_log = "overnight_run.log"
            if os.path.exists(default_log):
                analyze_log(default_log)
            else:
                print(f"No data found. Use --matches or --log")

    if args.watch:
        import time
        try:
            while True:
                os.system('clear')
                check_process()
                run()
                print("\n[Ctrl+C to exit, refreshing in 30s...]")
                time.sleep(30)
        except KeyboardInterrupt:
            pass
    else:
        check_process()
        run()
