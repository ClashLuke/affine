#!/usr/bin/env python3
"""
Clean ELO matches JSON — remove forfeits, skipped games, and games with too few moves.

Usage:
    python scripts/clean_elo_matches.py [input.json] [output.json]
"""

import json
import sys
from collections import defaultdict


def is_valid_match(match):
    moves = match.get("move_history", [])
    if not moves:
        return False, "no_moves"
    for m in moves:
        if m.get("forfeit"): return False, "forfeit"
        if m.get("skipped"): return False, "skipped"
    if sum(1 for m in moves if m.get("latency_ms") is not None) < 2:
        return False, "too_few_moves"
    return True, "valid"


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "elo_matches.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "elo_matches_clean.json"

    with open(input_path) as f:
        data = json.load(f)

    matches = data.get("matches", [])
    print(f"Total matches: {len(matches)}")

    valid = []
    invalid = defaultdict(list)
    for match in matches:
        ok, reason = is_valid_match(match)
        (valid if ok else invalid[reason]).append(match)

    print(f"\n  Valid: {len(valid)}")
    for reason, lst in sorted(invalid.items(), key=lambda x: -len(x[1])):
        print(f"  Invalid ({reason}): {len(lst)}")

    # Forfeit/skip analysis
    for category, label in [("forfeit", "forfeit"), ("skipped", "skip")]:
        by_miner = defaultdict(int)
        for match in invalid.get(category, []):
            for move in match.get("move_history", []):
                if move.get(category):
                    idx = move.get("player", 0)
                    parts = match.get("participants", [])
                    if idx < len(parts):
                        by_miner[parts[idx].get("miner_hotkey", "")[:16]] += 1
        if by_miner:
            print(f"\nTop {label}s by miner:")
            for hk, count in sorted(by_miner.items(), key=lambda x: -x[1])[:10]:
                print(f"  {hk}...: {count}")

    cleaned = {
        "timestamp": data.get("timestamp"),
        "total_matches": len(valid),
        "original_total": len(matches),
        "removed": {r: len(l) for r, l in invalid.items()},
        "miners": data.get("miners", []),
        "matches": valid,
    }

    with open(output_path, "w") as f:
        json.dump(cleaned, f, indent=2, default=str)
    print(f"\nSaved {len(valid)} valid matches to {output_path}")


if __name__ == "__main__":
    main()
