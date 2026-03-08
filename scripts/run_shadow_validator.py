#!/usr/bin/env python3
"""
Shadow Validator — compares ELO vs absolute scoring in parallel.
Does NOT set weights on chain.

Usage:
    python scripts/run_shadow_validator.py
    python scripts/run_shadow_validator.py --single
    python scripts/run_shadow_validator.py --rounds 10 --interval 300
"""

import asyncio
import argparse
import json
import os
import signal
import sys
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from affine.core.setup import logger
from affine.src.integration import ShadowValidator

shutdown_requested = False


def signal_handler(signum, frame):
    global shutdown_requested
    logger.info("Shutdown signal received, completing current round...")
    shutdown_requested = True


async def run_validation(args):
    global shutdown_requested
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    os.makedirs(args.output_dir, exist_ok=True)

    validator = ShadowValidator(
        netuid=args.netuid, dry_run=True,
        save_reports=True, report_dir=args.output_dir,
    )
    if args.k_factor:
        validator.elo_config.K_FACTOR_NEW_PLAYER = args.k_factor

    await validator.initialize()

    mode = "single" if args.single else f"{args.rounds or 'unlimited'} rounds"
    logger.info(f"Shadow Validator: netuid={args.netuid}, mode={mode}, interval={args.interval}s")

    rounds = 0
    max_rounds = 1 if args.single else args.rounds

    while not shutdown_requested:
        if max_rounds and rounds >= max_rounds:
            break

        try:
            report = await validator.run_validation_round()
            if report:
                rounds += 1
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(args.output_dir, f"validation_report_{ts}.json")
                with open(path, "w") as f:
                    json.dump(report.to_dict(), f, indent=2)

                logger.info(f"[Round {rounds}] miners={len(report.absolute_weights)} "
                            f"matches={report.matches_generated} corr={report.rank_correlation:.4f} "
                            f"top32_overlap={report.top_k_overlap} rmse={report.weight_rmse:.6f}")

                if report.rank_correlation < 0.7:
                    logger.warning("LOW CORRELATION: ELO rankings diverge from absolute")
            else:
                logger.warning("No data for validation")

        except Exception as e:
            logger.error(f"[Round {rounds + 1}] Error: {e}", exc_info=True)

        if not args.single and not shutdown_requested:
            if max_rounds is None or rounds < max_rounds:
                await asyncio.sleep(args.interval)

    logger.info(f"Done: {rounds} rounds, {len(validator.elo_state.ratings)} ratings")

    if validator.elo_state.ratings:
        path = os.path.join(args.output_dir, "final_elo_state.json")
        with open(path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "rounds": rounds,
                "matches": len(validator.elo_state.matches),
                "ratings": {
                    k: {"rating": float(v.rating), "matches": v.matches_played,
                         "wins": v.wins, "losses": v.losses, "draws": v.draws}
                    for k, v in validator.elo_state.ratings.items()
                },
            }, f, indent=2)
        logger.info(f"State saved to {path}")


def main():
    p = argparse.ArgumentParser(description="ELO Shadow Validator")
    p.add_argument("--netuid", type=int, default=120)
    p.add_argument("--single", action="store_true")
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--interval", type=int, default=300)
    p.add_argument("--output-dir", type=str, default="./validation_reports")
    p.add_argument("--k-factor", type=int, default=None)
    args = p.parse_args()
    asyncio.run(run_validation(args))


if __name__ == "__main__":
    main()
