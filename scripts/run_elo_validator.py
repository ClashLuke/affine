#!/usr/bin/env python3
"""
ELO Validator CLI — query ratings from API or replay from match history.

Usage:
    python scripts/run_elo_validator.py
    python scripts/run_elo_validator.py --replay --since 2024-01-01
    python scripts/run_elo_validator.py --local --output-format json -o ratings.json
"""

import argparse
import os
import sys
from datetime import datetime


def parse_date(s):
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return int(datetime.strptime(s, fmt).timestamp() * 1000)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"Invalid date: {s}")


def parse_args():
    p = argparse.ArgumentParser(description="ELO Validator")
    p.add_argument("--env", nargs="+", default=["game:chess", "game:tictactoe"])
    p.add_argument("--replay", action="store_true")
    p.add_argument("--since", type=str, default=None)
    p.add_argument("--until", type=str, default=None)
    p.add_argument("--k-factor", type=int, default=None)
    p.add_argument("--k-factor-new", type=int, default=None)
    p.add_argument("--min-matches", type=int, default=5)
    p.add_argument("--top-k", type=int, default=32)
    p.add_argument("--output-format", choices=["console", "json", "csv"], default="console")
    p.add_argument("--output", "-o", type=str, default=None)
    p.add_argument("--api-url", type=str, default=None)
    p.add_argument("--local", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


_ARGS = parse_args()

import asyncio
import json
import random
from decimal import Decimal

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from affine.core.setup import logger, setup_logging
from affine.src.elo.config import EloConfig
from affine.src.elo.models import EloRating
from affine.src.integration.elo_validator import EloValidatorEngine

setup_logging(verbosity=1)


def _make_elo_config(args):
    config = EloConfig()
    if args.k_factor:
        config.K_FACTOR = args.k_factor
    if args.k_factor_new:
        config.K_FACTOR_NEW_PLAYER = args.k_factor_new
    return config


def _write_output(report, args):
    if args.output_format == "console":
        report.print_console(top_k=args.top_k)
    else:
        output = report.to_json() if args.output_format == "json" else report.to_csv()
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            logger.info(f"Written to {args.output}")
        else:
            print(output)


async def run_validation(args):
    config = _make_elo_config(args)
    validator = EloValidatorEngine(
        environments=args.env, elo_config=config,
        api_url=args.api_url, local_mode=args.local,
    )

    if args.local:
        logger.info("Generating sample ELO data for local testing...")
        hotkeys = [
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy",
            "5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcCj68kUMaw",
            "5CiPPseXPECbkjWCa6MnjNokrgYjMqmKndv2rSnekmSK2DjL",
            "5GNJqTPyNqANBkUVMN1LPPrxXnFouWXoe2wNSmmEoLctxiZY",
            "5HpG9w8EBLe5XCrbczpwq5TSXvedjrBGCwqxK1iQ7qUsSWFc",
            "5Ck5SLSHYac6WFt5UZRSsdJjwmpSZq85fd5TRNAdZQVzEAPT",
        ]
        for env in args.env:
            for i, hk in enumerate(hotkeys):
                elo = 1500 + (len(hotkeys) - i) * 50 + random.randint(-30, 30)
                matches = random.randint(20, 100)
                wins = int(matches * (0.4 + (elo - 1400) / 1000))
                validator.add_rating(EloRating(
                    miner_hotkey=hk, model_revision="v1", env=env,
                    rating=Decimal(str(elo)), peak_rating=Decimal(str(elo + random.randint(0, 50))),
                    matches_played=matches, wins=wins, losses=matches - wins, draws=0,
                ))

    if not args.quiet:
        logger.info(f"ELO Validator: envs={args.env}, mode={'REPLAY' if args.replay else 'CURRENT'}")

    report = await validator.validate(
        replay=args.replay,
        since_timestamp=parse_date(args.since) if args.since else None,
        until_timestamp=parse_date(args.until) if args.until else None,
        min_matches=args.min_matches,
    )
    _write_output(report, args)
    return report


def main():
    args = _ARGS
    try:
        asyncio.run(run_validation(args))
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
