#!/usr/bin/env python3
"""
Run game evaluations using affinetes Docker containers.

Usage:
    source .env && python scripts/run_affinetes_eval.py --miners /tmp/hot_miners.json
"""

import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Run game evaluations via affinetes")
    parser.add_argument("--miners", type=str, required=True, help="Path to miners JSON")
    parser.add_argument("--max-miners", type=int, default=5)
    parser.add_argument("--games", type=int, default=3, help="Games per pair")
    return parser.parse_args()

_ARGS = parse_args()

import asyncio
import json
import time
from dataclasses import dataclass

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from affine.core.setup import logger, setup_logging
from affine.core.environments import SDKEnvironment

setup_logging(verbosity=1)


@dataclass
class Miner:
    """Shaped for SDKEnvironment.evaluate() interface."""
    uid: int
    hotkey: str
    model: str
    slug: str
    revision: str = "v1"


async def run_evaluation():
    args = _ARGS

    with open(args.miners, "r") as f:
        miner_data = json.load(f)

    miners = [
        Miner(uid=m.get("uid", 0), hotkey=m.get("hotkey", ""),
              model=m.get("model", ""), slug=m.get("slug", ""))
        for m in miner_data[:args.max_miners]
    ]

    logger.info(f"Loaded {len(miners)} miners")

    try:
        game_env = SDKEnvironment("game")
        await game_env.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize game environment: {e}")
        return

    results = []
    for i, miner in enumerate(miners):
        logger.info(f"Evaluating UID {miner.uid} ({miner.hotkey[:16]}...)...")
        try:
            result = await game_env.evaluate(miner=miner, task_id=int(time.time() * 1000) + i)
            logger.info(f"  Score: {result.score}, Latency: {result.latency_seconds:.2f}s")
            results.append({"uid": miner.uid, "hotkey": miner.hotkey,
                            "score": result.score, "latency": result.latency_seconds,
                            "success": result.success, "error": result.error})
        except Exception as e:
            logger.error(f"  Failed: {e}")
            results.append({"uid": miner.uid, "hotkey": miner.hotkey, "score": 0, "error": str(e)})

    logger.info(f"\nResults: {sum(1 for r in results if r.get('success'))}/{len(results)} succeeded")
    with open("affinetes_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to affinetes_results.json")


if __name__ == "__main__":
    try:
        asyncio.run(run_evaluation())
    except KeyboardInterrupt:
        sys.exit(1)
