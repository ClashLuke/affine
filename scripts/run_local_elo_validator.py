#!/usr/bin/env python3
"""
Shadow ELO Validator — run games against real miners, store results locally.

Usage:
    python scripts/run_local_elo_validator.py
    python scripts/run_local_elo_validator.py --miners miners.json --concurrent 4
"""

import argparse
import os
import sys


def load_env_file():
    if os.getenv("CHUTES_API_KEY"):
        return
    for path in [".env", "../.env"]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key, value = key.strip(), value.strip().strip('"').strip("'")
                        if key and value:
                            os.environ[key] = value
            break


load_env_file()


def parse_args():
    p = argparse.ArgumentParser(description="Shadow ELO Validator")
    p.add_argument("--miners", type=str, help="Path to miners JSON (default: metagraph)")
    p.add_argument("--max-miners", type=int, default=100)
    p.add_argument("--public-only", action="store_true")
    p.add_argument("--concurrent", type=int, default=8)
    p.add_argument("--game-type", action="append", dest="game_types")
    p.add_argument("--output", type=str, default="elo_matches.json")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--persist", action="store_true", help="Exit on load failure")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-runtime-balancing", action="store_true")
    return p.parse_args()


_ARGS = parse_args()
os.environ["LOCAL_DB_MODE"] = "true"

import asyncio
import json
import threading
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from affine.core.setup import logger, setup_logging
from affine.database.local_backend import init_local_client, close_local_client
from affine.src.elo.config import EloConfig
from affine.src.integration.local_validator_engine import LocalValidatorEngine
from affine.src.integration.elo_validator import MinerInfo

setup_logging(verbosity=0 if _ARGS.quiet else 1)


async def load_miners_from_metagraph(max_miners: int, public_only: bool) -> list[MinerInfo]:
    logger.info("Fetching miners from metagraph...")
    try:
        from affine.core.setup import NETUID
        from affine.utils.subtensor import get_subtensor
        from affine.utils.api_client import cli_api_client

        sub = await get_subtensor()
        meta = await sub.metagraph(NETUID)
        commits = await sub.get_all_revealed_commitments(NETUID)
        logger.info(f"Found {len(commits)} miners with commits")
    except Exception as e:
        logger.error(f"Failed to fetch metagraph: {e}")
        import traceback; traceback.print_exc()
        return []

    miners = []
    sem = asyncio.Semaphore(20)
    api_failures = [0]

    async with cli_api_client() as api_client:
        async def fetch_one(uid, hotkey):
            if hotkey not in commits:
                return None
            try:
                _, commit_data = commits[hotkey][-1]
                data = json.loads(commit_data)
                model, chute_id = data.get("model"), data.get("chute_id")
                if not model or not chute_id:
                    return None

                async with sem:
                    chute = await api_client.get_chute_info(chute_id)
                if not chute or not chute.get("hot", False):
                    return None
                slug = chute.get("slug")
                if not slug:
                    return None
                if public_only and not chute.get("public", False):
                    return None

                return MinerInfo(hotkey=hotkey, model=model, chute_slug=slug,
                                 uid=uid, model_revision=data.get("revision") or "v1",
                                 chute_id=chute_id)
            except Exception:
                api_failures[0] += 1
                return None

        # Test API with first 10
        test_results = await asyncio.gather(*[
            fetch_one(uid, meta.hotkeys[uid]) for uid in range(min(10, len(meta.hotkeys)))
        ])
        if api_failures[0] >= 5 and not any(test_results):
            logger.warning("Chutes API not accessible, falling back to existing data...")
            return await _load_miners_from_existing_data(max_miners)

        # Full fetch
        api_failures[0] = 0
        results = await asyncio.gather(*[
            fetch_one(uid, meta.hotkeys[uid]) for uid in range(len(meta.hotkeys))
        ])

    for m in results:
        if m is not None:
            miners.append(m)
            logger.info(f"  UID {m.uid}: {m.model}")
            if len(miners) >= max_miners:
                break

    logger.info(f"Loaded {len(miners)} miners")
    return miners


async def _load_miners_from_existing_data(max_miners: int) -> list[MinerInfo]:
    for path in ["elo_matches.json", "overnight_elo_matches.json"]:
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            miners = [
                MinerInfo(hotkey=m.get("hotkey", ""), model=m.get("model", ""),
                          chute_slug=m.get("chute_slug", ""), uid=m.get("uid"),
                          model_revision=m.get("model_revision", "v1"))
                for m in data.get("miners", [])[:max_miners]
            ]
            logger.info(f"Loaded {len(miners)} miners from {path}")
            return miners
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")

    logger.error("No existing miner data found. Use --miners.")
    return []


async def load_miners_from_file(path: str) -> list[MinerInfo]:
    with open(path) as f:
        data = json.load(f)
    miners = [
        MinerInfo(
            hotkey=m.get("hotkey", ""),
            model=m.get("model", m.get("registered_model", "")),
            chute_slug=m.get("chute_slug", m.get("slug", "")),
            uid=m.get("uid"),
            model_revision=m.get("model_revision", m.get("revision", "v1")),
        )
        for m in data
    ]
    logger.info(f"Loaded {len(miners)} miners from {path}")
    return miners


async def main():
    args = _ARGS

    if not os.getenv("CHUTES_API_KEY"):
        logger.error("CHUTES_API_KEY not set. Run: source .env")
        sys.exit(1)

    miners = (await load_miners_from_file(args.miners) if args.miners
              else await load_miners_from_metagraph(args.max_miners, args.public_only))

    if len(miners) < 2:
        logger.error("Need at least 2 miners")
        sys.exit(1)

    game_types = args.game_types or ["tictactoe", "chess", "connect4", "nim"]

    db_client = await init_local_client()
    engine = LocalValidatorEngine(
        miners=miners, game_types=game_types, elo_config=EloConfig(),
        concurrent_games=args.concurrent,
        enable_runtime_balancing=not args.no_runtime_balancing,
    )

    # Load existing matches
    existing_miners_data = []
    if not args.fresh and os.path.exists(args.output):
        try:
            with open(args.output) as f:
                existing_data = json.load(f)
            existing_miners_data = existing_data.get("miners", [])
            replayed = engine.game_validator.replay_matches(existing_data.get("matches", []))
            logger.info(f"Replayed {replayed} matches from {args.output}")
        except Exception as e:
            if args.persist:
                logger.error(f"Failed to load (--persist mode): {e}")
                sys.exit(1)
            logger.warning(f"Could not load existing matches: {e}")
    elif args.persist and not os.path.exists(args.output):
        logger.error(f"--persist but {args.output} does not exist")
        sys.exit(1)

    save_lock = threading.Lock()
    last_save_count = [0]

    def save_matches():
        with save_lock:
            current_miners = engine.miners if engine else miners
            all_miners = []
            seen = set()
            current_by_hk = {m.hotkey: m for m in current_miners}

            for m in existing_miners_data:
                hk = m.get("hotkey")
                if hk not in seen:
                    if hk in current_by_hk:
                        m["chute_slug"] = current_by_hk[hk].chute_slug
                    all_miners.append(m)
                    seen.add(hk)
            for m in current_miners:
                if m.hotkey not in seen:
                    all_miners.append({"uid": m.uid, "hotkey": m.hotkey, "model": m.model, "chute_slug": m.chute_slug})
                    seen.add(m.hotkey)

            match_count = len(engine.game_validator._matches)
            with open(args.output, "w") as f:
                json.dump({
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "total_matches": match_count,
                    "miners": all_miners,
                    "matches": engine.game_validator._matches,
                }, f, indent=2, default=str)

            if match_count >= last_save_count[0] + 10:
                logger.info(f"[Checkpoint] {match_count} matches saved")
                last_save_count[0] = match_count

    engine.game_validator.on_game_complete = lambda _: save_matches()

    logger.info(f"Starting: {len(miners)} miners, games={game_types}, workers={args.concurrent}")
    logger.info(f"Runtime balancing: {'disabled' if args.no_runtime_balancing else 'enabled'}")

    try:
        await engine.run_forever()
    finally:
        save_matches()
        logger.info(f"Final: {len(engine.game_validator._matches)} matches → {args.output}")
        await close_local_client()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nInterrupted. Data saved after each game.")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
