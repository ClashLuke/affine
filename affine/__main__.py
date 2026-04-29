import asyncio
import logging
import os
import sys

import bittensor  # noqa: F401 -- eager import so its logging init runs before ours

from .chain import _truthy_env
from .config import Config
from .dotenv import load_dotenv
from .loop import bittensor_chain, run
from .vllm import LocalSlots


def _local_slots() -> LocalSlots:
    champ = os.environ.get("CHAMPION_URL", "http://localhost:8000/v1")
    chal = os.environ.get("CHALLENGER_URL", "http://localhost:8001/v1")
    return LocalSlots(champ, chal)


async def _main():
    if _truthy_env("AFFINE_LOAD_DOTENV"):
        load_dotenv()
    cfg = Config.from_env()
    level = getattr(logging, cfg.log_level)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("affine").setLevel(level)  # bittensor silences sibling loggers; undo
    for noisy in ("boto3", "botocore", "s3transfer", "urllib3", "smart_open"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    slots = _local_slots() if _truthy_env("AFFINE_LOCAL") else None
    if slots is not None:
        logging.getLogger("affine").info("AFFINE_LOCAL=1 → using LocalSlots")
    chain, sub = bittensor_chain(cfg)
    try:
        await run(cfg, chain, slots=slots)
    finally:
        await sub.close()


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
