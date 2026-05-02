import asyncio
import logging
import sys

import bittensor  # noqa: F401 -- eager import so its logging init runs before ours

from .config import Config
from .loop import bittensor_chain, run


async def _main():
    cfg = Config.from_env()
    level = getattr(logging, cfg.log_level)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("affine").setLevel(level)
    for noisy in ("boto3", "botocore", "s3transfer", "urllib3", "smart_open"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    chain, sub = bittensor_chain(cfg)
    try:
        await run(cfg, chain)
    finally:
        await sub.close()


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
