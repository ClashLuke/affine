import asyncio
import logging
import os
import sys

from .config import Config
from .loop import run
from .vllm import LocalSlots


def main():
    config = Config.from_env()
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    slots = None
    if os.getenv("AFFINE_LOCAL"):
        slots = LocalSlots(
            os.getenv("CHAMPION_URL", "http://localhost:8000/v1"),
            os.getenv("CHALLENGER_URL", "http://localhost:8001/v1"),
        )

    asyncio.run(run(config, slots))


if __name__ == "__main__":
    main()
