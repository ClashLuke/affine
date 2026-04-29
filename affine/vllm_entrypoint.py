from __future__ import annotations

import argparse
import os
from pathlib import Path

from .backup import restore_from_env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hf", "s3"], default="hf")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--served-model-name", required=True)
    ap.add_argument("--manifest-key")
    ap.add_argument("vllm_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    model_path = args.model
    if args.source == "s3":
        if not args.manifest_key:
            raise SystemExit("--manifest-key is required for --source=s3")
        dest = Path(os.getenv("AFFINE_MODEL_DIR", "/models")) / args.served_model_name.replace("/", "__")
        restore_from_env(args.manifest_key, dest)
        model_path = str(dest)

    extra = args.vllm_args[1:] if args.vllm_args[:1] == ["--"] else args.vllm_args
    argv = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", args.served_model_name,
    ]
    if args.source == "hf":
        argv.extend(["--revision", args.revision])
    argv.extend(extra)
    os.execvp(argv[0], argv)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
