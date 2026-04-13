#!/usr/bin/env python3
"""Wrapper daemon for periodic evaluator runs on dedicated nodes."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluator_worker")


def run_once(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "distributed_model_evaluator.py"),
        "--run",
        "--workers",
        str(args.workers),
    ]
    if args.fast:
        cmd.append("--fast")
    if args.force:
        cmd.append("--force")
    if args.board:
        cmd.extend(["--board", args.board])
    logger.info("Starting evaluator cycle: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(AI_SERVICE_ROOT),
        env={**os.environ, "PYTHONPATH": str(AI_SERVICE_ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout:
        logger.info(result.stdout.strip())
    if result.stderr:
        logger.warning(result.stderr.strip())
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Periodic evaluator worker")
    parser.add_argument("--interval", type=int, default=3600)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--board", default="")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        code = run_once(args)
        if args.once:
            return code
        time.sleep(max(args.interval, 1))


if __name__ == "__main__":
    raise SystemExit(main())
