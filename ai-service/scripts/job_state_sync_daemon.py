#!/usr/bin/env python3
"""
Job State Sync Daemon - Keeps unified job states aligned with backend reality.

Usage:
    python scripts/job_state_sync_daemon.py --interval 60 --duration-hours 10
    python scripts/job_state_sync_daemon.py --once
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))

LOG_FILE = AI_SERVICE_ROOT / "logs" / "job_state_sync_daemon.log"


def log(message: str, level: str = "INFO") -> None:
    timestamp = datetime.datetime.now().isoformat()
    line = f"[{timestamp}] [{level}] {message}"
    print(line)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


async def run_once(json_output: bool = False) -> dict[str, int]:
    from app.coordination.unified_scheduler import get_scheduler

    scheduler = get_scheduler()
    updates = await scheduler.sync_job_states()
    if json_output:
        print(json.dumps({"updates": updates}, indent=2))
    else:
        log(f"sync_jobs updates={updates}")
    return updates


async def run_loop(interval_seconds: int, duration_hours: float, json_output: bool = False) -> None:
    start_time = time.time()
    end_time = None if duration_hours <= 0 else start_time + duration_hours * 3600

    while True:
        await run_once(json_output=json_output)

        if end_time is not None and time.time() >= end_time:
            break

        await asyncio_sleep(interval_seconds)


async def asyncio_sleep(seconds: int) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Job State Sync Daemon")
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Sync interval in seconds",
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=0,
        help="Total runtime in hours (0 = run forever)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single sync and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output",
    )
    args = parser.parse_args()

    os.environ.setdefault("PYTHONPATH", str(AI_SERVICE_ROOT))

    if args.once:
        import asyncio

        asyncio.run(run_once(json_output=args.json))
        return 0

    log(
        "Job state sync daemon starting "
        f"(interval={args.interval}s, duration_hours={args.duration_hours})"
    )

    import asyncio

    asyncio.run(run_loop(args.interval, args.duration_hours, json_output=args.json))
    log("Job state sync daemon finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
