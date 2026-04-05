#!/usr/bin/env python3
"""Fleet Health Check - reads S3 heartbeats and reports fleet status.

Standalone script that reads heartbeat JSON files from S3 and displays
the health status of all training nodes. No P2P dependency -- only
needs AWS CLI configured with S3 access.

Usage:
    PYTHONPATH=. python3 scripts/fleet_health_check.py
    PYTHONPATH=. python3 scripts/fleet_health_check.py --json
    PYTHONPATH=. python3 scripts/fleet_health_check.py --stale-hours 4 --dead-hours 8
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

S3_HEARTBEAT_PREFIX = "s3://ringrift-models-20251214/consolidated/heartbeats/"

# Default thresholds (seconds)
DEFAULT_STALE_THRESHOLD_H = 2.0
DEFAULT_DEAD_THRESHOLD_H = 6.0
DEFAULT_NO_PROGRESS_H = 24.0

STATUS_HEALTHY = "HEALTHY"
STATUS_STALE = "STALE"
STATUS_DEAD = "DEAD"
STATUS_NO_PROGRESS = "NO_PROGRESS"


def classify_status(
    age_seconds: float,
    stale_threshold_s: float,
    dead_threshold_s: float,
) -> str:
    """Classify node status based on heartbeat age.

    Args:
        age_seconds: Seconds since last heartbeat.
        stale_threshold_s: Seconds after which a node is STALE.
        dead_threshold_s: Seconds after which a node is DEAD.

    Returns:
        One of STATUS_HEALTHY, STATUS_STALE, STATUS_DEAD.
    """
    if age_seconds > dead_threshold_s:
        return STATUS_DEAD
    if age_seconds > stale_threshold_s:
        return STATUS_STALE
    return STATUS_HEALTHY


def format_age(age_seconds: float) -> str:
    """Format age in seconds to a human-readable string."""
    if age_seconds < 60:
        return f"{int(age_seconds)}s ago"
    if age_seconds < 3600:
        return f"{int(age_seconds / 60)}min ago"
    if age_seconds < 86400:
        h = age_seconds / 3600
        return f"{h:.1f}h ago"
    d = age_seconds / 86400
    return f"{d:.1f}d ago"


def list_heartbeat_keys(s3_prefix: str) -> list[str]:
    """List all .json keys under the S3 heartbeat prefix."""
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", s3_prefix],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"ERROR: aws s3 ls failed: {result.stderr.strip()}", file=sys.stderr)
            return []
        keys = []
        for line in result.stdout.strip().splitlines():
            # aws s3 ls output: "2026-04-05 10:30:00  512 gh200-8.json"
            parts = line.split()
            if parts and parts[-1].endswith(".json"):
                keys.append(parts[-1])
        return keys
    except FileNotFoundError:
        print("ERROR: aws CLI not found. Install it or add to PATH.", file=sys.stderr)
        return []
    except subprocess.TimeoutExpired:
        print("ERROR: aws s3 ls timed out.", file=sys.stderr)
        return []


def fetch_heartbeat(s3_prefix: str, key: str) -> dict[str, Any] | None:
    """Download and parse a single heartbeat JSON from S3."""
    try:
        result = subprocess.run(
            ["aws", "s3", "cp", f"{s3_prefix}{key}", "-"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return None


def fetch_all_heartbeats(s3_prefix: str) -> list[dict[str, Any]]:
    """Fetch all heartbeat JSON files from S3."""
    keys = list_heartbeat_keys(s3_prefix)
    heartbeats = []
    for key in keys:
        hb = fetch_heartbeat(s3_prefix, key)
        if hb is not None:
            heartbeats.append(hb)
    return heartbeats


def detect_no_progress(
    heartbeats: list[dict[str, Any]],
    no_progress_threshold_s: float,
) -> set[str]:
    """Detect configs where Elo has not changed across all nodes for a period.

    Groups heartbeats by config_key, checks if any node shows Elo change.
    Returns a set of config_keys that show NO_PROGRESS.
    """
    now = time.time()
    configs_with_progress: dict[str, bool] = {}
    configs_oldest_ts: dict[str, float] = {}

    for hb in heartbeats:
        config = hb.get("config_key", "")
        ts = hb.get("timestamp", 0)
        elo = hb.get("estimated_elo", 0)
        iteration = hb.get("iteration", 0)
        if not config:
            continue

        age = now - ts
        if config not in configs_oldest_ts:
            configs_oldest_ts[config] = ts
        else:
            configs_oldest_ts[config] = min(configs_oldest_ts[config], ts)

        # If any node has >1 iteration and non-default Elo, there's progress
        if iteration > 1 or elo != 1500.0:
            configs_with_progress[config] = True

    no_progress = set()
    for config, oldest_ts in configs_oldest_ts.items():
        age = now - oldest_ts
        if age > no_progress_threshold_s and not configs_with_progress.get(config, False):
            no_progress.add(config)
    return no_progress


def build_fleet_report(
    heartbeats: list[dict[str, Any]],
    stale_threshold_h: float = DEFAULT_STALE_THRESHOLD_H,
    dead_threshold_h: float = DEFAULT_DEAD_THRESHOLD_H,
    no_progress_h: float = DEFAULT_NO_PROGRESS_H,
) -> list[dict[str, Any]]:
    """Build a structured fleet report from heartbeats.

    Returns a list of dicts with node info and status classification.
    """
    now = time.time()
    stale_s = stale_threshold_h * 3600
    dead_s = dead_threshold_h * 3600
    no_progress_s = no_progress_h * 3600

    no_progress_configs = detect_no_progress(heartbeats, no_progress_s)

    report = []
    for hb in heartbeats:
        node_id = hb.get("node_id", "unknown")
        config_key = hb.get("config_key", "unknown")
        iteration = hb.get("iteration", 0)
        elo = hb.get("estimated_elo", 0)
        promos = hb.get("promotions", 0)
        ts = hb.get("timestamp", 0)
        dq_score = hb.get("data_quality_score")

        age = now - ts
        status = classify_status(age, stale_s, dead_s)

        # Override with NO_PROGRESS if applicable (but keep DEAD as worse)
        if config_key in no_progress_configs and status != STATUS_DEAD:
            status = STATUS_NO_PROGRESS

        report.append({
            "node_id": node_id,
            "config_key": config_key,
            "iteration": iteration,
            "estimated_elo": elo,
            "promotions": promos,
            "timestamp": ts,
            "age_seconds": age,
            "age_human": format_age(age),
            "status": status,
            "data_quality_score": dq_score,
        })

    # Sort by node_id for stable output
    report.sort(key=lambda r: r["node_id"])
    return report


def print_table(report: list[dict[str, Any]]) -> None:
    """Print fleet status as a formatted table."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\nFLEET HEALTH CHECK - {now_str}")
    print(f"{'Node':<16} {'Config':<14} {'Iter':>5} {'Elo':>6} {'Promos':>6} {'Last Seen':<14} Status")
    print("-" * 85)

    healthy = stale = dead = no_prog = 0
    for r in report:
        status = r["status"]
        if status == STATUS_HEALTHY:
            healthy += 1
            marker = ""
        elif status == STATUS_STALE:
            stale += 1
            marker = " !!"
        elif status == STATUS_DEAD:
            dead += 1
            marker = " !!!"
        elif status == STATUS_NO_PROGRESS:
            no_prog += 1
            marker = " !!"
        else:
            marker = ""

        print(
            f"{r['node_id']:<16} "
            f"{r['config_key']:<14} "
            f"{r['iteration']:>5} "
            f"{r['estimated_elo']:>6.0f} "
            f"{r['promotions']:>6} "
            f"{r['age_human']:<14} "
            f"{status}{marker}"
        )

    print("-" * 85)
    total = len(report)
    print(f"Total: {total} nodes | {healthy} healthy, {stale} stale, {dead} dead, {no_prog} no_progress")
    if not report:
        print("No heartbeats found. Are any nodes running minimal_alphazero_loop.py?")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fleet Health Check - S3 heartbeat monitor")
    ap.add_argument("--json", action="store_true", help="Output as JSON")
    ap.add_argument("--stale-hours", type=float, default=DEFAULT_STALE_THRESHOLD_H,
                    help=f"Hours before a node is STALE (default: {DEFAULT_STALE_THRESHOLD_H})")
    ap.add_argument("--dead-hours", type=float, default=DEFAULT_DEAD_THRESHOLD_H,
                    help=f"Hours before a node is DEAD (default: {DEFAULT_DEAD_THRESHOLD_H})")
    ap.add_argument("--no-progress-hours", type=float, default=DEFAULT_NO_PROGRESS_H,
                    help=f"Hours with no Elo change = NO_PROGRESS (default: {DEFAULT_NO_PROGRESS_H})")
    ap.add_argument("--s3-prefix", type=str, default=S3_HEARTBEAT_PREFIX,
                    help="S3 prefix for heartbeats")
    args = ap.parse_args()

    heartbeats = fetch_all_heartbeats(args.s3_prefix)
    report = build_fleet_report(
        heartbeats,
        stale_threshold_h=args.stale_hours,
        dead_threshold_h=args.dead_hours,
        no_progress_h=args.no_progress_hours,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_table(report)


if __name__ == "__main__":
    main()
