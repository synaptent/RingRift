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
import fnmatch
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

from scripts.lib.alerts import AlertSeverity, send_slack_notification

S3_HEARTBEAT_PREFIX = "s3://ringrift-models-20251214/consolidated/heartbeats/"

# Default thresholds (seconds)
DEFAULT_STALE_THRESHOLD_H = 2.0
DEFAULT_DEAD_THRESHOLD_H = 6.0
DEFAULT_NO_PROGRESS_H = 24.0

# End-of-iteration heartbeats can legitimately be sparse on larger or
# higher-game-count loops. These config-aware minima prevent false DEAD
# classifications for healthy long-running jobs while still allowing callers
# to pass even more conservative thresholds on the CLI.
CONFIG_THRESHOLD_OVERRIDES_H: dict[str, tuple[float, float]] = {
    "square8_2p": (6.0, 16.0),
    "square8_3p": (10.0, 24.0),
    "square8_4p": (10.0, 24.0),
    "hex8_2p": (12.0, 24.0),
    "hex8_3p": (10.0, 24.0),
    "hex8_4p": (10.0, 24.0),
    "square19_*": (48.0, 96.0),
    "hexagonal_*": (48.0, 96.0),
}

STATUS_HEALTHY = "HEALTHY"
STATUS_STALE = "STALE"
STATUS_DEAD = "DEAD"
STATUS_NO_PROGRESS = "NO_PROGRESS"


def get_thresholds_for_config(
    config_key: str,
    stale_threshold_h: float,
    dead_threshold_h: float,
) -> tuple[float, float]:
    """Return effective stale/dead thresholds for a config.

    The global CLI thresholds remain the baseline. For known long-running
    configs, we raise those minima to avoid false DEAD classifications when a
    node only publishes heartbeats at iteration completion.
    """
    for pattern, (override_stale_h, override_dead_h) in CONFIG_THRESHOLD_OVERRIDES_H.items():
        if fnmatch.fnmatch(config_key, pattern):
            return max(stale_threshold_h, override_stale_h), max(dead_threshold_h, override_dead_h)
    return stale_threshold_h, dead_threshold_h


def summarize_report(report: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize report counts by status."""
    counts = {
        STATUS_HEALTHY: 0,
        STATUS_STALE: 0,
        STATUS_DEAD: 0,
        STATUS_NO_PROGRESS: 0,
    }
    for item in report:
        status = item.get("status", STATUS_HEALTHY)
        counts[status] = counts.get(status, 0) + 1
    return counts


def build_slack_report(report: list[dict[str, Any]]) -> tuple[AlertSeverity, str, str]:
    """Build a Slack-friendly summary for the fleet report."""
    counts = summarize_report(report)
    total = len(report)

    if total == 0:
        return (
            AlertSeverity.WARNING,
            "Fleet Health Check",
            "No heartbeats found in S3. Fleet visibility is missing.",
        )

    lines = [
        (
            f"Total {total} nodes | {counts.get(STATUS_HEALTHY, 0)} healthy, "
            f"{counts.get(STATUS_STALE, 0)} stale, "
            f"{counts.get(STATUS_DEAD, 0)} dead, "
            f"{counts.get(STATUS_NO_PROGRESS, 0)} no_progress"
        )
    ]

    for status in (STATUS_DEAD, STATUS_STALE, STATUS_NO_PROGRESS):
        flagged = [item for item in report if item.get("status") == status]
        if not flagged:
            continue
        details = ", ".join(
            f"{item['node_id']} ({item['config_key']}, {item['age_human']})"
            for item in flagged[:8]
        )
        if len(flagged) > 8:
            details += f", +{len(flagged) - 8} more"
        lines.append(f"{status}: {details}")

    severity = AlertSeverity.INFO
    if counts.get(STATUS_DEAD, 0) > 0:
        severity = AlertSeverity.CRITICAL
    elif counts.get(STATUS_STALE, 0) > 0 or counts.get(STATUS_NO_PROGRESS, 0) > 0:
        severity = AlertSeverity.WARNING

    return (severity, "Fleet Health Check", "\n".join(lines))


def send_fleet_slack_report(
    report: list[dict[str, Any]],
    webhook_url: str | None = None,
    *,
    send_on_healthy: bool = False,
) -> bool:
    """Send a Slack alert for non-healthy fleet states.

    Returns True when a notification was attempted and sent successfully.
    Returns False when no alert was needed or sending failed.
    """
    severity, title, message = build_slack_report(report)
    if severity == AlertSeverity.INFO and not send_on_healthy:
        return False
    return send_slack_notification(
        message=message,
        severity=severity,
        title=title,
        webhook_url=webhook_url,
        username="Fleet Health Check",
    )


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


def parse_s3_prefix(s3_prefix: str) -> tuple[str, str]:
    """Parse an s3://bucket/prefix URL into bucket and key prefix."""
    if not s3_prefix.startswith("s3://"):
        raise ValueError(f"Invalid S3 prefix: {s3_prefix}")
    bucket_and_prefix = s3_prefix[5:]
    bucket, _, prefix = bucket_and_prefix.partition("/")
    if not bucket:
        raise ValueError(f"Invalid S3 prefix: {s3_prefix}")
    return bucket, prefix


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
            bucket, prefix = parse_s3_prefix(s3_prefix)
            fallback = subprocess.run(
                [
                    "aws",
                    "s3api",
                    "list-objects-v2",
                    "--bucket",
                    bucket,
                    "--prefix",
                    prefix,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if fallback.returncode != 0:
                detail = fallback.stderr.strip() or result.stderr.strip()
                print(f"ERROR: aws s3 ls failed: {detail}", file=sys.stderr)
                return []
            payload = json.loads(fallback.stdout or "{}")
            return [
                PurePosixPath(item["Key"]).name
                for item in payload.get("Contents", [])
                if item.get("Key", "").endswith(".json")
            ]
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
    except (subprocess.TimeoutExpired, ValueError):
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
        effective_stale_h, effective_dead_h = get_thresholds_for_config(
            config_key,
            stale_threshold_h,
            dead_threshold_h,
        )
        status = classify_status(age, effective_stale_h * 3600, effective_dead_h * 3600)

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
            "stale_threshold_hours": effective_stale_h,
            "dead_threshold_hours": effective_dead_h,
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

    counts = summarize_report(report)
    for r in report:
        status = r["status"]
        if status == STATUS_HEALTHY:
            marker = ""
        elif status == STATUS_STALE:
            marker = " !!"
        elif status == STATUS_DEAD:
            marker = " !!!"
        elif status == STATUS_NO_PROGRESS:
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
    print(
        f"Total: {total} nodes | "
        f"{counts.get(STATUS_HEALTHY, 0)} healthy, "
        f"{counts.get(STATUS_STALE, 0)} stale, "
        f"{counts.get(STATUS_DEAD, 0)} dead, "
        f"{counts.get(STATUS_NO_PROGRESS, 0)} no_progress"
    )
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
    ap.add_argument("--slack", action="store_true",
                    help="Send a Slack alert when stale, dead, or no-progress nodes are found")
    ap.add_argument("--slack-webhook", type=str, default=None,
                    help="Override Slack webhook URL (otherwise uses environment/file config)")
    ap.add_argument("--slack-on-healthy", action="store_true",
                    help="Also send an informational Slack message when the fleet is fully healthy")
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

    if args.slack:
        sent = send_fleet_slack_report(
            report,
            webhook_url=args.slack_webhook,
            send_on_healthy=args.slack_on_healthy,
        )
        if not sent:
            counts = summarize_report(report)
            if counts.get(STATUS_DEAD, 0) == 0 and counts.get(STATUS_STALE, 0) == 0 and counts.get(STATUS_NO_PROGRESS, 0) == 0:
                print("Slack alert skipped: fleet is healthy.")
            else:
                print("Slack alert failed or webhook not configured.", file=sys.stderr)


if __name__ == "__main__":
    main()
