#!/usr/bin/env python3
"""Terminal dashboard for supported minimal-loop training status.

This is intentionally a thin wrapper around scripts/training_status.py so the
machine-readable and human-readable paths share one collector.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
STATUS_SCRIPT = SCRIPT_DIR / "training_status.py"

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
CYAN = "\033[36m"


def _color(enabled: bool, color: str, text: str) -> str:
    return f"{color}{text}{RESET}" if enabled else text


def _status_label(row: dict[str, Any]) -> str:
    if row.get("ssh_ok") is False:
        return "SSH_ERROR"
    if row.get("supervisor_alive") is False and row.get("loop_alive") is True:
        return "SUPERVISOR_DEAD_LOOP_RUNNING"
    if row.get("supervisor_alive") is True and row.get("loop_alive") is False:
        return "SUPERVISOR_RUNNING_LOOP_DEAD"
    if row.get("process_alive") is False:
        return "PROCESS_DEAD"
    status = str(row.get("s3_status") or "UNKNOWN")
    latest = row.get("latest_metrics") or {}
    evaluation = latest.get("evaluation") or {}
    win_rate = evaluation.get("win_rate")
    if win_rate is not None:
        try:
            if float(win_rate) < 0.35 and status == "HEALTHY":
                return "REGRESSING"
        except (TypeError, ValueError):
            pass
    return status


def _status_color(label: str) -> str:
    if label in {"HEALTHY"}:
        return GREEN
    if label in {"STALE", "NO_PROGRESS", "REGRESSING"}:
        return YELLOW
    if label in {"DEAD", "PROCESS_DEAD", "SSH_ERROR", "SUPERVISOR_RUNNING_LOOP_DEAD"}:
        return RED
    return CYAN


def collect_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    cmd = [
        sys.executable,
        str(STATUS_SCRIPT),
        "--json",
        "--stale-hours",
        str(args.stale_hours),
        "--dead-hours",
        str(args.dead_hours),
        "--no-progress-hours",
        str(args.no_progress_hours),
    ]
    if args.ssh:
        cmd.append("--ssh")
    if args.no_s3:
        cmd.append("--no-s3")
    if args.ssh_key:
        cmd.extend(["--ssh-key", args.ssh_key])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"training_status rc={result.returncode}")
    return json.loads(result.stdout)


def render(rows: list[dict[str, Any]], *, color: bool) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(_color(color, BOLD, f"RingRift Training Dashboard - {now}"))
    print(_color(color, DIM, "Ctrl-C to exit"))
    print()
    print(
        f"{'Node':<10} {'Config':<12} {'Iter':>5} {'Elo':>7} {'Promos':>6} "
        f"{'HB Age':<12} {'Loop':<5} {'Sup':<5} {'Sup HB':<12} Status"
    )
    print("-" * 112)
    for row in rows:
        label = _status_label(row)
        loop_alive = row.get("loop_alive")
        supervisor_alive = row.get("supervisor_alive")
        status = _color(color, _status_color(label), label)
        print(
            f"{row['node']:<10} {row['config']:<12} "
            f"{int(row.get('iteration') or 0):>5} {float(row.get('elo') or 0):>7.0f} "
            f"{int(row.get('promotions') or 0):>6} {row.get('heartbeat_age', 'unknown'):<12} "
            f"{'y' if loop_alive else 'n' if loop_alive is False else '?':<5} "
            f"{'y' if supervisor_alive else 'n' if supervisor_alive is False else '?':<5} "
            f"{row.get('supervisor_heartbeat_age', 'unknown'):<12} {status}"
        )
        if row.get("last_error"):
            print(_color(color, RED, f"  {row['node']} last_error: {str(row['last_error']).splitlines()[-1][:150]}"))
        elif row.get("ssh_error"):
            print(_color(color, YELLOW, f"  {row['node']} ssh_error: {str(row['ssh_error'])[:150]}"))
    print("-" * 112)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refreshing terminal dashboard for training status")
    parser.add_argument("--interval", type=int, default=60, help="Refresh interval seconds")
    parser.add_argument("--once", action="store_true", help="Render once and exit")
    parser.add_argument("--ssh", action="store_true", help="Probe nodes over SSH")
    parser.add_argument("--no-s3", action="store_true", help="Skip S3 heartbeat reads")
    parser.add_argument("--ssh-key", default="~/.ssh/id_cluster")
    parser.add_argument("--timeout", type=int, default=40)
    parser.add_argument("--stale-hours", type=float, default=2.0)
    parser.add_argument("--dead-hours", type=float, default=6.0)
    parser.add_argument("--no-progress-hours", type=float, default=24.0)
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    while True:
        if not args.once:
            print("\033[2J\033[H", end="")
        try:
            rows = collect_rows(args)
            render(rows, color=not args.no_color)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(_color(not args.no_color, RED, f"dashboard error: {exc}"))
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
