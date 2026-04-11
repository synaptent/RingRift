#!/usr/bin/env python3
"""Comprehensive health report for supported training nodes.

This extends scripts/training_status.py with a higher-signal human report that
surfaces process state, supervisor heartbeat details, GPU usage, disk capacity,
and connectivity in one place.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from scripts.fleet_health_check import (
    DEFAULT_DEAD_THRESHOLD_H,
    DEFAULT_NO_PROGRESS_H,
    DEFAULT_STALE_THRESHOLD_H,
    S3_HEARTBEAT_PREFIX,
)
from scripts.training_status import collect_status


def _format_duration(seconds: float | int | None) -> str:
    if not isinstance(seconds, (int, float)):
        return "unknown"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def _status_label(row: dict[str, Any]) -> str:
    if row.get("ssh_ok") is False:
        return "SSH_ERROR"
    if row.get("supervisor_alive") is False and row.get("loop_alive") is True:
        return "SUPERVISOR_DEAD_LOOP_RUNNING"
    if row.get("supervisor_alive") is True and row.get("loop_alive") is False:
        return "SUPERVISOR_RUNNING_LOOP_DEAD"
    if row.get("process_alive") is False:
        return "PROCESS_DEAD"
    latest = row.get("latest_metrics") or {}
    evaluation = latest.get("evaluation") or {}
    win_rate = evaluation.get("win_rate")
    if row.get("s3_status") == "HEALTHY" and isinstance(win_rate, (int, float)) and float(win_rate) < 0.35:
        return "REGRESSING"
    return str(row.get("s3_status") or "UNKNOWN")


def _elo_trend(row: dict[str, Any]) -> str:
    values: list[float] = []
    for item in row.get("metrics_tail") or []:
        value = item.get("estimated_elo", item.get("elo"))
        if isinstance(value, (int, float)):
            values.append(float(value))
    if len(values) < 2:
        return "n/a"
    delta = values[-1] - values[0]
    if abs(delta) < 1.0:
        return "flat"
    return f"{delta:+.1f}"


def _gpu_summary(row: dict[str, Any]) -> str:
    if not row.get("gpu_available"):
        return "n/a"
    name = row.get("gpu_name") or "gpu"
    util = row.get("gpu_utilization_pct")
    used = row.get("gpu_memory_used_mb")
    total = row.get("gpu_memory_total_mb")
    util_text = f"{util:.0f}%" if isinstance(util, (int, float)) else "?"
    mem_text = (
        f"{used:.0f}/{total:.0f}MB"
        if isinstance(used, (int, float)) and isinstance(total, (int, float))
        else "?"
    )
    return f"{name} {util_text} {mem_text}"


def _disk_summary(row: dict[str, Any]) -> str:
    free = row.get("disk_free_gb")
    used_pct = row.get("disk_used_percent")
    if not isinstance(free, (int, float)):
        return "n/a"
    if isinstance(used_pct, (int, float)):
        return f"{free:.1f}GB free ({used_pct:.0f}% used)"
    return f"{free:.1f}GB free"


def _network_summary(row: dict[str, Any]) -> str:
    if row.get("ssh_ok") is False:
        return "down"
    latency = row.get("ssh_latency_ms")
    host_ip = row.get("tailscale_ip") or row.get("host_ip")
    latency_text = f"{latency:.0f}ms" if isinstance(latency, (int, float)) else "?"
    if host_ip:
        return f"{latency_text} {host_ip}"
    return latency_text


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_nodes": len(rows),
        "status_counts": {},
        "dead_nodes": [],
        "stale_nodes": [],
        "regressing_nodes": [],
    }
    for row in rows:
        label = _status_label(row)
        summary["status_counts"][label] = summary["status_counts"].get(label, 0) + 1
        if label in {"PROCESS_DEAD", "SUPERVISOR_RUNNING_LOOP_DEAD", "SUPERVISOR_DEAD_LOOP_RUNNING", "SSH_ERROR", "DEAD"}:
            summary["dead_nodes"].append(row["node"])
        if label in {"STALE", "NO_PROGRESS"}:
            summary["stale_nodes"].append(row["node"])
        if label == "REGRESSING":
            summary["regressing_nodes"].append(row["node"])
    return summary


def print_report(rows: list[dict[str, Any]]) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\nCLUSTER HEALTH - {now}")
    print(
        f"{'Node':<10} {'Config':<12} {'Iter':>5} {'Elo':>7} {'Trend':>7} "
        f"{'HB Age':<12} {'Loop':<5} {'Sup':<5} Status"
    )
    print("-" * 100)
    for row in rows:
        loop_alive = row.get("loop_alive")
        supervisor_alive = row.get("supervisor_alive")
        print(
            f"{row['node']:<10} {row['config']:<12} "
            f"{int(row.get('iteration') or 0):>5} {float(row.get('elo') or 0):>7.1f} "
            f"{_elo_trend(row):>7} {row.get('heartbeat_age', 'unknown'):<12} "
            f"{'y' if loop_alive else 'n' if loop_alive is False else '?':<5} "
            f"{'y' if supervisor_alive else 'n' if supervisor_alive is False else '?':<5} "
            f"{_status_label(row)}"
        )
        print(
            f"{'':<10} gpu={_gpu_summary(row)} | disk={_disk_summary(row)} | "
            f"net={_network_summary(row)}"
        )
        print(
            f"{'':<10} supervisor_hb={row.get('supervisor_heartbeat_age', 'unknown')} | "
            f"restarts={row.get('supervisor_restart_count', 'n/a')} | "
            f"uptime={_format_duration(row.get('supervisor_uptime_seconds'))}"
        )
        if row.get("supervisor_last_restart_time"):
            print(f"{'':<10} last_restart={row['supervisor_last_restart_time']}")
        if row.get("last_error"):
            print(f"{'':<10} last_error={str(row['last_error']).splitlines()[-1][:180]}")
        elif row.get("ssh_error"):
            print(f"{'':<10} ssh_error={str(row['ssh_error'])[:180]}")
    print("-" * 100)
    summary = build_summary(rows)
    print(
        "Summary: "
        f"{summary['total_nodes']} nodes | "
        + ", ".join(f"{label}={count}" for label, count in sorted(summary["status_counts"].items()))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive health report for supported training nodes")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--no-ssh", action="store_true", help="Skip SSH probes")
    parser.add_argument("--no-s3", action="store_true", help="Skip S3 heartbeat reads")
    parser.add_argument("--s3-prefix", default=S3_HEARTBEAT_PREFIX)
    parser.add_argument("--stale-hours", type=float, default=DEFAULT_STALE_THRESHOLD_H)
    parser.add_argument("--dead-hours", type=float, default=DEFAULT_DEAD_THRESHOLD_H)
    parser.add_argument("--no-progress-hours", type=float, default=DEFAULT_NO_PROGRESS_H)
    parser.add_argument("--ssh-user", default="ubuntu")
    parser.add_argument("--ssh-key", default="~/.ssh/id_cluster")
    parser.add_argument("--ssh-timeout", type=int, default=20)
    parser.add_argument("--ssh-connect-timeout", type=int, default=8)
    args = parser.parse_args()

    status_args = argparse.Namespace(
        json=False,
        ssh=not args.no_ssh,
        no_s3=args.no_s3,
        s3_prefix=args.s3_prefix,
        stale_hours=args.stale_hours,
        dead_hours=args.dead_hours,
        no_progress_hours=args.no_progress_hours,
        ssh_user=args.ssh_user,
        ssh_key=args.ssh_key,
        ssh_timeout=args.ssh_timeout,
        ssh_connect_timeout=args.ssh_connect_timeout,
    )
    rows = collect_status(status_args)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": build_summary(rows),
        "nodes": rows,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_report(rows)


if __name__ == "__main__":
    main()
