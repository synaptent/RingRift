#!/usr/bin/env python3
"""Report supported minimal-loop training status.

Reads S3 fleet heartbeats and can optionally SSH-probe the supported training
nodes for process state, metrics tail, supervisor heartbeat, and crash hints.

Usage:
    PYTHONPATH=. python scripts/training_status.py
    PYTHONPATH=. python scripts/training_status.py --ssh
    PYTHONPATH=. python scripts/training_status.py --ssh --json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.fleet_health_check import (
    DEFAULT_DEAD_THRESHOLD_H,
    DEFAULT_NO_PROGRESS_H,
    DEFAULT_STALE_THRESHOLD_H,
    S3_HEARTBEAT_PREFIX,
    build_fleet_report,
    fetch_all_heartbeats,
    format_age,
)


@dataclass(frozen=True)
class TrainingNode:
    node: str
    host: str
    config: str
    work_dir: str
    model: str


SUPPORTED_NODES: tuple[TrainingNode, ...] = (
    TrainingNode(
        node="gh200-8",
        host="100.121.230.110",
        config="hex8_2p",
        work_dir="data/minimal_loop_gh200-8",
        model="models/canonical_hex8_2p.pth",
    ),
    TrainingNode(
        node="gh200-9",
        host="100.127.168.116",
        config="square8_2p",
        work_dir="data/minimal_loop_square8_2p",
        model="models/canonical_square8_2p.pth",
    ),
    TrainingNode(
        node="gh200-12",
        host="100.86.51.4",
        config="square8_3p",
        work_dir="data/minimal_loop_square8_3p",
        model="models/canonical_square8_3p.pth",
    ),
    TrainingNode(
        node="gh200-10",
        host="100.100.19.96",
        config="square8_4p",
        work_dir="data/minimal_loop_square8_4p",
        model="models/canonical_square8_4p.pth",
    ),
)


def _latest_by_node_or_config(report: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    by_config: dict[str, dict[str, Any]] = {}
    for item in sorted(report, key=lambda r: float(r.get("timestamp", 0)), reverse=True):
        node = str(item.get("node_id") or "")
        config = str(item.get("config_key") or "")
        if node and node not in out:
            out[node] = item
        if config and config not in by_config:
            by_config[config] = item
    for node in SUPPORTED_NODES:
        if node.node not in out and node.config in by_config:
            out[node.node] = by_config[node.config]
    return out


def _ssh_args(host: str, user: str, key: str, connect_timeout: int) -> list[str]:
    args = ["ssh"]
    expanded_key = os.path.expanduser(key)
    if expanded_key:
        args.extend(["-i", expanded_key])
    args.extend([
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"ConnectTimeout={connect_timeout}",
        f"{user}@{host}",
    ])
    return args


def _remote_probe_script(node: TrainingNode) -> str:
    return f"""python3 - <<'PY'
import json
import os
import re
import subprocess
import time
from pathlib import Path

config = {json.dumps(node.config)}
work_dir = {json.dumps(node.work_dir)}
model = {json.dumps(node.model)}

def run(cmd):
    try:
        result = subprocess.run(cmd, shell=True, text=True, capture_output=True, timeout=8)
        return result.returncode, (result.stdout + result.stderr).strip()
    except Exception as exc:
        return 124, str(exc)

def pgrep(pattern):
    rc, out = run("pgrep -af " + json.dumps(pattern))
    if rc != 0:
        return []
    return [line for line in out.splitlines() if line.strip()]

def tail(path, lines=20):
    if not Path(path).exists():
        return ""
    rc, out = run("tail -n %d %s" % (lines, json.dumps(path)))
    return out if rc == 0 else ""

def parse_json_lines(text):
    rows = []
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows

def file_age(path):
    p = Path(path)
    if not p.exists():
        return None
    return max(0.0, time.time() - p.stat().st_mtime)

supervisor_lines = pgrep("scripts/[m]inimal_loop_supervisor.sh.*" + re.escape(work_dir))
loop_lines = pgrep("scripts/[m]inimal_alphazero_loop.py.*--work-dir " + re.escape(work_dir))
supervisor_hb = "/tmp/supervisor_%s.heartbeat" % config
metrics_tail = tail(str(Path(work_dir) / "metrics.jsonl"), 3)
log_paths = [
    "/tmp/minimal_alphazero_%s.log" % config,
    "/tmp/minimal_alphazero.log",
    str(Path(work_dir) / "minimal_alphazero.log"),
]
log_tail = ""
log_path = ""
for candidate in log_paths:
    log_tail = tail(candidate, 80)
    if log_tail:
        log_path = candidate
        break

error_lines = []
for line in log_tail.splitlines():
    lowered = line.lower()
    if (
        "traceback" in lowered
        or "error" in lowered
        or "exception" in lowered
        or "critical" in lowered
        or "circuit breaker" in lowered
        or "exited rc=" in lowered
    ):
        error_lines.append(line.strip())

print(json.dumps({{
    "ssh_ok": True,
    "supervisor_alive": bool(supervisor_lines),
    "loop_alive": bool(loop_lines),
    "process_alive": bool(loop_lines),
    "supervisor_pid": supervisor_lines[0].split()[0] if supervisor_lines else None,
    "loop_pid": loop_lines[0].split()[0] if loop_lines else None,
    "supervisor_heartbeat_path": supervisor_hb,
    "supervisor_heartbeat_age_seconds": file_age(supervisor_hb),
    "metrics_tail": parse_json_lines(metrics_tail),
    "latest_metrics": parse_json_lines(metrics_tail)[-1] if parse_json_lines(metrics_tail) else None,
    "log_path": log_path,
    "last_error": "\\n".join(error_lines[-6:]),
    "model_exists": Path(model).exists(),
}}))
PY"""


def ssh_probe_node(
    node: TrainingNode,
    *,
    user: str,
    key: str,
    timeout: int,
    connect_timeout: int,
) -> dict[str, Any]:
    command = (
        "cd ~/ringrift/ai-service 2>/dev/null || cd /home/ubuntu/ringrift/ai-service 2>/dev/null || exit 2; "
        + _remote_probe_script(node)
    )
    try:
        result = subprocess.run(
            [*_ssh_args(node.host, user, key, connect_timeout), command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ssh_ok": False, "ssh_error": "timeout"}

    stdout = result.stdout.strip()
    if result.returncode != 0:
        return {
            "ssh_ok": False,
            "ssh_error": (result.stderr.strip() or stdout or f"rc={result.returncode}")[-500:],
        }
    try:
        return json.loads(stdout.splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"ssh_ok": False, "ssh_error": f"unparseable SSH output: {stdout[-500:]}"}


def _merge_status(
    node: TrainingNode,
    heartbeat: dict[str, Any] | None,
    ssh_status: dict[str, Any] | None,
) -> dict[str, Any]:
    latest_metrics = (ssh_status or {}).get("latest_metrics") or {}
    iteration = (heartbeat or {}).get("iteration", latest_metrics.get("iteration", 0))
    elo = (heartbeat or {}).get("estimated_elo", latest_metrics.get("estimated_elo", 0))
    promotions = (heartbeat or {}).get("promotions", latest_metrics.get("total_promotions", 0))
    age_seconds = (heartbeat or {}).get("age_seconds")
    status = (heartbeat or {}).get("status", "NO_HEARTBEAT")

    out = {
        "node": node.node,
        "host": node.host,
        "config": node.config,
        "work_dir": node.work_dir,
        "model": node.model,
        "iteration": iteration,
        "elo": elo,
        "promotions": promotions,
        "heartbeat_age_seconds": age_seconds,
        "heartbeat_age": format_age(age_seconds) if isinstance(age_seconds, (int, float)) else "unknown",
        "s3_status": status,
        "stage": (heartbeat or {}).get("stage"),
        "timestamp": (heartbeat or {}).get("timestamp"),
        "process_alive": None,
        "loop_alive": None,
        "supervisor_alive": None,
        "supervisor_heartbeat_age_seconds": None,
        "supervisor_heartbeat_age": "unknown",
        "last_error": "",
        "latest_metrics": latest_metrics or None,
        "metrics_tail": [],
        "ssh_ok": None,
        "ssh_error": None,
        "model_exists": None,
    }
    if ssh_status:
        out.update(ssh_status)
        hb_age = ssh_status.get("supervisor_heartbeat_age_seconds")
        out["supervisor_heartbeat_age"] = format_age(hb_age) if isinstance(hb_age, (int, float)) else "unknown"
    return out


def collect_status(args: argparse.Namespace) -> list[dict[str, Any]]:
    heartbeats = [] if args.no_s3 else fetch_all_heartbeats(args.s3_prefix)
    fleet_report = build_fleet_report(
        heartbeats,
        stale_threshold_h=args.stale_hours,
        dead_threshold_h=args.dead_hours,
        no_progress_h=args.no_progress_hours,
    )
    heartbeat_by_node = _latest_by_node_or_config(fleet_report)

    statuses = []
    for node in SUPPORTED_NODES:
        ssh_status = None
        if args.ssh:
            ssh_status = ssh_probe_node(
                node,
                user=args.ssh_user,
                key=args.ssh_key,
                timeout=args.ssh_timeout,
                connect_timeout=args.ssh_connect_timeout,
            )
        statuses.append(_merge_status(node, heartbeat_by_node.get(node.node), ssh_status))
    return statuses


def _status_label(row: dict[str, Any]) -> str:
    if row.get("ssh_ok") is False:
        return "SSH_ERROR"
    if row.get("supervisor_alive") is False and row.get("loop_alive") is True:
        return "SUPERVISOR_DEAD_LOOP_RUNNING"
    if row.get("supervisor_alive") is True and row.get("loop_alive") is False:
        return "SUPERVISOR_RUNNING_LOOP_DEAD"
    if row.get("process_alive") is False:
        return "PROCESS_DEAD"
    return str(row.get("s3_status") or "UNKNOWN")


def print_table(rows: list[dict[str, Any]]) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\nTRAINING STATUS - {now}")
    print(
        f"{'Node':<10} {'Config':<12} {'Iter':>5} {'Elo':>7} {'Promos':>6} "
        f"{'Heartbeat':<12} {'Loop':<5} {'Sup':<5} Status"
    )
    print("-" * 100)
    for row in rows:
        loop_alive = row.get("loop_alive")
        supervisor_alive = row.get("supervisor_alive")
        print(
            f"{row['node']:<10} {row['config']:<12} "
            f"{int(row.get('iteration') or 0):>5} {float(row.get('elo') or 0):>7.0f} "
            f"{int(row.get('promotions') or 0):>6} {row.get('heartbeat_age', 'unknown'):<12} "
            f"{'y' if loop_alive else 'n' if loop_alive is False else '?':<5} "
            f"{'y' if supervisor_alive else 'n' if supervisor_alive is False else '?':<5} "
            f"{_status_label(row)}"
        )
        if row.get("last_error"):
            print(f"{'':<10} last_error: {str(row['last_error']).splitlines()[-1][:140]}")
        elif row.get("ssh_error"):
            print(f"{'':<10} ssh_error: {str(row['ssh_error'])[:140]}")
    print("-" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description="Supported minimal-loop training status")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument("--ssh", action="store_true", help="Probe nodes over SSH for process/log state")
    parser.add_argument("--no-s3", action="store_true", help="Skip S3 heartbeat reads")
    parser.add_argument("--s3-prefix", default=S3_HEARTBEAT_PREFIX, help="S3 heartbeat prefix")
    parser.add_argument("--stale-hours", type=float, default=DEFAULT_STALE_THRESHOLD_H)
    parser.add_argument("--dead-hours", type=float, default=DEFAULT_DEAD_THRESHOLD_H)
    parser.add_argument("--no-progress-hours", type=float, default=DEFAULT_NO_PROGRESS_H)
    parser.add_argument("--ssh-user", default="ubuntu")
    parser.add_argument("--ssh-key", default="~/.ssh/id_cluster")
    parser.add_argument("--ssh-timeout", type=int, default=20)
    parser.add_argument("--ssh-connect-timeout", type=int, default=8)
    args = parser.parse_args()

    rows = collect_status(args)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print_table(rows)


if __name__ == "__main__":
    main()
