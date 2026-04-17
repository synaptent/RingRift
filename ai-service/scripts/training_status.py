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

def read_json(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def file_age(path):
    p = Path(path)
    if not p.exists():
        return None
    return max(0.0, time.time() - p.stat().st_mtime)

def disk_stats(path):
    candidate = Path(path)
    if not candidate.exists():
        candidate = candidate.parent if candidate.parent.exists() else Path(".")
    try:
        stat = os.statvfs(str(candidate))
    except Exception:
        return {{}}
    total = stat.f_frsize * stat.f_blocks
    free = stat.f_frsize * stat.f_bavail
    used = max(total - free, 0)
    return {{
        "disk_total_gb": round(total / (1024 ** 3), 2),
        "disk_free_gb": round(free / (1024 ** 3), 2),
        "disk_used_percent": round((used / total) * 100.0, 1) if total else None,
    }}

def gpu_stats():
    rc, out = run(
        "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
    )
    if rc != 0 or not out.strip():
        return {{
            "gpu_available": False,
            "gpu_name": None,
            "gpu_utilization_pct": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
        }}
    first = out.splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 4:
        return {{
            "gpu_available": False,
            "gpu_name": first.strip() or None,
            "gpu_utilization_pct": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
        }}
    return {{
        "gpu_available": True,
        "gpu_name": parts[0] or None,
        "gpu_utilization_pct": float(parts[1]) if parts[1] else None,
        "gpu_memory_used_mb": float(parts[2]) if parts[2] else None,
        "gpu_memory_total_mb": float(parts[3]) if parts[3] else None,
    }}

def first_line(text):
    return text.splitlines()[0].strip() if text.splitlines() else None

supervisor_lines = pgrep("scripts/[m]inimal_loop_supervisor.sh.*" + re.escape(work_dir))
loop_lines = pgrep("scripts/[m]inimal_alphazero_loop.py.*--work-dir " + re.escape(work_dir))
supervisor_hb = "/tmp/supervisor_%s.heartbeat" % config
metrics_tail = tail(str(Path(work_dir) / "metrics.jsonl"), 3)
metrics_rows = parse_json_lines(metrics_tail)
supervisor_payload = read_json(supervisor_hb) or {{}}

progress_path = Path(work_dir) / "progress.json"
progress_payload = read_json(str(progress_path)) or None

eval_checkpoint = None
eval_checkpoint_path = None
eval_checkpoint_iteration = None
try:
    eval_candidates = sorted(Path(work_dir).glob("iter_*_eval.json"))
    if eval_candidates:
        latest = eval_candidates[-1]
        eval_checkpoint_path = str(latest)
        eval_checkpoint = read_json(eval_checkpoint_path) or None
        try:
            eval_checkpoint_iteration = int(latest.stem.split("_")[1])
        except (IndexError, ValueError):
            eval_checkpoint_iteration = None
except OSError:
    pass
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

hostname_rc, hostname_out = run("hostname")
host_ip_rc, host_ip_out = run("hostname -I")
tailscale_rc, tailscale_out = run("tailscale ip -4")

print(json.dumps({{
    "ssh_ok": True,
    "supervisor_alive": bool(supervisor_lines),
    "loop_alive": bool(loop_lines),
    "process_alive": bool(loop_lines),
    "supervisor_pid": supervisor_lines[0].split()[0] if supervisor_lines else None,
    "loop_pid": loop_lines[0].split()[0] if loop_lines else None,
    "supervisor_heartbeat_path": supervisor_hb,
    "supervisor_heartbeat_age_seconds": file_age(supervisor_hb),
    "supervisor_heartbeat": supervisor_payload or None,
    "supervisor_state": supervisor_payload.get("state"),
    "supervisor_restart_count": supervisor_payload.get("restart_count"),
    "supervisor_last_restart_time": supervisor_payload.get("last_restart_time"),
    "supervisor_uptime_seconds": supervisor_payload.get("uptime_seconds"),
    "supervisor_pid_from_heartbeat": supervisor_payload.get("supervisor_pid"),
    "metrics_tail": metrics_rows,
    "latest_metrics": metrics_rows[-1] if metrics_rows else None,
    "progress_payload": progress_payload,
    "eval_checkpoint": eval_checkpoint,
    "eval_checkpoint_path": eval_checkpoint_path,
    "eval_checkpoint_iteration": eval_checkpoint_iteration,
    "log_path": log_path,
    "last_error": "\\n".join(error_lines[-6:]),
    "model_exists": Path(model).exists(),
    "hostname": first_line(hostname_out) if hostname_rc == 0 else None,
    "host_ip": first_line(host_ip_out) if host_ip_rc == 0 else None,
    "tailscale_ip": first_line(tailscale_out) if tailscale_rc == 0 else None,
    **disk_stats(work_dir),
    **gpu_stats(),
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
    started_at = time.time()
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
        return {"ssh_ok": False, "ssh_error": "timeout", "ssh_latency_ms": round((time.time() - started_at) * 1000.0, 1)}

    stdout = result.stdout.strip()
    if result.returncode != 0:
        return {
            "ssh_ok": False,
            "ssh_error": (result.stderr.strip() or stdout or f"rc={result.returncode}")[-500:],
            "ssh_latency_ms": round((time.time() - started_at) * 1000.0, 1),
        }
    try:
        payload = json.loads(stdout.splitlines()[-1])
        payload["ssh_latency_ms"] = round((time.time() - started_at) * 1000.0, 1)
        return payload
    except (json.JSONDecodeError, IndexError):
        return {
            "ssh_ok": False,
            "ssh_error": f"unparseable SSH output: {stdout[-500:]}",
            "ssh_latency_ms": round((time.time() - started_at) * 1000.0, 1),
        }


# Standard cumulative eval-stage targets (games played by end of each stage).
# Matches staged_evaluate() in minimal_alphazero_loop.py.
EVAL_STAGE_TARGETS: tuple[int, ...] = (50, 100, 200, 400)


def _derive_eval_progress(
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Turn an iter_N_eval.json checkpoint into a live progress summary.

    Returns None when no checkpoint is available. When a checkpoint is present
    but malformed, returns a best-effort summary with the fields that could be
    parsed — callers should check ``games_played`` before trusting derived
    fields.
    """
    if not isinstance(checkpoint, dict):
        return None
    try:
        cw = int(checkpoint.get("candidate_wins", 0) or 0)
        bw = int(checkpoint.get("best_wins", 0) or 0)
        draws = int(checkpoint.get("draws", 0) or 0)
        played = int(checkpoint.get("games_played", 0) or 0)
    except (TypeError, ValueError):
        return None
    seat_outcomes = checkpoint.get("seat_outcomes") or []
    decided = cw + bw
    running_wr = round(cw / decided, 3) if decided > 0 else None

    stage_targets = list(EVAL_STAGE_TARGETS)
    # A stage is "complete" once games_played reaches its target. The stage
    # currently being worked through is the next one after the last completed
    # stage, capped at the final stage.
    stages_completed = sum(1 for target in stage_targets if played >= target)
    current_stage = min(stages_completed + 1, len(stage_targets))
    remaining_targets = [t for t in stage_targets if played < t]
    next_target = remaining_targets[0] if remaining_targets else stage_targets[-1]

    seat_games: dict[int, int] = {}
    seat_wins: dict[int, int] = {}
    for outcome in seat_outcomes:
        if not isinstance(outcome, dict):
            continue
        try:
            seat = int(outcome["candidate_player"])
        except (KeyError, TypeError, ValueError):
            continue
        seat_games[seat] = seat_games.get(seat, 0) + 1
        if outcome.get("won"):
            seat_wins[seat] = seat_wins.get(seat, 0) + 1

    seat_wr_partial: dict[int, float] | None = None
    seat_imbalance_ratio: float | None = None
    if seat_games:
        seat_wr_partial = {
            seat: round(seat_wins.get(seat, 0) / games, 3) if games > 0 else 0.0
            for seat, games in sorted(seat_games.items())
        }
        wrs = list(seat_wr_partial.values())
        if wrs and min(wrs) > 0:
            seat_imbalance_ratio = round(max(wrs) / min(wrs), 2)

    return {
        "games_played": played,
        "candidate_wins": cw,
        "best_wins": bw,
        "draws": draws,
        "running_wr": running_wr,
        "current_stage": current_stage,
        "next_stage_target": next_target,
        "stage_targets": stage_targets,
        "seat_games": dict(sorted(seat_games.items())) or None,
        "seat_wr_partial": seat_wr_partial,
        "seat_imbalance_ratio": seat_imbalance_ratio,
    }


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
        out["eval_progress"] = _derive_eval_progress(ssh_status.get("eval_checkpoint"))
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
        ep = row.get("eval_progress")
        if ep:
            parts = [
                f"stage {ep['current_stage']}/{len(ep['stage_targets'])}",
                f"games {ep['games_played']}/{ep['next_stage_target']}",
            ]
            running_wr = ep.get("running_wr")
            if running_wr is not None:
                parts.append(f"wr={running_wr:.1%}")
            parts.append(f"cand {ep['candidate_wins']}-{ep['best_wins']} best")
            seat_wr = ep.get("seat_wr_partial")
            if seat_wr:
                seat_str = " ".join(f"s{s}={wr:.0%}" for s, wr in seat_wr.items())
                parts.append(f"seats[{seat_str}]")
            imbalance = ep.get("seat_imbalance_ratio")
            if imbalance is not None and imbalance >= 1.5:
                parts.append(f"imbalance={imbalance}x")
            print(f"{'':<10} eval: {' | '.join(parts)}")
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
