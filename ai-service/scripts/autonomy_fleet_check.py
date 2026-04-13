#!/usr/bin/env python3
"""Manifest-aware health checks for the active autonomy fleet.

This script complements the S3 heartbeat-based fleet health check by probing
the active GH200 autonomy roles directly over SSH. It answers the operational
questions that matter for the current training architecture:

- is the role-specific systemd unit active?
- is ringrift-p2p active on the same node?
- is the expected process actually present?
- for trainers, is the supplemental-data lane wired into the live process?
- for selfplay workers, are supplemental shards landing locally and on the
  trainer feed path?

It is intentionally scoped to the manifest-driven GH200 autonomy fleet under
``config/node_roles.yaml`` and does not try to normalize every historical
provider/runtime combination in the repository.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

AI_SERVICE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROLE_CONFIG_PATH = AI_SERVICE_ROOT / "config" / "node_roles.yaml"
DEFAULT_HOSTS_CONFIG_PATH = AI_SERVICE_ROOT / "config" / "distributed_hosts.yaml"
DEFAULT_SSH_KEY_PATH = Path.home() / ".ssh" / "id_cluster"
REMOTE_AI_SERVICE_ROOT = "/home/ubuntu/ringrift/ai-service"


@dataclass(frozen=True)
class TrainerSpec:
    board_type: str
    num_players: int
    work_dir: str


TRAINER_SPECS: dict[str, TrainerSpec] = {
    "hex8_2p": TrainerSpec(board_type="hex8", num_players=2, work_dir="data/minimal_loop_gh200-8"),
    "square8_2p": TrainerSpec(board_type="square8", num_players=2, work_dir="data/minimal_loop_square8_2p"),
    "square8_3p": TrainerSpec(board_type="square8", num_players=3, work_dir="data/minimal_loop_square8_3p"),
    "square19_2p": TrainerSpec(board_type="square19", num_players=2, work_dir="data/minimal_loop_square19_2p"),
}


@dataclass(frozen=True)
class AutonomyTarget:
    node_name: str
    host_name: str
    ip: str
    role: str
    service_name: str
    target_config: str = ""
    assigned_configs: tuple[str, ...] = ()
    feeds_trainer: str | None = None
    primary_dir: str | None = None
    secondary_dir: str | None = None
    trainer_feed_dir: str | None = None


@dataclass
class NodeHealthReport:
    node_name: str
    role: str
    service_name: str
    service_state: str
    p2p_state: str
    process_count: int
    process_sample: str = ""
    primary_file_count: int | None = None
    secondary_file_count: int | None = None
    trainer_feed_file_count: int | None = None
    has_supplemental_flag: bool | None = None
    status: str = "healthy"
    issues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    ssh_error: str | None = None


def _normalize(value: str) -> str:
    return (value or "").lower().replace("-", "").replace("_", "")


def _find_host_config(name: str, host_nodes: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    norm = _normalize(name)
    for host_name, cfg in host_nodes.items():
        host_norm = _normalize(host_name)
        if host_norm == norm or norm in host_norm or host_norm in norm:
            if isinstance(cfg, dict):
                return host_name, cfg
    raise KeyError(f"Host not found for node role entry: {name}")


def _service_name_for_role(role: str) -> str:
    role = (role or "").strip()
    if role == "trainer":
        return "ringrift-training"
    if role == "selfplay-worker":
        return "ringrift-selfplay-worker"
    if role == "evaluator":
        return "ringrift-evaluator"
    if role == "sync-only":
        return "ringrift-p2p"
    raise ValueError(f"Unsupported autonomy role: {role}")


def build_autonomy_targets(
    *,
    role_config_path: Path = DEFAULT_ROLE_CONFIG_PATH,
    hosts_config_path: Path = DEFAULT_HOSTS_CONFIG_PATH,
    only: str = "",
) -> list[AutonomyTarget]:
    hosts = yaml.safe_load(hosts_config_path.read_text()) or {}
    roles = yaml.safe_load(role_config_path.read_text()) or {}

    role_nodes = roles.get("nodes", {})
    host_nodes = hosts.get("hosts", {})
    targets: list[AutonomyTarget] = []

    for node_name, role_cfg in role_nodes.items():
        if not isinstance(role_cfg, dict):
            continue
        role = str(role_cfg.get("role", "")).strip()
        target_config = str(role_cfg.get("target_config", "")).strip()
        if only and only not in {node_name, role, target_config}:
            continue

        host_name, host_cfg = _find_host_config(node_name, host_nodes)
        ip = host_cfg.get("tailscale_ip") or host_cfg.get("ssh_host") or host_cfg.get("host")
        if not ip:
            continue

        feeds_trainer = str(role_cfg.get("feeds_trainer", "")).strip() or None
        primary_dir = None
        secondary_dir = None
        trainer_feed_dir = None
        if role == "trainer":
            spec = TRAINER_SPECS.get(target_config)
            if spec:
                primary_dir = f"{REMOTE_AI_SERVICE_ROOT}/{spec.work_dir}/supplemental"
        elif role == "selfplay-worker":
            primary_dir = f"{REMOTE_AI_SERVICE_ROOT}/data/selfplay/policy_gumbel/{target_config}/supplemental"
            secondary_dir = f"{REMOTE_AI_SERVICE_ROOT}/data/selfplay/policy_gumbel/{target_config}/raw"
            if feeds_trainer:
                feed_cfg = role_nodes.get(feeds_trainer, {})
                feed_target = str(feed_cfg.get("target_config", "")).strip()
                feed_spec = TRAINER_SPECS.get(feed_target)
                if feed_spec:
                    trainer_feed_dir = f"{REMOTE_AI_SERVICE_ROOT}/{feed_spec.work_dir}/supplemental"

        targets.append(
            AutonomyTarget(
                node_name=node_name,
                host_name=host_name,
                ip=str(ip),
                role=role,
                service_name=_service_name_for_role(role),
                target_config=target_config,
                assigned_configs=tuple(role_cfg.get("assigned_configs", [])),
                feeds_trainer=feeds_trainer,
                primary_dir=primary_dir,
                secondary_dir=secondary_dir,
                trainer_feed_dir=trainer_feed_dir,
            )
        )

    return targets


def _make_remote_probe_command(target: AutonomyTarget) -> str:
    payload = {
        "role": target.role,
        "service_name": target.service_name,
        "primary_dir": target.primary_dir or "",
        "secondary_dir": target.secondary_dir or "",
        "trainer_feed_dir": target.trainer_feed_dir or "",
    }
    return f"""python3 - <<'PY'
import json
import subprocess
from pathlib import Path

payload = {json.dumps(payload)}

def run_text(*args):
    result = subprocess.run(list(args), capture_output=True, text=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()

def systemctl_state(unit):
    code, out, _err = run_text("systemctl", "is-active", unit)
    if code == 0:
        return out or "active"
    return out or "unknown"

def process_matches(pattern):
    _code, out, _err = run_text("ps", "-eo", "args")
    matches = []
    for line in out.splitlines():
        if pattern in line and "grep" not in line:
            matches.append(line.strip())
    return matches

def dir_stats(path_str):
    if not path_str:
        return {{"count": None}}
    path = Path(path_str)
    if not path.exists():
        return {{"count": 0, "exists": False}}
    files = [entry for entry in path.iterdir() if entry.is_file()]
    latest_age_s = None
    if files:
        latest_mtime = max(entry.stat().st_mtime for entry in files)
        import time
        latest_age_s = max(0.0, time.time() - latest_mtime)
    return {{"count": len(files), "exists": True, "latest_age_s": latest_age_s}}

pattern = {{
    "trainer": "minimal_alphazero_loop.py",
    "selfplay-worker": "policy_selfplay_worker.py",
    "evaluator": "evaluator_worker.py",
    "sync-only": "p2p_orchestrator.py",
}}[payload["role"]]
matches = process_matches(pattern)
sample = matches[0] if matches else ""
print(json.dumps({{
    "service_state": systemctl_state(payload["service_name"]),
    "p2p_state": systemctl_state("ringrift-p2p"),
    "process_count": len(matches),
    "process_sample": sample,
    "has_supplemental_flag": ("--supplemental-data-dir" in sample) if payload["role"] == "trainer" else None,
    "primary": dir_stats(payload["primary_dir"]),
    "secondary": dir_stats(payload["secondary_dir"]),
    "trainer_feed": dir_stats(payload["trainer_feed_dir"]),
}}))
PY"""


def probe_target(
    target: AutonomyTarget,
    *,
    ssh_key_path: Path = DEFAULT_SSH_KEY_PATH,
    timeout_s: int = 20,
) -> NodeHealthReport:
    ssh_cmd = [
        "ssh",
        "-o",
        "IdentitiesOnly=yes",
        "-i",
        str(ssh_key_path),
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=no",
        f"ubuntu@{target.ip}",
        _make_remote_probe_command(target),
    ]
    result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout_s)
    if result.returncode != 0:
        return NodeHealthReport(
            node_name=target.node_name,
            role=target.role,
            service_name=target.service_name,
            service_state="unknown",
            p2p_state="unknown",
            process_count=0,
            status="critical",
            issues=[f"ssh probe failed ({result.returncode})"],
            ssh_error=result.stderr.strip() or result.stdout.strip(),
        )

    payload = json.loads(result.stdout)
    report = NodeHealthReport(
        node_name=target.node_name,
        role=target.role,
        service_name=target.service_name,
        service_state=str(payload.get("service_state", "unknown")),
        p2p_state=str(payload.get("p2p_state", "unknown")),
        process_count=int(payload.get("process_count", 0)),
        process_sample=str(payload.get("process_sample", "")),
        primary_file_count=payload.get("primary", {}).get("count"),
        secondary_file_count=payload.get("secondary", {}).get("count"),
        trainer_feed_file_count=payload.get("trainer_feed", {}).get("count"),
        has_supplemental_flag=payload.get("has_supplemental_flag"),
    )
    assess_node_health(target, report)
    return report


def _escalate_status(current: str, new: str) -> str:
    rank = {"healthy": 0, "warning": 1, "critical": 2}
    return new if rank[new] > rank[current] else current


def assess_node_health(target: AutonomyTarget, report: NodeHealthReport) -> None:
    if report.service_state != "active":
        report.status = _escalate_status(report.status, "critical")
        report.issues.append(f"{target.service_name} is {report.service_state}")
    if report.p2p_state != "active":
        report.status = _escalate_status(report.status, "critical")
        report.issues.append(f"ringrift-p2p is {report.p2p_state}")

    if target.role in {"trainer", "evaluator"} and report.process_count < 1:
        report.status = _escalate_status(report.status, "critical")
        report.issues.append("expected long-running process is missing")
    elif target.role == "selfplay-worker" and report.process_count < 1:
        report.status = _escalate_status(report.status, "warning")
        report.notes.append("worker service is between batches or process sample was not visible")

    if target.role == "trainer" and report.has_supplemental_flag is False:
        report.status = _escalate_status(report.status, "critical")
        report.issues.append("trainer process missing --supplemental-data-dir")

    if target.role == "trainer" and report.primary_file_count == 0:
        report.notes.append("no supplemental shards visible on trainer yet")
    if target.role == "selfplay-worker":
        if (report.primary_file_count or 0) == 0 and (report.secondary_file_count or 0) > 0:
            report.notes.append("worker batch in progress: raw selfplay output exists but no completed supplemental shard yet")
        if (report.primary_file_count or 0) > 0 and (report.trainer_feed_file_count or 0) == 0:
            report.status = _escalate_status(report.status, "warning")
            report.issues.append("worker has local supplemental output but trainer feed dir is empty")


def apply_feed_relationship_checks(
    targets: list[AutonomyTarget],
    reports: list[NodeHealthReport],
) -> None:
    reports_by_name = {report.node_name: report for report in reports}
    targets_by_name = {target.node_name: target for target in targets}

    for target in targets:
        if target.role != "selfplay-worker" or not target.feeds_trainer:
            continue
        worker = reports_by_name.get(target.node_name)
        trainer = reports_by_name.get(target.feeds_trainer)
        trainer_target = targets_by_name.get(target.feeds_trainer)
        if not worker or not trainer or not trainer_target:
            continue
        if (worker.primary_file_count or 0) > 0 and (trainer.primary_file_count or 0) == 0:
            message = f"{target.node_name} produced supplemental shards but {target.feeds_trainer} still has none"
            worker.status = _escalate_status(worker.status, "warning")
            trainer.status = _escalate_status(trainer.status, "warning")
            if message not in worker.issues:
                worker.issues.append(message)
            if message not in trainer.issues:
                trainer.issues.append(message)


def summarize_reports(reports: list[NodeHealthReport]) -> dict[str, int]:
    summary = {"healthy": 0, "warning": 0, "critical": 0}
    for report in reports:
        summary[report.status] = summary.get(report.status, 0) + 1
    return summary


def _render_human_table(reports: list[NodeHealthReport]) -> str:
    lines = []
    for report in reports:
        counts = []
        if report.primary_file_count is not None:
            counts.append(f"primary={report.primary_file_count}")
        if report.secondary_file_count is not None:
            counts.append(f"secondary={report.secondary_file_count}")
        if report.trainer_feed_file_count is not None:
            counts.append(f"trainer_feed={report.trainer_feed_file_count}")
        detail = ", ".join(counts) if counts else "no-data-dirs"
        lines.append(
            f"{report.node_name:8} {report.role:16} {report.status:8} "
            f"svc={report.service_state:10} p2p={report.p2p_state:10} proc={report.process_count:<2} {detail}"
        )
        for issue in report.issues:
            lines.append(f"  issue: {issue}")
        for note in report.notes:
            lines.append(f"  note: {note}")
    summary = summarize_reports(reports)
    lines.append(
        f"summary: healthy={summary['healthy']} warning={summary['warning']} critical={summary['critical']}"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Health checks for the manifest-driven autonomy fleet")
    parser.add_argument("--only", default="", help="Limit by node name, role, or target config")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a human-readable table")
    parser.add_argument("--role-config-path", type=Path, default=DEFAULT_ROLE_CONFIG_PATH)
    parser.add_argument("--hosts-config-path", type=Path, default=DEFAULT_HOSTS_CONFIG_PATH)
    parser.add_argument("--ssh-key", type=Path, default=DEFAULT_SSH_KEY_PATH)
    args = parser.parse_args()

    targets = build_autonomy_targets(
        role_config_path=args.role_config_path,
        hosts_config_path=args.hosts_config_path,
        only=args.only,
    )
    reports = [probe_target(target, ssh_key_path=args.ssh_key) for target in targets]
    apply_feed_relationship_checks(targets, reports)

    if args.json:
        print(json.dumps({"summary": summarize_reports(reports), "reports": [asdict(r) for r in reports]}, indent=2))
    else:
        print(_render_human_table(reports))

    return 1 if any(report.status == "critical" for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
