from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from scripts.autonomy_fleet_check import (
    AutonomyTarget,
    NodeHealthReport,
    apply_feed_relationship_checks,
    assess_node_health,
    build_autonomy_targets,
)


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(dedent(content).strip() + "\n")


def test_build_autonomy_targets_resolves_roles_and_dirs(tmp_path: Path) -> None:
    role_path = tmp_path / "node_roles.yaml"
    hosts_path = tmp_path / "distributed_hosts.yaml"
    _write_yaml(
        role_path,
        """
        version: 1
        nodes:
          gh200-8:
            role: trainer
            target_config: hex8_2p
          gh200-11:
            role: selfplay-worker
            target_config: hex8_2p
            feeds_trainer: gh200-8
          gh200-10:
            role: evaluator
            assigned_configs: [hex8_2p]
        """,
    )
    _write_yaml(
        hosts_path,
        """
        hosts:
          lambda-gh200-8:
            tailscale_ip: 100.121.230.110
          lambda-gh200-10:
            tailscale_ip: 100.100.19.96
          lambda-gh200-11:
            tailscale_ip: 100.106.87.89
        """,
    )

    targets = build_autonomy_targets(role_config_path=role_path, hosts_config_path=hosts_path)
    by_name = {target.node_name: target for target in targets}

    assert by_name["gh200-8"].service_name == "ringrift-training"
    assert by_name["gh200-8"].primary_dir and by_name["gh200-8"].primary_dir.endswith(
        "data/minimal_loop_gh200-8/supplemental"
    )

    assert by_name["gh200-11"].service_name == "ringrift-selfplay-worker"
    assert by_name["gh200-11"].primary_dir and by_name["gh200-11"].primary_dir.endswith(
        "data/selfplay/policy_gumbel/hex8_2p/supplemental"
    )
    assert by_name["gh200-11"].trainer_feed_dir and by_name["gh200-11"].trainer_feed_dir.endswith(
        "data/minimal_loop_gh200-8/supplemental"
    )

    assert by_name["gh200-10"].service_name == "ringrift-evaluator"
    assert by_name["gh200-10"].primary_dir is None


def test_assess_node_health_flags_missing_trainer_supplemental_flag() -> None:
    target = AutonomyTarget(
        node_name="gh200-8",
        host_name="lambda-gh200-8",
        ip="100.121.230.110",
        role="trainer",
        service_name="ringrift-training",
        target_config="hex8_2p",
        primary_dir="/tmp/supplemental",
    )
    report = NodeHealthReport(
        node_name=target.node_name,
        role=target.role,
        service_name=target.service_name,
        service_state="active",
        p2p_state="active",
        process_count=1,
        has_supplemental_flag=False,
        primary_file_count=0,
    )

    assess_node_health(target, report)

    assert report.status == "critical"
    assert any("--supplemental-data-dir" in issue for issue in report.issues)
    assert any("no supplemental shards" in note for note in report.notes)


def test_assess_node_health_notes_worker_batch_in_progress() -> None:
    target = AutonomyTarget(
        node_name="gh200-11",
        host_name="lambda-gh200-11",
        ip="100.106.87.89",
        role="selfplay-worker",
        service_name="ringrift-selfplay-worker",
        target_config="hex8_2p",
        feeds_trainer="gh200-8",
        primary_dir="/tmp/local_supplemental",
        secondary_dir="/tmp/raw",
        trainer_feed_dir="/tmp/trainer_feed",
    )
    report = NodeHealthReport(
        node_name=target.node_name,
        role=target.role,
        service_name=target.service_name,
        service_state="active",
        p2p_state="active",
        process_count=1,
        primary_file_count=0,
        secondary_file_count=2,
        trainer_feed_file_count=0,
    )

    assess_node_health(target, report)

    assert report.status == "healthy"
    assert any("batch in progress" in note for note in report.notes)


def test_apply_feed_relationship_checks_warns_when_worker_not_visible_on_trainer() -> None:
    trainer_target = AutonomyTarget(
        node_name="gh200-8",
        host_name="lambda-gh200-8",
        ip="100.121.230.110",
        role="trainer",
        service_name="ringrift-training",
        target_config="hex8_2p",
        primary_dir="/tmp/trainer_supplemental",
    )
    worker_target = AutonomyTarget(
        node_name="gh200-11",
        host_name="lambda-gh200-11",
        ip="100.106.87.89",
        role="selfplay-worker",
        service_name="ringrift-selfplay-worker",
        target_config="hex8_2p",
        feeds_trainer="gh200-8",
        primary_dir="/tmp/worker_supplemental",
        trainer_feed_dir="/tmp/trainer_supplemental",
    )
    trainer_report = NodeHealthReport(
        node_name="gh200-8",
        role="trainer",
        service_name="ringrift-training",
        service_state="active",
        p2p_state="active",
        process_count=1,
        primary_file_count=0,
        has_supplemental_flag=True,
    )
    worker_report = NodeHealthReport(
        node_name="gh200-11",
        role="selfplay-worker",
        service_name="ringrift-selfplay-worker",
        service_state="active",
        p2p_state="active",
        process_count=1,
        primary_file_count=3,
        trainer_feed_file_count=0,
    )

    apply_feed_relationship_checks([trainer_target, worker_target], [trainer_report, worker_report])

    assert trainer_report.status == "warning"
    assert worker_report.status == "warning"
    assert any("produced supplemental shards" in issue for issue in trainer_report.issues)
    assert any("produced supplemental shards" in issue for issue in worker_report.issues)
