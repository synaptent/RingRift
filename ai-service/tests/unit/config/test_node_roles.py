from __future__ import annotations

from pathlib import Path

from app.config.node_roles import (
    clear_node_role_manifest_cache,
    get_node_workload_policy,
)


def _write_text(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


class TestNodeWorkloadPolicy:
    def teardown_method(self) -> None:
        clear_node_role_manifest_cache()

    def test_trainer_manifest_role_overrides_legacy_selfplay_flags(self, tmp_path: Path) -> None:
        cluster_path = tmp_path / "distributed_hosts.yaml"
        roles_path = tmp_path / "node_roles.yaml"

        _write_text(
            cluster_path,
            """
            hosts:
              gh200-8:
                role: gpu_training_selfplay
                gpu: GH200 (96 GB)
                gpu_vram_gb: 96
                selfplay_enabled: true
                training_enabled: true
                p2p_enabled: true
            """,
        )
        _write_text(
            roles_path,
            """
            nodes:
              gh200-8:
                role: trainer
                target_config: hex8_2p
            """,
        )

        policy = get_node_workload_policy(
            "lambda-gh200-8",
            cluster_config_path=cluster_path,
            role_config_path=roles_path,
        )

        assert policy.resolved is True
        assert policy.role == "trainer"
        assert policy.selfplay_enabled is False
        assert policy.training_enabled is True
        assert policy.job_preference == "training_only"
        assert policy.allowed_config_keys == ("hex8_2p",)

    def test_selfplay_worker_manifest_disables_p2p_selfplay_lane(self, tmp_path: Path) -> None:
        cluster_path = tmp_path / "distributed_hosts.yaml"
        roles_path = tmp_path / "node_roles.yaml"

        _write_text(
            cluster_path,
            """
            hosts:
              gh200-11:
                role: gpu_selfplay
                gpu: GH200 (96 GB)
                gpu_vram_gb: 96
                selfplay_enabled: true
                training_enabled: false
            """,
        )
        _write_text(
            roles_path,
            """
            nodes:
              gh200-11:
                role: selfplay-worker
                target_config: hex8_2p
                feeds_trainer: gh200-8
            """,
        )

        policy = get_node_workload_policy(
            "gh200-11",
            cluster_config_path=cluster_path,
            role_config_path=roles_path,
        )

        assert policy.resolved is True
        assert policy.role == "selfplay-worker"
        assert policy.selfplay_enabled is False
        assert policy.training_enabled is False
        assert policy.selfplay_profile == "disabled"
        assert policy.job_preference == "disabled"
        assert policy.allowed_config_keys == ("hex8_2p",)
        assert policy.feeds_trainer == "gh200-8"

    def test_legacy_gpu_selfplay_role_falls_back_without_overlay(self, tmp_path: Path) -> None:
        cluster_path = tmp_path / "distributed_hosts.yaml"

        _write_text(
            cluster_path,
            """
            hosts:
              vast-1:
                role: gpu_selfplay
                gpu: RTX 4090
                gpu_vram_gb: 24
                selfplay_enabled: true
                training_enabled: false
            """,
        )

        policy = get_node_workload_policy(
            "vast-1",
            cluster_config_path=cluster_path,
            role_config_path=tmp_path / "missing-node-roles.yaml",
        )

        assert policy.resolved is True
        assert policy.role == "selfplay-worker"
        assert policy.selfplay_enabled is True
        assert policy.training_enabled is False
        assert policy.job_preference == "gpu_only"
        assert policy.selfplay_profile == "policy-gumbel"

    def test_unresolved_node_returns_unresolved_policy(self, tmp_path: Path) -> None:
        cluster_path = tmp_path / "distributed_hosts.yaml"
        _write_text(cluster_path, "hosts: {}\n")

        policy = get_node_workload_policy(
            "unknown-node",
            cluster_config_path=cluster_path,
            role_config_path=tmp_path / "missing-node-roles.yaml",
        )

        assert policy.resolved is False
        assert policy.role == "sync-only"
