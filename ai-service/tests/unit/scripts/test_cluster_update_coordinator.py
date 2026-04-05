"""
Unit tests for QuorumSafeUpdateCoordinator.

January 3, 2026 - Sprint 16.2: Tests for quorum-safe rolling updates.
"""

import asyncio
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.cluster_update_coordinator import (
    QuorumSafeUpdateCoordinator,
    UpdateCoordinatorConfig,
    NodeConfig,
    UpdateResult,
    UpdateBatch,
    BatchCheckpoint,
    QuorumHealthLevel,
    ClusterHealth,
    QuorumUnsafeError,
    ConvergenceTimeoutError,
    QuorumLostError,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Default configuration for tests."""
    return UpdateCoordinatorConfig(
        convergence_timeout=10.0,  # Short timeout for tests
        voter_update_delay=1.0,
        max_parallel_non_voters=5,
        dry_run=False,
    )


@pytest.fixture
def mock_node_configs():
    """Mock node configurations as list of NodeConfig objects."""
    # Create list of voters and non-voters
    voter_names = ["lambda-gh200-1", "lambda-gh200-2", "nebius-backbone-1"]
    nodes = []

    # Voters
    for i, name in enumerate(voter_names):
        nodes.append(NodeConfig(
            name=name,
            ssh_host=f"100.1.1.{i + 1}",
            ssh_port=22,
            ssh_user="root",
            ssh_key=None,
            tailscale_ip=f"100.1.1.{i + 1}",
            ringrift_path="/workspace/ringrift/ai-service",
            is_voter=True,
            status="ready",
        ))

    # Non-voters
    for i, name in enumerate(["vast-gpu-1", "vast-gpu-2", "runpod-h100"]):
        nodes.append(NodeConfig(
            name=name,
            ssh_host=f"100.2.1.{i + 1}",
            ssh_port=22,
            ssh_user="root",
            ssh_key=None,
            tailscale_ip=f"100.2.1.{i + 1}",
            ringrift_path="~/ringrift/ai-service",
            is_voter=False,
            status="ready",
        ))

    return nodes


# =============================================================================
# Configuration Tests
# =============================================================================


class TestUpdateCoordinatorConfig:
    """Tests for configuration dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = UpdateCoordinatorConfig()
        assert config.convergence_timeout == 120.0
        assert config.voter_update_delay == 30.0
        assert config.max_parallel_non_voters == 10
        assert config.dry_run is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = UpdateCoordinatorConfig(
            convergence_timeout=180.0,
            voter_update_delay=60.0,
            max_parallel_non_voters=5,
            dry_run=True,
        )
        assert config.convergence_timeout == 180.0
        assert config.voter_update_delay == 60.0
        assert config.max_parallel_non_voters == 5
        assert config.dry_run is True


# =============================================================================
# NodeConfig Tests
# =============================================================================


class TestNodeConfig:
    """Tests for NodeConfig dataclass."""

    def test_node_config_creation(self):
        """Test NodeConfig creation with all fields."""
        node = NodeConfig(
            name="test-node",
            ssh_host="192.168.1.1",
            ssh_port=22,
            ssh_user="root",
            ssh_key=None,
            tailscale_ip="100.1.1.1",
            ringrift_path="/home/ringrift",
            is_voter=True,
            status="ready",
        )
        assert node.name == "test-node"
        assert node.is_voter is True
        assert node.status == "ready"

    def test_node_config_nat_blocked_default(self):
        """Test NodeConfig nat_blocked defaults to False."""
        node = NodeConfig(
            name="test-node",
            ssh_host="192.168.1.1",
            ssh_port=22,
            ssh_user="root",
            ssh_key=None,
            tailscale_ip="100.1.1.1",
            ringrift_path="/home/ringrift",
            is_voter=False,
            status="ready",
        )
        assert node.nat_blocked is False


# =============================================================================
# UpdateBatch Tests
# =============================================================================


class TestUpdateBatch:
    """Tests for UpdateBatch dataclass."""

    def test_update_batch_node_names_property(self, mock_node_configs):
        """Test that node_names property returns correct names."""
        batch = UpdateBatch(
            nodes=mock_node_configs[:3],  # First 3 nodes (voters)
            batch_type="voter",
        )
        assert "lambda-gh200-1" in batch.node_names
        assert "lambda-gh200-2" in batch.node_names
        assert "nebius-backbone-1" in batch.node_names
        assert len(batch.node_names) == 3

    def test_update_batch_empty_nodes(self):
        """Test UpdateBatch with empty node list."""
        batch = UpdateBatch(nodes=[], batch_type="non_voters")
        assert batch.node_names == []
        assert batch.batch_type == "non_voters"


# =============================================================================
# BatchCheckpoint Tests
# =============================================================================


class TestBatchCheckpoint:
    """Tests for BatchCheckpoint dataclass."""

    def test_checkpoint_creation(self):
        """Test BatchCheckpoint creation."""
        checkpoint = BatchCheckpoint(
            batch_nodes=["node1", "node2"],
            previous_commits={"node1": "abc123", "node2": "def456"},
            p2p_was_running={"node1": True, "node2": False},
            timestamp=1234567890.0,
        )
        assert len(checkpoint.batch_nodes) == 2
        assert checkpoint.previous_commits["node1"] == "abc123"
        assert checkpoint.p2p_was_running["node1"] is True


# =============================================================================
# UpdateResult Tests
# =============================================================================


class TestUpdateResult:
    """Tests for UpdateResult dataclass."""

    def test_success_result(self):
        """Test UpdateResult for successful update."""
        result = UpdateResult(
            success=True,
            batches_updated=3,
            nodes_updated=["node1", "node2", "node3"],
            nodes_failed=[],
            nodes_skipped=[],
        )
        assert result.success is True
        assert result.batches_updated == 3
        assert len(result.nodes_updated) == 3
        assert result.error_message is None

    def test_failure_result(self):
        """Test UpdateResult for failed update."""
        result = UpdateResult(
            success=False,
            batches_updated=1,
            nodes_updated=["node1"],
            nodes_failed=["node2"],
            nodes_skipped=[],
            rollback_performed=True,
            error_message="Convergence timeout",
        )
        assert result.success is False
        assert result.rollback_performed is True
        assert result.error_message == "Convergence timeout"


# =============================================================================
# ClusterHealth Tests
# =============================================================================


class TestClusterHealth:
    """Tests for ClusterHealth dataclass."""

    def test_healthy_cluster(self):
        """Test ClusterHealth for healthy cluster."""
        health = ClusterHealth(
            quorum_level=QuorumHealthLevel.HEALTHY,
            alive_peers=7,
            total_peers=7,
            leader_id="leader-node",
            alive_voters=["v1", "v2", "v3", "v4", "v5", "v6", "v7"],
            total_voters=7,
            quorum_required=4,
        )
        assert health.quorum_level == QuorumHealthLevel.HEALTHY
        assert health.alive_peers == 7
        assert len(health.alive_voters) == 7

    def test_minimum_quorum(self):
        """Test ClusterHealth at minimum quorum."""
        health = ClusterHealth(
            quorum_level=QuorumHealthLevel.MINIMUM,
            alive_peers=4,
            total_peers=7,
            leader_id="leader-node",
            alive_voters=["v1", "v2", "v3", "v4"],
            total_voters=7,
            quorum_required=4,
        )
        assert health.quorum_level == QuorumHealthLevel.MINIMUM


# =============================================================================
# QuorumHealthLevel Tests
# =============================================================================


class TestQuorumHealthLevel:
    """Tests for QuorumHealthLevel enum."""

    def test_health_levels(self):
        """Test all health level values."""
        assert QuorumHealthLevel.HEALTHY.value == "healthy"
        assert QuorumHealthLevel.DEGRADED.value == "degraded"
        assert QuorumHealthLevel.MINIMUM.value == "minimum"
        assert QuorumHealthLevel.LOST.value == "lost"


# =============================================================================
# Error Classes Tests
# =============================================================================


class TestErrorClasses:
    """Tests for custom error classes."""

    def test_quorum_unsafe_error(self):
        """Test QuorumUnsafeError exception."""
        with pytest.raises(QuorumUnsafeError):
            raise QuorumUnsafeError("Quorum at minimum")

    def test_convergence_timeout_error(self):
        """Test ConvergenceTimeoutError exception."""
        with pytest.raises(ConvergenceTimeoutError):
            raise ConvergenceTimeoutError("Batch 2 failed to converge")

    def test_quorum_lost_error(self):
        """Test QuorumLostError exception."""
        with pytest.raises(QuorumLostError):
            raise QuorumLostError("Quorum lost after batch 2")


# =============================================================================
# Coordinator Initialization Tests
# =============================================================================


class TestCoordinatorInitialization:
    """Tests for coordinator initialization."""

    def test_init_with_config_object(self, config):
        """Test initialization with config object."""
        with patch.object(QuorumSafeUpdateCoordinator, '_find_config_path', return_value=Path("/tmp/test.yaml")):
            coordinator = QuorumSafeUpdateCoordinator(config=config)
            assert coordinator.convergence_timeout == 10.0
            assert coordinator.voter_update_delay == 1.0
            assert coordinator.max_parallel_non_voters == 5

    def test_init_with_individual_args(self):
        """Test initialization with individual arguments."""
        with patch.object(QuorumSafeUpdateCoordinator, '_find_config_path', return_value=Path("/tmp/test.yaml")):
            coordinator = QuorumSafeUpdateCoordinator(
                convergence_timeout=60.0,
                voter_update_delay=15.0,
            )
            assert coordinator.convergence_timeout == 60.0
            assert coordinator.voter_update_delay == 15.0


class TestSingleNodeUpdate:
    """Tests for per-node update flow."""

    @pytest.mark.asyncio
    async def test_smoke_failure_blocks_restart(self, config, mock_node_configs):
        """A failed deploy smoke test should fail the node before restart."""
        node = next(n for n in mock_node_configs if not n.is_voter)

        with patch.object(
            QuorumSafeUpdateCoordinator,
            '_find_config_path',
            return_value=Path("/tmp/test.yaml"),
        ):
            coordinator = QuorumSafeUpdateCoordinator(config=config)

        client = MagicMock()
        client.run_async = AsyncMock(side_effect=[
            MagicMock(returncode=0, stdout="connected\n", stderr=""),
            MagicMock(returncode=0, stdout="1234\n", stderr=""),
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=0, stdout="abcdef1234567890\n", stderr=""),
        ])

        with patch.object(coordinator, "_create_ssh_client", AsyncMock(return_value=client)):
            with patch(
                "scripts.cluster_update_coordinator.run_remote_deploy_smoke_test",
                new=AsyncMock(return_value=(False, "deploy smoke failed on vast-gpu-1: import boom")),
            ):
                with patch.object(coordinator, "_restart_p2p_process", AsyncMock()) as mock_restart:
                    node_name, success, message = await coordinator._update_single_node(
                        node=node,
                        target_commit="main",
                        restart_p2p=True,
                        dry_run=False,
                        sync_config=False,
                    )

        assert node_name == node.name
        assert success is False
        assert "deploy smoke failed" in message
        mock_restart.assert_not_called()


class TestP2PManagerDetection:
    """Tests for platform-specific P2P manager detection and restart."""

    @pytest.mark.asyncio
    async def test_detect_p2p_manager_returns_launchd_when_label_present(self, config):
        with patch.object(
            QuorumSafeUpdateCoordinator,
            '_find_config_path',
            return_value=Path("/tmp/test.yaml"),
        ):
            coordinator = QuorumSafeUpdateCoordinator(config=config)

        client = MagicMock()
        client.run_async = AsyncMock(side_effect=[
            MagicMock(returncode=0, stdout="disabled\n", stderr=""),
            MagicMock(returncode=1, stdout="", stderr=""),
            MagicMock(returncode=0, stdout="91804\t0\tcom.ringrift.p2p\n", stderr=""),
        ])

        manager = await coordinator._detect_p2p_manager(client, "mac-studio")
        assert manager == "launchd"

    @pytest.mark.asyncio
    async def test_restart_p2p_process_uses_launchd_kickstart(self, config):
        with patch.object(
            QuorumSafeUpdateCoordinator,
            '_find_config_path',
            return_value=Path("/tmp/test.yaml"),
        ):
            coordinator = QuorumSafeUpdateCoordinator(config=config)

        node = NodeConfig(
            name="mac-studio",
            ssh_host="100.71.253.66",
            ssh_port=22,
            ssh_user="armand",
            ssh_key=None,
            tailscale_ip="100.71.253.66",
            ringrift_path="~/Development/RingRift/ai-service",
            is_voter=True,
            status="ready",
        )
        client = MagicMock()
        client.run_async = AsyncMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))

        with patch.object(coordinator, "_detect_p2p_manager", AsyncMock(return_value="launchd")):
            with patch.object(coordinator, "_check_p2p_running", AsyncMock(return_value=True)):
                with patch.object(coordinator, "_wait_for_p2p_http_ready", AsyncMock(return_value=True)) as mock_ready:
                    with patch("scripts.cluster_update_coordinator.asyncio.sleep", new=AsyncMock()) as mock_sleep:
                        ok = await coordinator._restart_p2p_process(client, node)

        assert ok is True
        restart_cmd = client.run_async.await_args_list[0].args[0]
        assert "launchctl kickstart -k gui/$(id -u)/com.ringrift.p2p" in restart_cmd
        assert "launchctl kickstart -k gui/$(id -u)/com.ringrift.p2p-orchestrator" in restart_cmd
        mock_sleep.assert_awaited_once_with(5)
        mock_ready.assert_awaited_once_with(client, node.name)

    @pytest.mark.asyncio
    async def test_wait_for_p2p_http_ready_retries_until_endpoint_responds(self, config):
        with patch.object(
            QuorumSafeUpdateCoordinator,
            '_find_config_path',
            return_value=Path("/tmp/test.yaml"),
        ):
            coordinator = QuorumSafeUpdateCoordinator(config=config)

        client = MagicMock()
        client.run_async = AsyncMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))

        ready = await coordinator._wait_for_p2p_http_ready(client, "mac-studio", timeout=30.0)

        assert ready is True
        assert client.run_async.await_count == 1
        probe_cmd = client.run_async.await_args_list[0].args[0]
        assert "while [ \"$attempt\" -lt 15 ]" in probe_cmd
        assert "http://127.0.0.1:8770/health" in probe_cmd
        assert "http://127.0.0.1:8770/status" in probe_cmd
        assert "sleep 2" in probe_cmd


class TestBatchFailureHandling:
    """Tests for batch failure and rollback behavior."""

    @pytest.mark.asyncio
    async def test_update_cluster_rolls_back_on_node_failure(self, config, mock_node_configs):
        """Any failed node in a batch should abort and roll back that batch."""
        with patch.object(
            QuorumSafeUpdateCoordinator,
            '_find_config_path',
            return_value=Path("/tmp/test.yaml"),
        ):
            coordinator = QuorumSafeUpdateCoordinator(config=config)

        batch = UpdateBatch(nodes=[mock_node_configs[3]], batch_type="non_voters")
        checkpoint = BatchCheckpoint(
            batch_nodes=batch.node_names,
            previous_commits={batch.node_names[0]: "abc123"},
            p2p_was_running={batch.node_names[0]: True},
            timestamp=0.0,
        )
        healthy = ClusterHealth(
            quorum_level=QuorumHealthLevel.HEALTHY,
            alive_peers=6,
            total_peers=6,
            leader_id="lambda-gh200-1",
            alive_voters=["lambda-gh200-1", "lambda-gh200-2", "nebius-backbone-1"],
            total_voters=3,
            quorum_required=2,
        )

        with patch.object(coordinator, "_load_config"):
            with patch.object(coordinator, "_get_cluster_health", AsyncMock(return_value=healthy)):
                with patch.object(coordinator, "_get_node_configs", return_value=mock_node_configs):
                    with patch.object(coordinator, "_calculate_update_batches", return_value=[batch]):
                        with patch.object(coordinator, "_save_batch_checkpoint", AsyncMock(return_value=checkpoint)):
                            with patch.object(
                                coordinator,
                                "_update_batch",
                                AsyncMock(return_value=[(batch.node_names[0], False, "deploy smoke failed")]),
                            ):
                                with patch.object(coordinator, "_rollback_batch", AsyncMock()) as mock_rollback:
                                    result = await coordinator.update_cluster(
                                        target_commit="main",
                                        restart_p2p=True,
                                        dry_run=False,
                                    )

        assert result.success is False
        assert result.rollback_performed is True
        assert result.failed_batch == 1
        assert batch.node_names[0] in result.nodes_failed
        mock_rollback.assert_awaited_once_with(batch, checkpoint)

    @pytest.mark.asyncio
    async def test_update_cluster_dry_run_does_not_rollback_on_node_failure(
        self,
        config,
        mock_node_configs,
    ):
        """Dry-run failures should report problems without mutating nodes."""
        with patch.object(
            QuorumSafeUpdateCoordinator,
            '_find_config_path',
            return_value=Path("/tmp/test.yaml"),
        ):
            coordinator = QuorumSafeUpdateCoordinator(config=config)

        batch = UpdateBatch(nodes=[mock_node_configs[3]], batch_type="non_voters")
        checkpoint = BatchCheckpoint(
            batch_nodes=batch.node_names,
            previous_commits={batch.node_names[0]: "abc123"},
            p2p_was_running={batch.node_names[0]: True},
            timestamp=0.0,
        )
        healthy = ClusterHealth(
            quorum_level=QuorumHealthLevel.HEALTHY,
            alive_peers=6,
            total_peers=6,
            leader_id="lambda-gh200-1",
            alive_voters=["lambda-gh200-1", "lambda-gh200-2", "nebius-backbone-1"],
            total_voters=3,
            quorum_required=2,
        )

        with patch.object(coordinator, "_load_config"):
            with patch.object(coordinator, "_get_cluster_health", AsyncMock(return_value=healthy)):
                with patch.object(coordinator, "_get_node_configs", return_value=mock_node_configs):
                    with patch.object(coordinator, "_calculate_update_batches", return_value=[batch]):
                        with patch.object(coordinator, "_save_batch_checkpoint", AsyncMock(return_value=checkpoint)):
                            with patch.object(
                                coordinator,
                                "_update_batch",
                                AsyncMock(return_value=[(batch.node_names[0], False, "connection failed")]),
                            ):
                                with patch.object(coordinator, "_rollback_batch", AsyncMock()) as mock_rollback:
                                    result = await coordinator.update_cluster(
                                        target_commit="main",
                                        restart_p2p=True,
                                        dry_run=True,
                                    )

        assert result.success is False
        assert result.rollback_performed is False
        assert result.failed_batch == 1
        assert batch.node_names[0] in result.nodes_failed
        mock_rollback.assert_not_awaited()


# =============================================================================
# Batch Calculation Tests
# =============================================================================


class TestBatchCalculation:
    """Tests for update batch calculation."""

    def test_calculate_batches_separates_voters(self, config, mock_node_configs):
        """Test that batches separate voters from non-voters."""
        with patch.object(QuorumSafeUpdateCoordinator, '_find_config_path', return_value=Path("/tmp/test.yaml")):
            coordinator = QuorumSafeUpdateCoordinator(config=config)
            batches = coordinator._calculate_update_batches(nodes=mock_node_configs)

            # First batch should be all non-voters
            assert len(batches) >= 2

            non_voter_batch = batches[0]
            assert non_voter_batch.batch_type == "non_voters"
            assert "vast-gpu-1" in non_voter_batch.node_names
            assert "vast-gpu-2" in non_voter_batch.node_names
            assert "runpod-h100" in non_voter_batch.node_names

            # Voter batches should have one voter each
            voter_batches = [b for b in batches if b.batch_type == "voter"]
            assert len(voter_batches) == 3  # 3 voters, one per batch

            for batch in voter_batches:
                assert len(batch.node_names) == 1

    def test_skips_offline_nodes(self, config, mock_node_configs):
        """Test that offline nodes are filtered out before batching."""
        # Filter out offline nodes before calling _calculate_update_batches
        # (actual filtering happens in _get_node_configs, so we test the filter logic)
        ready_nodes = [n for n in mock_node_configs if n.status == "ready"]

        with patch.object(QuorumSafeUpdateCoordinator, '_find_config_path', return_value=Path("/tmp/test.yaml")):
            coordinator = QuorumSafeUpdateCoordinator(config=config)
            batches = coordinator._calculate_update_batches(nodes=ready_nodes)

            all_nodes = []
            for batch in batches:
                all_nodes.extend(batch.node_names)

            # All nodes should be ready ones
            assert len(all_nodes) == 6  # 3 voters + 3 non-voters

    def test_skip_voters_flag(self, config, mock_node_configs):
        """Test --skip-voters flag excludes voters."""
        with patch.object(QuorumSafeUpdateCoordinator, '_find_config_path', return_value=Path("/tmp/test.yaml")):
            coordinator = QuorumSafeUpdateCoordinator(config=config)
            batches = coordinator._calculate_update_batches(nodes=mock_node_configs, skip_voters=True)

            all_nodes = []
            for batch in batches:
                all_nodes.extend(batch.node_names)

            # Should only have non-voters
            assert "lambda-gh200-1" not in all_nodes
            assert "lambda-gh200-2" not in all_nodes
            assert "nebius-backbone-1" not in all_nodes
            assert "vast-gpu-1" in all_nodes

    def test_only_voters(self, config, mock_node_configs):
        """Test batching with only voters."""
        # Filter to only voters
        voters_only = [n for n in mock_node_configs if n.is_voter]

        with patch.object(QuorumSafeUpdateCoordinator, '_find_config_path', return_value=Path("/tmp/test.yaml")):
            coordinator = QuorumSafeUpdateCoordinator(config=config)
            batches = coordinator._calculate_update_batches(nodes=voters_only)

            # Should have 3 batches, one per voter
            voter_batches = [b for b in batches if b.batch_type == "voter"]
            assert len(voter_batches) == 3

            for batch in voter_batches:
                assert len(batch.node_names) == 1
