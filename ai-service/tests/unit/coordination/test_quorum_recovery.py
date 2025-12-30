"""Tests for QuorumRecoveryManager.

December 30, 2025: Tests for quorum-aware reconnection prioritization.
"""

import asyncio
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from app.coordination.quorum_recovery import (
    QuorumRecoveryManager,
    QuorumRecoveryConfig,
    ReconnectionPriority,
    NodeReconnectionInfo,
    get_quorum_manager,
    reset_quorum_manager,
)


@pytest.fixture(autouse=True)
def reset_singleton() -> None:
    """Reset singleton before each test."""
    reset_quorum_manager()
    yield
    reset_quorum_manager()


class TestReconnectionPriority:
    """Tests for ReconnectionPriority enum."""

    def test_priority_ordering(self) -> None:
        """Test that priorities are ordered correctly (lower = higher priority)."""
        assert ReconnectionPriority.VOTER_QUORUM_NEEDED < ReconnectionPriority.VOTER_EXTRA
        assert ReconnectionPriority.VOTER_EXTRA < ReconnectionPriority.GPU_HIGH
        assert ReconnectionPriority.GPU_HIGH < ReconnectionPriority.GPU_STANDARD
        assert ReconnectionPriority.GPU_STANDARD < ReconnectionPriority.CPU_ONLY

    def test_voter_highest_priority(self) -> None:
        """Test that voter needing quorum has highest priority."""
        assert ReconnectionPriority.VOTER_QUORUM_NEEDED.value == 0


class TestQuorumRecoveryConfig:
    """Tests for QuorumRecoveryConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = QuorumRecoveryConfig()
        assert config.quorum_size == 3
        assert config.max_concurrent_reconnections == 3
        assert config.batch_delay == 1.0
        assert config.max_retries_per_node == 3
        assert config.reconnection_timeout == 30.0

    def test_high_end_gpus(self) -> None:
        """Test high-end GPU set."""
        config = QuorumRecoveryConfig()
        assert "h100" in config.high_end_gpus
        assert "a100" in config.high_end_gpus
        assert "gh200" in config.high_end_gpus
        assert "l40s" in config.high_end_gpus
        assert "rtx4090" in config.high_end_gpus

    def test_from_defaults(self) -> None:
        """Test loading from coordination defaults."""
        # Should not raise even if defaults module has issues
        config = QuorumRecoveryConfig.from_defaults()
        assert config.quorum_size > 0


class TestQuorumRecoveryManager:
    """Tests for QuorumRecoveryManager."""

    # =========================================================================
    # Voter Management
    # =========================================================================

    def test_set_voters(self) -> None:
        """Test setting voter list."""
        manager = QuorumRecoveryManager()
        manager.set_voters(["voter-1", "voter-2", "voter-3"])

        assert "voter-1" in manager._voter_ids
        assert "voter-2" in manager._voter_ids
        assert "voter-3" in manager._voter_ids
        assert len(manager._voter_ids) == 3

    def test_set_voters_from_set(self) -> None:
        """Test setting voters from a set."""
        manager = QuorumRecoveryManager()
        manager.set_voters({"voter-a", "voter-b"})
        assert len(manager._voter_ids) == 2

    def test_update_online_voters(self) -> None:
        """Test updating online voters."""
        manager = QuorumRecoveryManager()
        manager.set_voters(["voter-1", "voter-2", "voter-3"])
        manager.update_online_voters(["voter-1", "voter-2"])

        assert len(manager._online_voters) == 2
        assert "voter-1" in manager._online_voters
        assert "voter-3" not in manager._online_voters

    def test_update_online_voters_filters_non_voters(self) -> None:
        """Test that update_online_voters only includes actual voters."""
        manager = QuorumRecoveryManager()
        manager.set_voters(["voter-1", "voter-2"])
        # Include a non-voter in the online list
        manager.update_online_voters(["voter-1", "not-a-voter", "voter-2"])

        assert len(manager._online_voters) == 2
        assert "not-a-voter" not in manager._online_voters

    def test_register_node_gpu(self) -> None:
        """Test registering node GPU info."""
        manager = QuorumRecoveryManager()
        manager.register_node_gpu("node-1", "H100")
        manager.register_node_gpu("node-2", "RTX4090")

        assert manager._node_gpu_info["node-1"] == "h100"  # Lowercased
        assert manager._node_gpu_info["node-2"] == "rtx4090"

    # =========================================================================
    # Quorum State
    # =========================================================================

    def test_has_quorum_true(self) -> None:
        """Test has_quorum when quorum is met."""
        config = QuorumRecoveryConfig(quorum_size=3)
        manager = QuorumRecoveryManager(config)
        manager.set_voters(["v1", "v2", "v3", "v4"])
        manager.update_online_voters(["v1", "v2", "v3"])

        assert manager.has_quorum()

    def test_has_quorum_false(self) -> None:
        """Test has_quorum when quorum is not met."""
        config = QuorumRecoveryConfig(quorum_size=3)
        manager = QuorumRecoveryManager(config)
        manager.set_voters(["v1", "v2", "v3", "v4"])
        manager.update_online_voters(["v1", "v2"])

        assert not manager.has_quorum()

    def test_needs_more_voters(self) -> None:
        """Test needs_more_voters detection."""
        config = QuorumRecoveryConfig(quorum_size=3)
        manager = QuorumRecoveryManager(config)
        manager.set_voters(["v1", "v2", "v3", "v4"])

        manager.update_online_voters(["v1"])
        assert manager.needs_more_voters()

        manager.update_online_voters(["v1", "v2", "v3"])
        assert not manager.needs_more_voters()

    def test_voters_needed_for_quorum(self) -> None:
        """Test voters_needed_for_quorum calculation."""
        config = QuorumRecoveryConfig(quorum_size=3)
        manager = QuorumRecoveryManager(config)
        manager.set_voters(["v1", "v2", "v3", "v4"])

        manager.update_online_voters([])
        assert manager.voters_needed_for_quorum() == 3

        manager.update_online_voters(["v1"])
        assert manager.voters_needed_for_quorum() == 2

        manager.update_online_voters(["v1", "v2", "v3"])
        assert manager.voters_needed_for_quorum() == 0

        manager.update_online_voters(["v1", "v2", "v3", "v4"])
        assert manager.voters_needed_for_quorum() == 0

    # =========================================================================
    # Priority Calculation
    # =========================================================================

    def test_priority_voter_quorum_needed(self) -> None:
        """Test priority for voter when quorum is needed."""
        config = QuorumRecoveryConfig(quorum_size=3)
        manager = QuorumRecoveryManager(config)
        manager.set_voters(["voter-1", "voter-2", "voter-3"])
        manager.update_online_voters(["voter-1"])  # Only 1 online, need 2 more

        priority = manager.get_reconnection_priority("voter-2")
        assert priority == ReconnectionPriority.VOTER_QUORUM_NEEDED

    def test_priority_voter_extra(self) -> None:
        """Test priority for voter when quorum is met."""
        config = QuorumRecoveryConfig(quorum_size=2)
        manager = QuorumRecoveryManager(config)
        manager.set_voters(["voter-1", "voter-2", "voter-3"])
        manager.update_online_voters(["voter-1", "voter-2"])  # Quorum met

        priority = manager.get_reconnection_priority("voter-3")
        assert priority == ReconnectionPriority.VOTER_EXTRA

    def test_priority_gpu_high(self) -> None:
        """Test priority for high-end GPU node."""
        manager = QuorumRecoveryManager()
        manager.register_node_gpu("gpu-node", "h100")

        priority = manager.get_reconnection_priority("gpu-node")
        assert priority == ReconnectionPriority.GPU_HIGH

    def test_priority_gpu_standard(self) -> None:
        """Test priority for standard GPU node."""
        manager = QuorumRecoveryManager()
        manager.register_node_gpu("gpu-node", "rtx3090")

        priority = manager.get_reconnection_priority("gpu-node")
        assert priority == ReconnectionPriority.GPU_STANDARD

    def test_priority_cpu_only(self) -> None:
        """Test priority for CPU-only node."""
        manager = QuorumRecoveryManager()
        # Node with no GPU info registered

        priority = manager.get_reconnection_priority("cpu-node")
        assert priority == ReconnectionPriority.CPU_ONLY

    # =========================================================================
    # Prioritized Reconnection Order
    # =========================================================================

    def test_get_prioritized_order_voters_first(self) -> None:
        """Test that voters needing quorum come first."""
        config = QuorumRecoveryConfig(quorum_size=3)
        manager = QuorumRecoveryManager(config)
        manager.set_voters(["voter-1", "voter-2"])
        manager.update_online_voters([])  # No voters online
        manager.register_node_gpu("gpu-h100", "h100")
        manager.register_node_gpu("gpu-3090", "rtx3090")

        offline_nodes = ["gpu-h100", "voter-2", "cpu-node", "voter-1", "gpu-3090"]
        prioritized = manager.get_prioritized_reconnection_order(offline_nodes)

        # Voters should come first
        assert prioritized[0] in ["voter-1", "voter-2"]
        assert prioritized[1] in ["voter-1", "voter-2"]
        # Then high-end GPU
        assert prioritized[2] == "gpu-h100"
        # Then standard GPU
        assert prioritized[3] == "gpu-3090"
        # CPU last
        assert prioritized[4] == "cpu-node"

    def test_get_prioritized_order_empty_list(self) -> None:
        """Test with empty offline nodes list."""
        manager = QuorumRecoveryManager()
        prioritized = manager.get_prioritized_reconnection_order([])
        assert prioritized == []

    def test_get_prioritized_order_preserves_all_nodes(self) -> None:
        """Test that all nodes are in the output."""
        manager = QuorumRecoveryManager()
        offline_nodes = {"node-1", "node-2", "node-3"}
        prioritized = manager.get_prioritized_reconnection_order(offline_nodes)

        assert set(prioritized) == offline_nodes

    # =========================================================================
    # Reconnection Processing
    # =========================================================================

    @pytest.mark.asyncio
    async def test_process_reconnections_success(self) -> None:
        """Test successful reconnection processing."""
        config = QuorumRecoveryConfig(
            quorum_size=2,
            max_concurrent_reconnections=2,
            batch_delay=0.01,
        )
        manager = QuorumRecoveryManager(config)

        # Mock reconnect function that always succeeds
        reconnect_fn = AsyncMock(return_value=True)

        offline_nodes = ["node-1", "node-2", "node-3"]
        reconnected = []

        async for batch in manager.process_reconnections(offline_nodes, reconnect_fn):
            reconnected.extend(batch)

        assert len(reconnected) == 3
        assert reconnect_fn.call_count == 3

    @pytest.mark.asyncio
    async def test_process_reconnections_partial_failure(self) -> None:
        """Test reconnection with some failures."""
        config = QuorumRecoveryConfig(
            quorum_size=2,
            max_concurrent_reconnections=2,
            batch_delay=0.01,
        )
        manager = QuorumRecoveryManager(config)

        # Mock reconnect that fails for node-2
        async def reconnect_fn(node_id: str) -> bool:
            return node_id != "node-2"

        offline_nodes = ["node-1", "node-2", "node-3"]
        reconnected = []

        async for batch in manager.process_reconnections(offline_nodes, reconnect_fn):
            reconnected.extend(batch)

        assert len(reconnected) == 2
        assert "node-2" not in reconnected
        assert manager._stats["reconnections_succeeded"] == 2
        assert manager._stats["reconnections_failed"] == 1

    @pytest.mark.asyncio
    async def test_process_reconnections_timeout(self) -> None:
        """Test reconnection with timeout."""
        config = QuorumRecoveryConfig(
            quorum_size=2,
            max_concurrent_reconnections=1,
            reconnection_timeout=0.1,
            batch_delay=0.01,
        )
        manager = QuorumRecoveryManager(config)

        # Mock reconnect that times out
        async def slow_reconnect(node_id: str) -> bool:
            await asyncio.sleep(1.0)  # Longer than timeout
            return True

        offline_nodes = ["node-1"]
        reconnected = []

        async for batch in manager.process_reconnections(offline_nodes, slow_reconnect):
            reconnected.extend(batch)

        assert len(reconnected) == 0
        assert manager._stats["reconnections_failed"] == 1

    @pytest.mark.asyncio
    async def test_process_reconnections_updates_voter_status(self) -> None:
        """Test that reconnecting voters updates online voters."""
        config = QuorumRecoveryConfig(
            quorum_size=2,
            max_concurrent_reconnections=2,
            batch_delay=0.01,
        )
        manager = QuorumRecoveryManager(config)
        manager.set_voters(["voter-1", "voter-2"])
        manager.update_online_voters([])

        assert not manager.has_quorum()

        reconnect_fn = AsyncMock(return_value=True)
        offline_nodes = ["voter-1", "voter-2"]

        async for batch in manager.process_reconnections(offline_nodes, reconnect_fn):
            pass

        assert manager.has_quorum()
        assert manager._stats["quorum_restorations"] >= 1

    # =========================================================================
    # Status and Health
    # =========================================================================

    def test_get_status(self) -> None:
        """Test get_status method."""
        config = QuorumRecoveryConfig(quorum_size=3)
        manager = QuorumRecoveryManager(config)
        manager.set_voters(["v1", "v2", "v3"])
        manager.update_online_voters(["v1", "v2"])

        status = manager.get_status()

        assert status["has_quorum"] is False
        assert status["online_voters"] == 2
        assert status["total_voters"] == 3
        assert status["quorum_size"] == 3
        assert status["voters_needed"] == 1
        assert "stats" in status

    def test_health_check_healthy(self) -> None:
        """Test health check when quorum is met."""
        config = QuorumRecoveryConfig(quorum_size=2)
        manager = QuorumRecoveryManager(config)
        manager.set_voters(["v1", "v2", "v3"])
        manager.update_online_voters(["v1", "v2"])

        health = manager.health_check()

        # Check if it's a HealthCheckResult or dict
        if hasattr(health, "healthy"):
            assert health.healthy
        else:
            assert health.get("healthy") is True

    def test_health_check_degraded(self) -> None:
        """Test health check when quorum is not met."""
        config = QuorumRecoveryConfig(quorum_size=3)
        manager = QuorumRecoveryManager(config)
        manager.set_voters(["v1", "v2", "v3"])
        manager.update_online_voters(["v1"])

        health = manager.health_check()

        # Check if it's a HealthCheckResult or dict
        if hasattr(health, "healthy"):
            # HealthCheckResult.degraded() sets healthy=True but status="degraded"
            assert health.status == "degraded"
        else:
            assert health.get("status") == "degraded"


class TestSingletonAccessors:
    """Tests for singleton accessor functions."""

    def test_get_quorum_manager_singleton(self) -> None:
        """Test that get_quorum_manager returns singleton."""
        manager1 = get_quorum_manager()
        manager2 = get_quorum_manager()

        assert manager1 is manager2

    def test_reset_quorum_manager(self) -> None:
        """Test that reset creates new instance."""
        manager1 = get_quorum_manager()
        reset_quorum_manager()
        manager2 = get_quorum_manager()

        assert manager1 is not manager2

    def test_get_quorum_manager_auto_loads_config(self) -> None:
        """Test that get_quorum_manager tries to load cluster config."""
        with patch("app.coordination.quorum_recovery.QuorumRecoveryManager.load_from_cluster_config") as mock_load:
            reset_quorum_manager()
            manager = get_quorum_manager()
            # Should attempt to load config on first access
            mock_load.assert_called_once()
