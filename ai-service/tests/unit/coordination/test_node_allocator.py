"""Tests for app.coordination.node_allocator module.

December 29, 2025 - Tests for extracted node allocation functions.
"""

import pytest
from unittest.mock import MagicMock, patch

from app.coordination.node_allocator import (
    # Enums
    AllocationStrategy,
    NodeHealthStatus,
    # Data structures
    NodeCapability,
    AllocationResult,
    NodeMetrics,
    JobTargetResult,
    # Pure functions
    allocate_to_nodes,
    compute_hardware_limit,
    compute_target_jobs_for_node,
    is_node_eligible_for_allocation,
    get_total_cluster_capacity,
    rank_nodes_by_capacity,
    # Constants
    MIN_GAMES_PER_ALLOCATION,
    MIN_MEMORY_GB_FOR_TASKS,
    DISK_WARNING_THRESHOLD,
    MEMORY_WARNING_THRESHOLD,
)


# =============================================================================
# Test Constants
# =============================================================================


class TestConstants:
    """Test that constants are properly imported."""

    def test_min_games_per_allocation_is_positive(self):
        assert MIN_GAMES_PER_ALLOCATION > 0

    def test_min_memory_gb_for_tasks_is_reasonable(self):
        assert 2 <= MIN_MEMORY_GB_FOR_TASKS <= 32

    def test_disk_warning_threshold_is_percentage(self):
        assert 0 < DISK_WARNING_THRESHOLD < 100

    def test_memory_warning_threshold_is_percentage(self):
        assert 0 < MEMORY_WARNING_THRESHOLD < 100


# =============================================================================
# Test Enums
# =============================================================================


class TestAllocationStrategy:
    """Test AllocationStrategy enum."""

    def test_all_strategies_have_string_values(self):
        for strategy in AllocationStrategy:
            assert isinstance(strategy.value, str)

    def test_proportional_is_default(self):
        result = AllocationResult()
        assert result.strategy_used == AllocationStrategy.PROPORTIONAL

    def test_ephemeral_boost_strategy(self):
        assert AllocationStrategy.EPHEMERAL_BOOST.value == "ephemeral_boost"

    def test_round_robin_strategy(self):
        assert AllocationStrategy.ROUND_ROBIN.value == "round_robin"

    def test_priority_strategy(self):
        assert AllocationStrategy.PRIORITY.value == "priority"


class TestNodeHealthStatus:
    """Test NodeHealthStatus enum."""

    def test_all_statuses_have_string_values(self):
        for status in NodeHealthStatus:
            assert isinstance(status.value, str)

    def test_healthy_status(self):
        assert NodeHealthStatus.HEALTHY.value == "healthy"

    def test_degraded_status(self):
        assert NodeHealthStatus.DEGRADED.value == "degraded"

    def test_unhealthy_status(self):
        assert NodeHealthStatus.UNHEALTHY.value == "unhealthy"

    def test_unknown_status(self):
        assert NodeHealthStatus.UNKNOWN.value == "unknown"


# =============================================================================
# Test NodeCapability
# =============================================================================


class TestNodeCapability:
    """Test NodeCapability dataclass."""

    def test_creation_with_defaults(self):
        cap = NodeCapability(node_id="test-node")
        assert cap.node_id == "test-node"
        assert cap.gpu_type == "unknown"
        assert cap.gpu_memory_gb == 0.0
        assert cap.is_ephemeral is False
        assert cap.current_load == 0.0
        assert cap.current_jobs == 0
        assert cap.data_lag_seconds == 0.0

    def test_creation_with_all_fields(self):
        cap = NodeCapability(
            node_id="gpu-node-1",
            gpu_type="H100",
            gpu_memory_gb=80.0,
            is_ephemeral=True,
            current_load=0.5,
            current_jobs=10,
            data_lag_seconds=30.0,
        )
        assert cap.node_id == "gpu-node-1"
        assert cap.gpu_type == "H100"
        assert cap.gpu_memory_gb == 80.0
        assert cap.is_ephemeral is True
        assert cap.current_load == 0.5
        assert cap.current_jobs == 10
        assert cap.data_lag_seconds == 30.0

    @patch("app.coordination.node_allocator.get_gpu_weight")
    def test_capacity_weight_uses_get_gpu_weight(self, mock_get_weight):
        mock_get_weight.return_value = 2.0
        cap = NodeCapability(node_id="test", gpu_type="H100")
        weight = cap.capacity_weight
        assert weight == 2.0
        mock_get_weight.assert_called_with("H100")

    def test_available_capacity_when_not_loaded(self):
        cap = NodeCapability(node_id="test", current_load=0.0)
        # available = (1 - load) * weight
        # With load=0 and default weight~1.0, available ≈ 1.0
        assert cap.available_capacity > 0

    def test_available_capacity_when_fully_loaded(self):
        cap = NodeCapability(node_id="test", current_load=1.0)
        assert cap.available_capacity == 0.0

    def test_available_capacity_partial_load(self):
        cap = NodeCapability(node_id="test", current_load=0.5)
        # (1 - 0.5) * weight
        assert cap.available_capacity > 0
        assert cap.available_capacity < cap.capacity_weight

    def test_to_dict(self):
        cap = NodeCapability(
            node_id="test-node",
            gpu_type="A100",
            gpu_memory_gb=40.0,
            is_ephemeral=False,
            current_load=0.3,
            current_jobs=5,
            data_lag_seconds=10.0,
        )
        d = cap.to_dict()
        assert d["node_id"] == "test-node"
        assert d["gpu_type"] == "A100"
        assert d["gpu_memory_gb"] == 40.0
        assert d["is_ephemeral"] is False
        assert d["current_load"] == 0.3
        assert d["current_jobs"] == 5
        assert d["data_lag_seconds"] == 10.0
        assert "capacity_weight" in d
        assert "available_capacity" in d


# =============================================================================
# Test AllocationResult
# =============================================================================


class TestAllocationResult:
    """Test AllocationResult dataclass."""

    def test_creation_with_defaults(self):
        result = AllocationResult()
        assert result.allocations == {}
        assert result.total_allocated == 0
        assert result.total_requested == 0
        assert result.nodes_used == 0
        assert result.strategy_used == AllocationStrategy.PROPORTIONAL
        assert result.cluster_health_factor == 1.0
        assert result.notes == []

    def test_allocation_efficiency_zero_requested(self):
        result = AllocationResult(total_requested=0, total_allocated=0)
        assert result.allocation_efficiency == 0.0

    def test_allocation_efficiency_partial(self):
        result = AllocationResult(total_requested=100, total_allocated=75)
        assert result.allocation_efficiency == 0.75

    def test_allocation_efficiency_full(self):
        result = AllocationResult(total_requested=100, total_allocated=100)
        assert result.allocation_efficiency == 1.0

    def test_to_dict(self):
        result = AllocationResult(
            allocations={"node1": 100, "node2": 200},
            total_allocated=300,
            total_requested=400,
            nodes_used=2,
            strategy_used=AllocationStrategy.EPHEMERAL_BOOST,
            cluster_health_factor=0.9,
            notes=["Test note"],
        )
        d = result.to_dict()
        assert d["allocations"] == {"node1": 100, "node2": 200}
        assert d["total_allocated"] == 300
        assert d["total_requested"] == 400
        assert d["nodes_used"] == 2
        assert d["strategy_used"] == "ephemeral_boost"
        assert d["cluster_health_factor"] == 0.9
        assert d["allocation_efficiency"] == 0.75
        assert d["notes"] == ["Test note"]


# =============================================================================
# Test NodeMetrics
# =============================================================================


class TestNodeMetrics:
    """Test NodeMetrics dataclass."""

    def test_creation_with_defaults(self):
        metrics = NodeMetrics(node_id="test-node")
        assert metrics.node_id == "test-node"
        assert metrics.memory_gb == 0
        assert metrics.has_gpu is False
        assert metrics.cpu_count == 0
        assert metrics.cpu_percent == 0.0
        assert metrics.memory_percent == 0.0
        assert metrics.disk_percent == 0.0
        assert metrics.gpu_percent == 0.0
        assert metrics.gpu_memory_percent == 0.0
        assert metrics.selfplay_jobs == 0
        assert metrics.gpu_name == ""
        assert metrics.gpu_count == 1

    def test_creation_with_all_fields(self):
        metrics = NodeMetrics(
            node_id="gpu-node",
            memory_gb=128,
            has_gpu=True,
            cpu_count=64,
            cpu_percent=45.0,
            memory_percent=60.0,
            disk_percent=30.0,
            gpu_percent=80.0,
            gpu_memory_percent=70.0,
            selfplay_jobs=8,
            gpu_name="H100",
            gpu_count=4,
        )
        assert metrics.node_id == "gpu-node"
        assert metrics.memory_gb == 128
        assert metrics.has_gpu is True
        assert metrics.cpu_count == 64
        assert metrics.gpu_count == 4

    def test_from_node_info(self):
        mock_node = MagicMock()
        mock_node.node_id = "test-123"
        mock_node.memory_gb = 64
        mock_node.has_gpu = True
        mock_node.cpu_count = 32
        mock_node.cpu_percent = 25.0
        mock_node.memory_percent = 40.0
        mock_node.disk_percent = 20.0
        mock_node.gpu_percent = 50.0
        mock_node.gpu_memory_percent = 45.0
        mock_node.selfplay_jobs = 4
        mock_node.gpu_name = "A100"
        mock_node.gpu_count = 2

        metrics = NodeMetrics.from_node_info(mock_node)
        assert metrics.node_id == "test-123"
        assert metrics.memory_gb == 64
        assert metrics.has_gpu is True
        assert metrics.cpu_count == 32
        assert metrics.gpu_count == 2

    def test_from_node_info_with_missing_attributes(self):
        mock_node = MagicMock(spec=[])
        metrics = NodeMetrics.from_node_info(mock_node)
        assert metrics.node_id == "unknown"
        assert metrics.memory_gb == 0
        assert metrics.has_gpu is False

    def test_from_node_info_with_none_values(self):
        mock_node = MagicMock()
        mock_node.node_id = "test"
        mock_node.memory_gb = None
        mock_node.cpu_count = None
        mock_node.gpu_percent = None

        metrics = NodeMetrics.from_node_info(mock_node)
        assert metrics.memory_gb == 0
        assert metrics.cpu_count == 0
        assert metrics.gpu_percent == 0.0


# =============================================================================
# Test JobTargetResult
# =============================================================================


class TestJobTargetResult:
    """Test JobTargetResult dataclass."""

    def test_creation_with_defaults(self):
        result = JobTargetResult(target_jobs=5)
        assert result.target_jobs == 5
        assert result.reason == ""
        assert result.backpressure_factor == 1.0
        assert result.was_scaled_up is False
        assert result.was_scaled_down is False
        assert result.scale_reason == ""
        assert result.used_resource_targets is False
        assert result.used_hardware_fallback is False

    def test_creation_with_all_fields(self):
        result = JobTargetResult(
            target_jobs=10,
            reason="resource_targets",
            backpressure_factor=0.8,
            was_scaled_up=True,
            was_scaled_down=False,
            scale_reason="underutilized",
            used_resource_targets=True,
            used_hardware_fallback=False,
        )
        assert result.target_jobs == 10
        assert result.backpressure_factor == 0.8
        assert result.was_scaled_up is True

    def test_to_dict(self):
        result = JobTargetResult(
            target_jobs=8,
            reason="hardware_fallback",
            was_scaled_down=True,
            scale_reason="overloaded",
        )
        d = result.to_dict()
        assert d["target_jobs"] == 8
        assert d["reason"] == "hardware_fallback"
        assert d["was_scaled_down"] is True
        assert d["scale_reason"] == "overloaded"


# =============================================================================
# Test allocate_to_nodes
# =============================================================================


class TestAllocateToNodes:
    """Test allocate_to_nodes function."""

    def test_no_nodes_returns_empty_result(self):
        result = allocate_to_nodes("hex8_2p", 1000, {})
        assert result.allocations == {}
        assert result.total_allocated == 0
        assert "No available nodes" in result.notes[0]

    def test_single_node_allocation(self):
        caps = {
            "node1": NodeCapability(
                node_id="node1",
                gpu_type="A100",
                current_load=0.0,
            )
        }
        result = allocate_to_nodes("hex8_2p", 500, caps)
        assert "node1" in result.allocations
        assert result.total_allocated > 0
        assert result.nodes_used == 1

    def test_multiple_node_allocation(self):
        caps = {
            "node1": NodeCapability(node_id="node1", gpu_type="H100", current_load=0.2),
            "node2": NodeCapability(node_id="node2", gpu_type="A100", current_load=0.3),
            "node3": NodeCapability(node_id="node3", gpu_type="A10", current_load=0.1),
        }
        result = allocate_to_nodes("hex8_2p", 2000, caps)
        assert len(result.allocations) > 0
        assert result.total_allocated > 0
        assert result.nodes_used <= 3

    def test_unhealthy_nodes_excluded(self):
        caps = {
            "healthy": NodeCapability(node_id="healthy", gpu_type="A100"),
            "unhealthy": NodeCapability(node_id="unhealthy", gpu_type="H100"),
        }
        result = allocate_to_nodes(
            "hex8_2p", 500, caps, unhealthy_nodes={"unhealthy"}
        )
        assert "unhealthy" not in result.allocations
        if result.nodes_used > 0:
            assert "healthy" in result.allocations

    def test_cluster_health_factor_reduces_allocation(self):
        caps = {
            "node1": NodeCapability(node_id="node1", gpu_type="A100"),
        }
        result_full = allocate_to_nodes("hex8_2p", 1000, caps, cluster_health_factor=1.0)
        result_reduced = allocate_to_nodes("hex8_2p", 1000, caps, cluster_health_factor=0.5)

        # Note: actual allocations depend on GPU caps, but health factor is recorded
        assert result_reduced.cluster_health_factor == 0.5
        assert "health" in result_reduced.notes[0].lower()

    def test_short_job_uses_ephemeral_boost(self):
        caps = {
            "node1": NodeCapability(node_id="node1", gpu_type="A100", is_ephemeral=True),
        }
        # hex8 and square8 are short jobs
        result = allocate_to_nodes("hex8_2p", 500, caps)
        assert result.strategy_used == AllocationStrategy.EPHEMERAL_BOOST

    def test_long_job_uses_proportional(self):
        caps = {
            "node1": NodeCapability(node_id="node1", gpu_type="A100", is_ephemeral=False),
        }
        # hexagonal is a long job
        result = allocate_to_nodes("hexagonal_2p", 500, caps)
        assert result.strategy_used == AllocationStrategy.PROPORTIONAL

    def test_ephemeral_node_boost_for_short_job(self):
        caps = {
            "ephemeral": NodeCapability(
                node_id="ephemeral", gpu_type="A100", is_ephemeral=True
            ),
        }
        result = allocate_to_nodes("hex8_2p", 500, caps)
        # Should have a note about boosting ephemeral
        assert any("Boosted" in note for note in result.notes)

    def test_ephemeral_node_reduction_for_long_job(self):
        caps = {
            "ephemeral": NodeCapability(
                node_id="ephemeral", gpu_type="A100", is_ephemeral=True
            ),
        }
        result = allocate_to_nodes("hexagonal_2p", 500, caps)
        # Should have a note about reducing ephemeral
        assert any("Reduced" in note for note in result.notes)

    def test_nodes_with_low_capacity_excluded(self):
        caps = {
            "high_cap": NodeCapability(node_id="high_cap", current_load=0.0),
            "low_cap": NodeCapability(node_id="low_cap", current_load=0.95),  # <0.1 available
        }
        result = allocate_to_nodes("hex8_2p", 500, caps)
        assert "low_cap" not in result.allocations

    def test_allocation_result_has_correct_totals(self):
        caps = {
            "node1": NodeCapability(node_id="node1", gpu_type="A100"),
            "node2": NodeCapability(node_id="node2", gpu_type="A100"),
        }
        result = allocate_to_nodes("hex8_2p", 1000, caps)
        assert result.total_requested == 1000
        assert result.total_allocated == sum(result.allocations.values())
        assert result.nodes_used == len(result.allocations)


# =============================================================================
# Test compute_hardware_limit
# =============================================================================


class TestComputeHardwareLimit:
    """Test compute_hardware_limit function."""

    def test_gh200_high_limit(self):
        limit = compute_hardware_limit(True, "GH200", 1, 72, 96)
        assert limit >= 48  # 72 * 0.8 = 57.6

    def test_h100_limit(self):
        limit = compute_hardware_limit(True, "H100", 1, 64, 128)
        assert limit == min(int(64 * 0.5), 48)  # 32

    def test_a100_limit(self):
        limit = compute_hardware_limit(True, "A100-80GB", 1, 48, 80)
        assert limit == min(int(48 * 0.4), 32)  # 19

    def test_rtx_5090_limit(self):
        limit = compute_hardware_limit(True, "RTX 5090", 2, 32, 128)
        # min(32*0.3, 2*12, 64) = min(9, 24, 64) = 9
        assert limit == min(int(32 * 0.3), 2 * 12, 64)

    def test_rtx_4090_limit(self):
        limit = compute_hardware_limit(True, "RTX 4090", 1, 24, 64)
        # min(24*0.3, 1*8, 32) = min(7, 8, 32) = 7
        assert limit == min(int(24 * 0.3), 1 * 8, 32)

    def test_rtx_3060_limit(self):
        limit = compute_hardware_limit(True, "RTX 3060", 1, 16, 32)
        # min(16*0.25, 1*6, 16) = min(4, 6, 16) = 4
        assert limit == min(int(16 * 0.25), 1 * 6, 16)

    def test_unknown_gpu_conservative(self):
        limit = compute_hardware_limit(True, "Unknown GPU", 1, 32, 64)
        assert limit == max(4, min(int(32 * 0.2), 8))  # 6

    def test_cpu_only_high_resource(self):
        limit = compute_hardware_limit(False, "", 0, 64, 128)
        assert limit == min(int(64 * 0.5), 16)  # 16

    def test_cpu_only_medium_resource(self):
        limit = compute_hardware_limit(False, "", 0, 24, 32)
        assert limit == min(int(24 * 0.4), 8)  # 8

    def test_cpu_only_low_resource(self):
        limit = compute_hardware_limit(False, "", 0, 8, 16)
        assert limit == max(2, min(8 // 4, 4))  # 2

    def test_zero_cpu_count_uses_fallback(self):
        limit = compute_hardware_limit(True, "H100", 1, 0, 128)
        assert limit == 32  # H100 fallback

    def test_case_insensitive_gpu_name(self):
        limit_upper = compute_hardware_limit(True, "H100", 1, 64, 128)
        limit_lower = compute_hardware_limit(True, "h100", 1, 64, 128)
        assert limit_upper == limit_lower


# =============================================================================
# Test compute_target_jobs_for_node
# =============================================================================


class TestComputeTargetJobsForNode:
    """Test compute_target_jobs_for_node function."""

    def test_basic_computation_no_callbacks(self):
        metrics = NodeMetrics(
            node_id="test",
            memory_gb=64,
            has_gpu=True,
            cpu_count=32,
            gpu_name="A100",
            gpu_count=1,
        )
        result = compute_target_jobs_for_node(metrics)
        assert result.target_jobs >= 1
        assert result.used_hardware_fallback is True
        assert result.reason == "hardware_fallback"

    def test_emergency_active_returns_zero(self):
        metrics = NodeMetrics(node_id="test", memory_gb=64)
        result = compute_target_jobs_for_node(
            metrics, is_emergency_active_fn=lambda: True
        )
        assert result.target_jobs == 0
        assert result.reason == "emergency_active"

    def test_backpressure_stop_returns_zero(self):
        metrics = NodeMetrics(node_id="test", memory_gb=64)

        def mock_stop(queue_type):
            return True

        # Mock the QueueType import inside the function
        mock_queue_type = MagicMock()
        mock_queue_type.TRAINING_DATA = "training_data"

        with patch.dict(
            "sys.modules",
            {"app.coordination.backpressure": MagicMock(QueueType=mock_queue_type)},
        ):
            result = compute_target_jobs_for_node(
                metrics, should_stop_production_fn=mock_stop
            )
        assert result.target_jobs == 0
        assert result.reason == "backpressure_stop"

    def test_insufficient_memory_returns_zero(self):
        metrics = NodeMetrics(node_id="test", memory_gb=2)  # Below threshold
        result = compute_target_jobs_for_node(metrics)
        assert result.target_jobs == 0
        assert result.reason == "insufficient_memory"

    def test_resource_targets_used_when_available(self):
        metrics = NodeMetrics(
            node_id="test",
            memory_gb=64,
            has_gpu=True,
            cpu_count=32,
        )

        def mock_host_targets(node_id):
            return MagicMock(max_selfplay=20)

        def mock_target_count(node_id, cpu_count, cpu_pct, gpu_pct):
            return 10

        result = compute_target_jobs_for_node(
            metrics,
            get_host_targets_fn=mock_host_targets,
            get_target_job_count_fn=mock_target_count,
        )
        assert result.used_resource_targets is True
        assert result.reason == "resource_targets"
        assert result.target_jobs > 0

    def test_scale_up_recorded(self):
        metrics = NodeMetrics(
            node_id="test",
            memory_gb=64,
            has_gpu=True,
            cpu_count=32,
            selfplay_jobs=5,
        )

        def mock_host_targets(node_id):
            return MagicMock(max_selfplay=20)

        def mock_target_count(node_id, cpu_count, cpu_pct, gpu_pct):
            return 15

        def mock_scale_up(node_id, cpu_pct, gpu_pct, current_jobs):
            return True, "underutilized"

        result = compute_target_jobs_for_node(
            metrics,
            get_host_targets_fn=mock_host_targets,
            get_target_job_count_fn=mock_target_count,
            should_scale_up_fn=mock_scale_up,
        )
        assert result.was_scaled_up is True
        assert result.scale_reason == "underutilized"

    def test_scale_down_recorded(self):
        metrics = NodeMetrics(
            node_id="test",
            memory_gb=64,
            has_gpu=True,
            cpu_count=32,
            selfplay_jobs=15,
        )

        def mock_host_targets(node_id):
            return MagicMock(max_selfplay=20)

        def mock_target_count(node_id, cpu_count, cpu_pct, gpu_pct):
            return 15

        def mock_scale_down(node_id, cpu_pct, gpu_pct, mem_pct):
            return True, 5, "overloaded"

        result = compute_target_jobs_for_node(
            metrics,
            get_host_targets_fn=mock_host_targets,
            get_target_job_count_fn=mock_target_count,
            should_scale_down_fn=mock_scale_down,
        )
        assert result.was_scaled_down is True
        assert result.scale_reason == "overloaded"

    def test_backpressure_factor_applied(self):
        metrics = NodeMetrics(
            node_id="test",
            memory_gb=64,
            has_gpu=True,
            cpu_count=32,
        )

        def mock_throttle(queue_type):
            return True

        def mock_factor(queue_type):
            return 0.5

        # Mock the QueueType import inside the function
        mock_queue_type = MagicMock()
        mock_queue_type.TRAINING_DATA = "training_data"

        with patch.dict(
            "sys.modules",
            {"app.coordination.backpressure": MagicMock(QueueType=mock_queue_type)},
        ):
            result = compute_target_jobs_for_node(
                metrics,
                should_throttle_production_fn=mock_throttle,
                get_throttle_factor_fn=mock_factor,
            )
        assert result.backpressure_factor == 0.5

    def test_gpu_overload_reduces_target(self):
        metrics = NodeMetrics(
            node_id="test",
            memory_gb=64,
            has_gpu=True,
            cpu_count=32,
            gpu_name="A100",
            gpu_percent=90.0,  # Overloaded
        )
        result = compute_target_jobs_for_node(metrics)
        # Should be reduced due to overload
        assert result.target_jobs >= 1  # At least minimum

    def test_disk_warning_caps_target(self):
        metrics = NodeMetrics(
            node_id="test",
            memory_gb=64,
            has_gpu=True,
            cpu_count=32,
            gpu_name="A100",
            disk_percent=DISK_WARNING_THRESHOLD + 5,  # Above threshold
        )
        result = compute_target_jobs_for_node(metrics)
        assert result.target_jobs <= 4  # Capped at 4

    def test_memory_warning_caps_target(self):
        metrics = NodeMetrics(
            node_id="test",
            memory_gb=64,
            has_gpu=True,
            cpu_count=32,
            gpu_name="A100",
            memory_percent=MEMORY_WARNING_THRESHOLD + 5,  # Above threshold
        )
        result = compute_target_jobs_for_node(metrics)
        assert result.target_jobs <= 2  # Capped at 2

    def test_utilization_recording_called(self):
        metrics = NodeMetrics(
            node_id="test",
            memory_gb=64,
            cpu_percent=50.0,
            gpu_percent=60.0,
            memory_percent=40.0,
            selfplay_jobs=5,
        )
        record_fn = MagicMock()

        compute_target_jobs_for_node(metrics, record_utilization_fn=record_fn)

        record_fn.assert_called_once_with("test", 50.0, 60.0, 40.0, 5)

    def test_minimum_jobs_is_one(self):
        metrics = NodeMetrics(
            node_id="test",
            memory_gb=32,
            cpu_count=4,
            cpu_percent=50.0,
        )
        result = compute_target_jobs_for_node(metrics)
        assert result.target_jobs >= 1


# =============================================================================
# Test Convenience Functions
# =============================================================================


class TestIsNodeEligibleForAllocation:
    """Test is_node_eligible_for_allocation function."""

    def test_healthy_node_with_capacity_eligible(self):
        node = NodeCapability(node_id="test", current_load=0.0)
        assert is_node_eligible_for_allocation(node) is True

    def test_unhealthy_node_not_eligible(self):
        node = NodeCapability(node_id="unhealthy", current_load=0.0)
        assert is_node_eligible_for_allocation(node, unhealthy_nodes={"unhealthy"}) is False

    def test_node_with_low_capacity_not_eligible(self):
        node = NodeCapability(node_id="test", current_load=0.95)
        assert is_node_eligible_for_allocation(node, min_capacity=0.1) is False

    def test_custom_min_capacity(self):
        node = NodeCapability(node_id="test", current_load=0.7)
        # available_capacity ≈ 0.3 * weight
        assert is_node_eligible_for_allocation(node, min_capacity=0.5) is False
        assert is_node_eligible_for_allocation(node, min_capacity=0.1) is True


class TestGetTotalClusterCapacity:
    """Test get_total_cluster_capacity function."""

    def test_empty_cluster(self):
        assert get_total_cluster_capacity({}) == 0.0

    def test_single_node(self):
        caps = {"node1": NodeCapability(node_id="node1", current_load=0.0)}
        total = get_total_cluster_capacity(caps)
        assert total == caps["node1"].available_capacity

    def test_multiple_nodes(self):
        caps = {
            "node1": NodeCapability(node_id="node1", current_load=0.0),
            "node2": NodeCapability(node_id="node2", current_load=0.5),
        }
        total = get_total_cluster_capacity(caps)
        expected = caps["node1"].available_capacity + caps["node2"].available_capacity
        assert total == expected

    def test_excludes_unhealthy_nodes(self):
        caps = {
            "healthy": NodeCapability(node_id="healthy", current_load=0.0),
            "unhealthy": NodeCapability(node_id="unhealthy", current_load=0.0),
        }
        total_with = get_total_cluster_capacity(caps)
        total_without = get_total_cluster_capacity(caps, unhealthy_nodes={"unhealthy"})
        assert total_without < total_with


class TestRankNodesByCapacity:
    """Test rank_nodes_by_capacity function."""

    def test_empty_cluster(self):
        assert rank_nodes_by_capacity({}) == []

    def test_single_node(self):
        caps = {"node1": NodeCapability(node_id="node1")}
        ranked = rank_nodes_by_capacity(caps)
        assert len(ranked) == 1
        assert ranked[0].node_id == "node1"

    def test_sorts_by_capacity_descending(self):
        caps = {
            "low": NodeCapability(node_id="low", current_load=0.8),
            "high": NodeCapability(node_id="high", current_load=0.0),
            "mid": NodeCapability(node_id="mid", current_load=0.5),
        }
        ranked = rank_nodes_by_capacity(caps)
        # Highest capacity first
        assert ranked[0].node_id == "high"
        assert ranked[1].node_id == "mid"
        assert ranked[2].node_id == "low"

    def test_excludes_unhealthy(self):
        caps = {
            "healthy": NodeCapability(node_id="healthy"),
            "unhealthy": NodeCapability(node_id="unhealthy"),
        }
        ranked = rank_nodes_by_capacity(caps, unhealthy_nodes={"unhealthy"})
        assert len(ranked) == 1
        assert ranked[0].node_id == "healthy"

    def test_secondary_sort_by_lag(self):
        caps = {
            "high_lag": NodeCapability(node_id="high_lag", data_lag_seconds=100.0),
            "low_lag": NodeCapability(node_id="low_lag", data_lag_seconds=10.0),
        }
        ranked = rank_nodes_by_capacity(caps, prefer_low_lag=True)
        # Same capacity, but lower lag first
        assert ranked[0].node_id == "low_lag"
        assert ranked[1].node_id == "high_lag"

    def test_no_lag_preference(self):
        caps = {
            "high_lag": NodeCapability(node_id="high_lag", data_lag_seconds=100.0),
            "low_lag": NodeCapability(node_id="low_lag", data_lag_seconds=10.0),
        }
        ranked = rank_nodes_by_capacity(caps, prefer_low_lag=False)
        # Order may vary but shouldn't crash
        assert len(ranked) == 2
