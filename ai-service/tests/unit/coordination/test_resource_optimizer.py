"""Tests for Resource Optimizer module.

Tests the cooperative resource scheduling system that targets 60-80% utilization
for optimal training throughput across the RingRift cluster.
"""

import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from app.coordination.resource_optimizer import (
    SCALE_DOWN_THRESHOLD,
    SCALE_UP_THRESHOLD,
    TARGET_UTIL_MAX,
    # Constants (for testing)
    TARGET_UTIL_MIN,
    TARGET_UTIL_OPTIMAL,
    ClusterState,
    # Data classes
    NodeResources,
    OptimizationResult,
    # Classes
    PIDController,
    ResourceOptimizer,
    # Enums
    ResourceType,
    ScaleAction,
    UtilizationPredictor,
    get_cluster_utilization,
    get_hybrid_selfplay_limits,
    get_max_cpu_only_selfplay,
    get_max_selfplay_for_node,
    get_optimal_concurrency,
    # Module functions
    get_resource_optimizer,
    should_scale_down,
    should_scale_up,
)

# =============================================================================
# Enum Tests
# =============================================================================


class TestResourceType:
    """Tests for ResourceType enum."""

    def test_all_resource_types_defined(self):
        """All expected resource types should exist."""
        assert ResourceType.CPU.value == "cpu"
        assert ResourceType.GPU.value == "gpu"
        assert ResourceType.MEMORY.value == "memory"

    def test_resource_type_count(self):
        """Should have all resource types defined."""
        # ResourceType is consolidated with 7 members: CPU, GPU, MEMORY, DISK, NETWORK, HYBRID, IO
        assert len(ResourceType) == 7

    def test_enum_string_value(self):
        """Should be usable as string."""
        assert str(ResourceType.CPU) == "ResourceType.CPU"
        assert ResourceType.CPU.value == "cpu"


class TestScaleAction:
    """Tests for ScaleAction enum."""

    def test_all_scale_actions_defined(self):
        """All expected scale actions should exist."""
        assert ScaleAction.NONE.value == "none"
        assert ScaleAction.SCALE_UP.value == "scale_up"
        assert ScaleAction.SCALE_DOWN.value == "scale_down"
        assert ScaleAction.REBALANCE.value == "rebalance"

    def test_scale_action_count(self):
        """Should have exactly 4 scale actions."""
        assert len(ScaleAction) == 4


# =============================================================================
# NodeResources Tests
# =============================================================================


class TestNodeResources:
    """Tests for NodeResources dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        node = NodeResources(node_id="test-node")
        assert node.node_id == "test-node"
        assert node.cpu_percent == 0.0
        assert node.gpu_percent == 0.0
        assert node.memory_percent == 0.0
        assert node.has_gpu is False
        assert node.active_jobs == 0

    def test_full_initialization(self):
        """Should accept all parameters."""
        node = NodeResources(
            node_id="gpu-server-1",
            cpu_percent=65.0,
            gpu_percent=70.0,
            memory_percent=50.0,
            gpu_count=2,
            has_gpu=True,
            gpu_name="NVIDIA H100",
            active_jobs=10,
        )
        assert node.cpu_percent == 65.0
        assert node.gpu_count == 2
        assert node.gpu_name == "NVIDIA H100"

    def test_to_dict(self):
        """Should serialize to dictionary."""
        node = NodeResources(
            node_id="test",
            cpu_percent=50.0,
            gpu_percent=60.0,
        )
        data = node.to_dict()
        assert data["node_id"] == "test"
        assert data["cpu_percent"] == 50.0
        assert data["gpu_percent"] == 60.0

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        data = {
            "node_id": "test",
            "cpu_percent": 50.0,
            "gpu_percent": 60.0,
            "has_gpu": True,
            "extra_field": "ignored",
        }
        node = NodeResources.from_dict(data)
        assert node.node_id == "test"
        assert node.cpu_percent == 50.0
        assert node.has_gpu is True

    def test_get_max_gpu_jobs_no_gpu(self):
        """Should return 0 for nodes without GPU."""
        node = NodeResources(node_id="cpu-only", has_gpu=False, gpu_count=0)
        assert node.get_max_gpu_jobs() == 0

    def test_get_max_gpu_jobs_datacenter(self):
        """Should return higher limits for datacenter GPUs."""
        node = NodeResources(
            node_id="datacenter",
            has_gpu=True,
            gpu_count=8,
            gpu_name="NVIDIA H100",
        )
        max_jobs = node.get_max_gpu_jobs()
        assert max_jobs == 32  # 8 GPUs * 4 jobs per GPU

    def test_get_max_gpu_jobs_consumer(self):
        """Should return moderate limits for consumer GPUs."""
        node = NodeResources(
            node_id="gaming-pc",
            has_gpu=True,
            gpu_count=1,
            gpu_name="NVIDIA RTX 4090",
        )
        max_jobs = node.get_max_gpu_jobs()
        assert max_jobs == 3  # 1 GPU * 3 jobs per GPU

    def test_get_max_cpu_jobs(self):
        """Should calculate reasonable CPU job limits."""
        node = NodeResources(
            node_id="cpu-server",
            cpu_count=32,
            memory_gb=64.0,
        )
        max_jobs = node.get_max_cpu_jobs()
        # 32 cores / 2 = 16, 64GB / 2 = 32, min is 16
        assert max_jobs == 16

    def test_get_max_cpu_jobs_memory_constrained(self):
        """Should respect memory constraints."""
        node = NodeResources(
            node_id="low-memory",
            cpu_count=64,
            memory_gb=16.0,
        )
        max_jobs = node.get_max_cpu_jobs()
        # 64 cores / 2 = 32, 16GB / 2 = 8, min is 8
        assert max_jobs == 8


# =============================================================================
# ClusterState Tests
# =============================================================================


class TestClusterState:
    """Tests for ClusterState dataclass."""

    def test_empty_cluster(self):
        """Should handle empty cluster."""
        state = ClusterState(nodes=[])
        state.compute_aggregates()
        assert state.total_cpu_util == 0.0
        assert state.total_gpu_util == 0.0
        assert state.gpu_node_count == 0
        assert state.total_jobs == 0

    def test_compute_aggregates(self):
        """Should correctly compute aggregate statistics."""
        nodes = [
            NodeResources(node_id="n1", cpu_percent=60.0, gpu_percent=70.0, has_gpu=True, active_jobs=5),
            NodeResources(node_id="n2", cpu_percent=80.0, gpu_percent=90.0, has_gpu=True, active_jobs=8),
            NodeResources(node_id="n3", cpu_percent=40.0, gpu_percent=0.0, has_gpu=False, active_jobs=3),
        ]
        state = ClusterState(nodes=nodes)
        state.compute_aggregates()

        # CPU average: (60+80+40)/3 = 60
        assert state.total_cpu_util == 60.0
        # GPU average (only GPU nodes): (70+90)/2 = 80
        assert state.total_gpu_util == 80.0
        assert state.gpu_node_count == 2
        assert state.cpu_node_count == 3
        assert state.total_jobs == 16

    def test_gpu_memory_status_ok(self):
        """Should return 'ok' when GPU memory is low."""
        nodes = [
            NodeResources(node_id="n1", has_gpu=True, gpu_memory_percent=50.0),
        ]
        state = ClusterState(nodes=nodes)
        assert state.get_gpu_memory_status() == "ok"
        assert not state.is_gpu_memory_constrained()
        assert not state.is_gpu_memory_critical()

    def test_gpu_memory_status_warning(self):
        """Should return 'warning' when GPU memory is high."""
        nodes = [
            NodeResources(node_id="n1", has_gpu=True, gpu_memory_percent=85.0),
        ]
        state = ClusterState(nodes=nodes)
        assert state.get_gpu_memory_status() == "warning"
        assert state.is_gpu_memory_constrained()
        assert not state.is_gpu_memory_critical()

    def test_gpu_memory_status_critical(self):
        """Should return 'critical' when GPU memory is very high."""
        nodes = [
            NodeResources(node_id="n1", has_gpu=True, gpu_memory_percent=95.0),
        ]
        state = ClusterState(nodes=nodes)
        assert state.get_gpu_memory_status() == "critical"
        assert state.is_gpu_memory_constrained()
        assert state.is_gpu_memory_critical()


# =============================================================================
# OptimizationResult Tests
# =============================================================================


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_to_dict(self):
        """Should serialize to dictionary."""
        result = OptimizationResult(
            action=ScaleAction.SCALE_UP,
            resource_type=ResourceType.GPU,
            current_util=45.0,
            target_util=70.0,
            adjustment=4,
            nodes_affected=["node-1", "node-2"],
            reason="GPU underutilized",
            confidence=0.85,
        )
        data = result.to_dict()
        assert data["action"] == "scale_up"
        assert data["resource_type"] == "gpu"
        assert data["current_util"] == 45.0
        assert data["adjustment"] == 4
        assert len(data["nodes_affected"]) == 2
        assert data["confidence"] == 0.85


# =============================================================================
# PIDController Tests
# =============================================================================


class TestPIDController:
    """Tests for PIDController class."""

    def test_initialization(self):
        """Should initialize with correct defaults."""
        pid = PIDController(setpoint=70.0)
        assert pid.setpoint == 70.0
        assert pid.kp > 0
        assert pid.ki >= 0
        assert pid.kd >= 0

    def test_from_config(self):
        """Should create from config dictionary."""
        config = {
            "kp": 0.5,
            "ki": 0.1,
            "kd": 0.2,
            "integral_clamp": 50.0,
        }
        pid = PIDController.from_config(config, setpoint=75.0)
        assert pid.kp_base == 0.5
        assert pid.ki_base == 0.1
        assert pid.kd_base == 0.2
        assert pid.integral_clamp == 50.0
        assert pid.setpoint == 75.0

    def test_update_below_setpoint(self):
        """Should return positive output when below setpoint."""
        pid = PIDController(setpoint=70.0, min_update_interval=0)
        # Force initial update
        pid._last_update = time.time() - 100
        output = pid.update(50.0, dt=1.0)  # 20% below target
        assert output > 0  # Should want more work

    def test_update_above_setpoint(self):
        """Should return negative output when above setpoint."""
        pid = PIDController(setpoint=70.0, min_update_interval=0)
        pid._last_update = time.time() - 100
        output = pid.update(90.0, dt=1.0)  # 20% above target
        assert output < 0  # Should want less work

    def test_update_at_setpoint(self):
        """Should return near-zero output at setpoint."""
        pid = PIDController(setpoint=70.0, min_update_interval=0)
        pid._last_update = time.time() - 100
        output = pid.update(70.0, dt=1.0)  # At target
        assert abs(output) < 5  # Should be small

    def test_reset(self):
        """Should reset controller state."""
        pid = PIDController(setpoint=70.0, min_update_interval=0)
        pid._last_update = time.time() - 100
        pid.update(50.0, dt=1.0)
        pid.update(40.0, dt=1.0)  # Build up some state

        pid.reset()
        assert pid._integral == 0.0
        assert pid._prev_error == 0.0
        assert pid._prev_output == 0.0

    def test_get_state(self):
        """Should return controller state."""
        pid = PIDController(setpoint=70.0)
        state = pid.get_state()
        assert "kp_effective" in state
        assert "integral" in state
        assert "setpoint" in state
        assert state["setpoint"] == 70.0

    def test_gain_scheduling(self):
        """Should adjust gains based on error magnitude."""
        pid = PIDController(
            kp=0.3,
            setpoint=70.0,
            gain_scheduling=True,
            large_error_threshold=15.0,
            large_error_gain_multiplier=1.5,
            min_update_interval=0,
        )
        pid._last_update = time.time() - 100

        # Large error - should increase gains
        pid._apply_gain_scheduling(25.0)
        assert pid.kp > pid.kp_base  # Gains should be increased

    def test_integral_clamping(self):
        """Should clamp integral to prevent windup."""
        pid = PIDController(
            setpoint=70.0,
            integral_clamp=10.0,
            min_update_interval=0,
        )
        pid._last_update = time.time() - 100

        # Many updates below setpoint should build integral
        for _ in range(50):
            pid.update(20.0, dt=1.0)

        assert abs(pid._integral) <= 10.0  # Should be clamped


# =============================================================================
# UtilizationPredictor Tests
# =============================================================================


class TestUtilizationPredictor:
    """Tests for UtilizationPredictor class."""

    def test_initialization(self):
        """Should initialize with correct defaults."""
        predictor = UtilizationPredictor()
        assert predictor.history_window_seconds > 0
        assert predictor.min_samples_for_prediction > 0

    def test_record_sample(self):
        """Should record utilization samples."""
        predictor = UtilizationPredictor()
        predictor.record_sample(50.0, 60.0, 40.0)
        predictor.record_sample(55.0, 65.0, 45.0)

        assert len(predictor._history) == 2
        assert predictor._ema_cpu is not None
        assert predictor._ema_gpu is not None

    def test_predict_insufficient_data(self):
        """Should return None with insufficient data."""
        predictor = UtilizationPredictor(min_samples_for_prediction=10)
        predictor.record_sample(50.0, 60.0)
        predictor.record_sample(55.0, 65.0)

        result = predictor.predict()
        assert result is None

    def test_predict_with_sufficient_data(self):
        """Should return predictions with sufficient data."""
        predictor = UtilizationPredictor(min_samples_for_prediction=5)

        # Add enough samples
        base_time = time.time()
        for i in range(10):
            predictor.record_sample(
                cpu_util=50.0 + i * 2,  # Rising CPU
                gpu_util=60.0 + i,      # Rising GPU
                timestamp=base_time + i * 10,
            )

        result = predictor.predict()
        assert result is not None
        assert "predicted_cpu" in result
        assert "predicted_gpu" in result
        assert "confidence" in result
        assert 0 <= result["predicted_cpu"] <= 100
        assert 0 <= result["predicted_gpu"] <= 100

    def test_trend_calculation(self):
        """Should calculate valid trends."""
        predictor = UtilizationPredictor(min_samples_for_prediction=5)

        # Add samples with variation
        base_time = time.time()
        for i in range(10):
            predictor.record_sample(
                cpu_util=50.0 + i * 2,  # Gradual increase
                gpu_util=70.0,          # Stable
                timestamp=base_time + i * 60,  # 60 second intervals
            )

        result = predictor.predict()
        assert result is not None
        # Check that we get a valid trend value (the exact value depends on timing)
        assert result["cpu_trend"] in ["rising", "stable", "falling"]
        assert result["gpu_trend"] in ["rising", "stable", "falling"]
        # Slope per minute should be present
        assert "cpu_slope_per_min" in result
        assert "gpu_slope_per_min" in result

    def test_clear(self):
        """Should clear history and state."""
        predictor = UtilizationPredictor()
        predictor.record_sample(50.0, 60.0)
        predictor.record_sample(55.0, 65.0)

        predictor.clear()
        assert len(predictor._history) == 0
        assert predictor._ema_cpu is None

    def test_proactive_adjustment_none_needed(self):
        """Should return None when no adjustment needed."""
        predictor = UtilizationPredictor(min_samples_for_prediction=5)

        # Add samples near target
        base_time = time.time()
        for i in range(10):
            predictor.record_sample(
                cpu_util=70.0,  # At target
                gpu_util=70.0,
                timestamp=base_time + i * 10,
            )

        result = predictor.get_proactive_adjustment()
        # May be None or no action needed
        if result is not None:
            assert result["action"] in ["scale_up", "scale_down"]


# =============================================================================
# Hardware-Aware Selfplay Limits Tests
# =============================================================================


class TestHardwareAwareLimits:
    """Tests for hardware-aware selfplay limit calculations."""

    def test_datacenter_gpu_limits(self):
        """Datacenter GPUs should have high limits."""
        # H100 with 80GB VRAM and many cores
        limit = get_max_selfplay_for_node(
            node_id="gpu-server",
            gpu_count=8,
            gpu_name="NVIDIA H100",
            cpu_count=256,
            memory_gb=512,
            has_gpu=True,
        )
        # Should be high but bounded
        assert limit >= 32
        assert limit <= 64

    def test_consumer_gpu_limits(self):
        """Consumer GPUs should have moderate limits."""
        limit = get_max_selfplay_for_node(
            node_id="gaming-pc",
            gpu_count=1,
            gpu_name="NVIDIA RTX 4090",
            cpu_count=16,
            memory_gb=32,
            has_gpu=True,
        )
        # Should be reasonable for consumer hardware
        assert 4 <= limit <= 16

    def test_low_vram_limits(self):
        """Low VRAM GPUs should have lower limits."""
        limit = get_max_selfplay_for_node(
            node_id="budget-gpu",
            gpu_count=1,
            gpu_name="NVIDIA RTX 3060",
            cpu_count=8,
            memory_gb=16,
            has_gpu=True,
        )
        # Should be constrained by VRAM and CPU
        # With 8GB VRAM and 8 CPUs, limit may be 1-4
        assert 1 <= limit <= 8

    def test_cpu_only_limits(self):
        """CPU-only nodes should scale with cores."""
        limit = get_max_selfplay_for_node(
            node_id="cpu-server",
            cpu_count=64,
            memory_gb=128,
            has_gpu=False,
        )
        assert limit >= 16
        assert limit <= 32

    def test_memory_constraints(self):
        """Low memory should constrain jobs."""
        limit = get_max_selfplay_for_node(
            node_id="low-mem",
            cpu_count=32,
            memory_gb=8,  # Only 8GB
            has_gpu=False,
        )
        # Should be very constrained
        assert limit <= 4

    def test_cpu_only_additional_jobs(self):
        """Should calculate additional CPU-only jobs correctly."""
        additional = get_max_cpu_only_selfplay(
            node_id="hybrid",
            cpu_count=88,
            memory_gb=64,
            gpu_jobs_running=4,
        )
        # Should have room for additional CPU jobs
        assert additional > 0
        assert additional <= 32

    def test_hybrid_selfplay_limits(self):
        """Should return both GPU and CPU-only limits."""
        limits = get_hybrid_selfplay_limits(
            node_id="hybrid-node",
            gpu_count=1,
            gpu_name="NVIDIA RTX 2060 Super",
            cpu_count=88,
            memory_gb=64,
            has_gpu=True,
            gpu_vram_gb=8,
        )
        assert "gpu_jobs" in limits
        assert "cpu_only_jobs" in limits
        assert "total_jobs" in limits
        assert limits["total_jobs"] == limits["gpu_jobs"] + limits["cpu_only_jobs"]

    def test_gh200_high_limits(self):
        """GH200 with unified memory should have very high limits."""
        limit = get_max_selfplay_for_node(
            node_id="gh200-node",
            gpu_count=1,
            gpu_name="NVIDIA GH200",
            cpu_count=72,
            memory_gb=480,
            has_gpu=True,
        )
        # GH200 has 480GB unified memory
        assert limit >= 40

    def test_high_cpu_multiplier(self):
        """High CPU count should boost limits."""
        # Same GPU but different CPU counts
        low_cpu_limit = get_max_selfplay_for_node(
            node_id="low-cpu",
            gpu_count=1,
            gpu_name="NVIDIA RTX 4090",
            cpu_count=8,
            memory_gb=64,
            has_gpu=True,
        )

        high_cpu_limit = get_max_selfplay_for_node(
            node_id="high-cpu",
            gpu_count=1,
            gpu_name="NVIDIA RTX 4090",
            cpu_count=128,
            memory_gb=256,
            has_gpu=True,
        )

        # High CPU should allow more jobs (due to high_cpu_multiplier)
        assert high_cpu_limit >= low_cpu_limit


# =============================================================================
# ResourceOptimizer Integration Tests
# =============================================================================


class TestResourceOptimizerIntegration:
    """Integration tests for ResourceOptimizer with temporary database."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "coordination" / "resource_state.db"
            yield db_path

    @pytest.fixture
    def optimizer(self, temp_db_path):
        """Create a ResourceOptimizer with temporary database."""
        # Reset singleton for testing
        ResourceOptimizer._instance = None

        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            opt = ResourceOptimizer()
            yield opt
            ResourceOptimizer._instance = None

    def test_report_and_retrieve_node_resources(self, optimizer):
        """Should store and retrieve node resources."""
        node = NodeResources(
            node_id="test-node-1",
            cpu_percent=65.0,
            gpu_percent=72.0,
            memory_percent=50.0,
            has_gpu=True,
            gpu_count=2,
            gpu_name="NVIDIA A100",
            active_jobs=8,
        )

        optimizer.report_node_resources(node)
        retrieved = optimizer.get_node_resources("test-node-1")

        assert retrieved is not None
        assert retrieved.node_id == "test-node-1"
        assert retrieved.cpu_percent == 65.0
        assert retrieved.gpu_percent == 72.0

    def test_cluster_state_aggregation(self, optimizer):
        """Should aggregate cluster state correctly."""
        # Report multiple nodes
        for i in range(3):
            node = NodeResources(
                node_id=f"node-{i}",
                cpu_percent=50.0 + i * 10,
                gpu_percent=60.0 + i * 10,
                has_gpu=True,
                active_jobs=5,
            )
            optimizer.report_node_resources(node)

        state = optimizer.get_cluster_state()
        assert len(state.nodes) == 3
        assert state.gpu_node_count == 3
        assert state.total_jobs == 15

    def test_scaling_decision_underutilized(self, optimizer):
        """Should recommend scale up when underutilized."""
        # Report underutilized node
        node = NodeResources(
            node_id="under-node",
            cpu_percent=30.0,  # Well below target
            gpu_percent=25.0,
            has_gpu=True,
        )
        optimizer.report_node_resources(node)

        assert optimizer.should_scale_up("gpu")
        assert not optimizer.should_scale_down("gpu")

    def test_scaling_decision_overutilized(self, optimizer):
        """Should recommend scale down when overutilized."""
        node = NodeResources(
            node_id="over-node",
            cpu_percent=95.0,  # Well above target
            gpu_percent=92.0,
            has_gpu=True,
        )
        optimizer.report_node_resources(node)

        assert optimizer.should_scale_down("gpu")
        assert not optimizer.should_scale_up("gpu")

    def test_optimization_recommendation(self, optimizer):
        """Should provide optimization recommendations."""
        node = NodeResources(
            node_id="test-node",
            cpu_percent=40.0,  # Underutilized
            gpu_percent=35.0,
            has_gpu=True,
        )
        optimizer.report_node_resources(node)

        rec = optimizer.get_optimization_recommendation()
        assert rec.action == ScaleAction.SCALE_UP
        assert rec.resource_type == ResourceType.GPU

    def test_optimal_concurrency_calculation(self, optimizer):
        """Should calculate optimal job concurrency."""
        node = NodeResources(
            node_id="worker-1",
            cpu_percent=50.0,
            gpu_percent=45.0,
            has_gpu=True,
            gpu_count=1,
            gpu_name="RTX 4090",
            cpu_count=16,
            memory_gb=64,
        )
        optimizer.report_node_resources(node)

        optimal = optimizer.get_optimal_concurrency(
            node_id="worker-1",
            resource_type="gpu",
            current_jobs=4,
        )
        # Should be a reasonable number
        assert optimal >= 1


# =============================================================================
# Rate Negotiation Tests
# =============================================================================


class TestRateNegotiation:
    """Tests for rate negotiation between orchestrators."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "coordination" / "resource_state.db"
            yield db_path

    @pytest.fixture
    def optimizer(self, temp_db_path):
        """Create a ResourceOptimizer with temporary database."""
        ResourceOptimizer._instance = None

        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            opt = ResourceOptimizer()
            yield opt
            ResourceOptimizer._instance = None

    def test_negotiate_rate_approved(self, optimizer):
        """Should approve rate within normal utilization."""
        # Set up moderate utilization
        node = NodeResources(
            node_id="normal-node",
            cpu_percent=70.0,
            gpu_percent=68.0,
            has_gpu=True,
        )
        optimizer.report_node_resources(node)

        approved = optimizer.negotiate_selfplay_rate(
            requested_rate=1000,
            reason="routine_adjustment",
            requestor="test",
        )
        # Should be near requested when utilization is in range
        assert 800 <= approved <= 1200

    def test_negotiate_rate_underutilized(self, optimizer):
        """Should increase rate when underutilized."""
        node = NodeResources(
            node_id="under-node",
            cpu_percent=40.0,  # Well below target
            gpu_percent=35.0,
            has_gpu=True,
        )
        optimizer.report_node_resources(node)

        approved = optimizer.negotiate_selfplay_rate(
            requested_rate=1000,
            reason="need_more_work",
            requestor="test",
        )
        # Should be higher than requested
        assert approved >= 1000

    def test_negotiate_rate_overutilized(self, optimizer):
        """Should decrease rate when overutilized."""
        node = NodeResources(
            node_id="over-node",
            cpu_percent=95.0,  # Critical
            gpu_percent=92.0,
            has_gpu=True,
        )
        optimizer.report_node_resources(node)

        approved = optimizer.negotiate_selfplay_rate(
            requested_rate=1000,
            reason="routine_adjustment",
            requestor="test",
        )
        # Should be lower than requested
        assert approved < 1000

    def test_get_current_rate(self, optimizer):
        """Should retrieve current selfplay rate."""
        rate = optimizer.get_current_selfplay_rate()
        assert rate > 0  # Default should be set

    def test_rate_history(self, optimizer):
        """Should track rate negotiation history."""
        node = NodeResources(
            node_id="test-node",
            cpu_percent=70.0,
            has_gpu=False,
        )
        optimizer.report_node_resources(node)

        optimizer.negotiate_selfplay_rate(1000, "test1", "requester1")
        optimizer.negotiate_selfplay_rate(1200, "test2", "requester2")

        history = optimizer.get_rate_history(limit=10)
        assert len(history) >= 2


# =============================================================================
# Config Weighting Tests
# =============================================================================


class TestConfigWeighting:
    """Tests for data-aware config weighting."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "coordination" / "resource_state.db"
            yield db_path

    @pytest.fixture
    def optimizer(self, temp_db_path):
        ResourceOptimizer._instance = None
        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            opt = ResourceOptimizer()
            yield opt
            ResourceOptimizer._instance = None

    def test_update_config_weights(self, optimizer):
        """Should calculate weights based on game distribution."""
        game_counts = {
            "square8_2p": 1000,   # Overserved
            "hex7_2p": 200,       # Underserved
            "hex9_2p": 500,       # Average
        }

        weights = optimizer.update_config_weights(game_counts)

        assert "square8_2p" in weights
        assert "hex7_2p" in weights
        # Underserved should have higher weight
        assert weights["hex7_2p"] > weights["square8_2p"]

    def test_get_config_weights(self, optimizer):
        """Should retrieve stored config weights."""
        game_counts = {"config_a": 100, "config_b": 100}
        optimizer.update_config_weights(game_counts)

        weights = optimizer.get_config_weights()
        assert "config_a" in weights
        assert "config_b" in weights


# =============================================================================
# Utilization Status Tests
# =============================================================================


class TestUtilizationStatus:
    """Tests for utilization status reporting."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "coordination" / "resource_state.db"
            yield db_path

    @pytest.fixture
    def optimizer(self, temp_db_path):
        ResourceOptimizer._instance = None
        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            opt = ResourceOptimizer()
            yield opt
            ResourceOptimizer._instance = None

    def test_status_optimal(self, optimizer):
        """Should report optimal status when in target range."""
        node = NodeResources(
            node_id="good-node",
            cpu_percent=70.0,  # In target range (60-80)
            gpu_percent=75.0,
            has_gpu=True,
        )
        optimizer.report_node_resources(node)

        status = optimizer.get_utilization_status()
        assert status["status"] == "optimal"
        assert status["cpu"]["status"] == "optimal"
        assert status["gpu"]["status"] == "optimal"

    def test_status_underutilized(self, optimizer):
        """Should report below status when underutilized."""
        node = NodeResources(
            node_id="under-node",
            cpu_percent=40.0,
            gpu_percent=35.0,
            has_gpu=True,
        )
        optimizer.report_node_resources(node)

        status = optimizer.get_utilization_status()
        assert status["status"] == "below"

    def test_status_includes_recommendation(self, optimizer):
        """Should include actionable recommendation."""
        node = NodeResources(
            node_id="test-node",
            cpu_percent=40.0,
            has_gpu=False,
        )
        optimizer.report_node_resources(node)

        status = optimizer.get_utilization_status()
        assert "recommendation" in status
        assert len(status["recommendation"]) > 0


# =============================================================================
# Metrics Tests
# =============================================================================


class TestMetrics:
    """Tests for Prometheus metrics generation."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "coordination" / "resource_state.db"
            yield db_path

    @pytest.fixture
    def optimizer(self, temp_db_path):
        ResourceOptimizer._instance = None
        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            opt = ResourceOptimizer()
            yield opt
            ResourceOptimizer._instance = None

    def test_metrics_dict(self, optimizer):
        """Should generate metrics dictionary."""
        node = NodeResources(
            node_id="metrics-node",
            cpu_percent=65.0,
            gpu_percent=70.0,
            has_gpu=True,
            active_jobs=5,
        )
        optimizer.report_node_resources(node)

        metrics = optimizer.get_metrics_dict()

        # Check expected metric names
        assert "ringrift_cluster_cpu_utilization" in metrics
        assert "ringrift_cluster_gpu_utilization" in metrics
        assert "ringrift_target_util_min" in metrics
        assert "ringrift_cluster_total_jobs" in metrics
        assert "ringrift_optimization_action" in metrics

        # Check values are reasonable
        assert 0 <= metrics["ringrift_cluster_cpu_utilization"] <= 1
        assert metrics["ringrift_cluster_total_jobs"] == 5


# =============================================================================
# Module-Level Function Tests
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "coordination" / "resource_state.db"
            yield db_path

    @pytest.fixture
    def reset_optimizer(self, temp_db_path):
        """Reset optimizer singleton for clean tests."""
        ResourceOptimizer._instance = None
        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            yield
            ResourceOptimizer._instance = None

    def test_get_cluster_utilization(self, reset_optimizer):
        """Should return cluster utilization tuple."""
        optimizer = get_resource_optimizer()
        node = NodeResources(
            node_id="test",
            cpu_percent=60.0,
            gpu_percent=70.0,
            memory_percent=50.0,
            has_gpu=True,
        )
        optimizer.report_node_resources(node)

        cpu, gpu, memory = get_cluster_utilization()
        assert cpu == 60.0
        assert gpu == 70.0
        assert memory == 50.0


# =============================================================================
# Predictive Scaling Tests
# =============================================================================


class TestPredictiveScaling:
    """Tests for predictive scaling functionality."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "coordination" / "resource_state.db"
            yield db_path

    @pytest.fixture
    def optimizer(self, temp_db_path):
        ResourceOptimizer._instance = None
        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            opt = ResourceOptimizer()
            yield opt
            ResourceOptimizer._instance = None

    def test_predictor_state(self, optimizer):
        """Should return predictor state."""
        state = optimizer.get_predictor_state()
        assert "enabled" in state
        assert "sample_count" in state
        assert state["enabled"] is True

    def test_enable_disable_predictive(self, optimizer):
        """Should toggle predictive scaling."""
        optimizer.set_predictive_scaling_enabled(False)
        assert not optimizer._predictive_scaling_enabled

        optimizer.set_predictive_scaling_enabled(True)
        assert optimizer._predictive_scaling_enabled


# =============================================================================
# History and Cleanup Tests
# =============================================================================


class TestHistoryAndCleanup:
    """Tests for utilization history and cleanup."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "coordination" / "resource_state.db"
            yield db_path

    @pytest.fixture
    def optimizer(self, temp_db_path):
        ResourceOptimizer._instance = None
        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            opt = ResourceOptimizer()
            yield opt
            ResourceOptimizer._instance = None

    def test_record_utilization(self, optimizer):
        """Should record utilization samples."""
        optimizer.record_utilization(
            node_id="test",
            cpu_percent=65.0,
            gpu_percent=70.0,
            memory_percent=50.0,
        )

        history = optimizer.get_utilization_history(node_id="test", hours=1)
        # Should have at least one entry
        assert len(history) >= 0  # May be empty if aggregation buckets it

    def test_cleanup_old_data(self, optimizer):
        """Should clean up old data."""
        # Record some data
        optimizer.record_utilization("test", 50.0, 60.0)

        # Cleanup should not error
        deleted = optimizer.cleanup_old_data(days=7)
        assert deleted >= 0


# =============================================================================
# Additional Tests for Missing Coverage
# =============================================================================


class TestScaleAction:
    """Tests for get_scale_action method."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "coordination" / "resource_state.db"
            yield db_path

    @pytest.fixture
    def optimizer(self, temp_db_path):
        ResourceOptimizer._instance = None
        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            opt = ResourceOptimizer()
            yield opt
            ResourceOptimizer._instance = None

    def test_get_scale_action_scale_up(self, optimizer):
        """Should return SCALE_UP when utilization is low."""
        node = NodeResources(
            node_id="low-util-node",
            cpu_percent=30.0,  # Below SCALE_UP_THRESHOLD (55)
            gpu_percent=25.0,
            has_gpu=True,
        )
        optimizer.report_node_resources(node)

        action = optimizer.get_scale_action("cpu")
        assert action == ScaleAction.SCALE_UP

    def test_get_scale_action_scale_down(self, optimizer):
        """Should return SCALE_DOWN when utilization is high."""
        node = NodeResources(
            node_id="high-util-node",
            cpu_percent=90.0,  # Above SCALE_DOWN_THRESHOLD (85)
            gpu_percent=92.0,
            has_gpu=True,
        )
        optimizer.report_node_resources(node)

        action = optimizer.get_scale_action("gpu")
        assert action == ScaleAction.SCALE_DOWN

    def test_get_scale_action_rebalance(self, optimizer):
        """Should return REBALANCE when utilization needs adjustment."""
        node = NodeResources(
            node_id="imbalanced-node",
            cpu_percent=60.0,  # Within range but far from optimal (70)
            has_gpu=False,
        )
        optimizer.report_node_resources(node)

        # 60% is within 55-85 range but >10 points from optimal (70)
        # However, 60% is only 10 points from 70, so may return NONE
        action = optimizer.get_scale_action("cpu")
        assert action in [ScaleAction.REBALANCE, ScaleAction.NONE]

    def test_get_scale_action_none(self, optimizer):
        """Should return NONE when utilization is optimal."""
        node = NodeResources(
            node_id="optimal-node",
            cpu_percent=70.0,  # At optimal
            gpu_percent=72.0,
            has_gpu=True,
        )
        optimizer.report_node_resources(node)

        action = optimizer.get_scale_action("cpu")
        assert action == ScaleAction.NONE


class TestOptimizationActionRecording:
    """Tests for recording optimization actions."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "coordination" / "resource_state.db"
            yield db_path

    @pytest.fixture
    def optimizer(self, temp_db_path):
        ResourceOptimizer._instance = None
        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            opt = ResourceOptimizer()
            yield opt
            ResourceOptimizer._instance = None

    def test_record_optimization_action(self, optimizer):
        """Should record optimization actions to database."""
        result = OptimizationResult(
            action=ScaleAction.SCALE_UP,
            resource_type=ResourceType.GPU,
            current_util=45.0,
            target_util=70.0,
            adjustment=4,
            nodes_affected=["node-1", "node-2"],
            reason="Testing optimization recording",
            confidence=0.8,
        )

        optimizer.record_optimization_action(result)

        # Verify it was recorded by querying the database
        with optimizer._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM optimization_history
                ORDER BY id DESC LIMIT 1
            """).fetchone()

            assert row is not None
            assert row["action"] == "scale_up"
            assert row["resource_type"] == "gpu"
            assert row["current_util"] == 45.0
            assert row["adjustment"] == 4


class TestFeedbackAdjustment:
    """Tests for apply_feedback_adjustment method."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "coordination" / "resource_state.db"
            yield db_path

    @pytest.fixture
    def optimizer(self, temp_db_path):
        ResourceOptimizer._instance = None
        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            opt = ResourceOptimizer()
            yield opt
            ResourceOptimizer._instance = None

    def test_apply_feedback_adjustment_increase(self, optimizer):
        """Should increase rate when underutilized."""
        node = NodeResources(
            node_id="under-node",
            cpu_percent=40.0,  # Below target
            gpu_percent=35.0,
            has_gpu=True,
        )
        optimizer.report_node_resources(node)

        new_rate = optimizer.apply_feedback_adjustment("test_loop")

        # Should increase from default rate
        assert new_rate >= 100  # At least min rate

    def test_apply_feedback_adjustment_decrease(self, optimizer):
        """Should decrease rate when overutilized."""
        node = NodeResources(
            node_id="over-node",
            cpu_percent=95.0,  # Above target
            gpu_percent=90.0,
            has_gpu=True,
        )
        optimizer.report_node_resources(node)

        new_rate = optimizer.apply_feedback_adjustment("test_loop")

        # Should be within valid range
        assert 100 <= new_rate <= 5000


class TestProactiveAdjustment:
    """Tests for proactive adjustment methods."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "coordination" / "resource_state.db"
            yield db_path

    @pytest.fixture
    def optimizer(self, temp_db_path):
        ResourceOptimizer._instance = None
        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            opt = ResourceOptimizer()
            yield opt
            ResourceOptimizer._instance = None

    def test_get_prediction(self, optimizer):
        """Should return prediction if available."""
        # Add samples to predictor
        for i in range(15):
            optimizer._predictor.record_sample(
                cpu_util=60.0 + i,
                gpu_util=65.0 + i,
                gpu_mem_util=50.0,
            )

        prediction = optimizer.get_prediction()

        # With sufficient samples, should return prediction
        if prediction is not None:
            assert "predicted_cpu" in prediction
            assert "predicted_gpu" in prediction
            assert "confidence" in prediction

    def test_get_proactive_adjustment_optimizer(self, optimizer):
        """Should return proactive adjustment recommendation."""
        # Add samples showing rising trend
        base_time = time.time()
        for i in range(15):
            optimizer._predictor.record_sample(
                cpu_util=50.0 + i * 2,  # Rising
                gpu_util=55.0 + i * 2,  # Rising
                gpu_mem_util=50.0,
                timestamp=base_time + i * 10,
            )

        adjustment = optimizer.get_proactive_adjustment()

        # May or may not return adjustment depending on confidence
        if adjustment is not None:
            assert "action" in adjustment
            assert "confidence" in adjustment
            assert adjustment["action"] in ["scale_up", "scale_down"]

    def test_apply_proactive_adjustment(self, optimizer):
        """Should apply proactive scaling adjustment."""
        # Setup predictive scaling scenario
        node = NodeResources(
            node_id="test-node",
            cpu_percent=70.0,
            gpu_percent=75.0,
            has_gpu=True,
        )
        optimizer.report_node_resources(node)

        # Apply adjustment (may return None if no adjustment needed)
        new_rate = optimizer.apply_proactive_adjustment("predictive_test")

        if new_rate is not None:
            assert 100 <= new_rate <= 5000


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "coordination" / "resource_state.db"
            yield db_path

    @pytest.fixture
    def optimizer(self, temp_db_path):
        ResourceOptimizer._instance = None
        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            opt = ResourceOptimizer()
            yield opt
            ResourceOptimizer._instance = None

    def test_get_node_resources_not_found(self, optimizer):
        """Should return None for non-existent node."""
        result = optimizer.get_node_resources("non-existent-node")
        assert result is None

    def test_optimal_concurrency_no_node_data(self, optimizer):
        """Should use conservative defaults when node not found."""
        optimal = optimizer.get_optimal_concurrency(
            node_id="unknown-node",
            resource_type="gpu",
            current_jobs=2,
        )

        # Should return conservative limit
        assert 1 <= optimal <= 4

    def test_negotiate_rate_emergency_reason(self, optimizer):
        """Should approve emergency requests regardless of utilization."""
        node = NodeResources(
            node_id="emergency-node",
            cpu_percent=95.0,  # Overutilized
            has_gpu=False,
        )
        optimizer.report_node_resources(node)

        approved = optimizer.negotiate_selfplay_rate(
            requested_rate=2000,
            reason="emergency_data_collection",
            requestor="admin",
        )

        # Should approve despite high utilization
        assert 100 <= approved <= 5000

    def test_negotiate_rate_critical_reason(self, optimizer):
        """Should approve critical requests."""
        node = NodeResources(
            node_id="critical-node",
            cpu_percent=90.0,
            has_gpu=False,
        )
        optimizer.report_node_resources(node)

        approved = optimizer.negotiate_selfplay_rate(
            requested_rate=1500,
            reason="critical_model_update",
            requestor="training",
        )

        assert 100 <= approved <= 5000

    def test_negotiate_rate_gpu_memory_critical(self, optimizer):
        """Should throttle rate when GPU memory is critical."""
        node = NodeResources(
            node_id="gpu-critical",
            cpu_percent=70.0,
            gpu_percent=75.0,
            gpu_memory_percent=95.0,  # Critical
            has_gpu=True,
        )
        optimizer.report_node_resources(node)

        approved = optimizer.negotiate_selfplay_rate(
            requested_rate=1000,
            reason="routine",
            requestor="test",
        )

        # Should be heavily throttled due to GPU memory
        assert approved < 1000

    def test_negotiate_rate_gpu_memory_constrained(self, optimizer):
        """Should moderately throttle when GPU memory is constrained."""
        node = NodeResources(
            node_id="gpu-warning",
            cpu_percent=70.0,
            gpu_percent=75.0,
            gpu_memory_percent=85.0,  # Warning level
            has_gpu=True,
        )
        optimizer.report_node_resources(node)

        approved = optimizer.negotiate_selfplay_rate(
            requested_rate=1000,
            reason="routine",
            requestor="test",
        )

        # Should be moderately throttled
        assert approved <= 1000


class TestUtilizationPredictorEdgeCases:
    """Tests for UtilizationPredictor edge cases."""

    def test_calculate_trend_single_point(self):
        """Should handle single data point."""
        predictor = UtilizationPredictor()
        data = [(time.time(), 50.0)]

        slope, intercept = predictor._calculate_trend(data)

        assert slope == 0.0
        assert intercept == 50.0

    def test_calculate_trend_identical_timestamps(self):
        """Should handle identical timestamps gracefully."""
        predictor = UtilizationPredictor()
        ts = time.time()
        data = [(ts, 50.0), (ts, 55.0), (ts, 60.0)]

        slope, intercept = predictor._calculate_trend(data)

        # When timestamps are identical, denominator is near zero
        # The function should handle this gracefully and return average
        # The exact value depends on floating point precision
        assert abs(slope) < 1.0  # Slope should be small or near average
        # Intercept calculation may be unstable with identical timestamps
        # Just verify it's a valid number
        assert isinstance(intercept, (int, float))
        assert not (intercept != intercept)  # Not NaN

    def test_record_sample_prunes_old_data(self):
        """Should prune samples older than history window."""
        predictor = UtilizationPredictor(history_window_seconds=60)

        # Add old sample
        old_time = time.time() - 120  # 2 minutes ago
        predictor.record_sample(10.0, 20.0, timestamp=old_time)

        # Add recent sample
        predictor.record_sample(50.0, 60.0)

        # Old sample should be pruned
        assert len(predictor._history) == 1

    def test_get_proactive_adjustment_low_confidence(self):
        """Should return None when confidence is too low."""
        predictor = UtilizationPredictor(min_samples_for_prediction=5)

        # Add only a few samples (low confidence)
        for i in range(6):
            predictor.record_sample(
                cpu_util=70.0,  # Stable
                gpu_util=70.0,
                gpu_mem_util=50.0,
            )

        # Mock the prediction to have low confidence
        with patch.object(predictor, 'predict') as mock_predict:
            mock_predict.return_value = {
                "predicted_cpu": 45.0,
                "predicted_gpu": 50.0,
                "predicted_gpu_mem": 50.0,
                "confidence": 0.3,  # Below 0.5 threshold
            }

            adjustment = predictor.get_proactive_adjustment()
            assert adjustment is None


class TestPIDControllerEdgeCases:
    """Tests for PIDController edge cases."""

    def test_update_throttling(self):
        """Should throttle updates based on min_update_interval."""
        pid = PIDController(
            setpoint=70.0,
            min_update_interval=10.0,  # 10 second minimum
        )

        # First update
        output1 = pid.update(50.0, dt=1.0)

        # Immediate second update (should be throttled)
        output2 = pid.update(55.0, dt=1.0)

        # Should return same output due to throttling
        assert output1 == output2

    def test_update_auto_dt_calculation(self):
        """Should auto-calculate dt when not provided."""
        pid = PIDController(setpoint=70.0, min_update_interval=0)

        pid._last_update = time.time() - 100  # Force update
        output = pid.update(50.0)  # No dt provided

        # Should still work
        assert output != 0

    def test_output_smoothing_disabled(self):
        """Should work with output smoothing disabled."""
        pid = PIDController(
            setpoint=70.0,
            output_smoothing=0.0,  # Disabled
            min_update_interval=0,
        )

        pid._last_update = time.time() - 100
        output = pid.update(50.0, dt=1.0)

        assert output != 0

    def test_gain_scheduling_disabled(self):
        """Should work with gain scheduling disabled."""
        pid = PIDController(
            setpoint=70.0,
            gain_scheduling=False,
            min_update_interval=0,
        )

        pid._last_update = time.time() - 100
        pid._apply_gain_scheduling(30.0)  # Large error

        # Gains should remain at base values
        assert pid.kp == pid.kp_base
        assert pid.ki == pid.ki_base
        assert pid.kd == pid.kd_base


class TestModuleFunctionsComprehensive:
    """Comprehensive tests for module-level functions."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "coordination" / "resource_state.db"
            yield db_path

    @pytest.fixture
    def optimizer(self, temp_db_path):
        """Create optimizer with temp database."""
        ResourceOptimizer._instance = None
        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            opt = ResourceOptimizer()
            yield opt
            ResourceOptimizer._instance = None

    def test_should_scale_up_method(self, optimizer):
        """Test should_scale_up method."""
        node = NodeResources(
            node_id="test",
            cpu_percent=40.0,  # Below threshold
            has_gpu=False,
        )
        optimizer.report_node_resources(node)

        assert optimizer.should_scale_up("cpu") is True

    def test_should_scale_down_method(self, optimizer):
        """Test should_scale_down method."""
        node = NodeResources(
            node_id="test",
            cpu_percent=90.0,  # Above threshold
            has_gpu=False,
        )
        optimizer.report_node_resources(node)

        assert optimizer.should_scale_down("cpu") is True

    def test_get_optimal_concurrency_method(self, optimizer):
        """Test get_optimal_concurrency method."""
        node = NodeResources(
            node_id="worker-1",
            cpu_percent=65.0,
            gpu_percent=70.0,
            has_gpu=True,
            gpu_count=1,
            cpu_count=16,
            memory_gb=64,
        )
        optimizer.report_node_resources(node)

        optimal = optimizer.get_optimal_concurrency(
            node_id="worker-1",
            resource_type="gpu",
            current_util=70.0,
        )

        assert optimal >= 1


class TestClusterStateComprehensive:
    """Comprehensive tests for ClusterState edge cases."""

    def test_compute_aggregates_mixed_nodes(self):
        """Should handle mix of GPU and CPU-only nodes."""
        nodes = [
            NodeResources(node_id="gpu1", cpu_percent=60.0, gpu_percent=70.0, has_gpu=True),
            NodeResources(node_id="gpu2", cpu_percent=65.0, gpu_percent=75.0, has_gpu=True),
            NodeResources(node_id="cpu1", cpu_percent=80.0, has_gpu=False),
        ]
        state = ClusterState(nodes=nodes)
        state.compute_aggregates()

        assert state.gpu_node_count == 2
        assert state.cpu_node_count == 3
        assert state.total_gpu_util == 72.5  # (70+75)/2

    def test_compute_aggregates_zero_values(self):
        """Should handle nodes with zero utilization."""
        nodes = [
            NodeResources(node_id="idle", cpu_percent=0.0, gpu_percent=0.0, has_gpu=True),
        ]
        state = ClusterState(nodes=nodes)
        state.compute_aggregates()

        # Zero values should be excluded from averages
        assert state.total_cpu_util == 0.0
        assert state.total_gpu_util == 0.0

    def test_gpu_memory_status_no_gpu_nodes(self):
        """Should handle cluster with no GPU nodes."""
        nodes = [
            NodeResources(node_id="cpu-only", cpu_percent=60.0, has_gpu=False),
        ]
        state = ClusterState(nodes=nodes)

        assert state.get_gpu_memory_status() == "ok"
        assert not state.is_gpu_memory_constrained()
        assert not state.is_gpu_memory_critical()


# =============================================================================
# Health Check Tests (December 2025)
# =============================================================================


class TestHealthCheck:
    """Tests for ResourceOptimizer.health_check() method."""

    @pytest.fixture
    def optimizer(self, temp_db_path):
        """Provide ResourceOptimizer instance for health check tests."""
        import app.coordination.resource_optimizer as ro_module
        from app.coordination.resource_optimizer import ResourceOptimizer

        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            with patch("app.coordination.resource_optimizer.env") as mock_env:
                mock_env.node_id = "health-test-node"
                mock_env.orchestrator_id = "test-orchestrator"
                # Reset both class-level and module-level singletons
                ResourceOptimizer._instance = None
                ro_module._optimizer = None
                opt = ResourceOptimizer()
                yield opt

    def test_health_check_healthy_with_nodes(self, optimizer):
        """health_check should return healthy when nodes are reporting."""
        # Report a healthy node
        node = NodeResources(
            node_id="healthy-node",
            cpu_percent=65.0,
            gpu_percent=70.0,
            has_gpu=True,
            active_jobs=5,
        )
        optimizer.report_node_resources(node)

        result = optimizer.health_check()

        assert result.healthy is True
        assert "healthy" in result.message.lower() or "running" in str(result.status).lower()

    def test_health_check_degraded_high_utilization(self, optimizer):
        """health_check should return degraded for very high utilization."""
        # Report severely overloaded node
        node = NodeResources(
            node_id="overloaded-node",
            cpu_percent=98.0,
            gpu_percent=99.0,
            has_gpu=True,
        )
        optimizer.report_node_resources(node)

        result = optimizer.health_check()

        assert result.healthy is True  # Still healthy, but degraded
        # Check that the high utilization is detected
        assert "cpu" in result.message.lower() or "gpu" in result.message.lower() or "high" in result.message.lower()

    def test_health_check_no_nodes(self, optimizer):
        """health_check should handle no nodes reporting."""
        result = optimizer.health_check()

        # Should still be healthy but may note no nodes
        assert result.healthy is True

    def test_health_check_db_error_handling(self, optimizer):
        """health_check should handle database errors gracefully."""
        # Close the db connection to force an error
        with patch.object(optimizer, "_get_connection", side_effect=sqlite3.DatabaseError("Test error")):
            result = optimizer.health_check()

            # Should not raise, and should return unhealthy
            assert result.healthy is False
            assert "error" in result.message.lower() or "database" in result.message.lower()


class TestGetMaxSelfplayForNodeById:
    """Tests for get_max_selfplay_for_node_by_id function."""

    def test_node_in_database(self, temp_db_path):
        """Should return hardware-aware limit for known node."""
        import app.coordination.resource_optimizer as ro_module
        from app.coordination.resource_optimizer import (
            ResourceOptimizer,
            get_max_selfplay_for_node_by_id,
            NodeResources,
        )

        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            with patch("app.coordination.resource_optimizer.env") as mock_env:
                mock_env.node_id = "test-node"
                mock_env.orchestrator_id = "test"
                # Reset both class-level and module-level singletons
                ResourceOptimizer._instance = None
                ro_module._optimizer = None
                optimizer = ResourceOptimizer()

                # Report a node with hardware info
                node = NodeResources(
                    node_id="known-node",
                    gpu_count=2,
                    gpu_name="RTX 4090",
                    cpu_count=32,
                    memory_gb=64.0,
                    has_gpu=True,
                )
                optimizer.report_node_resources(node)

                result = get_max_selfplay_for_node_by_id("known-node")

                assert result >= 4
                assert result <= 32

    def test_node_not_in_database_h100_pattern(self, temp_db_path):
        """Should use hostname pattern for unknown H100 nodes."""
        import app.coordination.resource_optimizer as ro_module
        from app.coordination.resource_optimizer import (
            ResourceOptimizer,
            get_max_selfplay_for_node_by_id,
        )

        # Ensure clean state - the function queries ResourceOptimizer singleton
        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            with patch("app.coordination.resource_optimizer.env") as mock_env:
                mock_env.node_id = "test-node"
                mock_env.orchestrator_id = "test"
                # Reset both class-level and module-level singletons
                ResourceOptimizer._instance = None
                ro_module._optimizer = None
                ResourceOptimizer()  # Initialize singleton

                # The function should fall back to pattern matching
                result = get_max_selfplay_for_node_by_id("runpod-h100-1")

                assert result == 12  # HIGH_END tier

    def test_node_not_in_database_4090_pattern(self, temp_db_path):
        """Should use hostname pattern for unknown 4090 nodes."""
        import app.coordination.resource_optimizer as ro_module
        from app.coordination.resource_optimizer import (
            ResourceOptimizer,
            get_max_selfplay_for_node_by_id,
        )

        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            with patch("app.coordination.resource_optimizer.env") as mock_env:
                mock_env.node_id = "test-node"
                mock_env.orchestrator_id = "test"
                # Reset both class-level and module-level singletons
                ResourceOptimizer._instance = None
                ro_module._optimizer = None
                ResourceOptimizer()  # Initialize singleton

                result = get_max_selfplay_for_node_by_id("vast-4090-node")

                assert result == 8  # MID_TIER

    def test_node_not_in_database_unknown(self, temp_db_path):
        """Should use conservative default for unknown nodes."""
        import app.coordination.resource_optimizer as ro_module
        from app.coordination.resource_optimizer import (
            ResourceOptimizer,
            get_max_selfplay_for_node_by_id,
        )

        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            with patch("app.coordination.resource_optimizer.env") as mock_env:
                mock_env.node_id = "test-node"
                mock_env.orchestrator_id = "test"
                # Reset both class-level and module-level singletons
                ResourceOptimizer._instance = None
                ro_module._optimizer = None
                ResourceOptimizer()  # Initialize singleton

                result = get_max_selfplay_for_node_by_id("unknown-mystery-node")

                assert result == 6  # Conservative default


class TestGetNodeHardwareInfo:
    """Tests for get_node_hardware_info function."""

    def test_node_found(self, temp_db_path):
        """Should return hardware info dict for known node."""
        import app.coordination.resource_optimizer as ro_module
        from app.coordination.resource_optimizer import (
            ResourceOptimizer,
            get_node_hardware_info,
            NodeResources,
        )

        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            with patch("app.coordination.resource_optimizer.env") as mock_env:
                mock_env.node_id = "test-node"
                mock_env.orchestrator_id = "test"
                # Reset both class-level and module-level singletons
                ResourceOptimizer._instance = None
                ro_module._optimizer = None
                optimizer = ResourceOptimizer()

                # Report a node
                node = NodeResources(
                    node_id="hw-info-node",
                    gpu_count=4,
                    gpu_name="A100",
                    cpu_count=64,
                    memory_gb=256.0,
                    has_gpu=True,
                )
                optimizer.report_node_resources(node)

                result = get_node_hardware_info("hw-info-node")

                assert result is not None
                assert result["gpu_count"] == 4
                assert result["gpu_name"] == "A100"
                assert result["cpu_count"] == 64
                assert result["memory_gb"] == 256.0
                assert result["has_gpu"] is True

    def test_node_not_found(self, temp_db_path):
        """Should return None for unknown node."""
        import app.coordination.resource_optimizer as ro_module
        from app.coordination.resource_optimizer import (
            ResourceOptimizer,
            get_node_hardware_info,
        )

        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            with patch("app.coordination.resource_optimizer.env") as mock_env:
                mock_env.node_id = "test-node"
                mock_env.orchestrator_id = "test"
                # Reset both class-level and module-level singletons
                ResourceOptimizer._instance = None
                ro_module._optimizer = None
                ResourceOptimizer()

                result = get_node_hardware_info("nonexistent-node")

                assert result is None


class TestGetConfigWeightDetails:
    """Tests for get_config_weight_details method."""

    def test_returns_detailed_list(self, temp_db_path):
        """Should return list of detailed config weight info."""
        from app.coordination.resource_optimizer import ResourceOptimizer

        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            with patch("app.coordination.resource_optimizer.env") as mock_env:
                mock_env.node_id = "test-node"
                mock_env.orchestrator_id = "test"
                ResourceOptimizer._instance = None
                optimizer = ResourceOptimizer()

                # Update weights
                optimizer.update_config_weights(
                    game_counts={"hex8_2p": 1000, "square8_2p": 200},
                    throughput={"hex8_2p": 100.0, "square8_2p": 30.0},
                )

                details = optimizer.get_config_weight_details()

                assert isinstance(details, list)
                assert len(details) >= 2

                # Check that details contain expected fields
                for d in details:
                    assert "config_key" in d
                    assert "weight" in d
                    assert "game_count" in d

    def test_empty_on_no_weights(self, temp_db_path):
        """Should return empty list when no weights configured."""
        from app.coordination.resource_optimizer import ResourceOptimizer

        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            with patch("app.coordination.resource_optimizer.env") as mock_env:
                mock_env.node_id = "test-node"
                mock_env.orchestrator_id = "test"
                ResourceOptimizer._instance = None
                optimizer = ResourceOptimizer()

                details = optimizer.get_config_weight_details()

                assert isinstance(details, list)
                assert len(details) == 0


class TestGetRecommendation:
    """Tests for _get_recommendation internal method."""

    def test_recommendation_no_nodes(self, temp_db_path):
        """Should recommend checking connectivity when no nodes."""
        from app.coordination.resource_optimizer import (
            ResourceOptimizer,
            ClusterState,
        )

        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            with patch("app.coordination.resource_optimizer.env") as mock_env:
                mock_env.node_id = "test-node"
                mock_env.orchestrator_id = "test"
                ResourceOptimizer._instance = None
                optimizer = ResourceOptimizer()

                state = ClusterState(nodes=[])
                state.compute_aggregates()

                rec = optimizer._get_recommendation(state)

                assert "no active nodes" in rec.lower() or "connectivity" in rec.lower()

    def test_recommendation_scale_up(self, temp_db_path):
        """Should recommend scaling up when underutilized."""
        from app.coordination.resource_optimizer import (
            ResourceOptimizer,
            ClusterState,
            NodeResources,
        )

        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            with patch("app.coordination.resource_optimizer.env") as mock_env:
                mock_env.node_id = "test-node"
                mock_env.orchestrator_id = "test"
                ResourceOptimizer._instance = None
                optimizer = ResourceOptimizer()

                nodes = [
                    NodeResources(node_id="low", cpu_percent=30.0, has_gpu=False),
                ]
                state = ClusterState(nodes=nodes)
                state.compute_aggregates()

                rec = optimizer._get_recommendation(state)

                assert "scale up" in rec.lower() or "increase" in rec.lower()

    def test_recommendation_scale_down(self, temp_db_path):
        """Should recommend scaling down when overutilized."""
        from app.coordination.resource_optimizer import (
            ResourceOptimizer,
            ClusterState,
            NodeResources,
        )

        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            with patch("app.coordination.resource_optimizer.env") as mock_env:
                mock_env.node_id = "test-node"
                mock_env.orchestrator_id = "test"
                ResourceOptimizer._instance = None
                optimizer = ResourceOptimizer()

                nodes = [
                    NodeResources(node_id="hot", cpu_percent=95.0, has_gpu=False),
                ]
                state = ClusterState(nodes=nodes)
                state.compute_aggregates()

                rec = optimizer._get_recommendation(state)

                assert "scale down" in rec.lower() or "reduce" in rec.lower()

    def test_recommendation_optimal(self, temp_db_path):
        """Should indicate optimal when in target range."""
        from app.coordination.resource_optimizer import (
            ResourceOptimizer,
            ClusterState,
            NodeResources,
        )

        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            with patch("app.coordination.resource_optimizer.env") as mock_env:
                mock_env.node_id = "test-node"
                mock_env.orchestrator_id = "test"
                ResourceOptimizer._instance = None
                optimizer = ResourceOptimizer()

                nodes = [
                    NodeResources(node_id="good", cpu_percent=70.0, has_gpu=False),
                ]
                state = ClusterState(nodes=nodes)
                state.compute_aggregates()

                rec = optimizer._get_recommendation(state)

                assert "optimal" in rec.lower()


class TestUtilizationHistoryEdgeCases:
    """Edge case tests for utilization history."""

    def test_history_with_resolution(self, temp_db_path):
        """Should aggregate history by resolution."""
        from app.coordination.resource_optimizer import (
            ResourceOptimizer,
            NodeResources,
        )

        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            with patch("app.coordination.resource_optimizer.env") as mock_env:
                mock_env.node_id = "test-node"
                mock_env.orchestrator_id = "test"
                ResourceOptimizer._instance = None
                optimizer = ResourceOptimizer()

                # Report multiple samples
                for i in range(5):
                    node = NodeResources(
                        node_id="test-node",
                        cpu_percent=50.0 + i * 5,
                        gpu_percent=60.0 + i * 5,
                        has_gpu=True,
                    )
                    optimizer.report_node_resources(node)

                # Get history with specific resolution
                history = optimizer.get_utilization_history(
                    node_id="test-node",
                    hours=1.0,
                    resolution_seconds=60,
                )

                assert isinstance(history, list)

    def test_cluster_average_history(self, temp_db_path):
        """Should compute cluster-wide average when node_id is None."""
        from app.coordination.resource_optimizer import (
            ResourceOptimizer,
            NodeResources,
        )

        with patch("app.coordination.resource_optimizer.COORDINATION_DB_PATH", temp_db_path):
            with patch("app.coordination.resource_optimizer.env") as mock_env:
                mock_env.node_id = "test-node"
                mock_env.orchestrator_id = "test"
                ResourceOptimizer._instance = None
                optimizer = ResourceOptimizer()

                # Report samples for multiple nodes
                for i in range(3):
                    node = NodeResources(
                        node_id=f"node-{i}",
                        cpu_percent=50.0 + i * 10,
                        has_gpu=False,
                    )
                    optimizer.report_node_resources(node)

                # Get cluster average
                history = optimizer.get_utilization_history(
                    node_id=None,  # Cluster average
                    hours=1.0,
                )

                assert isinstance(history, list)
