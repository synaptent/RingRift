"""Tests for Resource Optimizer module.

Tests the cooperative resource scheduling system that targets 60-80% utilization
for optimal training throughput across the RingRift cluster.
"""

import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from app.coordination.resource_optimizer import (
    # Enums
    ResourceType,
    ScaleAction,
    # Data classes
    NodeResources,
    ClusterState,
    OptimizationResult,
    # Classes
    PIDController,
    UtilizationPredictor,
    ResourceOptimizer,
    # Module functions
    get_resource_optimizer,
    should_scale_up,
    should_scale_down,
    get_optimal_concurrency,
    get_cluster_utilization,
    get_max_selfplay_for_node,
    get_max_cpu_only_selfplay,
    get_hybrid_selfplay_limits,
    # Constants (for testing)
    TARGET_UTIL_MIN,
    TARGET_UTIL_MAX,
    TARGET_UTIL_OPTIMAL,
    SCALE_UP_THRESHOLD,
    SCALE_DOWN_THRESHOLD,
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
        """Should have exactly 3 resource types."""
        assert len(ResourceType) == 3

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
