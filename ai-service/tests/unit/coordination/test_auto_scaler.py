"""Unit tests for AutoScaler.

December 2025: Tests for queue-based auto-scaling of GPU instances.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.auto_scaler import (
    AutoScaler,
    MonitoringAwareAutoScaler,
    NodeMetrics,
    ScaleEvent,
    ScalingAction,
    ScalingConfig,
    ScalingDecision,
)


# =============================================================================
# Test Dataclasses
# =============================================================================


class TestScalingConfig:
    """Tests for ScalingConfig dataclass."""

    def test_default_values(self):
        """ScalingConfig has expected defaults."""
        config = ScalingConfig()
        assert config.queue_depth_scale_up == 10
        assert config.queue_depth_scale_down == 2
        assert config.gpu_idle_minutes == 15
        assert config.gpu_idle_threshold_percent == 10.0
        assert config.min_instances == 2
        assert config.max_instances == 20
        assert config.max_hourly_cost == 2.00
        assert config.scale_cooldown_seconds == 600
        assert config.max_scale_up_per_cycle == 3
        assert config.max_scale_down_per_cycle == 2
        assert config.predictive_scaling is True
        assert config.enabled is True

    def test_custom_values(self):
        """ScalingConfig accepts custom values."""
        config = ScalingConfig(
            queue_depth_scale_up=20,
            min_instances=5,
            max_instances=50,
            max_hourly_cost=10.00,
            enabled=False,
        )
        assert config.queue_depth_scale_up == 20
        assert config.min_instances == 5
        assert config.max_instances == 50
        assert config.max_hourly_cost == 10.00
        assert config.enabled is False


class TestScalingDecision:
    """Tests for ScalingDecision dataclass."""

    def test_default_values(self):
        """ScalingDecision has expected defaults."""
        decision = ScalingDecision(action=ScalingAction.NONE)
        assert decision.action == ScalingAction.NONE
        assert decision.count == 0
        assert decision.node_ids == []
        assert decision.reason == ""
        assert decision.estimated_cost_change == 0.0

    def test_scale_up_decision(self):
        """ScalingDecision for scale up."""
        decision = ScalingDecision(
            action=ScalingAction.SCALE_UP,
            count=3,
            reason="queue_depth_15_exceeds_threshold_10",
            estimated_cost_change=1.50,
        )
        assert decision.action == ScalingAction.SCALE_UP
        assert decision.count == 3
        assert decision.estimated_cost_change == 1.50


class TestScaleEvent:
    """Tests for ScaleEvent dataclass."""

    def test_scale_event_creation(self):
        """ScaleEvent is created correctly."""
        event = ScaleEvent(
            timestamp=1000.0,
            action=ScalingAction.SCALE_DOWN,
            count=2,
            node_ids=["node-1", "node-2"],
            reason="idle_nodes",
            success=True,
        )
        assert event.timestamp == 1000.0
        assert event.action == ScalingAction.SCALE_DOWN
        assert event.count == 2
        assert event.node_ids == ["node-1", "node-2"]
        assert event.success is True
        assert event.error is None


class TestNodeMetrics:
    """Tests for NodeMetrics dataclass."""

    def test_node_metrics_creation(self):
        """NodeMetrics is created correctly."""
        metrics = NodeMetrics(
            node_id="vast-12345",
            gpu_utilization=75.5,
            last_job_time=1000.0,
            hourly_cost=0.50,
            instance_type="RTX4090",
            is_idle=False,
        )
        assert metrics.node_id == "vast-12345"
        assert metrics.gpu_utilization == 75.5
        assert metrics.hourly_cost == 0.50
        assert metrics.is_idle is False


# =============================================================================
# Test AutoScaler Initialization
# =============================================================================


class TestAutoScalerInit:
    """Tests for AutoScaler initialization."""

    def test_default_initialization(self):
        """AutoScaler initializes with defaults."""
        scaler = AutoScaler()
        assert scaler.config.enabled is True
        assert scaler._last_scale_time == 0
        assert scaler._scale_history == []
        assert scaler._active_instances == {}
        assert scaler._work_queue is None

    def test_custom_config(self):
        """AutoScaler accepts custom config."""
        config = ScalingConfig(max_instances=100)
        scaler = AutoScaler(config=config)
        assert scaler.config.max_instances == 100


# =============================================================================
# Test Node Metrics Management
# =============================================================================


class TestNodeMetricsManagement:
    """Tests for node metrics tracking."""

    @pytest.fixture
    def scaler(self):
        """Create a fresh scaler for each test."""
        return AutoScaler()

    def test_update_node_metrics(self, scaler):
        """Updates node metrics correctly."""
        metrics = NodeMetrics(
            node_id="node-1",
            gpu_utilization=80.0,
            last_job_time=time.time(),
            hourly_cost=0.50,
            instance_type="RTX4090",
            is_idle=False,
        )
        scaler.update_node_metrics("node-1", metrics)
        assert "node-1" in scaler._active_instances
        assert scaler._active_instances["node-1"].gpu_utilization == 80.0

    def test_remove_node(self, scaler):
        """Removes node from tracking."""
        metrics = NodeMetrics("node-1", 80.0, time.time(), 0.50, "RTX4090")
        scaler.update_node_metrics("node-1", metrics)
        assert "node-1" in scaler._active_instances

        scaler.remove_node("node-1")
        assert "node-1" not in scaler._active_instances

    def test_idle_node_tracking(self, scaler):
        """Tracks idle nodes correctly."""
        # Non-idle node
        metrics = NodeMetrics("node-1", 80.0, time.time(), 0.50, "RTX4090", is_idle=False)
        scaler.update_node_metrics("node-1", metrics)
        assert "node-1" not in scaler._node_idle_since

        # Mark as idle
        metrics.is_idle = True
        scaler.update_node_metrics("node-1", metrics)
        assert "node-1" in scaler._node_idle_since

        # Mark as active again
        metrics.is_idle = False
        scaler.update_node_metrics("node-1", metrics)
        assert "node-1" not in scaler._node_idle_since


# =============================================================================
# Test Queue Status
# =============================================================================


class TestQueueStatus:
    """Tests for queue status methods."""

    @pytest.fixture
    def scaler(self):
        """Create scaler with mock work queue."""
        s = AutoScaler()
        s._work_queue = MagicMock()
        return s

    def test_get_pending_count(self, scaler):
        """Gets pending count from work queue."""
        scaler._work_queue.get_queue_status.return_value = {
            "by_status": {"pending": 15, "running": 5}
        }
        assert scaler.get_pending_count() == 15

    def test_get_running_count(self, scaler):
        """Gets running count from work queue."""
        scaler._work_queue.get_queue_status.return_value = {
            "by_status": {"pending": 15, "running": 5, "claimed": 2}
        }
        assert scaler.get_running_count() == 7  # running + claimed

    def test_no_work_queue(self):
        """Returns 0 when no work queue set."""
        scaler = AutoScaler()
        assert scaler.get_pending_count() == 0
        assert scaler.get_running_count() == 0


# =============================================================================
# Test Idle Node Detection
# =============================================================================


class TestIdleNodeDetection:
    """Tests for idle node detection."""

    def test_get_idle_nodes_none(self):
        """Returns empty list when no idle nodes."""
        scaler = AutoScaler()
        assert scaler._get_idle_nodes() == []

    def test_get_idle_nodes_recent(self):
        """Doesn't return recently-idle nodes."""
        scaler = AutoScaler()
        scaler._node_idle_since["node-1"] = time.time()  # Just became idle
        # Default threshold is 15 minutes
        assert scaler._get_idle_nodes() == []

    def test_get_idle_nodes_old(self):
        """Returns nodes idle beyond threshold."""
        config = ScalingConfig(gpu_idle_minutes=1)  # 1 minute threshold
        scaler = AutoScaler(config=config)
        # Node idle for 2 minutes
        scaler._node_idle_since["node-1"] = time.time() - 120
        assert scaler._get_idle_nodes() == ["node-1"]


# =============================================================================
# Test Cost Tracking
# =============================================================================


class TestCostTracking:
    """Tests for cost tracking."""

    @pytest.fixture
    def scaler(self):
        """Create scaler with some instances."""
        s = AutoScaler()
        s._active_instances = {
            "node-1": NodeMetrics("node-1", 80, time.time(), 0.50, "RTX4090"),
            "node-2": NodeMetrics("node-2", 90, time.time(), 0.75, "A100"),
        }
        return s

    def test_get_current_hourly_cost(self, scaler):
        """Calculates current hourly cost."""
        cost = scaler._get_current_hourly_cost()
        assert cost == pytest.approx(1.25)  # 0.50 + 0.75

    def test_get_cost_metrics(self, scaler):
        """Gets comprehensive cost metrics."""
        metrics = scaler.get_cost_metrics()
        assert "current_hourly_cost" in metrics
        assert "total_cost_accumulated" in metrics
        assert "projected_monthly" in metrics
        assert "budget_utilization_pct" in metrics
        assert metrics["current_hourly_cost"] == pytest.approx(1.25)


# =============================================================================
# Test Cooldown
# =============================================================================


class TestCooldown:
    """Tests for cooldown handling."""

    def test_not_in_cooldown(self):
        """Not in cooldown when no recent scaling."""
        scaler = AutoScaler()
        assert scaler._is_in_cooldown() is False

    def test_in_cooldown(self):
        """In cooldown after recent scaling."""
        scaler = AutoScaler()
        scaler._last_scale_time = time.time()
        assert scaler._is_in_cooldown() is True


# =============================================================================
# Test Queue Sampling
# =============================================================================


class TestQueueSampling:
    """Tests for queue depth sampling."""

    @pytest.fixture
    def scaler(self):
        """Create scaler with mock work queue."""
        s = AutoScaler()
        s._work_queue = MagicMock()
        s._work_queue.get_queue_status.return_value = {
            "by_status": {"pending": 10, "running": 5}
        }
        return s

    def test_record_queue_sample(self, scaler):
        """Records queue sample."""
        scaler.record_queue_sample()
        assert len(scaler._queue_depth_samples) == 1
        sample = scaler._queue_depth_samples[0]
        assert sample[1] == 10  # pending
        assert sample[2] == 5   # running

    def test_ema_updates(self, scaler):
        """Updates EMA rates with multiple samples."""
        # First sample
        scaler.record_queue_sample()

        # Second sample with changed values
        scaler._work_queue.get_queue_status.return_value = {
            "by_status": {"pending": 15, "running": 3}
        }
        scaler.record_queue_sample()

        # EMAs should have updated
        assert scaler._ema_arrival_rate >= 0
        assert scaler._ema_completion_rate >= 0


# =============================================================================
# Test Demand Prediction
# =============================================================================


class TestDemandPrediction:
    """Tests for demand prediction."""

    def test_predict_with_no_history(self):
        """Predicts with growth assumption when no history."""
        scaler = AutoScaler()
        scaler._work_queue = MagicMock()
        scaler._work_queue.get_queue_status.return_value = {
            "by_status": {"pending": 10}
        }
        # With 10 pending and 10% growth per hour, 2 hours ahead = 10 * 1.2 = 12
        predicted = scaler._predict_demand(hours_ahead=2)
        assert predicted == 12

    def test_predict_disabled(self):
        """Returns current when prediction disabled."""
        config = ScalingConfig(predictive_scaling=False)
        scaler = AutoScaler(config=config)
        scaler._work_queue = MagicMock()
        scaler._work_queue.get_queue_status.return_value = {
            "by_status": {"pending": 10}
        }
        assert scaler._predict_demand(hours_ahead=2) == 10


# =============================================================================
# Test Evaluate Method
# =============================================================================


class TestEvaluate:
    """Tests for scaling evaluation."""

    @pytest.fixture
    def scaler(self):
        """Create scaler with mock work queue."""
        config = ScalingConfig(
            queue_depth_scale_up=10,
            queue_depth_scale_down=2,
            min_instances=2,
            max_instances=20,
            scale_cooldown_seconds=0,  # No cooldown for testing
        )
        s = AutoScaler(config=config)
        s._work_queue = MagicMock()
        return s

    @pytest.mark.asyncio
    async def test_evaluate_disabled(self, scaler):
        """Returns NONE when disabled."""
        scaler.config.enabled = False
        decision = await scaler.evaluate()
        assert decision.action == ScalingAction.NONE
        assert "disabled" in decision.reason

    @pytest.mark.asyncio
    async def test_evaluate_in_cooldown(self):
        """Returns NONE when in cooldown."""
        config = ScalingConfig(scale_cooldown_seconds=600)
        scaler = AutoScaler(config=config)
        scaler._last_scale_time = time.time()
        decision = await scaler.evaluate()
        assert decision.action == ScalingAction.NONE
        assert "cooldown" in decision.reason

    @pytest.mark.asyncio
    async def test_evaluate_scale_up(self, scaler):
        """Recommends scale up when queue depth exceeds threshold."""
        # High queue depth (50 pending) to ensure scale-up is triggered
        # With 2 instances and 50 pending: work_per_instance = 25, instances_needed = (50-10)/25 = 1.6 = 1
        scaler._work_queue.get_queue_status.return_value = {
            "by_status": {"pending": 50, "running": 5}  # >> 10 threshold
        }
        # Add some instances
        for i in range(2):
            scaler._active_instances[f"node-{i}"] = NodeMetrics(
                f"node-{i}", 80, time.time(), 0.50, "RTX4090"
            )

        decision = await scaler.evaluate()
        assert decision.action == ScalingAction.SCALE_UP
        assert decision.count > 0
        assert "exceeds_threshold" in decision.reason

    @pytest.mark.asyncio
    async def test_evaluate_scale_down(self, scaler):
        """Recommends scale down when queue depth below threshold and idle nodes."""
        scaler._work_queue.get_queue_status.return_value = {
            "by_status": {"pending": 1, "running": 0}  # < 2 threshold
        }
        # Add instances including idle ones
        for i in range(5):
            scaler._active_instances[f"node-{i}"] = NodeMetrics(
                f"node-{i}", 5, time.time(), 0.50, "RTX4090", is_idle=True
            )
            scaler._node_idle_since[f"node-{i}"] = time.time() - 3600  # Idle for 1 hour

        decision = await scaler.evaluate()
        assert decision.action == ScalingAction.SCALE_DOWN
        assert decision.count > 0
        assert len(decision.node_ids) > 0

    @pytest.mark.asyncio
    async def test_evaluate_at_min_instances(self, scaler):
        """Doesn't scale down below minimum."""
        scaler._work_queue.get_queue_status.return_value = {
            "by_status": {"pending": 0, "running": 0}
        }
        # Only 2 instances (minimum)
        for i in range(2):
            scaler._active_instances[f"node-{i}"] = NodeMetrics(
                f"node-{i}", 5, time.time(), 0.50, "RTX4090", is_idle=True
            )
            scaler._node_idle_since[f"node-{i}"] = time.time() - 3600

        decision = await scaler.evaluate()
        assert decision.action == ScalingAction.NONE
        assert "min_instances" in decision.reason

    @pytest.mark.asyncio
    async def test_evaluate_cost_budget(self, scaler):
        """Respects cost budget when scaling up."""
        scaler.config.max_hourly_cost = 1.00  # Low budget
        scaler._work_queue.get_queue_status.return_value = {
            "by_status": {"pending": 100, "running": 10}  # High queue depth
        }
        # Already at budget with current instances
        for i in range(2):
            scaler._active_instances[f"node-{i}"] = NodeMetrics(
                f"node-{i}", 80, time.time(), 0.50, "RTX4090"  # 0.50 * 2 = 1.00
            )

        decision = await scaler.evaluate()
        # Should not scale up or limit scale up due to budget
        if decision.action == ScalingAction.SCALE_UP:
            assert decision.count <= 0  # Limited by budget


# =============================================================================
# Test Scale Event Recording
# =============================================================================


class TestScaleEventRecording:
    """Tests for scale event recording."""

    def test_record_success(self):
        """Records successful scale event."""
        scaler = AutoScaler()
        decision = ScalingDecision(
            action=ScalingAction.SCALE_UP,
            count=2,
            reason="test",
        )
        scaler.record_scale_event(decision, success=True)

        assert len(scaler._scale_history) == 1
        assert scaler._scale_history[0].success is True
        assert scaler._last_scale_time > 0

    def test_record_failure(self):
        """Records failed scale event."""
        scaler = AutoScaler()
        old_scale_time = scaler._last_scale_time
        decision = ScalingDecision(action=ScalingAction.SCALE_DOWN, count=1)
        scaler.record_scale_event(decision, success=False, error="API error")

        assert len(scaler._scale_history) == 1
        assert scaler._scale_history[0].success is False
        assert scaler._scale_history[0].error == "API error"
        # Failed events don't update last_scale_time
        assert scaler._last_scale_time == old_scale_time

    def test_history_limit(self):
        """Keeps only last 100 events."""
        scaler = AutoScaler()
        for i in range(110):
            decision = ScalingDecision(action=ScalingAction.NONE)
            scaler.record_scale_event(decision, success=True)

        assert len(scaler._scale_history) == 100


# =============================================================================
# Test Scaling Stats
# =============================================================================


class TestScalingStats:
    """Tests for scaling statistics."""

    def test_get_scaling_stats(self):
        """Gets comprehensive scaling stats."""
        scaler = AutoScaler()
        scaler._active_instances = {
            "node-1": NodeMetrics("node-1", 80, time.time(), 0.50, "RTX4090"),
        }
        stats = scaler.get_scaling_stats()

        assert "enabled" in stats
        assert "current_instances" in stats
        assert "pending_work" in stats
        assert "in_cooldown" in stats
        assert "current_hourly_cost" in stats
        assert stats["current_instances"] == 1


# =============================================================================
# Test MonitoringAwareAutoScaler
# =============================================================================


class TestMonitoringAwareAutoScaler:
    """Tests for MonitoringAwareAutoScaler."""

    def test_initialization(self):
        """Initializes with monitoring state."""
        scaler = MonitoringAwareAutoScaler()
        assert scaler._monitoring_alerts == {}
        assert scaler._unhealthy_nodes == {}
        assert scaler._cluster_healthy is True
        assert scaler._resource_constrained is False

    def test_inherits_from_autoscaler(self):
        """Inherits AutoScaler behavior."""
        scaler = MonitoringAwareAutoScaler()
        assert hasattr(scaler, "config")
        assert hasattr(scaler, "evaluate")
        assert hasattr(scaler, "get_cost_metrics")

    def test_custom_config(self):
        """Accepts custom config."""
        config = ScalingConfig(max_instances=100)
        scaler = MonitoringAwareAutoScaler(config=config)
        assert scaler.config.max_instances == 100


# =============================================================================
# Test ScalingAction Enum
# =============================================================================


class TestScalingAction:
    """Tests for ScalingAction enum."""

    def test_values(self):
        """ScalingAction has expected values."""
        assert ScalingAction.NONE.value == "none"
        assert ScalingAction.SCALE_UP.value == "scale_up"
        assert ScalingAction.SCALE_DOWN.value == "scale_down"

    def test_is_string_enum(self):
        """ScalingAction is a string enum."""
        assert isinstance(ScalingAction.NONE, str)
        assert ScalingAction.NONE == "none"
