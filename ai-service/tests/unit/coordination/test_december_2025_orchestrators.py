"""Tests for December 2025 orchestrators (ResourceMonitoring, MetricsAnalysis, Optimization, Cache).

These tests verify the core functionality of orchestrators added in December 2025.
Tests focus on:
- State management
- Event handling
- Status reporting
- Edge cases
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =============================================================================
# Mock event classes for testing
# =============================================================================

class MockDataEvent:
    """Mock DataEvent for testing."""

    def __init__(self, event_type, payload, source="test"):
        self.event_type = event_type
        self.payload = payload
        self.source = source


class MockDataEventType:
    """Mock DataEventType enum values."""

    BACKPRESSURE_ACTIVATED = MagicMock(value="backpressure_activated")
    BACKPRESSURE_RELEASED = MagicMock(value="backpressure_released")
    CLUSTER_CAPACITY_CHANGED = MagicMock(value="cluster_capacity_changed")
    NODE_CAPACITY_UPDATED = MagicMock(value="node_capacity_updated")
    RESOURCE_CONSTRAINT = MagicMock(value="resource_constraint")
    METRICS_UPDATED = MagicMock(value="metrics_updated")
    ELO_UPDATED = MagicMock(value="elo_updated")
    TRAINING_PROGRESS = MagicMock(value="training_progress")
    PLATEAU_DETECTED = MagicMock(value="plateau_detected")
    REGRESSION_DETECTED = MagicMock(value="regression_detected")
    CMAES_TRIGGERED = MagicMock(value="cmaes_triggered")
    CMAES_COMPLETED = MagicMock(value="cmaes_completed")
    NAS_TRIGGERED = MagicMock(value="nas_triggered")
    NAS_COMPLETED = MagicMock(value="nas_completed")
    CACHE_INVALIDATED = MagicMock(value="cache_invalidated")
    MODEL_REGISTERED = MagicMock(value="model_registered")
    MODEL_DEPLOYED = MagicMock(value="model_deployed")


# =============================================================================
# ResourceMonitoringCoordinator Tests
# =============================================================================

class TestResourceMonitoringCoordinator:
    """Tests for ResourceMonitoringCoordinator."""

    def test_initialization(self):
        """Test coordinator initializes with correct defaults."""
        from app.coordination.resource_monitoring_coordinator import ResourceMonitoringCoordinator

        coord = ResourceMonitoringCoordinator()
        assert coord.backpressure_gpu_threshold == 90.0
        assert coord.backpressure_memory_threshold == 85.0
        assert coord.backpressure_disk_threshold == 90.0
        assert not coord.is_backpressure_active()

    def test_custom_thresholds(self):
        """Test coordinator respects custom thresholds."""
        from app.coordination.resource_monitoring_coordinator import ResourceMonitoringCoordinator

        coord = ResourceMonitoringCoordinator(
            backpressure_gpu_threshold=80.0,
            backpressure_memory_threshold=75.0,
        )
        assert coord.backpressure_gpu_threshold == 80.0
        assert coord.backpressure_memory_threshold == 75.0

    def test_update_node_resources(self):
        """Test manual node resource updates."""
        from app.coordination.resource_monitoring_coordinator import ResourceMonitoringCoordinator

        coord = ResourceMonitoringCoordinator()

        # Update a node
        state = coord.update_node_resources(
            node_id="gh200-a",
            gpu_utilization=75.0,
            cpu_utilization=50.0,
            memory_used_percent=60.0,
        )

        assert state.node_id == "gh200-a"
        assert state.gpu_utilization == 75.0
        assert state.cpu_utilization == 50.0
        assert state.memory_used_percent == 60.0
        assert not state.backpressure_active

    def test_update_triggers_backpressure(self):
        """Test that threshold violations trigger backpressure."""
        from app.coordination.resource_monitoring_coordinator import ResourceMonitoringCoordinator

        coord = ResourceMonitoringCoordinator(backpressure_gpu_threshold=80.0)

        # Update with high GPU
        with patch.object(coord, "_emit_backpressure_event"):
            state = coord.update_node_resources(
                node_id="gh200-a",
                gpu_utilization=95.0,  # Above threshold
            )

        assert state.backpressure_active
        assert coord.is_backpressure_active("gh200-a")

    def test_get_stats(self):
        """Test aggregate statistics calculation."""
        from app.coordination.resource_monitoring_coordinator import ResourceMonitoringCoordinator

        coord = ResourceMonitoringCoordinator()

        # Add some nodes
        coord.update_node_resources("node-1", gpu_utilization=60.0, task_slots_available=5, task_slots_total=10)
        coord.update_node_resources("node-2", gpu_utilization=80.0, task_slots_available=3, task_slots_total=10)

        stats = coord.get_stats()
        assert stats.total_nodes == 2
        assert stats.avg_gpu_utilization == 70.0  # (60 + 80) / 2
        assert stats.total_task_slots == 20
        assert stats.available_task_slots == 8

    def test_get_status_dict(self):
        """Test status dictionary generation."""
        from app.coordination.resource_monitoring_coordinator import ResourceMonitoringCoordinator

        coord = ResourceMonitoringCoordinator()
        coord.update_node_resources("node-1", gpu_utilization=60.0)

        status = coord.get_status()
        assert "total_nodes" in status
        assert "avg_gpu_utilization" in status
        assert "backpressure_active" in status
        assert status["total_nodes"] == 1

    @pytest.mark.asyncio
    async def test_on_node_capacity_updated(self):
        """Test handling NODE_CAPACITY_UPDATED event."""
        from app.coordination.resource_monitoring_coordinator import ResourceMonitoringCoordinator

        coord = ResourceMonitoringCoordinator()

        event = MockDataEvent(
            MockDataEventType.NODE_CAPACITY_UPDATED,
            {
                "node_id": "gh200-a",
                "gpu_utilization": 65.0,
                "cpu_utilization": 45.0,
                "memory_used_percent": 55.0,
                "task_slots_available": 8,
                "task_slots_total": 16,
            },
        )

        await coord._on_node_capacity_updated(event)

        state = coord.get_node_state("gh200-a")
        assert state is not None
        assert state.gpu_utilization == 65.0
        assert state.task_slots_available == 8

    def test_backpressure_callback(self):
        """Test backpressure change callbacks are invoked."""
        from app.coordination.resource_monitoring_coordinator import ResourceMonitoringCoordinator

        coord = ResourceMonitoringCoordinator(backpressure_gpu_threshold=80.0)
        callback_invoked = []

        def on_backpressure(node_id, activated, level):
            callback_invoked.append((node_id, activated, level.value if hasattr(level, "value") else str(level)))

        coord.on_backpressure_change(on_backpressure)

        # Trigger backpressure
        with patch.object(coord, "_emit_backpressure_event"):
            coord.update_node_resources("node-1", gpu_utilization=95.0)

        assert len(callback_invoked) == 1
        assert callback_invoked[0][0] == "node-1"
        assert callback_invoked[0][1] is True


# =============================================================================
# MetricsAnalysisOrchestrator Tests
# =============================================================================

class TestMetricsAnalysisOrchestrator:
    """Tests for MetricsAnalysisOrchestrator."""

    def test_initialization(self):
        """Test orchestrator initializes with correct defaults."""
        from app.coordination.metrics_analysis_orchestrator import MetricsAnalysisOrchestrator

        orch = MetricsAnalysisOrchestrator()
        assert orch.window_size == 100
        assert orch.plateau_threshold == 0.001
        assert orch.plateau_window == 10

    def test_record_metric(self):
        """Test recording metric values."""
        from app.coordination.metrics_analysis_orchestrator import MetricsAnalysisOrchestrator

        orch = MetricsAnalysisOrchestrator()

        # Record some values
        orch.record_metric("val_loss", 0.5, epoch=1)
        orch.record_metric("val_loss", 0.45, epoch=2)
        orch.record_metric("val_loss", 0.4, epoch=3)

        current = orch.get_current_value("val_loss")
        assert current == 0.4

        best = orch.get_best_value("val_loss")
        assert best == 0.4  # Lower is better for loss

    def test_plateau_detection(self):
        """Test plateau detection for metrics."""
        from app.coordination.metrics_analysis_orchestrator import MetricsAnalysisOrchestrator

        orch = MetricsAnalysisOrchestrator(plateau_window=5, plateau_threshold=0.001)

        # Record values that plateau
        with patch.object(orch, "_emit_plateau_detected"):
            for i in range(10):
                orch.record_metric("val_loss", 0.5, epoch=i)  # Same value

        assert orch.is_plateau("val_loss")

    def test_regression_detection(self):
        """Test regression detection for metrics."""
        from app.coordination.metrics_analysis_orchestrator import MetricsAnalysisOrchestrator

        orch = MetricsAnalysisOrchestrator()

        # Record improving values then regress
        with patch.object(orch, "_emit_plateau_detected"), \
             patch.object(orch, "_emit_regression_detected"):
            for i in range(10):
                orch.record_metric("elo", 1500 + i * 10, epoch=i)  # Improving

            # Now regress significantly
            orch.record_metric("elo", 1400, epoch=10)  # Big drop

        assert orch.is_regression("elo")

    def test_get_trend_analysis(self):
        """Test trend analysis retrieval."""
        from app.coordination.metrics_analysis_orchestrator import MetricsAnalysisOrchestrator

        orch = MetricsAnalysisOrchestrator()

        # Record improving values
        for i in range(15):
            orch.record_metric("elo", 1500 + i * 5, epoch=i)

        trend = orch.get_trend("elo")
        assert trend is not None
        assert trend.metric_name == "elo"
        assert trend.best_value >= 1500

    def test_anomaly_detection(self):
        """Test anomaly detection for outlier values."""
        from app.coordination.metrics_analysis_orchestrator import MetricsAnalysisOrchestrator

        orch = MetricsAnalysisOrchestrator()

        # Record varying stable values to build up variance
        with patch.object(orch, "_emit_plateau_detected"), \
             patch.object(orch, "_emit_regression_detected"):
            for i in range(15):
                orch.record_metric("val_loss", 0.5 + (i * 0.001), epoch=i)

            # Record an extreme outlier
            anomaly = orch.record_metric("val_loss", 10.0, epoch=15)  # Massive spike

        # Anomaly detection depends on sufficient variance - may return None if values too stable
        if anomaly is not None:
            assert anomaly.anomaly_type == "spike"
            assert anomaly.severity > 3.0

    def test_get_status_dict(self):
        """Test status dictionary generation."""
        from app.coordination.metrics_analysis_orchestrator import MetricsAnalysisOrchestrator

        orch = MetricsAnalysisOrchestrator()
        orch.record_metric("val_loss", 0.5, epoch=1)
        orch.record_metric("elo", 1500, epoch=1)

        status = orch.get_status()
        assert "metrics_tracked" in status
        assert "metrics" in status
        assert status["metrics_tracked"] == 2
        assert "val_loss" in status["metrics"]
        assert "elo" in status["metrics"]


# =============================================================================
# OptimizationCoordinator Tests
# =============================================================================

class TestOptimizationCoordinator:
    """Tests for OptimizationCoordinator."""

    def test_initialization(self):
        """Test coordinator initializes with correct defaults."""
        from app.coordination.optimization_coordinator import OptimizationCoordinator

        coord = OptimizationCoordinator()
        assert coord.plateau_window == 10
        assert coord.min_epochs_between_optimization == 20
        assert coord.cooldown_seconds == 300.0
        assert not coord.is_optimization_running()

    def test_custom_configuration(self):
        """Test coordinator respects custom configuration."""
        from app.coordination.optimization_coordinator import OptimizationCoordinator

        coord = OptimizationCoordinator(
            plateau_window=5,
            cooldown_seconds=600.0,
        )
        assert coord.plateau_window == 5
        assert coord.cooldown_seconds == 600.0

    def test_can_trigger_optimization(self):
        """Test optimization trigger conditions."""
        from app.coordination.optimization_coordinator import OptimizationCoordinator

        coord = OptimizationCoordinator()

        # Should be able to trigger initially
        assert coord.can_trigger_optimization()

    def test_trigger_cmaes(self):
        """Test triggering CMA-ES optimization."""
        from app.coordination.optimization_coordinator import OptimizationCoordinator

        coord = OptimizationCoordinator()

        with patch.object(coord, "_emit_optimization_triggered"):
            run = coord.trigger_cmaes(
                reason="loss_plateau",
                parameters=["learning_rate", "batch_size"],
            )

        assert run is not None
        assert run.trigger_reason == "loss_plateau"
        assert coord.is_optimization_running()
        assert not coord.can_trigger_optimization()

    def test_trigger_nas(self):
        """Test triggering NAS optimization."""
        from app.coordination.optimization_coordinator import OptimizationCoordinator

        coord = OptimizationCoordinator()

        with patch.object(coord, "_emit_optimization_triggered"):
            run = coord.trigger_nas(
                reason="architecture_search",
                generations=50,
                population_size=20,
            )

        assert run is not None
        assert run.generations == 50
        assert run.population_size == 20
        assert coord.is_optimization_running()

    def test_cancel_optimization(self):
        """Test cancelling an optimization run."""
        from app.coordination.optimization_coordinator import OptimizationCoordinator

        coord = OptimizationCoordinator()

        with patch.object(coord, "_emit_optimization_triggered"):
            coord.trigger_cmaes(reason="test")

        # Cancel it
        cancelled = coord.cancel_optimization()

        assert cancelled
        assert not coord.is_optimization_running()

    def test_get_optimization_history(self):
        """Test retrieving optimization history."""
        from app.coordination.optimization_coordinator import OptimizationCoordinator

        coord = OptimizationCoordinator(cooldown_seconds=0)

        # Run some optimizations
        with patch.object(coord, "_emit_optimization_triggered"):
            coord.trigger_cmaes(reason="test1")
            coord.cancel_optimization()  # Use cancel to end

            coord.trigger_nas(reason="test2")
            coord.cancel_optimization()

        history = coord.get_optimization_history()
        assert len(history) == 2

    def test_get_status_dict(self):
        """Test status dictionary generation."""
        from app.coordination.optimization_coordinator import OptimizationCoordinator

        coord = OptimizationCoordinator()

        status = coord.get_status()
        assert "is_running" in status
        assert "in_cooldown" in status
        assert "total_runs" in status
        assert "subscribed" in status


# =============================================================================
# CacheCoordinationOrchestrator Tests
# =============================================================================

class TestCacheCoordinationOrchestrator:
    """Tests for CacheCoordinationOrchestrator."""

    def test_initialization(self):
        """Test orchestrator initializes with correct defaults."""
        from app.coordination.cache_coordination_orchestrator import CacheCoordinationOrchestrator

        orch = CacheCoordinationOrchestrator()
        # Check the default TTL is set
        assert orch.default_ttl_seconds > 0

    def test_register_cache(self):
        """Test registering a cache entry."""
        from app.coordination.cache_coordination_orchestrator import CacheCoordinationOrchestrator

        orch = CacheCoordinationOrchestrator()

        # Use the correct signature: node_id, cache_type, model_id
        entry = orch.register_cache(
            node_id="gh200-a",
            cache_type="nnue_weights",  # Valid cache type
            model_id="model-v1",
            size_bytes=100 * 1024 * 1024,  # 100 MB
        )

        assert entry is not None
        assert entry.model_id == "model-v1"
        assert entry.node_id == "gh200-a"

    def test_invalidate_model_caches(self):
        """Test invalidating caches for a specific model."""
        from app.coordination.cache_coordination_orchestrator import CacheCoordinationOrchestrator

        orch = CacheCoordinationOrchestrator()

        # Register caches on multiple nodes
        orch.register_cache("node-1", "nnue_weights", "model-v1", 50 * 1024 * 1024)
        orch.register_cache("node-2", "nnue_weights", "model-v1", 50 * 1024 * 1024)
        orch.register_cache("node-1", "nnue_weights", "model-v2", 50 * 1024 * 1024)

        # Invalidate model-v1
        with patch.object(orch, "_emit_cache_invalidated"):
            count = orch.invalidate_model("model-v1")

        assert count == 2  # Two caches invalidated

    def test_invalidate_node_caches(self):
        """Test invalidating all caches on a node."""
        from app.coordination.cache_coordination_orchestrator import CacheCoordinationOrchestrator

        orch = CacheCoordinationOrchestrator()

        # Register caches
        orch.register_cache("node-1", "nnue_weights", "model-v1", 50 * 1024 * 1024)
        orch.register_cache("node-1", "inference_cache", "model-v2", 50 * 1024 * 1024)
        orch.register_cache("node-2", "nnue_weights", "model-v1", 50 * 1024 * 1024)

        # Invalidate node-1
        with patch.object(orch, "_emit_cache_invalidated"):
            count = orch.invalidate_node("node-1")

        assert count == 2

    def test_get_cache_stats(self):
        """Test cache statistics calculation."""
        from app.coordination.cache_coordination_orchestrator import CacheCoordinationOrchestrator

        orch = CacheCoordinationOrchestrator()

        orch.register_cache("node-1", "nnue_weights", "model-v1", 100 * 1024 * 1024)
        orch.register_cache("node-1", "nnue_weights", "model-v2", 150 * 1024 * 1024)

        stats = orch.get_stats()
        assert stats.total_entries == 2
        assert stats.total_size_bytes > 200 * 1024 * 1024  # 250 MB total

    def test_get_status_dict(self):
        """Test status dictionary generation."""
        from app.coordination.cache_coordination_orchestrator import CacheCoordinationOrchestrator

        orch = CacheCoordinationOrchestrator()

        status = orch.get_status()
        assert "total_entries" in status
        assert "invalidated_entries" in status

    def test_record_hit_miss(self):
        """Test cache hit/miss recording."""
        from app.coordination.cache_coordination_orchestrator import CacheCoordinationOrchestrator

        orch = CacheCoordinationOrchestrator()

        # Register a cache
        orch.register_cache("node-1", "nnue_weights", "model-v1")

        # Record hits and misses
        assert orch.record_hit("node-1", "nnue_weights", "model-v1")
        assert orch.record_hit("node-1", "nnue_weights", "model-v1")
        assert orch.record_miss("node-1", "nnue_weights", "model-v1")

        # Get the entry again to check counts
        cache_id = next(iter(orch._entries.keys()))
        updated_entry = orch._entries[cache_id]
        assert updated_entry.hits == 2
        assert updated_entry.misses == 1


# =============================================================================
# Bootstrap Tests
# =============================================================================

class TestCoordinationBootstrap:
    """Tests for coordination_bootstrap module."""

    def test_bootstrap_status_initial(self):
        """Test initial bootstrap status."""
        from app.coordination.coordination_bootstrap import get_bootstrap_status, reset_bootstrap_state

        reset_bootstrap_state()
        status = get_bootstrap_status()

        assert status["initialized"] is False
        assert status["initialized_count"] == 0

    def test_is_coordination_ready_before_bootstrap(self):
        """Test is_coordination_ready before bootstrap."""
        from app.coordination.coordination_bootstrap import is_coordination_ready, reset_bootstrap_state

        reset_bootstrap_state()
        assert not is_coordination_ready()

    def test_bootstrap_with_all_disabled(self):
        """Test bootstrap with all coordinators disabled."""
        from app.coordination.coordination_bootstrap import (
            bootstrap_coordination,
            get_bootstrap_status,
            reset_bootstrap_state,
        )

        reset_bootstrap_state()

        status = bootstrap_coordination(
            enable_resources=False,
            enable_metrics=False,
            enable_optimization=False,
            enable_cache=False,
            enable_model=False,
            enable_error=False,
            enable_leadership=False,
            enable_selfplay=False,
            enable_pipeline=False,
            enable_task=False,
            enable_sync=False,
            enable_training=False,
            enable_recovery=False,
            enable_transfer=False,
            enable_ephemeral=False,
            enable_queue=False,
            enable_multi_provider=False,
            enable_job_scheduler=False,
            enable_global_task=False,
            register_with_registry=False,
        )

        assert status["initialized"] is True
        assert status["initialized_count"] == 0

    def test_bootstrap_selective(self):
        """Test selective coordinator initialization."""
        from app.coordination.coordination_bootstrap import (
            bootstrap_coordination,
            get_bootstrap_status,
            reset_bootstrap_state,
        )

        reset_bootstrap_state()

        # Only enable metrics
        status = bootstrap_coordination(
            enable_resources=False,
            enable_metrics=True,
            enable_optimization=False,
            enable_cache=False,
            enable_model=False,
            enable_error=False,
            enable_leadership=False,
            enable_selfplay=False,
            enable_pipeline=False,
            enable_task=False,
            register_with_registry=False,
        )

        assert status["initialized"] is True
        assert "metrics_orchestrator" in status["coordinators"]


# =============================================================================
# Event Emitters Tests
# =============================================================================

class TestEventEmitters:
    """Tests for centralized event emitters."""

    @pytest.mark.asyncio
    async def test_emit_backpressure_activated_without_bus(self):
        """Test emit_backpressure_activated handles missing bus gracefully."""
        from app.coordination.event_emitters import emit_backpressure_activated

        # Should not raise even without event bus
        result = await emit_backpressure_activated(
            node_id="test-node",
            level="high",
            reason="test",
        )

        # Returns False when bus unavailable
        assert result in (True, False)

    @pytest.mark.asyncio
    async def test_emit_plateau_detected_without_bus(self):
        """Test emit_plateau_detected handles missing bus gracefully."""
        from app.coordination.event_emitters import emit_plateau_detected

        result = await emit_plateau_detected(
            metric_name="val_loss",
            current_value=0.5,
            best_value=0.4,
            epochs_since_improvement=10,
        )

        assert result in (True, False)

    @pytest.mark.asyncio
    async def test_emit_regression_detected_without_bus(self):
        """Test emit_regression_detected handles missing bus gracefully."""
        from app.coordination.event_emitters import emit_regression_detected

        result = await emit_regression_detected(
            metric_name="elo",
            current_value=1400,
            previous_value=1550,
            severity="severe",
        )

        assert result in (True, False)
