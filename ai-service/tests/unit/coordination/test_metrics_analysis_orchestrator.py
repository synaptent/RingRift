"""Tests for MetricsAnalysisOrchestrator (trend detection).

Tests cover:
- TrendDirection and MetricType enums
- MetricPoint, TrendAnalysis, AnomalyDetection dataclasses
- MetricTracker class
- MetricsAnalysisOrchestrator event handling
- Module functions
"""

import time
import pytest
from unittest.mock import MagicMock, patch


# =============================================================================
# Test TrendDirection Enum
# =============================================================================

class TestTrendDirection:
    """Tests for TrendDirection enum."""

    def test_improving_value(self):
        """Test IMPROVING direction value."""
        from app.coordination.metrics_analysis_orchestrator import TrendDirection
        assert TrendDirection.IMPROVING.value == "improving"

    def test_degrading_value(self):
        """Test DEGRADING direction value."""
        from app.coordination.metrics_analysis_orchestrator import TrendDirection
        assert TrendDirection.DEGRADING.value == "degrading"

    def test_stable_value(self):
        """Test STABLE direction value."""
        from app.coordination.metrics_analysis_orchestrator import TrendDirection
        assert TrendDirection.STABLE.value == "stable"

    def test_plateau_value(self):
        """Test PLATEAU direction value."""
        from app.coordination.metrics_analysis_orchestrator import TrendDirection
        assert TrendDirection.PLATEAU.value == "plateau"

    def test_unknown_value(self):
        """Test UNKNOWN direction value."""
        from app.coordination.metrics_analysis_orchestrator import TrendDirection
        assert TrendDirection.UNKNOWN.value == "unknown"


# =============================================================================
# Test MetricType Enum
# =============================================================================

class TestMetricType:
    """Tests for MetricType enum."""

    def test_minimize_value(self):
        """Test MINIMIZE type value."""
        from app.coordination.metrics_analysis_orchestrator import MetricType
        assert MetricType.MINIMIZE.value == "minimize"

    def test_maximize_value(self):
        """Test MAXIMIZE type value."""
        from app.coordination.metrics_analysis_orchestrator import MetricType
        assert MetricType.MAXIMIZE.value == "maximize"


# =============================================================================
# Test MetricPoint Dataclass
# =============================================================================

class TestMetricPoint:
    """Tests for MetricPoint dataclass."""

    def test_create_with_value(self):
        """Test creating metric point with value."""
        from app.coordination.metrics_analysis_orchestrator import MetricPoint

        point = MetricPoint(value=0.95)
        assert point.value == 0.95
        assert point.epoch == 0
        assert point.metadata == {}

    def test_create_full(self):
        """Test creating metric point with all fields."""
        from app.coordination.metrics_analysis_orchestrator import MetricPoint

        point = MetricPoint(
            value=0.85,
            epoch=10,
            metadata={"batch": 1000},
        )
        assert point.value == 0.85
        assert point.epoch == 10
        assert point.metadata["batch"] == 1000

    def test_timestamp_default(self):
        """Test timestamp is auto-populated."""
        from app.coordination.metrics_analysis_orchestrator import MetricPoint

        before = time.time()
        point = MetricPoint(value=1.0)
        after = time.time()

        assert before <= point.timestamp <= after


# =============================================================================
# Test TrendAnalysis Dataclass
# =============================================================================

class TestTrendAnalysis:
    """Tests for TrendAnalysis dataclass."""

    def test_create(self):
        """Test creating trend analysis."""
        from app.coordination.metrics_analysis_orchestrator import (
            TrendAnalysis,
            TrendDirection,
        )

        analysis = TrendAnalysis(
            metric_name="val_loss",
            direction=TrendDirection.IMPROVING,
            current_value=0.15,
            best_value=0.12,
            worst_value=0.35,
            change_rate=-0.01,
            std_dev=0.02,
            samples=50,
            is_plateau=False,
            is_regression=False,
        )

        assert analysis.metric_name == "val_loss"
        assert analysis.direction == TrendDirection.IMPROVING
        assert analysis.current_value == 0.15
        assert analysis.samples == 50

    def test_default_values(self):
        """Test default values."""
        from app.coordination.metrics_analysis_orchestrator import (
            TrendAnalysis,
            TrendDirection,
        )

        analysis = TrendAnalysis(
            metric_name="test",
            direction=TrendDirection.UNKNOWN,
            current_value=0.0,
            best_value=0.0,
            worst_value=0.0,
            change_rate=0.0,
            std_dev=0.0,
            samples=0,
            is_plateau=False,
            is_regression=False,
        )

        assert analysis.plateau_epochs == 0
        assert analysis.regression_severity == 0.0
        assert analysis.window_start == 0.0
        assert analysis.window_end == 0.0


# =============================================================================
# Test AnomalyDetection Dataclass
# =============================================================================

class TestAnomalyDetection:
    """Tests for AnomalyDetection dataclass."""

    def test_create(self):
        """Test creating anomaly detection."""
        from app.coordination.metrics_analysis_orchestrator import AnomalyDetection

        anomaly = AnomalyDetection(
            metric_name="train_loss",
            anomaly_type="spike",
            value=1.5,
            expected_range=(0.1, 0.3),
            severity=5.0,
        )

        assert anomaly.metric_name == "train_loss"
        assert anomaly.anomaly_type == "spike"
        assert anomaly.value == 1.5
        assert anomaly.expected_range == (0.1, 0.3)
        assert anomaly.severity == 5.0


# =============================================================================
# Test MetricTracker
# =============================================================================

class TestMetricTracker:
    """Tests for MetricTracker class."""

    def test_initialization(self):
        """Test tracker initialization."""
        from app.coordination.metrics_analysis_orchestrator import (
            MetricTracker,
            MetricType,
        )

        tracker = MetricTracker(
            name="loss",
            metric_type=MetricType.MINIMIZE,
        )

        assert tracker.name == "loss"
        assert tracker.metric_type == MetricType.MINIMIZE
        assert tracker.window_size == 100
        assert len(tracker._history) == 0

    def test_add_point(self):
        """Test adding data points."""
        from app.coordination.metrics_analysis_orchestrator import (
            MetricTracker,
            MetricType,
        )

        tracker = MetricTracker(name="loss", metric_type=MetricType.MINIMIZE)

        tracker.add_point(0.5, epoch=1)
        tracker.add_point(0.4, epoch=2)
        tracker.add_point(0.3, epoch=3)

        assert len(tracker._history) == 3
        assert tracker._best_value == 0.3

    def test_get_values(self):
        """Test getting values."""
        from app.coordination.metrics_analysis_orchestrator import (
            MetricTracker,
            MetricType,
        )

        tracker = MetricTracker(name="elo", metric_type=MetricType.MAXIMIZE)
        tracker.add_point(1500)
        tracker.add_point(1520)
        tracker.add_point(1510)

        values = tracker.get_values()
        assert values == [1500, 1520, 1510]

    def test_get_mean(self):
        """Test getting mean."""
        from app.coordination.metrics_analysis_orchestrator import (
            MetricTracker,
            MetricType,
        )

        tracker = MetricTracker(name="test", metric_type=MetricType.MINIMIZE)
        tracker.add_point(10)
        tracker.add_point(20)
        tracker.add_point(30)

        assert tracker.get_mean() == 20.0

    def test_get_std_dev(self):
        """Test getting standard deviation."""
        from app.coordination.metrics_analysis_orchestrator import (
            MetricTracker,
            MetricType,
        )

        tracker = MetricTracker(name="test", metric_type=MetricType.MINIMIZE)
        tracker.add_point(10)
        tracker.add_point(20)
        tracker.add_point(30)

        std = tracker.get_std_dev()
        assert std == pytest.approx(10.0, abs=0.01)

    def test_plateau_detection_minimize(self):
        """Test plateau detection for minimize metric."""
        from app.coordination.metrics_analysis_orchestrator import (
            MetricTracker,
            MetricType,
        )

        tracker = MetricTracker(
            name="loss",
            metric_type=MetricType.MINIMIZE,
            plateau_window=5,
            plateau_threshold=0.01,
        )

        # Add improving values
        tracker.add_point(0.5)
        tracker.add_point(0.4)
        tracker.add_point(0.3)

        assert tracker.is_plateau() is False

        # Add stable values
        for _ in range(7):
            tracker.add_point(0.3)

        assert tracker.is_plateau() is True

    def test_plateau_detection_maximize(self):
        """Test plateau detection for maximize metric."""
        from app.coordination.metrics_analysis_orchestrator import (
            MetricTracker,
            MetricType,
        )

        tracker = MetricTracker(
            name="elo",
            metric_type=MetricType.MAXIMIZE,
            plateau_window=5,
        )

        # Add improving values
        tracker.add_point(1500)
        tracker.add_point(1550)
        tracker.add_point(1600)

        assert tracker.is_plateau() is False

        # Add stable values
        for _ in range(7):
            tracker.add_point(1600)

        assert tracker.is_plateau() is True

    def test_regression_detection_minimize(self):
        """Test regression detection for minimize metric."""
        from app.coordination.metrics_analysis_orchestrator import (
            MetricTracker,
            MetricType,
        )

        tracker = MetricTracker(
            name="loss",
            metric_type=MetricType.MINIMIZE,
            regression_threshold=0.05,
        )

        # Set best value
        tracker.add_point(0.1)

        # Add regressed value
        tracker.add_point(0.2)  # 100% worse

        assert tracker.is_regression() is True

    def test_regression_detection_maximize(self):
        """Test regression detection for maximize metric."""
        from app.coordination.metrics_analysis_orchestrator import (
            MetricTracker,
            MetricType,
        )

        tracker = MetricTracker(
            name="elo",
            metric_type=MetricType.MAXIMIZE,
            regression_threshold=0.05,
        )

        # Set best value
        tracker.add_point(1600)

        # Add regressed value
        tracker.add_point(1400)  # Significant drop

        assert tracker.is_regression() is True

    def test_anomaly_detection(self):
        """Test anomaly detection."""
        from app.coordination.metrics_analysis_orchestrator import (
            MetricTracker,
            MetricType,
        )

        tracker = MetricTracker(
            name="loss",
            metric_type=MetricType.MINIMIZE,
            anomaly_threshold=3.0,
        )

        # Add normal values
        for i in range(15):
            tracker.add_point(0.2 + i * 0.001)  # Slight trend

        # Add anomaly
        tracker.add_point(1.0)  # Big spike

        anomaly = tracker.check_anomaly()
        assert anomaly is not None
        assert anomaly.anomaly_type == "spike"

    def test_trend_direction_improving(self):
        """Test trend direction detection for improving metric."""
        from app.coordination.metrics_analysis_orchestrator import (
            MetricTracker,
            MetricType,
            TrendDirection,
        )

        tracker = MetricTracker(
            name="loss",
            metric_type=MetricType.MINIMIZE,
            plateau_window=5,
        )

        # Add steadily decreasing values
        for i in range(15):
            tracker.add_point(1.0 - i * 0.05)

        direction = tracker.get_trend_direction()
        assert direction == TrendDirection.IMPROVING

    def test_trend_direction_degrading(self):
        """Test trend direction detection for degrading metric."""
        from app.coordination.metrics_analysis_orchestrator import (
            MetricTracker,
            MetricType,
            TrendDirection,
        )

        tracker = MetricTracker(
            name="elo",
            metric_type=MetricType.MAXIMIZE,  # Maximize: decreasing is degrading
            plateau_window=10,  # Larger window so we don't hit plateau
            plateau_threshold=0.001,
        )

        # Start with an improvement to reset epochs_since_improvement
        tracker.add_point(1500)
        tracker.add_point(1600)  # Improvement

        # Then add degrading values (but fewer than plateau_window)
        for i in range(8):
            tracker.add_point(1550 - i * 30)  # Degrading trend

        direction = tracker.get_trend_direction()
        # For maximize metric, decreasing values = degrading
        assert direction == TrendDirection.DEGRADING

    def test_analyze(self):
        """Test full analysis."""
        from app.coordination.metrics_analysis_orchestrator import (
            MetricTracker,
            MetricType,
        )

        tracker = MetricTracker(
            name="val_loss",
            metric_type=MetricType.MINIMIZE,
        )

        for i in range(20):
            tracker.add_point(0.5 - i * 0.01)

        analysis = tracker.analyze()

        assert analysis.metric_name == "val_loss"
        assert analysis.samples == 20
        assert analysis.current_value == pytest.approx(0.31, abs=0.01)
        assert analysis.best_value == pytest.approx(0.31, abs=0.01)


# =============================================================================
# Test MetricsAnalysisOrchestrator
# =============================================================================

class TestMetricsAnalysisOrchestrator:
    """Tests for MetricsAnalysisOrchestrator class."""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator instance."""
        from app.coordination.metrics_analysis_orchestrator import (
            MetricsAnalysisOrchestrator,
        )
        return MetricsAnalysisOrchestrator()

    def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.window_size == 100
        assert orchestrator.plateau_threshold == 0.001
        assert orchestrator.plateau_window == 10
        assert len(orchestrator._trackers) == 0
        assert orchestrator._subscribed is False

    def test_initialization_custom(self):
        """Test initialization with custom values."""
        from app.coordination.metrics_analysis_orchestrator import (
            MetricsAnalysisOrchestrator,
        )

        orch = MetricsAnalysisOrchestrator(
            window_size=50,
            plateau_threshold=0.01,
            plateau_window=5,
        )

        assert orch.window_size == 50
        assert orch.plateau_threshold == 0.01
        assert orch.plateau_window == 5

    def test_record_metric_creates_tracker(self, orchestrator):
        """Test recording metric creates tracker."""
        orchestrator.record_metric("test_metric", 0.5)

        assert "test_metric" in orchestrator._trackers

    def test_record_multiple_metrics(self, orchestrator):
        """Test recording multiple different metrics."""
        orchestrator.record_metric("train_loss", 0.5)
        orchestrator.record_metric("val_loss", 0.6)
        orchestrator.record_metric("elo", 1500)

        assert len(orchestrator._trackers) == 3

    def test_set_metric_type(self, orchestrator):
        """Test setting metric type."""
        from app.coordination.metrics_analysis_orchestrator import MetricType

        orchestrator.set_metric_type("custom_metric", MetricType.MAXIMIZE)
        orchestrator.record_metric("custom_metric", 100)

        assert orchestrator._trackers["custom_metric"].metric_type == MetricType.MAXIMIZE

    def test_get_trend(self, orchestrator):
        """Test getting trend for metric."""
        for i in range(15):
            orchestrator.record_metric("loss", 1.0 - i * 0.05)

        trend = orchestrator.get_trend("loss")

        assert trend is not None
        assert trend.metric_name == "loss"

    def test_get_trend_not_found(self, orchestrator):
        """Test getting trend for non-existent metric."""
        trend = orchestrator.get_trend("nonexistent")
        assert trend is None

    def test_get_all_trends(self, orchestrator):
        """Test getting all trends."""
        for i in range(15):
            orchestrator.record_metric("loss", 1.0 - i * 0.05)
            orchestrator.record_metric("elo", 1500 + i * 10)

        trends = orchestrator.get_all_trends()

        assert "loss" in trends
        assert "elo" in trends
        assert len(trends) == 2

    def test_is_plateau(self, orchestrator):
        """Test plateau check."""
        # Record without plateau
        for i in range(5):
            orchestrator.record_metric("improving", 1.0 - i * 0.1)

        assert orchestrator.is_plateau("improving") is False
        assert orchestrator.is_plateau("nonexistent") is False

    def test_is_regression(self, orchestrator):
        """Test regression check."""
        assert orchestrator.is_regression("nonexistent") is False

    def test_get_current_value(self, orchestrator):
        """Test getting current value."""
        orchestrator.record_metric("test", 42.5)

        assert orchestrator.get_current_value("test") == 42.5
        assert orchestrator.get_current_value("nonexistent") is None

    def test_get_best_value(self, orchestrator):
        """Test getting best value."""
        orchestrator.record_metric("loss", 0.5)
        orchestrator.record_metric("loss", 0.3)
        orchestrator.record_metric("loss", 0.4)

        best = orchestrator.get_best_value("loss")
        assert best == 0.3  # Minimize: lower is better

    def test_on_plateau_callback(self, orchestrator):
        """Test plateau callback registration."""
        callback = MagicMock()
        orchestrator.on_plateau(callback)

        assert callback in orchestrator._plateau_callbacks

    def test_on_regression_callback(self, orchestrator):
        """Test regression callback registration."""
        callback = MagicMock()
        orchestrator.on_regression(callback)

        assert callback in orchestrator._regression_callbacks

    def test_on_anomaly_callback(self, orchestrator):
        """Test anomaly callback registration."""
        callback = MagicMock()
        orchestrator.on_anomaly(callback)

        assert callback in orchestrator._anomaly_callbacks

    def test_get_anomalies(self, orchestrator):
        """Test getting anomalies."""
        # Initially empty
        assert orchestrator.get_anomalies() == []

        # Add normal values with some variance, then spike
        for i in range(15):
            orchestrator.record_metric("anomaly_test", 0.2 + i * 0.001)  # Small variance
        orchestrator.record_metric("anomaly_test", 5.0)  # Large spike (>> 3 stdev)

        anomalies = orchestrator.get_anomalies()
        assert len(anomalies) >= 1

    def test_reset_window(self, orchestrator):
        """Test resetting a metric window."""
        orchestrator.record_metric("test", 1.0)
        orchestrator.record_metric("test", 2.0)

        assert len(orchestrator._trackers["test"]._history) == 2

        result = orchestrator.reset_window("test")
        assert result is True
        assert len(orchestrator._trackers["test"]._history) == 0

    def test_reset_window_not_found(self, orchestrator):
        """Test resetting non-existent window."""
        result = orchestrator.reset_window("nonexistent")
        assert result is False

    def test_reset_all_windows(self, orchestrator):
        """Test resetting all windows."""
        orchestrator.record_metric("a", 1.0)
        orchestrator.record_metric("b", 2.0)

        count = orchestrator.reset_all_windows()
        assert count == 2

    def test_get_status(self, orchestrator):
        """Test getting status."""
        for i in range(15):
            orchestrator.record_metric("loss", 1.0 - i * 0.05)

        status = orchestrator.get_status()

        assert "metrics_tracked" in status
        assert "metrics" in status
        assert "plateaus" in status
        assert "regressions" in status
        assert "trends" in status
        assert "subscribed" in status


# =============================================================================
# Test Module Functions
# =============================================================================

class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_metrics_orchestrator_singleton(self):
        """Test singleton pattern."""
        from app.coordination.metrics_analysis_orchestrator import (
            get_metrics_orchestrator,
        )

        orch1 = get_metrics_orchestrator()
        orch2 = get_metrics_orchestrator()

        assert orch1 is orch2

    def test_record_metric_function(self):
        """Test record_metric convenience function."""
        from app.coordination.metrics_analysis_orchestrator import (
            record_metric,
            get_metrics_orchestrator,
        )

        record_metric("convenience_test", 42.0)
        orch = get_metrics_orchestrator()

        assert orch.get_current_value("convenience_test") == 42.0

    def test_is_metric_plateau_function(self):
        """Test is_metric_plateau convenience function."""
        from app.coordination.metrics_analysis_orchestrator import is_metric_plateau

        result = is_metric_plateau("nonexistent")
        assert result is False

    def test_get_metric_trend_function(self):
        """Test get_metric_trend convenience function."""
        from app.coordination.metrics_analysis_orchestrator import (
            get_metric_trend,
            record_metric,
        )

        for i in range(15):
            record_metric("trend_test", 1.0 - i * 0.05)

        trend = get_metric_trend("trend_test")
        assert trend is not None

    def test_analyze_metrics_function(self):
        """Test analyze_metrics convenience function."""
        from app.coordination.metrics_analysis_orchestrator import (
            analyze_metrics,
            record_metric,
        )

        for i in range(15):
            record_metric("analyze_test", 0.5 + i * 0.01)

        trends = analyze_metrics()
        assert "analyze_test" in trends


# =============================================================================
# Integration Tests
# =============================================================================

class TestMetricsIntegration:
    """Integration tests for metrics analysis orchestrator."""

    def test_default_metric_types(self):
        """Test default metric type assignments."""
        from app.coordination.metrics_analysis_orchestrator import (
            MetricsAnalysisOrchestrator,
            MetricType,
        )

        orch = MetricsAnalysisOrchestrator()

        # Minimize metrics
        assert orch._metric_types["train_loss"] == MetricType.MINIMIZE
        assert orch._metric_types["val_loss"] == MetricType.MINIMIZE
        assert orch._metric_types["value_mse"] == MetricType.MINIMIZE

        # Maximize metrics
        assert orch._metric_types["elo"] == MetricType.MAXIMIZE
        assert orch._metric_types["win_rate"] == MetricType.MAXIMIZE
        assert orch._metric_types["accuracy"] == MetricType.MAXIMIZE

    def test_plateau_triggers_callback(self):
        """Test plateau detection triggers callback."""
        from app.coordination.metrics_analysis_orchestrator import (
            MetricsAnalysisOrchestrator,
        )

        orch = MetricsAnalysisOrchestrator(plateau_window=3)
        callback = MagicMock()
        orch.on_plateau(callback)

        # Record stable values to trigger plateau
        orch.record_metric("test", 0.5)
        orch.record_metric("test", 0.5)
        orch.record_metric("test", 0.5)
        orch.record_metric("test", 0.5)  # Plateau detected

        callback.assert_called()

    def test_analysis_result_alias(self):
        """Test AnalysisResult is aliased to TrendAnalysis."""
        from app.coordination.metrics_analysis_orchestrator import (
            AnalysisResult,
            TrendAnalysis,
        )

        assert AnalysisResult is TrendAnalysis
