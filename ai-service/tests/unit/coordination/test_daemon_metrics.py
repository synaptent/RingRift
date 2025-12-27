"""Unit tests for daemon_metrics.py - Prometheus-style metrics collection.

Tests the DaemonMetricsCollector class:
- Metrics rendering
- Daemon health metrics collection
- Optional subsystem metrics (selfplay, sync, event router)
- Prometheus format compliance
- Singleton pattern

December 2025: Created for daemon metrics extraction testing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Test DaemonMetricsCollector creation
# =============================================================================


class TestDaemonMetricsCollectorCreation:
    """Test DaemonMetricsCollector initialization."""

    def test_basic_creation(self):
        """Verify collector can be created with health summary function."""
        from app.coordination.daemon_metrics import DaemonMetricsCollector

        def mock_health_summary():
            return {"running": 5, "failed": 0, "total": 5, "score": 1.0}

        collector = DaemonMetricsCollector(mock_health_summary)
        assert collector is not None
        assert collector._health_summary_fn is mock_health_summary


# =============================================================================
# Test daemon health metrics
# =============================================================================


class TestDaemonHealthMetrics:
    """Test daemon health metrics collection."""

    def test_collect_daemon_health_metrics(self):
        """Verify daemon health metrics are collected correctly."""
        from app.coordination.daemon_metrics import DaemonMetricsCollector

        def mock_health_summary():
            return {
                "running": 10,
                "failed": 2,
                "total": 15,
                "score": 0.8,
                "liveness": {"uptime_seconds": 3600},
            }

        collector = DaemonMetricsCollector(mock_health_summary)
        lines = collector._collect_daemon_health_metrics()

        # Check for expected metrics
        metrics_text = "\n".join(lines)

        assert "daemon_count" in metrics_text
        assert 'state="running"} 10' in metrics_text
        assert 'state="failed"} 2' in metrics_text
        assert 'state="stopped"} 3' in metrics_text  # 15 - 10 - 2 = 3
        assert "daemon_health_score 0.8" in metrics_text
        assert "daemon_uptime_seconds 3600" in metrics_text

    def test_collect_daemon_health_metrics_handles_missing_liveness(self):
        """Verify metrics handle missing liveness data."""
        from app.coordination.daemon_metrics import DaemonMetricsCollector

        def mock_health_summary():
            return {"running": 5, "failed": 0, "total": 5, "score": 1.0}

        collector = DaemonMetricsCollector(mock_health_summary)
        lines = collector._collect_daemon_health_metrics()

        metrics_text = "\n".join(lines)
        assert "daemon_uptime_seconds 0" in metrics_text


# =============================================================================
# Test render_metrics
# =============================================================================


class TestRenderMetrics:
    """Test full metrics rendering."""

    def test_render_metrics_includes_daemon_health(self):
        """Verify render_metrics includes daemon health metrics."""
        from app.coordination.daemon_metrics import DaemonMetricsCollector

        def mock_health_summary():
            return {
                "running": 5,
                "failed": 0,
                "total": 5,
                "score": 1.0,
                "liveness": {"uptime_seconds": 100},
            }

        collector = DaemonMetricsCollector(mock_health_summary)

        # Patch subsystem collectors to return empty (avoid import issues)
        with patch.object(collector, "_collect_selfplay_metrics", return_value=[]):
            with patch.object(collector, "_collect_sync_metrics", return_value=[]):
                with patch.object(
                    collector, "_collect_event_router_metrics", return_value=[]
                ):
                    metrics = collector.render_metrics()

        assert "daemon_count" in metrics
        assert "daemon_health_score" in metrics

    def test_render_metrics_returns_string(self):
        """Verify render_metrics returns a string."""
        from app.coordination.daemon_metrics import DaemonMetricsCollector

        def mock_health_summary():
            return {"running": 0, "failed": 0, "total": 0, "score": 0.0}

        collector = DaemonMetricsCollector(mock_health_summary)

        with patch.object(collector, "_collect_selfplay_metrics", return_value=[]):
            with patch.object(collector, "_collect_sync_metrics", return_value=[]):
                with patch.object(
                    collector, "_collect_event_router_metrics", return_value=[]
                ):
                    metrics = collector.render_metrics()

        assert isinstance(metrics, str)


# =============================================================================
# Test optional subsystem metrics
# =============================================================================


class TestOptionalSubsystemMetrics:
    """Test optional metrics from subsystems."""

    def test_selfplay_metrics_handles_runtime_error(self):
        """Verify selfplay metrics handles runtime errors gracefully."""
        from app.coordination.daemon_metrics import DaemonMetricsCollector

        def mock_health_summary():
            return {"running": 0, "failed": 0, "total": 0, "score": 0.0}

        collector = DaemonMetricsCollector(mock_health_summary)

        # Mock the import path inside the method
        with patch(
            "app.coordination.selfplay_scheduler.get_selfplay_scheduler",
            side_effect=RuntimeError("mock"),
        ):
            lines = collector._collect_selfplay_metrics()

        # Returns empty list on error (graceful fallback)
        assert isinstance(lines, list)

    def test_sync_metrics_handles_runtime_error(self):
        """Verify sync metrics handles runtime errors gracefully."""
        from app.coordination.daemon_metrics import DaemonMetricsCollector

        def mock_health_summary():
            return {"running": 0, "failed": 0, "total": 0, "score": 0.0}

        collector = DaemonMetricsCollector(mock_health_summary)

        # Mock to raise runtime error
        with patch(
            "app.coordination.auto_sync_daemon.get_auto_sync_daemon",
            side_effect=RuntimeError("mock"),
        ):
            lines = collector._collect_sync_metrics()

        # Returns empty list on error (graceful fallback)
        assert isinstance(lines, list)

    def test_event_router_metrics_returns_list(self):
        """Verify event router metrics returns a list."""
        from app.coordination.daemon_metrics import DaemonMetricsCollector

        def mock_health_summary():
            return {"running": 0, "failed": 0, "total": 0, "score": 0.0}

        collector = DaemonMetricsCollector(mock_health_summary)

        # The method handles its own errors internally
        lines = collector._collect_event_router_metrics()

        # Should return a list (may be empty or have content)
        assert isinstance(lines, list)


# =============================================================================
# Test Prometheus metrics format
# =============================================================================


class TestPrometheusFormat:
    """Test Prometheus format compliance."""

    def test_metrics_have_help_comments(self):
        """Verify metrics include HELP comments."""
        from app.coordination.daemon_metrics import DaemonMetricsCollector

        def mock_health_summary():
            return {"running": 1, "failed": 0, "total": 1, "score": 1.0}

        collector = DaemonMetricsCollector(mock_health_summary)
        lines = collector._collect_daemon_health_metrics()

        metrics_text = "\n".join(lines)
        assert "# HELP daemon_count" in metrics_text
        assert "# HELP daemon_health_score" in metrics_text

    def test_metrics_have_type_comments(self):
        """Verify metrics include TYPE comments."""
        from app.coordination.daemon_metrics import DaemonMetricsCollector

        def mock_health_summary():
            return {"running": 1, "failed": 0, "total": 1, "score": 1.0}

        collector = DaemonMetricsCollector(mock_health_summary)
        lines = collector._collect_daemon_health_metrics()

        metrics_text = "\n".join(lines)
        assert "# TYPE daemon_count gauge" in metrics_text
        assert "# TYPE daemon_health_score gauge" in metrics_text

    def test_metrics_use_labels_correctly(self):
        """Verify metrics use Prometheus label syntax."""
        from app.coordination.daemon_metrics import DaemonMetricsCollector

        def mock_health_summary():
            return {"running": 5, "failed": 0, "total": 5, "score": 1.0}

        collector = DaemonMetricsCollector(mock_health_summary)
        lines = collector._collect_daemon_health_metrics()

        metrics_text = "\n".join(lines)
        # Check for proper label format: metric_name{label="value"} value
        assert 'daemon_count{state="running"} 5' in metrics_text


# =============================================================================
# Test singleton pattern
# =============================================================================


class TestSingletonPattern:
    """Test get_daemon_metrics_collector singleton."""

    def test_singleton_requires_health_fn_on_first_call(self):
        """Verify singleton raises error without health_summary_fn on first call."""
        from app.coordination.daemon_metrics import (
            _metrics_collector,
            get_daemon_metrics_collector,
        )

        # Reset singleton for test
        import app.coordination.daemon_metrics as dm

        original = dm._metrics_collector
        dm._metrics_collector = None

        try:
            with pytest.raises(ValueError, match="health_summary_fn required"):
                get_daemon_metrics_collector(health_summary_fn=None)
        finally:
            # Restore original state
            dm._metrics_collector = original

    def test_singleton_returns_same_instance(self):
        """Verify singleton returns same instance on subsequent calls."""
        from app.coordination.daemon_metrics import get_daemon_metrics_collector

        import app.coordination.daemon_metrics as dm

        original = dm._metrics_collector
        dm._metrics_collector = None

        try:

            def mock_fn():
                return {}

            c1 = get_daemon_metrics_collector(mock_fn)
            c2 = get_daemon_metrics_collector()  # No fn needed on second call

            assert c1 is c2
        finally:
            dm._metrics_collector = original


# =============================================================================
# Test edge cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_negative_stopped_count_prevented(self):
        """Verify stopped count can't go negative."""
        from app.coordination.daemon_metrics import DaemonMetricsCollector

        # Edge case: running + failed > total (shouldn't happen, but handle it)
        def mock_health_summary():
            return {
                "running": 10,
                "failed": 10,
                "total": 15,  # Less than running + failed
                "score": 0.5,
            }

        collector = DaemonMetricsCollector(mock_health_summary)
        lines = collector._collect_daemon_health_metrics()

        metrics_text = "\n".join(lines)
        # stopped = max(0, 15 - 10 - 10) = max(0, -5) = 0
        assert 'state="stopped"} 0' in metrics_text

    def test_empty_health_summary(self):
        """Verify handling of empty health summary."""
        from app.coordination.daemon_metrics import DaemonMetricsCollector

        def mock_health_summary():
            return {}

        collector = DaemonMetricsCollector(mock_health_summary)
        lines = collector._collect_daemon_health_metrics()

        metrics_text = "\n".join(lines)
        # Should use defaults (0)
        assert 'state="running"} 0' in metrics_text
        assert "daemon_health_score 0" in metrics_text
