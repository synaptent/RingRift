"""Tests for MetricsManager from p2p module.

Tests cover:
- Metric recording and buffering
- Flush behavior (automatic and manual)
- History retrieval with filters
- Summary aggregation
- Thread safety
"""

from __future__ import annotations

import tempfile
import threading
import time
import sqlite3
from pathlib import Path

import pytest

from scripts.p2p.metrics_manager import MetricsManager, MetricsManagerMixin


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_metrics.db"


@pytest.fixture
def manager(temp_db):
    """Create a MetricsManager instance with large buffer to avoid auto-flush."""
    return MetricsManager(temp_db, flush_interval=3600.0, max_buffer=1000)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestInitialization:
    """Test MetricsManager initialization."""

    def test_init_creates_table(self, temp_db):
        """Test that initialization creates the metrics table."""
        manager = MetricsManager(temp_db)

        # Table should exist
        import sqlite3

        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='metrics_history'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_init_custom_settings(self, temp_db):
        """Test initialization with custom settings."""
        manager = MetricsManager(temp_db, flush_interval=60.0, max_buffer=50)

        assert manager._metrics_flush_interval == 60.0
        assert manager._metrics_max_buffer == 50


# =============================================================================
# Recording Tests
# =============================================================================


class TestRecordMetric:
    """Test metric recording."""

    def test_record_adds_to_buffer(self, manager):
        """Test that recording adds to buffer."""
        manager.record_metric("test_metric", 42.0)

        assert manager.get_pending_count() == 1

    def test_record_with_metadata(self, manager):
        """Test recording with metadata."""
        manager.record_metric(
            "gpu_utilization",
            85.5,
            board_type="hex8",
            num_players=2,
            metadata={"node": "test-node"},
        )

        assert manager.get_pending_count() == 1

    def test_record_triggers_flush_on_max_buffer(self, temp_db):
        """Test that recording triggers flush when buffer is full."""
        manager = MetricsManager(temp_db, max_buffer=5)

        for i in range(6):
            manager.record_metric("test", float(i))

        # Should have flushed, so pending should be 1 (the 6th entry)
        # or 0 if it flushed again
        assert manager.get_pending_count() <= 1


# =============================================================================
# Flush Tests
# =============================================================================


class TestFlush:
    """Test flush behavior."""

    def test_flush_empty_buffer(self, manager):
        """Test flushing empty buffer returns 0."""
        count = manager.flush()
        assert count == 0

    def test_flush_clears_buffer(self, manager):
        """Test flushing clears the buffer."""
        manager.record_metric("test", 1.0)
        manager.record_metric("test", 2.0)

        count = manager.flush()

        assert count == 2
        assert manager.get_pending_count() == 0

    def test_flush_writes_to_database(self, manager, temp_db):
        """Test that flush writes to database."""
        manager.record_metric("gpu_util", 75.0, board_type="hex8")
        manager.flush()

        import sqlite3

        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM metrics_history")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1


# =============================================================================
# History Tests
# =============================================================================


class TestGetHistory:
    """Test history retrieval."""

    def test_get_history_empty(self, manager):
        """Test getting history when empty."""
        history = manager.get_history("nonexistent")
        assert history == []

    def test_get_history_basic(self, manager):
        """Test basic history retrieval."""
        manager.record_metric("cpu_util", 50.0)
        manager.record_metric("cpu_util", 60.0)
        manager.flush()

        history = manager.get_history("cpu_util")

        assert len(history) == 2
        assert history[0]["value"] == 60.0  # Most recent first
        assert history[1]["value"] == 50.0

    def test_get_history_breaks_timestamp_ties_by_id(self, manager):
        """History should stay newest-first even when timestamps tie."""
        timestamp = time.time()

        conn = sqlite3.connect(str(manager.db_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO metrics_history
            (timestamp, metric_type, board_type, num_players, value, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (timestamp, "cpu_util", None, None, 50.0, None),
        )
        cursor.execute(
            """
            INSERT INTO metrics_history
            (timestamp, metric_type, board_type, num_players, value, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (timestamp, "cpu_util", None, None, 60.0, None),
        )
        conn.commit()
        conn.close()

        history = manager.get_history("cpu_util")

        assert len(history) == 2
        assert history[0]["value"] == 60.0
        assert history[1]["value"] == 50.0

    def test_get_history_with_board_filter(self, manager):
        """Test history retrieval with board type filter."""
        manager.record_metric("loss", 0.5, board_type="hex8")
        manager.record_metric("loss", 0.6, board_type="square8")
        manager.flush()

        history = manager.get_history("loss", board_type="hex8")

        assert len(history) == 1
        assert history[0]["value"] == 0.5

    def test_get_history_with_player_filter(self, manager):
        """Test history retrieval with player count filter."""
        manager.record_metric("elo", 1500.0, num_players=2)
        manager.record_metric("elo", 1400.0, num_players=4)
        manager.flush()

        history = manager.get_history("elo", num_players=2)

        assert len(history) == 1
        assert history[0]["value"] == 1500.0

    def test_get_history_respects_limit(self, manager):
        """Test history respects limit parameter."""
        for i in range(10):
            manager.record_metric("test", float(i))
        manager.flush()

        history = manager.get_history("test", limit=5)

        assert len(history) == 5


# =============================================================================
# Summary Tests
# =============================================================================


class TestGetSummary:
    """Test summary aggregation."""

    def test_get_summary_empty(self, manager):
        """Test getting summary when empty."""
        summary = manager.get_summary()

        assert "period_hours" in summary
        assert summary.get("metrics", {}) == {}

    def test_get_summary_basic(self, manager):
        """Test basic summary aggregation."""
        manager.record_metric("gpu_util", 50.0)
        manager.record_metric("gpu_util", 70.0)
        manager.record_metric("gpu_util", 90.0)
        manager.flush()

        summary = manager.get_summary()

        assert "gpu_util" in summary["metrics"]
        gpu_summary = summary["metrics"]["gpu_util"]
        assert gpu_summary["count"] == 3
        assert gpu_summary["avg"] == 70.0
        assert gpu_summary["min"] == 50.0
        assert gpu_summary["max"] == 90.0

    def test_get_summary_multiple_types(self, manager):
        """Test summary with multiple metric types."""
        manager.record_metric("cpu", 40.0)
        manager.record_metric("gpu", 80.0)
        manager.flush()

        summary = manager.get_summary()

        assert "cpu" in summary["metrics"]
        assert "gpu" in summary["metrics"]


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Test thread safety of MetricsManager."""

    def test_concurrent_recording(self, manager):
        """Test concurrent metric recording."""
        errors = []

        def record_metrics():
            try:
                for i in range(100):
                    manager.record_metric("concurrent", float(i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_metrics) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

        # Flush and verify
        manager.flush()
        history = manager.get_history("concurrent", limit=1000)
        assert len(history) == 500  # 5 threads * 100 records


# =============================================================================
# Mixin Tests
# =============================================================================


class TestMetricsManagerMixin:
    """Test MetricsManagerMixin."""

    def test_mixin_delegates_to_manager(self, temp_db):
        """Test that mixin methods delegate to manager."""

        class TestOrchestrator(MetricsManagerMixin):
            def __init__(self, db_path):
                self._metrics_manager = MetricsManager(db_path)

        orch = TestOrchestrator(temp_db)
        orch.record_metric("test", 42.0)
        orch._flush_metrics_buffer()

        history = orch.get_metrics_history("test")
        assert len(history) == 1

        summary = orch.get_metrics_summary()
        assert "test" in summary["metrics"]
