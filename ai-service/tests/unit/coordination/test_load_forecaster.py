"""Unit tests for LoadForecaster.

Tests cluster-wide load prediction and capacity planning.
"""

from __future__ import annotations

import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.load_forecaster import (
    DurationAccuracy,
    LoadForecaster,
    LoadPrediction,
    SchedulingWindow,
    ThroughputForecast,
)


class TestLoadPrediction:
    """Tests for LoadPrediction dataclass."""

    def test_initialization_defaults(self):
        """Test default initialization values."""
        pred = LoadPrediction(timestamp=time.time(), hours_ahead=1.0)

        assert pred.active_jobs == 0
        assert pred.busy_hosts == 0
        assert pred.gpu_utilization == 0.0
        assert pred.cpu_utilization == 0.0
        assert pred.total_hosts == 0
        assert pred.gpu_hosts == 0
        assert pred.available_hosts == 0
        assert pred.confidence == 0.0

    def test_load_percentage_zero_hosts(self):
        """Test load_percentage with zero hosts returns 0."""
        pred = LoadPrediction(timestamp=time.time(), hours_ahead=1.0, total_hosts=0)
        assert pred.load_percentage == 0.0

    def test_load_percentage_calculation(self):
        """Test load_percentage calculation."""
        pred = LoadPrediction(
            timestamp=time.time(),
            hours_ahead=1.0,
            total_hosts=10,
            busy_hosts=7,
        )
        assert pred.load_percentage == 70.0

    def test_is_peak_high_load_percentage(self):
        """Test is_peak with high load percentage."""
        pred = LoadPrediction(
            timestamp=time.time(),
            hours_ahead=1.0,
            total_hosts=10,
            busy_hosts=8,  # 80% load
            gpu_utilization=0.5,
        )
        assert pred.is_peak is True

    def test_is_peak_high_gpu_utilization(self):
        """Test is_peak with high GPU utilization."""
        pred = LoadPrediction(
            timestamp=time.time(),
            hours_ahead=1.0,
            total_hosts=10,
            busy_hosts=3,  # 30% load
            gpu_utilization=0.85,
        )
        assert pred.is_peak is True

    def test_is_peak_low_load(self):
        """Test is_peak with low load."""
        pred = LoadPrediction(
            timestamp=time.time(),
            hours_ahead=1.0,
            total_hosts=10,
            busy_hosts=5,  # 50% load
            gpu_utilization=0.5,
        )
        assert pred.is_peak is False


class TestDurationAccuracy:
    """Tests for DurationAccuracy dataclass."""

    def test_initialization_defaults(self):
        """Test default initialization values."""
        accuracy = DurationAccuracy(task_type="selfplay")

        assert accuracy.task_type == "selfplay"
        assert accuracy.sample_count == 0
        assert accuracy.mean_error == 0.0
        assert accuracy.median_error == 0.0
        assert accuracy.std_error == 0.0
        assert accuracy.mean_pct_error == 0.0
        assert accuracy.accuracy_score == 1.0

    def test_needs_recalibration_insufficient_samples(self):
        """Test needs_recalibration with insufficient samples."""
        accuracy = DurationAccuracy(
            task_type="selfplay",
            sample_count=5,  # < 10
            mean_pct_error=0.5,  # > 0.3
        )
        assert accuracy.needs_recalibration is False

    def test_needs_recalibration_low_error(self):
        """Test needs_recalibration with low error."""
        accuracy = DurationAccuracy(
            task_type="selfplay",
            sample_count=20,  # >= 10
            mean_pct_error=0.1,  # <= 0.3
        )
        assert accuracy.needs_recalibration is False

    def test_needs_recalibration_true(self):
        """Test needs_recalibration is true when conditions met."""
        accuracy = DurationAccuracy(
            task_type="selfplay",
            sample_count=15,  # >= 10
            mean_pct_error=0.5,  # > 0.3
        )
        assert accuracy.needs_recalibration is True

    def test_needs_recalibration_negative_error(self):
        """Test needs_recalibration with negative error (overestimate)."""
        accuracy = DurationAccuracy(
            task_type="selfplay",
            sample_count=15,
            mean_pct_error=-0.4,  # |error| > 0.3
        )
        assert accuracy.needs_recalibration is True


class TestThroughputForecast:
    """Tests for ThroughputForecast dataclass."""

    def test_initialization_defaults(self):
        """Test default initialization values."""
        forecast = ThroughputForecast(task_type="training")

        assert forecast.task_type == "training"
        assert forecast.window_hours == 24.0
        assert forecast.current_rate == 0.0
        assert forecast.predicted_rate == 0.0
        assert forecast.trend == 0.0
        assert forecast.expected_completions == 0
        assert forecast.expected_failures == 0

    def test_custom_values(self):
        """Test with custom values."""
        forecast = ThroughputForecast(
            task_type="selfplay",
            window_hours=12.0,
            current_rate=5.0,
            predicted_rate=6.0,
            trend=0.2,
            expected_completions=72,
            expected_failures=3,
        )

        assert forecast.task_type == "selfplay"
        assert forecast.window_hours == 12.0
        assert forecast.current_rate == 5.0
        assert forecast.predicted_rate == 6.0
        assert forecast.trend == 0.2
        assert forecast.expected_completions == 72
        assert forecast.expected_failures == 3


class TestSchedulingWindow:
    """Tests for SchedulingWindow dataclass."""

    def test_initialization(self):
        """Test initialization."""
        now = time.time()
        window = SchedulingWindow(
            start_timestamp=now,
            end_timestamp=now + 3600,  # 1 hour later
            expected_load=0.3,
            reason="low_load",
        )

        assert window.start_timestamp == now
        assert window.end_timestamp == now + 3600
        assert window.expected_load == 0.3
        assert window.reason == "low_load"

    def test_start_time_property(self):
        """Test start_time returns datetime."""
        now = time.time()
        window = SchedulingWindow(
            start_timestamp=now,
            end_timestamp=now + 3600,
            expected_load=0.3,
            reason="test",
        )

        start = window.start_time
        assert isinstance(start, datetime)
        assert abs(start.timestamp() - now) < 1

    def test_end_time_property(self):
        """Test end_time returns datetime."""
        now = time.time()
        window = SchedulingWindow(
            start_timestamp=now,
            end_timestamp=now + 7200,
            expected_load=0.3,
            reason="test",
        )

        end = window.end_time
        assert isinstance(end, datetime)
        assert abs(end.timestamp() - (now + 7200)) < 1

    def test_duration_hours_property(self):
        """Test duration_hours calculation."""
        now = time.time()
        window = SchedulingWindow(
            start_timestamp=now,
            end_timestamp=now + 7200,  # 2 hours
            expected_load=0.3,
            reason="test",
        )

        assert window.duration_hours == 2.0

    def test_duration_hours_fractional(self):
        """Test duration_hours with fractional hours."""
        now = time.time()
        window = SchedulingWindow(
            start_timestamp=now,
            end_timestamp=now + 5400,  # 1.5 hours
            expected_load=0.3,
            reason="test",
        )

        assert window.duration_hours == 1.5


class TestLoadForecaster:
    """Tests for LoadForecaster class."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "load_forecaster.db"

    @pytest.fixture
    def forecaster(self, temp_db_path):
        """Create a fresh forecaster instance."""
        return LoadForecaster(db_path=temp_db_path)

    def test_initialization(self, temp_db_path):
        """Test forecaster initializes correctly."""
        forecaster = LoadForecaster(db_path=temp_db_path)
        assert forecaster._db_path == temp_db_path

    def test_database_created(self, temp_db_path):
        """Test database file is created."""
        forecaster = LoadForecaster(db_path=temp_db_path)
        assert temp_db_path.exists()

    def test_get_connection_returns_connection(self, forecaster):
        """Test _get_connection returns a connection."""
        conn = forecaster._get_connection()
        assert conn is not None

    def test_get_connection_thread_local(self, forecaster):
        """Test _get_connection returns same connection in same thread."""
        conn1 = forecaster._get_connection()
        conn2 = forecaster._get_connection()
        assert conn1 is conn2

    def test_database_tables_exist(self, forecaster):
        """Test required database tables are created."""
        conn = forecaster._get_connection()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}

        assert "hourly_load" in tables
        assert "duration_accuracy" in tables
        assert "throughput_samples" in tables

    def test_get_cluster_size_defaults(self, forecaster):
        """Test _get_cluster_size returns defaults when cluster config unavailable."""
        with patch.object(forecaster, "_cluster_size_cache", None):
            # Patch the imports to simulate missing cluster config
            with patch("app.coordination.load_forecaster.get_cluster_nodes", None):
                with patch("app.coordination.load_forecaster.get_gpu_nodes", None):
                    total, gpu = forecaster._get_cluster_size()

        # Should return defaults
        assert total >= 1
        assert gpu >= 1

    def test_get_cluster_size_caching(self, forecaster):
        """Test _get_cluster_size uses cache."""
        # Prime the cache
        forecaster._cluster_size_cache = (40, 32, time.time())

        total, gpu = forecaster._get_cluster_size()

        assert total == 40
        assert gpu == 32

    def test_get_cluster_size_cache_expired(self, forecaster):
        """Test _get_cluster_size refreshes expired cache."""
        # Set expired cache
        old_time = time.time() - forecaster._cache_ttl - 100
        forecaster._cluster_size_cache = (40, 32, old_time)

        # This should refresh the cache (returns defaults with mocked imports)
        total, gpu = forecaster._get_cluster_size()

        # Cache should be refreshed with new timestamp
        assert forecaster._cluster_size_cache is not None
        assert forecaster._cluster_size_cache[2] > old_time

    def test_predict_load_returns_prediction(self, forecaster):
        """Test predict_load returns LoadPrediction."""
        prediction = forecaster.predict_load(hours_ahead=1.0)

        assert isinstance(prediction, LoadPrediction)
        assert prediction.hours_ahead == 1.0
        assert prediction.timestamp > 0

    def test_predict_load_different_horizons(self, forecaster):
        """Test predict_load with different time horizons."""
        pred_1h = forecaster.predict_load(hours_ahead=1.0)
        pred_6h = forecaster.predict_load(hours_ahead=6.0)
        pred_24h = forecaster.predict_load(hours_ahead=24.0)

        assert pred_1h.hours_ahead == 1.0
        assert pred_6h.hours_ahead == 6.0
        assert pred_24h.hours_ahead == 24.0

        # Timestamps should reflect the prediction time
        assert pred_6h.timestamp > pred_1h.timestamp
        assert pred_24h.timestamp > pred_6h.timestamp

    def test_record_load_snapshot(self, forecaster):
        """Test recording a load snapshot."""
        forecaster.record_load_snapshot(
            active_jobs=100,
            busy_hosts=20,
            total_hosts=30,
            gpu_hosts=25,
            gpu_utilization=0.75,
            cpu_utilization=0.60,
        )

        # Verify it was stored
        conn = forecaster._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM hourly_load")
        count = cursor.fetchone()[0]
        assert count >= 1

    def test_get_duration_accuracy_returns_list(self, forecaster):
        """Test get_duration_accuracy returns list."""
        # Without scheduler, returns empty list
        accuracies = forecaster.get_duration_accuracy()
        assert isinstance(accuracies, list)

    def test_get_duration_accuracy_for_task_type(self, forecaster):
        """Test get_duration_accuracy with specific task type."""
        # Without scheduler, returns empty list
        accuracies = forecaster.get_duration_accuracy("selfplay")
        assert isinstance(accuracies, list)

    def test_get_throughput_forecast_returns_list(self, forecaster):
        """Test get_throughput_forecast returns list."""
        # Without scheduler, returns empty list
        forecasts = forecaster.get_throughput_forecast()
        assert isinstance(forecasts, list)

    def test_get_throughput_forecast_for_task_type(self, forecaster):
        """Test get_throughput_forecast with specific task type."""
        # Without scheduler, returns empty list
        forecasts = forecaster.get_throughput_forecast("training")
        assert isinstance(forecasts, list)

    def test_get_optimal_scheduling_window(self, forecaster):
        """Test getting optimal scheduling window."""
        # Record some load data first
        forecaster.record_load_snapshot(
            active_jobs=50,
            busy_hosts=15,
            total_hosts=30,
            gpu_hosts=25,
            gpu_utilization=0.5,
            cpu_utilization=0.4,
        )

        window = forecaster.get_optimal_scheduling_window(
            task_type="training",
            min_duration_hours=2.0,
            max_hours_ahead=24.0,
        )

        # Should return a SchedulingWindow or None
        if window is not None:
            assert isinstance(window, SchedulingWindow)
            assert window.duration_hours >= 2.0

    def test_thread_safety(self, temp_db_path):
        """Test thread safety of forecaster operations."""
        import threading

        forecaster = LoadForecaster(db_path=temp_db_path)
        errors = []

        def record_snapshots():
            try:
                for _ in range(20):
                    forecaster.record_load_snapshot(
                        active_jobs=50,
                        busy_hosts=15,
                        total_hosts=30,
                        gpu_hosts=25,
                        gpu_utilization=0.5,
                        cpu_utilization=0.4,
                    )
            except Exception as e:
                errors.append(e)

        def predict_loads():
            try:
                for _ in range(20):
                    forecaster.predict_load(hours_ahead=1.0)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_snapshots),
            threading.Thread(target=predict_loads),
            threading.Thread(target=record_snapshots),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_module_imports(self):
        """Test module-level functions are importable."""
        from app.coordination.load_forecaster import (
            get_load_forecaster,
            predict_cluster_load,
            get_optimal_scheduling_window,
        )

        assert callable(get_load_forecaster)
        assert callable(predict_cluster_load)
        assert callable(get_optimal_scheduling_window)
