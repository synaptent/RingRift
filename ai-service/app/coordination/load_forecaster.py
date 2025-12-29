#!/usr/bin/env python3
"""Load Forecasting for Cluster-Wide Capacity Planning.

This module provides cluster-wide load prediction and capacity planning
based on historical job data from duration_scheduler.

Features:
- Cluster load prediction at future time points
- GPU utilization forecasting
- Job duration accuracy tracking (estimate vs actual)
- Throughput forecasting (jobs/hour by type)
- Peak detection and optimal scheduling windows
- Training data readiness prediction

December 2025: Created as part of Phase 2 training improvements.

Usage:
    from app.coordination.load_forecaster import (
        get_load_forecaster,
        predict_cluster_load,
        get_optimal_scheduling_window,
        predict_training_readiness,
    )

    # Predict load 6 hours from now
    load = predict_cluster_load(hours_ahead=6)

    # Find best time to schedule training
    window = get_optimal_scheduling_window(
        task_type="training",
        min_duration_hours=4,
    )

    # Predict when config will have enough training data
    ready_at = predict_training_readiness("hex8_2p")
"""

from __future__ import annotations

import json
import logging
import sqlite3
import statistics
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from app.coordination.protocols import HealthCheckResult

logger = logging.getLogger(__name__)

# Import duration scheduler for historical data
try:
    from app.coordination.duration_scheduler import (
        DurationScheduler,
        get_scheduler as get_duration_scheduler,
        DEFAULT_DURATIONS,
        INTENSIVE_TASK_TYPES,
    )
except ImportError:
    get_duration_scheduler = None
    DEFAULT_DURATIONS = {}
    INTENSIVE_TASK_TYPES = set()

# Import cluster config for node information
try:
    from app.config.cluster_config import get_cluster_nodes, get_gpu_nodes
except ImportError:
    get_cluster_nodes = None
    get_gpu_nodes = None


@dataclass
class LoadPrediction:
    """Prediction of cluster load at a future time point."""

    timestamp: float  # Unix timestamp of prediction
    hours_ahead: float  # Hours from now

    # Predicted load metrics
    active_jobs: int = 0  # Expected running jobs
    busy_hosts: int = 0  # Expected busy hosts
    gpu_utilization: float = 0.0  # 0-1 expected GPU util
    cpu_utilization: float = 0.0  # 0-1 expected CPU util

    # Capacity
    total_hosts: int = 0
    gpu_hosts: int = 0
    available_hosts: int = 0

    # Confidence (0-1, based on historical data availability)
    confidence: float = 0.0

    @property
    def load_percentage(self) -> float:
        """Load as percentage of total capacity."""
        if self.total_hosts == 0:
            return 0.0
        return (self.busy_hosts / self.total_hosts) * 100

    @property
    def is_peak(self) -> bool:
        """Whether this is a peak load period."""
        return self.load_percentage >= 80 or self.gpu_utilization >= 0.8


@dataclass
class DurationAccuracy:
    """Accuracy metrics for duration estimation."""

    task_type: str
    sample_count: int = 0

    # Errors (actual - estimated, in seconds)
    mean_error: float = 0.0  # Positive = underestimate
    median_error: float = 0.0
    std_error: float = 0.0

    # Percentage errors
    mean_pct_error: float = 0.0  # Average (actual/estimated - 1)

    # Accuracy score (0-1, where 1 = perfect)
    accuracy_score: float = 1.0

    @property
    def needs_recalibration(self) -> bool:
        """Whether default duration needs adjustment."""
        return abs(self.mean_pct_error) > 0.3 and self.sample_count >= 10


@dataclass
class ThroughputForecast:
    """Forecast of job completion rate."""

    task_type: str
    window_hours: float = 24.0

    # Historical rates (jobs per hour)
    current_rate: float = 0.0
    predicted_rate: float = 0.0

    # Trend (-1 to 1, negative = declining)
    trend: float = 0.0

    # Predictions
    expected_completions: int = 0  # In next window
    expected_failures: int = 0


@dataclass
class SchedulingWindow:
    """Optimal time window for scheduling."""

    start_timestamp: float
    end_timestamp: float
    expected_load: float  # 0-1
    reason: str

    @property
    def start_time(self) -> datetime:
        return datetime.fromtimestamp(self.start_timestamp)

    @property
    def end_time(self) -> datetime:
        return datetime.fromtimestamp(self.end_timestamp)

    @property
    def duration_hours(self) -> float:
        return (self.end_timestamp - self.start_timestamp) / 3600


class LoadForecaster:
    """Cluster-wide load forecasting based on historical data."""

    def __init__(
        self,
        db_path: Path | None = None,
        duration_scheduler: DurationScheduler | None = None,
    ):
        """Initialize load forecaster.

        Args:
            db_path: Path for forecast-specific data (default: uses duration scheduler DB)
            duration_scheduler: Optional custom scheduler (default: global singleton)
        """
        self._scheduler = duration_scheduler or (
            get_duration_scheduler() if get_duration_scheduler else None
        )
        self._db_path = db_path or Path("/tmp/ringrift_coordination/load_forecaster.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

        # Cache for expensive computations
        self._cluster_size_cache: tuple[int, int, float] | None = None  # (total, gpu, timestamp)
        self._cache_ttl = 300  # 5 minutes

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self._db_path), timeout=30.0)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=10000")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema for forecaster-specific data."""
        conn = self._get_connection()
        conn.executescript('''
            -- Hourly load snapshots (for pattern learning)
            CREATE TABLE IF NOT EXISTS hourly_load (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                hour_timestamp REAL NOT NULL,  -- Hour start (rounded)
                active_jobs INTEGER NOT NULL DEFAULT 0,
                busy_hosts INTEGER NOT NULL DEFAULT 0,
                total_hosts INTEGER NOT NULL DEFAULT 0,
                gpu_hosts INTEGER NOT NULL DEFAULT 0,
                gpu_utilization REAL NOT NULL DEFAULT 0.0,
                cpu_utilization REAL NOT NULL DEFAULT 0.0,
                jobs_completed INTEGER NOT NULL DEFAULT 0,
                jobs_failed INTEGER NOT NULL DEFAULT 0,
                day_of_week INTEGER NOT NULL DEFAULT 0,  -- 0=Monday
                hour_of_day INTEGER NOT NULL DEFAULT 0
            );

            -- Duration accuracy tracking
            CREATE TABLE IF NOT EXISTS duration_accuracy (
                task_type TEXT PRIMARY KEY,
                sample_count INTEGER NOT NULL DEFAULT 0,
                total_estimated REAL NOT NULL DEFAULT 0.0,
                total_actual REAL NOT NULL DEFAULT 0.0,
                sum_squared_error REAL NOT NULL DEFAULT 0.0,
                last_updated REAL NOT NULL
            );

            -- Throughput samples (rolling window)
            CREATE TABLE IF NOT EXISTS throughput_samples (
                sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                jobs_completed INTEGER NOT NULL DEFAULT 0,
                jobs_failed INTEGER NOT NULL DEFAULT 0,
                window_seconds INTEGER NOT NULL DEFAULT 3600  -- 1 hour default
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_hourly_timestamp ON hourly_load(hour_timestamp);
            CREATE INDEX IF NOT EXISTS idx_hourly_dow ON hourly_load(day_of_week, hour_of_day);
            CREATE INDEX IF NOT EXISTS idx_throughput_type ON throughput_samples(task_type, timestamp);
        ''')
        conn.commit()

    def _get_cluster_size(self) -> tuple[int, int]:
        """Get cluster size (total hosts, GPU hosts) with caching."""
        now = time.time()
        if self._cluster_size_cache:
            total, gpu, cached_at = self._cluster_size_cache
            if now - cached_at < self._cache_ttl:
                return total, gpu

        # Try to get from cluster config
        total_hosts = 0
        gpu_hosts = 0

        if get_cluster_nodes:
            try:
                nodes = get_cluster_nodes()
                total_hosts = len([n for n in nodes.values() if n.status == "ready"])
            except (ValueError, KeyError, TypeError, AttributeError, OSError):
                pass  # Config loading or attribute access issues

        if get_gpu_nodes:
            try:
                gpu_nodes = get_gpu_nodes()
                gpu_hosts = len(gpu_nodes)
            except (ValueError, KeyError, TypeError, AttributeError, OSError):
                pass  # Config loading or attribute access issues

        # Fallback to reasonable defaults
        if total_hosts == 0:
            total_hosts = 36  # Default cluster size
        if gpu_hosts == 0:
            gpu_hosts = int(total_hosts * 0.8)  # Assume 80% have GPUs

        self._cluster_size_cache = (total_hosts, gpu_hosts, now)
        return total_hosts, gpu_hosts

    def predict_load(self, hours_ahead: float = 1.0) -> LoadPrediction:
        """Predict cluster load at a future time point.

        Args:
            hours_ahead: Hours from now to predict

        Returns:
            LoadPrediction with expected load metrics
        """
        now = time.time()
        target_time = now + (hours_ahead * 3600)
        target_dt = datetime.fromtimestamp(target_time)
        target_dow = target_dt.weekday()
        target_hour = target_dt.hour

        total_hosts, gpu_hosts = self._get_cluster_size()
        conn = self._get_connection()

        # Get historical data for same day/hour (last 4 weeks)
        cursor = conn.execute('''
            SELECT
                AVG(active_jobs) as avg_jobs,
                AVG(busy_hosts) as avg_busy,
                AVG(gpu_utilization) as avg_gpu,
                AVG(cpu_utilization) as avg_cpu,
                COUNT(*) as sample_count
            FROM hourly_load
            WHERE day_of_week = ? AND hour_of_day = ?
            AND hour_timestamp > ?
        ''', (target_dow, target_hour, now - 28 * 86400))

        row = cursor.fetchone()

        if row and row["sample_count"] >= 2:
            # Use historical pattern
            confidence = min(0.9, row["sample_count"] / 10)
            return LoadPrediction(
                timestamp=target_time,
                hours_ahead=hours_ahead,
                active_jobs=int(row["avg_jobs"]),
                busy_hosts=int(row["avg_busy"]),
                gpu_utilization=row["avg_gpu"] or 0.0,
                cpu_utilization=row["avg_cpu"] or 0.0,
                total_hosts=total_hosts,
                gpu_hosts=gpu_hosts,
                available_hosts=total_hosts - int(row["avg_busy"]),
                confidence=confidence,
            )

        # Fallback: use current running jobs from duration scheduler
        if self._scheduler:
            try:
                sched_conn = self._scheduler._get_connection()
                cursor = sched_conn.execute('''
                    SELECT COUNT(*) as cnt FROM running_tasks
                    WHERE expected_end > ?
                ''', (target_time,))
                running = cursor.fetchone()["cnt"]

                # Estimate based on running tasks that will still be active
                return LoadPrediction(
                    timestamp=target_time,
                    hours_ahead=hours_ahead,
                    active_jobs=running,
                    busy_hosts=min(running, total_hosts),
                    gpu_utilization=running / gpu_hosts if gpu_hosts > 0 else 0.0,
                    cpu_utilization=running / total_hosts if total_hosts > 0 else 0.0,
                    total_hosts=total_hosts,
                    gpu_hosts=gpu_hosts,
                    available_hosts=max(0, total_hosts - running),
                    confidence=0.5,  # Medium confidence when using current state
                )
            except Exception as e:
                logger.debug(f"Duration scheduler query failed: {e}")

        # Very low confidence fallback
        return LoadPrediction(
            timestamp=target_time,
            hours_ahead=hours_ahead,
            total_hosts=total_hosts,
            gpu_hosts=gpu_hosts,
            available_hosts=total_hosts,
            confidence=0.1,
        )

    def record_load_snapshot(
        self,
        active_jobs: int,
        busy_hosts: int,
        total_hosts: int | None = None,
        gpu_hosts: int | None = None,
        gpu_utilization: float = 0.0,
        cpu_utilization: float = 0.0,
        jobs_completed: int = 0,
        jobs_failed: int = 0,
    ) -> None:
        """Record a load snapshot for pattern learning.

        Call this periodically (e.g., hourly) to build historical patterns.
        """
        now = time.time()
        now_dt = datetime.fromtimestamp(now)

        # Round to hour start
        hour_start = now - (now % 3600)

        if total_hosts is None or gpu_hosts is None:
            total_hosts, gpu_hosts = self._get_cluster_size()

        conn = self._get_connection()
        conn.execute('''
            INSERT INTO hourly_load (
                hour_timestamp, active_jobs, busy_hosts, total_hosts, gpu_hosts,
                gpu_utilization, cpu_utilization, jobs_completed, jobs_failed,
                day_of_week, hour_of_day
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            hour_start, active_jobs, busy_hosts, total_hosts, gpu_hosts,
            gpu_utilization, cpu_utilization, jobs_completed, jobs_failed,
            now_dt.weekday(), now_dt.hour,
        ))
        conn.commit()

    def get_duration_accuracy(self, task_type: str | None = None) -> list[DurationAccuracy]:
        """Get duration estimation accuracy metrics.

        Args:
            task_type: Specific type or None for all

        Returns:
            List of DurationAccuracy records
        """
        if not self._scheduler:
            return []

        results = []
        conn = self._scheduler._get_connection()

        # Get recent completions
        query = '''
            SELECT task_type, duration_seconds, metadata
            FROM duration_history
            WHERE success = 1 AND started_at > ?
        '''
        params: list[Any] = [time.time() - 30 * 86400]  # Last 30 days

        if task_type:
            query += ' AND task_type = ?'
            params.append(task_type)

        cursor = conn.execute(query, params)

        # Group by task type
        by_type: dict[str, list[tuple[float, float]]] = {}
        for row in cursor.fetchall():
            tt = row["task_type"]
            actual = row["duration_seconds"]

            # Get estimated duration (from defaults)
            estimated = DEFAULT_DURATIONS.get(tt, 3600)

            if tt not in by_type:
                by_type[tt] = []
            by_type[tt].append((estimated, actual))

        # Compute accuracy metrics
        for tt, samples in by_type.items():
            if not samples:
                continue

            errors = [actual - estimated for estimated, actual in samples]
            pct_errors = [
                (actual / estimated - 1) if estimated > 0 else 0
                for estimated, actual in samples
            ]

            mean_error = statistics.mean(errors)
            std_error = statistics.stdev(errors) if len(errors) > 1 else 0.0
            mean_pct = statistics.mean(pct_errors)

            # Accuracy score: 1 - mean absolute percentage error (capped at 0)
            accuracy = max(0.0, 1.0 - abs(mean_pct))

            results.append(DurationAccuracy(
                task_type=tt,
                sample_count=len(samples),
                mean_error=mean_error,
                median_error=statistics.median(errors) if errors else 0.0,
                std_error=std_error,
                mean_pct_error=mean_pct,
                accuracy_score=accuracy,
            ))

        return results

    def get_throughput_forecast(
        self,
        task_type: str | None = None,
        window_hours: float = 24.0,
    ) -> list[ThroughputForecast]:
        """Forecast job completion rates.

        Args:
            task_type: Specific type or None for all
            window_hours: Forecast window

        Returns:
            List of ThroughputForecast records
        """
        if not self._scheduler:
            return []

        results = []
        conn = self._scheduler._get_connection()
        now = time.time()

        # Get completions by hour for trend analysis
        query = '''
            SELECT
                task_type,
                CAST((started_at / 3600) AS INTEGER) * 3600 as hour_bucket,
                COUNT(*) as completions,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes
            FROM duration_history
            WHERE started_at > ?
        '''
        params: list[Any] = [now - 7 * 86400]  # Last 7 days

        if task_type:
            query += ' AND task_type = ?'
            params.append(task_type)

        query += ' GROUP BY task_type, hour_bucket ORDER BY hour_bucket'
        cursor = conn.execute(query, params)

        # Group by task type
        by_type: dict[str, list[tuple[float, int, int]]] = {}
        for row in cursor.fetchall():
            tt = row["task_type"]
            if tt not in by_type:
                by_type[tt] = []
            by_type[tt].append((row["hour_bucket"], row["completions"], row["successes"]))

        # Compute forecasts
        for tt, hourly in by_type.items():
            if not hourly:
                continue

            # Recent rate (last 24 hours)
            recent = [c for ts, c, s in hourly if ts > now - 86400]
            current_rate = sum(recent) / 24 if recent else 0

            # Calculate trend (last 3 days vs previous 4 days)
            recent_3d = [c for ts, c, s in hourly if ts > now - 3 * 86400]
            older_4d = [c for ts, c, s in hourly if now - 7 * 86400 < ts <= now - 3 * 86400]

            recent_avg = sum(recent_3d) / len(recent_3d) if recent_3d else 0
            older_avg = sum(older_4d) / len(older_4d) if older_4d else recent_avg

            if older_avg > 0:
                trend = (recent_avg - older_avg) / older_avg
                trend = max(-1.0, min(1.0, trend))  # Clamp to [-1, 1]
            else:
                trend = 0.0

            # Predict future rate (apply trend)
            predicted_rate = current_rate * (1 + trend * 0.5)  # Dampen trend impact

            # Total failures in recent window
            failures = sum(c - s for ts, c, s in hourly if ts > now - 86400)

            results.append(ThroughputForecast(
                task_type=tt,
                window_hours=window_hours,
                current_rate=current_rate,
                predicted_rate=predicted_rate,
                trend=trend,
                expected_completions=int(predicted_rate * window_hours),
                expected_failures=int(failures * (window_hours / 24)),
            ))

        return results

    def get_optimal_scheduling_window(
        self,
        task_type: str,
        min_duration_hours: float = 1.0,
        max_hours_ahead: float = 24.0,
        prefer_off_peak: bool = True,
    ) -> SchedulingWindow | None:
        """Find optimal time window for scheduling a task.

        Args:
            task_type: Type of task to schedule
            min_duration_hours: Minimum required window duration
            max_hours_ahead: Maximum hours to look ahead
            prefer_off_peak: Prefer low-load windows

        Returns:
            SchedulingWindow or None if no suitable window found
        """
        now = time.time()
        best_window: SchedulingWindow | None = None
        best_score = -1.0

        # Check each hour in the window
        for hours_offset in range(int(max_hours_ahead)):
            prediction = self.predict_load(hours_ahead=hours_offset)

            # Skip if already at peak
            if prediction.is_peak and prefer_off_peak:
                continue

            # Calculate window score (lower load = higher score)
            load_score = 1.0 - (prediction.load_percentage / 100)

            # Prefer higher confidence predictions
            conf_score = prediction.confidence

            # Combined score
            score = load_score * 0.7 + conf_score * 0.3

            if score > best_score:
                best_score = score
                best_window = SchedulingWindow(
                    start_timestamp=now + (hours_offset * 3600),
                    end_timestamp=now + ((hours_offset + min_duration_hours) * 3600),
                    expected_load=prediction.load_percentage / 100,
                    reason=f"Low load ({prediction.load_percentage:.0f}%), "
                           f"confidence={prediction.confidence:.1%}",
                )

        return best_window

    def predict_training_readiness(
        self,
        config_key: str,
        min_games: int = 1000,
    ) -> tuple[float, float] | None:
        """Predict when a config will have enough training data.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            min_games: Minimum games needed for training

        Returns:
            Tuple of (ready_timestamp, games_per_hour) or None if cannot predict
        """
        # This would integrate with game database to track generation rate
        # For now, use throughput forecast for selfplay
        forecasts = self.get_throughput_forecast(task_type="selfplay")

        if not forecasts:
            return None

        selfplay = forecasts[0]
        if selfplay.predicted_rate <= 0:
            return None

        # Estimate games per hour (assume average 50 games per selfplay job)
        games_per_hour = selfplay.predicted_rate * 50

        # Would need current game count from DB
        # For now, return rate only
        return (time.time(), games_per_hour)

    def health_check(self) -> HealthCheckResult:
        """Health check for monitoring integration.

        Returns:
            HealthCheckResult with current health status and metrics.
        """
        try:
            conn = self._get_connection()

            # Check hourly load records
            cursor = conn.execute(
                "SELECT COUNT(*) as cnt FROM hourly_load WHERE hour_timestamp > ?",
                (time.time() - 7 * 86400,)
            )
            snapshots = cursor.fetchone()["cnt"]

            # Get accuracy metrics count
            accuracy = self.get_duration_accuracy()

            # Determine health status
            is_healthy = snapshots > 0 or self._scheduler is not None
            message = "OK" if is_healthy else "No recent data and no scheduler"

            return HealthCheckResult(
                healthy=is_healthy,
                message=message,
                details={
                    "snapshots_7d": snapshots,
                    "accuracy_tracked_types": len(accuracy),
                    "has_scheduler": self._scheduler is not None,
                },
            )
        except (sqlite3.Error, OSError) as e:
            return HealthCheckResult(
                healthy=False,
                message=f"Database error: {e}",
                details={"error": str(e)},
            )

    def cleanup_old_data(self, max_age_days: int = 90) -> int:
        """Clean up old forecasting data."""
        conn = self._get_connection()
        cutoff = time.time() - (max_age_days * 86400)

        cursor = conn.execute(
            "DELETE FROM hourly_load WHERE hour_timestamp < ?",
            (cutoff,)
        )
        deleted = cursor.rowcount

        conn.execute(
            "DELETE FROM throughput_samples WHERE timestamp < ?",
            (cutoff,)
        )

        conn.commit()
        return deleted

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# =============================================================================
# Singleton and convenience functions
# =============================================================================

_forecaster: LoadForecaster | None = None
_forecaster_lock = threading.RLock()


def get_load_forecaster() -> LoadForecaster:
    """Get global load forecaster singleton."""
    global _forecaster
    with _forecaster_lock:
        if _forecaster is None:
            _forecaster = LoadForecaster()
        return _forecaster


def reset_load_forecaster() -> None:
    """Reset the global forecaster (for testing)."""
    global _forecaster
    with _forecaster_lock:
        if _forecaster is not None:
            _forecaster.close()
        _forecaster = None


def predict_cluster_load(hours_ahead: float = 1.0) -> LoadPrediction:
    """Predict cluster load at a future time point."""
    return get_load_forecaster().predict_load(hours_ahead)


def get_optimal_scheduling_window(
    task_type: str,
    min_duration_hours: float = 1.0,
) -> SchedulingWindow | None:
    """Find optimal scheduling window for a task."""
    return get_load_forecaster().get_optimal_scheduling_window(
        task_type, min_duration_hours
    )


def record_hourly_load(
    active_jobs: int,
    busy_hosts: int,
    **kwargs: Any,
) -> None:
    """Record current load for pattern learning."""
    get_load_forecaster().record_load_snapshot(active_jobs, busy_hosts, **kwargs)


def predict_training_readiness(config_key: str) -> tuple[float, float] | None:
    """Predict when training data will be ready."""
    return get_load_forecaster().predict_training_readiness(config_key)


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Classes
    "LoadForecaster",
    "LoadPrediction",
    "DurationAccuracy",
    "ThroughputForecast",
    "SchedulingWindow",
    # Singleton
    "get_load_forecaster",
    "reset_load_forecaster",
    # Convenience functions
    "predict_cluster_load",
    "get_optimal_scheduling_window",
    "record_hourly_load",
    "predict_training_readiness",
]
