"""Training Dashboard & Monitoring System for RingRift AI.

Provides real-time visualization and monitoring of:
- Training metrics (loss, accuracy, learning rate)
- Elo progression across models
- Cluster utilization (CPU, GPU, memory)
- Self-play statistics (games/hour, win rates)
- Model performance benchmarks

Features:
- Web-based dashboard with auto-refresh
- Historical data storage and querying
- Alerting for anomalies
- Export to common formats

Usage:
    # Start dashboard server
    python -m app.monitoring.training_dashboard --port 8080

    # Or use programmatically
    from app.monitoring.training_dashboard import DashboardServer, MetricsCollector

    collector = MetricsCollector(db_path="data/metrics.db")
    collector.record_training_step(epoch=1, loss=0.5, accuracy=0.8)

    server = DashboardServer(collector)
    server.run(port=8080)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.utils.datetime_utils import iso_now, time_ago, to_iso, utc_now
from app.utils.paths import AI_SERVICE_ROOT
from app.utils.optional_imports import (
    PROMETHEUS_AVAILABLE as HAS_PROMETHEUS,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

import numpy as np

logger = logging.getLogger(__name__)

# Prometheus metrics (initialized if prometheus_client is available)
if HAS_PROMETHEUS:
    # Training metrics
    PROM_TRAINING_LOSS = Gauge(
        'ringrift_dashboard_training_loss',
        'Current training loss',
        ['model_id']
    )
    PROM_TRAINING_ACCURACY = Gauge(
        'ringrift_dashboard_training_accuracy',
        'Current training accuracy',
        ['model_id']
    )
    PROM_TRAINING_EPOCH = Gauge(
        'ringrift_dashboard_training_epoch',
        'Current training epoch',
        ['model_id']
    )
    PROM_SAMPLES_PER_SECOND = Gauge(
        'ringrift_dashboard_samples_per_second',
        'Training samples per second',
        ['model_id']
    )

    # Elo metrics
    PROM_MODEL_ELO = Gauge(
        'ringrift_dashboard_model_elo',
        'Model Elo rating',
        ['model_id', 'board_type', 'num_players']
    )
    PROM_MODEL_WIN_RATE = Gauge(
        'ringrift_dashboard_model_win_rate',
        'Model win rate',
        ['model_id', 'board_type', 'num_players']
    )
    PROM_MODEL_GAMES_PLAYED = Gauge(
        'ringrift_dashboard_model_games_played',
        'Games played by model',
        ['model_id', 'board_type', 'num_players']
    )

    # Cluster metrics
    PROM_CLUSTER_CPU = Gauge(
        'ringrift_dashboard_cluster_cpu_percent',
        'Cluster CPU usage percent',
        ['host']
    )
    PROM_CLUSTER_GPU = Gauge(
        'ringrift_dashboard_cluster_gpu_percent',
        'Cluster GPU usage percent',
        ['host']
    )
    PROM_CLUSTER_MEMORY = Gauge(
        'ringrift_dashboard_cluster_memory_percent',
        'Cluster memory usage percent',
        ['host']
    )
    PROM_CLUSTER_ACTIVE_JOBS = Gauge(
        'ringrift_dashboard_cluster_active_jobs',
        'Active jobs on host',
        ['host']
    )
    PROM_SELFPLAY_GAMES_PER_HOUR = Gauge(
        'ringrift_dashboard_selfplay_games_per_hour',
        'Self-play games per hour',
        ['host']
    )


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TrainingMetrics:
    """Metrics from a training step."""
    timestamp: str
    epoch: int
    step: int
    loss: float
    policy_loss: float = 0.0
    value_loss: float = 0.0
    accuracy: float = 0.0
    learning_rate: float = 0.0
    batch_size: int = 0
    samples_per_second: float = 0.0
    gpu_memory_mb: float = 0.0
    model_id: str = ""


@dataclass
class EloSnapshot:
    """Elo rating snapshot for a model."""
    timestamp: str
    model_id: str
    elo: float
    games_played: int
    win_rate: float
    board_type: str
    num_players: int
    confidence_interval: Tuple[float, float] = (0.0, 0.0)


@dataclass
class ClusterMetrics:
    """Metrics from the compute cluster."""
    timestamp: str
    host_name: str
    cpu_percent: float
    memory_percent: float
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    disk_percent: float = 0.0
    active_jobs: int = 0
    selfplay_games_per_hour: float = 0.0


@dataclass
class SelfPlayMetrics:
    """Metrics from self-play generation."""
    timestamp: str
    board_type: str
    num_players: int
    games_completed: int
    games_per_hour: float
    avg_game_length: float
    draw_rate: float
    first_player_win_rate: float


@dataclass
class Alert:
    """An alert/notification."""
    timestamp: str
    severity: str  # "info", "warning", "error", "critical"
    category: str  # "training", "cluster", "model", "selfplay"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False


# =============================================================================
# Metrics Database
# =============================================================================

class MetricsDatabase:
    """SQLite database for storing metrics."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Training metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                epoch INTEGER,
                step INTEGER,
                loss REAL,
                policy_loss REAL,
                value_loss REAL,
                accuracy REAL,
                learning_rate REAL,
                batch_size INTEGER,
                samples_per_second REAL,
                gpu_memory_mb REAL,
                model_id TEXT
            )
        """)

        # Elo snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS elo_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_id TEXT NOT NULL,
                elo REAL,
                games_played INTEGER,
                win_rate REAL,
                board_type TEXT,
                num_players INTEGER,
                ci_low REAL,
                ci_high REAL
            )
        """)

        # Cluster metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cluster_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                host_name TEXT NOT NULL,
                cpu_percent REAL,
                memory_percent REAL,
                gpu_percent REAL,
                gpu_memory_percent REAL,
                disk_percent REAL,
                active_jobs INTEGER,
                selfplay_games_per_hour REAL
            )
        """)

        # Self-play metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS selfplay_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                board_type TEXT,
                num_players INTEGER,
                games_completed INTEGER,
                games_per_hour REAL,
                avg_game_length REAL,
                draw_rate REAL,
                first_player_win_rate REAL
            )
        """)

        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                severity TEXT,
                category TEXT,
                message TEXT,
                details TEXT,
                acknowledged INTEGER DEFAULT 0
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_timestamp ON training_metrics(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_elo_model ON elo_snapshots(model_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cluster_host ON cluster_metrics(host_name)")

        conn.commit()
        conn.close()

    def insert_training_metrics(self, metrics: TrainingMetrics):
        """Insert training metrics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO training_metrics
            (timestamp, epoch, step, loss, policy_loss, value_loss, accuracy,
             learning_rate, batch_size, samples_per_second, gpu_memory_mb, model_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.timestamp, metrics.epoch, metrics.step, metrics.loss,
            metrics.policy_loss, metrics.value_loss, metrics.accuracy,
            metrics.learning_rate, metrics.batch_size, metrics.samples_per_second,
            metrics.gpu_memory_mb, metrics.model_id
        ))

        conn.commit()
        conn.close()

    def insert_elo_snapshot(self, snapshot: EloSnapshot):
        """Insert Elo snapshot."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO elo_snapshots
            (timestamp, model_id, elo, games_played, win_rate, board_type,
             num_players, ci_low, ci_high)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snapshot.timestamp, snapshot.model_id, snapshot.elo,
            snapshot.games_played, snapshot.win_rate, snapshot.board_type,
            snapshot.num_players, snapshot.confidence_interval[0],
            snapshot.confidence_interval[1]
        ))

        conn.commit()
        conn.close()

    def insert_cluster_metrics(self, metrics: ClusterMetrics):
        """Insert cluster metrics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO cluster_metrics
            (timestamp, host_name, cpu_percent, memory_percent, gpu_percent,
             gpu_memory_percent, disk_percent, active_jobs, selfplay_games_per_hour)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.timestamp, metrics.host_name, metrics.cpu_percent,
            metrics.memory_percent, metrics.gpu_percent, metrics.gpu_memory_percent,
            metrics.disk_percent, metrics.active_jobs, metrics.selfplay_games_per_hour
        ))

        conn.commit()
        conn.close()

    def insert_selfplay_metrics(self, metrics: SelfPlayMetrics):
        """Insert self-play metrics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO selfplay_metrics
            (timestamp, board_type, num_players, games_completed, games_per_hour,
             avg_game_length, draw_rate, first_player_win_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.timestamp, metrics.board_type, metrics.num_players,
            metrics.games_completed, metrics.games_per_hour, metrics.avg_game_length,
            metrics.draw_rate, metrics.first_player_win_rate
        ))

        conn.commit()
        conn.close()

    def insert_alert(self, alert: Alert):
        """Insert alert."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO alerts
            (timestamp, severity, category, message, details, acknowledged)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            alert.timestamp, alert.severity, alert.category,
            alert.message, json.dumps(alert.details), 0
        ))

        conn.commit()
        conn.close()

    def get_training_metrics(
        self,
        since: Optional[datetime] = None,
        model_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[TrainingMetrics]:
        """Query training metrics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        query = "SELECT * FROM training_metrics WHERE 1=1"
        params = []

        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [TrainingMetrics(
            timestamp=row[1], epoch=row[2], step=row[3], loss=row[4],
            policy_loss=row[5], value_loss=row[6], accuracy=row[7],
            learning_rate=row[8], batch_size=row[9], samples_per_second=row[10],
            gpu_memory_mb=row[11], model_id=row[12]
        ) for row in rows]

    def get_elo_history(
        self,
        model_id: Optional[str] = None,
        board_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[EloSnapshot]:
        """Query Elo history."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        query = "SELECT * FROM elo_snapshots WHERE 1=1"
        params = []

        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)

        if board_type:
            query += " AND board_type = ?"
            params.append(board_type)

        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [EloSnapshot(
            timestamp=row[1], model_id=row[2], elo=row[3], games_played=row[4],
            win_rate=row[5], board_type=row[6], num_players=row[7],
            confidence_interval=(row[8], row[9])
        ) for row in rows]

    def get_cluster_metrics(
        self,
        host_name: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[ClusterMetrics]:
        """Query cluster metrics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        query = "SELECT * FROM cluster_metrics WHERE 1=1"
        params = []

        if host_name:
            query += " AND host_name = ?"
            params.append(host_name)

        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [ClusterMetrics(
            timestamp=row[1], host_name=row[2], cpu_percent=row[3],
            memory_percent=row[4], gpu_percent=row[5], gpu_memory_percent=row[6],
            disk_percent=row[7], active_jobs=row[8], selfplay_games_per_hour=row[9]
        ) for row in rows]

    def get_alerts(
        self,
        severity: Optional[str] = None,
        category: Optional[str] = None,
        unacknowledged_only: bool = False,
        limit: int = 100,
    ) -> List[Alert]:
        """Query alerts."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        query = "SELECT * FROM alerts WHERE 1=1"
        params = []

        if severity:
            query += " AND severity = ?"
            params.append(severity)

        if category:
            query += " AND category = ?"
            params.append(category)

        if unacknowledged_only:
            query += " AND acknowledged = 0"

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [Alert(
            timestamp=row[1], severity=row[2], category=row[3],
            message=row[4], details=json.loads(row[5] or "{}"),
            acknowledged=bool(row[6])
        ) for row in rows]


# =============================================================================
# Metrics Collector
# =============================================================================

class MetricsCollector:
    """Collects and stores metrics from various sources."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        alert_thresholds: Optional[Dict[str, Any]] = None,
    ):
        """Initialize collector.

        Args:
            db_path: Path to metrics database
            alert_thresholds: Custom thresholds for alerting
        """
        if db_path is None:
            db_path = AI_SERVICE_ROOT / "data" / "metrics" / "training_metrics.db"

        self.db = MetricsDatabase(db_path)

        # Default alert thresholds - aligned with resource_guard limits
        # Import limits from resource_guard for consistency
        try:
            from app.utils.resource_guard import LIMITS as RESOURCE_LIMITS
            gpu_limit = RESOURCE_LIMITS.GPU_MAX_PERCENT
            disk_limit = RESOURCE_LIMITS.DISK_MAX_PERCENT
            cpu_limit = RESOURCE_LIMITS.CPU_MAX_PERCENT
            memory_limit = RESOURCE_LIMITS.MEMORY_MAX_PERCENT
        except ImportError:
            gpu_limit, disk_limit, cpu_limit, memory_limit = 80.0, 70.0, 80.0, 80.0

        self.alert_thresholds = alert_thresholds or {
            "loss_spike": 2.0,  # Alert if loss > 2x moving average
            "gpu_memory_high": gpu_limit,  # Alert at resource_guard limit (80%)
            "disk_high": disk_limit,  # Alert at resource_guard limit (70%)
            "cpu_high": cpu_limit,  # Alert at resource_guard limit (80%)
            "memory_high": memory_limit,  # Alert at resource_guard limit (80%)
            "elo_drop": 50.0,  # Alert if Elo drops > 50
            "games_per_hour_low": 10.0,  # Alert if self-play < 10 games/hour
        }

        # Moving averages for anomaly detection
        self._loss_history: List[float] = []
        self._elo_history: Dict[str, float] = {}

    def record_training_step(
        self,
        epoch: int,
        step: int,
        loss: float,
        policy_loss: float = 0.0,
        value_loss: float = 0.0,
        accuracy: float = 0.0,
        learning_rate: float = 0.0,
        batch_size: int = 0,
        samples_per_second: float = 0.0,
        gpu_memory_mb: float = 0.0,
        model_id: str = "",
    ):
        """Record a training step."""
        metrics = TrainingMetrics(
            timestamp=iso_now(),
            epoch=epoch,
            step=step,
            loss=loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            accuracy=accuracy,
            learning_rate=learning_rate,
            batch_size=batch_size,
            samples_per_second=samples_per_second,
            gpu_memory_mb=gpu_memory_mb,
            model_id=model_id,
        )
        self.db.insert_training_metrics(metrics)

        # Export to Prometheus if available
        if HAS_PROMETHEUS:
            label = model_id or "default"
            PROM_TRAINING_LOSS.labels(model_id=label).set(loss)
            PROM_TRAINING_ACCURACY.labels(model_id=label).set(accuracy)
            PROM_TRAINING_EPOCH.labels(model_id=label).set(epoch)
            PROM_SAMPLES_PER_SECOND.labels(model_id=label).set(samples_per_second)

        # Check for anomalies
        self._check_loss_anomaly(loss)

    def record_elo(
        self,
        model_id: str,
        elo: float,
        games_played: int,
        win_rate: float,
        board_type: str,
        num_players: int,
        confidence_interval: Tuple[float, float] = (0.0, 0.0),
    ):
        """Record Elo snapshot."""
        snapshot = EloSnapshot(
            timestamp=iso_now(),
            model_id=model_id,
            elo=elo,
            games_played=games_played,
            win_rate=win_rate,
            board_type=board_type,
            num_players=num_players,
            confidence_interval=confidence_interval,
        )
        self.db.insert_elo_snapshot(snapshot)

        # Export to Prometheus if available
        if HAS_PROMETHEUS:
            PROM_MODEL_ELO.labels(
                model_id=model_id,
                board_type=board_type,
                num_players=str(num_players)
            ).set(elo)
            PROM_MODEL_WIN_RATE.labels(
                model_id=model_id,
                board_type=board_type,
                num_players=str(num_players)
            ).set(win_rate)
            PROM_MODEL_GAMES_PLAYED.labels(
                model_id=model_id,
                board_type=board_type,
                num_players=str(num_players)
            ).set(games_played)

        # Check for Elo drops
        self._check_elo_drop(model_id, elo)

    def record_cluster_status(
        self,
        host_name: str,
        cpu_percent: float,
        memory_percent: float,
        gpu_percent: float = 0.0,
        gpu_memory_percent: float = 0.0,
        disk_percent: float = 0.0,
        active_jobs: int = 0,
        selfplay_games_per_hour: float = 0.0,
    ):
        """Record cluster metrics."""
        metrics = ClusterMetrics(
            timestamp=iso_now(),
            host_name=host_name,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            gpu_percent=gpu_percent,
            gpu_memory_percent=gpu_memory_percent,
            disk_percent=disk_percent,
            active_jobs=active_jobs,
            selfplay_games_per_hour=selfplay_games_per_hour,
        )
        self.db.insert_cluster_metrics(metrics)

        # Export to Prometheus if available
        if HAS_PROMETHEUS:
            PROM_CLUSTER_CPU.labels(host=host_name).set(cpu_percent)
            PROM_CLUSTER_GPU.labels(host=host_name).set(gpu_percent)
            PROM_CLUSTER_MEMORY.labels(host=host_name).set(memory_percent)
            PROM_CLUSTER_ACTIVE_JOBS.labels(host=host_name).set(active_jobs)
            PROM_SELFPLAY_GAMES_PER_HOUR.labels(host=host_name).set(selfplay_games_per_hour)

        # Check for resource issues against resource_guard 80% limits
        self._check_resource_issues(host_name, cpu_percent, gpu_memory_percent, disk_percent, memory_percent)

    def record_selfplay_stats(
        self,
        board_type: str,
        num_players: int,
        games_completed: int,
        games_per_hour: float,
        avg_game_length: float,
        draw_rate: float,
        first_player_win_rate: float,
    ):
        """Record self-play statistics."""
        metrics = SelfPlayMetrics(
            timestamp=iso_now(),
            board_type=board_type,
            num_players=num_players,
            games_completed=games_completed,
            games_per_hour=games_per_hour,
            avg_game_length=avg_game_length,
            draw_rate=draw_rate,
            first_player_win_rate=first_player_win_rate,
        )
        self.db.insert_selfplay_metrics(metrics)

        # Check for low throughput
        if games_per_hour < self.alert_thresholds["games_per_hour_low"]:
            self._create_alert(
                "warning", "selfplay",
                f"Low self-play throughput: {games_per_hour:.1f} games/hour",
                {"board_type": board_type, "num_players": num_players}
            )

    def record_data_quality(
        self,
        config_key: str,
        parity_passed: bool,
        parity_failure_rate: float,
        games_checked: int,
        holdout_loss: Optional[float] = None,
        overfit_gap: Optional[float] = None,
    ):
        """Record data quality metrics for dashboard display.

        Args:
            config_key: Configuration key (e.g., 'square8_2p')
            parity_passed: Whether parity validation passed
            parity_failure_rate: Rate of parity failures (0-1)
            games_checked: Number of games checked in validation
            holdout_loss: Optional holdout evaluation loss
            overfit_gap: Optional overfitting gap (holdout_loss - train_loss)
        """
        now = iso_now()

        # Store in SQLite for historical tracking
        try:
            self.db.conn.execute("""
                INSERT INTO data_quality_metrics
                (timestamp, config_key, parity_passed, parity_failure_rate,
                 games_checked, holdout_loss, overfit_gap)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (now, config_key, 1 if parity_passed else 0, parity_failure_rate,
                  games_checked, holdout_loss, overfit_gap))
            self.db.conn.commit()
        except sqlite3.OperationalError:
            # Table may not exist yet - create it
            self.db.conn.execute("""
                CREATE TABLE IF NOT EXISTS data_quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    config_key TEXT NOT NULL,
                    parity_passed INTEGER NOT NULL,
                    parity_failure_rate REAL NOT NULL,
                    games_checked INTEGER NOT NULL,
                    holdout_loss REAL,
                    overfit_gap REAL
                )
            """)
            self.db.conn.execute("""
                INSERT INTO data_quality_metrics
                (timestamp, config_key, parity_passed, parity_failure_rate,
                 games_checked, holdout_loss, overfit_gap)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (now, config_key, 1 if parity_passed else 0, parity_failure_rate,
                  games_checked, holdout_loss, overfit_gap))
            self.db.conn.commit()

        # Create alert if parity failure rate is high
        if parity_failure_rate > 0.10:  # 10% threshold
            self._create_alert(
                "warning", "data_quality",
                f"High parity failure rate for {config_key}: {parity_failure_rate:.1%}",
                {"config_key": config_key, "failure_rate": parity_failure_rate}
            )

        # Create alert if overfitting detected
        if overfit_gap is not None and overfit_gap > 0.15:  # 15% overfit gap threshold
            self._create_alert(
                "warning", "data_quality",
                f"Overfitting detected for {config_key}: gap={overfit_gap:.4f}",
                {"config_key": config_key, "overfit_gap": overfit_gap}
            )

    def _check_loss_anomaly(self, loss: float):
        """Check for loss spikes."""
        self._loss_history.append(loss)
        if len(self._loss_history) > 100:
            self._loss_history = self._loss_history[-100:]

        if len(self._loss_history) >= 10:
            avg_loss = np.mean(self._loss_history[:-1])
            if loss > avg_loss * self.alert_thresholds["loss_spike"]:
                self._create_alert(
                    "warning", "training",
                    f"Loss spike detected: {loss:.4f} (avg: {avg_loss:.4f})",
                    {"current_loss": loss, "avg_loss": avg_loss}
                )

    def _check_elo_drop(self, model_id: str, elo: float):
        """Check for Elo drops."""
        if model_id in self._elo_history:
            prev_elo = self._elo_history[model_id]
            drop = prev_elo - elo
            if drop > self.alert_thresholds["elo_drop"]:
                self._create_alert(
                    "warning", "model",
                    f"Elo drop for {model_id}: {prev_elo:.0f} -> {elo:.0f} (-{drop:.0f})",
                    {"model_id": model_id, "previous_elo": prev_elo, "current_elo": elo}
                )
        self._elo_history[model_id] = elo

    def _check_resource_issues(
        self,
        host_name: str,
        cpu_percent: float,
        gpu_memory_percent: float,
        disk_percent: float,
        memory_percent: float = 0.0,
    ):
        """Check for resource issues against resource_guard limits."""
        if cpu_percent > self.alert_thresholds["cpu_high"]:
            self._create_alert(
                "warning", "cluster",
                f"High CPU usage on {host_name}: {cpu_percent:.1f}% (limit: {self.alert_thresholds['cpu_high']:.0f}%)",
                {"host": host_name, "cpu_percent": cpu_percent, "limit": self.alert_thresholds["cpu_high"]}
            )

        if memory_percent > self.alert_thresholds.get("memory_high", 80.0):
            self._create_alert(
                "warning", "cluster",
                f"High memory usage on {host_name}: {memory_percent:.1f}% (limit: {self.alert_thresholds.get('memory_high', 80.0):.0f}%)",
                {"host": host_name, "memory_percent": memory_percent, "limit": self.alert_thresholds.get("memory_high", 80.0)}
            )

        if gpu_memory_percent > self.alert_thresholds["gpu_memory_high"]:
            self._create_alert(
                "warning", "cluster",
                f"High GPU memory on {host_name}: {gpu_memory_percent:.1f}% (limit: {self.alert_thresholds['gpu_memory_high']:.0f}%)",
                {"host": host_name, "gpu_memory_percent": gpu_memory_percent, "limit": self.alert_thresholds["gpu_memory_high"]}
            )

        if disk_percent > self.alert_thresholds["disk_high"]:
            self._create_alert(
                "error", "cluster",
                f"High disk usage on {host_name}: {disk_percent:.1f}% (limit: {self.alert_thresholds['disk_high']:.0f}%)",
                {"host": host_name, "disk_percent": disk_percent, "limit": self.alert_thresholds["disk_high"]}
            )

    def _create_alert(
        self,
        severity: str,
        category: str,
        message: str,
        details: Dict[str, Any],
    ):
        """Create and store an alert."""
        alert = Alert(
            timestamp=iso_now(),
            severity=severity,
            category=category,
            message=message,
            details=details,
        )
        self.db.insert_alert(alert)
        logger.warning(f"[ALERT] [{severity.upper()}] {message}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics."""
        now = utc_now()
        one_hour_ago = time_ago(hours=1)

        # Recent training metrics
        training_metrics = self.db.get_training_metrics(since=one_hour_ago, limit=100)
        avg_loss = np.mean([m.loss for m in training_metrics]) if training_metrics else 0

        # Latest Elo
        elo_snapshots = self.db.get_elo_history(limit=10)
        latest_elos = {}
        for s in elo_snapshots:
            if s.model_id not in latest_elos:
                latest_elos[s.model_id] = s.elo

        # Cluster status
        cluster_metrics = self.db.get_cluster_metrics(since=one_hour_ago, limit=100)
        hosts_status = {}
        for m in cluster_metrics:
            if m.host_name not in hosts_status:
                hosts_status[m.host_name] = {
                    "cpu": m.cpu_percent,
                    "gpu": m.gpu_percent,
                    "memory": m.memory_percent,
                }

        # Unacknowledged alerts
        alerts = self.db.get_alerts(unacknowledged_only=True, limit=10)

        return {
            "timestamp": to_iso(now),
            "training": {
                "recent_steps": len(training_metrics),
                "avg_loss": avg_loss,
            },
            "elo": latest_elos,
            "cluster": hosts_status,
            "alerts": [{"severity": a.severity, "message": a.message} for a in alerts],
        }


# =============================================================================
# Dashboard Server
# =============================================================================

class DashboardServer:
    """HTTP server for the monitoring dashboard."""

    def __init__(
        self,
        collector: MetricsCollector,
        static_dir: Optional[Path] = None,
    ):
        """Initialize server.

        Args:
            collector: Metrics collector instance
            static_dir: Directory for static files
        """
        self.collector = collector
        self.static_dir = static_dir or AI_SERVICE_ROOT / "app" / "monitoring" / "static"

    def _generate_dashboard_html(self) -> str:
        """Generate dashboard HTML."""
        summary = self.collector.get_summary()

        # Get historical data for charts
        one_day_ago = time_ago(days=1)
        training_history = self.collector.db.get_training_metrics(since=one_day_ago, limit=500)
        elo_history = self.collector.db.get_elo_history(since=one_day_ago, limit=200)

        # Prepare chart data
        loss_data = [{"x": m.timestamp, "y": m.loss} for m in reversed(training_history)]
        elo_data = {}
        for s in reversed(elo_history):
            if s.model_id not in elo_data:
                elo_data[s.model_id] = []
            elo_data[s.model_id].append({"x": s.timestamp, "y": s.elo})

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RingRift AI Training Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #eee; }}
        .header {{ background: #16213e; padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; }}
        .header h1 {{ font-size: 1.5rem; color: #0f4c75; }}
        .header h1 {{ color: #3282b8; }}
        .container {{ padding: 1rem 2rem; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; }}
        .card {{ background: #16213e; border-radius: 8px; padding: 1rem; }}
        .card h2 {{ color: #3282b8; font-size: 1rem; margin-bottom: 0.5rem; border-bottom: 1px solid #0f4c75; padding-bottom: 0.5rem; }}
        .metric {{ display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #0f4c7522; }}
        .metric:last-child {{ border-bottom: none; }}
        .metric-label {{ color: #888; }}
        .metric-value {{ font-weight: bold; }}
        .metric-value.good {{ color: #4caf50; }}
        .metric-value.warning {{ color: #ff9800; }}
        .metric-value.error {{ color: #f44336; }}
        .alert {{ padding: 0.5rem; margin: 0.25rem 0; border-radius: 4px; font-size: 0.85rem; }}
        .alert.warning {{ background: #ff980022; border-left: 3px solid #ff9800; }}
        .alert.error {{ background: #f4433622; border-left: 3px solid #f44336; }}
        .alert.info {{ background: #2196f322; border-left: 3px solid #2196f3; }}
        .chart-container {{ height: 200px; margin-top: 1rem; }}
        .refresh-btn {{ background: #3282b8; color: white; border: none; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer; }}
        .refresh-btn:hover {{ background: #0f4c75; }}
        .status-dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 0.5rem; }}
        .status-dot.online {{ background: #4caf50; }}
        .status-dot.warning {{ background: #ff9800; }}
        .status-dot.offline {{ background: #f44336; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎮 RingRift AI Training Dashboard</h1>
        <div>
            <span id="last-update">Last updated: {summary['timestamp'][:19]}</span>
            <button class="refresh-btn" onclick="location.reload()">↻ Refresh</button>
        </div>
    </div>

    <div class="container">
        <div class="grid">
            <!-- Training Metrics -->
            <div class="card">
                <h2>📊 Training</h2>
                <div class="metric">
                    <span class="metric-label">Recent Steps</span>
                    <span class="metric-value">{summary['training']['recent_steps']}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Loss (1h)</span>
                    <span class="metric-value">{summary['training']['avg_loss']:.4f}</span>
                </div>
                <div class="chart-container">
                    <canvas id="lossChart"></canvas>
                </div>
            </div>

            <!-- Elo Ratings -->
            <div class="card">
                <h2>🏆 Elo Ratings</h2>
                {''.join(f'''<div class="metric">
                    <span class="metric-label">{model}</span>
                    <span class="metric-value">{elo:.0f}</span>
                </div>''' for model, elo in list(summary['elo'].items())[:5])}
                <div class="chart-container">
                    <canvas id="eloChart"></canvas>
                </div>
            </div>

            <!-- Cluster Status -->
            <div class="card">
                <h2>🖥️ Cluster Status</h2>
                {''.join(f'''<div class="metric">
                    <span class="metric-label">
                        <span class="status-dot {'online' if status['cpu'] < 90 else 'warning'}"></span>
                        {host}
                    </span>
                    <span class="metric-value">CPU: {status['cpu']:.0f}% | GPU: {status['gpu']:.0f}%</span>
                </div>''' for host, status in list(summary['cluster'].items())[:6])}
            </div>

            <!-- Alerts -->
            <div class="card">
                <h2>⚠️ Alerts</h2>
                {''.join(f'''<div class="alert {a['severity']}">
                    {a['message']}
                </div>''' for a in summary['alerts'][:5]) or '<p style="color: #4caf50;">No active alerts</p>'}
            </div>
        </div>
    </div>

    <script>
        // Loss chart
        const lossCtx = document.getElementById('lossChart').getContext('2d');
        new Chart(lossCtx, {{
            type: 'line',
            data: {{
                datasets: [{{
                    label: 'Loss',
                    data: {json.dumps(loss_data[-100:])},
                    borderColor: '#3282b8',
                    backgroundColor: '#3282b822',
                    fill: true,
                    tension: 0.1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{ type: 'time', display: false }},
                    y: {{ beginAtZero: false }}
                }},
                plugins: {{ legend: {{ display: false }} }}
            }}
        }});

        // Elo chart
        const eloCtx = document.getElementById('eloChart').getContext('2d');
        const eloDatasets = [];
        const colors = ['#3282b8', '#4caf50', '#ff9800', '#f44336', '#9c27b0'];
        let colorIdx = 0;
        for (const [model, data] of Object.entries({json.dumps(elo_data)})) {{
            eloDatasets.push({{
                label: model,
                data: data.slice(-50),
                borderColor: colors[colorIdx % colors.length],
                fill: false,
                tension: 0.1
            }});
            colorIdx++;
        }}
        new Chart(eloCtx, {{
            type: 'line',
            data: {{ datasets: eloDatasets }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{ type: 'time', display: false }},
                    y: {{ beginAtZero: false }}
                }},
                plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 10 }} }} }}
            }}
        }});

        // Auto-refresh every 30 seconds
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>"""
        return html

    def _handle_api_summary(self) -> Dict[str, Any]:
        """Handle API summary request."""
        return self.collector.get_summary()

    def _handle_api_training(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Handle API training metrics request."""
        since = time_ago(hours=hours)
        metrics = self.collector.db.get_training_metrics(since=since, limit=1000)
        return [asdict(m) for m in metrics]

    def _handle_api_elo(self, hours: int = 168) -> List[Dict[str, Any]]:
        """Handle API Elo history request."""
        since = time_ago(hours=hours)
        snapshots = self.collector.db.get_elo_history(since=since, limit=500)
        return [asdict(s) for s in snapshots]

    def run(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the dashboard server.

        Uses simple HTTP server for compatibility.
        """
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import urllib.parse

        dashboard = self

        class DashboardHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path

                if path == "/" or path == "/dashboard":
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(dashboard._generate_dashboard_html().encode())

                elif path == "/api/summary":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(dashboard._handle_api_summary()).encode())

                elif path == "/api/training":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(dashboard._handle_api_training()).encode())

                elif path == "/api/elo":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(dashboard._handle_api_elo()).encode())

                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass  # Suppress logging

        server = HTTPServer((host, port), DashboardHandler)
        logger.info(f"Dashboard server running at http://{host}:{port}")
        print(f"🎮 RingRift Training Dashboard: http://localhost:{port}")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="RingRift Training Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--db", type=str, help="Metrics database path")

    args = parser.parse_args()

    db_path = Path(args.db) if args.db else None
    collector = MetricsCollector(db_path=db_path)
    server = DashboardServer(collector)
    server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
