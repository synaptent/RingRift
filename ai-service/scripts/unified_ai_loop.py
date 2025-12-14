#!/usr/bin/env python3
"""Unified AI Self-Improvement Loop - Single coordinator for the complete improvement cycle.

This daemon integrates all components of the AI improvement loop:
1. Streaming Data Collection - 60-second incremental sync from all hosts
2. Shadow Tournament Service - 15-minute lightweight evaluation
3. Training Scheduler - Auto-trigger when data thresholds met
4. Model Promoter - Auto-deploy on Elo threshold
5. Adaptive Curriculum - Elo-weighted training focus

Replaces the need for separate daemons by providing a single entry point
that coordinates all improvement activities with tight integration.

Usage:
    # Start the unified loop
    python scripts/unified_ai_loop.py --start

    # Run in foreground with verbose output
    python scripts/unified_ai_loop.py --foreground --verbose

    # Check status
    python scripts/unified_ai_loop.py --status

    # Stop gracefully
    python scripts/unified_ai_loop.py --stop

    # Use custom config
    python scripts/unified_ai_loop.py --config config/unified_loop.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import sqlite3
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import yaml

# Optional Prometheus client
try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

# Allow imports from app/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
RINGRIFT_ROOT = AI_SERVICE_ROOT.parent


# =============================================================================
# Prometheus Metrics
# =============================================================================

if HAS_PROMETHEUS:
    # Data collection metrics
    GAMES_SYNCED_TOTAL = Counter(
        'ringrift_games_synced_total',
        'Total games synced from remote hosts',
        ['host']
    )
    SYNC_DURATION_SECONDS = Histogram(
        'ringrift_sync_duration_seconds',
        'Time taken to sync games from a host',
        ['host'],
        buckets=[1, 5, 10, 30, 60, 120, 300]
    )
    SYNC_ERRORS_TOTAL = Counter(
        'ringrift_sync_errors_total',
        'Total sync errors by host',
        ['host', 'error_type']
    )
    GAMES_PENDING_TRAINING = Gauge(
        'ringrift_games_pending_training',
        'Games collected but not yet used for training',
        ['config']
    )

    # Training metrics
    TRAINING_RUNS_TOTAL = Counter(
        'ringrift_training_runs_total',
        'Total training runs',
        ['config', 'status']
    )
    TRAINING_DURATION_SECONDS = Histogram(
        'ringrift_training_duration_seconds',
        'Training run duration in seconds',
        ['config'],
        buckets=[60, 300, 600, 1800, 3600, 7200]
    )
    TRAINING_IN_PROGRESS = Gauge(
        'ringrift_training_in_progress',
        'Whether training is currently running',
        ['config']
    )

    # Evaluation metrics
    EVALUATIONS_TOTAL = Counter(
        'ringrift_evaluations_total',
        'Total evaluation runs',
        ['config', 'type']
    )
    EVALUATION_DURATION_SECONDS = Histogram(
        'ringrift_evaluation_duration_seconds',
        'Evaluation duration in seconds',
        ['config', 'type'],
        buckets=[30, 60, 120, 300, 600, 1200]
    )
    CURRENT_ELO = Gauge(
        'ringrift_current_elo',
        'Current Elo rating for configuration',
        ['config', 'model']
    )
    ELO_TREND = Gauge(
        'ringrift_elo_trend',
        'Elo trend (positive = improving)',
        ['config']
    )

    # Promotion metrics
    PROMOTIONS_TOTAL = Counter(
        'ringrift_promotions_total',
        'Total model promotions',
        ['config', 'status']
    )
    ELO_GAIN_ON_PROMOTION = Histogram(
        'ringrift_elo_gain_on_promotion',
        'Elo gain when model is promoted',
        ['config'],
        buckets=[5, 10, 20, 30, 50, 100]
    )
    PROMOTION_CANDIDATES = Gauge(
        'ringrift_promotion_candidates',
        'Number of promotion candidates',
        []
    )

    # Curriculum metrics
    CURRICULUM_WEIGHT = Gauge(
        'ringrift_curriculum_weight',
        'Training weight for configuration',
        ['config']
    )
    CURRICULUM_REBALANCES_TOTAL = Counter(
        'ringrift_curriculum_rebalances_total',
        'Total curriculum rebalancing events',
        []
    )

    # System metrics
    LOOP_CYCLES_TOTAL = Counter(
        'ringrift_loop_cycles_total',
        'Total improvement loop cycles',
        ['loop']
    )
    LOOP_ERRORS_TOTAL = Counter(
        'ringrift_loop_errors_total',
        'Total loop errors',
        ['loop', 'error_type']
    )
    UPTIME_SECONDS = Gauge(
        'ringrift_uptime_seconds',
        'Daemon uptime in seconds',
        []
    )
    HOSTS_ACTIVE = Gauge(
        'ringrift_hosts_active',
        'Number of active hosts',
        []
    )
    HOSTS_FAILED = Gauge(
        'ringrift_hosts_failed',
        'Number of failed hosts (consecutive failures)',
        []
    )


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics endpoint."""

    def do_GET(self):
        if self.path == '/metrics' and HAS_PROMETHEUS:
            self.send_response(200)
            self.send_header('Content-Type', CONTENT_TYPE_LATEST)
            self.end_headers()
            self.wfile.write(generate_latest())
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress request logging


def start_metrics_server(port: int = 9090) -> Optional[HTTPServer]:
    """Start the Prometheus metrics HTTP server."""
    if not HAS_PROMETHEUS:
        print("[Metrics] prometheus_client not installed, metrics disabled")
        return None

    try:
        server = HTTPServer(('0.0.0.0', port), MetricsHandler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"[Metrics] Prometheus metrics available at http://0.0.0.0:{port}/metrics")
        return server
    except Exception as e:
        print(f"[Metrics] Failed to start metrics server: {e}")
        return None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DataIngestionConfig:
    """Configuration for streaming data collection."""
    poll_interval_seconds: int = 60
    sync_method: str = "incremental"  # "incremental" or "full"
    deduplication: bool = True
    min_games_per_sync: int = 10
    remote_db_pattern: str = "data/games/*.db"


@dataclass
class TrainingConfig:
    """Configuration for automatic training triggers."""
    trigger_threshold_games: int = 1000
    min_interval_seconds: int = 1800  # 30 minutes minimum between training
    max_concurrent_jobs: int = 1
    prefer_gpu_hosts: bool = True
    training_script: str = "scripts/train_nnue.py"


@dataclass
class EvaluationConfig:
    """Configuration for continuous evaluation."""
    shadow_interval_seconds: int = 900  # 15 minutes
    shadow_games_per_config: int = 10
    full_tournament_interval_seconds: int = 3600  # 1 hour
    full_tournament_games: int = 50
    baseline_models: List[str] = field(default_factory=lambda: ["random", "heuristic", "mcts_100"])


@dataclass
class PromotionConfig:
    """Configuration for automatic model promotion."""
    auto_promote: bool = True
    elo_threshold: int = 20  # Must beat current best by 20 Elo
    min_games: int = 50
    significance_level: float = 0.05
    sync_to_cluster: bool = True


@dataclass
class CurriculumConfig:
    """Configuration for adaptive curriculum."""
    adaptive: bool = True
    rebalance_interval_seconds: int = 3600  # 1 hour
    max_weight_multiplier: float = 2.0
    min_weight_multiplier: float = 0.5


@dataclass
class UnifiedLoopConfig:
    """Complete configuration for the unified AI loop."""
    data_ingestion: DataIngestionConfig = field(default_factory=DataIngestionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    # Host configuration
    hosts_config_path: str = "config/remote_hosts.yaml"

    # Database paths
    unified_elo_db: str = "data/unified_elo.db"
    data_manifest_db: str = "data/data_manifest.db"

    # Logging
    log_dir: str = "logs/unified_loop"
    verbose: bool = False

    # Metrics
    metrics_port: int = 9090
    metrics_enabled: bool = True

    # Operation modes
    dry_run: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> "UnifiedLoopConfig":
        """Load configuration from YAML file."""
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        config = cls()

        if "data_ingestion" in data:
            for k, v in data["data_ingestion"].items():
                if hasattr(config.data_ingestion, k):
                    setattr(config.data_ingestion, k, v)

        if "training" in data:
            for k, v in data["training"].items():
                if hasattr(config.training, k):
                    setattr(config.training, k, v)

        if "evaluation" in data:
            for k, v in data["evaluation"].items():
                if hasattr(config.evaluation, k):
                    setattr(config.evaluation, k, v)

        if "promotion" in data:
            for k, v in data["promotion"].items():
                if hasattr(config.promotion, k):
                    setattr(config.promotion, k, v)

        if "curriculum" in data:
            for k, v in data["curriculum"].items():
                if hasattr(config.curriculum, k):
                    setattr(config.curriculum, k, v)

        for key in ["hosts_config_path", "unified_elo_db", "data_manifest_db", "log_dir",
                    "verbose", "metrics_port", "metrics_enabled", "dry_run"]:
            if key in data:
                setattr(config, key, data[key])

        return config


# =============================================================================
# Event System
# =============================================================================

class DataEventType(Enum):
    """Types of data pipeline events."""
    NEW_GAMES_AVAILABLE = "new_games"
    TRAINING_THRESHOLD_REACHED = "training_threshold"
    TRAINING_STARTED = "training_started"
    TRAINING_COMPLETED = "training_completed"
    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_COMPLETED = "evaluation_completed"
    PROMOTION_CANDIDATE = "promotion_candidate"
    MODEL_PROMOTED = "model_promoted"
    CURRICULUM_REBALANCED = "curriculum_rebalanced"


@dataclass
class DataEvent:
    """A data pipeline event."""
    event_type: DataEventType
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class EventBus:
    """Simple async event bus for component coordination."""

    def __init__(self):
        self._subscribers: Dict[DataEventType, List[Callable]] = {}
        self._event_history: List[DataEvent] = []
        self._max_history = 1000

    def subscribe(self, event_type: DataEventType, callback: Callable):
        """Subscribe to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    async def publish(self, event: DataEvent):
        """Publish an event to all subscribers."""
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        if event.event_type in self._subscribers:
            for callback in self._subscribers[event.event_type]:
                try:
                    result = callback(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    print(f"[EventBus] Error in subscriber: {e}")

    def get_recent_events(self, event_type: Optional[DataEventType] = None, limit: int = 100) -> List[DataEvent]:
        """Get recent events, optionally filtered by type."""
        events = self._event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]


# =============================================================================
# State Management
# =============================================================================

@dataclass
class HostState:
    """State for a remote host."""
    name: str
    ssh_host: str
    ssh_user: str = "ubuntu"
    ssh_port: int = 22
    last_sync_time: float = 0.0
    last_game_count: int = 0
    consecutive_failures: int = 0
    enabled: bool = True


@dataclass
class ConfigState:
    """State for a board/player configuration."""
    board_type: str
    num_players: int
    game_count: int = 0
    games_since_training: int = 0
    last_training_time: float = 0.0
    last_evaluation_time: float = 0.0
    current_elo: float = 1500.0
    elo_trend: float = 0.0  # Positive = improving
    training_weight: float = 1.0


@dataclass
class UnifiedLoopState:
    """Complete state for the unified AI loop."""
    started_at: str = ""
    last_cycle_at: str = ""

    # Cycle counters
    total_data_syncs: int = 0
    total_training_runs: int = 0
    total_evaluations: int = 0
    total_promotions: int = 0

    # Host states
    hosts: Dict[str, HostState] = field(default_factory=dict)

    # Configuration states (keyed by "board_type_num_players")
    configs: Dict[str, ConfigState] = field(default_factory=dict)

    # Current training state
    training_in_progress: bool = False
    training_config: str = ""
    training_started_at: float = 0.0

    # Games pending training
    total_games_pending: int = 0

    # Error tracking
    consecutive_failures: int = 0
    last_error: str = ""
    last_error_time: str = ""

    # Curriculum weights (config_key -> weight)
    curriculum_weights: Dict[str, float] = field(default_factory=dict)
    last_curriculum_rebalance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "started_at": self.started_at,
            "last_cycle_at": self.last_cycle_at,
            "total_data_syncs": self.total_data_syncs,
            "total_training_runs": self.total_training_runs,
            "total_evaluations": self.total_evaluations,
            "total_promotions": self.total_promotions,
            "hosts": {k: asdict(v) for k, v in self.hosts.items()},
            "configs": {k: asdict(v) for k, v in self.configs.items()},
            "training_in_progress": self.training_in_progress,
            "training_config": self.training_config,
            "training_started_at": self.training_started_at,
            "total_games_pending": self.total_games_pending,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "last_error_time": self.last_error_time,
            "curriculum_weights": self.curriculum_weights,
            "last_curriculum_rebalance": self.last_curriculum_rebalance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedLoopState":
        """Create state from dictionary."""
        state = cls()
        for key in ["started_at", "last_cycle_at", "total_data_syncs", "total_training_runs",
                    "total_evaluations", "total_promotions", "training_in_progress",
                    "training_config", "training_started_at", "total_games_pending",
                    "consecutive_failures", "last_error", "last_error_time",
                    "last_curriculum_rebalance"]:
            if key in data:
                setattr(state, key, data[key])

        if "hosts" in data:
            for name, host_data in data["hosts"].items():
                state.hosts[name] = HostState(**host_data)

        if "configs" in data:
            for key, config_data in data["configs"].items():
                state.configs[key] = ConfigState(**config_data)

        if "curriculum_weights" in data:
            state.curriculum_weights = data["curriculum_weights"]

        return state


# =============================================================================
# Data Collection Component
# =============================================================================

class StreamingDataCollector:
    """Collects game data from remote hosts with 60-second incremental sync."""

    def __init__(self, config: DataIngestionConfig, state: UnifiedLoopState, event_bus: EventBus):
        self.config = config
        self.state = state
        self.event_bus = event_bus
        self._known_game_ids: Set[str] = set()

    async def sync_host(self, host: HostState) -> int:
        """Sync games from a single host. Returns count of new games."""
        if not host.enabled:
            return 0

        try:
            # Query game count on remote host
            ssh_target = f"{host.ssh_user}@{host.ssh_host}"
            port_arg = f"-p {host.ssh_port}" if host.ssh_port != 22 else ""

            # Get game count from all DBs
            cmd = f'ssh -o ConnectTimeout=10 {port_arg} {ssh_target} "cd ~/ringrift/ai-service && find data/games -name \'*.db\' -exec sqlite3 {{}} \'SELECT COUNT(*) FROM games\' \\; 2>/dev/null | awk \'{{s+=$1}} END {{print s}}\'"'

            result = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(result.communicate(), timeout=30)

            current_count = int(stdout.decode().strip() or "0")
            new_games = max(0, current_count - host.last_game_count)

            if new_games >= self.config.min_games_per_sync:
                # Trigger rsync for incremental sync
                if self.config.sync_method == "incremental":
                    await self._incremental_sync(host)
                else:
                    await self._full_sync(host)

                host.last_game_count = current_count
                host.last_sync_time = time.time()

                # Publish event
                await self.event_bus.publish(DataEvent(
                    event_type=DataEventType.NEW_GAMES_AVAILABLE,
                    payload={
                        "host": host.name,
                        "new_games": new_games,
                        "total_games": current_count,
                    }
                ))

            host.consecutive_failures = 0
            return new_games

        except Exception as e:
            host.consecutive_failures += 1
            print(f"[DataCollector] Failed to sync {host.name}: {e}")
            return 0

    async def _incremental_sync(self, host: HostState):
        """Perform incremental rsync of new data."""
        ssh_target = f"{host.ssh_user}@{host.ssh_host}"
        local_dir = AI_SERVICE_ROOT / "data" / "games" / "synced" / host.name
        local_dir.mkdir(parents=True, exist_ok=True)

        # Rsync with append mode for incremental transfer
        cmd = f'rsync -avz --progress -e "ssh -o ConnectTimeout=10" {ssh_target}:~/ringrift/ai-service/data/games/*.db {local_dir}/'

        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(process.communicate(), timeout=300)

    async def _full_sync(self, host: HostState):
        """Perform full sync (same as incremental for now)."""
        await self._incremental_sync(host)

    async def run_collection_cycle(self) -> int:
        """Run one data collection cycle across all hosts."""
        total_new = 0
        tasks = []

        for host in self.state.hosts.values():
            if host.enabled:
                tasks.append(self.sync_host(host))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, int):
                    total_new += result

        self.state.total_data_syncs += 1
        self.state.total_games_pending += total_new

        return total_new


# =============================================================================
# Shadow Tournament Component
# =============================================================================

class ShadowTournamentService:
    """Runs lightweight continuous evaluation."""

    def __init__(self, config: EvaluationConfig, state: UnifiedLoopState, event_bus: EventBus):
        self.config = config
        self.state = state
        self.event_bus = event_bus

    async def run_shadow_tournament(self, config_key: str) -> Dict[str, Any]:
        """Run a quick shadow tournament for a configuration."""
        parts = config_key.rsplit("_", 1)
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))

        await self.event_bus.publish(DataEvent(
            event_type=DataEventType.EVALUATION_STARTED,
            payload={"config": config_key, "type": "shadow"}
        ))

        try:
            # Run quick tournament
            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / "scripts" / "run_model_elo_tournament.py"),
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--games", str(self.config.shadow_games_per_config),
                "--quick",
                "--include-baselines",
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=600)

            # Parse results (simplified - real implementation would parse JSON output)
            result = {
                "config": config_key,
                "games_played": self.config.shadow_games_per_config,
                "success": process.returncode == 0,
            }

            if config_key in self.state.configs:
                self.state.configs[config_key].last_evaluation_time = time.time()

            self.state.total_evaluations += 1

            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.EVALUATION_COMPLETED,
                payload=result
            ))

            return result

        except Exception as e:
            print(f"[ShadowTournament] Error running tournament for {config_key}: {e}")
            return {"config": config_key, "error": str(e), "success": False}

    async def run_full_tournament(self) -> Dict[str, Any]:
        """Run a full tournament across all configurations."""
        await self.event_bus.publish(DataEvent(
            event_type=DataEventType.EVALUATION_STARTED,
            payload={"type": "full"}
        ))

        try:
            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / "scripts" / "run_model_elo_tournament.py"),
                "--all-configs",
                "--games", str(self.config.full_tournament_games),
                "--include-baselines",
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=3600)

            result = {
                "type": "full",
                "success": process.returncode == 0,
            }

            self.state.total_evaluations += 1

            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.EVALUATION_COMPLETED,
                payload=result
            ))

            return result

        except Exception as e:
            print(f"[ShadowTournament] Error running full tournament: {e}")
            return {"type": "full", "error": str(e), "success": False}


# =============================================================================
# Training Scheduler Component
# =============================================================================

class TrainingScheduler:
    """Schedules and manages training runs."""

    def __init__(self, config: TrainingConfig, state: UnifiedLoopState, event_bus: EventBus):
        self.config = config
        self.state = state
        self.event_bus = event_bus
        self._training_process: Optional[asyncio.subprocess.Process] = None

    def should_trigger_training(self) -> Optional[str]:
        """Check if training should be triggered. Returns config key or None."""
        if self.state.training_in_progress:
            return None

        # Check minimum interval
        now = time.time()

        for config_key, config_state in self.state.configs.items():
            if config_state.games_since_training >= self.config.trigger_threshold_games:
                if now - config_state.last_training_time >= self.config.min_interval_seconds:
                    return config_key

        return None

    async def start_training(self, config_key: str) -> bool:
        """Start a training run for the given configuration."""
        if self.state.training_in_progress:
            return False

        parts = config_key.rsplit("_", 1)
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))

        await self.event_bus.publish(DataEvent(
            event_type=DataEventType.TRAINING_STARTED,
            payload={"config": config_key}
        ))

        try:
            self.state.training_in_progress = True
            self.state.training_config = config_key
            self.state.training_started_at = time.time()

            # Use v3 for all board types (best architecture with spatial policy heads)
            model_version = "v3"

            # Start training process
            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / self.config.training_script),
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--model-version", model_version,
            ]

            self._training_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
            )

            return True

        except Exception as e:
            print(f"[TrainingScheduler] Error starting training: {e}")
            self.state.training_in_progress = False
            return False

    async def check_training_status(self) -> Optional[Dict[str, Any]]:
        """Check if current training has completed."""
        if not self.state.training_in_progress or not self._training_process:
            return None

        # Check if process has finished
        if self._training_process.returncode is not None:
            stdout, stderr = await self._training_process.communicate()

            success = self._training_process.returncode == 0
            config_key = self.state.training_config

            self.state.training_in_progress = False
            self.state.training_config = ""
            self.state.total_training_runs += 1
            self._training_process = None

            if config_key in self.state.configs:
                self.state.configs[config_key].last_training_time = time.time()
                self.state.configs[config_key].games_since_training = 0

            result = {
                "config": config_key,
                "success": success,
                "duration": time.time() - self.state.training_started_at,
            }

            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.TRAINING_COMPLETED,
                payload=result
            ))

            return result

        return None


# =============================================================================
# Model Promoter Component
# =============================================================================

class ModelPromoter:
    """Handles automatic model promotion based on Elo."""

    def __init__(self, config: PromotionConfig, state: UnifiedLoopState, event_bus: EventBus):
        self.config = config
        self.state = state
        self.event_bus = event_bus

    async def check_promotion_candidates(self) -> List[Dict[str, Any]]:
        """Check for models that should be promoted."""
        if not self.config.auto_promote:
            return []

        candidates = []

        try:
            # Query Elo database for candidates
            elo_db_path = AI_SERVICE_ROOT / "data" / "unified_elo.db"
            if not elo_db_path.exists():
                return []

            conn = sqlite3.connect(elo_db_path)
            cursor = conn.cursor()

            # Find models that beat current best by threshold
            cursor.execute("""
                SELECT participant_id, board_type, num_players, rating, games_played
                FROM elo_ratings
                WHERE games_played >= ?
                ORDER BY board_type, num_players, rating DESC
            """, (self.config.min_games,))

            rows = cursor.fetchall()
            conn.close()

            # Group by config and find candidates
            by_config: Dict[str, List[Tuple]] = {}
            for row in rows:
                config_key = f"{row[1]}_{row[2]}p"
                if config_key not in by_config:
                    by_config[config_key] = []
                by_config[config_key].append(row)

            for config_key, models in by_config.items():
                if len(models) < 2:
                    continue

                best = models[0]
                current_best_id = f"ringrift_best_{config_key.replace('_', '_')}"

                # Check if top model beats current best by threshold
                for model in models:
                    if model[0] == current_best_id:
                        continue
                    if model[3] - best[3] >= self.config.elo_threshold:
                        candidates.append({
                            "model_id": model[0],
                            "config": config_key,
                            "elo": model[3],
                            "games": model[4],
                            "elo_gain": model[3] - best[3],
                        })
                        break

            return candidates

        except Exception as e:
            print(f"[ModelPromoter] Error checking candidates: {e}")
            return []

    async def execute_promotion(self, candidate: Dict[str, Any]) -> bool:
        """Execute a model promotion."""
        try:
            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.PROMOTION_CANDIDATE,
                payload=candidate
            ))

            # Run promotion script
            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / "scripts" / "auto_promote_best_models.py"),
                "--config", candidate["config"],
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)

            success = process.returncode == 0

            if success:
                self.state.total_promotions += 1

                await self.event_bus.publish(DataEvent(
                    event_type=DataEventType.MODEL_PROMOTED,
                    payload=candidate
                ))

                # Sync to cluster if enabled
                if self.config.sync_to_cluster:
                    await self._sync_to_cluster(candidate)

            return success

        except Exception as e:
            print(f"[ModelPromoter] Error executing promotion: {e}")
            return False

    async def _sync_to_cluster(self, candidate: Dict[str, Any]):
        """Sync promoted model to cluster."""
        try:
            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / "scripts" / "sync_models.py"),
                "--push-promoted",
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
            )
            await asyncio.wait_for(process.communicate(), timeout=300)

        except Exception as e:
            print(f"[ModelPromoter] Error syncing to cluster: {e}")


# =============================================================================
# Adaptive Curriculum Component
# =============================================================================

class AdaptiveCurriculum:
    """Manages Elo-weighted training curriculum."""

    def __init__(self, config: CurriculumConfig, state: UnifiedLoopState, event_bus: EventBus):
        self.config = config
        self.state = state
        self.event_bus = event_bus

    async def rebalance_weights(self) -> Dict[str, float]:
        """Recompute training weights based on Elo performance."""
        if not self.config.adaptive:
            return {}

        try:
            # Query Elo by config
            elo_db_path = AI_SERVICE_ROOT / "data" / "unified_elo.db"
            if not elo_db_path.exists():
                return {}

            conn = sqlite3.connect(elo_db_path)
            cursor = conn.cursor()

            # Get best Elo for each config
            cursor.execute("""
                SELECT board_type, num_players, MAX(rating) as best_elo
                FROM elo_ratings
                WHERE participant_id LIKE 'ringrift_%'
                GROUP BY board_type, num_players
            """)

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return {}

            elo_by_config = {
                f"{row[0]}_{row[1]}p": row[2]
                for row in rows
            }

            # Compute weights based on deviation from median
            elos = list(elo_by_config.values())
            median_elo = statistics.median(elos)

            new_weights = {}
            for config_key, elo in elo_by_config.items():
                # Boost weight for underperforming configs
                deficit = median_elo - elo
                weight = 1.0 + (deficit / 200.0)

                # Clamp to configured range
                weight = max(self.config.min_weight_multiplier,
                           min(self.config.max_weight_multiplier, weight))

                new_weights[config_key] = weight

            # Update state
            self.state.curriculum_weights = new_weights
            self.state.last_curriculum_rebalance = time.time()

            # Update config states
            for config_key, weight in new_weights.items():
                if config_key in self.state.configs:
                    self.state.configs[config_key].training_weight = weight

            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.CURRICULUM_REBALANCED,
                payload={"weights": new_weights}
            ))

            return new_weights

        except Exception as e:
            print(f"[AdaptiveCurriculum] Error rebalancing: {e}")
            return {}


# =============================================================================
# Main Unified Loop
# =============================================================================

class UnifiedAILoop:
    """Single coordinator for the complete AI improvement loop."""

    def __init__(self, config: UnifiedLoopConfig):
        self.config = config
        self.state = UnifiedLoopState()
        self.event_bus = EventBus()

        # Initialize components
        self.data_collector = StreamingDataCollector(
            config.data_ingestion, self.state, self.event_bus
        )
        self.shadow_tournament = ShadowTournamentService(
            config.evaluation, self.state, self.event_bus
        )
        self.training_scheduler = TrainingScheduler(
            config.training, self.state, self.event_bus
        )
        self.model_promoter = ModelPromoter(
            config.promotion, self.state, self.event_bus
        )
        self.adaptive_curriculum = AdaptiveCurriculum(
            config.curriculum, self.state, self.event_bus
        )

        # State management
        self._state_path = AI_SERVICE_ROOT / config.log_dir / "unified_loop_state.json"
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Timing trackers
        self._last_shadow_eval: Dict[str, float] = {}
        self._last_full_eval: float = 0.0
        self._started_time: float = 0.0

    def _update_metrics(self):
        """Update Prometheus metrics from current state."""
        if not HAS_PROMETHEUS:
            return

        # Update uptime
        if self._started_time > 0:
            UPTIME_SECONDS.set(time.time() - self._started_time)

        # Update host counts
        active_hosts = sum(1 for h in self.state.hosts.values() if h.enabled and h.consecutive_failures < 3)
        failed_hosts = sum(1 for h in self.state.hosts.values() if h.consecutive_failures >= 3)
        HOSTS_ACTIVE.set(active_hosts)
        HOSTS_FAILED.set(failed_hosts)

        # Update curriculum weights
        for config_key, weight in self.state.curriculum_weights.items():
            CURRICULUM_WEIGHT.labels(config=config_key).set(weight)

        # Update pending games
        for config_key, config_state in self.state.configs.items():
            GAMES_PENDING_TRAINING.labels(config=config_key).set(config_state.games_since_training)
            if config_state.current_elo > 0:
                CURRENT_ELO.labels(config=config_key, model="best").set(config_state.current_elo)
            ELO_TREND.labels(config=config_key).set(config_state.elo_trend)

        # Training in progress
        if self.state.training_in_progress:
            TRAINING_IN_PROGRESS.labels(config=self.state.training_config).set(1)
        else:
            for config_key in self.state.configs:
                TRAINING_IN_PROGRESS.labels(config=config_key).set(0)

    def _load_state(self):
        """Load state from checkpoint file."""
        if self._state_path.exists():
            try:
                with open(self._state_path) as f:
                    data = json.load(f)
                self.state = UnifiedLoopState.from_dict(data)
                print(f"[UnifiedLoop] Loaded state: {self.state.total_data_syncs} syncs, {self.state.total_training_runs} training runs")
            except Exception as e:
                print(f"[UnifiedLoop] Error loading state: {e}")

    def _save_state(self):
        """Save state to checkpoint file."""
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_path, "w") as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except Exception as e:
            print(f"[UnifiedLoop] Error saving state: {e}")

    def _load_hosts(self):
        """Load host configuration from YAML."""
        hosts_path = AI_SERVICE_ROOT / self.config.hosts_config_path
        if not hosts_path.exists():
            print(f"[UnifiedLoop] Hosts config not found: {hosts_path}")
            return

        try:
            with open(hosts_path) as f:
                hosts_data = yaml.safe_load(f)

            # Load standard hosts
            if "standard_hosts" in hosts_data:
                for name, data in hosts_data["standard_hosts"].items():
                    self.state.hosts[name] = HostState(
                        name=name,
                        ssh_host=data.get("ssh_host", ""),
                        ssh_user=data.get("ssh_user", "ubuntu"),
                        ssh_port=data.get("ssh_port", 22),
                    )

            print(f"[UnifiedLoop] Loaded {len(self.state.hosts)} hosts")

        except Exception as e:
            print(f"[UnifiedLoop] Error loading hosts: {e}")

    def _init_configs(self):
        """Initialize board/player configurations."""
        for board_type in ["square8", "square19", "hexagonal"]:
            for num_players in [2, 3, 4]:
                config_key = f"{board_type}_{num_players}p"
                if config_key not in self.state.configs:
                    self.state.configs[config_key] = ConfigState(
                        board_type=board_type,
                        num_players=num_players,
                    )

    async def _data_collection_loop(self):
        """Main data collection loop - runs every 60 seconds."""
        while self._running:
            try:
                new_games = await self.data_collector.run_collection_cycle()
                if new_games > 0:
                    print(f"[DataCollection] Synced {new_games} new games")

                    # Check if training threshold reached
                    trigger_config = self.training_scheduler.should_trigger_training()
                    if trigger_config:
                        await self.event_bus.publish(DataEvent(
                            event_type=DataEventType.TRAINING_THRESHOLD_REACHED,
                            payload={"config": trigger_config}
                        ))

            except Exception as e:
                print(f"[DataCollection] Error: {e}")
                self.state.consecutive_failures += 1

            self._save_state()

            # Wait for next cycle
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.config.data_ingestion.poll_interval_seconds
                )
                break  # Shutdown requested
            except asyncio.TimeoutError:
                pass  # Continue loop

    async def _evaluation_loop(self):
        """Main evaluation loop - shadow every 15 min, full every 1 hour."""
        while self._running:
            try:
                now = time.time()

                # Check for shadow tournaments
                for config_key in self.state.configs:
                    last_eval = self._last_shadow_eval.get(config_key, 0)
                    if now - last_eval >= self.config.evaluation.shadow_interval_seconds:
                        print(f"[Evaluation] Running shadow tournament for {config_key}")
                        await self.shadow_tournament.run_shadow_tournament(config_key)
                        self._last_shadow_eval[config_key] = now
                        break  # One at a time to avoid overload

                # Check for full tournament
                if now - self._last_full_eval >= self.config.evaluation.full_tournament_interval_seconds:
                    print("[Evaluation] Running full tournament")
                    await self.shadow_tournament.run_full_tournament()
                    self._last_full_eval = now

            except Exception as e:
                print(f"[Evaluation] Error: {e}")

            # Wait 60 seconds between checks
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=60)
                break
            except asyncio.TimeoutError:
                pass

    async def _training_loop(self):
        """Main training management loop."""
        while self._running:
            try:
                # Check if training completed
                result = await self.training_scheduler.check_training_status()
                if result:
                    print(f"[Training] Completed: {result}")

                # Check if we should start training
                if not self.state.training_in_progress:
                    trigger_config = self.training_scheduler.should_trigger_training()
                    if trigger_config:
                        print(f"[Training] Starting training for {trigger_config}")
                        await self.training_scheduler.start_training(trigger_config)

            except Exception as e:
                print(f"[Training] Error: {e}")

            # Check every 30 seconds
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=30)
                break
            except asyncio.TimeoutError:
                pass

    async def _promotion_loop(self):
        """Main promotion checking loop."""
        while self._running:
            try:
                candidates = await self.model_promoter.check_promotion_candidates()
                for candidate in candidates:
                    print(f"[Promotion] Found candidate: {candidate['model_id']} (+{candidate['elo_gain']} Elo)")
                    await self.model_promoter.execute_promotion(candidate)

            except Exception as e:
                print(f"[Promotion] Error: {e}")

            # Check every 5 minutes
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=300)
                break
            except asyncio.TimeoutError:
                pass

    async def _curriculum_loop(self):
        """Main curriculum rebalancing loop."""
        while self._running:
            try:
                now = time.time()
                if now - self.state.last_curriculum_rebalance >= self.config.curriculum.rebalance_interval_seconds:
                    weights = await self.adaptive_curriculum.rebalance_weights()
                    if weights:
                        print(f"[Curriculum] Rebalanced weights: {weights}")

            except Exception as e:
                print(f"[Curriculum] Error: {e}")

            # Check every 10 minutes
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=600)
                break
            except asyncio.TimeoutError:
                pass

    async def _metrics_loop(self):
        """Periodically update Prometheus metrics."""
        while self._running:
            try:
                self._update_metrics()
                if HAS_PROMETHEUS:
                    LOOP_CYCLES_TOTAL.labels(loop="metrics").inc()
            except Exception as e:
                print(f"[Metrics] Error: {e}")
                if HAS_PROMETHEUS:
                    LOOP_ERRORS_TOTAL.labels(loop="metrics", error_type=type(e).__name__).inc()

            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=15)
                break
            except asyncio.TimeoutError:
                pass

    async def _external_drive_sync_loop(self):
        """External drive sync loop - syncs data to Mac Studio external drive."""
        # Check if external drive sync is enabled in config
        config_path = AI_SERVICE_ROOT / "config" / "unified_loop.yaml"
        if not config_path.exists():
            return

        try:
            with open(config_path) as f:
                full_config = yaml.safe_load(f) or {}

            ext_config = full_config.get("external_drive_sync", {})
            if not ext_config.get("enabled", False):
                print("[ExternalDriveSync] Disabled in config")
                return

            target_dir = Path(ext_config.get("target_dir", "/Volumes/RingRift-Data/selfplay_repository"))
            sync_interval = ext_config.get("sync_interval_seconds", 300)
            sync_models = ext_config.get("sync_models", True)
            run_analysis = ext_config.get("run_analysis", True)

            # Check if target directory parent exists (drive is mounted)
            if not target_dir.parent.exists():
                print(f"[ExternalDriveSync] External drive not mounted at {target_dir.parent}")
                return

            print(f"[ExternalDriveSync] Starting with target={target_dir}, interval={sync_interval}s")

        except Exception as e:
            print(f"[ExternalDriveSync] Config error: {e}")
            return

        while self._running:
            try:
                # Run external_drive_sync_daemon.py --once
                cmd = [
                    sys.executable,
                    str(AI_SERVICE_ROOT / "scripts" / "external_drive_sync_daemon.py"),
                    "--once",
                    "--target", str(target_dir),
                    "--config", str(AI_SERVICE_ROOT / "config" / "distributed_hosts.yaml"),
                ]

                if not sync_models:
                    cmd.append("--no-models")
                if not run_analysis:
                    cmd.append("--no-analysis")

                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=AI_SERVICE_ROOT,
                )
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=1800)

                if process.returncode == 0:
                    print(f"[ExternalDriveSync] Cycle complete")
                else:
                    print(f"[ExternalDriveSync] Error: {stderr.decode()[:200]}")

            except asyncio.TimeoutError:
                print("[ExternalDriveSync] Sync timed out")
            except Exception as e:
                print(f"[ExternalDriveSync] Error: {e}")

            # Wait for next cycle
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=sync_interval)
                break
            except asyncio.TimeoutError:
                pass

    async def run(self):
        """Main entry point - runs all loops concurrently."""
        self._running = True
        self._started_time = time.time()
        self.state.started_at = datetime.now().isoformat()

        # Load previous state and configuration
        self._load_state()
        self._load_hosts()
        self._init_configs()

        dry_run_msg = " (DRY RUN)" if self.config.dry_run else ""
        print(f"[UnifiedLoop] Starting with {len(self.state.hosts)} hosts, {len(self.state.configs)} configs{dry_run_msg}")
        print(f"[UnifiedLoop] Data sync: {self.config.data_ingestion.poll_interval_seconds}s")
        print(f"[UnifiedLoop] Shadow eval: {self.config.evaluation.shadow_interval_seconds}s")
        print(f"[UnifiedLoop] Full eval: {self.config.evaluation.full_tournament_interval_seconds}s")

        if self.config.dry_run:
            print("[UnifiedLoop] Dry run - showing planned operations:")
            for host_name, host in self.state.hosts.items():
                print(f"  - Would sync from {host.ssh_user}@{host.ssh_host}")
            for config_key in self.state.configs:
                print(f"  - Would run evaluations for {config_key}")
            print("[UnifiedLoop] Dry run complete - exiting")
            return

        # Start all loops including metrics and external drive sync
        await asyncio.gather(
            self._data_collection_loop(),
            self._evaluation_loop(),
            self._training_loop(),
            self._promotion_loop(),
            self._curriculum_loop(),
            self._metrics_loop(),
            self._external_drive_sync_loop(),
        )

        print("[UnifiedLoop] Shutdown complete")

    def stop(self):
        """Request graceful shutdown."""
        self._running = False
        self._shutdown_event.set()
        self._save_state()


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified AI Self-Improvement Loop")
    parser.add_argument("--start", action="store_true", help="Start the daemon in background")
    parser.add_argument("--foreground", "-f", action="store_true", help="Run in foreground")
    parser.add_argument("--stop", action="store_true", help="Stop the daemon")
    parser.add_argument("--status", action="store_true", help="Show daemon status")
    parser.add_argument("--config", type=str, default="config/unified_loop.yaml", help="Config file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode - simulate without executing")
    parser.add_argument("--metrics-port", type=int, default=9090, help="Prometheus metrics port")
    parser.add_argument("--no-metrics", action="store_true", help="Disable Prometheus metrics")

    args = parser.parse_args()

    config_path = AI_SERVICE_ROOT / args.config
    config = UnifiedLoopConfig.from_yaml(config_path)
    config.verbose = args.verbose
    config.dry_run = args.dry_run
    config.metrics_port = args.metrics_port
    config.metrics_enabled = not args.no_metrics

    if args.status:
        state_path = AI_SERVICE_ROOT / config.log_dir / "unified_loop_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            print("Unified AI Loop Status:")
            print(f"  Started: {state.get('started_at', 'N/A')}")
            print(f"  Last cycle: {state.get('last_cycle_at', 'N/A')}")
            print(f"  Data syncs: {state.get('total_data_syncs', 0)}")
            print(f"  Training runs: {state.get('total_training_runs', 0)}")
            print(f"  Evaluations: {state.get('total_evaluations', 0)}")
            print(f"  Promotions: {state.get('total_promotions', 0)}")
        else:
            print("No state file found - daemon may not be running")
        return

    if args.stop:
        pid_path = AI_SERVICE_ROOT / config.log_dir / "unified_loop.pid"
        if pid_path.exists():
            pid = int(pid_path.read_text().strip())
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"Sent SIGTERM to PID {pid}")
            except ProcessLookupError:
                print(f"Process {pid} not found")
            pid_path.unlink()
        else:
            print("No PID file found")
        return

    if args.start or args.foreground:
        if config.dry_run:
            print("[UnifiedLoop] DRY RUN MODE - no actual operations will be performed")

        # Start metrics server
        metrics_server = None
        if config.metrics_enabled and not config.dry_run:
            metrics_server = start_metrics_server(config.metrics_port)

        loop = UnifiedAILoop(config)

        # Handle signals
        def signal_handler(sig, frame):
            print("\n[UnifiedLoop] Received shutdown signal")
            loop.stop()
            if metrics_server:
                metrics_server.shutdown()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        if args.start and not args.foreground:
            # Daemonize
            pid = os.fork()
            if pid > 0:
                # Parent
                pid_path = AI_SERVICE_ROOT / config.log_dir / "unified_loop.pid"
                pid_path.parent.mkdir(parents=True, exist_ok=True)
                pid_path.write_text(str(pid))
                print(f"Started daemon with PID {pid}")
                return

        # Run the loop
        asyncio.run(loop.run())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
