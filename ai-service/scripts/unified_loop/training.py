"""Unified Loop Training Scheduler.

This module contains the training scheduler for the unified AI loop:
- TrainingScheduler: Schedules and manages training runs with cluster-wide coordination

Extracted from unified_ai_loop.py for better modularity (Phase 2 refactoring).
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .config import (
    DataEvent,
    DataEventType,
    FeedbackConfig,
    TrainingConfig,
)

if TYPE_CHECKING:
    from unified_ai_loop import ConfigPriorityQueue, EventBus, UnifiedLoopState
    from app.integration.pipeline_feedback import PipelineFeedbackController

# Path constants
AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]

# Feature flag: disable local tasks when running on dedicated hosts
DISABLE_LOCAL_TASKS = os.environ.get("RINGRIFT_DISABLE_LOCAL_TASKS", "").lower() in ("1", "true", "yes")

# Optional ELO service import
try:
    from app.training.elo_service import get_elo_service
    HAS_ELO_SERVICE = True
except ImportError:
    HAS_ELO_SERVICE = False
    get_elo_service = None

# Optional Prometheus metrics - avoid duplicate registration
try:
    from prometheus_client import Counter, REGISTRY
    HAS_PROMETHEUS = True
    if 'ringrift_data_quality_blocked_training_total' in REGISTRY._names_to_collectors:
        DATA_QUALITY_BLOCKED_TRAINING = REGISTRY._names_to_collectors['ringrift_data_quality_blocked_training_total']
    else:
        DATA_QUALITY_BLOCKED_TRAINING = Counter(
            'ringrift_data_quality_blocked_training_total',
            'Training runs blocked by data quality gate',
            ['reason']
        )
except ImportError:
    HAS_PROMETHEUS = False
    DATA_QUALITY_BLOCKED_TRAINING = None

# Improvement optimizer for positive feedback acceleration
try:
    from app.training.improvement_optimizer import (
        get_improvement_optimizer,
        should_fast_track_training,
    )
    HAS_IMPROVEMENT_OPTIMIZER = True
except ImportError:
    HAS_IMPROVEMENT_OPTIMIZER = False
    get_improvement_optimizer = None
    should_fast_track_training = None

# Resource optimizer for utilization-based scheduling
try:
    from app.coordination.resource_optimizer import get_utilization_status
    HAS_RESOURCE_OPTIMIZER = True
except ImportError:
    HAS_RESOURCE_OPTIMIZER = False
    get_utilization_status = None

# Coordination utilities
try:
    from app.coordination import estimate_task_duration
    HAS_COORDINATION = True
except ImportError:
    HAS_COORDINATION = False
    estimate_task_duration = None

# Pre-spawn health checking
try:
    from app.distributed.pre_spawn_health import gate_on_cluster_health
    HAS_PRE_SPAWN_HEALTH = True
except ImportError:
    HAS_PRE_SPAWN_HEALTH = False
    gate_on_cluster_health = None

# Feedback accelerator for momentum-based training
try:
    from app.integration.feedback_accelerator import (
        get_feedback_accelerator,
        record_games_generated,
    )
    HAS_FEEDBACK_ACCELERATOR = True
except ImportError:
    HAS_FEEDBACK_ACCELERATOR = False
    get_feedback_accelerator = None
    record_games_generated = None

# Value calibration
try:
    from app.training.value_calibration import CalibrationTracker
    HAS_VALUE_CALIBRATION = True
except ImportError:
    HAS_VALUE_CALIBRATION = False
    CalibrationTracker = None

# Temperature scheduling
try:
    from app.ai.temperature_schedule import create_temp_scheduler
    HAS_TEMPERATURE_SCHEDULING = True
except ImportError:
    HAS_TEMPERATURE_SCHEDULING = False
    create_temp_scheduler = None


class TrainingScheduler:
    """Schedules and manages training runs with cluster-wide coordination."""

    def __init__(
        self,
        config: TrainingConfig,
        state: "UnifiedLoopState",
        event_bus: "EventBus",
        feedback_config: Optional[FeedbackConfig] = None,
        feedback: Optional["PipelineFeedbackController"] = None,
        config_priority: Optional["ConfigPriorityQueue"] = None
    ):
        # Import ConfigPriorityQueue at runtime to avoid circular imports
        from scripts.unified_ai_loop import ConfigPriorityQueue as CPQ

        self.config = config
        self.state = state
        self.event_bus = event_bus
        self.feedback_config = feedback_config or FeedbackConfig()
        self.feedback = feedback
        self.config_priority = config_priority or CPQ()
        self._training_process: Optional[asyncio.subprocess.Process] = None
        # Dynamic threshold tracking (for promotion velocity calculation)
        self._promotion_history: List[float] = []  # Timestamps of recent promotions
        self._training_history: List[float] = []   # Timestamps of recent training runs
        # Cluster-wide training coordination
        self._training_lock_fd: Optional[int] = None
        self._training_lock_path: Optional[Path] = None
        # Calibration tracking (per config)
        self._calibration_trackers: Dict[str, Any] = {}
        if HAS_VALUE_CALIBRATION:
            for config_key in state.configs:
                self._calibration_trackers[config_key] = CalibrationTracker(window_size=1000)
        # Temperature scheduler for self-play exploration
        self._temp_scheduler: Optional[Any] = None
        if HAS_TEMPERATURE_SCHEDULING:
            self._temp_scheduler = create_temp_scheduler(state.current_temperature_preset or "default")

    def _get_dynamic_threshold(self, config_key: str) -> int:
        """Calculate dynamic training threshold based on promotion velocity."""
        base_threshold = self.config.trigger_threshold_games  # Default: 500

        now = time.time()
        recent_promotions = [t for t in self._promotion_history if now - t < 3600 * 6]
        promotions_per_hour = len(recent_promotions) / 6.0 if recent_promotions else 0

        recent_training = [t for t in self._training_history if now - t < 3600 * 6]
        training_per_hour = len(recent_training) / 6.0 if recent_training else 0

        adjustment = 1.0

        if promotions_per_hour > 0.5:
            adjustment *= 0.7
        elif promotions_per_hour < 0.1:
            adjustment *= 0.8
        else:
            adjustment *= 0.9

        if training_per_hour > 2 and promotions_per_hour < 0.2:
            adjustment *= 1.5
            print(f"[Training] Dynamic: High training ({training_per_hour:.1f}/hr) but low promotion ({promotions_per_hour:.1f}/hr) - increasing threshold")

        config_state = self.state.configs.get(config_key)
        if config_state:
            time_since_promotion = now - config_state.last_promotion_time
            if time_since_promotion > 3600 * 2:
                adjustment *= 0.75
                print(f"[Training] Dynamic: {time_since_promotion/3600:.1f}h since last promotion for {config_key} - lowering threshold")

        dynamic_threshold = int(base_threshold * adjustment)
        min_threshold = base_threshold // 4
        max_threshold = base_threshold * 2

        final_threshold = max(min_threshold, min(max_threshold, dynamic_threshold))

        # Improvement optimizer acceleration
        if HAS_IMPROVEMENT_OPTIMIZER:
            try:
                optimizer = get_improvement_optimizer()
                optimizer_threshold = optimizer.get_dynamic_threshold(config_key)
                metrics = optimizer.get_improvement_metrics()

                if optimizer_threshold < final_threshold:
                    streak_info = f"streak={metrics.get('consecutive_promotions', 0)}"
                    print(f"[ImprovementOptimizer] Accelerating threshold for {config_key}: "
                          f"{final_threshold} → {optimizer_threshold} ({streak_info})")
                    final_threshold = optimizer_threshold

                if should_fast_track_training(config_key):
                    fast_threshold = max(min_threshold, final_threshold * 8 // 10)
                    if fast_threshold < final_threshold:
                        print(f"[ImprovementOptimizer] Fast-tracking {config_key}: {final_threshold} → {fast_threshold}")
                        final_threshold = fast_threshold
            except Exception as e:
                if self.config.verbose:
                    print(f"[ImprovementOptimizer] Error getting threshold: {e}")

        # Utilization-based adjustment
        if HAS_RESOURCE_OPTIMIZER and get_utilization_status is not None:
            try:
                util_status = get_utilization_status()
                cpu_util = util_status.get('cpu_util', 70)
                gpu_util = util_status.get('gpu_util', 70)
                avg_util = (cpu_util + gpu_util) / 2 if gpu_util > 0 else cpu_util
                if avg_util < 50:
                    final_threshold = max(min_threshold, final_threshold * 6 // 10)
                elif avg_util < 60:
                    final_threshold = max(min_threshold, final_threshold * 8 // 10)
                elif avg_util > 85:
                    final_threshold = min(max_threshold, final_threshold * 12 // 10)
            except Exception:
                pass

        # Underrepresented config priority
        if hasattr(self.config_priority, '_trained_model_counts'):
            self.config_priority._update_trained_model_counts()
            model_count = self.config_priority._trained_model_counts.get(config_key, 0)

            if model_count == 0:
                bootstrap_threshold = 50
                if final_threshold > bootstrap_threshold:
                    print(f"[Training] BOOTSTRAP: {config_key} has 0 trained models - threshold {final_threshold} → {bootstrap_threshold}")
                    final_threshold = bootstrap_threshold
            elif model_count == 1:
                single_model_threshold = min_threshold
                if final_threshold > single_model_threshold:
                    print(f"[Training] UNDERREPRESENTED: {config_key} has 1 model - threshold {final_threshold} → {single_model_threshold}")
                    final_threshold = single_model_threshold
            elif model_count <= 3:
                catchup_threshold = min_threshold * 3 // 2
                if final_threshold > catchup_threshold:
                    print(f"[Training] CATCHUP: {config_key} has {model_count} models - threshold {final_threshold} → {catchup_threshold}")
                    final_threshold = catchup_threshold

        if final_threshold != base_threshold:
            print(f"[Training] Dynamic threshold for {config_key}: {final_threshold} (base: {base_threshold}, adj: {adjustment:.2f})")

        return final_threshold

    def record_promotion(self):
        """Record a successful promotion for velocity tracking."""
        now = time.time()
        self._promotion_history.append(now)
        self._promotion_history = [t for t in self._promotion_history if now - t < 86400]

    def record_training_start(self):
        """Record a training run start for velocity tracking."""
        now = time.time()
        self._training_history.append(now)
        self._training_history = [t for t in self._training_history if now - t < 86400]

    def set_feedback_controller(self, feedback: "PipelineFeedbackController"):
        """Set the feedback controller (called after initialization)."""
        self.feedback = feedback

    def _acquire_training_lock(self) -> bool:
        """Acquire cluster-wide training lock using file locking."""
        import fcntl
        import socket

        lock_dir = AI_SERVICE_ROOT / "data" / "coordination"
        lock_dir.mkdir(parents=True, exist_ok=True)

        hostname = socket.gethostname()
        self._training_lock_path = lock_dir / f"training.{hostname}.lock"

        try:
            self._training_lock_fd = os.open(
                str(self._training_lock_path),
                os.O_RDWR | os.O_CREAT
            )
            fcntl.flock(self._training_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(self._training_lock_fd, 0)
            os.lseek(self._training_lock_fd, 0, os.SEEK_SET)
            os.write(self._training_lock_fd, f"{os.getpid()}\n".encode())
            print(f"[Training] Acquired cluster-wide training lock on {hostname}")
            return True
        except (OSError, BlockingIOError) as e:
            if self._training_lock_fd is not None:
                os.close(self._training_lock_fd)
                self._training_lock_fd = None
            print(f"[Training] Lock acquisition failed: {e}")
            return False

    def _release_training_lock(self):
        """Release the cluster-wide training lock."""
        import fcntl

        if self._training_lock_fd is not None:
            try:
                fcntl.flock(self._training_lock_fd, fcntl.LOCK_UN)
                os.close(self._training_lock_fd)
                print("[Training] Released cluster-wide training lock")
            except Exception as e:
                print(f"[Training] Error releasing lock: {e}")
            finally:
                self._training_lock_fd = None
            if self._training_lock_path and self._training_lock_path.exists():
                try:
                    self._training_lock_path.unlink()
                except Exception:
                    pass

    def is_training_locked_elsewhere(self) -> bool:
        """Check if training is running on another host in the cluster."""
        import socket

        lock_dir = AI_SERVICE_ROOT / "data" / "coordination"
        if not lock_dir.exists():
            return False

        hostname = socket.gethostname()
        for lock_file in lock_dir.glob("training.*.lock"):
            if f"training.{hostname}.lock" in lock_file.name:
                continue
            try:
                if lock_file.stat().st_size > 0:
                    age = time.time() - lock_file.stat().st_mtime
                    if age < 3600:
                        other_host = lock_file.name.replace("training.", "").replace(".lock", "")
                        print(f"[Training] Training lock held by {other_host}")
                        return True
            except Exception:
                continue
        return False

    def should_trigger_training(self) -> Optional[str]:
        """Check if training should be triggered. Returns config key or None."""
        if self.state.training_in_progress:
            return None

        if self.is_training_locked_elsewhere():
            return None

        # Duration-aware scheduling
        if HAS_COORDINATION:
            import socket
            from app.coordination.duration_scheduler import get_scheduler
            node_id = socket.gethostname()
            scheduler = get_scheduler()
            can_schedule, schedule_reason = scheduler.can_schedule_now(
                "training", node_id, avoid_peak_hours=False
            )
            if not can_schedule:
                if self.config.verbose:
                    print(f"[Training] Deferred by duration scheduler: {schedule_reason}")
                return None

        # Health-aware training
        if HAS_RESOURCE_OPTIMIZER and get_utilization_status is not None:
            try:
                util_status = get_utilization_status()
                gpu_util = util_status.get('gpu_util', 70)
                if gpu_util > 85:
                    return None
            except Exception:
                pass

        # Cluster health gate
        if HAS_PRE_SPAWN_HEALTH and gate_on_cluster_health is not None:
            try:
                can_proceed, health_msg = gate_on_cluster_health(
                    "training", min_healthy=2, min_healthy_ratio=0.4
                )
                if not can_proceed:
                    if self.config.verbose:
                        print(f"[Training] Deferred: {health_msg}")
                    return None
            except Exception:
                pass

        now = time.time()

        # Import ConfigPriorityQueue at runtime
        from scripts.unified_ai_loop import ConfigPriorityQueue
        priority_queue = ConfigPriorityQueue()
        prioritized_configs = priority_queue.get_prioritized_configs(self.state.configs)

        for config_key, priority_score in prioritized_configs:
            config_state = self.state.configs[config_key]
            if now - config_state.last_training_time < self.config.min_interval_seconds:
                continue

            # Momentum-based acceleration
            if HAS_FEEDBACK_ACCELERATOR:
                try:
                    decision = get_feedback_accelerator().get_training_decision(config_key)
                    if decision.should_train:
                        record_games_generated(config_key, config_state.games_since_training)
                        intensity_str = decision.intensity.value if decision.intensity else "normal"
                        momentum_str = decision.momentum.value if decision.momentum else "stable"
                        print(f"[Training] Trigger: momentum-based acceleration for {config_key} "
                              f"(intensity={intensity_str}, momentum={momentum_str})")
                        return config_key
                except Exception:
                    pass

            # Dynamic game count threshold
            dynamic_threshold = self._get_dynamic_threshold(config_key)
            if config_state.games_since_training >= dynamic_threshold:
                print(f"[Training] Trigger: game threshold reached for {config_key} "
                      f"({config_state.games_since_training} >= {dynamic_threshold} games)")
                return config_key

            # Elo plateau detection
            if self.feedback:
                if self.feedback.eval_analyzer.is_plateau(
                    config_key,
                    min_improvement=self.feedback_config.elo_plateau_threshold,
                    lookback=self.feedback_config.elo_plateau_lookback
                ):
                    if config_state.games_since_training >= self.config.trigger_threshold_games // 4:
                        print(f"[Training] Trigger: Elo plateau detected for {config_key}")
                        return config_key

            # Win rate degradation
            if self.feedback:
                weak_configs = self.feedback.eval_analyzer.get_weak_configs(
                    threshold=self.feedback_config.win_rate_degradation_threshold
                )
                if config_key in weak_configs:
                    if config_state.games_since_training >= self.config.trigger_threshold_games // 4:
                        print(f"[Training] Trigger: Win rate degradation for {config_key}")
                        return config_key

        return None

    async def start_training(self, config_key: str) -> bool:
        """Start a training run for the given configuration."""
        if DISABLE_LOCAL_TASKS:
            print(f"[Training] Skipping local training (RINGRIFT_DISABLE_LOCAL_TASKS=true)")
            return False

        if self.state.training_in_progress:
            return False

        self.record_training_start()

        # Data quality gate
        if self.feedback:
            parity_failure_rate = self.feedback.data_monitor.get_parity_failure_rate()
            if parity_failure_rate > self.feedback_config.max_parity_failure_rate:
                print(f"[Training] BLOCKED by data quality gate: parity failure rate {parity_failure_rate:.1%}")
                if HAS_PROMETHEUS:
                    DATA_QUALITY_BLOCKED_TRAINING.labels(reason="parity_failure").inc()
                return False

            if self.feedback.should_quarantine_data():
                print(f"[Training] BLOCKED by data quality gate: data quarantined")
                if HAS_PROMETHEUS:
                    DATA_QUALITY_BLOCKED_TRAINING.labels(reason="quarantined").inc()
                return False

            if self.feedback.state.data_quality_score < self.feedback_config.min_data_quality_score:
                print(f"[Training] BLOCKED by data quality gate: score {self.feedback.state.data_quality_score:.2f}")
                if HAS_PROMETHEUS:
                    DATA_QUALITY_BLOCKED_TRAINING.labels(reason="low_score").inc()
                return False

        parts = config_key.rsplit("_", 1)
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))

        if not self._acquire_training_lock():
            print(f"[Training] Could not acquire cluster-wide lock for {config_key}")
            return False

        await self.event_bus.publish(DataEvent(
            event_type=DataEventType.TRAINING_STARTED,
            payload={"config": config_key}
        ))

        try:
            self.state.training_in_progress = True
            self.state.training_config = config_key
            self.state.training_started_at = time.time()

            if HAS_COORDINATION and estimate_task_duration:
                est_duration = estimate_task_duration("training", config=config_key)
                eta_time = datetime.fromtimestamp(time.time() + est_duration).strftime("%H:%M:%S")
                print(f"[Training] Estimated duration: {est_duration/3600:.1f}h (ETA: {eta_time})")

            model_version = "v3"

            games_dir = AI_SERVICE_ROOT / "data" / "games"
            synced_dir = games_dir / "synced"
            gpu_selfplay_dir = games_dir / "gpu_selfplay"

            jsonl_path = gpu_selfplay_dir / config_key / "games.jsonl"
            has_jsonl_data = jsonl_path.exists() and jsonl_path.stat().st_size > 0

            game_dbs = list(games_dir.glob("*.db"))
            if synced_dir.exists():
                game_dbs.extend(synced_dir.rglob("*.db"))

            if not game_dbs and not has_jsonl_data:
                print(f"[Training] No game data found")
                self.state.training_in_progress = False
                self._release_training_lock()
                return False

            # Auto-consolidate databases
            consolidated_db = games_dir / "consolidated_training_v2.db"
            consolidation_max_age = 6 * 3600

            should_consolidate = False
            if has_jsonl_data:
                print(f"[Training] Using JSONL data, skipping DB consolidation")
            elif not consolidated_db.exists():
                should_consolidate = True
            elif time.time() - consolidated_db.stat().st_mtime > consolidation_max_age:
                should_consolidate = True

            if should_consolidate:
                merge_dbs = []
                for db_path in game_dbs:
                    db_str = str(db_path)
                    if any(skip in db_str for skip in ['quarantine', 'corrupted', 'backup', 'consolidated']):
                        continue
                    if 'selfplay' in db_path.name or 'merged' in db_path.name or 'training' in db_path.name:
                        try:
                            conn = sqlite3.connect(db_path)
                            cursor = conn.cursor()
                            cursor.execute("SELECT COUNT(*) FROM games")
                            count = cursor.fetchone()[0]
                            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'")
                            has_moves = cursor.fetchone() is not None
                            conn.close()
                            if count > 0 and has_moves:
                                merge_dbs.append(db_path)
                        except Exception:
                            pass

                if len(merge_dbs) > 1:
                    print(f"[Training] Consolidating {len(merge_dbs)} databases...")
                    merge_cmd = [
                        sys.executable,
                        str(AI_SERVICE_ROOT / "scripts" / "merge_game_dbs.py"),
                        "--output", str(consolidated_db),
                        "--dedupe-by-game-id",
                    ]
                    for db in merge_dbs:
                        merge_cmd.extend(["--db", str(db)])

                    try:
                        merge_process = await asyncio.create_subprocess_exec(
                            *merge_cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            cwd=AI_SERVICE_ROOT,
                        )
                        await asyncio.wait_for(merge_process.communicate(), timeout=1800)
                    except Exception:
                        pass

            if consolidated_db.exists():
                largest_db = consolidated_db
            else:
                largest_db = max(game_dbs, key=lambda p: p.stat().st_size)

            training_dir = AI_SERVICE_ROOT / "data" / "training"
            training_dir.mkdir(parents=True, exist_ok=True)
            data_path = training_dir / f"unified_{config_key}.npz"

            export_max_age = 6 * 3600
            skip_export = False
            if data_path.exists():
                npz_age = time.time() - data_path.stat().st_mtime
                if npz_age < export_max_age:
                    skip_export = True

            encoder_version = "v2" if board_type == "hexagonal" else "default"

            if not skip_export:
                if has_jsonl_data:
                    export_cmd = [
                        sys.executable,
                        str(AI_SERVICE_ROOT / "scripts" / "jsonl_to_npz.py"),
                        "--input", str(jsonl_path),
                        "--output", str(data_path),
                        "--board-type", board_type,
                        "--num-players", str(num_players),
                        "--gpu-selfplay",
                        "--max-games", "10000",
                    ]
                else:
                    export_cmd = [
                        sys.executable,
                        str(AI_SERVICE_ROOT / self.config.export_script),
                        "--db", str(largest_db),
                        "--output", str(data_path),
                        "--board-type", board_type,
                        "--num-players", str(num_players),
                        "--sample-every", "2",
                        "--require-completed",
                        "--min-moves", "10",
                        "--exclude-recovery",
                    ]
                if encoder_version != "default":
                    export_cmd.extend(["--encoder-version", encoder_version])

                export_env = os.environ.copy()
                export_env["PYTHONPATH"] = str(AI_SERVICE_ROOT)
                export_process = await asyncio.create_subprocess_exec(
                    *export_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=AI_SERVICE_ROOT,
                    env=export_env,
                )
                stdout, stderr = await export_process.communicate()

                if export_process.returncode != 0:
                    self.state.training_in_progress = False
                    self._release_training_lock()
                    return False

            timestamp = int(time.time())
            model_id = f"{config_key}_v3_{timestamp}"
            run_dir = AI_SERVICE_ROOT / "logs" / "unified_training" / model_id

            base_epochs = 100
            if self.feedback:
                epochs_multiplier = self.feedback.get_epochs_multiplier()
                epochs = max(50, int(base_epochs * epochs_multiplier))
            else:
                epochs = base_epochs

            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / self.config.nn_training_script),
                "--board", board_type,
                "--num-players", str(num_players),
                "--data-path", str(data_path),
                "--run-dir", str(run_dir),
                "--model-id", model_id,
                "--model-version", model_version,
                "--epochs", str(epochs),
            ]

            print(f"[Training] Starting training for {model_id}...")
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
            self._release_training_lock()
            return False

    async def check_training_status(self) -> Optional[Dict[str, Any]]:
        """Check if current training has completed."""
        if not self.state.training_in_progress or not self._training_process:
            return None

        if self._training_process.returncode is not None:
            stdout, stderr = await self._training_process.communicate()

            success = self._training_process.returncode == 0
            config_key = self.state.training_config

            self.state.training_in_progress = False
            self.state.training_config = ""
            self.state.total_training_runs += 1
            self._training_process = None
            self._release_training_lock()

            if config_key in self.state.configs:
                self.state.configs[config_key].last_training_time = time.time()
                self.state.configs[config_key].games_since_training = 0

            result = {
                "config": config_key,
                "success": success,
                "duration": time.time() - self.state.training_started_at,
            }

            if success and HAS_VALUE_CALIBRATION:
                calibration_report = await self._run_calibration_analysis(config_key)
                if calibration_report:
                    result["calibration"] = calibration_report

            if HAS_IMPROVEMENT_OPTIMIZER and success:
                try:
                    optimizer = get_improvement_optimizer()
                    calibration_ece = result.get("calibration", {}).get("ece")
                    optimizer.record_training_complete(
                        config_key=config_key,
                        duration_seconds=result["duration"],
                        val_loss=0.0,
                        calibration_ece=calibration_ece,
                    )
                except Exception:
                    pass

            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.TRAINING_COMPLETED,
                payload=result
            ))

            return result

        return None

    async def _run_calibration_analysis(self, config_key: str) -> Optional[Dict[str, Any]]:
        """Run value calibration analysis on recent training data."""
        if not HAS_VALUE_CALIBRATION:
            return None

        try:
            if config_key not in self._calibration_trackers:
                self._calibration_trackers[config_key] = CalibrationTracker(window_size=1000)

            tracker = self._calibration_trackers[config_key]
            report = tracker.compute_current_calibration()
            if report is None:
                return None

            report_dict = report.to_dict()
            self.state.calibration_reports[config_key] = report_dict
            self.state.last_calibration_time = time.time()

            print(f"[Calibration] {config_key}: ECE={report.ece:.4f}, MCE={report.mce:.4f}")

            return report_dict

        except Exception as e:
            print(f"[Calibration] Error analyzing {config_key}: {e}")
            return None

    def get_temperature_for_move(self, move_number: int, game_state: Optional[Any] = None) -> float:
        """Get exploration temperature for a given move in self-play."""
        if self._temp_scheduler is None:
            return 1.0
        return self._temp_scheduler.get_temperature(move_number, game_state)

    def update_training_progress(self, progress: float):
        """Update training progress for curriculum-based temperature scheduling."""
        if self._temp_scheduler is not None:
            self._temp_scheduler.set_training_progress(progress)

    async def request_urgent_training(self, configs: List[str], reason: str) -> bool:
        """Request urgent training for specified configs due to feedback signal."""
        if self.state.training_in_progress:
            print(f"[Training] Urgent training request deferred: training already in progress")
            return False

        now = time.time()
        for config_key in configs:
            if config_key not in self.state.configs:
                continue

            config_state = self.state.configs[config_key]
            urgent_cooldown = self.config.min_interval_seconds / 2
            if now - config_state.last_training_time < urgent_cooldown:
                continue

            print(f"[Training] URGENT TRAINING triggered for {config_key}: {reason}")
            started = await self.start_training(config_key)
            if started:
                return True

        print(f"[Training] Urgent training request could not be fulfilled for configs: {configs}")
        return False
