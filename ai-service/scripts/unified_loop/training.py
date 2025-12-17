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

# Simplified training triggers (2024-12)
try:
    from app.training.training_triggers import (
        TrainingTriggers,
        TriggerConfig,
        TriggerDecision,
    )
    HAS_SIMPLIFIED_TRIGGERS = True
except ImportError:
    HAS_SIMPLIFIED_TRIGGERS = False
    TrainingTriggers = None
    TriggerConfig = None
    TriggerDecision = None

# Curriculum feedback loop (2024-12)
try:
    from app.training.curriculum_feedback import (
        CurriculumFeedback,
        get_curriculum_feedback,
        record_selfplay_game,
        get_curriculum_weights,
    )
    HAS_CURRICULUM_FEEDBACK = True
except ImportError:
    HAS_CURRICULUM_FEEDBACK = False
    CurriculumFeedback = None
    get_curriculum_feedback = None
    record_selfplay_game = None
    get_curriculum_weights = None


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
        # Per-config training coordination (allows parallel training of different configs)
        self._training_locks: Dict[str, int] = {}  # config_key -> fd
        self._training_lock_paths: Dict[str, Path] = {}
        self._max_concurrent_training = 3  # Allow up to 3 parallel training runs
        # Calibration tracking (per config)
        self._calibration_trackers: Dict[str, Any] = {}
        if HAS_VALUE_CALIBRATION:
            for config_key in state.configs:
                self._calibration_trackers[config_key] = CalibrationTracker(window_size=1000)
        # Temperature scheduler for self-play exploration
        self._temp_scheduler: Optional[Any] = None
        if HAS_TEMPERATURE_SCHEDULING:
            self._temp_scheduler = create_temp_scheduler(state.current_temperature_preset or "default")

        # Simplified 3-signal trigger system (2024-12)
        self._simplified_triggers: Optional[Any] = None
        if HAS_SIMPLIFIED_TRIGGERS and getattr(config, 'use_simplified_triggers', True):
            trigger_cfg = TriggerConfig(
                freshness_threshold=config.trigger_threshold_games,
                staleness_hours=getattr(config, 'staleness_hours', 6.0),
                min_win_rate=getattr(config, 'min_win_rate_threshold', 0.45),
                min_interval_minutes=config.min_interval_seconds / 60,
                max_concurrent_training=self._max_concurrent_training,
                bootstrap_threshold=getattr(config, 'bootstrap_threshold', 50),
            )
            self._simplified_triggers = TrainingTriggers(trigger_cfg)
            print("[Training] Using simplified 3-signal trigger system")

        # Curriculum feedback loop (2024-12)
        self._curriculum_feedback: Optional[Any] = None
        if HAS_CURRICULUM_FEEDBACK:
            self._curriculum_feedback = get_curriculum_feedback()
            print("[Training] Using curriculum feedback loop for adaptive weights")

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

        # Phase 2.4: Win rate → training feedback
        # Adjust threshold based on evaluation win rates to avoid redundant training
        if config_state:
            win_rate = getattr(config_state, 'win_rate', 0.5)
            win_rate_trend = getattr(config_state, 'win_rate_trend', 0.0)
            consecutive_high = getattr(config_state, 'consecutive_high_win_rate', 0)

            # High win rate (>70%) consistently → de-prioritize training (already strong)
            if consecutive_high >= 3 and win_rate > 0.7:
                skip_factor = 1.5 + (consecutive_high - 3) * 0.2  # Max ~2.5x
                skip_factor = min(skip_factor, 2.5)
                old_threshold = final_threshold
                final_threshold = min(max_threshold, int(final_threshold * skip_factor))
                if final_threshold > old_threshold:
                    print(f"[Training] WIN_RATE_SKIP: {config_key} has {win_rate:.1%} win rate "
                          f"(consecutive_high={consecutive_high}) - threshold {old_threshold} → {final_threshold}")

            # Low win rate (<50%) or declining → prioritize training urgently
            elif win_rate < 0.5 or win_rate_trend < -0.05:
                urgency_factor = 0.6 if win_rate < 0.4 else 0.8
                old_threshold = final_threshold
                final_threshold = max(min_threshold, int(final_threshold * urgency_factor))
                if final_threshold < old_threshold:
                    trend_str = f", trend={win_rate_trend:+.1%}" if win_rate_trend < -0.05 else ""
                    print(f"[Training] WIN_RATE_URGENT: {config_key} has {win_rate:.1%} win rate{trend_str} "
                          f"- threshold {old_threshold} → {final_threshold}")

            # Declining win rate but still decent → moderate priority increase
            elif win_rate_trend < -0.02 and win_rate < 0.65:
                old_threshold = final_threshold
                final_threshold = max(min_threshold, int(final_threshold * 0.9))
                if final_threshold < old_threshold:
                    print(f"[Training] WIN_RATE_DECLINING: {config_key} win rate declining ({win_rate_trend:+.1%}) "
                          f"- threshold {old_threshold} → {final_threshold}")

        # Curriculum feedback weight adjustment (2024-12)
        # Applies weight from recent selfplay performance (0.5 to 2.0)
        curriculum_weight = self.get_curriculum_weight(config_key)
        if curriculum_weight != 1.0:
            old_threshold = final_threshold
            # Weight > 1.0 = needs more training (lower threshold)
            # Weight < 1.0 = already strong (higher threshold)
            final_threshold = int(final_threshold / curriculum_weight)
            final_threshold = max(min_threshold, min(max_threshold, final_threshold))
            if final_threshold != old_threshold:
                print(f"[Training] CURRICULUM_WEIGHT: {config_key} weight={curriculum_weight:.2f} "
                      f"- threshold {old_threshold} → {final_threshold}")

        if final_threshold != base_threshold:
            print(f"[Training] Dynamic threshold for {config_key}: {final_threshold} (base: {base_threshold}, adj: {adjustment:.2f})")

        return final_threshold

    def should_train_simplified(
        self,
        config_key: str,
        games_since_training: int,
        win_rate: float = 0.5,
        model_count: int = 1,
    ) -> tuple[bool, str, float]:
        """Check if training should run using simplified 3-signal system.

        Args:
            config_key: Config identifier (e.g., "square8_2p")
            games_since_training: Number of new games since last training
            win_rate: Current win rate (0.0 to 1.0)
            model_count: Number of trained models for this config

        Returns:
            Tuple of (should_train, reason, priority)
        """
        if self._simplified_triggers is None:
            # Fall back to legacy system
            threshold = self._get_dynamic_threshold(config_key)
            should = games_since_training >= threshold
            return should, f"games={games_since_training} >= threshold={threshold}", float(games_since_training)

        # Update state in simplified triggers
        self._simplified_triggers.update_config_state(
            config_key,
            games_count=games_since_training + self._simplified_triggers.get_config_state(config_key).last_training_games,
            win_rate=win_rate,
            model_count=model_count,
        )

        # Get decision from simplified system
        decision = self._simplified_triggers.should_train(config_key)
        return decision.should_train, decision.reason, decision.priority

    def record_training_complete_simplified(self, config_key: str, games_at_training: int) -> None:
        """Record training completion in simplified trigger system."""
        if self._simplified_triggers is not None:
            self._simplified_triggers.record_training_complete(config_key, games_at_training)

    def get_next_training_config_simplified(self) -> Optional[str]:
        """Get the highest priority config that should train using simplified system."""
        if self._simplified_triggers is None:
            return None

        decision = self._simplified_triggers.get_next_training_config()
        if decision is not None:
            print(f"[Training] Simplified trigger: {decision.config_key} (priority={decision.priority:.2f}, reason={decision.reason})")
            return decision.config_key
        return None

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

    def record_selfplay_result(
        self,
        config_key: str,
        winner: int,
        model_elo: float = 1500.0,
    ) -> None:
        """Record a selfplay game result for curriculum feedback.

        Args:
            config_key: Config identifier (e.g., "square8_2p")
            winner: 1 = model won, -1 = model lost, 0 = draw
            model_elo: Current model Elo rating
        """
        if self._curriculum_feedback is not None:
            self._curriculum_feedback.record_game(
                config_key, winner, model_elo, opponent_type="selfplay"
            )

    def get_curriculum_weights(self) -> Dict[str, float]:
        """Get curriculum weights for all configs.

        Returns:
            Dict mapping config_key → weight (0.5 to 2.0)
            Higher weight = more training attention needed
        """
        if self._curriculum_feedback is not None:
            return self._curriculum_feedback.get_curriculum_weights()
        return {}

    def get_curriculum_weight(self, config_key: str) -> float:
        """Get curriculum weight for a specific config.

        Returns:
            Weight between 0.5 (de-prioritize) and 2.0 (high priority)
        """
        weights = self.get_curriculum_weights()
        return weights.get(config_key, 1.0)

    def record_training_complete_for_curriculum(self, config_key: str) -> None:
        """Record training completion for curriculum feedback."""
        if self._curriculum_feedback is not None:
            self._curriculum_feedback.record_training(config_key)

    def set_feedback_controller(self, feedback: "PipelineFeedbackController"):
        """Set the feedback controller (called after initialization)."""
        self.feedback = feedback

    def _acquire_training_lock(self, config_key: str = None) -> bool:
        """Acquire per-config training lock using file locking.

        Allows up to _max_concurrent_training parallel training runs.
        Different configs can train simultaneously (e.g., square8 + hexagonal).
        """
        import fcntl
        import socket

        # Check if we've hit the concurrent training limit
        active_locks = len(self._training_locks)
        if active_locks >= self._max_concurrent_training:
            print(f"[Training] Max concurrent training ({self._max_concurrent_training}) reached")
            return False

        lock_dir = AI_SERVICE_ROOT / "data" / "coordination"
        lock_dir.mkdir(parents=True, exist_ok=True)

        hostname = socket.gethostname()
        lock_name = f"training.{config_key or 'global'}.{hostname}.lock"
        lock_path = lock_dir / lock_name

        try:
            fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, f"{os.getpid()}\n".encode())

            self._training_locks[config_key or 'global'] = fd
            self._training_lock_paths[config_key or 'global'] = lock_path
            print(f"[Training] Acquired lock for {config_key or 'global'} on {hostname} ({active_locks + 1}/{self._max_concurrent_training} slots used)")
            return True
        except (OSError, BlockingIOError) as e:
            try:
                os.close(fd)
            except:
                pass
            print(f"[Training] Lock acquisition failed for {config_key}: {e}")
            return False

    def _release_training_lock(self, config_key: str = None):
        """Release the per-config training lock."""
        import fcntl

        key = config_key or 'global'
        fd = self._training_locks.get(key)
        lock_path = self._training_lock_paths.get(key)

        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
                print(f"[Training] Released lock for {key}")
            except Exception as e:
                print(f"[Training] Error releasing lock for {key}: {e}")
            finally:
                self._training_locks.pop(key, None)
                self._training_lock_paths.pop(key, None)
            if lock_path and lock_path.exists():
                try:
                    lock_path.unlink()
                except Exception:
                    pass

    def is_training_locked_elsewhere(self, config_key: str = None) -> bool:
        """Check if training is running on another host for a specific config."""
        import socket

        lock_dir = AI_SERVICE_ROOT / "data" / "coordination"
        if not lock_dir.exists():
            return False

        hostname = socket.gethostname()
        # Look for locks matching this config (or any lock if config_key is None)
        pattern = f"training.{config_key}.*" if config_key else "training.*"

        for lock_file in lock_dir.glob(pattern + ".lock"):
            # Skip our own locks
            if hostname in lock_file.name:
                continue
            try:
                if lock_file.stat().st_size > 0:
                    age = time.time() - lock_file.stat().st_mtime
                    if age < 3600:
                        parts = lock_file.name.replace("training.", "").replace(".lock", "").split(".")
                        other_config = parts[0] if len(parts) > 1 else "unknown"
                        other_host = parts[-1] if len(parts) > 1 else parts[0]
                        print(f"[Training] Config {other_config} locked by {other_host}")
                        return True
            except Exception:
                continue
        return False

    def count_active_training_runs(self) -> int:
        """Count how many training runs are active across the cluster."""
        lock_dir = AI_SERVICE_ROOT / "data" / "coordination"
        if not lock_dir.exists():
            return 0

        count = 0
        for lock_file in lock_dir.glob("training.*.lock"):
            try:
                if lock_file.stat().st_size > 0:
                    age = time.time() - lock_file.stat().st_mtime
                    if age < 3600:
                        count += 1
            except Exception:
                continue
        return count

    def should_trigger_training(self) -> Optional[str]:
        """Check if training should be triggered. Returns config key or None.

        Supports parallel training: different configs can train simultaneously,
        up to _max_concurrent_training total runs.
        """
        # Check global training_in_progress flag (legacy check for backwards compatibility)
        if self.state.training_in_progress:
            return None

        # Check concurrent training limit (cluster-wide)
        active_runs = self.count_active_training_runs()
        if active_runs >= self._max_concurrent_training:
            if self.config.verbose:
                print(f"[Training] Max concurrent training ({self._max_concurrent_training}) reached cluster-wide")
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

            # Skip configs already training (per-config lock check)
            if self.is_training_locked_elsewhere(config_key):
                continue
            if config_key in self._training_locks:
                continue  # Already training locally

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

            # Phase 2.5: Incremental DB consolidation
            # Track which DBs have been merged to avoid redundant work
            consolidated_db = games_dir / "consolidated_training_v2.db"
            consolidation_state_file = games_dir / ".consolidation_state.json"

            should_consolidate = False
            if has_jsonl_data:
                print(f"[Training] Using JSONL data, skipping DB consolidation")
            elif not consolidated_db.exists():
                should_consolidate = True
            else:
                # Check if any source DB has changed since last consolidation
                last_state = {}
                if consolidation_state_file.exists():
                    try:
                        import json
                        with open(consolidation_state_file) as f:
                            last_state = json.load(f)
                    except Exception:
                        pass

                # Find DBs that need merging
                merge_dbs = []
                changed_dbs = []
                for db_path in game_dbs:
                    db_str = str(db_path)
                    if any(skip in db_str for skip in ['quarantine', 'corrupted', 'backup', 'consolidated']):
                        continue
                    if 'selfplay' in db_path.name or 'merged' in db_path.name or 'training' in db_path.name:
                        try:
                            current_mtime = db_path.stat().st_mtime
                            current_size = db_path.stat().st_size
                            db_key = str(db_path)
                            last_mtime = last_state.get(db_key, {}).get('mtime', 0)
                            last_size = last_state.get(db_key, {}).get('size', 0)

                            conn = sqlite3.connect(db_path)
                            cursor = conn.cursor()
                            cursor.execute("SELECT COUNT(*) FROM games")
                            count = cursor.fetchone()[0]
                            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'")
                            has_moves = cursor.fetchone() is not None
                            conn.close()

                            if count > 0 and has_moves:
                                merge_dbs.append(db_path)
                                # Check if this DB has new data
                                if current_mtime > last_mtime + 60 or current_size > last_size:
                                    changed_dbs.append(db_path.name)
                        except Exception:
                            pass

                # Only re-consolidate if there are changed DBs
                if changed_dbs:
                    should_consolidate = True
                    print(f"[Training] Incremental consolidation: {len(changed_dbs)} DBs with new data")

            if should_consolidate and not has_jsonl_data:
                # Re-scan for merge_dbs if we didn't already
                if 'merge_dbs' not in locals() or not merge_dbs:
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

                        # Save consolidation state for incremental tracking
                        import json
                        new_state = {}
                        for db_path in merge_dbs:
                            try:
                                new_state[str(db_path)] = {
                                    'mtime': db_path.stat().st_mtime,
                                    'size': db_path.stat().st_size,
                                }
                            except Exception:
                                pass
                        with open(consolidation_state_file, 'w') as f:
                            json.dump(new_state, f)
                        print(f"[Training] Consolidation state saved for {len(new_state)} DBs")
                    except Exception as e:
                        print(f"[Training] Consolidation error: {e}")

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

            # Get optimized training settings
            batch_size = self.config.batch_size or 256
            sampling_weights = self.config.sampling_weights or "victory_type"

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
                "--batch-size", str(batch_size),
                "--sampling-weights", sampling_weights,
                "--use-optimized-hyperparams",
                "--warmup-epochs", str(self.config.warmup_epochs),
            ]

            # Advanced training optimizations
            if self.config.use_spectral_norm:
                cmd.append("--spectral-norm")
            if self.config.use_lars:
                cmd.append("--lars")
            if self.config.use_cyclic_lr:
                cmd.extend(["--cyclic-lr", "--cyclic-lr-period", str(self.config.cyclic_lr_period)])
            if self.config.use_gradient_profiling:
                cmd.append("--gradient-profiling")
            if self.config.use_mixed_precision:
                cmd.extend(["--mixed-precision", "--amp-dtype", self.config.amp_dtype])
            if self.config.gradient_accumulation > 1:
                cmd.extend(["--gradient-accumulation", str(self.config.gradient_accumulation)])

            # Knowledge distillation
            if self.config.use_knowledge_distill and self.config.teacher_model_path:
                cmd.extend([
                    "--knowledge-distill",
                    "--teacher-model", self.config.teacher_model_path,
                    "--distill-alpha", str(self.config.distill_alpha),
                    "--distill-temperature", str(self.config.distill_temperature),
                ])

            
            # 2024-12 Advanced Training Improvements
            if getattr(self.config, 'use_value_whitening', True):
                cmd.extend([
                    "--value-whitening",
                    "--value-whitening-momentum", str(getattr(self.config, 'value_whitening_momentum', 0.99)),
                ])
            if getattr(self.config, 'use_ema', True):
                cmd.extend([
                    "--ema",
                    "--ema-decay", str(getattr(self.config, 'ema_decay', 0.999)),
                ])
            if getattr(self.config, 'use_stochastic_depth', True):
                cmd.extend([
                    "--stochastic-depth",
                    "--stochastic-depth-prob", str(getattr(self.config, 'stochastic_depth_prob', 0.1)),
                ])
            if getattr(self.config, 'use_adaptive_warmup', True):
                cmd.append("--adaptive-warmup")
            if getattr(self.config, 'use_hard_example_mining', True):
                cmd.extend([
                    "--hard-example-mining",
                    "--hard-example-top-k", str(getattr(self.config, 'hard_example_top_k', 0.3)),
                ])
            if getattr(self.config, 'use_dynamic_batch', False):
                cmd.extend([
                    "--dynamic-batch",
                    "--dynamic-batch-schedule", getattr(self.config, 'dynamic_batch_schedule', 'linear'),
                ])
            # Cross-board transfer learning
            if getattr(self.config, 'transfer_from_model', None):
                cmd.extend([
                    "--transfer-from", self.config.transfer_from_model,
                    "--transfer-freeze-epochs", str(getattr(self.config, 'transfer_freeze_epochs', 5)),
                ])

            print(f"[Training] Starting training for {model_id}...")

            # Also start NNUE policy training in parallel if configured
            nnue_policy_process = None
            if hasattr(self.config, 'nnue_policy_script'):
                nnue_policy_process = await self._start_nnue_policy_training(
                    board_type, num_players, largest_db, epochs, run_dir,
                    jsonl_path=jsonl_path if has_jsonl_data else None
                )
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

    async def _start_nnue_policy_training(
        self,
        board_type: str,
        num_players: int,
        db_path: Path,
        epochs: int,
        run_dir: Path,
        jsonl_path: Optional[Path] = None,
    ) -> Optional[asyncio.subprocess.Process]:
        """Start NNUE policy training with advanced optimizations.

        Uses the new training features: SWA, EMA, progressive batching,
        focal loss, auto-KL loss detection, and D6 hex augmentation.
        """
        try:
            nnue_run_dir = run_dir / "nnue_policy"
            nnue_run_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / self.config.nnue_policy_script),
                "--db", str(db_path),
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--epochs", str(epochs),
                "--run-dir", str(nnue_run_dir),
                "--batch-size", str(self.config.batch_size or 256),
                "--lr-scheduler", "cosine_warmup",
                "--grad-clip", "1.0",
            ]

            # Add JSONL source for MCTS policy data
            if jsonl_path and jsonl_path.exists():
                cmd.extend(["--jsonl", str(jsonl_path)])

            # Auto-KL loss detection (uses MCTS visit distributions when available)
            cmd.extend([
                "--auto-kl-loss",
                "--kl-min-coverage", "0.3",
                "--kl-min-samples", "50",
            ])

            # Mixed precision (AMP)
            if self.config.use_mixed_precision:
                cmd.append("--use-amp")

            # Stochastic Weight Averaging
            if getattr(self.config, 'use_swa', True):
                cmd.append("--use-swa")
                swa_start = int(epochs * getattr(self.config, 'swa_start_fraction', 0.75))
                cmd.extend(["--swa-start-epoch", str(swa_start)])

            # Exponential Moving Average
            if getattr(self.config, 'use_ema', True):
                cmd.append("--use-ema")
                cmd.extend(["--ema-decay", str(getattr(self.config, 'ema_decay', 0.999))])

            # Progressive batch sizing
            if getattr(self.config, 'use_progressive_batch', True):
                cmd.append("--progressive-batch")
                cmd.extend(["--min-batch-size", str(getattr(self.config, 'min_batch_size', 64))])
                cmd.extend(["--max-batch-size", str(getattr(self.config, 'max_batch_size', 512))])

            # Focal loss for hard sample mining
            focal_gamma = getattr(self.config, 'focal_gamma', 2.0)
            if focal_gamma > 0:
                cmd.extend(["--focal-gamma", str(focal_gamma)])

            # Label smoothing warmup
            warmup = getattr(self.config, 'label_smoothing_warmup', 5)
            if warmup > 0:
                cmd.extend(["--label-smoothing-warmup", str(warmup)])

            # Save learning curves
            cmd.append("--save-curves")

            jsonl_status = f"JSONL: {jsonl_path.name}" if jsonl_path and jsonl_path.exists() else "JSONL: None"
            print(f"[Training] Starting NNUE policy training with advanced optimizations...")
            print(f"[Training]   SWA: {getattr(self.config, 'use_swa', True)}, "
                  f"EMA: {getattr(self.config, 'use_ema', True)}, "
                  f"Progressive batch: {getattr(self.config, 'use_progressive_batch', True)}")
            print(f"[Training]   Auto-KL: True (min_coverage=30%, min_samples=50), {jsonl_status}")

            env = os.environ.copy()
            env["PYTHONPATH"] = str(AI_SERVICE_ROOT)

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
                env=env,
            )

            return process

        except Exception as e:
            print(f"[Training] Error starting NNUE policy training: {e}")
            return None

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
