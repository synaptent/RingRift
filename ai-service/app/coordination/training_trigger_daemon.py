"""Training Trigger Daemon - Automatic training decision logic (December 2025).

This daemon decides WHEN to trigger training automatically, eliminating
the human "train now" decision. It monitors multiple conditions to ensure
training starts at the optimal time.

Decision Conditions:
1. Data freshness - NPZ data < configured max age (default: 1 hour)
2. Training not active - No training already running for that config
3. Idle GPU available - At least one training GPU with < threshold utilization
4. Quality trajectory - Model still improving OR evaluation overdue
5. Minimum samples - Sufficient training samples available

Key features:
- Subscribes to NPZ_EXPORT_COMPLETE events for immediate trigger
- Periodic scan for training opportunities
- Tracks per-config training state
- Integrates with TrainingCoordinator to prevent duplicates
- Emits TRAINING_STARTED event when triggering

Usage:
    from app.coordination.training_trigger_daemon import TrainingTriggerDaemon

    daemon = TrainingTriggerDaemon()
    await daemon.start()

December 2025: Created as part of Phase 1 automation improvements.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.config.coordination_defaults import SyncDefaults
from app.coordination.handler_base import HandlerBase, HealthCheckResult

logger = logging.getLogger(__name__)

# Circuit breaker integration (Phase 4 - December 2025)
try:
    from app.distributed.circuit_breaker import get_training_breaker
    HAS_CIRCUIT_BREAKER = True
except ImportError:
    HAS_CIRCUIT_BREAKER = False
    get_training_breaker = None


@dataclass
class TrainingTriggerConfig:
    """Configuration for training trigger decisions."""

    enabled: bool = True
    # Data freshness
    max_data_age_hours: float = 1.0
    # December 2025: Use training_freshness to trigger sync when data is stale
    enforce_freshness_with_sync: bool = True  # If True, trigger sync instead of just rejecting
    freshness_sync_timeout_seconds: float = field(
        default_factory=lambda: SyncDefaults.SYNC_TIMEOUT  # Wait up to 5 min for sync
    )
    # December 29, 2025: Strict mode - fail immediately if data is stale (no sync attempt)
    # Useful for high-quality training where only fresh data should be used
    strict_freshness_mode: bool = field(
        default_factory=lambda: os.environ.get("RINGRIFT_STRICT_DATA_FRESHNESS", "").lower()
        in ("true", "1", "yes")
    )
    # Minimum samples to trigger training
    # December 29, 2025: Reduced from 10000 to 5000 for faster iteration cycles
    min_samples_threshold: int = 5000
    # Cooldown between training runs for same config
    # December 29, 2025: Reduced from 4.0 to 1.0 for faster iteration cycles
    training_cooldown_hours: float = 1.0
    # Maximum concurrent training jobs
    max_concurrent_training: int = 2
    # GPU utilization threshold for "idle"
    gpu_idle_threshold_percent: float = 20.0
    # Timeout for training subprocess (24 hours)
    training_timeout_seconds: int = 86400
    # Check interval for periodic scans
    # December 29, 2025: Reduced from 300s to 120s for faster detection
    scan_interval_seconds: int = 120  # 2 minutes
    # Training epochs
    default_epochs: int = 50
    default_batch_size: int = 512
    # Model version
    model_version: str = "v2"


@dataclass
class ConfigTrainingState:
    """Tracks training state for a single configuration."""

    config_key: str
    board_type: str
    num_players: int
    # Training status
    last_training_time: float = 0.0
    training_in_progress: bool = False
    training_pid: int | None = None
    # Data status
    last_npz_update: float = 0.0
    npz_sample_count: int = 0
    npz_path: str = ""
    # Quality tracking
    last_elo: float = 1500.0
    elo_trend: float = 0.0  # positive = improving
    # Training intensity (set by master_loop or FeedbackLoopController)
    training_intensity: str = "normal"  # hot_path, accelerated, normal, reduced, paused
    consecutive_failures: int = 0
    # December 29, 2025: Track model path for event emission
    _pending_model_path: str = ""  # Path where current training will save model


class TrainingTriggerDaemon(HandlerBase):
    """Daemon that automatically triggers training when conditions are met.

    Inherits from HandlerBase (December 2025 migration) providing:
    - Automatic event subscription via _get_event_subscriptions()
    - Singleton pattern via get_instance()
    - Standardized health check format
    - Lifecycle management (start/stop)
    """

    def __init__(self, config: TrainingTriggerConfig | None = None):
        self._daemon_config = config or TrainingTriggerConfig()
        super().__init__(
            name="training_trigger",
            config=self._daemon_config,
            cycle_interval=float(self._daemon_config.scan_interval_seconds),
        )
        self._training_states: dict[str, ConfigTrainingState] = {}
        self._training_semaphore = asyncio.Semaphore(self._daemon_config.max_concurrent_training)
        self._active_training_tasks: dict[str, asyncio.Task] = {}
        # Track whether we should skip due to coordinator mode
        self._coordinator_skip = False

    @property
    def config(self) -> TrainingTriggerConfig:
        """Get the daemon configuration."""
        return self._daemon_config

    def _get_event_subscriptions(self) -> dict[str, Callable]:
        """Return event subscriptions for HandlerBase.

        Subscribes to:
        - NPZ_EXPORT_COMPLETE: Immediate training trigger after export
        - TRAINING_COMPLETED: Track state after training finishes
        - TRAINING_THRESHOLD_REACHED: Honor master_loop-triggered requests
        - QUALITY_SCORE_UPDATED: Keep intensity in sync
        - TRAINING_BLOCKED_BY_QUALITY: Pause intensity
        - EVALUATION_COMPLETED: Gauntlet -> training feedback
        - TRAINING_INTENSITY_CHANGED: Updates from unified_feedback orchestrator
        """
        return {
            "npz_export_complete": self._on_npz_export_complete,
            "training_completed": self._on_training_completed,
            "training_threshold_reached": self._on_training_threshold_reached,
            "quality_score_updated": self._on_quality_score_updated,
            "training_blocked_by_quality": self._on_training_blocked_by_quality,
            "evaluation_completed": self._on_evaluation_completed,
            "training_intensity_changed": self._on_training_intensity_changed,
        }

    async def _on_start(self) -> None:
        """Hook called before main loop - check coordinator mode."""
        from app.config.env import env
        if env.is_coordinator or not env.training_enabled:
            logger.info(
                f"[TrainingTriggerDaemon] Skipped on coordinator node: {env.node_id} "
                f"(is_coordinator={env.is_coordinator}, training_enabled={env.training_enabled})"
            )
            self._coordinator_skip = True

    async def _on_stop(self) -> None:
        """Hook called when stopping - cancel active training tasks."""
        for config_key, task in self._active_training_tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"[TrainingTriggerDaemon] Cancelled training for {config_key}")

    def _get_training_params_for_intensity(
        self, intensity: str
    ) -> tuple[int, int, float]:
        """Map training intensity to (epochs, batch_size, lr_multiplier).

        December 2025: Fixes Gap 2 - training_intensity was defined but never consumed.
        The FeedbackLoopController sets intensity based on quality score:
          - hot_path (quality >= 0.90): Fast iteration, high LR
          - accelerated (quality >= 0.80): Increased training, moderate LR boost
          - normal (quality >= 0.65): Default parameters
          - reduced (quality >= 0.50): More epochs at lower LR for struggling configs
          - paused: Skip training entirely (handled in _maybe_trigger_training)

        Returns:
            Tuple of (epochs, batch_size, learning_rate_multiplier)
        """
        intensity_params = {
            # hot_path: Fast iteration with larger batches, higher LR
            "hot_path": (30, 1024, 1.5),
            # accelerated: More aggressive training
            "accelerated": (40, 768, 1.2),
            # normal: Default parameters
            "normal": (self.config.default_epochs, self.config.default_batch_size, 1.0),
            # reduced: Slower, more careful training for struggling configs
            "reduced": (60, 256, 0.8),
            # paused: Should not reach here, but use minimal params
            "paused": (10, 128, 0.5),
        }

        params = intensity_params.get(intensity)
        if params is None:
            logger.warning(
                f"[TrainingTriggerDaemon] Unknown intensity '{intensity}', using 'normal'"
            )
            params = intensity_params["normal"]

        return params

    def _get_dynamic_sample_threshold(self, config_key: str) -> int:
        """Get dynamically adjusted sample threshold for training.

        Phase 5 (Dec 2025): Uses ImprovementOptimizer to adjust thresholds
        based on training success patterns:
        - On promotion streak: Lower threshold → faster iteration
        - Struggling/regression: Higher threshold → more conservative

        Args:
            config_key: Configuration identifier

        Returns:
            Minimum sample count required to trigger training
        """
        try:
            from app.training.improvement_optimizer import get_dynamic_threshold

            dynamic_threshold = get_dynamic_threshold(config_key)

            # Log significant deviations from base threshold
            if dynamic_threshold != self.config.min_samples_threshold:
                logger.debug(
                    f"[TrainingTriggerDaemon] Dynamic threshold for {config_key}: "
                    f"{dynamic_threshold} (base: {self.config.min_samples_threshold})"
                )

            return dynamic_threshold

        except ImportError:
            logger.debug("[TrainingTriggerDaemon] improvement_optimizer not available")
        except Exception as e:
            logger.debug(f"[TrainingTriggerDaemon] Error getting dynamic threshold: {e}")

        # Fallback to static config threshold
        return self.config.min_samples_threshold

    async def _on_npz_export_complete(self, result: Any) -> None:
        """Handle NPZ export completion - immediate training trigger."""
        try:
            metadata = getattr(result, "metadata", {})
            config_key = metadata.get("config")
            board_type = metadata.get("board_type")
            num_players = metadata.get("num_players")
            npz_path = metadata.get("output_path", "")
            samples = metadata.get("samples", 0)

            if not config_key:
                # Try to build from board_type and num_players
                if board_type and num_players:
                    config_key = f"{board_type}_{num_players}p"
                else:
                    logger.debug("[TrainingTriggerDaemon] Missing config info in NPZ export result")
                    return

            # Update state
            state = self._get_or_create_state(config_key, board_type, num_players)
            state.last_npz_update = time.time()
            state.npz_sample_count = samples or 0
            state.npz_path = npz_path

            logger.info(
                f"[TrainingTriggerDaemon] NPZ export complete for {config_key}: "
                f"{samples} samples at {npz_path}"
            )

            # Check if we should trigger training
            await self._maybe_trigger_training(config_key)

        except Exception as e:
            logger.error(f"[TrainingTriggerDaemon] Error handling NPZ export: {e}")

    async def _on_training_completed(self, event: Any) -> None:
        """Handle training completion to update state."""
        try:
            payload = getattr(event, "payload", {})
            config_key = payload.get("config")

            if config_key and config_key in self._training_states:
                state = self._training_states[config_key]
                state.training_in_progress = False
                state.training_pid = None
                state.last_training_time = time.time()

                # Update ELO tracking if available
                if "elo" in payload:
                    old_elo = state.last_elo
                    state.last_elo = payload["elo"]
                    state.elo_trend = state.last_elo - old_elo

                logger.info(f"[TrainingTriggerDaemon] Training completed for {config_key}")

        except Exception as e:
            logger.error(f"[TrainingTriggerDaemon] Error handling training completion: {e}")

    def _intensity_from_quality(self, quality_score: float) -> str:
        """Map quality scores to training intensity."""
        if quality_score >= 0.90:
            return "hot_path"
        if quality_score >= 0.80:
            return "accelerated"
        if quality_score >= 0.65:
            return "normal"
        if quality_score >= 0.50:
            return "reduced"
        return "paused"

    async def _on_training_threshold_reached(self, event: Any) -> None:
        """Handle training threshold reached events from master_loop."""
        try:
            payload = getattr(event, "payload", {})
            config_key = payload.get("config") or payload.get("config_key")
            if not config_key:
                return

            board_type = payload.get("board_type")
            num_players = payload.get("num_players")
            state = self._get_or_create_state(config_key, board_type, num_players)

            intensity = payload.get("priority") or payload.get("training_intensity")
            if intensity:
                state.training_intensity = intensity
                logger.debug(
                    f"[TrainingTriggerDaemon] {config_key}: "
                    f"training_intensity set to {intensity}"
                )

            await self._maybe_trigger_training(config_key)

        except Exception as e:
            logger.error(f"[TrainingTriggerDaemon] Error handling training threshold: {e}")

    async def _on_quality_score_updated(self, event: Any) -> None:
        """Handle quality score updates to keep intensity in sync."""
        try:
            payload = getattr(event, "payload", {})
            config_key = payload.get("config_key") or payload.get("config")
            if not config_key:
                return

            quality_score = float(payload.get("quality_score", 0.0))
            state = self._get_or_create_state(config_key)
            state.training_intensity = self._intensity_from_quality(quality_score)

            logger.debug(
                f"[TrainingTriggerDaemon] {config_key}: "
                f"quality={quality_score:.2f} → intensity={state.training_intensity}"
            )
        except Exception as e:
            logger.error(f"[TrainingTriggerDaemon] Error handling quality update: {e}")

    async def _on_training_intensity_changed(self, event: Any) -> None:
        """Handle training intensity changes from unified_feedback orchestrator.

        December 2025: Enables event-driven quality feedback instead of direct
        object assignment. The unified_feedback.py emits TRAINING_INTENSITY_CHANGED
        when quality metrics change, and this handler updates local state.

        Payload:
            config_key: str - The board config (e.g., "hex8_2p")
            intensity: str - The new intensity level
            quality: float - The quality score that triggered the change
        """
        try:
            payload = getattr(event, "payload", {})
            config_key = payload.get("config_key") or payload.get("config")
            if not config_key:
                return

            new_intensity = payload.get("intensity")
            if not new_intensity:
                return

            state = self._get_or_create_state(config_key)
            old_intensity = state.training_intensity

            # Only update if intensity actually changed
            if old_intensity != new_intensity:
                state.training_intensity = new_intensity
                quality = payload.get("quality", 0.0)
                logger.info(
                    f"[TrainingTriggerDaemon] {config_key}: "
                    f"intensity changed via event: {old_intensity} → {new_intensity} "
                    f"(quality={quality:.2f})"
                )
        except Exception as e:
            logger.error(f"[TrainingTriggerDaemon] Error handling intensity change: {e}")

    async def _on_training_blocked_by_quality(self, event: Any) -> None:
        """Handle training blocked events to pause intensity."""
        try:
            payload = getattr(event, "payload", {})
            config_key = payload.get("config_key") or payload.get("config")
            if not config_key:
                return

            state = self._get_or_create_state(config_key)
            state.training_intensity = "paused"

            logger.info(f"[TrainingTriggerDaemon] {config_key}: training paused due to quality gate")
        except Exception as e:
            logger.error(f"[TrainingTriggerDaemon] Error handling training blocked: {e}")

    async def _on_evaluation_completed(self, event: Any) -> None:
        """Handle gauntlet evaluation completion - adjust training parameters (Dec 2025).

        This closes the critical feedback loop: gauntlet performance → training parameters.

        Adjustments based on win rate:
        - Win rate < 40%: Boost training intensity, increase epochs, trigger extra selfplay
        - Win rate 40-60%: Increase training to "accelerated" mode
        - Win rate 60-75%: Normal training, model is improving
        - Win rate > 75%: Reduce intensity, model is strong

        Expected improvement: 50-100 Elo by closing the feedback loop.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event

            config_key = payload.get("config", "") or payload.get("config_key", "")
            win_rate = payload.get("win_rate", 0.5)
            elo = payload.get("elo", 1500.0)
            games_played = payload.get("games_played", 0)

            # Try to parse config_key from model_id if not provided
            if not config_key:
                model_id = payload.get("model_id", "")
                if model_id:
                    # Extract config from model_id like "canonical_hex8_2p" -> "hex8_2p"
                    parts = model_id.replace("canonical_", "").rsplit("_", 1)
                    if len(parts) == 2 and parts[1].endswith("p"):
                        config_key = f"{parts[0]}_{parts[1]}"

            if not config_key:
                logger.debug("[TrainingTriggerDaemon] No config_key in evaluation event")
                return

            state = self._get_or_create_state(config_key)

            # Calculate Elo change if we have previous Elo
            elo_delta = elo - state.last_elo if state.last_elo > 0 else 0.0
            state.elo_trend = elo_delta
            old_elo = state.last_elo
            state.last_elo = elo

            logger.info(
                f"[TrainingTriggerDaemon] Evaluation complete for {config_key}: "
                f"win_rate={win_rate:.1%}, elo={elo:.0f} (delta={elo_delta:+.0f}), "
                f"games={games_played}"
            )

            # Determine new training intensity based on win rate
            old_intensity = state.training_intensity

            if win_rate < 0.40:
                # Struggling model - aggressive training boost
                state.training_intensity = "accelerated"
                logger.warning(
                    f"[TrainingTriggerDaemon] {config_key} struggling (win_rate={win_rate:.1%}), "
                    f"boosting training intensity to 'accelerated'"
                )
                # Trigger extra selfplay to generate more training data
                await self._trigger_selfplay_boost(config_key, multiplier=1.5)

            elif win_rate < 0.60:
                # Below target but not terrible - increase training
                state.training_intensity = "accelerated"
                logger.info(
                    f"[TrainingTriggerDaemon] {config_key} below target (win_rate={win_rate:.1%}), "
                    f"setting intensity to 'accelerated'"
                )

            elif win_rate < 0.75:
                # Reasonable performance - normal training
                state.training_intensity = "normal"
                if old_intensity != "normal":
                    logger.info(
                        f"[TrainingTriggerDaemon] {config_key} recovering (win_rate={win_rate:.1%}), "
                        f"returning to 'normal' intensity"
                    )

            else:
                # Strong model - can reduce training intensity
                if state.training_intensity != "reduced":
                    state.training_intensity = "reduced"
                    logger.info(
                        f"[TrainingTriggerDaemon] {config_key} strong (win_rate={win_rate:.1%}), "
                        f"reducing training intensity"
                    )

            # Check for Elo plateau (no improvement over multiple evaluations)
            if elo_delta <= 5 and old_elo > 0:
                state.consecutive_failures += 1
                if state.consecutive_failures >= 3:
                    logger.warning(
                        f"[TrainingTriggerDaemon] {config_key} Elo plateau detected "
                        f"({state.consecutive_failures} evals with minimal improvement), "
                        f"consider curriculum advancement"
                    )
                    await self._signal_curriculum_advancement(config_key)
            else:
                # Elo improved - reset failure counter
                state.consecutive_failures = 0

            # Record to FeedbackAccelerator for Elo momentum tracking
            await self._record_to_feedback_accelerator(config_key, elo, elo_delta)

        except Exception as e:
            logger.error(f"[TrainingTriggerDaemon] Error handling evaluation: {e}")

    async def _trigger_selfplay_boost(self, config_key: str, multiplier: float = 1.5) -> None:
        """Trigger additional selfplay for struggling configurations."""
        try:
            from app.coordination.selfplay_scheduler import get_selfplay_scheduler

            scheduler = get_selfplay_scheduler()
            if scheduler:
                # Boost allocation for this config
                scheduler.boost_config_allocation(config_key, multiplier)
                logger.info(
                    f"[TrainingTriggerDaemon] Boosted selfplay for {config_key} by {multiplier}x"
                )
        except Exception as e:
            logger.debug(f"[TrainingTriggerDaemon] Could not boost selfplay: {e}")

    async def _signal_curriculum_advancement(self, config_key: str) -> None:
        """Signal that curriculum should advance for a stagnant configuration."""
        try:
            from app.coordination.event_router import publish

            await publish(
                event_type="CURRICULUM_ADVANCEMENT_NEEDED",
                payload={
                    "config_key": config_key,
                    "reason": "elo_plateau",
                    "timestamp": time.time(),
                },
                source="training_trigger_daemon",
            )
            logger.info(
                f"[TrainingTriggerDaemon] Signaled curriculum advancement for {config_key}"
            )
        except Exception as e:
            logger.debug(f"[TrainingTriggerDaemon] Could not signal curriculum: {e}")

    async def _record_to_feedback_accelerator(
        self, config_key: str, elo: float, elo_delta: float
    ) -> None:
        """Record Elo update to FeedbackAccelerator for momentum tracking."""
        try:
            from app.training.feedback_accelerator import get_feedback_accelerator

            accelerator = get_feedback_accelerator()
            if accelerator:
                accelerator.record_elo_update(config_key, elo, elo_delta)
                logger.debug(
                    f"[TrainingTriggerDaemon] Recorded Elo to FeedbackAccelerator: "
                    f"{config_key}={elo:.0f} (delta={elo_delta:+.0f})"
                )
        except Exception as e:
            logger.debug(f"[TrainingTriggerDaemon] Could not record to accelerator: {e}")

    def _get_or_create_state(
        self, config_key: str, board_type: str | None = None, num_players: int | None = None
    ) -> ConfigTrainingState:
        """Get or create training state for a config."""
        if config_key not in self._training_states:
            # Parse config_key if board_type/num_players not provided
            if not board_type or not num_players:
                parts = config_key.rsplit("_", 1)
                board_type = parts[0] if len(parts) == 2 else config_key
                try:
                    num_players = int(parts[1].replace("p", "")) if len(parts) == 2 else 2
                except ValueError:
                    num_players = 2

            self._training_states[config_key] = ConfigTrainingState(
                config_key=config_key,
                board_type=board_type,
                num_players=num_players,
            )

        return self._training_states[config_key]

    async def _maybe_trigger_training(self, config_key: str) -> bool:
        """Check conditions and trigger training if appropriate."""
        state = self._training_states.get(config_key)
        if not state:
            return False

        # Check all conditions
        can_train, reason = await self._check_training_conditions(config_key)

        if not can_train:
            logger.debug(f"[TrainingTriggerDaemon] {config_key}: Cannot train - {reason}")
            return False

        # Trigger training
        logger.info(f"[TrainingTriggerDaemon] Triggering training for {config_key}")
        task = asyncio.create_task(self._run_training(config_key))
        task.add_done_callback(lambda t: self._on_training_task_done(t, config_key))
        self._active_training_tasks[config_key] = task

        return True

    async def _check_training_conditions(self, config_key: str) -> tuple[bool, str]:
        """Check all conditions for training trigger.

        Returns:
            Tuple of (can_train, reason)
        """
        state = self._training_states.get(config_key)
        if not state:
            return False, "no state"

        # 1. Check if training already in progress
        if state.training_in_progress:
            return False, "training already in progress"

        if state.training_intensity == "paused":
            return False, "training intensity paused"

        # Phase 4: Check circuit breaker before triggering training
        if HAS_CIRCUIT_BREAKER and get_training_breaker:
            breaker = get_training_breaker()
            if not breaker.can_execute(config_key):
                return False, f"circuit open for {config_key}"

        # 2. Check training cooldown
        time_since_training = time.time() - state.last_training_time
        cooldown_seconds = self.config.training_cooldown_hours * 3600
        if time_since_training < cooldown_seconds:
            remaining = (cooldown_seconds - time_since_training) / 3600
            return False, f"cooldown active ({remaining:.1f}h remaining)"

        # 3. Check data freshness (December 2025: use training_freshness for sync)
        data_age_hours = (time.time() - state.last_npz_update) / 3600
        if data_age_hours > self.config.max_data_age_hours:
            # December 29, 2025: Strict mode - fail immediately without sync attempt
            if self.config.strict_freshness_mode:
                return False, f"data too old ({data_age_hours:.1f}h) [strict mode - no sync]"
            elif self.config.enforce_freshness_with_sync:
                # Try to sync and wait for fresh data
                fresh = await self._ensure_fresh_data(state.board_type, state.num_players)
                if not fresh:
                    return False, f"data too old ({data_age_hours:.1f}h), sync failed"
                # Sync succeeded, continue with training check
                logger.info(
                    f"[TrainingTriggerDaemon] {config_key}: data refreshed via sync"
                )
            else:
                return False, f"data too old ({data_age_hours:.1f}h)"

        # 4. Check minimum samples
        # Phase 5 (Dec 2025): Use dynamic threshold from ImprovementOptimizer
        # Lower threshold when on a promotion streak, higher when struggling
        min_samples = self._get_dynamic_sample_threshold(config_key)
        if state.npz_sample_count < min_samples:
            return False, f"insufficient samples ({state.npz_sample_count} < {min_samples})"

        # 5. Check if idle GPU available (optional - allow training anyway)
        gpu_available = await self._check_gpu_availability()
        if not gpu_available:
            logger.warning(f"[TrainingTriggerDaemon] {config_key}: No idle GPU, proceeding anyway")

        # 6. Check concurrent training limit
        active_count = sum(
            1 for s in self._training_states.values() if s.training_in_progress
        )
        if active_count >= self.config.max_concurrent_training:
            return False, f"max concurrent training reached ({active_count})"

        # December 29, 2025: Auto-boost intensity for very fresh data
        # Fresh data (< 30 min old) suggests active selfplay → accelerate training
        if data_age_hours < 0.5:  # Less than 30 minutes old
            if state.training_intensity == "normal":
                state.training_intensity = "accelerated"
                logger.info(
                    f"[TrainingTriggerDaemon] {config_key}: boosted to 'accelerated' "
                    f"(data is {data_age_hours * 60:.0f}min fresh)"
                )
            elif state.training_intensity == "accelerated":
                state.training_intensity = "hot_path"
                logger.info(
                    f"[TrainingTriggerDaemon] {config_key}: boosted to 'hot_path' "
                    f"(data is {data_age_hours * 60:.0f}min fresh)"
                )

        return True, "all conditions met"

    async def _check_gpu_availability(self) -> bool:
        """Check if any GPU is available for training."""
        try:
            # Try to get GPU utilization via nvidia-smi
            process = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=10)

            if process.returncode == 0:
                for line in stdout.decode().strip().split("\n"):
                    try:
                        util = float(line.strip())
                        if util < self.config.gpu_idle_threshold_percent:
                            return True
                    except ValueError:
                        continue
                return False

        except (FileNotFoundError, asyncio.TimeoutError):
            pass
        except Exception as e:
            logger.debug(f"[TrainingTriggerDaemon] GPU check failed: {e}")

        # Assume GPU available if we can't check
        return True

    async def _ensure_fresh_data(self, board_type: str, num_players: int) -> bool:
        """Ensure training data is fresh, triggering sync if needed (December 2025).

        Uses training_freshness module to check data age and trigger sync
        if data is stale. This closes the data freshness feedback loop.

        Args:
            board_type: Board type for training
            num_players: Number of players

        Returns:
            True if data is now fresh, False if sync failed or timed out
        """
        try:
            from app.coordination.training_freshness import (
                DataFreshnessChecker,
                FreshnessConfig,
            )

            config = FreshnessConfig(
                max_age_hours=self.config.max_data_age_hours,
                trigger_sync=True,
                wait_for_sync=True,
                sync_timeout_seconds=self.config.freshness_sync_timeout_seconds,
            )

            checker = DataFreshnessChecker(config)
            result = await checker.ensure_fresh_data(board_type, num_players)

            if result.is_fresh:
                # Update local state with fresh data info
                config_key = f"{board_type}_{num_players}p"
                if config_key in self._training_states:
                    self._training_states[config_key].last_npz_update = time.time()
                    if result.games_available:
                        self._training_states[config_key].npz_sample_count = result.games_available
                return True

            logger.warning(
                f"[TrainingTriggerDaemon] Data freshness check failed for "
                f"{board_type}_{num_players}p: {result.error}"
            )
            return False

        except ImportError:
            logger.debug("[TrainingTriggerDaemon] training_freshness module not available")
            return False
        except Exception as e:
            logger.warning(f"[TrainingTriggerDaemon] ensure_fresh_data failed: {e}")
            return False

    async def _run_training(self, config_key: str) -> bool:
        """Run training subprocess for a configuration."""
        state = self._training_states.get(config_key)
        if not state:
            return False

        # Check for paused intensity - skip training
        if state.training_intensity == "paused":
            logger.info(
                f"[TrainingTriggerDaemon] Skipping training for {config_key}: "
                "intensity is 'paused' (quality score < 0.50)"
            )
            return False

        async with self._training_semaphore:
            state.training_in_progress = True

            try:
                # Get intensity-adjusted training parameters
                epochs, batch_size, lr_mult = self._get_training_params_for_intensity(
                    state.training_intensity
                )

                logger.info(
                    f"[TrainingTriggerDaemon] Starting training for {config_key} "
                    f"({state.npz_sample_count} samples, intensity={state.training_intensity}, "
                    f"epochs={epochs}, batch={batch_size}, lr_mult={lr_mult:.1f})"
                )

                # Build training command
                base_dir = Path(__file__).resolve().parent.parent.parent
                npz_path = state.npz_path or f"data/training/{config_key}.npz"

                # December 29, 2025: Use canonical model paths for consistent naming
                # Training outputs to models/canonical_{board}_{n}p.pth
                # The training script also saves timestamped checkpoints alongside for history
                model_filename = f"canonical_{config_key}.pth"
                model_path = str(base_dir / "models" / model_filename)

                cmd = [
                    sys.executable,
                    "-m", "app.training.train",
                    "--board-type", state.board_type,
                    "--num-players", str(state.num_players),
                    "--data-path", npz_path,
                    "--model-version", self.config.model_version,
                    "--epochs", str(epochs),
                    "--batch-size", str(batch_size),
                    "--save-path", model_path,  # December 29, 2025: Explicit save path
                    # December 2025: Allow stale data to unblock training when
                    # selfplay rate is slower than freshness threshold.
                    # The freshness check was blocking ALL training because game
                    # databases have content ages of 7-100+ hours while threshold is 1h.
                    "--allow-stale-data",
                    "--max-data-age-hours", "168",  # 1 week threshold
                ]

                # Store model_path in state for event emission
                state.npz_path = npz_path  # Keep npz_path
                state._pending_model_path = model_path  # Track expected model path

                # Compute adjusted learning rate (base 1e-3 * multiplier)
                # The training CLI uses --learning-rate for explicit LR setting
                if lr_mult != 1.0:
                    base_lr = 1e-3  # Default from TrainingConfig
                    adjusted_lr = base_lr * lr_mult
                    cmd.extend(["--learning-rate", f"{adjusted_lr:.6f}"])

                # Run training subprocess
                start_time = time.time()
                # December 29, 2025: Add RINGRIFT_ALLOW_PENDING_GATE to bypass parity
                # validation on cluster nodes that lack Node.js/npx
                training_env = {
                    **__import__("os").environ,
                    "PYTHONPATH": str(base_dir),
                    "RINGRIFT_ALLOW_PENDING_GATE": "true",
                }
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(base_dir),
                    env=training_env,
                )

                state.training_pid = process.pid

                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.config.training_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    logger.error(f"[TrainingTriggerDaemon] Training timed out for {config_key}")
                    state.consecutive_failures += 1
                    return False

                duration = time.time() - start_time

                if process.returncode == 0:
                    # Success
                    state.last_training_time = time.time()
                    state.consecutive_failures = 0

                    logger.info(
                        f"[TrainingTriggerDaemon] Training complete for {config_key}: "
                        f"{duration/3600:.1f}h"
                    )

                    # Emit training complete event
                    await self._emit_training_complete(config_key, success=True)
                    return True

                else:
                    # Failure
                    state.consecutive_failures += 1

                    # Adjust training intensity on consecutive failures (December 2025)
                    # This prevents wasting compute on configs that repeatedly fail
                    if state.consecutive_failures >= 3:
                        old_intensity = state.training_intensity
                        state.training_intensity = "paused"
                        logger.warning(
                            f"[TrainingTriggerDaemon] {config_key}: {state.consecutive_failures} "
                            f"consecutive failures, pausing training (was: {old_intensity})"
                        )
                    elif state.consecutive_failures >= 2:
                        old_intensity = state.training_intensity
                        if state.training_intensity not in ("reduced", "paused"):
                            state.training_intensity = "reduced"
                            logger.info(
                                f"[TrainingTriggerDaemon] {config_key}: 2 failures, reducing intensity "
                                f"(was: {old_intensity})"
                            )

                    logger.error(
                        f"[TrainingTriggerDaemon] Training failed for {config_key}: "
                        f"exit code {process.returncode}\n"
                        f"stderr: {stderr.decode()[:500]}"
                    )
                    await self._emit_training_complete(config_key, success=False)
                    return False

            except Exception as e:
                state.consecutive_failures += 1

                # Also adjust intensity on exceptions (December 2025)
                if state.consecutive_failures >= 2:
                    old_intensity = state.training_intensity
                    state.training_intensity = "reduced"
                    logger.info(
                        f"[TrainingTriggerDaemon] {config_key}: {state.consecutive_failures} failures "
                        f"(exception), reducing intensity (was: {old_intensity})"
                    )

                logger.error(f"[TrainingTriggerDaemon] Training error for {config_key}: {e}")
                return False

            finally:
                state.training_in_progress = False
                state.training_pid = None
                # Remove from active tasks
                self._active_training_tasks.pop(config_key, None)

    def _on_training_task_done(self, task: asyncio.Task, config_key: str) -> None:
        """Handle training task completion."""
        try:
            exc = task.exception()
            if exc:
                logger.error(f"[TrainingTriggerDaemon] Training task error for {config_key}: {exc}")
        except asyncio.CancelledError:
            pass
        except asyncio.InvalidStateError:
            pass

    async def _emit_training_complete(self, config_key: str, success: bool) -> None:
        """Emit training completion event."""
        # Phase 4: Record circuit breaker success/failure
        if HAS_CIRCUIT_BREAKER and get_training_breaker:
            breaker = get_training_breaker()
            if success:
                breaker.record_success(config_key)
            else:
                breaker.record_failure(config_key)

        try:
            from app.coordination.event_router import (
                StageEvent,
                StageCompletionResult,
                get_stage_event_bus,
            )

            state = self._training_states.get(config_key)

            # December 29, 2025: Include model_path in event for EvaluationDaemon
            model_path = ""
            if state and hasattr(state, "_pending_model_path"):
                model_path = state._pending_model_path
                # Verify model exists before including path
                if model_path and success:
                    from pathlib import Path
                    if not Path(model_path).exists():
                        logger.warning(
                            f"[TrainingTriggerDaemon] Model not found at {model_path}, "
                            "EvaluationDaemon may fail"
                        )

            bus = get_stage_event_bus()
            await bus.emit(
                StageCompletionResult(
                    event=StageEvent.TRAINING_COMPLETE if success else StageEvent.TRAINING_FAILED,
                    success=success,
                    timestamp=__import__("datetime").datetime.now().isoformat(),
                    metadata={
                        "config": config_key,
                        "board_type": state.board_type if state else "",
                        "num_players": state.num_players if state else 0,
                        "samples_trained": state.npz_sample_count if state else 0,
                        # December 29, 2025: Critical for evaluation pipeline
                        "model_path": model_path,
                        "checkpoint_path": model_path,  # Alias for compatibility
                    },
                )
            )
            logger.info(
                f"[TrainingTriggerDaemon] Emitted TRAINING_{'COMPLETE' if success else 'FAILED'} "
                f"for {config_key} (model_path={model_path})"
            )

        except Exception as e:
            logger.warning(f"[TrainingTriggerDaemon] Failed to emit training event: {e}")

    async def _run_cycle(self) -> None:
        """Main work loop iteration - called by HandlerBase at scan_interval_seconds."""
        # Skip if we're on a coordinator node
        if self._coordinator_skip:
            return

        # Scan for training opportunities
        await self._scan_for_training_opportunities()

    async def _scan_for_training_opportunities(self) -> None:
        """Scan for configs that may need training."""
        try:
            # Check existing states
            for config_key in list(self._training_states.keys()):
                await self._maybe_trigger_training(config_key)

            # Also scan for NPZ files that haven't been tracked
            training_dir = Path(__file__).resolve().parent.parent.parent / "data" / "training"
            if training_dir.exists():
                for npz_path in training_dir.glob("*.npz"):
                    # Parse config from filename
                    name = npz_path.stem
                    if "_" not in name:
                        continue

                    config_key = name
                    if config_key not in self._training_states:
                        # Create state and check
                        parts = config_key.rsplit("_", 1)
                        if len(parts) == 2:
                            board_type = parts[0]
                            try:
                                num_players = int(parts[1].replace("p", ""))
                            except ValueError:
                                continue

                            state = self._get_or_create_state(config_key, board_type, num_players)
                            state.npz_path = str(npz_path)
                            state.last_npz_update = npz_path.stat().st_mtime

                            # Get sample count from file (approximate)
                            try:
                                from app.utils.numpy_utils import safe_load_npz
                                with safe_load_npz(npz_path) as data:
                                    state.npz_sample_count = len(data.get("values", []))
                            except (FileNotFoundError, OSError, ValueError, ImportError):
                                pass

                            await self._maybe_trigger_training(config_key)

        except Exception as e:
            logger.error(f"[TrainingTriggerDaemon] Error scanning for opportunities: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get current daemon status."""
        return {
            "running": self._running,
            "configs_tracked": len(self._training_states),
            "active_training": sum(
                1 for s in self._training_states.values() if s.training_in_progress
            ),
            "states": {
                key: {
                    "training_in_progress": state.training_in_progress,
                    "training_intensity": state.training_intensity,
                    "last_training": state.last_training_time,
                    "npz_samples": state.npz_sample_count,
                    "last_elo": state.last_elo,
                    "failures": state.consecutive_failures,
                }
                for key, state in self._training_states.items()
            },
        }

    def health_check(self) -> HealthCheckResult:
        """Check daemon health.

        Returns:
            Health check result with training trigger status and metrics.
        """
        # Count active training tasks
        active_training = sum(
            1 for state in self._training_states.values()
            if state.training_in_progress
        )

        # Count failed configs
        failed_configs = sum(
            1 for state in self._training_states.values()
            if state.consecutive_failures > 0
        )

        # Determine health status
        healthy = self._running

        message = "Running" if healthy else "Daemon stopped"

        return HealthCheckResult(
            healthy=healthy,
            message=message,
            details={
                "running": self._running,
                "enabled": self.config.enabled,
                "configs_tracked": len(self._training_states),
                "active_training_tasks": active_training,
                "failed_configs": failed_configs,
                "max_concurrent_training": self.config.max_concurrent_training,
                "max_data_age_hours": self.config.max_data_age_hours,
            },
        )


# December 2025: Using HandlerBase singleton pattern
def get_training_trigger_daemon() -> TrainingTriggerDaemon:
    """Get or create the singleton training trigger daemon.

    December 2025: Now uses HandlerBase.get_instance() singleton pattern.
    """
    return TrainingTriggerDaemon.get_instance()


def reset_training_trigger_daemon() -> None:
    """Reset the singleton instance (for testing).

    December 2025: Added for test isolation.
    """
    TrainingTriggerDaemon.reset_instance()


async def start_training_trigger_daemon() -> TrainingTriggerDaemon:
    """Start the training trigger daemon (convenience function)."""
    daemon = get_training_trigger_daemon()
    await daemon.start()
    return daemon


__all__ = [
    "ConfigTrainingState",
    "TrainingTriggerConfig",
    "TrainingTriggerDaemon",
    "get_training_trigger_daemon",
    "reset_training_trigger_daemon",
    "start_training_trigger_daemon",
]
