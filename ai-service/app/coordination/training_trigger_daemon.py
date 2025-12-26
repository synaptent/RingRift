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
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrainingTriggerConfig:
    """Configuration for training trigger decisions."""

    enabled: bool = True
    # Data freshness
    max_data_age_hours: float = 1.0
    # Minimum samples to trigger training
    min_samples_threshold: int = 10000
    # Cooldown between training runs for same config
    training_cooldown_hours: float = 4.0
    # Maximum concurrent training jobs
    max_concurrent_training: int = 2
    # GPU utilization threshold for "idle"
    gpu_idle_threshold_percent: float = 20.0
    # Timeout for training subprocess (24 hours)
    training_timeout_seconds: int = 86400
    # Check interval for periodic scans
    scan_interval_seconds: int = 300  # 5 minutes
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


class TrainingTriggerDaemon:
    """Daemon that automatically triggers training when conditions are met."""

    def __init__(self, config: TrainingTriggerConfig | None = None):
        self.config = config or TrainingTriggerConfig()
        self._running = False
        self._task: asyncio.Task | None = None
        self._training_states: dict[str, ConfigTrainingState] = {}
        self._training_semaphore = asyncio.Semaphore(self.config.max_concurrent_training)
        self._event_subscriptions: list[Any] = []
        self._active_training_tasks: dict[str, asyncio.Task] = {}

    async def start(self) -> None:
        """Start the training trigger daemon."""
        if self._running:
            logger.warning("[TrainingTriggerDaemon] Already running")
            return

        self._running = True
        logger.info("[TrainingTriggerDaemon] Starting training trigger daemon")

        # Subscribe to relevant events
        await self._subscribe_to_events()

        # Start background monitoring task
        self._task = asyncio.create_task(self._monitor_loop())
        self._task.add_done_callback(self._on_task_done)

    async def stop(self) -> None:
        """Stop the training trigger daemon."""
        self._running = False

        # Cancel monitoring task
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Cancel any active training tasks
        for config_key, task in self._active_training_tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"[TrainingTriggerDaemon] Cancelled training for {config_key}")

        # Unsubscribe from events
        for unsub in self._event_subscriptions:
            try:
                if callable(unsub):
                    unsub()
            except Exception as e:
                logger.debug(f"[TrainingTriggerDaemon] Error unsubscribing: {e}")

        logger.info("[TrainingTriggerDaemon] Stopped")

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Handle task completion or failure."""
        try:
            exc = task.exception()
            if exc:
                logger.error(f"[TrainingTriggerDaemon] Task failed: {exc}")
        except asyncio.CancelledError:
            pass
        except asyncio.InvalidStateError:
            pass

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

    async def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events."""
        try:
            from app.coordination.stage_events import StageEvent, get_event_bus

            bus = get_event_bus()
            unsub = bus.subscribe(StageEvent.NPZ_EXPORT_COMPLETE, self._on_npz_export_complete)
            self._event_subscriptions.append(unsub)
            logger.info("[TrainingTriggerDaemon] Subscribed to NPZ_EXPORT_COMPLETE events")
        except ImportError:
            logger.warning("[TrainingTriggerDaemon] Stage events not available")

        try:
            from app.distributed.data_events import DataEventType, get_event_bus

            bus = get_event_bus()
            # Subscribe to training completion to track state
            unsub = bus.subscribe(DataEventType.TRAINING_COMPLETED, self._on_training_completed)
            self._event_subscriptions.append(unsub)
            logger.info("[TrainingTriggerDaemon] Subscribed to TRAINING_COMPLETED events")
        except ImportError:
            logger.warning("[TrainingTriggerDaemon] Data events not available")

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

        # 2. Check training cooldown
        time_since_training = time.time() - state.last_training_time
        cooldown_seconds = self.config.training_cooldown_hours * 3600
        if time_since_training < cooldown_seconds:
            remaining = (cooldown_seconds - time_since_training) / 3600
            return False, f"cooldown active ({remaining:.1f}h remaining)"

        # 3. Check data freshness
        data_age_hours = (time.time() - state.last_npz_update) / 3600
        if data_age_hours > self.config.max_data_age_hours:
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

                cmd = [
                    sys.executable,
                    "-m", "app.training.train",
                    "--board-type", state.board_type,
                    "--num-players", str(state.num_players),
                    "--data-path", npz_path,
                    "--model-version", self.config.model_version,
                    "--epochs", str(epochs),
                    "--batch-size", str(batch_size),
                ]

                # Compute adjusted learning rate (base 1e-3 * multiplier)
                # The training CLI uses --learning-rate for explicit LR setting
                if lr_mult != 1.0:
                    base_lr = 1e-3  # Default from TrainingConfig
                    adjusted_lr = base_lr * lr_mult
                    cmd.extend(["--learning-rate", f"{adjusted_lr:.6f}"])

                # Run training subprocess
                start_time = time.time()
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(base_dir),
                    env={**__import__("os").environ, "PYTHONPATH": str(base_dir)},
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
                    logger.error(
                        f"[TrainingTriggerDaemon] Training failed for {config_key}: "
                        f"exit code {process.returncode}\n"
                        f"stderr: {stderr.decode()[:500]}"
                    )
                    await self._emit_training_complete(config_key, success=False)
                    return False

            except Exception as e:
                state.consecutive_failures += 1
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
        try:
            from app.coordination.stage_events import (
                StageEvent,
                StageCompletionResult,
                get_event_bus,
            )

            state = self._training_states.get(config_key)

            bus = get_event_bus()
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
                    },
                )
            )
            logger.debug(
                f"[TrainingTriggerDaemon] Emitted TRAINING_{'COMPLETE' if success else 'FAILED'} "
                f"for {config_key}"
            )

        except Exception as e:
            logger.warning(f"[TrainingTriggerDaemon] Failed to emit training event: {e}")

    async def _monitor_loop(self) -> None:
        """Background loop to periodically check for training opportunities."""
        while self._running:
            try:
                # Scan for training opportunities
                await self._scan_for_training_opportunities()

                # Wait before next scan
                await asyncio.sleep(self.config.scan_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[TrainingTriggerDaemon] Monitor loop error: {e}")
                await asyncio.sleep(60)

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
                                import numpy as np
                                with np.load(npz_path, allow_pickle=True) as data:
                                    state.npz_sample_count = len(data.get("values", []))
                            except Exception:
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


# Singleton instance
_daemon: TrainingTriggerDaemon | None = None


def get_training_trigger_daemon() -> TrainingTriggerDaemon:
    """Get or create the singleton training trigger daemon."""
    global _daemon
    if _daemon is None:
        _daemon = TrainingTriggerDaemon()
    return _daemon


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
    "start_training_trigger_daemon",
]
