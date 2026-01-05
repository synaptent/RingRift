"""Training execution and subprocess management.

Jan 4, 2026 - Sprint 17.9: Extracted from training_trigger_daemon.py as part of
daemon decomposition (Phase 4).

This module handles:
- Training subprocess creation and management
- Work queue dispatch for coordinator nodes
- Graceful process termination (SIGTERM → SIGKILL)
- Training event emission (complete, failed, timeout)

Usage:
    from app.coordination.training_execution import (
        TrainingExecutor,
        TrainingExecutionConfig,
        dispatch_training_to_queue,
        graceful_kill_process,
    )

    executor = TrainingExecutor(config)
    success = await executor.run_training(config_key, state, arch)
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from app.coordination.training_trigger_types import (
        ArchitectureSpec,
        ConfigTrainingState,
        TrainingTriggerConfig,
    )

logger = logging.getLogger(__name__)

__all__ = [
    "TrainingExecutionConfig",
    "TrainingExecutor",
    "dispatch_training_to_queue",
    "graceful_kill_process",
    "emit_training_complete",
    "emit_training_failed",
]


@dataclass
class TrainingExecutionConfig:
    """Configuration for training execution.

    Extracted from TrainingTriggerConfig for execution-specific settings.
    """

    # Timeout settings
    training_timeout_seconds: int = 86400  # 24 hours
    training_timeout_hours: float = 4.0
    graceful_kill_timeout_seconds: float = 30.0

    # Training parameters
    default_epochs: int = 50
    default_batch_size: int = 512
    model_version: str = "v2"

    # Environment
    allow_pending_gate: bool = True
    pythonpath: str = ""

    @classmethod
    def from_trigger_config(cls, config: "TrainingTriggerConfig") -> "TrainingExecutionConfig":
        """Create execution config from trigger config."""
        return cls(
            training_timeout_seconds=config.training_timeout_seconds,
            training_timeout_hours=config.training_timeout_hours,
            graceful_kill_timeout_seconds=config.graceful_kill_timeout_seconds,
            default_epochs=config.default_epochs,
            default_batch_size=config.default_batch_size,
            model_version=config.model_version,
        )


@dataclass
class TrainingResult:
    """Result of a training execution."""

    success: bool
    config_key: str
    duration_seconds: float = 0.0
    model_path: str = ""
    error_message: str = ""
    exit_code: int | None = None

    @property
    def duration_hours(self) -> float:
        return self.duration_seconds / 3600


@dataclass
class ExecutionStats:
    """Statistics for training execution."""

    jobs_started: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    jobs_timed_out: int = 0
    jobs_killed: int = 0
    total_training_hours: float = 0.0
    last_execution_time: float = 0.0


class TrainingExecutor:
    """Manages training subprocess execution.

    This class handles:
    - Running training as subprocess with proper environment
    - Timeout monitoring and graceful termination
    - Event emission for completion/failure
    - Optional work queue dispatch for coordinator nodes

    Session 17.9: Extracted from TrainingTriggerDaemon for independent testability.
    """

    def __init__(
        self,
        config: TrainingExecutionConfig | None = None,
        dispatch_to_queue: bool = False,
        get_training_params: Callable[[str], tuple[int, int, float]] | None = None,
    ) -> None:
        """Initialize training executor.

        Args:
            config: Execution configuration
            dispatch_to_queue: If True, dispatch to work queue instead of running locally
            get_training_params: Callback to get (epochs, batch_size, lr_mult) for intensity
        """
        self.config = config or TrainingExecutionConfig()
        self._dispatch_to_queue = dispatch_to_queue
        self._get_training_params = get_training_params or self._default_training_params
        self.stats = ExecutionStats()
        self._base_dir = Path(__file__).resolve().parent.parent.parent

    def _default_training_params(self, intensity: str) -> tuple[int, int, float]:
        """Default training parameters based on intensity."""
        params = {
            "hot_path": (self.config.default_epochs * 2, 256, 2.0),
            "accelerated": (int(self.config.default_epochs * 1.5), 384, 1.5),
            "normal": (self.config.default_epochs, self.config.default_batch_size, 1.0),
            "reduced": (self.config.default_epochs // 2, self.config.default_batch_size, 0.5),
            "paused": (0, 0, 0.0),
        }
        return params.get(intensity, params["normal"])

    async def dispatch_to_work_queue(
        self,
        config_key: str,
        state: "ConfigTrainingState",
        arch: "ArchitectureSpec | None" = None,
    ) -> bool:
        """Dispatch training job to work queue for remote execution.

        Used on coordinator nodes that don't have GPUs. Training jobs
        are dispatched to the centralized work queue for GPU nodes to claim.

        Args:
            config_key: Configuration identifier (e.g., "hex8_2p")
            state: Current training state for this config
            arch: Optional architecture specification

        Returns:
            True if job was successfully queued
        """
        try:
            from app.coordination.work_distributor import (
                get_work_distributor,
                DistributedWorkConfig,
            )

            distributor = get_work_distributor()

            # Get intensity-adjusted training parameters
            epochs, batch_size, lr_mult = self._get_training_params(
                state.training_intensity
            )

            # Apply architecture-specific overrides
            arch_name = "v5"
            if arch is not None:
                arch_name = arch.name
                if arch.epochs is not None:
                    epochs = arch.epochs
                if arch.batch_size is not None:
                    batch_size = arch.batch_size

            # Compute priority based on config characteristics
            priority = self._compute_work_priority(state)

            # Build config for work queue submission
            work_config = DistributedWorkConfig(
                require_gpu=True,
                require_high_memory=state.board_type in ("square19", "hexagonal"),
                priority=priority,
            )

            # Submit to work queue
            work_id = await distributor.submit_training(
                board=state.board_type,
                num_players=state.num_players,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=1e-3 * lr_mult,
                config=work_config,
                model_version=arch_name,
            )

            if work_id:
                logger.info(
                    f"[TrainingExecutor] Dispatched training to queue: {config_key} "
                    f"(work_id={work_id}, arch={arch_name}, epochs={epochs}, batch={batch_size})"
                )
                self.stats.jobs_started += 1
                return True
            else:
                logger.warning(
                    f"[TrainingExecutor] Failed to dispatch training for {config_key}: "
                    "work queue returned None"
                )
                return False

        except ImportError as e:
            logger.warning(
                f"[TrainingExecutor] Cannot dispatch to work queue (module not available): {e}"
            )
            return False
        except Exception as e:
            logger.error(
                f"[TrainingExecutor] Failed to dispatch training for {config_key}: {e}"
            )
            return False

    def _compute_work_priority(self, state: "ConfigTrainingState") -> int:
        """Compute priority for work queue submission."""
        priority = 50
        # Higher priority for underrepresented configs
        if state.board_type in ("square19", "hexagonal"):
            priority = min(100, priority + 20)
        if state.num_players in (3, 4):
            priority = min(100, priority + 15)
        # Boost priority for accelerating configs
        if state.elo_velocity > 10.0:
            priority = min(100, priority + 10)
        return priority

    async def run_training(
        self,
        config_key: str,
        state: "ConfigTrainingState",
        arch: "ArchitectureSpec | None" = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> TrainingResult:
        """Run training subprocess for a configuration.

        Args:
            config_key: Configuration identifier (e.g., "hex8_2p")
            state: Current training state
            arch: Optional architecture specification
            semaphore: Optional semaphore for concurrency control

        Returns:
            TrainingResult with success status and details
        """
        # Check for paused intensity
        if state.training_intensity == "paused":
            logger.info(
                f"[TrainingExecutor] Skipping training for {config_key}: "
                "intensity is 'paused' (quality score < 0.50)"
            )
            return TrainingResult(
                success=False,
                config_key=config_key,
                error_message="Training paused due to low quality score",
            )

        # Dispatch to queue if configured
        if self._dispatch_to_queue:
            success = await self.dispatch_to_work_queue(config_key, state, arch)
            return TrainingResult(success=success, config_key=config_key)

        # Default to v5 if no architecture specified
        if arch is None:
            from app.coordination.training_trigger_types import ArchitectureSpec
            arch = ArchitectureSpec(
                name="v5", enabled=True, configs=["*"], priority=1.0
            )

        # Run with optional semaphore
        if semaphore:
            async with semaphore:
                return await self._run_training_subprocess(config_key, state, arch)
        else:
            return await self._run_training_subprocess(config_key, state, arch)

    async def _run_training_subprocess(
        self,
        config_key: str,
        state: "ConfigTrainingState",
        arch: "ArchitectureSpec",
    ) -> TrainingResult:
        """Execute training as a subprocess.

        Args:
            config_key: Configuration identifier
            state: Current training state
            arch: Architecture specification

        Returns:
            TrainingResult with execution details
        """
        start_time = time.time()
        self.stats.jobs_started += 1

        try:
            # Get training parameters
            epochs, batch_size, lr_mult = self._get_training_params(
                state.training_intensity
            )

            # Apply architecture overrides
            if arch.epochs is not None:
                epochs = arch.epochs
            if arch.batch_size is not None:
                batch_size = arch.batch_size

            logger.info(
                f"[TrainingExecutor] Starting training for {config_key} "
                f"with architecture {arch.name} "
                f"({state.npz_sample_count} samples, intensity={state.training_intensity}, "
                f"epochs={epochs}, batch={batch_size}, lr_mult={lr_mult:.2f})"
            )

            # Build training command
            npz_path = state.npz_path or f"data/training/{config_key}.npz"
            model_filename = f"canonical_{config_key}_{arch.name}.pth"
            model_path = str(self._base_dir / "models" / model_filename)

            cmd = self._build_training_command(
                board_type=state.board_type,
                num_players=state.num_players,
                npz_path=npz_path,
                model_path=model_path,
                arch_name=arch.name,
                epochs=epochs,
                batch_size=batch_size,
                lr_mult=lr_mult,
            )

            # Set up environment
            training_env = {
                **os.environ,
                "PYTHONPATH": self.config.pythonpath or str(self._base_dir),
                "RINGRIFT_ALLOW_PENDING_GATE": "true" if self.config.allow_pending_gate else "false",
            }

            # Run subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._base_dir),
                env=training_env,
            )

            # Wait with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.training_timeout_seconds,
                )
            except asyncio.TimeoutError:
                await graceful_kill_process(
                    process.pid,
                    config_key,
                    grace_seconds=self.config.graceful_kill_timeout_seconds,
                )
                self.stats.jobs_timed_out += 1
                duration = time.time() - start_time
                return TrainingResult(
                    success=False,
                    config_key=config_key,
                    duration_seconds=duration,
                    error_message=f"Training timed out after {duration/3600:.1f}h",
                )

            duration = time.time() - start_time
            self.stats.total_training_hours += duration / 3600
            self.stats.last_execution_time = time.time()

            if process.returncode == 0:
                self.stats.jobs_completed += 1
                logger.info(
                    f"[TrainingExecutor] Training complete for {config_key}: "
                    f"{duration/3600:.1f}h"
                )
                return TrainingResult(
                    success=True,
                    config_key=config_key,
                    duration_seconds=duration,
                    model_path=model_path,
                    exit_code=0,
                )
            else:
                self.stats.jobs_failed += 1
                error_msg = stderr.decode()[:500] if stderr else "Unknown error"
                logger.error(
                    f"[TrainingExecutor] Training failed for {config_key}: "
                    f"exit code {process.returncode}\n"
                    f"stderr: {error_msg}"
                )
                return TrainingResult(
                    success=False,
                    config_key=config_key,
                    duration_seconds=duration,
                    error_message=error_msg,
                    exit_code=process.returncode,
                )

        except Exception as e:
            self.stats.jobs_failed += 1
            duration = time.time() - start_time
            logger.error(f"[TrainingExecutor] Training error for {config_key}: {e}")
            return TrainingResult(
                success=False,
                config_key=config_key,
                duration_seconds=duration,
                error_message=str(e),
            )

    def _build_training_command(
        self,
        board_type: str,
        num_players: int,
        npz_path: str,
        model_path: str,
        arch_name: str,
        epochs: int,
        batch_size: int,
        lr_mult: float,
    ) -> list[str]:
        """Build the training CLI command."""
        cmd = [
            sys.executable,
            "-m", "app.training.train",
            "--board-type", board_type,
            "--num-players", str(num_players),
            "--data-path", npz_path,
            "--model-version", arch_name,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--save-path", model_path,
            "--allow-stale-data",
            "--max-data-age-hours", "168",
        ]

        # Add learning rate if modified
        if lr_mult != 1.0:
            base_lr = 1e-3
            adjusted_lr = base_lr * lr_mult
            cmd.extend(["--learning-rate", f"{adjusted_lr:.6f}"])

        return cmd


async def dispatch_training_to_queue(
    config_key: str,
    state: "ConfigTrainingState",
    arch: "ArchitectureSpec | None" = None,
    get_training_params: Callable[[str], tuple[int, int, float]] | None = None,
) -> bool:
    """Dispatch training job to work queue.

    Convenience function for coordinator nodes.

    Args:
        config_key: Configuration identifier
        state: Training state
        arch: Optional architecture spec
        get_training_params: Optional callback for parameters

    Returns:
        True if successfully queued
    """
    executor = TrainingExecutor(
        dispatch_to_queue=True,
        get_training_params=get_training_params,
    )
    return await executor.dispatch_to_work_queue(config_key, state, arch)


async def graceful_kill_process(
    pid: int,
    context: str,
    grace_seconds: float = 30.0,
    emit_events: bool = True,
) -> bool:
    """Gracefully kill a process - SIGTERM first, then SIGKILL.

    Sends SIGTERM to allow process to save checkpoints, then SIGKILL
    after grace period if still running.

    Args:
        pid: Process ID to kill
        context: Context string for logging (e.g., config_key)
        grace_seconds: Time to wait between SIGTERM and SIGKILL
        emit_events: Whether to emit training timeout events

    Returns:
        True if process was killed, False on error
    """
    try:
        # Emit timeout event if enabled
        if emit_events:
            from app.coordination.event_emission_helpers import safe_emit_event

            safe_emit_event(
                "TRAINING_TIMEOUT_REACHED",
                {
                    "config_key": context,
                    "pid": pid,
                    "grace_seconds": grace_seconds,
                    "timestamp": time.time(),
                },
                context="graceful_kill_process",
            )

        # Send SIGTERM for graceful shutdown
        os.kill(pid, signal.SIGTERM)
        logger.info(
            f"[graceful_kill_process] Sent SIGTERM to PID {pid} for {context}, "
            f"waiting {grace_seconds}s for graceful exit"
        )

        # Wait for process to exit gracefully
        start_wait = time.time()
        while time.time() - start_wait < grace_seconds:
            try:
                os.kill(pid, 0)  # Check if process exists
                await asyncio.sleep(1.0)
            except ProcessLookupError:
                logger.info(
                    f"[graceful_kill_process] Process PID {pid} exited gracefully for {context}"
                )
                return True

        # Process still running - send SIGKILL
        try:
            os.kill(pid, signal.SIGKILL)
            logger.warning(
                f"[graceful_kill_process] Sent SIGKILL to PID {pid} for {context} "
                f"(did not exit after {grace_seconds}s SIGTERM)"
            )
            return True
        except ProcessLookupError:
            logger.info(
                f"[graceful_kill_process] Process PID {pid} exited just before SIGKILL for {context}"
            )
            return True

    except ProcessLookupError:
        logger.debug(f"[graceful_kill_process] Process {pid} already dead for {context}")
        return True
    except PermissionError:
        logger.error(f"[graceful_kill_process] Permission denied killing PID {pid} for {context}")
        return False
    except OSError as e:
        logger.error(f"[graceful_kill_process] OS error killing PID {pid} for {context}: {e}")
        return False


async def emit_training_complete(
    config_key: str,
    success: bool,
    model_path: str = "",
    board_type: str = "",
    num_players: int = 0,
    sample_count: int = 0,
) -> bool:
    """Emit training completion event.

    Args:
        config_key: Configuration identifier
        success: Whether training succeeded
        model_path: Path to saved model
        board_type: Board type (e.g., "hex8")
        num_players: Number of players
        sample_count: Number of samples trained on

    Returns:
        True if event was emitted successfully
    """
    try:
        from app.coordination.event_router import (
            StageEvent,
            StageCompletionResult,
            get_stage_event_bus,
        )

        # Verify model exists if success
        if model_path and success:
            if not Path(model_path).exists():
                logger.warning(
                    f"[emit_training_complete] Model not found at {model_path}, "
                    "EvaluationDaemon may fail"
                )

        bus = get_stage_event_bus()
        await bus.emit(
            StageCompletionResult(
                event=StageEvent.TRAINING_COMPLETE if success else StageEvent.TRAINING_FAILED,
                success=success,
                timestamp=datetime.datetime.now().isoformat(),
                metadata={
                    "config": config_key,
                    "board_type": board_type,
                    "num_players": num_players,
                    "samples_trained": sample_count,
                    "model_path": model_path,
                    "checkpoint_path": model_path,
                },
            )
        )
        logger.info(
            f"[emit_training_complete] Emitted TRAINING_{'COMPLETE' if success else 'FAILED'} "
            f"for {config_key} (model_path={model_path})"
        )
        return True

    except Exception as e:
        logger.warning(f"[emit_training_complete] Failed to emit training event: {e}")
        return False


async def emit_training_failed(config_key: str, reason: str) -> bool:
    """Emit TRAINING_FAILED event for timed-out or errored training.

    Args:
        config_key: Configuration identifier
        reason: Failure reason (e.g., "timeout", "error")

    Returns:
        True if event was emitted successfully
    """
    from app.coordination.event_emission_helpers import safe_emit_event

    success = safe_emit_event(
        "TRAINING_FAILED",
        {
            "config_key": config_key,
            "reason": reason,
            "timestamp": time.time(),
            "source": "training_execution",
        },
        context="emit_training_failed",
    )
    if success:
        logger.info(f"[emit_training_failed] Emitted TRAINING_FAILED for {config_key}: {reason}")
    return success
