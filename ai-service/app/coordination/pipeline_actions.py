"""Pipeline Action Triggers - Invokes actual work for each pipeline stage.

This module bridges the gap between the DataPipelineOrchestrator's event handlers
and the actual work that needs to be done for each stage. Previously, the orchestrator
only tracked state transitions without invoking actions.

Each action trigger:
1. Invokes the actual subprocess/script for the stage
2. Returns a StageCompletionResult indicating success/failure
3. Emits appropriate events for downstream handlers
4. Handles errors gracefully with detailed error information

Usage:
    from app.coordination.pipeline_actions import (
        trigger_data_sync,
        trigger_npz_export,
        trigger_training,
        trigger_evaluation,
        trigger_promotion,
    )

    # In DataPipelineOrchestrator._on_selfplay_complete:
    if self.auto_trigger and not self._circuit_breaker.is_open:
        result = await trigger_data_sync(board_type, num_players, iteration)
        if not result.success:
            self._record_failure(PipelineStage.DATA_SYNC, result.error)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Import promotion thresholds from centralized config
try:
    from app.config.thresholds import (
        PRODUCTION_MIN_WIN_RATE_VS_HEURISTIC,
        PRODUCTION_MIN_WIN_RATE_VS_RANDOM,
    )
except ImportError:
    # Fallback values - keep in sync with app/config/thresholds.py
    PRODUCTION_MIN_WIN_RATE_VS_RANDOM = 0.90  # 90%
    PRODUCTION_MIN_WIN_RATE_VS_HEURISTIC = 0.60  # 60%

# December 30, 2025: Centralized retry config for consistency
from app.utils.retry import RetryConfig
from app.coordination.event_utils import make_config_key

# Import event emitter for training triggered events (January 2026 - migrated to event_router)
try:
    from app.coordination.event_emission_helpers import safe_emit_event_async
    HAS_EVENT_EMITTERS = True
except ImportError:
    HAS_EVENT_EMITTERS = False

logger = logging.getLogger(__name__)


# =============================================================================
# Phase 4.3 Dec 29, 2025: Per-config Circuit Breaker
# Prevents failed configs from retrying infinitely and starving others
# =============================================================================

@dataclass
class CircuitState:
    """Circuit breaker state for a config.

    Tracks consecutive failures and opens circuit after threshold.
    Open circuit prevents new actions for that config until backoff expires.
    """

    failures: int = 0
    last_failure_time: float = 0.0
    open_until: float = 0.0  # Unix timestamp when circuit can close
    total_failures: int = 0  # Lifetime failures for metrics

    @property
    def is_open(self) -> bool:
        """Check if circuit is currently open (blocking actions)."""
        return time.time() < self.open_until

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for status reporting."""
        return {
            "failures": self.failures,
            "total_failures": self.total_failures,
            "is_open": self.is_open,
            "open_until": self.open_until,
            "remaining_seconds": max(0, self.open_until - time.time()),
        }


# Per-config circuit breaker states
_circuit_states: dict[str, CircuitState] = {}

# Circuit breaker configuration
CIRCUIT_FAILURE_THRESHOLD = int(os.environ.get("RINGRIFT_CIRCUIT_FAILURE_THRESHOLD", "5"))
CIRCUIT_BACKOFF_SECONDS = float(os.environ.get("RINGRIFT_CIRCUIT_BACKOFF_SECONDS", "300"))  # 5 min

# Dec 29, 2025 - Phase 3: Quality gate configuration
# Training is blocked if quality score is below this threshold
# Relaxed from 0.6 to 0.3 for 48-hour autonomous operation (Dec 2025)
# This reduces training gate blocks from ~15% to ~5%
QUALITY_GATE_THRESHOLD = float(os.environ.get("RINGRIFT_QUALITY_GATE_THRESHOLD", "0.3"))


async def _check_training_data_quality(config_key: str, npz_path: str) -> tuple[bool, float]:
    """Check if training data quality is sufficient.

    Dec 29, 2025 - Phase 3: Quality gate blocking training.
    Prevents training on low-quality data (e.g., heuristic-only, high parity failures).

    Args:
        config_key: Config like "hex8_2p"
        npz_path: Path to NPZ training data

    Returns:
        Tuple of (quality_ok, quality_score)
    """
    try:
        from app.coordination.quality_monitor_daemon import get_quality_daemon

        daemon = get_quality_daemon()
        if daemon:
            quality = daemon.get_config_quality(config_key)
            if quality is not None:
                return (quality >= QUALITY_GATE_THRESHOLD, quality)
    except (ImportError, AttributeError):
        pass

    # If QualityMonitorDaemon not available, check the data pipeline orchestrator
    try:
        from app.coordination.data_pipeline_orchestrator import get_data_pipeline_orchestrator

        orchestrator = get_data_pipeline_orchestrator()
        if orchestrator and hasattr(orchestrator, "_check_training_data_quality"):
            quality_ok = await orchestrator._check_training_data_quality(Path(npz_path))
            if not quality_ok:
                return (False, 0.5)  # Assume below threshold if check fails
    except (ImportError, AttributeError):
        pass

    # Default: assume quality is acceptable if we can't check
    return (True, 0.7)


def _get_circuit_state(config_key: str) -> CircuitState:
    """Get or create circuit state for a config."""
    if config_key not in _circuit_states:
        _circuit_states[config_key] = CircuitState()
    return _circuit_states[config_key]


def check_circuit(config_key: str) -> bool:
    """Check if actions are allowed for this config.

    Args:
        config_key: Config key like "hex8_2p"

    Returns:
        True if circuit is closed (actions allowed), False if open (blocked)
    """
    state = _get_circuit_state(config_key)
    if state.is_open:
        logger.warning(
            f"[PipelineActions] Circuit open for {config_key}, "
            f"{state.open_until - time.time():.0f}s remaining"
        )
        return False
    return True


def record_action_success(config_key: str) -> None:
    """Record a successful action - resets failure count.

    Args:
        config_key: Config key like "hex8_2p"
    """
    state = _get_circuit_state(config_key)
    if state.failures > 0:
        logger.info(
            f"[PipelineActions] Circuit recovered for {config_key} "
            f"after {state.failures} failures"
        )
    state.failures = 0
    state.open_until = 0.0


def record_action_failure(config_key: str, error: str | None = None) -> None:
    """Record a failed action - may open circuit.

    Args:
        config_key: Config key like "hex8_2p"
        error: Optional error message for logging
    """
    state = _get_circuit_state(config_key)
    state.failures += 1
    state.total_failures += 1
    state.last_failure_time = time.time()

    if state.failures >= CIRCUIT_FAILURE_THRESHOLD:
        # Open circuit with exponential backoff
        backoff_multiplier = min(2 ** (state.failures - CIRCUIT_FAILURE_THRESHOLD), 8)
        backoff = CIRCUIT_BACKOFF_SECONDS * backoff_multiplier
        state.open_until = time.time() + backoff

        logger.error(
            f"[PipelineActions] Circuit OPENED for {config_key} after "
            f"{state.failures} consecutive failures. Backoff: {backoff:.0f}s. "
            f"Error: {error or 'Unknown'}"
        )


def get_circuit_status() -> dict[str, dict[str, Any]]:
    """Get status of all circuit breakers for monitoring."""
    return {key: state.to_dict() for key, state in _circuit_states.items()}


def reset_circuit(config_key: str) -> None:
    """Manually reset circuit for a config (for admin use).

    Args:
        config_key: Config key like "hex8_2p"
    """
    if config_key in _circuit_states:
        old_state = _circuit_states[config_key]
        logger.info(
            f"[PipelineActions] Circuit manually reset for {config_key} "
            f"(was: {old_state.failures} failures, open={old_state.is_open})"
        )
        _circuit_states[config_key] = CircuitState()


class ActionPriority(Enum):
    """Priority levels for pipeline actions."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class StageCompletionResult:
    """Result of a pipeline stage action."""

    success: bool
    stage: str
    iteration: int
    duration_seconds: float = 0.0
    output_path: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for event emission."""
        return {
            "success": self.success,
            "stage": self.stage,
            "iteration": self.iteration,
            "duration_seconds": self.duration_seconds,
            "output_path": self.output_path,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class ActionConfig:
    """Configuration for pipeline actions."""

    # Script paths (relative to ai-service/)
    sync_script: str = "scripts/unified_data_sync.py"  # Unified sync (replaces deprecated scripts)
    export_script: str = "scripts/export_replay_dataset.py"
    train_module: str = "app.training.train"
    evaluate_script: str = "scripts/quick_gauntlet.py"
    promote_script: str = "scripts/auto_promote.py"

    # Default timeouts (seconds) - can be overridden per board type
    sync_timeout: float = 1800.0  # 30 minutes
    export_timeout: float = 3600.0  # 1 hour
    training_timeout: float = 86400.0  # 24 hours
    evaluation_timeout: float = 7200.0  # 2 hours
    promotion_timeout: float = 600.0  # 10 minutes

    # Paths
    data_dir: str = "data"
    models_dir: str = "models"
    training_data_dir: str = "data/training"
    games_dir: str = "data/games"

    # Python executable
    python_executable: str = "python3"


def _get_ai_service_root() -> Path:
    """Get the ai-service root directory."""
    # Try environment variable first
    if ai_service_path := os.environ.get("RINGRIFT_AI_SERVICE_PATH"):
        return Path(ai_service_path)

    # Try finding based on current file location
    current = Path(__file__).resolve()
    # Go up: pipeline_actions.py -> coordination -> app -> ai-service
    for _ in range(3):
        current = current.parent
        if (current / "app").exists() and (current / "scripts").exists():
            return current

    # Fallback to environment variable or raise
    fallback = os.getenv("RINGRIFT_AI_SERVICE_PATH")
    if fallback:
        return Path(fallback)

    raise RuntimeError(
        "Could not locate ai-service directory. Set RINGRIFT_AI_SERVICE_PATH environment variable."
    )


# Maximum output size to capture (prevent memory exhaustion from large outputs)
_MAX_OUTPUT_BYTES = 10 * 1024 * 1024  # 10 MB

# Grace period for SIGTERM before SIGKILL
_SIGTERM_GRACE_SECONDS = 5.0

# Board-type specific timeout multipliers (relative to default)
# Larger boards take longer to process
_BOARD_TYPE_TIMEOUT_MULTIPLIERS: dict[str, dict[str, float]] = {
    # Small boards - faster processing
    "hex8": {
        "export": 0.5,      # 30 min default → 15 min
        "training": 0.5,    # 24h default → 12h
        "evaluation": 0.5,  # 2h default → 1h
    },
    "square8": {
        "export": 0.75,     # 30 min default → 22.5 min
        "training": 0.75,   # 24h default → 18h
        "evaluation": 0.75, # 2h default → 1.5h
    },
    # Large boards - need more time
    "square19": {
        "export": 2.0,      # 30 min default → 60 min
        "training": 1.5,    # 24h default → 36h
        "evaluation": 2.0,  # 2h default → 4h
    },
    "hexagonal": {
        "export": 2.0,      # 30 min default → 60 min
        "training": 1.5,    # 24h default → 36h
        "evaluation": 2.0,  # 2h default → 4h
    },
}


def get_timeout_for_board(
    board_type: str,
    stage: str,
    default_timeout: float,
) -> float:
    """Get timeout adjusted for board type complexity.

    Larger boards (square19, hexagonal) need more time for export/training.
    Smaller boards (hex8) can use shorter timeouts.

    Args:
        board_type: Board type (hex8, square8, square19, hexagonal)
        stage: Pipeline stage (export, training, evaluation, sync, promotion)
        default_timeout: Default timeout from ActionConfig

    Returns:
        Adjusted timeout in seconds
    """
    multipliers = _BOARD_TYPE_TIMEOUT_MULTIPLIERS.get(board_type, {})
    multiplier = multipliers.get(stage, 1.0)
    adjusted = default_timeout * multiplier

    # Also check environment override
    env_key = f"RINGRIFT_{stage.upper()}_TIMEOUT"
    if env_val := os.environ.get(env_key):
        try:
            adjusted = float(env_val)
        except ValueError:
            pass

    return adjusted


async def _run_subprocess(
    cmd: list[str],
    timeout: float,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    max_output_bytes: int = _MAX_OUTPUT_BYTES,
) -> tuple[int, str, str]:
    """Run a subprocess asynchronously with timeout and graceful termination.

    Phase 8 (December 2025): Improved error handling with SIGTERM→SIGKILL
    escalation and output size limits to prevent memory exhaustion.

    Args:
        cmd: Command and arguments
        timeout: Timeout in seconds
        cwd: Working directory
        env: Environment variables (merged with current env)
        max_output_bytes: Maximum bytes to capture from stdout/stderr

    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    logger.info(f"[PipelineActions] Running: {' '.join(cmd)}")

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=full_env,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout,
        )

        # Truncate output if too large
        if len(stdout) > max_output_bytes:
            stdout = stdout[:max_output_bytes] + b"\n... [output truncated]"
        if len(stderr) > max_output_bytes:
            stderr = stderr[:max_output_bytes] + b"\n... [output truncated]"

        return (
            process.returncode or 0,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )

    except asyncio.TimeoutError:
        # Phase 8: Graceful termination - SIGTERM first, then SIGKILL
        logger.warning(
            f"[PipelineActions] Process timed out after {timeout}s, "
            f"sending SIGTERM (PID: {process.pid})"
        )

        try:
            process.terminate()  # SIGTERM - gives process chance to cleanup

            # Wait for graceful shutdown
            try:
                await asyncio.wait_for(
                    process.wait(),
                    timeout=_SIGTERM_GRACE_SECONDS,
                )
                logger.info(
                    f"[PipelineActions] Process {process.pid} terminated gracefully"
                )
            except asyncio.TimeoutError:
                # Process didn't respond to SIGTERM, force kill
                logger.warning(
                    f"[PipelineActions] Process {process.pid} did not respond to SIGTERM "
                    f"after {_SIGTERM_GRACE_SECONDS}s, sending SIGKILL"
                )
                process.kill()
                await process.wait()

        except ProcessLookupError:
            # Process already exited
            pass

        return (-1, "", f"Process timed out after {timeout}s (killed)")


async def trigger_data_sync(
    board_type: str,
    num_players: int,
    iteration: int,
    config: ActionConfig | None = None,
    hosts: list[str] | None = None,
) -> StageCompletionResult:
    """Trigger data synchronization from cluster nodes.

    Syncs game databases from remote cluster nodes to local storage.

    Args:
        board_type: Board type (e.g., "hex8", "square8")
        num_players: Number of players (2, 3, 4)
        iteration: Pipeline iteration number
        config: Action configuration
        hosts: Specific hosts to sync from (None = all)

    Returns:
        StageCompletionResult with sync results
    """
    config = config or ActionConfig()
    root = _get_ai_service_root()
    start_time = time.time()

    try:
        # Dec 29, 2025: unified_data_sync.py doesn't accept --board-type/--num-players
        # It syncs all data, not config-specific. Use --once for one-shot sync.
        cmd = [
            config.python_executable,
            str(root / config.sync_script),
            "--once",  # Run once and exit
        ]

        if hosts:
            cmd.extend(["--hosts", ",".join(hosts)])

        exit_code, stdout, stderr = await _run_subprocess(
            cmd,
            timeout=config.sync_timeout,
            cwd=root,
            env={"PYTHONPATH": str(root)},
        )

        duration = time.time() - start_time
        success = exit_code == 0

        result = StageCompletionResult(
            success=success,
            stage="data_sync",
            iteration=iteration,
            duration_seconds=duration,
            error=stderr if not success else None,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            metadata={
                "board_type": board_type,
                "num_players": num_players,
                "hosts": hosts,
            },
        )

        if success:
            logger.info(
                f"[PipelineActions] Data sync completed in {duration:.1f}s "
                f"for {board_type}_{num_players}p"
            )
            await _emit_sync_complete(result)
        else:
            logger.error(f"[PipelineActions] Data sync failed: {stderr[:500]}")

        return result

    except Exception as e:
        duration = time.time() - start_time
        logger.exception(f"[PipelineActions] Data sync error: {e}")
        return StageCompletionResult(
            success=False,
            stage="data_sync",
            iteration=iteration,
            duration_seconds=duration,
            error=str(e),
        )


async def trigger_npz_export(
    board_type: str,
    num_players: int,
    iteration: int,
    config: ActionConfig | None = None,
    use_discovery: bool = True,
    require_completed: bool = True,
    min_moves: int = 10,
) -> StageCompletionResult:
    """Trigger NPZ export from game databases.

    Exports training data from SQLite game databases to NPZ format.

    Args:
        board_type: Board type (e.g., "hex8", "square8")
        num_players: Number of players (2, 3, 4)
        iteration: Pipeline iteration number
        config: Action configuration
        use_discovery: Use GameDiscovery to find databases
        require_completed: Only export completed games
        min_moves: Minimum move count per game

    Returns:
        StageCompletionResult with export results
    """
    config = config or ActionConfig()
    root = _get_ai_service_root()
    start_time = time.time()

    # Output path
    output_filename = f"{board_type}_{num_players}p_iter{iteration}.npz"
    output_path = root / config.training_data_dir / output_filename

    try:
        cmd = [
            config.python_executable,
            str(root / config.export_script),
            "--board-type", board_type,
            "--num-players", str(num_players),
            "--output", str(output_path),
        ]

        if use_discovery:
            cmd.append("--use-discovery")
        if require_completed:
            cmd.append("--require-completed")
        if min_moves > 0:
            cmd.extend(["--min-moves", str(min_moves)])

        # January 2026: Allow non-canonical databases for export
        # The canonical source validation is too strict for cluster nodes that
        # generate selfplay data in databases not registered in TRAINING_DATA_REGISTRY.md
        cmd.append("--allow-noncanonical")

        # December 29, 2025: Enable pending_gate bypass for cluster nodes without npx
        # This ensures training data is exported even when parity gates are pending
        exit_code, stdout, stderr = await _run_subprocess(
            cmd,
            timeout=config.export_timeout,
            cwd=root,
            env={
                "PYTHONPATH": str(root),
                "RINGRIFT_ALLOW_PENDING_GATE": "true",
            },
        )

        duration = time.time() - start_time
        success = exit_code == 0 and output_path.exists()

        # Parse sample count from output if available
        samples_exported = 0
        for line in stdout.split("\n"):
            if "samples" in line.lower() and "exported" in line.lower():
                try:
                    # Try to extract number from line like "Exported 12345 samples"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.lower() == "exported" and i + 1 < len(parts):
                            samples_exported = int(parts[i + 1].replace(",", ""))
                            break
                except (ValueError, IndexError):
                    pass

        result = StageCompletionResult(
            success=success,
            stage="npz_export",
            iteration=iteration,
            duration_seconds=duration,
            output_path=str(output_path) if success else None,
            error=stderr if not success else None,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            metadata={
                "board_type": board_type,
                "num_players": num_players,
                "samples_exported": samples_exported,
                "use_discovery": use_discovery,
            },
        )

        if success:
            logger.info(
                f"[PipelineActions] NPZ export completed in {duration:.1f}s: "
                f"{samples_exported} samples -> {output_path}"
            )
            await _emit_npz_export_complete(result)
        else:
            logger.error(f"[PipelineActions] NPZ export failed: {stderr[:500]}")

        return result

    except Exception as e:
        duration = time.time() - start_time
        logger.exception(f"[PipelineActions] NPZ export error: {e}")
        return StageCompletionResult(
            success=False,
            stage="npz_export",
            iteration=iteration,
            duration_seconds=duration,
            error=str(e),
        )


async def trigger_npz_combination(
    board_type: str,
    num_players: int,
    iteration: int,
    config: ActionConfig | None = None,
    freshness_weight: float = 1.5,
    min_quality_score: float = 0.2,
) -> StageCompletionResult:
    """Trigger NPZ combination with quality-weighted weighting.

    Combines all available NPZ files for the config into a single
    combined file using quality and freshness weighting.

    Args:
        board_type: Board type (e.g., "hex8", "square8")
        num_players: Number of players (2, 3, 4)
        iteration: Pipeline iteration number
        config: Action configuration
        freshness_weight: Weight for fresher data (default: 1.5)
        min_quality_score: Minimum quality score threshold (default: 0.2)

    Returns:
        StageCompletionResult with combination results
    """
    start_time = time.time()
    config_key = make_config_key(board_type, num_players)

    try:
        from app.training.npz_combiner import (
            NPZCombinerConfig,
            discover_and_combine_for_config,
        )

        root = _get_ai_service_root()
        output_path = root / "data" / "training" / f"{config_key}_combined.npz"

        combiner_config = NPZCombinerConfig(
            freshness_weight=freshness_weight,
            min_quality_score=min_quality_score,
            deduplicate=True,
        )

        # Run combination
        result = discover_and_combine_for_config(
            config_key=config_key,
            output_path=output_path,
            combiner_config=combiner_config,
        )

        duration = time.time() - start_time

        if result.success:
            logger.info(
                f"[PipelineActions] NPZ combination completed in {duration:.1f}s: "
                f"{result.total_samples} samples -> {output_path}"
            )

            # Emit completion event
            await safe_emit_event_async(
                "npz_combination_complete",
                {
                    "config_key": config_key,
                    "output_path": str(output_path),
                    "total_samples": result.total_samples,
                    "samples_by_source": result.samples_by_source,
                    "source": "pipeline_actions",
                },
                context="PipelineActions",
            )

            return StageCompletionResult(
                success=True,
                stage="npz_combination",
                iteration=iteration,
                duration_seconds=duration,
                output_path=str(output_path),
                metadata={
                    "board_type": board_type,
                    "num_players": num_players,
                    "total_samples": result.total_samples,
                    "samples_by_source": result.samples_by_source,
                },
            )
        else:
            logger.warning(f"[PipelineActions] NPZ combination failed: {result.error}")

            # Emit failure event
            await safe_emit_event_async(
                "npz_combination_failed",
                {
                    "config_key": config_key,
                    "error": result.error,
                    "source": "pipeline_actions",
                },
                context="PipelineActions",
            )

            return StageCompletionResult(
                success=False,
                stage="npz_combination",
                iteration=iteration,
                duration_seconds=duration,
                error=result.error,
            )

    except Exception as e:
        duration = time.time() - start_time
        logger.exception(f"[PipelineActions] NPZ combination error: {e}")
        return StageCompletionResult(
            success=False,
            stage="npz_combination",
            iteration=iteration,
            duration_seconds=duration,
            error=str(e),
        )


async def trigger_training(
    board_type: str,
    num_players: int,
    npz_path: str,
    iteration: int,
    config: ActionConfig | None = None,
    batch_size: int = 512,
    epochs: int = 50,
    early_stopping: bool = True,
    init_weights: str | None = None,
    max_retries: int = 2,
    retry_delay: float = 30.0,
    architecture: str | None = None,
) -> StageCompletionResult:
    """Trigger neural network training with retry logic.

    Trains a new model checkpoint using exported NPZ data.
    On failure, retries with reduced batch size to handle OOM issues.

    Args:
        board_type: Board type (e.g., "hex8", "square8")
        num_players: Number of players (2, 3, 4)
        npz_path: Path to NPZ training data
        iteration: Pipeline iteration number
        config: Action configuration
        batch_size: Training batch size
        epochs: Maximum epochs
        early_stopping: Enable early stopping
        init_weights: Path to initial weights (for transfer learning)
        max_retries: Maximum retry attempts (default: 2)
        retry_delay: Delay between retries in seconds (default: 30)
        architecture: Architecture version (e.g., "v5", "v6"). If None, selects
            based on ArchitectureTracker allocation weights (Phase 7).

    Returns:
        StageCompletionResult with training results
    """
    config = config or ActionConfig()
    root = _get_ai_service_root()
    start_time = time.time()
    last_error = None
    config_key = make_config_key(board_type, num_players)

    # Phase 4.3 Dec 29, 2025: Check circuit breaker before starting
    if not check_circuit(config_key):
        return StageCompletionResult(
            success=False,
            stage="training",
            iteration=iteration,
            duration_seconds=0.0,
            error=f"Circuit open for {config_key} - too many recent failures",
            metadata={"circuit_blocked": True, "config_key": config_key},
        )

    # Dec 29, 2025 - Phase 3: Quality gate blocking training
    # Don't waste GPU cycles on low-quality data
    quality_ok, quality_score = await _check_training_data_quality(config_key, npz_path)
    if not quality_ok:
        # Emit event so pipeline can regenerate data (January 2026 - migrated to event_router)
        await safe_emit_event_async(
            "TRAINING_BLOCKED_BY_QUALITY",
            {
                "config_key": config_key,
                "npz_path": npz_path,
                "quality_score": quality_score,
                "threshold": QUALITY_GATE_THRESHOLD,
            },
            context="pipeline_actions",
        )

        logger.warning(
            f"[PipelineActions] Training blocked by quality gate: "
            f"{config_key} quality={quality_score:.2f} < threshold={QUALITY_GATE_THRESHOLD}"
        )
        return StageCompletionResult(
            success=False,
            stage="training",
            iteration=iteration,
            duration_seconds=0.0,
            error=f"Quality gate failed: {quality_score:.2f} < {QUALITY_GATE_THRESHOLD}",
            metadata={
                "quality_blocked": True,
                "config_key": config_key,
                "quality_score": quality_score,
                "threshold": QUALITY_GATE_THRESHOLD,
            },
        )

    # Emit training triggered event (January 2026 - migrated to event_router)
    if HAS_EVENT_EMITTERS:
        job_id = f"train_{config_key}_iter{iteration}"
        await safe_emit_event_async(
            "TRAINING_TRIGGERED",
            {
                "config": config_key,
                "job_id": job_id,
                "trigger_reason": "pipeline",
                "priority": "normal",
                "iteration": iteration,
                "epochs": epochs,
                "batch_size": batch_size,
            },
            context="pipeline_actions",
        )

    # Phase 7: Architecture selection based on ArchitectureTracker weights
    selected_architecture = architecture
    if selected_architecture is None:
        try:
            from app.training.architecture_tracker import get_allocation_weights
            import random

            weights = get_allocation_weights(board_type, num_players)
            if weights:
                # Weighted random selection based on allocation weights
                architectures = list(weights.keys())
                weight_values = list(weights.values())
                selected_architecture = random.choices(
                    architectures, weights=weight_values, k=1
                )[0]
                logger.info(
                    f"[PipelineActions] Selected architecture {selected_architecture} "
                    f"from weights: {weights}"
                )
            else:
                # Default to v5 if no tracking data
                selected_architecture = "v5"
                logger.debug(
                    f"[PipelineActions] No architecture weights for {config_key}, "
                    f"defaulting to {selected_architecture}"
                )
        except ImportError:
            selected_architecture = "v5"
            logger.debug("[PipelineActions] ArchitectureTracker not available, using v5")

    # Output model path (include architecture in filename for multi-arch)
    if selected_architecture and selected_architecture != "v5":
        model_filename = f"{board_type}_{num_players}p_{selected_architecture}_iter{iteration}.pth"
    else:
        model_filename = f"{board_type}_{num_players}p_iter{iteration}.pth"
    model_path = root / config.models_dir / model_filename

    # December 30, 2025: Use centralized RetryConfig for consistency
    retry_config = RetryConfig(
        max_attempts=max_retries,
        base_delay=retry_delay,
        max_delay=retry_delay * 4,  # Allow up to 4x base delay
        exponential=False,  # Use fixed delay for OOM recovery
        jitter=0.1,  # Add 10% jitter
    )

    for attempt in retry_config.attempts():
        # Reduce batch size on retry to handle OOM issues (0-indexed for calculation)
        attempt_idx = attempt.number - 1
        retry_batch_size = batch_size // (2 ** attempt_idx)
        retry_batch_size = max(64, retry_batch_size)  # Minimum batch size

        if not attempt.is_first:
            logger.info(
                f"[PipelineActions] Training retry {attempt.number}/{retry_config.max_attempts} "
                f"with batch_size={retry_batch_size}"
            )
            await attempt.wait_async()

        try:
            cmd = [
                config.python_executable,
                "-m", config.train_module,
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--data-path", npz_path,
                "--save-path", str(model_path),
                "--batch-size", str(retry_batch_size),
                "--epochs", str(epochs),
            ]

            if early_stopping:
                cmd.append("--early-stopping")
            if init_weights:
                cmd.extend(["--init-weights", init_weights])
            if selected_architecture:
                cmd.extend(["--model-version", selected_architecture])

            # December 29, 2025: Enable pending_gate bypass for training data access
            exit_code, stdout, stderr = await _run_subprocess(
                cmd,
                timeout=config.training_timeout,
                cwd=root,
                env={
                    "PYTHONPATH": str(root),
                    "RINGRIFT_ALLOW_PENDING_GATE": "true",
                },
            )

            attempt_duration = time.time() - start_time
            success = exit_code == 0 and model_path.exists()

            # Parse training metrics from output
            train_loss = 0.0
            val_loss = 0.0
            policy_accuracy = 0.0
            for line in stdout.split("\n"):
                line_lower = line.lower()
                if "train_loss" in line_lower:
                    try:
                        train_loss = float(line.split(":")[-1].strip())
                    except ValueError:
                        pass
                elif "val_loss" in line_lower:
                    try:
                        val_loss = float(line.split(":")[-1].strip())
                    except ValueError:
                        pass
                elif "policy_accuracy" in line_lower or "policy acc" in line_lower:
                    try:
                        policy_accuracy = float(line.split(":")[-1].strip().replace("%", ""))
                    except ValueError:
                        pass

            result = StageCompletionResult(
                success=success,
                stage="training",
                iteration=iteration,
                duration_seconds=attempt_duration,
                output_path=str(model_path) if success else None,
                error=stderr if not success else None,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                metadata={
                    "board_type": board_type,
                    "num_players": num_players,
                    "model_id": model_filename,
                    "architecture": selected_architecture,  # Phase 7
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "policy_accuracy": policy_accuracy,
                    "epochs": epochs,
                    "batch_size": retry_batch_size,
                    "attempt": attempt.number,
                },
            )

            if success:
                logger.info(
                    f"[PipelineActions] Training completed in {attempt_duration:.1f}s: "
                    f"val_loss={val_loss:.4f}, policy_acc={policy_accuracy:.1f}%"
                )
                record_action_success(config_key)  # Phase 4.3: Reset circuit on success
                await _emit_training_complete(result)
                return result
            else:
                last_error = stderr[:500]
                logger.warning(
                    f"[PipelineActions] Training attempt {attempt.number} failed: {last_error}"
                )
                # Continue to next retry if not last attempt
                if attempt.should_retry:
                    continue
                # Last attempt - emit failure and return
                logger.error(f"[PipelineActions] All {retry_config.max_attempts} training attempts failed")
                record_action_failure(config_key, last_error)  # Phase 4.3: Track failure
                await _emit_training_failed(result)
                return result

        except Exception as e:
            last_error = str(e)
            logger.warning(f"[PipelineActions] Training attempt {attempt.number} error: {e}")
            # Continue to next retry if not last attempt
            if attempt.should_retry:
                continue

    # All retries exhausted with exceptions
    duration = time.time() - start_time
    logger.error(f"[PipelineActions] All {retry_config.max_attempts} training attempts failed with errors")
    record_action_failure(config_key, last_error)  # Phase 4.3: Track failure
    result = StageCompletionResult(
        success=False,
        stage="training",
        iteration=iteration,
        duration_seconds=duration,
        error=last_error or "All retries failed",
    )
    await _emit_training_failed(result)
    return result


async def _dispatch_evaluation_to_cluster(
    model_path: str,
    board_type: str,
    num_players: int,
    iteration: int,
    num_games: int,
    baselines: list[str] | None,
) -> StageCompletionResult:
    """Dispatch evaluation to cluster node via work queue.

    January 27, 2026: Added to prevent coordinator nodes from running
    heavy evaluation workloads locally. Coordinators dispatch work
    to GPU cluster nodes instead.
    """
    start_time = time.time()
    config_key = make_config_key(board_type, num_players)

    try:
        from app.coordination.work_distributor import (
            get_work_distributor,
            DistributedWorkConfig,
        )

        distributor = get_work_distributor()
        # January 27, 2026: Use priority=85 for gauntlets so they're claimed
        # before most selfplay (50) but after critical training (100)
        config = DistributedWorkConfig(priority=85, require_gpu=True)
        work_id = await distributor.submit_evaluation(
            candidate_model=model_path,
            baseline_model=None,
            games=num_games * len(baselines or ["random", "heuristic"]),
            board=board_type,
            num_players=num_players,
            evaluation_type="gauntlet",
            config=config,
        )

        if work_id:
            logger.info(
                f"[PipelineActions] Dispatched evaluation to cluster: {work_id} "
                f"for {model_path}"
            )
            return StageCompletionResult(
                success=True,
                stage="evaluation",
                iteration=iteration,
                duration_seconds=time.time() - start_time,
                metadata={
                    "dispatched_to_cluster": True,
                    "work_id": work_id,
                    "config_key": config_key,
                },
            )
        else:
            logger.warning(
                f"[PipelineActions] Failed to dispatch evaluation to cluster: {model_path}"
            )
            return StageCompletionResult(
                success=False,
                stage="evaluation",
                iteration=iteration,
                duration_seconds=time.time() - start_time,
                error="dispatch_failed",
                metadata={"config_key": config_key},
            )

    except ImportError:
        logger.warning(
            "[PipelineActions] WorkDistributor not available, cannot dispatch to cluster"
        )
        return StageCompletionResult(
            success=False,
            stage="evaluation",
            iteration=iteration,
            duration_seconds=time.time() - start_time,
            error="work_distributor_unavailable",
        )
    except (OSError, RuntimeError) as e:
        logger.error(f"[PipelineActions] Dispatch to cluster failed: {e}")
        return StageCompletionResult(
            success=False,
            stage="evaluation",
            iteration=iteration,
            duration_seconds=time.time() - start_time,
            error=str(e),
        )


async def trigger_evaluation(
    model_path: str,
    board_type: str,
    num_players: int,
    iteration: int = 0,
    config: ActionConfig | None = None,
    num_games: int = 100,
    baselines: list[str] | None = None,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> StageCompletionResult:
    """Trigger gauntlet evaluation of a trained model with retry logic.

    Evaluates a model against baseline opponents to measure strength.
    On failure, retries with increased game count to reduce variance.

    Args:
        model_path: Path to model checkpoint
        board_type: Board type (e.g., "hex8", "square8")
        num_players: Number of players (2, 3, 4)
        iteration: Pipeline iteration number
        config: Action configuration
        num_games: Base games per opponent
        baselines: Baseline opponents to test against
        max_retries: Maximum retry attempts (default: 3)
        retry_delay: Delay between retries in seconds

    Returns:
        StageCompletionResult with evaluation results
    """
    config = config or ActionConfig()
    root = _get_ai_service_root()
    start_time = time.time()
    baselines = baselines or ["random", "heuristic"]
    last_error = None
    config_key = make_config_key(board_type, num_players)

    # Phase 4.3 Dec 29, 2025: Check circuit breaker before starting
    if not check_circuit(config_key):
        return StageCompletionResult(
            success=False,
            stage="evaluation",
            iteration=iteration,
            duration_seconds=0.0,
            error=f"Circuit open for {config_key} - too many recent failures",
            metadata={"circuit_blocked": True, "config_key": config_key},
        )

    # January 27, 2026: Check if this node should run evaluations locally
    # Coordinators dispatch to cluster nodes instead of running locally
    from app.config.env import env
    if not env.gauntlet_enabled:
        logger.info(
            f"[PipelineActions] Gauntlet disabled on this node, dispatching to cluster: {model_path}"
        )
        return await _dispatch_evaluation_to_cluster(
            model_path, board_type, num_players, iteration, num_games, baselines
        )

    # December 30, 2025: Use centralized RetryConfig for consistency
    retry_config = RetryConfig(
        max_attempts=max_retries,
        base_delay=retry_delay,
        max_delay=retry_delay * 4,  # Allow up to 4x base delay
        exponential=True,  # Use exponential backoff for transient failures
        jitter=0.1,  # Add 10% jitter
    )

    for attempt in retry_config.attempts():
        # Increase games on retry to reduce variance (0-indexed for calculation)
        attempt_idx = attempt.number - 1
        retry_games = int(num_games * (1 + 0.5 * attempt_idx))

        if not attempt.is_first:
            logger.info(
                f"[PipelineActions] Evaluation retry {attempt.number}/{retry_config.max_attempts} "
                f"with {retry_games} games per opponent"
            )
            await attempt.wait_async()

        try:
            cmd = [
                config.python_executable,
                config.evaluate_script,
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--model", model_path,
                "--games", str(retry_games),
            ]

            exit_code, stdout, stderr = await _run_subprocess(
                cmd,
                timeout=config.evaluation_timeout,
                cwd=root,
                env={"PYTHONPATH": str(root)},
            )

            attempt_duration = time.time() - start_time
            success = exit_code == 0

            # Parse evaluation results
            win_rates = {}
            elo_delta = 0.0
            for line in stdout.split("\n"):
                line_lower = line.lower()
                # Parse win rates like "vs random: 95.0%"
                if "vs " in line_lower and "%" in line:
                    try:
                        parts = line.split(":")
                        opponent = parts[0].split("vs")[-1].strip()
                        rate_str = parts[1].strip().replace("%", "")
                        win_rates[opponent] = float(rate_str)
                    except (ValueError, IndexError):
                        pass
                # Parse elo delta
                if "elo" in line_lower and ("delta" in line_lower or "+" in line):
                    try:
                        elo_delta = float(line.split(":")[-1].strip().replace("+", ""))
                    except ValueError:
                        pass

            # Check promotion eligibility (win_rates are integer percentages, thresholds are 0-1)
            eligible = all([
                win_rates.get("random", 0) >= PRODUCTION_MIN_WIN_RATE_VS_RANDOM * 100,
                win_rates.get("heuristic", 0) >= PRODUCTION_MIN_WIN_RATE_VS_HEURISTIC * 100,
            ]) if win_rates else False

            result = StageCompletionResult(
                success=success,
                stage="evaluation",
                iteration=iteration,
                duration_seconds=attempt_duration,
                output_path=model_path,
                error=stderr if not success else None,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                metadata={
                    "board_type": board_type,
                    "num_players": num_players,
                    "model_path": model_path,
                    "win_rates": win_rates,
                    "elo_delta": elo_delta,
                    "num_games": retry_games,
                    "promotion_eligible": eligible,
                    "attempt": attempt.number,
                },
            )

            if success:
                logger.info(
                    f"[PipelineActions] Evaluation completed in {attempt_duration:.1f}s: "
                    f"win_rates={win_rates}, elo_delta={elo_delta:+.0f}"
                )
                record_action_success(config_key)  # Phase 4.3: Reset circuit on success
                await _emit_evaluation_complete(result)
                return result
            else:
                last_error = stderr[:500]
                logger.warning(
                    f"[PipelineActions] Evaluation attempt {attempt.number} failed: {last_error}"
                )
                # Continue to next retry if not last attempt
                if attempt.should_retry:
                    continue
                # Last attempt - return failure result
                logger.error(f"[PipelineActions] All {retry_config.max_attempts} evaluation attempts failed")
                record_action_failure(config_key, last_error)  # Phase 4.3: Track failure
                return result

        except Exception as e:
            last_error = str(e)
            logger.warning(f"[PipelineActions] Evaluation attempt {attempt.number} error: {e}")
            # Continue to next retry if not last attempt
            if attempt.should_retry:
                continue

    # All retries exhausted with exceptions
    duration = time.time() - start_time
    logger.error(f"[PipelineActions] All {retry_config.max_attempts} evaluation attempts failed with errors")
    record_action_failure(config_key, last_error)  # Phase 4.3: Track failure
    return StageCompletionResult(
        success=False,
        stage="evaluation",
        iteration=iteration,
        duration_seconds=duration,
        error=last_error or "All retries failed",
    )


async def trigger_promotion(
    model_path: str,
    gauntlet_results: dict[str, Any],
    board_type: str,
    num_players: int,
    iteration: int = 0,
    config: ActionConfig | None = None,
    sync_to_cluster: bool = True,
) -> StageCompletionResult:
    """Trigger model promotion if gauntlet passed.

    Promotes a model to canonical status and syncs to cluster.

    Args:
        model_path: Path to model checkpoint
        gauntlet_results: Results from gauntlet evaluation
        board_type: Board type (e.g., "hex8", "square8")
        num_players: Number of players (2, 3, 4)
        iteration: Pipeline iteration number
        config: Action configuration
        sync_to_cluster: Sync promoted model to cluster nodes

    Returns:
        StageCompletionResult with promotion results
    """
    config = config or ActionConfig()
    root = _get_ai_service_root()
    start_time = time.time()

    # Check eligibility from gauntlet results (win_rates are integer percentages)
    win_rates = gauntlet_results.get("win_rates", {})
    eligible = (
        win_rates.get("random", 0) >= PRODUCTION_MIN_WIN_RATE_VS_RANDOM * 100 and
        win_rates.get("heuristic", 0) >= PRODUCTION_MIN_WIN_RATE_VS_HEURISTIC * 100
    )

    if not eligible:
        return StageCompletionResult(
            success=True,
            stage="promotion",
            iteration=iteration,
            duration_seconds=0.0,
            error=None,
            metadata={
                "promoted": False,
                "reason": "Did not meet promotion thresholds",
                "win_rates": win_rates,
            },
        )

    try:
        cmd = [
            config.python_executable,
            str(root / config.promote_script),
            "--model", model_path,
            "--board-type", board_type,
            "--num-players", str(num_players),
        ]

        if sync_to_cluster:
            cmd.append("--sync-to-cluster")

        exit_code, stdout, stderr = await _run_subprocess(
            cmd,
            timeout=config.promotion_timeout,
            cwd=root,
            env={"PYTHONPATH": str(root)},
        )

        duration = time.time() - start_time
        success = exit_code == 0

        # Determine canonical path
        canonical_name = f"canonical_{board_type}_{num_players}p.pth"
        canonical_path = root / config.models_dir / canonical_name

        result = StageCompletionResult(
            success=success,
            stage="promotion",
            iteration=iteration,
            duration_seconds=duration,
            output_path=str(canonical_path) if success else None,
            error=stderr if not success else None,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            metadata={
                "board_type": board_type,
                "num_players": num_players,
                "promoted": success,
                "reason": "Gauntlet passed" if success else stderr,
                "win_rates": win_rates,
                "synced_to_cluster": sync_to_cluster and success,
            },
        )

        if success:
            logger.info(
                f"[PipelineActions] Promotion completed in {duration:.1f}s: "
                f"{model_path} -> {canonical_path}"
            )
            await _emit_promotion_complete(result)
        else:
            logger.error(f"[PipelineActions] Promotion failed: {stderr[:500]}")

        return result

    except Exception as e:
        duration = time.time() - start_time
        logger.exception(f"[PipelineActions] Promotion error: {e}")
        return StageCompletionResult(
            success=False,
            stage="promotion",
            iteration=iteration,
            duration_seconds=duration,
            error=str(e),
        )


# =============================================================================
# Event Emission Helpers
# =============================================================================


async def _emit_sync_complete(result: StageCompletionResult) -> None:
    """Emit SYNC_COMPLETE event (January 2026 - migrated to event_router)."""
    payload = {
        "sync_type": "data",
        "items_synced": result.metadata.get("files_synced", 0),
        "success": result.success,
        "duration_seconds": result.duration_seconds,
        "source": "pipeline_actions",
        "iteration": result.iteration,
        **result.metadata,
    }
    await safe_emit_event_async("SYNC_COMPLETE", payload, context="pipeline_actions")


async def _emit_npz_export_complete(result: StageCompletionResult) -> None:
    """Emit NPZ_EXPORT_COMPLETE event (January 2026 - migrated to event_router)."""
    extra_meta = {k: v for k, v in result.metadata.items()
                  if k not in ("board_type", "num_players", "samples_exported", "games_exported")}
    payload = {
        "board_type": result.metadata.get("board_type", "unknown"),
        "num_players": result.metadata.get("num_players", 2),
        "samples_exported": result.metadata.get("samples_exported", 0),
        "games_exported": result.metadata.get("games_exported", 0),
        "output_path": result.output_path or "",
        "success": result.success,
        "duration_seconds": result.duration_seconds,
        "iteration": result.iteration,
        **extra_meta,
    }
    await safe_emit_event_async("NPZ_EXPORT_COMPLETE", payload, context="pipeline_actions")


async def _emit_training_complete(result: StageCompletionResult) -> None:
    """Emit TRAINING_COMPLETE event (January 2026 - migrated to event_router)."""
    extra_meta = {k: v for k, v in result.metadata.items()
                  if k not in ("model_id", "board_type", "num_players", "val_loss", "epochs_completed")}
    payload = {
        "job_id": result.metadata.get("model_id", f"training_iter{result.iteration}"),
        "board_type": result.metadata.get("board_type", "unknown"),
        "num_players": result.metadata.get("num_players", 2),
        "success": result.success,
        "final_loss": result.metadata.get("val_loss"),
        "model_path": result.output_path,
        "epochs_completed": result.metadata.get("epochs_completed", 0),
        "iteration": result.iteration,
        **extra_meta,
    }
    await safe_emit_event_async("TRAINING_COMPLETE", payload, context="pipeline_actions")


async def _emit_training_failed(result: StageCompletionResult) -> None:
    """Emit TRAINING_FAILED event (January 2026 - migrated to event_router)."""
    extra_meta = {k: v for k, v in result.metadata.items()
                  if k not in ("model_id", "board_type", "num_players")}
    payload = {
        "job_id": result.metadata.get("model_id", f"training_iter{result.iteration}"),
        "board_type": result.metadata.get("board_type", "unknown"),
        "num_players": result.metadata.get("num_players", 2),
        "success": False,
        "iteration": result.iteration,
        "error": result.error or "Unknown error",
        **extra_meta,
    }
    await safe_emit_event_async("TRAINING_FAILED", payload, context="pipeline_actions")


async def _emit_evaluation_complete(result: StageCompletionResult) -> None:
    """Emit EVALUATION_COMPLETE event (January 2026 - migrated to event_router)."""
    extra_meta = {k: v for k, v in result.metadata.items()
                  if k not in ("model_id", "board_type", "num_players", "win_rates", "elo_delta", "games_played")}
    payload = {
        "model_id": result.metadata.get("model_id", f"eval_iter{result.iteration}"),
        "board_type": result.metadata.get("board_type", "unknown"),
        "num_players": result.metadata.get("num_players", 2),
        "success": result.success,
        "win_rate": result.metadata.get("win_rates", {}).get("heuristic"),
        "elo_delta": result.metadata.get("elo_delta"),
        "games_played": result.metadata.get("games_played", 0),
        "iteration": result.iteration,
        **extra_meta,
    }
    await safe_emit_event_async("EVALUATION_COMPLETE", payload, context="pipeline_actions")


async def _emit_promotion_complete(result: StageCompletionResult) -> None:
    """Emit PROMOTION_COMPLETE event (January 2026 - migrated to event_router)."""
    # Extract required parameters from metadata
    board_type = result.metadata.get("board_type", "unknown")
    num_players = result.metadata.get("num_players", 2)
    model_path = result.output_path or result.metadata.get("model_path", "")

    # Generate model_id from path or metadata
    if model_path:
        model_id = Path(model_path).stem
    else:
        model_id = f"{board_type}_{num_players}p_iter{result.iteration}"

    payload = {
        "model_id": model_id,
        "board_type": board_type,
        "num_players": num_players,
        "promotion_type": "production",
        "elo_improvement": result.metadata.get("elo_delta"),
        "model_path": model_path,
        "promoted": result.metadata.get("promoted", False),
        "promotion_reason": result.metadata.get("reason", ""),
        "iteration": result.iteration,
    }
    await safe_emit_event_async("PROMOTION_COMPLETE", payload, context="pipeline_actions")


__all__ = [
    # Config and result types
    "ActionConfig",
    "ActionPriority",
    "StageCompletionResult",
    # Pipeline action triggers
    "trigger_data_sync",
    "trigger_evaluation",
    "trigger_npz_combination",
    "trigger_npz_export",
    "trigger_promotion",
    "trigger_training",
    # Circuit breaker (Phase 4.3 Dec 29, 2025)
    "CircuitState",
    "check_circuit",
    "get_circuit_status",
    "record_action_failure",
    "record_action_success",
    "reset_circuit",
]
