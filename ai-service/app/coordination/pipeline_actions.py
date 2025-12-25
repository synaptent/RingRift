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
import subprocess
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

logger = logging.getLogger(__name__)


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
    sync_script: str = "scripts/sync_cluster_data.py"
    export_script: str = "scripts/export_replay_dataset.py"
    train_module: str = "app.training.train"
    evaluate_module: str = "app.gauntlet.runner"
    promote_script: str = "scripts/auto_promote.py"

    # Timeouts (seconds)
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

    # Fallback to common location
    return Path("/Users/armand/Development/RingRift/ai-service")


async def _run_subprocess(
    cmd: list[str],
    timeout: float,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """Run a subprocess asynchronously with timeout.

    Args:
        cmd: Command and arguments
        timeout: Timeout in seconds
        cwd: Working directory
        env: Environment variables (merged with current env)

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
        return (
            process.returncode or 0,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return (-1, "", f"Process timed out after {timeout}s")


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
        cmd = [
            config.python_executable,
            str(root / config.sync_script),
            "--board-type", board_type,
            "--num-players", str(num_players),
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

        exit_code, stdout, stderr = await _run_subprocess(
            cmd,
            timeout=config.export_timeout,
            cwd=root,
            env={"PYTHONPATH": str(root)},
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
) -> StageCompletionResult:
    """Trigger neural network training.

    Trains a new model checkpoint using exported NPZ data.

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

    Returns:
        StageCompletionResult with training results
    """
    config = config or ActionConfig()
    root = _get_ai_service_root()
    start_time = time.time()

    # Output model path
    model_filename = f"{board_type}_{num_players}p_iter{iteration}.pth"
    model_path = root / config.models_dir / model_filename

    try:
        cmd = [
            config.python_executable,
            "-m", config.train_module,
            "--board-type", board_type,
            "--num-players", str(num_players),
            "--data-path", npz_path,
            "--save-path", str(model_path),
            "--batch-size", str(batch_size),
            "--epochs", str(epochs),
        ]

        if early_stopping:
            cmd.append("--early-stopping")
        if init_weights:
            cmd.extend(["--init-weights", init_weights])

        exit_code, stdout, stderr = await _run_subprocess(
            cmd,
            timeout=config.training_timeout,
            cwd=root,
            env={"PYTHONPATH": str(root)},
        )

        duration = time.time() - start_time
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
            duration_seconds=duration,
            output_path=str(model_path) if success else None,
            error=stderr if not success else None,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            metadata={
                "board_type": board_type,
                "num_players": num_players,
                "model_id": model_filename,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "policy_accuracy": policy_accuracy,
                "epochs": epochs,
                "batch_size": batch_size,
            },
        )

        if success:
            logger.info(
                f"[PipelineActions] Training completed in {duration:.1f}s: "
                f"val_loss={val_loss:.4f}, policy_acc={policy_accuracy:.1f}%"
            )
            await _emit_training_complete(result)
        else:
            logger.error(f"[PipelineActions] Training failed: {stderr[:500]}")
            await _emit_training_failed(result)

        return result

    except Exception as e:
        duration = time.time() - start_time
        logger.exception(f"[PipelineActions] Training error: {e}")
        result = StageCompletionResult(
            success=False,
            stage="training",
            iteration=iteration,
            duration_seconds=duration,
            error=str(e),
        )
        await _emit_training_failed(result)
        return result


async def trigger_evaluation(
    model_path: str,
    board_type: str,
    num_players: int,
    iteration: int = 0,
    config: ActionConfig | None = None,
    num_games: int = 100,
    baselines: list[str] | None = None,
) -> StageCompletionResult:
    """Trigger gauntlet evaluation of a trained model.

    Evaluates a model against baseline opponents to measure strength.

    Args:
        model_path: Path to model checkpoint
        board_type: Board type (e.g., "hex8", "square8")
        num_players: Number of players (2, 3, 4)
        iteration: Pipeline iteration number
        config: Action configuration
        num_games: Games per opponent
        baselines: Baseline opponents to test against

    Returns:
        StageCompletionResult with evaluation results
    """
    config = config or ActionConfig()
    root = _get_ai_service_root()
    start_time = time.time()
    baselines = baselines or ["random", "heuristic"]

    try:
        cmd = [
            config.python_executable,
            "-m", config.evaluate_module,
            "--board-type", board_type,
            "--num-players", str(num_players),
            "--model-path", model_path,
            "--games", str(num_games),
        ]

        exit_code, stdout, stderr = await _run_subprocess(
            cmd,
            timeout=config.evaluation_timeout,
            cwd=root,
            env={"PYTHONPATH": str(root)},
        )

        duration = time.time() - start_time
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
            duration_seconds=duration,
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
                "num_games": num_games,
                "promotion_eligible": eligible,
            },
        )

        if success:
            logger.info(
                f"[PipelineActions] Evaluation completed in {duration:.1f}s: "
                f"win_rates={win_rates}, elo_delta={elo_delta:+.0f}"
            )
            await _emit_evaluation_complete(result)
        else:
            logger.error(f"[PipelineActions] Evaluation failed: {stderr[:500]}")

        return result

    except Exception as e:
        duration = time.time() - start_time
        logger.exception(f"[PipelineActions] Evaluation error: {e}")
        return StageCompletionResult(
            success=False,
            stage="evaluation",
            iteration=iteration,
            duration_seconds=duration,
            error=str(e),
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
    """Emit SYNC_COMPLETE event."""
    try:
        from app.coordination.event_emitters import emit_sync_complete

        await emit_sync_complete(
            sync_type="data",  # Required: type of sync
            items_synced=result.metadata.get("files_synced", 0),  # Required
            success=result.success,
            duration_seconds=result.duration_seconds,
            source="pipeline_actions",
            iteration=result.iteration,
            **result.metadata,
        )
    except Exception as e:
        logger.warning(f"[PipelineActions] Could not emit sync_complete: {e}")


async def _emit_npz_export_complete(result: StageCompletionResult) -> None:
    """Emit NPZ_EXPORT_COMPLETE event."""
    try:
        from app.coordination.event_emitters import emit_npz_export_complete

        await emit_npz_export_complete(
            board_type=result.metadata.get("board_type", "unknown"),
            num_players=result.metadata.get("num_players", 2),
            samples_exported=result.metadata.get("samples_exported", 0),
            games_exported=result.metadata.get("games_exported", 0),
            output_path=result.output_path or "",
            success=result.success,
            duration_seconds=result.duration_seconds,
            iteration=result.iteration,
            **{k: v for k, v in result.metadata.items()
               if k not in ("board_type", "num_players", "samples_exported", "games_exported")},
        )
    except Exception as e:
        logger.warning(f"[PipelineActions] Could not emit npz_export_complete: {e}")


async def _emit_training_complete(result: StageCompletionResult) -> None:
    """Emit TRAINING_COMPLETE event."""
    try:
        from app.coordination.event_emitters import emit_training_complete

        await emit_training_complete(
            job_id=result.metadata.get("model_id", f"training_iter{result.iteration}"),
            board_type=result.metadata.get("board_type", "unknown"),
            num_players=result.metadata.get("num_players", 2),
            success=result.success,
            final_loss=result.metadata.get("val_loss"),
            model_path=result.output_path,
            epochs_completed=result.metadata.get("epochs_completed", 0),
            iteration=result.iteration,
            **{k: v for k, v in result.metadata.items()
               if k not in ("model_id", "board_type", "num_players", "val_loss", "epochs_completed")},
        )
    except Exception as e:
        logger.warning(f"[PipelineActions] Could not emit training_complete: {e}")


async def _emit_training_failed(result: StageCompletionResult) -> None:
    """Emit TRAINING_FAILED event."""
    try:
        from app.coordination.event_emitters import emit_training_complete

        # Use emit_training_complete with success=False
        await emit_training_complete(
            job_id=result.metadata.get("model_id", f"training_iter{result.iteration}"),
            board_type=result.metadata.get("board_type", "unknown"),
            num_players=result.metadata.get("num_players", 2),
            success=False,
            iteration=result.iteration,
            error=result.error or "Unknown error",
            **{k: v for k, v in result.metadata.items()
               if k not in ("model_id", "board_type", "num_players")},
        )
    except Exception as e:
        logger.warning(f"[PipelineActions] Could not emit training_failed: {e}")


async def _emit_evaluation_complete(result: StageCompletionResult) -> None:
    """Emit EVALUATION_COMPLETE event."""
    try:
        from app.coordination.event_emitters import emit_evaluation_complete

        await emit_evaluation_complete(
            model_id=result.metadata.get("model_id", f"eval_iter{result.iteration}"),
            board_type=result.metadata.get("board_type", "unknown"),
            num_players=result.metadata.get("num_players", 2),
            success=result.success,
            win_rate=result.metadata.get("win_rates", {}).get("heuristic"),
            elo_delta=result.metadata.get("elo_delta"),
            games_played=result.metadata.get("games_played", 0),
            iteration=result.iteration,
            **{k: v for k, v in result.metadata.items()
               if k not in ("model_id", "board_type", "num_players", "win_rates", "elo_delta", "games_played")},
        )
    except Exception as e:
        logger.warning(f"[PipelineActions] Could not emit evaluation_complete: {e}")


async def _emit_promotion_complete(result: StageCompletionResult) -> None:
    """Emit PROMOTION_COMPLETE event."""
    try:
        from app.coordination.event_emitters import emit_promotion_complete

        await emit_promotion_complete(
            iteration=result.iteration,
            promoted=result.metadata.get("promoted", False),
            promotion_reason=result.metadata.get("reason", ""),
            metadata=result.metadata,
        )
    except Exception as e:
        logger.debug(f"[PipelineActions] Could not emit promotion_complete: {e}")


__all__ = [
    "ActionConfig",
    "ActionPriority",
    "StageCompletionResult",
    "trigger_data_sync",
    "trigger_evaluation",
    "trigger_npz_export",
    "trigger_promotion",
    "trigger_training",
]
