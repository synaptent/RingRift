"""Pipeline triggers with prerequisite validation.

This module provides trigger functions that validate prerequisites before
invoking pipeline stages. This prevents wasted work and provides clear
error messages when prerequisites are not met.

Usage:
    from app.coordination.pipeline_triggers import PipelineTrigger

    trigger = PipelineTrigger()

    # After selfplay completes
    result = await trigger.trigger_sync_after_selfplay("hex8", 2)

    # After sync completes
    result = await trigger.trigger_export_after_sync("hex8", 2)

    # After export completes
    result = await trigger.trigger_training_after_export("hex8", 2)
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.coordination.pipeline_actions import (
    ActionConfig,
    StageCompletionResult,
    trigger_data_sync,
    trigger_npz_export,
    trigger_training,
    trigger_evaluation,
    trigger_promotion,
)

logger = logging.getLogger(__name__)


@dataclass
class PrerequisiteResult:
    """Result of a prerequisite check."""

    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed


@dataclass
class TriggerConfig:
    """Configuration for pipeline triggers."""

    # Minimum games required for sync to be worthwhile
    min_games_for_sync: int = 100

    # Minimum samples required for training
    min_samples_for_training: int = 10000

    # Minimum NPZ file size in bytes (sanity check)
    min_npz_size_bytes: int = 100_000  # 100KB

    # Minimum win rate vs random for promotion
    min_win_rate_vs_random: float = 0.85

    # Minimum win rate vs heuristic for promotion
    min_win_rate_vs_heuristic: float = 0.60

    # Root directory for ai-service
    ai_service_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)


class PipelineTrigger:
    """Trigger functions with prerequisite validation.

    Each trigger validates that prerequisites are met before invoking
    the corresponding pipeline action. This prevents:
    - Exporting when no databases exist
    - Training when NPZ has too few samples
    - Evaluating when model file doesn't exist
    - Promoting when model doesn't meet thresholds
    """

    def __init__(self, config: TriggerConfig | None = None):
        self.config = config or TriggerConfig()
        self._iteration_counter: dict[str, int] = {}

    def _get_iteration(self, config_key: str) -> int:
        """Get next iteration number for a config."""
        if config_key not in self._iteration_counter:
            self._iteration_counter[config_key] = 0
        self._iteration_counter[config_key] += 1
        return self._iteration_counter[config_key]

    # =========================================================================
    # Prerequisite Checks
    # =========================================================================

    async def check_databases_exist(
        self,
        board_type: str,
        num_players: int,
    ) -> PrerequisiteResult:
        """Check if game databases exist for the config."""
        try:
            from app.utils.game_discovery import GameDiscovery

            discovery = GameDiscovery()
            databases = discovery.find_databases_for_config(board_type, num_players)

            if not databases:
                return PrerequisiteResult(
                    passed=False,
                    message=f"No game databases found for {board_type}_{num_players}p",
                    details={"databases_found": 0},
                )

            total_games = sum(db.game_count for db in databases)

            return PrerequisiteResult(
                passed=True,
                message=f"Found {len(databases)} database(s) with {total_games:,} games",
                details={
                    "databases_found": len(databases),
                    "total_games": total_games,
                    "database_paths": [str(db.path) for db in databases],
                },
            )
        except Exception as e:
            return PrerequisiteResult(
                passed=False,
                message=f"Error checking databases: {e}",
                details={"error": str(e)},
            )

    async def check_npz_exists(
        self,
        board_type: str,
        num_players: int,
    ) -> PrerequisiteResult:
        """Check if NPZ training file exists with sufficient samples."""
        try:
            import numpy as np

            training_dir = self.config.ai_service_root / "data" / "training"
            config_key = f"{board_type}_{num_players}p"

            # Look for NPZ files matching this config
            npz_patterns = [
                f"{config_key}_v*.npz",
                f"{config_key}_iter*.npz",
                f"{config_key}.npz",
                f"canonical_{config_key}.npz",
            ]

            best_npz: Path | None = None
            best_samples = 0

            for pattern in npz_patterns:
                for npz_path in training_dir.glob(pattern):
                    if npz_path.stat().st_size < self.config.min_npz_size_bytes:
                        continue

                    try:
                        with np.load(npz_path) as data:
                            samples = len(data.get("features", data.get("states", [])))
                            if samples > best_samples:
                                best_samples = samples
                                best_npz = npz_path
                    except Exception:
                        continue

            if best_npz is None:
                return PrerequisiteResult(
                    passed=False,
                    message=f"No valid NPZ file found for {config_key}",
                    details={"searched_dir": str(training_dir)},
                )

            if best_samples < self.config.min_samples_for_training:
                return PrerequisiteResult(
                    passed=False,
                    message=f"NPZ has {best_samples:,} samples, need {self.config.min_samples_for_training:,}",
                    details={
                        "npz_path": str(best_npz),
                        "samples": best_samples,
                        "min_required": self.config.min_samples_for_training,
                    },
                )

            return PrerequisiteResult(
                passed=True,
                message=f"Found {best_npz.name} with {best_samples:,} samples",
                details={
                    "npz_path": str(best_npz),
                    "samples": best_samples,
                },
            )
        except Exception as e:
            return PrerequisiteResult(
                passed=False,
                message=f"Error checking NPZ: {e}",
                details={"error": str(e)},
            )

    async def check_model_exists(
        self,
        board_type: str,
        num_players: int,
        model_path: str | None = None,
    ) -> PrerequisiteResult:
        """Check if model file exists."""
        try:
            config_key = f"{board_type}_{num_players}p"

            if model_path:
                path = Path(model_path)
            else:
                # Default model location
                models_dir = self.config.ai_service_root / "models" / config_key
                path = models_dir / "latest.pth"

            if not path.exists():
                return PrerequisiteResult(
                    passed=False,
                    message=f"Model not found: {path}",
                    details={"model_path": str(path)},
                )

            size_mb = path.stat().st_size / (1024 * 1024)

            return PrerequisiteResult(
                passed=True,
                message=f"Found model: {path.name} ({size_mb:.1f}MB)",
                details={
                    "model_path": str(path),
                    "size_mb": size_mb,
                },
            )
        except Exception as e:
            return PrerequisiteResult(
                passed=False,
                message=f"Error checking model: {e}",
                details={"error": str(e)},
            )

    async def check_no_training_running(
        self,
        board_type: str,
        num_players: int,
    ) -> PrerequisiteResult:
        """Check that no training is currently running for this config."""
        try:
            import subprocess

            config_key = f"{board_type}_{num_players}p"

            # Check for running training processes
            result = await asyncio.to_thread(
                subprocess.run,
                ["pgrep", "-f", f"train.*{board_type}.*{num_players}"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                return PrerequisiteResult(
                    passed=False,
                    message=f"Training already running for {config_key} (PIDs: {', '.join(pids)})",
                    details={"running_pids": pids},
                )

            return PrerequisiteResult(
                passed=True,
                message="No training currently running",
                details={},
            )
        except Exception as e:
            # If we can't check, assume it's safe to proceed
            return PrerequisiteResult(
                passed=True,
                message=f"Could not check for running training: {e}",
                details={"warning": str(e)},
            )

    async def check_evaluation_passed(
        self,
        win_rate_vs_random: float,
        win_rate_vs_heuristic: float,
    ) -> PrerequisiteResult:
        """Check if evaluation results meet promotion thresholds."""
        issues = []

        if win_rate_vs_random < self.config.min_win_rate_vs_random:
            issues.append(
                f"Win rate vs random: {win_rate_vs_random:.1%} < "
                f"{self.config.min_win_rate_vs_random:.1%} required"
            )

        if win_rate_vs_heuristic < self.config.min_win_rate_vs_heuristic:
            issues.append(
                f"Win rate vs heuristic: {win_rate_vs_heuristic:.1%} < "
                f"{self.config.min_win_rate_vs_heuristic:.1%} required"
            )

        if issues:
            return PrerequisiteResult(
                passed=False,
                message="Model does not meet promotion thresholds",
                details={
                    "issues": issues,
                    "win_rate_vs_random": win_rate_vs_random,
                    "win_rate_vs_heuristic": win_rate_vs_heuristic,
                },
            )

        return PrerequisiteResult(
            passed=True,
            message="Model meets all promotion thresholds",
            details={
                "win_rate_vs_random": win_rate_vs_random,
                "win_rate_vs_heuristic": win_rate_vs_heuristic,
            },
        )

    # =========================================================================
    # Trigger Functions with Validation
    # =========================================================================

    async def trigger_sync_after_selfplay(
        self,
        board_type: str,
        num_players: int,
        hosts: list[str] | None = None,
        skip_validation: bool = False,
    ) -> StageCompletionResult:
        """Trigger data sync after selfplay completes.

        Prerequisites:
        - Game databases exist (at least on remote hosts)

        Args:
            board_type: Board type (e.g., "hex8", "square8")
            num_players: Number of players
            hosts: Specific hosts to sync from
            skip_validation: Skip prerequisite checks

        Returns:
            StageCompletionResult
        """
        config_key = f"{board_type}_{num_players}p"
        iteration = self._get_iteration(config_key)

        logger.info(f"[PipelineTrigger] Triggering sync for {config_key} (iteration {iteration})")

        # For sync, we don't have strong prerequisites since data may be on remote hosts
        # Just log and proceed

        return await trigger_data_sync(
            board_type=board_type,
            num_players=num_players,
            iteration=iteration,
            hosts=hosts,
        )

    async def trigger_export_after_sync(
        self,
        board_type: str,
        num_players: int,
        skip_validation: bool = False,
    ) -> StageCompletionResult:
        """Trigger NPZ export after sync completes.

        Prerequisites:
        - Game databases exist locally

        Args:
            board_type: Board type
            num_players: Number of players
            skip_validation: Skip prerequisite checks

        Returns:
            StageCompletionResult
        """
        config_key = f"{board_type}_{num_players}p"
        iteration = self._get_iteration(config_key)

        logger.info(f"[PipelineTrigger] Triggering export for {config_key} (iteration {iteration})")

        # Check prerequisites
        if not skip_validation:
            db_check = await self.check_databases_exist(board_type, num_players)
            if not db_check:
                logger.warning(f"[PipelineTrigger] Export prerequisite failed: {db_check.message}")
                return StageCompletionResult(
                    success=False,
                    stage="npz_export",
                    iteration=iteration,
                    duration_seconds=0,
                    error=f"Prerequisite failed: {db_check.message}",
                    metadata={"prerequisite_check": db_check.details},
                )
            logger.info(f"[PipelineTrigger] Prerequisite passed: {db_check.message}")

        return await trigger_npz_export(
            board_type=board_type,
            num_players=num_players,
            iteration=iteration,
        )

    async def trigger_training_after_export(
        self,
        board_type: str,
        num_players: int,
        data_path: str | None = None,
        epochs: int = 30,
        skip_validation: bool = False,
    ) -> StageCompletionResult:
        """Trigger training after NPZ export completes.

        Prerequisites:
        - NPZ file exists with sufficient samples
        - No training already running for this config

        Args:
            board_type: Board type
            num_players: Number of players
            data_path: Path to NPZ file (auto-detected if None)
            epochs: Training epochs
            skip_validation: Skip prerequisite checks

        Returns:
            StageCompletionResult
        """
        config_key = f"{board_type}_{num_players}p"
        iteration = self._get_iteration(config_key)

        logger.info(f"[PipelineTrigger] Triggering training for {config_key} (iteration {iteration})")

        # Check prerequisites
        if not skip_validation:
            # Check NPZ exists
            npz_check = await self.check_npz_exists(board_type, num_players)
            if not npz_check:
                logger.warning(f"[PipelineTrigger] Training prerequisite failed: {npz_check.message}")
                return StageCompletionResult(
                    success=False,
                    stage="training",
                    iteration=iteration,
                    duration_seconds=0,
                    error=f"Prerequisite failed: {npz_check.message}",
                    metadata={"prerequisite_check": npz_check.details},
                )
            logger.info(f"[PipelineTrigger] NPZ check passed: {npz_check.message}")

            # Use discovered data path if not provided
            if data_path is None:
                data_path = npz_check.details.get("npz_path")

            # Check no training running
            running_check = await self.check_no_training_running(board_type, num_players)
            if not running_check:
                logger.warning(f"[PipelineTrigger] Training blocked: {running_check.message}")
                return StageCompletionResult(
                    success=False,
                    stage="training",
                    iteration=iteration,
                    duration_seconds=0,
                    error=f"Prerequisite failed: {running_check.message}",
                    metadata={"prerequisite_check": running_check.details},
                )
            logger.info(f"[PipelineTrigger] No conflicting training: {running_check.message}")

        return await trigger_training(
            board_type=board_type,
            num_players=num_players,
            iteration=iteration,
            data_path=data_path,
            epochs=epochs,
        )

    async def trigger_evaluation_after_training(
        self,
        board_type: str,
        num_players: int,
        model_path: str | None = None,
        games_per_opponent: int = 50,
        skip_validation: bool = False,
    ) -> StageCompletionResult:
        """Trigger evaluation after training completes.

        Prerequisites:
        - Model file exists

        Args:
            board_type: Board type
            num_players: Number of players
            model_path: Path to model (auto-detected if None)
            games_per_opponent: Games per evaluation opponent
            skip_validation: Skip prerequisite checks

        Returns:
            StageCompletionResult
        """
        config_key = f"{board_type}_{num_players}p"
        iteration = self._get_iteration(config_key)

        logger.info(f"[PipelineTrigger] Triggering evaluation for {config_key} (iteration {iteration})")

        # Check prerequisites
        if not skip_validation:
            model_check = await self.check_model_exists(board_type, num_players, model_path)
            if not model_check:
                logger.warning(f"[PipelineTrigger] Evaluation prerequisite failed: {model_check.message}")
                return StageCompletionResult(
                    success=False,
                    stage="evaluation",
                    iteration=iteration,
                    duration_seconds=0,
                    error=f"Prerequisite failed: {model_check.message}",
                    metadata={"prerequisite_check": model_check.details},
                )
            logger.info(f"[PipelineTrigger] Model check passed: {model_check.message}")

            # Use discovered model path if not provided
            if model_path is None:
                model_path = model_check.details.get("model_path")

        return await trigger_evaluation(
            board_type=board_type,
            num_players=num_players,
            iteration=iteration,
            model_path=model_path,
            games_per_opponent=games_per_opponent,
        )

    async def trigger_promotion_after_evaluation(
        self,
        board_type: str,
        num_players: int,
        model_path: str,
        win_rate_vs_random: float,
        win_rate_vs_heuristic: float,
        skip_validation: bool = False,
    ) -> StageCompletionResult:
        """Trigger promotion after evaluation completes.

        Prerequisites:
        - Model meets win rate thresholds

        Args:
            board_type: Board type
            num_players: Number of players
            model_path: Path to model to promote
            win_rate_vs_random: Win rate against random baseline
            win_rate_vs_heuristic: Win rate against heuristic baseline
            skip_validation: Skip prerequisite checks

        Returns:
            StageCompletionResult
        """
        config_key = f"{board_type}_{num_players}p"
        iteration = self._get_iteration(config_key)

        logger.info(f"[PipelineTrigger] Triggering promotion for {config_key} (iteration {iteration})")

        # Check prerequisites
        if not skip_validation:
            eval_check = await self.check_evaluation_passed(
                win_rate_vs_random=win_rate_vs_random,
                win_rate_vs_heuristic=win_rate_vs_heuristic,
            )
            if not eval_check:
                logger.warning(f"[PipelineTrigger] Promotion prerequisite failed: {eval_check.message}")
                return StageCompletionResult(
                    success=False,
                    stage="promotion",
                    iteration=iteration,
                    duration_seconds=0,
                    error=f"Prerequisite failed: {eval_check.message}",
                    metadata={"prerequisite_check": eval_check.details},
                )
            logger.info(f"[PipelineTrigger] Evaluation check passed: {eval_check.message}")

        return await trigger_promotion(
            board_type=board_type,
            num_players=num_players,
            iteration=iteration,
            model_path=model_path,
        )


# =============================================================================
# Singleton and convenience functions
# =============================================================================

_pipeline_trigger: PipelineTrigger | None = None


def get_pipeline_trigger() -> PipelineTrigger:
    """Get the global PipelineTrigger singleton."""
    global _pipeline_trigger
    if _pipeline_trigger is None:
        _pipeline_trigger = PipelineTrigger()
    return _pipeline_trigger


async def validate_pipeline_prerequisites(
    board_type: str,
    num_players: int,
    stage: str,
) -> PrerequisiteResult:
    """Validate prerequisites for a specific pipeline stage.

    Args:
        board_type: Board type
        num_players: Number of players
        stage: Stage to validate ("sync", "export", "training", "evaluation")

    Returns:
        PrerequisiteResult
    """
    trigger = get_pipeline_trigger()

    if stage == "sync":
        # Sync has no strong prerequisites
        return PrerequisiteResult(passed=True, message="Sync has no prerequisites")
    elif stage == "export":
        return await trigger.check_databases_exist(board_type, num_players)
    elif stage == "training":
        npz_check = await trigger.check_npz_exists(board_type, num_players)
        if not npz_check:
            return npz_check
        return await trigger.check_no_training_running(board_type, num_players)
    elif stage == "evaluation":
        return await trigger.check_model_exists(board_type, num_players)
    else:
        return PrerequisiteResult(passed=False, message=f"Unknown stage: {stage}")


__all__ = [
    "PipelineTrigger",
    "PrerequisiteResult",
    "TriggerConfig",
    "get_pipeline_trigger",
    "validate_pipeline_prerequisites",
]
