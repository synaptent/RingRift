"""Pipeline stage mixin - stage callback handlers for DataPipelineOrchestrator.

December 2025: Extracted from data_pipeline_orchestrator.py as part of mixin-based refactoring.
December 2025: Updated to inherit from PipelineMixinBase for common patterns.

This mixin provides stage callback handlers (different from data event handlers):
- _on_selfplay_complete: Handle selfplay completion
- _on_sync_complete: Handle data sync completion
- _on_npz_export_complete: Handle NPZ export completion
- _on_training_started: Handle training start
- _on_training_complete: Handle training completion
- _on_training_failed: Handle training failure
- _on_evaluation_complete: Handle evaluation completion
- _on_promotion_complete: Handle promotion completion
- _on_iteration_complete: Handle iteration completion
- _extract_stage_result: Extract StageCompletionResult from RouterEvent (overrides base)

Inherits from PipelineMixinBase which provides:
- DataPipelineOrchestratorProtocol (documents expected interface)
- Common utility methods (_get_config_key, _log_stage_event, etc.)
- Circuit breaker helpers
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from app.coordination.pipeline_mixin_base import PipelineMixinBase

if TYPE_CHECKING:
    from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

logger = logging.getLogger(__name__)


class PipelineStageMixin(PipelineMixinBase):
    """Mixin providing stage callback handlers for DataPipelineOrchestrator.

    This mixin handles StageEvent callbacks that drive pipeline stage transitions.
    Stage handlers update pipeline state, trigger downstream stages, and maintain
    iteration records.

    Inherits from PipelineMixinBase for common utilities.
    """

    # Additional type hints specific to this mixin
    if TYPE_CHECKING:
        _completed_iterations: list
        _total_games: int
        _total_models: int
        _total_promotions: int
        quality_gate_enabled: bool
        _last_quality_score: float
        max_history: int

    # =========================================================================
    # Stage Result Extraction (overrides base with more fields)
    # =========================================================================

    def _extract_stage_result(self, event_or_result) -> Any:
        """Extract StageCompletionResult from RouterEvent or return as-is.

        December 27, 2025: Fix for RouterEvent.iteration AttributeError.
        When subscribers are called via the unified router, they may receive
        either a StageCompletionResult directly or a RouterEvent wrapper.
        This helper extracts the underlying result for consistent handling.
        """
        # If it's a RouterEvent, extract the stage_result
        if hasattr(event_or_result, "stage_result") and event_or_result.stage_result is not None:
            return event_or_result.stage_result

        # If it's a RouterEvent without stage_result, try payload
        if hasattr(event_or_result, "payload") and hasattr(event_or_result, "event_type"):
            # It's a RouterEvent - create a minimal result-like object from payload
            payload = event_or_result.payload or {}

            # Create a SimpleNamespace to mimic StageCompletionResult
            return SimpleNamespace(
                iteration=payload.get("iteration", 0),
                success=payload.get("success", True),
                games_generated=payload.get("games_generated", payload.get("games_count", 0)),
                board_type=payload.get("board_type"),
                num_players=payload.get("num_players"),
                error=payload.get("error"),
                metadata=payload.get("metadata", {}),
                model_path=payload.get("model_path"),
                train_loss=payload.get("train_loss"),
                val_loss=payload.get("val_loss"),
                win_rate=payload.get("win_rate"),
                elo_delta=payload.get("elo_delta"),
                promoted=payload.get("promoted", False),
            )

        # It's already a StageCompletionResult or similar
        return event_or_result

    # =========================================================================
    # Selfplay Stage Handler
    # =========================================================================

    async def _on_selfplay_complete(self, result) -> None:
        """Handle selfplay completion."""
        from app.coordination.data_pipeline_orchestrator import PipelineStage

        # December 27, 2025: Handle both RouterEvent and StageCompletionResult
        result = self._extract_stage_result(result)

        iteration = getattr(result, "iteration", 0)
        self._ensure_iteration_record(iteration)

        # Track board configuration for downstream stages
        self._current_board_type = getattr(result, "board_type", None)
        self._current_num_players = getattr(result, "num_players", None)

        games_generated = getattr(result, "games_generated", 0)
        self._iteration_records[iteration].games_generated = games_generated
        self._total_games += games_generated

        if getattr(result, "success", True):
            self._transition_to(
                PipelineStage.DATA_SYNC,
                iteration,
                metadata={"games_generated": result.games_generated},
            )

            # Auto-trigger data sync if enabled (December 2025)
            if self.auto_trigger and self.auto_trigger_sync:
                await self._auto_trigger_sync(iteration)
        else:
            self._transition_to(
                PipelineStage.IDLE,
                iteration,
                success=False,
                metadata={"error": getattr(result, "error", "Unknown error")},
            )

    # =========================================================================
    # Sync Stage Handler
    # =========================================================================

    async def _on_sync_complete(self, result) -> None:
        """Handle data sync completion."""
        from app.coordination.data_pipeline_orchestrator import PipelineStage

        # December 27, 2025: Handle both RouterEvent and StageCompletionResult
        result = self._extract_stage_result(result)

        iteration = getattr(result, "iteration", 0)

        if getattr(result, "success", True):
            self._transition_to(
                PipelineStage.NPZ_EXPORT,
                iteration,
                metadata=getattr(result, "metadata", {}),
            )

            # Auto-trigger NPZ export if enabled (December 2025)
            if self.auto_trigger and self.auto_trigger_export:
                await self._auto_trigger_export(iteration)
        else:
            self._transition_to(
                PipelineStage.IDLE,
                iteration,
                success=False,
                metadata={"error": getattr(result, "error", "Unknown error")},
            )

    # =========================================================================
    # NPZ Export Stage Handler
    # =========================================================================

    async def _on_npz_export_complete(self, result) -> None:
        """Handle NPZ export completion."""
        from app.coordination.data_pipeline_orchestrator import PipelineStage

        # December 27, 2025: Handle both RouterEvent and StageCompletionResult
        result = self._extract_stage_result(result)

        iteration = getattr(result, "iteration", 0)

        if getattr(result, "success", True):
            # Quality gate check before training (December 2025 - Phase 14)
            metadata = getattr(result, "metadata", {}) or {}
            npz_path = getattr(result, "output_path", None) or metadata.get("output_path")
            if self.quality_gate_enabled and npz_path:
                quality_ok = await self._check_training_data_quality(npz_path, iteration)
                if not quality_ok:
                    logger.warning(
                        f"[DataPipelineOrchestrator] Quality gate blocked training for "
                        f"iteration {iteration} (quality={self._last_quality_score:.2f})"
                    )
                    await self._emit_training_blocked_by_quality(iteration, npz_path)
                    # Don't transition to training - stay at export complete
                    return

            self._transition_to(
                PipelineStage.TRAINING,
                iteration,
                metadata=metadata,
            )

            # Auto-trigger training if enabled (December 2025)
            if self.auto_trigger and self.auto_trigger_training:
                if npz_path:
                    await self._auto_trigger_training(iteration, npz_path)
        else:
            self._transition_to(
                PipelineStage.IDLE,
                iteration,
                success=False,
                metadata={"error": getattr(result, "error", "Unknown error")},
            )

    # =========================================================================
    # Training Stage Handlers
    # =========================================================================

    async def _on_training_started(self, result) -> None:
        """Handle training start."""
        from app.coordination.data_pipeline_orchestrator import PipelineStage

        # December 27, 2025: Handle both RouterEvent and StageCompletionResult
        result = self._extract_stage_result(result)

        iteration = getattr(result, "iteration", 0)
        self._ensure_iteration_record(iteration)
        self._stage_start_times[PipelineStage.TRAINING] = time.time()

    async def _on_training_complete(self, result) -> None:
        """Handle training completion."""
        from app.coordination.data_pipeline_orchestrator import PipelineStage

        # December 27, 2025: Handle both RouterEvent and StageCompletionResult
        result = self._extract_stage_result(result)

        iteration = getattr(result, "iteration", 0)
        record = self._ensure_iteration_record(iteration)

        record.model_id = getattr(result, "model_id", None)
        self._total_models += 1

        if getattr(result, "success", True):
            self._transition_to(
                PipelineStage.EVALUATION,
                iteration,
                metadata={
                    "model_id": getattr(result, "model_id", None),
                    "train_loss": getattr(result, "train_loss", None),
                    "val_loss": getattr(result, "val_loss", None),
                },
            )

            # Auto-trigger evaluation if enabled (December 2025)
            if self.auto_trigger and self.auto_trigger_evaluation:
                metadata = getattr(result, "metadata", {}) or {}
                model_path = getattr(result, "model_path", None) or metadata.get("model_path")
                if model_path:
                    await self._auto_trigger_evaluation(iteration, model_path)
        else:
            self._transition_to(
                PipelineStage.IDLE,
                iteration,
                success=False,
                metadata={"error": getattr(result, "error", "Unknown error")},
            )

    async def _on_training_failed(self, result) -> None:
        """Handle training failure."""
        from app.coordination.data_pipeline_orchestrator import PipelineStage

        # December 27, 2025: Handle both RouterEvent and StageCompletionResult
        result = self._extract_stage_result(result)

        iteration = getattr(result, "iteration", 0)
        record = self._ensure_iteration_record(iteration)
        record.error = getattr(result, "error", "Unknown error")

        self._transition_to(
            PipelineStage.IDLE,
            iteration,
            success=False,
            metadata={"error": getattr(result, "error", "Unknown error")},
        )

    # =========================================================================
    # Evaluation Stage Handler
    # =========================================================================

    async def _on_evaluation_complete(self, result) -> None:
        """Handle evaluation completion."""
        from app.coordination.data_pipeline_orchestrator import PipelineStage

        # December 27, 2025: Handle both RouterEvent and StageCompletionResult
        result = self._extract_stage_result(result)

        iteration = getattr(result, "iteration", 0)
        record = self._ensure_iteration_record(iteration)

        record.elo_delta = getattr(result, "elo_delta", 0.0) or 0.0

        if getattr(result, "success", True):
            self._transition_to(
                PipelineStage.PROMOTION,
                iteration,
                metadata={
                    "win_rate": getattr(result, "win_rate", None),
                    "elo_delta": getattr(result, "elo_delta", None),
                },
            )

            # Auto-trigger promotion if enabled (December 2025)
            if self.auto_trigger and self.auto_trigger_promotion:
                metadata = getattr(result, "metadata", {}) or {}
                model_path = getattr(result, "model_path", None) or metadata.get("model_path")
                gauntlet_results = metadata
                if model_path:
                    await self._auto_trigger_promotion(iteration, model_path, gauntlet_results)

            # December 27, 2025: Trigger model sync after successful evaluation
            # This ensures evaluated models are distributed to training nodes
            if self.auto_trigger and self.auto_trigger_sync:
                await self._trigger_model_sync_after_evaluation(result)
        else:
            self._transition_to(
                PipelineStage.IDLE,
                iteration,
                success=False,
                metadata={"error": getattr(result, "error", "Unknown error")},
            )

    # =========================================================================
    # Promotion Stage Handler
    # =========================================================================

    async def _on_promotion_complete(self, result) -> None:
        """Handle promotion completion."""
        from app.coordination.data_pipeline_orchestrator import PipelineStage

        # December 27, 2025: Handle both RouterEvent and StageCompletionResult
        result = self._extract_stage_result(result)

        iteration = getattr(result, "iteration", 0)
        record = self._ensure_iteration_record(iteration)

        promoted = getattr(result, "promoted", False)
        record.promoted = promoted
        if promoted:
            self._total_promotions += 1

        self._transition_to(
            PipelineStage.COMPLETE,
            iteration,
            metadata={
                "promoted": promoted,
                "reason": getattr(result, "promotion_reason", ""),
            },
        )

        # Feed back to curriculum (December 2025)
        await self._update_curriculum_on_promotion(result)

        # December 27, 2025: Trigger model sync after successful promotion
        # This ensures promoted models are distributed to all cluster nodes
        if promoted and self.auto_trigger and self.auto_trigger_sync:
            await self._trigger_model_sync_after_promotion(result)

    # =========================================================================
    # Iteration Completion Handler
    # =========================================================================

    async def _on_iteration_complete(self, result) -> None:
        """Handle iteration completion."""
        from app.coordination.data_pipeline_orchestrator import PipelineStage

        # December 27, 2025: Handle both RouterEvent and StageCompletionResult
        result = self._extract_stage_result(result)

        iteration = getattr(result, "iteration", 0)
        if iteration in self._iteration_records:
            record = self._iteration_records[iteration]
            record.end_time = time.time()
            record.success = getattr(result, "success", True)

            # Move to completed history
            self._completed_iterations.append(record)
            if len(self._completed_iterations) > self.max_history:
                self._completed_iterations = self._completed_iterations[
                    -self.max_history :
                ]

            del self._iteration_records[iteration]

        self._transition_to(PipelineStage.IDLE, iteration + 1)
