"""Pipeline trigger mixin - stage triggering methods for DataPipelineOrchestrator.

December 2025: Extracted from data_pipeline_orchestrator.py as part of mixin-based refactoring.

This mixin provides stage triggering methods:
- _auto_trigger_sync: Trigger data sync after selfplay
- _auto_trigger_export: Trigger NPZ export after sync
- _auto_trigger_training: Trigger training after export
- _auto_trigger_evaluation: Trigger evaluation after training
- _auto_trigger_promotion: Trigger promotion after evaluation
- _trigger_orphan_recovery_sync: Priority sync for orphan games
- _trigger_data_regeneration: Regenerate training data when quality is low
- _trigger_model_sync_after_evaluation: Sync models after evaluation
- _trigger_model_sync_after_promotion: Sync models after promotion
- _update_curriculum_on_promotion: Update curriculum weights after promotion

Expected attributes from main class:
- auto_trigger: bool
- auto_trigger_sync: bool
- auto_trigger_export: bool
- auto_trigger_training: bool
- auto_trigger_evaluation: bool
- auto_trigger_promotion: bool
- _current_board_type: str | None
- _current_num_players: int | None
- _iteration_records: dict
- _circuit_breaker: PipelineCircuitBreaker | None
- _last_quality_score: float
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

logger = logging.getLogger(__name__)


class PipelineTriggerMixin:
    """Mixin providing stage triggering methods for DataPipelineOrchestrator.

    This mixin handles automatic triggering of downstream pipeline stages
    after upstream stages complete successfully.
    """

    # Type hints for attributes expected from main class
    if TYPE_CHECKING:
        auto_trigger: bool
        auto_trigger_sync: bool
        auto_trigger_export: bool
        auto_trigger_training: bool
        auto_trigger_evaluation: bool
        auto_trigger_promotion: bool
        _current_board_type: str | None
        _current_num_players: int | None
        _iteration_records: dict
        _circuit_breaker: Any
        _last_quality_score: float

    # =========================================================================
    # Auto-Trigger Methods
    # =========================================================================

    async def _auto_trigger_sync(self, iteration: int) -> None:
        """Auto-trigger data synchronization with prerequisite validation."""
        if not self._can_auto_trigger():
            return

        board_type, num_players = self._get_board_config()
        if not board_type or not num_players:
            logger.warning("[DataPipelineOrchestrator] Cannot auto-trigger sync: missing board config")
            return

        try:
            # Use PipelineTrigger for prerequisite validation (December 2025)
            from app.coordination.pipeline_triggers import PipelineTrigger
            trigger = PipelineTrigger()
            result = await trigger.trigger_sync_after_selfplay(board_type, num_players)

            if result.success:
                self._record_circuit_success("data_sync")
                logger.info(f"[DataPipelineOrchestrator] Sync triggered successfully: {result.output_path or 'completed'}")
            else:
                self._record_circuit_failure("data_sync", result.error or "Prerequisite check failed")
                logger.warning(f"[DataPipelineOrchestrator] Sync trigger failed: {result.error or 'Unknown error'}")
        except Exception as e:
            logger.error(
                f"[DataPipelineOrchestrator] Auto-trigger sync failed for "
                f"{board_type}_{num_players}p: {e}",
                exc_info=True,
            )
            self._record_circuit_failure("data_sync", str(e))

    async def _auto_trigger_export(self, iteration: int) -> None:
        """Auto-trigger NPZ export with prerequisite validation."""
        if not self._can_auto_trigger():
            return

        board_type, num_players = self._get_board_config()
        if not board_type or not num_players:
            logger.warning("[DataPipelineOrchestrator] Cannot auto-trigger export: missing board config")
            return

        try:
            # Use PipelineTrigger for prerequisite validation (December 2025)
            from app.coordination.pipeline_triggers import PipelineTrigger
            trigger = PipelineTrigger()
            result = await trigger.trigger_export_after_sync(board_type, num_players)

            if result.success:
                self._record_circuit_success("npz_export")
                # Store output path for training stage
                if iteration in self._iteration_records:
                    self._iteration_records[iteration].metadata = {
                        "npz_path": result.output_path
                    }
                logger.info(f"[DataPipelineOrchestrator] Export triggered successfully: {result.output_path or 'completed'}")
            else:
                self._record_circuit_failure("npz_export", result.error or "Prerequisite check failed")
                logger.warning(f"[DataPipelineOrchestrator] Export trigger failed: {result.error or 'Unknown error'}")
        except Exception as e:
            logger.error(
                f"[DataPipelineOrchestrator] Auto-trigger export failed for "
                f"{board_type}_{num_players}p, iteration={iteration}: {e}",
                exc_info=True,
            )
            self._record_circuit_failure("npz_export", str(e))

    async def _auto_trigger_training(self, iteration: int, npz_path: str) -> None:
        """Auto-trigger neural network training with prerequisite validation."""
        if not self._can_auto_trigger():
            return

        board_type, num_players = self._get_board_config()
        if not board_type or not num_players:
            logger.warning("[DataPipelineOrchestrator] Cannot auto-trigger training: missing board config")
            return

        try:
            # Use PipelineTrigger for prerequisite validation (December 2025)
            from app.coordination.pipeline_triggers import PipelineTrigger
            trigger = PipelineTrigger()
            result = await trigger.trigger_training_after_export(board_type, num_players)

            if result.success:
                self._record_circuit_success("training")
                # Store model path for evaluation stage
                if iteration in self._iteration_records:
                    self._iteration_records[iteration].model_id = result.metadata.get("model_id")
                logger.info(f"[DataPipelineOrchestrator] Training triggered successfully: {result.output_path or 'completed'}")
            else:
                self._record_circuit_failure("training", result.error or "Prerequisite check failed")
                logger.warning(f"[DataPipelineOrchestrator] Training trigger failed: {result.error or 'Unknown error'}")
        except Exception as e:
            logger.error(
                f"[DataPipelineOrchestrator] Auto-trigger training failed for "
                f"{board_type}_{num_players}p, iteration={iteration}: {e}",
                exc_info=True,
            )
            self._record_circuit_failure("training", str(e))

    async def _auto_trigger_evaluation(self, iteration: int, model_path: str) -> None:
        """Auto-trigger gauntlet evaluation."""
        if not self._can_auto_trigger():
            return

        board_type, num_players = self._get_board_config()
        if not board_type or not num_players:
            logger.warning("[DataPipelineOrchestrator] Cannot auto-trigger evaluation: missing board config")
            return

        try:
            # Phase 7: Use PipelineTrigger for prerequisite validation
            from app.coordination.pipeline_triggers import get_pipeline_trigger

            trigger = get_pipeline_trigger()
            logger.info(f"[DataPipelineOrchestrator] Auto-triggering evaluation for {model_path}")
            result = await trigger.trigger_evaluation_after_training(
                board_type=board_type,
                num_players=num_players,
                model_path=model_path,
            )

            if result.success:
                self._record_circuit_success("evaluation")
                # Store evaluation results for promotion stage
                if iteration in self._iteration_records:
                    self._iteration_records[iteration].elo_delta = result.metadata.get("elo_delta", 0.0)
            else:
                self._record_circuit_failure("evaluation", result.error or "Unknown error")
        except Exception as e:
            logger.error(
                f"[DataPipelineOrchestrator] Auto-trigger evaluation failed for "
                f"iteration={iteration}, model={model_path}: {e}",
                exc_info=True,
            )
            self._record_circuit_failure("evaluation", str(e))

    async def _auto_trigger_promotion(
        self, iteration: int, model_path: str, gauntlet_results: dict
    ) -> None:
        """Auto-trigger model promotion."""
        if not self._can_auto_trigger():
            return

        board_type, num_players = self._get_board_config()
        if not board_type or not num_players:
            logger.warning("[DataPipelineOrchestrator] Cannot auto-trigger promotion: missing board config")
            return

        try:
            # Phase 7: Use PipelineTrigger for prerequisite validation
            from app.coordination.pipeline_triggers import get_pipeline_trigger

            # Extract win rates from gauntlet_results
            win_rates = gauntlet_results.get("win_rates", {})
            win_rate_vs_random = win_rates.get("random", gauntlet_results.get("win_rate_vs_random", 0.0))
            win_rate_vs_heuristic = win_rates.get("heuristic", gauntlet_results.get("win_rate_vs_heuristic", 0.0))

            trigger = get_pipeline_trigger()
            logger.info(f"[DataPipelineOrchestrator] Auto-triggering promotion for {model_path}")
            result = await trigger.trigger_promotion_after_evaluation(
                board_type=board_type,
                num_players=num_players,
                model_path=model_path,
                win_rate_vs_random=win_rate_vs_random,
                win_rate_vs_heuristic=win_rate_vs_heuristic,
            )

            if result.success:
                self._record_circuit_success("promotion")
            else:
                # Promotion failure is not a circuit-breaking event
                logger.info(f"[DataPipelineOrchestrator] Promotion skipped: {result.metadata.get('reason', 'Unknown')}")
        except Exception as e:
            logger.error(
                f"[DataPipelineOrchestrator] Auto-trigger promotion failed for "
                f"{board_type}_{num_players}p, model={model_path}: {e}",
                exc_info=True,
            )
            # Don't record as circuit failure - promotion is optional

    # =========================================================================
    # Priority Trigger Methods
    # =========================================================================

    async def _trigger_orphan_recovery_sync(
        self, source_node: str | None, config_key: str | None, orphan_count: int
    ) -> bool:
        """Trigger priority sync for orphan recovery with retry.

        Uses exponential backoff to retry failed sync operations.
        December 2025: Added retry to address orphan detection -> sync disconnect.

        Args:
            source_node: The node where orphans were detected
            config_key: The board config (e.g., "hex8_2p")
            orphan_count: Number of orphan games detected

        Returns:
            True if sync was successfully triggered, False otherwise
        """
        from app.utils.retry import RetryConfig

        # Retry config: 4 attempts, 2s base delay, 30s max (exponential backoff)
        retry_config = RetryConfig(
            max_attempts=4,
            base_delay=2.0,
            max_delay=30.0,
            exponential=True,
            jitter=0.2,
        )

        for attempt in retry_config.attempts():
            try:
                from app.coordination.sync_facade import get_sync_facade

                facade = get_sync_facade()
                if facade:
                    # Priority sync for orphan recovery
                    await facade.trigger_priority_sync(
                        reason="orphan_games_recovery",
                        source_node=source_node,
                        config_key=config_key,
                    )
                    logger.info(
                        f"[DataPipelineOrchestrator] Triggered priority sync for orphan recovery "
                        f"({orphan_count} games from {source_node})"
                    )
                    return True
                else:
                    logger.warning("[DataPipelineOrchestrator] SyncFacade not available")
                    return False

            except (ConnectionError, TimeoutError, OSError) as e:
                # Retryable network/connection errors
                if attempt.is_last:
                    logger.error(
                        f"[DataPipelineOrchestrator] Orphan sync failed after "
                        f"{attempt.number} attempts: {e}"
                    )
                    return False
                else:
                    logger.warning(
                        f"[DataPipelineOrchestrator] Orphan sync attempt {attempt.number} "
                        f"failed: {e}, retrying in {retry_config.get_delay(attempt.number):.1f}s"
                    )
                    await attempt.wait_async()

            except ImportError:
                # sync_facade not available - don't retry
                logger.debug("[DataPipelineOrchestrator] sync_facade not available")
                return False

            except Exception as e:
                # Other errors - log and don't retry
                logger.warning(f"[DataPipelineOrchestrator] Failed to trigger orphan sync: {e}")
                return False

        return False

    async def _trigger_data_regeneration(
        self, board_type: str, num_players: int, iteration: int
    ) -> None:
        """Trigger regeneration of training data when quality is low.

        Args:
            board_type: Board type
            num_players: Number of players
            iteration: Pipeline iteration
        """
        try:
            from app.coordination.event_router import DataEventType, get_router

            router = get_router()
            if router:
                await router.publish(
                    event_type=DataEventType.SELFPLAY_TARGET_UPDATED,
                    payload={
                        "config": f"{board_type}_{num_players}p",
                        "board_type": board_type,
                        "num_players": num_players,
                        "extra_games": 2000,  # Request more data
                        "reason": "quality_gate_failed",
                        "quality_score": self._last_quality_score,
                        "iteration": iteration,
                    },
                    source="DataPipelineOrchestrator",
                )
                logger.info(
                    f"[QualityGate] Triggered data regeneration for {board_type}_{num_players}p"
                )
        except Exception as e:
            logger.warning(f"[QualityGate] Failed to trigger data regeneration: {e}")

    # =========================================================================
    # Model Sync Triggers
    # =========================================================================

    async def _trigger_model_sync_after_evaluation(self, result) -> None:
        """Trigger model sync after evaluation completes.

        December 27, 2025: Ensures evaluated models are distributed to training
        nodes so they can be used for further selfplay or comparison.

        Args:
            result: Evaluation result with model_path and metadata
        """
        try:
            from app.coordination.sync_facade import get_sync_facade

            # December 27, 2025: Use getattr for safe attribute access
            metadata = getattr(result, "metadata", {}) or {}

            # Get config key for logging
            config_key = metadata.get("config_key")
            if not config_key and self._current_board_type and self._current_num_players:
                config_key = f"{self._current_board_type}_{self._current_num_players}p"

            facade = get_sync_facade()
            logger.info(
                f"[DataPipelineOrchestrator] Triggering model sync after evaluation "
                f"({config_key or 'unknown config'})"
            )
            await facade.trigger_priority_sync(
                reason="post_evaluation_sync",
                config_key=config_key,
                data_type="models",
            )
        except ImportError:
            logger.debug("[DataPipelineOrchestrator] sync_facade not available for eval sync")
        except Exception as e:
            logger.warning(f"[DataPipelineOrchestrator] Model sync after evaluation failed: {e}")

    async def _trigger_model_sync_after_promotion(self, result) -> None:
        """Trigger model sync after promotion completes.

        December 27, 2025: Ensures promoted models are distributed to all cluster
        nodes so they can use the new best model for selfplay.

        Args:
            result: Promotion result with model info and metadata
        """
        try:
            from app.coordination.sync_facade import get_sync_facade

            # December 27, 2025: Use getattr for safe attribute access
            metadata = getattr(result, "metadata", {}) or {}

            # Get config key
            config_key = None
            board_type = getattr(result, "board_type", None)
            num_players = getattr(result, "num_players", None)
            if board_type and num_players:
                config_key = f"{board_type}_{num_players}p"
            elif metadata:
                config_key = metadata.get("config_key")
            if not config_key and self._current_board_type and self._current_num_players:
                config_key = f"{self._current_board_type}_{self._current_num_players}p"

            facade = get_sync_facade()
            logger.info(
                f"[DataPipelineOrchestrator] Triggering model sync after promotion "
                f"({config_key or 'unknown config'})"
            )
            await facade.trigger_priority_sync(
                reason="post_promotion_sync",
                config_key=config_key,
                data_type="models",
            )
        except ImportError:
            logger.debug("[DataPipelineOrchestrator] sync_facade not available for promotion sync")
        except Exception as e:
            logger.warning(f"[DataPipelineOrchestrator] Model sync after promotion failed: {e}")

    # =========================================================================
    # Curriculum Update
    # =========================================================================

    async def _update_curriculum_on_promotion(self, result) -> None:
        """Update curriculum weights based on promotion result.

        This closes the feedback loop: promotion results affect future
        training resource allocation via curriculum weights.

        December 2025: Added to complete the self-improvement loop.
        """
        try:
            from app.training.curriculum_feedback import get_curriculum_feedback

            # December 27, 2025: Use getattr for safe attribute access
            metadata = getattr(result, "metadata", {}) or {}

            # Get config key from result or tracked state
            config_key = None
            board_type = getattr(result, "board_type", None)
            num_players = getattr(result, "num_players", None)
            if board_type and num_players:
                config_key = f"{board_type}_{num_players}p"
            elif self._current_board_type and self._current_num_players:
                config_key = f"{self._current_board_type}_{self._current_num_players}p"
            else:
                config_key = metadata.get("config_key")

            if not config_key:
                logger.debug("[DataPipelineOrchestrator] No config_key for curriculum update")
                return

            promoted = getattr(result, "promoted", False)
            feedback = get_curriculum_feedback()
            feedback.record_promotion(
                config_key=config_key,
                promoted=promoted,
                new_elo=getattr(result, "new_elo", None) or metadata.get("new_elo"),
                promotion_reason=getattr(result, "promotion_reason", "") or metadata.get("reason", ""),
            )

            logger.info(
                f"[DataPipelineOrchestrator] Curriculum updated for {config_key}: "
                f"promoted={promoted}"
            )

        except ImportError:
            logger.debug("[DataPipelineOrchestrator] curriculum_feedback not available")
        except Exception as e:
            logger.warning(f"[DataPipelineOrchestrator] Curriculum update failed: {e}")
