"""Pipeline event handler mixin - data event handlers for DataPipelineOrchestrator.

December 2025: Extracted from data_pipeline_orchestrator.py as part of mixin-based refactoring.
December 2025: Updated to inherit from PipelineMixinBase for common patterns.

This mixin provides all `_on_*` event handlers for DataEventType events:
- Core pipeline events: selfplay complete, sync, training, evaluation, promotion
- Data lifecycle events: orphan games, consolidation, repair, NPZ combination
- Quality and feedback events: quality score updates, curriculum changes
- Resource events: backpressure, resource constraints
- Infrastructure events: S3 backup, database creation, work queue

Inherits from PipelineMixinBase which provides:
- DataPipelineOrchestratorProtocol (documents expected interface)
- Common utility methods (_get_config_key, _log_stage_event, etc.)
- Stage result extraction (_extract_stage_result)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from app.coordination.event_handler_utils import extract_config_key
from app.coordination.event_utils import make_config_key, parse_config_key
from app.coordination.event_emission_helpers import safe_emit_event
from app.coordination.pipeline_mixin_base import PipelineMixinBase

if TYPE_CHECKING:
    from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

logger = logging.getLogger(__name__)


class PipelineEventHandlerMixin(PipelineMixinBase):
    """Mixin providing data event handlers for DataPipelineOrchestrator.

    This mixin handles DataEventType events that drive the training pipeline.
    Event handlers update pipeline state and trigger downstream actions.

    Inherits from PipelineMixinBase for common utilities.
    """

    # Additional type hints specific to this mixin
    if TYPE_CHECKING:
        _paused: bool
        _pause_reason: str | None
        _backpressure_active: bool
        _resource_constraints: dict
        _stage_metadata: dict
        _quality_distribution: dict
        _last_quality_update: float
        _cache_invalidation_count: int
        _pending_cache_refresh: bool
        _active_optimization: str | None
        _optimization_run_id: str | None
        _optimization_start_time: float

    # =========================================================================
    # Core Pipeline Data Event Handlers
    # =========================================================================

    async def _on_data_selfplay_complete(self, event: Any) -> None:
        """Handle SELFPLAY_COMPLETE data events as a fallback."""
        if not self._should_process_stage_data_event("SELFPLAY_COMPLETE"):
            return

        payload = getattr(event, "payload", {}) or {}
        config_key = extract_config_key(payload)
        games_generated = payload.get("games_played", payload.get("games_generated", 0))
        metadata = {"config_key": config_key, **payload}

        if not config_key or not games_generated:
            return

        board_type, num_players = self._get_board_config(metadata=metadata)
        iteration = self._next_data_event_iteration()

        result = SimpleNamespace(
            iteration=iteration,
            board_type=board_type,
            num_players=num_players,
            games_generated=games_generated,
            success=payload.get("success", True),
            error=payload.get("error"),
            metadata=metadata,
        )
        await self._on_selfplay_complete(result)

    async def _on_data_sync_completed(self, event: Any) -> None:
        """Handle DATA_SYNC_COMPLETED data events as a fallback."""
        if not self._should_process_stage_data_event("DATA_SYNC_COMPLETED"):
            return

        payload = getattr(event, "payload", {}) or {}
        config_key = extract_config_key(payload)
        games_synced = payload.get("games_synced", 0) or payload.get("files_synced", 0)
        metadata = {"config_key": config_key, **payload}
        iteration = self._current_iteration_for_data_event()

        result = SimpleNamespace(
            iteration=iteration,
            success=True,
            error=None,
            metadata=metadata,
        )
        if games_synced:
            metadata["games_synced"] = games_synced

        await self._on_sync_complete(result)

    async def _on_data_sync_failed(self, event: Any) -> None:
        """Handle DATA_SYNC_FAILED data events as a fallback."""
        if not self._should_process_stage_data_event("DATA_SYNC_FAILED"):
            return

        payload = getattr(event, "payload", {}) or {}
        metadata = {"config_key": extract_config_key(payload), **payload}
        iteration = self._current_iteration_for_data_event()

        result = SimpleNamespace(
            iteration=iteration,
            success=False,
            error=payload.get("error"),
            metadata=metadata,
        )
        await self._on_sync_complete(result)

    async def _on_data_training_completed(self, event: Any) -> None:
        """Handle TRAINING_COMPLETED data events as a fallback."""
        if not self._should_process_stage_data_event("TRAINING_COMPLETED"):
            return

        payload = getattr(event, "payload", {}) or {}
        config_key = extract_config_key(payload)
        metadata = {"config_key": config_key, **payload}
        iteration = self._current_iteration_for_data_event()

        model_path = payload.get("checkpoint_path") or payload.get("model_path")
        model_id = payload.get("model_id")
        if not model_id and model_path:
            model_id = Path(model_path).stem

        result = SimpleNamespace(
            iteration=iteration,
            board_type=payload.get("board_type"),
            num_players=payload.get("num_players"),
            model_id=model_id,
            model_path=model_path,
            train_loss=payload.get("final_train_loss") or payload.get("train_loss"),
            val_loss=payload.get("final_val_loss") or payload.get("best_val_loss") or payload.get("val_loss"),
            success=True,
            error=None,
            metadata=metadata,
        )
        await self._on_training_complete(result)

    async def _on_data_training_failed(self, event: Any) -> None:
        """Handle TRAINING_FAILED data events as a fallback."""
        if not self._should_process_stage_data_event("TRAINING_FAILED"):
            return

        payload = getattr(event, "payload", {}) or {}
        metadata = {"config_key": extract_config_key(payload), **payload}
        iteration = self._current_iteration_for_data_event()

        result = SimpleNamespace(
            iteration=iteration,
            success=False,
            error=payload.get("error"),
            metadata=metadata,
        )
        await self._on_training_failed(result)

    async def _on_data_evaluation_completed(self, event: Any) -> None:
        """Handle EVALUATION_COMPLETED data events as a fallback."""
        if not self._should_process_stage_data_event("EVALUATION_COMPLETED"):
            return

        payload = getattr(event, "payload", {}) or {}
        metadata = {"config_key": extract_config_key(payload), **payload}
        iteration = self._current_iteration_for_data_event()

        result = SimpleNamespace(
            iteration=iteration,
            win_rate=payload.get("win_rate", 0.0),
            elo_delta=payload.get("elo_delta", 0.0),
            model_path=payload.get("model_path") or payload.get("checkpoint_path"),
            success=True,
            error=None,
            metadata=metadata,
        )
        await self._on_evaluation_complete(result)

    async def _on_data_evaluation_failed(self, event: Any) -> None:
        """Handle EVALUATION_FAILED data events as a fallback."""
        if not self._should_process_stage_data_event("EVALUATION_FAILED"):
            return

        payload = getattr(event, "payload", {}) or {}
        metadata = {"config_key": extract_config_key(payload), **payload}
        iteration = self._current_iteration_for_data_event()

        result = SimpleNamespace(
            iteration=iteration,
            success=False,
            error=payload.get("error"),
            metadata=metadata,
        )
        await self._on_evaluation_complete(result)

    async def _on_data_model_promoted(self, event: Any) -> None:
        """Handle MODEL_PROMOTED data events as a fallback."""
        if not self._should_process_stage_data_event("MODEL_PROMOTED"):
            return

        payload = getattr(event, "payload", {}) or {}
        metadata = {"config_key": extract_config_key(payload), **payload}
        iteration = self._current_iteration_for_data_event()

        result = SimpleNamespace(
            iteration=iteration,
            promoted=payload.get("promoted", True),
            promotion_reason=payload.get("promotion_reason") or payload.get("reason"),
            board_type=payload.get("board_type"),
            num_players=payload.get("num_players"),
            metadata=metadata,
        )
        await self._on_promotion_complete(result)

    # =========================================================================
    # Orphan Games Event Handlers
    # =========================================================================

    async def _on_orphan_games_detected(self, event: Any) -> None:
        """Handle ORPHAN_GAMES_DETECTED events - trigger sync to recover orphan games.

        Orphan games are selfplay games stored on ephemeral nodes that haven't been
        synced to training nodes. This event is emitted by OrphanDetectionDaemon
        when it finds unsynced games. We trigger a priority sync to recover them.
        """
        payload = getattr(event, "payload", {}) or {}
        orphan_count = payload.get("orphan_count", 0)
        source_node = payload.get("source_node")
        config_key = extract_config_key(payload)

        if orphan_count == 0:
            return

        logger.info(
            f"[DataPipelineOrchestrator] Orphan games detected: "
            f"{orphan_count} games from {source_node} ({config_key})"
        )

        # Record as pending work that needs sync
        self._orphan_games_pending = getattr(self, "_orphan_games_pending", 0) + orphan_count

        # Trigger priority sync if auto-trigger enabled
        if self.auto_trigger and self.auto_trigger_sync:
            await self._trigger_orphan_recovery_sync(source_node, config_key, orphan_count)

    async def _on_orphan_games_registered(self, event: Any) -> None:
        """Handle ORPHAN_GAMES_REGISTERED events - update pipeline state.

        This event is emitted after orphan games are successfully synced and
        registered in the training data catalog. We can now proceed with export.
        """
        payload = getattr(event, "payload", {}) or {}
        registered_count = payload.get("registered_count", 0)
        config_key = extract_config_key(payload)
        board_type = payload.get("board_type")
        num_players = payload.get("num_players")

        if registered_count == 0:
            return

        logger.info(
            f"[DataPipelineOrchestrator] Orphan games registered: "
            f"{registered_count} games for {config_key}"
        )

        # Update pending count
        pending = getattr(self, "_orphan_games_pending", 0)
        self._orphan_games_pending = max(0, pending - registered_count)

        # Emit NEW_GAMES_AVAILABLE for downstream consumers (e.g., export triggers)
        safe_emit_event(
            "new_games_available",
            {
                "board_type": board_type or "unknown",
                "num_players": num_players or 2,
                "new_games": registered_count,
                "source": "orphan_recovery",
                "config_key": config_key,
            },
            context="DataPipeline",
        )

    # =========================================================================
    # Consolidation Event Handlers
    # =========================================================================

    async def _on_consolidation_started(self, event: Any) -> None:
        """Handle CONSOLIDATION_STARTED events - track consolidation progress.

        December 2025: Part of training pipeline fix. Tracks when scattered
        selfplay games are being merged into canonical databases.
        """
        payload = getattr(event, "payload", {}) or {}
        board_type = payload.get("board_type")
        num_players = payload.get("num_players")
        config_key = extract_config_key(payload) or make_config_key(board_type, num_players)

        logger.info(
            f"[DataPipelineOrchestrator] Consolidation started for {config_key}"
        )

        # Track consolidation in progress
        if not hasattr(self, "_consolidations_in_progress"):
            self._consolidations_in_progress: set[str] = set()
        self._consolidations_in_progress.add(config_key)

    async def _on_consolidation_complete(self, event: Any) -> None:
        """Handle CONSOLIDATION_COMPLETE events - trigger export after consolidation.

        December 2025: Part of training pipeline fix. After games are consolidated
        into canonical databases, we trigger NPZ export for training.
        """
        payload = getattr(event, "payload", {}) or {}
        board_type = payload.get("board_type")
        num_players = payload.get("num_players")
        config_key = extract_config_key(payload) or make_config_key(board_type, num_players)
        games_consolidated = payload.get("games_consolidated", 0)
        canonical_db = payload.get("canonical_db")

        logger.info(
            f"[DataPipelineOrchestrator] Consolidation complete for {config_key}: "
            f"{games_consolidated} games merged into {canonical_db}"
        )

        # Remove from in-progress set
        if hasattr(self, "_consolidations_in_progress"):
            self._consolidations_in_progress.discard(config_key)

        # Emit NEW_GAMES_AVAILABLE to trigger export pipeline
        if games_consolidated > 0:
            safe_emit_event(
                "new_games_available",
                {
                    "board_type": board_type,
                    "num_players": num_players,
                    "new_games": games_consolidated,
                    "source": "consolidation",
                    "config_key": config_key,
                    "canonical_db": canonical_db,
                },
                context="DataPipeline",
            )
            logger.info(
                f"[DataPipelineOrchestrator] Triggered export pipeline after consolidation"
            )

    # =========================================================================
    # NPZ Combination Handlers (December 28, 2025)
    # =========================================================================

    async def _on_npz_combination_complete(self, event: Any) -> None:
        """Handle NPZ_COMBINATION_COMPLETE - trigger training with combined data.

        December 2025: After NPZ files are combined with quality-weighted sampling,
        trigger training using the combined file instead of single NPZ files.
        """
        from app.coordination.data_pipeline_orchestrator import PipelineStage

        payload = getattr(event, "payload", {}) or {}
        config_key = extract_config_key(payload)
        output_path = payload.get("output_path", "")
        total_samples = payload.get("total_samples", 0)
        samples_by_source = payload.get("samples_by_source", {})

        logger.info(
            f"[DataPipelineOrchestrator] NPZ combination complete for {config_key}: "
            f"{total_samples} samples in {output_path}"
        )

        # Transition to NPZ_COMBINATION stage and mark complete
        iteration = self._current_iteration
        self._transition_to(PipelineStage.NPZ_COMBINATION, iteration, success=True)

        # Track combination stats
        if not hasattr(self, "_combination_stats"):
            self._combination_stats: dict[str, Any] = {}
        self._combination_stats[config_key] = {
            "output_path": output_path,
            "total_samples": total_samples,
            "samples_by_source": samples_by_source,
            "timestamp": time.time(),
        }

        # Trigger training with the combined NPZ file
        if output_path and total_samples > 0:
            safe_emit_event(
                "training_threshold_reached",
                {
                    "config_key": config_key,
                    "data_path": output_path,
                    "total_samples": total_samples,
                    "source": "npz_combination",
                    "combined": True,
                },
                context="DataPipeline",
            )
            logger.info(
                f"[DataPipelineOrchestrator] Triggered training for {config_key} "
                f"with combined data ({total_samples} samples)"
            )

    async def _on_npz_combination_failed(self, event: Any) -> None:
        """Handle NPZ_COMBINATION_FAILED - fall back to single file training.

        December 2025: If NPZ combination fails, we fall back to using the best
        single NPZ file (largest or freshest) for training instead.
        """
        from app.coordination.data_pipeline_orchestrator import PipelineStage

        payload = getattr(event, "payload", {}) or {}
        config_key = extract_config_key(payload)
        error = payload.get("error", "Unknown error")

        logger.warning(
            f"[DataPipelineOrchestrator] NPZ combination failed for {config_key}: {error}. "
            f"Falling back to single file training."
        )

        # Record circuit breaker failure
        if hasattr(self, "circuit_breaker"):
            self.circuit_breaker.record_failure("npz_combination", error)

        # Transition with failure but don't block pipeline
        iteration = self._current_iteration
        self._transition_to(
            PipelineStage.NPZ_COMBINATION,
            iteration,
            success=False,
            metadata={"error": error, "fallback": True},
        )

        # Emit training threshold with fallback indicator
        # Training trigger will pick best single file instead
        safe_emit_event(
            "training_threshold_reached",
            {
                "config_key": config_key,
                "source": "npz_combination_fallback",
                "combined": False,
                "combination_error": error,
            },
            context="DataPipeline",
        )
        logger.info(
            f"[DataPipelineOrchestrator] Triggered fallback training for {config_key}"
        )

    # =========================================================================
    # Repair Event Handlers (December 27, 2025)
    # =========================================================================

    async def _on_repair_completed(self, event: Any) -> None:
        """Handle REPAIR_COMPLETED - retrigger blocked pipeline stages.

        December 2025: After replication repair completes, retrigger any pipeline
        stages that were blocked waiting for data availability.
        """
        from app.coordination.data_pipeline_orchestrator import PipelineStage

        payload = getattr(event, "payload", {}) or {}
        repair_type = payload.get("repair_type", "unknown")
        files_repaired = payload.get("files_repaired", 0)
        source_node = payload.get("source_node")

        logger.info(
            f"[DataPipelineOrchestrator] Repair completed: {repair_type}, "
            f"{files_repaired} files from {source_node}"
        )

        # If we were blocked on data availability, retry the pending stage
        if self._current_stage == PipelineStage.SELFPLAY:
            # Repair may have fixed missing game data - check if we can proceed
            logger.info("[DataPipelineOrchestrator] Checking if we can proceed after repair")
            # Emit event to trigger sync check
            safe_emit_event(
                "sync_triggered",
                {"reason": "post_repair", "source": "data_pipeline_orchestrator"},
                context="DataPipeline",
            )

    async def _on_repair_failed(self, event: Any) -> None:
        """Handle REPAIR_FAILED - log and potentially escalate."""
        payload = getattr(event, "payload", {}) or {}
        repair_type = payload.get("repair_type", "unknown")
        error = payload.get("error", "unknown error")
        source_node = payload.get("source_node")

        logger.warning(
            f"[DataPipelineOrchestrator] Repair failed: {repair_type} "
            f"from {source_node}: {error}"
        )

        # Increment error count for circuit breaker consideration
        if hasattr(self, "_repair_failure_count"):
            self._repair_failure_count += 1
        else:
            self._repair_failure_count = 1

    async def _on_sync_checksum_failed(self, event: Any) -> None:
        """Handle SYNC_CHECKSUM_FAILED - trigger repair for corrupted data.

        December 2025: When sync checksum validation fails, queue the affected
        data for re-sync via the replication daemon.
        """
        payload = getattr(event, "payload", {}) or {}
        file_path = payload.get("file_path", "unknown")
        expected = payload.get("expected_checksum", "")[:12]
        actual = payload.get("actual_checksum", "")[:12]
        source_node = payload.get("source_node", "unknown")

        logger.warning(
            f"[DataPipelineOrchestrator] Sync checksum failed: {file_path} "
            f"from {source_node} (expected={expected}..., got={actual}...)"
        )

        # Try to trigger repair via unified replication daemon
        try:
            from app.coordination.unified_replication_daemon import (
                get_unified_replication_daemon,
            )

            daemon = get_unified_replication_daemon()
            # Jan 2026: is_running is a property, not a method
            if daemon and daemon.is_running:
                # Queue file for re-sync (extract game_id from path if possible)
                game_id = payload.get("game_id")
                if game_id:
                    await daemon.trigger_repair([game_id])
                    logger.info(
                        f"[DataPipelineOrchestrator] Queued game {game_id} for repair "
                        f"after checksum failure"
                    )
        except ImportError:
            logger.debug("[DataPipelineOrchestrator] Replication daemon not available")
        except Exception as e:
            logger.debug(f"[DataPipelineOrchestrator] Failed to trigger repair: {e}")

    # =========================================================================
    # Quality and Curriculum Event Handlers
    # =========================================================================

    async def _on_quality_score_updated(self, event: Any) -> None:
        """Handle QUALITY_SCORE_UPDATED - track per-game quality changes.

        December 2025: Aggregates quality scores for quality gating decisions.
        If quality drops significantly, may block training.
        """
        payload = getattr(event, "payload", {}) or {}
        game_id = payload.get("game_id")
        quality_score = payload.get("quality_score", 0.0)
        config_key = extract_config_key(payload)

        # Track quality scores
        if not hasattr(self, "_recent_quality_scores"):
            self._recent_quality_scores: list = []

        self._recent_quality_scores.append(quality_score)
        # Keep only last 100 scores
        if len(self._recent_quality_scores) > 100:
            self._recent_quality_scores = self._recent_quality_scores[-100:]

        # Check for quality degradation
        if len(self._recent_quality_scores) >= 10:
            avg_quality = sum(self._recent_quality_scores[-10:]) / 10
            if avg_quality < 0.3:
                logger.warning(
                    f"[DataPipelineOrchestrator] Low quality trend detected: "
                    f"avg={avg_quality:.2f} for {config_key}"
                )

    async def _on_curriculum_rebalanced(self, event: Any) -> None:
        """Handle CURRICULUM_REBALANCED - update pipeline priorities.

        December 2025: When curriculum weights change, adjust which configs
        get priority in the training pipeline.
        """
        payload = getattr(event, "payload", {}) or {}
        config_key = extract_config_key(payload)
        new_weights = payload.get("weights", {})

        logger.info(
            f"[DataPipelineOrchestrator] Curriculum rebalanced for {config_key}: "
            f"weights={new_weights}"
        )

        # Store curriculum weights for training priority decisions
        if not hasattr(self, "_curriculum_weights"):
            self._curriculum_weights: dict = {}
        self._curriculum_weights[config_key] = new_weights

    async def _on_curriculum_advanced(self, event: Any) -> None:
        """Handle CURRICULUM_ADVANCED - log curriculum tier progression.

        December 2025: When curriculum advances to a harder tier, log
        for tracking training progress.
        """
        payload = getattr(event, "payload", {}) or {}
        config_key = extract_config_key(payload)
        new_tier = payload.get("tier")
        old_tier = payload.get("old_tier")

        logger.info(
            f"[DataPipelineOrchestrator] Curriculum advanced for {config_key}: "
            f"{old_tier} -> {new_tier}"
        )

        # Track curriculum progression
        if not hasattr(self, "_curriculum_tiers"):
            self._curriculum_tiers: dict = {}
        self._curriculum_tiers[config_key] = new_tier

    # =========================================================================
    # S3 Backup Event Handler
    # =========================================================================

    async def _on_s3_backup_completed(self, event: Any) -> None:
        """Handle S3_BACKUP_COMPLETED - track S3 backup status for pipeline health.

        December 2025: Tracks S3 backup completions for monitoring and
        pipeline health reporting. Part of Phase 3 S3 infrastructure.
        """
        payload = getattr(event, "payload", {}) or {}
        files_count = payload.get("uploaded_count", 0)
        bucket = payload.get("bucket", "")
        duration = payload.get("duration_seconds", 0.0)
        promotions = payload.get("promotions", [])

        # Track S3 backup metrics
        if not hasattr(self, "_s3_backup_stats"):
            self._s3_backup_stats = {
                "backups_completed": 0,
                "files_backed_up": 0,
                "last_backup_time": 0.0,
            }

        self._s3_backup_stats["backups_completed"] += 1
        self._s3_backup_stats["files_backed_up"] += files_count
        self._s3_backup_stats["last_backup_time"] = time.time()

        logger.info(
            f"[DataPipelineOrchestrator] S3 backup completed: "
            f"{files_count} files to {bucket} in {duration:.1f}s "
            f"(promotions: {len(promotions)})"
        )

    # =========================================================================
    # Quality and Cache Event Handlers (December 2025)
    # =========================================================================

    async def _on_quality_distribution_changed(self, event) -> None:
        """Handle QUALITY_DISTRIBUTION_CHANGED - track quality changes."""
        payload = event.payload

        # Update quality distribution
        distribution = payload.get("distribution", {})
        if distribution:
            self._quality_distribution = distribution
            self._last_quality_update = time.time()

        high_quality = distribution.get("high", 0.0)
        low_quality = distribution.get("low", 0.0)

        logger.info(
            f"[DataPipelineOrchestrator] Quality distribution updated: "
            f"high={high_quality:.1%}, low={low_quality:.1%}"
        )

        # If quality distribution shifted significantly, may need to adjust training
        if low_quality > 0.5:
            logger.warning(
                "[DataPipelineOrchestrator] Low quality data >50% - "
                "consider curriculum adjustments"
            )

    async def _on_cache_invalidated(self, event) -> None:
        """Handle CACHE_INVALIDATED - track cache changes for data pipeline."""
        payload = event.payload

        invalidation_type = payload.get("invalidation_type", "")
        count = payload.get("count", 0)
        target_id = payload.get("target_id", "")

        self._cache_invalidation_count += count

        # Check if this affects NPZ data caches
        if invalidation_type == "model":
            # Model cache invalidation may require NPZ re-export
            self._pending_cache_refresh = True
            logger.info(
                f"[DataPipelineOrchestrator] Cache invalidated for model {target_id}: "
                f"{count} entries - NPZ data may need refresh"
            )
        else:
            logger.debug(
                f"[DataPipelineOrchestrator] Cache invalidated: "
                f"{invalidation_type}={target_id}, count={count}"
            )

    async def _on_optimization_triggered(self, event) -> None:
        """Handle CMAES_TRIGGERED or NAS_TRIGGERED - track optimization state."""
        payload = event.payload
        event_type = str(event.event_type.value).lower()

        # Determine optimization type from event
        if "cmaes" in event_type:
            opt_type = "cmaes"
        elif "nas" in event_type:
            opt_type = "nas"
        else:
            opt_type = "unknown"

        run_id = payload.get("run_id", "")
        reason = payload.get("reason", "")

        self._active_optimization = opt_type
        self._optimization_run_id = run_id
        self._optimization_start_time = time.time()

        logger.info(
            f"[DataPipelineOrchestrator] {opt_type.upper()} optimization triggered: "
            f"run_id={run_id}, reason={reason}"
        )

        # Note: Pipeline can continue but training may be coordinated differently
        # during optimization runs (e.g., different hyperparameters being tested)

    # =========================================================================
    # Resource Constraint Event Handlers (December 2025)
    # =========================================================================

    async def _on_resource_constraint_detected(self, event) -> None:
        """Handle RESOURCE_CONSTRAINT_DETECTED - pause pipeline on critical constraints."""
        from app.coordination.data_pipeline_orchestrator import PipelineStage

        payload = event.payload

        resource_type = payload.get("resource_type", "unknown")
        severity = payload.get("severity", "warning")
        current_value = payload.get("current_value", 0)
        threshold = payload.get("threshold", 0)
        node_id = payload.get("node_id", "")

        # Track the constraint
        self._resource_constraints[resource_type] = {
            "severity": severity,
            "current_value": current_value,
            "threshold": threshold,
            "node_id": node_id,
            "time": time.time(),
        }

        logger.warning(
            f"[DataPipelineOrchestrator] Resource constraint detected: "
            f"{resource_type}={current_value} (threshold={threshold}, severity={severity})"
        )

        # Pause on critical constraints during resource-intensive stages
        critical_stages = {PipelineStage.TRAINING, PipelineStage.NPZ_EXPORT}
        if severity == "critical" and self._current_stage in critical_stages:
            await self._pause_pipeline(
                reason=f"Critical {resource_type} constraint: {current_value}/{threshold}"
            )

    async def _on_backpressure_activated(self, event) -> None:
        """Handle BACKPRESSURE_ACTIVATED - pause pipeline under heavy load."""
        payload = event.payload

        source = payload.get("source", "unknown")
        level = payload.get("level", "unknown")

        self._backpressure_active = True

        logger.warning(
            f"[DataPipelineOrchestrator] Backpressure activated: source={source}, level={level}"
        )

        # Pause if backpressure is severe
        if level in ("high", "critical"):
            await self._pause_pipeline(reason=f"Backpressure from {source}: {level}")

    async def _on_backpressure_released(self, event) -> None:
        """Handle BACKPRESSURE_RELEASED - potentially resume pipeline."""
        payload = event.payload

        source = payload.get("source", "unknown")

        self._backpressure_active = False

        logger.info(f"[DataPipelineOrchestrator] Backpressure released: source={source}")

        # Auto-resume if paused due to backpressure and no other constraints
        if (self._paused and "Backpressure" in (self._pause_reason or "")
                and not self._has_critical_constraints()):
            await self._resume_pipeline()

    # =========================================================================
    # Game and Sync Feedback Event Handlers
    # =========================================================================

    async def _on_game_synced(self, event) -> None:
        """Handle GAME_SYNCED - games synced to training nodes.

        December 2025: Wire previously orphaned event. When games are synced
        to training nodes (via AutoSyncDaemon), this can trigger NPZ export
        if sufficient games have accumulated.

        Actions:
        - Track total games synced for metrics
        - Optionally trigger export if auto_trigger_export enabled and
          games synced exceed threshold
        """
        from app.coordination.data_pipeline_orchestrator import PipelineStage

        payload = event.payload if hasattr(event, 'payload') else event

        node_id = payload.get("node_id", "unknown")
        games_pushed = payload.get("games_pushed", 0)
        target_nodes = payload.get("target_nodes", [])
        is_ephemeral = payload.get("is_ephemeral", False)

        # Track cumulative games synced
        if not hasattr(self, "_games_synced_count"):
            self._games_synced_count = 0
        self._games_synced_count += games_pushed

        logger.debug(
            f"[DataPipelineOrchestrator] Games synced: {games_pushed} from {node_id} "
            f"to {len(target_nodes)} nodes (ephemeral={is_ephemeral})"
        )

        # If in DATA_SYNC stage and auto-trigger is enabled, consider triggering export
        if (self._current_stage == PipelineStage.DATA_SYNC
                and self.auto_trigger
                and self.auto_trigger_export):
            # Transition to NPZ_EXPORT stage
            self._transition_to(
                PipelineStage.NPZ_EXPORT,
                self._current_iteration,
                metadata={
                    "games_synced": games_pushed,
                    "source_node": node_id,
                    "target_nodes": target_nodes,
                },
            )
            await self._auto_trigger_export(self._current_iteration)

    async def _on_exploration_boost(self, event) -> None:
        """Handle EXPLORATION_BOOST - request to boost exploration temperature.

        December 2025: Wire previously orphaned event. Emitted when curriculum
        feedback detects that exploration diversity is low or training is
        plateauing on certain configurations.

        Actions:
        - Log the boost request for metrics
        - Forward to curriculum integration if available
        - Update exploration multiplier in pipeline metadata
        """
        payload = event.payload if hasattr(event, "payload") else event

        config_key = extract_config_key(payload, default="unknown")
        boost_factor = payload.get("boost_factor", 1.2)
        reason = payload.get("reason", "exploration_plateau")

        logger.info(
            f"[DataPipelineOrchestrator] Exploration boost requested: "
            f"config={config_key}, factor={boost_factor:.2f}, reason={reason}"
        )

        # Track exploration boost events for pipeline metrics
        if not hasattr(self, "_exploration_boost_count"):
            self._exploration_boost_count = 0
        self._exploration_boost_count += 1

        # Forward to curriculum integration if wired
        try:
            from app.training.curriculum_integration import (
                get_curriculum_integration,
            )
            curriculum = get_curriculum_integration()
            if curriculum and hasattr(curriculum, "apply_exploration_boost"):
                await curriculum.apply_exploration_boost(config_key, boost_factor)
                logger.debug(
                    f"[DataPipelineOrchestrator] Forwarded exploration boost to curriculum"
                )
        except ImportError:
            pass  # Curriculum integration not available
        except Exception as e:
            logger.warning(f"[DataPipelineOrchestrator] Failed to forward exploration boost: {e}")

    async def _on_sync_triggered(self, event) -> None:
        """Handle SYNC_TRIGGERED - data sync initiated due to staleness.

        December 2025: Wire previously orphaned event. Emitted when AutoSyncDaemon
        or SyncFacade triggers a sync due to stale data detection.

        Actions:
        - Log the sync trigger for metrics
        - Update pipeline stage if appropriate
        - Track sync frequency for feedback
        """
        from app.coordination.data_pipeline_orchestrator import PipelineStage

        payload = event.payload if hasattr(event, "payload") else event

        reason = payload.get("reason", "stale_data")
        config_key = extract_config_key(payload)
        data_age_hours = payload.get("data_age_hours", 0)
        source = payload.get("source", "unknown")

        logger.info(
            f"[DataPipelineOrchestrator] Sync triggered: reason={reason}, "
            f"config={config_key}, age={data_age_hours:.1f}h, source={source}"
        )

        # Track sync trigger frequency
        if not hasattr(self, "_sync_trigger_count"):
            self._sync_trigger_count = 0
        self._sync_trigger_count += 1

        # If we're idle and sync was triggered, transition to DATA_SYNC stage
        if self._current_stage == PipelineStage.IDLE and self.auto_trigger:
            self._transition_to(
                PipelineStage.DATA_SYNC,
                self._current_iteration,
                metadata={
                    "trigger_reason": reason,
                    "config_key": config_key,
                    "data_age_hours": data_age_hours,
                },
            )

    async def _on_data_stale(self, event) -> None:
        """Handle DATA_STALE - training data has become stale.

        December 2025: Wire this previously orphaned event. Emitted by
        TrainingFreshness or train_cli.py when data age exceeds threshold.

        Actions:
        - Log the stale data alert
        - Trigger priority sync via SyncFacade
        - Track stale data frequency for health monitoring
        """
        payload = event.payload if hasattr(event, "payload") else event

        config_key = extract_config_key(payload)
        data_age_hours = payload.get("data_age_hours", 0)
        max_age_hours = payload.get("max_age_hours", 1.0)
        source = payload.get("source", "unknown")

        logger.warning(
            f"[DataPipelineOrchestrator] DATA_STALE received: "
            f"config={config_key}, age={data_age_hours:.1f}h (max={max_age_hours:.1f}h), "
            f"source={source}"
        )

        # Track stale data frequency for health monitoring
        if not hasattr(self, "_stale_data_count"):
            self._stale_data_count = 0
        self._stale_data_count += 1

        # Trigger priority sync if we have SyncFacade available
        try:
            from app.coordination.sync_facade import get_sync_facade

            facade = get_sync_facade()
            if facade:
                from app.core.async_context import fire_and_forget

                async def trigger_sync():
                    await facade.trigger_priority_sync(
                        reason="stale_data",
                        config_key=config_key,
                        data_type="games",
                    )

                fire_and_forget(
                    trigger_sync(),
                    error_callback=lambda exc: logger.debug(
                        f"Priority sync trigger failed: {exc}"
                    ),
                )
                logger.info(
                    f"[DataPipelineOrchestrator] Triggered priority sync for {config_key}"
                )
        except ImportError:
            logger.debug("[DataPipelineOrchestrator] SyncFacade not available")

    # =========================================================================
    # Pipeline Status Event Handlers (Phase 11)
    # =========================================================================

    async def _on_new_games_available(self, event) -> None:
        """Handle NEW_GAMES_AVAILABLE - new game data ready for processing.

        December 2025 Phase 11: Wire NEW_GAMES_AVAILABLE to trigger NPZ export
        when new games are available. This closes the loop from selfplay -> export.

        Actions:
        - Track new game availability
        - Consider triggering NPZ export if threshold met
        - Update pipeline state for monitoring
        """
        payload = event.payload if hasattr(event, "payload") else event

        config_key = extract_config_key(payload)
        # Read with fallback chain for backward compatibility (canonical: new_games)
        game_count = payload.get(
            "new_games",
            payload.get("games_count", payload.get("count", payload.get("game_count", 0)))
        )
        source = payload.get("source", payload.get("host", "unknown"))

        logger.debug(
            f"[DataPipelineOrchestrator] NEW_GAMES_AVAILABLE: "
            f"config={config_key}, count={game_count}, source={source}"
        )

        # Track new games for this iteration
        if not hasattr(self, "_new_games_tracker"):
            self._new_games_tracker: dict[str, int] = {}
        if config_key:
            self._new_games_tracker[config_key] = (
                self._new_games_tracker.get(config_key, 0) + game_count
            )

        # Update stats
        if not hasattr(self, "_stats"):
            self._stats = {"total_games": 0}
        self._stats["total_games"] = self._stats.get("total_games", 0) + game_count

    async def _on_regression_detected(self, event) -> None:
        """Handle REGRESSION_DETECTED - model performance dropped.

        December 2025 Phase 11: Wire REGRESSION_DETECTED to pause training
        progression and track regression events for health monitoring.

        Actions:
        - Log the regression alert
        - Track regression events for health monitoring
        - Consider pausing training progression
        """
        payload = event.payload if hasattr(event, "payload") else event

        config_key = extract_config_key(payload)
        severity = payload.get("severity", "unknown")
        elo_change = payload.get("elo_change", 0)
        reason = payload.get("reason", "")

        logger.warning(
            f"[DataPipelineOrchestrator] REGRESSION_DETECTED: "
            f"config={config_key}, severity={severity}, elo_change={elo_change}, "
            f"reason={reason}"
        )

        # Track regression events for health monitoring
        if not hasattr(self, "_regression_count"):
            self._regression_count = 0
        self._regression_count += 1

        # Store last regression for diagnostics
        self._last_regression = {
            "config_key": config_key,
            "severity": severity,
            "elo_change": elo_change,
            "reason": reason,
            "timestamp": time.time(),
        }

        # If severity is severe/critical, consider pausing training
        if severity in ("severe", "critical"):
            logger.warning(
                f"[DataPipelineOrchestrator] Severe regression detected for {config_key}, "
                "training progression may be paused"
            )
            # Update stage metadata to reflect regression
            self._stage_metadata["regression_detected"] = True
            self._stage_metadata["regression_severity"] = severity

    async def _on_promotion_failed(self, event) -> None:
        """Handle PROMOTION_FAILED - model failed promotion criteria.

        December 2025 Phase 11: Wire PROMOTION_FAILED to track failed promotions
        and update pipeline state appropriately.

        Actions:
        - Log the promotion failure
        - Track failed promotions for monitoring
        - Update pipeline state (stay in EVALUATION or reset)
        """
        from app.coordination.data_pipeline_orchestrator import PipelineStage

        payload = event.payload if hasattr(event, "payload") else event

        config_key = extract_config_key(payload)
        reason = payload.get("reason", payload.get("error", "unknown"))
        model_path = payload.get("model_path", payload.get("model_id", ""))

        logger.warning(
            f"[DataPipelineOrchestrator] PROMOTION_FAILED: "
            f"config={config_key}, reason={reason}, model={model_path}"
        )

        # Track promotion failures for monitoring
        if not hasattr(self, "_promotion_failure_count"):
            self._promotion_failure_count = 0
        self._promotion_failure_count += 1

        # Store last failure for diagnostics
        self._last_promotion_failure = {
            "config_key": config_key,
            "reason": reason,
            "model_path": model_path,
            "timestamp": time.time(),
        }

        # If we're in promotion stage, transition back to evaluation
        if self._current_stage == PipelineStage.PROMOTION:
            self._transition_to(
                PipelineStage.EVALUATION,
                self._current_iteration,
                success=False,
                metadata={"promotion_failed": True, "reason": reason},
            )

    async def _on_promotion_candidate(self, event) -> None:
        """Handle PROMOTION_CANDIDATE - model ready for promotion evaluation.

        This handler was added December 2025 to wire the previously orphaned
        PROMOTION_CANDIDATE event. It's emitted by PromotionController when
        a model exceeds win rate thresholds in evaluation.

        Actions:
        - Logs the candidate for tracking
        - Updates pipeline state if in EVALUATION stage
        - Emits curriculum feedback event for training adjustments
        """
        from app.coordination.data_pipeline_orchestrator import PipelineStage

        payload = event.payload if hasattr(event, 'payload') else event

        model_id = payload.get("model_id", "unknown")
        board_type = payload.get("board_type", "unknown")
        num_players = payload.get("num_players", 2)
        win_rate = payload.get("win_rate_vs_heuristic", 0.0)

        logger.info(
            f"[DataPipelineOrchestrator] Promotion candidate: {model_id} "
            f"({board_type}_{num_players}p, {win_rate:.1%} vs heuristic)"
        )

        # Track candidates for this iteration
        if not hasattr(self, "_promotion_candidates"):
            self._promotion_candidates = []
        self._promotion_candidates.append({
            "model_id": model_id,
            "board_type": board_type,
            "num_players": num_players,
            "win_rate": win_rate,
            "timestamp": time.time(),
        })

        # If we're in evaluation stage, update transition tracking
        if self._current_stage == PipelineStage.EVALUATION:
            self._stage_metadata["candidates"] = len(self._promotion_candidates)

    async def _on_database_created(self, event) -> None:
        """Handle DATABASE_CREATED - new game database file created.

        This handler enables immediate registration and pipeline triggering
        when new databases are created, preventing orphaned databases.

        Added: December 2025 - Phase 4A.3
        """
        payload = event.payload if hasattr(event, 'payload') else event
        db_path = payload.get("db_path", "")
        board_type = payload.get("board_type", "")
        num_players = payload.get("num_players", 0)

        logger.info(
            f"[DataPipelineOrchestrator] New database created: {db_path} "
            f"({board_type}_{num_players}p)"
        )

        # Track for sync triggering if threshold is met
        if not hasattr(self, "_new_databases"):
            self._new_databases = []
        self._new_databases.append({
            "db_path": db_path,
            "board_type": board_type,
            "num_players": num_players,
            "timestamp": time.time(),
        })

    async def _on_training_threshold_reached(self, event) -> None:
        """Handle TRAINING_THRESHOLD_REACHED - enough games for training.

        Triggers NPZ export and training when game threshold is met.
        Added: December 2025
        """
        from app.coordination.data_pipeline_orchestrator import PipelineStage

        payload = event.payload if hasattr(event, 'payload') else event
        config = extract_config_key(payload)
        games = payload.get("games", 0)

        logger.info(
            f"[DataPipelineOrchestrator] Training threshold reached: "
            f"{config} ({games} games)"
        )

        # Auto-trigger export if enabled and we're in appropriate stage
        if self.auto_trigger and self.auto_trigger_export:
            if self._current_stage in [PipelineStage.IDLE, PipelineStage.DATA_SYNC]:
                parsed = parse_config_key(config)
                if parsed:
                    iteration = self._current_iteration + 1
                    await self._auto_trigger_export(iteration)
                else:
                    logger.warning(f"Failed to parse config {config}")

    async def _on_promotion_started(self, event) -> None:
        """Handle PROMOTION_STARTED - promotion process initiated.

        Tracks promotion attempts and updates pipeline stage.
        Added: December 2025
        """
        from app.coordination.data_pipeline_orchestrator import PipelineStage

        payload = event.payload if hasattr(event, 'payload') else event
        config = extract_config_key(payload)
        model_id = payload.get("model_id", "")

        logger.info(
            f"[DataPipelineOrchestrator] Promotion started: {model_id} ({config})"
        )

        # Transition to promotion stage if in evaluation
        if self._current_stage == PipelineStage.EVALUATION:
            iteration = self._current_iteration
            self._transition_to(
                PipelineStage.PROMOTION,
                iteration,
                metadata={"model_id": model_id, "config": config},
            )

    async def _on_work_queued(self, event) -> None:
        """Handle WORK_QUEUED - work added to distributed queue.

        Logs work queue activity for pipeline observability.
        Added: December 2025
        """
        payload = event.payload if hasattr(event, 'payload') else event
        work_type = payload.get("work_type", "unknown")
        config = extract_config_key(payload)

        logger.debug(
            f"[DataPipelineOrchestrator] Work queued: {work_type} ({config})"
        )

        # Track work queue depth for backpressure decisions
        if not hasattr(self, "_queued_work_count"):
            self._queued_work_count = 0
        self._queued_work_count += 1

    # =========================================================================
    # Task Lifecycle Event Handlers (December 2025)
    # =========================================================================

    async def _on_task_abandoned(self, event) -> None:
        """Handle TASK_ABANDONED - update pending counts and track abandonments.

        December 2025: Wire previously orphaned event. Emitted when jobs are
        intentionally cancelled (e.g., due to backpressure, resource constraints,
        or pipeline requirements). Different from TASK_FAILED which indicates errors.

        Actions:
        - Log the abandonment for metrics
        - Decrement pending work count
        - Track abandonment reason for debugging
        """
        payload = event.payload if hasattr(event, "payload") else event

        task_id = payload.get("task_id", "unknown")
        task_type = payload.get("task_type", "unknown")
        reason = payload.get("reason", "unknown")
        node_id = payload.get("node_id", "")
        config_key = extract_config_key(payload)

        logger.info(
            f"[DataPipelineOrchestrator] Task abandoned: {task_id} "
            f"(type={task_type}, node={node_id}, reason={reason})"
        )

        # Track abandonments for metrics
        if not hasattr(self, "_abandoned_task_count"):
            self._abandoned_task_count = 0
        self._abandoned_task_count += 1

        # Decrement pending work count if we were tracking it
        if hasattr(self, "_queued_work_count") and self._queued_work_count > 0:
            self._queued_work_count -= 1

        # Store last abandonment for debugging
        if not hasattr(self, "_last_task_abandonment"):
            self._last_task_abandonment = None
        self._last_task_abandonment = {
            "task_id": task_id,
            "task_type": task_type,
            "reason": reason,
            "node_id": node_id,
            "config_key": config_key,
            "timestamp": time.time(),
        }

        # Track by config for per-config analysis
        if config_key:
            if not hasattr(self, "_abandonments_by_config"):
                self._abandonments_by_config: dict[str, int] = {}
            self._abandonments_by_config[config_key] = (
                self._abandonments_by_config.get(config_key, 0) + 1
            )
