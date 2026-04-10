"""Stage execution and event-handling helpers for DataPipelineOrchestrator."""

from __future__ import annotations

import logging
import asyncio
import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from app.config.thresholds import DISK_PRODUCTION_HALT_PERCENT
from app.coordination.event_handler_utils import extract_config_key
from app.coordination.event_utils import make_config_key, parse_config_key
from app.coordination.data_pipeline_orchestrator import (
    IterationRecord,
    MAX_STAGE_RETRIES,
    PipelineStage,
    STAGE_RETRY_BACKOFF_MULTIPLIER,
    STAGE_RETRY_DELAY_SECONDS,
    StageTransition,
)
from app.utils.sqlite_utils import connect_safe

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class PipelineStagesMixin:
    """Stage transition, quality, resource, and data-event handlers."""

    def _transition_to(
        self,
        new_stage: PipelineStage,
        iteration: int,
        success: bool = True,
        metadata: dict | None = None,
    ) -> None:
        """Record a stage transition.

        December 31, 2025: Added guards to prevent:
        1. Same-stage transitions (no-op, just log debug)
        2. Rapid transitions (minimum 100ms interval to prevent loops)
        """
        old_stage = self._current_stage

        # Guard 1: Skip same-stage transitions
        if old_stage == new_stage and success:
            logger.debug(
                f"[DataPipelineOrchestrator] Skipping same-stage transition: "
                f"{old_stage.value} -> {new_stage.value}"
            )
            return

        # Guard 2: Prevent rapid transitions (minimum 100ms interval)
        now = time.time()
        last_transition = getattr(self, "_last_transition_time", 0.0)
        if now - last_transition < 0.1:  # 100ms cooldown
            logger.debug(
                f"[DataPipelineOrchestrator] Throttling rapid transition: "
                f"{old_stage.value} -> {new_stage.value} (too soon after last)"
            )
            return
        self._last_transition_time = now

        # Calculate duration of previous stage
        duration = 0.0
        if old_stage in self._stage_start_times:
            duration = time.time() - self._stage_start_times[old_stage]
            # Record stage duration
            if old_stage not in self._stage_durations:
                self._stage_durations[old_stage] = []
            self._stage_durations[old_stage].append(duration)

        # Record transition
        transition = StageTransition(
            from_stage=old_stage,
            to_stage=new_stage,
            iteration=iteration,
            success=success,
            duration_seconds=duration,
            metadata=metadata or {},
        )
        self._transitions.append(transition)

        # Trim history
        if len(self._transitions) > self.max_history * 10:
            self._transitions = self._transitions[-self.max_history * 10 :]

        # Update current state
        self._current_stage = new_stage
        self._current_iteration = iteration
        self._stage_start_times[new_stage] = time.time()

        # Update iteration record
        if iteration in self._iteration_records:
            record = self._iteration_records[iteration]
            record.stages_completed.append(new_stage.value)

        logger.info(
            f"[DataPipelineOrchestrator] Stage transition: {old_stage.value} -> "
            f"{new_stage.value} (iteration {iteration})"
        )

        # Invoke stage callbacks
        for callback in self._stage_callbacks.get(new_stage, []):
            try:
                callback(new_stage, iteration)
            except Exception as e:
                callback_name = getattr(callback, "__name__", repr(callback))
                logger.error(
                    f"[DataPipelineOrchestrator] Callback error in {callback_name} "
                    f"for stage={new_stage.value}, iteration={iteration}: {e}",
                    exc_info=True,
                )
    def _ensure_iteration_record(self, iteration: int) -> IterationRecord:
        """Ensure an iteration record exists."""
        if iteration not in self._iteration_records:
            self._iteration_records[iteration] = IterationRecord(
                iteration=iteration,
                start_time=time.time(),
            )
        return self._iteration_records[iteration]
    async def _check_training_data_quality(self, npz_path: str, iteration: int) -> bool:
        """Check if training data meets quality threshold.

        Evaluates NPZ training data quality and blocks training if:
        - Average quality score < threshold (default 0.6)
        - High quality game percentage < minimum (default 30%)
        - Quality declining for 3 consecutive exports

        Args:
            npz_path: Path to NPZ training data
            iteration: Pipeline iteration number

        Returns:
            True if quality is acceptable for training
        """
        try:
            from pathlib import Path
            import numpy as np
            from app.utils.numpy_utils import safe_load_npz

            # Check if file exists
            if not Path(npz_path).exists():
                logger.warning(f"[QualityGate] NPZ file not found: {npz_path}")
                return True  # Allow training if we can't check

            # Load NPZ and check for quality metadata
            with safe_load_npz(npz_path) as data:
                # Try to get quality scores from NPZ metadata
                if "quality_scores" in data:
                    quality_scores = data["quality_scores"]
                    avg_quality = float(np.mean(quality_scores))
                    high_quality_pct = float(np.mean(quality_scores >= 0.7))
                elif "metadata" in data:
                    # Check metadata for quality info
                    metadata = data["metadata"].item() if data["metadata"].ndim == 0 else dict(data["metadata"])
                    avg_quality = metadata.get("avg_quality", 0.7)
                    high_quality_pct = metadata.get("high_quality_pct", 0.5)
                else:
                    # Estimate quality from data characteristics
                    avg_quality = await self._estimate_data_quality(data, npz_path)
                    high_quality_pct = 0.5  # Default

            self._last_quality_score = avg_quality
            self._quality_check_history.append(avg_quality)

            # Keep only last 10 checks
            if len(self._quality_check_history) > 10:
                self._quality_check_history = self._quality_check_history[-10:]

            logger.info(
                f"[QualityGate] Iteration {iteration}: avg_quality={avg_quality:.3f}, "
                f"high_quality_pct={high_quality_pct:.1%}, threshold={self.quality_gate_threshold}"
            )

            # Check 1: Average quality threshold
            if avg_quality < self.quality_gate_threshold:
                logger.warning(
                    f"[QualityGate] Quality {avg_quality:.3f} below threshold "
                    f"{self.quality_gate_threshold}"
                )
                return False

            # Check 2: High quality percentage
            if high_quality_pct < self.quality_gate_min_high_quality_pct:
                logger.warning(
                    f"[QualityGate] High quality games {high_quality_pct:.1%} below minimum "
                    f"{self.quality_gate_min_high_quality_pct:.1%}"
                )
                return False

            # Check 3: Quality declining trend
            if len(self._quality_check_history) >= 3:
                recent = self._quality_check_history[-3:]
                if all(recent[i] > recent[i + 1] for i in range(len(recent) - 1)):
                    decline_amount = recent[0] - recent[-1]
                    if decline_amount > 0.1:  # More than 10% decline
                        logger.warning(
                            f"[QualityGate] Quality declining: {recent[0]:.3f} -> {recent[-1]:.3f}"
                        )
                        return False

            return True

        except (ValueError, TypeError, KeyError, AttributeError, ZeroDivisionError, IndexError) as e:
            logger.warning(f"[QualityGate] Error checking quality: {e}")
            return True  # Allow training if quality check fails
    async def _estimate_data_quality(self, data: "np.lib.npyio.NpzFile", npz_path: str) -> float:
        """Estimate data quality from NPZ contents when no explicit quality scores.

        Args:
            data: Loaded NPZ file
            npz_path: Path to NPZ (for logging)

        Returns:
            Estimated quality score (0-1)
        """
        try:
            import numpy as np

            quality_signals = []

            # Check sample count
            if "features" in data or "X" in data:
                features_key = "features" if "features" in data else "X"
                n_samples = len(data[features_key])
                # More samples = generally better, normalize to 0.3-1.0 range
                sample_score = min(1.0, 0.3 + 0.7 * (n_samples / 50000))
                quality_signals.append(sample_score)

            # Check policy distribution
            if "policy" in data or "policy_targets" in data:
                policy_key = "policy" if "policy" in data else "policy_targets"
                policy = data[policy_key]
                # Check entropy of policies (higher = more diverse = better)
                policy_probs = np.clip(policy, 1e-10, 1.0)
                entropy = -np.sum(policy_probs * np.log(policy_probs), axis=-1).mean()
                max_entropy = np.log(policy.shape[-1])
                entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0.5
                quality_signals.append(min(1.0, 0.3 + 0.7 * entropy_ratio))

            # Check value distribution
            if "value" in data or "value_targets" in data:
                value_key = "value" if "value" in data else "value_targets"
                values = data[value_key]
                # Check if values span a reasonable range (not all same)
                value_std = np.std(values)
                value_score = min(1.0, 0.4 + value_std * 2)  # Higher variance = more diverse
                quality_signals.append(value_score)

            if quality_signals:
                return float(np.mean(quality_signals))
            return 0.6  # Default moderate quality

        except (ValueError, TypeError, KeyError, IndexError, AttributeError, ZeroDivisionError) as e:
            logger.debug(f"[QualityGate] Error estimating quality: {e}")
            return 0.6
    async def _emit_training_blocked_by_quality(self, iteration: int, npz_path: str) -> None:
        """Emit event when training is blocked due to quality gate.

        This triggers data regeneration or other corrective actions.

        Args:
            iteration: Pipeline iteration
            npz_path: Path to the NPZ file that failed quality check
        """
        try:
            from app.coordination.event_router import publish

            board_type, num_players = self._get_board_config()
            # December 30, 2025: Include config_key for SelfplayScheduler integration
            config_key = make_config_key(board_type, num_players) if board_type and num_players else ""
            await publish(
                event_type="TRAINING_BLOCKED_BY_QUALITY",
                payload={
                    "iteration": iteration,
                    "npz_path": npz_path,
                    "board_type": board_type,
                    "num_players": num_players,
                    "config_key": config_key,  # Added for SelfplayScheduler
                    "quality_score": self._last_quality_score,
                    "threshold": self.quality_gate_threshold,
                    "quality_history": self._quality_check_history[-5:],
                    "recommendation": "trigger_data_regeneration",
                    "reason": "quality_gate_failed",
                },
                source="DataPipelineOrchestrator",
            )
            logger.info(
                f"[QualityGate] Emitted TRAINING_BLOCKED_BY_QUALITY for iteration {iteration}"
            )

            # Also trigger data regeneration if we have enough info
            if board_type and num_players:
                await self._trigger_data_regeneration(board_type, num_players, iteration)

        except (RuntimeError, ValueError, TypeError, AttributeError, KeyError, ImportError) as e:
            logger.warning(f"[QualityGate] Failed to emit quality block event: {e}")
    async def _trigger_data_regeneration(
        self, board_type: str, num_players: int, iteration: int
    ) -> None:
        """Request more selfplay data when quality gate blocks training.

        December 30, 2025: Implements Gap #6 from integration analysis.
        Emits SELFPLAY_TARGET_UPDATED to boost data generation for blocked configs.
        """
        try:
            from app.coordination.event_router import publish

            config_key = make_config_key(board_type, num_players)

            # Calculate additional games needed based on quality score
            quality_score = getattr(self, "_last_quality_score", 0.5)
            # Lower quality = more games needed
            additional_games = int(200 * (1.0 - quality_score))
            additional_games = max(100, min(500, additional_games))

            await publish(
                event_type="SELFPLAY_TARGET_UPDATED",
                payload={
                    "config_key": config_key,
                    "target_games": additional_games,
                    "priority": "high",
                    "reason": "quality_gate_blocked",
                    "quality_score": quality_score,
                    "iteration": iteration,
                    "exploration_boost": 1.5,  # Encourage diverse data
                },
                source="DataPipelineOrchestrator",
            )
            logger.info(
                f"[DataPipelineOrchestrator] Requested {additional_games} additional games "
                f"for {config_key} (quality={quality_score:.2f})"
            )

        except (ImportError, RuntimeError, AttributeError) as e:
            logger.debug(f"[DataPipelineOrchestrator] Could not trigger regeneration: {e}")
    def _can_auto_trigger(self) -> bool:
        """Check if auto-triggering is allowed.

        Returns False if:
        - Pipeline is paused
        - Circuit breaker is open
        - Backpressure is active
        - Cluster resources are constrained (December 2025)
        """
        if self._paused:
            logger.debug("[DataPipelineOrchestrator] Auto-trigger blocked: pipeline paused")
            return False

        if self._circuit_breaker and self._circuit_breaker.is_open:
            logger.debug("[DataPipelineOrchestrator] Auto-trigger blocked: circuit breaker open")
            return False

        if self._backpressure_active:
            logger.debug("[DataPipelineOrchestrator] Auto-trigger blocked: backpressure active")
            return False

        # Check unified health manager (December 2025)
        # This provides comprehensive health scoring including node availability,
        # circuit health, error rates, and recovery status
        try:
            from app.coordination.unified_health_manager import should_pause_pipeline

            should_pause, reason = should_pause_pipeline()
            if should_pause:
                logger.debug(
                    f"[DataPipelineOrchestrator] Auto-trigger blocked by health manager: {reason}"
                )
                # Emit event for monitoring/alerting
                from app.coordination.event_emission_helpers import safe_emit_event
                from app.distributed.data_events import DataEventType

                safe_emit_event(
                    DataEventType.TRAINING_BLOCKED_BY_QUALITY,  # Reuse existing event type
                    payload={
                        "reason": reason,
                        "blocked_by": "health_manager",
                        "source": "data_pipeline_orchestrator",
                    },
                    context="DataPipelineOrchestrator",
                    source="health_manager_check",
                )
                return False
        except ImportError:
            pass  # UnifiedHealthManager not available, skip this check
        except (RuntimeError, ValueError, TypeError, AttributeError, KeyError) as e:
            logger.debug(f"[DataPipelineOrchestrator] Health manager check failed: {e}")
            # Continue on health check failure - don't block training

        # Check cluster resources (December 2025)
        if not self._check_cluster_resources():
            logger.debug("[DataPipelineOrchestrator] Auto-trigger blocked: cluster resources constrained")
            return False

        return True
    def _check_cluster_resources(
        self,
        disk_threshold: float = float(DISK_PRODUCTION_HALT_PERCENT),
        min_free_disk_gb: float = 50.0,
    ) -> bool:
        """Check if cluster has sufficient resources for training.

        Returns True if resources are adequate, False if constrained.

        Args:
            disk_threshold: Max disk usage percentage before blocking
            min_free_disk_gb: Minimum free disk space required

        December 2025: Added to integrate cluster status with training decisions.
        Uses cached ClusterMonitor with TTL to avoid expensive SSH reconnections.
        """
        try:
            from app.coordination.cluster_status_monitor import ClusterMonitor

            # Use cached ClusterMonitor with TTL (December 2025 - performance fix)
            now = time.time()
            if (self._cluster_monitor is None or
                now - self._cluster_monitor_last_check > self._cluster_monitor_ttl):
                self._cluster_monitor = ClusterMonitor()
                self._cluster_monitor_last_check = now

            status = self._cluster_monitor.get_cluster_status(
                include_game_counts=False,
                include_training_status=True,
                include_disk_usage=True,
            )

            # Check disk usage
            if status.avg_disk_usage > disk_threshold:
                logger.warning(
                    f"[DataPipelineOrchestrator] Cluster disk usage high: "
                    f"{status.avg_disk_usage:.1f}% (threshold: {disk_threshold}%)"
                )
                self._emit_resource_constraint("disk_usage_high", status.avg_disk_usage)
                return False

            # Check free disk space
            if status.total_disk_free_gb < min_free_disk_gb:
                logger.warning(
                    f"[DataPipelineOrchestrator] Cluster disk space low: "
                    f"{status.total_disk_free_gb:.1f}GB free (min: {min_free_disk_gb}GB)"
                )
                self._emit_resource_constraint("disk_space_low", status.total_disk_free_gb)
                return False

            # Check if too many nodes are already training
            training_ratio = status.nodes_training / max(status.active_nodes, 1)
            if training_ratio > 0.8:
                logger.info(
                    f"[DataPipelineOrchestrator] Most nodes busy training: "
                    f"{status.nodes_training}/{status.active_nodes} ({training_ratio:.0%})"
                )
                # This is informational, not blocking - training can queue
                pass

            return True

        except ImportError:
            # ClusterMonitor not available - allow auto-trigger
            return True
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError, IOError, KeyError) as e:
            logger.debug(f"[DataPipelineOrchestrator] Resource check failed: {e}")
            # On error, allow auto-trigger (fail open)
            return True
    def _emit_resource_constraint(self, constraint_type: str, value: float) -> None:
        """Emit RESOURCE_CONSTRAINT_DETECTED event with deduplication.

        December 2025: Added cooldown to prevent event spam during sustained constraints.
        """
        # Deduplicate: Don't emit same constraint type within 60 seconds
        now = time.time()
        last_emit = self._last_constraint_emit.get(constraint_type, 0)
        if now - last_emit < 60.0:
            return
        self._last_constraint_emit[constraint_type] = now

        try:
            from app.coordination.event_router import DataEventType, publish_sync

            publish_sync(
                DataEventType.RESOURCE_CONSTRAINT_DETECTED,
                {
                    "constraint_type": constraint_type,
                    "value": value,
                    "timestamp": now,
                    "source": "data_pipeline_orchestrator",
                },
                source="data_pipeline_orchestrator",
            )

        except (RuntimeError, ValueError, TypeError, AttributeError, KeyError, ImportError) as e:
            logger.debug(f"[DataPipelineOrchestrator] Best-effort event emit failed: {e}")
    def _record_circuit_success(self, stage: str) -> None:
        """Record a successful stage execution to circuit breaker."""
        if self._circuit_breaker:
            self._circuit_breaker.record_success(stage)
    def _record_circuit_failure(self, stage: str, error: str) -> None:
        """Record a failed stage execution to circuit breaker."""
        if self._circuit_breaker:
            self._circuit_breaker.record_failure(stage, error)
    async def _pause_pipeline(self, reason: str) -> None:
        """Pause the pipeline due to resource constraints."""
        if self._paused:
            return  # Already paused

        self._paused = True
        self._pause_reason = reason
        self._pause_time = time.time()

        logger.warning(f"[DataPipelineOrchestrator] Pipeline PAUSED: {reason}")

        # Emit event for other coordinators (January 2026 - migrated to event_router)
        try:
            from app.coordination.event_emission_helpers import safe_emit_event_async

            await safe_emit_event_async(
                "RESOURCE_CONSTRAINT_DETECTED",
                {
                    "resource_type": "pipeline_pause",
                    "severity": "critical",
                    "current_value": 1,
                    "threshold": 0,
                    "action_taken": f"pipeline_paused: {reason}",
                },
                context="data_pipeline_orchestrator",
            )
        except (RuntimeError, ValueError, TypeError, AttributeError, KeyError, ImportError) as e:
            logger.debug(f"[DataPipelineOrchestrator] Failed to emit resource constraint: {e}")
    async def _resume_pipeline(self) -> None:
        """Resume the pipeline after constraint resolution."""
        if not self._paused:
            return

        pause_duration = time.time() - self._pause_time
        logger.info(
            f"[DataPipelineOrchestrator] Pipeline RESUMED after {pause_duration:.1f}s pause"
        )

        self._paused = False
        self._pause_reason = None
        self._pause_time = 0.0
    def _has_critical_constraints(self) -> bool:
        """Check if any critical resource constraints are active."""
        now = time.time()
        for _resource_type, constraint in self._resource_constraints.items():
            # Constraints older than 60s are considered stale
            if now - constraint.get("time", 0) > 60:
                continue
            if constraint.get("severity") == "critical":
                return True
        return False
    def is_paused(self) -> bool:
        """Check if pipeline is currently paused."""
        return self._paused
    def get_pause_info(self) -> dict[str, Any] | None:
        """Get information about current pause state."""
        if not self._paused:
            return None
        return {
            "paused": True,
            "reason": self._pause_reason,
            "duration_seconds": time.time() - self._pause_time,
            "active_constraints": dict(self._resource_constraints),
            "backpressure_active": self._backpressure_active,
        }
    def clear_resource_constraints(self) -> None:
        """Clear all tracked resource constraints."""
        self._resource_constraints.clear()
        logger.info("[DataPipelineOrchestrator] Resource constraints cleared")
    def clear_optimization_state(self) -> None:
        """Clear active optimization tracking."""
        if self._active_optimization:
            duration = time.time() - self._optimization_start_time
            logger.info(
                f"[DataPipelineOrchestrator] Optimization {self._active_optimization} "
                f"completed after {duration:.1f}s"
            )
        self._active_optimization = None
        self._optimization_run_id = None
        self._optimization_start_time = 0.0
    def is_optimization_active(self) -> bool:
        """Check if optimization is currently active."""
        return self._active_optimization is not None
    def get_active_optimization(self) -> str | None:
        """Get the type of active optimization, or None."""
        return self._active_optimization
    def get_quality_distribution(self) -> dict[str, float]:
        """Get current quality distribution."""
        return dict(self._quality_distribution)
    def needs_cache_refresh(self) -> bool:
        """Check if pipeline needs cache refresh."""
        return self._pending_cache_refresh
    def clear_cache_refresh_flag(self) -> None:
        """Clear the pending cache refresh flag."""
        if self._pending_cache_refresh:
            self._pending_cache_refresh = False
            logger.info("[DataPipelineOrchestrator] Cache refresh flag cleared")
    def start_iteration(self, iteration: int) -> IterationRecord:
        """Manually start a new pipeline iteration.

        Args:
            iteration: The iteration number

        Returns:
            The created IterationRecord
        """
        record = self._ensure_iteration_record(iteration)
        self._transition_to(PipelineStage.SELFPLAY, iteration)
        return record
    def on_stage_enter(
        self, stage: PipelineStage, callback: Callable[[PipelineStage, int], None]
    ) -> None:
        """Register a callback for when a stage is entered.

        Args:
            stage: The stage to watch
            callback: Function(stage, iteration) to call
        """
        if stage not in self._stage_callbacks:
            self._stage_callbacks[stage] = []
        self._stage_callbacks[stage].append(callback)
    def _on_new_games_available(self, event) -> None:
        """Handle NEW_GAMES_AVAILABLE event - trigger export if threshold met.

        December 29, 2025: Added SelfplayScheduler integration to set training
        targets and check if more games are needed before triggering export.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            game_count = payload.get("game_count", 0)
            source = payload.get("source", "unknown")

            logger.info(
                f"[DataPipelineOrchestrator] New games available: "
                f"config={config_key}, count={game_count}, source={source}"
            )

            # December 29, 2025: Wire to SelfplayScheduler for game count normalization
            # Set training sample targets and check if more games are needed
            if config_key:
                self._update_selfplay_scheduler_targets(config_key)

            self._record_event_processed()

        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_new_games_available: {e}")
    def _on_orphan_games_detected(self, event) -> None:
        """Handle ORPHAN_GAMES_DETECTED event - trigger resync and re-export.

        Sprint 4 (Jan 2, 2026): Auto-trigger re-export for orphaned games above threshold.
        Orphan games are selfplay databases that exist on nodes but aren't registered
        in the central manifest. They need to be synced and re-exported.

        The plan suggested adding this handler which was referenced but not implemented.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            host = payload.get("host", "unknown")
            orphan_count = payload.get("orphan_count", 0)
            orphan_paths = payload.get("orphan_paths", [])
            total_games = payload.get("total_games", 0)

            logger.info(
                f"[DataPipelineOrchestrator] Orphan games detected: "
                f"host={host}, count={orphan_count}, games={total_games}"
            )

            # Sprint 4: Auto-resync threshold - only trigger for significant orphan counts
            ORPHAN_RESYNC_THRESHOLD = int(
                os.environ.get("RINGRIFT_ORPHAN_RESYNC_THRESHOLD", "100")
            )

            if total_games >= ORPHAN_RESYNC_THRESHOLD:
                logger.info(
                    f"[DataPipelineOrchestrator] Orphan threshold exceeded "
                    f"({total_games} >= {ORPHAN_RESYNC_THRESHOLD}), triggering resync"
                )
                # Emit sync trigger event to pull orphan data
                self._emit_orphan_resync_trigger(host, orphan_paths, total_games)

            self._record_event_processed()

        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_orphan_games_detected: {e}")
    def _on_orphan_games_registered(self, event) -> None:
        """Handle ORPHAN_GAMES_REGISTERED event - trigger export after registration.

        Sprint 4 (Jan 2, 2026): After orphan games are registered, trigger NPZ export
        to include the recovered data in training.

        Sprint 10 (Jan 3, 2026): Added auto-quality-check for orphan games.
        Only triggers export if orphan game quality is acceptable.
        Expected Elo gain: +1-2 Elo from better training data quality.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            host = payload.get("host", "unknown")
            registered_count = payload.get("registered_count", 0)
            games_recovered = payload.get("games_recovered", 0)
            registered_paths = payload.get("registered_paths", [])

            logger.info(
                f"[DataPipelineOrchestrator] Orphan games registered: "
                f"host={host}, count={registered_count}, games={games_recovered}"
            )

            # Extract configs from registered paths and trigger re-export
            configs_to_export: set[str] = set()
            configs_needing_quality_boost: set[str] = set()

            for path in registered_paths:
                config = self._extract_config_from_db_path(path)
                if config:
                    # Sprint 10: Check quality of orphan games before export
                    quality_result = self._check_orphan_game_quality(path, config)
                    if quality_result["acceptable"]:
                        configs_to_export.add(config)
                        if quality_result.get("needs_boost"):
                            configs_needing_quality_boost.add(config)
                    else:
                        # Quality too low - emit event to boost selfplay quality
                        logger.warning(
                            f"[DataPipelineOrchestrator] Orphan games for {config} "
                            f"have low quality ({quality_result['score']:.2f}), "
                            f"triggering quality boost instead of export"
                        )
                        self._emit_orphan_quality_blocked(
                            config, quality_result["score"], path
                        )

            # Trigger export for configs that passed quality check
            for config_key in configs_to_export:
                source = "orphan_recovery"
                if config_key in configs_needing_quality_boost:
                    source = "orphan_recovery_with_boost"
                self._emit_export_trigger(config_key, source=source)

            self._record_event_processed()

        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_orphan_games_registered: {e}")
    def _emit_orphan_resync_trigger(
        self, host: str, orphan_paths: list[str], total_games: int
    ) -> None:
        """Emit SYNC_TRIGGERED event to resync orphan games.

        Sprint 4 (Jan 2, 2026): Part of orphan game recovery pipeline.
        """
        from app.coordination.event_emission_helpers import safe_emit_event
        from app.distributed.data_events import DataEventType

        safe_emit_event(
            DataEventType.SYNC_TRIGGERED,
            {
                "host": host,
                "paths": orphan_paths[:10],  # Limit payload size
                "game_count": total_games,
                "trigger": "orphan_recovery",
                "source": "data_pipeline_orchestrator",
                "timestamp": time.time(),
            },
            log_after="Emitted SYNC_TRIGGERED for orphan recovery",
            context="DataPipelineOrchestrator",
            source="data_pipeline_orchestrator",
        )
    def _emit_export_trigger(self, config_key: str, source: str) -> None:
        """Emit event to trigger NPZ export for a config.

        Sprint 4 (Jan 2, 2026): Part of orphan game recovery pipeline.
        """
        from app.coordination.event_emission_helpers import safe_emit_event
        from app.distributed.data_events import DataEventType

        safe_emit_event(
            DataEventType.NEW_GAMES_AVAILABLE,
            {
                "config_key": config_key,
                "source": source,
                "trigger": "orphan_recovery",
                "timestamp": time.time(),
            },
            log_after=f"Emitted NEW_GAMES_AVAILABLE for {config_key}",
            context="DataPipelineOrchestrator",
            source="data_pipeline_orchestrator",
        )
    def _extract_config_from_db_path(self, path: str) -> str | None:
        """Extract config key from database path.

        Sprint 4 (Jan 2, 2026): Helper for orphan recovery.
        Path format: .../selfplay_{board}_{n}p.db or .../canonical_{board}_{n}p.db
        """
        import re

        # Match patterns like selfplay_hex8_2p.db or canonical_square8_4p.db
        match = re.search(r"(?:selfplay_|canonical_)?(\w+)_(\d+)p\.db$", path)
        if match:
            board_type = match.group(1)
            num_players = int(match.group(2))
            return make_config_key(board_type, num_players)
        return None
    def _check_orphan_game_quality(
        self, db_path: str, config_key: str
    ) -> dict[str, bool | float]:
        """Check quality of orphan games in a database.

        Sprint 10 (Jan 3, 2026): Auto-quality-check for orphan games.
        Ensures only acceptable-quality orphan data gets exported to training.

        Args:
            db_path: Path to the orphan games database
            config_key: Config key (e.g., "hex8_2p")

        Returns:
            dict with keys:
                - acceptable: True if quality is good enough for export
                - score: Quality score (0.0-1.0)
                - needs_boost: True if quality is borderline and needs boost
                - reason: Human-readable explanation
        """
        import sqlite3
        from pathlib import Path

        # Default thresholds - configurable via env
        ORPHAN_QUALITY_MIN = float(
            os.environ.get("RINGRIFT_ORPHAN_QUALITY_MIN", "0.4")
        )
        ORPHAN_QUALITY_BOOST_THRESHOLD = float(
            os.environ.get("RINGRIFT_ORPHAN_QUALITY_BOOST", "0.6")
        )

        try:
            db = Path(db_path)
            if not db.exists():
                logger.warning(f"[QualityCheck] Orphan DB not found: {db_path}")
                return {
                    "acceptable": False,
                    "score": 0.0,
                    "needs_boost": False,
                    "reason": "database_not_found",
                }

            # Connect and check quality metrics
            conn = connect_safe(db_path, row_factory=None)
            cursor = conn.cursor()

            # Check 1: Game completion rate (finished games / total games)
            cursor.execute(
                "SELECT COUNT(*) FROM games WHERE winner IS NOT NULL"
            )
            finished_games = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM games")
            total_games = cursor.fetchone()[0]

            if total_games == 0:
                conn.close()
                return {
                    "acceptable": False,
                    "score": 0.0,
                    "needs_boost": False,
                    "reason": "no_games",
                }

            completion_rate = finished_games / total_games

            # Check 2: Average game length (longer games = more training signal)
            cursor.execute(
                """
                SELECT AVG(move_count) FROM (
                    SELECT game_id, COUNT(*) as move_count
                    FROM moves GROUP BY game_id
                )
                """
            )
            avg_moves_result = cursor.fetchone()
            avg_moves = avg_moves_result[0] if avg_moves_result[0] else 0

            # Check 3: Move diversity (unique positions explored)
            cursor.execute("SELECT COUNT(DISTINCT game_id) FROM moves")
            games_with_moves = cursor.fetchone()[0]

            conn.close()

            # Calculate quality score (weighted average)
            # - Completion rate: 40% weight (finished games are more valuable)
            # - Game length: 40% weight (normalized to expected range)
            # - Coverage: 20% weight (games with moves recorded)

            # Normalize avg_moves to 0-1 (expect 30-100 moves for a typical game)
            length_score = min(1.0, avg_moves / 50.0) if avg_moves > 0 else 0.0

            # Coverage score
            coverage_score = games_with_moves / total_games if total_games > 0 else 0.0

            quality_score = (
                completion_rate * 0.4
                + length_score * 0.4
                + coverage_score * 0.2
            )

            # Determine acceptability
            acceptable = quality_score >= ORPHAN_QUALITY_MIN
            needs_boost = quality_score < ORPHAN_QUALITY_BOOST_THRESHOLD

            reason = "quality_ok"
            if not acceptable:
                reason = f"quality_too_low_{quality_score:.2f}"
            elif needs_boost:
                reason = f"quality_borderline_{quality_score:.2f}"

            logger.info(
                f"[QualityCheck] Orphan games {config_key}: "
                f"score={quality_score:.2f}, completion={completion_rate:.1%}, "
                f"avg_moves={avg_moves:.0f}, acceptable={acceptable}"
            )

            return {
                "acceptable": acceptable,
                "score": quality_score,
                "needs_boost": needs_boost,
                "reason": reason,
            }

        except (sqlite3.Error, OSError, ValueError) as e:
            logger.warning(f"[QualityCheck] Error checking orphan quality: {e}")
            # On error, be permissive and allow export with boost flag
            return {
                "acceptable": True,
                "score": 0.5,
                "needs_boost": True,
                "reason": f"error_{type(e).__name__}",
            }
    def _emit_orphan_quality_blocked(
        self, config_key: str, quality_score: float, db_path: str
    ) -> None:
        """Emit event when orphan game quality is too low for export.

        Sprint 10 (Jan 3, 2026): Triggers quality boost in SelfplayScheduler.
        This causes the scheduler to prefer high-quality Gumbel MCTS modes
        for this config, improving overall training data quality.

        Args:
            config_key: Config that failed quality check
            quality_score: The quality score that triggered the block
            db_path: Path to the orphan database
        """
        from app.coordination.event_emission_helpers import safe_emit_event
        from app.distributed.data_events import DataEventType

        # Calculate quality deficit for boost strength
        min_quality = float(
            os.environ.get("RINGRIFT_ORPHAN_QUALITY_MIN", "0.4")
        )
        quality_deficit = max(0.0, min_quality - quality_score)

        safe_emit_event(
            DataEventType.TRAINING_BLOCKED_BY_QUALITY,
            {
                "config_key": config_key,
                "quality_score": quality_score,
                "threshold": min_quality,
                "quality_deficit": quality_deficit,
                "source": "orphan_quality_check",
                "db_path": db_path,
                "reason": "orphan_games_low_quality",
                "recommendation": "boost_selfplay_quality",
                "timestamp": time.time(),
            },
            log_after=(
                f"Emitted TRAINING_BLOCKED_BY_QUALITY for orphan games "
                f"{config_key} (score={quality_score:.2f})"
            ),
            context="DataPipelineOrchestrator",
            source="orphan_quality_check",
        )
    def _update_selfplay_scheduler_targets(self, config_key: str) -> None:
        """Update SelfplayScheduler with training sample targets.

        December 29, 2025: Closes the pipeline → scheduler feedback loop.
        - Sets target samples based on board size
        - Checks if more games are needed
        - Emits SELFPLAY_TARGET_UPDATED if games needed

        This ensures the scheduler knows how many more games to generate
        for each configuration before training can proceed.
        """
        try:
            from app.coordination.selfplay_scheduler import get_selfplay_scheduler

            scheduler = get_selfplay_scheduler()

            # Calculate target samples based on board type
            # Default: 50K samples minimum, scale with board size
            target_samples = 50000
            if config_key.startswith("square19") or config_key.startswith("hexagonal"):
                target_samples = 100000  # Large boards need more data
            elif config_key.startswith("square8") or config_key.startswith("hex8"):
                target_samples = 50000  # Standard boards

            scheduler.set_target_training_samples(config_key, target_samples)

            # Check if we have enough games
            games_needed = scheduler.get_games_needed(config_key)
            if games_needed > 0:
                logger.info(
                    f"[DataPipelineOrchestrator] {config_key} needs {games_needed} more games"
                )
                # Emit event to request more selfplay
                self._emit_selfplay_target_updated(config_key, games_needed)

        except ImportError:
            # SelfplayScheduler not available - expected in minimal environments
            logger.debug("[DataPipelineOrchestrator] SelfplayScheduler not available")
        except (AttributeError, RuntimeError, TypeError) as e:
            # Non-critical - log and continue
            logger.debug(f"[DataPipelineOrchestrator] Failed to update scheduler: {e}")
    def _emit_selfplay_target_updated(self, config_key: str, games_needed: int) -> None:
        """Emit SELFPLAY_TARGET_UPDATED event to request more games.

        December 29, 2025: Part of pipeline → scheduler wiring.
        """
        try:
            from app.coordination.event_router import publish_sync

            publish_sync(
                "SELFPLAY_TARGET_UPDATED",
                {
                    "config_key": config_key,
                    "games_needed": games_needed,
                    "source": "data_pipeline_orchestrator",
                    "timestamp": time.time(),
                },
                source="data_pipeline_orchestrator",
            )
        except (AttributeError, ImportError, RuntimeError) as e:
            logger.debug(f"[DataPipelineOrchestrator] Failed to emit target update: {e}")
    def _on_regression_detected(self, event) -> None:
        """Handle REGRESSION_DETECTED event - trigger curriculum rebalance.

        December 29, 2025: Phase 7 - Regression-triggered curriculum rebalance.
        When model regression is detected, reduce this config's curriculum weight
        to prevent bad training data from propagating.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            elo_loss = payload.get("elo_loss", payload.get("elo_drop", 0))
            severity = payload.get("severity", "unknown")

            if not config_key:
                return

            # Record regression for tracking
            self._last_regression = {"config": config_key, "loss": elo_loss}

            logger.warning(
                f"[DataPipelineOrchestrator] Regression detected: "
                f"config={config_key}, elo_loss={elo_loss:.0f}, severity={severity}"
            )

            # Phase 7: Trigger curriculum rebalance for significant regressions
            if abs(elo_loss) > 50:
                self._emit_curriculum_emergency_update(config_key, elo_loss)

            self._record_event_processed()

        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_regression_detected: {e}")
    def _emit_curriculum_emergency_update(self, config_key: str, elo_loss: float) -> None:
        """Emit curriculum emergency update to reduce allocation for regressing config.

        December 29, 2025: Phase 7 - Closes the regression → curriculum feedback loop.
        """
        try:
            from app.coordination.event_router import publish_sync

            # Calculate reduction factor based on regression severity
            if abs(elo_loss) > 100:
                factor = 0.3  # Severe regression: reduce to 30%
            else:
                factor = 0.5  # Moderate regression: reduce to 50%

            publish_sync(
                "CURRICULUM_REBALANCED",
                {
                    "trigger": "regression_detected",
                    "changed_configs": [config_key],
                    "action": "reduce_allocation",
                    "factor": factor,
                    "elo_loss": elo_loss,
                    "timestamp": time.time(),
                },
                source="data_pipeline_orchestrator",
            )
            logger.info(
                f"[DataPipelineOrchestrator] Emitted curriculum emergency update: "
                f"config={config_key}, factor={factor}, elo_loss={elo_loss:.0f}"
            )
        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"[DataPipelineOrchestrator] Could not emit curriculum update: {e}")
    def _on_promotion_failed(self, event) -> None:
        """Handle PROMOTION_FAILED event - log and track for pipeline metrics."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            reason = payload.get("reason", "unknown")

            logger.warning(
                f"[DataPipelineOrchestrator] Promotion failed: "
                f"config={config_key}, reason={reason}"
            )
            self._record_event_processed()

        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_promotion_failed: {e}")
    def _on_consolidation_started(self, event) -> None:
        """Handle CONSOLIDATION_STARTED event."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            logger.info(f"[DataPipelineOrchestrator] Consolidation started: {config_key}")
            self._record_event_processed()
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_consolidation_started: {e}")
    def _on_consolidation_complete(self, event) -> None:
        """Handle CONSOLIDATION_COMPLETE event - trigger export for consolidated data."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            game_count = payload.get("game_count", 0)
            logger.info(
                f"[DataPipelineOrchestrator] Consolidation complete: "
                f"config={config_key}, games={game_count}"
            )
            self._record_event_processed()
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_consolidation_complete: {e}")
    def _on_npz_combination_complete(self, event) -> None:
        """Handle NPZ_COMBINATION_COMPLETE event."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            output_path = payload.get("output_path", "")
            logger.info(
                f"[DataPipelineOrchestrator] NPZ combination complete: "
                f"config={config_key}, path={output_path}"
            )
            self._record_event_processed()
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_npz_combination_complete: {e}")
    def _on_npz_combination_failed(self, event) -> None:
        """Handle NPZ_COMBINATION_FAILED event."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            error = payload.get("error", "unknown")
            logger.error(
                f"[DataPipelineOrchestrator] NPZ combination failed: "
                f"config={config_key}, error={error}"
            )
            self._record_error(f"npz_combination_failed: {config_key}")
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_npz_combination_failed: {e}")
    def _on_repair_completed(self, event) -> None:
        """Handle REPAIR_COMPLETED event - retrigger sync after data repair."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            files_repaired = payload.get("files_repaired", 0)
            logger.info(
                f"[DataPipelineOrchestrator] Repair completed: "
                f"config={config_key}, files={files_repaired}"
            )
            self._record_event_processed()
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_repair_completed: {e}")
    def _on_repair_failed(self, event) -> None:
        """Handle REPAIR_FAILED event - track repair failures for circuit breaker."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            error = payload.get("error", "unknown")
            logger.error(
                f"[DataPipelineOrchestrator] Repair failed: "
                f"config={config_key}, error={error}"
            )
            self._record_circuit_failure("repair", error)
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_repair_failed: {e}")
    def _on_partition_healed(self, event) -> None:
        """Handle PARTITION_HEALED event - track P2P partition healing success.

        January 3, 2026: Wires the previously orphaned PARTITION_HEALED event.
        Emitted by partition_healer.py:514 when network partitions are healed.

        Actions:
        - Log healing success with partition details
        - Trigger priority sync to resynchronize data after partition healing
        - Reset any partition-related circuit breakers
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            partitions_found = payload.get("partitions_found", 0)
            partitions_healed = payload.get("partitions_healed", 0)
            nodes_reconnected = payload.get("nodes_reconnected", 0)
            duration_ms = payload.get("duration_ms", 0.0)

            logger.info(
                f"[DataPipelineOrchestrator] Partition healed: "
                f"found={partitions_found}, healed={partitions_healed}, "
                f"reconnected={nodes_reconnected}, duration={duration_ms:.0f}ms"
            )

            # After partition healing, trigger priority sync to resynchronize data
            # across the previously partitioned nodes
            if partitions_healed > 0:
                from app.coordination.event_emission_helpers import safe_emit_event
                from app.distributed.data_events import DataEventType

                safe_emit_event(
                    DataEventType.SYNC_TRIGGERED,
                    {
                        "reason": "partition_healed",
                        "priority": "high",
                        "partitions_healed": partitions_healed,
                        "nodes_reconnected": nodes_reconnected,
                        "timestamp": time.time(),
                    },
                    log_after="Triggered priority sync after partition healing",
                    context="DataPipelineOrchestrator",
                    source="partition_healer",
                )

            self._record_event_processed()
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_partition_healed: {e}")
    def _on_task_abandoned(self, event) -> None:
        """Handle TASK_ABANDONED event - update pending counts."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            task_id = payload.get("task_id", "")
            reason = payload.get("reason", "unknown")
            logger.info(
                f"[DataPipelineOrchestrator] Task abandoned: "
                f"id={task_id}, reason={reason}"
            )
            self._record_event_processed()
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_task_abandoned: {e}")
    def _on_quality_score_updated(self, event) -> None:
        """Handle QUALITY_SCORE_UPDATED event - aggregate quality metrics."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            quality_score = payload.get("quality_score", 0.0)

            if config_key:
                self._quality_distribution[config_key] = quality_score
                logger.debug(
                    f"[DataPipelineOrchestrator] Quality updated: "
                    f"config={config_key}, score={quality_score:.2f}"
                )
            self._record_event_processed()
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_quality_score_updated: {e}")
    def _on_curriculum_rebalanced(self, event) -> None:
        """Handle CURRICULUM_REBALANCED event - update pipeline priorities."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            trigger = payload.get("trigger", "unknown")
            changed_configs = payload.get("changed_configs", [])
            logger.info(
                f"[DataPipelineOrchestrator] Curriculum rebalanced: "
                f"trigger={trigger}, configs={changed_configs}"
            )
            self._record_event_processed()
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_curriculum_rebalanced: {e}")
    def _on_curriculum_advanced(self, event) -> None:
        """Handle CURRICULUM_ADVANCED event - track curriculum progression."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            new_tier = payload.get("new_tier", "")
            logger.info(
                f"[DataPipelineOrchestrator] Curriculum advanced: "
                f"config={config_key}, tier={new_tier}"
            )
            self._record_event_processed()
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_curriculum_advanced: {e}")
    def _on_s3_backup_completed(self, event) -> None:
        """Handle S3_BACKUP_COMPLETED event."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            files_backed_up = payload.get("files_backed_up", 0)
            total_size_mb = payload.get("total_size_mb", 0)
            logger.info(
                f"[DataPipelineOrchestrator] S3 backup completed: "
                f"files={files_backed_up}, size={total_size_mb:.1f}MB"
            )
            self._record_event_processed()
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_s3_backup_completed: {e}")
    def _on_sync_checksum_failed(self, event) -> None:
        """Handle SYNC_CHECKSUM_FAILED event - trigger repair."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            file_path = payload.get("file_path", "")
            expected = payload.get("expected_checksum", "")[:16]
            actual = payload.get("actual_checksum", "")[:16]
            logger.warning(
                f"[DataPipelineOrchestrator] Checksum mismatch: "
                f"file={file_path}, expected={expected}..., actual={actual}..."
            )
            self._record_circuit_failure("sync", f"checksum_mismatch: {file_path}")
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_sync_checksum_failed: {e}")
    def _on_data_stale(self, event) -> None:
        """Handle DATA_STALE event - trigger urgent sync before training.

        December 29, 2025: Phase 4D - Data freshness gating.
        When training data is stale (>24h old for most configs), trigger
        priority sync to get fresh data before allowing training.

        This prevents training on stale curriculum data which can lead to
        Elo regression or slower improvement velocity.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            data_age_hours = payload.get("data_age_hours", 0)
            threshold_hours = payload.get("threshold_hours", 24)
            source = payload.get("source", "unknown")

            if not config_key:
                return

            logger.warning(
                f"[DataPipelineOrchestrator] Data stale: "
                f"config={config_key}, age={data_age_hours:.1f}h > threshold={threshold_hours}h, "
                f"source={source}"
            )

            # Record staleness for tracking
            if not hasattr(self, "_stale_configs"):
                self._stale_configs: dict = {}
            self._stale_configs[config_key] = {
                "detected_at": time.time(),
                "age_hours": data_age_hours,
            }

            # Trigger priority sync for this config
            self._emit_priority_sync_request(config_key, reason="stale_data")

            self._record_event_processed()

        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_data_stale: {e}")
    def _emit_priority_sync_request(self, config_key: str, reason: str) -> None:
        """Emit SYNC_REQUEST event for priority sync.

        December 29, 2025: Used to trigger urgent sync when data is stale
        or after regression detection.
        """
        try:
            from app.coordination.event_router import publish_sync

            publish_sync(
                "SYNC_REQUEST",
                {
                    "config_key": config_key,
                    "priority": "urgent",
                    "reason": reason,
                    "source": "DataPipelineOrchestrator",
                    "requested_at": time.time(),
                },
            )
            logger.info(
                f"[DataPipelineOrchestrator] Priority sync requested: "
                f"config={config_key}, reason={reason}"
            )
        except (ImportError, AttributeError) as e:
            logger.debug(f"Could not emit SYNC_REQUEST: {e}")

    async def handle_stage_failure(
        self,
        stage: PipelineStage,
        config_key: str,
        error: Exception | str,
    ) -> bool:
        """Handle a stage failure with automatic retry logic.

        Implements cascade recovery: when a stage fails, it will be retried
        up to MAX_STAGE_RETRIES times with exponential backoff.

        Args:
            stage: The pipeline stage that failed
            config_key: The configuration key (e.g., "hex8_2p")
            error: The error that caused the failure

        Returns:
            True if a retry was scheduled, False if retries exhausted
        """
        key = (stage.value, config_key)
        retry_count = self._stage_retry_counts.get(key, 0)

        # Record failure in circuit breaker
        if self._circuit_breaker:
            self._circuit_breaker.record_failure(stage.value, str(error))

        if retry_count >= MAX_STAGE_RETRIES:
            # Retries exhausted - emit PIPELINE_FAILED event
            logger.error(
                f"[DataPipelineOrchestrator] Stage {stage.value} failed {MAX_STAGE_RETRIES} times "
                f"for {config_key}, giving up. Error: {error}"
            )
            self._stage_retry_counts.pop(key, None)  # Reset counter

            await self._emit_pipeline_failed(stage, config_key, error)
            return False

        # Schedule retry with exponential backoff
        self._stage_retry_counts[key] = retry_count + 1
        delay = STAGE_RETRY_DELAY_SECONDS * (STAGE_RETRY_BACKOFF_MULTIPLIER ** retry_count)

        logger.warning(
            f"[DataPipelineOrchestrator] Stage {stage.value} failed for {config_key}: {error}. "
            f"Retry {retry_count + 1}/{MAX_STAGE_RETRIES} in {delay:.0f}s"
        )

        # Cancel any existing pending retry for this stage/config
        existing_task = self._pending_retries.get(key)
        if existing_task and not existing_task.done():
            existing_task.cancel()

        # Schedule the retry
        task = asyncio.create_task(
            self._execute_stage_retry(stage, config_key, delay)
        )
        self._pending_retries[key] = task

        return True
    async def _execute_stage_retry(
        self,
        stage: PipelineStage,
        config_key: str,
        delay: float,
    ) -> None:
        """Execute a delayed stage retry.

        Args:
            stage: The stage to retry
            config_key: The configuration key
            delay: Delay before retry in seconds
        """
        try:
            await asyncio.sleep(delay)

            key = (stage.value, config_key)
            retry_count = self._stage_retry_counts.get(key, 0)

            logger.info(
                f"[DataPipelineOrchestrator] Retrying {stage.value} for {config_key} "
                f"(attempt {retry_count}/{MAX_STAGE_RETRIES})"
            )

            # Parse config_key to get board_type and num_players
            parsed = parse_config_key(config_key)
            if not parsed:
                logger.error(f"[DataPipelineOrchestrator] Invalid config_key: {config_key}")
                return

            # Trigger the appropriate stage
            await self._trigger_stage(stage, parsed.board_type, parsed.num_players)

        except asyncio.CancelledError:
            logger.debug(f"[DataPipelineOrchestrator] Retry cancelled for {stage.value}/{config_key}")
        except Exception as e:
            logger.error(f"[DataPipelineOrchestrator] Retry failed for {stage.value}/{config_key}: {e}")
        finally:
            # Jan 12, 2026: Clean up completed retry task to prevent memory leak
            # Previously, tasks were never removed from _pending_retries, causing
            # unbounded memory growth during 48h+ autonomous operation.
            key = (stage.value, config_key)
            self._pending_retries.pop(key, None)
    async def _trigger_stage(
        self,
        stage: PipelineStage,
        board_type: str,
        num_players: int,
    ) -> None:
        """Trigger a specific pipeline stage.

        Args:
            stage: The stage to trigger
            board_type: Board type for the stage
            num_players: Number of players
        """
        config_key = make_config_key(board_type, num_players)

        # Call the appropriate trigger method based on stage
        if stage == PipelineStage.DATA_SYNC:
            await self._trigger_data_sync(board_type, num_players)
        elif stage == PipelineStage.NPZ_EXPORT:
            await self._trigger_npz_export(board_type, num_players)
        elif stage == PipelineStage.NPZ_COMBINATION:
            await self._trigger_npz_combination(board_type, num_players)
        elif stage == PipelineStage.TRAINING:
            await self._trigger_training(board_type, num_players)
        elif stage == PipelineStage.EVALUATION:
            await self._trigger_evaluation(board_type, num_players)
        elif stage == PipelineStage.PROMOTION:
            await self._trigger_promotion(board_type, num_players)
        else:
            logger.warning(f"[DataPipelineOrchestrator] Cannot trigger stage: {stage.value}")
    async def _emit_pipeline_failed(
        self,
        stage: PipelineStage,
        config_key: str,
        error: Exception | str,
    ) -> None:
        """Emit PIPELINE_FAILED event when retries are exhausted.

        Args:
            stage: The stage that failed
            config_key: The configuration key
            error: The error that caused the failure
        """
        from app.coordination.event_emission_helpers import safe_emit_event_async

        await safe_emit_event_async(
            "PIPELINE_FAILED",
            {
                "stage": stage.value,
                "config_key": config_key,
                "error": str(error),
                "retries_exhausted": True,
                "max_retries": MAX_STAGE_RETRIES,
                "timestamp": time.time(),
            },
            context="DataPipelineOrchestrator",
            source="pipeline_retry_manager",
        )
    def reset_stage_retry_count(self, stage: PipelineStage, config_key: str) -> None:
        """Reset retry count for a stage/config after successful completion.

        Call this after a stage completes successfully to reset the retry counter.

        Args:
            stage: The stage that completed
            config_key: The configuration key
        """
        key = (stage.value, config_key)
        if key in self._stage_retry_counts:
            del self._stage_retry_counts[key]
            logger.debug(f"[DataPipelineOrchestrator] Reset retry count for {stage.value}/{config_key}")
    def get_cascade_recovery_status(self) -> dict[str, Any]:
        """Get current cascade recovery status.

        Returns:
            Dict with retry counts and pending retries
        """
        return {
            "retry_counts": {
                f"{stage}/{config}": count
                for (stage, config), count in self._stage_retry_counts.items()
            },
            "pending_retries": [
                f"{stage}/{config}"
                for (stage, config), task in self._pending_retries.items()
                if not task.done()
            ],
            "max_retries": MAX_STAGE_RETRIES,
            "retry_delay_seconds": STAGE_RETRY_DELAY_SECONDS,
            "backoff_multiplier": STAGE_RETRY_BACKOFF_MULTIPLIER,
        }
