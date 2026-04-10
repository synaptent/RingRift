"""Training executor actions for TrainingTriggerDaemon.

April 2026: Extracted from training_trigger_daemon.py (Part 3 Phase 3).
This module contains the operational training decision, data readiness,
queue dispatch, local execution, timeout, and diagnostic helpers. The daemon
module keeps lifecycle, scheduling, and event subscriptions.
"""
from __future__ import annotations

import asyncio
import contextlib
import datetime
import logging
import math
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from app.config.coordination_defaults import QualityGateDefaults, SyncDefaults
from app.config.env import env
from app.config.ports import get_local_p2p_status_url
from app.coordination.event_utils import make_config_key, parse_config_key
from app.coordination.training_architecture_selector import (
    apply_velocity_amplification,
    get_training_params_for_intensity,
    select_architecture_for_training,
)
from app.coordination.training_data_availability import (
    DataAvailabilityChecker,
    DataAvailabilityConfig,
    check_cluster_availability,
    check_gpu_availability,
    parse_config_from_filename,
    scan_local_npz_files,
)
from app.coordination.training_decision_engine import (
    check_confidence_early_trigger as check_confidence_early_trigger_fn,
    compute_adaptive_max_data_age,
    compute_dynamic_sample_threshold,
    compute_velocity_adjusted_cooldown,
)
from app.coordination.training_execution import (
    TrainingExecutionConfig,
    TrainingExecutor,
    TrainingResult,
    emit_training_complete as _emit_training_complete_impl,
    emit_training_failed as _emit_training_failed_impl,
    graceful_kill_process as _graceful_kill_process_impl,
)
from app.coordination.training_quality_gates import (
    DATA_STARVED_THRESHOLD,
    MINIMUM_QUALITY_FLOOR,
    TRAINING_STALL_HOURS,
    QualityGateResult,
    apply_confidence_weighting,
    check_quality_gate_conditions,
    compute_decayed_quality_score,
    compute_quality_confidence,
    get_quality_from_state,
    intensity_from_quality,
)
from app.coordination.training_retry_manager import (
    get_adaptive_max_data_age,
    get_velocity_adjusted_cooldown,
)
from app.coordination.training_trigger_types import (
    ArchitectureSpec,
    ConfigTrainingState,
    MultiArchitectureConfig,
    TrainingDecision,
)

logger = logging.getLogger(__name__)

try:
    from app.distributed.circuit_breaker import get_training_breaker
    HAS_CIRCUIT_BREAKER = True
except ImportError:
    HAS_CIRCUIT_BREAKER = False
    get_training_breaker = None

try:
    from app.coordination.p2p_integration import (
        is_p2p_available,
        with_training_lock,
    )
    HAS_DISTRIBUTED_LOCK = True
except ImportError:
    HAS_DISTRIBUTED_LOCK = False
    is_p2p_available = None  # type: ignore[assignment]
    with_training_lock = None  # type: ignore[assignment]


class TrainingExecutorActionsMixin:
    """Operational training helpers used by TrainingTriggerDaemon."""

    async def _maybe_trigger_training(self, config_key: str) -> bool:
        """Check conditions and trigger training for all applicable architectures.

        December 30, 2025: Updated to support multi-architecture training.
        Iterates over architectures configured for this config and triggers
        training for each one that hasn't trained recently.
        """
        state = self._training_states.get(config_key)
        if not state:
            return False

        # Check base conditions (applies to all architectures)
        can_train, reason = await self._check_training_conditions(config_key)

        if not can_train:
            logger.debug(f"[TrainingTriggerDaemon] {config_key}: Cannot train - {reason}")
            return False

        # April 2026: Only train with the canonical model architecture (v2).
        # Multi-architecture training (v3/v4/v5/v5-heavy-large) crashes because
        # init weights from canonical v2 models are incompatible with other archs,
        # causing 4/5 dispatches to fail with cuda_error:rc=1 and accumulating
        # failure counts that block all future training for that config.
        architectures = [ArchitectureSpec(
            name="v2", enabled=True, configs=["*"], priority=1.0
        )]

        triggered_any = False
        for arch in architectures:
            # Check architecture-specific cooldown
            arch_key = (config_key, arch.name)
            last_train_time = self._architecture_training_times.get(arch_key, 0.0)
            time_since_training = time.time() - last_train_time
            cooldown_seconds = self._architecture_config.min_hours_between_runs * 3600

            if time_since_training < cooldown_seconds:
                remaining_hours = (cooldown_seconds - time_since_training) / 3600
                logger.debug(
                    f"[TrainingTriggerDaemon] {config_key}/{arch.name}: "
                    f"Architecture cooldown ({remaining_hours:.1f}h remaining)"
                )
                continue

            # Check if already training this architecture
            if self._active_architecture_training.get(arch_key, False):
                logger.debug(
                    f"[TrainingTriggerDaemon] {config_key}/{arch.name}: "
                    f"Already training this architecture"
                )
                continue

            # Trigger training for this architecture with safe error handling (Sprint 17.4)
            logger.info(
                f"[TrainingTriggerDaemon] Triggering training for {config_key} "
                f"with architecture {arch.name}"
            )
            task = self._safe_create_task(
                self._run_training(config_key, arch),
                context=f"run_training:{config_key}:{arch.name}",
            )
            task.add_done_callback(
                lambda t, ck=config_key, a=arch.name: self._on_training_task_done(t, ck, a)
            )
            # Track with architecture suffix
            task_key = f"{config_key}:{arch.name}"
            self._active_training_tasks[task_key] = task
            self._active_architecture_training[arch_key] = True
            self._architecture_training_times[arch_key] = time.time()
            triggered_any = True

        return triggered_any

    async def _process_training_retry_queue(self) -> None:
        """Process pending training retries whose delay has elapsed.

        December 29, 2025 (Phase 3): Called at the start of each cycle
        to re-attempt failed training jobs with exponential backoff.
        """
        if not self._training_retry_queue:
            return

        now = time.time()
        ready_for_retry: list[tuple[str, str, int, int, str]] = []
        remaining: list[tuple[str, str, int, int, float, str]] = []

        while self._training_retry_queue:
            item = self._training_retry_queue.popleft()
            config_key, board_type, num_players, attempts, next_retry_time, error = item

            if next_retry_time <= now:
                ready_for_retry.append((config_key, board_type, num_players, attempts, error))
            else:
                remaining.append(item)

        # Put back items not yet ready
        for item in remaining:
            self._training_retry_queue.append(item)

        # Process ready items
        for config_key, board_type, num_players, attempts, error in ready_for_retry:
            state = self._get_or_create_state(config_key, board_type, num_players)

            # Skip if already training
            if state.training_in_progress:
                logger.debug(
                    f"[TrainingTriggerDaemon] Retry deferred (already training): {config_key}"
                )
                # Re-queue with same attempt count but short delay
                self._training_retry_queue.append(
                    (config_key, board_type, num_players, attempts, now + 60.0, error)
                )
                continue

            logger.info(
                f"[TrainingTriggerDaemon] Retrying training #{attempts} for {config_key}"
            )

            # Trigger training check (will go through normal validation)
            can_train, reason = await self._check_training_conditions(config_key)
            if can_train:
                success = await self._trigger_training(config_key, state)
                if success:
                    self._retry_stats["retries_succeeded"] += 1
                    logger.info(
                        f"[TrainingTriggerDaemon] Retry #{attempts} succeeded for {config_key}"
                    )
                else:
                    # Re-queue for next attempt
                    self._queue_training_retry(
                        config_key, board_type, num_players,
                        f"retry failed: {reason}", attempts
                    )
            else:
                # Re-queue for later (conditions not met yet)
                # December 30, 2025: Use RetryConfig base_delay for consistency
                delay = self._retry_config.base_delay / 2  # Shorter delay for condition check
                self._training_retry_queue.append(
                    (config_key, board_type, num_players, attempts, now + delay, error)
                )
                logger.debug(
                    f"[TrainingTriggerDaemon] Retry deferred for {config_key}: {reason}"
                )

    async def _trigger_priority_sync(
        self, config_key: str, board_type: str, num_players: int
    ) -> bool:
        """Trigger priority data sync for a configuration (Dec 2025 Phase 2A).

        Uses SyncFacade to request immediate sync of training data for the
        specified configuration.

        Args:
            config_key: Configuration identifier (e.g., "hex8_2p")
            board_type: Board type
            num_players: Number of players

        Returns:
            True if sync was triggered successfully
        """
        try:
            from app.coordination.sync_facade import get_sync_facade

            facade = get_sync_facade()
            # Dec 30, 2025: Add timeout protection to prevent hanging indefinitely
            # on network issues during 48h autonomous operation
            response = await asyncio.wait_for(
                facade.trigger_priority_sync(
                    reason="training_data_stale",
                    config_key=config_key,
                    data_type="training",
                ),
                timeout=300.0,  # 5 minute timeout for sync operation
            )

            if response.get("success"):
                logger.info(
                    f"[TrainingTriggerDaemon] Priority sync triggered for {config_key}"
                )
                return True
            else:
                logger.warning(
                    f"[TrainingTriggerDaemon] Priority sync failed for {config_key}: "
                    f"{response.get('error', 'unknown')}"
                )
                return False

        except asyncio.TimeoutError:
            logger.warning(
                f"[TrainingTriggerDaemon] Priority sync timed out for {config_key} after 5min"
            )
            return False
        except ImportError:
            logger.debug("[TrainingTriggerDaemon] SyncFacade not available for priority sync")
            return False
        except Exception as e:
            logger.warning(f"[TrainingTriggerDaemon] Error triggering priority sync: {e}")
            return False

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
        """Signal that curriculum should advance for a stagnant configuration.

        Dec 29, 2025: Now emits DataEventType.CURRICULUM_ADVANCEMENT_NEEDED
        which is handled by MomentumToCurriculumBridge._on_curriculum_advancement_needed().
        """
        try:
            from app.coordination.event_router import publish
            from app.distributed.data_events import DataEventType

            await publish(
                event_type=DataEventType.CURRICULUM_ADVANCEMENT_NEEDED,
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
        except (ImportError, AttributeError, RuntimeError) as e:
            # ImportError: event modules not available
            # AttributeError: DataEventType enum missing
            # RuntimeError: publish operation failed
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
        # January 2026: Defensive validation - ensure board_type is a string
        # This protects against event payloads containing tuples instead of strings
        if board_type is not None and not isinstance(board_type, str):
            logger.warning(
                f"[TrainingTriggerDaemon] Invalid board_type type for {config_key}: "
                f"expected str, got {type(board_type).__name__}={board_type}"
            )
            # Try to extract string if it's a tuple (board_type, num_players)
            if isinstance(board_type, tuple) and len(board_type) >= 1:
                board_type = str(board_type[0]) if board_type[0] else None
            else:
                board_type = None

        if config_key not in self._training_states:
            # Parse config_key if board_type/num_players not provided
            if not board_type or not num_players:
                parsed_board, parsed_players = self._parse_config_from_filename(config_key)
                if parsed_board and parsed_players:
                    board_type = parsed_board
                    num_players = parsed_players
                else:
                    # Use canonical parse_config_key utility
                    parsed = parse_config_key(config_key)
                    if parsed:
                        board_type = parsed.board_type
                        num_players = parsed.num_players
                    else:
                        board_type = config_key
                        num_players = 2

            self._training_states[config_key] = ConfigTrainingState(
                config_key=config_key,
                board_type=board_type,
                num_players=num_players,
            )

        return self._training_states[config_key]

    def _parse_config_from_filename(self, name: str) -> tuple[str | None, int | None]:
        """Parse board_type and num_players from filename.

        December 30, 2025: Migrated to use consolidated extraction utilities.

        Handles various naming patterns:
        - hex8_2p.npz -> (hex8, 2)
        - square8_3p_fresh.npz -> (square8, 3)
        - canonical_hexagonal_4p_trained.npz -> (hexagonal, 4)

        Returns:
            (board_type, num_players) or (None, None) if not parseable.
        """
        # Use consolidated utilities for config extraction
        config_key = extract_config_from_path(name)
        if config_key:
            parsed = parse_config_key(config_key)
            if parsed:
                return parsed.board_type, parsed.num_players
        return None, None

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

        # December 29, 2025 (Phase 4): Check evaluation backpressure
        # When evaluation queue is full, pause training to let evaluations catch up
        if self._evaluation_backpressure:
            return False, "evaluation backpressure active (queue full)"

        # Phase 4: Check circuit breaker before triggering training
        if HAS_CIRCUIT_BREAKER and get_training_breaker:
            breaker = get_training_breaker()
            if not breaker.can_execute(config_key):
                return False, f"circuit open for {config_key}"

        # 2. Check training cooldown (December 29, 2025: velocity-adjusted)
        time_since_training = time.time() - state.last_training_time
        # Use velocity-adjusted cooldown instead of fixed cooldown
        cooldown_seconds = compute_velocity_adjusted_cooldown(
            self.config.training_cooldown_hours, state.elo_velocity, state.elo_velocity_trend,
        )
        if time_since_training < cooldown_seconds:
            remaining = (cooldown_seconds - time_since_training) / 3600
            trend_info = f", velocity_trend={state.elo_velocity_trend}" if state.elo_velocity_trend != "stable" else ""
            return False, f"cooldown active ({remaining:.1f}h remaining{trend_info})"

        # 3. Check minimum samples before expensive freshness/quality/aggregation work.
        # Configs that are obviously under the sample floor cannot train anyway, so
        # fail fast here instead of scanning manifests or remote sources.
        if state.npz_sample_count < self.config.confidence_min_samples:
            min_samples = compute_dynamic_sample_threshold(
                config_key, state.num_players or 2,
                base_threshold=self.config.min_samples_threshold,
            )
            if state.npz_sample_count < min_samples:
                return False, f"insufficient samples ({state.npz_sample_count} < {min_samples})"

        # 4. Check data freshness (December 2025: use training_freshness for sync)
        # January 2026 (Phase 4.1): Auto-sync on stale data instead of blocking
        # January 3, 2026: Adaptive data freshness based on velocity trend
        # - Plateauing configs get 3x threshold (more lenient) to break stalls
        # - Accelerating configs get 0.5x threshold (stricter) to maintain quality
        data_age_hours = (time.time() - state.last_npz_update) / 3600
        # Feb 2026: Compute game_count for starved config detection in adaptive age
        _game_count_for_age = None
        try:
            from app.utils.game_discovery import count_games_for_config as _cgfc
            parsed_ck = parse_config_key(config_key)
            if parsed_ck:
                _game_count_for_age = _cgfc(parsed_ck.board_type, parsed_ck.num_players)
        except (ImportError, ValueError, OSError):
            pass
        adaptive_max_age = compute_adaptive_max_data_age(
            self.config.max_data_age_hours, state.elo_velocity_trend,
            state.last_training_time, time.time(), game_count=_game_count_for_age,
        )
        if data_age_hours > adaptive_max_age:
            # December 29, 2025: Strict mode - fail immediately without sync attempt
            if self.config.strict_freshness_mode:
                return False, f"data too old ({data_age_hours:.1f}h) [strict mode - no sync]"

            # January 2026 (Phase 4.1): Check if data is "very stale" (>2x adaptive threshold)
            # Very stale data → proceed with warning (don't block indefinitely)
            very_stale_threshold = adaptive_max_age * 2
            if data_age_hours > very_stale_threshold:
                # Data is very old - proceed anyway with warning to prevent indefinite blocks
                logger.warning(
                    f"[TrainingTriggerDaemon] {config_key}: proceeding with very stale data "
                    f"(age={data_age_hours:.1f}h > {very_stale_threshold:.1f}h threshold). "
                    f"Triggering background sync."
                )
                # Trigger background sync with safe error handling (Sprint 17.4)
                self._safe_create_task(
                    self._trigger_priority_sync(config_key, state.board_type, state.num_players),
                    context=f"priority_sync_very_stale:{config_key}",
                )
                # Continue with training (data will be fresher next time)
            elif self.config.enforce_freshness_with_sync:
                # Moderately stale - try to sync and wait for fresh data
                fresh = await self._ensure_fresh_data(state.board_type, state.num_players)
                if not fresh:
                    return False, f"data stale ({data_age_hours:.1f}h), sync triggered but not ready"
                # Sync succeeded, continue with training check
                logger.info(
                    f"[TrainingTriggerDaemon] {config_key}: data refreshed via sync"
                )
            else:
                # Not enforcing with sync - just trigger background sync and block
                # Use safe task creation for error handling (Sprint 17.4)
                self._safe_create_task(
                    self._trigger_priority_sync(config_key, state.board_type, state.num_players),
                    context=f"priority_sync_stale:{config_key}",
                )
                return False, f"data stale ({data_age_hours:.1f}h), sync triggered"

        # January 2026: Log cluster-wide game counts for visibility.
        # This is informational only, so keep it off the RPC critical path.
        if self._running:
            self._safe_create_task(
                self._log_aggregated_game_counts(config_key, state.board_type, state.num_players),
                context=f"log_aggregated_game_counts:{config_key}",
            )

        # 3.5 January 2026 Sprint 10: Check data quality before training
        # This ensures training only proceeds with high-quality data.
        # Expected improvement: +15-20 Elo from tighter quality feedback.
        quality_ok, quality_reason = await self._check_quality_gate(config_key)
        if not quality_ok:
            return False, quality_reason

        # 3.6 January 6, 2026 (Session 17.41): Graduated minimum game requirement
        # Use graduated thresholds based on player count to enable training earlier
        # for 4p configs while still requiring sufficient data quality.
        # 2p: 50, 3p: 70, 4p: 100 (synced with PromotionGameDefaults)
        try:
            from app.config.coordination_defaults import PromotionGameDefaults
            from app.utils.game_discovery import count_games_for_config

            game_count = count_games_for_config(state.board_type, state.num_players)
            min_games = PromotionGameDefaults.get_min_games(state.num_players)

            if game_count < min_games:
                return False, (
                    f"insufficient games for {state.num_players}p config "
                    f"({game_count} < {min_games} graduated minimum)"
                )
        except Exception as e:
            logger.warning(
                f"[TrainingTriggerDaemon] {config_key}: could not check game count: {e}"
            )

        # 5. Check minimum samples (with confidence-based early trigger)
        # Dec 29, 2025: Try confidence-based early trigger first
        # This allows training to start earlier when statistical confidence is high
        if state.npz_sample_count >= self.config.confidence_min_samples:
            early_trigger, early_reason = check_confidence_early_trigger_fn(
                config_key, state.npz_sample_count,
                min_samples=self.config.confidence_min_samples,
                target_ci_width=self.config.confidence_target_ci_width,
                confidence_enabled=self.config.confidence_early_trigger_enabled,
            )
            if early_trigger:
                logger.info(
                    f"[TrainingTriggerDaemon] {config_key}: early trigger - {early_reason}"
                )
                # Skip the min_samples check - confidence is high enough
            else:
                # Fall back to dynamic threshold from ImprovementOptimizer
                # Phase 5 (Dec 2025): Lower when on promotion streak, higher when struggling
                min_samples = compute_dynamic_sample_threshold(
                    config_key, state.num_players or 2,
                    base_threshold=self.config.min_samples_threshold,
                )
                if state.npz_sample_count < min_samples:
                    return False, f"insufficient samples ({state.npz_sample_count} < {min_samples}), {early_reason}"
        else:
            # Below confidence minimum - use dynamic threshold
            min_samples = compute_dynamic_sample_threshold(
                config_key, state.num_players or 2,
                base_threshold=self.config.min_samples_threshold,
            )
            if state.npz_sample_count < min_samples:
                return False, f"insufficient samples ({state.npz_sample_count} < {min_samples})"

        # 6. Check if idle GPU available (optional - allow training anyway)
        gpu_available = await self._check_gpu_availability()
        if not gpu_available:
            logger.warning(f"[TrainingTriggerDaemon] {config_key}: No idle GPU, proceeding anyway")

        # 7. Check concurrent training limit
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

    async def get_training_decision(self, config_key: str) -> TrainingDecision:
        """Get detailed training decision for a config (December 30, 2025 - RPC API).

        This method exposes the full training decision logic for external callers,
        including the P2P orchestrator's /training/trigger-decision endpoint.

        Args:
            config_key: Config key like "hex8_2p"

        Returns:
            TrainingDecision with full condition details
        """
        state = self._training_states.get(config_key)
        if not state:
            return TrainingDecision(
                config_key=config_key,
                can_trigger=False,
                reason="config not tracked",
            )

        # Calculate all condition values
        time_since_training = time.time() - state.last_training_time
        cooldown_seconds = compute_velocity_adjusted_cooldown(
            self.config.training_cooldown_hours, state.elo_velocity, state.elo_velocity_trend,
        )
        cooldown_remaining = max(0, cooldown_seconds - time_since_training) / 3600

        data_age_hours = (time.time() - state.last_npz_update) / 3600

        min_samples = compute_dynamic_sample_threshold(
            config_key, state.num_players or 2,
            base_threshold=self.config.min_samples_threshold,
        )

        active_count = sum(
            1 for s in self._training_states.values() if s.training_in_progress
        )

        # Check circuit breaker
        circuit_breaker_open = False
        if HAS_CIRCUIT_BREAKER and get_training_breaker:
            breaker = get_training_breaker()
            circuit_breaker_open = not breaker.can_execute(config_key)

        # GPU availability (quick check, don't block)
        # Jan 2026: Reduced timeout from 5s to 2s for faster training trigger decisions
        gpu_available = True
        try:
            gpu_available = await asyncio.wait_for(
                self._check_gpu_availability(), timeout=2.0
            )
        except asyncio.TimeoutError:
            pass

        # Get the actual decision
        can_trigger, reason = await self._check_training_conditions(config_key)

        return TrainingDecision(
            config_key=config_key,
            can_trigger=can_trigger,
            reason=reason,
            training_in_progress=state.training_in_progress,
            intensity_paused=state.training_intensity == "paused",
            evaluation_backpressure=self._evaluation_backpressure,
            circuit_breaker_open=circuit_breaker_open,
            cooldown_remaining_hours=cooldown_remaining,
            data_age_hours=data_age_hours,
            max_data_age_hours=self.config.max_data_age_hours,
            sample_count=state.npz_sample_count,
            sample_threshold=min_samples,
            gpu_available=gpu_available,
            concurrent_training_count=active_count,
            max_concurrent_training=self.config.max_concurrent_training,
            npz_path=state.npz_path,
            current_elo=state.last_elo,
            elo_velocity=state.elo_velocity,
            elo_velocity_trend=state.elo_velocity_trend,
        )

    def get_tracked_configs(self) -> list[str]:
        """Get list of all tracked config keys (December 30, 2025 - RPC API)."""
        return list(self._training_states.keys())

    def _get_cached_quality(self, config_key: str) -> float | None:
        """Get cached quality score if fresh.

        January 5, 2026 (Phase 7.9): Quality assessment cache to reduce SQLite lookups.
        Returns cached score if within TTL, None otherwise.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")

        Returns:
            Cached quality score if fresh, None if stale or not cached
        """
        if config_key in self._quality_cache:
            score, timestamp = self._quality_cache[config_key]
            if time.time() - timestamp < self._quality_cache_ttl:
                return score
        return None

    def _update_quality_cache(self, config_key: str, quality: float) -> None:
        """Update quality cache with fresh score.

        January 5, 2026 (Phase 7.9): Cache quality results for 10 seconds.
        """
        self._quality_cache[config_key] = (quality, time.time())

    async def _check_quality_gate(self, config_key: str) -> tuple[bool, str]:
        """Check if data quality meets minimum threshold for training.

        January 2026 Sprint 10: Tighter quality feedback before training.
        Blocks training if data quality is below threshold, ensuring we only
        train on high-quality data.

        January 3, 2026: Added quality confidence decay. When quality data is stale
        (no recent updates), the effective quality score decays toward a floor value.
        This prevents stale high-quality assessments from blocking training indefinitely.

        Expected improvement: +15-20 Elo from better training data quality.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")

        Returns:
            Tuple of (quality_ok, reason):
            - (True, "quality ok (X.XX)") if quality >= threshold
            - (False, "quality too low (X.XX < threshold)") if quality < threshold
        """
        # January 3, 2026 (Sprint 10): Use board-specific quality thresholds
        # Larger/more complex boards need higher quality data for effective training
        # See QualityGateDefaults.QUALITY_GATES for per-config thresholds
        try:
            from app.config.coordination_defaults import QualityGateDefaults
            quality_threshold = QualityGateDefaults.get_quality_threshold(config_key)
        except ImportError:
            quality_threshold = 0.50  # Fallback to default

        try:
            # January 5, 2026 (Phase 7.9): Check cache first to reduce SQLite lookups
            cached = self._get_cached_quality(config_key)
            if cached is not None:
                quality = cached
                logger.debug(
                    f"[TrainingTriggerDaemon] {config_key}: using cached quality {quality:.2f}"
                )
            else:
                # Cache miss - fetch from quality_monitor
                from app.coordination.quality_monitor_daemon import get_quality_monitor

                quality_monitor = get_quality_monitor()
                quality = quality_monitor.get_quality_for_config(config_key)

                if quality is None:
                    # Jan 3, 2026: No fresh quality data - try decayed stored quality
                    state = self._training_states.get(config_key)
                    if state and state.last_quality_update > 0:
                        decayed_quality = compute_decayed_quality_score(
                            last_quality_score=state.last_quality_score,
                            last_quality_update=state.last_quality_update,
                            current_time=time.time(),
                            decay_enabled=self.config.quality_decay_enabled,
                            half_life_hours=self.config.quality_decay_half_life_hours,
                            decay_floor=self.config.quality_decay_floor,
                        )
                        logger.debug(
                            f"[TrainingTriggerDaemon] {config_key}: using decayed quality "
                            f"{decayed_quality:.2f} (original: {state.last_quality_score:.2f})"
                        )
                        quality = decayed_quality
                    else:
                        # No quality data at all — allow training with default quality.
                        # Blocking here creates a chicken-and-egg deadlock: training needs
                        # quality scores, but quality scores require evaluated models.
                        logger.info(
                            f"[TrainingTriggerDaemon] {config_key}: no quality data available, "
                            f"proceeding with default quality (bootstrap mode)"
                        )
                        quality = quality_threshold  # Use threshold as default to pass gate

                # Update cache with fresh quality score
                if quality is not None:
                    self._update_quality_cache(config_key, quality)

            if quality < quality_threshold:
                # January 3, 2026: Relax quality gate for data-starved or stalled configs
                # This prevents configs with limited data from being permanently blocked
                # while maintaining quality floor to prevent garbage data training
                # Jan 4, 2026 - Sprint 17.9: Constants now imported from training_quality_gates.py

                allow_degraded = False
                degraded_reason = ""

                # Check if quality meets minimum floor
                if quality >= MINIMUM_QUALITY_FLOOR:
                    # Get game count for this config
                    try:
                        from app.utils.game_discovery import count_games_for_config
                        from app.coordination.event_utils import parse_config_key

                        parsed = parse_config_key(config_key)
                        if parsed:
                            game_count = count_games_for_config(
                                parsed.board_type, parsed.num_players
                            )
                            if game_count < DATA_STARVED_THRESHOLD:
                                allow_degraded = True
                                degraded_reason = (
                                    f"bootstrap mode ({game_count} < {DATA_STARVED_THRESHOLD} games)"
                                )
                    except (ImportError, ValueError, OSError) as e:
                        logger.debug(f"[TrainingTriggerDaemon] Game count check failed: {e}")

                    # Check if training is stalled (emergency override)
                    if not allow_degraded:
                        state = self._training_states.get(config_key)
                        if state and state.last_training_time > 0:
                            hours_since = (time.time() - state.last_training_time) / 3600
                            if hours_since > TRAINING_STALL_HOURS:
                                allow_degraded = True
                                degraded_reason = f"training stalled ({hours_since:.1f}h > {TRAINING_STALL_HOURS}h)"

                if allow_degraded:
                    logger.info(
                        f"[TrainingTriggerDaemon] {config_key}: quality gate RELAXED - "
                        f"allowing degraded quality ({quality:.2f}) for {degraded_reason}"
                    )
                    return True, f"quality degraded but allowed ({quality:.2f}, {degraded_reason})"

                # Quality too low - block training and emit event
                logger.warning(
                    f"[TrainingTriggerDaemon] {config_key}: quality gate FAILED "
                    f"(quality={quality:.2f} < threshold={quality_threshold})"
                )

                # Emit TRAINING_BLOCKED_BY_QUALITY event for feedback loop
                from app.coordination.event_emission_helpers import safe_emit_event

                safe_emit_event(
                    "TRAINING_BLOCKED_BY_QUALITY",
                    {
                        "config_key": config_key,
                        "quality_score": quality,
                        "threshold": quality_threshold,
                        "reason": "pre_training_quality_gate",
                        "source": "training_trigger_daemon",
                    },
                    context="TrainingTriggerDaemon",
                )

                return False, f"quality too low ({quality:.2f} < {quality_threshold})"

            # Quality is acceptable
            logger.debug(
                f"[TrainingTriggerDaemon] {config_key}: quality gate passed "
                f"(quality={quality:.2f} >= {quality_threshold})"
            )
            return True, f"quality ok ({quality:.2f})"

        except ImportError:
            logger.error(
                f"[TrainingTriggerDaemon] {config_key}: quality monitor not available, "
                f"blocking training"
            )
            return False, "quality monitor not available"
        except Exception as e:
            # Mar 30, 2026: FAIL on error. Previous optimistic default allowed
            # training on potentially corrupt data when quality check code was broken.
            logger.error(f"[TrainingTriggerDaemon] Quality check FAILED: {e}")
            return False, f"quality check error: {e}"

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

    async def _check_cluster_availability(self) -> bool:
        """Check if cluster is available with fast timeout (Jan 2, 2026).

        Used by auto_detect_local_mode to determine if we should fall back
        to local-only mode when cluster is unreachable.

        Returns:
            True if cluster is reachable, False otherwise
        """
        timeout = self._daemon_config.cluster_availability_timeout_seconds

        try:
            # Check P2P status endpoint
            import aiohttp

            p2p_url = get_local_p2p_status_url()
            async with aiohttp.ClientSession() as session:
                async with session.get(p2p_url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        alive_peers = data.get("alive_peers", 0)
                        if alive_peers > 0:
                            return True
                        # No peers alive - cluster not functional
                        logger.debug(
                            "[TrainingTriggerDaemon] Cluster check: no alive peers"
                        )
                        return False

        except ImportError:
            logger.debug("[TrainingTriggerDaemon] aiohttp not available for cluster check")
        except asyncio.TimeoutError:
            logger.debug(
                f"[TrainingTriggerDaemon] Cluster check timed out after {timeout}s"
            )
        except Exception as e:
            logger.debug(f"[TrainingTriggerDaemon] Cluster check failed: {e}")

        return False

    def _scan_local_npz_files(self) -> list[tuple[str, str, int, Path]]:
        """Scan local NPZ files for training in local-only mode (Jan 2, 2026).

        Returns list of (config_key, board_type, num_players, npz_path) tuples
        for all valid NPZ files found locally.
        """
        results: list[tuple[str, str, int, Path]] = []

        training_dir = Path(__file__).resolve().parent.parent.parent / "data" / "training"
        if not training_dir.exists():
            return results

        for npz_path in training_dir.glob("*.npz"):
            board_type, num_players = self._parse_config_from_filename(npz_path.stem)
            if board_type is None or num_players is None:
                continue

            config_key = make_config_key(board_type, num_players)
            results.append((config_key, board_type, num_players, npz_path))

        return results

    @staticmethod
    def _extract_npz_sample_count(data: Any) -> int:
        """Extract sample count from a loaded NPZ archive."""
        for key in ("values", "features", "states"):
            array = data.get(key)
            if array is not None:
                return len(array)
        return 0

    def _get_npz_metadata(
        self,
        config_key: str,
        npz_path: Path,
        *,
        validate: bool = False,
    ) -> tuple[float, int, str] | None:
        """Return cached NPZ metadata, loading the file only when it changed."""
        path_str = str(npz_path)

        try:
            current_mtime = npz_path.stat().st_mtime
        except OSError as e:
            logger.debug(f"[TrainingTriggerDaemon] Could not stat NPZ {npz_path}: {e}")
            return None

        cached = self._npz_cache.get(config_key)
        if cached and cached[2] == path_str and current_mtime <= cached[0]:
            return cached

        if validate:
            try:
                from app.training.data_validation import is_npz_valid
                if not is_npz_valid(npz_path):
                    logger.warning(f"Skipping invalid NPZ: {npz_path}")
                    return None
            except ImportError:
                pass

        try:
            from app.utils.numpy_utils import safe_load_npz

            with safe_load_npz(npz_path) as data:
                sample_count = self._extract_npz_sample_count(data)
        except (FileNotFoundError, OSError, ValueError, ImportError) as e:
            logger.debug(f"[TrainingTriggerDaemon] Failed to load NPZ metadata for {npz_path}: {e}")
            return None

        metadata = (current_mtime, sample_count, path_str)
        self._npz_cache[config_key] = metadata
        return metadata

    async def _ensure_fresh_data(self, board_type: str, num_players: int) -> bool:
        """Ensure training data is fresh, triggering sync if needed (December 2025).

        Uses training_freshness module to check data age and trigger sync
        if data is stale. This closes the data freshness feedback loop.

        Jan 2, 2026: In local-only mode, skips sync and just checks if local data exists.

        Args:
            board_type: Board type for training
            num_players: Number of players

        Returns:
            True if data is now fresh, False if sync failed or timed out
        """
        # Jan 2, 2026: In local-only mode, just check if local NPZ exists
        if self._local_only_mode:
            config_key = make_config_key(board_type, num_players)
            local_npz = Path(f"data/training/{config_key}.npz")
            if local_npz.exists():
                logger.debug(
                    f"[TrainingTriggerDaemon] Local-only mode: using existing NPZ for {config_key}"
                )
                return True
            logger.debug(
                f"[TrainingTriggerDaemon] Local-only mode: no NPZ for {config_key}"
            )
            return False

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
                config_key = make_config_key(board_type, num_players)
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

    async def _check_all_data_sources(
        self, config_key: str, min_samples_needed: int
    ) -> tuple[int, str | None]:
        """Check all sources for available training data (January 2026).

        Queries local NPZ files, TrainingDataManifest (S3/OWC), and ClusterManifest
        to find total available samples across all data sources.

        Jan 2, 2026: In local-only mode, skips remote data sources (S3, OWC, Cluster).

        Args:
            config_key: Configuration identifier (e.g., "hex8_2p")
            min_samples_needed: Minimum samples required for training

        Returns:
            Tuple of (total_samples_available, best_remote_path_if_any)
        """
        total_samples = 0
        best_remote_path: str | None = None

        # 1. Check local NPZ files
        try:
            local_npz = Path(f"data/training/{config_key}.npz")
            if local_npz.exists():
                metadata = self._get_npz_metadata(config_key, local_npz)
                if metadata is not None:
                    _mtime, local_count, _path = metadata
                    total_samples += local_count
                    logger.debug(
                        f"[TrainingTriggerDaemon] Local NPZ for {config_key}: {local_count} samples"
                    )
        except Exception as e:
            logger.debug(f"[TrainingTriggerDaemon] Local NPZ check failed: {e}")

        # Jan 2, 2026: Skip remote sources in local-only mode
        if self._local_only_mode:
            logger.debug(
                f"[TrainingTriggerDaemon] Local-only mode: skipping remote data sources for {config_key}"
            )
            return total_samples, best_remote_path

        # 2. Check TrainingDataManifest for S3/OWC data
        try:
            from app.coordination.training_data_manifest import (
                get_training_manifest,
                DataSource,
            )

            manifest = get_training_manifest()

            # Check S3
            s3_data = manifest.get_data_for_config(config_key, source=DataSource.S3)
            if s3_data and s3_data.sample_count > 0:
                logger.debug(
                    f"[TrainingTriggerDaemon] S3 has {s3_data.sample_count} samples for {config_key}"
                )
                if s3_data.sample_count > total_samples:
                    total_samples = s3_data.sample_count
                    best_remote_path = s3_data.path

            # Check OWC
            owc_data = manifest.get_data_for_config(config_key, source=DataSource.OWC)
            if owc_data and owc_data.sample_count > 0:
                logger.debug(
                    f"[TrainingTriggerDaemon] OWC has {owc_data.sample_count} samples for {config_key}"
                )
                if owc_data.sample_count > total_samples:
                    total_samples = owc_data.sample_count
                    best_remote_path = owc_data.path

        except ImportError:
            logger.debug("[TrainingTriggerDaemon] TrainingDataManifest not available")
        except Exception as e:
            logger.debug(f"[TrainingTriggerDaemon] Manifest check failed: {e}")

        # 3. Check ClusterManifest for games on other nodes (estimate samples)
        try:
            from app.distributed.cluster_manifest import get_cluster_manifest

            cluster_manifest = get_cluster_manifest()
            remote_games = cluster_manifest.get_game_count(config_key)
            if remote_games > 0:
                # Estimate ~50 samples per game (typical move count)
                estimated_samples = remote_games * 50
                logger.debug(
                    f"[TrainingTriggerDaemon] Cluster has ~{remote_games} games "
                    f"(~{estimated_samples} samples) for {config_key}"
                )
                # Don't override total_samples, just log for awareness
                # The games need to be synced and exported first

        except ImportError:
            logger.debug("[TrainingTriggerDaemon] ClusterManifest not available")
        except Exception as e:
            logger.debug(f"[TrainingTriggerDaemon] Cluster check failed: {e}")

        return total_samples, best_remote_path

    async def _fetch_remote_data_if_needed(
        self, config_key: str, local_count: int, min_samples_needed: int
    ) -> bool:
        """Fetch remote data if local is insufficient (January 2026).

        When local training data is below threshold, attempts to download
        from S3 or OWC to enable training.

        Args:
            config_key: Configuration identifier (e.g., "hex8_2p")
            local_count: Current local sample count
            min_samples_needed: Minimum samples required for training

        Returns:
            True if data was fetched and is now available locally
        """
        if local_count >= min_samples_needed:
            return True  # Already have enough locally

        try:
            from app.coordination.training_data_manifest import (
                get_training_manifest,
                DataSource,
            )

            manifest = get_training_manifest()

            # Find best remote source
            best_source = None
            best_count = local_count

            for source in [DataSource.S3, DataSource.OWC]:
                data = manifest.get_data_for_config(config_key, source=source)
                if data and data.sample_count > best_count:
                    best_source = data
                    best_count = data.sample_count

            if best_source and best_count >= min_samples_needed:
                logger.info(
                    f"[TrainingTriggerDaemon] Fetching {config_key} from "
                    f"{best_source.source.value} ({best_count} samples)"
                )

                # Download to local training directory
                local_path = await manifest.download_to_local(best_source)
                if local_path and local_path.exists():
                    logger.info(
                        f"[TrainingTriggerDaemon] Downloaded {config_key} to {local_path}"
                    )
                    return True
                else:
                    logger.warning(
                        f"[TrainingTriggerDaemon] Download failed for {config_key}"
                    )
                    return False

            logger.debug(
                f"[TrainingTriggerDaemon] No remote source with enough data for {config_key}"
            )
            return False

        except ImportError:
            logger.debug("[TrainingTriggerDaemon] TrainingDataManifest not available")
            return False
        except Exception as e:
            logger.warning(f"[TrainingTriggerDaemon] Remote fetch failed: {e}")
            return False

    async def _dispatch_training_to_queue(
        self,
        config_key: str,
        state: ConfigTrainingState,
        arch: ArchitectureSpec | None = None,
    ) -> bool:
        """Dispatch training job to work queue for remote execution.

        December 30, 2025: Added to support coordinator-based training dispatch.
        When the daemon runs on a coordinator node (no GPU), it dispatches
        training jobs to the centralized work queue. GPU nodes in the cluster
        will claim and execute these jobs.

        Args:
            config_key: Configuration identifier (e.g., "hex8_2p")
            state: Current training state for this config
            arch: Optional architecture specification

        Returns:
            True if job was successfully queued
        """
        try:
            from app.coordination.work_distributor import get_work_distributor

            distributor = get_work_distributor()

            # Get intensity-adjusted training parameters
            epochs, batch_size, lr_mult = get_training_params_for_intensity(
                state.training_intensity,
                default_epochs=self.config.default_epochs,
                default_batch_size=self.config.default_batch_size,
            )

            # Apply architecture-specific overrides if provided
            # Session 17.22: Use tracker-informed selection when arch is not explicitly specified
            if arch is not None:
                arch_name = arch.name
                if arch.epochs is not None:
                    epochs = arch.epochs
                if arch.batch_size is not None:
                    batch_size = arch.batch_size
            else:
                # No explicit arch - use ArchitectureTracker for performance-based selection
                arch_name = select_architecture_for_training(
                    board_type=state.board_type,
                    num_players=state.num_players,
                )

            # Compute priority based on config characteristics
            priority = 50
            # Higher priority for underrepresented configs
            if state.board_type in ("square19", "hexagonal"):
                priority = min(100, priority + 20)
            if state.num_players in (3, 4):
                priority = min(100, priority + 15)
            # Boost priority for accelerating configs (positive Elo velocity)
            if state.elo_velocity > 10.0:
                priority = min(100, priority + 10)

            # Build config for work queue submission
            from app.coordination.work_distributor import DistributedWorkConfig
            work_config = DistributedWorkConfig(
                require_gpu=True,  # Training requires GPU
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
                    f"[TrainingTriggerDaemon] Dispatched training to queue: {config_key} "
                    f"(work_id={work_id}, arch={arch_name}, epochs={epochs}, batch={batch_size})"
                )
                # Update state to track that training was dispatched
                state.training_in_progress = True
                state.training_start_time = time.time()
                # Store work_id for tracking (use _pending prefix to avoid conflicts)
                if not hasattr(state, "_pending_work_id"):
                    state._pending_work_id = None
                state._pending_work_id = work_id
                return True
            else:
                logger.warning(
                    f"[TrainingTriggerDaemon] Failed to dispatch training for {config_key}: "
                    "work queue returned None"
                )
                return False

        except ImportError as e:
            logger.warning(
                f"[TrainingTriggerDaemon] Cannot dispatch to work queue (module not available): {e}"
            )
            return False
        except Exception as e:
            logger.error(
                f"[TrainingTriggerDaemon] Failed to dispatch training for {config_key}: {e}"
            )
            return False

    async def _run_training(
        self,
        config_key: str,
        arch: ArchitectureSpec | None = None,
    ) -> bool:
        """Run training subprocess for a configuration.

        December 30, 2025: Added arch parameter for multi-architecture training.
        January 2026 (Sprint 3): Added distributed lock to prevent duplicate jobs.
        """
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

        # January 2026 (Sprint 3): Acquire distributed lock to prevent duplicate training
        # This ensures only one node trains a given config at a time across the cluster.
        # Lock timeout is 30 minutes (1800s) to cover long training runs.
        if HAS_DISTRIBUTED_LOCK and with_training_lock:
            try:
                # Check if P2P is available before attempting lock
                if is_p2p_available and await is_p2p_available():
                    arch_suffix = f":{arch.name}" if arch else ""
                    lock_name = f"{config_key}{arch_suffix}"
                    async with with_training_lock(lock_name, timeout_seconds=1800.0) as lock_result:
                        if not lock_result.acquired:
                            logger.info(
                                f"[TrainingTriggerDaemon] Training lock for {config_key} "
                                f"not acquired (held by another node), skipping"
                            )
                            return False
                        logger.debug(
                            f"[TrainingTriggerDaemon] Acquired training lock for {config_key}"
                        )
                        # Run training within lock context
                        return await self._run_training_inner(config_key, state, arch)
            except Exception as e:
                # If lock acquisition fails, log and proceed without lock
                # This ensures training still works when P2P is unavailable
                logger.warning(
                    f"[TrainingTriggerDaemon] Distributed lock error for {config_key}: {e}, "
                    "proceeding without cluster lock"
                )

        # Fallback: run without distributed lock (P2P unavailable or error)
        return await self._run_training_inner(config_key, state, arch)

    async def _run_training_inner(
        self,
        config_key: str,
        state: "TrainingConfigState",
        arch: ArchitectureSpec | None = None,
    ) -> bool:
        """Inner training logic (called by _run_training with lock held).

        January 2026 (Sprint 3): Extracted to enable lock wrapper.
        """
        # December 30, 2025: Dispatch to work queue on coordinator nodes
        # This allows the coordinator to trigger training on remote GPU nodes
        if self._dispatch_to_queue:
            return await self._dispatch_training_to_queue(config_key, state, arch)

        # Default to the supported v2 architecture family if no architecture is
        # specified; v5 remains outside the corrected minimal-loop contract.
        if arch is None:
            arch = ArchitectureSpec(
                name="v2", enabled=True, configs=["*"], priority=1.0
            )

        # Mar 6, 2026: Cross-process governor for training.
        # Prevents training + evaluation/export from overloading the node.
        _governor_slot = None
        try:
            from app.utils.coordinator_governor import get_governor, OperationType
            _governor_slot = get_governor().try_acquire(
                OperationType.TRAINING,
                description=f"training:{config_key}",
            )
            if _governor_slot is None:
                logger.info(
                    f"[TrainingTriggerDaemon] Governor denied training for "
                    f"{config_key}: system at capacity"
                )
                return False
        except Exception as _gov_err:
            logger.debug(f"[TrainingTriggerDaemon] Governor unavailable: {_gov_err}")

        async with self._training_semaphore:
            state.training_in_progress = True
            state.training_start_time = time.time()  # Phase 2: Timeout watchdog

            try:
                # Get intensity-adjusted training parameters
                epochs, batch_size, lr_mult = get_training_params_for_intensity(
                    state.training_intensity,
                    default_epochs=self.config.default_epochs,
                    default_batch_size=self.config.default_batch_size,
                )

                # January 3, 2026 (Sprint 12): Apply Elo velocity-based amplification
                # High velocity configs get more aggressive training to capitalize on momentum
                # Low velocity configs get more conservative LR to avoid overshooting
                epochs, batch_size, lr_mult = apply_velocity_amplification(
                    (epochs, batch_size, lr_mult),
                    state.elo_velocity,
                    state.elo_velocity_trend,
                )

                # December 30, 2025: Apply architecture-specific overrides
                if arch.epochs is not None:
                    epochs = arch.epochs
                if arch.batch_size is not None:
                    batch_size = arch.batch_size

                logger.info(
                    f"[TrainingTriggerDaemon] Starting training for {config_key} "
                    f"with architecture {arch.name} "
                    f"({state.npz_sample_count} samples, intensity={state.training_intensity}, "
                    f"velocity={state.elo_velocity:.2f}, trend={state.elo_velocity_trend}, "
                    f"epochs={epochs}, batch={batch_size}, lr_mult={lr_mult:.2f})"
                )

                # Build training command
                base_dir = Path(__file__).resolve().parent.parent.parent
                npz_path = state.npz_path or f"data/training/{config_key}.npz"

                # Candidate artifacts must be evaluated and promoted before they
                # become canonical models.
                model_filename = f"candidate_{config_key}_{arch.name}.pth"
                model_path = str(base_dir / "models" / model_filename)

                cmd = [
                    sys.executable,
                    "-m", "app.training.train",
                    "--board-type", state.board_type,
                    "--num-players", str(state.num_players),
                    "--data-path", npz_path,
                    "--model-version", arch.name,  # December 30, 2025: Use architecture name
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
                    **os.environ,
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
                    self._last_failure_time[config_key] = time.time()
                    return False

                duration = time.time() - start_time

                if process.returncode == 0:
                    # Success
                    state.last_training_time = time.time()
                    state.consecutive_failures = 0
                    self._last_failure_time.pop(config_key, None)

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
                    self._last_failure_time[config_key] = time.time()

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
                self._last_failure_time[config_key] = time.time()

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
                # December 30, 2025: Remove from active tasks using architecture-aware key
                # Note: The _on_training_task_done callback also cleans up,
                # but this ensures cleanup on exceptions before callback fires
                task_key = f"{config_key}:{arch.name}"
                self._active_training_tasks.pop(task_key, None)
                arch_key = (config_key, arch.name)
                self._active_architecture_training.pop(arch_key, None)
                # Mar 6, 2026: Release governor slot
                if _governor_slot is not None:
                    try:
                        from app.utils.coordinator_governor import get_governor
                        get_governor().release(_governor_slot)
                    except Exception:
                        pass

    def _on_training_task_done(
        self, task: asyncio.Task, config_key: str, arch_name: str | None = None
    ) -> None:
        """Handle training task completion.

        December 30, 2025: Added arch_name parameter for multi-architecture tracking.
        """
        try:
            exc = task.exception()
            if exc:
                logger.error(
                    f"[TrainingTriggerDaemon] Training task error for "
                    f"{config_key}/{arch_name or 'v5'}: {exc}"
                )
        except asyncio.CancelledError:
            pass
        except asyncio.InvalidStateError:
            pass

        # December 30, 2025: Clear architecture-specific tracking
        if arch_name:
            arch_key = (config_key, arch_name)
            self._active_architecture_training.pop(arch_key, None)
            # Remove task with architecture suffix
            task_key = f"{config_key}:{arch_name}"
            self._active_training_tasks.pop(task_key, None)

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
            from app.coordination.event_emission_helpers import safe_emit_event_async

            state = self._training_states.get(config_key)

            # December 29, 2025: Include model_path in event for EvaluationDaemon
            model_path = ""
            if state and hasattr(state, "_pending_model_path"):
                model_path = state._pending_model_path
                # Verify model exists before including path
                if model_path and success:
                    if not Path(model_path).exists():
                        logger.warning(
                            f"[TrainingTriggerDaemon] Model not found at {model_path}, "
                            "EvaluationDaemon may fail"
                        )

            await safe_emit_event_async(
                "TRAINING_COMPLETE" if success else "TRAINING_FAILED",
                {
                    "config": config_key,
                    "board_type": state.board_type if state else "",
                    "num_players": state.num_players if state else 0,
                    "samples_trained": state.npz_sample_count if state else 0,
                    # December 29, 2025: Critical for evaluation pipeline
                    "model_path": model_path,
                    "checkpoint_path": model_path,  # Alias for compatibility
                    "success": success,
                    "timestamp": datetime.datetime.now().isoformat(),
                },
                context="TrainingTriggerDaemon",
            )
            logger.info(
                f"[TrainingTriggerDaemon] Emitted TRAINING_{'COMPLETE' if success else 'FAILED'} "
                f"for {config_key} (model_path={model_path})"
            )

        except Exception as e:
            logger.warning(f"[TrainingTriggerDaemon] Failed to emit training event: {e}")

    async def _check_failure_recovery(self) -> None:
        """Auto-recover configs stuck in paused/reduced intensity from past failures.

        Mar 2026: Without this, consecutive_failures and training_intensity never
        auto-reset after the root cause is resolved. The circuit breaker has its own
        TTL decay (1h), but the daemon's consecutive_failures counter is independent.
        This creates a deadlock: training is "paused" so it can never succeed, and
        success is the only way to clear consecutive_failures.

        Recovery logic: If no new failures have occurred within the recovery cooldown
        period (30 min), reset consecutive_failures and restore training_intensity
        to "normal", allowing training to retry.
        """
        now = time.time()
        for config_key, state in self._training_states.items():
            if state.consecutive_failures == 0:
                continue
            if state.training_in_progress:
                continue

            last_fail = self._last_failure_time.get(config_key, 0.0)
            if last_fail <= 0:
                # No tracked failure time — use a conservative estimate
                # (treat it as recent to avoid premature recovery)
                continue

            time_since_failure = now - last_fail
            if time_since_failure < self._failure_recovery_cooldown:
                continue

            old_failures = state.consecutive_failures
            old_intensity = state.training_intensity
            state.consecutive_failures = 0

            # Only restore intensity if it was degraded by failures
            if old_intensity in ("paused", "reduced"):
                state.training_intensity = "normal"

            logger.info(
                f"[TrainingTriggerDaemon] Auto-recovered {config_key}: "
                f"consecutive_failures {old_failures} -> 0, "
                f"intensity {old_intensity} -> {state.training_intensity} "
                f"(no failures for {time_since_failure / 60:.0f}min)"
            )

            # Clear the tracked failure time
            self._last_failure_time.pop(config_key, None)

    async def _check_training_timeouts(self) -> None:
        """Check for and kill training jobs that exceed the timeout.

        December 29, 2025 (Phase 2): Training timeout watchdog for 48h autonomous operation.
        This catches hung training processes even if the daemon restarts.
        """
        timeout_seconds = self.config.training_timeout_hours * 3600
        now = time.time()

        for config_key, state in self._training_states.items():
            if not state.training_in_progress:
                continue

            if state.training_start_time <= 0:
                continue  # No start time recorded (shouldn't happen)

            elapsed = now - state.training_start_time
            if elapsed < timeout_seconds:
                continue

            # Training has exceeded timeout
            elapsed_hours = elapsed / 3600
            self._timeout_stats["timeouts_detected"] += 1
            self._timeout_stats["last_timeout_time"] = now
            logger.warning(
                f"[TrainingTriggerDaemon] Training timeout for {config_key}: "
                f"running for {elapsed_hours:.1f}h (limit: {self.config.training_timeout_hours}h)"
            )

            # Kill the training process if we have a PID
            # January 2, 2026: Use graceful shutdown - SIGTERM first to allow checkpoint save,
            # then SIGKILL after grace period if still running
            if state.training_pid is not None:
                await self._graceful_kill_process(
                    state.training_pid,
                    config_key,
                    grace_seconds=self.config.graceful_kill_timeout_seconds,
                )

            # Reset state
            state.training_in_progress = False
            state.training_pid = None
            state.training_start_time = 0.0
            state.consecutive_failures += 1
            self._last_failure_time[config_key] = time.time()

            # Cancel the asyncio task if it exists
            if config_key in self._active_training_tasks:
                task = self._active_training_tasks.pop(config_key)
                if not task.done():
                    task.cancel()

            # Emit training failed event
            await self._emit_training_failed(config_key, "timeout")

    async def _graceful_kill_process(
        self, pid: int, config_key: str, grace_seconds: float = 30.0
    ) -> None:
        """Gracefully kill a training process - SIGTERM first, then SIGKILL.

        January 2, 2026: Added to prevent model checkpoint corruption during timeout.
        Sends SIGTERM first to allow the training process to save checkpoints,
        waits up to grace_seconds, then sends SIGKILL if still running.

        Args:
            pid: Process ID to kill
            config_key: Config key for logging
            grace_seconds: Time to wait between SIGTERM and SIGKILL
        """
        try:
            # Jan 3, 2026: Emit TRAINING_TIMEOUT_REACHED before killing to allow
            # other systems (curriculum, feedback loop) to react
            from app.coordination.event_emission_helpers import safe_emit_event

            safe_emit_event(
                "TRAINING_TIMEOUT_REACHED",
                {
                    "config_key": config_key,
                    "pid": pid,
                    "timeout_hours": self.config.training_timeout_hours,
                    "grace_seconds": grace_seconds,
                    "timestamp": time.time(),
                },
                context="TrainingTriggerDaemon",
                log_after=f"Emitted TRAINING_TIMEOUT_REACHED for {config_key}",
            )

            # First, send SIGTERM for graceful shutdown
            os.kill(pid, signal.SIGTERM)
            logger.info(
                f"[TrainingTriggerDaemon] Sent SIGTERM to training process "
                f"PID {pid} for {config_key}, waiting {grace_seconds}s for graceful exit"
            )

            # Wait for process to exit gracefully
            start_wait = time.time()
            while time.time() - start_wait < grace_seconds:
                try:
                    # Check if process still exists (os.kill with signal 0 checks existence)
                    os.kill(pid, 0)
                    await asyncio.sleep(1.0)  # Check every second
                except ProcessLookupError:
                    # Process has exited gracefully
                    logger.info(
                        f"[TrainingTriggerDaemon] Training process PID {pid} "
                        f"exited gracefully after SIGTERM for {config_key}"
                    )
                    self._timeout_stats["processes_killed"] += 1
                    return

            # Process still running after grace period - send SIGKILL
            try:
                os.kill(pid, signal.SIGKILL)
                self._timeout_stats["processes_killed"] += 1
                logger.warning(
                    f"[TrainingTriggerDaemon] Sent SIGKILL to training process "
                    f"PID {pid} for {config_key} (did not exit after {grace_seconds}s SIGTERM)"
                )
            except ProcessLookupError:
                # Process exited between our check and SIGKILL - that's fine
                logger.info(
                    f"[TrainingTriggerDaemon] Training process PID {pid} "
                    f"exited just before SIGKILL for {config_key}"
                )

        except ProcessLookupError:
            logger.debug(
                f"[TrainingTriggerDaemon] Process {pid} already dead for {config_key}"
            )
        except PermissionError:
            logger.error(
                f"[TrainingTriggerDaemon] Permission denied killing PID {pid} for {config_key}"
            )
        except OSError as e:
            logger.error(
                f"[TrainingTriggerDaemon] OS error killing PID {pid} for {config_key}: {e}"
            )

    async def _emit_training_failed(self, config_key: str, reason: str) -> None:
        """Emit TRAINING_FAILED event for timed-out or errored training."""
        try:
            from app.distributed.data_events import DataEventType

            bus = self._get_event_bus()
            if bus:
                bus.publish_sync(
                    DataEventType.TRAINING_FAILED.value,
                    {
                        "config_key": config_key,
                        "reason": reason,
                        "timestamp": time.time(),
                        "source": "TrainingTriggerDaemon",
                    },
                )
                logger.info(
                    f"[TrainingTriggerDaemon] Emitted TRAINING_FAILED for {config_key}: {reason}"
                )
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[TrainingTriggerDaemon] Failed to emit TRAINING_FAILED: {e}")

    async def _log_training_diagnostic_summary(self) -> None:
        """Log periodic diagnostic summary of training trigger status.

        January 10, 2026: Added to help diagnose why training is or isn't triggering.
        Logs every 5 minutes (not every cycle) to avoid log spam.
        Shows:
        - NPZ files found and sample counts
        - Quality scores per config
        - Why training was NOT triggered (which condition failed)
        - Active training jobs

        This is critical for debugging the self-improvement loop.
        """
        # Rate-limit to every 5 minutes to avoid log spam
        now = time.time()
        if not hasattr(self, "_last_diagnostic_log"):
            self._last_diagnostic_log = 0.0
        if now - self._last_diagnostic_log < 300.0:  # 5 minutes
            return
        self._last_diagnostic_log = now

        if not self._training_states:
            logger.info(
                "[TrainingTriggerDaemon] DIAGNOSTIC: No training states tracked yet. "
                "Waiting for NPZ_EXPORT_COMPLETE events or scan to discover configs."
            )
            return

        # Build diagnostic summary
        lines = ["[TrainingTriggerDaemon] DIAGNOSTIC SUMMARY:"]
        lines.append(f"  Tracked configs: {len(self._training_states)}")
        lines.append(f"  Evaluation backpressure: {'ACTIVE (paused)' if self._evaluation_backpressure else 'OK'}")
        lines.append(f"  Local-only mode: {'YES' if self._local_only_mode else 'NO'}")

        active_training = []
        blocked_configs = []
        ready_configs = []

        for config_key, state in sorted(self._training_states.items()):
            # Check if actively training
            if state.training_in_progress:
                active_training.append(
                    f"    - {config_key}: IN PROGRESS (started {self._format_age(state.training_start_time)})"
                )
                continue

            # Check why training is blocked
            can_train, reason = await self._check_training_conditions(config_key)

            if can_train:
                ready_configs.append(
                    f"    - {config_key}: READY ({state.npz_sample_count:,} samples, "
                    f"Elo={state.last_elo:.0f}, intensity={state.training_intensity})"
                )
            else:
                blocked_configs.append(
                    f"    - {config_key}: BLOCKED - {reason} "
                    f"(samples={state.npz_sample_count:,}, Elo={state.last_elo:.0f})"
                )

        # Log summary sections
        if active_training:
            lines.append(f"  Active training ({len(active_training)}):")
            lines.extend(active_training)

        if ready_configs:
            lines.append(f"  Ready to train ({len(ready_configs)}):")
            lines.extend(ready_configs)

        if blocked_configs:
            lines.append(f"  Blocked ({len(blocked_configs)}):")
            lines.extend(blocked_configs)

        # Log all at once to keep summary together
        logger.info("\n".join(lines))

        # If everything is blocked, log a warning with suggestions
        if blocked_configs and not active_training and not ready_configs:
            # Count common blockers
            blockers = {}
            for line in blocked_configs:
                if "cooldown" in line.lower():
                    blockers["cooldown"] = blockers.get("cooldown", 0) + 1
                elif "insufficient samples" in line.lower():
                    blockers["insufficient_samples"] = blockers.get("insufficient_samples", 0) + 1
                elif "quality" in line.lower():
                    blockers["quality"] = blockers.get("quality", 0) + 1
                elif "paused" in line.lower():
                    blockers["paused"] = blockers.get("paused", 0) + 1

            if blockers:
                top_blocker = max(blockers.items(), key=lambda x: x[1])
                logger.warning(
                    f"[TrainingTriggerDaemon] All {len(blocked_configs)} configs are blocked! "
                    f"Top blocker: {top_blocker[0]} ({top_blocker[1]} configs). "
                    f"Check: 1) NPZ export daemon running? 2) Quality scores? 3) Cooldown settings?"
                )

    def _format_age(self, timestamp: float) -> str:
        """Format a timestamp as human-readable age string."""
        if timestamp <= 0:
            return "unknown"
        age_seconds = time.time() - timestamp
        if age_seconds < 60:
            return f"{age_seconds:.0f}s ago"
        elif age_seconds < 3600:
            return f"{age_seconds/60:.0f}m ago"
        else:
            return f"{age_seconds/3600:.1f}h ago"

    async def _sync_elo_from_unified_db(self) -> None:
        """Periodically sync Elo ratings from unified_elo.db to training trigger state.

        Feb 2026: Fixes stale last_elo values that caused incorrect simulation budgets.
        The training_trigger_state only updates last_elo when EVALUATION_COMPLETED events
        fire. If evaluations don't run for a config, last_elo stays at default 1500,
        causing the budget calculator to use bootstrap-tier budgets even for strong models.
        """
        now = time.time()
        if now - self._last_elo_db_sync < self._elo_db_sync_interval:
            return

        self._last_elo_db_sync = now

        def _do_sync() -> int:
            """Blocking sync (runs in thread)."""
            import sqlite3

            db_path = Path("data/unified_elo.db")
            if not db_path.exists():
                return 0

            updated = 0
            try:
                conn = sqlite3.connect(str(db_path))
                try:
                    conn.row_factory = sqlite3.Row
                    # Filter out heuristic/random/mcts engines to get actual NN Elo.
                    # Without this filter, heuristic engines inflate reported Elo
                    # (e.g. square8_2p heuristic=1910 vs NN=1695), causing the
                    # system to incorrectly believe training is complete.
                    rows = conn.execute(
                        "SELECT board_type || '_' || num_players || 'p' as config_key, "
                        "MAX(rating) as best_rating "
                        "FROM elo_ratings "
                        "WHERE participant_id NOT LIKE '%heuristic%' "
                        "AND participant_id NOT LIKE '%random%' "
                        "AND participant_id NOT LIKE '%mcts_medium%' "
                        "AND participant_id NOT LIKE 'none:%' "
                        "GROUP BY board_type, num_players"
                    ).fetchall()
                finally:
                    conn.close()

                for row in rows:
                    config_key = row["config_key"]
                    best_elo = row["best_rating"]
                    if config_key in self._training_states:
                        state = self._training_states[config_key]
                        if best_elo > state.last_elo + 5.0:
                            state.last_elo = best_elo
                            updated += 1
            except (sqlite3.Error, OSError) as e:
                logger.debug(f"[TrainingTriggerDaemon] Elo DB sync failed: {e}")

            return updated

        try:
            updated = await asyncio.to_thread(_do_sync)
            if updated > 0:
                logger.info(
                    f"[TrainingTriggerDaemon] Synced Elo from unified_elo.db: "
                    f"{updated} configs updated"
                )
        except Exception as e:
            logger.debug(f"[TrainingTriggerDaemon] Elo sync error: {e}")

    async def _check_backpressure_recovery_timeout(self) -> None:
        """Check if evaluation backpressure has exceeded max duration and auto-release.

        January 2, 2026: Added to prevent indefinite training pause if
        EVALUATION_BACKPRESSURE_RELEASED event is lost or never received.
        """
        if not self._evaluation_backpressure:
            return

        last_backpressure = self._backpressure_stats.get("last_backpressure_time", 0)
        if last_backpressure <= 0:
            return

        now = time.time()
        duration = now - last_backpressure

        if duration >= self.config.backpressure_max_duration_seconds:
            self._evaluation_backpressure = False
            self._backpressure_stats["resumes_after_backpressure"] += 1
            self._backpressure_stats["auto_recovery_count"] = (
                self._backpressure_stats.get("auto_recovery_count", 0) + 1
            )

            duration_minutes = duration / 60
            max_minutes = self.config.backpressure_max_duration_seconds / 60
            logger.warning(
                f"[TrainingTriggerDaemon] Auto-released evaluation backpressure after "
                f"{duration_minutes:.1f}m (max: {max_minutes:.0f}m). "
                f"Training RESUMED - possible lost BACKPRESSURE_RELEASED event."
            )

            # Emit an event for visibility
            try:
                from app.distributed.data_events import DataEventType

                bus = self._get_event_bus()
                if bus:
                    bus.publish_sync(
                        "training_backpressure_auto_released",
                        {
                            "duration_seconds": duration,
                            "max_duration_seconds": self.config.backpressure_max_duration_seconds,
                            "auto_recovery_count": self._backpressure_stats["auto_recovery_count"],
                            "timestamp": now,
                            "source": "TrainingTriggerDaemon",
                        },
                    )
            except Exception:
                pass  # Event emission is optional

    async def _scan_for_training_opportunities(self) -> None:
        """Scan for configs that may need training."""
        try:
            # Check existing states
            for config_key in list(self._training_states.keys()):
                await self._maybe_trigger_training(config_key)

            # Also scan for NPZ files that haven't been tracked
            # January 3, 2026: Skip files already known via event-driven cache
            training_dir = Path(__file__).resolve().parent.parent.parent / "data" / "training"
            if training_dir.exists():
                for npz_path in training_dir.glob("*.npz"):
                    # Parse config from filename using robust regex
                    board_type, num_players = self._parse_config_from_filename(npz_path.stem)
                    if board_type is None or num_players is None:
                        continue

                    config_key = make_config_key(board_type, num_players)

                    # January 3, 2026: Skip if already in cache with same or newer mtime
                    # This avoids redundant disk I/O when events already informed us
                    if config_key in self._npz_cache:
                        cached_mtime, _cached_samples, cached_path = self._npz_cache[config_key]
                        current_mtime = npz_path.stat().st_mtime
                        if cached_path == str(npz_path) and current_mtime <= cached_mtime:
                            # File hasn't changed since last event, skip disk read
                            continue

                    if config_key not in self._training_states:
                        state = self._get_or_create_state(config_key, board_type, num_players)
                        state.npz_path = str(npz_path)
                        metadata = self._get_npz_metadata(
                            config_key,
                            npz_path,
                            validate=True,
                        )
                        if metadata is None:
                            continue

                        state.last_npz_update, state.npz_sample_count, state.npz_path = metadata

                        await self._maybe_trigger_training(config_key)

        except Exception as e:
            logger.error(f"[TrainingTriggerDaemon] Error scanning for opportunities: {e}")
