"""Training Stale Data Fallback Controller.

December 2025: Part of 48-hour autonomous operation plan.
When sync failures block training indefinitely, allow training to proceed
with stale data after N failures or timeout.

Jan 21, 2026: Phase 5 - Added tiered warnings before fallback.
- Tier 0: No issues, data quality is good
- Tier 1 (NOTICE): 15 min OR 2 failures - log info
- Tier 2 (WARNING): 30 min OR 4 failures - emit warning event
- Tier 3 (CRITICAL): 45 min OR 6 failures - allow fallback
Rollback: RINGRIFT_STALE_FALLBACK_LEGACY=true

Problem: Sync failures can block training indefinitely, freezing progress.
Solution: After configurable failures or timeout, allow training with
older data while continuing to attempt sync in background.

Usage:
    from app.coordination.stale_fallback import (
        TrainingFallbackController,
        get_training_fallback_controller,
        should_allow_stale_training,
    )

    # Check data quality tier before sync
    controller = get_training_fallback_controller()
    tier, message = controller.check_data_quality_tier("hex8_2p")
    if tier >= 2:
        logger.warning(f"Sync degraded: {message}")

    # Check if stale training should be allowed
    allowed, reason = controller.should_allow_training(
        config_key="hex8_2p",
        data_age_hours=2.5,
        sync_failures=3,
        elapsed_sync_time=1800.0,
    )
    if allowed:
        logger.warning(f"Proceeding with stale data: {reason}")
        # Continue training...
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.config.coordination_defaults import StaleFallbackDefaults

logger = logging.getLogger(__name__)

__all__ = [
    "TrainingFallbackController",
    "FallbackDecision",
    "FallbackState",
    "get_training_fallback_controller",
    "reset_training_fallback_controller",
    "should_allow_stale_training",
    "check_data_quality_tier",  # Jan 21, 2026: Phase 5
]


@dataclass
class FallbackDecision:
    """Result of a stale fallback decision.

    Attributes:
        allowed: Whether training should proceed with stale data
        reason: Human-readable explanation of the decision
        is_fallback: True if this is a fallback (not normal fresh data)
        data_age_hours: Age of the training data
        sync_failures: Number of sync failures that led to this decision
        elapsed_time: Time spent trying to sync
    """
    allowed: bool
    reason: str
    is_fallback: bool = False
    data_age_hours: float = 0.0
    sync_failures: int = 0
    elapsed_time: float = 0.0


@dataclass
class FallbackState:
    """Per-config tracking state for fallback controller.

    Attributes:
        config_key: Config identifier (e.g., "hex8_2p")
        sync_failures: Cumulative sync failures for this config
        first_failure_time: Timestamp of first failure in current sequence
        last_fallback_time: Timestamp of last fallback allowed
        fallback_count: Number of times fallback was used
    """
    config_key: str
    sync_failures: int = 0
    first_failure_time: float = 0.0
    last_fallback_time: float = 0.0
    fallback_count: int = 0

    def reset(self) -> None:
        """Reset failure tracking (called after successful sync)."""
        self.sync_failures = 0
        self.first_failure_time = 0.0

    def record_failure(self) -> None:
        """Record a sync failure."""
        self.sync_failures += 1
        if self.first_failure_time == 0.0:
            self.first_failure_time = time.time()

    def record_fallback(self) -> None:
        """Record that fallback was used."""
        self.last_fallback_time = time.time()
        self.fallback_count += 1


class TrainingFallbackController:
    """Controls when training can proceed with stale data.

    This controller tracks sync failures per config and decides when
    to allow training to proceed with older data rather than waiting
    indefinitely for sync to complete.

    Fallback is allowed when ANY of these conditions are met:
    1. Sync has failed >= MAX_SYNC_FAILURES times
    2. Time spent trying to sync >= MAX_SYNC_DURATION

    Fallback is BLOCKED when:
    1. Stale fallback is disabled (ENABLE_STALE_FALLBACK=false)
    2. Data age exceeds ABSOLUTE_MAX_DATA_AGE (hard limit)
    3. Not enough games available (< MIN_GAMES_FOR_FALLBACK)
    4. Within FALLBACK_COOLDOWN of last fallback for this config

    December 2025: Part of 48-hour autonomous operation plan.

    Jan 21, 2026: Phase 5 - Added tiered warnings before fallback.
    - Tier 0: No issues
    - Tier 1 (NOTICE): 15 min OR 2 failures - log info
    - Tier 2 (WARNING): 30 min OR 4 failures - emit warning event
    - Tier 3 (CRITICAL): 45 min OR 6 failures - allow fallback
    Rollback: RINGRIFT_STALE_FALLBACK_LEGACY=true
    """

    def __init__(self) -> None:
        """Initialize the fallback controller."""
        self._states: dict[str, FallbackState] = {}
        self._start_time = time.time()
        # Jan 21, 2026: Track emitted tier events to avoid spam
        self._tier_events_emitted: dict[str, int] = {}  # config_key -> highest tier emitted

    def _get_state(self, config_key: str) -> FallbackState:
        """Get or create state for a config."""
        if config_key not in self._states:
            self._states[config_key] = FallbackState(config_key=config_key)
        return self._states[config_key]

    def record_sync_failure(self, config_key: str) -> None:
        """Record a sync failure for a config.

        Args:
            config_key: Config identifier (e.g., "hex8_2p")
        """
        state = self._get_state(config_key)
        state.record_failure()
        logger.debug(
            f"[StaleFallback] Sync failure #{state.sync_failures} for {config_key}"
        )

    def record_sync_success(self, config_key: str) -> None:
        """Record successful sync (resets failure tracking).

        Args:
            config_key: Config identifier (e.g., "hex8_2p")
        """
        state = self._get_state(config_key)
        if state.sync_failures > 0:
            logger.info(
                f"[StaleFallback] Sync succeeded for {config_key} after "
                f"{state.sync_failures} failures"
            )
        state.reset()
        # Reset tier event tracking on success
        if config_key in self._tier_events_emitted:
            del self._tier_events_emitted[config_key]

    def check_data_quality_tier(
        self,
        config_key: str,
        sync_failures: int | None = None,
        elapsed_sync_time: float | None = None,
    ) -> tuple[int, str]:
        """Check the current data quality tier based on sync issues.

        Jan 21, 2026: Phase 5 - Gradual warnings before fallback.

        Args:
            config_key: Config identifier (e.g., "hex8_2p")
            sync_failures: Override for sync failures (uses tracked count if None)
            elapsed_sync_time: Time spent trying to sync (seconds)

        Returns:
            Tuple of (tier 0-3, message describing the tier)
            - Tier 0: No issues, data quality is good
            - Tier 1: NOTICE - early warning, sync has some issues
            - Tier 2: WARNING - escalated warning, sync struggling
            - Tier 3: CRITICAL - fallback threshold reached
        """
        # Check legacy mode
        if StaleFallbackDefaults.LEGACY_MODE:
            # In legacy mode, only report tier 0 or 3
            state = self._get_state(config_key)
            failures = sync_failures if sync_failures is not None else state.sync_failures
            elapsed = elapsed_sync_time if elapsed_sync_time is not None else state.elapsed_time
            if failures >= 5 or elapsed >= 2700:  # Legacy thresholds
                return 3, f"Sync issues: {failures} failures, {elapsed:.0f}s elapsed (legacy mode)"
            return 0, "Data quality: OK"

        # Get current state
        state = self._get_state(config_key)
        failures = sync_failures if sync_failures is not None else state.sync_failures
        elapsed = elapsed_sync_time if elapsed_sync_time is not None else state.elapsed_time

        # Determine tier based on thresholds
        tier = 0
        message = "Data quality: OK"

        # Check Tier 3 (CRITICAL) first
        if failures >= StaleFallbackDefaults.TIER_3_FAILURES or elapsed >= StaleFallbackDefaults.TIER_3_DURATION:
            tier = 3
            message = f"CRITICAL: {failures} failures, {elapsed:.0f}s elapsed - fallback threshold reached"
        # Check Tier 2 (WARNING)
        elif failures >= StaleFallbackDefaults.TIER_2_FAILURES or elapsed >= StaleFallbackDefaults.TIER_2_DURATION:
            tier = 2
            message = f"WARNING: {failures} failures, {elapsed:.0f}s elapsed - sync struggling"
        # Check Tier 1 (NOTICE)
        elif failures >= StaleFallbackDefaults.TIER_1_FAILURES or elapsed >= StaleFallbackDefaults.TIER_1_DURATION:
            tier = 1
            message = f"NOTICE: {failures} failures, {elapsed:.0f}s elapsed - early warning"

        # Emit tier event if this is a new/higher tier
        self._emit_tier_event(config_key, tier, failures, elapsed)

        return tier, message

    def _emit_tier_event(
        self,
        config_key: str,
        tier: int,
        failures: int,
        elapsed: float,
    ) -> None:
        """Emit event for tier changes.

        Jan 21, 2026: Phase 5 - Only emits when tier increases to avoid spam.
        """
        if tier == 0:
            return

        # Check if we've already emitted for this tier or higher
        previous_tier = self._tier_events_emitted.get(config_key, 0)
        if tier <= previous_tier:
            return

        # Record that we've emitted for this tier
        self._tier_events_emitted[config_key] = tier

        # Log based on tier
        tier_names = {1: "NOTICE", 2: "WARNING", 3: "CRITICAL"}
        tier_name = tier_names.get(tier, f"TIER_{tier}")

        if tier == 1:
            logger.info(f"[StaleFallback] {tier_name} for {config_key}: {failures} failures, {elapsed:.0f}s elapsed")
        elif tier == 2:
            logger.warning(f"[StaleFallback] {tier_name} for {config_key}: {failures} failures, {elapsed:.0f}s elapsed")
        else:
            logger.warning(f"[StaleFallback] {tier_name} for {config_key}: {failures} failures, {elapsed:.0f}s elapsed - fallback imminent")

        # Emit event for tier 2 and 3
        if tier >= 2 and StaleFallbackDefaults.EMIT_FALLBACK_EVENTS:
            try:
                from app.coordination.event_router import emit_event
                emit_event(
                    f"STALE_DATA_QUALITY_TIER_{tier}",
                    {
                        "config_key": config_key,
                        "tier": tier,
                        "tier_name": tier_name,
                        "sync_failures": failures,
                        "elapsed_seconds": elapsed,
                    },
                )
            except (ImportError, AttributeError, RuntimeError) as e:
                logger.debug(f"[StaleFallback] Could not emit tier event: {e}")

    def should_allow_training(
        self,
        config_key: str,
        data_age_hours: float,
        sync_failures: int | None = None,
        elapsed_sync_time: float | None = None,
        games_available: int = 0,
    ) -> FallbackDecision:
        """Determine if training should proceed with potentially stale data.

        Args:
            config_key: Config identifier (e.g., "hex8_2p")
            data_age_hours: Age of the training data in hours
            sync_failures: Override for sync failures (uses tracked count if None)
            elapsed_sync_time: Time spent trying to sync (seconds)
            games_available: Number of games in the training data

        Returns:
            FallbackDecision with allowed status and reason
        """
        state = self._get_state(config_key)
        now = time.time()

        # Use tracked failures if not overridden
        if sync_failures is not None:
            # Update tracked state with provided value
            if sync_failures > state.sync_failures:
                state.sync_failures = sync_failures
        failures = state.sync_failures

        # Calculate elapsed time from first failure if not provided
        if elapsed_sync_time is None and state.first_failure_time > 0:
            elapsed_sync_time = now - state.first_failure_time
        elapsed = elapsed_sync_time or 0.0

        # Check if fallback is disabled
        if not StaleFallbackDefaults.ENABLE_STALE_FALLBACK:
            return FallbackDecision(
                allowed=False,
                reason="Stale fallback disabled (RINGRIFT_ENABLE_STALE_FALLBACK=false)",
                data_age_hours=data_age_hours,
                sync_failures=failures,
                elapsed_time=elapsed,
            )

        # Check absolute maximum data age (hard limit)
        if data_age_hours > StaleFallbackDefaults.ABSOLUTE_MAX_DATA_AGE:
            return FallbackDecision(
                allowed=False,
                reason=(
                    f"Data too old ({data_age_hours:.1f}h > "
                    f"{StaleFallbackDefaults.ABSOLUTE_MAX_DATA_AGE}h limit)"
                ),
                data_age_hours=data_age_hours,
                sync_failures=failures,
                elapsed_time=elapsed,
            )

        # Check minimum games requirement
        if games_available > 0 and games_available < StaleFallbackDefaults.MIN_GAMES_FOR_FALLBACK:
            return FallbackDecision(
                allowed=False,
                reason=(
                    f"Insufficient games ({games_available} < "
                    f"{StaleFallbackDefaults.MIN_GAMES_FOR_FALLBACK} required)"
                ),
                data_age_hours=data_age_hours,
                sync_failures=failures,
                elapsed_time=elapsed,
            )

        # Check cooldown
        if state.last_fallback_time > 0:
            cooldown_remaining = (
                state.last_fallback_time
                + StaleFallbackDefaults.FALLBACK_COOLDOWN
                - now
            )
            if cooldown_remaining > 0:
                return FallbackDecision(
                    allowed=False,
                    reason=(
                        f"Fallback cooldown active ({cooldown_remaining:.0f}s remaining)"
                    ),
                    data_age_hours=data_age_hours,
                    sync_failures=failures,
                    elapsed_time=elapsed,
                )

        # Check if fallback should be allowed
        should_fallback = False
        reason = ""

        # Condition 1: Too many sync failures
        if failures >= StaleFallbackDefaults.MAX_SYNC_FAILURES:
            should_fallback = True
            reason = (
                f"Sync failed {failures} times "
                f"(threshold: {StaleFallbackDefaults.MAX_SYNC_FAILURES})"
            )

        # Condition 2: Sync taking too long
        elif elapsed >= StaleFallbackDefaults.MAX_SYNC_DURATION:
            should_fallback = True
            reason = (
                f"Sync duration {elapsed:.0f}s exceeded "
                f"{StaleFallbackDefaults.MAX_SYNC_DURATION:.0f}s limit"
            )

        if should_fallback:
            state.record_fallback()

            # Jan 3, 2026: Enhanced logging for better observability
            # Calculate how long this config has been experiencing sync issues
            time_in_fallback = 0.0
            if state.first_failure_time > 0:
                time_in_fallback = now - state.first_failure_time

            logger.warning(
                f"[StaleFallback] Allowing stale training for {config_key}: {reason} "
                f"(data_age={data_age_hours:.1f}h, games={games_available}, "
                f"fallback_count={state.fallback_count}, "
                f"sync_issue_duration={time_in_fallback:.0f}s)"
            )

            # Additional info-level context for debugging
            if state.fallback_count > 1:
                logger.info(
                    f"[StaleFallback] Config {config_key} has used fallback "
                    f"{state.fallback_count} times. Consider investigating sync issues."
                )

            # Emit event if configured
            if StaleFallbackDefaults.EMIT_FALLBACK_EVENTS:
                self._emit_fallback_event(
                    config_key, data_age_hours, failures, elapsed, reason
                )

            return FallbackDecision(
                allowed=True,
                reason=reason,
                is_fallback=True,
                data_age_hours=data_age_hours,
                sync_failures=failures,
                elapsed_time=elapsed,
            )

        # Not yet time to fallback
        return FallbackDecision(
            allowed=False,
            reason=(
                f"Sync still in progress ({failures} failures, {elapsed:.0f}s elapsed)"
            ),
            data_age_hours=data_age_hours,
            sync_failures=failures,
            elapsed_time=elapsed,
        )

    def _emit_fallback_event(
        self,
        config_key: str,
        data_age_hours: float,
        sync_failures: int,
        elapsed_time: float,
        reason: str,
    ) -> None:
        """Emit event when fallback is triggered."""
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            if router:
                router.publish_sync(
                    event_type="STALE_TRAINING_FALLBACK",
                    payload={
                        "config_key": config_key,
                        "data_age_hours": data_age_hours,
                        "sync_failures": sync_failures,
                        "elapsed_time": elapsed_time,
                        "reason": reason,
                        "timestamp": time.time(),
                    },
                    source="TrainingFallbackController",
                )
        except Exception as e:
            logger.debug(f"[StaleFallback] Could not emit fallback event: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get controller status."""
        return {
            "enabled": StaleFallbackDefaults.ENABLE_STALE_FALLBACK,
            "config": {
                "max_sync_failures": StaleFallbackDefaults.MAX_SYNC_FAILURES,
                "max_sync_duration": StaleFallbackDefaults.MAX_SYNC_DURATION,
                "absolute_max_data_age": StaleFallbackDefaults.ABSOLUTE_MAX_DATA_AGE,
                "min_games_for_fallback": StaleFallbackDefaults.MIN_GAMES_FOR_FALLBACK,
                "fallback_cooldown": StaleFallbackDefaults.FALLBACK_COOLDOWN,
            },
            "states": {
                key: {
                    "sync_failures": state.sync_failures,
                    "first_failure_time": state.first_failure_time,
                    "last_fallback_time": state.last_fallback_time,
                    "fallback_count": state.fallback_count,
                }
                for key, state in self._states.items()
            },
            "uptime_seconds": time.time() - self._start_time,
        }

    def health_check(self) -> "HealthCheckResult":
        """Perform health check for DaemonManager integration.

        December 29, 2025: Added to enable health monitoring.

        Returns:
            HealthCheckResult indicating controller health:
            - HEALTHY: Fallback rarely used, sync working normally
            - DEGRADED: Fallback being used frequently, sync may have issues
            - UNHEALTHY: Fallback disabled when needed (shouldn't happen)
        """
        try:
            from app.coordination.contracts import HealthCheckResult, CoordinatorStatus

            # Calculate health metrics
            total_fallbacks = sum(s.fallback_count for s in self._states.values())
            recent_failures = sum(s.sync_failures for s in self._states.values())
            configs_in_fallback = sum(
                1 for s in self._states.values()
                if s.sync_failures >= StaleFallbackDefaults.MAX_SYNC_FAILURES
            )
            uptime = time.time() - self._start_time

            # Determine health status
            if not StaleFallbackDefaults.ENABLE_STALE_FALLBACK:
                # Fallback disabled - check if it would be needed
                if recent_failures > 0:
                    return HealthCheckResult(
                        healthy=False,
                        status=CoordinatorStatus.DEGRADED,
                        message=f"Stale fallback disabled but {recent_failures} sync failures detected",
                        details={
                            "fallback_enabled": False,
                            "configs_needing_fallback": configs_in_fallback,
                            "total_recent_failures": recent_failures,
                            "uptime_seconds": uptime,
                        },
                    )

            # Healthy if fallback rarely used (< 1 per 10 min average)
            fallback_rate = total_fallbacks / max(uptime / 600, 1)  # Per 10 min

            if fallback_rate < 0.1:
                return HealthCheckResult(
                    healthy=True,
                    status=CoordinatorStatus.RUNNING,
                    message="Stale fallback controller healthy",
                    details={
                        "fallback_enabled": StaleFallbackDefaults.ENABLE_STALE_FALLBACK,
                        "total_fallbacks": total_fallbacks,
                        "configs_tracked": len(self._states),
                        "uptime_seconds": uptime,
                    },
                )
            elif fallback_rate < 0.5:
                return HealthCheckResult(
                    healthy=True,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"Elevated fallback usage: {total_fallbacks} total, {configs_in_fallback} configs affected",
                    details={
                        "fallback_enabled": StaleFallbackDefaults.ENABLE_STALE_FALLBACK,
                        "total_fallbacks": total_fallbacks,
                        "fallback_rate_per_10min": fallback_rate,
                        "configs_in_fallback": configs_in_fallback,
                        "uptime_seconds": uptime,
                    },
                )
            else:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"High fallback usage indicates sync issues: {total_fallbacks} total",
                    details={
                        "fallback_enabled": StaleFallbackDefaults.ENABLE_STALE_FALLBACK,
                        "total_fallbacks": total_fallbacks,
                        "fallback_rate_per_10min": fallback_rate,
                        "configs_in_fallback": configs_in_fallback,
                        "uptime_seconds": uptime,
                    },
                )

        except ImportError:
            # Fallback if contracts not available
            return {  # type: ignore
                "healthy": True,
                "message": "Stale fallback controller running",
                "uptime_seconds": time.time() - self._start_time,
            }


# Module-level singleton
_fallback_controller: TrainingFallbackController | None = None


def get_training_fallback_controller() -> TrainingFallbackController:
    """Get the singleton TrainingFallbackController instance."""
    global _fallback_controller
    if _fallback_controller is None:
        _fallback_controller = TrainingFallbackController()
    return _fallback_controller


def reset_training_fallback_controller() -> None:
    """Reset the singleton (for testing)."""
    global _fallback_controller
    _fallback_controller = None


def should_allow_stale_training(
    config_key: str,
    data_age_hours: float,
    sync_failures: int = 0,
    elapsed_sync_time: float = 0.0,
    games_available: int = 0,
) -> tuple[bool, str]:
    """Convenience function to check if stale training should be allowed.

    Args:
        config_key: Config identifier (e.g., "hex8_2p")
        data_age_hours: Age of the training data in hours
        sync_failures: Number of sync failures
        elapsed_sync_time: Time spent trying to sync (seconds)
        games_available: Number of games in the training data

    Returns:
        Tuple of (allowed: bool, reason: str)
    """
    controller = get_training_fallback_controller()
    decision = controller.should_allow_training(
        config_key=config_key,
        data_age_hours=data_age_hours,
        sync_failures=sync_failures,
        elapsed_sync_time=elapsed_sync_time,
        games_available=games_available,
    )
    return decision.allowed, decision.reason


def check_data_quality_tier(
    config_key: str,
    sync_failures: int = 0,
    elapsed_sync_time: float = 0.0,
) -> tuple[int, str]:
    """Convenience function to check data quality tier.

    Jan 21, 2026: Phase 5 - Gradual warnings before fallback.

    Args:
        config_key: Config identifier (e.g., "hex8_2p")
        sync_failures: Number of sync failures
        elapsed_sync_time: Time spent trying to sync (seconds)

    Returns:
        Tuple of (tier: int 0-3, message: str)
        - Tier 0: No issues
        - Tier 1: NOTICE (early warning)
        - Tier 2: WARNING (sync struggling)
        - Tier 3: CRITICAL (fallback threshold)
    """
    controller = get_training_fallback_controller()
    return controller.check_data_quality_tier(
        config_key=config_key,
        sync_failures=sync_failures,
        elapsed_sync_time=elapsed_sync_time,
    )
