"""Queue health, backpressure, and circuit-breaker helpers."""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any

from app.coordination.event_emission_helpers import safe_emit_event
from app.coordination.types import BackpressureLevel

logger = logging.getLogger(__name__)


class QueuePopulationHealthMixin:
    """Extracted queue population behavior."""

    def get_current_queue_depth(self) -> int:
        """Get current work queue depth."""
        if self._work_queue is None:
            return 0
        status = self._work_queue.get_queue_status()
        pending = len(status.get("pending", []))
        running = len(status.get("running", []))
        return pending + running
    def get_pending_by_type(self) -> dict[str, int]:
        """Get pending counts grouped by work type.

        January 30, 2026: Added for per-type limit enforcement to prevent
        backlog accumulation (e.g., 2,686 pending jobs).

        Returns:
            Dict mapping work_type to pending count
        """
        if self._work_queue is None:
            return {}
        try:
            # Mar 2026: Primary path — use in-memory items dict (live state).
            # The SQLite db has stale 'pending' items from before orchestrator
            # restarts (items expire in memory but SQLite is never updated),
            # causing false "over limit" signals. The in-memory items dict is
            # the source of truth and resets cleanly on each restart.
            items_dict = getattr(self._work_queue, "_items", None)
            if items_dict is not None:
                lock = getattr(self._work_queue, "lock", None)
                counts: dict[str, int] = {}
                items_snapshot = list(items_dict.values()) if lock is None else []
                if lock is not None:
                    with lock:
                        items_snapshot = list(items_dict.values())
                for item in items_snapshot:
                    status_val = getattr(item, "status", None)
                    if status_val is not None and getattr(status_val, "value", status_val) == "pending":
                        wtype = getattr(item, "work_type", None)
                        if wtype is not None:
                            wtype_str = getattr(wtype, "value", str(wtype))
                            counts[wtype_str] = counts.get(wtype_str, 0) + 1
                return counts
            # Fallback: direct SQL with age filter to exclude post-restart stale items
            db_path = getattr(self._work_queue, "db_path", None)
            if db_path:
                with sqlite3.connect(db_path, timeout=10.0) as conn:
                    now = time.time()
                    cursor = conn.execute(
                        "SELECT work_type, COUNT(*) FROM work_items "
                        "WHERE status = 'pending' "
                        "AND created_at > (? - timeout_seconds * max_attempts) "
                        "GROUP BY work_type",
                        (now,),
                    )
                    result = {row[0]: row[1] for row in cursor.fetchall()}
                return result
            # Final fallback: use queue status dict
            status = self._work_queue.get_queue_status()
            pending_items = status.get("pending", [])
            counts = {}
            for item in pending_items:
                wtype = getattr(item, "work_type", "unknown")
                counts[wtype] = counts.get(wtype, 0) + 1
            return counts
        except Exception as e:
            logger.debug(f"[QueuePopulator] Failed to get pending by type: {e}")
            return {}
    def _get_pending_tournament_by_config(self) -> dict[str, int]:
        """Get pending tournament counts grouped by config key.

        Feb 2026: Added to prevent evaluation starvation. When the global
        tournament limit is reached, some configs may have 0 pending
        tournaments while others have 40+. This enables per-config fairness.

        Returns:
            Dict mapping config_key (e.g. "hex8_2p") to pending tournament count
        """
        if self._work_queue is None:
            return {}
        try:
            db_path = getattr(self._work_queue, "db_path", None)
            if db_path:
                with sqlite3.connect(db_path, timeout=10.0) as conn:
                    now = time.time()
                    cursor = conn.execute(
                        "SELECT json_extract(config, '$.board_type'), "
                        "json_extract(config, '$.num_players'), COUNT(*) "
                        "FROM work_items "
                        "WHERE status = 'pending' AND work_type = 'tournament' "
                        "AND created_at > (? - timeout_seconds * max_attempts) "
                        "GROUP BY json_extract(config, '$.board_type'), "
                        "json_extract(config, '$.num_players')",
                        (now,),
                    )
                    result = {}
                    for row in cursor.fetchall():
                        if row[0] and row[1]:
                            config_key = f"{row[0]}_{row[1]}p"
                            result[config_key] = row[2]
                return result
            return {}
        except Exception as e:
            logger.debug(f"[QueuePopulator] Failed to get tournament by config: {e}")
            return {}
    def _is_type_over_limit(self, work_type: str, pending_by_type: dict[str, int]) -> bool:
        """Check if a work type is over its pending limit.

        January 30, 2026: Per-type limits prevent backlog accumulation.
        """
        current = pending_by_type.get(work_type, 0)
        limit = {
            "selfplay": self.config.max_pending_selfplay,
            "training": self.config.max_pending_training,
            "tournament": self.config.max_pending_tournament,
            "hyperparam_sweep": self.config.max_pending_hyperparam_sweep,
        }.get(work_type, 100)
        return current >= limit
    def calculate_items_needed(self) -> int:
        """Calculate how many items to add to reach target depth.

        December 29, 2025: Now targets target_queue_depth instead of min_queue_depth,
        and caps at max_batch_per_cycle to prevent burst releases that cause
        queue variance spikes (was 2,170% variance, target <50%).
        """
        current = self.get_current_queue_depth()
        # Use target_queue_depth for filling, but only add if below min_queue_depth
        if current >= self.config.min_queue_depth:
            # Already above minimum, gradually fill to target
            needed = max(0, self.config.target_queue_depth - current)
        else:
            # Below minimum, fill more aggressively to min_queue_depth
            needed = max(0, self.config.target_queue_depth - current)

        # Cap at max_batch_per_cycle to prevent burst releases
        return min(needed, self.config.max_batch_per_cycle)
    def _check_backpressure(self) -> tuple[BackpressureLevel, float]:
        """Check current backpressure level."""
        try:
            from app.coordination.queue_monitor import get_queue_monitor

            monitor = get_queue_monitor()
            if monitor:
                status = monitor.get_overall_status()
                bp_level = status.get("backpressure_level", "none")
                if isinstance(bp_level, str):
                    bp_level = BackpressureLevel(bp_level)
                elif hasattr(bp_level, "value"):
                    bp_level = BackpressureLevel(bp_level.value)

                if bp_level.should_stop():
                    return bp_level, 0.0

                return bp_level, bp_level.reduction_factor()

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[QueuePopulator] Backpressure check failed: {e}")

        return BackpressureLevel.NONE, 1.0
    def _maybe_emit_backpressure_event(
        self, current_level: BackpressureLevel, status: dict[str, Any]
    ) -> None:
        """Emit events on backpressure state changes with hysteresis.

        Only emits events when crossing the MEDIUM threshold to prevent event
        spam during level oscillation. Uses hysteresis: emit BACKPRESSURE when
        transitioning from low→high, emit BACKPRESSURE_RELEASED when high→low.

        Args:
            current_level: The current backpressure level from _check_backpressure()
            status: Queue status dict with depth, utilization, etc.
        """
        # Define HIGH_LEVELS for hysteresis (MEDIUM and above trigger events)
        HIGH_LEVELS = (
            BackpressureLevel.MEDIUM,
            BackpressureLevel.HARD,
            BackpressureLevel.HIGH,
            BackpressureLevel.CRITICAL,
            BackpressureLevel.STOP,
        )

        was_activated = self._last_backpressure_level in HIGH_LEVELS
        is_activated = current_level in HIGH_LEVELS

        # Only emit on transitions
        if is_activated and not was_activated:
            # Low → High: emit backpressure activated
            from app.distributed.data_events import DataEventType

            safe_emit_event(
                event_type=DataEventType.WORK_QUEUE_BACKPRESSURE,
                payload={
                    "level": current_level.value,
                    "reduction_factor": current_level.reduction_factor(),
                    "queue_depth": status.get("queue_depth", 0),
                    "utilization": status.get("utilization", 0.0),
                    "threshold_crossed": "medium",
                    "previous_level": self._last_backpressure_level.value,
                },
                log_before=True,
                log_level=logging.WARNING,
                context="queue_populator_backpressure",
            )
            logger.warning(
                f"[QueuePopulator] Backpressure ACTIVATED: {self._last_backpressure_level.value} → {current_level.value}"
            )
        elif not is_activated and was_activated:
            # High → Low: emit backpressure released
            from app.distributed.data_events import DataEventType

            safe_emit_event(
                event_type=DataEventType.WORK_QUEUE_BACKPRESSURE_RELEASED,
                payload={
                    "level": current_level.value,
                    "previous_level": self._last_backpressure_level.value,
                    "queue_depth": status.get("queue_depth", 0),
                    "utilization": status.get("utilization", 0.0),
                },
                log_before=True,
                log_level=logging.INFO,
                context="queue_populator_backpressure_released",
            )
            logger.info(
                f"[QueuePopulator] Backpressure RELEASED: {self._last_backpressure_level.value} → {current_level.value}"
            )

        # Update state for next check
        self._last_backpressure_level = current_level
    def _maybe_emit_queue_exhausted_event(self) -> None:
        """Emit WORK_QUEUE_EXHAUSTED event when queue becomes empty.

        January 2026 - Phase 3 Task 4: Wire underutilization handler to event bus.
        This event triggers UnderutilizationRecoveryHandler to inject high-priority
        work items for underserved configurations.

        Uses hysteresis to prevent event spam:
        - Emits WORK_QUEUE_EXHAUSTED when queue transitions from non-empty to empty
        - Does not emit if queue was already empty (no repeated emissions)
        """
        current_depth = self.get_current_queue_depth()
        is_exhausted = current_depth == 0

        # Only emit on transitions from non-empty to empty
        if is_exhausted and not self._was_queue_exhausted:
            from app.distributed.data_events import DataEventType

            safe_emit_event(
                event_type=DataEventType.WORK_QUEUE_EXHAUSTED,
                payload={
                    "queue_depth": 0,
                    "min_queue_depth": self.config.min_queue_depth,
                    "target_queue_depth": self.config.target_queue_depth,
                    "cluster_health_factor": self._cluster_health_factor,
                    "dead_nodes_count": len(self._dead_nodes),
                    "timestamp": time.time(),
                },
                log_before=True,
                log_level=logging.WARNING,
                context="queue_populator_exhausted",
            )
            logger.warning(
                f"[QueuePopulator] WORK_QUEUE_EXHAUSTED emitted - "
                f"queue is empty (min_depth={self.config.min_queue_depth})"
            )

        elif not is_exhausted and self._was_queue_exhausted:
            # Queue recovered from empty state - log for visibility
            logger.info(
                f"[QueuePopulator] Queue recovered from exhaustion - "
                f"depth now {current_depth}"
            )

        # Update state for next check
        self._was_queue_exhausted = is_exhausted
    def _calculate_backoff(self) -> float:
        """Calculate next backoff duration with exponential growth and jitter.

        Returns:
            Next backoff duration in seconds.
        """
        import random

        if not self.config.backoff_enabled:
            return 0.0

        if self._backoff_current_seconds == 0.0:
            self._backoff_current_seconds = self.config.backoff_initial_seconds
        else:
            self._backoff_current_seconds = min(
                self._backoff_current_seconds * self.config.backoff_multiplier,
                self.config.backoff_max_seconds,
            )

        # Apply jitter: ±jitter% randomization
        jitter_range = self._backoff_current_seconds * self.config.backoff_jitter
        jitter = random.uniform(-jitter_range, jitter_range)

        return self._backoff_current_seconds + jitter
    def _apply_backoff(self) -> None:
        """Apply exponential backoff after hitting queue hard limit."""
        backoff_duration = self._calculate_backoff()
        self._backoff_until = time.time() + backoff_duration
        self._consecutive_hard_limit_hits += 1

        logger.warning(
            f"[QueuePopulator] Backing off for {backoff_duration:.1f}s "
            f"(consecutive hard limit hits: {self._consecutive_hard_limit_hits}, "
            f"current backoff: {self._backoff_current_seconds:.1f}s)"
        )

        # Emit event for monitoring
        safe_emit_event(
            event_type="QUEUE_POPULATOR_BACKOFF",
            payload={
                "backoff_seconds": backoff_duration,
                "consecutive_hits": self._consecutive_hard_limit_hits,
                "backoff_level": self._backoff_current_seconds,
            },
            log_before=False,
            context="queue_populator_backoff",
        )
    def _reset_backoff(self) -> None:
        """Reset backoff state after successful operation."""
        if self._backoff_current_seconds > 0.0:
            logger.info(
                f"[QueuePopulator] Backoff reset after {self._consecutive_hard_limit_hits} "
                f"consecutive hard limit hits"
            )
        self._backoff_current_seconds = 0.0
        self._backoff_until = 0.0
        self._consecutive_hard_limit_hits = 0
    def _is_backing_off(self) -> bool:
        """Check if currently in backoff period."""
        if not self.config.backoff_enabled:
            return False
        return time.time() < self._backoff_until
    def _log_health_status(self, force: bool = False) -> None:
        """Log health status periodically during backpressure.

        Args:
            force: If True, log regardless of interval.
        """
        now = time.time()
        interval = self.config.health_log_interval_seconds

        if not force and (now - self._last_health_log_time) < interval:
            return

        self._last_health_log_time = now

        # Calculate backpressure duration
        bp_duration = 0.0
        if self._backpressure_start_time > 0:
            bp_duration = now - self._backpressure_start_time

        queue_depth = self.get_current_queue_depth()
        bp_level, _ = self._check_backpressure()

        # Get drain rate
        drain_rate = self._calculate_drain_rate()

        logger.info(
            f"[QueuePopulator] HEALTH: queue_depth={queue_depth}, "
            f"backpressure={bp_level.value}, bp_duration={bp_duration:.1f}s, "
            f"drain_rate={drain_rate:.2f}/min, "
            f"partition_detected={self._partition_detected}, "
            f"circuit_state={self._circuit_state.value if hasattr(self, '_circuit_state') else 'unknown'}, "
            f"backoff_until={self._backoff_until:.1f}, "
            f"consecutive_hard_hits={self._consecutive_hard_limit_hits}"
        )
    def record_completion(self, timestamp: float | None = None) -> None:
        """Record a work item completion for drain rate calculation.

        Args:
            timestamp: Unix timestamp of completion. Uses current time if None.
        """
        ts = timestamp or time.time()
        self._completion_timestamps.append(ts)

        # Prune old timestamps (keep last 5 minutes for analysis)
        cutoff = ts - 300.0
        self._completion_timestamps = [t for t in self._completion_timestamps if t > cutoff]
    def _calculate_drain_rate(self) -> float:
        """Calculate queue drain rate (completions per minute).

        Returns:
            Drain rate in completions per minute.
        """
        if not self._completion_timestamps:
            return 0.0

        now = time.time()
        window = self.config.partition_drain_window_seconds
        cutoff = now - window

        completions_in_window = sum(1 for t in self._completion_timestamps if t > cutoff)
        # Convert to per-minute rate
        return (completions_in_window / window) * 60.0
    def _check_partition(self) -> bool:
        """Check for cluster partition based on queue drain rate.

        Returns:
            True if partition is detected.
        """
        if not self.config.partition_detection_enabled:
            return False

        now = time.time()
        window = self.config.partition_drain_window_seconds
        cutoff = now - window

        completions_in_window = sum(1 for t in self._completion_timestamps if t > cutoff)

        if completions_in_window < self.config.partition_min_completions:
            self._consecutive_zero_drain_windows += 1
        else:
            self._consecutive_zero_drain_windows = 0
            if self._partition_detected:
                # Partition recovered
                partition_duration = now - self._partition_detected_at
                logger.info(
                    f"[QueuePopulator] Partition RECOVERED after {partition_duration:.1f}s "
                    f"(drain rate: {self._calculate_drain_rate():.2f}/min)"
                )
                safe_emit_event(
                    event_type="CLUSTER_PARTITION_RECOVERED",
                    payload={
                        "partition_duration_seconds": partition_duration,
                        "drain_rate": self._calculate_drain_rate(),
                    },
                    log_before=False,
                    context="queue_populator_partition",
                )
            self._partition_detected = False
            self._partition_detected_at = 0.0

        if self._consecutive_zero_drain_windows >= self.config.partition_alert_threshold:
            if not self._partition_detected:
                # New partition detected
                self._partition_detected = True
                self._partition_detected_at = now
                logger.error(
                    f"[QueuePopulator] CLUSTER PARTITION DETECTED: "
                    f"No queue drain for {self._consecutive_zero_drain_windows} consecutive windows "
                    f"({window * self._consecutive_zero_drain_windows:.0f}s). "
                    f"Workers may be unreachable."
                )
                safe_emit_event(
                    event_type="CLUSTER_PARTITION_DETECTED",
                    payload={
                        "consecutive_zero_windows": self._consecutive_zero_drain_windows,
                        "window_seconds": window,
                        "queue_depth": self.get_current_queue_depth(),
                    },
                    log_before=True,
                    log_level=logging.ERROR,
                    context="queue_populator_partition",
                )

        return self._partition_detected
    def _circuit_breaker_allow(self) -> bool:
        """Check if circuit breaker allows operation.

        Returns:
            True if operation is allowed.
        """
        if not self.config.circuit_breaker_enabled:
            return True

        now = time.time()

        if self._circuit_state == self._CircuitState.CLOSED:
            return True

        if self._circuit_state == self._CircuitState.OPEN:
            # Check if reset timeout has elapsed
            if now - self._circuit_opened_at >= self.config.circuit_breaker_reset_timeout_seconds:
                self._circuit_state = self._CircuitState.HALF_OPEN
                self._circuit_half_open_successes = 0
                logger.info("[QueuePopulator] Circuit breaker entering HALF_OPEN state")
                return True
            return False

        if self._circuit_state == self._CircuitState.HALF_OPEN:
            return True

        return True
    def _circuit_breaker_record_success(self) -> None:
        """Record a successful operation for circuit breaker."""
        if not self.config.circuit_breaker_enabled:
            return

        if self._circuit_state == self._CircuitState.HALF_OPEN:
            self._circuit_half_open_successes += 1
            if self._circuit_half_open_successes >= self.config.circuit_breaker_half_open_successes:
                self._circuit_state = self._CircuitState.CLOSED
                self._circuit_failure_count = 0
                logger.info("[QueuePopulator] Circuit breaker CLOSED after recovery")
                safe_emit_event(
                    event_type="CIRCUIT_BREAKER_CLOSED",
                    payload={"component": "queue_populator"},
                    log_before=False,
                    context="queue_populator_circuit",
                )
        elif self._circuit_state == self._CircuitState.CLOSED:
            # Decay failure count on success
            if self._circuit_failure_count > 0:
                self._circuit_failure_count -= 1
    def _circuit_breaker_record_failure(self) -> None:
        """Record a failed operation for circuit breaker."""
        if not self.config.circuit_breaker_enabled:
            return

        self._circuit_failure_count += 1

        if self._circuit_state == self._CircuitState.HALF_OPEN:
            # Immediate return to OPEN on failure during half-open
            self._circuit_state = self._CircuitState.OPEN
            self._circuit_opened_at = time.time()
            logger.warning("[QueuePopulator] Circuit breaker REOPENED after half-open failure")

        elif self._circuit_failure_count >= self.config.circuit_breaker_failure_threshold:
            self._circuit_state = self._CircuitState.OPEN
            self._circuit_opened_at = time.time()
            logger.error(
                f"[QueuePopulator] Circuit breaker OPENED after "
                f"{self._circuit_failure_count} failures"
            )
            safe_emit_event(
                event_type="CIRCUIT_BREAKER_OPENED",
                payload={
                    "component": "queue_populator",
                    "failure_count": self._circuit_failure_count,
                },
                log_before=True,
                log_level=logging.ERROR,
                context="queue_populator_circuit",
            )
    def get_health_metrics(self) -> dict[str, Any]:
        """Get current health metrics for monitoring.

        Returns:
            Dictionary with health metrics.
        """
        now = time.time()
        bp_level, reduction_factor = self._check_backpressure()

        return {
            "queue_depth": self.get_current_queue_depth(),
            "backpressure_level": bp_level.value,
            "backpressure_reduction_factor": reduction_factor,
            "backoff_active": self._is_backing_off(),
            "backoff_current_seconds": self._backoff_current_seconds,
            "backoff_remaining_seconds": max(0, self._backoff_until - now),
            "consecutive_hard_limit_hits": self._consecutive_hard_limit_hits,
            "drain_rate_per_minute": self._calculate_drain_rate(),
            "partition_detected": self._partition_detected,
            "partition_duration_seconds": (now - self._partition_detected_at) if self._partition_detected else 0,
            "consecutive_zero_drain_windows": self._consecutive_zero_drain_windows,
            "circuit_state": self._circuit_state.value if hasattr(self, '_circuit_state') else "unknown",
            "circuit_failure_count": self._circuit_failure_count,
            "cluster_health_factor": self._cluster_health_factor,
            "dead_nodes_count": len(self._dead_nodes),
        }
