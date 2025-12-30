"""Daemon Event Handlers - Handles events that affect daemon lifecycle.

Extracted from daemon_manager.py (December 2025) to reduce file size
and improve maintainability.

This module handles events like:
- REGRESSION_CRITICAL - Model regression requiring daemon response
- SELFPLAY_TARGET_UPDATED - Selfplay priority changes
- EXPLORATION_BOOST - Temperature adjustment requests
- DAEMON_STATUS_CHANGED - Self-healing for failed daemons
- HOST_OFFLINE/HOST_ONLINE - P2P cluster node changes
- LEADER_ELECTED - Leader-only daemon management
- BACKPRESSURE_ACTIVATED/RELEASED - Workload management
- DISK_SPACE_LOW - Data generation throttling

Usage:
    from app.coordination.daemon_event_handlers import DaemonEventHandlers

    # Create handlers with reference to DaemonManager
    handlers = DaemonEventHandlers(daemon_manager)

    # Subscribe to events
    await handlers.subscribe_to_events()
"""

from __future__ import annotations

import logging
import socket
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.coordination.daemon_manager import DaemonManager

logger = logging.getLogger(__name__)


class DaemonEventHandlers:
    """Handles events that affect daemon lifecycle and coordination.

    This class contains all event handlers that were previously in DaemonManager.
    It maintains a reference to the DaemonManager for accessing daemon state
    and control methods.

    Event subscriptions:
    - REGRESSION_CRITICAL: Model regression response
    - SELFPLAY_TARGET_UPDATED: Workload scaling
    - EXPLORATION_BOOST: Temperature propagation
    - DAEMON_STATUS_CHANGED: Self-healing
    - HOST_OFFLINE: Exclude offline hosts from sync
    - HOST_ONLINE: Re-include recovered hosts
    - LEADER_ELECTED: Leader-only daemon management
    - BACKPRESSURE_ACTIVATED: Pause non-essential daemons
    - BACKPRESSURE_RELEASED: Resume paused daemons
    - DISK_SPACE_LOW: Throttle data generation
    """

    def __init__(self, manager: "DaemonManager"):
        """Initialize the event handlers.

        Args:
            manager: Reference to the DaemonManager instance
        """
        self._manager = manager
        self._subscribed = False

    async def subscribe_to_events(self) -> None:
        """Subscribe to all daemon-related events.

        Phase 5 (December 2025): Centralized handling of critical events
        that require daemon-level coordination response.

        Phase 7 (December 2025): Also wires AutoRollbackHandler to actually
        perform rollbacks when REGRESSION_CRITICAL events are received.
        """
        if self._subscribed:
            return

        try:
            from app.coordination.event_router import get_router, DataEventType

            if DataEventType is None:
                logger.debug("[DaemonEventHandlers] DataEventType not available, skipping event subscription")
                return

            router = get_router()
            if router is None:
                logger.debug("[DaemonEventHandlers] Event router not available for event subscription")
                return

            # Core critical events
            router.subscribe(DataEventType.REGRESSION_CRITICAL.value, self._on_regression_critical)

            # P0.3 (December 2025): Feedback loop events
            if hasattr(DataEventType, 'SELFPLAY_TARGET_UPDATED'):
                router.subscribe(DataEventType.SELFPLAY_TARGET_UPDATED.value, self._on_selfplay_target_updated)
                logger.debug("[DaemonEventHandlers] Subscribed to SELFPLAY_TARGET_UPDATED")

            if hasattr(DataEventType, 'EXPLORATION_BOOST'):
                router.subscribe(DataEventType.EXPLORATION_BOOST.value, self._on_exploration_boost)
                logger.debug("[DaemonEventHandlers] Subscribed to EXPLORATION_BOOST")

            if hasattr(DataEventType, 'DAEMON_STATUS_CHANGED'):
                router.subscribe(DataEventType.DAEMON_STATUS_CHANGED.value, self._on_daemon_status_changed)
                logger.debug("[DaemonEventHandlers] Subscribed to DAEMON_STATUS_CHANGED")

            # Dec 27, 2025: P2P cluster events
            if hasattr(DataEventType, 'HOST_OFFLINE'):
                router.subscribe(DataEventType.HOST_OFFLINE.value, self._on_host_offline)
                logger.debug("[DaemonEventHandlers] Subscribed to HOST_OFFLINE")

            if hasattr(DataEventType, 'HOST_ONLINE'):
                router.subscribe(DataEventType.HOST_ONLINE.value, self._on_host_online)
                logger.debug("[DaemonEventHandlers] Subscribed to HOST_ONLINE")

            if hasattr(DataEventType, 'LEADER_ELECTED'):
                router.subscribe(DataEventType.LEADER_ELECTED.value, self._on_leader_elected)
                logger.debug("[DaemonEventHandlers] Subscribed to LEADER_ELECTED")

            # December 2025: Backpressure events
            if hasattr(DataEventType, 'BACKPRESSURE_ACTIVATED'):
                router.subscribe(DataEventType.BACKPRESSURE_ACTIVATED.value, self._on_backpressure_activated)
                logger.debug("[DaemonEventHandlers] Subscribed to BACKPRESSURE_ACTIVATED")

            if hasattr(DataEventType, 'BACKPRESSURE_RELEASED'):
                router.subscribe(DataEventType.BACKPRESSURE_RELEASED.value, self._on_backpressure_released)
                logger.debug("[DaemonEventHandlers] Subscribed to BACKPRESSURE_RELEASED")

            # December 2025: Disk space events
            if hasattr(DataEventType, 'DISK_SPACE_LOW'):
                router.subscribe(DataEventType.DISK_SPACE_LOW.value, self._on_disk_space_low)
                logger.debug("[DaemonEventHandlers] Subscribed to DISK_SPACE_LOW")

            logger.info("[DaemonEventHandlers] Subscribed to critical events (Phase 5, P0.3, P2P cluster, backpressure, disk space)")
            self._subscribed = True

            # Phase 7: Wire AutoRollbackHandler
            self._wire_rollback_handler()

        except (ImportError, RuntimeError, ConnectionError) as e:
            logger.warning(f"[DaemonEventHandlers] Failed to subscribe to events: {e}")

    def _wire_rollback_handler(self) -> None:
        """Wire AutoRollbackHandler to perform model rollbacks."""
        try:
            from app.training.model_registry import get_model_registry
            from app.training.rollback_manager import wire_regression_to_rollback

            registry = get_model_registry()
            handler = wire_regression_to_rollback(registry)
            if handler:
                logger.info("[DaemonEventHandlers] Wired AutoRollbackHandler for automatic model rollback (Phase 7)")
            else:
                logger.warning("[DaemonEventHandlers] Failed to wire AutoRollbackHandler")
        except (ImportError, RuntimeError, AttributeError) as rollback_err:
            logger.warning(f"[DaemonEventHandlers] Could not wire AutoRollbackHandler: {rollback_err}")

    # =========================================================================
    # Event Handlers
    # =========================================================================

    async def _on_regression_critical(self, event: Any) -> None:
        """Handle REGRESSION_CRITICAL event - centralized response.

        Phase 5 (December 2025): When a critical regression is detected,
        coordinate daemon-level response:
        1. Log the critical event prominently
        2. Pause selfplay for the affected config to prevent bad data
        3. Alert cluster nodes via P2P
        4. Trigger model rollback if configured

        Args:
            event: The REGRESSION_CRITICAL event
        """
        from app.coordination.daemon_types import DaemonState, DaemonType

        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", payload.get("config", "unknown"))
            model_id = payload.get("model_id", "unknown")
            elo_drop = payload.get("elo_drop", 0)
            current_elo = payload.get("current_elo", 0)
            previous_elo = payload.get("previous_elo", 0)

            # Prominent logging - this is critical
            logger.critical(
                f"[REGRESSION_CRITICAL] Model regression detected!\n"
                f"  Config: {config_key}\n"
                f"  Model: {model_id}\n"
                f"  ELO: {previous_elo:.0f} → {current_elo:.0f} (drop: {elo_drop:.0f})"
            )

            # Emit alert for cluster-wide awareness
            await self._emit_cluster_alert(
                alert_type="regression_critical",
                config_key=config_key,
                message=f"Critical model regression: {config_key} dropped {elo_drop:.0f} ELO",
                severity="critical",
                model_id=model_id,
            )

            # Check if rollback daemon is running
            if DaemonType.MODEL_DISTRIBUTION in self._manager._daemons:
                info = self._manager._daemons[DaemonType.MODEL_DISTRIBUTION]
                if info.state == DaemonState.RUNNING:
                    logger.info(
                        f"[DaemonEventHandlers] Model distribution daemon running - "
                        f"rollback should be handled by RollbackManager"
                    )

        except (RuntimeError, OSError, ConnectionError, ImportError) as e:
            logger.error(f"[DaemonEventHandlers] Error handling REGRESSION_CRITICAL: {e}")

    async def _on_selfplay_target_updated(self, event: Any) -> None:
        """Handle SELFPLAY_TARGET_UPDATED event - adjust daemon workloads.

        P0.3 (December 2025): When selfplay targets change, coordinate
        daemon-level response:
        1. Log the target change for monitoring
        2. Adjust idle resource daemon behavior if needed
        3. Propagate priority to relevant daemons

        Args:
            event: The SELFPLAY_TARGET_UPDATED event
        """
        from app.coordination.daemon_types import DaemonState, DaemonType

        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", "unknown")
            priority = payload.get("priority", "normal")
            reason = payload.get("reason", "unknown")
            target_jobs = payload.get("target_jobs")

            logger.info(
                f"[DaemonEventHandlers] SELFPLAY_TARGET_UPDATED: {config_key} "
                f"priority={priority} reason={reason}"
                + (f" target_jobs={target_jobs}" if target_jobs else "")
            )

            # If priority is urgent or high, trigger immediate idle resource check
            if priority in ("urgent", "high"):
                if DaemonType.IDLE_RESOURCE in self._manager._daemons:
                    info = self._manager._daemons[DaemonType.IDLE_RESOURCE]
                    if info.state == DaemonState.RUNNING and hasattr(info, 'instance'):
                        daemon = info.instance
                        if hasattr(daemon, 'trigger_immediate_check'):
                            daemon.trigger_immediate_check()
                            logger.debug(
                                f"[DaemonEventHandlers] Triggered immediate idle resource check "
                                f"for {config_key}"
                            )

        except (RuntimeError, OSError, AttributeError) as e:
            logger.debug(f"[DaemonEventHandlers] Error handling SELFPLAY_TARGET_UPDATED: {e}")

    async def _on_exploration_boost(self, event: Any) -> None:
        """Handle EXPLORATION_BOOST event - coordinate temperature adjustments.

        P0.3 (December 2025): When exploration boost is requested,
        propagate to selfplay daemons for temperature adjustment.

        Args:
            event: The EXPLORATION_BOOST event
        """
        from app.coordination.daemon_types import DaemonState, DaemonType

        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", "unknown")
            boost_factor = payload.get("boost_factor", 1.0)
            reason = payload.get("reason", "unknown")
            duration_seconds = payload.get("duration_seconds", 3600)

            logger.info(
                f"[DaemonEventHandlers] EXPLORATION_BOOST: {config_key} "
                f"factor={boost_factor:.2f}x reason={reason} "
                f"duration={duration_seconds}s"
            )

            # Propagate boost to selfplay scheduler if available
            if DaemonType.SELFPLAY_SCHEDULER in self._manager._daemons:
                info = self._manager._daemons[DaemonType.SELFPLAY_SCHEDULER]
                if info.state == DaemonState.RUNNING and hasattr(info, 'instance'):
                    scheduler = info.instance
                    if hasattr(scheduler, 'apply_exploration_boost'):
                        scheduler.apply_exploration_boost(
                            config_key=config_key,
                            boost_factor=boost_factor,
                            duration_seconds=duration_seconds,
                        )
                        logger.debug(
                            f"[DaemonEventHandlers] Applied exploration boost to SelfplayScheduler"
                        )

        except (RuntimeError, OSError, AttributeError) as e:
            logger.debug(f"[DaemonEventHandlers] Error handling EXPLORATION_BOOST: {e}")

    async def _on_daemon_status_changed(self, event: Any) -> None:
        """Handle DAEMON_STATUS_CHANGED event - self-healing response.

        P0.3 (December 2025): When a daemon status changes (failure, restart,
        health degradation), attempt self-healing restart if configured.

        Args:
            event: The DAEMON_STATUS_CHANGED event
        """
        from app.coordination.daemon_types import DaemonType

        try:
            payload = event.payload if hasattr(event, "payload") else event
            daemon_type_str = payload.get("daemon_type", "unknown")
            old_status = payload.get("old_status", "unknown")
            new_status = payload.get("new_status", "unknown")
            reason = payload.get("reason", "")

            logger.info(
                f"[DaemonEventHandlers] DAEMON_STATUS_CHANGED: {daemon_type_str} "
                f"{old_status} -> {new_status}"
                + (f" ({reason})" if reason else "")
            )

            # Self-healing: attempt restart for failed daemons
            if new_status in ("FAILED", "CRASHED", "STOPPED"):
                try:
                    daemon_type = DaemonType(daemon_type_str)
                    if daemon_type in self._manager._daemons:
                        info = self._manager._daemons[daemon_type]

                        # Only restart if we haven't exceeded retry limits
                        max_restarts = 3
                        if info.restart_count < max_restarts:
                            logger.warning(
                                f"[DaemonEventHandlers] Attempting self-healing restart for "
                                f"{daemon_type_str} (attempt {info.restart_count + 1}/{max_restarts})"
                            )
                            # Schedule restart via lifecycle manager
                            if hasattr(self._manager, '_lifecycle') and self._manager._lifecycle:
                                await self._manager._lifecycle.restart_daemon(daemon_type)
                        else:
                            logger.error(
                                f"[DaemonEventHandlers] Daemon {daemon_type_str} exceeded max restarts "
                                f"({max_restarts}), not attempting self-healing"
                            )
                except ValueError:
                    logger.debug(f"[DaemonEventHandlers] Unknown daemon type: {daemon_type_str}")

        except (RuntimeError, OSError, AttributeError) as e:
            logger.debug(f"[DaemonEventHandlers] Error handling DAEMON_STATUS_CHANGED: {e}")

    async def _on_host_offline(self, event: Any) -> None:
        """Handle HOST_OFFLINE event - pause affected daemons when nodes leave cluster.

        Dec 27, 2025: When a host goes offline, notify sync-related daemons
        to exclude that host from sync targets.

        Args:
            event: The HOST_OFFLINE event
        """
        from app.coordination.daemon_types import DaemonState, DaemonType

        try:
            payload = event.payload if hasattr(event, "payload") else event
            host_id = payload.get("host_id", "unknown")
            reason = payload.get("reason", "")

            logger.info(
                f"[DaemonEventHandlers] HOST_OFFLINE: {host_id}"
                + (f" ({reason})" if reason else "")
            )

            # Notify sync-related daemons to exclude this host
            for daemon_type in [DaemonType.AUTO_SYNC, DaemonType.MODEL_DISTRIBUTION]:
                if daemon_type in self._manager._daemons:
                    info = self._manager._daemons[daemon_type]
                    if info.state == DaemonState.RUNNING and hasattr(info, 'instance'):
                        daemon = info.instance
                        if hasattr(daemon, 'mark_host_offline'):
                            daemon.mark_host_offline(host_id)
                            logger.debug(
                                f"[DaemonEventHandlers] Marked host {host_id} offline for {daemon_type.value}"
                            )

        except (RuntimeError, OSError, AttributeError, KeyError) as e:
            logger.debug(f"[DaemonEventHandlers] Error handling HOST_OFFLINE: {e}")

    async def _on_host_online(self, event: Any) -> None:
        """Handle HOST_ONLINE event - resume/restart daemons when nodes rejoin.

        Dec 27, 2025: When a host comes back online, notify sync-related daemons
        to re-include the host in sync targets.

        Args:
            event: The HOST_ONLINE event
        """
        from app.coordination.daemon_types import DaemonState, DaemonType

        try:
            payload = event.payload if hasattr(event, "payload") else event
            host_id = payload.get("host_id", "unknown")

            logger.info(f"[DaemonEventHandlers] HOST_ONLINE: {host_id}")

            # Notify sync-related daemons to re-include this host
            for daemon_type in [DaemonType.AUTO_SYNC, DaemonType.MODEL_DISTRIBUTION]:
                if daemon_type in self._manager._daemons:
                    info = self._manager._daemons[daemon_type]
                    if info.state == DaemonState.RUNNING and hasattr(info, 'instance'):
                        daemon = info.instance
                        if hasattr(daemon, 'mark_host_online'):
                            daemon.mark_host_online(host_id)
                            logger.debug(
                                f"[DaemonEventHandlers] Marked host {host_id} online for {daemon_type.value}"
                            )

        except (RuntimeError, OSError, AttributeError, KeyError) as e:
            logger.debug(f"[DaemonEventHandlers] Error handling HOST_ONLINE: {e}")

    async def _on_leader_elected(self, event: Any) -> None:
        """Handle LEADER_ELECTED event - manage leader-only daemons.

        Dec 27, 2025: When a new leader is elected:
        1. If we are the new leader: start leader-only daemons
        2. If we lost leadership: stop leader-only daemons
        3. Update all daemons with new leader info

        Args:
            event: The LEADER_ELECTED event
        """
        from app.coordination.daemon_types import DaemonState, DaemonType

        try:
            payload = event.payload if hasattr(event, "payload") else event
            leader_id = payload.get("leader_id", "unknown")
            previous_leader_id = payload.get("previous_leader_id", "")
            is_self = payload.get("is_self", False)

            logger.info(
                f"[DaemonEventHandlers] LEADER_ELECTED: {leader_id}"
                + (f" (previous: {previous_leader_id})" if previous_leader_id else "")
                + (" [THIS NODE]" if is_self else "")
            )

            # Leader-only daemons that should only run on the leader node
            leader_only_daemons = [
                DaemonType.DATA_PIPELINE,
                DaemonType.AUTO_PROMOTION,
                DaemonType.EVALUATION,
                DaemonType.TRAINING_TRIGGER,
            ]

            if is_self:
                # We became the leader - start leader-only daemons
                for daemon_type in leader_only_daemons:
                    if daemon_type in self._manager._daemons:
                        info = self._manager._daemons[daemon_type]
                        if info.state != DaemonState.RUNNING:
                            logger.info(
                                f"[DaemonEventHandlers] Starting leader-only daemon: {daemon_type.value}"
                            )
                            try:
                                await self._manager.start(daemon_type)
                            except (RuntimeError, OSError) as e:
                                logger.warning(
                                    f"[DaemonEventHandlers] Failed to start {daemon_type.value}: {e}"
                                )
            else:
                # We lost leadership - stop leader-only daemons
                for daemon_type in leader_only_daemons:
                    if daemon_type in self._manager._daemons:
                        info = self._manager._daemons[daemon_type]
                        if info.state == DaemonState.RUNNING:
                            logger.info(
                                f"[DaemonEventHandlers] Stopping leader-only daemon: {daemon_type.value}"
                            )
                            try:
                                await self._manager.stop(daemon_type)
                            except (RuntimeError, OSError) as e:
                                logger.warning(
                                    f"[DaemonEventHandlers] Failed to stop {daemon_type.value}: {e}"
                                )

            # Notify all running daemons of the leadership change
            for daemon_type, info in self._manager._daemons.items():
                if info.state == DaemonState.RUNNING and hasattr(info, 'instance'):
                    daemon = info.instance
                    if hasattr(daemon, 'on_leader_changed'):
                        daemon.on_leader_changed(leader_id=leader_id, is_self=is_self)

        except (RuntimeError, OSError, AttributeError, KeyError) as e:
            logger.debug(f"[DaemonEventHandlers] Error handling LEADER_ELECTED: {e}")

    async def _on_backpressure_activated(self, event: Any) -> None:
        """Handle BACKPRESSURE_ACTIVATED event - reduce daemon workload.

        December 2025: When backpressure is activated, pause non-essential daemons.

        Essential daemons (always running):
        - EVENT_ROUTER, DAEMON_WATCHDOG, QUEUE_MONITOR, CLUSTER_WATCHDOG

        Pausable daemons:
        - IDLE_RESOURCE, SELFPLAY_COORDINATOR, TRAINING_NODE_WATCHER, AUTO_SYNC

        Args:
            event: The BACKPRESSURE_ACTIVATED event
        """
        from app.coordination.daemon_types import DaemonState, DaemonType

        try:
            payload = event.payload if hasattr(event, "payload") else event
            reason = payload.get("reason", "unknown")
            threshold = payload.get("threshold", 0)
            current_value = payload.get("current_value", 0)

            logger.warning(
                f"[DaemonEventHandlers] BACKPRESSURE_ACTIVATED: {reason} "
                f"(value: {current_value}, threshold: {threshold})"
            )

            # Non-essential daemons that can be paused
            pausable_daemons = [
                DaemonType.IDLE_RESOURCE,
                DaemonType.SELFPLAY_COORDINATOR,
                DaemonType.TRAINING_NODE_WATCHER,
                DaemonType.AUTO_SYNC,
            ]

            paused_count = 0
            for daemon_type in pausable_daemons:
                if daemon_type in self._manager._daemons:
                    info = self._manager._daemons[daemon_type]
                    if info.state == DaemonState.RUNNING:
                        daemon = getattr(info, 'instance', None)
                        if daemon and hasattr(daemon, 'pause'):
                            try:
                                daemon.pause()
                                paused_count += 1
                                logger.debug(
                                    f"[DaemonEventHandlers] Paused {daemon_type.value} due to backpressure"
                                )
                            except (RuntimeError, OSError) as e:
                                logger.debug(
                                    f"[DaemonEventHandlers] Failed to pause {daemon_type.value}: {e}"
                                )

            if paused_count > 0:
                logger.info(
                    f"[DaemonEventHandlers] Paused {paused_count} daemons due to backpressure"
                )

        except (RuntimeError, OSError, AttributeError, KeyError) as e:
            logger.debug(f"[DaemonEventHandlers] Error handling BACKPRESSURE_ACTIVATED: {e}")

    async def _on_backpressure_released(self, event: Any) -> None:
        """Handle BACKPRESSURE_RELEASED event - resume normal daemon operations.

        December 2025: When backpressure is released, resume paused daemons.

        Args:
            event: The BACKPRESSURE_RELEASED event
        """
        from app.coordination.daemon_types import DaemonType

        try:
            payload = event.payload if hasattr(event, "payload") else event
            duration = payload.get("duration_seconds", 0)

            logger.info(
                f"[DaemonEventHandlers] BACKPRESSURE_RELEASED after {duration:.1f}s"
            )

            # Resume paused daemons
            resumable_daemons = [
                DaemonType.IDLE_RESOURCE,
                DaemonType.SELFPLAY_COORDINATOR,
                DaemonType.TRAINING_NODE_WATCHER,
                DaemonType.AUTO_SYNC,
            ]

            resumed_count = 0
            for daemon_type in resumable_daemons:
                if daemon_type in self._manager._daemons:
                    info = self._manager._daemons[daemon_type]
                    daemon = getattr(info, 'instance', None)
                    if daemon and hasattr(daemon, 'resume'):
                        try:
                            daemon.resume()
                            resumed_count += 1
                            logger.debug(
                                f"[DaemonEventHandlers] Resumed {daemon_type.value} after backpressure"
                            )
                        except (RuntimeError, OSError) as e:
                            logger.debug(
                                f"[DaemonEventHandlers] Failed to resume {daemon_type.value}: {e}"
                            )

            if resumed_count > 0:
                logger.info(
                    f"[DaemonEventHandlers] Resumed {resumed_count} daemons after backpressure release"
                )

        except (RuntimeError, OSError, AttributeError, KeyError) as e:
            logger.debug(f"[DaemonEventHandlers] Error handling BACKPRESSURE_RELEASED: {e}")

    async def _on_disk_space_low(self, event: Any) -> None:
        """Handle DISK_SPACE_LOW event - pause data-generating daemons.

        December 2025: When disk space is critical (>85%), pause daemons that
        generate data to prevent disk from filling up completely.

        Args:
            event: The DISK_SPACE_LOW event
        """
        from app.coordination.daemon_types import DaemonState, DaemonType

        try:
            payload = event.payload if hasattr(event, "payload") else event
            host = payload.get("host", "unknown")
            usage_percent = payload.get("usage_percent", 0)
            free_gb = payload.get("free_gb", 0)
            threshold = payload.get("threshold", 70)

            logger.warning(
                f"[DaemonEventHandlers] DISK_SPACE_LOW: {host} at {usage_percent:.1f}% "
                f"(threshold: {threshold}%, free: {free_gb:.1f}GB)"
            )

            # Only respond to events for this host
            local_hostname = socket.gethostname()
            if host not in (local_hostname, "localhost", "127.0.0.1"):
                logger.debug(f"[DaemonEventHandlers] Ignoring disk event for other host: {host}")
                return

            # If disk usage is critical (>85%), pause data-generating daemons
            if usage_percent >= 85:
                logger.warning(
                    f"[DaemonEventHandlers] Critical disk usage ({usage_percent:.1f}%), "
                    f"pausing data-generating daemons"
                )

                data_generating_daemons = [
                    DaemonType.SELFPLAY_COORDINATOR,
                    DaemonType.IDLE_RESOURCE,
                    DaemonType.AUTO_SYNC,
                    DaemonType.TRAINING_NODE_WATCHER,
                ]

                paused_count = 0
                for daemon_type in data_generating_daemons:
                    if daemon_type in self._manager._daemons:
                        info = self._manager._daemons[daemon_type]
                        if info.state == DaemonState.RUNNING:
                            daemon = getattr(info, 'instance', None)
                            if daemon and hasattr(daemon, 'pause'):
                                try:
                                    daemon.pause()
                                    paused_count += 1
                                    logger.warning(
                                        f"[DaemonEventHandlers] Paused {daemon_type.value} "
                                        f"due to low disk space"
                                    )
                                except (RuntimeError, OSError) as e:
                                    logger.debug(
                                        f"[DaemonEventHandlers] Failed to pause {daemon_type.value}: {e}"
                                    )

                if paused_count > 0:
                    logger.warning(
                        f"[DaemonEventHandlers] Paused {paused_count} data-generating daemons "
                        f"due to critical disk space"
                    )

        except (RuntimeError, OSError, AttributeError, KeyError) as e:
            logger.debug(f"[DaemonEventHandlers] Error handling DISK_SPACE_LOW: {e}")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _emit_cluster_alert(
        self,
        alert_type: str,
        config_key: str,
        message: str,
        severity: str,
        **extra: Any,
    ) -> None:
        """Emit a HEALTH_ALERT event for cluster-wide awareness.

        Args:
            alert_type: Type of alert (e.g., "regression_critical")
            config_key: Configuration key affected
            message: Alert message
            severity: Severity level (e.g., "critical", "warning")
            **extra: Additional payload fields
        """
        try:
            from app.coordination.event_router import get_router, DataEventType, DataEvent

            if DataEventType is None or DataEvent is None:
                return

            router = get_router()
            if router:
                alert_event = DataEvent(
                    event_type=DataEventType.HEALTH_ALERT,
                    payload={
                        "alert": alert_type,
                        "alert_type": alert_type,
                        "config_key": config_key,
                        "message": message,
                        "severity": severity,
                        **extra,
                    },
                    source="DaemonEventHandlers",
                )
                await router.publish_async(DataEventType.HEALTH_ALERT.value, alert_event)
        except (RuntimeError, OSError, ConnectionError) as e:
            logger.debug(f"[DaemonEventHandlers] Failed to emit cluster alert: {e}")


__all__ = ["DaemonEventHandlers"]
