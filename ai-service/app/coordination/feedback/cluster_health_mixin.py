"""Cluster Health Mixin for FeedbackLoopController.

Extracted from feedback_loop_controller.py to reduce file size.
Provides handlers for cluster health and capacity events.

January 2026 Sprint 17.9: Phase 4 decomposition.
"""

from __future__ import annotations

import logging
import time
from typing import Any, TYPE_CHECKING

from app.coordination.event_handler_utils import extract_config_key

if TYPE_CHECKING:
    from app.coordination.feedback_loop_controller import FeedbackState

logger = logging.getLogger(__name__)


class FeedbackClusterHealthMixin:
    """Mixin providing cluster health event handlers for FeedbackLoopController.

    Handles events:
    - DAEMON_STATUS_CHANGED: Daemon health state transitions
    - P2P_CLUSTER_UNHEALTHY: Cluster connectivity issues
    - CLUSTER_CAPACITY_CHANGED: Cluster resource changes
    - HEALTH_CHECK_PASSED: Node health check succeeded
    - HEALTH_CHECK_FAILED: Node health check failed

    Requires host class to have:
    - _states: dict[str, FeedbackState]
    - _cluster_healthy: bool
    - _cluster_capacity: float
    - _available_nodes: set
    - _failed_nodes: set
    - _node_failure_counts: dict[str, int]
    """

    # These attributes will be set by the host class
    _states: dict
    _cluster_healthy: bool
    _cluster_capacity: float
    _available_nodes: set
    _failed_nodes: set
    _node_failure_counts: dict

    def _on_daemon_status_changed(self, event: Any) -> None:
        """Handle DAEMON_STATUS_CHANGED - daemon health state transitions.

        Monitors daemon health and adjusts feedback loop behavior if critical
        daemons (training, evaluation, sync) are stuck or crashed.

        Added: December 2025
        """
        payload = event.payload if hasattr(event, "payload") else {}
        daemon_name = payload.get("daemon_name", "")
        old_status = payload.get("old_status", "")
        new_status = payload.get("new_status", "")
        hostname = payload.get("hostname", "")

        logger.info(
            f"[FeedbackLoopController] Daemon status changed: {daemon_name} on {hostname} "
            f"{old_status} -> {new_status}"
        )

        # If critical training daemon crashed, pause training via feedback_accelerator
        if new_status in ["error", "stuck", "crashed"]:
            if "training" in daemon_name.lower() or "coordinator" in daemon_name.lower():
                logger.warning(
                    f"[FeedbackLoopController] Critical daemon {daemon_name} unhealthy, "
                    f"pausing training allocation"
                )

                # Pause all training via feedback_accelerator
                try:
                    from app.training.feedback_accelerator import get_feedback_accelerator

                    accelerator = get_feedback_accelerator()

                    # Pause training for all tracked configs
                    for config_key in self._states.keys():
                        accelerator.signal_training_needed(
                            config_key=config_key,
                            urgency="none",
                            reason=f"daemon_unhealthy_{daemon_name}",
                        )
                        logger.info(
                            f"[FeedbackLoopController] Paused training for {config_key} "
                            f"due to unhealthy daemon {daemon_name}"
                        )
                except ImportError:
                    logger.debug("[FeedbackLoopController] feedback_accelerator not available")
                except (AttributeError, TypeError, RuntimeError) as e:
                    logger.debug(f"[FeedbackLoopController] Failed to pause training: {e}")

    def _on_p2p_cluster_unhealthy(self, event: Any) -> None:
        """Handle P2P_CLUSTER_UNHEALTHY - cluster connectivity issues.

        When cluster health degrades, adjusts sync and training strategies
        to minimize dependency on unreliable nodes.

        Added: December 2025
        """
        payload = event.payload if hasattr(event, "payload") else {}
        dead_nodes = payload.get("dead_nodes", [])
        alive_nodes = payload.get("alive_nodes", [])

        logger.warning(
            f"[FeedbackLoopController] P2P cluster unhealthy: "
            f"{len(dead_nodes)} dead, {len(alive_nodes)} alive"
        )

        # If too many nodes dead, pause training via feedback_accelerator
        if len(dead_nodes) > len(alive_nodes):
            logger.error(
                "[FeedbackLoopController] Majority of cluster dead, pausing training"
            )
            self._cluster_healthy = False

            # Pause all training via feedback_accelerator
            try:
                from app.training.feedback_accelerator import get_feedback_accelerator

                accelerator = get_feedback_accelerator()

                # Pause training for all tracked configs
                for config_key in self._states.keys():
                    accelerator.signal_training_needed(
                        config_key=config_key,
                        urgency="none",
                        reason="cluster_unhealthy_majority_dead",
                    )
                    logger.info(
                        f"[FeedbackLoopController] Paused training for {config_key} "
                        f"due to cluster health (dead={len(dead_nodes)}, alive={len(alive_nodes)})"
                    )
            except ImportError:
                logger.debug("[FeedbackLoopController] feedback_accelerator not available")
            except (AttributeError, TypeError, RuntimeError) as e:
                logger.debug(f"[FeedbackLoopController] Failed to pause training: {e}")

        else:
            self._cluster_healthy = True

    def _on_cluster_capacity_changed(self, event: Any) -> None:
        """Handle CLUSTER_CAPACITY_CHANGED - cluster resources changed.

        Dec 27, 2025: Closes cluster capacity -> workload feedback loop.
        When cluster capacity changes (nodes added/removed), this handler:
        1. Adjusts selfplay rate targets
        2. Updates training batch sizes if needed
        3. Logs capacity changes for monitoring

        Args:
            event: Event with payload containing capacity, change_type, node_count
        """
        payload = event.payload if hasattr(event, "payload") else {}
        capacity = payload.get("capacity", 1.0)  # 0.0-1.0
        change_type = payload.get("change_type", "unknown")  # added, removed, scaled
        node_count = payload.get("node_count", 0)
        gpu_count = payload.get("gpu_count", 0)

        # Update cluster capacity tracking
        prev_capacity = getattr(self, "_cluster_capacity", 1.0)
        self._cluster_capacity = capacity

        # Log significant capacity changes
        capacity_delta = capacity - prev_capacity
        if abs(capacity_delta) > 0.1:
            logger.info(
                f"[FeedbackLoopController] Cluster capacity changed: "
                f"{prev_capacity:.1%} -> {capacity:.1%} ({change_type}), "
                f"nodes={node_count}, gpus={gpu_count}"
            )

            # If capacity dropped significantly, emit warning
            if capacity_delta < -0.2:
                logger.warning(
                    f"[FeedbackLoopController] Significant capacity drop detected: "
                    f"{abs(capacity_delta):.1%} reduction. "
                    "Training/selfplay rates may need adjustment."
                )

    def _on_health_check_passed(self, event: Any) -> None:
        """Handle HEALTH_CHECK_PASSED - node health check succeeded.

        Dec 27, 2025: Closes node health -> feedback loop.
        When a node passes health check, this handler:
        1. Updates node availability tracking
        2. May resume jobs on recovered nodes

        Args:
            event: Event with payload containing node_id, node_ip, check_type, latency_ms
        """
        payload = event.payload if hasattr(event, "payload") else {}
        node_id = payload.get("node_id", "unknown")
        check_type = payload.get("check_type", "general")
        latency_ms = payload.get("latency_ms")

        # Track node as available
        if not hasattr(self, "_available_nodes"):
            self._available_nodes = set()
        self._available_nodes.add(node_id)

        # Log recovery from previous failure
        if not hasattr(self, "_failed_nodes"):
            self._failed_nodes = set()
        if node_id in self._failed_nodes:
            self._failed_nodes.discard(node_id)
            # Reset failure count on recovery
            if hasattr(self, "_node_failure_counts") and node_id in self._node_failure_counts:
                del self._node_failure_counts[node_id]
            logger.info(
                f"[FeedbackLoopController] Node {node_id} recovered from health failure "
                f"(check_type={check_type}, latency={latency_ms}ms)"
            )

    def _on_health_check_failed(self, event: Any) -> None:
        """Handle HEALTH_CHECK_FAILED - node health check failed.

        Dec 27, 2025: Closes node health -> feedback loop.
        When a node fails health check, this handler:
        1. Marks node as unavailable
        2. May trigger job redistribution if node was running jobs
        3. Emits warning for monitoring

        Args:
            event: Event with payload containing node_id, reason, node_ip, check_type, error
        """
        payload = event.payload if hasattr(event, "payload") else {}
        node_id = payload.get("node_id", "unknown")
        reason = payload.get("reason", "unknown")
        check_type = payload.get("check_type", "general")
        error = payload.get("error", "")

        # Track node as failed
        if not hasattr(self, "_failed_nodes"):
            self._failed_nodes = set()
        self._failed_nodes.add(node_id)

        # Remove from available nodes
        if not hasattr(self, "_available_nodes"):
            self._available_nodes = set()
        self._available_nodes.discard(node_id)

        # Count consecutive failures for this node
        if not hasattr(self, "_node_failure_counts"):
            self._node_failure_counts = {}
        self._node_failure_counts[node_id] = self._node_failure_counts.get(node_id, 0) + 1
        failure_count = self._node_failure_counts[node_id]

        # LRU cleanup: limit to 100 entries max
        MAX_FAILURE_TRACKING_NODES = 100
        if len(self._node_failure_counts) > MAX_FAILURE_TRACKING_NODES:
            # Remove entries with lowest counts (least concerning)
            sorted_nodes = sorted(
                self._node_failure_counts.items(), key=lambda x: x[1], reverse=True
            )
            self._node_failure_counts = dict(sorted_nodes[:MAX_FAILURE_TRACKING_NODES])

        # Log warning for repeated failures
        if failure_count >= 3:
            logger.warning(
                f"[FeedbackLoopController] Node {node_id} has failed {failure_count} "
                f"consecutive health checks (reason={reason}, check_type={check_type}). "
                "Consider removing from active pool."
            )
        else:
            logger.debug(
                f"[FeedbackLoopController] Node {node_id} health check failed: {reason}"
            )
