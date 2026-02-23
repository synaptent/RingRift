"""Event Subscription Wiring for P2P Orchestrator.

February 2026: Extracted from p2p_orchestrator.py to reduce file size.

This module provides standalone functions that wire event subscriptions
for the orchestrator. Each function takes the orchestrator instance and
subscribes to a group of related events.

Usage:
    from scripts.p2p.event_wiring import (
        wire_feedback_loops,
        subscribe_to_daemon_events,
        subscribe_to_feedback_signals,
        subscribe_to_manager_events,
    )

    ok = wire_feedback_loops(orchestrator)
    ok = subscribe_to_daemon_events(orchestrator)
    ok = subscribe_to_feedback_signals(orchestrator)
    ok = subscribe_to_manager_events(orchestrator)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def wire_feedback_loops(orchestrator: Any) -> bool:
    """Wire curriculum feedback loops for self-improvement.

    Connects P2P orchestrator to the training feedback system:
    - Curriculum weights adjust based on Elo velocity
    - Weak configs get boosted/penalized based on evaluation results
    - Quality scores influence exploration temperature
    - Failed promotions reduce config priority

    Returns True if wiring succeeded, False otherwise.
    """
    try:
        from app.coordination.curriculum_integration import wire_all_feedback_loops

        status = wire_all_feedback_loops()
        if status.get("success", False):
            wired_count = status.get("wired_count", 0)
            logger.info(f"Feedback loops: wired {wired_count} bridges successfully")
            return True
        else:
            error = status.get("error", "Unknown error")
            logger.warning(f"Feedback loops: partial wiring - {error}")
            return False
    except ImportError as e:
        logger.debug(f"Feedback loops: curriculum_integration not available: {e}")
        return False
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Feedback loops: failed to wire: {e}")
        return False


def subscribe_to_daemon_events(orchestrator: Any) -> bool:
    """Subscribe to daemon status events for observability.

    Receives DAEMON_STATUS_CHANGED events from daemon_manager to track
    daemon health across the cluster. This enables:
    - Tracking which daemons are running/crashed on each node
    - Auto-recovery of critical daemons
    - Cluster-wide daemon health reporting via /status endpoint

    Returns True if subscription succeeded, False otherwise.
    """
    try:
        from app.coordination.event_router import subscribe

        def handle_daemon_status(event: Any) -> None:
            """Handle daemon status change events."""
            try:
                payload = event.payload if hasattr(event, "payload") else event
                daemon_name = payload.get("daemon_name", "unknown")
                new_status = payload.get("new_status", "unknown")
                hostname = payload.get("hostname", "unknown")
                error = payload.get("error")

                # Track daemon states for cluster health reporting
                if not hasattr(orchestrator, "_daemon_states"):
                    orchestrator._daemon_states = {}
                orchestrator._daemon_states[f"{hostname}:{daemon_name}"] = {
                    "status": new_status,
                    "last_update": time.time(),
                    "error": error,
                }

                # Log critical daemon failures
                if new_status in ("crashed", "failed") and error:
                    logger.warning(
                        f"Daemon {daemon_name} on {hostname} {new_status}: {error}"
                    )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Error handling daemon status event: {e}")

        subscribe("DAEMON_STATUS_CHANGED", handle_daemon_status)
        logger.info("Subscribed to daemon status events")
        return True
    except ImportError as e:
        logger.debug(f"Daemon events: event_router not available: {e}")
        return False
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Daemon events: failed to subscribe: {e}")
        return False


def subscribe_to_feedback_signals(orchestrator: Any) -> bool:
    """Subscribe to training feedback signals for dynamic orchestration.

    Subscribes to key feedback events that should influence cluster
    orchestration decisions:
    - ELO_VELOCITY_CHANGED: Adjust selfplay allocation based on training velocity
    - QUALITY_DEGRADED: Pause/slow selfplay when data quality drops
    - EVALUATION_COMPLETED: Trigger model promotion decisions
    - PROMOTION_FAILED: Revert curriculum weights if promotion fails
    - PLATEAU_DETECTED: Trigger hyperparameter search or curriculum changes
    - EXPLORATION_BOOST: Boost selfplay exploration on training anomalies
    - HANDLER_FAILED: Track event handler failures for monitoring

    Returns True if subscription succeeded, False otherwise.
    """
    try:
        from app.coordination.event_router import subscribe

        def handle_quality_degraded(event: Any) -> None:
            """Handle quality degradation events."""
            try:
                payload = event.payload if hasattr(event, "payload") else event
                config_key = payload.get("config_key", "unknown")
                quality_score = payload.get("quality_score", 0)
                threshold = payload.get("threshold", 0)

                logger.warning(
                    f"Quality degraded for {config_key}: {quality_score:.2f} < {threshold:.2f}"
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Error handling quality degraded event: {e}")

        def handle_elo_velocity_changed(event: Any) -> None:
            """Handle Elo velocity change events."""
            try:
                payload = event.payload if hasattr(event, "payload") else event
                config_key = payload.get("config_key", "unknown")
                velocity = payload.get("velocity", 0)

                if velocity < -50:  # Significant regression
                    logger.warning(f"Elo regression for {config_key}: velocity={velocity}")
                elif velocity > 50:  # Good progress
                    logger.info(f"Elo progress for {config_key}: velocity={velocity}")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Error handling Elo velocity event: {e}")

        def handle_evaluation_completed(event: Any) -> None:
            """Handle evaluation completion events."""
            try:
                payload = event.payload if hasattr(event, "payload") else event
                config_key = payload.get("config_key", "unknown")
                win_rate = payload.get("win_rate", 0)
                opponent = payload.get("opponent", "unknown")

                logger.info(
                    f"Evaluation completed for {config_key}: {win_rate:.1%} vs {opponent}"
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Error handling evaluation completed event: {e}")

        def handle_plateau_detected(event: Any) -> None:
            """Handle training plateau detection events."""
            try:
                payload = event.payload if hasattr(event, "payload") else event
                config_key = payload.get("config_key", "unknown")
                epochs_stalled = payload.get("epochs_stalled", 0)

                logger.warning(
                    f"Training plateau for {config_key}: stalled {epochs_stalled} epochs"
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Error handling plateau detected event: {e}")

        def handle_exploration_boost(event: Any) -> None:
            """Handle exploration boost events from training feedback.

            When training anomalies (loss spikes, stalls) are detected, this
            signals that we should boost exploration in selfplay to generate
            more diverse training data.
            """
            try:
                payload = event.payload if hasattr(event, "payload") else event
                config_key = payload.get("config_key", payload.get("config", "unknown"))
                boost_factor = payload.get("boost_factor", payload.get("boost", 1.0))
                reason = payload.get("reason", "training_anomaly")
                duration = payload.get("duration_seconds", 900)

                logger.info(
                    f"Exploration boost for {config_key}: {boost_factor:.2f}x "
                    f"(reason={reason}, duration={duration}s)"
                )

                # Forward to selfplay scheduler if available
                if hasattr(orchestrator, "selfplay_scheduler") and orchestrator.selfplay_scheduler:
                    orchestrator.selfplay_scheduler.set_exploration_boost(
                        config_key, boost_factor, duration
                    )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Error handling exploration boost event: {e}")

        def handle_promotion_failed(event: Any) -> None:
            """Handle model promotion failure events.

            When model promotion fails (gauntlet failure, threshold not met),
            we should revert curriculum weights and potentially pause training
            for that config until issues are resolved.
            """
            try:
                payload = event.payload if hasattr(event, "payload") else event
                config_key = payload.get("config_key", "unknown")
                model_path = payload.get("model_path", "unknown")
                reason = payload.get("reason", "unknown")
                win_rate = payload.get("win_rate", 0.0)

                logger.warning(
                    f"[P2P] Promotion FAILED for {config_key}: {reason} "
                    f"(model={model_path}, win_rate={win_rate:.1%})"
                )

                # Revert curriculum weights if selfplay scheduler available
                if hasattr(orchestrator, "selfplay_scheduler") and orchestrator.selfplay_scheduler:
                    orchestrator.selfplay_scheduler.record_promotion_failure(config_key)
                    logger.info(f"[P2P] Reduced selfplay priority for {config_key} after promotion failure")

                # Track failed promotions for monitoring
                if not hasattr(orchestrator, "_promotion_failures"):
                    orchestrator._promotion_failures = {}
                if config_key not in orchestrator._promotion_failures:
                    orchestrator._promotion_failures[config_key] = []
                orchestrator._promotion_failures[config_key].append({
                    "timestamp": time.time(),
                    "reason": reason,
                    "win_rate": win_rate,
                })
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Error handling promotion failed event: {e}")

        def handle_handler_failed(event: Any) -> None:
            """Handle event handler failure events.

            When a coordination event handler throws an exception, this event
            is emitted. We track these for monitoring and potentially trigger
            alerts for critical handler failures.
            """
            try:
                payload = event.payload if hasattr(event, "payload") else event
                handler_name = payload.get("handler_name", "unknown")
                event_type = payload.get("event_type", "unknown")
                error = payload.get("error", "unknown")
                coordinator = payload.get("coordinator", "unknown")

                logger.error(
                    f"[P2P] Handler FAILED: {handler_name} for {event_type} "
                    f"in {coordinator}: {error}"
                )

                # Track handler failures for health monitoring
                if not hasattr(orchestrator, "_handler_failures"):
                    orchestrator._handler_failures = {}
                failure_key = f"{coordinator}.{handler_name}"
                if failure_key not in orchestrator._handler_failures:
                    orchestrator._handler_failures[failure_key] = []
                orchestrator._handler_failures[failure_key].append({
                    "timestamp": time.time(),
                    "event_type": event_type,
                    "error": str(error)[:200],  # Truncate long errors
                })

                # Keep only last 10 failures per handler
                if len(orchestrator._handler_failures[failure_key]) > 10:
                    orchestrator._handler_failures[failure_key] = (
                        orchestrator._handler_failures[failure_key][-10:]
                    )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Error handling handler failed event: {e}")

        # Subscribe to all feedback signals
        subscribe("QUALITY_DEGRADED", handle_quality_degraded)
        subscribe("ELO_VELOCITY_CHANGED", handle_elo_velocity_changed)
        subscribe("EVALUATION_COMPLETED", handle_evaluation_completed)
        subscribe("PLATEAU_DETECTED", handle_plateau_detected)
        subscribe("EXPLORATION_BOOST", handle_exploration_boost)
        subscribe("PROMOTION_FAILED", handle_promotion_failed)
        subscribe("HANDLER_FAILED", handle_handler_failed)

        logger.info("Subscribed to training feedback signals")
        return True
    except ImportError as e:
        logger.debug(f"Feedback signals: event_router not available: {e}")
        return False
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Feedback signals: failed to subscribe: {e}")
        return False


def subscribe_to_manager_events(orchestrator: Any) -> bool:
    """Subscribe to manager lifecycle events for coordination.

    Subscribes to critical manager events:
    - TRAINING_STARTED/COMPLETED: Coordinate training transitions
    - TASK_SPAWNED/COMPLETED/FAILED: Track job lifecycle
    - DATA_SYNC_STARTED/COMPLETED: Coordinate data freshness
    - NODE_UNHEALTHY/RECOVERED: Track node health
    - P2P_CLUSTER_HEALTHY/UNHEALTHY: Track cluster health

    Returns True if subscription succeeded, False otherwise.
    """
    try:
        from app.coordination.event_router import subscribe

        def handle_training_started(event: Any) -> None:
            """Handle training start events."""
            try:
                payload = event.payload if hasattr(event, "payload") else event
                config_key = payload.get("config_key", "unknown")
                node_id = payload.get("node_id", "unknown")
                logger.info(f"[P2P] Training started: {config_key} on {node_id}")
                # Track active training in cluster state
                if not hasattr(orchestrator, "_active_training"):
                    orchestrator._active_training = {}
                orchestrator._active_training[config_key] = {
                    "node_id": node_id,
                    "started_at": time.time(),
                }
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Error handling training started event: {e}")

        def handle_training_completed(event: Any) -> None:
            """Handle training completion events."""
            try:
                payload = event.payload if hasattr(event, "payload") else event
                config_key = payload.get("config_key", "unknown")
                model_path = payload.get("model_path", "")
                final_loss = payload.get("final_loss", 0)
                logger.info(
                    f"[P2P] Training completed: {config_key} "
                    f"(loss={final_loss:.4f}, model={model_path})"
                )
                # Clear from active training
                if hasattr(orchestrator, "_active_training"):
                    orchestrator._active_training.pop(config_key, None)
                # Trigger selfplay allocation refresh
                if hasattr(orchestrator, "selfplay_scheduler"):
                    orchestrator.selfplay_scheduler.on_training_complete(config_key)

                # Bridge to coordination event bus for EvaluationDaemon
                try:
                    from app.coordination.event_router import emit_event
                    from app.coordination.data_events import DataEventType
                    emit_event(DataEventType.TRAINING_COMPLETED, {
                        "config_key": config_key,
                        "model_path": model_path,
                        "final_loss": final_loss,
                        "training_samples": payload.get("training_samples", 0),
                        "training_games": payload.get("training_games", 0),
                        "source": "p2p_bridge",
                    })
                    logger.debug("[P2P] Bridged TRAINING_COMPLETED to coordination bus")
                except Exception as bridge_err:  # noqa: BLE001
                    logger.debug(f"Could not bridge TRAINING_COMPLETED: {bridge_err}")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Error handling training completed event: {e}")

        def handle_task_spawned(event: Any) -> None:
            """Handle task spawn events."""
            try:
                payload = event.payload if hasattr(event, "payload") else event
                job_id = payload.get("job_id", "unknown")
                job_type = payload.get("job_type", "unknown")
                node_id = payload.get("node_id", "unknown")
                logger.debug(f"[P2P] Task spawned: {job_type} {job_id} on {node_id}")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Error handling task spawned event: {e}")

        def handle_task_completed(event: Any) -> None:
            """Handle task completion events."""
            try:
                payload = event.payload if hasattr(event, "payload") else event
                job_id = payload.get("job_id", "unknown")
                job_type = payload.get("job_type", "unknown")
                duration = payload.get("duration", 0)
                logger.debug(f"[P2P] Task completed: {job_type} {job_id} ({duration:.1f}s)")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Error handling task completed event: {e}")

        def handle_task_failed(event: Any) -> None:
            """Handle task failure events."""
            try:
                payload = event.payload if hasattr(event, "payload") else event
                job_id = payload.get("job_id", "unknown")
                job_type = payload.get("job_type", "unknown")
                error = payload.get("error", "unknown error")
                logger.warning(f"[P2P] Task failed: {job_type} {job_id} - {error}")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Error handling task failed event: {e}")

        def handle_data_sync_started(event: Any) -> None:
            """Handle data sync start events."""
            try:
                payload = event.payload if hasattr(event, "payload") else event
                sync_type = payload.get("sync_type", "unknown")
                target_count = payload.get("target_nodes", 0)
                logger.info(f"[P2P] Data sync started: {sync_type} to {target_count} nodes")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Error handling data sync started event: {e}")

        def handle_data_sync_completed(event: Any) -> None:
            """Handle data sync completion events."""
            try:
                payload = event.payload if hasattr(event, "payload") else event
                sync_type = payload.get("sync_type", "unknown")
                duration = payload.get("duration", 0)
                files_synced = payload.get("files_synced", 0)
                logger.info(
                    f"[P2P] Data sync completed: {sync_type} "
                    f"({files_synced} files in {duration:.1f}s)"
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Error handling data sync completed event: {e}")

        def handle_node_unhealthy(event: Any) -> None:
            """Handle NODE_UNHEALTHY events - pause jobs on unhealthy nodes."""
            try:
                payload = event.payload if hasattr(event, "payload") else event
                node_id = payload.get("node_id", "unknown")
                reason = payload.get("reason", "")
                logger.warning(f"[P2P] Node {node_id} unhealthy: {reason}")
                if hasattr(orchestrator, "node_selector") and orchestrator.node_selector:
                    orchestrator.node_selector.mark_node_unhealthy(node_id, reason)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Error handling node unhealthy event: {e}")

        def handle_node_recovered(event: Any) -> None:
            """Handle NODE_RECOVERED events - resume jobs on recovered nodes."""
            try:
                payload = event.payload if hasattr(event, "payload") else event
                node_id = payload.get("node_id", "unknown")
                logger.info(f"[P2P] Node {node_id} recovered")
                if hasattr(orchestrator, "node_selector") and orchestrator.node_selector:
                    orchestrator.node_selector.mark_node_healthy(node_id)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Error handling node recovered event: {e}")

        def handle_cluster_healthy(event: Any) -> None:
            """Handle P2P_CLUSTER_HEALTHY events."""
            try:
                logger.info("[P2P] Cluster is healthy - resuming normal operations")
                orchestrator._cluster_health_degraded = False
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Error handling cluster healthy event: {e}")

        def handle_cluster_unhealthy(event: Any) -> None:
            """Handle P2P_CLUSTER_UNHEALTHY events - pause non-critical operations."""
            try:
                payload = event.payload if hasattr(event, "payload") else event
                reason = payload.get("reason", "")
                alive_nodes = payload.get("alive_nodes", 0)
                logger.warning(
                    f"[P2P] Cluster unhealthy: {reason} (alive_nodes={alive_nodes})"
                )
                orchestrator._cluster_health_degraded = True
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Error handling cluster unhealthy event: {e}")

        # Subscribe to all manager events
        subscribe("TRAINING_STARTED", handle_training_started)
        subscribe("TRAINING_COMPLETED", handle_training_completed)
        subscribe("TASK_SPAWNED", handle_task_spawned)
        subscribe("TASK_COMPLETED", handle_task_completed)
        subscribe("TASK_FAILED", handle_task_failed)
        subscribe("DATA_SYNC_STARTED", handle_data_sync_started)
        subscribe("DATA_SYNC_COMPLETED", handle_data_sync_completed)

        # Subscribe to health events
        subscribe("NODE_UNHEALTHY", handle_node_unhealthy)
        subscribe("NODE_RECOVERED", handle_node_recovered)
        subscribe("P2P_CLUSTER_HEALTHY", handle_cluster_healthy)
        subscribe("P2P_CLUSTER_UNHEALTHY", handle_cluster_unhealthy)

        logger.info("Subscribed to manager lifecycle and health events")
        return True
    except ImportError as e:
        logger.debug(f"Manager events: event_router not available: {e}")
        return False
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Manager events: failed to subscribe: {e}")
        return False
