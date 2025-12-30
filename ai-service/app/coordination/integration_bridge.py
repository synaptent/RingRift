"""Integration Bridge - Wires integration modules to coordination event system.

This module connects the 3 standalone integration modules to the unified event router:
1. ModelLifecycleManager (model_lifecycle.py) - Model promotion/rollback lifecycle
2. P2PIntegrationManager (p2p_integration.py) - Cluster coordination
3. PipelineFeedbackController (pipeline_feedback.py) - Closed-loop feedback

The bridge:
- Converts callback-based events to unified router events
- Subscribes integration modules to relevant router events
- Provides single initialization point for bootstrap

Created: December 2025
Purpose: C2 consolidation - wire integration modules to coordination bootstrap
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.coordination.event_router import (
    RouterEvent,
    publish_sync,
    subscribe,
)

if TYPE_CHECKING:
    from app.integration.model_lifecycle import ModelLifecycleManager
    from app.integration.p2p_integration import P2PIntegrationManager
    from app.integration.pipeline_feedback import (
        FeedbackSignalRouter,
        PipelineFeedbackController,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# Event Type Constants (matching data_events.py DataEventType)
# =============================================================================

EVENT_MODEL_REGISTERED = "model_registered"
EVENT_MODEL_PROMOTED = "model_promoted"
EVENT_MODEL_REJECTED = "model_rejected"
EVENT_MODEL_ROLLBACK = "model_rollback"
EVENT_TRAINING_TRIGGERED = "training_triggered"
EVENT_TRAINING_COMPLETED = "training_completed"
EVENT_EVALUATION_COMPLETE = "evaluation_completed"  # Fixed: matches DataEventType.EVALUATION_COMPLETED
EVENT_EVALUATION_SCHEDULED = "evaluation_scheduled"
EVENT_CLUSTER_HEALTH_CHANGED = "cluster_health_changed"
EVENT_SELFPLAY_SCALED = "selfplay_scaled"
EVENT_FEEDBACK_SIGNAL = "feedback_signal"
EVENT_PARITY_VALIDATION_COMPLETE = "parity_validation_completed"  # Fixed: matches DataEventType
EVENT_ELO_UPDATED = "elo_updated"
EVENT_REGISTRY_SYNC_NEEDED = "registry_sync_needed"
EVENT_CURRICULUM_UPDATED = "curriculum_updated"


# =============================================================================
# Model Lifecycle Bridge
# =============================================================================

def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _infer_version_from_model_id(model_id: str | None) -> int | None:
    if not model_id:
        return None
    for token in (":v", "_v", "-v"):
        if token in model_id:
            suffix = model_id.rsplit(token, 1)[-1]
            if suffix.isdigit():
                return int(suffix)
    return None


def _first_present(payload: dict[str, Any], metadata: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
        if key in metadata and metadata[key] is not None:
            return metadata[key]
    return None


def _build_evaluation_result(payload: dict[str, Any]) -> "EvaluationResult" | None:
    from app.integration.model_lifecycle import EvaluationResult

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    model_id = _first_present(payload, metadata, ["model_id", "model", "id"])
    version_raw = _first_present(payload, metadata, ["version", "model_version", "model_ver"])
    version = _coerce_int(version_raw) or _infer_version_from_model_id(model_id)

    if not model_id or version is None:
        return None

    elo = _first_present(payload, metadata, ["elo", "elo_rating", "final_elo", "new_elo"])
    elo_uncertainty = _first_present(payload, metadata, ["elo_uncertainty"])
    games_played = _coerce_int(_first_present(payload, metadata, ["games_played", "games", "games_generated"])) or 0
    win_rate = _first_present(payload, metadata, ["win_rate"])
    draw_rate = _first_present(payload, metadata, ["draw_rate"])
    value_mse = _first_present(payload, metadata, ["value_mse"])
    policy_accuracy = _first_present(payload, metadata, ["policy_accuracy"])
    elo_vs_production = _first_present(payload, metadata, ["elo_vs_production"])
    win_rate_vs_production = _first_present(payload, metadata, ["win_rate_vs_production"])
    games_vs_production = _coerce_int(_first_present(payload, metadata, ["games_vs_production"])) or 0

    return EvaluationResult(
        model_id=str(model_id),
        version=version,
        elo=float(elo) if elo is not None else None,
        elo_uncertainty=float(elo_uncertainty) if elo_uncertainty is not None else None,
        games_played=games_played,
        win_rate=float(win_rate) if win_rate is not None else None,
        draw_rate=float(draw_rate) if draw_rate is not None else None,
        value_mse=float(value_mse) if value_mse is not None else None,
        policy_accuracy=float(policy_accuracy) if policy_accuracy is not None else None,
        elo_vs_production=float(elo_vs_production) if elo_vs_production is not None else None,
        win_rate_vs_production=float(win_rate_vs_production) if win_rate_vs_production is not None else None,
        games_vs_production=games_vs_production,
    )


def _task_error_callback(task: asyncio.Task) -> None:
    """Handle errors from fire-and-forget tasks.

    Dec 2025: Added to fix CRITICAL issue where task exceptions were silently lost.
    """
    try:
        exc = task.exception()
        if exc is not None:
            logger.error(f"[IntegrationBridge] Background task failed: {exc}", exc_info=exc)
    except (asyncio.CancelledError, asyncio.InvalidStateError):
        pass  # Task was cancelled or still pending


def _run_coroutine(coro: Any) -> None:
    """Run coroutine in fire-and-forget mode with error handling.

    Dec 2025: Fixed to add error callback so task exceptions are logged
    rather than silently lost.
    """
    if coro is None:
        return
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(coro)
        # Add error callback to catch and log task exceptions
        task.add_done_callback(_task_error_callback)
    except RuntimeError:
        # No running loop - create new one (blocking)
        try:
            asyncio.run(coro)
        except Exception as e:
            logger.error(f"[IntegrationBridge] Failed to run coroutine: {e}")


def wire_model_lifecycle_events(manager: ModelLifecycleManager) -> None:
    """Wire ModelLifecycleManager callbacks to event router.

    Converts callback-based events from ModelLifecycleManager to
    unified router events that other components can subscribe to.

    Args:
        manager: The ModelLifecycleManager instance to wire
    """
    logger.info("[IntegrationBridge] Wiring ModelLifecycleManager to event router")

    # Register callbacks that publish to router
    def on_model_registered(**event_data: Any) -> None:
        publish_sync(EVENT_MODEL_REGISTERED, dict(event_data), source="model_lifecycle")

    def on_model_promoted(**event_data: Any) -> None:
        publish_sync(EVENT_MODEL_PROMOTED, dict(event_data), source="model_lifecycle")

    def on_training_triggered(**event_data: Any) -> None:
        publish_sync(EVENT_TRAINING_TRIGGERED, dict(event_data), source="model_lifecycle")

    def on_model_rollback(**event_data: Any) -> None:
        publish_sync(EVENT_MODEL_ROLLBACK, dict(event_data), source="model_lifecycle")

    # Register with manager's callback system
    manager.register_callback("model_registered", on_model_registered)
    manager.register_callback("model_promoted", on_model_promoted)
    manager.register_callback("training_triggered", on_training_triggered)
    manager.register_callback("model_rollback", on_model_rollback)

    # Subscribe to evaluation complete events for promotion decisions
    def on_evaluation_complete(event: RouterEvent) -> None:
        if not isinstance(event.payload, dict):
            logger.debug("[IntegrationBridge] Evaluation payload missing or invalid")
            return
        if event.payload.get("success") is False:
            logger.info("[IntegrationBridge] Evaluation reported failure; skipping submission")
            return

        result = _build_evaluation_result(event.payload)
        if not result:
            logger.debug("[IntegrationBridge] Evaluation payload missing model_id/version")
            return

        try:
            _run_coroutine(manager.submit_evaluation(result.model_id, result.version, result))
        except Exception as e:
            logger.error(f"[IntegrationBridge] Error submitting evaluation: {e}")

    subscribe(EVENT_EVALUATION_COMPLETE, on_evaluation_complete)

    logger.info("[IntegrationBridge] ModelLifecycleManager wired successfully")


# =============================================================================
# P2P Integration Bridge
# =============================================================================

def wire_p2p_integration_events(manager: P2PIntegrationManager) -> None:
    """Wire P2PIntegrationManager callbacks to event router.

    Converts P2P cluster events to unified router events.

    Args:
        manager: The P2PIntegrationManager instance to wire
    """
    logger.info("[IntegrationBridge] Wiring P2PIntegrationManager to event router")

    # Register callbacks that publish to router
    def on_cluster_unhealthy(**event_data: Any) -> None:
        publish_sync(
            EVENT_CLUSTER_HEALTH_CHANGED,
            {"healthy": False, **event_data},
            source="p2p_integration",
        )

    def on_cluster_healthy(**event_data: Any) -> None:
        publish_sync(
            EVENT_CLUSTER_HEALTH_CHANGED,
            {"healthy": True, **event_data},
            source="p2p_integration",
        )

    def on_selfplay_scaled(**event_data: Any) -> None:
        publish_sync(EVENT_SELFPLAY_SCALED, dict(event_data), source="p2p_integration")

    def on_evaluation_complete(**event_data: Any) -> None:
        publish_sync(EVENT_EVALUATION_COMPLETE, dict(event_data), source="p2p_integration")

    # Register with manager's callback system
    manager.register_callback("cluster_unhealthy", on_cluster_unhealthy)
    manager.register_callback("cluster_healthy", on_cluster_healthy)
    manager.register_callback("selfplay_scaled", on_selfplay_scaled)
    manager.register_callback("evaluation_complete", on_evaluation_complete)

    # Subscribe to model promoted events to trigger P2P sync
    def on_model_promoted(event: RouterEvent) -> None:
        if not isinstance(event.payload, dict):
            return
        metadata = event.payload.get("metadata") if isinstance(event.payload.get("metadata"), dict) else {}
        model_id = _first_present(event.payload, metadata, ["model_id", "model", "id"])
        model_path = _first_present(event.payload, metadata, ["model_path", "path"])
        promotion_type = _first_present(event.payload, metadata, ["promotion_type", "stage", "tier"])
        if promotion_type and str(promotion_type).lower() not in {"production", "champion"}:
            return
        if not model_id or not model_path:
            logger.debug("[IntegrationBridge] Model promotion missing model_id/model_path; skipping sync")
            return

        async def _sync() -> None:
            try:
                await manager.sync_model_to_cluster(str(model_id), Path(str(model_path)))
            except Exception as e:
                logger.error(f"[IntegrationBridge] Error syncing model to cluster: {e}")

        _run_coroutine(_sync())

    # Subscribe to training triggered events
    def on_training_triggered(event: RouterEvent) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}

        async def _start_training() -> None:
            try:
                if payload and hasattr(manager, "training"):
                    await manager.training.start_training(payload)
                else:
                    await manager.trigger_training()
            except Exception as e:
                logger.error(f"[IntegrationBridge] Error starting cluster training: {e}")

        _run_coroutine(_start_training())

    subscribe(EVENT_MODEL_PROMOTED, on_model_promoted)
    subscribe(EVENT_TRAINING_TRIGGERED, on_training_triggered)

    logger.info("[IntegrationBridge] P2PIntegrationManager wired successfully")


# =============================================================================
# Pipeline Feedback Bridge
# =============================================================================

def wire_pipeline_feedback_events(
    controller: PipelineFeedbackController,
    router: FeedbackSignalRouter | None = None,
) -> None:
    """Wire PipelineFeedbackController to event router.

    Converts feedback signals to unified router events.

    Args:
        controller: The PipelineFeedbackController instance
        router: Optional FeedbackSignalRouter for signal routing
    """
    logger.info("[IntegrationBridge] Wiring PipelineFeedbackController to event router")

    # Wrap the controller's signal emission to also publish to router
    original_emit = getattr(controller, "_emit_signal", None)
    if original_emit:
        def wrapped_emit(signal: Any) -> None:
            # Call original
            original_emit(signal)
            # Publish to router
            publish_sync(
                EVENT_FEEDBACK_SIGNAL,
                {
                    "action": signal.action.value if hasattr(signal.action, "value") else str(signal.action),
                    "reason": signal.reason,
                    "metadata": signal.metadata,
                    "severity": signal.severity.value if hasattr(signal.severity, "value") else str(signal.severity),
                },
                source="pipeline_feedback",
            )

        controller._emit_signal = wrapped_emit  # type: ignore[attr-defined]

    # Subscribe to stage completion events
    # Note: on_stage_complete is async, so we schedule it on the event loop
    def _schedule_async(coro):
        """Schedule an async coroutine from a sync callback."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            # No running loop - run directly (blocking)
            asyncio.run(coro)

    def on_training_completed(event: RouterEvent) -> None:
        if event.payload:
            try:
                _schedule_async(controller.on_stage_complete("training", event.payload))
            except Exception as e:
                logger.error(f"[IntegrationBridge] Error handling training complete: {e}")

    def on_evaluation_complete(event: RouterEvent) -> None:
        if event.payload:
            try:
                _schedule_async(controller.on_stage_complete("evaluation", event.payload))
            except Exception as e:
                logger.error(f"[IntegrationBridge] Error handling evaluation complete: {e}")

    def on_parity_complete(event: RouterEvent) -> None:
        if event.payload:
            try:
                _schedule_async(controller.on_stage_complete("parity_validation", event.payload))
            except Exception as e:
                logger.error(f"[IntegrationBridge] Error handling parity complete: {e}")

    subscribe(EVENT_TRAINING_COMPLETED, on_training_completed)
    subscribe(EVENT_EVALUATION_COMPLETE, on_evaluation_complete)
    subscribe(EVENT_PARITY_VALIDATION_COMPLETE, on_parity_complete)

    logger.info("[IntegrationBridge] PipelineFeedbackController wired successfully")


# =============================================================================
# Sync Manager Integration (December 2025)
# =============================================================================

def wire_sync_manager_events() -> None:
    """Wire sync managers to respond to model and ELO events.

    This enables automatic sync triggers when:
    - A model is promoted (triggers registry sync)
    - ELO ratings are updated (triggers ELO sync)
    """
    logger.info("[IntegrationBridge] Wiring sync managers to event router")

    # Wire model_promoted → registry sync
    def on_model_promoted_sync_registry(event: RouterEvent) -> None:
        """Handle MODEL_PROMOTED events by triggering registry sync.

        Synchronizes model metadata to RegistrySyncManager when a model is
        promoted. This ensures all cluster nodes have up-to-date model
        registry information for discovery and loading.

        Args:
            event: RouterEvent with payload containing 'model_id' or 'model'

        Side Effects:
            Triggers async RegistrySyncManager.sync_model_metadata() call
        """
        if not isinstance(event.payload, dict):
            return

        model_id = event.payload.get("model_id") or event.payload.get("model")
        if not model_id:
            return

        async def _sync_registry() -> None:
            try:
                from app.training.registry_sync_manager import get_registry_sync_manager

                sync_mgr = get_registry_sync_manager()
                if sync_mgr:
                    await sync_mgr.sync_model_metadata(str(model_id))
                    logger.debug(f"[IntegrationBridge] Registry sync triggered for {model_id}")
            except ImportError:
                pass  # Module not available
            except Exception as e:
                logger.warning(f"[IntegrationBridge] Registry sync error: {e}")

        _run_coroutine(_sync_registry())

    # Wire elo_updated → ELO sync
    def on_elo_updated_sync(event: RouterEvent) -> None:
        """Handle ELO_UPDATED events by triggering cluster-wide Elo sync.

        Synchronizes Elo ratings database across cluster nodes when ratings
        change. This ensures consistent Elo data for selfplay priority
        allocation and model evaluation decisions.

        Args:
            event: RouterEvent with payload containing 'config' key

        Side Effects:
            Triggers async EloSyncManager.sync_with_cluster() call
        """
        if not isinstance(event.payload, dict):
            return

        config = event.payload.get("config")
        if not config:
            return

        async def _sync_elo() -> None:
            try:
                from app.tournament.elo_sync_manager import get_elo_sync_manager

                sync_mgr = get_elo_sync_manager()
                if sync_mgr:
                    await sync_mgr.sync_with_cluster()
                    logger.debug(f"[IntegrationBridge] ELO sync triggered for {config}")
            except ImportError:
                pass  # Module not available
            except Exception as e:
                logger.warning(f"[IntegrationBridge] ELO sync error: {e}")

        _run_coroutine(_sync_elo())

    subscribe(EVENT_MODEL_PROMOTED, on_model_promoted_sync_registry)
    subscribe(EVENT_ELO_UPDATED, on_elo_updated_sync)

    logger.info("[IntegrationBridge] Sync managers wired successfully")


# =============================================================================
# Evaluation Curriculum Bridge (December 2025)
# =============================================================================


def wire_evaluation_curriculum_bridge() -> bool:
    """Wire evaluation results to curriculum weight adjustments.

    Connects evaluation completion events to the EvaluationCurriculumBridge,
    which adjusts selfplay config weights based on performance trends.

    Returns:
        True if wiring successful, False otherwise
    """
    try:
        from app.integration.evaluation_curriculum_bridge import (
            EvaluationCurriculumBridge,
        )

        bridge = EvaluationCurriculumBridge()

        def on_evaluation_complete(event: RouterEvent) -> None:
            """Handle EVALUATION_COMPLETED events to update curriculum weights.

            Processes evaluation results (Elo, win_rate) and feeds them to
            EvaluationCurriculumBridge, which adjusts selfplay priority weights
            based on performance trends. Configs with stagnant Elo receive
            increased selfplay allocation.

            Args:
                event: RouterEvent with payload containing:
                    - config_key: Board/player config identifier
                    - metrics: Dict with optional 'elo' and 'win_rate' values

            Side Effects:
                - Updates curriculum weight tracking
                - May emit CURRICULUM_WEIGHTS_UPDATED event
            """
            if not isinstance(event.payload, dict):
                return

            config_key = event.payload.get("config_key")
            metrics = event.payload.get("metrics", {})
            elo = metrics.get("elo")
            win_rate = metrics.get("win_rate")

            if config_key and (elo is not None or win_rate is not None):
                # Update curriculum with evaluation results
                bridge.add_evaluation_result(
                    config_key=config_key,
                    elo=elo,
                    win_rate=win_rate,
                )
                # Emit curriculum update event
                curriculum_weights = bridge.get_curriculum_weights()
                publish_sync(RouterEvent(
                    event_type=EVENT_CURRICULUM_UPDATED,
                    payload={
                        "config_key": config_key,
                        "weights": curriculum_weights,
                        "trigger": "evaluation_complete",
                    },
                    source="integration_bridge",
                ))

        subscribe(EVENT_EVALUATION_COMPLETE, on_evaluation_complete)
        logger.info("[IntegrationBridge] Evaluation curriculum bridge wired successfully")
        return True

    except ImportError as e:
        logger.debug(f"[IntegrationBridge] EvaluationCurriculumBridge not available: {e}")
        return False
    except Exception as e:
        logger.error(f"[IntegrationBridge] Error wiring evaluation bridge: {e}")
        return False


# =============================================================================
# Unified Wiring Entry Point
# =============================================================================

_integration_wired = False


async def wire_all_integrations() -> dict[str, bool]:
    """Wire all integration modules to the event router.

    This is called by coordination_bootstrap.py during initialization.

    Returns:
        Dict of component name to success status
    """
    global _integration_wired

    if _integration_wired:
        logger.debug("[IntegrationBridge] Already wired, skipping")
        return {"already_wired": True}

    results: dict[str, bool] = {}

    # Wire ModelLifecycleManager
    try:
        from app.integration.model_lifecycle import create_lifecycle_manager

        manager = await create_lifecycle_manager()
        wire_model_lifecycle_events(manager)
        results["model_lifecycle"] = True
    except ImportError as e:
        logger.debug(f"[IntegrationBridge] ModelLifecycleManager not available: {e}")
        results["model_lifecycle"] = False
    except Exception as e:
        logger.error(f"[IntegrationBridge] Error wiring ModelLifecycleManager: {e}")
        results["model_lifecycle"] = False

    # Wire P2PIntegrationManager
    try:
        from app.integration.p2p_integration import connect_to_cluster

        manager = await connect_to_cluster()
        if manager:
            wire_p2p_integration_events(manager)
            results["p2p_integration"] = True
        else:
            results["p2p_integration"] = False
    except ImportError as e:
        logger.debug(f"[IntegrationBridge] P2PIntegrationManager not available: {e}")
        results["p2p_integration"] = False
    except Exception as e:
        logger.error(f"[IntegrationBridge] Error wiring P2PIntegrationManager: {e}")
        results["p2p_integration"] = False

    # Wire PipelineFeedbackController
    try:
        from app.integration.pipeline_feedback import (
            create_feedback_controller,
            create_feedback_router,
        )
        from app.utils.paths import AI_SERVICE_ROOT

        controller = create_feedback_controller(AI_SERVICE_ROOT)
        router = create_feedback_router()
        wire_pipeline_feedback_events(controller, router)
        results["pipeline_feedback"] = True
    except ImportError as e:
        logger.debug(f"[IntegrationBridge] PipelineFeedbackController not available: {e}")
        results["pipeline_feedback"] = False
    except Exception as e:
        logger.error(f"[IntegrationBridge] Error wiring PipelineFeedbackController: {e}")
        results["pipeline_feedback"] = False

    # Wire sync managers (December 2025)
    try:
        wire_sync_manager_events()
        results["sync_managers"] = True
    except Exception as e:
        logger.error(f"[IntegrationBridge] Error wiring sync managers: {e}")
        results["sync_managers"] = False

    # Wire evaluation curriculum bridge (December 2025)
    try:
        results["evaluation_curriculum"] = wire_evaluation_curriculum_bridge()
    except Exception as e:
        logger.error(f"[IntegrationBridge] Error wiring evaluation curriculum: {e}")
        results["evaluation_curriculum"] = False

    _integration_wired = True
    logger.info(f"[IntegrationBridge] Integration wiring complete: {results}")

    return results


def wire_all_integrations_sync() -> dict[str, bool]:
    """Synchronous wrapper for wire_all_integrations."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
        # We're in an async context, schedule it
        future = asyncio.ensure_future(wire_all_integrations())
        return {"scheduled": True}
    except RuntimeError:
        # No loop running, create one
        return asyncio.run(wire_all_integrations())


def reset_integration_wiring() -> None:
    """Reset wiring state (for testing)."""
    global _integration_wired
    _integration_wired = False


# =============================================================================
# Integration Health Verification (December 2025)
# =============================================================================

_health_check_subscriptions: dict[str, bool] = {}
_last_event_time: float = 0.0
_events_processed: int = 0
_errors_count: int = 0


def health_check() -> "HealthCheckResult":
    """Return health status for DaemonManager integration.

    Returns:
        HealthCheckResult with integration bridge health status
    """
    from app.coordination.contracts import CoordinatorStatus, HealthCheckResult

    if not _integration_wired:
        return HealthCheckResult(
            healthy=False,
            status=CoordinatorStatus.STOPPED,
            message="Integration bridge not wired",
            details={
                "wired": False,
                "events_processed": _events_processed,
                "errors_count": _errors_count,
            },
        )

    # Check error rate
    total_ops = max(_events_processed, 1)
    error_rate = _errors_count / total_ops

    if error_rate > 0.5:
        return HealthCheckResult(
            healthy=False,
            status=CoordinatorStatus.ERROR,
            message=f"High error rate: {error_rate:.1%}",
            details={
                "wired": True,
                "events_processed": _events_processed,
                "errors_count": _errors_count,
                "error_rate": error_rate,
            },
        )

    if error_rate > 0.2:
        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.DEGRADED,
            message=f"Elevated error rate: {error_rate:.1%}",
            details={
                "wired": True,
                "events_processed": _events_processed,
                "errors_count": _errors_count,
                "error_rate": error_rate,
            },
        )

    return HealthCheckResult(
        healthy=True,
        status=CoordinatorStatus.RUNNING,
        message="Integration bridge operating normally",
        details={
            "wired": True,
            "events_processed": _events_processed,
            "errors_count": _errors_count,
            "last_event_time": _last_event_time,
        },
    )


def _register_health_subscription(event_type: str) -> None:
    """Register that an event was received for health tracking."""
    _health_check_subscriptions[event_type] = True


async def verify_integration_health(
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    """Verify that integration event wiring is functional.

    Emits test events and verifies they propagate through the router
    and reach subscribed handlers.

    Args:
        timeout_seconds: How long to wait for event propagation

    Returns:
        Dictionary with health status and event verification results
    """
    global _health_check_subscriptions
    _health_check_subscriptions.clear()

    results: dict[str, Any] = {
        "wired": _integration_wired,
        "events_tested": [],
        "events_received": [],
        "health_ok": False,
        "details": {},
    }

    if not _integration_wired:
        results["details"]["error"] = "Integration not wired - call wire_all_integrations first"
        return results

    # Subscribe to test events
    test_events = [
        EVENT_MODEL_REGISTERED,
        EVENT_TRAINING_COMPLETED,
        EVENT_EVALUATION_COMPLETE,
        EVENT_FEEDBACK_SIGNAL,
    ]

    for event_type in test_events:
        try:
            subscribe(event_type, lambda e, et=event_type: _register_health_subscription(et))
        except Exception as e:
            results["details"][event_type] = f"subscription_failed: {e}"

    # Emit test events
    for event_type in test_events:
        results["events_tested"].append(event_type)
        try:
            test_event = RouterEvent(
                event_type=event_type,
                payload={"_health_check": True, "test_event": event_type},
                source="integration_bridge_health_check",
            )
            publish_sync(test_event)
        except Exception as e:
            results["details"][event_type] = f"publish_failed: {e}"

    # Wait briefly for async propagation
    await asyncio.sleep(min(0.5, timeout_seconds))

    # Check which events were received
    for event_type in test_events:
        if _health_check_subscriptions.get(event_type):
            results["events_received"].append(event_type)
            results["details"][event_type] = "ok"
        else:
            results["details"][event_type] = "not_received"

    # Determine overall health (75% threshold)
    received_count = len(results["events_received"])
    tested_count = len(results["events_tested"])
    results["health_ok"] = received_count >= tested_count * 0.75

    logger.info(
        f"[IntegrationBridge] Health check: {received_count}/{tested_count} events verified"
    )

    return results


def verify_integration_health_sync(timeout_seconds: float = 5.0) -> dict[str, Any]:
    """Synchronous wrapper for verify_integration_health."""
    try:
        asyncio.get_running_loop()
        return {"scheduled": True, "async_context": True}
    except RuntimeError:
        return asyncio.run(verify_integration_health(timeout_seconds))


def get_wiring_status() -> dict[str, Any]:
    """Get current integration wiring status.

    Returns:
        Dictionary with wiring status and component details
    """
    return {
        "wired": _integration_wired,
        "event_types_registered": [
            EVENT_MODEL_REGISTERED,
            EVENT_MODEL_PROMOTED,
            EVENT_MODEL_REJECTED,
            EVENT_MODEL_ROLLBACK,
            EVENT_TRAINING_TRIGGERED,
            EVENT_TRAINING_COMPLETED,
            EVENT_EVALUATION_COMPLETE,
            EVENT_EVALUATION_SCHEDULED,
            EVENT_CLUSTER_HEALTH_CHANGED,
            EVENT_SELFPLAY_SCALED,
            EVENT_FEEDBACK_SIGNAL,
            EVENT_PARITY_VALIDATION_COMPLETE,
            EVENT_ELO_UPDATED,
            EVENT_REGISTRY_SYNC_NEEDED,
        ],
        "components": [
            "model_lifecycle",
            "p2p_integration",
            "pipeline_feedback",
            "sync_managers",
        ],
    }


__all__ = [
    # Main wiring functions
    "wire_all_integrations",
    "wire_all_integrations_sync",
    "wire_model_lifecycle_events",
    "wire_p2p_integration_events",
    "wire_pipeline_feedback_events",
    "wire_sync_manager_events",
    # Health and status
    "health_check",
    "verify_integration_health",
    "verify_integration_health_sync",
    "get_wiring_status",
    "reset_integration_wiring",
    # Event constants
    "EVENT_MODEL_REGISTERED",
    "EVENT_MODEL_PROMOTED",
    "EVENT_MODEL_REJECTED",
    "EVENT_MODEL_ROLLBACK",
    "EVENT_TRAINING_TRIGGERED",
    "EVENT_TRAINING_COMPLETED",
    "EVENT_EVALUATION_COMPLETE",
    "EVENT_EVALUATION_SCHEDULED",
    "EVENT_CLUSTER_HEALTH_CHANGED",
    "EVENT_SELFPLAY_SCALED",
    "EVENT_FEEDBACK_SIGNAL",
    "EVENT_PARITY_VALIDATION_COMPLETE",
    "EVENT_ELO_UPDATED",
    "EVENT_REGISTRY_SYNC_NEEDED",
]
