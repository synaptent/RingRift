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
EVENT_EVALUATION_COMPLETE = "evaluation_complete"
EVENT_EVALUATION_SCHEDULED = "evaluation_scheduled"
EVENT_CLUSTER_HEALTH_CHANGED = "cluster_health_changed"
EVENT_SELFPLAY_SCALED = "selfplay_scaled"
EVENT_FEEDBACK_SIGNAL = "feedback_signal"
EVENT_PARITY_VALIDATION_COMPLETE = "parity_validation_complete"


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


def _run_coroutine(coro: Any) -> None:
    if coro is None:
        return
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        asyncio.run(coro)


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
    def on_training_completed(event: RouterEvent) -> None:
        if event.payload:
            try:
                controller.on_stage_complete("training", event.payload)
            except Exception as e:
                logger.error(f"[IntegrationBridge] Error handling training complete: {e}")

    def on_evaluation_complete(event: RouterEvent) -> None:
        if event.payload:
            try:
                controller.on_stage_complete("evaluation", event.payload)
            except Exception as e:
                logger.error(f"[IntegrationBridge] Error handling evaluation complete: {e}")

    def on_parity_complete(event: RouterEvent) -> None:
        if event.payload:
            try:
                controller.on_stage_complete("parity_validation", event.payload)
            except Exception as e:
                logger.error(f"[IntegrationBridge] Error handling parity complete: {e}")

    subscribe(EVENT_TRAINING_COMPLETED, on_training_completed)
    subscribe(EVENT_EVALUATION_COMPLETE, on_evaluation_complete)
    subscribe(EVENT_PARITY_VALIDATION_COMPLETE, on_parity_complete)

    logger.info("[IntegrationBridge] PipelineFeedbackController wired successfully")


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

        controller = create_feedback_controller()
        router = create_feedback_router()
        wire_pipeline_feedback_events(controller, router)
        results["pipeline_feedback"] = True
    except ImportError as e:
        logger.debug(f"[IntegrationBridge] PipelineFeedbackController not available: {e}")
        results["pipeline_feedback"] = False
    except Exception as e:
        logger.error(f"[IntegrationBridge] Error wiring PipelineFeedbackController: {e}")
        results["pipeline_feedback"] = False

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


__all__ = [
    "wire_all_integrations",
    "wire_all_integrations_sync",
    "wire_model_lifecycle_events",
    "wire_p2p_integration_events",
    "wire_pipeline_feedback_events",
    "reset_integration_wiring",
]
