"""Event Management HTTP handlers for P2P orchestrator.

January 2026 - P2P Modularization Phase 5b

This mixin provides HTTP handlers for event subscription visibility
and feedback loop health monitoring.

Must be mixed into a class that provides:
- self.node_id: str
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aiohttp import web

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class EventManagementHandlersMixin:
    """Mixin providing event subscription management HTTP handlers.

    Endpoints:
    - GET /subscriptions - Get event subscription dashboard with feedback loop health
    """

    # Required attributes (provided by orchestrator)
    node_id: str

    async def handle_subscriptions(self, request: web.Request) -> web.Response:
        """GET /subscriptions - Get event subscription dashboard.

        Phase 5 (December 2025): Visibility into feedback loop event wiring.

        Returns list of events and their subscribers to verify the feedback
        loop is properly wired. Critical for debugging dead-end events.

        Returns:
            JSON with event types and their subscriber counts/names
        """
        try:
            from app.coordination.event_router import DataEventType

            subscriptions: dict[str, dict] = {}
            router_info: dict = {"available": False, "type": "unknown"}

            try:
                from app.coordination.event_router import get_router

                router = get_router()
                if router is not None:
                    router_info["available"] = True
                    router_info["type"] = type(router).__name__

                    # Get all subscriptions from router
                    if hasattr(router, '_subscribers'):
                        for event_key, handlers in router._subscribers.items():
                            handler_names = []
                            for handler in handlers:
                                if hasattr(handler, '__name__'):
                                    handler_names.append(handler.__name__)
                                elif hasattr(handler, '__class__'):
                                    handler_names.append(handler.__class__.__name__)
                                else:
                                    handler_names.append(str(type(handler)))

                            subscriptions[event_key] = {
                                "count": len(handlers),
                                "handlers": handler_names[:10],  # Limit to first 10
                            }
            except Exception as e:  # noqa: BLE001
                router_info["error"] = str(e)

            # Define critical events for feedback loop
            critical_events = [
                "hyperparameter_updated",
                "curriculum_advanced",
                "adaptive_params_changed",
                "regression_critical",
                "evaluation_completed",
                "model_promoted",
                "training_complete",
                "selfplay_complete",
            ]

            critical_status: dict[str, dict] = {}
            for event in critical_events:
                if event in subscriptions:
                    critical_status[event] = {
                        "status": "active",
                        "subscribers": subscriptions[event]["count"],
                    }
                else:
                    critical_status[event] = {
                        "status": "missing",
                        "subscribers": 0,
                    }

            missing_count = sum(1 for e in critical_status.values() if e["status"] == "missing")

            return web.json_response({
                "node_id": self.node_id,
                "router": router_info,
                "feedback_loop_health": "healthy" if missing_count == 0 else f"{missing_count} missing",
                "critical_events": critical_status,
                "all_subscriptions": subscriptions,
                "total_event_types": len(subscriptions),
                "phase": "Phase 5 - December 2025",
            })

        except Exception as e:  # noqa: BLE001
            logger.error(f"in handle_subscriptions: {e}")
            return web.json_response({"error": str(e)}, status=500)
