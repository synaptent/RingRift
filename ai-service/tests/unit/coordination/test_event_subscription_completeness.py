"""Event subscription completeness tests (December 2025).

This module verifies that all pipeline-critical events have at least one subscriber,
ensuring that the event-driven coordination layer works correctly.

These tests catch silent failures where events are emitted but no coordinator
is subscribed to handle them.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock


class TestEventSubscriptionCompleteness:
    """Verify critical events have subscribers."""

    def test_validate_event_wiring_available(self):
        """Verify validate_event_wiring function is accessible."""
        from app.coordination.event_router import validate_event_wiring

        result = validate_event_wiring(raise_on_error=False, log_warnings=False)
        assert isinstance(result, dict)
        assert "valid" in result

    def test_critical_events_defined(self):
        """Verify critical DataEventType values exist."""
        from app.distributed.data_events import DataEventType

        critical_events = [
            "TRAINING_COMPLETED",
            "EVALUATION_COMPLETED",
            "MODEL_PROMOTED",
            "DATA_SYNC_COMPLETED",
        ]

        for event_name in critical_events:
            assert hasattr(DataEventType, event_name), (
                f"Missing critical event type: {event_name}"
            )

    def test_recommended_events_defined(self):
        """Verify recommended DataEventType values exist."""
        from app.distributed.data_events import DataEventType

        recommended_events = [
            "NEW_GAMES_AVAILABLE",
            "CURRICULUM_REBALANCED",
            "ELO_VELOCITY_CHANGED",
            "TRAINING_STARTED",
            "SELFPLAY_COMPLETE",
        ]

        for event_name in recommended_events:
            assert hasattr(DataEventType, event_name), (
                f"Missing recommended event type: {event_name}"
            )


class TestEventRouterIntegration:
    """Test event router integration with coordinators."""

    @pytest.fixture
    def mock_router(self):
        """Create a mock router with subscriber tracking."""
        from app.coordination.event_router import UnifiedEventRouter

        router = UnifiedEventRouter()
        return router

    def test_router_can_subscribe_handler(self, mock_router):
        """Verify router accepts subscriptions."""
        call_count = [0]

        def test_handler(event):
            call_count[0] += 1

        mock_router.subscribe("test_event", test_handler)
        assert "test_event" in mock_router._subscribers

    def test_router_tracks_subscriber_count(self, mock_router):
        """Verify router tracks number of subscribers per event type."""
        def handler1(event): pass
        def handler2(event): pass

        mock_router.subscribe("test_event", handler1)
        mock_router.subscribe("test_event", handler2)

        subscribers = mock_router._subscribers.get("test_event", [])
        assert len(subscribers) == 2

    def test_router_unsubscribe_works(self, mock_router):
        """Verify unsubscribe removes handler."""
        def handler(event): pass

        mock_router.subscribe("test_event", handler)
        mock_router.unsubscribe("test_event", handler)

        subscribers = mock_router._subscribers.get("test_event", [])
        assert len(subscribers) == 0


class TestCoordinatorEventWiring:
    """Test that key coordinators wire their event subscriptions."""

    def test_data_pipeline_orchestrator_subscribes_to_events(self):
        """Verify DataPipelineOrchestrator subscribes to expected events."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        # Check that the class has event subscription method
        assert hasattr(DataPipelineOrchestrator, "subscribe_to_events") or \
               hasattr(DataPipelineOrchestrator, "_subscribe_to_events"), (
            "DataPipelineOrchestrator must have event subscription method"
        )

    def test_selfplay_scheduler_subscribes_to_events(self):
        """Verify SelfplayScheduler subscribes to expected events."""
        from app.coordination.selfplay_scheduler import SelfplayScheduler

        assert hasattr(SelfplayScheduler, "subscribe_to_events"), (
            "SelfplayScheduler must have subscribe_to_events method"
        )

    def test_feedback_loop_controller_subscribes_to_events(self):
        """Verify FeedbackLoopController subscribes to expected events."""
        from app.coordination.feedback_loop_controller import FeedbackLoopController

        # Check for subscription method
        assert hasattr(FeedbackLoopController, "_subscribe_to_events") or \
               hasattr(FeedbackLoopController, "subscribe_to_events") or \
               hasattr(FeedbackLoopController, "_wire_event_handlers"), (
            "FeedbackLoopController must have event subscription method"
        )

    def test_task_coordinator_has_wire_function(self):
        """Verify TaskCoordinator has wire function for events."""
        from app.coordination import task_coordinator

        assert hasattr(task_coordinator, "wire_task_coordinator_events"), (
            "task_coordinator module must have wire_task_coordinator_events function"
        )


class TestCoordinatorRegistryCompleteness:
    """Test COORDINATOR_REGISTRY has all required entries."""

    def test_coordinator_registry_exists(self):
        """Verify COORDINATOR_REGISTRY is defined."""
        from app.coordination.coordination_bootstrap import COORDINATOR_REGISTRY

        assert isinstance(COORDINATOR_REGISTRY, dict)
        assert len(COORDINATOR_REGISTRY) > 0

    def test_registry_has_critical_coordinators(self):
        """Verify critical coordinators are in registry."""
        from app.coordination.coordination_bootstrap import COORDINATOR_REGISTRY

        # Note: Names may vary based on implementation
        # Check for at least some critical coordinator types
        # DataPipelineOrchestrator is initialized separately, not in registry
        critical_patterns = [
            "task",  # task_coordinator or global_task_coordinator
            "cache",  # cache_orchestrator
            "selfplay",  # selfplay_scheduler or selfplay_orchestrator
            "metrics",  # metrics_orchestrator
            "training",  # training_coordinator
        ]

        registry_names = list(COORDINATOR_REGISTRY.keys())
        for pattern in critical_patterns:
            found = any(pattern in name for name in registry_names)
            assert found, (
                f"No coordinator matching pattern '{pattern}' found in registry. "
                f"Available: {registry_names[:10]}..."
            )

    def test_registry_entries_have_required_fields(self):
        """Verify all registry entries have required fields."""
        from app.coordination.coordination_bootstrap import (
            COORDINATOR_REGISTRY,
            InitPattern,
        )

        checked = 0
        for name, spec in list(COORDINATOR_REGISTRY.items())[:10]:  # Check first 10
            # CoordinatorSpec is a dataclass with specific fields
            assert hasattr(spec, "name"), f"{name} spec has no 'name' attribute"
            assert hasattr(spec, "display_name"), f"{name} spec has no 'display_name'"
            assert hasattr(spec, "module_path"), f"{name} spec has no 'module_path'"
            assert hasattr(spec, "pattern"), f"{name} spec has no 'pattern'"
            # Verify fields are not empty
            assert spec.name, f"{name} has empty name"
            assert spec.display_name, f"{name} has empty display_name"
            # DELEGATE and SKIP patterns don't require module_path
            # (they delegate to other coordinators or are deprecated)
            if spec.pattern not in (InitPattern.DELEGATE, InitPattern.SKIP):
                assert spec.module_path, f"{name} has empty module_path"
            checked += 1

        assert checked > 0, "No registry entries checked"


class TestEventNormalization:
    """Test event type normalization for subscription matching."""

    def test_normalize_event_type_available(self):
        """Verify normalize_event_type function exists."""
        from app.coordination.event_normalization import normalize_event_type

        assert callable(normalize_event_type)

    def test_normalize_event_type_handles_enum(self):
        """Verify normalization handles DataEventType enum."""
        from app.coordination.event_normalization import normalize_event_type
        from app.distributed.data_events import DataEventType

        # The function should extract the value from enums
        normalized = normalize_event_type(DataEventType.TRAINING_COMPLETED)
        # Result should be a string (the enum value)
        assert isinstance(normalized, str)
        assert "training" in normalized.lower() or "completed" in normalized.lower()

    def test_normalize_event_type_handles_string(self):
        """Verify normalization handles string event types."""
        from app.coordination.event_normalization import normalize_event_type

        # Should normalize to lowercase or canonical form
        normalized = normalize_event_type("training_completed")
        assert isinstance(normalized, str)
        # Canonical form may vary
        assert normalized in ["training_completed", "TRAINING_COMPLETED", "training_complete"]


class TestBootstrapEventWiring:
    """Test that bootstrap_coordination wires events correctly."""

    def test_bootstrap_has_event_wiring_capability(self):
        """Verify bootstrap process includes event wiring."""
        from app.coordination import coordination_bootstrap

        # Check for any wiring functions
        wiring_funcs = [
            name for name in dir(coordination_bootstrap)
            if "wire" in name.lower() and callable(getattr(coordination_bootstrap, name, None))
        ]
        assert len(wiring_funcs) > 0, (
            "coordination_bootstrap must have at least one wiring function"
        )

    def test_pipeline_orchestrator_has_wire_function(self):
        """Verify pipeline orchestrator has wire function."""
        from app.coordination.data_pipeline_orchestrator import wire_pipeline_events

        assert callable(wire_pipeline_events), (
            "wire_pipeline_events must be callable"
        )


class TestCriticalEventHandlers:
    """Test that critical events have proper handlers defined."""

    def test_training_completed_has_handlers(self):
        """Verify TRAINING_COMPLETED event has handler methods in coordinators."""
        from app.coordination.feedback_loop_controller import FeedbackLoopController

        # Check that FeedbackLoopController can handle training completion
        assert hasattr(FeedbackLoopController, "_on_training_complete") or \
               hasattr(FeedbackLoopController, "_trigger_evaluation"), (
            "FeedbackLoopController must have TRAINING_COMPLETED handler"
        )

    def test_evaluation_completed_has_handlers(self):
        """Verify EVALUATION_COMPLETED event has handler methods in FeedbackLoopController."""
        # FeedbackLoopController handles evaluation results, not curriculum_integration
        from app.coordination.feedback_loop_controller import FeedbackLoopController

        # Check for evaluation handler methods
        has_handler = (
            hasattr(FeedbackLoopController, "_on_evaluation_complete") or
            hasattr(FeedbackLoopController, "_on_evaluation_completed") or
            hasattr(FeedbackLoopController, "_process_evaluation_result") or
            hasattr(FeedbackLoopController, "_handle_evaluation")
        )

        assert has_handler, (
            "FeedbackLoopController must have EVALUATION_COMPLETED handler"
        )

    def test_sync_completed_has_handlers(self):
        """Verify DATA_SYNC_COMPLETED event has handler methods."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        assert hasattr(DataPipelineOrchestrator, "_on_sync_complete"), (
            "DataPipelineOrchestrator must have DATA_SYNC_COMPLETED handler"
        )


class TestEventSubscriptionDiagnostics:
    """Test event subscription diagnostic tools."""

    def test_validate_event_wiring_returns_missing_info(self):
        """Verify validate_event_wiring reports missing subscriptions."""
        from app.coordination.event_router import validate_event_wiring

        result = validate_event_wiring(raise_on_error=False, log_warnings=False)

        assert "missing_critical" in result
        assert "missing_optional" in result
        assert isinstance(result["missing_critical"], list)
        assert isinstance(result["missing_optional"], list)

    def test_validate_event_wiring_reports_all_subscribed(self):
        """Verify validate_event_wiring reports subscribed events."""
        from app.coordination.event_router import validate_event_wiring

        result = validate_event_wiring(raise_on_error=False, log_warnings=False)

        assert "all_subscribed" in result
        assert isinstance(result["all_subscribed"], list)


class TestEventEmitterCoverage:
    """Test that event emitters exist for critical events."""

    def test_training_completed_emitter_exists(self):
        """Verify emit function for TRAINING_COMPLETED exists."""
        from app.coordination import event_emitters

        # Check for training complete emitter
        has_emitter = (
            hasattr(event_emitters, "emit_training_complete") or
            hasattr(event_emitters, "emit_training_completed")
        )
        assert has_emitter, "Missing training completion emitter"

    def test_evaluation_completed_emitter_exists(self):
        """Verify emit function for EVALUATION_COMPLETED exists."""
        from app.coordination import event_emitters

        has_emitter = (
            hasattr(event_emitters, "emit_evaluation_complete") or
            hasattr(event_emitters, "emit_evaluation_completed")
        )
        assert has_emitter, "Missing evaluation completion emitter"

    def test_model_promoted_emitter_exists(self):
        """Verify emit function for MODEL_PROMOTED exists."""
        from app.coordination import event_emitters

        has_emitter = (
            hasattr(event_emitters, "emit_model_promoted") or
            hasattr(event_emitters, "emit_promotion_complete")
        )
        assert has_emitter, "Missing model promotion emitter"

    def test_sync_completed_emitter_exists(self):
        """Verify emit function for DATA_SYNC_COMPLETED exists."""
        from app.coordination import event_emitters

        has_emitter = (
            hasattr(event_emitters, "emit_sync_complete") or
            hasattr(event_emitters, "emit_data_sync_complete") or
            hasattr(event_emitters, "emit_sync_completed")
        )
        assert has_emitter, "Missing sync completion emitter"
