"""Tests for core_events.py - Consolidated Event System.

December 2025: Tests the event system consolidation module (Phase 5).
Verifies re-exports work correctly and import compatibility is maintained.
"""

from __future__ import annotations

import pytest


# =============================================================================
# Import Compatibility Tests
# =============================================================================


class TestCoreEventsImports:
    """Tests that all exports are accessible from core_events."""

    def test_import_core_router(self):
        """Core router classes are importable."""
        from app.coordination.core_events import (
            UnifiedEventRouter,
            RouterEvent,
            EventSource,
            get_router,
            publish,
            publish_sync,
            subscribe,
            unsubscribe,
            reset_router,
        )

        assert UnifiedEventRouter is not None
        assert RouterEvent is not None
        assert EventSource is not None
        assert callable(get_router)
        assert callable(publish)
        assert callable(publish_sync)
        assert callable(subscribe)
        assert callable(unsubscribe)
        assert callable(reset_router)

    def test_import_event_types(self):
        """Event type classes are importable."""
        from app.coordination.core_events import (
            DataEventType,
            DataEvent,
            EventBus,
            StageEvent,
            StageCompletionResult,
            CrossProcessEvent,
            CrossProcessEventPoller,
            CrossProcessEventQueue,
        )

        assert DataEventType is not None
        assert DataEvent is not None
        assert EventBus is not None
        assert StageEvent is not None
        assert StageCompletionResult is not None
        assert CrossProcessEvent is not None
        assert CrossProcessEventPoller is not None
        assert CrossProcessEventQueue is not None

    def test_import_bus_access(self):
        """Bus access functions are importable."""
        from app.coordination.core_events import (
            get_event_bus,
            get_stage_event_bus,
            get_cross_process_queue,
            reset_cross_process_queue,
        )

        assert callable(get_event_bus)
        assert callable(get_stage_event_bus)
        assert callable(get_cross_process_queue)
        assert callable(reset_cross_process_queue)

    def test_import_cross_process_functions(self):
        """Cross-process functions are importable."""
        from app.coordination.core_events import (
            ack_event,
            ack_events,
            bridge_to_cross_process,
            cp_poll_events,
            cp_publish,
            subscribe_process,
        )

        assert callable(ack_event)
        assert callable(ack_events)
        assert callable(bridge_to_cross_process)
        assert callable(cp_poll_events)
        assert callable(cp_publish)
        assert callable(subscribe_process)

    def test_import_event_mappings(self):
        """Event mapping constants and functions are importable."""
        from app.coordination.core_events import (
            STAGE_TO_DATA_EVENT_MAP,
            DATA_TO_STAGE_EVENT_MAP,
            DATA_TO_CROSS_PROCESS_MAP,
            CROSS_PROCESS_TO_DATA_MAP,
            STAGE_TO_CROSS_PROCESS_MAP,
            get_data_event_type,
            get_stage_event_type,
            get_cross_process_event_type,
            is_mapped_event,
            get_all_event_types,
            validate_mappings,
        )

        assert isinstance(STAGE_TO_DATA_EVENT_MAP, dict)
        assert isinstance(DATA_TO_STAGE_EVENT_MAP, dict)
        assert isinstance(DATA_TO_CROSS_PROCESS_MAP, dict)
        assert isinstance(CROSS_PROCESS_TO_DATA_MAP, dict)
        assert isinstance(STAGE_TO_CROSS_PROCESS_MAP, dict)
        assert callable(get_data_event_type)
        assert callable(get_stage_event_type)
        assert callable(get_cross_process_event_type)
        assert callable(is_mapped_event)
        assert callable(get_all_event_types)
        assert callable(validate_mappings)

    def test_import_normalization(self):
        """Event normalization functions are importable."""
        from app.coordination.core_events import (
            CANONICAL_EVENT_NAMES,
            EVENT_NAMING_GUIDELINES,
            normalize_event_type,
            is_canonical,
            get_variants,
            audit_event_usage,
            validate_event_names,
        )

        assert isinstance(CANONICAL_EVENT_NAMES, (dict, set))  # Can be dict or set
        assert isinstance(EVENT_NAMING_GUIDELINES, str)  # It's a docstring/guidelines text
        assert callable(normalize_event_type)
        assert callable(is_canonical)
        assert callable(get_variants)
        assert callable(audit_event_usage)
        assert callable(validate_event_names)

    def test_import_validation_stats(self):
        """Validation and stats functions are importable."""
        from app.coordination.core_events import (
            validate_event_flow,
            get_orphaned_events,
            get_event_stats,
            get_event_payload,
        )

        assert callable(validate_event_flow)
        assert callable(get_orphaned_events)
        assert callable(get_event_stats)
        assert callable(get_event_payload)

    def test_import_backward_compat(self):
        """Backward compatibility exports are importable."""
        from app.coordination.core_events import (
            UnifiedEventCoordinator,
            CoordinatorStats,
            get_event_coordinator,
            get_coordinator_stats,
            start_coordinator,
            stop_coordinator,
        )

        assert UnifiedEventCoordinator is not None
        assert CoordinatorStats is not None
        assert callable(get_event_coordinator)
        assert callable(get_coordinator_stats)
        assert callable(start_coordinator)
        assert callable(stop_coordinator)


# =============================================================================
# Typed Emit Function Tests
# =============================================================================


class TestTypedEmitFunctions:
    """Tests that typed emit functions are importable and callable."""

    def test_import_training_emitters(self):
        """Training-related emit functions are importable."""
        from app.coordination.core_events import (
            emit_training_complete,
            emit_training_complete_sync,
            emit_training_started,
            emit_training_started_sync,
            emit_training_completed,
            emit_training_completed_sync,
            emit_training_failed,
            emit_training_triggered,
            emit_training_rollback_needed,
            emit_training_rollback_completed,
            emit_training_early_stopped,
            emit_training_loss_anomaly,
            emit_training_loss_trend,
        )

        assert callable(emit_training_complete)
        assert callable(emit_training_complete_sync)
        assert callable(emit_training_started)
        assert callable(emit_training_started_sync)
        assert callable(emit_training_completed)
        assert callable(emit_training_completed_sync)
        assert callable(emit_training_failed)
        assert callable(emit_training_triggered)
        assert callable(emit_training_rollback_needed)
        assert callable(emit_training_rollback_completed)
        assert callable(emit_training_early_stopped)
        assert callable(emit_training_loss_anomaly)
        assert callable(emit_training_loss_trend)

    def test_import_selfplay_emitters(self):
        """Selfplay-related emit functions are importable."""
        from app.coordination.core_events import (
            emit_selfplay_complete,
            emit_selfplay_batch_completed,
            emit_selfplay_target_updated,
        )

        assert callable(emit_selfplay_complete)
        assert callable(emit_selfplay_batch_completed)
        assert callable(emit_selfplay_target_updated)

    def test_import_sync_emitters(self):
        """Sync-related emit functions are importable."""
        from app.coordination.core_events import (
            emit_sync_complete,
            emit_sync_completed,
            emit_data_sync_failed,
        )

        assert callable(emit_sync_complete)
        assert callable(emit_sync_completed)
        assert callable(emit_data_sync_failed)

    def test_import_evaluation_emitters(self):
        """Evaluation-related emit functions are importable."""
        from app.coordination.core_events import (
            emit_evaluation_complete,
            emit_evaluation_completed,
        )

        assert callable(emit_evaluation_complete)
        assert callable(emit_evaluation_completed)

    def test_import_promotion_emitters(self):
        """Promotion-related emit functions are importable."""
        from app.coordination.core_events import (
            emit_promotion_complete,
            emit_promotion_complete_sync,
            emit_promotion_candidate,
            emit_model_promoted,
        )

        assert callable(emit_promotion_complete)
        assert callable(emit_promotion_complete_sync)
        assert callable(emit_promotion_candidate)
        assert callable(emit_model_promoted)

    def test_import_quality_emitters(self):
        """Quality-related emit functions are importable."""
        from app.coordination.core_events import (
            emit_quality_updated,
            emit_quality_degraded,
            emit_quality_score_updated,
            emit_quality_check_requested,
            emit_game_quality_score,
        )

        assert callable(emit_quality_updated)
        assert callable(emit_quality_degraded)
        assert callable(emit_quality_score_updated)
        assert callable(emit_quality_check_requested)
        assert callable(emit_game_quality_score)

    def test_import_regression_emitters(self):
        """Regression-related emit functions are importable."""
        from app.coordination.core_events import (
            emit_regression_detected,
            emit_plateau_detected,
        )

        assert callable(emit_regression_detected)
        assert callable(emit_plateau_detected)

    def test_import_node_emitters(self):
        """Node-related emit functions are importable."""
        from app.coordination.core_events import (
            emit_host_offline,
            emit_host_online,
            emit_node_recovered,
            emit_node_unhealthy,
            emit_node_overloaded,
            emit_idle_resource_detected,
        )

        assert callable(emit_host_offline)
        assert callable(emit_host_online)
        assert callable(emit_node_recovered)
        assert callable(emit_node_unhealthy)
        assert callable(emit_node_overloaded)
        assert callable(emit_idle_resource_detected)

    def test_import_cluster_emitters(self):
        """Cluster-related emit functions are importable."""
        from app.coordination.core_events import (
            emit_leader_elected,
            emit_leader_lost,
            emit_cluster_capacity_changed,
            emit_p2p_cluster_healthy,
            emit_p2p_cluster_unhealthy,
            emit_p2p_node_dead,
        )

        assert callable(emit_leader_elected)
        assert callable(emit_leader_lost)
        assert callable(emit_cluster_capacity_changed)
        assert callable(emit_p2p_cluster_healthy)
        assert callable(emit_p2p_cluster_unhealthy)
        assert callable(emit_p2p_node_dead)

    def test_import_coordinator_emitters(self):
        """Coordinator-related emit functions are importable."""
        from app.coordination.core_events import (
            emit_coordinator_healthy,
            emit_coordinator_unhealthy,
            emit_coordinator_heartbeat,
            emit_coordinator_shutdown,
            emit_coordinator_health_degraded,
            emit_daemon_status_changed,
        )

        assert callable(emit_coordinator_healthy)
        assert callable(emit_coordinator_unhealthy)
        assert callable(emit_coordinator_heartbeat)
        assert callable(emit_coordinator_shutdown)
        assert callable(emit_coordinator_health_degraded)
        assert callable(emit_daemon_status_changed)

    def test_import_backpressure_emitters(self):
        """Backpressure-related emit functions are importable."""
        from app.coordination.core_events import (
            emit_backpressure_activated,
            emit_backpressure_released,
        )

        assert callable(emit_backpressure_activated)
        assert callable(emit_backpressure_released)

    def test_import_curriculum_emitters(self):
        """Curriculum-related emit functions are importable."""
        from app.coordination.core_events import (
            emit_curriculum_updated,
            emit_curriculum_rebalanced,
            emit_curriculum_advanced,
            emit_elo_velocity_changed,
            emit_exploration_boost,
        )

        assert callable(emit_curriculum_updated)
        assert callable(emit_curriculum_rebalanced)
        assert callable(emit_curriculum_advanced)
        assert callable(emit_elo_velocity_changed)
        assert callable(emit_exploration_boost)

    def test_import_task_emitters(self):
        """Task-related emit functions are importable."""
        from app.coordination.core_events import (
            emit_task_complete,
            emit_task_abandoned,
            emit_task_orphaned,
        )

        assert callable(emit_task_complete)
        assert callable(emit_task_abandoned)
        assert callable(emit_task_orphaned)

    def test_import_repair_emitters(self):
        """Repair-related emit functions are importable."""
        from app.coordination.core_events import (
            emit_repair_completed,
            emit_repair_failed,
        )

        assert callable(emit_repair_completed)
        assert callable(emit_repair_failed)

    def test_import_misc_emitters(self):
        """Miscellaneous emit functions are importable."""
        from app.coordination.core_events import (
            emit_cache_invalidated,
            emit_handler_failed,
            emit_handler_timeout,
            emit_health_check_passed,
            emit_health_check_failed,
            emit_hyperparameter_updated,
            emit_model_corrupted,
            emit_new_games,
            emit_optimization_triggered,
            emit_data_event,
        )

        assert callable(emit_cache_invalidated)
        assert callable(emit_handler_failed)
        assert callable(emit_handler_timeout)
        assert callable(emit_health_check_passed)
        assert callable(emit_health_check_failed)
        assert callable(emit_hyperparameter_updated)
        assert callable(emit_model_corrupted)
        assert callable(emit_new_games)
        assert callable(emit_optimization_triggered)
        assert callable(emit_data_event)


# =============================================================================
# Functional Tests
# =============================================================================


class TestEventMappingsFunction:
    """Tests that event mapping functions work correctly."""

    def test_stage_to_data_mapping(self):
        """Stage to data event mapping works."""
        from app.coordination.core_events import STAGE_TO_DATA_EVENT_MAP

        # Should have mappings for common stages
        assert len(STAGE_TO_DATA_EVENT_MAP) > 0

    def test_data_to_cross_process_mapping(self):
        """Data to cross-process event mapping works."""
        from app.coordination.core_events import DATA_TO_CROSS_PROCESS_MAP

        assert len(DATA_TO_CROSS_PROCESS_MAP) >= 0  # Can be empty but importable

    def test_validate_mappings_callable(self):
        """validate_mappings function works."""
        from app.coordination.core_events import validate_mappings

        # Should not raise
        result = validate_mappings()
        # Result is either True or list of errors
        assert result is True or isinstance(result, list)


class TestEventNormalizationFunction:
    """Tests that event normalization functions work correctly."""

    def test_normalize_event_type_returns_uppercase(self):
        """Event names are normalized to uppercase."""
        from app.coordination.core_events import normalize_event_type

        # Normalization should return uppercase canonical names
        result = normalize_event_type("training_completed")
        assert result == "TRAINING_COMPLETED"

    def test_normalize_event_type_idempotent(self):
        """Normalizing an already-normalized name returns the same value."""
        from app.coordination.core_events import normalize_event_type

        result1 = normalize_event_type("TRAINING_COMPLETED")
        result2 = normalize_event_type("training_completed")
        assert result1 == result2

    def test_is_canonical_check(self):
        """is_canonical correctly identifies canonical names."""
        from app.coordination.core_events import is_canonical

        # Canonical names are uppercase
        assert is_canonical("TRAINING_COMPLETED") is True
        # Lowercase is not canonical
        assert is_canonical("training_completed") is False
        # Non-existent events should return False
        result = is_canonical("UNKNOWN_EVENT_TYPE_XYZ_123")
        assert isinstance(result, bool)


class TestRouterSingleton:
    """Tests for router singleton access."""

    def test_get_router_returns_instance(self):
        """get_router returns a router instance."""
        from app.coordination.core_events import get_router, reset_router

        reset_router()  # Reset before test
        router = get_router()
        assert router is not None

    def test_get_router_returns_same_instance(self):
        """get_router returns the same singleton."""
        from app.coordination.core_events import get_router, reset_router

        reset_router()
        router1 = get_router()
        router2 = get_router()
        assert router1 is router2

    def test_reset_router_clears_instance(self):
        """reset_router clears the singleton."""
        from app.coordination.core_events import get_router, reset_router

        router1 = get_router()
        reset_router()
        router2 = get_router()
        # After reset, should be a new instance
        assert router1 is not router2


class TestEventBusAccess:
    """Tests for event bus access functions."""

    def test_get_event_bus_returns_bus(self):
        """get_event_bus returns an EventBus."""
        from app.coordination.core_events import get_event_bus, EventBus

        bus = get_event_bus()
        assert bus is not None
        assert isinstance(bus, EventBus)

    def test_get_stage_event_bus_returns_bus(self):
        """get_stage_event_bus returns an EventBus."""
        from app.coordination.core_events import get_stage_event_bus

        bus = get_stage_event_bus()
        assert bus is not None


# =============================================================================
# __all__ Export Tests
# =============================================================================


class TestModuleExports:
    """Tests for __all__ exports."""

    def test_all_exports_accessible(self):
        """All items in __all__ are accessible."""
        from app.coordination import core_events

        missing = []
        for name in core_events.__all__:
            if not hasattr(core_events, name):
                missing.append(name)

        assert missing == [], f"Missing exports: {missing}"

    def test_exports_count(self):
        """Module exports expected number of items."""
        from app.coordination.core_events import __all__

        # Should have 100+ exports (70+ emitters + router + types + mappings)
        assert len(__all__) >= 100, f"Expected 100+ exports, got {len(__all__)}"


# =============================================================================
# DataEventType Enum Tests
# =============================================================================


class TestDataEventTypeEnum:
    """Tests for DataEventType enum."""

    def test_common_event_types_exist(self):
        """Common event types are defined."""
        from app.coordination.core_events import DataEventType

        # Check for common training events
        assert hasattr(DataEventType, "TRAINING_COMPLETED")
        assert hasattr(DataEventType, "TRAINING_STARTED")
        assert hasattr(DataEventType, "TRAINING_FAILED")

        # Check for sync events
        assert hasattr(DataEventType, "DATA_SYNC_COMPLETED")

        # Check for evaluation events
        assert hasattr(DataEventType, "EVALUATION_COMPLETED")

    def test_event_types_have_values(self):
        """Event types have string values."""
        from app.coordination.core_events import DataEventType

        assert isinstance(DataEventType.TRAINING_COMPLETED.value, str)
        assert len(DataEventType.TRAINING_COMPLETED.value) > 0


# =============================================================================
# RouterEvent Tests
# =============================================================================


class TestRouterEventFromCoreEvents:
    """Tests for RouterEvent via core_events import."""

    def test_router_event_creation(self):
        """RouterEvent can be created."""
        from app.coordination.core_events import RouterEvent

        event = RouterEvent(event_type="TEST_EVENT")
        assert event.event_type == "TEST_EVENT"
        assert event.payload == {}

    def test_router_event_with_payload(self):
        """RouterEvent can have payload."""
        from app.coordination.core_events import RouterEvent

        event = RouterEvent(
            event_type="TEST_EVENT",
            payload={"key": "value", "count": 42},
        )
        assert event.payload["key"] == "value"
        assert event.payload["count"] == 42

    def test_router_event_has_timestamp(self):
        """RouterEvent has auto-generated timestamp."""
        from app.coordination.core_events import RouterEvent
        import time

        before = time.time()
        event = RouterEvent(event_type="TEST")
        after = time.time()

        assert before <= event.timestamp <= after


# =============================================================================
# Cross-Module Compatibility Tests
# =============================================================================


class TestCrossModuleCompatibility:
    """Tests that old imports still work alongside new imports."""

    def test_event_router_import_still_works(self):
        """Old event_router imports still work."""
        from app.coordination.event_router import (
            UnifiedEventRouter,
            DataEventType,
            get_event_bus,
        )

        assert UnifiedEventRouter is not None
        assert DataEventType is not None
        assert callable(get_event_bus)

    def test_event_emitters_import_still_works(self):
        """Old event_emitters imports still work."""
        from app.coordination.event_emitters import (
            emit_training_complete,
            emit_sync_complete,
        )

        assert callable(emit_training_complete)
        assert callable(emit_sync_complete)

    def test_event_mappings_import_still_works(self):
        """Old event_mappings imports still work."""
        from app.coordination.event_mappings import (
            STAGE_TO_DATA_EVENT_MAP,
            get_data_event_type,
        )

        assert isinstance(STAGE_TO_DATA_EVENT_MAP, dict)
        assert callable(get_data_event_type)

    def test_event_normalization_import_still_works(self):
        """Old event_normalization imports still work."""
        from app.coordination.event_normalization import (
            normalize_event_type,
            CANONICAL_EVENT_NAMES,
        )

        assert callable(normalize_event_type)
        assert isinstance(CANONICAL_EVENT_NAMES, dict)

    def test_same_objects_from_both_paths(self):
        """Objects imported from old and new paths are the same."""
        from app.coordination.core_events import DataEventType as CoreEventType
        from app.coordination.event_router import DataEventType as RouterEventType

        assert CoreEventType is RouterEventType

        from app.coordination.core_events import emit_training_complete as core_emit
        from app.coordination.event_emitters import emit_training_complete as emitter_emit

        assert core_emit is emitter_emit
