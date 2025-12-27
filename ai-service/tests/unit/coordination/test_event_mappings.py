"""Tests for event mapping utilities (December 2025).

Tests for app/coordination/event_mappings.py:
- STAGE_TO_DATA_EVENT_MAP - Stage-to-DataEventType mapping
- DATA_TO_STAGE_EVENT_MAP - Reverse mapping
- DATA_TO_CROSS_PROCESS_MAP - Cross-process queue routing
- CROSS_PROCESS_TO_DATA_MAP - Reverse mapping
- STAGE_TO_CROSS_PROCESS_MAP - Direct stage to cross-process
- Helper functions: get_data_event_type, get_cross_process_event_type, etc.
- validate_mappings() - Mapping consistency validation
"""

from __future__ import annotations

import pytest

from app.coordination.event_mappings import (
    CROSS_PROCESS_TO_DATA_MAP,
    DATA_TO_CROSS_PROCESS_MAP,
    DATA_TO_STAGE_EVENT_MAP,
    STAGE_TO_CROSS_PROCESS_MAP,
    STAGE_TO_DATA_EVENT_MAP,
    get_all_event_types,
    get_cross_process_event_type,
    get_data_event_type,
    get_stage_event_type,
    is_mapped_event,
    validate_mappings,
)


# =============================================================================
# Stage-to-Data Mapping Tests
# =============================================================================


class TestStageToDataMapping:
    """Tests for STAGE_TO_DATA_EVENT_MAP."""

    def test_selfplay_complete_mapping(self):
        """selfplay_complete maps to selfplay_complete."""
        assert STAGE_TO_DATA_EVENT_MAP["selfplay_complete"] == "selfplay_complete"

    def test_canonical_selfplay_mapping(self):
        """canonical_selfplay_complete maps to selfplay_complete."""
        assert STAGE_TO_DATA_EVENT_MAP["canonical_selfplay_complete"] == "selfplay_complete"

    def test_gpu_selfplay_mapping(self):
        """gpu_selfplay_complete maps to selfplay_complete."""
        assert STAGE_TO_DATA_EVENT_MAP["gpu_selfplay_complete"] == "selfplay_complete"

    def test_training_complete_mapping(self):
        """training_complete maps to training_completed."""
        assert STAGE_TO_DATA_EVENT_MAP["training_complete"] == "training_completed"

    def test_training_started_mapping(self):
        """training_started maps correctly."""
        assert STAGE_TO_DATA_EVENT_MAP["training_started"] == "training_started"

    def test_evaluation_complete_mapping(self):
        """evaluation_complete maps to evaluation_completed."""
        assert STAGE_TO_DATA_EVENT_MAP["evaluation_complete"] == "evaluation_completed"

    def test_promotion_complete_mapping(self):
        """promotion_complete maps to model_promoted."""
        assert STAGE_TO_DATA_EVENT_MAP["promotion_complete"] == "model_promoted"

    def test_sync_complete_mapping(self):
        """sync_complete maps to sync_completed."""
        assert STAGE_TO_DATA_EVENT_MAP["sync_complete"] == "sync_completed"

    def test_cmaes_complete_mapping(self):
        """cmaes_complete maps to cmaes_completed."""
        assert STAGE_TO_DATA_EVENT_MAP["cmaes_complete"] == "cmaes_completed"


class TestDataToStageMapping:
    """Tests for DATA_TO_STAGE_EVENT_MAP (reverse)."""

    def test_training_completed_reverse(self):
        """training_completed maps back to training_complete."""
        assert DATA_TO_STAGE_EVENT_MAP["training_completed"] == "training_complete"

    def test_evaluation_completed_reverse(self):
        """evaluation_completed maps back to evaluation_complete."""
        assert DATA_TO_STAGE_EVENT_MAP["evaluation_completed"] == "evaluation_complete"

    def test_model_promoted_reverse(self):
        """model_promoted maps back to promotion_complete."""
        assert DATA_TO_STAGE_EVENT_MAP["model_promoted"] == "promotion_complete"

    def test_selfplay_complete_reverse(self):
        """selfplay_complete maps back to selfplay_complete."""
        assert DATA_TO_STAGE_EVENT_MAP["selfplay_complete"] == "selfplay_complete"


# =============================================================================
# Data-to-CrossProcess Mapping Tests
# =============================================================================


class TestDataToCrossProcessMapping:
    """Tests for DATA_TO_CROSS_PROCESS_MAP."""

    def test_training_events_use_uppercase(self):
        """Training events map to UPPERCASE cross-process format."""
        assert DATA_TO_CROSS_PROCESS_MAP["training_started"] == "TRAINING_STARTED"
        assert DATA_TO_CROSS_PROCESS_MAP["training_completed"] == "TRAINING_COMPLETED"
        assert DATA_TO_CROSS_PROCESS_MAP["training_failed"] == "TRAINING_FAILED"

    def test_evaluation_events_mapped(self):
        """Evaluation events are properly mapped."""
        assert DATA_TO_CROSS_PROCESS_MAP["evaluation_started"] == "EVALUATION_STARTED"
        assert DATA_TO_CROSS_PROCESS_MAP["evaluation_completed"] == "EVALUATION_COMPLETED"
        assert DATA_TO_CROSS_PROCESS_MAP["elo_updated"] == "ELO_UPDATED"

    def test_promotion_events_mapped(self):
        """Promotion events are properly mapped."""
        assert DATA_TO_CROSS_PROCESS_MAP["model_promoted"] == "MODEL_PROMOTED"
        assert DATA_TO_CROSS_PROCESS_MAP["promotion_failed"] == "PROMOTION_FAILED"

    def test_sync_events_mapped(self):
        """Sync events are properly mapped."""
        assert DATA_TO_CROSS_PROCESS_MAP["sync_completed"] == "DATA_SYNC_COMPLETED"
        assert DATA_TO_CROSS_PROCESS_MAP["sync_started"] == "DATA_SYNC_STARTED"
        assert DATA_TO_CROSS_PROCESS_MAP["p2p_model_synced"] == "P2P_MODEL_SYNCED"

    def test_quality_events_mapped(self):
        """Quality events are properly mapped."""
        assert DATA_TO_CROSS_PROCESS_MAP["quality_score_updated"] == "QUALITY_SCORE_UPDATED"
        assert DATA_TO_CROSS_PROCESS_MAP["quality_degraded"] == "QUALITY_DEGRADED"

    def test_regression_events_mapped(self):
        """Regression events are properly mapped."""
        assert DATA_TO_CROSS_PROCESS_MAP["regression_detected"] == "REGRESSION_DETECTED"
        assert DATA_TO_CROSS_PROCESS_MAP["regression_critical"] == "REGRESSION_CRITICAL"
        assert DATA_TO_CROSS_PROCESS_MAP["regression_cleared"] == "REGRESSION_CLEARED"

    def test_p2p_events_mapped(self):
        """P2P cluster events are properly mapped."""
        assert DATA_TO_CROSS_PROCESS_MAP["p2p_node_dead"] == "P2P_NODE_DEAD"
        assert DATA_TO_CROSS_PROCESS_MAP["leader_elected"] == "LEADER_ELECTED"
        assert DATA_TO_CROSS_PROCESS_MAP["leader_lost"] == "LEADER_LOST"

    def test_work_queue_events_mapped(self):
        """Work queue events are properly mapped."""
        assert DATA_TO_CROSS_PROCESS_MAP["work_queued"] == "WORK_QUEUED"
        assert DATA_TO_CROSS_PROCESS_MAP["work_completed"] == "WORK_COMPLETED"
        assert DATA_TO_CROSS_PROCESS_MAP["work_failed"] == "WORK_FAILED"

    def test_task_lifecycle_events_mapped(self):
        """Task lifecycle events are properly mapped."""
        assert DATA_TO_CROSS_PROCESS_MAP["task_spawned"] == "TASK_SPAWNED"
        assert DATA_TO_CROSS_PROCESS_MAP["task_completed"] == "TASK_COMPLETED"
        assert DATA_TO_CROSS_PROCESS_MAP["task_abandoned"] == "TASK_ABANDONED"

    def test_resource_events_mapped(self):
        """Resource events are properly mapped."""
        assert DATA_TO_CROSS_PROCESS_MAP["backpressure_activated"] == "BACKPRESSURE_ACTIVATED"
        assert DATA_TO_CROSS_PROCESS_MAP["backpressure_released"] == "BACKPRESSURE_RELEASED"


class TestCrossProcessToDataMapping:
    """Tests for CROSS_PROCESS_TO_DATA_MAP (reverse)."""

    def test_reverse_mapping_is_complete(self):
        """Every DATA_TO_CROSS_PROCESS value has reverse mapping."""
        for data_event, cp_event in DATA_TO_CROSS_PROCESS_MAP.items():
            assert cp_event in CROSS_PROCESS_TO_DATA_MAP
            assert CROSS_PROCESS_TO_DATA_MAP[cp_event] == data_event

    def test_training_completed_reverse(self):
        """TRAINING_COMPLETED maps back to training_completed."""
        assert CROSS_PROCESS_TO_DATA_MAP["TRAINING_COMPLETED"] == "training_completed"

    def test_model_promoted_reverse(self):
        """MODEL_PROMOTED maps back to model_promoted."""
        assert CROSS_PROCESS_TO_DATA_MAP["MODEL_PROMOTED"] == "model_promoted"


# =============================================================================
# Stage-to-CrossProcess Mapping Tests
# =============================================================================


class TestStageToCrossProcessMapping:
    """Tests for STAGE_TO_CROSS_PROCESS_MAP (direct)."""

    def test_training_complete_direct(self):
        """training_complete maps directly to TRAINING_COMPLETED."""
        assert STAGE_TO_CROSS_PROCESS_MAP["training_complete"] == "TRAINING_COMPLETED"

    def test_selfplay_complete_direct(self):
        """selfplay_complete maps directly to SELFPLAY_BATCH_COMPLETE."""
        assert STAGE_TO_CROSS_PROCESS_MAP["selfplay_complete"] == "SELFPLAY_BATCH_COMPLETE"

    def test_evaluation_complete_direct(self):
        """evaluation_complete maps directly to EVALUATION_COMPLETED."""
        assert STAGE_TO_CROSS_PROCESS_MAP["evaluation_complete"] == "EVALUATION_COMPLETED"

    def test_promotion_complete_direct(self):
        """promotion_complete maps directly to MODEL_PROMOTED."""
        assert STAGE_TO_CROSS_PROCESS_MAP["promotion_complete"] == "MODEL_PROMOTED"

    def test_sync_complete_direct(self):
        """sync_complete maps directly to DATA_SYNC_COMPLETED."""
        assert STAGE_TO_CROSS_PROCESS_MAP["sync_complete"] == "DATA_SYNC_COMPLETED"


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestGetDataEventType:
    """Tests for get_data_event_type helper."""

    def test_valid_stage_event(self):
        """Returns data event for valid stage event."""
        result = get_data_event_type("training_complete")
        assert result == "training_completed"

    def test_unmapped_stage_event(self):
        """Returns None for unmapped stage event."""
        result = get_data_event_type("unknown_stage")
        assert result is None

    def test_selfplay_stages_return_same_event(self):
        """Multiple selfplay stages map to same data event."""
        assert get_data_event_type("selfplay_complete") == "selfplay_complete"
        assert get_data_event_type("canonical_selfplay_complete") == "selfplay_complete"
        assert get_data_event_type("gpu_selfplay_complete") == "selfplay_complete"


class TestGetCrossProcessEventType:
    """Tests for get_cross_process_event_type helper."""

    def test_from_data_event(self):
        """Converts data event to cross-process format."""
        result = get_cross_process_event_type("training_completed", source="data")
        assert result == "TRAINING_COMPLETED"

    def test_from_stage_event(self):
        """Converts stage event to cross-process format."""
        result = get_cross_process_event_type("training_complete", source="stage")
        assert result == "TRAINING_COMPLETED"

    def test_unmapped_event(self):
        """Returns None for unmapped event."""
        result = get_cross_process_event_type("unknown_event")
        assert result is None

    def test_default_source_is_data(self):
        """Default source is 'data'."""
        result = get_cross_process_event_type("model_promoted")
        assert result == "MODEL_PROMOTED"


class TestGetStageEventType:
    """Tests for get_stage_event_type helper."""

    def test_valid_data_event(self):
        """Returns stage event for valid data event."""
        result = get_stage_event_type("training_completed")
        assert result == "training_complete"

    def test_unmapped_data_event(self):
        """Returns None for unmapped data event."""
        result = get_stage_event_type("unknown_data")
        assert result is None


class TestIsMappedEvent:
    """Tests for is_mapped_event helper."""

    def test_lowercase_stage_event(self):
        """Recognizes lowercase stage events."""
        assert is_mapped_event("training_complete") is True
        assert is_mapped_event("selfplay_complete") is True

    def test_lowercase_data_event(self):
        """Recognizes lowercase data events."""
        assert is_mapped_event("training_completed") is True
        assert is_mapped_event("model_promoted") is True

    def test_uppercase_cross_process_event(self):
        """Recognizes uppercase cross-process events."""
        assert is_mapped_event("TRAINING_COMPLETED") is True
        assert is_mapped_event("MODEL_PROMOTED") is True

    def test_unmapped_event(self):
        """Returns False for unknown events."""
        assert is_mapped_event("totally_unknown") is False
        assert is_mapped_event("NOT_A_REAL_EVENT") is False


class TestGetAllEventTypes:
    """Tests for get_all_event_types helper."""

    def test_returns_set(self):
        """Returns a set of event types."""
        result = get_all_event_types()
        assert isinstance(result, set)

    def test_includes_stage_events(self):
        """Includes stage event types."""
        result = get_all_event_types()
        assert "training_complete" in result
        assert "selfplay_complete" in result

    def test_includes_data_events(self):
        """Includes data event types."""
        result = get_all_event_types()
        assert "training_completed" in result
        assert "model_promoted" in result

    def test_includes_cross_process_events(self):
        """Includes cross-process event types."""
        result = get_all_event_types()
        assert "TRAINING_COMPLETED" in result
        assert "MODEL_PROMOTED" in result

    def test_has_substantial_size(self):
        """Has a substantial number of event types."""
        result = get_all_event_types()
        # Should have at least 100 distinct event types
        assert len(result) >= 100


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidateMappings:
    """Tests for validate_mappings consistency checker."""

    def test_returns_list(self):
        """Returns a list of warnings."""
        result = validate_mappings()
        assert isinstance(result, list)

    def test_no_critical_errors(self):
        """No critical errors in current mappings."""
        warnings = validate_mappings()
        # Filter out expected warnings (if any)
        critical_warnings = [w for w in warnings if "CRITICAL" in w.upper()]
        assert len(critical_warnings) == 0

    def test_cross_process_names_are_uppercase(self):
        """Cross-process event names should be uppercase."""
        for cp_event in CROSS_PROCESS_TO_DATA_MAP:
            assert cp_event == cp_event.upper(), f"{cp_event} is not uppercase"


# =============================================================================
# Mapping Consistency Tests
# =============================================================================


class TestMappingConsistency:
    """Tests for overall mapping consistency."""

    def test_stage_to_data_values_are_strings(self):
        """All STAGE_TO_DATA_EVENT_MAP values are strings."""
        for stage, data in STAGE_TO_DATA_EVENT_MAP.items():
            assert isinstance(stage, str)
            assert isinstance(data, str)

    def test_data_to_cross_process_values_are_uppercase(self):
        """All DATA_TO_CROSS_PROCESS_MAP values are uppercase."""
        for data, cp in DATA_TO_CROSS_PROCESS_MAP.items():
            assert cp == cp.upper(), f"Value {cp} for {data} is not uppercase"

    def test_no_duplicate_stage_keys(self):
        """Stage keys are unique."""
        keys = list(STAGE_TO_DATA_EVENT_MAP.keys())
        assert len(keys) == len(set(keys))

    def test_no_duplicate_cross_process_keys(self):
        """Cross-process keys are unique."""
        keys = list(CROSS_PROCESS_TO_DATA_MAP.keys())
        assert len(keys) == len(set(keys))

    def test_training_lifecycle_complete(self):
        """Training lifecycle events are complete."""
        required_training = ["training_started", "training_completed", "training_failed"]
        for event in required_training:
            assert event in DATA_TO_CROSS_PROCESS_MAP

    def test_evaluation_lifecycle_complete(self):
        """Evaluation lifecycle events are complete."""
        required_eval = ["evaluation_started", "evaluation_completed", "evaluation_failed"]
        for event in required_eval:
            assert event in DATA_TO_CROSS_PROCESS_MAP

    def test_sync_lifecycle_complete(self):
        """Sync lifecycle events are complete."""
        required_sync = ["sync_started", "sync_completed", "sync_failed"]
        for event in required_sync:
            assert event in DATA_TO_CROSS_PROCESS_MAP

    def test_task_lifecycle_complete(self):
        """Task lifecycle events are complete."""
        required_task = ["task_spawned", "task_completed", "task_failed", "task_abandoned"]
        for event in required_task:
            assert event in DATA_TO_CROSS_PROCESS_MAP


class TestDecember2025Additions:
    """Tests for December 2025 event additions."""

    def test_p2p_node_dead_exists(self):
        """P2P_NODE_DEAD event added in Dec 2025."""
        assert "p2p_node_dead" in DATA_TO_CROSS_PROCESS_MAP
        assert DATA_TO_CROSS_PROCESS_MAP["p2p_node_dead"] == "P2P_NODE_DEAD"

    def test_task_abandoned_exists(self):
        """TASK_ABANDONED event added in Dec 2025."""
        assert "task_abandoned" in DATA_TO_CROSS_PROCESS_MAP
        assert DATA_TO_CROSS_PROCESS_MAP["task_abandoned"] == "TASK_ABANDONED"

    def test_handler_failed_exists(self):
        """HANDLER_FAILED event added in Dec 2025."""
        assert "handler_failed" in DATA_TO_CROSS_PROCESS_MAP
        assert DATA_TO_CROSS_PROCESS_MAP["handler_failed"] == "HANDLER_FAILED"

    def test_selfplay_feedback_events_exist(self):
        """Selfplay feedback loop events added in Dec 2025."""
        selfplay_events = [
            "selfplay_complete",
            "selfplay_target_updated",
            "selfplay_rate_changed",
        ]
        for event in selfplay_events:
            assert event in DATA_TO_CROSS_PROCESS_MAP
