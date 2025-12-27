"""Tests for app/coordination/types.py - Canonical coordination types.

December 27, 2025: Created as part of high-risk module test coverage effort.
types.py is the most-imported coordination module with 42+ importers.
"""

from __future__ import annotations

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


# =============================================================================
# BackpressureLevel Tests
# =============================================================================


class TestBackpressureLevel:
    """Tests for BackpressureLevel enum."""

    def test_all_values_exist(self):
        """Test that all expected backpressure levels exist."""
        from app.coordination.types import BackpressureLevel

        expected = ["NONE", "LOW", "SOFT", "MEDIUM", "HARD", "HIGH", "CRITICAL", "STOP"]
        for name in expected:
            assert hasattr(BackpressureLevel, name), f"Missing {name}"

    def test_string_values(self):
        """Test that enum values are lowercase strings."""
        from app.coordination.types import BackpressureLevel

        assert BackpressureLevel.NONE.value == "none"
        assert BackpressureLevel.SOFT.value == "soft"
        assert BackpressureLevel.HARD.value == "hard"
        assert BackpressureLevel.STOP.value == "stop"

    def test_from_legacy_queue_valid(self):
        """Test from_legacy_queue with valid values."""
        from app.coordination.types import BackpressureLevel

        assert BackpressureLevel.from_legacy_queue("none") == BackpressureLevel.NONE
        assert BackpressureLevel.from_legacy_queue("soft") == BackpressureLevel.SOFT
        assert BackpressureLevel.from_legacy_queue("hard") == BackpressureLevel.HARD
        assert BackpressureLevel.from_legacy_queue("stop") == BackpressureLevel.STOP

    def test_from_legacy_queue_case_insensitive(self):
        """Test from_legacy_queue handles case variations."""
        from app.coordination.types import BackpressureLevel

        assert BackpressureLevel.from_legacy_queue("NONE") == BackpressureLevel.NONE
        assert BackpressureLevel.from_legacy_queue("Soft") == BackpressureLevel.SOFT
        assert BackpressureLevel.from_legacy_queue("HARD") == BackpressureLevel.HARD

    def test_from_legacy_queue_unknown(self):
        """Test from_legacy_queue returns NONE for unknown values."""
        from app.coordination.types import BackpressureLevel

        assert BackpressureLevel.from_legacy_queue("unknown") == BackpressureLevel.NONE
        assert BackpressureLevel.from_legacy_queue("") == BackpressureLevel.NONE

    def test_from_legacy_resource_valid(self):
        """Test from_legacy_resource with valid values."""
        from app.coordination.types import BackpressureLevel

        assert BackpressureLevel.from_legacy_resource("none") == BackpressureLevel.NONE
        assert BackpressureLevel.from_legacy_resource("low") == BackpressureLevel.LOW
        assert BackpressureLevel.from_legacy_resource("medium") == BackpressureLevel.MEDIUM
        assert BackpressureLevel.from_legacy_resource("high") == BackpressureLevel.HIGH
        assert BackpressureLevel.from_legacy_resource("critical") == BackpressureLevel.CRITICAL

    def test_from_legacy_resource_case_insensitive(self):
        """Test from_legacy_resource handles case variations."""
        from app.coordination.types import BackpressureLevel

        assert BackpressureLevel.from_legacy_resource("LOW") == BackpressureLevel.LOW
        assert BackpressureLevel.from_legacy_resource("Medium") == BackpressureLevel.MEDIUM

    def test_from_legacy_resource_unknown(self):
        """Test from_legacy_resource returns NONE for unknown values."""
        from app.coordination.types import BackpressureLevel

        assert BackpressureLevel.from_legacy_resource("invalid") == BackpressureLevel.NONE

    def test_is_throttling_none(self):
        """Test is_throttling returns False for NONE."""
        from app.coordination.types import BackpressureLevel

        assert BackpressureLevel.NONE.is_throttling() is False

    def test_is_throttling_others(self):
        """Test is_throttling returns True for all non-NONE levels."""
        from app.coordination.types import BackpressureLevel

        throttling_levels = [
            BackpressureLevel.LOW,
            BackpressureLevel.SOFT,
            BackpressureLevel.MEDIUM,
            BackpressureLevel.HARD,
            BackpressureLevel.HIGH,
            BackpressureLevel.CRITICAL,
            BackpressureLevel.STOP,
        ]
        for level in throttling_levels:
            assert level.is_throttling() is True, f"{level} should be throttling"

    def test_should_stop_stop_and_critical(self):
        """Test should_stop returns True for STOP and CRITICAL."""
        from app.coordination.types import BackpressureLevel

        assert BackpressureLevel.STOP.should_stop() is True
        assert BackpressureLevel.CRITICAL.should_stop() is True

    def test_should_stop_others(self):
        """Test should_stop returns False for non-stop levels."""
        from app.coordination.types import BackpressureLevel

        non_stop_levels = [
            BackpressureLevel.NONE,
            BackpressureLevel.LOW,
            BackpressureLevel.SOFT,
            BackpressureLevel.MEDIUM,
            BackpressureLevel.HARD,
            BackpressureLevel.HIGH,
        ]
        for level in non_stop_levels:
            assert level.should_stop() is False, f"{level} should not stop"

    def test_reduction_factor_values(self):
        """Test reduction_factor returns expected values."""
        from app.coordination.types import BackpressureLevel

        # Test all levels have correct reduction factors
        assert BackpressureLevel.NONE.reduction_factor() == 1.0
        assert BackpressureLevel.LOW.reduction_factor() == 0.75
        assert BackpressureLevel.SOFT.reduction_factor() == 0.50
        assert BackpressureLevel.MEDIUM.reduction_factor() == 0.25
        assert BackpressureLevel.HARD.reduction_factor() == 0.10
        assert BackpressureLevel.HIGH.reduction_factor() == 0.05
        assert BackpressureLevel.CRITICAL.reduction_factor() == 0.01
        assert BackpressureLevel.STOP.reduction_factor() == 0.0

    def test_reduction_factor_monotonic_decrease(self):
        """Test reduction factors decrease with severity."""
        from app.coordination.types import BackpressureLevel

        # Order from least to most severe
        ordered = [
            BackpressureLevel.NONE,
            BackpressureLevel.LOW,
            BackpressureLevel.SOFT,
            BackpressureLevel.MEDIUM,
            BackpressureLevel.HARD,
            BackpressureLevel.HIGH,
            BackpressureLevel.CRITICAL,
            BackpressureLevel.STOP,
        ]

        for i in range(len(ordered) - 1):
            assert (
                ordered[i].reduction_factor() >= ordered[i + 1].reduction_factor()
            ), f"{ordered[i]} should have >= reduction than {ordered[i + 1]}"


# =============================================================================
# TaskType Tests
# =============================================================================


class TestTaskType:
    """Tests for TaskType enum."""

    def test_core_task_types_exist(self):
        """Test that core task types exist."""
        from app.coordination.types import TaskType

        core_types = [
            "SELFPLAY",
            "GPU_SELFPLAY",
            "HYBRID_SELFPLAY",
            "TRAINING",
            "EVALUATION",
            "EXPORT",
            "SYNC",
            "TOURNAMENT",
            "PARITY",
        ]
        for name in core_types:
            assert hasattr(TaskType, name), f"Missing core task type: {name}"

    def test_optimization_task_types(self):
        """Test optimization task types exist."""
        from app.coordination.types import TaskType

        assert hasattr(TaskType, "CMAES")

    def test_pipeline_task_types(self):
        """Test pipeline task types exist."""
        from app.coordination.types import TaskType

        pipeline_types = ["PIPELINE", "IMPROVEMENT_LOOP", "BACKGROUND_LOOP"]
        for name in pipeline_types:
            assert hasattr(TaskType, name), f"Missing pipeline task type: {name}"

    def test_unknown_fallback(self):
        """Test UNKNOWN fallback type exists."""
        from app.coordination.types import TaskType

        assert TaskType.UNKNOWN.value == "unknown"


# =============================================================================
# BoardType Tests
# =============================================================================


class TestBoardType:
    """Tests for BoardType enum."""

    def test_all_board_types_exist(self):
        """Test that all expected board types exist."""
        from app.coordination.types import BoardType

        expected = ["HEX8", "SQUARE8", "SQUARE19", "HEXAGONAL"]
        for name in expected:
            assert hasattr(BoardType, name), f"Missing board type: {name}"

    def test_string_values(self):
        """Test that enum values are lowercase strings."""
        from app.coordination.types import BoardType

        assert BoardType.HEX8.value == "hex8"
        assert BoardType.SQUARE8.value == "square8"
        assert BoardType.SQUARE19.value == "square19"
        assert BoardType.HEXAGONAL.value == "hexagonal"

    def test_cell_count_property(self):
        """Test cell_count property returns expected values."""
        from app.coordination.types import BoardType

        assert BoardType.HEX8.cell_count == 61
        assert BoardType.SQUARE8.cell_count == 64
        assert BoardType.SQUARE19.cell_count == 361
        assert BoardType.HEXAGONAL.cell_count == 469

    def test_vram_requirement_gb_property(self):
        """Test vram_requirement_gb property returns expected values."""
        from app.coordination.types import BoardType

        assert BoardType.HEX8.vram_requirement_gb == 4.0
        assert BoardType.SQUARE8.vram_requirement_gb == 4.0
        assert BoardType.SQUARE19.vram_requirement_gb == 8.0
        assert BoardType.HEXAGONAL.vram_requirement_gb == 12.0

    def test_from_string_exact_match(self):
        """Test from_string with exact matches."""
        from app.coordination.types import BoardType

        assert BoardType.from_string("hex8") == BoardType.HEX8
        assert BoardType.from_string("square8") == BoardType.SQUARE8
        assert BoardType.from_string("square19") == BoardType.SQUARE19
        assert BoardType.from_string("hexagonal") == BoardType.HEXAGONAL

    def test_from_string_case_insensitive(self):
        """Test from_string is case insensitive."""
        from app.coordination.types import BoardType

        assert BoardType.from_string("HEX8") == BoardType.HEX8
        assert BoardType.from_string("Square8") == BoardType.SQUARE8
        assert BoardType.from_string("HEXAGONAL") == BoardType.HEXAGONAL

    def test_from_string_variations(self):
        """Test from_string handles common variations."""
        from app.coordination.types import BoardType

        # Aliases
        assert BoardType.from_string("hexsmall") == BoardType.HEX8
        assert BoardType.from_string("sq8") == BoardType.SQUARE8
        assert BoardType.from_string("sq19") == BoardType.SQUARE19
        assert BoardType.from_string("hexlarge") == BoardType.HEXAGONAL

    def test_from_string_strips_separators(self):
        """Test from_string handles dashes and underscores."""
        from app.coordination.types import BoardType

        assert BoardType.from_string("hex-8") == BoardType.HEX8
        assert BoardType.from_string("square_8") == BoardType.SQUARE8

    def test_from_string_unknown_raises(self):
        """Test from_string raises ValueError for unknown types."""
        from app.coordination.types import BoardType

        with pytest.raises(ValueError, match="Unknown board type"):
            BoardType.from_string("invalid")

        with pytest.raises(ValueError):
            BoardType.from_string("hex16")


# =============================================================================
# WorkStatus Tests
# =============================================================================


class TestWorkStatus:
    """Tests for WorkStatus enum."""

    def test_all_statuses_exist(self):
        """Test that all expected work statuses exist."""
        from app.coordination.types import WorkStatus

        expected = [
            "PENDING",
            "CLAIMED",
            "RUNNING",
            "COMPLETED",
            "FAILED",
            "TIMEOUT",
            "CANCELLED",
        ]
        for name in expected:
            assert hasattr(WorkStatus, name), f"Missing work status: {name}"

    def test_string_values(self):
        """Test that enum values are lowercase strings."""
        from app.coordination.types import WorkStatus

        assert WorkStatus.PENDING.value == "pending"
        assert WorkStatus.COMPLETED.value == "completed"
        assert WorkStatus.FAILED.value == "failed"


# =============================================================================
# HealthLevel Tests
# =============================================================================


class TestHealthLevel:
    """Tests for HealthLevel enum."""

    def test_all_levels_exist(self):
        """Test that all expected health levels exist."""
        from app.coordination.types import HealthLevel

        expected = ["HEALTHY", "DEGRADED", "UNHEALTHY", "CRITICAL", "UNKNOWN"]
        for name in expected:
            assert hasattr(HealthLevel, name), f"Missing health level: {name}"

    def test_string_values(self):
        """Test that enum values are lowercase strings."""
        from app.coordination.types import HealthLevel

        assert HealthLevel.HEALTHY.value == "healthy"
        assert HealthLevel.DEGRADED.value == "degraded"
        assert HealthLevel.CRITICAL.value == "critical"


# =============================================================================
# TaskStatus Tests
# =============================================================================


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_all_statuses_exist(self):
        """Test that all expected task statuses exist."""
        from app.coordination.types import TaskStatus

        expected = [
            "PENDING",
            "RUNNING",
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMED_OUT",
            "ORPHANED",
        ]
        for name in expected:
            assert hasattr(TaskStatus, name), f"Missing task status: {name}"

    def test_string_values(self):
        """Test that enum values are lowercase strings."""
        from app.coordination.types import TaskStatus

        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.TIMED_OUT.value == "timed_out"
        assert TaskStatus.ORPHANED.value == "orphaned"


# =============================================================================
# ResourceType Tests
# =============================================================================


class TestResourceType:
    """Tests for ResourceType enum."""

    def test_all_resource_types_exist(self):
        """Test that all expected resource types exist."""
        from app.coordination.types import ResourceType

        expected = ["CPU", "GPU", "MEMORY", "DISK", "NETWORK", "HYBRID", "IO"]
        for name in expected:
            assert hasattr(ResourceType, name), f"Missing resource type: {name}"

    def test_string_values(self):
        """Test that enum values are lowercase strings."""
        from app.coordination.types import ResourceType

        assert ResourceType.CPU.value == "cpu"
        assert ResourceType.GPU.value == "gpu"
        assert ResourceType.HYBRID.value == "hybrid"


# =============================================================================
# TransferPriority Tests
# =============================================================================


class TestTransferPriority:
    """Tests for TransferPriority enum."""

    def test_all_priorities_exist(self):
        """Test that all expected transfer priorities exist."""
        from app.coordination.types import TransferPriority

        expected = ["CRITICAL", "HIGH", "NORMAL", "LOW", "BACKGROUND"]
        for name in expected:
            assert hasattr(TransferPriority, name), f"Missing priority: {name}"

    def test_string_values(self):
        """Test that enum values are lowercase strings."""
        from app.coordination.types import TransferPriority

        assert TransferPriority.CRITICAL.value == "critical"
        assert TransferPriority.NORMAL.value == "normal"
        assert TransferPriority.BACKGROUND.value == "background"


# =============================================================================
# CoordinatorHealthState Tests
# =============================================================================


class TestCoordinatorHealthState:
    """Tests for CoordinatorHealthState enum."""

    def test_all_states_exist(self):
        """Test that all expected coordinator health states exist."""
        from app.coordination.types import CoordinatorHealthState

        expected = [
            "UNKNOWN",
            "HEALTHY",
            "UNHEALTHY",
            "DEGRADED",
            "SHUTDOWN",
            "INIT_FAILED",
        ]
        for name in expected:
            assert hasattr(CoordinatorHealthState, name), f"Missing state: {name}"

    def test_string_values(self):
        """Test that enum values are lowercase strings."""
        from app.coordination.types import CoordinatorHealthState

        assert CoordinatorHealthState.HEALTHY.value == "healthy"
        assert CoordinatorHealthState.INIT_FAILED.value == "init_failed"


# =============================================================================
# CoordinatorRunState Tests
# =============================================================================


class TestCoordinatorRunState:
    """Tests for CoordinatorRunState enum."""

    def test_all_states_exist(self):
        """Test that all expected coordinator run states exist."""
        from app.coordination.types import CoordinatorRunState

        expected = ["RUNNING", "PAUSED", "DRAINING", "EMERGENCY", "STOPPED"]
        for name in expected:
            assert hasattr(CoordinatorRunState, name), f"Missing state: {name}"

    def test_string_values(self):
        """Test that enum values are lowercase strings."""
        from app.coordination.types import CoordinatorRunState

        assert CoordinatorRunState.RUNNING.value == "running"
        assert CoordinatorRunState.DRAINING.value == "draining"
        assert CoordinatorRunState.STOPPED.value == "stopped"


# =============================================================================
# ErrorSeverity Tests
# =============================================================================


class TestErrorSeverity:
    """Tests for ErrorSeverity enum."""

    def test_all_severities_exist(self):
        """Test that all expected error severities exist."""
        from app.coordination.types import ErrorSeverity

        expected = ["INFO", "WARNING", "ERROR", "CRITICAL"]
        for name in expected:
            assert hasattr(ErrorSeverity, name), f"Missing severity: {name}"

    def test_string_values(self):
        """Test that enum values are lowercase strings."""
        from app.coordination.types import ErrorSeverity

        assert ErrorSeverity.INFO.value == "info"
        assert ErrorSeverity.WARNING.value == "warning"
        assert ErrorSeverity.CRITICAL.value == "critical"


# =============================================================================
# Re-export Tests
# =============================================================================


class TestReExports:
    """Tests for re-exported types from app.core.node."""

    def test_node_types_are_exported(self):
        """Test that core node types are re-exported."""
        from app.coordination.types import (
            GPUInfo,
            NodeHealth,
            NodeRole,
            NodeState,
            Provider,
        )

        # Verify they are importable (not None)
        assert GPUInfo is not None
        assert NodeHealth is not None
        assert NodeRole is not None
        assert NodeState is not None
        assert Provider is not None

    def test_all_exports_in_dunder_all(self):
        """Test that __all__ contains all expected exports."""
        from app.coordination import types

        expected_exports = {
            # Core node types (re-exported)
            "GPUInfo",
            "NodeHealth",
            "NodeRole",
            "NodeState",
            "Provider",
            # Coordination types
            "BackpressureLevel",
            "TaskType",
            "BoardType",
            "WorkStatus",
            "HealthLevel",
            # December 2025 consolidation
            "TaskStatus",
            "ResourceType",
            "TransferPriority",
            "CoordinatorHealthState",
            "CoordinatorRunState",
            "ErrorSeverity",
        }

        actual_all = set(types.__all__)
        assert expected_exports == actual_all, f"Missing from __all__: {expected_exports - actual_all}"
