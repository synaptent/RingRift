"""Tests for pipeline_mixin_base.py.

December 2025: Created to complete unit test coverage for all pipeline modules.
Tests the base class utilities used by all 4 pipeline mixins.
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.pipeline_mixin_base import (
    DataPipelineOrchestratorProtocol,
    PipelineMixinBase,
    PipelineProtocol,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class MockCircuitBreaker:
    """Mock circuit breaker for testing."""

    def __init__(self, is_open: bool = False):
        self._is_open = is_open
        self.failures_recorded: list[str] = []
        self.successes_recorded = 0

    def is_open(self) -> bool:
        return self._is_open

    def record_failure(self, error_msg: str) -> None:
        self.failures_recorded.append(error_msg)

    def record_success(self) -> None:
        self.successes_recorded += 1


class MockCircuitBreakerWithState:
    """Mock circuit breaker using state attribute instead of is_open()."""

    def __init__(self, state_name: str = "CLOSED"):
        self.state = SimpleNamespace(name=state_name)


class MockOrchestrator(PipelineMixinBase):
    """Mock orchestrator for testing mixin methods."""

    def __init__(
        self,
        board_type: str | None = "hex8",
        num_players: int | None = 2,
        current_iteration: int = 1,
        auto_trigger: bool = True,
        circuit_breaker: Any = None,
    ):
        self._current_board_type = board_type
        self._current_num_players = num_players
        self._current_iteration = current_iteration
        self.auto_trigger = auto_trigger
        self.auto_trigger_sync = True
        self.auto_trigger_export = True
        self.auto_trigger_training = False
        self.auto_trigger_evaluation = True
        self.auto_trigger_promotion = False
        self._circuit_breaker = circuit_breaker


class MockRouterEvent:
    """Mock RouterEvent with stage_result."""

    def __init__(
        self,
        stage_result: Any = None,
        payload: dict | None = None,
        event_type: str = "test_event",
    ):
        self.stage_result = stage_result
        self.payload = payload
        self.event_type = event_type


class MockStageResult:
    """Mock StageCompletionResult."""

    def __init__(
        self,
        success: bool = True,
        stage: str = "EXPORT",
        iteration: int = 1,
        error: str | None = None,
    ):
        self.success = success
        self.stage = stage
        self.iteration = iteration
        self.timestamp = time.time()
        self.data = {}
        self.error = error


# =============================================================================
# Protocol Tests
# =============================================================================


class TestDataPipelineOrchestratorProtocol:
    """Tests for the DataPipelineOrchestratorProtocol runtime checkable protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Verify the protocol can be checked at runtime."""
        assert hasattr(DataPipelineOrchestratorProtocol, "__protocol_attrs__") or hasattr(
            DataPipelineOrchestratorProtocol, "_is_protocol"
        )

    def test_protocol_defines_core_attributes(self):
        """Verify protocol defines expected core attributes."""
        annotations = getattr(
            DataPipelineOrchestratorProtocol, "__annotations__", {}
        )
        # Check for some key attributes (protocol annotations)
        expected_attrs = [
            "_current_stage",
            "_current_iteration",
            "auto_trigger",
            "_circuit_breaker",
            "_paused",
        ]
        for attr in expected_attrs:
            assert attr in annotations, f"Protocol missing {attr}"

    def test_pipeline_protocol_alias(self):
        """Verify PipelineProtocol is an alias for DataPipelineOrchestratorProtocol."""
        assert PipelineProtocol is DataPipelineOrchestratorProtocol


# =============================================================================
# Config Key Tests
# =============================================================================


class TestGetConfigKey:
    """Tests for _get_config_key() method."""

    def test_returns_config_key_when_both_set(self):
        """Returns 'board_type_Np' when both values set."""
        orch = MockOrchestrator(board_type="hex8", num_players=2)
        assert orch._get_config_key() == "hex8_2p"

    def test_returns_config_key_for_4_players(self):
        """Returns correct key for 4 players."""
        orch = MockOrchestrator(board_type="square8", num_players=4)
        assert orch._get_config_key() == "square8_4p"

    def test_returns_unknown_when_board_type_none(self):
        """Returns 'unknown' when board_type is None."""
        orch = MockOrchestrator(board_type=None, num_players=2)
        assert orch._get_config_key() == "unknown"

    def test_returns_unknown_when_num_players_none(self):
        """Returns 'unknown' when num_players is None."""
        orch = MockOrchestrator(board_type="hex8", num_players=None)
        assert orch._get_config_key() == "unknown"

    def test_returns_unknown_when_both_none(self):
        """Returns 'unknown' when both values are None."""
        orch = MockOrchestrator(board_type=None, num_players=None)
        assert orch._get_config_key() == "unknown"

    def test_handles_missing_attributes(self):
        """Returns 'unknown' when attributes don't exist."""
        mixin = PipelineMixinBase()  # No attributes set
        assert mixin._get_config_key() == "unknown"


# =============================================================================
# Stage Result Extraction Tests
# =============================================================================


class TestExtractStageResult:
    """Tests for _extract_stage_result() method."""

    def test_extracts_from_router_event_with_stage_result(self):
        """Extracts stage_result from RouterEvent when present."""
        stage_result = MockStageResult(success=True, stage="EXPORT", iteration=5)
        router_event = MockRouterEvent(stage_result=stage_result)

        mixin = PipelineMixinBase()
        result = mixin._extract_stage_result(router_event)

        assert result is stage_result
        assert result.success is True
        assert result.stage == "EXPORT"
        assert result.iteration == 5

    def test_creates_namespace_from_router_event_payload(self):
        """Creates SimpleNamespace from RouterEvent payload when no stage_result."""
        payload = {
            "success": True,
            "stage": "TRAINING",
            "iteration": 10,
            "data": {"model": "test.pth"},
            "error": None,
        }
        router_event = MockRouterEvent(stage_result=None, payload=payload, event_type="test")

        mixin = PipelineMixinBase()
        result = mixin._extract_stage_result(router_event)

        assert isinstance(result, SimpleNamespace)
        assert result.success is True
        assert result.stage == "TRAINING"
        assert result.iteration == 10
        assert result.data == {"model": "test.pth"}
        assert result.error is None

    def test_returns_as_is_for_direct_stage_result(self):
        """Returns stage result directly when not wrapped."""
        stage_result = MockStageResult(success=False, stage="EVALUATION", iteration=3, error="Test error")

        mixin = PipelineMixinBase()
        result = mixin._extract_stage_result(stage_result)

        assert result is stage_result
        assert result.success is False
        assert result.error == "Test error"

    def test_handles_router_event_with_empty_payload(self):
        """Handles RouterEvent with None payload gracefully."""
        router_event = MockRouterEvent(stage_result=None, payload=None, event_type="test")

        mixin = PipelineMixinBase()
        result = mixin._extract_stage_result(router_event)

        assert isinstance(result, SimpleNamespace)
        assert result.success is True  # Default
        assert result.stage == "unknown"
        assert result.iteration == 0

    def test_handles_plain_dict(self):
        """Returns plain dict as-is (no special handling)."""
        data = {"success": True, "stage": "SYNC"}

        mixin = PipelineMixinBase()
        result = mixin._extract_stage_result(data)

        assert result == data

    def test_handles_none_input(self):
        """Returns None as-is."""
        mixin = PipelineMixinBase()
        result = mixin._extract_stage_result(None)
        assert result is None


# =============================================================================
# Logging Tests
# =============================================================================


class TestLogStageEvent:
    """Tests for _log_stage_event() method."""

    def test_logs_successful_event(self, caplog):
        """Logs successful events at INFO level."""
        orch = MockOrchestrator(board_type="hex8", num_players=2, current_iteration=5)
        result = MockStageResult(success=True, iteration=5)

        with caplog.at_level(logging.INFO):
            orch._log_stage_event("SYNC_COMPLETE", result)

        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.INFO
        assert "[hex8_2p]" in caplog.records[0].message
        assert "SYNC_COMPLETE" in caplog.records[0].message
        assert "iteration=5" in caplog.records[0].message

    def test_logs_failed_event_as_warning(self, caplog):
        """Logs failed events at WARNING level."""
        orch = MockOrchestrator(board_type="square8", num_players=4)
        result = MockStageResult(success=False, error="Connection timeout")

        with caplog.at_level(logging.WARNING):
            orch._log_stage_event("TRAINING_FAILED", result)

        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.WARNING
        assert "FAILED" in caplog.records[0].message
        assert "Connection timeout" in caplog.records[0].message

    def test_includes_extra_data_in_log(self, caplog):
        """Includes extra data in log message."""
        orch = MockOrchestrator()
        result = MockStageResult(success=True)
        extra = {"games": 100, "model": "best.pth"}

        with caplog.at_level(logging.INFO):
            orch._log_stage_event("EXPORT_COMPLETE", result, extra=extra)

        assert "games" in caplog.records[0].message or "100" in caplog.records[0].message

    def test_uses_result_iteration_if_available(self, caplog):
        """Uses iteration from result rather than orchestrator if present."""
        orch = MockOrchestrator(current_iteration=1)
        result = MockStageResult(success=True, iteration=99)

        with caplog.at_level(logging.INFO):
            orch._log_stage_event("TEST_EVENT", result)

        assert "iteration=99" in caplog.records[0].message


class TestLogTrigger:
    """Tests for _log_trigger() method."""

    def test_logs_trigger_with_reason(self, caplog):
        """Logs trigger with reason."""
        orch = MockOrchestrator(board_type="hexagonal", num_players=3, current_iteration=7)

        with caplog.at_level(logging.INFO):
            orch._log_trigger("EXPORT", reason="Sync completed successfully")

        assert len(caplog.records) == 1
        assert "[hexagonal_3p]" in caplog.records[0].message
        assert "Triggering EXPORT" in caplog.records[0].message
        assert "iteration=7" in caplog.records[0].message
        assert "Sync completed successfully" in caplog.records[0].message

    def test_logs_trigger_without_reason(self, caplog):
        """Logs trigger without reason."""
        orch = MockOrchestrator()

        with caplog.at_level(logging.INFO):
            orch._log_trigger("TRAINING")

        assert "Triggering TRAINING" in caplog.records[0].message
        assert ":" not in caplog.records[0].message.split(")")[-1]  # No trailing colon


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestIsCircuitOpen:
    """Tests for _is_circuit_open() method."""

    def test_returns_false_when_no_circuit_breaker(self):
        """Returns False when circuit breaker is None."""
        orch = MockOrchestrator(circuit_breaker=None)
        assert orch._is_circuit_open() is False

    def test_returns_true_when_circuit_open_via_is_open(self):
        """Returns True when circuit breaker.is_open() returns True."""
        cb = MockCircuitBreaker(is_open=True)
        orch = MockOrchestrator(circuit_breaker=cb)
        assert orch._is_circuit_open() is True

    def test_returns_false_when_circuit_closed_via_is_open(self):
        """Returns False when circuit breaker.is_open() returns False."""
        cb = MockCircuitBreaker(is_open=False)
        orch = MockOrchestrator(circuit_breaker=cb)
        assert orch._is_circuit_open() is False

    def test_falls_back_to_state_attribute(self):
        """Falls back to checking state.name when is_open() not available."""
        cb = MockCircuitBreakerWithState(state_name="OPEN")
        orch = MockOrchestrator(circuit_breaker=cb)
        assert orch._is_circuit_open() is True

    def test_state_closed_returns_false(self):
        """Returns False when state.name is CLOSED."""
        cb = MockCircuitBreakerWithState(state_name="CLOSED")
        orch = MockOrchestrator(circuit_breaker=cb)
        assert orch._is_circuit_open() is False

    def test_handles_missing_attributes(self):
        """Returns False for circuit breaker without expected attributes."""
        cb = object()  # Empty object with no is_open or state
        orch = MockOrchestrator(circuit_breaker=cb)
        assert orch._is_circuit_open() is False


class TestRecordCircuitFailure:
    """Tests for _record_circuit_failure() method."""

    def test_records_failure_with_exception(self):
        """Records failure when given an exception."""
        cb = MockCircuitBreaker()
        orch = MockOrchestrator(circuit_breaker=cb)

        orch._record_circuit_failure(ValueError("Test error"))

        assert len(cb.failures_recorded) == 1
        assert "Test error" in cb.failures_recorded[0]

    def test_records_failure_with_string(self):
        """Records failure when given a string."""
        cb = MockCircuitBreaker()
        orch = MockOrchestrator(circuit_breaker=cb)

        orch._record_circuit_failure("Connection failed")

        assert len(cb.failures_recorded) == 1
        assert cb.failures_recorded[0] == "Connection failed"

    def test_no_op_when_no_circuit_breaker(self):
        """Does nothing when circuit breaker is None."""
        orch = MockOrchestrator(circuit_breaker=None)
        # Should not raise
        orch._record_circuit_failure("Error")

    def test_no_op_when_no_record_failure_method(self):
        """Does nothing when circuit breaker lacks record_failure()."""
        cb = object()  # Empty object
        orch = MockOrchestrator(circuit_breaker=cb)
        # Should not raise
        orch._record_circuit_failure("Error")


class TestRecordCircuitSuccess:
    """Tests for _record_circuit_success() method."""

    def test_records_success(self):
        """Records success on the circuit breaker."""
        cb = MockCircuitBreaker()
        orch = MockOrchestrator(circuit_breaker=cb)

        orch._record_circuit_success()

        assert cb.successes_recorded == 1

    def test_records_multiple_successes(self):
        """Records multiple successes."""
        cb = MockCircuitBreaker()
        orch = MockOrchestrator(circuit_breaker=cb)

        orch._record_circuit_success()
        orch._record_circuit_success()
        orch._record_circuit_success()

        assert cb.successes_recorded == 3

    def test_no_op_when_no_circuit_breaker(self):
        """Does nothing when circuit breaker is None."""
        orch = MockOrchestrator(circuit_breaker=None)
        # Should not raise
        orch._record_circuit_success()


# =============================================================================
# Auto-Trigger Tests
# =============================================================================


class TestShouldAutoTrigger:
    """Tests for _should_auto_trigger() method."""

    def test_returns_false_when_auto_trigger_disabled(self):
        """Returns False when master auto_trigger is disabled."""
        orch = MockOrchestrator(auto_trigger=False)
        assert orch._should_auto_trigger("sync") is False
        assert orch._should_auto_trigger("export") is False

    def test_checks_sync_flag(self):
        """Returns auto_trigger_sync value for 'sync' stage."""
        orch = MockOrchestrator(auto_trigger=True)
        orch.auto_trigger_sync = True
        assert orch._should_auto_trigger("sync") is True

        orch.auto_trigger_sync = False
        assert orch._should_auto_trigger("sync") is False

    def test_checks_export_flag(self):
        """Returns auto_trigger_export value for 'export' stage."""
        orch = MockOrchestrator(auto_trigger=True)
        orch.auto_trigger_export = True
        assert orch._should_auto_trigger("export") is True

        orch.auto_trigger_export = False
        assert orch._should_auto_trigger("export") is False

    def test_checks_training_flag(self):
        """Returns auto_trigger_training value for 'training' stage."""
        orch = MockOrchestrator(auto_trigger=True)
        orch.auto_trigger_training = False
        assert orch._should_auto_trigger("training") is False

        orch.auto_trigger_training = True
        assert orch._should_auto_trigger("training") is True

    def test_checks_evaluation_flag(self):
        """Returns auto_trigger_evaluation value for 'evaluation' stage."""
        orch = MockOrchestrator(auto_trigger=True)
        assert orch._should_auto_trigger("evaluation") is True  # Default True

    def test_checks_promotion_flag(self):
        """Returns auto_trigger_promotion value for 'promotion' stage."""
        orch = MockOrchestrator(auto_trigger=True)
        assert orch._should_auto_trigger("promotion") is False  # Default False

    def test_case_insensitive_stage_lookup(self):
        """Handles uppercase/mixed case stage names."""
        orch = MockOrchestrator(auto_trigger=True)
        orch.auto_trigger_sync = True

        assert orch._should_auto_trigger("SYNC") is True
        assert orch._should_auto_trigger("Sync") is True
        assert orch._should_auto_trigger("sYnC") is True

    def test_returns_false_for_unknown_stage(self):
        """Returns False for unknown stage names."""
        orch = MockOrchestrator(auto_trigger=True)
        assert orch._should_auto_trigger("unknown_stage") is False
        assert orch._should_auto_trigger("") is False

    def test_handles_missing_stage_flag(self):
        """Returns False when stage-specific flag doesn't exist."""
        orch = MockOrchestrator(auto_trigger=True)
        # Delete a flag
        del orch.auto_trigger_sync

        # Should return False, not raise
        assert orch._should_auto_trigger("sync") is False


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestPipelineMixinBaseIntegration:
    """Integration-style tests for the mixin base class."""

    def test_workflow_sync_to_export_trigger(self, caplog):
        """Test typical workflow: sync completes, trigger export."""
        orch = MockOrchestrator(
            board_type="square19",
            num_players=2,
            current_iteration=42,
            auto_trigger=True,
        )
        cb = MockCircuitBreaker(is_open=False)
        orch._circuit_breaker = cb

        # Simulate sync completion
        sync_result = MockStageResult(success=True, stage="SYNC", iteration=42)

        with caplog.at_level(logging.INFO):
            # Log the sync completion
            orch._log_stage_event("SYNC_COMPLETE", sync_result)

            # Check if we should auto-trigger export
            if orch._should_auto_trigger("export") and not orch._is_circuit_open():
                orch._log_trigger("EXPORT", reason="Sync completed")
                orch._record_circuit_success()

        # Verify logging
        assert len(caplog.records) == 2
        assert "SYNC_COMPLETE" in caplog.records[0].message
        assert "Triggering EXPORT" in caplog.records[1].message

        # Verify circuit breaker
        assert cb.successes_recorded == 1

    def test_workflow_with_failure(self, caplog):
        """Test workflow with stage failure."""
        orch = MockOrchestrator(board_type="hex8", num_players=4)
        cb = MockCircuitBreaker()
        orch._circuit_breaker = cb

        # Simulate training failure
        training_result = MockStageResult(
            success=False,
            stage="TRAINING",
            iteration=10,
            error="Out of memory",
        )

        with caplog.at_level(logging.WARNING):
            orch._log_stage_event("TRAINING_FAILED", training_result)
            orch._record_circuit_failure(ValueError("Out of memory"))

        assert len(caplog.records) == 1
        assert "FAILED" in caplog.records[0].message
        assert len(cb.failures_recorded) == 1

    def test_workflow_with_open_circuit(self, caplog):
        """Test that operations are blocked when circuit is open."""
        cb = MockCircuitBreaker(is_open=True)
        orch = MockOrchestrator(auto_trigger=True, circuit_breaker=cb)

        # Sync completed
        sync_result = MockStageResult(success=True, stage="SYNC", iteration=1)

        with caplog.at_level(logging.INFO):
            orch._log_stage_event("SYNC_COMPLETE", sync_result)

            # Should not trigger export because circuit is open
            if orch._should_auto_trigger("export") and not orch._is_circuit_open():
                orch._log_trigger("EXPORT")

        # Only sync complete should be logged, no export trigger
        assert len(caplog.records) == 1
        assert "SYNC_COMPLETE" in caplog.records[0].message
