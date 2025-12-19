"""Tests for OptimizationOrchestrator (CMAES/NAS/PBT monitoring).

Tests cover:
- OptimizationType and OptimizationState enums
- OptimizationRun dataclass
- OptimizationOrchestrator event handling
- Module functions
"""

import time
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass


# =============================================================================
# Test OptimizationType Enum
# =============================================================================

class TestOptimizationType:
    """Tests for OptimizationType enum."""

    def test_cmaes_value(self):
        """Test CMAES type value."""
        from app.training.optimization_orchestrator import OptimizationType
        assert OptimizationType.CMAES.value == "cmaes"

    def test_nas_value(self):
        """Test NAS type value."""
        from app.training.optimization_orchestrator import OptimizationType
        assert OptimizationType.NAS.value == "nas"

    def test_pbt_value(self):
        """Test PBT type value."""
        from app.training.optimization_orchestrator import OptimizationType
        assert OptimizationType.PBT.value == "pbt"

    def test_all_types_exist(self):
        """Test all expected types exist."""
        from app.training.optimization_orchestrator import OptimizationType
        types = [t.value for t in OptimizationType]
        assert "cmaes" in types
        assert "nas" in types
        assert "pbt" in types


# =============================================================================
# Test OptimizationState Enum
# =============================================================================

class TestOptimizationState:
    """Tests for OptimizationState enum."""

    def test_pending_value(self):
        """Test PENDING state value."""
        from app.training.optimization_orchestrator import OptimizationState
        assert OptimizationState.PENDING.value == "pending"

    def test_running_value(self):
        """Test RUNNING state value."""
        from app.training.optimization_orchestrator import OptimizationState
        assert OptimizationState.RUNNING.value == "running"

    def test_generation_complete_value(self):
        """Test GENERATION_COMPLETE state value."""
        from app.training.optimization_orchestrator import OptimizationState
        assert OptimizationState.GENERATION_COMPLETE.value == "generation_complete"

    def test_completed_value(self):
        """Test COMPLETED state value."""
        from app.training.optimization_orchestrator import OptimizationState
        assert OptimizationState.COMPLETED.value == "completed"

    def test_failed_value(self):
        """Test FAILED state value."""
        from app.training.optimization_orchestrator import OptimizationState
        assert OptimizationState.FAILED.value == "failed"


# =============================================================================
# Test OptimizationRun Dataclass
# =============================================================================

class TestOptimizationRun:
    """Tests for OptimizationRun dataclass."""

    def test_create_minimal(self):
        """Test creating run with required fields."""
        from app.training.optimization_orchestrator import (
            OptimizationRun,
            OptimizationType,
            OptimizationState,
        )

        run = OptimizationRun(
            run_id="cmaes_sq8_001",
            opt_type=OptimizationType.CMAES,
            state=OptimizationState.RUNNING,
            config="square8_2p",
            started_at=time.time(),
            updated_at=time.time(),
        )

        assert run.run_id == "cmaes_sq8_001"
        assert run.opt_type == OptimizationType.CMAES
        assert run.state == OptimizationState.RUNNING
        assert run.config == "square8_2p"

    def test_default_values(self):
        """Test default values."""
        from app.training.optimization_orchestrator import (
            OptimizationRun,
            OptimizationType,
            OptimizationState,
        )

        run = OptimizationRun(
            run_id="test",
            opt_type=OptimizationType.NAS,
            state=OptimizationState.PENDING,
            config="test_config",
            started_at=time.time(),
            updated_at=time.time(),
        )

        assert run.current_generation == 0
        assert run.total_generations == 0
        assert run.best_score is None
        assert run.best_params == {}
        assert run.metadata == {}

    def test_to_dict(self):
        """Test to_dict method."""
        from app.training.optimization_orchestrator import (
            OptimizationRun,
            OptimizationType,
            OptimizationState,
        )

        now = time.time()
        run = OptimizationRun(
            run_id="nas_test_001",
            opt_type=OptimizationType.NAS,
            state=OptimizationState.RUNNING,
            config="square8_2p",
            started_at=now,
            updated_at=now,
            current_generation=5,
            total_generations=20,
            best_score=0.85,
            best_params={"lr": 0.001},
        )

        result = run.to_dict()

        assert result["run_id"] == "nas_test_001"
        assert result["type"] == "nas"
        assert result["state"] == "running"
        assert result["config"] == "square8_2p"
        assert result["current_generation"] == 5
        assert result["total_generations"] == 20
        assert result["best_score"] == 0.85
        assert result["best_params"] == {"lr": 0.001}
        assert "duration_seconds" in result

    def test_to_dict_includes_duration(self):
        """Test to_dict calculates duration dynamically."""
        from app.training.optimization_orchestrator import (
            OptimizationRun,
            OptimizationType,
            OptimizationState,
        )

        past_time = time.time() - 60  # Started 60 seconds ago
        run = OptimizationRun(
            run_id="test",
            opt_type=OptimizationType.PBT,
            state=OptimizationState.RUNNING,
            config="test",
            started_at=past_time,
            updated_at=time.time(),
        )

        result = run.to_dict()
        assert result["duration_seconds"] >= 59  # At least 59 seconds


# =============================================================================
# Test OptimizationOrchestrator
# =============================================================================

class TestOptimizationOrchestrator:
    """Tests for OptimizationOrchestrator class."""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator instance."""
        from app.training.optimization_orchestrator import (
            OptimizationOrchestrator,
            reset_optimization_orchestrator,
        )

        reset_optimization_orchestrator()
        orch = OptimizationOrchestrator()
        yield orch
        reset_optimization_orchestrator()

    def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator._active_runs == {}
        assert orchestrator._completed_runs == []
        assert orchestrator._max_history == 100
        assert orchestrator._subscribed is False
        assert orchestrator._event_counts == {}

    def test_initialization_custom_history(self):
        """Test initialization with custom history limit."""
        from app.training.optimization_orchestrator import OptimizationOrchestrator

        orch = OptimizationOrchestrator(max_history=50)
        assert orch._max_history == 50

    def test_track_event(self, orchestrator):
        """Test event counting."""
        orchestrator._track_event("cmaes_triggered")
        orchestrator._track_event("cmaes_triggered")
        orchestrator._track_event("nas_started")

        assert orchestrator._event_counts["cmaes_triggered"] == 2
        assert orchestrator._event_counts["nas_started"] == 1

    def test_get_run_id_with_explicit_id(self, orchestrator):
        """Test run ID extraction from payload."""
        from app.training.optimization_orchestrator import OptimizationType

        payload = {"run_id": "explicit_run_123"}
        run_id = orchestrator._get_run_id(payload, OptimizationType.CMAES)
        assert run_id == "explicit_run_123"

    def test_get_run_id_generated(self, orchestrator):
        """Test run ID generation when not in payload."""
        from app.training.optimization_orchestrator import OptimizationType

        payload = {"config": "square8_2p"}
        run_id = orchestrator._get_run_id(payload, OptimizationType.CMAES)
        assert run_id.startswith("cmaes_square8_2p_")

    def test_on_cmaes_triggered(self, orchestrator):
        """Test CMAES triggered handler."""
        mock_event = MagicMock()
        mock_event.payload = {
            "run_id": "cmaes_001",
            "config": "square8_2p",
            "generations": 50,
        }

        orchestrator._on_cmaes_triggered(mock_event)

        assert "cmaes_001" in orchestrator._active_runs
        run = orchestrator._active_runs["cmaes_001"]
        assert run.config == "square8_2p"
        assert run.total_generations == 50
        assert orchestrator._event_counts["cmaes_triggered"] == 1

    def test_on_cmaes_completed(self, orchestrator):
        """Test CMAES completed handler."""
        from app.training.optimization_orchestrator import (
            OptimizationRun,
            OptimizationType,
            OptimizationState,
        )

        # Add active run first
        orchestrator._active_runs["cmaes_001"] = OptimizationRun(
            run_id="cmaes_001",
            opt_type=OptimizationType.CMAES,
            state=OptimizationState.RUNNING,
            config="square8_2p",
            started_at=time.time(),
            updated_at=time.time(),
        )

        mock_event = MagicMock()
        mock_event.payload = {
            "run_id": "cmaes_001",
            "best_score": 0.92,
            "best_params": {"lr": 0.001},
        }

        orchestrator._on_cmaes_completed(mock_event)

        assert "cmaes_001" not in orchestrator._active_runs
        assert len(orchestrator._completed_runs) == 1
        assert orchestrator._completed_runs[0].best_score == 0.92

    def test_on_nas_triggered(self, orchestrator):
        """Test NAS triggered handler."""
        mock_event = MagicMock()
        mock_event.payload = {
            "run_id": "nas_001",
            "config": "hex6_3p",
            "generations": 100,
        }

        orchestrator._on_nas_triggered(mock_event)

        assert "nas_001" in orchestrator._active_runs
        run = orchestrator._active_runs["nas_001"]
        assert run.opt_type.value == "nas"

    def test_on_nas_started(self, orchestrator):
        """Test NAS started handler updates state."""
        from app.training.optimization_orchestrator import (
            OptimizationRun,
            OptimizationType,
            OptimizationState,
        )

        # Add pending run
        orchestrator._active_runs["nas_001"] = OptimizationRun(
            run_id="nas_001",
            opt_type=OptimizationType.NAS,
            state=OptimizationState.PENDING,
            config="test",
            started_at=time.time(),
            updated_at=time.time(),
        )

        mock_event = MagicMock()
        mock_event.payload = {"run_id": "nas_001"}

        orchestrator._on_nas_started(mock_event)

        assert orchestrator._active_runs["nas_001"].state == OptimizationState.RUNNING

    def test_on_nas_generation(self, orchestrator):
        """Test NAS generation handler."""
        from app.training.optimization_orchestrator import (
            OptimizationRun,
            OptimizationType,
            OptimizationState,
        )

        orchestrator._active_runs["nas_001"] = OptimizationRun(
            run_id="nas_001",
            opt_type=OptimizationType.NAS,
            state=OptimizationState.RUNNING,
            config="test",
            started_at=time.time(),
            updated_at=time.time(),
        )

        mock_event = MagicMock()
        mock_event.payload = {
            "run_id": "nas_001",
            "generation": 10,
            "best_score": 0.88,
        }

        orchestrator._on_nas_generation(mock_event)

        run = orchestrator._active_runs["nas_001"]
        assert run.current_generation == 10
        assert run.best_score == 0.88
        assert run.state == OptimizationState.GENERATION_COMPLETE

    def test_on_pbt_started(self, orchestrator):
        """Test PBT started handler."""
        mock_event = MagicMock()
        mock_event.payload = {
            "run_id": "pbt_001",
            "config": "square8_2p",
            "generations": 30,
        }

        orchestrator._on_pbt_started(mock_event)

        assert "pbt_001" in orchestrator._active_runs
        run = orchestrator._active_runs["pbt_001"]
        assert run.opt_type.value == "pbt"
        assert run.state.value == "running"

    def test_on_pbt_generation(self, orchestrator):
        """Test PBT generation handler."""
        from app.training.optimization_orchestrator import (
            OptimizationRun,
            OptimizationType,
            OptimizationState,
        )

        orchestrator._active_runs["pbt_001"] = OptimizationRun(
            run_id="pbt_001",
            opt_type=OptimizationType.PBT,
            state=OptimizationState.RUNNING,
            config="test",
            started_at=time.time(),
            updated_at=time.time(),
        )

        mock_event = MagicMock()
        mock_event.payload = {
            "run_id": "pbt_001",
            "generation": 15,
            "best_score": 0.75,
        }

        orchestrator._on_pbt_generation(mock_event)

        run = orchestrator._active_runs["pbt_001"]
        assert run.current_generation == 15
        assert run.best_score == 0.75

    def test_find_active_run_by_id(self, orchestrator):
        """Test finding run by explicit ID."""
        from app.training.optimization_orchestrator import OptimizationType

        payload = {"run_id": "test_001"}
        result = orchestrator._find_active_run(OptimizationType.CMAES, payload)
        assert result == "test_001"

    def test_find_active_run_by_config(self, orchestrator):
        """Test finding run by config match."""
        from app.training.optimization_orchestrator import (
            OptimizationRun,
            OptimizationType,
            OptimizationState,
        )

        orchestrator._active_runs["cmaes_sq8"] = OptimizationRun(
            run_id="cmaes_sq8",
            opt_type=OptimizationType.CMAES,
            state=OptimizationState.RUNNING,
            config="square8_2p",
            started_at=time.time(),
            updated_at=time.time(),
        )

        payload = {"config": "square8_2p"}
        result = orchestrator._find_active_run(OptimizationType.CMAES, payload)
        assert result == "cmaes_sq8"

    def test_add_to_history_respects_limit(self, orchestrator):
        """Test history limit is respected."""
        from app.training.optimization_orchestrator import (
            OptimizationRun,
            OptimizationType,
            OptimizationState,
        )

        orchestrator._max_history = 5

        for i in range(10):
            run = OptimizationRun(
                run_id=f"run_{i}",
                opt_type=OptimizationType.CMAES,
                state=OptimizationState.COMPLETED,
                config="test",
                started_at=time.time(),
                updated_at=time.time(),
            )
            orchestrator._add_to_history(run)

        assert len(orchestrator._completed_runs) == 5
        assert orchestrator._completed_runs[0].run_id == "run_5"

    def test_get_active_runs(self, orchestrator):
        """Test getting active runs."""
        from app.training.optimization_orchestrator import (
            OptimizationRun,
            OptimizationType,
            OptimizationState,
        )

        orchestrator._active_runs["run1"] = OptimizationRun(
            run_id="run1",
            opt_type=OptimizationType.CMAES,
            state=OptimizationState.RUNNING,
            config="sq8",
            started_at=time.time(),
            updated_at=time.time(),
        )
        orchestrator._active_runs["run2"] = OptimizationRun(
            run_id="run2",
            opt_type=OptimizationType.NAS,
            state=OptimizationState.RUNNING,
            config="hex6",
            started_at=time.time(),
            updated_at=time.time(),
        )

        result = orchestrator.get_active_runs()
        assert len(result) == 2
        assert all(isinstance(r, dict) for r in result)

    def test_get_completed_runs(self, orchestrator):
        """Test getting completed runs with limit."""
        from app.training.optimization_orchestrator import (
            OptimizationRun,
            OptimizationType,
            OptimizationState,
        )

        for i in range(10):
            run = OptimizationRun(
                run_id=f"run_{i}",
                opt_type=OptimizationType.CMAES,
                state=OptimizationState.COMPLETED,
                config="test",
                started_at=time.time(),
                updated_at=time.time(),
            )
            orchestrator._completed_runs.append(run)

        result = orchestrator.get_completed_runs(limit=5)
        assert len(result) == 5

    def test_get_run_active(self, orchestrator):
        """Test getting specific active run."""
        from app.training.optimization_orchestrator import (
            OptimizationRun,
            OptimizationType,
            OptimizationState,
        )

        orchestrator._active_runs["run_123"] = OptimizationRun(
            run_id="run_123",
            opt_type=OptimizationType.CMAES,
            state=OptimizationState.RUNNING,
            config="test",
            started_at=time.time(),
            updated_at=time.time(),
        )

        result = orchestrator.get_run("run_123")
        assert result is not None
        assert result["run_id"] == "run_123"

    def test_get_run_completed(self, orchestrator):
        """Test getting specific completed run."""
        from app.training.optimization_orchestrator import (
            OptimizationRun,
            OptimizationType,
            OptimizationState,
        )

        run = OptimizationRun(
            run_id="completed_456",
            opt_type=OptimizationType.NAS,
            state=OptimizationState.COMPLETED,
            config="test",
            started_at=time.time(),
            updated_at=time.time(),
        )
        orchestrator._completed_runs.append(run)

        result = orchestrator.get_run("completed_456")
        assert result is not None
        assert result["run_id"] == "completed_456"

    def test_get_run_not_found(self, orchestrator):
        """Test getting non-existent run."""
        result = orchestrator.get_run("nonexistent")
        assert result is None

    def test_get_status(self, orchestrator):
        """Test getting orchestrator status."""
        from app.training.optimization_orchestrator import (
            OptimizationRun,
            OptimizationType,
            OptimizationState,
        )

        orchestrator._subscribed = True
        orchestrator._event_counts = {"cmaes_triggered": 5, "nas_started": 3}

        orchestrator._active_runs["run1"] = OptimizationRun(
            run_id="run1",
            opt_type=OptimizationType.CMAES,
            state=OptimizationState.RUNNING,
            config="test",
            started_at=time.time(),
            updated_at=time.time(),
        )
        orchestrator._active_runs["run2"] = OptimizationRun(
            run_id="run2",
            opt_type=OptimizationType.CMAES,
            state=OptimizationState.RUNNING,
            config="test",
            started_at=time.time(),
            updated_at=time.time(),
        )

        status = orchestrator.get_status()

        assert status["subscribed"] is True
        assert status["active_runs"] == 2
        assert status["active_by_type"]["cmaes"] == 2
        assert status["event_counts"]["cmaes_triggered"] == 5


# =============================================================================
# Test Module Functions
# =============================================================================

class TestModuleFunctions:
    """Tests for module-level functions."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before and after each test."""
        from app.training.optimization_orchestrator import reset_optimization_orchestrator

        reset_optimization_orchestrator()
        yield
        reset_optimization_orchestrator()

    def test_get_optimization_orchestrator_creates_singleton(self):
        """Test that get_optimization_orchestrator creates a singleton."""
        from app.training.optimization_orchestrator import (
            get_optimization_orchestrator,
            OptimizationOrchestrator,
        )

        orch1 = get_optimization_orchestrator()
        orch2 = get_optimization_orchestrator()

        assert orch1 is orch2
        assert isinstance(orch1, OptimizationOrchestrator)

    def test_reset_optimization_orchestrator(self):
        """Test resetting the orchestrator singleton."""
        from app.training.optimization_orchestrator import (
            get_optimization_orchestrator,
            reset_optimization_orchestrator,
        )

        orch1 = get_optimization_orchestrator()
        reset_optimization_orchestrator()
        orch2 = get_optimization_orchestrator()

        assert orch1 is not orch2

    def test_wire_optimization_events(self):
        """Test wiring optimization events."""
        from app.training.optimization_orchestrator import (
            wire_optimization_events,
            reset_optimization_orchestrator,
        )

        with patch("app.distributed.data_events.get_event_bus") as mock_bus:
            mock_bus.return_value = MagicMock()
            orch = wire_optimization_events()
            assert orch._subscribed is True


# =============================================================================
# Integration Tests
# =============================================================================

class TestOptimizationIntegration:
    """Integration tests for optimization orchestrator."""

    @pytest.fixture(autouse=True)
    def reset(self):
        """Reset singleton."""
        from app.training.optimization_orchestrator import reset_optimization_orchestrator

        reset_optimization_orchestrator()
        yield
        reset_optimization_orchestrator()

    def test_full_cmaes_lifecycle(self):
        """Test complete CMAES run lifecycle."""
        from app.training.optimization_orchestrator import OptimizationOrchestrator

        orch = OptimizationOrchestrator()

        # Trigger CMAES run
        trigger_event = MagicMock()
        trigger_event.payload = {
            "run_id": "cmaes_lifecycle",
            "config": "square8_2p",
            "generations": 10,
        }
        orch._on_cmaes_triggered(trigger_event)

        assert "cmaes_lifecycle" in orch._active_runs

        # Complete CMAES run
        complete_event = MagicMock()
        complete_event.payload = {
            "run_id": "cmaes_lifecycle",
            "best_score": 0.95,
            "best_params": {"lr": 0.002},
        }
        orch._on_cmaes_completed(complete_event)

        assert "cmaes_lifecycle" not in orch._active_runs
        assert len(orch._completed_runs) == 1
        assert orch._completed_runs[0].best_score == 0.95

    def test_full_nas_lifecycle(self):
        """Test complete NAS run lifecycle."""
        from app.training.optimization_orchestrator import OptimizationOrchestrator

        orch = OptimizationOrchestrator()

        # Trigger NAS
        orch._on_nas_triggered(MagicMock(payload={"run_id": "nas_life", "config": "test"}))
        assert orch._active_runs["nas_life"].state.value == "pending"

        # Start NAS
        orch._on_nas_started(MagicMock(payload={"run_id": "nas_life"}))
        assert orch._active_runs["nas_life"].state.value == "running"

        # Complete generation
        orch._on_nas_generation(MagicMock(payload={"run_id": "nas_life", "generation": 1}))
        assert orch._active_runs["nas_life"].current_generation == 1

        # Complete NAS
        orch._on_nas_completed(MagicMock(payload={"run_id": "nas_life", "best_score": 0.88}))
        assert "nas_life" not in orch._active_runs
        assert len(orch._completed_runs) == 1

    def test_concurrent_optimization_runs(self):
        """Test handling concurrent runs of different types."""
        from app.training.optimization_orchestrator import OptimizationOrchestrator

        orch = OptimizationOrchestrator()

        # Start all three types
        orch._on_cmaes_triggered(MagicMock(payload={"run_id": "cmaes_1", "config": "sq8"}))
        orch._on_nas_triggered(MagicMock(payload={"run_id": "nas_1", "config": "hex6"}))
        orch._on_pbt_started(MagicMock(payload={"run_id": "pbt_1", "config": "sq8"}))

        assert len(orch._active_runs) == 3

        status = orch.get_status()
        assert status["active_by_type"]["cmaes"] == 1
        assert status["active_by_type"]["nas"] == 1
        assert status["active_by_type"]["pbt"] == 1
