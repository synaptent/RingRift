"""Tests for training coordination re-export modules.

These modules consolidate imports from multiple coordination modules.
The underlying modules have comprehensive tests - these tests verify
the re-exports work correctly.

December 2025 - Phase 2 test coverage
"""

import pytest


class TestOrchestratorReexports:
    """Tests for app.coordination.training.orchestrator re-exports."""

    def test_training_coordinator_exports(self):
        """TrainingCoordinator and related functions are importable."""
        from app.coordination.training.orchestrator import (
            TrainingCoordinator,
            get_training_coordinator,
            get_training_status,
            TrainingJob,
            can_train,
            request_training_slot,
            release_training_slot,
            wire_training_events,
        )
        assert TrainingCoordinator is not None
        assert callable(get_training_coordinator)
        assert callable(get_training_status)
        assert TrainingJob is not None
        assert callable(can_train)
        assert callable(request_training_slot)
        assert callable(release_training_slot)
        assert callable(wire_training_events)

    def test_selfplay_orchestrator_exports(self):
        """SelfplayOrchestrator and related functions are importable."""
        from app.coordination.training.orchestrator import (
            SelfplayOrchestrator,
            get_selfplay_orchestrator,
            get_selfplay_stats,
            is_large_board,
            get_engine_for_board,
            get_simulation_budget_for_board,
            SelfplayStats,
            SelfplayType,
            wire_selfplay_events,
        )
        assert SelfplayOrchestrator is not None
        assert callable(get_selfplay_orchestrator)
        assert callable(get_selfplay_stats)
        assert callable(is_large_board)
        assert callable(get_engine_for_board)
        assert callable(get_simulation_budget_for_board)
        assert SelfplayStats is not None
        assert SelfplayType is not None
        assert callable(wire_selfplay_events)

    def test_all_exports_defined(self):
        """__all__ contains expected exports."""
        from app.coordination.training import orchestrator

        expected = [
            "TrainingCoordinator",
            "get_training_coordinator",
            "get_training_status",
            "TrainingJob",
            "can_train",
            "request_training_slot",
            "release_training_slot",
            "wire_training_events",
            "SelfplayOrchestrator",
            "get_selfplay_orchestrator",
            "get_selfplay_stats",
            "is_large_board",
            "get_engine_for_board",
            "get_simulation_budget_for_board",
            "SelfplayStats",
            "SelfplayType",
            "wire_selfplay_events",
        ]
        for name in expected:
            assert name in orchestrator.__all__, f"{name} missing from __all__"

    def test_is_large_board_function(self):
        """is_large_board works correctly."""
        from app.coordination.training.orchestrator import is_large_board

        # Small boards
        assert is_large_board("hex8") is False
        assert is_large_board("square8") is False
        # Large boards
        assert is_large_board("hexagonal") is True
        assert is_large_board("square19") is True


class TestSchedulerReexports:
    """Tests for app.coordination.training.scheduler re-exports."""

    def test_job_scheduler_exports(self):
        """PriorityJobScheduler and related classes are importable."""
        from app.coordination.training.scheduler import (
            PriorityJobScheduler,
            JobPriority,
            ScheduledJob,
            HostDeadJobMigrator,
        )
        assert PriorityJobScheduler is not None
        assert JobPriority is not None
        assert ScheduledJob is not None
        assert HostDeadJobMigrator is not None

    def test_duration_scheduler_exports(self):
        """DurationScheduler and related functions are importable."""
        from app.coordination.training.scheduler import (
            DurationScheduler,
            ScheduledTask,
            TaskDurationRecord,
            estimate_task_duration,
            can_schedule_task,
        )
        assert DurationScheduler is not None
        assert ScheduledTask is not None
        assert TaskDurationRecord is not None
        assert callable(estimate_task_duration)
        assert callable(can_schedule_task)

    def test_work_distributor_export(self):
        """WorkDistributor is importable."""
        from app.coordination.training.scheduler import WorkDistributor
        assert WorkDistributor is not None

    def test_unified_scheduler_exports(self):
        """UnifiedScheduler and factory function are importable."""
        from app.coordination.training.scheduler import (
            UnifiedScheduler,
            get_unified_scheduler,
        )
        assert UnifiedScheduler is not None
        assert callable(get_unified_scheduler)

    def test_all_exports_defined(self):
        """__all__ contains expected exports."""
        from app.coordination.training import scheduler

        expected = [
            "PriorityJobScheduler",
            "JobPriority",
            "ScheduledJob",
            "HostDeadJobMigrator",
            "DurationScheduler",
            "ScheduledTask",
            "TaskDurationRecord",
            "estimate_task_duration",
            "can_schedule_task",
            "WorkDistributor",
            "UnifiedScheduler",
            "get_unified_scheduler",
        ]
        for name in expected:
            assert name in scheduler.__all__, f"{name} missing from __all__"

    def test_job_priority_enum(self):
        """JobPriority enum has expected values."""
        from app.coordination.training.scheduler import JobPriority

        # Should have priority levels
        assert hasattr(JobPriority, "HIGH") or hasattr(JobPriority, "CRITICAL")
        assert hasattr(JobPriority, "NORMAL") or hasattr(JobPriority, "LOW")


class TestModuleInit:
    """Tests for training package __init__.py."""

    def test_package_importable(self):
        """Training package can be imported."""
        import app.coordination.training
        assert app.coordination.training is not None

    def test_orchestrator_submodule_importable(self):
        """Orchestrator submodule can be imported."""
        from app.coordination.training import orchestrator
        assert orchestrator is not None

    def test_scheduler_submodule_importable(self):
        """Scheduler submodule can be imported."""
        from app.coordination.training import scheduler
        assert scheduler is not None
