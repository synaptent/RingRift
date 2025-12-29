"""Tests for deprecated re-export modules (December 2025).

These modules are backward-compatibility shims scheduled for Q2 2026 removal.
Tests verify:
1. Imports work for backward compatibility
2. Deprecation warnings are emitted (where applicable)
3. Re-exports point to correct canonical modules
"""

import warnings

import pytest


class TestClusterSyncReexports:
    """Tests for app.coordination.cluster.sync deprecated re-exports."""

    def test_import_sync_scheduler(self):
        """Verify SyncScheduler is accessible."""
        from app.coordination.cluster.sync import SyncScheduler

        assert SyncScheduler is not None

    def test_import_sync_coordinator(self):
        """Verify SyncCoordinator is accessible."""
        from app.coordination.cluster.sync import SyncCoordinator

        assert SyncCoordinator is not None

    def test_import_bandwidth_coordinated_rsync(self):
        """Verify BandwidthCoordinatedRsync is accessible."""
        from app.coordination.cluster.sync import BandwidthCoordinatedRsync

        assert BandwidthCoordinatedRsync is not None

    def test_import_sync_mutex(self):
        """Verify SyncMutex is accessible."""
        from app.coordination.cluster.sync import SyncMutex

        assert SyncMutex is not None


class TestTrainingOrchestratorReexports:
    """Tests for app.coordination.training.orchestrator deprecated re-exports."""

    def test_import_training_coordinator(self):
        """Verify TrainingCoordinator is accessible."""
        from app.coordination.training.orchestrator import TrainingCoordinator

        assert TrainingCoordinator is not None

    def test_import_selfplay_orchestrator(self):
        """Verify SelfplayOrchestrator is accessible."""
        from app.coordination.training.orchestrator import SelfplayOrchestrator

        assert SelfplayOrchestrator is not None

    def test_import_all_exports(self):
        """Verify __all__ exports are accessible."""
        from app.coordination.training import orchestrator

        # Should have multiple exports
        assert hasattr(orchestrator, "__all__")
        for name in orchestrator.__all__:
            obj = getattr(orchestrator, name, None)
            assert obj is not None, f"Export '{name}' is None or missing"


class TestTrainingSchedulerReexports:
    """Tests for app.coordination.training.scheduler deprecated re-exports."""

    def test_import_priority_job_scheduler(self):
        """Verify PriorityJobScheduler is accessible."""
        from app.coordination.training.scheduler import PriorityJobScheduler

        assert PriorityJobScheduler is not None

    def test_import_unified_scheduler(self):
        """Verify UnifiedScheduler is accessible."""
        from app.coordination.training.scheduler import UnifiedScheduler

        assert UnifiedScheduler is not None

    def test_import_all_exports(self):
        """Verify __all__ exports are accessible."""
        from app.coordination.training import scheduler

        assert hasattr(scheduler, "__all__")
        for name in scheduler.__all__:
            obj = getattr(scheduler, name, None)
            assert obj is not None, f"Export '{name}' is None or missing"


class TestBaseHandlerDeprecated:
    """Tests for app.coordination.base_handler deprecated module."""

    def test_import_emits_deprecation_warning(self):
        """Verify importing base_handler emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Force reload to get the warning
            import importlib

            import app.coordination.base_handler

            importlib.reload(app.coordination.base_handler)

            # Check for deprecation warning
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) > 0, "Expected DeprecationWarning"

    def test_import_handler_base(self):
        """Verify HandlerBase is accessible from deprecated path."""
        from app.coordination.base_handler import HandlerBase

        assert HandlerBase is not None

    def test_handler_base_from_canonical_path(self):
        """Verify HandlerBase can be imported from canonical path."""
        from app.coordination.handler_base import HandlerBase

        assert HandlerBase is not None


class TestNoCircularImportsInReexports:
    """Verify no circular imports in re-export modules."""

    def test_cluster_sync_no_circular_import(self):
        """Verify cluster.sync imports without circular errors."""
        from app.coordination.cluster import sync

        assert sync is not None

    def test_training_orchestrator_no_circular_import(self):
        """Verify training.orchestrator imports without circular errors."""
        from app.coordination.training import orchestrator

        assert orchestrator is not None

    def test_training_scheduler_no_circular_import(self):
        """Verify training.scheduler imports without circular errors."""
        from app.coordination.training import scheduler

        assert scheduler is not None
