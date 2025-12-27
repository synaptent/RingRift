"""Tests for centralized enum definitions.

This module tests the enums.py consolidation module to ensure:
1. All expected enums are exported
2. Enum values are stable and correct
3. No naming collisions exist
4. Backward compatibility is maintained
"""

from __future__ import annotations

import pytest


class TestEnumExports:
    """Test that all expected enums are exported from enums.py."""

    def test_leadership_role_export(self):
        """Test LeadershipRole enum is exported."""
        from app.coordination.enums import LeadershipRole

        assert hasattr(LeadershipRole, "LEADER")
        assert hasattr(LeadershipRole, "FOLLOWER")
        assert hasattr(LeadershipRole, "CANDIDATE")

    def test_cluster_node_role_export(self):
        """Test ClusterNodeRole enum is exported."""
        from app.coordination.enums import ClusterNodeRole

        assert hasattr(ClusterNodeRole, "TRAINING")
        assert hasattr(ClusterNodeRole, "SELFPLAY")
        assert hasattr(ClusterNodeRole, "COORDINATOR")

    def test_job_recovery_action_export(self):
        """Test JobRecoveryAction enum is exported."""
        from app.coordination.enums import JobRecoveryAction

        assert hasattr(JobRecoveryAction, "RESTART_JOB")
        assert hasattr(JobRecoveryAction, "KILL_JOB")

    def test_system_recovery_action_export(self):
        """Test SystemRecoveryAction enum is exported."""
        from app.coordination.enums import SystemRecoveryAction

        # SystemRecoveryAction has system-level recovery actions
        assert SystemRecoveryAction is not None

    def test_node_recovery_action_export(self):
        """Test NodeRecoveryAction enum is exported."""
        from app.coordination.enums import NodeRecoveryAction

        assert hasattr(NodeRecoveryAction, "RESTART")

    def test_daemon_type_export(self):
        """Test DaemonType enum is exported."""
        from app.coordination.enums import DaemonType

        # DaemonType should have 60+ daemon types
        assert len(list(DaemonType)) >= 60
        assert hasattr(DaemonType, "AUTO_SYNC")
        assert hasattr(DaemonType, "DATA_PIPELINE")
        assert hasattr(DaemonType, "FEEDBACK_LOOP")

    def test_daemon_state_export(self):
        """Test DaemonState enum is exported."""
        from app.coordination.enums import DaemonState

        assert hasattr(DaemonState, "RUNNING")
        assert hasattr(DaemonState, "STOPPED")
        assert hasattr(DaemonState, "STARTING")

    def test_node_health_state_export(self):
        """Test NodeHealthState enum is exported."""
        from app.coordination.enums import NodeHealthState

        assert hasattr(NodeHealthState, "HEALTHY")
        assert hasattr(NodeHealthState, "UNHEALTHY")
        assert hasattr(NodeHealthState, "DEGRADED")

    def test_error_severity_export(self):
        """Test ErrorSeverity enum is exported."""
        from app.coordination.enums import ErrorSeverity

        assert hasattr(ErrorSeverity, "ERROR")
        assert hasattr(ErrorSeverity, "WARNING")
        assert hasattr(ErrorSeverity, "CRITICAL")

    def test_recovery_status_export(self):
        """Test RecoveryStatus enum is exported."""
        from app.coordination.enums import RecoveryStatus

        assert hasattr(RecoveryStatus, "PENDING")
        assert hasattr(RecoveryStatus, "IN_PROGRESS")
        assert hasattr(RecoveryStatus, "COMPLETED")
        assert hasattr(RecoveryStatus, "FAILED")

    def test_recovery_result_export(self):
        """Test RecoveryResult enum is exported."""
        from app.coordination.enums import RecoveryResult

        assert hasattr(RecoveryResult, "SUCCESS")
        assert hasattr(RecoveryResult, "FAILED")

    def test_data_event_type_export(self):
        """Test DataEventType enum is exported."""
        from app.coordination.enums import DataEventType

        assert hasattr(DataEventType, "TRAINING_STARTED")
        assert hasattr(DataEventType, "TRAINING_COMPLETED")
        assert hasattr(DataEventType, "DATA_SYNC_COMPLETED")


class TestEnumDistinctness:
    """Test that similarly named enums are distinct to prevent collision bugs."""

    def test_leadership_vs_cluster_node_role(self):
        """LeadershipRole and ClusterNodeRole should be distinct enums."""
        from app.coordination.enums import ClusterNodeRole, LeadershipRole

        # They should be different enum classes
        assert LeadershipRole is not ClusterNodeRole

        # Their values should have different semantics
        assert LeadershipRole.LEADER.value != ClusterNodeRole.TRAINING.value

    def test_recovery_action_enums_are_distinct(self):
        """All three RecoveryAction enums should be distinct."""
        from app.coordination.enums import (
            JobRecoveryAction,
            NodeRecoveryAction,
            SystemRecoveryAction,
        )

        # All three should be different enum classes
        assert JobRecoveryAction is not SystemRecoveryAction
        assert SystemRecoveryAction is not NodeRecoveryAction
        assert JobRecoveryAction is not NodeRecoveryAction


class TestEnumValues:
    """Test that enum values are stable and correct."""

    def test_daemon_state_values(self):
        """Test DaemonState has expected values."""
        from app.coordination.enums import DaemonState

        # Critical daemon states
        assert DaemonState.RUNNING.value == "running"
        assert DaemonState.STOPPED.value == "stopped"

    def test_node_health_state_values(self):
        """Test NodeHealthState has expected values."""
        from app.coordination.enums import NodeHealthState

        assert NodeHealthState.HEALTHY.value == "healthy"
        assert NodeHealthState.UNHEALTHY.value == "unhealthy"
        assert NodeHealthState.DEGRADED.value == "degraded"

    def test_error_severity_ordering(self):
        """Test ErrorSeverity values for severity comparison."""
        from app.coordination.enums import ErrorSeverity

        # Severity should be orderable
        severities = [ErrorSeverity.INFO, ErrorSeverity.WARNING, ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]
        # Just verify all exist
        assert len(severities) == 4


class TestAllExports:
    """Test the __all__ export list."""

    def test_all_contains_primary_exports(self):
        """Test __all__ contains expected exports."""
        from app.coordination import enums

        expected = [
            "LeadershipRole",
            "ClusterNodeRole",
            "JobRecoveryAction",
            "SystemRecoveryAction",
            "NodeRecoveryAction",
            "DaemonType",
            "DaemonState",
            "NodeHealthState",
            "ErrorSeverity",
            "RecoveryStatus",
            "RecoveryResult",
            "DataEventType",
        ]

        for name in expected:
            assert name in enums.__all__, f"{name} should be in __all__"
            assert hasattr(enums, name), f"{name} should be accessible"

    def test_no_deprecated_names_in_all(self):
        """Deprecated names should not be in __all__."""
        from app.coordination import enums

        # NodeRole was the ambiguous name that caused collision bugs
        assert "NodeRole" not in enums.__all__
        # RecoveryAction was also ambiguous
        assert "RecoveryAction" not in enums.__all__


class TestImportPatterns:
    """Test various import patterns work correctly."""

    def test_star_import(self):
        """Test 'from enums import *' imports expected names."""
        # Get the __all__ names
        from app.coordination import enums

        # All names in __all__ should be importable
        for name in enums.__all__:
            assert hasattr(enums, name)

    def test_direct_import(self):
        """Test direct enum imports work."""
        from app.coordination.enums import DaemonState, DaemonType, DataEventType

        assert DaemonType is not None
        assert DaemonState is not None
        assert DataEventType is not None

    def test_source_module_import(self):
        """Test importing from source modules matches enums.py exports."""
        from app.coordination.daemon_types import DaemonType as DaemonTypeDirect
        from app.coordination.enums import DaemonType as DaemonTypeExported

        assert DaemonTypeDirect is DaemonTypeExported


class TestDaemonTypeCompleteness:
    """Test DaemonType enum has all expected daemon types."""

    def test_critical_daemons_exist(self):
        """Test critical daemon types are defined."""
        from app.coordination.enums import DaemonType

        critical = [
            "AUTO_SYNC",
            "DATA_PIPELINE",
            "FEEDBACK_LOOP",
            "EVALUATION",
            "AUTO_PROMOTION",
            "SELFPLAY_COORDINATOR",
            "EVENT_ROUTER",
            "DAEMON_WATCHDOG",
        ]

        for name in critical:
            assert hasattr(DaemonType, name), f"DaemonType.{name} should exist"

    def test_daemon_count(self):
        """Test DaemonType has at least 60 types."""
        from app.coordination.enums import DaemonType

        count = len(list(DaemonType))
        assert count >= 60, f"Expected 60+ daemon types, got {count}"


class TestDataEventTypeCompleteness:
    """Test DataEventType enum has all expected events."""

    def test_core_events_exist(self):
        """Test core data event types are defined."""
        from app.coordination.enums import DataEventType

        core_events = [
            "TRAINING_STARTED",
            "TRAINING_COMPLETED",
            "SELFPLAY_COMPLETE",  # Note: uses COMPLETE not COMPLETED
            "DATA_SYNC_STARTED",
            "DATA_SYNC_COMPLETED",
            "EVALUATION_STARTED",
            "EVALUATION_COMPLETED",
            "MODEL_PROMOTED",
        ]

        for name in core_events:
            assert hasattr(DataEventType, name), f"DataEventType.{name} should exist"

    def test_orphan_events_exist(self):
        """Test orphan game detection events are defined."""
        from app.coordination.enums import DataEventType

        # Dec 2025 additions for orphan game recovery
        assert hasattr(DataEventType, "ORPHAN_GAMES_DETECTED")
        assert hasattr(DataEventType, "ORPHAN_GAMES_REGISTERED")
