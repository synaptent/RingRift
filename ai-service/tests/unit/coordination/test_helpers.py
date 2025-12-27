"""Tests for app.coordination.helpers module.

Tests the safe coordination helper utilities that wrap try/except import patterns:
- has_coordination() availability check
- Coordinator safe wrappers (get_coordinator_safe, can_spawn_safe, etc.)
- Role management safe wrappers (acquire_role_safe, release_role_safe, etc.)
- Safeguards functions (check_spawn_allowed, get_safeguards)
- Queue backpressure functions (should_throttle_safe, should_stop_safe)
- Sync mutex functions (acquire_sync_lock_safe, release_sync_lock_safe)
- Bandwidth management functions (request_bandwidth_safe, release_bandwidth_safe)
- Duration scheduling functions (can_schedule_task_safe, estimate_duration_safe)
- Cross-process event functions (publish_event_safe, poll_events_safe)
- Resource targets functions (get_resource_targets_safe, should_scale_up_safe)

December 2025 - Phase 3 test coverage for critical untested modules.
"""

from unittest.mock import MagicMock, patch
import pytest
import socket


class TestHasCoordination:
    """Tests for coordination availability check."""

    def test_has_coordination_returns_bool(self):
        """Should return boolean indicating coordination availability."""
        from app.coordination.helpers import has_coordination

        result = has_coordination()
        assert isinstance(result, bool)

    def test_has_coordination_consistent(self):
        """Should return consistent result across calls."""
        from app.coordination.helpers import has_coordination

        result1 = has_coordination()
        result2 = has_coordination()
        assert result1 == result2


class TestGetTaskTypes:
    """Tests for get_task_types()."""

    def test_get_task_types_returns_type_or_none(self):
        """Should return TaskType enum or None."""
        from app.coordination.helpers import get_task_types

        result = get_task_types()
        # Should be either None or a type/enum
        assert result is None or hasattr(result, '__members__')


class TestGetOrchestratorRoles:
    """Tests for get_orchestrator_roles()."""

    def test_get_orchestrator_roles_returns_type_or_none(self):
        """Should return OrchestratorRole enum or None."""
        from app.coordination.helpers import get_orchestrator_roles

        result = get_orchestrator_roles()
        assert result is None or hasattr(result, '__members__')


class TestGetCoordinatorSafe:
    """Tests for get_coordinator_safe()."""

    def test_get_coordinator_safe_when_coordination_available(self):
        """Should return coordinator or None without raising."""
        from app.coordination.helpers import get_coordinator_safe

        # Should not raise even if coordination unavailable
        result = get_coordinator_safe()
        # Result could be None or a coordinator instance
        assert result is None or hasattr(result, 'can_spawn_task')

    @patch('app.coordination.helpers._get_coordinator')
    @patch('app.coordination.helpers._HAS_COORDINATION', True)
    def test_get_coordinator_safe_handles_exception(self, mock_get_coordinator):
        """Should return None on exception."""
        mock_get_coordinator.side_effect = RuntimeError("Init failed")

        from app.coordination.helpers import get_coordinator_safe
        result = get_coordinator_safe()
        assert result is None

    @patch('app.coordination.helpers._HAS_COORDINATION', False)
    def test_get_coordinator_safe_no_coordination(self):
        """Should return None when coordination not available."""
        from app.coordination.helpers import get_coordinator_safe
        result = get_coordinator_safe()
        assert result is None


class TestCanSpawnSafe:
    """Tests for can_spawn_safe()."""

    def test_can_spawn_safe_returns_tuple(self):
        """Should return tuple of (bool, str)."""
        from app.coordination.helpers import can_spawn_safe

        result = can_spawn_safe(None)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    @patch('app.coordination.helpers._HAS_COORDINATION', False)
    def test_can_spawn_safe_no_coordination(self):
        """Should return (True, 'coordination_unavailable') when no coordination."""
        from app.coordination.helpers import can_spawn_safe

        allowed, reason = can_spawn_safe(None)
        assert allowed is True
        assert reason == "coordination_unavailable"

    def test_can_spawn_safe_uses_hostname_if_no_node_id(self):
        """Should use hostname when node_id not provided."""
        from app.coordination.helpers import can_spawn_safe

        # Should not raise
        result = can_spawn_safe(None, node_id=None)
        assert isinstance(result, tuple)


class TestRegisterTaskSafe:
    """Tests for register_task_safe()."""

    @patch('app.coordination.helpers._HAS_COORDINATION', False)
    def test_register_task_safe_no_coordination(self):
        """Should return False when no coordination."""
        from app.coordination.helpers import register_task_safe

        result = register_task_safe("test-task-1", None)
        assert result is False

    @patch('app.coordination.helpers.get_coordinator_safe')
    @patch('app.coordination.helpers._HAS_COORDINATION', True)
    def test_register_task_safe_no_coordinator(self, mock_get_coordinator):
        """Should return False when coordinator not available."""
        mock_get_coordinator.return_value = None

        from app.coordination.helpers import register_task_safe
        result = register_task_safe("test-task-2", None)
        assert result is False

    @patch('app.coordination.helpers.get_coordinator_safe')
    @patch('app.coordination.helpers._HAS_COORDINATION', True)
    def test_register_task_safe_success(self, mock_get_coordinator):
        """Should return True on successful registration."""
        mock_coordinator = MagicMock()
        mock_get_coordinator.return_value = mock_coordinator

        from app.coordination.helpers import register_task_safe
        result = register_task_safe("test-task-3", "SELFPLAY", "node-1", 12345)
        assert result is True
        mock_coordinator.register_task.assert_called_once()

    @patch('app.coordination.helpers.get_coordinator_safe')
    @patch('app.coordination.helpers._HAS_COORDINATION', True)
    def test_register_task_safe_handles_exception(self, mock_get_coordinator):
        """Should return False on exception."""
        mock_coordinator = MagicMock()
        mock_coordinator.register_task.side_effect = RuntimeError("DB error")
        mock_get_coordinator.return_value = mock_coordinator

        from app.coordination.helpers import register_task_safe
        result = register_task_safe("test-task-4", None)
        assert result is False


class TestCompleteTaskSafe:
    """Tests for complete_task_safe()."""

    @patch('app.coordination.helpers._HAS_COORDINATION', False)
    def test_complete_task_safe_no_coordination(self):
        """Should return False when no coordination."""
        from app.coordination.helpers import complete_task_safe

        result = complete_task_safe("test-task")
        assert result is False

    @patch('app.coordination.helpers.get_coordinator_safe')
    @patch('app.coordination.helpers._HAS_COORDINATION', True)
    def test_complete_task_safe_success(self, mock_get_coordinator):
        """Should return True on success."""
        mock_coordinator = MagicMock()
        mock_get_coordinator.return_value = mock_coordinator

        from app.coordination.helpers import complete_task_safe
        result = complete_task_safe("test-task")
        assert result is True
        mock_coordinator.complete_task.assert_called_once_with("test-task")


class TestFailTaskSafe:
    """Tests for fail_task_safe()."""

    @patch('app.coordination.helpers._HAS_COORDINATION', False)
    def test_fail_task_safe_no_coordination(self):
        """Should return False when no coordination."""
        from app.coordination.helpers import fail_task_safe

        result = fail_task_safe("test-task", "error message")
        assert result is False

    @patch('app.coordination.helpers.get_coordinator_safe')
    @patch('app.coordination.helpers._HAS_COORDINATION', True)
    def test_fail_task_safe_success(self, mock_get_coordinator):
        """Should return True on success."""
        mock_coordinator = MagicMock()
        mock_get_coordinator.return_value = mock_coordinator

        from app.coordination.helpers import fail_task_safe
        result = fail_task_safe("test-task", "test error")
        assert result is True
        mock_coordinator.fail_task.assert_called_once_with("test-task", "test error")


class TestGetRegistrySafe:
    """Tests for get_registry_safe()."""

    def test_get_registry_safe_returns_registry_or_none(self):
        """Should return registry or None without raising."""
        from app.coordination.helpers import get_registry_safe

        result = get_registry_safe()
        assert result is None or hasattr(result, 'is_role_held')


class TestAcquireReleasRoleSafe:
    """Tests for acquire_role_safe() and release_role_safe()."""

    @patch('app.coordination.helpers._HAS_COORDINATION', False)
    def test_acquire_role_safe_no_coordination(self):
        """Should return False when no coordination."""
        from app.coordination.helpers import acquire_role_safe

        result = acquire_role_safe(None)
        assert result is False

    @patch('app.coordination.helpers._HAS_COORDINATION', False)
    def test_release_role_safe_no_coordination(self):
        """Should return False when no coordination."""
        from app.coordination.helpers import release_role_safe

        result = release_role_safe(None)
        assert result is False


class TestHasRole:
    """Tests for has_role()."""

    @patch('app.coordination.helpers.get_registry_safe')
    def test_has_role_no_registry(self, mock_get_registry):
        """Should return False when no registry."""
        mock_get_registry.return_value = None

        from app.coordination.helpers import has_role
        result = has_role(None)
        assert result is False

    @patch('app.coordination.helpers.get_registry_safe')
    def test_has_role_checks_registry(self, mock_get_registry):
        """Should delegate to registry."""
        mock_registry = MagicMock()
        mock_registry.is_role_held.return_value = True
        mock_get_registry.return_value = mock_registry

        from app.coordination.helpers import has_role
        result = has_role("TEST_ROLE")
        assert result is True
        mock_registry.is_role_held.assert_called_once()


class TestGetRoleHolder:
    """Tests for get_role_holder()."""

    @patch('app.coordination.helpers.get_registry_safe')
    def test_get_role_holder_no_registry(self, mock_get_registry):
        """Should return None when no registry."""
        mock_get_registry.return_value = None

        from app.coordination.helpers import get_role_holder
        result = get_role_holder(None)
        assert result is None


class TestCheckSpawnAllowed:
    """Tests for check_spawn_allowed()."""

    @patch('app.coordination.helpers._HAS_COORDINATION', False)
    def test_check_spawn_allowed_no_coordination(self):
        """Should return (True, 'safeguards_unavailable') when no coordination."""
        from app.coordination.helpers import check_spawn_allowed

        allowed, reason = check_spawn_allowed("selfplay", "hex8_2p")
        assert allowed is True
        assert reason == "safeguards_unavailable"

    def test_check_spawn_allowed_returns_tuple(self):
        """Should return tuple of (bool, str)."""
        from app.coordination.helpers import check_spawn_allowed

        result = check_spawn_allowed("selfplay", "hex8_2p")
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestGetSafeguards:
    """Tests for get_safeguards()."""

    @patch('app.coordination.helpers._Safeguards', None)
    def test_get_safeguards_not_available(self):
        """Should return None when Safeguards not available."""
        from app.coordination.helpers import get_safeguards

        result = get_safeguards()
        assert result is None


class TestGetCurrentNodeId:
    """Tests for get_current_node_id()."""

    def test_get_current_node_id_returns_hostname(self):
        """Should return current hostname."""
        from app.coordination.helpers import get_current_node_id

        result = get_current_node_id()
        assert result == socket.gethostname()


class TestIsUnifiedLoopRunning:
    """Tests for is_unified_loop_running()."""

    @patch('app.coordination.helpers._HAS_COORDINATION', False)
    def test_is_unified_loop_running_no_coordination(self):
        """Should return False when no coordination."""
        from app.coordination.helpers import is_unified_loop_running

        result = is_unified_loop_running()
        assert result is False


class TestWarnIfOrchestratorRunning:
    """Tests for warn_if_orchestrator_running()."""

    @patch('app.coordination.helpers._HAS_COORDINATION', False)
    def test_warn_if_orchestrator_running_no_coordination(self, capsys):
        """Should not print warning when no coordination."""
        from app.coordination.helpers import warn_if_orchestrator_running

        warn_if_orchestrator_running("test_daemon")
        captured = capsys.readouterr()
        # Should not print anything when coordination unavailable
        assert "WARNING" not in captured.out


class TestQueueBackpressureFunctions:
    """Tests for queue backpressure functions."""

    def test_get_queue_types(self):
        """Should return QueueType enum or None."""
        from app.coordination.helpers import get_queue_types

        result = get_queue_types()
        assert result is None or hasattr(result, '__members__')

    @patch('app.coordination.helpers._should_throttle_production', None)
    def test_should_throttle_safe_no_function(self):
        """Should return False when function not available."""
        from app.coordination.helpers import should_throttle_safe

        result = should_throttle_safe()
        assert result is False

    @patch('app.coordination.helpers._should_stop_production', None)
    def test_should_stop_safe_no_function(self):
        """Should return False when function not available."""
        from app.coordination.helpers import should_stop_safe

        result = should_stop_safe()
        assert result is False

    @patch('app.coordination.helpers._get_throttle_factor', None)
    def test_get_throttle_factor_safe_no_function(self):
        """Should return 1.0 when function not available."""
        from app.coordination.helpers import get_throttle_factor_safe

        result = get_throttle_factor_safe()
        assert result == 1.0

    @patch('app.coordination.helpers._report_queue_depth', None)
    def test_report_queue_depth_safe_no_function(self):
        """Should not raise when function not available."""
        from app.coordination.helpers import report_queue_depth_safe

        # Should not raise
        report_queue_depth_safe(None, 100)


class TestSyncMutexFunctions:
    """Tests for sync mutex functions."""

    @patch('app.coordination.helpers._sync_lock', None)
    def test_has_sync_lock_no_lock(self):
        """Should return False when sync_lock not available."""
        from app.coordination.helpers import has_sync_lock

        result = has_sync_lock()
        assert result is False

    def test_get_sync_lock_context(self):
        """Should return context manager or None."""
        from app.coordination.helpers import get_sync_lock_context

        result = get_sync_lock_context()
        # Could be None or a context manager
        assert result is None or callable(result)

    @patch('app.coordination.helpers._acquire_sync_lock', None)
    def test_acquire_sync_lock_safe_no_function(self):
        """Should return True when function not available."""
        from app.coordination.helpers import acquire_sync_lock_safe

        result = acquire_sync_lock_safe("host1")
        assert result is True

    @patch('app.coordination.helpers._release_sync_lock', None)
    def test_release_sync_lock_safe_no_function(self):
        """Should not raise when function not available."""
        from app.coordination.helpers import release_sync_lock_safe

        # Should not raise
        release_sync_lock_safe("host1")


class TestBandwidthManagementFunctions:
    """Tests for bandwidth management functions."""

    @patch('app.coordination.helpers._request_bandwidth', None)
    def test_has_bandwidth_manager_no_function(self):
        """Should return False when function not available."""
        from app.coordination.helpers import has_bandwidth_manager

        result = has_bandwidth_manager()
        assert result is False

    def test_get_transfer_priorities(self):
        """Should return TransferPriority enum or None."""
        from app.coordination.helpers import get_transfer_priorities

        result = get_transfer_priorities()
        assert result is None or hasattr(result, '__members__')

    @patch('app.coordination.helpers._request_bandwidth', None)
    def test_request_bandwidth_safe_no_function(self):
        """Should return (True, requested_mbps) when function not available."""
        from app.coordination.helpers import request_bandwidth_safe

        granted, mbps = request_bandwidth_safe("host1", 100.0)
        assert granted is True
        assert mbps == 100.0

    @patch('app.coordination.helpers._release_bandwidth', None)
    def test_release_bandwidth_safe_no_function(self):
        """Should not raise when function not available."""
        from app.coordination.helpers import release_bandwidth_safe

        # Should not raise
        release_bandwidth_safe("host1")

    @patch('app.coordination.helpers._bandwidth_allocation', None)
    def test_get_bandwidth_context_no_function(self):
        """Should return None when function not available."""
        from app.coordination.helpers import get_bandwidth_context

        result = get_bandwidth_context()
        assert result is None


class TestDurationSchedulingFunctions:
    """Tests for duration scheduling functions."""

    @patch('app.coordination.helpers._can_schedule_task', None)
    def test_has_duration_scheduler_no_function(self):
        """Should return False when function not available."""
        from app.coordination.helpers import has_duration_scheduler

        result = has_duration_scheduler()
        assert result is False

    @patch('app.coordination.helpers._can_schedule_task', None)
    def test_can_schedule_task_safe_no_function(self):
        """Should return True when function not available."""
        from app.coordination.helpers import can_schedule_task_safe

        result = can_schedule_task_safe("selfplay", 60.0)
        assert result is True

    @patch('app.coordination.helpers._register_running_task', None)
    def test_register_running_task_safe_no_function(self):
        """Should return False when function not available."""
        from app.coordination.helpers import register_running_task_safe

        result = register_running_task_safe("task1", "selfplay", 60.0)
        assert result is False

    @patch('app.coordination.helpers._record_task_completion', None)
    def test_record_task_completion_safe_no_function(self):
        """Should return False when function not available."""
        from app.coordination.helpers import record_task_completion_safe

        result = record_task_completion_safe("task1", "selfplay", 120.0)
        assert result is False

    @patch('app.coordination.helpers._estimate_task_duration', None)
    def test_estimate_duration_safe_no_function(self):
        """Should return default when function not available."""
        from app.coordination.helpers import estimate_duration_safe

        result = estimate_duration_safe("selfplay", default=120.0)
        assert result == 120.0


class TestCrossProcessEventFunctions:
    """Tests for cross-process event functions."""

    @patch('app.coordination.helpers._publish_event', None)
    def test_has_cross_process_events_no_function(self):
        """Should return False when function not available."""
        from app.coordination.helpers import has_cross_process_events

        result = has_cross_process_events()
        assert result is False

    @patch('app.coordination.helpers._publish_event', None)
    def test_publish_event_safe_no_function(self):
        """Should return False when function not available."""
        from app.coordination.helpers import publish_event_safe

        result = publish_event_safe("TEST_EVENT", {"key": "value"})
        assert result is False

    @patch('app.coordination.helpers._poll_events', None)
    def test_poll_events_safe_no_function(self):
        """Should return empty list when function not available."""
        from app.coordination.helpers import poll_events_safe

        result = poll_events_safe()
        assert result == []

    @patch('app.coordination.helpers._ack_event', None)
    def test_ack_event_safe_no_function(self):
        """Should return False when function not available."""
        from app.coordination.helpers import ack_event_safe

        result = ack_event_safe(123)
        assert result is False

    @patch('app.coordination.helpers._subscribe_process', None)
    def test_subscribe_process_safe_no_function(self):
        """Should return False when function not available."""
        from app.coordination.helpers import subscribe_process_safe

        result = subscribe_process_safe("test_process")
        assert result is False

    @patch('app.coordination.helpers._CrossProcessEventPoller', None)
    def test_get_event_poller_class_no_class(self):
        """Should return None when class not available."""
        from app.coordination.helpers import get_event_poller_class

        result = get_event_poller_class()
        assert result is None


class TestResourceTargetsFunctions:
    """Tests for resource targets functions."""

    @patch('app.coordination.helpers._get_resource_targets', None)
    def test_has_resource_targets_no_function(self):
        """Should return False when function not available."""
        from app.coordination.helpers import has_resource_targets

        result = has_resource_targets()
        assert result is False

    @patch('app.coordination.helpers._get_resource_targets', None)
    def test_get_resource_targets_safe_no_function(self):
        """Should return None when function not available."""
        from app.coordination.helpers import get_resource_targets_safe

        result = get_resource_targets_safe()
        assert result is None

    @patch('app.coordination.helpers._get_host_targets', None)
    def test_get_host_targets_safe_no_function(self):
        """Should return None when function not available."""
        from app.coordination.helpers import get_host_targets_safe

        result = get_host_targets_safe("host1")
        assert result is None

    @patch('app.coordination.helpers._get_cluster_summary', None)
    def test_get_cluster_summary_safe_no_function(self):
        """Should return empty dict when function not available."""
        from app.coordination.helpers import get_cluster_summary_safe

        result = get_cluster_summary_safe()
        assert result == {}

    @patch('app.coordination.helpers._should_scale_up_targets', None)
    def test_should_scale_up_safe_no_function(self):
        """Should return False when function not available."""
        from app.coordination.helpers import should_scale_up_safe

        result = should_scale_up_safe("host1")
        assert result is False

    @patch('app.coordination.helpers._should_scale_down_targets', None)
    def test_should_scale_down_safe_no_function(self):
        """Should return False when function not available."""
        from app.coordination.helpers import should_scale_down_safe

        result = should_scale_down_safe("host1")
        assert result is False

    @patch('app.coordination.helpers._set_backpressure', None)
    def test_set_backpressure_safe_no_function(self):
        """Should not raise when function not available."""
        from app.coordination.helpers import set_backpressure_safe

        # Should not raise
        set_backpressure_safe(True)


class TestModuleExports:
    """Tests for module-level re-exports."""

    def test_tasktype_exported(self):
        """TaskType should be exported (may be None)."""
        from app.coordination.helpers import TaskType
        # Just verify import doesn't fail

    def test_orchestratorrole_exported(self):
        """OrchestratorRole should be exported (may be None)."""
        from app.coordination.helpers import OrchestratorRole
        # Just verify import doesn't fail

    def test_all_exports_complete(self):
        """Should have all documented exports in __all__."""
        from app.coordination import helpers

        # Check that __all__ is defined
        assert hasattr(helpers, '__all__')

        # Check that all items in __all__ are actually exported
        for name in helpers.__all__:
            assert hasattr(helpers, name), f"Missing export: {name}"

    def test_core_safe_functions_exported(self):
        """Core safe wrapper functions should be exported."""
        from app.coordination.helpers import (
            has_coordination,
            get_coordinator_safe,
            can_spawn_safe,
            register_task_safe,
            complete_task_safe,
            fail_task_safe,
            get_registry_safe,
            acquire_role_safe,
            release_role_safe,
            has_role,
            check_spawn_allowed,
            get_current_node_id,
        )
        # All imports should succeed
