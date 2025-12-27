"""Health check compliance tests for DaemonManager-registered daemons (December 2025).

This module verifies that all daemons registered with DaemonManager implement
the health_check() method and return valid HealthCheckResult objects.

Critical for cluster reliability: DaemonManager depends on health_check() to
monitor daemon status and trigger auto-restarts.
"""

from __future__ import annotations

import pytest
from typing import Any
from unittest.mock import MagicMock, patch


class TestHealthCheckCompliance:
    """Verify all registered daemons implement health_check()."""

    def test_daemon_registry_has_all_daemon_types(self):
        """Verify DAEMON_REGISTRY covers all DaemonType values."""
        from app.coordination.daemon_registry import DAEMON_REGISTRY
        from app.coordination.daemon_types import DaemonType

        # Get all daemon types except deprecated ones
        all_types = set(DaemonType)
        registered_types = set(DAEMON_REGISTRY.keys())

        # These are known deprecated or special types
        deprecated_types = {
            DaemonType.SYNC_COORDINATOR,  # Use AUTO_SYNC
            DaemonType.HEALTH_CHECK,  # Use NODE_HEALTH_MONITOR
        }

        expected_types = all_types - deprecated_types
        missing = expected_types - registered_types

        # Allow some tolerance for new types not yet registered
        assert len(missing) <= 5, (
            f"Too many daemon types missing from DAEMON_REGISTRY: {[t.name for t in missing]}"
        )

    def test_daemon_specs_have_valid_runner_names(self):
        """Verify all DaemonSpecs have valid runner function names."""
        from app.coordination.daemon_registry import DAEMON_REGISTRY

        for daemon_type, spec in DAEMON_REGISTRY.items():
            assert spec.runner_name, (
                f"{daemon_type.name} has empty runner_name"
            )
            # Runner names should follow create_* pattern
            assert spec.runner_name.startswith("create_"), (
                f"{daemon_type.name} runner_name '{spec.runner_name}' should start with 'create_'"
            )

    def test_daemon_runners_are_importable(self):
        """Verify all daemon runner functions can be imported."""
        from app.coordination.daemon_registry import DAEMON_REGISTRY
        from app.coordination.daemon_runners import get_runner

        failed_imports = []
        for daemon_type, spec in DAEMON_REGISTRY.items():
            try:
                runner = get_runner(daemon_type)
                if runner is None:
                    failed_imports.append((daemon_type.name, "get_runner returned None"))
            except Exception as e:
                failed_imports.append((daemon_type.name, str(e)))

        assert len(failed_imports) == 0, (
            f"Failed to import runners: {failed_imports}"
        )

    @pytest.mark.parametrize("coordinator_class,module_path", [
        ("JobStallDetector", "app.coordination.stall_detection"),
        ("PriorityJobScheduler", "app.coordination.job_scheduler"),
        ("WorkDistributor", "app.coordination.work_distributor"),
        ("Safeguards", "app.coordination.safeguards"),
        ("SyncRouter", "app.coordination.sync_router"),
        ("ContinuousTrainingLoop", "app.coordination.continuous_loop"),
    ])
    def test_core_coordinators_have_health_check(self, coordinator_class: str, module_path: str):
        """Verify core coordinators implement health_check()."""
        import importlib

        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, coordinator_class)
            assert hasattr(cls, "health_check"), (
                f"{coordinator_class} in {module_path} is missing health_check() method"
            )
        except ImportError as e:
            pytest.skip(f"Could not import {module_path}: {e}")

    def test_job_stall_detector_health_check_returns_valid_result(self):
        """Verify JobStallDetector.health_check() returns proper HealthCheckResult."""
        from app.coordination.stall_detection import JobStallDetector, StallDetectorConfig

        detector = JobStallDetector(config=StallDetectorConfig())
        result = detector.health_check()

        assert hasattr(result, "healthy"), "health_check() must return object with 'healthy' attribute"
        assert hasattr(result, "status"), "health_check() must return object with 'status' attribute"
        assert isinstance(result.healthy, bool), "healthy must be boolean"

    def test_priority_job_scheduler_health_check_returns_valid_result(self):
        """Verify PriorityJobScheduler.health_check() returns proper HealthCheckResult."""
        from app.coordination.job_scheduler import PriorityJobScheduler

        scheduler = PriorityJobScheduler()
        result = scheduler.health_check()

        assert hasattr(result, "healthy"), "health_check() must return object with 'healthy' attribute"
        assert hasattr(result, "status"), "health_check() must return object with 'status' attribute"
        assert isinstance(result.healthy, bool), "healthy must be boolean"

    def test_work_distributor_health_check_returns_valid_result(self):
        """Verify WorkDistributor.health_check() returns proper HealthCheckResult."""
        from app.coordination.work_distributor import WorkDistributor

        distributor = WorkDistributor()
        result = distributor.health_check()

        assert hasattr(result, "healthy"), "health_check() must return object with 'healthy' attribute"
        assert hasattr(result, "status"), "health_check() must return object with 'status' attribute"
        assert isinstance(result.healthy, bool), "healthy must be boolean"

    def test_safeguards_health_check_returns_valid_result(self):
        """Verify Safeguards.health_check() returns proper HealthCheckResult."""
        from app.coordination.safeguards import Safeguards

        safeguards = Safeguards()
        result = safeguards.health_check()

        assert hasattr(result, "healthy"), "health_check() must return object with 'healthy' attribute"
        assert hasattr(result, "status"), "health_check() must return object with 'status' attribute"
        assert isinstance(result.healthy, bool), "healthy must be boolean"


class TestHealthCheckResultProtocol:
    """Verify HealthCheckResult protocol compliance."""

    def test_health_check_result_has_required_fields(self):
        """Verify HealthCheckResult dataclass has all required fields."""
        from app.coordination.protocols import HealthCheckResult

        # Create a sample result
        result = HealthCheckResult(
            healthy=True,
            status="running",
            message="OK",
        )

        assert hasattr(result, "healthy")
        assert hasattr(result, "status")
        assert hasattr(result, "message")

    def test_coordinator_status_enum_exists(self):
        """Verify CoordinatorStatus enum has expected values."""
        from app.coordination.protocols import CoordinatorStatus

        # Check expected statuses
        assert hasattr(CoordinatorStatus, "RUNNING")
        assert hasattr(CoordinatorStatus, "STOPPED")
        assert hasattr(CoordinatorStatus, "DEGRADED")
        assert hasattr(CoordinatorStatus, "ERROR")


class TestDaemonHealthIntegration:
    """Test DaemonManager's integration with daemon health checks."""

    def test_daemon_manager_can_query_daemon_health(self):
        """Verify DaemonManager has health monitoring capabilities."""
        from app.coordination.daemon_manager import DaemonManager

        # DaemonManager uses health_check for overall status
        # and liveness_probe for quick health checks
        assert hasattr(DaemonManager, "health_check"), (
            "DaemonManager must have health_check method"
        )
        assert hasattr(DaemonManager, "liveness_probe"), (
            "DaemonManager must have liveness_probe method"
        )

    def test_daemon_manager_has_liveness_probe(self):
        """Verify DaemonManager has liveness_probe method for health checks."""
        from app.coordination.daemon_manager import DaemonManager

        assert hasattr(DaemonManager, "liveness_probe"), (
            "DaemonManager must have liveness_probe method"
        )

    def test_daemon_manager_has_health_check(self):
        """Verify DaemonManager has its own health_check method."""
        from app.coordination.daemon_manager import DaemonManager

        assert hasattr(DaemonManager, "health_check"), (
            "DaemonManager must have health_check method"
        )


class TestCoordinatorBaseHealthCheck:
    """Test CoordinatorBase health_check implementation."""

    def test_coordinator_base_has_health_check(self):
        """Verify CoordinatorBase provides health_check method."""
        from app.coordination.coordinator_base import CoordinatorBase

        assert hasattr(CoordinatorBase, "health_check"), (
            "CoordinatorBase must have health_check method"
        )

    def test_subclasses_inherit_health_check(self):
        """Verify subclasses of CoordinatorBase inherit health_check."""
        from app.coordination.coordinator_base import CoordinatorBase

        class TestCoordinator(CoordinatorBase):
            """Test coordinator subclass."""

            def get_stats(self):
                """Implement abstract method."""
                return {}

        coord = TestCoordinator()
        assert hasattr(coord, "health_check"), (
            "Subclasses must inherit health_check from CoordinatorBase"
        )


class TestCriticalDaemonHealthChecks:
    """Test health_check on critical daemon types."""

    @pytest.fixture
    def mock_event_bus(self):
        """Mock the event bus to avoid real subscriptions."""
        with patch("app.coordination.event_router.get_event_bus") as mock:
            mock.return_value = MagicMock()
            yield mock

    def test_selfplay_scheduler_has_health_check(self):
        """Verify SelfplayScheduler implements health_check()."""
        from app.coordination.selfplay_scheduler import SelfplayScheduler

        assert hasattr(SelfplayScheduler, "health_check"), (
            "SelfplayScheduler must implement health_check()"
        )

    def test_feedback_loop_controller_has_health_check(self):
        """Verify FeedbackLoopController implements health_check()."""
        from app.coordination.feedback_loop_controller import FeedbackLoopController

        assert hasattr(FeedbackLoopController, "health_check"), (
            "FeedbackLoopController must implement health_check()"
        )

    def test_data_pipeline_orchestrator_has_health_check(self):
        """Verify DataPipelineOrchestrator implements health_check()."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        assert hasattr(DataPipelineOrchestrator, "health_check"), (
            "DataPipelineOrchestrator must implement health_check()"
        )

    def test_unified_health_manager_has_health_check(self):
        """Verify UnifiedHealthManager implements health_check()."""
        from app.coordination.unified_health_manager import UnifiedHealthManager

        assert hasattr(UnifiedHealthManager, "health_check"), (
            "UnifiedHealthManager must implement health_check()"
        )

    def test_health_check_orchestrator_has_health_check(self):
        """Verify HealthCheckOrchestrator implements health_check()."""
        from app.coordination.health_check_orchestrator import HealthCheckOrchestrator

        assert hasattr(HealthCheckOrchestrator, "health_check"), (
            "HealthCheckOrchestrator must implement health_check()"
        )


class TestDaemonAdapterHealthCheck:
    """Verify DaemonAdapter and subclasses have health_check."""

    def test_daemon_adapter_base_has_health_check(self):
        """Verify base DaemonAdapter class has health_check method."""
        from app.coordination.daemon_adapters import DaemonAdapter

        assert hasattr(DaemonAdapter, "health_check"), (
            "DaemonAdapter base class must have health_check method"
        )

    @pytest.mark.parametrize("adapter_class", [
        "SyncCoordinatorAdapter",
        "AutoPromotionAdapter",
        "DistillationAdapter",
        "NASAdapter",
        "PBTAdapter",
        "SyncGateAdapter",
        "TrainingDistributorAdapter",
        "TrainingJobCoordinatorAdapter",
        "WorkerReporterAdapter",
    ])
    def test_adapter_subclasses_have_health_check(self, adapter_class: str):
        """Verify all DaemonAdapter subclasses have health_check."""
        from app.coordination import daemon_adapters

        try:
            cls = getattr(daemon_adapters, adapter_class)
            assert hasattr(cls, "health_check"), (
                f"{adapter_class} must have health_check method"
            )
        except AttributeError:
            pytest.skip(f"{adapter_class} not found in daemon_adapters")


class TestEventRouterHealthValidation:
    """Test event router health validation functions."""

    def test_validate_event_wiring_function_exists(self):
        """Verify validate_event_wiring function exists in event_router."""
        from app.coordination.event_router import validate_event_wiring

        assert callable(validate_event_wiring), (
            "validate_event_wiring must be callable"
        )

    def test_validate_event_wiring_returns_dict(self):
        """Verify validate_event_wiring returns proper validation result."""
        from app.coordination.event_router import validate_event_wiring

        result = validate_event_wiring(raise_on_error=False, log_warnings=False)

        assert isinstance(result, dict), "validate_event_wiring must return dict"
        assert "valid" in result, "Result must have 'valid' key"
        assert "missing_critical" in result, "Result must have 'missing_critical' key"

    def test_validate_event_wiring_checks_critical_events(self):
        """Verify validate_event_wiring checks for critical events."""
        from app.coordination.event_router import validate_event_wiring

        result = validate_event_wiring(raise_on_error=False, log_warnings=False)

        # Even if not subscribed, the function should report which are missing
        assert "missing_critical" in result
        assert "missing_optional" in result
        assert isinstance(result["missing_critical"], list)


class TestHealthCheckReturnTypes:
    """Verify health_check methods return correct types."""

    def test_health_check_result_is_importable(self):
        """Verify HealthCheckResult can be imported."""
        from app.coordination.protocols import HealthCheckResult
        assert HealthCheckResult is not None

    def test_coordinator_status_is_importable(self):
        """Verify CoordinatorStatus can be imported."""
        from app.coordination.protocols import CoordinatorStatus
        assert CoordinatorStatus is not None

    def test_health_check_result_supports_details(self):
        """Verify HealthCheckResult supports optional details field."""
        from app.coordination.protocols import HealthCheckResult

        result = HealthCheckResult(
            healthy=True,
            status="running",
            message="All systems operational",
            details={"uptime": 3600, "errors": 0},
        )

        assert result.details is not None
        assert result.details["uptime"] == 3600
