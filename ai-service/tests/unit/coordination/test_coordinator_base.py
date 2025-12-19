"""Tests for coordinator_base.py - base classes and registry for coordinators."""

import pytest
import asyncio
from unittest.mock import MagicMock, patch


class TestCoordinatorBaseImports:
    """Test that coordinator_base provides correct imports."""

    def test_import_enums_and_dataclasses(self):
        """Test importing enums and dataclasses."""
        from app.coordination.coordinator_base import (
            CoordinatorStatus,
            CoordinatorStats,
        )
        assert CoordinatorStatus is not None
        assert CoordinatorStats is not None

    def test_import_protocols(self):
        """Test importing protocols."""
        from app.coordination.coordinator_base import (
            CoordinatorProtocol,
        )
        assert CoordinatorProtocol is not None

    def test_import_base_classes(self):
        """Test importing base classes."""
        from app.coordination.coordinator_base import (
            CoordinatorBase,
        )
        assert CoordinatorBase is not None

    def test_import_mixins(self):
        """Test importing mixins."""
        from app.coordination.coordinator_base import (
            SQLitePersistenceMixin,
            SingletonMixin,
            CallbackMixin,
        )
        assert SQLitePersistenceMixin is not None
        assert SingletonMixin is not None
        assert CallbackMixin is not None

    def test_import_registry(self):
        """Test importing registry."""
        from app.coordination.coordinator_base import (
            CoordinatorRegistry,
            get_coordinator_registry,
        )
        assert CoordinatorRegistry is not None
        assert get_coordinator_registry is not None

    def test_import_helper_functions(self):
        """Test importing new helper functions."""
        from app.coordination.coordinator_base import (
            get_all_coordinators,
            get_coordinator_statuses,
            shutdown_all_coordinators,
        )
        assert callable(get_all_coordinators)
        assert callable(get_coordinator_statuses)
        assert callable(shutdown_all_coordinators)


class TestCoordinatorStatus:
    """Test CoordinatorStatus enum."""

    def test_status_values_exist(self):
        """Test that expected status values exist."""
        from app.coordination.coordinator_base import CoordinatorStatus

        assert hasattr(CoordinatorStatus, 'INITIALIZING')
        assert hasattr(CoordinatorStatus, 'READY')
        assert hasattr(CoordinatorStatus, 'RUNNING')
        assert hasattr(CoordinatorStatus, 'PAUSED')
        assert hasattr(CoordinatorStatus, 'DRAINING')
        assert hasattr(CoordinatorStatus, 'ERROR')
        assert hasattr(CoordinatorStatus, 'STOPPED')

    def test_status_is_string_enum(self):
        """Test that status values are strings."""
        from app.coordination.coordinator_base import CoordinatorStatus

        assert CoordinatorStatus.RUNNING.value == "running"
        assert CoordinatorStatus.STOPPED.value == "stopped"


class TestCoordinatorStats:
    """Test CoordinatorStats dataclass."""

    def test_default_values(self):
        """Test default values."""
        from app.coordination.coordinator_base import (
            CoordinatorStats,
            CoordinatorStatus,
        )

        stats = CoordinatorStats()
        assert stats.status == CoordinatorStatus.INITIALIZING
        assert stats.uptime_seconds == 0.0
        assert stats.operations_count == 0
        assert stats.errors_count == 0

    def test_to_dict(self):
        """Test converting to dict."""
        from app.coordination.coordinator_base import (
            CoordinatorStats,
            CoordinatorStatus,
        )

        stats = CoordinatorStats(
            status=CoordinatorStatus.RUNNING,
            uptime_seconds=120.5,
            operations_count=50,
        )

        data = stats.to_dict()
        assert isinstance(data, dict)
        assert data["status"] == "running"
        assert data["uptime_seconds"] == 120.5
        assert data["operations_count"] == 50


class TestGetAllCoordinators:
    """Test get_all_coordinators function."""

    def test_returns_dict(self):
        """Test that function returns a dict."""
        from app.coordination.coordinator_base import get_all_coordinators

        result = get_all_coordinators()
        assert isinstance(result, dict)

    def test_returns_copy(self):
        """Test that function returns a copy, not the original."""
        from app.coordination.coordinator_base import (
            get_all_coordinators,
            get_coordinator_registry,
        )

        result1 = get_all_coordinators()
        result2 = get_all_coordinators()

        # Should be equal but not the same object
        assert result1 == result2
        # Modifying one shouldn't affect the other
        result1["test"] = "value"
        assert "test" not in result2


class TestGetCoordinatorStatuses:
    """Test get_coordinator_statuses function."""

    def test_returns_dict(self):
        """Test that function returns a dict."""
        from app.coordination.coordinator_base import get_coordinator_statuses

        result = get_coordinator_statuses()
        assert isinstance(result, dict)


class TestCoordinatorRegistry:
    """Test CoordinatorRegistry class."""

    def test_singleton_pattern(self):
        """Test that get_coordinator_registry returns singleton."""
        from app.coordination.coordinator_base import get_coordinator_registry

        r1 = get_coordinator_registry()
        r2 = get_coordinator_registry()
        assert r1 is r2

    def test_get_health_summary(self):
        """Test getting health summary."""
        from app.coordination.coordinator_base import get_coordinator_registry

        registry = get_coordinator_registry()
        summary = registry.get_health_summary()

        assert isinstance(summary, dict)
        assert "healthy" in summary
        assert "coordinator_count" in summary
        assert "coordinators" in summary
