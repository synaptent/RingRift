"""
Tests for app.coordination.orchestrator_registry module.

Tests the SQLite-based orchestrator coordination system including:
- OrchestratorRole and OrchestratorState enums
- OrchestratorInfo dataclass
- OrchestratorRegistry singleton
- CrossCoordinatorHealthProtocol
- Coordinator registration functions
"""

import os
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.orchestrator_registry import (
    HEARTBEAT_INTERVAL_SECONDS,
    HEARTBEAT_TIMEOUT_SECONDS,
    REGISTRY_DB,
    # Module-level constants
    REGISTRY_DIR,
    CoordinatorHealth,
    CrossCoordinatorHealthProtocol,
    # Dataclasses
    OrchestratorInfo,
    # Main class
    OrchestratorRegistry,
    # Enums
    OrchestratorRole,
    OrchestratorState,
    acquire_orchestrator_role,
    auto_register_known_coordinators,
    check_cluster_health,
    # Discovery
    discover_and_register_orchestrators,
    get_coordinator,
    # Health functions
    get_cross_coordinator_health,
    get_orchestrator_inventory,
    get_registered_coordinators,
    # Convenience functions
    get_registry,
    is_orchestrator_role_available,
    orchestrator_role,
    # Coordinator registration
    register_coordinator,
    release_orchestrator_role,
    shutdown_all_coordinators,
    unregister_coordinator,
)

# ============================================
# Test Fixtures
# ============================================

@pytest.fixture
def temp_registry_dir(tmp_path):
    """Create a temporary registry directory for testing."""
    registry_dir = tmp_path / "ringrift_coordination"
    registry_dir.mkdir(parents=True, exist_ok=True)
    return registry_dir


@pytest.fixture
def mock_registry(temp_registry_dir, monkeypatch):
    """Create a mock registry instance with isolated database."""
    # Patch the module-level REGISTRY_DIR and REGISTRY_DB
    monkeypatch.setattr(
        "app.coordination.orchestrator_registry.REGISTRY_DIR",
        temp_registry_dir
    )
    monkeypatch.setattr(
        "app.coordination.orchestrator_registry.REGISTRY_DB",
        temp_registry_dir / "orchestrator_registry.db"
    )

    # Clear singleton
    OrchestratorRegistry._instance = None

    # Create fresh instance
    registry = OrchestratorRegistry()
    registry._db_path = temp_registry_dir / "orchestrator_registry.db"

    yield registry

    # Cleanup
    registry.release_role()
    OrchestratorRegistry._instance = None


@pytest.fixture
def fresh_coordinator_registry(monkeypatch):
    """Reset the coordinator registry for testing."""
    import app.coordination.orchestrator_registry as module

    # Clear the coordinator registry
    original_registry = module._coordinator_registry.copy()
    module._coordinator_registry.clear()

    yield module._coordinator_registry

    # Restore
    module._coordinator_registry.clear()
    module._coordinator_registry.update(original_registry)


# ============================================
# Test OrchestratorRole Enum
# ============================================

class TestOrchestratorRole:
    """Tests for OrchestratorRole enum."""

    def test_cluster_orchestrator_value(self):
        assert OrchestratorRole.CLUSTER_ORCHESTRATOR.value == "cluster_orchestrator"

    def test_improvement_daemon_value(self):
        assert OrchestratorRole.IMPROVEMENT_DAEMON.value == "improvement_daemon"

    def test_pipeline_orchestrator_value(self):
        assert OrchestratorRole.PIPELINE_ORCHESTRATOR.value == "pipeline_orchestrator"

    def test_p2p_leader_value(self):
        assert OrchestratorRole.P2P_LEADER.value == "p2p_leader"

    def test_tournament_runner_value(self):
        assert OrchestratorRole.TOURNAMENT_RUNNER.value == "tournament_runner"

    def test_model_sync_value(self):
        assert OrchestratorRole.MODEL_SYNC.value == "model_sync"

    def test_data_sync_value(self):
        assert OrchestratorRole.DATA_SYNC.value == "data_sync"

    def test_unified_loop_value(self):
        assert OrchestratorRole.UNIFIED_LOOP.value == "unified_loop"

    def test_all_roles_defined(self):
        """Ensure all expected roles are defined."""
        expected_roles = {
            "cluster_orchestrator",
            "improvement_daemon",
            "pipeline_orchestrator",
            "p2p_leader",
            "tournament_runner",
            "model_sync",
            "data_sync",
            "unified_loop",
        }
        actual_roles = {role.value for role in OrchestratorRole}
        assert actual_roles == expected_roles


# ============================================
# Test OrchestratorState Enum
# ============================================

class TestOrchestratorState:
    """Tests for OrchestratorState enum."""

    def test_starting_value(self):
        assert OrchestratorState.STARTING.value == "starting"

    def test_running_value(self):
        assert OrchestratorState.RUNNING.value == "running"

    def test_stopping_value(self):
        assert OrchestratorState.STOPPING.value == "stopping"

    def test_stopped_value(self):
        assert OrchestratorState.STOPPED.value == "stopped"

    def test_dead_value(self):
        assert OrchestratorState.DEAD.value == "dead"

    def test_all_states_defined(self):
        """Ensure all expected states are defined."""
        expected_states = {"starting", "running", "stopping", "stopped", "dead"}
        actual_states = {state.value for state in OrchestratorState}
        assert actual_states == expected_states


# ============================================
# Test OrchestratorInfo Dataclass
# ============================================

class TestOrchestratorInfo:
    """Tests for OrchestratorInfo dataclass."""

    def test_creation(self):
        now = datetime.now().isoformat()
        info = OrchestratorInfo(
            id="test-id",
            role="cluster_orchestrator",
            hostname="localhost",
            pid=12345,
            state="running",
            started_at=now,
            last_heartbeat=now,
            metadata={"key": "value"},
        )

        assert info.id == "test-id"
        assert info.role == "cluster_orchestrator"
        assert info.hostname == "localhost"
        assert info.pid == 12345
        assert info.state == "running"
        assert info.started_at == now
        assert info.last_heartbeat == now
        assert info.metadata == {"key": "value"}

    def test_is_alive_running(self):
        """Test is_alive returns True for recent heartbeat."""
        now = datetime.now().isoformat()
        info = OrchestratorInfo(
            id="test-id",
            role="cluster_orchestrator",
            hostname="localhost",
            pid=12345,
            state="running",
            started_at=now,
            last_heartbeat=now,
            metadata={},
        )

        assert info.is_alive() is True

    def test_is_alive_stopped(self):
        """Test is_alive returns False for stopped state."""
        now = datetime.now().isoformat()
        info = OrchestratorInfo(
            id="test-id",
            role="cluster_orchestrator",
            hostname="localhost",
            pid=12345,
            state="stopped",
            started_at=now,
            last_heartbeat=now,
            metadata={},
        )

        assert info.is_alive() is False

    def test_is_alive_dead(self):
        """Test is_alive returns False for dead state."""
        now = datetime.now().isoformat()
        info = OrchestratorInfo(
            id="test-id",
            role="cluster_orchestrator",
            hostname="localhost",
            pid=12345,
            state="dead",
            started_at=now,
            last_heartbeat=now,
            metadata={},
        )

        assert info.is_alive() is False

    def test_is_alive_stale_heartbeat(self):
        """Test is_alive returns False for stale heartbeat."""
        old_time = (datetime.now() - timedelta(seconds=HEARTBEAT_TIMEOUT_SECONDS + 10)).isoformat()
        info = OrchestratorInfo(
            id="test-id",
            role="cluster_orchestrator",
            hostname="localhost",
            pid=12345,
            state="running",
            started_at=old_time,
            last_heartbeat=old_time,
            metadata={},
        )

        assert info.is_alive() is False

    def test_is_alive_invalid_heartbeat(self):
        """Test is_alive handles invalid heartbeat timestamp."""
        info = OrchestratorInfo(
            id="test-id",
            role="cluster_orchestrator",
            hostname="localhost",
            pid=12345,
            state="running",
            started_at="invalid",
            last_heartbeat="invalid",
            metadata={},
        )

        assert info.is_alive() is False

    def test_to_dict(self):
        """Test to_dict serialization."""
        now = datetime.now().isoformat()
        info = OrchestratorInfo(
            id="test-id",
            role="cluster_orchestrator",
            hostname="localhost",
            pid=12345,
            state="running",
            started_at=now,
            last_heartbeat=now,
            metadata={"key": "value"},
        )

        result = info.to_dict()
        assert result == {
            "id": "test-id",
            "role": "cluster_orchestrator",
            "hostname": "localhost",
            "pid": 12345,
            "state": "running",
            "started_at": now,
            "last_heartbeat": now,
            "metadata": {"key": "value"},
        }


# ============================================
# Test CoordinatorHealth Dataclass
# ============================================

class TestCoordinatorHealth:
    """Tests for CoordinatorHealth dataclass."""

    def test_minimal_creation(self):
        health = CoordinatorHealth(
            coordinator_id="coord-1",
            role="cluster_orchestrator",
            is_healthy=True,
            last_seen=datetime.now().isoformat(),
        )

        assert health.coordinator_id == "coord-1"
        assert health.role == "cluster_orchestrator"
        assert health.is_healthy is True
        assert health.response_time_ms is None
        assert health.error is None
        assert health.metadata is None

    def test_full_creation(self):
        now = datetime.now().isoformat()
        health = CoordinatorHealth(
            coordinator_id="coord-1",
            role="cluster_orchestrator",
            is_healthy=False,
            last_seen=now,
            response_time_ms=150.5,
            error="Connection timeout",
            metadata={"attempts": 3},
        )

        assert health.coordinator_id == "coord-1"
        assert health.is_healthy is False
        assert health.response_time_ms == 150.5
        assert health.error == "Connection timeout"
        assert health.metadata == {"attempts": 3}

    def test_to_dict(self):
        now = datetime.now().isoformat()
        health = CoordinatorHealth(
            coordinator_id="coord-1",
            role="cluster_orchestrator",
            is_healthy=True,
            last_seen=now,
            response_time_ms=50.0,
            metadata={"key": "value"},
        )

        result = health.to_dict()
        assert result["coordinator_id"] == "coord-1"
        assert result["role"] == "cluster_orchestrator"
        assert result["is_healthy"] is True
        assert result["last_seen"] == now
        assert result["response_time_ms"] == 50.0
        assert result["error"] is None
        assert result["metadata"] == {"key": "value"}


# ============================================
# Test OrchestratorRegistry
# ============================================

class TestOrchestratorRegistry:
    """Tests for OrchestratorRegistry class."""

    def test_singleton_pattern(self, mock_registry):
        """Test that get_instance returns same instance."""
        OrchestratorRegistry._instance = mock_registry
        instance1 = OrchestratorRegistry.get_instance()
        instance2 = OrchestratorRegistry.get_instance()
        assert instance1 is instance2

    def test_init_creates_db(self, temp_registry_dir, monkeypatch):
        """Test that initialization creates database."""
        monkeypatch.setattr(
            "app.coordination.orchestrator_registry.REGISTRY_DIR",
            temp_registry_dir
        )
        monkeypatch.setattr(
            "app.coordination.orchestrator_registry.REGISTRY_DB",
            temp_registry_dir / "orchestrator_registry.db"
        )
        OrchestratorRegistry._instance = None

        registry = OrchestratorRegistry()
        registry._db_path = temp_registry_dir / "orchestrator_registry.db"

        assert registry._db_path.exists()

        # Clean up
        registry.release_role()
        OrchestratorRegistry._instance = None

    def test_acquire_role_success(self, mock_registry):
        """Test successful role acquisition."""
        result = mock_registry.acquire_role(OrchestratorRole.CLUSTER_ORCHESTRATOR)

        assert result is True
        assert mock_registry._my_role == OrchestratorRole.CLUSTER_ORCHESTRATOR
        assert mock_registry._my_id is not None

        # Clean up
        mock_registry.release_role()

    def test_acquire_role_with_metadata(self, mock_registry):
        """Test role acquisition with metadata."""
        result = mock_registry.acquire_role(
            OrchestratorRole.CLUSTER_ORCHESTRATOR,
            metadata={"version": "1.0", "mode": "test"}
        )

        assert result is True

        # Verify metadata stored
        holder = mock_registry.get_role_holder(OrchestratorRole.CLUSTER_ORCHESTRATOR)
        assert holder is not None
        assert holder.metadata.get("version") == "1.0"
        assert holder.metadata.get("mode") == "test"

        # Clean up
        mock_registry.release_role()

    def test_acquire_role_denied_when_held(self, mock_registry):
        """Test that role acquisition is denied when already held."""
        # First acquisition succeeds
        result1 = mock_registry.acquire_role(OrchestratorRole.CLUSTER_ORCHESTRATOR)
        assert result1 is True

        # Store the ID and role
        saved_id = mock_registry._my_id
        saved_role = mock_registry._my_role

        # Simulate different process trying to acquire
        mock_registry._my_id = None
        mock_registry._my_role = None

        # Second acquisition should fail (same process but pretending different)
        # The check looks at hostname + pid, so we need to mock it
        with patch("socket.gethostname", return_value="different-host"):
            result2 = mock_registry.acquire_role(OrchestratorRole.CLUSTER_ORCHESTRATOR)
            # Should fail because heartbeat is recent
            assert result2 is False

        # Restore and clean up
        mock_registry._my_id = saved_id
        mock_registry._my_role = saved_role
        mock_registry.release_role()

    def test_release_role(self, mock_registry):
        """Test role release."""
        mock_registry.acquire_role(OrchestratorRole.CLUSTER_ORCHESTRATOR)
        assert mock_registry._my_role is not None

        mock_registry.release_role()

        assert mock_registry._my_role is None
        assert mock_registry._my_id is None

    def test_release_role_when_not_held(self, mock_registry):
        """Test release when no role held."""
        # Should not raise
        mock_registry.release_role()
        assert mock_registry._my_role is None

    def test_heartbeat(self, mock_registry):
        """Test heartbeat update."""
        mock_registry.acquire_role(OrchestratorRole.CLUSTER_ORCHESTRATOR)

        # Record initial heartbeat
        holder1 = mock_registry.get_role_holder(OrchestratorRole.CLUSTER_ORCHESTRATOR)
        initial_heartbeat = holder1.last_heartbeat

        # Wait a bit and heartbeat
        time.sleep(0.01)
        mock_registry.heartbeat()

        # Check heartbeat updated
        holder2 = mock_registry.get_role_holder(OrchestratorRole.CLUSTER_ORCHESTRATOR)
        assert holder2.last_heartbeat >= initial_heartbeat

        # Clean up
        mock_registry.release_role()

    def test_heartbeat_with_metadata_update(self, mock_registry):
        """Test heartbeat with metadata update."""
        mock_registry.acquire_role(
            OrchestratorRole.CLUSTER_ORCHESTRATOR,
            metadata={"initial": True}
        )

        mock_registry.heartbeat(metadata_update={"counter": 1})

        holder = mock_registry.get_role_holder(OrchestratorRole.CLUSTER_ORCHESTRATOR)
        assert holder.metadata.get("initial") is True
        assert holder.metadata.get("counter") == 1

        # Clean up
        mock_registry.release_role()

    def test_heartbeat_when_not_registered(self, mock_registry):
        """Test heartbeat when not registered does nothing."""
        # Should not raise
        mock_registry.heartbeat()

    def test_get_active_orchestrators(self, mock_registry):
        """Test getting active orchestrators."""
        mock_registry.acquire_role(OrchestratorRole.CLUSTER_ORCHESTRATOR)

        active = mock_registry.get_active_orchestrators()

        assert len(active) == 1
        assert active[0].role == OrchestratorRole.CLUSTER_ORCHESTRATOR.value

        # Clean up
        mock_registry.release_role()

    def test_get_active_orchestrators_empty(self, mock_registry):
        """Test getting active orchestrators when none registered."""
        active = mock_registry.get_active_orchestrators()
        assert len(active) == 0

    def test_is_role_held(self, mock_registry):
        """Test is_role_held check."""
        assert mock_registry.is_role_held(OrchestratorRole.CLUSTER_ORCHESTRATOR) is False

        mock_registry.acquire_role(OrchestratorRole.CLUSTER_ORCHESTRATOR)
        assert mock_registry.is_role_held(OrchestratorRole.CLUSTER_ORCHESTRATOR) is True

        mock_registry.release_role()
        assert mock_registry.is_role_held(OrchestratorRole.CLUSTER_ORCHESTRATOR) is False

    def test_get_role_holder(self, mock_registry):
        """Test getting role holder info."""
        assert mock_registry.get_role_holder(OrchestratorRole.CLUSTER_ORCHESTRATOR) is None

        mock_registry.acquire_role(OrchestratorRole.CLUSTER_ORCHESTRATOR)

        holder = mock_registry.get_role_holder(OrchestratorRole.CLUSTER_ORCHESTRATOR)
        assert holder is not None
        assert holder.role == OrchestratorRole.CLUSTER_ORCHESTRATOR.value
        assert holder.pid == os.getpid()

        # Clean up
        mock_registry.release_role()

    def test_get_status_summary(self, mock_registry):
        """Test status summary."""
        mock_registry.acquire_role(OrchestratorRole.CLUSTER_ORCHESTRATOR)

        summary = mock_registry.get_status_summary()

        assert summary["active_count"] == 1
        assert "cluster_orchestrator" in summary["roles_held"]
        assert summary["my_role"] == "cluster_orchestrator"
        assert summary["my_id"] is not None

        # Clean up
        mock_registry.release_role()

    def test_get_recent_events(self, mock_registry):
        """Test getting recent events."""
        mock_registry.acquire_role(OrchestratorRole.CLUSTER_ORCHESTRATOR)
        mock_registry.release_role()

        events = mock_registry.get_recent_events(limit=10)

        assert len(events) >= 2  # At least ACQUIRED and RELEASED
        event_types = {e["event_type"] for e in events}
        assert "ACQUIRED" in event_types
        assert "RELEASED" in event_types

    def test_get_recent_events_by_role(self, mock_registry):
        """Test getting events filtered by role."""
        mock_registry.acquire_role(OrchestratorRole.CLUSTER_ORCHESTRATOR)
        mock_registry.release_role()

        events = mock_registry.get_recent_events(
            limit=10,
            role="cluster_orchestrator"
        )

        for event in events:
            assert event["role"] == "cluster_orchestrator"

    def test_generate_id_format(self, mock_registry):
        """Test ID generation format."""
        id_str = mock_registry._generate_id(OrchestratorRole.CLUSTER_ORCHESTRATOR)

        parts = id_str.split("_")
        assert parts[0] == "cluster"
        assert parts[1] == "orchestrator"
        # hostname_pid_timestamp


# ============================================
# Test OrchestratorRegistry Data Availability
# ============================================

class TestOrchestratorRegistryDataAvailability:
    """Tests for data availability methods."""

    def test_get_data_availability_no_catalog(self, mock_registry):
        """Test data availability when catalog not available."""
        with patch(
            "app.coordination.orchestrator_registry.HAS_DATA_CATALOG",
            False
        ):
            result = mock_registry.get_data_availability()

            assert result["available"] is False
            assert "error" in result

    def test_has_sufficient_data_no_catalog(self, mock_registry):
        """Test sufficient data check when catalog not available."""
        with patch(
            "app.coordination.orchestrator_registry.HAS_DATA_CATALOG",
            False
        ):
            result = mock_registry.has_sufficient_data(min_games=100)
            assert result is False

    def test_get_training_readiness(self, mock_registry):
        """Test training readiness check."""
        with patch(
            "app.coordination.orchestrator_registry.HAS_DATA_CATALOG",
            False
        ):
            result = mock_registry.get_training_readiness(
                config_key="square8_2p",
                min_games=100,
                min_quality=0.5
            )

            assert "ready" in result
            assert result["ready"] is False
            assert "reasons" in result
            assert "thresholds" in result
            assert result["thresholds"]["min_games"] == 100


# ============================================
# Test CrossCoordinatorHealthProtocol
# ============================================

class TestCrossCoordinatorHealthProtocol:
    """Tests for CrossCoordinatorHealthProtocol class."""

    def test_initialization(self, mock_registry):
        """Test protocol initialization."""
        protocol = CrossCoordinatorHealthProtocol(registry=mock_registry)

        assert protocol._registry is mock_registry
        assert protocol._health_check_timeout_ms == 5000.0

    def test_initialization_custom_timeout(self, mock_registry):
        """Test protocol with custom timeout."""
        protocol = CrossCoordinatorHealthProtocol(
            registry=mock_registry,
            health_check_timeout_ms=10000.0
        )

        assert protocol._health_check_timeout_ms == 10000.0

    def test_check_all_coordinators_empty(self, mock_registry):
        """Test check when no coordinators registered."""
        protocol = CrossCoordinatorHealthProtocol(registry=mock_registry)

        result = protocol.check_all_coordinators()

        assert result == {}

    def test_check_all_coordinators_with_active(self, mock_registry):
        """Test check with active coordinators."""
        mock_registry.acquire_role(OrchestratorRole.CLUSTER_ORCHESTRATOR)

        protocol = CrossCoordinatorHealthProtocol(registry=mock_registry)

        result = protocol.check_all_coordinators()

        assert len(result) == 1
        health = next(iter(result.values()))
        assert health.is_healthy is True
        assert health.role == "cluster_orchestrator"

        # Clean up
        mock_registry.release_role()

    def test_is_coordinator_healthy(self, mock_registry):
        """Test checking specific coordinator health."""
        mock_registry.acquire_role(OrchestratorRole.CLUSTER_ORCHESTRATOR)

        protocol = CrossCoordinatorHealthProtocol(registry=mock_registry)

        assert protocol.is_coordinator_healthy(OrchestratorRole.CLUSTER_ORCHESTRATOR) is True
        assert protocol.is_coordinator_healthy(OrchestratorRole.TOURNAMENT_RUNNER) is False

        # Clean up
        mock_registry.release_role()

    def test_get_healthy_coordinators(self, mock_registry):
        """Test getting list of healthy coordinators."""
        mock_registry.acquire_role(OrchestratorRole.CLUSTER_ORCHESTRATOR)

        protocol = CrossCoordinatorHealthProtocol(registry=mock_registry)

        healthy = protocol.get_healthy_coordinators()

        assert len(healthy) == 1
        assert healthy[0].role == "cluster_orchestrator"

        # Clean up
        mock_registry.release_role()

    def test_get_unhealthy_coordinators(self, mock_registry):
        """Test getting list of unhealthy coordinators."""
        protocol = CrossCoordinatorHealthProtocol(registry=mock_registry)

        unhealthy = protocol.get_unhealthy_coordinators()

        # No coordinators, so none unhealthy
        assert len(unhealthy) == 0

    def test_get_role_health(self, mock_registry):
        """Test getting health for specific role."""
        mock_registry.acquire_role(OrchestratorRole.CLUSTER_ORCHESTRATOR)

        protocol = CrossCoordinatorHealthProtocol(registry=mock_registry)

        health = protocol.get_role_health(OrchestratorRole.CLUSTER_ORCHESTRATOR)

        assert health is not None
        assert health.is_healthy is True
        assert health.role == "cluster_orchestrator"

        # Non-existent role
        health_none = protocol.get_role_health(OrchestratorRole.TOURNAMENT_RUNNER)
        assert health_none is None

        # Clean up
        mock_registry.release_role()

    def test_get_cluster_health_summary_empty(self, mock_registry):
        """Test cluster health summary with no coordinators."""
        protocol = CrossCoordinatorHealthProtocol(registry=mock_registry)

        summary = protocol.get_cluster_health_summary()

        assert summary["total_coordinators"] == 0
        assert summary["healthy_count"] == 0
        assert summary["cluster_healthy"] is False  # No healthy coordinators

    def test_get_cluster_health_summary_with_active(self, mock_registry):
        """Test cluster health summary with active coordinators."""
        mock_registry.acquire_role(OrchestratorRole.CLUSTER_ORCHESTRATOR)

        protocol = CrossCoordinatorHealthProtocol(registry=mock_registry)

        summary = protocol.get_cluster_health_summary()

        assert summary["total_coordinators"] == 1
        assert summary["healthy_count"] == 1
        assert summary["health_percentage"] == 100.0
        assert "cluster_orchestrator" in summary["by_role"]
        assert summary["cluster_healthy"] is True  # Critical role is up

        # Clean up
        mock_registry.release_role()

    def test_cache_usage(self, mock_registry):
        """Test that health checks use cache."""
        mock_registry.acquire_role(OrchestratorRole.CLUSTER_ORCHESTRATOR)

        protocol = CrossCoordinatorHealthProtocol(registry=mock_registry)
        protocol._check_cooldown_seconds = 60.0  # Long cooldown

        # First check
        result1 = protocol.check_all_coordinators()

        # Second check should use cache
        result2 = protocol.check_all_coordinators()

        # Same results from cache
        assert len(result1) == len(result2)

        # Clean up
        mock_registry.release_role()

    def test_on_coordinator_failure_callback(self, mock_registry):
        """Test registering failure callback."""
        protocol = CrossCoordinatorHealthProtocol(registry=mock_registry)

        callback = MagicMock()
        protocol.on_coordinator_failure(callback)

        assert hasattr(protocol, "_failure_callbacks")
        assert callback in protocol._failure_callbacks


# ============================================
# Test Convenience Functions
# ============================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_registry(self, mock_registry, monkeypatch):
        """Test get_registry returns singleton."""
        OrchestratorRegistry._instance = mock_registry

        result = get_registry()

        assert result is mock_registry

    def test_acquire_orchestrator_role(self, mock_registry, monkeypatch):
        """Test acquire_orchestrator_role convenience function."""
        OrchestratorRegistry._instance = mock_registry

        result = acquire_orchestrator_role(OrchestratorRole.CLUSTER_ORCHESTRATOR)

        assert result is True

        # Clean up
        release_orchestrator_role()

    def test_release_orchestrator_role(self, mock_registry, monkeypatch):
        """Test release_orchestrator_role convenience function."""
        OrchestratorRegistry._instance = mock_registry

        acquire_orchestrator_role(OrchestratorRole.CLUSTER_ORCHESTRATOR)
        release_orchestrator_role()

        assert mock_registry._my_role is None

    def test_is_orchestrator_role_available(self, mock_registry, monkeypatch):
        """Test is_orchestrator_role_available function."""
        OrchestratorRegistry._instance = mock_registry

        assert is_orchestrator_role_available(OrchestratorRole.CLUSTER_ORCHESTRATOR) is True

        acquire_orchestrator_role(OrchestratorRole.CLUSTER_ORCHESTRATOR)

        assert is_orchestrator_role_available(OrchestratorRole.CLUSTER_ORCHESTRATOR) is False

        # Clean up
        release_orchestrator_role()

    def test_orchestrator_role_context_manager(self, mock_registry, monkeypatch):
        """Test orchestrator_role context manager."""
        OrchestratorRegistry._instance = mock_registry

        with orchestrator_role(OrchestratorRole.CLUSTER_ORCHESTRATOR) as registry:
            assert registry._my_role == OrchestratorRole.CLUSTER_ORCHESTRATOR

        # Should be released after context
        assert mock_registry._my_role is None

    def test_orchestrator_role_context_manager_failure(self, mock_registry, monkeypatch):
        """Test context manager raises on failed acquisition."""
        OrchestratorRegistry._instance = mock_registry

        # Acquire the role first
        mock_registry.acquire_role(OrchestratorRole.CLUSTER_ORCHESTRATOR)

        # Try to acquire again from "different host"
        with patch("socket.gethostname", return_value="different-host"):
            with pytest.raises(RuntimeError, match="Failed to acquire"):
                with orchestrator_role(OrchestratorRole.CLUSTER_ORCHESTRATOR):
                    pass

        # Clean up
        mock_registry.release_role()


# ============================================
# Test Health Functions
# ============================================

class TestHealthFunctions:
    """Tests for health-related module functions."""

    def test_get_cross_coordinator_health(self):
        """Test get_cross_coordinator_health returns singleton."""
        protocol1 = get_cross_coordinator_health()
        protocol2 = get_cross_coordinator_health()

        assert protocol1 is protocol2

    def test_check_cluster_health(self, mock_registry, monkeypatch):
        """Test check_cluster_health function."""
        OrchestratorRegistry._instance = mock_registry

        # Reset health protocol singleton
        import app.coordination.orchestrator_registry as module
        module._health_protocol = None

        result = check_cluster_health()

        assert "total_coordinators" in result
        assert "healthy_count" in result
        assert "cluster_healthy" in result


# ============================================
# Test Coordinator Registration
# ============================================

class TestCoordinatorRegistration:
    """Tests for coordinator registration functions."""

    def test_register_coordinator(self, fresh_coordinator_registry):
        """Test registering a coordinator."""
        mock_coordinator = MagicMock()

        result = register_coordinator(
            "test_coordinator",
            mock_coordinator,
            health_callback=lambda: True,
            shutdown_callback=mock_coordinator.shutdown,
            metadata={"type": "test"},
        )

        assert result is True
        assert "test_coordinator" in fresh_coordinator_registry

    def test_unregister_coordinator(self, fresh_coordinator_registry):
        """Test unregistering a coordinator."""
        mock_coordinator = MagicMock()
        register_coordinator("test_coordinator", mock_coordinator)

        result = unregister_coordinator("test_coordinator")

        assert result is True
        assert "test_coordinator" not in fresh_coordinator_registry

    def test_unregister_nonexistent(self, fresh_coordinator_registry):
        """Test unregistering non-existent coordinator."""
        result = unregister_coordinator("nonexistent")
        assert result is False

    def test_get_coordinator(self, fresh_coordinator_registry):
        """Test getting a registered coordinator."""
        mock_coordinator = MagicMock()
        register_coordinator("test_coordinator", mock_coordinator)

        result = get_coordinator("test_coordinator")

        assert result is mock_coordinator

    def test_get_coordinator_nonexistent(self, fresh_coordinator_registry):
        """Test getting non-existent coordinator."""
        result = get_coordinator("nonexistent")
        assert result is None

    def test_get_registered_coordinators(self, fresh_coordinator_registry):
        """Test getting all registered coordinators."""
        mock1 = MagicMock()
        mock2 = MagicMock()

        register_coordinator("coord1", mock1, health_callback=lambda: True)
        register_coordinator("coord2", mock2, health_callback=lambda: False)

        result = get_registered_coordinators()

        assert len(result) == 2
        assert "coord1" in result
        assert "coord2" in result
        assert result["coord1"]["healthy"] is True
        assert result["coord2"]["healthy"] is False

    def test_get_registered_coordinators_health_exception(self, fresh_coordinator_registry):
        """Test health callback that raises exception."""
        mock_coordinator = MagicMock()

        def bad_health():
            raise RuntimeError("Health check failed")

        register_coordinator(
            "bad_coordinator",
            mock_coordinator,
            health_callback=bad_health,
        )

        result = get_registered_coordinators()

        # Should handle exception and report unhealthy
        assert result["bad_coordinator"]["healthy"] is False

    def test_shutdown_all_coordinators(self, fresh_coordinator_registry):
        """Test shutting down all coordinators."""
        mock1 = MagicMock()
        mock2 = MagicMock()

        register_coordinator("coord1", mock1, shutdown_callback=mock1.shutdown)
        register_coordinator("coord2", mock2, shutdown_callback=mock2.shutdown)

        results = shutdown_all_coordinators()

        assert results["coord1"] is True
        assert results["coord2"] is True
        mock1.shutdown.assert_called_once()
        mock2.shutdown.assert_called_once()

    def test_shutdown_with_exception(self, fresh_coordinator_registry):
        """Test shutdown handles exceptions."""
        mock_coordinator = MagicMock()
        mock_coordinator.shutdown.side_effect = RuntimeError("Shutdown failed")

        register_coordinator(
            "bad_coordinator",
            mock_coordinator,
            shutdown_callback=mock_coordinator.shutdown,
        )

        results = shutdown_all_coordinators()

        assert results["bad_coordinator"] is False


# ============================================
# Test Discovery Functions
# ============================================

class TestDiscoveryFunctions:
    """Tests for orchestrator discovery functions."""

    def test_discover_and_register_orchestrators(self):
        """Test orchestrator discovery."""
        # This may fail to import some modules in test environment
        results = discover_and_register_orchestrators()

        assert isinstance(results, dict)
        # At least one should be attempted
        assert len(results) > 0

    def test_get_orchestrator_inventory(self, mock_registry, monkeypatch):
        """Test getting orchestrator inventory."""
        OrchestratorRegistry._instance = mock_registry

        # Reset health protocol singleton
        import app.coordination.orchestrator_registry as module
        module._health_protocol = None

        result = get_orchestrator_inventory()

        assert "discovered" in result
        assert "active" in result
        assert "roles_held" in result
        assert "health" in result
        assert "timestamp" in result


# ============================================
# Test Auto Registration
# ============================================

class TestAutoRegistration:
    """Tests for auto registration of known coordinators."""

    def test_auto_register_known_coordinators(self):
        """Test auto registration discovers coordinators."""
        results = auto_register_known_coordinators()

        assert isinstance(results, dict)
        # Results should have entries for all known coordinators
        # Some may succeed, some may fail depending on imports
        assert len(results) > 0


# ============================================
# Test Wait for Role
# ============================================

class TestWaitForRole:
    """Tests for wait_for_role functionality."""

    def test_wait_for_role_immediate_success(self, mock_registry):
        """Test wait_for_role when role immediately available."""
        result = mock_registry.wait_for_role(
            OrchestratorRole.CLUSTER_ORCHESTRATOR,
            timeout=1.0,
            poll_interval=0.1
        )

        assert result is True
        assert mock_registry._my_role == OrchestratorRole.CLUSTER_ORCHESTRATOR

        # Clean up
        mock_registry.release_role()

    def test_wait_for_role_timeout(self, mock_registry):
        """Test wait_for_role times out when role held."""
        # First acquire the role
        mock_registry.acquire_role(OrchestratorRole.CLUSTER_ORCHESTRATOR)
        saved_id = mock_registry._my_id
        saved_role = mock_registry._my_role

        # Simulate different process
        mock_registry._my_id = None
        mock_registry._my_role = None

        with patch("socket.gethostname", return_value="different-host"):
            result = mock_registry.wait_for_role(
                OrchestratorRole.CLUSTER_ORCHESTRATOR,
                timeout=0.2,
                poll_interval=0.05
            )

            # Should timeout
            assert result is False

        # Restore and clean up
        mock_registry._my_id = saved_id
        mock_registry._my_role = saved_role
        mock_registry.release_role()


# ============================================
# Test Stale Cleanup
# ============================================

class TestStaleCleanup:
    """Tests for stale orchestrator cleanup."""

    def test_cleanup_stale_orchestrators(self, mock_registry):
        """Test cleanup of stale entries."""
        # Register with old heartbeat
        mock_registry.acquire_role(OrchestratorRole.CLUSTER_ORCHESTRATOR)

        # Manually set old heartbeat in DB
        old_time = (datetime.now() - timedelta(seconds=HEARTBEAT_TIMEOUT_SECONDS + 60)).isoformat()

        with mock_registry._get_conn() as conn:
            conn.execute(
                'UPDATE orchestrators SET last_heartbeat = ? WHERE id = ?',
                (old_time, mock_registry._my_id)
            )
            conn.commit()

        # Stop heartbeat thread to prevent updates
        mock_registry._stop_heartbeat()

        # Run cleanup
        mock_registry._cleanup_stale_orchestrators()

        # Entry should be removed
        active = mock_registry.get_active_orchestrators()
        assert len(active) == 0


# ============================================
# Test Thread Safety
# ============================================

class TestThreadSafety:
    """Tests for thread safety of registry operations."""

    def test_concurrent_heartbeats(self, mock_registry):
        """Test concurrent heartbeat updates."""
        mock_registry.acquire_role(OrchestratorRole.CLUSTER_ORCHESTRATOR)

        errors = []

        def heartbeat_worker():
            try:
                for _ in range(10):
                    mock_registry.heartbeat(metadata_update={"thread": threading.current_thread().name})
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=heartbeat_worker) for _ in range(5)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0

        # Clean up
        mock_registry.release_role()


# ============================================
# Integration Tests
# ============================================

class TestOrchestratorRegistryIntegration:
    """Integration tests for orchestrator registry."""

    def test_full_lifecycle(self, mock_registry):
        """Test full orchestrator lifecycle."""
        # Acquire role
        assert mock_registry.acquire_role(
            OrchestratorRole.CLUSTER_ORCHESTRATOR,
            metadata={"version": "1.0"}
        )

        # Send heartbeats
        for i in range(3):
            mock_registry.heartbeat(metadata_update={"iteration": i})
            time.sleep(0.01)

        # Check status
        summary = mock_registry.get_status_summary()
        assert summary["active_count"] == 1

        # Check events
        events = mock_registry.get_recent_events(limit=10)
        assert len(events) >= 1

        # Release
        mock_registry.release_role()

        # Verify released
        assert mock_registry._my_role is None
        assert mock_registry.is_role_held(OrchestratorRole.CLUSTER_ORCHESTRATOR) is False

    def test_health_protocol_integration(self, mock_registry):
        """Test health protocol with registry."""
        mock_registry.acquire_role(OrchestratorRole.CLUSTER_ORCHESTRATOR)

        protocol = CrossCoordinatorHealthProtocol(registry=mock_registry)

        # Check health
        health = protocol.get_role_health(OrchestratorRole.CLUSTER_ORCHESTRATOR)
        assert health is not None
        assert health.is_healthy is True

        # Get summary
        summary = protocol.get_cluster_health_summary()
        assert summary["total_coordinators"] == 1
        assert summary["healthy_count"] == 1

        # Clean up
        mock_registry.release_role()

    def test_coordinator_registration_integration(self, fresh_coordinator_registry):
        """Test coordinator registration with health checks."""
        class TestCoordinator:
            def __init__(self):
                self._healthy = True

            def is_healthy(self):
                return self._healthy

            def shutdown(self):
                self._healthy = False

        coord = TestCoordinator()

        # Register
        register_coordinator(
            "test_coord",
            coord,
            health_callback=coord.is_healthy,
            shutdown_callback=coord.shutdown,
        )

        # Check registered
        registered = get_registered_coordinators()
        assert "test_coord" in registered
        assert registered["test_coord"]["healthy"] is True

        # Shutdown
        results = shutdown_all_coordinators()
        assert results["test_coord"] is True

        # Check unhealthy after shutdown
        registered = get_registered_coordinators()
        assert registered["test_coord"]["healthy"] is False
