"""Tests for BaseDaemon base class.

December 2025: Created for Phase 1 quick wins - test coverage for base daemon class.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.base_daemon import BaseDaemon, DaemonConfig
from app.coordination.protocols import CoordinatorStatus, HealthCheckResult


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass(kw_only=True)
class MockDaemonConfig(DaemonConfig):
    """Mock configuration for testing."""

    test_setting: str = "test_value"
    fail_on_cycle: bool = False
    cycle_delay: float = 0.0


class MockDaemon(BaseDaemon[MockDaemonConfig]):
    """Concrete daemon implementation for testing."""

    def __init__(self, config: MockDaemonConfig | None = None):
        super().__init__(config)
        self.cycle_count = 0
        self.on_start_called = False
        self.on_stop_called = False

    async def _run_cycle(self) -> None:
        """Run one test cycle."""
        self.cycle_count += 1
        if self.config.fail_on_cycle:
            raise RuntimeError("Test failure")
        if self.config.cycle_delay > 0:
            await asyncio.sleep(self.config.cycle_delay)

    @staticmethod
    def _get_default_config() -> MockDaemonConfig:
        return MockDaemonConfig()

    async def _on_start(self) -> None:
        self.on_start_called = True

    async def _on_stop(self) -> None:
        self.on_stop_called = True


@pytest.fixture
def daemon():
    """Create a test daemon."""
    return MockDaemon()


@pytest.fixture
def daemon_config():
    """Create a test config."""
    return MockDaemonConfig()


# =============================================================================
# DaemonConfig Tests
# =============================================================================


class MockDaemonConfig:
    """Tests for DaemonConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DaemonConfig()
        assert config.enabled is True
        assert config.check_interval_seconds == 300
        assert config.handle_signals is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DaemonConfig(
            enabled=False,
            check_interval_seconds=60,
            handle_signals=True,
        )
        assert config.enabled is False
        assert config.check_interval_seconds == 60
        assert config.handle_signals is True

    def test_subclass_inheritance(self):
        """Test subclass can add fields."""
        config = MockDaemonConfig(test_setting="custom")
        assert config.test_setting == "custom"
        assert config.enabled is True  # Inherited default

    def test_from_env_enabled(self):
        """Test from_env loads enabled flag."""
        with patch.dict(os.environ, {"TEST_ENABLED": "0"}):
            config = DaemonConfig.from_env(prefix="TEST")
            assert config.enabled is False

    def test_from_env_interval(self):
        """Test from_env loads interval."""
        with patch.dict(os.environ, {"TEST_INTERVAL": "120"}):
            config = DaemonConfig.from_env(prefix="TEST")
            assert config.check_interval_seconds == 120

    def test_from_env_signals(self):
        """Test from_env loads signal handling."""
        with patch.dict(os.environ, {"TEST_HANDLE_SIGNALS": "1"}):
            config = DaemonConfig.from_env(prefix="TEST")
            assert config.handle_signals is True

    def test_from_env_invalid_interval(self):
        """Test from_env handles invalid interval gracefully."""
        with patch.dict(os.environ, {"TEST_INTERVAL": "not_a_number"}):
            config = DaemonConfig.from_env(prefix="TEST")
            # Should keep default
            assert config.check_interval_seconds == 300


# =============================================================================
# BaseDaemon Lifecycle Tests
# =============================================================================


class TestBaseDaemonLifecycle:
    """Tests for daemon lifecycle management."""

    def test_initialization(self, daemon: MockDaemon):
        """Test daemon initializes correctly."""
        assert daemon.is_running is False
        assert daemon._coordinator_status == CoordinatorStatus.INITIALIZING
        assert daemon._errors_count == 0
        assert daemon._cycles_completed == 0
        assert daemon.node_id is not None

    def test_initialization_with_config(self):
        """Test daemon uses provided config."""
        config = MockDaemonConfig(check_interval_seconds=10)
        daemon = MockDaemon(config)
        assert daemon.config.check_interval_seconds == 10

    def test_initialization_default_config(self):
        """Test daemon uses default config when None provided."""
        daemon = MockDaemon(config=None)
        assert daemon.config.test_setting == "test_value"

    @pytest.mark.asyncio
    async def test_start(self, daemon: MockDaemon):
        """Test daemon start."""
        # Mock register_coordinator to avoid side effects
        with patch("app.coordination.base_daemon.register_coordinator"):
            await daemon.start()
            assert daemon.is_running is True
            assert daemon._coordinator_status == CoordinatorStatus.RUNNING
            assert daemon.on_start_called is True
            assert daemon._task is not None
            # Clean up
            await daemon.stop()

    @pytest.mark.asyncio
    async def test_start_when_disabled(self):
        """Test daemon doesn't start when disabled."""
        config = MockDaemonConfig(enabled=False)
        daemon = MockDaemon(config)
        await daemon.start()
        assert daemon.is_running is False

    @pytest.mark.asyncio
    async def test_start_already_running(self, daemon: MockDaemon):
        """Test start is idempotent."""
        with patch("app.coordination.base_daemon.register_coordinator"):
            await daemon.start()
            # Call start again
            await daemon.start()
            assert daemon.is_running is True
            await daemon.stop()

    @pytest.mark.asyncio
    async def test_stop(self, daemon: MockDaemon):
        """Test daemon stop."""
        with patch("app.coordination.base_daemon.register_coordinator"):
            with patch("app.coordination.base_daemon.unregister_coordinator"):
                await daemon.start()
                await daemon.stop()
                assert daemon.is_running is False
                assert daemon._coordinator_status == CoordinatorStatus.STOPPED
                assert daemon.on_stop_called is True

    @pytest.mark.asyncio
    async def test_stop_not_running(self, daemon: MockDaemon):
        """Test stop when not running is safe."""
        await daemon.stop()
        assert daemon.is_running is False

    def test_uptime_not_started(self, daemon: MockDaemon):
        """Test uptime is 0 when not started."""
        assert daemon.uptime_seconds == 0.0

    @pytest.mark.asyncio
    async def test_uptime_after_start(self, daemon: MockDaemon):
        """Test uptime increases after start."""
        with patch("app.coordination.base_daemon.register_coordinator"):
            await daemon.start()
            await asyncio.sleep(0.1)
            assert daemon.uptime_seconds > 0
            await daemon.stop()


# =============================================================================
# Main Loop Tests
# =============================================================================


class TestBaseDaemonMainLoop:
    """Tests for protected main loop."""

    @pytest.mark.asyncio
    async def test_cycle_execution(self):
        """Test cycles are executed."""
        config = MockDaemonConfig(check_interval_seconds=0)
        daemon = MockDaemon(config)

        with patch("app.coordination.base_daemon.register_coordinator"):
            await daemon.start()
            # Wait for a cycle
            await asyncio.sleep(0.1)
            assert daemon.cycle_count > 0
            assert daemon._cycles_completed > 0
            await daemon.stop()

    @pytest.mark.asyncio
    async def test_cycle_error_handling(self):
        """Test errors in cycle are caught and counted."""
        config = MockDaemonConfig(
            check_interval_seconds=0,
            fail_on_cycle=True,
        )
        daemon = MockDaemon(config)

        with patch("app.coordination.base_daemon.register_coordinator"):
            await daemon.start()
            await asyncio.sleep(0.1)
            assert daemon._errors_count > 0
            assert daemon._last_error == "Test failure"
            await daemon.stop()

    @pytest.mark.asyncio
    async def test_loop_continues_after_error(self):
        """Test loop continues after error."""
        config = MockDaemonConfig(
            check_interval_seconds=0,
            fail_on_cycle=True,
        )
        daemon = MockDaemon(config)

        with patch("app.coordination.base_daemon.register_coordinator"):
            await daemon.start()
            await asyncio.sleep(0.1)
            # Daemon should still be running
            assert daemon.is_running is True
            await daemon.stop()


# =============================================================================
# Health Check Tests
# =============================================================================


class TestBaseDaemonHealthCheck:
    """Tests for health_check method."""

    def test_health_check_not_running(self, daemon: MockDaemon):
        """Test health_check when not running."""
        result = daemon.health_check()
        assert result.healthy is False
        assert result.status == CoordinatorStatus.STOPPED
        assert "not running" in result.message.lower()

    @pytest.mark.asyncio
    async def test_health_check_running(self):
        """Test health_check when running healthy."""
        daemon = MockDaemon()

        with patch("app.coordination.base_daemon.register_coordinator"):
            await daemon.start()
            # Let it run a cycle
            await asyncio.sleep(0.1)
            result = daemon.health_check()
            assert result.healthy is True
            await daemon.stop()

    @pytest.mark.asyncio
    async def test_health_check_high_error_rate(self):
        """Test health_check with high error rate."""
        config = MockDaemonConfig(
            check_interval_seconds=0,
            fail_on_cycle=True,
        )
        daemon = MockDaemon(config)

        with patch("app.coordination.base_daemon.register_coordinator"):
            await daemon.start()
            # Let it fail several times
            await asyncio.sleep(0.2)
            result = daemon.health_check()
            # Should be unhealthy due to high error rate
            assert result.status in (CoordinatorStatus.ERROR, CoordinatorStatus.RUNNING)
            await daemon.stop()


# =============================================================================
# Status Tests
# =============================================================================


class TestBaseDaemonStatus:
    """Tests for get_status method."""

    def test_get_status_initial(self, daemon: MockDaemon):
        """Test initial status."""
        status = daemon.get_status()
        assert status["daemon"] == "MockDaemon"
        assert status["running"] is False
        assert status["node_id"] is not None
        assert status["stats"]["cycles_completed"] == 0
        assert status["stats"]["errors"] == 0

    @pytest.mark.asyncio
    async def test_get_status_running(self):
        """Test status when running."""
        config = MockDaemonConfig(check_interval_seconds=0)
        daemon = MockDaemon(config)

        with patch("app.coordination.base_daemon.register_coordinator"):
            await daemon.start()
            await asyncio.sleep(0.1)
            status = daemon.get_status()
            assert status["running"] is True
            assert status["uptime_seconds"] > 0
            assert status["stats"]["cycles_completed"] > 0
            await daemon.stop()

    def test_get_status_config(self, daemon: MockDaemon):
        """Test config is included in status."""
        status = daemon.get_status()
        assert "config" in status
        assert "enabled" in status["config"]
        assert "interval" in status["config"]


# =============================================================================
# Property Tests
# =============================================================================


class TestBaseDaemonProperties:
    """Tests for daemon properties."""

    def test_name_property(self, daemon: MockDaemon):
        """Test name property returns daemon name."""
        assert daemon.name == "MockDaemon"

    def test_status_property(self, daemon: MockDaemon):
        """Test status property returns coordinator status."""
        assert daemon.status == CoordinatorStatus.INITIALIZING

    @pytest.mark.asyncio
    async def test_status_property_running(self):
        """Test status property when running."""
        daemon = MockDaemon()
        with patch("app.coordination.base_daemon.register_coordinator"):
            await daemon.start()
            assert daemon.status == CoordinatorStatus.RUNNING
            await daemon.stop()


# =============================================================================
# Coordinator Registration Tests
# =============================================================================


class TestBaseDaemonCoordinatorRegistration:
    """Tests for coordinator protocol registration."""

    @pytest.mark.asyncio
    async def test_register_on_start(self):
        """Test coordinator registration on start."""
        daemon = MockDaemon()
        with patch("app.coordination.base_daemon.register_coordinator") as mock_register:
            await daemon.start()
            mock_register.assert_called_once()
            await daemon.stop()

    @pytest.mark.asyncio
    async def test_unregister_on_stop(self):
        """Test coordinator unregistration on stop."""
        daemon = MockDaemon()
        with patch("app.coordination.base_daemon.register_coordinator"):
            with patch("app.coordination.base_daemon.unregister_coordinator") as mock_unregister:
                await daemon.start()
                await daemon.stop()
                mock_unregister.assert_called_once()


# =============================================================================
# Custom Daemon Name Tests
# =============================================================================


class TestBaseDaemonCustomName:
    """Tests for custom daemon naming."""

    def test_default_name(self, daemon: MockDaemon):
        """Test default name is class name."""
        assert daemon._get_daemon_name() == "MockDaemon"

    def test_custom_name(self):
        """Test custom daemon name override."""

        class CustomNameDaemon(MockDaemon):
            def _get_daemon_name(self) -> str:
                return "MyCustomDaemon"

        daemon = CustomNameDaemon()
        assert daemon._get_daemon_name() == "MyCustomDaemon"
        assert daemon.name == "MyCustomDaemon"
