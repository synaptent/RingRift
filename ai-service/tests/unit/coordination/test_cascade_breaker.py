"""Tests for cascade_breaker.py - Hierarchical cascade circuit breaker.

January 7, 2026 - Phase 4.1: Comprehensive test coverage for CascadeBreakerManager.

Tests cover:
- Configuration dataclasses (CategoryBreakerConfig, CascadeBreakerConfig)
- CascadeBreakerManager functionality:
  - Critical daemon exemption
  - Startup grace period
  - Category-level circuit breaker tripping and cooldown
  - Global circuit breaker tripping and cooldown
  - Category exemption from global breaker
  - Status reporting
  - Reset functionality
- Singleton pattern (get_cascade_breaker, reset_cascade_breaker)
- Environment variable configuration
"""

from __future__ import annotations

import os
import time
from unittest.mock import patch

import pytest

from app.coordination.cascade_breaker import (
    CategoryBreakerConfig,
    CategoryBreakerState,
    CascadeBreakerConfig,
    CascadeBreakerManager,
    get_cascade_breaker,
    reset_cascade_breaker,
    _load_config_from_env,
)
from app.coordination.daemon_types import DaemonCategory, DaemonType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_breaker_singleton():
    """Reset singleton before and after each test."""
    reset_cascade_breaker()
    yield
    reset_cascade_breaker()


@pytest.fixture
def short_grace_config():
    """Config with very short grace period for testing."""
    return CascadeBreakerConfig(
        global_threshold=5,
        global_window_seconds=60,
        global_cooldown_seconds=10,
        startup_grace_period=0,  # No grace period
        startup_threshold=10,
    )


@pytest.fixture
def low_threshold_config():
    """Config with low thresholds for easy testing."""
    return CascadeBreakerConfig(
        global_threshold=3,
        global_window_seconds=300,
        global_cooldown_seconds=5,
        startup_grace_period=0,
        startup_threshold=10,
        category_configs={
            DaemonCategory.SYNC: CategoryBreakerConfig(
                threshold=2,
                window_seconds=60,
                cooldown_seconds=5,
                exempt_from_global=False,
            ),
            DaemonCategory.EVENT: CategoryBreakerConfig(
                threshold=3,
                window_seconds=60,
                cooldown_seconds=5,
                exempt_from_global=True,
            ),
        },
    )


# =============================================================================
# CategoryBreakerConfig Tests
# =============================================================================


class TestCategoryBreakerConfig:
    """Tests for CategoryBreakerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CategoryBreakerConfig()
        assert config.threshold == 5
        assert config.window_seconds == 300
        assert config.cooldown_seconds == 60
        assert config.exempt_from_global is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CategoryBreakerConfig(
            threshold=10,
            window_seconds=600,
            cooldown_seconds=120,
            exempt_from_global=True,
        )
        assert config.threshold == 10
        assert config.window_seconds == 600
        assert config.cooldown_seconds == 120
        assert config.exempt_from_global is True

    def test_frozen(self):
        """Test that config is immutable."""
        config = CategoryBreakerConfig()
        with pytest.raises(AttributeError):
            config.threshold = 10  # type: ignore


# =============================================================================
# CategoryBreakerState Tests
# =============================================================================


class TestCategoryBreakerState:
    """Tests for CategoryBreakerState dataclass."""

    def test_default_values(self):
        """Test default state values."""
        state = CategoryBreakerState()
        assert state.restart_timestamps == []
        assert state.breaker_open is False
        assert state.opened_at == 0.0
        assert state.total_restarts == 0
        assert state.total_blocked == 0

    def test_mutable(self):
        """Test that state is mutable."""
        state = CategoryBreakerState()
        state.breaker_open = True
        state.total_restarts = 5
        state.restart_timestamps.append(time.time())
        assert state.breaker_open is True
        assert state.total_restarts == 5
        assert len(state.restart_timestamps) == 1


# =============================================================================
# CascadeBreakerConfig Tests
# =============================================================================


class TestCascadeBreakerConfig:
    """Tests for CascadeBreakerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CascadeBreakerConfig()
        assert config.global_threshold == 25
        assert config.global_window_seconds == 300
        assert config.global_cooldown_seconds == 120
        assert config.startup_grace_period == 300
        assert config.startup_threshold == 100
        assert len(config.category_configs) > 0
        assert "event_router" in config.critical_exempt_daemons

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CascadeBreakerConfig(
            global_threshold=10,
            startup_grace_period=60,
        )
        assert config.global_threshold == 10
        assert config.startup_grace_period == 60


# =============================================================================
# CascadeBreakerManager - Basic Tests
# =============================================================================


class TestCascadeBreakerManagerBasic:
    """Basic tests for CascadeBreakerManager."""

    def test_initialization(self, short_grace_config):
        """Test manager initialization."""
        breaker = CascadeBreakerManager(short_grace_config)
        assert breaker._config == short_grace_config
        assert len(breaker._category_states) == len(DaemonCategory)
        assert breaker._global_breaker_open is False

    def test_can_restart_allowed(self, short_grace_config):
        """Test that restarts are allowed by default."""
        breaker = CascadeBreakerManager(short_grace_config)
        # Use ELO_SYNC which is NOT in CRITICAL_DAEMONS (AUTO_SYNC is critical)
        allowed, reason = breaker.can_restart(DaemonType.ELO_SYNC)
        assert allowed is True
        assert reason == "allowed"

    def test_record_restart(self, short_grace_config):
        """Test recording a restart."""
        breaker = CascadeBreakerManager(short_grace_config)
        breaker.record_restart(DaemonType.AUTO_SYNC)

        category = DaemonCategory.SYNC
        state = breaker._category_states[category]
        assert state.total_restarts == 1
        assert len(state.restart_timestamps) == 1

    def test_get_status(self, short_grace_config):
        """Test status reporting."""
        breaker = CascadeBreakerManager(short_grace_config)
        breaker.record_restart(DaemonType.AUTO_SYNC)

        status = breaker.get_status()
        assert "uptime_seconds" in status
        assert "global" in status
        assert "categories" in status
        assert status["total_allowed"] == 0
        assert status["global"]["breaker_open"] is False

    def test_reset(self, short_grace_config):
        """Test reset functionality."""
        breaker = CascadeBreakerManager(short_grace_config)

        # Record some restarts
        for _ in range(3):
            breaker.record_restart(DaemonType.AUTO_SYNC)

        # Reset
        breaker.reset()

        # Check state is cleared
        state = breaker._category_states[DaemonCategory.SYNC]
        assert len(state.restart_timestamps) == 0
        assert state.breaker_open is False
        assert breaker._global_breaker_open is False


# =============================================================================
# CascadeBreakerManager - Critical Daemon Tests
# =============================================================================


class TestCascadeBreakerManagerCritical:
    """Tests for critical daemon exemption."""

    def test_critical_daemon_exempt_by_name(self, short_grace_config):
        """Test that critical daemons by name are always allowed."""
        config = CascadeBreakerConfig(
            startup_grace_period=0,
            critical_exempt_daemons=frozenset({"auto_sync"}),
        )
        breaker = CascadeBreakerManager(config)

        # Record many restarts to trip breakers
        for _ in range(30):
            breaker.record_restart(DaemonType.AUTO_SYNC)

        # Should still be allowed (exempt by name)
        allowed, reason = breaker.can_restart(DaemonType.AUTO_SYNC)
        assert allowed is True
        assert reason == "critical_daemon_exempt"

    def test_critical_daemon_exempt_by_set(self):
        """Test that CRITICAL_DAEMONS set is respected."""
        from app.coordination.daemon_types import CRITICAL_DAEMONS

        # Find a daemon in CRITICAL_DAEMONS but NOT in critical_exempt_daemons
        # AUTO_SYNC, QUEUE_POPULATOR, IDLE_RESOURCE, SOCKET_LEAK_RECOVERY are in CRITICAL_DAEMONS
        # but not in the default critical_exempt_daemons frozenset
        config = CascadeBreakerConfig(startup_grace_period=0)

        # Find a critical daemon NOT in the exempt set
        exempt_names = {d for d in config.critical_exempt_daemons}
        non_exempt_critical = None
        for daemon in CRITICAL_DAEMONS:
            if daemon.value not in exempt_names:
                non_exempt_critical = daemon
                break

        if non_exempt_critical:
            breaker = CascadeBreakerManager(config)
            allowed, reason = breaker.can_restart(non_exempt_critical)
            assert allowed is True
            assert reason == "critical_daemon_set"


# =============================================================================
# CascadeBreakerManager - Startup Grace Period Tests
# =============================================================================


class TestCascadeBreakerManagerStartupGrace:
    """Tests for startup grace period."""

    def test_grace_period_allows_restarts(self):
        """Test that restarts are allowed during grace period."""
        config = CascadeBreakerConfig(
            startup_grace_period=300,  # 5 minutes
            startup_threshold=100,
        )
        breaker = CascadeBreakerManager(config)

        # Immediately after creation, should be in grace period
        # Use ELO_SYNC which is NOT in CRITICAL_DAEMONS (critical daemons bypass grace period check)
        allowed, reason = breaker.can_restart(DaemonType.ELO_SYNC)
        assert allowed is True
        assert reason == "startup_grace_period"

    def test_grace_period_expires(self, short_grace_config):
        """Test that grace period expires."""
        # Grace period is 0, so should not be in grace period
        breaker = CascadeBreakerManager(short_grace_config)

        # Use ELO_SYNC which is NOT in CRITICAL_DAEMONS
        allowed, reason = breaker.can_restart(DaemonType.ELO_SYNC)
        assert allowed is True
        # Should be "allowed" not "startup_grace_period"
        assert reason == "allowed"


# =============================================================================
# CascadeBreakerManager - Category Breaker Tests
# =============================================================================


class TestCascadeBreakerManagerCategory:
    """Tests for category-level circuit breaker."""

    def test_category_breaker_trips(self, low_threshold_config):
        """Test that category breaker trips at threshold."""
        breaker = CascadeBreakerManager(low_threshold_config)

        # SYNC category has threshold of 2
        breaker.record_restart(DaemonType.AUTO_SYNC)

        # First restart - still allowed
        allowed, _ = breaker.can_restart(DaemonType.AUTO_SYNC)
        assert allowed is True

        # Second restart - trips the breaker
        breaker.record_restart(DaemonType.AUTO_SYNC)

        # Third attempt - blocked
        allowed, reason = breaker.can_restart(DaemonType.AUTO_SYNC)
        assert allowed is False
        assert "category_sync_breaker_open" in reason

    def test_category_cooldown_expires(self, low_threshold_config):
        """Test that category breaker cooldown expires."""
        breaker = CascadeBreakerManager(low_threshold_config)

        # Trip the breaker
        for _ in range(2):
            breaker.record_restart(DaemonType.AUTO_SYNC)

        # Verify blocked
        allowed, _ = breaker.can_restart(DaemonType.AUTO_SYNC)
        assert allowed is False

        # Simulate cooldown expiry (modify opened_at)
        category = DaemonCategory.SYNC
        breaker._category_states[category].opened_at = time.time() - 10

        # Should be allowed again
        allowed, reason = breaker.can_restart(DaemonType.AUTO_SYNC)
        assert allowed is True
        assert reason == "allowed"

    def test_different_categories_independent(self, low_threshold_config):
        """Test that different categories are independent."""
        breaker = CascadeBreakerManager(low_threshold_config)

        # Trip SYNC category
        for _ in range(2):
            breaker.record_restart(DaemonType.AUTO_SYNC)

        # SYNC should be blocked
        allowed, _ = breaker.can_restart(DaemonType.AUTO_SYNC)
        assert allowed is False

        # EVENT category should still work
        allowed, _ = breaker.can_restart(DaemonType.EVENT_ROUTER)
        assert allowed is True


# =============================================================================
# CascadeBreakerManager - Global Breaker Tests
# =============================================================================


class TestCascadeBreakerManagerGlobal:
    """Tests for global circuit breaker."""

    def test_global_breaker_trips(self, low_threshold_config):
        """Test that global breaker trips at threshold."""
        breaker = CascadeBreakerManager(low_threshold_config)

        # Global threshold is 3
        # Use SYNC daemon which is not exempt from global
        for _ in range(3):
            breaker.record_restart(DaemonType.AUTO_SYNC)

        # Global breaker should be open
        assert breaker._global_breaker_open is True

        # Non-exempt category should be blocked
        allowed, reason = breaker.can_restart(DaemonType.COORDINATOR_DISK_MANAGER)
        assert allowed is False
        assert "global_breaker_open" in reason

    def test_exempt_category_ignores_global(self, low_threshold_config):
        """Test that exempt categories ignore global breaker."""
        breaker = CascadeBreakerManager(low_threshold_config)

        # Trip global breaker
        for _ in range(3):
            breaker.record_restart(DaemonType.AUTO_SYNC)

        assert breaker._global_breaker_open is True

        # EVENT is exempt from global - should still be allowed
        allowed, _ = breaker.can_restart(DaemonType.EVENT_ROUTER)
        assert allowed is True

    def test_global_cooldown_expires(self, low_threshold_config):
        """Test that global breaker cooldown expires."""
        breaker = CascadeBreakerManager(low_threshold_config)

        # Trip global breaker
        for _ in range(3):
            breaker.record_restart(DaemonType.AUTO_SYNC)

        assert breaker._global_breaker_open is True

        # Simulate cooldown expiry
        breaker._global_opened_at = time.time() - 10

        # Should be allowed again
        allowed, _ = breaker.can_restart(DaemonType.COORDINATOR_DISK_MANAGER)
        assert allowed is True
        assert breaker._global_breaker_open is False


# =============================================================================
# CascadeBreakerManager - Statistics Tests
# =============================================================================


class TestCascadeBreakerManagerStats:
    """Tests for statistics tracking."""

    def test_total_allowed_tracked(self, short_grace_config):
        """Test that allowed restarts are counted."""
        breaker = CascadeBreakerManager(short_grace_config)

        for _ in range(5):
            allowed, _ = breaker.can_restart(DaemonType.AUTO_SYNC)
            assert allowed is True

        assert breaker._total_allowed == 5

    def test_total_blocked_tracked(self, low_threshold_config):
        """Test that blocked restarts are counted."""
        breaker = CascadeBreakerManager(low_threshold_config)

        # Trip the breaker
        for _ in range(2):
            breaker.record_restart(DaemonType.AUTO_SYNC)

        # Try blocked restarts
        for _ in range(3):
            allowed, _ = breaker.can_restart(DaemonType.AUTO_SYNC)
            assert allowed is False

        assert breaker._total_blocked == 3

    def test_blocked_by_category_tracked(self, low_threshold_config):
        """Test that blocked-by-category is tracked."""
        breaker = CascadeBreakerManager(low_threshold_config)

        # Trip the breaker
        for _ in range(2):
            breaker.record_restart(DaemonType.AUTO_SYNC)

        # Try blocked restart
        allowed, _ = breaker.can_restart(DaemonType.AUTO_SYNC)
        assert allowed is False

        assert breaker._blocked_by_category[DaemonCategory.SYNC] == 1


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingletonFunctions:
    """Tests for singleton access functions."""

    def test_get_cascade_breaker_returns_instance(self):
        """Test that get_cascade_breaker returns an instance."""
        breaker = get_cascade_breaker()
        assert isinstance(breaker, CascadeBreakerManager)

    def test_get_cascade_breaker_returns_same_instance(self):
        """Test that get_cascade_breaker returns same instance."""
        breaker1 = get_cascade_breaker()
        breaker2 = get_cascade_breaker()
        assert breaker1 is breaker2

    def test_reset_cascade_breaker_resets_state(self):
        """Test that reset_cascade_breaker resets state."""
        breaker = get_cascade_breaker()
        breaker.record_restart(DaemonType.AUTO_SYNC)

        reset_cascade_breaker()

        # Get new instance
        breaker2 = get_cascade_breaker()
        state = breaker2._category_states[DaemonCategory.SYNC]
        assert state.total_restarts == 0

    def test_reset_cascade_breaker_clears_singleton(self):
        """Test that reset_cascade_breaker clears singleton."""
        breaker1 = get_cascade_breaker()
        reset_cascade_breaker()
        breaker2 = get_cascade_breaker()

        # Should be different instances
        assert breaker1 is not breaker2


# =============================================================================
# Environment Variable Tests
# =============================================================================


class TestEnvironmentConfig:
    """Tests for environment variable configuration."""

    def test_load_config_from_env_no_vars(self):
        """Test loading config with no environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            config = _load_config_from_env()
            assert config is None

    def test_load_config_from_env_global_threshold(self):
        """Test loading global threshold from env."""
        with patch.dict(os.environ, {"RINGRIFT_CASCADE_GLOBAL_THRESHOLD": "50"}):
            config = _load_config_from_env()
            assert config is not None
            assert config.global_threshold == 50

    def test_load_config_from_env_global_cooldown(self):
        """Test loading global cooldown from env."""
        with patch.dict(os.environ, {"RINGRIFT_CASCADE_GLOBAL_COOLDOWN": "180"}):
            config = _load_config_from_env()
            assert config is not None
            assert config.global_cooldown_seconds == 180

    def test_load_config_from_env_startup_grace(self):
        """Test loading startup grace period from env."""
        with patch.dict(os.environ, {"RINGRIFT_CASCADE_STARTUP_GRACE": "600"}):
            config = _load_config_from_env()
            assert config is not None
            assert config.startup_grace_period == 600

    def test_load_config_from_env_startup_threshold(self):
        """Test loading startup threshold from env."""
        with patch.dict(os.environ, {"RINGRIFT_CASCADE_STARTUP_THRESHOLD": "200"}):
            config = _load_config_from_env()
            assert config is not None
            assert config.startup_threshold == 200

    def test_load_config_from_env_invalid_value(self, caplog):
        """Test that invalid env values are logged."""
        with patch.dict(os.environ, {"RINGRIFT_CASCADE_GLOBAL_THRESHOLD": "not_a_number"}):
            config = _load_config_from_env()
            assert config is None
            assert "Invalid RINGRIFT_CASCADE_GLOBAL_THRESHOLD" in caplog.text


# =============================================================================
# Status Reporting Tests
# =============================================================================


class TestStatusReporting:
    """Tests for get_status() method."""

    def test_status_contains_global_info(self, short_grace_config):
        """Test that status contains global breaker info."""
        breaker = CascadeBreakerManager(short_grace_config)
        status = breaker.get_status()

        assert "global" in status
        assert "breaker_open" in status["global"]
        assert "recent_restarts" in status["global"]
        assert "threshold" in status["global"]
        assert "window_seconds" in status["global"]

    def test_status_contains_category_info(self, short_grace_config):
        """Test that status contains category info."""
        breaker = CascadeBreakerManager(short_grace_config)
        status = breaker.get_status()

        assert "categories" in status
        assert len(status["categories"]) > 0

        # Check first category has expected fields
        for cat_name, cat_status in status["categories"].items():
            assert "breaker_open" in cat_status
            assert "recent_restarts" in cat_status
            assert "threshold" in cat_status
            assert "total_restarts" in cat_status

    def test_status_shows_cooldown_remaining(self, low_threshold_config):
        """Test that status shows cooldown remaining when breaker is open."""
        breaker = CascadeBreakerManager(low_threshold_config)

        # Trip the breaker
        for _ in range(2):
            breaker.record_restart(DaemonType.AUTO_SYNC)

        status = breaker.get_status()

        sync_status = status["categories"]["sync"]
        assert sync_status["breaker_open"] is True
        assert "cooldown_remaining" in sync_status
        assert sync_status["cooldown_remaining"] > 0

    def test_status_in_startup_grace(self):
        """Test status during startup grace period."""
        config = CascadeBreakerConfig(startup_grace_period=300)
        breaker = CascadeBreakerManager(config)

        status = breaker.get_status()
        assert status["in_startup_grace"] is True


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestEventEmission:
    """Tests for event emission."""

    def test_category_breaker_emits_event(self, low_threshold_config):
        """Test that tripping category breaker emits event."""
        breaker = CascadeBreakerManager(low_threshold_config)

        with patch(
            "app.coordination.cascade_breaker.safe_emit_event"
        ) as mock_emit:
            # Trip the breaker
            for _ in range(2):
                breaker.record_restart(DaemonType.AUTO_SYNC)

            # Verify event emitted
            mock_emit.assert_called()
            call_args = mock_emit.call_args
            assert call_args[0][0] == "CATEGORY_BREAKER_TRIPPED"
            assert call_args[0][1]["category"] == "sync"

    def test_global_breaker_emits_event(self, low_threshold_config):
        """Test that tripping global breaker emits event."""
        breaker = CascadeBreakerManager(low_threshold_config)

        with patch(
            "app.coordination.cascade_breaker.safe_emit_event"
        ) as mock_emit:
            # Trip global breaker
            for _ in range(3):
                breaker.record_restart(DaemonType.AUTO_SYNC)

            # Verify global event emitted
            calls = mock_emit.call_args_list
            global_calls = [c for c in calls if c[0][0] == "GLOBAL_BREAKER_TRIPPED"]
            assert len(global_calls) >= 1
