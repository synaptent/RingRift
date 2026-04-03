"""Tests for BaseCoordinationConfig and related config classes.

Sprint 17.2 (January 4, 2026): Tests for type-safe configuration base class.
"""

import os
from dataclasses import dataclass
from typing import ClassVar
from unittest.mock import patch

import pytest

from app.coordination.base_config import (
    BaseCoordinationConfig,
    MonitorDaemonConfig,
    RecoveryDaemonConfig,
    SyncDaemonConfig,
)


# =============================================================================
# BaseCoordinationConfig Tests
# =============================================================================


class TestBaseCoordinationConfig:
    """Tests for the base configuration class."""

    def test_default_values(self):
        """Test default field values."""
        config = BaseCoordinationConfig()
        assert config.enabled is True
        assert config.check_interval_seconds == 60.0
        assert config.startup_delay_seconds == 0.0
        assert config.description == ""

    def test_custom_values(self):
        """Test setting custom values."""
        config = BaseCoordinationConfig(
            enabled=False,
            check_interval_seconds=120.0,
            startup_delay_seconds=5.0,
            description="Test daemon",
        )
        assert config.enabled is False
        assert config.check_interval_seconds == 120.0
        assert config.startup_delay_seconds == 5.0
        assert config.description == "Test daemon"

    def test_is_enabled(self):
        """Test is_enabled convenience method."""
        config_enabled = BaseCoordinationConfig(enabled=True)
        config_disabled = BaseCoordinationConfig(enabled=False)
        assert config_enabled.is_enabled() is True
        assert config_disabled.is_enabled() is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = BaseCoordinationConfig(
            enabled=True,
            check_interval_seconds=30.0,
            description="Test",
        )
        result = config.to_dict()
        assert result["enabled"] is True
        assert result["check_interval_seconds"] == 30.0
        assert result["description"] == "Test"
        # Should not include private fields
        assert "_env_prefix" not in result


# =============================================================================
# Environment Variable Helper Tests
# =============================================================================


class TestEnvVarHelpers:
    """Tests for environment variable loading helpers."""

    def test_make_env_key(self):
        """Test environment key construction."""
        key = BaseCoordinationConfig._make_env_key("ENABLED")
        assert key == "RINGRIFT_ENABLED"

    def test_get_env_bool_true_values(self):
        """Test boolean parsing for true values."""
        true_values = ["true", "TRUE", "True", "1", "yes", "YES", "on", "ON"]
        for value in true_values:
            with patch.dict(os.environ, {"RINGRIFT_TEST": value}):
                result = BaseCoordinationConfig._get_env_bool("TEST", False)
                assert result is True, f"'{value}' should parse as True"

    def test_get_env_bool_false_values(self):
        """Test boolean parsing for false values."""
        false_values = ["false", "FALSE", "0", "no", "NO", "off", "OFF", "anything"]
        for value in false_values:
            with patch.dict(os.environ, {"RINGRIFT_TEST": value}):
                result = BaseCoordinationConfig._get_env_bool("TEST", True)
                assert result is False, f"'{value}' should parse as False"

    def test_get_env_bool_default(self):
        """Test boolean default when env var not set."""
        # Ensure env var is not set
        env = os.environ.copy()
        env.pop("RINGRIFT_UNSET_VAR", None)
        with patch.dict(os.environ, env, clear=True):
            assert BaseCoordinationConfig._get_env_bool("UNSET_VAR", True) is True
            assert BaseCoordinationConfig._get_env_bool("UNSET_VAR", False) is False

    def test_get_env_int(self):
        """Test integer parsing."""
        with patch.dict(os.environ, {"RINGRIFT_COUNT": "42"}):
            result = BaseCoordinationConfig._get_env_int("COUNT", 0)
            assert result == 42

    def test_get_env_int_invalid(self):
        """Test integer parsing with invalid value."""
        with patch.dict(os.environ, {"RINGRIFT_COUNT": "not_a_number"}):
            result = BaseCoordinationConfig._get_env_int("COUNT", 100)
            assert result == 100  # Returns default

    def test_get_env_int_default(self):
        """Test integer default when env var not set."""
        result = BaseCoordinationConfig._get_env_int("NONEXISTENT_INT", 999)
        assert result == 999

    def test_get_env_float(self):
        """Test float parsing."""
        with patch.dict(os.environ, {"RINGRIFT_RATE": "3.14"}):
            result = BaseCoordinationConfig._get_env_float("RATE", 0.0)
            assert abs(result - 3.14) < 0.001

    def test_get_env_float_invalid(self):
        """Test float parsing with invalid value."""
        with patch.dict(os.environ, {"RINGRIFT_RATE": "invalid"}):
            result = BaseCoordinationConfig._get_env_float("RATE", 1.5)
            assert result == 1.5  # Returns default

    def test_get_env_str(self):
        """Test string retrieval."""
        with patch.dict(os.environ, {"RINGRIFT_NAME": "  test_value  "}):
            result = BaseCoordinationConfig._get_env_str("NAME", "default")
            assert result == "test_value"  # Stripped

    def test_get_env_str_default(self):
        """Test string default when env var not set."""
        result = BaseCoordinationConfig._get_env_str("NONEXISTENT_STR", "fallback")
        assert result == "fallback"

    def test_get_env_list(self):
        """Test list parsing."""
        with patch.dict(os.environ, {"RINGRIFT_HOSTS": "host1, host2, host3"}):
            result = BaseCoordinationConfig._get_env_list("HOSTS")
            assert result == ["host1", "host2", "host3"]

    def test_get_env_list_custom_separator(self):
        """Test list parsing with custom separator."""
        with patch.dict(os.environ, {"RINGRIFT_ITEMS": "a:b:c"}):
            result = BaseCoordinationConfig._get_env_list("ITEMS", separator=":")
            assert result == ["a", "b", "c"]

    def test_get_env_list_empty_items(self):
        """Test that empty items are filtered."""
        with patch.dict(os.environ, {"RINGRIFT_ITEMS": "a,,b,  ,c"}):
            result = BaseCoordinationConfig._get_env_list("ITEMS")
            assert result == ["a", "b", "c"]

    def test_get_env_list_default(self):
        """Test list default when env var not set."""
        result = BaseCoordinationConfig._get_env_list("NONEXISTENT_LIST", default=["x", "y"])
        assert result == ["x", "y"]


# =============================================================================
# Custom Config Subclass Tests
# =============================================================================


class TestCustomConfigSubclass:
    """Tests for custom config subclass patterns."""

    def test_custom_env_prefix(self):
        """Test that subclasses can override env prefix."""

        @dataclass
        class MyDaemonConfig(BaseCoordinationConfig):
            _env_prefix: ClassVar[str] = "MY_DAEMON"
            custom_field: int = 10

        key = MyDaemonConfig._make_env_key("CUSTOM")
        assert key == "MY_DAEMON_CUSTOM"

    def test_from_env_loads_base_fields(self):
        """Test that from_env loads base class fields."""
        with patch.dict(
            os.environ,
            {
                "RINGRIFT_ENABLED": "false",
                "RINGRIFT_CHECK_INTERVAL": "120.5",
                "RINGRIFT_STARTUP_DELAY": "10.0",
            },
        ):
            config = BaseCoordinationConfig.from_env()
            assert config.enabled is False
            assert config.check_interval_seconds == 120.5
            assert config.startup_delay_seconds == 10.0

    def test_custom_from_env(self):
        """Test custom from_env in subclass."""

        @dataclass
        class MyConfig(BaseCoordinationConfig):
            _env_prefix: ClassVar[str] = "MY_SERVICE"
            max_retries: int = 3
            timeout: float = 30.0

            @classmethod
            def from_env(cls) -> "MyConfig":
                return cls(
                    enabled=cls._get_env_bool("ENABLED", True),
                    check_interval_seconds=cls._get_env_float("INTERVAL", 60.0),
                    max_retries=cls._get_env_int("MAX_RETRIES", 3),
                    timeout=cls._get_env_float("TIMEOUT", 30.0),
                )

        with patch.dict(
            os.environ,
            {
                "MY_SERVICE_ENABLED": "true",
                "MY_SERVICE_INTERVAL": "45.0",
                "MY_SERVICE_MAX_RETRIES": "5",
                "MY_SERVICE_TIMEOUT": "60.0",
            },
        ):
            config = MyConfig.from_env()
            assert config.enabled is True
            assert config.check_interval_seconds == 45.0
            assert config.max_retries == 5
            assert config.timeout == 60.0


# =============================================================================
# Pre-built Template Tests
# =============================================================================


class TestSyncDaemonConfig:
    """Tests for SyncDaemonConfig template."""

    def test_default_values(self):
        """Test default sync config values."""
        config = SyncDaemonConfig()
        assert config.sync_timeout_seconds == 300.0
        assert config.max_concurrent_syncs == 1
        assert config.retry_count == 3
        assert config.retry_delay_seconds == 5.0

    def test_env_prefix(self):
        """Test sync config env prefix."""
        key = SyncDaemonConfig._make_env_key("TIMEOUT")
        assert key == "RINGRIFT_SYNC_TIMEOUT"


class TestMonitorDaemonConfig:
    """Tests for MonitorDaemonConfig template."""

    def test_default_values(self):
        """Test default monitor config values."""
        config = MonitorDaemonConfig()
        assert config.health_check_interval_seconds == 30.0
        assert config.alert_threshold_count == 3
        assert config.alert_cooldown_seconds == 300.0

    def test_env_prefix(self):
        """Test monitor config env prefix."""
        key = MonitorDaemonConfig._make_env_key("INTERVAL")
        assert key == "RINGRIFT_MONITOR_INTERVAL"


class TestRecoveryDaemonConfig:
    """Tests for RecoveryDaemonConfig template."""

    def test_default_values(self):
        """Test default recovery config values."""
        config = RecoveryDaemonConfig()
        assert config.grace_period_seconds == 30.0
        assert config.max_recovery_attempts == 3
        assert config.recovery_cooldown_seconds == 60.0
        assert config.escalation_enabled is True

    def test_env_prefix(self):
        """Test recovery config env prefix."""
        key = RecoveryDaemonConfig._make_env_key("GRACE_PERIOD")
        assert key == "RINGRIFT_RECOVERY_GRACE_PERIOD"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_empty_string_env_var(self):
        """Test handling of empty string in env var."""
        with patch.dict(os.environ, {"RINGRIFT_EMPTY": ""}):
            # String returns empty (after strip)
            assert BaseCoordinationConfig._get_env_str("EMPTY", "default") == ""
            # Bool returns False (not in true values)
            assert BaseCoordinationConfig._get_env_bool("EMPTY", True) is False

    def test_handles_whitespace_only_env_var(self):
        """Test handling of whitespace-only env var."""
        with patch.dict(os.environ, {"RINGRIFT_WHITESPACE": "   "}):
            assert BaseCoordinationConfig._get_env_str("WHITESPACE", "default") == ""

    def test_int_handles_negative_values(self):
        """Test integer parsing handles negative values."""
        with patch.dict(os.environ, {"RINGRIFT_NEGATIVE": "-42"}):
            result = BaseCoordinationConfig._get_env_int("NEGATIVE", 0)
            assert result == -42

    def test_float_handles_scientific_notation(self):
        """Test float parsing handles scientific notation."""
        with patch.dict(os.environ, {"RINGRIFT_SCIENTIFIC": "1.5e-3"}):
            result = BaseCoordinationConfig._get_env_float("SCIENTIFIC", 0.0)
            assert abs(result - 0.0015) < 0.0001
