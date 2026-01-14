"""Tests for app/config/env.py - Centralized environment configuration.

Tests the RingRiftEnv singleton and its cached properties for environment
variable access with proper defaults and type conversion.
"""

from __future__ import annotations

import os
import socket
from unittest.mock import patch

import pytest

from app.config.env import RingRiftEnv, env


class TestRingRiftEnvNodeIdentity:
    """Tests for node identity properties."""

    def test_node_id_from_env(self):
        """node_id should use RINGRIFT_NODE_ID if set."""
        e = RingRiftEnv()
        with patch.dict(os.environ, {"RINGRIFT_NODE_ID": "test-node-123"}):
            # Need a fresh instance to avoid cached_property
            e2 = RingRiftEnv()
            assert e2.node_id == "test-node-123"

    def test_node_id_defaults_to_canonical(self):
        """node_id should resolve to canonical ID if env not set.

        The Unified Node Identity System resolves node IDs from config,
        so the result may differ from the raw hostname. The key requirement
        is that we get a valid, non-empty node ID.
        """
        # Remove the env var if it exists
        env_copy = os.environ.copy()
        env_copy.pop("RINGRIFT_NODE_ID", None)
        with patch.dict(os.environ, env_copy, clear=True):
            e = RingRiftEnv()
            # Should return a non-empty string (canonical ID or hostname fallback)
            assert isinstance(e.node_id, str)
            assert len(e.node_id) > 0

    def test_orchestrator_id_from_env(self):
        """orchestrator_id should use RINGRIFT_ORCHESTRATOR if set."""
        with patch.dict(os.environ, {"RINGRIFT_ORCHESTRATOR": "my-orchestrator"}):
            e = RingRiftEnv()
            assert e.orchestrator_id == "my-orchestrator"

    def test_orchestrator_id_default(self):
        """orchestrator_id should default to 'unknown'."""
        env_copy = os.environ.copy()
        env_copy.pop("RINGRIFT_ORCHESTRATOR", None)
        with patch.dict(os.environ, env_copy, clear=True):
            e = RingRiftEnv()
            assert e.orchestrator_id == "unknown"

    def test_hostname(self):
        """hostname should return socket.gethostname()."""
        e = RingRiftEnv()
        assert e.hostname == socket.gethostname()


class TestRingRiftEnvLogging:
    """Tests for logging configuration properties."""

    def test_log_level_from_env(self):
        """log_level should use RINGRIFT_LOG_LEVEL if set."""
        with patch.dict(os.environ, {"RINGRIFT_LOG_LEVEL": "debug"}):
            e = RingRiftEnv()
            assert e.log_level == "DEBUG"  # Should be uppercased

    def test_log_level_default(self):
        """log_level should default to INFO."""
        env_copy = os.environ.copy()
        env_copy.pop("RINGRIFT_LOG_LEVEL", None)
        with patch.dict(os.environ, env_copy, clear=True):
            e = RingRiftEnv()
            assert e.log_level == "INFO"

    def test_log_json_true(self):
        """log_json should be True when set to 'true'."""
        with patch.dict(os.environ, {"RINGRIFT_LOG_JSON": "true"}):
            e = RingRiftEnv()
            assert e.log_json is True

    def test_log_json_false_default(self):
        """log_json should default to False."""
        env_copy = os.environ.copy()
        env_copy.pop("RINGRIFT_LOG_JSON", None)
        with patch.dict(os.environ, env_copy, clear=True):
            e = RingRiftEnv()
            assert e.log_json is False

    def test_trace_debug_enabled(self):
        """trace_debug should be True when RINGRIFT_TRACE_DEBUG is '1'."""
        with patch.dict(os.environ, {"RINGRIFT_TRACE_DEBUG": "1"}):
            e = RingRiftEnv()
            assert e.trace_debug is True

    def test_trace_debug_disabled(self):
        """trace_debug should be False by default."""
        env_copy = os.environ.copy()
        env_copy.pop("RINGRIFT_TRACE_DEBUG", None)
        with patch.dict(os.environ, env_copy, clear=True):
            e = RingRiftEnv()
            assert e.trace_debug is False


class TestRingRiftEnvResourceManagement:
    """Tests for resource management properties."""

    def test_target_util_min_from_env(self):
        """target_util_min should use RINGRIFT_TARGET_UTIL_MIN if set."""
        with patch.dict(os.environ, {"RINGRIFT_TARGET_UTIL_MIN": "50"}):
            e = RingRiftEnv()
            assert e.target_util_min == 50.0

    def test_target_util_min_default(self):
        """target_util_min should default to 60.0."""
        env_copy = os.environ.copy()
        env_copy.pop("RINGRIFT_TARGET_UTIL_MIN", None)
        with patch.dict(os.environ, env_copy, clear=True):
            e = RingRiftEnv()
            assert e.target_util_min == 60.0

    def test_pid_values(self):
        """PID controller values should have correct defaults."""
        env_copy = os.environ.copy()
        for key in ["RINGRIFT_PID_KP", "RINGRIFT_PID_KI", "RINGRIFT_PID_KD"]:
            env_copy.pop(key, None)
        with patch.dict(os.environ, env_copy, clear=True):
            e = RingRiftEnv()
            assert e.pid_kp == 0.3
            assert e.pid_ki == 0.05
            assert e.pid_kd == 0.1

    def test_idle_check_interval_from_env(self):
        """idle_check_interval should use env var if set."""
        with patch.dict(os.environ, {"RINGRIFT_IDLE_CHECK_INTERVAL": "120"}):
            e = RingRiftEnv()
            assert e.idle_check_interval == 120

    def test_idle_check_interval_default(self):
        """idle_check_interval should default to 60."""
        env_copy = os.environ.copy()
        env_copy.pop("RINGRIFT_IDLE_CHECK_INTERVAL", None)
        with patch.dict(os.environ, env_copy, clear=True):
            e = RingRiftEnv()
            assert e.idle_check_interval == 60


class TestRingRiftEnvFeatureFlags:
    """Tests for feature flag properties."""

    def test_skip_shadow_contracts_true_explicit(self):
        """skip_shadow_contracts should be True when set to '1'."""
        with patch.dict(os.environ, {"RINGRIFT_SKIP_SHADOW_CONTRACTS": "1"}):
            e = RingRiftEnv()
            assert e.skip_shadow_contracts is True

    def test_skip_shadow_contracts_true_by_default(self):
        """skip_shadow_contracts should be True by default.

        Dec 29, 2025: The code defaults to 'true' (skip shadow contracts)
        which is the safe default for cluster nodes that lack Node.js.
        """
        env_copy = os.environ.copy()
        env_copy.pop("RINGRIFT_SKIP_SHADOW_CONTRACTS", None)
        with patch.dict(os.environ, env_copy, clear=True):
            e = RingRiftEnv()
            assert e.skip_shadow_contracts is True

    def test_skip_shadow_contracts_false_explicit(self):
        """skip_shadow_contracts should be False when explicitly set to '0'."""
        with patch.dict(os.environ, {"RINGRIFT_SKIP_SHADOW_CONTRACTS": "0"}):
            e = RingRiftEnv()
            assert e.skip_shadow_contracts is False

    def test_idle_resource_enabled_true(self):
        """idle_resource_enabled should be True when set to '1'."""
        with patch.dict(os.environ, {"RINGRIFT_IDLE_RESOURCE_ENABLED": "1"}):
            e = RingRiftEnv()
            assert e.idle_resource_enabled is True

    def test_idle_resource_enabled_false(self):
        """idle_resource_enabled should be False when set to '0'."""
        with patch.dict(os.environ, {"RINGRIFT_IDLE_RESOURCE_ENABLED": "0"}):
            e = RingRiftEnv()
            assert e.idle_resource_enabled is False


class TestRingRiftEnvHelperMethods:
    """Tests for helper methods."""

    def test_get_with_prefix(self):
        """get should add RINGRIFT_ prefix if not present."""
        with patch.dict(os.environ, {"RINGRIFT_TEST_VAR": "test_value"}):
            e = RingRiftEnv()
            assert e.get("TEST_VAR") == "test_value"

    def test_get_without_prefix(self):
        """get should work with full env var name."""
        with patch.dict(os.environ, {"RINGRIFT_TEST_VAR": "test_value"}):
            e = RingRiftEnv()
            assert e.get("RINGRIFT_TEST_VAR") == "test_value"

    def test_get_default(self):
        """get should return default when env var not set."""
        e = RingRiftEnv()
        assert e.get("NONEXISTENT_VAR", "default") == "default"

    def test_get_int(self):
        """get_int should return integer value."""
        with patch.dict(os.environ, {"RINGRIFT_TEST_INT": "42"}):
            e = RingRiftEnv()
            assert e.get_int("TEST_INT") == 42

    def test_get_int_default(self):
        """get_int should return default for missing/invalid values."""
        e = RingRiftEnv()
        assert e.get_int("NONEXISTENT_INT", 100) == 100

    def test_get_float(self):
        """get_float should return float value."""
        with patch.dict(os.environ, {"RINGRIFT_TEST_FLOAT": "3.14"}):
            e = RingRiftEnv()
            assert e.get_float("TEST_FLOAT") == 3.14

    def test_get_bool_true_values(self):
        """get_bool should return True for '1', 'true', 'yes', 'on'."""
        e = RingRiftEnv()
        for value in ["1", "true", "TRUE", "yes", "YES", "on", "ON"]:
            with patch.dict(os.environ, {"RINGRIFT_TEST_BOOL": value}):
                e2 = RingRiftEnv()
                assert e2.get_bool("TEST_BOOL") is True, f"Failed for value: {value}"

    def test_get_bool_false_values(self):
        """get_bool should return False for other values."""
        e = RingRiftEnv()
        for value in ["0", "false", "no", "off", ""]:
            with patch.dict(os.environ, {"RINGRIFT_TEST_BOOL": value}):
                e2 = RingRiftEnv()
                assert e2.get_bool("TEST_BOOL") is False, f"Failed for value: {value}"

    def test_is_set_true(self):
        """is_set should return True when env var is set."""
        with patch.dict(os.environ, {"RINGRIFT_TEST_SET": "value"}):
            e = RingRiftEnv()
            assert e.is_set("TEST_SET") is True

    def test_is_set_false(self):
        """is_set should return False when env var is not set."""
        env_copy = os.environ.copy()
        env_copy.pop("RINGRIFT_NONEXISTENT", None)
        with patch.dict(os.environ, env_copy, clear=True):
            e = RingRiftEnv()
            assert e.is_set("NONEXISTENT") is False


class TestRingRiftEnvSingleton:
    """Tests for the global env singleton."""

    def test_global_env_exists(self):
        """Global env singleton should exist."""
        assert env is not None
        assert isinstance(env, RingRiftEnv)

    def test_env_repr(self):
        """env repr should include node_id."""
        e = RingRiftEnv()
        repr_str = repr(e)
        assert "RingRiftEnv" in repr_str
        assert "node_id" in repr_str
