"""Unit tests for availability/recovery_engine.py.

Tests for RecoveryEngine daemon that performs escalating node recovery.

Created: Dec 28, 2025
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.availability.recovery_engine import (
    RecoveryAction,
    RecoveryEngineConfig,
    RecoveryEngine,
    RecoveryResult,
    RecoveryState,
    get_recovery_engine,
    reset_recovery_engine,
)
from app.coordination.availability.node_monitor import NodeHealthResult


class TestRecoveryAction:
    """Tests for RecoveryAction enum."""

    def test_action_values(self):
        """Test all action values exist."""
        assert RecoveryAction.RESTART_P2P is not None
        assert RecoveryAction.RESTART_TAILSCALE is not None
        assert RecoveryAction.REBOOT_INSTANCE is not None
        assert RecoveryAction.RECREATE_INSTANCE is not None

    def test_action_timeout(self):
        """Test action timeout properties."""
        assert RecoveryAction.RESTART_P2P.timeout_seconds == 30
        assert RecoveryAction.RESTART_TAILSCALE.timeout_seconds == 60
        assert RecoveryAction.REBOOT_INSTANCE.timeout_seconds == 180
        assert RecoveryAction.RECREATE_INSTANCE.timeout_seconds == 600

    def test_escalation_order(self):
        """Test actions have correct escalation order."""
        # RESTART_P2P should be tried first
        assert RecoveryAction.RESTART_P2P.value < RecoveryAction.RESTART_TAILSCALE.value
        assert RecoveryAction.RESTART_TAILSCALE.value < RecoveryAction.REBOOT_INSTANCE.value
        assert RecoveryAction.REBOOT_INSTANCE.value < RecoveryAction.RECREATE_INSTANCE.value

    def test_action_description(self):
        """Test action descriptions."""
        assert "P2P" in RecoveryAction.RESTART_P2P.description
        assert "Tailscale" in RecoveryAction.RESTART_TAILSCALE.description


class TestRecoveryState:
    """Tests for RecoveryState dataclass."""

    def test_default_state(self):
        """Test default recovery state."""
        state = RecoveryState()
        assert state.current_action_index == 0
        assert state.attempts == {}
        assert state.last_attempt is None
        assert state.last_success is None
        assert state.cooldown_until is None

    def test_state_with_attempts(self):
        """Test state with prior attempts."""
        now = datetime.now()
        state = RecoveryState(
            current_action_index=2,  # REBOOT_INSTANCE
            attempts={RecoveryAction.RESTART_P2P: 3, RecoveryAction.RESTART_TAILSCALE: 2},
            last_attempt=now,
        )
        assert state.current_action_index == 2
        assert state.attempts[RecoveryAction.RESTART_P2P] == 3
        assert state.last_attempt == now


class TestRecoveryResult:
    """Tests for RecoveryResult dataclass."""

    def test_successful_result(self):
        """Test successful recovery result."""
        result = RecoveryResult(
            node_id="test-node",
            action=RecoveryAction.RESTART_P2P,
            success=True,
            duration_seconds=5.0,
        )
        assert result.success
        assert result.node_id == "test-node"
        assert result.error is None

    def test_failed_result(self):
        """Test failed recovery result."""
        result = RecoveryResult(
            node_id="test-node",
            action=RecoveryAction.RESTART_TAILSCALE,
            success=False,
            duration_seconds=60.0,
            error="Connection timeout",
        )
        assert not result.success
        assert "timeout" in result.error.lower()

    def test_to_dict(self):
        """Test serialization to dict."""
        result = RecoveryResult(
            node_id="test-node",
            action=RecoveryAction.RESTART_P2P,
            success=True,
            duration_seconds=2.5,
        )
        d = result.to_dict()
        assert d["node_id"] == "test-node"
        assert d["action"] == "RESTART_P2P"
        assert d["success"] is True
        assert d["duration_seconds"] == 2.5


class TestRecoveryEngineConfig:
    """Tests for RecoveryEngineConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RecoveryEngineConfig()
        assert config.max_attempts_per_action == 3
        assert config.backoff_multiplier == 2.0
        assert config.backoff_base_seconds == 30.0
        assert config.cooldown_after_success_seconds == 300.0
        assert config.cooldown_after_exhausted_seconds == 3600.0

    def test_enabled_actions(self):
        """Test default enabled actions include all recovery types."""
        config = RecoveryEngineConfig()
        assert RecoveryAction.RESTART_P2P in config.enabled_actions
        assert RecoveryAction.RESTART_TAILSCALE in config.enabled_actions
        assert RecoveryAction.REBOOT_INSTANCE in config.enabled_actions
        assert RecoveryAction.RECREATE_INSTANCE in config.enabled_actions

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RecoveryEngineConfig(
            max_attempts_per_action=5,
            backoff_base_seconds=60.0,
            enabled_actions=[RecoveryAction.RESTART_P2P, RecoveryAction.RESTART_TAILSCALE],
        )
        assert config.max_attempts_per_action == 5
        assert config.backoff_base_seconds == 60.0
        assert len(config.enabled_actions) == 2


class TestRecoveryEngine:
    """Tests for RecoveryEngine daemon."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_recovery_engine()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_recovery_engine()

    def test_singleton_pattern(self):
        """Test that get_recovery_engine returns singleton."""
        engine1 = get_recovery_engine()
        engine2 = get_recovery_engine()

        assert engine1 is engine2

    def test_event_subscriptions(self):
        """Test event subscription setup."""
        engine = get_recovery_engine()

        subs = engine._get_event_subscriptions()

        # Should subscribe to NODE_UNHEALTHY
        assert "NODE_UNHEALTHY" in subs
        assert "RECOVERY_INITIATED" in subs
        assert "NODE_RECOVERED" in subs

    def test_get_recovery_state(self):
        """Test getting recovery state for a node."""
        engine = get_recovery_engine()

        # Initially no state
        state = engine.get_recovery_state("unknown-node")
        assert state is None

    def test_get_recovery_history(self):
        """Test getting recovery history."""
        engine = get_recovery_engine()

        history = engine.get_recovery_history()
        assert isinstance(history, list)
        assert len(history) == 0  # Empty initially

    def test_health_check(self):
        """Test health_check method returns valid result."""
        engine = get_recovery_engine()

        result = engine.health_check()

        assert result.healthy is True
        assert result.message is not None
        assert result.details is not None
        assert "queue_size" in result.details


class TestRecoveryEngineEventHandlers:
    """Tests for RecoveryEngine event handlers."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_recovery_engine()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_recovery_engine()

    @pytest.mark.asyncio
    async def test_on_node_unhealthy(self):
        """Test handling NODE_UNHEALTHY event."""
        engine = get_recovery_engine()

        event = {
            "payload": {
                "node_id": "test-node",
                "layer": "p2p",
                "error": "Connection refused",
            }
        }

        await engine._on_node_unhealthy(event)

        # Node should be queued for recovery
        assert engine._recovery_queue.qsize() >= 1

    @pytest.mark.asyncio
    async def test_on_node_recovered(self):
        """Test handling NODE_RECOVERED event."""
        engine = get_recovery_engine()

        # Add a recovery state first
        engine._recovery_states["test-node"] = RecoveryState(
            current_action_index=1,
            attempts={RecoveryAction.RESTART_P2P: 2},
        )

        event = {
            "payload": {
                "node_id": "test-node",
            }
        }

        await engine._on_node_recovered(event)

        # State should be reset
        state = engine._recovery_states.get("test-node")
        assert state is not None
        assert state.last_success is not None


class TestRecoveryEngineActions:
    """Tests for recovery action execution."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_recovery_engine()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_recovery_engine()

    def test_escalation_order_constant(self):
        """Test ESCALATION_ORDER constant is correct."""
        engine = get_recovery_engine()

        assert RecoveryEngine.ESCALATION_ORDER == [
            RecoveryAction.RESTART_P2P,
            RecoveryAction.RESTART_TAILSCALE,
            RecoveryAction.REBOOT_INSTANCE,
            RecoveryAction.RECREATE_INSTANCE,
        ]

    def test_recovery_state_tracking(self):
        """Test that recovery states are tracked per node."""
        engine = get_recovery_engine()

        # Add recovery states for different nodes
        engine._recovery_states["node-1"] = RecoveryState(current_action_index=1)
        engine._recovery_states["node-2"] = RecoveryState(current_action_index=2)

        state1 = engine.get_recovery_state("node-1")
        state2 = engine.get_recovery_state("node-2")

        assert state1 is not None
        assert state2 is not None
        assert state1["current_action_index"] == 1
        assert state2["current_action_index"] == 2
