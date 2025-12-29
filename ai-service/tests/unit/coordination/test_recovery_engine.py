"""Unit tests for recovery_engine module.

Tests the escalating node recovery engine for cluster resilience.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.availability.recovery_engine import (
    RecoveryAction,
    RecoveryEngine,
    RecoveryEngineConfig,
    RecoveryResult,
    RecoveryState,
)


# =============================================================================
# RecoveryAction Tests
# =============================================================================


class TestRecoveryAction:
    """Tests for RecoveryAction enum."""

    def test_has_restart_p2p(self):
        """RecoveryAction should have RESTART_P2P."""
        assert hasattr(RecoveryAction, "RESTART_P2P")

    def test_has_restart_tailscale(self):
        """RecoveryAction should have RESTART_TAILSCALE."""
        assert hasattr(RecoveryAction, "RESTART_TAILSCALE")

    def test_has_reboot_instance(self):
        """RecoveryAction should have REBOOT_INSTANCE."""
        assert hasattr(RecoveryAction, "REBOOT_INSTANCE")

    def test_has_recreate_instance(self):
        """RecoveryAction should have RECREATE_INSTANCE."""
        assert hasattr(RecoveryAction, "RECREATE_INSTANCE")

    def test_action_names_are_strings(self):
        """RecoveryAction names should be strings."""
        assert isinstance(RecoveryAction.RESTART_P2P.name, str)
        assert isinstance(RecoveryAction.RESTART_TAILSCALE.name, str)


# =============================================================================
# RecoveryResult Tests
# =============================================================================


class TestRecoveryResult:
    """Tests for RecoveryResult dataclass."""

    def test_create_success_result(self):
        """Should create a success recovery result."""
        result = RecoveryResult(
            node_id="test-node",
            action=RecoveryAction.RESTART_P2P,
            success=True,
            duration_seconds=5.0,
        )
        assert result.node_id == "test-node"
        assert result.action == RecoveryAction.RESTART_P2P
        assert result.success is True
        assert result.duration_seconds == 5.0
        assert result.error is None

    def test_create_failure_result(self):
        """Should create a failure recovery result."""
        result = RecoveryResult(
            node_id="test-node",
            action=RecoveryAction.REBOOT_INSTANCE,
            success=False,
            duration_seconds=30.0,
            error="SSH connection failed",
        )
        assert result.success is False
        assert result.error == "SSH connection failed"

    def test_to_dict(self):
        """Should convert to dictionary."""
        result = RecoveryResult(
            node_id="test-node",
            action=RecoveryAction.RESTART_P2P,
            success=True,
            duration_seconds=5.0,
        )
        d = result.to_dict()
        assert d["node_id"] == "test-node"
        assert d["action"] == "RESTART_P2P"
        assert d["success"] is True
        assert d["duration_seconds"] == 5.0
        assert "timestamp" in d

    def test_default_timestamp(self):
        """Should have default timestamp."""
        result = RecoveryResult(
            node_id="test-node",
            action=RecoveryAction.RESTART_P2P,
            success=True,
            duration_seconds=5.0,
        )
        assert isinstance(result.timestamp, datetime)

    def test_default_details(self):
        """Should have empty default details."""
        result = RecoveryResult(
            node_id="test-node",
            action=RecoveryAction.RESTART_P2P,
            success=True,
            duration_seconds=5.0,
        )
        assert result.details == {}


# =============================================================================
# RecoveryState Tests
# =============================================================================


class TestRecoveryState:
    """Tests for RecoveryState dataclass."""

    def test_default_state(self):
        """Should have sensible defaults."""
        state = RecoveryState()
        assert state.current_action_index == 0
        assert state.attempts == {}
        assert state.last_attempt is None
        assert state.last_success is None
        assert state.cooldown_until is None

    def test_track_attempts(self):
        """Should track attempts per action."""
        state = RecoveryState()
        state.attempts[RecoveryAction.RESTART_P2P] = 1
        state.attempts[RecoveryAction.RESTART_TAILSCALE] = 2
        assert state.attempts[RecoveryAction.RESTART_P2P] == 1
        assert state.attempts[RecoveryAction.RESTART_TAILSCALE] == 2

    def test_update_cooldown(self):
        """Should update cooldown_until."""
        state = RecoveryState()
        future = datetime.now() + timedelta(minutes=5)
        state.cooldown_until = future
        assert state.cooldown_until == future


# =============================================================================
# RecoveryEngineConfig Tests
# =============================================================================


class TestRecoveryEngineConfig:
    """Tests for RecoveryEngineConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = RecoveryEngineConfig()
        assert config.check_interval_seconds == 60
        assert config.max_attempts_per_action == 3
        assert config.backoff_base_seconds == 30.0
        assert config.backoff_multiplier == 2.0
        assert config.cooldown_after_success_seconds == 300.0
        assert config.cooldown_after_exhausted_seconds == 3600.0

    def test_enabled_actions_default(self):
        """Should have all actions enabled by default."""
        config = RecoveryEngineConfig()
        assert RecoveryAction.RESTART_P2P in config.enabled_actions
        assert RecoveryAction.RESTART_TAILSCALE in config.enabled_actions
        assert RecoveryAction.REBOOT_INSTANCE in config.enabled_actions
        assert RecoveryAction.RECREATE_INSTANCE in config.enabled_actions

    def test_custom_config(self):
        """Should accept custom configuration."""
        config = RecoveryEngineConfig(
            max_attempts_per_action=5,
            backoff_base_seconds=60.0,
            enabled_actions=[RecoveryAction.RESTART_P2P, RecoveryAction.RESTART_TAILSCALE],
        )
        assert config.max_attempts_per_action == 5
        assert config.backoff_base_seconds == 60.0
        assert len(config.enabled_actions) == 2


# =============================================================================
# RecoveryEngine Tests
# =============================================================================


class TestRecoveryEngineInit:
    """Tests for RecoveryEngine initialization."""

    def test_init_default_config(self):
        """Should initialize with default config."""
        engine = RecoveryEngine()
        assert engine.config is not None
        assert isinstance(engine.config, RecoveryEngineConfig)

    def test_init_custom_config(self):
        """Should accept custom config."""
        config = RecoveryEngineConfig(max_attempts_per_action=5)
        engine = RecoveryEngine(config=config)
        assert engine.config.max_attempts_per_action == 5

    def test_init_empty_recovery_states(self):
        """Should start with empty recovery states."""
        engine = RecoveryEngine()
        assert engine._recovery_states == {}

    def test_init_empty_history(self):
        """Should start with empty history."""
        engine = RecoveryEngine()
        assert engine._recovery_history == []

    def test_get_daemon_name(self):
        """Should return correct daemon name."""
        engine = RecoveryEngine()
        assert engine._get_daemon_name() == "RecoveryEngine"

    def test_get_default_config(self):
        """Should return RecoveryEngineConfig."""
        engine = RecoveryEngine()
        config = engine._get_default_config()
        assert isinstance(config, RecoveryEngineConfig)


class TestRecoveryEngineEventSubscriptions:
    """Tests for RecoveryEngine event subscriptions."""

    def test_subscribes_to_node_unhealthy(self):
        """Should subscribe to NODE_UNHEALTHY."""
        engine = RecoveryEngine()
        subs = engine._get_event_subscriptions()
        assert "NODE_UNHEALTHY" in subs

    def test_subscribes_to_recovery_initiated(self):
        """Should subscribe to RECOVERY_INITIATED."""
        engine = RecoveryEngine()
        subs = engine._get_event_subscriptions()
        assert "RECOVERY_INITIATED" in subs

    def test_subscribes_to_node_recovered(self):
        """Should subscribe to NODE_RECOVERED."""
        engine = RecoveryEngine()
        subs = engine._get_event_subscriptions()
        assert "NODE_RECOVERED" in subs


class TestRecoveryEngineOnNodeUnhealthy:
    """Tests for _on_node_unhealthy handler."""

    @pytest.mark.asyncio
    async def test_queues_recovery_for_unhealthy_node(self):
        """Should queue recovery for unhealthy node."""
        engine = RecoveryEngine()
        event = {
            "payload": {
                "node_id": "test-node",
                "layer": "p2p",
                "error": "Connection refused",
            }
        }

        await engine._on_node_unhealthy(event)

        # Check queue has the item
        assert not engine._recovery_queue.empty()
        node_id, health_result = await engine._recovery_queue.get()
        assert node_id == "test-node"
        assert health_result.healthy is False

    @pytest.mark.asyncio
    async def test_handles_missing_node_id(self):
        """Should handle missing node_id gracefully."""
        engine = RecoveryEngine()
        event = {"payload": {"layer": "p2p"}}

        await engine._on_node_unhealthy(event)

        # Queue should be empty
        assert engine._recovery_queue.empty()


class TestRecoveryEngineOnNodeRecovered:
    """Tests for _on_node_recovered handler."""

    @pytest.mark.asyncio
    async def test_resets_state_for_recovered_node(self):
        """Should reset recovery state for recovered node."""
        engine = RecoveryEngine()
        # Set up existing state
        engine._recovery_states["test-node"] = RecoveryState(
            current_action_index=2,
            attempts={RecoveryAction.RESTART_P2P: 3},
        )

        event = {"payload": {"node_id": "test-node"}}
        await engine._on_node_recovered(event)

        # State should be reset
        state = engine._recovery_states["test-node"]
        assert state.current_action_index == 0
        assert state.attempts == {}
        assert state.last_success is not None

    @pytest.mark.asyncio
    async def test_ignores_unknown_node(self):
        """Should not create state for unknown node."""
        engine = RecoveryEngine()
        event = {"payload": {"node_id": "unknown-node"}}

        await engine._on_node_recovered(event)

        assert "unknown-node" not in engine._recovery_states


class TestRecoveryEngineEscalation:
    """Tests for escalation order."""

    def test_escalation_order(self):
        """Should have correct escalation order."""
        assert RecoveryEngine.ESCALATION_ORDER == [
            RecoveryAction.RESTART_P2P,
            RecoveryAction.RESTART_TAILSCALE,
            RecoveryAction.REBOOT_INSTANCE,
            RecoveryAction.RECREATE_INSTANCE,
        ]

    def test_escalation_order_length(self):
        """Escalation order should have 4 actions."""
        assert len(RecoveryEngine.ESCALATION_ORDER) == 4


class TestRecoveryEngineRunCycle:
    """Tests for _run_cycle method."""

    @pytest.mark.asyncio
    async def test_processes_queue_items(self):
        """Should process items from queue."""
        engine = RecoveryEngine()
        engine._attempt_recovery = AsyncMock()

        # Add item to queue
        from app.coordination.availability.node_monitor import (
            HealthCheckLayer,
            NodeHealthResult,
        )

        health_result = NodeHealthResult(
            node_id="test-node",
            layer=HealthCheckLayer.P2P,
            healthy=False,
            latency_ms=0.0,
        )
        await engine._recovery_queue.put(("test-node", health_result))

        await engine._run_cycle()

        engine._attempt_recovery.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_empty_queue(self):
        """Should handle empty queue gracefully."""
        engine = RecoveryEngine()
        engine._attempt_recovery = AsyncMock()

        # No items in queue
        await engine._run_cycle()

        engine._attempt_recovery.assert_not_called()

    @pytest.mark.asyncio
    async def test_processes_max_5_items(self):
        """Should process at most 5 items per cycle."""
        engine = RecoveryEngine()
        engine._attempt_recovery = AsyncMock()

        from app.coordination.availability.node_monitor import (
            HealthCheckLayer,
            NodeHealthResult,
        )

        # Add 10 items
        for i in range(10):
            health_result = NodeHealthResult(
                node_id=f"node-{i}",
                layer=HealthCheckLayer.P2P,
                healthy=False,
                latency_ms=0.0,
            )
            await engine._recovery_queue.put((f"node-{i}", health_result))

        await engine._run_cycle()

        assert engine._attempt_recovery.call_count == 5
        assert engine._recovery_queue.qsize() == 5  # 5 remaining


class TestRecoveryEngineCooldown:
    """Tests for cooldown behavior."""

    @pytest.mark.asyncio
    async def test_respects_cooldown(self):
        """Should not attempt recovery during cooldown."""
        engine = RecoveryEngine()
        engine._get_node = AsyncMock(return_value=MagicMock())

        # Set cooldown
        future = datetime.now() + timedelta(minutes=5)
        engine._recovery_states["test-node"] = RecoveryState(cooldown_until=future)

        result = await engine._attempt_recovery("test-node", None)

        assert result is None  # No recovery attempted

    @pytest.mark.asyncio
    async def test_recovers_after_cooldown_expires(self):
        """Should attempt recovery after cooldown expires."""
        engine = RecoveryEngine()
        mock_node = MagicMock()
        mock_node.node_id = "test-node"
        engine._get_node = AsyncMock(return_value=mock_node)
        engine._execute_recovery = AsyncMock(
            return_value=RecoveryResult(
                node_id="test-node",
                action=RecoveryAction.RESTART_P2P,
                success=True,
                duration_seconds=1.0,
            )
        )
        engine._emit_recovery_success = AsyncMock()

        # Set expired cooldown
        past = datetime.now() - timedelta(minutes=5)
        engine._recovery_states["test-node"] = RecoveryState(cooldown_until=past)

        result = await engine._attempt_recovery("test-node", None)

        assert result is not None
        engine._execute_recovery.assert_called_once()


class TestRecoveryEngineGetHistory:
    """Tests for recovery history."""

    def test_history_starts_empty(self):
        """History should start empty."""
        engine = RecoveryEngine()
        assert engine._recovery_history == []

    @pytest.mark.asyncio
    async def test_history_records_attempts(self):
        """History should record recovery attempts."""
        engine = RecoveryEngine()
        mock_node = MagicMock()
        mock_node.node_id = "test-node"
        engine._get_node = AsyncMock(return_value=mock_node)

        result = RecoveryResult(
            node_id="test-node",
            action=RecoveryAction.RESTART_P2P,
            success=True,
            duration_seconds=1.0,
        )
        engine._execute_recovery = AsyncMock(return_value=result)
        engine._emit_recovery_success = AsyncMock()

        await engine._attempt_recovery("test-node", None)

        assert len(engine._recovery_history) == 1
        assert engine._recovery_history[0].node_id == "test-node"
