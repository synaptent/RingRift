"""Tests for RecoveryEngine - escalating node recovery.

Created: Dec 29, 2025
Phase 3: Test coverage for critical untested modules.
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
    get_recovery_engine,
    reset_recovery_engine,
)


# =============================================================================
# RecoveryAction Tests
# =============================================================================


class TestRecoveryAction:
    """Tests for RecoveryAction enum."""

    def test_action_values_exist(self) -> None:
        """Test all recovery actions exist."""
        assert RecoveryAction.RESTART_P2P
        assert RecoveryAction.RESTART_TAILSCALE
        assert RecoveryAction.REBOOT_INSTANCE
        assert RecoveryAction.RECREATE_INSTANCE

    def test_timeout_seconds_property(self) -> None:
        """Test timeout_seconds property for each action."""
        assert RecoveryAction.RESTART_P2P.timeout_seconds == 30
        assert RecoveryAction.RESTART_TAILSCALE.timeout_seconds == 60
        assert RecoveryAction.REBOOT_INSTANCE.timeout_seconds == 180
        assert RecoveryAction.RECREATE_INSTANCE.timeout_seconds == 600

    def test_description_property(self) -> None:
        """Test description property for each action."""
        assert "P2P" in RecoveryAction.RESTART_P2P.description
        assert "Tailscale" in RecoveryAction.RESTART_TAILSCALE.description
        assert "Reboot" in RecoveryAction.REBOOT_INSTANCE.description
        assert "recreate" in RecoveryAction.RECREATE_INSTANCE.description.lower()

    def test_escalation_order(self) -> None:
        """Test escalation order is correct (least to most disruptive)."""
        expected_order = [
            RecoveryAction.RESTART_P2P,
            RecoveryAction.RESTART_TAILSCALE,
            RecoveryAction.REBOOT_INSTANCE,
            RecoveryAction.RECREATE_INSTANCE,
        ]
        assert RecoveryEngine.ESCALATION_ORDER == expected_order


# =============================================================================
# RecoveryResult Tests
# =============================================================================


class TestRecoveryResult:
    """Tests for RecoveryResult dataclass."""

    def test_basic_result(self) -> None:
        """Test creating a basic result."""
        result = RecoveryResult(
            node_id="test-node",
            action=RecoveryAction.RESTART_P2P,
            success=True,
            duration_seconds=5.5,
        )
        assert result.node_id == "test-node"
        assert result.action == RecoveryAction.RESTART_P2P
        assert result.success is True
        assert result.duration_seconds == 5.5
        assert result.error is None

    def test_result_with_error(self) -> None:
        """Test result with error."""
        result = RecoveryResult(
            node_id="test-node",
            action=RecoveryAction.RESTART_P2P,
            success=False,
            duration_seconds=10.0,
            error="SSH connection failed",
        )
        assert result.success is False
        assert result.error == "SSH connection failed"

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        result = RecoveryResult(
            node_id="test-node",
            action=RecoveryAction.RESTART_P2P,
            success=True,
            duration_seconds=5.5,
        )
        result_dict = result.to_dict()

        assert result_dict["node_id"] == "test-node"
        assert result_dict["action"] == "RESTART_P2P"
        assert result_dict["success"] is True
        assert result_dict["duration_seconds"] == 5.5
        assert "timestamp" in result_dict
        assert isinstance(result_dict["details"], dict)


# =============================================================================
# RecoveryState Tests
# =============================================================================


class TestRecoveryState:
    """Tests for RecoveryState dataclass."""

    def test_default_state(self) -> None:
        """Test default state values."""
        state = RecoveryState()
        assert state.current_action_index == 0
        assert state.attempts == {}
        assert state.last_attempt is None
        assert state.last_success is None
        assert state.cooldown_until is None

    def test_state_with_attempts(self) -> None:
        """Test state with attempt tracking."""
        state = RecoveryState(
            current_action_index=1,
            attempts={RecoveryAction.RESTART_P2P: 3},
            last_attempt=datetime.now(),
        )
        assert state.current_action_index == 1
        assert state.attempts[RecoveryAction.RESTART_P2P] == 3


# =============================================================================
# RecoveryEngineConfig Tests
# =============================================================================


class TestRecoveryEngineConfig:
    """Tests for RecoveryEngineConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = RecoveryEngineConfig()
        assert config.check_interval_seconds == 60
        assert config.max_attempts_per_action == 3
        assert config.backoff_base_seconds == 30.0
        assert config.backoff_multiplier == 2.0
        assert config.cooldown_after_success_seconds == 300.0
        assert config.cooldown_after_exhausted_seconds == 3600.0

    def test_enabled_actions_default(self) -> None:
        """Test default enabled actions."""
        config = RecoveryEngineConfig()
        assert RecoveryAction.RESTART_P2P in config.enabled_actions
        assert RecoveryAction.RESTART_TAILSCALE in config.enabled_actions
        assert RecoveryAction.REBOOT_INSTANCE in config.enabled_actions
        assert RecoveryAction.RECREATE_INSTANCE in config.enabled_actions

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = RecoveryEngineConfig(
            max_attempts_per_action=5,
            backoff_base_seconds=60.0,
            enabled_actions=[RecoveryAction.RESTART_P2P],
        )
        assert config.max_attempts_per_action == 5
        assert config.backoff_base_seconds == 60.0
        assert config.enabled_actions == [RecoveryAction.RESTART_P2P]


# =============================================================================
# RecoveryEngine Tests
# =============================================================================


class TestRecoveryEngineInit:
    """Tests for RecoveryEngine initialization."""

    def test_default_init(self) -> None:
        """Test default initialization."""
        engine = RecoveryEngine()
        assert engine._recovery_states == {}
        assert engine._recovery_history == []
        assert engine._recovery_queue is not None

    def test_init_with_config(self) -> None:
        """Test initialization with custom config."""
        config = RecoveryEngineConfig(max_attempts_per_action=5)
        engine = RecoveryEngine(config=config)
        assert engine.config.max_attempts_per_action == 5

    def test_get_default_config(self) -> None:
        """Test _get_default_config method."""
        engine = RecoveryEngine()
        config = engine._get_default_config()
        assert isinstance(config, RecoveryEngineConfig)

    def test_get_daemon_name(self) -> None:
        """Test daemon name."""
        engine = RecoveryEngine()
        assert engine._get_daemon_name() == "RecoveryEngine"


class TestRecoveryEngineEventSubscriptions:
    """Tests for event subscriptions."""

    def test_get_event_subscriptions(self) -> None:
        """Test event subscription registration."""
        engine = RecoveryEngine()
        subs = engine._get_event_subscriptions()

        assert "NODE_UNHEALTHY" in subs
        assert "RECOVERY_INITIATED" in subs
        assert "NODE_RECOVERED" in subs

    @pytest.mark.asyncio
    async def test_on_node_unhealthy(self) -> None:
        """Test NODE_UNHEALTHY event handler."""
        engine = RecoveryEngine()

        event = {
            "payload": {
                "node_id": "test-node",
                "layer": "p2p",
                "error": "Connection refused",
            }
        }

        await engine._on_node_unhealthy(event)

        # Check queue has item
        node_id, health_result = await engine._recovery_queue.get()
        assert node_id == "test-node"
        assert health_result.healthy is False

    @pytest.mark.asyncio
    async def test_on_node_recovered(self) -> None:
        """Test NODE_RECOVERED event handler."""
        engine = RecoveryEngine()

        # Setup initial state
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
    async def test_on_recovery_initiated(self) -> None:
        """Test RECOVERY_INITIATED event handler."""
        engine = RecoveryEngine()

        event = {
            "payload": {
                "node_id": "test-node",
                "health_result": {"healthy": False},
            }
        }

        await engine._on_recovery_initiated(event)

        # Check queue has item
        node_id, health_result = await engine._recovery_queue.get()
        assert node_id == "test-node"


class TestRecoveryEngineCycle:
    """Tests for recovery cycle."""

    @pytest.mark.asyncio
    async def test_run_cycle_empty_queue(self) -> None:
        """Test cycle with empty queue."""
        engine = RecoveryEngine()
        await engine._run_cycle()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_run_cycle_with_items(self) -> None:
        """Test cycle processes queue items."""
        engine = RecoveryEngine()

        # Mock the recovery attempt
        engine._attempt_recovery = AsyncMock(return_value=None)

        # Queue some items
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

        # Attempt recovery should be called
        engine._attempt_recovery.assert_called_once()


class TestRecoveryEngineBackoff:
    """Tests for backoff calculation."""

    def test_backoff_first_attempt(self) -> None:
        """Test backoff for first attempt."""
        engine = RecoveryEngine()
        backoff = engine._get_backoff(RecoveryAction.RESTART_P2P, 1)
        assert backoff == 30.0  # base

    def test_backoff_second_attempt(self) -> None:
        """Test backoff for second attempt."""
        engine = RecoveryEngine()
        backoff = engine._get_backoff(RecoveryAction.RESTART_P2P, 2)
        assert backoff == 60.0  # base * 2

    def test_backoff_third_attempt(self) -> None:
        """Test backoff for third attempt."""
        engine = RecoveryEngine()
        backoff = engine._get_backoff(RecoveryAction.RESTART_P2P, 3)
        assert backoff == 120.0  # base * 4

    def test_backoff_cap(self) -> None:
        """Test backoff is capped at 300 seconds."""
        engine = RecoveryEngine()
        backoff = engine._get_backoff(RecoveryAction.RESTART_P2P, 10)
        assert backoff == 300.0  # Capped


class TestRecoveryEngineCooldown:
    """Tests for cooldown calculation."""

    def test_cooldown_after_success(self) -> None:
        """Test cooldown after successful recovery."""
        engine = RecoveryEngine()
        cooldown = engine._get_cooldown(success=True)
        assert cooldown.total_seconds() == 300.0

    def test_cooldown_after_exhausted(self) -> None:
        """Test cooldown after all actions exhausted."""
        engine = RecoveryEngine()
        cooldown = engine._get_cooldown(success=False)
        assert cooldown.total_seconds() == 3600.0


class TestRecoveryEngineState:
    """Tests for state management."""

    def test_get_recovery_state_nonexistent(self) -> None:
        """Test getting state for nonexistent node."""
        engine = RecoveryEngine()
        state = engine.get_recovery_state("nonexistent")
        assert state is None

    def test_get_recovery_state_existing(self) -> None:
        """Test getting state for existing node."""
        engine = RecoveryEngine()
        engine._recovery_states["test-node"] = RecoveryState(
            current_action_index=1,
            attempts={RecoveryAction.RESTART_P2P: 2},
            last_attempt=datetime.now(),
        )

        state = engine.get_recovery_state("test-node")
        assert state is not None
        assert state["node_id"] == "test-node"
        assert state["current_action_index"] == 1
        assert "RESTART_P2P" in state["attempts"]

    def test_get_recovery_state_cooldown(self) -> None:
        """Test state shows cooldown status."""
        engine = RecoveryEngine()
        engine._recovery_states["test-node"] = RecoveryState(
            cooldown_until=datetime.now() + timedelta(seconds=60)
        )

        state = engine.get_recovery_state("test-node")
        assert state["in_cooldown"] is True


class TestRecoveryEngineHistory:
    """Tests for recovery history."""

    def test_get_recovery_history_empty(self) -> None:
        """Test getting empty history."""
        engine = RecoveryEngine()
        history = engine.get_recovery_history()
        assert history == []

    def test_get_recovery_history_with_entries(self) -> None:
        """Test getting history with entries."""
        engine = RecoveryEngine()
        engine._recovery_history = [
            RecoveryResult(
                node_id="node1",
                action=RecoveryAction.RESTART_P2P,
                success=True,
                duration_seconds=5.0,
            ),
            RecoveryResult(
                node_id="node2",
                action=RecoveryAction.RESTART_P2P,
                success=False,
                duration_seconds=10.0,
                error="Failed",
            ),
        ]

        history = engine.get_recovery_history()
        assert len(history) == 2
        assert history[0]["node_id"] == "node1"
        assert history[1]["node_id"] == "node2"

    def test_get_recovery_history_limit(self) -> None:
        """Test history limit."""
        engine = RecoveryEngine()
        # Add 60 entries
        for i in range(60):
            engine._recovery_history.append(
                RecoveryResult(
                    node_id=f"node{i}",
                    action=RecoveryAction.RESTART_P2P,
                    success=True,
                    duration_seconds=1.0,
                )
            )

        history = engine.get_recovery_history(limit=10)
        assert len(history) == 10
        # Should be the last 10
        assert history[0]["node_id"] == "node50"


class TestRecoveryEngineHealthCheck:
    """Tests for health check."""

    def test_health_check_healthy(self) -> None:
        """Test health check with no issues."""
        engine = RecoveryEngine()
        health = engine.health_check()

        assert health["healthy"] is True
        assert "nodes_in_recovery" in health["details"]
        assert "queue_size" in health["details"]

    def test_health_check_with_nodes_in_recovery(self) -> None:
        """Test health check with nodes being recovered."""
        engine = RecoveryEngine()
        engine._recovery_states["node1"] = RecoveryState(current_action_index=2)
        engine._recovery_states["node2"] = RecoveryState(current_action_index=0)

        health = engine.health_check()

        assert health["details"]["nodes_in_recovery"] == 1  # Only node1


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_recovery_engine()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_recovery_engine()

    def test_get_recovery_engine_creates_instance(self) -> None:
        """Test get_recovery_engine creates instance."""
        engine = get_recovery_engine()
        assert engine is not None
        assert isinstance(engine, RecoveryEngine)

    def test_get_recovery_engine_returns_same_instance(self) -> None:
        """Test get_recovery_engine returns same instance."""
        engine1 = get_recovery_engine()
        engine2 = get_recovery_engine()
        assert engine1 is engine2

    def test_reset_recovery_engine(self) -> None:
        """Test reset creates new instance."""
        engine1 = get_recovery_engine()
        reset_recovery_engine()
        engine2 = get_recovery_engine()
        assert engine1 is not engine2


# =============================================================================
# Integration Tests
# =============================================================================


class TestRecoveryAttempt:
    """Integration tests for recovery attempts."""

    @pytest.mark.asyncio
    async def test_attempt_recovery_no_node(self) -> None:
        """Test recovery attempt for unknown node."""
        engine = RecoveryEngine()
        engine._get_node = AsyncMock(return_value=None)

        result = await engine._attempt_recovery("unknown-node", None)
        assert result is None

    @pytest.mark.asyncio
    async def test_attempt_recovery_in_cooldown(self) -> None:
        """Test recovery skips nodes in cooldown."""
        engine = RecoveryEngine()
        engine._recovery_states["test-node"] = RecoveryState(
            cooldown_until=datetime.now() + timedelta(seconds=60)
        )

        result = await engine._attempt_recovery("test-node", None)
        assert result is None

    @pytest.mark.asyncio
    async def test_attempt_recovery_success(self) -> None:
        """Test successful recovery."""
        engine = RecoveryEngine()

        # Mock node
        mock_node = MagicMock()
        mock_node.name = "test-node"
        mock_node.best_ip = "192.168.1.1"
        mock_node.ssh_user = "root"
        mock_node.ssh_port = 22
        engine._get_node = AsyncMock(return_value=mock_node)

        # Mock execute_recovery to succeed
        success_result = RecoveryResult(
            node_id="test-node",
            action=RecoveryAction.RESTART_P2P,
            success=True,
            duration_seconds=5.0,
        )
        engine._execute_recovery = AsyncMock(return_value=success_result)
        engine._emit_recovery_success = AsyncMock()

        result = await engine._attempt_recovery("test-node", None)

        assert result is not None
        assert result.success is True
        engine._emit_recovery_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_recovery_restart_p2p(self) -> None:
        """Test P2P restart execution."""
        engine = RecoveryEngine()

        mock_node = MagicMock()
        mock_node.name = "test-node"
        mock_node.best_ip = "192.168.1.1"
        mock_node.ssh_user = "root"
        mock_node.ssh_port = 22

        # Mock subprocess
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.wait = AsyncMock(return_value=0)
            mock_exec.return_value = mock_proc

            engine._verify_recovery = AsyncMock(return_value=True)
            result = await engine._execute_recovery(mock_node, RecoveryAction.RESTART_P2P)

            assert result.success is True
            assert result.action == RecoveryAction.RESTART_P2P
