"""Unit tests for availability/recovery_engine.py.

Tests the escalating node recovery engine that provides:
- RecoveryResult/RecoveryState data classes
- RecoveryEngineConfig configuration
- RecoveryEngine daemon with escalating strategies

This is critical infrastructure (666 LOC) for automated cluster recovery.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.availability.recovery_engine import (
    RecoveryEngine,
    RecoveryEngineConfig,
    RecoveryResult,
    RecoveryState,
    get_recovery_engine,
    reset_recovery_engine,
)
from app.coordination.enums import SystemRecoveryAction

# Use canonical name for tests
RecoveryAction = SystemRecoveryAction


# =============================================================================
# RecoveryResult Tests
# =============================================================================


class TestRecoveryResult:
    """Tests for RecoveryResult dataclass."""

    def test_recovery_result_creation(self) -> None:
        """Test creating a RecoveryResult."""
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
        assert result.timestamp is not None

    def test_recovery_result_with_error(self) -> None:
        """Test RecoveryResult with error."""
        result = RecoveryResult(
            node_id="test-node",
            action=RecoveryAction.RESTART_TAILSCALE,
            success=False,
            duration_seconds=10.0,
            error="SSH connection failed",
        )

        assert result.success is False
        assert result.error == "SSH connection failed"

    def test_recovery_result_to_dict(self) -> None:
        """Test serialization to dictionary."""
        result = RecoveryResult(
            node_id="node-1",
            action=RecoveryAction.REBOOT_INSTANCE,
            success=True,
            duration_seconds=60.0,
            details={"provider": "vast"},
        )

        d = result.to_dict()

        assert d["node_id"] == "node-1"
        assert d["action"] == "REBOOT_INSTANCE"
        assert d["success"] is True
        assert d["duration_seconds"] == 60.0
        assert "timestamp" in d
        assert d["details"]["provider"] == "vast"


# =============================================================================
# RecoveryState Tests
# =============================================================================


class TestRecoveryState:
    """Tests for RecoveryState dataclass."""

    def test_recovery_state_defaults(self) -> None:
        """Test default RecoveryState."""
        state = RecoveryState()

        assert state.current_action_index == 0
        assert state.attempts == {}
        assert state.last_attempt is None
        assert state.last_success is None
        assert state.cooldown_until is None

    def test_recovery_state_with_attempts(self) -> None:
        """Test RecoveryState with attempts tracking."""
        state = RecoveryState(
            current_action_index=1,
            attempts={RecoveryAction.RESTART_P2P: 2},
            last_attempt=datetime.now(),
        )

        assert state.current_action_index == 1
        assert state.attempts[RecoveryAction.RESTART_P2P] == 2

    def test_recovery_state_cooldown(self) -> None:
        """Test RecoveryState cooldown tracking."""
        future = datetime.now() + timedelta(minutes=5)
        state = RecoveryState(cooldown_until=future)

        assert state.cooldown_until == future
        assert datetime.now() < state.cooldown_until


# =============================================================================
# RecoveryEngineConfig Tests
# =============================================================================


class TestRecoveryEngineConfig:
    """Tests for RecoveryEngineConfig."""

    def test_config_defaults(self) -> None:
        """Test default configuration."""
        config = RecoveryEngineConfig()

        assert config.check_interval_seconds == 60
        assert config.max_attempts_per_action == 3
        assert config.backoff_base_seconds == 30.0
        assert config.backoff_multiplier == 2.0
        assert config.cooldown_after_success_seconds == 300.0
        assert config.cooldown_after_exhausted_seconds == 3600.0

    def test_config_enabled_actions(self) -> None:
        """Test enabled_actions default."""
        config = RecoveryEngineConfig()

        assert RecoveryAction.RESTART_P2P in config.enabled_actions
        assert RecoveryAction.RESTART_TAILSCALE in config.enabled_actions
        assert RecoveryAction.REBOOT_INSTANCE in config.enabled_actions
        assert RecoveryAction.RECREATE_INSTANCE in config.enabled_actions

    def test_config_custom(self) -> None:
        """Test custom configuration."""
        config = RecoveryEngineConfig(
            max_attempts_per_action=5,
            enabled_actions=[RecoveryAction.RESTART_P2P],
        )

        assert config.max_attempts_per_action == 5
        assert len(config.enabled_actions) == 1


# =============================================================================
# RecoveryEngine Tests
# =============================================================================


class TestRecoveryEngine:
    """Tests for RecoveryEngine class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self) -> None:
        """Reset singleton before each test."""
        reset_recovery_engine()
        yield
        reset_recovery_engine()

    def test_engine_creation(self) -> None:
        """Test creating RecoveryEngine."""
        engine = RecoveryEngine()

        assert engine._recovery_states == {}
        assert engine._recovery_history == []

    def test_engine_with_config(self) -> None:
        """Test engine with custom config."""
        config = RecoveryEngineConfig(max_attempts_per_action=5)
        engine = RecoveryEngine(config=config)

        assert engine.config.max_attempts_per_action == 5

    def test_daemon_name(self) -> None:
        """Test daemon name."""
        engine = RecoveryEngine()
        assert engine._get_daemon_name() == "RecoveryEngine"

    def test_event_subscriptions(self) -> None:
        """Test event subscriptions."""
        engine = RecoveryEngine()
        subs = engine._get_event_subscriptions()

        assert "NODE_UNHEALTHY" in subs
        assert "RECOVERY_INITIATED" in subs
        assert "NODE_RECOVERED" in subs

    def test_escalation_order(self) -> None:
        """Test escalation order is correct."""
        assert RecoveryEngine.ESCALATION_ORDER == [
            RecoveryAction.RESTART_P2P,
            RecoveryAction.RESTART_TAILSCALE,
            RecoveryAction.REBOOT_INSTANCE,
            RecoveryAction.RECREATE_INSTANCE,
        ]

    def test_backoff_calculation(self) -> None:
        """Test backoff calculation."""
        engine = RecoveryEngine()

        # First attempt
        backoff1 = engine._get_backoff(RecoveryAction.RESTART_P2P, 1)
        assert backoff1 == 30.0

        # Second attempt (30 * 2^1 = 60)
        backoff2 = engine._get_backoff(RecoveryAction.RESTART_P2P, 2)
        assert backoff2 == 60.0

        # Third attempt (30 * 2^2 = 120)
        backoff3 = engine._get_backoff(RecoveryAction.RESTART_P2P, 3)
        assert backoff3 == 120.0

    def test_backoff_max(self) -> None:
        """Test backoff is capped at 300 seconds."""
        engine = RecoveryEngine()

        # High attempt should be capped
        backoff = engine._get_backoff(RecoveryAction.RESTART_P2P, 10)
        assert backoff == 300.0

    def test_cooldown_success(self) -> None:
        """Test cooldown after success."""
        engine = RecoveryEngine()
        cooldown = engine._get_cooldown(success=True)

        assert cooldown.total_seconds() == 300.0

    def test_cooldown_exhausted(self) -> None:
        """Test cooldown after all actions exhausted."""
        engine = RecoveryEngine()
        cooldown = engine._get_cooldown(success=False)

        assert cooldown.total_seconds() == 3600.0

    def test_health_check(self) -> None:
        """Test health check returns proper status."""
        engine = RecoveryEngine()

        # Empty state
        health = engine.health_check()

        assert health["healthy"] is True
        assert "Recovery engine:" in health["message"]
        assert health["details"]["nodes_in_recovery"] == 0
        assert health["details"]["queue_size"] == 0

    def test_health_check_with_state(self) -> None:
        """Test health check with recovery state."""
        engine = RecoveryEngine()

        # Add a node in recovery (not at index 0)
        engine._recovery_states["node-1"] = RecoveryState(current_action_index=1)

        health = engine.health_check()
        assert health["details"]["nodes_in_recovery"] == 1

    def test_get_recovery_state_unknown(self) -> None:
        """Test getting state for unknown node."""
        engine = RecoveryEngine()

        state = engine.get_recovery_state("unknown-node")
        assert state is None

    def test_get_recovery_state(self) -> None:
        """Test getting recovery state."""
        engine = RecoveryEngine()
        engine._recovery_states["node-1"] = RecoveryState(
            current_action_index=1,
            attempts={RecoveryAction.RESTART_P2P: 2},
        )

        state = engine.get_recovery_state("node-1")

        assert state is not None
        assert state["node_id"] == "node-1"
        assert state["current_action_index"] == 1
        assert state["attempts"]["RESTART_P2P"] == 2

    def test_get_recovery_history_empty(self) -> None:
        """Test getting empty history."""
        engine = RecoveryEngine()

        history = engine.get_recovery_history()
        assert history == []

    def test_get_recovery_history(self) -> None:
        """Test getting recovery history."""
        engine = RecoveryEngine()
        engine._recovery_history.append(
            RecoveryResult(
                node_id="node-1",
                action=RecoveryAction.RESTART_P2P,
                success=True,
                duration_seconds=5.0,
            )
        )

        history = engine.get_recovery_history()
        assert len(history) == 1
        assert history[0]["node_id"] == "node-1"

    def test_get_recovery_history_limit(self) -> None:
        """Test recovery history respects limit."""
        engine = RecoveryEngine()

        # Add 100 results
        for i in range(100):
            engine._recovery_history.append(
                RecoveryResult(
                    node_id=f"node-{i}",
                    action=RecoveryAction.RESTART_P2P,
                    success=True,
                    duration_seconds=1.0,
                )
            )

        # Get with default limit
        history = engine.get_recovery_history(limit=50)
        assert len(history) == 50

        # Should be the last 50
        assert history[0]["node_id"] == "node-50"

    @pytest.mark.asyncio
    async def test_on_node_unhealthy(self) -> None:
        """Test handling NODE_UNHEALTHY event."""
        engine = RecoveryEngine()

        event = {
            "payload": {
                "node_id": "test-node",
                "layer": "p2p",
                "error": "Connection timeout",
            }
        }

        await engine._on_node_unhealthy(event)

        # Should have queued the node for recovery
        assert engine._recovery_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_on_node_recovered(self) -> None:
        """Test handling NODE_RECOVERED event."""
        engine = RecoveryEngine()

        # Set up recovery state
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
    async def test_run_cycle_empty_queue(self) -> None:
        """Test run cycle with empty queue."""
        engine = RecoveryEngine()

        # Should not raise
        await engine._run_cycle()

    @pytest.mark.asyncio
    async def test_attempt_recovery_cooldown(self) -> None:
        """Test recovery respects cooldown."""
        engine = RecoveryEngine()

        # Set cooldown
        engine._recovery_states["node-1"] = RecoveryState(
            cooldown_until=datetime.now() + timedelta(hours=1)
        )

        result = await engine._attempt_recovery("node-1", None)

        assert result is None

    @pytest.mark.asyncio
    async def test_attempt_recovery_unknown_node(self) -> None:
        """Test recovery with unknown node."""
        engine = RecoveryEngine()

        with patch.object(engine, "_get_node", return_value=None):
            result = await engine._attempt_recovery("unknown-node", None)

        assert result is None


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self) -> None:
        """Reset singleton before each test."""
        reset_recovery_engine()
        yield
        reset_recovery_engine()

    def test_get_recovery_engine(self) -> None:
        """Test getting singleton instance."""
        engine1 = get_recovery_engine()
        engine2 = get_recovery_engine()

        assert engine1 is engine2

    def test_reset_recovery_engine(self) -> None:
        """Test resetting singleton."""
        engine1 = get_recovery_engine()
        reset_recovery_engine()
        engine2 = get_recovery_engine()

        assert engine1 is not engine2


# =============================================================================
# Recovery Action Execution Tests
# =============================================================================


class TestRecoveryActions:
    """Tests for recovery action execution."""

    @pytest.fixture
    def engine(self) -> RecoveryEngine:
        reset_recovery_engine()
        return RecoveryEngine()

    @pytest.fixture
    def mock_node(self) -> MagicMock:
        """Create a mock ClusterNode."""
        node = MagicMock()
        node.name = "test-node"
        node.best_ip = "192.168.1.1"
        node.tailscale_ip = "100.64.1.1"
        node.ssh_user = "root"
        node.ssh_port = 22
        node.provider = "vast"
        node.instance_id = "inst-123"
        node.gpu = "RTX 4090"
        return node

    @pytest.mark.asyncio
    async def test_restart_p2p_no_ip(self, engine: RecoveryEngine) -> None:
        """Test restart P2P fails without IP."""
        node = MagicMock()
        node.best_ip = None
        node.tailscale_ip = None

        success, error = await engine._restart_p2p(node)

        assert success is False
        assert error == "No IP address"

    @pytest.mark.asyncio
    async def test_restart_tailscale_no_ip(self, engine: RecoveryEngine) -> None:
        """Test restart Tailscale fails without IP."""
        node = MagicMock()
        node.best_ip = None
        node.tailscale_ip = None

        success, error = await engine._restart_tailscale(node)

        assert success is False
        assert error == "No IP address"

    @pytest.mark.asyncio
    async def test_reboot_instance_no_provider(self, engine: RecoveryEngine) -> None:
        """Test reboot fails without provider."""
        node = MagicMock()
        node.provider = None

        success, error = await engine._reboot_instance(node)

        assert success is False
        assert error == "No provider configured"

    @pytest.mark.asyncio
    async def test_reboot_instance_no_instance_id(self, engine: RecoveryEngine) -> None:
        """Test reboot fails without instance_id."""
        node = MagicMock()
        node.provider = "vast"
        node.instance_id = None

        success, error = await engine._reboot_instance(node)

        assert success is False
        assert error == "No instance_id configured"

    @pytest.mark.asyncio
    async def test_recreate_instance_no_provider(self, engine: RecoveryEngine) -> None:
        """Test recreate fails without provider."""
        node = MagicMock()
        node.provider = None

        success, error = await engine._recreate_instance(node)

        assert success is False
        assert error == "No provider configured"

    @pytest.mark.asyncio
    async def test_execute_recovery_restart_p2p(
        self, engine: RecoveryEngine, mock_node: MagicMock
    ) -> None:
        """Test execute_recovery for RESTART_P2P."""
        with patch.object(engine, "_restart_p2p", return_value=(True, None)):
            with patch.object(engine, "_verify_recovery", return_value=True):
                result = await engine._execute_recovery(
                    mock_node, RecoveryAction.RESTART_P2P
                )

        assert result.success is True
        assert result.action == RecoveryAction.RESTART_P2P
        assert result.error is None

    @pytest.mark.asyncio
    async def test_execute_recovery_verification_failed(
        self, engine: RecoveryEngine, mock_node: MagicMock
    ) -> None:
        """Test execute_recovery when verification fails."""
        with patch.object(engine, "_restart_p2p", return_value=(True, None)):
            with patch.object(engine, "_verify_recovery", return_value=False):
                result = await engine._execute_recovery(
                    mock_node, RecoveryAction.RESTART_P2P
                )

        assert result.success is False
        assert result.error == "Recovery verification failed"

    @pytest.mark.asyncio
    async def test_execute_recovery_exception(
        self, engine: RecoveryEngine, mock_node: MagicMock
    ) -> None:
        """Test execute_recovery handles exceptions."""
        with patch.object(
            engine, "_restart_p2p", side_effect=Exception("Unexpected error")
        ):
            result = await engine._execute_recovery(
                mock_node, RecoveryAction.RESTART_P2P
            )

        assert result.success is False
        assert "Unexpected error" in result.error


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestEventEmission:
    """Tests for event emission."""

    @pytest.fixture
    def engine(self) -> RecoveryEngine:
        reset_recovery_engine()
        return RecoveryEngine()

    @pytest.mark.asyncio
    async def test_emit_recovery_success(self, engine: RecoveryEngine) -> None:
        """Test emitting RECOVERY_COMPLETED event."""
        result = RecoveryResult(
            node_id="node-1",
            action=RecoveryAction.RESTART_P2P,
            success=True,
            duration_seconds=5.0,
        )

        with patch.object(engine, "_safe_emit_event") as mock_emit:
            await engine._emit_recovery_success(result)

        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        # Event type is lowercase (DataEventType.RECOVERY_COMPLETED.value)
        assert "recovery_completed" in str(call_args)

    @pytest.mark.asyncio
    async def test_emit_recovery_failed(self, engine: RecoveryEngine) -> None:
        """Test emitting RECOVERY_FAILED event."""
        result = RecoveryResult(
            node_id="node-1",
            action=RecoveryAction.RESTART_P2P,
            success=False,
            duration_seconds=5.0,
            error="SSH failed",
        )

        with patch.object(engine, "_safe_emit_event") as mock_emit:
            await engine._emit_recovery_failed(result)

        mock_emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_node_failed_permanently(self, engine: RecoveryEngine) -> None:
        """Test emitting NODE_FAILED_PERMANENTLY event."""
        with patch.object(engine, "_safe_emit_event") as mock_emit:
            await engine._emit_node_failed_permanently("node-1")

        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        assert call_args[0][0] == "NODE_FAILED_PERMANENTLY"
