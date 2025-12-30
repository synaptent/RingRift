"""Tests for ConnectivityRecoveryCoordinator.

Comprehensive test suite covering:
- RecoveryStrategy enum
- RecoveryAttempt dataclass
- NodeConnectivityState dataclass
- RecoveryConfig dataclass with from_env()
- ConnectivityRecoveryCoordinator event-driven recovery
- Factory functions

December 2025 - Created as part of test coverage improvements.
"""

from __future__ import annotations

import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.connectivity_recovery_coordinator import (
    ConnectivityRecoveryCoordinator,
    NodeConnectivityState,
    RecoveryAttempt,
    RecoveryConfig,
    RecoveryStrategy,
    create_connectivity_recovery_coordinator,
    get_connectivity_recovery_coordinator,
)


# =============================================================================
# RecoveryStrategy Enum Tests
# =============================================================================


class TestRecoveryStrategy:
    """Tests for RecoveryStrategy enum."""

    def test_all_strategies_defined(self):
        """All expected recovery strategies are defined."""
        assert RecoveryStrategy.TAILSCALE_UP.value == "tailscale_up"
        assert RecoveryStrategy.TAILSCALE_RESTART.value == "tailscale_restart"
        assert RecoveryStrategy.SSH_RECOVERY.value == "ssh_recovery"
        assert RecoveryStrategy.P2P_RECONNECT.value == "p2p_reconnect"
        assert RecoveryStrategy.NODE_RESTART.value == "node_restart"
        assert RecoveryStrategy.ALERT.value == "alert"

    def test_strategy_count(self):
        """Exactly 6 recovery strategies exist."""
        assert len(RecoveryStrategy) == 6

    def test_strategy_iteration(self):
        """Can iterate over all strategies."""
        strategies = list(RecoveryStrategy)
        assert len(strategies) == 6
        assert RecoveryStrategy.SSH_RECOVERY in strategies


# =============================================================================
# RecoveryAttempt Dataclass Tests
# =============================================================================


class TestRecoveryAttempt:
    """Tests for RecoveryAttempt dataclass."""

    def test_required_fields(self):
        """All required fields are set correctly."""
        attempt = RecoveryAttempt(
            node_name="test-node",
            strategy=RecoveryStrategy.SSH_RECOVERY,
            timestamp=1234567890.0,
            success=True,
        )
        assert attempt.node_name == "test-node"
        assert attempt.strategy == RecoveryStrategy.SSH_RECOVERY
        assert attempt.timestamp == 1234567890.0
        assert attempt.success is True

    def test_error_message_optional(self):
        """error_message is optional and defaults to None."""
        attempt = RecoveryAttempt(
            node_name="node1",
            strategy=RecoveryStrategy.ALERT,
            timestamp=time.time(),
            success=False,
        )
        assert attempt.error_message is None

    def test_error_message_provided(self):
        """error_message can be explicitly set."""
        attempt = RecoveryAttempt(
            node_name="node1",
            strategy=RecoveryStrategy.TAILSCALE_UP,
            timestamp=time.time(),
            success=False,
            error_message="Connection refused",
        )
        assert attempt.error_message == "Connection refused"

    def test_failed_attempt(self):
        """Failed attempt has success=False."""
        attempt = RecoveryAttempt(
            node_name="failed-node",
            strategy=RecoveryStrategy.NODE_RESTART,
            timestamp=time.time(),
            success=False,
            error_message="Timeout",
        )
        assert attempt.success is False
        assert "Timeout" in attempt.error_message


# =============================================================================
# NodeConnectivityState Dataclass Tests
# =============================================================================


class TestNodeConnectivityState:
    """Tests for NodeConnectivityState dataclass."""

    def test_defaults(self):
        """Default values are correct."""
        state = NodeConnectivityState(node_name="test-node")
        assert state.node_name == "test-node"
        assert state.tailscale_connected is False
        assert state.p2p_connected is False
        assert state.last_seen == 0.0
        assert state.recovery_attempts == 0
        assert state.last_recovery_time == 0.0
        assert state.consecutive_failures == 0
        assert state.pending_recovery is False

    def test_connected_state(self):
        """Can represent a fully connected node."""
        state = NodeConnectivityState(
            node_name="healthy-node",
            tailscale_connected=True,
            p2p_connected=True,
            last_seen=time.time(),
        )
        assert state.tailscale_connected is True
        assert state.p2p_connected is True
        assert state.last_seen > 0

    def test_recovery_in_progress(self):
        """Can represent a node with pending recovery."""
        state = NodeConnectivityState(
            node_name="recovering-node",
            tailscale_connected=False,
            p2p_connected=False,
            recovery_attempts=2,
            pending_recovery=True,
        )
        assert state.pending_recovery is True
        assert state.recovery_attempts == 2

    def test_consecutive_failures_tracking(self):
        """Tracks consecutive failures."""
        state = NodeConnectivityState(
            node_name="problematic-node",
            consecutive_failures=5,
            recovery_attempts=3,
        )
        assert state.consecutive_failures == 5


# =============================================================================
# RecoveryConfig Dataclass Tests
# =============================================================================


class TestRecoveryConfig:
    """Tests for RecoveryConfig dataclass."""

    def test_defaults(self):
        """Default configuration values."""
        config = RecoveryConfig()
        assert config.recovery_cooldown_seconds == 300.0
        assert config.max_recovery_attempts == 5
        assert config.escalation_threshold == 3
        assert config.ssh_recovery_enabled is True
        assert config.ssh_timeout_seconds == 30.0
        assert config.alert_after_failures == 5
        assert config.slack_webhook is None

    def test_custom_values(self):
        """Can override default values."""
        config = RecoveryConfig(
            recovery_cooldown_seconds=600.0,
            max_recovery_attempts=10,
            ssh_recovery_enabled=False,
            slack_webhook="https://hooks.slack.com/xxx",
        )
        assert config.recovery_cooldown_seconds == 600.0
        assert config.max_recovery_attempts == 10
        assert config.ssh_recovery_enabled is False
        assert config.slack_webhook == "https://hooks.slack.com/xxx"

    def test_from_env_defaults(self):
        """from_env() uses defaults when no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            config = RecoveryConfig.from_env()
            assert config.recovery_cooldown_seconds == 300.0
            assert config.max_recovery_attempts == 5
            assert config.ssh_recovery_enabled is True
            assert config.slack_webhook is None

    def test_from_env_custom(self):
        """from_env() reads environment variables."""
        env_vars = {
            "RINGRIFT_RECOVERY_COOLDOWN": "120",
            "RINGRIFT_MAX_RECOVERY_ATTEMPTS": "3",
            "RINGRIFT_SSH_RECOVERY_ENABLED": "false",
            "RINGRIFT_SLACK_WEBHOOK": "https://hooks.slack.com/test",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = RecoveryConfig.from_env()
            assert config.recovery_cooldown_seconds == 120.0
            assert config.max_recovery_attempts == 3
            assert config.ssh_recovery_enabled is False
            assert config.slack_webhook == "https://hooks.slack.com/test"

    def test_from_env_ssh_enabled_variations(self):
        """from_env() handles different boolean string values."""
        # True variations
        for value in ["true", "True", "TRUE"]:
            with patch.dict(os.environ, {"RINGRIFT_SSH_RECOVERY_ENABLED": value}):
                config = RecoveryConfig.from_env()
                assert config.ssh_recovery_enabled is True

        # False variations
        for value in ["false", "False", "0", "no"]:
            with patch.dict(os.environ, {"RINGRIFT_SSH_RECOVERY_ENABLED": value}):
                config = RecoveryConfig.from_env()
                assert config.ssh_recovery_enabled is False


# =============================================================================
# ConnectivityRecoveryCoordinator Sync Tests
# =============================================================================


class TestConnectivityRecoveryCoordinatorSync:
    """Synchronous tests for ConnectivityRecoveryCoordinator."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        ConnectivityRecoveryCoordinator.reset_instance()
        yield
        ConnectivityRecoveryCoordinator.reset_instance()

    def test_initialization(self):
        """Coordinator initializes with default config."""
        coordinator = create_connectivity_recovery_coordinator()
        assert coordinator._config is not None
        assert coordinator._node_states == {}
        assert coordinator._recovery_history == []
        assert coordinator._pending_recoveries == set()

    def test_initialization_with_custom_config(self):
        """Coordinator uses provided config."""
        config = RecoveryConfig(
            recovery_cooldown_seconds=60.0,
            max_recovery_attempts=3,
        )
        coordinator = create_connectivity_recovery_coordinator(config=config)
        assert coordinator._config.recovery_cooldown_seconds == 60.0
        assert coordinator._config.max_recovery_attempts == 3

    def test_event_subscriptions(self):
        """Returns correct event subscriptions."""
        coordinator = create_connectivity_recovery_coordinator()
        subs = coordinator._get_event_subscriptions()

        expected_events = [
            "TAILSCALE_DISCONNECTED",
            "TAILSCALE_RECOVERED",
            "TAILSCALE_RECOVERY_FAILED",
            "P2P_NODE_DEAD",
            "HOST_OFFLINE",
            "HOST_ONLINE",
        ]
        for event in expected_events:
            assert event in subs
            assert callable(subs[event])

    def test_health_check_empty_state(self):
        """health_check returns healthy with no tracked nodes."""
        coordinator = create_connectivity_recovery_coordinator()
        result = coordinator.health_check()

        assert result.healthy is True
        assert "0 nodes tracked" in result.message
        assert result.details["nodes_tracked"] == 0
        assert result.details["pending_recoveries"] == 0

    def test_health_check_with_nodes(self):
        """health_check reports tracked node count."""
        coordinator = create_connectivity_recovery_coordinator()
        coordinator._get_or_create_state("node-1")
        coordinator._get_or_create_state("node-2")
        coordinator._get_or_create_state("node-3")

        result = coordinator.health_check()
        assert result.details["nodes_tracked"] == 3

    def test_health_check_with_pending_recoveries(self):
        """health_check reports pending recoveries."""
        coordinator = create_connectivity_recovery_coordinator()
        coordinator._pending_recoveries.add("node-1")
        coordinator._pending_recoveries.add("node-2")

        result = coordinator.health_check()
        assert result.details["pending_recoveries"] == 2

    def test_health_check_unhealthy_too_many_pending(self):
        """health_check returns unhealthy with 5+ pending recoveries."""
        coordinator = create_connectivity_recovery_coordinator()
        for i in range(5):
            coordinator._pending_recoveries.add(f"node-{i}")

        result = coordinator.health_check()
        assert result.healthy is False

    def test_health_check_unhealthy_too_many_failures(self):
        """health_check returns unhealthy with 10+ recent failures."""
        coordinator = create_connectivity_recovery_coordinator()
        for i in range(10):
            state = coordinator._get_or_create_state(f"node-{i}")
            state.consecutive_failures = 1

        result = coordinator.health_check()
        assert result.healthy is False

    def test_get_or_create_state_new(self):
        """Creates new state if not exists."""
        coordinator = create_connectivity_recovery_coordinator()
        state = coordinator._get_or_create_state("new-node")

        assert state.node_name == "new-node"
        assert state.tailscale_connected is False
        assert "new-node" in coordinator._node_states

    def test_get_or_create_state_existing(self):
        """Returns existing state if present."""
        coordinator = create_connectivity_recovery_coordinator()
        state1 = coordinator._get_or_create_state("node-x")
        state1.tailscale_connected = True

        state2 = coordinator._get_or_create_state("node-x")
        assert state2 is state1
        assert state2.tailscale_connected is True

    def test_get_node_state_existing(self):
        """get_node_state returns state if exists."""
        coordinator = create_connectivity_recovery_coordinator()
        coordinator._get_or_create_state("test-node")

        result = coordinator.get_node_state("test-node")
        assert result is not None
        assert result.node_name == "test-node"

    def test_get_node_state_missing(self):
        """get_node_state returns None for unknown node."""
        coordinator = create_connectivity_recovery_coordinator()
        result = coordinator.get_node_state("unknown-node")
        assert result is None

    def test_get_all_states(self):
        """get_all_states returns copy of all states."""
        coordinator = create_connectivity_recovery_coordinator()
        coordinator._get_or_create_state("node-a")
        coordinator._get_or_create_state("node-b")

        states = coordinator.get_all_states()
        assert len(states) == 2
        assert "node-a" in states
        assert "node-b" in states

    def test_get_disconnected_nodes(self):
        """get_disconnected_nodes returns nodes not fully connected."""
        coordinator = create_connectivity_recovery_coordinator()

        # Fully connected
        state1 = coordinator._get_or_create_state("healthy")
        state1.tailscale_connected = True
        state1.p2p_connected = True

        # Tailscale down
        state2 = coordinator._get_or_create_state("ts-down")
        state2.tailscale_connected = False
        state2.p2p_connected = True

        # P2P down
        state3 = coordinator._get_or_create_state("p2p-down")
        state3.tailscale_connected = True
        state3.p2p_connected = False

        disconnected = coordinator.get_disconnected_nodes()
        assert "healthy" not in disconnected
        assert "ts-down" in disconnected
        assert "p2p-down" in disconnected

    def test_get_recovery_history(self):
        """get_recovery_history returns recent attempts."""
        coordinator = create_connectivity_recovery_coordinator()

        for i in range(5):
            attempt = RecoveryAttempt(
                node_name=f"node-{i}",
                strategy=RecoveryStrategy.SSH_RECOVERY,
                timestamp=time.time(),
                success=i % 2 == 0,
            )
            coordinator._recovery_history.append(attempt)

        history = coordinator.get_recovery_history(limit=3)
        assert len(history) == 3

    def test_get_recovery_history_limit(self):
        """get_recovery_history respects limit parameter."""
        coordinator = create_connectivity_recovery_coordinator()

        for i in range(150):
            attempt = RecoveryAttempt(
                node_name=f"node-{i}",
                strategy=RecoveryStrategy.ALERT,
                timestamp=time.time(),
                success=False,
            )
            coordinator._recovery_history.append(attempt)

        # Default limit
        history = coordinator.get_recovery_history()
        assert len(history) == 100

        # Custom limit
        history = coordinator.get_recovery_history(limit=50)
        assert len(history) == 50

    def test_build_recovery_command_regular(self):
        """Builds systemctl-based command for regular hosts."""
        coordinator = create_connectivity_recovery_coordinator()
        ssh_info = {"is_container": False}

        cmd = coordinator._build_recovery_command(ssh_info)
        assert "systemctl restart tailscaled" in cmd
        assert "tailscale up --accept-routes" in cmd

    def test_build_recovery_command_container(self):
        """Builds userspace networking command for containers."""
        coordinator = create_connectivity_recovery_coordinator()
        ssh_info = {"is_container": True}

        cmd = coordinator._build_recovery_command(ssh_info)
        assert "--tun=userspace-networking" in cmd
        assert "pkill -9 tailscaled" in cmd


# =============================================================================
# ConnectivityRecoveryCoordinator Async Tests
# =============================================================================


@pytest.mark.asyncio
class TestConnectivityRecoveryCoordinatorAsync:
    """Async tests for ConnectivityRecoveryCoordinator."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        ConnectivityRecoveryCoordinator.reset_instance()
        yield
        ConnectivityRecoveryCoordinator.reset_instance()

    async def test_on_tailscale_disconnected_missing_node_name(self):
        """Ignores event without node_name."""
        coordinator = create_connectivity_recovery_coordinator()
        await coordinator._on_tailscale_disconnected({})
        assert len(coordinator._node_states) == 0

    async def test_on_tailscale_disconnected_creates_state(self):
        """Creates state and marks as disconnected."""
        coordinator = create_connectivity_recovery_coordinator()
        coordinator._config.ssh_recovery_enabled = False  # Disable SSH for test

        await coordinator._on_tailscale_disconnected({"node_name": "test-node"})

        state = coordinator.get_node_state("test-node")
        assert state is not None
        assert state.tailscale_connected is False
        assert state.consecutive_failures == 1

    async def test_on_tailscale_disconnected_local_daemon_handling(self):
        """Skips recovery when local daemon is handling."""
        coordinator = create_connectivity_recovery_coordinator()

        await coordinator._on_tailscale_disconnected({
            "node_name": "local-node",
            "local_daemon_handling": True,
        })

        assert "local-node" not in coordinator._pending_recoveries

    async def test_on_tailscale_disconnected_queues_recovery(self):
        """Queues SSH recovery when enabled."""
        coordinator = create_connectivity_recovery_coordinator()
        coordinator._config.ssh_recovery_enabled = True

        # Mock the SSH recovery to avoid actual SSH
        coordinator._attempt_ssh_recovery = AsyncMock(return_value=False)

        await coordinator._on_tailscale_disconnected({"hostname": "remote-node"})

        assert "remote-node" in coordinator._pending_recoveries

    async def test_on_tailscale_recovered_clears_state(self):
        """Clears failure state on recovery."""
        coordinator = create_connectivity_recovery_coordinator()

        # Set up failed state
        state = coordinator._get_or_create_state("recovering-node")
        state.tailscale_connected = False
        state.consecutive_failures = 3
        coordinator._pending_recoveries.add("recovering-node")

        await coordinator._on_tailscale_recovered({"node_name": "recovering-node"})

        assert state.tailscale_connected is True
        assert state.consecutive_failures == 0
        assert "recovering-node" not in coordinator._pending_recoveries

    async def test_on_tailscale_recovery_failed_increments_attempts(self):
        """Increments recovery attempts on failure."""
        coordinator = create_connectivity_recovery_coordinator()
        coordinator._escalate_recovery = AsyncMock()  # Mock escalation

        state = coordinator._get_or_create_state("failing-node")
        state.recovery_attempts = 0

        await coordinator._on_tailscale_recovery_failed({"node_name": "failing-node"})

        assert state.recovery_attempts == 1
        assert state.last_recovery_time > 0

    async def test_on_tailscale_recovery_failed_escalates(self):
        """Escalates after threshold failures."""
        config = RecoveryConfig(escalation_threshold=2)
        coordinator = create_connectivity_recovery_coordinator(config=config)
        coordinator._escalate_recovery = AsyncMock()

        state = coordinator._get_or_create_state("problem-node")
        state.recovery_attempts = 1

        await coordinator._on_tailscale_recovery_failed({"hostname": "problem-node"})

        # recovery_attempts is now 2 >= escalation_threshold of 2
        coordinator._escalate_recovery.assert_called_once()

    async def test_on_p2p_node_dead(self):
        """Handles P2P node dead event."""
        coordinator = create_connectivity_recovery_coordinator()

        await coordinator._on_p2p_node_dead({"node_id": "dead-peer"})

        state = coordinator.get_node_state("dead-peer")
        assert state is not None
        assert state.p2p_connected is False
        assert state.consecutive_failures == 1

    async def test_on_host_offline(self):
        """Handles host offline event."""
        coordinator = create_connectivity_recovery_coordinator()

        await coordinator._on_host_offline({"hostname": "offline-host"})

        state = coordinator.get_node_state("offline-host")
        assert state.tailscale_connected is False
        assert state.p2p_connected is False
        assert "offline-host" in coordinator._pending_recoveries

    async def test_on_host_online(self):
        """Handles host online event."""
        coordinator = create_connectivity_recovery_coordinator()

        # Set up offline state
        state = coordinator._get_or_create_state("returning-host")
        state.tailscale_connected = False
        state.p2p_connected = False
        state.consecutive_failures = 5
        coordinator._pending_recoveries.add("returning-host")

        await coordinator._on_host_online({"node_id": "returning-host"})

        assert state.tailscale_connected is True
        assert state.p2p_connected is True
        assert state.consecutive_failures == 0
        assert state.last_seen > 0
        assert "returning-host" not in coordinator._pending_recoveries

    async def test_check_pending_recoveries_cooldown(self):
        """Respects recovery cooldown period."""
        config = RecoveryConfig(recovery_cooldown_seconds=300.0)
        coordinator = create_connectivity_recovery_coordinator(config=config)
        coordinator._attempt_ssh_recovery = AsyncMock()

        state = coordinator._get_or_create_state("cooling-node")
        state.last_recovery_time = time.time()  # Just attempted
        coordinator._pending_recoveries.add("cooling-node")

        await coordinator._check_pending_recoveries()

        # Should not attempt recovery during cooldown
        coordinator._attempt_ssh_recovery.assert_not_called()

    async def test_check_pending_recoveries_max_attempts(self):
        """Escalates when max attempts exceeded."""
        config = RecoveryConfig(max_recovery_attempts=3)
        coordinator = create_connectivity_recovery_coordinator(config=config)
        coordinator._escalate_recovery = AsyncMock()

        state = coordinator._get_or_create_state("exhausted-node")
        state.recovery_attempts = 3
        state.last_recovery_time = 0  # Cooldown expired
        coordinator._pending_recoveries.add("exhausted-node")

        await coordinator._check_pending_recoveries()

        coordinator._escalate_recovery.assert_called_once()
        assert "exhausted-node" not in coordinator._pending_recoveries

    async def test_attempt_ssh_recovery_no_ssh_info(self):
        """Returns False when no SSH info available."""
        coordinator = create_connectivity_recovery_coordinator()
        coordinator._get_ssh_info = AsyncMock(return_value=None)

        result = await coordinator._attempt_ssh_recovery("unknown-node")

        assert result is False

    async def test_attempt_ssh_recovery_success(self):
        """Records successful recovery attempt."""
        coordinator = create_connectivity_recovery_coordinator()
        coordinator._get_ssh_info = AsyncMock(return_value={
            "host": "100.1.2.3",
            "port": 22,
            "user": "root",
            "key": None,
            "is_container": False,
        })
        coordinator._execute_ssh_command = AsyncMock(return_value=True)
        coordinator._emit_event = AsyncMock()

        result = await coordinator._attempt_ssh_recovery("success-node")

        assert result is True
        state = coordinator.get_node_state("success-node")
        assert state.consecutive_failures == 0
        assert "success-node" not in coordinator._pending_recoveries

        # Check recovery was recorded
        assert len(coordinator._recovery_history) == 1
        assert coordinator._recovery_history[0].success is True

    async def test_attempt_ssh_recovery_failure(self):
        """Records failed recovery attempt."""
        coordinator = create_connectivity_recovery_coordinator()
        coordinator._get_ssh_info = AsyncMock(return_value={
            "host": "192.168.1.100",
            "port": 22,
            "user": "ubuntu",
            "key": "~/.ssh/id_rsa",
            "is_container": True,
        })
        coordinator._execute_ssh_command = AsyncMock(return_value=False)

        result = await coordinator._attempt_ssh_recovery("fail-node")

        assert result is False
        assert len(coordinator._recovery_history) == 1
        assert coordinator._recovery_history[0].success is False

    async def test_execute_ssh_command_success(self):
        """Returns True when SSH command succeeds with Tailscale IP."""
        coordinator = create_connectivity_recovery_coordinator()

        # Mock subprocess
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"100.123.456.789\n", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await coordinator._execute_ssh_command(
                host="192.168.1.1",
                port=22,
                user="root",
                key_path=None,
                command="tailscale ip -4",
            )

        assert result is True

    async def test_execute_ssh_command_no_tailscale_ip(self):
        """Returns False when output lacks Tailscale IP."""
        coordinator = create_connectivity_recovery_coordinator()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"192.168.1.1\n", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await coordinator._execute_ssh_command(
                host="192.168.1.1",
                port=22,
                user="root",
                key_path=None,
                command="tailscale ip -4",
            )

        assert result is False

    async def test_execute_ssh_command_timeout(self):
        """Returns False on SSH timeout."""
        coordinator = create_connectivity_recovery_coordinator()

        async def slow_communicate():
            await asyncio.sleep(100)
            return b"", b""

        mock_proc = MagicMock()
        mock_proc.communicate = slow_communicate

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await coordinator._execute_ssh_command(
                host="slow-host",
                port=22,
                user="root",
                key_path=None,
                command="sleep 100",
            )

        assert result is False

    async def test_escalate_recovery_emits_event(self):
        """Escalation emits event."""
        coordinator = create_connectivity_recovery_coordinator()
        coordinator._emit_event = AsyncMock()
        coordinator._send_slack_alert = AsyncMock()

        state = NodeConnectivityState(
            node_name="escalated-node",
            recovery_attempts=5,
            consecutive_failures=10,
        )

        await coordinator._escalate_recovery("escalated-node", state)

        coordinator._emit_event.assert_called_once()
        call_args = coordinator._emit_event.call_args
        assert call_args[0][0] == "CONNECTIVITY_RECOVERY_ESCALATED"
        assert call_args[0][1]["node_name"] == "escalated-node"

    async def test_escalate_recovery_sends_slack_alert(self):
        """Escalation sends Slack alert if configured."""
        config = RecoveryConfig(slack_webhook="https://hooks.slack.com/test")
        coordinator = create_connectivity_recovery_coordinator(config=config)
        coordinator._emit_event = AsyncMock()
        coordinator._send_slack_alert = AsyncMock()

        state = NodeConnectivityState(node_name="alert-node")

        await coordinator._escalate_recovery("alert-node", state)

        coordinator._send_slack_alert.assert_called_once_with("alert-node", state)

    async def test_run_cycle_updates_stats(self):
        """Run cycle updates completion stats."""
        coordinator = create_connectivity_recovery_coordinator()
        coordinator._check_pending_recoveries = AsyncMock()

        initial_cycles = coordinator._stats.cycles_completed

        await coordinator._run_cycle()

        assert coordinator._stats.cycles_completed == initial_cycles + 1
        assert coordinator._stats.last_activity > 0

    async def test_run_cycle_error_handling(self):
        """Run cycle handles errors gracefully."""
        coordinator = create_connectivity_recovery_coordinator()
        coordinator._check_pending_recoveries = AsyncMock(
            side_effect=Exception("Test error")
        )

        await coordinator._run_cycle()

        assert coordinator._stats.errors_count == 1
        assert "Test error" in coordinator._stats.last_error

    async def test_get_ssh_info_found(self):
        """Returns SSH info from cluster config."""
        coordinator = create_connectivity_recovery_coordinator()

        mock_node = MagicMock()
        mock_node.name = "test-node"
        mock_node.ssh_host = "192.168.1.100"
        mock_node.ssh_port = 22
        mock_node.ssh_user = "ubuntu"
        mock_node.ssh_key = "~/.ssh/id_cluster"
        mock_node.tailscale_ip = "100.1.2.3"
        mock_node.is_container = False

        with patch(
            "app.config.cluster_config.get_cluster_nodes",
            return_value=[mock_node],
        ):
            info = await coordinator._get_ssh_info("test-node")

        assert info is not None
        assert info["host"] == "192.168.1.100"
        assert info["port"] == 22
        assert info["user"] == "ubuntu"

    async def test_get_ssh_info_not_found(self):
        """Returns None when node not in config."""
        coordinator = create_connectivity_recovery_coordinator()

        with patch(
            "app.config.cluster_config.get_cluster_nodes",
            return_value=[],
        ):
            info = await coordinator._get_ssh_info("unknown-node")

        assert info is None

    async def test_emit_event_graceful_failure(self):
        """Event emission fails gracefully."""
        coordinator = create_connectivity_recovery_coordinator()

        # Should not raise
        await coordinator._emit_event("TEST_EVENT", {"data": "test"})


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        ConnectivityRecoveryCoordinator.reset_instance()
        yield
        ConnectivityRecoveryCoordinator.reset_instance()

    def test_get_connectivity_recovery_coordinator(self):
        """get_connectivity_recovery_coordinator returns singleton."""
        coord1 = get_connectivity_recovery_coordinator()
        coord2 = get_connectivity_recovery_coordinator()
        assert coord1 is coord2

    def test_create_connectivity_recovery_coordinator(self):
        """create_connectivity_recovery_coordinator returns new instance."""
        coord1 = create_connectivity_recovery_coordinator()
        coord2 = create_connectivity_recovery_coordinator()
        # Note: These are NOT the same because create_ doesn't use singleton
        # The function actually creates new instances for testing
        assert coord1 is not coord2

    def test_create_with_config(self):
        """create_connectivity_recovery_coordinator accepts config."""
        config = RecoveryConfig(max_recovery_attempts=10)
        coordinator = create_connectivity_recovery_coordinator(config=config)
        assert coordinator._config.max_recovery_attempts == 10


# =============================================================================
# Integration-like Tests
# =============================================================================


@pytest.mark.asyncio
class TestConnectivityRecoveryIntegration:
    """Integration-like tests for end-to-end flows."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        ConnectivityRecoveryCoordinator.reset_instance()
        yield
        ConnectivityRecoveryCoordinator.reset_instance()

    async def test_disconnect_recover_flow(self):
        """Full disconnect -> recovery flow."""
        coordinator = create_connectivity_recovery_coordinator()
        coordinator._config.ssh_recovery_enabled = False

        # Step 1: Node disconnects
        await coordinator._on_tailscale_disconnected({"node_name": "flow-node"})

        state = coordinator.get_node_state("flow-node")
        assert state.tailscale_connected is False
        assert state.consecutive_failures == 1

        # Step 2: Node recovers
        await coordinator._on_tailscale_recovered({"node_name": "flow-node"})

        assert state.tailscale_connected is True
        assert state.consecutive_failures == 0

    async def test_multiple_failures_escalation(self):
        """Multiple failures lead to escalation."""
        config = RecoveryConfig(escalation_threshold=2)
        coordinator = create_connectivity_recovery_coordinator(config=config)
        coordinator._escalate_recovery = AsyncMock()

        # First failure
        await coordinator._on_tailscale_recovery_failed({"node_name": "multi-fail"})
        coordinator._escalate_recovery.assert_not_called()

        # Second failure (reaches threshold)
        await coordinator._on_tailscale_recovery_failed({"node_name": "multi-fail"})
        coordinator._escalate_recovery.assert_called_once()

    async def test_p2p_and_tailscale_combined(self):
        """Node with both P2P and Tailscale issues."""
        coordinator = create_connectivity_recovery_coordinator()

        # P2P goes down
        await coordinator._on_p2p_node_dead({"node_id": "combo-node"})
        state = coordinator.get_node_state("combo-node")
        assert state.p2p_connected is False
        assert state.consecutive_failures == 1

        # Tailscale goes down
        coordinator._config.ssh_recovery_enabled = False
        await coordinator._on_tailscale_disconnected({"node_name": "combo-node"})
        assert state.tailscale_connected is False
        assert state.consecutive_failures == 2

        # Both come back via HOST_ONLINE
        await coordinator._on_host_online({"hostname": "combo-node"})
        assert state.tailscale_connected is True
        assert state.p2p_connected is True
        assert state.consecutive_failures == 0
