"""Tests for ConnectivityRecoveryCoordinator.

December 29, 2025: Comprehensive tests for unified connectivity recovery.
"""

from __future__ import annotations

import asyncio
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
# RecoveryStrategy Tests
# =============================================================================


class TestRecoveryStrategy:
    """Tests for RecoveryStrategy enum."""

    def test_all_strategies_exist(self):
        """All expected strategies are defined."""
        assert RecoveryStrategy.TAILSCALE_UP.value == "tailscale_up"
        assert RecoveryStrategy.TAILSCALE_RESTART.value == "tailscale_restart"
        assert RecoveryStrategy.SSH_RECOVERY.value == "ssh_recovery"
        assert RecoveryStrategy.P2P_RECONNECT.value == "p2p_reconnect"
        assert RecoveryStrategy.NODE_RESTART.value == "node_restart"
        assert RecoveryStrategy.ALERT.value == "alert"

    def test_strategies_are_unique(self):
        """All strategy values are unique."""
        values = [s.value for s in RecoveryStrategy]
        assert len(values) == len(set(values))


# =============================================================================
# RecoveryAttempt Tests
# =============================================================================


class TestRecoveryAttempt:
    """Tests for RecoveryAttempt dataclass."""

    def test_basic_creation(self):
        """Test basic RecoveryAttempt creation."""
        attempt = RecoveryAttempt(
            node_name="test-node",
            strategy=RecoveryStrategy.SSH_RECOVERY,
            timestamp=1000.0,
            success=True,
        )
        assert attempt.node_name == "test-node"
        assert attempt.strategy == RecoveryStrategy.SSH_RECOVERY
        assert attempt.timestamp == 1000.0
        assert attempt.success is True
        assert attempt.error_message is None

    def test_with_error_message(self):
        """Test RecoveryAttempt with error."""
        attempt = RecoveryAttempt(
            node_name="test-node",
            strategy=RecoveryStrategy.SSH_RECOVERY,
            timestamp=1000.0,
            success=False,
            error_message="Connection refused",
        )
        assert attempt.success is False
        assert attempt.error_message == "Connection refused"


# =============================================================================
# NodeConnectivityState Tests
# =============================================================================


class TestNodeConnectivityState:
    """Tests for NodeConnectivityState dataclass."""

    def test_default_values(self):
        """Test default state values."""
        state = NodeConnectivityState(node_name="test-node")
        assert state.node_name == "test-node"
        assert state.tailscale_connected is False
        assert state.p2p_connected is False
        assert state.last_seen == 0.0
        assert state.recovery_attempts == 0
        assert state.consecutive_failures == 0
        assert state.pending_recovery is False

    def test_connectivity_flags(self):
        """Test connectivity flag updates."""
        state = NodeConnectivityState(
            node_name="test-node",
            tailscale_connected=True,
            p2p_connected=True,
        )
        assert state.tailscale_connected is True
        assert state.p2p_connected is True


# =============================================================================
# RecoveryConfig Tests
# =============================================================================


class TestRecoveryConfig:
    """Tests for RecoveryConfig dataclass."""

    def test_default_values(self):
        """Test default config values."""
        config = RecoveryConfig()
        assert config.recovery_cooldown_seconds == 300.0
        assert config.max_recovery_attempts == 5
        assert config.escalation_threshold == 3
        assert config.ssh_recovery_enabled is True
        assert config.ssh_timeout_seconds == 30.0
        assert config.alert_after_failures == 5
        assert config.slack_webhook is None

    def test_from_env_defaults(self):
        """Test from_env with defaults."""
        with patch.dict("os.environ", {}, clear=True):
            config = RecoveryConfig.from_env()
            assert config.recovery_cooldown_seconds == 300.0
            assert config.max_recovery_attempts == 5

    def test_from_env_overrides(self):
        """Test from_env with environment overrides."""
        env = {
            "RINGRIFT_RECOVERY_COOLDOWN": "60",
            "RINGRIFT_MAX_RECOVERY_ATTEMPTS": "10",
            "RINGRIFT_SSH_RECOVERY_ENABLED": "false",
            "RINGRIFT_SLACK_WEBHOOK": "https://hooks.slack.com/test",
        }
        with patch.dict("os.environ", env, clear=True):
            config = RecoveryConfig.from_env()
            assert config.recovery_cooldown_seconds == 60.0
            assert config.max_recovery_attempts == 10
            assert config.ssh_recovery_enabled is False
            assert config.slack_webhook == "https://hooks.slack.com/test"


# =============================================================================
# ConnectivityRecoveryCoordinator Init Tests
# =============================================================================


class TestCoordinatorInit:
    """Tests for coordinator initialization."""

    def test_default_init(self):
        """Test default initialization."""
        coord = create_connectivity_recovery_coordinator()
        assert coord._config is not None
        assert coord.name == "connectivity_recovery"
        assert coord._node_states == {}
        assert coord._recovery_history == []
        assert coord._pending_recoveries == set()

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = RecoveryConfig(
            max_recovery_attempts=10,
            ssh_recovery_enabled=False,
        )
        coord = create_connectivity_recovery_coordinator(config=config)
        assert coord._config.max_recovery_attempts == 10
        assert coord._config.ssh_recovery_enabled is False

    def test_event_subscriptions(self):
        """Test event subscriptions are set up."""
        coord = create_connectivity_recovery_coordinator()
        subs = coord._get_event_subscriptions()

        assert "TAILSCALE_DISCONNECTED" in subs
        assert "TAILSCALE_RECOVERED" in subs
        assert "TAILSCALE_RECOVERY_FAILED" in subs
        assert "P2P_NODE_DEAD" in subs
        assert "HOST_OFFLINE" in subs
        assert "HOST_ONLINE" in subs


# =============================================================================
# Event Handler Tests
# =============================================================================


class TestEventHandlers:
    """Tests for event handlers."""

    @pytest.mark.asyncio
    async def test_on_tailscale_disconnected(self):
        """Test handling Tailscale disconnection."""
        coord = create_connectivity_recovery_coordinator(
            config=RecoveryConfig(ssh_recovery_enabled=False)
        )

        event = {"node_name": "test-node"}
        await coord._on_tailscale_disconnected(event)

        state = coord.get_node_state("test-node")
        assert state is not None
        assert state.tailscale_connected is False
        assert state.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_on_tailscale_disconnected_local_handling(self):
        """Test local daemon handling bypasses recovery."""
        coord = create_connectivity_recovery_coordinator()

        event = {"node_name": "test-node", "local_daemon_handling": True}
        await coord._on_tailscale_disconnected(event)

        # Should not queue for pending recovery
        assert "test-node" not in coord._pending_recoveries

    @pytest.mark.asyncio
    async def test_on_tailscale_recovered(self):
        """Test handling Tailscale recovery."""
        coord = create_connectivity_recovery_coordinator()

        # Set up disconnected state
        coord._node_states["test-node"] = NodeConnectivityState(
            node_name="test-node",
            tailscale_connected=False,
            consecutive_failures=3,
        )
        coord._pending_recoveries.add("test-node")

        event = {"node_name": "test-node"}
        await coord._on_tailscale_recovered(event)

        state = coord.get_node_state("test-node")
        assert state.tailscale_connected is True
        assert state.consecutive_failures == 0
        assert "test-node" not in coord._pending_recoveries

    @pytest.mark.asyncio
    async def test_on_p2p_node_dead(self):
        """Test handling P2P node dead event."""
        coord = create_connectivity_recovery_coordinator()

        event = {"node_id": "peer-123"}
        await coord._on_p2p_node_dead(event)

        state = coord.get_node_state("peer-123")
        assert state is not None
        assert state.p2p_connected is False
        assert state.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_on_host_offline(self):
        """Test handling host offline event."""
        coord = create_connectivity_recovery_coordinator()

        event = {"hostname": "worker-1"}
        await coord._on_host_offline(event)

        state = coord.get_node_state("worker-1")
        assert state.tailscale_connected is False
        assert state.p2p_connected is False
        assert "worker-1" in coord._pending_recoveries

    @pytest.mark.asyncio
    async def test_on_host_online(self):
        """Test handling host online event."""
        coord = create_connectivity_recovery_coordinator()

        # Set up offline state
        coord._node_states["worker-1"] = NodeConnectivityState(
            node_name="worker-1",
            tailscale_connected=False,
            p2p_connected=False,
            consecutive_failures=5,
        )
        coord._pending_recoveries.add("worker-1")

        event = {"hostname": "worker-1"}
        await coord._on_host_online(event)

        state = coord.get_node_state("worker-1")
        assert state.tailscale_connected is True
        assert state.p2p_connected is True
        assert state.consecutive_failures == 0
        assert "worker-1" not in coord._pending_recoveries

    @pytest.mark.asyncio
    async def test_missing_node_name(self):
        """Test handling events with missing node name."""
        coord = create_connectivity_recovery_coordinator()

        # Should not raise, just return early
        await coord._on_tailscale_disconnected({})
        await coord._on_tailscale_recovered({})
        await coord._on_p2p_node_dead({})

        assert len(coord._node_states) == 0


# =============================================================================
# Recovery Logic Tests
# =============================================================================


class TestRecoveryLogic:
    """Tests for recovery logic."""

    def test_build_recovery_command_container(self):
        """Test recovery command for containers."""
        coord = create_connectivity_recovery_coordinator()

        cmd = coord._build_recovery_command({"is_container": True})

        assert "userspace-networking" in cmd
        assert "pkill -9 tailscaled" in cmd
        assert "tailscale up" in cmd

    def test_build_recovery_command_host(self):
        """Test recovery command for regular hosts."""
        coord = create_connectivity_recovery_coordinator()

        cmd = coord._build_recovery_command({"is_container": False})

        assert "systemctl restart tailscaled" in cmd
        assert "tailscale up" in cmd

    @pytest.mark.asyncio
    async def test_execute_ssh_command_success(self):
        """Test successful SSH command execution."""
        coord = create_connectivity_recovery_coordinator()

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"100.64.0.1\n", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            success = await coord._execute_ssh_command(
                host="192.168.1.1",
                port=22,
                user="root",
                key_path=None,
                command="tailscale ip",
            )

        assert success is True

    @pytest.mark.asyncio
    async def test_execute_ssh_command_failure(self):
        """Test failed SSH command execution."""
        coord = create_connectivity_recovery_coordinator()

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"Error"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            success = await coord._execute_ssh_command(
                host="192.168.1.1",
                port=22,
                user="root",
                key_path=None,
                command="tailscale ip",
            )

        assert success is False

    @pytest.mark.asyncio
    async def test_execute_ssh_command_timeout(self):
        """Test SSH command timeout."""
        coord = create_connectivity_recovery_coordinator(
            config=RecoveryConfig(ssh_timeout_seconds=0.1)
        )

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            success = await coord._execute_ssh_command(
                host="192.168.1.1",
                port=22,
                user="root",
                key_path=None,
                command="sleep 10",
            )

        assert success is False

    @pytest.mark.asyncio
    async def test_attempt_ssh_recovery_success(self):
        """Test successful SSH recovery attempt."""
        coord = create_connectivity_recovery_coordinator()

        # Mock SSH info and command execution
        with patch.object(coord, "_get_ssh_info", return_value={
            "host": "192.168.1.1",
            "port": 22,
            "user": "root",
        }):
            with patch.object(coord, "_execute_ssh_command", return_value=True):
                with patch.object(coord, "_emit_event", new_callable=AsyncMock):
                    success = await coord._attempt_ssh_recovery("test-node")

        assert success is True
        state = coord.get_node_state("test-node")
        assert state.consecutive_failures == 0
        assert len(coord._recovery_history) == 1
        assert coord._recovery_history[0].success is True

    @pytest.mark.asyncio
    async def test_attempt_ssh_recovery_no_ssh_info(self):
        """Test recovery when SSH info not available."""
        coord = create_connectivity_recovery_coordinator()

        with patch.object(coord, "_get_ssh_info", return_value=None):
            success = await coord._attempt_ssh_recovery("test-node")

        assert success is False


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health check functionality."""

    def test_health_check_healthy(self):
        """Test healthy state."""
        coord = create_connectivity_recovery_coordinator()

        result = coord.health_check()

        assert result.healthy is True
        assert "0 nodes tracked" in result.message
        assert result.details["pending_recoveries"] == 0

    def test_health_check_with_pending(self):
        """Test health with pending recoveries."""
        coord = create_connectivity_recovery_coordinator()
        coord._pending_recoveries = {"node1", "node2", "node3"}

        result = coord.health_check()

        assert result.healthy is True  # Still healthy with < 5 pending
        assert result.details["pending_recoveries"] == 3

    def test_health_check_degraded(self):
        """Test degraded health with many pending recoveries."""
        coord = create_connectivity_recovery_coordinator()
        coord._pending_recoveries = {f"node{i}" for i in range(10)}

        result = coord.health_check()

        assert result.healthy is False
        assert result.details["pending_recoveries"] == 10


# =============================================================================
# State Management Tests
# =============================================================================


class TestStateManagement:
    """Tests for state management."""

    def test_get_or_create_state(self):
        """Test get_or_create_state creates new state."""
        coord = create_connectivity_recovery_coordinator()

        state = coord._get_or_create_state("new-node")

        assert state.node_name == "new-node"
        assert "new-node" in coord._node_states

    def test_get_or_create_state_returns_existing(self):
        """Test get_or_create_state returns existing state."""
        coord = create_connectivity_recovery_coordinator()

        state1 = coord._get_or_create_state("test-node")
        state1.consecutive_failures = 5

        state2 = coord._get_or_create_state("test-node")

        assert state2.consecutive_failures == 5
        assert state1 is state2

    def test_get_all_states(self):
        """Test get_all_states returns copy."""
        coord = create_connectivity_recovery_coordinator()

        coord._get_or_create_state("node1")
        coord._get_or_create_state("node2")

        states = coord.get_all_states()

        assert len(states) == 2
        assert "node1" in states
        assert "node2" in states

    def test_get_disconnected_nodes(self):
        """Test get_disconnected_nodes."""
        coord = create_connectivity_recovery_coordinator()

        coord._node_states["connected"] = NodeConnectivityState(
            node_name="connected",
            tailscale_connected=True,
            p2p_connected=True,
        )
        coord._node_states["disconnected"] = NodeConnectivityState(
            node_name="disconnected",
            tailscale_connected=False,
            p2p_connected=True,
        )
        coord._node_states["offline"] = NodeConnectivityState(
            node_name="offline",
            tailscale_connected=False,
            p2p_connected=False,
        )

        disconnected = coord.get_disconnected_nodes()

        assert "connected" not in disconnected
        assert "disconnected" in disconnected
        assert "offline" in disconnected

    def test_get_recovery_history(self):
        """Test get_recovery_history with limit."""
        coord = create_connectivity_recovery_coordinator()

        for i in range(150):
            coord._recovery_history.append(
                RecoveryAttempt(
                    node_name=f"node-{i}",
                    strategy=RecoveryStrategy.SSH_RECOVERY,
                    timestamp=float(i),
                    success=True,
                )
            )

        history = coord.get_recovery_history(limit=50)

        assert len(history) == 50
        assert history[0].node_name == "node-100"  # Last 50


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_connectivity_recovery_coordinator(self):
        """Test create function."""
        coord = create_connectivity_recovery_coordinator()
        assert isinstance(coord, ConnectivityRecoveryCoordinator)

    def test_create_with_config(self):
        """Test create with custom config."""
        config = RecoveryConfig(max_recovery_attempts=20)
        coord = create_connectivity_recovery_coordinator(config=config)
        assert coord._config.max_recovery_attempts == 20


# =============================================================================
# Run Cycle Tests
# =============================================================================


class TestRunCycle:
    """Tests for run cycle."""

    @pytest.mark.asyncio
    async def test_run_cycle_success(self):
        """Test successful run cycle."""
        coord = create_connectivity_recovery_coordinator()

        with patch.object(coord, "_check_pending_recoveries", new_callable=AsyncMock):
            await coord._run_cycle()

        assert coord._stats.cycles_completed == 1
        assert coord._stats.last_activity > 0

    @pytest.mark.asyncio
    async def test_run_cycle_error(self):
        """Test run cycle with error."""
        coord = create_connectivity_recovery_coordinator()

        with patch.object(
            coord,
            "_check_pending_recoveries",
            side_effect=Exception("Test error"),
        ):
            await coord._run_cycle()

        assert coord._stats.errors_count == 1
        assert "Test error" in coord._stats.last_error
