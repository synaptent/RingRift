"""Tests for Network Loops - P2P network management.

Tests cover:
- Config dataclass validation for all loop types
- Basic loop lifecycle (start/stop)
- Statistics tracking
- Core loop logic with mocked callbacks
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

from scripts.p2p.loops.network_loops import (
    IpDiscoveryConfig,
    IpDiscoveryLoop,
    TailscaleRecoveryConfig,
    TailscaleRecoveryLoop,
    NATManagementConfig,
    NATManagementLoop,
    TailscalePeerDiscoveryConfig,
    TailscalePeerDiscoveryLoop,
    HeartbeatConfig,
    HeartbeatLoop,
    VoterHeartbeatConfig,
    VoterHeartbeatLoop,
)


# =============================================================================
# IpDiscoveryConfig Tests
# =============================================================================


class TestIpDiscoveryConfig:
    """Tests for IpDiscoveryConfig dataclass."""

    def test_default_values(self):
        """Test IpDiscoveryConfig has correct defaults."""
        config = IpDiscoveryConfig()

        assert config.check_interval_seconds == 300.0
        assert config.dns_timeout_seconds == 10.0
        assert config.max_nodes_per_cycle == 20

    def test_validation_check_interval_zero(self):
        """Test validation rejects check_interval_seconds <= 0."""
        with pytest.raises(ValueError, match="check_interval_seconds must be > 0"):
            IpDiscoveryConfig(check_interval_seconds=0)

    def test_validation_dns_timeout_zero(self):
        """Test validation rejects dns_timeout_seconds <= 0."""
        with pytest.raises(ValueError, match="dns_timeout_seconds must be > 0"):
            IpDiscoveryConfig(dns_timeout_seconds=0)

    def test_validation_max_nodes_zero(self):
        """Test validation rejects max_nodes_per_cycle <= 0."""
        with pytest.raises(ValueError, match="max_nodes_per_cycle must be > 0"):
            IpDiscoveryConfig(max_nodes_per_cycle=0)


# =============================================================================
# IpDiscoveryLoop Tests
# =============================================================================


class TestIpDiscoveryLoop:
    """Tests for IpDiscoveryLoop."""

    def test_init(self):
        """Test loop initialization."""
        loop = IpDiscoveryLoop(
            get_nodes=lambda: {},
            update_node_ip=AsyncMock(),
        )

        assert loop.name == "ip_discovery"
        assert loop.interval == 300.0
        assert loop._updates_count == 0

    def test_init_custom_config(self):
        """Test loop initialization with custom config."""
        config = IpDiscoveryConfig(check_interval_seconds=60.0)
        loop = IpDiscoveryLoop(
            get_nodes=lambda: {},
            update_node_ip=AsyncMock(),
            config=config,
        )

        assert loop.interval == 60.0

    @pytest.mark.asyncio
    async def test_run_once_empty_nodes(self):
        """Test _run_once with no nodes."""
        loop = IpDiscoveryLoop(
            get_nodes=lambda: {},
            update_node_ip=AsyncMock(),
        )

        # Should not raise
        await loop._run_once()

    @pytest.mark.asyncio
    async def test_run_once_with_nodes(self):
        """Test _run_once processes nodes."""
        update_ip = AsyncMock()
        nodes = {
            "node1": {"ip": "192.168.1.1", "hostname": "node1.local"},
            "node2": {"ip": "192.168.1.2"},
        }
        loop = IpDiscoveryLoop(
            get_nodes=lambda: nodes,
            update_node_ip=update_ip,
        )

        # Mock reachability to return True
        with patch.object(loop, "_is_reachable", return_value=True):
            with patch.object(loop, "_resolve_hostname", return_value=None):
                await loop._run_once()

        # No updates since all IPs reachable
        assert update_ip.call_count == 0

    @pytest.mark.asyncio
    async def test_run_once_switches_to_tailscale(self):
        """Test _run_once switches to Tailscale IP when primary unreachable."""
        update_ip = AsyncMock()
        nodes = {
            "node1": {"ip": "192.168.1.1", "tailscale_ip": "100.64.0.1"},
        }
        loop = IpDiscoveryLoop(
            get_nodes=lambda: nodes,
            update_node_ip=update_ip,
        )

        async def mock_reachable(ip, *args, **kwargs):
            return ip == "100.64.0.1"

        with patch.object(loop, "_is_reachable", mock_reachable):
            with patch.object(loop, "_resolve_hostname", return_value=None):
                await loop._run_once()

        update_ip.assert_called_once_with("node1", "100.64.0.1")
        assert loop._updates_count == 1

    def test_get_discovery_stats(self):
        """Test get_discovery_stats returns correct data."""
        loop = IpDiscoveryLoop(
            get_nodes=lambda: {},
            update_node_ip=AsyncMock(),
        )
        loop._updates_count = 5

        stats = loop.get_discovery_stats()

        assert stats["total_updates"] == 5
        assert "name" in stats


# =============================================================================
# TailscaleRecoveryConfig Tests
# =============================================================================


class TestTailscaleRecoveryConfig:
    """Tests for TailscaleRecoveryConfig dataclass."""

    def test_default_values(self):
        """Test TailscaleRecoveryConfig has correct defaults."""
        config = TailscaleRecoveryConfig()

        assert config.check_interval_seconds == 120.0
        assert config.recovery_timeout_seconds == 60.0
        assert config.max_recovery_attempts == 3
        assert config.cooldown_after_recovery_seconds == 300.0

    def test_validation_check_interval_zero(self):
        """Test validation rejects check_interval_seconds <= 0."""
        with pytest.raises(ValueError, match="check_interval_seconds must be > 0"):
            TailscaleRecoveryConfig(check_interval_seconds=0)

    def test_validation_recovery_timeout_zero(self):
        """Test validation rejects recovery_timeout_seconds <= 0."""
        with pytest.raises(ValueError, match="recovery_timeout_seconds must be > 0"):
            TailscaleRecoveryConfig(recovery_timeout_seconds=0)

    def test_validation_max_recovery_attempts_zero(self):
        """Test validation rejects max_recovery_attempts <= 0."""
        with pytest.raises(ValueError, match="max_recovery_attempts must be > 0"):
            TailscaleRecoveryConfig(max_recovery_attempts=0)

    def test_validation_cooldown_negative(self):
        """Test validation rejects negative cooldown."""
        with pytest.raises(ValueError, match="cooldown_after_recovery_seconds must be >= 0"):
            TailscaleRecoveryConfig(cooldown_after_recovery_seconds=-1)


# =============================================================================
# TailscaleRecoveryLoop Tests
# =============================================================================


class TestTailscaleRecoveryLoop:
    """Tests for TailscaleRecoveryLoop."""

    def test_init(self):
        """Test loop initialization."""
        loop = TailscaleRecoveryLoop(
            get_tailscale_status=lambda: {},
            run_ssh_command=AsyncMock(),
        )

        assert loop.name == "tailscale_recovery"
        assert loop.interval == 120.0
        assert loop._recoveries_count == 0

    @pytest.mark.asyncio
    async def test_run_once_empty_status(self):
        """Test _run_once with no status."""
        loop = TailscaleRecoveryLoop(
            get_tailscale_status=lambda: {},
            run_ssh_command=AsyncMock(),
        )

        await loop._run_once()

    @pytest.mark.asyncio
    async def test_run_once_healthy_nodes(self):
        """Test _run_once with all healthy nodes."""
        status = {
            "node1": {"tailscale_state": "running", "tailscale_online": True},
        }
        run_ssh = AsyncMock()
        loop = TailscaleRecoveryLoop(
            get_tailscale_status=lambda: status,
            run_ssh_command=run_ssh,
        )

        await loop._run_once()

        # No recovery attempts for healthy nodes
        run_ssh.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_once_triggers_recovery(self):
        """Test _run_once triggers recovery for offline nodes."""
        status = {
            "node1": {"tailscale_state": "stopped", "tailscale_online": False},
        }

        # Mock SSH result
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"Self": {"Online": true}}'
        run_ssh = AsyncMock(return_value=mock_result)

        loop = TailscaleRecoveryLoop(
            get_tailscale_status=lambda: status,
            run_ssh_command=run_ssh,
        )

        await loop._run_once()

        assert run_ssh.called
        assert loop._recoveries_count == 1

    def test_get_recovery_stats(self):
        """Test get_recovery_stats returns correct data."""
        loop = TailscaleRecoveryLoop(
            get_tailscale_status=lambda: {},
            run_ssh_command=AsyncMock(),
        )
        loop._recoveries_count = 3
        loop._recovery_attempts = {"node1": 2, "node2": 3}

        stats = loop.get_recovery_stats()

        assert stats["total_recoveries"] == 3
        assert stats["nodes_at_max_attempts"] == 1  # node2 at max (3)
        assert "node1" in stats["pending_recoveries"]


# =============================================================================
# NATManagementConfig Tests
# =============================================================================


class TestNATManagementConfig:
    """Tests for NATManagementConfig dataclass."""

    def test_default_values(self):
        """Test NATManagementConfig has correct defaults."""
        config = NATManagementConfig()

        assert config.check_interval_seconds == 60.0
        assert config.stun_probe_interval_seconds == 300.0
        assert config.symmetric_detection_enabled is True

    def test_validation_check_interval_zero(self):
        """Test validation rejects check_interval_seconds <= 0."""
        with pytest.raises(ValueError, match="check_interval_seconds must be > 0"):
            NATManagementConfig(check_interval_seconds=0)

    def test_validation_stun_probe_interval_zero(self):
        """Test validation rejects stun_probe_interval_seconds <= 0."""
        with pytest.raises(ValueError, match="stun_probe_interval_seconds must be > 0"):
            NATManagementConfig(stun_probe_interval_seconds=0)


# =============================================================================
# NATManagementLoop Tests
# =============================================================================


class TestNATManagementLoop:
    """Tests for NATManagementLoop."""

    def test_init(self):
        """Test loop initialization."""
        loop = NATManagementLoop(
            detect_nat_type=AsyncMock(),
            probe_nat_blocked_peers=AsyncMock(),
            update_relay_preferences=AsyncMock(),
        )

        assert loop.name == "nat_management"
        assert loop.interval == 60.0
        assert loop._stun_probes_count == 0

    @pytest.mark.asyncio
    async def test_run_once_calls_callbacks(self):
        """Test _run_once calls all NAT management callbacks."""
        detect_nat = AsyncMock()
        probe_blocked = AsyncMock()
        update_relay = AsyncMock()

        loop = NATManagementLoop(
            detect_nat_type=detect_nat,
            probe_nat_blocked_peers=probe_blocked,
            update_relay_preferences=update_relay,
        )

        # Force STUN probe by setting last probe time to 0
        loop._last_stun_probe = 0

        await loop._run_once()

        detect_nat.assert_called_once()
        probe_blocked.assert_called_once()
        update_relay.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_once_skips_stun_when_recent(self):
        """Test _run_once skips STUN probe when recently done."""
        import time

        detect_nat = AsyncMock()
        probe_blocked = AsyncMock()
        update_relay = AsyncMock()

        loop = NATManagementLoop(
            detect_nat_type=detect_nat,
            probe_nat_blocked_peers=probe_blocked,
            update_relay_preferences=update_relay,
        )

        # Set last probe to recent time
        loop._last_stun_probe = time.time()

        await loop._run_once()

        detect_nat.assert_not_called()
        probe_blocked.assert_called_once()
        update_relay.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_once_calls_validate_relay(self):
        """Test _run_once calls validate_relay_assignments when provided."""
        validate_relay = AsyncMock()

        loop = NATManagementLoop(
            detect_nat_type=AsyncMock(),
            probe_nat_blocked_peers=AsyncMock(),
            update_relay_preferences=AsyncMock(),
            validate_relay_assignments=validate_relay,
        )

        await loop._run_once()

        validate_relay.assert_called_once()
        assert loop._relay_validations_count == 1

    def test_get_nat_stats(self):
        """Test get_nat_stats returns correct data."""
        loop = NATManagementLoop(
            detect_nat_type=AsyncMock(),
            probe_nat_blocked_peers=AsyncMock(),
            update_relay_preferences=AsyncMock(),
        )
        loop._stun_probes_count = 5
        loop._nat_recovery_attempts = 10

        stats = loop.get_nat_stats()

        assert stats["stun_probes"] == 5
        assert stats["nat_recovery_attempts"] == 10


# =============================================================================
# TailscalePeerDiscoveryConfig Tests
# =============================================================================


class TestTailscalePeerDiscoveryConfig:
    """Tests for TailscalePeerDiscoveryConfig dataclass."""

    def test_default_values(self):
        """Test TailscalePeerDiscoveryConfig has correct defaults."""
        config = TailscalePeerDiscoveryConfig()

        assert config.bootstrap_interval_seconds == 60.0
        assert config.maintenance_interval_seconds == 120.0
        assert config.min_peers_for_maintenance == 5
        assert config.interval_jitter == 0.1
        assert config.connect_timeout_seconds == 10.0
        assert config.max_nodes_per_cycle == 10
        assert config.p2p_port == 8770

    def test_validation_bootstrap_interval_zero(self):
        """Test validation rejects bootstrap_interval_seconds <= 0."""
        with pytest.raises(ValueError, match="bootstrap_interval_seconds must be > 0"):
            TailscalePeerDiscoveryConfig(bootstrap_interval_seconds=0)

    def test_validation_maintenance_interval_zero(self):
        """Test validation rejects maintenance_interval_seconds <= 0."""
        with pytest.raises(ValueError, match="maintenance_interval_seconds must be > 0"):
            TailscalePeerDiscoveryConfig(maintenance_interval_seconds=0)

    def test_validation_min_peers_negative(self):
        """Test validation rejects min_peers_for_maintenance < 0."""
        with pytest.raises(ValueError, match="min_peers_for_maintenance must be >= 0"):
            TailscalePeerDiscoveryConfig(min_peers_for_maintenance=-1)

    def test_validation_jitter_negative(self):
        """Test validation rejects jitter < 0."""
        with pytest.raises(ValueError, match="interval_jitter must be between 0 and 1"):
            TailscalePeerDiscoveryConfig(interval_jitter=-0.1)

    def test_validation_jitter_greater_than_one(self):
        """Test validation rejects jitter > 1."""
        with pytest.raises(ValueError, match="interval_jitter must be between 0 and 1"):
            TailscalePeerDiscoveryConfig(interval_jitter=1.5)

    def test_validation_connect_timeout_zero(self):
        """Test validation rejects connect_timeout_seconds <= 0."""
        with pytest.raises(ValueError, match="connect_timeout_seconds must be > 0"):
            TailscalePeerDiscoveryConfig(connect_timeout_seconds=0)

    def test_validation_max_nodes_zero(self):
        """Test validation rejects max_nodes_per_cycle <= 0."""
        with pytest.raises(ValueError, match="max_nodes_per_cycle must be > 0"):
            TailscalePeerDiscoveryConfig(max_nodes_per_cycle=0)

    def test_validation_port_zero(self):
        """Test validation rejects port < 1."""
        with pytest.raises(ValueError, match="p2p_port must be between 1 and 65535"):
            TailscalePeerDiscoveryConfig(p2p_port=0)

    def test_validation_port_too_high(self):
        """Test validation rejects port > 65535."""
        with pytest.raises(ValueError, match="p2p_port must be between 1 and 65535"):
            TailscalePeerDiscoveryConfig(p2p_port=65536)


# =============================================================================
# TailscalePeerDiscoveryLoop Tests
# =============================================================================


class TestTailscalePeerDiscoveryLoop:
    """Tests for TailscalePeerDiscoveryLoop."""

    def test_init(self):
        """Test loop initialization."""
        loop = TailscalePeerDiscoveryLoop(
            is_leader=lambda: True,
            get_current_peers=lambda: set(),
            get_alive_peer_count=lambda: 0,
            probe_and_connect=AsyncMock(return_value=True),
        )

        assert loop.name == "tailscale_peer_discovery"
        assert loop._current_mode == "bootstrap"
        assert loop._nodes_discovered == 0

    @pytest.mark.asyncio
    async def test_run_once_updates_mode_bootstrap(self):
        """Test _run_once stays in bootstrap mode with few peers."""
        loop = TailscalePeerDiscoveryLoop(
            is_leader=lambda: True,
            get_current_peers=lambda: set(),
            get_alive_peer_count=lambda: 2,  # < 5 = bootstrap mode
            probe_and_connect=AsyncMock(return_value=True),
        )

        with patch.object(loop, "_get_tailscale_peers", return_value=None):
            with patch.object(loop, "_probe_yaml_hosts", return_value=[]):
                await loop._run_once()

        assert loop._current_mode == "bootstrap"

    @pytest.mark.asyncio
    async def test_run_once_switches_to_maintenance(self):
        """Test _run_once switches to maintenance mode with enough peers."""
        loop = TailscalePeerDiscoveryLoop(
            is_leader=lambda: True,
            get_current_peers=lambda: set(),
            get_alive_peer_count=lambda: 10,  # >= 5 = maintenance mode
            probe_and_connect=AsyncMock(return_value=True),
        )

        with patch.object(loop, "_get_tailscale_peers", return_value=None):
            with patch.object(loop, "_probe_yaml_hosts", return_value=[]):
                await loop._run_once()

        assert loop._current_mode == "maintenance"

    def test_get_jittered_interval(self):
        """Test _get_jittered_interval applies jitter."""
        config = TailscalePeerDiscoveryConfig(interval_jitter=0.1)
        loop = TailscalePeerDiscoveryLoop(
            is_leader=lambda: True,
            get_current_peers=lambda: set(),
            get_alive_peer_count=lambda: 0,
            probe_and_connect=AsyncMock(),
            config=config,
        )

        # With 10% jitter on 100s, should be between 90 and 110
        intervals = [loop._get_jittered_interval(100.0) for _ in range(100)]
        assert min(intervals) >= 90.0
        assert max(intervals) <= 110.0

    def test_find_missing_compute_nodes(self):
        """Test _find_missing_compute_nodes filters correctly."""
        loop = TailscalePeerDiscoveryLoop(
            is_leader=lambda: True,
            get_current_peers=lambda: {"existing-node"},
            get_alive_peer_count=lambda: 0,
            probe_and_connect=AsyncMock(),
        )

        ts_peers = {
            "peer1": {
                "HostName": "lambda-gpu-1",
                "Online": True,
                "TailscaleIPs": ["100.64.0.1"],
            },
            "peer2": {
                "HostName": "existing-node",
                "Online": True,
                "TailscaleIPs": ["100.64.0.2"],
            },
            "peer3": {
                "HostName": "non-compute-server",  # Not matching patterns
                "Online": True,
                "TailscaleIPs": ["100.64.0.3"],
            },
            "peer4": {
                "HostName": "vast-4090",
                "Online": False,  # Offline
                "TailscaleIPs": ["100.64.0.4"],
            },
        }

        missing = loop._find_missing_compute_nodes(ts_peers, {"existing-node"})

        # Should only find lambda-gpu-1 (online, compute pattern, not in current peers)
        assert len(missing) == 1
        assert missing[0][0] == "lambda-gpu-1"
        assert missing[0][1] == "100.64.0.1"


# =============================================================================
# HeartbeatConfig Tests
# =============================================================================


class TestHeartbeatConfig:
    """Tests for HeartbeatConfig dataclass."""

    def test_default_values(self):
        """Test HeartbeatConfig has correct defaults."""
        config = HeartbeatConfig()

        assert config.interval_seconds == 15.0
        assert config.relay_heartbeat_interval == 30.0
        assert config.leader_lease_duration == 60.0

    def test_validation_interval_zero(self):
        """Test validation rejects interval_seconds <= 0."""
        with pytest.raises(ValueError, match="interval_seconds must be > 0"):
            HeartbeatConfig(interval_seconds=0)

    def test_validation_relay_interval_zero(self):
        """Test validation rejects relay_heartbeat_interval <= 0."""
        with pytest.raises(ValueError, match="relay_heartbeat_interval must be > 0"):
            HeartbeatConfig(relay_heartbeat_interval=0)

    def test_validation_lease_duration_zero(self):
        """Test validation rejects leader_lease_duration <= 0."""
        with pytest.raises(ValueError, match="leader_lease_duration must be > 0"):
            HeartbeatConfig(leader_lease_duration=0)


# =============================================================================
# HeartbeatLoop Tests
# =============================================================================


class TestHeartbeatLoop:
    """Tests for HeartbeatLoop."""

    def _create_loop(self, **overrides):
        """Create a HeartbeatLoop with default mocked callbacks."""
        defaults = {
            "get_known_peers": lambda: [],
            "get_relay_peers": lambda: set(),
            "get_peers_snapshot": lambda: [],
            "send_heartbeat_to_peer": AsyncMock(return_value=None),
            "send_relay_heartbeat": AsyncMock(return_value={}),
            "update_peer": AsyncMock(),
            "parse_peer_address": lambda addr: ("http", addr.split(":")[0], 8770),
            "get_node_id": lambda: "test-node",
            "get_role": lambda: MagicMock(),
            "get_leader_id": lambda: None,
            "set_leader": AsyncMock(),
            "is_leader_eligible": lambda peer, conflicts: True,
            "is_leader_lease_valid": lambda: False,
            "endpoint_conflict_keys": lambda peers: set(),
            "bootstrap_from_known_peers": AsyncMock(),
            "emit_host_online": AsyncMock(),
            "get_tailscale_ip_for_peer": lambda node_id: None,
            "get_self_info": lambda: MagicMock(),
        }
        defaults.update(overrides)
        return HeartbeatLoop(**defaults)

    def test_init(self):
        """Test loop initialization."""
        loop = self._create_loop()

        assert loop.name == "heartbeat"
        assert loop.interval == 15.0
        assert loop._heartbeats_sent == 0

    @pytest.mark.asyncio
    async def test_run_once_no_peers(self):
        """Test _run_once with no known peers."""
        loop = self._create_loop()

        await loop._run_once()

        assert loop._heartbeats_sent == 0

    @pytest.mark.asyncio
    async def test_first_contact_uses_advertised_capabilities_without_selfplay_fallback(self):
        """First-contact events should preserve role-gated capabilities."""
        info = MagicMock(
            node_id="lambda-gh200-8",
            host="10.0.0.8",
            port=8770,
            capabilities=["training", "gauntlet"],
            has_gpu=True,
            memory_gb=96,
            last_heartbeat=0.0,
            role=MagicMock(),
        )
        emit_host_online = AsyncMock()
        update_peer = AsyncMock()
        send_heartbeat = AsyncMock(return_value=info)
        loop = self._create_loop(
            get_known_peers=lambda: ["10.0.0.8:8770"],
            send_heartbeat_to_peer=send_heartbeat,
            update_peer=update_peer,
            emit_host_online=emit_host_online,
        )

        await loop._run_once()

        emit_host_online.assert_awaited_once_with(
            "lambda-gh200-8",
            ["training", "gauntlet"],
        )


# =============================================================================
# VoterHeartbeatConfig Tests
# =============================================================================


class TestVoterHeartbeatConfig:
    """Tests for VoterHeartbeatConfig dataclass."""

    def test_default_values(self):
        """Test VoterHeartbeatConfig has correct defaults."""
        config = VoterHeartbeatConfig()

        assert config.interval_seconds == 10.0  # Faster than regular heartbeat
        assert config.heartbeat_timeout_seconds == 5.0
        assert config.mesh_refresh_interval_seconds == 60.0
        assert config.nat_recovery_aggressive is True

    def test_validation_interval_zero(self):
        """Test validation rejects interval_seconds <= 0."""
        with pytest.raises(ValueError, match="interval_seconds must be > 0"):
            VoterHeartbeatConfig(interval_seconds=0)

    def test_validation_heartbeat_timeout_zero(self):
        """Test validation rejects heartbeat_timeout_seconds <= 0."""
        with pytest.raises(ValueError, match="heartbeat_timeout_seconds must be > 0"):
            VoterHeartbeatConfig(heartbeat_timeout_seconds=0)

    def test_validation_mesh_refresh_interval_zero(self):
        """Test validation rejects mesh_refresh_interval_seconds <= 0."""
        with pytest.raises(ValueError, match="mesh_refresh_interval_seconds must be > 0"):
            VoterHeartbeatConfig(mesh_refresh_interval_seconds=0)


# =============================================================================
# VoterHeartbeatLoop Tests
# =============================================================================


class TestVoterHeartbeatLoop:
    """Tests for VoterHeartbeatLoop."""

    def _create_loop(self, **overrides):
        """Create a VoterHeartbeatLoop with default mocked callbacks."""
        defaults = {
            "get_voter_node_ids": lambda: [],
            "get_node_id": lambda: "test-node",
            "get_peer": lambda node_id: None,
            "send_voter_heartbeat": AsyncMock(return_value=True),
            "try_alternative_endpoints": AsyncMock(return_value=False),
            "discover_voter_peer": AsyncMock(),
            "refresh_voter_mesh": AsyncMock(),
            "clear_nat_blocked": AsyncMock(),
            "increment_failures": MagicMock(),
        }
        defaults.update(overrides)
        return VoterHeartbeatLoop(**defaults)

    def test_init(self):
        """Test loop initialization."""
        loop = self._create_loop()

        assert loop.name == "voter_heartbeat"
        assert loop.interval == 10.0  # Faster than regular
        assert loop._heartbeats_sent == 0

    @pytest.mark.asyncio
    async def test_on_start_non_voter(self):
        """Test _on_start sets _is_voter to False for non-voters."""
        loop = self._create_loop(
            get_voter_node_ids=lambda: ["voter1", "voter2"],
            get_node_id=lambda: "non-voter-node",
        )

        await loop._on_start()

        assert loop._is_voter is False

    @pytest.mark.asyncio
    async def test_on_start_is_voter(self):
        """Test _on_start sets _is_voter to True for voters."""
        loop = self._create_loop(
            get_voter_node_ids=lambda: ["test-node", "voter2"],
            get_node_id=lambda: "test-node",
        )

        await loop._on_start()

        assert loop._is_voter is True

    @pytest.mark.asyncio
    async def test_run_once_skips_non_voter(self):
        """Test _run_once skips when not a voter."""
        send_hb = AsyncMock(return_value=True)
        loop = self._create_loop(
            get_voter_node_ids=lambda: ["voter1", "voter2"],
            send_voter_heartbeat=send_hb,
        )
        loop._is_voter = False

        await loop._run_once()

        send_hb.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_once_sends_heartbeats(self):
        """Test _run_once sends heartbeats to other voters."""
        mock_peer = MagicMock()
        send_hb = AsyncMock(return_value=True)

        loop = self._create_loop(
            get_voter_node_ids=lambda: ["test-node", "voter2", "voter3"],
            get_node_id=lambda: "test-node",
            get_peer=lambda node_id: mock_peer if node_id != "test-node" else None,
            send_voter_heartbeat=send_hb,
        )
        loop._is_voter = True

        await loop._run_once()

        # Should send to voter2 and voter3 (not self)
        assert send_hb.call_count == 2
        assert loop._heartbeats_sent == 2
        assert loop._heartbeats_succeeded == 2

    @pytest.mark.asyncio
    async def test_run_once_discovers_unknown_peers(self):
        """Test _run_once discovers peers for unknown voters."""
        discover = AsyncMock()

        loop = self._create_loop(
            get_voter_node_ids=lambda: ["test-node", "unknown-voter"],
            get_node_id=lambda: "test-node",
            get_peer=lambda node_id: None,  # Unknown
            discover_voter_peer=discover,
        )
        loop._is_voter = True

        await loop._run_once()

        discover.assert_called_once_with("unknown-voter")


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestNetworkLoopsLifecycle:
    """Integration-style tests for loop lifecycle."""

    @pytest.mark.asyncio
    async def test_ip_discovery_loop_start_stop(self):
        """Test IpDiscoveryLoop can be started and stopped."""
        loop = IpDiscoveryLoop(
            get_nodes=lambda: {},
            update_node_ip=AsyncMock(),
            config=IpDiscoveryConfig(check_interval_seconds=0.1),
        )

        task = loop.start_background()
        await asyncio.sleep(0.15)

        assert loop.running

        loop.stop()
        await asyncio.sleep(0.1)

        assert not loop.running
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_nat_management_loop_start_stop(self):
        """Test NATManagementLoop can be started and stopped."""
        loop = NATManagementLoop(
            detect_nat_type=AsyncMock(),
            probe_nat_blocked_peers=AsyncMock(),
            update_relay_preferences=AsyncMock(),
            config=NATManagementConfig(check_interval_seconds=0.1),
        )

        task = loop.start_background()
        await asyncio.sleep(0.15)

        assert loop.running

        loop.stop()
        await asyncio.sleep(0.1)

        assert not loop.running
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
