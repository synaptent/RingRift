"""Tests for scripts.p2p.gossip_protocol module.

Phase 3 extraction tests - Dec 26, 2025.
"""

import asyncio
import threading
import time
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from scripts.p2p.gossip_protocol import GossipProtocolMixin
from scripts.p2p.models import NodeInfo

# Note: GossipMetricsMixin was merged into GossipProtocolMixin (Dec 2025)
# The deprecated gossip_metrics module now re-exports GossipProtocolMixin


class MockNodeRole:
    """Mock NodeRole enum."""
    LEADER = "leader"
    FOLLOWER = "follower"

    @property
    def value(self):
        return self._value

    def __init__(self, value):
        self._value = value


class GossipProtocolTestClass(GossipProtocolMixin):
    """Test class that uses the gossip protocol mixin."""

    def __init__(self):
        self.node_id = "test-node"
        self.peers: dict = {}
        self.peers_lock = threading.RLock()
        self.leader_id = None
        self.role = MockNodeRole("follower")
        self.self_info = MagicMock()
        self.self_info.selfplay_jobs = 2
        self.self_info.training_jobs = 1
        self.self_info.gpu_percent = 45.0
        self.self_info.cpu_percent = 30.0
        self.self_info.memory_percent = 50.0
        self.self_info.disk_percent = 60.0
        self.self_info.has_gpu = True
        self.self_info.gpu_name = "RTX 4090"
        self.verbose = False
        self.last_leader_seen = 0.0
        self._cluster_epoch = 1
        self._gossip_peer_states = {}
        self._gossip_peer_manifests = {}
        self._gossip_learned_endpoints = {}

        # Initialize gossip metrics
        self._init_gossip_metrics()

    def _update_self_info(self):
        """Mock update_self_info."""
        pass

    def _urls_for_peer(self, peer, path):
        """Mock URL generator."""
        return [f"http://{peer.host}:{peer.port}{path}"]

    def _auth_headers(self):
        """Mock auth headers."""
        return {"Authorization": "test-token"}

    def _has_voter_quorum(self):
        """Mock voter quorum check."""
        return True

    def _save_cluster_epoch(self):
        """Mock save cluster epoch."""
        pass

    async def _send_heartbeat_to_peer(self, host, port):
        """Mock heartbeat sender."""
        return None

    def _save_peer_to_cache(self, node_id, host, port, tailscale_ip):
        """Mock peer cache saver."""
        pass


class TestGossipProtocolMixin:
    """Tests for GossipProtocolMixin."""

    def test_init_gossip_protocol(self):
        """Test gossip protocol initialization."""
        obj = GossipProtocolTestClass()
        obj._init_gossip_protocol()

        assert hasattr(obj, "_gossip_peer_states")
        assert hasattr(obj, "_gossip_peer_manifests")
        assert hasattr(obj, "_gossip_learned_endpoints")
        assert hasattr(obj, "_last_gossip_time")
        assert hasattr(obj, "_last_anti_entropy_repair")

    def test_build_local_gossip_state(self):
        """Test building local gossip state."""
        obj = GossipProtocolTestClass()
        now = time.time()

        state = obj._build_local_gossip_state(now)

        assert state["node_id"] == "test-node"
        assert state["timestamp"] == now
        assert state["version"] == int(now * 1000)
        assert state["selfplay_jobs"] == 2
        assert state["training_jobs"] == 1
        assert state["has_gpu"] is True
        assert state["gpu_name"] == "RTX 4090"
        assert state["voter_quorum_ok"] is True

    def test_get_gossip_known_states_returns_recent(self):
        """Test that only recent states are returned."""
        obj = GossipProtocolTestClass()
        now = time.time()

        # Add a recent state
        obj._gossip_peer_states["peer-1"] = {
            "node_id": "peer-1",
            "timestamp": now - 60,  # 1 minute ago
            "version": 1000,
        }

        # Add an old state (should be excluded)
        obj._gossip_peer_states["peer-2"] = {
            "node_id": "peer-2",
            "timestamp": now - 400,  # > 5 minutes ago
            "version": 900,
        }

        known = obj._get_gossip_known_states()

        assert "peer-1" in known
        assert "peer-2" not in known

    def test_get_peer_endpoints_for_gossip(self):
        """Test peer endpoint collection for gossip."""
        obj = GossipProtocolTestClass()

        # Add a mock peer
        peer = MagicMock()
        peer.node_id = "peer-1"
        peer.host = "192.168.1.1"
        peer.port = 8770
        peer.tailscale_ip = "100.1.2.3"
        peer.last_heartbeat = time.time()
        peer.is_alive.return_value = True
        peer.retired = False

        obj.peers["peer-1"] = peer

        endpoints = obj._get_peer_endpoints_for_gossip()

        assert len(endpoints) == 1
        assert endpoints[0]["node_id"] == "peer-1"
        assert endpoints[0]["host"] == "192.168.1.1"
        assert endpoints[0]["port"] == 8770
        assert endpoints[0]["tailscale_ip"] == "100.1.2.3"

    def test_process_sender_state_updates_leader(self):
        """Test that processing sender state updates leader info."""
        obj = GossipProtocolTestClass()
        obj.leader_id = None

        sender_state = {
            "node_id": "peer-1",
            "version": 2000,
            "leader_id": "peer-leader",
            "leader_lease_expires": time.time() + 60,  # Future lease
        }

        obj._process_sender_state(sender_state)

        assert obj.leader_id == "peer-leader"
        assert obj.last_leader_seen > 0

    def test_process_sender_state_ignores_self(self):
        """Test that processing sender state ignores own state."""
        obj = GossipProtocolTestClass()

        sender_state = {
            "node_id": "test-node",  # Same as self
            "version": 2000,
        }

        obj._process_sender_state(sender_state)

        assert "test-node" not in obj._gossip_peer_states

    def test_process_known_states(self):
        """Test processing known states from propagation."""
        obj = GossipProtocolTestClass()

        known_states = {
            "peer-1": {"node_id": "peer-1", "version": 1000},
            "peer-2": {"node_id": "peer-2", "version": 2000},
            "test-node": {"node_id": "test-node", "version": 3000},  # Self, should be ignored
        }

        obj._process_known_states(known_states)

        assert "peer-1" in obj._gossip_peer_states
        assert "peer-2" in obj._gossip_peer_states
        assert "test-node" not in obj._gossip_peer_states  # Self excluded

    def test_process_known_states_version_wins(self):
        """Test that higher version states win."""
        obj = GossipProtocolTestClass()

        # Add existing state
        obj._gossip_peer_states["peer-1"] = {"node_id": "peer-1", "version": 2000}

        # Try to update with older version
        known_states = {"peer-1": {"node_id": "peer-1", "version": 1000}}
        obj._process_known_states(known_states)

        assert obj._gossip_peer_states["peer-1"]["version"] == 2000  # Old version kept

        # Now update with newer version
        known_states = {"peer-1": {"node_id": "peer-1", "version": 3000}}
        obj._process_known_states(known_states)

        assert obj._gossip_peer_states["peer-1"]["version"] == 3000  # New version adopted

    def test_handle_incoming_cluster_epoch_adopts_higher(self):
        """Test that higher cluster epoch is adopted."""
        obj = GossipProtocolTestClass()
        obj._cluster_epoch = 1

        response = {"sender_state": {"leader_id": None}}
        obj._handle_incoming_cluster_epoch(5, response)

        assert obj._cluster_epoch == 5

    def test_handle_incoming_cluster_epoch_ignores_lower(self):
        """Test that lower cluster epoch is ignored."""
        obj = GossipProtocolTestClass()
        obj._cluster_epoch = 10

        response = {"sender_state": {"leader_id": None}}
        obj._handle_incoming_cluster_epoch(5, response)

        assert obj._cluster_epoch == 10

    def test_handle_incoming_cluster_epoch_invalid_value(self):
        """Test that invalid epoch values are handled gracefully."""
        obj = GossipProtocolTestClass()
        obj._cluster_epoch = 1

        response = {"sender_state": {"leader_id": None}}

        # Test with invalid values
        obj._handle_incoming_cluster_epoch("invalid", response)
        assert obj._cluster_epoch == 1

        obj._handle_incoming_cluster_epoch(None, response)
        assert obj._cluster_epoch == 1

    @pytest.mark.asyncio
    async def test_process_gossip_peer_endpoints_adds_to_learned(self):
        """Test that peer endpoints are added to learned endpoints."""
        obj = GossipProtocolTestClass()

        peer_endpoints = [
            {
                "node_id": "new-peer",
                "host": "192.168.1.100",
                "port": 8770,
                "tailscale_ip": "100.1.2.100",
                "last_heartbeat": time.time(),
            }
        ]

        with patch.object(obj, "_try_connect_gossip_peer", new_callable=AsyncMock):
            obj._process_gossip_peer_endpoints(peer_endpoints)
            # Allow async task to run
            await asyncio.sleep(0)

        assert "new-peer" in obj._gossip_learned_endpoints
        assert obj._gossip_learned_endpoints["new-peer"]["host"] == "100.1.2.100"  # Tailscale preferred

    def test_process_gossip_peer_endpoints_ignores_self(self):
        """Test that own endpoints are ignored."""
        obj = GossipProtocolTestClass()

        peer_endpoints = [
            {
                "node_id": "test-node",  # Same as self
                "host": "192.168.1.100",
                "port": 8770,
            }
        ]

        obj._process_gossip_peer_endpoints(peer_endpoints)

        assert "test-node" not in obj._gossip_learned_endpoints

    def test_get_gossip_peer_states(self):
        """Test public API for getting peer states."""
        obj = GossipProtocolTestClass()
        obj._gossip_peer_states = {"peer-1": {"version": 1000}}

        result = obj.get_gossip_peer_states()

        assert result == {"peer-1": {"version": 1000}}
        # Verify it's a copy
        result["peer-2"] = {"version": 2000}
        assert "peer-2" not in obj._gossip_peer_states

    def test_get_gossip_learned_endpoints(self):
        """Test public API for getting learned endpoints."""
        obj = GossipProtocolTestClass()
        obj._gossip_learned_endpoints = {"peer-1": {"host": "192.168.1.1"}}

        result = obj.get_gossip_learned_endpoints()

        assert result == {"peer-1": {"host": "192.168.1.1"}}
        # Verify it's a copy
        result["peer-2"] = {"host": "192.168.1.2"}
        assert "peer-2" not in obj._gossip_learned_endpoints

    def test_build_anti_entropy_state(self):
        """Test building full anti-entropy state."""
        obj = GossipProtocolTestClass()
        obj._gossip_peer_states = {
            "peer-1": {"node_id": "peer-1", "version": 1000}
        }
        now = time.time()

        state = obj._build_anti_entropy_state(now)

        assert state["anti_entropy"] is True
        assert state["sender"] == "test-node"
        assert "peer-1" in state["all_known_states"]
        assert "test-node" in state["all_known_states"]  # Our own state included

    def test_process_anti_entropy_response_counts_updates(self):
        """Test that anti-entropy response processing counts updates."""
        obj = GossipProtocolTestClass()
        now = time.time()

        response_data = {
            "all_known_states": {
                "peer-1": {"node_id": "peer-1", "version": 1000},
                "peer-2": {"node_id": "peer-2", "version": 2000},
            }
        }

        updates = obj._process_anti_entropy_response(response_data, now)

        assert updates == 2
        assert "peer-1" in obj._gossip_peer_states
        assert "peer-2" in obj._gossip_peer_states


class TestGossipProtocolAsync:
    """Async tests for GossipProtocolMixin."""

    @pytest.mark.asyncio
    async def test_gossip_state_to_peers_rate_limited(self):
        """Test that gossip is rate limited."""
        obj = GossipProtocolTestClass()
        obj._last_gossip_time = time.time()  # Just gossiped

        # Add a peer
        peer = MagicMock()
        peer.node_id = "peer-1"
        peer.is_alive.return_value = True
        peer.retired = False
        obj.peers["peer-1"] = peer

        # Should return early due to rate limiting
        await obj._gossip_state_to_peers()

        # No network calls should have been made
        # (rate limiting kicked in)

    @pytest.mark.asyncio
    async def test_gossip_state_to_peers_no_peers(self):
        """Test gossip with no peers."""
        obj = GossipProtocolTestClass()
        obj._last_gossip_time = 0  # Allow gossip

        # Should complete without error
        await obj._gossip_state_to_peers()

    @pytest.mark.asyncio
    async def test_gossip_anti_entropy_repair_rate_limited(self):
        """Test that anti-entropy is rate limited."""
        obj = GossipProtocolTestClass()
        obj._last_anti_entropy_repair = time.time()  # Just repaired

        # Should return early due to rate limiting
        await obj._gossip_anti_entropy_repair()

    @pytest.mark.asyncio
    async def test_try_connect_gossip_peer_already_connected(self):
        """Test connecting to already connected peer."""
        obj = GossipProtocolTestClass()

        # Add already-connected peer
        peer = MagicMock()
        peer.is_alive.return_value = True
        obj.peers["peer-1"] = peer

        # Should return early
        await obj._try_connect_gossip_peer("peer-1", "192.168.1.1", 8770)
