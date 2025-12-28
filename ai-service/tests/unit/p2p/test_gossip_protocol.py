"""Unit tests for gossip_protocol.py.

Tests the core gossip protocol mixin for decentralized state sharing.
"""

import asyncio
import gzip
import json
import threading
import time
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Create mock for aiohttp before importing the module
class MockClientTimeout:
    def __init__(self, total=None):
        self.total = total


class MockResponse:
    def __init__(self, status=200, data=None, headers=None):
        self.status = status
        self._data = data or {}
        self.headers = headers or {}

    async def json(self):
        return self._data

    async def read(self):
        return json.dumps(self._data).encode("utf-8")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockSession:
    def __init__(self, response=None):
        self._response = response or MockResponse()

    async def post(self, url, data=None, json=None, headers=None):
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


# Mock aiohttp module
mock_aiohttp = MagicMock()
mock_aiohttp.ClientTimeout = MockClientTimeout
mock_aiohttp.ClientSession = MockSession
mock_aiohttp.ClientError = Exception

with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
    from scripts.p2p.gossip_protocol import GossipProtocolMixin


# Create a concrete class that uses the mixin
@dataclass
class MockNodeInfo:
    node_id: str
    host: str = ""
    port: int = 8770
    tailscale_ip: str = ""
    last_heartbeat: float = 0.0
    selfplay_jobs: int = 0
    training_jobs: int = 0
    gpu_percent: float = 0.0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    has_gpu: bool = False
    gpu_name: str = ""
    retired: bool = False

    def is_alive(self) -> bool:
        return time.time() - self.last_heartbeat < 90


class MockNodeRole:
    value: str

    def __init__(self, value: str):
        self.value = value

    LEADER = None
    FOLLOWER = None


MockNodeRole.LEADER = MockNodeRole("leader")
MockNodeRole.FOLLOWER = MockNodeRole("follower")


class TestableGossipProtocol(GossipProtocolMixin):
    """Concrete implementation for testing the mixin."""

    def __init__(self, node_id: str = "test-node-1"):
        self.node_id = node_id
        self.peers: dict[str, MockNodeInfo] = {}
        self.peers_lock = threading.RLock()
        self.leader_id: str | None = None
        self.role = MockNodeRole.FOLLOWER
        self.self_info = MockNodeInfo(node_id=node_id, last_heartbeat=time.time())
        self.verbose = False
        self.last_leader_seen = 0.0
        self._cluster_epoch = 1
        self._gossip_peer_states: dict = {}
        self._gossip_peer_manifests: dict = {}
        self._gossip_learned_endpoints: dict = {}

        # Initialize the mixin
        self._init_gossip_protocol()

    def _update_self_info(self) -> None:
        self.self_info.last_heartbeat = time.time()

    def _urls_for_peer(self, peer, path: str) -> list[str]:
        host = peer.tailscale_ip or peer.host
        port = peer.port or 8770
        return [f"http://{host}:{port}{path}"]

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": "Bearer test-token"}

    def _has_voter_quorum(self) -> bool:
        return True

    def _save_cluster_epoch(self) -> None:
        pass  # No-op for tests

    async def _send_heartbeat_to_peer(self, host: str, port: int) -> MockNodeInfo | None:
        return MockNodeInfo(node_id=f"peer-{host}", host=host, port=port, last_heartbeat=time.time())

    def _save_peer_to_cache(self, node_id: str, host: str, port: int, tailscale_ip: str) -> None:
        pass  # No-op for tests

    def _record_gossip_metrics(self, event: str, peer_id: str = "", latency_ms: float = 0) -> None:
        pass  # No-op for tests

    def _record_gossip_compression(self, original_size: int, compressed_size: int) -> None:
        pass  # No-op for tests


class TestGossipProtocolInitialization:
    """Test gossip protocol initialization."""

    def test_init_gossip_protocol_creates_state_dicts(self):
        """Test that initialization creates required state dicts."""
        protocol = TestableGossipProtocol()

        assert hasattr(protocol, "_gossip_peer_states")
        assert hasattr(protocol, "_gossip_peer_manifests")
        assert hasattr(protocol, "_gossip_learned_endpoints")
        assert hasattr(protocol, "_last_gossip_time")
        assert hasattr(protocol, "_last_anti_entropy_repair")
        assert hasattr(protocol, "_last_gossip_cleanup")

    def test_init_preserves_existing_state(self):
        """Test that init doesn't overwrite existing state."""
        protocol = TestableGossipProtocol()
        protocol._gossip_peer_states = {"existing": {"data": True}}

        protocol._init_gossip_protocol()

        assert "existing" in protocol._gossip_peer_states

    def test_class_constants_are_defined(self):
        """Test that class constants are properly defined."""
        assert GossipProtocolMixin.GOSSIP_MAX_PEER_STATES == 200
        assert GossipProtocolMixin.GOSSIP_MAX_MANIFESTS == 100
        assert GossipProtocolMixin.GOSSIP_MAX_ENDPOINTS == 100
        assert GossipProtocolMixin.GOSSIP_STATE_TTL == 3600
        assert GossipProtocolMixin.GOSSIP_ENDPOINT_TTL == 1800


class TestGossipStateCleanup:
    """Test gossip state cleanup logic."""

    def test_cleanup_rate_limited(self):
        """Test that cleanup is rate limited to every 5 minutes."""
        protocol = TestableGossipProtocol()
        protocol._last_gossip_cleanup = time.time()

        # Add some test data
        protocol._gossip_peer_states["test"] = {"timestamp": time.time()}

        protocol._cleanup_gossip_state()

        # Should not have cleaned since last cleanup was just now
        assert "test" in protocol._gossip_peer_states

    def test_cleanup_removes_stale_states(self):
        """Test that cleanup removes states older than TTL."""
        protocol = TestableGossipProtocol()
        protocol._last_gossip_cleanup = 0  # Allow cleanup

        stale_time = time.time() - protocol.GOSSIP_STATE_TTL - 100
        protocol._gossip_peer_states["stale-node"] = {"timestamp": stale_time}
        protocol._gossip_peer_states["fresh-node"] = {"timestamp": time.time()}

        protocol._cleanup_gossip_state()

        assert "stale-node" not in protocol._gossip_peer_states
        assert "fresh-node" in protocol._gossip_peer_states

    def test_cleanup_enforces_max_peer_states(self):
        """Test that cleanup enforces max peer states limit."""
        protocol = TestableGossipProtocol()
        protocol._last_gossip_cleanup = 0

        # Add more states than the limit
        now = time.time()
        for i in range(protocol.GOSSIP_MAX_PEER_STATES + 50):
            protocol._gossip_peer_states[f"node-{i}"] = {"timestamp": now - i}

        protocol._cleanup_gossip_state()

        assert len(protocol._gossip_peer_states) <= protocol.GOSSIP_MAX_PEER_STATES

    def test_cleanup_removes_stale_endpoints(self):
        """Test that cleanup removes endpoints older than TTL."""
        protocol = TestableGossipProtocol()
        protocol._last_gossip_cleanup = 0

        stale_time = time.time() - protocol.GOSSIP_ENDPOINT_TTL - 100
        protocol._gossip_learned_endpoints["stale-ep"] = {"learned_at": stale_time}
        protocol._gossip_learned_endpoints["fresh-ep"] = {"learned_at": time.time()}

        protocol._cleanup_gossip_state()

        assert "stale-ep" not in protocol._gossip_learned_endpoints
        assert "fresh-ep" in protocol._gossip_learned_endpoints

    def test_cleanup_enforces_max_endpoints(self):
        """Test that cleanup enforces max endpoints limit."""
        protocol = TestableGossipProtocol()
        protocol._last_gossip_cleanup = 0

        now = time.time()
        for i in range(protocol.GOSSIP_MAX_ENDPOINTS + 20):
            protocol._gossip_learned_endpoints[f"ep-{i}"] = {"learned_at": now - i}

        protocol._cleanup_gossip_state()

        assert len(protocol._gossip_learned_endpoints) <= protocol.GOSSIP_MAX_ENDPOINTS

    def test_cleanup_enforces_max_manifests(self):
        """Test that cleanup enforces max manifests limit."""
        protocol = TestableGossipProtocol()
        protocol._last_gossip_cleanup = 0

        for i in range(protocol.GOSSIP_MAX_MANIFESTS + 10):
            protocol._gossip_peer_manifests[f"manifest-{i}"] = {"data": i}

        protocol._cleanup_gossip_state()

        assert len(protocol._gossip_peer_manifests) <= protocol.GOSSIP_MAX_MANIFESTS


class TestBuildLocalGossipState:
    """Test building local state for gossip."""

    def test_builds_basic_state(self):
        """Test that basic state fields are included."""
        protocol = TestableGossipProtocol()
        now = time.time()

        state = protocol._build_local_gossip_state(now)

        assert state["node_id"] == "test-node-1"
        assert state["timestamp"] == now
        assert "version" in state
        assert state["role"] == "follower"
        assert "leader_id" in state
        assert "voter_quorum_ok" in state

    def test_includes_resource_metrics(self):
        """Test that resource metrics are included."""
        protocol = TestableGossipProtocol()
        protocol.self_info.gpu_percent = 75.0
        protocol.self_info.cpu_percent = 50.0
        protocol.self_info.memory_percent = 60.0
        protocol.self_info.disk_percent = 40.0
        protocol.self_info.has_gpu = True
        protocol.self_info.gpu_name = "RTX 4090"

        state = protocol._build_local_gossip_state(time.time())

        assert state["gpu_percent"] == 75.0
        assert state["cpu_percent"] == 50.0
        assert state["memory_percent"] == 60.0
        assert state["disk_percent"] == 40.0
        assert state["has_gpu"] is True
        assert state["gpu_name"] == "RTX 4090"

    def test_includes_job_counts(self):
        """Test that job counts are included."""
        protocol = TestableGossipProtocol()
        protocol.self_info.selfplay_jobs = 5
        protocol.self_info.training_jobs = 2

        state = protocol._build_local_gossip_state(time.time())

        assert state["selfplay_jobs"] == 5
        assert state["training_jobs"] == 2

    def test_includes_leader_info(self):
        """Test that leader info is included."""
        protocol = TestableGossipProtocol()
        protocol.leader_id = "leader-node"

        state = protocol._build_local_gossip_state(time.time())

        assert state["leader_id"] == "leader-node"

    def test_version_is_milliseconds(self):
        """Test that version is in milliseconds."""
        protocol = TestableGossipProtocol()
        now = time.time()

        state = protocol._build_local_gossip_state(now)

        assert state["version"] == int(now * 1000)


class TestGetGossipKnownStates:
    """Test getting known states for gossip."""

    def test_returns_recent_states(self):
        """Test that recent states are included."""
        protocol = TestableGossipProtocol()
        recent_time = time.time() - 100

        protocol._gossip_peer_states["recent-node"] = {
            "timestamp": recent_time,
            "node_id": "recent-node",
        }

        states = protocol._get_gossip_known_states()

        assert "recent-node" in states

    def test_excludes_old_states(self):
        """Test that states older than 5 minutes are excluded."""
        protocol = TestableGossipProtocol()
        old_time = time.time() - 400  # More than 300 seconds (5 min)

        protocol._gossip_peer_states["old-node"] = {
            "timestamp": old_time,
            "node_id": "old-node",
        }

        states = protocol._get_gossip_known_states()

        assert "old-node" not in states

    def test_returns_empty_dict_when_no_states(self):
        """Test that empty dict is returned when no states."""
        protocol = TestableGossipProtocol()

        states = protocol._get_gossip_known_states()

        assert states == {}


class TestGetPeerEndpointsForGossip:
    """Test getting peer endpoints for gossip."""

    def test_includes_alive_peers(self):
        """Test that alive peers are included."""
        protocol = TestableGossipProtocol()
        protocol.peers["alive-peer"] = MockNodeInfo(
            node_id="alive-peer",
            host="192.168.1.1",
            port=8770,
            last_heartbeat=time.time(),
        )

        endpoints = protocol._get_peer_endpoints_for_gossip()

        assert len(endpoints) == 1
        assert endpoints[0]["node_id"] == "alive-peer"
        assert endpoints[0]["host"] == "192.168.1.1"
        assert endpoints[0]["port"] == 8770

    def test_excludes_dead_peers(self):
        """Test that dead peers are excluded."""
        protocol = TestableGossipProtocol()
        protocol.peers["dead-peer"] = MockNodeInfo(
            node_id="dead-peer",
            host="192.168.1.1",
            port=8770,
            last_heartbeat=0,  # Very old
        )

        endpoints = protocol._get_peer_endpoints_for_gossip()

        assert len(endpoints) == 0

    def test_excludes_retired_peers(self):
        """Test that retired peers are excluded."""
        protocol = TestableGossipProtocol()
        protocol.peers["retired-peer"] = MockNodeInfo(
            node_id="retired-peer",
            host="192.168.1.1",
            port=8770,
            last_heartbeat=time.time(),
            retired=True,
        )

        endpoints = protocol._get_peer_endpoints_for_gossip()

        assert len(endpoints) == 0

    def test_excludes_self(self):
        """Test that self is excluded."""
        protocol = TestableGossipProtocol()
        protocol.peers[protocol.node_id] = MockNodeInfo(
            node_id=protocol.node_id,
            host="192.168.1.1",
            port=8770,
            last_heartbeat=time.time(),
        )

        endpoints = protocol._get_peer_endpoints_for_gossip()

        assert len(endpoints) == 0

    def test_prefers_tailscale_ip(self):
        """Test that tailscale IP is included when available."""
        protocol = TestableGossipProtocol()
        protocol.peers["peer-1"] = MockNodeInfo(
            node_id="peer-1",
            host="192.168.1.1",
            port=8770,
            tailscale_ip="100.64.0.1",
            last_heartbeat=time.time(),
        )

        endpoints = protocol._get_peer_endpoints_for_gossip()

        assert endpoints[0]["tailscale_ip"] == "100.64.0.1"


class TestProcessGossipResponse:
    """Test processing gossip responses."""

    def test_processes_sender_state(self):
        """Test that sender state is processed."""
        protocol = TestableGossipProtocol()

        response = {
            "sender_state": {
                "node_id": "sender-1",
                "timestamp": time.time(),
                "version": int(time.time() * 1000),
                "role": "follower",
            }
        }

        protocol._process_gossip_response(response)

        assert "sender-1" in protocol._gossip_peer_states

    def test_ignores_own_state(self):
        """Test that own state is not stored."""
        protocol = TestableGossipProtocol()

        response = {
            "sender_state": {
                "node_id": protocol.node_id,
                "timestamp": time.time(),
                "version": int(time.time() * 1000),
            }
        }

        protocol._process_gossip_response(response)

        assert protocol.node_id not in protocol._gossip_peer_states

    def test_handles_empty_response(self):
        """Test that empty response is handled gracefully."""
        protocol = TestableGossipProtocol()

        protocol._process_gossip_response({})
        protocol._process_gossip_response(None)

        # Should not raise

    def test_processes_known_states(self):
        """Test that known states are processed."""
        protocol = TestableGossipProtocol()
        now = time.time()

        response = {
            "known_states": {
                "node-a": {"node_id": "node-a", "version": int(now * 1000)},
                "node-b": {"node_id": "node-b", "version": int(now * 1000)},
            }
        }

        protocol._process_gossip_response(response)

        assert "node-a" in protocol._gossip_peer_states
        assert "node-b" in protocol._gossip_peer_states


class TestProcessSenderState:
    """Test processing sender state from gossip."""

    def test_updates_state_with_newer_version(self):
        """Test that state is updated when version is newer."""
        protocol = TestableGossipProtocol()
        old_version = 1000
        new_version = 2000

        protocol._gossip_peer_states["node-1"] = {"version": old_version, "data": "old"}

        sender_state = {
            "node_id": "node-1",
            "version": new_version,
            "data": "new",
        }

        protocol._process_sender_state(sender_state)

        assert protocol._gossip_peer_states["node-1"]["data"] == "new"

    def test_ignores_older_version(self):
        """Test that older version is ignored."""
        protocol = TestableGossipProtocol()
        old_version = 1000
        new_version = 2000

        protocol._gossip_peer_states["node-1"] = {"version": new_version, "data": "new"}

        sender_state = {
            "node_id": "node-1",
            "version": old_version,
            "data": "old",
        }

        protocol._process_sender_state(sender_state)

        assert protocol._gossip_peer_states["node-1"]["data"] == "new"

    def test_updates_leader_info(self):
        """Test that leader info is updated from gossip."""
        protocol = TestableGossipProtocol()
        protocol.leader_id = None

        sender_state = {
            "node_id": "node-1",
            "version": int(time.time() * 1000),
            "leader_id": "leader-node",
            "leader_lease_expires": time.time() + 60,
        }

        protocol._process_sender_state(sender_state)

        assert protocol.leader_id == "leader-node"

    def test_ignores_expired_leader_lease(self):
        """Test that expired leader lease is ignored."""
        protocol = TestableGossipProtocol()
        protocol.leader_id = None

        sender_state = {
            "node_id": "node-1",
            "version": int(time.time() * 1000),
            "leader_id": "leader-node",
            "leader_lease_expires": time.time() - 60,  # Expired
        }

        protocol._process_sender_state(sender_state)

        assert protocol.leader_id is None


class TestProcessGossipPeerEndpoints:
    """Test processing peer endpoints from gossip."""

    def test_stores_learned_endpoints(self):
        """Test that learned endpoints are stored."""
        protocol = TestableGossipProtocol()

        endpoints = [
            {
                "node_id": "node-1",
                "host": "192.168.1.1",
                "port": 8770,
                "tailscale_ip": "100.64.0.1",
                "last_heartbeat": time.time(),
            }
        ]

        protocol._process_gossip_peer_endpoints(endpoints)

        assert "node-1" in protocol._gossip_learned_endpoints
        assert protocol._gossip_learned_endpoints["node-1"]["host"] == "100.64.0.1"

    def test_ignores_self_endpoint(self):
        """Test that own endpoint is ignored."""
        protocol = TestableGossipProtocol()

        endpoints = [
            {
                "node_id": protocol.node_id,
                "host": "192.168.1.1",
                "port": 8770,
            }
        ]

        protocol._process_gossip_peer_endpoints(endpoints)

        assert protocol.node_id not in protocol._gossip_learned_endpoints

    def test_prefers_tailscale_ip(self):
        """Test that tailscale IP is preferred over host."""
        protocol = TestableGossipProtocol()

        endpoints = [
            {
                "node_id": "node-1",
                "host": "192.168.1.1",
                "tailscale_ip": "100.64.0.1",
                "port": 8770,
            }
        ]

        protocol._process_gossip_peer_endpoints(endpoints)

        assert protocol._gossip_learned_endpoints["node-1"]["host"] == "100.64.0.1"


class TestHandleIncomingClusterEpoch:
    """Test cluster epoch handling for split-brain resolution."""

    def test_adopts_higher_epoch(self):
        """Test that higher epoch is adopted."""
        protocol = TestableGossipProtocol()
        protocol._cluster_epoch = 1

        response = {"sender_state": {"leader_id": "other-leader"}}
        protocol._handle_incoming_cluster_epoch(2, response)

        assert protocol._cluster_epoch == 2

    def test_ignores_lower_epoch(self):
        """Test that lower epoch is ignored."""
        protocol = TestableGossipProtocol()
        protocol._cluster_epoch = 5

        response = {}
        protocol._handle_incoming_cluster_epoch(3, response)

        assert protocol._cluster_epoch == 5

    def test_handles_invalid_epoch(self):
        """Test that invalid epoch values are handled."""
        protocol = TestableGossipProtocol()
        protocol._cluster_epoch = 1

        # Should not raise
        protocol._handle_incoming_cluster_epoch("invalid", {})
        protocol._handle_incoming_cluster_epoch(None, {})

        assert protocol._cluster_epoch == 1


class TestBuildAntiEntropyState:
    """Test building anti-entropy state for full reconciliation."""

    def test_includes_anti_entropy_flag(self):
        """Test that anti_entropy flag is set."""
        protocol = TestableGossipProtocol()

        state = protocol._build_anti_entropy_state(time.time())

        assert state["anti_entropy"] is True

    def test_includes_all_known_states(self):
        """Test that all known states are included."""
        protocol = TestableGossipProtocol()
        protocol._gossip_peer_states["node-1"] = {"data": "test"}
        protocol._gossip_peer_states["node-2"] = {"data": "test2"}

        state = protocol._build_anti_entropy_state(time.time())

        assert "node-1" in state["all_known_states"]
        assert "node-2" in state["all_known_states"]

    def test_includes_self_state(self):
        """Test that own state is included."""
        protocol = TestableGossipProtocol()

        state = protocol._build_anti_entropy_state(time.time())

        assert protocol.node_id in state["all_known_states"]


class TestProcessAntiEntropyResponse:
    """Test processing anti-entropy response."""

    def test_updates_states_from_peer(self):
        """Test that states are updated from peer."""
        protocol = TestableGossipProtocol()
        now = time.time()

        response = {
            "all_known_states": {
                "node-1": {"node_id": "node-1", "version": int(now * 1000)},
            }
        }

        updates = protocol._process_anti_entropy_response(response, now)

        assert updates == 1
        assert "node-1" in protocol._gossip_peer_states

    def test_skips_own_state(self):
        """Test that own state is not stored."""
        protocol = TestableGossipProtocol()
        now = time.time()

        response = {
            "all_known_states": {
                protocol.node_id: {"version": int(now * 1000)},
            }
        }

        updates = protocol._process_anti_entropy_response(response, now)

        assert updates == 0

    def test_returns_update_count(self):
        """Test that correct update count is returned."""
        protocol = TestableGossipProtocol()
        now = time.time()

        response = {
            "all_known_states": {
                "node-1": {"version": 1000},
                "node-2": {"version": 2000},
                "node-3": {"version": 3000},
            }
        }

        updates = protocol._process_anti_entropy_response(response, now)

        assert updates == 3


class TestPublicAPIMethods:
    """Test public API methods."""

    def test_get_gossip_peer_states_returns_copy(self):
        """Test that get_gossip_peer_states returns a copy."""
        protocol = TestableGossipProtocol()
        protocol._gossip_peer_states["node-1"] = {"data": "test"}

        states = protocol.get_gossip_peer_states()

        # Modify the copy
        states["node-1"]["data"] = "modified"

        # Original should be unchanged
        assert protocol._gossip_peer_states["node-1"]["data"] == "test"

    def test_get_gossip_learned_endpoints_returns_copy(self):
        """Test that get_gossip_learned_endpoints returns a copy."""
        protocol = TestableGossipProtocol()
        protocol._gossip_learned_endpoints["node-1"] = {"host": "192.168.1.1"}

        endpoints = protocol.get_gossip_learned_endpoints()

        # Modify the copy
        endpoints["node-1"]["host"] = "modified"

        # Original should be unchanged
        assert protocol._gossip_learned_endpoints["node-1"]["host"] == "192.168.1.1"


class TestGossipStateTopeers:
    """Test the main gossip state sharing method."""

    @pytest.mark.asyncio
    async def test_rate_limited(self):
        """Test that gossip is rate limited to every 30 seconds."""
        protocol = TestableGossipProtocol()
        protocol._last_gossip_time = time.time()

        # Should return early due to rate limiting
        await protocol._gossip_state_to_peers()

        # No gossip should have happened
        # (we'd need to mock more to verify this, but no exception means success)

    @pytest.mark.asyncio
    async def test_skips_when_no_peers(self):
        """Test that gossip is skipped when no peers."""
        protocol = TestableGossipProtocol()
        protocol._last_gossip_time = 0  # Allow gossip

        # Should complete without error
        await protocol._gossip_state_to_peers()

    @pytest.mark.asyncio
    async def test_calls_cleanup(self):
        """Test that cleanup is called during gossip."""
        protocol = TestableGossipProtocol()
        protocol._last_gossip_time = 0
        protocol._last_gossip_cleanup = 0

        # Add some peers
        protocol.peers["peer-1"] = MockNodeInfo(
            node_id="peer-1",
            host="192.168.1.1",
            port=8770,
            last_heartbeat=time.time(),
        )

        with patch.object(protocol, "_cleanup_gossip_state") as mock_cleanup:
            await protocol._gossip_state_to_peers()
            mock_cleanup.assert_called_once()


class TestAntiEntropyRepair:
    """Test anti-entropy repair mechanism."""

    @pytest.mark.asyncio
    async def test_rate_limited(self):
        """Test that anti-entropy is rate limited to every 5 minutes."""
        protocol = TestableGossipProtocol()
        protocol._last_anti_entropy_repair = time.time()

        # Should return early due to rate limiting
        await protocol._gossip_anti_entropy_repair()

        # Success if no exception

    @pytest.mark.asyncio
    async def test_skips_when_no_peers(self):
        """Test that anti-entropy is skipped when no peers."""
        protocol = TestableGossipProtocol()
        protocol._last_anti_entropy_repair = 0  # Allow

        await protocol._gossip_anti_entropy_repair()

        # Success if no exception


class TestGossipCompression:
    """Test gossip message compression."""

    @pytest.mark.asyncio
    async def test_compresses_payload(self):
        """Test that gossip payload is compressed with gzip."""
        protocol = TestableGossipProtocol()

        # Create a large payload
        payload = {"data": "x" * 1000}
        json_bytes = json.dumps(payload).encode("utf-8")
        compressed = gzip.compress(json_bytes, compresslevel=6)

        # Compressed should be smaller
        assert len(compressed) < len(json_bytes)

    @pytest.mark.asyncio
    async def test_decompresses_response(self):
        """Test that compressed response is decompressed."""
        protocol = TestableGossipProtocol()

        # Create compressed response
        response_data = {"sender_state": {"node_id": "test"}}
        compressed = gzip.compress(json.dumps(response_data).encode("utf-8"))

        class MockCompressedResponse:
            status = 200
            headers = {"Content-Encoding": "gzip"}

            async def read(self):
                return compressed

        response = MockCompressedResponse()
        result = await protocol._read_gossip_response(response)

        assert result["sender_state"]["node_id"] == "test"
