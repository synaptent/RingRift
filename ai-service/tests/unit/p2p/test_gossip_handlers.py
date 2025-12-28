"""Tests for scripts.p2p.handlers.gossip module.

Tests the HTTP handler mixin for gossip protocol endpoints.
December 2025.
"""

import gzip
import json
import time
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from scripts.p2p.handlers.gossip import GossipHandlersMixin


class MockNodeRole:
    """Mock NodeRole enum."""

    def __init__(self, value: str):
        self._value = value

    @property
    def value(self):
        return self._value


class MockNodeInfo:
    """Mock NodeInfo with common attributes."""

    def __init__(self):
        self.selfplay_jobs = 2
        self.training_jobs = 1
        self.gpu_percent = 45.0
        self.cpu_percent = 30.0
        self.memory_percent = 50.0
        self.disk_percent = 60.0
        self.has_gpu = True
        self.gpu_name = "RTX 4090"


class MockDataManifest:
    """Mock DataManifest for testing."""

    def __init__(self, total_files: int = 100, selfplay_games: int = 1000):
        self.total_files = total_files
        self.selfplay_games = selfplay_games
        self.collected_at = time.time()

    def to_dict(self):
        return {
            "total_files": self.total_files,
            "selfplay_games": self.selfplay_games,
            "collected_at": self.collected_at,
        }


class GossipHandlersTestClass(GossipHandlersMixin):
    """Test class that uses the gossip handlers mixin."""

    def __init__(self):
        self.node_id = "test-node-1"
        self.leader_id = "leader-node"
        self.auth_token = None
        self.role = MockNodeRole("follower")
        self.self_info = MockNodeInfo()
        self.local_data_manifest = MockDataManifest()
        self.leader_lease_expires = time.time() + 3600
        self._gossip_peer_states = {}
        self._authorization_failed = False
        self._gossip_metrics_recorded = []
        self._voter_quorum_ok = True

    def _is_request_authorized(self, request):
        return not self._authorization_failed

    def _process_gossip_response(self, data):
        """Mock processing of gossip response."""
        pass

    def _record_gossip_metrics(self, metric_type: str, node_id: str | None = None):
        self._gossip_metrics_recorded.append((metric_type, node_id))

    def _update_self_info(self):
        """Mock update_self_info."""
        pass

    def _get_gossip_known_states(self):
        return {"peer-1": {"node_id": "peer-1", "version": 1000}}

    def _has_voter_quorum(self):
        return self._voter_quorum_ok


@pytest.fixture
def handler():
    """Create a test handler instance."""
    return GossipHandlersTestClass()


class TestGossipHandlersMixinAttributes:
    """Test required attributes on the mixin."""

    def test_has_node_id_hint(self):
        """Mixin should have node_id type hint."""
        hints = GossipHandlersMixin.__annotations__
        assert "node_id" in hints
        assert hints["node_id"] is str or hints["node_id"] == "str"

    def test_has_leader_id_hint(self):
        """Mixin should have leader_id type hint."""
        hints = GossipHandlersMixin.__annotations__
        assert "leader_id" in hints

    def test_has_auth_token_hint(self):
        """Mixin should have auth_token type hint."""
        hints = GossipHandlersMixin.__annotations__
        assert "auth_token" in hints


class TestHandleGossip:
    """Tests for the /gossip endpoint handler."""

    @pytest.mark.asyncio
    async def test_gossip_returns_our_state(self, handler):
        """Gossip response should include our node's state."""
        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={
            "sender": "peer-node",
            "sender_state": {"node_id": "peer-node"},
            "known_states": {},
        })

        response = await handler.handle_gossip(request)

        # Response is gzip-compressed
        assert response.headers.get("Content-Encoding") == "gzip"
        decompressed = gzip.decompress(response.body)
        data = json.loads(decompressed)

        assert "sender_state" in data
        assert data["sender_state"]["node_id"] == "test-node-1"
        assert data["sender_state"]["leader_id"] == "leader-node"

    @pytest.mark.asyncio
    async def test_gossip_includes_role(self, handler):
        """Response should include our role."""
        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={})

        response = await handler.handle_gossip(request)
        decompressed = gzip.decompress(response.body)
        data = json.loads(decompressed)

        assert data["sender_state"]["role"] == "follower"

    @pytest.mark.asyncio
    async def test_gossip_includes_resource_metrics(self, handler):
        """Response should include resource metrics from self_info."""
        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={})

        response = await handler.handle_gossip(request)
        decompressed = gzip.decompress(response.body)
        data = json.loads(decompressed)

        state = data["sender_state"]
        assert state["selfplay_jobs"] == 2
        assert state["training_jobs"] == 1
        assert state["gpu_percent"] == 45.0
        assert state["cpu_percent"] == 30.0
        assert state["memory_percent"] == 50.0
        assert state["disk_percent"] == 60.0
        assert state["has_gpu"] is True
        assert state["gpu_name"] == "RTX 4090"

    @pytest.mark.asyncio
    async def test_gossip_includes_manifest_summary(self, handler):
        """Response should include manifest summary if available."""
        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={})

        response = await handler.handle_gossip(request)
        decompressed = gzip.decompress(response.body)
        data = json.loads(decompressed)

        assert "manifest_summary" in data["sender_state"]
        summary = data["sender_state"]["manifest_summary"]
        assert summary["total_files"] == 100
        assert summary["selfplay_games"] == 1000

    @pytest.mark.asyncio
    async def test_gossip_includes_known_states(self, handler):
        """Response should include known peer states."""
        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={})

        response = await handler.handle_gossip(request)
        decompressed = gzip.decompress(response.body)
        data = json.loads(decompressed)

        assert "known_states" in data
        assert "peer-1" in data["known_states"]

    @pytest.mark.asyncio
    async def test_gossip_includes_peer_manifests(self, handler):
        """Response should include peer manifests for sync planning."""
        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={})

        response = await handler.handle_gossip(request)
        decompressed = gzip.decompress(response.body)
        data = json.loads(decompressed)

        assert "peer_manifests" in data
        assert "test-node-1" in data["peer_manifests"]

    @pytest.mark.asyncio
    async def test_gossip_includes_voter_quorum(self, handler):
        """Response should include voter quorum status."""
        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={})

        response = await handler.handle_gossip(request)
        decompressed = gzip.decompress(response.body)
        data = json.loads(decompressed)

        assert data["sender_state"]["voter_quorum_ok"] is True

    @pytest.mark.asyncio
    async def test_gossip_voter_quorum_false(self, handler):
        """Response reflects quorum failure."""
        handler._voter_quorum_ok = False
        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={})

        response = await handler.handle_gossip(request)
        decompressed = gzip.decompress(response.body)
        data = json.loads(decompressed)

        assert data["sender_state"]["voter_quorum_ok"] is False

    @pytest.mark.asyncio
    async def test_gossip_records_received_metric(self, handler):
        """Gossip handler should record received metric."""
        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={})

        await handler.handle_gossip(request)

        assert ("received", None) in handler._gossip_metrics_recorded

    @pytest.mark.asyncio
    async def test_gossip_unauthorized_with_token(self, handler):
        """Gossip should return 401 when auth fails."""
        handler.auth_token = "secret"
        handler._authorization_failed = True

        request = MagicMock()
        request.headers = {}

        response = await handler.handle_gossip(request)

        assert response.status == 401

    @pytest.mark.asyncio
    async def test_gossip_handles_json_decode_error(self, handler):
        """Handler should gracefully handle JSON decode errors."""
        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(side_effect=json.JSONDecodeError("msg", "doc", 0))

        # Should not raise, should return a valid response
        response = await handler.handle_gossip(request)
        assert response.status == 200


class TestGossipGzipHandling:
    """Tests for gzip compression handling in gossip."""

    @pytest.mark.asyncio
    async def test_gossip_handles_gzip_request(self, handler):
        """Handler should decompress gzip requests."""
        payload = {"sender": "peer", "sender_state": {}, "known_states": {}}
        compressed = gzip.compress(json.dumps(payload).encode("utf-8"))

        request = MagicMock()
        request.headers = {"Content-Encoding": "gzip"}
        request.read = AsyncMock(return_value=compressed)

        response = await handler.handle_gossip(request)

        assert response.status == 200

    @pytest.mark.asyncio
    async def test_gossip_handles_fake_gzip_header(self, handler):
        """Handler should handle requests claiming gzip but sending raw JSON."""
        # Client sets Content-Encoding: gzip but sends raw JSON (no magic bytes)
        payload = {"sender": "peer", "sender_state": {}, "known_states": {}}
        raw_json = json.dumps(payload).encode("utf-8")

        request = MagicMock()
        request.headers = {"Content-Encoding": "gzip"}
        request.read = AsyncMock(return_value=raw_json)

        # Should handle gracefully by parsing as raw JSON
        response = await handler.handle_gossip(request)

        assert response.status == 200

    @pytest.mark.asyncio
    async def test_gossip_checks_gzip_magic_bytes(self, handler):
        """Handler should check for gzip magic bytes before decompression."""
        # Magic bytes are 0x1f 0x8b
        payload = {"sender": "peer"}
        compressed = gzip.compress(json.dumps(payload).encode("utf-8"))

        assert compressed[:2] == b"\x1f\x8b"  # Verify magic bytes

        request = MagicMock()
        request.headers = {"Content-Encoding": "gzip"}
        request.read = AsyncMock(return_value=compressed)

        response = await handler.handle_gossip(request)
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_gossip_response_always_gzipped(self, handler):
        """Response should always be gzip-compressed."""
        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={})

        response = await handler.handle_gossip(request)

        assert response.headers.get("Content-Encoding") == "gzip"
        assert response.content_type == "application/json"


class TestHandleGossipAntiEntropy:
    """Tests for the /gossip/anti-entropy endpoint handler."""

    @pytest.mark.asyncio
    async def test_anti_entropy_returns_all_states(self, handler):
        """Anti-entropy should return all known states."""
        handler._gossip_peer_states = {
            "peer-1": {"node_id": "peer-1", "version": 1000},
            "peer-2": {"node_id": "peer-2", "version": 2000},
        }

        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={
            "anti_entropy": True,
            "sender": "peer-3",
            "all_known_states": {},
        })

        response = await handler.handle_gossip_anti_entropy(request)
        data = json.loads(response.body)

        assert data["anti_entropy"] is True
        assert data["sender"] == "test-node-1"
        assert "all_known_states" in data
        assert "peer-1" in data["all_known_states"]
        assert "peer-2" in data["all_known_states"]
        assert "test-node-1" in data["all_known_states"]

    @pytest.mark.asyncio
    async def test_anti_entropy_includes_our_state(self, handler):
        """Anti-entropy response includes our own state."""
        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={})

        response = await handler.handle_gossip_anti_entropy(request)
        data = json.loads(response.body)

        our_state = data["all_known_states"]["test-node-1"]
        assert our_state["node_id"] == "test-node-1"
        assert our_state["role"] == "follower"
        assert our_state["selfplay_jobs"] == 2

    @pytest.mark.asyncio
    async def test_anti_entropy_processes_peer_states(self, handler):
        """Anti-entropy should update from peer's states."""
        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={
            "anti_entropy": True,
            "sender": "peer-3",
            "all_known_states": {
                "peer-4": {"node_id": "peer-4", "version": 5000, "gpu_percent": 80},
            },
        })

        response = await handler.handle_gossip_anti_entropy(request)
        data = json.loads(response.body)

        # We should have learned about peer-4
        assert "peer-4" in handler._gossip_peer_states
        assert handler._gossip_peer_states["peer-4"]["version"] == 5000

        # Response should include update count
        assert data["updates_applied"] >= 1

    @pytest.mark.asyncio
    async def test_anti_entropy_ignores_stale_updates(self, handler):
        """Anti-entropy should ignore stale (lower version) updates."""
        handler._gossip_peer_states = {
            "peer-1": {"node_id": "peer-1", "version": 5000},
        }

        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={
            "all_known_states": {
                "peer-1": {"node_id": "peer-1", "version": 1000},  # Stale
            },
        })

        await handler.handle_gossip_anti_entropy(request)

        # Our version should remain at 5000
        assert handler._gossip_peer_states["peer-1"]["version"] == 5000

    @pytest.mark.asyncio
    async def test_anti_entropy_ignores_own_state(self, handler):
        """Anti-entropy should not update our own state from peers."""
        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={
            "all_known_states": {
                "test-node-1": {"node_id": "test-node-1", "version": 99999},
            },
        })

        await handler.handle_gossip_anti_entropy(request)

        # Should not store our own node in peer states
        assert "test-node-1" not in handler._gossip_peer_states

    @pytest.mark.asyncio
    async def test_anti_entropy_records_metrics(self, handler):
        """Anti-entropy should record gossip metrics."""
        handler._gossip_peer_states = {}

        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={
            "all_known_states": {
                "peer-1": {"node_id": "peer-1", "version": 1000},
            },
        })

        await handler.handle_gossip_anti_entropy(request)

        # Should record received metric
        assert ("received", None) in handler._gossip_metrics_recorded
        # Should record update metric
        assert ("update", "peer-1") in handler._gossip_metrics_recorded
        # Should record anti_entropy metric
        assert ("anti_entropy", None) in handler._gossip_metrics_recorded

    @pytest.mark.asyncio
    async def test_anti_entropy_unauthorized(self, handler):
        """Anti-entropy should return 401 when auth fails."""
        handler.auth_token = "secret"
        handler._authorization_failed = True

        request = MagicMock()
        request.headers = {}

        response = await handler.handle_gossip_anti_entropy(request)

        assert response.status == 401

    @pytest.mark.asyncio
    async def test_anti_entropy_handles_attribute_error(self, handler):
        """Handler should gracefully handle request parsing issues."""
        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(side_effect=AttributeError("no json"))

        # Should not raise
        response = await handler.handle_gossip_anti_entropy(request)
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_anti_entropy_initializes_peer_states(self, handler):
        """Anti-entropy should initialize peer states dict if missing."""
        delattr(handler, "_gossip_peer_states")

        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={
            "all_known_states": {"peer-1": {"node_id": "peer-1", "version": 1}},
        })

        await handler.handle_gossip_anti_entropy(request)

        assert hasattr(handler, "_gossip_peer_states")
        assert "peer-1" in handler._gossip_peer_states

    @pytest.mark.asyncio
    async def test_anti_entropy_includes_timestamp(self, handler):
        """Anti-entropy response includes timestamp."""
        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={})

        before = time.time()
        response = await handler.handle_gossip_anti_entropy(request)
        after = time.time()

        data = json.loads(response.body)
        assert before <= data["timestamp"] <= after


class TestAntiEntropyEventEmission:
    """Tests for event emission in anti-entropy handler."""

    @pytest.mark.asyncio
    async def test_anti_entropy_emits_node_online_for_new_peers(self, handler):
        """Anti-entropy should emit node online events for new peers."""
        # Mock the event bridge
        mock_bridge = MagicMock()
        mock_bridge.emit = AsyncMock()

        with patch("scripts.p2p.handlers.gossip._event_bridge", mock_bridge):

            handler._gossip_peer_states = {}

            request = MagicMock()
            request.headers = {}
            request.json = AsyncMock(return_value={
                "all_known_states": {
                    "new-peer": {
                        "node_id": "new-peer",
                        "version": 1000,
                        "role": "worker",
                        "has_gpu": True,
                        "gpu_name": "RTX 4090",
                    },
                },
            })

            await handler.handle_gossip_anti_entropy(request)

            mock_bridge.emit.assert_called_once()
            args = mock_bridge.emit.call_args
            assert args[0][0] == "p2p_node_online"
            event_data = args[0][1]
            assert event_data["node_id"] == "new-peer"
            assert event_data["host_type"] == "worker"
            assert event_data["capabilities"]["has_gpu"] is True

    @pytest.mark.asyncio
    async def test_anti_entropy_no_event_for_existing_peers(self, handler):
        """Anti-entropy should not emit events for known peers (even with updates)."""
        # Mock the event bridge
        mock_bridge = MagicMock()
        mock_bridge.emit = AsyncMock()

        with patch("scripts.p2p.handlers.gossip._event_bridge", mock_bridge):

            # Peer already known
            handler._gossip_peer_states = {
                "existing-peer": {"node_id": "existing-peer", "version": 100},
            }

            request = MagicMock()
            request.headers = {}
            request.json = AsyncMock(return_value={
                "all_known_states": {
                    "existing-peer": {"node_id": "existing-peer", "version": 200},  # Update
                },
            })

            await handler.handle_gossip_anti_entropy(request)

            # Should not emit for existing peers
            mock_bridge.emit.assert_not_called()


class TestGossipErrorHandling:
    """Tests for error handling in gossip handlers."""

    @pytest.mark.asyncio
    async def test_gossip_handles_exception_gracefully(self, handler):
        """Gossip handler should return 500 on unexpected error."""
        # Make _process_gossip_response raise an exception
        handler._process_gossip_response = MagicMock(
            side_effect=RuntimeError("Unexpected error")
        )

        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={})

        response = await handler.handle_gossip(request)

        assert response.status == 500
        data = json.loads(response.body)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_anti_entropy_handles_exception_gracefully(self, handler):
        """Anti-entropy handler should return 500 on unexpected error."""
        handler._record_gossip_metrics = MagicMock(
            side_effect=RuntimeError("Metrics error")
        )

        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={})

        response = await handler.handle_gossip_anti_entropy(request)

        assert response.status == 500
        data = json.loads(response.body)
        assert "error" in data


class TestGossipStateVersion:
    """Tests for state version handling."""

    @pytest.mark.asyncio
    async def test_gossip_includes_version_timestamp(self, handler):
        """State version should be based on current timestamp."""
        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={})

        before = int(time.time() * 1000)
        response = await handler.handle_gossip(request)
        after = int(time.time() * 1000)

        decompressed = gzip.decompress(response.body)
        data = json.loads(decompressed)

        version = data["sender_state"]["version"]
        assert before <= version <= after + 1  # Allow 1ms tolerance

    @pytest.mark.asyncio
    async def test_gossip_includes_leader_lease_expires(self, handler):
        """State should include leader lease expiration."""
        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={})

        response = await handler.handle_gossip(request)
        decompressed = gzip.decompress(response.body)
        data = json.loads(decompressed)

        assert "leader_lease_expires" in data["sender_state"]
        assert data["sender_state"]["leader_lease_expires"] > 0


class TestGossipWithoutManifest:
    """Tests for gossip behavior when manifest is not available."""

    @pytest.mark.asyncio
    async def test_gossip_works_without_manifest(self, handler):
        """Gossip should work when local_data_manifest is None."""
        handler.local_data_manifest = None

        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={})

        response = await handler.handle_gossip(request)
        decompressed = gzip.decompress(response.body)
        data = json.loads(decompressed)

        # Should work, manifest_summary should be absent
        assert response.status == 200
        assert "manifest_summary" not in data["sender_state"]

    @pytest.mark.asyncio
    async def test_gossip_peer_manifests_empty_without_manifest(self, handler):
        """Peer manifests should be empty when manifest has no to_dict."""
        handler.local_data_manifest = MagicMock(spec=[])  # No to_dict method

        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={})

        response = await handler.handle_gossip(request)
        decompressed = gzip.decompress(response.body)
        data = json.loads(decompressed)

        # peer_manifests should exist but be empty
        assert data["peer_manifests"] == {}
