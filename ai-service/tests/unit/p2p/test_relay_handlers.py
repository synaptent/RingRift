"""Tests for scripts.p2p.handlers.relay module.

Tests cover:
- RelayHandlersMixin relay heartbeat endpoint
- Relay enqueue endpoint for NAT-blocked nodes
- Relay peers endpoint
- Relay status endpoint
- Command queue management and delivery

December 2025.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from scripts.p2p.handlers.relay import RelayHandlersMixin


# =============================================================================
# Mock Classes
# =============================================================================


@dataclass
class MockNodeInfo:
    """Mock NodeInfo for testing."""

    node_id: str = "test-node"
    host: str = "192.168.1.100"
    port: int = 8770
    reported_host: str = ""
    reported_port: int = 0
    last_heartbeat: float = 0.0
    nat_blocked: bool = False
    nat_blocked_since: float = 0.0
    relay_via: str = ""
    role: Any = None
    selfplay_jobs: int = 0
    training_jobs: int = 0

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "nat_blocked": self.nat_blocked,
            "relay_via": self.relay_via,
            "role": getattr(self.role, "value", str(self.role)) if self.role else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MockNodeInfo":
        return cls(
            node_id=data.get("node_id", "unknown"),
            host=data.get("host", ""),
            port=data.get("port", 8770),
            reported_host=data.get("reported_host", ""),
            reported_port=data.get("reported_port", 0),
        )


class MockNodeRole:
    """Mock NodeRole enum."""

    LEADER = MagicMock(value="leader")
    FOLLOWER = MagicMock(value="follower")


class RelayHandlersTestClass(RelayHandlersMixin):
    """Test class that uses the relay handlers mixin."""

    def __init__(
        self,
        node_id: str = "relay-node",
        leader_id: str = "leader-node",
    ):
        self.node_id = node_id
        self.leader_id = leader_id
        self.auth_token = None

        # Mock NodeInfo
        self.self_info = MockNodeInfo(node_id=node_id)
        self.peers: dict[str, MockNodeInfo] = {}
        self.peers_lock = threading.RLock()
        self.relay_lock = threading.RLock()
        self.relay_command_queue: dict[str, list] = {}

        # Voter configuration
        self.voter_node_ids = ["voter-1", "voter-2", "voter-3"]
        self.voter_quorum_size = 2
        self.voter_config_source = "yaml"
        self._voter_quorum_ok = True

        # Authorization tracking
        self._auth_failed = False

    def _update_self_info(self) -> None:
        pass

    def get_peers_ro(self) -> dict[str, MockNodeInfo]:
        """Return a lock-free read-only peer snapshot."""
        return dict(self.peers)

    def _get_leader_peer(self) -> MockNodeInfo | None:
        return self.peers.get(self.leader_id)

    def _has_voter_quorum(self) -> bool:
        return self._voter_quorum_ok

    def _is_request_authorized(self, request) -> bool:
        return not self._auth_failed

    def _enqueue_relay_command(
        self, target_node_id: str, cmd_type: str, payload: dict
    ) -> str | None:
        if len(self.relay_command_queue.get(target_node_id, [])) >= 100:
            return None  # Queue full

        cmd_id = f"cmd_{int(time.time() * 1000)}"
        cmd = {
            "id": cmd_id,
            "cmd": cmd_type,
            "payload": payload,
            "ts": time.time(),
            "expires_at": time.time() + 300,
        }

        if target_node_id not in self.relay_command_queue:
            self.relay_command_queue[target_node_id] = []
        self.relay_command_queue[target_node_id].append(cmd)

        return cmd_id


class MockRequest:
    """Mock aiohttp Request for testing."""

    def __init__(
        self,
        headers: dict | None = None,
        json_data: dict | None = None,
        remote: str = "127.0.0.1",
    ):
        self.headers = headers or {}
        self._json_data = json_data or {}
        self.remote = remote

    async def json(self):
        return self._json_data


# =============================================================================
# Relay Heartbeat Tests
# =============================================================================


class TestHandleRelayHeartbeat:
    """Tests for handle_relay_heartbeat endpoint."""

    @pytest.fixture
    def handler(self):
        return RelayHandlersTestClass()

    @pytest.mark.asyncio
    async def test_accepts_heartbeat(self, handler):
        """Heartbeat is accepted and stored."""
        request = MockRequest(
            json_data={
                "node_id": "nat-blocked-1",
                "host": "10.0.0.1",
                "port": 8770,
            }
        )

        # Patch NodeInfo.from_dict to use our mock
        with patch("scripts.p2p.handlers.relay.NodeInfo", MockNodeInfo):
            response = await handler.handle_relay_heartbeat(request)

        body = json.loads(response.body)
        assert body["success"] is True
        assert "nat-blocked-1" in handler.peers

    @pytest.mark.asyncio
    async def test_marks_node_as_nat_blocked(self, handler):
        """Node is marked as NAT-blocked."""
        request = MockRequest(
            json_data={
                "node_id": "nat-blocked-1",
                "host": "10.0.0.1",
                "port": 8770,
            }
        )

        with patch("scripts.p2p.handlers.relay.NodeInfo", MockNodeInfo):
            await handler.handle_relay_heartbeat(request)

        peer = handler.peers.get("nat-blocked-1")
        assert peer is not None
        assert peer.nat_blocked is True

    @pytest.mark.asyncio
    async def test_returns_relay_node_info(self, handler):
        """Response includes relay node identifier."""
        request = MockRequest(json_data={"node_id": "nat-1"})

        with patch("scripts.p2p.handlers.relay.NodeInfo", MockNodeInfo):
            response = await handler.handle_relay_heartbeat(request)

        body = json.loads(response.body)
        assert body["relay_node"] == "relay-node"

    @pytest.mark.asyncio
    async def test_returns_pending_commands(self, handler):
        """Response includes queued commands for the node."""
        handler.relay_command_queue["nat-1"] = [
            {
                "id": "cmd1",
                "cmd": "start_job",
                "payload": {},
                "ts": time.time(),
                "expires_at": time.time() + 300,
            }
        ]
        request = MockRequest(json_data={"node_id": "nat-1"})

        with patch("scripts.p2p.handlers.relay.NodeInfo", MockNodeInfo):
            response = await handler.handle_relay_heartbeat(request)

        body = json.loads(response.body)
        assert len(body["commands"]) == 1
        assert body["commands"][0]["id"] == "cmd1"

    @pytest.mark.asyncio
    async def test_processes_relay_ack(self, handler):
        """ACKed commands are removed from queue."""
        handler.relay_command_queue["nat-1"] = [
            {"id": "cmd1", "cmd": "test", "ts": time.time(), "expires_at": time.time() + 300},
            {"id": "cmd2", "cmd": "test", "ts": time.time(), "expires_at": time.time() + 300},
        ]
        request = MockRequest(
            json_data={
                "node_id": "nat-1",
                "relay_ack": ["cmd1"],
            }
        )

        with patch("scripts.p2p.handlers.relay.NodeInfo", MockNodeInfo):
            await handler.handle_relay_heartbeat(request)

        queue = handler.relay_command_queue["nat-1"]
        assert len(queue) == 1
        assert queue[0]["id"] == "cmd2"

    @pytest.mark.asyncio
    async def test_returns_voter_config(self, handler):
        """Response includes voter configuration."""
        request = MockRequest(json_data={"node_id": "nat-1"})

        with patch("scripts.p2p.handlers.relay.NodeInfo", MockNodeInfo):
            response = await handler.handle_relay_heartbeat(request)

        body = json.loads(response.body)
        assert "voter_node_ids" in body
        assert body["voter_quorum_size"] == 2

    @pytest.mark.asyncio
    async def test_removes_expired_commands(self, handler):
        """Expired commands are not returned."""
        handler.relay_command_queue["nat-1"] = [
            {
                "id": "expired",
                "cmd": "old",
                "ts": time.time() - 400,
                "expires_at": time.time() - 100,  # Expired
            },
            {
                "id": "valid",
                "cmd": "new",
                "ts": time.time(),
                "expires_at": time.time() + 300,
            },
        ]
        request = MockRequest(json_data={"node_id": "nat-1"})

        with patch("scripts.p2p.handlers.relay.NodeInfo", MockNodeInfo):
            response = await handler.handle_relay_heartbeat(request)

        body = json.loads(response.body)
        cmd_ids = [c["id"] for c in body["commands"]]
        assert "expired" not in cmd_ids
        assert "valid" in cmd_ids

    @pytest.mark.asyncio
    async def test_uses_real_ip_from_headers(self, handler):
        """Real IP is extracted from X-Forwarded-For header."""
        request = MockRequest(
            headers={"X-Forwarded-For": "203.0.113.50, 10.0.0.1"},
            json_data={"node_id": "nat-1", "host": "10.0.0.1"},
        )

        with patch("scripts.p2p.handlers.relay.NodeInfo", MockNodeInfo):
            await handler.handle_relay_heartbeat(request)

        peer = handler.peers["nat-1"]
        assert peer.host == "203.0.113.50"

    @pytest.mark.asyncio
    async def test_error_returns_400(self, handler):
        """Invalid request returns 400."""
        request = MockRequest()
        request._json_data = None  # Will cause error

        with patch.object(request, "json", side_effect=Exception("Parse error")):
            response = await handler.handle_relay_heartbeat(request)

        assert response.status == 400


# =============================================================================
# Relay Enqueue Tests
# =============================================================================


class TestHandleRelayEnqueue:
    """Tests for handle_relay_enqueue endpoint."""

    @pytest.fixture
    def handler(self):
        return RelayHandlersTestClass()

    @pytest.mark.asyncio
    async def test_enqueue_command(self, handler):
        """Command is enqueued for target node."""
        request = MockRequest(
            json_data={
                "target_node_id": "nat-1",
                "type": "start_job",
                "payload": {"job_type": "selfplay"},
            }
        )

        response = await handler.handle_relay_enqueue(request)

        body = json.loads(response.body)
        assert body["success"] is True
        assert "id" in body
        assert len(handler.relay_command_queue["nat-1"]) == 1

    @pytest.mark.asyncio
    async def test_missing_target_node_id(self, handler):
        """Missing target_node_id returns 400."""
        request = MockRequest(
            json_data={
                "type": "start_job",
            }
        )

        response = await handler.handle_relay_enqueue(request)

        assert response.status == 400
        body = json.loads(response.body)
        assert "target_node_id" in body["error"]

    @pytest.mark.asyncio
    async def test_missing_type(self, handler):
        """Missing type returns 400."""
        request = MockRequest(
            json_data={
                "target_node_id": "nat-1",
            }
        )

        response = await handler.handle_relay_enqueue(request)

        assert response.status == 400
        body = json.loads(response.body)
        assert "type" in body["error"]

    @pytest.mark.asyncio
    async def test_queue_full_returns_429(self, handler):
        """Full queue returns 429 Too Many Requests."""
        # Fill the queue
        handler.relay_command_queue["nat-1"] = [
            {"id": f"cmd{i}"} for i in range(100)
        ]

        request = MockRequest(
            json_data={
                "target_node_id": "nat-1",
                "type": "test",
            }
        )

        response = await handler.handle_relay_enqueue(request)

        assert response.status == 429

    @pytest.mark.asyncio
    async def test_payload_defaults_to_empty(self, handler):
        """Missing payload defaults to empty dict."""
        request = MockRequest(
            json_data={
                "target_node_id": "nat-1",
                "type": "cleanup",
            }
        )

        response = await handler.handle_relay_enqueue(request)

        body = json.loads(response.body)
        assert body["success"] is True

        cmd = handler.relay_command_queue["nat-1"][0]
        assert cmd["payload"] == {}


# =============================================================================
# Relay Peers Tests
# =============================================================================


class TestHandleRelayPeers:
    """Tests for handle_relay_peers endpoint."""

    @pytest.fixture
    def handler(self):
        h = RelayHandlersTestClass()
        # Add some peers
        h.peers["direct-1"] = MockNodeInfo(node_id="direct-1", nat_blocked=False)
        h.peers["nat-1"] = MockNodeInfo(node_id="nat-1", nat_blocked=True)
        h.peers["nat-2"] = MockNodeInfo(node_id="nat-2", nat_blocked=True)
        return h

    @pytest.mark.asyncio
    async def test_returns_all_peers(self, handler):
        """Returns all peers including NAT-blocked."""
        request = MockRequest()

        response = await handler.handle_relay_peers(request)

        body = json.loads(response.body)
        assert body["total_peers"] == 3

    @pytest.mark.asyncio
    async def test_separates_direct_and_nat_blocked(self, handler):
        """Reports direct and NAT-blocked counts."""
        request = MockRequest()

        response = await handler.handle_relay_peers(request)

        body = json.loads(response.body)
        assert body["direct_peers"] == 1
        assert body["nat_blocked_peers"] == 2

    @pytest.mark.asyncio
    async def test_includes_voter_config(self, handler):
        """Response includes voter configuration."""
        request = MockRequest()

        response = await handler.handle_relay_peers(request)

        body = json.loads(response.body)
        assert "voter_node_ids" in body
        assert body["voter_quorum_ok"] is True

    @pytest.mark.asyncio
    async def test_auth_required_when_configured(self, handler):
        """Auth token required when configured."""
        handler.auth_token = "secret"
        handler._auth_failed = True
        request = MockRequest()

        response = await handler.handle_relay_peers(request)

        assert response.status == 401


# =============================================================================
# Relay Status Tests
# =============================================================================


class TestHandleRelayStatus:
    """Tests for handle_relay_status endpoint."""

    @pytest.fixture
    def handler(self):
        h = RelayHandlersTestClass()
        h.peers["nat-1"] = MockNodeInfo(node_id="nat-1", nat_blocked=True)
        return h

    @pytest.mark.asyncio
    async def test_returns_queue_status(self, handler):
        """Returns pending command counts."""
        handler.relay_command_queue["nat-1"] = [
            {"id": "cmd1", "cmd": "test", "ts": time.time()},
            {"id": "cmd2", "cmd": "test", "ts": time.time()},
        ]
        request = MockRequest()

        response = await handler.handle_relay_status(request)

        body = json.loads(response.body)
        assert body["total_pending_commands"] == 2
        assert "nat-1" in body["nodes_with_pending"]

    @pytest.mark.asyncio
    async def test_shows_command_age(self, handler):
        """Shows age of pending commands."""
        handler.relay_command_queue["nat-1"] = [
            {"id": "cmd1", "cmd": "test", "ts": time.time() - 60},  # 60 secs old
        ]
        request = MockRequest()

        response = await handler.handle_relay_status(request)

        body = json.loads(response.body)
        queue = body["queues"]["nat-1"]
        assert queue["commands"][0]["age_secs"] >= 59

    @pytest.mark.asyncio
    async def test_marks_stale_commands(self, handler):
        """Commands older than 5 minutes are marked stale."""
        handler.relay_command_queue["nat-1"] = [
            {"id": "old", "cmd": "test", "ts": time.time() - 400},  # > 5 min
            {"id": "new", "cmd": "test", "ts": time.time() - 10},
        ]
        request = MockRequest()

        response = await handler.handle_relay_status(request)

        body = json.loads(response.body)
        cmds = body["queues"]["nat-1"]["commands"]

        old_cmd = next(c for c in cmds if c["id"] == "old")
        new_cmd = next(c for c in cmds if c["id"] == "new")

        assert old_cmd["stale"] is True
        assert new_cmd["stale"] is False

    @pytest.mark.asyncio
    async def test_lists_nat_blocked_nodes(self, handler):
        """Lists all NAT-blocked nodes."""
        request = MockRequest()

        response = await handler.handle_relay_status(request)

        body = json.loads(response.body)
        assert "nat-1" in body["nat_blocked_nodes"]

    @pytest.mark.asyncio
    async def test_empty_queues_not_shown(self, handler):
        """Empty queues are not included in response."""
        handler.relay_command_queue["nat-1"] = []
        request = MockRequest()

        response = await handler.handle_relay_status(request)

        body = json.loads(response.body)
        assert "nat-1" not in body["queues"]

    @pytest.mark.asyncio
    async def test_auth_required_when_configured(self, handler):
        """Auth required when token configured."""
        handler.auth_token = "secret"
        handler._auth_failed = True
        request = MockRequest()

        response = await handler.handle_relay_status(request)

        assert response.status == 401
