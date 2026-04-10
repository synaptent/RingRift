"""Integration tests for SWIM/Raft HTTP endpoints.

Tests the SWIM and Raft handler endpoints in both enabled and disabled modes.
These tests mock the underlying protocol implementations to verify endpoint behavior.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

# Import handlers
from scripts.p2p.handlers.swim import SwimHandlersMixin
from scripts.p2p.handlers.raft import RaftHandlersMixin

pytestmark = pytest.mark.timeout(30)


class MockOrchestrator(SwimHandlersMixin, RaftHandlersMixin):
    """Mock orchestrator class for testing handlers."""

    def __init__(
        self,
        *,
        swim_started: bool = False,
        swim_manager: MagicMock | None = None,
        raft_initialized: bool = False,
        raft_work_queue: MagicMock | None = None,
        raft_job_assignments: MagicMock | None = None,
    ):
        self.node_id = "test-node"
        self.auth_token = None
        self._swim_started = swim_started
        self._swim_manager = swim_manager
        self._raft_initialized = raft_initialized
        self._raft_work_queue = raft_work_queue
        self._raft_job_assignments = raft_job_assignments

    def _is_request_authorized(self, request: web.Request) -> bool:
        """Check if request is authorized (always True in tests without auth_token)."""
        if self.auth_token is None:
            return True
        auth_header = request.headers.get("Authorization", "")
        return auth_header == f"Bearer {self.auth_token}"

    def get_swim_membership_summary(self) -> dict:
        """Get SWIM membership summary for tests."""
        return {
            "swim_enabled": True,
            "swim_available": True,
            "swim_started": self._swim_started,
            "membership_mode": "hybrid" if self._swim_started else "http",
            "members": 10,
            "alive": 8,
            "suspected": 1,
            "failed": 1,
        }


class TestSwimStatusEndpoint(AioHTTPTestCase):
    """Tests for GET /swim/status endpoint."""

    async def get_application(self) -> web.Application:
        """Create test application."""
        self.mock_orchestrator = MockOrchestrator()
        app = web.Application()
        app.router.add_get("/swim/status", self.mock_orchestrator.handle_swim_status)
        return app

    async def test_swim_status_disabled(self):
        """Test /swim/status when SWIM is disabled."""
        async with self.client.get("/swim/status") as resp:
            assert resp.status == 200
            data = await resp.json()
            assert data["node_id"] == "test-node"
            assert "swim_enabled" in data
            assert "config" in data
            assert "timestamp" in data

    async def test_swim_status_with_manager(self):
        """Test /swim/status when SWIM manager is available."""
        self.mock_orchestrator._swim_started = True
        async with self.client.get("/swim/status") as resp:
            assert resp.status == 200
            data = await resp.json()
            assert data["swim_started"] is True
            assert data["members"] == 10
            assert data["alive"] == 8


class TestSwimMembersEndpoint(AioHTTPTestCase):
    """Tests for GET /swim/members endpoint."""

    async def get_application(self) -> web.Application:
        """Create test application."""
        self.mock_orchestrator = MockOrchestrator()
        app = web.Application()
        app.router.add_get("/swim/members", self.mock_orchestrator.handle_swim_members)
        return app

    async def test_swim_members_not_started(self):
        """Test /swim/members when SWIM not started."""
        async with self.client.get("/swim/members") as resp:
            assert resp.status == 200
            data = await resp.json()
            assert data["swim_started"] is False
            assert data["members"] == []
            assert "message" in data

    async def test_swim_members_started_no_members(self):
        """Test /swim/members when SWIM started but no internal access."""
        # Create a swim manager mock where _swim is explicitly None
        # so the handler falls back to get_alive_peers()
        swim_manager = MagicMock()
        swim_manager._swim = None  # Explicitly set to None to trigger fallback
        swim_manager.get_alive_peers.return_value = ["peer-1", "peer-2"]
        self.mock_orchestrator._swim_started = True
        self.mock_orchestrator._swim_manager = swim_manager

        async with self.client.get("/swim/members") as resp:
            assert resp.status == 200
            data = await resp.json()
            assert data["swim_started"] is True
            assert len(data["members"]) == 2
            assert data["alive_count"] == 2


class TestRaftStatusEndpoint(AioHTTPTestCase):
    """Tests for GET /raft/status endpoint."""

    async def get_application(self) -> web.Application:
        """Create test application."""
        self.mock_orchestrator = MockOrchestrator()
        app = web.Application()
        app.router.add_get("/raft/status", self.mock_orchestrator.handle_raft_status)
        return app

    async def test_raft_status_disabled(self):
        """Test /raft/status when Raft is disabled."""
        async with self.client.get("/raft/status") as resp:
            assert resp.status == 200
            data = await resp.json()
            assert data["node_id"] == "test-node"
            assert data["raft_initialized"] is False
            assert data["cluster_health"] == "disabled"

    async def test_raft_status_initialized(self):
        """Test /raft/status when Raft is initialized."""
        work_queue = MagicMock()
        work_queue.is_ready = True
        work_queue.is_leader = False
        work_queue.leader_address = "192.168.1.10:4321"

        job_assignments = MagicMock()
        job_assignments.is_ready = True
        job_assignments.is_leader = False
        job_assignments.leader_address = "192.168.1.10:4322"

        self.mock_orchestrator._raft_initialized = True
        self.mock_orchestrator._raft_work_queue = work_queue
        self.mock_orchestrator._raft_job_assignments = job_assignments

        with patch("scripts.p2p.handlers.raft.RAFT_ENABLED", True):
            async with self.client.get("/raft/status") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["raft_initialized"] is True
                assert data["work_queue"]["is_ready"] is True
                assert data["cluster_health"] == "healthy"


class TestRaftWorkQueueEndpoint(AioHTTPTestCase):
    """Tests for GET /raft/work endpoint."""

    async def get_application(self) -> web.Application:
        """Create test application."""
        self.mock_orchestrator = MockOrchestrator()
        app = web.Application()
        app.router.add_get("/raft/work", self.mock_orchestrator.handle_raft_work_queue)
        return app

    async def test_raft_work_queue_disabled(self):
        """Test /raft/work when Raft is disabled."""
        async with self.client.get("/raft/work") as resp:
            assert resp.status == 200
            data = await resp.json()
            assert data["enabled"] is False
            assert "message" in data

    async def test_raft_work_queue_enabled(self):
        """Test /raft/work when Raft is enabled."""
        work_queue = MagicMock()
        work_queue.is_ready = True
        work_queue.is_leader = True
        work_queue.leader_address = "localhost:4321"
        work_queue.get_queue_stats.return_value = {
            "pending": 10,
            "claimed": 3,
            "running": 2,
            "completed": 100,
            "failed": 5,
        }

        self.mock_orchestrator._raft_initialized = True
        self.mock_orchestrator._raft_work_queue = work_queue

        with patch("scripts.p2p.handlers.raft.RAFT_ENABLED", True):
            async with self.client.get("/raft/work") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["enabled"] is True
                assert data["is_leader"] is True
                assert data["stats"]["pending"] == 10


class TestRaftJobsEndpoint(AioHTTPTestCase):
    """Tests for GET /raft/jobs endpoint."""

    async def get_application(self) -> web.Application:
        """Create test application."""
        self.mock_orchestrator = MockOrchestrator()
        app = web.Application()
        app.router.add_get("/raft/jobs", self.mock_orchestrator.handle_raft_jobs)
        return app

    async def test_raft_jobs_disabled(self):
        """Test /raft/jobs when Raft is disabled."""
        async with self.client.get("/raft/jobs") as resp:
            assert resp.status == 200
            data = await resp.json()
            assert data["enabled"] is False
            assert "message" in data

    async def test_raft_jobs_enabled(self):
        """Test /raft/jobs when Raft is enabled."""
        job_assignments = MagicMock()
        job_assignments.is_ready = True
        job_assignments.is_leader = False
        job_assignments.leader_address = "192.168.1.10:4322"
        job_assignments.get_assignment_stats.return_value = {
            "assigned": 5,
            "running": 8,
            "completed": 50,
            "failed": 2,
        }

        self.mock_orchestrator._raft_initialized = True
        self.mock_orchestrator._raft_job_assignments = job_assignments

        with patch("scripts.p2p.handlers.raft.RAFT_ENABLED", True):
            async with self.client.get("/raft/jobs") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["enabled"] is True
                assert data["stats"]["assigned"] == 5


class TestRaftLockEndpoints(AioHTTPTestCase):
    """Tests for POST/DELETE /raft/lock/{name} endpoints."""

    async def get_application(self) -> web.Application:
        """Create test application."""
        self.mock_orchestrator = MockOrchestrator()
        app = web.Application()
        app.router.add_post("/raft/lock/{name}", self.mock_orchestrator.handle_raft_lock)
        app.router.add_delete("/raft/lock/{name}", self.mock_orchestrator.handle_raft_unlock)
        return app

    async def test_lock_raft_disabled(self):
        """Test acquiring lock when Raft is disabled."""
        async with self.client.post("/raft/lock/my-lock") as resp:
            assert resp.status == 503
            data = await resp.json()
            assert data["acquired"] is False
            assert "error" in data

    async def test_unlock_raft_disabled(self):
        """Test releasing lock when Raft is disabled."""
        async with self.client.delete("/raft/lock/my-lock") as resp:
            assert resp.status == 503
            data = await resp.json()
            assert data["released"] is False
            assert "error" in data

    async def test_lock_missing_name(self):
        """Test acquiring lock without name (should be handled by router)."""
        # This is implicitly tested by router - can't reach handler without name
        pass

    async def test_lock_with_auth(self):
        """Test lock endpoint requires auth when auth_token is set."""
        self.mock_orchestrator.auth_token = "secret-token"
        async with self.client.post("/raft/lock/my-lock") as resp:
            assert resp.status == 401
            data = await resp.json()
            assert data["error"] == "unauthorized"


class TestSwimStatusEndpointError(AioHTTPTestCase):
    """Tests for error handling in SWIM endpoints."""

    async def get_application(self) -> web.Application:
        """Create test application."""
        self.mock_orchestrator = MockOrchestrator()
        # Make get_swim_membership_summary raise an exception
        self.mock_orchestrator.get_swim_membership_summary = MagicMock(
            side_effect=RuntimeError("Test error")
        )
        app = web.Application()
        app.router.add_get("/swim/status", self.mock_orchestrator.handle_swim_status)
        return app

    async def test_swim_status_error_handling(self):
        """Test /swim/status handles exceptions gracefully."""
        async with self.client.get("/swim/status") as resp:
            assert resp.status == 500
            data = await resp.json()
            assert "error" in data
            assert "Test error" in data["error"]


class TestRaftStatusEndpointError(AioHTTPTestCase):
    """Tests for error handling in Raft endpoints."""

    async def get_application(self) -> web.Application:
        """Create test application."""
        # Create work queue that raises on property access
        work_queue = MagicMock()
        type(work_queue).is_ready = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Work queue error"))
        )

        self.mock_orchestrator = MockOrchestrator(
            raft_initialized=True,
            raft_work_queue=work_queue,
        )
        app = web.Application()
        app.router.add_get("/raft/status", self.mock_orchestrator.handle_raft_status)
        return app

    async def test_raft_status_work_queue_error(self):
        """Test /raft/status handles work queue errors."""
        with patch("scripts.p2p.handlers.raft.RAFT_ENABLED", True):
            async with self.client.get("/raft/status") as resp:
                # Should still return 200 with error in work_queue field
                assert resp.status == 200
                data = await resp.json()
                assert "error" in data["work_queue"]


# Unit tests (non-async, simpler assertions)
class TestSwimHandlersMixin:
    """Unit tests for SwimHandlersMixin methods."""

    def test_mixin_has_handlers(self):
        """Test mixin has expected handler methods."""
        assert hasattr(SwimHandlersMixin, "handle_swim_status")
        assert hasattr(SwimHandlersMixin, "handle_swim_members")

    def test_handlers_are_async(self):
        """Test handlers are coroutines."""
        import asyncio
        assert asyncio.iscoroutinefunction(SwimHandlersMixin.handle_swim_status)
        assert asyncio.iscoroutinefunction(SwimHandlersMixin.handle_swim_members)


class TestRaftHandlersMixin:
    """Unit tests for RaftHandlersMixin methods."""

    def test_mixin_has_handlers(self):
        """Test mixin has expected handler methods."""
        assert hasattr(RaftHandlersMixin, "handle_raft_status")
        assert hasattr(RaftHandlersMixin, "handle_raft_work_queue")
        assert hasattr(RaftHandlersMixin, "handle_raft_jobs")
        assert hasattr(RaftHandlersMixin, "handle_raft_lock")
        assert hasattr(RaftHandlersMixin, "handle_raft_unlock")

    def test_handlers_are_async(self):
        """Test handlers are coroutines."""
        import asyncio
        assert asyncio.iscoroutinefunction(RaftHandlersMixin.handle_raft_status)
        assert asyncio.iscoroutinefunction(RaftHandlersMixin.handle_raft_work_queue)
        assert asyncio.iscoroutinefunction(RaftHandlersMixin.handle_raft_jobs)
        assert asyncio.iscoroutinefunction(RaftHandlersMixin.handle_raft_lock)
        assert asyncio.iscoroutinefunction(RaftHandlersMixin.handle_raft_unlock)
