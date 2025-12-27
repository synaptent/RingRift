"""Tests for p2p_integration.py (December 2025).

This module tests the P2P integration facade that provides a unified
interface for coordination modules to interact with the P2P cluster.

Coverage targets:
- P2PNodeStatus dataclass
- P2PJobResult dataclass
- P2POrchestratorShim compatibility class
- get_p2p_status() function
- get_p2p_nodes() function
- submit_p2p_job() function
- is_p2p_available() function
- Cache management functions
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


class TestP2PNodeStatus:
    """Tests for P2PNodeStatus dataclass."""

    def test_default_values(self):
        """Test default values for P2PNodeStatus."""
        from app.coordination.p2p_integration import P2PNodeStatus

        status = P2PNodeStatus(node_id="test-node", host="192.168.1.100")

        assert status.node_id == "test-node"
        assert status.host == "192.168.1.100"
        assert status.port == 8770
        assert status.is_alive is True
        assert status.is_healthy is True
        assert status.gpu_utilization == 0.0
        assert status.active_jobs == 0
        assert status.provider == "unknown"
        assert status.has_gpu is False

    def test_from_dict_basic(self):
        """Test creating P2PNodeStatus from dict."""
        from app.coordination.p2p_integration import P2PNodeStatus

        data = {
            "node_id": "runpod-h100",
            "host": "102.210.171.65",
            "port": 8770,
            "is_alive": True,
            "is_healthy": True,
            "has_gpu": True,
            "gpu_name": "H100",
            "provider": "runpod",
        }

        status = P2PNodeStatus.from_dict(data)

        assert status.node_id == "runpod-h100"
        assert status.host == "102.210.171.65"
        assert status.has_gpu is True
        assert status.gpu_name == "H100"
        assert status.provider == "runpod"

    def test_from_dict_with_gpu_memory(self):
        """Test from_dict handles different GPU memory field names."""
        from app.coordination.p2p_integration import P2PNodeStatus

        # Test with gpu_memory_total (P2P format)
        data1 = {"node_id": "n1", "host": "h1", "gpu_memory_total": 80.0}
        status1 = P2PNodeStatus.from_dict(data1)
        assert status1.gpu_memory_total_gb == 80.0

        # Test with gpu_memory_total_gb (alternative format)
        data2 = {"node_id": "n2", "host": "h2", "gpu_memory_total_gb": 96.0}
        status2 = P2PNodeStatus.from_dict(data2)
        assert status2.gpu_memory_total_gb == 96.0

    def test_from_dict_empty(self):
        """Test from_dict handles empty dict."""
        from app.coordination.p2p_integration import P2PNodeStatus

        status = P2PNodeStatus.from_dict({})

        assert status.node_id == ""
        assert status.host == ""
        assert status.is_alive is True  # Default


class TestP2PJobResult:
    """Tests for P2PJobResult dataclass."""

    def test_success_result(self):
        """Test successful job result."""
        from app.coordination.p2p_integration import P2PJobResult

        result = P2PJobResult(
            success=True,
            job_id="job-123",
            details={"started_at": time.time()},
        )

        assert result.success is True
        assert result.job_id == "job-123"
        assert result.error == ""
        assert result.details is not None

    def test_failure_result(self):
        """Test failed job result."""
        from app.coordination.p2p_integration import P2PJobResult

        result = P2PJobResult(
            success=False,
            error="No available nodes",
        )

        assert result.success is False
        assert result.error == "No available nodes"
        assert result.job_id == ""


class TestP2POrchestratorShim:
    """Tests for P2POrchestratorShim compatibility class."""

    def test_shim_initialization(self):
        """Test shim can be created."""
        from app.coordination.p2p_integration import P2POrchestratorShim

        shim = P2POrchestratorShim()
        assert shim is not None

    def test_get_p2p_orchestrator_returns_shim(self):
        """Test get_p2p_orchestrator returns shim instance."""
        from app.coordination.p2p_integration import get_p2p_orchestrator

        # Should return shim or None depending on availability
        result = get_p2p_orchestrator()
        if result is not None:
            from app.coordination.p2p_integration import P2POrchestratorShim
            assert isinstance(result, P2POrchestratorShim)

    @pytest.mark.asyncio
    async def test_shim_get_status(self):
        """Test shim get_status method."""
        from app.coordination.p2p_integration import P2POrchestratorShim

        shim = P2POrchestratorShim()

        # Mock the underlying get_p2p_status
        with patch("app.coordination.p2p_integration.get_p2p_status") as mock_status:
            mock_status.return_value = {"leader_id": "node-1", "alive_peers": 5}

            result = await shim.get_status()

            assert result == {"leader_id": "node-1", "alive_peers": 5}
            mock_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_shim_submit_job(self):
        """Test shim submit_job method."""
        from app.coordination.p2p_integration import (
            P2POrchestratorShim,
            P2PJobResult,
        )

        shim = P2POrchestratorShim()

        # Mock submit_p2p_job
        with patch("app.coordination.p2p_integration.submit_p2p_job") as mock_submit:
            mock_submit.return_value = P2PJobResult(
                success=True,
                job_id="job-456",
            )

            job_spec = {"type": "selfplay", "board_type": "hex8"}
            result = await shim.submit_job(job_spec)

            assert result["success"] is True
            assert result["job_id"] == "job-456"
            mock_submit.assert_called_once_with(job_spec)


class TestGetP2PStatus:
    """Tests for get_p2p_status function."""

    @pytest.mark.asyncio
    async def test_returns_none_without_backend(self):
        """Test returns None when backend unavailable."""
        from app.coordination.p2p_integration import get_p2p_status, clear_p2p_cache

        clear_p2p_cache()

        with patch("app.coordination.p2p_integration._get_backend") as mock_backend:
            mock_backend.return_value = None

            result = await get_p2p_status()

            assert result is None

    @pytest.mark.asyncio
    async def test_uses_cache(self):
        """Test that cache is used when valid."""
        from app.coordination.p2p_integration import (
            get_p2p_status,
            clear_p2p_cache,
        )
        import app.coordination.p2p_integration as module

        clear_p2p_cache()

        # Set up cache manually
        module._status_cache = {"leader_id": "cached-leader"}
        module._status_cache_time = time.time()

        result = await get_p2p_status(use_cache=True, cache_ttl=60.0)

        assert result == {"leader_id": "cached-leader"}

        clear_p2p_cache()

    @pytest.mark.asyncio
    async def test_bypasses_cache_when_disabled(self):
        """Test that cache is bypassed when use_cache=False."""
        from app.coordination.p2p_integration import (
            get_p2p_status,
            clear_p2p_cache,
        )
        import app.coordination.p2p_integration as module

        clear_p2p_cache()

        # Set up stale cache
        module._status_cache = {"leader_id": "stale"}
        module._status_cache_time = time.time()

        with patch("app.coordination.p2p_integration._get_backend") as mock_backend:
            mock_backend.return_value = None

            result = await get_p2p_status(use_cache=False)

            # Should try backend even though cache exists
            mock_backend.assert_called_once()

        clear_p2p_cache()


class TestGetP2PNodes:
    """Tests for get_p2p_nodes function."""

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_status(self):
        """Test returns empty list when status unavailable."""
        from app.coordination.p2p_integration import get_p2p_nodes

        with patch("app.coordination.p2p_integration.get_p2p_status") as mock_status:
            mock_status.return_value = None

            result = await get_p2p_nodes()

            assert result == []

    @pytest.mark.asyncio
    async def test_parses_alive_peers_dicts(self):
        """Test parsing alive_peers with dict entries."""
        from app.coordination.p2p_integration import get_p2p_nodes

        with patch("app.coordination.p2p_integration.get_p2p_status") as mock_status:
            mock_status.return_value = {
                "alive_peers": [
                    {"node_id": "node-1", "host": "192.168.1.1", "has_gpu": True},
                    {"node_id": "node-2", "host": "192.168.1.2", "has_gpu": False},
                ]
            }

            result = await get_p2p_nodes()

            assert len(result) == 2
            assert result[0].node_id == "node-1"
            assert result[0].has_gpu is True
            assert result[1].node_id == "node-2"

    @pytest.mark.asyncio
    async def test_parses_alive_peers_strings(self):
        """Test parsing alive_peers with string entries."""
        from app.coordination.p2p_integration import get_p2p_nodes

        with patch("app.coordination.p2p_integration.get_p2p_status") as mock_status:
            mock_status.return_value = {
                "alive_peers": ["node-1", "node-2", "node-3"]
            }

            result = await get_p2p_nodes()

            assert len(result) == 3
            assert result[0].node_id == "node-1"
            assert result[0].host == ""


class TestGetP2PFilteredNodes:
    """Tests for filtered node functions."""

    @pytest.mark.asyncio
    async def test_get_alive_nodes(self):
        """Test get_p2p_alive_nodes filters correctly."""
        from app.coordination.p2p_integration import get_p2p_alive_nodes

        with patch("app.coordination.p2p_integration.get_p2p_nodes") as mock_nodes:
            from app.coordination.p2p_integration import P2PNodeStatus

            mock_nodes.return_value = [
                P2PNodeStatus(node_id="alive", host="h1", is_alive=True),
                P2PNodeStatus(node_id="dead", host="h2", is_alive=False),
            ]

            result = await get_p2p_alive_nodes()

            assert len(result) == 1
            assert result[0].node_id == "alive"

    @pytest.mark.asyncio
    async def test_get_healthy_nodes(self):
        """Test get_p2p_healthy_nodes filters correctly."""
        from app.coordination.p2p_integration import get_p2p_healthy_nodes

        with patch("app.coordination.p2p_integration.get_p2p_nodes") as mock_nodes:
            from app.coordination.p2p_integration import P2PNodeStatus

            mock_nodes.return_value = [
                P2PNodeStatus(node_id="healthy", host="h1", is_alive=True, is_healthy=True),
                P2PNodeStatus(node_id="unhealthy", host="h2", is_alive=True, is_healthy=False),
                P2PNodeStatus(node_id="dead", host="h3", is_alive=False, is_healthy=True),
            ]

            result = await get_p2p_healthy_nodes()

            assert len(result) == 1
            assert result[0].node_id == "healthy"

    @pytest.mark.asyncio
    async def test_get_gpu_nodes(self):
        """Test get_p2p_gpu_nodes filters correctly."""
        from app.coordination.p2p_integration import get_p2p_gpu_nodes

        with patch("app.coordination.p2p_integration.get_p2p_healthy_nodes") as mock_nodes:
            from app.coordination.p2p_integration import P2PNodeStatus

            mock_nodes.return_value = [
                P2PNodeStatus(node_id="gpu", host="h1", has_gpu=True),
                P2PNodeStatus(node_id="cpu", host="h2", has_gpu=False),
            ]

            result = await get_p2p_gpu_nodes()

            assert len(result) == 1
            assert result[0].node_id == "gpu"


class TestSubmitP2PJob:
    """Tests for submit_p2p_job function."""

    @pytest.mark.asyncio
    async def test_returns_error_without_aiohttp(self):
        """Test returns error when aiohttp unavailable."""
        from app.coordination.p2p_integration import submit_p2p_job

        with patch("app.coordination.p2p_integration.HAS_AIOHTTP", False):
            result = await submit_p2p_job({"type": "selfplay"})

            assert result.success is False
            assert "aiohttp" in result.error

    @pytest.mark.asyncio
    async def test_returns_error_without_backend(self):
        """Test returns error when backend unavailable."""
        from app.coordination.p2p_integration import submit_p2p_job

        with patch("app.coordination.p2p_integration._get_backend") as mock_backend:
            mock_backend.return_value = None

            result = await submit_p2p_job({"type": "selfplay"})

            assert result.success is False
            assert "backend" in result.error.lower()


class TestIsP2PAvailable:
    """Tests for is_p2p_available function."""

    @pytest.mark.asyncio
    async def test_returns_false_without_aiohttp(self):
        """Test returns False when aiohttp unavailable."""
        from app.coordination.p2p_integration import is_p2p_available

        with patch("app.coordination.p2p_integration.HAS_AIOHTTP", False):
            result = await is_p2p_available()
            assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_without_backend(self):
        """Test returns False when backend unavailable."""
        from app.coordination.p2p_integration import is_p2p_available

        with patch("app.coordination.p2p_integration._get_backend") as mock_backend:
            mock_backend.return_value = None

            result = await is_p2p_available()
            assert result is False


class TestCacheManagement:
    """Tests for cache management functions."""

    def test_clear_cache(self):
        """Test clear_p2p_cache clears cache."""
        import app.coordination.p2p_integration as module
        from app.coordination.p2p_integration import clear_p2p_cache

        module._status_cache = {"test": "data"}
        module._status_cache_time = time.time()

        clear_p2p_cache()

        assert module._status_cache == {}
        assert module._status_cache_time == 0.0

    @pytest.mark.asyncio
    async def test_close_connection(self):
        """Test close_p2p_connection cleans up."""
        import app.coordination.p2p_integration as module
        from app.coordination.p2p_integration import close_p2p_connection

        # Mock backend instance
        mock_backend = MagicMock()
        mock_backend.close = AsyncMock()
        module._backend_instance = mock_backend

        await close_p2p_connection()

        mock_backend.close.assert_called_once()
        assert module._backend_instance is None


class TestGetP2PLeader:
    """Tests for leader-related functions."""

    @pytest.mark.asyncio
    async def test_get_leader_id(self):
        """Test get_p2p_leader_id returns leader ID."""
        from app.coordination.p2p_integration import get_p2p_leader_id

        with patch("app.coordination.p2p_integration.get_p2p_status") as mock_status:
            mock_status.return_value = {"leader_id": "leader-node"}

            result = await get_p2p_leader_id()

            assert result == "leader-node"

    @pytest.mark.asyncio
    async def test_get_leader_id_fallback(self):
        """Test get_p2p_leader_id uses effective_leader_id as fallback."""
        from app.coordination.p2p_integration import get_p2p_leader_id

        with patch("app.coordination.p2p_integration.get_p2p_status") as mock_status:
            mock_status.return_value = {"effective_leader_id": "effective-leader"}

            result = await get_p2p_leader_id()

            assert result == "effective-leader"

    @pytest.mark.asyncio
    async def test_get_leader_id_none(self):
        """Test get_p2p_leader_id returns None when unavailable."""
        from app.coordination.p2p_integration import get_p2p_leader_id

        with patch("app.coordination.p2p_integration.get_p2p_status") as mock_status:
            mock_status.return_value = None

            result = await get_p2p_leader_id()

            assert result is None


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_importable(self):
        """Test that all __all__ exports are importable."""
        from app.coordination import p2p_integration

        for name in p2p_integration.__all__:
            assert hasattr(p2p_integration, name), f"Missing export: {name}"

    def test_key_functions_exported(self):
        """Test that key functions are exported."""
        from app.coordination.p2p_integration import (
            get_p2p_status,
            get_p2p_nodes,
            get_p2p_alive_nodes,
            get_p2p_healthy_nodes,
            get_p2p_gpu_nodes,
            submit_p2p_job,
            is_p2p_available,
            get_p2p_orchestrator,
            clear_p2p_cache,
            close_p2p_connection,
        )

        # All should be callable
        assert callable(get_p2p_status)
        assert callable(get_p2p_nodes)
        assert callable(submit_p2p_job)
        assert callable(is_p2p_available)
        assert callable(get_p2p_orchestrator)


class TestIntegrationWithDaemons:
    """Integration tests verifying daemon compatibility."""

    def test_idle_resource_daemon_can_import(self):
        """Test idle_resource_daemon can import from p2p_integration."""
        # This tests the fix for the broken import
        from app.coordination.idle_resource_daemon import IdleResourceDaemon

        assert IdleResourceDaemon is not None

    def test_node_recovery_daemon_can_import(self):
        """Test node_recovery_daemon can import from p2p_integration."""
        # This tests the fix for the broken import
        from app.coordination.node_recovery_daemon import NodeRecoveryDaemon

        assert NodeRecoveryDaemon is not None
