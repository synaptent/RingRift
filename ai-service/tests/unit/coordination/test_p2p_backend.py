"""Tests for p2p_backend.py - P2P Backend client for communication with P2P orchestrator.

This module tests:
- P2PNodeInfo dataclass
- P2PBackend class
- Helper functions for URL normalization and IP checking
- discover_p2p_leader_url function
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.p2p_backend import (
    P2P_DEFAULT_PORT,
    P2PBackend,
    P2PNodeInfo,
    _is_loopback,
    _is_tailscale_ip,
    _normalize_p2p_seed_url,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_aiohttp_response():
    """Create a mock aiohttp response."""

    def create_response(status: int, json_data: dict | list | None = None):
        mock_resp = AsyncMock()
        mock_resp.status = status
        mock_resp.json = AsyncMock(return_value=json_data or {})
        return mock_resp

    return create_response


@pytest.fixture
def mock_session(mock_aiohttp_response):
    """Create a mock aiohttp ClientSession."""

    def create_session(responses: dict[str, tuple[int, dict]]):
        """Create session with predefined responses for URLs.

        Args:
            responses: Dict mapping URL patterns to (status_code, json_response)

        Returns:
            Tuple of (session, async_get_session) where async_get_session
            is an async function to patch _get_session with.
        """
        session = AsyncMock()

        def get_response(url, **kwargs):
            # Accept and ignore extra kwargs like json=
            for pattern, (status, data) in responses.items():
                if pattern in url:
                    resp = mock_aiohttp_response(status, data)
                    cm = AsyncMock()
                    cm.__aenter__ = AsyncMock(return_value=resp)
                    # Must return False to not suppress exceptions
                    cm.__aexit__ = AsyncMock(return_value=False)
                    return cm
            # Default 404 response
            resp = mock_aiohttp_response(404, {"error": "not found"})
            cm = AsyncMock()
            cm.__aenter__ = AsyncMock(return_value=resp)
            cm.__aexit__ = AsyncMock(return_value=False)
            return cm

        session.get = MagicMock(side_effect=get_response)
        session.post = MagicMock(side_effect=get_response)
        session.closed = False
        session.close = AsyncMock()

        # Create async wrapper for _get_session
        async def async_get_session():
            return session

        # Return tuple: (session for direct use, async function for patching)
        return session, async_get_session

    return create_session


# =============================================================================
# P2PNodeInfo Tests
# =============================================================================


class TestP2PNodeInfo:
    """Tests for P2PNodeInfo dataclass."""

    def test_creation_minimal(self):
        """Test creating node info with required fields."""
        node = P2PNodeInfo(
            node_id="node-1",
            host="192.168.1.100",
            port=8770,
            role="follower",
            has_gpu=False,
            gpu_name="",
            memory_gb=16,
            capabilities=["selfplay"],
        )

        assert node.node_id == "node-1"
        assert node.host == "192.168.1.100"
        assert node.port == 8770
        assert node.is_alive is True
        assert node.is_healthy is True

    def test_creation_with_gpu(self):
        """Test creating GPU node info."""
        node = P2PNodeInfo(
            node_id="gpu-node-1",
            host="192.168.1.101",
            port=8770,
            role="leader",
            has_gpu=True,
            gpu_name="NVIDIA H100",
            memory_gb=96,
            capabilities=["selfplay", "training", "gpu_cmaes"],
            cpu_percent=45.5,
            memory_percent=30.2,
            selfplay_jobs=3,
            training_jobs=1,
        )

        assert node.has_gpu is True
        assert node.gpu_name == "NVIDIA H100"
        assert node.cpu_percent == 45.5
        assert node.selfplay_jobs == 3

    def test_to_worker_config(self):
        """Test converting to worker config dict."""
        node = P2PNodeInfo(
            node_id="worker-1",
            host="10.0.0.50",
            port=8770,
            role="follower",
            has_gpu=True,
            gpu_name="NVIDIA A10",
            memory_gb=24,
            capabilities=["selfplay", "training"],
        )

        config = node.to_worker_config()

        assert config["name"] == "worker-1"
        assert config["host"] == "10.0.0.50"
        assert config["role"] == "mixed"  # has_gpu=True
        assert config["gpu"] == "NVIDIA A10"
        assert config["memory_gb"] == 24

    def test_to_worker_config_cpu_only(self):
        """Test worker config for CPU-only node."""
        node = P2PNodeInfo(
            node_id="cpu-worker",
            host="10.0.0.51",
            port=8770,
            role="follower",
            has_gpu=False,
            gpu_name="",
            memory_gb=8,
            capabilities=["selfplay"],
        )

        config = node.to_worker_config()

        assert config["role"] == "selfplay"  # CPU-only
        assert config["max_parallel_jobs"] == 2  # CPU default


# =============================================================================
# P2PBackend Tests
# =============================================================================


class TestP2PBackend:
    """Tests for P2PBackend class."""

    def test_init(self):
        """Test backend initialization."""
        backend = P2PBackend("http://localhost:8770")

        assert backend.leader_url == "http://localhost:8770"
        assert backend.timeout == 30

    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is stripped from URL."""
        backend = P2PBackend("http://localhost:8770/")

        assert backend.leader_url == "http://localhost:8770"

    def test_init_with_auth_token(self):
        """Test initialization with auth token."""
        backend = P2PBackend("http://localhost:8770", auth_token="secret123")

        assert backend.auth_token == "secret123"

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_session):
        """Test async context manager."""
        session = mock_session({"/health": (200, {"status": "ok"})})

        with patch.object(P2PBackend, "_get_session", return_value=session):
            async with P2PBackend("http://localhost:8770") as backend:
                assert backend is not None

    @pytest.mark.asyncio
    async def test_get_session_creates_session(self):
        """Test session creation on first use."""
        backend = P2PBackend("http://localhost:8770")

        with patch("app.coordination.p2p_backend.aiohttp") as mock_aiohttp:
            mock_session = AsyncMock()
            mock_aiohttp.ClientSession.return_value = mock_session

            session = await backend._get_session()

            assert session is mock_session
            mock_aiohttp.ClientSession.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_session(self):
        """Test closing session."""
        backend = P2PBackend("http://localhost:8770")
        mock_session = AsyncMock()
        mock_session.closed = False
        backend._session = mock_session

        await backend.close()

        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cluster_status(self, mock_session):
        """Test getting cluster status."""
        status_data = {
            "leader_id": "node-1",
            "nodes": [{"node_id": "node-1", "is_healthy": True}],
        }
        _, async_get_session = mock_session({"/api/cluster/status": (200, status_data)})

        backend = P2PBackend("http://localhost:8770")

        with patch.object(backend, "_get_session", async_get_session):
            status = await backend.get_cluster_status()

        assert status["leader_id"] == "node-1"
        assert len(status["nodes"]) == 1

    @pytest.mark.asyncio
    async def test_get_cluster_status_error(self, mock_session):
        """Test cluster status error handling."""
        _, async_get_session = mock_session({"/api/cluster/status": (500, {"error": "internal"})})

        backend = P2PBackend("http://localhost:8770")

        with patch.object(backend, "_get_session", async_get_session):
            with pytest.raises(RuntimeError, match="Failed to get cluster status"):
                await backend.get_cluster_status()

    @pytest.mark.asyncio
    async def test_get_nodes(self, mock_session):
        """Test getting node list."""
        status_data = {
            "nodes": [
                {
                    "node_id": "node-1",
                    "host": "192.168.1.100",
                    "port": 8770,
                    "role": "leader",
                    "has_gpu": True,
                    "gpu_name": "H100",
                    "memory_gb": 80,
                    "capabilities": ["selfplay", "training"],
                    "is_alive": True,
                    "is_healthy": True,
                },
                {
                    "node_id": "node-2",
                    "host": "192.168.1.101",
                    "port": 8770,
                    "role": "follower",
                    "has_gpu": False,
                    "memory_gb": 16,
                    "capabilities": ["selfplay"],
                    "is_alive": True,
                    "is_healthy": False,
                },
            ]
        }
        _, async_get_session = mock_session({"/api/cluster/status": (200, status_data)})

        backend = P2PBackend("http://localhost:8770")

        with patch.object(backend, "_get_session", async_get_session):
            nodes = await backend.get_nodes()

        assert len(nodes) == 2
        assert nodes[0].node_id == "node-1"
        assert nodes[0].has_gpu is True
        assert nodes[1].is_healthy is False

    @pytest.mark.asyncio
    async def test_get_healthy_nodes(self, mock_session):
        """Test filtering to healthy nodes only."""
        status_data = {
            "nodes": [
                {"node_id": "healthy", "is_alive": True, "is_healthy": True},
                {"node_id": "unhealthy", "is_alive": True, "is_healthy": False},
                {"node_id": "dead", "is_alive": False, "is_healthy": True},
            ]
        }
        _, async_get_session = mock_session({"/api/cluster/status": (200, status_data)})

        backend = P2PBackend("http://localhost:8770")

        with patch.object(backend, "_get_session", async_get_session):
            nodes = await backend.get_healthy_nodes()

        assert len(nodes) == 1
        assert nodes[0].node_id == "healthy"

    @pytest.mark.asyncio
    async def test_get_gpu_nodes(self, mock_session):
        """Test filtering to GPU nodes only."""
        status_data = {
            "nodes": [
                {"node_id": "gpu-1", "has_gpu": True, "is_alive": True, "is_healthy": True},
                {"node_id": "cpu-1", "has_gpu": False, "is_alive": True, "is_healthy": True},
            ]
        }
        _, async_get_session = mock_session({"/api/cluster/status": (200, status_data)})

        backend = P2PBackend("http://localhost:8770")

        with patch.object(backend, "_get_session", async_get_session):
            nodes = await backend.get_gpu_nodes()

        assert len(nodes) == 1
        assert nodes[0].node_id == "gpu-1"

    @pytest.mark.asyncio
    async def test_start_canonical_selfplay(self, mock_session):
        """Test starting canonical selfplay."""
        result_data = {"success": True, "job_id": "job-123"}
        _, async_get_session = mock_session({"/pipeline/start": (200, result_data)})

        backend = P2PBackend("http://localhost:8770")

        with patch.object(backend, "_get_session", async_get_session):
            result = await backend.start_canonical_selfplay(
                board_type="square8",
                num_players=2,
                games_per_node=500,
            )

        assert result["success"] is True
        assert result["job_id"] == "job-123"

    @pytest.mark.asyncio
    async def test_start_canonical_selfplay_failure(self, mock_session):
        """Test handling selfplay start failure."""
        result_data = {"success": False, "error": "cluster busy"}
        _, async_get_session = mock_session({"/pipeline/start": (200, result_data)})

        backend = P2PBackend("http://localhost:8770")

        with patch.object(backend, "_get_session", async_get_session):
            with pytest.raises(RuntimeError, match="Failed to start canonical selfplay"):
                await backend.start_canonical_selfplay()

    @pytest.mark.asyncio
    async def test_start_training(self, mock_session):
        """Test starting training."""
        result_data = {"success": True, "job_id": "train-456"}
        _, async_get_session = mock_session({"/pipeline/start": (200, result_data)})

        backend = P2PBackend("http://localhost:8770")

        with patch.object(backend, "_get_session", async_get_session):
            result = await backend.start_training(
                board_type="hex8",
                num_players=4,
                epochs=50,
            )

        assert result["job_id"] == "train-456"

    @pytest.mark.asyncio
    async def test_get_pipeline_status(self, mock_session):
        """Test getting pipeline status."""
        status_data = {
            "current_job": {"job_id": "job-123", "status": "running"},
            "queue": [],
        }
        _, async_get_session = mock_session({"/pipeline/status": (200, status_data)})

        backend = P2PBackend("http://localhost:8770")

        with patch.object(backend, "_get_session", async_get_session):
            status = await backend.get_pipeline_status()

        assert status["current_job"]["status"] == "running"

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_session):
        """Test health check when healthy."""
        _, async_get_session = mock_session({"/health": (200, {"status": "ok"})})

        backend = P2PBackend("http://localhost:8770")

        with patch.object(backend, "_get_session", async_get_session):
            is_healthy = await backend.health_check()

        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, mock_session):
        """Test health check when unhealthy."""
        _, async_get_session = mock_session({"/health": (503, {"status": "unhealthy"})})

        backend = P2PBackend("http://localhost:8770")

        with patch.object(backend, "_get_session", async_get_session):
            is_healthy = await backend.health_check()

        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_cancel_job(self, mock_session):
        """Test canceling a job."""
        result_data = {"success": True, "message": "Job cancelled"}
        _, async_get_session = mock_session({"/jobs/job-123/cancel": (200, result_data)})

        backend = P2PBackend("http://localhost:8770")

        with patch.object(backend, "_get_session", async_get_session):
            result = await backend.cancel_job("job-123")

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_trigger_data_sync(self, mock_session):
        """Test triggering data sync."""
        result_data = {"success": True, "synced_nodes": 5}
        _, async_get_session = mock_session({"/sync/start": (200, result_data)})

        backend = P2PBackend("http://localhost:8770")

        with patch.object(backend, "_get_session", async_get_session):
            result = await backend.trigger_data_sync()

        assert result["synced_nodes"] == 5


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_normalize_p2p_seed_url_empty(self):
        """Test normalizing empty URL."""
        assert _normalize_p2p_seed_url("") == ""
        assert _normalize_p2p_seed_url(None) == ""  # type: ignore

    def test_normalize_p2p_seed_url_adds_scheme(self):
        """Test that http:// is added when missing."""
        result = _normalize_p2p_seed_url("192.168.1.100:8770")
        assert result == "http://192.168.1.100:8770"

    def test_normalize_p2p_seed_url_keeps_scheme(self):
        """Test that existing scheme is preserved."""
        result = _normalize_p2p_seed_url("https://secure.example.com:8770")
        assert result == "https://secure.example.com:8770"

    def test_normalize_p2p_seed_url_strips_trailing_slash(self):
        """Test trailing slash is stripped."""
        result = _normalize_p2p_seed_url("http://example.com:8770/")
        assert result == "http://example.com:8770"

    def test_is_loopback(self):
        """Test loopback detection."""
        assert _is_loopback("localhost") is True
        assert _is_loopback("127.0.0.1") is True
        assert _is_loopback("::1") is True
        assert _is_loopback("0.0.0.0") is True

        assert _is_loopback("192.168.1.1") is False
        assert _is_loopback("10.0.0.1") is False

    def test_is_loopback_handles_edge_cases(self):
        """Test loopback with edge cases."""
        assert _is_loopback("") is False
        assert _is_loopback(None) is False  # type: ignore
        assert _is_loopback("  localhost  ") is True  # Whitespace

    def test_is_tailscale_ip(self):
        """Test Tailscale IP detection."""
        # Tailscale IPs are in 100.64.0.0/10 range
        assert _is_tailscale_ip("100.64.0.1") is True
        assert _is_tailscale_ip("100.100.50.25") is True
        assert _is_tailscale_ip("100.127.255.255") is True

        # Non-Tailscale IPs
        assert _is_tailscale_ip("192.168.1.1") is False
        assert _is_tailscale_ip("10.0.0.1") is False
        assert _is_tailscale_ip("100.128.0.1") is False  # Just outside range

    def test_is_tailscale_ip_handles_edge_cases(self):
        """Test Tailscale detection edge cases."""
        assert _is_tailscale_ip("") is False
        assert _is_tailscale_ip(None) is False  # type: ignore
        assert _is_tailscale_ip("not.an.ip") is False
        assert _is_tailscale_ip("::1") is False  # IPv6


# =============================================================================
# discover_p2p_leader_url Tests
# =============================================================================


class TestDiscoverP2PLeaderUrl:
    """Tests for discover_p2p_leader_url function."""

    @pytest.mark.asyncio
    async def test_discover_empty_seeds(self):
        """Test discovery with no seeds returns None."""
        from app.coordination.p2p_backend import discover_p2p_leader_url

        result = await discover_p2p_leader_url([])

        assert result is None

    @pytest.mark.asyncio
    async def test_discover_leader_from_seed(self):
        """Test discovering leader from seed node."""
        from app.coordination.p2p_backend import discover_p2p_leader_url

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "node_id": "seed-node",
                "leader_id": "seed-node",
                "effective_leader_id": "seed-node",
                "self": {
                    "node_id": "seed-node",
                    "host": "192.168.1.100",
                    "port": 8770,
                },
            }
        )

        health_response = AsyncMock()
        health_response.status = 200

        mock_session = AsyncMock()

        def mock_get(url):
            cm = AsyncMock()
            if "/status" in url:
                cm.__aenter__ = AsyncMock(return_value=mock_response)
            else:
                cm.__aenter__ = AsyncMock(return_value=health_response)
            cm.__aexit__ = AsyncMock()
            return cm

        mock_session.get = MagicMock(side_effect=mock_get)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        with patch("app.coordination.p2p_backend.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession.return_value = mock_session
            mock_aiohttp.ClientTimeout = MagicMock()

            result = await discover_p2p_leader_url(["http://192.168.1.100:8770"])

        # Should return the seed URL since it claims to be leader
        assert result is not None
        assert "192.168.1.100" in result
