"""Unit tests for HybridTransport - Multi-Protocol Fallback Transport Layer.

Tests cover:
- HybridTransport initialization
- Transport selection logic (HTTP -> Tailscale -> Cloudflare -> SSH)
- send_request() with mocked transports
- Failover behavior when primary transport fails
- Performance estimation
- Timeout handling
- NodeTransportState tracking
- Transport statistics
- File download with aria2 fallback
- Transport probing
- Cloudflare Zero Trust configuration
- aria2 configuration
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import module under test
from app.distributed.hybrid_transport import (
    HTTP_FAILURES_BEFORE_SSH,
    SSH_SUCCESS_BEFORE_HTTP_RETRY,
    TRANSPORT_HEALTH_CHECK_INTERVAL,
    HybridTransport,
    NodeTransportState,
    TransportType,
    diagnose_node_connectivity,
    get_hybrid_transport,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def transport() -> HybridTransport:
    """Create a fresh HybridTransport instance for testing."""
    return HybridTransport()


@pytest.fixture
def node_state() -> NodeTransportState:
    """Create a fresh NodeTransportState for testing."""
    return NodeTransportState(node_id="test-node")


@pytest.fixture
def mock_aiohttp_success():
    """Mock aiohttp to return successful responses."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"status": "ok"})

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.request = MagicMock(return_value=mock_response)
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    return mock_session


@pytest.fixture
def mock_aiohttp_failure():
    """Mock aiohttp to simulate connection failures."""
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(side_effect=ConnectionError("Connection refused"))
    mock_session.__aexit__ = AsyncMock(return_value=None)

    return mock_session


# =============================================================================
# TransportType Enum Tests
# =============================================================================


class TestTransportType:
    """Tests for TransportType enum."""

    def test_transport_type_values(self):
        """Test that all transport types have expected values."""
        assert TransportType.HTTP.value == "http"
        assert TransportType.TAILSCALE.value == "tailscale"
        assert TransportType.CLOUDFLARE.value == "cloudflare"
        assert TransportType.ARIA2.value == "aria2"
        assert TransportType.SSH.value == "ssh"

    def test_transport_type_is_string_enum(self):
        """Test that TransportType values are strings."""
        for t in TransportType:
            assert isinstance(t.value, str)


# =============================================================================
# NodeTransportState Tests
# =============================================================================


class TestNodeTransportState:
    """Tests for NodeTransportState tracking."""

    def test_initial_state(self, node_state: NodeTransportState):
        """Test initial state values."""
        assert node_state.node_id == "test-node"
        assert node_state.preferred_transport == TransportType.HTTP
        assert node_state.http_consecutive_failures == 0
        assert node_state.tailscale_consecutive_failures == 0
        assert node_state.cloudflare_consecutive_failures == 0
        assert node_state.ssh_consecutive_successes == 0
        assert not node_state.ssh_available
        assert not node_state.aria2_available

    def test_record_http_success_resets_failures(self, node_state: NodeTransportState):
        """Test that HTTP success resets failure counter."""
        node_state.http_consecutive_failures = 5
        node_state.preferred_transport = TransportType.SSH

        node_state.record_http_success()

        assert node_state.http_consecutive_failures == 0
        assert node_state.preferred_transport == TransportType.HTTP
        assert node_state.http_last_success > 0

    def test_record_http_failure_increments_counter(self, node_state: NodeTransportState):
        """Test that HTTP failure increments counter."""
        node_state.record_http_failure()
        assert node_state.http_consecutive_failures == 1

        node_state.record_http_failure()
        assert node_state.http_consecutive_failures == 2

    def test_http_failure_triggers_ssh_switch(self, node_state: NodeTransportState):
        """Test that enough HTTP failures trigger switch to SSH."""
        node_state.ssh_available = True

        # Record failures up to threshold
        for _ in range(HTTP_FAILURES_BEFORE_SSH):
            node_state.record_http_failure()

        assert node_state.preferred_transport == TransportType.SSH

    def test_http_failure_no_switch_without_ssh(self, node_state: NodeTransportState):
        """Test that HTTP failures don't switch to SSH if unavailable."""
        node_state.ssh_available = False

        for _ in range(HTTP_FAILURES_BEFORE_SSH + 5):
            node_state.record_http_failure()

        # Should still prefer HTTP since SSH not available
        assert node_state.preferred_transport == TransportType.HTTP

    def test_record_tailscale_success(self, node_state: NodeTransportState):
        """Test Tailscale success tracking."""
        node_state.tailscale_consecutive_failures = 3
        node_state.record_tailscale_success()

        assert node_state.tailscale_consecutive_failures == 0
        assert node_state.tailscale_last_success > 0

    def test_record_tailscale_failure(self, node_state: NodeTransportState):
        """Test Tailscale failure tracking."""
        node_state.record_tailscale_failure()
        assert node_state.tailscale_consecutive_failures == 1

    def test_record_cloudflare_success(self, node_state: NodeTransportState):
        """Test Cloudflare success tracking."""
        node_state.cloudflare_consecutive_failures = 2
        node_state.record_cloudflare_success()

        assert node_state.cloudflare_consecutive_failures == 0
        assert node_state.cloudflare_last_success > 0

    def test_record_cloudflare_failure(self, node_state: NodeTransportState):
        """Test Cloudflare failure tracking."""
        node_state.record_cloudflare_failure()
        assert node_state.cloudflare_consecutive_failures == 1

    def test_record_aria2_success(self, node_state: NodeTransportState):
        """Test aria2 success tracking."""
        assert not node_state.aria2_available
        node_state.record_aria2_success()

        assert node_state.aria2_available
        assert node_state.aria2_last_success > 0

    def test_record_ssh_success(self, node_state: NodeTransportState):
        """Test SSH success tracking."""
        node_state.record_ssh_success()

        assert node_state.ssh_consecutive_successes == 1
        assert node_state.ssh_available
        assert node_state.ssh_last_success > 0

    def test_ssh_success_resets_http_failures_after_threshold(
        self, node_state: NodeTransportState
    ):
        """Test that SSH successes eventually reset HTTP failure counter."""
        node_state.http_consecutive_failures = HTTP_FAILURES_BEFORE_SSH

        for _ in range(SSH_SUCCESS_BEFORE_HTTP_RETRY):
            node_state.record_ssh_success()

        assert node_state.http_consecutive_failures == 0
        assert node_state.ssh_consecutive_successes == 0

    def test_record_ssh_failure(self, node_state: NodeTransportState):
        """Test SSH failure resets success counter."""
        node_state.ssh_consecutive_successes = 3
        node_state.record_ssh_failure()

        assert node_state.ssh_consecutive_successes == 0

    def test_should_try_http_initially(self, node_state: NodeTransportState):
        """Test that HTTP should be tried initially."""
        assert node_state.should_try_http()

    def test_should_try_http_after_failures_below_threshold(
        self, node_state: NodeTransportState
    ):
        """Test HTTP should still be tried if failures below threshold."""
        node_state.http_consecutive_failures = HTTP_FAILURES_BEFORE_SSH - 1
        assert node_state.should_try_http()

    def test_should_try_ssh_when_available(self, node_state: NodeTransportState):
        """Test SSH should be tried when available."""
        node_state.ssh_available = True
        assert node_state.should_try_ssh()

    def test_should_not_try_ssh_when_unavailable(self, node_state: NodeTransportState):
        """Test SSH should not be tried when unavailable."""
        assert not node_state.should_try_ssh()

    def test_should_try_cloudflare_with_tunnel_configured(
        self, node_state: NodeTransportState
    ):
        """Test Cloudflare should be tried with tunnel configured."""
        node_state.cloudflare_tunnel = "node.tunnel.example.com"
        assert node_state.should_try_cloudflare()

    def test_should_not_try_cloudflare_without_tunnel(
        self, node_state: NodeTransportState
    ):
        """Test Cloudflare should not be tried without tunnel."""
        assert not node_state.should_try_cloudflare()

    def test_should_not_try_cloudflare_after_failures(
        self, node_state: NodeTransportState
    ):
        """Test Cloudflare should not be tried after too many failures."""
        node_state.cloudflare_tunnel = "node.tunnel.example.com"
        node_state.cloudflare_consecutive_failures = 3
        assert not node_state.should_try_cloudflare()

    def test_should_try_aria2_when_available(self, node_state: NodeTransportState):
        """Test aria2 should be tried when available."""
        node_state.aria2_available = True
        assert node_state.should_try_aria2()

    def test_should_not_try_aria2_when_unavailable(self, node_state: NodeTransportState):
        """Test aria2 should not be tried when unavailable."""
        assert not node_state.should_try_aria2()


# =============================================================================
# HybridTransport Initialization Tests
# =============================================================================


class TestHybridTransportInit:
    """Tests for HybridTransport initialization."""

    def test_init_creates_empty_states(self, transport: HybridTransport):
        """Test that initialization creates empty state dict."""
        assert transport._states == {}

    def test_init_creates_lock(self, transport: HybridTransport):
        """Test that initialization creates asyncio lock."""
        assert isinstance(transport._lock, asyncio.Lock)

    def test_init_no_ssh_transport(self, transport: HybridTransport):
        """Test that SSH transport is not loaded initially."""
        assert transport._ssh_transport is None

    def test_init_no_http_session(self, transport: HybridTransport):
        """Test that HTTP session is not created initially."""
        assert transport._http_session is None


# =============================================================================
# HybridTransport State Management Tests
# =============================================================================


class TestHybridTransportStateManagement:
    """Tests for HybridTransport state management."""

    def test_get_state_creates_new_state(self, transport: HybridTransport):
        """Test that _get_state creates new state for unknown node."""
        state = transport._get_state("new-node")

        assert state.node_id == "new-node"
        assert "new-node" in transport._states

    def test_get_state_returns_existing_state(self, transport: HybridTransport):
        """Test that _get_state returns existing state."""
        state1 = transport._get_state("test-node")
        state1.http_consecutive_failures = 5

        state2 = transport._get_state("test-node")

        assert state1 is state2
        assert state2.http_consecutive_failures == 5

    def test_configure_cloudflare(self, transport: HybridTransport):
        """Test Cloudflare tunnel configuration."""
        transport.configure_cloudflare("test-node", "tunnel.example.com")

        state = transport._get_state("test-node")
        assert state.cloudflare_tunnel == "tunnel.example.com"
        assert state.cloudflare_consecutive_failures == 0

    def test_configure_aria2_enabled(self, transport: HybridTransport):
        """Test aria2 availability configuration (enabled)."""
        transport.configure_aria2("test-node", available=True)

        state = transport._get_state("test-node")
        assert state.aria2_available

    def test_configure_aria2_disabled(self, transport: HybridTransport):
        """Test aria2 availability configuration (disabled)."""
        transport.configure_aria2("test-node", available=False)

        state = transport._get_state("test-node")
        assert not state.aria2_available


# =============================================================================
# HybridTransport HTTP Transport Tests
# =============================================================================


class TestHybridTransportHttp:
    """Tests for HybridTransport HTTP communication."""

    @pytest.mark.asyncio
    async def test_try_http_success(self, transport: HybridTransport):
        """Test successful HTTP request."""
        # Patch at aiohttp module level since import is inside the function
        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "ok"})
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_session = AsyncMock()
            mock_session.request = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_cls.return_value = mock_session

            success, response = await transport._try_http(
                "test-node", "http://localhost:8080/test", "GET"
            )

            assert success
            assert response == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_try_http_failure_status(self, transport: HybridTransport):
        """Test HTTP request with non-200 status."""
        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Internal Server Error")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_session = AsyncMock()
            mock_session.request = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_cls.return_value = mock_session

            success, response = await transport._try_http(
                "test-node", "http://localhost:8080/test", "GET"
            )

            assert not success
            assert response["status"] == 500

    @pytest.mark.asyncio
    async def test_try_http_connection_error(self, transport: HybridTransport):
        """Test HTTP request with connection error."""
        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(
                side_effect=ConnectionError("Connection refused")
            )
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_cls.return_value = mock_session

            success, response = await transport._try_http(
                "test-node", "http://localhost:8080/test", "GET"
            )

            assert not success
            assert response is None


# =============================================================================
# HybridTransport Tailscale Transport Tests
# =============================================================================


class TestHybridTransportTailscale:
    """Tests for HybridTransport Tailscale communication."""

    @pytest.mark.asyncio
    async def test_try_tailscale_no_registry(self, transport: HybridTransport):
        """Test Tailscale fails gracefully without registry."""
        # Patch at the module that contains get_registry since it's imported inside the function
        with patch(
            "app.distributed.dynamic_registry.get_registry",
            side_effect=ImportError("No registry"),
        ):
            success, response = await transport._try_tailscale(
                "test-node", 8080, "/test", "GET"
            )

            assert not success
            assert response is None

    @pytest.mark.asyncio
    async def test_try_tailscale_no_node_info(self, transport: HybridTransport):
        """Test Tailscale fails gracefully when node not in registry."""
        mock_registry = MagicMock()
        mock_registry._nodes = {}

        with patch(
            "app.distributed.dynamic_registry.get_registry", return_value=mock_registry
        ):
            success, response = await transport._try_tailscale(
                "test-node", 8080, "/test", "GET"
            )

            assert not success
            assert response is None

    @pytest.mark.asyncio
    async def test_try_tailscale_no_tailscale_ip(self, transport: HybridTransport):
        """Test Tailscale fails gracefully when node has no Tailscale IP."""
        mock_node_info = MagicMock()
        mock_node_info.tailscale_ip = None

        mock_registry = MagicMock()
        mock_registry._nodes = {"test-node": mock_node_info}

        with patch(
            "app.distributed.dynamic_registry.get_registry", return_value=mock_registry
        ):
            success, response = await transport._try_tailscale(
                "test-node", 8080, "/test", "GET"
            )

            assert not success
            assert response is None


# =============================================================================
# HybridTransport Cloudflare Transport Tests
# =============================================================================


class TestHybridTransportCloudflare:
    """Tests for HybridTransport Cloudflare Zero Trust communication."""

    @pytest.mark.asyncio
    async def test_try_cloudflare_no_tunnel(self, transport: HybridTransport):
        """Test Cloudflare fails gracefully without tunnel configured."""
        success, response = await transport._try_cloudflare(
            "test-node", "/test", "GET"
        )

        assert not success
        assert response is None

    @pytest.mark.asyncio
    async def test_try_cloudflare_with_tunnel(self, transport: HybridTransport):
        """Test Cloudflare request with tunnel configured."""
        transport.configure_cloudflare("test-node", "tunnel.example.com")

        with patch.object(transport, "_try_http") as mock_http:
            mock_http.return_value = (True, {"status": "ok"})

            success, response = await transport._try_cloudflare(
                "test-node", "/test", "GET"
            )

            assert success
            assert response == {"status": "ok"}
            mock_http.assert_called_once_with(
                "test-node", "https://tunnel.example.com/test", "GET", None, 15.0
            )


# =============================================================================
# HybridTransport aria2 Transport Tests
# =============================================================================


class TestHybridTransportAria2:
    """Tests for HybridTransport aria2 file downloads."""

    @pytest.mark.asyncio
    async def test_try_aria2_not_installed(self, transport: HybridTransport):
        """Test aria2 fails gracefully when not installed."""
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("aria2c not found"),
        ):
            success, path = await transport._try_aria2(
                "test-node",
                ["http://example.com/file.npz"],
                "/tmp/file.npz",
            )

            assert not success
            assert path is None

    @pytest.mark.asyncio
    async def test_try_aria2_timeout(self, transport: HybridTransport):
        """Test aria2 handles timeout."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_proc
        ), patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp.return_value.__enter__ = MagicMock(
                return_value=MagicMock(name="/tmp/input.txt")
            )
            mock_temp.return_value.__exit__ = MagicMock(return_value=None)
            mock_temp.return_value.name = "/tmp/input.txt"

            success, path = await transport._try_aria2(
                "test-node",
                ["http://example.com/file.npz"],
                "/tmp/file.npz",
            )

            assert not success
            assert path is None


# =============================================================================
# HybridTransport SSH Transport Tests
# =============================================================================


class TestHybridTransportSsh:
    """Tests for HybridTransport SSH communication."""

    @pytest.mark.asyncio
    async def test_get_ssh_transport_lazy_load(self, transport: HybridTransport):
        """Test SSH transport is lazily loaded."""
        mock_ssh = MagicMock()

        # Patch at the module that contains get_ssh_transport since it's imported inside the function
        with patch(
            "app.distributed.ssh_transport.get_ssh_transport",
            return_value=mock_ssh,
        ):
            ssh = await transport._get_ssh_transport()
            assert ssh is mock_ssh

    @pytest.mark.asyncio
    async def test_get_ssh_transport_import_error(self, transport: HybridTransport):
        """Test SSH transport handles import error gracefully."""
        # Patch the import itself to raise ImportError
        with patch.dict("sys.modules", {"app.distributed.ssh_transport": None}):
            # Reset the transport's cached SSH transport
            transport._ssh_transport = None
            ssh = await transport._get_ssh_transport()
            # Should be None since import fails
            assert ssh is None

    @pytest.mark.asyncio
    async def test_try_ssh_no_transport(self, transport: HybridTransport):
        """Test SSH request fails gracefully without transport."""
        with patch.object(transport, "_get_ssh_transport", return_value=None):
            success, response = await transport._try_ssh(
                "test-node", "heartbeat", {"node_id": "test"}
            )

            assert not success
            assert response is None

    @pytest.mark.asyncio
    async def test_try_ssh_success(self, transport: HybridTransport):
        """Test successful SSH request."""
        mock_ssh = AsyncMock()
        mock_ssh.send_command = AsyncMock(return_value=(True, {"status": "ok"}))

        with patch.object(transport, "_get_ssh_transport", return_value=mock_ssh):
            success, response = await transport._try_ssh(
                "test-node", "heartbeat", {"node_id": "test"}
            )

            assert success
            assert response == {"status": "ok"}


# =============================================================================
# HybridTransport send_request Tests
# =============================================================================


class TestHybridTransportSendRequest:
    """Tests for HybridTransport.send_request() with fallback chain."""

    @pytest.mark.asyncio
    async def test_send_request_http_success(self, transport: HybridTransport):
        """Test send_request succeeds with HTTP."""
        with patch.object(transport, "_try_http") as mock_http:
            mock_http.return_value = (True, {"status": "ok"})

            success, response = await transport.send_request(
                node_id="test-node",
                host="localhost",
                port=8080,
                path="/test",
                method="GET",
            )

            assert success
            assert response == {"status": "ok"}
            mock_http.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_request_http_fail_tailscale_success(
        self, transport: HybridTransport
    ):
        """Test send_request falls back to Tailscale after HTTP failure."""
        with patch.object(transport, "_try_http") as mock_http, patch.object(
            transport, "_try_tailscale"
        ) as mock_ts:
            mock_http.return_value = (False, None)
            mock_ts.return_value = (True, {"status": "ok"})

            success, response = await transport.send_request(
                node_id="test-node",
                host="localhost",
                port=8080,
                path="/test",
                method="GET",
            )

            assert success
            assert response == {"status": "ok"}
            mock_http.assert_called_once()
            mock_ts.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_request_cloudflare_fallback(self, transport: HybridTransport):
        """Test send_request falls back to Cloudflare after Tailscale failure."""
        transport.configure_cloudflare("test-node", "tunnel.example.com")

        with patch.object(transport, "_try_http") as mock_http, patch.object(
            transport, "_try_tailscale"
        ) as mock_ts, patch.object(transport, "_try_cloudflare") as mock_cf:
            mock_http.return_value = (False, None)
            mock_ts.return_value = (False, None)
            mock_cf.return_value = (True, {"status": "ok"})

            success, response = await transport.send_request(
                node_id="test-node",
                host="localhost",
                port=8080,
                path="/test",
                method="GET",
            )

            assert success
            assert response == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_send_request_ssh_fallback(self, transport: HybridTransport):
        """Test send_request falls back to SSH after other failures."""
        with patch.object(transport, "_try_http") as mock_http, patch.object(
            transport, "_try_tailscale"
        ) as mock_ts, patch.object(transport, "_check_ssh_available") as mock_ssh_check, patch.object(
            transport, "_try_ssh"
        ) as mock_ssh:
            mock_http.return_value = (False, None)
            mock_ts.return_value = (False, None)
            mock_ssh_check.return_value = True
            mock_ssh.return_value = (True, {"status": "ok"})

            success, response = await transport.send_request(
                node_id="test-node",
                host="localhost",
                port=8080,
                path="/test",
                method="GET",
                command_type="heartbeat",
            )

            assert success
            assert response == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_send_request_all_fail(self, transport: HybridTransport):
        """Test send_request returns failure when all transports fail."""
        with patch.object(transport, "_try_http") as mock_http, patch.object(
            transport, "_try_tailscale"
        ) as mock_ts:
            mock_http.return_value = (False, None)
            mock_ts.return_value = (False, None)

            success, response = await transport.send_request(
                node_id="test-node",
                host="localhost",
                port=8080,
                path="/test",
                method="GET",
            )

            assert not success
            assert response is None


# =============================================================================
# HybridTransport Convenience Method Tests
# =============================================================================


class TestHybridTransportConvenienceMethods:
    """Tests for HybridTransport convenience methods."""

    @pytest.mark.asyncio
    async def test_send_heartbeat(self, transport: HybridTransport):
        """Test send_heartbeat convenience method."""
        with patch.object(transport, "send_request") as mock_request:
            mock_request.return_value = (True, {"status": "ok"})

            success, response = await transport.send_heartbeat(
                node_id="test-node",
                host="localhost",
                port=8080,
                self_info={"node_id": "self"},
            )

            assert success
            mock_request.assert_called_once_with(
                node_id="test-node",
                host="localhost",
                port=8080,
                path="/heartbeat",
                method="POST",
                payload={"node_id": "self"},
                command_type="heartbeat",
            )

    @pytest.mark.asyncio
    async def test_send_relay_heartbeat(self, transport: HybridTransport):
        """Test send_relay_heartbeat convenience method."""
        with patch.object(transport, "send_request") as mock_request:
            mock_request.return_value = (True, {"status": "ok"})

            success, response = await transport.send_relay_heartbeat(
                node_id="relay-node",
                host="localhost",
                port=8080,
                self_info={"node_id": "self"},
            )

            assert success
            mock_request.assert_called_once_with(
                node_id="relay-node",
                host="localhost",
                port=8080,
                path="/relay/heartbeat",
                method="POST",
                payload={"node_id": "self"},
                command_type="relay_heartbeat",
            )

    @pytest.mark.asyncio
    async def test_request_job_start(self, transport: HybridTransport):
        """Test request_job_start convenience method."""
        with patch.object(transport, "send_request") as mock_request:
            mock_request.return_value = (True, {"job_id": "123"})

            success, response = await transport.request_job_start(
                node_id="worker-node",
                host="localhost",
                port=8080,
                job_payload={"type": "selfplay", "config": "hex8_2p"},
            )

            assert success
            mock_request.assert_called_once_with(
                node_id="worker-node",
                host="localhost",
                port=8080,
                path="/jobs/start",
                method="POST",
                payload={"type": "selfplay", "config": "hex8_2p"},
                command_type="start_job",
            )

    @pytest.mark.asyncio
    async def test_get_node_status(self, transport: HybridTransport):
        """Test get_node_status convenience method."""
        with patch.object(transport, "send_request") as mock_request:
            mock_request.return_value = (True, {"role": "leader", "alive_peers": 5})

            response = await transport.get_node_status(
                node_id="test-node",
                host="localhost",
                port=8080,
            )

            assert response == {"role": "leader", "alive_peers": 5}
            mock_request.assert_called_once_with(
                node_id="test-node",
                host="localhost",
                port=8080,
                path="/status",
                method="GET",
                command_type="status",
            )

    @pytest.mark.asyncio
    async def test_get_node_status_failure(self, transport: HybridTransport):
        """Test get_node_status returns None on failure."""
        with patch.object(transport, "send_request") as mock_request:
            mock_request.return_value = (False, None)

            response = await transport.get_node_status(
                node_id="test-node",
                host="localhost",
                port=8080,
            )

            assert response is None


# =============================================================================
# HybridTransport Transport Stats Tests
# =============================================================================


class TestHybridTransportStats:
    """Tests for HybridTransport statistics."""

    def test_get_transport_stats_empty(self, transport: HybridTransport):
        """Test transport stats with no nodes."""
        stats = transport.get_transport_stats()
        assert stats == {}

    def test_get_transport_stats_with_nodes(self, transport: HybridTransport):
        """Test transport stats with tracked nodes."""
        # Create some state
        state = transport._get_state("node-1")
        state.http_consecutive_failures = 2
        state.ssh_available = True
        state.cloudflare_tunnel = "tunnel.example.com"
        state.aria2_available = True

        stats = transport.get_transport_stats()

        assert "node-1" in stats
        assert stats["node-1"]["preferred_transport"] == "http"
        assert stats["node-1"]["http_failures"] == 2
        assert stats["node-1"]["ssh_available"]
        assert stats["node-1"]["cloudflare_available"]
        assert stats["node-1"]["aria2_available"]


# =============================================================================
# HybridTransport File Download Tests
# =============================================================================


class TestHybridTransportDownload:
    """Tests for HybridTransport file downloads."""

    @pytest.mark.asyncio
    async def test_download_file_with_aria2(self, transport: HybridTransport):
        """Test download_file with aria2."""
        transport.configure_aria2("test-node", available=True)

        with patch.object(transport, "_try_aria2") as mock_aria2:
            mock_aria2.return_value = (True, "/tmp/file.npz")

            success, path = await transport.download_file(
                node_id="test-node",
                urls=["http://example.com/file.npz"],
                local_path="/tmp/file.npz",
            )

            assert success
            assert path == "/tmp/file.npz"

    @pytest.mark.asyncio
    async def test_download_file_http_fallback(self, transport: HybridTransport):
        """Test download_file falls back to HTTP when aria2 fails."""
        with patch.object(transport, "_try_aria2") as mock_aria2, patch(
            "aiohttp.ClientSession"
        ) as mock_session_cls:
            mock_aria2.return_value = (False, None)

            # Create a proper async iterator for content
            async def async_chunk_gen():
                yield b"data"

            mock_content = MagicMock()
            mock_content.iter_chunked = MagicMock(return_value=async_chunk_gen())

            # Mock HTTP download
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.content = mock_content
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_cls.return_value = mock_session

            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = Path(tmpdir) / "file.npz"

                success, path = await transport.download_file(
                    node_id="test-node",
                    urls=["http://example.com/file.npz"],
                    local_path=str(local_path),
                )

                # Verify aria2 was tried first
                mock_aria2.assert_called_once()
                # HTTP fallback should be triggered
                mock_session_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_file_all_fail(self, transport: HybridTransport):
        """Test download_file returns failure when all methods fail."""
        with patch.object(transport, "_try_aria2") as mock_aria2, patch(
            "aiohttp.ClientSession"
        ) as mock_session_cls:
            mock_aria2.return_value = (False, None)

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(
                side_effect=ConnectionError("Connection refused")
            )
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_cls.return_value = mock_session

            success, path = await transport.download_file(
                node_id="test-node",
                urls=["http://example.com/file.npz"],
                local_path="/tmp/file.npz",
            )

            assert not success
            assert path is None


# =============================================================================
# HybridTransport Probe Tests
# =============================================================================


class TestHybridTransportProbe:
    """Tests for HybridTransport transport probing."""

    @pytest.mark.asyncio
    async def test_probe_all_transports(self, transport: HybridTransport):
        """Test probing all transports."""
        with patch.object(transport, "_try_http") as mock_http, patch.object(
            transport, "_try_tailscale"
        ) as mock_ts, patch.object(transport, "_get_ssh_transport") as mock_ssh_get:
            mock_http.return_value = (True, None)
            mock_ts.return_value = (False, None)
            mock_ssh_get.return_value = None

            results = await transport.probe_all_transports(
                node_id="test-node",
                host="localhost",
                port=8080,
            )

            assert "http" in results
            assert results["http"][0]  # HTTP succeeded
            assert "tailscale" in results
            assert not results["tailscale"][0]  # Tailscale failed


# =============================================================================
# HybridTransport SSH Availability Tests
# =============================================================================


class TestHybridTransportSshAvailability:
    """Tests for HybridTransport SSH availability checking."""

    @pytest.mark.asyncio
    async def test_check_ssh_available_cached(self, transport: HybridTransport):
        """Test SSH availability uses cached state."""
        state = transport._get_state("test-node")
        state.ssh_available = True

        result = await transport._check_ssh_available("test-node")
        assert result

    @pytest.mark.asyncio
    async def test_check_ssh_available_from_config(self, transport: HybridTransport):
        """Test SSH availability checks cluster config."""
        mock_host_config = MagicMock()
        mock_host_config.ssh_host = "192.168.1.1"
        mock_host_config.tailscale_ip = None

        with patch(
            "app.config.cluster_config.get_cluster_nodes",
            return_value={"test-node": mock_host_config},
        ):
            result = await transport._check_ssh_available("test-node")
            assert result

    @pytest.mark.asyncio
    async def test_check_ssh_available_vast_provider(self, transport: HybridTransport):
        """Test SSH availability enabled for Vast provider."""
        with patch(
            "app.config.cluster_config.get_cluster_nodes",
            return_value={},
        ), patch(
            "app.config.cluster_config.get_host_provider",
            return_value="vast",
        ):
            result = await transport._check_ssh_available("vast-12345")
            assert result


# =============================================================================
# Module-Level Function Tests
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_hybrid_transport_singleton(self):
        """Test get_hybrid_transport returns singleton."""
        # Reset global for clean test
        import app.distributed.hybrid_transport as module

        module._hybrid_transport = None

        t1 = get_hybrid_transport()
        t2 = get_hybrid_transport()

        assert t1 is t2

    @pytest.mark.asyncio
    async def test_diagnose_node_connectivity(self):
        """Test diagnose_node_connectivity runs probes."""
        with patch(
            "app.distributed.hybrid_transport.get_hybrid_transport"
        ) as mock_get_transport:
            mock_transport = AsyncMock()
            mock_transport.probe_all_transports = AsyncMock(
                return_value={
                    "http": (True, 10.5),
                    "tailscale": (False, 5000.0),
                    "ssh": (True, 50.2),
                }
            )
            mock_get_transport.return_value = mock_transport

            result = await diagnose_node_connectivity(
                node_id="test-node",
                host="localhost",
                port=8080,
            )

            assert result["node_id"] == "test-node"
            assert result["best_transport"] == "http"
            assert result["best_latency_ms"] == 10.5
            assert "HTTP working normally" in result["recommendation"]


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_http_failures_before_ssh(self):
        """Test HTTP failures threshold."""
        assert HTTP_FAILURES_BEFORE_SSH == 3

    def test_ssh_success_before_http_retry(self):
        """Test SSH success threshold."""
        assert SSH_SUCCESS_BEFORE_HTTP_RETRY == 5

    def test_transport_health_check_interval(self):
        """Test health check interval."""
        assert TRANSPORT_HEALTH_CHECK_INTERVAL == 300


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_node_state_multiple_operations(self, node_state: NodeTransportState):
        """Test state consistency across multiple operations."""
        # Simulate a realistic sequence
        node_state.record_http_failure()
        node_state.record_http_failure()
        node_state.record_http_failure()

        node_state.ssh_available = True
        node_state.record_http_failure()  # Should trigger switch

        assert node_state.preferred_transport == TransportType.SSH

        # SSH successes
        for _ in range(SSH_SUCCESS_BEFORE_HTTP_RETRY):
            node_state.record_ssh_success()

        # Should reset HTTP failures
        assert node_state.http_consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, transport: HybridTransport):
        """Test handling of concurrent requests."""
        with patch.object(transport, "_try_http") as mock_http:
            mock_http.return_value = (True, {"status": "ok"})

            # Run multiple concurrent requests
            tasks = [
                transport.send_request(
                    node_id="test-node",
                    host="localhost",
                    port=8080,
                    path="/test",
                    method="GET",
                )
                for _ in range(5)
            ]

            results = await asyncio.gather(*tasks)

            # All should succeed
            assert all(r[0] for r in results)
            assert mock_http.call_count == 5

    def test_transport_stats_multiple_nodes(self, transport: HybridTransport):
        """Test transport stats with multiple nodes."""
        for i in range(5):
            state = transport._get_state(f"node-{i}")
            state.http_consecutive_failures = i
            state.ssh_available = i % 2 == 0

        stats = transport.get_transport_stats()

        assert len(stats) == 5
        for i in range(5):
            assert f"node-{i}" in stats
            assert stats[f"node-{i}"]["http_failures"] == i
            assert stats[f"node-{i}"]["ssh_available"] == (i % 2 == 0)
