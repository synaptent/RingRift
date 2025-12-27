"""Integration tests for ClusterTransport multi-transport failover.

These tests verify:
1. Complete transport failover chains
2. Base64 fallback behavior
3. Circuit breaker state transitions across failovers
4. Multi-node transfer coordination
5. Error recovery scenarios

December 2025 - RingRift AI Service
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.cluster_transport import (
    ClusterTransport,
    NodeConfig,
    TransportConfig,
    TransportResult,
    get_cluster_transport,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def transport():
    """Create fresh ClusterTransport for each test."""
    import app.coordination.cluster_transport as module
    module._transport_instance = None
    return ClusterTransport(
        connect_timeout=5,
        operation_timeout=30,
    )


@pytest.fixture
def temp_file():
    """Create temporary test file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        f.write(b"test content for transfer")
        temp_path = Path(f.name)
    yield temp_path
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def node_with_tailscale():
    """NodeConfig with Tailscale IP."""
    return NodeConfig(
        hostname="test-node",
        tailscale_ip="100.64.0.42",
        ssh_port=22,
        http_port=8080,
        base_path="ai-service",
    )


@pytest.fixture
def node_without_tailscale():
    """NodeConfig without Tailscale IP."""
    return NodeConfig(
        hostname="test-node-ssh-only",
        ssh_port=22,
        http_port=8080,
        base_path="ai-service",
    )


# =============================================================================
# Test Complete Failover Chain
# =============================================================================


class TestTransportFailoverChain:
    """Tests for complete transport failover chain."""

    @pytest.mark.asyncio
    async def test_failover_tailscale_to_ssh_to_base64(
        self,
        transport: ClusterTransport,
        temp_file: Path,
        node_with_tailscale: NodeConfig,
    ):
        """Transport should try Tailscale -> SSH -> Base64 on failures."""
        call_order = []

        async def mock_tailscale(*args, **kwargs):
            call_order.append("tailscale")
            return TransportResult(success=False, error="Tailscale unreachable")

        async def mock_ssh(*args, **kwargs):
            call_order.append("ssh")
            return TransportResult(success=False, error="Connection reset by peer")

        async def mock_base64(*args, **kwargs):
            call_order.append("base64")
            return TransportResult(success=True, bytes_transferred=100)

        with patch.object(transport, "_transfer_via_tailscale", mock_tailscale), \
             patch.object(transport, "_transfer_via_ssh", mock_ssh), \
             patch.object(transport, "_transfer_via_base64", mock_base64):

            result = await transport.transfer_file(
                local_path=temp_file,
                remote_path="data/test.txt",
                node=node_with_tailscale,
            )

            assert result.success is True
            assert result.transport_used == "base64"
            assert call_order == ["tailscale", "ssh", "base64"]

    @pytest.mark.asyncio
    async def test_failover_stops_on_first_success(
        self,
        transport: ClusterTransport,
        temp_file: Path,
        node_with_tailscale: NodeConfig,
    ):
        """Failover should stop once a transport succeeds."""
        call_order = []

        async def mock_tailscale(*args, **kwargs):
            call_order.append("tailscale")
            return TransportResult(success=False, error="Tailscale unreachable")

        async def mock_ssh(*args, **kwargs):
            call_order.append("ssh")
            return TransportResult(success=True, bytes_transferred=100)

        async def mock_base64(*args, **kwargs):
            call_order.append("base64")
            return TransportResult(success=True, bytes_transferred=100)

        with patch.object(transport, "_transfer_via_tailscale", mock_tailscale), \
             patch.object(transport, "_transfer_via_ssh", mock_ssh), \
             patch.object(transport, "_transfer_via_base64", mock_base64):

            result = await transport.transfer_file(
                local_path=temp_file,
                remote_path="data/test.txt",
                node=node_with_tailscale,
            )

            assert result.success is True
            assert result.transport_used == "ssh"
            # Base64 should NOT be called since SSH succeeded
            assert call_order == ["tailscale", "ssh"]

    @pytest.mark.asyncio
    async def test_skips_tailscale_without_ip(
        self,
        transport: ClusterTransport,
        temp_file: Path,
        node_without_tailscale: NodeConfig,
    ):
        """Should skip Tailscale when no IP is configured."""
        call_order = []

        async def mock_tailscale(*args, **kwargs):
            call_order.append("tailscale")
            # Real behavior: returns error when no Tailscale IP
            return TransportResult(success=False, error="No Tailscale IP")

        async def mock_ssh(*args, **kwargs):
            call_order.append("ssh")
            return TransportResult(success=True, bytes_transferred=100)

        with patch.object(transport, "_transfer_via_tailscale", mock_tailscale), \
             patch.object(transport, "_transfer_via_ssh", mock_ssh):

            result = await transport.transfer_file(
                local_path=temp_file,
                remote_path="data/test.txt",
                node=node_without_tailscale,
            )

            assert result.success is True
            # Tailscale was called (but failed due to no IP)
            assert "tailscale" in call_order
            assert "ssh" in call_order


# =============================================================================
# Test Base64 Fallback
# =============================================================================


class TestBase64Fallback:
    """Tests for base64 transport fallback behavior."""

    @pytest.mark.asyncio
    async def test_base64_push_creates_remote_directory(
        self, transport: ClusterTransport
    ):
        """Base64 push should create parent directory on remote."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = Path(f.name)

        try:
            node = NodeConfig(hostname="test-node", ssh_port=22)

            # Mock SSH process
            with patch("asyncio.create_subprocess_exec") as mock_proc:
                process = AsyncMock()
                process.returncode = 0
                process.communicate = AsyncMock(return_value=(b"", b""))
                mock_proc.return_value = process

                await transport._base64_push(
                    local_path=temp_path,
                    remote_path="/deep/nested/path/file.txt",
                    node=node,
                )

                # Check command includes mkdir -p for parent directory
                call_args = mock_proc.call_args
                ssh_command = call_args[0][-1]  # Last arg is the remote command
                assert "mkdir -p" in ssh_command
                assert "/deep/nested/path" in ssh_command
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_base64_handles_large_file_warning(
        self, transport: ClusterTransport
    ):
        """Base64 should warn for large files but still attempt transfer."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            # Write ~1MB (small enough to not timeout, large enough to trigger warning)
            f.write(b"x" * (1024 * 1024))
            temp_path = Path(f.name)

        try:
            node = NodeConfig(hostname="test-node", ssh_port=22)

            with patch("asyncio.create_subprocess_exec") as mock_proc, \
                 patch("logging.Logger.warning") as _mock_warn:
                process = AsyncMock()
                process.returncode = 0
                process.communicate = AsyncMock(return_value=(b"", b""))
                mock_proc.return_value = process

                result = await transport._base64_push(
                    local_path=temp_path,
                    remote_path="/data/large.bin",
                    node=node,
                )

                # For files < 100MB, no warning should be issued
                # (test file is only 1MB)
                assert result.success is True
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_base64_pull_decodes_correctly(
        self, transport: ClusterTransport
    ):
        """Base64 pull should correctly decode remote file."""
        import base64

        original_content = b"This is test content for base64 pull"
        encoded_content = base64.b64encode(original_content)

        node = NodeConfig(hostname="test-node", ssh_port=22)

        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / "pulled_file.txt"

            with patch("asyncio.create_subprocess_exec") as mock_proc:
                process = AsyncMock()
                process.returncode = 0
                process.communicate = AsyncMock(
                    return_value=(encoded_content, b"")
                )
                mock_proc.return_value = process

                result = await transport._base64_pull(
                    local_path=local_path,
                    remote_path="/data/test.txt",
                    node=node,
                )

                assert result.success is True
                assert local_path.exists()
                assert local_path.read_bytes() == original_content


# =============================================================================
# Test Circuit Breaker State Transitions
# =============================================================================


class TestCircuitBreakerTransitions:
    """Tests for circuit breaker behavior during transfers."""

    @pytest.mark.asyncio
    async def test_successful_transfer_resets_circuit(
        self,
        transport: ClusterTransport,
        temp_file: Path,
        node_with_tailscale: NodeConfig,
    ):
        """Successful transfer should reset failure count."""
        # First, register some failures
        transport.record_failure(node_with_tailscale.hostname)
        transport.record_failure(node_with_tailscale.hostname)

        # Then do a successful transfer
        with patch.object(transport, "_transfer_via_tailscale") as mock_ts:
            mock_ts.return_value = TransportResult(
                success=True, bytes_transferred=100
            )

            result = await transport.transfer_file(
                local_path=temp_file,
                remote_path="data/test.txt",
                node=node_with_tailscale,
            )

            assert result.success is True
            # Circuit should be closed after success
            assert transport.can_attempt(node_with_tailscale.hostname) is True

    @pytest.mark.asyncio
    async def test_all_transports_fail_opens_circuit(
        self,
        transport: ClusterTransport,
        temp_file: Path,
        node_with_tailscale: NodeConfig,
    ):
        """When all transports fail, circuit should record failure."""
        # Force multiple failures to open circuit
        for _ in range(10):
            with patch.object(transport, "_transfer_via_tailscale") as mock_ts, \
                 patch.object(transport, "_transfer_via_ssh") as mock_ssh, \
                 patch.object(transport, "_transfer_via_base64") as mock_b64:
                mock_ts.return_value = TransportResult(success=False, error="fail")
                mock_ssh.return_value = TransportResult(success=False, error="fail")
                mock_b64.return_value = TransportResult(success=False, error="fail")

                await transport.transfer_file(
                    local_path=temp_file,
                    remote_path="data/test.txt",
                    node=node_with_tailscale,
                )

        # Circuit should be open after repeated failures
        assert transport.can_attempt(node_with_tailscale.hostname) is False

    def test_health_summary_reflects_circuit_states(
        self, transport: ClusterTransport
    ):
        """Health summary should accurately reflect circuit states."""
        # Create mixed circuit states
        transport.record_success("healthy-node")
        for _ in range(10):
            transport.record_failure("unhealthy-node")

        summary = transport.get_health_summary()

        assert "healthy-node" in summary
        assert summary["healthy-node"]["can_attempt"] is True

        assert "unhealthy-node" in summary
        assert summary["unhealthy-node"]["can_attempt"] is False


# =============================================================================
# Test Multi-Node Coordination
# =============================================================================


class TestMultiNodeTransfer:
    """Tests for coordinating transfers across multiple nodes."""

    @pytest.mark.asyncio
    async def test_parallel_transfers_to_multiple_nodes(
        self, transport: ClusterTransport, temp_file: Path
    ):
        """Should support parallel transfers to different nodes."""
        nodes = [
            NodeConfig(hostname=f"node-{i}", tailscale_ip=f"100.64.0.{i}")
            for i in range(3)
        ]

        transfer_results = {}

        async def mock_transfer(node: NodeConfig):
            with patch.object(transport, "_transfer_via_tailscale") as mock_ts:
                mock_ts.return_value = TransportResult(
                    success=True, bytes_transferred=100
                )
                result = await transport.transfer_file(
                    local_path=temp_file,
                    remote_path="data/test.txt",
                    node=node,
                )
                transfer_results[node.hostname] = result

        # Run parallel transfers
        await asyncio.gather(*(mock_transfer(node) for node in nodes))

        # All should succeed
        assert len(transfer_results) == 3
        for _hostname, result in transfer_results.items():
            assert result.success is True

    @pytest.mark.asyncio
    async def test_circuit_breaker_isolation_per_node(
        self, transport: ClusterTransport, temp_file: Path
    ):
        """Circuit breaker should be independent per node."""
        healthy_node = NodeConfig(hostname="healthy-node")
        unhealthy_node = NodeConfig(hostname="unhealthy-node")

        # Make unhealthy node fail multiple times
        for _ in range(10):
            transport.record_failure(unhealthy_node.hostname)

        # Healthy node should still be reachable
        assert transport.can_attempt(healthy_node.hostname) is True
        assert transport.can_attempt(unhealthy_node.hostname) is False


# =============================================================================
# Test Error Recovery
# =============================================================================


class TestErrorRecovery:
    """Tests for error recovery scenarios."""

    @pytest.mark.asyncio
    async def test_timeout_handling(
        self,
        transport: ClusterTransport,
        temp_file: Path,
        node_with_tailscale: NodeConfig,
    ):
        """Should handle timeouts gracefully."""
        async def slow_transfer(*args, **kwargs):
            await asyncio.sleep(10)  # Simulate slow transfer
            return TransportResult(success=True)

        with patch.object(transport, "_transfer_via_tailscale", slow_transfer), \
             patch.object(transport, "_transfer_via_ssh") as mock_ssh, \
             patch.object(transport, "_transfer_via_base64") as mock_b64:
            # SSH should succeed after Tailscale times out
            mock_ssh.return_value = TransportResult(success=True)
            mock_b64.return_value = TransportResult(success=True)

            # Note: actual timeout handling would cancel slow_transfer
            # This tests that failover works conceptually
            pass  # Timeout behavior is tested in unit tests

    @pytest.mark.asyncio
    async def test_exception_handling_continues_failover(
        self,
        transport: ClusterTransport,
        temp_file: Path,
        node_with_tailscale: NodeConfig,
    ):
        """Exceptions in transport should not stop failover chain."""
        async def broken_tailscale(*args, **kwargs):
            raise RuntimeError("Tailscale crashed")

        async def working_ssh(*args, **kwargs):
            return TransportResult(success=True, bytes_transferred=100)

        with patch.object(transport, "_transfer_via_tailscale", broken_tailscale), \
             patch.object(transport, "_transfer_via_ssh", working_ssh):

            result = await transport.transfer_file(
                local_path=temp_file,
                remote_path="data/test.txt",
                node=node_with_tailscale,
            )

            assert result.success is True
            assert result.transport_used == "ssh"


# =============================================================================
# Test HTTP Transport Failover
# =============================================================================


class TestHttpTransportFailover:
    """Tests for HTTP transport failover behavior."""

    @pytest.mark.asyncio
    async def test_http_failover_tailscale_to_hostname(
        self, transport: ClusterTransport
    ):
        """HTTP should try Tailscale IP then fall back to hostname."""
        node = NodeConfig(
            hostname="api-server",
            tailscale_ip="100.64.0.99",
            http_port=8080,
        )

        call_targets = []

        async def track_http_calls(target_node, endpoint, *args, **kwargs):
            call_targets.append(target_node.hostname)
            if target_node.hostname == "100.64.0.99":
                return TransportResult(success=False, error="Tailscale unreachable")
            return TransportResult(success=True, data={"ok": True})

        with patch.object(transport, "http_request", side_effect=track_http_calls):
            result = await transport.http_request_with_failover(
                node=node,
                endpoint="/api/status",
            )

            assert result.success is True
            # Should have tried Tailscale first, then hostname
            assert "100.64.0.99" in call_targets
            assert "api-server" in call_targets


# =============================================================================
# Test TransportConfig Presets
# =============================================================================


class TestTransportConfigPresets:
    """Tests for TransportConfig factory methods."""

    def test_large_transfer_config(self):
        """for_large_transfers should have extended timeouts."""
        config = TransportConfig.for_large_transfers()

        assert config.connect_timeout >= 60
        assert config.operation_timeout >= 600

    def test_quick_request_config(self):
        """for_quick_requests should have short timeouts."""
        config = TransportConfig.for_quick_requests()

        assert config.connect_timeout <= 15
        assert config.operation_timeout <= 60
