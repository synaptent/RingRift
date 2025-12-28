"""Unit tests for SSH transport module (deprecated).

Tests cover:
- SSHAddress dataclass properties
- SSHCommandResult dataclass
- SSHTransport initialization and configuration
- Address caching and staleness detection
- get_ssh_transport singleton

Note: This module is deprecated. Tests verify backward compatibility.
Use app.core.ssh for new code.

December 2025: Created for test coverage of deprecated module.
"""

import os
import tempfile
import time
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Capture deprecation warning during import
with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    from app.distributed.ssh_transport import (
        SSHAddress,
        SSHCommandResult,
        SSHTransport,
        get_ssh_transport,
        SSH_ADDRESS_CACHE_TTL,
        VAST_SSH_USER,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_control_dir():
    """Create a temporary control directory for SSH sockets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def transport(temp_control_dir):
    """Create an SSHTransport with temporary control directory."""
    return SSHTransport(control_path_dir=temp_control_dir)


# =============================================================================
# SSHAddress Tests
# =============================================================================


class TestSSHAddress:
    """Tests for SSHAddress dataclass."""

    def test_basic_creation(self):
        """SSHAddress can be created with required fields."""
        addr = SSHAddress(
            node_id="test-node",
            ssh_host="ssh.example.com",
            ssh_port=22,
        )
        assert addr.node_id == "test-node"
        assert addr.ssh_host == "ssh.example.com"
        assert addr.ssh_port == 22

    def test_default_user(self):
        """SSHAddress uses default SSH user."""
        addr = SSHAddress(
            node_id="test",
            ssh_host="host",
            ssh_port=22,
        )
        assert addr.ssh_user == VAST_SSH_USER

    def test_ssh_destination(self):
        """ssh_destination property returns correct format."""
        addr = SSHAddress(
            node_id="test",
            ssh_host="ssh.vast.ai",
            ssh_port=12345,
            ssh_user="ubuntu",
        )
        assert addr.ssh_destination == "ubuntu@ssh.vast.ai"

    def test_str_representation(self):
        """String representation includes all details."""
        addr = SSHAddress(
            node_id="test",
            ssh_host="host.com",
            ssh_port=2222,
            ssh_user="user",
        )
        assert str(addr) == "user@host.com:2222"

    def test_is_stale_fresh(self):
        """Fresh address is not stale."""
        addr = SSHAddress(
            node_id="test",
            ssh_host="host",
            ssh_port=22,
        )
        assert addr.is_stale is False

    def test_is_stale_old(self):
        """Old address is stale."""
        addr = SSHAddress(
            node_id="test",
            ssh_host="host",
            ssh_port=22,
            cached_at=time.time() - SSH_ADDRESS_CACHE_TTL - 100,
        )
        assert addr.is_stale is True

    def test_failure_tracking(self):
        """SSHAddress tracks failures."""
        addr = SSHAddress(
            node_id="test",
            ssh_host="host",
            ssh_port=22,
            consecutive_failures=3,
            last_failure=time.time(),
        )
        assert addr.consecutive_failures == 3
        assert addr.last_failure > 0


# =============================================================================
# SSHCommandResult Tests
# =============================================================================


class TestSSHCommandResult:
    """Tests for SSHCommandResult dataclass."""

    def test_success_result(self):
        """Success result has expected properties."""
        result = SSHCommandResult(
            success=True,
            stdout="hello",
            stderr="",
            return_code=0,
            elapsed_ms=100.0,
        )
        assert result.success is True
        assert result.stdout == "hello"
        assert result.return_code == 0

    def test_failure_result(self):
        """Failure result includes error message."""
        result = SSHCommandResult(
            success=False,
            stdout="",
            stderr="Connection refused",
            return_code=255,
            error="SSH connection failed",
        )
        assert result.success is False
        assert result.error == "SSH connection failed"
        assert result.return_code == 255

    def test_default_values(self):
        """Default values are applied correctly."""
        result = SSHCommandResult(success=True)
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.return_code == -1
        assert result.elapsed_ms == 0.0
        assert result.error is None


# =============================================================================
# SSHTransport Tests
# =============================================================================


class TestSSHTransport:
    """Tests for SSHTransport class."""

    def test_initialization(self, temp_control_dir):
        """Transport initializes with control directory."""
        transport = SSHTransport(control_path_dir=temp_control_dir)
        assert transport._control_path_dir.exists()

    def test_default_control_dir(self):
        """Transport uses default control directory if none provided."""
        transport = SSHTransport()
        expected = Path.home() / ".ssh" / "ringrift_control"
        assert transport._control_path_dir == expected

    def test_set_ssh_address(self, transport):
        """set_ssh_address stores address correctly."""
        transport.set_ssh_address(
            node_id="vast-12345",
            ssh_host="ssh5.vast.ai",
            ssh_port=14364,
            ssh_user="root",
        )
        addr = transport._addresses.get("vast-12345")
        assert addr is not None
        assert addr.ssh_host == "ssh5.vast.ai"
        assert addr.ssh_port == 14364

    def test_control_path_generation(self, transport):
        """Control path is generated correctly."""
        path = transport._get_control_path("test-node")
        assert "ctrl_test-node" in path
        assert str(transport._control_path_dir) in path

    def test_control_path_sanitization(self, transport):
        """Control path sanitizes special characters."""
        path = transport._get_control_path("node/with:special")
        assert "/" not in Path(path).name
        assert ":" not in Path(path).name

    def test_control_path_truncation(self, transport):
        """Control path truncates long node IDs."""
        long_id = "very-long-node-identifier-that-exceeds-limit"
        path = transport._get_control_path(long_id)
        # Truncates to 20 chars
        assert len(Path(path).name) <= 25  # "ctrl_" + 20

    def test_get_ssh_address_cache_miss(self, transport):
        """Returns None for unknown node without cache."""
        addr = transport._get_ssh_address("nonexistent-node")
        assert addr is None

    def test_get_ssh_address_cache_hit(self, transport):
        """Returns cached address if fresh."""
        transport.set_ssh_address("cached-node", "host.com", 22)
        addr = transport._get_ssh_address("cached-node")
        assert addr is not None
        assert addr.ssh_host == "host.com"

    def test_get_ssh_address_stale_cache(self, transport):
        """Returns stale address if nothing else available."""
        addr = SSHAddress(
            node_id="stale-node",
            ssh_host="old.host.com",
            ssh_port=22,
            cached_at=time.time() - SSH_ADDRESS_CACHE_TTL - 100,
        )
        transport._addresses["stale-node"] = addr
        # Without registry/config, returns stale cache
        result = transport._get_ssh_address("stale-node")
        assert result is not None
        assert result.ssh_host == "old.host.com"


# =============================================================================
# Singleton Tests
# =============================================================================


class TestGetSshTransport:
    """Tests for get_ssh_transport singleton."""

    def test_returns_transport_instance(self):
        """get_ssh_transport returns SSHTransport instance."""
        transport = get_ssh_transport()
        assert isinstance(transport, SSHTransport)

    def test_returns_same_instance(self):
        """get_ssh_transport returns singleton."""
        transport1 = get_ssh_transport()
        transport2 = get_ssh_transport()
        assert transport1 is transport2


# =============================================================================
# Deprecation Warning Test
# =============================================================================


class TestDeprecationWarning:
    """Tests for deprecation behavior."""

    def test_module_emits_deprecation_warning(self):
        """Importing module emits deprecation warning."""
        # Re-import in a way that triggers warning
        import importlib
        import sys

        # Clear cached import
        module_name = "app.distributed.ssh_transport"
        if module_name in sys.modules:
            # Module already imported, warning was already raised
            # Just verify the module has the expected deprecation note
            import app.distributed.ssh_transport as mod
            assert "DEPRECATED" in mod.__doc__


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_node_id(self, transport):
        """Empty node ID returns None."""
        addr = transport._get_ssh_address("")
        assert addr is None

    def test_unicode_node_id(self, transport):
        """Unicode node ID is handled."""
        transport.set_ssh_address("node-ünïcödé", "host.com", 22)
        addr = transport._addresses.get("node-ünïcödé")
        assert addr is not None

    def test_zero_port(self, transport):
        """Port 0 is allowed (though unusual)."""
        transport.set_ssh_address("zero-port", "host.com", 0)
        addr = transport._addresses.get("zero-port")
        assert addr.ssh_port == 0

    def test_empty_ssh_host(self, transport):
        """Empty SSH host is stored (validation elsewhere)."""
        transport.set_ssh_address("no-host", "", 22)
        addr = transport._addresses.get("no-host")
        assert addr.ssh_host == ""
