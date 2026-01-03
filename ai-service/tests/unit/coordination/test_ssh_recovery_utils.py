"""Tests for SSH recovery utilities.

Created: Jan 3, 2026
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.ssh_recovery_utils import (
    SSHConfig,
    SSHRecoveryHelper,
    SSHResult,
    RecoveryCommandConfig,
    execute_ssh_recovery,
    get_ssh_recovery_helper,
    restart_tailscale_via_ssh,
    restart_p2p_via_ssh,
)


class TestSSHConfig:
    """Tests for SSHConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SSHConfig(host="test-host")
        assert config.host == "test-host"
        assert config.port == 22
        assert config.user == "root"
        assert config.key_path is None
        assert config.connect_timeout == 10
        assert config.command_timeout == 30

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = SSHConfig(
            host="custom-host",
            port=2222,
            user="admin",
            key_path="~/.ssh/custom",
            connect_timeout=20,
            command_timeout=60,
        )
        assert config.host == "custom-host"
        assert config.port == 2222
        assert config.user == "admin"
        assert config.key_path == "~/.ssh/custom"


class TestSSHResult:
    """Tests for SSHResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful result."""
        result = SSHResult(success=True, exit_code=0, stdout="OK")
        assert result.success
        assert result.exit_code == 0
        assert result.output == "OK"

    def test_failure_result(self) -> None:
        """Test failure result."""
        result = SSHResult(
            success=False,
            exit_code=1,
            stdout="",
            stderr="Error",
            error="Connection failed",
        )
        assert not result.success
        assert result.exit_code == 1
        assert result.output == "Error"
        assert result.error == "Connection failed"

    def test_combined_output(self) -> None:
        """Test combined stdout/stderr output."""
        result = SSHResult(
            success=True,
            exit_code=0,
            stdout="line1\n",
            stderr="line2\n",
        )
        assert result.output == "line1\nline2\n"


class TestRecoveryCommandConfig:
    """Tests for RecoveryCommandConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = RecoveryCommandConfig()
        assert not config.is_container
        assert config.hostname == "unknown"
        assert config.authkey is None
        assert config.ringrift_path == "~/ringrift/ai-service"

    def test_container_config(self) -> None:
        """Test container configuration."""
        config = RecoveryCommandConfig(
            is_container=True,
            hostname="vast-node-1",
            authkey="tskey-xxx",
        )
        assert config.is_container
        assert config.hostname == "vast-node-1"
        assert config.authkey == "tskey-xxx"


class TestSSHRecoveryHelper:
    """Tests for SSHRecoveryHelper class."""

    @pytest.fixture
    def helper(self) -> SSHRecoveryHelper:
        """Create helper instance."""
        return SSHRecoveryHelper(max_retries=2, base_delay=0.01, max_delay=0.1)

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        helper = SSHRecoveryHelper()
        assert helper._max_retries == 3
        assert helper._base_delay == 1.0
        assert helper._max_delay == 16.0

    def test_init_custom(self, helper: SSHRecoveryHelper) -> None:
        """Test custom initialization."""
        assert helper._max_retries == 2
        assert helper._base_delay == 0.01

    @pytest.mark.asyncio
    async def test_execute_once_success(self, helper: SSHRecoveryHelper) -> None:
        """Test single successful SSH execution."""
        config = SSHConfig(host="test-host")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"output", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await helper._execute_once(config, "echo test", 30)

        assert result.success
        assert result.exit_code == 0
        assert result.stdout == "output"

    @pytest.mark.asyncio
    async def test_execute_once_failure(self, helper: SSHRecoveryHelper) -> None:
        """Test single failed SSH execution."""
        config = SSHConfig(host="test-host")

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await helper._execute_once(config, "false", 30)

        assert not result.success
        assert result.exit_code == 1
        assert result.stderr == "error"

    @pytest.mark.asyncio
    async def test_execute_with_retry_success_first_try(
        self,
        helper: SSHRecoveryHelper,
    ) -> None:
        """Test execution succeeds on first try."""
        config = SSHConfig(host="test-host")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"OK", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await helper.execute_ssh_command(config, "echo test")

        assert result.success
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_execute_with_retry_success_after_failure(
        self,
        helper: SSHRecoveryHelper,
    ) -> None:
        """Test execution succeeds after initial failure."""
        config = SSHConfig(host="test-host")

        # First call fails with exit code 255 (SSH failure), second succeeds
        call_count = 0

        async def mock_communicate():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return b"", b"Connection refused"
            return b"OK", b""

        mock_proc = MagicMock()
        mock_proc.communicate = mock_communicate

        def set_returncode():
            mock_proc.returncode = 255 if call_count <= 1 else 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            # Patch returncode to change based on call count
            type(mock_proc).returncode = property(
                lambda self: 255 if call_count <= 1 else 0
            )
            result = await helper.execute_ssh_command(config, "echo test")

        assert result.success
        assert result.attempts == 2

    @pytest.mark.asyncio
    async def test_execute_all_retries_exhausted(
        self,
        helper: SSHRecoveryHelper,
    ) -> None:
        """Test all retries exhausted."""
        config = SSHConfig(host="test-host")

        mock_proc = MagicMock()
        mock_proc.returncode = 255
        mock_proc.communicate = AsyncMock(return_value=(b"", b"Connection refused"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await helper.execute_ssh_command(config, "echo test")

        assert not result.success
        assert result.attempts == 2  # max_retries=2
        assert "failed after 2 attempts" in (result.error or "")

    @pytest.mark.asyncio
    async def test_execute_timeout(self, helper: SSHRecoveryHelper) -> None:
        """Test execution timeout."""
        config = SSHConfig(host="test-host", command_timeout=1)

        async def slow_communicate():
            await asyncio.sleep(10)
            return b"", b""

        mock_proc = MagicMock()
        mock_proc.communicate = slow_communicate

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await helper.execute_ssh_command(
                config,
                "sleep 100",
                timeout=0.01,
                retry_on_timeout=False,
            )

        assert not result.success
        assert result.exit_code == 124  # Timeout exit code

    # =========================================================================
    # Command Builder Tests
    # =========================================================================

    def test_build_tailscale_command_host(self, helper: SSHRecoveryHelper) -> None:
        """Test Tailscale command for regular host."""
        config = RecoveryCommandConfig(
            is_container=False,
            hostname="lambda-node-1",
        )
        cmd = helper.build_tailscale_recovery_command(config)

        assert "systemctl restart tailscaled" in cmd
        assert "tailscale up" in cmd
        assert "--hostname='lambda-node-1'" in cmd
        assert "tailscale ip -4" in cmd

    def test_build_tailscale_command_container(
        self,
        helper: SSHRecoveryHelper,
    ) -> None:
        """Test Tailscale command for container."""
        config = RecoveryCommandConfig(
            is_container=True,
            hostname="vast-node-1",
        )
        cmd = helper.build_tailscale_recovery_command(config)

        assert "pkill -9 tailscaled" in cmd
        assert "--tun=userspace-networking" in cmd
        assert "tailscale up" in cmd
        assert "--hostname='vast-node-1'" in cmd

    def test_build_tailscale_command_with_authkey(
        self,
        helper: SSHRecoveryHelper,
    ) -> None:
        """Test Tailscale command includes authkey."""
        config = RecoveryCommandConfig(
            is_container=False,
            hostname="test",
            authkey="tskey-test-123",
        )
        cmd = helper.build_tailscale_recovery_command(config)

        assert "--authkey=tskey-test-123" in cmd

    def test_build_p2p_restart_command(self, helper: SSHRecoveryHelper) -> None:
        """Test P2P restart command."""
        config = RecoveryCommandConfig(ringrift_path="/opt/ringrift/ai-service")
        cmd = helper.build_p2p_restart_command(config)

        assert "pkill -f 'python.*p2p_orchestrator'" in cmd
        assert "cd /opt/ringrift/ai-service" in cmd
        assert "screen -dmS p2p" in cmd
        assert "PYTHONPATH=." in cmd
        assert "pgrep -f p2p_orchestrator" in cmd

    def test_build_health_check_command(self, helper: SSHRecoveryHelper) -> None:
        """Test health check command."""
        cmd = helper.build_health_check_command()

        assert "curl" in cmd
        assert "8770/status" in cmd

    # =========================================================================
    # Verification Tests
    # =========================================================================

    def test_verify_tailscale_output_success(
        self,
        helper: SSHRecoveryHelper,
    ) -> None:
        """Test Tailscale verification with valid IP."""
        assert helper.verify_tailscale_output("100.123.45.67")
        assert helper.verify_tailscale_output("Tailscale IP: 100.64.0.1")

    def test_verify_tailscale_output_failure(
        self,
        helper: SSHRecoveryHelper,
    ) -> None:
        """Test Tailscale verification without IP."""
        assert not helper.verify_tailscale_output("Connection failed")
        assert not helper.verify_tailscale_output("192.168.1.1")

    def test_verify_p2p_output_success(self, helper: SSHRecoveryHelper) -> None:
        """Test P2P verification with PID."""
        assert helper.verify_p2p_output("12345")
        assert helper.verify_p2p_output("Starting...\n54321\n")

    def test_verify_p2p_output_failure(self, helper: SSHRecoveryHelper) -> None:
        """Test P2P verification without PID."""
        assert not helper.verify_p2p_output("")
        assert not helper.verify_p2p_output("No process found")


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_ssh_recovery_helper_singleton(self) -> None:
        """Test singleton pattern."""
        helper1 = get_ssh_recovery_helper()
        helper2 = get_ssh_recovery_helper()
        assert helper1 is helper2

    @pytest.mark.asyncio
    async def test_execute_ssh_recovery(self) -> None:
        """Test convenience execute function."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"OK", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await execute_ssh_recovery(
                host="test-host",
                command="echo test",
                port=22,
                user="root",
            )

        assert result.success
        assert result.stdout == "OK"

    @pytest.mark.asyncio
    async def test_restart_tailscale_via_ssh(self) -> None:
        """Test Tailscale restart convenience function."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"100.64.0.1", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await restart_tailscale_via_ssh(
                host="test-host",
                hostname="test-node",
                is_container=False,
            )

        assert result.success

    @pytest.mark.asyncio
    async def test_restart_p2p_via_ssh(self) -> None:
        """Test P2P restart convenience function."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"12345", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await restart_p2p_via_ssh(
                host="test-host",
                port=22,
            )

        assert result.success
