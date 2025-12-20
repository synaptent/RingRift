"""Tests for scripts.lib.ssh module.

Tests SSH command execution utilities.
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from scripts.lib.ssh import (
    SSHConfig,
    SSHResult,
    run_ssh_command,
    run_ssh_command_with_config,
)


class TestSSHConfig:
    """Tests for SSHConfig dataclass."""

    def test_default_values(self):
        """Test SSHConfig with default values."""
        config = SSHConfig(host="example.com")

        assert config.host == "example.com"
        assert config.port == 22
        assert config.user == "root"
        assert config.ssh_key is None
        assert config.connect_timeout == 10
        assert config.batch_mode is True

    def test_custom_values(self):
        """Test SSHConfig with custom values."""
        config = SSHConfig(
            host="192.168.1.1",
            port=2222,
            user="ubuntu",
            ssh_key="~/.ssh/custom_key",
            connect_timeout=30,
            batch_mode=False,
        )

        assert config.host == "192.168.1.1"
        assert config.port == 2222
        assert config.user == "ubuntu"
        assert config.ssh_key == "~/.ssh/custom_key"
        assert config.connect_timeout == 30
        assert config.batch_mode is False

    def test_from_dict(self):
        """Test creating SSHConfig from dictionary."""
        data = {
            "ssh_host": "10.0.0.1",
            "ssh_port": 22022,
            "ssh_user": "admin",
        }
        config = SSHConfig.from_dict(data)

        assert config.host == "10.0.0.1"
        assert config.port == 22022
        assert config.user == "admin"

    def test_from_dict_with_tailscale_fallback(self):
        """Test from_dict uses tailscale_ip as fallback."""
        data = {
            "tailscale_ip": "100.64.0.1",
            "ssh_user": "ubuntu",
        }
        config = SSHConfig.from_dict(data)

        assert config.host == "100.64.0.1"

    def test_build_ssh_args_basic(self):
        """Test building SSH command arguments."""
        config = SSHConfig(host="example.com", user="ubuntu")
        args = config.build_ssh_args()

        assert "ssh" in args
        assert "ubuntu@example.com" in args
        assert "-o" in args
        assert "BatchMode=yes" in args

    def test_build_ssh_args_with_key(self):
        """Test SSH args include key path."""
        config = SSHConfig(
            host="example.com",
            ssh_key="/path/to/key",
        )
        args = config.build_ssh_args()

        assert "-i" in args
        key_index = args.index("-i")
        assert args[key_index + 1] == "/path/to/key"

    def test_build_ssh_args_with_custom_port(self):
        """Test SSH args include custom port."""
        config = SSHConfig(host="example.com", port=2222)
        args = config.build_ssh_args()

        assert "-p" in args
        port_index = args.index("-p")
        assert args[port_index + 1] == "2222"

    def test_build_ssh_args_default_port_omitted(self):
        """Test default port 22 is not explicitly added."""
        config = SSHConfig(host="example.com", port=22)
        args = config.build_ssh_args()

        assert "-p" not in args


class TestSSHResult:
    """Tests for SSHResult dataclass."""

    def test_success_result(self):
        """Test successful SSH result."""
        result = SSHResult(
            success=True,
            output="uptime: 10 days",
            exit_code=0,
        )

        assert result.success is True
        assert result.output == "uptime: 10 days"
        assert result.exit_code == 0
        assert result.error is None

    def test_failure_result(self):
        """Test failed SSH result."""
        result = SSHResult(
            success=False,
            output="",
            exit_code=1,
            error="Connection refused",
        )

        assert result.success is False
        assert result.exit_code == 1
        assert result.error == "Connection refused"


class TestRunSSHCommand:
    """Tests for run_ssh_command function."""

    @patch("scripts.lib.ssh.subprocess.run")
    def test_successful_command(self, mock_run):
        """Test successful SSH command execution."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Hello World",
            stderr="",
        )

        success, output = run_ssh_command("example.com", "echo Hello World")

        assert success is True
        assert "Hello World" in output
        mock_run.assert_called_once()

    @patch("scripts.lib.ssh.subprocess.run")
    def test_failed_command(self, mock_run):
        """Test failed SSH command execution."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Permission denied",
        )

        success, output = run_ssh_command("example.com", "restricted_command")

        assert success is False
        assert "Permission denied" in output

    @patch("scripts.lib.ssh.subprocess.run")
    def test_timeout_handling(self, mock_run):
        """Test SSH command timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=30)

        success, output = run_ssh_command("example.com", "long_command", timeout=30)

        assert success is False
        assert "timeout" in output.lower()

    @patch("scripts.lib.ssh.subprocess.run")
    def test_connection_error(self, mock_run):
        """Test SSH connection error."""
        mock_run.return_value = MagicMock(
            returncode=255,
            stdout="",
            stderr="Connection refused",
        )

        success, _output = run_ssh_command("example.com", "echo test")

        assert success is False

    @patch("scripts.lib.ssh.subprocess.run")
    def test_with_custom_user(self, mock_run):
        """Test SSH with custom user."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="ok",
            stderr="",
        )

        run_ssh_command("example.com", "whoami", user="admin")

        # Verify the call included the user
        call_args = mock_run.call_args
        cmd = call_args[0][0] if call_args[0] else call_args[1].get("args", [])
        assert any("admin@example.com" in str(arg) for arg in cmd)

    @patch("scripts.lib.ssh.subprocess.run")
    def test_with_custom_port(self, mock_run):
        """Test SSH with custom port."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="ok",
            stderr="",
        )

        run_ssh_command("example.com", "uptime", port=2222)

        call_args = mock_run.call_args
        cmd = call_args[0][0] if call_args[0] else call_args[1].get("args", [])
        # Check that -p 2222 is in the command
        assert "-p" in cmd
        assert "2222" in cmd


class TestRunSSHCommandWithConfig:
    """Tests for run_ssh_command_with_config function."""

    @patch("scripts.lib.ssh.subprocess.run")
    def test_with_config(self, mock_run):
        """Test SSH command with SSHConfig."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="GPU available",
            stderr="",
        )

        config = SSHConfig(
            host="gpu-server",
            port=22,
            user="ubuntu",
        )
        success, output = run_ssh_command_with_config(config, "nvidia-smi")

        assert success is True
        assert "GPU available" in output

    @patch("scripts.lib.ssh.subprocess.run")
    def test_config_with_ssh_key(self, mock_run):
        """Test SSH command with key from config."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="ok",
            stderr="",
        )

        config = SSHConfig(
            host="secure-server",
            ssh_key="/home/user/.ssh/id_rsa",
        )
        run_ssh_command_with_config(config, "ls")

        call_args = mock_run.call_args
        cmd = call_args[0][0] if call_args[0] else call_args[1].get("args", [])
        assert "-i" in cmd
