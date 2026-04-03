"""Tests for container_tailscale_setup.py module.

December 2025: Tests for Container Tailscale Setup.
Used for P2P connectivity in Docker/Vast.ai/RunPod containers.
"""

import asyncio
import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.container_tailscale_setup import (
    ContainerNetworkStatus,
    ContainerTailscaleConfig,
    _check_socks5_proxy,
    _check_tailscale_installed,
    _check_tailscale_status,
    _is_tailscaled_running,
    _run_command,
    _start_tailscaled,
    detect_container_environment,
    ensure_userspace_tailscale,
    get_container_network_status,
    health_check,
    setup_container_networking,
    verify_tailscale_connectivity,
)


class TestContainerTailscaleConfig:
    """Tests for ContainerTailscaleConfig dataclass."""

    def test_default_values(self):
        """Test default values for ContainerTailscaleConfig."""
        config = ContainerTailscaleConfig()
        assert config.auth_key is None
        assert config.socks5_port == 1055
        assert config.timeout == 60
        assert config.accept_routes is True
        assert config.accept_dns is False

    def test_from_env_with_defaults(self):
        """Test from_env() with no environment variables set."""
        with patch.dict(os.environ, {}, clear=True):
            config = ContainerTailscaleConfig.from_env()
            assert config.auth_key is None
            assert config.socks5_port == 1055

    def test_from_env_with_auth_key(self):
        """Test from_env() with TAILSCALE_AUTH_KEY set."""
        with patch.dict(os.environ, {"TAILSCALE_AUTH_KEY": "tskey-12345"}, clear=True):
            config = ContainerTailscaleConfig.from_env()
            assert config.auth_key == "tskey-12345"

    def test_from_env_with_all_vars(self):
        """Test from_env() with all environment variables set."""
        env = {
            "TAILSCALE_AUTH_KEY": "tskey-abc123",
            "RINGRIFT_TAILSCALE_SOCKS_PORT": "2080",
            "RINGRIFT_TAILSCALE_TIMEOUT": "120",
            "RINGRIFT_TAILSCALE_ACCEPT_ROUTES": "false",
            "RINGRIFT_TAILSCALE_ACCEPT_DNS": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            config = ContainerTailscaleConfig.from_env()
            assert config.auth_key == "tskey-abc123"
            assert config.socks5_port == 2080
            assert config.timeout == 120
            assert config.accept_routes is False
            assert config.accept_dns is True

    def test_from_env_invalid_port(self):
        """Test from_env() with invalid SOCKS5 port raises ValueError."""
        with patch.dict(os.environ, {"RINGRIFT_TAILSCALE_SOCKS_PORT": "not-a-number"}, clear=True):
            with pytest.raises(ValueError):
                ContainerTailscaleConfig.from_env()


class TestContainerNetworkStatus:
    """Tests for ContainerNetworkStatus dataclass."""

    def test_default_values(self):
        """Test default values for ContainerNetworkStatus."""
        status = ContainerNetworkStatus()
        assert status.is_container is False
        assert status.container_type is None
        assert status.tailscale_installed is False
        assert status.tailscale_running is False
        assert status.tailscale_connected is False
        assert status.tailscale_ip is None
        assert status.socks5_available is False
        assert status.socks5_port == 1055
        assert status.error is None

    def test_is_ready_not_container(self):
        """Test is_ready property when not in container."""
        status = ContainerNetworkStatus(is_container=False)
        assert status.is_ready is True

    def test_is_ready_fully_configured(self):
        """Test is_ready property when fully configured (all requirements met)."""
        status = ContainerNetworkStatus(
            is_container=True,
            tailscale_installed=True,
            tailscale_running=True,
            tailscale_connected=True,
            socks5_available=True,
        )
        assert status.is_ready is True

    def test_is_ready_missing_installed(self):
        """Test is_ready property when tailscale not installed."""
        status = ContainerNetworkStatus(
            is_container=True,
            tailscale_installed=False,
            tailscale_running=True,
            tailscale_connected=True,
            socks5_available=True,
        )
        assert status.is_ready is False

    def test_is_ready_missing_socks5(self):
        """Test is_ready property when SOCKS5 not available."""
        status = ContainerNetworkStatus(
            is_container=True,
            tailscale_installed=True,
            tailscale_running=True,
            tailscale_connected=True,
            socks5_available=False,
        )
        assert status.is_ready is False

    def test_is_ready_not_connected(self):
        """Test is_ready property when not connected."""
        status = ContainerNetworkStatus(
            is_container=True,
            tailscale_connected=False,
        )
        assert status.is_ready is False


class TestDetectContainerEnvironment:
    """Tests for detect_container_environment() function."""

    def test_not_in_container(self):
        """Test detection when not in container."""
        with patch("os.path.exists", return_value=False):
            with patch("builtins.open", side_effect=FileNotFoundError()):
                with patch.dict(os.environ, {}, clear=True):
                    result = detect_container_environment()
                    assert result is None

    def test_docker_via_dockerenv(self):
        """Test Docker detection via /.dockerenv file."""
        def mock_exists(path):
            return path == "/.dockerenv"

        with patch("os.path.exists", side_effect=mock_exists):
            with patch.dict(os.environ, {}, clear=True):
                result = detect_container_environment()
                assert result == "docker"

    def test_podman_via_env(self):
        """Test Podman detection via environment variable."""
        with patch("os.path.exists", return_value=False):
            with patch("builtins.open", side_effect=FileNotFoundError()):
                with patch.dict(os.environ, {"container": "podman"}, clear=True):
                    result = detect_container_environment()
                    assert result == "podman"

    def test_kubernetes_via_env(self):
        """Test Kubernetes detection via KUBERNETES_SERVICE_HOST env var."""
        with patch("os.path.exists", return_value=False):
            with patch("builtins.open", side_effect=FileNotFoundError()):
                with patch.dict(os.environ, {"KUBERNETES_SERVICE_HOST": "10.0.0.1"}, clear=True):
                    result = detect_container_environment()
                    assert result == "kubernetes"

    def test_vastai_via_env(self):
        """Test Vast.ai detection returns docker (uses Docker runtime)."""
        with patch("os.path.exists", return_value=False):
            with patch("builtins.open", side_effect=FileNotFoundError()):
                with patch.dict(os.environ, {"VAST_CONTAINERLABEL": "vastai-123"}, clear=True):
                    result = detect_container_environment()
                    assert result == "docker"  # Vast.ai uses Docker

    def test_runpod_via_env(self):
        """Test RunPod detection returns docker (uses Docker runtime)."""
        with patch("os.path.exists", return_value=False):
            with patch("builtins.open", side_effect=FileNotFoundError()):
                with patch.dict(os.environ, {"RUNPOD_POD_ID": "pod-12345"}, clear=True):
                    result = detect_container_environment()
                    assert result == "docker"  # RunPod uses Docker

    def test_docker_via_cgroup(self):
        """Test Docker detection via /proc/1/cgroup."""
        mock_open = MagicMock()
        mock_open.return_value.__enter__.return_value.read.return_value = "1:name=docker:/docker/abc123"

        with patch("os.path.exists", return_value=False):
            with patch("builtins.open", mock_open):
                with patch.dict(os.environ, {}, clear=True):
                    result = detect_container_environment()
                    assert result == "docker"


class TestCheckTailscaleInstalled:
    """Tests for _check_tailscale_installed() function."""

    def test_tailscale_installed(self):
        """Test when Tailscale is installed."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/tailscale"
            assert _check_tailscale_installed() is True
            mock_which.assert_called_with("tailscale")

    def test_tailscale_not_installed(self):
        """Test when Tailscale is not installed."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None
            assert _check_tailscale_installed() is False


class TestRunCommand:
    """Tests for _run_command() async helper."""

    @pytest.mark.asyncio
    async def test_successful_command(self):
        """Test successful command execution."""
        mock_process = MagicMock()
        mock_process.communicate = AsyncMock(return_value=(b"output", b""))
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            returncode, stdout, stderr = await _run_command(["echo", "test"])
            assert returncode == 0
            assert stdout == "output"
            assert stderr == ""

    @pytest.mark.asyncio
    async def test_failed_command(self):
        """Test failed command execution."""
        mock_process = MagicMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b"error message"))
        mock_process.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            returncode, stdout, stderr = await _run_command(["false"])
            assert returncode == 1
            assert stderr == "error message"

    @pytest.mark.asyncio
    async def test_command_exception(self):
        """Test command execution with exception."""
        with patch("asyncio.create_subprocess_exec", side_effect=OSError("Command not found")):
            returncode, stdout, stderr = await _run_command(["nonexistent"])
            assert returncode == -1
            assert "Command not found" in stderr


class TestCheckTailscaleStatus:
    """Tests for _check_tailscale_status() function."""

    @pytest.mark.asyncio
    async def test_tailscale_connected(self):
        """Test when Tailscale is connected (parses JSON output)."""
        import json
        status_json = json.dumps({
            "TailscaleIPs": ["100.64.0.1"],
            "Self": {"Online": True}
        })
        with patch(
            "app.coordination.container_tailscale_setup._run_command",
            return_value=(0, status_json, ""),
        ):
            connected, ip = await _check_tailscale_status()
            assert connected is True
            assert ip == "100.64.0.1"

    @pytest.mark.asyncio
    async def test_tailscale_not_connected(self):
        """Test when Tailscale is not connected."""
        with patch(
            "app.coordination.container_tailscale_setup._run_command",
            return_value=(1, "", "Tailscale is stopped"),
        ):
            connected, ip = await _check_tailscale_status()
            assert connected is False
            assert ip is None

    @pytest.mark.asyncio
    async def test_tailscale_no_ip(self):
        """Test when Tailscale status JSON has no IPs."""
        import json
        status_json = json.dumps({
            "TailscaleIPs": [],
            "Self": {"Online": False}
        })
        with patch(
            "app.coordination.container_tailscale_setup._run_command",
            return_value=(0, status_json, ""),
        ):
            connected, ip = await _check_tailscale_status()
            assert connected is False
            assert ip is None

    @pytest.mark.asyncio
    async def test_tailscale_invalid_json(self):
        """Test when Tailscale status returns invalid JSON."""
        with patch(
            "app.coordination.container_tailscale_setup._run_command",
            return_value=(0, "not valid json", ""),
        ):
            connected, ip = await _check_tailscale_status()
            assert connected is False
            assert ip is None


class TestCheckSocks5Proxy:
    """Tests for _check_socks5_proxy() function."""

    @pytest.mark.asyncio
    async def test_socks5_available(self):
        """Test when SOCKS5 proxy is available."""
        mock_reader = MagicMock()
        mock_writer = MagicMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            result = await _check_socks5_proxy(1055)
            assert result is True
            mock_writer.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_socks5_not_available(self):
        """Test when SOCKS5 proxy is not available."""
        with patch("asyncio.open_connection", side_effect=ConnectionRefusedError()):
            result = await _check_socks5_proxy(1055)
            assert result is False

    @pytest.mark.asyncio
    async def test_socks5_timeout(self):
        """Test when SOCKS5 proxy connection times out."""
        with patch("asyncio.open_connection", side_effect=asyncio.TimeoutError()):
            result = await _check_socks5_proxy(1055)
            assert result is False


class TestIsTailscaledRunning:
    """Tests for _is_tailscaled_running() function."""

    @pytest.mark.asyncio
    async def test_tailscaled_running(self):
        """Test when tailscaled is running."""
        with patch(
            "app.coordination.container_tailscale_setup._run_command",
            return_value=(0, "12345 tailscaled", ""),
        ):
            result = await _is_tailscaled_running()
            assert result is True

    @pytest.mark.asyncio
    async def test_tailscaled_not_running(self):
        """Test when tailscaled is not running."""
        with patch(
            "app.coordination.container_tailscale_setup._run_command",
            return_value=(1, "", ""),
        ):
            result = await _is_tailscaled_running()
            assert result is False


class TestStartTailscaled:
    """Tests for _start_tailscaled() function."""

    @pytest.mark.asyncio
    async def test_tailscaled_not_installed(self):
        """Test when tailscaled is not installed."""
        with patch(
            "app.coordination.container_tailscale_setup._check_tailscaled_installed",
            return_value=False,
        ):
            config = ContainerTailscaleConfig()
            result = await _start_tailscaled(config)
            assert result is False

    @pytest.mark.asyncio
    async def test_already_running(self):
        """Test when tailscaled is already running."""
        with patch(
            "app.coordination.container_tailscale_setup._check_tailscaled_installed",
            return_value=True,
        ):
            with patch(
                "app.coordination.container_tailscale_setup._is_tailscaled_running",
                return_value=True,
            ):
                config = ContainerTailscaleConfig()
                result = await _start_tailscaled(config)
                assert result is True

    @pytest.mark.asyncio
    async def test_start_success(self):
        """Test successful tailscaled startup."""
        mock_process = MagicMock()
        mock_process.pid = 12345

        with patch(
            "app.coordination.container_tailscale_setup._check_tailscaled_installed",
            return_value=True,
        ):
            with patch(
                "app.coordination.container_tailscale_setup._is_tailscaled_running",
                side_effect=[False, True],  # Not running initially, then running after start
            ):
                with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                    with patch("asyncio.sleep", new_callable=AsyncMock):
                        with patch("pathlib.Path.exists", return_value=True):
                            with patch("pathlib.Path.mkdir"):
                                config = ContainerTailscaleConfig()
                                result = await _start_tailscaled(config)
                                assert result is True

    @pytest.mark.asyncio
    async def test_start_failure(self):
        """Test failed tailscaled startup (never becomes running)."""
        mock_process = MagicMock()
        mock_process.pid = 12345

        with patch(
            "app.coordination.container_tailscale_setup._check_tailscaled_installed",
            return_value=True,
        ):
            with patch(
                "app.coordination.container_tailscale_setup._is_tailscaled_running",
                return_value=False,  # Never becomes running
            ):
                with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                    with patch("asyncio.sleep", new_callable=AsyncMock):
                        with patch("pathlib.Path.exists", return_value=True):
                            config = ContainerTailscaleConfig()
                            result = await _start_tailscaled(config)
                            assert result is False


class TestEnsureUserspaceTailscale:
    """Tests for ensure_userspace_tailscale() function."""

    @pytest.mark.asyncio
    async def test_tailscale_not_installed(self):
        """Test when Tailscale is not installed."""
        with patch(
            "app.coordination.container_tailscale_setup._check_tailscale_installed",
            return_value=False,
        ):
            result = await ensure_userspace_tailscale()
            assert result is False

    @pytest.mark.asyncio
    async def test_tailscaled_start_failure(self):
        """Test when tailscaled fails to start."""
        with patch(
            "app.coordination.container_tailscale_setup._check_tailscale_installed",
            return_value=True,
        ):
            with patch(
                "app.coordination.container_tailscale_setup._start_tailscaled",
                return_value=False,
            ):
                result = await ensure_userspace_tailscale()
                assert result is False

    @pytest.mark.asyncio
    async def test_authentication_failure(self):
        """Test when Tailscale authentication fails."""
        with patch(
            "app.coordination.container_tailscale_setup._check_tailscale_installed",
            return_value=True,
        ):
            with patch(
                "app.coordination.container_tailscale_setup._start_tailscaled",
                return_value=True,
            ):
                with patch(
                    "app.coordination.container_tailscale_setup._authenticate_tailscale",
                    return_value=False,
                ):
                    result = await ensure_userspace_tailscale()
                    assert result is False

    @pytest.mark.asyncio
    async def test_socks5_not_available(self):
        """Test when SOCKS5 proxy doesn't become available."""
        with patch(
            "app.coordination.container_tailscale_setup._check_tailscale_installed",
            return_value=True,
        ):
            with patch(
                "app.coordination.container_tailscale_setup._start_tailscaled",
                return_value=True,
            ):
                with patch(
                    "app.coordination.container_tailscale_setup._authenticate_tailscale",
                    return_value=True,
                ):
                    with patch(
                        "app.coordination.container_tailscale_setup._check_socks5_proxy",
                        return_value=False,
                    ):
                        with patch("asyncio.sleep", new_callable=AsyncMock):
                            result = await ensure_userspace_tailscale()
                            assert result is False

    @pytest.mark.asyncio
    async def test_success(self):
        """Test successful userspace Tailscale setup."""
        with patch(
            "app.coordination.container_tailscale_setup._check_tailscale_installed",
            return_value=True,
        ):
            with patch(
                "app.coordination.container_tailscale_setup._start_tailscaled",
                return_value=True,
            ):
                with patch(
                    "app.coordination.container_tailscale_setup._authenticate_tailscale",
                    return_value=True,
                ):
                    with patch(
                        "app.coordination.container_tailscale_setup._check_socks5_proxy",
                        return_value=True,
                    ):
                        result = await ensure_userspace_tailscale()
                        assert result is True


class TestVerifyTailscaleConnectivity:
    """Tests for verify_tailscale_connectivity() function."""

    @pytest.mark.asyncio
    async def test_connected(self):
        """Test when Tailscale is connected."""
        with patch(
            "app.coordination.container_tailscale_setup._check_tailscale_status",
            return_value=(True, "100.64.0.1"),
        ):
            result = await verify_tailscale_connectivity()
            assert result is True

    @pytest.mark.asyncio
    async def test_not_connected(self):
        """Test when Tailscale is not connected."""
        with patch(
            "app.coordination.container_tailscale_setup._check_tailscale_status",
            return_value=(False, None),
        ):
            result = await verify_tailscale_connectivity()
            assert result is False


class TestSetupContainerNetworking:
    """Tests for setup_container_networking() function."""

    @pytest.mark.asyncio
    async def test_not_in_container(self):
        """Test when not running in container."""
        with patch(
            "app.coordination.container_tailscale_setup.detect_container_environment",
            return_value=None,
        ):
            success, message = await setup_container_networking()
            assert success is True
            assert "Not running in container" in message

    @pytest.mark.asyncio
    async def test_tailscale_not_installed(self):
        """Test when Tailscale is not installed in container."""
        with patch(
            "app.coordination.container_tailscale_setup.detect_container_environment",
            return_value="docker",
        ):
            with patch(
                "app.coordination.container_tailscale_setup._check_tailscale_installed",
                return_value=False,
            ):
                success, message = await setup_container_networking()
                assert success is False
                assert "not installed" in message

    @pytest.mark.asyncio
    async def test_kernel_tailscale_working(self):
        """Test when kernel Tailscale is already working."""
        with patch(
            "app.coordination.container_tailscale_setup.detect_container_environment",
            return_value="docker",
        ):
            with patch(
                "app.coordination.container_tailscale_setup._check_tailscale_installed",
                return_value=True,
            ):
                with patch(
                    "app.coordination.container_tailscale_setup._check_tailscale_status",
                    return_value=(True, "100.64.0.1"),
                ):
                    with patch(
                        "app.coordination.container_tailscale_setup._check_socks5_proxy",
                        return_value=False,
                    ):
                        success, message = await setup_container_networking()
                        assert success is True
                        assert "kernel mode" in message

    @pytest.mark.asyncio
    async def test_userspace_setup_success(self):
        """Test successful userspace Tailscale setup."""
        with patch(
            "app.coordination.container_tailscale_setup.detect_container_environment",
            return_value="vastai",
        ):
            with patch(
                "app.coordination.container_tailscale_setup._check_tailscale_installed",
                return_value=True,
            ):
                with patch(
                    "app.coordination.container_tailscale_setup._check_tailscale_status",
                    side_effect=[
                        (False, None),  # First check - not connected
                        (True, "100.64.0.2"),  # After setup - connected
                    ],
                ):
                    with patch(
                        "app.coordination.container_tailscale_setup.ensure_userspace_tailscale",
                        return_value=True,
                    ):
                        with patch(
                            "app.coordination.container_tailscale_setup._check_socks5_proxy",
                            return_value=True,
                        ):
                            success, message = await setup_container_networking()
                            assert success is True
                            assert "100.64.0.2" in message


class TestGetContainerNetworkStatus:
    """Tests for get_container_network_status() function."""

    @pytest.mark.asyncio
    async def test_cached_status(self):
        """Test that cached status is returned when fresh."""
        import app.coordination.container_tailscale_setup as module

        # Set up a fresh cached status
        cached = ContainerNetworkStatus(
            is_container=True,
            tailscale_connected=True,
            tailscale_ip="100.64.0.3",
            last_check=datetime.now(),
        )
        module._status_cache = cached

        result = await get_container_network_status()
        assert result.tailscale_ip == "100.64.0.3"

        # Cleanup
        module._status_cache = None

    @pytest.mark.asyncio
    async def test_fresh_check_when_not_container(self):
        """Test fresh check when not in container."""
        import app.coordination.container_tailscale_setup as module

        module._status_cache = None

        with patch(
            "app.coordination.container_tailscale_setup.detect_container_environment",
            return_value=None,
        ):
            result = await get_container_network_status()
            assert result.is_container is False

        # Cleanup
        module._status_cache = None


class TestHealthCheck:
    """Tests for health_check() function."""

    def test_not_container_no_cache(self):
        """Test health check when not in container (no cache)."""
        import app.coordination.container_tailscale_setup as module

        module._status_cache = None

        with patch(
            "app.coordination.container_tailscale_setup.detect_container_environment",
            return_value=None,
        ):
            result = health_check()
            assert result.healthy is True
            assert result.status.value == "running"
            assert "native networking" in result.message

    def test_container_no_cache(self):
        """Test health check in container without cache."""
        import app.coordination.container_tailscale_setup as module

        module._status_cache = None

        with patch(
            "app.coordination.container_tailscale_setup.detect_container_environment",
            return_value="docker",
        ):
            result = health_check()
            assert result.healthy is True
            assert result.status.value == "initializing"
            assert "async status check required" in result.message

    def test_cached_healthy_kernel_mode(self):
        """Test health check with cached healthy status (kernel mode)."""
        import app.coordination.container_tailscale_setup as module

        module._status_cache = ContainerNetworkStatus(
            is_container=True,
            container_type="docker",
            tailscale_connected=True,
            tailscale_ip="100.64.0.5",
            socks5_available=False,
        )

        result = health_check()
        assert result.healthy is True
        assert result.details["mode"] == "kernel"
        assert result.details["tailscale_ip"] == "100.64.0.5"

        # Cleanup
        module._status_cache = None

    def test_cached_healthy_userspace_mode(self):
        """Test health check with cached healthy status (userspace mode)."""
        import app.coordination.container_tailscale_setup as module

        module._status_cache = ContainerNetworkStatus(
            is_container=True,
            container_type="vastai",
            tailscale_connected=True,
            tailscale_ip="100.64.0.6",
            socks5_available=True,
            socks5_port=1055,
        )

        result = health_check()
        assert result.healthy is True
        assert result.details["mode"] == "userspace"
        assert result.details["socks5_port"] == 1055

        # Cleanup
        module._status_cache = None

    def test_cached_not_container(self):
        """Test health check with cached non-container status."""
        import app.coordination.container_tailscale_setup as module

        module._status_cache = ContainerNetworkStatus(
            is_container=False,
        )

        result = health_check()
        assert result.healthy is True
        assert "native networking" in result.message

        # Cleanup
        module._status_cache = None
