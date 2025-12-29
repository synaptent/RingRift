"""Container Tailscale Setup for P2P Connectivity.

Container environments (Docker on Vast.ai, RunPod, etc.) cannot use kernel Tailscale
and require userspace Tailscale with SOCKS5 proxy for P2P network access.

This module provides:
- Container environment detection
- Userspace Tailscale setup with SOCKS5 proxy
- Connectivity verification
- Health check integration

Usage:
    from app.coordination.container_tailscale_setup import (
        setup_container_networking,
        detect_container_environment,
        verify_tailscale_connectivity,
    )

    # Main entry point for automatic setup
    success, message = await setup_container_networking()

    # Or use individual functions
    container_type = detect_container_environment()
    if container_type:
        await ensure_userspace_tailscale()
        connected = await verify_tailscale_connectivity()

Environment Variables:
    TAILSCALE_AUTH_KEY: Tailscale auth key for container authentication
    RINGRIFT_SOCKS_PROXY: Set automatically to socks5://localhost:1055
    RINGRIFT_TAILSCALE_SOCKS_PORT: SOCKS5 proxy port (default: 1055)
    RINGRIFT_TAILSCALE_TIMEOUT: Setup timeout in seconds (default: 60)

December 2025
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__ = [
    "ContainerTailscaleConfig",
    "ContainerNetworkStatus",
    "detect_container_environment",
    "ensure_userspace_tailscale",
    "verify_tailscale_connectivity",
    "setup_container_networking",
    "get_container_network_status",
    "health_check",
]

logger = logging.getLogger(__name__)

# Default SOCKS5 proxy port for userspace Tailscale
DEFAULT_SOCKS5_PORT = 1055

# Timeout for Tailscale operations
DEFAULT_TIMEOUT = 60


@dataclass
class ContainerTailscaleConfig:
    """Configuration for container Tailscale setup."""

    socks5_port: int = DEFAULT_SOCKS5_PORT
    timeout: int = DEFAULT_TIMEOUT
    accept_routes: bool = True
    accept_dns: bool = False  # Container may have its own DNS
    auth_key: str | None = None

    @classmethod
    def from_env(cls) -> ContainerTailscaleConfig:
        """Create config from environment variables."""
        return cls(
            socks5_port=int(os.environ.get("RINGRIFT_TAILSCALE_SOCKS_PORT", str(DEFAULT_SOCKS5_PORT))),
            timeout=int(os.environ.get("RINGRIFT_TAILSCALE_TIMEOUT", str(DEFAULT_TIMEOUT))),
            accept_routes=os.environ.get("RINGRIFT_TAILSCALE_ACCEPT_ROUTES", "true").lower() in ("1", "true", "yes"),
            accept_dns=os.environ.get("RINGRIFT_TAILSCALE_ACCEPT_DNS", "false").lower() in ("1", "true", "yes"),
            auth_key=os.environ.get("TAILSCALE_AUTH_KEY"),
        )


@dataclass
class ContainerNetworkStatus:
    """Status of container networking setup."""

    is_container: bool = False
    container_type: str | None = None
    tailscale_installed: bool = False
    tailscale_running: bool = False
    tailscale_connected: bool = False
    socks5_available: bool = False
    socks5_port: int = DEFAULT_SOCKS5_PORT
    tailscale_ip: str | None = None
    last_check: datetime = field(default_factory=datetime.now)
    error: str | None = None

    @property
    def is_ready(self) -> bool:
        """Check if container networking is fully ready."""
        if not self.is_container:
            return True  # Not a container, networking is native
        return (
            self.tailscale_installed
            and self.tailscale_running
            and self.tailscale_connected
            and self.socks5_available
        )


# Global status cache
_status_cache: ContainerNetworkStatus | None = None


def detect_container_environment() -> str | None:
    """Detect if running in a container environment.

    Returns:
        Container type string ('docker', 'podman', 'lxc') or None if not in container.
    """
    # Check for Docker
    if os.path.exists("/.dockerenv"):
        return "docker"

    # Check cgroup for container indicators
    try:
        with open("/proc/1/cgroup", "r") as f:
            cgroup_content = f.read().lower()
            if "docker" in cgroup_content:
                return "docker"
            if "lxc" in cgroup_content:
                return "lxc"
            if "kubepods" in cgroup_content:
                return "kubernetes"
    except (FileNotFoundError, PermissionError):
        pass

    # Check for Podman
    if os.path.exists("/run/.containerenv"):
        return "podman"

    # Check environment variables set by container runtimes
    if os.environ.get("container") == "podman":
        return "podman"
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return "kubernetes"

    # Check for Vast.ai specific indicators
    if os.environ.get("VAST_CONTAINERLABEL"):
        return "docker"  # Vast.ai uses Docker

    # Check for RunPod specific indicators
    if os.environ.get("RUNPOD_POD_ID"):
        return "docker"  # RunPod uses Docker

    return None


def _check_tailscale_installed() -> bool:
    """Check if Tailscale is installed."""
    return shutil.which("tailscale") is not None


def _check_tailscaled_installed() -> bool:
    """Check if tailscaled daemon is installed."""
    return shutil.which("tailscaled") is not None


async def _run_command(
    cmd: list[str],
    timeout: int = 30,
    check: bool = False,
) -> tuple[int, str, str]:
    """Run a command asynchronously.

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout,
        )
        return (
            process.returncode or 0,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )
    except asyncio.TimeoutError:
        if process:
            process.kill()
        return -1, "", "Command timed out"
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"
    except (OSError, PermissionError) as e:
        return -1, "", str(e)


async def _check_tailscale_status() -> tuple[bool, str | None]:
    """Check Tailscale connection status.

    Returns:
        Tuple of (is_connected, tailscale_ip)
    """
    code, stdout, stderr = await _run_command(["tailscale", "status", "--json"], timeout=10)
    if code != 0:
        return False, None

    try:
        import json
        status = json.loads(stdout)
        # Check if we have a Tailscale IP
        tailscale_ips = status.get("TailscaleIPs", [])
        if tailscale_ips:
            return True, tailscale_ips[0]
        # Also check Self field
        self_info = status.get("Self", {})
        if self_info.get("Online"):
            ips = self_info.get("TailscaleIPs", [])
            return True, ips[0] if ips else None
    except (json.JSONDecodeError, KeyError, IndexError):
        pass

    return False, None


async def _check_socks5_proxy(port: int) -> bool:
    """Check if SOCKS5 proxy is available on the given port."""
    try:
        # Try to connect to the SOCKS5 proxy
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", port),
            timeout=5,
        )
        writer.close()
        await writer.wait_closed()
        return True
    except (ConnectionRefusedError, asyncio.TimeoutError, OSError):
        return False


async def _is_tailscaled_running() -> bool:
    """Check if tailscaled daemon is running."""
    code, stdout, _ = await _run_command(["pgrep", "-x", "tailscaled"], timeout=5)
    return code == 0 and stdout.strip() != ""


async def _start_tailscaled(config: ContainerTailscaleConfig) -> bool:
    """Start tailscaled in userspace mode with SOCKS5 proxy.

    Returns:
        True if tailscaled started successfully.
    """
    if not _check_tailscaled_installed():
        logger.error("tailscaled not installed")
        return False

    # Check if already running
    if await _is_tailscaled_running():
        logger.info("tailscaled already running")
        return True

    # Build command
    cmd = [
        "tailscaled",
        "--tun=userspace-networking",
        f"--socks5-server=localhost:{config.socks5_port}",
        "--state=/var/lib/tailscale/tailscaled.state",
        "--socket=/var/run/tailscale/tailscaled.sock",
    ]

    # Start tailscaled in background
    try:
        # Create state directory if needed
        state_dir = Path("/var/lib/tailscale")
        run_dir = Path("/var/run/tailscale")
        for d in [state_dir, run_dir]:
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)

        # Start as background process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,  # Detach from parent
        )

        # Wait a bit for startup
        await asyncio.sleep(2)

        # Verify it's running
        if await _is_tailscaled_running():
            logger.info(f"tailscaled started with SOCKS5 on port {config.socks5_port}")
            return True
        else:
            logger.error("tailscaled failed to start")
            return False

    except (OSError, PermissionError) as e:
        logger.error(f"Failed to start tailscaled: {e}")
        return False


async def _authenticate_tailscale(config: ContainerTailscaleConfig) -> bool:
    """Authenticate Tailscale with auth key.

    Returns:
        True if authentication successful.
    """
    if not config.auth_key:
        logger.warning("No TAILSCALE_AUTH_KEY set, skipping authentication")
        # Check if already authenticated
        connected, _ = await _check_tailscale_status()
        return connected

    cmd = ["tailscale", "up", f"--auth-key={config.auth_key}"]

    if config.accept_routes:
        cmd.append("--accept-routes")
    if config.accept_dns:
        cmd.append("--accept-dns")

    code, stdout, stderr = await _run_command(cmd, timeout=config.timeout)

    if code == 0:
        logger.info("Tailscale authenticated successfully")
        return True
    else:
        # Tailscale up returns non-zero if already authenticated
        # Check actual status
        connected, _ = await _check_tailscale_status()
        if connected:
            logger.info("Tailscale already authenticated")
            return True
        logger.error(f"Tailscale authentication failed: {stderr}")
        return False


async def ensure_userspace_tailscale(
    config: ContainerTailscaleConfig | None = None,
) -> bool:
    """Ensure Tailscale is running in userspace mode with SOCKS5 proxy.

    Args:
        config: Optional configuration. Uses environment defaults if not provided.

    Returns:
        True if Tailscale is ready for use.
    """
    if config is None:
        config = ContainerTailscaleConfig.from_env()

    # Step 1: Check/install Tailscale
    if not _check_tailscale_installed():
        logger.error("Tailscale not installed. Run: curl -fsSL https://tailscale.com/install.sh | sh")
        return False

    # Step 2: Start tailscaled
    if not await _start_tailscaled(config):
        return False

    # Step 3: Authenticate
    if not await _authenticate_tailscale(config):
        return False

    # Step 4: Wait for SOCKS5 proxy
    for _ in range(10):
        if await _check_socks5_proxy(config.socks5_port):
            logger.info(f"SOCKS5 proxy ready on port {config.socks5_port}")
            # Set environment variable for P2P
            os.environ["RINGRIFT_SOCKS_PROXY"] = f"socks5://localhost:{config.socks5_port}"
            return True
        await asyncio.sleep(1)

    logger.error(f"SOCKS5 proxy not available on port {config.socks5_port}")
    return False


async def verify_tailscale_connectivity() -> bool:
    """Verify Tailscale connectivity to P2P network.

    Returns:
        True if connected and can reach P2P peers.
    """
    connected, ip = await _check_tailscale_status()
    if not connected:
        logger.warning("Tailscale not connected")
        return False

    logger.info(f"Tailscale connected with IP: {ip}")
    return True


async def setup_container_networking(
    config: ContainerTailscaleConfig | None = None,
) -> tuple[bool, str]:
    """Main entry point for container networking setup.

    Detects container environment and sets up userspace Tailscale if needed.

    Args:
        config: Optional configuration. Uses environment defaults if not provided.

    Returns:
        Tuple of (success, message) describing the result.
    """
    global _status_cache

    # Initialize status
    status = ContainerNetworkStatus()

    # Detect container environment
    container_type = detect_container_environment()
    status.container_type = container_type
    status.is_container = container_type is not None

    if not status.is_container:
        status.tailscale_connected = True  # Native networking works
        _status_cache = status
        return True, "Not running in container, native networking available"

    logger.info(f"Detected container environment: {container_type}")

    if config is None:
        config = ContainerTailscaleConfig.from_env()

    status.socks5_port = config.socks5_port

    # Check Tailscale installation
    status.tailscale_installed = _check_tailscale_installed()
    if not status.tailscale_installed:
        status.error = "Tailscale not installed"
        _status_cache = status
        return False, "Tailscale not installed. Run: curl -fsSL https://tailscale.com/install.sh | sh"

    # Setup userspace Tailscale
    if not await ensure_userspace_tailscale(config):
        status.error = "Failed to setup userspace Tailscale"
        _status_cache = status
        return False, "Failed to setup userspace Tailscale with SOCKS5 proxy"

    status.tailscale_running = True

    # Verify connectivity
    connected, ip = await _check_tailscale_status()
    status.tailscale_connected = connected
    status.tailscale_ip = ip

    if not connected:
        status.error = "Tailscale not connected to network"
        _status_cache = status
        return False, "Tailscale daemon running but not connected to network"

    # Verify SOCKS5
    status.socks5_available = await _check_socks5_proxy(config.socks5_port)
    if not status.socks5_available:
        status.error = f"SOCKS5 proxy not available on port {config.socks5_port}"
        _status_cache = status
        return False, f"SOCKS5 proxy not available on port {config.socks5_port}"

    status.last_check = datetime.now()
    _status_cache = status

    return True, f"Container networking ready (Tailscale IP: {ip}, SOCKS5: localhost:{config.socks5_port})"


async def get_container_network_status() -> ContainerNetworkStatus:
    """Get current container network status.

    Returns cached status if available, otherwise performs fresh check.
    """
    global _status_cache

    if _status_cache is not None:
        # Refresh if older than 60 seconds
        age = (datetime.now() - _status_cache.last_check).total_seconds()
        if age < 60:
            return _status_cache

    # Perform fresh check
    status = ContainerNetworkStatus()
    status.container_type = detect_container_environment()
    status.is_container = status.container_type is not None

    if status.is_container:
        status.tailscale_installed = _check_tailscale_installed()
        status.tailscale_running = await _is_tailscaled_running()
        status.tailscale_connected, status.tailscale_ip = await _check_tailscale_status()

        config = ContainerTailscaleConfig.from_env()
        status.socks5_port = config.socks5_port
        status.socks5_available = await _check_socks5_proxy(config.socks5_port)

    status.last_check = datetime.now()
    _status_cache = status
    return status


def health_check() -> dict:
    """Synchronous health check for daemon integration.

    Returns:
        Health check result dict compatible with DaemonManager.
    """
    # Use cached status if available
    if _status_cache is not None:
        status = _status_cache
    else:
        # Quick sync check
        container_type = detect_container_environment()
        if container_type is None:
            return {
                "healthy": True,
                "status": "healthy",
                "message": "Not a container, native networking",
                "details": {"is_container": False},
            }

        # For containers, we need async check - return unknown
        return {
            "healthy": None,
            "status": "unknown",
            "message": "Container detected, async status check required",
            "details": {"is_container": True, "container_type": container_type},
        }

    if not status.is_container:
        return {
            "healthy": True,
            "status": "healthy",
            "message": "Not a container, native networking",
            "details": {"is_container": False},
        }

    if status.is_ready:
        return {
            "healthy": True,
            "status": "healthy",
            "message": f"Container networking ready (IP: {status.tailscale_ip})",
            "details": {
                "is_container": True,
                "container_type": status.container_type,
                "tailscale_ip": status.tailscale_ip,
                "socks5_port": status.socks5_port,
            },
        }

    return {
        "healthy": False,
        "status": "unhealthy",
        "message": status.error or "Container networking not ready",
        "details": {
            "is_container": True,
            "container_type": status.container_type,
            "tailscale_installed": status.tailscale_installed,
            "tailscale_running": status.tailscale_running,
            "tailscale_connected": status.tailscale_connected,
            "socks5_available": status.socks5_available,
            "error": status.error,
        },
    }
