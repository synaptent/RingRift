"""TailscaleHealthDaemon - Local daemon for monitoring and auto-recovering Tailscale.

This daemon runs on each cluster node to ensure Tailscale connectivity is maintained.
It detects when Tailscale disconnects and attempts automatic recovery.

Key Features:
- Monitors Tailscale connectivity every 30 seconds (configurable)
- Automatically restarts tailscaled when it crashes
- Re-authenticates with preserved state when possible
- Supports userspace networking for containers (Vast.ai, RunPod)
- Reports status to P2P orchestrator via HTTP

Recovery Strategy:
1. Try `tailscale up` (reconnect with existing auth)
2. If that fails, restart tailscaled daemon
3. If container, use userspace networking mode
4. Log failure for manual intervention if all recovery fails

December 2025 - Created as part of P2P stability improvements.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from app.coordination.contracts import HealthCheckResult
from app.coordination.handler_base import HandlerBase, HandlerStats

logger = logging.getLogger(__name__)


class TailscaleStatus(Enum):
    """Tailscale connection status."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    STARTING = "starting"
    NEEDS_AUTH = "needs_auth"
    UNKNOWN = "unknown"
    NOT_INSTALLED = "not_installed"


@dataclass
class TailscaleState:
    """Current state of Tailscale on this node."""

    status: TailscaleStatus = TailscaleStatus.UNKNOWN
    tailscale_ip: str | None = None
    hostname: str | None = None
    backend_state: str | None = None
    is_online: bool = False
    is_running: bool = False
    last_check_time: float = 0.0
    last_connected_time: float = 0.0
    recovery_attempts: int = 0
    last_recovery_time: float = 0.0
    error_message: str | None = None


@dataclass
class TailscaleHealthConfig:
    """Configuration for TailscaleHealthDaemon."""

    # Check intervals
    check_interval_seconds: float = 30.0
    health_check_timeout_seconds: float = 10.0

    # Recovery settings
    max_recovery_attempts: int = 3
    recovery_cooldown_seconds: float = 300.0  # 5 minutes between recovery attempts

    # Container/userspace networking
    use_userspace_networking: bool = True  # For Vast.ai/RunPod containers
    tailscale_state_dir: str = "/var/lib/tailscale"
    tailscale_socket_dir: str = "/var/run/tailscale"

    # P2P reporting
    report_to_p2p: bool = True
    p2p_status_endpoint: str = "http://localhost:8770/tailscale_health"

    # Environment overrides
    @classmethod
    def from_env(cls) -> TailscaleHealthConfig:
        """Create config from environment variables."""
        return cls(
            check_interval_seconds=float(
                os.environ.get("RINGRIFT_TAILSCALE_CHECK_INTERVAL", "30.0")
            ),
            max_recovery_attempts=int(
                os.environ.get("RINGRIFT_TAILSCALE_MAX_RECOVERY", "3")
            ),
            use_userspace_networking=os.environ.get(
                "RINGRIFT_TAILSCALE_USERSPACE", "true"
            ).lower() == "true",
            report_to_p2p=os.environ.get(
                "RINGRIFT_TAILSCALE_REPORT_P2P", "true"
            ).lower() == "true",
        )


class TailscaleHealthDaemon(HandlerBase):
    """Daemon that monitors and auto-recovers Tailscale connectivity.

    This daemon runs locally on each cluster node and:
    1. Checks Tailscale status every check_interval_seconds
    2. Detects disconnections or daemon crashes
    3. Attempts automatic recovery
    4. Reports status to P2P orchestrator

    Usage:
        # On each cluster node
        daemon = TailscaleHealthDaemon.get_instance()
        await daemon.start()

    Environment Variables:
        RINGRIFT_TAILSCALE_CHECK_INTERVAL: Check interval in seconds (default: 30)
        RINGRIFT_TAILSCALE_MAX_RECOVERY: Max recovery attempts (default: 3)
        RINGRIFT_TAILSCALE_USERSPACE: Use userspace networking (default: true)
        RINGRIFT_TAILSCALE_REPORT_P2P: Report to P2P (default: true)
    """

    def __init__(self, config: TailscaleHealthConfig | None = None):
        """Initialize daemon."""
        self._ts_config = config or TailscaleHealthConfig.from_env()
        super().__init__(
            name="tailscale_health",
            config=self._ts_config,
            cycle_interval=self._ts_config.check_interval_seconds,
        )
        self._state = TailscaleState()
        self._recovery_in_progress = False

    # =========================================================================
    # Core Interface (HandlerBase)
    # =========================================================================

    async def _run_cycle(self) -> None:
        """Main work loop - check Tailscale and recover if needed."""
        try:
            # Check current status
            await self._check_tailscale_status()

            # Attempt recovery if disconnected
            if self._state.status in (
                TailscaleStatus.DISCONNECTED,
                TailscaleStatus.NEEDS_AUTH,
            ):
                await self._attempt_recovery()

            # Report to P2P
            if self._ts_config.report_to_p2p:
                await self._report_status_to_p2p()

            self._stats.cycles_completed += 1
            self._stats.last_activity = time.time()

        except Exception as e:
            # Dec 29, 2025: Intentionally broad - daemon must keep running
            # Individual methods have their own narrowed exception handlers;
            # this catches any unexpected errors to maintain daemon continuity.
            # Note: Exception (not BaseException) already excludes SystemExit,
            # KeyboardInterrupt, and GeneratorExit.
            self._stats.errors_count += 1
            self._stats.last_error = str(e)
            self._stats.last_error_time = time.time()
            logger.error(f"Tailscale health check failed: {e}")

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """No event subscriptions needed - this is a polling daemon."""
        return {}

    def health_check(self) -> HealthCheckResult:
        """Return health check result."""
        is_healthy = self._state.status == TailscaleStatus.CONNECTED

        details = {
            "status": self._state.status.value,
            "tailscale_ip": self._state.tailscale_ip,
            "hostname": self._state.hostname,
            "is_online": self._state.is_online,
            "recovery_attempts": self._state.recovery_attempts,
            "last_check_time": self._state.last_check_time,
            "cycles_completed": self._stats.cycles_completed,
        }

        return HealthCheckResult(
            healthy=is_healthy,
            message=f"Tailscale {self._state.status.value}",
            details=details,
        )

    # =========================================================================
    # Tailscale Status Checking
    # =========================================================================

    async def _check_tailscale_status(self) -> None:
        """Check current Tailscale status."""
        self._state.last_check_time = time.time()

        # Check if tailscale binary exists
        tailscale_path = shutil.which("tailscale")
        if not tailscale_path:
            self._state.status = TailscaleStatus.NOT_INSTALLED
            self._state.is_online = False
            self._state.is_running = False
            logger.warning("Tailscale not installed on this node")
            return

        # Check if tailscaled is running
        tailscaled_running = await self._is_tailscaled_running()
        self._state.is_running = tailscaled_running

        if not tailscaled_running:
            self._state.status = TailscaleStatus.DISCONNECTED
            self._state.is_online = False
            logger.warning("tailscaled is not running")
            return

        # Get detailed status
        try:
            result = await asyncio.wait_for(
                self._run_command(["tailscale", "status", "--json"]),
                timeout=self._ts_config.health_check_timeout_seconds,
            )

            if result["returncode"] != 0:
                # Check for "not running" message
                if "not running" in result["stderr"].lower():
                    self._state.status = TailscaleStatus.DISCONNECTED
                    self._state.is_online = False
                elif "logged out" in result["stderr"].lower():
                    self._state.status = TailscaleStatus.NEEDS_AUTH
                    self._state.is_online = False
                else:
                    self._state.status = TailscaleStatus.UNKNOWN
                    self._state.error_message = result["stderr"]
                return

            # Parse JSON status
            status_data = json.loads(result["stdout"])
            self._parse_status_json(status_data)

        except asyncio.TimeoutError:
            self._state.status = TailscaleStatus.UNKNOWN
            self._state.error_message = "Status check timed out"
            logger.warning("Tailscale status check timed out")
        except json.JSONDecodeError as e:
            self._state.status = TailscaleStatus.UNKNOWN
            self._state.error_message = f"Invalid JSON: {e}"
            logger.warning(f"Failed to parse Tailscale status: {e}")
        except (KeyError, TypeError, OSError) as e:
            # Dec 29, 2025: Narrowed from bare Exception
            # - KeyError: accessing missing dict keys from _run_command result
            # - TypeError: if result is unexpected type
            # - OSError: system/process errors from command execution
            self._state.status = TailscaleStatus.UNKNOWN
            self._state.error_message = str(e)
            logger.error(f"Error checking Tailscale status: {e}")

    def _parse_status_json(self, data: dict[str, Any]) -> None:
        """Parse Tailscale status JSON."""
        backend_state = data.get("BackendState", "Unknown")
        self._state.backend_state = backend_state

        if backend_state == "Running":
            self._state.status = TailscaleStatus.CONNECTED
            self._state.is_online = True
            self._state.last_connected_time = time.time()
            self._state.recovery_attempts = 0  # Reset on success
        elif backend_state == "NeedsLogin":
            self._state.status = TailscaleStatus.NEEDS_AUTH
            self._state.is_online = False
        elif backend_state == "Starting":
            self._state.status = TailscaleStatus.STARTING
            self._state.is_online = False
        else:
            self._state.status = TailscaleStatus.DISCONNECTED
            self._state.is_online = False

        # Get self info
        if "Self" in data:
            self_info = data["Self"]
            tailscale_ips = self_info.get("TailscaleIPs", [])
            self._state.tailscale_ip = tailscale_ips[0] if tailscale_ips else None
            self._state.hostname = self_info.get("HostName")

    async def _is_tailscaled_running(self) -> bool:
        """Check if tailscaled process is running."""
        try:
            result = await self._run_command(["pgrep", "-x", "tailscaled"])
            return result["returncode"] == 0
        except (OSError, asyncio.TimeoutError, KeyError) as e:
            # Dec 29, 2025: Narrowed from bare Exception
            # - OSError: subprocess/process errors
            # - TimeoutError: command timeout (though _run_command handles this)
            # - KeyError: defensive if result dict changes
            logger.debug(f"Error checking tailscaled: {e}")
            return False

    # =========================================================================
    # Recovery Logic
    # =========================================================================

    async def _attempt_recovery(self) -> bool:
        """Attempt to recover Tailscale connectivity.

        Returns:
            True if recovery succeeded, False otherwise
        """
        if self._recovery_in_progress:
            logger.debug("Recovery already in progress, skipping")
            return False

        # Check recovery cooldown
        if self._state.last_recovery_time:
            elapsed = time.time() - self._state.last_recovery_time
            if elapsed < self._ts_config.recovery_cooldown_seconds:
                logger.debug(
                    f"Recovery cooldown active, {elapsed:.0f}s since last attempt"
                )
                return False

        # Check max attempts
        if self._state.recovery_attempts >= self._ts_config.max_recovery_attempts:
            logger.warning(
                f"Max recovery attempts ({self._ts_config.max_recovery_attempts}) "
                f"reached, giving up"
            )
            return False

        self._recovery_in_progress = True
        self._state.recovery_attempts += 1
        self._state.last_recovery_time = time.time()

        try:
            logger.info(
                f"Attempting Tailscale recovery (attempt "
                f"{self._state.recovery_attempts}/{self._ts_config.max_recovery_attempts})"
            )

            # Strategy 1: Try tailscale up (reconnect with existing auth)
            if await self._try_tailscale_up():
                logger.info("Tailscale recovered via 'tailscale up'")
                return True

            # Strategy 2: Restart tailscaled daemon
            if await self._restart_tailscaled():
                # Wait for startup then try up again
                await asyncio.sleep(5)
                if await self._try_tailscale_up():
                    logger.info("Tailscale recovered after daemon restart")
                    return True

            # Strategy 3: Full restart with userspace networking (containers)
            if self._ts_config.use_userspace_networking:
                if await self._restart_tailscaled_userspace():
                    await asyncio.sleep(5)
                    if await self._try_tailscale_up():
                        logger.info("Tailscale recovered with userspace networking")
                        return True

            logger.error("All Tailscale recovery strategies failed")
            return False

        finally:
            self._recovery_in_progress = False

    async def _try_tailscale_up(self) -> bool:
        """Try to bring Tailscale up with existing authentication."""
        try:
            result = await asyncio.wait_for(
                self._run_command(
                    ["tailscale", "up", "--accept-routes"],
                    timeout=30,
                ),
                timeout=35,
            )

            if result["returncode"] == 0:
                # Verify connection
                await asyncio.sleep(2)
                await self._check_tailscale_status()
                return self._state.status == TailscaleStatus.CONNECTED

            logger.warning(f"tailscale up failed: {result['stderr']}")
            return False

        except asyncio.TimeoutError:
            logger.warning("tailscale up timed out")
            return False
        except (KeyError, TypeError, OSError) as e:
            # Dec 29, 2025: Narrowed from bare Exception
            # - KeyError: accessing result dict keys
            # - TypeError: unexpected result types
            # - OSError: process/system errors
            logger.error(f"tailscale up error: {e}")
            return False

    async def _restart_tailscaled(self) -> bool:
        """Restart the tailscaled daemon."""
        try:
            # Try systemctl first
            result = await self._run_command(
                ["systemctl", "restart", "tailscaled"],
                timeout=30,
            )
            if result["returncode"] == 0:
                return True

            # Fallback: kill and start manually
            await self._run_command(["pkill", "-9", "tailscaled"])
            await asyncio.sleep(2)

            # Start tailscaled
            result = await self._run_command(
                ["tailscaled", "--state=/var/lib/tailscale/tailscaled.state"],
                timeout=10,
                background=True,
            )
            return True

        except (KeyError, OSError, asyncio.TimeoutError) as e:
            # Dec 29, 2025: Narrowed from bare Exception
            # - KeyError: accessing result dict keys
            # - OSError: process/permission errors
            # - TimeoutError: command timeout
            logger.error(f"Failed to restart tailscaled: {e}")
            return False

    async def _restart_tailscaled_userspace(self) -> bool:
        """Restart tailscaled in userspace networking mode (for containers)."""
        try:
            # Kill any existing tailscaled
            await self._run_command(["pkill", "-9", "tailscaled"])
            await asyncio.sleep(2)

            # Ensure directories exist
            Path(self._ts_config.tailscale_state_dir).mkdir(
                parents=True, exist_ok=True
            )
            Path(self._ts_config.tailscale_socket_dir).mkdir(
                parents=True, exist_ok=True
            )

            # Start in userspace mode
            cmd = [
                "tailscaled",
                "--tun=userspace-networking",
                f"--statedir={self._ts_config.tailscale_state_dir}",
            ]

            result = await self._run_command(cmd, timeout=10, background=True)
            logger.info("Started tailscaled in userspace mode")
            return True

        except (KeyError, OSError, PermissionError, asyncio.TimeoutError) as e:
            # Dec 29, 2025: Narrowed from bare Exception
            # - KeyError: accessing result dict keys
            # - OSError: process/file errors
            # - PermissionError: directory creation failures
            # - TimeoutError: command timeout
            logger.error(f"Failed to start tailscaled in userspace mode: {e}")
            return False

    # =========================================================================
    # P2P Reporting
    # =========================================================================

    async def _report_status_to_p2p(self) -> None:
        """Report Tailscale status to P2P orchestrator."""
        try:
            import aiohttp

            status_data = {
                "status": self._state.status.value,
                "tailscale_ip": self._state.tailscale_ip,
                "hostname": self._state.hostname,
                "is_online": self._state.is_online,
                "is_running": self._state.is_running,
                "recovery_attempts": self._state.recovery_attempts,
                "last_check_time": self._state.last_check_time,
                "timestamp": time.time(),
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._ts_config.p2p_status_endpoint,
                    json=status_data,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status != 200:
                        logger.debug(
                            f"P2P status report failed: {resp.status}"
                        )

        except ImportError:
            logger.debug("aiohttp not available, skipping P2P report")
        except asyncio.TimeoutError:
            # Dec 29, 2025: Narrowed from bare Exception
            logger.debug("P2P report timed out")
        except OSError as e:
            # Socket/network level errors
            logger.debug(f"Failed to report to P2P (network): {e}")
        except aiohttp.ClientError as e:
            # Dec 29, 2025: Added explicit aiohttp error handling
            # aiohttp.ClientError is not a subclass of OSError
            logger.debug(f"Failed to report to P2P (client): {e}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def _run_command(
        self,
        cmd: list[str],
        timeout: float = 30,
        background: bool = False,
    ) -> dict[str, Any]:
        """Run a shell command asynchronously."""
        try:
            if background:
                # Fire and forget
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                return {"returncode": 0, "stdout": "", "stderr": ""}

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )

            return {
                "returncode": proc.returncode,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
            }

        except asyncio.TimeoutError:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": "Command timed out",
            }
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            # Dec 29, 2025: Narrowed from bare Exception
            # - OSError: process/system errors, file not found
            # - SubprocessError: Popen failures
            # - ValueError: invalid arguments
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
            }

    # =========================================================================
    # State Access
    # =========================================================================

    def get_state(self) -> TailscaleState:
        """Get current Tailscale state."""
        return self._state

    def get_tailscale_ip(self) -> str | None:
        """Get current Tailscale IP address."""
        return self._state.tailscale_ip


# =============================================================================
# Factory Functions
# =============================================================================

def get_tailscale_health_daemon(
    config: TailscaleHealthConfig | None = None,
) -> TailscaleHealthDaemon:
    """Get or create the TailscaleHealthDaemon singleton."""
    return TailscaleHealthDaemon.get_instance()  # type: ignore


def create_tailscale_health_daemon(
    config: TailscaleHealthConfig | None = None,
) -> TailscaleHealthDaemon:
    """Create a new TailscaleHealthDaemon instance (for testing)."""
    return TailscaleHealthDaemon(config=config)
