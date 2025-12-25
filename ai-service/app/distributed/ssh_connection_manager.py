"""SSH Connection Manager with Smart Transport Selection.

Provides intelligent SSH connection management with:
- Quick preflight TCP probe before SSH
- Smart transport ordering based on health history
- Automatic fallback to alternate transports
- Integration with TransportHealthTracker

Usage:
    from app.distributed.ssh_connection_manager import SSHConnectionManager

    manager = SSHConnectionManager()

    # Smart connection with automatic transport selection
    success, output = await manager.run_command(
        node_id="lambda-gh200-a",
        command="nvidia-smi",
        transports={"tailscale": "100.123.183.70", "direct": "192.222.51.29"},
    )

    # With explicit transport preference
    success, output = await manager.run_command(
        node_id="lambda-gh200-a",
        command="nvidia-smi",
        transports={"tailscale": "100.123.183.70"},
        prefer_transport="tailscale",
    )
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from dataclasses import dataclass
from typing import Any

from app.distributed.transport_health import (
    TransportHealthTracker,
    TransportType,
    get_tracker,
)

logger = logging.getLogger(__name__)


@dataclass
class ConnectionResult:
    """Result of an SSH connection attempt."""
    success: bool
    output: str
    transport_used: str | None = None
    latency_ms: float = 0.0
    error: str | None = None
    attempts: int = 0


class SSHConnectionManager:
    """Manages SSH connections with smart transport selection.

    Features:
    - TCP preflight probe to quickly detect unreachable hosts
    - Transport health tracking for automatic failover
    - Adaptive timeouts based on historical latency
    - Circuit breaker pattern for failing transports
    """

    # Configuration
    TCP_PROBE_TIMEOUT = float(os.environ.get("RINGRIFT_TCP_PROBE_TIMEOUT", "3.0"))
    SSH_DEFAULT_TIMEOUT = int(os.environ.get("RINGRIFT_SSH_DEFAULT_TIMEOUT", "30"))
    MAX_RETRY_ATTEMPTS = int(os.environ.get("RINGRIFT_SSH_MAX_RETRIES", "2"))
    RETRY_DELAY_SECONDS = float(os.environ.get("RINGRIFT_SSH_RETRY_DELAY", "1.0"))

    def __init__(
        self,
        ssh_key: str | None = None,
        ssh_user: str = "ubuntu",
        health_tracker: TransportHealthTracker | None = None,
    ):
        """Initialize SSH Connection Manager.

        Args:
            ssh_key: Path to SSH private key
            ssh_user: Default SSH user
            health_tracker: Optional health tracker (uses global singleton if not provided)
        """
        self.ssh_key = ssh_key or os.environ.get("RINGRIFT_SSH_KEY", "~/.ssh/id_cluster")
        self.ssh_user = ssh_user
        self.health_tracker = health_tracker or get_tracker()

    async def tcp_probe(
        self,
        host: str,
        port: int = 22,
        timeout: float | None = None,
    ) -> tuple[bool, float]:
        """Quick TCP probe to check if host:port is reachable.

        Args:
            host: Hostname or IP address
            port: Port to probe (default: 22 for SSH)
            timeout: Probe timeout in seconds

        Returns:
            Tuple of (reachable: bool, latency_ms: float)
        """
        timeout = timeout or self.TCP_PROBE_TIMEOUT
        start = time.time()

        try:
            # Use asyncio to avoid blocking
            loop = asyncio.get_event_loop()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setblocking(False)

            try:
                await asyncio.wait_for(
                    loop.sock_connect(sock, (host, port)),
                    timeout=timeout,
                )
                latency_ms = (time.time() - start) * 1000
                return True, latency_ms
            finally:
                sock.close()

        except (socket.error, asyncio.TimeoutError, OSError) as e:
            logger.debug(f"TCP probe failed for {host}:{port}: {e}")
            return False, 0.0

    async def run_command(
        self,
        node_id: str,
        command: str,
        transports: dict[str, str],  # {"tailscale": "100.x.x.x", "direct": "192.x.x.x"}
        *,
        port: int = 22,
        user: str | None = None,
        timeout: int | None = None,
        prefer_transport: str | None = None,
        use_smart_transport: bool = True,
        retry_on_failure: bool = True,
    ) -> ConnectionResult:
        """Execute SSH command with smart transport selection.

        Args:
            node_id: Node identifier for health tracking
            command: Command to execute
            transports: Dict of transport_type -> IP address
            port: SSH port
            user: SSH user (defaults to instance user)
            timeout: Command timeout in seconds
            prefer_transport: Explicitly prefer this transport
            use_smart_transport: Enable smart transport selection (default: True)
            retry_on_failure: Retry with alternate transport on failure

        Returns:
            ConnectionResult with success status and output
        """
        user = user or self.ssh_user
        timeout = timeout or self.SSH_DEFAULT_TIMEOUT

        # Build ordered list of transports to try
        transport_order = self._get_transport_order(
            node_id, transports, prefer_transport, use_smart_transport
        )

        if not transport_order:
            return ConnectionResult(
                success=False,
                output="",
                error="No transports available",
                attempts=0,
            )

        result = ConnectionResult(success=False, output="", attempts=0)
        last_error = None

        for transport_name, host in transport_order:
            result.attempts += 1
            transport_type = TransportType(transport_name)

            # TCP preflight probe
            reachable, probe_latency = await self.tcp_probe(host, port)
            if not reachable:
                logger.debug(f"TCP probe failed for {node_id}/{transport_name} ({host})")
                self.health_tracker.record_failure(node_id, transport_type, "TCP probe failed")
                last_error = f"TCP probe failed for {host}"
                continue

            # Get adaptive timeout based on history
            adaptive_timeout = self.health_tracker.get_adaptive_timeout(
                node_id, transport_type, timeout
            )

            # Execute SSH command
            start_time = time.time()
            success, output, error = await self._execute_ssh(
                host=host,
                port=port,
                user=user,
                command=command,
                timeout=int(adaptive_timeout),
            )
            latency_ms = (time.time() - start_time) * 1000

            if success:
                self.health_tracker.record_success(node_id, transport_type, latency_ms)
                return ConnectionResult(
                    success=True,
                    output=output,
                    transport_used=transport_name,
                    latency_ms=latency_ms,
                    attempts=result.attempts,
                )
            else:
                self.health_tracker.record_failure(node_id, transport_type, error)
                last_error = error
                logger.debug(
                    f"SSH failed for {node_id}/{transport_name}: {error}"
                )

            if not retry_on_failure:
                break

        # All transports failed
        return ConnectionResult(
            success=False,
            output="",
            error=last_error or "All transports failed",
            attempts=result.attempts,
        )

    def _get_transport_order(
        self,
        node_id: str,
        transports: dict[str, str],
        prefer_transport: str | None,
        use_smart_transport: bool,
    ) -> list[tuple[str, str]]:
        """Get ordered list of transports to try.

        Returns:
            List of (transport_name, ip_address) tuples in order of preference
        """
        available = list(transports.keys())

        if not available:
            return []

        if prefer_transport and prefer_transport in transports:
            # Explicit preference - put it first
            order = [prefer_transport] + [t for t in available if t != prefer_transport]
        elif use_smart_transport:
            # Use health tracker to determine best order
            available_types = [TransportType(t) for t in available]
            best = self.health_tracker.get_best_transport(node_id, available_types)
            if best:
                best_name = best.value
                order = [best_name] + [t for t in available if t != best_name]
            else:
                # Default order: Tailscale first (more reliable for Lambda)
                if "tailscale" in available:
                    order = ["tailscale"] + [t for t in available if t != "tailscale"]
                else:
                    order = available
        else:
            # No smart selection - use provided order
            order = available

        return [(t, transports[t]) for t in order if t in transports]

    async def _execute_ssh(
        self,
        host: str,
        port: int,
        user: str,
        command: str,
        timeout: int,
    ) -> tuple[bool, str, str | None]:
        """Execute SSH command.

        Returns:
            Tuple of (success, output, error)
        """
        ssh_key_path = os.path.expanduser(self.ssh_key)

        # Build SSH command with hardened options
        ssh_args = [
            "ssh",
            "-o", f"ConnectTimeout={min(timeout, 30)}",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=4",
            "-o", "TCPKeepAlive=yes",
        ]

        if os.path.exists(ssh_key_path):
            ssh_args.extend(["-i", ssh_key_path])

        if port != 22:
            ssh_args.extend(["-p", str(port)])

        ssh_args.append(f"{user}@{host}")
        ssh_args.append(command)

        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )

            output = stdout.decode("utf-8", errors="replace").strip()
            error_output = stderr.decode("utf-8", errors="replace").strip()

            if proc.returncode == 0:
                return True, output, None
            else:
                error_msg = error_output or f"Exit code {proc.returncode}"
                return False, output, error_msg

        except asyncio.TimeoutError:
            return False, "", f"Timeout after {timeout}s"
        except Exception as e:
            return False, "", str(e)

    async def probe_all_transports(
        self,
        node_id: str,
        transports: dict[str, str],
        port: int = 22,
    ) -> dict[str, dict[str, Any]]:
        """Probe all available transports for a node.

        Useful for diagnostics and initial transport selection.

        Args:
            node_id: Node identifier
            transports: Dict of transport_type -> IP address
            port: SSH port to probe

        Returns:
            Dict of transport results with reachability and latency
        """
        results = {}

        async def probe_one(transport_name: str, host: str) -> None:
            reachable, latency = await self.tcp_probe(host, port)
            results[transport_name] = {
                "host": host,
                "reachable": reachable,
                "latency_ms": latency,
            }

        # Probe all transports concurrently
        await asyncio.gather(
            *[probe_one(name, host) for name, host in transports.items()]
        )

        return results


# Convenience function for synchronous use
def run_ssh_command_smart(
    node_id: str,
    command: str,
    transports: dict[str, str],
    **kwargs: Any,
) -> ConnectionResult:
    """Synchronous wrapper for SSHConnectionManager.run_command."""
    manager = SSHConnectionManager()
    return asyncio.run(
        manager.run_command(node_id, command, transports, **kwargs)
    )
