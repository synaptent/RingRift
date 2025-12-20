"""SSH Transport Layer for P2P Communication.

Provides SSH-based communication as a fallback when HTTP fails for Vast.ai
instances behind NAT with userland Tailscale (no TUN device).

The key insight: Vast's SSH proxy addresses (sshX.vast.ai:port) are ALWAYS
reachable regardless of the instance's internal networking situation.

Architecture:
- Primary: HTTP via Tailscale mesh (when it works)
- Fallback: SSH exec commands through Vast proxy

Usage:
    from app.distributed.ssh_transport import SSHTransport

    transport = SSHTransport()

    # Send command to NAT-blocked Vast instance
    result = await transport.send_command(
        node_id="vast-5090-quad",
        command_type="heartbeat",
        payload={"status": "active"},
    )

    # Execute arbitrary command
    output = await transport.ssh_exec(
        node_id="vast-5090-quad",
        command="curl -s localhost:8770/status",
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Import circuit breaker for fault tolerance
try:
    from app.distributed.circuit_breaker import (
        CircuitOpenError,
        CircuitState,
        get_adaptive_timeout,
        get_operation_breaker,
    )
    HAS_CIRCUIT_BREAKER = True
except ImportError:
    HAS_CIRCUIT_BREAKER = False
    CircuitOpenError = Exception

# Try to load from unified config, with fallback defaults
try:
    from app.config.unified_config import get_config
    _config = get_config()
    SSH_CONNECT_TIMEOUT = _config.ssh.connect_timeout_seconds
    SSH_COMMAND_TIMEOUT = _config.ssh.transport_command_timeout_seconds
    SSH_MAX_RETRIES = _config.ssh.max_retries
    SSH_RETRY_DELAY = _config.ssh.retry_delay_seconds
    SSH_ADDRESS_CACHE_TTL = _config.ssh.address_cache_ttl_seconds
    LOCAL_P2P_PORT = _config.distributed.p2p_port
except ImportError:
    # Fallback defaults if config not available
    SSH_CONNECT_TIMEOUT = 10  # seconds
    SSH_COMMAND_TIMEOUT = 30  # seconds
    SSH_MAX_RETRIES = 2
    SSH_RETRY_DELAY = 1.0  # seconds
    SSH_ADDRESS_CACHE_TTL = 300  # 5 minutes
    LOCAL_P2P_PORT = int(os.environ.get("RINGRIFT_P2P_PORT", "8770"))

# Default SSH user for Vast instances (env override always available)
VAST_SSH_USER = os.environ.get("RINGRIFT_VAST_SSH_USER", "root")


@dataclass
class SSHAddress:
    """SSH connection details for a node."""
    node_id: str
    ssh_host: str  # e.g., "ssh5.vast.ai"
    ssh_port: int  # e.g., 14364
    ssh_user: str = VAST_SSH_USER
    last_success: float = 0.0
    last_failure: float = 0.0
    consecutive_failures: int = 0
    cached_at: float = field(default_factory=time.time)

    @property
    def is_stale(self) -> bool:
        """Check if cached address is stale."""
        return time.time() - self.cached_at > SSH_ADDRESS_CACHE_TTL

    @property
    def ssh_destination(self) -> str:
        """Full SSH destination string."""
        return f"{self.ssh_user}@{self.ssh_host}"

    def __str__(self) -> str:
        return f"{self.ssh_user}@{self.ssh_host}:{self.ssh_port}"


@dataclass
class SSHCommandResult:
    """Result of an SSH command execution."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = -1
    elapsed_ms: float = 0.0
    error: str | None = None


class SSHTransport:
    """SSH-based transport for P2P communication.

    Provides reliable communication with Vast instances by using their
    SSH proxy addresses as a fallback when HTTP fails.

    Features:
    - Automatic address discovery from DynamicHostRegistry
    - Connection pooling (via SSH ControlMaster)
    - Retry with exponential backoff
    - Health tracking per node
    """

    def __init__(self, control_path_dir: Path | None = None):
        """Initialize SSH transport.

        Args:
            control_path_dir: Directory for SSH ControlMaster sockets
        """
        self._addresses: dict[str, SSHAddress] = {}
        self._lock = asyncio.Lock()

        # SSH ControlMaster directory for connection pooling
        if control_path_dir is None:
            control_path_dir = Path.home() / ".ssh" / "ringrift_control"
        self._control_path_dir = control_path_dir
        self._control_path_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

        # Try to load DynamicHostRegistry
        self._registry = None
        try:
            from app.distributed.dynamic_registry import get_registry
            self._registry = get_registry()
        except ImportError:
            logger.warning("DynamicHostRegistry not available, SSH addresses must be set manually")

    def _get_control_path(self, node_id: str) -> str:
        """Get SSH ControlMaster socket path for a node."""
        # Use short path to avoid socket path length limits
        safe_id = node_id.replace("/", "_").replace(":", "_")[:20]
        return str(self._control_path_dir / f"ctrl_{safe_id}")

    def _get_ssh_address(self, node_id: str) -> SSHAddress | None:
        """Get SSH address for a node, using cache or registry.

        Args:
            node_id: Node identifier

        Returns:
            SSHAddress if found, None otherwise
        """
        # Check cache first
        cached = self._addresses.get(node_id)
        if cached and not cached.is_stale:
            return cached

        # Try to get from registry
        if self._registry:
            try:
                node_info = self._registry._nodes.get(node_id)
                if node_info:
                    # Get SSH host/port from dynamic or static config
                    ssh_host = node_info.dynamic_host or node_info.static_host
                    ssh_port = node_info.dynamic_port or node_info.static_port

                    if ssh_host and ssh_port:
                        addr = SSHAddress(
                            node_id=node_id,
                            ssh_host=ssh_host,
                            ssh_port=ssh_port,
                        )
                        self._addresses[node_id] = addr
                        return addr
            except Exception as e:
                logger.debug(f"Failed to get SSH address from registry for {node_id}: {e}")

        return cached  # Return stale cache if nothing else available

    def set_ssh_address(
        self,
        node_id: str,
        ssh_host: str,
        ssh_port: int,
        ssh_user: str = VAST_SSH_USER,
    ) -> None:
        """Manually set SSH address for a node.

        Args:
            node_id: Node identifier
            ssh_host: SSH hostname (e.g., "ssh5.vast.ai")
            ssh_port: SSH port
            ssh_user: SSH username (default: root)
        """
        self._addresses[node_id] = SSHAddress(
            node_id=node_id,
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            ssh_user=ssh_user,
        )

    def _build_ssh_command(
        self,
        addr: SSHAddress,
        remote_command: str,
        use_control_master: bool = True,
    ) -> list[str]:
        """Build SSH command with proper options.

        Args:
            addr: SSH address details
            remote_command: Command to execute on remote host
            use_control_master: Whether to use SSH ControlMaster

        Returns:
            List of command arguments for subprocess
        """
        cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}",
            "-o", "BatchMode=yes",  # No interactive prompts
            "-o", "LogLevel=ERROR",  # Reduce noise
        ]

        # Add ControlMaster options for connection pooling
        if use_control_master:
            control_path = self._get_control_path(addr.node_id)
            cmd.extend([
                "-o", f"ControlPath={control_path}",
                "-o", "ControlMaster=auto",
                "-o", "ControlPersist=60",  # Reduced from 300s for faster reconnection
                # TCP keepalive settings to detect dead connections
                "-o", "ServerAliveInterval=15",  # Send keepalive every 15 seconds
                "-o", "ServerAliveCountMax=3",   # Fail after 3 missed keepalives (45s total)
                "-o", "TCPKeepAlive=yes",        # Enable TCP-level keepalive
            ])

        # Add port
        cmd.extend(["-p", str(addr.ssh_port)])

        # Add destination
        cmd.append(addr.ssh_destination)

        # Add remote command
        cmd.append(remote_command)

        return cmd

    async def ssh_exec(
        self,
        node_id: str,
        command: str,
        timeout: float = SSH_COMMAND_TIMEOUT,
    ) -> SSHCommandResult:
        """Execute a command on a remote node via SSH.

        Args:
            node_id: Target node identifier
            command: Shell command to execute
            timeout: Command timeout in seconds

        Returns:
            SSHCommandResult with execution details
        """
        addr = self._get_ssh_address(node_id)
        if not addr:
            return SSHCommandResult(
                success=False,
                error=f"No SSH address found for {node_id}",
            )

        # Circuit breaker protection
        if HAS_CIRCUIT_BREAKER:
            breaker = get_operation_breaker("ssh")
            if not breaker.can_execute(node_id):
                return SSHCommandResult(
                    success=False,
                    error=f"Circuit breaker open for {node_id}",
                )
            # Use adaptive timeout based on circuit state
            timeout = get_adaptive_timeout("ssh", node_id, timeout)

        ssh_cmd = self._build_ssh_command(addr, command)
        start_time = time.time()

        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                elapsed = (time.time() - start_time) * 1000
                addr.consecutive_failures += 1
                addr.last_failure = time.time()
                # Record timeout as failure in circuit breaker
                if HAS_CIRCUIT_BREAKER:
                    get_operation_breaker("ssh").record_failure(
                        node_id, TimeoutError(f"SSH command timeout after {timeout}s")
                    )
                return SSHCommandResult(
                    success=False,
                    error=f"Command timed out after {timeout}s",
                    elapsed_ms=elapsed,
                )

            elapsed = (time.time() - start_time) * 1000
            success = proc.returncode == 0

            if success:
                addr.consecutive_failures = 0
                addr.last_success = time.time()
                # Record success in circuit breaker
                if HAS_CIRCUIT_BREAKER:
                    get_operation_breaker("ssh").record_success(node_id)
            else:
                addr.consecutive_failures += 1
                addr.last_failure = time.time()
                # Record failure in circuit breaker
                if HAS_CIRCUIT_BREAKER:
                    get_operation_breaker("ssh").record_failure(node_id)

            return SSHCommandResult(
                success=success,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                return_code=proc.returncode,
                elapsed_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            addr.consecutive_failures += 1
            addr.last_failure = time.time()
            # Record failure in circuit breaker
            if HAS_CIRCUIT_BREAKER:
                get_operation_breaker("ssh").record_failure(node_id, e)
            return SSHCommandResult(
                success=False,
                error=str(e),
                elapsed_ms=elapsed,
            )

    async def send_command(
        self,
        node_id: str,
        command_type: str,
        payload: dict[str, Any],
        retries: int = SSH_MAX_RETRIES,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Send a P2P command to a node via SSH.

        This is the main entry point for SSH-based P2P communication.
        It executes a curl command on the remote host to hit its local
        P2P API.

        Args:
            node_id: Target node identifier
            command_type: Command type (heartbeat, start_job, etc.)
            payload: Command payload as dict
            retries: Number of retries on failure

        Returns:
            Tuple of (success, response_dict or None)
        """
        # Build the curl command to hit local P2P API
        endpoint = self._command_type_to_endpoint(command_type)
        payload_json = json.dumps(payload)

        # Use shlex.quote() for proper shell escaping to prevent command injection
        # This handles all shell metacharacters safely, not just single quotes
        escaped_payload = shlex.quote(payload_json)

        curl_cmd = (
            f"curl -s -X POST 'http://localhost:{LOCAL_P2P_PORT}{endpoint}' "
            f"-H 'Content-Type: application/json' "
            f"-d {escaped_payload} "
            f"--connect-timeout 5 --max-time 20"
        )

        for attempt in range(retries + 1):
            result = await self.ssh_exec(node_id, curl_cmd)

            if result.success and result.stdout:
                try:
                    response = json.loads(result.stdout)
                    return True, response
                except json.JSONDecodeError:
                    logger.warning(
                        f"Invalid JSON response from {node_id}: {result.stdout[:200]}"
                    )

            if attempt < retries:
                await asyncio.sleep(SSH_RETRY_DELAY * (attempt + 1))

        return False, None

    async def check_connectivity(self, node_id: str) -> tuple[bool, str]:
        """Check if a node is reachable via SSH.

        Args:
            node_id: Node to check

        Returns:
            Tuple of (reachable, status_message)
        """
        result = await self.ssh_exec(node_id, "echo 'ping'", timeout=10)

        if result.success and "ping" in result.stdout:
            return True, f"SSH OK ({result.elapsed_ms:.0f}ms)"
        else:
            error = result.error or result.stderr or "Unknown error"
            return False, f"SSH failed: {error}"

    async def get_node_status(self, node_id: str) -> dict[str, Any] | None:
        """Get P2P status from a node via SSH.

        Args:
            node_id: Node to query

        Returns:
            Status dict if successful, None otherwise
        """
        result = await self.ssh_exec(
            node_id,
            f"curl -s http://localhost:{LOCAL_P2P_PORT}/status --connect-timeout 5",
        )

        if result.success and result.stdout:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                pass

        return None

    async def send_heartbeat(
        self,
        node_id: str,
        self_info: dict[str, Any],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Send heartbeat to a node via SSH.

        This is used as a fallback when HTTP heartbeats fail.

        Args:
            node_id: Target node
            self_info: Our node info dict

        Returns:
            Tuple of (success, response_dict or None)
        """
        return await self.send_command(node_id, "heartbeat", self_info)

    async def relay_command(
        self,
        node_id: str,
        target_node_id: str,
        command_type: str,
        payload: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """Relay a command through one node to another.

        Useful when node A can reach node B via SSH, but node B
        needs to forward a command to node C.

        Args:
            node_id: Relay node
            target_node_id: Ultimate target
            command_type: Command type
            payload: Command payload

        Returns:
            Tuple of (success, command_id or None)
        """
        relay_payload = {
            "target_node_id": target_node_id,
            "type": command_type,
            "payload": payload,
        }
        success, response = await self.send_command(
            node_id, "relay_enqueue", relay_payload
        )

        if success and response:
            return True, response.get("id")

        return False, None

    def _command_type_to_endpoint(self, command_type: str) -> str:
        """Map command type to P2P API endpoint."""
        mapping = {
            "heartbeat": "/heartbeat",
            "relay_heartbeat": "/relay/heartbeat",
            "relay_enqueue": "/relay/enqueue",
            "start_job": "/jobs/start",
            "stop_job": "/jobs/stop",
            "cleanup": "/cleanup",
            "cleanup_files": "/cleanup/files",
            "reduce_selfplay": "/selfplay/reduce",
            "restart_stuck_jobs": "/jobs/restart_stuck",
            "status": "/status",
            "health": "/health",
        }
        return mapping.get(command_type, f"/command/{command_type}")

    async def close(self) -> None:
        """Close all SSH ControlMaster connections."""
        for node_id in list(self._addresses.keys()):
            control_path = self._get_control_path(node_id)
            if Path(control_path).exists():
                try:
                    # Send exit command to ControlMaster
                    proc = await asyncio.create_subprocess_exec(
                        "ssh", "-O", "exit", "-o", f"ControlPath={control_path}",
                        "dummy",  # Destination ignored for -O exit
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await proc.wait()
                except Exception:
                    pass


# Global instance
_transport: SSHTransport | None = None


def get_ssh_transport() -> SSHTransport:
    """Get or create global SSH transport instance."""
    global _transport
    if _transport is None:
        _transport = SSHTransport()
    return _transport


async def probe_vast_nodes_via_ssh() -> dict[str, tuple[bool, str]]:
    """Probe all Vast nodes via SSH and return connectivity status.

    Returns:
        Dict mapping node_id to (reachable, status_message)
    """
    transport = get_ssh_transport()
    results = {}

    try:
        from app.distributed.dynamic_registry import get_registry
        registry = get_registry()

        vast_nodes = [
            node_id for node_id, node in registry._nodes.items()
            if node_id.startswith("vast-")
        ]

        tasks = [transport.check_connectivity(node_id) for node_id in vast_nodes]
        connectivity = await asyncio.gather(*tasks, return_exceptions=True)

        for node_id, result in zip(vast_nodes, connectivity, strict=False):
            if isinstance(result, Exception):
                results[node_id] = (False, str(result))
            else:
                results[node_id] = result

    except ImportError:
        logger.warning("DynamicHostRegistry not available for SSH probing")

    return results
