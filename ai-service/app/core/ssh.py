"""Unified SSH Helper Module for RingRift AI Service.

CANONICAL SSH UTILITY
=====================
This is the single source of truth for ALL SSH operations in the codebase.
Other modules should import from here instead of implementing their own SSH logic.

Features:
- Unified SSHClient class with async and sync methods
- Connection pooling via SSH ControlMaster (5 min persist)
- Automatic timeout handling
- SSH config loading from cluster_hosts.yaml
- Multi-transport fallback (Tailscale -> Direct -> Cloudflare)
- Backward-compatible function exports

Usage:
    from app.core.ssh import (
        SSHClient,
        get_ssh_client,
        run_ssh_command,
        run_ssh_command_async,
        SSHConfig,
        SSHResult,
    )

    # Async usage (recommended)
    client = get_ssh_client("runpod-h100")
    result = await client.run_async("nvidia-smi")

    # Sync usage
    result = client.run("nvidia-smi", timeout=30)

    # Convenience functions (backward compatible)
    result = await run_ssh_command_async("runpod-h100", "echo hello")

Migration from existing modules:
    - app/execution/executor.py:SSHExecutor -> app/core/ssh.SSHClient
    - app/distributed/hosts.py:SSHExecutor -> app/core/ssh.SSHClient
    - app/distributed/ssh_transport.py:SSHTransport -> app/core/ssh.SSHClient
    - app/providers/base.py:run_ssh_command -> app/core/ssh.run_ssh_command_async
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Import canonical CircuitState from circuit_breaker module (Dec 28, 2025 consolidation)
from app.distributed.circuit_breaker import CircuitState

__all__ = [
    "SSHClient",
    "SSHConfig",
    "SSHResult",
    "SSHHealth",
    "SSHMetrics",
    "CircuitState",  # Re-exported from circuit_breaker.py (Dec 2025)
    "get_ssh_client",
    "get_ssh_metrics",
    "run_ssh_command",
    "run_ssh_command_async",
    "run_ssh_command_sync",
    # Vast.ai convenience functions
    "run_vast_ssh_command",
    "run_vast_ssh_command_async",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SSHConfig:
    """SSH connection configuration."""
    host: str
    port: int = 22
    user: str | None = None
    key_path: str | None = None
    connect_timeout: int = 10
    command_timeout: int = 60
    use_control_master: bool = True
    control_persist: int = 300  # 5 minutes
    server_alive_interval: int = 30
    server_alive_count_max: int = 4
    tailscale_ip: str | None = None
    work_dir: str | None = None
    venv_activate: str | None = None

    # January 5, 2026: Ephemeral host support for Vast.ai containers
    # Ephemeral hosts regenerate SSH host keys on restart, causing connection failures
    # with strict host key checking. When True, uses StrictHostKeyChecking=no and
    # UserKnownHostsFile=/dev/null to handle dynamic host keys gracefully.
    is_ephemeral: bool = False

    # Cloudflare Zero Trust support
    cloudflare_tunnel: str | None = None
    cloudflare_service_token_id: str | None = None
    cloudflare_service_token_secret: str | None = None

    @property
    def use_cloudflare(self) -> bool:
        """Check if Cloudflare Zero Trust tunnel is configured."""
        return bool(self.cloudflare_tunnel)

    @classmethod
    def from_cluster_node(cls, node_id: str) -> SSHConfig | None:
        """Load SSH configuration from cluster_hosts.yaml."""
        try:
            from app.config.cluster_config import get_cluster_nodes
            nodes = get_cluster_nodes()
            if node_id not in nodes:
                return None

            node = nodes[node_id]
            # January 5, 2026: Detect ephemeral hosts (Vast.ai containers, etc.)
            # These regenerate SSH host keys on restart, requiring lenient host key checking
            is_ephemeral = getattr(node, "is_ephemeral", False) or getattr(node, "is_container", False)
            return cls(
                host=node.ssh_host or node.tailscale_ip or "",
                port=node.ssh_port,
                user=node.ssh_user,
                key_path=node.ssh_key,
                tailscale_ip=node.tailscale_ip,
                work_dir=node.ringrift_path,
                is_ephemeral=is_ephemeral,
                cloudflare_tunnel=getattr(node, "cloudflare_tunnel", None),
                cloudflare_service_token_id=getattr(node, "cloudflare_service_token_id", None),
                cloudflare_service_token_secret=getattr(node, "cloudflare_service_token_secret", None),
            )
        except Exception as e:
            logger.debug(f"Failed to load SSH config for {node_id}: {e}")
            return None

    @classmethod
    def from_host_config(cls, host_config: Any) -> SSHConfig:
        """Create SSHConfig from a HostConfig object.

        Args:
            host_config: HostConfig instance from distributed_hosts.yaml

        Returns:
            SSHConfig instance
        """
        # January 5, 2026: Detect ephemeral hosts (Vast.ai containers, etc.)
        # These regenerate SSH host keys on restart, requiring lenient host key checking
        is_ephemeral = getattr(host_config, "is_ephemeral", False) or getattr(host_config, "is_container", False)
        return cls(
            host=getattr(host_config, "ssh_host", getattr(host_config, "tailscale_ip", "")),
            port=getattr(host_config, "ssh_port", 22),
            user=getattr(host_config, "ssh_user", "ubuntu"),
            key_path=getattr(host_config, "ssh_key", None),
            tailscale_ip=getattr(host_config, "tailscale_ip", None),
            work_dir=getattr(host_config, "ringrift_path", "~/ringrift/ai-service"),
            is_ephemeral=is_ephemeral,
            cloudflare_tunnel=getattr(host_config, "cloudflare_tunnel", None),
            cloudflare_service_token_id=getattr(host_config, "cloudflare_service_token_id", None),
            cloudflare_service_token_secret=getattr(host_config, "cloudflare_service_token_secret", None),
        )


@dataclass
class SSHResult:
    """Result of SSH command execution."""
    success: bool
    returncode: int
    stdout: str
    stderr: str
    elapsed_ms: float
    command: str
    host: str
    transport_used: str | None = None
    timed_out: bool = False
    error: str | None = None

    @property
    def output(self) -> str:
        return f"{self.stdout}\n{self.stderr}".strip()

    def __bool__(self) -> bool:
        return self.success


# CircuitState is now imported from app.distributed.circuit_breaker (Dec 28, 2025 consolidation)
# This class was removed as it duplicated the canonical CircuitState(Enum) definition.
# The enum values are identical: CLOSED="closed", OPEN="open", HALF_OPEN="half_open"


@dataclass
class SSHHealth:
    """SSH connection health tracking with circuit breaker support.

    Features (Dec 2025):
    - Exponential backoff: 10s -> 20s -> 40s -> 80s -> 160s -> 300s max
    - Half-open state: Allows one test request after timeout
    - Early recovery probe: Tests at 50% of timeout elapsed
    """
    last_success: float | None = None
    last_failure: float | None = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_successes: int = 0
    total_failures: int = 0

    # Circuit breaker configuration
    failure_threshold: int = 3  # Default, overridden from coordination_defaults

    # Exponential backoff configuration
    initial_recovery_timeout: float = 30.0  # Start with 30s (was 300s)
    max_recovery_timeout: float = 300.0  # Cap at 5 minutes
    backoff_multiplier: float = 2.0  # Double each time

    # State tracking
    _consecutive_opens: int = 0  # Track how many times circuit has opened
    _circuit_state: CircuitState = field(default=CircuitState.CLOSED)  # Now uses enum (Dec 28, 2025)
    _circuit_opened_at: float | None = None  # When circuit entered OPEN state
    _last_probe_time: float | None = None  # When we last tried early recovery probe

    def __post_init__(self):
        """Load circuit breaker settings from centralized config."""
        try:
            from app.config.coordination_defaults import CircuitBreakerDefaults
            # Use centralized SSH circuit breaker settings
            self.failure_threshold = CircuitBreakerDefaults.SSH_FAILURE_THRESHOLD
            # Note: We use our own initial_recovery_timeout (30s) instead of the old 300s
        except ImportError:
            pass  # Use defaults if coordination_defaults not available

    @property
    def current_recovery_timeout(self) -> float:
        """Calculate current recovery timeout with exponential backoff.

        Backoff sequence: 30s -> 60s -> 120s -> 240s -> 300s (capped)
        The first open (consecutive_opens=1) uses base timeout (30s).
        Subsequent opens use exponentially increasing timeouts.
        """
        # Use (opens - 1) so first open uses base timeout
        exponent = max(0, self._consecutive_opens - 1)
        timeout = self.initial_recovery_timeout * (self.backoff_multiplier ** exponent)
        return min(timeout, self.max_recovery_timeout)

    # Keep legacy property for backward compatibility
    @property
    def recovery_timeout(self) -> float:
        """Legacy property - returns current backoff timeout."""
        return self.current_recovery_timeout

    @property
    def circuit_state(self) -> str:
        """Get current circuit state as string value (for backward compat)."""
        return self._circuit_state.value  # Return string value from enum (Dec 28, 2025)

    @property
    def is_healthy(self) -> bool:
        """Check if connection is considered healthy."""
        # Healthy if circuit is closed
        return self._circuit_state == CircuitState.CLOSED

    @property
    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open (fail-fast mode).

        Circuit is open when:
        - State is OPEN AND
        - Not enough time has passed for recovery attempt
        """
        if self._circuit_state == CircuitState.CLOSED:
            return False

        if self._circuit_state == CircuitState.HALF_OPEN:
            # In half-open, we allow exactly one request through
            return False

        # State is OPEN - check if we should transition to HALF_OPEN
        if self._circuit_opened_at is None:
            return False

        time_since_open = time.time() - self._circuit_opened_at
        if time_since_open >= self.current_recovery_timeout:
            # Timeout elapsed, transition to half-open
            self._circuit_state = CircuitState.HALF_OPEN
            return False

        return True

    @property
    def seconds_until_recovery(self) -> float:
        """Seconds until next recovery attempt is allowed."""
        if self._circuit_state == CircuitState.CLOSED:
            return 0.0
        if self._circuit_state == CircuitState.HALF_OPEN:
            return 0.0
        if self._circuit_opened_at is None:
            return 0.0
        elapsed = time.time() - self._circuit_opened_at
        return max(0.0, self.current_recovery_timeout - elapsed)

    @property
    def should_try_early_probe(self) -> bool:
        """Check if we should try an early recovery probe.

        Returns True if:
        - Circuit is OPEN
        - At least 50% of timeout has elapsed
        - We haven't probed in the last 10 seconds
        """
        if self._circuit_state != CircuitState.OPEN:
            return False
        if self._circuit_opened_at is None:
            return False

        now = time.time()
        time_since_open = now - self._circuit_opened_at
        half_timeout = self.current_recovery_timeout / 2

        if time_since_open < half_timeout:
            return False

        # Check if we've probed recently (avoid spamming probes)
        if self._last_probe_time is not None:
            if now - self._last_probe_time < 10.0:
                return False

        return True

    def mark_probe_attempted(self) -> None:
        """Mark that we attempted an early recovery probe."""
        self._last_probe_time = time.time()

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        total = self.total_successes + self.total_failures
        if total == 0:
            return 0.0
        return self.total_successes / total

    def open_circuit(self) -> None:
        """Open the circuit breaker (enter fail-fast mode)."""
        if self._circuit_state != CircuitState.OPEN:
            self._consecutive_opens += 1
            self._circuit_state = CircuitState.OPEN
            self._circuit_opened_at = time.time()
            logger.warning(
                f"Circuit breaker OPENED (attempt #{self._consecutive_opens}), "
                f"recovery timeout: {self.current_recovery_timeout:.0f}s"
            )

    def close_circuit(self) -> None:
        """Close the circuit breaker (resume normal operation)."""
        if self._circuit_state != CircuitState.CLOSED:
            logger.info(
                f"Circuit breaker CLOSED after {self._consecutive_opens} consecutive opens"
            )
            self._circuit_state = CircuitState.CLOSED
            self._circuit_opened_at = None
            self._consecutive_opens = 0  # Reset backoff on successful recovery
            self._last_probe_time = None

    def record_success(self) -> None:
        """Record successful operation - may close circuit."""
        self.last_success = time.time()
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.total_successes += 1

        # If we were in half-open state, success means we can close
        if self._circuit_state == CircuitState.HALF_OPEN:
            self.close_circuit()
        elif self._circuit_state == CircuitState.OPEN:
            # Early probe succeeded
            self.close_circuit()

    def record_failure(self) -> None:
        """Record failed operation - may open circuit."""
        self.last_failure = time.time()
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.total_failures += 1

        # If we were in half-open state, failure means re-open with increased backoff
        if self._circuit_state == CircuitState.HALF_OPEN:
            self._circuit_state = CircuitState.OPEN
            self._circuit_opened_at = time.time()
            # Note: _consecutive_opens was already incremented when we first opened
            logger.warning(
                f"Circuit breaker RE-OPENED from half-open, "
                f"recovery timeout: {self.current_recovery_timeout:.0f}s"
            )
        elif self.consecutive_failures >= self.failure_threshold:
            self.open_circuit()

    def reset_circuit(self) -> None:
        """Manually reset circuit breaker (e.g., after known recovery)."""
        self.consecutive_failures = 0
        self.close_circuit()


# =============================================================================
# SSH Metrics (December 2025 - Phase 5 Observability)
# =============================================================================

@dataclass
class HostMetrics:
    """Per-host SSH metrics."""
    host: str
    total_successes: int = 0
    total_failures: int = 0
    total_timeouts: int = 0
    last_success_time: float | None = None
    last_failure_time: float | None = None
    last_error: str | None = None
    total_latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate for this host."""
        total = self.total_successes + self.total_failures
        if total == 0:
            return 0.0
        return self.total_successes / total

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency for successful operations."""
        if self.total_successes == 0:
            return 0.0
        return self.total_latency_ms / self.total_successes


@dataclass
class TransportMetrics:
    """Per-transport SSH metrics."""
    transport: str  # "tailscale", "direct", "cloudflare"
    total_successes: int = 0
    total_failures: int = 0

    @property
    def success_rate(self) -> float:
        total = self.total_successes + self.total_failures
        if total == 0:
            return 0.0
        return self.total_successes / total


class SSHMetrics:
    """Global SSH failure metrics for observability.

    December 2025: Added as part of cluster node availability fixes.
    Provides visibility into SSH connection health across the cluster.

    Usage:
        from app.core.ssh import get_ssh_metrics

        metrics = get_ssh_metrics()

        # Get summary
        summary = metrics.get_summary()
        print(f"Total failures: {summary['total_failures']}")

        # Get per-host metrics
        host_metrics = metrics.get_host_metrics("runpod-h100")
        print(f"Host success rate: {host_metrics.success_rate:.1%}")

        # Get unhealthy hosts
        unhealthy = metrics.get_unhealthy_hosts(min_failures=3)
        for host in unhealthy:
            print(f"Unhealthy: {host}")
    """

    _instance: SSHMetrics | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._host_metrics: dict[str, HostMetrics] = {}
        self._transport_metrics: dict[str, TransportMetrics] = {}
        self._error_categories: dict[str, int] = {}  # category -> count
        self._start_time = time.time()

        # Initialize transport metrics
        for transport in ["tailscale", "direct", "cloudflare", "circuit_breaker"]:
            self._transport_metrics[transport] = TransportMetrics(transport=transport)

    @classmethod
    def get_instance(cls) -> SSHMetrics:
        """Get singleton instance."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._instance_lock:
            cls._instance = None

    def _get_or_create_host_metrics(self, host: str) -> HostMetrics:
        """Get or create metrics for a host."""
        with self._lock:
            if host not in self._host_metrics:
                self._host_metrics[host] = HostMetrics(host=host)
            return self._host_metrics[host]

    def _categorize_error(self, error: str | None) -> str:
        """Categorize an error message for aggregation."""
        if not error:
            return "unknown"

        error_lower = error.lower()

        if "timeout" in error_lower or "timed out" in error_lower:
            return "timeout"
        elif "connection refused" in error_lower:
            return "connection_refused"
        elif "permission denied" in error_lower:
            return "permission_denied"
        elif "host key" in error_lower:
            return "host_key_error"
        elif "no route to host" in error_lower:
            return "no_route"
        elif "network unreachable" in error_lower or "network is unreachable" in error_lower:
            return "network_unreachable"
        elif "circuit breaker" in error_lower:
            return "circuit_breaker"
        elif "connection reset" in error_lower:
            return "connection_reset"
        elif "broken pipe" in error_lower:
            return "broken_pipe"
        else:
            return "other"

    def record_success(
        self,
        host: str,
        transport: str | None = None,
        latency_ms: float = 0.0,
    ) -> None:
        """Record a successful SSH operation.

        Args:
            host: The host that was contacted
            transport: The transport used (tailscale, direct, cloudflare)
            latency_ms: Operation latency in milliseconds
        """
        with self._lock:
            # Update host metrics
            hm = self._get_or_create_host_metrics(host)
            hm.total_successes += 1
            hm.last_success_time = time.time()
            hm.total_latency_ms += latency_ms

            # Update transport metrics
            if transport and transport in self._transport_metrics:
                self._transport_metrics[transport].total_successes += 1

    def record_failure(
        self,
        host: str,
        transport: str | None = None,
        error: str | None = None,
        timed_out: bool = False,
    ) -> None:
        """Record a failed SSH operation.

        Args:
            host: The host that was contacted
            transport: The transport used (tailscale, direct, cloudflare)
            error: Error message or description
            timed_out: Whether the operation timed out
        """
        with self._lock:
            # Update host metrics
            hm = self._get_or_create_host_metrics(host)
            hm.total_failures += 1
            hm.last_failure_time = time.time()
            hm.last_error = error
            if timed_out:
                hm.total_timeouts += 1

            # Update transport metrics
            if transport and transport in self._transport_metrics:
                self._transport_metrics[transport].total_failures += 1

            # Update error category
            category = self._categorize_error(error)
            self._error_categories[category] = self._error_categories.get(category, 0) + 1

            # Log at WARNING level for visibility
            logger.warning(
                f"[SSHMetrics] Failure recorded: host={host}, transport={transport}, "
                f"category={category}, error={error}"
            )

    def get_host_metrics(self, host: str) -> HostMetrics | None:
        """Get metrics for a specific host."""
        with self._lock:
            return self._host_metrics.get(host)

    def get_all_host_metrics(self) -> dict[str, HostMetrics]:
        """Get metrics for all hosts."""
        with self._lock:
            return dict(self._host_metrics)

    def get_transport_metrics(self) -> dict[str, TransportMetrics]:
        """Get metrics for all transports."""
        with self._lock:
            return dict(self._transport_metrics)

    def get_error_categories(self) -> dict[str, int]:
        """Get error counts by category."""
        with self._lock:
            return dict(self._error_categories)

    def get_unhealthy_hosts(
        self,
        min_failures: int = 3,
        max_success_rate: float = 0.5,
    ) -> list[str]:
        """Get list of hosts considered unhealthy.

        Args:
            min_failures: Minimum failures to consider
            max_success_rate: Maximum success rate to be considered unhealthy

        Returns:
            List of unhealthy host names
        """
        with self._lock:
            unhealthy = []
            for host, metrics in self._host_metrics.items():
                if metrics.total_failures >= min_failures:
                    if metrics.success_rate <= max_success_rate:
                        unhealthy.append(host)
            return unhealthy

    def get_summary(self) -> dict[str, Any]:
        """Get summary of SSH metrics for monitoring.

        Returns:
            Dictionary with aggregated metrics suitable for dashboards.
        """
        with self._lock:
            total_successes = sum(m.total_successes for m in self._host_metrics.values())
            total_failures = sum(m.total_failures for m in self._host_metrics.values())
            total_timeouts = sum(m.total_timeouts for m in self._host_metrics.values())
            total_ops = total_successes + total_failures

            return {
                "uptime_seconds": time.time() - self._start_time,
                "total_operations": total_ops,
                "total_successes": total_successes,
                "total_failures": total_failures,
                "total_timeouts": total_timeouts,
                "success_rate": total_successes / total_ops if total_ops > 0 else 0.0,
                "hosts_tracked": len(self._host_metrics),
                "unhealthy_hosts": len(self.get_unhealthy_hosts()),
                "error_categories": dict(self._error_categories),
                "transport_stats": {
                    t: {"successes": m.total_successes, "failures": m.total_failures, "success_rate": m.success_rate}
                    for t, m in self._transport_metrics.items()
                    if m.total_successes + m.total_failures > 0
                },
            }

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        with self._lock:
            self._host_metrics.clear()
            self._error_categories.clear()
            self._start_time = time.time()
            for transport in self._transport_metrics.values():
                transport.total_successes = 0
                transport.total_failures = 0


# Module-level singleton accessor
def get_ssh_metrics() -> SSHMetrics:
    """Get the singleton SSHMetrics instance."""
    return SSHMetrics.get_instance()


# =============================================================================
# SSH Client
# =============================================================================

class SSHClient:
    """Unified SSH client with connection pooling and multi-transport support."""

    _clients: dict[str, SSHClient] = {}
    _lock = asyncio.Lock() if hasattr(asyncio, "Lock") else None

    def __init__(
        self,
        config: SSHConfig,
        control_path_dir: Path | None = None,
    ):
        self._config = config
        self._control_path_dir = control_path_dir or Path.home() / ".ssh" / "ringrift_control"
        self._control_path_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._health = SSHHealth()

    @property
    def config(self) -> SSHConfig:
        return self._config

    @property
    def health(self) -> SSHHealth:
        """Get connection health status."""
        return self._health

    def _record_success(
        self,
        transport: str | None = None,
        latency_ms: float = 0.0,
    ) -> None:
        """Record successful command execution.

        Args:
            transport: The transport used (tailscale, direct, cloudflare)
            latency_ms: Operation latency in milliseconds
        """
        self._health.record_success()
        # Update global metrics for observability
        get_ssh_metrics().record_success(
            host=self._config.host,
            transport=transport,
            latency_ms=latency_ms,
        )

    def _record_failure(
        self,
        transport: str | None = None,
        error: str | None = None,
        timed_out: bool = False,
    ) -> None:
        """Record failed command execution.

        Args:
            transport: The transport used (tailscale, direct, cloudflare)
            error: Error message or description
            timed_out: Whether the operation timed out
        """
        self._health.record_failure()
        # Update global metrics for observability
        get_ssh_metrics().record_failure(
            host=self._config.host,
            transport=transport,
            error=error,
            timed_out=timed_out,
        )

    async def _try_early_recovery_probe(self) -> bool:
        """Try an early recovery probe to test if host is back.

        Returns True if probe succeeded and circuit was closed.
        """
        if not self._health.should_try_early_probe:
            return False

        self._health.mark_probe_attempted()
        logger.debug(f"Attempting early recovery probe for {self._config.host}")

        # Use a lightweight probe command with short timeout
        probe_cmd = self._build_ssh_command("echo ok", use_tailscale=bool(self._config.tailscale_ip))

        try:
            proc = await asyncio.create_subprocess_exec(
                *probe_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=5.0,  # Short timeout for probe
                )

                if proc.returncode == 0 and b"ok" in stdout:
                    logger.info(f"Early recovery probe succeeded for {self._config.host}")
                    self._health.record_success()  # This closes the circuit
                    return True
                else:
                    logger.debug(f"Early recovery probe failed for {self._config.host}: exit={proc.returncode}")
                    return False

            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                logger.debug(f"Early recovery probe timed out for {self._config.host}")
                return False

        except (OSError, ValueError) as e:
            logger.debug(f"Early recovery probe error for {self._config.host}: {e}")
            return False

    def _get_control_path(self) -> str:
        """Get ControlMaster socket path for this host."""
        safe_host = self._config.host.replace(".", "_").replace(":", "_")
        return str(self._control_path_dir / f"{safe_host}_{self._config.port}")

    def _build_cloudflare_proxy_command(self) -> str:
        """Build cloudflared ProxyCommand for Cloudflare Zero Trust tunnel.

        Returns:
            ProxyCommand string for SSH config (-o ProxyCommand=...)
        """
        if not self._config.cloudflare_tunnel:
            raise ValueError("Cloudflare tunnel not configured")

        cmd_parts = ["cloudflared", "access", "ssh", "--hostname", self._config.cloudflare_tunnel]

        # Add service token authentication if configured
        if self._config.cloudflare_service_token_id and self._config.cloudflare_service_token_secret:
            cmd_parts.extend([
                "--service-token-id", self._config.cloudflare_service_token_id,
                "--service-token-secret", self._config.cloudflare_service_token_secret,
            ])

        return " ".join(cmd_parts)

    def _build_ssh_command(
        self,
        remote_command: str,
        use_tailscale: bool = False,
        use_cloudflare: bool = False,
    ) -> list[str]:
        """Build SSH command with proper options."""
        cmd = ["ssh"]

        # January 5, 2026: Ephemeral hosts (Vast.ai containers) regenerate SSH host keys
        # on restart. Use lenient host key checking to handle dynamic keys gracefully.
        if self._config.is_ephemeral:
            cmd.extend(["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"])
        else:
            cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])

        cmd.extend([
            "-o", f"ConnectTimeout={self._config.connect_timeout}",
            "-o", "BatchMode=yes",
            "-o", "LogLevel=ERROR",
            "-o", f"ServerAliveInterval={self._config.server_alive_interval}",
            "-o", f"ServerAliveCountMax={self._config.server_alive_count_max}",
            "-o", "TCPKeepAlive=yes",
        ])

        # Cloudflare Zero Trust tunnel via ProxyCommand
        if use_cloudflare:
            proxy_cmd = self._build_cloudflare_proxy_command()
            cmd.extend(["-o", f"ProxyCommand={proxy_cmd}"])

        # Connection pooling via ControlMaster
        if self._config.use_control_master:
            control_path = self._get_control_path()
            cmd.extend([
                "-o", f"ControlPath={control_path}",
                "-o", "ControlMaster=auto",
                "-o", f"ControlPersist={self._config.control_persist}",
            ])

        # Port - use port 22 for Tailscale IPs, otherwise use configured port
        effective_port = 22 if use_tailscale else self._config.port
        if effective_port != 22:
            cmd.extend(["-p", str(effective_port)])

        # Key file
        if self._config.key_path:
            key_path = os.path.expanduser(self._config.key_path)
            if os.path.exists(key_path):
                cmd.extend(["-i", key_path])

        # SSH target
        target_host = self._config.tailscale_ip if use_tailscale else self._config.host
        if self._config.user:
            target = f"{self._config.user}@{target_host}"
        else:
            target = target_host

        cmd.append(target)
        cmd.append(remote_command)

        return cmd

    async def run_async(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        bypass_circuit_breaker: bool = False,
    ) -> SSHResult:
        """Execute command asynchronously.

        Args:
            command: Command to execute on remote host
            timeout: Command timeout in seconds (default: config.command_timeout)
            cwd: Working directory on remote host
            env: Environment variables to set
            bypass_circuit_breaker: Skip circuit breaker check (for health checks)
        """
        # Try early recovery probe if circuit is open but probe window is available
        if not bypass_circuit_breaker and self._health.circuit_state == CircuitState.OPEN:
            if self._health.should_try_early_probe:
                probe_succeeded = await self._try_early_recovery_probe()
                if probe_succeeded:
                    logger.info(f"Early probe recovered {self._config.host}, proceeding with request")
                    # Circuit is now closed, continue with request

        # Circuit breaker check - fail fast for hosts with repeated failures
        if not bypass_circuit_breaker and self._health.is_circuit_open:
            recovery_seconds = self._health.seconds_until_recovery
            state = self._health.circuit_state
            backoff_level = self._health._consecutive_opens
            logger.warning(
                f"Circuit breaker {state.upper()} for {self._config.host}: "
                f"{self._health.consecutive_failures} failures, "
                f"backoff level {backoff_level}, "
                f"recovery in {recovery_seconds:.0f}s"
            )
            return SSHResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr=f"Circuit breaker {state}: host unavailable, retry in {recovery_seconds:.0f}s",
                elapsed_ms=0.0,
                command=command,
                host=self._config.host,
                transport_used="circuit_breaker",
            )

        # Log if we're in half-open state (testing recovery)
        if self._health.circuit_state == CircuitState.HALF_OPEN:
            logger.info(f"Circuit breaker HALF-OPEN for {self._config.host}: testing recovery with this request")

        timeout = timeout or self._config.command_timeout
        start_time = time.time()

        # Build remote command
        remote_cmd = command
        if cwd or self._config.work_dir:
            work_dir = cwd or self._config.work_dir
            remote_cmd = f"cd {work_dir} && {command}"
        if env:
            env_str = " ".join(f"{k}={v}" for k, v in env.items())
            remote_cmd = f"{env_str} {remote_cmd}"
        if self._config.venv_activate:
            # Handle both "source /path/activate" and just "/path/activate" formats
            venv = self._config.venv_activate
            if venv.startswith("source "):
                remote_cmd = f"{venv} && {remote_cmd}"
            else:
                remote_cmd = f"source {venv} && {remote_cmd}"

        # Transport fallback chain: Tailscale -> Direct -> Cloudflare
        transports = []
        if self._config.tailscale_ip:
            transports.append(("tailscale", True, False))
        transports.append(("direct", False, False))
        if self._config.use_cloudflare:
            transports.append(("cloudflare", False, True))

        last_error = None
        for transport_name, use_tailscale, use_cloudflare in transports:
            try:
                ssh_cmd = self._build_ssh_command(remote_cmd, use_tailscale, use_cloudflare)

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
                    elapsed_ms = (time.time() - start_time) * 1000

                    result = SSHResult(
                        success=proc.returncode == 0,
                        returncode=proc.returncode or 0,
                        stdout=stdout.decode("utf-8", errors="replace"),
                        stderr=stderr.decode("utf-8", errors="replace"),
                        elapsed_ms=elapsed_ms,
                        command=command,
                        host=self._config.host,
                        transport_used=transport_name,
                    )

                    # Record health status with transport and latency for metrics
                    if result.success:
                        self._record_success(transport=transport_name, latency_ms=elapsed_ms)
                    else:
                        self._record_failure(transport=transport_name, error=result.stderr)

                    return result

                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    elapsed_ms = (time.time() - start_time) * 1000
                    timeout_error = f"Command timed out after {timeout}s"
                    self._record_failure(transport=transport_name, error=timeout_error, timed_out=True)
                    return SSHResult(
                        success=False,
                        returncode=-1,
                        stdout="",
                        stderr=timeout_error,
                        elapsed_ms=elapsed_ms,
                        command=command,
                        host=self._config.host,
                        transport_used=transport_name,
                        timed_out=True,
                    )

            except Exception as e:
                last_error = str(e)
                # December 2025: Elevated from DEBUG to WARNING for visibility
                logger.warning(f"SSH via {transport_name} to {self._config.host} failed: {e}")
                continue

        elapsed_ms = (time.time() - start_time) * 1000
        all_failed_error = last_error or "All transports failed"
        self._record_failure(transport=None, error=all_failed_error)
        # December 2025: Elevated from DEBUG to WARNING for visibility
        logger.warning(
            f"SSH to {self._config.host} failed: {all_failed_error} "
            f"(tried {len(transports)} transports)"
        )
        return SSHResult(
            success=False,
            returncode=-1,
            stdout="",
            stderr=all_failed_error,
            elapsed_ms=elapsed_ms,
            command=command,
            host=self._config.host,
            error=last_error,
        )

    async def run_async_with_retry(
        self,
        command: str,
        timeout: int | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> SSHResult:
        """Execute with automatic retry on transient failures."""
        last_result = None
        for attempt in range(max_retries):
            result = await self.run_async(command, timeout)
            if result.success:
                return result
            last_result = result
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
        return last_result or SSHResult(
            success=False, returncode=-1, stdout="", stderr="No attempts made",
            elapsed_ms=0, command=command, host=self._config.host,
        )

    def run(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SSHResult:
        """Execute command synchronously."""
        timeout = timeout or self._config.command_timeout
        start_time = time.time()

        remote_cmd = command
        if cwd or self._config.work_dir:
            work_dir = cwd or self._config.work_dir
            remote_cmd = f"cd {work_dir} && {command}"
        if env:
            env_str = " ".join(f"{k}={v}" for k, v in env.items())
            remote_cmd = f"{env_str} {remote_cmd}"

        # Transport fallback chain: Tailscale -> Direct -> Cloudflare
        transports = []
        if self._config.tailscale_ip:
            transports.append(("tailscale", True, False))
        transports.append(("direct", False, False))
        if self._config.use_cloudflare:
            transports.append(("cloudflare", False, True))

        last_error = None
        for transport_name, use_tailscale, use_cloudflare in transports:
            try:
                ssh_cmd = self._build_ssh_command(remote_cmd, use_tailscale, use_cloudflare)

                result = subprocess.run(
                    ssh_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                elapsed_ms = (time.time() - start_time) * 1000

                ssh_result = SSHResult(
                    success=result.returncode == 0,
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    elapsed_ms=elapsed_ms,
                    command=command,
                    host=self._config.host,
                    transport_used=transport_name,
                )

                # Record health status with transport and latency for metrics
                if ssh_result.success:
                    self._record_success(transport=transport_name, latency_ms=elapsed_ms)
                else:
                    self._record_failure(transport=transport_name, error=result.stderr)

                return ssh_result

            except subprocess.TimeoutExpired:
                elapsed_ms = (time.time() - start_time) * 1000
                timeout_error = f"Command timed out after {timeout}s"
                self._record_failure(transport=transport_name, error=timeout_error, timed_out=True)
                return SSHResult(
                    success=False,
                    returncode=-1,
                    stdout="",
                    stderr=timeout_error,
                    elapsed_ms=elapsed_ms,
                    command=command,
                    host=self._config.host,
                    transport_used=transport_name,
                    timed_out=True,
                )

            except Exception as e:
                last_error = str(e)
                # December 2025: Elevated from DEBUG to WARNING for visibility
                logger.warning(f"SSH via {transport_name} to {self._config.host} failed: {e}")
                continue

        elapsed_ms = (time.time() - start_time) * 1000
        all_failed_error = last_error or "All transports failed"
        self._record_failure(transport=None, error=all_failed_error)
        # December 2025: Elevated from DEBUG to WARNING for visibility
        logger.warning(
            f"SSH to {self._config.host} failed: {all_failed_error} "
            f"(tried {len(transports)} transports)"
        )
        return SSHResult(
            success=False,
            returncode=-1,
            stdout="",
            stderr=all_failed_error,
            elapsed_ms=elapsed_ms,
            command=command,
            host=self._config.host,
            error=last_error,
        )

    async def check_connectivity(self) -> tuple[bool, str]:
        """Check if host is reachable via SSH."""
        result = await self.run_async("echo ok", timeout=10)
        if result.success and "ok" in result.stdout:
            return True, f"Connected via {result.transport_used}"
        return False, result.stderr or result.error or "Unknown error"

    def is_alive(self) -> bool:
        """Check if remote host is reachable (sync)."""
        result = self.run("echo ok", timeout=10)
        return result.success and "ok" in result.stdout

    def scp_from(
        self,
        remote_path: str,
        local_path: str,
        timeout: int = 300,
    ) -> SSHResult:
        """Copy file from remote host."""
        start_time = time.time()
        target_host = self._config.tailscale_ip or self._config.host
        if self._config.user:
            target = f"{self._config.user}@{target_host}"
        else:
            target = target_host

        cmd = [
            "scp",
            "-o", f"ConnectTimeout={self._config.connect_timeout}",
            "-o", "TCPKeepAlive=yes",
            "-o", f"ServerAliveInterval={self._config.server_alive_interval}",
            "-o", f"ServerAliveCountMax={self._config.server_alive_count_max}",
            "-o", "BatchMode=yes",
        ]
        # Use port 22 for Tailscale IPs, otherwise use configured port
        use_tailscale = self._config.tailscale_ip and target_host == self._config.tailscale_ip
        effective_port = 22 if use_tailscale else self._config.port
        if effective_port != 22:
            cmd.extend(["-P", str(effective_port)])
        if self._config.key_path:
            key_path = os.path.expanduser(self._config.key_path)
            if os.path.exists(key_path):
                cmd.extend(["-i", key_path])
        cmd.extend([f"{target}:{remote_path}", local_path])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            elapsed_ms = (time.time() - start_time) * 1000
            return SSHResult(
                success=result.returncode == 0,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                elapsed_ms=elapsed_ms,
                command=f"scp {remote_path} -> {local_path}",
                host=self._config.host,
            )
        except subprocess.TimeoutExpired:
            elapsed_ms = (time.time() - start_time) * 1000
            return SSHResult(
                success=False, returncode=-1, stdout="", stderr=f"SCP timed out after {timeout}s",
                elapsed_ms=elapsed_ms, command=f"scp {remote_path}", host=self._config.host, timed_out=True,
            )

    def scp_to(
        self,
        local_path: str,
        remote_path: str,
        timeout: int = 300,
    ) -> SSHResult:
        """Copy file to remote host."""
        start_time = time.time()
        target_host = self._config.tailscale_ip or self._config.host
        if self._config.user:
            target = f"{self._config.user}@{target_host}"
        else:
            target = target_host

        cmd = [
            "scp",
            "-o", f"ConnectTimeout={self._config.connect_timeout}",
            "-o", "TCPKeepAlive=yes",
            "-o", f"ServerAliveInterval={self._config.server_alive_interval}",
            "-o", f"ServerAliveCountMax={self._config.server_alive_count_max}",
            "-o", "BatchMode=yes",
        ]
        # Use port 22 for Tailscale IPs, otherwise use configured port
        use_tailscale = self._config.tailscale_ip and target_host == self._config.tailscale_ip
        effective_port = 22 if use_tailscale else self._config.port
        if effective_port != 22:
            cmd.extend(["-P", str(effective_port)])
        if self._config.key_path:
            key_path = os.path.expanduser(self._config.key_path)
            if os.path.exists(key_path):
                cmd.extend(["-i", key_path])
        cmd.extend([local_path, f"{target}:{remote_path}"])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            elapsed_ms = (time.time() - start_time) * 1000
            return SSHResult(
                success=result.returncode == 0,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                elapsed_ms=elapsed_ms,
                command=f"scp {local_path} -> {remote_path}",
                host=self._config.host,
            )
        except subprocess.TimeoutExpired:
            elapsed_ms = (time.time() - start_time) * 1000
            return SSHResult(
                success=False, returncode=-1, stdout="", stderr=f"SCP timed out after {timeout}s",
                elapsed_ms=elapsed_ms, command=f"scp {local_path}", host=self._config.host, timed_out=True,
            )

    async def run_background(
        self,
        command: str,
        log_file: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SSHResult:
        """Execute command in background using nohup.

        Args:
            command: Command to execute
            log_file: Optional log file path (defaults to nohup.out)
            cwd: Working directory
            env: Environment variables

        Returns:
            SSHResult with PID in stdout if successful
        """
        # Build nohup command
        log_redirect = f"> {log_file} 2>&1" if log_file else "> /dev/null 2>&1"
        bg_command = f"nohup {command} {log_redirect} & echo $!"

        result = await self.run_async(bg_command, cwd=cwd, env=env)

        if result.success:
            # Extract PID from output
            pid = result.stdout.strip()
            logger.debug(f"Started background process on {self._config.host}: PID={pid}")

        return result

    async def get_process_memory(self, pid: int) -> tuple[int, str] | None:
        """Get memory usage of a process in MB.

        Args:
            pid: Process ID to check

        Returns:
            Tuple of (memory_mb, command_name) or None if process not found
        """
        # Use ps to get RSS (resident set size) in KB and command name
        cmd = f"ps -p {pid} -o rss=,comm= 2>/dev/null || echo ''"
        result = await self.run_async(cmd)

        if not result.success or not result.stdout.strip():
            return None

        try:
            parts = result.stdout.strip().split(None, 1)
            if len(parts) < 2:
                return None

            rss_kb = int(parts[0])
            cmd_name = parts[1]
            memory_mb = rss_kb // 1024

            return memory_mb, cmd_name
        except (ValueError, IndexError):
            return None

    async def close(self) -> None:
        """Close ControlMaster connection."""
        if self._config.use_control_master:
            control_path = self._get_control_path()
            try:
                await asyncio.create_subprocess_exec(
                    "ssh", "-O", "exit", "-o", f"ControlPath={control_path}",
                    f"{self._config.user}@{self._config.host}" if self._config.user else self._config.host,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            except (subprocess.SubprocessError, OSError, ValueError):
                pass


# =============================================================================
# Client Pool / Factory (December 2025: Added TTL, health checks, LRU eviction)
# January 2026: Added dynamic pool scaling (Phase 7 P2P Stability)
# =============================================================================

import threading
import time as time_module

_client_pool: dict[str, SSHClient] = {}
_client_last_used: dict[str, float] = {}  # Track last use time
_client_pool_lock = threading.Lock()

# Pool configuration - static defaults (used if dynamic scaling disabled)
_STATIC_MAX_POOL_SIZE = 50  # Maximum clients to keep in pool
_STATIC_CLIENT_TTL_SECONDS = 1800  # 30 minutes max age
HEALTH_CHECK_INTERVAL = 300  # Check health every 5 minutes

# Jan 21, 2026: Phase 7 P2P Stability - Rollback via environment variable
_USE_STATIC_POOL = os.environ.get("RINGRIFT_SSH_STATIC_POOL", "").lower() in (
    "true", "1", "yes"
)


def _get_max_pool_size() -> int:
    """Get maximum pool size, scaling with active cluster nodes.

    Jan 21, 2026: Phase 7 P2P Stability Plan - prevents ControlMaster pool
    exhaustion with 41+ nodes under load.

    Returns:
        Dynamic pool size = max(30, active_nodes * 1.5), or static 50 if disabled
    """
    if _USE_STATIC_POOL:
        return _STATIC_MAX_POOL_SIZE

    try:
        from app.config.cluster_config import get_active_nodes
        active_count = len(get_active_nodes())
        # Scale pool: 1.5x overhead for active nodes, minimum 30
        return max(30, int(active_count * 1.5))
    except (ImportError, Exception):
        return _STATIC_MAX_POOL_SIZE


def _get_client_ttl() -> int:
    """Get client TTL, reducing under pool pressure.

    Jan 21, 2026: Phase 7 P2P Stability Plan - reduces TTL when pool is
    under pressure to make room for new connections.

    Returns:
        TTL in seconds (300-1800 depending on utilization)
    """
    if _USE_STATIC_POOL:
        return _STATIC_CLIENT_TTL_SECONDS

    max_pool = _get_max_pool_size()
    current_size = len(_client_pool)

    if max_pool == 0:
        return _STATIC_CLIENT_TTL_SECONDS

    utilization = current_size / max_pool

    if utilization > 0.9:
        return 300   # 5 min under high pressure
    elif utilization > 0.7:
        return 900   # 15 min under moderate pressure
    return 1800      # 30 min default


# Legacy aliases for backward compatibility
MAX_POOL_SIZE = _STATIC_MAX_POOL_SIZE
CLIENT_TTL_SECONDS = _STATIC_CLIENT_TTL_SECONDS


def _evict_stale_clients() -> int:
    """Evict stale clients from pool. Returns number evicted.

    Jan 21, 2026: Uses dynamic TTL and pool size from Phase 7 scaling.
    """
    now = time_module.time()
    evicted = 0
    to_remove = []

    # Get dynamic TTL and pool size
    ttl_seconds = _get_client_ttl()
    max_pool_size = _get_max_pool_size()

    with _client_pool_lock:
        for node_id, last_used in list(_client_last_used.items()):
            if now - last_used > ttl_seconds:
                to_remove.append(node_id)

        # Also evict LRU if pool is too large
        if len(_client_pool) > max_pool_size:
            sorted_by_age = sorted(_client_last_used.items(), key=lambda x: x[1])
            excess = len(_client_pool) - max_pool_size
            for node_id, _ in sorted_by_age[:excess]:
                if node_id not in to_remove:
                    to_remove.append(node_id)

        for node_id in to_remove:
            if node_id in _client_pool:
                del _client_pool[node_id]
                del _client_last_used[node_id]
                evicted += 1

    if evicted > 0:
        logger.debug(
            f"[SSHPool] Evicted {evicted} stale clients, pool size: {len(_client_pool)}/{max_pool_size}, "
            f"TTL: {ttl_seconds}s"
        )

    return evicted


def get_ssh_client(node_id_or_host: str) -> SSHClient:
    """Get SSH client for a node, creating if needed.

    Loads config from cluster_hosts.yaml if node_id matches.
    Automatically evicts stale clients when pool grows too large.

    Jan 21, 2026: Uses dynamic pool size from Phase 7 scaling.
    """
    # Periodically evict stale clients (use dynamic pool size)
    max_pool_size = _get_max_pool_size()
    if len(_client_pool) > max_pool_size // 2:
        _evict_stale_clients()

    with _client_pool_lock:
        if node_id_or_host in _client_pool:
            _client_last_used[node_id_or_host] = time_module.time()
            return _client_pool[node_id_or_host]

    # Try to load from cluster config
    config = SSHConfig.from_cluster_node(node_id_or_host)
    if config is None:
        # Fallback: treat as hostname
        config = SSHConfig(host=node_id_or_host)

    client = SSHClient(config)

    with _client_pool_lock:
        _client_pool[node_id_or_host] = client
        _client_last_used[node_id_or_host] = time_module.time()

    return client


def get_pool_stats() -> dict:
    """Get SSH connection pool statistics.

    Jan 21, 2026: Returns dynamic pool size and TTL from Phase 7 scaling.
    """
    max_pool_size = _get_max_pool_size()
    ttl_seconds = _get_client_ttl()

    with _client_pool_lock:
        now = time_module.time()
        current_size = len(_client_pool)
        utilization = current_size / max_pool_size if max_pool_size > 0 else 0.0

        return {
            "pool_size": current_size,
            "max_pool_size": max_pool_size,
            "ttl_seconds": ttl_seconds,
            "utilization_pct": round(utilization * 100, 1),
            "dynamic_scaling_enabled": not _USE_STATIC_POOL,
            "clients": {
                node_id: {
                    "age_seconds": now - _client_last_used.get(node_id, now),
                    "host": client.config.host,
                }
                for node_id, client in _client_pool.items()
            },
        }


async def close_all_clients() -> None:
    """Close all SSH connections."""
    with _client_pool_lock:
        for client in _client_pool.values():
            await client.close()
        _client_pool.clear()
        _client_last_used.clear()


# =============================================================================
# Backward-Compatible Function Exports
# =============================================================================

async def run_ssh_command_async(
    host: str,
    command: str,
    user: str | None = None,
    timeout: int | None = None,
    max_retries: int = 3,
    **kwargs: Any,
) -> SSHResult:
    """Run SSH command asynchronously (compatible with executor.py)."""
    client = get_ssh_client(host)
    if user and client.config.user != user:
        # Create a new config with the specified user
        new_config = SSHConfig(
            host=client.config.host,
            port=client.config.port,
            user=user,
            key_path=client.config.key_path,
            tailscale_ip=client.config.tailscale_ip,
        )
        client = SSHClient(new_config)
    return await client.run_async_with_retry(command, timeout, max_retries)


def run_ssh_command_sync(
    host: str,
    command: str,
    user: str | None = None,
    port: int = 22,
    key_path: str | None = None,
    timeout: int = 60,
    **kwargs: Any,
) -> SSHResult:
    """Run SSH command synchronously (compatible with executor.py)."""
    config = SSHConfig(
        host=host,
        port=port,
        user=user,
        key_path=key_path,
        command_timeout=timeout,
    )
    client = SSHClient(config)
    return client.run(command, timeout)


# Default to async for backward compatibility
run_ssh_command = run_ssh_command_async


# =============================================================================
# Vast.ai SSH Convenience Functions
# =============================================================================

def run_vast_ssh_command(
    host: str,
    port: int,
    command: str,
    *,
    timeout: int = 30,
    retries: int = 0,
    key_path: str | None = None,
) -> SSHResult:
    """Run SSH command on a Vast.ai instance (sync).

    Vast.ai instances use:
    - Custom SSH ports (not 22)
    - root user
    - accept-new host key checking
    - batch mode

    Args:
        host: Vast.ai instance hostname or IP
        port: SSH port (varies per instance)
        command: Command to execute
        timeout: Command timeout in seconds (default: 30)
        retries: Number of retry attempts (default: 0)
        key_path: Optional SSH key path (default: ~/.ssh/id_ed25519)

    Returns:
        SSHResult with command output

    Example:
        result = run_vast_ssh_command("ssh5.vast.ai", 12345, "nvidia-smi")
        if result.success:
            print(result.stdout)
    """
    config = SSHConfig(
        host=host,
        port=port,
        user="root",
        key_path=key_path or "~/.ssh/id_ed25519",
        connect_timeout=10,
        command_timeout=timeout,
        use_control_master=True,
    )
    client = SSHClient(config)

    last_result = None
    for attempt in range(retries + 1):
        result = client.run(command, timeout=timeout)
        if result.success:
            return result
        last_result = result
        if attempt < retries:
            time.sleep(2)  # Brief delay between retries

    return last_result or SSHResult(
        success=False,
        returncode=-1,
        stdout="",
        stderr="No attempts made",
        elapsed_ms=0,
        command=command,
        host=host,
    )


async def run_vast_ssh_command_async(
    host: str,
    port: int,
    command: str,
    *,
    timeout: int = 30,
    retries: int = 0,
    key_path: str | None = None,
) -> SSHResult:
    """Run SSH command on a Vast.ai instance (async).

    Vast.ai instances use:
    - Custom SSH ports (not 22)
    - root user
    - accept-new host key checking
    - batch mode

    Args:
        host: Vast.ai instance hostname or IP
        port: SSH port (varies per instance)
        command: Command to execute
        timeout: Command timeout in seconds (default: 30)
        retries: Number of retry attempts (default: 0)
        key_path: Optional SSH key path (default: ~/.ssh/id_ed25519)

    Returns:
        SSHResult with command output

    Example:
        result = await run_vast_ssh_command_async("ssh5.vast.ai", 12345, "nvidia-smi")
        if result.success:
            print(result.stdout)
    """
    config = SSHConfig(
        host=host,
        port=port,
        user="root",
        key_path=key_path or "~/.ssh/id_ed25519",
        connect_timeout=10,
        command_timeout=timeout,
        use_control_master=True,
    )
    client = SSHClient(config)

    if retries > 0:
        return await client.run_async_with_retry(command, timeout, retries)
    return await client.run_async(command, timeout=timeout)
