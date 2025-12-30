"""Base classes for cloud provider management.

All provider managers inherit from ProviderManager and implement
a consistent interface for instance discovery, health checking,
and lifecycle management.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from app.config.ports import P2P_DEFAULT_PORT

logger = logging.getLogger(__name__)


class Provider(str, Enum):
    """Supported cloud providers."""

    LAMBDA = "lambda"
    VAST = "vast"
    HETZNER = "hetzner"
    AWS = "aws"
    LOCAL = "local"


class InstanceState(str, Enum):
    """Unified instance state across providers."""

    RUNNING = "running"
    STARTING = "starting"
    STOPPING = "stopping"
    STOPPED = "stopped"
    TERMINATED = "terminated"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class ProviderInstance:
    """Unified representation of a cloud instance."""

    # Identity
    instance_id: str
    provider: Provider
    name: str

    # Network
    public_ip: str | None = None
    private_ip: str | None = None
    tailscale_ip: str | None = None
    ssh_port: int = 22

    # State
    state: InstanceState = InstanceState.UNKNOWN
    created_at: datetime | None = None
    last_seen: datetime | None = None

    # Resources
    gpu_type: str | None = None
    gpu_count: int = 0
    gpu_memory_gb: int = 0
    cpu_count: int = 0
    memory_gb: int = 0

    # Cost (hourly rate in USD)
    hourly_cost: float = 0.0

    # Provider-specific metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ssh_host(self) -> str | None:
        """Get best SSH host (prefer Tailscale, then private, then public)."""
        return self.tailscale_ip or self.private_ip or self.public_ip

    def __str__(self) -> str:
        gpu_str = f"{self.gpu_count}x {self.gpu_type}" if self.gpu_type else "CPU"
        return f"{self.name} ({self.provider.value}) - {gpu_str} - {self.state.value}"


@dataclass
class ProviderHealthCheckResult:
    """Result of a provider-specific health check.

    December 2025: Renamed from HealthCheckResult to avoid collision with
    app.coordination.contracts.HealthCheckResult (canonical for coordinators).
    """

    healthy: bool
    check_type: str
    message: str
    latency_ms: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    details: dict[str, Any] = field(default_factory=dict)


# Backward-compat alias (deprecated Dec 2025)
HealthCheckResult = ProviderHealthCheckResult


@dataclass
class RecoveryResult:
    """Result of a recovery action."""

    success: bool
    action: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: dict[str, Any] = field(default_factory=dict)


class ProviderManager(ABC):
    """Abstract base class for provider managers.

    Each provider (Lambda, Vast, Hetzner, AWS) implements this interface
    for consistent instance management.
    """

    provider: Provider

    @abstractmethod
    async def list_instances(self) -> list[ProviderInstance]:
        """List all instances from this provider."""
        pass

    @abstractmethod
    async def get_instance(self, instance_id: str) -> ProviderInstance | None:
        """Get details of a specific instance."""
        pass

    @abstractmethod
    async def check_health(self, instance: ProviderInstance) -> HealthCheckResult:
        """Check health of an instance."""
        pass

    async def reboot_instance(self, instance_id: str) -> bool:
        """Reboot an instance. Override if supported."""
        logger.warning(f"{self.provider.value}: reboot not supported")
        return False

    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate an instance. Override if supported."""
        logger.warning(f"{self.provider.value}: terminate not supported")
        return False

    async def launch_instance(self, config: dict[str, Any]) -> str | None:
        """Launch a new instance. Override if supported."""
        logger.warning(f"{self.provider.value}: launch not supported")
        return None

    async def run_ssh_command(
        self,
        instance: ProviderInstance,
        command: str,
        timeout: int = 30,
    ) -> tuple[int, str, str]:
        """Run SSH command on instance.

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        if not instance.ssh_host:
            return -1, "", "No SSH host available"

        ssh_cmd = [
            "ssh",
            "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "BatchMode=yes",
            "-p", str(instance.ssh_port),
        ]

        # Add key if in metadata
        if ssh_key := instance.metadata.get("ssh_key"):
            from pathlib import Path
            ssh_cmd.extend(["-i", str(Path(ssh_key).expanduser())])

        ssh_user = instance.metadata.get("ssh_user", "ubuntu")
        ssh_cmd.append(f"{ssh_user}@{instance.ssh_host}")
        ssh_cmd.append(command)

        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
            return proc.returncode or 0, stdout.decode(), stderr.decode()
        except asyncio.TimeoutError:
            return -1, "", f"SSH timeout after {timeout}s"
        except Exception as e:
            return -1, "", str(e)

    async def check_ssh_connectivity(
        self,
        instance: ProviderInstance,
    ) -> HealthCheckResult:
        """Check if SSH is reachable."""
        import time

        start = time.time()
        code, stdout, stderr = await self.run_ssh_command(instance, "echo ok", timeout=15)
        latency = (time.time() - start) * 1000

        if code == 0 and "ok" in stdout:
            return HealthCheckResult(
                healthy=True,
                check_type="ssh",
                message="SSH connectivity OK",
                latency_ms=latency,
            )
        return HealthCheckResult(
            healthy=False,
            check_type="ssh",
            message=f"SSH failed: {stderr or 'no response'}",
            latency_ms=latency,
        )

    async def check_p2p_health(
        self,
        instance: ProviderInstance,
        port: int = P2P_DEFAULT_PORT,
    ) -> HealthCheckResult:
        """Check if P2P daemon is responding."""
        import time

        start = time.time()
        cmd = f"curl -s --connect-timeout 5 http://localhost:{port}/health"
        code, stdout, stderr = await self.run_ssh_command(instance, cmd, timeout=20)
        latency = (time.time() - start) * 1000

        if code == 0 and stdout.strip():
            return HealthCheckResult(
                healthy=True,
                check_type="p2p",
                message="P2P daemon responding",
                latency_ms=latency,
                details={"response": stdout.strip()[:200]},
            )
        return HealthCheckResult(
            healthy=False,
            check_type="p2p",
            message=f"P2P daemon not responding: {stderr or 'timeout'}",
            latency_ms=latency,
        )

    async def check_tailscale(
        self,
        instance: ProviderInstance,
    ) -> HealthCheckResult:
        """Check Tailscale connectivity."""
        import time

        start = time.time()
        cmd = "tailscale status --json 2>/dev/null | head -1"
        code, stdout, stderr = await self.run_ssh_command(instance, cmd, timeout=15)
        latency = (time.time() - start) * 1000

        if code == 0 and stdout.strip().startswith("{"):
            return HealthCheckResult(
                healthy=True,
                check_type="tailscale",
                message="Tailscale connected",
                latency_ms=latency,
            )
        return HealthCheckResult(
            healthy=False,
            check_type="tailscale",
            message=f"Tailscale not connected: {stderr or 'no status'}",
            latency_ms=latency,
        )

    async def get_utilization(
        self,
        instance: ProviderInstance,
    ) -> dict[str, float]:
        """Get CPU/GPU/memory utilization."""
        # CPU and memory
        cmd = """python3 -c "
import psutil
import json
print(json.dumps({
    'cpu_percent': psutil.cpu_percent(interval=0.1),
    'memory_percent': psutil.virtual_memory().percent,
    'disk_percent': psutil.disk_usage('/').percent
}))
" 2>/dev/null"""

        code, stdout, stderr = await self.run_ssh_command(instance, cmd, timeout=15)

        result = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "disk_percent": 0.0,
            "gpu_percent": 0.0,
            "gpu_memory_percent": 0.0,
        }

        if code == 0 and stdout.strip():
            try:
                import json
                data = json.loads(stdout.strip())
                result.update(data)
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        # GPU (if available)
        gpu_cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null"
        code, stdout, stderr = await self.run_ssh_command(instance, gpu_cmd, timeout=10)

        if code == 0 and stdout.strip():
            try:
                parts = stdout.strip().split(",")
                if len(parts) >= 3:
                    result["gpu_percent"] = float(parts[0].strip())
                    mem_used = float(parts[1].strip())
                    mem_total = float(parts[2].strip())
                    if mem_total > 0:
                        result["gpu_memory_percent"] = (mem_used / mem_total) * 100
            except (ValueError, TypeError, IndexError):
                pass

        return result
