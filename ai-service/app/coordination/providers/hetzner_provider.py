"""Hetzner cloud provider implementation.

Hetzner provides CPU-only cloud servers at low cost.
Uses hcloud CLI or REST API for instance management.

Configuration:
    Environment variable: HCLOUD_TOKEN
    Or config file: ~/.config/hcloud/cli.toml

December 30, 2025: Added circuit breaker protection for API resilience.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime

from app.coordination.providers.base import (
    CloudProvider,
    GPUType,
    Instance,
    InstanceStatus,
    ProviderType,
)
from app.distributed.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)
from app.utils.async_utils import (
    async_subprocess_run,
    SubprocessTimeoutError,
)

logger = logging.getLogger(__name__)

# Circuit breaker for Hetzner API calls (December 30, 2025)
# Hetzner has 3 CPU-only nodes for P2P voting, lower priority
_hetzner_circuit_breaker: CircuitBreaker | None = None


def get_hetzner_circuit_breaker() -> CircuitBreaker:
    """Get the Hetzner API circuit breaker singleton."""
    global _hetzner_circuit_breaker
    if _hetzner_circuit_breaker is None:
        _hetzner_circuit_breaker = CircuitBreaker(
            failure_threshold=3,  # Open after 3 consecutive failures
            recovery_timeout=45.0,  # Wait 45s before testing recovery
            half_open_max_calls=1,  # Single test call in half-open
            success_threshold=1,
            operation_type="hetzner_api",
            max_backoff=300.0,  # Cap at 5 minutes
        )
    return _hetzner_circuit_breaker


def reset_hetzner_circuit_breaker() -> None:
    """Reset the circuit breaker (for testing)."""
    global _hetzner_circuit_breaker
    if _hetzner_circuit_breaker is not None:
        _hetzner_circuit_breaker.reset_all()
    _hetzner_circuit_breaker = None

# Hetzner server types (CPU only)
HETZNER_PLANS = {
    # type: (vCPUs, RAM GB, cost_per_hour)
    "cx22": (2, 4, 0.007),
    "cx32": (4, 8, 0.014),
    "cx42": (8, 16, 0.028),
    "cx52": (16, 32, 0.056),
    "ccx13": (2, 8, 0.014),  # Dedicated CPU
    "ccx23": (4, 16, 0.028),
    "ccx33": (8, 32, 0.056),
    "ccx43": (16, 64, 0.112),
    "ccx53": (32, 128, 0.224),
    "ccx63": (48, 192, 0.336),
}


class HetznerProvider(CloudProvider):
    """Hetzner cloud provider (CPU only)."""

    def __init__(self, token: str | None = None):
        """Initialize Hetzner provider.

        Args:
            token: Hetzner API token (reads from HCLOUD_TOKEN if not specified)
        """
        self._token = token or os.environ.get("HCLOUD_TOKEN")
        self._cli_path = shutil.which("hcloud")

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.HETZNER

    @property
    def name(self) -> str:
        return "Hetzner"

    def is_configured(self) -> bool:
        """Check if hcloud CLI is available and configured."""
        return bool(self._token) or bool(self._cli_path)

    async def _run_cli(self, *args: str) -> tuple[str, str, int]:
        """Run hcloud CLI command with circuit breaker protection.

        December 30, 2025: Added circuit breaker to prevent cascading failures
        when Hetzner API is experiencing issues.

        Returns:
            Tuple of (stdout, stderr, returncode)

        Raises:
            RuntimeError: If CLI not found
            CircuitOpenError: If circuit is open due to recent failures
        """
        if not self._cli_path:
            raise RuntimeError("hcloud CLI not found")

        breaker = get_hetzner_circuit_breaker()
        target = "hetzner_api"

        # Check circuit state before attempting call
        if not breaker.can_execute(target):
            state = breaker.get_status(target)
            logger.warning(
                f"Hetzner circuit breaker is {state.state.value}, "
                f"skipping CLI call (failures={state.failure_count})"
            )
            raise CircuitOpenError(f"Hetzner API circuit is {state.state.value}")

        cmd = [self._cli_path] + list(args) + ["-o", "json"]

        env = os.environ.copy()
        if self._token:
            env["HCLOUD_TOKEN"] = self._token

        try:
            # Use native async subprocess (Dec 30, 2025 - replaced run_in_executor)
            result = await async_subprocess_run(cmd, timeout=60.0, env=env)

            # Check for API-level errors (non-zero exit code)
            if result.returncode != 0:
                # Some errors are expected (e.g., server not found) - don't trip circuit
                stderr_lower = result.stderr.lower()
                if "not found" in stderr_lower or "no servers" in stderr_lower:
                    # Expected condition, still success for circuit purposes
                    breaker.record_success(target)
                else:
                    # Actual API failure - record but don't raise
                    logger.debug(f"Hetzner CLI returned {result.returncode}: {result.stderr}")
                    breaker.record_failure(target)
            else:
                breaker.record_success(target)

            return result.stdout, result.stderr, result.returncode

        except SubprocessTimeoutError as e:
            logger.warning(f"Hetzner CLI timeout: {e}")
            breaker.record_failure(target)
            raise
        except OSError as e:
            logger.warning(f"Hetzner CLI error: {e}")
            breaker.record_failure(target)
            raise

    def _parse_instance(self, data: dict) -> Instance:
        """Parse Hetzner server JSON into Instance object."""
        status_map = {
            "initializing": InstanceStatus.PENDING,
            "starting": InstanceStatus.STARTING,
            "running": InstanceStatus.RUNNING,
            "stopping": InstanceStatus.STOPPING,
            "off": InstanceStatus.STOPPED,
            "deleting": InstanceStatus.TERMINATED,
        }
        status = status_map.get(data.get("status", ""), InstanceStatus.UNKNOWN)

        # Get server type info
        server_type = data.get("server_type", {}).get("name", "")
        plan_info = HETZNER_PLANS.get(server_type, (1, 1, 0.01))

        # Get IP address
        ip = None
        public_net = data.get("public_net", {})
        if public_net.get("ipv4"):
            ip = public_net["ipv4"].get("ip")

        return Instance(
            id=str(data.get("id", "")),
            provider=ProviderType.HETZNER,
            name=data.get("name", ""),
            status=status,
            gpu_type=GPUType.CPU_ONLY,
            gpu_count=0,
            gpu_memory_gb=0,
            ip_address=ip,
            ssh_port=22,
            ssh_user="root",
            created_at=datetime.fromisoformat(data["created"].replace("Z", "+00:00"))
            if data.get("created") else None,
            cost_per_hour=plan_info[2],
            region=data.get("datacenter", {}).get("name", ""),
            tags={label: "true" for label in data.get("labels", {})},
            raw_data=data,
        )

    async def list_instances(self) -> list[Instance]:
        """List all Hetzner servers."""
        stdout, stderr, rc = await self._run_cli("server", "list")

        if rc != 0:
            logger.error(f"hcloud server list failed: {stderr}")
            return []

        try:
            servers = json.loads(stdout)
            return [self._parse_instance(srv) for srv in servers]
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse hcloud output: {e}")
            return []

    async def get_instance(self, instance_id: str) -> Instance | None:
        """Get specific server by ID."""
        stdout, stderr, rc = await self._run_cli("server", "describe", instance_id)

        if rc != 0:
            return None

        try:
            data = json.loads(stdout)
            return self._parse_instance(data)
        except json.JSONDecodeError:
            return None

    # get_instance_status() uses base class default

    async def scale_up(
        self,
        gpu_type: GPUType,
        count: int = 1,
        region: str | None = None,
        name_prefix: str = "ringrift",
    ) -> list[Instance]:
        """Create new CPU servers (GPU type ignored - Hetzner is CPU only)."""
        # Use a reasonable default server type for training
        server_type = "ccx33"  # 8 dedicated vCPUs, 32GB RAM

        created = []
        for i in range(count):
            name = f"{name_prefix}-cpu-{i}"

            args = [
                "server", "create",
                "--type", server_type,
                "--image", "ubuntu-22.04",
                "--name", name,
            ]

            if region:
                args.extend(["--datacenter", region])
            else:
                args.extend(["--location", "fsn1"])  # Falkenstein

            stdout, stderr, rc = await self._run_cli(*args)

            if rc != 0:
                logger.error(f"Failed to create Hetzner server: {stderr}")
                continue

            try:
                data = json.loads(stdout)
                server_data = data.get("server", data)
                instance = self._parse_instance(server_data)
                created.append(instance)
                logger.info(f"Created Hetzner server: {instance.id}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to parse create response: {e}")

        return created

    async def scale_down(self, instance_ids: list[str]) -> dict[str, bool]:
        """Delete servers."""
        results = {}

        for instance_id in instance_ids:
            stdout, stderr, rc = await self._run_cli("server", "delete", instance_id)
            success = rc == 0

            if success:
                logger.info(f"Deleted Hetzner server: {instance_id}")
            else:
                logger.error(f"Failed to delete {instance_id}: {stderr}")

            results[instance_id] = success

        return results

    def get_cost_per_hour(self, gpu_type: GPUType) -> float:
        """Get hourly cost (CPU only, uses default server type)."""
        # Hetzner doesn't have GPUs, return CPU server cost
        return HETZNER_PLANS.get("ccx33", (0, 0, 0.056))[2]

    async def get_available_gpus(self) -> dict[GPUType, int]:
        """Hetzner has no GPUs."""
        return {GPUType.CPU_ONLY: 100}  # Effectively unlimited CPU

    def health_check(self) -> "HealthCheckResult":
        """Check provider health for CoordinatorProtocol compliance.

        December 2025: Added for daemon health monitoring integration.
        December 30, 2025: Added circuit breaker status to health check.
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        cli_available = self._cli_path is not None
        configured = self.is_configured()

        # Get circuit breaker status
        breaker = get_hetzner_circuit_breaker()
        circuit_status = breaker.get_status("hetzner_api")
        circuit_state = circuit_status.state

        if not cli_available and not self._token:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message="HetznerProvider: hcloud CLI not found and no API token",
                details={
                    "cli_path": None,
                    "has_token": False,
                    "configured": False,
                    "circuit_state": circuit_state.value,
                },
            )

        if not configured:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message="HetznerProvider: not configured",
                details={
                    "cli_path": self._cli_path,
                    "has_token": bool(self._token),
                    "configured": False,
                    "circuit_state": circuit_state.value,
                },
            )

        # Check if circuit breaker is open
        if circuit_state == CircuitState.OPEN:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"HetznerProvider: API circuit open (failures={circuit_status.failure_count})",
                details={
                    "cli_path": self._cli_path,
                    "has_token": bool(self._token),
                    "configured": True,
                    "circuit_state": circuit_state.value,
                    "circuit_failures": circuit_status.failure_count,
                    "circuit_opened_at": circuit_status.opened_at,
                },
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="HetznerProvider: configured",
            details={
                "cli_path": self._cli_path,
                "has_token": bool(self._token),
                "configured": True,
                "circuit_state": circuit_state.value,
                "circuit_failures": circuit_status.failure_count,
            },
        )
