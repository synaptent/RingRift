"""Vultr cloud provider implementation.

Uses the vultr-cli command-line tool for instance management.
Requires vultr-cli to be installed and configured with API key.

Configuration:
    ~/.vultr-cli.yaml:
        api-key: YOUR_API_KEY

December 30, 2025: Added circuit breaker protection for API resilience.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

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

# Circuit breaker for Vultr API calls (December 30, 2025)
# Vultr has 2 A100 nodes, medium priority
_vultr_circuit_breaker: CircuitBreaker | None = None


def get_vultr_circuit_breaker() -> CircuitBreaker:
    """Get the Vultr API circuit breaker singleton."""
    global _vultr_circuit_breaker
    if _vultr_circuit_breaker is None:
        _vultr_circuit_breaker = CircuitBreaker(
            failure_threshold=3,  # Open after 3 consecutive failures
            recovery_timeout=45.0,  # Wait 45s before testing recovery
            half_open_max_calls=1,  # Single test call in half-open
            success_threshold=1,
            operation_type="vultr_api",
            max_backoff=300.0,  # Cap at 5 minutes
        )
    return _vultr_circuit_breaker


def reset_vultr_circuit_breaker() -> None:
    """Reset the circuit breaker (for testing)."""
    global _vultr_circuit_breaker
    if _vultr_circuit_breaker is not None:
        _vultr_circuit_breaker.reset_all()
    _vultr_circuit_breaker = None

# Vultr GPU instance plans (as of Dec 2025)
VULTR_GPU_PLANS = {
    # plan_id: (gpu_type, gpu_memory_gb, cost_per_hour)
    "vcg-a100-1c-6g-4vram": (GPUType.A100_40GB, 20, 0.62),  # A100 20GB
    "vcg-a100-2c-12g-8vram": (GPUType.A100_40GB, 40, 1.24),  # A100 40GB
    "vcg-a100-3c-24g-16vram": (GPUType.A100_80GB, 80, 2.48),  # A100 80GB
    "vcg-h100-1c-80g": (GPUType.H100_80GB, 80, 3.99),  # H100 80GB
}

# Default SSH key name (must be added to Vultr account)
DEFAULT_SSH_KEY = "ringrift-cluster"


class VultrProvider(CloudProvider):
    """Vultr cloud provider using vultr-cli."""

    def __init__(self, cli_path: str | None = None):
        """Initialize Vultr provider.

        Args:
            cli_path: Path to vultr-cli binary (auto-detected if not specified)
        """
        self._cli_path = cli_path or self._find_cli()
        self._ssh_key_id: str | None = None

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.VULTR

    @property
    def name(self) -> str:
        return "Vultr"

    def _find_cli(self) -> str | None:
        """Find vultr-cli binary."""
        # Check common locations
        paths = [
            shutil.which("vultr-cli"),
            str(Path.home() / ".local/bin/vultr-cli"),
            "/usr/local/bin/vultr-cli",
        ]
        for path in paths:
            if path and Path(path).exists():
                return path
        return None

    def is_configured(self) -> bool:
        """Check if vultr-cli is available and configured."""
        if not self._cli_path:
            return False

        # Check if config file exists
        config_path = Path.home() / ".vultr-cli.yaml"
        return config_path.exists()

    async def _run_cli(self, *args: str) -> tuple[str, str, int]:
        """Run vultr-cli command with circuit breaker protection.

        December 30, 2025: Added circuit breaker to prevent cascading failures
        when Vultr API is experiencing issues.

        Returns:
            Tuple of (stdout, stderr, returncode)

        Raises:
            RuntimeError: If CLI not found
            CircuitOpenError: If circuit is open due to recent failures
        """
        if not self._cli_path:
            raise RuntimeError("vultr-cli not found")

        breaker = get_vultr_circuit_breaker()
        target = "vultr_api"

        # Check circuit state before attempting call
        if not breaker.can_execute(target):
            state = breaker.get_status(target)
            logger.warning(
                f"Vultr circuit breaker is {state.state.value}, "
                f"skipping CLI call (failures={state.failure_count})"
            )
            raise CircuitOpenError(f"Vultr API circuit is {state.state.value}")

        cmd = [self._cli_path] + list(args) + ["--output", "json"]

        try:
            # Use native async subprocess (Dec 30, 2025 - replaced run_in_executor)
            result = await async_subprocess_run(cmd, timeout=60.0)

            # Check for API-level errors (non-zero exit code)
            if result.returncode != 0:
                # Some errors are expected (e.g., instance not found) - don't trip circuit
                stderr_lower = result.stderr.lower()
                if "not found" in stderr_lower or "no instances" in stderr_lower:
                    # Expected condition, still success for circuit purposes
                    breaker.record_success(target)
                else:
                    # Actual API failure - record but don't raise
                    logger.debug(f"Vultr CLI returned {result.returncode}: {result.stderr}")
                    breaker.record_failure(target)
            else:
                breaker.record_success(target)

            return result.stdout, result.stderr, result.returncode

        except SubprocessTimeoutError as e:
            logger.warning(f"Vultr CLI timeout: {e}")
            breaker.record_failure(target)
            raise
        except OSError as e:
            logger.warning(f"Vultr CLI error: {e}")
            breaker.record_failure(target)
            raise

    def _parse_instance(self, data: dict) -> Instance:
        """Parse Vultr instance JSON into Instance object."""
        # Parse status
        status_map = {
            "pending": InstanceStatus.PENDING,
            "active": InstanceStatus.RUNNING,
            "suspended": InstanceStatus.STOPPED,
            "resizing": InstanceStatus.STARTING,
        }
        status = status_map.get(data.get("status", ""), InstanceStatus.UNKNOWN)

        # Parse GPU type from plan
        plan = data.get("plan", "")
        gpu_info = VULTR_GPU_PLANS.get(plan, (GPUType.UNKNOWN, 0, 0.0))

        return Instance(
            id=data.get("id", ""),
            provider=ProviderType.VULTR,
            name=data.get("label", ""),
            status=status,
            gpu_type=gpu_info[0],
            gpu_count=1,
            gpu_memory_gb=gpu_info[1],
            ip_address=data.get("main_ip"),
            ssh_port=22,
            ssh_user="root",
            created_at=datetime.fromisoformat(data["date_created"].replace("Z", "+00:00"))
            if data.get("date_created") else None,
            cost_per_hour=gpu_info[2],
            region=data.get("region", ""),
            tags=data.get("tags", {}),
            raw_data=data,
        )

    async def list_instances(self) -> list[Instance]:
        """List all Vultr instances."""
        stdout, stderr, rc = await self._run_cli("instance", "list")

        if rc != 0:
            logger.error(f"vultr-cli instance list failed: {stderr}")
            return []

        try:
            data = json.loads(stdout)
            instances = data.get("instances", [])
            return [self._parse_instance(inst) for inst in instances]
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse vultr-cli output: {e}")
            return []

    async def get_instance(self, instance_id: str) -> Instance | None:
        """Get specific instance by ID."""
        stdout, stderr, rc = await self._run_cli("instance", "get", instance_id)

        if rc != 0:
            return None

        try:
            data = json.loads(stdout)
            return self._parse_instance(data.get("instance", data))
        except (json.JSONDecodeError, KeyError):
            return None

    # get_instance_status() uses base class default

    async def _get_ssh_key_id(self) -> str | None:
        """Get SSH key ID from Vultr account."""
        if self._ssh_key_id:
            return self._ssh_key_id

        stdout, stderr, rc = await self._run_cli("ssh-key", "list")
        if rc != 0:
            return None

        try:
            data = json.loads(stdout)
            keys = data.get("ssh_keys", [])
            for key in keys:
                if key.get("name") == DEFAULT_SSH_KEY:
                    self._ssh_key_id = key.get("id")
                    return self._ssh_key_id
        except json.JSONDecodeError:
            pass

        return None

    async def scale_up(
        self,
        gpu_type: GPUType,
        count: int = 1,
        region: str | None = None,
        name_prefix: str = "ringrift",
    ) -> list[Instance]:
        """Create new GPU instances."""
        # Map GPU type to Vultr plan
        plan_map = {
            GPUType.A100_40GB: "vcg-a100-1c-6g-4vram",  # A100 20GB (cheapest)
            GPUType.A100_80GB: "vcg-a100-3c-24g-16vram",
            GPUType.H100_80GB: "vcg-h100-1c-80g",
        }

        plan = plan_map.get(gpu_type)
        if not plan:
            logger.error(f"No Vultr plan for GPU type: {gpu_type}")
            return []

        # Get SSH key
        ssh_key_id = await self._get_ssh_key_id()

        created = []
        for i in range(count):
            label = f"{name_prefix}-{gpu_type.value}-{i}"

            args = [
                "instance", "create",
                "--plan", plan,
                "--region", region or "ewr",  # Default to US East
                "--os", "2136",  # Ubuntu 22.04 x64
                "--label", label,
            ]

            if ssh_key_id:
                args.extend(["--ssh-keys", ssh_key_id])

            stdout, stderr, rc = await self._run_cli(*args)

            if rc != 0:
                logger.error(f"Failed to create instance: {stderr}")
                continue

            try:
                data = json.loads(stdout)
                instance = self._parse_instance(data.get("instance", data))
                created.append(instance)
                logger.info(f"Created Vultr instance: {instance.id}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to parse create response: {e}")

        return created

    async def scale_down(self, instance_ids: list[str]) -> dict[str, bool]:
        """Terminate instances."""
        results = {}

        for instance_id in instance_ids:
            stdout, stderr, rc = await self._run_cli("instance", "delete", instance_id)
            success = rc == 0

            if success:
                logger.info(f"Terminated Vultr instance: {instance_id}")
            else:
                logger.error(f"Failed to terminate {instance_id}: {stderr}")

            results[instance_id] = success

        return results

    def get_cost_per_hour(self, gpu_type: GPUType) -> float:
        """Get hourly cost for GPU type."""
        cost_map = {
            GPUType.A100_40GB: 0.62,  # A100 20GB actually
            GPUType.A100_80GB: 2.48,
            GPUType.H100_80GB: 3.99,
        }
        return cost_map.get(gpu_type, 0.0)

    def health_check(self) -> "HealthCheckResult":
        """Check provider health for CoordinatorProtocol compliance.

        December 2025: Added for daemon health monitoring integration.
        December 30, 2025: Added circuit breaker status to health check.
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        cli_available = self._cli_path is not None
        configured = self.is_configured()

        # Get circuit breaker status
        breaker = get_vultr_circuit_breaker()
        circuit_status = breaker.get_status("vultr_api")
        circuit_state = circuit_status.state

        if not cli_available:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message="VultrProvider: vultr-cli not found",
                details={
                    "cli_path": None,
                    "configured": False,
                    "circuit_state": circuit_state.value,
                },
            )

        if not configured:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message="VultrProvider: config file not found",
                details={
                    "cli_path": self._cli_path,
                    "configured": False,
                    "circuit_state": circuit_state.value,
                },
            )

        # Check if circuit breaker is open
        if circuit_state == CircuitState.OPEN:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"VultrProvider: API circuit open (failures={circuit_status.failure_count})",
                details={
                    "cli_path": self._cli_path,
                    "configured": True,
                    "circuit_state": circuit_state.value,
                    "circuit_failures": circuit_status.failure_count,
                    "circuit_opened_at": circuit_status.opened_at,
                },
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="VultrProvider: CLI available and configured",
            details={
                "cli_path": self._cli_path,
                "configured": True,
                "circuit_state": circuit_state.value,
                "circuit_failures": circuit_status.failure_count,
            },
        )
