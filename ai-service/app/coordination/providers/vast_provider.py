"""Vast.ai cloud provider implementation.

Vast.ai provides marketplace GPU instances at varying prices.
Uses the vastai CLI tool for instance management.

Configuration:
    ~/.vast_api_key (created by `vastai set api-key`)

December 30, 2025: Added circuit breaker protection for API resilience.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
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

logger = logging.getLogger(__name__)

# Circuit breaker for Vast.ai API calls (December 30, 2025)
# Vast.ai is marketplace-based with ephemeral instances, so we use
# conservative thresholds to avoid blocking during transient issues
_vast_circuit_breaker: CircuitBreaker | None = None


def get_vast_circuit_breaker() -> CircuitBreaker:
    """Get the Vast.ai API circuit breaker singleton."""
    global _vast_circuit_breaker
    if _vast_circuit_breaker is None:
        _vast_circuit_breaker = CircuitBreaker(
            failure_threshold=3,  # Open after 3 consecutive failures
            recovery_timeout=45.0,  # Wait 45s before testing recovery
            half_open_max_calls=1,  # Single test call in half-open
            success_threshold=1,
            operation_type="vast_api",
            max_backoff=300.0,  # Cap at 5 minutes
        )
    return _vast_circuit_breaker


def reset_vast_circuit_breaker() -> None:
    """Reset the circuit breaker (for testing)."""
    global _vast_circuit_breaker
    if _vast_circuit_breaker is not None:
        _vast_circuit_breaker.reset_all()
    _vast_circuit_breaker = None


class VastProvider(CloudProvider):
    """Vast.ai marketplace provider."""

    def __init__(self, cli_path: str | None = None):
        """Initialize Vast provider.

        Args:
            cli_path: Path to vastai CLI (auto-detected if not specified)
        """
        self._cli_path = cli_path or shutil.which("vastai")

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.VAST

    @property
    def name(self) -> str:
        return "Vast.ai"

    def is_configured(self) -> bool:
        """Check if vastai CLI is available and configured."""
        if not self._cli_path:
            return False
        # Check for API key file
        api_key_file = os.path.expanduser("~/.vast_api_key")
        return os.path.exists(api_key_file)

    async def _run_cli(self, *args: str) -> tuple[str, str, int]:
        """Run vastai CLI command with circuit breaker protection.

        December 30, 2025: Added circuit breaker to prevent cascading failures
        when Vast.ai API is experiencing issues.

        Raises:
            RuntimeError: If CLI not found
            CircuitOpenError: If circuit is open due to recent failures
        """
        if not self._cli_path:
            raise RuntimeError("vastai CLI not found")

        breaker = get_vast_circuit_breaker()
        target = "vast_api"

        # Check circuit state before attempting call
        if not breaker.can_execute(target):
            state = breaker.get_status(target)
            logger.warning(
                f"Vast.ai circuit breaker is {state.state.value}, "
                f"skipping CLI call (failures={state.failure_count})"
            )
            raise CircuitOpenError(f"Vast.ai API circuit is {state.state.value}")

        cmd = [self._cli_path] + list(args) + ["--raw"]

        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            )

            # Check for API-level errors (non-zero exit code)
            if result.returncode != 0:
                # Some errors are expected (e.g., no instances) - don't trip circuit
                stderr_lower = result.stderr.lower()
                if "no instances" in stderr_lower or "not found" in stderr_lower:
                    # Expected condition, still success for circuit purposes
                    breaker.record_success(target)
                else:
                    # Actual API failure - record but don't raise
                    logger.debug(f"Vast.ai CLI returned {result.returncode}: {result.stderr}")
                    breaker.record_failure(target)
            else:
                breaker.record_success(target)

            return result.stdout, result.stderr, result.returncode

        except subprocess.TimeoutExpired as e:
            logger.warning(f"Vast.ai CLI timeout: {e}")
            breaker.record_failure(target)
            raise
        except (OSError, subprocess.SubprocessError) as e:
            logger.warning(f"Vast.ai CLI error: {e}")
            breaker.record_failure(target)
            raise

    def _parse_instance(self, data: dict) -> Instance:
        """Parse Vast.ai instance JSON into Instance object."""
        status_map = {
            "running": InstanceStatus.RUNNING,
            "loading": InstanceStatus.STARTING,
            "exited": InstanceStatus.STOPPED,
            "created": InstanceStatus.PENDING,
        }
        actual_status = data.get("actual_status", "unknown")
        status = status_map.get(actual_status, InstanceStatus.UNKNOWN)

        # Parse GPU info using base class method
        gpu_name = data.get("gpu_name", "")
        gpu_type = GPUType.from_string(gpu_name)
        gpu_count = data.get("num_gpus", 1)

        return Instance(
            id=str(data.get("id", "")),
            provider=ProviderType.VAST,
            name=data.get("label", f"vast-{data.get('id', 'unknown')}"),
            status=status,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            gpu_memory_gb=data.get("gpu_ram", 0) / 1024,  # Convert MB to GB
            ip_address=data.get("public_ipaddr"),
            ssh_port=data.get("ssh_port", 22),
            ssh_user="root",
            created_at=datetime.fromtimestamp(data["start_date"])
            if data.get("start_date") else None,
            cost_per_hour=data.get("dph_total", 0),
            region=data.get("geolocation", ""),
            tags={"label": data.get("label", "")},
            raw_data=data,
        )

    async def list_instances(self) -> list[Instance]:
        """List all Vast.ai instances."""
        stdout, stderr, rc = await self._run_cli("show", "instances")

        if rc != 0:
            logger.error(f"vastai show instances failed: {stderr}")
            return []

        try:
            instances = json.loads(stdout)
            return [self._parse_instance(inst) for inst in instances]
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse vastai output: {e}")
            return []

    # get_instance() and get_instance_status() use base class defaults

    async def scale_up(
        self,
        gpu_type: GPUType,
        count: int = 1,
        region: str | None = None,
        name_prefix: str = "ringrift",
    ) -> list[Instance]:
        """Create new Vast.ai instances.

        Note: Vast.ai is a marketplace - this searches for and rents available offers.
        """
        # Map GPU type to Vast.ai search term
        gpu_search = {
            GPUType.RTX_3090: "RTX_3090",
            GPUType.RTX_4090: "RTX_4090",
            GPUType.RTX_5090: "RTX_5090",
            GPUType.A10: "A10",
            GPUType.A100_40GB: "A100",
            GPUType.A100_80GB: "A100_80GB",
            GPUType.H100_80GB: "H100",
            GPUType.GH200_96GB: "GH200",
        }.get(gpu_type, "")

        if not gpu_search:
            logger.error(f"No Vast search term for GPU: {gpu_type}")
            return []

        # Search for offers
        search_args = [
            "search", "offers",
            f"gpu_name={gpu_search}",
            "rentable=True",
            "reliability>0.95",
            "-o", "dph_total",  # Sort by price
        ]

        stdout, stderr, rc = await self._run_cli(*search_args)
        if rc != 0:
            logger.error(f"Vast offer search failed: {stderr}")
            return []

        try:
            offers = json.loads(stdout)
        except json.JSONDecodeError:
            return []

        if not offers:
            logger.warning(f"No Vast offers for {gpu_type}")
            return []

        created = []
        for i, offer in enumerate(offers[:count]):
            offer_id = offer.get("id")
            if not offer_id:
                continue

            # Create instance from offer
            create_args = [
                "create", "instance",
                str(offer_id),
                "--image", "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
                "--disk", "50",
                "--label", f"{name_prefix}-{i}",
            ]

            stdout, stderr, rc = await self._run_cli(*create_args)
            if rc != 0:
                logger.error(f"Failed to create Vast instance: {stderr}")
                continue

            try:
                result = json.loads(stdout)
                instance_id = result.get("new_contract")
                if instance_id:
                    # Fetch full instance info
                    instance = await self.get_instance(str(instance_id))
                    if instance:
                        created.append(instance)
                        logger.info(f"Created Vast instance: {instance_id}")
            except json.JSONDecodeError:
                pass

        return created

    async def scale_down(self, instance_ids: list[str]) -> dict[str, bool]:
        """Destroy Vast.ai instances."""
        results = {}

        for instance_id in instance_ids:
            stdout, stderr, rc = await self._run_cli("destroy", "instance", instance_id)
            success = rc == 0

            if success:
                logger.info(f"Destroyed Vast instance: {instance_id}")
            else:
                logger.error(f"Failed to destroy {instance_id}: {stderr}")

            results[instance_id] = success

        return results

    def get_cost_per_hour(self, gpu_type: GPUType) -> float:
        """Get approximate hourly cost (varies by marketplace)."""
        # Approximate median prices on Vast.ai
        cost_map = {
            GPUType.RTX_3090: 0.30,
            GPUType.RTX_4090: 0.50,
            GPUType.RTX_5090: 0.80,
            GPUType.A10: 0.35,
            GPUType.A100_40GB: 0.90,
            GPUType.A100_80GB: 1.30,
            GPUType.H100_80GB: 2.50,
            GPUType.GH200_96GB: 3.00,
        }
        return cost_map.get(gpu_type, 0.5)

    def health_check(self) -> "HealthCheckResult":
        """Check provider health for CoordinatorProtocol compliance.

        December 2025: Added for daemon health monitoring integration.
        December 30, 2025: Added circuit breaker status to health check.
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        cli_available = self._cli_path is not None
        configured = self.is_configured()

        # Get circuit breaker status
        breaker = get_vast_circuit_breaker()
        circuit_status = breaker.get_status("vast_api")
        circuit_state = circuit_status.state

        if not cli_available:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message="VastProvider: vastai CLI not found",
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
                message="VastProvider: API key not configured",
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
                message=f"VastProvider: API circuit open (failures={circuit_status.failure_count})",
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
            message="VastProvider: CLI available and configured",
            details={
                "cli_path": self._cli_path,
                "configured": True,
                "circuit_state": circuit_state.value,
                "circuit_failures": circuit_status.failure_count,
            },
        )
