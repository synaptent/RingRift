"""ConfigValidatorDaemon - Validates distributed_hosts.yaml against provider APIs.

Jan 12, 2026: Created to detect configuration drift across providers.

This daemon periodically validates the distributed_hosts.yaml configuration
against actual provider status from Lambda Labs, Vast.ai, RunPod, and Tailscale.

Features:
1. Checks if hosts in config actually exist in provider APIs
2. Detects IP address mismatches
3. Identifies terminated instances still in config
4. Validates Tailscale connectivity for all nodes

Events emitted:
- CONFIG_VALIDATION_COMPLETED: Validation passed with all nodes valid
- CONFIG_VALIDATION_WARNING: Minor issues (some nodes offline but reachable)
- CONFIG_VALIDATION_ERROR: Critical issues (wrong IPs, missing nodes)

Usage:
    from app.coordination.config_validator_daemon import (
        ConfigValidatorDaemon,
        get_config_validator_daemon,
    )

    daemon = get_config_validator_daemon()
    await daemon.start()

    # Manual validation
    result = await daemon.validate_config()
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from app.coordination.contracts import HealthCheckResult
from app.coordination.handler_base import HandlerBase

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity level for validation issues."""

    INFO = "info"  # Informational, no action needed
    WARNING = "warning"  # Minor issue, node offline but expected
    ERROR = "error"  # Critical issue, config mismatch
    CRITICAL = "critical"  # Severe issue, config fundamentally broken


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""

    node_id: str
    provider: str
    severity: ValidationSeverity
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of config validation."""

    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    validated_at: float = 0.0
    nodes_checked: int = 0
    nodes_valid: int = 0
    nodes_warning: int = 0
    nodes_error: int = 0

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue to the result."""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.nodes_error += 1
            self.is_valid = False
        elif issue.severity == ValidationSeverity.WARNING:
            self.nodes_warning += 1
        elif issue.severity == ValidationSeverity.CRITICAL:
            self.nodes_error += 1
            self.is_valid = False


@dataclass
class ConfigValidatorConfig:
    """Configuration for ConfigValidatorDaemon."""

    # Path to distributed_hosts.yaml
    config_path: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent
        / "config"
        / "distributed_hosts.yaml"
    )

    # How often to run validation (seconds)
    validation_interval: float = 600.0  # 10 minutes

    # Whether to validate Tailscale connectivity
    validate_tailscale: bool = True

    # Whether to validate Lambda instances
    validate_lambda: bool = True

    # Whether to validate Vast.ai instances
    validate_vast: bool = True

    # Whether to validate RunPod instances
    validate_runpod: bool = True

    # Timeout for provider API calls (seconds)
    api_timeout: float = 30.0

    # Emit events on validation complete
    emit_events: bool = True

    # Cache TTL for provider data (seconds)
    cache_ttl: float = 300.0


class ConfigValidatorDaemon(HandlerBase):
    """Daemon for validating config against provider APIs.

    Checks distributed_hosts.yaml against Lambda Labs, Vast.ai,
    RunPod, and Tailscale to detect configuration drift.
    """

    _event_source = "ConfigValidatorDaemon"
    _instance: ConfigValidatorDaemon | None = None
    _instance_lock = asyncio.Lock()

    def __init__(self, config: ConfigValidatorConfig | None = None) -> None:
        """Initialize ConfigValidatorDaemon.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self._config = config or ConfigValidatorConfig()
        super().__init__(
            name="config_validator", cycle_interval=self._config.validation_interval
        )

        self._last_result: ValidationResult | None = None
        self._provider_cache: dict[str, tuple[float, Any]] = {}

    @classmethod
    async def get_instance(cls) -> ConfigValidatorDaemon:
        """Get singleton instance."""
        async with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """No events subscribed - validation is self-triggered."""
        return {}

    async def _run_cycle(self) -> None:
        """Main daemon cycle - run validation."""
        try:
            # Feb 2026: Guard against _config being None after daemon restart
            if self._config is None:
                self._config = ConfigValidatorConfig()
            result = await self.validate_config()
            self._last_result = result

            if self._config.emit_events:
                await self._emit_validation_events(result)

        except Exception as e:
            logger.error(f"[ConfigValidator] Cycle error: {e}")
            self._stats.errors_count += 1

    async def validate_config(self) -> ValidationResult:
        """Validate distributed_hosts.yaml against provider APIs.

        Returns:
            ValidationResult with all issues found
        """
        result = ValidationResult(is_valid=True, validated_at=time.time())

        try:
            # Load config
            if not self._config.config_path.exists():
                result.add_issue(
                    ValidationIssue(
                        node_id="",
                        provider="config",
                        severity=ValidationSeverity.CRITICAL,
                        message="Config file not found",
                        details={"path": str(self._config.config_path)},
                    )
                )
                return result

            config = yaml.safe_load(self._config.config_path.read_text())
            hosts = config.get("hosts", {})

            # Group hosts by provider
            provider_hosts: dict[str, list[tuple[str, dict]]] = {
                "lambda": [],
                "vast": [],
                "runpod": [],
                "nebius": [],
                "vultr": [],
                "hetzner": [],
                "local": [],
                "unknown": [],
            }

            for node_id, host_cfg in hosts.items():
                provider = self._detect_provider(node_id, host_cfg)
                provider_hosts[provider].append((node_id, host_cfg))
                result.nodes_checked += 1

            # Validate each provider
            if self._config.validate_tailscale:
                tailscale_issues = await self._validate_tailscale(hosts)
                for issue in tailscale_issues:
                    result.add_issue(issue)

            if self._config.validate_lambda and provider_hosts["lambda"]:
                lambda_issues = await self._validate_lambda(provider_hosts["lambda"])
                for issue in lambda_issues:
                    result.add_issue(issue)

            if self._config.validate_vast and provider_hosts["vast"]:
                vast_issues = await self._validate_vast(provider_hosts["vast"])
                for issue in vast_issues:
                    result.add_issue(issue)

            if self._config.validate_runpod and provider_hosts["runpod"]:
                runpod_issues = await self._validate_runpod(provider_hosts["runpod"])
                for issue in runpod_issues:
                    result.add_issue(issue)

            # Jan 15, 2026: Validate relay configuration
            # Phase 4 of P2P Resilience Plan - checks relay node availability
            relay_issues = self._validate_relay_config(hosts)
            for issue in relay_issues:
                result.add_issue(issue)

            # Count valid nodes
            result.nodes_valid = (
                result.nodes_checked - result.nodes_error - result.nodes_warning
            )

            logger.info(
                f"[ConfigValidator] Validation complete: "
                f"{result.nodes_valid}/{result.nodes_checked} valid, "
                f"{result.nodes_warning} warnings, {result.nodes_error} errors"
            )

            return result

        except Exception as e:
            logger.error(f"[ConfigValidator] Validation error: {e}")
            result.add_issue(
                ValidationIssue(
                    node_id="",
                    provider="validator",
                    severity=ValidationSeverity.ERROR,
                    message=f"Validation failed: {e}",
                )
            )
            return result

    def _detect_provider(self, node_id: str, host_cfg: dict) -> str:
        """Detect provider from node_id or config."""
        node_lower = node_id.lower()

        if "lambda" in node_lower or "gh200" in node_lower:
            return "lambda"
        if "vast" in node_lower:
            return "vast"
        if "runpod" in node_lower:
            return "runpod"
        if "nebius" in node_lower:
            return "nebius"
        if "vultr" in node_lower:
            return "vultr"
        if "hetzner" in node_lower:
            return "hetzner"
        if "local" in node_lower or "mac" in node_lower:
            return "local"

        return "unknown"

    async def _validate_tailscale(
        self, hosts: dict[str, dict]
    ) -> list[ValidationIssue]:
        """Validate all hosts have valid Tailscale IPs.

        Args:
            hosts: Dict of node_id -> host config

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        try:
            # Get Tailscale status
            tailscale_peers = await self._get_cached_data(
                "tailscale", self._fetch_tailscale_status
            )

            if tailscale_peers is None:
                issues.append(
                    ValidationIssue(
                        node_id="",
                        provider="tailscale",
                        severity=ValidationSeverity.WARNING,
                        message="Could not fetch Tailscale status",
                    )
                )
                return issues

            # Build IP -> hostname map
            tailscale_ips = {
                ip: info for ip, info in tailscale_peers.items() if ip
            }

            # Check each host
            for node_id, host_cfg in hosts.items():
                ts_ip = host_cfg.get("tailscale_ip")

                if not ts_ip:
                    # No Tailscale IP configured - might be expected for local nodes
                    if self._detect_provider(node_id, host_cfg) not in ("local", "hetzner"):
                        issues.append(
                            ValidationIssue(
                                node_id=node_id,
                                provider="tailscale",
                                severity=ValidationSeverity.INFO,
                                message="No tailscale_ip configured",
                            )
                        )
                    continue

                if ts_ip not in tailscale_ips:
                    issues.append(
                        ValidationIssue(
                            node_id=node_id,
                            provider="tailscale",
                            severity=ValidationSeverity.ERROR,
                            message=f"Tailscale IP {ts_ip} not found in network",
                            details={"configured_ip": ts_ip},
                        )
                    )
                elif not tailscale_ips[ts_ip].get("online", False):
                    issues.append(
                        ValidationIssue(
                            node_id=node_id,
                            provider="tailscale",
                            severity=ValidationSeverity.WARNING,
                            message=f"Node offline in Tailscale",
                            details={
                                "configured_ip": ts_ip,
                                "last_seen": tailscale_ips[ts_ip].get("last_seen", "unknown"),
                            },
                        )
                    )

            return issues

        except Exception as e:
            logger.debug(f"[ConfigValidator] Tailscale validation error: {e}")
            issues.append(
                ValidationIssue(
                    node_id="",
                    provider="tailscale",
                    severity=ValidationSeverity.WARNING,
                    message=f"Tailscale validation failed: {e}",
                )
            )
            return issues

    async def _validate_lambda(
        self, hosts: list[tuple[str, dict]]
    ) -> list[ValidationIssue]:
        """Validate Lambda Labs hosts against API.

        Args:
            hosts: List of (node_id, host_cfg) tuples for Lambda nodes

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        try:
            # Get Lambda instances
            lambda_instances = await self._get_cached_data(
                "lambda", self._fetch_lambda_instances
            )

            if lambda_instances is None:
                issues.append(
                    ValidationIssue(
                        node_id="",
                        provider="lambda",
                        severity=ValidationSeverity.INFO,
                        message="Could not fetch Lambda instances (API key missing?)",
                    )
                )
                return issues

            # Build instance map by IP
            instance_ips = {
                inst.get("ip"): inst
                for inst in lambda_instances
                if inst.get("ip")
            }

            # Check each host
            for node_id, host_cfg in hosts:
                ssh_host = host_cfg.get("ssh_host") or host_cfg.get("tailscale_ip")

                if not ssh_host:
                    continue

                # Check if IP exists in Lambda
                if ssh_host not in instance_ips:
                    issues.append(
                        ValidationIssue(
                            node_id=node_id,
                            provider="lambda",
                            severity=ValidationSeverity.WARNING,
                            message=f"IP {ssh_host} not found in Lambda instances",
                            details={"configured_ip": ssh_host},
                        )
                    )
                else:
                    instance = instance_ips[ssh_host]
                    if instance.get("status") != "active":
                        issues.append(
                            ValidationIssue(
                                node_id=node_id,
                                provider="lambda",
                                severity=ValidationSeverity.WARNING,
                                message=f"Lambda instance not active: {instance.get('status')}",
                                details={
                                    "status": instance.get("status"),
                                    "instance_id": instance.get("id"),
                                },
                            )
                        )

            return issues

        except Exception as e:
            logger.debug(f"[ConfigValidator] Lambda validation error: {e}")
            return issues

    async def _validate_vast(
        self, hosts: list[tuple[str, dict]]
    ) -> list[ValidationIssue]:
        """Validate Vast.ai hosts against CLI.

        Args:
            hosts: List of (node_id, host_cfg) tuples for Vast nodes

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        try:
            # Get Vast instances
            vast_instances = await self._get_cached_data(
                "vast", self._fetch_vast_instances
            )

            if vast_instances is None:
                issues.append(
                    ValidationIssue(
                        node_id="",
                        provider="vast",
                        severity=ValidationSeverity.INFO,
                        message="Could not fetch Vast instances (vastai CLI missing?)",
                    )
                )
                return issues

            # Build instance map by ID from node_id (e.g., vast-12345678)
            instance_ids = {
                str(inst.get("id")): inst
                for inst in vast_instances
                if inst.get("id")
            }

            # Check each host
            for node_id, host_cfg in hosts:
                # Extract instance ID from node_id (vast-12345678 -> 12345678)
                instance_id = None
                if "-" in node_id:
                    parts = node_id.split("-")
                    for part in parts:
                        if part.isdigit() and len(part) >= 6:
                            instance_id = part
                            break

                if not instance_id:
                    continue

                if instance_id not in instance_ids:
                    issues.append(
                        ValidationIssue(
                            node_id=node_id,
                            provider="vast",
                            severity=ValidationSeverity.ERROR,
                            message=f"Vast instance {instance_id} not found (terminated?)",
                            details={"instance_id": instance_id},
                        )
                    )
                else:
                    instance = instance_ids[instance_id]
                    status = instance.get("actual_status", instance.get("status", "unknown"))
                    if status != "running":
                        issues.append(
                            ValidationIssue(
                                node_id=node_id,
                                provider="vast",
                                severity=ValidationSeverity.WARNING,
                                message=f"Vast instance not running: {status}",
                                details={
                                    "status": status,
                                    "instance_id": instance_id,
                                },
                            )
                        )

            return issues

        except Exception as e:
            logger.debug(f"[ConfigValidator] Vast validation error: {e}")
            return issues

    async def _validate_runpod(
        self, hosts: list[tuple[str, dict]]
    ) -> list[ValidationIssue]:
        """Validate RunPod hosts against API.

        Args:
            hosts: List of (node_id, host_cfg) tuples for RunPod nodes

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        try:
            # Get RunPod pods
            runpod_pods = await self._get_cached_data(
                "runpod", self._fetch_runpod_pods
            )

            if runpod_pods is None:
                issues.append(
                    ValidationIssue(
                        node_id="",
                        provider="runpod",
                        severity=ValidationSeverity.INFO,
                        message="Could not fetch RunPod pods (API key missing?)",
                    )
                )
                return issues

            # Build pod map by ID
            pod_ids = {pod.get("id"): pod for pod in runpod_pods if pod.get("id")}

            # Check each host
            for node_id, host_cfg in hosts:
                # Extract pod ID from node_id or config
                pod_id = host_cfg.get("runpod_id")
                if not pod_id and "-" in node_id:
                    # Try to extract from node_id
                    parts = node_id.split("-")
                    for part in parts:
                        if len(part) >= 10:  # RunPod IDs are typically long
                            pod_id = part
                            break

                if not pod_id:
                    continue

                if pod_id not in pod_ids:
                    issues.append(
                        ValidationIssue(
                            node_id=node_id,
                            provider="runpod",
                            severity=ValidationSeverity.ERROR,
                            message=f"RunPod pod {pod_id} not found (terminated?)",
                            details={"pod_id": pod_id},
                        )
                    )
                else:
                    pod = pod_ids[pod_id]
                    status = pod.get("desiredStatus", pod.get("status", "unknown"))
                    if status not in ("RUNNING", "running"):
                        issues.append(
                            ValidationIssue(
                                node_id=node_id,
                                provider="runpod",
                                severity=ValidationSeverity.WARNING,
                                message=f"RunPod pod not running: {status}",
                                details={
                                    "status": status,
                                    "pod_id": pod_id,
                                },
                            )
                        )

            return issues

        except Exception as e:
            logger.debug(f"[ConfigValidator] RunPod validation error: {e}")
            return issues

    async def _get_cached_data(
        self, cache_key: str, fetch_func: Any
    ) -> Any | None:
        """Get data from cache or fetch fresh.

        Args:
            cache_key: Key for cache lookup
            fetch_func: Async function to fetch data if cache miss

        Returns:
            Cached or freshly fetched data, or None on error
        """
        now = time.time()

        # Check cache
        if cache_key in self._provider_cache:
            cached_at, data = self._provider_cache[cache_key]
            if now - cached_at < self._config.cache_ttl:
                return data

        # Fetch fresh data
        try:
            data = await fetch_func()
            self._provider_cache[cache_key] = (now, data)
            return data
        except Exception as e:
            logger.debug(f"[ConfigValidator] Failed to fetch {cache_key}: {e}")
            return None

    async def _fetch_tailscale_status(self) -> dict[str, dict]:
        """Fetch Tailscale peer status.

        Returns:
            Dict mapping IP to peer info
        """
        try:
            import subprocess

            result = await asyncio.to_thread(
                subprocess.run,
                ["tailscale", "status", "--json"],
                capture_output=True,
                text=True,
                timeout=self._config.api_timeout,
            )

            if result.returncode != 0:
                return {}

            import json
            status = json.loads(result.stdout)

            # Build IP -> peer info map
            peers: dict[str, dict] = {}
            for peer in status.get("Peer", {}).values():
                for ip in peer.get("TailscaleIPs", []):
                    peers[ip] = {
                        "online": peer.get("Online", False),
                        "hostname": peer.get("HostName", ""),
                        "last_seen": peer.get("LastSeen", ""),
                    }

            return peers

        except Exception as e:
            logger.debug(f"[ConfigValidator] Tailscale fetch error: {e}")
            return {}

    async def _fetch_lambda_instances(self) -> list[dict]:
        """Fetch Lambda Labs instances.

        Returns:
            List of instance dicts
        """
        try:
            from app.coordination.node_availability.providers.lambda_checker import (
                LambdaChecker,
            )

            checker = LambdaChecker()
            instances = await asyncio.to_thread(checker.get_instances)
            return instances or []

        except ImportError:
            logger.debug("[ConfigValidator] LambdaChecker not available")
            return []
        except Exception as e:
            logger.debug(f"[ConfigValidator] Lambda fetch error: {e}")
            return []

    async def _fetch_vast_instances(self) -> list[dict]:
        """Fetch Vast.ai instances.

        Returns:
            List of instance dicts
        """
        try:
            from app.coordination.node_availability.providers.vast_checker import (
                VastChecker,
            )

            checker = VastChecker()
            instances = await asyncio.to_thread(checker.get_instances)
            return instances or []

        except ImportError:
            logger.debug("[ConfigValidator] VastChecker not available")
            return []
        except Exception as e:
            logger.debug(f"[ConfigValidator] Vast fetch error: {e}")
            return []

    async def _fetch_runpod_pods(self) -> list[dict]:
        """Fetch RunPod pods.

        Returns:
            List of pod dicts
        """
        try:
            from app.coordination.node_availability.providers.runpod_checker import (
                RunPodChecker,
            )

            checker = RunPodChecker()
            pods = await asyncio.to_thread(checker.get_pods)
            return pods or []

        except ImportError:
            logger.debug("[ConfigValidator] RunPodChecker not available")
            return []
        except Exception as e:
            logger.debug(f"[ConfigValidator] RunPod fetch error: {e}")
            return []

    def _validate_relay_config(
        self, hosts: dict[str, dict]
    ) -> list[ValidationIssue]:
        """Validate relay configuration for NAT-blocked nodes.

        Jan 15, 2026: Phase 4 of P2P Resilience Plan.

        Checks:
        1. Relay nodes exist in config
        2. Relay nodes have relay_capable: true
        3. Relay nodes are not status: offline
        4. Warn if node has < 2 configured relays

        Args:
            hosts: Dict of node_id -> host config

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        # Build set of relay-capable, online nodes
        relay_capable_nodes: set[str] = set()
        for node_id, cfg in hosts.items():
            if cfg.get("relay_capable", False) and cfg.get("status") == "active":
                relay_capable_nodes.add(node_id)

        # Check each host's relay configuration
        for node_id, cfg in hosts.items():
            # Skip offline nodes
            if cfg.get("status") == "offline":
                continue

            # Check each relay level
            relay_fields = [
                "relay_primary",
                "relay_secondary",
                "relay_tertiary",
                "relay_quaternary",
            ]

            configured_relays = 0
            valid_relays = 0

            for relay_field in relay_fields:
                relay_node = cfg.get(relay_field)
                if not relay_node:
                    continue

                configured_relays += 1

                # Check relay node exists
                if relay_node not in hosts:
                    issues.append(
                        ValidationIssue(
                            node_id=node_id,
                            provider="relay",
                            severity=ValidationSeverity.ERROR,
                            message=f"{relay_field} '{relay_node}' not found in config",
                            details={"field": relay_field, "relay_node": relay_node},
                        )
                    )
                    continue

                relay_cfg = hosts[relay_node]

                # Check relay_capable flag
                if not relay_cfg.get("relay_capable", False):
                    issues.append(
                        ValidationIssue(
                            node_id=node_id,
                            provider="relay",
                            severity=ValidationSeverity.WARNING,
                            message=f"{relay_field} '{relay_node}' not relay_capable",
                            details={"field": relay_field, "relay_node": relay_node},
                        )
                    )
                    continue

                # Check relay status
                if relay_cfg.get("status") == "offline":
                    issues.append(
                        ValidationIssue(
                            node_id=node_id,
                            provider="relay",
                            severity=ValidationSeverity.ERROR,
                            message=f"{relay_field} '{relay_node}' is offline",
                            details={"field": relay_field, "relay_node": relay_node},
                        )
                    )
                    continue

                valid_relays += 1

            # Warn if fewer than 2 valid relays for a NAT-blocked node
            if configured_relays > 0 and valid_relays < 2:
                issues.append(
                    ValidationIssue(
                        node_id=node_id,
                        provider="relay",
                        severity=ValidationSeverity.WARNING,
                        message=f"Only {valid_relays} valid relay(s) configured (recommend 2+)",
                        details={
                            "configured_relays": configured_relays,
                            "valid_relays": valid_relays,
                        },
                    )
                )

        # Summary log
        if issues:
            error_count = sum(1 for i in issues if i.severity == ValidationSeverity.ERROR)
            warn_count = sum(1 for i in issues if i.severity == ValidationSeverity.WARNING)
            logger.info(
                f"[ConfigValidator] Relay validation: {error_count} errors, {warn_count} warnings"
            )

        return issues

    async def _emit_validation_events(self, result: ValidationResult) -> None:
        """Emit events based on validation result.

        Args:
            result: Validation result to emit events for
        """
        event_data = {
            "is_valid": result.is_valid,
            "nodes_checked": result.nodes_checked,
            "nodes_valid": result.nodes_valid,
            "nodes_warning": result.nodes_warning,
            "nodes_error": result.nodes_error,
            "issues_count": len(result.issues),
            "validated_at": result.validated_at,
        }

        if result.nodes_error > 0:
            event_type = "CONFIG_VALIDATION_ERROR"
            event_data["error_issues"] = [
                {"node_id": i.node_id, "provider": i.provider, "message": i.message}
                for i in result.issues
                if i.severity == ValidationSeverity.ERROR
            ]
        elif result.nodes_warning > 0:
            event_type = "CONFIG_VALIDATION_WARNING"
            event_data["warning_issues"] = [
                {"node_id": i.node_id, "provider": i.provider, "message": i.message}
                for i in result.issues
                if i.severity == ValidationSeverity.WARNING
            ]
        else:
            event_type = "CONFIG_VALIDATION_COMPLETED"

        await self._safe_emit_event_async(event_type, event_data)

    def health_check(self) -> HealthCheckResult:
        """Return health status for DaemonManager integration."""
        healthy = self._running
        details: dict[str, Any] = {
            "running": self._running,
        }

        if self._last_result:
            details.update({
                "last_validation": self._last_result.validated_at,
                "is_valid": self._last_result.is_valid,
                "nodes_checked": self._last_result.nodes_checked,
                "nodes_valid": self._last_result.nodes_valid,
                "nodes_warning": self._last_result.nodes_warning,
                "nodes_error": self._last_result.nodes_error,
            })

        return HealthCheckResult(healthy=healthy, details=details)


def get_config_validator_daemon() -> ConfigValidatorDaemon:
    """Get ConfigValidatorDaemon instance (sync wrapper).

    For async context, use ConfigValidatorDaemon.get_instance() directly.
    """
    return ConfigValidatorDaemon()
