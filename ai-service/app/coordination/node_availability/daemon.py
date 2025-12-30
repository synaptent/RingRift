"""Node Availability Daemon (December 2025).

Synchronizes cloud provider instance state with distributed_hosts.yaml.
Periodically queries all configured providers and updates the YAML config
when instances change state (e.g., terminated, stopped).

Key features:
- Multi-provider support (Vast.ai, Lambda, RunPod, Vultr, Hetzner)
- Atomic YAML updates with backup
- Dry-run mode for testing
- Event emission for downstream coordination
- Graceful degradation when providers unavailable

Usage:
    from app.coordination.node_availability.daemon import (
        NodeAvailabilityDaemon,
        get_node_availability_daemon,
    )

    daemon = get_node_availability_daemon()
    await daemon.start()

Environment variables:
    RINGRIFT_NODE_AVAILABILITY_ENABLED: Enable daemon (default: true)
    RINGRIFT_NODE_AVAILABILITY_DRY_RUN: Log only, no writes (default: true)
    RINGRIFT_NODE_AVAILABILITY_INTERVAL: Check interval in seconds (default: 300)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.protocols import CoordinatorStatus
from app.coordination.node_availability.config_updater import (
    ConfigUpdater,
    ConfigUpdateResult,
)
from app.coordination.node_availability.state_checker import (
    InstanceInfo,
    ProviderInstanceState,
    StateChecker,
    STATE_TO_YAML_STATUS,
)
from app.coordination.node_availability.providers.vast_checker import VastChecker
from app.coordination.node_availability.providers.lambda_checker import LambdaChecker
from app.coordination.node_availability.providers.runpod_checker import RunPodChecker
from app.coordination.node_availability.providers.tailscale_checker import TailscaleChecker

logger = logging.getLogger(__name__)


@dataclass
class DaemonStats:
    """Statistics tracking for NodeAvailabilityDaemon.

    December 2025: Re-added after HandlerBase migration.
    Tracks provider-specific metrics not covered by HandlerStats.
    """

    cycles_completed: int = 0
    last_cycle_time: Optional[datetime] = None
    last_cycle_duration_seconds: float = 0.0
    provider_checks: dict = field(default_factory=dict)  # provider -> check count
    provider_errors: dict = field(default_factory=dict)  # provider -> error count
    total_updates: int = 0
    nodes_updated: int = 0
    dry_run_updates: int = 0

    def record_cycle(self, duration_seconds: float) -> None:
        """Record a completed cycle."""
        self.cycles_completed += 1
        self.last_cycle_time = datetime.now()
        self.last_cycle_duration_seconds = duration_seconds

    def record_provider_check(self, provider: str, success: bool) -> None:
        """Record a provider check result."""
        self.provider_checks[provider] = self.provider_checks.get(provider, 0) + 1
        if not success:
            self.provider_errors[provider] = self.provider_errors.get(provider, 0) + 1

    def record_update(self, result) -> None:
        """Record an update result.

        Args:
            result: Object with update_count and dry_run attributes.
        """
        self.total_updates += 1
        self.nodes_updated += result.update_count
        if result.dry_run:
            self.dry_run_updates += 1


@dataclass
class NodeAvailabilityConfig:
    """Configuration for NodeAvailabilityDaemon.

    December 2025: Simplified - no longer inherits from DaemonConfig.
    HandlerBase uses cycle_interval directly.
    """

    # Check interval (passed to HandlerBase as cycle_interval)
    check_interval_seconds: float = 300.0  # 5 minutes

    # Daemon control
    enabled: bool = True

    # Daemon-specific settings
    dry_run: bool = True  # Log only, no YAML writes
    grace_period_seconds: float = 60.0  # Wait before marking as terminated

    # Per-provider toggles
    vast_enabled: bool = True
    lambda_enabled: bool = True
    runpod_enabled: bool = True
    vultr_enabled: bool = True
    hetzner_enabled: bool = True

    # Tailscale mesh connectivity checker (December 2025)
    # Unlike cloud provider checkers, this checks actual P2P connectivity
    tailscale_enabled: bool = True

    # P2P integration
    auto_update_voters: bool = False  # Auto-remove offline nodes from P2P voters

    @classmethod
    def from_env(cls, prefix: str = "RINGRIFT_NODE_AVAILABILITY") -> "NodeAvailabilityConfig":
        """Load configuration from environment variables."""
        config = cls()
        config.enabled = os.environ.get(f"{prefix}_ENABLED", "1").lower() in ("1", "true")
        config.dry_run = os.environ.get(f"{prefix}_DRY_RUN", "1").lower() in ("1", "true")

        if interval := os.environ.get(f"{prefix}_INTERVAL"):
            config.check_interval_seconds = float(interval)

        if grace := os.environ.get(f"{prefix}_GRACE_PERIOD"):
            config.grace_period_seconds = float(grace)

        # Provider toggles
        config.vast_enabled = os.environ.get(f"{prefix}_VAST", "1").lower() in ("1", "true")
        config.lambda_enabled = os.environ.get(f"{prefix}_LAMBDA", "1").lower() in ("1", "true")
        config.runpod_enabled = os.environ.get(f"{prefix}_RUNPOD", "1").lower() in ("1", "true")
        config.vultr_enabled = os.environ.get(f"{prefix}_VULTR", "1").lower() in ("1", "true")
        config.hetzner_enabled = os.environ.get(f"{prefix}_HETZNER", "1").lower() in ("1", "true")
        config.tailscale_enabled = os.environ.get(f"{prefix}_TAILSCALE", "1").lower() in ("1", "true")

        # P2P integration
        config.auto_update_voters = os.environ.get(f"{prefix}_AUTO_VOTERS", "0").lower() in ("1", "true")

        return config


class NodeAvailabilityDaemon(HandlerBase):
    """Daemon that synchronizes provider instance state with config.

    Periodically queries all cloud provider APIs and updates
    distributed_hosts.yaml when instances change state.

    December 2025: Migrated to HandlerBase pattern.
    - Uses HandlerBase singleton (get_instance/reset_instance)
    - Uses _stats for metrics tracking
    """

    def __init__(self, config: Optional[NodeAvailabilityConfig] = None):
        """Initialize the daemon.

        Args:
            config: Optional configuration (uses env if None)
        """
        self._daemon_config = config or NodeAvailabilityConfig.from_env()

        super().__init__(
            name="NodeAvailabilityDaemon",
            config=self._daemon_config,
            cycle_interval=float(self._daemon_config.check_interval_seconds),
        )

        # Daemon-specific stats (supplement HandlerBase._stats)
        self._daemon_stats = DaemonStats()
        # Legacy direct attributes for backward compatibility
        self._provider_checks: dict[str, int] = {}
        self._provider_errors: dict[str, int] = {}
        self._total_updates = 0
        self._nodes_updated = 0
        self._dry_run_updates = 0

        self._config_updater = ConfigUpdater(dry_run=self.config.dry_run)
        self._checkers: dict[str, StateChecker] = {}
        self._pending_terminations: dict[str, float] = {}  # node -> first_seen_timestamp

        # Initialize provider checkers
        self._init_checkers()

        # Log startup mode prominently
        if self.config.dry_run:
            logger.warning(
                "NodeAvailabilityDaemon started in DRY-RUN mode "
                "(set RINGRIFT_NODE_AVAILABILITY_DRY_RUN=0 to enable config updates)"
            )
        else:
            logger.info(
                "NodeAvailabilityDaemon started with config updates ENABLED"
            )

    @property
    def config(self) -> NodeAvailabilityConfig:
        """Get daemon configuration."""
        return self._daemon_config

    def _record_provider_check(self, provider: str, success: bool) -> None:
        """Record a provider check result."""
        self._daemon_stats.record_provider_check(provider, success)
        # Also update legacy attributes for backward compatibility
        self._provider_checks[provider] = self._provider_checks.get(provider, 0) + 1
        if not success:
            self._provider_errors[provider] = self._provider_errors.get(provider, 0) + 1

    def _record_update(self, result: ConfigUpdateResult) -> None:
        """Record an update result."""
        self._daemon_stats.record_update(result)
        # Also update legacy attributes for backward compatibility
        self._total_updates += 1
        self._nodes_updated += result.update_count
        if result.dry_run:
            self._dry_run_updates += 1

    def _init_checkers(self) -> None:
        """Initialize provider state checkers based on config."""
        if self.config.vast_enabled:
            checker = VastChecker()
            if checker.is_enabled:
                self._checkers["vast"] = checker
            else:
                logger.info("Vast.ai checker disabled (no API key)")

        if self.config.lambda_enabled:
            checker = LambdaChecker()
            if checker.is_enabled:
                self._checkers["lambda"] = checker
            else:
                logger.info("Lambda Labs checker disabled (no API key)")

        if self.config.runpod_enabled:
            checker = RunPodChecker()
            if checker.is_enabled:
                self._checkers["runpod"] = checker
            else:
                logger.info("RunPod checker disabled (no API key)")

        # Tailscale mesh connectivity checker (December 2025)
        # Unlike cloud providers, this checks actual P2P connectivity
        if self.config.tailscale_enabled:
            ts_checker = TailscaleChecker()
            if ts_checker.is_enabled:
                self._checkers["tailscale"] = ts_checker
            else:
                logger.info("Tailscale checker disabled (not installed)")

        logger.info(f"Initialized {len(self._checkers)} provider checkers: {list(self._checkers.keys())}")

    async def _run_cycle(self) -> None:
        """Run one check cycle across all providers.

        Called by BaseDaemon main loop at configured interval.
        """
        start_time = time.time()

        try:
            # Load current config
            config = self._config_updater.load_config()
            hosts = config.get("hosts", {})

            # Collect updates from all providers
            all_updates: dict[str, str] = {}

            for provider, checker in self._checkers.items():
                try:
                    updates = await self._check_provider(checker, hosts)
                    all_updates.update(updates)
                    self._record_provider_check(provider, success=True)
                except (OSError, asyncio.TimeoutError, ValueError, KeyError) as e:
                    # OSError: network/file issues, TimeoutError: API timeout
                    # ValueError: JSON/data parsing, KeyError: missing response fields
                    logger.error(f"Error checking {provider}: {e}")
                    self._record_provider_check(provider, success=False)

            # Apply updates if any
            if all_updates:
                result = await self._config_updater.update_node_statuses(
                    all_updates,
                    reason="provider_sync",
                )
                self._record_update(result)

                if result.success:
                    # Emit events for each state change
                    for node, (old_status, new_status) in result.changes.items():
                        await self._emit_state_change_event(node, old_status, new_status)

                    if result.dry_run:
                        logger.info(f"[DRY RUN] Would update {result.update_count} nodes")
                    else:
                        logger.info(f"Updated {result.update_count} nodes")
                else:
                    logger.error(f"Config update failed: {result.error}")

        finally:
            duration = time.time() - start_time
            self._daemon_stats.record_cycle(duration)
            logger.debug(f"Check cycle completed in {duration:.2f}s")

    async def _check_provider(
        self,
        checker: StateChecker,
        config_hosts: dict[str, dict],
    ) -> dict[str, str]:
        """Check a single provider and return updates.

        Args:
            checker: Provider state checker
            config_hosts: Current hosts configuration

        Returns:
            Dict of node_name -> new_status for nodes that need updating
        """
        updates: dict[str, str] = {}

        # Get current instance states
        instances = await checker.get_instance_states()
        if not instances:
            # No instances returned - check for terminated nodes
            terminated = await checker.get_terminated_instances(config_hosts)
            for node in terminated:
                # Apply grace period before marking as terminated
                if self._check_grace_period(node):
                    updates[node] = "retired"
            return updates

        # Correlate with config
        instances = checker.correlate_with_config(instances, config_hosts)

        # Find nodes that need updating
        for instance in instances:
            if not instance.node_name:
                continue  # No matching config node

            current_status = config_hosts.get(instance.node_name, {}).get("status", "unknown")
            new_status = instance.yaml_status

            if current_status != new_status:
                # State changed
                if instance.state == ProviderInstanceState.TERMINATED:
                    # Apply grace period for terminations
                    if self._check_grace_period(instance.node_name):
                        updates[instance.node_name] = new_status
                else:
                    updates[instance.node_name] = new_status

        # Check for nodes that disappeared from provider
        terminated = await checker.get_terminated_instances(config_hosts)
        for node in terminated:
            if node not in updates:
                if self._check_grace_period(node):
                    updates[node] = "retired"

        return updates

    def _check_grace_period(self, node_name: str) -> bool:
        """Check if node has been in terminated state long enough.

        Args:
            node_name: Name of the node

        Returns:
            True if grace period has passed, False otherwise
        """
        now = time.time()

        if node_name not in self._pending_terminations:
            # First time seeing this termination
            self._pending_terminations[node_name] = now
            logger.debug(f"Node {node_name} first seen as terminated, starting grace period")
            return False

        first_seen = self._pending_terminations[node_name]
        elapsed = now - first_seen

        if elapsed >= self.config.grace_period_seconds:
            # Grace period passed
            del self._pending_terminations[node_name]
            return True

        logger.debug(f"Node {node_name} in grace period ({elapsed:.0f}s / {self.config.grace_period_seconds}s)")
        return False

    async def _emit_state_change_event(
        self,
        node_name: str,
        old_status: str,
        new_status: str,
    ) -> None:
        """Emit event for node state change.

        Args:
            node_name: Name of the node
            old_status: Previous status
            new_status: New status
        """
        try:
            from app.coordination.event_emitters import emit_generic_event
            from app.distributed.data_events import DataEventType

            await emit_generic_event(
                DataEventType.CLUSTER_STATUS_CHANGED,
                {
                    "node": node_name,
                    "old_status": old_status,
                    "new_status": new_status,
                    "source": "node_availability_daemon",
                    "timestamp": datetime.now().isoformat(),
                },
            )
        except Exception as e:
            logger.debug(f"Failed to emit state change event: {e}")

    def health_check(self) -> HealthCheckResult:
        """Check daemon health.

        Returns:
            HealthCheckResult with current status
        """
        # Health is based on configuration validity, not running state
        # A properly configured daemon is healthy even if not started
        is_healthy = True
        messages = []

        # Check if any checkers are enabled
        if not self._checkers:
            messages.append("No provider checkers enabled")
            is_healthy = False

        # Check for recent errors
        for provider, error_count in self._daemon_stats.provider_errors.items():
            check_count = self._daemon_stats.provider_checks.get(provider, 0)
            if check_count > 0 and error_count / check_count > 0.5:
                messages.append(f"{provider} has high error rate ({error_count}/{check_count})")
                is_healthy = False

        return HealthCheckResult(
            healthy=is_healthy,
            status=CoordinatorStatus.RUNNING if is_healthy else CoordinatorStatus.STOPPED,
            message="; ".join(messages) if messages else "OK",
            details={
                "enabled_providers": list(self._checkers.keys()),
                "cycles_completed": self._daemon_stats.cycles_completed,
                "last_cycle": self._daemon_stats.last_cycle_time.isoformat() if self._daemon_stats.last_cycle_time else None,
                "total_updates": self._daemon_stats.total_updates,
                "nodes_updated": self._daemon_stats.nodes_updated,
                "dry_run": self.config.dry_run,
                "provider_checks": self._daemon_stats.provider_checks,
                "provider_errors": self._daemon_stats.provider_errors,
                "error_count": self._stats.errors_count,
            },
        )

    def get_status(self) -> dict:
        """Get daemon status for API endpoints."""
        return {
            "running": self._running,
            "config": {
                "enabled": self.config.enabled,
                "dry_run": self.config.dry_run,
                "interval_seconds": self.config.check_interval_seconds,
                "grace_period_seconds": self.config.grace_period_seconds,
            },
            "stats": {
                "cycles_completed": self._daemon_stats.cycles_completed,
                "last_cycle_time": self._daemon_stats.last_cycle_time.isoformat() if self._daemon_stats.last_cycle_time else None,
                "last_cycle_duration": self._daemon_stats.last_cycle_duration_seconds,
                "total_updates": self._daemon_stats.total_updates,
                "nodes_updated": self._daemon_stats.nodes_updated,
            },
            "providers": {
                name: checker.get_status()
                for name, checker in self._checkers.items()
            },
            "pending_terminations": list(self._pending_terminations.keys()),
        }

    async def stop(self) -> None:
        """Stop the daemon and cleanup."""
        await super().stop()

        # Close HTTP sessions
        for checker in self._checkers.values():
            if hasattr(checker, "close"):
                await checker.close()


def get_node_availability_daemon(
    config: Optional[NodeAvailabilityConfig] = None,
) -> NodeAvailabilityDaemon:
    """Get or create the singleton NodeAvailabilityDaemon instance.

    Uses HandlerBase.get_instance() for thread-safe singleton access.

    Args:
        config: Optional configuration. Only used on first call when
            creating the instance; ignored on subsequent calls.
    """
    if config is not None:
        return NodeAvailabilityDaemon.get_instance(config=config)
    return NodeAvailabilityDaemon.get_instance()


def reset_daemon_instance() -> None:
    """Reset the singleton instance (for testing).

    Uses HandlerBase.reset_instance() for thread-safe cleanup.
    """
    NodeAvailabilityDaemon.reset_instance()
