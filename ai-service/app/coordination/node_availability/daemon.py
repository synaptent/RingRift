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

from app.coordination.base_daemon import BaseDaemon, DaemonConfig
from app.coordination.contracts import HealthCheckResult
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

logger = logging.getLogger(__name__)

# Singleton instance
_daemon_instance: Optional["NodeAvailabilityDaemon"] = None


@dataclass
class NodeAvailabilityConfig(DaemonConfig):
    """Configuration for NodeAvailabilityDaemon."""

    # Override base config
    check_interval_seconds: float = 300.0  # 5 minutes

    # Daemon-specific settings
    dry_run: bool = True  # Log only, no YAML writes
    grace_period_seconds: float = 60.0  # Wait before marking as terminated

    # Per-provider toggles
    vast_enabled: bool = True
    lambda_enabled: bool = True
    runpod_enabled: bool = True
    vultr_enabled: bool = True
    hetzner_enabled: bool = True

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

        # P2P integration
        config.auto_update_voters = os.environ.get(f"{prefix}_AUTO_VOTERS", "0").lower() in ("1", "true")

        return config


@dataclass
class DaemonStats:
    """Statistics for the daemon."""

    cycles_completed: int = 0
    last_cycle_time: Optional[datetime] = None
    last_cycle_duration_seconds: float = 0.0

    # Per-provider stats
    provider_checks: dict[str, int] = field(default_factory=dict)
    provider_errors: dict[str, int] = field(default_factory=dict)

    # Update stats
    total_updates: int = 0
    nodes_updated: int = 0
    dry_run_updates: int = 0

    def record_cycle(self, duration: float) -> None:
        """Record a completed check cycle."""
        self.cycles_completed += 1
        self.last_cycle_time = datetime.now()
        self.last_cycle_duration_seconds = duration

    def record_provider_check(self, provider: str, success: bool) -> None:
        """Record a provider check result."""
        self.provider_checks[provider] = self.provider_checks.get(provider, 0) + 1
        if not success:
            self.provider_errors[provider] = self.provider_errors.get(provider, 0) + 1

    def record_update(self, result: ConfigUpdateResult) -> None:
        """Record an update result."""
        self.total_updates += 1
        self.nodes_updated += result.update_count
        if result.dry_run:
            self.dry_run_updates += 1


class NodeAvailabilityDaemon(BaseDaemon[NodeAvailabilityConfig]):
    """Daemon that synchronizes provider instance state with config.

    Periodically queries all cloud provider APIs and updates
    distributed_hosts.yaml when instances change state.

    Inherits from BaseDaemon which provides:
    - Lifecycle management (start/stop)
    - Coordinator protocol registration
    - Protected main loop with error handling
    - Health check interface
    """

    def __init__(self, config: Optional[NodeAvailabilityConfig] = None):
        """Initialize the daemon.

        Args:
            config: Optional configuration (uses env if None)
        """
        super().__init__(config)
        self._stats = DaemonStats()
        self._config_updater = ConfigUpdater(dry_run=self.config.dry_run)
        self._checkers: dict[str, StateChecker] = {}
        self._pending_terminations: dict[str, float] = {}  # node -> first_seen_timestamp

        # Initialize provider checkers
        self._init_checkers()

    @staticmethod
    def _get_default_config() -> NodeAvailabilityConfig:
        """Return default configuration."""
        return NodeAvailabilityConfig.from_env()

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
                    self._stats.record_provider_check(provider, success=True)
                except (OSError, asyncio.TimeoutError, ValueError, KeyError) as e:
                    # OSError: network/file issues, TimeoutError: API timeout
                    # ValueError: JSON/data parsing, KeyError: missing response fields
                    logger.error(f"Error checking {provider}: {e}")
                    self._stats.record_provider_check(provider, success=False)

            # Apply updates if any
            if all_updates:
                result = await self._config_updater.update_node_statuses(
                    all_updates,
                    reason="provider_sync",
                )
                self._stats.record_update(result)

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
            self._stats.record_cycle(duration)
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
        is_healthy = True
        messages = []

        # Check if any checkers are enabled
        if not self._checkers:
            messages.append("No provider checkers enabled")
            is_healthy = False

        # Check for recent errors
        for provider, error_count in self._stats.provider_errors.items():
            check_count = self._stats.provider_checks.get(provider, 0)
            if check_count > 0 and error_count / check_count > 0.5:
                messages.append(f"{provider} has high error rate ({error_count}/{check_count})")
                is_healthy = False

        return HealthCheckResult(
            healthy=is_healthy,
            message="; ".join(messages) if messages else "OK",
            details={
                "enabled_providers": list(self._checkers.keys()),
                "cycles_completed": self._stats.cycles_completed,
                "last_cycle": self._stats.last_cycle_time.isoformat() if self._stats.last_cycle_time else None,
                "total_updates": self._stats.total_updates,
                "nodes_updated": self._stats.nodes_updated,
                "dry_run": self.config.dry_run,
                "provider_checks": self._stats.provider_checks,
                "provider_errors": self._stats.provider_errors,
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
                "cycles_completed": self._stats.cycles_completed,
                "last_cycle_time": self._stats.last_cycle_time.isoformat() if self._stats.last_cycle_time else None,
                "last_cycle_duration": self._stats.last_cycle_duration_seconds,
                "total_updates": self._stats.total_updates,
                "nodes_updated": self._stats.nodes_updated,
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

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        The daemon instance
    """
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = NodeAvailabilityDaemon(config)
    return _daemon_instance


def reset_daemon_instance() -> None:
    """Reset the singleton instance (for testing)."""
    global _daemon_instance
    _daemon_instance = None
