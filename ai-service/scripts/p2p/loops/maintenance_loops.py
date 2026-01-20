"""Maintenance Loops for P2P Orchestrator.

December 2025: Background loops for system maintenance tasks.

Loops:
- GitUpdateLoop: Periodically checks for and applies git updates with auto-restart

Usage:
    from scripts.p2p.loops import GitUpdateLoop, GitUpdateConfig

    git_loop = GitUpdateLoop(
        check_for_updates=orchestrator._check_for_updates,
        perform_update=orchestrator._perform_git_update,
        restart_orchestrator=orchestrator._restart_orchestrator,
    )
    await git_loop.run_forever()
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

from .base import BaseLoop

from .loop_constants import LoopIntervals, AUTO_UPDATE_ENABLED

logger = logging.getLogger(__name__)


# Backward-compat alias (Sprint 10: use LoopIntervals.GIT_UPDATE_CHECK instead)
GIT_UPDATE_CHECK_INTERVAL = LoopIntervals.GIT_UPDATE_CHECK
# AUTO_UPDATE_ENABLED is now imported from loop_constants


@dataclass
class GitUpdateConfig:
    """Configuration for git update loop.

    December 2025: Extracted from p2p_orchestrator._git_update_loop
    """

    check_interval_seconds: float = GIT_UPDATE_CHECK_INTERVAL
    error_retry_seconds: float = 60.0  # Wait before retry on error
    enabled: bool = AUTO_UPDATE_ENABLED

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")
        if self.error_retry_seconds <= 0:
            raise ValueError("error_retry_seconds must be > 0")


class GitUpdateLoop(BaseLoop):
    """Background loop to periodically check for and apply git updates.

    When updates are detected:
    1. Calculates commits behind remote
    2. Performs git pull
    3. Restarts orchestrator to apply changes

    December 2025: Extracted from p2p_orchestrator._git_update_loop
    """

    def __init__(
        self,
        check_for_updates: Callable[[], tuple[bool, str | None, str | None]],
        perform_update: Callable[[], Coroutine[Any, Any, tuple[bool, str]]],
        restart_orchestrator: Callable[[], Coroutine[Any, Any, None]],
        get_commits_behind: Callable[[str, str], int] | None = None,
        config: GitUpdateConfig | None = None,
    ):
        """Initialize git update loop.

        Args:
            check_for_updates: Callback returning (has_updates, local_commit, remote_commit)
            perform_update: Async callback to perform git pull, returns (success, message)
            restart_orchestrator: Async callback to restart the orchestrator
            get_commits_behind: Optional callback to count commits behind (local, remote) -> count
            config: Update configuration
        """
        self.config = config or GitUpdateConfig()
        super().__init__(
            name="git_update",
            interval=self.config.check_interval_seconds,
        )
        self._check_for_updates = check_for_updates
        self._perform_update = perform_update
        self._restart_orchestrator = restart_orchestrator
        self._get_commits_behind = get_commits_behind

        # Statistics
        self._checks_count = 0
        self._updates_found = 0
        self._updates_applied = 0
        self._update_failures = 0

    async def _on_start(self) -> None:
        """Check if auto-update is enabled."""
        if not self.config.enabled:
            logger.info("[GitUpdate] Auto-update disabled, loop will skip checks")
        else:
            logger.info(
                f"[GitUpdate] Auto-update loop started "
                f"(interval: {self.config.check_interval_seconds}s)"
            )

    async def _run_once(self) -> None:
        """Check for updates and apply if available."""
        if not self.config.enabled:
            return

        self._checks_count += 1

        try:
            # Check for updates
            has_updates, local_commit, remote_commit = self._check_for_updates()

            if has_updates and local_commit and remote_commit:
                self._updates_found += 1

                # Calculate commits behind (if callback provided)
                commits_behind = 0
                if self._get_commits_behind:
                    commits_behind = self._get_commits_behind(local_commit, remote_commit)
                    logger.info(f"[GitUpdate] Update available: {commits_behind} commits behind")

                logger.info(f"[GitUpdate] Local:  {local_commit[:8]}")
                logger.info(f"[GitUpdate] Remote: {remote_commit[:8]}")

                # Perform update
                success, message = await self._perform_update()

                if success:
                    self._updates_applied += 1
                    logger.info("[GitUpdate] Update successful, restarting...")
                    await self._restart_orchestrator()
                else:
                    self._update_failures += 1
                    logger.warning(f"[GitUpdate] Update failed: {message}")

        except Exception as e:
            logger.warning(f"[GitUpdate] Error in update check: {e}")
            await asyncio.sleep(self.config.error_retry_seconds)

    def get_update_stats(self) -> dict[str, Any]:
        """Get update statistics."""
        return {
            "enabled": self.config.enabled,
            "checks_count": self._checks_count,
            "updates_found": self._updates_found,
            "updates_applied": self._updates_applied,
            "update_failures": self._update_failures,
            "success_rate": (
                self._updates_applied / self._updates_found * 100
                if self._updates_found > 0
                else 100.0
            ),
            **self.stats.to_dict(),
        }

    def health_check(self) -> dict[str, Any]:
        """Return health status for DaemonManager integration.

        Returns:
            HealthCheckResult-compatible dict with status, message, and details.
        """
        stats = self.get_update_stats()

        if not self.running:
            status = "ERROR"
            message = "Git update loop not running"
        elif not self.config.enabled:
            status = "HEALTHY"
            message = "Auto-update disabled"
        elif self._update_failures > 3:
            status = "DEGRADED"
            message = f"Multiple update failures: {self._update_failures}"
        else:
            status = "HEALTHY"
            message = f"Applied {self._updates_applied} updates"

        return {
            "status": status,
            "message": message,
            "details": {
                "is_running": self.running,
                "enabled": self.config.enabled,
                "checks_count": self._checks_count,
                "updates_found": self._updates_found,
                "updates_applied": self._updates_applied,
                "update_failures": self._update_failures,
                "run_count": self.stats.total_runs,
            },
        }


# =============================================================================
# Circuit Breaker Decay Loop (Sprint 17.6)
# =============================================================================


@dataclass
class CircuitBreakerDecayConfig:
    """Configuration for circuit breaker TTL decay loop.

    January 2026 Sprint 17.6: Prevents circuits from staying OPEN indefinitely.
    January 5, 2026 (Phase 3): Reduced TTL from 6h to 1h for faster node recovery.
    """

    check_interval_seconds: float = 1800.0  # Check every 30 minutes (was 1 hour)
    ttl_seconds: float = 3600.0  # Reset circuits open > 1 hour (was 6 hours)
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")


class CircuitBreakerDecayLoop(BaseLoop):
    """Background loop to periodically decay old circuit breakers.

    This prevents circuits from being stuck OPEN indefinitely after transient
    failures. Circuits that have been OPEN longer than ttl_seconds are
    automatically reset to CLOSED.

    January 2026 Sprint 17.6: Added as part of stability improvements.
    January 5, 2026 (Phase 3): TTL reduced from 6h to 1h for faster recovery.
    January 20, 2026: Added external_alive_check for gossip-integrated recovery.

    Benefits:
    - Prevents 1h+ stuck circuits blocking health checks (was 6h)
    - Reduces manual interventions from stuck states
    - Enables graceful recovery after network partitions
    - External alive check enables immediate recovery when gossip reports node alive
    """

    def __init__(
        self,
        config: CircuitBreakerDecayConfig | None = None,
    ):
        """Initialize circuit breaker decay loop.

        Args:
            config: Optional configuration (uses defaults if not provided)
        """
        self.config = config or CircuitBreakerDecayConfig()
        super().__init__(
            name="circuit_breaker_decay",
            interval=self.config.check_interval_seconds,
        )
        self._decay_count = 0
        self._last_decay_result: dict[str, Any] = {}
        # Jan 20, 2026: External alive check for gossip-integrated recovery
        self._external_alive_check: callable | None = None

    def set_external_alive_check(self, callback: callable) -> None:
        """Set callback for external alive verification (e.g., from gossip).

        January 20, 2026: Enables immediate circuit recovery when gossip
        reports a node is alive, instead of waiting for TTL expiry.

        Args:
            callback: Callable(host: str) -> bool that returns True if the
                host is known to be alive from external source (gossip/P2P).
        """
        self._external_alive_check = callback
        logger.info("[CircuitBreakerDecay] External alive check callback configured")

    async def _run_once(self) -> None:
        """Run one decay cycle."""
        if not self.config.enabled:
            return

        try:
            # Import here to avoid circular imports
            from app.coordination.circuit_breaker_base import (
                decay_all_circuit_breakers,
            )
            from app.coordination.node_circuit_breaker import (
                get_node_circuit_registry,
            )

            # Decay operation and transport circuits (uniform TTL)
            result = decay_all_circuit_breakers(self.config.ttl_seconds)

            # Also decay node circuits
            try:
                node_registry = get_node_circuit_registry()
                node_result = node_registry.decay_all_old_circuits(self.config.ttl_seconds)
                result["node_circuits"] = node_result
            except Exception as e:
                logger.debug(f"[CircuitBreakerDecay] Node registry not available: {e}")

            # Jan 5, 2026: Also decay with transport-specific TTLs
            # This provides faster recovery for transports that typically
            # recover quickly (relay: 15min, tailscale: 30min, ssh: 1hr)
            # Jan 20, 2026: Added external_alive_check for gossip-integrated recovery
            try:
                from app.distributed.circuit_breaker import (
                    decay_transport_circuit_breakers,
                )

                transport_result = decay_transport_circuit_breakers(
                    external_alive_check=self._external_alive_check
                )
                result["transport_specific_decay"] = transport_result

                # Track external recoveries separately
                external_count = len(transport_result.get("external_recovered", []))
                if external_count > 0:
                    logger.info(
                        f"[CircuitBreakerDecay] {external_count} circuits recovered via gossip"
                    )
            except Exception as e:
                logger.debug(
                    f"[CircuitBreakerDecay] Transport-specific decay not available: {e}"
                )

            self._last_decay_result = result

            # Count total decayed (including transport-specific decay)
            total_decayed = (
                result.get("operation_registry", {}).get("total_decayed", 0)
                + len(result.get("transport_breakers", {}).get("decayed", []))
                + result.get("node_circuits", {}).get("total_decayed", 0)
                + len(result.get("transport_specific_decay", {}).get("decayed", []))
            )

            if total_decayed > 0:
                self._decay_count += total_decayed
                logger.info(
                    f"[CircuitBreakerDecay] Decayed {total_decayed} old circuits "
                    f"(total lifetime: {self._decay_count})"
                )

        except Exception as e:
            logger.warning(f"[CircuitBreakerDecay] Error in decay cycle: {e}")

    def get_decay_stats(self) -> dict[str, Any]:
        """Get decay statistics."""
        return {
            "enabled": self.config.enabled,
            "ttl_seconds": self.config.ttl_seconds,
            "check_interval_seconds": self.config.check_interval_seconds,
            "total_decayed_lifetime": self._decay_count,
            "last_result": self._last_decay_result,
            **self.stats.to_dict(),
        }

    def health_check(self) -> dict[str, Any]:
        """Return health status for DaemonManager integration.

        Returns:
            HealthCheckResult-compatible dict with status, message, and details.
        """
        if not self.running:
            status = "ERROR"
            message = "Circuit breaker decay loop not running"
        elif not self.config.enabled:
            status = "HEALTHY"
            message = "CB decay disabled"
        else:
            status = "HEALTHY"
            message = f"Decayed {self._decay_count} circuits lifetime"

        return {
            "status": status,
            "message": message,
            "details": {
                "is_running": self.running,
                "enabled": self.config.enabled,
                "ttl_seconds": self.config.ttl_seconds,
                "total_decayed_lifetime": self._decay_count,
                "run_count": self.stats.total_runs,
            },
        }


__all__ = [
    "GitUpdateConfig",
    "GitUpdateLoop",
    "CircuitBreakerDecayConfig",
    "CircuitBreakerDecayLoop",
]
