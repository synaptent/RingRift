"""Adaptive dead peer cooldown with probe-based early recovery.

This module replaces the hardcoded 1-hour dead peer cooldown with an adaptive
system that:
1. Uses tiered cooldowns based on recent failure frequency
2. Allows probe-based early recovery when a node comes back online
3. Reduces technical debt by making cooldown behavior configurable

The 1-hour cooldown was causing 25-40% of nodes to be excluded after brief
network blips. This adaptive approach starts with a 30-second cooldown and
only escalates with repeated failures.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Tiered cooldown: starts low, increases with repeated failures within window
COOLDOWN_TIERS: dict[int, float] = {
    0: 30.0,    # First failure: 30 seconds
    1: 120.0,   # Second failure within window: 2 minutes
    2: 600.0,   # Third failure: 10 minutes
    3: 1800.0,  # Max: 30 minutes (down from 60!)
}

# Track failures within this window for tier calculation
FAILURE_WINDOW_SECONDS: float = 600.0  # 10 minutes

# Probe timeout for early recovery attempts
PROBE_TIMEOUT_SECONDS: float = 5.0


@dataclass
class DeadPeerCooldownManager:
    """Manages adaptive cooldown for dead peers with probe-based recovery.

    This replaces the static DEAD_PEER_COOLDOWN_SECONDS = 3600 with an adaptive
    system that:
    - Starts with a short cooldown (30s) for first-time failures
    - Escalates cooldown for repeated failures within a 10-minute window
    - Caps cooldown at 30 minutes (not 60 minutes)
    - Allows probe-based early recovery when gossip reports a node as alive

    Usage:
        manager = DeadPeerCooldownManager()
        manager.set_probe_func(tcp_probe_func)

        # When a node dies:
        manager.record_death(node_id)

        # When checking if we should reconnect:
        if manager.is_in_cooldown(node_id):
            # Try probe-based recovery
            if await manager.probe_and_recover(node_id, host, port):
                # Node recovered early, proceed with connection
                pass
            else:
                # Still in cooldown, skip
                return
    """

    _failure_history: dict[str, list[float]] = field(default_factory=dict)
    _dead_timestamps: dict[str, float] = field(default_factory=dict)
    _active_probes: set[str] = field(default_factory=set)
    _probe_func: Callable[[str, str, int], Awaitable[bool]] | None = field(default=None)
    _cooldown_tiers: dict[int, float] = field(default_factory=lambda: dict(COOLDOWN_TIERS))
    _failure_window: float = field(default=FAILURE_WINDOW_SECONDS)

    # Stats for monitoring
    _stats: dict[str, int] = field(default_factory=lambda: {
        "deaths_recorded": 0,
        "cooldowns_checked": 0,
        "cooldowns_active": 0,
        "probes_attempted": 0,
        "probes_successful": 0,
        "early_recoveries": 0,
    })

    def set_probe_func(self, probe_func: Callable[[str, str, int], Awaitable[bool]]) -> None:
        """Set the TCP probe function for early recovery attempts.

        Args:
            probe_func: Async function(node_id, host, port) -> bool indicating success
        """
        self._probe_func = probe_func

    def record_death(self, node_id: str) -> None:
        """Record node death and update failure history.

        This should be called when a node is marked as dead/failed.
        The failure is recorded in the history for tier calculation.

        Args:
            node_id: The unique identifier of the failed node
        """
        now = time.time()
        self._dead_timestamps[node_id] = now
        self._stats["deaths_recorded"] += 1

        # Update failure history (keep failures within window)
        history = self._failure_history.get(node_id, [])
        history = [t for t in history if now - t < self._failure_window]
        history.append(now)
        self._failure_history[node_id] = history

        tier = self.get_cooldown_tier(node_id)
        cooldown = self.get_cooldown_seconds(node_id)
        logger.info(
            f"Dead peer recorded: {node_id}, tier={tier}, cooldown={cooldown:.0f}s, "
            f"failures_in_window={len(history)}"
        )

    def get_cooldown_tier(self, node_id: str) -> int:
        """Get current cooldown tier based on recent failure count.

        Args:
            node_id: The unique identifier of the node

        Returns:
            Tier number (0-3), higher means longer cooldown
        """
        history = self._failure_history.get(node_id, [])
        now = time.time()
        recent = [t for t in history if now - t < self._failure_window]
        # Tier is based on failure count, capped at max tier
        return min(len(recent) - 1, max(self._cooldown_tiers.keys()))

    def get_cooldown_seconds(self, node_id: str) -> float:
        """Get cooldown duration for this node based on its tier.

        Args:
            node_id: The unique identifier of the node

        Returns:
            Cooldown duration in seconds
        """
        tier = self.get_cooldown_tier(node_id)
        # Ensure we have a valid tier
        tier = max(0, min(tier, max(self._cooldown_tiers.keys())))
        return self._cooldown_tiers.get(tier, self._cooldown_tiers[0])

    def is_in_cooldown(self, node_id: str) -> bool:
        """Check if node is still in cooldown period.

        Args:
            node_id: The unique identifier of the node

        Returns:
            True if node is in cooldown and should not be reconnected yet
        """
        self._stats["cooldowns_checked"] += 1

        if node_id not in self._dead_timestamps:
            return False

        dead_time = self._dead_timestamps[node_id]
        cooldown = self.get_cooldown_seconds(node_id)
        in_cooldown = time.time() - dead_time < cooldown

        if in_cooldown:
            self._stats["cooldowns_active"] += 1

        return in_cooldown

    def get_remaining_cooldown(self, node_id: str) -> float:
        """Get remaining cooldown time for a node.

        Args:
            node_id: The unique identifier of the node

        Returns:
            Remaining cooldown in seconds, or 0 if not in cooldown
        """
        if node_id not in self._dead_timestamps:
            return 0.0

        dead_time = self._dead_timestamps[node_id]
        cooldown = self.get_cooldown_seconds(node_id)
        remaining = cooldown - (time.time() - dead_time)
        return max(0.0, remaining)

    def clear_cooldown(self, node_id: str) -> None:
        """Clear cooldown for a node (e.g., after successful probe).

        This removes the node from the dead timestamps, allowing immediate
        reconnection. The failure history is preserved for tier calculation
        in case the node fails again soon.

        Args:
            node_id: The unique identifier of the node
        """
        if node_id in self._dead_timestamps:
            self._dead_timestamps.pop(node_id)
            self._stats["early_recoveries"] += 1
            logger.info(f"Cooldown cleared for {node_id} (early recovery)")

    def clear_failure_history(self, node_id: str) -> None:
        """Clear all failure history for a node.

        Use this when a node has been stable for a long period and you want
        to reset its tier to 0.

        Args:
            node_id: The unique identifier of the node
        """
        self._failure_history.pop(node_id, None)
        self._dead_timestamps.pop(node_id, None)
        logger.debug(f"Failure history cleared for {node_id}")

    async def probe_and_recover(
        self,
        node_id: str,
        host: str,
        port: int,
    ) -> bool:
        """Attempt early recovery via TCP probe.

        If a node is in cooldown but gossip reports it might be alive,
        this method attempts a TCP probe to verify. If successful, the
        cooldown is cleared and the node can be reconnected immediately.

        Args:
            node_id: The unique identifier of the node
            host: The host address to probe
            port: The port to probe

        Returns:
            True if probe succeeded and cooldown was cleared
        """
        # Don't probe if already probing this node
        if node_id in self._active_probes:
            logger.debug(f"Skipping probe for {node_id} - already in progress")
            return False

        # Don't probe if no probe function configured
        if not self._probe_func:
            logger.debug(f"Skipping probe for {node_id} - no probe function configured")
            return False

        self._active_probes.add(node_id)
        self._stats["probes_attempted"] += 1

        try:
            logger.debug(f"Probing {node_id} at {host}:{port} for early recovery")
            success = await asyncio.wait_for(
                self._probe_func(node_id, host, port),
                timeout=PROBE_TIMEOUT_SECONDS,
            )

            if success:
                self._stats["probes_successful"] += 1
                self.clear_cooldown(node_id)
                logger.info(f"Peer {node_id} recovered via probe during cooldown")
                return True
            else:
                logger.debug(f"Probe failed for {node_id} - staying in cooldown")
                return False

        except asyncio.TimeoutError:
            logger.debug(f"Probe timed out for {node_id} - staying in cooldown")
            return False
        except Exception as e:
            logger.warning(f"Probe error for {node_id}: {e}")
            return False
        finally:
            self._active_probes.discard(node_id)

    def get_all_in_cooldown(self) -> dict[str, dict[str, Any]]:
        """Get all nodes currently in cooldown with their details.

        Returns:
            Dict mapping node_id to cooldown details (tier, remaining, etc.)
        """
        result = {}
        now = time.time()

        for node_id, dead_time in self._dead_timestamps.items():
            cooldown = self.get_cooldown_seconds(node_id)
            remaining = cooldown - (now - dead_time)

            if remaining > 0:
                result[node_id] = {
                    "tier": self.get_cooldown_tier(node_id),
                    "cooldown_seconds": cooldown,
                    "remaining_seconds": remaining,
                    "dead_since": dead_time,
                    "failures_in_window": len([
                        t for t in self._failure_history.get(node_id, [])
                        if now - t < self._failure_window
                    ]),
                }

        return result

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about cooldown manager activity.

        Returns:
            Dict with counts of deaths, probes, recoveries, etc.
        """
        return {
            **self._stats,
            "nodes_in_cooldown": len(self.get_all_in_cooldown()),
            "nodes_with_history": len(self._failure_history),
            "active_probes": len(self._active_probes),
        }

    def cleanup_old_history(self, max_age_seconds: float = 3600.0) -> int:
        """Clean up old failure history entries.

        This should be called periodically to prevent unbounded memory growth.

        Args:
            max_age_seconds: Remove entries older than this (default 1 hour)

        Returns:
            Number of nodes cleaned up
        """
        now = time.time()
        cleaned = 0

        # Clean failure history
        nodes_to_clean = []
        for node_id, history in self._failure_history.items():
            # Keep only recent failures
            recent = [t for t in history if now - t < max_age_seconds]
            if not recent:
                nodes_to_clean.append(node_id)
            else:
                self._failure_history[node_id] = recent

        for node_id in nodes_to_clean:
            del self._failure_history[node_id]
            cleaned += 1

        # Clean dead timestamps for expired cooldowns
        expired = [
            node_id for node_id, dead_time in self._dead_timestamps.items()
            if now - dead_time > max_age_seconds
        ]
        for node_id in expired:
            del self._dead_timestamps[node_id]

        if cleaned or expired:
            logger.debug(
                f"Cleaned up {cleaned} history entries and {len(expired)} expired timestamps"
            )

        return cleaned + len(expired)
