"""
Work Discovery Manager - Multi-channel work discovery for P2P cluster resilience.

January 4, 2026: Phase 5 of P2P Cluster Resilience plan.

Problem: WorkerPullLoop returns early if no leader_id, leaving workers idle.

Solution: Try multiple channels to discover work:
1. Leader work queue (fastest - normal path)
2. Peer discovery (query other peers for work)
3. Local autonomous queue (from AutonomousQueueLoop)
4. Direct selfplay (last resort when all else fails)

This ensures workers can always find work even during partitions or leader elections.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import socket
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine

import yaml

logger = logging.getLogger(__name__)

# Module-level cached check for selfplay enabled
_selfplay_enabled_checked: bool = False
_selfplay_enabled: bool | None = None


def _is_selfplay_enabled_for_node() -> bool:
    """Check if selfplay is enabled for this node via YAML config.

    This prevents coordinator nodes from spawning direct selfplay
    when WorkDiscoveryManager tries to generate fallback work.

    Jan 5, 2026: Same logic as AutonomousQueuePopulationLoop._is_selfplay_enabled_for_node()
    """
    global _selfplay_enabled_checked, _selfplay_enabled

    if _selfplay_enabled_checked:
        return _selfplay_enabled if _selfplay_enabled is not None else True

    _selfplay_enabled_checked = True

    try:
        hostname = socket.gethostname()
        hostname_lower = hostname.lower().replace("-", "").replace("_", "")

        # Try to find cluster config
        config_paths = [
            Path(__file__).parent.parent.parent.parent / "config" / "distributed_hosts.yaml",
            Path.cwd() / "config" / "distributed_hosts.yaml",
            Path("/Users/armand/Development/RingRift/ai-service/config/distributed_hosts.yaml"),
        ]

        cluster_config = None
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        cluster_config = yaml.safe_load(f)
                    break
                except (OSError, IOError):
                    # File read error
                    continue
                except yaml.YAMLError:
                    # Malformed YAML
                    continue

        if not cluster_config:
            return True

        # YAML uses "hosts" key, not "nodes"
        hosts = cluster_config.get("hosts", {})
        if hosts:
            # Search for matching host config by hostname
            for config_name, cfg in hosts.items():
                if not isinstance(cfg, dict):
                    continue
                config_name_lower = config_name.lower().replace("-", "").replace("_", "")

                # Direct hostname match
                if hostname_lower in config_name_lower or config_name_lower in hostname_lower:
                    _selfplay_enabled = cfg.get("selfplay_enabled", True)
                    if not _selfplay_enabled:
                        logger.info(
                            f"[WorkDiscovery] Node {hostname} matched config {config_name}, "
                            "selfplay_enabled=false - disabling direct selfplay"
                        )
                    return _selfplay_enabled

                # Special case: MacBook hostname maps to mac-studio or local-mac config
                if "macbook" in hostname_lower and config_name_lower in ("macstudio", "localmac"):
                    _selfplay_enabled = cfg.get("selfplay_enabled", True)
                    if not _selfplay_enabled:
                        logger.info(
                            f"[WorkDiscovery] MacBook {hostname} matched {config_name} config, "
                            "selfplay_enabled=false - disabling direct selfplay"
                        )
                    return _selfplay_enabled

        # Also check elo_sync.coordinator as an indicator
        elo_sync = cluster_config.get("elo_sync", {})
        coordinator_name = elo_sync.get("coordinator", "")
        if coordinator_name:
            coord_lower = coordinator_name.lower().replace("-", "").replace("_", "")
            if coord_lower in hostname_lower or hostname_lower in coord_lower:
                logger.info(
                    f"[WorkDiscovery] Node {hostname} matches elo_sync coordinator {coordinator_name}, "
                    "disabling direct selfplay"
                )
                _selfplay_enabled = False
                return False

        return True

    except Exception as e:
        logger.debug(f"[WorkDiscovery] Error checking selfplay_enabled config: {e}")
        return True


def set_selfplay_disabled_override(disabled: bool = True) -> None:
    """Set a runtime override to disable selfplay for this node.

    January 2026: Added to support --training-only flag in P2P orchestrator.
    This allows runtime disabling of selfplay without modifying YAML config.

    Args:
        disabled: If True, selfplay will be disabled for this node.
    """
    global _selfplay_enabled_checked, _selfplay_enabled
    _selfplay_enabled_checked = True
    _selfplay_enabled = not disabled
    logger.info(
        f"[WorkDiscovery] Selfplay override set: selfplay_enabled={not disabled}"
    )


class DiscoveryChannel(Enum):
    """Work discovery channels in priority order."""

    LEADER = "leader"  # Normal path: leader work queue
    PEER = "peer"  # Query other peers for work
    AUTONOMOUS = "autonomous"  # Local autonomous queue
    DIRECT = "direct"  # Direct selfplay generation


@dataclass
class WorkDiscoveryConfig:
    """Configuration for work discovery manager."""

    # Enable/disable channels
    leader_enabled: bool = True
    peer_discovery_enabled: bool = True
    autonomous_enabled: bool = True
    direct_selfplay_enabled: bool = True

    # Peer discovery settings
    peer_query_limit: int = 3  # Max peers to query
    peer_query_timeout_seconds: float = 5.0

    # Direct selfplay settings
    direct_selfplay_game_count: int = 10  # Games per direct selfplay batch

    # Cooldowns
    peer_discovery_cooldown_seconds: float = 30.0
    direct_selfplay_cooldown_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> WorkDiscoveryConfig:
        """Create config from environment variables."""
        return cls(
            leader_enabled=os.environ.get("RINGRIFT_WORK_DISCOVERY_LEADER", "true").lower() == "true",
            peer_discovery_enabled=os.environ.get("RINGRIFT_WORK_DISCOVERY_PEER", "true").lower() == "true",
            autonomous_enabled=os.environ.get("RINGRIFT_WORK_DISCOVERY_AUTONOMOUS", "true").lower() == "true",
            direct_selfplay_enabled=os.environ.get("RINGRIFT_WORK_DISCOVERY_DIRECT", "true").lower() == "true",
            peer_query_limit=int(os.environ.get("RINGRIFT_WORK_DISCOVERY_PEER_LIMIT", "3")),
            peer_query_timeout_seconds=float(os.environ.get("RINGRIFT_WORK_DISCOVERY_PEER_TIMEOUT", "5.0")),
            direct_selfplay_game_count=int(os.environ.get("RINGRIFT_WORK_DISCOVERY_DIRECT_GAMES", "10")),
        )


@dataclass
class DiscoveryResult:
    """Result of a work discovery attempt."""

    work_item: dict[str, Any] | None
    channel: DiscoveryChannel
    duration_seconds: float
    peer_id: str | None = None  # For peer channel
    error: str | None = None


@dataclass
class WorkDiscoveryStats:
    """Statistics for work discovery."""

    attempts_by_channel: dict[str, int] = field(default_factory=lambda: {c.value: 0 for c in DiscoveryChannel})
    successes_by_channel: dict[str, int] = field(default_factory=lambda: {c.value: 0 for c in DiscoveryChannel})
    failures_by_channel: dict[str, int] = field(default_factory=lambda: {c.value: 0 for c in DiscoveryChannel})
    last_success_time: float | None = None
    last_success_channel: str | None = None

    def record_attempt(self, channel: DiscoveryChannel, success: bool) -> None:
        """Record a discovery attempt."""
        self.attempts_by_channel[channel.value] += 1
        if success:
            self.successes_by_channel[channel.value] += 1
            self.last_success_time = time.time()
            self.last_success_channel = channel.value
        else:
            self.failures_by_channel[channel.value] += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "attempts": self.attempts_by_channel.copy(),
            "successes": self.successes_by_channel.copy(),
            "failures": self.failures_by_channel.copy(),
            "last_success_time": self.last_success_time,
            "last_success_channel": self.last_success_channel,
        }


class WorkDiscoveryManager:
    """Discovers work through multiple channels for resilience.

    Priority order:
    1. Leader work queue - fastest, normal path
    2. Peer discovery - query other peers for their work queues
    3. Autonomous queue - local queue populated by AutonomousQueueLoop
    4. Direct selfplay - generate work directly when all else fails
    """

    def __init__(
        self,
        # Channel 1: Leader
        get_leader_id: Callable[[], str | None],
        claim_from_leader: Callable[[list[str]], Coroutine[Any, Any, dict[str, Any] | None]],
        # Channel 2: Peer discovery
        get_alive_peers: Callable[[], list[str]] | None = None,
        query_peer_work: Callable[[str, list[str]], Coroutine[Any, Any, dict[str, Any] | None]] | None = None,
        # Channel 3: Autonomous queue
        pop_autonomous_work: Callable[[], Coroutine[Any, Any, dict[str, Any] | None]] | None = None,
        # Channel 4: Direct selfplay
        create_direct_selfplay_work: Callable[[list[str]], dict[str, Any] | None] | None = None,
        # Config
        config: WorkDiscoveryConfig | None = None,
    ):
        """Initialize work discovery manager.

        Args:
            get_leader_id: Callback returning current leader node ID
            claim_from_leader: Async callback to claim work from leader
            get_alive_peers: Callback returning list of alive peer IDs
            query_peer_work: Async callback to query a peer for work
            pop_autonomous_work: Async callback to pop from autonomous queue
            create_direct_selfplay_work: Callback to create direct selfplay work item
            config: Discovery configuration
        """
        self.config = config or WorkDiscoveryConfig.from_env()
        self._get_leader_id = get_leader_id
        self._claim_from_leader = claim_from_leader
        self._get_alive_peers = get_alive_peers
        self._query_peer_work = query_peer_work
        self._pop_autonomous_work = pop_autonomous_work
        self._create_direct_selfplay_work = create_direct_selfplay_work

        self._stats = WorkDiscoveryStats()
        self._last_peer_query_time: float = 0.0
        self._last_direct_selfplay_time: float = 0.0

    async def discover_work(self, capabilities: list[str]) -> DiscoveryResult:
        """Discover work through multiple channels in priority order.

        Args:
            capabilities: List of work types this worker can handle

        Returns:
            DiscoveryResult with work item (or None) and discovery metadata
        """
        start_time = time.time()

        # Channel 1: Leader work queue (fastest)
        if self.config.leader_enabled:
            result = await self._try_leader_channel(capabilities)
            if result.work_item:
                return result

        # Channel 2: Peer discovery
        if self.config.peer_discovery_enabled and self._can_query_peers():
            result = await self._try_peer_channel(capabilities)
            if result.work_item:
                return result

        # Channel 3: Autonomous queue
        if self.config.autonomous_enabled:
            result = await self._try_autonomous_channel()
            if result.work_item:
                return result

        # Channel 4: Direct selfplay (last resort)
        if self.config.direct_selfplay_enabled and self._can_direct_selfplay(capabilities):
            result = await self._try_direct_channel(capabilities)
            if result.work_item:
                return result

        # No work found from any channel
        duration = time.time() - start_time
        return DiscoveryResult(
            work_item=None,
            channel=DiscoveryChannel.LEADER,  # Primary attempted channel
            duration_seconds=duration,
            error="No work available from any channel",
        )

    async def _try_leader_channel(self, capabilities: list[str]) -> DiscoveryResult:
        """Try to get work from leader."""
        start_time = time.time()
        channel = DiscoveryChannel.LEADER

        leader_id = self._get_leader_id()
        if not leader_id:
            self._stats.record_attempt(channel, False)
            return DiscoveryResult(
                work_item=None,
                channel=channel,
                duration_seconds=time.time() - start_time,
                error="No leader available",
            )

        try:
            work_item = await self._claim_from_leader(capabilities)
            success = work_item is not None
            self._stats.record_attempt(channel, success)
            return DiscoveryResult(
                work_item=work_item,
                channel=channel,
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            self._stats.record_attempt(channel, False)
            return DiscoveryResult(
                work_item=None,
                channel=channel,
                duration_seconds=time.time() - start_time,
                error=str(e),
            )

    async def _try_peer_channel(self, capabilities: list[str]) -> DiscoveryResult:
        """Try to get work from other peers."""
        start_time = time.time()
        channel = DiscoveryChannel.PEER

        if not self._get_alive_peers or not self._query_peer_work:
            self._stats.record_attempt(channel, False)
            return DiscoveryResult(
                work_item=None,
                channel=channel,
                duration_seconds=time.time() - start_time,
                error="Peer discovery not configured",
            )

        try:
            peers = self._get_alive_peers()
            leader_id = self._get_leader_id()

            # Filter out leader (already tried) and shuffle for load balancing
            peers = [p for p in peers if p != leader_id]
            random.shuffle(peers)
            peers = peers[: self.config.peer_query_limit]

            for peer_id in peers:
                try:
                    work_item = await asyncio.wait_for(
                        self._query_peer_work(peer_id, capabilities),
                        timeout=self.config.peer_query_timeout_seconds,
                    )
                    if work_item:
                        self._stats.record_attempt(channel, True)
                        self._last_peer_query_time = time.time()
                        return DiscoveryResult(
                            work_item=work_item,
                            channel=channel,
                            duration_seconds=time.time() - start_time,
                            peer_id=peer_id,
                        )
                except asyncio.TimeoutError:
                    logger.debug(f"[WorkDiscovery] Peer {peer_id} query timed out")
                except Exception as e:
                    logger.debug(f"[WorkDiscovery] Peer {peer_id} query failed: {e}")

            self._stats.record_attempt(channel, False)
            self._last_peer_query_time = time.time()
            return DiscoveryResult(
                work_item=None,
                channel=channel,
                duration_seconds=time.time() - start_time,
                error=f"No work from {len(peers)} peers",
            )

        except Exception as e:
            self._stats.record_attempt(channel, False)
            return DiscoveryResult(
                work_item=None,
                channel=channel,
                duration_seconds=time.time() - start_time,
                error=str(e),
            )

    async def _try_autonomous_channel(self) -> DiscoveryResult:
        """Try to get work from autonomous queue."""
        start_time = time.time()
        channel = DiscoveryChannel.AUTONOMOUS

        if not self._pop_autonomous_work:
            self._stats.record_attempt(channel, False)
            return DiscoveryResult(
                work_item=None,
                channel=channel,
                duration_seconds=time.time() - start_time,
                error="Autonomous queue not configured",
            )

        try:
            work_item = await self._pop_autonomous_work()
            success = work_item is not None
            self._stats.record_attempt(channel, success)
            return DiscoveryResult(
                work_item=work_item,
                channel=channel,
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            self._stats.record_attempt(channel, False)
            return DiscoveryResult(
                work_item=None,
                channel=channel,
                duration_seconds=time.time() - start_time,
                error=str(e),
            )

    async def _try_direct_channel(self, capabilities: list[str]) -> DiscoveryResult:
        """Create direct selfplay work as last resort."""
        start_time = time.time()
        channel = DiscoveryChannel.DIRECT

        if not self._create_direct_selfplay_work:
            self._stats.record_attempt(channel, False)
            return DiscoveryResult(
                work_item=None,
                channel=channel,
                duration_seconds=time.time() - start_time,
                error="Direct selfplay not configured",
            )

        try:
            work_item = self._create_direct_selfplay_work(capabilities)
            success = work_item is not None
            self._stats.record_attempt(channel, success)
            if success:
                self._last_direct_selfplay_time = time.time()
            return DiscoveryResult(
                work_item=work_item,
                channel=channel,
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            self._stats.record_attempt(channel, False)
            return DiscoveryResult(
                work_item=None,
                channel=channel,
                duration_seconds=time.time() - start_time,
                error=str(e),
            )

    def _can_query_peers(self) -> bool:
        """Check if we can query peers (respecting cooldown)."""
        if self._last_peer_query_time == 0:
            return True
        elapsed = time.time() - self._last_peer_query_time
        return elapsed >= self.config.peer_discovery_cooldown_seconds

    def _can_direct_selfplay(self, capabilities: list[str]) -> bool:
        """Check if we can do direct selfplay."""
        # Must have selfplay capability
        if "selfplay" not in capabilities:
            return False

        # Jan 5, 2026: Check if this node should do selfplay (coordinator check)
        # This prevents coordinator nodes from spawning selfplay even as fallback
        if not _is_selfplay_enabled_for_node():
            return False

        # Respect cooldown
        if self._last_direct_selfplay_time == 0:
            return True
        elapsed = time.time() - self._last_direct_selfplay_time
        return elapsed >= self.config.direct_selfplay_cooldown_seconds

    def get_stats(self) -> dict[str, Any]:
        """Get discovery statistics."""
        return {
            **self._stats.to_dict(),
            "can_query_peers": self._can_query_peers(),
            "can_direct_selfplay": self._can_direct_selfplay(["selfplay"]),
        }

    def health_check(self) -> dict[str, Any]:
        """Return health status for DaemonManager integration.

        Returns a dict with:
        - healthy: True if discovery is functional
        - status: "active" if channels are configured, "limited" if some missing
        - channels_enabled: Which discovery channels are active
        - stats: Discovery statistics by channel
        - last_success: Time since last successful discovery
        """
        # Calculate total attempts and successes
        total_attempts = sum(self._stats.attempts_by_channel.values())
        total_successes = sum(self._stats.successes_by_channel.values())

        # Determine health based on success rate (if we've had attempts)
        if total_attempts == 0:
            # No attempts yet - healthy but idle
            healthy = True
            status = "idle"
        elif total_successes > 0:
            # At least one success - healthy
            success_rate = total_successes / total_attempts
            healthy = True
            status = "active" if success_rate >= 0.1 else "degraded"
        else:
            # All attempts failed
            healthy = False
            status = "failing"

        # Check which channels are properly configured
        channels_enabled = {
            "leader": self.config.leader_enabled,
            "peer": self.config.peer_discovery_enabled and self._get_alive_peers is not None,
            "autonomous": self.config.autonomous_enabled and self._pop_autonomous_work is not None,
            "direct": self.config.direct_selfplay_enabled and self._create_direct_selfplay_work is not None,
        }

        # If no channels enabled, mark as degraded
        if not any(channels_enabled.values()):
            healthy = False
            status = "no_channels"

        # Calculate time since last success
        last_success_ago = None
        if self._stats.last_success_time:
            last_success_ago = time.time() - self._stats.last_success_time

        total_failures = sum(self._stats.failures_by_channel.values())

        return {
            "healthy": healthy,
            "status": status,
            "channels_enabled": channels_enabled,
            "stats": {
                "attempts": self._stats.attempts_by_channel.copy(),
                "successes": self._stats.successes_by_channel.copy(),
                "failures": self._stats.failures_by_channel.copy(),
                "total_attempts": total_attempts,
                "total_successes": total_successes,
                "total_failures": total_failures,
                "success_rate": total_successes / total_attempts if total_attempts > 0 else None,
            },
            "last_success_channel": self._stats.last_success_channel,
            "last_success_seconds_ago": last_success_ago,
            "cooldowns": {
                "can_query_peers": self._can_query_peers(),
                "can_direct_selfplay": self._can_direct_selfplay(["selfplay"]),
            },
        }


# Singleton instance
_work_discovery_manager: WorkDiscoveryManager | None = None


def get_work_discovery_manager() -> WorkDiscoveryManager | None:
    """Get singleton work discovery manager instance."""
    return _work_discovery_manager


def set_work_discovery_manager(manager: WorkDiscoveryManager) -> None:
    """Set the singleton work discovery manager."""
    global _work_discovery_manager
    _work_discovery_manager = manager


def reset_work_discovery_manager() -> None:
    """Reset the singleton (for testing)."""
    global _work_discovery_manager
    _work_discovery_manager = None


__all__ = [
    "DiscoveryChannel",
    "DiscoveryResult",
    "WorkDiscoveryConfig",
    "WorkDiscoveryManager",
    "WorkDiscoveryStats",
    "get_work_discovery_manager",
    "reset_work_discovery_manager",
    "set_work_discovery_manager",
]
