"""DistributedLeadershipCoordinator - Multi-daemon coordination (December 2025).

This module provides centralized coordination for distributed leadership
across multiple daemon instances. It tracks leader elections, handles
failover, and coordinates work distribution.

Event Integration:
- Subscribes to LEADER_ELECTED: Track new leader elections
- Subscribes to LEADER_LOST: Track leader loss/disconnection
- Subscribes to LEADER_STEPDOWN: Track voluntary leader stepdown
- Subscribes to HOST_ONLINE: Track node joining
- Subscribes to HOST_OFFLINE: Track node leaving

Key Responsibilities:
1. Track current leader for each resource/domain
2. Handle leader failover gracefully
3. Coordinate work distribution among followers
4. Prevent split-brain scenarios
5. Provide leadership status for decisions

Usage:
    from app.coordination.leadership_coordinator import (
        LeadershipCoordinator,
        wire_leadership_events,
        get_leadership_coordinator,
    )

    # Wire leadership events
    coordinator = wire_leadership_events()

    # Check if we're the leader
    if coordinator.is_leader("training"):
        # We're responsible for triggering training
        trigger_training()

    # Get current leader
    leader = coordinator.get_leader("training")
    print(f"Training leader: {leader}")
"""

from __future__ import annotations

import logging
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

logger = logging.getLogger(__name__)


class LeadershipDomain(Enum):
    """Domains that have leaders."""

    TRAINING = "training"  # Training job coordination
    EVALUATION = "evaluation"  # Evaluation coordination
    PROMOTION = "promotion"  # Model promotion decisions
    DATA_SYNC = "data_sync"  # Data synchronization
    CLUSTER = "cluster"  # Overall cluster coordination


class LeadershipRole(Enum):
    """Role of a node in leader election (Raft-like).

    NOTE (Dec 2025): Renamed from NodeRole to avoid collision with
    ClusterNodeRole in multi_provider_orchestrator.py which has different semantics.
    """

    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    OFFLINE = "offline"


# Backward-compat alias (deprecated)
NodeRole = LeadershipRole


@dataclass
class NodeInfo:
    """Information about a cluster node."""

    node_id: str
    hostname: str = ""
    ip_address: str = ""
    role: NodeRole = NodeRole.FOLLOWER
    last_heartbeat: float = field(default_factory=time.time)
    leader_domains: set[str] = field(default_factory=set)
    capabilities: set[str] = field(default_factory=set)
    joined_at: float = field(default_factory=time.time)

    @property
    def is_alive(self) -> bool:
        """Check if node is considered alive."""
        return time.time() - self.last_heartbeat < 30.0


@dataclass
class LeadershipRecord:
    """Record of leadership for a domain."""

    domain: str
    leader_node_id: str
    elected_at: float = field(default_factory=time.time)
    term: int = 1
    previous_leader: str | None = None
    election_reason: str = ""


@dataclass
class LeadershipEvent:
    """Record of a leadership event."""

    event_type: str  # elected, lost, stepdown
    domain: str
    node_id: str
    term: int
    timestamp: float = field(default_factory=time.time)
    reason: str = ""


@dataclass
class LeadershipStats:
    """Aggregate leadership statistics."""

    total_nodes: int = 0
    online_nodes: int = 0
    leaders_by_domain: dict[str, str] = field(default_factory=dict)
    total_elections: int = 0
    total_failovers: int = 0
    current_term: int = 0
    cluster_healthy: bool = True


class LeadershipCoordinator:
    """Coordinates distributed leadership across the cluster.

    Tracks leader elections, handles failover, and provides leadership
    status for distributed decision making.
    """

    def __init__(
        self,
        local_node_id: str | None = None,
        heartbeat_timeout: float = 30.0,
        election_timeout: float = 10.0,
        max_event_history: int = 200,
    ):
        """Initialize LeadershipCoordinator.

        Args:
            local_node_id: ID of the local node (default: hostname)
            heartbeat_timeout: Seconds without heartbeat to mark offline
            election_timeout: Seconds to wait for election
            max_event_history: Maximum events to retain
        """
        self.local_node_id = local_node_id or socket.gethostname()
        self.heartbeat_timeout = heartbeat_timeout
        self.election_timeout = election_timeout
        self.max_event_history = max_event_history

        # Node tracking
        self._nodes: dict[str, NodeInfo] = {}
        self._local_node = NodeInfo(
            node_id=self.local_node_id,
            hostname=socket.gethostname(),
        )
        self._nodes[self.local_node_id] = self._local_node

        # Leadership tracking by domain
        self._leadership: dict[str, LeadershipRecord] = {}
        self._current_term = 0

        # Event history
        self._events: list[LeadershipEvent] = []

        # Statistics
        self._total_elections = 0
        self._total_failovers = 0

        # Callbacks
        self._leader_change_callbacks: list[Callable[[str, str, str], None]] = []  # domain, old, new

        # Subscription state
        self._subscribed = False

    def subscribe_to_events(self) -> bool:
        """Subscribe to leadership-related events.

        Returns:
            True if successfully subscribed
        """
        if self._subscribed:
            return True

        try:
            from app.coordination.event_router import get_router
            from app.coordination.event_router import DataEventType

            router = get_router()

            router.subscribe(DataEventType.LEADER_ELECTED.value, self._on_leader_elected)
            router.subscribe(DataEventType.LEADER_LOST.value, self._on_leader_lost)
            router.subscribe(DataEventType.LEADER_STEPDOWN.value, self._on_leader_stepdown)
            router.subscribe(DataEventType.HOST_ONLINE.value, self._on_host_online)
            router.subscribe(DataEventType.HOST_OFFLINE.value, self._on_host_offline)

            self._subscribed = True
            logger.info("[LeadershipCoordinator] Subscribed to leadership events")
            return True

        except ImportError:
            logger.warning("[LeadershipCoordinator] data_events not available")
            return False
        except (AttributeError, TypeError) as e:
            # December 29, 2025: Narrowed from bare except Exception
            # - AttributeError: Router or subscribe method not available
            # - TypeError: Wrong argument types for subscribe
            logger.error(f"[LeadershipCoordinator] Failed to subscribe: {e}")
            return False

    def _record_event(
        self, event_type: str, domain: str, node_id: str, reason: str = ""
    ) -> None:
        """Record a leadership event."""
        event = LeadershipEvent(
            event_type=event_type,
            domain=domain,
            node_id=node_id,
            term=self._current_term,
            reason=reason,
        )
        self._events.append(event)

        # Trim history
        if len(self._events) > self.max_event_history:
            self._events = self._events[-self.max_event_history:]

    async def _on_leader_elected(self, event) -> None:
        """Handle LEADER_ELECTED event."""
        # Dec 29, 2025: Robust payload extraction (handles RouterEvent, dict, etc.)
        payload = event.payload if hasattr(event, "payload") else event
        if not isinstance(payload, dict):
            payload = {}
        domain = payload.get("domain", "cluster")
        # Dec 29, 2025: Handle both "node_id" and "leader_id" keys (emit uses "leader_id")
        node_id = payload.get("node_id") or payload.get("leader_id", "")
        term = payload.get("term", self._current_term + 1)

        old_leader = None
        if domain in self._leadership:
            old_leader = self._leadership[domain].leader_node_id

        # Update leadership
        self._leadership[domain] = LeadershipRecord(
            domain=domain,
            leader_node_id=node_id,
            term=term,
            previous_leader=old_leader,
            election_reason=payload.get("reason", "elected"),
        )

        # Update term
        self._current_term = max(self._current_term, term)
        self._total_elections += 1

        # Update node info
        if node_id in self._nodes:
            self._nodes[node_id].role = NodeRole.LEADER
            self._nodes[node_id].leader_domains.add(domain)
        if old_leader and old_leader in self._nodes:
            self._nodes[old_leader].role = NodeRole.FOLLOWER
            self._nodes[old_leader].leader_domains.discard(domain)

        # Record event
        self._record_event("elected", domain, node_id, payload.get("reason", ""))

        # Notify callbacks
        for callback in self._leader_change_callbacks:
            try:
                callback(domain, old_leader or "", node_id)
            except Exception as e:
                logger.error(f"[LeadershipCoordinator] Leader change callback error: {e}")

        logger.info(
            f"[LeadershipCoordinator] Leader elected for {domain}: {node_id} "
            f"(term {term})"
        )

    async def _on_leader_lost(self, event) -> None:
        """Handle LEADER_LOST event."""
        # Dec 29, 2025: Robust payload extraction (handles RouterEvent, dict, etc.)
        payload = event.payload if hasattr(event, "payload") else event
        if not isinstance(payload, dict):
            payload = {}
        domain = payload.get("domain", "cluster")
        # Dec 29, 2025: Handle both "node_id" and "old_leader_id" keys (emit uses "old_leader_id")
        node_id = payload.get("node_id") or payload.get("old_leader_id", "")

        if domain in self._leadership and self._leadership[domain].leader_node_id == node_id:
            old_leader = self._leadership[domain].leader_node_id
            del self._leadership[domain]
            self._total_failovers += 1

            # Update node info
            if node_id in self._nodes:
                self._nodes[node_id].leader_domains.discard(domain)

            # Record event
            self._record_event("lost", domain, node_id, payload.get("reason", ""))

            # Notify callbacks
            for callback in self._leader_change_callbacks:
                try:
                    callback(domain, old_leader, "")
                except Exception as e:
                    logger.error(f"[LeadershipCoordinator] Callback error: {e}")

            logger.warning(
                f"[LeadershipCoordinator] Leader lost for {domain}: {node_id}"
            )

    async def _on_leader_stepdown(self, event) -> None:
        """Handle LEADER_STEPDOWN event."""
        # Dec 29, 2025: Robust payload extraction (handles RouterEvent, dict, etc.)
        payload = event.payload if hasattr(event, "payload") else event
        if not isinstance(payload, dict):
            payload = {}
        domain = payload.get("domain", "cluster")
        # Dec 29, 2025: Handle multiple key variations for node ID
        node_id = payload.get("node_id") or payload.get("leader_id", "")

        if domain in self._leadership and self._leadership[domain].leader_node_id == node_id:
            old_leader = self._leadership[domain].leader_node_id
            del self._leadership[domain]

            # Update node info
            if node_id in self._nodes:
                self._nodes[node_id].role = NodeRole.FOLLOWER
                self._nodes[node_id].leader_domains.discard(domain)

            # Record event
            self._record_event("stepdown", domain, node_id, payload.get("reason", ""))

            # Notify callbacks
            for callback in self._leader_change_callbacks:
                try:
                    callback(domain, old_leader, "")
                except Exception as e:
                    logger.error(f"[LeadershipCoordinator] Callback error: {e}")

            logger.info(
                f"[LeadershipCoordinator] Leader stepped down for {domain}: {node_id}"
            )

    async def _on_host_online(self, event) -> None:
        """Handle HOST_ONLINE event."""
        # Dec 29, 2025: Robust payload extraction (handles RouterEvent, dict, etc.)
        payload = event.payload if hasattr(event, "payload") else event
        if not isinstance(payload, dict):
            payload = {}
        # Dec 29, 2025: Handle both "node_id" and "host" keys (emit uses "host")
        node_id = payload.get("node_id") or payload.get("host", "")
        hostname = payload.get("hostname", "")

        if node_id not in self._nodes:
            self._nodes[node_id] = NodeInfo(
                node_id=node_id,
                hostname=hostname,
                ip_address=payload.get("ip_address", ""),
                capabilities=set(payload.get("capabilities", [])),
            )
        else:
            self._nodes[node_id].last_heartbeat = time.time()
            self._nodes[node_id].role = NodeRole.FOLLOWER

        logger.info(f"[LeadershipCoordinator] Node online: {node_id}")

    async def _on_host_offline(self, event) -> None:
        """Handle HOST_OFFLINE event."""
        # Dec 29, 2025: Robust payload extraction (handles RouterEvent, dict, etc.)
        payload = event.payload if hasattr(event, "payload") else event
        if not isinstance(payload, dict):
            payload = {}
        # Dec 29, 2025: Handle both "node_id" and "host" keys (emit uses "host")
        node_id = payload.get("node_id") or payload.get("host", "")

        if node_id in self._nodes:
            self._nodes[node_id].role = NodeRole.OFFLINE

            # Check if this node was a leader
            for domain, record in list(self._leadership.items()):
                if record.leader_node_id == node_id:
                    del self._leadership[domain]
                    self._total_failovers += 1
                    self._record_event("lost", domain, node_id, "host offline")

                    logger.warning(
                        f"[LeadershipCoordinator] Leader offline for {domain}: {node_id}"
                    )

        logger.info(f"[LeadershipCoordinator] Node offline: {node_id}")

    def is_leader(self, domain: str = "cluster") -> bool:
        """Check if the local node is the leader for a domain.

        Args:
            domain: The leadership domain to check

        Returns:
            True if local node is the leader
        """
        if domain not in self._leadership:
            return False
        return self._leadership[domain].leader_node_id == self.local_node_id

    def get_leader(self, domain: str = "cluster") -> str | None:
        """Get the current leader for a domain.

        Args:
            domain: The leadership domain

        Returns:
            Leader node ID, or None if no leader
        """
        if domain not in self._leadership:
            return None
        return self._leadership[domain].leader_node_id

    def has_leader(self, domain: str = "cluster") -> bool:
        """Check if a domain has a leader.

        Args:
            domain: The leadership domain

        Returns:
            True if domain has a leader
        """
        return domain in self._leadership

    def claim_leadership(self, domain: str, reason: str = "manual") -> bool:
        """Attempt to claim leadership for a domain.

        This is a local operation. In a real distributed system,
        this would need consensus.

        Returns:
            True if leadership was claimed
        """
        if domain in self._leadership:
            return False

        self._current_term += 1
        self._leadership[domain] = LeadershipRecord(
            domain=domain,
            leader_node_id=self.local_node_id,
            term=self._current_term,
            election_reason=reason,
        )

        self._local_node.role = NodeRole.LEADER
        self._local_node.leader_domains.add(domain)
        self._total_elections += 1

        self._record_event("elected", domain, self.local_node_id, reason)

        logger.info(f"[LeadershipCoordinator] Claimed leadership for {domain}")
        return True

    def release_leadership(self, domain: str, reason: str = "voluntary") -> bool:
        """Release leadership for a domain.

        Returns:
            True if leadership was released
        """
        if not self.is_leader(domain):
            return False

        del self._leadership[domain]
        self._local_node.leader_domains.discard(domain)
        if not self._local_node.leader_domains:
            self._local_node.role = NodeRole.FOLLOWER

        self._record_event("stepdown", domain, self.local_node_id, reason)

        logger.info(f"[LeadershipCoordinator] Released leadership for {domain}")
        return True

    def on_leader_change(self, callback: Callable[[str, str, str], None]) -> None:
        """Register callback for leader changes.

        Args:
            callback: Function(domain, old_leader, new_leader)
        """
        self._leader_change_callbacks.append(callback)

    def get_node(self, node_id: str) -> NodeInfo | None:
        """Get info for a specific node."""
        return self._nodes.get(node_id)

    def get_all_nodes(self) -> list[NodeInfo]:
        """Get all known nodes."""
        return list(self._nodes.values())

    def get_online_nodes(self) -> list[NodeInfo]:
        """Get all online nodes."""
        return [n for n in self._nodes.values() if n.is_alive and n.role != NodeRole.OFFLINE]

    def get_leaders(self) -> dict[str, str]:
        """Get all current leaders by domain."""
        return {domain: record.leader_node_id for domain, record in self._leadership.items()}

    def get_event_history(self, limit: int = 50) -> list[LeadershipEvent]:
        """Get recent leadership events."""
        return self._events[-limit:]

    def get_stats(self) -> LeadershipStats:
        """Get aggregate leadership statistics."""
        online = [n for n in self._nodes.values() if n.is_alive]
        leaders = {d: r.leader_node_id for d, r in self._leadership.items()}

        # Check cluster health
        required_domains = {"cluster", "training"}
        has_leaders = all(d in self._leadership for d in required_domains)
        healthy = len(online) >= 1 and has_leaders

        return LeadershipStats(
            total_nodes=len(self._nodes),
            online_nodes=len(online),
            leaders_by_domain=leaders,
            total_elections=self._total_elections,
            total_failovers=self._total_failovers,
            current_term=self._current_term,
            cluster_healthy=healthy,
        )

    def get_status(self) -> dict[str, Any]:
        """Get coordinator status for monitoring."""
        stats = self.get_stats()

        return {
            "local_node_id": self.local_node_id,
            "local_role": self._local_node.role.value,
            "local_leader_domains": list(self._local_node.leader_domains),
            "total_nodes": stats.total_nodes,
            "online_nodes": stats.online_nodes,
            "leaders": stats.leaders_by_domain,
            "current_term": stats.current_term,
            "total_elections": stats.total_elections,
            "total_failovers": stats.total_failovers,
            "cluster_healthy": stats.cluster_healthy,
            "subscribed": self._subscribed,
        }

    def health_check(self) -> HealthCheckResult:
        """Check coordinator health for DaemonManager integration.

        Returns:
            HealthCheckResult with status and details

        December 2025 Session 2: Added exception handling.
        """
        try:
            stats = self.get_stats()

            details = {
                "local_node_id": self.local_node_id,
                "subscribed": self._subscribed,
                "online_nodes": stats.online_nodes,
                "total_nodes": stats.total_nodes,
                "leaders_by_domain": stats.leaders_by_domain,
                "total_elections": stats.total_elections,
                "total_failovers": stats.total_failovers,
                "current_term": stats.current_term,
            }

            # Not subscribed to events = degraded
            if not self._subscribed:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.DEGRADED,
                    message="LeadershipCoordinator not subscribed to events",
                    details=details,
                )

            # No online nodes = error
            if stats.online_nodes == 0:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.ERROR,
                    message="No online nodes in cluster",
                    details=details,
                )

            # Cluster not healthy (missing required leaders)
            if not stats.cluster_healthy:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.DEGRADED,
                    message="Cluster missing required leaders",
                    details=details,
                )

            # All good
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message=f"LeadershipCoordinator healthy: {stats.online_nodes} nodes, {len(stats.leaders_by_domain)} domains",
                details=details,
            )
        except (AttributeError, TypeError, KeyError) as e:
            # December 29, 2025: Narrowed from bare except Exception
            # - AttributeError: Stats method or attributes not available
            # - TypeError: Wrong types passed to HealthCheckResult
            # - KeyError: Missing dict keys in stats
            logger.warning(f"[LeadershipCoordinator] health_check error: {e}")
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Health check error: {e}",
                details={"error": str(e)},
            )


# =============================================================================
# Singleton and convenience functions
# =============================================================================

_leadership_coordinator: LeadershipCoordinator | None = None


def get_leadership_coordinator() -> LeadershipCoordinator:
    """Get the global LeadershipCoordinator singleton."""
    global _leadership_coordinator
    if _leadership_coordinator is None:
        _leadership_coordinator = LeadershipCoordinator()
    return _leadership_coordinator


def wire_leadership_events() -> LeadershipCoordinator:
    """Wire leadership events to the coordinator.

    Returns:
        The wired LeadershipCoordinator instance
    """
    coordinator = get_leadership_coordinator()
    coordinator.subscribe_to_events()
    return coordinator


def is_leader(domain: str = "cluster") -> bool:
    """Convenience function to check if local node is leader."""
    return get_leadership_coordinator().is_leader(domain)


def get_current_leader(domain: str = "cluster") -> str | None:
    """Convenience function to get current leader."""
    return get_leadership_coordinator().get_leader(domain)


__all__ = [
    "LeadershipCoordinator",
    "LeadershipDomain",
    "LeadershipEvent",
    "LeadershipRecord",
    "LeadershipStats",
    "NodeInfo",
    "NodeRole",
    "get_current_leader",
    "get_leadership_coordinator",
    "is_leader",
    "wire_leadership_events",
]
