"""Unified Timeout Profile for P2P Operations.

Jan 29, 2026: Created as single source of truth for all P2P timeouts.
Eliminates timeout mismatches that caused split-brain (18s disagreement window).

Design principles:
1. All timeouts derived from two base values (HEARTBEAT_INTERVAL, PEER_TIMEOUT)
2. Relationships are enforced mathematically (e.g., SUSPECT = PEER_TIMEOUT * 0.5)
3. Environment variables tune base values; derived values auto-adjust
4. Deterministic jitter eliminates node disagreement

Usage:
    from app.config.timeout_profile import get_timeout_profile, get_jittered_timeout

    profile = get_timeout_profile()
    timeout = profile.PEER_TIMEOUT  # 180s (base)
    suspect = profile.SUSPECT_TIMEOUT  # 90s (derived: PEER_TIMEOUT * 0.5)

    # Deterministic jitter based on node_id
    jittered = get_jittered_timeout(profile.PEER_TIMEOUT, "node-a")  # Same value every call
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class TimeoutProfile:
    """Single source of truth for all P2P timeouts.

    All timeouts are derived from two base values:
    - HEARTBEAT_INTERVAL: How often nodes send heartbeats
    - PEER_TIMEOUT: How long before a peer is considered dead

    This ensures all timeout relationships stay in sync when tuning.
    """

    # ============================================
    # Base Values (tune these only)
    # ============================================

    HEARTBEAT_INTERVAL: float = 15.0
    """Seconds between heartbeat probes. Lower = faster detection, higher = less traffic."""

    PEER_TIMEOUT: float = 180.0
    """Seconds before marking peer as DEAD. Should be >= HEARTBEAT_INTERVAL * 10."""

    # ============================================
    # Jitter Configuration
    # ============================================

    JITTER_FACTOR: float = 0.05
    """Jitter percentage (±5%). Deterministic per node_id, not random."""

    # ============================================
    # Derived Values (computed properties)
    # ============================================

    @property
    def SUSPECT_TIMEOUT(self) -> float:
        """Seconds before marking peer as SUSPECT (half of PEER_TIMEOUT)."""
        return self.PEER_TIMEOUT * 0.5  # 90s for 180s PEER_TIMEOUT

    @property
    def VOTER_HEARTBEAT_TIMEOUT(self) -> float:
        """Seconds before marking voter unhealthy (4 missed heartbeats)."""
        return self.HEARTBEAT_INTERVAL * 4  # 60s for 15s HEARTBEAT_INTERVAL

    @property
    def LEADER_LEASE_DURATION(self) -> float:
        """Seconds leader lease is valid (1.67x PEER_TIMEOUT)."""
        return self.PEER_TIMEOUT * 1.67  # 300s for 180s PEER_TIMEOUT

    @property
    def LEADER_PROBE_THRESHOLD(self) -> float:
        """Seconds of failures before triggering election (0.56x PEER_TIMEOUT)."""
        return self.PEER_TIMEOUT * 0.56  # 100s for 180s PEER_TIMEOUT

    @property
    def PEER_RETIRE_AFTER(self) -> float:
        """Seconds before retiring dead peer (PEER_TIMEOUT + 60s gap)."""
        return self.PEER_TIMEOUT + 60  # 240s for 180s PEER_TIMEOUT

    @property
    def GOSSIP_INTERVAL(self) -> float:
        """Seconds between gossip rounds (0.67x HEARTBEAT_INTERVAL)."""
        return self.HEARTBEAT_INTERVAL * 0.67  # 10s for 15s HEARTBEAT_INTERVAL

    @property
    def HEALTH_CHECK_TIMEOUT(self) -> float:
        """Seconds for health check probe (0.53x HEARTBEAT_INTERVAL)."""
        return self.HEARTBEAT_INTERVAL * 0.53  # 8s for 15s HEARTBEAT_INTERVAL

    @property
    def ELECTION_TIMEOUT(self) -> float:
        """Seconds for election request timeout."""
        return self.HEARTBEAT_INTERVAL  # 15s, matches heartbeat

    @property
    def MAX_CONSECUTIVE_FAILURES(self) -> int:
        """Number of failures before marking node dead."""
        return int(self.PEER_TIMEOUT / self.HEARTBEAT_INTERVAL)  # 12 for 180s/15s

    # ============================================
    # SWIM Protocol Alignment
    # ============================================

    @property
    def SWIM_FAILURE_TIMEOUT(self) -> float:
        """SWIM failure detection (0.75x PEER_TIMEOUT for early warning)."""
        return self.PEER_TIMEOUT * 0.75  # 135s for 180s PEER_TIMEOUT

    @property
    def SWIM_SUSPICION_TIMEOUT(self) -> float:
        """SWIM suspicion period (0.5x PEER_TIMEOUT)."""
        return self.PEER_TIMEOUT * 0.5  # 90s for 180s PEER_TIMEOUT

    # ============================================
    # HTTP Configuration
    # ============================================

    @property
    def HTTP_TOTAL_TIMEOUT(self) -> float:
        """HTTP request total timeout (0.5x PEER_TIMEOUT for retries)."""
        return self.PEER_TIMEOUT * 0.5  # 90s for 180s PEER_TIMEOUT

    @property
    def HTTP_CONNECT_TIMEOUT(self) -> float:
        """HTTP connection timeout (2x HEARTBEAT_INTERVAL)."""
        return self.HEARTBEAT_INTERVAL * 2  # 30s for 15s HEARTBEAT_INTERVAL

    # ============================================
    # Validation
    # ============================================

    def __post_init__(self) -> None:
        """Validate timeout relationships."""
        # Ensure PEER_TIMEOUT is at least 10 heartbeats
        min_peer_timeout = self.HEARTBEAT_INTERVAL * 10
        if self.PEER_TIMEOUT < min_peer_timeout:
            raise ValueError(
                f"PEER_TIMEOUT ({self.PEER_TIMEOUT}s) must be >= "
                f"HEARTBEAT_INTERVAL * 10 ({min_peer_timeout}s)"
            )

    def validate_relationships(self) -> list[str]:
        """Return list of any timeout relationship warnings."""
        warnings = []

        if self.SUSPECT_TIMEOUT >= self.PEER_TIMEOUT:
            warnings.append(
                f"SUSPECT_TIMEOUT ({self.SUSPECT_TIMEOUT}s) should be < "
                f"PEER_TIMEOUT ({self.PEER_TIMEOUT}s)"
            )

        if self.LEADER_PROBE_THRESHOLD >= self.LEADER_LEASE_DURATION:
            warnings.append(
                f"LEADER_PROBE_THRESHOLD ({self.LEADER_PROBE_THRESHOLD}s) should be < "
                f"LEADER_LEASE_DURATION ({self.LEADER_LEASE_DURATION}s)"
            )

        if self.HEALTH_CHECK_TIMEOUT >= self.HEARTBEAT_INTERVAL:
            warnings.append(
                f"HEALTH_CHECK_TIMEOUT ({self.HEALTH_CHECK_TIMEOUT}s) should be < "
                f"HEARTBEAT_INTERVAL ({self.HEARTBEAT_INTERVAL}s)"
            )

        return warnings


# ============================================
# Singleton Instance
# ============================================

_profile_instance: TimeoutProfile | None = None


def get_timeout_profile() -> TimeoutProfile:
    """Get the singleton TimeoutProfile instance.

    Base values are read from environment variables on first call:
    - RINGRIFT_P2P_HEARTBEAT_INTERVAL (default: 15)
    - RINGRIFT_P2P_PEER_TIMEOUT (default: 180)
    - RINGRIFT_P2P_PEER_TIMEOUT_JITTER (default: 0.05)
    """
    global _profile_instance

    if _profile_instance is None:
        heartbeat = float(os.environ.get("RINGRIFT_P2P_HEARTBEAT_INTERVAL", "15") or 15)
        peer_timeout = float(os.environ.get("RINGRIFT_P2P_PEER_TIMEOUT", "180") or 180)
        jitter = float(os.environ.get("RINGRIFT_P2P_PEER_TIMEOUT_JITTER", "0.05") or 0.05)

        _profile_instance = TimeoutProfile(
            HEARTBEAT_INTERVAL=heartbeat,
            PEER_TIMEOUT=peer_timeout,
            JITTER_FACTOR=jitter,
        )

    return _profile_instance


def reset_timeout_profile() -> None:
    """Reset the singleton instance (for testing)."""
    global _profile_instance
    _profile_instance = None


# ============================================
# Deterministic Jitter
# ============================================


def get_jittered_timeout(base_timeout: float, node_id: str, jitter_factor: float | None = None) -> float:
    """Get timeout with deterministic jitter based on node_id.

    Unlike random jitter, this returns the SAME value for a given (timeout, node_id) pair.
    All nodes in the cluster will agree on the jittered timeout for any peer.

    Args:
        base_timeout: Base timeout value in seconds
        node_id: Node identifier (e.g., "hetzner-cpu1")
        jitter_factor: Jitter factor ±percentage (default: from profile)

    Returns:
        Timeout with deterministic jitter applied
    """
    if jitter_factor is None:
        jitter_factor = get_timeout_profile().JITTER_FACTOR

    # Hash the node_id to get a deterministic value 0-1
    hash_bytes = hashlib.md5(node_id.encode()).digest()
    hash_int = int.from_bytes(hash_bytes[:4], "big")
    normalized = hash_int / 0xFFFFFFFF  # 0.0 to 1.0

    # Map to jitter range: -jitter_factor to +jitter_factor
    jitter_range = base_timeout * jitter_factor * 2
    offset = (normalized * jitter_range) - (jitter_range / 2)

    return base_timeout + offset


def get_jittered_peer_timeout(node_id: str) -> float:
    """Convenience function for peer timeout with deterministic jitter."""
    profile = get_timeout_profile()
    return get_jittered_timeout(profile.PEER_TIMEOUT, node_id, profile.JITTER_FACTOR)


# ============================================
# Convenience Exports
# ============================================

# Export computed values for direct import (uses singleton)
def _get_heartbeat_interval() -> float:
    return get_timeout_profile().HEARTBEAT_INTERVAL


def _get_peer_timeout() -> float:
    return get_timeout_profile().PEER_TIMEOUT


def _get_suspect_timeout() -> float:
    return get_timeout_profile().SUSPECT_TIMEOUT


def _get_voter_heartbeat_timeout() -> float:
    return get_timeout_profile().VOTER_HEARTBEAT_TIMEOUT


def _get_leader_lease_duration() -> float:
    return get_timeout_profile().LEADER_LEASE_DURATION


def _get_peer_retire_after() -> float:
    return get_timeout_profile().PEER_RETIRE_AFTER


def _get_gossip_interval() -> float:
    return get_timeout_profile().GOSSIP_INTERVAL
