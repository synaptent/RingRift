"""Voter Configuration Types for P2P Cluster Stability.

Part of Phase 2: Voter Configuration Management
P2P Cluster Stability Plan - Jan 13, 2026

This module provides versioned voter configuration with integrity verification,
enabling strict quorum enforcement and preventing split-brain elections.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QuorumResult(Enum):
    """Result of a quorum check.

    Used to provide detailed feedback on why an election might be blocked.
    """

    OK = "ok"  # Quorum achieved, election can proceed
    LOST = "lost"  # Had quorum but lost it (voters went offline)
    CONFIG_MISSING = "config_missing"  # No voter configuration loaded
    CONFIG_INVALID = "config_invalid"  # Config integrity check failed
    INSUFFICIENT_VOTERS = "insufficient"  # Not enough voters configured (< 3)
    PARTITION_DETECTED = "partition"  # Network partition suspected


@dataclass
class VoterConfigVersion:
    """Versioned voter configuration with integrity verification.

    This structure ensures all nodes agree on the voter list and quorum size.
    The SHA256 hash allows nodes to detect configuration drift via gossip.

    Attributes:
        version: Monotonically increasing version number
        voters: Sorted list of voter node IDs
        quorum_size: Minimum voters needed for quorum (floor(n/2) + 1)
        sha256_hash: Integrity hash for drift detection
        created_at: Unix timestamp when config was created
        created_by: Node ID that created this version
    """

    version: int
    voters: list[str]
    quorum_size: int
    sha256_hash: str
    created_at: float = field(default_factory=time.time)
    created_by: str = "unknown"

    def __post_init__(self):
        """Ensure voters list is sorted for consistent hashing."""
        self.voters = sorted(self.voters)

    @classmethod
    def create(
        cls,
        voters: list[str],
        version: int = 1,
        created_by: str = "unknown",
    ) -> VoterConfigVersion:
        """Create a new voter config version with computed hash.

        Args:
            voters: List of voter node IDs
            version: Version number (default 1 for new configs)
            created_by: Node ID creating this config

        Returns:
            VoterConfigVersion with computed hash and quorum size
        """
        sorted_voters = sorted(voters)
        quorum_size = len(sorted_voters) // 2 + 1

        # Compute integrity hash
        hash_input = f"{version}:{quorum_size}:{','.join(sorted_voters)}"
        sha256_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        return cls(
            version=version,
            voters=sorted_voters,
            quorum_size=quorum_size,
            sha256_hash=sha256_hash,
            created_at=time.time(),
            created_by=created_by,
        )

    def verify_integrity(self) -> bool:
        """Verify the configuration integrity using SHA256 hash.

        Returns:
            True if hash matches computed value, False otherwise
        """
        hash_input = f"{self.version}:{self.quorum_size}:{','.join(sorted(self.voters))}"
        expected = hashlib.sha256(hash_input.encode()).hexdigest()
        return self.sha256_hash == expected

    def has_minimum_voters(self, minimum: int = 3) -> bool:
        """Check if configuration has minimum required voters.

        Args:
            minimum: Minimum voters required (default 3 for fault tolerance)

        Returns:
            True if len(voters) >= minimum
        """
        return len(self.voters) >= minimum

    def is_voter(self, node_id: str) -> bool:
        """Check if a node ID is in the voter list.

        Args:
            node_id: Node ID to check

        Returns:
            True if node_id is a voter
        """
        return node_id in self.voters

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "voters": self.voters,
            "quorum_size": self.quorum_size,
            "sha256_hash": self.sha256_hash,
            "hash_short": self.sha256_hash[:16],  # For display
            "created_at": self.created_at,
            "created_by": self.created_by,
            "voter_count": len(self.voters),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VoterConfigVersion:
        """Create from dictionary (deserialization)."""
        return cls(
            version=data["version"],
            voters=data["voters"],
            quorum_size=data["quorum_size"],
            sha256_hash=data["sha256_hash"],
            created_at=data.get("created_at", time.time()),
            created_by=data.get("created_by", "unknown"),
        )

    def __eq__(self, other: object) -> bool:
        """Configs are equal if their hashes match."""
        if not isinstance(other, VoterConfigVersion):
            return False
        return self.sha256_hash == other.sha256_hash

    def __hash__(self) -> int:
        """Hash based on SHA256 for use in sets/dicts."""
        return hash(self.sha256_hash)


@dataclass
class QuorumHealth:
    """Detailed quorum health status for monitoring.

    Provides rich diagnostics for debugging quorum issues.
    """

    result: QuorumResult
    reason: str
    voters_total: int = 0
    voters_alive: int = 0
    voters_needed: int = 0
    config_version: int = 0
    config_hash: str = ""
    alive_voter_ids: list[str] = field(default_factory=list)
    dead_voter_ids: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def is_healthy(self) -> bool:
        """Quick check if quorum is healthy."""
        return self.result == QuorumResult.OK

    @property
    def is_degraded(self) -> bool:
        """Check if quorum is degraded but functional."""
        return self.result == QuorumResult.OK and self.voters_alive < self.voters_total

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "result": self.result.value,
            "reason": self.reason,
            "is_healthy": self.is_healthy,
            "is_degraded": self.is_degraded,
            "voters_total": self.voters_total,
            "voters_alive": self.voters_alive,
            "voters_needed": self.voters_needed,
            "config_version": self.config_version,
            "config_hash": self.config_hash[:16] if self.config_hash else "",
            "alive_voter_ids": self.alive_voter_ids,
            "dead_voter_ids": self.dead_voter_ids,
            "timestamp": self.timestamp,
        }
