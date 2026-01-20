"""Voter Configuration Manager for P2P Cluster Stability.

Part of Phase 2: Voter Configuration Management
P2P Cluster Stability Plan - Jan 13, 2026

This module provides the single source of truth for voter configuration,
with version tracking, integrity verification, and P2P sync support.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from app.coordination.voter_config_types import (
    QuorumHealth,
    QuorumResult,
    VoterConfigVersion,
)

logger = logging.getLogger(__name__)

# Feature flag for strict quorum enforcement
STRICT_QUORUM_ENABLED = os.environ.get(
    "RINGRIFT_STRICT_QUORUM_ENFORCEMENT", "false"
).lower() == "true"

# Minimum voters required for fault-tolerant quorum
MIN_VOTERS_FOR_QUORUM = 3


class VoterConfigManager:
    """Manages voter configuration with version tracking and integrity.

    This is a singleton that provides:
    - Single source of truth for voter configuration
    - Version tracking with SHA256 integrity hashes
    - Persistence with atomic writes
    - P2P sync protocol support
    - Strict quorum enforcement (when enabled)

    Usage:
        manager = get_voter_config_manager()
        config = manager.get_current()

        # Check quorum
        health = manager.check_quorum(alive_voter_ids=["node1", "node2"])
        if health.is_healthy:
            proceed_with_election()
    """

    _instance: VoterConfigManager | None = None
    _lock = threading.Lock()

    def __init__(self, persist_path: Path | None = None):
        """Initialize the voter config manager.

        Args:
            persist_path: Path to persist voter config. If None, uses default.
        """
        self._config: VoterConfigVersion | None = None
        self._persist_path = persist_path or self._default_persist_path()
        self._update_callbacks: list[Callable[[VoterConfigVersion], None]] = []
        self._config_lock = threading.RLock()

        # Load persisted config
        self._load_persisted()

    @classmethod
    def get_instance(cls, persist_path: Path | None = None) -> VoterConfigManager:
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(persist_path)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def _default_persist_path(self) -> Path:
        """Get default persistence path."""
        # Try multiple locations
        paths = [
            Path(__file__).parent.parent.parent.parent / "data" / "voter_config.json",
            Path.cwd() / "data" / "voter_config.json",
        ]
        for path in paths:
            if path.parent.exists():
                return path
        # Fallback to first option, creating parent if needed
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        return paths[0]

    def _load_persisted(self) -> None:
        """Load voter config from persistent storage."""
        if not self._persist_path.exists():
            logger.debug(f"No persisted voter config at {self._persist_path}")
            return

        try:
            with open(self._persist_path) as f:
                data = json.load(f)
            config = VoterConfigVersion.from_dict(data)

            # Verify integrity
            if not config.verify_integrity():
                logger.error(
                    f"Persisted voter config failed integrity check, ignoring"
                )
                return

            self._config = config
            logger.info(
                f"Loaded voter config v{config.version} with {len(config.voters)} voters"
            )
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.error(f"Failed to load voter config: {e}")

    def _persist(self) -> None:
        """Persist voter config to storage atomically."""
        if self._config is None:
            return

        try:
            temp_path = self._persist_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(self._config.to_dict(), f, indent=2)

            # Atomic rename
            temp_path.rename(self._persist_path)
            logger.debug(f"Persisted voter config v{self._config.version}")
        except OSError as e:
            logger.error(f"Failed to persist voter config: {e}")

    def get_current(self) -> VoterConfigVersion | None:
        """Get current voter configuration.

        Returns:
            Current VoterConfigVersion or None if not configured
        """
        with self._config_lock:
            return self._config

    def update(
        self,
        voters: list[str],
        created_by: str = "unknown",
        force: bool = False,
    ) -> VoterConfigVersion:
        """Update voter configuration.

        Creates a new version with incremented version number.

        Args:
            voters: New list of voter node IDs
            created_by: Node ID making the update
            force: If True, allow update even if config hasn't changed

        Returns:
            New VoterConfigVersion
        """
        with self._config_lock:
            # Compute new version number
            new_version = 1 if self._config is None else self._config.version + 1

            # Create new config
            new_config = VoterConfigVersion.create(
                voters=voters,
                version=new_version,
                created_by=created_by,
            )

            # Check if actually changed
            if not force and self._config is not None:
                if set(self._config.voters) == set(new_config.voters):
                    logger.debug("Voter config unchanged, skipping update")
                    return self._config

            self._config = new_config
            self._persist()

            logger.info(
                f"Updated voter config to v{new_version}: "
                f"{len(voters)} voters, hash={new_config.sha256_hash[:16]}"
            )

            # Notify callbacks
            for callback in self._update_callbacks:
                try:
                    callback(new_config)
                except Exception as e:
                    logger.error(f"Voter config callback failed: {e}")

            return new_config

    def load_from_yaml(self, config_path: Path | None = None) -> VoterConfigVersion | None:
        """Load voter configuration from distributed_hosts.yaml.

        Args:
            config_path: Path to distributed_hosts.yaml. If None, searches default locations.

        Returns:
            New VoterConfigVersion if loaded, None otherwise
        """
        import yaml

        if config_path is None:
            search_paths = [
                Path(__file__).parent.parent.parent.parent / "config" / "distributed_hosts.yaml",
                Path.cwd() / "config" / "distributed_hosts.yaml",
            ]
            for path in search_paths:
                if path.exists():
                    config_path = path
                    break

        if config_path is None or not config_path.exists():
            logger.error("distributed_hosts.yaml not found")
            return None

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            voters = config.get("p2p_voters", [])
            if not voters:
                logger.warning("No p2p_voters found in config")
                return None

            return self.update(voters, created_by="yaml_load")
        except (yaml.YAMLError, OSError) as e:
            logger.error(f"Failed to load voters from YAML: {e}")
            return None

    def apply_remote_config(
        self,
        remote_config: VoterConfigVersion,
        source: str = "sync",
    ) -> tuple[bool, str]:
        """Apply a voter config received from a remote peer.

        This is the core sync mechanism. Only applies if:
        1. Remote version > local version (never downgrade)
        2. Remote config passes integrity check

        Jan 20, 2026: Added for automated voter config synchronization.

        Args:
            remote_config: VoterConfigVersion received from remote peer
            source: String identifying source (e.g., "sync", "leader", "gossip")

        Returns:
            Tuple of (success: bool, reason: str)
        """
        with self._config_lock:
            # Validate integrity
            if not remote_config.verify_integrity():
                logger.error(
                    f"[VoterConfigSync] Remote config v{remote_config.version} "
                    f"failed integrity check, rejecting"
                )
                return False, "integrity_check_failed"

            # Check version ordering
            local_version = self._config.version if self._config else 0
            if remote_config.version <= local_version:
                logger.debug(
                    f"[VoterConfigSync] Remote v{remote_config.version} <= local v{local_version}, "
                    f"ignoring (source={source})"
                )
                return False, "version_not_newer"

            # Check minimum voters
            if not remote_config.has_minimum_voters(MIN_VOTERS_FOR_QUORUM):
                logger.warning(
                    f"[VoterConfigSync] Remote config has only {len(remote_config.voters)} voters, "
                    f"need {MIN_VOTERS_FOR_QUORUM}"
                )
                return False, "insufficient_voters"

            # Apply the remote config
            old_voters = self._config.voters if self._config else []
            self._config = remote_config
            self._persist()

            logger.info(
                f"[VoterConfigSync] Applied remote config v{remote_config.version} from {source}: "
                f"{len(remote_config.voters)} voters, hash={remote_config.sha256_hash[:16]}"
            )

            # Notify callbacks
            for callback in self._update_callbacks:
                try:
                    callback(remote_config)
                except Exception as e:
                    logger.error(f"Voter config callback failed: {e}")

            # Log voter changes for audit
            added = set(remote_config.voters) - set(old_voters)
            removed = set(old_voters) - set(remote_config.voters)
            if added or removed:
                logger.info(
                    f"[VoterConfigSync] Voter changes - added: {sorted(added)}, removed: {sorted(removed)}"
                )

            return True, "applied"

    def check_quorum(
        self,
        alive_voter_ids: list[str] | None = None,
        alive_check_fn: Callable[[str], bool] | None = None,
    ) -> QuorumHealth:
        """Check quorum health with detailed diagnostics.

        Args:
            alive_voter_ids: List of voter IDs known to be alive
            alive_check_fn: Optional function to check if a voter is alive

        Returns:
            QuorumHealth with detailed status
        """
        with self._config_lock:
            config = self._config

            # No config loaded
            if config is None:
                return QuorumHealth(
                    result=QuorumResult.CONFIG_MISSING,
                    reason="No voter configuration loaded",
                )

            # Verify integrity
            if not config.verify_integrity():
                return QuorumHealth(
                    result=QuorumResult.CONFIG_INVALID,
                    reason="Voter configuration integrity check failed",
                    config_version=config.version,
                    config_hash=config.sha256_hash,
                )

            # Check minimum voters
            if not config.has_minimum_voters(MIN_VOTERS_FOR_QUORUM):
                return QuorumHealth(
                    result=QuorumResult.INSUFFICIENT_VOTERS,
                    reason=f"Only {len(config.voters)} voters configured, need {MIN_VOTERS_FOR_QUORUM}",
                    voters_total=len(config.voters),
                    voters_needed=config.quorum_size,
                    config_version=config.version,
                    config_hash=config.sha256_hash,
                )

            # Determine alive voters
            alive_set: set[str] = set()
            dead_set: set[str] = set()

            if alive_voter_ids is not None:
                alive_set = set(alive_voter_ids) & set(config.voters)
                dead_set = set(config.voters) - alive_set
            elif alive_check_fn is not None:
                for voter_id in config.voters:
                    if alive_check_fn(voter_id):
                        alive_set.add(voter_id)
                    else:
                        dead_set.add(voter_id)
            else:
                # No liveness info provided - assume all alive (permissive)
                alive_set = set(config.voters)

            # Check quorum
            voters_alive = len(alive_set)
            has_quorum = voters_alive >= config.quorum_size

            if has_quorum:
                return QuorumHealth(
                    result=QuorumResult.OK,
                    reason=f"Quorum achieved: {voters_alive}/{len(config.voters)} voters alive",
                    voters_total=len(config.voters),
                    voters_alive=voters_alive,
                    voters_needed=config.quorum_size,
                    config_version=config.version,
                    config_hash=config.sha256_hash,
                    alive_voter_ids=sorted(alive_set),
                    dead_voter_ids=sorted(dead_set),
                )
            else:
                return QuorumHealth(
                    result=QuorumResult.LOST,
                    reason=f"Quorum lost: {voters_alive}/{len(config.voters)} voters alive, need {config.quorum_size}",
                    voters_total=len(config.voters),
                    voters_alive=voters_alive,
                    voters_needed=config.quorum_size,
                    config_version=config.version,
                    config_hash=config.sha256_hash,
                    alive_voter_ids=sorted(alive_set),
                    dead_voter_ids=sorted(dead_set),
                )

    def check_quorum_strict(
        self,
        alive_voter_ids: list[str] | None = None,
        alive_check_fn: Callable[[str], bool] | None = None,
    ) -> tuple[QuorumResult, str]:
        """Strict quorum check - NO bypass paths.

        This is the enforcement point for elections. When strict quorum
        is enabled, elections CANNOT proceed without proper quorum.

        Args:
            alive_voter_ids: List of voter IDs known to be alive
            alive_check_fn: Optional function to check if a voter is alive

        Returns:
            Tuple of (QuorumResult, reason_string)
        """
        health = self.check_quorum(alive_voter_ids, alive_check_fn)

        # Log for monitoring
        if health.result != QuorumResult.OK:
            if STRICT_QUORUM_ENABLED:
                logger.warning(
                    f"[STRICT QUORUM] {health.result.value}: {health.reason}"
                )
            else:
                logger.debug(
                    f"[QUORUM] {health.result.value}: {health.reason} "
                    f"(strict mode disabled, would block election)"
                )

        return health.result, health.reason

    def on_config_update(self, callback: Callable[[VoterConfigVersion], None]) -> None:
        """Register a callback for config updates.

        Args:
            callback: Function called with new config when updated
        """
        self._update_callbacks.append(callback)

    def get_config_for_gossip(self) -> dict[str, Any]:
        """Get minimal config info for gossip protocol.

        Returns:
            Dict with hash, version, and timestamp for drift detection
        """
        with self._config_lock:
            if self._config is None:
                return {"hash": None, "version": 0, "timestamp": 0}

            return {
                "hash": self._config.sha256_hash[:16],
                "version": self._config.version,
                "timestamp": self._config.created_at,
                "voter_count": len(self._config.voters),
            }

    def should_pull_config(self, remote_gossip: dict[str, Any]) -> bool:
        """Check if we should pull config from a remote node.

        Args:
            remote_gossip: Gossip data from remote node

        Returns:
            True if remote has newer/different config we should pull
        """
        with self._config_lock:
            remote_version = remote_gossip.get("version", 0)
            remote_hash = remote_gossip.get("hash", "")

            if self._config is None:
                return remote_version > 0

            # Pull if remote has higher version
            if remote_version > self._config.version:
                return True

            # Pull if same version but different hash (conflict)
            if (
                remote_version == self._config.version
                and remote_hash
                and remote_hash != self._config.sha256_hash[:16]
            ):
                logger.warning(
                    f"Voter config hash mismatch at v{remote_version}: "
                    f"local={self._config.sha256_hash[:16]}, remote={remote_hash}"
                )
                return True

            return False


# Module-level singleton accessor
_manager_instance: VoterConfigManager | None = None


def get_voter_config_manager() -> VoterConfigManager:
    """Get the global VoterConfigManager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = VoterConfigManager.get_instance()
    return _manager_instance


def reset_voter_config_manager() -> None:
    """Reset the global instance (for testing)."""
    global _manager_instance
    _manager_instance = None
    VoterConfigManager.reset_instance()
