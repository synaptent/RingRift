"""Gossip Persistence Mixin - SQLite persistence for gossip state.

Extracted from gossip_protocol.py for modularity.

This mixin provides SQLite-backed persistence for gossip peer states
and learned endpoints. It enables fast cluster state recovery after
P2P orchestrator restarts.

The mixin expects the implementing class to have (via P2PMixinBase):
- _execute_db_query(...) for database operations
- _ensure_table(...) for table creation
- _log_info/debug/warning/error for logging
- _gossip_peer_states: dict[str, dict]
- _gossip_learned_endpoints: dict[str, dict]
- _gossip_state_sync_lock: threading.RLock (optional)
- _cluster_epoch: int

December 2025: Added for SWIM/Raft stability improvements.
February 2026: Extracted as part of gossip_protocol.py decomposition.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class GossipPersistenceMixin:
    """Mixin providing SQLite persistence for gossip state.

    Persists gossip peer states and learned endpoints to SQLite
    for fast recovery after P2P restarts.

    Expects the implementing class to provide:
    - _execute_db_query() from P2PMixinBase
    - _ensure_table() from P2PMixinBase
    - _log_info/debug/warning() from P2PMixinBase
    - _gossip_peer_states: dict
    - _gossip_learned_endpoints: dict
    - _cluster_epoch: int
    """

    # Table names for gossip persistence
    GOSSIP_PERSISTENCE_TABLE = "gossip_peer_states"
    GOSSIP_ENDPOINT_TABLE = "gossip_learned_endpoints"

    def _ensure_gossip_tables(self) -> bool:
        """Ensure gossip persistence tables exist in SQLite.

        Creates tables for storing:
        1. Peer states - Last known state from each peer (jobs, resources, health)
        2. Learned endpoints - Discovered peer connection info

        Returns:
            True if tables are ready, False on error

        December 2025: Added for SWIM/Raft stability improvements.
        Allows gossip state to survive P2P orchestrator restarts, reducing
        the time to recover cluster state after a coordinator failover.
        """
        # Table for peer states
        state_schema = f"""
            CREATE TABLE IF NOT EXISTS {self.GOSSIP_PERSISTENCE_TABLE} (
                node_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                timestamp REAL NOT NULL,
                cluster_epoch INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """
        state_index = f"""
            CREATE INDEX IF NOT EXISTS idx_gossip_timestamp
            ON {self.GOSSIP_PERSISTENCE_TABLE}(timestamp DESC)
        """

        # Table for learned endpoints
        endpoint_schema = f"""
            CREATE TABLE IF NOT EXISTS {self.GOSSIP_ENDPOINT_TABLE} (
                node_id TEXT PRIMARY KEY,
                host TEXT NOT NULL,
                port INTEGER NOT NULL,
                tailscale_ip TEXT,
                learned_at REAL NOT NULL,
                last_verified REAL,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0
            )
        """
        endpoint_index = f"""
            CREATE INDEX IF NOT EXISTS idx_endpoint_learned
            ON {self.GOSSIP_ENDPOINT_TABLE}(learned_at DESC)
        """

        # Use base class helper to create tables
        state_ok = self._ensure_table(
            self.GOSSIP_PERSISTENCE_TABLE,
            state_schema,
            state_index,
        )
        endpoint_ok = self._ensure_table(
            self.GOSSIP_ENDPOINT_TABLE,
            endpoint_schema,
            endpoint_index,
        )

        return state_ok and endpoint_ok

    def _persist_gossip_state(self, node_id: str, state: dict) -> bool:
        """Persist a peer's gossip state to SQLite.

        Args:
            node_id: The peer's node ID
            state: The state dictionary to persist

        Returns:
            True on success, False on error

        Note: Uses upsert pattern (INSERT OR REPLACE) for simplicity.
        """
        now = time.time()
        timestamp = state.get("timestamp", now)
        epoch = state.get("cluster_epoch", getattr(self, "_cluster_epoch", 0))

        try:
            state_json = json.dumps(state)
        except (TypeError, ValueError) as e:
            self._log_warning(f"Failed to serialize state for {node_id}: {e}")
            return False

        result = self._execute_db_query(
            f"""
            INSERT OR REPLACE INTO {self.GOSSIP_PERSISTENCE_TABLE}
            (node_id, state_json, timestamp, cluster_epoch, created_at, updated_at)
            VALUES (?, ?, ?, ?,
                COALESCE((SELECT created_at FROM {self.GOSSIP_PERSISTENCE_TABLE} WHERE node_id = ?), ?),
                ?)
            """,
            (node_id, state_json, timestamp, epoch, node_id, now, now),
            fetch=False,
            commit=True,
        )
        return result is not None and result > 0

    def _persist_learned_endpoint(
        self,
        node_id: str,
        host: str,
        port: int,
        tailscale_ip: str | None = None,
    ) -> bool:
        """Persist a learned peer endpoint to SQLite.

        Args:
            node_id: The peer's node ID
            host: The peer's host/IP address
            port: The peer's port number
            tailscale_ip: Optional Tailscale IP

        Returns:
            True on success, False on error
        """
        now = time.time()

        result = self._execute_db_query(
            f"""
            INSERT INTO {self.GOSSIP_ENDPOINT_TABLE}
            (node_id, host, port, tailscale_ip, learned_at, success_count)
            VALUES (?, ?, ?, ?, ?, 1)
            ON CONFLICT(node_id) DO UPDATE SET
                host = excluded.host,
                port = excluded.port,
                tailscale_ip = COALESCE(excluded.tailscale_ip, tailscale_ip),
                last_verified = excluded.learned_at,
                success_count = success_count + 1
            """,
            (node_id, host, port, tailscale_ip, now),
            fetch=False,
            commit=True,
        )
        return result is not None

    def _load_persisted_gossip_states(self, max_age_seconds: float = 1800.0) -> dict[str, dict]:
        """Load persisted gossip states from SQLite.

        Args:
            max_age_seconds: Only load states newer than this (default 30min)
                            Jan 25, 2026: Reduced from 1h to 30m for cleaner startup

        Returns:
            Dictionary of {node_id: state_dict}

        Called during P2P startup to recover cluster state quickly.
        """
        cutoff = time.time() - max_age_seconds

        rows = self._execute_db_query(
            f"""
            SELECT node_id, state_json, timestamp, cluster_epoch
            FROM {self.GOSSIP_PERSISTENCE_TABLE}
            WHERE timestamp > ?
            ORDER BY timestamp DESC
            """,
            (cutoff,),
            fetch=True,
            commit=False,
        )

        if not rows:
            return {}

        states = {}
        for row in rows:
            try:
                node_id, state_json, timestamp, epoch = row
                state = json.loads(state_json)
                state["timestamp"] = timestamp  # Ensure timestamp is set
                state["cluster_epoch"] = epoch
                state["_persisted"] = True  # Mark as loaded from disk
                states[node_id] = state
            except (json.JSONDecodeError, ValueError, IndexError) as e:
                self._log_warning(f"Failed to load persisted state: {e}")
                continue

        self._log_info(f"Loaded {len(states)} persisted gossip states")
        return states

    def _load_persisted_endpoints(self, max_age_seconds: float = 1800.0) -> dict[str, dict]:
        """Load persisted learned endpoints from SQLite.

        Args:
            max_age_seconds: Only load endpoints newer than this (default 30 min)

        Returns:
            Dictionary of {node_id: endpoint_info}
        """
        cutoff = time.time() - max_age_seconds

        rows = self._execute_db_query(
            f"""
            SELECT node_id, host, port, tailscale_ip, learned_at, success_count, failure_count
            FROM {self.GOSSIP_ENDPOINT_TABLE}
            WHERE learned_at > ?
            ORDER BY success_count DESC
            """,
            (cutoff,),
            fetch=True,
            commit=False,
        )

        if not rows:
            return {}

        endpoints = {}
        for row in rows:
            try:
                node_id, host, port, tailscale_ip, learned_at, successes, failures = row
                endpoints[node_id] = {
                    "host": host,
                    "port": port,
                    "tailscale_ip": tailscale_ip,
                    "learned_at": learned_at,
                    "success_count": successes,
                    "failure_count": failures,
                    "_persisted": True,
                }
            except (ValueError, IndexError) as e:
                self._log_warning(f"Failed to load persisted endpoint: {e}")
                continue

        self._log_info(f"Loaded {len(endpoints)} persisted endpoints")
        return endpoints

    def _cleanup_persisted_gossip_state(self, max_age_seconds: float = 7200.0) -> int:
        """Clean up old persisted gossip state from SQLite.

        Args:
            max_age_seconds: Delete states older than this (default 2 hours)

        Returns:
            Number of rows deleted
        """
        cutoff = time.time() - max_age_seconds

        # Clean peer states
        state_deleted = self._execute_db_query(
            f"DELETE FROM {self.GOSSIP_PERSISTENCE_TABLE} WHERE timestamp < ?",
            (cutoff,),
            fetch=False,
            commit=True,
        ) or 0

        # Clean learned endpoints
        endpoint_deleted = self._execute_db_query(
            f"DELETE FROM {self.GOSSIP_ENDPOINT_TABLE} WHERE learned_at < ?",
            (cutoff,),
            fetch=False,
            commit=True,
        ) or 0

        total = state_deleted + endpoint_deleted
        if total > 0:
            self._log_debug(f"Cleaned {state_deleted} states, {endpoint_deleted} endpoints from persistence")

        return total

    def _save_gossip_state_periodic(self) -> None:
        """Periodically save current gossip state to SQLite.

        Called from the gossip loop to persist state. This ensures that
        if the P2P orchestrator restarts, it can quickly recover the
        cluster state from SQLite rather than waiting for fresh gossip.

        Rate limited to once per minute to avoid excessive disk I/O.
        """
        now = time.time()

        # Rate limit: persist every 60 seconds
        last_persist = getattr(self, "_last_gossip_persist", 0)
        if now - last_persist < 60:
            return
        self._last_gossip_persist = now

        # Ensure tables exist (lazy initialization)
        if not self._ensure_gossip_tables():
            return

        # Persist current peer states
        persisted_count = 0
        for node_id, state in list(self._gossip_peer_states.items()):
            if self._persist_gossip_state(node_id, state):
                persisted_count += 1

        # Persist learned endpoints
        endpoint_count = 0
        for node_id, endpoint in list(self._gossip_learned_endpoints.items()):
            host = endpoint.get("host")
            port = endpoint.get("port")
            if host and port:
                if self._persist_learned_endpoint(
                    node_id,
                    host,
                    port,
                    endpoint.get("tailscale_ip"),
                ):
                    endpoint_count += 1

        # Clean up old persisted data
        self._cleanup_persisted_gossip_state()

        if persisted_count > 0 or endpoint_count > 0:
            self._log_debug(
                f"Persisted {persisted_count} peer states, {endpoint_count} endpoints"
            )

    def _restore_gossip_state_on_startup(self) -> None:
        """Restore gossip state from SQLite on startup.

        Called during P2P initialization to recover cluster state quickly
        after a restart. This reduces the time needed to rebuild the
        cluster view from O(minutes) to O(seconds).

        December 2025: Part of SWIM/Raft stability improvements.
        """
        # Ensure tables exist
        if not self._ensure_gossip_tables():
            self._log_warning("Gossip tables not available, starting with empty state")
            return

        # Load persisted peer states
        persisted_states = self._load_persisted_gossip_states()
        if persisted_states:
            # Jan 3, 2026 (Sprint 15.1): Use sync lock for thread safety during restore
            # Jan 25, 2026: Add timeout to prevent indefinite blocking at startup
            lock = getattr(self, "_gossip_state_sync_lock", None)
            if lock is not None:
                if not lock.acquire(blocking=True, timeout=10.0):
                    self._log_warning("Gossip lock acquisition timed out during startup restore")
                    return
            try:
                # Merge with any existing states (prefer fresher data)
                for node_id, state in persisted_states.items():
                    existing = self._gossip_peer_states.get(node_id)
                    if existing is None or state.get("timestamp", 0) > existing.get("timestamp", 0):
                        self._gossip_peer_states[node_id] = state
            finally:
                if lock is not None:
                    lock.release()

        # Load persisted endpoints
        persisted_endpoints = self._load_persisted_endpoints()
        if persisted_endpoints:
            for node_id, endpoint in persisted_endpoints.items():
                existing = self._gossip_learned_endpoints.get(node_id)
                if existing is None or endpoint.get("learned_at", 0) > existing.get("learned_at", 0):
                    self._gossip_learned_endpoints[node_id] = endpoint

        total_restored = len(persisted_states) + len(persisted_endpoints)
        if total_restored > 0:
            self._log_info(
                f"Restored gossip state: {len(persisted_states)} peer states, "
                f"{len(persisted_endpoints)} endpoints"
            )
