"""Peer Manager Mixin - Peer Discovery & Management.

Extracted from p2p_orchestrator.py for modularity.
This mixin provides peer discovery, reputation tracking, and cache management.

Usage:
    class P2POrchestrator(PeerManagerMixin, ...):
        pass

Phase 2.1 extraction - Dec 26, 2025
Refactored to use P2PMixinBase - Dec 27, 2025
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from scripts.p2p.p2p_mixin_base import P2PMixinBase

if TYPE_CHECKING:
    from pathlib import Path

    from scripts.p2p.models import NodeInfo

logger = logging.getLogger(__name__)

# Load constants with fallbacks using base class helper
_CONSTANTS = P2PMixinBase._load_config_constants({
    "PEER_CACHE_MAX_ENTRIES": 200,
    "PEER_CACHE_TTL_SECONDS": 604800,  # 7 days
    "PEER_REPUTATION_ALPHA": 0.2,
})

PEER_CACHE_MAX_ENTRIES = _CONSTANTS["PEER_CACHE_MAX_ENTRIES"]
PEER_CACHE_TTL_SECONDS = _CONSTANTS["PEER_CACHE_TTL_SECONDS"]
PEER_REPUTATION_ALPHA = _CONSTANTS["PEER_REPUTATION_ALPHA"]


class PeerManagerMixin(P2PMixinBase):
    """Mixin providing peer management functionality.

    Inherits from P2PMixinBase for shared database and state helpers.

    Requires the implementing class to have:
    - db_path: Path - Path to SQLite database
    - bootstrap_seeds: list[str] - Initial peer addresses
    - verbose: bool - Verbose logging flag
    - node_id: str - This node's ID
    - peers: dict[str, NodeInfo] - Active peers
    - peers_lock: threading.RLock - Lock for peers dict
    - _peer_reputations: dict[str, float] - Peer reputation scores
    - _peer_last_seen: dict[str, float] - Last heartbeat times
    - _nat_blocked_peers: set[str] - Peers behind NAT
    """

    MIXIN_TYPE = "peer_manager"

    # Type hints for IDE support (implemented by P2POrchestrator)
    db_path: "Path"
    bootstrap_seeds: list[str]
    verbose: bool
    node_id: str
    peers: dict[str, Any]  # dict[str, NodeInfo]

    def _update_peer_reputation(self, peer_addr_or_node_id: str, success: bool) -> None:
        """Update peer reputation based on interaction success.

        Uses exponential moving average (EMA) for smooth reputation updates.
        Higher reputation = more reliable peer for bootstrap.
        """
        with self._db_connection() as conn:
            if not conn:
                return

            try:
                cursor = conn.cursor()

                # Get current reputation
                cursor.execute(
                    "SELECT reputation_score, success_count, failure_count FROM peer_cache WHERE node_id = ?",
                    (peer_addr_or_node_id,),
                )
                row = cursor.fetchone()

                if row:
                    current_score = float(row[0] or 0.5)
                    success_count = int(row[1] or 0)
                    failure_count = int(row[2] or 0)
                else:
                    current_score = 0.5
                    success_count = 0
                    failure_count = 0

                # EMA update: new_score = alpha * outcome + (1-alpha) * current
                outcome = 1.0 if success else 0.0
                new_score = PEER_REPUTATION_ALPHA * outcome + (1 - PEER_REPUTATION_ALPHA) * current_score

                # Update counts
                if success:
                    success_count += 1
                else:
                    failure_count += 1

                # Upsert
                cursor.execute(
                    """
                    INSERT INTO peer_cache (node_id, host, port, reputation_score, success_count, failure_count, last_seen)
                    VALUES (?, '', 0, ?, ?, ?, ?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        reputation_score = ?,
                        success_count = ?,
                        failure_count = ?,
                        last_seen = ?
                """,
                    (
                        peer_addr_or_node_id,
                        new_score,
                        success_count,
                        failure_count,
                        time.time(),
                        new_score,
                        success_count,
                        failure_count,
                        time.time(),
                    ),
                )
                conn.commit()
            except Exception as e:
                if self.verbose:
                    self._log_debug(f"Error updating peer reputation: {e}")

    def _save_peer_to_cache(
        self,
        node_id: str,
        host: str,
        port: int,
        tailscale_ip: str | None = None,
    ) -> None:
        """Save a peer to the cache for persistence across restarts."""
        if not node_id or node_id == self.node_id:
            return

        with self._db_connection() as conn:
            if not conn:
                return

            try:
                cursor = conn.cursor()

                # Check if this is a known bootstrap seed
                is_seed = f"{host}:{port}" in self.bootstrap_seeds

                cursor.execute(
                    """
                    INSERT INTO peer_cache (node_id, host, port, tailscale_ip, last_seen, is_bootstrap_seed)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        host = COALESCE(NULLIF(?, ''), host),
                        port = CASE WHEN ? > 0 THEN ? ELSE port END,
                        tailscale_ip = COALESCE(NULLIF(?, ''), tailscale_ip),
                        last_seen = ?,
                        is_bootstrap_seed = ?
                """,
                    (
                        node_id,
                        host,
                        port,
                        tailscale_ip or "",
                        time.time(),
                        is_seed,
                        host,
                        port,
                        port,
                        tailscale_ip or "",
                        time.time(),
                        is_seed,
                    ),
                )
                conn.commit()

                # Prune old entries if needed
                cursor.execute("SELECT COUNT(*) FROM peer_cache")
                count = cursor.fetchone()[0]
                if count > PEER_CACHE_MAX_ENTRIES:
                    # Delete oldest entries with lowest reputation
                    cursor.execute(
                        """
                        DELETE FROM peer_cache
                        WHERE node_id IN (
                            SELECT node_id FROM peer_cache
                            WHERE is_bootstrap_seed = 0
                            ORDER BY reputation_score ASC, last_seen ASC
                            LIMIT ?
                        )
                    """,
                        (count - PEER_CACHE_MAX_ENTRIES,),
                    )
                    conn.commit()

            except Exception as e:
                if self.verbose:
                    self._log_debug(f"Error saving peer to cache: {e}")

    def _get_bootstrap_peers_by_reputation(self, limit: int = 5) -> list[str]:
        """Get most reliable cached peers for bootstrap.

        Returns list of "host:port" strings ordered by reputation.
        Filters out peers not seen in the last 7 days.
        """
        with self._db_connection() as conn:
            if not conn:
                return []

            try:
                cursor = conn.cursor()

                # Get peers seen recently, ordered by seed status then reputation
                cutoff = time.time() - PEER_CACHE_TTL_SECONDS
                cursor.execute(
                    """
                    SELECT host, port, tailscale_ip FROM peer_cache
                    WHERE last_seen > ? AND host != '' AND port > 0
                    ORDER BY is_bootstrap_seed DESC, reputation_score DESC
                    LIMIT ?
                """,
                    (cutoff, limit),
                )

                result = []
                for row in cursor.fetchall():
                    host = row[0]
                    port = row[1]
                    ts_ip = row[2]
                    # Prefer Tailscale IP if available
                    if ts_ip:
                        result.append(f"{ts_ip}:{port}")
                    elif host:
                        result.append(f"{host}:{port}")
                return result

            except Exception as e:
                if self.verbose:
                    self._log_debug(f"Error getting cached peers: {e}")
                return []

    def _get_peer_health_score(self, peer_id: str) -> float:
        """Get health score for a peer based on reputation and sync history.

        Returns score between 0.0 (unhealthy) and 1.0 (healthy).
        """
        # Start with reputation score
        reputation = getattr(self, "_peer_reputations", {}).get(peer_id, 0.5)

        # Factor in recent sync success
        self._ensure_state_attr("_p2p_sync_results", {})
        sync_history = self._p2p_sync_results.get(peer_id, [])
        if sync_history:
            recent_syncs = sync_history[-10:]  # Last 10 syncs
            sync_success_rate = sum(1 for s in recent_syncs if s) / len(recent_syncs)
        else:
            sync_success_rate = 0.5  # Neutral if no history

        # Weight reputation more heavily than sync history
        health_score = 0.7 * reputation + 0.3 * sync_success_rate
        return min(1.0, max(0.0, health_score))

    def _record_p2p_sync_result(self, peer_id: str, success: bool) -> None:
        """Record sync result for a peer.

        Used for health scoring and routing decisions.
        """
        # Ensure sync results dict exists using base class helper
        self._ensure_state_attr("_p2p_sync_results", {})

        if peer_id not in self._p2p_sync_results:
            self._p2p_sync_results[peer_id] = []

        self._p2p_sync_results[peer_id].append(success)

        # Keep only last 50 results per peer
        if len(self._p2p_sync_results[peer_id]) > 50:
            self._p2p_sync_results[peer_id] = self._p2p_sync_results[peer_id][-50:]

        # Also update reputation
        self._update_peer_reputation(peer_id, success)

    def _get_cached_peer_count(self) -> int:
        """Get count of peers in the cache."""
        result = self._execute_db_query(
            "SELECT COUNT(*) FROM peer_cache",
            fetch=True,
            commit=False,
        )
        if result and len(result) > 0:
            return result[0][0]
        return 0

    def _clear_peer_cache(self) -> int:
        """Clear all non-seed peers from cache. Returns count deleted."""
        result = self._execute_db_query(
            "DELETE FROM peer_cache WHERE is_bootstrap_seed = 0",
            fetch=False,
        )
        return result if result is not None else 0

    def _prune_stale_peers(self, max_age_seconds: float = 86400) -> int:
        """Remove peers not seen in max_age_seconds. Returns count pruned."""
        cutoff = time.time() - max_age_seconds
        result = self._execute_db_query(
            "DELETE FROM peer_cache WHERE last_seen < ? AND is_bootstrap_seed = 0",
            (cutoff,),
            fetch=False,
        )
        return result if result is not None else 0


# Convenience function to get singleton (if P2POrchestrator uses this)
_peer_manager: PeerManagerMixin | None = None


def get_peer_manager() -> PeerManagerMixin | None:
    """Get the peer manager instance (set by P2POrchestrator on init)."""
    return _peer_manager


def set_peer_manager(instance: PeerManagerMixin) -> None:
    """Set the peer manager instance."""
    global _peer_manager
    _peer_manager = instance
