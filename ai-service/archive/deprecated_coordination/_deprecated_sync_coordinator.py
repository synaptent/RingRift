#!/usr/bin/env python3
"""Smart Sync Coordinator for unified cluster-wide data management.

.. deprecated:: December 2025
    This module is being consolidated into sync_core.py as part of the 67→15
    module consolidation effort. For new code, prefer:

    - For sync scheduling: Use :class:`AutoSyncDaemon` from auto_sync_daemon.py
    - For sync execution: Use :class:`SyncCoordinator` from app.distributed.sync_coordinator
    - For ephemeral syncs: Use :class:`EphemeralSyncDaemon` from ephemeral_sync.py

    This module will be archived in Q2 2026.

Architecture Note:
    This module is the SCHEDULING layer for cluster-wide sync coordination.
    It decides WHEN and WHAT to sync based on data freshness and priority.

    For EXECUTION (actually performing syncs), use:
    - :class:`SyncCoordinator` from :mod:`app.distributed.sync_coordinator`
    - Re-exported as `DistributedSyncCoordinator` from :mod:`app.coordination`

    The two layers work together:
    1. This module tracks data freshness and recommends sync operations
    2. The distributed layer executes syncs via aria2/SSH/P2P transports

This module provides centralized coordination for data synchronization across
all distributed hosts, preventing data silos and ensuring efficient transfers.

Features:
1. Data freshness tracking across all hosts
2. Priority-based sync scheduling (hosts with most unsynced data first)
3. Bandwidth-aware transfer balancing
4. Single dashboard view of cluster data state
5. Automatic failover and recovery
6. Integration with existing sync_mutex and bandwidth_manager

Goals:
- Prevent data silos by ensuring all hosts have fresh data
- Maximize training data availability
- Minimize sync latency for high-priority data
- Provide visibility into cluster-wide data state

Usage:
    from app.coordination.sync_coordinator import (
        SyncScheduler,  # Preferred name (2025-12)
        get_sync_scheduler,
        get_cluster_data_status,
        schedule_priority_sync,
        get_sync_recommendations,
    )

    # Get cluster-wide status
    status = get_cluster_data_status()
    print(f"Hosts with stale data: {status.stale_hosts}")

    # Schedule priority sync for hosts with most unsynced data
    await schedule_priority_sync()

    # Get recommendations for sync operations
    recommendations = get_sync_recommendations()
    for rec in recommendations:
        print(f"{rec.host}: {rec.action} - {rec.reason}")
"""

from __future__ import annotations

import warnings

# Emit deprecation warning at import time (December 2025)
warnings.warn(
    "SyncScheduler is deprecated and will be archived in Q2 2026. "
    "Use AutoSyncDaemon for automated sync or SyncFacade for manual sync:\n"
    "\n"
    "For automated P2P sync:\n"
    "  from app.coordination import AutoSyncDaemon\n"
    "  daemon = AutoSyncDaemon()\n"
    "  await daemon.start()\n"
    "\n"
    "For one-time sync operations:\n"
    "  from app.coordination.sync_facade import sync\n"
    "  await sync('games', priority='high')\n"
    "\n"
    "See SYNC_CONSOLIDATION_PLAN.md for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from app.coordination.coordinator_base import (
    CoordinatorBase,
    CoordinatorStatus,
    SQLitePersistenceMixin,
)

logger = logging.getLogger(__name__)

# Backpressure integration (December 2025)
try:
    from app.coordination.queue_monitor import (
        BackpressureLevel,
        QueueType,
        check_backpressure,
        get_throttle_factor,
        report_queue_depth,
        should_stop_production,
        should_throttle_production,
    )
    HAS_QUEUE_MONITOR = True
except ImportError:
    HAS_QUEUE_MONITOR = False
    QueueType = None
    BackpressureLevel = None

    def check_backpressure(*args, **kwargs):
        return False

    def should_throttle_production(*args, **kwargs):
        return False

    def should_stop_production(*args, **kwargs):
        return False

    def get_throttle_factor(*args, **kwargs):
        return 1.0

    def report_queue_depth(*args, **kwargs):
        pass

# Default paths
from app.utils.paths import CONFIG_DIR, DATA_DIR

# Import canonical SyncPriority from sync_constants (consolidated December 2025)
from app.coordination.sync_constants import SyncPriority

DEFAULT_COORDINATOR_DB = DATA_DIR / "coordination" / "sync_coordinator.db"
# Canonical host configuration (December 2025 - consolidated from remote_hosts.yaml)
HOST_CONFIG_PATH = CONFIG_DIR / "distributed_hosts.yaml"

# Thresholds - import from centralized config (December 2025)
try:
    from app.config.thresholds import (
        CRITICAL_STALE_THRESHOLD_SECONDS,
        FRESHNESS_CHECK_INTERVAL,
        MAX_SYNC_QUEUE_SIZE,
        STALE_DATA_THRESHOLD_SECONDS,
    )
except ImportError:
    # Try coordination_defaults first
    try:
        from app.config.coordination_defaults import SyncCoordinatorDefaults
        STALE_DATA_THRESHOLD_SECONDS = 1800  # 30 minutes
        CRITICAL_STALE_THRESHOLD_SECONDS = SyncCoordinatorDefaults.CRITICAL_STALE_THRESHOLD
        MAX_SYNC_QUEUE_SIZE = 20
        FRESHNESS_CHECK_INTERVAL = SyncCoordinatorDefaults.FRESHNESS_CHECK_INTERVAL
    except ImportError:
        # Final fallback defaults
        STALE_DATA_THRESHOLD_SECONDS = 1800  # 30 minutes
        CRITICAL_STALE_THRESHOLD_SECONDS = 3600  # 1 hour
        MAX_SYNC_QUEUE_SIZE = 20
        FRESHNESS_CHECK_INTERVAL = 60
SYNC_PRIORITY_WEIGHTS = {
    "games_behind": 1.0,      # Weight per game behind
    "time_since_sync": 0.01,  # Weight per second since last sync
    "host_priority": 10.0,    # Weight for high-priority hosts
}


# SyncPriority is now imported from sync_constants.py (consolidated December 2025)


class HostType(Enum):
    """Types of hosts with different sync strategies."""
    EPHEMERAL = "ephemeral"  # Vast.ai - can terminate anytime
    PERSISTENT = "persistent"  # AWS, Lambda - stable
    LOCAL = "local"          # Local machine
    ARCHIVE = "archive"      # External drive / archival storage


class SyncAction(Enum):
    """Recommended sync actions."""
    SYNC_NOW = "sync_now"
    SCHEDULE_SYNC = "schedule_sync"
    SKIP = "skip"
    VERIFY_DATA = "verify_data"
    RECOVER_MANIFEST = "recover_manifest"


@dataclass
class HostDataState:
    """Data state for a single host."""
    host: str
    host_type: HostType
    last_sync_time: float = 0.0
    last_sync_games: int = 0
    total_games: int = 0
    estimated_unsynced_games: int = 0
    last_heartbeat: float = 0.0
    is_reachable: bool = True
    sync_in_progress: bool = False
    sync_failures_24h: int = 0
    last_error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def seconds_since_sync(self) -> float:
        return time.time() - self.last_sync_time if self.last_sync_time > 0 else float('inf')

    @property
    def is_stale(self) -> bool:
        return self.seconds_since_sync > STALE_DATA_THRESHOLD_SECONDS

    @property
    def is_critical(self) -> bool:
        # Ephemeral hosts are always critical if they have unsynced data
        if self.host_type == HostType.EPHEMERAL and self.estimated_unsynced_games > 0:
            return True
        return self.seconds_since_sync > CRITICAL_STALE_THRESHOLD_SECONDS

    @property
    def sync_priority_score(self) -> float:
        """Calculate priority score for sync scheduling. Higher = more urgent."""
        score = 0.0

        # Games behind weight
        score += self.estimated_unsynced_games * SYNC_PRIORITY_WEIGHTS["games_behind"]

        # Time since sync weight
        score += self.seconds_since_sync * SYNC_PRIORITY_WEIGHTS["time_since_sync"]

        # Host type multiplier
        if self.host_type == HostType.EPHEMERAL:
            score *= 3.0  # Ephemeral hosts get 3x priority
        elif self.host_type == HostType.PERSISTENT:
            score *= 1.5  # Persistent hosts get 1.5x priority

        # Penalty for recent failures
        if self.sync_failures_24h > 0:
            score *= 0.8 ** self.sync_failures_24h  # Exponential backoff

        return score

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "host_type": self.host_type.value,
            "last_sync_time": datetime.fromtimestamp(self.last_sync_time).isoformat() if self.last_sync_time > 0 else None,
            "seconds_since_sync": round(self.seconds_since_sync, 1),
            "total_games": self.total_games,
            "estimated_unsynced_games": self.estimated_unsynced_games,
            "is_stale": self.is_stale,
            "is_critical": self.is_critical,
            "sync_priority_score": round(self.sync_priority_score, 2),
            "is_reachable": self.is_reachable,
            "sync_in_progress": self.sync_in_progress,
            "sync_failures_24h": self.sync_failures_24h,
        }


@dataclass
class SyncRecommendation:
    """A recommended sync action for a host."""
    host: str
    action: SyncAction
    priority: SyncPriority
    reason: str
    estimated_games: int = 0
    estimated_duration_seconds: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "action": self.action.value,
            "priority": self.priority.value,
            "reason": self.reason,
            "estimated_games": self.estimated_games,
            "estimated_duration_seconds": self.estimated_duration_seconds,
        }


@dataclass
class ClusterDataStatus:
    """Overall cluster data synchronization status."""
    total_hosts: int
    healthy_hosts: int
    stale_hosts: list[str]
    critical_hosts: list[str]
    syncing_hosts: list[str]
    unreachable_hosts: list[str]
    total_games_cluster: int
    estimated_unsynced_games: int
    last_full_sync_time: float
    recommendations: list[SyncRecommendation]
    host_states: dict[str, HostDataState]

    @property
    def cluster_health_score(self) -> float:
        """0-100 score of cluster data health."""
        if self.total_hosts == 0:
            return 100.0

        # Base score from healthy hosts
        health_ratio = self.healthy_hosts / self.total_hosts
        score = health_ratio * 70  # Max 70 points for host health

        # Penalty for stale data
        stale_ratio = len(self.stale_hosts) / self.total_hosts if self.total_hosts > 0 else 0
        score -= stale_ratio * 20

        # Penalty for critical hosts
        critical_ratio = len(self.critical_hosts) / self.total_hosts if self.total_hosts > 0 else 0
        score -= critical_ratio * 30

        # Bonus for recent full sync
        time_since_full = time.time() - self.last_full_sync_time if self.last_full_sync_time > 0 else float('inf')
        if time_since_full < 1800:  # Within 30 min
            score += 30
        elif time_since_full < 3600:  # Within 1 hour
            score += 20
        elif time_since_full < 7200:  # Within 2 hours
            score += 10

        return max(0, min(100, score))

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_hosts": self.total_hosts,
            "healthy_hosts": self.healthy_hosts,
            "stale_hosts": self.stale_hosts,
            "critical_hosts": self.critical_hosts,
            "syncing_hosts": self.syncing_hosts,
            "unreachable_hosts": self.unreachable_hosts,
            "total_games_cluster": self.total_games_cluster,
            "estimated_unsynced_games": self.estimated_unsynced_games,
            "cluster_health_score": round(self.cluster_health_score, 1),
            "last_full_sync_time": datetime.fromtimestamp(self.last_full_sync_time).isoformat() if self.last_full_sync_time > 0 else None,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "host_states": {k: v.to_dict() for k, v in self.host_states.items()},
        }


class SyncScheduler(CoordinatorBase, SQLitePersistenceMixin):
    """Centralized SCHEDULER for cluster-wide data synchronization.

    .. note::
        This class handles SCHEDULING decisions (when/what to sync).
        For EXECUTION (actually performing syncs), use:
        ``from app.distributed.sync_coordinator import SyncCoordinator``

    Extends CoordinatorBase for standardized lifecycle and SQLitePersistenceMixin
    for thread-safe database access.

    This class provides:
    1. Unified view of data state across all hosts
    2. Priority-based sync scheduling
    3. Bandwidth-aware transfer management
    4. Automatic recovery from sync failures
    """

    _instance: SyncScheduler | None = None
    _singleton_lock = threading.RLock()

    def __init__(self, db_path: Path | None = None):
        CoordinatorBase.__init__(self, name="SyncScheduler")
        self._host_states: dict[str, HostDataState] = {}
        self._sync_queue: list[SyncRecommendation] = []
        self._callbacks: list[Callable[[SyncRecommendation], None]] = []
        self._last_full_sync_time: float = 0.0

        # Initialize SQLite persistence
        db_path = db_path or DEFAULT_COORDINATOR_DB
        self.init_db(db_path)

        self._load_host_config()
        self._load_state()

        # Mark as ready
        self._status = CoordinatorStatus.READY

    @classmethod
    def get_instance(cls, db_path: Path | None = None) -> SyncScheduler:
        """Get or create singleton instance."""
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls(db_path)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton for testing."""
        with cls._singleton_lock:
            if cls._instance is not None:
                cls._instance._save_state()
            cls._instance = None

    # =========================================================================
    # Database Management (via SQLitePersistenceMixin)
    # =========================================================================

    def _get_schema(self) -> str:
        """Get database schema SQL."""
        return """
            -- Host state table
            CREATE TABLE IF NOT EXISTS host_state (
                host TEXT PRIMARY KEY,
                host_type TEXT NOT NULL,
                last_sync_time REAL DEFAULT 0,
                last_sync_games INTEGER DEFAULT 0,
                total_games INTEGER DEFAULT 0,
                estimated_unsynced INTEGER DEFAULT 0,
                last_heartbeat REAL DEFAULT 0,
                is_reachable INTEGER DEFAULT 1,
                sync_failures_24h INTEGER DEFAULT 0,
                last_error TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}'
            );

            -- Sync history table
            CREATE TABLE IF NOT EXISTS sync_history (
                sync_id INTEGER PRIMARY KEY AUTOINCREMENT,
                host TEXT NOT NULL,
                started_at REAL NOT NULL,
                completed_at REAL,
                games_synced INTEGER DEFAULT 0,
                bytes_transferred INTEGER DEFAULT 0,
                success INTEGER DEFAULT 0,
                error_message TEXT DEFAULT '',
                duration_seconds REAL DEFAULT 0
            );

            -- Create indexes
            CREATE INDEX IF NOT EXISTS idx_sync_history_host
            ON sync_history(host, started_at DESC);

            -- Coordinator state table
            CREATE TABLE IF NOT EXISTS coordinator_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at REAL
            );
        """

    def _load_state(self) -> None:
        """Load state from database."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Load host states
        cursor.execute("SELECT * FROM host_state")
        for row in cursor.fetchall():
            host = row["host"]
            self._host_states[host] = HostDataState(
                host=host,
                host_type=HostType(row["host_type"]),
                last_sync_time=row["last_sync_time"] or 0,
                last_sync_games=row["last_sync_games"] or 0,
                total_games=row["total_games"] or 0,
                estimated_unsynced_games=row["estimated_unsynced"] or 0,
                last_heartbeat=row["last_heartbeat"] or 0,
                is_reachable=bool(row["is_reachable"]),
                sync_failures_24h=row["sync_failures_24h"] or 0,
                last_error=row["last_error"] or "",
                metadata=json.loads(row["metadata"] or "{}"),
            )

        # Load coordinator state
        cursor.execute("SELECT value FROM coordinator_state WHERE key = 'last_full_sync_time'")
        row = cursor.fetchone()
        if row:
            self._last_full_sync_time = float(row["value"])

        logger.info(f"[SyncCoordinator] Loaded state for {len(self._host_states)} hosts")

    def _save_state(self) -> None:
        """Save state to database with transaction isolation."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Wrap all state updates in single transaction for atomicity
            cursor.execute("BEGIN IMMEDIATE")

            for host, state in self._host_states.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO host_state
                    (host, host_type, last_sync_time, last_sync_games, total_games,
                     estimated_unsynced, last_heartbeat, is_reachable, sync_failures_24h,
                     last_error, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    host,
                    state.host_type.value,
                    state.last_sync_time,
                    state.last_sync_games,
                    state.total_games,
                    state.estimated_unsynced_games,
                    state.last_heartbeat,
                    1 if state.is_reachable else 0,
                    state.sync_failures_24h,
                    state.last_error,
                    json.dumps(state.metadata),
                ))

            cursor.execute("""
                INSERT OR REPLACE INTO coordinator_state (key, value, updated_at)
                VALUES ('last_full_sync_time', ?, ?)
            """, (str(self._last_full_sync_time), time.time()))

            conn.commit()

        except Exception as e:
            try:
                conn.rollback()
            except sqlite3.Error as rollback_err:
                logger.debug(f"[SyncCoordinator] Rollback after save state error: {rollback_err}")
            logger.warning(f"[SyncCoordinator] Failed to save state: {e}")

    def _load_host_config(self) -> None:
        """Load host configuration from YAML."""
        try:
            import yaml
            if HOST_CONFIG_PATH.exists():
                with open(HOST_CONFIG_PATH) as f:
                    config = yaml.safe_load(f)

                # Add standard hosts (dict format: {host_name: {config...}})
                standard_hosts = config.get("standard_hosts", {})
                if isinstance(standard_hosts, dict):
                    for host_name, host_config in standard_hosts.items():
                        if host_name and host_name not in self._host_states:
                            host_type = HostType.PERSISTENT
                            if "vast" in host_name.lower():
                                host_type = HostType.EPHEMERAL
                            self._host_states[host_name] = HostDataState(
                                host=host_name,
                                host_type=host_type,
                                metadata=host_config if isinstance(host_config, dict) else {},
                            )

                # Add vast hosts (ephemeral) - also dict format
                vast_hosts = config.get("vast_hosts", {})
                if isinstance(vast_hosts, dict):
                    for host_name, host_config in vast_hosts.items():
                        if host_name and host_name not in self._host_states:
                            self._host_states[host_name] = HostDataState(
                                host=host_name,
                                host_type=HostType.EPHEMERAL,
                                metadata=host_config if isinstance(host_config, dict) else {},
                            )

                standard_count = len(standard_hosts) if isinstance(standard_hosts, dict) else 0
                vast_count = len(vast_hosts) if isinstance(vast_hosts, dict) else 0
                logger.info(f"[SyncCoordinator] Loaded {standard_count} standard hosts, "
                           f"{vast_count} vast hosts from config")
        except Exception as e:
            logger.warning(f"[SyncCoordinator] Failed to load host config: {e}")

    # =========================================================================
    # Host State Management
    # =========================================================================

    def register_host(
        self,
        host: str,
        host_type: HostType = HostType.PERSISTENT,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a host for sync coordination."""
        if host not in self._host_states:
            self._host_states[host] = HostDataState(
                host=host,
                host_type=host_type,
                metadata=metadata or {},
            )
            self._save_state()
            logger.info(f"[SyncCoordinator] Registered host: {host} ({host_type.value})")

    def update_host_state(
        self,
        host: str,
        total_games: int | None = None,
        estimated_unsynced: int | None = None,
        is_reachable: bool | None = None,
        heartbeat: bool = False,
    ) -> None:
        """Update the data state for a host."""
        if host not in self._host_states:
            self.register_host(host)

        state = self._host_states[host]

        if total_games is not None:
            state.total_games = total_games

        if estimated_unsynced is not None:
            state.estimated_unsynced_games = estimated_unsynced

        if is_reachable is not None:
            state.is_reachable = is_reachable

        if heartbeat:
            state.last_heartbeat = time.time()

        self._save_state()

    def record_sync_start(self, host: str) -> int:
        """Record that a sync operation has started with transaction isolation."""
        if host in self._host_states:
            self._host_states[host].sync_in_progress = True

        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            cursor.execute("""
                INSERT INTO sync_history (host, started_at)
                VALUES (?, ?)
            """, (host, time.time()))
            # Capture lastrowid BEFORE commit (critical for thread safety)
            sync_id = cursor.lastrowid
            conn.commit()
            return sync_id
        except Exception as e:
            try:
                conn.rollback()
            except sqlite3.Error as rollback_err:
                logger.debug(f"[SyncCoordinator] Rollback after sync start error: {rollback_err}")
            logger.warning(f"[SyncCoordinator] Failed to record sync start: {e}")
            return -1

    def record_sync_complete(
        self,
        host: str,
        sync_id: int,
        games_synced: int,
        bytes_transferred: int = 0,
        success: bool = True,
        error_message: str = "",
    ) -> None:
        """Record that a sync operation has completed with transaction isolation."""
        now = time.time()

        if host in self._host_states:
            state = self._host_states[host]
            state.sync_in_progress = False
            state.last_sync_time = now

            if success:
                state.last_sync_games = games_synced
                state.estimated_unsynced_games = max(0, state.estimated_unsynced_games - games_synced)
                state.last_error = ""
            else:
                state.sync_failures_24h += 1
                state.last_error = error_message

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Wrap SELECT + UPDATE in single transaction for consistency
            cursor.execute("BEGIN IMMEDIATE")

            # Get start time for duration calculation
            cursor.execute("SELECT started_at FROM sync_history WHERE sync_id = ?", (sync_id,))
            row = cursor.fetchone()
            started_at = row["started_at"] if row else now
            duration = now - started_at

            cursor.execute("""
                UPDATE sync_history
                SET completed_at = ?, games_synced = ?, bytes_transferred = ?,
                    success = ?, error_message = ?, duration_seconds = ?
                WHERE sync_id = ?
            """, (now, games_synced, bytes_transferred, 1 if success else 0,
                  error_message, duration, sync_id))
            conn.commit()
        except Exception as e:
            try:
                conn.rollback()
            except sqlite3.Error as rollback_err:
                logger.debug(f"[SyncCoordinator] Rollback after sync complete error: {rollback_err}")
            logger.warning(f"[SyncCoordinator] Failed to record sync complete: {e}")
            return

        self._save_state()

        if success:
            logger.info(f"[SyncCoordinator] Sync completed for {host}: {games_synced} games in {duration:.1f}s")
        else:
            logger.warning(f"[SyncCoordinator] Sync failed for {host}: {error_message}")

    def record_games_generated(self, host: str, games: int) -> None:
        """Record that games were generated on a host (increases unsynced count)."""
        if host not in self._host_states:
            self.register_host(host)

        state = self._host_states[host]
        state.total_games += games
        state.estimated_unsynced_games += games
        state.last_heartbeat = time.time()

        # Don't save on every game - batch saves
        # self._save_state()

    # =========================================================================
    # Cluster Status
    # =========================================================================

    def get_cluster_status(self) -> ClusterDataStatus:
        """Get overall cluster data synchronization status."""
        stale_hosts = []
        critical_hosts = []
        syncing_hosts = []
        unreachable_hosts = []
        healthy_hosts = 0
        total_games = 0
        total_unsynced = 0

        for host, state in self._host_states.items():
            total_games += state.total_games
            total_unsynced += state.estimated_unsynced_games

            if not state.is_reachable:
                unreachable_hosts.append(host)
            elif state.sync_in_progress:
                syncing_hosts.append(host)
            elif state.is_critical:
                critical_hosts.append(host)
            elif state.is_stale:
                stale_hosts.append(host)
            else:
                healthy_hosts += 1

        recommendations = self.get_sync_recommendations()

        return ClusterDataStatus(
            total_hosts=len(self._host_states),
            healthy_hosts=healthy_hosts,
            stale_hosts=stale_hosts,
            critical_hosts=critical_hosts,
            syncing_hosts=syncing_hosts,
            unreachable_hosts=unreachable_hosts,
            total_games_cluster=total_games,
            estimated_unsynced_games=total_unsynced,
            last_full_sync_time=self._last_full_sync_time,
            recommendations=recommendations,
            host_states=dict(self._host_states),
        )

    def get_host_state(self, host: str) -> HostDataState | None:
        """Get the data state for a specific host."""
        return self._host_states.get(host)

    async def get_stats(self) -> dict[str, Any]:
        """Get sync coordinator statistics.

        Implements CoordinatorBase.get_stats() interface.
        """
        # Get base stats from CoordinatorBase
        base_stats = await super().get_stats()

        # Get cluster status
        status = self.get_cluster_status()

        # Merge with sync-specific stats
        base_stats.update({
            "total_hosts": status.total_hosts,
            "healthy_hosts": status.healthy_hosts,
            "stale_hosts": len(status.stale_hosts),
            "critical_hosts": len(status.critical_hosts),
            "syncing_hosts": len(status.syncing_hosts),
            "unreachable_hosts": len(status.unreachable_hosts),
            "total_games_cluster": status.total_games_cluster,
            "estimated_unsynced_games": status.estimated_unsynced_games,
            "cluster_health_score": round(status.cluster_health_score, 1),
            "last_full_sync_time": status.last_full_sync_time,
            "pending_recommendations": len(status.recommendations),
        })
        return base_stats

    def get_stats_sync(self) -> dict[str, Any]:
        """Synchronous version of get_stats for non-async contexts."""
        status = self.get_cluster_status()
        return {
            "name": self.name,
            "status": self.status.value,
            "total_hosts": status.total_hosts,
            "healthy_hosts": status.healthy_hosts,
            "stale_hosts": len(status.stale_hosts),
            "critical_hosts": len(status.critical_hosts),
            "syncing_hosts": len(status.syncing_hosts),
            "unreachable_hosts": len(status.unreachable_hosts),
            "total_games_cluster": status.total_games_cluster,
            "estimated_unsynced_games": status.estimated_unsynced_games,
            "cluster_health_score": round(status.cluster_health_score, 1),
            "last_full_sync_time": status.last_full_sync_time,
            "pending_recommendations": len(status.recommendations),
        }

    # =========================================================================
    # Sync Scheduling
    # =========================================================================

    def get_sync_recommendations(
        self,
        max_recommendations: int = 5,
        respect_backpressure: bool = True,
    ) -> list[SyncRecommendation]:
        """Get prioritized sync recommendations for the cluster.

        Args:
            max_recommendations: Maximum number of recommendations to return
            respect_backpressure: If True, reduce recommendations when under backpressure

        Returns:
            List of SyncRecommendation objects
        """
        recommendations = []

        # Check backpressure (December 2025)
        backpressure_info = self.check_sync_backpressure()
        if respect_backpressure and backpressure_info["should_stop"]:
            logger.info(f"Sync backpressure: stopping new syncs ({backpressure_info['reason']})")
            return []  # No new syncs when hard limit reached

        # Adjust max recommendations based on throttle factor
        if respect_backpressure and backpressure_info["should_throttle"]:
            throttle_factor = backpressure_info["throttle_factor"]
            max_recommendations = max(1, int(max_recommendations * throttle_factor))
            logger.debug(f"Sync backpressure: throttling to {max_recommendations} recommendations")

        # Score and sort hosts by sync priority
        scored_hosts = [
            (host, state, state.sync_priority_score)
            for host, state in self._host_states.items()
            if state.is_reachable and not state.sync_in_progress
        ]
        scored_hosts.sort(key=lambda x: x[2], reverse=True)

        for host, state, _score in scored_hosts[:max_recommendations]:
            # Determine action and priority
            if state.is_critical:
                action = SyncAction.SYNC_NOW
                priority = SyncPriority.CRITICAL
                reason = f"Critical: {state.estimated_unsynced_games} unsynced games, {state.seconds_since_sync/60:.0f}min since sync"
            elif state.is_stale:
                action = SyncAction.SYNC_NOW
                priority = SyncPriority.HIGH
                reason = f"Stale: {state.seconds_since_sync/60:.0f}min since last sync"
            elif state.estimated_unsynced_games > 100:
                action = SyncAction.SCHEDULE_SYNC
                priority = SyncPriority.NORMAL
                reason = f"{state.estimated_unsynced_games} games waiting to sync"
            elif state.estimated_unsynced_games > 0:
                action = SyncAction.SCHEDULE_SYNC
                priority = SyncPriority.LOW
                reason = f"{state.estimated_unsynced_games} games to sync"
            else:
                continue  # No recommendation needed

            # Under backpressure, downgrade non-critical actions
            if (respect_backpressure and backpressure_info["should_throttle"]
                    and priority not in (SyncPriority.CRITICAL, SyncPriority.HIGH)):
                action = SyncAction.SKIP
                reason = f"Deferred due to backpressure: {reason}"

            # Estimate duration based on historical data
            estimated_duration = self._estimate_sync_duration(host, state.estimated_unsynced_games)

            recommendations.append(SyncRecommendation(
                host=host,
                action=action,
                priority=priority,
                reason=reason,
                estimated_games=state.estimated_unsynced_games,
                estimated_duration_seconds=estimated_duration,
                metadata={"backpressure": backpressure_info},
            ))

        return recommendations

    # =========================================================================
    # Backpressure Integration (December 2025)
    # =========================================================================

    def check_sync_backpressure(self) -> dict[str, Any]:
        """Check if sync operations should be throttled due to backpressure.

        Returns:
            Dict with backpressure status:
            - should_throttle: True if soft limit exceeded
            - should_stop: True if hard limit exceeded
            - throttle_factor: 0.0-1.0 multiplier for sync rate
            - reason: Human-readable reason
        """
        if not HAS_QUEUE_MONITOR:
            return {
                "should_throttle": False,
                "should_stop": False,
                "throttle_factor": 1.0,
                "reason": "Queue monitor not available",
            }

        # Check sync queue specifically
        should_stop = should_stop_production(QueueType.SYNC_QUEUE)
        should_throttle = should_throttle_production(QueueType.SYNC_QUEUE)
        throttle_factor = get_throttle_factor(QueueType.SYNC_QUEUE)

        # Also check training data queue - don't sync if training is backlogged
        training_throttle = should_throttle_production(QueueType.TRAINING_DATA)
        training_stop = should_stop_production(QueueType.TRAINING_DATA)

        if training_stop:
            return {
                "should_throttle": True,
                "should_stop": True,
                "throttle_factor": 0.0,
                "reason": "Training data queue at hard limit",
            }

        if should_stop:
            return {
                "should_throttle": True,
                "should_stop": True,
                "throttle_factor": 0.0,
                "reason": "Sync queue at hard limit",
            }

        if training_throttle:
            # Reduce throttle factor further if training is backlogged
            training_factor = get_throttle_factor(QueueType.TRAINING_DATA)
            throttle_factor = min(throttle_factor, training_factor)
            should_throttle = True

        if should_throttle:
            return {
                "should_throttle": True,
                "should_stop": False,
                "throttle_factor": throttle_factor,
                "reason": f"Throttling at {throttle_factor:.0%} capacity",
            }

        return {
            "should_throttle": False,
            "should_stop": False,
            "throttle_factor": 1.0,
            "reason": "No backpressure",
        }

    def report_sync_queue_depth(self, depth: int | None = None) -> None:
        """Report the current sync queue depth for backpressure tracking.

        Args:
            depth: Queue depth to report. If None, computes from host states.
        """
        if not HAS_QUEUE_MONITOR:
            return

        if depth is None:
            # Compute depth from host states
            depth = sum(
                state.estimated_unsynced_games
                for state in self._host_states.values()
                if state.is_reachable
            )

        report_queue_depth(QueueType.SYNC_QUEUE, depth)

    def should_allow_sync(self, priority: SyncPriority = SyncPriority.NORMAL) -> bool:
        """Check if a sync operation should be allowed given current backpressure.

        Critical and high priority syncs are always allowed.
        Normal and low priority are subject to backpressure.

        Args:
            priority: Priority of the sync operation

        Returns:
            True if sync should proceed
        """
        # Critical and high priority always allowed
        if priority in (SyncPriority.CRITICAL, SyncPriority.HIGH):
            return True

        backpressure = self.check_sync_backpressure()

        if backpressure["should_stop"]:
            return False

        if backpressure["should_throttle"]:
            # Probabilistic throttling based on factor
            import random
            return random.random() < backpressure["throttle_factor"]

        return True

    def get_next_sync_target(self) -> str | None:
        """Get the highest priority host that should be synced next."""
        recommendations = self.get_sync_recommendations(max_recommendations=1)
        if recommendations and recommendations[0].action in (SyncAction.SYNC_NOW, SyncAction.SCHEDULE_SYNC):
            return recommendations[0].host
        return None

    # =========================================================================
    # Execution Bridge (December 2025)
    # Delegates to app.distributed.sync_coordinator for actual sync execution
    # =========================================================================

    async def execute_priority_sync(
        self,
        max_syncs: int = 3,
    ) -> dict[str, Any]:
        """Execute sync operations for highest priority hosts.

        This bridges the scheduling logic in this module with the execution
        layer in app.distributed.sync_coordinator.

        Args:
            max_syncs: Maximum number of hosts to sync in this batch

        Returns:
            Dict with sync execution results
        """
        results = {
            "syncs_attempted": 0,
            "syncs_completed": 0,
            "total_files": 0,
            "total_bytes": 0,
            "errors": [],
        }

        # Get recommendations
        recommendations = self.get_sync_recommendations(max_recommendations=max_syncs)
        if not recommendations:
            logger.debug("[SyncCoordinator] No sync recommendations to execute")
            return results

        # Filter to actionable recommendations
        actionable = [r for r in recommendations if r.action in (SyncAction.SYNC_NOW, SyncAction.SCHEDULE_SYNC)]
        if not actionable:
            return results

        # Import distributed layer executor
        try:
            from app.distributed.sync_coordinator import SyncCoordinator as DistributedSyncCoordinator
            executor = DistributedSyncCoordinator.get_instance()
        except ImportError as e:
            logger.warning(f"[SyncCoordinator] Distributed executor not available: {e}")
            results["errors"].append(f"Distributed layer unavailable: {e}")
            return results

        # Execute syncs for each recommended host
        for rec in actionable:
            host = rec.host
            sync_id = self.record_sync_start(host)
            results["syncs_attempted"] += 1

            try:
                # Execute via distributed layer
                sync_result = await executor.full_cluster_sync()

                # Update coordination layer tracking
                games_synced = sync_result.total_files_synced
                bytes_transferred = sync_result.total_bytes_transferred

                self.record_sync_complete(
                    host=host,
                    sync_id=sync_id,
                    games_synced=games_synced,
                    bytes_transferred=bytes_transferred,
                    success=True,
                )

                results["syncs_completed"] += 1
                results["total_files"] += games_synced
                results["total_bytes"] += bytes_transferred

                logger.info(f"[SyncCoordinator] Executed sync for {host}: "
                           f"{games_synced} files, {bytes_transferred / (1024*1024):.1f}MB")

            except Exception as e:
                self.record_sync_complete(
                    host=host,
                    sync_id=sync_id,
                    games_synced=0,
                    success=False,
                    error_message=str(e),
                )
                results["errors"].append(f"{host}: {e}")
                logger.warning(f"[SyncCoordinator] Sync execution failed for {host}: {e}")

        if results["syncs_completed"] == len(actionable):
            self.record_full_sync_complete()

        return results

    def _estimate_sync_duration(self, host: str, games: int) -> int:
        """Estimate sync duration based on historical data."""
        if games == 0:
            return 0

        conn = self._get_connection()
        cursor = conn.cursor()

        # Get average sync rate for this host
        cursor.execute("""
            SELECT AVG(games_synced / NULLIF(duration_seconds, 0)) as games_per_second
            FROM sync_history
            WHERE host = ? AND success = 1 AND duration_seconds > 0
            ORDER BY completed_at DESC
            LIMIT 10
        """, (host,))
        row = cursor.fetchone()

        if row and row["games_per_second"]:
            games_per_second = row["games_per_second"]
            return int(games / games_per_second)

        # Default: assume 10 games per second
        return games // 10

    def record_full_sync_complete(self) -> None:
        """Record that a full cluster sync has completed."""
        self._last_full_sync_time = time.time()
        self._save_state()
        logger.info("[SyncCoordinator] Full cluster sync completed")

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup_old_history(self, days: int = 7) -> int:
        """Remove sync history older than specified days."""
        cutoff = time.time() - (days * 86400)

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sync_history WHERE completed_at < ?", (cutoff,))
        deleted = cursor.rowcount
        conn.commit()

        if deleted > 0:
            logger.info(f"[SyncCoordinator] Cleaned up {deleted} old sync history records")

        return deleted

    def reset_failure_counts(self) -> None:
        """Reset 24-hour failure counts (called daily)."""
        for state in self._host_states.values():
            state.sync_failures_24h = 0
        self._save_state()


# =============================================================================
# Module-level convenience functions
# =============================================================================

_coordinator: SyncScheduler | None = None


def get_sync_coordinator() -> SyncScheduler:
    """Get the singleton sync scheduler.

    .. deprecated:: 2025-12
        Use :func:`get_sync_scheduler` instead for clarity.
    """
    return SyncScheduler.get_instance()


def get_cluster_data_status() -> ClusterDataStatus:
    """Get overall cluster data synchronization status."""
    return get_sync_coordinator().get_cluster_status()


def get_sync_recommendations(max_recommendations: int = 5) -> list[SyncRecommendation]:
    """Get prioritized sync recommendations."""
    return get_sync_coordinator().get_sync_recommendations(max_recommendations)


def get_next_sync_target() -> str | None:
    """Get the highest priority host to sync next."""
    return get_sync_coordinator().get_next_sync_target()


def register_host(
    host: str,
    host_type: HostType = HostType.PERSISTENT,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Register a host for sync coordination."""
    get_sync_coordinator().register_host(host, host_type, metadata)


def update_host_state(
    host: str,
    total_games: int | None = None,
    estimated_unsynced: int | None = None,
    is_reachable: bool | None = None,
    heartbeat: bool = False,
) -> None:
    """Update the data state for a host."""
    get_sync_coordinator().update_host_state(
        host, total_games, estimated_unsynced, is_reachable, heartbeat
    )


def record_sync_start(host: str) -> int:
    """Record that a sync operation has started."""
    return get_sync_coordinator().record_sync_start(host)


def record_sync_complete(
    host: str,
    sync_id: int,
    games_synced: int,
    bytes_transferred: int = 0,
    success: bool = True,
    error_message: str = "",
) -> None:
    """Record that a sync operation has completed."""
    get_sync_coordinator().record_sync_complete(
        host, sync_id, games_synced, bytes_transferred, success, error_message
    )


def record_games_generated(host: str, games: int) -> None:
    """Record that games were generated on a host."""
    get_sync_coordinator().record_games_generated(host, games)


async def execute_priority_sync(max_syncs: int = 3) -> dict[str, Any]:
    """Execute sync operations for highest priority hosts.

    This bridges the scheduling layer with the distributed execution layer.
    """
    return await get_sync_coordinator().execute_priority_sync(max_syncs)


def reset_sync_coordinator() -> None:
    """Reset the coordinator singleton.

    .. deprecated:: 2025-12
        Use :func:`reset_sync_scheduler` instead.
    """
    SyncScheduler.reset_instance()


# =============================================================================
# Deprecation and Aliasing (December 2025)
# =============================================================================
#
# This module provides SCHEDULING for sync operations (when/what to sync).
# For EXECUTION (how to sync), use app.distributed.sync_coordinator.
#
# To reduce confusion from the name collision, we provide these aliases:
# =============================================================================
# Naming Convention (2025-12)
# =============================================================================
# - SyncScheduler: Canonical name for this SCHEDULING layer
# - SyncCoordinator: Deprecated alias for backwards compatibility
#
# Import recommendation:
#   # For scheduling (this module) - PREFERRED
#   from app.coordination.sync_coordinator import SyncScheduler, get_sync_scheduler
#
#   # For execution (distributed module)
#   from app.distributed.sync_coordinator import SyncCoordinator
#

# =============================================================================
# DEPRECATED ALIAS (December 2025)
# =============================================================================
# SyncCoordinator is a deprecated alias for SyncScheduler.
# This alias exists for backwards compatibility but should not be used in new code.
#
# The naming was changed to avoid collision with:
#   app.distributed.sync_coordinator.SyncCoordinator (execution layer)
#
# New code should use:
#   from app.coordination.sync_coordinator import SyncScheduler
#   or: from app.coordination import SyncScheduler
#
# For execution layer:
#   from app.distributed.sync_coordinator import SyncCoordinator
#   or: from app.coordination import DistributedSyncCoordinator
# =============================================================================
SyncCoordinator = SyncScheduler  # DEPRECATED: Use SyncScheduler


def get_sync_scheduler(db_path: Path | None = None) -> SyncScheduler:
    """Get the sync scheduler singleton.

    This is the preferred function for getting the scheduling coordinator.
    """
    return SyncScheduler.get_instance(db_path)


def reset_sync_scheduler() -> None:
    """Reset the sync scheduler singleton."""
    SyncScheduler.reset_instance()


def wire_sync_events() -> SyncScheduler:
    """Wire sync coordinator to the event router for automatic updates.

    Subscribes to:
    - DATA_SYNC_COMPLETED: Update host state after successful sync
    - DATA_SYNC_FAILED: Mark sync failure for scheduling
    - NEW_GAMES_AVAILABLE: Track new games for sync prioritization

    Returns:
        The configured SyncScheduler instance
    """
    scheduler = get_sync_scheduler()

    try:
        from app.coordination.event_router import get_router
        from app.coordination.event_router import DataEventType  # Types still needed

        router = get_router()

        def _event_payload(event: Any) -> dict[str, Any]:
            if isinstance(event, dict):
                return event
            payload = getattr(event, "payload", None)
            return payload if isinstance(payload, dict) else {}

        def _on_sync_completed(event: Any) -> None:
            """Handle sync completion - update host state."""
            payload = _event_payload(event)
            host = payload.get("host") or payload.get("source_host")
            games_synced = payload.get("games_synced", 0)
            if host:
                scheduler.record_sync_complete(host, success=True, games_synced=games_synced)

        def _on_sync_failed(event: Any) -> None:
            """Handle sync failure - update scheduling."""
            payload = _event_payload(event)
            host = payload.get("host") or payload.get("source_host")
            error = payload.get("error", "Unknown error")
            if host:
                scheduler.record_sync_complete(host, success=False, error=error)

        def _on_new_games(event: Any) -> None:
            """Handle new games - update priority."""
            payload = _event_payload(event)
            host = payload.get("host") or payload.get("source_host")
            count = payload.get("count", 1)
            if host:
                scheduler.record_games_generated(host, count)

        def _on_sync_stalled(event: Any) -> None:
            """Handle sync stall - mark host as temporarily unavailable and try alternative sources.

            Dec 2025: Critical handler to prevent sync deadlocks from blocking training.
            When sync stalls (timeout/failure), we:
            1. Mark source host as temporarily unavailable
            2. Log stall for monitoring
            3. SyncScheduler will automatically select alternative sources on next sync
            """
            payload = _event_payload(event)
            source_host = payload.get("source_host", "unknown")
            target_host = payload.get("target_host", "unknown")
            timeout = payload.get("timeout_seconds", 0)
            retry_count = payload.get("retry_count", 0)

            logger.warning(
                f"[SyncScheduler] SYNC_STALLED: {source_host} -> {target_host} "
                f"(timeout: {timeout}s, retries: {retry_count}). "
                f"Marking {source_host} as temporarily unavailable for failover."
            )

            # Mark failed host - SyncScheduler will avoid it temporarily
            scheduler.record_sync_complete(source_host, success=False,
                                          error=f"Sync stalled after {timeout}s")

        router.subscribe(DataEventType.DATA_SYNC_COMPLETED.value, _on_sync_completed)
        router.subscribe(DataEventType.DATA_SYNC_FAILED.value, _on_sync_failed)
        router.subscribe(DataEventType.NEW_GAMES_AVAILABLE.value, _on_new_games)
        router.subscribe(DataEventType.SYNC_STALLED.value, _on_sync_stalled)

        logger.info("[SyncScheduler] Wired to event router (DATA_SYNC_COMPLETED, DATA_SYNC_FAILED, NEW_GAMES_AVAILABLE, SYNC_STALLED)")

    except ImportError:
        logger.warning("[SyncScheduler] data_events not available, running without event router")

    return scheduler


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "ClusterDataStatus",
    # Data classes
    "HostDataState",
    "HostType",
    "SyncAction",
    "SyncCoordinator",  # Deprecated alias
    # Enums
    "SyncPriority",
    "SyncRecommendation",
    # Main class
    "SyncScheduler",
    "execute_priority_sync",
    "get_cluster_data_status",
    "get_next_sync_target",
    # Functions
    "get_sync_coordinator",
    "get_sync_recommendations",
    "get_sync_scheduler",
    "record_games_generated",
    "record_sync_complete",
    "record_sync_start",
    "register_host",
    "reset_sync_coordinator",
    "reset_sync_scheduler",
    "update_host_state",
    "wire_sync_events",
]
