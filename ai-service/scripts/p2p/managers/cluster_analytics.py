"""Cluster Analytics Manager for P2P Orchestrator.

January 2026: Phase 4 P2P Orchestrator Deep Decomposition

This module extracts cluster-wide analytics and summary generation
from p2p_orchestrator.py for better modularity and testability.

Responsibilities:
- Training distribution statistics
- ELO rating aggregation (local and cluster-wide)
- Peer reputation scoring and aggregation
- Sync interval tracking
- Data deduplication metrics

Usage:
    from scripts.p2p.managers.cluster_analytics import (
        ClusterAnalytics,
        ClusterAnalyticsConfig,
        get_cluster_analytics,
    )

    # Create with callbacks
    analytics = ClusterAnalytics(
        config=ClusterAnalyticsConfig(),
        get_gossip_states=lambda: {},
        get_peer_reputation=lambda: {},
        get_sync_intervals=lambda: {},
        get_dedup_stats=lambda: {},
    )

    # Get cluster-wide ELO summary
    elo_summary = analytics.get_cluster_elo_summary()

    # Get training distribution
    training_summary = analytics.get_distributed_training_summary()
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Callable

logger = logging.getLogger(__name__)

# Module-level singleton
_cluster_analytics: ClusterAnalytics | None = None
_analytics_lock = threading.Lock()


@dataclass
class ClusterAnalyticsConfig:
    """Configuration for cluster analytics.

    Attributes:
        elo_cache_ttl: TTL for ELO summary cache in seconds
        startup_grace_period: Seconds to wait before querying ELO DB
        stale_gossip_threshold: Seconds before gossip state is considered stale
        top_models_limit: Number of top models to include in summaries
    """

    elo_cache_ttl: float = 600.0
    startup_grace_period: float = 60.0
    stale_gossip_threshold: float = 300.0
    top_models_limit: int = 5


@dataclass
class EloSummary:
    """ELO rating summary for a node or cluster.

    Attributes:
        top_models: List of top models with ratings
        total_models: Total number of rated models
        last_update: Timestamp of last ELO update
        deferred: Whether query was deferred due to startup grace period
    """

    top_models: list[dict[str, Any]] = field(default_factory=list)
    total_models: int = 0
    last_update: float = 0.0
    deferred: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "top_models": self.top_models,
            "total_models": self.total_models,
            "last_update": self.last_update,
            "deferred": self.deferred,
        }


@dataclass
class ReputationSummary:
    """Peer reputation summary.

    Attributes:
        reliable_peers: List of high-reputation peers
        unreliable_peers: List of low-reputation peers
    """

    reliable_peers: list[dict[str, Any]] = field(default_factory=list)
    unreliable_peers: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reliable_peers": self.reliable_peers,
            "unreliable_peers": self.unreliable_peers,
        }


class ClusterAnalytics:
    """Cluster-wide analytics aggregation and reporting.

    Provides summary methods for training, ELO, reputation, sync, and
    deduplication metrics. Uses callbacks to access orchestrator state
    for loose coupling.
    """

    def __init__(
        self,
        config: ClusterAnalyticsConfig | None = None,
        get_gossip_states: Callable[[], dict[str, dict]] | None = None,
        get_peer_reputation: Callable[[], dict[str, dict]] | None = None,
        get_reputation_score: Callable[[str], float] | None = None,
        get_sync_intervals: Callable[[], dict[str, float]] | None = None,
        get_sync_streaks: Callable[[], tuple[dict[str, int], dict[str, int]]] | None = None,
        get_activity_factor: Callable[[], float] | None = None,
        get_dedup_stats: Callable[[], dict[str, Any]] | None = None,
        get_training_configs: Callable[[], dict[str, list[str]]] | None = None,
    ):
        """Initialize cluster analytics.

        Args:
            config: Analytics configuration
            get_gossip_states: Callback to get gossip peer states
            get_peer_reputation: Callback to get peer reputation data
            get_reputation_score: Callback to compute reputation score for a peer
            get_sync_intervals: Callback to get adaptive sync intervals
            get_sync_streaks: Callback to get (success_streaks, failure_streaks)
            get_activity_factor: Callback to get cluster activity factor
            get_dedup_stats: Callback to get deduplication stats
            get_training_configs: Callback to get active training configs by node
        """
        self._config = config or ClusterAnalyticsConfig()
        self._get_gossip_states = get_gossip_states or (lambda: {})
        self._get_peer_reputation = get_peer_reputation or (lambda: {})
        self._get_reputation_score = get_reputation_score or (lambda _: 50.0)
        self._get_sync_intervals = get_sync_intervals or (lambda: {})
        self._get_sync_streaks = get_sync_streaks or (lambda: ({}, {}))
        self._get_activity_factor = get_activity_factor or (lambda: 1.0)
        self._get_dedup_stats = get_dedup_stats or (lambda: {})
        self._get_training_configs = get_training_configs or (lambda: {})

        self._lock = threading.Lock()
        self._startup_time = time.time()

        # ELO cache
        self._elo_cache: EloSummary | None = None
        self._elo_cache_time: float = 0.0

    # =========================================================================
    # DISTRIBUTED TRAINING ANALYTICS
    # =========================================================================

    def get_distributed_training_summary(self) -> dict[str, Any]:
        """Get summary of distributed training state.

        Returns dict with:
        - active_configs: List of config keys with active training
        - total_training_jobs: Total training jobs across cluster
        - configs_by_node_count: Map of config -> number of nodes training it
        """
        cluster_configs = self._get_training_configs()
        return {
            "active_configs": list(cluster_configs.keys()),
            "total_training_jobs": sum(len(nodes) for nodes in cluster_configs.values()),
            "configs_by_node_count": {k: len(v) for k, v in cluster_configs.items()},
        }

    # =========================================================================
    # ELO ANALYTICS
    # =========================================================================

    def get_local_elo_summary(self) -> EloSummary:
        """Get summary of local ELO ratings for gossip propagation.

        Uses lazy loading to defer ELO query until after startup grace period.
        Uses TTL-based cache to reduce database load.

        Returns:
            EloSummary with top models and statistics
        """
        now = time.time()

        # Lazy loading: don't query during startup grace period
        if now - self._startup_time < self._config.startup_grace_period:
            return EloSummary(deferred=True)

        # Check cache
        with self._lock:
            if (
                self._elo_cache is not None
                and now - self._elo_cache_time < self._config.elo_cache_ttl
            ):
                return self._elo_cache

        # Query ELO database
        summary = EloSummary()

        try:
            from app.tournament import get_elo_database

            db = get_elo_database()

            with db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT participant_id, rating, games_played, last_update,
                           (SELECT COUNT(*) FROM elo_ratings) as total,
                           (SELECT MAX(last_update) FROM elo_ratings) as max_updated
                    FROM elo_ratings
                    ORDER BY rating DESC
                    LIMIT ?
                """,
                    (self._config.top_models_limit,),
                )
                rows = cursor.fetchall()

                if rows:
                    summary.total_models = rows[0][4] if rows[0][4] else 0
                    summary.last_update = rows[0][5] if rows[0][5] else 0

                for row in rows:
                    summary.top_models.append(
                        {
                            "model": row[0],
                            "elo": round(row[1]),
                            "games": row[2],
                        }
                    )

        except (
            KeyError,
            IndexError,
            AttributeError,
            ImportError,
            sqlite3.OperationalError,
        ):
            # Silently fail - ELO summary is optional
            pass

        # Update cache
        with self._lock:
            self._elo_cache = summary
            self._elo_cache_time = now

        return summary

    def get_cluster_elo_summary(self) -> dict[str, Any]:
        """Get cluster-wide ELO summary from gossip state.

        Aggregates ELO info from all nodes via gossip to get a cluster-wide
        view of model performance. Keeps highest ELO seen for each model.

        Returns dict with:
        - top_models: Top 10 models by ELO across cluster
        - total_unique_models: Total unique models tracked
        """
        all_models: dict[str, dict] = {}
        gossip_states = self._get_gossip_states()
        now = time.time()

        # Include our own ELO summary
        local_summary = self.get_local_elo_summary()
        for model_info in local_summary.top_models:
            model_name = model_info.get("model", "")
            if model_name:
                all_models[model_name] = model_info

        # Include ELO summaries from gossip
        for _node_id, state in gossip_states.items():
            if state.get("timestamp", 0) < now - self._config.stale_gossip_threshold:
                continue

            elo_summary = state.get("elo_summary", {})
            for model_info in elo_summary.get("top_models", []):
                model_name = model_info.get("model", "")
                if model_name:
                    existing = all_models.get(model_name, {})
                    if model_info.get("elo", 0) > existing.get("elo", 0):
                        all_models[model_name] = model_info

        # Sort by ELO and return top 10
        sorted_models = sorted(
            all_models.values(), key=lambda x: x.get("elo", 0), reverse=True
        )
        return {
            "top_models": sorted_models[:10],
            "total_unique_models": len(all_models),
        }

    # =========================================================================
    # PEER REPUTATION ANALYTICS
    # =========================================================================

    def get_peer_reputation_summary(self) -> ReputationSummary:
        """Get summary of peer reputation for gossip propagation.

        Returns top/bottom peers by reputation score to help cluster
        converge on reliable peer selection.
        """
        reputation_data = self._get_peer_reputation()

        scores = []
        for peer_id in reputation_data:
            score = self._get_reputation_score(peer_id)
            scores.append((peer_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        return ReputationSummary(
            reliable_peers=[
                {"peer": p, "score": round(s)} for p, s in scores[:5] if s >= 70
            ],
            unreliable_peers=[
                {"peer": p, "score": round(s)} for p, s in scores[-3:] if s < 30
            ],
        )

    def get_cluster_peer_reputation(self) -> dict[str, Any]:
        """Aggregate peer reputation from gossip for cluster-wide view.

        Returns dict with:
        - most_reliable: Top 10 most reliable peers with average scores
        - peers_tracked: Total number of peers tracked
        """
        all_scores: dict[str, list[float]] = {}
        gossip_states = self._get_gossip_states()
        now = time.time()

        # Include our own reputation data
        local_summary = self.get_peer_reputation_summary()
        for peer_info in local_summary.reliable_peers:
            peer_id = peer_info["peer"]
            if peer_id not in all_scores:
                all_scores[peer_id] = []
            all_scores[peer_id].append(peer_info["score"])

        # Include reputation from gossip
        for _node_id, state in gossip_states.items():
            if state.get("timestamp", 0) < now - self._config.stale_gossip_threshold:
                continue

            rep_summary = state.get("peer_reputation", {})
            for peer_info in rep_summary.get("reliable_peers", []):
                peer_id = peer_info["peer"]
                if peer_id not in all_scores:
                    all_scores[peer_id] = []
                all_scores[peer_id].append(peer_info["score"])

        # Calculate average scores
        avg_scores = {
            peer: sum(scores) / len(scores)
            for peer, scores in all_scores.items()
            if scores
        }
        sorted_peers = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            "most_reliable": [
                {"peer": p, "avg_score": round(s)} for p, s in sorted_peers[:10]
            ],
            "peers_tracked": len(all_scores),
        }

    # =========================================================================
    # SYNC INTERVAL ANALYTICS
    # =========================================================================

    def get_sync_interval_summary(self) -> dict[str, Any]:
        """Get summary of current adaptive sync intervals for monitoring.

        Returns dict with:
        - data_interval: Current data sync interval
        - model_interval: Current model sync interval
        - training_db_interval: Current training DB sync interval
        - activity_factor: Cluster activity factor
        - data_streak: Success/failure streak for data sync
        - model_streak: Success/failure streak for model sync
        """
        intervals = self._get_sync_intervals()
        success_streaks, failure_streaks = self._get_sync_streaks()

        return {
            "data_interval": round(intervals.get("data", 0)),
            "model_interval": round(intervals.get("model", 0)),
            "training_db_interval": round(intervals.get("training_db", 0)),
            "activity_factor": round(self._get_activity_factor(), 2),
            "data_streak": {
                "success": success_streaks.get("data", 0),
                "failure": failure_streaks.get("data", 0),
            },
            "model_streak": {
                "success": success_streaks.get("model", 0),
                "failure": failure_streaks.get("model", 0),
            },
        }

    # =========================================================================
    # DEDUPLICATION ANALYTICS
    # =========================================================================

    def get_dedup_summary(self) -> dict[str, Any]:
        """Get deduplication metrics summary.

        Returns dict with:
        - files_skipped: Number of files skipped due to deduplication
        - games_skipped: Number of games skipped
        - bytes_saved_mb: Bytes saved in MB
        - known_file_hashes: Number of tracked file hashes
        - known_game_ids: Number of tracked game IDs
        """
        stats = self._get_dedup_stats()

        return {
            "files_skipped": stats.get("files_skipped", 0),
            "games_skipped": stats.get("games_skipped", 0),
            "bytes_saved_mb": round(stats.get("bytes_saved", 0) / (1024 * 1024), 2),
            "known_file_hashes": stats.get("known_file_hashes", 0),
            "known_game_ids": stats.get("known_game_ids", 0),
        }

    # =========================================================================
    # COMBINED SUMMARY
    # =========================================================================

    def get_full_analytics_summary(self) -> dict[str, Any]:
        """Get combined analytics summary for monitoring endpoints.

        Returns dict with all analytics sections.
        """
        return {
            "training": self.get_distributed_training_summary(),
            "elo": {
                "local": self.get_local_elo_summary().to_dict(),
                "cluster": self.get_cluster_elo_summary(),
            },
            "reputation": {
                "local": self.get_peer_reputation_summary().to_dict(),
                "cluster": self.get_cluster_peer_reputation(),
            },
            "sync_intervals": self.get_sync_interval_summary(),
            "deduplication": self.get_dedup_summary(),
        }

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    def health_check(self) -> dict[str, Any]:
        """Return health status for daemon manager integration.

        Returns dict with:
        - status: "healthy", "degraded", or "error"
        - message: Human-readable status message
        - details: Additional details for debugging
        """
        try:
            # Try to get basic summaries
            elo_summary = self.get_local_elo_summary()
            training_summary = self.get_distributed_training_summary()

            status = "healthy"
            message = "ClusterAnalytics operational"

            if elo_summary.deferred:
                status = "degraded"
                message = "ELO queries deferred during startup"

            return {
                "status": status,
                "message": message,
                "details": {
                    "elo_models_tracked": elo_summary.total_models,
                    "elo_cache_age_seconds": round(
                        time.time() - self._elo_cache_time
                    )
                    if self._elo_cache_time
                    else None,
                    "training_active_configs": len(
                        training_summary.get("active_configs", [])
                    ),
                    "startup_age_seconds": round(time.time() - self._startup_time),
                },
            }
        except Exception as e:
            logger.debug(f"[ClusterAnalytics] Health check error: {e}")
            return {
                "status": "error",
                "message": f"Health check failed: {e}",
                "details": {},
            }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def get_cluster_analytics() -> ClusterAnalytics | None:
    """Get the singleton ClusterAnalytics instance.

    Returns:
        ClusterAnalytics instance or None if not created
    """
    return _cluster_analytics


def set_cluster_analytics(analytics: ClusterAnalytics | None) -> None:
    """Set the singleton ClusterAnalytics instance.

    Args:
        analytics: ClusterAnalytics instance or None to clear
    """
    global _cluster_analytics
    with _analytics_lock:
        _cluster_analytics = analytics


def create_cluster_analytics(
    config: ClusterAnalyticsConfig | None = None,
    get_gossip_states: Callable[[], dict[str, dict]] | None = None,
    get_peer_reputation: Callable[[], dict[str, dict]] | None = None,
    get_reputation_score: Callable[[str], float] | None = None,
    get_sync_intervals: Callable[[], dict[str, float]] | None = None,
    get_sync_streaks: Callable[[], tuple[dict[str, int], dict[str, int]]] | None = None,
    get_activity_factor: Callable[[], float] | None = None,
    get_dedup_stats: Callable[[], dict[str, Any]] | None = None,
    get_training_configs: Callable[[], dict[str, list[str]]] | None = None,
) -> ClusterAnalytics:
    """Create and register a ClusterAnalytics instance.

    Args:
        config: Analytics configuration
        get_gossip_states: Callback to get gossip peer states
        get_peer_reputation: Callback to get peer reputation data
        get_reputation_score: Callback to compute reputation score for a peer
        get_sync_intervals: Callback to get adaptive sync intervals
        get_sync_streaks: Callback to get (success_streaks, failure_streaks)
        get_activity_factor: Callback to get cluster activity factor
        get_dedup_stats: Callback to get deduplication stats
        get_training_configs: Callback to get active training configs by node

    Returns:
        ClusterAnalytics instance
    """
    analytics = ClusterAnalytics(
        config=config,
        get_gossip_states=get_gossip_states,
        get_peer_reputation=get_peer_reputation,
        get_reputation_score=get_reputation_score,
        get_sync_intervals=get_sync_intervals,
        get_sync_streaks=get_sync_streaks,
        get_activity_factor=get_activity_factor,
        get_dedup_stats=get_dedup_stats,
        get_training_configs=get_training_configs,
    )
    set_cluster_analytics(analytics)
    return analytics
