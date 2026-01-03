"""Manifest Collection Loop - Periodic manifest collection for dashboard/training/sync.

Created as part of Phase 4 of the P2P orchestrator decomposition (December 2025).

This loop periodically collects data manifests from cluster nodes (if leader)
or locally (if not leader). The manifests track available selfplay data,
training data, and model states across the cluster.

Usage:
    from scripts.p2p.loops import ManifestCollectionLoop

    loop = ManifestCollectionLoop(
        get_role=lambda: orchestrator.role,
        collect_cluster_manifest=lambda: orchestrator._collect_cluster_manifest(),
        collect_local_manifest=lambda: orchestrator._collect_local_data_manifest(),
        update_manifest=lambda m: orchestrator._update_manifest(m),
    )
    await loop.run_forever()
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from .base import BackoffConfig, BaseLoop
from .loop_constants import LoopIntervals

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Backward-compat aliases (Sprint 10: use LoopIntervals.* instead)
DEFAULT_COLLECTION_INTERVAL = LoopIntervals.MANIFEST_COLLECTION
INITIAL_DELAY = LoopIntervals.MANIFEST_INITIAL_DELAY


class NodeRole(Enum):
    """Node role enum (mirrors the one in p2p_orchestrator)."""
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    VOTER = "voter"


class ManifestCollectionLoop(BaseLoop):
    """Background loop for periodic manifest collection.

    This loop:
    1. Waits for initial delay before first collection (allows HTTP server to start)
    2. If leader: Collects cluster-wide manifest from all nodes
    3. If not leader: Collects local manifest only
    4. Updates the manifest storage via callback
    5. Optionally updates improvement cycle manager

    Attributes:
        _get_role: Callback to get current node role
        _collect_cluster_manifest: Async callback to collect cluster manifest
        _collect_local_manifest: Sync callback to collect local manifest
        _update_manifest: Callback to store the collected manifest
    """

    def __init__(
        self,
        get_role: Callable[[], NodeRole | str],
        collect_cluster_manifest: Callable[[], Coroutine[Any, Any, Any]],
        collect_local_manifest: Callable[[], Any],
        update_manifest: Callable[[Any, bool], None],
        *,
        update_improvement_cycle: Callable[[dict[str, Any]], None] | None = None,
        record_stats_sample: Callable[[Any], None] | None = None,
        interval: float = DEFAULT_COLLECTION_INTERVAL,
        initial_delay: float = INITIAL_DELAY,
        backoff_config: BackoffConfig | None = None,
        enabled: bool = True,
    ):
        """Initialize the manifest collection loop.

        Args:
            get_role: Callback that returns current node role (LEADER, FOLLOWER, etc.)
            collect_cluster_manifest: Async callback that collects cluster-wide manifest
            collect_local_manifest: Sync callback that collects local manifest
            update_manifest: Callback to store manifest; args: (manifest, is_cluster)
            update_improvement_cycle: Optional callback to update ImprovementCycleManager
            record_stats_sample: Optional callback to record stats sample for dashboard
            interval: Seconds between collection attempts (default: 300)
            initial_delay: Delay before first collection (default: 60s)
            backoff_config: Custom backoff configuration for errors
            enabled: Whether the loop is enabled
        """
        super().__init__(
            name="manifest_collection",
            interval=interval,
            backoff_config=backoff_config or BackoffConfig(
                initial_delay=10.0,
                max_delay=300.0,
                multiplier=2.0,
            ),
            enabled=enabled,
        )
        self._get_role = get_role
        self._collect_cluster_manifest = collect_cluster_manifest
        self._collect_local_manifest = collect_local_manifest
        self._update_manifest = update_manifest
        self._update_improvement_cycle = update_improvement_cycle
        self._record_stats_sample = record_stats_sample
        self._initial_delay = initial_delay

        # Track collection stats
        self._first_run = True
        self._total_collections = 0
        self._cluster_collections = 0
        self._local_collections = 0
        self._last_collection_time = 0.0

    async def _on_start(self) -> None:
        """Wait for initial delay before starting collections."""
        logger.info(
            f"[{self.name}] Starting (first collection in {self._initial_delay}s)"
        )
        await asyncio.sleep(self._initial_delay)
        self._first_run = False

    async def _run_once(self) -> None:
        """Execute one iteration of the manifest collection loop."""
        role = self._get_role()

        # Normalize role to string if it's an enum
        if hasattr(role, "value"):
            role_str = role.value
        else:
            role_str = str(role).lower()

        is_leader = role_str == "leader"
        self._total_collections += 1

        if is_leader:
            # Leader collects cluster-wide manifest
            manifest = await self._collect_cluster_manifest()
            self._update_manifest(manifest, True)  # is_cluster=True
            self._cluster_collections += 1

            # Record stats sample for dashboard
            if self._record_stats_sample is not None:
                try:
                    self._record_stats_sample(manifest)
                except Exception as e:
                    logger.debug(f"[{self.name}] Stats sample recording error: {e}")

            # Update improvement cycle manager
            if self._update_improvement_cycle is not None:
                try:
                    by_board = getattr(manifest, "by_board_type", {})
                    self._update_improvement_cycle(by_board)
                except Exception as e:
                    logger.debug(f"[{self.name}] ImprovementCycleManager update error: {e}")

            logger.debug(f"[{self.name}] Collected cluster manifest")
        else:
            # Non-leader collects local manifest only
            manifest = await asyncio.to_thread(self._collect_local_manifest)
            self._update_manifest(manifest, False)  # is_cluster=False
            self._local_collections += 1

            logger.debug(f"[{self.name}] Collected local manifest")

        self._last_collection_time = time.time()

    def get_status(self) -> dict[str, Any]:
        """Get extended loop status including collection statistics."""
        status = super().get_status()

        role = self._get_role()
        if hasattr(role, "value"):
            role_str = role.value
        else:
            role_str = str(role)

        status["collection_stats"] = {
            "total_collections": self._total_collections,
            "cluster_collections": self._cluster_collections,
            "local_collections": self._local_collections,
            "last_collection_time": self._last_collection_time,
        }
        status["role"] = role_str
        status["initial_delay"] = self._initial_delay
        status["first_run_completed"] = not self._first_run
        return status
