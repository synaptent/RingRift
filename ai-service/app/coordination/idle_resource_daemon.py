"""Idle Resource Daemon (December 2025 - Phase 20).

Monitors cluster for idle GPU resources and automatically spawns selfplay jobs
to maximize resource utilization.

Key features:
- Monitors GPU utilization across cluster nodes
- Spawns selfplay jobs on idle GPUs (>15min at <10% utilization)
- Matches board size to GPU memory capacity
- Queue-depth aware scaling (more aggressive when queue is deep)
- Integrates with FeedbackAccelerator for config prioritization

Usage:
    from app.coordination.idle_resource_daemon import IdleResourceDaemon

    daemon = IdleResourceDaemon()
    await daemon.start()
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.coordination.protocols import (
    CoordinatorStatus,
    register_coordinator,
    unregister_coordinator,
)
from app.core.async_context import safe_create_task

logger = logging.getLogger(__name__)

# Job scheduler integration (Phase 21.2 - Dec 2025)
try:
    from app.coordination.job_scheduler import (
        JobPriority,
        PriorityJobScheduler,
        ScheduledJob,
        get_scheduler,
    )
    HAS_JOB_SCHEDULER = True
except ImportError:
    HAS_JOB_SCHEDULER = False
    get_scheduler = None
    PriorityJobScheduler = None
    ScheduledJob = None
    JobPriority = None


@dataclass
class IdleResourceConfig:
    """Configuration for idle resource monitoring."""
    enabled: bool = True
    # Reduced from 300s (5min) to 60s for faster detection (Dec 2025)
    check_interval_seconds: int = 60  # 1 minute (was 5 minutes)
    idle_threshold_percent: float = 10.0  # <10% GPU utilization
    # Reduced from 900s (15min) to 120s (2min) for faster spawning (Dec 2025)
    idle_duration_seconds: int = 120  # 2 minutes before spawning (was 15 minutes)
    # Base max concurrent spawns (scaled dynamically based on idle nodes)
    max_concurrent_spawns: int = 4
    # Maximum absolute cap for spawns (Phase 2B.2 - December 2025)
    max_spawns_cap: int = 40
    # Board size to GPU memory mapping (GB)
    gpu_memory_thresholds: dict[str, int] = field(default_factory=lambda: {
        "hexagonal_4p": 80,   # Largest board, 4 players
        "hexagonal_2p": 48,
        "square19_4p": 48,
        "square19_2p": 24,
        "square8_4p": 12,
        "hex8_4p": 12,
        "square8_2p": 8,
        "hex8_2p": 8,
    })
    # Default games per spawn
    default_games_per_spawn: int = 100
    # Queue depth thresholds for scaling
    high_queue_depth: int = 20
    medium_queue_depth: int = 10

    @classmethod
    def from_env(cls) -> IdleResourceConfig:
        """Load configuration from environment variables."""
        config = cls()
        config.enabled = os.environ.get("RINGRIFT_IDLE_RESOURCE_ENABLED", "1") == "1"
        # Faster detection: reduced from 300s to 60s (Dec 2025)
        config.check_interval_seconds = int(
            os.environ.get("RINGRIFT_IDLE_CHECK_INTERVAL", "60")
        )
        config.idle_threshold_percent = float(
            os.environ.get("RINGRIFT_IDLE_THRESHOLD", "10.0")
        )
        # Faster spawning: reduced from 900s to 120s (Dec 2025)
        config.idle_duration_seconds = int(
            os.environ.get("RINGRIFT_IDLE_DURATION", "120")
        )
        return config


@dataclass
class NodeStatus:
    """Status of a cluster node."""
    node_id: str
    host: str
    gpu_utilization: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_memory_used_gb: float = 0.0
    last_seen: float = 0.0
    idle_since: float = 0.0  # When node became idle
    active_jobs: int = 0
    provider: str = "unknown"


@dataclass
class SpawnStats:
    """Statistics for spawn operations."""
    total_spawns: int = 0
    successful_spawns: int = 0
    failed_spawns: int = 0
    games_spawned: int = 0
    last_spawn_time: float = 0.0
    last_error: str | None = None


class IdleResourceDaemon:
    """Daemon that monitors idle resources and spawns selfplay jobs.

    Continuously scans cluster for underutilized GPUs and automatically
    spawns selfplay jobs to maximize resource usage.
    """

    def __init__(self, config: IdleResourceConfig | None = None):
        self.config = config or IdleResourceConfig.from_env()
        self.node_id = socket.gethostname()
        self._running = False
        self._stats = SpawnStats()
        self._monitor_task: asyncio.Task | None = None
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_spawns)

        # Track node states
        self._node_states: dict[str, NodeStatus] = {}

        # CoordinatorProtocol state
        self._coordinator_status = CoordinatorStatus.INITIALIZING
        self._start_time: float = 0.0
        self._events_processed: int = 0
        self._errors_count: int = 0
        self._last_error: str = ""

        logger.info(
            f"IdleResourceDaemon initialized: node={self.node_id}, "
            f"interval={self.config.check_interval_seconds}s, "
            f"idle_threshold={self.config.idle_threshold_percent}%"
        )

    def _get_dynamic_max_spawns(self) -> int:
        """Calculate max concurrent spawns based on idle node count.

        December 2025 - Phase 2B.2: Scale job spawning proportionally
        to idle capacity instead of fixed 4.

        Returns:
            Max spawns: base * (idle_nodes / 4), capped at max_spawns_cap
        """
        # Count idle nodes
        idle_nodes = sum(
            1 for node in self._node_states.values()
            if node.gpu_utilization < self.config.idle_threshold_percent
            and node.idle_since > 0
        )

        if idle_nodes <= 0:
            return self.config.max_concurrent_spawns

        # Scale: 2 spawns per idle node, minimum of base (4), max of cap (40)
        scaled = max(
            self.config.max_concurrent_spawns,
            min(idle_nodes * 2, self.config.max_spawns_cap)
        )

        return scaled

    def is_running(self) -> bool:
        """Check if the daemon is running."""
        return self._running

    async def start(self) -> None:
        """Start the idle resource daemon."""
        if not self.config.enabled:
            self._coordinator_status = CoordinatorStatus.STOPPED
            logger.info("IdleResourceDaemon disabled by config")
            return

        if self._coordinator_status == CoordinatorStatus.RUNNING:
            return  # Already running

        self._running = True
        self._coordinator_status = CoordinatorStatus.RUNNING
        self._start_time = time.time()
        logger.info(f"Starting IdleResourceDaemon on {self.node_id}")

        # Register with coordinator protocol
        try:
            register_coordinator("idle_resource", self)
        except Exception as e:
            logger.debug(f"Failed to register coordinator: {e}")

        # Start monitoring loop
        self._monitor_task = safe_create_task(
            self._monitor_loop(),
            name="idle_resource_monitor"
        )

    async def stop(self) -> None:
        """Stop the idle resource daemon."""
        if self._coordinator_status == CoordinatorStatus.STOPPED:
            return  # Already stopped

        self._coordinator_status = CoordinatorStatus.STOPPING
        logger.info("Stopping IdleResourceDaemon...")
        self._running = False

        # Stop monitor task
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        # Unregister coordinator
        try:
            unregister_coordinator("idle_resource")
        except Exception:
            pass

        self._coordinator_status = CoordinatorStatus.STOPPED
        logger.info(
            f"IdleResourceDaemon stopped. Stats: "
            f"{self._stats.successful_spawns}/{self._stats.total_spawns} spawns, "
            f"{self._stats.games_spawned} games"
        )

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_and_spawn()
                await asyncio.sleep(self.config.check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._errors_count += 1
                self._last_error = str(e)
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(60)  # Back off on error

    async def _check_and_spawn(self) -> None:
        """Check for idle nodes and spawn selfplay jobs.

        December 2025 - Phase 2B.2: Dynamically scales spawning based on
        idle node count instead of fixed max of 4.
        """
        try:
            # Get cluster status
            nodes = await self._get_cluster_nodes()

            if not nodes:
                logger.debug("No cluster nodes found")
                return

            # Get queue depth for scaling decisions
            queue_depth = await self._get_queue_depth()

            # Get dynamic max spawns based on current idle capacity
            max_spawns = self._get_dynamic_max_spawns()

            # Collect nodes that need spawning
            spawn_candidates = [
                node for node in nodes
                if self._should_spawn(node, queue_depth)
            ]

            if not spawn_candidates:
                return

            # Log scaling decision
            logger.info(
                f"[IdleResourceDaemon] Spawn check: {len(spawn_candidates)} candidates, "
                f"max_spawns={max_spawns} (dynamic), queue_depth={queue_depth}"
            )

            # Spawn up to max_spawns jobs concurrently
            spawn_tasks = []
            for node in spawn_candidates[:max_spawns]:
                spawn_tasks.append(self._spawn_selfplay(node))

            if spawn_tasks:
                results = await asyncio.gather(*spawn_tasks, return_exceptions=True)
                successful = sum(1 for r in results if r is True)
                logger.info(
                    f"[IdleResourceDaemon] Spawned {successful}/{len(spawn_tasks)} jobs"
                )

        except Exception as e:
            logger.warning(f"Check and spawn error: {e}")

    async def _get_cluster_nodes(self) -> list[NodeStatus]:
        """Get status of all cluster nodes."""
        nodes: list[NodeStatus] = []

        try:
            from app.distributed.p2p_orchestrator import get_p2p_orchestrator

            p2p = get_p2p_orchestrator()
            if p2p is None:
                return nodes

            # Get cluster status
            status = await p2p.get_status()
            if not status:
                return nodes

            # Parse alive peers
            alive_peers = status.get("alive_peers", [])
            if isinstance(alive_peers, int):
                # Just a count, not detailed info
                return nodes

            for peer_info in alive_peers:
                if isinstance(peer_info, dict):
                    node = NodeStatus(
                        node_id=peer_info.get("node_id", ""),
                        host=peer_info.get("host", ""),
                        gpu_utilization=peer_info.get("gpu_utilization", 0.0),
                        gpu_memory_total_gb=peer_info.get("gpu_memory_total", 0.0),
                        gpu_memory_used_gb=peer_info.get("gpu_memory_used", 0.0),
                        last_seen=peer_info.get("last_seen", time.time()),
                        active_jobs=peer_info.get("active_jobs", 0),
                        provider=peer_info.get("provider", "unknown"),
                    )
                    nodes.append(node)
                    self._update_node_state(node)

        except ImportError:
            logger.debug("P2P orchestrator not available")
        except Exception as e:
            logger.debug(f"Failed to get cluster nodes: {e}")

        return nodes

    def _update_node_state(self, node: NodeStatus) -> None:
        """Update tracked node state for idle duration tracking."""
        now = time.time()
        existing = self._node_states.get(node.node_id)

        if existing:
            # Check if node transitioned to idle
            if node.gpu_utilization < self.config.idle_threshold_percent:
                if existing.gpu_utilization >= self.config.idle_threshold_percent:
                    # Just became idle
                    node.idle_since = now
                else:
                    # Still idle, preserve idle_since
                    node.idle_since = existing.idle_since
            else:
                # Not idle
                node.idle_since = 0.0
        else:
            # New node
            if node.gpu_utilization < self.config.idle_threshold_percent:
                node.idle_since = now

        self._node_states[node.node_id] = node

    async def _get_queue_depth(self) -> int:
        """Get current queue depth for scaling decisions."""
        try:
            from app.coordination.job_scheduler import get_job_scheduler

            scheduler = get_job_scheduler()
            if scheduler:
                return scheduler.get_queue_depth()
        except ImportError:
            pass
        except Exception:
            pass

        return 0  # Default to no queue

    def _should_spawn(self, node: NodeStatus, queue_depth: int) -> bool:
        """Decide whether to spawn selfplay on a node."""
        now = time.time()

        # Check if node is idle long enough
        if node.idle_since <= 0:
            return False

        idle_duration = now - node.idle_since

        # Adjust threshold based on queue depth
        if queue_depth > self.config.high_queue_depth:
            # More aggressive spawning when queue is deep
            threshold = self.config.idle_threshold_percent * 3  # 30%
            required_idle_time = self.config.idle_duration_seconds / 3  # 5 minutes
        elif queue_depth > self.config.medium_queue_depth:
            threshold = self.config.idle_threshold_percent * 2  # 20%
            required_idle_time = self.config.idle_duration_seconds / 2  # 7.5 minutes
        else:
            threshold = self.config.idle_threshold_percent  # 10%
            required_idle_time = self.config.idle_duration_seconds  # 15 minutes

        # Check conditions
        if node.gpu_utilization > threshold:
            return False

        if idle_duration < required_idle_time:
            return False

        # Don't spawn if node already has active jobs
        if node.active_jobs > 0:
            return False

        return True

    def _select_config_for_gpu(self, gpu_memory_gb: float) -> str:
        """Select appropriate board config for GPU memory."""
        # Sort by memory requirement descending
        sorted_configs = sorted(
            self.config.gpu_memory_thresholds.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for config_key, required_memory in sorted_configs:
            if gpu_memory_gb >= required_memory:
                return config_key

        # Default to smallest config
        return "hex8_2p"

    async def _spawn_selfplay(self, node: NodeStatus) -> bool:
        """Spawn a selfplay job on the given node."""
        async with self._semaphore:
            self._stats.total_spawns += 1

            try:
                config_key = self._select_config_for_gpu(node.gpu_memory_total_gb)

                # Get multiplier from FeedbackAccelerator
                games = self.config.default_games_per_spawn
                try:
                    from app.training.feedback_accelerator import get_selfplay_multiplier
                    multiplier = get_selfplay_multiplier(config_key)
                    games = int(games * multiplier)
                except ImportError:
                    pass

                logger.info(
                    f"[IdleResourceDaemon] Spawning selfplay on {node.node_id}: "
                    f"config={config_key}, games={games}, "
                    f"gpu_memory={node.gpu_memory_total_gb:.0f}GB"
                )

                # Phase 21.2: Also schedule via PriorityJobScheduler for tracking
                if HAS_JOB_SCHEDULER and get_scheduler:
                    try:
                        scheduler = get_scheduler()
                        if scheduler:
                            # Parse config key for job config
                            parts = config_key.rsplit("_", 1)
                            board_type = parts[0] if parts else config_key
                            num_players = int(parts[1].replace("p", "")) if len(parts) > 1 else 2

                            job = ScheduledJob(
                                job_type="selfplay",
                                priority=JobPriority.NORMAL,  # Idle-spawned jobs are normal priority
                                config={
                                    "board_type": board_type,
                                    "num_players": num_players,
                                    "games": games,
                                    "config_key": config_key,
                                    "source": "idle_resource_daemon",
                                },
                                host_preference=node.node_id,
                                requires_gpu=True,
                                estimated_duration_seconds=games * 10.0,  # ~10s per game estimate
                            )
                            scheduler.schedule(job)
                            logger.debug(
                                f"[IdleResourceDaemon] Scheduled job via JobScheduler: {config_key}"
                            )
                    except Exception as e:
                        logger.debug(f"[IdleResourceDaemon] JobScheduler integration failed: {e}")

                # Spawn via P2P job distribution
                success = await self._distribute_job(node, config_key, games)

                if success:
                    self._stats.successful_spawns += 1
                    self._stats.games_spawned += games
                    self._stats.last_spawn_time = time.time()

                    # Emit event
                    self._emit_spawn_event(node, config_key, games)

                    # Reset idle tracking for this node
                    if node.node_id in self._node_states:
                        self._node_states[node.node_id].idle_since = 0.0

                    return True
                else:
                    self._stats.failed_spawns += 1
                    return False

            except Exception as e:
                self._stats.failed_spawns += 1
                self._stats.last_error = str(e)
                logger.error(f"Failed to spawn selfplay on {node.node_id}: {e}")
                return False

    async def _distribute_job(
        self,
        node: NodeStatus,
        config_key: str,
        games: int,
    ) -> bool:
        """Distribute a selfplay job to a node."""
        try:
            from app.distributed.p2p_orchestrator import get_p2p_orchestrator

            p2p = get_p2p_orchestrator()
            if p2p is None:
                return False

            # Parse config key
            parts = config_key.rsplit("_", 1)
            if len(parts) != 2:
                logger.warning(f"Invalid config key: {config_key}")
                return False

            board_type = parts[0]
            num_players = int(parts[1].replace("p", ""))

            # Submit job via P2P
            job_spec = {
                "type": "selfplay",
                "board_type": board_type,
                "num_players": num_players,
                "num_games": games,
                "engine": "gumbel",
                "target_node": node.node_id,
            }

            result = await p2p.submit_job(job_spec)
            return result.get("success", False)

        except ImportError:
            logger.debug("P2P orchestrator not available for job distribution")
            return False
        except Exception as e:
            logger.warning(f"Job distribution failed: {e}")
            return False

    def _emit_spawn_event(
        self,
        node: NodeStatus,
        config_key: str,
        games: int,
    ) -> None:
        """Emit event for selfplay spawn."""
        try:
            from app.coordination.event_router import get_router, DataEventType

            router = get_router()
            # Phase 22.2 fix: Use publish_sync instead of emit (which doesn't exist)
            router.publish_sync(
                DataEventType.P2P_SELFPLAY_SCALED.value
                if hasattr(DataEventType, 'P2P_SELFPLAY_SCALED')
                else "p2p_selfplay_scaled",
                {
                    "node_id": node.node_id,
                    "config": config_key,
                    "games": games,
                    "reason": "idle_resource",
                    "gpu_memory_gb": node.gpu_memory_total_gb,
                    "timestamp": time.time(),
                },
                source="idle_resource_daemon",
            )
        except Exception as e:
            logger.debug(f"Could not publish spawn event: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get daemon statistics."""
        return {
            "running": self._running,
            "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
            "total_spawns": self._stats.total_spawns,
            "successful_spawns": self._stats.successful_spawns,
            "failed_spawns": self._stats.failed_spawns,
            "games_spawned": self._stats.games_spawned,
            "last_spawn_time": self._stats.last_spawn_time,
            "last_error": self._stats.last_error,
            "tracked_nodes": len(self._node_states),
            "errors_count": self._errors_count,
        }

    # CoordinatorProtocol methods
    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        return {
            "healthy": self._running,
            "status": self._coordinator_status.value,
            "stats": self.get_stats(),
        }

    def get_status(self) -> CoordinatorStatus:
        """Get coordinator status."""
        return self._coordinator_status
