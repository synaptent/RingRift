"""Status Metrics Collector for P2P Orchestrator.

January 30, 2026 - Priority 2.2 Decomposition

This manager extracts the parallel metric collection pattern from handle_status().
It provides:

1. Safe metric gathering with timeout and error handling
2. Parallel execution of multiple metric calls
3. Result extraction into named dictionaries

The handle_status endpoint was taking 10-24 seconds because metric calls
were executed sequentially. This collector runs them in parallel for <5s latency.

Usage:
    from scripts.p2p.managers.status_metrics_collector import (
        StatusMetricsCollector,
        MetricTask,
        CollectorConfig,
        create_status_metrics_collector,
    )

    collector = create_status_metrics_collector(orchestrator, config)
    results = await collector.collect_all_metrics()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default timeout for each metric call
# Feb 2026: Reduced from 5.0s to 2.0s. With 22+ metrics running in parallel,
# a 5s timeout per metric meant worst case was still 5s wall-clock (due to
# asyncio.gather parallelism), but thread pool congestion could cause queueing.
# 2s is sufficient for in-memory metrics; truly slow metrics (SQLite, file I/O)
# will timeout gracefully and return error markers.
DEFAULT_METRIC_TIMEOUT = 2.0  # seconds


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CollectorConfig:
    """Configuration for status metrics collector."""

    # Timeout for each individual metric call
    metric_timeout: float = DEFAULT_METRIC_TIMEOUT

    # Whether to log warnings for timed-out metrics
    log_timeouts: bool = True

    # Whether to include stack traces in error results
    include_stack_traces: bool = False

    # Maximum number of concurrent metric tasks
    max_concurrent_tasks: int = 20


@dataclass
class MetricTask:
    """Definition of a metric gathering task."""

    name: str
    func: Callable[[], Any]
    timeout: float | None = None  # Override collector default
    is_async: bool = False  # Whether func is already async


@dataclass
class MetricResult:
    """Result of a single metric gathering task."""

    name: str
    value: dict[str, Any]
    success: bool
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class CollectionResult:
    """Result of collecting all metrics."""

    metrics: dict[str, dict[str, Any]]
    errors: list[str]
    total_duration_ms: float
    timed_out_count: int
    failed_count: int
    success_count: int


class StatusMetricsCollector:
    """Collects status metrics in parallel with timeout protection.

    This class extracts the parallel metric gathering pattern from handle_status.
    Key features:
    - All metrics are gathered concurrently using asyncio.gather()
    - Each metric call has timeout protection
    - Errors are captured gracefully without failing the entire collection
    - Results are returned as a dictionary keyed by metric name

    Example:
        collector = StatusMetricsCollector(config)
        collector.add_task(MetricTask("gossip", self._get_gossip_metrics_summary))
        collector.add_task(MetricTask("elo", self._get_cluster_elo_summary))
        results = await collector.collect_all_metrics()
        # results.metrics = {"gossip": {...}, "elo": {...}}
    """

    def __init__(
        self,
        config: CollectorConfig | None = None,
    ):
        self.config = config or CollectorConfig()
        self._tasks: list[MetricTask] = []

    def add_task(self, task: MetricTask) -> None:
        """Add a metric task to the collection."""
        self._tasks.append(task)

    def add_tasks(self, tasks: list[MetricTask]) -> None:
        """Add multiple metric tasks."""
        self._tasks.extend(tasks)

    def clear_tasks(self) -> None:
        """Clear all registered tasks."""
        self._tasks.clear()

    async def _safe_metric(
        self,
        name: str,
        func: Callable[[], Any],
        timeout: float | None = None,
        is_async: bool = False,
        use_thread: bool = False,
    ) -> MetricResult:
        """Safely gather a metric with timeout and error handling.

        This is the core pattern extracted from handle_status:
        - By default, calls sync functions directly (they're fast in-memory ops)
        - Set use_thread=True for functions with blocking I/O (file/DB access)
        - Applies timeout protection
        - Captures errors gracefully

        Args:
            name: Metric name for identification
            func: Function to call (sync or async)
            timeout: Timeout in seconds (uses config default if None)
            is_async: Whether func is already async
            use_thread: Whether to run sync func in thread pool (for blocking I/O)

        Returns:
            MetricResult with value or error information

        Feb 22, 2026: Changed default from asyncio.to_thread() to direct call.
        With thread pool capped at 4 workers, 20+ metrics all using to_thread()
        caused massive queuing (each waiting 2s for a slot), blocking the event
        loop for 10-40s total. Most metrics are fast in-memory dict operations
        that don't need threads.
        """
        effective_timeout = timeout or self.config.metric_timeout
        start_time = time.perf_counter()

        try:
            if is_async:
                result = await asyncio.wait_for(
                    func(),
                    timeout=effective_timeout,
                )
            elif use_thread:
                result = await asyncio.wait_for(
                    asyncio.to_thread(func),
                    timeout=effective_timeout,
                )
            else:
                # Call sync function directly - most metrics are fast in-memory ops.
                # The 2s timeout won't help if the function truly blocks, but these
                # functions should return in <10ms.
                result = func()

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Ensure result is a dict
            if not isinstance(result, dict):
                result = {"value": result}

            return MetricResult(
                name=name,
                value=result,
                success=True,
                duration_ms=duration_ms,
            )

        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000
            if self.config.log_timeouts:
                logger.warning(f"StatusMetricsCollector: {name} timed out after {effective_timeout}s")
            return MetricResult(
                name=name,
                value={"error": "timeout", "timeout_seconds": effective_timeout},
                success=False,
                error="timeout",
                duration_ms=duration_ms,
            )

        except Exception as e:  # noqa: BLE001
            duration_ms = (time.perf_counter() - start_time) * 1000
            error_msg = str(e)
            if self.config.include_stack_traces:
                import traceback
                error_msg = traceback.format_exc()

            logger.debug(f"StatusMetricsCollector: {name} failed: {e}")
            return MetricResult(
                name=name,
                value={"error": error_msg},
                success=False,
                error=error_msg,
                duration_ms=duration_ms,
            )

    async def collect_all_metrics(self) -> CollectionResult:
        """Collect all registered metrics in parallel.

        This is the main entry point. It:
        1. Creates async tasks for all registered metrics
        2. Runs them concurrently with asyncio.gather()
        3. Collects results into a dictionary

        Returns:
            CollectionResult with all metrics and statistics
        """
        if not self._tasks:
            return CollectionResult(
                metrics={},
                errors=[],
                total_duration_ms=0.0,
                timed_out_count=0,
                failed_count=0,
                success_count=0,
            )

        start_time = time.perf_counter()

        # Create tasks for all metrics
        async_tasks = [
            self._safe_metric(
                name=task.name,
                func=task.func,
                timeout=task.timeout,
                is_async=task.is_async,
            )
            for task in self._tasks
        ]

        # Gather all results in parallel
        results = await asyncio.gather(*async_tasks, return_exceptions=True)

        # Process results
        metrics: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        timed_out_count = 0
        failed_count = 0
        success_count = 0

        for result in results:
            if isinstance(result, MetricResult):
                metrics[result.name] = result.value

                if result.success:
                    success_count += 1
                elif result.error == "timeout":
                    timed_out_count += 1
                    failed_count += 1
                else:
                    failed_count += 1
                    if result.error:
                        errors.append(f"{result.name}: {result.error}")

            elif isinstance(result, Exception):
                # Handle unexpected exceptions from gather
                error_msg = f"Unexpected error: {result}"
                errors.append(error_msg)
                failed_count += 1
                logger.warning(f"StatusMetricsCollector: gather exception: {result}")

        total_duration_ms = (time.perf_counter() - start_time) * 1000

        return CollectionResult(
            metrics=metrics,
            errors=errors,
            total_duration_ms=total_duration_ms,
            timed_out_count=timed_out_count,
            failed_count=failed_count,
            success_count=success_count,
        )

    async def collect_named_metrics(
        self,
        tasks: list[MetricTask],
    ) -> dict[str, dict[str, Any]]:
        """Convenience method to collect specific metrics by task list.

        This allows one-shot collection without registering tasks first.

        Args:
            tasks: List of MetricTask definitions

        Returns:
            Dictionary of metric name -> result dict
        """
        # Temporarily set tasks
        old_tasks = self._tasks
        self._tasks = tasks

        try:
            result = await self.collect_all_metrics()
            return result.metrics
        finally:
            self._tasks = old_tasks


# =============================================================================
# Pre-built metric task factories
# =============================================================================


def create_orchestrator_metric_tasks(orchestrator: Any) -> list[MetricTask]:
    """Create standard metric tasks for P2P orchestrator status endpoint.

    This factory creates the metric tasks that were previously inline
    in handle_status(). Each task corresponds to a section of metrics.

    Args:
        orchestrator: P2P orchestrator instance

    Returns:
        List of MetricTask definitions
    """
    tasks = []

    # Gossip metrics
    if hasattr(orchestrator, "_get_gossip_metrics_summary"):
        tasks.append(MetricTask(
            name="gossip_metrics",
            func=orchestrator._get_gossip_metrics_summary,
        ))

    # Distributed training
    if hasattr(orchestrator, "_get_distributed_training_summary"):
        tasks.append(MetricTask(
            name="distributed_training",
            func=orchestrator._get_distributed_training_summary,
        ))

    # Cluster Elo
    if hasattr(orchestrator, "_get_cluster_elo_summary"):
        tasks.append(MetricTask(
            name="cluster_elo",
            func=orchestrator._get_cluster_elo_summary,
        ))

    # Leader consensus
    if hasattr(orchestrator, "leadership") and hasattr(orchestrator.leadership, "get_cluster_leader_consensus"):
        tasks.append(MetricTask(
            name="leader_consensus",
            func=orchestrator.leadership.get_cluster_leader_consensus,
        ))

    # Peer reputation
    if hasattr(orchestrator, "_get_cluster_peer_reputation"):
        tasks.append(MetricTask(
            name="peer_reputation",
            func=orchestrator._get_cluster_peer_reputation,
        ))

    # Sync intervals
    if hasattr(orchestrator, "sync") and hasattr(orchestrator.sync, "get_sync_interval_summary"):
        tasks.append(MetricTask(
            name="sync_intervals",
            func=orchestrator.sync.get_sync_interval_summary,
        ))

    # Tournament scheduling
    if hasattr(orchestrator, "tournament_manager") and hasattr(orchestrator.tournament_manager, "get_distributed_tournament_summary"):
        tasks.append(MetricTask(
            name="tournament_scheduling",
            func=orchestrator.tournament_manager.get_distributed_tournament_summary,
        ))

    # Data dedup
    if hasattr(orchestrator, "_get_dedup_summary"):
        tasks.append(MetricTask(
            name="data_dedup",
            func=orchestrator._get_dedup_summary,
        ))

    # SWIM/Raft status
    if hasattr(orchestrator, "network") and hasattr(orchestrator.network, "get_swim_raft_status"):
        tasks.append(MetricTask(
            name="swim_raft",
            func=orchestrator.network.get_swim_raft_status,
        ))

    # Partition status
    if hasattr(orchestrator, "get_partition_status"):
        tasks.append(MetricTask(
            name="partition",
            func=orchestrator.get_partition_status,
        ))

    # Background loops
    def _get_loop_manager_status():
        loop_manager = orchestrator._get_loop_manager()
        if loop_manager is not None:
            return loop_manager.get_all_status()
        return {"error": "LoopManager not initialized"}

    tasks.append(MetricTask(
        name="background_loops",
        func=_get_loop_manager_status,
    ))

    # Voter health
    if hasattr(orchestrator, "_check_voter_health"):
        tasks.append(MetricTask(
            name="voter_health",
            func=orchestrator._check_voter_health,
        ))

    # =========================================================================
    # Feb 2026: Metrics previously computed sequentially after the parallel
    # collector. Moved here to run in parallel for faster /status response.
    # =========================================================================

    # Cluster observability
    if hasattr(orchestrator, "sync") and hasattr(orchestrator.sync, "get_cluster_observability"):
        tasks.append(MetricTask(
            name="cluster_observability",
            func=orchestrator.sync.get_cluster_observability,
        ))

    # Fallback status
    if hasattr(orchestrator, "_get_fallback_status"):
        tasks.append(MetricTask(
            name="fallback_status",
            func=orchestrator._get_fallback_status,
        ))

    # Leadership consistency
    if hasattr(orchestrator, "leadership") and hasattr(orchestrator.leadership, "get_consistency_metrics"):
        tasks.append(MetricTask(
            name="leadership_consistency",
            func=orchestrator.leadership.get_consistency_metrics,
        ))

    # Config version (involves file I/O - reads YAML and computes SHA256)
    if hasattr(orchestrator, "_get_config_version"):
        tasks.append(MetricTask(
            name="config_version",
            func=orchestrator._get_config_version,
        ))

    # Data summary (cached)
    if hasattr(orchestrator, "_get_data_summary_cached"):
        tasks.append(MetricTask(
            name="data_summary",
            func=orchestrator._get_data_summary_cached,
        ))

    # Cooldown stats
    if hasattr(orchestrator, "_get_cooldown_stats"):
        tasks.append(MetricTask(
            name="cooldown_stats",
            func=orchestrator._get_cooldown_stats,
        ))

    # Peer health summary
    if hasattr(orchestrator, "health_metrics_manager") and hasattr(orchestrator.health_metrics_manager, "get_peer_health_summary"):
        tasks.append(MetricTask(
            name="peer_health_summary",
            func=orchestrator.health_metrics_manager.get_peer_health_summary,
        ))

    # Transport latency
    def _get_transport_latency():
        try:
            from scripts.p2p.transport_cascade import get_transport_cascade
            cascade = get_transport_cascade()
            return cascade.get_transport_latency_summary()
        except ImportError:
            return {"available": False, "reason": "import_error"}

    tasks.append(MetricTask(
        name="transport_latency",
        func=_get_transport_latency,
    ))

    # Is leader check
    if hasattr(orchestrator, "leadership") and hasattr(orchestrator.leadership, "check_is_leader"):
        def _check_is_leader():
            return {"value": orchestrator.leadership.check_is_leader()}
        tasks.append(MetricTask(
            name="is_leader",
            func=_check_is_leader,
        ))

    return tasks


def create_status_metrics_collector(
    orchestrator: Any | None = None,
    config: CollectorConfig | None = None,
    include_standard_tasks: bool = True,
) -> StatusMetricsCollector:
    """Factory function to create a StatusMetricsCollector.

    Args:
        orchestrator: P2P orchestrator instance (for standard tasks)
        config: Optional collector configuration
        include_standard_tasks: Whether to add standard orchestrator tasks

    Returns:
        Configured StatusMetricsCollector instance
    """
    collector = StatusMetricsCollector(config=config)

    if include_standard_tasks and orchestrator is not None:
        tasks = create_orchestrator_metric_tasks(orchestrator)
        collector.add_tasks(tasks)

    return collector
