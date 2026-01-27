"""NodeTargetingMixin - Per-node job calculation for SelfplayScheduler.

January 2026 Sprint 17.9: Extracted from selfplay_scheduler.py (~250 LOC)
to reduce main file size from 3,429 LOC toward ~1,800 LOC target.

This mixin provides:
- Target job calculation based on hardware capabilities
- Node selfplay eligibility checks
- Hardware-based concurrency limits
- Backpressure and resource-aware scaling

Usage:
    class SelfplayScheduler(NodeTargetingMixin, ...):
        pass

The mixin expects the following attributes on the class:
- _is_emergency_active_fn: Callable[[], bool] | None
- _should_stop_production_fn: Callable[..., bool] | None
- _should_throttle_production_fn: Callable[..., bool] | None
- _get_throttle_factor_fn: Callable[..., float] | None
- _record_utilization_fn: Callable[..., None] | None
- _get_host_targets_fn: Callable[[str], Any] | None
- _get_target_job_count_fn: Callable[..., int] | None
- _should_scale_up_fn: Callable[..., tuple[bool, str]] | None
- _should_scale_down_fn: Callable[..., tuple[bool, int, str]] | None
- _get_max_selfplay_for_node_fn: Callable[..., int] | None
- _node_capabilities: dict[str, NodeCapability]
- _verbose: bool
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

from app.config.cluster_config import get_cluster_nodes
from app.config.coordination_defaults import SelfplayAllocationDefaults

if TYPE_CHECKING:
    from app.coordination.node_allocator import NodeCapability

logger = logging.getLogger(__name__)

# Resource management thresholds (from SelfplayAllocationDefaults)
MIN_MEMORY_GB_FOR_TASKS = SelfplayAllocationDefaults.MIN_MEMORY_GB
DISK_WARNING_THRESHOLD = SelfplayAllocationDefaults.DISK_WARNING_THRESHOLD
MEMORY_WARNING_THRESHOLD = SelfplayAllocationDefaults.MEMORY_WARNING_THRESHOLD


class NodeTargetingMixin:
    """Mixin providing per-node job targeting methods.

    This mixin extracts node targeting responsibilities from SelfplayScheduler:
    - Target job calculation based on hardware
    - Selfplay eligibility checks
    - Hardware-based concurrency limits
    - Backpressure and resource-aware scaling

    Attributes expected from main class:
        _is_emergency_active_fn: Emergency check callback
        _should_stop_production_fn: Backpressure stop callback
        _should_throttle_production_fn: Backpressure throttle callback
        _get_throttle_factor_fn: Throttle factor callback
        _record_utilization_fn: Utilization recording callback
        _get_host_targets_fn: Host targets callback
        _get_target_job_count_fn: Target job count callback
        _should_scale_up_fn: Scale-up decision callback
        _should_scale_down_fn: Scale-down decision callback
        _get_max_selfplay_for_node_fn: Max selfplay limit callback
        _node_capabilities: Dict of node capabilities
        _verbose: Verbosity flag
    """

    # Type hints for attributes provided by SelfplayScheduler
    _is_emergency_active_fn: Optional[Callable[[], bool]]
    _should_stop_production_fn: Optional[Callable[..., bool]]
    _should_throttle_production_fn: Optional[Callable[..., bool]]
    _get_throttle_factor_fn: Optional[Callable[..., float]]
    _record_utilization_fn: Optional[Callable[..., None]]
    _get_host_targets_fn: Optional[Callable[[str], Any]]
    _get_target_job_count_fn: Optional[Callable[..., int]]
    _should_scale_up_fn: Optional[Callable[..., tuple[bool, str]]]
    _should_scale_down_fn: Optional[Callable[..., tuple[bool, int, str]]]
    _get_max_selfplay_for_node_fn: Optional[Callable[..., int]]
    _node_capabilities: dict[str, "NodeCapability"]
    _verbose: bool

    # =========================================================================
    # Node Eligibility Checks
    # =========================================================================

    def _is_selfplay_enabled(self, node_id: str) -> bool:
        """Check if selfplay is enabled for a node.

        January 2026: Added to support training-only nodes that should not
        receive selfplay jobs (prevents OOM from training + selfplay conflicts).

        Args:
            node_id: The node ID to check.

        Returns:
            True if selfplay is allowed on this node, False otherwise.
        """
        try:
            cluster_nodes = get_cluster_nodes()
            if node_id in cluster_nodes:
                return cluster_nodes[node_id].selfplay_enabled
            # If node not in config, default to enabled
            return True
        except Exception as e:
            # Graceful fallback: if config unavailable, allow selfplay
            logger.debug(f"[SelfplayScheduler] Could not check selfplay_enabled for {node_id}: {e}")
            return True

    def _is_cpu_only_node(self, node_id: str) -> bool:
        """Check if a node is CPU-only (no GPU).

        Jan 5, 2026: CPU-only nodes (like Hetzner) can contribute
        heuristic-only selfplay data. This helper identifies them
        for appropriate work assignment.

        Args:
            node_id: Node identifier to check

        Returns:
            True if node has no GPU, False otherwise
        """
        cap = self._node_capabilities.get(node_id)
        if not cap:
            return False
        return cap.gpu_memory_gb == 0

    # =========================================================================
    # Per-Node Job Targeting (Dec 2025)
    # =========================================================================

    def get_target_jobs_for_node(self, node: Any) -> int:
        """Return the desired selfplay concurrency for a node.

        Uses unified resource targets for consistent 60-80% utilization:
        - Backpressure-aware: Reduces jobs when training queue is full
        - Adaptive scaling: Increases jobs when underutilized, decreases when overloaded
        - Host-tier aware: Adjusts targets based on hardware capability

        Args:
            node: NodeInfo-like object with attributes:
                - node_id: str
                - memory_gb: int
                - has_gpu: bool
                - cpu_count: int
                - cpu_percent: float
                - memory_percent: float
                - disk_percent: float
                - gpu_percent: float
                - gpu_memory_percent: float
                - selfplay_jobs: int
                - gpu_name: str (optional)
                - gpu_count: int (optional)

        Returns:
            Target number of selfplay jobs for this node (always >= 1 unless blocked)
        """
        # Check safeguards first
        if self._is_emergency_active_fn:
            try:
                if self._is_emergency_active_fn():
                    return 0
            except (TypeError, AttributeError, RuntimeError) as e:
                # Dec 2025: Narrow exception - callback may be misconfigured
                logger.debug(f"[SelfplayScheduler] Emergency check callback error: {e}")

        # Check backpressure - reduce production when training queue is full
        backpressure_factor = 1.0
        if self._should_stop_production_fn:
            try:
                # Import QueueType lazily to avoid circular imports
                from app.coordination.backpressure import QueueType
                if self._should_stop_production_fn(QueueType.TRAINING_DATA):
                    if self._verbose:
                        logger.info(f"Backpressure STOP: training queue full, halting selfplay on {getattr(node, 'node_id', 'unknown')}")
                    return 0
            except Exception as e:
                if self._verbose:
                    logger.debug(f"Backpressure stop check error: {e}")

        if self._should_throttle_production_fn and self._get_throttle_factor_fn:
            try:
                from app.coordination.backpressure import QueueType
                if self._should_throttle_production_fn(QueueType.TRAINING_DATA):
                    backpressure_factor = self._get_throttle_factor_fn(QueueType.TRAINING_DATA)
                    if self._verbose:
                        logger.info(f"Backpressure throttle: factor={backpressure_factor:.2f}")
            except Exception as e:
                if self._verbose:
                    logger.debug(f"Backpressure throttle check error: {e}")

        # Extract node metrics
        node_id = getattr(node, "node_id", "unknown")
        memory_gb = int(getattr(node, "memory_gb", 0) or 0)
        has_gpu = bool(getattr(node, "has_gpu", False))
        cpu_count = int(getattr(node, "cpu_count", 0) or 0)
        cpu_percent = float(getattr(node, "cpu_percent", 0.0) or 0.0)
        mem_percent = float(getattr(node, "memory_percent", 0.0) or 0.0)
        disk_percent = float(getattr(node, "disk_percent", 0.0) or 0.0)
        gpu_percent = float(getattr(node, "gpu_percent", 0.0) or 0.0)
        gpu_mem_percent = float(getattr(node, "gpu_memory_percent", 0.0) or 0.0)
        current_jobs = int(getattr(node, "selfplay_jobs", 0) or 0)
        gpu_name = getattr(node, "gpu_name", "") or ""
        gpu_count = int(getattr(node, "gpu_count", 1) or 1) if has_gpu else 0

        # Minimum memory requirement
        if memory_gb > 0 and memory_gb < MIN_MEMORY_GB_FOR_TASKS:
            return 0

        # Record utilization for adaptive feedback
        if self._record_utilization_fn:
            with contextlib.suppress(Exception):
                self._record_utilization_fn(node_id, cpu_percent, gpu_percent, mem_percent, current_jobs)

        # Try unified resource targets if available
        if self._get_host_targets_fn and self._get_target_job_count_fn:
            try:
                host_targets = self._get_host_targets_fn(node_id)
                target_selfplay = self._get_target_job_count_fn(
                    node_id,
                    cpu_count if cpu_count > 0 else 8,
                    cpu_percent,
                    gpu_percent if has_gpu else 0.0,
                )

                # Scale up if underutilized
                if self._should_scale_up_fn:
                    scale_up, reason = self._should_scale_up_fn(
                        node_id, cpu_percent, gpu_percent, current_jobs
                    )
                    if scale_up and current_jobs < target_selfplay:
                        scale_up_increment = min(4, target_selfplay - current_jobs)
                        target_selfplay = current_jobs + scale_up_increment
                        if self._verbose:
                            logger.info(f"Scale-up on {node_id}: {reason}, target={target_selfplay}")

                # Scale down if overloaded
                if self._should_scale_down_fn:
                    scale_down, reduction, reason = self._should_scale_down_fn(
                        node_id, cpu_percent, gpu_percent, mem_percent
                    )
                    if scale_down:
                        target_selfplay = max(1, current_jobs - reduction)
                        logger.info(f"Scale-down on {node_id}: {reason}, target={target_selfplay}")

                # Apply backpressure factor
                target_selfplay = int(target_selfplay * backpressure_factor)

                # Apply host-specific max
                max_selfplay = getattr(host_targets, "max_selfplay", target_selfplay)
                target_selfplay = min(target_selfplay, max_selfplay)

                return int(max(1, target_selfplay))

            except Exception as e:
                if self._verbose:
                    logger.info(f"Resource targets error, falling back to hardware-aware: {e}")

        # FALLBACK: Use hardware-aware limits
        if self._get_max_selfplay_for_node_fn:
            max_selfplay = self._get_max_selfplay_for_node_fn(
                node_id=node_id,
                gpu_count=gpu_count,
                gpu_name=gpu_name,
                cpu_count=cpu_count,
                memory_gb=memory_gb,
                has_gpu=has_gpu,
            )
        else:
            # Minimal fallback when callback unavailable
            max_selfplay = self._compute_hardware_limit(
                has_gpu, gpu_name, gpu_count, cpu_count, memory_gb
            )

        target_selfplay = max_selfplay

        # Utilization-aware adjustments
        gpu_overloaded = gpu_percent > 85 or gpu_mem_percent > 85
        cpu_overloaded = cpu_percent > 80
        gpu_has_headroom = gpu_percent < 60 and gpu_mem_percent < 75
        cpu_has_headroom = cpu_percent < 60

        if gpu_overloaded:
            target_selfplay = max(2, target_selfplay - 2)
        if cpu_overloaded:
            target_selfplay = max(2, target_selfplay - 1)

        if ((has_gpu and gpu_has_headroom and cpu_has_headroom) or
            (not has_gpu and cpu_has_headroom)) and current_jobs < target_selfplay:
            target_selfplay = min(target_selfplay, current_jobs + 2)

        # Resource pressure warnings
        if disk_percent >= DISK_WARNING_THRESHOLD:
            target_selfplay = min(target_selfplay, 4)
        if mem_percent >= MEMORY_WARNING_THRESHOLD:
            target_selfplay = min(target_selfplay, 2)

        # Apply backpressure factor
        target_selfplay = int(target_selfplay * backpressure_factor)

        return int(max(1, target_selfplay))

    def _compute_hardware_limit(
        self,
        has_gpu: bool,
        gpu_name: str,
        gpu_count: int,
        cpu_count: int,
        memory_gb: int,
    ) -> int:
        """Compute hardware-based max selfplay limit.

        This is a fallback when resource_optimizer callbacks are unavailable.
        """
        if has_gpu:
            gpu_upper = gpu_name.upper()
            if any(g in gpu_upper for g in ["GH200"]):
                return int(cpu_count * 0.8) if cpu_count > 0 else 48
            elif any(g in gpu_upper for g in ["H100", "H200"]):
                return min(int(cpu_count * 0.5), 48) if cpu_count > 0 else 32
            elif any(g in gpu_upper for g in ["A100", "L40"]):
                return min(int(cpu_count * 0.4), 32) if cpu_count > 0 else 24
            elif any(g in gpu_upper for g in ["5090"]):
                return min(int(cpu_count * 0.3), gpu_count * 12, 64) if cpu_count > 0 else 48
            elif any(g in gpu_upper for g in ["A10", "4090", "3090"]):
                return min(int(cpu_count * 0.3), 24) if cpu_count > 0 else 16
            elif any(g in gpu_upper for g in ["4080", "4070", "3080", "4060"]):
                return min(int(cpu_count * 0.25), 12) if cpu_count > 0 else 8
            elif any(g in gpu_upper for g in ["3070", "3060", "2060", "2070", "2080"]):
                return min(int(cpu_count * 0.2), 10) if cpu_count > 0 else 6
            else:
                return min(int(cpu_count * 0.2), 8) if cpu_count > 0 else 6
        else:
            return min(int(cpu_count * 0.3), 32) if cpu_count > 0 else 8
