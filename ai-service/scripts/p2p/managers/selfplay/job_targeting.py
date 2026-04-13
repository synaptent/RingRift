"""JobTargetingMixin: Job targeting and resource allocation for selfplay scheduling.

Extracted from SelfplayScheduler for better modularity.
Handles memory-aware job allocation, GPU detection, multi-config support,
and per-node job targeting.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scripts.p2p.models import NodeInfo

# Import constants from canonical source to avoid duplication
try:
    from scripts.p2p.constants import (
        DISK_WARNING_THRESHOLD,
        MEMORY_WARNING_THRESHOLD,
        MIN_MEMORY_GB_FOR_TASKS,
    )
    from app.p2p.constants import (
        CPU_ONLY_JOB_MIN_CPUS,
    )
except ImportError:
    # Fallback for testing/standalone use - match canonical values
    MIN_MEMORY_GB_FOR_TASKS = 64
    DISK_WARNING_THRESHOLD = 65
    MEMORY_WARNING_THRESHOLD = 75
    CPU_ONLY_JOB_MIN_CPUS = 128

logger = logging.getLogger(__name__)

# Memory-aware job allocation constants (P1 - Sprint 6, Jan 2026)
# Job-type specific memory requirements in GB
JOB_MEMORY_REQUIREMENTS: dict[str, float] = {
    "gpu_gumbel": 8.0,  # High-quality Gumbel MCTS on GPU
    "gpu_policy": 6.0,  # Policy-only inference on GPU
    "cpu_heuristic": 2.0,  # CPU heuristic selfplay
    "cpu_gumbel": 4.0,  # CPU Gumbel MCTS
    "training": 16.0,  # Training job (needs extra headroom)
    "evaluation": 4.0,  # Evaluation/gauntlet job
    "default": 4.0,  # Default for unknown job types
}
# System reserved memory (OS, drivers, etc.)
SYSTEM_RESERVED_MEMORY_GB = 4.0
# Minimum free memory to maintain after job allocation
MIN_FREE_MEMORY_GB = 2.0

# Session 17.34: Multi-config per node constants
# Large GPUs can run multiple different configs simultaneously for better utilization
# VRAM thresholds (GB) -> max concurrent distinct configs
MAX_CONCURRENT_CONFIGS_BY_VRAM: dict[int, int] = {
    96: 3,  # GH200 96GB: 3 concurrent configs (improved GPU utilization)
    80: 2,  # A100/H100 80GB: 2 concurrent configs
    48: 2,  # L40S 48GB: 2 concurrent configs
    40: 1,  # A100 40GB: 1 config (tighter memory)
    24: 1,  # RTX 4090/3090: 1 config
}
# Default for GPUs not in the list
DEFAULT_MAX_CONCURRENT_CONFIGS = 1


class JobTargetingMixin:
    """Mixin providing job targeting and resource allocation methods.

    Extracted from SelfplayScheduler. Provides:
    - Memory-aware job allocation
    - GPU detection with YAML fallback
    - Multi-config per node support
    - Per-node job target calculation
    - Hybrid GPU/CPU job targeting
    """

    # =========================================================================
    # Memory-Aware Job Allocation (P1 - Sprint 6, Jan 2026)
    # =========================================================================

    def _get_job_memory_requirement(self, job_type: str) -> float:
        """Get memory requirement in GB for a specific job type.

        Args:
            job_type: Type of job (gpu_gumbel, cpu_heuristic, training, etc.)

        Returns:
            Memory requirement in GB for this job type.
        """
        return JOB_MEMORY_REQUIREMENTS.get(
            job_type, JOB_MEMORY_REQUIREMENTS["default"]
        )

    def _check_memory_available(
        self, node: Any, job_type: str = "gpu_gumbel"
    ) -> bool:
        """Check if a node has enough memory for a specific job type.

        This method checks CURRENT memory usage, not just total memory.
        It accounts for:
        - Job-type specific memory requirements
        - System reserved memory (OS, drivers)
        - Minimum free memory buffer

        Args:
            node: Node info object with memory_gb and memory_percent attributes.
            job_type: Type of job being considered.

        Returns:
            True if node has sufficient memory for this job type.
        """
        try:
            # Get total and current memory usage
            total_memory_gb = float(getattr(node, "memory_gb", 0) or 0)
            memory_percent = float(getattr(node, "memory_percent", 0.0) or 0.0)

            if total_memory_gb <= 0:
                # No memory info available - fall back to basic check
                return True

            # Calculate current memory usage
            current_usage_gb = (memory_percent / 100.0) * total_memory_gb

            # Calculate available memory after accounting for:
            # 1. Current usage
            # 2. System reserved memory
            # 3. Minimum free memory buffer
            available_gb = (
                total_memory_gb
                - current_usage_gb
                - SYSTEM_RESERVED_MEMORY_GB
                - MIN_FREE_MEMORY_GB
            )

            # Get job-specific memory requirement
            job_memory_gb = self._get_job_memory_requirement(job_type)

            # Check if we have enough available memory
            has_enough = available_gb >= job_memory_gb

            if not has_enough and self.verbose:
                node_id = getattr(node, "node_id", "unknown")
                logger.debug(
                    f"Memory check failed for {node_id}: "
                    f"available={available_gb:.1f}GB, "
                    f"needed={job_memory_gb:.1f}GB "
                    f"(total={total_memory_gb:.1f}GB, "
                    f"used={memory_percent:.1f}%)"
                )

            return has_enough

        except (TypeError, ValueError, AttributeError) as e:
            # On any error, fall back to allowing the job
            logger.debug(f"Memory check error: {e}")
            return True

    def _get_recommended_job_type(self, node: Any) -> str:
        """Get the recommended job type based on node capabilities and memory.

        Considers:
        - GPU availability
        - Current memory usage
        - GPU memory usage

        Args:
            node: Node info object with hardware attributes.

        Returns:
            Recommended job type string.
        """
        has_gpu = bool(getattr(node, "has_gpu", False))
        gpu_mem_percent = float(getattr(node, "gpu_memory_percent", 0.0) or 0.0)

        if has_gpu:
            # GPU node - check GPU memory for job type
            if gpu_mem_percent < 50:
                # Plenty of GPU memory - can run high-quality Gumbel
                return "gpu_gumbel"
            elif gpu_mem_percent < 80:
                # Moderate GPU memory - use policy-only
                return "gpu_policy"
            else:
                # GPU memory constrained - fall back to CPU
                return "cpu_heuristic"
        else:
            # CPU-only node
            if self._check_memory_available(node, "cpu_gumbel"):
                return "cpu_gumbel"
            else:
                return "cpu_heuristic"

    # =========================================================================
    # GPU Detection and Config Filtering
    # =========================================================================

    def _lookup_yaml_gpu_config(self, node_id: str) -> tuple[bool, str, int]:
        """Lookup GPU config from distributed_hosts.yaml.

        This provides an authoritative fallback when runtime GPU detection
        fails (e.g., nvidia-smi not available, subprocess issues, etc.).

        Args:
            node_id: The node identifier to look up.

        Returns:
            Tuple of (has_gpu, gpu_name, gpu_vram_gb).

        January 2026: Added to fix GPU underutilization where runtime
        detection fails but YAML config has reliable GPU info.
        """
        try:
            from app.config.cluster_config import load_cluster_config

            config = load_cluster_config()
            host_cfg = config.hosts_raw.get(node_id, {})

            gpu_name = host_cfg.get("gpu", "")
            gpu_vram_gb = int(host_cfg.get("gpu_vram_gb", 0) or 0)

            # Has GPU if either gpu name or vram specified (excluding MPS/Apple)
            if gpu_name and "MPS" not in gpu_name.upper() and "APPLE" not in gpu_name.upper():
                return (True, gpu_name, gpu_vram_gb)
            if gpu_vram_gb > 0:
                return (True, gpu_name or "Unknown GPU", gpu_vram_gb)

            return (False, "", 0)
        except Exception as e:
            logger.debug(f"Could not load YAML config for {node_id}: {e}")
            return (False, "", 0)

    def _node_has_gpu(self, node: "NodeInfo") -> bool:
        """Check if a node has GPU capability.

        Args:
            node: Node information object

        Returns:
            True if the node has GPU (CUDA-capable), False otherwise.

        December 2025: Added for GPU-aware config selection.
        January 2026: Added YAML config fallback for when runtime detection fails.
            This fixes GPU underutilization on Lambda GH200 nodes where runtime
            GPU detection via NodeInfo methods fails silently.
        January 2026 FIX: Runtime methods returning False should NOT short-circuit.
            The YAML fallback must be checked when runtime detection returns False.
        """
        # Try runtime GPU detection methods - if ANY returns True, we have a GPU
        # NOTE: Methods returning False means detection failed, NOT that there's no GPU

        # Try has_cuda_gpu() method first (NodeInfo from scripts/p2p/models.py)
        if hasattr(node, "has_cuda_gpu"):
            result = node.has_cuda_gpu()
            if result is True:
                return True
            # If False/None, continue to next check (detection may have failed)

        # Try is_gpu_node() method
        if hasattr(node, "is_gpu_node"):
            result = node.is_gpu_node()
            if result is True:
                return True

        # Try has_gpu attribute
        if hasattr(node, "has_gpu") and node.has_gpu:
            return True

        # Try gpu_info attribute directly
        if hasattr(node, "gpu_info"):
            gpu_info = node.gpu_info
            if gpu_info is not None:
                gpu_count = getattr(gpu_info, "gpu_count", 0)
                if gpu_count > 0:
                    return True

        # YAML config fallback - use as authoritative source when runtime detection fails
        # January 2026: This is the critical fallback for nodes where runtime detection
        # returns False but we KNOW they have GPUs from our cluster configuration.
        node_id = getattr(node, "node_id", "")
        if node_id:
            yaml_has_gpu, yaml_gpu_name, _ = self._lookup_yaml_gpu_config(node_id)
            if yaml_has_gpu:
                logger.warning(
                    f"[GPU Detection] Using YAML fallback for {node_id}: "
                    f"runtime detection failed but YAML shows GPU: {yaml_gpu_name}"
                )
                return True

        # Fallback: assume no GPU if we can't determine
        return False

    def _filter_configs_by_gpu(
        self,
        configs: list[dict[str, Any]],
        node: "NodeInfo",
    ) -> list[dict[str, Any]]:
        """Filter configs to only those compatible with node's GPU capability.

        For CPU-only nodes, removes configs that require GPU (e.g., gumbel-mcts).
        GPU nodes can run all configs.

        Args:
            configs: List of selfplay config dicts
            node: Node information object

        Returns:
            Filtered list of configs compatible with the node.

        December 2025: Core GPU-aware filtering for config selection.
        """
        if self._node_has_gpu(node):
            # GPU nodes can run any config
            return configs

        # CPU-only nodes: filter out GPU-required configs
        cpu_compatible = [
            cfg
            for cfg in configs
            if not self._engine_mode_requires_gpu(cfg.get("engine_mode", ""))
        ]

        if len(cpu_compatible) < len(configs):
            filtered_count = len(configs) - len(cpu_compatible)
            node_id = getattr(node, "node_id", "unknown")
            logger.info(
                f"Filtered {filtered_count} GPU-required configs for CPU-only node {node_id}"
            )

        return cpu_compatible

    # =========================================================================
    # Multi-Config Per Node Support (Session 17.34)
    # =========================================================================

    def _get_max_concurrent_configs(self, node: "NodeInfo") -> int:
        """Get maximum concurrent distinct configs for a node based on GPU VRAM.

        Session 17.34: Large GPUs can run multiple different selfplay configs
        simultaneously for better GPU utilization. This method returns how many
        distinct configs a node should run concurrently.

        Args:
            node: Node information object

        Returns:
            Maximum number of distinct configs to run concurrently (1-3)
        """
        if not self._node_has_gpu(node):
            # CPU nodes only run one config
            return 1

        # Get GPU VRAM
        gpu_vram = int(
            getattr(node, "gpu_vram_gb", 0)
            or getattr(node, "gpu_memory_gb", 0)
            or 0
        )

        if gpu_vram <= 0:
            return DEFAULT_MAX_CONCURRENT_CONFIGS

        # Find the highest VRAM threshold we meet or exceed
        best_match = DEFAULT_MAX_CONCURRENT_CONFIGS
        for vram_threshold, max_configs in sorted(
            MAX_CONCURRENT_CONFIGS_BY_VRAM.items(),
            reverse=True,  # Check highest first
        ):
            if gpu_vram >= vram_threshold:
                best_match = max_configs
                break

        return best_match

    def _get_configs_running_on_node(self, node_id: str) -> set[str]:
        """Get the set of config keys currently running on a node.

        Session 17.34: Used for multi-config diversity - ensures large GPUs
        run different configs simultaneously rather than duplicates.

        Args:
            node_id: The node identifier

        Returns:
            Set of config keys (e.g., {"hex8_2p", "square8_3p"}) running on node
        """
        if self.get_active_configs_for_node is None:
            return set()

        try:
            active_configs = self.get_active_configs_for_node(node_id)
            return set(active_configs) if active_configs else set()
        except (TypeError, AttributeError, RuntimeError, KeyError) as e:
            logger.debug(f"Failed to get active configs for {node_id}: {e}")
            return set()

    def _apply_multi_config_preference(
        self,
        configs: list[dict[str, Any]],
        node: "NodeInfo",
    ) -> list[dict[str, Any]]:
        """Apply multi-config preference for large GPUs.

        Session 17.34: When a node can run multiple configs and has running jobs,
        boost priority for configs NOT currently running to improve diversity.

        Args:
            configs: List of selfplay config dicts with effective_priority set
            node: Node information object

        Returns:
            Configs with adjusted priorities for multi-config diversity
        """
        node_id = getattr(node, "node_id", "")
        if not node_id:
            return configs

        max_configs = self._get_max_concurrent_configs(node)
        if max_configs <= 1:
            # Single config node - no adjustment needed
            return configs

        running_configs = self._get_configs_running_on_node(node_id)
        if not running_configs:
            # No running configs - no adjustment needed
            return configs

        # If already at max distinct configs, allow duplicates
        if len(running_configs) >= max_configs:
            return configs

        # Boost priority for configs NOT currently running
        # This encourages multi-config diversity on large GPUs
        DIVERSITY_BOOST = 5  # Priority boost for non-running configs

        for cfg in configs:
            config_key = f"{cfg.get('board_type', '')}_{cfg.get('num_players', 2)}p"
            current_priority = cfg.get("effective_priority", 1)

            if config_key not in running_configs:
                # Boost configs not currently running on this node
                cfg["effective_priority"] = current_priority + DIVERSITY_BOOST
                cfg["_multi_config_boosted"] = True

        # Log the adjustment
        boosted_count = sum(1 for c in configs if c.get("_multi_config_boosted"))
        if boosted_count > 0:
            logger.info(
                f"Multi-config diversity: Boosted {boosted_count} non-running configs "
                f"for {node_id} (running: {running_configs}, max: {max_configs})"
            )

        return configs

    # =========================================================================
    # Per-Node Job Targeting
    # =========================================================================

    def get_target_jobs_for_node(self, node: "NodeInfo") -> int:
        """Return the desired selfplay concurrency for a node.

        Uses unified resource targets for consistent 60-80% utilization:
        - Backpressure-aware: Reduces jobs when training queue is full
        - Adaptive scaling: Increases jobs when underutilized, decreases when overloaded
        - Host-tier aware: Adjusts targets based on hardware capability

        Args:
            node: Node information

        Returns:
            Target number of selfplay jobs (minimum 1)

        Target: 60-80% CPU/GPU utilization for optimal training throughput.
        """
        # Dec 29, 2025: Skip coordinator nodes (no selfplay capability)
        node_caps = getattr(node, "capabilities", None) or []
        if not node_caps or "selfplay" not in node_caps:
            return 0

        # Check safeguards first - halt all selfplay during emergency
        if self.is_emergency_active is not None:
            try:
                if self.is_emergency_active():
                    return 0
            except (TypeError, AttributeError, RuntimeError, KeyError):
                pass  # Ignore errors in safeguards callback (non-critical)

        # Dec 2025 Phase 5: Check evaluation backpressure - pause when eval queue full
        # Dec 31, 2025: Add minimum job floor for high-end GPUs during backpressure
        # High-end GPUs should never be completely idle to maximize cluster throughput
        if self._evaluation_backpressure_active:
            has_gpu = bool(getattr(node, "has_gpu", False))
            # Use gpu_power_score to identify high-end GPUs:
            # H100 80GB = 4000, A100 80GB = 6864, GH200 = 15000
            gpu_power = int(getattr(node, "gpu_power_score", 0) or 0)
            gpu_name = str(getattr(node, "gpu_name", "") or "")
            # High-end GPU: A100, H100, GH200, RTX 5090 (power_score >= 4000)
            is_high_end_gpu = has_gpu and gpu_power >= 4000

            if is_high_end_gpu:
                # Allow 25% capacity during backpressure for high-end GPUs
                # This prevents total starvation while still reducing throughput
                base_target = int(getattr(node, "max_selfplay_jobs", 4) or 4)
                min_jobs = max(1, base_target // 4)
                logger.info(
                    f"Backpressure active but allowing {min_jobs} jobs on high-end GPU "
                    f"{node.node_id} ({gpu_name}, power={gpu_power})"
                )
                return min_jobs

            logger.debug(
                f"Evaluation backpressure active, halting selfplay on {node.node_id}"
            )
            return 0

        # Check backpressure - reduce production when training queue is full
        backpressure_factor = 1.0
        if (
            self.should_stop_production is not None
            and self.should_throttle_production is not None
        ):
            try:
                # Import QueueType here to avoid circular imports
                try:
                    from app.coordination.queue_monitor import QueueType

                    queue_type = QueueType.TRAINING_DATA
                except ImportError:
                    queue_type = "TRAINING_DATA"

                if self.should_stop_production(queue_type):
                    logger.info(
                        f"Backpressure STOP: training queue full, halting selfplay on {node.node_id}"
                    )
                    return 0
                if self.should_throttle_production(queue_type):
                    if self.get_throttle_factor is not None:
                        backpressure_factor = self.get_throttle_factor(queue_type)
                        logger.info(f"Backpressure throttle: factor={backpressure_factor:.2f}")
            except (TypeError, AttributeError, ValueError, RuntimeError) as e:
                logger.info(f"Backpressure check error: {e}")

        # Minimum memory requirement - skip low-memory machines to avoid OOM
        memory_gb = int(getattr(node, "memory_gb", 0) or 0)
        if memory_gb > 0 and memory_gb < MIN_MEMORY_GB_FOR_TASKS:
            return 0

        # Memory-aware job allocation (P1 - Sprint 6, Jan 2026)
        # Check current memory usage, not just total memory
        has_gpu = bool(getattr(node, "has_gpu", False))
        recommended_job_type = self._get_recommended_job_type(node)
        if not self._check_memory_available(node, recommended_job_type):
            # Not enough memory for even the lightest job type
            if not self._check_memory_available(node, "cpu_heuristic"):
                if self.verbose:
                    node_id = getattr(node, "node_id", "unknown")
                    logger.debug(
                        f"Node {node_id} has insufficient memory even for cpu_heuristic, "
                        f"skipping job allocation"
                    )
                return 0

        # Extract node metrics
        cpu_count = int(getattr(node, "cpu_count", 0) or 0)
        cpu_percent = float(getattr(node, "cpu_percent", 0.0) or 0.0)
        mem_percent = float(getattr(node, "memory_percent", 0.0) or 0.0)
        disk_percent = float(getattr(node, "disk_percent", 0.0) or 0.0)
        gpu_percent = float(getattr(node, "gpu_percent", 0.0) or 0.0)
        gpu_mem_percent = float(getattr(node, "gpu_memory_percent", 0.0) or 0.0)
        current_jobs = int(getattr(node, "selfplay_jobs", 0) or 0)

        # Record utilization for adaptive feedback
        if self.record_utilization is not None:
            with contextlib.suppress(Exception):
                self.record_utilization(
                    node.node_id, cpu_percent, gpu_percent, mem_percent, current_jobs
                )

        # Use unified resource targets if available
        if (
            self.get_host_targets is not None
            and self.get_target_job_count is not None
        ):
            try:
                # Get host-specific targets adjusted for tier and backpressure
                host_targets = self.get_host_targets(node.node_id)

                # Use the unified target calculator
                target_selfplay = self.get_target_job_count(
                    node.node_id,
                    cpu_count if cpu_count > 0 else 8,
                    cpu_percent,
                    gpu_percent if has_gpu else 0.0,
                )

                # Check if we should scale up (underutilized)
                if self.should_scale_up is not None:
                    scale_up, reason = self.should_scale_up(
                        node.node_id, cpu_percent, gpu_percent, current_jobs
                    )
                    if scale_up and current_jobs < target_selfplay:
                        # Controlled scale-up: Add 2-4 jobs at a time, not all at once
                        scale_up_increment = min(4, target_selfplay - current_jobs)
                        target_selfplay = current_jobs + scale_up_increment
                        if self.verbose:
                            logger.info(
                                f"Scale-up on {node.node_id}: {reason}, target={target_selfplay}"
                            )

                # Check if we should scale down (overloaded)
                if self.should_scale_down is not None:
                    scale_down, reduction, reason = self.should_scale_down(
                        node.node_id, cpu_percent, gpu_percent, mem_percent
                    )
                    if scale_down:
                        target_selfplay = max(1, current_jobs - reduction)
                        logger.info(
                            f"Scale-down on {node.node_id}: {reason}, target={target_selfplay}"
                        )

                # Apply backpressure factor
                target_selfplay = int(target_selfplay * backpressure_factor)

                # Apply host-specific max
                target_selfplay = min(target_selfplay, host_targets.max_selfplay)

                final_target = int(max(1, target_selfplay))

                # P0.2 (Dec 2025): Emit event when target changes significantly
                old_target = self._previous_targets.get(node.node_id, final_target)
                target_change = abs(final_target - old_target)
                relative_change = target_change / max(1, old_target)
                if target_change >= 3 or relative_change >= 0.5:
                    reason = "target_increased" if final_target > old_target else "target_decreased"
                    evt_priority = "high" if target_change >= 5 or relative_change >= 0.75 else "normal"
                    self._emit_selfplay_target_updated(
                        config_key=f"node:{node.node_id}",
                        priority=evt_priority,
                        reason=f"{reason}:{target_change}",
                        target_jobs=final_target,
                    )
                    self._previous_targets[node.node_id] = final_target

                return final_target

            except (TypeError, AttributeError, ValueError, KeyError, RuntimeError) as e:
                logger.info(f"Resource targets error, falling back to hardware-aware: {e}")

        # FALLBACK: Use unified hardware-aware limits from resource_optimizer
        # This ensures consistent limits across all orchestrators
        gpu_name = getattr(node, "gpu_name", "") or ""
        gpu_count = int(getattr(node, "gpu_count", 1) or 1) if has_gpu else 0

        if self.get_max_selfplay_for_node is not None:
            # Use single source of truth from resource_optimizer
            max_selfplay = self.get_max_selfplay_for_node(
                node_id=node.node_id,
                gpu_count=gpu_count,
                gpu_name=gpu_name,
                cpu_count=cpu_count,
                memory_gb=memory_gb,
                has_gpu=has_gpu,
            )
        else:
            # Minimal fallback when resource_optimizer unavailable
            # Values calibrated from observed workloads (GH200: 48 jobs at 70% GPU)
            if has_gpu:
                gpu_upper = gpu_name.upper()
                if any(g in gpu_upper for g in ["GH200"]):
                    # GH200 with unified 480GB memory - CPU is bottleneck
                    max_selfplay = int(cpu_count * 0.8) if cpu_count > 0 else 48
                elif any(g in gpu_upper for g in ["H100", "H200"]):
                    max_selfplay = (
                        min(int(cpu_count * 0.5), 48) if cpu_count > 0 else 32
                    )
                elif any(g in gpu_upper for g in ["A100", "L40"]):
                    max_selfplay = (
                        min(int(cpu_count * 0.4), 32) if cpu_count > 0 else 24
                    )
                elif any(g in gpu_upper for g in ["5090"]):
                    # RTX 5090 (32GB) - very high capacity
                    max_selfplay = (
                        min(int(cpu_count * 0.3), gpu_count * 12, 64)
                        if cpu_count > 0
                        else 48
                    )
                elif any(g in gpu_upper for g in ["A10", "4090", "3090"]):
                    max_selfplay = (
                        min(int(cpu_count * 0.3), 24) if cpu_count > 0 else 16
                    )
                elif any(g in gpu_upper for g in ["4080", "4070", "3080", "4060"]):
                    max_selfplay = (
                        min(int(cpu_count * 0.25), 12) if cpu_count > 0 else 8
                    )
                elif any(
                    g in gpu_upper for g in ["3070", "3060", "2060", "2070", "2080"]
                ):
                    max_selfplay = (
                        min(int(cpu_count * 0.2), 10) if cpu_count > 0 else 6
                    )
                else:
                    max_selfplay = min(int(cpu_count * 0.2), 8) if cpu_count > 0 else 6
            else:
                # CPU-only: ~0.3 jobs per core, capped at 32
                max_selfplay = min(int(cpu_count * 0.3), 32) if cpu_count > 0 else 8

        target_selfplay = max_selfplay

        # Utilization-aware adjustments (target 60-90%)
        # Jan 5, 2026: Raised CPU threshold 80%->90% for +10-15% cluster utilization
        gpu_overloaded = gpu_percent > 85 or gpu_mem_percent > 85
        cpu_overloaded = cpu_percent > 90
        gpu_has_headroom = gpu_percent < 60 and gpu_mem_percent < 75
        cpu_has_headroom = cpu_percent < 60

        # Scale DOWN if overloaded
        if gpu_overloaded:
            target_selfplay = max(2, target_selfplay - 2)
        if cpu_overloaded:
            target_selfplay = max(2, target_selfplay - 1)

        # Scale UP only if both resources have headroom (gradual)
        if (
            not gpu_overloaded
            and not cpu_overloaded
            and current_jobs > 0
            and (has_gpu and gpu_has_headroom and cpu_has_headroom)
        ) or ((not has_gpu and cpu_has_headroom) and current_jobs < target_selfplay):
            target_selfplay = min(target_selfplay, current_jobs + 2)

        # Resource pressure warnings
        if disk_percent >= DISK_WARNING_THRESHOLD:
            target_selfplay = min(target_selfplay, 4)
        if mem_percent >= MEMORY_WARNING_THRESHOLD:
            target_selfplay = min(target_selfplay, 2)

        # Apply backpressure factor
        target_selfplay = int(target_selfplay * backpressure_factor)

        final_target = int(max(1, target_selfplay))

        # P0.2 (Dec 2025): Emit event when target changes significantly
        node_id = getattr(node, "node_id", "unknown")
        old_target = self._previous_targets.get(node_id, final_target)
        target_change = abs(final_target - old_target)

        # Emit if target changed by 3+ jobs or 50%+ relative change
        relative_change = target_change / max(1, old_target)
        if target_change >= 3 or relative_change >= 0.5:
            reason = "target_increased" if final_target > old_target else "target_decreased"
            priority = "high" if target_change >= 5 or relative_change >= 0.75 else "normal"
            self._emit_selfplay_target_updated(
                config_key=f"node:{node_id}",
                priority=priority,
                reason=f"{reason}:{target_change}",
                target_jobs=final_target,
            )
            self._previous_targets[node_id] = final_target

        return final_target

    def get_hybrid_job_targets(self, node: "NodeInfo") -> dict[str, int]:
        """Get separate GPU and CPU-only selfplay job targets for hybrid mode.

        For high-CPU nodes with limited GPU VRAM (like Vast hosts), this enables:
        - Running GPU jobs up to VRAM limit
        - Running additional CPU-only jobs to utilize excess CPU capacity

        Args:
            node: Node information

        Returns:
            Dict with 'gpu_jobs', 'cpu_only_jobs', 'total_jobs'
        """
        has_gpu = bool(getattr(node, "has_gpu", False))
        cpu_count = int(getattr(node, "cpu_count", 0) or 0)
        memory_gb = int(getattr(node, "memory_gb", 0) or 0)
        gpu_name = getattr(node, "gpu_name", "") or ""
        gpu_count = int(getattr(node, "gpu_count", 1) or 1) if has_gpu else 0

        # Use hybrid limits function if available
        if self.get_hybrid_selfplay_limits is not None:
            try:
                limits = self.get_hybrid_selfplay_limits(
                    node_id=node.node_id,
                    gpu_count=gpu_count,
                    gpu_name=gpu_name,
                    cpu_count=cpu_count,
                    memory_gb=memory_gb,
                    has_gpu=has_gpu,
                )
                return limits
            except (TypeError, AttributeError, ValueError, KeyError, RuntimeError) as e:
                logger.info(f"Hybrid limits error: {e}")

        # Fallback: No CPU-only jobs, use standard target
        gpu_jobs = self.get_target_jobs_for_node(node)
        return {"gpu_jobs": gpu_jobs, "cpu_only_jobs": 0, "total_jobs": gpu_jobs}

    def should_spawn_cpu_only_jobs(self, node: "NodeInfo") -> bool:
        """Check if a node should spawn CPU-only jobs in addition to GPU jobs.

        CPU-only jobs are beneficial when:
        1. Node has many CPU cores (64+)
        2. Node has limited GPU VRAM (<=16GB per GPU)
        3. GPU jobs are already at capacity (VRAM-limited)

        Args:
            node: Node information

        Returns:
            True if CPU-only jobs should be spawned
        """
        if self.get_hybrid_selfplay_limits is None:
            return False

        cpu_count = int(getattr(node, "cpu_count", 0) or 0)
        has_gpu = bool(getattr(node, "has_gpu", False))
        gpu_name = (getattr(node, "gpu_name", "") or "").upper()

        # Must have significant CPU resources (64+ cores)
        if cpu_count < 64:
            return False

        # For GPU nodes, only spawn CPU-only if GPU has limited VRAM
        if has_gpu:
            # High-end datacenter GPUs don't need CPU-only jobs (plenty of VRAM)
            if any(g in gpu_name for g in ["GH200", "H100", "H200", "A100", "L40"]):
                return False
            # Consumer GPUs with limited VRAM benefit from CPU-only supplement
            if any(
                g in gpu_name
                for g in ["3070", "3060", "2060", "2070", "2080", "4060", "4070"]
            ):
                return True
            # 5090/4090 with 24-32GB might not need it unless very high CPU count
            if any(g in gpu_name for g in ["5090", "4090", "3090"]):
                return cpu_count >= CPU_ONLY_JOB_MIN_CPUS

        # CPU-only nodes always benefit from full CPU utilization
        return True
