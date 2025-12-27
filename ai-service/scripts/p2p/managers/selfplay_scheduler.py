"""SelfplayScheduler: Priority-based selfplay configuration selection.

Extracted from p2p_orchestrator.py for better modularity.
Handles weighted config selection, job targeting, diversity tracking, and Elo-based priority.
"""

from __future__ import annotations

import contextlib
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from scripts.p2p.models import NodeInfo

logger = logging.getLogger(__name__)


# Import constants from canonical source to avoid duplication
try:
    from scripts.p2p.constants import (
        DISK_WARNING_THRESHOLD,
        MEMORY_WARNING_THRESHOLD,
        MIN_MEMORY_GB_FOR_TASKS,
    )
except ImportError:
    # Fallback for testing/standalone use
    MIN_MEMORY_GB_FOR_TASKS = 16
    DISK_WARNING_THRESHOLD = 65  # Conservative: match constants.py
    MEMORY_WARNING_THRESHOLD = 75  # Conservative: match constants.py


@dataclass
class DiversityMetrics:
    """Diversity tracking metrics for selfplay scheduling."""

    games_by_engine_mode: dict[str, int] = field(default_factory=dict)
    games_by_board_config: dict[str, int] = field(default_factory=dict)
    games_by_difficulty: dict[str, int] = field(default_factory=dict)
    asymmetric_games: int = 0
    symmetric_games: int = 0
    last_reset: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with computed metrics."""
        total_games = self.asymmetric_games + self.symmetric_games
        asymmetric_ratio = (
            self.asymmetric_games / total_games if total_games > 0 else 0.0
        )

        engine_total = sum(self.games_by_engine_mode.values())
        engine_mode_distribution = (
            {k: v / engine_total for k, v in self.games_by_engine_mode.items()}
            if engine_total > 0
            else {}
        )

        return {
            "games_by_engine_mode": dict(self.games_by_engine_mode),
            "games_by_board_config": dict(self.games_by_board_config),
            "games_by_difficulty": dict(self.games_by_difficulty),
            "asymmetric_games": self.asymmetric_games,
            "symmetric_games": self.symmetric_games,
            "asymmetric_ratio": asymmetric_ratio,
            "engine_mode_distribution": engine_mode_distribution,
            "uptime_seconds": time.time() - self.last_reset,
        }


class SelfplayScheduler:
    """Manages selfplay configuration selection and job targeting.

    Responsibilities:
    - Weighted config selection based on static priority, Elo performance, curriculum
    - Job targeting per node based on hardware capabilities and utilization
    - Diversity tracking for monitoring
    - Integration with backpressure and resource optimization

    Usage:
        scheduler = SelfplayScheduler(
            get_cluster_elo_fn=lambda: orchestrator._get_cluster_elo_summary(),
            verbose=True
        )

        # Pick a config for a node
        config = scheduler.pick_weighted_config(node)

        # Get job target for a node
        target = scheduler.get_target_jobs_for_node(node)

        # Track diversity
        scheduler.track_diversity(config)

        # Get metrics
        metrics = scheduler.get_diversity_metrics()
    """

    def __init__(
        self,
        get_cluster_elo_fn: Callable[[], dict[str, Any]] | None = None,
        load_curriculum_weights_fn: Callable[[], dict[str, float]] | None = None,
        get_board_priority_overrides_fn: Callable[[], dict[str, int]] | None = None,
        should_stop_production_fn: Callable[[Any], bool] | None = None,
        should_throttle_production_fn: Callable[[Any], bool] | None = None,
        get_throttle_factor_fn: Callable[[Any], float] | None = None,
        record_utilization_fn: Callable[[str, float, float, float, int], None]
        | None = None,
        get_host_targets_fn: Callable[[str], Any] | None = None,
        get_target_job_count_fn: Callable[[str, int, float, float], int] | None = None,
        should_scale_up_fn: Callable[[str, float, float, int], tuple[bool, str]]
        | None = None,
        should_scale_down_fn: Callable[
            [str, float, float, float], tuple[bool, int, str]
        ]
        | None = None,
        get_max_selfplay_for_node_fn: Callable[..., int] | None = None,
        get_hybrid_selfplay_limits_fn: Callable[..., dict[str, int]] | None = None,
        verbose: bool = False,
    ):
        """Initialize the SelfplayScheduler.

        Args:
            get_cluster_elo_fn: Function to get cluster-wide Elo summary
            load_curriculum_weights_fn: Function to load curriculum weights
            get_board_priority_overrides_fn: Function to load board priority overrides
            should_stop_production_fn: Function to check if production should stop (backpressure)
            should_throttle_production_fn: Function to check if production should throttle
            get_throttle_factor_fn: Function to get throttle factor
            record_utilization_fn: Function to record node utilization
            get_host_targets_fn: Function to get host-specific targets
            get_target_job_count_fn: Function to calculate target job count
            should_scale_up_fn: Function to check if scaling up is needed
            should_scale_down_fn: Function to check if scaling down is needed
            get_max_selfplay_for_node_fn: Function to get max selfplay jobs for node
            get_hybrid_selfplay_limits_fn: Function to get hybrid selfplay limits
            verbose: Enable verbose logging
        """
        self.get_cluster_elo = get_cluster_elo_fn or (lambda: {})
        self.load_curriculum_weights = load_curriculum_weights_fn or (lambda: {})
        self.get_board_priority_overrides = get_board_priority_overrides_fn or (
            lambda: {}
        )
        self.should_stop_production = should_stop_production_fn
        self.should_throttle_production = should_throttle_production_fn
        self.get_throttle_factor = get_throttle_factor_fn
        self.record_utilization = record_utilization_fn
        self.get_host_targets = get_host_targets_fn
        self.get_target_job_count = get_target_job_count_fn
        self.should_scale_up = should_scale_up_fn
        self.should_scale_down = should_scale_down_fn
        self.get_max_selfplay_for_node = get_max_selfplay_for_node_fn
        self.get_hybrid_selfplay_limits = get_hybrid_selfplay_limits_fn
        self.verbose = verbose

        # Diversity tracking
        self.diversity_metrics = DiversityMetrics()

    def get_elo_based_priority_boost(self, board_type: str, num_players: int) -> int:
        """Get priority boost based on ELO performance for this config.

        PRIORITY-BASED SCHEDULING: Configs with high-performing models get
        priority boost to allocate more resources to promising configurations.

        Args:
            board_type: Board type (e.g., "hex8", "square8")
            num_players: Number of players (2, 3, or 4)

        Returns:
            Priority boost (0-5) based on:
            - Top model ELO for this config
            - Recent improvement rate
            - Data coverage (inverse - underrepresented get boost)
        """
        boost = 0

        try:
            cluster_elo = self.get_cluster_elo()
            top_models = cluster_elo.get("top_models", [])

            # Find best model for this board/player combo
            best_elo = 0
            for model in top_models:
                model_name = model.get("name", "")
                # Model names typically include board type and player count
                if board_type in model_name or str(num_players) in model_name:
                    best_elo = max(best_elo, model.get("elo", 0))

            # ELO-based boost (every 100 ELO above 1200 = +1 priority)
            if best_elo > 1200:
                boost += min(3, (best_elo - 1200) // 100)

            # Underrepresented config boost
            # (hex and square19 often have fewer games)
            if board_type in ("hexagonal", "square19"):
                boost += 1
            if num_players > 2:
                boost += 1

        except AttributeError:
            pass

        return min(5, boost)  # Cap at +5

    def pick_weighted_config(self, node: NodeInfo) -> dict[str, Any] | None:
        """Pick a selfplay config weighted by priority and node capabilities.

        PRIORITY-BASED SCHEDULING: Combines static priority with dynamic
        ELO-based boosts to allocate more resources to high-performing configs.

        Args:
            node: Node information for capability filtering

        Returns:
            Config dict with board_type, num_players, engine_mode, or None if no valid config
        """
        # Get the selfplay configs - DIVERSE mode prioritized for high-quality training data
        # Uses "mixed" engine mode for varied AI matchups (NNUE, MCTS, heuristic combinations)
        selfplay_configs = [
            # Priority 8: Underrepresented hex/sq19 combos with diverse AI (highest priority)
            {
                "board_type": "hexagonal",
                "num_players": 3,
                "engine_mode": "mixed",
                "priority": 8,
            },
            {
                "board_type": "hexagonal",
                "num_players": 2,
                "engine_mode": "mixed",
                "priority": 8,
            },
            {
                "board_type": "hexagonal",
                "num_players": 4,
                "engine_mode": "mixed",
                "priority": 8,
            },
            {
                "board_type": "hex8",
                "num_players": 2,
                "engine_mode": "mixed",
                "priority": 8,
            },
            {
                "board_type": "hex8",
                "num_players": 3,
                "engine_mode": "mixed",
                "priority": 8,
            },
            {
                "board_type": "hex8",
                "num_players": 4,
                "engine_mode": "mixed",
                "priority": 8,
            },
            {
                "board_type": "square19",
                "num_players": 3,
                "engine_mode": "mixed",
                "priority": 8,
            },
            {
                "board_type": "square19",
                "num_players": 2,
                "engine_mode": "mixed",
                "priority": 8,
            },
            {
                "board_type": "square19",
                "num_players": 4,
                "engine_mode": "mixed",
                "priority": 8,
            },
            # Priority 7: Square8 multi-player with diverse AI
            {
                "board_type": "square8",
                "num_players": 3,
                "engine_mode": "mixed",
                "priority": 7,
            },
            {
                "board_type": "square8",
                "num_players": 4,
                "engine_mode": "mixed",
                "priority": 7,
            },
            # Priority 6: Cross-AI matches (specific matchup types)
            {
                "board_type": "square8",
                "num_players": 2,
                "engine_mode": "heuristic-vs-mcts",
                "priority": 6,
            },
            {
                "board_type": "hexagonal",
                "num_players": 3,
                "engine_mode": "heuristic-vs-mcts",
                "priority": 6,
            },
            {
                "board_type": "square19",
                "num_players": 2,
                "engine_mode": "heuristic-vs-mcts",
                "priority": 6,
            },
            # Priority 5: Standard 2p square8 with diverse AI
            {
                "board_type": "square8",
                "num_players": 2,
                "engine_mode": "mixed",
                "priority": 5,
            },
            # Priority 4: Tournament varied (for evaluation-style games)
            {
                "board_type": "square8",
                "num_players": 2,
                "engine_mode": "tournament-varied",
                "priority": 4,
            },
            {
                "board_type": "hexagonal",
                "num_players": 2,
                "engine_mode": "tournament-varied",
                "priority": 4,
            },
            # Priority 3: CPU-bound specialized modes
            {
                "board_type": "square8",
                "num_players": 2,
                "engine_mode": "mcts-only",
                "priority": 3,
            },
            {
                "board_type": "hexagonal",
                "num_players": 2,
                "engine_mode": "descent-only",
                "priority": 3,
            },
        ]

        # Filter by node memory (avoid large boards on small nodes)
        node_mem = int(getattr(node, "memory_gb", 0) or 0)
        if node_mem and node_mem < 48:
            selfplay_configs = [
                c for c in selfplay_configs if c.get("board_type") == "square8"
            ]

        if not selfplay_configs:
            return None

        # PRIORITY-BASED SCHEDULING: Add ELO-based priority boosts
        # Phase 3.1: Also incorporate curriculum weights from unified AI loop
        curriculum_weights = {}
        try:
            curriculum_weights = self.load_curriculum_weights()
        except (OSError, ValueError, KeyError, ImportError):
            pass  # Use empty weights on error

        # Load board priority overrides from config (0=CRITICAL, 1=HIGH, 2=MEDIUM, 3=LOW)
        board_priority_overrides = self.get_board_priority_overrides()

        for cfg in selfplay_configs:
            elo_boost = self.get_elo_based_priority_boost(
                cfg.get("board_type", ""),
                cfg.get("num_players", 2),
            )

            # Phase 3.1: Apply curriculum weight boost
            # Config keys are formatted as "board_type_Np" (e.g., "square8_2p")
            config_key = f"{cfg.get('board_type', '')}_{cfg.get('num_players', 2)}p"
            curriculum_weight = curriculum_weights.get(config_key, 1.0)
            # Convert weight (0.7-1.5) to priority boost (0-3)
            # weight 0.7 = -1 boost, weight 1.0 = 0 boost, weight 1.5 = +2 boost
            curriculum_boost = int((curriculum_weight - 1.0) * 4)
            curriculum_boost = max(-2, min(3, curriculum_boost))  # Clamp to -2..+3

            # Apply board priority overrides from config
            # 0=CRITICAL adds +6, 1=HIGH adds +4, 2=MEDIUM adds +2, 3=LOW adds 0
            board_priority = board_priority_overrides.get(
                config_key, 3
            )  # Default to LOW (3)
            board_priority_boost = (3 - board_priority) * 2  # 0->6, 1->4, 2->2, 3->0

            cfg["effective_priority"] = (
                cfg.get("priority", 1)
                + elo_boost
                + curriculum_boost
                + board_priority_boost
            )

        # Build weighted list by effective priority
        weighted = []
        for cfg in selfplay_configs:
            # Ensure minimum priority of 1
            priority = max(1, cfg.get("effective_priority", 1))
            weighted.extend([cfg] * priority)

        return random.choice(weighted) if weighted else None

    def get_target_jobs_for_node(self, node: NodeInfo) -> int:
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
        # Check backpressure - reduce production when training queue is full
        backpressure_factor = 1.0
        if (
            self.should_stop_production is not None
            and self.should_throttle_production is not None
        ):
            try:
                # Import QueueType here to avoid circular imports
                try:
                    from app.coordination import QueueType

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
            except Exception as e:
                logger.info(f"Backpressure check error: {e}")

        # Minimum memory requirement - skip low-memory machines to avoid OOM
        memory_gb = int(getattr(node, "memory_gb", 0) or 0)
        if memory_gb > 0 and memory_gb < MIN_MEMORY_GB_FOR_TASKS:
            return 0

        # Extract node metrics
        has_gpu = bool(getattr(node, "has_gpu", False))
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

                return int(max(1, target_selfplay))

            except Exception as e:
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

        # Utilization-aware adjustments (target 60-80%)
        gpu_overloaded = gpu_percent > 85 or gpu_mem_percent > 85
        cpu_overloaded = cpu_percent > 80
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

        return int(max(1, target_selfplay))

    def get_hybrid_job_targets(self, node: NodeInfo) -> dict[str, int]:
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
            except Exception as e:
                logger.info(f"Hybrid limits error: {e}")

        # Fallback: No CPU-only jobs, use standard target
        gpu_jobs = self.get_target_jobs_for_node(node)
        return {"gpu_jobs": gpu_jobs, "cpu_only_jobs": 0, "total_jobs": gpu_jobs}

    def should_spawn_cpu_only_jobs(self, node: NodeInfo) -> bool:
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
                return cpu_count >= 128

        # CPU-only nodes always benefit from full CPU utilization
        return True

    def track_diversity(self, config: dict[str, Any]) -> None:
        """Track diversity metrics for a scheduled selfplay game.

        Args:
            config: Selfplay configuration dict with engine_mode, board_type, num_players, etc.
        """
        # Track engine mode
        engine_mode = config.get("engine_mode", "unknown")
        if engine_mode not in self.diversity_metrics.games_by_engine_mode:
            self.diversity_metrics.games_by_engine_mode[engine_mode] = 0
        self.diversity_metrics.games_by_engine_mode[engine_mode] += 1

        # Track board config
        board_key = (
            f"{config.get('board_type', 'unknown')}_{config.get('num_players', 0)}p"
        )
        if board_key not in self.diversity_metrics.games_by_board_config:
            self.diversity_metrics.games_by_board_config[board_key] = 0
        self.diversity_metrics.games_by_board_config[board_key] += 1

        # Track asymmetric vs symmetric
        if config.get("asymmetric"):
            self.diversity_metrics.asymmetric_games += 1
            strong = config.get("strong_config", {})
            weak = config.get("weak_config", {})
            logger.info(
                f"DIVERSE: Asymmetric game scheduled - "
                f"Strong({strong.get('engine_mode')}@D{strong.get('difficulty')}) vs "
                f"Weak({weak.get('engine_mode')}@D{weak.get('difficulty')}) "
                f"on {board_key}"
            )
        else:
            self.diversity_metrics.symmetric_games += 1

        # Track difficulty if available
        difficulty = config.get("difficulty", config.get("difficulty_band"))
        if difficulty:
            diff_key = str(difficulty)
            if diff_key not in self.diversity_metrics.games_by_difficulty:
                self.diversity_metrics.games_by_difficulty[diff_key] = 0
            self.diversity_metrics.games_by_difficulty[diff_key] += 1

    def get_diversity_metrics(self) -> dict[str, Any]:
        """Get diversity tracking metrics for monitoring.

        Returns:
            Dictionary with diversity metrics including computed statistics
        """
        return self.diversity_metrics.to_dict()
