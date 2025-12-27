"""Utilization Optimizer for cluster workload management.

Ensures all nodes are productively utilized by:
1. Detecting underutilized nodes
2. Spawning selfplay jobs on idle GPUs/CPUs
3. Balancing workload across the cluster
4. Matching GPU capabilities to appropriate board sizes

Usage:
    from app.coordination.utilization_optimizer import (
        UtilizationOptimizer,
        get_utilization_optimizer,
    )

    optimizer = get_utilization_optimizer()
    await optimizer.optimize_cluster()
"""

from __future__ import annotations

__all__ = [  # noqa: RUF022
    # Enums
    "BoardType",
    "WorkloadType",
    # Data classes
    "NodeWorkload",
    "OptimizationResult",
    "WorkloadConfig",
    # Constants
    "BOARD_PRIORITIES",
    "GPU_CAPABILITIES",
    # Main class and singleton
    "UtilizationOptimizer",
    "get_utilization_optimizer",
]

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from app.coordination.health_check_orchestrator import (
    HealthCheckOrchestrator,
    NodeHealthDetails,
    get_health_orchestrator,
)

# December 2025: Use canonical BoardType from coordination.types
from app.coordination.types import BoardType
from app.providers import Provider

logger = logging.getLogger(__name__)


class WorkloadType(str, Enum):
    """Types of workloads that can be assigned."""

    SELFPLAY = "selfplay"
    TRAINING = "training"
    EVALUATION = "evaluation"
    SYNC = "sync"


@dataclass
class WorkloadConfig:
    """Configuration for a workload."""

    workload_type: WorkloadType
    board_type: BoardType
    num_players: int = 2
    engine: str = "gumbel"
    num_games: int = 1000
    priority: int = 50  # 0-100, higher = more important


@dataclass
class NodeWorkload:
    """Current workload state of a node."""

    node_id: str
    selfplay_jobs: int = 0
    training_running: bool = False
    current_board_type: BoardType | None = None
    current_engine: str | None = None
    games_generated: int = 0
    utilization_score: float = 0.0  # 0-100


@dataclass
class OptimizationResult:
    """Result of an optimization action."""

    node_id: str
    action: str
    success: bool
    message: str
    workload: WorkloadConfig | None = None
    timestamp: datetime = field(default_factory=datetime.now)


# GPU capability mapping - determines which boards a GPU can handle efficiently
# This is the static default; dynamic profiles from distributed_hosts.yaml augment this
GPU_CAPABILITIES = {
    # Small GPUs (8-12GB) - hex8, square8
    "RTX 3060": {"max_board": BoardType.SQUARE8, "preferred": BoardType.HEX8},
    "RTX 3060 Ti": {"max_board": BoardType.SQUARE8, "preferred": BoardType.HEX8},
    "RTX 3070": {"max_board": BoardType.SQUARE8, "preferred": BoardType.HEX8},
    "RTX 2060": {"max_board": BoardType.HEX8, "preferred": BoardType.HEX8},
    "RTX 2060 SUPER": {"max_board": BoardType.SQUARE8, "preferred": BoardType.HEX8},
    "RTX 2080 Ti": {"max_board": BoardType.SQUARE8, "preferred": BoardType.HEX8},
    # Medium GPUs (16-24GB) - up to square19
    "RTX 4060 Ti": {"max_board": BoardType.SQUARE19, "preferred": BoardType.SQUARE8},
    "RTX 4070": {"max_board": BoardType.SQUARE19, "preferred": BoardType.SQUARE8},
    "RTX 4080": {"max_board": BoardType.SQUARE19, "preferred": BoardType.SQUARE19},
    "RTX 4080 SUPER": {"max_board": BoardType.SQUARE19, "preferred": BoardType.SQUARE19},
    "RTX 5070": {"max_board": BoardType.HEXAGONAL, "preferred": BoardType.HEXAGONAL},
    "RTX 5080": {"max_board": BoardType.SQUARE19, "preferred": BoardType.SQUARE19},
    "A10": {"max_board": BoardType.HEXAGONAL, "preferred": BoardType.HEXAGONAL},
    # Large GPUs (40-96GB) - all boards
    "A40": {"max_board": BoardType.HEXAGONAL, "preferred": BoardType.HEXAGONAL},
    "RTX 5090": {"max_board": BoardType.HEXAGONAL, "preferred": BoardType.HEXAGONAL},
    "H100": {"max_board": BoardType.HEXAGONAL, "preferred": BoardType.HEXAGONAL},
    "GH200": {"max_board": BoardType.HEXAGONAL, "preferred": BoardType.HEXAGONAL},
    "A100": {"max_board": BoardType.HEXAGONAL, "preferred": BoardType.HEXAGONAL},
    # New GPUs added dynamically (December 2025)
    "L40S": {"max_board": BoardType.HEXAGONAL, "preferred": BoardType.HEXAGONAL},
    "RTX 4090": {"max_board": BoardType.HEXAGONAL, "preferred": BoardType.HEXAGONAL},
}


def _infer_capability_from_vram(vram_gb: float) -> dict[str, BoardType]:
    """Infer GPU capability from VRAM size.

    Args:
        vram_gb: GPU VRAM in gigabytes

    Returns:
        Dict with max_board and preferred BoardType
    """
    if vram_gb >= 40:
        return {"max_board": BoardType.HEXAGONAL, "preferred": BoardType.HEXAGONAL}
    elif vram_gb >= 20:
        return {"max_board": BoardType.SQUARE19, "preferred": BoardType.SQUARE19}
    elif vram_gb >= 12:
        return {"max_board": BoardType.SQUARE19, "preferred": BoardType.SQUARE8}
    elif vram_gb >= 8:
        return {"max_board": BoardType.SQUARE8, "preferred": BoardType.HEX8}
    else:
        return {"max_board": BoardType.HEX8, "preferred": BoardType.HEX8}


def load_gpu_profiles_from_config() -> dict[str, dict[str, BoardType]]:
    """Load GPU profiles dynamically from distributed_hosts.yaml.

    December 2025: Consolidated to use cluster_config.py helpers.

    This augments the static GPU_CAPABILITIES with host-specific profiles
    from the cluster configuration, ensuring new GPU types are automatically
    supported based on their VRAM specs.

    Returns:
        Dict mapping GPU names to capability dicts
    """
    profiles: dict[str, dict[str, BoardType]] = {}

    try:
        from app.config.cluster_config import get_gpu_types

        gpu_types = get_gpu_types()

        for gpu_name, gpu_vram in gpu_types.items():
            # Skip if already in static capabilities
            if gpu_name in GPU_CAPABILITIES:
                continue

            # Skip GPUs with no VRAM info
            if gpu_vram <= 0:
                continue

            # Infer capability from VRAM
            capability = _infer_capability_from_vram(gpu_vram)
            profiles[gpu_name] = capability

            logger.debug(
                f"[UtilizationOptimizer] Added dynamic GPU profile: {gpu_name} "
                f"(vram={gpu_vram}GB) -> max={capability['max_board'].value}"
            )

        if profiles:
            logger.info(
                f"[UtilizationOptimizer] Loaded {len(profiles)} dynamic GPU profiles "
                f"from cluster_config"
            )

    except ImportError:
        logger.debug("[UtilizationOptimizer] cluster_config not available, using static profiles")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[UtilizationOptimizer] Failed to load dynamic GPU profiles: {e}")

    return profiles


# Load dynamic profiles on module import
_DYNAMIC_GPU_PROFILES: dict[str, dict[str, BoardType]] = {}


def get_gpu_capabilities() -> dict[str, dict[str, Any]]:
    """Get combined static and dynamic GPU capabilities.

    Returns:
        Dict mapping GPU names to capability dicts
    """
    global _DYNAMIC_GPU_PROFILES

    # Lazy load dynamic profiles
    if not _DYNAMIC_GPU_PROFILES:
        _DYNAMIC_GPU_PROFILES = load_gpu_profiles_from_config()

    # Combine static and dynamic profiles (static takes precedence)
    combined = {**_DYNAMIC_GPU_PROFILES, **GPU_CAPABILITIES}
    return combined

# Default priority for board types (what we need most training data for)
BOARD_PRIORITIES = {
    BoardType.HEX8: 60,  # Well trained, still useful
    BoardType.SQUARE8: 70,  # Main competitive format
    BoardType.SQUARE19: 80,  # Needs more data
    BoardType.HEXAGONAL: 90,  # Needs most data
}


class UtilizationOptimizer:
    """Optimizes cluster utilization by assigning workloads to idle nodes.

    Features:
    - Matches GPU capabilities to appropriate board sizes
    - Prioritizes underserved board types
    - Respects AWS staging node (light utilization only)
    - Balances load across providers
    - Routes GPU selfplay to GPU-rich nodes (Lambda)
    - Routes CPU selfplay to CPU-rich nodes (Vast, Hetzner)
    """

    # Utilization thresholds
    UNDERUTILIZED_GPU_THRESHOLD = 20.0  # GPU < 20% = underutilized
    UNDERUTILIZED_CPU_THRESHOLD = 30.0  # CPU < 30% = underutilized
    TARGET_GPU_UTILIZATION = 80.0  # Aim for 80% GPU usage
    TARGET_CPU_UTILIZATION = 70.0  # Aim for 70% CPU usage

    # AWS staging node limits
    AWS_MAX_SELFPLAY_JOBS = 2  # Keep AWS lightly loaded

    # Engine selection by provider - Lambda gets GPU selfplay, others get CPU
    GPU_SELFPLAY_PROVIDERS = {Provider.LAMBDA}  # Providers with high-end GPUs for GPU selfplay
    CPU_SELFPLAY_PROVIDERS = {Provider.VAST, Provider.HETZNER}  # CPU-rich providers

    def __init__(
        self,
        health_orchestrator: HealthCheckOrchestrator | None = None,
    ):
        """Initialize utilization optimizer.

        Args:
            health_orchestrator: Health check orchestrator instance
        """
        self.health_orchestrator = health_orchestrator or get_health_orchestrator()

        # Track workload state
        self.node_workloads: dict[str, NodeWorkload] = {}

        # Track what boards need more data
        self.board_data_needs: dict[BoardType, int] = {
            BoardType.HEX8: 50,
            BoardType.SQUARE8: 60,
            BoardType.SQUARE19: 80,
            BoardType.HEXAGONAL: 90,
        }

    def _get_gpu_capability(self, health: NodeHealthDetails) -> dict[str, Any] | None:
        """Get GPU capability info for a node.

        Uses combined static and dynamic profiles from distributed_hosts.yaml.
        Falls back to VRAM-based inference for unknown GPUs.
        """
        if not health.instance:
            return None

        gpu_type = health.instance.gpu_type or ""

        # Get combined static and dynamic capabilities
        capabilities = get_gpu_capabilities()

        # Try to match GPU type against known profiles
        for gpu_name, caps in capabilities.items():
            if gpu_name.lower() in gpu_type.lower():
                return caps

        # Fallback: infer capability from VRAM for unknown GPUs
        # This ensures new GPU types work automatically
        vram_gb = health.instance.gpu_memory_gb or 0
        if vram_gb > 0:
            return _infer_capability_from_vram(vram_gb)

        # Ultimate fallback for unknown GPUs with no VRAM info
        return {"max_board": BoardType.SQUARE8, "preferred": BoardType.HEX8}

    def _select_board_for_node(self, health: NodeHealthDetails) -> BoardType:
        """Select the best board type for a node based on its capabilities."""
        caps = self._get_gpu_capability(health)
        if not caps:
            return BoardType.SQUARE8  # Safe default

        max_board = caps["max_board"]
        preferred = caps["preferred"]

        # Check data needs - if we need more data for larger boards, use those
        # Filter to boards this GPU can handle
        valid_boards = [b for b in BoardType if self._board_size_order(b) <= self._board_size_order(max_board)]

        # Weight by data need
        weighted = [(b, self.board_data_needs.get(b, 50)) for b in valid_boards]
        weighted.sort(key=lambda x: -x[1])  # Highest need first

        # 70% chance to pick highest need, 30% chance for preferred
        if random.random() < 0.7 and weighted:
            return weighted[0][0]
        return preferred

    def _board_size_order(self, board: BoardType) -> int:
        """Get size ordering for boards."""
        order = {
            BoardType.HEX8: 1,
            BoardType.SQUARE8: 2,
            BoardType.SQUARE19: 3,
            BoardType.HEXAGONAL: 4,
        }
        return order.get(board, 2)

    def _select_engine_for_node(self, health: NodeHealthDetails) -> str:
        """Select the appropriate selfplay engine for a node.

        GPU-rich Lambda nodes should run GPU-based Gumbel MCTS.
        CPU-rich Vast/Hetzner nodes should run CPU-based heuristic.

        Returns:
            Engine name: "gumbel" for GPU nodes, "heuristic" for CPU nodes
        """
        if health.provider in self.GPU_SELFPLAY_PROVIDERS:
            # Lambda nodes with high-end GPUs - use GPU selfplay
            return "gumbel"
        elif health.provider in self.CPU_SELFPLAY_PROVIDERS:
            # Vast/Hetzner - CPU-rich but GPU is often consumer-grade
            return "heuristic"
        elif health.provider == Provider.AWS:
            # AWS staging - light duty, use heuristic
            return "heuristic"
        else:
            # Unknown - default to heuristic for safety
            return "heuristic"

    async def get_underutilized_nodes(self) -> list[tuple[str, NodeHealthDetails]]:
        """Get list of underutilized nodes.

        Returns:
            List of (node_id, health) tuples for underutilized nodes
        """
        underutilized = []

        for node_id, health in self.health_orchestrator.node_health.items():
            if not health.is_available():
                continue

            # Check utilization
            is_underutilized = False

            if health.instance and health.instance.gpu_count > 0:
                # GPU node
                if health.gpu_percent < self.UNDERUTILIZED_GPU_THRESHOLD:
                    is_underutilized = True
            else:
                # CPU-only node
                if health.cpu_percent < self.UNDERUTILIZED_CPU_THRESHOLD:
                    is_underutilized = True

            if is_underutilized:
                underutilized.append((node_id, health))

        return underutilized

    async def spawn_selfplay_job(
        self,
        node_id: str,
        board_type: BoardType | None = None,
        num_players: int = 2,
        engine: str = "gumbel",
        num_games: int = 1000,
    ) -> OptimizationResult:
        """Spawn a selfplay job on a node.

        Args:
            node_id: Target node
            board_type: Board type (auto-selected if None)
            num_players: Number of players
            engine: Selfplay engine (gumbel, descent, mcts)
            num_games: Number of games to generate

        Returns:
            OptimizationResult with outcome
        """
        health = self.health_orchestrator.get_node_health(node_id)
        if not health:
            return OptimizationResult(
                node_id=node_id,
                action="spawn_selfplay",
                success=False,
                message="Node not found",
            )

        if not health.is_available():
            return OptimizationResult(
                node_id=node_id,
                action="spawn_selfplay",
                success=False,
                message=f"Node not available (state={health.state.value})",
            )

        # AWS limits
        if health.provider == Provider.AWS:
            workload = self.node_workloads.get(node_id)
            if workload and workload.selfplay_jobs >= self.AWS_MAX_SELFPLAY_JOBS:
                return OptimizationResult(
                    node_id=node_id,
                    action="spawn_selfplay",
                    success=False,
                    message="AWS node at max selfplay jobs",
                )

        # Auto-select board type
        if board_type is None:
            board_type = self._select_board_for_node(health)

        # Determine work directory based on provider
        if health.provider == Provider.VAST:
            work_dir = "/root/ringrift/ai-service"
        else:
            work_dir = "~/ringrift/ai-service"

        # Build selfplay command
        db_name = f"selfplay_{board_type.value}_{num_players}p_{node_id}"
        cmd = f"""
cd {work_dir}
mkdir -p data/games logs
source venv/bin/activate 2>/dev/null || true
PYTHONPATH=. RINGRIFT_DISABLE_TORCH_COMPILE=1 nohup python3 scripts/selfplay.py \\
    --board {board_type.value} --num-players {num_players} \\
    --num-games {num_games} --engine {engine} \\
    --output-dir data/games/{db_name} \\
    > logs/selfplay_{board_type.value}.log 2>&1 &
sleep 2
pgrep -f selfplay && echo "SELFPLAY_STARTED"
"""

        # Get manager and run
        manager = self._get_manager_for_provider(health.provider)
        if not manager:
            return OptimizationResult(
                node_id=node_id,
                action="spawn_selfplay",
                success=False,
                message=f"No manager for provider {health.provider}",
            )

        code, stdout, stderr = await manager.run_ssh_command(
            health.instance, cmd, timeout=60
        )

        if code == 0 and "SELFPLAY_STARTED" in stdout:
            # Update workload tracking
            if node_id not in self.node_workloads:
                self.node_workloads[node_id] = NodeWorkload(node_id=node_id)
            self.node_workloads[node_id].selfplay_jobs += 1
            self.node_workloads[node_id].current_board_type = board_type
            self.node_workloads[node_id].current_engine = engine

            return OptimizationResult(
                node_id=node_id,
                action="spawn_selfplay",
                success=True,
                message=f"Started {engine} selfplay on {board_type.value}_{num_players}p",
                workload=WorkloadConfig(
                    workload_type=WorkloadType.SELFPLAY,
                    board_type=board_type,
                    num_players=num_players,
                    engine=engine,
                    num_games=num_games,
                ),
            )

        return OptimizationResult(
            node_id=node_id,
            action="spawn_selfplay",
            success=False,
            message=f"Failed to start selfplay: {stderr or stdout}",
        )

    def _get_manager_for_provider(self, provider: Provider):
        """Get manager for provider."""
        from app.providers import (
            AWSManager,
            HetznerManager,
            LambdaManager,
            VastManager,
        )

        managers = {
            Provider.LAMBDA: LambdaManager(),
            Provider.VAST: VastManager(),
            Provider.HETZNER: HetznerManager(),
            Provider.AWS: AWSManager(),
        }
        return managers.get(provider)

    async def stop_cpu_selfplay_on_gpu_nodes(self, node_id: str, health: NodeHealthDetails) -> OptimizationResult:
        """Stop CPU-based selfplay on GPU-capable nodes.

        Lambda GH200/H100 nodes should run GPU selfplay, not heuristic.
        This kills heuristic selfplay to free resources for Gumbel MCTS.
        """
        if health.provider not in self.GPU_SELFPLAY_PROVIDERS:
            return OptimizationResult(
                node_id=node_id,
                action="stop_cpu_selfplay",
                success=True,
                message="Not a GPU provider, skipping",
            )

        manager = self._get_manager_for_provider(health.provider)
        if not manager or not health.instance:
            return OptimizationResult(
                node_id=node_id,
                action="stop_cpu_selfplay",
                success=False,
                message="No manager or instance",
            )

        # Kill heuristic selfplay processes (but not Gumbel MCTS ones)
        cmd = """
pkill -f 'engine-mode.*heuristic' 2>/dev/null || true
pkill -f 'engine.*heuristic' 2>/dev/null || true
pkill -f 'run_self_play_soak' 2>/dev/null || true
sleep 1
echo "CPU_SELFPLAY_STOPPED"
"""
        _code, stdout, stderr = await manager.run_ssh_command(health.instance, cmd, timeout=30)

        if "CPU_SELFPLAY_STOPPED" in stdout:
            logger.info(f"[UtilizationOptimizer] Stopped CPU selfplay on {node_id}")
            return OptimizationResult(
                node_id=node_id,
                action="stop_cpu_selfplay",
                success=True,
                message="Stopped CPU selfplay processes",
            )

        return OptimizationResult(
            node_id=node_id,
            action="stop_cpu_selfplay",
            success=False,
            message=f"Failed: {stderr or stdout}",
        )

    async def optimize_cluster(self) -> list[OptimizationResult]:
        """Optimize the entire cluster.

        1. Stop CPU selfplay on GPU nodes (they should use GPU selfplay)
        2. Find underutilized nodes
        3. Spawn appropriate workloads (GPU/CPU based on provider)

        Returns:
            List of optimization results
        """
        results = []

        # First pass: Stop CPU selfplay on GPU nodes
        for node_id, health in self.health_orchestrator.node_health.items():
            if (
                health.provider in self.GPU_SELFPLAY_PROVIDERS
                and health.is_available()
                and health.gpu_percent < 20
                and health.cpu_percent > 50
            ):
                logger.info(
                    f"[UtilizationOptimizer] Node {node_id} running CPU selfplay "
                    f"(GPU={health.gpu_percent:.0f}%, CPU={health.cpu_percent:.0f}%)"
                )
                result = await self.stop_cpu_selfplay_on_gpu_nodes(node_id, health)
                results.append(result)
                await asyncio.sleep(1)

        # Get underutilized nodes
        underutilized = await self.get_underutilized_nodes()

        if not underutilized:
            logger.info("[UtilizationOptimizer] No underutilized nodes found")
            return results

        logger.info(
            f"[UtilizationOptimizer] Found {len(underutilized)} underutilized nodes"
        )

        # Prioritize by data needs
        for node_id, health in underutilized:
            # Select appropriate workload
            board_type = self._select_board_for_node(health)

            # Select engine based on provider capability
            engine = self._select_engine_for_node(health)

            # Determine num_players (mostly 2p, sometimes 4p for variety)
            num_players = 4 if random.random() < 0.2 else 2

            logger.info(
                f"[UtilizationOptimizer] Spawning {engine} selfplay on {node_id} "
                f"(provider={health.provider.value if health.provider else 'unknown'}, "
                f"board={board_type.value}, players={num_players})"
            )

            # Spawn selfplay with provider-appropriate engine
            result = await self.spawn_selfplay_job(
                node_id=node_id,
                board_type=board_type,
                num_players=num_players,
                engine=engine,
            )
            results.append(result)

            # Small delay between spawns
            await asyncio.sleep(0.5)

        # Summary
        stopped = sum(1 for r in results if r.action == "stop_cpu_selfplay" and r.success)
        spawned = sum(1 for r in results if r.action == "spawn_selfplay" and r.success)
        logger.info(
            f"[UtilizationOptimizer] Optimization complete: "
            f"{stopped} CPU jobs stopped, {spawned} GPU/CPU jobs spawned"
        )

        return results

    async def get_workload_distribution(self) -> dict[str, Any]:
        """Get current workload distribution across the cluster.

        Returns:
            Dict with workload statistics
        """
        distribution = {
            "by_board_type": {b.value: 0 for b in BoardType},
            "by_provider": {},
            "total_selfplay_jobs": 0,
            "total_training_jobs": 0,
            "underutilized_count": 0,
        }

        for node_id, health in self.health_orchestrator.node_health.items():
            provider = health.provider.value if health.provider else "unknown"

            if provider not in distribution["by_provider"]:
                distribution["by_provider"][provider] = {
                    "nodes": 0,
                    "selfplay_jobs": 0,
                    "gpu_utilization": 0,
                }

            distribution["by_provider"][provider]["nodes"] += 1
            distribution["by_provider"][provider]["gpu_utilization"] += health.gpu_percent

            # Check workload tracking
            workload = self.node_workloads.get(node_id)
            if workload:
                distribution["total_selfplay_jobs"] += workload.selfplay_jobs
                distribution["by_provider"][provider]["selfplay_jobs"] += workload.selfplay_jobs
                if workload.current_board_type:
                    distribution["by_board_type"][workload.current_board_type.value] += 1
                if workload.training_running:
                    distribution["total_training_jobs"] += 1

        # Calculate average utilization per provider
        for _provider, stats in distribution["by_provider"].items():
            if stats["nodes"] > 0:
                stats["avg_gpu_utilization"] = stats["gpu_utilization"] / stats["nodes"]

        # Count underutilized
        underutilized = await self.get_underutilized_nodes()
        distribution["underutilized_count"] = len(underutilized)

        return distribution

    def update_board_data_needs(self, board_needs: dict[BoardType, int]) -> None:
        """Update board data needs for workload prioritization.

        Args:
            board_needs: Dict mapping BoardType to priority (0-100)
        """
        self.board_data_needs.update(board_needs)
        logger.info(f"[UtilizationOptimizer] Updated board data needs: {board_needs}")

    def health_check(self) -> "HealthCheckResult":
        """Check utilization optimizer health for CoordinatorProtocol compliance.

        December 2025: Added for unified daemon health monitoring.

        Returns:
            HealthCheckResult with health status and metrics.
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        try:
            # Check if health orchestrator is available
            if self.health_orchestrator is None:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.ERROR,
                    message="Health orchestrator not available",
                )

            # Get tracked node count
            tracked_nodes = len(self.node_workloads)

            # Check if we're tracking any workloads
            if tracked_nodes == 0:
                return HealthCheckResult(
                    healthy=True,
                    status=CoordinatorStatus.DEGRADED,
                    message="No nodes being tracked",
                    details={"tracked_nodes": 0},
                )

            # Count active workloads
            active_selfplay = sum(w.selfplay_jobs for w in self.node_workloads.values())
            training_jobs = sum(1 for w in self.node_workloads.values() if w.training_running)

            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message=f"Tracking {tracked_nodes} nodes, {active_selfplay} selfplay, {training_jobs} training",
                details={
                    "tracked_nodes": tracked_nodes,
                    "active_selfplay_jobs": active_selfplay,
                    "training_jobs": training_jobs,
                    "board_data_needs": {k.value: v for k, v in self.board_data_needs.items()},
                },
            )

        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Health check failed: {e}",
            )


# Global instance
_utilization_optimizer: UtilizationOptimizer | None = None


def get_utilization_optimizer() -> UtilizationOptimizer:
    """Get or create the global utilization optimizer."""
    global _utilization_optimizer

    if _utilization_optimizer is None:
        _utilization_optimizer = UtilizationOptimizer()

    return _utilization_optimizer


async def optimize_cluster() -> list[OptimizationResult]:
    """Optimize cluster utilization.

    Convenience function using global optimizer.
    """
    return await get_utilization_optimizer().optimize_cluster()
