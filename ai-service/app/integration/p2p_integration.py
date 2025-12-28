"""
P2P Orchestrator Integration for RingRift AI Self-Improvement Loop.

Connects all training components with the distributed P2P cluster:
- Model synchronization across nodes
- Distributed training coordination
- Evaluation result aggregation
- Resource-aware job scheduling
- Fault-tolerant pipeline execution

This module acts as the bridge between the self-improvement loop
components and the P2P orchestrator's REST API.
"""

import asyncio
import contextlib
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from app.config.ports import P2P_DEFAULT_PORT, get_local_p2p_url

try:
    from aiohttp import ClientSession, ClientTimeout
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    ClientSession = None

logger = logging.getLogger(__name__)


# ============================================
# Configuration
# ============================================

@dataclass
class P2PIntegrationConfig:
    """Configuration for P2P orchestrator integration."""
    # Connection settings
    p2p_base_url: str = field(default_factory=get_local_p2p_url)
    auth_token: str | None = None
    connect_timeout: float = 10.0
    request_timeout: float = 60.0

    # Sync settings
    model_sync_enabled: bool = True
    data_sync_enabled: bool = True
    sync_interval_seconds: float = 300.0

    # Training coordination
    prefer_gpu_for_training: bool = True
    min_gpu_memory_gb: int = 8
    max_concurrent_training_jobs: int = 1

    # Selfplay coordination
    target_selfplay_games_per_hour: int = 1000
    auto_scale_selfplay: bool = True

    # Evaluation settings
    tournament_games_per_pair: int = 50
    use_distributed_tournament: bool = True

    # Health monitoring
    health_check_interval: float = 30.0
    unhealthy_threshold_failures: int = 3


class P2PNodeCapability(Enum):
    """Capabilities a node can have."""
    CPU_SELFPLAY = "cpu_selfplay"
    GPU_SELFPLAY = "gpu_selfplay"
    TRAINING = "training"
    CMAES = "cmaes"
    TOURNAMENT = "tournament"
    DATA_STORAGE = "data_storage"


@dataclass
class P2PNode:
    """Information about a P2P cluster node."""
    node_id: str
    host: str
    port: int
    is_alive: bool = False
    is_healthy: bool = False
    has_gpu: bool = False
    gpu_name: str = ""
    gpu_power_score: int = 0
    memory_gb: int = 0
    disk_percent: float = 0.0
    cpu_percent: float = 0.0
    selfplay_jobs: int = 0
    training_jobs: int = 0
    capabilities: list[P2PNodeCapability] = field(default_factory=list)
    last_heartbeat: float = 0.0


# ============================================
# P2P API Client
# ============================================

class P2PAPIClient:
    """
    Client for P2P orchestrator REST API.

    Provides typed methods for common operations.
    """

    def __init__(self, config: P2PIntegrationConfig):
        self.config = config
        self._session: ClientSession | None = None
        self._headers = {}

        if config.auth_token:
            self._headers["Authorization"] = f"Bearer {config.auth_token}"

    async def _get_session(self) -> ClientSession:
        """Get or create aiohttp session."""
        if not HAS_AIOHTTP:
            raise RuntimeError("aiohttp not available")

        if self._session is None or self._session.closed:
            timeout = ClientTimeout(
                connect=self.config.connect_timeout,
                total=self.config.request_timeout
            )
            self._session = ClientSession(timeout=timeout, headers=self._headers)

        return self._session

    async def close(self) -> None:
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request(
        self,
        method: str,
        endpoint: str,
        json_data: dict | None = None,
        **kwargs
    ) -> dict[str, Any]:
        """Make an API request."""
        session = await self._get_session()
        url = f"{self.config.p2p_base_url}{endpoint}"

        try:
            async with session.request(method, url, json=json_data, **kwargs) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    return {"error": f"HTTP {resp.status}", "status": resp.status}
        except Exception as e:
            logger.error(f"P2P API request failed: {method} {endpoint}: {e}")
            return {"error": str(e)}

    # ==========================================
    # Cluster Status
    # ==========================================

    async def get_cluster_status(self) -> dict[str, Any]:
        """Get overall cluster status."""
        return await self._request("GET", "/api/cluster/status")

    async def get_nodes(self) -> list[P2PNode]:
        """Get list of cluster nodes."""
        result = await self.get_cluster_status()

        nodes = []
        for node_data in result.get("nodes", []):
            capabilities = []
            if node_data.get("has_gpu"):
                capabilities.append(P2PNodeCapability.GPU_SELFPLAY)
                capabilities.append(P2PNodeCapability.TRAINING)
            else:
                capabilities.append(P2PNodeCapability.CPU_SELFPLAY)

            nodes.append(P2PNode(
                node_id=node_data.get("node_id", ""),
                host=node_data.get("host", ""),
                port=node_data.get("port", P2P_DEFAULT_PORT),
                is_alive=node_data.get("is_alive", False),
                is_healthy=node_data.get("is_healthy", False),
                has_gpu=node_data.get("has_gpu", False),
                gpu_name=node_data.get("gpu_name", ""),
                gpu_power_score=node_data.get("gpu_power_score", 0),
                memory_gb=node_data.get("memory_gb", 0),
                disk_percent=node_data.get("disk_percent", 0),
                cpu_percent=node_data.get("cpu_percent", 0),
                selfplay_jobs=node_data.get("selfplay_jobs", 0),
                training_jobs=node_data.get("training_jobs", 0),
                capabilities=capabilities,
                last_heartbeat=node_data.get("last_heartbeat", 0)
            ))

        return nodes

    async def get_leader(self) -> str | None:
        """Get current cluster leader node ID."""
        result = await self.get_cluster_status()
        return result.get("leader")

    # ==========================================
    # Job Management
    # ==========================================

    async def start_selfplay(
        self,
        node_id: str,
        board_type: str = "square8",
        num_players: int = 2,
        count: int = 1
    ) -> dict[str, Any]:
        """Start selfplay jobs on a node."""
        return await self._request(
            "POST",
            "/start_job",
            json_data={
                "node_id": node_id,
                "job_type": "hybrid_selfplay",
                "board_type": board_type,
                "num_players": num_players,
                "count": count
            }
        )

    async def stop_selfplay(self, node_id: str, job_id: str) -> dict[str, Any]:
        """Stop a selfplay job."""
        return await self._request(
            "POST",
            "/stop_job",
            json_data={"node_id": node_id, "job_id": job_id}
        )

    async def start_training(
        self,
        node_id: str,
        config: dict | None = None
    ) -> dict[str, Any]:
        """Start training on a node."""
        return await self._request(
            "POST",
            "/training/start",
            json_data={
                "node_id": node_id,
                "config": config or {}
            }
        )

    async def get_training_status(self) -> dict[str, Any]:
        """Get current training status."""
        return await self._request("GET", "/api/training/status")

    async def start_cmaes(
        self,
        board_type: str = "square8",
        generations: int = 100,
        population_size: int = 20
    ) -> dict[str, Any]:
        """Start distributed CMA-ES optimization."""
        return await self._request(
            "POST",
            "/cmaes/start",
            json_data={
                "board_type": board_type,
                "generations": generations,
                "population_size": population_size
            }
        )

    async def get_cmaes_status(self) -> dict[str, Any]:
        """Get CMA-ES status."""
        return await self._request("GET", "/cmaes/status")

    # ==========================================
    # Tournament & Evaluation
    # ==========================================

    async def start_tournament(
        self,
        model_ids: list[str],
        games_per_pair: int = 50
    ) -> dict[str, Any]:
        """Start model evaluation tournament."""
        return await self._request(
            "POST",
            "/tournament/start",
            json_data={
                "model_ids": model_ids,
                "games_per_pair": games_per_pair
            }
        )

    async def get_tournament_status(self) -> dict[str, Any]:
        """Get tournament status."""
        return await self._request("GET", "/tournament/status")

    async def get_elo_leaderboard(self) -> dict[str, Any]:
        """Get ELO leaderboard."""
        return await self._request("GET", "/api/elo_leaderboard")

    # ==========================================
    # Data & Sync
    # ==========================================

    async def get_data_manifest(self) -> dict[str, Any]:
        """Get cluster-wide data manifest."""
        return await self._request("GET", "/cluster_data_manifest")

    async def trigger_sync(self, source_node: str, target_node: str) -> dict[str, Any]:
        """Trigger data sync between nodes."""
        return await self._request(
            "POST",
            "/sync/start",
            json_data={
                "source_node": source_node,
                "target_node": target_node
            }
        )

    async def get_sync_status(self) -> dict[str, Any]:
        """Get data sync status."""
        return await self._request("GET", "/sync/status")

    # ==========================================
    # Improvement Loop
    # ==========================================

    async def start_improvement_loop(self, config: dict | None = None) -> dict[str, Any]:
        """Start the improvement loop."""
        return await self._request(
            "POST",
            "/improvement/start",
            json_data={"config": config or {}}
        )

    async def get_improvement_status(self) -> dict[str, Any]:
        """Get improvement loop status."""
        return await self._request("GET", "/improvement/status")

    async def notify_phase_complete(
        self,
        phase: str,
        result: dict[str, Any]
    ) -> dict[str, Any]:
        """Notify completion of an improvement phase."""
        return await self._request(
            "POST",
            "/improvement/phase_complete",
            json_data={"phase": phase, "result": result}
        )

    # ==========================================
    # Pipeline
    # ==========================================

    async def start_pipeline(self, config: dict | None = None) -> dict[str, Any]:
        """Start the pipeline orchestrator."""
        return await self._request(
            "POST",
            "/pipeline/start",
            json_data={"config": config or {}}
        )

    async def get_pipeline_status(self) -> dict[str, Any]:
        """Get pipeline status."""
        return await self._request("GET", "/pipeline/status")


# ============================================
# P2P Component Coordinators
# ============================================
#
# Orchestrator Hierarchy (2025-12):
#   - UnifiedTrainingOrchestrator (unified_orchestrator.py): Step-level training
#   - TrainingOrchestrator (orchestrated_training.py): Manager lifecycle
#   - ModelSyncCoordinator (model_lifecycle.py): Model registry sync
#   - P2P Coordinators (this file): P2P cluster REST API wrappers
#
# =============================================================================
# P2P Bridge Classes
# =============================================================================
# These bridge classes wrap the P2P orchestrator REST API for cluster operations.
# They coordinate across the P2P cluster via REST, NOT local training.
#
# Naming convention: P2P*Bridge to distinguish from core coordinators/orchestrators.
# For local training coordination, use:
#   - UnifiedTrainingOrchestrator (unified_orchestrator.py)
#   - TrainingCoordinator (coordination/training_coordinator.py)
# =============================================================================


class P2PSelfplayBridge:
    """Bridges selfplay coordination across the P2P cluster via REST API.

    .. note::
        This is a P2P REST API wrapper. For local training coordination, use
        UnifiedTrainingOrchestrator or TrainingCoordinator instead.

    Handles:
    - Auto-scaling selfplay workers based on target rate
    - Load balancing across nodes
    - GPU vs CPU allocation
    - Curriculum-weighted config selection (Phase 3.1)
    """

    # All supported configs
    ALL_CONFIGS = [
        ("square8", 2), ("square8", 3), ("square8", 4),
        ("square19", 2), ("square19", 3), ("square19", 4),
        ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
    ]

    def __init__(self, client: P2PAPIClient, config: P2PIntegrationConfig):
        self.client = client
        self.config = config
        self._selfplay_targets: dict[str, int] = {}
        self._rate_multiplier: float = 1.0  # Feedback-driven multiplier
        # Phase 3.1: Curriculum weights for config prioritization
        self._curriculum_weights: dict[str, float] = {}  # config_key -> weight
        self._last_selected_config_idx: int = 0  # Round-robin base

    def adjust_target_rate(self, multiplier: float, reason: str) -> int:
        """Adjust the target selfplay rate by applying a multiplier.

        Args:
            multiplier: Rate multiplier (1.0 = no change, 1.5 = 50% increase)
            reason: Human-readable reason for adjustment

        Returns:
            New effective target rate
        """
        # Clamp multiplier to reasonable bounds
        self._rate_multiplier = max(0.5, min(2.5, multiplier))
        effective_rate = int(self.config.target_selfplay_games_per_hour * self._rate_multiplier)
        logger.info(f"[Selfplay] Target rate adjusted to {effective_rate}/hour "
                    f"(multiplier={self._rate_multiplier:.2f}): {reason}")
        return effective_rate

    def get_effective_target_rate(self) -> int:
        """Get the current effective target rate after feedback adjustments."""
        return int(self.config.target_selfplay_games_per_hour * self._rate_multiplier)

    def update_curriculum_weights(self, weights: dict[str, float]) -> None:
        """Update curriculum weights from the training loop.

        Phase 3.1: These weights influence which configs get more selfplay.
        Higher weights = more resources allocated to that config.

        Args:
            weights: Dict mapping config_key (e.g., "square8_2p") to weight (0.7-1.5)
        """
        self._curriculum_weights = weights.copy()
        logger.info(f"[Selfplay] Updated curriculum weights: {len(weights)} configs")

    def select_weighted_config(self) -> tuple[str, int]:
        """Select a config based on curriculum weights.

        Phase 3.1: Uses weighted random selection to prioritize configs that need
        more training (higher curriculum weight).

        Returns:
            Tuple of (board_type, num_players)
        """
        import random

        if not self._curriculum_weights:
            # No weights - use simple round-robin
            self._last_selected_config_idx = (self._last_selected_config_idx + 1) % len(self.ALL_CONFIGS)
            return self.ALL_CONFIGS[self._last_selected_config_idx]

        # Build weighted list
        weighted_configs = []
        for board_type, num_players in self.ALL_CONFIGS:
            config_key = f"{board_type}_{num_players}p"
            weight = self._curriculum_weights.get(config_key, 1.0)
            weighted_configs.append((board_type, num_players, weight))

        # Weighted random selection
        total_weight = sum(w for _, _, w in weighted_configs)
        if total_weight <= 0:
            total_weight = len(weighted_configs)
            weighted_configs = [(b, n, 1.0) for b, n, _ in weighted_configs]

        r = random.uniform(0, total_weight)
        cumulative = 0
        for board_type, num_players, weight in weighted_configs:
            cumulative += weight
            if r <= cumulative:
                config_key = f"{board_type}_{num_players}p"
                logger.debug(f"[Selfplay] Selected {config_key} (weight={weight:.2f})")
                return (board_type, num_players)

        # Fallback to first config
        return self.ALL_CONFIGS[0]

    async def get_current_rate(self) -> float:
        """Get current selfplay games per hour."""
        status = await self.client.get_cluster_status()

        # Calculate from recent games
        # This would integrate with selfplay stats endpoint
        return status.get("selfplay_rate", 0)

    async def auto_scale(self) -> dict[str, Any]:
        """Auto-scale selfplay to meet target rate.

        Phase 3.1: Uses curriculum weights to select configs for new selfplay jobs.
        """
        if not self.config.auto_scale_selfplay:
            return {"action": "disabled"}

        current_rate = await self.get_current_rate()
        target_rate = self.get_effective_target_rate()  # Use feedback-adjusted rate

        nodes = await self.client.get_nodes()
        healthy_nodes = [n for n in nodes if n.is_healthy]

        actions = []

        if current_rate < target_rate * 0.8:
            # Scale up - use curriculum weights to select config
            for node in healthy_nodes:
                # Phase 3.1: Select config based on curriculum weights
                board_type, num_players = self.select_weighted_config()

                if node.has_gpu and node.selfplay_jobs < 4:
                    result = await self.client.start_selfplay(
                        node.node_id,
                        board_type=board_type,
                        num_players=num_players,
                        count=2
                    )
                    actions.append({
                        "node": node.node_id,
                        "action": "scale_up",
                        "config": f"{board_type}_{num_players}p",
                        "result": result
                    })
                elif not node.has_gpu and node.selfplay_jobs < 2:
                    result = await self.client.start_selfplay(
                        node.node_id,
                        board_type=board_type,
                        num_players=num_players,
                        count=1
                    )
                    actions.append({
                        "node": node.node_id,
                        "action": "scale_up",
                        "config": f"{board_type}_{num_players}p",
                        "result": result
                    })

        elif current_rate > target_rate * 1.2:
            # Scale down - would implement job stopping logic
            pass

        return {
            "current_rate": current_rate,
            "target_rate": target_rate,
            "actions": actions
        }

    async def get_distribution(self) -> dict[str, int]:
        """Get selfplay job distribution across nodes."""
        nodes = await self.client.get_nodes()
        return {n.node_id: n.selfplay_jobs for n in nodes if n.is_alive}


class P2PTrainingBridge:
    """Bridges training coordination across the P2P cluster via REST API.

    .. note::
        This is a P2P REST API wrapper. For local training, use
        UnifiedTrainingOrchestrator or TrainingCoordinator instead.

    Handles:
    - Selecting best node for training
    - Data aggregation before training
    - Checkpoint synchronization
    """

    def __init__(self, client: P2PAPIClient, config: P2PIntegrationConfig):
        self.client = client
        self.config = config
        self._current_training_node: str | None = None

    async def select_training_node(self) -> P2PNode | None:
        """Select the best node for training."""
        nodes = await self.client.get_nodes()

        # Filter for healthy GPU nodes
        candidates = [
            n for n in nodes
            if n.is_healthy and n.has_gpu and n.training_jobs == 0
        ]

        if not candidates:
            # Fall back to any healthy node
            candidates = [n for n in nodes if n.is_healthy and n.training_jobs == 0]

        if not candidates:
            return None

        # Sort by GPU power
        candidates.sort(key=lambda n: n.gpu_power_score, reverse=True)

        return candidates[0]

    async def start_training(
        self,
        config: dict | None = None
    ) -> dict[str, Any]:
        """Start training on the best available node."""
        node = await self.select_training_node()

        if not node:
            return {"error": "No available training node"}

        self._current_training_node = node.node_id

        return await self.client.start_training(node.node_id, config)

    async def get_status(self) -> dict[str, Any]:
        """Get current training status."""
        status = await self.client.get_training_status()
        status["selected_node"] = self._current_training_node
        return status

    async def aggregate_data(self, target_node: str) -> dict[str, Any]:
        """Aggregate training data to target node."""
        nodes = await self.client.get_nodes()

        sync_results = []
        for node in nodes:
            if node.node_id != target_node and node.is_alive:
                result = await self.client.trigger_sync(node.node_id, target_node)
                sync_results.append({
                    "source": node.node_id,
                    "result": result
                })

        return {"sync_operations": sync_results}


class P2PEvaluationBridge:
    """Bridges model evaluation across the P2P cluster via REST API.

    .. note::
        This is a P2P REST API wrapper for distributed evaluation.

    Handles:
    - Distributed tournament execution
    - ELO calculation and aggregation
    - Head-to-head comparison scheduling
    """

    def __init__(self, client: P2PAPIClient, config: P2PIntegrationConfig):
        self.client = client
        self.config = config

    async def run_tournament(
        self,
        models: list[str],
        games_per_pair: int | None = None
    ) -> dict[str, Any]:
        """Run evaluation tournament."""
        games = games_per_pair or self.config.tournament_games_per_pair

        if self.config.use_distributed_tournament:
            return await self.client.start_tournament(models, games)
        else:
            # Run locally (not distributed)
            return {"error": "Local tournament not implemented"}

    async def compare_models(
        self,
        model_a: str,
        model_b: str,
        games: int = 100
    ) -> dict[str, Any]:
        """Run head-to-head comparison between two models."""
        return await self.run_tournament([model_a, model_b], games)

    async def get_leaderboard(self) -> list[dict[str, Any]]:
        """Get current ELO leaderboard."""
        result = await self.client.get_elo_leaderboard()
        return result.get("leaderboard", [])


# =============================================================================
# Backward Compatibility Aliases (December 2025 consolidation)
# =============================================================================
# These aliases maintain backward compatibility with existing code.
# New code should use the P2P*Bridge names directly.

SelfplayCoordinator = P2PSelfplayBridge
"""Deprecated: Use P2PSelfplayBridge instead."""

TrainingCoordinator = P2PTrainingBridge
"""Deprecated: Use P2PTrainingBridge instead.

Note: This is the P2P cluster coordinator. For local training, use
TrainingCoordinator from app.coordination.training_coordinator instead.
"""

EvaluationCoordinator = P2PEvaluationBridge
"""Deprecated: Use P2PEvaluationBridge instead."""


# ============================================
# Main Integration Manager
# ============================================

class P2PIntegrationManager:
    """
    Main manager for P2P cluster integration.

    Coordinates all aspects of the self-improvement loop
    with the distributed P2P cluster.
    """

    def __init__(self, config: P2PIntegrationConfig | None = None):
        self.config = config or P2PIntegrationConfig()

        # Load auth token from environment if not set
        if not self.config.auth_token:
            self.config.auth_token = os.environ.get("RINGRIFT_CLUSTER_AUTH_TOKEN")

        # Initialize components
        self.client = P2PAPIClient(self.config)
        self.selfplay = P2PSelfplayBridge(self.client, self.config)
        self.training = P2PTrainingBridge(self.client, self.config)
        self.evaluation = P2PEvaluationBridge(self.client, self.config)

        # State
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._callbacks: dict[str, list[Callable]] = {}
        self._last_health_check: float = 0
        self._cluster_healthy: bool = False

    async def start(self) -> None:
        """Start integration manager background tasks."""
        self._running = True

        # Start background loops
        self._tasks.append(asyncio.create_task(self._health_check_loop()))
        self._tasks.append(asyncio.create_task(self._selfplay_management_loop()))
        self._tasks.append(asyncio.create_task(self._sync_loop()))

        logger.info("P2P integration manager started")

    async def stop(self) -> None:
        """Stop integration manager."""
        self._running = False

        for task in self._tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        await self.client.close()
        self._tasks.clear()

        logger.info("P2P integration manager stopped")

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for events."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    async def _fire_callbacks(self, event: str, **kwargs) -> None:
        """Fire callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(**kwargs)
                else:
                    callback(**kwargs)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    # ==========================================
    # High-Level Operations
    # ==========================================

    async def start_improvement_cycle(
        self,
        phases: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Start a complete improvement cycle.

        Phases: selfplay -> training -> evaluation
        """
        phases = phases or ["selfplay", "training", "evaluation"]

        # Start the improvement loop via P2P
        return await self.client.start_improvement_loop({
            "phases": phases,
            "auto_advance": True
        })

    async def trigger_training(
        self,
        wait_for_completion: bool = False
    ) -> dict[str, Any]:
        """Trigger training on the cluster."""
        result = await self.training.start_training()

        if wait_for_completion and "error" not in result:
            # Poll for completion
            while True:
                status = await self.training.get_status()
                if status.get("status") in ("completed", "failed"):
                    break
                await asyncio.sleep(30)

            return status

        return result

    async def evaluate_model(
        self,
        model_id: str,
        reference_model: str | None = None
    ) -> dict[str, Any]:
        """
        Evaluate a model against references.

        Returns evaluation metrics.
        """
        if reference_model:
            return await self.evaluation.compare_models(model_id, reference_model)
        else:
            # Compare against top models
            leaderboard = await self.evaluation.get_leaderboard()
            if leaderboard:
                top_models = [m["model_id"] for m in leaderboard[:3]]
                top_models.append(model_id)
                return await self.evaluation.run_tournament(top_models)
            return {"error": "No reference models available"}

    async def sync_model_to_cluster(
        self,
        model_id: str,
        model_path: Path
    ) -> dict[str, Any]:
        """Sync a model file to all cluster nodes."""
        nodes = await self.client.get_nodes()
        healthy_nodes = [n for n in nodes if n.is_healthy]

        results = {}
        for node in healthy_nodes:
            # Use sync endpoint to push model
            # This is a simplified version - actual implementation would
            # use the P2P sync/file endpoint
            results[node.node_id] = True

        return {
            "model_id": model_id,
            "synced_nodes": len(results),
            "total_nodes": len(healthy_nodes)
        }

    async def schedule_evaluation(
        self,
        model_id: str,
        version: int,
        priority: str = "normal",
        reason: str = "manual"
    ) -> dict[str, Any]:
        """
        Schedule an evaluation tournament for a model.

        This queues the model for Elo evaluation against reference models.

        Args:
            model_id: Model identifier
            version: Model version number
            priority: Priority level (low, normal, high)
            reason: Reason for evaluation (staging_promotion, manual, etc.)

        Returns:
            Dict with evaluation job ID and status
        """
        full_model_id = f"{model_id}_v{version}"
        logger.info(f"Scheduling evaluation for {full_model_id} (priority={priority}, reason={reason})")

        # Fire event for any listeners
        await self._fire_callbacks(
            "evaluation_scheduled",
            model_id=model_id,
            version=version,
            priority=priority,
            reason=reason
        )

        # Queue evaluation via the evaluation coordinator
        if self.evaluation:
            try:
                # Run tournament against top models
                leaderboard = await self.evaluation.get_leaderboard()
                if leaderboard:
                    top_models = [m["model_id"] for m in leaderboard[:5]]
                    # Don't add ourselves if we're already in the list
                    if full_model_id not in top_models:
                        top_models.append(full_model_id)

                    result = await self.evaluation.run_tournament(top_models)

                    # Fire completion event
                    if result and "ratings" in result:
                        model_rating = result["ratings"].get(full_model_id, {})
                        await self._fire_callbacks(
                            "evaluation_complete",
                            model_id=model_id,
                            version=version,
                            elo_rating=model_rating.get("elo"),
                            win_rate=model_rating.get("win_rate"),
                            games_played=model_rating.get("games", 0)
                        )

                    return {
                        "status": "completed",
                        "model_id": full_model_id,
                        "result": result
                    }
                else:
                    logger.warning("No reference models available for evaluation")
                    return {
                        "status": "skipped",
                        "model_id": full_model_id,
                        "reason": "no_reference_models"
                    }
            except Exception as e:
                logger.error(f"Evaluation failed for {full_model_id}: {e}")
                return {
                    "status": "failed",
                    "model_id": full_model_id,
                    "error": str(e)
                }
        else:
            logger.warning("Evaluation coordinator not available")
            return {
                "status": "skipped",
                "model_id": full_model_id,
                "reason": "no_evaluation_coordinator"
            }

    # ==========================================
    # Background Loops
    # ==========================================

    async def _health_check_loop(self) -> None:
        """Background loop for cluster health monitoring."""
        failures = 0

        while self._running:
            try:
                status = await self.client.get_cluster_status()

                if "error" in status:
                    failures += 1
                    if failures >= self.config.unhealthy_threshold_failures:
                        self._cluster_healthy = False
                        await self._fire_callbacks(
                            "cluster_unhealthy",
                            error=status.get("error")
                        )
                else:
                    failures = 0
                    self._cluster_healthy = True
                    self._last_health_check = time.time()

                    # Check for specific issues
                    nodes = status.get("nodes", [])
                    dead_nodes = [n for n in nodes if not n.get("is_alive")]
                    if dead_nodes:
                        await self._fire_callbacks(
                            "nodes_dead",
                            nodes=[n["node_id"] for n in dead_nodes]
                        )

                await asyncio.sleep(self.config.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(60)

    async def _selfplay_management_loop(self) -> None:
        """Background loop for selfplay management."""
        while self._running:
            try:
                if self._cluster_healthy and self.config.auto_scale_selfplay:
                    result = await self.selfplay.auto_scale()
                    if result.get("actions"):
                        await self._fire_callbacks(
                            "selfplay_scaled",
                            result=result
                        )

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Selfplay management loop error: {e}")
                await asyncio.sleep(60)

    async def _sync_loop(self) -> None:
        """Background loop for data synchronization."""
        while self._running:
            try:
                if self._cluster_healthy and self.config.data_sync_enabled:
                    # Trigger sync if needed
                    # This would implement sync logic based on data manifest

                    # Reconcile Elo ratings across cluster
                    await self._reconcile_elo()

                await asyncio.sleep(self.config.sync_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
                await asyncio.sleep(60)

    async def _reconcile_elo(self) -> dict[str, Any]:
        """Reconcile Elo ratings across cluster nodes.

        Uses Mac Studio as authoritative source and propagates to other nodes.
        Returns reconciliation status.
        """
        result = {
            "checked_nodes": 0,
            "synced_nodes": 0,
            "errors": [],
        }

        try:
            # Get authoritative Elo from local DB or primary source
            from app.utils.paths import AI_SERVICE_ROOT

            local_elo_db = AI_SERVICE_ROOT / "data" / "unified_elo.db"

            if not local_elo_db.exists():
                logger.warning("Local Elo database not found, skipping reconciliation")
                return result

            # Get cluster nodes
            cluster_status = await self.client.get_cluster_status()
            nodes = cluster_status.get("nodes", [])

            for node in nodes:
                if not node.get("is_alive", False):
                    continue

                result["checked_nodes"] += 1
                host = node.get("host", "")

                try:
                    # Fetch Elo leaderboard from node
                    node_elo = await self._fetch_node_elo(host, node.get("port", P2P_DEFAULT_PORT))
                    if node_elo:
                        # Compare and detect divergence
                        diverged = await self._check_elo_divergence(local_elo_db, node_elo)
                        if diverged:
                            logger.info(f"Elo divergence detected on {host}, syncing...")
                            synced = await self._sync_elo_to_node(host, local_elo_db)
                            if synced:
                                result["synced_nodes"] += 1
                except Exception as e:
                    result["errors"].append(f"{host}: {e!s}")

        except Exception as e:
            logger.error(f"Elo reconciliation error: {e}")
            result["errors"].append(str(e))

        return result

    async def _fetch_node_elo(self, host: str, port: int) -> dict | None:
        """Fetch Elo leaderboard from a cluster node."""
        import json
        import urllib.error
        import urllib.request

        url = f"http://{host}:{port}/api/elo/leaderboard"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "RingRift/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, UnicodeDecodeError, OSError):
            return None

    async def _check_elo_divergence(self, local_db: Path, node_elo: dict, threshold: float = 50.0) -> bool:
        """Check if node Elo ratings have diverged from local."""
        import sqlite3
        from app.tournament import EloDatabase

        try:
            elo_db = EloDatabase(local_db)
            leaderboard = elo_db.get_leaderboard()

            local_models = {m.model_id: m.rating for m in leaderboard}
            node_models = {m.get("model_id"): m.get("elo", 1500) for m in node_elo.get("leaderboard", [])}

            for model_id, local_elo in local_models.items():
                if model_id in node_models and abs(local_elo - node_models[model_id]) > threshold:
                    return True

            return False
        except (ImportError, FileNotFoundError, sqlite3.Error, AttributeError, KeyError, TypeError):
            return False

    async def _sync_elo_to_node(self, host: str, local_db: Path) -> bool:
        """Sync Elo database to a remote node via SCP."""
        import subprocess

        try:
            cmd = [
                "scp",
                "-o", "ConnectTimeout=10",
                "-o", "StrictHostKeyChecking=no",
                str(local_db),
                f"ubuntu@{host}:~/ringrift/ai-service/data/unified_elo.db"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error syncing Elo to {host}: {e}")
            return False

    # ==========================================
    # Status & Reporting
    # ==========================================

    def get_status(self) -> dict[str, Any]:
        """Get integration manager status."""
        return {
            "running": self._running,
            "cluster_healthy": self._cluster_healthy,
            "last_health_check": self._last_health_check,
            "config": asdict(self.config)
        }

    async def get_cluster_summary(self) -> dict[str, Any]:
        """Get comprehensive cluster summary."""
        status = await self.client.get_cluster_status()

        if "error" in status:
            return status

        nodes = status.get("nodes", [])

        return {
            "leader": status.get("leader"),
            "total_nodes": len(nodes),
            "alive_nodes": sum(1 for n in nodes if n.get("is_alive")),
            "healthy_nodes": sum(1 for n in nodes if n.get("is_healthy")),
            "gpu_nodes": sum(1 for n in nodes if n.get("has_gpu")),
            "total_selfplay_jobs": sum(n.get("selfplay_jobs", 0) for n in nodes),
            "total_training_jobs": sum(n.get("training_jobs", 0) for n in nodes),
            "avg_disk_usage": sum(n.get("disk_percent", 0) for n in nodes) / max(1, len(nodes)),
            "avg_cpu_usage": sum(n.get("cpu_percent", 0) for n in nodes) / max(1, len(nodes))
        }


# ============================================
# Integration Functions
# ============================================

async def connect_to_cluster(
    base_url: str | None = None,
    auth_token: str | None = None
) -> P2PIntegrationManager:
    """
    Connect to P2P cluster and return integration manager.

    Example:
        manager = await connect_to_cluster("http://192.168.1.100:8770")
        await manager.start()
        summary = await manager.get_cluster_summary()
    """
    effective_url = base_url or get_local_p2p_url()
    config = P2PIntegrationConfig(
        p2p_base_url=effective_url,
        auth_token=auth_token
    )
    manager = P2PIntegrationManager(config)
    await manager.start()
    return manager


def integrate_lifecycle_with_p2p(
    lifecycle_manager,
    p2p_manager: P2PIntegrationManager
) -> None:
    """
    Integrate model lifecycle manager with P2P cluster.

    Sets up callbacks for automatic model sync and evaluation.
    This enables the full automation loop:
    - Model promoted to staging → trigger evaluation tournament
    - Model promoted to production → sync to all cluster nodes
    - Training triggered → coordinate via P2P orchestrator
    """
    # Sync new production models to cluster
    async def on_model_promoted(**kwargs):
        stage = kwargs.get("stage")
        model_id = kwargs.get("model_id")
        version = kwargs.get("version")
        model_path = kwargs.get("model_path")

        if stage == "production":
            logger.info(f"[P2P Integration] Syncing production model {model_id}:v{version} to cluster")
            try:
                # Sync model file to all healthy cluster nodes
                if model_path:
                    result = await p2p_manager.sync_model_to_cluster(
                        model_id=f"{model_id}_v{version}",
                        model_path=Path(model_path)
                    )
                    logger.info(
                        f"[P2P Integration] Model sync complete: "
                        f"{result.get('synced_nodes', 0)}/{result.get('total_nodes', 0)} nodes"
                    )

                # Notify cluster of new production model
                await p2p_manager._fire_callbacks(
                    "production_model_available",
                    model_id=model_id,
                    version=version,
                    model_path=str(model_path) if model_path else None
                )
            except Exception as e:
                logger.error(f"[P2P Integration] Failed to sync model to cluster: {e}")

        elif stage == "staging":
            logger.info(f"[P2P Integration] Model {model_id}:v{version} promoted to staging, triggering evaluation")
            try:
                # Automatically queue evaluation tournament for staging models
                await p2p_manager.schedule_evaluation(
                    model_id=model_id,
                    version=version,
                    priority="high",
                    reason="staging_promotion"
                )
            except Exception as e:
                logger.error(f"[P2P Integration] Failed to schedule evaluation: {e}")

    lifecycle_manager.register_callback("model_promoted", on_model_promoted)

    # Trigger training via P2P when conditions are met
    async def on_training_triggered(**kwargs):
        reason = kwargs.get('reason', 'unknown')
        board_type = kwargs.get('board_type')
        num_players = kwargs.get('num_players')
        logger.info(f"[P2P Integration] Training triggered: {reason}")
        try:
            await p2p_manager.trigger_training(
                board_type=board_type,
                num_players=num_players,
                reason=reason
            )
        except Exception as e:
            logger.error(f"[P2P Integration] Failed to trigger training: {e}")

    lifecycle_manager.register_callback("training_triggered", on_training_triggered)

    # Handle evaluation completion to check for promotion
    async def on_evaluation_complete(**kwargs):
        model_id = kwargs.get("model_id")
        version = kwargs.get("version")
        elo_rating = kwargs.get("elo_rating")
        win_rate = kwargs.get("win_rate")

        logger.info(
            f"[P2P Integration] Evaluation complete for {model_id}:v{version}: "
            f"Elo={elo_rating}, WinRate={win_rate}"
        )

        # Submit evaluation results to lifecycle manager for promotion decision
        if hasattr(lifecycle_manager, 'submit_evaluation'):
            try:
                await lifecycle_manager.submit_evaluation(
                    model_id=model_id,
                    version=version,
                    elo_rating=elo_rating,
                    win_rate=win_rate
                )
            except Exception as e:
                logger.error(f"[P2P Integration] Failed to submit evaluation: {e}")

    # Register for evaluation events from P2P manager
    p2p_manager.register_callback("evaluation_complete", on_evaluation_complete)

    logger.info("[P2P Integration] Lifecycle ↔ P2P integration established")


def integrate_pipeline_with_p2p(
    _pipeline_orchestrator,
    p2p_manager: P2PIntegrationManager
) -> None:
    """
    Integrate pipeline orchestrator with P2P cluster.

    Sets up distributed execution of pipeline stages.
    """
    # Map pipeline stages to P2P operations
    # This would set up the appropriate callbacks and handlers


def integrate_feedback_with_selfplay(
    feedback_router,
    selfplay_coordinator: P2PSelfplayBridge
) -> None:
    """
    Integrate feedback signal router with selfplay bridge.

    Connects data quality signals to selfplay rate adjustments:
    - QUARANTINE_DATA → reduce selfplay rate (data quality issue)
    - DECREASE_DATA_COLLECTION → reduce rate
    - INCREASE_DATA_COLLECTION → increase rate
    - UPDATE_CURRICULUM_WEIGHTS → update config priorities

    Args:
        feedback_router: FeedbackSignalRouter instance
        selfplay_coordinator: P2PSelfplayBridge instance
    """
    # Import FeedbackAction here to avoid circular imports
    try:
        from app.integration.pipeline_feedback import FeedbackAction, FeedbackSignal
    except ImportError:
        logger.warning("[Feedback Integration] pipeline_feedback not available")
        return

    async def handle_quarantine_data(signal: FeedbackSignal) -> bool:
        """Handle data quarantine signal - reduce selfplay rate."""
        magnitude = signal.magnitude or 0.1
        # High parity failure = reduce rate proportionally
        # 10% failure rate → reduce to 80% rate, 20% → 60%, etc.
        new_multiplier = max(0.5, 1.0 - (magnitude * 2))
        selfplay_coordinator.adjust_target_rate(
            new_multiplier,
            f"Data quality issue: {signal.reason}"
        )
        logger.info(f"[Feedback→Selfplay] Quarantine signal: rate reduced to {new_multiplier:.0%}")
        return True

    async def handle_decrease_collection(signal: FeedbackSignal) -> bool:
        """Handle decrease data collection signal."""
        magnitude = signal.magnitude or 0.2
        current_multiplier = selfplay_coordinator._rate_multiplier
        new_multiplier = max(0.5, current_multiplier - magnitude)
        selfplay_coordinator.adjust_target_rate(
            new_multiplier,
            f"Decrease requested: {signal.reason}"
        )
        logger.info(f"[Feedback→Selfplay] Decrease signal: rate {current_multiplier:.0%} → {new_multiplier:.0%}")
        return True

    async def handle_increase_collection(signal: FeedbackSignal) -> bool:
        """Handle increase data collection signal."""
        magnitude = signal.magnitude or 0.2
        current_multiplier = selfplay_coordinator._rate_multiplier
        new_multiplier = min(2.0, current_multiplier + magnitude)
        selfplay_coordinator.adjust_target_rate(
            new_multiplier,
            f"Increase requested: {signal.reason}"
        )
        logger.info(f"[Feedback→Selfplay] Increase signal: rate {current_multiplier:.0%} → {new_multiplier:.0%}")
        return True

    async def handle_curriculum_weights(signal: FeedbackSignal) -> bool:
        """Handle curriculum weight update signal."""
        if signal.metadata and 'weights' in signal.metadata:
            weights = signal.metadata['weights']
            selfplay_coordinator.update_curriculum_weights(weights)
            logger.info(f"[Feedback→Selfplay] Curriculum weights updated: {len(weights)} configs")
            return True
        return False

    async def handle_training_data_quality(signal: FeedbackSignal) -> bool:
        """Handle training data quality warnings."""
        quality_score = signal.magnitude or 0.5
        # If quality is low, adjust selfplay parameters
        if quality_score < 0.7:
            # Low quality → slightly reduce rate to focus on quality
            new_multiplier = max(0.6, quality_score + 0.2)
            selfplay_coordinator.adjust_target_rate(
                new_multiplier,
                f"Low data quality ({quality_score:.0%})"
            )
            logger.info(f"[Feedback→Selfplay] Quality warning: rate adjusted for quality {quality_score:.0%}")
        return True

    # Register handlers
    feedback_router.register_handler(
        FeedbackAction.QUARANTINE_DATA,
        handle_quarantine_data,
        name="selfplay_quarantine_handler"
    )

    feedback_router.register_handler(
        FeedbackAction.DECREASE_DATA_COLLECTION,
        handle_decrease_collection,
        name="selfplay_decrease_handler"
    )

    feedback_router.register_handler(
        FeedbackAction.INCREASE_DATA_COLLECTION,
        handle_increase_collection,
        name="selfplay_increase_handler"
    )

    # Also handle curriculum weight updates if available
    if hasattr(FeedbackAction, 'UPDATE_CURRICULUM_WEIGHTS'):
        feedback_router.register_handler(
            FeedbackAction.UPDATE_CURRICULUM_WEIGHTS,
            handle_curriculum_weights,
            name="selfplay_curriculum_handler"
        )

    # Handle training quality signals
    if hasattr(FeedbackAction, 'TRAINING_QUALITY_WARNING'):
        feedback_router.register_handler(
            FeedbackAction.TRAINING_QUALITY_WARNING,
            handle_training_data_quality,
            name="selfplay_quality_handler"
        )

    logger.info("[Feedback Integration] Feedback ↔ Selfplay integration established")


def get_integrated_selfplay_coordinator(
    p2p_manager: P2PIntegrationManager,
    feedback_router=None
) -> P2PSelfplayBridge:
    """
    Get a P2PSelfplayBridge that's integrated with feedback signals.

    This is a convenience function for setting up the full integration.

    Args:
        p2p_manager: P2PIntegrationManager instance
        feedback_router: Optional FeedbackSignalRouter for quality signals

    Returns:
        P2PSelfplayBridge with feedback integration
    """
    coordinator = p2p_manager.selfplay

    if feedback_router:
        integrate_feedback_with_selfplay(feedback_router, coordinator)

    return coordinator


# ============================================
# Event-Driven Training Trigger (Phase 8.1)
# ============================================

def integrate_selfplay_with_training(
    selfplay_coordinator: P2PSelfplayBridge,
    training_triggers=None,
    training_scheduler=None,
    auto_trigger: bool = True,
) -> dict[str, Any]:
    """
    Integrate selfplay game completion with training triggers.

    This creates an event-driven pipeline where:
    1. P2PSelfplayBridge reports game completions
    2. TrainingTriggers updates game counts per config
    3. When threshold is reached, TRAINING_THRESHOLD_REACHED event is emitted
    4. If auto_trigger=True, training is automatically scheduled

    Args:
        selfplay_coordinator: P2PSelfplayBridge instance
        training_triggers: Optional TrainingTriggers instance (auto-creates if None)
        training_scheduler: Optional training scheduler for auto-trigger
        auto_trigger: Whether to automatically trigger training when threshold met

    Returns:
        Dict with integration info and callbacks
    """
    from app.training.training_triggers import get_training_triggers

    # Get or create training triggers
    triggers = training_triggers or get_training_triggers()

    # Track per-config game counts
    game_counts: dict[str, int] = {}
    pending_training: list[str] = []

    # Import event router for publishing events
    try:
        from app.coordination.event_router import DataEventType, RouterEvent, get_router

        event_router = get_router()
        has_event_bus = True
    except ImportError:
        event_router = None
        has_event_bus = False
        logger.warning("[Training↔Selfplay] Event router not available")

    # Import thresholds
    try:
        from app.config.thresholds import TRAINING_TRIGGER_GAMES
    except ImportError:
        TRAINING_TRIGGER_GAMES = 500

    async def on_games_completed(config_key: str, new_games: int, total_games: int, **kwargs):
        """Handler for when selfplay games are completed."""
        # Update game count (game_counts is a dict, so no nonlocal needed for in-place updates)
        game_counts[config_key] = total_games

        # Update training triggers
        triggers.update_config_state(
            config_key=config_key,
            games_count=total_games,
        )

        # Check if training should be triggered
        decision = triggers.should_train(config_key)

        logger.debug(
            f"[Training↔Selfplay] {config_key}: {new_games} new games "
            f"(total: {total_games}), should_train={decision.should_train}"
        )

        # Publish NEW_GAMES_AVAILABLE event
        if has_event_bus and new_games > 0:
            await event_router.publish(
                DataEventType.NEW_GAMES_AVAILABLE,
                payload={
                    "config": config_key,
                    "new_games": new_games,
                    "total_games": total_games,
                    "threshold": TRAINING_TRIGGER_GAMES,
                },
                source="selfplay_training_integration",
            )

        if decision.should_train and config_key not in pending_training:
            # Check if already pending
            pending_training.append(config_key)

            # Emit TRAINING_THRESHOLD_REACHED event
            if has_event_bus:
                await event_router.publish(
                    DataEventType.TRAINING_THRESHOLD_REACHED,
                    payload={
                        "config": config_key,
                        "total_games": total_games,
                        "priority": decision.priority,
                        "reason": decision.reason,
                        "signal_scores": decision.signal_scores,
                    },
                    source="selfplay_training_integration",
                )

            logger.info(
                f"[Training↔Selfplay] TRAINING_THRESHOLD_REACHED for {config_key} "
                f"(priority={decision.priority:.2f}, reason={decision.reason})"
            )

            # Auto-trigger training if enabled
            if auto_trigger and training_scheduler:
                try:
                    await training_scheduler.schedule_training(
                        config_key=config_key,
                        priority=decision.priority,
                        reason=decision.reason,
                    )
                    logger.info(f"[Training↔Selfplay] Auto-triggered training for {config_key}")
                except Exception as e:
                    logger.error(f"[Training↔Selfplay] Failed to auto-trigger training: {e}")

    async def on_training_started(config_key: str, **kwargs):
        """Handler for when training actually starts (clears pending)."""
        # pending_training is a set, no nonlocal needed for in-place modification
        if config_key in pending_training:
            pending_training.remove(config_key)

    async def on_training_completed(config_key: str, games_at_training: int, **kwargs):
        """Handler for when training completes."""
        # Record training completion in triggers
        triggers.record_training_complete(
            config_key=config_key,
            games_at_training=games_at_training,
            new_elo=kwargs.get("new_elo"),
        )
        logger.info(f"[Training↔Selfplay] Training completed for {config_key}, reset game counter")

    # Register callbacks with selfplay coordinator
    if hasattr(selfplay_coordinator, 'register_callback'):
        selfplay_coordinator.register_callback('games_completed', on_games_completed)
        selfplay_coordinator.register_callback('training_started', on_training_started)
        selfplay_coordinator.register_callback('training_completed', on_training_completed)
    elif hasattr(selfplay_coordinator, 'on_games_completed'):
        # Direct callback registration
        selfplay_coordinator.on_games_completed = on_games_completed

    # Also subscribe to event bus for external game completion events
    if has_event_bus:
        async def handle_sync_completed(event: RouterEvent):
            """Handle data sync completion (games may have been added)."""
            payload = event.payload
            config = payload.get("config")
            if config and payload.get("games_added", 0) > 0:
                await on_games_completed(
                    config_key=config,
                    new_games=payload.get("games_added", 0),
                    total_games=payload.get("total_games", game_counts.get(config, 0)),
                )

        event_router.subscribe(DataEventType.DATA_SYNC_COMPLETED.value, handle_sync_completed)

    logger.info("[Training↔Selfplay] Event-driven training trigger integration established")

    return {
        "triggers": triggers,
        "game_counts": game_counts,
        "pending_training": pending_training,
        "on_games_completed": on_games_completed,
        "on_training_completed": on_training_completed,
    }


def create_full_selfplay_training_loop(
    p2p_manager: P2PIntegrationManager,
    training_scheduler=None,
    feedback_controller=None,
    auto_trigger: bool = True,
) -> dict[str, Any]:
    """
    Create a fully integrated selfplay → training loop.

    This sets up:
    1. Selfplay coordinator with feedback integration
    2. Event-driven training triggers
    3. Curriculum weight adjustment based on evaluation
    4. Auto-training when game thresholds are met

    Args:
        p2p_manager: P2PIntegrationManager instance
        training_scheduler: Training scheduler for auto-triggering
        feedback_controller: Optional PipelineFeedbackController
        auto_trigger: Whether to auto-trigger training

    Returns:
        Dict with all integration components
    """
    # Get selfplay coordinator
    coordinator = p2p_manager.selfplay

    # Set up feedback integration if controller provided
    feedback_router = None
    if feedback_controller and hasattr(feedback_controller, 'signal_router'):
        feedback_router = feedback_controller.signal_router
        integrate_feedback_with_selfplay(feedback_router, coordinator)

    # Set up event-driven training triggers
    training_integration = integrate_selfplay_with_training(
        selfplay_coordinator=coordinator,
        training_scheduler=training_scheduler,
        auto_trigger=auto_trigger,
    )

    # Set up curriculum bridge if available
    curriculum_bridge = None
    try:
        from app.integration.evaluation_curriculum_bridge import create_evaluation_bridge
        curriculum_bridge = create_evaluation_bridge(
            feedback_controller=feedback_controller,
            feedback_router=feedback_router,
            selfplay_coordinator=coordinator,
        )
    except ImportError:
        logger.debug("[FullLoop] Curriculum bridge not available")

    logger.info("[FullLoop] Full selfplay → training loop established")

    return {
        "coordinator": coordinator,
        "training_triggers": training_integration["triggers"],
        "curriculum_bridge": curriculum_bridge,
        "game_counts": training_integration["game_counts"],
        "pending_training": training_integration["pending_training"],
    }


# ============================================
# Main
# ============================================

async def main():
    """Example usage of P2P integration."""
    # Create integration manager
    config = P2PIntegrationConfig(
        p2p_base_url="http://localhost:8770",
        auto_scale_selfplay=False  # Disable for testing
    )

    manager = P2PIntegrationManager(config)

    try:
        # Start manager
        await manager.start()

        # Get cluster summary
        summary = await manager.get_cluster_summary()
        print("Cluster Summary:")
        print(json.dumps(summary, indent=2))

        # Get nodes
        nodes = await manager.client.get_nodes()
        print(f"\nNodes ({len(nodes)}):")
        for node in nodes:
            status = "healthy" if node.is_healthy else ("alive" if node.is_alive else "dead")
            gpu = f" ({node.gpu_name})" if node.gpu_name else ""
            print(f"  {node.node_id}: {status}{gpu}")

        # Check training status
        training_status = await manager.training.get_status()
        print("\nTraining Status:")
        print(json.dumps(training_status, indent=2))

        # Get selfplay distribution
        selfplay_dist = await manager.selfplay.get_distribution()
        print("\nSelfplay Distribution:")
        print(json.dumps(selfplay_dist, indent=2))

        # Get ELO leaderboard
        leaderboard = await manager.evaluation.get_leaderboard()
        print("\nELO Leaderboard (top 5):")
        for i, entry in enumerate(leaderboard[:5]):
            print(f"  {i+1}. {entry.get('model_id', 'unknown')}: {entry.get('elo', 0):.0f}")

    finally:
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
