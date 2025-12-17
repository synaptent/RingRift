"""P2P Orchestrator Type Definitions.

This module contains enums used throughout the P2P orchestrator.
Extracted from p2p_orchestrator.py for better modularity.
"""

from enum import Enum


class NodeRole(str, Enum):
    """Role a node plays in the cluster."""
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"


class JobType(str, Enum):
    """Types of jobs nodes can run."""
    SELFPLAY = "selfplay"
    GPU_SELFPLAY = "gpu_selfplay"  # GPU-accelerated parallel selfplay (pure GPU, experimental)
    HYBRID_SELFPLAY = "hybrid_selfplay"  # Hybrid CPU/GPU selfplay (100% rule fidelity, GPU-accelerated eval)
    CPU_SELFPLAY = "cpu_selfplay"  # Pure CPU selfplay to utilize excess CPU on high-CPU/low-VRAM nodes
    TRAINING = "training"
    CMAES = "cmaes"
    # Distributed job types
    DISTRIBUTED_CMAES_COORDINATOR = "distributed_cmaes_coordinator"
    DISTRIBUTED_CMAES_WORKER = "distributed_cmaes_worker"
    DISTRIBUTED_TOURNAMENT_COORDINATOR = "distributed_tournament_coordinator"
    DISTRIBUTED_TOURNAMENT_WORKER = "distributed_tournament_worker"
    IMPROVEMENT_LOOP = "improvement_loop"
    # CPU-intensive data processing jobs
    DATA_EXPORT = "data_export"  # NPZ export (CPU-intensive, route to high-CPU nodes)
    DATA_AGGREGATION = "data_aggregation"  # JSONL aggregation (CPU-intensive)
