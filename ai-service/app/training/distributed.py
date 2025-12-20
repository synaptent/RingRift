"""Distributed Training Orchestration for RingRift AI.

.. deprecated:: 2025-12
    Use ``distributed_unified.py`` for new code. This module is deprecated
    except for helper functions which remain canonical.

    Migration::

        # DistributedTrainer → UnifiedDistributedTrainer
        # Old:
        from app.training.distributed import DistributedTrainer, DistributedConfig
        # New:
        from app.training.distributed_unified import (
            UnifiedDistributedTrainer,
            UnifiedDistributedConfig,
        )

    Keep using this module for helper functions:
    - setup_distributed(), cleanup_distributed()
    - is_main_process(), get_rank(), get_world_size()
    - get_distributed_sampler(), wrap_model_ddp()
    - seed_everything(), scale_learning_rate()
    - DistributedMetrics

Coordinates training across multiple GPUs and nodes using PyTorch
Distributed Data Parallel (DDP) with gradient synchronization.

Features:
1. Multi-GPU training on a single node
2. Multi-node training with TCP/NCCL backend
3. Gradient averaging and synchronization
4. Fault tolerance with checkpoint recovery
5. Elastic scaling support

Helper Functions (use these regardless of trainer choice):
    - setup_distributed(): Initialize process group
    - cleanup_distributed(): Destroy process group
    - is_main_process(): Check if rank 0
    - get_rank(), get_world_size(), get_local_rank()
    - get_distributed_sampler(): Create DistributedSampler
    - wrap_model_ddp(): Wrap model with DDP
    - seed_everything(): Seed all RNGs
    - scale_learning_rate(): Scale LR for distributed
    - DistributedMetrics: Cross-process metric aggregation

Usage:
    from app.training.distributed import DistributedTrainer, DistributedConfig

    config = DistributedConfig(
        world_size=4,
        backend="nccl",
    )

    trainer = DistributedTrainer(model, config)
    trainer.setup()

    for batch in dataloader:
        loss = trainer.train_step(batch)
"""

from __future__ import annotations

import logging
import os
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Try to import torch distributed
try:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler
    HAS_TORCH_DISTRIBUTED = True
except ImportError:
    HAS_TORCH_DISTRIBUTED = False
    torch = None
    dist = None
    DDP = None


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"
    master_addr: str = "localhost"
    master_port: int = 29500
    init_method: str | None = None
    gradient_sync_every: int = 1
    use_sync_batchnorm: bool = True
    checkpoint_dir: str = "data/distributed_checkpoints"
    checkpoint_interval: int = 1000
    auto_resume: bool = True
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25


@dataclass
class NodeInfo:
    """Information about a node in the cluster."""
    node_id: str
    hostname: str
    rank: int
    local_rank: int
    device: str
    status: str = "active"
    last_heartbeat: float = 0.0


class DistributedTrainer:
    """Orchestrates distributed training across multiple GPUs/nodes.

    .. deprecated:: 2025-12
        Use :class:`app.training.distributed_unified.UnifiedDistributedTrainer`
        for new code. This class remains for backward compatibility.
    """

    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        import warnings
        warnings.warn(
            "DistributedTrainer is deprecated. "
            "Use app.training.distributed_unified.UnifiedDistributedTrainer for new code.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not HAS_TORCH_DISTRIBUTED:
            raise ImportError("PyTorch distributed not available")

        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.ddp_model: DDP | None = None
        self.is_initialized = False
        self.step_count = 0
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.nodes: dict[int, NodeInfo] = {}

    def setup(self) -> bool:
        """Initialize distributed training environment."""
        config = self.config
        os.environ["MASTER_ADDR"] = config.master_addr
        os.environ["MASTER_PORT"] = str(config.master_port)

        try:
            if config.init_method:
                dist.init_process_group(
                    backend=config.backend,
                    init_method=config.init_method,
                    world_size=config.world_size,
                    rank=config.rank,
                )
            else:
                dist.init_process_group(
                    backend=config.backend,
                    world_size=config.world_size,
                    rank=config.rank,
                )

            if torch.cuda.is_available() and config.backend == "nccl":
                torch.cuda.set_device(config.local_rank)
                device = torch.device(f"cuda:{config.local_rank}")
            else:
                device = torch.device("cpu")

            self.model = self.model.to(device)

            if config.use_sync_batchnorm and config.world_size > 1:
                self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

            self.ddp_model = DDP(
                self.model,
                device_ids=[config.local_rank] if torch.cuda.is_available() else None,
                output_device=config.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=config.find_unused_parameters,
                broadcast_buffers=config.broadcast_buffers,
                bucket_cap_mb=config.bucket_cap_mb,
            )

            self.is_initialized = True
            self._register_node()

            if config.auto_resume:
                self._try_resume()

            logger.info(f"[Distributed] Initialized rank {config.rank}/{config.world_size}")
            return True

        except Exception as e:
            logger.error(f"[Distributed] Setup failed: {e}")
            return False

    def _register_node(self):
        config = self.config
        self.nodes[config.rank] = NodeInfo(
            node_id=f"node_{config.rank}",
            hostname=socket.gethostname(),
            rank=config.rank,
            local_rank=config.local_rank,
            device=f"cuda:{config.local_rank}" if torch.cuda.is_available() else "cpu",
            status="active",
            last_heartbeat=time.time(),
        )

    def cleanup(self):
        if self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False

    def train_step(self, batch: tuple[torch.Tensor, ...], loss_fn: Callable) -> float:
        if not self.is_initialized or self.ddp_model is None:
            raise RuntimeError("Distributed trainer not initialized")

        self.ddp_model.train()
        inputs = batch[0]
        targets = batch[1] if len(batch) > 1 else None

        outputs = self.ddp_model(inputs)
        loss = loss_fn(outputs, targets) if targets is not None else loss_fn(outputs)

        if self.optimizer:
            self.optimizer.zero_grad()
        loss.backward()

        if self.optimizer and self.step_count % self.config.gradient_sync_every == 0:
            self.optimizer.step()

        self.step_count += 1

        if self.step_count % self.config.checkpoint_interval == 0:
            self.save_checkpoint()

        return loss.item()

    def save_checkpoint(self, path: Path | None = None):
        if self.config.rank != 0:
            return
        path = path or (self.checkpoint_dir / f"checkpoint_{self.step_count}.pt")
        checkpoint = {
            "step_count": self.step_count,
            "model_state_dict": self.model.state_dict(),
        }
        if self.optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        torch.save(checkpoint, path)
        logger.info(f"[Distributed] Checkpoint saved to {path}")

    def _try_resume(self):
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return
        latest = checkpoints[-1]
        try:
            checkpoint = torch.load(latest, map_location=self._get_device())
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.step_count = checkpoint["step_count"]
            if self.optimizer and "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info(f"[Distributed] Resumed from {latest}")
        except Exception as e:
            logger.warning(f"[Distributed] Failed to resume: {e}")

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available() and self.config.backend == "nccl":
            return torch.device(f"cuda:{self.config.local_rank}")
        return torch.device("cpu")

    @property
    def is_main_process(self) -> bool:
        return self.config.rank == 0

    def barrier(self):
        if self.is_initialized:
            dist.barrier()


def create_distributed_trainer(
    model: nn.Module,
    world_size: int = 1,
    rank: int = 0,
    backend: str = "nccl",
) -> DistributedTrainer:
    """Factory function to create a distributed trainer."""
    config = DistributedConfig(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        backend=backend,
    )
    return DistributedTrainer(model, config)


# =============================================================================
# Helper Functions for train.py Compatibility
# =============================================================================

def setup_distributed(
    local_rank: int = -1,
    rank: int | None = None,
    world_size: int | None = None,
    backend: str | None = None,
) -> None:
    """Initialize distributed training process group.

    Args:
        local_rank: Local rank from torchrun/launcher (-1 for auto-detect)
        rank: Global rank (auto-detect from env if not provided)
        world_size: Total processes (auto-detect from env if not provided)
        backend: Communication backend ('nccl', 'gloo', auto-detect if not provided)
    """
    if not HAS_TORCH_DISTRIBUTED:
        return

    if dist.is_initialized():
        return

    # Auto-detect from environment
    if local_rank == -1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if rank is None:
        rank = int(os.environ.get("RANK", 0))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size == 1:
        logger.info("Single process, skipping distributed setup")
        return

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Determine backend
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    logger.info(f"Initialized distributed: rank={rank}, world_size={world_size}, backend={backend}")


def cleanup_distributed() -> None:
    """Cleanup distributed training resources."""
    if HAS_TORCH_DISTRIBUTED and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if not HAS_TORCH_DISTRIBUTED or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    if not HAS_TORCH_DISTRIBUTED or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get total number of processes."""
    if not HAS_TORCH_DISTRIBUTED or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_distributed() -> bool:
    """Check if distributed training is available and initialized."""
    if not HAS_TORCH_DISTRIBUTED:
        return False
    return dist.is_available() and dist.is_initialized()


def get_local_rank() -> int:
    """Get local rank from environment variable."""
    return int(os.environ.get("LOCAL_RANK", 0))


def synchronize() -> None:
    """Synchronize all processes with a barrier."""
    if is_distributed():
        dist.barrier()


def reduce_tensor(tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """Reduce tensor across all processes.

    Args:
        tensor: Tensor to reduce
        op: Reduce operation ('sum', 'mean', 'max', 'min')

    Returns:
        Reduced tensor
    """
    if not is_distributed():
        return tensor

    ops = {
        "sum": dist.ReduceOp.SUM,
        "mean": dist.ReduceOp.SUM,  # Will divide by world_size
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
    }
    reduce_op = ops.get(op, dist.ReduceOp.SUM)

    dist.all_reduce(tensor, op=reduce_op)

    if op == "mean":
        tensor.div_(get_world_size())

    return tensor


def all_gather_object(obj: Any) -> list:
    """Gather objects from all processes.

    Args:
        obj: Object to gather

    Returns:
        List of objects from all processes
    """
    if not is_distributed():
        return [obj]

    output = [None] * get_world_size()
    dist.all_gather_object(output, obj)
    return output


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast object from source rank to all processes.

    Args:
        obj: Object to broadcast (only used on src rank)
        src: Source rank

    Returns:
        Broadcasted object
    """
    if not is_distributed():
        return obj

    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def get_device_for_rank() -> torch.device:
    """Get the device for the current rank.

    Returns:
        torch.device for this rank
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{get_local_rank()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_distributed_sampler(
    dataset: Any,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = False,
) -> DistributedSampler | None:
    """Create a DistributedSampler for the dataset.

    Args:
        dataset: Dataset to sample from
        shuffle: Whether to shuffle
        seed: Random seed
        drop_last: Drop last incomplete batch

    Returns:
        DistributedSampler or None if not distributed
    """
    if not HAS_TORCH_DISTRIBUTED or not dist.is_initialized():
        return None

    return DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )


def wrap_model_ddp(
    model: nn.Module,
    device: Any,
    find_unused_parameters: bool = False,
    broadcast_buffers: bool = True,
) -> nn.Module:
    """Wrap model with DistributedDataParallel.

    Args:
        model: Model to wrap
        device: Device to use
        find_unused_parameters: Enable for models with unused parameters
        broadcast_buffers: Broadcast buffers from rank 0

    Returns:
        DDP-wrapped model or original if not distributed
    """
    if not HAS_TORCH_DISTRIBUTED or not dist.is_initialized():
        return model

    # Move to device first
    model = model.to(device)

    # Wrap with DDP
    ddp_model = DDP(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        output_device=device.index if device.type == "cuda" else None,
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=broadcast_buffers,
    )

    return ddp_model


def seed_everything(seed: int = 42, rank_offset: bool = False) -> None:
    """Seed all random number generators for reproducibility.

    Args:
        seed: Base random seed
        rank_offset: If True, add rank to seed for different random states per process
    """
    import random

    import numpy as np

    if rank_offset and HAS_TORCH_DISTRIBUTED and dist.is_initialized():
        seed = seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)

    if HAS_TORCH_DISTRIBUTED:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def scale_learning_rate(
    base_lr: float,
    world_size: int | None = None,
    scale_type: str = "linear",
) -> float:
    """Scale learning rate for distributed training.

    Args:
        base_lr: Base learning rate
        world_size: Number of processes (defaults to get_world_size())
        scale_type: "linear", "sqrt", or "none"

    Returns:
        Scaled learning rate
    """
    if world_size is None:
        world_size = get_world_size()

    if scale_type == "none":
        return base_lr

    if world_size <= 1:
        return base_lr

    if scale_type == "sqrt":
        return base_lr * (world_size ** 0.5)
    elif scale_type == "linear":
        return base_lr * world_size
    else:
        raise ValueError(f"Unknown scale_type: {scale_type}. Use 'linear', 'sqrt', or 'none'")


class DistributedMetrics:
    """Track and aggregate metrics across distributed processes."""

    def __init__(self):
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    @property
    def _metrics(self) -> dict[str, float]:
        """Alias for backward compatibility."""
        return self._sums

    def reset(self):
        """Reset all metrics."""
        self._sums.clear()
        self._counts.clear()

    def update(self, name: str, value: float, count: int = 1):
        """Update a metric with a new value."""
        if name not in self._sums:
            self._sums[name] = 0.0
            self._counts[name] = 0
        self._sums[name] += value * count
        self._counts[name] += count

    def add(self, name: str, value: float, count: int = 1):
        """Add a metric value. Alias for update()."""
        self.update(name, value, count)

    def reduce_all(self) -> dict[str, float]:
        """Reduce metrics across all processes and return averages."""
        if not HAS_TORCH_DISTRIBUTED or not dist.is_initialized():
            return {k: v / max(self._counts[k], 1) for k, v in self._sums.items()}

        result = {}
        for name, value in self._sums.items():
            count = self._counts[name]

            # Create tensors for reduction
            value_tensor = torch.tensor([value], dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")
            count_tensor = torch.tensor([count], dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")

            # All-reduce sum
            dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

            # Compute average
            result[name] = value_tensor.item() / max(count_tensor.item(), 1)

        return result

    def reduce_and_reset(self) -> dict[str, float]:
        """Reduce metrics and reset accumulators."""
        result = self.reduce_all()
        self.reset()
        return result

    def get(self, name: str, default: float = 0.0) -> float:
        """Get a metric value (local, not reduced)."""
        if name not in self._sums:
            return default
        return self._sums[name] / max(self._counts[name], 1)
