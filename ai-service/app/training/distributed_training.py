"""
Distributed Training Support for RingRift AI.

Provides multi-node distributed training with:
- Gradient averaging across nodes
- Gradient compression for bandwidth efficiency
- Async SGD with staleness bounds
- Fault-tolerant training with checkpointing

Usage:
    from app.training.distributed_training import (
        DistributedTrainer,
        GradientCompressor,
        AsyncSGD,
    )
"""

from __future__ import annotations

import logging
import os
import socket
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    # Communication backend
    backend: str = "nccl"  # "nccl" for GPU, "gloo" for CPU
    init_method: str = "env://"

    # Gradient compression
    compress_gradients: bool = True
    compression_ratio: float = 0.01  # Keep top 1% of gradients

    # Async SGD settings
    async_sgd: bool = False
    max_staleness: int = 3  # Maximum allowed staleness for gradients

    # Fault tolerance
    checkpoint_interval: int = 100  # Steps between checkpoints
    elastic_training: bool = True  # Allow node failures

    # Bandwidth optimization
    bucket_size_mb: int = 25
    gradient_as_bucket_view: bool = True

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"


class GradientCompressor:
    """
    Compresses gradients for bandwidth-efficient distributed training.

    Uses top-K sparsification with error feedback to maintain convergence.
    """

    def __init__(
        self,
        compression_ratio: float = 0.01,
        warmup_steps: int = 100,
    ):
        """
        Args:
            compression_ratio: Fraction of gradients to keep (0.01 = top 1%)
            warmup_steps: Steps before enabling compression
        """
        self.compression_ratio = compression_ratio
        self.warmup_steps = warmup_steps
        self._step = 0

        # Error feedback buffers
        self._error_buffers: Dict[str, torch.Tensor] = {}

    def compress(
        self,
        gradients: Dict[str, torch.Tensor],
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compress gradients using top-K sparsification.

        Args:
            gradients: Dictionary of parameter name -> gradient tensor

        Returns:
            Dictionary of parameter name -> (values, indices) sparse representation
        """
        self._step += 1

        # Skip compression during warmup
        if self._step < self.warmup_steps:
            return {
                name: (grad.flatten(), torch.arange(grad.numel(), device=grad.device))
                for name, grad in gradients.items()
            }

        compressed = {}

        for name, grad in gradients.items():
            # Add error feedback
            if name in self._error_buffers:
                grad = grad + self._error_buffers[name]

            flat_grad = grad.flatten()
            k = max(1, int(len(flat_grad) * self.compression_ratio))

            # Get top-K by magnitude
            _, indices = torch.topk(flat_grad.abs(), k)
            values = flat_grad[indices]

            # Store error for next iteration
            error = flat_grad.clone()
            error[indices] = 0
            self._error_buffers[name] = error.view_as(grad)

            compressed[name] = (values, indices)

        return compressed

    def decompress(
        self,
        compressed: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        shapes: Dict[str, torch.Size],
    ) -> Dict[str, torch.Tensor]:
        """
        Decompress gradients from sparse representation.

        Args:
            compressed: Dictionary of parameter name -> (values, indices)
            shapes: Original tensor shapes

        Returns:
            Dictionary of parameter name -> decompressed gradient tensor
        """
        decompressed = {}

        for name, (values, indices) in compressed.items():
            shape = shapes[name]
            numel = 1
            for dim in shape:
                numel *= dim

            flat = torch.zeros(numel, device=values.device, dtype=values.dtype)
            flat[indices] = values
            decompressed[name] = flat.view(shape)

        return decompressed


class AsyncSGD:
    """
    Asynchronous SGD with staleness-bounded gradient updates.

    Allows workers to proceed without waiting for global synchronization,
    improving throughput at the cost of slightly stale gradients.
    """

    def __init__(
        self,
        model: nn.Module,
        max_staleness: int = 3,
        learning_rate_decay: float = 0.9,
    ):
        """
        Args:
            model: Model being trained
            max_staleness: Maximum allowed gradient staleness
            learning_rate_decay: LR decay factor per staleness level
        """
        self.model = model
        self.max_staleness = max_staleness
        self.lr_decay = learning_rate_decay

        self._gradient_queue: deque = deque(maxlen=max_staleness + 1)
        self._current_step = 0
        self._pending_updates = 0

    def push_gradients(
        self,
        gradients: Dict[str, torch.Tensor],
        step: int,
    ) -> None:
        """
        Push gradients from a worker.

        Args:
            gradients: Gradients to push
            step: Step number when gradients were computed
        """
        self._gradient_queue.append({
            'gradients': gradients,
            'step': step,
            'timestamp': time.time(),
        })
        self._pending_updates += 1

    def get_update(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get aggregated gradient update.

        Returns:
            Aggregated gradients or None if no updates pending
        """
        if not self._gradient_queue:
            return None

        # Filter out stale gradients
        current_step = self._current_step
        valid_updates = []

        while self._gradient_queue:
            update = self._gradient_queue.popleft()
            staleness = current_step - update['step']

            if staleness <= self.max_staleness:
                # Apply staleness-based weight decay
                weight = self.lr_decay ** staleness
                valid_updates.append((update['gradients'], weight))
            else:
                logger.debug(f"Discarding stale gradient (staleness={staleness})")

        if not valid_updates:
            return None

        # Weighted average of gradients
        aggregated = {}
        total_weight = sum(w for _, w in valid_updates)

        for grads, weight in valid_updates:
            for name, grad in grads.items():
                if name not in aggregated:
                    aggregated[name] = torch.zeros_like(grad)
                aggregated[name] += (weight / total_weight) * grad

        self._current_step += 1
        return aggregated

    @property
    def pending_updates(self) -> int:
        """Number of pending gradient updates."""
        return len(self._gradient_queue)


class DistributedTrainer:
    """
    Manages distributed training across multiple nodes.

    Features:
    - Automatic process group initialization
    - DDP model wrapping
    - Gradient compression
    - Async SGD support
    - Fault tolerance with checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[DistributedConfig] = None,
    ):
        """
        Args:
            model: Model to train
            config: Distributed training configuration
        """
        self.config = config or DistributedConfig()
        self._original_model = model
        self._ddp_model: Optional[DDP] = None
        self._compressor: Optional[GradientCompressor] = None
        self._async_sgd: Optional[AsyncSGD] = None

        self._initialized = False
        self._rank = 0
        self._world_size = 1
        self._local_rank = 0

        # Checkpointing
        self._step = 0
        self._checkpoint_dir: Optional[str] = None

        # AMP
        self._scaler: Optional[torch.cuda.amp.GradScaler] = None

    def initialize(
        self,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        local_rank: Optional[int] = None,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
    ) -> None:
        """
        Initialize distributed training environment.

        Args:
            rank: Process rank (default: from env)
            world_size: Total number of processes (default: from env)
            local_rank: Local GPU rank (default: from env)
            master_addr: Master address (default: from env)
            master_port: Master port (default: from env)
        """
        if self._initialized:
            logger.warning("Distributed trainer already initialized")
            return

        # Get from environment if not provided
        self._rank = rank if rank is not None else int(os.environ.get('RANK', 0))
        self._world_size = world_size if world_size is not None else int(os.environ.get('WORLD_SIZE', 1))
        self._local_rank = local_rank if local_rank is not None else int(os.environ.get('LOCAL_RANK', 0))

        if master_addr:
            os.environ['MASTER_ADDR'] = master_addr
        if master_port:
            os.environ['MASTER_PORT'] = str(master_port)

        # Set default master if not provided
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'

        # Initialize process group
        if self._world_size > 1:
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                rank=self._rank,
                world_size=self._world_size,
            )

            # Set device
            if torch.cuda.is_available():
                torch.cuda.set_device(self._local_rank)
                device = torch.device(f"cuda:{self._local_rank}")
            else:
                device = torch.device("cpu")

            # Wrap model with DDP
            self._original_model = self._original_model.to(device)
            self._ddp_model = DDP(
                self._original_model,
                device_ids=[self._local_rank] if torch.cuda.is_available() else None,
                output_device=self._local_rank if torch.cuda.is_available() else None,
                bucket_cap_mb=self.config.bucket_size_mb,
                gradient_as_bucket_view=self.config.gradient_as_bucket_view,
            )

            logger.info(
                f"Initialized distributed training: rank={self._rank}/{self._world_size}, "
                f"local_rank={self._local_rank}, device={device}"
            )
        else:
            logger.info("Single-process training (no DDP)")
            if torch.cuda.is_available():
                self._original_model = self._original_model.cuda()

        # Initialize compression
        if self.config.compress_gradients and self._world_size > 1:
            self._compressor = GradientCompressor(
                compression_ratio=self.config.compression_ratio,
            )

        # Initialize async SGD
        if self.config.async_sgd:
            self._async_sgd = AsyncSGD(
                model=self._original_model,
                max_staleness=self.config.max_staleness,
            )

        # Initialize AMP
        if self.config.use_amp and torch.cuda.is_available():
            self._scaler = torch.cuda.amp.GradScaler()

        self._initialized = True

    @property
    def model(self) -> nn.Module:
        """Get the wrapped model."""
        return self._ddp_model if self._ddp_model is not None else self._original_model

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self._rank == 0

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Execute a single training step with distributed support.

        Args:
            batch: Training batch
            criterion: Loss function
            optimizer: Optimizer

        Returns:
            Dictionary of loss values
        """
        self._step += 1

        # Move batch to device
        device = next(self.model.parameters()).device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass with AMP
        if self._scaler is not None:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = self.model(batch['features'])
                loss = criterion(outputs, batch)
        else:
            outputs = self.model(batch['features'])
            loss = criterion(outputs, batch)

        # Backward pass
        optimizer.zero_grad()

        if self._scaler is not None:
            self._scaler.scale(loss).backward()
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Checkpointing
        if (
            self._checkpoint_dir is not None and
            self._step % self.config.checkpoint_interval == 0 and
            self.is_main_process
        ):
            self.save_checkpoint()

        return {'loss': loss.item()}

    def save_checkpoint(
        self,
        path: Optional[str] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a training checkpoint.

        Args:
            path: Checkpoint path (default: auto-generated)
            extra_state: Additional state to save

        Returns:
            Path to saved checkpoint
        """
        if path is None:
            if self._checkpoint_dir is None:
                self._checkpoint_dir = "/tmp/ringrift_distributed_ckpt"
            os.makedirs(self._checkpoint_dir, exist_ok=True)
            path = os.path.join(self._checkpoint_dir, f"checkpoint_step{self._step}.pt")

        state = {
            'step': self._step,
            'model_state_dict': self._original_model.state_dict(),
            'rank': self._rank,
            'world_size': self._world_size,
            'timestamp': datetime.now().isoformat(),
        }

        if extra_state:
            state.update(extra_state)

        torch.save(state, path)
        logger.info(f"Saved checkpoint to {path}")
        return path

    def load_checkpoint(
        self,
        path: str,
    ) -> Dict[str, Any]:
        """
        Load a training checkpoint.

        Args:
            path: Checkpoint path

        Returns:
            Extra state from checkpoint
        """
        device = next(self._original_model.parameters()).device
        state = torch.load(path, map_location=device)

        self._original_model.load_state_dict(state['model_state_dict'])
        self._step = state['step']

        logger.info(f"Loaded checkpoint from {path} (step {self._step})")
        return state

    def cleanup(self) -> None:
        """Cleanup distributed training resources."""
        if self._initialized and self._world_size > 1:
            dist.destroy_process_group()
            self._initialized = False
            logger.info("Cleaned up distributed training")


def all_reduce_gradients(
    model: nn.Module,
    async_op: bool = False,
) -> None:
    """
    All-reduce gradients across all processes.

    Args:
        model: Model with gradients to reduce
        async_op: Whether to use async operation
    """
    if not dist.is_initialized():
        return

    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=async_op)


def broadcast_model(
    model: nn.Module,
    src: int = 0,
) -> None:
    """
    Broadcast model parameters from source rank.

    Args:
        model: Model to broadcast
        src: Source rank
    """
    if not dist.is_initialized():
        return

    for param in model.parameters():
        dist.broadcast(param.data, src=src)


def get_distributed_info() -> Dict[str, Any]:
    """
    Get current distributed training information.

    Returns:
        Dictionary with rank, world_size, etc.
    """
    if dist.is_initialized():
        return {
            'initialized': True,
            'rank': dist.get_rank(),
            'world_size': dist.get_world_size(),
            'backend': dist.get_backend(),
            'hostname': socket.gethostname(),
        }
    return {
        'initialized': False,
        'rank': 0,
        'world_size': 1,
        'backend': None,
        'hostname': socket.gethostname(),
    }
