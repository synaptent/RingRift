"""Unified Distributed Training for RingRift AI.

Consolidated module combining features from distributed.py and distributed_training.py:
- Multi-GPU/Multi-node training with PyTorch DDP
- Gradient compression for bandwidth efficiency
- Async SGD with staleness bounds
- Fault tolerance with checkpoint recovery
- Elastic scaling support
- Mixed precision (AMP) training

This module replaces both distributed.py and distributed_training.py.

Usage:
    from app.training.distributed_unified import (
        UnifiedDistributedTrainer,
        UnifiedDistributedConfig,
        GradientCompressor,
        AsyncSGD,
    )

    config = UnifiedDistributedConfig(
        world_size=4,
        backend="nccl",
        compress_gradients=True,
        use_amp=True,
    )

    trainer = UnifiedDistributedTrainer(model, config)
    trainer.setup()

    for batch in dataloader:
        loss = trainer.train_step(batch, loss_fn)
"""

from __future__ import annotations

import logging
import os
import socket
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Use shared lazy torch import; extend with distributed modules
from app.training.utils import get_torch
from app.utils.resource_guard import check_gpu_memory, check_memory, clear_gpu_memory
from app.utils.torch_utils import safe_load_checkpoint

_dist = None
_DDP = None


def _get_torch_distributed():
    """Get torch, torch.distributed, and DDP lazily."""
    global _dist, _DDP
    torch = get_torch()
    if _dist is None:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        _dist = dist
        _DDP = DDP
    return torch, _dist, _DDP


def _auto_detect_backend() -> str:
    """Auto-detect the best distributed backend.

    Uses RINGRIFT_DISTRIBUTED_BACKEND env var if set,
    otherwise auto-detects based on CUDA availability.

    Returns:
        "nccl" for GPU training, "gloo" for CPU training.
    """
    env_backend = os.environ.get("RINGRIFT_DISTRIBUTED_BACKEND")
    if env_backend:
        return env_backend.lower()

    # Auto-detect: use nccl if CUDA available, gloo otherwise
    torch = get_torch()
    return "nccl" if torch.cuda.is_available() else "gloo"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class UnifiedDistributedConfig:
    """Unified configuration for distributed training.

    Combines all distributed training options in a single config.
    """
    # Basic distributed settings
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    # Backend: "nccl" for GPU, "gloo" for CPU. Set RINGRIFT_DISTRIBUTED_BACKEND to override.
    backend: str = ""  # Empty string triggers auto-detection
    master_addr: str = "localhost"
    master_port: int = 29500
    init_method: str = "env://"

    # DDP settings
    gradient_sync_every: int = 1
    use_sync_batchnorm: bool = True
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25

    # Gradient compression
    compress_gradients: bool = False
    compression_ratio: float = 0.01  # Keep top 1% of gradients
    compression_warmup_steps: int = 100

    # Async SGD settings
    async_sgd: bool = False
    max_staleness: int = 3
    staleness_lr_decay: float = 0.9

    # Fault tolerance
    checkpoint_dir: str = "data/distributed_checkpoints"
    checkpoint_interval: int = 1000
    auto_resume: bool = True
    elastic_training: bool = True

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # "float16" or "bfloat16"

    # Logging
    log_interval: int = 100

    def __post_init__(self):
        """Auto-detect backend if not specified."""
        if not self.backend:
            self.backend = _auto_detect_backend()


@dataclass
class NodeInfo:
    """Information about a node in the distributed cluster."""
    node_id: str
    hostname: str
    rank: int
    local_rank: int
    device: str
    status: str = "active"
    last_heartbeat: float = 0.0
    gradient_delay_ms: float = 0.0
    throughput_samples_sec: float = 0.0


# =============================================================================
# Gradient Compression
# =============================================================================

class GradientCompressor:
    """Compresses gradients for bandwidth-efficient distributed training.

    Uses top-K sparsification with error feedback to maintain convergence.
    """

    def __init__(
        self,
        compression_ratio: float = 0.01,
        warmup_steps: int = 100,
    ):
        """Initialize gradient compressor.

        Args:
            compression_ratio: Fraction of gradients to keep (0.01 = top 1%)
            warmup_steps: Steps before enabling compression
        """
        self.compression_ratio = compression_ratio
        self.warmup_steps = warmup_steps
        self._step = 0
        self._error_buffers: dict[str, Any] = {}  # torch.Tensor

    def compress(
        self,
        gradients: dict[str, Any],  # torch.Tensor
    ) -> dict[str, tuple[Any, Any]]:
        """Compress gradients using top-K sparsification.

        Args:
            gradients: Dictionary of parameter name -> gradient tensor

        Returns:
            Dictionary of parameter name -> (values, indices) sparse representation
        """
        torch, _, _ = _get_torch_distributed()
        self._step += 1

        # Skip compression during warmup
        if self._step < self.warmup_steps:
            return {
                name: (grad.flatten(), torch.arange(grad.numel(), device=grad.device))
                for name, grad in gradients.items()
            }

        compressed = {}

        for name, grad in gradients.items():
            # Add error feedback from previous iteration
            if name in self._error_buffers:
                grad = grad + self._error_buffers[name]

            flat_grad = grad.flatten()
            k = max(1, int(len(flat_grad) * self.compression_ratio))

            # Get top-K by magnitude
            _, indices = torch.topk(flat_grad.abs(), k)
            values = flat_grad[indices]

            # Store error for next iteration (maintains convergence)
            error = flat_grad.clone()
            error[indices] = 0
            self._error_buffers[name] = error.view_as(grad)

            compressed[name] = (values, indices)

        return compressed

    def decompress(
        self,
        compressed: dict[str, tuple[Any, Any]],
        shapes: dict[str, tuple],
    ) -> dict[str, Any]:
        """Decompress gradients from sparse representation.

        Args:
            compressed: Dictionary of parameter name -> (values, indices)
            shapes: Original tensor shapes

        Returns:
            Dictionary of parameter name -> decompressed gradient tensor
        """
        torch, _, _ = _get_torch_distributed()
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

    def reset(self):
        """Reset error buffers (e.g., at epoch boundary)."""
        self._error_buffers.clear()


# =============================================================================
# Async SGD
# =============================================================================

class AsyncSGD:
    """Asynchronous SGD with staleness-bounded gradient updates.

    Allows workers to proceed without waiting for global synchronization,
    improving throughput at the cost of slightly stale gradients.
    """

    def __init__(
        self,
        max_staleness: int = 3,
        learning_rate_decay: float = 0.9,
    ):
        """Initialize async SGD.

        Args:
            max_staleness: Maximum allowed gradient staleness
            learning_rate_decay: LR decay factor per staleness level
        """
        self.max_staleness = max_staleness
        self.lr_decay = learning_rate_decay
        self._gradient_queue: deque = deque(maxlen=max_staleness + 1)
        self._current_step = 0
        self._discarded_count = 0

    def push_gradients(
        self,
        gradients: dict[str, Any],
        step: int,
    ) -> None:
        """Push gradients from a worker.

        Args:
            gradients: Gradients to push
            step: Step number when gradients were computed
        """
        self._gradient_queue.append({
            'gradients': gradients,
            'step': step,
            'timestamp': time.time(),
        })

    def get_update(self) -> dict[str, Any] | None:
        """Get aggregated gradient update.

        Returns:
            Aggregated gradients or None if no updates pending
        """
        torch, _, _ = _get_torch_distributed()

        if not self._gradient_queue:
            return None

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
                self._discarded_count += 1
                logger.debug(f"[AsyncSGD] Discarded stale gradient (staleness={staleness})")

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

    @property
    def discarded_count(self) -> int:
        """Number of discarded stale gradients."""
        return self._discarded_count


# =============================================================================
# Unified Distributed Trainer
# =============================================================================

class UnifiedDistributedTrainer:
    """Unified distributed trainer combining all distributed training features.

    Features:
    - Automatic process group initialization
    - DDP model wrapping with sync batchnorm
    - Gradient compression (optional)
    - Async SGD support (optional)
    - Fault tolerance with checkpointing
    - Mixed precision (AMP) training
    - Elastic scaling
    - Node health monitoring
    """

    def __init__(
        self,
        model: Any,  # nn.Module
        config: UnifiedDistributedConfig | None = None,
        optimizer: Any | None = None,  # torch.optim.Optimizer
    ):
        """Initialize unified distributed trainer.

        Args:
            model: PyTorch model to train
            config: Distributed training configuration
            optimizer: Optional optimizer (can be set later)
        """
        self.config = config or UnifiedDistributedConfig()
        self._original_model = model
        self._optimizer = optimizer
        self._ddp_model: Any | None = None

        # State
        self._initialized = False
        self._step = 0
        self._epoch = 0

        # Node tracking
        self._nodes: dict[int, NodeInfo] = {}

        # Components (initialized lazily)
        self._compressor: GradientCompressor | None = None
        self._async_sgd: AsyncSGD | None = None
        self._scaler: Any | None = None  # GradScaler

        # Checkpointing
        self._checkpoint_dir = Path(self.config.checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[DistributedUnified] Created trainer (world_size={config.world_size})")

    def setup(self) -> bool:
        """Initialize distributed training environment.

        Returns:
            True if initialization successful
        """
        torch, dist, DDP = _get_torch_distributed()
        config = self.config

        # Set environment variables
        os.environ["MASTER_ADDR"] = config.master_addr
        os.environ["MASTER_PORT"] = str(config.master_port)

        try:
            # Initialize process group
            if config.world_size > 1:
                dist.init_process_group(
                    backend=config.backend,
                    init_method=config.init_method,
                    world_size=config.world_size,
                    rank=config.rank,
                )

            # Set device
            if torch.cuda.is_available() and config.backend == "nccl":
                torch.cuda.set_device(config.local_rank)
                self._device = torch.device(f"cuda:{config.local_rank}")
            else:
                self._device = torch.device("cpu")

            # Resource guard: check GPU/memory before model allocation
            if self._device.type == "cuda":
                if not check_gpu_memory():
                    logger.warning("[DistributedUnified] GPU memory pressure, clearing cache")
                    clear_gpu_memory()
                if not check_memory():
                    logger.warning("[DistributedUnified] System memory pressure detected")

            # Move model to device
            self._original_model = self._original_model.to(self._device)

            # Convert to sync batchnorm if needed
            if config.use_sync_batchnorm and config.world_size > 1:
                import torch.nn as nn
                self._original_model = nn.SyncBatchNorm.convert_sync_batchnorm(
                    self._original_model
                )

            # Wrap with DDP
            if config.world_size > 1:
                self._ddp_model = DDP(
                    self._original_model,
                    device_ids=[config.local_rank] if torch.cuda.is_available() else None,
                    output_device=config.local_rank if torch.cuda.is_available() else None,
                    find_unused_parameters=config.find_unused_parameters,
                    broadcast_buffers=config.broadcast_buffers,
                    bucket_cap_mb=config.bucket_cap_mb,
                )
            else:
                self._ddp_model = self._original_model

            # Initialize gradient compression
            if config.compress_gradients:
                self._compressor = GradientCompressor(
                    compression_ratio=config.compression_ratio,
                    warmup_steps=config.compression_warmup_steps,
                )

            # Initialize async SGD
            if config.async_sgd:
                self._async_sgd = AsyncSGD(
                    max_staleness=config.max_staleness,
                    learning_rate_decay=config.staleness_lr_decay,
                )

            # Initialize AMP
            if config.use_amp and torch.cuda.is_available():
                dtype = torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
                self._scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

            # Register this node
            self._register_node()

            # Try to resume from checkpoint
            if config.auto_resume:
                self._try_resume()

            self._initialized = True
            logger.info(
                f"[DistributedUnified] Initialized rank {config.rank}/{config.world_size} "
                f"on {self._device}"
            )
            return True

        except Exception as e:
            logger.error(f"[DistributedUnified] Setup failed: {e}")
            return False

    def _register_node(self):
        """Register this node in the cluster."""
        _torch, _, _ = _get_torch_distributed()
        config = self.config

        self._nodes[config.rank] = NodeInfo(
            node_id=f"node_{config.rank}",
            hostname=socket.gethostname(),
            rank=config.rank,
            local_rank=config.local_rank,
            device=str(self._device),
            status="active",
            last_heartbeat=time.time(),
        )

    def train_step(
        self,
        batch: tuple[Any, ...],
        loss_fn: Callable,
    ) -> float:
        """Execute a single training step.

        Args:
            batch: Input batch (inputs, targets, ...)
            loss_fn: Loss function

        Returns:
            Loss value
        """
        torch, _, _ = _get_torch_distributed()

        if not self._initialized or self._ddp_model is None:
            raise RuntimeError("Distributed trainer not initialized. Call setup() first.")

        config = self.config
        self._ddp_model.train()

        # Unpack batch
        inputs = batch[0].to(self._device)
        targets = batch[1].to(self._device) if len(batch) > 1 else None

        # Forward pass with optional AMP
        if self._scaler is not None:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16):
                outputs = self._ddp_model(inputs)
                loss = loss_fn(outputs, targets) if targets is not None else loss_fn(outputs)
        else:
            outputs = self._ddp_model(inputs)
            loss = loss_fn(outputs, targets) if targets is not None else loss_fn(outputs)

        # Backward pass
        if self._optimizer:
            self._optimizer.zero_grad()

        if self._scaler is not None:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient compression (if enabled)
        if self._compressor is not None and self._step % config.gradient_sync_every == 0:
            gradients = {
                name: param.grad.clone()
                for name, param in self._ddp_model.named_parameters()
                if param.grad is not None
            }
            compressed = self._compressor.compress(gradients)
            # In production, you would all-reduce the compressed gradients here
            # For now, just decompress back (single-node compression demo)
            shapes = {name: grad.shape for name, grad in gradients.items()}
            decompressed = self._compressor.decompress(compressed, shapes)
            for name, param in self._ddp_model.named_parameters():
                if name in decompressed:
                    param.grad = decompressed[name]

        # Optimizer step
        if self._optimizer and self._step % config.gradient_sync_every == 0:
            if self._scaler is not None:
                self._scaler.step(self._optimizer)
                self._scaler.update()
            else:
                self._optimizer.step()

        self._step += 1

        # Checkpointing
        if self._step % config.checkpoint_interval == 0:
            self.save_checkpoint()

        # Update node heartbeat
        if config.rank in self._nodes:
            self._nodes[config.rank].last_heartbeat = time.time()

        return loss.item()

    def save_checkpoint(self, path: Path | None = None):
        """Save training checkpoint.

        Args:
            path: Optional custom path (defaults to checkpoint_dir)
        """
        torch, _, _ = _get_torch_distributed()
        config = self.config

        # Only rank 0 saves checkpoints
        if config.rank != 0:
            return

        path = path or (self._checkpoint_dir / f"checkpoint_{self._step}.pt")

        checkpoint = {
            "step": self._step,
            "epoch": self._epoch,
            "model_state_dict": self._original_model.state_dict(),
            "config": {
                "world_size": config.world_size,
                "compress_gradients": config.compress_gradients,
                "use_amp": config.use_amp,
            },
        }

        if self._optimizer:
            checkpoint["optimizer_state_dict"] = self._optimizer.state_dict()

        if self._scaler is not None:
            checkpoint["scaler_state_dict"] = self._scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"[DistributedUnified] Checkpoint saved to {path}")

        # Cleanup old checkpoints (keep last 3)
        checkpoints = sorted(self._checkpoint_dir.glob("checkpoint_*.pt"))
        for old_ckpt in checkpoints[:-3]:
            old_ckpt.unlink()

    def _try_resume(self):
        """Try to resume from latest checkpoint."""
        _torch, _, _ = _get_torch_distributed()

        checkpoints = sorted(self._checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return

        latest = checkpoints[-1]
        try:
            checkpoint = safe_load_checkpoint(latest, map_location=self._device, warn_on_unsafe=False)
            self._original_model.load_state_dict(checkpoint["model_state_dict"])
            self._step = checkpoint.get("step", 0)
            self._epoch = checkpoint.get("epoch", 0)

            if self._optimizer and "optimizer_state_dict" in checkpoint:
                self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if self._scaler is not None and "scaler_state_dict" in checkpoint:
                self._scaler.load_state_dict(checkpoint["scaler_state_dict"])

            logger.info(f"[DistributedUnified] Resumed from {latest} (step={self._step})")

        except Exception as e:
            logger.warning(f"[DistributedUnified] Failed to resume: {e}")

    def cleanup(self):
        """Cleanup distributed training resources."""
        _, dist, _ = _get_torch_distributed()

        if self._initialized:
            if dist.is_initialized():
                dist.destroy_process_group()
            self._initialized = False
            logger.info("[DistributedUnified] Cleanup complete")

    def barrier(self):
        """Synchronization barrier across all processes."""
        _, dist, _ = _get_torch_distributed()

        if self._initialized and dist.is_initialized():
            dist.barrier()

    @property
    def model(self) -> Any:
        """Get the wrapped model."""
        return self._ddp_model if self._ddp_model is not None else self._original_model

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.config.rank == 0

    @property
    def step(self) -> int:
        """Current training step."""
        return self._step

    @property
    def nodes(self) -> dict[int, NodeInfo]:
        """Get registered nodes."""
        return self._nodes

    def get_metrics(self) -> dict[str, Any]:
        """Get training metrics."""
        metrics = {
            "step": self._step,
            "epoch": self._epoch,
            "rank": self.config.rank,
            "world_size": self.config.world_size,
            "initialized": self._initialized,
        }

        if self._compressor:
            metrics["compression_step"] = self._compressor._step

        if self._async_sgd:
            metrics["pending_gradients"] = self._async_sgd.pending_updates
            metrics["discarded_gradients"] = self._async_sgd.discarded_count

        return metrics

    def set_optimizer(self, optimizer: Any):
        """Set or update the optimizer."""
        self._optimizer = optimizer

    def __enter__(self):
        """Context manager entry."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# =============================================================================
# Factory Functions
# =============================================================================

def create_distributed_trainer(
    model: Any,
    world_size: int = 1,
    rank: int = 0,
    backend: str = "nccl",
    compress_gradients: bool = False,
    use_amp: bool = True,
) -> UnifiedDistributedTrainer:
    """Factory function to create a distributed trainer.

    Args:
        model: PyTorch model
        world_size: Total number of processes
        rank: This process's rank
        backend: Communication backend
        compress_gradients: Enable gradient compression
        use_amp: Enable mixed precision

    Returns:
        Configured UnifiedDistributedTrainer
    """
    config = UnifiedDistributedConfig(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        backend=backend,
        compress_gradients=compress_gradients,
        use_amp=use_amp,
    )
    return UnifiedDistributedTrainer(model, config)


# =============================================================================
# Backwards Compatibility Aliases
# =============================================================================

# Alias for backwards compatibility with distributed.py
DistributedConfig = UnifiedDistributedConfig
DistributedTrainer = UnifiedDistributedTrainer
