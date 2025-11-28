"""
Distributed training utilities for PyTorch DistributedDataParallel (DDP).

This module provides helper functions for setting up and managing distributed
training across multiple GPUs on a single machine or across multiple nodes.

Usage:
    # In training script when distributed mode is enabled:
    setup_distributed(rank, world_size)
    model = wrap_model_ddp(model, device)
    sampler = get_distributed_sampler(dataset)
    # ... training loop ...
    cleanup_distributed()

Launch with torchrun:
    torchrun --nproc_per_node=NUM_GPUS train.py --distributed --data data.npz

For the DistributedTrainer class:
    from app.training.distributed import DistributedTrainer
    from app.training.config import TrainConfig

    config = TrainConfig(epochs_per_iter=100, batch_size=64)
    trainer = DistributedTrainer(config, world_size=4)
    trainer.setup_data_parallel()
    trainer.train()
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)


def setup_distributed(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    backend: Optional[str] = None,
    init_method: Optional[str] = None,
) -> None:
    """
    Initialize the distributed process group.

    When using torchrun, rank and world_size are automatically set via
    environment variables (LOCAL_RANK, RANK, WORLD_SIZE). This function
    reads those if not explicitly provided.

    Args:
        rank: Process rank. If None, reads from RANK env var.
        world_size: Total number of processes. If None, reads from
            WORLD_SIZE env var.
        backend: Communication backend. If None, uses 'nccl' for CUDA,
            'gloo' otherwise.
        init_method: URL for process group initialization. If None,
            uses env:// method.

    Raises:
        RuntimeError: If distributed initialization fails.
    """
    # Read from environment if not provided (torchrun sets these)
    if rank is None:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Choose backend based on device availability
    if backend is None:
        if torch.cuda.is_available():
            backend = "nccl"
        else:
            backend = "gloo"

    # Default init method is env:// which uses MASTER_ADDR and MASTER_PORT
    if init_method is None:
        # Ensure MASTER_ADDR and MASTER_PORT are set for env://
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        init_method = "env://"

    # Initialize the process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )

    # Set the device for this process
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)


def cleanup_distributed() -> None:
    """
    Clean up the distributed process group.

    Should be called at the end of training to properly release resources.
    Safe to call even if distributed is not initialized.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    """
    Check if distributed training is currently active.

    Returns:
        True if running in distributed mode with initialized process group.
    """
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    """
    Check if this is the main process (rank 0).

    In distributed training, only rank 0 should perform logging, checkpointing,
    and other singleton operations.

    Returns:
        True if this is rank 0 or if not in distributed mode.
    """
    if not is_distributed():
        return True
    return get_rank() == 0


def get_rank() -> int:
    """
    Get the rank of the current process.

    Returns:
        Process rank (0 to world_size - 1), or 0 if not in distributed mode.
    """
    if not is_distributed():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """
    Get the total number of processes in the distributed group.

    Returns:
        World size, or 1 if not in distributed mode.
    """
    if not is_distributed():
        return 1
    return dist.get_world_size()


def get_local_rank() -> int:
    """
    Get the local rank of the current process on this node.

    The local rank is used to determine which GPU to use on multi-GPU nodes.

    Returns:
        Local rank from LOCAL_RANK env var, or 0 if not set.
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def get_distributed_sampler(
    dataset: Dataset,
    shuffle: bool = True,
    seed: int = 42,
    drop_last: bool = False,
) -> DistributedSampler:
    """
    Create a DistributedSampler for the given dataset.

    The DistributedSampler ensures each process in distributed training
    gets a unique subset of the data, preventing duplicate samples.

    Args:
        dataset: The dataset to sample from.
        shuffle: Whether to shuffle the data. Should be True for training.
        seed: Random seed for shuffling. Must be the same across all processes.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        A DistributedSampler configured for the current distributed setup.
    """
    return DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )


def wrap_model_ddp(
    model: torch.nn.Module,
    device: torch.device,
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = True,
) -> DDP:
    """
    Wrap a model with DistributedDataParallel.

    DDP wraps the model to synchronize gradients across all processes
    during the backward pass, enabling efficient distributed training.

    Args:
        model: The model to wrap. Should already be moved to the target device.
        device: The device the model is on. Used to set device_ids for CUDA.
        find_unused_parameters: Set to True if not all model outputs are used
            in the loss. This adds overhead but handles dynamic architectures.
        gradient_as_bucket_view: Reduces memory overhead by storing gradients
            directly in gradient buckets.

    Returns:
        The model wrapped with DistributedDataParallel.

    Note:
        Before wrapping, ensure the model is moved to the correct device and
        the distributed process group is initialized.
    """
    if device.type == "cuda":
        local_rank = get_local_rank()
        return DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )
    else:
        # For CPU or other devices, don't specify device_ids
        return DDP(
            model,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )


def synchronize() -> None:
    """
    Synchronize all processes by waiting at a barrier.

    Useful for ensuring all processes have completed a particular step
    before proceeding.
    """
    if is_distributed():
        dist.barrier()


def reduce_tensor(
    tensor: torch.Tensor,
    op: str = "sum",
) -> torch.Tensor:
    """
    Reduce a tensor across all processes.

    Args:
        tensor: The tensor to reduce. Will be modified in-place.
        op: The reduction operation name: "sum", "product", "min", "max".

    Returns:
        The reduced tensor (modified in-place).
    """
    if not is_distributed():
        return tensor
    
    op_map = {
        "sum": dist.ReduceOp.SUM,
        "product": dist.ReduceOp.PRODUCT,
        "min": dist.ReduceOp.MIN,
        "max": dist.ReduceOp.MAX,
    }
    reduce_op = op_map.get(op.lower(), dist.ReduceOp.SUM)
    dist.all_reduce(tensor, op=reduce_op)
    return tensor


def all_gather_object(obj: object) -> list:
    """
    Gather arbitrary Python objects from all processes.

    Args:
        obj: Any picklable Python object.

    Returns:
        List of objects from all processes, ordered by rank.
    """
    if not is_distributed():
        return [obj]
    
    world_size = get_world_size()
    output = [None] * world_size
    dist.all_gather_object(output, obj)
    return output


def broadcast_object(obj: object, src: int = 0) -> object:
    """
    Broadcast a Python object from source rank to all processes.

    Args:
        obj: Any picklable Python object (only used on src rank).
        src: Source rank to broadcast from.

    Returns:
        The broadcasted object.
    """
    if not is_distributed():
        return obj
    
    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def seed_everything(seed: int, rank_offset: bool = True) -> None:
    """
    Set random seeds for reproducibility across distributed processes.

    Args:
        seed: Base random seed.
        rank_offset: If True, add the rank to the seed so each process
            has a different but deterministic random state. This is typically
            desired for data augmentation but not for model initialization.
    """
    import random
    import numpy as np

    if rank_offset and is_distributed():
        seed = seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set deterministic algorithms where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device_for_rank() -> torch.device:
    """
    Get the appropriate device for the current process rank.

    In distributed training with GPUs, each rank should use a different GPU.
    This function maps local ranks to GPU indices.

    Returns:
        The device (cuda:N for GPU or cpu for CPU training).
    """
    if torch.cuda.is_available():
        local_rank = get_local_rank()
        return torch.device(f"cuda:{local_rank}")
    elif torch.backends.mps.is_available():
        # MPS (Apple Silicon) - only one device, no distribution
        return torch.device("mps")
    else:
        return torch.device("cpu")


def scale_learning_rate(
    base_lr: float,
    world_size: Optional[int] = None,
    scale_type: str = "linear",
) -> float:
    """
    Scale learning rate for distributed training.

    When using larger effective batch sizes (batch_size * world_size),
    the learning rate may need to be scaled accordingly.

    Args:
        base_lr: The base learning rate for single-GPU training.
        world_size: Number of processes. If None, uses get_world_size().
        scale_type: Scaling method:
            - "linear": Scale linearly with world size (LR * world_size)
            - "sqrt": Scale with square root (LR * sqrt(world_size))
            - "none": No scaling

    Returns:
        The scaled learning rate.
    """
    if world_size is None:
        world_size = get_world_size()

    if scale_type == "linear":
        return base_lr * world_size
    elif scale_type == "sqrt":
        import math
        return base_lr * math.sqrt(world_size)
    elif scale_type == "none":
        return base_lr
    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")


class DistributedMetrics:
    """
    Helper class for accumulating and reducing metrics across
    distributed processes.

    Usage:
        metrics = DistributedMetrics()
        for batch in loader:
            loss = compute_loss(batch)
            metrics.add("loss", loss.item())
            metrics.add("accuracy", compute_accuracy(batch))
        
        # After epoch, get averaged metrics across all processes
        avg_metrics = metrics.reduce_and_reset()
        if is_main_process():
            print(f"Average loss: {avg_metrics['loss']}")
    """

    def __init__(self):
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def add(self, name: str, value: float, count: int = 1) -> None:
        """
        Add a metric value.

        Args:
            name: Metric name.
            value: Metric value (will be multiplied by count for averaging).
            count: Number of samples this value represents.
        """
        if name not in self._sums:
            self._sums[name] = 0.0
            self._counts[name] = 0
        self._sums[name] += value * count
        self._counts[name] += count

    def reduce_and_reset(
        self,
        device: Optional[torch.device] = None,
    ) -> dict[str, float]:
        """
        Reduce metrics across all processes and reset accumulators.

        Args:
            device: Device to use for reduction tensors. If None, uses CPU.

        Returns:
            Dictionary of metric names to averaged values.
        """
        if device is None:
            device = torch.device("cpu")

        result = {}
        for name in self._sums:
            if is_distributed():
                # Create tensors for sum and count
                sum_tensor = torch.tensor(self._sums[name], device=device)
                count_tensor = torch.tensor(self._counts[name], device=device)
                
                # Reduce across all processes
                reduce_tensor(sum_tensor)
                reduce_tensor(count_tensor)
                
                # Compute average
                total_count = count_tensor.item()
                if total_count > 0:
                    result[name] = sum_tensor.item() / total_count
                else:
                    result[name] = 0.0
            else:
                # Non-distributed: just compute local average
                if self._counts[name] > 0:
                    result[name] = self._sums[name] / self._counts[name]
                else:
                    result[name] = 0.0

        # Reset accumulators
        self._sums.clear()
        self._counts.clear()

        return result

    def reset(self) -> None:
        """Reset all accumulators without reducing."""
        self._sums.clear()
        self._counts.clear()


# =============================================================================
# Distributed Trainer Class
# =============================================================================


class DistributedTrainer:
    """
    High-level distributed training coordinator.

    Integrates StreamingDataLoader for distributed data sharding,
    ModelVersionManager for versioned checkpoint management, and
    TrainConfig for configuration.

    Example usage::

        from app.training.config import TrainConfig
        from app.training.distributed import DistributedTrainer

        config = TrainConfig(
            epochs_per_iter=100,
            batch_size=64,
            learning_rate=0.001,
        )

        # Launch with torchrun:
        # torchrun --nproc_per_node=4 train_distributed.py

        trainer = DistributedTrainer(
            config=config,
            data_paths=['data1.npz', 'data2.npz'],
            model=my_model,
        )
        trainer.setup()
        trainer.train()
        trainer.cleanup()

    Attributes:
        config: Training configuration
        rank: Process rank (0 to world_size - 1)
        world_size: Total number of distributed processes
        is_main: True if this is rank 0 (main process)
    """

    def __init__(
        self,
        config: Any,  # TrainConfig - use Any to avoid circular import
        data_paths: Union[str, List[str]],
        model: nn.Module,
        checkpoint_dir: str = "checkpoints",
        resume_path: Optional[str] = None,
        scale_lr: bool = True,
        lr_scale_mode: str = "linear",
    ):
        """
        Initialize the distributed trainer.

        Args:
            config: TrainConfig instance with training parameters
            data_paths: Path(s) to training data files (.npz or .hdf5)
            model: PyTorch model to train (will be wrapped with DDP)
            checkpoint_dir: Directory for saving checkpoints
            resume_path: Optional path to checkpoint to resume from
            scale_lr: Whether to scale learning rate with world size
            lr_scale_mode: LR scaling mode ('linear' or 'sqrt')
        """
        self.config = config
        self.data_paths = (
            [data_paths] if isinstance(data_paths, str) else data_paths
        )
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.resume_path = resume_path
        self.scale_lr = scale_lr
        self.lr_scale_mode = lr_scale_mode

        # Will be set during setup
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.is_main = True
        self.device: torch.device = torch.device("cpu")

        # Training components (initialized in setup)
        self.wrapped_model: Optional[DDP] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.train_loader: Optional[Any] = None
        self.val_loader: Optional[Any] = None

        # Metrics and state
        self.metrics = DistributedMetrics()
        self.start_epoch = 0
        self.best_val_loss = float('inf')

        # Version manager for checkpoints
        self._version_manager: Optional[Any] = None

    def setup(self) -> None:
        """
        Set up distributed training environment.

        This method:
        1. Initializes the distributed process group
        2. Sets up the device for this rank
        3. Wraps the model with DDP
        4. Creates data loaders with proper sharding
        5. Initializes optimizer and scheduler
        6. Loads checkpoint if resuming
        """
        # Initialize distributed process group
        setup_distributed()

        self.rank = get_rank()
        self.world_size = get_world_size()
        self.local_rank = get_local_rank()
        self.is_main = is_main_process()

        # Set up device
        self.device = get_device_for_rank()

        if self.is_main:
            logger.info(
                f"Distributed training initialized: "
                f"rank {self.rank}/{self.world_size}, device {self.device}"
            )

        # Seed for reproducibility (with rank offset for different data)
        seed_everything(self.config.seed, rank_offset=True)

        # Move model to device and wrap with DDP
        self.model = self.model.to(self.device)
        self.wrapped_model = wrap_model_ddp(self.model, self.device)

        # Set up data loaders with sharding
        self.setup_data_parallel()

        # Scale learning rate if requested
        lr = self.config.learning_rate
        if self.scale_lr and self.world_size > 1:
            lr = scale_learning_rate(
                lr, self.world_size, scale_type=self.lr_scale_mode
            )
            if self.is_main:
                logger.info(
                    f"Scaled LR from {self.config.learning_rate:.6f} "
                    f"to {lr:.6f} ({self.lr_scale_mode})"
                )

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.wrapped_model.parameters(),
            lr=lr,
            weight_decay=self.config.weight_decay,
        )

        # Initialize version manager for checkpoints
        from app.training.model_versioning import ModelVersionManager
        self._version_manager = ModelVersionManager(
            default_device=self.device
        )

        # Create checkpoint directory
        if self.is_main:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Resume from checkpoint if specified
        if self.resume_path is not None:
            self._load_checkpoint(self.resume_path)

    def setup_data_parallel(self) -> None:
        """
        Setup data parallel training with proper sharding.

        Creates StreamingDataLoader instances with rank-based sharding
        so each worker gets unique samples.
        """
        from app.training.data_loader import StreamingDataLoader

        # Create training data loader with sharding
        self.train_loader = StreamingDataLoader(
            data_paths=self.data_paths,
            batch_size=self.config.batch_size,
            shuffle=True,
            seed=self.config.seed,
            rank=self.rank,
            world_size=self.world_size,
        )

        # Create validation loader (also sharded)
        self.val_loader = StreamingDataLoader(
            data_paths=self.data_paths,
            batch_size=self.config.batch_size,
            shuffle=False,
            seed=self.config.seed + 1000,
            rank=self.rank,
            world_size=self.world_size,
        )

        if self.is_main:
            total = self.train_loader.total_samples
            shard = self.train_loader.shard_size
            logger.info(
                f"Data sharding: {total} total samples, "
                f"{shard} per rank"
            )

    def train(self) -> Dict[str, float]:
        """
        Run the distributed training loop.

        Returns:
            Dictionary of final metrics (loss, etc.)
        """
        if self.wrapped_model is None or self.optimizer is None:
            raise RuntimeError("Call setup() before train()")

        if self.train_loader is None:
            raise RuntimeError("Data loaders not initialized")

        # Loss functions
        value_criterion = nn.MSELoss()
        policy_criterion = nn.KLDivLoss(reduction='batchmean')

        policy_weight = getattr(self.config, 'policy_weight', 1.0)

        final_metrics: Dict[str, float] = {}

        for epoch in range(self.start_epoch, self.config.epochs_per_iter):
            # Set epoch for shuffling
            self.train_loader.set_epoch(epoch)

            # Training phase
            self.wrapped_model.train()
            self.metrics.reset()

            for batch_idx, batch in enumerate(self.train_loader):
                (features, globals_vec), (value_targets, policy_targets) = (
                    batch
                )

                features = features.to(self.device)
                globals_vec = globals_vec.to(self.device)
                value_targets = value_targets.to(self.device)
                policy_targets = policy_targets.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                value_pred, policy_pred = self.wrapped_model(
                    features, globals_vec
                )

                # Compute loss
                policy_log_probs = torch.log_softmax(policy_pred, dim=1)
                value_loss = value_criterion(value_pred, value_targets)
                policy_loss = policy_criterion(
                    policy_log_probs, policy_targets
                )
                loss = value_loss + (policy_weight * policy_loss)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.wrapped_model.parameters(), max_norm=1.0
                )
                self.optimizer.step()

                # Track metrics
                self.metrics.add('train_loss', loss.item(), features.size(0))
                self.metrics.add('value_loss', value_loss.item(), 1)
                self.metrics.add('policy_loss', policy_loss.item(), 1)

                if batch_idx % 10 == 0 and self.is_main:
                    logger.info(
                        f"Epoch {epoch+1}, Batch {batch_idx}: "
                        f"loss={loss.item():.4f}"
                    )

            # Reduce metrics across all ranks
            train_metrics = self.metrics.reduce_and_reset(device=self.device)

            # Validation phase
            val_metrics = self._validate(value_criterion, policy_criterion)

            if self.is_main:
                logger.info(
                    f"Epoch [{epoch+1}/{self.config.epochs_per_iter}] "
                    f"Train Loss: {train_metrics.get('train_loss', 0):.4f}, "
                    f"Val Loss: {val_metrics.get('val_loss', 0):.4f}"
                )

            # Checkpoint (only rank 0)
            val_loss = val_metrics.get('val_loss', float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.checkpoint(epoch, val_loss, is_best=True)
            elif (epoch + 1) % 5 == 0:
                self.checkpoint(epoch, val_loss, is_best=False)

            final_metrics = {
                'train_loss': train_metrics.get('train_loss', 0),
                'val_loss': val_loss,
                'epoch': epoch + 1,
            }

        return final_metrics

    def _validate(
        self,
        value_criterion: nn.Module,
        policy_criterion: nn.Module,
    ) -> Dict[str, float]:
        """Run validation phase."""
        if self.wrapped_model is None or self.val_loader is None:
            return {}

        self.wrapped_model.eval()
        self.metrics.reset()

        policy_weight = getattr(self.config, 'policy_weight', 1.0)

        with torch.no_grad():
            for batch in self.val_loader:
                (features, globals_vec), (value_targets, policy_targets) = (
                    batch
                )

                features = features.to(self.device)
                globals_vec = globals_vec.to(self.device)
                value_targets = value_targets.to(self.device)
                policy_targets = policy_targets.to(self.device)

                value_pred, policy_pred = self.wrapped_model(
                    features, globals_vec
                )

                policy_log_probs = torch.log_softmax(policy_pred, dim=1)
                value_loss = value_criterion(value_pred, value_targets)
                policy_loss = policy_criterion(
                    policy_log_probs, policy_targets
                )
                loss = value_loss + (policy_weight * policy_loss)

                self.metrics.add('val_loss', loss.item(), features.size(0))

        return self.metrics.reduce_and_reset(device=self.device)

    def checkpoint(
        self,
        epoch: int,
        loss: float,
        is_best: bool = False,
    ) -> None:
        """
        Save a versioned checkpoint using ModelVersionManager.

        Only rank 0 saves checkpoints to avoid file conflicts.
        All ranks wait at a barrier after saving to ensure consistency.

        Args:
            epoch: Current epoch number
            loss: Current loss value
            is_best: Whether this is the best model so far
        """
        # Synchronize before checkpoint
        synchronize()

        # Only rank 0 saves
        if self.is_main and self._version_manager is not None:
            # Get underlying model (unwrap DDP)
            model_to_save = (
                self.wrapped_model.module
                if self.wrapped_model is not None
                else self.model
            )

            # Create checkpoint path
            if is_best:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, "best_model.pth"
                )
            else:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"
                )

            # Create metadata
            training_info = {
                'epoch': epoch + 1,
                'loss': float(loss),
                'world_size': self.world_size,
                'is_best': is_best,
            }

            metadata = self._version_manager.create_metadata(
                model_to_save,
                training_info=training_info,
            )

            # Save checkpoint
            self._version_manager.save_checkpoint(
                model_to_save,
                metadata,
                checkpoint_path,
                optimizer=self.optimizer,
                epoch=epoch,
                loss=loss,
            )

            logger.info(
                f"Saved checkpoint to {checkpoint_path} "
                f"(epoch {epoch+1}, loss {loss:.4f})"
            )

        # All ranks wait after checkpoint
        synchronize()

    def _load_checkpoint(self, path: str) -> None:
        """
        Load a checkpoint with barrier synchronization.

        All ranks load the checkpoint. Rank 0 logs the status.

        Args:
            path: Path to checkpoint file
        """
        # Barrier before load to ensure checkpoint is complete
        synchronize()

        if not os.path.exists(path):
            if self.is_main:
                logger.warning(f"Checkpoint not found: {path}")
            return

        if self._version_manager is None:
            return

        try:
            state_dict, metadata = self._version_manager.load_checkpoint(
                path,
                strict=False,  # Allow version mismatches with warning
                verify_checksum=True,
                device=self.device,
            )

            # Load into unwrapped model
            model_to_load = (
                self.wrapped_model.module
                if self.wrapped_model is not None
                else self.model
            )
            model_to_load.load_state_dict(state_dict)

            # Try to load epoch from training info
            if 'epoch' in metadata.training_info:
                self.start_epoch = metadata.training_info['epoch']

            if self.is_main:
                logger.info(
                    f"Loaded checkpoint from {path} "
                    f"(version {metadata.architecture_version}, "
                    f"epoch {self.start_epoch})"
                )

        except Exception as e:
            if self.is_main:
                logger.error(f"Failed to load checkpoint: {e}")

        # Barrier after load
        synchronize()

    def cleanup(self) -> None:
        """Clean up distributed training resources."""
        # Close data loaders
        if self.train_loader is not None:
            if hasattr(self.train_loader, 'close'):
                self.train_loader.close()
        if self.val_loader is not None:
            if hasattr(self.val_loader, 'close'):
                self.val_loader.close()

        # Clean up distributed process group
        cleanup_distributed()

        if self.is_main:
            logger.info("Distributed training cleanup complete")