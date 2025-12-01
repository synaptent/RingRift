"""
Training script for RingRift Neural Network AI
Includes validation split, checkpointing, early stopping, LR warmup,
and distributed training support via PyTorch DDP.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import random
import os
import copy
import argparse
import glob
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import logging
from app.ai.neural_net import (
    RingRiftCNN,
    HexNeuralNet,
    HEX_BOARD_SIZE,
    P_HEX,
)
from app.training.config import TrainConfig
from app.models import BoardType
from app.training.hex_augmentation import HexSymmetryTransform
from app.training.distributed import (  # noqa: E402
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    get_distributed_sampler,
    wrap_model_ddp,
    seed_everything,
    scale_learning_rate,
    DistributedMetrics,
)
from app.training.data_loader import (  # noqa: E402
    StreamingDataLoader,
    get_sample_count,
)
from app.training.model_versioning import (  # noqa: E402
    ModelVersionManager,
    save_model_checkpoint,
    VersionMismatchError,
    LegacyCheckpointError,
)
from app.training.seed_utils import seed_all

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Backwards-compatible alias that forwards to the shared training
# seeding utility so that existing callers importing seed_all from this
# module continue to work.
def seed_all_legacy(seed: int = 42) -> None:
    seed_all(seed)


class EarlyStopping:
    """
    Early stopping to terminate training when validation loss stops improving.
    
    Args:
        patience: Number of epochs to wait before stopping after
            last improvement
        min_delta: Minimum change in validation loss to qualify as improvement
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_state: Optional[Dict[str, Any]] = None
        self.should_stop = False
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop based on validation loss.
        
        Args:
            val_loss: Current validation loss
            model: Model to save state from if this is best so far
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
    
    def restore_best_weights(self, model: nn.Module) -> None:
        """Restore the best weights to the model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            logger.info("Restored best model weights from early stopping")


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    scheduler: Optional[Any] = None,
    early_stopping: Optional[EarlyStopping] = None,
    use_versioning: bool = True,
) -> None:
    """
    Save a training checkpoint with optional versioning metadata.

    Args:
        model: The model to save
        optimizer: The optimizer to save state from
        epoch: Current epoch number
        loss: Current loss value
        path: Path to save checkpoint to
        scheduler: Optional LR scheduler to save state from
        early_stopping: Optional early stopping tracker to save state from
        use_versioning: Whether to include versioning metadata (default True)
    """
    # Ensure directory exists
    dir_path = os.path.dirname(path) if os.path.dirname(path) else '.'
    os.makedirs(dir_path, exist_ok=True)

    if use_versioning:
        # Use versioned checkpoint format
        manager = ModelVersionManager()
        training_info = {
            'epoch': epoch,
            'loss': float(loss),
        }
        if early_stopping is not None:
            training_info['early_stopping'] = {
                'best_loss': early_stopping.best_loss,
                'counter': early_stopping.counter,
            }

        metadata = manager.create_metadata(
            model,
            training_info=training_info,
        )

        manager.save_checkpoint(
            model,
            metadata,
            path,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            loss=loss,
        )

        # Also save early stopping best_state if needed for resume
        if (
            early_stopping is not None
            and early_stopping.best_state is not None
        ):
            # Save early stopping state separately so it survives reloading
            es_path = path.replace('.pth', '_early_stopping.pth')
            torch.save(
                {
                    'best_loss': early_stopping.best_loss,
                    'counter': early_stopping.counter,
                    'best_state': early_stopping.best_state,
                },
                es_path,
            )

        logger.info(
            f"Saved versioned checkpoint to {path} "
            f"(epoch {epoch}, loss {loss:.4f}, "
            f"version {metadata.architecture_version})"
        )
    else:
        # Legacy format for backwards compatibility
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if early_stopping is not None:
            checkpoint['early_stopping'] = {
                'best_loss': early_stopping.best_loss,
                'counter': early_stopping.counter,
                'best_state': early_stopping.best_state,
            }

        torch.save(checkpoint, path)
        logger.info(
            "Saved legacy checkpoint to %s (epoch %d, loss %.4f)",
            path,
            epoch,
            loss,
        )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    early_stopping: Optional[EarlyStopping] = None,
    device: Optional[torch.device] = None,
    strict_versioning: bool = False,
) -> Tuple[int, float]:
    """
    Load a training checkpoint with optional version validation.

    Args:
        path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional LR scheduler to load state into
        early_stopping: Optional early stopping tracker to restore state into
        device: Device to map checkpoint tensors to
        strict_versioning: If True, fail on version mismatch. If False,
            log warnings but continue (default: False for backwards compat)

    Returns:
        Tuple of (epoch, loss) from the checkpoint

    Raises:
        VersionMismatchError: If strict_versioning and version mismatch
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Check if this is a versioned checkpoint
    manager = ModelVersionManager(default_device=device)
    if manager.METADATA_KEY in checkpoint:
        # Versioned checkpoint
        try:
            state_dict, metadata = manager.load_checkpoint(
                path,
                strict=strict_versioning,
                verify_checksum=True,
                device=device,
            )
            model.load_state_dict(state_dict)
            logger.info(
                f"Loaded versioned checkpoint from {path} "
                f"(version {metadata.architecture_version})"
            )

            # Extract epoch/loss from metadata or checkpoint
            epoch = checkpoint.get('epoch', 0)
            loss = checkpoint.get('loss', float('inf'))

        except (VersionMismatchError, LegacyCheckpointError) as e:
            if strict_versioning:
                raise
            logger.warning(f"Version issue loading checkpoint: {e}")
            # Fall back to direct loading
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 0)
            loss = checkpoint.get('loss', float('inf'))
    else:
        # Legacy checkpoint format
        logger.info(f"Loading legacy checkpoint from {path}")
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))

    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Load early stopping state
    if early_stopping is not None:
        if 'early_stopping' in checkpoint:
            es_state = checkpoint['early_stopping']
            early_stopping.best_loss = es_state['best_loss']
            early_stopping.counter = es_state['counter']
            early_stopping.best_state = es_state.get('best_state')
        else:
            # Check for separate early stopping file
            es_path = path.replace('.pth', '_early_stopping.pth')
            if os.path.exists(es_path):
                es_state = torch.load(es_path, map_location=device)
                early_stopping.best_loss = es_state['best_loss']
                early_stopping.counter = es_state['counter']
                early_stopping.best_state = es_state.get('best_state')

    logger.info(
        f"Loaded checkpoint from {path} (epoch {epoch}, loss {loss:.4f})"
    )
    return epoch, loss


def get_warmup_scheduler(
    optimizer: optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    scheduler_type: str = 'none',
) -> Optional[Any]:
    """
    Create a learning rate scheduler with optional warmup.
    
    This is the legacy warmup scheduler that uses LambdaLR for simple
    scheduling. For advanced cosine annealing, use create_lr_scheduler()
    instead.
    
    Args:
        optimizer: The optimizer to schedule
        warmup_epochs: Number of epochs for linear warmup (0 to disable)
        total_epochs: Total number of training epochs
        scheduler_type: Type of scheduler after warmup
            ('none', 'step', 'cosine')
        
    Returns:
        LR scheduler or None if no scheduling requested
    """
    if warmup_epochs == 0 and scheduler_type == 'none':
        return None
    
    def lr_lambda(epoch: int) -> float:
        # Linear warmup phase
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        
        # Post-warmup phase
        if scheduler_type == 'none':
            return 1.0
        elif scheduler_type == 'step':
            # Step decay: reduce by 0.5 every 10 epochs after warmup
            steps = (epoch - warmup_epochs) // 10
            return 0.5 ** steps
        elif scheduler_type == 'cosine':
            # Cosine annealing after warmup
            remaining = max(1, total_epochs - warmup_epochs)
            progress = (epoch - warmup_epochs) / remaining
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        else:
            return 1.0
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_lr_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str,
    total_epochs: int,
    warmup_epochs: int = 0,
    lr_min: float = 1e-6,
    lr_t0: int = 10,
    lr_t_mult: int = 2,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """
    Create a learning rate scheduler with PyTorch's native implementations.
    
    Supports cosine annealing with optional warmup using SequentialLR to chain
    a linear warmup scheduler with the main scheduler.
    
    Args:
        optimizer: The optimizer to schedule
        scheduler_type: Type of scheduler:
            - 'none': No scheduling (returns None)
            - 'step': Step decay (legacy, uses LambdaLR)
            - 'cosine': CosineAnnealingLR to lr_min over total_epochs
            - 'cosine-warm-restarts': CosineAnnealingWarmRestarts with
              T_0, T_mult
        total_epochs: Total number of training epochs
        warmup_epochs: Number of epochs for linear warmup (0 to disable)
        lr_min: Minimum learning rate for cosine annealing (eta_min)
        lr_t0: T_0 parameter for CosineAnnealingWarmRestarts
            (initial restart period)
        lr_t_mult: T_mult parameter for CosineAnnealingWarmRestarts
            (period multiplier)
        
    Returns:
        LR scheduler or None if scheduler_type is 'none' and warmup_epochs is 0
    """
    # For legacy 'step' scheduler or 'none' with warmup, use the old function
    if scheduler_type in ('none', 'step'):
        return get_warmup_scheduler(
            optimizer, warmup_epochs, total_epochs, scheduler_type
        )
    
    # Create the main scheduler based on type
    if scheduler_type == 'cosine':
        # Calculate T_max: epochs for cosine annealing (after warmup)
        t_max = max(1, total_epochs - warmup_epochs)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=lr_min
        )
    elif scheduler_type == 'cosine-warm-restarts':
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=lr_t0, T_mult=lr_t_mult, eta_min=lr_min
        )
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}, using none")
        return None
    
    # If no warmup, return the main scheduler directly
    if warmup_epochs == 0:
        return main_scheduler
    
    # Create warmup scheduler using LinearLR
    # LinearLR scales the learning rate from start_factor to end_factor
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / max(1, warmup_epochs),  # Start at lr/warmup_epochs
        end_factor=1.0,  # End at full lr
        total_iters=warmup_epochs,
    )
    
    # Chain warmup and main scheduler using SequentialLR
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )
    
    return combined_scheduler


class RingRiftDataset(Dataset):
    """
    Dataset of self-play positions for a single board geometry.

    Each .npz file is expected to be *homogeneous* in board type/size so that
    mini-batches drawn by a vanilla DataLoader contain only one spatial shape.
    This keeps the training input compatible with the CNN and with
    NeuralNetAI.evaluate_batch, which enforces same-board-per-batch semantics.

    Future multi-board runs can either:
      - use separate datasets per board type/size, or
      - introduce a higher-level sampler/collate_fn that groups samples by
        geometry before feeding them to the network.
    
    Note: Terminal states (samples with empty policy arrays) are automatically
    filtered out during loading to prevent NaN losses when using KLDivLoss.
    Empty policy targets would otherwise cause the loss to become undefined.
    
    Args:
        data_path: Path to the .npz training data file
        board_type: Board geometry type (for augmentation)
        augment_hex: Enable D6 symmetry augmentation for hex boards
    """

    def __init__(
        self,
        data_path: str,
        board_type: BoardType = BoardType.SQUARE8,
        augment_hex: bool = False,
    ):
        self.data_path = data_path
        self.board_type = board_type
        self.augment_hex = augment_hex and board_type == BoardType.HEXAGONAL
        self.hex_transform: Optional[HexSymmetryTransform] = None
        
        # Initialize hex transform if augmentation enabled
        if self.augment_hex:
            self.hex_transform = HexSymmetryTransform(board_size=21)
            logger.info("Hex symmetry augmentation enabled (D6 group)")
        
        self.length = 0
        # Memory-mapped file object (np.lib.npyio.NpzFile) or in-memory dict
        self.data = None
        # Optional metadata inferred from the underlying npz file to aid
        # future multi-board training tooling.
        self.spatial_shape = None  # (H, W) of feature maps, if known
        self.board_type_meta = None
        self.board_size_meta = None
        # List of valid sample indices (those with non-empty policies)
        self.valid_indices = None

        if os.path.exists(data_path):
            try:
                # Load in mmap mode to avoid loading everything into RAM.
                # Note: np.load with mmap_mode returns a NpzFile where numeric
                # arrays are memory-mapped. We use allow_pickle=True so object
                # arrays (sparse policies) can still be loaded on demand.
                self.data = np.load(
                    data_path,
                    mmap_mode='r',
                    allow_pickle=True,
                )

                if 'features' in self.data:
                    total_samples = len(self.data['values'])
                    
                    # Filter out samples with empty policies (terminal states)
                    # These would cause NaN when computing KLDivLoss
                    policy_indices_arr = self.data['policy_indices']
                    self.valid_indices = [
                        i for i in range(total_samples)
                        if len(policy_indices_arr[i]) > 0
                    ]
                    
                    filtered_count = total_samples - len(self.valid_indices)
                    if filtered_count > 0:
                        logger.info(
                            f"Filtered {filtered_count} terminal states "
                            f"with empty policies out of {total_samples} "
                            f"total samples"
                        )
                    
                    self.length = len(self.valid_indices)
                    
                    if self.length == 0:
                        logger.warning(
                            f"All {total_samples} samples in {data_path} "
                            f"have empty policies (terminal states). "
                            f"Dataset is empty."
                        )
                    else:
                        logger.info(
                            f"Loaded {self.length} valid training samples "
                            f"from {data_path} (mmap)"
                        )

                    # Optional per-dataset metadata for multi-board training.
                    # Newer datasets may include scalar or per-sample arrays
                    # named 'board_type' and/or 'board_size'. Older datasets
                    # will simply omit these keys.
                    files = getattr(self.data, "files", [])
                    if "board_type" in files:
                        self.board_type_meta = self.data["board_type"]
                    if "board_size" in files:
                        self.board_size_meta = self.data["board_size"]

                    # Infer the canonical spatial shape (H, W) once so that
                    # callers can route samples into same-board batches if
                    # mixed-geometry datasets are ever introduced.
                    try:
                        # Use first valid sample if available
                        if self.valid_indices:
                            first_valid = self.valid_indices[0]
                            sample = self.data["features"][first_valid]
                        else:
                            sample = self.data["features"][0]
                        if sample.ndim >= 3:
                            self.spatial_shape = tuple(sample.shape[-2:])
                    except Exception:
                        # Best-effort only; training will still work as long
                        # as individual samples are well-formed.
                        self.spatial_shape = None
                else:
                    print("Invalid data format in npz")
                    self.length = 0
            except Exception as e:
                print(f"Error loading data: {e}")
                self.length = 0
        else:
            print(f"Data file {data_path} not found, generating dummy data")
            # Generate dummy data in memory for testing
            # Ensure all dummy samples have non-empty policies
            dummy_count = 100
            self.data = {
                'features': np.random.rand(
                    dummy_count, 40, 8, 8
                ).astype(np.float32),
                'globals': np.random.rand(dummy_count, 10).astype(np.float32),
                'values': np.random.choice(
                    [1.0, 0.0, -1.0],
                    size=dummy_count,
                ).astype(np.float32),
                'policy_indices': np.array([
                    np.random.choice(55000, 5, replace=False).astype(np.int32)
                    for _ in range(dummy_count)
                ], dtype=object),
                'policy_values': np.array([
                    np.random.rand(5).astype(np.float32)
                    for _ in range(dummy_count)
                ], dtype=object),
            }
            self.valid_indices = list(range(dummy_count))
            self.length = dummy_count
            self.spatial_shape = (8, 8)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.length == 0:
            raise IndexError("Dataset is empty")

        if self.data is None:
            raise RuntimeError(
                "RingRiftDataset backing store is not initialised. "
                "This usually indicates a failed load."
            )

        # Map through valid_indices to get actual data index
        # This skips terminal states with empty policies
        if self.valid_indices is not None:
            actual_idx = self.valid_indices[idx]
        else:
            actual_idx = idx

        # Access data from memory-mapped arrays. We copy to ensure we have a
        # writable tensor if needed, and to detach from the mmap backing
        # store.
        features = np.array(self.data['features'][actual_idx])
        globals_vec = np.array(self.data['globals'][actual_idx])
        value = np.array(self.data['values'][actual_idx])

        # Policy is stored as object array of arrays (sparse). mmap does not
        # support object arrays directly, so these may be fully loaded into
        # memory depending on how the npz was written. For very large datasets
        # a CSR-style encoding would be preferable, but for now we assume the
        # object array fits in memory or is handled by OS paging.
        policy_indices = self.data['policy_indices'][actual_idx]
        policy_values = self.data['policy_values'][actual_idx]
        
        # Apply hex symmetry augmentation on-the-fly if enabled
        # This expands effective dataset size by 12x without extra memory
        if self.augment_hex and self.hex_transform is not None:
            # Pick a random transformation from the D6 group (0-11)
            transform_id = random.randint(0, 11)
            
            if transform_id != 0:  # 0 is identity, skip for efficiency
                # Transform the feature tensor
                features = self.hex_transform.transform_board(
                    features, transform_id
                )
                
                # Transform sparse policy
                indices_arr = np.asarray(policy_indices, dtype=np.int32)
                values_arr = np.asarray(policy_values, dtype=np.float32)
                policy_indices, policy_values = (
                    self.hex_transform.transform_sparse_policy(
                        indices_arr, values_arr, transform_id
                    )
                )
            
        # Reconstruct dense policy vector on-the-fly
        # Since we filter for non-empty policies, this should always have data
        policy_vector = torch.zeros(55000, dtype=torch.float32)
        
        if len(policy_indices) > 0:
            # Convert to proper numpy arrays with correct dtype
            # The object array may contain arrays that need explicit casting
            indices_arr = np.asarray(policy_indices, dtype=np.int64)
            values_arr = np.asarray(policy_values, dtype=np.float32)
            policy_vector[indices_arr] = torch.from_numpy(values_arr)
            
        return (
            torch.from_numpy(features),
            torch.from_numpy(globals_vec),
            torch.tensor([value.item()], dtype=torch.float32),
            policy_vector
        )


def train_model(
    config: TrainConfig,
    data_path: Union[str, List[str]],
    save_path: str,
    early_stopping_patience: int = 10,
    checkpoint_dir: str = 'checkpoints',
    checkpoint_interval: int = 5,
    warmup_epochs: int = 0,
    lr_scheduler: str = 'none',
    lr_min: float = 1e-6,
    lr_t0: int = 10,
    lr_t_mult: int = 2,
    resume_path: Optional[str] = None,
    augment_hex_symmetry: bool = False,
    distributed: bool = False,
    local_rank: int = -1,
    scale_lr: bool = False,
    lr_scale_mode: str = 'linear',
    find_unused_parameters: bool = False,
    use_streaming: bool = False,
    data_dir: Optional[str] = None,
):
    """
    Train the RingRift neural network model.

    Args:
        config: Training configuration
        data_path: Path(s) to training data (.npz file or list of files)
        save_path: Path to save the best model weights
        early_stopping_patience: Number of epochs without improvement before
            stopping (0 to disable early stopping)
        checkpoint_dir: Directory for saving periodic checkpoints
        checkpoint_interval: Save checkpoint every N epochs
        warmup_epochs: Number of epochs for LR warmup (0 to disable)
        lr_scheduler: Type of LR scheduler:
            - 'none': No scheduling (constant LR after warmup)
            - 'step': Step decay by 0.5 every 10 epochs
            - 'cosine': CosineAnnealingLR over remaining epochs
            - 'cosine-warm-restarts': CosineAnnealingWarmRestarts
        lr_min: Minimum learning rate for cosine annealing (default: 1e-6)
        lr_t0: T_0 for CosineAnnealingWarmRestarts (initial restart period)
        lr_t_mult: T_mult for CosineAnnealingWarmRestarts (period multiplier)
        resume_path: Path to checkpoint to resume training from
        augment_hex_symmetry: Enable D6 symmetry augmentation for hex boards
        distributed: Enable distributed training with DDP
        local_rank: Local rank for distributed training (set by torchrun)
        scale_lr: Whether to scale learning rate with world size
        lr_scale_mode: LR scaling mode ('linear' or 'sqrt')
        find_unused_parameters: Enable find_unused_parameters for DDP
        use_streaming: Use StreamingDataLoader for large datasets
        data_dir: Directory containing multiple .npz files (for streaming)
    """
    # Set up distributed training if enabled
    if distributed:
        # Setup distributed process group
        setup_distributed(local_rank)
        world_size = get_world_size()
        
        # Seed with rank offset for different random state per process
        seed_everything(config.seed, rank_offset=True)
        
        # Scale learning rate if requested
        if scale_lr:
            config.learning_rate = scale_learning_rate(
                config.learning_rate, world_size, scale_type=lr_scale_mode
            )
            if is_main_process():
                logger.info(
                    f"Scaled learning rate to {config.learning_rate:.6f} "
                    f"({lr_scale_mode} scaling with world_size={world_size})"
                )
    else:
        seed_all(config.seed)

    # Device configuration
    if distributed:
        # In distributed mode, use the local_rank device
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        if is_main_process():
            logger.info(
                f"Distributed training on device: {device} "
                f"(rank {get_rank()}/{get_world_size()})"
            )
    else:
        # Standard single-device selection
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        logger.info(f"Using device: {device}")

    # Determine canonical spatial board_size for the CNN from config.
    if config.board_type == BoardType.SQUARE19:
        board_size = 19
    elif config.board_type == BoardType.HEXAGONAL:
        # For hex boards we use the canonical 2 * radius + 1 mapping used by
        # the feature encoder. With the default size parameter 11
        # (see create_initial_state), this yields a 21×21 grid.
        board_size = HEX_BOARD_SIZE  # 21
    else:
        # Default to 8×8.
        board_size = 8

    # Determine whether to use HexNeuralNet for hexagonal boards
    use_hex_model = config.board_type == BoardType.HEXAGONAL

    if not distributed or is_main_process():
        if use_hex_model:
            logger.info(
                f"Initializing HexNeuralNet with board_size={board_size}"
            )
        else:
            logger.info(
                f"Initializing RingRiftCNN with board_size={board_size}"
            )

    # Initialize model based on board type
    if use_hex_model:
        # HexNeuralNet for hexagonal boards
        # in_channels = 10 base channels * (history_length + 1)
        hex_in_channels = 10 * (config.history_length + 1)
        model = HexNeuralNet(
            in_channels=hex_in_channels,
            global_features=10,
            num_res_blocks=8,
            num_filters=128,
            board_size=board_size,
            policy_size=P_HEX,
        )
    else:
        # RingRiftCNN for square boards
        model = RingRiftCNN(
            board_size=board_size,
            in_channels=10,
            global_features=10,
            history_length=config.history_length
        )
    model.to(device)

    # Load existing weights if available to continue training
    if os.path.exists(save_path):
        try:
            model.load_state_dict(
                torch.load(save_path, map_location=device, weights_only=True)
            )
            if not distributed or is_main_process():
                logger.info(f"Loaded existing model weights from {save_path}")
        except Exception as e:
            if not distributed or is_main_process():
                logger.warning(
                    f"Could not load existing weights: {e}. Starting fresh."
                )

    # Wrap model with DDP if using distributed training
    if distributed:
        model = wrap_model_ddp(
            model, device,
            find_unused_parameters=find_unused_parameters
        )
        if is_main_process():
            logger.info("Model wrapped with DistributedDataParallel")

    # Loss functions
    value_criterion = nn.MSELoss()
    policy_criterion = nn.KLDivLoss(reduction='batchmean')

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler with optional warmup
    # Use the new create_lr_scheduler for advanced cosine options
    epoch_scheduler = create_lr_scheduler(
        optimizer,
        scheduler_type=lr_scheduler,
        total_epochs=config.epochs_per_iter,
        warmup_epochs=warmup_epochs,
        lr_min=lr_min,
        lr_t0=lr_t0,
        lr_t_mult=lr_t_mult,
    )
    
    # ReduceLROnPlateau as fallback if no scheduler configured
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    ) if epoch_scheduler is None else None
    
    # Early stopping
    early_stopper: Optional[EarlyStopping] = None
    if early_stopping_patience > 0:
        early_stopper = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=0.0001,
        )
    
    # Track starting epoch for resume
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if resume_path is not None and os.path.exists(resume_path):
        # For DDP, load into the underlying model
        model_to_load = cast(
            nn.Module,
            model.module if distributed else model,
        )
        start_epoch, _ = load_checkpoint(
            resume_path,
            model_to_load,
            optimizer,
            scheduler=epoch_scheduler,
            early_stopping=early_stopper,
            device=device,
        )
        start_epoch += 1  # Start from next epoch
        if not distributed or is_main_process():
            logger.info(f"Resuming training from epoch {start_epoch}")
    
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Mixed precision scaler
    # Note: GradScaler is primarily for CUDA.
    # For MPS, mixed precision support is evolving.
    # We'll enable it only for CUDA for now to be safe.
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    train_streaming_loader: Optional[StreamingDataLoader] = None
    val_streaming_loader: Optional[StreamingDataLoader] = None
    train_loader: Optional[DataLoader] = None
    val_loader: Optional[DataLoader] = None
    train_sampler = None
    val_sampler = None

    # Collect data paths for streaming mode
    data_paths: List[str] = []
    if use_streaming:
        # Use streaming data loader for large datasets
        if data_dir is not None:
            # Collect all .npz files from directory
            npz_pattern = os.path.join(data_dir, "*.npz")
            data_paths = sorted(glob.glob(npz_pattern))
            if not distributed or is_main_process():
                logger.info(
                    f"Found {len(data_paths)} .npz files in {data_dir}"
                )
        elif isinstance(data_path, list):
            data_paths = data_path
        else:
            data_paths = [data_path]

        if not data_paths:
            if not distributed or is_main_process():
                logger.warning("No data files found for streaming; skipping.")
            if distributed:
                cleanup_distributed()
            return

        # Get total sample count across all files
        total_samples = sum(
            get_sample_count(p) for p in data_paths if os.path.exists(p)
        )

        if total_samples == 0:
            if not distributed or is_main_process():
                logger.warning("No samples found in data files; skipping.")
            if distributed:
                cleanup_distributed()
            return

        if not distributed or is_main_process():
            logger.info(
                f"StreamingDataLoader: {total_samples} total samples "
                f"across {len(data_paths)} files"
            )

        # Create streaming data loaders (80/20 split approximated by files)
        # For simplicity, we use all data for training in streaming mode
        # and compute validation on a subset
        val_split = 0.2
        val_samples = int(total_samples * val_split)
        train_samples = total_samples - val_samples

        # Determine rank/world_size for distributed data sharding
        if distributed:
            stream_rank = get_rank()
            stream_world_size = get_world_size()
        else:
            stream_rank = 0
            stream_world_size = 1

        train_streaming_loader = StreamingDataLoader(
            data_paths=data_paths,
            batch_size=config.batch_size,
            shuffle=True,
            seed=config.seed,
            drop_last=False,
            rank=stream_rank,
            world_size=stream_world_size,
        )

        # For validation, we use the same files but with different seed
        # and limit samples conceptually
        val_streaming_loader = StreamingDataLoader(
            data_paths=data_paths,
            batch_size=config.batch_size,
            shuffle=False,
            seed=config.seed + 1000,
            drop_last=False,
            rank=stream_rank,
            world_size=stream_world_size,
        )

        train_sampler = None
        train_size = train_samples
        val_size = val_samples

    else:
        # Legacy single-file loading with RingRiftDataset
        if isinstance(data_path, list):
            data_path_str = data_path[0] if data_path else ""
        else:
            data_path_str = data_path

        full_dataset = RingRiftDataset(
            data_path_str,
            board_type=config.board_type,
            augment_hex=augment_hex_symmetry,
        )

        if len(full_dataset) == 0:
            if not distributed or is_main_process():
                logger.warning(
                    "Training dataset at %s is empty; skipping.",
                    data_path_str,
                )
            if distributed:
                cleanup_distributed()
            return

        # Log spatial shape if available
        shape = getattr(full_dataset, "spatial_shape", None)
        if shape is not None and (not distributed or is_main_process()):
            h, w = shape
            logger.info(
                "Dataset spatial feature shape inferred as %dx%d.",
                h,
                w,
            )

        # Split into train/val (80/20)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

        # Create data loaders with distributed samplers if needed
        if distributed:
            train_sampler = get_distributed_sampler(
                train_dataset,
                shuffle=True,
            )
            val_sampler = get_distributed_sampler(
                val_dataset,
                shuffle=False,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=False,  # Sampler handles shuffling
                sampler=train_sampler,
                num_workers=2,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=2,
                pin_memory=True,
            )
        else:
            train_sampler = None
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
            )

    if not distributed or is_main_process():
        logger.info(
            f"Starting training for {config.epochs_per_iter} epochs..."
        )
        logger.info(f"Train size: {train_size}, Val size: {val_size}")
        if use_streaming:
            logger.info("Using StreamingDataLoader for memory-efficient data")
            if distributed:
                logger.info(
                    f"  Data sharding: rank {get_rank()}/{get_world_size()}, "
                    f"~{train_size // get_world_size()} samples per rank"
                )
        if distributed:
            logger.info(
                f"Distributed training with {get_world_size()} processes"
            )
        if early_stopper is not None:
            logger.info(
                f"Early stopping enabled with patience: "
                f"{early_stopping_patience}"
            )
        if warmup_epochs > 0:
            logger.info(f"LR warmup enabled for {warmup_epochs} epochs")
        if lr_scheduler in ('cosine', 'cosine-warm-restarts'):
            logger.info(
                f"LR scheduler: {lr_scheduler} (min_lr={lr_min})"
            )
            if lr_scheduler == 'cosine-warm-restarts':
                logger.info(f"  T_0={lr_t0}, T_mult={lr_t_mult}")
        logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    # Initialize distributed metrics tracker
    dist_metrics = DistributedMetrics() if distributed else None

    best_val_loss = float('inf')
    avg_val_loss = float('inf')  # Initialize for final checkpoint

    try:
        for epoch in range(start_epoch, config.epochs_per_iter):
            # Set epoch for distributed sampler or streaming loader
            if distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if use_streaming:
                assert train_streaming_loader is not None
                assert val_streaming_loader is not None
                train_streaming_loader.set_epoch(epoch)
                val_streaming_loader.set_epoch(epoch)

            # Training
            model.train()
            train_loss = 0.0
            train_batches = 0
            if dist_metrics is not None:
                dist_metrics.reset()

            # Select appropriate data source
            if use_streaming:
                assert train_streaming_loader is not None
                train_data_iter = iter(train_streaming_loader)
            else:
                assert train_loader is not None
                train_data_iter = iter(train_loader)

            for i, batch_data in enumerate(train_data_iter):
                # Handle both streaming and legacy batch formats
                if use_streaming:
                    (
                        (features, globals_vec),
                        (value_targets, policy_targets),
                    ) = batch_data
                else:
                    (
                        features,
                        globals_vec,
                        value_targets,
                        policy_targets,
                    ) = batch_data

                features = features.to(device)
                globals_vec = globals_vec.to(device)
                value_targets = value_targets.to(device)
                policy_targets = policy_targets.to(device)

                optimizer.zero_grad()

                # Autocast for mixed precision (CUDA only usually).
                # For MPS, we might need to check torch.autocast with
                # device_type="mps", but it is safer to stick to float32
                # on MPS if unsure.
                use_amp = device.type == 'cuda'

                with torch.cuda.amp.autocast(enabled=use_amp):
                    value_pred, policy_pred = model(features, globals_vec)

                    # Apply log_softmax to policy prediction for KLDivLoss
                    policy_log_probs = torch.log_softmax(policy_pred, dim=1)

                    value_loss = value_criterion(value_pred, value_targets)
                    policy_loss = policy_criterion(
                        policy_log_probs,
                        policy_targets,
                    )
                    loss = value_loss + (config.policy_weight * policy_loss)

                scaler.scale(loss).backward()

                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                train_batches += 1

                # Track metrics for distributed reduction
                if dist_metrics is not None:
                    dist_metrics.add(
                        'train_loss',
                        loss.item(),
                        features.size(0),
                    )

                if i % 10 == 0 and (not distributed or is_main_process()):
                    logger.info(
                        f"Epoch {epoch+1}, Batch {i}: "
                        f"Loss={loss.item():.4f} "
                        f"(Val={value_loss.item():.4f}, "
                        f"Pol={policy_loss.item():.4f})"
                    )

            # Compute average training loss
            if distributed and dist_metrics is not None:
                # Synchronize metrics across all processes
                train_metrics = dist_metrics.reduce_and_reset(device=device)
                avg_train_loss = train_metrics.get('train_loss', 0.0)
            elif train_batches > 0:
                avg_train_loss = train_loss / train_batches
            else:
                avg_train_loss = 0.0

            # Validation
            model.eval()
            val_loss = 0.0
            val_batches = 0
            if dist_metrics is not None:
                dist_metrics.reset()

            # Select appropriate validation data source
            if use_streaming:
                assert val_streaming_loader is not None
                val_data_iter = iter(val_streaming_loader)
                # Limit validation to ~20% of batches for streaming
                max_val_batches = max(
                    1,
                    len(val_streaming_loader) // 5,
                )
            else:
                assert val_loader is not None
                val_data_iter = iter(val_loader)
                max_val_batches = float('inf')

            with torch.no_grad():
                for val_batch_idx, val_batch in enumerate(val_data_iter):
                    if val_batch_idx >= max_val_batches:
                        break

                    # Handle both streaming and legacy batch formats
                    if use_streaming:
                        (
                            (features, globals_vec),
                            (value_targets, policy_targets),
                        ) = val_batch
                    else:
                        (
                            features,
                            globals_vec,
                            value_targets,
                            policy_targets,
                        ) = val_batch

                    features = features.to(device)
                    globals_vec = globals_vec.to(device)
                    value_targets = value_targets.to(device)
                    policy_targets = policy_targets.to(device)

                    # For DDP, forward through the wrapped model
                    value_pred, policy_pred = model(features, globals_vec)

                    policy_log_probs = torch.log_softmax(policy_pred, dim=1)

                    v_loss = value_criterion(value_pred, value_targets)
                    p_loss = policy_criterion(
                        policy_log_probs, policy_targets
                    )
                    loss = v_loss + (config.policy_weight * p_loss)
                    val_loss += loss.item()
                    val_batches += 1

                    # Track metrics for distributed reduction
                    if dist_metrics is not None:
                        dist_metrics.add(
                            'val_loss', loss.item(), features.size(0)
                        )

            # Compute average validation loss
            if distributed and dist_metrics is not None:
                val_metrics = dist_metrics.reduce_and_reset(device=device)
                avg_val_loss = val_metrics.get('val_loss', 0.0)
            elif val_batches > 0:
                avg_val_loss = val_loss / val_batches
            else:
                avg_val_loss = 0.0

            # Update scheduler at end of epoch
            if epoch_scheduler is not None:
                epoch_scheduler.step()
            elif plateau_scheduler is not None:
                plateau_scheduler.step(avg_val_loss)
            
            # Always log current learning rate
            if not distributed or is_main_process():
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"  Current LR: {current_lr:.6f}")

            if not distributed or is_main_process():
                logger.info(
                    f"Epoch [{epoch+1}/{config.epochs_per_iter}], "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}"
                )

            # Check early stopping (only on main process for DDP)
            # Get model for checkpointing (unwrap DDP if needed)
            model_to_save = cast(
                nn.Module,
                model.module if distributed else model,
            )

            if early_stopper is not None:
                should_stop = early_stopper(avg_val_loss, model_to_save)
                if should_stop:
                    if not distributed or is_main_process():
                        logger.info(
                            f"Early stopping triggered at epoch {epoch+1} "
                            f"(best loss: {early_stopper.best_loss:.4f})"
                        )
                        # Restore best weights
                        early_stopper.restore_best_weights(model_to_save)
                        # Save final checkpoint with best weights
                        final_checkpoint_path = os.path.join(
                            checkpoint_dir,
                            f"checkpoint_early_stop_epoch_{epoch+1}.pth",
                        )
                        save_checkpoint(
                            model_to_save,
                            optimizer,
                            epoch,
                            early_stopper.best_loss,
                            final_checkpoint_path,
                            scheduler=epoch_scheduler,
                            early_stopping=early_stopper,
                        )
                        # Save best model with versioning
                        save_model_checkpoint(
                            model_to_save,
                            save_path,
                            training_info={
                                'epoch': epoch,
                                'loss': float(early_stopper.best_loss),
                                'early_stopped': True,
                            },
                        )
                        logger.info("Best model saved to %s", save_path)
                    break

            # Checkpoint at intervals (only on main process)
            if (
                checkpoint_interval > 0
                and (epoch + 1) % checkpoint_interval == 0
            ):
                if not distributed or is_main_process():
                    checkpoint_path = os.path.join(
                        checkpoint_dir,
                        f"checkpoint_epoch_{epoch+1}.pth",
                    )
                    save_checkpoint(
                        model_to_save,
                        optimizer,
                        epoch,
                        avg_val_loss,
                        checkpoint_path,
                        scheduler=epoch_scheduler,
                        early_stopping=early_stopper,
                    )

            # Save best model (only on main process)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if not distributed or is_main_process():
                    # Save with versioning metadata
                    save_model_checkpoint(
                        model_to_save,
                        save_path,
                        training_info={
                            'epoch': epoch + 1,
                            'samples_seen': train_size * (epoch + 1),
                            'val_loss': float(avg_val_loss),
                            'train_loss': float(avg_train_loss),
                        },
                    )
                    logger.info(
                        "  New best model saved (Val Loss: %.4f)",
                        avg_val_loss,
                    )

                    # Save timestamped checkpoint for history tracking
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    version_path = save_path.replace(
                        ".pth",
                        f"_{timestamp}.pth",
                    )
                    save_model_checkpoint(
                        model_to_save,
                        version_path,
                        training_info={
                            'epoch': epoch + 1,
                            'samples_seen': train_size * (epoch + 1),
                            'val_loss': float(avg_val_loss),
                            'train_loss': float(avg_train_loss),
                            'timestamp': timestamp,
                        },
                    )
                    logger.info(
                        "  Versioned checkpoint saved: %s",
                        version_path,
                    )
        else:
            # Final checkpoint at end of training (if not early stopped).
            # This else clause is for the for-loop and executes if no break
            # occurred.
            if not distributed or is_main_process():
                model_to_save_final = cast(
                    nn.Module,
                    model.module if distributed else model,
                )
                final_checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"checkpoint_final_epoch_{config.epochs_per_iter}.pth",
                )
                save_checkpoint(
                    model_to_save_final,
                    optimizer,
                    config.epochs_per_iter - 1,
                    avg_val_loss,
                    final_checkpoint_path,
                    scheduler=epoch_scheduler,
                    early_stopping=early_stopper,
                )
                logger.info("Training completed. Final checkpoint saved.")
    finally:
        # Clean up distributed process group
        if distributed:
            cleanup_distributed()


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments.
    
    Args:
        args: Optional list of argument strings. If None, uses sys.argv.
              Useful for testing.
    """
    parser = argparse.ArgumentParser(
        description='Train RingRift Neural Network AI'
    )
    
    # Data and model paths
    parser.add_argument(
        '--data-path', type=str, default=None,
        help='Path to training data (.npz file)'
    )
    parser.add_argument(
        '--save-path', type=str, default=None,
        help='Path to save best model weights'
    )
    
    # Training configuration
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Training batch size'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=None,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility'
    )
    
    # Early stopping
    parser.add_argument(
        '--early-stopping-patience', type=int, default=10,
        help='Early stopping patience (0 to disable)'
    )
    
    # Checkpointing
    parser.add_argument(
        '--checkpoint-dir', type=str, default='checkpoints',
        help='Directory for saving checkpoints'
    )
    parser.add_argument(
        '--checkpoint-interval', type=int, default=5,
        help='Save checkpoint every N epochs'
    )
    
    # Learning rate scheduling
    parser.add_argument(
        '--warmup-epochs', type=int, default=0,
        help='Number of warmup epochs (0 to disable)'
    )
    parser.add_argument(
        '--lr-scheduler', type=str, default='none',
        choices=['none', 'step', 'cosine', 'cosine-warm-restarts'],
        help='Learning rate scheduler type'
    )
    parser.add_argument(
        '--lr-min', type=float, default=1e-6,
        help='Minimum learning rate for cosine annealing (default: 1e-6)'
    )
    parser.add_argument(
        '--lr-t0', type=int, default=10,
        help='T_0 for CosineAnnealingWarmRestarts (initial restart period)'
    )
    parser.add_argument(
        '--lr-t-mult', type=int, default=2,
        help='T_mult for CosineAnnealingWarmRestarts (period multiplier)'
    )
    
    # Resume training
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume from'
    )
    
    # Board type
    parser.add_argument(
        '--board-type', type=str, default=None,
        choices=['square8', 'square19', 'hexagonal'],
        help='Board type for training'
    )
    
    # Hex symmetry augmentation
    parser.add_argument(
        '--augment-hex-symmetry', action='store_true',
        help='Enable D6 symmetry augmentation for hex boards (12x dataset)'
    )

    # Distributed training arguments
    parser.add_argument(
        '--distributed', action='store_true',
        help='Enable distributed training with PyTorch DDP'
    )
    parser.add_argument(
        '--local-rank', type=int, default=-1,
        help='Local rank for distributed training (set by torchrun)'
    )
    parser.add_argument(
        '--scale-lr', action='store_true',
        help='Scale learning rate based on world size'
    )
    parser.add_argument(
        '--lr-scale-mode', type=str, default='linear',
        choices=['linear', 'sqrt'],
        help='LR scaling mode: linear (lr * world_size) or sqrt'
    )
    parser.add_argument(
        '--find-unused-parameters', action='store_true',
        help='Enable find_unused_parameters in DDP (slower but handles '
             'unused params)'
    )

    return parser.parse_args(args)


def main():
    """Main entry point for training."""
    args = parse_args()
    
    # Create config
    config = TrainConfig()
    
    # Override config from CLI args
    if args.epochs is not None:
        config.epochs_per_iter = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.seed is not None:
        config.seed = args.seed
    if args.board_type is not None:
        board_type_map = {
            'square8': BoardType.SQUARE8,
            'square19': BoardType.SQUARE19,
            'hexagonal': BoardType.HEXAGONAL,
        }
        config.board_type = board_type_map[args.board_type]
    
    # Determine paths
    data_path = args.data_path or os.path.join(config.data_dir, "dataset.npz")
    save_path = args.save_path or os.path.join(
        config.model_dir,
        f"{config.model_id}.pth",
    )
    # Run training
    train_model(
        config=config,
        data_path=data_path,
        save_path=save_path,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        warmup_epochs=args.warmup_epochs,
        lr_scheduler=args.lr_scheduler,
        lr_min=args.lr_min,
        lr_t0=args.lr_t0,
        lr_t_mult=args.lr_t_mult,
        resume_path=args.resume,
        augment_hex_symmetry=args.augment_hex_symmetry,
        distributed=args.distributed,
        local_rank=args.local_rank,
        scale_lr=args.scale_lr,
        lr_scale_mode=args.lr_scale_mode,
        find_unused_parameters=args.find_unused_parameters,
    )


if __name__ == "__main__":
    main()