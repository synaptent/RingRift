"""
Training script for RingRift Neural Network AI
Includes validation split, checkpointing, early stopping, and LR warmup
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
from typing import Optional, Tuple, Dict, Any, List

import logging
from app.ai.neural_net import RingRiftCNN  # noqa: E402
from app.training.config import TrainConfig  # noqa: E402
from app.models import BoardType  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def seed_all(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save state from
        epoch: Current epoch number
        loss: Current loss value
        path: Path to save checkpoint to
        scheduler: Optional LR scheduler to save state from
        early_stopping: Optional early stopping tracker to save state from
    """
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
    
    # Ensure directory exists
    dir_path = os.path.dirname(path) if os.path.dirname(path) else '.'
    os.makedirs(dir_path, exist_ok=True)
    torch.save(checkpoint, path)
    logger.info(
        f"Saved checkpoint to {path} (epoch {epoch}, loss {loss:.4f})"
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    early_stopping: Optional[EarlyStopping] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, float]:
    """
    Load a training checkpoint.
    
    Args:
        path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional LR scheduler to load state into
        early_stopping: Optional early stopping tracker to restore state into
        device: Device to map checkpoint tensors to
        
    Returns:
        Tuple of (epoch, loss) from the checkpoint
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if early_stopping is not None and 'early_stopping' in checkpoint:
        es_state = checkpoint['early_stopping']
        early_stopping.best_loss = es_state['best_loss']
        early_stopping.counter = es_state['counter']
        early_stopping.best_state = es_state['best_state']
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
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
    """

    def __init__(self, data_path):
        self.data_path = data_path
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
    data_path: str,
    save_path: str,
    early_stopping_patience: int = 10,
    checkpoint_dir: str = 'checkpoints',
    checkpoint_interval: int = 5,
    warmup_epochs: int = 0,
    lr_scheduler: str = 'none',
    resume_path: Optional[str] = None,
):
    """
    Train the RingRift neural network model.
    
    Args:
        config: Training configuration
        data_path: Path to the training data (.npz file)
        save_path: Path to save the best model weights
        early_stopping_patience: Number of epochs without improvement before
            stopping (0 to disable early stopping)
        checkpoint_dir: Directory for saving periodic checkpoints
        checkpoint_interval: Save checkpoint every N epochs
        warmup_epochs: Number of epochs for LR warmup (0 to disable)
        lr_scheduler: Type of LR scheduler ('none', 'step', 'cosine')
        resume_path: Path to checkpoint to resume training from
    """
    seed_all(config.seed)

    # Device configuration
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
        board_size = 21
    else:
        # Default to 8×8.
        board_size = 8

    logger.info(f"Initializing RingRiftCNN with board_size={board_size}")

    # Initialize model
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
            logger.info(f"Loaded existing model weights from {save_path}")
        except Exception as e:
            logger.warning(
                f"Could not load existing weights: {e}. Starting fresh."
            )

    # Loss functions
    value_criterion = nn.MSELoss()
    policy_criterion = nn.KLDivLoss(reduction='batchmean')

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler with optional warmup
    warmup_scheduler = get_warmup_scheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=config.epochs_per_iter,
        scheduler_type=lr_scheduler,
    )
    
    # ReduceLROnPlateau as fallback if no warmup scheduler
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    ) if warmup_scheduler is None else None
    
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
        start_epoch, _ = load_checkpoint(
            resume_path,
            model,
            optimizer,
            scheduler=warmup_scheduler,
            early_stopping=early_stopper,
            device=device,
        )
        start_epoch += 1  # Start from next epoch
        logger.info(f"Resuming training from epoch {start_epoch}")
    
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Mixed precision scaler
    # Note: GradScaler is primarily for CUDA.
    # For MPS, mixed precision support is evolving.
    # We'll enable it only for CUDA for now to be safe.
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # Load data
    full_dataset = RingRiftDataset(data_path)

    if len(full_dataset) == 0:
        logger.warning(
            "Training dataset at %s is empty; skipping training step.",
            data_path,
        )
        return

    # For multi-board training, each .npz file is expected to contain positions
    # from a single board type/size so that every mini-batch drawn by the
    # DataLoader is same-board. Higher-level tooling can still group different
    # per-board datasets to implement cross-board training schedules without
    # violating the same-board-per-batch invariant enforced by
    # NeuralNetAI.evaluate_batch.
    shape = getattr(full_dataset, "spatial_shape", None)
    if shape is not None:
        h, w = shape
        logger.info(
            "Dataset spatial feature shape inferred as %dx%d.",
            h,
            w,
        )

    # Experience Replay: Sample a subset if dataset is too large
    # This ensures we don't overfit to the oldest data if the buffer is huge,
    # but for now we train on everything in the buffer (up to 50k).
    # If buffer grows larger, we might want to sample recent + random old.

    # Split into train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )

    logger.info(f"Starting training for {config.epochs_per_iter} epochs...")
    logger.info(f"Train size: {train_size}, Val size: {val_size}")
    if early_stopper is not None:
        logger.info(
            f"Early stopping enabled with patience: {early_stopping_patience}"
        )
    if warmup_epochs > 0:
        logger.info(f"LR warmup enabled for {warmup_epochs} epochs")
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    best_val_loss = float('inf')
    avg_val_loss = float('inf')  # Initialize for final checkpoint

    for epoch in range(start_epoch, config.epochs_per_iter):
        # Training
        model.train()
        train_loss = 0

        for i, (features, globals_vec, value_targets, policy_targets) in \
                enumerate(train_loader):
            
            features = features.to(device)
            globals_vec = globals_vec.to(device)
            value_targets = value_targets.to(device)
            policy_targets = policy_targets.to(device)

            optimizer.zero_grad()

            # Autocast for mixed precision (CUDA only usually)
            # For MPS, we might need to check torch.autocast(device_type='mps')
            # but it's safer to stick to float32 on MPS if unsure.
            use_amp = (device.type == 'cuda')
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                value_pred, policy_pred = model(features, globals_vec)

                # Apply log_softmax to policy prediction for KLDivLoss
                policy_log_probs = torch.log_softmax(policy_pred, dim=1)

                value_loss = value_criterion(value_pred, value_targets)
                policy_loss = policy_criterion(
                    policy_log_probs, policy_targets
                )
                loss = value_loss + (config.policy_weight * policy_loss)

            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            
            if i % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}, Batch {i}: "
                    f"Loss={loss.item():.4f} "
                    f"(Val={value_loss.item():.4f}, "
                    f"Pol={policy_loss.item():.4f})"
                )

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, globals_vec, value_targets, policy_targets in \
                    val_loader:
                
                features = features.to(device)
                globals_vec = globals_vec.to(device)
                value_targets = value_targets.to(device)
                policy_targets = policy_targets.to(device)

                value_pred, policy_pred = model(features, globals_vec)
                
                policy_log_probs = torch.log_softmax(policy_pred, dim=1)
                
                value_loss = value_criterion(value_pred, value_targets)
                policy_loss = policy_criterion(
                    policy_log_probs, policy_targets
                )
                loss = value_loss + (config.policy_weight * policy_loss)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        # Update scheduler
        if warmup_scheduler is not None:
            warmup_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"  Current LR: {current_lr:.6f}")
        elif plateau_scheduler is not None:
            plateau_scheduler.step(avg_val_loss)

        logger.info(
            f"Epoch [{epoch+1}/{config.epochs_per_iter}], "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        # Check early stopping
        if early_stopper is not None:
            should_stop = early_stopper(avg_val_loss, model)
            if should_stop:
                logger.info(
                    f"Early stopping triggered at epoch {epoch+1} "
                    f"(best loss: {early_stopper.best_loss:.4f})"
                )
                # Restore best weights
                early_stopper.restore_best_weights(model)
                # Save final checkpoint with best weights
                final_checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"checkpoint_early_stop_epoch_{epoch+1}.pth"
                )
                save_checkpoint(
                    model, optimizer, epoch, early_stopper.best_loss,
                    final_checkpoint_path,
                    scheduler=warmup_scheduler,
                    early_stopping=early_stopper,
                )
                # Save best model
                torch.save(model.state_dict(), save_path)
                logger.info(f"Best model saved to {save_path}")
                break

        # Checkpoint at intervals
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_epoch_{epoch+1}.pth"
            )
            save_checkpoint(
                model, optimizer, epoch, avg_val_loss, checkpoint_path,
                scheduler=warmup_scheduler,
                early_stopping=early_stopper,
            )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(
                f"  New best model saved (Val Loss: {avg_val_loss:.4f})"
            )

            # Versioning: Save timestamped checkpoint
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_path = save_path.replace(".pth", f"_{timestamp}.pth")
            torch.save(model.state_dict(), version_path)
            logger.info(f"  Versioned checkpoint saved: {version_path}")
    
    # Final checkpoint at end of training (if not early stopped)
    else:
        final_checkpoint_path = os.path.join(
            checkpoint_dir,
            f"checkpoint_final_epoch_{config.epochs_per_iter}.pth"
        )
        save_checkpoint(
            model, optimizer, config.epochs_per_iter - 1, avg_val_loss,
            final_checkpoint_path,
            scheduler=warmup_scheduler,
            early_stopping=early_stopper,
        )
        logger.info("Training completed. Final checkpoint saved.")


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
        choices=['none', 'step', 'cosine'],
        help='Learning rate scheduler type'
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
        config.model_dir, f"{config.model_id}.pth"
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
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()