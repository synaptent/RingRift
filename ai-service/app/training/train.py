"""
Training script for RingRift Neural Network AI
Includes validation split and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import random
import os

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
                    self.length = len(self.data['values'])
                    print(
                        f"Loaded {self.length} samples from {data_path} (mmap)"
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
            self.length = 100
            # Generate dummy data in memory for testing
            self.data = {
                'features': np.random.rand(100, 40, 8, 8).astype(np.float32),
                'globals': np.random.rand(100, 10).astype(np.float32),
                'values': np.random.choice(
                    [1.0, 0.0, -1.0],
                    size=100,
                ).astype(np.float32),
                'policy_indices': np.array([
                    np.random.choice(55000, 5, replace=False).astype(np.int32)
                    for _ in range(100)
                ], dtype=object),
                'policy_values': np.array([
                    np.random.rand(5).astype(np.float32)
                    for _ in range(100)
                ], dtype=object),
            }
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

        # Access data from memory-mapped arrays. We copy to ensure we have a
        # writable tensor if needed, and to detach from the mmap backing
        # store.
        features = np.array(self.data['features'][idx])
        globals_vec = np.array(self.data['globals'][idx])
        value = np.array(self.data['values'][idx])

        # Policy is stored as object array of arrays (sparse). mmap does not
        # support object arrays directly, so these may be fully loaded into
        # memory depending on how the npz was written. For very large datasets
        # a CSR-style encoding would be preferable, but for now we assume the
        # object array fits in memory or is handled by OS paging.
        policy_indices = self.data['policy_indices'][idx]
        policy_values = self.data['policy_values'][idx]
            
        # Reconstruct dense policy vector on-the-fly
        policy_vector = torch.zeros(55000, dtype=torch.float32)
        
        if len(policy_indices) > 0:
            policy_vector[policy_indices] = torch.from_numpy(policy_values)
            
        return (
            torch.from_numpy(features),
            torch.from_numpy(globals_vec),
            torch.tensor([value.item()], dtype=torch.float32),
            policy_vector
        )


def train_model(
    config: TrainConfig,
    data_path: str,
    save_path: str
):
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

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
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

    best_val_loss = float('inf')

    for epoch in range(config.epochs_per_iter):
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
        scheduler.step(avg_val_loss)

        logger.info(
            f"Epoch [{epoch+1}/{config.epochs_per_iter}], "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        # Checkpoint
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


if __name__ == "__main__":
    config = TrainConfig()
    train_model(
        config=config,
        data_path=os.path.join(config.data_dir, "dataset.npz"),
        save_path=os.path.join(config.model_dir, "ringrift_v1.pth")
    )