"""
Training script for RingRift Neural Network AI
Includes validation split and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import sys

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from app.ai.neural_net import RingRiftCNN  # noqa: E402


class RingRiftDataset(Dataset):
    def __init__(self, data_path):
        if os.path.exists(data_path):
            # Use mmap_mode='r' for large files to avoid loading everything into RAM
            # However, allow_pickle=True is required for object arrays, which doesn't work well with mmap
            # Ideally, we should save data as separate arrays (features, globals, outcomes, policies)
            # For now, we'll stick to loading but warn about scalability.
            # TODO: Refactor generate_data.py to save structured arrays (HDF5 or separate .npy files)
            try:
                self.data = np.load(data_path, allow_pickle=True)
            except Exception as e:
                print(f"Error loading data: {e}")
                self.data = []
        else:
            print(f"Data file {data_path} not found, generating dummy data")
            self.data = []
            for _ in range(100):
                features = np.random.rand(40, 8, 8).astype(np.float32)
                globals_vec = np.random.rand(10).astype(np.float32)
                outcome = np.random.choice([1.0, 0.0, -1.0])
                policy = np.random.rand(55000).astype(np.float32)
                policy /= policy.sum()
                self.data.append(((features, globals_vec), outcome, policy))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Handle potential unpacking issues if data structure varies
        if len(item) == 3:
            (features, globals_vec), outcome, policy_target = item
        elif len(item) == 2:
            # Fallback for older data format or error
            if isinstance(item[0], tuple) and len(item[0]) == 2:
                (features, globals_vec), outcome = item
                # Dummy policy
                policy_target = np.zeros(55000, dtype=np.float32)
            else:
                raise ValueError(f"Unknown data format at index {idx}: {item}")
        else:
            raise ValueError(f"Unknown data format at index {idx}: {item}")

        # Ensure features have correct shape (40, 8, 8)
        if features.shape[0] == 10:
            features = np.concatenate([features] * 4, axis=0)
             
        return (
            torch.FloatTensor(features),
            torch.FloatTensor(globals_vec),
            torch.FloatTensor([outcome]),
            torch.FloatTensor(policy_target)
        )


def train_model(
    data_path="ai-service/app/training/data/self_play_data.npy",
    save_path="ai-service/app/models/ringrift_v1.pth",
    epochs=10
):
    # Hyperparameters
    learning_rate = 0.001
    batch_size = 32
    weight_decay = 1e-4

    # Initialize model
    model = RingRiftCNN(board_size=8, in_channels=10, global_features=10)
    
    # Load existing weights if available to continue training
    if os.path.exists(save_path):
        try:
            model.load_state_dict(torch.load(save_path, weights_only=True))
            print(f"Loaded existing model weights from {save_path}")
        except Exception as e:
            print(f"Could not load existing weights: {e}. Starting fresh.")
    
    # Loss functions
    value_criterion = nn.MSELoss()
    # Use KLDivLoss for policy distribution matching
    # Input should be log-probabilities (log_softmax), target should be probabilities
    policy_criterion = nn.KLDivLoss(reduction='batchmean')
    
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Load data
    full_dataset = RingRiftDataset(data_path)

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
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0

        for i, (features, globals_vec, value_targets, policy_targets) in \
                enumerate(train_loader):
            value_pred, policy_pred = model(features, globals_vec)

            # Apply log_softmax to policy prediction for KLDivLoss
            policy_log_probs = torch.log_softmax(policy_pred, dim=1)

            value_loss = value_criterion(value_pred, value_targets)
            policy_loss = policy_criterion(policy_log_probs, policy_targets)
            loss = value_loss + policy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, globals_vec, value_targets, policy_targets in \
                    val_loader:
                value_pred, policy_pred = model(features, globals_vec)
                
                policy_log_probs = torch.log_softmax(policy_pred, dim=1)
                
                value_loss = value_criterion(value_pred, value_targets)
                policy_loss = policy_criterion(policy_log_probs, policy_targets)
                loss = value_loss + policy_loss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch [{epoch+1}/{epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  New best model saved (Val Loss: {avg_val_loss:.4f})")
            
            # Versioning: Save timestamped checkpoint
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_path = save_path.replace(".pth", f"_{timestamp}.pth")
            torch.save(model.state_dict(), version_path)
            print(f"  Versioned checkpoint saved: {version_path}")

if __name__ == "__main__":
    os.makedirs(os.path.dirname("ai-service/app/models/"), exist_ok=True)
    train_model(save_path="ai-service/app/models/ringrift_v1.pth")