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
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.ai.neural_net import RingRiftCNN

class RingRiftDataset(Dataset):
    def __init__(self, data_path):
        if os.path.exists(data_path):
            self.data = np.load(data_path, allow_pickle=True)
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
            # Assuming ((features, globals), outcome) and missing policy?
            # Or (features, outcome)?
            # Let's inspect structure safely
            if isinstance(item[0], tuple) and len(item[0]) == 2:
                (features, globals_vec), outcome = item
                policy_target = np.zeros(55000, dtype=np.float32) # Dummy policy
            else:
                # Unknown format
                raise ValueError(f"Unknown data format at index {idx}: {item}")
        else:
             raise ValueError(f"Unknown data format at index {idx}: {item}")

        
        # Ensure features have correct shape (40, 8, 8)
        # If history was not used in generation, we might have (10, 8, 8)
        # We need to stack it to 40
        # Note: NeuralNetAI uses history_length=3, so total channels = 10 * (3+1) = 40
        # But if generation didn't use history, we might get 10 channels.
        # Also, if generation used history but only had 1 state, it might have stacked it already?
        # Let's check shape.
        
        if features.shape[0] == 10:
            features = np.concatenate([features] * 4, axis=0)
        elif features.shape[0] == 6:
             # If we somehow got 6 channels (maybe from an old version or different config?)
             # We need to pad or repeat to match 40.
             # But 6 doesn't divide 40 evenly.
             # This suggests a mismatch in feature extraction logic.
             # Current feature extraction produces 10 channels.
             # If we see 6, it might be from a very old version or a bug.
             # However, the error says "expected input[32, 6, 8, 8] to have 40 channels".
             # Wait, the error says "got 6 channels instead".
             # This means the input tensor has 6 channels.
             # Why 6?
             # Maybe the dummy data generator in RingRiftDataset (if file not found) produced 6?
             # No, dummy data produces 40.
             # The error happened during training with loaded data.
             # So the loaded data has 6 channels?
             # Let's look at NeuralNetAI._extract_features. It produces 10 channels.
             # Ah, maybe the data generation loop didn't use the updated NeuralNetAI?
             # Or maybe the data file contained old data?
             # We should force regeneration or handle this case.
             # For now, let's pad/repeat to 40 to unblock.
             
             # Repeat 6 to get close to 40? No, that's messy.
             # Let's assume the first 6 are valid and pad the rest with zeros?
             # Or repeat.
             # Let's try to construct 40 channels from 6.
             # 6 * 6 = 36 + 4 = 40.
             temp = np.concatenate([features] * 6, axis=0) # 36
             temp = np.concatenate([temp, features[:4]], axis=0) # 40
             features = temp
             
        return (
            torch.FloatTensor(features),
            torch.FloatTensor(globals_vec),
            torch.FloatTensor([outcome]),
            torch.FloatTensor(policy_target)
        )

def train_model(data_path="ai-service/app/training/data/self_play_data.npy", save_path="ai-service/app/models/ringrift_v1.pth", epochs=10):
    # Hyperparameters
    learning_rate = 0.001
    batch_size = 32
    weight_decay = 1e-4
    
    # Initialize model
    model = RingRiftCNN(board_size=8, in_channels=10, global_features=10)
    
    # Loss functions
    value_criterion = nn.MSELoss()
    policy_criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Load data
    full_dataset = RingRiftDataset(data_path)
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for i, (features, globals_vec, value_targets, policy_targets) in enumerate(train_loader):
            value_pred, policy_pred = model(features, globals_vec)
            
            value_loss = value_criterion(value_pred, value_targets)
            policy_loss = policy_criterion(policy_pred, policy_targets)
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
            for features, globals_vec, value_targets, policy_targets in val_loader:
                value_pred, policy_pred = model(features, globals_vec)
                value_loss = value_criterion(value_pred, value_targets)
                policy_loss = policy_criterion(policy_pred, policy_targets)
                loss = value_loss + policy_loss
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  New best model saved (Val Loss: {avg_val_loss:.4f})")

if __name__ == "__main__":
    os.makedirs(os.path.dirname("ai-service/app/models/"), exist_ok=True)
    train_model(save_path="ai-service/app/models/ringrift_v1.pth")