"""
Training script for RingRift Neural Network AI
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from ..ai.neural_net import RingRiftCNN

class RingRiftDataset(Dataset):
    def __init__(self, data_path):
        # Load training data (states and outcomes)
        # Format: [features, outcome]
        # outcome: 1 (win), 0 (draw), -1 (loss)
        if os.path.exists(data_path):
            self.data = np.load(data_path, allow_pickle=True)
        else:
            # Generate dummy data for testing
            print(f"Data file {data_path} not found, generating dummy data")
            self.data = []
            for _ in range(1000):
                # 4 channels, 8x8 board
                features = np.random.rand(4, 8, 8).astype(np.float32)
                outcome = np.random.choice([1.0, 0.0, -1.0])
                self.data.append((features, outcome))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, outcome = self.data[idx]
        return torch.FloatTensor(features), torch.FloatTensor([outcome])

def train_model(data_path="training_data.npy", save_path="ringrift_model.pth", epochs=10):
    # Hyperparameters
    learning_rate = 0.001
    batch_size = 32

    # Initialize model, loss, optimizer
    model = RingRiftCNN(board_size=8, in_channels=4)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load data
    dataset = RingRiftDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        total_loss = 0
        for i, (inputs, targets) in enumerate(dataloader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs(os.path.dirname("ai-service/app/models/"), exist_ok=True)
    train_model(save_path="ai-service/app/models/ringrift_v1.pth")