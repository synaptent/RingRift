"""
Test script to verify training pipeline by overfitting a small batch
"""

import torch
import torch.nn as nn
import torch.optim as optim

from app.ai.neural_net import RingRiftCNN

def test_overfit():
    print("Testing overfitting capability...")

    # Create a small synthetic batch (e.g., 4 samples)
    batch_size = 4
    features = torch.randn(batch_size, 40, 8, 8)
    globals_vec = torch.randn(batch_size, 10)

    # Fixed targets
    value_targets = torch.tensor([[1.0], [-1.0], [0.5], [-0.5]])
    policy_targets = torch.randn(batch_size, 55000)
    policy_targets = torch.softmax(policy_targets, dim=1)  # Normalize

    # Model
    model = RingRiftCNN(board_size=8, in_channels=10, global_features=10)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    value_criterion = nn.MSELoss()
    # Use KLDivLoss for distribution matching
    policy_criterion = nn.KLDivLoss(reduction='batchmean')

    # Train loop
    for epoch in range(100):
        optimizer.zero_grad()

        value_pred, policy_pred = model(features, globals_vec)

        # Apply log_softmax for KLDivLoss
        policy_log_probs = torch.log_softmax(policy_pred, dim=1)

        value_loss = value_criterion(value_pred, value_targets)
        policy_loss = policy_criterion(policy_log_probs, policy_targets)
        loss = value_loss + policy_loss

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

    print("Final Loss:", loss.item())

    if loss.item() < 0.1:
        print("SUCCESS: Model successfully overfitted small batch.")
    else:
        print("FAILURE: Model failed to overfit small batch.")

if __name__ == "__main__":
    test_overfit()
