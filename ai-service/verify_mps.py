#!/usr/bin/env python3
"""Verify MPS (Metal Performance Shaders) availability for training."""

import sys

import torch

print("=" * 60)
print("MPS Environment Verification")
print("=" * 60)

print(f"\nPyTorch version: {torch.__version__}")
print(f"Python version: {sys.version}")

# Check MPS availability
mps_available = torch.backends.mps.is_available()
mps_built = torch.backends.mps.is_built()

print(f"\nMPS available: {mps_available}")
print(f"MPS built: {mps_built}")

# Try to create a tensor on MPS if available
if mps_available:
    try:
        device = torch.device("mps")
        test_tensor = torch.randn(10, 10).to(device)
        print("\n✅ Successfully created tensor on MPS device")
        print(f"   Device: {test_tensor.device}")
        print(f"   Shape: {test_tensor.shape}")
    except Exception as e:
        print(f"\n❌ Failed to create tensor on MPS: {e}")
        sys.exit(1)
else:
    print("\n⚠️  MPS not available - will fall back to CPU")

# Check available devices
print(f"\nCUDA available: {torch.cuda.is_available()}")
print(f"Number of CPU threads: {torch.get_num_threads()}")

print("\n" + "=" * 60)
if mps_available:
    print("✅ Environment ready for MPS-accelerated training")
else:
    print("⚠️  Environment will use CPU for training")
print("=" * 60)
