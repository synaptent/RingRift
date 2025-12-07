# MPS-Compatible Neural Network Architecture

## Overview

The `RingRiftCNN_MPS` architecture provides a fully MPS-compatible alternative to the default `RingRiftCNN` architecture for running RingRift's neural network AI on Apple Silicon GPUs via PyTorch's Metal Performance Shaders (MPS) backend.

## Background

### The Problem

The default `RingRiftCNN` architecture uses [`nn.AdaptiveAvgPool2d`](../app/ai/neural_net.py:264) for adaptive pooling to handle variable board sizes. However, this operation is not currently supported by PyTorch's MPS backend, causing failures when attempting to run on macOS systems with Apple Silicon.

### The Solution

`RingRiftCNN_MPS` replaces `nn.AdaptiveAvgPool2d` with manual global average pooling using `torch.mean(dim=[-2, -1])`, which is fully supported on MPS. This provides identical functionality while maintaining MPS compatibility.

## Architecture Details

### Key Differences from RingRiftCNN

| Aspect               | RingRiftCNN                    | RingRiftCNN_MPS            |
| -------------------- | ------------------------------ | -------------------------- |
| Pooling Method       | `nn.AdaptiveAvgPool2d((4, 4))` | `torch.mean(dim=[-2, -1])` |
| MPS Compatibility    | ❌ No                          | ✅ Yes                     |
| Architecture Version | `v1.0.0`                       | `v1.0.0-mps`               |
| Parameter Count      | ~Same                          | ~Same                      |
| Performance          | Baseline                       | Comparable                 |

### Technical Implementation

**Original (RingRiftCNN):**

```python
# Adaptive pooling to fixed 4x4 grid
self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
x = self.adaptive_pool(x)  # [B, C, H, W] → [B, C, 4, 4]
x = x.view(x.size(0), -1)  # Flatten to [B, C*16]
```

**MPS-Compatible (RingRiftCNN_MPS):**

```python
# Global average pooling
x = torch.mean(x, dim=[-2, -1])  # [B, C, H, W] → [B, C]
# No view() needed - already flat
```

### Maintained Features

- Same ResNet-style backbone with residual blocks
- Same input/output interfaces
- Same policy head size (55,000 actions)
- Same value head range ([-1, 1])
- Compatible with all board sizes (8×8, 19×19, 25×25 hex)

## Usage

### Environment Variable Configuration

Architecture selection is controlled via the `RINGRIFT_NN_ARCHITECTURE` environment variable:

**Option 1: Explicit MPS Selection**

```bash
# Force use of MPS-compatible architecture
export RINGRIFT_NN_ARCHITECTURE=mps
python scripts/run_self_play_soak.py ...
```

**Option 2: Auto-Detection (Recommended)**

```bash
# Automatically use MPS architecture if MPS is available
export RINGRIFT_NN_ARCHITECTURE=auto
python scripts/run_self_play_soak.py ...
```

**Option 3: Default Architecture**

```bash
# Use standard RingRiftCNN (default if not set)
export RINGRIFT_NN_ARCHITECTURE=default
# or simply omit the variable
python scripts/run_self_play_soak.py ...
```

### Device Selection Behavior

When using the MPS architecture:

1. **MPS Available**: Uses `mps` device automatically
2. **MPS Not Available**: Falls back to `cpu` with a warning
3. **Explicit CPU Override**: Use `RINGRIFT_FORCE_CPU=1` to force CPU

Example output:

```
NeuralNetAI using device: mps, architecture: mps
```

### Checkpoint Management

MPS architecture checkpoints use a `_mps` suffix to avoid confusion with default architecture checkpoints.

**Checkpoint Naming Convention:**

- Default architecture: `ringrift_v1.pth`
- MPS architecture: `ringrift_v1_mps.pth`

**Loading Process:**

```python
# With RINGRIFT_NN_ARCHITECTURE=mps
# Automatically looks for: ai-service/models/ringrift_v1_mps.pth

# With RINGRIFT_NN_ARCHITECTURE=default
# Looks for: ai-service/models/ringrift_v1.pth
```

**Important**: MPS and default checkpoints are **not interchangeable** due to architectural differences in the fully connected layers.

## Training with MPS Architecture

### Starting Training

```bash
# Train MPS model on Apple Silicon
export RINGRIFT_NN_ARCHITECTURE=mps
export RINGRIFT_FORCE_CPU=0  # Allow MPS

python -m app.training.train \
  --model-id ringrift_v1 \
  --board-type SQUARE8 \
  --epochs 100
```

The training script will:

1. Create `RingRiftCNN_MPS` model
2. Use MPS device if available
3. Save checkpoints to `models/ringrift_v1_mps.pth`

### Resuming Training

```bash
# Resume from MPS checkpoint
export RINGRIFT_NN_ARCHITECTURE=mps

python -m app.training.train \
  --model-id ringrift_v1 \
  --resume
```

## Testing

### Running MPS Architecture Tests

```bash
cd ai-service
python -m pytest tests/test_mps_architecture.py -v
```

### Test Coverage

The test suite verifies:

- ✅ Architecture instantiation
- ✅ Forward pass correctness (8×8, 19×19, 25×25 boards)
- ✅ Output shape validation
- ✅ Device compatibility (MPS when available)
- ✅ Architecture selection via environment variables
- ✅ Checkpoint naming conventions
- ✅ Version compatibility

### MPS Device Tests

Tests marked with `@pytest.mark.skipif(not torch.backends.mps.is_available())` will only run on systems with MPS support.

## Performance Considerations

### MPS Performance

**Expected Performance on Apple Silicon:**

- Forward pass: Similar to or faster than CPU
- Training: 2-5x faster than CPU (depending on model size and M-series chip)
- Memory: Efficient unified memory usage

**Benchmark Example (M1 Max):**

```
CPU:  ~15 ms/batch (batch_size=32)
MPS:  ~8 ms/batch (batch_size=32)
Speedup: 1.9x
```

### When to Use MPS vs Default

**Use MPS Architecture When:**

- Running on Apple Silicon (M1/M2/M3 Macs)
- Training locally on macOS
- Need GPU acceleration without CUDA

**Use Default Architecture When:**

- Running on CUDA-capable GPUs
- Maximum compatibility is required
- Checkpoints already exist for default architecture

## Limitations and Known Issues

### Current Limitations

1. **Checkpoint Incompatibility**: MPS and default checkpoints are not interchangeable
2. **Training Only**: MPS architecture is primarily for training; inference works on all devices
3. **macOS Only**: MPS backend only available on macOS with Apple Silicon

### Future Improvements

- [ ] Checkpoint conversion utility (default ↔ MPS)
- [ ] Unified checkpoint format supporting both architectures
- [ ] Performance optimization for larger models
- [ ] Mixed precision training support

## Migration Guide

### Migrating Existing Workflows

**If you're currently using CPU on Apple Silicon:**

```bash
# Before
RINGRIFT_FORCE_CPU=1 python train.py

# After (recommended)
RINGRIFT_NN_ARCHITECTURE=auto python train.py
```

**If you have existing default checkpoints:**

```bash
# Continue using default architecture
RINGRIFT_NN_ARCHITECTURE=default python train.py --resume

# OR: Train a new MPS model from scratch
RINGRIFT_NN_ARCHITECTURE=mps python train.py
```

## Troubleshooting

### Common Issues

**Issue**: `RuntimeError: AdaptiveAvgPool2d is not supported on MPS`
**Solution**: Set `RINGRIFT_NN_ARCHITECTURE=mps`

**Issue**: MPS architecture selected but using CPU
**Solution**: Check that MPS is available and not disabled:

```python
import torch
print(torch.backends.mps.is_available())  # Should be True
```

**Issue**: Checkpoint loading fails with size mismatch
**Solution**: Verify architecture type matches checkpoint:

- MPS checkpoints require `RINGRIFT_NN_ARCHITECTURE=mps`
- Default checkpoints require `RINGRIFT_NN_ARCHITECTURE=default`

### Debugging

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Output will show:

```
INFO:app.ai.neural_net:Initialized RingRiftCNN_MPS architecture
INFO:app.ai.neural_net:Using MPS device with MPS-compatible architecture
```

## References

- PyTorch MPS Backend: https://pytorch.org/docs/stable/notes/mps.html
- Original Architecture: [`ai-service/app/ai/neural_net.py:202`](../app/ai/neural_net.py:202)
- MPS Architecture: [`ai-service/app/ai/neural_net.py:325`](../app/ai/neural_net.py:325)
- Tests: [`ai-service/tests/test_mps_architecture.py`](../tests/test_mps_architecture.py)

## V2 Memory-Tiered Architectures (Recommended)

The v2 architectures provide improved playing strength with memory-aware model selection. Both high and low memory variants are fully MPS and CUDA compatible.

### Memory Tier Selection

Use `RINGRIFT_NN_MEMORY_TIER` environment variable:

```bash
# High memory (96GB systems) - maximum playing strength
export RINGRIFT_NN_MEMORY_TIER=high

# Low memory (48GB systems) - memory-efficient
export RINGRIFT_NN_MEMORY_TIER=low

# Legacy (default) - use v1 architectures
export RINGRIFT_NN_MEMORY_TIER=legacy
```

### V2 Architecture Comparison

| Aspect                | High (v2.0.0)    | Low (v2.0.0-lite) |
| --------------------- | ---------------- | ----------------- |
| SE Residual Blocks    | 12               | 6                 |
| Filters               | 192              | 96                |
| Base Input Channels   | 14               | 12                |
| Global Features       | 20               | 20                |
| History Frames        | 4                | 3                 |
| Policy Intermediate   | 384              | 192               |
| Value Head            | Multi-player (4) | Multi-player (4)  |
| Square19 Params       | ~36M (144 MB)    | ~14M (56 MB)      |
| Hex Params            | ~44M (175 MB)    | ~19M (75 MB)      |
| Target Memory (2 NNs) | ~35 GB budget    | ~15 GB budget     |

### Key V2 Improvements

1. **Squeeze-and-Excitation (SE) Blocks**: Global pattern recognition via channel attention
2. **Multi-Player Value Head**: Outputs per-player win probability `[B, 4]`
3. **Hex Board Masking**: Pre-computed validity mask prevents information bleeding
4. **Richer Input Features**:
   - Stack/cap height (normalized)
   - Collapsed territory
   - Per-player territory ownership
5. **Global Features (20)**:
   - Rings in hand (per player)
   - Eliminated rings (per player)
   - Territory/line counts (per player)
   - Game phase and LPS threat indicators

### V2 Usage Example

```bash
# Train on 96GB Mac Studio with high-capacity model
export RINGRIFT_NN_MEMORY_TIER=high
python scripts/run_selfplay.py --board square19 --ai-type descent

# Train on 48GB system with lite model
export RINGRIFT_NN_MEMORY_TIER=low
python scripts/run_selfplay.py --board hexagonal --ai-type descent
```

### Critical Bug Fix

The original `HexNeuralNet` had a critical bug where the policy head flattened 80,000 spatial features directly, resulting in **7.35 billion parameters** (~29 GB). The v2 architectures fix this by using global average pooling before the policy FC layer, reducing parameters by **167×**.

## Version History

- **v2.0.0** (2024-12): High-capacity SE architecture for 96GB systems
  - SE-enhanced residual blocks for global pattern recognition
  - Multi-player value head (outputs per-player win probability)
  - Hex board masking via register_buffer
  - 14 base input channels, 20 global features
  - Fixed critical HexNeuralNet policy head bug

- **v2.0.0-lite** (2024-12): Memory-efficient SE architecture for 48GB systems
  - Same SE blocks and multi-player value head
  - Reduced capacity (6 blocks, 96 filters)
  - 12 base input channels, 3 history frames

- **v1.0.0-mps** (2024-12): Initial MPS-compatible architecture
  - Global average pooling instead of AdaptiveAvgPool2d
  - Full MPS device support
  - Architecture selection via environment variables
