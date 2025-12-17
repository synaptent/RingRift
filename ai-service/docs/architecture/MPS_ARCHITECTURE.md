# MPS Support for Neural Nets (Apple Silicon)

## Overview

RingRift supports running neural-network inference/training on Apple Silicon GPUs
via PyTorch's Metal Performance Shaders (MPS) backend.

## Background / Current State

Historically, RingRift had an MPS-specific architecture variant to avoid
`nn.AdaptiveAvgPool2d` limitations on MPS.

Today, the canonical CNN architectures (`RingRiftCNN_v2`, `RingRiftCNN_v3` and
their Lite variants) use MPS-compatible global pooling (`torch.mean(...)`), so
the same checkpoint can run on CPU, CUDA, or MPS.

## Model IDs vs Architectures (v2/v3/v4)

There is no `RingRiftCNN_v4` class. “v4” is a **model ID / checkpoint lineage**
used in filenames (e.g. `ringrift_v4_sq8_2p.pth`). The checkpoint metadata
declares the actual model class (e.g. `RingRiftCNN_v2`) and its hyperparameters
(filters, blocks, policy size, history length).

Example:

- `ringrift_v4_sq8_2p.pth` currently loads as `RingRiftCNN_v2` (192 filters),
  with `value_fc1.in_features = 212 = 192 + 20`.
- `ringrift_v5_sq8_2p_2xh100.pth` currently loads as `RingRiftCNN_v3`
  (architecture version `v3.1.0`, spatial policy heads). The “v5” prefix is
  similarly a checkpoint lineage, not a Python class name.

### Troubleshooting: `value_fc1 in_features` mismatch (212 vs 148)

If you see an error like:

```
value_fc1 in_features: checkpoint=212, expected=148
```

That mismatch is **not** about a “v4 architecture”. It means the checkpoint was
trained with a different `num_filters` than the runtime model was constructed
with:

- `212 = 192 + 20` → 192 filters, 20 global features (current v2/v3 default).
- `148 = 128 + 20` → 128 filters, 20 global features (older/legacy default).

**Canonical fix:** use the modern loader that infers architecture hyperparameters
from checkpoint metadata/weights (see `app/ai/neural_net.py`), and use a
board-appropriate `nn_model_id`:

- Square8 2p v2 family: `ringrift_v4_sq8_2p`
- Square8 2p v3 family: `ringrift_v5_sq8_2p_2xh100`

**Update (2025-12-13):** `NeuralNetAI` now attempts to _self-heal_ this mismatch
by rebuilding its runtime model to match the checkpoint’s shapes (filters,
residual blocks, history length, policy size, and num_players) before loading.
This converts “212 vs 148” failures into a one-time warning and avoids silent
neural-tier fallback.

**Debug commands:**

```bash
PYTHONPATH=ai-service python ai-service/scripts/inspect_nn_checkpoint.py --nn-model-id ringrift_v4_sq8_2p --board-type square8
PYTHONPATH=ai-service python ai-service/scripts/inspect_nn_checkpoint.py --nn-model-id ringrift_v5_sq8_2p_2xh100 --board-type square8
```

**Fail-fast mode (recommended for tournaments):** set `RINGRIFT_REQUIRE_NEURAL_NET=1`
or pass `--require-neural-net` to the tournament scripts so MCTS/Descent tiers
error instead of silently falling back to heuristic rollouts.

### Troubleshooting: `policy_size` / policy-layout mismatch (7000 vs legacy MAX_N)

RingRift currently supports **two** policy layouts for square boards:

- **Board-specific policy heads** (preferred for training + compact models):
  - Square8: `policy_size=7000`
  - Square19: `policy_size=67000`
- **Legacy MAX_N policy head** (single fixed layout for square boards):
  - Historically used by some `ringrift_v4_*` checkpoints (e.g. `policy_size=54875`)

If you load a Square8 checkpoint with `policy_size=7000` but your runtime
encoder still emits indices from the legacy MAX_N layout, the neural policy
will be effectively ignored (most move indices will be out-of-range), and MCTS
will degrade toward heuristic rollouts/uniform priors.

**Canonical fix:** `NeuralNetAI.encode_move` now auto-selects the encoder based
on the loaded checkpoint’s `model.policy_size`:

- If `model.policy_size == get_policy_size_for_board(board.type)`, it uses
  `encode_move_for_board` (board-specific).
- Otherwise it falls back to the legacy MAX_N encoding for compatibility with
  older checkpoints.

**Debug:** inspect the checkpoint’s declared policy size:

```bash
PYTHONPATH=ai-service python ai-service/scripts/inspect_nn_checkpoint.py --nn-model-id sq8_2p_nn_baseline --board-type square8
```

## Usage

### Environment Variable Configuration

Architecture/device selection is controlled via the `RINGRIFT_NN_ARCHITECTURE`
environment variable (see `app/ai/neural_net.py`):

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

NeuralNetAI resolves checkpoints under `ai-service/models/` using:

- `AIConfig.nn_model_id` (explicit), or
- a board-aware default (square8 → prefer `ringrift_v5_sq8_2p_2xh100` when present, else `ringrift_v4_sq8_2p`).

When `RINGRIFT_NN_ARCHITECTURE=mps`, the loader will _prefer_ a `*_mps.pth`
checkpoint but will fall back to the non-suffixed `.pth` if needed. Since v2/v3
architectures are MPS compatible, checkpoints are portable across devices; the
`*_mps` suffix is kept primarily for legacy naming conventions.

Important: We intentionally do **not** default to deprecated `ringrift_v1` /
`ringrift_v1_mps` IDs anymore. Missing or incompatible v1 checkpoints are a
common cause of “neural fallback” in MCTS/Descent.

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

## Troubleshooting

- If you see an error like:
  `value_fc1 in_features: checkpoint=212 expected=148`
  it usually means the code instantiated a smaller legacy model (128 filters)
  and tried to load a canonical 192-filter checkpoint. Ensure you are running
  the current `app/ai/neural_net.py` loader and that your scripts do not force
  legacy defaults / deprecated model IDs.

- If you see errors like:
  - `value_fc2 out_features: checkpoint=2 expected=4` or
  - `rank_dist_fc2 out_features: checkpoint=4 expected=16`
    it usually means you are trying to load a checkpoint trained for a different
    `num_players` configuration (common with V3 checkpoints). The loader now
    infers `num_players` from `value_fc2.weight.shape[0]` during initialization;
    make sure your runtime has the updated loader and that `nn_model_id` resolves
    to the intended checkpoint file.

- Use `scripts/inspect_nn_checkpoint.py` to debug what a given checkpoint will
  load as (model class, architecture version, inferred filters, inferred
  `num_players`, and value-head shapes).

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
