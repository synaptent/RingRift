# AI Service Dependency Updates

> **Doc Status (2025-11-27): Active (AI service dependency audit, non-semantics)**
>
> - Role: records the dependency stack and compatibility decisions for the Python AI microservice (NumPy/PyTorch/ Gymnasium, etc.) and outlines an aspirational RL roadmap. It guides environment setup and ML stack evolution, not game semantics.
> - Not a semantics or lifecycle SSoT: for rules semantics and lifecycle / API contracts, defer to the shared TypeScript rules engine under `src/shared/engine/**`, the engine contracts under `src/shared/engine/contracts/**`, the v2 contract vectors in `tests/fixtures/contract-vectors/v2/**`, [`RULES_CANONICAL_SPEC.md`](../RULES_CANONICAL_SPEC.md), [`ringrift_complete_rules.md`](../ringrift_complete_rules.md), [`RULES_ENGINE_ARCHITECTURE.md`](../RULES_ENGINE_ARCHITECTURE.md), [`RULES_IMPLEMENTATION_MAPPING.md`](../RULES_IMPLEMENTATION_MAPPING.md), and [`docs/CANONICAL_ENGINE_API.md`](../docs/CANONICAL_ENGINE_API.md).
> - Related docs: service-level overview in [`ai-service/README.md`](./README.md), AI architecture narrative in [`AI_ARCHITECTURE.md`](../AI_ARCHITECTURE.md), training and dataset docs in [`docs/AI_TRAINING_AND_DATASETS.md`](../docs/AI_TRAINING_AND_DATASETS.md) and [`docs/AI_TRAINING_PREPARATION_GUIDE.md`](../docs/AI_TRAINING_PREPARATION_GUIDE.md), and security/supply-chain posture in [`docs/SUPPLY_CHAIN_AND_CI_SECURITY.md`](../docs/SUPPLY_CHAIN_AND_CI_SECURITY.md).

## Overview

Updated all dependencies to be compatible with **NumPy 2.2.1** and **Python 3.13**.

## Installation Status

✅ **All dependencies installed successfully** (2025-01-13)

## Key Versions

### Core ML Stack

- **NumPy**: 2.2.1 (latest)
- **SciPy**: 1.15.1 (numpy 2.x compatible)
- **Scikit-learn**: 1.6.1 (numpy 2.x compatible)

### Deep Learning

- **PyTorch**: 2.6.0 (numpy 2.x compatible)
- **TorchVision**: 0.21.0 (numpy 2.x compatible)
- **TensorBoard**: 2.18.0 (monitoring & visualization)
- **TensorBoardX**: 2.6.2.2 (extended features)

### Reinforcement Learning

- **Gymnasium**: 1.0.0 (modern RL environment interface, successor to OpenAI Gym)

### Data Processing

- **Pandas**: 2.2.3
- **Matplotlib**: 3.10.0
- **H5Py**: 3.13.0

## Packages Removed

### 1. stable-baselines3

**Reason**: Incompatible with NumPy 2.x

- Latest stable-baselines3 (v2.4.0) requires `numpy<2.0`
- This is a hard constraint that conflicts with other modern packages

**Alternative**: Custom RL implementation using PyTorch

- More control over algorithm specifics
- Can optimize for RingRift's unique game mechanics
- No dependency version constraints
- Better integration with our existing neural network code

### 2. numba

**Reason**: Not yet compatible with Python 3.13

- numba 0.61.0 doesn't support Python 3.13
- Expected to be added in future release

**Alternative**: PyTorch JIT compilation

- PyTorch provides `torch.jit.script()` and `torch.jit.trace()` for JIT compilation
- Native support without additional dependencies
- Can be added later when numba supports Python 3.13

## Future RL Implementation Plan

Since we removed stable-baselines3, we'll implement custom RL algorithms using PyTorch:

### Phase 1: Basic RL (Current)

- Random AI (difficulty 1-2) ✅
- Heuristic AI (difficulty 3-5) ✅

### Phase 2: Neural Network AI (Next)

- Deep Q-Network (DQN) for difficulty 6-7
- Minimax with neural network evaluation for difficulty 7-8
- Monte Carlo Tree Search (MCTS) for difficulty 8-10

### Phase 3: Advanced RL

- Policy Gradient methods (REINFORCE, A3C)
- Actor-Critic methods (A2C, PPO)
- Self-play training
- Experience replay with priority sampling

### Advantages of Custom Implementation

1. **RingRift-specific optimizations**: Can encode game knowledge directly
2. **Multi-board support**: Train different models for square8, square19, hexagonal
3. **Adaptive difficulty**: Fine-tune difficulty levels more granularly
4. **No version conflicts**: Full control over dependencies
5. **Better debugging**: Understand every component of the RL pipeline

## Testing

All imports verified:

```python
✅ fastapi
✅ torch (2.6.0)
✅ tensorboard
✅ gymnasium (1.0.0)
✅ numpy (2.2.1)
✅ scipy
✅ sklearn
✅ pandas
✅ matplotlib
```

## Docker Compatibility

All packages are available as pre-built wheels for:

- ✅ macOS ARM64 (Apple Silicon)
- ✅ Linux x86_64
- ✅ Linux ARM64

The Dockerfile uses Python 3.11-slim (not 3.13) for broader compatibility in production environments.

## Next Steps

1. ✅ Install dependencies
2. ✅ Verify imports
3. 🔄 Test AI service startup
4. ⏳ Implement neural network AI (Phase 2)
5. ⏳ Add self-play training pipeline
6. ⏳ Create model checkpointing and versioning
