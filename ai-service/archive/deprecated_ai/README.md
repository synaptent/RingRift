# Deprecated AI Modules

This directory contains AI modules that have been deprecated in favor of GNN-based alternatives.

## Archived Modules (December 2025)

### Legacy Game Engine (`_game_engine_legacy.py`)

**File:** `_game_engine_legacy.py` (~192KB, ~5,000 LOC)

**Deprecated**: December 2025

**What it was:**
The original monolithic game engine implementation containing all game rules, move generation, state management, and scoring logic in a single file.

**Why deprecated:**
Superseded by the modular `app/rules/` module which provides:

- Single source of truth (SSoT) move generators
- Better separation of concerns
- Improved testability and maintainability
- Cleaner architecture with dedicated modules for each game aspect

**Replacement**: Use `app/rules/game_engine.py` instead.

```python
# Old (deprecated):
from app._game_engine_legacy import GameEngine

# New (recommended):
from app.rules.game_engine import GameEngine
```

**Migration Notes**:

- The legacy file was used by `app/rules/legacy/replay_compatibility.py` for backward compatibility with old replay formats
- The `app/game_engine/__init__.py` shim provided a migration path with deprecation warnings
- After archival, remaining imports will need to be updated or will fail

---

### Legacy Neural Network (`_neural_net_legacy.py`)

**File:** `_neural_net_legacy.py` (7,080 LOC)

**What it was:**
The original monolithic neural network implementation containing:

- All board encoders (square, hex, hexagonal)
- All model architectures (v1-v5, hex variants)
- Move encoding/decoding functions
- Model caching and loading utilities
- Training utilities (data augmentation, etc.)

**Why deprecated:**
Superseded by the modular `app/ai/neural_net/` package which provides:

- Cleaner separation of concerns (encoding, architecture, utilities)
- Better testability with smaller, focused modules
- Easier maintenance and extension
- Type-safe interfaces

**Migration Path:**

```python
# Old (deprecated)
from app.ai._neural_net_legacy import NeuralNetAI, encode_move_for_board

# New (recommended)
from app.ai.neural_net import NeuralNetAI, encode_move_for_board
```

**Note:** The symlink at `app/ai/_neural_net_legacy.py` points to this archived file for backward compatibility. Imports from `app.ai._neural_net_legacy` will continue to work but emit deprecation warnings.

---

### Energy-Based Move Optimization (EBMO)

**Files:**

- `ebmo_ai.py` - EBMO AI agent implementation
- `ebmo_network.py` - EBMO neural network architecture

**What it was:**
EBMO used gradient descent on continuous action embeddings at inference time to find optimal moves. Instead of discrete policy/value networks, it encoded moves as learnable embeddings and optimized them via energy minimization.

**Why deprecated:**
GNN-based approaches (GNNPolicyNet, HybridPolicyNet) show better results with:

- 18x smaller model size
- 4x faster training
- Native hex geometry understanding via message passing
- Better territory connectivity modeling

### Gradient Move Optimization (GMO)

**Files:**

- `gmo_ai.py` - Original GMO implementation with entropy-guided gradient ascent
- `gmo_v2.py` - Enhanced GMO with attention and ensemble optimization
- `ig_gmo.py` - Information-Gain GMO with mutual information exploration

**What it was:**
GMO used entropy-guided gradient ascent in move embedding space to explore the action manifold. It combined:

- Gradient-based optimization in continuous move space
- Entropy regularization for exploration
- Information-theoretic move selection

**Why deprecated:**
While innovative, GMO showed limitations compared to GNN approaches:

- Complex gradient dynamics during inference
- Sensitivity to hyperparameters (learning rate, entropy weight)
- GNN's message passing provides cleaner territory reasoning
- Hybrid CNN-GNN combines best of pattern recognition + connectivity

## Migration Path

### For EBMO Users

Replace:

```python
from app.ai.ebmo_ai import EBMO_AI

ai = EBMO_AI(player_number=1, config=config)
```

With GNN-based alternative:

```python
from app.ai.gnn_ai import GNNAI

ai = GNNAI(player_number=1, config=config)
```

Or for hybrid CNN+GNN:

```python
from app.ai.hybrid_ai import HybridAI

ai = HybridAI(player_number=1, config=config)
```

### For GMO Users

Replace:

```python
from app.ai.gmo_ai import GMOAI

ai = GMOAI(player_number=1, config=config)
```

With GNN-based alternative:

```python
from app.ai.gnn_ai import GNNAI

ai = GNNAI(player_number=1, config=config, model_path="models/gnn_hex8_2p/gnn_policy_best.pt")
```

For hybrid approach with faster inference:

```python
from app.ai.hybrid_ai import HybridAI

ai = HybridAI(player_number=1, config=config)
```

### Factory Updates

The factory has been updated to import from archive locations with deprecation warnings:

```python
# Still works, but emits deprecation warning
from app.ai.factory import AIFactory

ai = AIFactory.create_from_difficulty(difficulty=12)  # EBMO
ai = AIFactory.create_from_difficulty(difficulty=13)  # GMO
```

Recommended alternatives:

```python
ai = AIFactory.create_from_difficulty(difficulty=22)  # GNN
ai = AIFactory.create_from_difficulty(difficulty=23)  # GNN Strong
ai = AIFactory.create_from_difficulty(difficulty=24)  # Hybrid CNN-GNN
```

## Removal Timeline

**Q2 2026 (June 2026):** Full removal of EBMO and GMO modules

- Archive will be removed from main codebase
- Factory will no longer support EBMO/GMO/IG-GMO difficulty levels
- Breaking change for any code still using these modules

**Before removal:**

- All selfplay databases will migrate to GNN-based engines
- Tournament configurations will update to GNN alternatives
- Any production code using D12-D14 will need updates

## Benchmark Comparison

Performance comparison showing why GNN approaches superseded EBMO/GMO:

| Metric                  | EBMO  | GMO v2 | GNN       | Hybrid CNN-GNN |
| ----------------------- | ----- | ------ | --------- | -------------- |
| Model size              | 125MB | 130MB  | 7MB       | 18MB           |
| Training time (1 epoch) | 45min | 50min  | 12min     | 20min          |
| Inference (ms/move)     | 180ms | 220ms  | 35ms      | 40ms           |
| Policy accuracy (hex8)  | 68%   | 71%    | 76%       | 78%            |
| Win rate vs Heuristic   | 73%   | 76%    | 82%       | 85%            |
| Hex geometry handling   | Poor  | Poor   | Native    | Native         |
| Territory connectivity  | Fair  | Good   | Excellent | Excellent      |

## Technical Notes

### Why Gradient-Based Approaches Struggled

1. **Continuous relaxation mismatch**: Games have discrete move spaces, continuous optimization introduces approximation errors
2. **Gradient instability**: Multi-step gradient descent during inference is fragile
3. **Local minima**: Energy landscapes have many local optima
4. **Hex geometry**: CNN backbones don't naturally encode hex connectivity

### Why GNN Works Better

1. **Native graph structure**: Message passing directly models territory connectivity
2. **Inductive bias**: Graph structure encodes game rules naturally
3. **Efficiency**: No iterative optimization at inference time
4. **Scalability**: 18x smaller models train 4x faster

## References

- **GNN Implementation**: `app/ai/gnn_ai.py`, `app/ai/neural_net/gnn_policy.py`
- **Hybrid Implementation**: `app/ai/hybrid_ai.py`, `app/ai/neural_net/hybrid_policy.py`
- **Training**: `scripts/train_gnn_policy.py`
- **Benchmarks**: `docs/benchmarks/gnn_vs_cnn_comparison.md`

## Support

For migration assistance or questions:

- Check GNN training examples: `ai-service/docs/GNN_MIGRATION.md`
- See hybrid policy docs: `ai-service/app/ai/neural_net/README.md`
- Open issue: https://github.com/your-org/ringrift/issues

---

**Last Updated**: December 28, 2025
**Scheduled Removal**: Q2 2026 (June 2026)
