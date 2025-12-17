# MCTS Integration Guide

This document describes the Monte Carlo Tree Search (MCTS) implementation in RingRift, including standard MCTS, Gumbel MCTS, and integration with neural network training.

## Overview

RingRift implements two MCTS variants:

1. **Standard MCTS** (`app/ai/mcts_ai.py`) - Traditional MCTS with UCT selection and neural network evaluation
2. **Gumbel MCTS** (`app/ai/gumbel_mcts_ai.py`) - AlphaZero-style MCTS with Sequential Halving for sample efficiency

## Standard MCTS

### Usage

```python
from app.ai.mcts_ai import MCTSAI
from app.models import AIConfig, GameState

config = AIConfig(
    difficulty=7,  # Triggers neural MCTS
    mcts_simulations=800,
    use_incremental_search=True,
)

ai = MCTSAI(game_state, config)
move = ai.choose_move(game_state)
```

### Configuration Options

| Parameter                | Type  | Default | Description                               |
| ------------------------ | ----- | ------- | ----------------------------------------- |
| `difficulty`             | int   | 7       | Tier 7+ enables neural network evaluation |
| `mcts_simulations`       | int   | 800     | Number of MCTS simulations per move       |
| `use_incremental_search` | bool  | True    | Use make/unmake for efficiency            |
| `exploration_constant`   | float | 1.41    | UCT exploration coefficient (c_puct)      |
| `temperature`            | float | 1.0     | Move selection temperature                |

### Search Modes

**Incremental Search (Default)**

- Uses `MutableGameState` with make/unmake operations
- Significantly reduces memory allocation overhead
- Recommended for production

**Legacy Search**

- Uses full `GameState` clones at each node
- More memory-intensive but useful for debugging
- Enable with `use_incremental_search=False`

## Gumbel MCTS

Implements the Gumbel AlphaZero algorithm for more sample-efficient search.

### Key Innovations

1. **Gumbel-Top-K Sampling**: Samples k actions without replacement using Gumbel noise added to policy logits
2. **Sequential Halving**: Divides simulation budget across log2(k) phases, progressively halving candidates
3. **Completed Q-values**: Principled value estimates accounting for visit count asymmetry

### Usage

```python
from app.ai.gumbel_mcts_ai import GumbelMCTSAI
from app.models import AIConfig, GameState

config = AIConfig(
    difficulty=8,
    mcts_simulations=200,  # More efficient - needs fewer sims
    gumbel_top_k=16,
    gumbel_c_visit=50.0,
)

ai = GumbelMCTSAI(game_state, config)
move = ai.choose_move(game_state)
```

### When to Use Gumbel MCTS

- **Faster search**: 2-3x fewer simulations needed for same quality
- **Better for training**: Produces cleaner policy targets
- **Self-play data generation**: Ideal for generating MCTS visit distributions

## Neural Network Integration

### Policy Network Evaluation

MCTS can use neural networks for:

1. **Prior policy (P)**: Initial action probabilities for tree expansion
2. **Value prediction (V)**: Position evaluation without rollouts

```python
# Automatic with difficulty >= 6
config = AIConfig(difficulty=7, mcts_simulations=400)
ai = MCTSAI(game_state, config)

# Manual network specification
from app.ai.neural_net import load_model
model = load_model("models/nnue/square8_2p.pt")
ai = MCTSAI(game_state, config, neural_model=model)
```

### MCTS Data for Training

MCTS generates training data with visit distributions:

```python
# After search
visit_distribution = ai.get_visit_distribution()
# Returns: {"d4": 0.45, "e4": 0.30, "c4": 0.15, ...}

# For training data
game_record = {
    "moves": [
        {
            "move": "d4",
            "mcts_policy": visit_distribution,  # KL loss target
            "value": ai.root_value,  # Value target
        }
    ]
}
```

### KL Divergence Training

Train policy heads to match MCTS distributions:

```bash
# Generate MCTS data
python scripts/run_hybrid_selfplay.py \
    --board-type square8 \
    --engine-mode gumbel-mcts \
    --mcts-sims 400 \
    --output data/selfplay/mcts_square8/games.jsonl

# Train with KL loss
python scripts/train_nnue_policy.py \
    --jsonl data/selfplay/mcts_square8/games.jsonl \
    --auto-kl-loss \
    --epochs 50
```

See [NNUE_POLICY_TRAINING.md](NNUE_POLICY_TRAINING.md) for full KL loss documentation.

## Selfplay Data Generation

### GPU Selfplay with MCTS

```bash
# Standard MCTS selfplay
python scripts/run_gpu_selfplay.py \
    --board-type square8 \
    --num-players 2 \
    --num-games 1000 \
    --mcts-sims 200

# Gumbel MCTS for softer policy targets
python scripts/run_hybrid_selfplay.py \
    --board-type hex8 \
    --engine-mode gumbel-mcts \
    --mcts-sims 200 \
    --gumbel-top-k 16 \
    --num-games 500
```

### Reanalyze Existing Games

Add MCTS distributions to games for KL training:

```bash
python scripts/reanalyze_mcts_policy.py \
    --input data/games/existing.jsonl \
    --output data/games/with_mcts.jsonl \
    --mcts-sims 400 \
    --parallel 4
```

## Transposition Table

MCTS uses a bounded transposition table for position sharing:

```python
from app.ai.bounded_transposition_table import BoundedTranspositionTable

# Default: 1M positions
table = BoundedTranspositionTable(capacity=1_000_000)

# MCTS automatically uses transposition table
ai = MCTSAI(game_state, config)
ai.transposition_table = table
```

### Memory Management

| Capacity | Approx Memory | Use Case              |
| -------- | ------------- | --------------------- |
| 100K     | ~50MB         | Low-memory devices    |
| 1M       | ~500MB        | Default               |
| 10M      | ~5GB          | Tournament evaluation |

## Performance Tuning

### Simulation Count vs Quality

| Board     | Min Sims | Recommended | Tournament |
| --------- | -------- | ----------- | ---------- |
| square8   | 100      | 400         | 800+       |
| hex8      | 200      | 600         | 1200+      |
| hexagonal | 400      | 1000        | 2000+      |

### Parallel MCTS

Multi-threaded leaf parallelization:

```python
config = AIConfig(
    mcts_simulations=800,
    mcts_threads=4,  # Parallel leaf expansion
)
```

### Neural Batching

For GPU efficiency with neural evaluation:

```python
from app.ai.async_nn_eval import AsyncNeuralBatcher

batcher = AsyncNeuralBatcher(
    model=neural_model,
    batch_size=32,
    timeout_ms=10,
)

ai = MCTSAI(game_state, config, neural_batcher=batcher)
```

## Difficulty Tiers

MCTS is enabled at different difficulty tiers:

| Tier | Simulations | Neural  | Description         |
| ---- | ----------- | ------- | ------------------- |
| 5    | 100         | No      | Basic MCTS          |
| 6    | 200         | NNUE    | NNUE-guided MCTS    |
| 7    | 400         | NN+NNUE | Full neural MCTS    |
| 8    | 800         | NN+NNUE | Tournament strength |
| 9    | 1600        | NN+NNUE | Analysis mode       |

## Related Files

- `app/ai/mcts_ai.py` - Standard MCTS implementation (122KB)
- `app/ai/gumbel_mcts_ai.py` - Gumbel MCTS implementation (21KB)
- `app/ai/async_nn_eval.py` - Neural network batching
- `app/ai/bounded_transposition_table.py` - Transposition table
- `scripts/run_hybrid_selfplay.py` - MCTS selfplay generation
- `scripts/reanalyze_mcts_policy.py` - MCTS policy reanalysis

## References

- [Gumbel AlphaZero Paper](https://arxiv.org/abs/2104.06303) - Policy improvement by planning with Gumbel
- [AlphaZero](https://arxiv.org/abs/1712.01815) - Mastering games with self-play
- [KL Divergence Training](NNUE_POLICY_TRAINING.md) - Training with MCTS distributions
