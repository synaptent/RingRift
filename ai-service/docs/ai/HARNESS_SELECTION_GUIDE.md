# AI Harness Selection Guide

**Last Updated**: January 3, 2026

This guide helps you choose the right AI harness for your use case. A harness wraps an AI algorithm and model to provide a unified interface for move selection and evaluation.

## Quick Reference

| Harness       | Best For                 | Players  | Model Types | Speed     |
| ------------- | ------------------------ | -------- | ----------- | --------- |
| `GUMBEL_MCTS` | Training data generation | 2-4      | NN          | Medium    |
| `GPU_GUMBEL`  | High-throughput selfplay | 2-4      | NN          | Fast      |
| `MINIMAX`     | 2-player evaluation      | 2 only   | NN, NNUE    | Fast      |
| `MAXN`        | Multiplayer games        | 3-4 only | NN, NNUE    | Medium    |
| `BRS`         | Fast multiplayer         | 3-4 only | NN, NNUE    | Fast      |
| `POLICY_ONLY` | Baselines, fast play     | 2-4      | NN, NNUE    | Very Fast |
| `DESCENT`     | Research, exploration    | 2-4      | NN          | Slow      |
| `HEURISTIC`   | Bootstrap, baselines     | 2-4      | None        | Very Fast |
| `RANDOM`      | Sanity checks, diversity | 2-4      | None        | Instant   |

## Decision Tree

```
Need training data with visit distributions?
├── Yes → GUMBEL_MCTS or GPU_GUMBEL
│         └── Batch processing? → GPU_GUMBEL
│         └── Single game? → GUMBEL_MCTS
│
├── 2-player game?
│   ├── Using NNUE? → MINIMAX
│   └── Using NN? → GUMBEL_MCTS or MINIMAX
│
├── 3-4 player game?
│   ├── Need accuracy? → MAXN
│   └── Need speed? → BRS
│
├── Just need a baseline?
│   ├── Any neural network → POLICY_ONLY
│   └── No neural network → HEURISTIC or RANDOM
│
└── Research/exploration? → DESCENT
```

## Harness Details

### GUMBEL_MCTS (Gumbel AlphaZero MCTS)

**Best for**: Generating high-quality training data with visit distributions

```python
from app.ai.harness import create_harness, HarnessType

harness = create_harness(
    HarnessType.GUMBEL_MCTS,
    model_path="models/canonical_hex8_2p.pth",
    board_type="hex8",
    num_players=2,
    simulations=200,  # Higher = better quality, slower
)

move, metadata = harness.evaluate(game_state, player_number=1)
visit_distribution = harness.get_visit_distribution()  # For soft targets
```

**Key Features**:

- Sequential halving for efficient simulation budget
- Visit distribution capture for training soft policy targets
- Supports difficulty levels via simulation count
- Requires policy head (NN only, not NNUE without policy)

**Simulation Budget Reference**:
| Use Case | Simulations | Quality | Speed |
|----------|-------------|---------|-------|
| Throughput (bootstrap) | 64 | Low | Fast |
| Standard | 150 | Medium | Medium |
| Quality | 800 | High | Slow |
| Ultimate | 1600 | Very High | Very Slow |

### GPU_GUMBEL (GPU-Accelerated Gumbel MCTS)

**Best for**: High-throughput batch selfplay on GPU

```python
harness = create_harness(
    HarnessType.GPU_GUMBEL,
    model_path="models/canonical_hex8_2p.pth",
    board_type="hex8",
    num_players=2,
    simulations=200,
    extra={"batch_size": 64},  # Process 64 games simultaneously
)
```

**Key Features**:

- 6-57x speedup on CUDA (GPU-dependent)
- Fully vectorized move generation
- Best for bulk selfplay data generation
- Requires policy head

**When to use GPU_GUMBEL vs GUMBEL_MCTS**:

- Use GPU_GUMBEL for generating 100+ games in batch
- Use GUMBEL_MCTS for single-game play or interactive use

### MINIMAX (Alpha-Beta Search)

**Best for**: Fast, strong 2-player evaluation

```python
harness = create_harness(
    HarnessType.MINIMAX,
    model_path="models/nnue_hex8_2p.pt",  # NNUE preferred
    board_type="hex8",
    num_players=2,
    depth=4,  # Search depth (higher = stronger, slower)
)
```

**Key Features**:

- Alpha-beta pruning for efficient search
- Works with NNUE (faster) or NN (slower)
- **2-player only** - will raise ValueError for 3-4 players
- No policy head required

**Depth Reference**:
| Depth | Strength | Time (typical) |
|-------|----------|----------------|
| 2 | Weak | <10ms |
| 4 | Medium | 50-200ms |
| 6 | Strong | 500ms-2s |
| 8 | Very Strong | 2-10s |

### MAXN (Max-N Multiplayer Search)

**Best for**: Accurate multiplayer evaluation (3-4 players)

```python
harness = create_harness(
    HarnessType.MAXN,
    model_path="models/canonical_hex8_4p.pth",
    board_type="hex8",
    num_players=4,  # 3 or 4 players only
    depth=3,
)
```

**Key Features**:

- Each player maximizes own score
- More accurate than BRS, but slower
- **3-4 players only** - will raise ValueError for 2 players
- Supports both NN and NNUE

### BRS (Best-Reply Search)

**Best for**: Fast multiplayer evaluation (3-4 players)

```python
harness = create_harness(
    HarnessType.BRS,
    model_path="models/canonical_hex8_4p.pth",
    board_type="hex8",
    num_players=4,  # 3 or 4 players only
    depth=4,
)
```

**Key Features**:

- Greedy search assuming opponents play best replies
- Faster than MAXN with reasonable accuracy
- **3-4 players only** - will raise ValueError for 2 players
- Good for quick gauntlet evaluation

### POLICY_ONLY (Direct Policy Sampling)

**Best for**: Fast baselines, sanity checks

```python
harness = create_harness(
    HarnessType.POLICY_ONLY,
    model_path="models/canonical_hex8_2p.pth",
    board_type="hex8",
    num_players=2,
)
```

**Key Features**:

- No search - directly samples from policy network
- Very fast (single forward pass)
- Useful for baseline comparisons
- Requires policy head

### HEURISTIC (Hand-Crafted Evaluation)

**Best for**: Bootstrap phase, baselines without neural networks

```python
harness = create_harness(
    HarnessType.HEURISTIC,
    board_type="hex8",
    num_players=2,
    # No model_path needed
)
```

**Key Features**:

- No neural network required
- Uses hand-crafted position evaluation
- Fast and deterministic
- Good for initial selfplay before NN training

### RANDOM (Uniform Random)

**Best for**: Sanity checks, diversity injection

```python
harness = create_harness(
    HarnessType.RANDOM,
    board_type="hex8",
    num_players=2,
)
```

**Key Features**:

- Uniformly random legal move selection
- No model or evaluation
- Useful for baseline win-rate sanity checks

### DESCENT (Gradient Descent Search)

**Best for**: Research, exploration

```python
harness = create_harness(
    HarnessType.DESCENT,
    model_path="models/canonical_hex8_2p.pth",
    board_type="hex8",
    num_players=2,
)
```

**Key Features**:

- Uses gradient descent on value function
- Experimental/research algorithm
- Requires full NN with gradients

## Compatibility Matrix

| Harness     | NN  | NNUE | Heuristic | Policy Required | Min Players | Max Players |
| ----------- | --- | ---- | --------- | --------------- | ----------- | ----------- |
| GUMBEL_MCTS | ✓   | -    | -         | Yes             | 2           | 4           |
| GPU_GUMBEL  | ✓   | -    | -         | Yes             | 2           | 4           |
| MINIMAX     | ✓   | ✓    | -         | No              | 2           | 2           |
| MAXN        | ✓   | ✓    | -         | No              | 3           | 4           |
| BRS         | ✓   | ✓    | -         | No              | 3           | 4           |
| POLICY_ONLY | ✓   | ✓\*  | -         | Yes             | 2           | 4           |
| DESCENT     | ✓   | -    | -         | Yes             | 2           | 4           |
| HEURISTIC   | -   | -    | ✓         | No              | 2           | 4           |
| RANDOM      | -   | -    | -         | No              | 2           | 4           |

\*NNUE with policy head only

## Common Patterns

### Gauntlet Evaluation

```python
from app.ai.harness import get_harnesses_for_model_and_players, create_harness, ModelType

# Get all compatible harnesses for model and player count
harnesses = get_harnesses_for_model_and_players(
    model_type=ModelType.NEURAL_NET,
    num_players=4,
)
# Returns: [GUMBEL_MCTS, GPU_GUMBEL, MAXN, BRS, POLICY_ONLY, DESCENT]
# Note: MINIMAX excluded (2-player only)

for harness_type in harnesses:
    harness = create_harness(
        harness_type,
        model_path="models/canonical_hex8_4p.pth",
        board_type="hex8",
        num_players=4,
    )
    # Run games and track Elo per harness
```

### Training Data Generation

```python
from app.ai.harness import create_harness, HarnessType

# Use GPU_GUMBEL for batch selfplay
harness = create_harness(
    HarnessType.GPU_GUMBEL,
    model_path="models/canonical_hex8_2p.pth",
    board_type="hex8",
    num_players=2,
    simulations=200,
    extra={"batch_size": 64},
)

# After each move, capture visit distribution for training
move, metadata = harness.evaluate(game_state, player_number)
visit_dist = harness.get_visit_distribution()  # Dict[action_key, visit_count]
```

### Composite Elo Tracking

```python
# Each harness generates a composite ID for Elo tracking
harness = create_harness(HarnessType.MINIMAX, ...)
participant_id = harness.get_composite_participant_id()
# Format: "{model_id}:{harness_type}:{config_hash}"
# Example: "ringrift_hex8_2p:minimax:a1b2c3d4"
```

## Troubleshooting

### "Harness requires 2-2 players, got 4"

You're trying to use MINIMAX for a 4-player game. Use MAXN or BRS instead:

```python
# Wrong
harness = create_harness(HarnessType.MINIMAX, num_players=4)  # ValueError!

# Correct
harness = create_harness(HarnessType.MAXN, num_players=4)  # OK
```

### "Harness does not support NNUE"

GUMBEL_MCTS, GPU_GUMBEL, and DESCENT require full NN with policy head:

```python
# Wrong
harness = create_harness(
    HarnessType.GUMBEL_MCTS,
    model_path="models/nnue_hex8_2p.pt",  # NNUE without policy
)  # ValueError!

# Correct
harness = create_harness(
    HarnessType.MINIMAX,  # MINIMAX supports NNUE
    model_path="models/nnue_hex8_2p.pt",
)
```

### Low Quality Training Data

Increase simulation count for GUMBEL_MCTS:

```python
# Low quality (fast)
harness = create_harness(HarnessType.GUMBEL_MCTS, simulations=64)

# High quality (slow)
harness = create_harness(HarnessType.GUMBEL_MCTS, simulations=800)
```

## See Also

- `app/ai/harness/base_harness.py` - Base class implementation
- `app/ai/harness/harness_registry.py` - Compatibility matrix
- `app/ai/harness/implementations.py` - Concrete harness implementations
- `docs/ai/GUMBEL_MCTS_GUIDE.md` - Detailed Gumbel MCTS documentation
