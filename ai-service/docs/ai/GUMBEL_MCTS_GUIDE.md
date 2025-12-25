# Gumbel MCTS Selection Guide

Guide for choosing the right Gumbel MCTS variant for your use case.

## Overview

RingRift has multiple Gumbel MCTS implementations optimized for different scenarios:

| Variant                  | Best For                  | Throughput | Quality |
| ------------------------ | ------------------------- | ---------- | ------- |
| `gumbel_mcts_ai.py`      | Single game, CPU          | Low        | High    |
| `tensor_gumbel_tree.py`  | Single game, GPU          | Medium     | High    |
| `batched_gumbel_mcts.py` | Multiple games, CPU batch | Medium     | High    |
| `multi_game_gumbel.py`   | Selfplay, GPU parallel    | Very High  | High    |

## Quick Start

### For Selfplay (Recommended)

```python
from app.ai.multi_game_gumbel import MultiGameGumbelRunner

runner = MultiGameGumbelRunner(
    num_games=64,           # Run 64 games in parallel
    simulation_budget=800,  # Simulations per move
    num_sampled_actions=16, # Actions to consider
    neural_net=my_nn,
    device="cuda",
)

results = runner.run_batch()
```

### For Single Game Evaluation

```python
from app.ai.gumbel_mcts_ai import GumbelMCTSAI

ai = GumbelMCTSAI(
    neural_net=my_nn,
    simulation_budget=800,
    num_sampled_actions=16,
)

move = ai.select_move(game_state)
```

## Budget Tiers

Use these predefined budgets from `gumbel_common.py`:

```python
from app.ai.gumbel_common import (
    GUMBEL_BUDGET_THROUGHPUT,  # 64 - Fast selfplay
    GUMBEL_BUDGET_STANDARD,    # 150 - Balanced
    GUMBEL_BUDGET_QUALITY,     # 800 - High quality
    GUMBEL_BUDGET_ULTIMATE,    # 1600 - Maximum quality
    get_budget_for_difficulty,
)

# Map difficulty to budget
budget = get_budget_for_difficulty("hard")  # Returns 800
```

## Variant Details

### 1. GumbelMCTSAI (`gumbel_mcts_ai.py`)

**Use when:** Playing single games, need interpretable search tree

```python
from app.ai.gumbel_mcts_ai import GumbelMCTSAI

ai = GumbelMCTSAI(
    neural_net=nn,
    simulation_budget=800,
    num_sampled_actions=16,
    c_puct=1.5,
    dirichlet_alpha=0.3,
)

move = ai.select_move(state)
```

**Pros:**

- Full search tree available for analysis
- Works on CPU
- Clear code structure

**Cons:**

- Slow for batch processing
- Sequential NN calls

### 2. TensorGumbelTree (`tensor_gumbel_tree.py`)

**Use when:** Single game on GPU, need speed

```python
from app.ai.tensor_gumbel_tree import TensorGumbelTree

tree = TensorGumbelTree(
    neural_net=nn,
    budget=800,
    device="cuda",
)

move = tree.search(state)
```

**Pros:**

- GPU-accelerated
- Vectorized operations

**Cons:**

- More memory usage
- Single game only

### 3. BatchedGumbelMCTS (`batched_gumbel_mcts.py`)

**Use when:** Multiple games, limited GPU memory

```python
from app.ai.batched_gumbel_mcts import BatchedGumbelMCTS

mcts = BatchedGumbelMCTS(
    neural_net=nn,
    batch_size=8,
    simulation_budget=800,
)

moves = mcts.search_batch(game_states)
```

**Pros:**

- Batched NN evaluation
- Memory efficient

**Cons:**

- Not as fast as full GPU parallel

### 4. MultiGameGumbelRunner (`multi_game_gumbel.py`)

**Use when:** Selfplay, maximum throughput

```python
from app.ai.multi_game_gumbel import MultiGameGumbelRunner

runner = MultiGameGumbelRunner(
    num_games=64,
    simulation_budget=800,
    neural_net=nn,
    device="cuda",
)

results = runner.run_batch()
# Returns list of GameResult with moves, winner, etc.
```

**Pros:**

- 10-20x faster than sequential
- Optimized for selfplay data generation
- Handles terminal states correctly (as of Dec 2025)

**Cons:**

- Higher memory usage
- Requires GPU

## Performance Comparison

Measured on GH200 (96GB), hex8 board, 800 budget:

| Variant             | Games/sec | GPU Util | Notes            |
| ------------------- | --------- | -------- | ---------------- |
| gumbel_mcts_ai      | 0.5       | 20%      | CPU bottleneck   |
| tensor_gumbel_tree  | 2.0       | 60%      | Single game      |
| batched_gumbel_mcts | 5.0       | 70%      | 8-game batch     |
| multi_game_gumbel   | 15.0      | 95%      | 64-game parallel |

## Configuration Tips

### For Training Data Quality

```python
runner = MultiGameGumbelRunner(
    simulation_budget=800,      # High quality
    num_sampled_actions=16,     # Good action coverage
    temperature=1.0,            # Exploration
    temperature_threshold=30,   # Switch to greedy after move 30
)
```

### For Fast Iteration

```python
runner = MultiGameGumbelRunner(
    simulation_budget=64,       # Fast
    num_sampled_actions=8,      # Fewer actions
    temperature=0.5,            # Less exploration
)
```

### For Evaluation

```python
ai = GumbelMCTSAI(
    simulation_budget=1600,     # Maximum quality
    num_sampled_actions=32,     # Consider more actions
    temperature=0.0,            # Deterministic
)
```

## Common Issues

### Out of Memory

Reduce `num_games` or `simulation_budget`:

```python
runner = MultiGameGumbelRunner(
    num_games=32,  # Reduced from 64
    simulation_budget=400,  # Reduced from 800
)
```

### Slow Performance

1. Check GPU is being used: `nvidia-smi`
2. Increase batch size if memory allows
3. Use `multi_game_gumbel` instead of sequential variants

### Poor Move Quality

1. Increase `simulation_budget`
2. Check neural network is loaded correctly
3. Verify temperature settings

## See Also

- `app/ai/gumbel_common.py` - Shared data structures
- `scripts/selfplay.py` - CLI for selfplay with Gumbel
- `app/training/temperature_scheduling.py` - Temperature schedules
