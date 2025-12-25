# MCTS Module

Monte Carlo Tree Search implementations for RingRift AI.

## Overview

This module provides advanced MCTS algorithms with features:

- PUCT exploration (AlphaZero-style)
- Progressive widening for large branching factors
- Virtual loss for parallel search
- Transposition tables for position caching
- Tree reuse between moves
- Pondering (think on opponent's time)

## Key Components

### `MCTSConfig` - Configuration

```python
from app.mcts import MCTSConfig

config = MCTSConfig(
    num_simulations=800,      # Simulations per search
    cpuct=1.5,                # Exploration constant
    temperature=1.0,          # Move selection temperature
    add_dirichlet_noise=True, # Root exploration noise
    dirichlet_alpha=0.3,      # Noise concentration
    dirichlet_epsilon=0.25,   # Noise weight
    virtual_loss=3.0,         # Parallel search virtual loss
    progressive_widening=True, # For large action spaces
)
```

### `ImprovedMCTS` - Core Algorithm

```python
from app.mcts import ImprovedMCTS, MCTSConfig

# Create with neural network evaluator
mcts = ImprovedMCTS(
    config=MCTSConfig(num_simulations=800),
    network=my_neural_net,
    transposition_table=TranspositionTable(max_size=100000),
)

# Search for best move
policy, value = mcts.search(game_state, player=1)
best_move = mcts.select_move(game_state, temperature=0.1)
```

### `ParallelMCTS` - Multi-threaded Search

```python
from app.mcts import ParallelMCTS, MCTSConfig

# Parallel search with virtual loss
mcts = ParallelMCTS(
    config=MCTSConfig(
        num_simulations=1600,
        virtual_loss=3.0,
    ),
    network=my_neural_net,
    num_workers=4,
)

# Search uses multiple threads
policy, value = mcts.search(game_state, player=1)
```

### `MCTSWithPonder` - Background Thinking

```python
from app.mcts import MCTSWithPonder

# Create with pondering enabled
mcts = MCTSWithPonder(
    config=config,
    network=network,
)

# Start pondering in background
mcts.start_ponder(game_state, player=1)

# When move received, retrieve tree
mcts.stop_ponder()
tree = mcts.get_tree()
```

### `TranspositionTable` - Position Caching

```python
from app.mcts import TranspositionTable

# Create with max size
tt = TranspositionTable(max_size=100000)

# Store position evaluation
tt.put(position_hash, policy=[0.1, 0.2, ...], value=0.6)

# Retrieve if exists
result = tt.get(position_hash)
if result:
    policy, value = result
```

### `MCTSNode` - Tree Node

```python
from app.mcts import MCTSNode

# Nodes track visit counts and values
node = MCTSNode(
    prior=0.15,           # Policy prior for this action
    parent=parent_node,
)

# PUCT score for selection
puct = node.compute_puct(parent_visits=100, cpuct=1.5)

# Update after simulation
node.backup(value=0.7)
```

## Integration with AI

For gameplay, use the integrated AI class:

```python
from app.ai.improved_mcts_ai import ImprovedMCTSAI

ai = ImprovedMCTSAI(
    board_type="hex8",
    num_players=2,
    model_path="models/hex8_2p.pth",
    simulations=800,
)

# Get move for position
move = ai.get_move(game_state)
```

## Configuration Options

| Parameter              | Default | Description                  |
| ---------------------- | ------- | ---------------------------- |
| `num_simulations`      | 800     | Simulations per search       |
| `cpuct`                | 1.5     | Exploration constant         |
| `temperature`          | 1.0     | Move selection randomness    |
| `add_dirichlet_noise`  | True    | Root exploration noise       |
| `dirichlet_alpha`      | 0.3     | Noise concentration          |
| `dirichlet_epsilon`    | 0.25    | Noise weight at root         |
| `virtual_loss`         | 3.0     | Penalty for parallel paths   |
| `progressive_widening` | False   | Gradual action expansion     |
| `fpu_reduction`        | 0.0     | First-play urgency reduction |

## Algorithm Details

### PUCT Formula

```
PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

- `Q(s,a)`: Mean action value
- `P(s,a)`: Prior probability from policy network
- `N(s)`: Parent visit count
- `N(s,a)`: Action visit count
- `c_puct`: Exploration constant

### Virtual Loss

During parallel search, temporarily reduce Q-value of selected path:

```
Q_virtual = (W - virtual_loss) / (N + 1)
```

This encourages threads to explore different paths.

### Progressive Widening

For large action spaces, limit children based on visit count:

```
max_children = k * N(s)^alpha
```

Where `k` and `alpha` are tunable parameters.

## See Also

- `app/ai/gumbel_common.py` - Gumbel MCTS variant (used for selfplay)
- `app/ai/improved_mcts_ai.py` - Integrated AI player class
