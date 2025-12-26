# Experimental AI Algorithms

> **Last Updated**: 2025-12-19
> **Status**: Experimental
> **Location**: `app/ai/`

This document describes the novel AI algorithms developed for RingRift that go beyond traditional approaches like MCTS and policy networks.

## Overview

RingRift includes three experimental gradient-based AI architectures:

| Algorithm | Key Idea                                                     | Status             |
| --------- | ------------------------------------------------------------ | ------------------ |
| **EBMO**  | Energy-based gradient descent on action embeddings           | Active development |
| **GMO**   | Information-theoretic gradient optimization with uncertainty | Active development |
| **CAGE**  | Graph neural networks with constrained energy optimization   | Experimental       |

All three algorithms share a common theme: **continuous optimization in action embedding space** rather than discrete search (MCTS) or direct policy sampling.

---

## EBMO: Energy-Based Move Optimization

**Location**: `app/ai/ebmo_ai.py`, `app/ai/ebmo_network.py`

### Concept

EBMO learns an energy function E(s, a) over (state, action) pairs, where lower energy indicates better moves. At inference time, it uses gradient descent to find low-energy actions.

### Algorithm

```
1. Encode game state → state_embedding (done once)
2. For each restart:
   a. Initialize action_embedding from random legal move
   b. Gradient descent: a' = a - lr * ∇_a E(s, a)
   c. Periodically project to legal move manifold
   d. Track best (lowest energy) move found
3. Return best move across all restarts
```

### Key Innovations

- **Continuous optimization**: Gradient-guided exploration instead of discrete search
- **Multi-restart optimization**: Escapes local minima
- **Projection to legal moves**: Maintains feasibility during optimization

### Configuration

```python
from app.ai.ebmo_network import EBMOConfig

config = EBMOConfig(
    # State encoding (56-channel input with frame stacking)
    state_channels=56,
    state_hidden_dim=256,
    state_embed_dim=128,

    # Action encoding
    action_feature_dim=12,
    action_hidden_dim=128,
    action_embed_dim=128,

    # Energy network
    energy_hidden_dim=256,

    # Inference optimization
    num_restarts=5,
    optim_steps=20,
    optim_lr=0.1,
    projection_interval=5,
)
```

### Usage

```python
from app.ai.ebmo_ai import EBMO_AI
from app.models import AIConfig

config = AIConfig(difficulty=5)
ai = EBMO_AI(player_number=1, config=config, model_path="models/ebmo/best.pt")
move = ai.select_move(game_state)
```

### Training Scripts

**Note:** EBMO scripts were archived in 2025-12; the active entry point is
`python -m app.training.ebmo_trainer`.

| Script                                           | Purpose                     |
| ------------------------------------------------ | --------------------------- |
| `scripts/archive/ebmo/train_ebmo.py`             | Basic training (archived)   |
| `scripts/archive/ebmo/train_ebmo_quality.py`     | Quality training (archived) |
| `scripts/archive/ebmo/train_ebmo_curriculum.py`  | Curriculum (archived)       |
| `scripts/archive/ebmo/generate_ebmo_selfplay.py` | Self-play data (archived)   |
| `scripts/archive/ebmo/benchmark_ebmo_ladder.py`  | Benchmark (archived)        |

### Environment Variables

| Variable                   | Default                            | Description |
| -------------------------- | ---------------------------------- | ----------- |
| `RINGRIFT_EBMO_MODEL_PATH` | `models/ebmo/ebmo_square8_best.pt` | Model path  |

---

## GMO: Gradient Move Optimization

**Location**: `app/ai/gmo_ai.py`

### Concept

GMO combines gradient optimization with information-theoretic exploration using UCB-style bonuses for uncertainty and novelty.

### Algorithm

```
1. Encode game state and all legal moves into embeddings
2. Rank candidates using: value + β*uncertainty + γ*novelty (UCB-style)
3. For top-k candidates, run gradient ascent on the objective
4. Project optimized embeddings back to nearest legal moves
5. Select the best projected move
```

### Key Innovations

- **Uncertainty estimation**: MC Dropout provides calibrated uncertainty
- **Novelty tracking**: Avoids repeating similar moves
- **UCB-style exploration**: Balances exploitation vs exploration
- **Gradient ascent**: Optimizes in continuous space before discretizing

### Configuration

```python
from app.ai.gmo_ai import GMOConfig

# Tuned from hyperparameter sweep (100% vs Random, 62.5% vs Heuristic)
config = GMOConfig(
    # Embedding dimensions
    state_dim=128,
    move_dim=128,
    hidden_dim=256,

    # Optimization (tuned)
    top_k=5,           # Number of candidates to optimize
    optim_steps=5,     # Gradient steps per candidate
    lr=0.1,            # Learning rate

    # Information-theoretic parameters (tuned)
    beta=0.1,          # Exploration coefficient (low = exploitation)
    gamma=0.0,         # Novelty coefficient (disabled)

    # MC Dropout (critical - do not reduce)
    dropout_rate=0.1,
    mc_samples=10,     # Required for good performance

    # Novelty tracking
    novelty_memory_size=1000,
)
```

### Usage

```python
from app.ai.gmo_ai import GMO_AI
from app.models import AIConfig

config = AIConfig(difficulty=5)
ai = GMO_AI(player_number=1, config=config)
move = ai.select_move(game_state)
```

### Training Scripts

| Script                               | Purpose                     |
| ------------------------------------ | --------------------------- |
| `app/training/train_gmo.py`          | Basic GMO training          |
| `app/training/train_gmo_selfplay.py` | Self-play data              |
| `app/training/train_gmo_online.py`   | Online training             |
| `app/training/train_gmo_diverse.py`  | Diverse data training       |
| `scripts/gmo_hyperparam_sweep.py`    | Hyperparameter optimization |
| `scripts/gmo_ablation_study.py`      | Ablation experiments        |

### Performance Notes

- MC Dropout is **critical**: 0% win rate without it
- `mc_samples=10` is the minimum recommended value
- Low `beta=0.1` works best (exploitation-focused)
- Novelty bonus (`gamma`) provides no benefit

---

## CAGE: Constraint-Aware Graph Energy-Based Move Optimization

**Location**: `app/ai/cage_ai.py`, `app/ai/cage_network.py`

### Concept

CAGE combines graph neural networks for board representation with primal-dual optimization to enforce legality constraints during energy minimization.

### Algorithm

```
1. Represent board as graph:
   - Nodes = cells with features (piece type, player, etc.)
   - Edges = adjacencies with features (distance, direction)
2. Run GNN message passing to compute state embedding
3. For each legal move, compute energy with learned constraints
4. Use primal-dual optimization to find feasible minimum
5. Return lowest-energy legal move
```

### Key Innovations

- **Graph representation**: Natural for board games with arbitrary geometry
- **Learned constraints**: Legality encoded in energy function
- **Primal-dual optimization**: Stays on legal move manifold
- **Interpretable decomposition**: Can analyze energy components

### Configuration

```python
from app.ai.cage_network import CAGEConfig

config = CAGEConfig(
    # Graph neural network
    node_feature_dim=32,    # Features per cell
    edge_feature_dim=8,     # Features per edge
    gnn_hidden_dim=128,
    gnn_num_layers=4,
    gnn_num_heads=4,        # Attention heads

    # Action representation
    action_embed_dim=64,
    action_hidden_dim=128,

    # Energy network
    energy_hidden_dim=256,

    # Primal-dual parameters
    dual_lr=0.01,
    primal_steps=10,
    constraint_margin=0.1,
)
```

### Usage

```python
from app.ai.cage_ai import CAGE_AI
from app.models import AIConfig

config = AIConfig(difficulty=5)
ai = CAGE_AI(player_number=1, config=config, model_path="models/cage/best.pt")
move = ai.select_move(game_state)
```

### Training Scripts

| Script | Purpose                              |
| ------ | ------------------------------------ |
| (none) | CAGE training not yet wired to a CLI |

### Environment Variables

| Variable                   | Default                    | Description |
| -------------------------- | -------------------------- | ----------- |
| `RINGRIFT_CAGE_MODEL_PATH` | `models/cage/cage_best.pt` | Model path  |

---

## Comparison with Traditional Approaches

| Aspect             | Policy Network      | MCTS           | EBMO/GMO/CAGE           |
| ------------------ | ------------------- | -------------- | ----------------------- |
| Move selection     | softmax → sample    | visit counts   | gradient optimization   |
| Search type        | None (forward pass) | Tree search    | Continuous optimization |
| Exploration        | Entropy bonus       | UCB/PUCT       | Gradient + restarts     |
| Computational cost | O(1) per move       | O(simulations) | O(restarts × steps)     |
| Gradient usage     | Training only       | Training only  | Training + inference    |

---

## Research Status

### EBMO

- Active development with 56-channel input architecture
- Curriculum training and self-play pipelines
- Benchmarking against AI ladder (Random, Heuristic, MCTS)

### GMO

- Hyperparameter sweep completed
- Best configuration: 100% vs Random, 62.5% vs Heuristic
- MC Dropout critical for uncertainty estimation

### CAGE

- Experimental stage
- Graph representation complete
- Primal-dual optimization in development

---

## See Also

- [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md) - General training infrastructure
- [UNIFIED_AI_LOOP.md](UNIFIED_AI_LOOP.md) - Orchestrator for training loops
- [AI_TRAINING_PLAN.md](../roadmaps/AI_TRAINING_PLAN.md) - Long-term AI roadmap
