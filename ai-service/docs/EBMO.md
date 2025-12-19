# Energy-Based Move Optimization (EBMO)

A novel game-playing AI algorithm that uses gradient descent on continuous action embeddings at inference time to find optimal moves.

## Overview

Unlike traditional approaches (policy networks, MCTS, minimax), EBMO navigates a continuous action manifold using gradient descent, then projects to the nearest legal discrete move.

| Approach       | Selection Method           | EBMO Difference                  |
| -------------- | -------------------------- | -------------------------------- |
| Policy Network | Softmax → argmax/sample    | Gradient descent on action space |
| MCTS           | Tree search → visit counts | Continuous optimization, no tree |
| Minimax        | Depth-limited search       | No explicit game tree traversal  |
| Q-Learning     | Q(s,a) lookup → argmax     | Optimizes embeddings, not lookup |

## Algorithm

```
Standard Policy:  state → NN → logits → softmax → move
EBMO:            state → optimize(action_embedding) via ∇E → project → move
```

### Inference Steps

1. **Encode state** once using CNN backbone
2. **Multi-restart optimization** (default: 5 restarts):
   - Initialize action embedding from random legal move
   - Run gradient descent (50 steps): `a' = a - lr * ∇_a E(s, a)`
   - Periodically project to legal move manifold
3. **Select best move** with lowest final energy

### Training

Uses contrastive learning with outcome weighting:

- **Positive samples**: Moves actually played (from winning games)
- **Negative samples**: Random legal moves or moves from losing games
- **Loss**: InfoNCE-style contrastive loss + outcome-weighted energy loss

## Usage

### Creating EBMO AI

```python
from app.ai.ebmo_ai import EBMO_AI
from app.models import AIConfig

# With trained model
ai = EBMO_AI(
    player_number=1,
    config=AIConfig(difficulty=5),
    model_path="models/ebmo/ebmo_square8.pt"
)

# Select move
move = ai.select_move(game_state)
```

### Via Factory

```python
from app.ai.factory import AIFactory
from app.models.core import AIType, AIConfig

# Using AIType enum
ai = AIFactory.create(AIType.EBMO, player_number=1, config=AIConfig(difficulty=5))

# For tournaments
ai = AIFactory.create_for_tournament("ebmo", player_number=1)
```

### Training

```bash
# Train on cluster
python scripts/train_ebmo.py \
  --data-dir data/training \
  --epochs 100 \
  --batch-size 512 \
  --lr 0.001 \
  --output-dir models/ebmo

# With specific options
python scripts/train_ebmo.py \
  --data-dir data/training/ebmo_sq8 \
  --epochs 100 \
  --batch-size 512 \
  --num-negatives 7 \
  --outcome-weight 0.5 \
  --device cuda
```

### Self-Play Data Generation

```python
from app.training.parallel_selfplay import generate_dataset_parallel
from app.models import BoardType

generate_dataset_parallel(
    num_games=1000,
    output_file="data/ebmo_selfplay.npz",
    board_type=BoardType.SQUARE8,
    engine="ebmo",  # Use EBMO for self-play
)
```

## Architecture

### Network Components

```
┌─────────────────────────────────────────────────────────────┐
│                    EBMO Network                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  State Encoder (CNN):                                        │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                │
│  │ 56×8×8   │ → │ ResBlocks│ → │ 256-dim  │ = s_embed      │
│  │ features │   │ (6 SE)   │   │ state    │                │
│  └──────────┘   └──────────┘   └──────────┘                │
│                                                              │
│  Action Encoder:                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                │
│  │ 14-dim   │ → │ MLP      │ → │ 128-dim  │ = a_embed      │
│  │ features │   │ (3 layer)│   │ action   │                │
│  └──────────┘   └──────────┘   └──────────┘                │
│                                                              │
│  Energy Head:                                                │
│  ┌────────────────┐   ┌──────────┐   ┌────────┐            │
│  │ concat(s,a)    │ → │ MLP      │ → │ scalar │ = E(s,a)   │
│  │ 384-dim        │   │ (3 layer)│   │ energy │            │
│  └────────────────┘   └──────────┘   └────────┘            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Action Embedding (14 dimensions)

| Dims  | Feature                           |
| ----- | --------------------------------- |
| 0-1   | from_x, from_y (normalized [0,1]) |
| 2-3   | to_x, to_y (normalized [0,1])     |
| 4-11  | move_type (8-dim one-hot)         |
| 12-13 | direction vector                  |

## Configuration

### EBMOConfig Defaults

```python
@dataclass
class EBMOConfig:
    # Embedding dimensions
    state_embed_dim: int = 256
    action_embed_dim: int = 128
    energy_hidden_dim: int = 256
    num_energy_layers: int = 3

    # Inference optimization
    optim_steps: int = 50        # Gradient descent steps
    optim_lr: float = 0.1        # Action embedding learning rate
    num_restarts: int = 5        # Multi-restart count
    projection_temperature: float = 0.5
    project_every_n_steps: int = 10

    # Training
    contrastive_temperature: float = 0.1
    num_negatives: int = 15
    outcome_weight: float = 0.5  # Balance contrastive vs outcome loss
    learning_rate: float = 0.001

    # Board
    board_size: int = 8
    num_input_channels: int = 56  # 14 planes × 4 history frames
```

### AIConfig Integration

Override EBMO parameters via `AIConfig.extra`:

```python
config = AIConfig(
    difficulty=5,
    extra={
        'ebmo_optim_steps': 100,    # More optimization steps
        'ebmo_restarts': 10,         # More restarts
        'ebmo_temperature': 0.3,     # Lower projection temperature
    }
)
```

## Performance

### Benchmark Results (Dec 2025)

| Opponent            | EBMO Win Rate    | Notes                    |
| ------------------- | ---------------- | ------------------------ |
| RandomAI            | **100%** (15/15) | After 25 epochs training |
| HeuristicAI         | 0% (0/10)        | Needs more training      |
| MinimaxAI (depth=2) | 0% (0/10)        | Needs more training      |

### Inference Speed

- ~148ms per move on GH200 GPU
- ~113ms per move on Apple M-series (MPS)
- Scales with `optim_steps × num_restarts`

## Files

| File                           | Description                          |
| ------------------------------ | ------------------------------------ |
| `app/ai/ebmo_network.py`       | Network architecture, loss functions |
| `app/ai/ebmo_ai.py`            | AI agent implementation              |
| `app/training/ebmo_dataset.py` | Dataset with contrastive sampling    |
| `app/training/ebmo_trainer.py` | Training loop                        |
| `scripts/train_ebmo.py`        | Training script                      |

## Theoretical Advantages

1. **Continuous exploration**: Can search between discrete move options
2. **Gradient guidance**: More directed than random sampling (MCTS rollouts)
3. **Amortized search**: Energy landscape encodes search knowledge
4. **Scalability**: O(optimization_steps) vs O(tree_size) for MCTS

## Potential Limitations

1. **Local minima**: Gradient descent may get stuck (mitigated by restarts)
2. **Projection quality**: Continuous → discrete mapping
3. **Training stability**: Energy-based models can be sensitive

## References

- Energy-Based Models: LeCun et al., "A Tutorial on Energy-Based Learning"
- Contrastive Learning: InfoNCE loss from CPC (van den Oord et al.)
- Game AI: AlphaZero (Silver et al.), Gumbel MuZero (Danihelka et al.)
