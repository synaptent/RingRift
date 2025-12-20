# Energy-Based Move Optimization (EBMO)

An energy-based approach to game move selection that applies SPEN-style test-time optimization to board game AI, using continuous action embeddings with projection to legal discrete moves.

## Overview

EBMO applies structured prediction energy networks (SPENs) to game move selection: learn an energy function E(s, a) over (state, action) pairs, then use gradient descent to find low-energy actions at inference time, projecting to the nearest legal discrete move.

| Approach       | Selection Method            | EBMO Approach                          |
| -------------- | --------------------------- | -------------------------------------- |
| Policy Network | Softmax -> argmax/sample    | Gradient descent in action space       |
| MCTS           | Tree search -> visit counts | Continuous optimization, no tree       |
| Minimax        | Depth-limited search        | No explicit game tree traversal        |
| Q-Learning     | Q(s,a) lookup -> argmax     | Optimizes embeddings, not table lookup |

## Related Work

EBMO builds on several established research areas:

### Structured Prediction Energy Networks (SPENs)

The core inference approach--optimizing over continuous relaxations of structured outputs, then rounding--follows the SPEN pattern introduced by Belanger & McCallum (2016). EBMO applies this to game move selection.

### Action Embeddings for Large Discrete Spaces

The "continuous embedding -> nearest discrete action" projection is similar to Wolpertinger-style architectures (Dulac-Arnold et al., 2015) used for RL with large discrete action spaces.

### Energy-Based Policies

Energy-based policy formulations where action selection is cast as inference (sampling/optimization from a Boltzmann distribution) are well-established in RL (Haarnoja et al., 2017; Levine, 2018).

## What EBMO Provides

Rather than claiming fundamental novelty, EBMO offers:

1. **Application to board game move selection**: SPEN-style inference applied to legal move selection with game-specific projections
2. **Temperature-weighted soft projection**: Differentiable projection to legal moves using softmax over embedding distances
3. **Game outcome-weighted training**: Contrastive loss weighted by game outcomes for better credit assignment
4. **Practical implementation**: A working, tested codebase for energy-based game AI

## Algorithm

```
Standard Policy:  state -> NN -> logits -> softmax -> move
EBMO:            state -> optimize(action_embedding) via gradE -> project -> move
```

### Inference Steps

1. **Encode state** once using CNN backbone
2. **Multi-restart optimization** (default: 5 restarts):
   - Initialize action embedding from random legal move
   - Run gradient descent (50 steps): `a' = a - lr * grad_a E(s, a)`
   - Periodically project to legal move manifold via temperature-weighted soft projection
3. **Select best move** with lowest final energy

### Projection Method

The projection to legal moves uses a **temperature-weighted soft projection**:

```python
# Not simple nearest-neighbor, but soft weighted combination
distances = torch.cdist(action_embed, legal_embeddings)
weights = F.softmax(-distances / temperature, dim=-1)
projected = weights @ legal_embeddings
```

This allows gradients to flow through the projection and provides smoother optimization dynamics than hard kNN projection.

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

### Expert Data Generation

```bash
# Generate training data from strong AI play
python scripts/generate_ebmo_expert_data.py \
  --num-games 500 \
  --engine heuristic \
  --depth 5 \
  --output data/training/ebmo_expert.npz
```

### Expert Data Training

```bash
# Train directly on the expert NPZ format
python scripts/train_ebmo_expert.py \
  --data data/training/ebmo_expert.npz \
  --epochs 50 \
  --batch-size 256 \
  --output-dir models/ebmo
```

## Architecture

### Network Components

```
+-------------------------------------------------------------+
|                    EBMO Network                              |
+-------------------------------------------------------------+
|                                                              |
|  State Encoder (CNN):                                        |
|  +----------+   +----------+   +----------+                |
|  | 56x8x8   | -> | ResBlocks| -> | 256-dim  | = s_embed      |
|  | features |   | (6 SE)   |   | state    |                |
|  +----------+   +----------+   +----------+                |
|                                                              |
|  Action Encoder:                                             |
|  +----------+   +----------+   +----------+                |
|  | 14-dim   | -> | MLP      | -> | 128-dim  | = a_embed      |
|  | features |   | (3 layer)|   | action   |                |
|  +----------+   +----------+   +----------+                |
|                                                              |
|  Energy Head:                                                |
|  +----------------+   +----------+   +--------+            |
|  | concat(s,a)    | -> | MLP      | -> | scalar | = E(s,a)   |
|  | 384-dim        |   | (3 layer)|   | energy |            |
|  +----------------+   +----------+   +--------+            |
|                                                              |
+-------------------------------------------------------------+
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

    # State encoder
    num_input_channels: int = 56
    num_global_features: int = 20
    num_residual_blocks: int = 6
    residual_filters: int = 128

    # Action encoder
    action_feature_dim: int = 14
    action_hidden_dim: int = 64

    # Inference optimization
    optim_steps: int = 100       # Gradient descent steps
    optim_lr: float = 0.1        # Action embedding learning rate
    num_restarts: int = 8        # Multi-restart count
    projection_temperature: float = 0.3
    project_every_n_steps: int = 10
    use_direct_eval: bool = True  # Skip gradient descent, score legal moves
    skip_penalty: float = 5.0     # Penalize skip/pass moves when alternatives exist

    # Training
    contrastive_temperature: float = 0.1
    num_negatives: int = 15
    outcome_weight: float = 0.5  # Balance contrastive vs outcome loss
    learning_rate: float = 0.001

    # Board
    board_size: int = 8
    board_type: BoardType = BoardType.SQUARE8
```

**Direct evaluation mode:** When `use_direct_eval=True`, EBMO skips gradient descent
and directly scores every legal move, picking the lowest-energy option. Set it to
`False` if you want the original gradient-descent projection pipeline. `skip_penalty`
adds an energy offset to skip/pass moves when non-skip alternatives exist.

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

| Opponent            | EBMO Win Rate    | Notes                |
| ------------------- | ---------------- | -------------------- |
| RandomAI            | **100%** (15/15) | Self-play trained    |
| HeuristicAI         | **20%** (2/10)   | Expert-trained model |
| MinimaxAI (depth=2) | 0% (0/10)        | Needs more training  |

**Status**: Early prototype. Expert training on HeuristicAI games shows improvement (0% -> 20% vs HeuristicAI). More training data and hyperparameter tuning needed.

### Inference Speed

- ~148ms per move on GH200 GPU
- ~113ms per move on Apple M-series (MPS)
- Scales with `optim_steps x num_restarts`

## Files

| File                                   | Description                          |
| -------------------------------------- | ------------------------------------ |
| `app/ai/ebmo_network.py`               | Network architecture, loss functions |
| `app/ai/ebmo_ai.py`                    | AI agent implementation              |
| `app/training/ebmo_dataset.py`         | Dataset with contrastive sampling    |
| `app/training/ebmo_trainer.py`         | Training loop                        |
| `scripts/train_ebmo.py`                | Training script                      |
| `scripts/generate_ebmo_expert_data.py` | Expert data generation               |

## Potential Advantages

1. **Continuous exploration**: Gradient descent can explore between discrete options
2. **Gradient guidance**: More directed than random sampling (e.g., MCTS rollouts)
3. **Amortized search**: Energy landscape encodes search knowledge in parameters
4. **Differentiable projection**: Soft projection allows end-to-end training

## Known Limitations

1. **Local minima**: Gradient descent may get stuck (mitigated by restarts)
2. **Projection quality**: Continuous -> discrete mapping can lose information
3. **Training stability**: Energy-based models can be sensitive to hyperparameters
4. **Compute cost**: Multiple optimization steps per move vs. single forward pass

## References

### Core Methods

- **SPENs**: Belanger & McCallum, "Structured Prediction Energy Networks" (2016)
- **EBMs**: LeCun et al., "A Tutorial on Energy-Based Learning" (2006)

### Action Embeddings

- **Wolpertinger**: Dulac-Arnold et al., "Deep Reinforcement Learning in Large Discrete Action Spaces" (2015)

### Energy-Based RL

- **Soft Actor-Critic**: Haarnoja et al., "Soft Actor-Critic" (2018)
- **MaxEnt RL**: Levine, "Reinforcement Learning and Control as Probabilistic Inference" (2018)

### Game AI

- **AlphaZero**: Silver et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go" (2018)
- **Gumbel MuZero**: Danihelka et al., "Policy improvement by planning with Gumbel" (2022)

### Contrastive Learning

- **InfoNCE**: van den Oord et al., "Representation Learning with Contrastive Predictive Coding" (2018)
