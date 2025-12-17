# Gumbel MCTS Implementation

> **Last Updated**: 2025-12-17
> **Status**: Active
> **Location**: `app/ai/gumbel_mcts_ai.py`

This document describes the Gumbel MCTS (Monte Carlo Tree Search) implementation used for efficient move selection and training data generation.

## Table of Contents

1. [Overview](#overview)
2. [Algorithm](#algorithm)
3. [Configuration](#configuration)
4. [Visit Distribution for Training](#visit-distribution-for-training)
5. [API Reference](#api-reference)
6. [Usage Examples](#usage-examples)

---

## Overview

Gumbel MCTS is an efficient variant of MCTS that uses Gumbel-Top-K sampling with Sequential Halving to select moves. It provides:

- **Efficient search**: Requires fewer simulations than standard MCTS
- **Policy improvement**: Generates soft policy targets from visit distributions
- **Exploration**: Gumbel noise ensures diverse action sampling

### Key Features

| Feature               | Description                                             |
| --------------------- | ------------------------------------------------------- |
| Gumbel-Top-K Sampling | Sample K actions without replacement using Gumbel noise |
| Sequential Halving    | Progressively eliminate low-value actions               |
| Visit Distribution    | Extract normalized visit counts for training            |
| Completed Q-Values    | Handle visit count asymmetry in Q estimates             |

---

## Algorithm

### Gumbel-Top-K Sampling

Instead of selecting actions greedily, Gumbel MCTS samples K actions using:

```
score(a) = log(prior(a)) + gumbel_noise(a)
selected_actions = top_k(scores, k=m)
```

This ensures diverse exploration while respecting the policy prior.

### Sequential Halving

Actions are evaluated in phases, with the worst half eliminated each round:

```
Phase 1: Evaluate m actions with B/(m * ceil(log2(m))) simulations each
Phase 2: Keep top m/2, evaluate with 2x simulations
Phase 3: Keep top m/4, evaluate with 4x simulations
...
Final: Return action with highest Q-value
```

### Completed Q-Values

To handle asymmetric visit counts, Q-values are "completed":

```python
def completed_q(action, c_visit=50.0):
    if action.visit_count == 0:
        return prior_value

    raw_q = action.total_value / action.visit_count
    # Mix with prior based on visit count
    weight = action.visit_count / (action.visit_count + c_visit)
    return weight * raw_q + (1 - weight) * prior_value
```

---

## Configuration

### AIConfig Parameters

```python
from app.models.core import AIConfig, AIType

config = AIConfig(
    ai_type=AIType.GUMBEL_MCTS,
    difficulty=7,
    gumbel_num_sampled_actions=16,   # Number of actions to sample (m)
    gumbel_simulation_budget=100,     # Total simulation budget (B)
    use_neural_net=True,              # Required for Gumbel MCTS
    nn_model_id="ringrift_hex8_2p_v3_retrained",  # Optional specific model
)
```

### Parameters

| Parameter                    | Default | Description                           |
| ---------------------------- | ------- | ------------------------------------- |
| `gumbel_num_sampled_actions` | 16      | Number of actions to sample initially |
| `gumbel_simulation_budget`   | 100     | Total simulations to allocate         |
| `use_neural_net`             | True    | Must be True for Gumbel MCTS          |
| `nn_model_id`                | None    | Specific neural network model to use  |

### Internal Constants

| Constant    | Value | Description                    |
| ----------- | ----- | ------------------------------ |
| `c_puct`    | 1.5   | PUCT exploration constant      |
| `c_visit`   | 50.0  | Visit count mixing coefficient |
| `max_depth` | 10    | Maximum tree traversal depth   |

---

## Visit Distribution for Training

Gumbel MCTS can extract visit distributions for use as soft policy training targets.

### Why Soft Targets?

Traditional selfplay stores only the played move (one-hot encoding). Soft targets from visit distributions provide:

- **Richer signal**: Captures relative action quality
- **Better exploration**: Non-zero probability for multiple good moves
- **Improved learning**: Reduces overconfidence

### Extracting Visit Distribution

```python
from app.ai.gumbel_mcts_ai import GumbelMCTSAI

# Initialize AI
gumbel_ai = GumbelMCTSAI(player_number=1, config=config, board_type=board_type)

# Select move (stores internal state)
move = gumbel_ai.select_move(game_state)

# Extract visit distribution
moves, probs = gumbel_ai.get_visit_distribution()
# moves: List[Move] - all visited moves
# probs: List[float] - normalized visit probabilities (sum to 1.0)
```

### Using in Selfplay

The `run_hybrid_selfplay.py` script automatically captures visit distributions:

```bash
python scripts/run_hybrid_selfplay.py \
  --board-type hex8 \
  --engine-mode gumbel-mcts \
  --nn-model-id ringrift_hex8_2p_v3_retrained \
  --num-games 100
```

The captured `mcts_policy_dist` is stored in move records and can be used for training.

---

## API Reference

### GumbelMCTSAI Class

```python
class GumbelMCTSAI:
    """Gumbel MCTS AI using Sequential Halving for efficient search."""

    def __init__(
        self,
        player_number: int,
        config: AIConfig,
        board_type: BoardType = BoardType.SQUARE8,
    ):
        """
        Initialize Gumbel MCTS AI.

        Args:
            player_number: Player number (1-4)
            config: AI configuration with gumbel parameters
            board_type: Board geometry type
        """

    def select_move(self, game_state: GameState) -> Optional[Move]:
        """
        Select the best move using Gumbel MCTS.

        Args:
            game_state: Current game state

        Returns:
            Best move, or None if no valid moves
        """

    def get_visit_distribution(self) -> Tuple[List[Move], List[float]]:
        """
        Extract normalized visit count distribution from last search.

        Returns:
            Tuple of (moves, visit_probabilities) where probabilities sum to 1.0.
            Returns ([], []) if no search has been performed.
        """
```

### GumbelAction Dataclass

```python
@dataclass
class GumbelAction:
    """Represents an action being considered by Gumbel MCTS."""

    move: Move              # The actual move
    prior: float           # Policy prior probability
    gumbel: float          # Sampled Gumbel noise
    visit_count: int = 0   # Number of times visited
    total_value: float = 0.0  # Sum of backed-up values

    def completed_q(self, c_visit: float = 50.0) -> float:
        """Compute completed Q-value accounting for visit asymmetry."""
```

---

## Usage Examples

### Basic Usage

```python
from app.ai.gumbel_mcts_ai import GumbelMCTSAI
from app.models.core import AIConfig, AIType
from app.models import BoardType

# Configure
config = AIConfig(
    ai_type=AIType.GUMBEL_MCTS,
    difficulty=7,
    gumbel_num_sampled_actions=16,
    gumbel_simulation_budget=150,
    use_neural_net=True,
)

# Create AI
ai = GumbelMCTSAI(player_number=1, config=config, board_type=BoardType.HEX8)

# Play game
while not game_state.is_terminal():
    ai.player_number = game_state.current_player
    move = ai.select_move(game_state)
    game_state = game_state.apply_move(move)
```

### Generating Training Data

```python
# Play with policy distribution capture
training_data = []

for game_idx in range(num_games):
    game_state = create_initial_state()

    while not game_state.is_terminal():
        ai.player_number = game_state.current_player
        move = ai.select_move(game_state)

        # Capture soft policy target
        moves, probs = ai.get_visit_distribution()

        training_data.append({
            'state': encode_state(game_state),
            'policy_moves': moves,
            'policy_probs': probs,
            'value': None,  # Fill in after game ends
        })

        game_state = game_state.apply_move(move)

    # Backfill values based on game outcome
    fill_values(training_data, game_state.winner)
```

### Selfplay with Gumbel MCTS

```bash
# Generate gumbel-mcts games with specific model
python scripts/run_hybrid_selfplay.py \
  --board-type hex8 \
  --num-games 200 \
  --engine-mode gumbel-mcts \
  --nn-model-id ringrift_hex8_2p_v3_retrained \
  --output-dir data/selfplay/hex8_gumbel
```

---

## Comparison with Standard MCTS

| Aspect                | Standard MCTS       | Gumbel MCTS          |
| --------------------- | ------------------- | -------------------- |
| Action Selection      | UCB/PUCT            | Gumbel-Top-K         |
| Simulation Allocation | Uniform             | Sequential Halving   |
| Efficiency            | O(n) per simulation | O(n \* log(k)) total |
| Exploration           | UCB-based           | Gumbel noise         |
| Policy Targets        | Visit counts        | Visit distributions  |

### When to Use Gumbel MCTS

- **Training data generation**: Soft policy targets improve learning
- **Limited compute**: More efficient than standard MCTS
- **Diverse games**: Gumbel noise provides natural exploration
- **Neural network guidance**: Leverages strong policy prior

---

## See Also

- [NEURAL_AI_ARCHITECTURE.md](NEURAL_AI_ARCHITECTURE.md) - Neural network architectures
- [TRAINING_FEATURES.md](TRAINING_FEATURES.md) - Training configuration
- [ai_types_implementation.md](ai_types_implementation.md) - All AI types overview
