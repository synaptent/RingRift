# EBMO (Energy-Based Move Optimization) - Results & Learnings

## Executive Summary

EBMO is a novel AI algorithm that uses **gradient descent on action embeddings** at inference time to select moves. Unlike traditional policy networks or MCTS, EBMO treats move selection as a continuous optimization problem.

**Current Status:** Functional prototype that beats Random AI but loses to Heuristic AI.

## Benchmark Results (December 2025)

| Matchup                  | Result | Win Rate | Assessment          |
| ------------------------ | ------ | -------- | ------------------- |
| EBMO vs Random           | 14-6   | **70%**  | Solid baseline      |
| EBMO vs Heuristic        | 7-13   | **35%**  | Needs improvement   |
| EBMO (earlier) vs Random | 8-2    | **80%**  | Variance in testing |

### Ladder Position

```
D11: Gumbel MCTS (Ultimate)     ← Production ceiling
D9-10: Gumbel MCTS              ← Production upper tier
D6-8: MCTS variants             ← Production mid tier
D4-5: Minimax/BRS               ← Production lower tier
D3: PolicyOnly (65% vs Random)  ← Current D3 threshold
D2: Heuristic                   ← EBMO loses here (35%)
    ↑ EBMO sits here (70% vs Random, 35% vs Heuristic)
D1: Random                      ← EBMO beats this (70%)
```

**Key Finding:** EBMO is between D1 and D2 in strength. It needs to beat Heuristic (D2) before it can replace PolicyOnly at D3.

## Technical Implementation

### Architecture

```
State Encoder:  14×8×8 features → 6 SE-ResBlocks → 256-dim embedding
Action Encoder: Move features → 3-layer MLP → 128-dim embedding
Energy Head:    concat(state, action) → 3-layer MLP → scalar energy

Inference: Gradient descent on action embedding → project to legal move
```

### Training Pipeline

1. **Data Generation:** MCTS-labeled positions (150,600 samples, 56-channel)
2. **Loss Function:** Contrastive energy loss + quality-weighted margin
3. **Optimization:** Adam with learning rate 1e-3, batch size 64
4. **Validation:** 99.75% accuracy on held-out set

### Online Learning (Implemented)

```python
from app.ai.ebmo_online import EBMOOnlineAI, EBMOOnlineConfig

config = EBMOOnlineConfig(
    buffer_size=20,      # Rolling buffer of recent games
    learning_rate=1e-5,  # Conservative for stability
    td_lambda=0.9,       # Eligibility trace decay
    gamma=0.99,          # Discount factor
)

ai = EBMOOnlineAI(player_number=1, config=ai_config,
                   model_path="models/ebmo_56ch/ebmo_quality_best.pt",
                   online_config=config)

# Play games normally
move = ai.select_move(state)

# After each game, trigger learning update
metrics = ai.end_game(winner)
# Returns: {'total_loss': 264.1, 'td_loss': 156.5, 'outcome_loss': 371.7, ...}
```

## Key Files

| File                                    | Purpose                          |
| --------------------------------------- | -------------------------------- |
| `app/ai/ebmo_network.py`                | EBMO neural network architecture |
| `app/ai/ebmo_ai.py`                     | EBMO AI agent implementation     |
| `app/ai/ebmo_online.py`                 | Online/continuous learning       |
| `models/ebmo_56ch/ebmo_quality_best.pt` | Trained 56-channel model         |
| `scripts/eval_ebmo_56ch.py`             | Evaluation script                |
| `scripts/benchmark_ebmo_ladder.py`      | Ladder benchmark script          |

## Learnings

### What Works

1. **Energy-based formulation:** The concept of low energy = good move is sound
2. **Gradient descent optimization:** Finding moves via continuous optimization works
3. **56-channel features:** Stacking 4 frames of history improves temporal reasoning
4. **Online learning:** TD-Energy updates decrease loss over games
5. **Beats Random:** 70-80% win rate is a solid baseline

### What Doesn't Work (Yet)

1. **Heuristic gap:** EBMO loses to simple rule-based Heuristic AI
2. **Training data quality:** MCTS labels may not teach strategic principles
3. **Optimization depth:** 100 steps may be insufficient for complex positions
4. **Projection accuracy:** Mapping continuous embeddings to discrete moves loses information

### Why EBMO Loses to Heuristic

The Heuristic AI uses explicit game knowledge:

- Territory control priorities
- Ring safety evaluation
- Capture opportunity detection
- Strategic positioning rules

EBMO learns only from MCTS visit counts, which may not capture these strategic principles. The energy function learns "what MCTS would do" rather than "what wins games."

## Recommended Next Steps

### Priority 1: Learn from Heuristic (High Impact)

Train EBMO on games where Heuristic wins, using contrastive loss:

- Winner's moves → low energy
- Loser's moves → high energy

This teaches EBMO the strategic patterns that make Heuristic effective.

### Priority 2: Curriculum Learning (Medium Impact)

Progressive difficulty:

1. Beat Random consistently (done: 70%)
2. Beat Heuristic (target: 60%+)
3. Match PolicyOnly (target: 50%+)
4. Challenge MCTS (aspirational)

### Priority 3: Hybrid EBMO+MCTS (Experimental)

Use EBMO for move pruning, MCTS for final selection:

```python
# EBMO proposes top-k candidate moves
candidates = ebmo.get_top_k_moves(state, k=10)

# MCTS evaluates candidates only
best_move = mcts.search(state, restricted_to=candidates)
```

### Priority 4: Architecture Improvements

- Deeper energy head (more layers)
- Attention over legal moves
- Learned projection (instead of nearest-neighbor)
- Position-aware action encoding

## Configuration Reference

### EBMOConfig (Inference)

```python
@dataclass
class EBMOConfig:
    state_embed_dim: int = 256
    action_embed_dim: int = 128
    energy_hidden_dim: int = 256
    num_energy_layers: int = 3
    optim_steps: int = 100      # Gradient descent steps
    optim_lr: float = 0.1       # Step size
    num_restarts: int = 8       # Random restarts
    projection_temp: float = 0.3 # Softmax temperature for projection
```

### EBMOOnlineConfig (Learning)

```python
@dataclass
class EBMOOnlineConfig:
    buffer_size: int = 20       # Games in rolling buffer
    learning_rate: float = 1e-5 # Very conservative
    td_lambda: float = 0.9      # Eligibility traces
    gamma: float = 0.99         # Discount factor
    batch_size: int = 8         # Games per update
    gradient_clip: float = 1.0  # Gradient norm clipping
```

## Difficulty Ladder Integration

EBMO is available at experimental difficulty levels:

| Level | AI Type     | Profile ID                      |
| ----- | ----------- | ------------------------------- |
| D12   | EBMO        | `v3-ebmo-12-experimental`       |
| D13   | GMO         | `v3-gmo-13-experimental`        |
| D14   | IG-GMO      | `v3-iggmo-14-experimental`      |
| D15   | GPU Minimax | `v3-gpuminimax-15-experimental` |

Enable EBMO at D3 via environment variable:

```bash
export RINGRIFT_USE_EBMO=1
```

## Conclusion

EBMO demonstrates that **gradient-based move optimization is viable** for game AI. The current implementation beats Random (70%) but loses to Heuristic (35%), placing it between D1 and D2 on the difficulty ladder.

The key insight is that EBMO's training data (MCTS labels) teaches "what MCTS would do" rather than "what wins games." Future work should focus on **outcome-weighted training** where the model learns directly from game winners, not search algorithms.

The online learning infrastructure is in place and functional, providing a path toward continuous improvement during gameplay.
