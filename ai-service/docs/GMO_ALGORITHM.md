# Gradient Move Optimization (GMO) Algorithm

**A Synthesis of Established Techniques for RingRift Game AI**

GMO combines several well-known ML/RL techniques into a practical move-selection pipeline for RingRift: gradient-based inference in embedding space, action embeddings with nearest-neighbor projection, MC Dropout uncertainty, and UCB-style exploration bonuses.

> **Note on Novelty**: GMO is **not** a fundamentally novel algorithm. It is a synthesis of established techniques applied to a specific game domain. This document honestly attributes prior art.

---

## Table of Contents

1. [Overview](#overview)
2. [Prior Art & Lineage](#prior-art--lineage)
3. [What GMO Actually Is](#what-gmo-actually-is)
4. [Architecture](#architecture)
5. [Algorithm Details](#algorithm-details)
6. [Exploration Components](#exploration-components)
7. [Configuration Reference](#configuration-reference)
8. [Training](#training)
9. [Usage](#usage)
10. [Evaluation Results](#evaluation-results)
11. [Comparison to Other Approaches](#comparison-to-other-approaches)
12. [Potential Novel Extensions](#potential-novel-extensions)
13. [IG-GMO (Experimental)](#ig-gmo-experimental)
14. [File Reference](#file-reference)
15. [Training Infrastructure Integration](#training-infrastructure-integration)
16. [Cluster Smoke Run (pending)](#cluster-smoke-run-pending)
17. [References](#references)

---

## Overview

GMO (Gradient Move Optimization) is a neural network-based game AI that uses:

1. **Gradient ascent on action embeddings at inference time** - optimize in continuous space
2. **Nearest-neighbor projection back to discrete actions** - map to legal moves
3. **UCB-style exploration bonus** - value + uncertainty term
4. **Novelty bonus** - encourage diverse play

| Component             | GMO Implementation                     |
| --------------------- | -------------------------------------- |
| Action representation | Continuous embeddings (128-dim)        |
| Action selection      | Gradient optimization + projection     |
| Exploration           | MC Dropout variance + novelty distance |
| Objective             | value + beta*sqrt(var) + gamma*novelty |

---

## Prior Art & Lineage

GMO draws directly from several established lines of research. **None of the core components are novel.**

### 1. Gradient-Based Inference on Output/Action Embeddings

**Prior Art**: Energy-Based Models (EBMs) and Structured Prediction Energy Networks (SPENs)

The pattern of "define a differentiable scoring function over outputs, then do iterative gradient-based inference over a relaxed/continuous output representation" is the core idea in SPENs:

> "SPENs produce predictions by backpropagation to iteratively optimize the score/energy w.r.t. outputs, using a continuous relaxation and (optionally) rounding back to discrete."
> -- Belanger & McCallum (2016)

**More recent**: Test-time embedding optimization appears in offline RL work like DROP, which does "gradient ascent at testing time to infer the optimal embedding z\*".

**Very close parallel**: A 2025 arXiv paper proposes "gradient ascent on text embeddings" to optimize outcomes, then a "decoding strategy" to translate optimized embeddings back into discrete natural-language actions.

**GMO's relationship**: Direct application of this pattern. Not novel.

### 2. Action Embeddings + Nearest-Neighbor Projection

**Prior Art**: Dulac-Arnold et al. (2015) "Deep Reinforcement Learning in Large Discrete Action Spaces"

> "Embed discrete actions into a continuous space and use approximate nearest neighbors for lookup."

**Also**: Chandak et al. (2019) summarizes this approach: using continuous actions and selecting the nearest discrete action.

**GMO's relationship**: The projection step ("nearest legal move by cosine similarity") is exactly this standard pattern.

### 3. MC Dropout for Uncertainty

**Prior Art**: Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"

Monte Carlo dropout as an approximate Bayesian uncertainty estimate is a classic technique. Using dropout-derived uncertainty in decision-making (including RL contexts) is well-established.

**GMO's relationship**: Standard application of MC Dropout.

### 4. UCB-Style Exploration Bonus

**Prior Art**: Auer et al. (2002) "Finite-time Analysis of the Multiarmed Bandit Problem"

The formula `value + c * uncertainty` is the canonical UCB exploration principle from bandits, with many RL analogues.

**GMO's relationship**: `score = E[V] + beta*sqrt(Var)` is UCB with dropout variance. Not novel.

### 5. Novelty Bonus

**Prior Art**: Lehman & Stanley (2011) "Abandoning Objectives: Evolution Through the Search for Novelty Alone"

Novelty search--rewarding behavioral novelty / distance in an embedding space--is an established exploration paradigm.

**GMO's relationship**: The novelty tracker is exactly a nearest-neighbor novelty score in embedding space. Standard technique.

### Summary: GMO Component Lineage

| GMO Component                                     | Closest Prior Art                                 | Novel? |
| ------------------------------------------------- | ------------------------------------------------- | ------ |
| Gradient ascent on move embedding at inference    | SPENs, EBMs, test-time embedding optimization     | No     |
| Project optimized embedding to nearest legal move | Action embeddings + NN lookup (Dulac-Arnold 2015) | No     |
| Explore using dropout variance                    | MC Dropout (Gal & Ghahramani 2016)                | No     |
| "value + beta\*uncertainty"                       | UCB (Auer et al. 2002)                            | No     |
| Novelty memory bonus                              | Novelty search (Lehman & Stanley 2011)            | No     |
| Two-phase pipeline (rank then optimize top-K)     | Engineering choice                                | No     |

---

## What GMO Actually Is

GMO is best described as:

> **An engineered synthesis** of gradient-based inference, action embeddings, uncertainty estimation, and exploration bonuses, packaged into a two-phase move-selection pipeline for RingRift.

### What GMO Is NOT

- **Not** a fundamentally new algorithmic principle
- **Not** a novel form of exploration (it uses standard UCB + novelty)
- **Not** a new way to handle uncertainty (it uses standard MC Dropout)
- **Not** a new action representation (action embeddings are established)

### What GMO Might Be

- A **practical synthesis** that works well for RingRift
- A **demonstration** that gradient-based action inference can work in board games
- A **reference implementation** combining these techniques

### Honest Framing

> "We apply gradient-based inference over action embeddings to board game move selection, combining established techniques from structured prediction (SPENs), action embeddings (Dulac-Arnold et al.), and uncertainty-driven exploration (MC Dropout, UCB). We demonstrate this synthesis on RingRift."

---

## Architecture

### System Diagram

```
+-----------------------------------------------------------------------------+
|                           GMO Architecture                                   |
+-----------------------------------------------------------------------------+
|                                                                              |
|  +----------------+         +----------------+                              |
|  |   GameState    |         |  Legal Moves   |                              |
|  |   (RingRift)   |         |   [m1..mN]     |                              |
|  +-------+--------+         +-------+--------+                              |
|          |                          |                                        |
|          v                          v                                        |
|  +----------------+         +----------------+                              |
|  |  StateEncoder  |         |  MoveEncoder   |                              |
|  |  768->256->128 |         |  112->128->128 |                              |
|  +-------+--------+         +-------+--------+                              |
|          |                          |                                        |
|          v                          v                                        |
|     state_embed              move_embeds                                     |
|      (128-dim)              (N x 128-dim)                                   |
|          |                          |                                        |
|          +----------+---------------+                                        |
|                     |                                                        |
|                     v                                                        |
|  +---------------------------------------------------------------------+   |
|  |              Phase 1: Initial Ranking (standard UCB)                 |   |
|  |                                                                      |   |
|  |  For each move m_i:                                                 |   |
|  |    1. MC Dropout -> mean_value, variance                            |   |
|  |    2. NoveltyTracker -> novelty_score                               |   |
|  |    3. score = value + beta*sqrt(var) + gamma*novelty                |   |
|  |                                                                      |   |
|  |  Select top-K candidates by score                                   |   |
|  +---------------------------------------------------------------------+   |
|                     |                                                        |
|                     v                                                        |
|  +---------------------------------------------------------------------+   |
|  |         Phase 2: Gradient Optimization (SPEN-style inference)        |   |
|  |                                                                      |   |
|  |  For each top-K candidate:                                          |   |
|  |    1. Initialize: m_opt = m_candidate.clone()                       |   |
|  |    2. For step in 1..10:                                            |   |
|  |         objective = value(s, m_opt) + exploration_bonus             |   |
|  |         m_opt += lr * grad_m objective                              |   |
|  |    3. Project: nearest legal move by cosine similarity              |   |
|  |    4. Evaluate final move                                           |   |
|  +---------------------------------------------------------------------+   |
|                     |                                                        |
|                     v                                                        |
|              +-------------+                                                 |
|              |  Best Move  |                                                 |
|              +-------------+                                                 |
|                                                                              |
+-----------------------------------------------------------------------------+
```

### Component Details

#### 1. StateEncoder (`gmo_ai.py:179-253`)

Encodes a RingRift GameState into a 128-dimensional embedding.

**Features (12 planes x 64 positions = 768 features):**

- Planes 0-3: Ring presence per player
- Planes 4-7: Stack control per player
- Planes 8-11: Territory ownership per player

**Architecture:**

```
Input: 768 features (flattened board representation)
  |
Linear(768 -> 256) + ReLU
  |
Linear(256 -> 128)
  |
Output: 128-dim state embedding
```

#### 2. MoveEncoder (`gmo_ai.py:95-172`)

Encodes a Move object into a 128-dimensional embedding.

**Components:**

- `type_embed`: 8 move types -> 32-dim
- `from_embed`: 65 positions -> 32-dim (64 board positions + None)
- `to_embed`: 65 positions -> 32-dim
- `placement_embed`: 4 values -> 16-dim

**Architecture:**

```
Move -> [type_idx, from_idx, to_idx, placement_count]
  |
[type_embed || from_embed || to_embed || placement_embed] (112 dim)
  |
Linear(112 -> 128) + ReLU -> Linear(128 -> 128)
  |
Output: 128-dim move embedding
```

#### 3. GMOValueNetWithUncertainty (`gmo_ai.py:260-326`)

Joint network that predicts value and learned uncertainty.

**Architecture:**

```
Input: concat(state_embed, move_embed) = 256-dim
  |
Linear(256 -> 256) + ReLU + Dropout(0.1)
  |
Linear(256 -> 256) + ReLU + Dropout(0.1)
  |
  +------------------+------------------+
  |                  |
value_head      uncertainty_head
Linear(256->1)    Linear(256->1)
  |                  |
tanh               raw
  |                  |
value in [-1,1]   log_variance
```

#### 4. NoveltyTracker (`gmo_ai.py:333-381`)

Standard nearest-neighbor novelty scorer (per Lehman & Stanley 2011).

- Ring buffer storing last 1000 move embeddings
- Novelty = minimum L2 distance to any stored embedding
- Reset at start of each game

---

## Algorithm Details

### Main Algorithm (`select_move` at `gmo_ai.py:597-712`)

```python
def select_move(game_state):
    # 1. Get all legal moves
    legal_moves = get_valid_moves(game_state)

    # 2. Encode state and moves
    state_embed = state_encoder.encode(game_state)
    move_embeds = [move_encoder.encode(m) for m in legal_moves]

    # 3. Phase 1: UCB-style ranking (standard technique)
    candidates = []
    for i, move_embed in enumerate(move_embeds):
        mean_value, variance = mc_dropout_estimate(state_embed, move_embed)
        novelty = novelty_tracker.compute_novelty(move_embed)
        score = mean_value + beta * sqrt(variance) + gamma * novelty
        candidates.append((i, score, move_embed))

    top_k = sorted(candidates, reverse=True)[:3]

    # 4. Phase 2: SPEN-style gradient optimization
    best_move, best_score = None, -inf

    for idx, _, initial_embed in top_k:
        # Gradient ascent in embedding space
        optimized_embed = gradient_optimize(state_embed, initial_embed)

        # Project to nearest legal move (Dulac-Arnold style)
        nearest_idx = argmax([cosine_sim(optimized_embed, me) for me in move_embeds])

        final_score = evaluate(state_embed, move_embeds[nearest_idx])
        if final_score > best_score:
            best_score = final_score
            best_move = legal_moves[nearest_idx]

    novelty_tracker.add(embed(best_move))
    return best_move
```

### Gradient Optimization

```python
def gradient_optimize(state_embed, initial_move_embed):
    move_embed = initial_move_embed.clone().requires_grad_(True)
    optimizer = Adam([move_embed], lr=0.1)

    for step in range(optim_steps):
        optimizer.zero_grad()
        mean_value, variance = mc_dropout_estimate(state_embed, move_embed)

        # Anneal exploration (engineering choice, not novel)
        exploration_weight = beta * (1 - step / max(optim_steps - 1, 1)) * temperature
        objective = mean_value + exploration_weight * sqrt(variance)

        loss = -objective
        loss.backward()
        optimizer.step()

    return move_embed.detach()
```

---

## Exploration Components

### 1. MC Dropout Uncertainty (Gal & Ghahramani 2016)

Standard technique: keep dropout enabled during inference, run multiple forward passes, measure variance.

```python
def mc_dropout_estimate(state_embed, move_embed, n_samples=10):
    value_net.train()  # Enable dropout
    values = [value_net(state_embed, move_embed) for _ in range(n_samples)]
    return mean(values), var(values)
```

### 2. UCB-Style Bonus (Auer et al. 2002)

Standard technique: add uncertainty bonus to value.

```
score = E[value] + beta * sqrt(variance)
```

### 3. Novelty Bonus (Lehman & Stanley 2011)

Standard technique: reward distance from previously seen states/actions.

```
novelty = min_distance(current_embed, memory_buffer)
score += gamma * novelty
```

---

## Configuration Reference

```python
@dataclass
class GMOConfig:
    # Embedding dimensions
    state_dim: int = 128
    move_dim: int = 128
    hidden_dim: int = 256

    # Optimization parameters
    top_k: int = 3           # Candidates to optimize
    optim_steps: int = 2     # Gradient steps per candidate
    lr: float = 0.1          # Optimization learning rate

    # Exploration parameters (UCB-style)
    beta: float = 0.1        # Uncertainty coefficient
    gamma: float = 0.05      # Novelty coefficient
    exploration_temp: float = 1.0

    # MC Dropout parameters
    dropout_rate: float = 0.1
    mc_samples: int = 10

    # Novelty tracking
    novelty_memory_size: int = 1000

    device: str = "cpu"
```

### Parameter Tuning Guide

| Parameter     | Low Value Effect            | High Value Effect           | Range      |
| ------------- | --------------------------- | --------------------------- | ---------- |
| `beta`        | Exploit known good moves    | Explore uncertain moves     | 0.05 - 0.3 |
| `gamma`       | Allow repetitive patterns   | Force diverse play          | 0.01 - 0.1 |
| `top_k`       | Faster, may miss good moves | Thorough, slower            | 2 - 7      |
| `optim_steps` | Quick, less refined         | Better optimization, slower | 1 - 10     |

---

## Training

### Data Format

GMO trains on game records in JSONL format:

```json
{
  "initial_state": {
    /* GameState */
  },
  "moves": [
    /* Move objects */
  ],
  "winner": 1,
  "board_type": "square8"
}
```

### Loss Function

Negative log-likelihood with learned uncertainty (Nix & Weigend 1994):

```
L = exp(-log_var) * (pred - target)^2 + log_var
```

### Training Command

```bash
python -m app.training.train_gmo \
    --data-path data/gumbel_selfplay/sq8_gumbel_kl_canonical.jsonl \
    --output-dir models/gmo \
    --epochs 50 \
    --batch-size 64 \
    --device mps
```

---

## Usage

### Creating a GMO AI

```python
from app.ai.factory import AIFactory
from app.ai.gmo_ai import GMOAI, GMOConfig
from app.models import AIConfig, AIType
from pathlib import Path

# Via factory
ai = AIFactory.create(AIType.GMO, player_number=1, config=AIConfig(difficulty=6))

# Load trained weights
ai.load_checkpoint(Path("models/gmo/gmo_best.pt"))
```

### Evaluation

```python
from app.training.train_gmo import evaluate_vs_random

results = evaluate_vs_random(gmo_ai, num_games=100)
print(f"Win rate vs Random: {results['win_rate']:.1%}")
```

---

## Evaluation Results

Evaluation on Square8 board type (December 2024):

| Matchup             | GMO Win Rate | Games | Notes                 |
| ------------------- | ------------ | ----- | --------------------- |
| GMO vs Random AI    | **74%**      | 50    | Target was >60%       |
| GMO vs Heuristic AI | **62.5%**    | 40    | Stronger than level 2 |

**Position in Difficulty Ladder:**

GMO's performance places it between Heuristic (level 2) and Policy-only (level 3):

```
Level 1: Random AI        <- GMO beats 74%
Level 2: Heuristic AI     <- GMO beats 62.5%
Level 3: Policy-only AI   <- GMO comparable
Level 4+: Minimax, MCTS   <- Not evaluated
```

GMO is available via `AIFactory.create_for_tournament("gmo", player_number)` but is not part of the main difficulty ladder. It serves as an alternative approach demonstrating gradient-based inference.

---

## Comparison to Other Approaches

| Approach           | Move Selection        | Exploration     | Based On                   |
| ------------------ | --------------------- | --------------- | -------------------------- |
| **Random AI**      | Uniform random        | N/A             | -                          |
| **Policy Network** | Softmax sampling      | Entropy bonus   | Standard RL                |
| **MCTS**           | Visit counts          | UCT formula     | Coulom 2006                |
| **AlphaZero**      | MCTS + Policy         | Dirichlet noise | Silver et al. 2017         |
| **GMO**            | Gradient + projection | UCB + novelty   | SPENs, UCB, Novelty Search |

### Advantages

1. No tree search required
2. Continuous refinement of candidates
3. Uncertainty-aware exploration

### Disadvantages

1. Projection may lose information
2. Multiple forward passes per move
3. No lookahead (single-move horizon)

---

## Potential Novel Extensions

Consider:

### 1. Learned Projection

Instead of nearest-neighbor cosine similarity, learn a projection network:

```
optimized_embed -> ProjectionNet -> legal_move_logits
```

This could reduce "projection regret."

### 2. True Information-Theoretic Objective

Replace UCB-style `sqrt(variance)` with actual information gain:

```
objective = E[value] + beta * I(outcome; action | state)
```

This would be closer to Bayesian optimization.

### 3. Theoretical Analysis

Provide bounds on when gradient optimization + projection improves over single-pass selection. Conditions for monotonic policy improvement.

### 4. Empirical Validation

Rigorous comparisons vs MCTS, policy networks at equal compute budgets, with ablations showing each component's contribution.

---

## IG-GMO (Experimental)

IG-GMO is a research extension that implements the "true information-theoretic objective" idea in code. It replaces
the variance-based exploration term with mutual information and swaps the MLP state encoder for a GNN encoder with
optional soft-legality constraints. It is **not** part of the canonical difficulty ladder and should be treated as
experimental.

Implementation: `app/ai/ig_gmo.py`

Usage examples:

- `AIFactory.create(AIType.IG_GMO, player_number, AIConfig(...))`
- `create_tournament_ai("ig_gmo", player_number, ...)`

Entry points:

- `app/ai/ig_gmo.py` (`IGGMO`)
- `app/ai/factory.py` via `AIType.IG_GMO` or tournament agent ID `ig_gmo`
- For evaluation, reuse the GMO harnesses by swapping candidate creation
  to IG-GMO in `scripts/gmo_eval_strong.py` (see `create_gmo`)

---

## File Reference

| File                         | Lines  | Description             |
| ---------------------------- | ------ | ----------------------- |
| `app/ai/gmo_ai.py`           | ~400   | Main implementation     |
| `app/training/train_gmo.py`  | ~350   | Training script         |
| `tests/test_gmo_ai.py`       | ~620   | Unit tests (28 tests)   |
| `docs/GMO_ALGORITHM.md`      | This   | Documentation           |
| `models/gmo/gmo_best.pt`     | ~1.5MB | Best trained checkpoint |
| `scripts/gmo_integration.py` | ~500   | Integration utilities   |

---

## Training Infrastructure Integration

GMO is fully integrated into the RingRift training infrastructure:

### Self-Play Generation

GMO is available as an engine mode for self-play data generation:

```bash
# Generate self-play games using GMO
python -m app.training.generate_data \
    --engine gmo \
    --num-games 1000 \
    --board square8 \
    --num-players 2
```

GMO can also be mixed with other engines:

```python
from app.training.selfplay_config import EngineMode

# GMO is available as an engine mode
config = SelfplayConfig(
    engine_mode=EngineMode.GMO,
    num_games=100,
)
```

### Elo System Registration

GMO can be registered as a participant in the unified Elo system:

```bash
# Register GMO in Elo database
python scripts/gmo_integration.py register
```

This creates a participant entry with:

- ID: `gmo_v1`
- Type: `gmo`
- Metadata: algorithm components and version

### Gauntlet Evaluation

GMO is automatically recognized in gauntlet evaluation:

- Models with IDs starting with `gmo` use the GMO AI type
- GMO can serve as both a candidate and a baseline
- Supports distributed gauntlet evaluation

### Self-Play Training Pipeline

Full pipeline for iterative self-play training:

```bash
# Run complete pipeline: selfplay -> train -> evaluate
python scripts/gmo_integration.py pipeline \
    --num-games 500 \
    --epochs 20 \
    --iterations 3

# Individual steps:
python scripts/gmo_integration.py selfplay --num-games 100
python scripts/gmo_integration.py train --epochs 10
python scripts/gmo_integration.py evaluate --num-games 50
```

### Tournament Support

GMO is available through the AIFactory for tournament play:

```python
from app.ai.factory import AIFactory

# Create GMO for tournament
gmo = AIFactory.create_for_tournament("gmo", player_number=1)
```

---

## Cluster Smoke Run (pending)

These runs should be executed on a cluster host; results are not yet recorded.

### GMO Baseline Smoke Run

```bash
python scripts/gmo_eval_strong.py \
    --opponents random,heuristic,mcts_100 \
    --games 20 \
    --device cuda \
    --output results/gmo_smoke.json
```

Record results here:

| Date (UTC) | Host | Opponents | Games | Win rate | Notes |
| ---------- | ---- | --------- | ----- | -------- | ----- |
| pending    |      |           |       |          |       |

---

## References

### Core Techniques Used

1. **Belanger, D., & McCallum, A.** (2016). Structured Prediction Energy Networks. ICML.
   _Gradient-based inference over continuous output relaxations_

2. **Dulac-Arnold, G., et al.** (2015). Deep Reinforcement Learning in Large Discrete Action Spaces. arXiv.
   _Action embeddings with nearest-neighbor selection_

3. **Gal, Y., & Ghahramani, Z.** (2016). Dropout as a Bayesian Approximation. ICML.
   _MC Dropout for uncertainty estimation_

4. **Auer, P., et al.** (2002). Finite-time Analysis of the Multiarmed Bandit Problem. Machine Learning.
   _UCB exploration principle_

5. **Lehman, J., & Stanley, K. O.** (2011). Abandoning Objectives: Evolution Through the Search for Novelty Alone. Evolutionary Computation.
   _Novelty search_

6. **Nix, D. A., & Weigend, A. S.** (1994). Estimating the mean and variance of the target probability distribution. ICNN.
   _Heteroscedastic regression / learned uncertainty_

### Related Work

7. **Silver, D., et al.** (2017). Mastering the Game of Go without Human Knowledge. Nature.
   _AlphaZero for comparison_

8. **Chandak, Y., et al.** (2019). Learning Action Representations for Reinforcement Learning. ICML.
   _Action embeddings survey_

---

_Last updated: December 2024_
_RingRift AI Service_
_Honest attribution: GMO is a synthesis of established techniques, not a novel algorithm._
