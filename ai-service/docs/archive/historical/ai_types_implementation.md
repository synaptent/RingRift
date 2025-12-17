# AI Types Implementation - Policy-Only & Gumbel MCTS

## Overview

This document describes the implementation of two new AI types added to RingRift's AI system for more robust neural network evaluation and faster inference options.

## New AI Types

### 1. Policy-Only AI (`AIType.POLICY_ONLY`)

**Location**: `app/ai/policy_only_ai.py`

**Description**: Direct neural network policy output without any search. Performs a single forward pass through the NN and samples from the resulting policy distribution.

**Key Features**:

- Extremely fast (~100x faster than MCTS)
- Single forward pass inference
- Configurable temperature for exploration/exploitation tradeoff
- Falls back to uniform policy if NN unavailable

**Configuration Options**:

```python
AIConfig(
    policy_temperature=1.0,  # 0.01=greedy, 1.0=sample, >1.0=more uniform
    nn_model_id="models/your_model.pth",
)
```

**Use Cases**:

- Fast baseline evaluation of NN policy quality
- Rapid selfplay data generation (behavioral cloning)
- Calibrating NN strength independent of search
- Quick prototype testing

**Algorithm**:

```
1. Get valid moves from rules engine
2. Forward pass through NN → policy vector
3. Map valid moves to policy indices
4. Apply temperature softmax
5. Sample from distribution (or argmax if temp ≤ 0.01)
```

### 2. Gumbel MCTS AI (`AIType.GUMBEL_MCTS`)

**Location**: `app/ai/gumbel_mcts_ai.py`

**Description**: Sample-efficient MCTS using Gumbel-Top-K sampling and Sequential Halving from the Gumbel AlphaZero paper (Danihelka et al., 2022).

**Key Features**:

- More sample-efficient than standard MCTS
- Gumbel-Top-K for action sampling without replacement
- Sequential Halving for budget allocation
- Focuses computation on promising actions

**Configuration Options**:

```python
AIConfig(
    gumbel_num_sampled_actions=16,  # Actions to sample at root (m)
    gumbel_simulation_budget=150,   # Total simulation budget (n)
    nn_model_id="models/your_model.pth",
)
```

**Use Cases**:

- Better search with limited compute budget
- Policy calibration and evaluation
- Sample-efficient tournament play

**Algorithm**:

```
1. Get policy logits from NN
2. Gumbel-Top-K: Add Gumbel(0,1) noise to logits, take top k
3. Sequential Halving:
   - Divide budget across log2(k) phases
   - Each phase: simulate remaining actions, keep top half
4. Return action with highest completed Q-value
```

**References**:

- Danihelka et al. "Policy improvement by planning with Gumbel" (2022)
- https://arxiv.org/abs/2104.06303

## Multiplayer AI Types (Existing, Now Integrated)

For 3-player and 4-player games, the tournament system now automatically includes:

### MaxN AI (`AIType.MAXN`)

- Each player maximizes their own score
- More realistic multiplayer model than paranoid search
- Location: `app/ai/maxn_ai.py`

### Best-Reply Search (`AIType.BRS`)

- Fast approximation for multiplayer
- Assumes opponents play greedy best replies
- Location: `app/ai/maxn_ai.py`

## Files Modified

### Core Model Changes

- **`app/models/core.py`**
  - Added `AIType.POLICY_ONLY` and `AIType.GUMBEL_MCTS` to enum (lines 225-226)
  - Added `policy_temperature` config field (lines 723-732)
  - Added `gumbel_num_sampled_actions` config field (lines 738-746)
  - Added `gumbel_simulation_budget` config field (lines 748-757)

### New AI Implementations

- **`app/ai/policy_only_ai.py`** - Policy-Only AI implementation
- **`app/ai/gumbel_mcts_ai.py`** - Gumbel MCTS implementation

### Factory Registration

- **`app/ai/factory.py`**
  - Added lazy loading for PolicyOnlyAI (lines 357-359)
  - Added lazy loading for GumbelMCTSAI (lines 360-362)
  - Added tournament support for `policy_only` and `gumbel_mcts` agent IDs (lines 560-606)

### Tournament Integration

- **`scripts/run_model_elo_tournament.py`**
  - Added `--ai-type` choices: `policy_only`, `gumbel_mcts`, `maxn`, `brs`
  - Changed `--both-ai-types` to default `True`
  - Added `--no-both-ai-types` flag to disable
  - Updated AI type combinations for 2-player (8 combos) and 3/4-player (10 combos)
  - Added AI creation logic for new types

## Tournament Configuration

### Default Behavior (--both-ai-types=True)

**2-Player Games** - 8 AI type combinations:

1. descent vs descent
2. mcts vs mcts
3. policy_only vs policy_only
4. gumbel_mcts vs gumbel_mcts
5. mcts vs descent
6. descent vs mcts
7. policy_only vs descent
8. gumbel_mcts vs descent

**3/4-Player Games** - 10 AI type combinations:

1. descent vs descent
2. mcts vs mcts
3. maxn vs maxn
4. brs vs brs
5. policy_only vs policy_only
6. gumbel_mcts vs gumbel_mcts
7. mcts vs maxn
8. descent vs maxn
9. brs vs descent
10. maxn vs brs

### CLI Usage

```bash
# Default: uses all AI types automatically
python scripts/run_model_elo_tournament.py --board square8 --players 2 --run

# Multiplayer with full AI coverage
python scripts/run_model_elo_tournament.py --board square8 --players 3 --run

# Single AI type only
python scripts/run_model_elo_tournament.py --no-both-ai-types --ai-type gumbel_mcts --run

# Specific AI type for all games
python scripts/run_model_elo_tournament.py --no-both-ai-types --ai-type policy_only --run
```

## Factory Usage

```python
from app.ai.factory import AIFactory

# Create via tournament interface
ai = AIFactory.create_for_tournament(
    agent_id="policy_only",  # or "gumbel_mcts", "policy_0.5", "gumbel_100"
    player_number=1,
    board_type="square8",
    nn_model_id="models/your_model.pth",
)

# Create via explicit type
from app.models import AIType, AIConfig
config = AIConfig(
    difficulty=5,
    nn_model_id="models/your_model.pth",
    policy_temperature=0.5,
)
ai = AIFactory.create(AIType.POLICY_ONLY, player_number=1, config=config)
```

## Expected Elo Ranges (To Be Calibrated)

Based on theoretical analysis and initial testing:

| AI Type                | Expected Elo Range | Speed (relative) |
| ---------------------- | ------------------ | ---------------- |
| Random                 | 1200               | 1000x            |
| Heuristic              | 1350               | 500x             |
| Policy-Only            | 1400-1500          | 100x             |
| Gumbel MCTS (150 sims) | 1500-1600          | 10x              |
| MCTS (800 sims)        | 1550-1650          | 2x               |
| Descent                | 1600-1700          | 1x               |

_Note: Actual Elo will depend on NN quality and should be calibrated via tournament._

## Performance Characteristics

### Benchmark Results (GH200 GPU, 2025-12-16)

| AI Type     | Moves/sec | Speedup vs Descent |
| ----------- | --------- | ------------------ |
| policy_only | 11.30     | 45.3x              |
| gumbel_mcts | 11.22     | 45.0x              |
| mcts        | 0.25      | 1.0x               |
| descent     | 0.25      | 1.0x               |

**Key Findings**:

- Policy-Only and Gumbel MCTS are ~45x faster than search-based methods
- MCTS and Descent have similar throughput (both are search-intensive)
- Gumbel MCTS achieves near-Policy-Only speed while adding search quality

### Memory & Resource Usage

| AI Type     | Memory | GPU Required |
| ----------- | ------ | ------------ |
| Policy-Only | Low    | Yes          |
| Gumbel MCTS | Medium | Yes          |
| MCTS        | High   | Yes          |
| Descent     | High   | Yes          |
| MaxN        | Medium | No           |
| BRS         | Low    | No           |

## Gumbel MCTS Parameter Tuning Results (2025-12-16)

### Summary

Tested 9 configurations of (m, budget) against Policy-Only baseline, 10 games each:

**100% Win Rate Configurations (Recommended):**
| m | budget | moves/sec | Notes |
|---|--------|-----------|-------|
| 16 | 100 | 21.6 | **Best speed/quality** |
| 32 | 100 | 20.9 | Same quality, slightly slower |
| 16 | 200 | 11.9 | Overkill for this baseline |
| 32 | 200 | 12.7 | Overkill |
| 32 | 400 | 6.6 | Diminishing returns |

**0% Win Rate Configurations (Insufficient Search):**
| m | budget | moves/sec | Notes |
|---|--------|-----------|-------|
| 16 | 50 | 36.7 | Budget too low |
| 8 | 50 | 29.2 | m too low |
| 8 | 100 | 22.7 | m too low |
| 8 | 200 | 13.0 | m too low |

### Key Findings

1. **Minimum viable m is 16**: m=8 never beats Policy-Only regardless of budget
2. **Budget ≥100 is sufficient**: With m=16, budget=100 achieves 100% win rate
3. **Diminishing returns above 100 sims**: Higher budgets only reduce speed
4. **m=32 ≈ m=16 in quality**: No benefit from more sampled actions

### Recommended Configurations

- **Fast mode**: m=16, budget=100 (21.6 moves/sec)
- **Balanced mode**: m=16, budget=150 (current default)
- **High quality**: m=16, budget=200 (11.9 moves/sec)

## Future Work

1. **Difficulty Level Mapping**: Once Elo is calibrated, map new AI types to difficulty levels 1-10
2. ~~**Parameter Tuning**: Optimize `gumbel_num_sampled_actions` and `gumbel_simulation_budget`~~ ✓ Done
3. **Fast Selfplay**: Use Policy-Only for rapid behavioral cloning data generation
4. **Frontend Integration**: Expose new AI types in user-facing difficulty selector
5. **Hybrid Search**: Combine Gumbel MCTS root selection with Descent tree search
