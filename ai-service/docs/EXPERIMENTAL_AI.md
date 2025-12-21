# Experimental AI Approaches

This document describes the experimental AI algorithms available in RingRift beyond the production ladder (D1-D11).

## Overview

| AI Type         | Difficulty | Status      | Win Rate vs Random |
| --------------- | ---------- | ----------- | ------------------ |
| **EBMO**        | D12        | Functional  | 70%                |
| **GMO**         | D13        | Implemented | Untested           |
| **IG-GMO**      | D14        | Implemented | Untested           |
| **CAGE**        | N/A        | Broken      | 0%                 |
| **GPU Minimax** | D15        | Implemented | Untested           |

## EBMO (Energy-Based Move Optimization)

**Core Idea:** Use gradient descent on action embeddings at inference time.

**How It Works:**

1. Encode board state into 256-dim embedding
2. Initialize action embedding (random or from prior)
3. Run gradient descent to minimize energy E(state, action)
4. Project optimized embedding to nearest legal move

**Strengths:**

- Novel continuous exploration of action space
- No tree search required
- Gradient guidance vs random sampling

**Weaknesses:**

- Loses to Heuristic AI (35% win rate)
- Training data may not capture strategic principles
- Local minima in energy landscape

**Files:**

- `app/ai/ebmo_ai.py` - Main AI agent
- `app/ai/ebmo_network.py` - Neural network
- `app/ai/ebmo_online.py` - Online learning

## GMO (Gradient Move Optimization)

**Core Idea:** Entropy-guided gradient ascent in move embedding space.

**How It Works:**

1. Sample moves from policy network
2. Compute entropy of current action distribution
3. Gradient ascent toward high-entropy, high-value regions
4. Rerank by estimated Q-value

**Status:** Implemented but not benchmarked against ladder.

**Files:**

- `app/ai/gmo_ai.py`
- `app/ai/gmo_network.py`

## IG-GMO (Information-Gain GMO)

**Core Idea:** Mutual information-based exploration with GNN state encoding.

**Difficulty Tier:** D14 (experimental)

**How It Works:**

1. Encode state using Graph Attention Network (3 layers, 4 heads)
2. Compute mutual information via MC Dropout sampling: MI(y; theta | s, a) = H(E[p]) - E[H(p)]
3. Select moves that maximize expected information gain
4. Balance exploration vs exploitation via beta coefficient (default 0.2)
5. Track novelty with gamma coefficient (default 0.05)

**Configuration:** `app/ai/factory.py:195-203`

```python
14: {
    "ai_type": AIType.IG_GMO,
    "randomness": 0.1,
    "think_time_ms": 2000,
    "profile_id": "v3-iggmo-14-experimental",
    "use_neural_net": True,
}
```

**Key Components:**

- `IGGMOConfig`: State/move embedding dims (128), GNN layers (3), attention heads (4)
- `GraphAttentionLayer`: Multi-head attention over spatial neighbors
- `GNNStateEncoder`: Converts GameState to graph embeddings (8-connected neighbors)
- `SoftLegalityPredictor`: Learned differentiable legality function

**Enable at D3 (Experimental Override):**

```bash
export RINGRIFT_USE_IG_GMO=1
```

This override is wired in `app/ai/factory.py` and affects difficulty-based AI selection.

**Status:** Fully wired into AI factory. Untested against ladder.

**Files:**

- `app/ai/ig_gmo.py` - Main implementation (676 lines)

## CAGE (Constraint-Aware Graph EBMO)

**Core Idea:** Graph neural network with primal-dual legality constraints.

**How It Works:**

1. Represent board as graph (cells = nodes, adjacency = edges)
2. Message passing to compute node/graph embeddings
3. Primal-dual optimization to stay on legal move manifold
4. Energy minimization subject to legality constraints

**Status:** Broken. Loses 100% to Random AI despite training.

**Known Issues:**

- Graph construction may not capture game semantics
- Primal-dual optimization may be unstable
- Training data insufficient for GNN

**Files:**

- `app/ai/cage_ai.py`
- `app/ai/cage_network.py`

## GPU Minimax

**Core Idea:** GPU-accelerated minimax with batched neural network evaluation.

**How It Works:**

1. Standard alpha-beta minimax tree search
2. Batch leaf node evaluations on GPU
3. Parallel move generation on GPU
4. Faster search at same depth as CPU minimax

**Status:** Implemented for CUDA/MPS devices.

**Files:**

- `app/ai/gpu_minimax.py`
- `app/ai/gpu_kernels.py`

## Using Experimental AIs

### Via Difficulty Level

```python
from app.ai.factory import create_ai
from app.models import AIConfig

# Use experimental difficulty levels (12-15)
config = AIConfig(difficulty=12)  # EBMO
ai = create_ai(player_number=1, config=config)
```

### Via Direct Import

```python
from app.ai.ebmo_ai import EBMO_AI
from app.models import AIConfig

config = AIConfig(difficulty=5)
ai = EBMO_AI(
    player_number=1,
    config=config,
    model_path="models/ebmo_56ch/ebmo_quality_best.pt"
)
```

### Enable Experimental Overrides at D3

```bash
# EBMO at D3
export RINGRIFT_USE_EBMO=1

# IG-GMO at D3
export RINGRIFT_USE_IG_GMO=1
```

This replaces PolicyOnly at D3 with the experimental AI for A/B testing.

## Benchmarking

Run the ladder benchmark:

```bash
python scripts/benchmark_ebmo_ladder.py
```

This tests EBMO against:

1. Random AI (D1)
2. Heuristic AI (D2)
3. PolicyOnly AI (D3)

## Research Directions

### Near-term (likely to improve results)

1. **Outcome-weighted training:** Train EBMO on game winners, not MCTS labels
2. **Curriculum learning:** Progressive difficulty from Random → Heuristic → MCTS
3. **Online learning:** Use `EBMOOnlineAI` for continuous improvement

### Medium-term (moderate effort)

1. **Hybrid EBMO+MCTS:** Use EBMO for pruning, MCTS for final selection
2. **Attention over actions:** Learn to weight legal moves differentially
3. **Learned projection:** Replace nearest-neighbor with learned decoder

### Long-term (research-grade)

1. **Energy-based planning:** Multi-step trajectory optimization
2. **Adversarial energy:** Train energy function via self-play
3. **Meta-learning:** Learn to optimize across game positions

## Lessons Learned

1. **Training signal matters:** MCTS labels teach imitation, not winning
2. **Heuristics are hard to beat:** Simple rules encode domain knowledge
3. **Gradient-based search is viable:** EBMO concept is sound
4. **Online learning works:** TD-Energy updates decrease loss
5. **Graph NNs are tricky:** CAGE needs fundamental rework
