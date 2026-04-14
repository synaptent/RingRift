# Titans/MIRAS Implementation Plan for RingRift AI

## Executive Summary

This plan outlines how to integrate concepts from Google's Titans architecture (Neural Memory Modules with test-time learning) into RingRift's AI system to improve opponent modeling and game-specific adaptation.

**Key Finding from Surprise Analysis**: Current training data uses sparse policy format (single target action per sample), preventing meaningful surprise/entropy analysis. Phase 0 addresses this prerequisite.

---

## Background: Titans/MIRAS Concepts

From Google Research (Dec 2024):

1. **Neural Memory Module**: Trainable weights updated at inference time via gradient descent
2. **MIRAS Framework**: Unified view of memory operations:
   - **Memorization** = parameter adjustment via gradient descent
   - **Forgetting** = regularization toward prior
   - **Recall** = applying current parameters
   - **Surprise** = gradient magnitude (how unexpected the input is)
3. **Test-Time Learning**: Model adapts during inference without changing core weights

**Why Titans is Relevant to RingRift**:

- Strategic board games benefit from opponent modeling
- Game states have "surprising" positions (unexpected moves)
- Longer games (50-200 moves) have meaningful history
- Multiplayer dynamics require tracking individual opponents

---

## Phase 0: Enable Soft Policy Targets (Prerequisite)

**Problem**: Current selfplay stores only the chosen action (`policy_indices`), not the full MCTS visit distribution. This prevents surprise metric computation.

### Changes Required

**File**: `app/ai/gumbel_mcts_ai.py` (and related MCTS implementations)

```python
# Current: Returns only best action
return best_action

# New: Return action with policy distribution
return MCTSResult(
    action=best_action,
    policy_distribution=softmax(visit_counts),  # Full distribution
    root_value=root_value,
    visit_counts=visit_counts,
)
```

**File**: `app/db/game_replay_db.py`

```python
# Add soft policy storage to game_moves table
# Current: move_probs is often NULL or sparse
# New: Always store full policy distribution

def store_move_with_policy(self, game_id, move, policy_dist: np.ndarray):
    """Store move with full MCTS policy distribution."""
    # Store as compressed numpy array
    policy_blob = self._compress_policy(policy_dist)
    cursor.execute("""
        INSERT INTO game_moves (game_id, move_number, move_json, policy_probs)
        VALUES (?, ?, ?, ?)
    """, (game_id, move_num, json.dumps(move), policy_blob))
```

**File**: `scripts/export_replay_dataset.py`

```python
# Current: Exports sparse policy (policy_indices, policy_values)
# New: Export soft policy targets

def export_sample(move):
    if move.policy_probs:  # Soft targets available
        policy = self._decompress_policy(move.policy_probs)
    else:  # Fallback to one-hot
        policy = np.zeros(action_space_size)
        policy[move.action_index] = 1.0
    return policy
```

### Validation

```bash
# Generate test games with soft policy
python scripts/selfplay.py --board hex8 --num-players 2 \
  --engine gumbel --num-games 100 --store-soft-policy

# Verify entropy analysis now works
python scripts/analyze_surprise_from_npz.py \
  --npz data/training/hex8_2p_soft.npz
```

**Expected Output**:

- Mean entropy > 0.5 (vs 0 for sparse)
- Confidence distribution (not all 100%)
- Effective actions > 1 per sample

---

## Phase 1: Surprise Detection Module

Implement surprise metric as a standalone module before full Titans integration.

### File: `app/ai/surprise_detector.py`

```python
"""
Surprise Detection for RingRift AI.

Surprise = -log(P(observed_move)) where P is the policy probability.
High surprise indicates unexpected moves that may benefit from memory updates.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class SurpriseResult:
    """Result of surprise computation."""
    surprise: float  # -log(P(move)) in nats
    probability: float  # P(move)
    is_surprising: bool  # Above threshold
    policy_entropy: float  # H(policy) for context


class SurpriseDetector:
    """Detects surprising moves based on policy network predictions."""

    def __init__(
        self,
        surprise_threshold: float = 2.0,  # ~13.5% probability
        high_surprise_threshold: float = 3.0,  # ~5% probability
    ):
        self.surprise_threshold = surprise_threshold
        self.high_surprise_threshold = high_surprise_threshold

    def compute_surprise(
        self,
        policy_logits: torch.Tensor,
        chosen_action: int,
    ) -> SurpriseResult:
        """
        Compute surprise for a chosen action given policy logits.

        Args:
            policy_logits: Raw logits from policy head [action_space]
            chosen_action: Index of the action that was played

        Returns:
            SurpriseResult with surprise value and metadata
        """
        # Convert to probabilities
        probs = torch.softmax(policy_logits, dim=-1)
        p_action = probs[chosen_action].item()

        # Clamp to avoid log(0)
        p_action = max(p_action, 1e-10)

        # Surprise = negative log probability
        surprise = -math.log(p_action)

        # Compute entropy for context
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

        return SurpriseResult(
            surprise=surprise,
            probability=p_action,
            is_surprising=surprise > self.surprise_threshold,
            policy_entropy=entropy,
        )

    def should_update_memory(self, surprise_result: SurpriseResult) -> bool:
        """Determine if memory should be updated based on surprise."""
        # Update on high surprise (unexpected move)
        if surprise_result.surprise > self.high_surprise_threshold:
            return True
        # Also update on moderate surprise in high-entropy positions
        if (surprise_result.surprise > self.surprise_threshold and
            surprise_result.policy_entropy > 2.0):
            return True
        return False
```

### Integration Point

```python
# In game loop (e.g., AIEngine or GameEngine)
class AIWithSurprise:
    def __init__(self, model, surprise_detector: SurpriseDetector):
        self.model = model
        self.surprise_detector = surprise_detector
        self.surprise_history = []

    def process_opponent_move(self, state, opponent_move):
        """Process opponent's move and track surprise."""
        # Get our model's prediction for this position
        policy_logits, value = self.model(state)

        # Compute surprise
        result = self.surprise_detector.compute_surprise(
            policy_logits, opponent_move
        )

        self.surprise_history.append(result)

        # Trigger memory update if surprising
        if self.surprise_detector.should_update_memory(result):
            self._update_opponent_model(state, opponent_move, result)
```

---

## Phase 2: Neural Memory Module

Implement the core Titans memory architecture.

### File: `app/ai/neural_memory.py`

```python
"""
Neural Memory Module for RingRift AI.

Implements Titans-style test-time learning with differentiable memory.
Memory is updated via gradient descent at inference time.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class MemoryConfig:
    """Configuration for neural memory module."""
    memory_dim: int = 128  # Dimension of memory vectors
    num_memory_slots: int = 32  # Number of addressable memory slots
    learning_rate: float = 0.01  # Learning rate for memory updates
    momentum: float = 0.9  # Momentum for memory updates
    forget_rate: float = 0.01  # Regularization toward prior (forgetting)


class NeuralMemory(nn.Module):
    """
    Neural Memory Module with test-time learning.

    Implements the Titans architecture where memory weights are updated
    during inference via gradient descent on surprising inputs.
    """

    def __init__(self, input_dim: int, config: Optional[MemoryConfig] = None):
        super().__init__()
        self.config = config or MemoryConfig()

        # Memory parameters (updated at test time)
        self.memory_keys = nn.Parameter(
            torch.randn(self.config.num_memory_slots, self.config.memory_dim)
        )
        self.memory_values = nn.Parameter(
            torch.randn(self.config.num_memory_slots, self.config.memory_dim)
        )

        # Prior memory (for forgetting/regularization)
        self.register_buffer(
            "prior_keys", self.memory_keys.data.clone()
        )
        self.register_buffer(
            "prior_values", self.memory_values.data.clone()
        )

        # Input projection
        self.query_proj = nn.Linear(input_dim, self.config.memory_dim)
        self.key_proj = nn.Linear(input_dim, self.config.memory_dim)
        self.value_proj = nn.Linear(input_dim, self.config.memory_dim)

        # Output projection
        self.output_proj = nn.Linear(self.config.memory_dim, input_dim)

        # Memory update optimizer (separate from main model)
        self.memory_optimizer = None

    def _init_memory_optimizer(self):
        """Initialize optimizer for memory parameters only."""
        self.memory_optimizer = torch.optim.SGD(
            [self.memory_keys, self.memory_values],
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Query memory and return augmented representation.

        Args:
            x: Input tensor [batch, input_dim]

        Returns:
            Memory-augmented representation [batch, input_dim]
        """
        # Project input to memory space
        query = self.query_proj(x)  # [batch, memory_dim]

        # Attention over memory slots
        attn_weights = torch.softmax(
            torch.matmul(query, self.memory_keys.T) /
            (self.config.memory_dim ** 0.5),
            dim=-1
        )  # [batch, num_slots]

        # Retrieve from memory
        retrieved = torch.matmul(attn_weights, self.memory_values)
        # [batch, memory_dim]

        # Project back and residual connection
        output = self.output_proj(retrieved) + x

        return output

    def memorize(
        self,
        state_repr: torch.Tensor,
        target_repr: torch.Tensor,
        surprise: float,
    ):
        """
        Update memory based on surprising observation.

        This is the test-time learning step from Titans.

        Args:
            state_repr: State representation to memorize
            target_repr: Target representation (what should be remembered)
            surprise: Surprise value (scales update magnitude)
        """
        if self.memory_optimizer is None:
            self._init_memory_optimizer()

        # Scale learning rate by surprise
        effective_lr = self.config.learning_rate * min(surprise / 2.0, 2.0)
        for pg in self.memory_optimizer.param_groups:
            pg["lr"] = effective_lr

        # Memory prediction
        query = self.query_proj(state_repr)
        attn_weights = torch.softmax(
            torch.matmul(query, self.memory_keys.T) /
            (self.config.memory_dim ** 0.5),
            dim=-1
        )
        predicted = torch.matmul(attn_weights, self.memory_values)

        # Loss: prediction error
        loss = torch.mean((predicted - self.value_proj(target_repr)) ** 2)

        # Forgetting regularization (pull toward prior)
        forget_loss = self.config.forget_rate * (
            torch.mean((self.memory_keys - self.prior_keys) ** 2) +
            torch.mean((self.memory_values - self.prior_values) ** 2)
        )

        total_loss = loss + forget_loss

        # Update memory
        self.memory_optimizer.zero_grad()
        total_loss.backward()
        self.memory_optimizer.step()

    def reset_memory(self):
        """Reset memory to prior (between games)."""
        self.memory_keys.data.copy_(self.prior_keys)
        self.memory_values.data.copy_(self.prior_values)
```

---

## Phase 3: Opponent Model Integration

Integrate memory into the game engine for opponent-specific adaptation.

### File: `app/ai/opponent_memory.py`

```python
"""
Opponent-specific memory for RingRift AI.

Tracks patterns for each opponent player and adapts predictions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from app.ai.neural_memory import NeuralMemory, MemoryConfig
from app.ai.surprise_detector import SurpriseDetector, SurpriseResult


@dataclass
class OpponentProfile:
    """Profile for a single opponent."""
    player_id: int
    memory: NeuralMemory
    move_history: List[int] = field(default_factory=list)
    surprise_history: List[float] = field(default_factory=list)
    games_played: int = 0


class OpponentMemoryManager:
    """
    Manages opponent-specific memories for multiplayer games.

    Each opponent gets their own memory module that learns their patterns
    during gameplay via test-time updates.
    """

    def __init__(
        self,
        state_dim: int,
        memory_config: Optional[MemoryConfig] = None,
        num_players: int = 2,
    ):
        self.state_dim = state_dim
        self.memory_config = memory_config or MemoryConfig()
        self.num_players = num_players

        self.surprise_detector = SurpriseDetector()

        # One memory per opponent (excluding self)
        self.opponent_memories: Dict[int, OpponentProfile] = {}

    def get_or_create_profile(self, player_id: int) -> OpponentProfile:
        """Get or create opponent profile."""
        if player_id not in self.opponent_memories:
            self.opponent_memories[player_id] = OpponentProfile(
                player_id=player_id,
                memory=NeuralMemory(self.state_dim, self.memory_config),
            )
        return self.opponent_memories[player_id]

    def process_opponent_move(
        self,
        player_id: int,
        state_repr: torch.Tensor,
        move_repr: torch.Tensor,
        policy_logits: torch.Tensor,
        chosen_action: int,
    ) -> SurpriseResult:
        """
        Process an opponent's move and update their profile.

        Args:
            player_id: ID of the opponent player
            state_repr: State representation from encoder
            move_repr: Move representation (what opponent played)
            policy_logits: Our model's policy prediction for this state
            chosen_action: The action the opponent chose

        Returns:
            SurpriseResult from the move
        """
        profile = self.get_or_create_profile(player_id)

        # Compute surprise
        surprise_result = self.surprise_detector.compute_surprise(
            policy_logits, chosen_action
        )

        # Track history
        profile.move_history.append(chosen_action)
        profile.surprise_history.append(surprise_result.surprise)

        # Update memory if surprising
        if self.surprise_detector.should_update_memory(surprise_result):
            profile.memory.memorize(
                state_repr=state_repr,
                target_repr=move_repr,
                surprise=surprise_result.surprise,
            )

        return surprise_result

    def augment_prediction(
        self,
        state_repr: torch.Tensor,
        current_player: int,
    ) -> torch.Tensor:
        """
        Augment state representation with opponent memory.

        Args:
            state_repr: Base state representation
            current_player: The player whose turn it is

        Returns:
            Memory-augmented state representation
        """
        # Get opponent profile (the one we're predicting for)
        profile = self.opponent_memories.get(current_player)

        if profile is None:
            return state_repr

        # Apply opponent memory
        return profile.memory(state_repr)

    def reset_for_new_game(self):
        """Reset memories between games (optional)."""
        for profile in self.opponent_memories.values():
            profile.memory.reset_memory()
            profile.games_played += 1
```

---

## Phase 4: Model Architecture Integration

Integrate memory into the existing CNN architecture.

### Changes to `app/ai/neural_net/` architecture modules

```python
# Add memory-augmented forward pass option

class NeuralNetWithMemory(nn.Module):
    """RingRift neural network with optional memory augmentation."""

    def __init__(self, base_model: nn.Module, memory_config: MemoryConfig):
        super().__init__()
        self.base = base_model

        # Memory module after feature extraction, before heads
        self.memory = NeuralMemory(
            input_dim=base_model.feature_dim,
            config=memory_config,
        )

        # Whether to use memory (can toggle at inference)
        self.use_memory = True

    def forward(self, x, use_memory: bool = True):
        """Forward pass with optional memory augmentation."""
        # Extract features through residual blocks
        features = self.base.extract_features(x)

        # Apply memory if enabled
        if use_memory and self.use_memory:
            features = self.memory(features)

        # Policy and value heads
        policy = self.base.policy_head(features)
        value = self.base.value_head(features)

        return policy, value

    def memorize_opponent_move(self, state, opponent_move, surprise):
        """Update memory with opponent observation."""
        features = self.base.extract_features(state)
        # Target is the one-hot encoding of opponent move
        target = torch.zeros_like(features)  # Placeholder - needs encoding
        self.memory.memorize(features, target, surprise)
```

Instantiate `base_model` through the supported `app.ai.neural_net` facade, for
example via `create_model_for_board(...)`, rather than relying on older
one-off model-file paths.

---

## Phase 5: Training Integration

Add memory module to training pipeline.

### Changes to `app/training/train.py`

```python
# Add surprise-weighted loss option

def compute_loss_with_surprise(
    policy_logits: torch.Tensor,
    policy_target: torch.Tensor,
    value_pred: torch.Tensor,
    value_target: torch.Tensor,
    surprise_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute loss with optional surprise weighting.

    Positions with higher surprise get more training weight,
    as they represent harder-to-predict situations.
    """
    # Cross-entropy for policy
    policy_loss = F.cross_entropy(policy_logits, policy_target, reduction='none')

    # MSE for value
    value_loss = F.mse_loss(value_pred, value_target, reduction='none')

    if surprise_weights is not None:
        # Weight by surprise (normalized)
        weights = F.softmax(surprise_weights, dim=0)
        policy_loss = policy_loss * weights
        value_loss = value_loss.squeeze() * weights

    return policy_loss.mean() + value_loss.mean()
```

### Training with Soft Targets

```bash
# Train with soft policy targets (required for surprise-weighted loss)
python -m app.training.train \
  --board-type hex8 --num-players 2 \
  --data-path data/training/hex8_2p_soft.npz \
  --use-soft-policy-targets \
  --surprise-weighting 0.5
```

---

## Phase 6: Inference Integration

Integrate memory into game playing.

### Changes to `src/server/game/ai/AIEngine.ts`

```typescript
// Add memory-augmented inference option

interface MemoryState {
  opponentProfiles: Map<number, OpponentProfile>;
  gameHistory: GameHistoryEntry[];
}

class AIEngineWithMemory extends AIEngine {
  private memoryState: MemoryState;

  async computeMove(gameState: GameState): Promise<Move> {
    // Get base prediction
    const { policy, value } = await this.model.predict(gameState);

    // Check last opponent move for surprise
    if (gameState.moveHistory.length > 0) {
      const lastMove = gameState.moveHistory[gameState.moveHistory.length - 1];
      if (lastMove.player !== this.ourPlayer) {
        const surprise = this.computeSurprise(gameState.previousState, lastMove, policy);

        if (surprise.isSurprising) {
          // Update opponent model
          await this.updateOpponentMemory(
            lastMove.player,
            gameState.previousState,
            lastMove,
            surprise
          );
        }
      }
    }

    // Apply MCTS with memory-augmented value estimates
    return this.mcts.search(gameState, { memoryAugmented: true });
  }
}
```

---

## Implementation Timeline

| Phase                    | Effort  | Prerequisites | Impact                        |
| ------------------------ | ------- | ------------- | ----------------------------- |
| 0: Soft Policy Targets   | 4-6 hrs | None          | Enables all subsequent phases |
| 1: Surprise Detection    | 2-3 hrs | Phase 0       | Standalone utility            |
| 2: Neural Memory         | 4-6 hrs | None          | Core Titans component         |
| 3: Opponent Memory       | 3-4 hrs | Phase 1, 2    | Multiplayer adaptation        |
| 4: Model Integration     | 4-6 hrs | Phase 2       | Full architecture             |
| 5: Training Integration  | 3-4 hrs | Phase 0, 4    | Surprise-weighted training    |
| 6: Inference Integration | 4-6 hrs | All above     | End-to-end system             |

**Total Estimated Effort**: 24-35 hours

---

## Success Metrics

1. **Surprise Distribution**:
   - Mean entropy > 1.0 (was 0 with sparse targets)
   - 10-30% of moves classified as "surprising"

2. **Opponent Modeling**:
   - Win rate improvement vs repeated opponents
   - Faster adaptation in game series

3. **Memory Effectiveness**:
   - Reduced prediction error on "surprising" positions
   - Lower variance in value estimates mid-game

4. **Elo Improvement**:
   - Target: +50-100 Elo from memory augmentation
   - Especially in longer games (>100 moves)

---

## Alternatives Considered

### Option B: Frame Stacking Enhancement (Simpler)

Instead of full Titans memory, enhance existing frame stacking:

- Increase history from 3-4 frames to 8-16 frames
- Add attention over history frames
- Estimated: 8-12 hours, +20-50 Elo

### Option C: Pattern Library (Hybrid)

Pre-computed pattern library with runtime matching:

- Store common board patterns with best responses
- Hash-based lookup during inference
- Estimated: 10-15 hours, +30-70 Elo

**Recommendation**: Start with Phase 0 + Phase 1 to enable data collection, then evaluate whether full Titans implementation is warranted based on surprise distribution analysis.

---

## Appendix: Current Architecture Compatibility

### RingRift Net v2 (Current Production)

- 96 channels, 6 residual blocks with SE attention
- Feature dimension: 96 (after global pooling)
- Memory integration point: After residual blocks, before heads

### RingRift Net v4 (Experimental)

- 128 channels with attention layers
- Feature dimension: 128
- Better suited for memory integration (has attention already)

### MCTS Integration

- Gumbel MCTS already batches evaluations
- Memory can be applied at the neural network level
- No changes needed to search algorithm itself

---

## References

1. [Titans: Learning to Memorize at Test Time](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/) - Google Research, Dec 2024
2. [MIRAS: Memorization, Inference, Recall, and Attention via Surprise](https://arxiv.org/abs/2312.xxxxx) - Arxiv
3. [AlphaZero](https://www.deepmind.com/research/highlighted-research/alphazero) - DeepMind

---

_Created: 2025-12-26_
_Status: Draft - Pending Phase 0 prerequisite completion_
