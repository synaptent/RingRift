# AI Module (`app/ai`)

The AI module provides neural network-powered and traditional game-playing agents for RingRift. It includes multiple MCTS variants, GPU-accelerated game simulation, and sophisticated neural network architectures for both hexagonal and square boards.

## Table of Contents

- [Module Purpose](#module-purpose)
- [Architecture Overview](#architecture-overview)
- [Main Components](#main-components)
- [Neural Network Architectures](#neural-network-architectures)
- [Usage Examples](#usage-examples)
- [Component Relationships](#component-relationships)
- [Performance Considerations](#performance-considerations)

## Module Purpose

This module serves three primary functions:

1. **Game Playing**: Provide AI opponents of varying skill levels (difficulty 1-11) for gameplay
2. **Training Data Generation**: Run high-quality selfplay games to create training datasets
3. **Model Evaluation**: Compare model performance through tournament play

The module is designed to support both CPU-only execution (heuristic players) and GPU-accelerated inference and search (neural network players with MCTS).

## Architecture Overview

```
app/ai/
├── Core Infrastructure
│   ├── base.py                 # BaseAI abstract class
│   ├── factory.py              # AI factory and difficulty profiles
│   ├── registry.py             # AI type registration system
│   └── model_cache.py          # Model weight caching
│
├── AI Player Implementations
│   ├── random_ai.py            # Random move selection (baseline)
│   ├── heuristic_ai.py         # Rule-based evaluation (75KB, highly optimized)
│   ├── policy_only_ai.py       # Pure neural policy (no search)
│   ├── minimax_ai.py           # Minimax for 2-player games
│   ├── maxn_ai.py              # MaxN for multi-player games
│   ├── mcts_ai.py              # Standard MCTS (132KB)
│   ├── gumbel_mcts_ai.py       # Gumbel AlphaZero MCTS (70KB, production)
│   ├── descent_ai.py           # Gradient descent search
│   ├── gmo_ai.py               # Gradient Move Optimization
│   ├── ebmo_ai.py              # Energy-Based Move Optimization
│   └── universal_ai.py         # Unified AI wrapper
│
├── Neural Network (`neural_net/`)
│   ├── __init__.py             # Package exports
│   ├── constants.py            # Policy sizes and encoding constants
│   ├── blocks.py               # Reusable network building blocks
│   ├── model_factory.py        # Board-specific model creation
│   │
│   ├── Architecture Families
│   │   ├── hex_architectures.py      # Hex board CNNs (v2, v3)
│   │   ├── square_architectures.py   # Square board CNNs (v2, v3, v4)
│   │   ├── gnn_policy.py             # Graph Neural Network
│   │   └── hybrid_cnn_gnn.py         # Hybrid CNN-GNN
│   │
│   └── Encoding & Utilities
│       ├── hex_encoding.py     # Hexagonal action encoding
│       ├── square_encoding.py  # Square action encoding
│       └── graph_encoding.py   # Graph construction for GNN
│
├── MCTS Variants
│   ├── gumbel_common.py        # Shared data structures (GumbelAction, GumbelNode)
│   ├── gumbel_mcts_ai.py       # Production Gumbel MCTS
│   ├── batched_gumbel_mcts.py  # GPU-batched Gumbel
│   ├── gumbel_mcts_gpu.py      # GPU kernel integration
│   ├── tensor_gumbel_tree.py   # Tensor-based tree (66KB)
│   ├── entropy_mcts.py         # Entropy-regularized MCTS
│   └── improved_mcts_ai.py     # Enhanced MCTS features
│
├── GPU Engine (Production-Ready)
│   ├── gpu_parallel_games.py   # Parallel game execution (170KB, 6-57x speedup on CUDA)
│   ├── gpu_batch_state.py      # Batched game state tensors (80KB)
│   ├── gpu_move_generation.py  # Vectorized move generation (103KB)
│   ├── gpu_move_application.py # Vectorized move application (102KB)
│   ├── gpu_line_detection.py   # GPU line detection (21KB)
│   ├── gpu_territory.py        # Territory computation (36KB)
│   ├── gpu_selection.py        # Move selection logic (28KB)
│   ├── gpu_heuristic.py        # GPU heuristic evaluation
│   ├── gpu_batch.py            # Batch utilities
│   └── gpu_game_types.py       # Type definitions
│
├── Specialized Components
│   ├── nnue.py                 # NNUE evaluation (60KB)
│   ├── nnue_policy.py          # NNUE with policy network (117KB)
│   ├── ebmo_network.py         # Energy-based network (37KB)
│   ├── ebmo_online_learner.py  # Online learning for EBMO (20KB)
│   ├── ensemble_inference.py   # Model ensemble support
│   └── unified_loader.py       # Unified model loading
│
├── Evaluation & Heuristics
│   ├── evaluators/             # Modular evaluation components
│   │   ├── material_evaluator.py    # Material balance
│   │   ├── positional_evaluator.py  # Position quality
│   │   ├── tactical_evaluator.py    # Tactical threats
│   │   ├── mobility_evaluator.py    # Piece mobility
│   │   ├── strategic_evaluator.py   # Long-term strategy
│   │   └── endgame_evaluator.py     # Endgame patterns
│   ├── heuristic_weights.py    # Weight profiles for heuristic AI
│   ├── evaluation_provider.py  # Unified model loading
│   └── swap_evaluation.py      # Pie rule (swap sides) evaluation
│
├── Search Support
│   ├── zobrist.py              # Zobrist hashing for transposition tables
│   ├── bounded_transposition_table.py  # Memory-bounded caching
│   ├── move_cache.py           # Move generation caching
│   ├── territory_cache.py      # Territory computation caching
│   └── move_ordering.py        # Move ordering heuristics
│
├── Game State Utilities
│   ├── game_state_utils.py     # State manipulation helpers
│   ├── lightweight_state.py    # Minimal state representation
│   ├── fast_geometry.py        # Fast board geometry ops
│   └── canonical_move_encoding.py  # Canonical move encoding
│
├── Evaluation & Inference
│   ├── batch_eval.py           # Batched neural inference
│   ├── async_nn_eval.py        # Async neural evaluation
│   ├── parallel_eval.py        # Parallel position evaluation
│   ├── lightweight_eval.py     # Lightweight evaluation
│   └── numba_eval.py           # Numba-optimized evaluation
│
├── Optimization & Validation
│   ├── shadow_validation.py    # GPU/CPU parity validation
│   ├── gpu_memory_profiler.py  # Memory profiling
│   ├── cache_invalidation.py   # Cache management
│   └── decision_log.py         # Decision logging for analysis
│
├── Research & Experimental
│   ├── gmo_v2.py               # GMO v2 implementation
│   ├── ig_gmo.py               # Information-Gain GMO
│   ├── gmo_shared.py           # GMO shared utilities
│   ├── gmo_policy_provider.py  # GMO policy interface
│   ├── cmaes_diversity.py      # CMA-ES with diversity
│   ├── multi_opponent_fitness.py  # Multi-opponent evaluation
│   ├── marl_framework.py       # Multi-agent RL framework
│   └── neural_losses.py        # Custom loss functions
│
└── Archive
    └── archive/                # Deprecated implementations
        ├── cage_ai.py          # Original cage-based AI
        └── cage_network.py     # Original cage network
```

## Main Components

### 1. AI Player Factory (`factory.py`)

Central factory for creating AI instances with difficulty-based profiles.

**Key Functions:**

- `create_ai_from_difficulty(difficulty, player_number, ...)` - Create AI from difficulty 1-11
- `create_ai(ai_type, player_number, config)` - Create AI with explicit type
- `get_difficulty_profile(difficulty)` - Get canonical difficulty settings
- `AIFactory.create(...)` - Main factory method

**Difficulty Ladder:**

- 1: Random (baseline)
- 2: Heuristic with randomness
- 3: Policy-only (neural network, no search)
- 4: Minimax (2P) / MaxN (3-4P)
- 5: Minimax+NNUE
- 6: Descent with neural guidance
- 7: MCTS with neural guidance
- 8-11: Gumbel MCTS with increasing budgets (strongest)

### 2. Neural Network Module (`neural_net/`)

Provides board-specific neural network architectures.

**Key Classes:**

- `NeuralNetAI` - Main AI wrapper for neural network inference
- `create_model_for_board(board_type)` - Factory for board-specific models
- `ActionEncoderSquare` / `ActionEncoderHex` - Move encoding/decoding

**Model Families:**

**v2 (High Capacity):**

- `RingRiftCNN_v2` - Square boards (96 channels, 12 SE-ResBlocks)
- `HexNeuralNet_v2` - Hex boards (192 channels, 12 SE-ResBlocks)
- `RingRiftCNN_v2_Lite` / `HexNeuralNet_v2_Lite` - Memory-efficient variants

**v3 (Spatial Policy):**

- `RingRiftCNN_v3` - Spatial policy heads for better move encoding
- `HexNeuralNet_v3` - Spatial policy for hex boards

**v4 (NAS-Optimized):**

- `RingRiftCNN_v4` - Attention-based architecture (square only)
- `HexNeuralNet_v4` - NAS-optimized for hex boards

**GNN Architectures:**

- `GNNPolicyNet` - Pure graph neural network (~255K params)
- `HybridPolicyNet` - CNN-GNN hybrid (~15.5M params)

### 3. Gumbel MCTS (`gumbel_mcts_ai.py`)

Production-quality AlphaZero-style MCTS with Sequential Halving.

**Features:**

- Gumbel-Top-K sampling for efficient exploration
- GPU-accelerated batch evaluation (5-50x speedup)
- Shadow validation for GPU/CPU parity
- Completed Q-values for asymmetric visit counts

**Budget Tiers (from `gumbel_common.py`):**

- `GUMBEL_BUDGET_THROUGHPUT` (64) - Fast selfplay
- `GUMBEL_BUDGET_STANDARD` (150) - Normal games
- `GUMBEL_BUDGET_QUALITY` (800) - Tournament play
- `GUMBEL_BUDGET_ULTIMATE` (1600) - Maximum strength

**Environment Variables:**

- `RINGRIFT_GPU_GUMBEL_DISABLE=1` - Disable GPU acceleration
- `RINGRIFT_GPU_GUMBEL_SHADOW_VALIDATE=1` - Enable shadow validation

### 4. GPU Parallel Games (`gpu_parallel_games.py`)

GPU-accelerated parallel game execution for selfplay and CMA-ES.

**Performance:**

- 6-57x speedup on CUDA depending on GPU (A10: 6.5x, RTX 5090: 57x at batch 200)
- 10-100 games/sec on RTX 3090
- 50-500 games/sec on A100/RTX 5090
- CPU fallback for systems without CUDA

**Optimization Status (Dec 2025):**

- ~14 `.item()` calls remain in critical paths (down from 80+ after Dec 2025 optimizations)
  - `gpu_parallel_games.py`: 1 call (statistics only)
  - `gpu_move_generation.py`: 1 call
  - `gpu_move_application.py`: ~12 calls (attack moves, max_dist)
- Most hot paths fully vectorized; limited further speedup potential
- See `gpu_parallel_games.py` for optimization comments

**Key Classes:**

- `GPUParallelGames` - Main parallel game runner
- `BatchGameState` - Batched game state representation
- `create_gpu_games(board_type, num_games, device)` - Factory

**Note:** MPS (Apple Silicon) is currently 100x slower than CPU due to excessive synchronization. Use `device="cpu"` on Apple Silicon.

### 5. Heuristic AI (`heuristic_ai.py`)

Fast rule-based evaluation without neural networks.

**Features:**

- Weighted sum of 20+ hand-crafted features
- Configurable weights via `heuristic_weights.py`
- Parallel evaluation for large boards
- Territory detection with NumPy acceleration

**Evaluator Modules (`evaluators/`):**

- `MaterialEvaluator` - Ring and marker counts
- `PositionalEvaluator` - Board control and center influence
- `TacticalEvaluator` - Threats and captures
- `MobilityEvaluator` - Movement options
- `StrategicEvaluator` - Long-term planning
- `EndgameEvaluator` - Victory conditions

**Optimization Flags:**

- `RINGRIFT_USE_FAST_TERRITORY=true` - NumPy territory (2x faster)
- `RINGRIFT_USE_MOVE_CACHE=true` - Cache move generation
- `RINGRIFT_USE_PARALLEL_EVAL=true` - Parallel evaluation

### 6. EBMO Online Learner (`ebmo_online_learner.py`)

Continuous learning during gameplay without batch training.

**Features:**

- TD-Energy updates during play
- Rolling buffer of recent games
- Outcome-weighted contrastive loss
- Real-time weight updates

**Usage:**

```python
from app.ai.ebmo_online_learner import EBMOOnlineLearner

learner = EBMOOnlineLearner(network, device='cuda')
learner.record_transition(state, move, player, next_state)
metrics = learner.update_from_game(winner)
```

### 7. Unified AI (`universal_ai.py`)

Single AI that adapts strategy based on game configuration.

**Features:**

- Board-aware architecture selection
- Player-count adaptive search
- Automatic model loading
- Graceful fallback to heuristics

## Neural Network Architectures

### Architecture Comparison

| Architecture        | Params | Memory (FP32) | Board Types       | Key Features                |
| ------------------- | ------ | ------------- | ----------------- | --------------------------- |
| RingRiftCNN_v2      | 43.8M  | ~180 MB       | square8, square19 | SE-ResBlocks, high capacity |
| RingRiftCNN_v2_Lite | 12.2M  | ~50 MB        | square8, square19 | Memory efficient            |
| RingRiftCNN_v3      | 45.1M  | ~185 MB       | square8, square19 | Spatial policy heads        |
| RingRiftCNN_v4      | 51.3M  | ~210 MB       | square8, square19 | NAS-optimized, attention    |
| HexNeuralNet_v2     | 44.2M  | ~180 MB       | hex8, hexagonal   | Hex masking, SE blocks      |
| HexNeuralNet_v3     | 46.5M  | ~190 MB       | hex8, hexagonal   | Spatial policy for hex      |
| HexNeuralNet_v4     | 53.1M  | ~215 MB       | hex8, hexagonal   | NAS attention for hex       |
| GNNPolicyNet        | 255K   | ~1 MB         | all               | Graph neural network        |
| HybridPolicyNet     | 15.5M  | ~63 MB        | all               | CNN-GNN hybrid              |

### v2 Architecture (Production)

**Input:**

- 14 base channels × 4 history frames = 56 spatial channels
- 20 global features (rings in hand, territory counts, etc.)

**Spatial Channels:**

1. Per-player stack presence (4 channels)
2. Per-player marker presence (4 channels)
3. Stack height (normalized)
4. Cap height (normalized)
5. Collapsed territory
6. Territory ownership (3 channels)

**Global Features:**

1. Rings in hand (per player × 4)
2. Eliminated rings (per player × 4)
3. Territory count (per player × 4)
4. Line count (per player × 4)
5. Current player indicator
6. Game phase (early/mid/late)
7. Total rings in play
8. LPS threat indicator

**Architecture:**

- Convolutional stem (56 → 96/192 channels)
- 6-12 SE-ResBlocks with Squeeze-and-Excitation
- Global average pooling + global feature injection
- **Policy head:** FC layer (384 intermediate) → action logits
- **Value head:** FC layer (128 intermediate) → per-player win probabilities

**Memory Profile (FP32):**

- Model weights: ~150-180 MB
- With activations: ~350-380 MB per instance
- Two models + MCTS: ~18 GB total

### v3 Architecture (Spatial Policy)

Similar to v2 but with **spatial policy heads** that preserve spatial structure:

- Separate convolutional layers for each action type (placement, movement, etc.)
- Better move discrimination for spatially-correlated actions
- Slight increase in parameters (~45M) but better policy accuracy

### v4 Architecture (NAS-Optimized)

**Key Improvements:**

- Multi-head self-attention layers
- Residual connections with layer normalization
- Discovered via Neural Architecture Search
- Best for large boards (19×19) and complex positions

**Architecture:**

- 8-12 attention-augmented residual blocks
- Channel attention (SE) + spatial attention
- Position-aware encoding with learned embeddings
- Higher parameter count (~51M) justified by performance

### GNN Architecture (Graph Neural Network)

**When to use:** Naturally handles hexagonal boards and territory connectivity.

**Architecture:**

- Input: Graph from `board_to_graph()` (nodes = cells, edges = adjacency)
- Encoder: GraphSAGE or GAT layers (3-5 layers)
- Global pooling: Attention-weighted readout
- Heads: Per-node policy + global value

**Advantages:**

- No masking needed for hex boards
- Natural territory modeling via message passing
- Better generalization to different board sizes

**Requirements:** PyTorch Geometric (`torch-geometric`, `torch-scatter`, `torch-sparse`)

### Hybrid CNN-GNN

**Best of both worlds:**

- CNN backbone for fast local pattern recognition
- GNN refinement for connectivity and territory
- Validated by MDPI 2024 research on Go

**Architecture Flow:**

1. CNN extracts spatial features
2. Convert features to graph
3. GNN message passing refines features
4. Fusion layer combines CNN + GNN
5. Policy and value heads

## Usage Examples

### Creating AI Players

```python
from app.ai.factory import create_ai_from_difficulty, AIFactory, AIType
from app.models import AIConfig, BoardType

# Simple: Create from difficulty level
ai = create_ai_from_difficulty(
    difficulty=8,
    player_number=1,
    board_type=BoardType.HEX8,
    num_players=2
)

# Advanced: Explicit configuration
config = AIConfig(
    difficulty=8,
    think_time=5000,
    randomness=0.0,
    rng_seed=42
)
ai = AIFactory.create(
    ai_type=AIType.GUMBEL_MCTS,
    player_number=1,
    config=config
)

# Select a move
move = ai.select_move(game_state)
```

### Using Neural Networks

```python
from app.ai.neural_net import create_model_for_board
from app.models import BoardType
import torch

# Create model for board type
model = create_model_for_board(
    board_type=BoardType.SQUARE8,
    num_players=2,
    memory_tier="high"  # or "low", "v3-high", "v4", "gnn", "hybrid"
)

# Load trained weights
checkpoint = torch.load("models/canonical_sq8_2p.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use with NeuralNetAI wrapper
from app.ai.neural_net import NeuralNetAI
from app.models import AIConfig

neural_ai = NeuralNetAI(
    player_number=1,
    config=AIConfig(difficulty=7),
    model_path="models/canonical_sq8_2p.pth"
)
move = neural_ai.select_move(game_state)
```

### GPU Parallel Selfplay

```python
from app.ai.gpu_parallel_games import create_gpu_games
import torch

# Create batch of games
games = create_gpu_games(
    board_type="square8",
    num_players=2,
    num_games=128,
    device="cuda",  # or "cpu", "mps"
    enable_shadow_validation=False  # Enable for GPU/CPU parity checks
)

# Run games with heuristic policy
max_turns = 1000
for turn in range(max_turns):
    active_mask = games.state.get_active_mask()
    if active_mask.sum() == 0:
        break
    games.step()  # All games advance one turn

# Get results
winners = games.state.winner.cpu().numpy()
move_counts = games.state.move_count.cpu().numpy()

print(f"Completed {len(winners)} games")
print(f"Average moves: {move_counts.mean():.1f}")
print(f"Player 1 wins: {(winners == 1).sum()}")
print(f"Player 2 wins: {(winners == 2).sum()}")
```

### Gumbel MCTS Configuration

```python
from app.ai.gumbel_mcts_ai import GumbelMCTSAI
from app.ai.gumbel_common import (
    GUMBEL_BUDGET_THROUGHPUT,
    GUMBEL_BUDGET_STANDARD,
    GUMBEL_BUDGET_QUALITY,
    GUMBEL_BUDGET_ULTIMATE
)
from app.models import AIConfig

# High-quality tournament play
config = AIConfig(
    difficulty=10,
    think_time=10000,
    randomness=0.0,
    mcts_simulations=GUMBEL_BUDGET_QUALITY  # 800 simulations
)
ai = GumbelMCTSAI(
    player_number=1,
    config=config,
    model_path="models/canonical_hex8_2p.pth"
)

# Fast selfplay
fast_config = AIConfig(
    difficulty=6,
    mcts_simulations=GUMBEL_BUDGET_THROUGHPUT  # 64 simulations
)
fast_ai = GumbelMCTSAI(
    player_number=1,
    config=fast_config,
    model_path="models/canonical_hex8_2p.pth"
)
```

### Custom Heuristic Weights

```python
from app.ai.heuristic_ai import HeuristicAI
from app.ai.heuristic_weights import HeuristicWeights
from app.models import AIConfig

# Load default weights
weights = HeuristicWeights.load_default()

# Or load from file
weights = HeuristicWeights.load_from_file("weights/custom.json")

# Create AI with custom weights
config = AIConfig(difficulty=5)
ai = HeuristicAI(
    player_number=1,
    config=config
)
ai.weights = weights  # Override default weights

# Save weights after tuning
weights.save_to_file("weights/tuned.json")
```

### EBMO Online Learning

```python
from app.ai.ebmo_online_learner import EBMOOnlineLearner, EBMOOnlineConfig
from app.ai.ebmo_network import EBMONetwork

# Create network and learner
network = EBMONetwork(board_size=8, num_players=2)
config = EBMOOnlineConfig(
    learning_rate=1e-5,
    buffer_size=20,
    td_lambda=0.9
)
learner = EBMOOnlineLearner(network, device='cuda', config=config)

# During gameplay
for move_num in range(100):
    state = get_current_state()
    move = select_move(state)
    next_state = apply_move(state, move)
    learner.record_transition(state, move, player=1, next_state=next_state)

# After game ends
winner = get_winner()
metrics = learner.update_from_game(winner)
print(f"Loss: {metrics['loss']:.4f}, Energy: {metrics['mean_energy']:.4f}")
```

## Component Relationships

### Flow Diagram: AI Decision Making

```
GameState
    ↓
BaseAI.select_move()
    ↓
    ├─→ RandomAI → random.choice(valid_moves)
    │
    ├─→ HeuristicAI → Evaluators → Weighted sum → Best move
    │       ├─→ MaterialEvaluator
    │       ├─→ PositionalEvaluator
    │       ├─→ TacticalEvaluator
    │       └─→ StrategicEvaluator
    │
    ├─→ PolicyOnlyAI → NeuralNet → Sample from policy
    │
    ├─→ MinimaxAI → Minimax search → HeuristicAI eval
    │
    ├─→ MCTSAI → MCTS tree → NeuralNet eval → Best move
    │
    └─→ GumbelMCTSAI → Gumbel MCTS tree
            ├─→ Gumbel-Top-K sampling
            ├─→ Sequential Halving
            ├─→ Batch NN evaluation (GPU)
            └─→ Completed Q-values
```

### Neural Network Pipeline

```
GameState
    ↓
Feature Extraction (neural_net/encoding)
    ├─→ ActionEncoderSquare (square boards)
    └─→ ActionEncoderHex (hex boards)
    ↓
Model Selection (model_factory)
    ├─→ RingRiftCNN_v2/v3/v4 (square)
    ├─→ HexNeuralNet_v2/v3/v4 (hex)
    ├─→ GNNPolicyNet (graph)
    └─→ HybridPolicyNet (CNN+GNN)
    ↓
Forward Pass
    ├─→ CNN Backbone (ResBlocks/Attention)
    ├─→ Global Average Pooling
    ├─→ Global Feature Injection
    ├─→ Policy Head → Action logits (policy_size,)
    └─→ Value Head → Win probabilities (num_players,)
    ↓
Output: (policy_logits, value_vector)
```

### GPU Parallel Games Flow

```
BatchGameState (N games)
    ↓
generate_moves_batch()
    ├─→ generate_placement_moves_batch()
    ├─→ generate_movement_moves_batch()
    ├─→ generate_capture_moves_batch()
    └─→ generate_recovery_moves_batch()
    ↓
BatchMoves (N × max_moves)
    ↓
select_moves_vectorized()
    ├─→ Heuristic policy (gpu_heuristic)
    ├─→ Neural policy (batch_eval)
    └─→ Random policy
    ↓
Selected moves (N,)
    ↓
apply_moves_batch()
    ├─→ apply_placement_moves_batch()
    ├─→ apply_movement_moves_batch()
    ├─→ apply_capture_moves_batch()
    └─→ apply_recovery_moves_vectorized()
    ↓
Updated BatchGameState
    ↓
detect_lines_vectorized()
    ↓
compute_territory_batch()
    ↓
Check terminal conditions → Update winners
```

### Training Pipeline Integration

```
Selfplay (scripts/selfplay.py)
    ↓
GPU Parallel Games OR GumbelMCTSAI
    ↓
Game Replay Database (.db)
    ↓
export_replay_dataset.py
    ↓
Training NPZ files
    ↓
app.training.train
    ├─→ DataLoader (batched examples)
    ├─→ Forward pass (neural_net)
    ├─→ Loss computation (neural_losses)
    │   ├─→ Policy loss (cross-entropy)
    │   └─→ Value loss (multi-player rank distribution)
    ├─→ Optimizer step
    └─→ Validation
    ↓
Model Checkpoint (.pth)
    ↓
Gauntlet Evaluation (app.training.game_gauntlet)
    ↓
Auto-Promotion (scripts/auto_promote.py)
    ↓
Production Model (models/canonical_*)
```

## Performance Considerations

### GPU vs CPU

**GPU Acceleration is beneficial for:**

- Neural network inference (10-50x faster)
- Parallel game simulation (6.5x faster on CUDA)
- Batch MCTS evaluation (5-50x faster)

**CPU is better for:**

- Heuristic evaluation (no GPU overhead)
- Small batch sizes (< 8 games)
- Apple Silicon (MPS has synchronization overhead)

**CUDA Performance:**

- Production-ready with 100% GPU/CPU parity
- Optimal batch sizes: 64-256 games
- RTX 3090: 10-100 games/sec
- A100: 50-500 games/sec

**MPS (Apple Silicon) Note:**

- Currently 100x slower than CPU for game simulation
- Remaining `.item()` calls and MPS synchronization overhead still cause slowdowns
- Use `device="cpu"` on Apple Silicon for game simulation
- Neural network inference on MPS is still beneficial

### Memory Management

**Model Caching:**

- Models are cached in `_MODEL_CACHE` to avoid repeated loading
- Use `clear_model_cache()` to free GPU memory
- Multiple models can share GPU memory on high-memory systems

**Batch Size Guidelines:**

- RTX 3090 (24GB): batch_size=256-512
- A10 (23GB): batch_size=256
- GH200 (96GB): batch_size=1024+
- CPU: batch_size=32-128

**Memory Tiers:**

- `high` (default): Full v2 models, 96GB target
- `low`: Lite models, 48GB target
- `v3-high/v3-low`: Spatial policy variants
- `gnn`: Lightweight GNN (~1 MB)

### Optimization Flags

**Heuristic AI:**

```bash
export RINGRIFT_USE_FAST_TERRITORY=true       # 2x faster (default: true)
export RINGRIFT_USE_MOVE_CACHE=true           # Cache moves (default: true)
export RINGRIFT_USE_PARALLEL_EVAL=true        # Parallel eval (default: auto)
export RINGRIFT_PARALLEL_MIN_MOVES=50         # Threshold for parallel
```

**Gumbel MCTS:**

```bash
export RINGRIFT_GPU_GUMBEL_DISABLE=1          # Disable GPU batching
export RINGRIFT_GPU_GUMBEL_SHADOW_VALIDATE=1  # Enable parity checks
export RINGRIFT_GPU_TREE_SHADOW_RATE=0.05     # Shadow validate 5% of searches
```

**GPU Parallel Games:**

```bash
export RINGRIFT_GPU_SHADOW_VALIDATE=1         # Enable shadow validation
export RINGRIFT_GPU_SHADOW_RATE=0.05          # Validate 5% of games
```

### Benchmark Results (Dec 2025)

**2-Player Games:**

- Gumbel+NN >> MCTS+NN > Descent+NN >> Heuristic

**3-Player Games:**

- Gumbel+NN >> (MaxN ≈ MCTS+NN ≈ Descent+NN) >> Heuristic

**4-Player Games:**

- Gumbel+NN >> MaxN > MCTS+NN >> Heuristic

**Selfplay Speed:**

- CPU (Heuristic): 1-2 games/sec
- CPU (Gumbel+NN): 0.1-0.2 games/sec
- GPU (Parallel Heuristic): 10-100 games/sec
- GPU (Gumbel+NN batched): 5-20 games/sec

## Related Modules

- **`app/training/`** - Model training pipeline
  - `train.py` - Main training script
  - `selfplay_runner.py` - Unified selfplay base class
  - `temperature_scheduling.py` - Exploration schedules
  - `online_learning.py` - EBMO online learning

- **`app/training/game_gauntlet.py`** - AI evaluation framework
  - `run_baseline_gauntlet()` - Head-to-head baselines

- **`app/rules/`** - Python game rules (mirrors TypeScript)
  - `engine/` - Core game logic
  - `factory.py` - Rules engine factory

- **`scripts/`** - CLI tools
  - `selfplay.py` - Unified selfplay entry point
  - `export_replay_dataset.py` - Convert .db to .npz
  - `auto_promote.py` - Automated model promotion
  - `quick_gauntlet.py` - Baseline gauntlet CLI
  - `run_training_loop.py` - Full training pipeline

## See Also

- **Main project docs:** `/Users/armand/Development/RingRift/CLAUDE.md`
- **AI service docs:** `/Users/armand/Development/RingRift/ai-service/CLAUDE.md`
- **Architecture notes:** `/Users/armand/Development/RingRift/ai-service/app/ai/neural_net_analysis.md`
