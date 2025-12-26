# GPU MCTS Integration Plan

## Executive Summary

RingRift has four GPU-accelerated Gumbel MCTS implementations. This document outlines how to integrate them to maximize:

1. **Training throughput** - Faster selfplay = more training data = better models
2. **Human player experience** - Lower latency = more responsive AI = better UX

## The Four Implementations

### 1. MultiTreeMCTS (tensor_gumbel_tree.py)

**Purpose**: Maximum throughput selfplay for training data generation

- Full GPU-resident tree (Structure of Arrays)
- Processes N games simultaneously
- 6,671x faster than CPU baseline in batch mode
- Best for: Cluster selfplay, training data generation

### 2. GumbelMCTSGPU (gumbel_mcts_gpu.py)

**Purpose**: GPU-accelerated simulation with CPU tree

- Uses BatchGameState for GPU playouts
- Tree remains on CPU for flexibility
- 2-3x speedup over pure CPU
- Best for: Single-game high-quality search

### 3. BatchedGumbelMCTS (batched_gumbel_mcts.py)

**Purpose**: Batch NN evaluations across multiple games

- CPU tree, batched GPU NN calls
- 3-4x speedup through larger batches
- Currently used by EventDrivenSelfplay
- Best for: Production multi-game scenarios

### 4. HybridNNValuePlayer (hybrid_gpu.py)

**Purpose**: Fast human-facing gameplay

- Heuristic generates candidates, NN ranks
- 5-10x faster than full MCTS
- Trades lookahead depth for speed
- Best for: Low-latency human games

## Integration Matrix

| Use Case                | Current           | Recommended             | Speedup             |
| ----------------------- | ----------------- | ----------------------- | ------------------- |
| **Selfplay Training**   | BatchedGumbelMCTS | MultiTreeMCTS           | 3-4x → 10-20x       |
| **Human vs AI (D8-11)** | GumbelMCTSAI      | HybridNNValuePlayer     | 10x faster response |
| **Gauntlet Eval**       | GumbelMCTSAI      | BatchedGumbelMCTS batch | 3-4x                |
| **Sandbox Testing**     | GumbelMCTSAI      | Configurable (all 4)    | Variable            |

## Implementation Plan

### Phase 1: Factory System Upgrade

Extend `app/ai/factory.py` to expose GPU variants:

```python
class AIFactory:
    @staticmethod
    def create_mcts(
        board_type: BoardType,
        num_players: int = 2,
        player_number: int = 1,
        mode: str = "standard",  # standard, gpu, batch, hybrid, tensor
        device: str = "cuda",
        simulation_budget: int = 800,
        neural_net: NeuralNetAI | None = None,
    ) -> BaseAI:
        """Create MCTS AI with specified acceleration mode.

        Modes:
        - standard: GumbelMCTSAI (CPU, single game)
        - gpu: GumbelMCTSGPU (GPU simulation, single game)
        - batch: BatchedGumbelMCTS (batched NN, multi-game)
        - hybrid: HybridNNValuePlayer (fast heuristic+NN)
        - tensor: GPUGumbelMCTS (full GPU tree, single game)
        """
```

### Phase 2: Difficulty Ladder Enhancement

Update difficulty profiles for optimal UX:

| Difficulty | AI Type                 | Config           | Latency Target |
| ---------- | ----------------------- | ---------------- | -------------- |
| D1-D4      | Heuristic               | -                | <100ms         |
| D5-D6      | HeuristicNN             | Depth 2-3        | <200ms         |
| D7         | **HybridNNValuePlayer** | Top-8 candidates | <500ms         |
| D8-D9      | GumbelMCTSAI            | Budget 200-400   | <2s            |
| D10        | GumbelMCTSAI            | Budget 800-1600  | <5s            |

Note: D11 is reserved for internal benchmarking and is not exposed via the public API.

Key change: Insert HybridNNValuePlayer at D7 for fast, strong play.

### Phase 3: Selfplay Pipeline Optimization

Make MultiTreeMCTS the default for selfplay:

```python
# In event_driven_selfplay.py
class EventDrivenSelfplay:
    def __init__(
        self,
        ...
        use_gpu_mcts: bool = True,  # Change default to True
        gpu_device: str = "cuda",
        gpu_eval_mode: str = "heuristic",  # Fast mode for throughput
    ):
```

Expected improvement: 3-4x more training games per hour.

### Phase 4: Batch Move Endpoint

Add FastAPI endpoint for batch move requests:

```python
@app.post("/ai/moves_batch")
async def get_moves_batch(
    game_states: list[GameState],
    difficulty: int = 8,
    max_batch_size: int = 32,
) -> list[Move]:
    """Get moves for multiple games in a single batched call."""

    mcts = create_batched_gumbel_mcts(...)
    return mcts.select_moves_batch(game_states)
```

Use cases:

- Gauntlet evaluation (run N games in parallel)
- Multi-game spectator mode
- Tournament simulations

### Phase 5: Sandbox Configurability

Add AI mode selection to sandbox UI:

```typescript
// SandboxSettings.tsx
interface AISettings {
  difficulty: 1-11;
  mode: "standard" | "gpu" | "hybrid" | "batch";
  showThinking: boolean;
  latencyTarget: number;  // ms
}
```

Backend `/sandbox/ai/config` endpoint to set mode.

## Enhancements Per Implementation

### MultiTreeMCTS Enhancements

1. **Vectorize value aggregation** (lines 1211-1217)
   - Current: Python dict loop for value aggregation
   - Fix: Use scatter_add on GPU tensor
   - Impact: 10-20% speedup

2. **Pre-allocate simulation state lists**
   - Current: Append to list in loop (lines 1178-1198)
   - Fix: Pre-size based on expected simulations
   - Impact: 5-10% speedup

3. **Fuse BatchGameState conversions**
   - Current: Convert per-phase (line 1204)
   - Fix: Keep states on GPU between phases
   - Impact: 15-25% speedup for GPU device

### GumbelMCTSGPU Enhancements

1. **Merge with TensorGumbelTree**
   - GumbelMCTSGPU is a subset of TensorGumbelTree
   - Recommend deprecating GumbelMCTSGPU
   - Keep GPUSimulationEngine as reusable component

2. **Add hybrid eval option**
   - Use GPU heuristic for early simulations
   - Switch to NN for final phases
   - Reduces NN calls by 50-70%

### BatchedGumbelMCTS Enhancements

1. **Adaptive batch sizing**
   - Current: Fixed batch size
   - Fix: Grow batch based on available games
   - Impact: Better GPU utilization

2. **Early termination for dominated actions**
   - Current: Run full budget for all actions
   - Fix: Prune clearly losing actions early
   - Impact: 20-30% fewer simulations

### HybridNNValuePlayer Enhancements

1. **Add to difficulty ladder**
   - Currently not integrated
   - Add as D7 with 500ms latency target
   - Best balance of speed and strength

2. **Configurable candidate count**
   - Current: Hardcoded top-K
   - Fix: Scale with difficulty (D7: K=8, D8: K=16)
   - Impact: Tunable speed/quality tradeoff

3. **Caching for repeated positions**
   - Cache NN evaluations for common positions
   - Useful for opening book positions
   - Impact: 2-3x faster for cached positions

## Deployment Recommendations

### Cluster Selfplay (GH200/H100 nodes)

```bash
# Use GPU parallel selfplay (bulk tier) with max batch size
PYTHONPATH=. python scripts/run_gpu_selfplay.py \
    --board square8 \
    --num-players 2 \
    --num-games 1000 \
    --batch-size 64 \
    --engine-mode nnue-guided \
    --output-dir data/selfplay/gpu_square8_2p
```

### Production Server (human games)

```bash
# Default to standard MCTS, enable hybrid for D7
export RINGRIFT_HYBRID_D7=1
export RINGRIFT_GPU_MCTS_DISABLE=1  # Avoid GPU overhead for single games
```

### Gauntlet Evaluation

```bash
# Quick baseline gauntlet
python scripts/quick_gauntlet.py \
    --model models/latest.pth \
    --board-type hex8 \
    --num-players 2 \
    --games 30
```

## Success Metrics

| Metric              | Current      | Target       | How to Measure     |
| ------------------- | ------------ | ------------ | ------------------ |
| Selfplay throughput | 10 games/min | 50 games/min | Cluster dashboard  |
| D8-D10 latency      | 3-5s         | 1-3s         | Response time logs |
| D7 latency          | N/A          | <500ms       | Response time logs |
| Gauntlet speed      | 1 game/min   | 10 games/min | Gauntlet duration  |

## Implementation Status (Dec 2024)

### Phase 1: Factory System Upgrade ✅ COMPLETED

The unified factory method is now available:

```python
from app.ai.factory import create_mcts

# Standard CPU-based Gumbel MCTS (human games D8+)
mcts = create_mcts("square8", mode="standard", simulation_budget=800)
move = mcts.select_move(game_state)

# Fast hybrid for responsive human games (D7)
hybrid = create_mcts("hex8", mode="hybrid", top_k=8)
move = hybrid.select_move(game_state, valid_moves)

# GPU tensor for selfplay training (6,671x speedup)
tensor_mcts = create_mcts("square8", mode="tensor", device="cuda", batch_size=64)
moves, policies = tensor_mcts.search_batch(game_states, neural_net)

# Batched MCTS for gauntlet evaluation
batch_mcts = create_mcts("hex8", mode="batch", batch_size=16)
moves = batch_mcts.select_moves_batch(game_states)
```

Available modes:
| Mode | Class | Use Case | Latency |
|------|-------|----------|---------|
| `standard` | GumbelMCTSAI | Human games D8+ | ~19s (budget=50) |
| `gpu` | GPUGumbelMCTS | Single game, heuristic | ~15ms |
| `batch` | BatchedGumbelMCTS | Multi-game, NN batched | 3-4x over single |
| `hybrid` | HybridNNValuePlayer | Fast human games D7 | ~900ms |
| `tensor` | MultiTreeMCTS | Selfplay training | ~7ms/game |

### Phase 2: Difficulty Ladder Enhancement ✅ COMPLETED

HybridNN is now available at D7 via environment variable:

```bash
# Enable fast hybrid mode for D7 human games
export RINGRIFT_USE_HYBRID_D7=1
```

Or use directly:

```python
from app.ai.factory import AIFactory
from app.models import AIType

ai = AIFactory.create(AIType.HYBRID_NN, player_number=1, config=config)
```

### Phase 3: Selfplay Pipeline ✅ ALREADY IMPLEMENTED

EventDrivenSelfplay already supports GPU MCTS:

```python
manager = EventDrivenSelfplay(
    use_gpu_mcts=True,
    gpu_device="cuda",
    gpu_eval_mode="heuristic",
)
```

### Phase 4: Batch Move Endpoint ✅ COMPLETED

FastAPI endpoint for batch move requests is now available:

```python
# POST /ai/moves_batch
# Request body:
{
    "game_states": [...],  # List of serialized game states
    "difficulty": 8,
    "mode": "batch"  # or "tensor" for GPU-accelerated
}

# Response:
{
    "moves": [...],
    "mode_used": "batch",
    "total_time_ms": 156.3
}
```

Benchmark results: 25.6 games/sec throughput with `mode="tensor"`.

### Phase 5: Sandbox Configurability ⏳ DEFERRED

AI mode selection for sandbox UI requires cross-stack changes (client → server → AI service).
Deferred pending API design review.

### Phase 6: MultiTreeMCTS Optimizations ✅ COMPLETED (Dec 2024)

Performance optimizations applied to `_parallel_sequential_halving`:

1. **Vectorized value aggregation** - Using `torch.scatter_add` instead of Python dict loop
2. **Pre-collect unique states** - Build states once, expand with `.repeat()` for sims_per_action
3. **Tensor-based index construction** - Avoid Python list → tensor conversion

Benchmark results (CPU, square8_2p):
| Batch Size | Budget | Total Time | Per-Game | Games/sec |
|------------|--------|------------|----------|-----------|
| 4 | 32 | 73ms | 18ms | 54.6 |
| 16 | 32 | 233ms | 15ms | 68.8 |
| 64 | 32 | 994ms | 16ms | 64.4 |
| 64 | 64 | 2619ms | 41ms | 24.4 |

### Phase 7: Hybrid Mode Latency ✅ COMPLETED (Dec 2024)

Optimized HybridNNValuePlayer to achieve <200ms latency:

1. **Move subsampling** - Evaluate max 4×top_k moves, prioritizing captures/placements
2. **Sequential heuristic evaluation** - More efficient than GPU batch for candidate scoring

Latency improvement: 900ms → 91-178ms (5-10x speedup)

## Code Changes Summary

| File                                    | Changes                                  | Status      |
| --------------------------------------- | ---------------------------------------- | ----------- |
| `app/ai/factory.py`                     | Add `create_mcts()` with mode parameter  | ✅ Done     |
| `app/ai/hybrid_gpu.py`                  | Add `HybridNNAI` wrapper + optimizations | ✅ Done     |
| `app/ai/tensor_gumbel_tree.py`          | Vectorized aggregation + pre-allocation  | ✅ Done     |
| `app/models/core.py`                    | Add `AIType.HYBRID_NN` and config fields | ✅ Done     |
| `app/training/event_driven_selfplay.py` | GPU MCTS integration                     | ✅ Done     |
| `app/main.py`                           | Add `/ai/moves_batch` endpoint           | ✅ Done     |
| `src/client/components/sandbox/`        | AI mode selector                         | ⏳ Deferred |
| `scripts/run_gpu_selfplay.py`           | Selfplay script                          | ✅ Done     |

## Production Deployment Configuration

### Cluster Selfplay (GH200/H100 nodes)

For maximum training throughput, use MultiTreeMCTS with large batch sizes:

```bash
# Environment setup
export PYTHONPATH=/path/to/ai-service
export CUDA_VISIBLE_DEVICES=0

# Run GPU-accelerated selfplay
python scripts/run_gpu_selfplay.py \
    --board square8 \
    --num-players 2 \
    --num-games 1000 \
    --batch-size 64 \
    --quality-tier hybrid \
    --output-db data/games/selfplay_gpu.db
```

Expected throughput: ~50-70 games/sec on GH200.

### Production Server (Human Games)

For human-facing games, prioritize latency:

```bash
# D1-D6: Standard heuristics (no GPU)
export RINGRIFT_GPU_MCTS_DISABLE=1

# D7: Fast hybrid mode (~150ms latency)
export RINGRIFT_USE_HYBRID_D7=1
export RINGRIFT_HYBRID_TOP_K=8

# D8+: Standard MCTS (GPU optional but not recommended for latency)
# Uses GumbelMCTSAI by default
```

### Gauntlet Evaluation

For parallel model evaluation:

```bash
python scripts/quick_gauntlet.py \
    --model models/latest.pth \
    --board-type hex8 \
    --num-players 2 \
    --games 100 \
    --parallel
```

Expected throughput: ~10 games/min with batched evaluation.

### Environment Variables Reference

| Variable                    | Default | Description                           |
| --------------------------- | ------- | ------------------------------------- |
| `RINGRIFT_GPU_MCTS_DISABLE` | 0       | Set to 1 to disable GPU MCTS globally |
| `RINGRIFT_USE_HYBRID_D7`    | 0       | Set to 1 to use HybridNN at D7        |
| `RINGRIFT_HYBRID_TOP_K`     | 8       | Number of candidate moves for hybrid  |
| `RINGRIFT_TRACE_DEBUG`      | 0       | Enable debug tracing for MCTS         |

### Hardware Recommendations

| Use Case          | Recommended GPU      | VRAM  | Notes                     |
| ----------------- | -------------------- | ----- | ------------------------- |
| Selfplay Training | GH200/H100           | 80GB+ | Batch size 64-128         |
| Gauntlet Eval     | Any CUDA-capable     | 8GB+  | Batch size 16-32          |
| Human Games       | None (CPU preferred) | -     | GPU adds latency overhead |
