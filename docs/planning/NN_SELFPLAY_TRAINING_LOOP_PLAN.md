# Plan: Closing the NN Self-Play Training Loop

## Problem Statement

The neural network training pipeline is bottlenecked on generating high-quality Gumbel MCTS + NN games. While GPU heuristic self-play achieves 10-500 games/sec, NN-guided MCTS games are significantly slower because:

1. **MCTS tree search runs in Python** - Even with GPU batch NN evaluation, the tree traversal is sequential
2. **Each move requires 100-150 simulations** - With 16 candidate actions, this means ~1,110 NN evaluations per move
3. **No tree reuse across moves** - Each move starts fresh, discarding previous search
4. **Self-play loop not closing** - Training on static datasets rather than iteratively regenerating with improved models

## Current Infrastructure Analysis

### What Exists (and works well):

| Component              | File                                | Status                                |
| ---------------------- | ----------------------------------- | ------------------------------------- |
| Gumbel MCTS            | `app/ai/gumbel_mcts_ai.py`          | Production - 150 sims default         |
| GPU Batch NN Eval      | `gumbel_mcts_ai.py:720-816`         | Implemented - batches leaf evals      |
| Parallel Selfplay      | `app/training/parallel_selfplay.py` | Supports Gumbel engine                |
| GPU Heuristic Selfplay | `app/ai/gpu_parallel_games.py`      | 10-500 games/sec                      |
| Training Loop          | `app/training/train_loop.py`        | Iterative self-play → train → promote |
| Curriculum Training    | `app/training/curriculum.py`        | Multi-generation with promotion       |

### Key Bottleneck:

```
Heuristic GPU selfplay: 10-500 games/sec (fast)
Gumbel MCTS + NN:       0.1-1 games/sec (100-1000x slower)
```

The NN training data quality is limited because we can't generate enough NN+MCTS games to provide good policy targets.

## Investigation Findings

### 1. Batch NN Evaluation IS Implemented

From `gumbel_mcts_ai.py:1073-1170`:

- `LeafEvaluationBuffer` collects all leaf states across simulations
- Single batch forward pass per Sequential Halving phase
- ~4-5 GPU calls per move (one per phase)

### 2. Parallel Game Generation EXISTS

From `parallel_selfplay.py:44-80`:

- `ProcessPoolExecutor` with configurable workers
- Supports `engine="gumbel"` mode
- Each worker initializes own AI (avoids pickle issues)
- Default: 64 simulations, 16 top-k actions

### 3. Training Loop IS Closed (at algorithm level)

From `train_loop.py:88-150`:

- Self-play → Train → Tournament → Promotion cycle
- GPU parallel mode uses heuristic (fast but no NN policy)
- CPU mode uses DescentAI (slow but higher quality)

## Proposed Solution: Multi-Process Gumbel MCTS at Scale

### Phase 1: Maximize Parallel NN-MCTS Throughput (High Impact)

**Goal:** Generate 10-50x more Gumbel MCTS games by fully utilizing multi-process parallelism

**Actions:**

1. **Enable multi-GPU parallel Gumbel selfplay**
   - Location: `scripts/run_parallel_self_play.py`
   - Use `--engine gumbel --num-workers N` where N = CPU cores
   - Each worker gets own CUDA stream for NN inference

2. **Reduce simulation budget for initial data**
   - Current: 150 simulations per move
   - Proposed: 64 simulations for rapid iteration, 150 for refinement
   - Rationale: AlphaZero used 800, but found 50-100 works for smaller games

3. **Batch games within workers**
   - Current: 1 game per worker
   - Proposed: 4-8 games per worker with shared NN model
   - Benefit: NN batch utilization improves from 1x to 4-8x

### Phase 2: Close the Iterative Training Loop (Critical)

**Goal:** Automate the self-play → train → promote → repeat cycle

**Actions:**

1. **Create unified training orchestrator script**
   - Monitor games generated per config
   - Trigger training when threshold reached (e.g., 10K new games)
   - Run tournament evaluation
   - Promote if win rate > 55%
   - Repeat with new model

2. **Integrate with P2P cluster**
   - Distribute Gumbel selfplay across GPU nodes
   - Each node generates games with current best model
   - Central coordinator collects and triggers training

3. **Mixed data strategy**
   - 75% Gumbel MCTS + NN games (high quality policy)
   - 25% GPU heuristic games (position diversity)
   - Rationale: Heuristic provides breadth, NN provides depth

### Phase 3: Optimize Gumbel MCTS Speed (Medium Impact)

**Goal:** 2-5x speedup per Gumbel MCTS game

**Actions:**

1. **Tree reuse across moves**
   - Keep subtree from chosen action
   - Saves ~30% of simulations (empirical from chess engines)

2. **Reduce top-k sampling**
   - Current: 16 actions
   - Proposed: 8 actions for sq8, 12 for hex8
   - Rationale: Smaller boards have fewer truly different top moves

3. **Async NN evaluation**
   - `app/ai/async_nn_eval.py` exists but not integrated
   - Queue leaf states across multiple games
   - Single large batch inference

4. **FP16 inference by default**
   - Already supported in `gpu_parallel_games.py:413`
   - Ensure Gumbel MCTS uses it consistently

### Phase 4: Training Data Quality (Medium Impact)

**Goal:** Ensure training data produces improving models

**Actions:**

1. **Verify value target accuracy**
   - Current: Win/loss/draw from game outcome
   - Check: Are multi-player values correct?
   - File: `app/training/generate_data.py`

2. **Add MCTS policy temperature**
   - Current: Visit count fractions
   - Proposed: Softmax with temperature for early moves
   - Rationale: More exploration in training data

3. **Position deduplication**
   - Track zobrist hashes per game
   - Warn if >10% duplicate positions
   - Already in `generate_data.py:DataQualityTracker`

## Implementation Order

1. **[IMMEDIATE]** Run parallel Gumbel selfplay with current infrastructure

   ```bash
   python scripts/run_parallel_self_play.py \
     --num-games 1000 \
     --num-workers 16 \
     --engine gumbel \
     --board square8 \
     --output-dir data/selfplay/gumbel_sq8_2p
   ```

2. **[IMMEDIATE]** Verify training loop closure
   - Check `train_loop.py` actually uses Gumbel data
   - Confirm tournament evaluation works

3. **[DAY 1]** Create cluster-wide Gumbel generation script
   - SSH to each node, run parallel Gumbel
   - Collect results to central location

4. **[DAY 2]** Implement mixed data training
   - Combine Gumbel MCTS + heuristic games
   - Train and evaluate

5. **[DAY 3]** Add tree reuse optimization
   - Modify `gumbel_mcts_ai.py` to preserve subtree
   - Benchmark improvement

6. **[WEEK 1]** Full automated pipeline
   - Continuous self-play on cluster
   - Automatic training triggers
   - Model promotion and deployment

## Success Metrics

| Metric                  | Current | Target |
| ----------------------- | ------- | ------ |
| Gumbel games/hour/node  | ~100    | 1,000+ |
| Training iterations/day | ~1-2    | 5-10   |
| Model ELO vs heuristic  | +50-100 | +300+  |
| Policy accuracy         | 26%     | 50%+   |

## Key Files to Modify

1. `scripts/run_parallel_self_play.py` - Add multi-game batching per worker
2. `app/training/parallel_selfplay.py` - Reduce simulation budget option
3. `app/ai/gumbel_mcts_ai.py` - Add tree reuse
4. `app/training/train_loop.py` - Integrate Gumbel data generation
5. `scripts/p2p_orchestrator.py` - Add Gumbel selfplay job type

## Risks and Mitigations

| Risk                                | Mitigation                                 |
| ----------------------------------- | ------------------------------------------ |
| NN inference bottleneck             | Batch across games, use FP16               |
| Memory exhaustion with many workers | Limit workers to CPU_COUNT-2               |
| Training data quality issues        | Monitor position diversity, value accuracy |
| Model divergence                    | Rollback manager already exists            |

## Ready to Implement

Starting with Phase 1: Run parallel Gumbel selfplay to establish baseline throughput and identify actual bottlenecks.
