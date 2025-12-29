# Plan: Closing the Self-Play Loop & Addressing Bottlenecks

## Problem Statement

Despite having good infrastructure and techniques, NNUE models barely outperform heuristic AI because:

1. **The self-play loop isn't closing** - training produces models but selfplay doesn't automatically use them
2. **NN+MCTS data generation is 3-8x slower** than heuristic selfplay, creating a bottleneck

## Key Findings from Investigation

### Speed Gap (Heuristic vs NN+MCTS)

| Metric        | GPU Heuristic   | NN+MCTS         |
| ------------- | --------------- | --------------- |
| Throughput    | 13-16 games/sec | 2-5 games/sec   |
| NN calls/move | 0               | 4-5 (per phase) |
| Parallelism   | 64 games        | 1 game          |
| Bottleneck    | Vectorized      | Sequential      |

### Loop Architecture Gap

- Training saves models to `data/checkpoints/`
- Selfplay requires manual `--model-path` specification
- No automatic model discovery or event-driven updates
- `unified_ai_loop.py` has components but they're disconnected

## Implementation Plan

### Phase 1: Quick Win - Parallel MCTS Games (2-3 days)

**Goal**: 4x speedup by batching multiple MCTS games together

**Approach**: Instead of running 1 game with MCTS at a time, run N games simultaneously and batch their NN evaluations together.

**Implementation**:

1. Create `ParallelGumbelMCTS` class that manages multiple game states
2. During Sequential Halving, collect leaf states from ALL games
3. Single batch NN call for all games' evaluations
4. Distribute results back to individual games

**Files to modify**:

- `ai-service/app/ai/gumbel_mcts_ai.py` - Add batched multi-game support
- `ai-service/scripts/run_hybrid_selfplay.py` - Add `--parallel-mcts N` option

**Expected speedup**: 3-4x (from batching NN calls across games)

### Phase 2: Model Discovery & Auto-Selection (1-2 days)

**Goal**: Selfplay automatically uses the latest trained model

**Implementation**:

1. Create `SelfplayModelSelector` in `app/training/selfplay_model_selector.py`:

   ```python
   def get_current_model(board_type: str, num_players: int) -> Path:
       # 1. Check model registry for PRODUCTION model
       # 2. Fall back to latest checkpoint in data/checkpoints/
       # 3. Fall back to canonical baseline model
   ```

2. Integrate into selfplay config initialization:
   - If no `--model-path` specified, use selector
   - Log which model was selected and why

3. Add model versioning to selfplay output metadata

**Files to create/modify**:

- `ai-service/app/training/selfplay_model_selector.py` (NEW)
- `ai-service/app/training/selfplay_config.py` - Use selector as default

### Phase 3: Event-Driven Model Updates (2-3 days)

**Goal**: Selfplay workers automatically switch to new models when promoted

**Implementation**:

1. Extend `ModelPromoter` to emit structured events:

   ```python
   event_router.emit(DataEventType.MODEL_PROMOTED, {
       "board_type": "square8",
       "num_players": 2,
       "model_path": "/path/to/new/model.pt",
       "elo_improvement": 45
   })
   ```

2. Add event subscription to selfplay workers:

   ```python
   event_router.subscribe(DataEventType.MODEL_PROMOTED, self._on_model_promoted)

   def _on_model_promoted(self, event):
       if self._matches_config(event):
           self._load_new_model(event["model_path"])
   ```

3. Implement graceful model hot-reload:
   - Finish current game
   - Load new model weights
   - Continue with new model

**Files to modify**:

- `ai-service/scripts/unified_loop/promotion.py` - Emit events
- `ai-service/app/coordination/event_router.py` - Add MODEL_PROMOTED type
- `ai-service/app/ai/gumbel_mcts_ai.py` - Add model reload method
- `ai-service/scripts/run_hybrid_selfplay.py` - Subscribe to events

### Phase 4: Orchestration Integration (2-3 days)

**Goal**: Unified daemon that manages the entire train → selfplay → retrain loop

**Implementation**:

1. Create `IterativeTrainingLoop` that coordinates:

   ```python
   class IterativeTrainingLoop:
       def run_iteration(self):
           # 1. Check data requirements
           needs = self.training_scheduler.get_data_needs()

           # 2. Start/adjust selfplay workers
           self.selfplay_coordinator.set_targets(needs)

           # 3. Wait for data threshold
           await self.wait_for_data_ready()

           # 4. Run training
           model = await self.training_executor.train()

           # 5. Evaluate model
           elo = await self.tournament.evaluate(model)

           # 6. Promote if improved
           if elo > self.champion_elo + THRESHOLD:
               self.promoter.promote(model)  # Triggers event

           # 7. Loop continues with new model in selfplay
   ```

2. Wire existing components:
   - `TrainingScheduler` - data needs
   - `SelfplayOrchestrator` - worker management
   - `ModelPromoter` - promotion events
   - `ShadowTournamentService` - evaluation

**Files to create/modify**:

- `ai-service/scripts/unified_loop/iterative_loop.py` (NEW)
- `ai-service/scripts/unified_ai_loop.py` - Add iteration mode

### Phase 5: Quality Gates & Monitoring (2-3 days)

**Goal**: Ensure loop produces improving models, not noise

**Implementation**:

1. Pre-training data quality check:
   - Parity validation on new games
   - Diversity metrics (unique positions, move variety)
   - Minimum game length filters

2. Post-training model quality check:
   - Quick tournament against previous champion
   - Reject if Elo drops more than 50
   - Alert on training instability

3. Dashboard integration:
   - Iteration number and current phase
   - Data generation rate vs consumption
   - Elo trend over iterations
   - Model lineage visualization

**Files to create/modify**:

- `ai-service/app/training/data_quality.py` - Pre-training checks
- `ai-service/scripts/unified_loop/quality_gates.py` (NEW)
- Add Prometheus metrics for iteration tracking

## Execution Order

**Immediate (Day 1-3)**: Phase 1 - Parallel MCTS

- This gives immediate 3-4x speedup to NN+MCTS generation
- Unblocks the data generation bottleneck

**Week 1 (Day 4-7)**: Phase 2 + 3 - Model Discovery & Events

- Closes the loop automatically
- Selfplay starts using trained models without manual intervention

**Week 2**: Phase 4 + 5 - Full Orchestration

- Complete automation
- Quality gates prevent regression
- Monitoring for observability

## Success Metrics

1. **NN+MCTS throughput**: 8-12 games/sec (up from 2-5)
2. **Loop closure time**: Model deployed to selfplay within 5 minutes of training completion
3. **Iteration rate**: 1 complete iteration (train → generate → retrain) per day
4. **Elo improvement**: +50 Elo per week of iterative training

## Alternative Approaches Considered

1. **Async NN inference with queuing** - More complex, less reliable
2. **Distillation from heuristic first** - Already doing this partially
3. **Reduce MCTS budget** - Trades quality for speed (last resort)
4. **Cloud burst for NN inference** - Cost prohibitive at scale

## Dependencies

- Existing event router infrastructure
- Model registry system
- P2P cluster communication
- Prometheus/Grafana monitoring

## Risks

1. **Hot model reload might cause instability** - Mitigate with graceful transition
2. **Batched MCTS might have different behavior** - Validate with parity tests
3. **Automated loop might promote bad models** - Quality gates essential
