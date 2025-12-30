# Data Pipeline Flow - RingRift AI Training

This document describes the data flow through the RingRift AI training pipeline, from selfplay game generation to model deployment.

## Pipeline Stages

```
SELFPLAY → SYNC → NPZ_EXPORT → NPZ_COMBINATION → TRAINING → EVALUATION → PROMOTION
```

### Stage 1: SELFPLAY

**Purpose**: Generate training data through self-play games.

**Components**:

- `SelfplayScheduler` - Priority-based config allocation
- `IdleResourceDaemon` - Spawn selfplay on idle GPUs
- `gpu_parallel_games.py` - Vectorized GPU selfplay engine

**Inputs**:

- Board configuration (hex8_2p, square8_4p, etc.)
- Current model weights
- Curriculum weights from feedback loops

**Outputs**:

- Game data stored in SQLite databases (`data/games/*.db`)
- Each game includes: moves, states, policy targets, value outcomes

**Events**:

- Subscribes to: `CURRICULUM_REBALANCED`, `SELFPLAY_RATE_CHANGED`
- Emits: `SELFPLAY_COMPLETE`, `NEW_GAMES_AVAILABLE`

### Stage 2: SYNC

**Purpose**: Replicate game data across cluster nodes.

**Components**:

- `AutoSyncDaemon` - Primary sync orchestrator
- `SyncRouter` - Intelligent node selection
- `SyncPlanner` - Manifest collection and planning
- `BandwidthManager` - Rate limiting

**Inputs**:

- Game databases from selfplay nodes
- Node capacity and availability data

**Outputs**:

- Game data replicated to training-eligible nodes
- Sync receipts for verification

**Events**:

- Subscribes to: `NEW_GAMES_AVAILABLE`, `TRAINING_STARTED`, `NODE_RECOVERED`, `DATA_STALE`, `SYNC_REQUEST`
- Emits: `DATA_SYNC_STARTED`, `DATA_SYNC_COMPLETED`, `DATA_SYNC_FAILED`

### Cross-cutting: Data Freshness Gate

`TrainingFreshness` emits `DATA_STALE` / `DATA_FRESH` when training data ages past
the configured threshold. `DATA_STALE` triggers a priority sync (`SYNC_REQUEST`)
and `TrainingTriggerDaemon` blocks training until data is fresh again.

### Stage 3: NPZ_EXPORT

**Purpose**: Convert game databases to NumPy training format.

**Components**:

- `export_replay_dataset.py` - Main export script
- `parallel_encoding.py` - Parallel feature encoding
- `GameDiscovery` - Database discovery

**Inputs**:

- Synced game databases
- Board configuration
- Feature extraction settings

**Outputs**:

- NPZ files with training data:
  - `features` + `globals`: Board/state tensors + global features
  - `policy_indices` + `policy_values`: Sparse move distributions
  - `values` (and `values_mp` + `num_players` for multi-player exports)
  - `phases`, `move_numbers`, `total_game_moves`, `victory_types`, `engine_modes`, `move_types`
  - `opponent_elo`, `quality_score`, `opponent_types` when recorded
  - `sample_weights` + `timestamps` when `--quality-weighted` is enabled
  - `heuristics` when heuristic extraction is enabled

**Events**:

- Triggered by: `DATA_SYNC_COMPLETED` (via DataPipelineOrchestrator)
- Emits: `NPZ_EXPORT_COMPLETE`

### Stage 3b: NPZ_COMBINATION (Optional)

**Purpose**: Combine multiple NPZ files into a single quality-weighted dataset.

**Components**:

- `NPZCombinationDaemon` - Orchestrates combination after export
- `npz_combiner.py` - Quality + freshness weighting and deduplication

**Inputs**:

- One or more NPZ export files for the config

**Outputs**:

- Combined NPZ file (typically `{config}_combined.npz`)

**Events**:

- Triggered by: `NPZ_EXPORT_COMPLETE`
- Emits: `NPZ_COMBINATION_STARTED`, `NPZ_COMBINATION_COMPLETE`, `NPZ_COMBINATION_FAILED`

If combination fails, the pipeline falls back to training on the latest export.

### Stage 4: TRAINING

**Purpose**: Train neural network on exported data.

**Components**:

- `app.training.train` - Main training entry point
- `TrainingCoordinator` - Distributed training coordination
- `FeedbackLoopController` - Adaptive training parameters

**Inputs**:

- NPZ training data
- Current model checkpoint
- Hyperparameters (LR, batch size, epochs)

**Outputs**:

- Updated model checkpoint
- Training metrics (loss, accuracy)

**Events**:

- Triggered by: `NPZ_EXPORT_COMPLETE` or threshold-based triggers
- Subscribes to: `PLATEAU_DETECTED`, `ADAPTIVE_PARAMS_CHANGED`
- Emits: `TRAINING_STARTED`, `TRAINING_COMPLETED`, `TRAINING_FAILED`

### Stage 5: EVALUATION

**Purpose**: Evaluate trained model against baselines.

**Components**:

- `EvaluationDaemon` - Evaluation orchestration
- `GameGauntlet` - Multi-opponent evaluation
- `GauntletFeedbackController` - Result analysis

**Inputs**:

- Newly trained model
- Baseline models (Random, Heuristic, Previous best)

**Outputs**:

- Win rates against each baseline
- Elo rating estimate
- Promotion recommendation

**Events**:

- Triggered by: `TRAINING_COMPLETED`
- Emits: `EVALUATION_COMPLETED`, `PROMOTION_CANDIDATE`

### Stage 6: PROMOTION

**Purpose**: Deploy successful models to production.

**Components**:

- `AutoPromotionDaemon` - Promotion decision logic
- `UnifiedDistributionDaemon` - Model distribution
- `ModelRegistry` - Model versioning

**Inputs**:

- Evaluated model with metrics
- Promotion thresholds

**Outputs**:

- Promoted model in canonical location
- Model distributed to cluster nodes

**Events**:

- Triggered by: `PROMOTION_CANDIDATE`
- Emits: `MODEL_PROMOTED`, `PROMOTION_FAILED`

---

## Event Flow Diagram

```
                    ┌─────────────────┐
                    │  SelfplayScheduler │
                    └─────────┬─────────┘
                              │ SELFPLAY_COMPLETE
                              ▼
                    ┌─────────────────┐
                    │   AutoSyncDaemon   │
                    └─────────┬─────────┘
                              │ DATA_SYNC_COMPLETED
                              ▼
              ┌───────────────────────────────┐
              │   DataPipelineOrchestrator    │
              │   (Central Coordination)       │
              └───────────────┬───────────────┘
                              │ trigger_export()
                              ▼
                    ┌─────────────────┐
                    │   NPZ Export      │
                    └─────────┬─────────┘
                              │ NPZ_EXPORT_COMPLETE
                              ▼
                    ┌─────────────────┐
                    │ NPZ Combination  │
                    └─────────┬─────────┘
                              │ NPZ_COMBINATION_COMPLETE
                              ▼
                    ┌─────────────────┐
                    │   Training        │
                    └─────────┬─────────┘
                              │ TRAINING_COMPLETED
                              ▼
                    ┌─────────────────┐
                    │   Evaluation      │
                    └─────────┬─────────┘
                              │ PROMOTION_CANDIDATE
                              ▼
                    ┌─────────────────┐
                    │   Promotion       │
                    └─────────┬─────────┘
                              │ MODEL_PROMOTED
                              ▼
              ┌───────────────────────────────┐
              │  UnifiedDistributionDaemon    │
              │  (Model Distribution)          │
              └───────────────────────────────┘
```

---

## Feedback Loops

### Curriculum Feedback Loop

```
EVALUATION_COMPLETED
        │
        ▼
┌───────────────────┐    CURRICULUM_REBALANCED    ┌──────────────────┐
│ CurriculumIntegration │ ──────────────────────▶ │ SelfplayScheduler  │
└───────────────────┘                             └──────────────────┘
        ▲                                                   │
        │                 SELFPLAY_ALLOCATION_UPDATED       │
        └───────────────────────────────────────────────────┘
```

**Loop Guard**: Source tracking prevents echo loops. Events with `source` field matching the handler's component are skipped.

### Elo Momentum Feedback Loop

```
EVALUATION_COMPLETED → FeedbackAccelerator → SELFPLAY_RATE_CHANGED → SelfplayScheduler
                                │
                                ▼
                    Adjust selfplay rate multiplier
                    (1.5x accelerating, 0.75x regressing)
```

### Quality Feedback Loop

```
QualityMonitorDaemon → QUALITY_DEGRADED → FeedbackLoopController
                                                    │
                                                    ▼
                                       Adjust training parameters
                                       (reduce batch size, increase epochs)
```

---

## Key Components

### DataPipelineOrchestrator

Central coordinator that:

- Tracks pipeline stage transitions
- Coordinates downstream triggering
- Provides pipeline-wide observability
- Implements circuit breaker for fault tolerance

**Location**: `app/coordination/data_pipeline_orchestrator.py`

### Event Router

Unified event system that:

- Bridges in-memory, stage, and cross-process events
- Provides content-based deduplication (SHA256)
- Implements dead letter queue for failed handlers

**Location**: `app/coordination/event_router.py`

### DaemonManager

Lifecycle management for 73 daemon types:

- Start/stop coordination
- Health monitoring
- Auto-restart on failure
- Dependency ordering

**Location**: `app/coordination/daemon_manager.py`

---

## Configuration

### Pipeline Thresholds

| Threshold                       | Default | Description                                                     |
| ------------------------------- | ------- | --------------------------------------------------------------- |
| `TRAINING_GAME_THRESHOLD`       | 1000    | Min games before training triggers                              |
| `EVALUATION_WIN_RATE_RANDOM`    | 85%     | Required vs Random for promotion                                |
| `EVALUATION_WIN_RATE_HEURISTIC` | 60%     | Required vs Heuristic for promotion                             |
| `SYNC_INTERVAL_SECONDS`         | 60      | AutoSyncDaemon cycle interval (AutoSyncConfig.interval_seconds) |

### Environment Variables

```bash
# Pipeline control
RINGRIFT_TRAINING_THRESHOLD=1000
RINGRIFT_EVALUATION_GAMES=50

# Sync settings
RINGRIFT_MIN_SYNC_INTERVAL=2.0
RINGRIFT_DATA_SYNC_INTERVAL=120
RINGRIFT_FAST_SYNC_INTERVAL=30
RINGRIFT_AUTO_SYNC_MAX_CONCURRENT=6

# Circuit breaker
RINGRIFT_CB_FAILURE_THRESHOLD=5
RINGRIFT_CB_RECOVERY_TIMEOUT=300
```

---

## Troubleshooting

### Pipeline Stalled

1. Check stage status:

   ```python
   from app.coordination.data_pipeline_orchestrator import get_pipeline_orchestrator
   orch = get_pipeline_orchestrator()
   print(orch.get_status())
   ```

2. Check event subscriptions:

   ```python
   from app.coordination.event_router import get_router
   router = get_router()
   print(router.get_subscription_count())
   ```

3. Check daemon health:
   ```bash
   python scripts/launch_daemons.py --status
   ```

### Missing Events

1. Verify event emitter is being called:

   ```bash
   grep "emit.*EVENT_NAME" logs/coordination.log
   ```

2. Check for handler errors:

   ```bash
   grep "Error handling" logs/coordination.log
   ```

3. Verify source tracking isn't blocking:
   - Check if `source` field matches handler's skip condition

---

## See Also

- `docs/EVENT_SYSTEM_REFERENCE.md` - Complete event type reference
- `docs/DAEMON_REGISTRY.md` - Daemon specifications
- `docs/runbooks/FEEDBACK_LOOP_TROUBLESHOOTING.md` - Feedback loop debugging
