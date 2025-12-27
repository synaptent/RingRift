# RingRift AI-Service Consolidation Roadmap

**Created**: December 2025
**Purpose**: Address architectural fragmentation identified in codebase review

## Executive Summary

The ai-service codebase has grown to include significant duplication and fragmentation:

| Category                    | Count | Lines    | Issue                       |
| --------------------------- | ----- | -------- | --------------------------- |
| Gumbel MCTS variants        | 7     | ~5,935   | Overlapping implementations |
| Export modules              | 9+    | ~5,400   | Fragmented responsibilities |
| Coordinator/Manager classes | 75+   | ~25,000+ | Excessive abstraction       |
| Training orchestrators      | 6+    | ~8,000   | Multiple competing systems  |

This roadmap provides a phased approach to consolidation while preserving valuable features.

---

## Phase 1: Gumbel MCTS Consolidation (HIGH PRIORITY)

### Current State

```
app/ai/
├── gumbel_mcts_ai.py       (1,843 lines) - Standard single-game
├── batched_gumbel_mcts.py  (509 lines)   - Multi-game batching wrapper
├── multi_game_gumbel.py    (509 lines)   - 64+ games parallel runner
├── tensor_gumbel_tree.py   (1,775 lines) - GPU tensor-based tree
├── gumbel_mcts_gpu.py      (282 lines)   - GPU-accelerated extension
├── gmo_gumbel_hybrid.py    (619 lines)   - GMO value network hybrid
└── gumbel_common.py        (398 lines)   - Shared structures (GOOD!)
```

### Issues

1. **Duplicate data structures**: `MultiGameGumbelAction` in `multi_game_gumbel.py` duplicates `GumbelAction` from `gumbel_common.py`
2. **Overlapping search logic**: Sequential Halving implemented 4 times
3. **Inconsistent batching**: Each variant has its own leaf batching
4. **No unified interface**: Factory (`factory.py`) has hardcoded branches

### Target Architecture

```
app/ai/
├── gumbel/
│   ├── __init__.py                    # Public API
│   ├── core.py                        # GumbelAction, GumbelNode, schedules
│   ├── search.py                      # Unified search: single/multi/GPU
│   ├── batching.py                    # Unified leaf batching
│   └── configs.py                     # GumbelConfig with mode enum
├── gumbel_mcts_ai.py                  # FACADE: delegates to gumbel/
└── [legacy variants -> deprecated/]
```

### Migration Steps

1. **Migrate `multi_game_gumbel.py` to use `gumbel_common.GumbelAction`**
   - Remove `MultiGameGumbelAction` class
   - Use `GumbelAction.from_gumbel_score()` factory method
   - Estimated savings: ~50 lines

2. **Unify Sequential Halving into `gumbel_common.py`**
   - Already have `compute_sequential_halving_schedule()`
   - Add `SequentialHalvingExecutor` class
   - Remove duplicate implementations from:
     - `gumbel_mcts_ai.py:552-600`
     - `batched_gumbel_mcts.py:300-350`
     - `multi_game_gumbel.py:281-341`
     - `tensor_gumbel_tree.py:561-650`

3. **Create unified `GumbelSearchEngine`**

   ```python
   class GumbelSearchMode(Enum):
       SINGLE_GAME = "single"      # Standard CPU search
       MULTI_GAME = "multi"        # 64+ games parallel
       GPU_TENSOR = "gpu_tensor"   # Full GPU tree
       GPU_BATCHED = "gpu_batched" # CPU tree, GPU eval

   class GumbelSearchEngine:
       def __init__(self, mode: GumbelSearchMode, config: GumbelConfig):
           ...

       def search(self, states: list[GameState]) -> list[Move]:
           # Dispatch to appropriate implementation
           ...
   ```

4. **Preserve valuable unique features**:
   - `tensor_gumbel_tree.py`: Full GPU tensor tree (unique)
   - `gmo_gumbel_hybrid.py`: GMO value mixing (unique)
   - `multi_game_gumbel.py`: Per-game parallel optimization (merge into engine)

### Expected Savings

- **Lines removed**: ~1,200-1,500 (duplicate search logic)
- **Maintenance burden**: 7 files → 4 files
- **Bug surface**: Single test suite for all modes

---

## Phase 2: Export Pipeline Consolidation (MEDIUM PRIORITY)

### Current State

```
app/training/
├── export_core.py              # Core export logic
├── game_record_export.py       # DB → record extraction
├── incremental_export.py       # Streaming/incremental
├── dynamic_export.py           # Dynamic feature export
└── export_cache.py             # Caching layer

scripts/
├── export_replay_dataset.py    # Main CLI
├── jsonl_to_npz.py            # Format conversion
└── export_gumbel_*.py         # Gumbel-specific exports
```

### Issues

1. **Format fragmentation**: SQLite DB, JSONL, NPZ all handled separately
2. **No unified pipeline**: Each script reimplements data loading
3. **Discovery not integrated**: `GameDiscovery` not used consistently
4. **Quality checks scattered**: `data_quality.py` separate from export

### Target Architecture

```
app/training/export/
├── __init__.py                 # Public API
├── pipeline.py                 # ExportPipeline class
├── sources.py                  # Unified data sources (DB, JSONL)
├── transformers.py             # Feature extraction, augmentation
├── writers.py                  # NPZ, HDF5, TFRecord writers
└── quality.py                  # Integrated quality checks

scripts/
└── export_training_data.py     # Single unified CLI
```

### Key Changes

1. **Unified `ExportPipeline` class**:

   ```python
   class ExportPipeline:
       def __init__(self):
           self.sources: list[DataSource] = []
           self.transformers: list[Transformer] = []
           self.quality_checks: list[QualityCheck] = []
           self.writer: DataWriter = NPZWriter()

       def add_source(self, source: DataSource) -> Self: ...
       def add_transformer(self, t: Transformer) -> Self: ...
       def export(self, output_path: Path) -> ExportResult: ...
   ```

2. **Integrate `GameDiscovery` as default source**

3. **Built-in quality gates**: Fail export if quality below threshold

4. **Streaming mode**: Process games without loading all into memory

---

## Phase 3: Coordinator/Manager Consolidation (HIGH PRIORITY)

### Current State

**75+ coordinator/manager/orchestrator classes** identified, including:

**Training-related (6 overlapping)**:

- `TrainingOrchestrator` (orchestrated_training.py)
- `UnifiedTrainingOrchestrator` (unified_orchestrator.py)
- `TrainingCoordinator` (training_coordinator.py)
- `TrainingCoordinator` (p2p_integration.py) - DUPLICATE NAME!
- `TrainingLifecycleManager` (lifecycle_integration.py)
- `IntegratedTrainingManager` (integrated_enhancements.py)

**Health/Monitoring (4 overlapping)**:

- `UnifiedHealthOrchestrator` (unified_health.py)
- `NodeHealthOrchestrator` (node_health_orchestrator.py)
- `UnifiedHealthManager` (unified_health_manager.py)
- `MonitoringManager` (p2p_monitoring.py)

**Sync/Coordination (4 overlapping)**:

- `SyncOrchestrator` (sync_orchestrator.py, legacy; prefer SyncFacade)
- `SyncCoordinator` (sync_coordinator.py)
- `SyncScheduler` (sync_coordinator.py)
- `DistributedSyncCoordinator` (alias in sync_coordinator.py)

### Issues

1. **Naming collisions**: Multiple `TrainingCoordinator` classes
2. **Unclear responsibilities**: Orchestrator vs Coordinator vs Manager undefined
3. **Deep hierarchies**: CoordinatorBase with 8+ mixins
4. **Circular dependencies**: Cross-imports between coordinators

### Target Architecture

```
app/coordination/
├── __init__.py                  # Public API
├── base.py                      # CoordinatorBase (simplified)
├── training/
│   └── orchestrator.py          # SINGLE UnifiedTrainingOrchestrator
├── health/
│   └── orchestrator.py          # SINGLE HealthOrchestrator
├── sync/
│   └── coordinator.py           # SINGLE SyncCoordinator
├── lifecycle/
│   └── manager.py               # Lifecycle management
└── registry.py                  # Global orchestrator registry
```

### Naming Conventions

| Suffix         | Responsibility                       | Lifecycle    |
| -------------- | ------------------------------------ | ------------ |
| `Orchestrator` | Active management, complex workflows | Long-running |
| `Coordinator`  | Communication between components     | On-demand    |
| `Manager`      | Resource lifecycle (create/destroy)  | Singleton    |

### Migration Steps

1. **Rename to eliminate collisions**:
   - `p2p_integration.TrainingCoordinator` → `P2PTrainingBridge`
   - `p2p_integration.SelfplayCoordinator` → `P2PSelfplayBridge`

2. **Merge training orchestrators**:
   - Keep `UnifiedTrainingOrchestrator` as base
   - Merge features from `TrainingOrchestrator`, `IntegratedTrainingManager`
   - Deprecate: `orchestrated_training.py`, `integrated_enhancements.py`

3. **Merge health orchestrators**:
   - Keep `UnifiedHealthManager` (extends CoordinatorBase)
   - Merge `NodeHealthOrchestrator`, `UnifiedHealthOrchestrator`

4. **Simplify CoordinatorBase**:
   - Remove unused mixins (keep: SQLitePersistence, Singleton, Callback)
   - Reduce from 8+ mixins to 3-4 essential ones

---

## Phase 4: Configuration Consolidation (MEDIUM PRIORITY)

### Current State

7+ distinct config dataclasses identified:

- `OrchestratorConfig` (unified_orchestrator.py)
- `TrainingOrchestratorConfig` (orchestrated_training.py)
- `SyncOrchestratorConfig` (sync_orchestrator.py, legacy)
- `CoordinatorConfig` (coordinator_config.py)
- `GPUGumbelMCTSConfig` (tensor_gumbel_tree.py)
- `GMOGumbelConfig` (gmo_gumbel_hybrid.py)
- Plus: board-specific, player-specific configs

### Target Architecture

```python
# app/config/training.py
@dataclass
class TrainingConfig:
    """Single source of truth for training configuration."""

    # Board settings
    board_type: BoardType = BoardType.SQUARE8
    num_players: int = 2

    # Training hyperparameters
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 0.001

    # Selfplay settings
    selfplay: SelfplayConfig = field(default_factory=SelfplayConfig)

    # Search settings
    search: SearchConfig = field(default_factory=SearchConfig)

    # Distributed settings
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    @classmethod
    def for_board(cls, board_type: BoardType, num_players: int) -> Self:
        """Factory for board-specific defaults."""
        ...
```

---

## Phase 5: Event System Consolidation (LOW PRIORITY)

### Current State

15+ event types across multiple modules:

- `training_events.py`: Training step, checkpoint events
- `p2p_notifications.py`: P2P cluster events
- `unified_event_coordinator.py`: Central event bus

### Issues

- Some events fire synchronously, some async
- No unified schema for event payloads
- Multiple subscription mechanisms

### Target Architecture

Single event bus with typed events:

```python
class TrainingEvents:
    STEP_COMPLETED = "training.step.completed"
    CHECKPOINT_SAVED = "training.checkpoint.saved"
    EPOCH_COMPLETED = "training.epoch.completed"

class ClusterEvents:
    NODE_JOINED = "cluster.node.joined"
    NODE_LEFT = "cluster.node.left"
    LEADER_ELECTED = "cluster.leader.elected"
```

---

## Valuable Features to Preserve

During consolidation, these unique features MUST be preserved:

### From Gumbel Variants

- **GPU tensor tree** (`tensor_gumbel_tree.py`): Full tree in GPU memory
- **GMO hybrid** (`gmo_gumbel_hybrid.py`): GMO value network mixing
- **Multi-game optimization** (`multi_game_gumbel.py`): Per-game parallelism

### From Training Infrastructure

- **Hot buffer** (unified_orchestrator.py): Priority experience replay
- **Curriculum feedback** (curriculum_feedback.py): Dynamic weight adjustment
- **Quality bridge** (quality_bridge.py): Quality-weighted sampling
- **Online learning** (online_learning.py): EBMO continuous learning

### From Coordination

- **Signal-driven decisions** (unified_signals.py): Training signal SSoT
- **Gauntlet-based promotion** (auto_promote.py): ELO gating
- **Adaptive resources** (adaptive_resource_manager.py): Dynamic allocation

---

## Implementation Timeline

### Immediate (1-2 weeks)

- [x] Fix `MultiGameGumbelAction` duplication (Dec 2025)
- [x] Rename collision `TrainingCoordinator` classes (Dec 2025)
- [ ] Document current architecture in diagrams

### Short-term (2-4 weeks)

- [x] Unify Sequential Halving implementations (Dec 2025)
- [x] Create unified `GumbelSearchEngine` (Dec 2025)
- [x] Document training orchestrator consolidation (Dec 2025) - Created ORCHESTRATOR_GUIDE.md - Added deprecation notices to orchestrated_training.py, integrated_enhancements.py - Target: 5 orchestrators → 3 (keeping UnifiedTrainingOrchestrator, TrainingCoordinator, TrainingLifecycleManager)

### Medium-term (1-2 months)

- [x] Document export pipeline consolidation (Dec 2025) - Created EXPORT_GUIDE.md with module roles and usage - Primary CLI: scripts/export_replay_dataset.py - Core utilities: app/training/export_core.py
- [x] Document CoordinatorBase hierarchy (Dec 2025) - Created COORDINATOR_GUIDE.md with class categorization - Identified 125 classes, categorized into Core/Deprecated/Specialized - Defined naming conventions: Orchestrator vs Coordinator vs Manager
- [x] Document configuration system (Dec 2025) - Created CONFIG_GUIDE.md with usage patterns and migration - UnifiedConfig (unified_config.py) is single source of truth - 28 sub-configuration dataclasses consolidated - Identified 7 legacy config files for deprecation

### Long-term (ongoing)

- [x] Event system consolidation (Dec 2025) - Created EVENT_GUIDE.md documenting three-tier architecture - EventBus (core), StageEventBus (pipeline), CrossProcessEventQueue (IPC) - Centralized mappings in event_mappings.py - UnifiedEventCoordinator bridges all buses
- [x] Quality scoring consolidation (Dec 2025) - UnifiedQualityScorer (app/quality/unified_quality.py) is SSoT - Added compute_freshness_score() for backwards compatibility - Deprecated: DataQualityScorer (training_enhancements.py), GameQualityScorer (scripts/lib/data_quality.py)
- [x] Event system hardening (Dec 2025) - Fixed fire-and-forget asyncio.create_task in event_router.py with error callbacks - Fixed silent print() statements in cross_process_events.py with proper logger.error() - Added dispatch failure tracking metrics
- [x] Feedback loop completion (Dec 2025) - Eval→curriculum: game_gauntlet.py now emits EVALUATION_COMPLETED events consumed by TournamentToCurriculumWatcher - Cluster→training: training_coordinator.py subscribes to P2P_CLUSTER_HEALTHY/UNHEALTHY, NODE_RECOVERED events - Training blocked when cluster unhealthy
- [x] Deprecation warnings added (Dec 2025) - orchestrated_training.py, integrated_enhancements.py, training_enhancements.DataQualityScorer, scripts/lib/data_quality.py - Files kept in place (still imported) with runtime DeprecationWarning - Gradual migration to unified modules recommended
- [ ] Remove deprecated code (when all imports migrated)
- [ ] Documentation updates

---

## Metrics for Success

| Metric                      | Current | Target      | Status                      |
| --------------------------- | ------- | ----------- | --------------------------- |
| Gumbel implementation files | 7       | 4           | GumbelSearchEngine created  |
| Lines in Gumbel modules     | ~5,935  | ~4,000      | ~200 lines saved            |
| Coordinator/Manager classes | 125+    | 40-50       | Documented, 3 deprecated    |
| Config dataclasses          | 28      | 1 (unified) | DONE - UnifiedConfig SSoT   |
| Export modules              | 9+      | 4-5         | Documented, unified CLI     |
| Quality scorers             | 4       | 1 (unified) | DONE - UnifiedQualityScorer |

---

## Appendix: Files for Deprecation

After consolidation, these files should be archived:

```
# Move to deprecated/ folder
app/training/orchestrated_training.py    → merged into unified_orchestrator
app/training/integrated_enhancements.py  → features extracted
app/ai/batched_gumbel_mcts.py           → replaced by GumbelSearchEngine
app/monitoring/unified_health.py         → merged into unified_health_manager
```

---

_This roadmap should be reviewed and updated quarterly._
