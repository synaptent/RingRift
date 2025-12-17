# Refactoring Plan for Large Orchestrator Scripts

This document outlines the recommended refactoring strategy for the large monolithic scripts identified in the resource management improvement initiative.

## Target Files

| File                              | Lines  | Functions | Priority |
| --------------------------------- | ------ | --------- | -------- |
| `scripts/p2p_orchestrator.py`     | 25,516 | 499+      | High     |
| `scripts/unified_ai_loop.py`      | 8,742  | 176+      | Medium   |
| `scripts/run_improvement_loop.py` | 2,649  | 40+       | Low      |

## p2p_orchestrator.py Refactoring Plan

### Current Structure

The file contains:

1. **Utility functions** (lines 137-310): Systemd notifications, resource checks
2. **Enums and Data Classes** (lines 758-1435): NodeRole, JobType, NodeInfo, ClusterJob, etc.
3. **P2POrchestrator class** (line 1561+): Main orchestrator (~23,000 lines)
4. **Helper functions** (line 9525+): Agent loading, utilities
5. **main()** (line 25456): Entry point

### Proposed Module Structure

```
scripts/p2p/
├── __init__.py              # Re-exports for backward compatibility
├── types.py                 # Enums: NodeRole, JobType
├── models.py                # Dataclasses: NodeInfo, ClusterJob, etc.
├── resource.py              # Resource checking functions
├── network.py               # HTTP client, peer communication
├── data_sync.py             # DataSyncJob, ClusterDataManifest
├── scheduling.py            # Job scheduling logic
├── tournament.py            # Tournament coordination
├── training.py              # Training job management
├── selfplay.py              # Selfplay coordination
├── orchestrator.py          # Main P2POrchestrator class (trimmed)
└── cli.py                   # main() and CLI handling
```

### Migration Strategy

1. **Phase 1: Extract Types** (Low risk)
   - Create `p2p/types.py` with enums
   - Create `p2p/models.py` with dataclasses
   - Update imports in p2p_orchestrator.py
   - Run tests to verify

2. **Phase 2: Extract Utilities** (Low risk)
   - Create `p2p/resource.py` with resource functions
   - Create `p2p/network.py` with HTTP client code
   - Update imports and test

3. **Phase 3: Extract Job Types** (Medium risk)
   - Create `p2p/selfplay.py` with selfplay logic
   - Create `p2p/training.py` with training logic
   - Create `p2p/tournament.py` with tournament logic
   - Maintain backward compatibility

4. **Phase 4: Slim Orchestrator** (High risk)
   - Move remaining logic to appropriate modules
   - Keep P2POrchestrator as a thin coordinator
   - Extensive testing required

### Backward Compatibility

The refactored code must maintain:

- Same CLI interface (`python scripts/p2p_orchestrator.py`)
- Same HTTP API endpoints
- Same state file format
- Same log message patterns

Create `scripts/p2p/__init__.py`:

```python
# Backward compatibility - import everything from submodules
from .types import NodeRole, JobType
from .models import NodeInfo, ClusterJob, ...
from .orchestrator import P2POrchestrator
from .cli import main

# Allow direct import for compatibility
__all__ = ['P2POrchestrator', 'NodeRole', 'JobType', 'NodeInfo', ...]
```

## unified_ai_loop.py Refactoring Plan

### Current Structure

The file mixes:

- Data collection from cluster
- Training orchestration
- Tournament management
- Model promotion logic

### Proposed Module Structure

```
scripts/unified_loop/
├── __init__.py
├── config.py               # Configuration loading
├── data_collection.py      # Data sync and collection
├── training.py             # Training orchestration
├── evaluation.py           # Tournament and evaluation
├── promotion.py            # Model promotion logic
├── loop.py                 # Main loop coordinator
└── cli.py                  # Entry point
```

## run_improvement_loop.py Refactoring Plan

### Current Structure

The file handles:

- Selfplay stage
- Training stage
- Evaluation stage
- Promotion logic

### Proposed Module Structure

```
scripts/improvement/
├── __init__.py
├── stages/
│   ├── selfplay.py
│   ├── training.py
│   └── evaluation.py
├── promotion.py
├── checkpointing.py
├── loop.py
└── cli.py
```

## Testing Strategy

For each refactoring phase:

1. **Before refactoring**: Ensure all existing tests pass
2. **During refactoring**: Add unit tests for extracted modules
3. **After refactoring**: Run integration tests
4. **Validation**: Run actual workloads in staging

## Risk Mitigation

1. **Feature flags**: Add environment variable to switch between old/new code
2. **Parallel deployment**: Run old and new code side-by-side initially
3. **Rollback plan**: Keep old code available for quick rollback
4. **Incremental migration**: One module at a time

## Timeline

This refactoring should be done incrementally over multiple sessions:

- Phase 1 (Types): Can be done immediately
- Phase 2 (Utilities): After Phase 1 is validated
- Phase 3 (Job Types): Requires careful testing
- Phase 4 (Slim Orchestrator): Final phase

## Acceptance Criteria

Refactoring is complete when:

1. No single file exceeds 2,000 lines
2. Each module has single responsibility
3. All existing tests pass
4. Coverage maintained or improved
5. Performance unchanged or improved
