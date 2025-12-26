# Architecture Naming Conventions

This document establishes naming conventions for modules and classes in the RingRift AI service codebase to improve consistency and maintainability.

## Module Suffix Conventions

| Suffix | Purpose | Examples |
|--------|---------|----------|
| `*Orchestrator` | Cross-module coordination, manages multiple subsystems | `TrainingOrchestrator`, `SelfplayOrchestrator`, `DataPipelineOrchestrator` |
| `*Scheduler` | Time/priority-based dispatch and allocation | `SelfplayScheduler`, `CurriculumScheduler`, `JobScheduler` |
| `*Manager` | State and lifecycle management for a single concern | `DaemonManager`, `CheckpointManager`, `ResourceManager` |
| `*Controller` | Request/response flow, handles specific actions | `PromotionController`, `FeedbackLoopController` |
| `*Daemon` | Long-running background worker process | `SyncDaemon`, `TournamentDaemon`, `IdleResourceDaemon` |
| `*Service` | Stateless utility providing operations | `EloService`, `ModelService`, `MetricsService` |
| `*Handler` | Event or request handler | `ElectionHandler`, `WorkQueueHandler` |
| `*Monitor` | Observes and reports on system state | `ClusterMonitor`, `QualityMonitor` |
| `*Runner` | Executes a specific task or workflow | `SelfplayRunner`, `GauntletRunner` |
| `*Config` | Configuration dataclass | `TrainingConfig`, `SelfplayConfig`, `GauntletConfig` |

## Package Structure

```
ai-service/
├── app/
│   ├── ai/                    # AI algorithms (MCTS, neural networks)
│   ├── config/                # Centralized configuration
│   ├── coordination/          # Training pipeline orchestration
│   │   ├── cluster/           # Cluster-specific coordination
│   │   └── daemons/           # Background daemon implementations
│   ├── core/                  # Core utilities (SSH, node info)
│   ├── db/                    # Database utilities
│   ├── distributed/           # Cluster management
│   ├── gauntlet/              # Model evaluation
│   ├── models/                # Model discovery and management
│   ├── monitoring/            # Metrics and monitoring
│   ├── rules/                 # Game rules engine
│   ├── training/              # Training pipeline
│   │   ├── temperature/       # Temperature scheduling
│   │   └── online/            # Online learning
│   └── utils/                 # General utilities
├── scripts/
│   ├── p2p/                   # P2P orchestrator components
│   │   └── handlers/          # HTTP API handlers
│   └── *.py                   # CLI entry points
└── tests/
    ├── unit/                  # Unit tests
    └── integration/           # Integration tests
```

## Naming Anti-Patterns

Avoid these patterns:

| Anti-Pattern | Problem | Better Alternative |
|--------------|---------|-------------------|
| `*Helper` | Too vague | Use specific suffix based on purpose |
| `*Util` / `*Utils` | Becomes dumping ground | Split into focused modules |
| `*Wrapper` | Unclear what it wraps | Name based on abstraction purpose |
| `*Base` in module names | Implementation detail | Keep in class name only |
| Abbreviations | Reduces readability | Spell out: `Scheduler` not `Sched` |

## Board Type Naming

Always use canonical board type identifiers:

| Canonical | Avoid |
|-----------|-------|
| `hex8` | `h8`, `hex_8`, `hexSmall` |
| `square8` | `sq8`, `s8`, `squareSmall` |
| `square19` | `sq19`, `s19`, `squareLarge` |
| `hexagonal` | `hex`, `hexLarge`, `bigHex` |

## Configuration Key Naming

Configuration keys follow these patterns:

```python
# Format: {board_type}_{num_players}p
# Examples:
"hex8_2p"
"square8_4p"
"hexagonal_3p"
```

## File Naming

| Type | Convention | Example |
|------|------------|---------|
| Module | snake_case | `selfplay_scheduler.py` |
| Test file | `test_*.py` | `test_selfplay_scheduler.py` |
| Config file | snake_case | `training_config.py` |
| Documentation | UPPER_SNAKE_CASE.md | `ARCHITECTURE_NAMING.md` |

## Import Path Conventions

Prefer the new package-based import paths:

```python
# Preferred (new)
from app.coordination.cluster.sync import SyncScheduler
from app.training.temperature import create_scheduler

# Deprecated (old flat structure)
from app.coordination.sync_coordinator import SyncScheduler  # DeprecationWarning
```

## Version Suffixes

When versioning modules or models:

```
canonical_{board}_{n}p.pth          # Production model
canonical_{board}_{n}p_v2.pth       # Version 2
{board}_{n}p_experimental.pth        # Experimental
```

## Deprecation Pattern

When deprecating a module:

1. Add docstring deprecation note with migration path
2. Add runtime warning at module level:
   ```python
   import warnings
   warnings.warn(
       "module_name is deprecated. Use new_module instead.",
       DeprecationWarning,
       stacklevel=2,
   )
   ```
3. Document removal timeline (e.g., "Q2 2026")
4. Update this document to track deprecated modules

## Currently Deprecated Modules

| Module | Replacement | Removal Target |
|--------|-------------|----------------|
| `orchestrated_training.py` | `unified_orchestrator.py` | Q1 2026 |
| `integrated_enhancements.py` | `unified_orchestrator.py` | Q1 2026 |
| `_game_engine_legacy.py` | `game_engine/` | Q2 2026 |

---

*Last updated: December 2025*
