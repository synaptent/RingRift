# Core Coordination Module

This package provides core coordination infrastructure for the training pipeline.

## Modules

| Module        | Description                                                    |
| ------------- | -------------------------------------------------------------- |
| `events.py`   | Unified event routing and emission via `UnifiedEventRouter`    |
| `tasks.py`    | Task lifecycle and coordination via `TaskCoordinator`          |
| `pipeline.py` | Training pipeline orchestration via `DataPipelineOrchestrator` |

## Usage

```python
from app.coordination.core.events import UnifiedEventRouter
from app.coordination.core.tasks import TaskCoordinator
from app.coordination.core.pipeline import DataPipelineOrchestrator

# Event routing
router = UnifiedEventRouter()
router.subscribe("TRAINING_COMPLETE", handler)
router.emit("TRAINING_COMPLETE", {"epoch": 50, "loss": 0.05})

# Task coordination
coordinator = TaskCoordinator()
task_id = coordinator.submit_task(my_task)
await coordinator.wait_for_completion(task_id)

# Pipeline orchestration
pipeline = DataPipelineOrchestrator()
pipeline.trigger_stage("export")
```

## Architecture

```
core/
├── events.py     # Event bus with deduplication (SHA256)
├── tasks.py      # Task submission and lifecycle
└── pipeline.py   # Pipeline stage tracking and triggering
```

## Event System

The event system provides:

- Content-based deduplication via SHA256 hashing
- Fire-and-forget async handlers
- Error callbacks for failed handlers
- Cross-process event propagation (optional)

## December 2025 Consolidation

This package was created during the December 2025 module consolidation effort
(75 → 15 modules). It brings together core coordination primitives that form
the foundation for higher-level orchestration.

## See Also

- `app.coordination.event_router` - Main event router (re-exports from here)
- `app.coordination.daemon_manager` - Daemon lifecycle management
- `app.coordination.training` - Training-specific orchestration
