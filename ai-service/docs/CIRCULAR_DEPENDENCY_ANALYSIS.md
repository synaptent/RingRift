# Circular Dependency Analysis

**Generated**: December 29, 2025
**Tool**: `scripts/audit_circular_deps.py`

## Summary

| Metric             | Value |
| ------------------ | ----- |
| Modules scanned    | 1,864 |
| Total dependencies | 6,076 |
| Cycles found       | 150   |
| Critical severity  | 114   |
| High severity      | 18    |
| Low severity       | 18    |

## Key Hub Modules

These modules appear in the most circular dependency chains:

| Module                                 | Role              | Cycles Involved |
| -------------------------------------- | ----------------- | --------------- |
| `app.config.unified_config`            | Configuration hub | ~40+ cycles     |
| `app.distributed.data_events`          | Event types       | ~30+ cycles     |
| `app.coordination.event_router`        | Event routing     | ~25+ cycles     |
| `app.training.integrated_enhancements` | Training facade   | ~20+ cycles     |
| `app.db.game_replay`                   | Database access   | ~15+ cycles     |

## Critical Cycle Patterns

### Pattern 1: Config → Training → Events → Config

The most common pattern involves the configuration module:

```
app.config.unified_config
    → app.training.integrated_enhancements
    → app.training.* (various)
    → app.coordination.event_router
    → app.distributed.data_events
    → app.config.unified_config
```

**Root cause**: `data_events.py` imports from `unified_config` for event configuration.

**Fix strategy**: Lazy load config in `data_events.py` or extract event type definitions to a separate module with no dependencies.

### Pattern 2: Database → Training → Database

```
app.db
    → app.db.game_replay
    → app.training.generate_data
    → app.db
```

**Root cause**: `generate_data.py` imports database utilities that re-export from `app.db.__init__`.

**Fix strategy**: Use explicit imports (`from app.db.game_replay import ...`) instead of package imports.

### Pattern 3: Mixin Cycles (Expected)

```
app.coordination.data_pipeline_orchestrator
    → app.coordination.pipeline_trigger_mixin
    → app.coordination.data_pipeline_orchestrator
```

**Note**: These are expected patterns for mixin-based architecture. The mixin imports the main class for type hints.

**Fix strategy**: Use `TYPE_CHECKING` guards or string annotations.

### Pattern 4: Package **init** Self-Cycles (Expected)

```
app.coordination → app.coordination
```

**Note**: Self-cycles at package level are expected when `__init__.py` re-exports from submodules.

**Status**: No action needed.

## Recommended Fixes

### High Priority

1. **Break `data_events` → `unified_config` cycle**
   - Extract `DataEventType` enum to `app.distributed.event_types.py`
   - Have both `data_events.py` and `unified_config.py` import from there
   - Impact: ~30 cycles eliminated

2. **Break `training` → `db` cycle**
   - In `generate_data.py`, use explicit imports
   - Change `from app.db import ...` to `from app.db.game_replay import ...`
   - Impact: ~5 cycles eliminated

3. **Lazy load in `integrated_enhancements`**
   - This module is a facade that imports many training modules
   - Convert heavy imports to lazy loading pattern
   - Impact: ~20 cycles eliminated

### Medium Priority

4. **Use TYPE_CHECKING for mixin imports**

   ```python
   from typing import TYPE_CHECKING
   if TYPE_CHECKING:
       from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator
   ```

5. **Break `event_router` → `data_events` cycle**
   - Extract core event routing to `event_router_core.py`
   - Keep type-specific handling in `event_router.py`

### Low Priority (Expected Patterns)

6. Package self-cycles in `__init__.py` - No action needed
7. Mixin forward references - Use string annotations

## Running the Audit

```bash
# Full audit
cd ai-service
PYTHONPATH=. python scripts/audit_circular_deps.py

# Focus on specific package
PYTHONPATH=. python scripts/audit_circular_deps.py --path app/coordination

# Verbose with dependency details
PYTHONPATH=. python scripts/audit_circular_deps.py --verbose

# JSON output
PYTHONPATH=. python scripts/audit_circular_deps.py --json > deps.json
```

## Impact Assessment

| Fix                          | Cycles Eliminated | Effort | Risk   |
| ---------------------------- | ----------------- | ------ | ------ |
| Extract event types          | ~30               | Low    | Low    |
| Explicit db imports          | ~5                | Low    | Low    |
| Lazy integrated_enhancements | ~20               | Medium | Medium |
| TYPE_CHECKING guards         | ~18               | Low    | Low    |
| **Total**                    | **~73**           | -      | -      |

Implementing the high-priority fixes would eliminate approximately 50% of circular dependencies.

## Appendix: Full Cycle List

Run `scripts/audit_circular_deps.py` for the complete list of all 150 cycles with their severity classifications.
