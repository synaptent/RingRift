# Circular Dependency Map

**Last Updated**: December 29, 2025
**Total Cycles Found**: 127 (93 critical, 19 low severity)
**Modules Scanned**: 1,778
**Total Dependencies**: 5,851

## Overview

This document maps circular import dependencies in the ai-service codebase. While Python handles most circular imports gracefully via lazy loading, excessive cycles increase cognitive load and can cause import-time failures.

## Critical Cycle Categories

### Category 1: Coordination → Training → Coordination (38 cycles)

The largest cycle family involves bidirectional dependencies between coordination and training modules.

**Root Cause**: Coordination modules need training utilities (model loading, Elo), while training modules emit events to coordination infrastructure.

**Key Cycles**:

```
app.coordination.daemon_manager
    → app.coordination.auto_sync_daemon
    → app.distributed.quality_extractor
    → app.training.elo_service
    → app.coordination.helpers
    → app.coordination
```

**Mitigation**:

- Lazy imports in `elo_service.py` using `if TYPE_CHECKING:`
- Event emission via string-based routing (no direct import)
- Factory functions defer instantiation

**Future Fix**: Extract `EloService` to `app.core.elo` (no coordination deps)

---

### Category 2: Config → Training → Config (12 cycles)

Configuration modules import training utilities for defaults, while training imports config.

**Key Cycles**:

```
app.config.unified_config
    → app.training.integrated_enhancements
    → app.training.elo_weighting
    → app.quality.unified_quality
    → app.config.unified_config
```

**Mitigation**:

- `unified_config.py` uses `typing.TYPE_CHECKING` for training types
- Config values are primitives (no class instances)

**Future Fix**: Move quality scoring config to `app.config.quality_config.py`

---

### Category 3: Database → Validation → Database (8 cycles)

Database modules import validators, validators import database for schema checks.

**Key Cycles**:

```
app.db.unified_recording
    → app.db.parity_validator
    → app.db.game_replay
    → app.db.unified_recording
```

**Mitigation**:

- Parity validator is lazily imported only when needed
- Schema validation happens post-initialization

**Future Fix**: Extract `SchemaValidator` to separate module

---

### Category 4: Metrics → Training → Metrics (7 cycles)

Metrics collection imports training modules for context, training imports metrics for reporting.

**Key Cycles**:

```
app.metrics.orchestrator
    → app.training.quality_bridge
    → app.metrics.orchestrator
```

**Mitigation**:

- Quality bridge uses string-based metric keys
- Metrics are primitive dicts, not class instances

**Future Fix**: Define metric schemas in `app.metrics.schemas.py`

---

### Category 5: Model Lifecycle (5 cycles)

Model promotion, registry, and lifecycle have bidirectional dependencies.

**Key Cycles**:

```
app.integration.model_lifecycle
    → app.training.model_registry
    → app.training.promotion_controller
    → app.integration.model_lifecycle
```

**Mitigation**:

- All use `if TYPE_CHECKING:` for type hints
- Runtime imports only when method is called

**Future Fix**: Merge into single `app.models.lifecycle.py` module

---

### Category 6: Event System (5 cycles)

Event router, subscription store, and data events have circular refs.

**Key Cycles**:

```
app.coordination.event_router
    → app.coordination.subscription_store
    → app.coordination.dead_letter_queue
    → app.distributed.data_events
    → app.config.unified_config
    → app.coordination.event_router
```

**Mitigation**:

- Event types are enums (no runtime circular import)
- Subscription store uses string event names

**Future Fix**: Already minimal - consider acceptable

---

## Low Severity Cycles (19)

These cycles exist but are unlikely to cause issues:

| Cycle              | Modules                                       | Reason Acceptable              |
| ------------------ | --------------------------------------------- | ------------------------------ |
| Game engine legacy | `_game_engine_legacy` ↔ `rules.capture_chain` | Legacy compat, rarely imported |
| Test fixtures      | `tests.*` ↔ `app.*`                           | Test-only, not production      |
| CLI scripts        | `scripts.*` ↔ `app.*`                         | Entry points, not libraries    |

---

## Verification Commands

```bash
# Run full circular dependency audit
PYTHONPATH=. python scripts/audit_circular_deps.py

# Check specific module
PYTHONPATH=. python scripts/audit_circular_deps.py --module app.coordination

# Fail on new cycles (for CI)
PYTHONPATH=. python scripts/audit_circular_deps.py --fail-on-new --baseline 127
```

---

## Refactoring Roadmap

### Phase 1: Quick Wins (1-2 weeks)

1. Extract `EloService` to `app.core.elo`
2. Move quality config to `app.config.quality_config.py`
3. Add `--fail-on-new` to CI

### Phase 2: Architecture (3-4 weeks)

1. Merge model lifecycle modules
2. Extract schema validators
3. Define metric schemas

### Phase 3: Long-term (Q2 2026)

1. Full coordination/training decoupling
2. Consider dependency injection framework

---

## Adding New Dependencies

Before adding a new import:

1. Run `python scripts/audit_circular_deps.py --baseline 127`
2. If new cycle introduced, document here
3. Use lazy import pattern:

   ```python
   from typing import TYPE_CHECKING
   if TYPE_CHECKING:
       from app.training.foo import Bar

   def my_function():
       from app.training.foo import Bar  # Runtime import
       return Bar()
   ```

---

## Cycle Count History

| Date         | Total Cycles | Critical | Notes         |
| ------------ | ------------ | -------- | ------------- |
| Dec 29, 2025 | 127          | 93       | Initial audit |

---

_Generated by `scripts/audit_circular_deps.py`_
