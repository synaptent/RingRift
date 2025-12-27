# Coordination Module Test Coverage Roadmap

**Last Updated:** 2025-12-27
**Status:** Active Implementation

---

## Overview

This document tracks test coverage for `app/coordination/` modules and provides a prioritized implementation plan.

**Current State:**

- Total coordination modules: 148
- Modules with tests: 111 (75%)
- Modules without tests: 49 (25%)
- Total untested LOC: ~22,000

---

## Prioritization Tiers

### Tier 1: Critical (Data Integrity & Job Safety)

These modules directly affect data integrity or can cause destructive behavior if buggy.

| Module                     | LOC | Risk                                                 | Priority |
| -------------------------- | --- | ---------------------------------------------------- | -------- |
| `resource_targets.py`      | 906 | Controls job dispatch; bad logic = wasted GPU cycles | P0       |
| `job_reaper.py`            | 549 | Kills jobs; overly aggressive = data loss            | P0       |
| `daemon_adapters.py`       | 749 | Adapter layer for all daemons; failures cascade      | P0       |
| `npz_validation.py`        | 381 | Training data validation; bugs = corrupt training    | P0       |
| `transfer_verification.py` | 649 | File integrity checks; bugs = silent corruption      | P1       |
| `sync_safety.py`           | 365 | Prevents concurrent sync conflicts                   | P1       |

**Estimated Tests:** ~150 tests
**Estimated Effort:** 3-4 days

### Tier 2: High Priority (Core Infrastructure)

Core infrastructure that affects cluster reliability and performance.

| Module                         | LOC | Risk                         | Priority |
| ------------------------------ | --- | ---------------------------- | -------- |
| `continuous_loop.py`           | 721 | Background loop framework    | P1       |
| `dead_letter_queue.py`         | 758 | Failed event recovery        | P1       |
| `adaptive_resource_manager.py` | 650 | Resource allocation          | P1       |
| `utilization_optimizer.py`     | 731 | GPU utilization optimization | P1       |
| `work_distributor.py`          | 606 | Work distribution logic      | P1       |
| `host_health_policy.py`        | 695 | Node health decisions        | P2       |
| `dynamic_thresholds.py`        | 444 | Adaptive thresholds          | P2       |

**Estimated Tests:** ~200 tests
**Estimated Effort:** 4-5 days

### Tier 3: Medium Priority (Supporting Infrastructure)

Important supporting modules but less critical to core operations.

| Module                          | LOC  | Risk                           | Priority |
| ------------------------------- | ---- | ------------------------------ | -------- |
| `helpers.py`                    | 1160 | Utility functions              | P2       |
| `base_orchestrator.py`          | 429  | Base class for orchestrators   | P2       |
| `coordinator_config.py`         | 529  | Configuration loading          | P2       |
| `duration_scheduler.py`         | 728  | Job duration scheduling        | P2       |
| `stage_events.py`               | 585  | Pipeline stage events          | P2       |
| `ephemeral_data_guard.py`       | 659  | Ephemeral node data protection | P2       |
| `quality_monitor_daemon.py`     | 602  | Data quality monitoring        | P2       |
| `unified_inventory.py`          | 602  | Resource inventory tracking    | P2       |
| `unified_node_health_daemon.py` | 622  | Node health daemon             | P2       |
| `unified_registry.py`           | 672  | Model registry                 | P2       |

**Estimated Tests:** ~250 tests
**Estimated Effort:** 5-6 days

### Tier 4: Low Priority (Simple/Deprecated/Internal)

Simple type definitions, deprecated modules, or internal utilities.

| Module                              | LOC | Notes                                   |
| ----------------------------------- | --- | --------------------------------------- |
| `enums.py`                          | 95  | Simple enumerations - low value         |
| `types.py`                          | 240 | Type definitions - TypeScript-like      |
| `node_status.py`                    | 223 | Status dataclasses - already typed      |
| `system_health_monitor.py`          | 83  | Deprecated - use unified_health_manager |
| `contracts.py`                      | 384 | Contract definitions - low complexity   |
| `core_base.py`                      | 142 | Base class only                         |
| `core_utils.py`                     | 203 | Simple utilities                        |
| `curriculum_weights.py`             | 107 | Weight calculations                     |
| `daemon_factory_implementations.py` | 259 | Factory methods                         |
| `daemon_metrics.py`                 | 247 | Metrics collection                      |
| `daemon_registry.py`                | 456 | Registry pattern                        |
| `alert_types.py`                    | 381 | Alert definitions                       |
| `event_mappings.py`                 | 420 | Event type mappings                     |
| `sync_base.py`                      | 346 | Sync base class                         |
| `sync_constants.py`                 | 272 | Constants only                          |
| `tracing.py`                        | 522 | Tracing infrastructure                  |
| `task_decorators.py`                | 539 | Async decorators                        |
| `master_loop_guard.py`              | 111 | Guard logic                             |
| `coordinator_dependencies.py`       | 344 | Dependency tracking                     |
| `core_events.py`                    | 378 | Event definitions                       |
| `utils.py`                          | 682 | General utilities                       |

**Estimated Tests:** ~150 tests (many can be skipped)
**Estimated Effort:** 2-3 days (if needed)

---

## Skipped/Deferred Modules

These modules don't need tests or are deferred:

| Module                     | LOC | Reason                                     |
| -------------------------- | --- | ------------------------------------------ |
| `slurm_backend.py`         | 789 | SLURM not used - future integration        |
| `p2p_auto_deployer.py`     | 720 | Complex E2E testing required               |
| `async_training_bridge.py` | 387 | Integration-level testing only             |
| `node_health_monitor.py`   | 386 | Deprecated - use health_check_orchestrator |

---

## Implementation Plan

### Phase 1: Critical Coverage (Week 1)

**Goal:** Ensure Tier 1 modules are fully tested.

1. **resource_targets.py** (~40 tests)
   - Target calculation logic
   - GPU allocation formulas
   - Edge cases (0 GPUs, saturation)

2. **job_reaper.py** (~30 tests)
   - Reap threshold logic
   - Protected job detection
   - Concurrent modification safety

3. **daemon_adapters.py** (~35 tests)
   - Adapter lifecycle
   - Event forwarding
   - Error handling

4. **npz_validation.py** (~25 tests)
   - Array shape validation
   - Type checking
   - Corruption detection

5. **transfer_verification.py** (~30 tests)
   - Checksum validation
   - Partial transfer detection
   - Retry logic

6. **sync_safety.py** (~20 tests)
   - Lock acquisition
   - Timeout handling
   - Concurrent access

### Phase 2: Infrastructure Coverage (Week 2)

**Goal:** Cover Tier 2 core infrastructure.

1. **continuous_loop.py** (~35 tests)
2. **dead_letter_queue.py** (~30 tests)
3. **adaptive_resource_manager.py** (~35 tests)
4. **utilization_optimizer.py** (~40 tests)
5. **work_distributor.py** (~30 tests)
6. **host_health_policy.py** (~30 tests)

### Phase 3: Supporting Modules (Week 3+)

**Goal:** Cover remaining Tier 3 modules based on bug history.

Prioritize based on:

- Recent bug occurrences
- User-facing impact
- Dependency count

---

## Test Templates

### Unit Test Template

```python
"""Tests for app/coordination/{module}.py."""

from __future__ import annotations

import pytest

from app.coordination.{module} import (
    MainClass,
    helper_function,
)


class TestMainClass:
    """Tests for MainClass."""

    def test_initialization(self):
        """Test default initialization."""
        obj = MainClass()
        assert obj is not None

    def test_core_method(self):
        """Test core functionality."""
        obj = MainClass()
        result = obj.core_method(input_data)
        assert result == expected

    def test_error_handling(self):
        """Test error conditions."""
        obj = MainClass()
        with pytest.raises(ValueError):
            obj.core_method(invalid_input)


class TestHelperFunction:
    """Tests for helper_function."""

    @pytest.mark.parametrize("input_val,expected", [
        (1, 2),
        (10, 20),
        (0, 0),
    ])
    def test_parametrized(self, input_val, expected):
        """Test with various inputs."""
        assert helper_function(input_val) == expected
```

### Integration Test Template

```python
"""Integration tests for {module} with related components."""

import pytest

from app.coordination.{module} import MainClass
from app.coordination.related_module import RelatedClass


@pytest.fixture
def setup_integration():
    """Set up integration test fixtures."""
    main = MainClass()
    related = RelatedClass()
    return main, related


class TestIntegration:
    """Integration tests."""

    def test_component_interaction(self, setup_integration):
        """Test interaction between components."""
        main, related = setup_integration
        main.connect(related)
        result = main.process()
        assert result.is_valid
```

---

## Tracking Progress

Update this section as tests are added:

| Date       | Module                   | Tests Added | Coverage % |
| ---------- | ------------------------ | ----------- | ---------- |
| 2025-12-27 | database_sync_manager.py | 47          | 95%        |
| -          | -                        | -           | -          |

---

## See Also

- [AGENTS.md](../../AGENTS.md) - Coding guidelines
- [Test Guidelines](../DEVELOPER_GUIDE.md#testing) - Testing conventions
- [Coordination Architecture](../CLUSTER_INTEGRATION_GUIDE.md) - System overview
