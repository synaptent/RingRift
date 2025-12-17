# Resource Management Architecture

This document describes the resource management system for RingRift AI, ensuring consistent 80% utilization limits across all components.

## Overview

The resource management system prevents CPU, GPU, RAM, and disk overloading by enforcing consistent limits across the codebase. All long-running operations must check resource availability before proceeding.

## Resource Limits

| Resource | Max Utilization | Warning Threshold | Reason                                   |
| -------- | --------------- | ----------------- | ---------------------------------------- |
| CPU      | 80%             | 70%               | Standard operating margin                |
| Memory   | 80%             | 70%               | Standard operating margin                |
| GPU      | 80%             | 70%               | Standard operating margin                |
| Disk     | 70%             | 65%               | Tighter limit because cleanup takes time |

## Core Module: `app/utils/resource_guard.py`

The unified resource guard provides all resource checking functionality:

```python
from app.utils.resource_guard import (
    # Simple checks
    check_disk_space,      # Check disk availability
    check_memory,          # Check RAM availability
    check_cpu,             # Check CPU load
    check_gpu_memory,      # Check GPU VRAM

    # Combined check
    can_proceed,           # Check all resources at once

    # Blocking wait
    wait_for_resources,    # Wait until resources available
    require_resources,     # Require resources or fail

    # Status reporting
    get_resource_status,   # Get full resource status dict
    print_resource_status, # Print formatted status

    # Context manager
    ResourceGuard,         # Context manager for resource-safe ops

    # Async support
    AsyncResourceLimiter,  # Async limiter with backoff

    # Decorator
    respect_resource_limits, # Decorator for resource-aware functions

    # Constants
    LIMITS,                # ResourceLimits dataclass
)
```

## Usage Patterns

### 1. Simple Check Before Operation

```python
from app.utils.resource_guard import check_disk_space, check_memory

def save_training_data():
    if not check_disk_space(required_gb=5.0):
        logger.warning("Insufficient disk space, skipping save")
        return

    if not check_memory(required_gb=2.0):
        logger.warning("Insufficient memory, skipping save")
        return

    # Proceed with save
    ...
```

### 2. Combined Check

```python
from app.utils.resource_guard import can_proceed

def run_selfplay():
    if not can_proceed(check_disk=True, check_mem=True, check_gpu=True):
        logger.warning("Resources not available")
        return

    # Proceed with selfplay
    ...
```

### 3. Wait for Resources

```python
from app.utils.resource_guard import wait_for_resources

def start_training():
    # Wait up to 5 minutes for resources
    if not wait_for_resources(timeout=300):
        raise RuntimeError("Resources not available after 5 minutes")

    # Proceed with training
    ...
```

### 4. Context Manager

```python
from app.utils.resource_guard import ResourceGuard

def process_games():
    with ResourceGuard(disk_required_gb=5.0, mem_required_gb=4.0) as guard:
        if not guard.ok:
            logger.warning("Resources not available")
            return

        # Proceed with processing
        ...
```

### 5. Async Support

```python
from app.utils.resource_guard import AsyncResourceLimiter

async def distributed_training():
    limiter = AsyncResourceLimiter(
        disk_required_gb=10.0,
        mem_required_gb=8.0,
        gpu_required_gb=4.0,
    )

    async with limiter.acquire("training"):
        # Resources guaranteed available
        await train_model()
```

### 6. Decorator

```python
from app.utils.resource_guard import respect_resource_limits

@respect_resource_limits(disk_gb=5.0, mem_gb=4.0, gpu_gb=2.0)
async def train_model():
    # Automatically waits for resources before executing
    ...
```

## Integration with Scripts

All major scripts should import and use resource_guard:

```python
# Unified resource guard - 80% utilization limits
try:
    from app.utils.resource_guard import (
        can_proceed as resource_can_proceed,
        check_disk_space,
        check_memory,
        check_gpu_memory,
        require_resources,
        LIMITS as RESOURCE_LIMITS,
    )
    HAS_RESOURCE_GUARD = True
except ImportError:
    HAS_RESOURCE_GUARD = False
    resource_can_proceed = lambda **kwargs: True
    check_disk_space = lambda *args, **kwargs: True
    check_memory = lambda *args, **kwargs: True
    check_gpu_memory = lambda *args, **kwargs: True
    require_resources = lambda *args, **kwargs: True
    RESOURCE_LIMITS = None
```

## Related Modules

### `app/coordination/resource_targets.py`

Provides tier-specific utilization targets for different host types:

- HIGH_END (96GB+ RAM): Target 65% utilization
- MID_TIER (32-64GB): Target 60% utilization
- LOW_TIER (16-32GB): Target 55% utilization
- CPU_ONLY: Target 50% utilization

### `app/coordination/safeguards.py`

Provides circuit breakers and backpressure mechanisms:

- Circuit breaker: Prevents spawning after repeated failures
- Spawn rate tracking: Limits new process creation rate
- Resource thresholds: Enforces 80% limits

### `app/config/config_validator.py`

Validates configuration files and ensures resource limits are consistent:

- Validates unified_loop.yaml
- Checks resource limit consistency
- Reports warnings for unsafe configurations

## Testing

Tests are located in `tests/test_resource_guard.py`:

```bash
PYTHONPATH=. pytest tests/test_resource_guard.py -v
```

## Configuration Validation

Run validation to check resource limit consistency:

```bash
PYTHONPATH=. python -c "from app.config.config_validator import validate_all_configs; print(validate_all_configs())"
```

## Enforcement Date

Resource limits were unified and enforced starting 2025-12-16:

- CPU: 80% max
- Memory: 80% max
- GPU: 80% max
- Disk: 70% max

All scripts were updated to use the unified resource_guard module.

## Graceful Degradation

The resource management system supports graceful degradation to reduce workload under resource pressure instead of failing completely.

### Degradation Levels

| Level | Name     | Max Ratio | Behavior                                           |
| ----- | -------- | --------- | -------------------------------------------------- |
| 0     | Normal   | <70%      | Full operation                                     |
| 1     | Light    | 70-85%    | Log warning, continue                              |
| 2     | Moderate | 85-95%    | Reduce num_games by 50%                            |
| 3     | Heavy    | 95-100%   | Reduce num_games by 75%, block low-priority spawns |
| 4     | Critical | ≥100%     | Abort operations, block all spawns                 |

### Priority Levels

Operations have assigned priorities that determine if they can proceed during degradation:

```python
class OperationPriority:
    BACKGROUND = 0     # Optional cleanup, stats collection
    LOW = 1            # Extra selfplay, backfill tasks
    NORMAL = 2         # Regular selfplay, data generation
    HIGH = 3           # Training, model evaluation
    CRITICAL = 4       # Data sync, model promotion, health checks
```

### Usage in Selfplay Scripts

```python
from app.utils.resource_guard import (
    should_proceed_with_priority,
    OperationPriority,
    get_degradation_level,
)

# Check if operation should proceed
degradation = get_degradation_level()
if degradation >= 4:  # CRITICAL
    logger.error("Resources at critical levels, aborting")
    sys.exit(1)
elif degradation >= 3:  # HEAVY
    if not should_proceed_with_priority(OperationPriority.NORMAL):
        args.num_games = max(10, args.num_games // 4)  # 75% reduction
elif degradation >= 2:  # MODERATE
    if not should_proceed_with_priority(OperationPriority.NORMAL):
        args.num_games = max(10, args.num_games // 2)  # 50% reduction
```

### Prometheus Metrics

Graceful degradation exposes Prometheus metrics for monitoring:

- `ringrift_resource_degradation_level`: Current degradation level (0-4)
- `ringrift_resource_disk_used_percent`: Current disk usage
- `ringrift_resource_memory_used_percent`: Current memory usage
- `ringrift_resource_cpu_used_percent`: Current CPU usage
- `ringrift_resource_gpu_used_percent`: Current GPU usage

### Alerting Rules

Prometheus alerts are configured in `monitoring/prometheus/rules/utilization_alerts.yml`:

| Alert                       | Threshold | Severity |
| --------------------------- | --------- | -------- |
| DiskApproachingLimit        | >65%      | warning  |
| DiskAtLimit                 | >70%      | critical |
| MemoryApproachingLimit      | >75%      | warning  |
| MemoryAtLimit               | >80%      | critical |
| GracefulDegradationActive   | level ≥2  | warning  |
| GracefulDegradationCritical | level ≥4  | critical |

### Recommendations

Get actionable recommendations based on current resource state:

```python
from app.utils.resource_guard import get_recommended_actions

actions = get_recommended_actions()
for action in actions:
    print(f"- {action}")
# Example output:
# - Disk at 72%: Archive old games or delete unused data
# - Memory OK at 65%
```

## Sync Tool Consolidation

The codebase has several data synchronization tools. To avoid confusion and reduce maintenance burden, some have been deprecated in favor of the canonical implementations.

### Active Sync Tools (Use These)

| Tool                                    | Purpose                       | When to Use                     |
| --------------------------------------- | ----------------------------- | ------------------------------- |
| `scripts/unified_data_sync.py`          | Primary data sync service     | General game/training data sync |
| `scripts/sync_models.py`                | Model checkpoint distribution | Deploying models to cluster     |
| `scripts/cluster_sync_coordinator.py`   | Cluster-wide coordination     | Orchestrating multi-node sync   |
| `scripts/aria2_data_sync.py`            | High-speed parallel downloads | Large file transfers, resumable |
| `scripts/external_drive_sync_daemon.py` | External storage sync + QA    | Archiving to external drives    |
| `scripts/elo_db_sync.py`                | Elo database sync CLI         | Manual/daemon Elo sync          |
| `app/tournament/elo_sync_manager.py`    | Elo sync library              | Programmatic Elo sync in Python |
| `app/p2p/gossip_sync.py`                | Decentralized P2P protocol    | NAT-traversing peer discovery   |

### Deprecated Sync Tools (Do Not Use)

| Tool                                  | Replacement                   | Reason                      |
| ------------------------------------- | ----------------------------- | --------------------------- |
| `scripts/cluster_sync_integration.py` | `cluster_sync_coordinator.py` | 85% functionality overlap   |
| `scripts/p2p_model_sync.py`           | `aria2_data_sync.py`          | Consolidate aria2 usage     |
| `scripts/streaming_data_collector.py` | `unified_data_sync.py`        | Merged into unified service |
| `scripts/collector_watchdog.py`       | `unified_data_sync.py`        | Merged into unified service |
| `scripts/sync_all_data.py`            | `unified_data_sync.py`        | Merged into unified service |

### Migration Guide

If you're using deprecated tools:

1. **cluster_sync_integration.py** → Use `cluster_sync_coordinator.py` with the same arguments
2. **p2p_model_sync.py** → Use `aria2_data_sync.py --mode models`
3. **streaming_data_collector.py** → Use `unified_data_sync.py --mode stream`

### Sync Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Sync Tool Hierarchy                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  unified_data_sync.py ─────────────┐                           │
│         ↓                          │                           │
│  cluster_sync_coordinator.py ──────┤                           │
│         ↓                          ├── Game/Training Data      │
│  aria2_data_sync.py ───────────────┘                           │
│                                                                 │
│  sync_models.py ─────────────────────── Model Checkpoints      │
│                                                                 │
│  elo_db_sync.py ─────────────────────── Elo Database           │
│  elo_sync_manager.py (library)                                  │
│                                                                 │
│  external_drive_sync_daemon.py ──────── Archive Storage        │
│                                                                 │
│  gossip_sync.py ─────────────────────── P2P Discovery          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Consolidation enforced starting 2025-12-16.
