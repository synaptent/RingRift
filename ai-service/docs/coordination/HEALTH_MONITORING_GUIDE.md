# Health Monitoring Guide

**Last Updated**: January 3, 2026

This guide explains when to use the different health monitoring components in RingRift. The system has three layers of health monitoring, each serving different purposes.

## Quick Reference

| What You Need                | Module to Use    | Function                     |
| ---------------------------- | ---------------- | ---------------------------- |
| Should I pause the pipeline? | `health_facade`  | `should_pause_pipeline()`    |
| Is a specific node healthy?  | `health_facade`  | `get_node_health(node_id)`   |
| Overall cluster health score | `health_facade`  | `get_system_health_score()`  |
| Which nodes are healthy?     | `health_facade`  | `get_healthy_nodes()`        |
| Daemon health status         | `daemon_manager` | `dm.get_daemon_health(type)` |
| Should new jobs be allowed?  | `health_facade`  | `should_allow_new_jobs()`    |

## The Three Layers

### Layer 1: System Health (Cluster-Wide)

**Purpose**: Aggregate health score for the entire cluster.

**When to use**: Making cluster-wide decisions (pause pipeline, reduce load).

```python
from app.coordination.health_facade import (
    get_system_health_score,
    get_system_health_level,
    should_pause_pipeline,
    SystemHealthLevel,
)

# Get overall health score (0.0 to 1.0)
score = get_system_health_score()
print(f"Cluster health: {score:.1%}")

# Get health level (enum)
level = get_system_health_level()
if level == SystemHealthLevel.CRITICAL:
    print("CRITICAL: Major issues detected")
elif level == SystemHealthLevel.DEGRADED:
    print("DEGRADED: Some issues but operational")
else:
    print("HEALTHY: All systems nominal")

# Check if pipeline should pause
should_pause, reasons = should_pause_pipeline()
if should_pause:
    print(f"Pipeline paused: {', '.join(reasons)}")
```

**Score Components**:

- **Node Availability (40%)**: What % of nodes are reachable
- **Circuit Health (25%)**: What % of circuit breakers are closed
- **Error Rate (20%)**: Rolling error rate across daemons
- **Recovery Progress (15%)**: How many nodes are recovering

### Layer 2: Node Health (Per-Node)

**Purpose**: Track individual node health for routing decisions.

**When to use**: Choosing which node to sync from, route work to.

```python
from app.coordination.health_facade import (
    get_node_health,
    get_healthy_nodes,
    get_unhealthy_nodes,
    NodeHealthState,
)

# Get specific node health
node = get_node_health("nebius-h100-1")
if node:
    print(f"State: {node.state.value}")
    print(f"Last seen: {node.last_seen}")
    print(f"Error count: {node.error_count}")

    if node.state == NodeHealthState.HEALTHY:
        # Safe to route work here
        pass
    elif node.state == NodeHealthState.DEGRADED:
        # Use with caution
        pass
    elif node.state == NodeHealthState.UNHEALTHY:
        # Avoid routing here
        pass

# Get all healthy nodes for routing
healthy = get_healthy_nodes()
print(f"Healthy nodes: {len(healthy)}")

# Get unhealthy nodes for monitoring
unhealthy = get_unhealthy_nodes()
for node_id in unhealthy:
    print(f"⚠️ {node_id} is unhealthy")
```

### Layer 3: Daemon Health (Per-Daemon)

**Purpose**: Monitor individual daemon health for auto-restart.

**When to use**: Daemon manager health checks, debugging specific daemons.

```python
from app.coordination.daemon_manager import get_daemon_manager
from app.coordination.daemon_types import DaemonType

dm = get_daemon_manager()

# Get specific daemon health
health = dm.get_daemon_health(DaemonType.AUTO_SYNC)
if health:
    print(f"Running: {health.running}")
    print(f"Healthy: {health.healthy}")
    print(f"Last error: {health.last_error}")

# Get all daemon health
all_health = dm.get_all_daemon_health()
for daemon_type, info in all_health.items():
    status = "✓" if info.healthy else "✗"
    print(f"{status} {daemon_type.value}: {info.status}")
```

## Common Patterns

### Pattern 1: Job Scheduling Gate

Before scheduling new work, check if the system is healthy:

```python
from app.coordination.health_facade import should_allow_new_jobs

def schedule_selfplay_job(config_key: str) -> bool:
    """Schedule selfplay only if cluster is healthy."""
    if not should_allow_new_jobs():
        logger.warning("Cluster unhealthy, deferring job")
        return False

    # Proceed with scheduling
    return submit_job(config_key)
```

### Pattern 2: Sync Target Selection

Choose healthy nodes for data sync:

```python
from app.coordination.health_facade import get_healthy_nodes, get_node_health

def select_sync_target(config_key: str) -> str | None:
    """Select healthiest node for sync."""
    healthy_nodes = get_healthy_nodes()

    if not healthy_nodes:
        return None

    # Prefer nodes with most data for this config
    best_node = None
    best_score = -1

    for node_id in healthy_nodes:
        health = get_node_health(node_id)
        if health and health.error_count < 5:
            score = calculate_data_locality(node_id, config_key)
            if score > best_score:
                best_score = score
                best_node = node_id

    return best_node
```

### Pattern 3: Pipeline Pause with Recovery

Handle pipeline pause and resume:

```python
from app.coordination.health_facade import should_pause_pipeline
import asyncio

async def pipeline_controller():
    """Main pipeline control loop with health awareness."""
    while True:
        should_pause, reasons = should_pause_pipeline()

        if should_pause:
            logger.warning(f"Pipeline paused: {reasons}")
            # Wait for recovery
            while should_pause_pipeline()[0]:
                await asyncio.sleep(60)
            logger.info("Pipeline resumed")

        # Normal pipeline work
        await process_next_stage()
        await asyncio.sleep(10)
```

### Pattern 4: Daemon with Health Check

Implement health check in your daemon:

```python
from app.coordination.handler_base import HandlerBase, HealthCheckResult

class MyDaemon(HandlerBase):
    def health_check(self) -> HealthCheckResult:
        """Report daemon health to daemon manager."""
        # Determine health based on your criteria
        is_healthy = (
            self.stats.errors_count < 10 and
            time.time() - self.stats.last_activity < 300
        )

        return HealthCheckResult(
            healthy=is_healthy,
            status="healthy" if is_healthy else "degraded",
            message=f"Processed {self.stats.events_processed} events",
            details={
                "events_processed": self.stats.events_processed,
                "errors_count": self.stats.errors_count,
                "last_activity": self.stats.last_activity,
            },
        )
```

## Health Status Values

### SystemHealthLevel (Cluster-Wide)

| Level      | Score Range | Meaning                        |
| ---------- | ----------- | ------------------------------ |
| `HEALTHY`  | 0.8 - 1.0   | All systems operational        |
| `DEGRADED` | 0.5 - 0.8   | Some issues, still operational |
| `CRITICAL` | 0.0 - 0.5   | Major issues, reduced capacity |

### NodeHealthState (Per-Node)

| State       | Meaning                     | Action             |
| ----------- | --------------------------- | ------------------ |
| `HEALTHY`   | Node responding normally    | Safe to use        |
| `DEGRADED`  | Some errors but operational | Use with caution   |
| `UNHEALTHY` | Many errors, unreliable     | Avoid using        |
| `OFFLINE`   | Not responding              | Skip in routing    |
| `UNKNOWN`   | No data available           | Probe before using |

### HealthCheckResult (Per-Daemon)

| Field     | Type | Purpose                            |
| --------- | ---- | ---------------------------------- |
| `healthy` | bool | Is daemon healthy?                 |
| `status`  | str  | "healthy", "degraded", "unhealthy" |
| `message` | str  | Human-readable status              |
| `details` | dict | Additional metrics                 |

## Which Layer to Use?

```
Making cluster-wide decisions?
├── Should pipeline pause? → should_pause_pipeline()
├── Overall health? → get_system_health_score()
└── Can we schedule jobs? → should_allow_new_jobs()

Routing work to specific nodes?
├── Which nodes are up? → get_healthy_nodes()
├── Is this node OK? → get_node_health(node_id)
└── Node error history? → get_node_health().error_count

Debugging daemon issues?
├── Is daemon running? → dm.get_daemon_health(type)
├── All daemon status? → dm.get_all_daemon_health()
└── Restart unhealthy? → dm.restart_unhealthy_daemons()
```

## HTTP Endpoints

Health is exposed via HTTP for external monitoring:

| Endpoint       | Port | Purpose                        |
| -------------- | ---- | ------------------------------ |
| `GET /health`  | 8790 | Liveness probe (200 if alive)  |
| `GET /ready`   | 8790 | Readiness probe (200 if ready) |
| `GET /status`  | 8790 | Detailed daemon status JSON    |
| `GET /metrics` | 8790 | Prometheus-format metrics      |
| `GET /status`  | 8770 | P2P cluster status             |

```bash
# Check if daemon manager is healthy
curl http://localhost:8790/health

# Get detailed status
curl http://localhost:8790/status | jq

# P2P cluster status
curl http://localhost:8770/status | python3 -c '
import sys, json
d = json.load(sys.stdin)
print(f"Leader: {d.get(\"leader_id\")}")
print(f"Alive: {d.get(\"alive_peers\")}")
'
```

## See Also

- `app/coordination/health_facade.py` - Unified health interface
- `app/coordination/unified_health_manager.py` - System-level health
- `app/coordination/health_check_orchestrator.py` - Node-level health
- `app/coordination/daemon_manager.py` - Daemon lifecycle
- `app/coordination/contracts.py` - HealthCheckResult dataclass
