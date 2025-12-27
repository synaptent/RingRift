# P2P Background Loops

Background loop implementations extracted from `p2p_orchestrator.py` for better modularity, testability, and maintainability.

Created as part of Phase 4 of the P2P orchestrator decomposition (December 2025).

## Architecture

All loops inherit from `BaseLoop`, which provides:

- **Consistent error handling** with exponential backoff
- **Lifecycle management** (start, stop, pause, resume)
- **Metrics collection** (run count, error count, average duration)
- **Health status reporting** for monitoring

```
BaseLoop (abstract)
    │
    ├── QueuePopulatorLoop       - Maintains work queue depth (leader only)
    ├── EloSyncLoop              - Keeps unified_elo.db consistent across cluster
    │
    ├── Coordination Loops (coordination_loops.py)
    │   ├── AutoScalingLoop      - Scales cluster based on work queue depth
    │   └── HealthAggregationLoop - Aggregates health metrics from all nodes
    │
    ├── Data Loops (data_loops.py)
    │   ├── ModelSyncLoop        - Synchronizes models across cluster nodes
    │   └── DataAggregationLoop  - Aggregates training data from distributed nodes
    │
    ├── Job Loops (job_loops.py)
    │   ├── JobReaperLoop        - Cleans up stale and stuck jobs
    │   └── IdleDetectionLoop    - Detects idle nodes for work assignment
    │
    └── Network Loops (network_loops.py)
        ├── IpDiscoveryLoop      - Updates node IP addresses
        └── TailscaleRecoveryLoop - Recovers Tailscale connections
```

## Usage

### Single Loop

```python
from scripts.p2p.loops import QueuePopulatorLoop

loop = QueuePopulatorLoop(
    get_role=lambda: orchestrator.role,
    get_selfplay_scheduler=lambda: orchestrator.selfplay_scheduler,
    notifier=orchestrator.notifier,
)
await loop.run_forever()  # Blocks until stopped
```

### Multiple Loops with LoopManager

```python
from scripts.p2p.loops import LoopManager, QueuePopulatorLoop, BackoffConfig

manager = LoopManager()

# Register loops
manager.register(QueuePopulatorLoop(
    get_role=lambda: role,
    get_selfplay_scheduler=lambda: scheduler,
))

# Start all loops
await manager.start_all()

# Check status
status = manager.get_status()
print(f"Running: {status['running_loops']}")

# Stop all loops gracefully
await manager.stop_all()
```

### Custom Loop

```python
from scripts.p2p.loops import BaseLoop, BackoffConfig

class MyLoop(BaseLoop):
    def __init__(self):
        super().__init__(
            name="my_loop",
            interval=30.0,  # Run every 30 seconds
            backoff_config=BackoffConfig(
                initial_delay=5.0,
                max_delay=300.0,
                multiplier=2.0,
            ),
        )

    async def _run_once(self) -> None:
        """Override to implement loop logic."""
        # Your logic here
        pass

    async def _on_start(self) -> None:
        """Optional: called once when loop starts."""
        pass

    async def _on_stop(self) -> None:
        """Optional: called once when loop stops."""
        pass
```

## Available Loops

### QueuePopulatorLoop

Maintains minimum work queue depth by populating selfplay jobs.

**Features:**

- Only runs on leader node
- Uses SelfplayScheduler for priority-based config selection
- Adjusts interval when all Elo targets are met (60s → 300s)
- Lazy initialization of QueuePopulator from YAML config

**Configuration:**

- Default interval: 60 seconds
- Initial delay: 30 seconds (startup grace period)
- All-targets-met interval: 300 seconds

**Usage:**

```python
from scripts.p2p.loops import QueuePopulatorLoop

loop = QueuePopulatorLoop(
    get_role=lambda: orchestrator.role,
    get_selfplay_scheduler=lambda: orchestrator.selfplay_scheduler,
    notifier=orchestrator.notifier,
    config_path=Path("config/unified_loop.yaml"),  # Optional
    interval=60.0,  # Optional
)
```

## Configuration

### BackoffConfig

Controls exponential backoff on errors:

```python
BackoffConfig(
    initial_delay=5.0,    # First retry delay (seconds)
    max_delay=300.0,      # Maximum retry delay (seconds)
    multiplier=2.0,       # Backoff multiplier
    jitter=True,          # Add random jitter to prevent thundering herd
)
```

### Loop-Specific Config

Most loops load configuration from `config/unified_loop.yaml`:

```yaml
queue_populator:
  enabled: true
  min_queue_depth: 50
  target_elo: 2000
  configs:
    - hex8_2p
    - square8_2p
    # ... etc
```

## Monitoring

### Get Loop Status

```python
# Single loop
status = loop.get_status()
# Returns:
# {
#     "name": "queue_populator",
#     "running": True,
#     "paused": False,
#     "enabled": True,
#     "interval": 60.0,
#     "run_count": 42,
#     "error_count": 0,
#     "consecutive_errors": 0,
#     "average_duration_ms": 150.5,
#     "last_run_time": 1703577600.0,
#     "last_error": None,
# }

# LoopManager
manager_status = manager.get_status()
# Returns:
# {
#     "total_loops": 5,
#     "running_loops": 5,
#     "paused_loops": 0,
#     "error_loops": 0,
#     "loops": { ... per-loop status ... }
# }
```

### P2P Status Endpoint

Loop status is exposed via `/status` endpoint when integrated with P2P orchestrator.

## Migration from p2p_orchestrator.py

To migrate an existing loop method from `p2p_orchestrator.py`:

1. Create new file in `scripts/p2p/loops/` (e.g., `my_loop.py`)
2. Create class inheriting from `BaseLoop`
3. Move loop logic to `_run_once()` method
4. Replace `self.` references with injected dependencies
5. Add export to `__init__.py`
6. Update orchestrator to use new loop class

**Before (in orchestrator):**

```python
async def _my_loop(self):
    while self.running:
        try:
            # Logic using self.some_dependency
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Error: {e}")
            await asyncio.sleep(300)
```

**After (in separate file):**

```python
class MyLoop(BaseLoop):
    def __init__(self, get_dependency: Callable[[], Any]):
        super().__init__(name="my_loop", interval=60.0)
        self._get_dependency = get_dependency

    async def _run_once(self) -> None:
        dependency = self._get_dependency()
        # Logic using dependency
```

## Loop Implementation Status

| Loop                  | Source File               | Status   | Notes                               |
| --------------------- | ------------------------- | -------- | ----------------------------------- |
| QueuePopulatorLoop    | `queue_populator_loop.py` | Complete | Leader-only, integrates scheduler   |
| EloSyncLoop           | `elo_sync_loop.py`        | Complete | Syncs unified_elo.db across cluster |
| AutoScalingLoop       | `coordination_loops.py`   | Complete | Scale up/down based on queue depth  |
| HealthAggregationLoop | `coordination_loops.py`   | Complete | Collects CPU/GPU/disk metrics       |
| ModelSyncLoop         | `data_loops.py`           | Complete | Prioritizes hex8/square8 configs    |
| DataAggregationLoop   | `data_loops.py`           | Complete | Aggregates selfplay databases       |
| JobReaperLoop         | `job_loops.py`            | Complete | Enforces job timeouts               |
| IdleDetectionLoop     | `job_loops.py`            | Complete | Detects nodes with no activity      |
| IpDiscoveryLoop       | `network_loops.py`        | Complete | Fetches node IPs via HTTP/Tailscale |
| TailscaleRecoveryLoop | `network_loops.py`        | Complete | Restarts Tailscale on failure       |

All loops were extracted in December 2025 as part of Phase 4 of the P2P decomposition.

## Testing

```bash
# Run loop tests
cd ai-service
pytest tests/unit/scripts/p2p/loops/ -v

# Test specific loop
pytest tests/unit/scripts/p2p/loops/test_queue_populator_loop.py -v
```

## See Also

- `scripts/p2p/managers/README.md` - Manager classes for domain logic
- `scripts/p2p/handlers/README.md` - HTTP endpoint handlers
- `app/coordination/README.md` - Coordination infrastructure
