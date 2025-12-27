# Daemon Startup Order Verification

**Date:** December 27, 2025
**Status:** Production Reference

This document describes the correct daemon startup order, critical dependencies, and how to verify proper initialization.

---

## Critical Startup Order

The daemon startup order is critical for proper event handling. Event-consuming daemons MUST start before event-emitting daemons.

### Correct Order (master_loop.py)

```
1. EVENT_ROUTER           # Event bus - MUST be first
2. DAEMON_WATCHDOG        # Health monitoring infrastructure
3. FEEDBACK_LOOP          # Subscribes to training events
4. DATA_PIPELINE          # Subscribes to sync events
5. AUTO_SYNC              # Emits DATA_SYNC_COMPLETED
6. MODEL_DISTRIBUTION     # Emits MODEL_DISTRIBUTED
7. EVALUATION             # Subscribes to TRAINING_COMPLETED
8. AUTO_PROMOTION         # Subscribes to EVALUATION_COMPLETED
9. [Other daemons...]
```

### Why Order Matters

If sync daemons start BEFORE pipeline daemons:

```
AUTO_SYNC starts
    ↓
Emits DATA_SYNC_COMPLETED
    ↓
No subscribers yet! Event is LOST
    ↓
DATA_PIPELINE starts (too late)
    ↓
Never receives sync events
    ↓
NPZ export never triggers automatically
```

---

## Dependency Graph

### Tier 1: Infrastructure (Start First)

| Daemon          | Dependencies | Purpose           |
| --------------- | ------------ | ----------------- |
| EVENT_ROUTER    | None         | Central event bus |
| DAEMON_WATCHDOG | EVENT_ROUTER | Health monitoring |

### Tier 2: Event Consumers (Start Before Emitters)

| Daemon               | Dependencies | Subscribes To                            |
| -------------------- | ------------ | ---------------------------------------- |
| FEEDBACK_LOOP        | EVENT_ROUTER | TRAINING*\*, EVALUATION*_, QUALITY\__    |
| DATA_PIPELINE        | EVENT_ROUTER | DATA*SYNC*_, NEW*GAMES*_, REGRESSION\_\* |
| SELFPLAY_COORDINATOR | EVENT_ROUTER | NODE*RECOVERED, BACKPRESSURE*\*          |

### Tier 3: Event Emitters

| Daemon             | Dependencies | Emits                                     |
| ------------------ | ------------ | ----------------------------------------- |
| AUTO_SYNC          | Tier 2       | DATA_SYNC_COMPLETED, SYNC_STARTED         |
| MODEL_DISTRIBUTION | Tier 2       | MODEL_DISTRIBUTED, DISTRIBUTION_FAILED    |
| EVALUATION         | Tier 2       | EVALUATION_COMPLETED, EVALUATION_PROGRESS |

### Tier 4: Response Daemons

| Daemon          | Dependencies         | Purpose                     |
| --------------- | -------------------- | --------------------------- |
| AUTO_PROMOTION  | EVALUATION           | Promote passing models      |
| QUALITY_MONITOR | DATA_PIPELINE        | Monitor data quality        |
| IDLE_RESOURCE   | SELFPLAY_COORDINATOR | Spawn selfplay on idle GPUs |

---

## Verification Commands

### Check Daemon Status

```bash
# Via DaemonManager
python scripts/launch_daemons.py --status

# Via P2P status
curl -s http://localhost:8770/status | jq '.daemons'

# Programmatically
python -c "
from app.coordination.daemon_manager import get_daemon_manager
dm = get_daemon_manager()
for name, info in dm.get_all_daemon_health().items():
    print(f'{name}: {info.get(\"status\", \"unknown\")}')
"
```

### Verify Event Wiring

```bash
# Check that DATA_PIPELINE subscribes to sync events
grep -n "DATA_SYNC_COMPLETED" app/coordination/data_pipeline_orchestrator.py

# Check that FEEDBACK_LOOP subscribes to training events
grep -n "TRAINING_COMPLETED" app/coordination/feedback_loop_controller.py

# Check subscription count for key events
python -c "
from app.coordination.event_router import get_router
router = get_router()
print(f'DATA_SYNC_COMPLETED subscribers: {len(router._subscribers.get(\"DATA_SYNC_COMPLETED\", []))}')
print(f'TRAINING_COMPLETED subscribers: {len(router._subscribers.get(\"TRAINING_COMPLETED\", []))}')
"
```

### Verify Startup Order in Logs

```bash
# Check daemon startup order in logs
grep "Starting daemon" logs/daemon_manager.log | head -20

# Verify no "lost event" warnings
grep -E "(lost|orphan|unhandled)" logs/event_router.log
```

---

## Common Issues & Fixes

### Issue: Sync Events Lost

**Symptom:** NPZ export never triggers after sync completes.

**Cause:** DATA_PIPELINE started after AUTO_SYNC.

**Fix:** In master_loop.py, ensure startup order:

```python
# Good order
await dm.start(DaemonType.DATA_PIPELINE)
await dm.start(DaemonType.AUTO_SYNC)

# Bad order
await dm.start(DaemonType.AUTO_SYNC)  # Emits events
await dm.start(DaemonType.DATA_PIPELINE)  # Too late!
```

### Issue: Health Loop Not Running

**Symptom:** Crashed daemons don't restart.

**Cause:** Health monitoring loop not started after individual daemon starts.

**Fix:** Verify `_ensure_health_loop_running()` is called after each `start()`.

```python
# In daemon_manager.py
async def start(self, daemon_type: DaemonType) -> bool:
    success = await self._start_daemon(daemon_type)
    if success:
        self._ensure_health_loop_running()  # CRITICAL
    return success
```

### Issue: Event Type Mismatch

**Symptom:** Events emitted but never received.

**Cause:** Emitter uses string literal, subscriber uses enum value.

**Example:**

```python
# Emitter (wrong)
emit("DATA_SYNC_COMPLETED", payload)

# Subscriber (expects enum value)
bus.subscribe(DataEventType.DATA_SYNC_COMPLETED, handler)
# DataEventType.DATA_SYNC_COMPLETED.value = "sync_completed"  # Different!
```

**Fix:** Use `_get_event_type_value()` helper:

```python
from app.coordination.data_events import DataEventType

def _get_event_type_value(name: str) -> str:
    """Map human-readable name to actual event type value."""
    mapping = {
        "DATA_SYNC_COMPLETED": DataEventType.DATA_SYNC_COMPLETED.value,
        "DATA_SYNC_STARTED": DataEventType.DATA_SYNC_STARTED.value,
    }
    return mapping.get(name, name)
```

---

## Test Coverage

### Unit Tests

```bash
# Run startup order tests
pytest tests/integration/coordination/test_daemon_startup_order.py -v

# Key tests:
# - test_feedback_loop_before_sync
# - test_data_pipeline_before_sync
# - test_event_router_first
# - test_health_loop_starts
```

### Integration Tests

```bash
# Full daemon lifecycle test
pytest tests/integration/coordination/test_daemon_lifecycle.py -v

# Event flow integration
pytest tests/integration/coordination/test_event_flow.py -v
```

---

## Configuration Reference

### DaemonManager Settings

```python
# In app/config/coordination_defaults.py
@dataclass(frozen=True)
class DaemonDefaults:
    HEALTH_CHECK_INTERVAL: float = 30.0  # seconds
    STARTUP_GRACE_PERIOD: float = 120.0  # seconds before health checks
    MAX_RESTART_ATTEMPTS: int = 5
    RESTART_BACKOFF_BASE: float = 2.0  # exponential backoff
```

### Daemon Registry

See `app/coordination/daemon_registry.py` for the complete daemon registry with:

- Dependencies (`depends_on` field)
- Categories (sync, event, health, pipeline, resource)
- Health check intervals
- Auto-restart settings

---

## Monitoring

### Prometheus Metrics

If metrics are enabled:

```
ringrift_daemon_status{daemon="AUTO_SYNC"} 1  # 1=running, 0=stopped
ringrift_daemon_restarts_total{daemon="AUTO_SYNC"} 0
ringrift_daemon_health_check_failures{daemon="AUTO_SYNC"} 0
```

### Alerting

Configure alerts for:

- `ringrift_daemon_status == 0` for critical daemons
- `ringrift_daemon_restarts_total > 3` within 1 hour
- `ringrift_daemon_health_check_failures > 2` consecutive

---

## Appendix: Event Reference

### Key Events for Pipeline Coordination

| Event                  | Emitter             | Subscribers                |
| ---------------------- | ------------------- | -------------------------- |
| DATA_SYNC_COMPLETED    | AUTO_SYNC           | DataPipeline               |
| TRAINING_COMPLETED     | TrainingCoordinator | FeedbackLoop, Evaluation   |
| EVALUATION_COMPLETED   | Evaluation          | AutoPromotion, Curriculum  |
| MODEL_PROMOTED         | AutoPromotion       | ModelDistribution          |
| REGRESSION_DETECTED    | ModelWatchdog       | DataPipeline, FeedbackLoop |
| BACKPRESSURE_ACTIVATED | ResourceMonitor     | SelfplayCoordinator        |

See `docs/EVENT_REFERENCE.md` for the complete event catalog.
