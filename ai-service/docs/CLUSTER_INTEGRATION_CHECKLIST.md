# Cluster Integration Checklist

Last Updated: December 27, 2025

This checklist ensures all cluster components are properly integrated and communicating.

## Pre-Flight Checks

### 1. P2P Orchestrator Startup

- [ ] **P2P Manager Event Subscriptions**
  - `selfplay_scheduler.subscribe_to_events()` - Called during init
  - `job_manager.subscribe_to_events()` - Called during init
  - `training_coordinator.subscribe_to_events()` - Called during init
  - Verify via `/status` endpoint: `_event_subscription_status.all_healthy = true`

- [ ] **P2P Connectivity Verification**
  - Voter quorum check (minimum 3/5 voters alive)
  - Master loop calls `_verify_p2p_connectivity()` on startup
  - Emits `P2P_CLUSTER_UNHEALTHY` if quorum not met

### 2. Daemon Startup Order

The correct startup order ensures event subscribers are ready before emitters:

```
1. EVENT_ROUTER          # Event infrastructure
2. NODE_HEALTH_MONITOR   # Health tracking
3. CLUSTER_MONITOR       # Cluster awareness
4. HEALTH_SERVER         # Health endpoints
5. FEEDBACK_LOOP         # Event subscriber (listens to TRAINING_COMPLETED, etc.)
6. DATA_PIPELINE         # Event subscriber (listens to DATA_SYNC_COMPLETED, etc.)
7. AUTO_SYNC             # Event emitter (emits DATA_SYNC_COMPLETED)
8. ELO_SYNC              # Event emitter
```

### 3. Event Routing

- [ ] **EventRouter Coverage**: 157/153 events mapped (97%+)
- [ ] **Cross-Process Events**: Verify key events bridge correctly
  - `DATA_SYNC_COMPLETED` → triggers pipeline export
  - `SELFPLAY_COMPLETE` → triggers AUTO_SYNC immediate sync
  - `TRAINING_COMPLETED` → triggers evaluation queue
  - `EVALUATION_COMPLETED` → triggers model promotion check

## Health Check Verification

### Daemon Health Checks

All critical daemons should implement `health_check()` returning:

```python
{
    "status": "healthy" | "unhealthy",
    "operations_count": int,
    "errors_count": int,
    "last_error": str | None,
}
```

Critical daemons with health_check():

- [x] AUTO_SYNC
- [x] SELFPLAY_COORDINATOR
- [x] TRAINING_TRIGGER
- [x] FEEDBACK_LOOP
- [x] CLUSTER_WATCHDOG
- [x] NODE_RECOVERY
- [x] EVALUATION

### P2P Manager Health Checks

All 7 P2P managers implement `health_check()`:

- [x] StateManager
- [x] JobManager
- [x] NodeSelector
- [x] SyncPlanner
- [x] TrainingCoordinator
- [x] SelfplayScheduler
- [x] LoopManager (if exists)

## Event Subscription Verification

### TrainingCoordinator Events

```python
# Subscribed to:
- SELFPLAY_COMPLETE       # → Queue training when data ready
- DATA_SYNC_COMPLETED     # → Check data availability
- EVALUATION_COMPLETED    # → Handle promotion decision
- REGRESSION_DETECTED     # → Trigger rollback
```

### JobManager Events

```python
# Subscribed to:
- HOST_OFFLINE           # → Reschedule jobs from offline host
- HOST_ONLINE            # → Consider host for new jobs
```

### SelfplayScheduler Events

```python
# Subscribed to:
- NODE_RECOVERED         # → Refresh node availability
```

### DataPipelineOrchestrator Events

```python
# Subscribed to all DataEventType pipeline events:
- SELFPLAY_COMPLETE
- DATA_SYNC_STARTED/COMPLETED
- TRAINING_STARTED/COMPLETED
- EVALUATION_STARTED/COMPLETED
- PROMOTION_STARTED/COMPLETED
- EXPORT_STARTED/COMPLETED
- ORPHAN_GAMES_DETECTED/REGISTERED
```

## Configuration Constants

### SQLite Timeouts (app/config/coordination_defaults.py)

```python
from app.config.coordination_defaults import SQLiteDefaults, get_sqlite_timeout

# Use appropriate tier:
# QUICK (2s): Health checks, existence tests
# READ (5s): Standard read operations
# STANDARD (10s): Normal read/write
# WRITE (30s): Registry, Elo updates
# HEAVY (60s): Database consolidation
# MERGE (120s): Database merge operations
```

### Network Timeouts

```python
from app.config.coordination_defaults import get_timeout

timeout = get_timeout("http")   # 30s
timeout = get_timeout("ssh")    # 60s
timeout = get_timeout("health") # 5s
timeout = get_timeout("rsync")  # 30s
```

## Verification Commands

### Check Cluster Health

```bash
# P2P status
curl -s http://localhost:8770/status | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"Healthy: {d.get(\"healthy\")}, Leader: {d.get(\"leader_id\")}, Alive: {d.get(\"alive_peers\")}")'

# Daemon status
python scripts/launch_daemons.py --status

# Event subscription status
curl -s http://localhost:8770/status | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"Events: {d.get(\"_event_subscription_status\")}")'
```

### Check Sync Status

```bash
# AUTO_SYNC daemon health
curl -s http://localhost:8770/daemon/AUTO_SYNC/health

# Sync statistics
python -c "from app.coordination.sync_facade import get_sync_facade; print(get_sync_facade().get_stats())"
```

### Check Training Pipeline

```bash
# Pipeline stage status
python -c "from app.coordination.data_pipeline_orchestrator import get_pipeline_orchestrator; print(get_pipeline_orchestrator().get_status())"
```

## Common Issues and Fixes

### Events Not Propagating

1. Check daemon startup order (subscribers before emitters)
2. Verify `subscribe_to_events()` called on managers
3. Check `_subscribed` flags are True

### Sync Not Triggering

1. Verify `SELFPLAY_COMPLETE` emitted after selfplay
2. Check `AutoSyncDaemon._on_selfplay_complete()` handler
3. Verify DATA_PIPELINE daemon is running

### Stale Data Training

1. Check data freshness: `--check-data-freshness` flag
2. Verify sync completed: `DATA_SYNC_COMPLETED` event
3. Check pipeline stage progression

## Deprecated Modules

Use unified replacements for:

| Deprecated                     | Replacement                            |
| ------------------------------ | -------------------------------------- |
| `cluster_data_sync.py`         | `AutoSyncDaemon(strategy="broadcast")` |
| `ephemeral_sync.py`            | `AutoSyncDaemon(strategy="ephemeral")` |
| `model_distribution_daemon.py` | `UnifiedDistributionDaemon`            |
| `npz_distribution_daemon.py`   | `UnifiedDistributionDaemon`            |
| `node_health_monitor.py`       | `health_check_orchestrator.py`         |
| `system_health_monitor.py`     | `unified_health_manager.py`            |

## Integration Test Commands

```bash
# Full integration test
python -m pytest tests/integration/test_cluster_integration.py -v

# Event routing test
python -c "
from app.coordination.event_router import get_event_router
router = get_event_router()
print(f'Mappings: {len(router._mappings)}')
print(f'Subscribers: {sum(len(s) for s in router._subscribers.values())}')
"

# Daemon factory test
python -c "
from app.coordination.daemon_factory import DaemonFactory
factory = DaemonFactory()
for dtype in ['AUTO_SYNC', 'DATA_PIPELINE', 'FEEDBACK_LOOP']:
    spec = factory.get_spec(dtype)
    print(f'{dtype}: {spec.import_path}.{spec.class_name}')
"
```
