# ADR-002: Daemon Lifecycle Management

**Status**: Accepted
**Date**: December 2025
**Author**: RingRift AI Team

## Context

The RingRift training infrastructure runs 62+ background daemons across 20+ nodes:

- Selfplay generators
- Data sync daemons
- Training coordinators
- Model distribution daemons
- Health monitors
- etc.

Managing these daemons manually was error-prone:

- Daemons crashed without restart
- Startup order dependencies violated
- Resource contention between daemons
- No visibility into daemon health

## Decision

Implement a centralized **DaemonManager** (`app/coordination/daemon_manager.py`) with:

### DaemonType Enum

66 daemon types organized by category:

- Core: `EVENT_ROUTER`, `P2P_BACKEND`
- Data: `AUTO_SYNC`, `EPHEMERAL_SYNC`, `SYNC_BANDWIDTH`
- Training: `TRAINING_COORDINATOR`, `GAUNTLET_FEEDBACK`
- Distribution: `MODEL_DISTRIBUTION`, `NPZ_DISTRIBUTION`
- Monitoring: `HEALTH_MONITOR`, `QUALITY_MONITOR`

### Daemon Profiles

Pre-configured daemon sets for different node roles:

- `coordinator`: Full daemon set for coordinator node
- `training_node`: Data pipeline + training daemons
- `selfplay`: Lightweight selfplay generation
- `ephemeral`: Aggressive sync for Vast.ai nodes

### Startup Order

Dependencies declared via `depends_on`:

```python
self.register_factory(
    DaemonType.GAUNTLET_FEEDBACK,
    self._create_gauntlet_feedback,
    depends_on=[DaemonType.EVENT_ROUTER],
)
```

### Health Monitoring

Each daemon reports health via `CoordinatorProtocol`:

- `name`: Unique daemon identifier
- `status`: RUNNING, STOPPED, FAILED
- `get_metrics()`: Custom metrics dict
- `health_check()`: Self-check result

## Consequences

### Positive

- Single entry point for all daemon management
- Automatic restart on failure (with backoff)
- Dependency-ordered startup
- Unified logging and metrics

### Negative

- Single point of failure (DaemonManager itself)
- Profile selection requires node role knowledge
- Some daemons still run outside DaemonManager (legacy)

## Implementation Notes

- Use `scripts/launch_daemons.py` for CLI daemon management
- DaemonManager emits `DAEMON_STARTED`, `DAEMON_STOPPED`, `DAEMON_FAILED` events
- Watchdog process restarts DaemonManager if it crashes
- Phase 7: AutoRollbackHandler wired during `_subscribe_to_critical_events()`

## Related ADRs

- ADR-001: Event-Driven Architecture (daemons use events)
