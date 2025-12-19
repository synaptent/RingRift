# Runbooks

Operational runbooks for RingRift AI service alerts and incidents.

## Available Runbooks

| Runbook                                                  | Alert                 | Severity | Description                  |
| -------------------------------------------------------- | --------------------- | -------- | ---------------------------- |
| [COORDINATOR_ERROR.md](COORDINATOR_ERROR.md)             | CoordinatorError      | Critical | Coordinator in error state   |
| [CLUSTER_HEALTH_CRITICAL.md](CLUSTER_HEALTH_CRITICAL.md) | ClusterHealthCritical | Critical | Cluster health below 50%     |
| [SYNC_HOST_CRITICAL.md](SYNC_HOST_CRITICAL.md)           | CriticalSyncHosts     | Critical | Hosts in critical sync state |

## Runbook Structure

Each runbook follows a standard format:

1. **Alert Information** - Severity, component, team
2. **Description** - What the alert means
3. **Impact** - What's affected
4. **Diagnosis** - Steps to investigate
5. **Resolution** - Actions to fix the issue
6. **Prevention** - How to avoid future occurrences
7. **Escalation** - When and how to escalate
8. **Related Alerts** - Other alerts to check

## Quick Reference

### Admin Endpoints

```bash
# Coordinator health
curl -H "X-Admin-Key: $KEY" http://localhost:8001/admin/health/coordinators

# Full system health
curl -H "X-Admin-Key: $KEY" http://localhost:8001/admin/health/full

# Sync coordinator status
curl -H "X-Admin-Key: $KEY" http://localhost:8001/admin/sync/status

# Trigger sync (optional categories: games, training, models)
curl -X POST -H "X-Admin-Key: $KEY" \
  "http://localhost:8001/admin/sync/trigger?categories=games&categories=training"

# Prometheus metrics
curl http://localhost:8001/metrics
```

### Common Commands

```bash
# Check coordinator status
python -c "
from app.coordination.coordinator_base import get_coordinator_registry
print(get_coordinator_registry().get_health_summary())
"

# Check sync status
python -c "
from app.coordination.sync_coordinator import SyncCoordinator
print(SyncCoordinator.get_instance().get_stats())
"
```

## Related Documentation

- [COORDINATION_SYSTEM.md](../architecture/COORDINATION_SYSTEM.md) - System architecture
- [OPERATIONAL_RUNBOOK.md](../infrastructure/OPERATIONAL_RUNBOOK.md) - General operations
- [CLUSTER_OPERATIONS_RUNBOOK.md](../infrastructure/CLUSTER_OPERATIONS_RUNBOOK.md) - Cluster management
