# Orphaned Daemons Documentation

**Created**: December 25, 2025
**Purpose**: Document daemons defined in DaemonType but not in production profiles

> **Deprecated:** This list is a December 2025 snapshot and is not maintained. For current profiles, see `ai-service/docs/DAEMON_REGISTRY.md` and `app/coordination/daemon_manager.py:DAEMON_PROFILES`.

## Summary

Out of 53 daemon types defined in `daemon_manager.py`, 19 are "orphaned" - meaning they are not included in any production profile (coordinator, training_node, ephemeral, selfplay, minimal).

## Daemon Categories

### Deprecated (4 daemons) - Scheduled for removal Q2 2026

| Daemon              | Lines | Replacement                      | Notes                                     |
| ------------------- | ----- | -------------------------------- | ----------------------------------------- |
| `SYNC_COORDINATOR`  | 62    | `AUTO_SYNC`                      | Legacy sync mechanism                     |
| `HEALTH_CHECK`      | 69    | `NODE_HEALTH_MONITOR`            | Replaced by unified health                |
| `HIGH_QUALITY_SYNC` | 63    | `AUTO_SYNC` + quality thresholds | Legacy quality-based sync                 |
| `ELO_SYNC`          | 64    | Tournament system                | Tournament-specific, not in training loop |

### Replaced by Better Implementations (4 daemons)

| Daemon                 | Lines | Replacement                         | Notes                                        |
| ---------------------- | ----- | ----------------------------------- | -------------------------------------------- |
| `MODEL_SYNC`           | 65    | `MODEL_DISTRIBUTION`                | Event-driven model sync                      |
| `QUEUE_MONITOR`        | 71    | `IDLE_RESOURCE` + `QUEUE_POPULATOR` | Superseded by active scheduling              |
| `SELFPLAY_COORDINATOR` | 81    | `IDLE_RESOURCE` + `QUEUE_POPULATOR` | Experimental, replaced by unified scheduling |
| `AUTO_PROMOTION`       | 137   | `UNIFIED_PROMOTION`                 | Both exist, UNIFIED uses PromotionController |

### Experimental/Not Production-Ready (3 daemons)

| Daemon                 | Lines | Purpose              | Status                                  |
| ---------------------- | ----- | -------------------- | --------------------------------------- |
| `GOSSIP_SYNC`          | 85    | P2P data replication | Experimental P2P mechanism              |
| `DATA_SERVER`          | 86    | Aria2 data server    | Legacy distributed download             |
| `CROSS_PROCESS_POLLER` | 76    | Cross-process events | Event infrastructure, background thread |

### Optional Enhancements (5 daemons)

| Daemon                | Lines | Purpose                   | When to Use                      |
| --------------------- | ----- | ------------------------- | -------------------------------- |
| `DISTILLATION`        | 89    | Create smaller models     | Model compression for deployment |
| `EXTERNAL_DRIVE_SYNC` | 91    | Backup to external drives | Manual backup operations         |
| `VAST_CPU_PIPELINE`   | 92    | CPU preprocessing on Vast | CPU-only Vast.ai instances       |
| `CACHE_COORDINATION`  | 173   | Model preloading          | Performance optimization         |
| `MAINTENANCE`         | 191   | Log rotation, cleanup     | System maintenance               |

### Advanced Orchestration (3 daemons)

| Daemon                  | Lines | Purpose              | Status                         |
| ----------------------- | ----- | -------------------- | ------------------------------ |
| `RECOVERY_ORCHESTRATOR` | 170   | Failure recovery     | Advanced, needs more testing   |
| `MULTI_PROVIDER`        | 182   | Multi-cloud failover | Not active in current setup    |
| `ADAPTIVE_RESOURCES`    | 179   | Dynamic scaling      | Advanced resource optimization |

## Recommendations

### Remove from Enum (Q2 2026)

- `SYNC_COORDINATOR`
- `HEALTH_CHECK`
- `HIGH_QUALITY_SYNC`
- `ELO_SYNC`

### Keep but Document as Optional

- `DISTILLATION` - Add to profiles when model compression needed
- `EXTERNAL_DRIVE_SYNC` - Add when backup automation needed
- `MAINTENANCE` - Consider adding to coordinator profile

### Archive (Superseded)

- `MODEL_SYNC` - Archive, MODEL_DISTRIBUTION is canonical
- `AUTO_PROMOTION` - Archive, UNIFIED_PROMOTION is canonical
- `SELFPLAY_COORDINATOR` - Archive, replaced by IDLE_RESOURCE + QUEUE_POPULATOR

### Keep for Future

- `GOSSIP_SYNC` - Keep for P2P improvements
- `RECOVERY_ORCHESTRATOR` - Keep for fault tolerance
- `MULTI_PROVIDER` - Keep for multi-cloud expansion

## Profile Coverage

| Profile       | Daemon Count | Notes                      |
| ------------- | ------------ | -------------------------- |
| coordinator   | 28           | Central orchestration      |
| training_node | 19           | GPU training nodes         |
| ephemeral     | 7            | Vast.ai/spot instances     |
| selfplay      | 4            | Minimal selfplay only      |
| minimal       | 1            | Just EVENT_ROUTER          |
| full          | 53           | All daemons (testing only) |
