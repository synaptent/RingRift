# RingRift AI Service - Comprehensive Action Plan

**Date**: December 25, 2025
**Status**: Phase 7 Complete → Consolidation & Optimization
**Cluster**: 23/24 nodes updated (95.8% coverage)

> Status: Historical snapshot (Dec 2025). Kept for reference; consult `ai-service/docs/README.md` for current guidance.

---

## Executive Summary

Five comprehensive assessments revealed that the RingRift AI training infrastructure is **~95% implemented but ~65-70% effective** due to final-mile integration gaps. The architecture is sound; the key issue is **wiring existing components together** rather than building new ones.

### Key Findings

| Area               | Implementation             | Effectiveness    | Gap                         |
| ------------------ | -------------------------- | ---------------- | --------------------------- |
| **Event System**   | 80+ event types            | 72% (8 orphaned) | Missing handlers            |
| **Feedback Loops** | All components exist       | 65-70%           | PFSP unused, loose coupling |
| **Data Sync**      | Multi-layer infrastructure | 75%              | Ephemeral data loss risk    |
| **Daemon System**  | 62 daemon types            | 60%              | 8 dormant, no systemd       |
| **Documentation**  | 6.3/10                     | -                | Missing ADRs, READMEs       |

---

## Part 1: Critical Wiring (Immediate Priority)

### 1.1 Wire PFSP Into Game Loop (HIGHEST PRIORITY)

**Problem**: PFSP exists (525 LOC) but `select_opponent()` and `record_pfsp_result()` are never called.

**Impact**: Selfplay uses random opponents instead of prioritizing ~50% win rate (maximum learning signal).

**Files**: `app/training/selfplay_runner.py`

**Change**: In each runner's `run_game()` method:

```python
def run_game(self, game_idx: int) -> GameResult:
    opponent = self.get_pfsp_opponent()  # ADD THIS
    # ... existing game logic ...
    if result.winner is not None:
        self.record_pfsp_result(...)  # ADD THIS AT END
    return result
```

**Locations to modify**:

- `GumbelMCTSSelfplayRunner.run_game()` (line ~1231)
- `HeuristicSelfplayRunner.run_game()` (line ~1115)
- `GNNSelfplayRunner.run_game()` (line ~1400)

**Estimated effort**: ~18 lines of code

---

### 1.2 Wire ADAPTIVE_PARAMS_CHANGED (Only Dead-End Event)

**Problem**: GauntletFeedbackController emits `ADAPTIVE_PARAMS_CHANGED` but nobody subscribes.

**Impact**: Temperature, search budget, and exploration adjustments never reach selfplay.

**File**: `app/training/selfplay_runner.py`

**Change**: Add subscription in `_subscribe_to_feedback_events()`:

```python
bus.subscribe(DataEventType.ADAPTIVE_PARAMS_CHANGED, self._on_adaptive_params_changed)

def _on_adaptive_params_changed(self, event):
    payload = event.payload
    if payload.get('config_key') != self.config.config_key:
        return
    if 'temperature_multiplier' in payload:
        self._temperature_scale *= payload['temperature_multiplier']
    # ... apply other adjustments
```

**Estimated effort**: ~25 lines of code

---

### 1.3 Handle Orphaned Events (8 Events)

**Events with zero subscribers**:

| Event                              | Expected Handler           | Action Required             |
| ---------------------------------- | -------------------------- | --------------------------- |
| `IDLE_RESOURCE_DETECTED`           | IdleResourceDaemon         | Spawn selfplay on idle GPUs |
| `NODE_OVERLOADED`                  | WorkDistributor            | Redistribute jobs           |
| `SYNC_STALLED`                     | SyncDaemon                 | Retry or alert              |
| `TRAINING_LOSS_ANOMALY`            | TrainingCoordinator        | Early stop, alert           |
| `TRAINING_EARLY_STOPPED`           | TrainingCoordinator        | Log, emit                   |
| `REGRESSION_DETECTED` (5 variants) | GauntletFeedbackController | Rollback model              |
| `PARITY_FAILURE_RATE_CHANGED`      | ParityMonitor              | Alert                       |

**Priority**: HIGH - These represent silent failures

---

## Part 2: Data Integrity (Next Priority)

### 2.1 Ephemeral Node Data Loss Prevention

**Problem**: Vast.ai nodes can lose 10-100 unsynced games per termination.

**Current state**:

- Write-through timeout: 10 seconds (too short)
- No persistent WAL on ephemeral nodes
- Async fallback has no guarantee

**Fixes**:

1. Increase write-through timeout to 30s (`ephemeral_sync.py`)
2. Implement local WAL before DB write
3. Add WAL recovery on node startup
4. Track `games_at_risk_on_termination` metric

**File**: `app/coordination/ephemeral_sync.py`

---

### 2.2 Enforce Minimum Replication

**Problem**: Single-copy games can accumulate and be lost permanently.

**Current state**:

- `replication_target: 2` in config but not enforced
- ReplicationMonitor only alerts, doesn't prevent

**Fixes**:

1. Add pre-training replication check (must have 2+ copies)
2. Block training if any game has single copy >60min old
3. Implement emergency replication daemon
4. Emit `REPLICATION_CRITICAL` alert

**File**: `app/coordination/replication_monitor.py`

---

### 2.3 Reduce Sync Latency

**Current**: 300 seconds (5 minutes) before games are discoverable

**Target**: 60 seconds

**Change** in `auto_sync_daemon.py`:

```python
MAIN_SYNC_INTERVAL = 60  # Was 300
GOSSIP_SYNC_INTERVAL = 30  # Was 60
```

---

## Part 3: Documentation Quick Wins

### Tier 1: Missing READMEs (< 1 hour total)

| Directory                 | Effort | Template                    |
| ------------------------- | ------ | --------------------------- |
| `coordination/resources/` | 15 min | Resource optimization docs  |
| `coordination/training/`  | 20 min | Training orchestration docs |
| `coordination/providers/` | 15 min | Cloud provider integration  |

### Tier 2: Consolidation Guides (1-2 hours each)

| Guide                        | Purpose                           |
| ---------------------------- | --------------------------------- |
| `HEALTH_MONITORING_GUIDE.md` | Map 9 health files to use cases   |
| `DAEMON_ADAPTERS_GUIDE.md`   | Document 11 daemon adapters       |
| `CONFIG_LOADING_GUIDE.md`    | Clarify config precedence         |
| `EVENT_TYPES_REGISTRY.md`    | Single source of truth for events |

### Tier 3: Architecture Decision Records

**Missing ADRs**:

- P2P vs centralized sync design
- Event bus consolidation (3 → 1)
- GPU vectorization approach
- Daemon lifecycle management
- Quality gate implementation

---

## Part 4: Automation Hardening

### 4.1 Create systemd Services

**Problem**: No auto-restart on node reboot.

**Files to create**:

- `scripts/systemd/ringrift-training.service`
- `scripts/systemd/ringrift-p2p.service`

**Template**:

```ini
[Unit]
Description=RingRift Training Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/ringrift/ai-service
ExecStart=/usr/bin/python3 scripts/master_loop.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

### 4.2 Add Daemon Watchdog

**Problem**: Crashed daemons aren't detected.

**Solution**: Create `scripts/watchdog.py`:

- Monitor daemon health endpoints
- Restart crashed daemons
- Alert if restart count > threshold
- Log to `/logs/watchdog.log`

---

### 4.3 Remove Single Point of Failure

**Problem**: mac-studio coordinator is single point of failure.

**Solutions**:

1. Run 2 coordinators (active-passive)
2. Use P2P consensus for critical decisions
3. Implement automatic failover
4. Test failover scenarios

---

## Part 5: Deprecated Code Cleanup

### 5.1 Archive Legacy Files

**Files confirmed safe to archive**:

- `_game_engine_legacy.py` (191KB, no imports)
- `_neural_net_legacy.py` (no imports)
- `registry_backup.py`

**Process**:

1. `grep -r "filename" app/` to verify no imports
2. Move to `archive/` with migration notes
3. Update any references

---

### 5.2 Dormant Daemons

**8 daemon types with incomplete integration**:

| Daemon                | Status                              | Action             |
| --------------------- | ----------------------------------- | ------------------ |
| `ELO_SYNC`            | Replaced by feedback loops          | Deprecate          |
| `MODEL_SYNC`          | Replaced by ModelDistributionDaemon | Deprecate          |
| `HEALTH_CHECK`        | Replaced by NodeHealthMonitor       | Remove             |
| `DISTILLATION`        | No integration                      | Document or remove |
| `UNIFIED_PROMOTION`   | Overshadowed                        | Merge or remove    |
| `EXTERNAL_DRIVE_SYNC` | Cluster-specific                    | Document           |
| `VAST_CPU_PIPELINE`   | Provider-specific                   | Document           |

---

## Part 6: Metrics & Observability

### Add These Metrics

```python
# Data Safety
ringrift_games_at_risk_on_ephemeral_termination
ringrift_single_copy_games
ringrift_replication_factor_avg

# Sync Performance
ringrift_sync_latency_p50/p99
ringrift_games_synced_per_cycle

# Feedback Loop Health
ringrift_dead_end_events_count
ringrift_feedback_loop_latency_seconds
ringrift_pfsp_opponent_selections_total
```

---

## Prioritized Execution Order

### Today (4-6 hours)

1. **Wire PFSP into game loop** (~1 hour)
   - 3 `run_game()` methods × 6 lines each = 18 LOC

2. **Wire ADAPTIVE_PARAMS_CHANGED** (~30 min)
   - Add subscription + handler = 25 LOC

3. **Create 3 missing READMEs** (~50 min)
   - resources/, training/, providers/

### This Week

4. **Fix ephemeral data loss** (~2 hours)
   - Increase write-through timeout
   - Add games_at_risk tracking

5. **Reduce sync latency** (~30 min)
   - Change interval constants

6. **Create Health Monitoring Guide** (~1 hour)
   - Map 9 health files

7. **Handle remaining orphaned events** (~3 hours)
   - Add handlers for 8 event types

### Next Week

8. **Create systemd services** (~2 hours)
9. **Create daemon watchdog** (~3 hours)
10. **Complete documentation guides** (~4 hours)
11. **Archive legacy code** (~1 hour)

---

## Success Metrics

| Metric                      | Before | After     | Verification                |
| --------------------------- | ------ | --------- | --------------------------- |
| PFSP opponent selection     | 0%     | 100%      | Logs show "PFSP selected:"  |
| Dead-end events             | 8+     | 0         | All events have subscribers |
| Feedback loop effectiveness | 65%    | 90%       | Training adapts to gauntlet |
| Sync latency                | 300s   | 60s       | Games discoverable <1 min   |
| Documentation score         | 6.3/10 | 8/10      | READMEs + ADRs exist        |
| 24/7 uptime                 | Manual | Automatic | 7+ days no intervention     |
| Ephemeral data loss risk    | HIGH   | LOW       | WAL + 30s timeout           |

---

## Files Modified Summary

### Critical Wiring (Part 1)

- `app/training/selfplay_runner.py` - PFSP + ADAPTIVE_PARAMS

### Data Integrity (Part 2)

- `app/coordination/ephemeral_sync.py` - Write-through timeout
- `app/coordination/auto_sync_daemon.py` - Sync intervals
- `app/coordination/replication_monitor.py` - Enforcement

### Documentation (Part 3)

- `app/coordination/resources/README.md` (NEW)
- `app/coordination/training/README.md` (NEW)
- `app/coordination/providers/README.md` (NEW)
- `app/coordination/HEALTH_MONITORING_GUIDE.md` (NEW)

### Automation (Part 4)

- `scripts/systemd/ringrift-training.service` (NEW)
- `scripts/watchdog.py` (NEW)

---

## Cluster Status (Dec 25, 2025)

**Updated**: 23/24 nodes (95.8%)
**Failed**: nebius-l40s-2 (connection timeout)

All Vast.ai, RunPod, Vultr, Hetzner nodes updated successfully.
Lambda Labs nodes remain offline (support ticket pending).

---

## Next Steps

After completing PFSP wiring (highest impact item), focus on:

1. **Run selfplay test** to verify PFSP is selecting opponents
2. **Monitor feedback events** to verify loop closure
3. **Check gauntlet results** for training adaptation
4. **Verify data sync** improvements

The system is well-architected; completing these final-mile integrations will bring effectiveness from ~65% to ~90%.
