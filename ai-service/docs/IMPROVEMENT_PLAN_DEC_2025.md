# RingRift AI Service Improvement Plan - December 2025

## Executive Summary

Comprehensive assessment completed on December 29, 2025. The RingRift AI training system is **75% complete** with strong fundamentals but faces architectural sprawl and cluster underutilization challenges.

**Key Metrics:**

- Model Elo: 1,000-1,111 (2-player), 850-950 (multiplayer)
- Cluster: 7/36 nodes active (19% utilization)
- Test Coverage: 63% coordination, 38% training, 22% P2P
- Training Data: 34,683 games across 12 configs (111 GB)

---

## Priority 1: Critical (Do This Week)

### 1.1 Restore Cluster Connectivity [2-3 hours]

**Impact**: 4-6x throughput gain

- Only 7/36 configured nodes are alive
- 15 nodes failed SSH during update (Tailscale/SSH issues)
- Immediate actions:
  1. SSH health check on RunPod/Nebius/Vast nodes
  2. Restart P2P orchestrator on recovered nodes
  3. Verify Tailscale connectivity

### 1.2 Generate 4-Player Training Data [4-6 hours]

**Impact**: Unlock multiplayer product release

Current data shortage:
| Config | Games | Target | Gap |
|--------|-------|--------|-----|
| hex8_4p | 372 | 2,000 | 1,628 |
| hexagonal_4p | 126 | 1,000 | 874 |

Commands:

```bash
python scripts/master_loop.py --configs hex8_4p,hexagonal_4p --selfplay-games 500
```

### 1.3 Fix GPU Batch State .item() Bottleneck [4 hours]

**Impact**: 25-35% GPU selfplay speedup

Location: `app/ai/gpu_batch_state.py:1024-1033, 1136-1226`

Current: 7x `.item()` calls per move × 64 games × 100 moves = 44,800 GPU syncs
Fix: Batch CPU transfer before loop (pattern exists at line 1061)

---

## Priority 2: High (This Sprint)

### 2.1 Test Coverage for Critical Paths [40-60 hours]

**Impact**: Catch bugs before cluster deployment

| Module                        | LOC   | Tests Needed | Priority |
| ----------------------------- | ----- | ------------ | -------- |
| gpu_mcts_selfplay.py          | 681   | 25-30        | P0       |
| selfplay_runner.py            | 2,112 | 35-40        | P0       |
| daemon_manager.py             | 3,804 | 30-35        | P0       |
| data_pipeline_orchestrator.py | 2,205 | 25-30        | P1       |

### 2.2 Event System Unification [3 days]

**Impact**: Prevent silent event routing bugs

Issues:

- 202 event types across 15 files
- Inconsistent naming: `TRAINING_COMPLETED` vs `sync_completed`
- Missing subscriber wiring discovered Dec 27

Solution:

- Create TypeSafe event registry with Pydantic models
- Add startup validation for subscriber wiring
- Consolidate into 3 event modules

### 2.3 Async Database Writes [2 days]

**Impact**: 15-25% cluster throughput improvement

Current: Synchronous `conn.commit()` blocks other nodes
Fix: Background task queue for database writes

```python
await background_db_queue.add_task((game_data, elo_updates))
```

---

## Priority 3: Medium (Next 2 Weeks)

### 3.1 Module Consolidation [5 days]

**Impact**: 40% import reduction, easier maintenance

- 248 coordination modules → target 200
- daemon_runners.py has 90 imports (structural smell)
- Create `app/coordination/core/` with base classes

### 3.2 Configuration Consolidation [3 days]

**Impact**: Prevent config drift

Current: 26 config modules with overlapping concerns
Target: 8 modules with Pydantic validation

### 3.3 Code Deduplication [3 days]

**Impact**: ~2,500 LOC reduction

- tier_eval_config.py duplicated (802 LOC)
- 1,455 LOC dead code in P2P orchestrator
- 200 LOC unused imports

---

## Priority 4: Lower (Ongoing)

### 4.1 Documentation Improvements

- Merge 3 CLAUDE.md files
- Add module architecture diagrams
- Document daemon startup order

### 4.2 Performance Optimizations

- Data loader multiprocessing (20-30% training speedup)
- Vectorize feature extraction loops (5-8% speedup)
- Compress database sync transfers (8-12% sync speedup)

### 4.3 Type Safety Improvements

- 868 instances of `: Any` across 209 files
- Add type hints to critical paths

---

## Summary Statistics

### Code Quality

- Total LOC: ~1.4M
- Dead code: ~2,500 LOC
- Complexity hotspots: 5 files >5,000 lines
- TODO/FIXME: 1,097 comments

### Architecture

- Modules: 248 coordination, 187 training, 95 P2P
- Daemon types: 77 active, 6 deprecated
- Event types: 207
- Base classes: HandlerBase (53 handlers), MonitorBase (15 monitors)

### Test Coverage

- Unit tests: 878 files, 448K LOC
- Pass rate: 98.5% (11,793 tests)
- Untested LOC: ~62K across 80+ modules

### Performance

- GPU speedup: 6-57x (hardware dependent)
- Remaining .item() calls: ~14
- Optimization potential: 60-120% throughput

---

## Scheduled for Q2 2026

### data_events Import Migration (Documented Dec 29, 2025)

**Status**: Tracked, scheduled for Q2 2026 removal
**Scope**: 78 import lines across 40+ files in `app/coordination/`
**Current State**: Deprecation warnings active, module functional

All imports from `app.distributed.data_events` should be migrated to:

- `DataEventType` → `app.coordination.event_router.DataEventType`
- `emit_*` functions → `app.coordination.event_emitters.*`
- `get_event_bus` → `app.coordination.event_router.get_event_bus`

Files affected (by import count):

- daemon_manager.py (5 imports)
- pipeline_event_handler_mixin.py (5 imports)
- dead_letter_queue.py (3 imports)
- data_consolidation_daemon.py (4 imports)
- npz_combination_daemon.py (4 imports)
- auto_sync_daemon.py (3 imports)
- p2p_recovery_daemon.py (4 imports)
- 33+ other files (1-2 imports each)

**Note**: The deprecated module works correctly with warnings. Migration is safe to defer.

---

## Success Metrics

By end of January 2025:

- [ ] 25+ cluster nodes active (from 7)
- [ ] 2,000+ games for all 12 configs
- [ ] Elo 1,100+ for 2-player, 1,050+ for multiplayer
- [ ] 80%+ test coverage for P0 modules
- [ ] 48h+ unattended operation demonstrated

---

## Appendix: Agent Reports

Full assessment reports from 5 exploration agents:

1. Code Quality Issues - 2,457 LOC dead code, 1,097 TODOs
2. Architectural Quality - 248 modules, 90-import daemon_runners
3. Test Coverage Gaps - 62K LOC untested, 80+ modules
4. High-Level Goals - 75% complete, cluster bottleneck
5. Performance Bottlenecks - 60-120% optimization potential
