# Known Issues

This document tracks known issues, workarounds, and their status.

## Critical Data Issues

### hex8_4p Data Corruption (FIXED Dec 26, 2025)

**Issue:** `hex8_4p_selfplay.db` contained 1,185 games with corrupted move data. All PLACE_RING moves had `to=None`.

**Root Cause:** Phase was extracted from post-move state instead of pre-move state. Terminal game moves (game_over phase) couldn't serialize positions correctly.

**Status:** FIXED in commit d840f4c4

**Impact:** Games generated before Dec 26, 2025 may have corrupted moves.

**Workaround:** Regenerate data using `scripts/selfplay.py`

---

## Cluster Issues

### Parity Gates Fail on Cluster Nodes

**Issue:** TypeScript parity gates require `npx` which isn't installed on cluster container images (Vast.ai, RunPod) or Nebius nodes.

**Symptoms:**

- `pending_gate` status in databases
- Parity validation scripts fail with "npx: command not found"

**Workaround:**

```bash
# Skip parity gates on cluster nodes
export RINGRIFT_ALLOW_PENDING_GATE=1

# Run parity validation locally (has npx) before syncing to cluster
python scripts/check_ts_python_replay_parity.py --db data/games/canonical_hex8.db
```

**Long-term Fix:** Add Node.js to cluster node images (requires image rebuild)

---

### SCP/Rsync Connection Resets

**Issue:** Some file transfers fail with "Connection reset by peer" error.

**Symptoms:**

- Large file transfers fail partway through
- Intermittent failures on Vast.ai and some RunPod nodes

**Workaround:** Use base64 transfer:

```bash
# Manual base64 transfer
cat local_file.npz | base64 | ssh user@host 'base64 -d > remote_file.npz'

# From Python
from scripts.lib.transfer import base64_push, robust_push

# Auto-failover (tries rsync -> scp -> base64)
result = robust_push("file.npz", "host", 22, "/path/file.npz", TransferConfig())
```

**Root Cause:** Firewall/proxy interference with binary streams on some providers.

---

### Lambda Labs Nodes (Dec 2025)

**History:**

- December 2025: Account terminated, all nodes offline
- December 28, 2025: Account restored, new GH200 instances being provisioned

**Current Status:** ACTIVE - 6 GH200 nodes being set up

**Nodes Configured:**

- lambda-gh200-1 through lambda-gh200-5 (96GB each)
- lambda-gh200-training (96GB)

**Setup Required:**

- Tailscale registration pending
- RingRift repository needs deployment
- See `config/distributed_hosts.yaml` for configuration

---

## Event System Issues

### P2P Events Not Reaching Coordinators (FIXED Dec 27, 2025)

**Issue:** P2P lifecycle events (HOST_OFFLINE, HOST_ONLINE, LEADER_ELECTED) were published to DataEventBus but never bridged to UnifiedEventRouter.

**Root Cause:** Missing DataEventBus -> Router bridge in `_setup_bus_bridges()`.

**Status:** FIXED (Dec 28, 2025)

**Fixes Applied:**

1. Added DataEventBus -> Router bridge in `event_router.py` (Dec 27)
2. Added subscriptions in `coordination_bootstrap.py:_wire_missing_event_subscriptions()` (Dec 28):
   - HOST_OFFLINE → UnifiedHealthManager.handle_node_offline()
   - HOST_ONLINE → UnifiedHealthManager.handle_node_online()
   - LEADER_ELECTED → LeadershipCoordinator.on_leader_change()

---

## Training Issues

### Elo Plateau at 1600-1700 (FIXED Dec 27, 2025)

**Issue:** AI models plateau at ~1600-1700 Elo, never reaching the 2000+ target.

**Root Causes (identified via exploration agents):**

1. **Tournament sample size too small:** Using only 10 games for promotion decisions (statistically meaningless)
2. **Selfplay MCTS budget too low:** Using THROUGHPUT (64 sims) instead of QUALITY (800)
3. **No master-level budget tier:** Maximum was 1600 simulations, insufficient for 2000+ Elo

**Fixes Applied (Dec 27, 2025):**

1. `train_loop.py`: Tournament games increased from 10 → 50
2. `mixed_opponent_selfplay.py`: MCTS budget changed from THROUGHPUT (64) → QUALITY (800)
3. `thresholds.py`: Added GUMBEL_BUDGET_MASTER = 3200 for 2000+ Elo training

**Status:** FIXED - Changes applied, requires regenerating training data and retraining

**Budget Tier Reference:**

| Tier       | Simulations | Use Case                          |
| ---------- | ----------- | --------------------------------- |
| THROUGHPUT | 64          | Fast bootstrap only               |
| STANDARD   | 800         | Normal training (1500-1800 Elo)   |
| QUALITY    | 800         | Evaluation/gauntlet               |
| ULTIMATE   | 1600        | Strong benchmarks (1800-2000 Elo) |
| MASTER     | 3200        | 2000+ Elo training                |

---

### Quality Score Blocks Training (FIXED Dec 28, 2025)

**Issue:** Master loop automation never triggered training because quality scores initialized to 0.0, which blocked the training readiness check (requires >= 0.5).

**Root Cause:** `ConfigState.last_quality_score` defaulted to `0.0`. The `needs_training` property required quality >= 0.5 to return True. For new configs with no prior training data, no quality events had been generated yet - creating a circular dependency where training couldn't start without quality data, but quality data required prior training.

**Symptoms:**

- `master_loop.py` running 51+ hours without triggering training
- `_check_training_readiness()` returning "Low quality: 0.00" even with sufficient games
- TRAINING_THRESHOLD_REACHED events emitted but training never started

**Fix Applied (Dec 28, 2025):**

- Changed `last_quality_score` default from `0.0` → `0.7` in both:
  - `ConfigState` dataclass (line 165)
  - SQLite schema default (line 318)
- 0.7 is above the 0.5 threshold, allowing initial training to proceed
- Quality score is overwritten once actual QUALITY_SCORE_UPDATED events arrive

**Status:** FIXED

---

### Gauntlet Semaphore Leak (Non-Critical)

**Issue:** Gauntlet evaluation shows `resource_tracker: 5 leaked semaphore objects` warning on macOS with Python 3.10 spawn multiprocessing.

**Symptoms:**

- Warning messages in test output
- No functional impact (games complete correctly)

**Status:** NON-BLOCKING - Warning only, games complete correctly

**Root Cause:** ThreadPoolExecutor + spawn multiprocessing interaction causes semaphore leaks.

---

## Model Issues

### Models Require Symlinks for Selfplay

**Issue:** GPU selfplay expects models at `models/ringrift_best_{board}_{n}p.pth` but training outputs to `models/canonical_{board}_{n}p.pth`.

**Workaround:**

```bash
# Create symlinks after training
cd models
ln -sf canonical_hex8_4p.pth ringrift_best_hex8_4p.pth
```

**Status:** DOCUMENTED - By design for version tracking

---

## Network Issues

### Tailscale Connectivity Intermittent

**Issue:** Tailscale mesh connectivity can be intermittent between some node pairs.

**Workaround:** P2P orchestrator automatically falls back to public IPs when Tailscale unavailable.

**Status:** MONITORED - Failover working correctly

---

## See Also

- `CLAUDE.local.md` - Additional operational context
- `TRAINING_DATA_REGISTRY.md` - Data quality tracking
- `docs/DAEMON_REGISTRY.md` - Daemon configuration reference
