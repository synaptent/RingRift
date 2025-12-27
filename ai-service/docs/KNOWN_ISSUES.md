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

### Lambda Labs Nodes Terminated (Dec 2025)

**Issue:** Lambda Labs account terminated December 2025. All Lambda nodes permanently offline.

**Impact:**

- GH200 nodes (a, b-new, d, g, h, i, o, p, q, r, s, t) unavailable
- H100 nodes (lambda-h100, lambda-2xh100) unavailable

**Status:** PERMANENT - Use alternative providers (Vast.ai, RunPod, Nebius, Vultr)

---

## Event System Issues

### P2P Events Not Reaching Coordinators (FIXED Dec 27, 2025)

**Issue:** P2P lifecycle events (HOST_OFFLINE, HOST_ONLINE, LEADER_ELECTED) were published to DataEventBus but never bridged to UnifiedEventRouter.

**Root Cause:** Missing DataEventBus -> Router bridge in `_setup_bus_bridges()`.

**Status:** FIXED in commit pending (Dec 27, 2025)

**Fix Applied:** Added `_on_data_bus_event()` handler and DataEventBus subscription loop in `event_router.py`.

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
