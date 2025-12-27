# Training Data Volume Remediation Runbook

This runbook provides step-by-step instructions for executing training data generation tasks on a cluster to address the 94% gap in training data volume.

## Overview

### Current State

| Database                  | Current Games | Target | Gap   |
| ------------------------- | ------------- | ------ | ----- |
| `canonical_square8_2p.db` | 200           | 1,000  | 80%   |
| `canonical_square8_3p.db` | 2             | 500    | 99.6% |
| `canonical_square8_4p.db` | 2             | 500    | 99.6% |
| `canonical_square19.db`   | 26            | 1,000  | 97.4% |
| `canonical_hexagonal.db`  | 14            | 1,000  | 98.6% |

**Total Current:** ~244 games  
**Total Target:** 4,000+ games  
**Overall Gap:** ~94%

### Why This Runbook Exists

- Only square8 2P has sufficient data (200+ games)
- Hexagonal boards were blocked by HEX-PARITY-02 (now fixed in [`global_actions.py:174-220`](../../ai-service/app/rules/global_actions.py))
- Neural network training requires large canonical datasets to be effective

---

## Prerequisites

### 1. Cluster Access Requirements

- SSH access to target cluster nodes (Runpod/Vast/etc.)
- Sufficient disk space (>50GB recommended)
- Python 3.10+ installed (Docker/CI uses 3.11)
- Node.js 18+ installed (for TS↔Python parity checks)

### 2. Python Environment Setup

```bash
# SSH to cluster node
ssh user@cluster-node

# Navigate to RingRift directory
cd ~/ringrift

# Set up Python virtual environment (if not already done)
cd ai-service
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```

### 3. Sync Local Code Changes to Cluster

**Critical:** Ensure the HEX-PARITY-02 fix is present on the cluster before running hexagonal tasks.

```bash
# From local machine
cd /path/to/RingRift

# Push changes to remote repository
git add -A && git commit -m "Sync for cluster training" && git push origin main

# On cluster node - pull latest changes
ssh user@cluster-node "cd ~/ringrift && git fetch --all && git reset --hard origin/main"
```

### 4. Environment Variables

```bash
# Required for all tasks
export PYTHONPATH="$HOME/ringrift/ai-service"
export RINGRIFT_SKIP_RESOURCE_GUARD=1
```

---

## Task 1: Verify HEX-PARITY-02 Fix (P0)

**Priority:** P0 (Must complete before Task 6)  
**Dependency:** None

Verify that the hexagonal parity fix in [`global_actions.py`](../../ai-service/app/rules/global_actions.py) is working correctly on the cluster.

### Command

```bash
cd ~/ringrift/ai-service
source venv/bin/activate
PYTHONPATH=. python scripts/check_ts_python_replay_parity.py \
  --db data/games/canonical_hexagonal.db \
  --emit-state-bundles-dir parity_bundles/ \
  --compact
```

### Expected Output

```
Parity Summary:
  games_checked: X
  games_with_semantic_divergence: 0
  semantic_divergence_rate: 0.00%
```

### Success Criteria

- [ ] `games_with_semantic_divergence: 0`
- [ ] No new state bundles in `parity_bundles/` indicating divergence
- [ ] Script completes without errors

### Troubleshooting

If divergences are found:

1. Check that the latest code with HEX-PARITY-02 fix is deployed
2. Review state bundles using:
   ```bash
   PYTHONPATH=. python scripts/diff_state_bundle.py --bundle parity_bundles/<bundle_file>.json
   ```

---

## Task 2: Scale Square8 2P to 1,000 Games (P1)

**Priority:** P1  
**Dependency:** None (can run in parallel with Tasks 3-4)

### Command

```bash
cd ~/ringrift/ai-service
source venv/bin/activate
nohup bash -c 'PYTHONPATH=. RINGRIFT_SKIP_RESOURCE_GUARD=1 \
  python scripts/generate_canonical_selfplay.py \
    --board square8 \
    --num-players 2 \
    --num-games 200 \
    --min-recorded-games 1000 \
    --max-soak-attempts 10 \
    --difficulty-band medium \
    --db data/games/canonical_square8_2p.db \
    --summary data/games/db_health.canonical_square8_2p.json' \
  > logs/task2_square8_2p.log 2>&1 &
```

### Estimated Time

- ~4-5 hours (single host)

### Success Criteria

- [ ] `games_total >= 1000` in `db_health.canonical_square8_2p.json`
- [ ] `canonical_ok: true` in summary JSON
- [ ] `games_with_semantic_divergence: 0`

### Verification

```bash
# Check game count
sqlite3 data/games/canonical_square8_2p.db "SELECT COUNT(*) FROM games"

# Check gate status
jq '{canonical_ok, games_total: .db_stats.games_total}' data/games/db_health.canonical_square8_2p.json
```

---

## Task 3: Generate Square8 3P Data (P1)

**Priority:** P1  
**Dependency:** None (can run in parallel with Tasks 2, 4)

### Command

```bash
cd ~/ringrift/ai-service
source venv/bin/activate
nohup bash -c 'PYTHONPATH=. RINGRIFT_SKIP_RESOURCE_GUARD=1 \
  python scripts/generate_canonical_selfplay.py \
    --board square8 \
    --num-players 3 \
    --num-games 100 \
    --min-recorded-games 500 \
    --max-soak-attempts 6 \
    --difficulty-band light \
    --db data/games/canonical_square8_3p.db \
    --summary data/games/db_health.canonical_square8_3p.json' \
  > logs/task3_square8_3p.log 2>&1 &
```

### Estimated Time

- ~6 hours (single host)

### Success Criteria

- [ ] `games_total >= 500` in summary JSON
- [ ] `canonical_ok: true`

### Verification

```bash
sqlite3 data/games/canonical_square8_3p.db "SELECT COUNT(*) FROM games"
jq '{canonical_ok, games_total: .db_stats.games_total}' data/games/db_health.canonical_square8_3p.json
```

---

## Task 4: Generate Square8 4P Data (P1)

**Priority:** P1  
**Dependency:** None (can run in parallel with Tasks 2-3)

### Command

```bash
cd ~/ringrift/ai-service
source venv/bin/activate
nohup bash -c 'PYTHONPATH=. RINGRIFT_SKIP_RESOURCE_GUARD=1 \
  python scripts/generate_canonical_selfplay.py \
    --board square8 \
    --num-players 4 \
    --num-games 100 \
    --min-recorded-games 500 \
    --max-soak-attempts 6 \
    --difficulty-band light \
    --db data/games/canonical_square8_4p.db \
    --summary data/games/db_health.canonical_square8_4p.json' \
  > logs/task4_square8_4p.log 2>&1 &
```

### Estimated Time

- ~8 hours (single host)

### Success Criteria

- [ ] `games_total >= 500` in summary JSON
- [ ] `canonical_ok: true`

### Verification

```bash
sqlite3 data/games/canonical_square8_4p.db "SELECT COUNT(*) FROM games"
jq '{canonical_ok, games_total: .db_stats.games_total}' data/games/db_health.canonical_square8_4p.json
```

---

## Task 5: Scale Square19 2P Data (P2)

**Priority:** P2  
**Dependency:** None (can start anytime, but prioritize after square8 tasks)

### Command

```bash
cd ~/ringrift/ai-service
source venv/bin/activate
nohup bash -c 'PYTHONPATH=. RINGRIFT_SKIP_RESOURCE_GUARD=1 \
  python scripts/generate_canonical_selfplay.py \
    --board square19 \
    --num-players 2 \
    --num-games 100 \
    --min-recorded-games 1000 \
    --max-soak-attempts 15 \
    --difficulty-band light \
    --db data/games/canonical_square19.db \
    --summary data/games/db_health.canonical_square19.json' \
  > logs/task5_square19.log 2>&1 &
```

### Estimated Time

- ~15 hours (single host)
- ~5 hours with distributed nodes

### Success Criteria

- [ ] `games_total >= 1000` in summary JSON
- [ ] `canonical_ok: true`
- [ ] `games_with_semantic_divergence: 0`

### Verification

```bash
sqlite3 data/games/canonical_square19.db "SELECT COUNT(*) FROM games"
jq '{canonical_ok, games_total: .db_stats.games_total}' data/games/db_health.canonical_square19.json
```

---

## Task 6: Regenerate Hexagonal Data (P2)

**Priority:** P2  
**Dependency:** Task 1 must pass first (HEX-PARITY-02 fix verification)

This task regenerates the hexagonal database after the HEX-PARITY-02 fix.

### Command

```bash
cd ~/ringrift/ai-service
source venv/bin/activate
nohup bash -c 'PYTHONPATH=. RINGRIFT_SKIP_RESOURCE_GUARD=1 \
  python scripts/run_canonical_selfplay_parity_gate.py \
    --board-type hexagonal \
    --num-games 300 \
    --output-db data/games/canonical_hexagonal.db' \
  > logs/task6_hexagonal.log 2>&1 &
```

### Estimated Time

- ~35 hours (single host) for 1,000 games
- ~10 hours for initial 300 games

### Success Criteria

- [ ] `games_total >= 1000` (after multiple runs)
- [ ] `canonical_ok: true`
- [ ] `games_with_semantic_divergence: 0`
- [ ] No HEX-PARITY-02 related failures

### Verification

```bash
sqlite3 data/games/canonical_hexagonal.db "SELECT COUNT(*) FROM games"
jq '{canonical_ok, games_total: .db_stats.games_total}' data/games/db_health.canonical_hexagonal.json
```

### Scale-Up to 1,000 Games

After initial generation, use iterative runs:

```bash
nohup bash -c 'PYTHONPATH=. RINGRIFT_SKIP_RESOURCE_GUARD=1 \
  python scripts/generate_canonical_selfplay.py \
    --board hexagonal \
    --num-players 2 \
    --num-games 50 \
    --min-recorded-games 1000 \
    --max-soak-attempts 25 \
    --difficulty-band light \
    --db data/games/canonical_hexagonal.db \
    --summary data/games/db_health.canonical_hexagonal.json' \
  > logs/task6_hexagonal_scaleup.log 2>&1 &
```

---

## Task 7: Generate Hex8 Data (P2)

**Priority:** P2  
**Dependency:** Task 1 (HEX-PARITY-02 fix verified)

### Command

```bash
cd ~/ringrift/ai-service
source venv/bin/activate
nohup bash -c 'PYTHONPATH=. RINGRIFT_SKIP_RESOURCE_GUARD=1 \
  python scripts/generate_canonical_selfplay.py \
    --board hex8 \
    --num-players 2 \
    --num-games 50 \
    --min-recorded-games 200 \
    --max-soak-attempts 5 \
    --difficulty-band light \
    --db data/games/canonical_hex8_2p.db \
    --summary data/games/db_health.canonical_hex8_2p.json' \
  > logs/task7_hex8.log 2>&1 &
```

### Estimated Time

- ~4-6 hours (single host)

### Success Criteria

- [ ] `games_total >= 200` in summary JSON
- [ ] `canonical_ok: true`

### Verification

```bash
sqlite3 data/games/canonical_hex8_2p.db "SELECT COUNT(*) FROM games"
jq '{canonical_ok, games_total: .db_stats.games_total}' data/games/db_health.canonical_hex8_2p.json
```

---

## Task 8: Train Neural Network Models (P3)

**Priority:** P3  
**Dependency:** Tasks 2-7 completed, GPU cluster access required

### Prerequisites

- Canonical databases are populated and pass gate
- GPU-capable cluster node (CUDA)

### Commands (One Per Board Type)

#### Square8 Training

```bash
cd ~/ringrift/ai-service
source venv/bin/activate
PYTHONPATH=. python scripts/run_canonical_training.py \
  --board-type square8 \
  --epochs 50 \
  2>&1 | tee logs/training_square8.log
```

#### Square19 Training

```bash
PYTHONPATH=. python scripts/run_canonical_training.py \
  --board-type square19 \
  --epochs 50 \
  2>&1 | tee logs/training_square19.log
```

#### Hexagonal Training

```bash
PYTHONPATH=. python scripts/run_canonical_training.py \
  --board-type hexagonal \
  --epochs 50 \
  2>&1 | tee logs/training_hexagonal.log
```

### Estimated Time

- ~2-4 hours per board type (with GPU)
- ~8-12 hours per board type (CPU only)

### Success Criteria

- [ ] Training loss decreases over epochs (target: < 0.5)
- [ ] Model checkpoints saved at intervals (epochs 10, 20, 30, 40, 50)
- [ ] Final model files created:
  - `checkpoints/ringrift_v2_square8.pth`
  - `checkpoints/ringrift_v2_square19.pth`
  - `checkpoints/ringrift_v2_hexagonal.pth`

### Verification

```bash
# Check for model files
ls -la checkpoints/ringrift_v2_*.pth

# Evaluate neural network performance
PYTHONPATH=. python scripts/evaluate_ai_models.py \
  --player1 neural \
  --player2 random \
  --games 50 \
  --board square8 \
  --output results/neural_vs_random_post_scaling.json
```

---

## Progress Tracking Table

Use this table to track execution progress across cluster nodes:

| Task | Status     | Games Generated | Time Taken | Node | Notes |
| ---- | ---------- | --------------- | ---------- | ---- | ----- |
| 1    | ⬜ Pending | N/A             |            |      |       |
| 2    | ⬜ Pending | 0/1000          |            |      |       |
| 3    | ⬜ Pending | 0/500           |            |      |       |
| 4    | ⬜ Pending | 0/500           |            |      |       |
| 5    | ⬜ Pending | 0/1000          |            |      |       |
| 6    | ⬜ Pending | 0/1000          |            |      |       |
| 7    | ⬜ Pending | 0/200           |            |      |       |
| 8    | ⬜ Pending | N/A             |            |      |       |

**Status Legend:**

- ⬜ Pending
- 🔄 In Progress
- ✅ Complete
- ❌ Failed

---

## Validation After All Tasks

### 1. Verify All Databases Pass Canonical Gate

```bash
cd ~/ringrift/ai-service
source venv/bin/activate

# Check all DBs
for db in canonical_square8_2p canonical_square8_3p canonical_square8_4p canonical_square19 canonical_hexagonal canonical_hex8_2p; do
  echo "=== $db ==="
  if [ -f "data/games/${db}.db" ]; then
    echo "Games: $(sqlite3 data/games/${db}.db 'SELECT COUNT(*) FROM games')"
    if [ -f "data/games/db_health.${db}.json" ]; then
      jq '{canonical_ok, games_total: .db_stats.games_total, divergences: .parity_gate.games_with_semantic_divergence}' \
        data/games/db_health.${db}.json
    else
      echo "No health summary found"
    fi
  else
    echo "Database not found"
  fi
done
```

### 2. Update TRAINING_DATA_REGISTRY.md

After all tasks complete successfully, update `ai-service/TRAINING_DATA_REGISTRY.md` (local-only, gitignored) with:

1. Updated game counts for each canonical database
2. Date of last gate summary
3. Any edge-case workarounds documented
4. Volume targets achieved

### 3. Final Summary

Run the full summary check:

```bash
# Generate summary report
echo "=== Training Data Volume Summary ===" > /tmp/volume_summary.txt
echo "" >> /tmp/volume_summary.txt
echo "Database,Games,Target,Pct" >> /tmp/volume_summary.txt

for db in canonical_square8_2p:1000 canonical_square8_3p:500 canonical_square8_4p:500 canonical_square19:1000 canonical_hexagonal:1000 canonical_hex8_2p:200; do
  name=$(echo $db | cut -d: -f1)
  target=$(echo $db | cut -d: -f2)
  count=$(sqlite3 data/games/${name}.db "SELECT COUNT(*) FROM games" 2>/dev/null || echo "0")
  pct=$(echo "scale=1; $count * 100 / $target" | bc)
  echo "$name,$count,$target,$pct%" >> /tmp/volume_summary.txt
done

cat /tmp/volume_summary.txt
```

---

## Related Documentation

- [`TRAINING_DATA_VOLUME_REMEDIATION_PLAN.md`](../planning/TRAINING_DATA_VOLUME_REMEDIATION_PLAN.md) - Detailed remediation plan
- [`VAST_CANONICAL_SELFPLAY.md`](VAST_CANONICAL_SELFPLAY.md) - Vast.ai-specific setup
- `ai-service/TRAINING_DATA_REGISTRY.md` (local-only, gitignored) - Database inventory and classification
- [`AI_TRAINING_AND_DATASETS.md`](../ai/AI_TRAINING_AND_DATASETS.md) - Training data requirements

## Appendix: Monitoring Long-Running Tasks

### Check Process Status

```bash
# List running generation processes
ps aux | grep generate_canonical_selfplay | grep -v grep

# Check specific log in real-time
tail -f logs/task2_square8_2p.log

# Check all task logs for progress
for log in logs/task*.log; do
  echo "=== $log ==="
  tail -3 "$log"
done
```

### Estimate Completion Time

```bash
# Get current game count and calculate remaining time
db="data/games/canonical_square8_2p.db"
target=1000
current=$(sqlite3 $db "SELECT COUNT(*) FROM games" 2>/dev/null || echo "0")
remaining=$((target - current))
echo "Current: $current / $target ($remaining remaining)"
```

### Resume After Failure

If a task fails mid-way:

1. Check the log file for errors
2. Re-run the same command - the script will resume from the existing database
3. The `--min-recorded-games` flag ensures it continues until target is met
