# Parity Gate Resolution Guide

**Last Updated:** 2025-12-27
**Status:** Critical - All 12 Canonical DBs Affected

---

## Overview

The parity gate validates that Python-generated games replay identically in TypeScript. This is essential because **TypeScript is the source of truth** for game rules.

**Current Status:** All 12 canonical databases are stuck in `pending_gate` status because cluster nodes lack `npx` (Node.js runtime).

---

## Understanding the Problem

### What is the Parity Gate?

The parity gate replays Python-generated selfplay games through the TypeScript game engine to verify:

1. Same moves produce same game states
2. Victory conditions match
3. Territory calculations are identical

```
Python Selfplay → GameReplayDB → TypeScript Replay → Validate Match
```

### Why It's Failing

**Root Cause:** Cluster nodes (Vast.ai, RunPod, Nebius containers) don't have Node.js installed.

```bash
# On cluster node:
$ which npx
# No output - npx not found

$ node --version
# bash: node: command not found
```

The parity script requires:

```bash
npx ts-node -T scripts/selfplay-db-ts-replay.ts --db <database>
```

### Affected Databases

All 12 canonical configurations:

| Database                    | Status       | Games   |
| --------------------------- | ------------ | ------- |
| `canonical_hex8_2p.db`      | pending_gate | ~50,000 |
| `canonical_hex8_3p.db`      | pending_gate | ~10,000 |
| `canonical_hex8_4p.db`      | pending_gate | ~3,000  |
| `canonical_square8_2p.db`   | pending_gate | ~80,000 |
| `canonical_square8_3p.db`   | pending_gate | ~15,000 |
| `canonical_square8_4p.db`   | pending_gate | ~20,000 |
| `canonical_square19_2p.db`  | pending_gate | ~5,000  |
| `canonical_square19_3p.db`  | pending_gate | ~2,000  |
| `canonical_square19_4p.db`  | pending_gate | ~1,500  |
| `canonical_hexagonal_2p.db` | pending_gate | ~3,000  |
| `canonical_hexagonal_3p.db` | pending_gate | ~1,500  |
| `canonical_hexagonal_4p.db` | pending_gate | ~1,000  |

---

## Resolution Options

### Option 1: Run Parity Gate Locally (Recommended)

Since the local development machine has Node.js, run validation there:

```bash
# On local machine with Node.js
cd ~/Development/RingRift/ai-service

# Sync database from cluster
rsync -avz ubuntu@nebius-backbone-1:~/ringrift/ai-service/data/games/canonical_hex8_2p.db \
  data/games/canonical_hex8_2p.db

# Run parity validation
python scripts/check_ts_python_replay_parity.py \
  --db data/games/canonical_hex8_2p.db \
  --max-games 1000

# If parity fails, emit a state bundle for debugging
python scripts/check_ts_python_replay_parity.py \
  --db data/games/canonical_hex8_2p.db \
  --emit-state-bundles-dir parity_bundles

# Diff a bundle to locate the first divergence
python scripts/diff_state_bundle.py \
  --bundle parity_bundles/<bundle>.state_bundle.json

# Or run the canonical gate script
python scripts/run_canonical_selfplay_parity_gate.py --board-type hex8 --num-players 2
```

**After validation passes:**

```bash
# Update database status
sqlite3 data/games/canonical_hex8_2p.db "UPDATE metadata SET value='passed' WHERE key='parity_gate_status'"

# Sync back to cluster
rsync -avz data/games/canonical_hex8_2p.db \
  ubuntu@nebius-backbone-1:~/ringrift/ai-service/data/games/
```

### Option 2: Install Node.js on Coordinator

Install Node.js on one coordinator node (nebius-backbone-1):

```bash
ssh ubuntu@nebius-backbone-1

# Install Node.js via nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install 20
nvm use 20

# Install TypeScript dependencies
cd ~/ringrift
npm install

# Verify
npx ts-node --version
```

Then run parity on the coordinator:

```bash
cd ~/ringrift/ai-service
python scripts/run_canonical_selfplay_parity_gate.py --board-type hex8 --num-players 2
```

### Option 3: Allow Pending Gate for Training (Workaround)

If you need to train urgently while parity is pending:

```bash
# Set environment variable to allow pending_gate databases
export RINGRIFT_ALLOW_PENDING_GATE=1

# Export training data
python scripts/export_replay_dataset.py \
  --use-discovery --board-type hex8 --num-players 2 \
  --output data/training/hex8_2p.npz

# Train
python -m app.training.train \
  --board-type hex8 --num-players 2 \
  --data-path data/training/hex8_2p.npz
```

**Warning:** Training on unvalidated data risks learning from buggy games.

---

## Step-by-Step Resolution

### Step 1: Identify Unvalidated Databases

```bash
# List all pending_gate databases
for db in data/games/canonical_*.db; do
  status=$(sqlite3 "$db" "SELECT value FROM metadata WHERE key='parity_gate_status'" 2>/dev/null)
  echo "$db: ${status:-not_set}"
done
```

### Step 2: Run Validation (Local)

```bash
# Validate one database
python scripts/check_ts_python_replay_parity.py \
  --db data/games/canonical_hex8_2p.db \
  --max-games 1000 \
  --verbose

# If all games pass:
# Output: "Parity check passed: 1000/1000 games replayed correctly"
```

### Step 3: Update Database Status

On validation pass:

```bash
sqlite3 data/games/canonical_hex8_2p.db <<EOF
INSERT OR REPLACE INTO metadata (key, value) VALUES ('parity_gate_status', 'passed');
INSERT OR REPLACE INTO metadata (key, value) VALUES ('parity_gate_timestamp', datetime('now'));
INSERT OR REPLACE INTO metadata (key, value) VALUES ('parity_gate_games_checked', '1000');
EOF
```

On validation fail:

```bash
sqlite3 data/games/canonical_hex8_2p.db <<EOF
INSERT OR REPLACE INTO metadata (key, value) VALUES ('parity_gate_status', 'failed');
INSERT OR REPLACE INTO metadata (key, value) VALUES ('parity_gate_error', 'Move 42 mismatch: expected PLACE at (3,5), got PLACE at (3,4)');
EOF
```

### Step 4: Sync Updated Database to Cluster

```bash
# Push to all GPU nodes
for host in nebius-backbone-1 runpod-h100 vast-12345; do
  rsync -avz data/games/canonical_hex8_2p.db \
    root@$host:~/ringrift/ai-service/data/games/
done
```

---

## Troubleshooting Parity Failures

### Common Failure Patterns

#### 1. Move Position Mismatch

```
Parity Error: Move 15 mismatch
  Python: PLACE_RING to=(3, 5)
  TypeScript: PLACE_RING to=(3, 4)
```

**Cause:** Coordinate system difference (Python 0-indexed vs TypeScript 1-indexed?)

**Fix:** Check `app/rules/coordinate_transforms.py`

#### 2. Phase Mismatch

```
Parity Error: Move 22 phase mismatch
  Python: ATTACK
  TypeScript: PLACE_RING
```

**Cause:** Different phase transition logic

**Fix:** Compare `app/rules/phase_transitions.py` with `src/shared/engine/phases.ts`

#### 3. Territory Calculation Mismatch

```
Parity Error: Final score mismatch
  Python: P1=24, P2=18
  TypeScript: P1=23, P2=19
```

**Cause:** Territory calculation difference

**Fix:** Compare `app/rules/territory.py` with `src/shared/engine/territory.ts`

#### 4. Chain Capture FSM Mismatch

```
Parity Error: Move 45 capture failure
  Python: captured_positions=[...]
  TypeScript: captured_positions=[]
```

**Cause:** Chain capture logic difference

**Fix:** Compare `app/rules/capture.py` with `src/shared/engine/capture.ts`

### Debug Workflow

1. **Identify failing game:**

```bash
python scripts/check_ts_python_replay_parity.py \
  --db data/games/canonical_hex8_2p.db \
  --max-games 100 --verbose 2>&1 | grep "FAILED"
```

2. **Dump state at failure:**

```bash
RINGRIFT_TS_REPLAY_DUMP_STATE_AT_K=42 \
  npx ts-node -T scripts/selfplay-db-ts-replay.ts \
    --db data/games/canonical_hex8_2p.db \
    --game <game_id>
```

3. **Compare Python state:**

```python
# In Python
from app.db.game_replay_db import GameReplayDB
db = GameReplayDB("data/games/canonical_hex8_2p.db")
game = db.get_game("<game_id>")
state = replay_to_move(game, move_index=42)
print(state)
```

---

## Automation

### Scheduled Parity Validation

Add to coordinator crontab:

```bash
# Run parity validation daily at 2 AM
0 2 * * * cd ~/ringrift/ai-service && python scripts/run_canonical_selfplay_parity_gate.py --all >> logs/parity_gate.log 2>&1
```

### CI/CD Integration

In `.github/workflows/parity.yml`:

```yaml
parity-check:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-node@v4
      with:
        node-version: '20'
    - run: npm install
    - run: |
        python scripts/check_ts_python_replay_parity.py \
          --db data/games/test_sample.db \
          --max-games 100
```

---

## Prevention

### Ensure Parity Before Production

1. **Local validation first:** Always run parity locally before syncing to cluster
2. **Sample validation:** Run on 100-1000 games, not full database
3. **Continuous monitoring:** Add parity checks to selfplay pipeline

### Recommended Workflow

```bash
# 1. Generate games locally (small batch)
python scripts/selfplay.py --board hex8 --num-players 2 --num-games 100 \
  --output-dir data/games/test_hex8_2p

# 2. Run parity check
python scripts/check_ts_python_replay_parity.py \
  --db data/games/test_hex8_2p/games.db

# 3. If pass, run full selfplay on cluster
ssh nebius-backbone-1 "cd ~/ringrift/ai-service && python scripts/selfplay.py ..."
```

---

## See Also

- [PARITY_MISMATCH_DEBUG.md](./PARITY_MISMATCH_DEBUG.md) - Deep debugging guide
- [TRAINING_DATA_REGISTRY.md](../training/TRAINING_DATA_REGISTRY.md) - Data quality tracking
- [DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md) - Development setup
