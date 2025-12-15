# Vast.ai Canonical Selfplay Generation Runbook

This runbook provides step-by-step instructions for generating canonical selfplay databases on Vast.ai instances.

## Overview

Canonical selfplay generates game databases that pass:

1. **TS↔Python parity checks** - Ensures Python and TypeScript engines produce identical results
2. **Canonical history validation** - Ensures all phase transitions are properly recorded
3. **FE/territory fixture tests** - Validates forced elimination and territory edge cases

**Critical Requirements:**

- **Node.js 18+** is required for TS↔Python parity checks (uses `npx ts-node`)
- **Python 3.11+** with the ai-service dependencies
- **npm dependencies** installed at repo root level

## Vast.ai Instance Selection

When selecting instances on Vast.ai, look for:

| Criteria | Recommended      | Notes                                            |
| -------- | ---------------- | ------------------------------------------------ |
| vCPUs    | 16-32            | Higher CPU count speeds up parity checks         |
| RAM      | 64GB+            | Needed for TS↔Python parity validation           |
| GPU      | Any CUDA-capable | Not strictly required but speeds up AI inference |
| Disk     | 50GB+            | For dependencies & generated databases           |
| $/hr     | $0.05-0.15       | Best value for CPU-bound selfplay                |

**Tip:** RTX 3060/3070/3080 class instances often have good CPU configs at low cost.

Use `vastai search offers` or the web interface to find available instances (IDs are ephemeral).

## Quick Start

### Option A: Use the Automated Script

```bash
# From local machine (RingRift project root)
cd ai-service/scripts

# Run on existing instance (replace YOUR_INSTANCE_ID with your Vast.ai instance)
./vast_canonical_selfplay.sh \
  --instance-id YOUR_INSTANCE_ID \
  --num-games 200 \
  --board-type square8

# Or create a new instance first
./vast_canonical_selfplay.sh --create-instance
```

### Option B: Manual Setup

#### Step 1: Connect to Instance

```bash
# Connect to your instance (port and host from Vast.ai dashboard)
ssh -p YOUR_PORT root@sshX.vast.ai
```

#### Step 2: Install Node.js (if not present)

```bash
# Check Node version
node --version

# If missing or < v18, install Node.js 20.x
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs

# Verify
node --version  # Should show v20.x.x
npm --version
```

#### Step 3: Clone/Update RingRift Repository

```bash
# Clone if first time
git clone https://github.com/SynaptentLLC/RingRift.git ~/ringrift

# Or update if exists
cd ~/ringrift && git fetch --all && git reset --hard origin/main

cd ~/ringrift
```

#### Step 4: Install npm Dependencies

```bash
# At repo root
npm install

# Build TypeScript (optional but recommended)
npm run build
```

#### Step 5: Set Up Python Environment

```bash
cd ~/ringrift/ai-service

# Create venv if not exists
python3 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install --upgrade pip wheel
pip install -r requirements.txt
```

#### Step 6: Run Canonical Selfplay

```bash
# Ensure we're in ai-service with venv active
cd ~/ringrift/ai-service
source venv/bin/activate

# Set environment variables
export PYTHONPATH="$HOME/ringrift/ai-service"
export RINGRIFT_STRICT_NO_MOVE_INVARIANT=1
export RINGRIFT_PARITY_VALIDATION=strict
export RINGRIFT_FORCE_BOOKKEEPING_MOVES=1

# Create output directories
mkdir -p data/games logs/selfplay

# Run canonical selfplay for square8 2-player (200 games)
python scripts/generate_canonical_selfplay.py \
  --board-type square8 \
  --num-games 200 \
  --num-players 2 \
  --difficulty-band light \
  --db data/games/canonical_square8_2p.db \
  --summary data/games/canonical_square8_2p.summary.json \
  2>&1 | tee logs/selfplay/canonical_square8_2p.log
```

## Target Game Counts

Per AI Training Pipeline requirements:

| Board Type | Players | Target Games | Status       |
| ---------- | ------- | ------------ | ------------ |
| square8    | 2P      | 200+         | **PRIORITY** |
| square19   | 2P      | 50+          | Secondary    |
| hexagonal  | 2P      | 50+          | Secondary    |

## Expected Output

After successful completion:

```
ai-service/data/games/
├── canonical_square8_2p.db              # Game replay database
├── canonical_square8_2p.summary.json    # Gate summary (canonical_ok: true/false)
└── canonical_square8_2p.db.parity_gate.json  # Detailed parity report
```

### Verifying Results

```bash
# Check summary
cat data/games/canonical_square8_2p.summary.json | python3 -m json.tool

# Should show:
# {
#   "canonical_ok": true,
#   "db_stats": { "games_total": 200, ... },
#   "parity_gate": { "passed_canonical_parity_gate": true, ... }
# }
```

## Retrieving Results

### From Vast.ai to Local

```bash
# On local machine (replace YOUR_PORT and sshX with your instance's SSH details)
scp -P YOUR_PORT root@sshX.vast.ai:~/ringrift/ai-service/data/games/canonical_square8_2p.db \
  ./ai-service/data/games/

scp -P YOUR_PORT root@sshX.vast.ai:~/ringrift/ai-service/data/games/canonical_square8_2p.summary.json \
  ./ai-service/data/games/
```

## Troubleshooting

### "node: command not found"

The TS↔Python parity checks require Node.js:

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
```

### "npx ts-node" Errors

Ensure npm dependencies are installed at repo root:

```bash
cd ~/ringrift && npm install
```

### "canonical_ok: false" in Summary

Check `sample_issues` in the summary JSON for details. Common causes:

- Missing `--difficulty-band light` (heavy MCTS/Descent can cause issues)
- Outdated repo version (run `git pull`)
- Python dependency version mismatch

### SSH Connection Refused / Timeout

1. Check instance status: `vastai show instances`
2. Ensure instance is "running" not "loading" or "starting"
3. Check your SSH key is attached: `vastai show user`

## Parallel Execution

To generate games faster, use multiple instances in parallel:

```bash
# Square8 2P on first instance
ssh -p PORT1 root@sshX.vast.ai "cd ~/ringrift/ai-service && ./run_canonical.sh square8 100"

# Square19 2P on second instance (more powerful)
ssh -p PORT2 root@sshY.vast.ai "cd ~/ringrift/ai-service && ./run_canonical.sh square19 50"
```

## Cost Estimation

Estimates based on ~$0.06/hr instances (RTX 3070 class):

| Board       | Games | Est. Time | Est. Cost |
| ----------- | ----- | --------- | --------- |
| square8 2P  | 200   | ~2 hours  | ~$0.12    |
| square19 2P | 50    | ~4 hours  | ~$0.25    |
| hex 2P      | 50    | ~4 hours  | ~$0.25    |

## Related Scripts

- [`ai-service/scripts/vast_canonical_selfplay.sh`](../../ai-service/scripts/vast_canonical_selfplay.sh) - Automated setup script
- [`ai-service/scripts/generate_canonical_selfplay.py`](../../ai-service/scripts/generate_canonical_selfplay.py) - Core canonical selfplay generator
- [`ai-service/scripts/run_canonical_selfplay_parity_gate.py`](../../ai-service/scripts/run_canonical_selfplay_parity_gate.py) - Parity gate driver
- [`ai-service/scripts/check_ts_python_replay_parity.py`](../../ai-service/scripts/check_ts_python_replay_parity.py) - TS↔Python parity checker

## See Also

- [`TRAINING_DATA_REGISTRY.md`](../../ai-service/TRAINING_DATA_REGISTRY.md) - Canonical vs legacy DB inventory
- [`AI_TRAINING_AND_DATASETS.md`](../ai/AI_TRAINING_AND_DATASETS.md) - Training data requirements
