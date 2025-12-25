# RingRift AI Training Pipeline

This document describes the self-improvement training loop architecture and operational procedures.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         UNIFIED AI TRAINING LOOP                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │  GPU Nodes   │    │  Cloud VMs   │    │  Train Node  │    │  Backup    │ │
│  │  (Selfplay)  │    │  (Selfplay)  │    │  (Training)  │    │  (Backup)  │ │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └────────────┘ │
│         │                   │                   │                           │
│         └───────────────────┼───────────────────┘                           │
│                             │                                               │
│                     ┌───────▼───────┐                                       │
│                     │  JSONL Files  │                                       │
│                     │  (Raw Data)   │                                       │
│                     └───────┬───────┘                                       │
│                             │                                               │
│                     ┌───────▼───────┐                                       │
│                     │  SQLite DBs   │                                       │
│                     │ (Aggregated)  │                                       │
│                     └───────┬───────┘                                       │
│                             │                                               │
│                     ┌───────▼───────┐                                       │
│                     │   NPZ Files   │                                       │
│                     │  (Training)   │                                       │
│                     └───────┬───────┘                                       │
│                             │                                               │
│                     ┌───────▼───────┐                                       │
│                     │   GPU Train   │                                       │
│                     │  (GPU Node)   │                                       │
│                     └───────┬───────┘                                       │
│                             │                                               │
│                     ┌───────▼───────┐                                       │
│                     │  Elo Eval &   │                                       │
│                     │  Promotion    │                                       │
│                     └───────────────┘                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Selfplay Generation

**Hosts**: Configure in `config/cluster_hosts.yaml` (GPU nodes for selfplay, cloud VMs optional)

**Output**: JSONL files in `data/selfplay/` directories

**Process**:

- Workers generate games via `run_gpu_selfplay.py` or `run_hybrid_selfplay.py`
- Games are written to JSONL format for efficiency
- Each worker writes to its own subdirectory

### 2. Data Synchronization

**Cron Job**: Every 15 minutes

```bash
# Sync from vast hosts via P2P or unified sync
python scripts/unified_data_sync.py --source vast --dest local
```

**Script**: `unified_data_sync.py` (supersedes deprecated `sync_vast_jsonl.sh`)

### 3. JSONL to SQLite Aggregation

**Cron Job**: Every 30 minutes

```bash
python3 scripts/aggregate_jsonl_to_db.py \
    --input-dir data/selfplay \
    --output-db data/games/cluster_merged.db
```

**Output**: `data/games/cluster_merged.db` (or config-specific DBs like `hex8_2p.db`)

### 4. NPZ Export

**Triggered by**: Unified loop when training threshold reached

```bash
python3 scripts/export_replay_dataset.py \
    --db data/games/all_jsonl_training.db \
    --output data/training/unified_{config}.npz
```

### 5. Model Training

**Host**: Training GPU Node

**Script**: `scripts/unified_ai_loop.py`

**Config**: `config/unified_loop.yaml`

```yaml
training:
  trigger_threshold_games: 300
  min_interval_seconds: 1200
```

### 6. Evaluation & Promotion

**Requirement**: +20 Elo vs current best model

**Process**:

1. Shadow tournament against current best
2. If Elo gain >= threshold, promote to production
3. Distribute to all selfplay workers

## Key Configuration Files

| File                                        | Purpose                             |
| ------------------------------------------- | ----------------------------------- |
| `config/unified_loop.yaml`                  | Main loop configuration             |
| `config/distributed_hosts.yaml`             | Cluster host inventory (canonical)  |
| `config/remote_hosts.yaml`                  | Data sync host definitions (legacy) |
| `logs/unified_loop/unified_loop_state.json` | Loop state persistence              |

## Monitoring

### Training Monitor

```bash
python3 scripts/training_monitor.py --verbose
```

### Database Health Check

```bash
python3 scripts/db_health_check.py
```

### GPU Status

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

### Process Status

```bash
ps aux | grep -E "unified_ai_loop|selfplay|training"
```

## Troubleshooting

### Training Not Triggering

1. Check `training_in_progress` flag:

   ```bash
   cat logs/unified_loop/unified_loop_state.json | grep training_in_progress
   ```

2. Reset if stuck:
   ```python
   import json
   with open("logs/unified_loop/unified_loop_state.json", "r+") as f:
       state = json.load(f)
       state["training_in_progress"] = False
       f.seek(0)
       json.dump(state, f, indent=2)
       f.truncate()
   ```

### Corrupted Timestamps

Check for unrealistic "hours since promotion":

```bash
grep "since last promotion" logs/unified_*.log
```

Fix by resetting timestamps:

```python
import json, time
with open("logs/unified_loop/unified_loop_state.json", "r+") as f:
    state = json.load(f)
    now = time.time()
    for config in state["configs"].values():
        if config["last_promotion_time"] == 0:
            config["last_promotion_time"] = now
    f.seek(0)
    json.dump(state, f, indent=2)
    f.truncate()
```

### SSH Key Issues

Remote hosts require SSH key configuration on your training node:

```bash
# Add to ~/.ssh/config on training node
Host gpu-node-*
    User ubuntu
    IdentityFile ~/.ssh/id_cluster
    StrictHostKeyChecking no
```

### Database Corruption

Run health check:

```bash
python3 scripts/db_health_check.py --repair --quarantine
```

## Cron Jobs (Training Node)

```cron
# Sync JSONL from vast hosts
*/15 * * * * ~/sync_vast_jsonl.sh >> ~/ringrift/ai-service/logs/vast_sync.log 2>&1

# Aggregate JSONL to SQLite
*/30 * * * * cd ~/ringrift/ai-service && python3 scripts/aggregate_jsonl_to_db.py --input-dir data/selfplay --output-db data/games/all_jsonl_training.db >> logs/aggregate_jsonl.log 2>&1

# Training monitor (hourly)
0 * * * * cd ~/ringrift/ai-service && python3 scripts/training_monitor.py --log >> logs/training_monitor.log 2>&1
```

## Metrics

Prometheus metrics available at: `http://TRAINING_NODE_IP:9090/metrics`

Key metrics:

- `ringrift_training_runs_total`
- `ringrift_promotions_total`
- `ringrift_games_synced_total`
- `ringrift_gpu_utilization`

## Operational Procedures

### Starting the Training Loop

```bash
cd ~/ringrift/ai-service
source venv/bin/activate
python scripts/unified_ai_loop.py --start
```

### Stopping the Training Loop

```bash
python scripts/unified_ai_loop.py --stop
```

### Checking Status

```bash
python scripts/unified_ai_loop.py --status
```

### Emergency Halt

```bash
python scripts/unified_ai_loop.py --halt
```

### Resume After Halt

```bash
python scripts/unified_ai_loop.py --resume
```

## File Locations

| Path                         | Contents                  |
| ---------------------------- | ------------------------- |
| `data/selfplay/`             | Raw JSONL selfplay data   |
| `data/games/`                | SQLite training databases |
| `data/training/`             | NPZ training files        |
| `models/`                    | Trained model checkpoints |
| `models/ringrift_best_*.pth` | Production models         |
| `logs/unified_loop/`         | Loop logs and state       |

## NNUE Policy Training with MCTS Data

The pipeline supports training NNUE models with policy heads using MCTS visit distributions via KL divergence loss.

### Direct JSONL Training

Train directly from JSONL files without SQLite conversion:

```bash
python scripts/train_nnue_policy.py \
    --jsonl data/selfplay/mcts_square8_2p/games.jsonl \
    --auto-kl-loss \
    --epochs 50
```

### KL Divergence Loss

When MCTS data is available, KL loss trains the policy head to match search distributions:

```bash
# Auto-enable when sufficient MCTS coverage
python scripts/train_nnue_policy.py \
    --jsonl data/selfplay/mcts_*.jsonl \
    --auto-kl-loss \
    --kl-min-coverage 0.3 \
    --kl-min-samples 50

# Force KL loss
python scripts/train_nnue_policy.py \
    --jsonl data/selfplay/mcts_*.jsonl \
    --use-kl-loss
```

**Benefits of KL Loss:**

- Learns from full MCTS visit distribution, not just final move
- Better policy calibration for uncertain positions
- Improved move ranking across all candidates

### Automated Training Pipeline

The `scripts/auto_training_pipeline.py` provides a complete training workflow including data collection, value training, policy training, A/B testing, and model distribution:

```bash
# Full pipeline
python scripts/auto_training_pipeline.py \
    --board-type square8 \
    --num-players 2

# Dry run to see what would be done
python scripts/auto_training_pipeline.py \
    --dry-run --board-type hexagonal

# Skip steps selectively
python scripts/auto_training_pipeline.py \
    --skip-collect --skip-backfill \
    --board-type square8
```

**Pipeline Steps:**

| Step             | Flag to Skip      | Description                    |
| ---------------- | ----------------- | ------------------------------ |
| 1. Collect       | `--skip-collect`  | Gather data from cluster nodes |
| 2. Backfill      | `--skip-backfill` | Add missing snapshots          |
| 3. Train Value   | `--skip-train`    | Train NNUE value model         |
| 3b. Train Policy | `--skip-policy`   | Train NNUE policy model        |
| 3c. A/B Test     | `--skip-ab-test`  | Validate policy improvement    |
| 3d. Selfplay     | `--skip-selfplay` | Generate policy-guided games   |
| 4. Sync          | `--skip-sync`     | Distribute models to nodes     |

**Policy Training Options:**

```bash
--use-curriculum         # Use staged curriculum training (default)
--no-curriculum          # Use direct training
--selfplay-games 100     # Games for policy selfplay
```

### Multi-Config Training

Multi-board training is configured in `config/unified_loop.yaml` and run via the unified loop:

```yaml
# config/unified_loop.yaml (example)
training:
  boards: [square8, square19, hex8]
  players: [2, 3, 4]
```

```bash
python scripts/unified_ai_loop.py --start --config config/unified_loop.yaml
```

**Notes:**

- Balance mode and curriculum selection are handled inside `scripts/unified_ai_loop.py`.
- Policy training passthrough uses the unified loop config (`policy_training` section).

### Environment Variables

| Variable                          | Description                       | Default |
| --------------------------------- | --------------------------------- | ------- |
| `RINGRIFT_ENABLE_POLICY_TRAINING` | Enable NNUE policy training       | `1`     |
| `RINGRIFT_POLICY_AUTO_KL_LOSS`    | Auto-detect and enable KL loss    | `1`     |
| `RINGRIFT_POLICY_KL_MIN_COVERAGE` | Min MCTS coverage for auto-KL     | `0.3`   |
| `RINGRIFT_POLICY_KL_MIN_SAMPLES`  | Min samples for auto-KL           | `50`    |
| `RINGRIFT_ENABLE_AUTO_HP_TUNING`  | Enable hyperparameter auto-tuning | `0`     |

See [NNUE_POLICY_TRAINING.md](../algorithms/NNUE_POLICY_TRAINING.md) for complete policy training documentation.

---

## Advanced Training Features (2025-12)

The training pipeline includes several advanced features for improved model quality.

### Policy Label Smoothing

Prevents overconfident predictions by mixing targets with uniform distribution:

```bash
python -m app.training.train \
  --data-path data/training/dataset.npz \
  --board-type hex8 \
  --policy-label-smoothing 0.05
```

**Configuration** (`unified_loop.yaml`):

```yaml
training:
  label_smoothing_warmup: 5 # Epochs before full smoothing
```

### Hex Board Augmentation

D6 symmetry augmentation provides 12x effective dataset size for hex boards:

```bash
python -m app.training.train \
  --data-path data/training/hex8_games.npz \
  --board-type hex8 \
  --augment-hex-symmetry
```

**Configuration**:

```yaml
training:
  use_hex_augmentation: true
```

### Soft Policy Targets (Gumbel MCTS)

Generate training data with soft policy targets from visit distributions:

```bash
python scripts/run_hybrid_selfplay.py \
  --board-type hex8 \
  --engine-mode gumbel-mcts \
  --nn-model-id ringrift_hex8_2p_v3_retrained \
  --num-games 200
```

### Training CLI Arguments

| Argument                    | Type  | Default | Description                        |
| --------------------------- | ----- | ------- | ---------------------------------- |
| `--policy-label-smoothing`  | float | 0.0     | Label smoothing factor             |
| `--augment-hex-symmetry`    | flag  | False   | Enable D6 hex augmentation         |
| `--warmup-epochs`           | int   | 0       | LR warmup epochs                   |
| `--lr-scheduler`            | str   | none    | cosine, step, plateau, warmrestart |
| `--early-stopping-patience` | int   | 10      | Early stopping patience            |
| `--sampling-weights`        | str   | uniform | uniform, late_game, phase_emphasis |

### Advanced Regularization

| Feature          | Config Key             | Default | Description                 |
| ---------------- | ---------------------- | ------- | --------------------------- |
| SWA              | `use_swa`              | true    | Stochastic Weight Averaging |
| EMA              | `use_ema`              | true    | Exponential Moving Average  |
| Spectral Norm    | `use_spectral_norm`    | true    | Gradient stability          |
| Stochastic Depth | `use_stochastic_depth` | true    | Dropout regularization      |
| Focal Loss       | `focal_gamma`          | 2.0     | Hard sample mining          |

### See Also

- [TRAINING_FEATURES.md](TRAINING_FEATURES.md) - Comprehensive training features reference
- [HEX_AUGMENTATION.md](../algorithms/HEX_AUGMENTATION.md) - D6 symmetry augmentation details
- [MCTS_INTEGRATION.md](../algorithms/MCTS_INTEGRATION.md) - MCTS implementation and training data generation

---

## Operational Runbook (2025-12-17)

### Common Bottlenecks and Fixes

#### 1. Data Sync Not Running

**Symptom**: `total_data_syncs: 0` in unified loop state

**Fix**:

```bash
# Start sync daemon
python scripts/unified_data_sync.py --watchdog &

# Or use cron script
./scripts/sync_training_data_cron.sh
```

#### 2. Training Not Triggering

**Symptom**: `total_training_runs: 0` despite sufficient games

**Fix**:

```bash
# Reset loop state and restart
python -c "
import json
with open('logs/unified_loop/unified_loop_state.json', 'r+') as f:
    state = json.load(f)
    state['training_in_progress'] = False
    for host in state.get('hosts', {}).values():
        host['last_sync_time'] = 0.0
    f.seek(0)
    json.dump(state, f, indent=2)
    f.truncate()
"
pkill -f unified_ai_loop.py
python scripts/unified_ai_loop.py --start &
```

#### 3. Underperforming Configs

**Symptom**: Config has <10 models or Elo below 1500

**Fix**: Launch dedicated training loop

```bash
ssh gpu-node-1 "cd ~/ringrift/ai-service && source venv/bin/activate && \
  python scripts/multi_config_training_loop.py --board hexagonal --players 3 --iterations 30 &"
```

#### 4. Missing MCTS Data for KL Loss

**Symptom**: NNUE policy training shows "No MCTS coverage"

**Fix**: Start Gumbel MCTS selfplay

```bash
python scripts/run_hybrid_selfplay.py --board-type square8 --engine-mode gumbel-mcts \
  --mcts-sims 200 --num-games 500
```

### Cron Jobs

Install with: `crontab config/crontab_training.txt`

| Schedule       | Job                          | Description                |
| -------------- | ---------------------------- | -------------------------- |
| `*/15 * * * *` | `sync_training_data_cron.sh` | Sync data from GPU cluster |
| `0 3 * * *`    | `prune_models.py --auto`     | Daily model pruning        |
| `*/30 * * * *` | `vast_lifecycle.py --check`  | Cloud VM health monitoring |

### Active Daemons

| Daemon                   | Host       | Purpose                     |
| ------------------------ | ---------- | --------------------------- |
| `unified_ai_loop.py`     | Local      | Main orchestration          |
| `unified_data_sync.py`   | Local      | Data collection             |
| `auto_elo_tournament.py` | Train Node | Model evaluation            |
| `baseline_gauntlet.py`   | GPU Node   | Continuous baseline testing |

### Quick Status Check

```bash
# Overall status
PYTHONPATH=. python scripts/unified_ai_loop.py --status

# Training jobs
ps aux | grep -E 'train|selfplay' | grep python | wc -l

# Game counts
sqlite3 data/games/all_jsonl_training.db \
  "SELECT board_type, num_players, COUNT(*) FROM games GROUP BY 1,2"

# Model counts
sqlite3 data/unified_elo.db \
  "SELECT board_type, num_players, COUNT(*), MAX(rating) FROM elo_ratings WHERE archived_at IS NULL GROUP BY 1,2"
```

### Curriculum Training

Progressive training from simple to complex game phases:

```bash
# Auto-progress through all stages
python scripts/curriculum_training.py --auto-progress \
  --board hexagonal --num-players 3 \
  --db data/games/jsonl_aggregated.db

# Train specific stage only
python scripts/curriculum_training.py --stage 1 \
  --board square8 --num-players 2
```

**Stages**:

1. Placement phase (moves 1-36)
2. Early-mid game (moves 37-100)
3. Mid-late game (moves 100-200)
4. Endgame (moves 200+)
5. Full game

### Transfer Learning

Bootstrap weak configs from strong models:

```bash
python scripts/train_nnue_policy.py \
  --board hexagonal --num-players 3 \
  --db data/games/jsonl_aggregated.db \
  --pretrained models/nnue/nnue_policy_square8_2p.pt \
  --freeze-value \
  --epochs 30 --use-swa
```

**Options**:

- `--pretrained`: Path to pre-trained model
- `--freeze-value`: Freeze value head, train policy only
- `--use-swa`: Stochastic weight averaging for stability

### Network Configuration

Configure your cluster nodes in `config/cluster_hosts.yaml.example`:

| Node Type    | Example Tailscale IP | Example Direct IP |
| ------------ | -------------------- | ----------------- |
| gpu-node-1   | 100.x.x.x            | 10.0.0.1          |
| gpu-node-2   | 100.x.x.x            | 10.0.0.2          |
| training-srv | 100.x.x.x            | 10.0.0.10         |

The sync script (`scripts/sync_training_data_cron.sh`) tries Tailscale first, then falls back to direct IP.

**Setup**: Copy `config/cluster_hosts.yaml.example` to `config/cluster_hosts.yaml` and fill in your IPs.

## Contact

For issues with the training pipeline, check:

1. This documentation
2. `logs/unified_ai_loop.log`
3. `logs/unified_loop/unified_loop_state.json`
