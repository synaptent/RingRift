# Model Lifecycle Documentation

This document describes the complete lifecycle of neural network models in the RingRift AI training pipeline.

## Overview

```
Training → Evaluation → Promotion → Deployment → Monitoring → Archival
```

## 1. Training Stage

### 1.1 Data Generation

Models are trained on data from canonical selfplay databases.

**Location**: `data/games/canonical_*.db`

**Required volume**: 200+ games per board/player configuration

**Current status** (Dec 2025):
| Config | Games | Status |
|--------|-------|--------|
| square8_2p | 1,152 | ✅ Ready |
| square8_4p | 11,514 | ✅ Ready |
| hex8_2p | 295 | ✅ Ready |
| hex8_3p | 528 | ✅ Ready |
| hex8_4p | 1,284 | ✅ Ready |
| square8_3p | 67 | ⚠️ Scaling |
| square19_2p | 56 | ⚠️ Scaling |
| hexagonal_4p | 10 | ⚠️ Scaling |

### 1.2 Training Execution

```bash
# Export training data
python scripts/export_replay_dataset.py \
  --use-discovery --board-type hex8 --num-players 2 \
  --output data/training/hex8_2p.npz

# Train model
python -m app.training.train \
  --board-type hex8 --num-players 2 \
  --data-path data/training/hex8_2p.npz \
  --save-path models/hex8_2p_new.pt
```

### 1.3 Auto-Registration

On save, models are automatically:

1. Checksummed (SHA256)
2. Versioned with metadata
3. Registered in `data/model_registry/registry.db` as DEVELOPMENT stage

**Key files**:

- `app/training/model_versioning.py` - Versioning utilities
- `app/training/model_registry.py` - Registry database

## 2. Evaluation Stage

### 2.1 Gauntlet Evaluation

Models are evaluated against fixed baselines:

```bash
python scripts/baseline_gauntlet.py models/hex8_2p_new.pt \
  --board hex8 --games 30
```

**Baselines**:

- Random AI (target: ≥85% win rate)
- Heuristic AI (target: ≥60% win rate)

### 2.2 ELO Tournament

For competitive comparison:

```bash
python -m app.tournament.distributed_gauntlet \
  --model models/hex8_2p_new.pt
```

## 3. Promotion Criteria

### 3.1 Stage Transitions

| Transition            | Required Criteria                                                |
| --------------------- | ---------------------------------------------------------------- |
| DEVELOPMENT → STAGING | 50+ games, positive ELO delta                                    |
| STAGING → PRODUCTION  | 100+ games, ELO ≥1650, WR vs Random ≥90%, WR vs Heuristic ≥60%   |
| Any → ARCHIVED        | Superseded by higher ELO model, or age >7 days without promotion |
| Any → REJECTED        | ELO <1400 after 100+ games                                       |

### 3.2 Promotion Command

```bash
python scripts/auto_promote.py --gauntlet \
  --model models/hex8_2p_new.pt \
  --board-type hex8 --num-players 2 \
  --games 50
```

**Key file**: `app/training/promotion_controller.py`

## 4. Production Deployment

### 4.1 Essential Models Directory

Production models are copied to `models_essential/`:

```
models_essential/
├── canonical_sq8_2p_v1.pth     # square8 2-player
├── ringrift_v7_hex_2p.pth      # hex 2-player
├── ebmo_square8_v1.pt          # EBMO variant
└── ...
```

### 4.2 Automated Model Distribution (December 2025)

Models are automatically distributed across the cluster via `ModelDistributionDaemon`:

```
MODEL_PROMOTED event
        │
        ▼
┌──────────────────────┐
│ ModelDistributionDaemon │
│  - Subscribes to EVENT_ROUTER │
│  - Detects promotion events    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Multi-Transport Sync │
│  1. BitTorrent (>50MB) │
│  2. aria2 HTTP (multi-source) │
│  3. rsync --checksum (fallback) │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Post-Transfer Verification │
│  - SHA256 checksum match │
│  - File size validation   │
│  - Model loadability test │
└──────────────────────┘
```

**Key files**:

- `app/coordination/model_distribution_daemon.py` - Automated distribution
- `app/coordination/npz_distribution_daemon.py` - Training data distribution
- `app/distributed/resilient_transfer.py` - Unified transfer abstraction

**Manual sync** (legacy):

```bash
# Sync to all nodes
for node in $(cat config/distributed_hosts.yaml | grep 'host:' | awk '{print $2}'); do
  scp models/production_model.pt ubuntu@$node:~/ringrift/ai-service/models/
done
```

### 4.3 NPZ Training Data Distribution

Training data (NPZ files) are distributed using similar infrastructure:

```bash
# NPZ files are automatically distributed when:
# 1. EXPORT_COMPLETED event fires
# 2. NPZDistributionDaemon detects new NPZ files
# 3. BitTorrent is preferred for large files (>50MB)
```

**Distribution priorities**:

- GPU nodes (training candidates) receive highest priority
- Checksum verification is mandatory
- NPZ structure validation (array shapes, sample counts)

## 5. Monitoring

### 5.1 Model Registry CLI

```bash
# List all models by stage
python scripts/model_registry_cli.py list --stage production

# Get model details
python scripts/model_registry_cli.py info MODEL_ID

# Compare two models
python scripts/model_registry_cli.py compare MODEL_A MODEL_B
```

### 5.2 Performance Tracking

Model performance is tracked in the registry database with:

- Win rates vs baselines
- ELO ratings over time
- Games played count

## 6. Archival Policy

### 6.1 Automatic Archival

Models are archived when:

- Age >7 days AND not in PRODUCTION/STAGING stage
- ELO <1400 after 100+ games evaluated
- Superseded by a higher ELO model for same config
- Match experiment patterns (`*_iter*`, `*_epoch*`, `*_trial*`) AND age >3 days

### 6.2 Archive Structure

```
models/archive/
└── 202512/                    # Year-month folder
    ├── archived_manifest.json # Metadata for all archived models
    ├── iter_checkpoints/      # Experiment checkpoints
    ├── low_elo/              # Performance-based archives
    └── duplicates/           # Superseded models
```

### 6.3 Manual Archival

```bash
python scripts/model_registry_cli.py archive MODEL_ID --reason "superseded"
```

## 7. Rollback Procedure

If a production model regresses:

```bash
# 1. Check model history
python scripts/model_registry_cli.py history --config hex8_2p

# 2. Promote previous version
python scripts/model_registry_cli.py promote PREVIOUS_MODEL_ID --stage production

# 3. Archive regressed model
python scripts/model_registry_cli.py archive REGRESSED_MODEL_ID --reason "regression"
```

## Key Files Reference

| File                                            | Purpose                              |
| ----------------------------------------------- | ------------------------------------ |
| `app/training/model_versioning.py`              | Checkpoint integrity & metadata      |
| `app/training/model_registry.py`                | SQLite lifecycle tracking            |
| `app/training/promotion_controller.py`          | Promotion decision logic             |
| `app/config/thresholds.py`                      | Centralized threshold values         |
| `app/coordination/model_distribution_daemon.py` | Automated cluster model distribution |
| `app/coordination/npz_distribution_daemon.py`   | Training data distribution           |
| `app/distributed/resilient_transfer.py`         | Verified file transfers              |
| `app/coordination/npz_validation.py`            | NPZ structure validation             |
| `scripts/model_registry_cli.py`                 | CLI interface                        |
| `scripts/auto_promote.py`                       | Automated promotion workflow         |
| `scripts/baseline_gauntlet.py`                  | Baseline evaluation                  |
