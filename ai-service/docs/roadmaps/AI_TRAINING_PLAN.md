# AI Training Plan: Heuristic Weight Optimization & Neural Network Training

> **Doc Status (2025-12-14): Active**

This document outlines the complete training pipeline for optimizing RingRift AI,
covering heuristic weight optimization via CMA-ES and neural network training.

## Phase Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 0: Prerequisites & Validation                                     │
├─────────────────────────────────────────────────────────────────────────┤
│ Phase 1: Generate Evaluation State Pools                                │
├─────────────────────────────────────────────────────────────────────────┤
│ Phase 2: CMA-ES Heuristic Weight Optimization                          │
│   └─ 2a: Square8 → 2b: Square19 → 2c: Hex (radius-12)                  │
├─────────────────────────────────────────────────────────────────────────┤
│ Phase 3: Validate & Promote Heuristic Weights                          │
├─────────────────────────────────────────────────────────────────────────┤
│ Phase 4: Generate Neural Network Training Data                         │
├─────────────────────────────────────────────────────────────────────────┤
│ Phase 5: Neural Network Training                                       │
│   └─ 5a: Square8 → 5b: Hex (radius-12) → 5c: Square19                  │
├─────────────────────────────────────────────────────────────────────────┤
│ Phase 6: Validation & Deployment                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 0: Prerequisites & Validation

```bash
cd ai-service

# 1. Verify environment
python scripts/training_preflight_check.py

# 2. Create output directories
mkdir -p logs/cmaes/runs data/eval_pools/{square8,square19,hex}
mkdir -p data/training/{square8,square19,hex}
mkdir -p models/checkpoints

# 3. Verify GPU/MPS availability
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}, CUDA: {torch.cuda.is_available()}')"

# 4. Install CMA-ES if needed
pip install cma>=3.3.0
```

---

## Phase 1: Generate Evaluation State Pools

Generate mid/late-game states for multi-start evaluation:

```bash
# Square8 (fast - ~30 min)
RINGRIFT_SKIP_SHADOW_CONTRACTS=true python scripts/run_self_play_soak.py \
    --board square8 \
    --games 200 \
    --sample-mid-game \
    --output data/eval_pools/square8/pool_v1.jsonl

# Square19 (slower - ~2-3 hours)
RINGRIFT_SKIP_SHADOW_CONTRACTS=true python scripts/run_self_play_soak.py \
    --board square19 \
    --games 100 \
    --sample-mid-game \
    --output data/eval_pools/square19/pool_v1.jsonl

# Hex radius-12 (moderate - ~1-2 hours)
RINGRIFT_SKIP_SHADOW_CONTRACTS=true python scripts/run_self_play_soak.py \
    --board hexagonal \
    --games 100 \
    --sample-mid-game \
    --output data/eval_pools/hex/pool_v1.jsonl
```

---

## Phase 2: CMA-ES Heuristic Weight Optimization

### 2a: Square8 (Fastest iteration, establish baseline)

```bash
# Quick test run first
RINGRIFT_SKIP_SHADOW_CONTRACTS=true python scripts/run_cmaes_optimization.py \
    --generations 5 \
    --population-size 8 \
    --games-per-eval 10 \
    --board square8 \
    --eval-boards square8 \
    --eval-mode multi-start \
    --state-pool-id v1 \
    --output logs/cmaes/square8_test

# Full optimization run
RINGRIFT_SKIP_SHADOW_CONTRACTS=true python scripts/run_cmaes_optimization.py \
    --generations 50 \
    --population-size 20 \
    --games-per-eval 32 \
    --board square8 \
    --eval-boards square8 \
    --eval-mode multi-start \
    --state-pool-id v1 \
    --eval-randomness 0.02 \
    --seed 42 \
    --output logs/cmaes/square8_v2
```

### 2b: Square19 (Use optimized performance flags)

```bash
RINGRIFT_SKIP_SHADOW_CONTRACTS=true \
RINGRIFT_USE_MAKE_UNMAKE=true \
RINGRIFT_USE_BATCH_EVAL=true \
RINGRIFT_BATCH_EVAL_THRESHOLD=50 \
RINGRIFT_USE_FAST_TERRITORY=true \
python scripts/run_cmaes_optimization.py \
    --generations 30 \
    --population-size 16 \
    --games-per-eval 24 \
    --board square19 \
    --eval-boards square19 \
    --eval-mode multi-start \
    --state-pool-id v1 \
    --max-moves 300 \
    --eval-randomness 0.02 \
    --seed 42 \
    --output logs/cmaes/square19_v2
```

### 2c: Hex (Radius-12, 25×25 grid)

```bash
RINGRIFT_SKIP_SHADOW_CONTRACTS=true \
RINGRIFT_USE_MAKE_UNMAKE=true \
RINGRIFT_USE_BATCH_EVAL=true \
python scripts/run_cmaes_optimization.py \
    --generations 30 \
    --population-size 16 \
    --games-per-eval 24 \
    --board hexagonal \
    --eval-boards hexagonal \
    --eval-mode multi-start \
    --state-pool-id v1 \
    --max-moves 300 \
    --eval-randomness 0.02 \
    --seed 42 \
    --output logs/cmaes/hex_v2
```

### 2d: Multi-Board Joint Optimization (Optional, for generalist weights)

```bash
RINGRIFT_SKIP_SHADOW_CONTRACTS=true \
python scripts/run_cmaes_optimization.py \
    --generations 40 \
    --population-size 24 \
    --games-per-eval 16 \
    --board square8 \
    --eval-boards square8,square19,hex \
    --eval-mode multi-start \
    --state-pool-id v1 \
    --output logs/cmaes/multiboard_v2
```

---

## Phase 3: Validate & Promote Heuristic Weights

```bash
# Validate individual board weights
python scripts/validate_and_promote_weights.py \
    logs/cmaes/square8_v2/runs/*/best_weights.json \
    --games 100 \
    --min-win-rate 0.52 \
    --board square8

# Run full post-training pipeline
python scripts/post_training_pipeline.py \
    --training-dir logs/cmaes/square8_v2 \
    --num-players 2 \
    --validation-games 100 \
    --min-win-rate 0.52 \
    --output-dir data

# Merge weights from multiple runs (if applicable)
python scripts/merge_trained_weights.py \
    logs/cmaes/square8_v2/runs/*/best_weights.json \
    --mode fitness-weighted \
    --output data/trained_heuristic_profiles.json
```

**Activate trained weights:**

```bash
export RINGRIFT_TRAINED_HEURISTIC_PROFILES=$(pwd)/data/trained_heuristic_profiles.json
```

---

## Phase 4: Generate Neural Network Training Data

Use trained heuristic weights to generate high-quality training data:

```bash
# Square8 dataset (~50k samples)
RINGRIFT_SKIP_SHADOW_CONTRACTS=true \
RINGRIFT_TRAINED_HEURISTIC_PROFILES=data/trained_heuristic_profiles.json \
python -m app.training.generate_data \
    --board-type square8 \
    --num-games 1000 \
    --output data/training/square8/dataset_v2.npz

# Hex11 dataset with D6 augmentation (~50k x 12 = 600k effective samples)
RINGRIFT_SKIP_SHADOW_CONTRACTS=true \
python -m app.training.generate_data \
    --board-type hexagonal \
    --num-games 1000 \
    --augment-hex \
    --output data/training/hex/dataset_v2.npz

# Square19 dataset
RINGRIFT_SKIP_SHADOW_CONTRACTS=true \
python -m app.training.generate_data \
    --board-type square19 \
    --num-games 500 \
    --output data/training/square19/dataset_v2.npz
```

---

## Phase 5: Neural Network Training

### 5a: Square8 (Fast iteration, validate pipeline)

```bash
python -m app.training.train \
    --data-path data/training/square8/dataset_v2.npz \
    --save-path models/ringrift_square8_v2.pth \
    --board-type square8 \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --lr-scheduler cosine \
    --warmup-epochs 5 \
    --early-stopping-patience 15 \
    --checkpoint-dir models/checkpoints/square8 \
    --seed 42
```

### 5b: Hex (Radius-12, 25×25 with D6 augmentation)

```bash
python -m app.training.train \
    --data-path data/training/hex/dataset_v2.npz \
    --save-path models/ringrift_hex_v2.pth \
    --board-type hexagonal \
    --augment-hex-symmetry \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --lr-scheduler cosine-warm-restarts \
    --lr-t0 10 \
    --lr-t-mult 2 \
    --warmup-epochs 5 \
    --early-stopping-patience 20 \
    --checkpoint-dir models/checkpoints/hex \
    --seed 42
```

### 5c: Square19 (Larger model, more epochs)

```bash
python -m app.training.train \
    --data-path data/training/square19/dataset_v2.npz \
    --save-path models/ringrift_square19_v2.pth \
    --board-type square19 \
    --epochs 150 \
    --batch-size 32 \
    --learning-rate 0.0005 \
    --lr-scheduler cosine \
    --warmup-epochs 10 \
    --early-stopping-patience 20 \
    --checkpoint-dir models/checkpoints/square19 \
    --seed 42
```

### Multi-GPU Training (If Available)

```bash
torchrun --nproc_per_node=2 -m app.training.train \
    --distributed \
    --scale-lr \
    --data-path data/training/square8/dataset_v2.npz \
    --save-path models/ringrift_square8_v2.pth \
    --board-type square8 \
    --epochs 100
```

---

## Phase 6: Validation & Deployment

```bash
# Evaluate trained neural net against heuristic AI
python scripts/evaluate_ai_models.py \
    --model models/ringrift_square8_v2.pth \
    --opponent heuristic \
    --games 100 \
    --board square8

# Run AI tournament
python scripts/run_ai_tournament.py \
    --models heuristic,neural \
    --games 50 \
    --board square8

# Long-running stability test
python scripts/run_self_play_soak.py \
    --ai neural \
    --games 500 \
    --board square8
```

---

## Recommended Execution Order (Priority)

| Priority | Phase | Board     | Reason                                            |
| -------- | ----- | --------- | ------------------------------------------------- |
| 1        | 1+2a  | Square8   | Fastest iteration, validates pipeline             |
| 2        | 3     | Square8   | Validate weights before proceeding                |
| 3        | 4+5a  | Square8   | Generate NN training data with optimized weights  |
| 4        | 2c    | Hex (r12) | Medium complexity, test larger board optimization |
| 5        | 4+5b  | Hex (r12) | NN training with D6 augmentation                  |
| 6        | 2b    | Square19  | Slowest optimization, benefits from learnings     |
| 7        | 4+5c  | Square19  | Final NN training                                 |
| 8        | 6     | All       | Comprehensive validation                          |

---

## Performance Tips

1. **Always use** `RINGRIFT_SKIP_SHADOW_CONTRACTS=true` during training
2. **For large boards**, enable optimization flags:
   - `RINGRIFT_USE_MAKE_UNMAKE=true`
   - `RINGRIFT_USE_BATCH_EVAL=true`
   - `RINGRIFT_USE_FAST_TERRITORY=true`
3. **Start with Square8** to validate pipeline before larger boards
4. **Use multi-start evaluation** (`--eval-mode multi-start`) for more robust weight optimization
5. **Monitor with** `--progress-interval-sec 10` to ensure jobs are running

---

## Environment Variables Reference

| Variable                              | Default | Description                             |
| ------------------------------------- | ------- | --------------------------------------- |
| `RINGRIFT_SKIP_SHADOW_CONTRACTS`      | `false` | Skip contract validation (2-3x speedup) |
| `RINGRIFT_USE_MAKE_UNMAKE`            | `false` | Enable incremental state updates        |
| `RINGRIFT_USE_BATCH_EVAL`             | `false` | Enable batch position evaluation        |
| `RINGRIFT_BATCH_EVAL_THRESHOLD`       | `50`    | Min moves for batch eval                |
| `RINGRIFT_USE_FAST_TERRITORY`         | `false` | NumPy-based territory detection         |
| `RINGRIFT_TRAINED_HEURISTIC_PROFILES` | -       | Path to trained weight profiles         |

---

## Troubleshooting

### CMA-ES stuck at fitness ~0.5

- Enable `--debug-plateau` to see per-candidate W/D/L
- Check if eval pool has sufficient variety
- Try increasing `--games-per-eval` for more stable fitness estimates

### Neural network loss not decreasing

- Check data quality with `python -m app.training.territory_dataset_validation`
- Verify feature dimensions match board type
- Try reducing learning rate

### Out of memory

- Reduce `--batch-size`
- Use `--use-streaming` for large datasets
- Enable gradient checkpointing (if implemented)

---

## Selfplay Data Processing Pipeline

The training pipeline uses JSONL selfplay data converted to NPZ format for neural network training.

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. Selfplay Generation (GPU cluster)                                   │
│    run_self_play_soak.py → data/selfplay/*.jsonl                       │
├─────────────────────────────────────────────────────────────────────────┤
│ 2. Data Conversion                                                      │
│    jsonl_to_npz.py → data/training/*.npz                               │
├─────────────────────────────────────────────────────────────────────────┤
│ 3. Neural Network Training                                              │
│    train.py → models/*.pth                                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### Converting JSONL to NPZ

The `jsonl_to_npz.py` script converts selfplay game records to training-ready NPZ format.
It replays each game move-by-move, extracting 56-channel feature tensors at each position.

**Basic conversion:**

```bash
PYTHONPATH=. python scripts/jsonl_to_npz.py \
    --input-dir data/selfplay/ \
    --output data/training/square8_2p.npz \
    --board-type square8 \
    --num-players 2
```

**With checkpointing (recommended for large datasets):**

```bash
PYTHONPATH=. python scripts/jsonl_to_npz.py \
    --input-dir data/selfplay/ \
    --output data/training/hex_2p.npz \
    --board-type hexagonal \
    --num-players 2 \
    --checkpoint-dir /tmp/hex_checkpoint \
    --checkpoint-interval 50
```

**Resume from checkpoint after interruption:**

```bash
PYTHONPATH=. python scripts/jsonl_to_npz.py \
    --input-dir data/selfplay/ \
    --output data/training/hex_2p.npz \
    --board-type hexagonal \
    --num-players 2 \
    --checkpoint-dir /tmp/hex_checkpoint \
    --resume
```

### NPZ Format Details

Output NPZ files contain:

| Array              | Shape         | Type    | Description                                     |
| ------------------ | ------------- | ------- | ----------------------------------------------- |
| `features`         | (N, 56, H, W) | float32 | 56-channel board features (14 base × 4 history) |
| `globals`          | (N, 20)       | float32 | Global game state features                      |
| `values`           | (N,)          | float32 | Game outcome from current player's perspective  |
| `policy_indices`   | (N,)          | object  | Sparse action indices per sample                |
| `policy_values`    | (N,)          | object  | Sparse action probabilities per sample          |
| `move_numbers`     | (N,)          | int32   | Move index within game                          |
| `total_game_moves` | (N,)          | int32   | Total moves in source game                      |
| `phases`           | (N,)          | object  | Game phase at each position                     |
| `values_mp`        | (N, 4)        | float32 | Multi-player value vectors                      |
| `num_players`      | (N,)          | int32   | Player count per sample                         |

### 56-Channel Feature Encoding

The 14 base channels (× 4 history frames = 56 total):

| Channels | Description                     |
| -------- | ------------------------------- |
| 0-3      | Ring ownership (players 1-4)    |
| 4-7      | Marker ownership (players 1-4)  |
| 8        | Empty cells                     |
| 9        | Ring heights (normalized)       |
| 10       | Valid ring positions            |
| 11       | Current player's turn indicator |
| 12       | Territory control               |
| 13       | Threat positions                |

### Checkpointing Details

Long-running NPZ conversions (3+ hours for large datasets) can be interrupted by SSH timeouts,
system restarts, or other failures. The checkpointing system prevents data loss:

- **Chunk saves**: Every N games, data is saved to a numbered chunk file (`chunk_0000.npz`, etc.)
- **Progress tracking**: A `progress.json` file tracks completed games and chunk metadata
- **Resume support**: With `--resume`, the script skips already-processed games
- **Automatic merge**: On completion, all chunks are merged into the final NPZ file
- **Cleanup**: Checkpoint directory is removed after successful completion

**Memory usage**: Without checkpointing, the script accumulates 6-8GB in RAM for large datasets.
With checkpointing, memory is cleared after each chunk save.

---

## GPU Cluster Operations

### Available Instances

| Alias                     | Hardware | Primary Use                     |
| ------------------------- | -------- | ------------------------------- |
| `lambda-gpu`              | H100     | Training, large model inference |
| `ringrift-staging`        | CPU      | Staging, API testing            |
| `ringrift-selfplay-extra` | CPU      | Additional selfplay capacity    |

### Syncing Code to Instances

```bash
# Lambda H100
ssh lambda-gpu "cd /home/ubuntu/ringrift && git pull"

# Staging
ssh ringrift-staging "cd ~/RingRift && git pull"
```

### Running Long Jobs

Use `nohup` and background processes for long-running tasks:

```bash
# NPZ export with checkpointing
ssh lambda-gpu "cd /home/ubuntu/ringrift/ai-service && \
    source venv/bin/activate && \
    nohup python -u scripts/jsonl_to_npz.py \
        --input-dir data/selfplay/ \
        --output data/training/hex_2p.npz \
        --board-type hexagonal \
        --num-players 2 \
        --checkpoint-dir /tmp/hex_checkpoint \
        --checkpoint-interval 50 \
        > /tmp/npz_export.log 2>&1 &"

# Monitor progress
ssh lambda-gpu "tail -f /tmp/npz_export.log"
```

### Training on GPU

```bash
ssh lambda-gpu "cd /home/ubuntu/ringrift/ai-service && \
    source venv/bin/activate && \
    nohup python -u -m app.training.train \
        --data-path data/training/square8_2p.npz \
        --save-path models/ringrift_v5_sq8_2p.pth \
        --checkpoint-dir checkpoints/v5_sq8_2p \
        --epochs 100 \
        --batch-size 256 \
        --learning-rate 0.001 \
        --early-stopping-patience 15 \
        --lr-scheduler cosine \
        > /tmp/train.log 2>&1 &"
```

### Monitoring Running Processes

```bash
# Check for running training/export processes
ssh lambda-gpu "ps aux | grep -E 'train|jsonl_to_npz' | grep python"

# Check GPU utilization
ssh lambda-gpu "nvidia-smi"

# Check memory usage
ssh lambda-gpu "free -h"
```

---

## Quick Reference

### Common Commands

```bash
# Check selfplay data counts
find data/selfplay -name "*.jsonl" -exec wc -l {} + | tail -1

# Inspect NPZ file
python -c "import numpy as np; d=np.load('data.npz', allow_pickle=True); print({k: d[k].shape for k in d.files})"

# Count training samples
python -c "import numpy as np; d=np.load('data.npz'); print(f'{len(d[\"values\"])} samples')"
```

### Board Type Reference

| Board Type  | Size  | Policy Size | Primary Use               |
| ----------- | ----- | ----------- | ------------------------- |
| `square8`   | 8×8   | 7,000       | Fast iteration, testing   |
| `square19`  | 19×19 | 67,000      | Standard competitive play |
| `hexagonal` | 25×25 | 54,244      | Hex board variant (r12)   |

**Note:** Hex board uses radius-12 geometry (469 cells, 96 rings, 25×25 input). Old radius-10 models are deprecated.

### Model Naming Convention

```
ringrift_v{version}_{board}_{players}p.pth

Examples:
- ringrift_v5_sq8_2p.pth    (Square8, 2-player, version 5)
- ringrift_v2_hex_2p.pth    (Hexagonal, 2-player, version 2)
- ringrift_v1_sq19_4p.pth   (Square19, 4-player, version 1)
```
