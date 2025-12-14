# Training Pipeline Status Report

> **Date:** 2025-12-11
> **Purpose:** Document current state of distributed selfplay and training infrastructure

---

## Executive Summary

The RingRift AI training pipeline is actively generating selfplay data across a heterogeneous 10-machine cluster. All major instances are online and producing games with good game balance metrics.

**Key Metrics:**

- **Total Active Processes:** 420+ across cluster
- **Game Balance:** P1 48.6%, P2 50.5%, Draw 0.7% (healthy balance)
- **Average Game Length:** 113 moves (realistic gameplay)
- **GPU Rules Parity:** **100%** (verified; see `ai-service/docs/GPU_RULES_PARITY_AUDIT.md`)

---

## Cluster Status

| Instance       | Type             | Processes | Data Size | Status        |
| -------------- | ---------------- | --------- | --------- | ------------- |
| Lambda H100    | GPU (H100)       | 4         | 122MB     | Active        |
| Lambda A10     | GPU (A10)        | 2         | 68MB      | Active        |
| Vast 5090 Dual | GPU (2x5090)     | 386       | 25MB      | Active        |
| Vast 5090 Quad | GPU (4x5090)     | 28        | 11MB      | Active        |
| Vast 3090      | GPU (3090)       | -         | -         | Pending check |
| AWS Staging    | CPU (r5.4xlarge) | -         | -         | Pending check |
| Mac Studio     | CPU (M3 Max)     | -         | -         | Local cluster |
| MBP 64GB       | CPU (Intel)      | -         | -         | Local cluster |
| MBP 16GB       | CPU (M1)         | -         | -         | Local cluster |

---

## Game Balance Analysis (2025-12-11)

### Square8 2-Player Games

Data from Vast 5090 Dual overnight run (2011 games):

| Metric       | Value                   |
| ------------ | ----------------------- |
| P1 Win Rate  | 48.6%                   |
| P2 Win Rate  | 50.5%                   |
| Draw Rate    | 0.7%                    |
| Avg Moves    | 113.0                   |
| Victory Type | ring_elimination (100%) |

**Assessment:** Excellent balance. The slight P2 advantage suggests the heuristic AI may favor reactive play. Draw rate is appropriately low for a decisive game.

---

## Data Formats

### Selfplay Output (JSONL)

Each game is recorded as a JSON line containing:

```json
{
  "game_id": "uuid",
  "board_type": "square8",
  "num_players": 2,
  "winner": 1,
  "victory_type": "ring_elimination",
  "move_count": 87,
  "moves": [...],
  "final_state": {...}
}
```

### Training Data (NPZ)

Exported for NN/NNUE training:

- `features.npy`: Board state features
- `policies.npy`: Move probability distributions
- `values.npy`: Game outcomes

---

## Current Training Activities

### Heuristic Weight Optimization (CMA-ES)

- **Status:** Ready to run
- **Script:** `scripts/run_iterative_cmaes.py`
- **Output:** `data/cmaes/`

### NNUE Training

- **Status:** Infrastructure ready
- **Script:** `scripts/train_nnue.py`
- **Data Required:** ~10k games minimum per board/player combo

### Neural Network Training

- **Status:** Pending sufficient data
- **Script:** `app/training/train.py`
- **Data Required:** ~50k games per board type

---

## Known Issues

1. **GPU Rules Parity - RESOLVED (100%)**
   - ✅ Chain captures: Loop until exhausted (implemented 2025-12-11)
   - ✅ Territory processing: Full flood-fill with cap eligibility
   - ✅ Overlength lines: Probabilistic Option 1/2 selection
   - ✅ Recovery cascade: Full implementation
   - See `ai-service/docs/GPU_FULL_PARITY_PLAN_2025_12_11.md` for verified status
   - All 82 GPU tests passing as of 2025-12-12

2. **Recovery Moves**
   - Canonical rules implemented (RR-CANON-R110)
   - Recovery slide mechanic available in MOVEMENT phase
   - Low occurrence rate in practice (~0.1% of games)

3. **Overlength Lines**
   - GPU uses stochastic Option 1/2 selection
   - 70% Option 1 (all markers, eliminate), 30% Option 2 (subset, no eliminate)

4. **Neural Checkpoint Compatibility (MCTS/Descent Tiers)**
   - **Symptom:** Neural tiers (D6–D10) can silently fall back to heuristic rollouts when NN loading fails.
   - **Common failure mode:** Feature-shape mismatch (e.g. `value_fc1 in_features: checkpoint=212, expected=148`)
     when a checkpoint with 192 filters is loaded into a 128-filter model.
   - **Mitigations implemented (2025-12-12):**
     - `NeuralNetAI` no longer defaults to deprecated `ringrift_v1` IDs. When `nn_model_id` is unset, it auto-selects the
       best available local model for the board (preferring `ringrift_v4_*` then `ringrift_v3_*`, then `*_nn_baseline`).
     - Default architecture for unknown metadata is now the canonical v2 "high" tier (12 res blocks, 192 filters) to avoid
       the `212 vs 148` mismatch when checkpoint metadata is missing/unreadable.
     - Checkpoint metadata inference is hardened (handles `module.conv1.weight` / suffix keys) and DDP (`module.*`) keys are
       normalized before shape checks and load.
   - **Debug helper:** `ai-service/scripts/inspect_nn_checkpoint.py` (use `--strict` for CI/tournament gating).
   - **Strict mode (debug/tournaments):** set `RINGRIFT_REQUIRE_NEURAL_NET=1` to fail fast instead of degrading.

---

## Recommended Next Steps

### Immediate (This Session)

1. Generate a fresh **canonical** self-play DB via `ai-service/scripts/generate_canonical_selfplay.py` and confirm `canonical_ok: true`
2. Export training NPZ from the canonical DB via `ai-service/scripts/export_replay_dataset.py`
3. Train a v2 baseline model (current stable) via `ai-service/app/training/train.py`
4. Run an AI strength tournament with neural tiers enabled and `RINGRIFT_REQUIRE_NEURAL_NET=1` to prevent silent fallback

### Short-Term (This Week)

1. Accumulate 50k+ canonical games for square8 2p (then square8 3p/4p)
2. If/when v3 architectures (`memory_tier=v3-high/v3-low`) are promoted beyond experimental, train v3 checkpoints and compare against v2 via tournaments
3. Add regression gating: reject checkpoints that fail `inspect_nn_checkpoint.py --strict`

### Medium-Term

1. Accumulate 50k+ games per board/player combination
2. Run CMA-ES optimization on accumulated data
3. Begin NNUE training with optimized heuristics

---

## Data Locations

| Type         | Path                   | Description               |
| ------------ | ---------------------- | ------------------------- |
| Raw Selfplay | `data/selfplay/*/`     | Per-run JSONL files       |
| Synced Data  | `data/games/synced_*/` | Consolidated from cluster |
| Training NPZ | `data/training/`       | Exported for NN training  |
| Models       | `models/`              | Trained weights           |

---

## Scripts Reference

| Script                             | Purpose                       |
| ---------------------------------- | ----------------------------- |
| `scripts/run_hybrid_selfplay.py`   | Generate selfplay games       |
| `scripts/export_replay_dataset.py` | Convert DB to training format |
| `scripts/train_nnue.py`            | Train NNUE network            |
| `scripts/run_iterative_cmaes.py`   | Optimize heuristic weights    |
| `scripts/sync_selfplay_data.sh`    | Pull data from cluster        |
| `scripts/p2p_orchestrator.py`      | Distributed job coordination  |

---

## Environment Configuration

Key environment variables for training:

```bash
# Performance flags
export RINGRIFT_SKIP_SHADOW_CONTRACTS=true  # 2-3x speedup
export RINGRIFT_USE_MAKE_UNMAKE=true        # Incremental state
export RINGRIFT_USE_BATCH_EVAL=true         # Batch evaluation
export RINGRIFT_USE_FAST_TERRITORY=true     # NumPy territory

# Training data
export RINGRIFT_TRAINED_HEURISTIC_PROFILES=data/trained_heuristic_profiles.json
```

---

## Appendix: Cluster Connection Details

See `config/distributed_hosts.yaml` and `config/remote_hosts.yaml` for full connection details (gitignored for security).

### Quick Access

```bash
# Lambda instances
ssh ubuntu@209.20.157.81  # H100
ssh ubuntu@150.136.65.197 # A10

# Vast instances (custom ports)
ssh -p 18080 root@178.43.61.252  # 5090 Dual
ssh -p 45875 root@211.72.13.202  # 5090 Quad
ssh -p 47070 root@79.116.93.241  # 3090

# Local Mac cluster (via Tailscale)
ssh armand@100.107.168.125  # Mac Studio
ssh armand@100.92.222.49    # MBP 64GB
ssh armand@100.66.142.46    # MBP 16GB
```
