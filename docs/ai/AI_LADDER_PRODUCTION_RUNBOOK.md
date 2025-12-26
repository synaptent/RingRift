# AI Ladder Production Runbook – Square-8 2-Player

> **Status:** Active
> Quick-start guide for running AI tier training and promotion in production.
> For full details, see [AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md](AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md).

---

## Prerequisites

### Environment

```bash
# From project root
cd ai-service
source venv/bin/activate  # or your Python env
pip install -r requirements.txt
```

### Compute Requirements

| Tier(s) | Training Time (Demo) | Training Time (Full) | Notes                                                |
| ------- | -------------------- | -------------------- | ---------------------------------------------------- |
| D1      | —                    | —                    | Baseline only (no training)                          |
| D2–D3   | ~30s                 | 5-20 min             | Heuristic CMA-ES optimization                        |
| D4–D5   | ~30s                 | 10-30 min            | Minimax persona + NNUE (CPU OK for Square-8)         |
| D6      | ~30s                 | 1-4 hours            | Neural Descent training (GPU required)               |
| D7      | ~30s                 | 10-30 min            | Heuristic-only MCTS persona (CPU heavy at full eval) |
| D8      | ~30s                 | 2-8 hours            | MCTS + NN training (GPU required)                    |
| D9–D10  | ~30s                 | 4-12 hours           | Gumbel MCTS + NN (GPU required)                      |

Full training runs require GPU for D6–D10 neural tiers (plus D5 if retraining NNUE checkpoints). Demo mode runs on CPU.

### Required Files

- Canonical selfplay DBs in `TRAINING_DATA_REGISTRY.md` with status=`canonical`
- Tier candidate registry: `ai-service/config/tier_candidate_registry.square8_2p.json`
- Ladder runtime overrides (optional): `ai-service/data/ladder_runtime_overrides.json`

### Cluster Config References

- `ai-service/config/cluster_nodes.yaml` – cluster inventory + SSH keys
- `ai-service/config/distributed_hosts.yaml` – roles, sync routing, resource metadata
- `ai-service/config/p2p_hosts.yaml` – P2P orchestrator host list
- `ai-service/config/node_policies.yaml` – allowed work types per node class
- `ai-service/config/selfplay_workers.example.yaml` – template (copy to `selfplay_workers.yaml` if needed)
- `ai-service/config/unified_loop.yaml` – unified loop config
- `ai-service/config/tier_training_pipeline.square8_2p.json` – tier training defaults
- `ai-service/config/pipeline.json` – legacy pipeline config (not used by unified loop)

---

## Quick Start: Demo Mode (CI/Smoke)

Run both training and gating in demo mode (fast, no heavy compute):

```bash
cd ai-service

# Step 1: Train a candidate (demo mode)
PYTHONPATH=. python scripts/run_tier_training_pipeline.py \
  --tier D4 \
  --board square8 \
  --num-players 2 \
  --output-dir /tmp/tier_training_demo \
  --demo \
  --seed 42

# Step 2: Gate the candidate (demo mode)
RUN_DIR=$(ls -td /tmp/tier_training_demo/D4_* | head -1)
CANDIDATE_ID=$(jq -r '.candidate_id' "$RUN_DIR/training_report.json")
PYTHONPATH=. python scripts/run_full_tier_gating.py \
  --tier D4 \
  --candidate-id "$CANDIDATE_ID" \
  --run-dir "$RUN_DIR" \
  --demo
```

Outputs in `--run-dir`:

- `training_report.json` – Training configuration and candidate ID
- `tier_eval_result.json` – Evaluation results vs baselines
- `promotion_plan.json` – Promotion decision (promote/reject)
- `tier_perf_report.json` – Performance benchmark results (when a budget exists)
- `gate_report.json` – Combined summary with `final_decision`
- `status.json` – Pipeline status tracker

---

## Production Run: Full Training Cycle

### Step 1: Create Run Directory

```bash
OUTPUT_DIR="ai-service/logs/tier_gate"
mkdir -p "$OUTPUT_DIR"
```

### Step 2: Train Candidate

```bash
cd ai-service
PYTHONPATH=. python scripts/run_tier_training_pipeline.py \
  --tier D4 \
  --board square8 \
  --num-players 2 \
  --output-dir "$OUTPUT_DIR" \
  --seed 1
```

Identify the run directory (the one containing `training_report.json`), then monitor progress in `status.json`:

```bash
RUN_DIR=$(ls -td "$OUTPUT_DIR"/D4_* | head -1)
```

### Step 3: Gate Candidate

```bash
CANDIDATE_ID=$(jq -r '.candidate_id' "$RUN_DIR/training_report.json")
PYTHONPATH=. python scripts/run_full_tier_gating.py \
  --tier D4 \
  --candidate-id "$CANDIDATE_ID" \
  --run-dir "$RUN_DIR"
```

### Step 4: Verify Results

```bash
# Check final decision
jq '.final_decision' "$RUN_DIR/gate_report.json"

# Check evaluation pass
jq '.tier_eval.overall_pass' "$RUN_DIR/gate_report.json"

# Check perf pass (D3–D8 only)
jq '.tier_perf.overall_pass' "$RUN_DIR/gate_report.json"
```

### Step 5: Promotion (if passed)

If `final_decision: "promote"`, update the ladder config:

1. Apply the promotion plan to the candidate registry:

   ```bash
   python scripts/apply_tier_promotion_plan.py \
     --plan-path "$RUN_DIR/promotion_plan.json"
   ```

2. Promote the ladder:
   - **Runtime override:** update `ai-service/data/ladder_runtime_overrides.json` (fastest).
   - **Permanent change:** update `ai-service/app/config/ladder_config.py`.

3. Archive artifacts:
   ```bash
   mv "$RUN_DIR" ai-service/data/promotions/square8_2p/D4/
   ```

---

## Tier-Specific Commands

### D2/D3 (Heuristic CMA-ES)

```bash
# Training (heuristic CMA-ES optimization)
PYTHONPATH=. python scripts/run_tier_training_pipeline.py \
  --tier D2 --board square8 --num-players 2 --output-dir "$OUTPUT_DIR"

# Gating (no perf budget for D2; D3 uses perf budget)
PYTHONPATH=. python scripts/run_full_tier_gating.py \
  --tier D2 --candidate-id "$CANDIDATE_ID" --run-dir "$RUN_DIR" --no-perf
```

### D5 (Minimax + NNUE)

```bash
# CPU OK for sq8; use GPU only if retraining NNUE checkpoints
PYTHONPATH=. python scripts/run_tier_training_pipeline.py \
  --tier D5 --board square8 --num-players 2 --output-dir "$OUTPUT_DIR"
```

### D6 (Neural Descent)

```bash
# Requires GPU for full training
PYTHONPATH=. python scripts/run_tier_training_pipeline.py \
  --tier D6 --board square8 --num-players 2 --output-dir "$OUTPUT_DIR"
```

### D7 (Heuristic-only MCTS)

```bash
# CPU-heavy but no neural training
PYTHONPATH=. python scripts/run_tier_training_pipeline.py \
  --tier D7 --board square8 --num-players 2 --output-dir "$OUTPUT_DIR"
```

### D8 (MCTS + Neural)

```bash
# Longest training time, requires GPU
PYTHONPATH=. python scripts/run_tier_training_pipeline.py \
  --tier D8 --board square8 --num-players 2 --output-dir "$OUTPUT_DIR"
```

### D9/D10 (Gumbel MCTS + Neural)

```bash
# Heaviest GPU workloads
PYTHONPATH=. python scripts/run_tier_training_pipeline.py \
  --tier D9 --board square8 --num-players 2 --output-dir "$OUTPUT_DIR"
```

---

## Gating Criteria Summary

| Tier | Min Win Rate vs Baseline | Max Regression vs Previous | Perf Budget |
| ---- | ------------------------ | -------------------------- | ----------- |
| D1   | N/A                      | N/A                        | None        |
| D2   | ≥60% vs D1 random        | ≤5% vs D1                  | None        |
| D3   | ≥55% vs baselines        | ≤5% vs D2                  | Required    |
| D4   | ≥68% vs baselines        | ≤5% vs D3                  | Required    |
| D5   | ≥60% vs baselines        | ≤5% vs D4                  | Required    |
| D6   | ≥72% vs baselines        | ≤5% vs D5                  | Required    |
| D7   | ≥65% vs baselines        | ≤5% vs D6                  | Required    |
| D8   | ≥75% vs baselines        | ≤5% vs D7                  | Required    |
| D9   | ≥75% vs baselines        | ≤5% vs D8                  | None        |
| D10  | ≥75% vs baselines        | ≤5% vs D9                  | None        |

From `ai-service/app/training/tier_eval_config.py` and `ai-service/app/config/perf_budgets.py`.

---

## Troubleshooting

### Training Report Missing

```bash
# Verify training completed
cat "$RUN_DIR/status.json"
# Should show: "training": {"status": "completed"}
```

### Gate Failed

```bash
# Check specific failure reason
jq '.tier_eval' "$RUN_DIR/gate_report.json"
jq '.reason' "$RUN_DIR/promotion_plan.json"
```

### Perf Budget Exceeded

Options:

1. Reduce search depth/iterations and retrain
2. Update perf budgets in `ai-service/app/config/perf_budgets.py` (requires justification)
3. Reject candidate and keep current production tier

### Missing Canonical Data

```bash
# Check registry for canonical DBs
cat ai-service/TRAINING_DATA_REGISTRY.md | grep -A5 "canonical"
```

Only DBs with status=`canonical` should be used for production training.

---

## Related Documentation

- [AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md](AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md) – Full pipeline design
- [AI_CALIBRATION_RUNBOOK.md](AI_CALIBRATION_RUNBOOK.md) – Human calibration procedures
- [AI_TIER_PERF_BUDGETS.md](AI_TIER_PERF_BUDGETS.md) – Performance budget specifications
- [AI_HUMAN_CALIBRATION_GUIDE.md](AI_HUMAN_CALIBRATION_GUIDE.md) – Human testing templates
