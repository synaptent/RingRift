# Anti-Regression Framework

**Created:** April 5, 2026
**Purpose:** Prevent the recurring "everything appears working but actually isn't" failure pattern.

## Components

| Priority | Component                    | Status       | Catches                                         |
| -------- | ---------------------------- | ------------ | ----------------------------------------------- |
| **P0**   | Training Effectiveness Probe | IMPLEMENTING | Model trains but doesn't learn                  |
| **P0**   | Data Quality Sentinel        | IMPLEMENTING | Degenerate/identical training data              |
| **P1**   | Fleet Health Aggregator      | IMPLEMENTING | Stalled nodes invisible to coordinator          |
| **P1**   | Pipeline Integrity Gate      | TODO         | Deploy-induced regressions                      |
| **P2**   | Model Quality Gate           | TODO         | Bad promotions (mode collapse, dead value head) |
| **P2**   | Self-Healing & Escalation    | TODO         | Auto-restart, GPU waste tracking                |

## How Each Historical Failure Would Be Caught

| Historical Failure                       | Component                                  | Detection Time      |
| ---------------------------------------- | ------------------------------------------ | ------------------- |
| Training data identical (randomness=0.0) | Data Quality Sentinel                      | 1st iteration       |
| Square boards 0 input channels           | Training Probe (inference)                 | 1st iteration       |
| Arch registry v5-heavy for v2            | Training Probe (forward pass)              | 1st iteration       |
| Eval fell back to random play            | Training Probe (fallback detection)        | 1st evaluation      |
| SSH key missing → thread starvation      | Fleet Aggregator (heartbeat stale)         | 2 hours             |
| Gauntlet budget 25x too shallow          | Model Quality Gate (behavioral diversity)  | 1st promotion       |
| Promotion criteria too loose             | Model Quality Gate (strengthened criteria) | 1st noisy promotion |

## Integration Points in minimal_alphazero_loop.py

```
1. SELFPLAY
2. EXPORT
   → 2.5 DATA QUALITY CHECK (Data Quality Sentinel)
3. TRAIN
   → 3.5 TRAINING PROBE (Training Effectiveness Probe)
4. EVALUATE
5. PROMOTE/REJECT
   → 5.5 PUSH HEARTBEAT (Fleet Health Aggregator)
```

## Thresholds

### Data Quality

| Metric                   | WARN       | CRITICAL           |
| ------------------------ | ---------- | ------------------ |
| Policy entropy median    | < 1.5 bits | < 0.5 bits         |
| Feature channel variance | < 0.01     | all zeros          |
| Cross-iteration delta    | < 0.05     | < 0.01 (identical) |
| Value target std         | < 0.05     | < 0.01             |

### Training Probe

| Check                     | FAIL Threshold         |
| ------------------------- | ---------------------- |
| Policy entropy (10 moves) | All < 0.5 bits         |
| Value head output         | All == 0.0             |
| Weight delta L2 norm      | < 1e-8 (zero gradient) |
| Loss convergence          | final >= initial       |

### Fleet Health

| Heartbeat Age | Status          |
| ------------- | --------------- |
| < 2 hours     | HEALTHY         |
| 2-6 hours     | STALE (warning) |
| > 6 hours     | DEAD (critical) |
