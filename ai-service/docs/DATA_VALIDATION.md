# Data Validation and Quality Gates

This document covers scripts and workflows for validating training data quality in the RingRift AI pipeline.

## Overview

Data quality is critical for AI training. The validation pipeline ensures:

1. **Parity** - TypeScript and Python game engines produce identical results
2. **Canonical History** - Game states follow legal move sequences
3. **Completeness** - Games have proper start/end states
4. **Consistency** - Features match expected patterns

## Validation Scripts

### check_ts_python_replay_parity.py (Primary Parity Gate)

Verifies that TypeScript and Python game engines produce identical game states.

```bash
# Canonical mode (strict validation)
python scripts/check_ts_python_replay_parity.py --canonical --verbose

# Legacy mode (backward compatible)
python scripts/check_ts_python_replay_parity.py --legacy

# Validate specific JSONL file
python scripts/check_ts_python_replay_parity.py --input data/selfplay/games.jsonl

# Output detailed report
python scripts/check_ts_python_replay_parity.py --canonical --report parity_report.json
```

**Validation Checks:**

- Post-move view semantics (board state after each move)
- Post-bridge view semantics (state after bridge placements)
- Structural issue detection (malformed game records)
- Semantic divergence checking (logic differences)

**Exit Codes:**

- `0` - All games pass parity
- `1` - Parity failures detected
- `2` - Structural errors (malformed data)

### build_canonical_training_pool_db.py (Data Aggregation Gate)

Aggregates training data with strict quality gates.

```bash
# Build canonical training pool
python scripts/build_canonical_training_pool_db.py --output data/games/canonical.db

# With holdout exclusion
python scripts/build_canonical_training_pool_db.py --output data/games/canonical.db --exclude-holdout

# Dry run (validate only)
python scripts/build_canonical_training_pool_db.py --dry-run --verbose
```

**Quality Gates:**

- Per-game canonical history validation
- Parity gate (TS/Python agreement)
- Holdout exclusion (prevent test data leakage)
- Quarantine of failing games

**Output:**

- `canonical.db` - Validated training data
- `quarantine/` - Games that failed validation
- `validation_report.json` - Detailed validation results

### generate_canonical_selfplay.py (End-to-End Validation)

Generates and validates canonical self-play data.

```bash
# Generate with validation
python scripts/generate_canonical_selfplay.py --games 1000 --validate

# Validate existing data
python scripts/generate_canonical_selfplay.py --validate-only --input data/selfplay/

# Run FE/territory fixture tests
python scripts/generate_canonical_selfplay.py --fixtures
```

**Validation Steps:**

1. Generate games via self-play
2. Replay games in both TS and Python
3. Compare final states
4. Run feature extraction validation
5. Territory calculation verification

### holdout_validation.py (Overfitting Detection)

Validates model performance on holdout game sets.

```bash
# View holdout statistics
python scripts/holdout_validation.py --stats

# Evaluate model on holdout
python scripts/holdout_validation.py --evaluate --model models/nnue/square8_2p.pt

# Check for overfitting
python scripts/holdout_validation.py --check-overfitting

# Compare train vs holdout loss
python scripts/holdout_validation.py --gap-analysis
```

**Metrics:**

- Holdout loss vs training loss
- Overfitting gap (warning >0.10, critical >0.15)
- Value head calibration
- Policy accuracy on held-out positions

### training_preflight_check.py (Pre-Training Validation)

Runs validation checks before starting training.

```bash
# Full preflight check
python scripts/training_preflight_check.py --full

# Check specific database
python scripts/training_preflight_check.py --db data/games/training.db

# Resource availability check
python scripts/training_preflight_check.py --resources
```

**Checks:**

- Database integrity (SQLite PRAGMA checks)
- Data volume (minimum samples required)
- Feature consistency (tensor shapes)
- Resource availability (GPU, disk, memory)
- Model checkpoint validity

## Quality Metrics

### Parity Metrics

| Metric              | Description                      | Threshold |
| ------------------- | -------------------------------- | --------- |
| Parity Rate         | % games with TS/Python agreement | >99.9%    |
| Structural Errors   | Malformed game records           | 0         |
| Semantic Divergence | Logic differences                | 0         |

### Data Quality Metrics

| Metric           | Description                 | Threshold |
| ---------------- | --------------------------- | --------- |
| Completeness     | Games with proper end state | >99%      |
| Move Count       | Avg moves per game          | 50-200    |
| Win Distribution | P1/P2 win balance           | 40-60%    |
| Draw Rate        | Games ending in draw        | <10%      |

### Training Quality Metrics

| Metric            | Description          | Warning | Critical |
| ----------------- | -------------------- | ------- | -------- |
| Holdout Gap       | Train - holdout loss | 0.10    | 0.15     |
| Value Calibration | Predicted vs actual  | 0.15    | 0.25     |
| Policy Accuracy   | Top-1 move accuracy  | <30%    | <20%     |

## Workflows

### Pre-Training Validation

Before starting any training run:

```bash
# 1. Run preflight checks
python scripts/training_preflight_check.py --full

# 2. Validate data parity
python scripts/check_ts_python_replay_parity.py --canonical --input data/selfplay/

# 3. Build canonical pool (quarantines bad data)
python scripts/build_canonical_training_pool_db.py --output data/games/canonical.db

# 4. Verify holdout separation
python scripts/holdout_validation.py --stats
```

### Post-Training Validation

After training completes:

```bash
# 1. Evaluate on holdout
python scripts/holdout_validation.py --evaluate --model models/nnue/new_model.pt

# 2. Check overfitting
python scripts/holdout_validation.py --check-overfitting

# 3. Run parity validation on model predictions
python scripts/check_ts_python_replay_parity.py --model models/nnue/new_model.pt
```

### Continuous Validation

Set up continuous validation in cron:

```bash
# Every 4 hours: validate new selfplay data
0 */4 * * * cd ~/ringrift/ai-service && python scripts/check_ts_python_replay_parity.py --canonical --input data/selfplay/new/ >> /tmp/parity.log 2>&1

# Daily: rebuild canonical pool
0 2 * * * cd ~/ringrift/ai-service && python scripts/build_canonical_training_pool_db.py --output data/games/canonical.db >> /tmp/canonical.log 2>&1
```

## Quarantine System

Games that fail validation are quarantined for investigation.

### Quarantine Location

```
data/quarantine/
├── parity_failures/     # TS/Python divergence
├── structural_errors/   # Malformed records
├── incomplete_games/    # Missing end state
└── metadata.json        # Quarantine summary
```

### Investigating Quarantined Games

```bash
# View quarantine summary
python scripts/analyze_game_statistics.py --quarantine

# Debug specific game with per-step trace (use --db for SQLite, --json for JSON files)
python scripts/check_ts_python_replay_parity.py --db data/quarantine/games.db --trace-game game_12345

# For JSON fixture files
python scripts/check_ts_python_replay_parity.py --json data/quarantine/parity_failures/game_12345.json

# Re-validate database after fix
python scripts/check_ts_python_replay_parity.py --db data/quarantine/parity_failures/games.db
```

## Configuration

### Validation Thresholds

Configure in `config/unified_loop.yaml`:

```yaml
validation:
  parity_threshold: 0.999 # 99.9% parity required
  min_samples: 10000 # Minimum training samples
  holdout_gap_warning: 0.10 # Overfitting warning threshold
  holdout_gap_critical: 0.15 # Overfitting critical threshold
  max_quarantine_rate: 0.01 # Max 1% quarantine rate
```

### Feature Validation

```yaml
features:
  expected_shape:
    square8_2p: [64, 18] # 8x8 board, 18 feature planes
    hex8_2p: [81, 18] # 9x9 embedded, 18 feature planes
  value_range: [-1, 1] # Expected value target range
  policy_sum: 1.0 # Policy should sum to 1
```

## Troubleshooting

### High Quarantine Rate

If >1% of games are quarantined:

1. Check recent code changes to game engine
2. Review quarantine metadata for patterns
3. Run targeted parity tests on failing configs
4. Consider rolling back to known-good version

### Parity Failures

When TS/Python produce different results:

1. Identify the divergence point (move number)
2. Extract the game state at that point
3. Step through both engines with debugging
4. Check for floating-point or ordering differences

### Overfitting Detected

When holdout gap exceeds threshold:

1. Reduce training epochs
2. Increase dropout/regularization
3. Check for data leakage (holdout in training)
4. Consider more diverse training data
