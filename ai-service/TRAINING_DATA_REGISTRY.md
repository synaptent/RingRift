# Training Data Registry

This document tracks the provenance and canonical status of all self-play databases and neural network models in the RingRift AI training pipeline.

## Data Classification

| Status                  | Meaning                                                                                                     |
| ----------------------- | ----------------------------------------------------------------------------------------------------------- |
| **canonical**           | Generated and gated via `scripts/generate_canonical_selfplay.py` (parity gate **and** canonical history OK) |
| **legacy_noncanonical** | Pre-dates 7-phase/FE/canonical-history fixes; **DO NOT** use for new training                               |
| **pending_gate**        | Not yet validated; requires `generate_canonical_selfplay.py` (or equivalent gate) before any training use   |

---

## Game Replay Databases

### Canonical (Parity + Canonical-History Gated)

| Database                | Board Type | Players | Status                | Gate Summary                           | Notes                                                                                                                                                                                                                                                                           |
| ----------------------- | ---------- | ------- | --------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `canonical_square8.db`  | square8    | 2       | **canonical**         | canonical_square8.db.parity_gate.json  | Parity + canonical phase history passed (20 games, no divergences). See `data/games/canonical_square8.db.parity_gate.json`.                                                                                                                                                     |
| `canonical_square19.db` | square19   | 2       | **canonical**         | canonical_square19.db.parity_gate.json | Parity + canonical phase history passed (32 games, no divergences). See `data/games/canonical_square19.db.parity_gate.json`.                                                                                                                                                    |
| `canonical_hex.db`      | hexagonal  | 2       | ⚠️ **DEPRECATED_R10** | N/A                                    | **DEPRECATED 2025-12-06:** Generated with old hex geometry (radius 10, 331 cells, 36 rings). Incompatible with current hex spec (radius 12, 469 cells, 48 rings). **File removed from repo.** See `data/HEX_DATA_DEPRECATION_NOTICE.md` and `docs/HEX_ARTIFACTS_DEPRECATED.md`. |
| `golden.db`             | mixed      | mixed   | **canonical**         | N/A                                    | Hand-curated golden games (small fixed set, inspected manually)                                                                                                                                                                                                                 |

### Legacy / Non-Canonical

These databases were generated **before** the following fixes were applied:

- Line-length validation fix (RR-CANON-R120)
- Explicit line decision flow (RR-CANON-R121-R122)
- All turn actions/skips must be explicit (RR-CANON-R074)
- Forced elimination checks for correct player rotation

**DO NOT use these for new training runs.** They are retained for historical comparison only.

| Database                    | Board Type | Players | Status                  | Notes                               |
| --------------------------- | ---------- | ------- | ----------------------- | ----------------------------------- |
| `selfplay_square8_2p.db`    | square8    | 2       | **legacy_noncanonical** | Pre-parity-fix self-play            |
| `selfplay_square19_2p.db`   | square19   | 2       | **legacy_noncanonical** | Pre-parity-fix self-play            |
| `selfplay_square19_3p.db`   | square19   | 3       | **legacy_noncanonical** | Pre-parity-fix self-play            |
| `selfplay_square19_4p.db`   | square19   | 4       | **legacy_noncanonical** | Pre-parity-fix self-play            |
| `selfplay_hexagonal_2p.db`  | hexagonal  | 2       | ⚠️ **DEPRECATED_R10**   | Radius 10 geometry + Pre-parity-fix |
| `selfplay_hexagonal_3p.db`  | hexagonal  | 3       | ⚠️ **DEPRECATED_R10**   | Radius 10 geometry + Pre-parity-fix |
| `selfplay_hexagonal_4p.db`  | hexagonal  | 4       | ⚠️ **DEPRECATED_R10**   | Radius 10 geometry + Pre-parity-fix |
| `selfplay.db`               | mixed      | mixed   | **legacy_noncanonical** | Ad-hoc testing DB                   |
| `square8_2p.db`             | square8    | 2       | **legacy_noncanonical** | Early development DB                |
| `minimal_test.db`           | mixed      | mixed   | **legacy_noncanonical** | Test fixture DB                     |
| `golden_hexagonal.db`       | hexagonal  | 2       | ⚠️ **DEPRECATED_R10**   | Radius 10 geometry (golden)         |
| `selfplay_hex_mps_smoke.db` | hexagonal  | 2       | ⚠️ **DEPRECATED_R10**   | Radius 10 geometry + MPS smoke      |

---

## Neural Network Models

### Model Provenance Table

| Model File                                | Training Data               | Status                  | Notes                                                                                                                                   |
| ----------------------------------------- | --------------------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `ringrift_v1.pth`                         | Legacy selfplay DBs         | **legacy_noncanonical** | Main model, needs retraining on canonical data                                                                                          |
| `ringrift_v1.pth.legacy`                  | Early selfplay DBs          | **legacy_noncanonical** | Original v1, historical only                                                                                                            |
| `ringrift_v1_legacy_nested.pth`           | Legacy nested replay export | **legacy_noncanonical** | Do not use                                                                                                                              |
| `ringrift_v1_mps.pth`                     | Legacy selfplay DBs         | **legacy_noncanonical** | MPS variant, needs retraining                                                                                                           |
| `ringrift_v1_mps_2025*.pth`               | Legacy selfplay DBs         | **legacy_noncanonical** | MPS checkpoints                                                                                                                         |
| `ringrift_v1_2025*.pth`                   | Legacy selfplay DBs         | **legacy_noncanonical** | v1 checkpoints from Nov 2025                                                                                                            |
| `ringrift_from_replays_square8.pth`       | Mixed replay DBs            | **legacy_noncanonical** | Trained from legacy replays                                                                                                             |
| `ringrift_from_replays_square8_2025*.pth` | Mixed replay DBs            | **legacy_noncanonical** | Checkpoints from legacy replays                                                                                                         |
| `ringrift_v1_hex*.pth`                    | Legacy hex DBs (radius 10)  | ⚠️ **DEPRECATED_R10**   | Old hex geometry (331 cells, 36 rings, 21×21 input). Do not load; retrain on new radius-12 geometry (469 cells, 48 rings, 25×25 input). |

### Target Canonical Models

Once canonical self-play DBs are generated and exported, retrain these models:

| Target Model               | Training Data Source       | Status                                              |
| -------------------------- | -------------------------- | --------------------------------------------------- |
| `ringrift_v2_square8.pth`  | canonical_square8.db       | Pending                                             |
| `ringrift_v2_square19.pth` | canonical_square19.db      | Pending                                             |
| `ringrift_v2_hex.pth`      | canonical_hex.db (removed) | Pending — regenerate a radius-12 hex DB and retrain |

---

## Training Data Allowlist Policy

1. **All new training runs** must ONLY use databases listed as `canonical` in this registry.

2. **To add a new DB to the canonical allowlist:**

   Use the unified canonical generator + gate:

   ```bash
   cd ai-service
   PYTHONPATH=. python scripts/generate_canonical_selfplay.py \
     --board-type <board> \
     --num-games 32 \
     --db data/games/canonical_<board>.db \
     --summary db_health.canonical_<board>.json
   ```

   A DB is eligible for `Status = canonical` **only if**:
   - `canonical_ok` in the generated summary is `true`, and
   - `parity_gate.passed_canonical_parity_gate` is `true`, and
   - `canonical_history.non_canonical_games == 0`.

   Older flows that called `run_canonical_selfplay_parity_gate.py` directly are
   considered **v1 gating** and SHOULD be migrated to the new script the next
   time those DBs are regenerated.

3. **Legacy DBs** may be used for:
   - Historical comparison experiments
   - Ablation studies (comparing legacy vs canonical)
   - Debugging parity issues

   They must NOT be used for:
   - Training official v2+ models
   - Evaluation baselines
   - Curriculum learning seeds

4. **Model version tracking:**
   - All new models must use the `ModelVersionManager` from `app/training/model_versioning.py`
   - Checkpoints include architecture version, config, and SHA256 checksum
   - Version mismatch errors are thrown explicitly (no silent fallback)

---

## Cleanup Recommendations

### Move to `data/games/legacy/`:

- `selfplay_square8_2p.db`
- `selfplay_square19_*.db`
- `selfplay_hexagonal_*.db`
- `selfplay.db`
- `square8_2p.db`
- `minimal_test.db`
- `selfplay_hex_mps_smoke.db`

### Move to `models/legacy/`:

- `ringrift_v1.pth.legacy`
- `ringrift_v1_legacy_nested.pth`
- `ringrift_v1_2025*.pth`
- `ringrift_from_replays_square8*.pth`

### Keep in place (but retrain when canonical data ready):

- `ringrift_v1.pth` (current production model)
- `ringrift_v1_mps.pth` (MPS production model)

---

## Parity Gate Results

When `run_canonical_selfplay_parity_gate.py` is used to generate and gate a new DB,
its summary should be stored alongside this document (for example as
`parity_gate.<board>.json`). The lower-level parity sweeps invoked directly by
`check_ts_python_replay_parity.py` can also be captured as
`parity_summary.<label>.json` for ad-hoc debugging (for example,
`parity_summary.canonical_square8.json`).

Each gate summary contains:

```json
{
  "board_type": "...",
  "db_path": "...",
  "num_games": N,
  "parity_summary": {
    "games_checked": N,
    "games_with_structural_issues": 0,
    "games_with_semantic_divergence": 0
  },
  "passed_canonical_parity_gate": true
}
```

For DBs that **fail** the gate (structural issues or semantic divergence), it is
often useful to also persist:

- Parity fixtures:

  ```bash
  cd ai-service
  PYTHONPATH=. python scripts/check_ts_python_replay_parity.py \
    --db data/games/<db>.db \
    --emit-fixtures-dir parity_fixtures
  ```

  which writes one compact JSON fixture per divergent `(db, game_id)` under
  `parity_fixtures/`.

- Rich state bundles:

  ```bash
  cd ai-service
  PYTHONPATH=. python scripts/check_ts_python_replay_parity.py \
    --db data/games/<db>.db \
    --emit-state-bundles-dir parity_fixtures/state_bundles
  ```

  which write `.state_bundle.json` files containing full TS and Python
  `GameState` JSON immediately before and at the first divergent step for each
  game. These can be inspected with:

  ```bash
  PYTHONPATH=. python scripts/diff_state_bundle.py \
    --bundle parity_fixtures/state_bundles/<bundle>.state_bundle.json
  ```

Capturing these artefacts alongside `parity_gate.*.json` and
`parity_summary.*.json` makes it much faster to diagnose and fix any remaining
parity issues before promoting a DB to the canonical training allowlist.

---

_Last updated: 2025-12-05_
