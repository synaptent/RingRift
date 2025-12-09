# Training Data Registry

This document tracks the provenance and canonical status of all self-play databases and neural network models in the RingRift AI training pipeline.

## Data Classification

| Status                  | Meaning                                                                                                                                                                                                                               |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **canonical**           | Generated and gated via `scripts/generate_canonical_selfplay.py` with `canonical_ok == true` in the gate summary JSON (for supported boards this also implies FE/territory fixtures and canonical history/parity gate have all passed) |
| **legacy_noncanonical** | Pre-dates 7-phase/FE/canonical-history fixes; **DO NOT** use for new training                                                                                                                                                         |
| **pending_gate**        | Not yet validated; requires `generate_canonical_selfplay.py` (or equivalent gate) before any training use                                                                                                                             |

---

## Game Replay Databases

### Canonical (Parity + Canonical-History Gated)

| Database | Board Type | Players | Status | Gate Summary | Notes                                                                        |
| -------- | ---------- | ------- | ------ | ------------ | ---------------------------------------------------------------------------- |
| _None_   | –          | –       | –      | –            | Square8 regeneration is pending (see below); rerun the gate before training. |

The `Status` column uses `canonical` only for DBs whose latest gate summary JSON has `canonical_ok == true`. For supported board types (currently `square8` and `hexagonal`), this also implies `fe_territory_fixtures_ok == true` as well as a passing parity gate and canonical phase history.

### Pending Re-Gate / Needs Regeneration

| Database               | Board Type | Players | Status           | Gate Summary                     | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ---------------------- | ---------- | ------- | ---------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `canonical_square8.db` | square8    | 2       | **pending_gate** | db_health.canonical_square8.json | 2025-12-09 local run of `generate_canonical_selfplay.py --board-type square8 --num-games 0 --num-players 2 --db data/games/canonical_square8.db --summary data/games/db_health.canonical_square8.json` re-gated the existing DB, which now contains 14 square8 2p `selfplay_soak` games. Canonical phase-history checks pass (`check_canonical_phase_history.py` exited 0 for all games), but the TS replay parity harness reports `games_with_structural_issues = 14` and `games_with_semantic_divergence = 0` (all `ts-replay-error` cases with `[PHASE_MOVE_INVARIANT] Cannot apply move type 'place_ring' in phase 'territory_processing'` when attempting to replay the second player's ring placement after `territory_processing`). `canonical_ok=false` and `parity_gate.passed_canonical_parity_gate=false`; treat this DB as non-canonical training data pending a follow-up parity task to resolve the TS↔Python replay mismatch. |

### Legacy / Non-Canonical

These databases were generated **before** the following fixes were applied:

- Line-length validation fix (RR-CANON-R120)
- Explicit line decision flow (RR-CANON-R121-R122)
- All turn actions/skips must be explicit (RR-CANON-R074)
- Forced elimination checks for correct player rotation

**DO NOT use these for new training runs.** They are retained for historical comparison only.

_None retained._ All legacy/non-canonical DBs were deleted as part of the 2025-12-08 cleanup.

---

### Gate Notes (2025-12-07)

- Replayed `canonical_square8.db` and `canonical_square8_2p.db` via `check_canonical_phase_history.py`; both hit late-turn `Not your turn` failures (matching the parity gate structural errors). Treat them as pending re-generation.
- `canonical_square19.db` currently has zero games despite an older parity gate artifact; regenerate and re-gate before use.
- Hex assets remain deprecated until a radius-12 canonical DB is generated. Only radius-12 hex DBs whose gate summary JSON reports `canonical_ok == true` and `fe_territory_fixtures_ok == true` **and** that are listed as `canonical` in this registry should be used for training hex models. Older radius-10/legacy hex DBs remain permanently non-canonical; see `docs/HEX_PARITY_AUDIT.md` and `ai-service/data/HEX_DATA_DEPRECATION_NOTICE.md` for deprecation and parity-audit context.
- 2025-12-08 sandbox attempt to regenerate `canonical_square8.db` via `generate_canonical_selfplay.py` failed before the parity gate due to OpenMP shared-memory permissions (`OMP: Error #179: Function Can't open SHM2 failed`). The resulting `db_health.canonical_square8.json` has `canonical_ok=false` and an empty `canonical_history` block. Re-run the generator on a host with SHM permissions, then replace the DB and summaries.
- 2025-12-09 local re-gate of `canonical_square8.db` via `generate_canonical_selfplay.py --board-type square8 --num-games 0 --num-players 2 --db data/games/canonical_square8.db --summary data/games/db_health.canonical_square8.json` (no new games requested) confirmed that the DB now contains 14 square8 2p `selfplay_soak` games and that canonical phase-history checks pass, but TS replay parity still reports 14 structural issues (all `[PHASE_MOVE_INVARIANT] Cannot apply move type 'place_ring' in phase 'territory_processing'` during second-player ring placements). Leave `canonical_square8.db` in **pending_gate** until this TS↔Python replay mismatch is resolved in a dedicated parity task.

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
| `ringrift_v1_hex*.pth`                    | Legacy hex DBs (radius 10)  | ⚠️ **DEPRECATED_R10**   | Old hex geometry (331 cells, 36 rings, 21×21 input). Do not load; retrain on new radius-12 geometry (469 cells, 72 rings, 25×25 input). |

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

   A DB is eligible for `Status = canonical` **only if** its latest gate summary JSON (written by `scripts/generate_canonical_selfplay.py`) reports:

   - `canonical_ok == true` (the top-level allowlist flag), and
   - for supported board types (currently `square8` and `hexagonal`), this in turn implies:
     - `parity_gate.passed_canonical_parity_gate == true`, and
     - `canonical_history.games_checked > 0` and `canonical_history.non_canonical_games == 0`, and
     - `fe_territory_fixtures_ok == true`.

   For other board types, `fe_territory_fixtures_ok` may be `true` by construction until dedicated FE/territory fixtures are added; once such fixtures exist, `canonical_ok` will also encode their success.

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

The legacy and pending DBs listed earlier have been removed. For any new replay DBs, place them under `data/games/canonical_<board>.db` only after passing the gate. Legacy models should still be parked under `models/legacy/` as noted in the Model Provenance table above; regenerate v2 models once new canonical data exists.

---

## Parity Gate Results

When `run_canonical_selfplay_parity_gate.py` is used to generate and gate a new DB,
its summary should be stored alongside this document (for example as
`parity_gate.<board>.json`). The lower-level parity sweeps invoked directly by
`check_ts_python_replay_parity.py` can also be captured as
`parity_summary.<label>.json` for ad-hoc debugging (for example,
`parity_summary.canonical_square8.json`).

The example below shows the inner `parity_summary` structure used both in historical `parity_gate.*.json` artefacts and inside the newer canonical gate summaries emitted by `scripts/generate_canonical_selfplay.py` (under the `parity_gate` key alongside `canonical_history`, `fe_territory_fixtures_ok`, and `canonical_ok`).

Each parity summary contains:

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

_Last updated: 2025-12-09_
