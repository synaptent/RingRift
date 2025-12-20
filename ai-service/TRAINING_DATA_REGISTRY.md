# Training Data Registry

This document tracks the provenance and canonical status of all self-play databases and neural network models in the RingRift AI training pipeline.

## Data Classification

| Status                  | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **canonical**           | Training-approved canonical rules data. A DB is canonical when every game it contains passes canonical config + canonical phase-history validation and TS↔Python replay parity (no structural issues, no semantic divergence). Canonical DBs may be regenerated with `scripts/generate_canonical_selfplay.py` or incrementally extended by ingesting other sources via `scripts/build_canonical_training_pool_db.py` (per-game fail-closed gates; tournament/eval treated as holdout). |
| **legacy_noncanonical** | Pre-dates 7-phase/FE/canonical-history fixes; **DO NOT** use for new training                                                                                                                                                                                                                                                                                                                                                                                                          |
| **pending_gate**        | Not yet validated; requires `generate_canonical_selfplay.py` (or equivalent gate) before any training use                                                                                                                                                                                                                                                                                                                                                                              |

---

## Game Replay Databases

### Canonical (Parity + Canonical-History Gated)

| Database                  | Board Type | Players | Status        | Gate Summary                                      | Notes                                                                                                                                                                                 |
| ------------------------- | ---------- | ------- | ------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `canonical_square8_2p.db` | square8    | 2       | **canonical** | canonical_square8_2p_200games.parity_summary.json | 2025-12-16 Vast.ai regeneration (200 games, 12,642 samples); 100% semantic parity verified; NPZ exported to `data/training/canonical_square8_2p.npz`                                  |
| `canonical_square8.db`    | square8    | 2       | **canonical** | db_health.canonical_square8.json                  | 2025-12-12 distributed regeneration (12 games) and re-gate after TS territory-control parity fix; `canonical_ok=true` (only end-of-game-only current_player mismatch).                |
| `canonical_square8_3p.db` | square8    | 3       | **canonical** | db_health.canonical_square8_3p.json               | 2025-12-12 initial 3P canonical DB (2 games) gated successfully (`canonical_ok=true`; parity only end-of-game-only current_player mismatch).                                          |
| `canonical_square8_4p.db` | square8    | 4       | **canonical** | db_health.canonical_square8_4p.json               | 2025-12-12 4P canonical DB (2 games) gated successfully (`canonical_ok=true`). Scale up for training.                                                                                 |
| `canonical_square19.db`   | square19   | 2       | **canonical** | db_health.canonical_square19.json                 | 2025-12-20 regenerated via direct soak (3 games, 1,903 moves) with `RINGRIFT_USE_MAKE_UNMAKE=true` (light band). Parity + canonical history gates passed; still below volume targets. |
| `canonical_hexagonal.db`  | hexagonal  | 2       | **canonical** | db_health.canonical_hexagonal.json                | 2025-12-20 regenerated via direct soak (1 game, 1,113 moves) with `RINGRIFT_USE_MAKE_UNMAKE=true` (light band). Parity + canonical history gates passed; still below volume targets.  |

The `Status` column uses `canonical` only for DBs whose latest gate summary JSON has `canonical_ok == true`. For supported board types (`square8`, `square19`, and `hexagonal`), this also implies `fe_territory_fixtures_ok == true` as well as a passing parity gate and canonical phase history.

### Volume Targets (Provisional)

These targets define when large-board datasets are considered ready for training and should be revisited after throughput profiling.

| Board Type | Baseline Gate Target | Training Target | Notes                                                                          |
| ---------- | -------------------- | --------------- | ------------------------------------------------------------------------------ |
| square19   | >=200 games          | >=1000 games    | Align with canonical_square8_2p baseline; scale once throughput is acceptable. |
| hexagonal  | >=200 games          | >=1000 games    | Same target as square19; adjust after large-board performance runs.            |

### Pending Re-Gate / Needs Regeneration

None at the moment (large-board DBs are canonical but still low-volume).

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

- 2025-12-16 Vast.ai regeneration: `canonical_square8_2p.db` generated with 200 games (12,642 training samples) via distributed self-play on Vast.ai instance. 100% semantic parity verified (`games_with_semantic_divergence: 0`, `passed_canonical_parity_gate: true`). NPZ exported to `data/training/canonical_square8_2p.npz` with board-aware encoding. This is the primary 2-player square8 canonical training source.
- 2025-12-12 distributed regeneration + parity fix: `canonical_square8.db` now passes the canonical gate (`canonical_ok=true`, parity only end-of-game-only mismatches). It is safe for new training.
- 2025-12-12: `canonical_square8_3p.db` has initial gated games; scale up before training.
- 2025-12-12: `canonical_square8_4p.db` is now canonical (`canonical_ok=true`); scale up before training.
- 2025-12-20: `canonical_square19.db` re-gated with direct soak (light band) and `RINGRIFT_USE_MAKE_UNMAKE=true`; parity + history gates passed (`canonical_ok=true`). Scale volume toward targets.
- 2025-12-20: `canonical_hexagonal.db` re-gated with direct soak (light band) and `RINGRIFT_USE_MAKE_UNMAKE=true`; parity + history gates passed (`canonical_ok=true`). Scale volume toward targets.
- Historical: the sandboxed environment can fail OpenMP shared-memory allocation (`OMP: Error #179: Function Can't open SHM2 failed`); run canonical self-play on a host/container with SHM permissions.
- (Historical) 2025-12-09 re-gate of `canonical_square8.db` found TS replay structural errors; resolved on 2025-12-12 by aligning TS territory-control victory to collapsed-territory counts.

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
| `ringrift_v1_hex*.pth`                    | Legacy hex DBs (radius 10)  | ⚠️ **DEPRECATED_R10**   | Old hex geometry (331 cells, 36 rings, 21×21 input). Do not load; retrain on new radius-12 geometry (469 cells, 96 rings, 25×25 input). |

### Target Canonical Models

Once canonical self-play DBs are generated and exported, retrain these models:

| Target Model               | Training Data Source   | Status  |
| -------------------------- | ---------------------- | ------- |
| `ringrift_v2_square8.pth`  | canonical_square8.db   | Pending |
| `ringrift_v2_square19.pth` | canonical_square19.db  | Pending |
| `ringrift_v2_hex.pth`      | canonical_hexagonal.db | Pending |

---

## Training Data Allowlist Policy

1. **All new training runs** must ONLY use databases listed as `canonical` in this registry.

2. **To add a new DB to the canonical allowlist (regenerate-from-scratch path):**

   Use the unified canonical generator + gate:

   ```bash
   cd ai-service
   PYTHONPATH=. python scripts/generate_canonical_selfplay.py \
     --board-type <board> \
     --num-games 32 \
     --db data/games/canonical_<board>.db \
     --summary db_health.canonical_<board>.json
   ```

   For distributed self-play across SSH hosts, pass `--hosts host1,host2,...`.
   Per-host DBs are merged into the destination DB (deduped by `game_id`).
   Use `--reset-db` to archive any existing destination DB before generating/merging.

   A DB is eligible for `Status = canonical` **only if** its latest gate summary JSON (written by `scripts/generate_canonical_selfplay.py`) reports:
   - `canonical_ok == true` (the top-level allowlist flag), and
   - for supported board types (`square8`, `square19`, and `hexagonal`), this in turn implies:
     - `parity_gate.passed_canonical_parity_gate == true`, and
     - `canonical_history.games_checked > 0` and `canonical_history.non_canonical_games == 0`, and
     - `fe_territory_fixtures_ok == true`, and
     - `anm_ok == true`, where ANM parity + invariants are enforced via:
       - `tests/parity/test_anm_global_actions_parity.py`
       - `tests/invariants/test_anm_and_termination_invariants.py`

   The gate summary also includes `db_stats` (informational only) with lightweight totals (games/moves) and `swap_sides` counts to help spot low-volume DBs and baseline coverage quickly.

   For other board types, `fe_territory_fixtures_ok` and `anm_ok` may be `true` by construction until dedicated FE/territory/ANM fixtures are added; once such fixtures exist, `canonical_ok` will also encode their success.

   Older flows that called `run_canonical_selfplay_parity_gate.py` directly are
   considered **v1 gating** and SHOULD be migrated to the new script the next
   time those DBs are regenerated.

3. **To incorporate additional self-play sources into canonical training data (incremental ingestion path):**

   Any generator (CMA-ES, tournaments, soaks, hybrid self-play, etc.) may write
   games into a **staging** `GameReplayDB`, as long as those games are rules‑correct.
   Use the strict per-game ingestion gate to merge only canonical games into a
   canonical training DB and quarantine anything that fails:

   ```bash
   cd ai-service
   PYTHONPATH=. python scripts/build_canonical_training_pool_db.py \
     --input-db data/games/staging/<source>.db \
     --output-db data/games/canonical_<board>.db \
     --board-type <board> \
     --num-players <N> \
     --require-completed \
     --holdout-db data/games/holdouts/holdout_<board>_<N>p.db \
     --quarantine-db data/games/quarantine/quarantine_<board>_<N>p.db \
     --report-json logs/ingest_reports/ingest_<board>_<N>p_<ts>.json
   ```

   Policy:
   - **Tournament/evaluation games are holdout**: they are excluded from training and may be copied into `--holdout-db`.
   - **Per-game gates are fail-closed**: any game failing canonical config/history or TS↔Python parity is excluded and may be copied into `--quarantine-db`.

4. **Legacy DBs** may be used for:
   - Historical comparison experiments
   - Ablation studies (comparing legacy vs canonical)
   - Debugging parity issues

   They must NOT be used for:
   - Training official v2+ models
   - Evaluation baselines
   - Curriculum learning seeds

5. **Model version tracking:**
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

## Elo Tournament and Holdout Data

Tournament and evaluation games are explicitly **holdout data** and are excluded from training. This section documents the Elo calibration tournament infrastructure.

### Elo Leaderboard Database

| Database             | Location                  | Purpose                                                         |
| -------------------- | ------------------------- | --------------------------------------------------------------- |
| `elo_leaderboard.db` | `data/elo_leaderboard.db` | Stores model Elo ratings, match history, and rating progression |

Schema tables:

- `models`: Registered models with paths, board types, and versions
- `elo_ratings`: Current Elo ratings per model/config (rating, games_played, wins, losses, draws)
- `match_history`: Individual match records (model_a, model_b, winner, game_length, timestamp)
- `rating_history`: Historical rating snapshots for tracking progression

### Tournament Orchestration

Elo calibration tournaments can be triggered via:

1. **Standalone script** (`scripts/run_model_elo_tournament.py`):

   ```bash
   python scripts/run_model_elo_tournament.py --board square8 --players 2 --games 100 --run
   python scripts/run_model_elo_tournament.py --all-configs --games 50 --run
   ```

2. **Pipeline integration** (`scripts/unified_ai_loop.py`):
   - Shadow tournaments run every 5 minutes across all board/player configs
   - Full tournaments run hourly for comprehensive Elo calibration
   - _Note: `pipeline_orchestrator.py` is deprecated; use unified_ai_loop.py_

3. **Improvement loop integration** (`scripts/run_improvement_loop.py`):
   - `--tournament-every-n-iterations N`: Run tournaments every N iterations
   - `--tournament-on-promotion`: Run tournaments after each model promotion
   - `--tournament-board-types`: Comma-separated board types (or "all")
   - `--tournament-player-counts`: Comma-separated player counts (or "all")

### Holdout Data Policy

Tournament games are written to holdout DBs and are NOT ingested into canonical training pools:

| Holdout DB                        | Board Type | Players | Purpose                    |
| --------------------------------- | ---------- | ------- | -------------------------- |
| `holdouts/holdout_square8_2p.db`  | square8    | 2       | Eval/tournament games only |
| `holdouts/holdout_square8_3p.db`  | square8    | 3       | Eval/tournament games only |
| `holdouts/holdout_square8_4p.db`  | square8    | 4       | Eval/tournament games only |
| `holdouts/holdout_square19_2p.db` | square19   | 2       | Eval/tournament games only |
| `holdouts/holdout_hex_2p.db`      | hexagonal  | 2       | Eval/tournament games only |

This separation ensures:

- Training data remains uncontaminated by evaluation games
- Elo ratings reflect true model strength (no training on test data)
- Holdout games can be used for independent validation

---

---

## 2025-12-20 Parity Fixes Applied

Two critical parity bugs were fixed:

1. **Phase Transition Bug (commit b8175468)**
   - Root cause: `no_territory_action` case in `applyMoveWithChainInfo` did not advance phase/player
   - Fix: Handle phase transition inline (check forced_elimination, then rotate to next player)
   - All data generated pre-fix is non-canonical

2. **FORCED_ELIMINATION Parity Bug**
   - Root cause: GPU selfplay scripts missing FORCED_ELIMINATION phase handling
   - Fix: Updated `run_gpu_selfplay.py` and `import_gpu_selfplay_to_db.py`
   - Synced to all 8 Lambda cluster nodes

**Impact:**

- All existing DBs (jsonl*converted*\*, staging/improvement_loop) generated before these fixes are non-canonical
- New selfplay data (post-fix) should pass parity gates
- Legacy DBs with v1 schema (missing `game_moves` table) cannot be parity-checked

**Verification:** ✅ COMPLETE (2025-12-20)

Parity check on fresh canonical data confirms all fixes working:

```bash
# Results: passed_canonical_parity_gate: true
# 10/10 games: 0 semantic divergence, 0 structural issues
PYTHONPATH=. python scripts/check_ts_python_replay_parity.py \
  --db data/games/parity_test/selfplay.db --compact
```

Test details:

- Board: square8, 2-player
- Engine: heuristic-only (canonical difficulty band)
- Games: 10
- Semantic divergence: 0
- Structural issues: 0
- End-of-game only divergence: 1 (minor current_player mismatch at game_over, acceptable)

**Scale Verification (2025-12-20):**

Additional parity verification on improvement loop data confirms fixes work at scale:

- `selfplay_iter84_square8_2p.db`: 190 games, 50 sampled → **0 semantic divergence**
- `selfplay_iter86_square8_2p.db`: 207 games (in progress)
- All games generated with `--difficulty-band canonical --fail-on-anomaly`
- Gate summary: `passed_canonical_parity_gate: true`

The improvement loop generates canonical-quality data and can be ingested via
`scripts/build_canonical_training_pool_db.py`.

_Last updated: 2025-12-20_
