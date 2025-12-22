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

| Database                  | Board Type | Players | Status        | Gate Summary                        | Notes                                                                                                                                                                                         |
| ------------------------- | ---------- | ------- | ------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `canonical_square8_2p.db` | square8    | 2       | **canonical** | db_health.canonical_square8_2p.json | 2025-12-16 Vast.ai regeneration (200 games, 12,642 samples); 100% semantic parity verified; NPZ exported to `data/training/canonical_square8_2p.npz`                                          |
| `canonical_square8.db`    | square8    | 2       | **canonical** | db_health.canonical_square8.json    | 2025-12-12 distributed regeneration (12 games) and re-gate after TS territory-control parity fix; `canonical_ok=true` (only end-of-game-only current_player mismatch). Small parity smoke DB. |
| `canonical_square8_3p.db` | square8    | 3       | **canonical** | db_health.canonical_square8_3p.json | 2025-12-12 initial 3P canonical DB (2 games) gated successfully (`canonical_ok=true`; parity only end-of-game-only current_player mismatch).                                                  |
| `canonical_square8_4p.db` | square8    | 4       | **canonical** | db_health.canonical_square8_4p.json | 2025-12-12 4P canonical DB (2 games) gated successfully (`canonical_ok=true`). Scale up for training.                                                                                         |
| `canonical_square19.db`   | square19   | 2       | **canonical** | db_health.canonical_square19.json   | 2025-12-20 regenerated via direct soak (3 games, 1,903 moves) with `RINGRIFT_USE_MAKE_UNMAKE=true` (light band). Parity + canonical history gates passed; still below volume targets.         |

The `Status` column uses `canonical` only for DBs whose latest gate summary JSON has `canonical_ok == true`. For supported board types (`square8`, `square19`, and `hexagonal`), this also implies `fe_territory_fixtures_ok == true` as well as a passing parity gate and canonical phase history.

### Coverage Matrix (2025-12-21)

Target: All 12 combinations (4 board types × 3 player counts) with canonical training data.

| Board     | 2P                      | 3P           | 4P           |
| --------- | ----------------------- | ------------ | ------------ |
| square8   | ✅ canonical (200+)     | ⚠️ small (2) | ⚠️ small (2) |
| square19  | ⚠️ small (3)            | ❌ missing   | ❌ missing   |
| hex8      | ❌ missing              | ❌ missing   | ❌ missing   |
| hexagonal | ❌ parity-blocked (300) | ❌ missing   | ❌ missing   |

Legend:

- ✅ = Canonical, sufficient volume (>=200 games)
- ⚠️ = Canonical/partial but insufficient volume (<200 games)
- ❌ = Not generated yet

**Priority Actions (2025-12-21):**

1. Scale up square8 3P/4P to 200+ games each
2. Scale up square19 2P to 200+ games
3. Generate square19 3P/4P databases
4. Generate hex8 2P/3P/4P databases (new board type)
5. ~~Fix hexagonal parity bug HEX-PARITY-01~~ FIXED (phase_machine.py:138) - new remaining issues (ANM state divergence)

**Generation Commands:**

```bash
# Scale up square8 3P/4P
python scripts/generate_canonical_selfplay.py --board square8 --num-players 3 --num-games 200 --min-recorded-games 200
python scripts/generate_canonical_selfplay.py --board square8 --num-players 4 --num-games 200 --min-recorded-games 200

# Scale up square19 2P and generate 3P/4P
python scripts/generate_canonical_selfplay.py --board square19 --num-players 2 --num-games 200 --min-recorded-games 200
python scripts/generate_canonical_selfplay.py --board square19 --num-players 3 --num-games 200
python scripts/generate_canonical_selfplay.py --board square19 --num-players 4 --num-games 200

# Generate hex8 databases (new board type)
python scripts/generate_canonical_selfplay.py --board hex8 --num-players 2 --num-games 200
python scripts/generate_canonical_selfplay.py --board hex8 --num-players 3 --num-games 200
python scripts/generate_canonical_selfplay.py --board hex8 --num-players 4 --num-games 200

# Hexagonal: Parity bug FIXED (commit 7f43c368) - now unblocked
python scripts/generate_canonical_selfplay.py --board hexagonal --num-players 2 --num-games 200
python scripts/generate_canonical_selfplay.py --board hexagonal --num-players 3 --num-games 200
python scripts/generate_canonical_selfplay.py --board hexagonal --num-players 4 --num-games 200
```

### Volume Targets (Provisional)

These targets define when large-board datasets are considered ready for training and should be revisited after throughput profiling.

| Board Type | Baseline Gate Target | Training Target | Notes                                                                          |
| ---------- | -------------------- | --------------- | ------------------------------------------------------------------------------ |
| square19   | >=200 games          | >=1000 games    | Align with canonical_square8_2p baseline; scale once throughput is acceptable. |
| hexagonal  | >=200 games          | >=1000 games    | Same target as square19; adjust after large-board performance runs.            |

### Pending Re-Gate / Needs Regeneration

| Database                 | Board Type | Players | Status             | Gate Summary                         | Issue                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------------------------ | ---------- | ------- | ------------------ | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `canonical_hexagonal.db` | hexagonal  | 2       | **parity_blocked** | canonical_hexagonal.parity_gate.json | 2025-12-21 fresh regeneration (300 games, random-only AI): **98% semantic divergence rate** (98/100 sampled). Primary issue is ANM state mismatch in `territory_processing` phase - Python reports `is_anm: true` while TS reports `is_anm: false`. State hashes match (game logic correct) but ANM tracking differs. 2 games also have structural issues (legacy phase coercion errors). **HEX-PARITY-02 is NOT fully resolved.** Cannot mark as canonical until ANM parity bug is fixed. |
| `all_jsonl_training.db`  | mixed      | 2,3,4   | **pending_gate**   | N/A                                  | 2025-12-21 aggregated JSONL selfplay data. Contains hex8 (662 2p, 597 3p, 526 4p games), hexagonal, square8, square19. Use with `--allow-pending-gate` flag for initial model training while canonical generation catches up. Generated by `aggregate_jsonl_to_db.py`.                                                                                                                                                                                                                     |

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

- 2025-12-21 Fresh Hexagonal Regeneration (300 games): Regenerated `canonical_hexagonal.db` with
  300 games using `run_self_play_soak.py --engine-mode random-only`. All 300 games recorded
  successfully with 0 invariant violations.

  **Parity Results (100 games sampled):**
  - 0 games pass full parity ❌
  - 98 games fail with semantic divergence (primarily ANM state mismatch in `territory_processing`)
  - 2 games fail with structural issues (legacy phase coercion errors)

  **Primary Divergence Pattern (98 games):** ANM state mismatch in `territory_processing` phase.
  Python reports `is_anm: true` while TS reports `is_anm: false`. State hashes match, confirming
  game logic is correct but ANM tracking differs. This indicates HEX-PARITY-02 (ANM state
  divergence) is **NOT resolved** contrary to the initial assessment.

  **Secondary Issues (6 games):** Full state divergence with `current_player`, `current_phase`,
  and `state_hash` mismatches. Python shows `ring_placement` phase while TS shows
  `territory_processing` - suggests phase transition logic divergence.

  **Structural Errors (2 games):**
  - `de0d9a9d-492a-496a-860a-48cdf38e9199`: `continue_capture_segment` in `forced_elimination` phase
  - `c5759dbb-710e-4723-8d64-43a70be38400`: `move_stack` with "No stack at origin" error

  **Status: PARITY BLOCKED.** Cannot mark as canonical until ANM parity bugs are fixed.
  The 300 games are recorded but not usable for canonical training.

- 2025-12-21 (earlier) Post HEX-PARITY-01 Fix Regeneration: Successfully regenerated `canonical_hexagonal.db`
  after the HEX-PARITY-01 fix was applied to `phase_machine.py:138`. The fix modified
  `_on_line_processing_complete()` to skip `territory_processing` when conditions match TS behavior.

  **Results (11 games checked):**
  - 3 games (27%) pass full parity ✅
  - 7 games fail with semantic divergence (ANM state mismatch in `line_processing` phase)
  - 1 game fails with structural issue (`skip_capture` not valid in `forced_elimination` phase)

  **HEX-PARITY-01 is RESOLVED:** The original pattern (phase mismatch after `no_territory_action`
  where Python stayed in `territory_processing` while TS moved to `forced_elimination`) no longer
  appears in any of the failure logs.

  **HEX-PARITY-02 is NOT RESOLVED:** The ANM state divergence persists in the fresh generation
  with even higher failure rate (98% vs 64%). Investigation ongoing.

- 2025-12-20 PAR-02b hex parity audit: Verified that the PAR-01 self-capture fix (removal of
  `controlling_player != player` check in [`capture_chain.py:266-272`](app/rules/capture_chain.py:266))
  applies universally to all board types including hexagonal. However, `canonical_hexagonal.db`
  failed parity verification with structural error: "Invalid recovery slide: No buried ring for
  player 2 at -12,1,11" at k=709. The `db_health.canonical_hexagonal.json` shows `canonical_ok=false`.
  **This DB needs regeneration with the fixed code.** The registry entry erroneously claimed the
  DB was canonical; this has been corrected to `pending_gate` status.

- 2025-12-20 GPU canonical upgrade: GPU batch selfplay now produces canonical-quality data with:
  - 7-column move history tensor (added phase column)
  - Phase-specific move types (NO_PLACEMENT_ACTION, NO_MOVEMENT_ACTION, etc.)
  - CAPTURE/CHAIN_CAPTURE phase tracking with canonical move types (OVERTAKING_CAPTURE, CONTINUE_CAPTURE_SEGMENT)
  - FORCED_ELIMINATION detection (RR-CANON-R160)
  - Export translation module (`app/ai/gpu_canonical_export.py`)
  - Parity validation script (`scripts/run_gpu_canonical_parity_gate.py`)

  GPU selfplay data can now pass TS↔Python parity verification when properly exported. Use `--canonical-mode` flag for canonical-quality data generation.

- 2025-12-16 Vast.ai regeneration: `canonical_square8_2p.db` generated with 200 games (12,642 training samples) via distributed self-play on Vast.ai instance. 100% semantic parity verified (`games_with_semantic_divergence: 0`, `passed_canonical_parity_gate: true`). NPZ exported to `data/training/canonical_square8_2p.npz` with board-aware encoding. This is the primary 2-player square8 canonical training source.
- 2025-12-12 distributed regeneration + parity fix: `canonical_square8.db` now passes the canonical gate (`canonical_ok=true`, parity only end-of-game-only mismatches). It is safe for new training.
- 2025-12-12: `canonical_square8_3p.db` has initial gated games; scale up before training.
- 2025-12-12: `canonical_square8_4p.db` is now canonical (`canonical_ok=true`); scale up before training.
- 2025-12-20: `canonical_square19.db` re-gated with direct soak (light band) and `RINGRIFT_USE_MAKE_UNMAKE=true`; parity + history gates passed (`canonical_ok=true`). Scale volume toward targets.
- Historical: the sandboxed environment can fail OpenMP shared-memory allocation (`OMP: Error #179: Function Can't open SHM2 failed`); run canonical self-play on a host/container with SHM permissions.
- (Historical) 2025-12-09 re-gate of `canonical_square8.db` found TS replay structural errors; resolved on 2025-12-12 by aligning TS territory-control victory to collapsed-territory counts.

---

## Neural Network Models

### Model Provenance Table

| Model File                                | Training Data                                        | Status                  | Notes                                                                                                                                                                                            |
| ----------------------------------------- | ---------------------------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ringrift_v1.pth`                         | Legacy selfplay DBs                                  | **legacy_noncanonical** | Main model, needs retraining on canonical data                                                                                                                                                   |
| `ringrift_v1.pth.legacy`                  | Early selfplay DBs                                   | **legacy_noncanonical** | Original v1, historical only                                                                                                                                                                     |
| `ringrift_v1_legacy_nested.pth`           | Legacy nested replay export                          | **legacy_noncanonical** | Do not use                                                                                                                                                                                       |
| `ringrift_v1_mps.pth`                     | Legacy selfplay DBs                                  | **legacy_noncanonical** | MPS variant, needs retraining                                                                                                                                                                    |
| `ringrift_v1_mps_2025*.pth`               | Legacy selfplay DBs                                  | **legacy_noncanonical** | MPS checkpoints                                                                                                                                                                                  |
| `ringrift_v1_2025*.pth`                   | Legacy selfplay DBs                                  | **legacy_noncanonical** | v1 checkpoints from Nov 2025                                                                                                                                                                     |
| `ringrift_from_replays_square8.pth`       | Mixed replay DBs                                     | **legacy_noncanonical** | Trained from legacy replays                                                                                                                                                                      |
| `ringrift_from_replays_square8_2025*.pth` | Mixed replay DBs                                     | **legacy_noncanonical** | Checkpoints from legacy replays                                                                                                                                                                  |
| `ringrift_v1_hex*.pth`                    | Legacy hex DBs (radius 10)                           | ⚠️ **DEPRECATED_R10**   | Old hex geometry (331 cells, 36 rings, 21×21 input). Do not load; retrain on new radius-12 geometry (469 cells, 96 rings, 25×25 input).                                                          |
| `nnue_policy_square8_2p_canonical.pth`    | Gumbel MCTS selfplay (838 games, early-game focused) | **canonical**           | 2025-12-21 V6 model: 60.23% policy accuracy, 55% win rate vs baseline (100-game A/B test). Conservative hyperparams (80 epochs, 0.0002 LR) on first 50 moves. Best performing NNUE policy model. |

### Target Canonical Models

Once canonical self-play DBs are generated and exported, retrain these models:

| Target Model               | Training Data Source    | Status  |
| -------------------------- | ----------------------- | ------- |
| `ringrift_v2_square8.pth`  | canonical_square8_2p.db | Pending |
| `ringrift_v2_square19.pth` | canonical_square19.db   | Pending |
| `ringrift_v2_hex.pth`      | canonical_hexagonal.db  | Pending |

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

   For player-specific DBs, use `canonical_<board>_<players>p.db` (for example,
   `canonical_square8_2p.db`). If you omit `--db`, the generator defaults to
   `canonical_<board>_<players>p.db` based on `--num-players`.

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

## Canonical Gate Results

When `generate_canonical_selfplay.py` is used to generate and gate a new DB,
its canonical gate summary (parity + canonical history + FE/ANM checks) is
written to `data/games/db_health.<db>.json`.
If `run_canonical_selfplay_parity_gate.py` is used directly, store its JSON
summary alongside this document (for example as `parity_gate.<board>.json`).
Lower-level parity sweeps invoked directly by `check_ts_python_replay_parity.py`
can also be captured as `parity_summary.<label>.json` for ad-hoc debugging.

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

For DBs that **fail** the parity gate (structural issues or semantic divergence),
it is often useful to also persist:

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
- New selfplay data (post-fix) should pass canonical gates
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

_Last updated: 2025-12-21_

---

## Cluster Deployment Workflow (Lambda GH200/H100)

Use the cluster deployment script to scale canonical selfplay across all 12 board/player combinations. This workflow is required for generating sufficient training data for 2000+ Elo models.

### Prerequisites

1. **SSH access** to Lambda cluster nodes via Tailscale
2. **Repository cloned** at `~/ringrift/ai-service` on each node
3. **Python venv** set up with dependencies at `~/ringrift/ai-service/venv`
4. **Node.js + npx** available for TS parity verification

### Deployment Script

```bash
# From ai-service/ directory
./scripts/deploy_multi_config_training.sh <command>
```

| Command    | Description                                             |
| ---------- | ------------------------------------------------------- |
| `selfplay` | Deploy canonical selfplay generation for all 12 configs |
| `status`   | Check training/selfplay status on all nodes             |
| `deploy`   | Deploy NNUE training for all 12 configs                 |
| `collect`  | Collect trained models from cluster to local            |
| `help`     | Show usage information                                  |

### Full Workflow

**Step 1: Generate Canonical Training Data (Cluster)**

```bash
# Deploy selfplay to all cluster nodes
./scripts/deploy_multi_config_training.sh selfplay

# Monitor progress
./scripts/deploy_multi_config_training.sh status

# Check completion on specific node
ssh ubuntu@lambda-gh200-a 'tail -f ~/ringrift/ai-service/logs/canonical_selfplay_square8_2p.log'
```

Each node generates games iteratively until `--min-recorded-games 200` is reached. Look for `"canonical_ok": true` in the summary JSON to confirm data is ready.

**Step 2: Export to NPZ Format (Cluster)**

Once selfplay completes, export to NPZ format on each node:

```bash
# SSH to node and run export
ssh ubuntu@lambda-gh200-a 'cd ~/ringrift/ai-service && source venv/bin/activate && \
  python scripts/db_to_training_npz.py \
    --db data/games/canonical_square8_2p.db \
    --output data/training/canonical_square8_2p.npz \
    --board-type square8 --num-players 2'
```

Or use the batch export script:

```bash
ssh ubuntu@lambda-gh200-a 'cd ~/ringrift/ai-service && \
  for board in square8 square19 hex8 hexagonal; do
    for players in 2 3 4; do
      db="data/games/canonical_${board}_${players}p.db"
      npz="data/training/canonical_${board}_${players}p.npz"
      if [[ -f "$db" ]]; then
        python scripts/db_to_training_npz.py --db "$db" --output "$npz" \
          --board-type "$board" --num-players "$players" && echo "Exported $npz"
      fi
    done
  done'
```

**Step 3: Train Models (Cluster)**

```bash
# Deploy training for all 12 configs
./scripts/deploy_multi_config_training.sh deploy

# Monitor training progress
./scripts/deploy_multi_config_training.sh status
```

**Step 4: Collect Trained Models (Local)**

```bash
# Collect models from all cluster nodes
./scripts/deploy_multi_config_training.sh collect
```

Models are saved to `models/nnue/cluster_collected_<timestamp>/`.

### Node Allocation

The script distributes configs across available nodes round-robin:

| GPU Type | Nodes                         | Best For                            |
| -------- | ----------------------------- | ----------------------------------- |
| GH200    | 10 nodes (a-l, excluding b,j) | square8, hex8 (smaller boards)      |
| H100     | 2 nodes                       | square19, hexagonal (larger boards) |

For large boards (square19, hexagonal), prefer H100 nodes for faster game generation.

### Data Volume Targets

| Board Type | Min Games | Training Target | Games/Hour (est.) |
| ---------- | --------- | --------------- | ----------------- |
| square8    | 200       | 1000+           | ~50-100           |
| square19   | 200       | 1000+           | ~10-20            |
| hex8       | 200       | 1000+           | ~30-50            |
| hexagonal  | 200       | 1000+           | ~5-15             |

### Troubleshooting

**"No cluster nodes available"**

- Check Tailscale is running: `tailscale status`
- Verify SSH connectivity: `ssh ubuntu@lambda-gh200-a 'echo ok'`

**Selfplay failing parity gate**

- Check the gate summary: `cat data/games/db_health.canonical_<board>_<N>p.json`
- Look for `passed_canonical_parity_gate: false` or `non_canonical_games > 0`
- May indicate rules engine divergence requiring code fix

**Training not finding data**

- Verify NPZ export completed: `ls -la data/training/canonical_*.npz`
- Check NPZ is non-empty: `python -c "import numpy as np; d=np.load('data/training/canonical_square8_2p.npz'); print(d['features'].shape)"`
