# Hex Parity Audit Report

## Audit Date: 2025-12-07

## Summary

**Overall Health Status: ⚠️ PENDING (Data Missing)**

Radius-10 assets remain deprecated and the HexNeuralNet alias/import fix landed, but we have no gated radius-12 data on disk. All `selfplay_hexagonal_*.db` and `selfplay_hex_mps_smoke.db` files are empty, so parity needs to be re-run after regenerating fresh radius-12 self-play.

## 1. Database Status

| Database                  | Exists | Games | Status                                                           |
| ------------------------- | ------ | ----- | ---------------------------------------------------------------- |
| canonical_hex.db          | **No** | -     | Deleted (deprecated radius-10)                                   |
| selfplay_hexagonal_2p.db  | Yes    | 0     | Empty radius-12 placeholder; needs fresh self-play before gating |
| selfplay_hexagonal_3p.db  | Yes    | 0     | Empty radius-12 placeholder; needs fresh self-play before gating |
| selfplay_hexagonal_4p.db  | Yes    | 0     | Empty radius-12 placeholder; needs fresh self-play before gating |
| selfplay_hex_mps_smoke.db | Yes    | 0     | Empty smoke DB; regenerate after alias/import fixes              |

**Note:** Historic parity fixtures still exist under `ai-service/parity_fixtures/*hex*`, but they pre-date the current alias/import fixes and should be treated as legacy until new games are generated.

## 2. Parity Test Results

### Core Hex Parity Tests

| Test                                                                         | Status      | Notes                                                   |
| ---------------------------------------------------------------------------- | ----------- | ------------------------------------------------------- |
| test_basic_capture_on_board_type[hexagonal]                                  | **PENDING** | Rerun once radius-12 self-play DBs are regenerated      |
| test_line_and_territory_scenario_parity[hexagonal]                           | **PENDING** | Blocked on missing replay data                          |
| test_get_valid_moves_line_processing_surfaces_only_line_decisions[hexagonal] | **PENDING** | Blocked on missing replay data                          |
| test_line_and_territory_ts_snapshot_parity[hexagonal]                        | **PENDING** | Blocked on missing replay data                          |
| test_overlength_line_option2_segments_exhaustive[hexagonal]                  | **PENDING** | Geometry code unchanged, but parity needs a fresh sweep |

### Hex Geometry Tests

| Test                              | Status         | Notes                                          |
| --------------------------------- | -------------- | ---------------------------------------------- |
| test_calculate_distance[...]      | **NOT RE-RUN** | Last known pass; rerun after data regeneration |
| test_get_path_positions_hexagonal | **NOT RE-RUN** | Last known pass; rerun after data regeneration |

### Replay Fixture Tests

| Test Category              | Status                             | Notes                                                                                 |
| -------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------- |
| Hex replay parity fixtures | **SKIPPED (legacy fixtures only)** | Requires TS runner and fresh radius-12 DB; current fixtures are pre-fix legacy traces |

## 3. Fixture Coverage

### Total Hex Fixtures: legacy-only (no fresh radius-12 replays)

| Player Count | Fixtures              | Coverage           |
| ------------ | --------------------- | ------------------ |
| 2-player     | Legacy traces present | Needs regeneration |
| 3-player     | Legacy traces present | Needs regeneration |
| 4-player     | Legacy traces present | Needs regeneration |
| MPS smoke    | Legacy traces present | Needs regeneration |

### Move Type Coverage in Hex Fixtures

| Move Type                  | Fixtures | Description                |
| -------------------------- | -------- | -------------------------- |
| `move_stack`               | ~15      | Basic stack movement       |
| `overtaking_capture`       | ~8       | Capture mechanics          |
| `continue_capture_segment` | ~5       | Chain capture continuation |
| `place_ring`               | ~3       | Ring placement phase       |
| Other phases               | ~1       | Various                    |

### State Bundle Directories

| Directory                      | Files | Purpose                      |
| ------------------------------ | ----- | ---------------------------- |
| `state_bundles_hex/`           | 3     | Pre-fix divergence snapshots |
| `state_bundles_hex_after_fix/` | 3     | Post-fix verification        |

## 4. Engine Support

### TypeScript

- [x] Board type enum includes 'hexagonal'
- [x] Geometry calculations in [`territoryDetection.ts`](../src/shared/engine/territoryDetection.ts)
- [x] Line detection in [`LineAggregate.ts`](../src/shared/engine/aggregates/LineAggregate.ts)
- [x] Placement validation in [`PlacementAggregate.ts`](../src/shared/engine/aggregates/PlacementAggregate.ts)
- [x] Territory processing in [`TerritoryAggregate.ts`](../src/shared/engine/aggregates/TerritoryAggregate.ts)

### Python

- [x] Board type support in [`core.py`](../ai-service/app/rules/core.py:32) (BoardConfig for HEXAGONAL)
- [x] Geometry calculations in [`geometry.py`](../ai-service/app/rules/geometry.py)
- [x] Capture chain support in [`capture_chain.py`](../ai-service/app/rules/capture_chain.py:136)
- [x] Mutable state handling in [`mutable_state.py`](../ai-service/app/rules/mutable_state.py:1736)

### New Hex Geometry Parameters (radius-12)

```python
# From ai-service/app/rules/core.py
BoardType.HEXAGONAL: BoardConfig(
    size=13,          # side length
    radius=12,        # hex radius
    total_cells=469,  # 3*12^2 + 3*12 + 1
    rings_per_player=48
)
```

## 5. Known Issues

### 5.1 Canonical Hex Database Does Not Exist

`canonical_hex.db` remains deleted and all radius-12 self-play DBs are empty. A new canonical database needs to be generated and gated after the HexNeuralNet alias/import fix.

**Recommendation:** Generate a canonical hex database using:

```bash
cd ai-service
PYTHONPATH=. python scripts/generate_canonical_selfplay.py \
  --board-type hexagonal \
  --num-players 2 \
  --num-games 100 \
  --db data/games/canonical_hex_r12.db
```

### 5.2 Replay Parity Tests Skipped

Replay fixture regression tests remain skipped until fresh radius-12 data exists and the TS runner is pointed at the new HexNeuralNet alias/import path.

## 6. Recommendations

### High Priority

1. **Generate Canonical Hex Database**: Create a proper `canonical_hex_r12.db` with radius-12 geometry and full parity validation.

2. **Fix Python 3.10 Compatibility**: Update the `NotRequired` import to use `typing_extensions` for compatibility.

### Medium Priority

3. **Expand Hex Fixture Coverage**: Add more fixtures for:
   - Territory processing on hex boards
   - Forced elimination scenarios
   - Multi-region territory claims

4. **Rename Self-Play Databases**: Consider renaming `selfplay_hexagonal_*.db` to indicate radius-12 explicitly (e.g., `selfplay_hex_r12_2p.db`).

### Low Priority

5. **Document Migration Path**: Update `HEX_DATA_DEPRECATION_NOTICE.md` with successful migration status.

6. **Clean Up State Bundles**: Archive or remove old state bundle directories once analysis is complete.

## 7. Verification Commands

```bash
# Run hex parity tests
cd ai-service && python -m pytest tests/parity/ -v -k "hex" --ignore=tests/parity/test_rules_parity_fixtures.py

# Run hex geometry tests
cd ai-service && python -m pytest tests/rules/test_geometry.py -v -k "hex"

# Check hex fixture count
ls ai-service/parity_fixtures/*hex* | wc -l

# List hex databases
ls -la ai-service/data/games/*hex*.db 2>/dev/null
```

## 8. Deprecation Status

### Legacy References Found

| Location                         | Type             | Status                              |
| -------------------------------- | ---------------- | ----------------------------------- |
| `test_eval_pools_multiplayer.py` | Skip message     | ✅ Correct (documents deprecation)  |
| `test_mcts_ai.py`                | Code comment     | ✅ Acceptable (legacy path testing) |
| `test_descent_ai.py`             | Code comment     | ✅ Acceptable (hex NN integration)  |
| `.mypy_cache/`                   | Cache files      | Not relevant                        |
| `venv/`                          | Third-party code | Not relevant                        |

**Conclusion:** No problematic legacy hex references remain in production code.

## 9. Audit Sign-off

- [ ] Hex geometry tests re-run (last pass pre-regeneration)
- [ ] Hex parity tests re-run (blocked on missing data)
- [ ] Hex parity fixtures refreshed (legacy-only traces remain)
- [ ] Canonical hex DB clean (does not exist - needs generation)
- [x] No legacy radius-10 data in production code
- [ ] Deprecation notice updated after regeneration

### Overall Status

| Criterion                   | Status                                    |
| --------------------------- | ----------------------------------------- |
| Hex geometry implementation | ⚠️ Pending re-verify with new data        |
| TS↔Python hex parity        | ❌ Blocked (no radius-12 games to replay) |
| Hex fixture coverage        | ⚠️ Legacy-only, needs regeneration        |
| Canonical hex DB            | ⚠️ Not present                            |
| Legacy data cleanup         | ✅ Radius-10 removed                      |

**Audit Result: PENDING – regenerate radius-12 data and rerun parity**

---

_Audit conducted: 2025-12-07_
_Auditor: AI Agent (W3-7)_
