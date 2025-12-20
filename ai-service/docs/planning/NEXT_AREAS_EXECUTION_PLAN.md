# Next Areas Execution Plan (Remote)

**Created:** 2025-12-20
**Updated:** 2025-12-20
**Status:** Lane 3 NEW BUG FOUND - Line Detection Parity; Lane 1 blocked

---

## Lane 1: Canonical Data Pipeline (cluster)

- [x] Confirm selfplay jobs are canonical pipeline or raw
- [ ] For each board (square19, hex), run canonical gate:
  - [x] Parity check (TS vs Python) - **FAILED**
  - [x] Canonical phase-history validation - **FAILED**
- [x] Write `db_health.canonical_<board>.json` for each DB
  - Created: `data/db_health/canonical_square19.json`
  - Created: `data/db_health/canonical_hexagonal.json`
  - Created: `data/db_health/canonical_square8.json`
- [ ] Update `TRAINING_DATA_REGISTRY.md` with `canonical_ok` status and provenance
- [x] **FIXED: Phase transition parity bug (commit b8175468)**

**Status:** Ready for re-validation. Issues addressed:

1. Legacy DBs (jsonl*converted*\*) use old schema v1 - structural issue, not fixable
2. ~~Phase transition parity bug~~ **FIXED** - `no_territory_action` now handles phase transition inline

### Lane 1 Progress Log

| Date       | Board                    | Parity                             | Phase-History         | DB Health JSON           | Notes                                                          |
| ---------- | ------------------------ | ---------------------------------- | --------------------- | ------------------------ | -------------------------------------------------------------- |
| 2025-12-20 | **PARITY FIX VERIFIED**  | **PASS (0 semantic divergence)**   | -                     | -                        | b8175468: `no_territory_action` inline phase transition fix    |
| 2025-12-20 | selfplay.db (post-fix)   | PASS (0/10 semantic)               | 10 structural         | -                        | Structural issues are legacy data quality, not parity          |
| 2025-12-20 | square8_2p               | FAIL (4/10 semantic divergence)    | FAIL (19+ violations) | canonical_square8.json   | Phase mismatch: Python=ring_placement, TS=territory_processing |
| 2025-12-20 | jsonl_aggregated (mixed) | FAIL (5/20 semantic divergence)    | -                     | -                        | Same phase transition bug                                      |
| 2025-12-20 | square19_2p              | FAIL (14/14 structural)            | N/A                   | canonical_square19.json  | Missing game_moves table (v1 schema)                           |
| 2025-12-20 | hexagonal/selfplay.db    | FAIL (429/429 structural)          | N/A                   | canonical_hexagonal.json | Missing game_moves table (v1 schema)                           |
| 2025-12-20 | staging/iter0_sq8_2p     | FAIL (151 semantic, 33 structural) | TBD                   | canonical_square8.json   | Pre-fix data + line detection bug - 71% semantic divergence    |
| 2025-12-20 | canonical_test (fresh)   | PASS (0/20 semantic)               | 20 structural         | -                        | Phase fix works; structural issues = line detection bug        |

### Critical Finding: Phase Transition Parity Bug - **RESOLVED**

**Symptom:** Python and TypeScript engines disagreed on phase transitions:

- Python reports phase: `ring_placement`
- TypeScript reports phase: `territory_processing`
- Divergence occurred mid-game (moves 50-125 typically)

**Root Cause (Fixed in b8175468):**

- `no_territory_action` case in `applyMoveWithChainInfo` returned state without advancing phase/player
- `applyMoveForReplay` does NOT call `processPostMovePhases`, so phase transition must be inline
- Fix: Added inline check for forced_elimination + turn rotation in `no_territory_action` case

**Legacy DBs:**

- `square8_2p.db`, `jsonl_aggregated.db`, `selfplay.db` - data generated pre-fix is non-canonical
- New selfplay data (post-fix) should pass parity gates

### Selfplay Configuration Verified

Current selfplay on lambda-gh200-e uses canonical pipeline:

```
--difficulty-band canonical --fail-on-anomaly --streaming-record
```

DB path: `data/games/distributed_soak_runs/distributed_soak_square19_3p_20251220_045423/`

---

## Lane 2: AI Factory - IG-GMO (experimental tier)

- [ ] Wire IG-GMO into AI factory mapping (server and ai-service)
- [ ] Gate it behind experimental flag or difficulty tier
- [ ] Add doc entry under AI difficulty ladder and service endpoints

**Status:** Pending (blocked on Lane 1)

---

## Lane 3: Parity Hardening

**Priority: NORMAL** - Core fix complete

- [x] **FIXED: Phase transition parity bug (commit b8175468)**
  - Root cause: `no_territory_action` case in `applyMoveWithChainInfo` did not advance phase/player
  - Fix: Handle phase transition inline (check forced_elimination, then rotate to next player)
  - Verified: semantic divergence 0 on post-fix data
- [ ] Add unit tests for territory detection (empty region semantics)
- [ ] Add replay contract tests for `forced_elimination` and `no_territory_action` sequencing

**Status:** Core fix deployed. Unit tests still pending.

### NEW Finding: Line Detection Parity Bug (2025-12-20)

**Symptom:** After `move_stack`, TS enters `line_processing` but Python recorded `no_line_action`.

**Error from TS replay:**

```
[PHASE_MOVE_INVARIANT] Cannot apply move type 'place_ring' in phase 'line_processing'
```

**Root Cause Analysis:**

- TS `deriveLineProcessingState` uses `findLinesForPlayer(state.board, player, state.players.length)`
- TS detects 1+ lines after `move_stack` → stays in `line_processing`
- Python `BoardManager.find_all_lines` detects 0 lines → emits `no_line_action` → transitions to `territory_processing`
- The engines disagree on whether a line was formed

**Location in code:**

- TS: `src/shared/engine/fsm/FSMAdapter.ts:703-710` - `deriveLineProcessingState`
- TS: `src/shared/engine/lineDetection.ts:90-96` - `findLinesForPlayer`
- Python: `app/board_manager.py:170-239` - `find_all_lines`

**Affected Data:**

- `canonical_square19.db` - 3/3 games failing (26MB, 2-player square19)
- All selfplay data with this bug is non-canonical

**Next Steps:**

1. Create minimal reproduction case with exact board state
2. Compare line detection output between Python and TS at the divergence point
3. Fix whichever implementation is incorrect

---

## Lane 4: Documentation Audit

- [ ] Search docs for outdated parity/canonical DB notes
- [ ] Update docs to reflect current gating flow and new IG-GMO tier

**Status:** Pending

---

## Cluster Nodes

| Node  | SSH Command                      | Purpose                     |
| ----- | -------------------------------- | --------------------------- |
| A40   | `ssh -p 38742 root@ssh8.vast.ai` | Primary training/validation |
| 5070  | `ssh -p 10042 root@ssh2.vast.ai` | Secondary training          |
| 4080S | `ssh -p 19940 root@ssh3.vast.ai` | Benchmarking                |

---

## Scripts Reference

```bash
# Parity gate
PYTHONPATH=. python ai-service/scripts/check_ts_python_replay_parity.py \
  --db <path_to_db> --compact

# Canonical history gate
PYTHONPATH=. python ai-service/scripts/check_canonical_phase_history.py \
  --db <path_to_db>
```

---

## Background Tasks

| Task       | PID   | Command                                                                               | Log                                | Started          |
| ---------- | ----- | ------------------------------------------------------------------------------------- | ---------------------------------- | ---------------- |
| Model Sync | 56000 | `sync_models.py --sync --use-sync-coordinator --config config/distributed_hosts.yaml` | `logs/sync_models_coordinator.log` | 2025-12-20 00:24 |

Monitor: `tail -f logs/sync_models_coordinator.log`
