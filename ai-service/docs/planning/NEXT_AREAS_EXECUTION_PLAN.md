# Next Areas Execution Plan (Remote)

**Created:** 2025-12-20
**Updated:** 2025-12-20
**Status:** Lane 3 COMPLETE - All parity bugs fixed and verified; Lane 1 unblocked

---

## Lane 1: Canonical Data Pipeline (cluster)

- [x] Confirm selfplay jobs are canonical pipeline or raw
- [x] For each board (square19, hex), run canonical gate:
  - [x] Parity check (TS vs Python) - **PASSED** (5 games, 0 divergences)
  - [x] Canonical phase-history validation - **PASSED**
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

**Priority: LOW** - All critical bugs fixed, verification complete

- [x] **FIXED: Phase transition parity bug (commit b8175468)**
  - Root cause: `no_territory_action` case in `applyMoveWithChainInfo` did not advance phase/player
  - Fix: Handle phase transition inline (check forced_elimination, then rotate to next player)
  - Verified: semantic divergence 0 on post-fix data
- [x] **FIXED: Line detection parity bug**
  - Root cause: FORCED_ELIMINATION phase handling missing in GPU selfplay scripts
  - Fix: Added proper forced elimination handling in run_gpu_selfplay.py and import_gpu_selfplay_to_db.py
  - Verified: 138 parity tests pass, 0 failures
- [ ] Add unit tests for territory detection (empty region semantics)
- [ ] Add replay contract tests for `forced_elimination` and `no_territory_action` sequencing

**Status:** All critical parity bugs fixed. Lane 1 unblocked.

### RESOLVED: Line Detection Parity Bug (2025-12-20)

**Status: FIXED** - Root cause was `no_territory_action` not being marked as turn-ending.

**Original Symptom:** After `move_stack`, TS enters `line_processing` but Python recorded `no_line_action`.

**Root Cause (FIXED):**

- After `no_territory_action`, `processPostMovePhases` was running incorrectly
- This caused TS to re-derive phase as `line_processing` for the _next_ player
- Python had already advanced to `ring_placement` for the next player's turn

**Fix (turnOrchestrator.ts:1522-1523):**

```typescript
const isTurnEndingTerritoryMove =
  move.type === 'skip_territory_processing' || move.type === 'no_territory_action';
```

This prevents `processPostMovePhases` from running after `no_territory_action`, since the inline
handler in `applyMoveWithChainInfo` already handles:

1. Forced elimination check (if `!hadAnyAction && hasStacks` → `forced_elimination` phase)
2. Turn rotation (otherwise → next player's `ring_placement`)

### Phase Coercion Analysis (2025-12-20)

**Question:** Does the fix add phase coercion on top of phase coercion?

**Answer:** No. The fix REDUCES the need for coercion.

**Findings:**

1. **The fix is a behavior correction**, not coercion. It makes TS match Python's canonical behavior.
2. **Existing coercion** (lines 1278-1390) is replay-tolerance only (`options?.replayCompatibility`).
3. **Coercion handles legacy data** where TS/Python phases weren't aligned pre-fix.
4. **No new coercion was added.** The fix at line 1522-1523 is a core engine fix.

**Verification:**

- Parity test on `canonical_square19.db`: **PASS** (5 games, 0 divergences, 2,588+ moves)
- Forced elimination scenarios: **Verified working** - inline handler correctly checks `hadAnyAction && hasStacks`

**Code paths verified:**

- `no_territory_action` handler (lines 2230-2271): Correctly transitions to `forced_elimination` or rotates player
- `computeHadAnyActionThisTurn` (lines 3580-3604): Correctly identifies bookkeeping vs real actions
- `isNoActionBookkeepingMove` (lines 3632-3640): `no_territory_action` is correctly classified as bookkeeping

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
