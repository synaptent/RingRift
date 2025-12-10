# FSM Extension Strategy

> **Status:** Graduated (FSM is Canonical)
> **Last Updated:** 2025-12-10
> **Related:** [STATE_MACHINES.md](./STATE_MACHINES.md), [CANONICAL_ENGINE_API.md](./CANONICAL_ENGINE_API.md)

---

## Executive Summary

**FSM is now the canonical game state orchestrator** (RR-CANON compliance). The Turn FSM validates all phase transitions and move legality. Shadow mode has been removed - FSM validation is either active (default) or disabled (for emergency rollback only).

This document tracks the FSM extension roadmap, now largely complete.

---

## Current State

### FSM Implementation (TS)

| File                                        | Purpose                                                  |
| ------------------------------------------- | -------------------------------------------------------- |
| `src/shared/engine/fsm/TurnStateMachine.ts` | Core FSM: phase transitions, state, and event validation |
| `src/shared/engine/fsm/FSMAdapter.ts`       | Bridge between FSM events and existing Move types        |
| `src/shared/engine/fsm/index.ts`            | Public exports for FSM validation                        |

### Integration Points

| Location                                              | Integration                                                       |
| ----------------------------------------------------- | ----------------------------------------------------------------- |
| `src/shared/engine/orchestration/turnOrchestrator.ts` | Imports `validateMoveWithFSM`, `isMoveTypeValidForPhase` from FSM |
| Environment Variable                                  | `RINGRIFT_FSM_VALIDATION_MODE` controls shadow/active/off modes   |

### Validation Modes

- **`off`**: FSM validation disabled (emergency rollback only, not recommended)
- **`active`** (default): FSM validation is authoritative; invalid moves rejected with FSM error codes

> **Note:** Shadow mode has been removed. FSM is canonical.

### Related Components

| File                                                   | Role                                               |
| ------------------------------------------------------ | -------------------------------------------------- |
| `src/shared/engine/orchestration/phaseStateMachine.ts` | Legacy phase transition helpers (being superseded) |
| `ai-service/app/rules/history_contract.py`             | Python phase↔move contract (must mirror FSM)       |
| `ai-service/app/game_engine.py`                        | Python GameEngine (parity target)                  |

---

## Extension Roadmap

### Phase 1: Validation Unification (P0) ✅ COMPLETE

**Goal:** Make FSM-active validation authoritative for move legality.

| Task                                             | Status  | Files                 |
| ------------------------------------------------ | ------- | --------------------- |
| Wire FSM validation into turnOrchestrator        | ✅ Done | `turnOrchestrator.ts` |
| Add shadow mode logging for divergence detection | ✅ Done | `turnOrchestrator.ts` |
| Enable `active` mode by default in test/parity   | ✅ Done | CI config             |
| Update fixtures that assume legacy coercions     | ✅ Done | `tests/fixtures/**`   |
| Remove shadow mode infrastructure                | ✅ Done | Multiple files        |
| Graduate FSM to canonical                        | ✅ Done | `envFlags.ts`         |

**Outcome:**

- FSM is now the canonical validator (default: active)
- Shadow mode removed (~2,300 lines of code)
- All tests pass with FSM validation enabled

### Phase 2: Orchestrator FSM Control (P1) ✅ COMPLETE

**Goal:** Replace manual phase-routing branches with FSM-driven transitions.

| Task                                              | Status  |
| ------------------------------------------------- | ------- |
| Add feature flag `RINGRIFT_FSM_ORCHESTRATOR_MODE` | ✅ Done |
| Add `computeFSMOrchestration()` function          | ✅ Done |
| Add `compareFSMWithLegacy()` for shadow mode      | ✅ Done |
| Fix `currentPlayer` tracking for bookkeeping      | ✅ Done |
| Integrate FSM orchestrator into turnOrchestrator  | ✅ Done |
| Validate via orchestrator soak + parity gates     | ✅ Done |
| Graduate to canonical (remove shadow mode)        | ✅ Done |

**Outcome:**

- FSM orchestrator is now canonical
- `RINGRIFT_FSM_VALIDATION_MODE` only supports 'off' and 'active' (default: active)
- Shadow mode infrastructure removed
- Bookkeeping moves correctly set `currentPlayer`
- `deriveStateFromGame()` handles player derivation for bookkeeping moves

### Phase 3: Decision Surfaces (P1) ✅ COMPLETE

**Goal:** Emit pending decisions (line/territory/FE choices) from FSM state.

| Task                                              | Status  |
| ------------------------------------------------- | ------- |
| Drive `pendingLines` from FSM state               | ✅ Done |
| Drive `pendingRegions` from FSM state             | ✅ Done |
| Add `FSMDecisionSurface` to orchestration result  | ✅ Done |
| Include chain continuations in decision surface   | ✅ Done |
| Include forced elimination count in surface       | ✅ Done |
| Ensure FE surfaces only after territory phase     | ✅ Done |
| Forbid auto-advance without explicit `no_*` moves | ✅ Done |

**Outcome:**

- `FSMDecisionSurface` provides concrete data for all decision types
- FSM guards enforce explicit `no_*` moves for phase transitions
- Hosts construct valid decision options from FSM state

### Phase 4: Python Parity (P1) ✅ COMPLETE

**Goal:** Align Python phase machine to TS FSM transitions exactly.

| Task                                          | Status  | Files                            |
| --------------------------------------------- | ------- | -------------------------------- |
| Mirror FSM transition table in Python         | ✅ Done | `ai-service/app/rules/fsm.py`    |
| Add `FSMDecisionSurface` equivalent in Python | ✅ Done | `ai-service/app/rules/fsm.py`    |
| Add `FSMOrchestrationResult` in Python        | ✅ Done | `ai-service/app/rules/fsm.py`    |
| Add `compute_fsm_orchestration()` function    | ✅ Done | `ai-service/app/rules/fsm.py`    |
| Add `compare_fsm_with_legacy()` function      | ✅ Done | `ai-service/app/rules/fsm.py`    |
| Update Python FSM parity tests                | ✅ Done | `tests/rules/test_fsm_parity.py` |
| Graduate Python FSM as canonical              | ✅ Done | `ai-service/app/rules/fsm.py`    |
| Wire `GameEngine._update_phase` to FSM        | ✅ Done | `ai-service/app/game_engine.py`  |

**Outcome:**

- Python FSM types mirror TypeScript exactly
- All 28 Python FSM parity tests pass
- FSM behavior is canonical in Python (skips empty phases correctly)
- `GameEngine._update_phase` now uses `compute_fsm_orchestration()` for phase transitions
- Legacy phase_machine kept as fallback (RINGRIFT_FSM_VALIDATION_MODE=off)

### Phase 5: UI/Telemetry Integration (P2) ✅ COMPLETE

**Goal:** Surface FSM state to UI and telemetry systems.

| Task                                      | Status     | Files                                   |
| ----------------------------------------- | ---------- | --------------------------------------- |
| Adapter for FSM → GameHUD view model      | ✅ Done    | `src/client/adapters/gameViewModels.ts` |
| FSM decision surface telemetry events     | ✅ Done    | `src/shared/telemetry/rulesUxEvents.ts` |
| `FSMDecisionSurfaceViewModel` type        | ✅ Done    | `src/client/adapters/gameViewModels.ts` |
| `toFSMDecisionSurfaceViewModel()` adapter | ✅ Done    | `src/client/adapters/gameViewModels.ts` |
| `extractFSMTelemetryFields()` helper      | ✅ Done    | `src/client/adapters/gameViewModels.ts` |
| FSM action traces in replay harness       | 🔜 Planned |                                         |
| Teaching overlay FSM-aware explanations   | 🔜 Planned |                                         |

**Outcome:**

- New telemetry event types: `fsm_decision_surface_shown`, `fsm_decision_made`, `fsm_phase_transition`
- `RulesUxEventPayload` extended with FSM fields: `fsmPhase`, `fsmDecisionType`, `fsmPendingLineCount`, etc.
- `FSMDecisionSurfaceViewModel` provides UI-ready decision surface data
- `toFSMDecisionSurfaceViewModel()` transforms FSM orchestration results for HUD consumption
- `extractFSMTelemetryFields()` prepares low-cardinality metrics for telemetry emission

### Phase 6: Testing & Fixtures (P2) ✅ COMPLETE

**Goal:** Comprehensive FSM test coverage.

| Task                                  | Status  | Files                                                                               |
| ------------------------------------- | ------- | ----------------------------------------------------------------------------------- |
| Property-based random event sequences | ✅ Done | `tests/unit/fsm/FSM.property.test.ts`                                               |
| Cross-language fixture generation     | ✅ Done | `tests/fixtures/fsm-parity/v1/`, `tests/unit/fsm/FSM.crossLanguageFixtures.test.ts` |
| Python fixture loader                 | ✅ Done | `ai-service/tests/rules/test_fsm_fixtures.py`                                       |
| FE entry/exit targeted tests          | ✅ Done | `tests/unit/fsm/FSM.forcedElimination.test.ts`                                      |
| Territory loop tests                  | ✅ Done | `tests/unit/fsm/FSM.territoryLoop.test.ts`                                          |

**Outcome:**

- Property-based FSM tests with fast-check cover:
  - State invariants (valid phases, player rotation, actions array)
  - Error handling (invalid events, wrong player)
  - Transition determinism (same input → same output)
  - Phase progression invariants (correct next phases)
  - TurnStateMachine class invariants (history growth, canSend consistency)
  - Global events (RESIGN, TIMEOUT → game_over)
- Forced Elimination (FE) targeted tests (31 tests):
  - Multi-elimination sequences (ringsOverLimit > 1)
  - Entry conditions from territory/line processing
  - FE counter invariants and boundary cases
  - Global event handling (RESIGN/TIMEOUT)
  - Player rotation across 2p and 4p games
- Territory loop tests (31 tests):
  - Multi-region processing loops
  - Elimination sequences within regions
  - Territory → FE transition
  - Region index bounds checking
  - Action emission verification
- Cross-language fixture generation:
  - JSON fixtures at `tests/fixtures/fsm-parity/v1/fsm_transitions.vectors.json`
  - 23 test vectors covering all FSM phases and transitions
  - TypeScript loader at `tests/unit/fsm/FSM.crossLanguageFixtures.test.ts` (24 tests)
  - Python loader at `ai-service/tests/rules/test_fsm_fixtures.py` (14 tests)
  - Both languages consume the same JSON fixtures for true parity validation

### Phase 7: Data Pipeline (P2) ✅ COMPLETE

**Goal:** Thread FSM validation into training data.

| Task                                       | Status  | Files                                           |
| ------------------------------------------ | ------- | ----------------------------------------------- |
| Add `fsm_valid` field to move metadata     | ✅ Done | `ai-service/app/db/game_replay.py`              |
| Add `fsm_error_code` field                 | ✅ Done | `ai-service/app/db/game_replay.py`              |
| Schema v7 migration                        | ✅ Done | `ai-service/app/db/game_replay.py`              |
| `validate_move_fsm()` helper               | ✅ Done | `ai-service/app/db/recording.py`                |
| Block dataset generation on FSM validation | ✅ Done | `ai-service/app/training/env.py`                |
| Add `fsmValidated` to GameRecordMetadata   | ✅ Done | `ai-service/app/models/game_record.py`          |
| Update `build_training_game_record()`      | ✅ Done | `ai-service/app/training/game_record_export.py` |

**Outcome:**

- Database schema v7 adds `fsm_valid` (INTEGER 0/1/NULL) and `fsm_error_code` (TEXT) to `game_history_entries`
- Training environment already blocks on FSM validation (raises `FSMValidationError` in active mode)
- `GameRecordMetadata.fsm_validated` field tracks per-game FSM validation status
- `validate_move_fsm()` convenience function for recording FSM validation results
- All training data generated with FSM active is implicitly FSM-validated

---

## Target Areas: Detailed Analysis

### 1. Turn Orchestrator Integration

**Current:** `turnOrchestrator.ts` uses manual phase-routing with FSM validation as a guard.

**Target:** FSM transitions drive phase advancement directly.

| Aspect              | Pros                    | Cons/Risks                          |
| ------------------- | ----------------------- | ----------------------------------- |
| **Correctness**     | Single source of truth  | Large refactor                      |
| **Maintainability** | Cleaner code            | Regression risk in sandbox          |
| **Parity**          | Easier Python alignment | Decision-resolution plumbing needed |

**Reward:** HIGH - Foundation for all other extensions.

### 2. Move Validation Unification

**Current:** FSM validation runs alongside existing validators.

**Target:** FSM-active becomes sole authority.

| Aspect            | Pros                               | Cons/Risks              |
| ----------------- | ---------------------------------- | ----------------------- |
| **API Surface**   | Cleaner, unified                   | Legacy fixture breakage |
| **Bug Reduction** | Eliminates phase-invariant escapes | Migration effort        |
| **Divergence**    | None possible                      | Testing overhead        |

**Reward:** HIGH - Closes entire class of phase/move bugs.

### 3. Decision Surfaces

**Current:** Pending decisions derived in multiple places.

**Target:** FSM state is authoritative source.

| Aspect             | Pros                        | Cons/Risks            |
| ------------------ | --------------------------- | --------------------- |
| **Predictability** | Deterministic loops         | UI adjustments needed |
| **UX**             | Better state explainability | Telemetry alignment   |
| **Bugs**           | Removes off-phase moves     | Careful testing       |

**Reward:** MEDIUM-HIGH - Reduces weird-state bugs.

### 4. Python Parity

**Current:** Manual mirroring of phase rules.

**Target:** Generated or tightly-coupled FSM definitions.

| Aspect        | Pros             | Cons/Risks                 |
| ------------- | ---------------- | -------------------------- |
| **Parity**    | Eliminates drift | Codegen complexity         |
| **Debugging** | Easy diff        | Dual maintenance if manual |
| **Training**  | Clean data       | Schema changes             |

**Reward:** HIGH - Directly impacts AI training quality.

---

## Implementation Guidelines

### Adding FSM Events

1. Define event type in `TurnStateMachine.ts`
2. Add transition rule to FSM table
3. Add adapter mapping in `FSMAdapter.ts`
4. Mirror in Python `phase_machine.py`
5. Add test coverage
6. Update parity fixtures

### Enabling Active Mode

```bash
# In test environment
export RINGRIFT_FSM_VALIDATION_MODE=active

# Run parity checks
npm run test:parity
```

### Debugging FSM Divergences

1. Enable shadow mode logging
2. Run failing test/scenario
3. Check logs for `[FSM_SHADOW_VALIDATION]` entries
4. Compare `fsmResult` vs `existingResult` in divergence logs
5. Fix either FSM rules or legacy validator

---

## Immediate Action Items

1. **Fix `current_player` mismatch** at `game_over` transition
   - Use existing state bundles from parity failures
   - Align TS and Python phase-machine transitions

2. **Enable FSM-active in CI parity tests**
   - Update CI config to set `RINGRIFT_FSM_VALIDATION_MODE=active`
   - Fix any resulting test failures

3. **Add Python phase-machine parity tests**
   - Generate from TS FSM transition table
   - Add to `ai-service/tests/parity/`

4. **Update fixtures with zero-ring skip_placement**
   - Audit fixtures assuming legacy behavior
   - Update to use explicit `no_placement_action`

5. **Prototype FSM-driven orchestrator flag**
   - Add `RINGRIFT_FSM_ORCHESTRATOR` env flag
   - Run orchestrator soak with flag enabled
   - Gate on canonical history validation

---

## Success Metrics

| Metric                                | Target | Current |
| ------------------------------------- | ------ | ------- |
| FSM-legacy divergences in shadow mode | 0      | TBD     |
| Parity tests passing with active mode | 100%   | ~95%    |
| Python phase-machine coverage         | 100%   | ~80%    |
| Canonical DBs passing FSM validation  | 100%   | TBD     |

---

## References

- **FSM Source:** `src/shared/engine/fsm/`
- **Orchestrator:** `src/shared/engine/orchestration/turnOrchestrator.ts`
- **Python Phase Contract:** `ai-service/app/rules/history_contract.py`
- **Parity Runbook:** [PARITY_VERIFICATION_RUNBOOK.md](../PARITY_VERIFICATION_RUNBOOK.md)
- **State Machines Overview:** [STATE_MACHINES.md](./STATE_MACHINES.md)
