# FSM Extension Strategy

> **Status:** Living Document  
> **Last Updated:** 2025-12-08  
> **Related:** [STATE_MACHINES.md](./STATE_MACHINES.md), [CANONICAL_ENGINE_API.md](./CANONICAL_ENGINE_API.md)

---

## Executive Summary

RingRift has an explicit Turn FSM for game phase validation. This document outlines the strategy for extending FSM coverage to improve determinism, TS↔Python parity, and debugging capabilities.

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

- **`off`**: FSM validation disabled (legacy behavior)
- **`shadow`**: FSM validation runs alongside existing validation; divergences logged but not enforced
- **`active`**: FSM validation enforced; invalid moves rejected with FSM error codes

### Related Components

| File                                                   | Role                                               |
| ------------------------------------------------------ | -------------------------------------------------- |
| `src/shared/engine/orchestration/phaseStateMachine.ts` | Legacy phase transition helpers (being superseded) |
| `ai-service/app/rules/history_contract.py`             | Python phase↔move contract (must mirror FSM)       |
| `ai-service/app/game_engine.py`                        | Python GameEngine (parity target)                  |

---

## Extension Roadmap

### Phase 1: Validation Unification (P0)

**Goal:** Make FSM-active validation authoritative for move legality.

| Task                                             | Status         | Files                 |
| ------------------------------------------------ | -------------- | --------------------- |
| Wire FSM validation into turnOrchestrator        | ✅ Done        | `turnOrchestrator.ts` |
| Add shadow mode logging for divergence detection | ✅ Done        | `turnOrchestrator.ts` |
| Enable `active` mode by default in test/parity   | 🔄 In Progress | CI config             |
| Update fixtures that assume legacy coercions     | 🔜 Planned     | `tests/fixtures/**`   |
| Remove redundant phase-invariant checks          | 🔜 Planned     | Various validators    |

**Success Criteria:**

- All parity tests pass with `FSM_VALIDATION_MODE=active`
- Zero divergences between FSM and legacy validation in canonical DBs

### Phase 2: Orchestrator FSM Control (P1)

**Goal:** Replace manual phase-routing branches with FSM-driven transitions.

| Task                                              | Status     |
| ------------------------------------------------- | ---------- |
| Add feature flag `RINGRIFT_FSM_ORCHESTRATOR`      | 🔜 Planned |
| Map FSM events to move shapes for compatibility   | 🔜 Planned |
| Replace `advancePhase` calls with FSM transitions | 🔜 Planned |
| Validate via orchestrator soak + parity gates     | 🔜 Planned |

**Benefits:**

- Single source of truth for phase transitions
- Eliminates duplicated transition logic
- Cleaner invariant enforcement
- Easier diffing vs Python

### Phase 3: Decision Surfaces (P1)

**Goal:** Emit pending decisions (line/territory/FE choices) from FSM state.

| Task                                              | Status     |
| ------------------------------------------------- | ---------- |
| Drive `pendingLines` from FSM state               | 🔜 Planned |
| Drive `pendingRegions` from FSM state             | 🔜 Planned |
| Ensure FE surfaces only after territory phase     | 🔜 Planned |
| Forbid auto-advance without explicit `no_*` moves | 🔜 Planned |

**Benefits:**

- Predictable loops for multiple regions/lines
- Fewer "silent skips"
- Better UX/state explainability

### Phase 4: Python Parity (P1)

**Goal:** Align Python phase machine to TS FSM transitions exactly.

| Task                                           | Status     | Files                                   |
| ---------------------------------------------- | ---------- | --------------------------------------- |
| Mirror FSM transition table in Python          | 🔜 Planned | `ai-service/app/rules/phase_machine.py` |
| Consider codegen from shared JSON/YAML spec    | 🔜 Planned | New spec file                           |
| Add parity bundles for end-of-turn ownership   | 🔜 Planned | `ai-service/parity_fixtures/`           |
| Fix `current_player` divergence at `game_over` | 🔜 Planned | Both engines                            |

**Benefits:**

- Tight TS↔Python parity
- Easier bundle diffing
- Reduced maintenance burden

### Phase 5: UI/Telemetry Integration (P2)

**Goal:** Surface FSM state to UI and telemetry systems.

| Task                                    | Status     |
| --------------------------------------- | ---------- |
| Adapter for FSM → GameHUD view model    | 🔜 Planned |
| FSM action traces in replay harness     | 🔜 Planned |
| Teaching overlay FSM-aware explanations | 🔜 Planned |

### Phase 6: Testing & Fixtures (P2)

**Goal:** Comprehensive FSM test coverage.

| Task                                  | Status     |
| ------------------------------------- | ---------- |
| Property-based random event sequences | 🔜 Planned |
| Cross-language fixture generation     | 🔜 Planned |
| FE entry/exit targeted tests          | 🔜 Planned |
| Territory loop tests                  | 🔜 Planned |

### Phase 7: Data Pipeline (P2)

**Goal:** Thread FSM validation into training data.

| Task                                       | Status     |
| ------------------------------------------ | ---------- |
| Add `fsm_valid` field to move metadata     | 🔜 Planned |
| Block dataset generation on FSM validation | 🔜 Planned |
| Tag non-canonical sequences in export      | 🔜 Planned |

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
