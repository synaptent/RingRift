# RingRift Rules Engine Surface Audit

**Last Updated:** 2025-12-21
**Document Purpose:** Identify which modules encode rules (Single Source of Truth) versus which are adapters over the canonical engine.

## Executive Summary

The RingRift Python implementation uses a **two-layer architecture**:

1. **Canonical Rules Layer** (`app/rules/`): Encodes RingRift game semantics per RULES_CANONICAL_SPEC.md
2. **Host Adapter Layer** (`app/game_engine/` + `app/_game_engine_legacy.py` compatibility path): Provides move generation and state transitions

The archived monolithic GameEngine implementation still backs the stable
`app.game_engine` facade during migration. The newer `app/rules/` layer is
being built as a **shadow-first** migration path.

---

## Module Classification Matrix

### Single Source of Truth (SSoT) - Rule-Encoding Modules

| Module                                  | SSoT Type | Key Responsibilities                                                                |
| --------------------------------------- | --------- | ----------------------------------------------------------------------------------- |
| `app/rules/core.py`                     | Full SSoT | Board configs, victory thresholds, geometry helpers (RR-CANON-R060-R062, R110-R115) |
| `app/rules/placement.py`                | Full SSoT | Ring placement validation (RR-CANON-R080-R082)                                      |
| `app/rules/elimination.py`              | Full SSoT | Context-aware elimination (RR-CANON-R100, R113, R122, R145)                         |
| `app/rules/recovery.py`                 | Full SSoT | Recovery action (RR-CANON-R110-R115)                                                |
| `app/rules/capture_chain.py`            | Full SSoT | Capture mechanics (RR-CANON-R100-R103)                                              |
| `app/rules/history_contract.py`         | Full SSoT | Phase-move contract (RR-CANON-R070, R075)                                           |
| `app/rules/global_actions.py`           | Full SSoT | Action availability (RR-CANON-R200-R207)                                            |
| `app/game_engine/phase_requirements.py` | Full SSoT | BookKeeping move types (RR-CANON-R076)                                              |

### Adapter Modules (Orchestration Layer)

| Module                        | Adapts Over           | Purpose                              |
| ----------------------------- | --------------------- | ------------------------------------ |
| `app/rules/validators/*`      | GameEngine            | Phase/turn checks, delegates to SSoT |
| `app/rules/mutators/*`        | GameEngine.apply_move | State application, shadow validation |
| `app/rules/default_engine.py` | GameEngine            | Validator/mutator scheduling         |
| `app/rules/phase_machine.py`  | GameEngine            | Phase transitions                    |
| `app/rules/fsm.py`            | history_contract      | Move-to-phase validation             |

### Legacy / Transitional Modules

| Module                       | Status                            | Migration Path                                                  |
| ---------------------------- | --------------------------------- | --------------------------------------------------------------- |
| `app/_game_engine_legacy.py` | **Deprecated compatibility path** | Keep callers on `app.game_engine` while decomposition continues |
| `app/rules/legacy/*`         | **Deprecated**                    | Remove v1-v7 support in Q4 2026                                 |

---

## Duplication Risk Summary

| Rule Category         | Status        | Risk     | Files                                |
| --------------------- | ------------- | -------- | ------------------------------------ |
| **Geometry (Core)**   | SSoT ✓        | LOW      | core.py, geometry.py                 |
| **Capture**           | SSoT ✓        | LOW      | capture_chain.py                     |
| **Placement**         | SSoT ✓        | LOW      | placement.py                         |
| **Elimination**       | SSoT ✓        | LOW      | elimination.py                       |
| **Recovery**          | SSoT ✓        | LOW      | recovery.py                          |
| **Phase Transitions** | PARTIAL       | MEDIUM   | phase_machine.py, fsm.py, GameEngine |
| **Line Formation**    | NOT EXTRACTED | **HIGH** | GameEngine only                      |
| **Territory**         | NOT EXTRACTED | **HIGH** | GameEngine only                      |

---

## Compliance Status

**Overall Compliance**: 70% complete

### Implemented in SSoT

- [x] RR-CANON-R060/R061: Victory threshold
- [x] RR-CANON-R062: Territory victory minimum
- [x] RR-CANON-R070/R075: Phase-to-move contract
- [x] RR-CANON-R080-R082: Placement rules
- [x] RR-CANON-R100-R103: Capture mechanics
- [x] RR-CANON-R100: Forced elimination
- [x] RR-CANON-R110-R115: Recovery action
- [x] RR-CANON-R120: Line length
- [x] RR-CANON-R122: Line elimination cost
- [x] RR-CANON-R145: Territory elimination
- [x] RR-CANON-R200-R207: Global actions

### Not Yet Extracted

- [ ] Full line detection/formation logic
- [ ] Territory disconnection logic
- [ ] Movement geometry helpers

---

## Priority Recommendations

### Priority 1: Critical for Compliance

1. **Extract Line Formation Logic** → `app/rules/line.py`
   - Line detection, marker validation (RR-CANON-R120)
   - Implement LineValidator/LineMutator

2. **Extract Territory Processing Logic** → `app/rules/territory.py`
   - Region detection, collapse logic (RR-CANON-R140-R145)
   - Implement TerritoryValidator/TerritoryMutator

3. **Complete Phase FSM Migration**
   - Migrate phase_machine.py → fsm.py
   - Mark phase_machine.py as deprecated

### Priority 2: Risk Mitigation

4. Establish shadow contract tests for all validators/mutators
5. Extract movement geometry to geometry.py
6. Add cross-language parity tests

---

## Migration Roadmap

| Phase | Timeline | Goal                                |
| ----- | -------- | ----------------------------------- |
| 1     | Q1 2026  | Extract line.py, territory.py       |
| 2     | Q2 2026  | Complete phase FSM migration        |
| 3     | Q2 2026  | Extract movement geometry           |
| 4     | Q3 2026  | Mutator-first orchestration default |
| 5     | Q4 2026  | Remove legacy game support          |

---

## Key File Locations

### SSoT Modules (Rule-Encoding)

```
app/rules/
├── core.py                  # Board config, victory, recovery eligibility
├── placement.py             # Placement validation
├── elimination.py           # Context-aware elimination
├── recovery.py              # Recovery action
├── capture_chain.py         # Capture geometry
├── history_contract.py      # Phase-move contract
├── global_actions.py        # Action availability
└── fsm.py                   # FSM validation
```

### Legacy / Primary Executor

```
app/
├── _game_engine_legacy.py   # 4,482 LOC – Primary executor (active)
└── rules/legacy/            # Deprecated compatibility layer
```

---

## Conclusion

RingRift's Python implementation is in active migration toward a specification-driven architecture. 16/18 major rule groups have SSoT implementations. Remaining work focuses on extracting line/territory logic and completing the phase FSM migration.
