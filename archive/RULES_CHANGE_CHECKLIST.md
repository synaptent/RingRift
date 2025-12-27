> **DEPRECATED**: This document is no longer maintained. Please refer to [RULES_CANONICAL_SPEC.md](../RULES_CANONICAL_SPEC.md) or the current documentation in the root directory.

# RingRift Rules/Engine Change Checklist

**Purpose:** This document provides a systematic workflow for making changes to RingRift rules or engine implementations. Following this checklist helps prevent drift between specifications, TypeScript/Python engines, tests, and documentation.

**When to use:** Any time you modify game rules, add new mechanics, clarify existing behavior, or update engine implementations.

---

## Quick Reference: Key Documents

| Document                                                                                                      | Purpose                                        |
| ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| [`RULES_CANONICAL_SPEC.md`](../RULES_CANONICAL_SPEC.md)                                                       | Canonical rules with RR-CANON-RXXX identifiers |
| [`docs/supplementary/RULES_RULESET_CLARIFICATIONS.md`](../docs/supplementary/RULES_RULESET_CLARIFICATIONS.md) | CLAR-XXX entries for ambiguous areas           |
<<<<<<< Updated upstream
| [`RULES_IMPLEMENTATION_MAPPING.md`](../docs/rules/RULES_IMPLEMENTATION_MAPPING.md)                                       | Maps rules to TS + Python code locations       |
| [`RULES_ENGINE_ARCHITECTURE.md`](../docs/architecture/RULES_ENGINE_ARCHITECTURE.md)                                             | Architecture, parity, and rollout strategy     |
| [`../docs/ai/AI_TRAINING_AND_DATASETS.md`](../docs/ai/AI_TRAINING_AND_DATASETS.md)                                        | AI training implications                       |
=======
| [`RULES_IMPLEMENTATION_MAPPING.md`](../docs/rules/RULES_IMPLEMENTATION_MAPPING.md)                            | Maps rules to TS + Python code locations       |
| [`RULES_ENGINE_ARCHITECTURE.md`](../docs/architecture/RULES_ENGINE_ARCHITECTURE.md)                           | Architecture, parity, and rollout strategy     |
| [`../docs/ai/AI_TRAINING_AND_DATASETS.md`](../docs/ai/AI_TRAINING_AND_DATASETS.md)                            | AI training implications                       |
>>>>>>> Stashed changes

---

## Phase 1: Before Making Changes

- [ ] **Identify affected RR-CANON rules**
  - Review [`../RULES_CANONICAL_SPEC.md`](../RULES_CANONICAL_SPEC.md) to determine which `RR-CANON-RXXX` rule(s) are affected.
  - Note the rule IDs for reference in commits and documentation.

- [ ] **Check existing clarifications**
  - Review [`docs/supplementary/RULES_RULESET_CLARIFICATIONS.md`](../docs/supplementary/RULES_RULESET_CLARIFICATIONS.md) for existing `CLAR-XXX` entries related to this area.
  - If a relevant clarification exists, ensure your change aligns with the resolved interpretation.

- [ ] **Assess ambiguity**
  - If the change is ambiguous or could have multiple interpretations, **add a new CLAR-XXX entry** before proceeding.
  - Document candidate interpretations and get consensus on the canonical choice.

- [ ] **Map affected code locations**
  - Consult [`../docs/rules/RULES_IMPLEMENTATION_MAPPING.md`](../docs/rules/RULES_IMPLEMENTATION_MAPPING.md) to identify all affected code:
    - TypeScript shared engine modules in `src/shared/engine/**`
    - TypeScript backend orchestration in `src/server/game/**`
    - TypeScript client sandbox in `src/client/sandbox/**`
    - Python rules engine in `ai-service/app/**`

---

## Phase 2: Specification Updates

- [ ] **Update canonical spec**
  - Modify [`../RULES_CANONICAL_SPEC.md`](../RULES_CANONICAL_SPEC.md) with new/changed rule semantics.
  - Use existing `RR-CANON-RXXX` format for rule identifiers.
  - If adding a new rule, assign the next available rule number in the appropriate section.

- [ ] **Update clarifications (if applicable)**
  - If this change resolves an ambiguity, update [`docs/supplementary/RULES_RULESET_CLARIFICATIONS.md`](../docs/supplementary/RULES_RULESET_CLARIFICATIONS.md).
  - Mark the CLAR entry as **Resolved** with the chosen interpretation.

- [ ] **Update player-facing rules (if applicable)**
  - If the change affects player-visible behavior, update:
    - [`../ringrift_complete_rules.md`](../ringrift_complete_rules.md) – narrative rules with examples
    - [`../ringrift_compact_rules.md`](../ringrift_compact_rules.md) – implementation-focused spec

---

## Phase 3: Implementation Updates

### TypeScript Shared Engine

- [ ] Update validators in `src/shared/engine/validators/**`:
  - [`PlacementValidator.ts`](src/shared/engine/validators/PlacementValidator.ts)
  - [`MovementValidator.ts`](src/shared/engine/validators/MovementValidator.ts)
  - [`CaptureValidator.ts`](src/shared/engine/validators/CaptureValidator.ts)
  - [`LineValidator.ts`](src/shared/engine/validators/LineValidator.ts)
  - [`TerritoryValidator.ts`](src/shared/engine/validators/TerritoryValidator.ts)

- [ ] Update mutators in `src/shared/engine/mutators/**`:
  - [`PlacementMutator.ts`](src/shared/engine/mutators/PlacementMutator.ts)
  - [`MovementMutator.ts`](src/shared/engine/mutators/MovementMutator.ts)
  - [`CaptureMutator.ts`](src/shared/engine/mutators/CaptureMutator.ts)
  - [`LineMutator.ts`](src/shared/engine/mutators/LineMutator.ts)
  - [`TerritoryMutator.ts`](src/shared/engine/mutators/TerritoryMutator.ts)
  - [`TurnMutator.ts`](src/shared/engine/mutators/TurnMutator.ts)

- [ ] Update shared helpers as needed:
  - [`core.ts`](src/shared/engine/core.ts) – geometry, S-invariant, board utilities
  - [`movementLogic.ts`](src/shared/engine/movementLogic.ts) – movement enumeration
  - [`captureLogic.ts`](src/shared/engine/captureLogic.ts) – capture enumeration
  - [`lineDetection.ts`](src/shared/engine/lineDetection.ts) – line discovery
  - [`lineDecisionHelpers.ts`](src/shared/engine/lineDecisionHelpers.ts) – line reward decisions
  - [`territoryDetection.ts`](src/shared/engine/territoryDetection.ts) – region discovery
  - [`territoryProcessing.ts`](src/shared/engine/territoryProcessing.ts) – region collapse
  - [`territoryDecisionHelpers.ts`](src/shared/engine/territoryDecisionHelpers.ts) – territory decisions
  - [`turnLogic.ts`](src/shared/engine/turnLogic.ts) – phase/turn state machine
  - [`victoryLogic.ts`](src/shared/engine/victoryLogic.ts) – win conditions

### TypeScript Backend Host

- [ ] Update backend orchestration if needed:
  - [`src/server/game/GameEngine.ts`](src/server/game/GameEngine.ts) – main game loop
  - [`src/server/game/RuleEngine.ts`](src/server/game/RuleEngine.ts) – move validation/generation
  - [`src/server/game/BoardManager.ts`](src/server/game/BoardManager.ts) – board utilities
  - [`src/server/game/turn/TurnEngine.ts`](src/server/game/turn/TurnEngine.ts) – turn progression

### TypeScript Client Sandbox

- [ ] Update sandbox if not using shared engine directly:
  - [`src/client/sandbox/ClientSandboxEngine.ts`](src/client/sandbox/ClientSandboxEngine.ts)
  - Relevant `sandbox*.ts` modules in `src/client/sandbox/`

### Python Rules Engine

- [ ] Update Python implementation:
  - [`ai-service/app/game_engine.py`](ai-service/app/game_engine.py) – main engine
  - [`ai-service/app/board_manager.py`](ai-service/app/board_manager.py) – board utilities
  - Validators in `ai-service/app/rules/validators/**`
  - Mutators in `ai-service/app/rules/mutators/**`

### Update Implementation Mapping

- [ ] If new functions/modules are added, update [`../docs/rules/RULES_IMPLEMENTATION_MAPPING.md`](../docs/rules/RULES_IMPLEMENTATION_MAPPING.md):
  - Add entries to the forward mapping (Rules → Implementation)
  - Add entries to the inverse mapping (Implementation → Rules)

---

## Phase 4: Test Updates

### Unit Tests

- [ ] **TypeScript unit tests** – add/update tests for affected mechanics:
  - Movement: [`tests/unit/movement.shared.test.ts`](tests/unit/movement.shared.test.ts)
  - Captures: [`tests/unit/captureLogic.shared.test.ts`](tests/unit/captureLogic.shared.test.ts)
  - Lines: [`tests/unit/lineDecisionHelpers.shared.test.ts`](tests/unit/lineDecisionHelpers.shared.test.ts)
  - Territory: [`tests/unit/territoryDecisionHelpers.shared.test.ts`](tests/unit/territoryDecisionHelpers.shared.test.ts)
  - Placement: [`tests/unit/placement.shared.test.ts`](tests/unit/placement.shared.test.ts)

- [ ] **Python unit tests** – mirror test coverage:
  - `ai-service/tests/test_rules_*.py`
  - `ai-service/tests/rules/test_validators.py`
  - `ai-service/tests/rules/test_mutators.py`

### Scenario Tests

- [ ] **RulesMatrix scenarios** – add/update if this is a core mechanic:
  - [`tests/scenarios/rulesMatrix.ts`](tests/scenarios/rulesMatrix.ts)
  - Related scenario test files in `tests/scenarios/`

### Parity Tests

- [ ] **TS ↔ Python parity** – ensure both engines agree:
  - TS fixtures: `tests/fixtures/rules-parity/v2/*.json`
  - TS parity tests: `tests/unit/Python_vs_TS.*.test.ts`
  - Python parity tests: `ai-service/tests/parity/test_rules_parity*.py`

### Determinism & No-Randomness Tests

- [ ] **Verify determinism tests pass:**
  - TS: [`tests/unit/EngineDeterminism.shared.test.ts`](tests/unit/EngineDeterminism.shared.test.ts)
  - Python: `ai-service/tests/test_engine_determinism.py`

- [ ] **Verify no-randomness guards pass:**
  - TS: [`tests/unit/NoRandomInCoreRules.test.ts`](tests/unit/NoRandomInCoreRules.test.ts)
  - Python: `ai-service/tests/test_no_random_in_rules_core.py`

---

## Phase 5: AI Training Alignment

_Only applicable if the change affects victory conditions, rewards, or state features._

- [ ] **Review reward computation:**
  - Check [`ai-service/app/training/env.py`](ai-service/app/training/env.py) for reward functions
  - Ensure victory/termination changes are reflected in episode termination

- [ ] **Review state encoding:**
  - Check [`ai-service/app/training/encoding.py`](ai-service/app/training/encoding.py) (if exists)
  - Check heuristic feature extraction if applicable

- [ ] **Update AI documentation:**
  - If needed, update [`../docs/ai/AI_TRAINING_AND_DATASETS.md`](../docs/ai/AI_TRAINING_AND_DATASETS.md)

---

## Phase 6: Documentation & UX

- [ ] **Update inline code comments**
  - Add comments explaining non-obvious behavior
  - Reference RR-CANON rule IDs in comments where helpful

- [ ] **Update architecture docs (if significant):**
  - [`../docs/architecture/RULES_ENGINE_ARCHITECTURE.md`](../docs/architecture/RULES_ENGINE_ARCHITECTURE.md)

- [ ] **Update UX components (if player-facing):**
  - Tooltips, modals, event log in `src/client/components/`
  - [`src/client/components/GameEventLog.tsx`](src/client/components/GameEventLog.tsx)
  - [`src/client/components/GameHUD.tsx`](src/client/components/GameHUD.tsx)

- [ ] **Consider audit report update (if significant):**
  - [`archive/FINAL_RULES_AUDIT_REPORT.md`](../archive/FINAL_RULES_AUDIT_REPORT.md)

---

## Phase 7: Final Verification

### Run Full Test Suites

- [ ] **TypeScript tests:**

  ```bash
  npm test
  ```

- [ ] **Python tests:**
  ```bash
  cd ai-service && pytest
  ```

### Run Targeted Verification

- [ ] **Parity tests specifically:**

  ```bash
  npm test -- Python_vs_TS
  cd ai-service && pytest tests/parity/
  ```

- [ ] **Determinism tests:**
  ```bash
  npm test -- EngineDeterminism NoRandomInCoreRules
  ```

### Board State Verification

- [ ] **Verify BoardManager repair counter stays zero on legal flows:**
  - If board state changes are involved, ensure no invariant violations occur
  - Check `BoardManager.assertBoardInvariants()` is not triggered during normal play

---

## Example: Adding a New Victory Condition

Suppose you want to add a "Ring Majority" victory condition at turn 50.

### Phase 1: Before Making Changes

- [x] Identify affected rules: RR-CANON-R170 (elimination), R171 (territory), R172 (LPS), R173 (stalemate)
- [x] Check clarifications: No existing CLAR entry for "timed victory"
- [x] Add CLAR entry: Create CLAR-004 documenting the new condition and its priority vs existing wins
- [x] Map affected code: `victoryLogic.ts`, `GameEngine.ts`, Python `game_engine.py`

### Phase 2: Specification Updates

- [x] Add RR-CANON-R174 to `RULES_CANONICAL_SPEC.md` defining Ring Majority victory
- [x] Update player-facing rules with new victory path

### Phase 3: Implementation Updates

- [x] Add `checkRingMajorityVictory()` to `victoryLogic.ts`
- [x] Wire into `evaluateVictory()` in correct priority order
- [x] Mirror in Python `game_engine.py`
- [x] Update `RULES_IMPLEMENTATION_MAPPING.md` with new function

### Phase 4: Test Updates

- [x] Add unit tests for ring majority detection
- [x] Add scenario test triggering majority at turn 50
- [x] Add parity fixture testing TS/Python agreement
- [x] Verify determinism tests still pass

### Phase 5: AI Training Alignment

- [x] Update reward function for new victory type
- [x] Document in AI training docs

### Phase 6: Documentation & UX

- [x] Add "Ring Majority" to victory modal messages
- [x] Update tooltips explaining new win condition

### Phase 7: Final Verification

- [x] Run `npm test` – all green
- [x] Run `pytest` – all green
- [x] Run parity tests – all green

---

## Example: Changing Line Length Requirement

Suppose you want to change `square8` line length from 3 to 4.

### Phase 1: Before Making Changes

- [x] Identify affected rules: RR-CANON-R001 (board config), R120-R122 (lines)
- [x] Check clarifications: Line length is in BOARD_CONFIGS, not ambiguous
- [x] Map affected code: `game.ts` types, `lineDetection.ts`, `lineDecisionHelpers.ts`, Python equivalents

### Phase 2: Specification Updates

- [x] Update RR-CANON-R001 table in `RULES_CANONICAL_SPEC.md`: square8 lineLength = 4
- [x] Update `ringrift_compact_rules.md` §1.1 version table
- [x] Update `ringrift_complete_rules.md` 8×8 section

### Phase 3: Implementation Updates

- [x] Update `BOARD_CONFIGS.square8.lineLength` in `src/shared/types/game.ts`
- [x] No code logic changes needed (uses config)
- [x] Update Python config to match

### Phase 4: Test Updates

- [x] Update line detection tests expecting 3-marker lines to expect 4
- [x] Update scenario tests for square8 line rewards
- [x] Generate new parity fixtures with updated config
- [x] Run determinism tests

### Phase 5: AI Training Alignment

- [x] No reward changes needed (line scoring already uses config)

### Phase 6: Documentation & UX

- [x] Update any UI that displays "3 markers" for square8

### Phase 7: Final Verification

- [x] Full test suite passes
- [x] Parity confirmed

---

## Document Metadata

**Created:** November 25, 2025

**Related Documents:**

- [`../RULES_CANONICAL_SPEC.md`](../RULES_CANONICAL_SPEC.md) – Canonical rules specification
- [`../docs/rules/RULES_IMPLEMENTATION_MAPPING.md`](../docs/rules/RULES_IMPLEMENTATION_MAPPING.md) – Code location mapping
- [`../docs/supplementary/RULES_RULESET_CLARIFICATIONS.md`](../docs/supplementary/RULES_RULESET_CLARIFICATIONS.md) – Ambiguity resolutions
- [`../docs/architecture/RULES_ENGINE_ARCHITECTURE.md`](../docs/architecture/RULES_ENGINE_ARCHITECTURE.md) – Engine architecture
- [`../docs/ai/AI_TRAINING_AND_DATASETS.md`](../docs/ai/AI_TRAINING_AND_DATASETS.md) – AI training documentation
- [`archive/FINAL_RULES_AUDIT_REPORT.md`](../archive/FINAL_RULES_AUDIT_REPORT.md) – Rules audit findings
