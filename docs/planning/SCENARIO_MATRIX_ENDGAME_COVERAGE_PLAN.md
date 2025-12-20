# Scenario Matrix + Endgame Coverage Plan

> **Created:** 2025-12-20
> **Owner:** Tool-driven agent
> **Goal:** Add deterministic, orchestrator-driven multi-phase endgame coverage and align the scenario matrix.

This plan assumes the working tree is unstable due to other agents. Changes are scoped to tests and docs directly related to multi-phase endgame coverage.

---

## Context

- **State:** Stable beta with consolidated orchestrator and strong TS-Python parity.
- **Gap:** Scenario matrix + endgame coverage still lacks explicit, orchestrator-driven multi-phase victory validation.
- **Focus:** Use contract vectors to drive chain -> line -> territory sequences and assert GameEndExplanation output.

---

## Scope

- `tests/scenarios/GameEndExplanation.multiPhase.test.ts`
- `tests/fixtures/contract-vectors/v2/multi_phase_turn.vectors.json`
- `docs/rules/RULES_SCENARIO_MATRIX.md`

---

## Plan

1. Identify contract-vector sequences for square8/square19/hex that traverse line -> territory.
2. Add a GameEndExplanation test that runs `processTurn`, resolves pending decisions, and forces a territory victory threshold for deterministic victory.
3. Update RULES_SCENARIO_MATRIX to reference the new test coverage and log progress here.

---

## Definition of Done

- Orchestrator-driven multi-phase test asserts territory victory and GameEndExplanation payload.
- Phases traversed include `line_processing` and `territory_processing`.
- Scenario matrix reflects the new coverage.

---

## Progress Log

- [x] Locate line -> territory contract vectors to drive the test (`multi_phase.full_sequence_with_territory*`).
- [x] Add orchestrator-driven GameEndExplanation multi-phase test.
- [x] Update RULES_SCENARIO_MATRIX to mention the new coverage.
