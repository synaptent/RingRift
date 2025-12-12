# Rules Docs Consistency Pass (2025-12-11)

> **Doc Status (2025-12-11): Active (worklog / tracker)**
>
> **Purpose:** Track a repo-wide consistency pass resolving known contradictions across rulebooks, canonical spec, UX/analysis docs, and TS engine comments/tests.

---

## Scope

This pass focuses on four high-impact canonical clarifications and their propagation:

1. LPS “real actions” vs recovery
2. Square8 `lineLength` (2p exception)
3. Recovery legality (“at least `lineLength`”, overlength allowed)
4. Territory processing optionality (subset vs mandatory exhaustion)

---

## Canonical Decisions (SSoT)

- **LPS real actions (RR-CANON-R172):** Real actions are ring placement, non-capture movement, and overtaking captures. **Recovery and forced elimination are not real actions** and do not reset LPS tracking.
- **Square8 `lineLength`:** `lineLength = 4` for **2-player** square8; `lineLength = 3` for **3–4 player** square8. `lineLength = 4` for square19 and hex.
- **Recovery legality (RR-CANON-R112):** A recovery slide completes a line of **at least** `lineLength`; overlength lines are allowed and use Option 1/2 semantics.
- **Territory processing (RR-CANON-R140–R145 + RR-CANON-R075):** Player may process **any subset** of eligible regions; “stop early” is recorded as `skip_territory_processing` (distinct from `no_territory_action` when no regions exist).

---

## Changes Applied (High Level)

- **Canonical spec:** `RULES_CANONICAL_SPEC.md`
- **Player-facing rules:** `ringrift_complete_rules.md`, `ringrift_compact_rules.md`, `ringrift_simple_human_rules.md`
- **TS engine alignment (comments + phase validation):**
  - `src/shared/engine/lpsTracking.ts`
  - `src/shared/engine/playerStateHelpers.ts`
  - `src/shared/engine/phaseValidation.ts`
  - `src/shared/engine/orchestration/turnOrchestrator.ts`
  - `src/shared/engine/aggregates/RecoveryAggregate.ts`
  - `src/shared/types/game.ts`
  - Host comments: `src/server/game/GameEngine.ts`, `src/client/sandbox/ClientSandboxEngine.ts`
- **TS tests updated:** `tests/unit/engine/phaseValidation.test.ts`
- **Derived / analysis / planning docs aligned:** examples include
  - `docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md`
  - `docs/testing/SKIPPED_TESTS_TRIAGE.md`
  - `docs/architecture/AI_ARCHITECTURE.md`
  - `docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md`
  - `docs/supplementary/rules_analysis/RECOVERY_ACTION_DESIGN_AND_IMPLEMENTATION.md`
  - `docs/supplementary/rules_analysis/rules_analysis_recovery_action_interactions.md`
  - `docs/archive/plans/RECOVERY_ACTION_IMPLEMENTATION_PLAN.md`
  - `TODO.md`

---

## Verification Commands

- **TS (focused):**
  - `npm test -- tests/unit/engine/phaseValidation.test.ts`
  - `npm test -- tests/unit/playerStateHelpers.test.ts`
- **Python (focused):**
  - `pytest ai-service/tests/rules/test_recovery.py`
  - `pytest ai-service/tests/parity/test_recovery_parity.py`

---

## Known Follow-ups (Not Addressed Here)

- **TS↔Python contract drift:** Python canonical contract includes move types not present in TS `MoveType` (e.g. `skip_recovery`, `choose_line_option`, `choose_territory_option`). Decide whether to implement on TS side or adjust the canonical contract so both languages match.
- **Legacy vs canonical replay support:** Some TS validation paths still mention backwards-compatibility allowances (e.g., legacy move types during forced elimination). If legacy support remains necessary, make it explicitly opt-in and keep canonical validators strict.

Notes:

- `skip_recovery` exists in TS; current drift is primarily around `choose_territory_option` (canonical + Python) and Python-only aliases like `choose_line_option`.
