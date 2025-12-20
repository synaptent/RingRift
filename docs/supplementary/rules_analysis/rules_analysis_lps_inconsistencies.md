# Last Player Standing (LPS) Inconsistency Analysis

This document tracks the Last Player Standing (LPS) victory condition threshold across the codebase and documentation.

**Status:** ✅ **RESOLVED** – All documentation and client-facing UX copy are now aligned on the **3-consecutive-full-round** requirement.

## Canonical Specification (SSoT)

According to `RULES_CANONICAL_SPEC.md` (RR-CANON-R172):

- **Requirement**: **3 consecutive full rounds** where the same player is the exclusive real-action holder.
- **Text**: "There exists at least one full round... After the first round completes... on the following round P remains the only player... After the second round completes... on the following round P remains the only player... After the third round completes... P is declared the winner".

## Findings

### 1. Documentation Status

- **Canonical rulebook:** `ringrift_complete_rules.md` §13.3 now explicitly describes the three-round requirement from `RULES_CANONICAL_SPEC.md`.
- **Implementation spec:** `ringrift_compact_rules.md` §7.3 is explicit about three consecutive full rounds with required real actions each round.
- **Human-readable rules:** `ringrift_simple_human_rules.md` §5.3 now mirrors the three-round flow.
- **UX specs:** `docs/UX_RULES_COPY_SPEC.md` and `docs/UX_RULES_WEIRD_STATES_SPEC.md` now both call out the three-round requirement in tooltips/teaching copy.

### 2. UX Copy Status (Updated 2025-12-08)

- ✅ **`src/client/components/TeachingOverlay.tsx`**: Updated to "THREE consecutive complete rounds"
- ✅ **`src/client/adapters/gameViewModels.ts`**: Updated LPS banner copy to "three consecutive full rounds"
- ✅ **`src/shared/teaching/teachingTopics.ts`**: Updated all LPS tip text to "THREE rounds" and "three-round countdown"
- ✅ **`src/shared/teaching/teachingScenarios.ts`**: Updated learning objective to "THREE consecutive rounds"
- ✅ **`src/client/components/GameHUD.tsx`**: Already correct ("three consecutive full rounds" in tooltip)

### 3. Codebase Consistency (TypeScript)

- **`src/shared/engine/lpsTracking.ts`**:
  - `export const LPS_DEFAULT_REQUIRED_ROUNDS = 3;`
  - `export const LPS_REQUIRED_CONSECUTIVE_ROUNDS = LPS_DEFAULT_REQUIRED_ROUNDS;`
  - **Consistent**: Matches the canonical spec (THREE rounds).

### 3. Codebase Consistency (Python)

- **`ai-service/app/rules/mutable_state.py`**:
  - `_lps_rounds_required: int = 3`
  - **Consistent**: Matches the canonical spec (THREE rounds).

## Remediation Plan

✅ **COMPLETE** – All engines, canonical docs, and UX copy now agree on **3 consecutive full rounds**.

- ✅ **Docs aligned:** `ringrift_complete_rules.md`, `ringrift_compact_rules.md`, `ringrift_simple_human_rules.md`, `docs/UX_RULES_COPY_SPEC.md`, and `docs/UX_RULES_WEIRD_STATES_SPEC.md` now all state the three-round requirement.
- ✅ **UX copy updated (2025-12-08):** All LPS teaching tips, tooltips, and victory banners in `TeachingOverlay.tsx`, `gameViewModels.ts`, `teachingTopics.ts`, and `teachingScenarios.ts` now correctly say "THREE consecutive complete rounds".
- ✅ **Code constants updated (2025-12-10):** Both TS and Python engines now use a 3-round LPS threshold (`LPS_DEFAULT_REQUIRED_ROUNDS = 3`, `lps_rounds_required = 3`).
