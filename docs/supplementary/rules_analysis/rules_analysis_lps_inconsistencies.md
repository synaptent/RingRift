# Last Player Standing (LPS) Inconsistency Analysis

This document tracks the Last Player Standing (LPS) victory condition threshold across the codebase and documentation.

**Status:** Canonical docs are now aligned on the **3-consecutive-full-round** requirement. Client-facing copy still needs to be updated to match.

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

### 2. Remaining Inconsistencies

- **`src/client/utils/rulesUxTelemetry.ts`** and **`src/client/components/GameHUD.tsx`/`gameViewModels.ts`**: LPS teaching/victory copy still talks about **two** rounds in tooltips and onboarding strings.
  - Risk: UX copy contradicts the canonical docs and engines; tests snapshot this text, so updates will require test fixture refreshes.

### 2. Codebase Consistency (TypeScript)

- **`src/shared/engine/lpsTracking.ts`**:
  - `export const LPS_REQUIRED_CONSECUTIVE_ROUNDS = 3;`
  - **Consistent**: Matches the canonical spec.

### 3. Codebase Consistency (Python)

- **`ai-service/app/game_engine.py`**:
  - `LPS_REQUIRED_CONSECUTIVE_ROUNDS = 3`
  - **Consistent**: Matches the canonical spec.

## Remediation Plan

The engines and canonical spec agree on **3 rounds**. The remaining gap is UX copy.

- ✅ **Docs aligned:** `ringrift_complete_rules.md`, `ringrift_compact_rules.md`, `ringrift_simple_human_rules.md`, `docs/UX_RULES_COPY_SPEC.md`, and `docs/UX_RULES_WEIRD_STATES_SPEC.md` now all state the three-round requirement.
- ⏳ **Update UX copy:** Refresh LPS text in `src/client/utils/rulesUxTelemetry.ts`, `src/client/adapters/gameViewModels.ts`, and related HUD/onboarding tooltips to say "THREE consecutive complete rounds".
- ⏳ **Refresh tests/snapshots:** Update any snapshot or tooltip tests (`GameHUD` / `VictoryModal`) that assert the old two-round wording once the copy changes land.
