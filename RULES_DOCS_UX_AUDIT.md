# RingRift Rules Documentation & UX Audit

**Date:** November 24, 2025
**Auditor:** Documentation and UX Agent
**Scope:** Player-facing text, developer documentation, inline comments, and configuration.

---

## 3.1 Summary

This audit compares the current documentation and user experience text against the authoritative **RR‑CANON** (`RULES_CANONICAL_SPEC.md`) and the actual codebase behaviour.

**Headline Findings:**

1.  **Critical UX Mismatch on Chain Captures:** The GameHUD explicitly tells players they can "end your turn" during chain captures, but RR‑CANON-R103 mandates **mandatory continuation** if any capture is available. This is a high-severity rule misstatement in the UI.
2.  **Victory Condition Misinformation:** The Victory Modal describes Ring Elimination as "eliminating **all** opponent rings," whereas the actual rule (RR‑CANON-R170) and implementation is **>50%** of total rings.
3.  **Canonical Rulebook Drift:** The player-facing `ringrift_complete_rules.md` contains obsolete rules regarding movement landing (Section 10.2) and forced elimination (FAQ Q24) that contradict the canonical spec and current engine behaviour.
4.  **Hidden Implementation Compromises:** Several known engineering compromises (e.g., board repair logic, placement cap approximations) are documented in internal analysis files but missing from the primary developer guides (`RULES_ENGINE_ARCHITECTURE.md` or code comments), creating a risk of future regression.
5.  **Strong Developer Alignment:** Recent engine code (GameEngine, RuleEngine, shared helpers) is exceptionally well-commented with direct references to RR‑CANON sections, reducing the risk of developer misunderstanding.

---

## 3.2 Player-Facing Rules & UX

This section reviews text visible to end-users in the web client and the primary rulebook.

### HUD & In-Game Text

| ID           | Severity   | Location                  | Issue                                                                                                                                                                                                | Remediation                                                                                                 |
| :----------- | :--------- | :------------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------- |
| **DOCUX-P1** | **High**   | `GameHUD.tsx` (PhaseInfo) | **Chain Capture:** Description reads "Continue capturing or **end your turn**". This implies optionality. RR‑CANON-R103 states chain capturing is **mandatory** while legal moves exist.             | Change to: "Continue capturing (Mandatory)". Remove "or end your turn".                                     |
| **DOCUX-P2** | **Medium** | `VictoryModal.tsx`        | **Ring Elimination:** Description reads "Victory by eliminating **all** opponent rings". RR‑CANON-R170 is **>50%** of total rings. Eliminating all is a win, but not the threshold.                  | Change to: "Victory by eliminating >50% of total rings."                                                    |
| **DOCUX-P3** | Low        | `GameHUD.tsx` (PhaseInfo) | **Line Reward:** Description "Choose how to process your line" implies choice exists for _all_ lines. For exact-length lines (RR‑CANON-R122 Case 1), there is no choice (must collapse & eliminate). | Change to: "Process line reward" (generic) or conditionally show "Choose reward" only for overlength lines. |
| **DOCUX-P4** | Low        | `GameHUD.tsx` (SubPhase)  | **Territory:** "Processing disconnected regions" is vague regarding the **self-elimination cost** (RR‑CANON-R145). Players may be surprised they have to lose a ring.                                | Add tooltip or sub-text: "Regions collapse; you must eliminate one outside ring per region."                |

### Public Rulebook (`ringrift_complete_rules.md`)

| ID           | Severity   | Location                             | Issue                                                                                                                                                                                                                                             | Remediation                                                                                 |
| :----------- | :--------- | :----------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------ |
| **DOCUX-P5** | **Medium** | `ringrift_complete_rules.md` §10.2   | **Movement Landing:** States capture landing "differs from non-capture movement in 19×19/Hex", implying non-capture must stop at first valid space. RR‑CANON-R091/R101 unifies this: **all** movement can land on any valid space beyond markers. | Update §10.2 to reflect the Unified Landing Rule (RR‑CANON-R091) matching the Compact Spec. |
| **DOCUX-P6** | Low        | `ringrift_complete_rules.md` FAQ Q24 | **Forced Elimination:** Suggests a player might be unable to perform forced elimination because "all caps have already been eliminated". RR‑CANON-R100/R022 clarifies any stack has a cap.                                                        | Update FAQ Q24 to state forced elimination is always possible if you control a stack.       |
| **DOCUX-P7** | Low        | `ringrift_complete_rules.md` §11.2   | **Line Length:** Uses "4 or 5" loosely.                                                                                                                                                                                                           | Standardize to "Required Length (3 for 8x8, 4 for 19x19/Hex)" to match RR‑CANON-R001.       |

---

## 3.3 Developer Documentation and Comments

This section reviews internal documentation for engineering accuracy.

| ID           | Classification              | Location                                  | Issue                                                                                                                                        | Recommended Action                                                                                             |
| :----------- | :-------------------------- | :---------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- |
| **DOCUX-D1** | **Historical / Superseded** | `deprecated/RULES_GAP_ANALYSIS_REPORT.md` | Describes forced elimination auto-selection as a "divergence". RR‑CANON-R100 now accepts this as a valid implementation choice (CCE-007).    | Ensure file remains in `deprecated/` and is not referenced as current status.                                  |
| **DOCUX-D2** | **Ambiguous**               | `RULES_ENGINE_ARCHITECTURE.md`            | Does not explicitly mention the **Board Repair** logic (`CCE-001`) in `BoardManager.ts`. Developers might assume invariants are strict-only. | Add a "Defensive Invariants" section to `RULES_ENGINE_ARCHITECTURE.md` documenting the repair pass.            |
| **DOCUX-D3** | **Design-intent match**     | `src/server/game/GameEngine.ts`           | Comments reference "automatic consequences" vs "move-driven decision phases".                                                                | Keep. This accurately reflects the transition state described in `RULES_IMPLEMENTATION_MAPPING.md`.            |
| **DOCUX-D4** | **Ambiguous**               | `src/client/sandbox/sandboxTerritory.ts`  | Contains deprecated region search logic alongside new shared delegates.                                                                      | Add `@deprecated` JSDoc tag to the legacy functions to prevent accidental usage.                               |
| **DOCUX-D5** | **Ambiguous**               | `AI_ARCHITECTURE.md`                      | Mentions AI "biasing against Option 2" for lines.                                                                                            | Update to reflect that `HeuristicAI` should eventually support Option 2 logic per `AI_IMPROVEMENT_BACKLOG.md`. |

---

## 3.4 Configuration and Defaults

| ID           | Config Key            | Location                       | Issue                                                                                                                                                | Recommendation                                                                                            |
| :----------- | :-------------------- | :----------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------- |
| **DOCUX-C1** | `RINGRIFT_RULES_MODE` | `src/shared/utils/envFlags.ts` | Defaults to `'ts'`. Docs explain `'python'` and `'shadow'`, but implications for **parity failures** in shadow mode aren't prominent in `README.md`. | Add a "Rules Engine Modes" section to `README.md` explaining that `shadow` logs mismatches but trusts TS. |
| **DOCUX-C2** | `ringsPerPlayer`      | `src/shared/types/game.ts`     | Used for placement caps. Implementation approximates this using stack heights (`CCE-002`).                                                           | Add comment to `BOARD_CONFIGS` in `types/game.ts` noting the implementation approximation (CCE-002).      |

---

## 3.5 Cross-Reference to Known Implementation Compromises

Mapping from `RULES_CONSISTENCY_EDGE_CASES.md` (CCE) to Documentation/UX needs.

| CCE ID      | Description                                                                     | Status       | Doc/UX Action Required                                                                            |
| :---------- | :------------------------------------------------------------------------------ | :----------- | :------------------------------------------------------------------------------------------------ |
| **CCE-001** | **Board Repair:** Overlapping markers are silently deleted.                     | Undocumented | **Create DOCUX-D2**: Document this defensive behaviour in Architecture docs.                      |
| **CCE-002** | **Placement Cap:** Counts total stack height, not specific ring colors.         | Undocumented | **Create DOCUX-C2**: Add comment to `BOARD_CONFIGS` or `PlacementValidator`.                      |
| **CCE-003** | **Sandbox Skip:** Sandbox has no `skip_placement` move; infers phase.           | Undocumented | Add note to `ClientSandboxEngine.ts` header about this divergence from backend.                   |
| **CCE-004** | **Capture Chain Helpers:** Shared helper is a stub; logic lives in GameEngine.  | Documented   | `RULES_IMPLEMENTATION_MAPPING.md` already notes this. No action.                                  |
| **CCE-005** | **Territory Self-Elimination:** "Outside" check is implicit in ordering.        | Undocumented | Add comment to `territoryDecisionHelpers.ts` clarifying that interior stacks are already gone.    |
| **CCE-006** | **Last Player Standing:** Not explicit; relies on game continuing to stalemate. | Undocumented | Update `RULES_ENGINE_ARCHITECTURE.md` to clarify R172 is handled via standard play-to-completion. |
| **CCE-007** | **Forced Elim Heuristic:** Auto-selects smallest cap.                           | Undocumented | Add comment to `TurnEngine.ts` that this is a deterministic tie-break choice.                     |
| **CCE-008** | **Phase Ordering:** Move->Line->Territory->Victory.                             | Documented   | Well-covered in `GameEngine.ts` comments. No action.                                              |
