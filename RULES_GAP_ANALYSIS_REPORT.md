# RingRift Rules Gap Analysis Report

**Date:** November 21, 2025
**Analyst:** Architect Mode
**Scope:** Backend Game Engine (`src/shared/engine/`, `src/server/game/`)

## Executive Summary

The RingRift backend implementation is highly compliant with the written rules (`ringrift_complete_rules.md` and `ringrift_compact_rules.md`). The core mechanics of movement, capture, line formation, and territory disconnection are implemented with high fidelity.

One minor divergence was identified regarding the **Forced Elimination** mechanic: while the engine correctly enforces the elimination when a player is blocked, it currently automates the selection of the stack to eliminate from, whereas the rules state the player should choose.

## Detailed Analysis

### 1. Board & Adjacency

- **Rule:** Square boards use Moore (8-way) adjacency for movement/lines and Von Neumann (4-way) for territory. Hex boards use 6-way for all.
- **Implementation:** `src/server/game/BoardManager.ts` correctly implements `getMooreNeighbors`, `getVonNeumannNeighbors`, and `getHexagonalNeighbors`. `RuleEngine.ts` uses the correct adjacency for each operation.
- **Status:** ✅ **Compliant**

### 2. Movement

- **Rule:** Move distance >= stack height. Can land on any valid space beyond markers (Unified Landing Rule).
- **Implementation:** `src/server/game/RuleEngine.ts` (`validateStackMovement`) and `src/shared/engine/core.ts` (`validateCaptureSegmentOnBoard`) correctly enforce distance and path constraints.
- **Status:** ✅ **Compliant**

### 3. Ring Placement

- **Rule:** Optional if rings on board + valid moves. Mandatory if no rings on board OR no valid moves. "No-dead-placement" rule (must leave at least one legal move).
- **Implementation:** `src/server/game/turn/TurnEngine.ts` (`advanceGameForCurrentPlayer`) correctly determines if placement is mandatory or optional. `src/server/game/rules/placementHelpers.ts` (`hasAnyLegalMoveOrCaptureFrom`) correctly implements the no-dead-placement check.
- **Status:** ✅ **Compliant**

### 4. Overtaking Captures

- **Rule:** Cap height >= target cap height. Mandatory chain captures. Player chooses path.
- **Implementation:** `src/server/game/rules/captureChainEngine.ts` manages the chain state. `GameEngine.ts` enforces mandatory continuation. `getCaptureOptionsFromPosition` correctly enumerates valid options.
- **Status:** ✅ **Compliant**

### 5. Line Formation & Rewards

- **Rule:** Lines of 4+ (8x8) or 5+ (19x19/Hex). Graduated rewards for overlength lines (Option 1: Collapse All + Eliminate, Option 2: Collapse Min + No Eliminate).
- **Implementation:** `src/server/game/rules/lineProcessing.ts` correctly identifies lines and implements the Option 1/2 logic. It supports `interactionManager` for player choice.
- **Status:** ✅ **Compliant**

### 6. Territory Disconnection

- **Rule:** Region disconnected if physically surrounded AND lacks representation from at least one _active_ player. Self-elimination prerequisite (must have stack outside region).
- **Implementation:** `src/server/game/BoardManager.ts` (`findDisconnectedRegions`) correctly checks for active player representation. `src/server/game/rules/territoryProcessing.ts` (`canProcessDisconnectedRegion`) correctly enforces the self-elimination prerequisite.
- **Status:** ✅ **Compliant**

### 7. Forced Elimination

- **Rule:** If a player has stacks but no legal moves/placements, they _must_ eliminate the cap of one of their stacks. The player _chooses_ which stack.
- **Implementation:** `src/server/game/turn/TurnEngine.ts` (`processForcedElimination`) correctly detects the blocked state and triggers elimination.
- **Divergence:** The current implementation automatically eliminates from the first available stack (`playerStacks[0]`) instead of offering a choice to the player.
  - _Code Reference:_ `src/server/game/turn/TurnEngine.ts:431`
  - _Impact:_ Minor strategic impact (player cannot optimize which stack to sacrifice), but ensures game progress.
- **Status:** ⚠️ **Divergent (Minor)**

### 8. Victory Conditions & Stalemate

- **Rule:** >50% Rings Eliminated, >50% Territory, or Last Player Standing. Stalemate Tiebreakers: Territory > Rings > Markers > Last Actor.
- **Implementation:** `src/server/game/RuleEngine.ts` (`checkGameEnd`) implements all victory conditions and the exact tiebreaker order specified in the rules.
- **Status:** ✅ **Compliant**

## Action Items

1.  **Refactor Forced Elimination:** Update `TurnEngine.ts` and `GameEngine.ts` to support a `ForcedEliminationChoice` interaction, allowing the player to select which stack to eliminate when blocked. This will align the implementation fully with the rules and remove the current auto-selection simplification.
2.  **Verify "Active Player" Definition:** Ensure `BoardManager.ts`'s definition of "active player" (players with stacks on board) aligns perfectly with edge cases where a player might have 0 stacks but rings in hand (though such a player wouldn't provide representation anyway, so this is likely correct).
3.  **Track this divergence explicitly:** Add a small P1 rules/engine‑parity entry to [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md:1) and/or [`TODO.md`](TODO.md:26) so that the forced‑elimination choice gap is visible alongside other parity and termination work.

> **OPEN QUESTION:** The written rules state that the player chooses which stack’s cap to eliminate when blocked, but the current backend implementation auto‑selects the first available stack. Maintainers should decide whether to treat the rules spec as authoritative (and update the engine to surface a `ForcedEliminationChoice`) or to intentionally relax the rules and update [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1) accordingly. Until that decision is made, treat this as a known divergence and avoid relying on the auto‑selection behaviour in long‑term UI copy or AI heuristics.

## Conclusion

The RingRift backend is in excellent shape. The only identified gap is the automation of the Forced Elimination choice, which is a known simplification. The complex rules around territory disconnection and chain captures are implemented with impressive fidelity.
