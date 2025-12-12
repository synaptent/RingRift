# Recovery Action Design and Implementation Plan

> **Doc Status (2025-12-08): Active (design document)**
>
> **Purpose:** Document the design rationale, rules analysis, and implementation plan for the Recovery Action mechanic.
>
> **Canonical Source:** Once implemented, canonical rules will be in `RULES_CANONICAL_SPEC.md` R110–R115

---

## 1. Executive Summary

The **Recovery Action** (also called "Marker Slide") is a proposed rule addition that allows temporarily eliminated players to remain active by sliding markers to form lines, paying the cost with buried rings. This document captures the design discussion, rules analysis, goal alignment assessment, and implementation plan.

### Key Decision

The Recovery Action **has been designed and partially implemented** on the `sandbox-replay-refactor` branch (commit `c0b66e2e`). This document serves to:

1. Record the design rationale and rules analysis
2. Track implementation artifacts across branches
3. Guide the integration into the main branch

---

## 2. Problem Statement

### Current State (Without Recovery)

When a player loses all their stacks and has no rings in hand, they become **temporarily inactive**:

- They cannot place (no rings in hand)
- They cannot move (no stacks)
- They cannot capture (no stacks)
- Their only path back is **passive recovery**: waiting for opponents to expose their buried rings

**Issues with Current State:**

1. **Dead agency**: Players with markers and buried rings have no way to influence the game
2. **Anticlimactic elimination**: A player can be "effectively dead" while still having material on the board
3. **LPS becomes easier**: Dominant players can secure LPS victory without truly eliminating opponent agency
4. **Buried rings are wasted**: Rings captured into opponent stacks become purely passive

### Designer Intent

Per the game designer (conversation 2025-12-08):

> "The desired additional mechanism is as follows: if a player has no legal ring placements, moves or captures, then buried rings become a potential resource for them... they may move a marker owned by them to an adjacent empty space if it forms a line, and if they can pay for the line reward/collapse with a buried ring of theirs or a ring from the cap of a stack they control."

---

## 3. Recovery Action Rules Specification

### RR-CANON-R110: Recovery Eligibility

A player P is **eligible for recovery** if ALL of the following hold:

1. P controls **no stacks** on the board (no stack has P's color as top ring)
2. P owns at least one **marker** on the board
3. P has at least one **buried ring** (a ring of P's color inside any stack, but not as the top ring)

**Note:** Rings in hand do **NOT** prevent recovery eligibility. A player with rings in hand but no stacks may still use recovery if the above conditions are met.

### RR-CANON-R111: Marker Slide Adjacency

When eligible, P may slide one of their markers to an **adjacent empty cell**:

- **Square boards (8×8, 19×19):** Moore adjacency (8 directions)
- **Hexagonal board:** Hex adjacency (6 directions)

The destination cell must be:

- Within board bounds
- Empty (no stack, no marker, not collapsed territory)

### RR-CANON-R112: Success Criteria

A marker slide is legal if **either** of these conditions is satisfied:

- **(a) Line formation:** Completes a line of **at least** `lineLength` consecutive markers of P's color
- **(b) Fallback repositioning:** If no slide satisfies (a), any adjacent slide is permitted (including slides that cause territory disconnection)

**Note:** Territory disconnection may occur as a side effect of any recovery slide (line-forming or fallback).

**Skip option:** P may elect to skip recovery entirely, preserving buried rings for a future turn.

**Line length requirements:**

- `lineLength = 3` for 8×8 (3-4 player) or as configured
- `lineLength = 4` for 19×19, hexagonal, and 8×8 (2-player)

**Line recovery (condition a):** The slid marker, in its new position, must participate in the completed line. Overlength lines are permitted. When an overlength line is formed, the player chooses between:

- **Option 1:** Collapse all markers in the line to territory and pay the self-elimination cost (one buried ring extraction).
- **Option 2:** Collapse exactly `lineLength` consecutive markers of the player's choice to territory **without** paying any self-elimination cost. The remaining markers stay on the board.

This mirrors normal line reward semantics (RR-CANON-R130–R134).

**Fallback recovery (condition b):** If no line-forming slide exists, P may slide any marker to an adjacent empty cell, including slides that cause territory disconnection. This costs one buried ring extraction but does not trigger line processing.

### RR-CANON-R113: Buried Ring Extraction Cost

The cost of a recovery slide depends on the chosen option:

- **Exact-length lines:** Cost = 1 buried ring extraction
- **Overlength lines with Option 1:** Cost = 1 buried ring extraction
- **Overlength lines with Option 2:** Cost = 0 (no extraction required)

If P has no buried rings and must pay a cost, that action is **illegal**.

### RR-CANON-R114: Cascade Processing

After a recovery slide:

1. **Line collapse:**
   - **Exact-length or Option 1:** All markers in the formed line become collapsed spaces (territory) owned by P
   - **Option 2:** Exactly `lineLength` consecutive markers chosen by P become collapsed spaces; remaining markers stay on board
2. **Buried ring extraction:** For exact-length lines or Option 1, one buried ring is removed and credited to P's `eliminatedRingsTotal`. Option 2 has no extraction cost.
3. **Territory processing:** Check for disconnected regions created by the collapse; process per standard territory rules (RR-CANON-R140–R145)
4. **Victory check:** After all processing, check victory conditions

### RR-CANON-R115: Recording Semantics

Recovery slides are recorded as `recovery_slide` moves:

```typescript
type RecoverySlideMove = {
  type: 'recovery_slide';
  player: number;
  from: Position; // Source marker position
  to: Position; // Adjacent destination
  option?: 1 | 2; // For overlength lines: 1 (collapse all) or 2 (collapse min)
  collapsedMarkers?: Position[]; // For Option 2: which markers to collapse
  extractionStack?: string; // Stack for buried ring extraction (if applicable)
  formedLines: LineInfo[];
};
```

**LPS Classification:** Recovery slides are **NOT "real actions"** for Last Player Standing purposes. Like forced elimination, recovery is a survival mechanism that does not reset the LPS countdown.

---

## 4. Design Goal Alignment Analysis

### Goal 1: High Emergent Complexity from Simple Rules

| Criterion           | Assessment                                                                        |
| ------------------- | --------------------------------------------------------------------------------- |
| Rule simplicity     | **MODERATE** - Adds conditional action using existing mechanics                   |
| Interaction depth   | **EXCELLENT** - Markers become offensive/defensive; buried rings become resources |
| Avoiding rule bloat | **GOOD** - Thematically consistent (markers = influence projection)               |

**Verdict:** Adds depth through interaction, not through rule complexity.

### Goal 2: Exciting, Tense, and Strategically Non-Trivial Games

| Criterion           | Assessment                                                                 |
| ------------------- | -------------------------------------------------------------------------- |
| Comeback potential  | **EXCELLENT** - "Eliminated" players can still threaten lines/territory    |
| Tension maintenance | **EXCELLENT** - Cannot safely ignore opponents with markers + buried rings |
| LPS impact          | **POSITIVE** - LPS requires truly eliminating agency, not just stacks      |
| Alliance dynamics   | **POSITIVE** - "Zombie" players remain politically relevant                |

**Verdict:** Strongly aligns with design goals by maintaining tension and comeback potential.

### Goal 3: Human–Computer Competitive Balance

| Criterion           | Assessment                                                         |
| ------------------- | ------------------------------------------------------------------ |
| Pattern recognition | **POSITIVE** - Humans can intuitively spot line opportunities      |
| Branching factor    | **MODERATE** - Limited additional moves (only when blocked)        |
| Social dynamics     | **EXCELLENT** - Maintains social relevance of "eliminated" players |

**Verdict:** Favors human players through positional intuition and social relevance.

---

## 5. System Interaction Analysis

### 5.1 Last Player Standing (LPS)

**Canonical:** Recovery is **NOT** a real action for LPS purposes (RR-CANON-R172).

| Scenario                             | Without Recovery   | With Recovery                        |
| ------------------------------------ | ------------------ | ------------------------------------ |
| B has only FE available              | B cannot block LPS | B cannot block LPS (FE ≠ real)       |
| B has valid recovery slide           | N/A                | B cannot block LPS (recovery ≠ real) |
| Only A has real actions for 2 rounds | A wins LPS         | A wins LPS (recovery doesn't count)  |

**Implication:** LPS evaluation must ignore recovery (and forced elimination) when determining real actions; recovery still matters for global-legal-action / ANM classification and turn rotation.

### 5.2 Forced Elimination (FE)

**Mutual Exclusivity:** FE and Recovery are mutually exclusive:

- FE requires: controlling ≥1 stack
- Recovery requires: controlling 0 stacks

A player transitions from FE eligibility to Recovery eligibility when their last stack is eliminated.

### 5.3 ANM (Active-No-Moves)

Updated global legal action set:

```
Global Legal Actions = {
  placements,
  movements,
  captures,
  forced_elimination (if has stacks, no other moves),
  recovery_slides (if eligible AND valid slide exists),
  line_decisions,
  territory_decisions
}
```

### 5.4 Turn Rotation

Updated skip logic must check:

1. Has stacks? → Don't skip
2. Has rings in hand? → Don't skip
3. Eligible for recovery AND has valid slides? → Don't skip
4. None of the above → Skip

---

## 6. Implementation Status

### 6.1 Artifacts on `sandbox-replay-refactor` Branch

| File                                                                               | Lines | Description          |
| ---------------------------------------------------------------------------------- | ----- | -------------------- |
| `src/shared/engine/aggregates/RecoveryAggregate.ts`                                | 807   | TS domain aggregate  |
| `ai-service/app/rules/recovery.py`                                                 | 686   | Python rules module  |
| `ai-service/app/rules/validators/recovery.py`                                      | 42    | Python validator     |
| `ai-service/app/rules/mutators/recovery.py`                                        | 32    | Python mutator       |
| `tests/unit/RecoveryAggregate.shared.test.ts`                                      | 495   | TS unit tests        |
| `ai-service/tests/rules/test_recovery.py`                                          | 437   | Python tests         |
| `docs/supplementary/rules_analysis/rules_analysis_recovery_action_interactions.md` | 374   | Interaction analysis |

**Commit:** `c0b66e2e` (2025-12-08)
**Branch:** `remotes/origin/sandbox-replay-refactor`

### 6.2 Integration Requirements

To bring Recovery into main:

1. Cherry-pick or merge relevant commits from `sandbox-replay-refactor`
2. Update `RULES_CANONICAL_SPEC.md` with R110–R115
3. Update `ringrift_complete_rules.md` §4 with Recovery Action section
4. Update `ringrift_compact_rules.md` §2 with formal specification
5. Add `recovery_slide` to `MoveType` in `src/shared/types/game.ts`
6. Integrate with turn orchestrator phase machine
7. Add contract vectors for parity testing
8. Ensure LPS evaluation does **not** count recovery as a real action (RR-CANON-R172)

---

## 7. Open Questions

### 7.1 Resolved

1. **Q: Is recovery a "real action" for LPS?**
   A: No - recovery slides do **not** count as real actions for LPS (RR-CANON-R172).

2. **Q: Can overlength lines be formed?**
   A: Yes - overlength lines are permitted; the canonical cost model is Option 1/Option 2 (Option 1 costs 1 buried ring extraction, Option 2 is free and only available for overlength).

3. **Q: What happens if collapse creates disconnected regions?**
   A: Standard territory processing applies; buried rings can pay for self-elimination.

### 7.2 To Confirm

1. **Extraction stack choice:** When multiple stacks contain buried rings, does the player choose which to extract from?
   - Current implementation: Yes, player specifies `extractionStacks[]`

2. **Cascading recovery:** If recovery collapse exhumes a ring (via territory processing), can the player immediately place it?
   - Tentative: No, turn ends after recovery processing

3. **Multiple lines:** If one slide completes multiple lines, are all processed?
   - Tentative: Yes, all formed lines are processed (higher total cost)

---

## 8. References

- **Design conversation:** 2025-12-08 discussion with game designer
- **Prior implementation:** `sandbox-replay-refactor` branch, commit `c0b66e2e`
- **Canonical rules:** `RULES_CANONICAL_SPEC.md` (pending R110–R115 addition)
- **Complete rules:** `ringrift_complete_rules.md` (pending §4.5 addition)
- **LPS analysis:** `docs/supplementary/rules_analysis/rules_analysis_lps_fe.md`

---

## 9. Recoverable Artifacts from Git History

### 9.1 Source Branch

**Branch:** `remotes/origin/sandbox-replay-refactor`
**Divergence Point:** Commit `77248883` (common ancestor with main)
**Recovery-specific Commit:** `c0b66e2e` (2025-12-08 14:49:58)

### 9.2 Recovery-Specific Files to Cherry-Pick

The following files contain recovery action implementation and should be extracted:

#### TypeScript Implementation

| File                                                | Lines | Purpose                                                    |
| --------------------------------------------------- | ----- | ---------------------------------------------------------- |
| `src/shared/engine/aggregates/RecoveryAggregate.ts` | 807   | Domain aggregate with enumeration, validation, application |
| `tests/unit/RecoveryAggregate.shared.test.ts`       | 495   | Unit tests for TS aggregate                                |

#### Python Implementation

| File                                          | Lines | Purpose                             |
| --------------------------------------------- | ----- | ----------------------------------- |
| `ai-service/app/rules/recovery.py`            | 686   | Core recovery rules and enumeration |
| `ai-service/app/rules/validators/recovery.py` | 42    | Move validation                     |
| `ai-service/app/rules/mutators/recovery.py`   | 32    | State mutation                      |
| `ai-service/tests/rules/test_recovery.py`     | 437   | Python unit tests                   |

#### Documentation

| File                                                                               | Lines | Purpose                     |
| ---------------------------------------------------------------------------------- | ----- | --------------------------- |
| `docs/supplementary/rules_analysis/rules_analysis_recovery_action_interactions.md` | 374   | System interaction analysis |

#### Supporting Changes (may need review)

| File                                        | Change Type | Notes                      |
| ------------------------------------------- | ----------- | -------------------------- |
| `src/shared/engine/fsm/FSMAdapter.ts`       | Modified    | Recovery phase integration |
| `src/shared/engine/fsm/TurnStateMachine.ts` | Modified    | Recovery state transitions |
| `ai-service/app/game_engine.py`             | Modified    | Python engine integration  |
| `tests/unit/fsm/TurnStateMachine.test.ts`   | Modified    | FSM tests for recovery     |
| `tests/utils/fixtures.ts`                   | Modified    | Test fixtures              |

### 9.3 Additional Valuable Artifacts (Non-Recovery)

The `sandbox-replay-refactor` branch also contains:

#### Tournament System (AI Evaluation)

- `ai-service/app/tournament/__init__.py`
- `ai-service/app/tournament/agents.py` (216 lines)
- `ai-service/app/tournament/elo.py` (356 lines)
- `ai-service/app/tournament/runner.py` (466 lines)
- `ai-service/app/tournament/scheduler.py` (402 lines)
- `ai-service/scripts/run_tournament.py` (272 lines)

#### Load Test Infrastructure

- `tests/load/configs/bcap-scenarios.json`
- `tests/load/scripts/compare-runs.js`
- `tests/load/scripts/run-ai-heavy.sh`

#### AI Improvements

- `ai-service/app/ai/evaluation_provider.py`
- `ai-service/app/ai/move_ordering.py`
- `ai-service/app/ai/swap_evaluation.py`

### 9.4 Integration Commands

To extract recovery-specific files:

```bash
# Create a patch file for recovery-specific changes
git format-patch -1 c0b66e2e --stdout -- \
  src/shared/engine/aggregates/RecoveryAggregate.ts \
  ai-service/app/rules/recovery.py \
  ai-service/app/rules/validators/recovery.py \
  ai-service/app/rules/mutators/recovery.py \
  ai-service/tests/rules/test_recovery.py \
  tests/unit/RecoveryAggregate.shared.test.ts \
  docs/supplementary/rules_analysis/rules_analysis_recovery_action_interactions.md \
  > recovery-feature.patch

# Apply the patch to main
git checkout main
git apply recovery-feature.patch
```

Alternatively, cherry-pick the entire commit:

```bash
git checkout main
git cherry-pick c0b66e2e
# Resolve any conflicts
```

### 9.5 Integration Steps

1. **Extract recovery files** from commit `c0b66e2e`
2. **Review FSM changes** in `TurnStateMachine.ts` and `FSMAdapter.ts`
3. **Add `recovery_slide` to MoveType** in `src/shared/types/game.ts`
4. **Update turn orchestrator** to handle recovery phase
5. **Add recovery eligibility check** to turn rotation logic
6. **Update LPS evaluation** to check for valid recovery moves
7. **Add canonical rules** (R110–R115) to `RULES_CANONICAL_SPEC.md`
8. **Add narrative rules** to `ringrift_complete_rules.md` §4.5
9. **Run parity tests** to validate TS↔Python alignment
10. **Add contract vectors** for recovery scenarios

---

## 10. Comparison: Branch Implementation vs Discussion Proposal

### 10.1 Key Differences

| Aspect                  | Branch Implementation              | Discussion Proposal                | Resolution               |
| ----------------------- | ---------------------------------- | ---------------------------------- | ------------------------ |
| **Eligibility**         | NO stacks, NO rings in hand        | No legal placements/moves/captures | **Use branch** (simpler) |
| **Payment source**      | Buried rings only                  | Buried rings OR cap rings          | **Use branch** (simpler) |
| **Overlength handling** | Entire line collapses, cost scales | Entire line collapses, cost scales | **Same**                 |
| **LPS status**          | IS a real action                   | IS a real action                   | **Same**                 |
| **Territory cascade**   | Yes, with buried ring self-elim    | Yes, with buried ring self-elim    | **Same**                 |

### 10.2 Analysis: Eligibility Scope

**Branch (Stricter):** Recovery only for players with:

- Zero controlled stacks
- Zero rings in hand
- Has markers + buried rings

**Discussion (Broader):** Recovery for players with:

- No legal placements, moves, or captures
- Could include players with trapped/immobile stacks
- Cap rings could pay for recovery

**Recommendation: Use Branch (Stricter) Approach**

Rationale:

1. **Simpler state detection:** "No stacks AND no rings" is easier to check than "no legal moves of any kind"
2. **Cleaner conceptual model:** Recovery is for "zombie" players who have lost all direct agency
3. **Avoids FE/Recovery overlap:** If player has stacks but can't move, they use FE first
4. **Consistent with existing rules:** Players with stacks have a clear path (FE loop until no stacks)

### 10.3 Analysis: Payment Source

**Branch:** Only buried rings can pay for recovery

**Discussion:** Buried rings OR cap rings from controlled stacks

**Recommendation: Use Branch (Buried Rings Only) Approach**

Rationale:

1. **Eligibility conflict:** If player controls stacks (needed for cap rings), they're not eligible for recovery anyway
2. **Cleaner narrative:** "Use the rings opponents captured from you" is more thematic
3. **No new complexity:** Don't need to track "can this player pay with cap OR buried"

### 10.4 Final Unified Rule Design

**RR-CANON-R110: Recovery Eligibility**
A player P is eligible for recovery if ALL hold:

1. P controls **zero stacks** on the board
2. P has **zero rings in hand**
3. P owns at least one **marker** on the board
4. P has at least one **buried ring** (P's color inside any stack, not as top ring)

**RR-CANON-R111: Marker Slide**
Slide one marker to an adjacent empty cell (Moore for square, hex-adjacent for hex).

**RR-CANON-R112: Success Criteria**
Slide is legal if **either**: (a) completes a line of ≥ `lineLength` consecutive markers, OR (b) if no line-forming slide exists, any adjacent slide is permitted (fallback). Note: Territory disconnection may occur as a side effect of any recovery slide (line-forming or fallback).

**RR-CANON-R113: Buried Ring Cost**
For line formation: Cost = 1 + max(0, actualLineLength - lineLength). Fallback slides cost 1 buried ring.

**RR-CANON-R114: Cascade Processing**
For line formation: collapse line → extract buried ring(s) → territory cascade → victory check. Fallback slides only extract 1 buried ring (no line collapse).

**RR-CANON-R115: Recording & LPS**

- Move type: `recovery_slide`
- Recovery **IS a real action** for LPS purposes

### 10.5 Simplifications Over Initial Discussion

1. **Removed cap ring payment** - Redundant since eligible players have no stacks
2. **Strict eligibility** - No stacks AND no rings, not just "no legal actions"
3. **Single payment type** - Always buried rings, no choice complexity
4. **Clear FE→Recovery transition** - FE loop exhausts stacks, then recovery becomes available

### 10.6 Consistency with Game Themes

| Theme                        | How Recovery Aligns                          |
| ---------------------------- | -------------------------------------------- |
| **Buried rings as resource** | Captured material remains valuable           |
| **Markers as influence**     | Marker network enables comeback              |
| **No dead agency**           | Players with material can always act         |
| **Tension maintenance**      | "Eliminated" players remain threats          |
| **LPS difficulty**           | Must truly eliminate agency, not just stacks |

---

## 11. Revision History

| Date       | Author      | Changes                                     |
| ---------- | ----------- | ------------------------------------------- |
| 2025-12-08 | Claude Code | Initial document from design discussion     |
| 2025-12-08 | Claude Code | Added git artifact recovery analysis (§9)   |
| 2025-12-08 | Claude Code | Added comparison and unified proposal (§10) |
