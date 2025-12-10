# Recovery Action Implementation Plan

> **Doc Status (2025-12-10): Active – P0, P1, P2, P3 Complete**
>
> **Purpose:** Complete implementation plan for the Recovery Action rule feature.
>
> **Canonical Source:** `RULES_CANONICAL_SPEC.md` §5.4 (RR-CANON-R110–R115)
>
> **Progress (2025-12-08):** All P0 (Critical Path), P1 (High Priority), P2 (UI, Teaching), and P3 (AI, Testing) tasks completed.

---

## Executive Summary

The **Recovery Action** allows temporarily eliminated players to remain active by sliding markers to form lines, paying costs with buried ring extraction. **All P0, P1, P2, and P3 tasks are complete.**

### Implementation Status Overview

| Component                       | Status      | Notes                                           |
| ------------------------------- | ----------- | ----------------------------------------------- |
| Rules Specification             | ✅ Complete | RR-CANON-R110–R115 in `RULES_CANONICAL_SPEC.md` |
| TS RecoveryAggregate            | ✅ Complete | Option 1/2 cost model implemented               |
| Python recovery.py              | ✅ Complete | Option 1/2 cost model implemented               |
| Turn Orchestrator Integration   | ✅ Complete | Wired into movement phase                       |
| GameEngine Integration (Python) | ✅ Complete | `get_valid_moves()` includes recovery           |
| LPS Integration                 | ✅ Complete | Recovery counted as real action                 |
| Contract Vectors                | ✅ Complete | `recovery_action.vectors.json` created          |
| Parity Tests                    | ✅ Complete | `test_recovery_parity.py` (15 tests)            |
| FSM Event Mapping               | ✅ Complete | `TurnStateMachine.ts`, `FSMAdapter.ts`          |
| RuleEngine Validation           | ✅ Complete | `RuleEngine.ts` validation case                 |
| Move Notation                   | ✅ Complete | `notation.ts` with `Rv` prefix                  |
| Unit Tests (Option 1/2)         | ✅ Complete | `RecoveryAggregate.shared.test.ts` (35 tests)   |
| UI Components                   | ✅ Complete | Highlighting, selection, overlength dialog      |
| Teaching Materials              | ✅ Complete | 7 tips in teachingTopics.ts                     |
| AI Heuristic Evaluation         | ✅ Complete | `heuristic_ai.py` recovery potential weights    |
| Extended Parity Tests           | ✅ Complete | `test_recovery_parity.py` (20 tests)            |
| Backend LPS Wiring              | ✅ Complete | `TurnEngine.ts` calls `evaluateLpsVictory`      |

---

## 1. Rule Summary (RR-CANON-R110–R115)

### 1.1 Eligibility (RR-CANON-R110)

A player P is eligible for recovery if **ALL** conditions hold:

1. P controls **no stacks** on the board
2. P has **zero rings in hand** (`ringsInHand[P] == 0`)
3. P has **at least one marker** on the board
4. P has **at least one buried ring** (their ring at a non-top position in some stack)

### 1.2 Marker Slide (RR-CANON-R111)

- Move one marker to an **adjacent empty cell**
- Square boards: Moore neighborhood (8 directions)
- Hexagonal boards: Hex-adjacency (6 directions)
- Destination must be: valid, empty (no stack/marker), not collapsed

### 1.3 Success Criteria (RR-CANON-R112) – **UPDATED**

The slide is legal if **either**: (a) completes a line of **at least `lineLength`** consecutive markers, OR (b) if no line-forming slide exists, any adjacent slide is permitted (fallback). Note: Territory disconnection may occur as a side effect of any recovery slide (line-forming or fallback).

**Line length requirements:**

| Board Type | Players | `lineLength` |
| ---------- | ------- | ------------ |
| square8    | 2       | 4            |
| square8    | 3-4     | 3            |
| square19   | any     | 4            |
| hexagonal  | any     | 4            |

**Overlength lines ARE permitted** with Option 1/Option 2 semantics:

- **Option 1:** Collapse all markers → pay 1 buried ring extraction
- **Option 2:** Collapse exactly `lineLength` markers of choice → pay 0

### 1.4 Buried Ring Extraction (RR-CANON-R113) – **UPDATED**

| Recovery Type         | Cost                     |
| --------------------- | ------------------------ |
| Exact length line     | 1 buried ring extraction |
| Overlength + Option 1 | 1 buried ring extraction |
| Overlength + Option 2 | 0 (no extraction)        |
| Fallback slide        | 1 buried ring extraction |

Extraction process:

1. Select any stack containing at least one of your buried rings
2. Remove your **bottommost** ring from that stack
3. Ring is eliminated and credited to your eliminated rings total
4. Stack height decreases by 1; control determined by new top ring

### 1.5 LPS Classification

Recovery is a **"real action"** for Last Player Standing purposes (RR-CANON-R172).

---

## 2. Implementation Phases

### Phase 0: Pre-requisites

| Task                                             | Priority | File(s)                     | Status  |
| ------------------------------------------------ | -------- | --------------------------- | ------- |
| Verify player-count dependent `lineLength` works | P0       | `rulesConfig.ts`, `core.py` | ✅ Done |
| Confirm `recovery_slide` in MoveType enum        | P0       | `game.ts`, `core.py`        | ✅ Done |

### Phase 1: Update Cost Model (Option 1/Option 2)

**Critical:** The existing implementation uses a graduated cost model (`1 + overlength`). This must be updated to match the new rules.

#### 1.1 TypeScript: RecoveryAggregate.ts

**File:** `src/shared/engine/aggregates/RecoveryAggregate.ts`

```typescript
// OLD (remove):
const cost = 1 + Math.max(0, formedLineLength - lineLength);

// NEW: Cost depends on option choice
// For enumeration, we only need to check if recovery is possible
// Option 2 always costs 0 for overlength, Option 1 costs 1
// Player must have at least 1 buried ring for exact-length lines
// but can always do Option 2 (cost 0) for overlength lines
```

**Changes needed:**

1. Update `RecoverySlideTarget` interface:

   ```typescript
   interface RecoverySlideTarget {
     from: Position;
     to: Position;
     formedLineLength: number;
     isOverlength: boolean;
     /** For exact-length or Option 1: 1. For Option 2: 0 */
     option1Cost: number;
     option2Cost: number;
   }
   ```

2. Update `enumerateRecoverySlideTargets()` to return overlength-eligible moves even when player has 0 buried rings (Option 2 is free)

3. Add `RecoverySlideMove` discriminated type with `option: 1 | 2` field for overlength lines

4. Update `applyRecoverySlide()` to handle Option 1 vs Option 2 collapse semantics

#### 1.2 Python: recovery.py

**File:** `ai-service/app/rules/recovery.py`

Mirror the TypeScript changes:

1. Update `RecoverySlideTarget` dataclass
2. Update enumeration logic
3. Update application to support Option 1/2

**File:** `ai-service/app/rules/mutators/recovery.py`

- Update mutator to handle option parameter

**File:** `ai-service/app/rules/validators/recovery.py`

- Update validator to accept overlength moves

---

### Phase 2: Integration into Move Generation

#### 2.1 TypeScript Turn Orchestrator

**File:** `src/shared/engine/orchestration/turnOrchestrator.ts`

In `getValidMovesForPhase()` for MOVEMENT phase:

```typescript
// In movement phase enumeration
case 'movement': {
  const movementMoves = enumerateSimpleMovesForPlayer(state, player);
  const captureMoves = enumerateAllCaptureMoves(state, player);

  // NEW: Add recovery moves if eligible
  const recoveryMoves = enumerateRecoverySlides(state, player);

  return [...movementMoves, ...captureMoves, ...recoveryMoves];
}
```

#### 2.2 Python GameEngine

**File:** `ai-service/app/game_engine.py`

In `get_valid_moves()` method, add recovery integration:

```python
def get_valid_moves(self, game_state: GameState, player_number: int) -> List[Move]:
    phase = game_state.current_phase

    if phase == GamePhase.MOVEMENT:
        movement_moves = self._get_movement_moves(game_state, player_number)
        capture_moves = self._get_capture_moves(game_state, player_number)
        # NEW: Add recovery moves
        recovery_moves = self._get_recovery_moves(game_state, player_number)
        return movement_moves + capture_moves + recovery_moves
    # ... rest of phases
```

Add the helper method:

```python
def _get_recovery_moves(self, game_state: GameState, player_number: int) -> List[Move]:
    """Get recovery moves for an eligible player (RR-CANON-R110–R115)."""
    from app.rules.recovery import enumerate_recovery_moves
    return enumerate_recovery_moves(game_state, player_number)
```

---

### Phase 3: Move Application & Dispatching

#### 3.1 TypeScript FSM Integration

**File:** `src/shared/engine/fsm/FSMAdapter.ts`

Add recovery event mapping in `moveToEvent()`:

```typescript
case 'recovery_slide':
  return { type: 'RECOVERY_SLIDE', move };
```

Update `isMoveTypeValidForPhase()`:

```typescript
case 'movement':
  return ['movement', 'overtaking_capture', 'recovery_slide'].includes(moveType);
```

#### 3.2 TypeScript RuleEngine

**File:** `src/server/game/RuleEngine.ts`

Add validation case in `validateMove()`:

```typescript
case 'recovery_slide':
  return this.validateRecoverySlide(move, gameState);
```

#### 3.3 Move Notation

**File:** `src/shared/engine/notation.ts`

Add recovery notation:

```typescript
case 'recovery_slide': {
  const optionSuffix = move.option === 2 ? '(min)' : '';
  return `R${posToNotation(move.from)}-${posToNotation(move.to)}${optionSuffix}`;
}
// Example: "Ra3-a4" or "Ra3-a4(min)" for Option 2
```

---

### Phase 4: LPS Integration (Critical)

Recovery action is a **"real action"** for Last Player Standing. This is essential for correct game termination.

#### 4.1 TypeScript playerStateHelpers

**File:** `src/shared/engine/playerStateHelpers.ts`

Update `hasAnyRealAction()`:

```typescript
export function hasAnyRealAction(
  state: GameState,
  playerNumber: number,
  delegates: ActionAvailabilityDelegates
): boolean {
  // Existing checks for placement, movement, capture...

  // NEW: Check recovery action
  if (delegates.hasRecovery?.(playerNumber)) {
    return true;
  }

  return false;
}
```

Update `ActionAvailabilityDelegates`:

```typescript
export interface ActionAvailabilityDelegates {
  hasPlacement: (playerNumber: number) => boolean;
  hasMovement: (playerNumber: number) => boolean;
  hasCapture: (playerNumber: number) => boolean;
  hasRecovery?: (playerNumber: number) => boolean; // NEW
}
```

#### 4.2 Python LPS Integration

**File:** `ai-service/app/game_engine.py`

Update real action detection to include recovery:

```python
def _has_real_action_for_player(self, game_state: GameState, player_number: int) -> bool:
    """R172 real-action availability predicate for LPS."""
    # Existing checks...

    # Check recovery moves
    recovery_moves = self._get_recovery_moves(game_state, player_number)
    if recovery_moves:
        return True

    return False
```

---

### Phase 5: UI Components

#### 5.1 Board Highlighting

**File:** `src/client/components/BoardView.tsx`

When current player is recovery-eligible:

- Highlight markers that can slide (source markers)
- On marker selection, highlight valid adjacent destinations
- Show line preview when hovering over destination

#### 5.2 Move Selection

**File:** `src/client/sandbox/sandboxMovement.ts`

Add recovery move handling:

- Detect when player clicks on their marker in recovery-eligible state
- Show valid slide destinations
- For overlength lines, present Option 1 vs Option 2 choice dialog

#### 5.3 Choice Dialog for Overlength

**File:** `src/client/components/ChoiceDialog.tsx`

Add `RecoveryLineOptionChoice` similar to existing `LineRewardChoice`:

```typescript
interface RecoveryLineOptionChoice extends PlayerChoice {
  choiceType: 'recovery_line_option';
  lineLength: number;
  requiredLength: number;
  options: [
    { option: 1; description: 'Collapse all markers (costs 1 buried ring)' },
    { option: 2; description: 'Collapse minimum markers (free)' },
  ];
}
```

#### 5.4 Move History Display

**File:** `src/client/components/MoveHistory.tsx`

Add recovery move rendering:

```typescript
case 'recovery_slide':
  return `Recovery: ${formatPos(move.from)} → ${formatPos(move.to)}${move.option === 2 ? ' (min)' : ''}`;
```

#### 5.5 Teaching Content

**File:** `src/shared/teaching/teachingTopics.ts`

Add recovery action teaching tips:

```typescript
export const RECOVERY_ACTION_TIPS: TeachingTip[] = [
  {
    text: 'When you have no stacks and no rings in hand, but still have markers and buried rings, you can perform a RECOVERY ACTION.',
    category: 'eligibility',
    emphasis: 'critical',
  },
  {
    text: 'Slide one of your markers to an adjacent empty cell to complete a line. If successful, extract a buried ring as payment.',
    category: 'action',
    emphasis: 'important',
  },
  {
    text: 'For lines longer than the minimum, you can choose Option 2 (collapse minimum markers) to avoid paying any buried rings.',
    category: 'overlength',
    emphasis: 'normal',
  },
];
```

**File:** `src/client/components/TeachingOverlay.tsx`

Add routing for recovery teaching.

---

### Phase 6: Testing

#### 6.1 Contract Vectors

**File:** `tests/fixtures/contract-vectors/v2/recovery.vectors.json` (NEW)

Create comprehensive test vectors:

```json
{
  "metadata": {
    "category": "recovery",
    "version": 2,
    "description": "Recovery action scenarios (RR-CANON-R110–R115)"
  },
  "vectors": [
    {
      "id": "recovery.eligibility.basic",
      "description": "Player eligible for recovery with exact-length line available",
      "input": { "state": "...", "move": { "type": "recovery_slide", ... } },
      "expectedOutput": { "valid": true, "assertions": { ... } }
    },
    {
      "id": "recovery.overlength.option1",
      "description": "Overlength line with Option 1 (collapse all, pay 1)",
      ...
    },
    {
      "id": "recovery.overlength.option2",
      "description": "Overlength line with Option 2 (collapse min, pay 0)",
      ...
    },
    {
      "id": "recovery.not_eligible.has_stacks",
      "description": "Player not eligible - still controls stacks",
      ...
    },
    {
      "id": "recovery.not_eligible.has_rings_in_hand",
      "description": "Player not eligible - has rings in hand",
      ...
    },
    {
      "id": "recovery.lps.counts_as_real_action",
      "description": "Recovery prevents LPS victory",
      ...
    }
  ]
}
```

#### 6.2 Unit Tests (TypeScript)

**File:** `tests/unit/RecoveryAggregate.shared.test.ts` (exists, extended)

Test coverage (35 tests):

- [x] Option 1/Option 2 semantics for overlength
- [x] Overlength move legal even with 0 buried rings (Option 2)
- [x] Exact-length move requires 1 buried ring
- [x] Line collapse positions for Option 2 (subset selection)
- [ ] Territory cascade after recovery

**File:** `tests/unit/lpsRecovery.test.ts` (NEW)

- [ ] Recovery counts as real action
- [ ] Player with recovery option blocks LPS
- [ ] LPS counter resets when recovery taken

#### 6.3 Unit Tests (Python)

**File:** `ai-service/tests/rules/test_recovery.py` (exists, extend)

Mirror TypeScript tests for:

- [ ] Option 1/Option 2 semantics
- [ ] Overlength with 0 buried rings
- [ ] LPS integration

#### 6.4 Parity Tests

**File:** `ai-service/tests/parity/test_recovery_parity.py` (NEW)

Verify TS↔Python produce identical results for:

- Eligibility checks
- Move enumeration (same moves generated)
- Move application (same state mutations)
- LPS real-action detection

---

### Phase 7: AI Considerations

#### 7.1 Move Ordering

**File:** `ai-service/app/ai/move_ordering.py`

Recovery moves should be evaluated with appropriate heuristics:

- Line length (longer = more territory)
- Option 1 vs Option 2 tradeoff (territory vs. preserving buried rings)
- Resulting board position strength

#### 7.2 Heuristic Evaluation

**File:** `ai-service/app/ai/heuristic_ai.py`

Consider adding recovery-specific evaluation:

- Value of having recovery available (threat potential)
- Cost of losing recovery eligibility
- Buried ring value as recovery resource

Note: Most AI algorithms (minimax, MCTS, descent) auto-inherit recovery moves from `get_valid_moves()`.

---

### Phase 8: Replay & Serialization

#### 8.1 Canonical Replay Engine

**File:** `src/shared/replay/CanonicalReplayEngine.ts`

Add recovery move replay support:

```typescript
case 'recovery_slide':
  return applyRecoverySlide(state, move);
```

#### 8.2 Database Serialization

**File:** `ai-service/app/db/game_replay.py`

Ensure recovery_slide moves are correctly serialized/deserialized.

---

## 3. Complete File Inventory

### Files to Create

| File                                                       | Purpose               |
| ---------------------------------------------------------- | --------------------- |
| `tests/fixtures/contract-vectors/v2/recovery.vectors.json` | Parity test vectors   |
| `tests/unit/lpsRecovery.test.ts`                           | LPS integration tests |
| `ai-service/tests/parity/test_recovery_parity.py`          | TS↔Python parity      |

### Files to Modify

| File                                                  | Changes                              |
| ----------------------------------------------------- | ------------------------------------ |
| `src/shared/engine/aggregates/RecoveryAggregate.ts`   | Option 1/2 semantics                 |
| `src/shared/engine/orchestration/turnOrchestrator.ts` | Wire recovery into movement phase    |
| `src/shared/engine/fsm/FSMAdapter.ts`                 | Add recovery event mapping           |
| `src/shared/engine/notation.ts`                       | Recovery move notation               |
| `src/shared/engine/playerStateHelpers.ts`             | LPS real-action check                |
| `src/server/game/RuleEngine.ts`                       | Validation case                      |
| `ai-service/app/rules/recovery.py`                    | Option 1/2 semantics                 |
| `ai-service/app/game_engine.py`                       | Integration into `get_valid_moves()` |
| `ai-service/app/rules/validators/recovery.py`         | Update validation                    |
| `ai-service/app/rules/mutators/recovery.py`           | Update mutation                      |
| `src/client/components/BoardView.tsx`                 | Recovery move highlighting           |
| `src/client/sandbox/sandboxMovement.ts`               | Recovery move selection              |
| `src/client/components/ChoiceDialog.tsx`              | Option 1/2 choice dialog             |
| `src/client/components/MoveHistory.tsx`               | Recovery move display                |
| `src/shared/teaching/teachingTopics.ts`               | Recovery teaching tips               |
| `tests/unit/RecoveryAggregate.shared.test.ts`         | Extend with Option 1/2               |
| `ai-service/tests/rules/test_recovery.py`             | Extend with Option 1/2               |

---

## 4. Implementation Priority Order

### Critical Path (P0) – ✅ COMPLETE

1. ✅ **Update cost model** in RecoveryAggregate.ts and recovery.py
2. ✅ **Integrate into turn orchestrator** (TS) and game_engine.py (Python)
3. ✅ **LPS integration** in both engines
4. ✅ **Contract vectors** for parity testing (`recovery_action.vectors.json`)
5. ✅ **Basic parity tests** (`test_recovery_parity.py` - 15 tests)

### High Priority (P1) – ✅ COMPLETE

6. ✅ FSM event mapping (`TurnStateMachine.ts`, `FSMAdapter.ts`)
7. ✅ RuleEngine validation case (`RuleEngine.ts`)
8. ✅ Unit tests for Option 1/2 semantics (`RecoveryAggregate.shared.test.ts`)
9. ✅ Move notation (`notation.ts`)

### Medium Priority (P2) – ✅ COMPLETE

10. ✅ UI highlighting and selection (`ClientSandboxEngine.ts`, `useSandboxInteractions.ts`)
11. ✅ Choice dialog for overlength (`useSandboxInteractions.ts` - confirm dialog)
12. ✅ Move history display (`MoveHistory.tsx`)
13. ✅ Teaching content (`teachingTopics.ts` - 7 tips)
14. ✅ AI move ordering (already in `move_ordering.py`)

### Lower Priority (P3)

15. Heuristic evaluation enhancements
16. Replay engine integration
17. Extended parity test coverage

---

## 5. Risk Considerations

| Risk                    | Impact                                                      | Mitigation                                                 |
| ----------------------- | ----------------------------------------------------------- | ---------------------------------------------------------- |
| **Cost model mismatch** | Existing code uses graduated cost; rules now use Option 1/2 | Comprehensive update of both TS and Python implementations |
| **LPS edge cases**      | Recovery affecting game termination                         | Extensive testing with LPS scenarios                       |
| **UI complexity**       | Option 1/2 choice adds interaction                          | Clear UI with good defaults                                |
| **Parity divergence**   | TS and Python may handle overlength differently             | Contract vectors and parity tests                          |
| **Recovery rarity**     | Hard to test rare scenarios                                 | Curated test scenarios and fuzzing                         |

---

## 6. Acceptance Criteria

- [ ] Recovery eligibility correctly detected in both engines
- [ ] Recovery moves enumerated when eligible (including overlength with Option 2)
- [ ] Option 1/Option 2 choice for overlength lines works correctly
- [ ] Line collapse follows chosen option (all vs. minimum)
- [ ] Buried ring extraction only for exact-length or Option 1
- [ ] Recovery counts as real action for LPS
- [ ] All contract vectors pass
- [ ] TS↔Python parity verified
- [ ] UI allows recovery move selection
- [ ] Move history correctly displays recovery moves
- [ ] Teaching overlay explains recovery action

---

## Appendix A: Code Examples

### A.1 Option 1/2 Move Structure

```typescript
// TypeScript Move for recovery with overlength
interface RecoverySlideMove extends Move {
  type: 'recovery_slide';
  player: number;
  from: Position;
  to: Position;
  /** For overlength lines: 1 (collapse all) or 2 (collapse min) */
  option?: 1 | 2;
  /** For Option 2: which markers to collapse (subset of line) */
  collapsedMarkers?: Position[];
  /** Stack to extract buried ring from (for exact-length or Option 1) */
  extractionStack?: string;
}
```

### A.2 Enumeration Logic

```typescript
function enumerateRecoverySlides(state: GameState, player: number): Move[] {
  if (!isEligibleForRecovery(state, player)) return [];

  const moves: Move[] = [];
  const lineLength = getEffectiveLineLengthThreshold(state.board.type, state.players.length);
  const buriedRings = countBuriedRings(state.board, player);

  for (const [posKey, marker] of state.board.markers) {
    if (marker.player !== player) continue;

    for (const dir of getAdjacencyDirections(state.board.type)) {
      const toPos = addPositions(marker.position, dir);
      if (!isValidEmptyCell(toPos, state.board)) continue;

      const formedLength = getFormedLineLength(state.board, marker.position, toPos, player);

      if (formedLength >= lineLength) {
        const isOverlength = formedLength > lineLength;

        if (isOverlength) {
          // Option 2 is always available (free)
          moves.push({
            type: 'recovery_slide',
            player,
            from: marker.position,
            to: toPos,
            option: 2,
          });

          // Option 1 requires at least 1 buried ring
          if (buriedRings >= 1) {
            moves.push({
              type: 'recovery_slide',
              player,
              from: marker.position,
              to: toPos,
              option: 1,
            });
          }
        } else {
          // Exact-length requires 1 buried ring
          if (buriedRings >= 1) {
            moves.push({ type: 'recovery_slide', player, from: marker.position, to: toPos });
          }
        }
      }
    }
  }

  return moves;
}
```

---

## Appendix B: Contract Vector Schema

```json
{
  "id": "recovery.overlength.option2.free",
  "category": "recovery",
  "description": "Overlength line with Option 2 has zero cost",
  "input": {
    "state": {
      "board": {
        "type": "square8",
        "stacks": {},
        "markers": {
          "3,3": { "player": 1 },
          "3,4": { "player": 1 },
          "3,5": { "player": 1 },
          "3,7": { "player": 1 }
        },
        "collapsedSpaces": {}
      },
      "players": [
        { "playerNumber": 1, "ringsInHand": 0 },
        { "playerNumber": 2, "ringsInHand": 5 }
      ],
      "currentPlayer": 1,
      "currentPhase": "movement"
    },
    "move": {
      "type": "recovery_slide",
      "player": 1,
      "from": { "x": 3, "y": 7 },
      "to": { "x": 3, "y": 6 },
      "option": 2
    }
  },
  "expectedOutput": {
    "valid": true,
    "assertions": {
      "linesProcessed": 1,
      "markersCollapsed": 3,
      "ringsExtracted": 0,
      "territoryGained": 3
    }
  }
}
```
