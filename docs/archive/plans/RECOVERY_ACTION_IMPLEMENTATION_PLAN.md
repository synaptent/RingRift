# Recovery Action Implementation Plan

## Overview

This document outlines the implementation plan for the **Recovery Action** rule, which allows temporarily eliminated players to recover by sliding a marker to complete a line.

### Rule Summary (from canonical specs)

**Eligibility:** A player is eligible for recovery if ALL of these conditions hold:
1. They control **no stacks** on the board
2. They have **zero rings in hand**
3. They have at least one **marker** on the board
4. They have **buried rings** in opponent-controlled stacks

**Action:** A **recovery slide** moves one of the player's markers orthogonally (Von Neumann adjacency) to an adjacent empty space. The slide is legal **only if** it completes a line of **exactly** `lineLength` consecutive markers of the player's colour. Overlength lines do NOT qualify.

**Effect:** On completing a line:
- The line of markers collapses into territory
- Buried rings are exhumed and returned to the player's hand
- The player can now place rings on subsequent turns

**LPS Impact:** Recovery action counts as a "real action" for Last Player Standing purposes (alongside placement, movement, and capture).

---

## Implementation Tasks

### Phase 1: Recovery Action Core Implementation

#### 1.1 New MoveType

**File: `src/shared/types/game.ts`**
Add to MoveType union:
```typescript
| 'recovery_slide'        // Marker slide that completes a line for recovery
```

**File: `ai-service/app/models/core.py`**
Add to MoveType enum:
```python
RECOVERY_SLIDE = "recovery_slide"
```

#### 1.2 Recovery Eligibility Predicate

**TypeScript: `src/shared/engine/playerStateHelpers.ts`**
```typescript
/**
 * Check if a player is eligible for recovery action.
 *
 * Eligibility (RR-CANON-R???):
 * - No stacks controlled
 * - Zero rings in hand
 * - Has markers on board
 * - Has buried rings in opponent stacks
 */
export function isEligibleForRecovery(
  state: GameState,
  playerNumber: number
): boolean {
  const player = state.players.find(p => p.playerNumber === playerNumber);
  if (!player) return false;

  // Must have zero rings in hand
  if (player.ringsInHand > 0) return false;

  // Must control no stacks
  if (playerControlsAnyStack(state.board, playerNumber)) return false;

  // Must have at least one marker
  const hasMarker = [...state.board.markers.values()]
    .some(m => m.player === playerNumber);
  if (!hasMarker) return false;

  // Must have buried rings in opponent stacks
  const hasBuriedRings = [...state.board.stacks.values()]
    .some(stack =>
      stack.controllingPlayer !== playerNumber &&
      stack.rings.includes(playerNumber)
    );

  return hasBuriedRings;
}
```

**Python: `ai-service/app/game_engine.py`**
```python
@staticmethod
def _is_eligible_for_recovery(game_state: GameState, player_number: int) -> bool:
    """Check if player is eligible for recovery action."""
    player = next((p for p in game_state.players if p.player_number == player_number), None)
    if not player:
        return False

    # Must have zero rings in hand
    if player.rings_in_hand > 0:
        return False

    # Must control no stacks
    board = game_state.board
    controls_stack = any(
        s.controlling_player == player_number
        for s in board.stacks.values()
    )
    if controls_stack:
        return False

    # Must have at least one marker
    has_marker = any(m.player == player_number for m in board.markers.values())
    if not has_marker:
        return False

    # Must have buried rings in opponent stacks
    has_buried = any(
        s.controlling_player != player_number and player_number in s.rings
        for s in board.stacks.values()
    )

    return has_buried
```

#### 1.3 Recovery Move Generation

**TypeScript: `src/shared/engine/aggregates/RecoveryAggregate.ts`** (new file)
```typescript
/**
 * Recovery Action Aggregate
 *
 * Handles enumeration and application of recovery slides.
 * Recovery is available when a player is temporarily eliminated
 * (no stacks, no rings in hand) but has markers and buried rings.
 */

export function enumerateRecoverySlides(
  state: GameState,
  playerNumber: number
): Move[] {
  if (!isEligibleForRecovery(state, playerNumber)) {
    return [];
  }

  const lineLength = getEffectiveLineLengthThreshold(
    state.board.type,
    state.players.length
  );

  const moves: Move[] = [];

  // For each marker owned by the player
  for (const [posKey, marker] of state.board.markers) {
    if (marker.player !== playerNumber) continue;

    const fromPos = stringToPosition(posKey);

    // Check each Von Neumann neighbor (4 orthogonal directions)
    const vonNeumannDirs = [
      { x: 1, y: 0 },
      { x: -1, y: 0 },
      { x: 0, y: 1 },
      { x: 0, y: -1 }
    ];

    for (const dir of vonNeumannDirs) {
      const toPos = { x: fromPos.x + dir.x, y: fromPos.y + dir.y };

      // Must be valid and empty (no stack, no marker, not collapsed)
      if (!isValidPosition(toPos, state.board)) continue;
      if (getStack(toPos, state.board)) continue;
      if (getMarker(toPos, state.board) !== undefined) continue;
      if (isCollapsedSpace(toPos, state.board)) continue;

      // Would this slide complete a line of EXACTLY lineLength?
      if (wouldCompleteExactLine(state, fromPos, toPos, playerNumber, lineLength)) {
        moves.push({
          type: 'recovery_slide',
          player: playerNumber,
          from: fromPos,
          to: toPos,
        });
      }
    }
  }

  return moves;
}

function wouldCompleteExactLine(
  state: GameState,
  fromPos: Position,
  toPos: Position,
  player: number,
  requiredLength: number
): boolean {
  // Simulate the marker move
  const tempBoard = cloneBoardState(state.board);
  tempBoard.markers.delete(positionToString(fromPos));
  tempBoard.markers.set(positionToString(toPos), {
    player,
    position: toPos,
    type: 'regular'
  });

  // Find lines containing the new position
  const lines = findLinesContainingPosition(
    { ...state, board: tempBoard },
    toPos
  );

  // Must have exactly one line of exactly lineLength (not overlength)
  return lines.some(line =>
    line.player === player &&
    line.length === requiredLength
  );
}
```

**Python: `ai-service/app/game_engine.py`**
```python
@staticmethod
def _get_recovery_moves(game_state: GameState, player_number: int) -> List[Move]:
    """Get recovery slide moves for an eligible player."""
    if not GameEngine._is_eligible_for_recovery(game_state, player_number):
        return []

    line_length = get_effective_line_length(
        game_state.board.type,
        len(game_state.players)
    )

    moves: List[Move] = []
    board = game_state.board

    # Von Neumann directions (orthogonal only)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for pos_key, marker in board.markers.items():
        if marker.player != player_number:
            continue

        from_pos = marker.position

        for dx, dy in directions:
            to_pos = Position(x=from_pos.x + dx, y=from_pos.y + dy)

            # Must be valid and empty
            if not BoardManager.is_valid_position(to_pos, board.type, board.size):
                continue
            if BoardManager.get_stack(to_pos, board):
                continue
            if board.markers.get(to_pos.to_key()):
                continue
            if BoardManager.is_collapsed_space(to_pos, board):
                continue

            # Would complete a line of exactly lineLength?
            if GameEngine._would_complete_exact_line(
                game_state, from_pos, to_pos, player_number, line_length
            ):
                moves.append(Move(
                    type=MoveType.RECOVERY_SLIDE,
                    player=player_number,
                    from_position=from_pos,
                    to=to_pos,
                ))

    return moves
```

#### 1.4 Recovery Move Application

**TypeScript: `src/shared/engine/aggregates/RecoveryAggregate.ts`**
```typescript
export function applyRecoverySlide(
  state: GameState,
  move: Move
): RecoveryApplicationOutcome {
  // 1. Move the marker from -> to
  const fromKey = positionToString(move.from);
  const toKey = positionToString(move.to);

  state.board.markers.delete(fromKey);
  state.board.markers.set(toKey, {
    player: move.player,
    position: move.to,
    type: 'regular'
  });

  // 2. Detect the formed line
  const lines = findLinesContainingPosition(state, move.to);
  const formedLine = lines.find(l => l.player === move.player);

  if (!formedLine) {
    throw new Error('Recovery slide did not form a line');
  }

  // 3. Process the line (collapse markers -> territory)
  // This follows the same logic as normal line processing
  // but happens immediately as part of the recovery action
  processLineForRecovery(state, formedLine);

  // 4. Exhume buried rings
  exhumeBuriedRings(state, move.player);

  return { success: true, formedLine };
}
```

**Python:** Similar implementation in `_apply_recovery_slide()`.

#### 1.5 Integration into Turn Orchestrator

**TypeScript: `src/shared/engine/orchestration/turnOrchestrator.ts`**

In the `movement` phase, add recovery slides to available moves:
```typescript
// In movement phase enumeration
if (phase === 'movement') {
  const movementMoves = enumerateSimpleMovesForPlayer(state, player);
  const captureMoves = enumerateAllCaptureMoves(state, player);
  const recoveryMoves = enumerateRecoverySlides(state, player);  // NEW
  moves = [...movementMoves, ...captureMoves, ...recoveryMoves];
}
```

**Python: `ai-service/app/game_engine.py`**

In `get_valid_moves()` for MOVEMENT phase:
```python
elif phase == GamePhase.MOVEMENT:
    movement_moves = GameEngine._get_movement_moves(state, player_number)
    capture_moves = GameEngine._get_capture_moves(state, player_number)
    recovery_moves = GameEngine._get_recovery_moves(state, player_number)  # NEW
    moves = movement_moves + capture_moves + recovery_moves
```

#### 2.6 LPS Real Action Update

**TypeScript: `src/shared/engine/playerStateHelpers.ts`**

Update `hasAnyRealAction()` to include recovery:
```typescript
export function hasAnyRealAction(
  state: GameState,
  playerNumber: number,
  delegates: ActionAvailabilityDelegates
): boolean {
  // ... existing checks ...

  // Check recovery action
  if (delegates.hasRecovery && delegates.hasRecovery(playerNumber)) {
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
  hasRecovery?: (playerNumber: number) => boolean;  // NEW
}
```

**Python: `ai-service/app/game_engine.py`**

Update `_has_real_action_for_player()`:
```python
@staticmethod
def _has_real_action_for_player(game_state: GameState, player_number: int) -> bool:
    """R172 real-action availability predicate for LPS."""
    if GameEngine._has_valid_actions(game_state, player_number):
        return True
    # Also check recovery action
    if GameEngine._get_recovery_moves(game_state, player_number):
        return True
    return False
```

---

### Phase 3: Tests

#### 3.1 TypeScript Tests

**File: `tests/unit/RecoveryAggregate.test.ts`** (new)
- Test recovery eligibility predicate
- Test recovery move enumeration
- Test that overlength lines don't qualify
- Test that Von Neumann adjacency is enforced
- Test recovery application and line collapse
- Test ring exhumation

**File: `tests/unit/lpsTracking.recovery.test.ts`** (new or extend existing)
- Test that recovery counts as real action for LPS

**File: `tests/unit/lineLength.playerCount.test.ts`** (new)
- Test that square8 2-player uses lineLength=4
- Test that square8 3-4 player uses lineLength=3

#### 3.2 Python Tests

**File: `ai-service/tests/test_recovery_action.py`** (new)
- Mirror TS tests for recovery eligibility
- Test recovery move generation
- Test overlength exclusion
- Test Von Neumann adjacency

**File: `ai-service/tests/test_linelength_player_count.py`** (new)
- Test player-count dependent lineLength

**File: `ai-service/tests/parity/test_recovery_parity.py`** (new)
- Test TS/Python parity for recovery scenarios

#### 3.3 Mutator Tests

**File: `ai-service/tests/rules/test_mutators.py`**
Add RecoveryMutator tests following existing patterns.

---

### Phase 4: Python Mutator/Validator Infrastructure

#### 4.1 RecoveryValidator

**File: `ai-service/app/rules/validators/recovery_validator.py`** (new)
```python
class RecoveryValidator:
    def validate(self, state: GameState, move: Move) -> ValidationResult:
        # Validate recovery_slide move
        pass
```

#### 4.2 RecoveryMutator

**File: `ai-service/app/rules/mutators/recovery_mutator.py`** (new)
```python
class RecoveryMutator:
    def apply(self, state: GameState, move: Move) -> GameState:
        # Apply recovery_slide move
        pass
```

#### 4.3 DefaultRulesEngine Integration

**File: `ai-service/app/rules/default_engine.py`**
Add recovery shadow contract following existing patterns.

---

### Phase 5: UI Considerations

#### 5.1 Move Display

Recovery slides should be displayed in the move list/history with appropriate labeling:
- "Recovery: A3 → A4 (forms line)"

#### 5.2 Board Highlighting

When a player is eligible for recovery, highlight:
- Their markers that can slide
- Valid destination squares that would complete a line

#### 5.3 Tutorial/Help Text

Add explanation of recovery action to help system.

---

## File Summary

### New Files
- `src/shared/engine/aggregates/RecoveryAggregate.ts`
- `tests/unit/RecoveryAggregate.test.ts`
- `tests/unit/lineLength.playerCount.test.ts`
- `ai-service/app/rules/validators/recovery_validator.py`
- `ai-service/app/rules/mutators/recovery_mutator.py`
- `ai-service/tests/test_recovery_action.py`
- `ai-service/tests/test_linelength_player_count.py`
- `ai-service/tests/parity/test_recovery_parity.py`

### Modified Files
- `src/shared/types/game.ts` (add MoveType)
- `src/shared/engine/rulesConfig.ts` (lineLength player-count)
- `src/shared/engine/playerStateHelpers.ts` (recovery eligibility, LPS)
- `src/shared/engine/orchestration/turnOrchestrator.ts` (integrate recovery)
- `ai-service/app/models/core.py` (add MoveType)
- `ai-service/app/rules/core.py` (lineLength player-count)
- `ai-service/app/game_engine.py` (recovery methods)
- `ai-service/app/rules/default_engine.py` (recovery contract)

---

## Implementation Order

1. **LineLength Player-Count** (pre-requisite)
   - Update TS `getEffectiveLineLengthThreshold()`
   - Update Python `get_effective_line_length()`
   - Add tests

2. **Recovery Eligibility**
   - Add eligibility predicate to both engines
   - Add tests

3. **Recovery Move Generation**
   - Add move enumeration to both engines
   - Ensure overlength exclusion
   - Add tests

4. **Recovery Move Application**
   - Implement application in both engines
   - Handle line collapse and ring exhumation
   - Add tests

5. **LPS Integration**
   - Update `hasAnyRealAction()` in both engines
   - Add LPS-specific tests

6. **Turn Orchestrator Integration**
   - Wire recovery into movement phase
   - Full integration tests

7. **Parity Tests**
   - Ensure TS/Python produce identical results

8. **Mutator/Validator Infrastructure** (Python)
   - Add shadow contracts

---

## Risk Considerations

1. **LineLength Change Impact:** Changing square8 2-player to lineLength=4 is a significant gameplay change. Existing games and replays may behave differently.

2. **Recovery Rarity:** Recovery scenarios are rare in practice. Extensive testing needed to ensure edge cases are covered.

3. **Performance:** Recovery move enumeration involves line detection simulation. May need optimization for AI search.

4. **UI Complexity:** Recovery adds another action type that needs clear visual representation.
