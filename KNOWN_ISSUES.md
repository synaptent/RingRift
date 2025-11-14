# Known Issues & Bugs

**Last Updated:** November 13, 2025  
**Status:** Code-verified assessment based on actual implementation  
**Related Documents:** [CURRENT_STATE_ASSESSMENT.md](./CURRENT_STATE_ASSESSMENT.md) | [TODO.md](./TODO.md) | [CONTRIBUTING.md](./CONTRIBUTING.md)

This document tracks specific bugs, missing features, and implementation issues in the RingRift codebase.

**VERIFICATION STATUS:** All issues verified through code analysis (November 13, 2025)

---

## 🔴 Critical Issues (Prevents Core Functionality)

### Issue #0: Player Choice System Not Implemented ⭐ NEW
**Priority:** P0 - CRITICAL  
**Component:** GameEngine.ts, NEW: PlayerInteractionManager.ts  
**Status:** Not Started  
**Severity:** BLOCKS STRATEGIC GAMEPLAY

**Description:**
The most critical architectural gap: NO mechanism exists for player choices during gameplay. All strategic decisions currently default to first option, completely eliminating player agency and strategic depth.

**Expected Behavior:**
Players should be able to make strategic choices at key decision points:
1. **Line Processing Order:** When multiple lines form, choose which to process first
2. **Graduated Line Rewards:** Choose Option 1 (collapse all + eliminate ring) vs Option 2 (collapse minimum, no elimination)
3. **Ring/Cap Elimination:** Choose which stack to eliminate ring/cap from
4. **Region Processing Order:** When multiple disconnected regions, choose processing order
5. **Capture Direction:** When multiple valid capture directions, choose which to pursue

**Current Behavior:**
- Line processing: Uses first line found (GameEngine.ts line 459)
- Graduated rewards: ALWAYS uses Option 2 (GameEngine.ts line 484)
- Ring elimination: Uses first stack (GameEngine.ts line 516)
- Region order: Uses first found
- Capture direction: Not implemented

**Code Evidence:**
```typescript
// GameEngine.ts line 459
// TODO: In full implementation, player should choose which line to process first
// For now, process in order found
const line = lines[0];

// GameEngine.ts line 484
// TODO: In full implementation, player should choose Option 1 or Option 2
// For now, always use Option 2 to preserve rings

// GameEngine.ts line 516
// TODO: In full implementation, player should choose which stack
// For now, eliminate from first stack
```

**Impact:**
- Game is unplayable at competitive level
- Strategic depth eliminated
- Cannot properly test rule compliance
- AI cannot make intelligent decisions

**Files Affected:**
- `src/server/game/GameEngine.ts` (needs integration)
- `src/shared/types/game.ts` (needs PlayerChoice types)
- NEW: `src/server/game/PlayerInteractionManager.ts` (needs creation)
- NEW: `src/client/components/PlayerChoiceDialog.tsx` (needs creation)

**Required Solution:**
Create async choice request/response system that:
- Prompts human players via UI
- Allows AI to make programmatic decisions
- Supports timeouts and validation
- Integrates with all choice points in game flow

**Rule References:**
- Section 11.2: Graduated line rewards
- Section 10.3: Chain capture direction choices
- Section 12.2: Region processing order
- All player agency rules

**Added:** November 13, 2025 (Code Verification)

---

### Issue #1: Marker System - RESOLVED ✅
**Priority:** P0 - CRITICAL  
**Component:** GameEngine.ts, BoardManager.ts  
**Status:** ✅ COMPLETED (November 13, 2025)

**Description:**
Marker system implementation completed with all core functionality working.

**Implemented Features:**
- ✅ Markers placed when rings move from position
- ✅ Opponent markers flip to mover's color when jumped
- ✅ Own markers collapse to territory when jumped
- ✅ Landing on own marker removes it
- ✅ Collapsed spaces block movement

**Files Completed:**
- ✅ `src/server/game/GameEngine.ts` (processMarkersAlongPath method)
- ✅ `src/server/game/BoardManager.ts` (all marker CRUD methods)
- ✅ `src/shared/types/game.ts` (BoardState with collapsedSpaces)

**Remaining (Deferred to Phase 2):**
- [ ] Comprehensive unit tests
- [ ] Integration tests

**Rule Reference:** Section 8.3, Section 4.2.1

**Verified:** Code analysis November 13, 2025

---

### Issue #2: Line Formation Not Detecting or Processing Lines
**Priority:** P0 - CRITICAL  
**Component:** GameEngine.ts, RuleEngine.ts, BoardManager.ts  
**Status:** Partial Implementation

**Description:**
Line formation is a core game mechanic where forming 4+ consecutive markers (8x8) or 5+ markers (19x19/hex) triggers special actions. Current implementation has basic detection but missing processing logic.

**Expected Behavior:**
1. Detect when 4+ (8x8) or 5+ (19x19/hex) consecutive markers of same color form a line
2. For exactly minimum length: Collapse all markers AND eliminate one player ring/cap
3. For longer lines: Player chooses Option 1 (collapse all + eliminate ring) or Option 2 (collapse only minimum without eliminating)
4. Process multiple lines in player-chosen order
5. Check for new lines after each collapse

**Current Behavior:**
- Basic line detection exists but incomplete
- No graduated reward system (Option 1 vs Option 2)
- No ring elimination on line collapse
- No player choice mechanism
- No multi-line processing order

**Files Affected:**
- `src/server/game/GameEngine.ts` (processAutomaticConsequences method)
- `src/server/game/BoardManager.ts` (findAllLines method)
- `src/server/game/RuleEngine.ts` (processLineFormation method)

**Rule Reference:**
- Section 11: Line Formation & Collapse
- Section 11.2: Collapse Process (Graduated Line Rewards)

**Code Example:**
```typescript
// Current (incomplete)
for (const line of lines) {
  if (line.positions.length >= config.lineLength) {
    for (const pos of line.positions) {
      this.boardManager.removeStack(pos, this.gameState.board);
    }
  }
}

// Needed
for (const line of lines) {
  if (line.positions.length === config.lineLength) {
    this.processLineCollapse(line, 'collapse_all');
    this.eliminatePlayerRingOrCap(line.player);
  } else if (line.positions.length > config.lineLength) {
    const choice = await this.promptPlayerChoice(line.player);
    // Process based on choice...
  }
}
```

---

### Issue #3: Territory Disconnection Not Implemented
**Priority:** P0 - CRITICAL  
**Component:** RuleEngine.ts, BoardManager.ts  
**Status:** Stub Only

**Description:**
Territory disconnection is one of the most powerful mechanics in RingRift. When a region becomes physically disconnected and lacks representation from active players, it should be claimed. This is completely missing.

**Expected Behavior:**
1. Detect regions surrounded by collapsed spaces/board edges/single-player markers using Von Neumann adjacency (4-direction for square, 6 for hex)
2. Check if region lacks representation (active player stacks) from at least one player on the board
3. Perform self-elimination prerequisite check (would player have rings left to self-eliminate?)
4. If check passes: Collapse region + border markers, eliminate all internal rings, mandatory self-elimination
5. Check for chain reactions (new disconnections)

**Current Behavior:**
- Method exists but only removes stacks without proper logic
- No Von Neumann adjacency checking
- No representation checking
- No self-elimination prerequisite
- No border marker collapse
- No chain reaction detection

**Files Affected:**
- `src/server/game/RuleEngine.ts` (processTerritoryDisconnection method)
- `src/server/game/BoardManager.ts` (needs findDisconnectedRegions method)

**Rule Reference:**
- Section 12: Area Disconnection & Collapse
- Section 12.2: Disconnection Process
- FAQ Q15: How is a region determined to be surrounded or disconnected?

---

### Issue #4: Movement Validation Incomplete
**Priority:** P0 - CRITICAL  
**Component:** RuleEngine.ts  
**Status:** Partial Implementation

**Description:**
Movement validation doesn't properly enforce RingRift's complex movement rules.

**Expected Behavior:**
1. Movement distance ≥ stack height (minimum distance rule)
2. Can land on any valid space beyond markers meeting distance requirement (not just first)
3. Path must be clear of other rings/stacks
4. Cannot move through collapsed spaces
5. Landing on same-color marker removes it

**Current Behavior:**
- Basic distance checking exists but incomplete
- Landing rule not implemented correctly
- Path validation incomplete
- No marker landing logic

**Files Affected:**
- `src/server/game/RuleEngine.ts` (validateStackMovement method)

**Rule Reference:**
- Section 8.2: Minimum Distance Requirements
- Q2 FAQ: What exactly counts for minimum jump requirement?

---

### Issue #5: Chain Captures Not Enforced ⭐ UPDATED
**Priority:** P0 - CRITICAL  
**Component:** GameEngine.ts, RuleEngine.ts  
**Status:** Partially Implemented (40% complete)

**Description:**
Single captures work correctly, but mandatory chain capture continuation is NOT enforced. This is a critical rule violation.

**Expected Behavior:**
1. ✅ Overtaking: Captured rings added to bottom of capturing stack (WORKING)
2. ✅ Elimination: Rings permanently removed (WORKING)
3. ❌ Chain captures mandatory once started (NOT IMPLEMENTED)
4. ✅ Cap height comparison required for overtaking (WORKING)
5. ✅ Flexible landing during captures (WORKING)

**Current Behavior:**
- ✅ Single captures work correctly
- ✅ Overtaking vs elimination distinction implemented
- ❌ Chain captures NOT mandatory
- ❌ Player cannot choose capture direction when multiple valid
- ❌ 180° reversal patterns not tested
- ❌ Cyclic capture patterns not tested

**Code Evidence:**
```typescript
// RuleEngine.ts has processChainReactions() but it's not fully enforced
// GameEngine does not force continuation of chain captures
```

**Impact:**
- Major rule violation
- Cannot play competitive games
- Strategic capture sequences impossible
- FAQ Q14 scenarios fail

**Files Affected:**
- `src/server/game/GameEngine.ts` (needs chain enforcement)
- `src/server/game/RuleEngine.ts` (has structure, needs completion)

**Rule Reference:**
- Section 10.3: Chain Overtaking - "Mandatory once started"
- FAQ Q14: Chain capture mechanics
- FAQ Q15.3.1: 180° reversal patterns
- FAQ Q15.3.2: Cyclic patterns

**Verified:** Code analysis November 13, 2025

---

## 🟡 High Priority Issues (Major Features Missing)

### Issue #6: Forced Elimination Not Implemented
**Priority:** P1 - HIGH  
**Component:** GameEngine.ts, RuleEngine.ts  
**Status:** Not Started

**Description:**
When a player has no valid moves but controls stacks, they must eliminate a stack cap. This rule is not implemented.

**Expected Behavior:**
At the start of a player's turn, if they have no valid placement, movement, or capture options but control stacks on the board, they must eliminate the entire cap of one controlled stack.

**Current Behavior:**
No logic to detect this situation or force elimination.

**Files Affected:**
- `src/server/game/GameEngine.ts` (turn start logic)
- `src/server/game/RuleEngine.ts` (getValidMoves method)

**Rule Reference:**
- Section 4.4: Forced Elimination When Blocked
- FAQ Q24: What happens if I control stacks but have no valid options?

---

### Issue #7: Game Phase Transitions Incorrect
**Priority:** P1 - HIGH  
**Component:** GameEngine.ts  
**Status:** Incorrect Implementation

**Description:**
The current phase system doesn't match the actual game flow from the rules.

**Expected Phases:**
1. `ring_placement` (optional unless no rings on board)
2. `movement` (required if possible)
3. `capture` (optional to start, mandatory chain)
4. `line_processing` (automatic)
5. `territory_processing` (automatic)

**Current Phases:**
- Has `main_game` phase (undefined in rules)
- Missing `line_processing` phase
- Phase transitions don't follow rules

**Files Affected:**
- `src/shared/types/game.ts` (GamePhase type)
- `src/server/game/GameEngine.ts` (advanceGame method)

**Rule Reference:**
- Section 4: Turn Sequence
- Section 15.2: Turn Sequence and Flow

---

### Issue #8: Player State Not Updated Correctly
**Priority:** P1 - HIGH  
**Component:** GameEngine.ts  
**Status:** Broken

**Description:**
Player state fields (ringsInHand, eliminatedRings, territorySpaces) are not updated during gameplay.

**Expected Behavior:**
- `ringsInHand` decrements when rings are placed
- `eliminatedRings` increments when rings are eliminated
- `territorySpaces` updates when territory is claimed

**Current Behavior:**
- Fields defined but never updated
- Ring counts don't match actual game state

**Files Affected:**
- `src/server/game/GameEngine.ts` (multiple methods)

---

### Issue #9: Board State Missing Collapsed Spaces
**Priority:** P1 - HIGH  
**Component:** BoardManager.ts, game.ts  
**Status:** Not Implemented

**Description:**
The BoardState interface doesn't properly track collapsed spaces (claimed territory).

**Expected Behavior:**
Separate tracking of:
- `stacks` - Ring stacks on board
- `markers` - Markers on board (player color)
- `collapsedSpaces` - Claimed territory (player color)

**Current Behavior:**
- All tracked together in unclear way
- No distinction between markers and collapsed spaces

**Files Affected:**
- `src/shared/types/game.ts` (BoardState interface)
- `src/server/game/BoardManager.ts` (all methods)

---

## 🟢 Medium Priority Issues (Feature Gaps)

### Issue #10: AI Players Not Implemented
**Priority:** P2 - MEDIUM  
**Component:** New AIPlayer class needed  
**Status:** Not Started

**Description:**
No AI implementation exists.

**Requirements:**
- Levels 1-3: Random valid moves
- Levels 4-7: Heuristic evaluation
- Levels 8-10: MCTS or minimax

**Files Needed:**
- `src/server/ai/AIPlayer.ts`
- `src/server/ai/MoveEvaluator.ts`

---

### Issue #11: Frontend UI Not Implemented
**Priority:** P2 - MEDIUM  
**Component:** Client components  
**Status:** Skeleton Only

**Description:**
No board rendering or game UI exists.

**Components Needed:**
- GameBoard.tsx
- BoardCell.tsx
- RingStack.tsx
- MarkerDisplay.tsx
- MoveInput.tsx
- GameStatus.tsx

**Files Affected:**
- `src/client/components/` (new files needed)

---

### Issue #12: WebSocket Game Events Incomplete
**Priority:** P2 - MEDIUM  
**Component:** WebSocket server  
**Status:** Basic Setup Only

**Description:**
Socket.IO configured but game-specific events not implemented.

**Events Needed:**
- player_move
- game_state_update
- game_ended
- spectator_joined
- move_validation

**Files Affected:**
- `src/server/websocket/server.ts`

---

### Issue #13: No Tests Written
**Priority:** P2 - MEDIUM  
**Component:** Tests  
**Status:** Not Started

**Description:**
Jest configured but no actual test files exist.

**Tests Needed:**
- Unit tests for game rules
- Integration tests for turn sequences
- Scenario tests from rules document
- E2E gameplay tests

**Files Needed:**
- `src/server/game/__tests__/`
- `src/client/__tests__/`

---

### Issue #14: Database Not Connected
**Priority:** P2 - MEDIUM  
**Component:** Database integration  
**Status:** Schema Only

**Description:**
Prisma schema defined but not connected to game engine.

**Needs:**
- Game persistence
- User data storage
- Move history recording
- Statistics tracking

**Files Affected:**
- `src/server/services/GameService.ts`
- `src/server/services/UserService.ts`

---

## 🔵 Low Priority Issues (Nice-to-Have)

### Issue #15: Spectator Mode Not Implemented
**Priority:** P3 - LOW  
**Component:** WebSocket, UI  
**Status:** Not Started

**Description:**
No spectator functionality exists.

---

### Issue #16: Replay System Not Implemented
**Priority:** P3 - LOW  
**Component:** Database, UI  
**Status:** Not Started

**Description:**
Game replays not supported.

---

### Issue #17: Rating System Incomplete
**Priority:** P3 - LOW  
**Component:** RatingService  
**Status:** Stub Only

**Description:**
ELO rating calculation not implemented.

---

## 📋 Issue Summary by Component

### GameEngine.ts
- Issue #1: Marker system
- Issue #2: Line formation
- Issue #5: Capture system
- Issue #6: Forced elimination
- Issue #7: Phase transitions
- Issue #8: Player state updates

### RuleEngine.ts
- Issue #3: Territory disconnection
- Issue #4: Movement validation
- Issue #5: Capture validation
- Issue #6: Forced elimination

### BoardManager.ts
- Issue #1: Marker methods
- Issue #2: Line detection
- Issue #3: Disconnection detection
- Issue #9: Collapsed spaces tracking

### Type Definitions (game.ts)
- Issue #7: GamePhase enum
- Issue #9: BoardState interface

### New Components Needed
- Issue #10: AI implementation
- Issue #11: Frontend UI
- Issue #13: Test suite

---

## 🚀 Recommended Fix Order

For efficient development, fix issues in this order:

### Phase 1: Core Logic Foundation
1. Issue #9: Fix BoardState data structure
2. Issue #1: Implement marker system
3. Issue #4: Fix movement validation
4. Issue #7: Fix phase transitions

### Phase 2: Game Mechanics
5. Issue #5: Complete capture system
6. Issue #2: Implement line formation
7. Issue #3: Implement territory disconnection
8. Issue #6: Add forced elimination
9. Issue #8: Fix player state updates

### Phase 3: Testing & Validation
10. Issue #13: Write comprehensive tests

### Phase 4: User Interface
11. Issue #11: Implement frontend UI
12. Issue #12: Complete WebSocket events

### Phase 5: Advanced Features
13. Issue #10: Implement AI players
14. Issue #14: Connect database
15. Issue #15-17: Additional features

---

## 📞 Reporting New Issues

When reporting new issues:

1. **Check existing issues** in this document first
2. **Include:**
   - Clear description of expected vs actual behavior
   - File(s) affected
   - Rule reference from `ringrift_complete_rules.md`
   - Steps to reproduce (if applicable)
   - Code snippets showing the problem
3. **Label priority:** P0 (Critical), P1 (High), P2 (Medium), P3 (Low)

---

**Document Version:** 1.0  
**Maintainer:** Development Team  
**Last Review:** November 13, 2025
