# RingRift Development TODO

**Last Updated:** November 13, 2025  
**Current Phase:** Phase 1 - Core Game Logic Implementation

This document tracks specific development tasks in prioritized order.

---

## 🚀 PHASE 1: Core Game Logic Implementation

**Status:** Not Started  
**Priority:** P0 - CRITICAL  
**Target Completion:** 3-5 weeks

### Task 1.1: Fix BoardState Data Structure ⏳
**Priority:** P0 - CRITICAL | **Estimated:** 1-2 days | **Status:** Not Started

**Files:**
- `src/shared/types/game.ts`
- `src/server/game/BoardManager.ts`

**Subtasks:**
- [ ] Update `BoardState` interface to separate stacks, markers, collapsed spaces
- [ ] Change markers from `Map<string, MarkerInfo>` to `Map<string, number>`
- [ ] Add `collapsedSpaces: Map<string, number>` field
- [ ] Remove `eliminatedRings` from BoardState (should be in Player state)
- [ ] Update all BoardManager methods to work with new structure
- [ ] Update GameEngine to use new BoardState structure
- [ ] Test backward compatibility

**Acceptance Criteria:**
- [ ] BoardState clearly separates rings, markers, and collapsed spaces
- [ ] All existing code compiles with new structure
- [ ] No functionality broken by refactor

**Rule Reference:** N/A - Internal structure improvement

---

### Task 1.2: Implement Marker System ⏳
**Priority:** P0 - CRITICAL | **Estimated:** 2-3 days | **Status:** Not Started

**Dependencies:** Task 1.1 (BoardState fix)

**Files:**
- `src/server/game/BoardManager.ts`
- `src/server/game/GameEngine.ts`

**Subtasks:**
- [ ] Add `setMarker(position, player, board)` to BoardManager
- [ ] Add `getMarker(position, board)` to BoardManager
- [ ] Add `removeMarker(position, board)` to BoardManager
- [ ] Add `flipMarker(position, newPlayer, board)` to BoardManager
- [ ] Add `collapseMarker(position, player, board)` to BoardManager
- [ ] Implement marker placement in `applyMove()` when ring moves
- [ ] Implement marker flipping when moving over opponent markers
- [ ] Implement marker collapsing when moving over own markers
- [ ] Implement marker removal when landing on same-color marker
- [ ] Write unit tests for all marker operations
- [ ] Write integration tests for marker flow during movement

**Acceptance Criteria:**
- [ ] Markers placed at starting position when rings move
- [ ] Opponent markers flip to your color when jumped over
- [ ] Own markers collapse to territory when jumped over
- [ ] Landing on own marker removes it
- [ ] All tests pass

**Rule Reference:** Section 8.3 - Marker Interaction

---

### Task 1.3: Fix Movement Validation ⏳
**Priority:** P0 - CRITICAL | **Estimated:** 2-3 days | **Status:** Not Started

**Dependencies:** Task 1.2 (Marker system)

**Files:**
- `src/server/game/RuleEngine.ts`

**Subtasks:**
- [ ] Implement `validateMovementDistance()` - enforce distance ≥ stack height
- [ ] Fix `validateStackMovement()` to check path for collapsed spaces
- [ ] Implement landing on any valid space beyond markers (not just first)
- [ ] Add validation for marker interactions during path
- [ ] Ensure cannot pass through other rings/stacks
- [ ] Add validation for landing on same-color marker
- [ ] Write unit tests for distance validation
- [ ] Write unit tests for path validation
- [ ] Write unit tests for landing rules
- [ ] Write tests for all edge cases from FAQ Q2

**Acceptance Criteria:**
- [ ] Movement requires distance ≥ stack height
- [ ] Can land on any valid space beyond markers meeting distance
- [ ] Path validation blocks movement through collapsed spaces
- [ ] Path validation blocks movement through other rings
- [ ] Landing on same-color marker is valid
- [ ] All tests pass

**Rule Reference:** Section 8.2, FAQ Q2

---

### Task 1.4: Fix Game Phase Transitions ⏳
**Priority:** P1 - HIGH | **Estimated:** 1-2 days | **Status:** Not Started

**Files:**
- `src/shared/types/game.ts`
- `src/server/game/GameEngine.ts`

**Subtasks:**
- [ ] Update `GamePhase` type: remove `main_game`, add `line_processing`
- [ ] Rewrite `advanceGame()` to follow correct phase flow
- [ ] Implement ring_placement → movement transition
- [ ] Implement movement → capture transition
- [ ] Implement capture → line_processing transition
- [ ] Implement line_processing → territory_processing transition
- [ ] Implement territory_processing → next player transition
- [ ] Add validation to prevent invalid phase transitions
- [ ] Write unit tests for each phase transition
- [ ] Write integration test for complete turn flow

**Acceptance Criteria:**
- [ ] Phases match rules: ring_placement → movement → capture → line_processing → territory_processing
- [ ] Phase transitions enforce correct game flow
- [ ] No invalid phase states possible
- [ ] All tests pass

**Rule Reference:** Section 4, Section 15.2

---

### Task 1.5: Complete Capture System ⏳
**Priority:** P1 - HIGH | **Estimated:** 3-4 days | **Status:** Not Started

**Dependencies:** Task 1.3 (Movement validation)

**Files:**
- `src/server/game/GameEngine.ts`
- `src/server/game/RuleEngine.ts`

**Subtasks:**
- [ ] Distinguish overtaking (rings stay in play) vs elimination (rings removed)
- [ ] Fix cap height comparison in `isValidCapture()`
- [ ] I mplement flexible landing during captures (any valid space beyond target)
- [ ] Implement chain capture detection
- [ ] Implement mandatory chain capture continuation
- [ ] Add captured ring to bottom of capturing stack (overtaking)
- [ ] Implement proper stack merging logic
- [ ] Handle player choice when multiple capture directions available
- [ ] Write unit tests for cap height comparison
- [ ] Write unit tests for chain capture sequences
- [ ] Write tests for 180° reversal pattern (FAQ)
- [ ] Write tests for cyclic pattern (FAQ)

**Acceptance Criteria:**
- [ ] Overtaking adds captured ring to bottom of stack
- [ ] Chain captures are mandatory once started
- [ ] Cap height correctly determines capture eligibility
- [ ] Can land on any valid space beyond target
- [ ] Multiple capture sequences work correctly
- [ ] All tests pass including FAQ examples

**Rule Reference:** Sections 9-10, Section 15.3

---

### Task 1.6: Implement Line Formation ⏳
**Priority:** P1 - HIGH | **Estimated:** 3-4 days | **Status:** Not Started

**Dependencies:** Task 1.2 (Marker system), Task 1.4 (Phase transitions)

**Files:**
- `src/server/game/GameEngine.ts`
- `src/server/game/BoardManager.ts`
- `src/server/game/RuleEngine.ts`

**Subtasks:**
- [ ] Fix `findAllLines()` to detect 4+ markers for 8x8
- [ ] Fix `findAllLines()` to detect 5+ markers for 19x19/hex
- [ ] Implement Option 1: Collapse all + eliminate ring (for exact or longer lines)
- [ ] Implement Option 2: Collapse only required + no elimination (for longer lines only)
- [ ] Add player choice mechanism for longer lines
- [ ] Implement `eliminatePlayerRingOrCap()` method
- [ ] Handle multiple line processing in player-chosen order
- [ ] Check for new lines after each collapse
- [ ] Update player's `eliminatedRings` counter
- [ ] Update player's `territorySpaces` counter
- [ ] Write unit tests for line detection
- [ ] Write unit tests for graduated rewards
- [ ] Write integration tests for multiple lines

**Acceptance Criteria:**
- [ ] Lines of 4+ (8x8) or 5+ (19x19/hex) detected correctly
- [ ] Exactly minimum length: must use Option 1
- [ ] Longer than minimum: player chooses Option 1 or 2
- [ ] Ring elimination tracked correctly
- [ ] Territory spaces tracked correctly
- [ ] Multiple lines processed in correct order
- [ ] All tests pass

**Rule Reference:** Section 11, Section 11.2

---

### Task 1.7: Implement Territory Disconnection ⏳
**Priority:** P1 - HIGH | **Estimated:** 4-5 days | **Status:** Not Started

**Dependencies:** Task 1.2 (Marker system), Task 1.4 (Phase transitions)

**Files:**
- `src/server/game/RuleEngine.ts`
- `src/server/game/BoardManager.ts`

**Subtasks:**
- [ ] Implement `findDisconnectedRegions()` in BoardManager
- [ ] Use Von Neumann adjacency for square boards
- [ ] Use hexagonal adjacency for hex board
- [ ] Detect regions surrounded by collapsed spaces/edges/single-player markers
- [ ] Implement representation check (region lacks active player's stacks)
- [ ] Implement self-elimination prerequisite check
- [ ] Collapse region to moving player's color
- [ ] Collapse border markers to moving player's color
- [ ] Eliminate all rings within region
- [ ] Mandatory self-elimination after region elimination
- [ ] Check for chain reactions (new disconnections)
- [ ] Update player elimination counts
- [ ] Update player territory counts
- [ ] Write unit tests for disconnection detection
- [ ] Write unit tests for representation checking
- [ ] Write unit tests for prerequisite validation
- [ ] Write tests for chain reactions
- [ ] Write test for example from Section 16.8.6

**Acceptance Criteria:**
- [ ] Correctly detects disconnected regions
- [ ] Uses correct adjacency type (Von Neumann for square, hex for hex)
- [ ] Representation check works correctly
- [ ] Self-elimination prerequisite prevents illegal disconnections
- [ ] All rings eliminated and counted correctly
- [ ] Border markers collapsed correctly
- [ ] Chain reactions processed correctly
- [ ] All tests pass including rules example

**Rule Reference:** Section 12, Section 12.2, FAQ Q15

---

### Task 1.8: Add Forced Elimination ⏳
**Priority:** P1 - HIGH | **Estimated:** 1 day | **Status:** Not Started

**Dependencies:** Task 1.6 (Line formation - for eliminatePlayerRingOrCap method)

**Files:**
- `src/server/game/GameEngine.ts`
- `src/server/game/RuleEngine.ts`

**Subtasks:**
- [ ] Add turn start check: player has no valid moves but controls stacks
- [ ] Implement forced cap elimination
- [ ] Update player's `eliminatedRings` counter
- [ ] Add player choice for which stack to eliminate from
- [ ] Write unit tests for forced elimination detection
- [ ] Write unit tests for cap elimination
- [ ] Write integration test for forced elimination scenario

**Acceptance Criteria:**
- [ ] Detects when player has no valid moves but controls stacks
- [ ] Forces cap elimination before turn ends
- [ ] Player can choose which stack to eliminate from
- [ ] Elimination counted correctly
- [ ] All tests pass

**Rule Reference:** Section 4.4, FAQ Q24

---

### Task 1.9: Fix Player State Updates ⏳
**Priority:** P1 - HIGH | **Estimated:** 1 day | **Status:** Not Started

**Dependencies:** Tasks 1.1-1.8 (all previous tasks)

**Files:**
- `src/server/game/GameEngine.ts`

**Subtasks:**
- [ ] Decrement `ringsInHand` when ring is placed
- [ ] Increment `eliminatedRings` when rings are eliminated
- [ ] Update `territorySpaces` when territory is collapsed
- [ ] Ensure all player state fields remain synchronized
- [ ] Add validation to prevent negative values
- [ ] Write unit tests for state updates
- [ ] Write integration tests for state synchronization

**Acceptance Criteria:**
- [ ] `ringsInHand` decreases on placement
- [ ] `eliminatedRings` increases on elimination
- [ ] `territorySpaces` increases on collapse
- [ ] State always matches actual game board
- [ ] No negative values possible
- [ ] All tests pass

**Rule Reference:** N/A - Correct state tracking

---

## 🧪 PHASE 2: Testing & Validation

**Status:** Not Started  
**Priority:** P1 - HIGH  
**Target Completion:** 2-3 weeks after Phase 1

### Task 2.1: Unit Tests
- [ ] BoardManager position utilities
- [ ] BoardManager adjacency calculations  
- [ ] BoardManager line detection
- [ ] RuleEngine movement validation
- [ ] RuleEngine capture validation
- [ ] RuleEngine line formation
- [ ] RuleEngine territory disconnection
- [ ] GameEngine state transitions

### Task 2.2: Integration Tests
- [ ] Complete turn sequence
- [ ] Ring placement → movement → capture flow
- [ ] Line formation → ring elimination
- [ ] Territory disconnection → ring elimination
- [ ] Chain capture sequences
- [ ] Forced elimination scenarios

### Task 2.3: Scenario Tests
- [ ] 180° reversal capture (FAQ example)
- [ ] Cyclic capture pattern (FAQ example)
- [ ] Territory disconnection (Section 16.8.6)
- [ ] Graduated line rewards
- [ ] Victory conditions
- [ ] All edge cases from FAQ Q1-Q24

---

## 🎨 PHASE 3: Frontend Implementation

**Status:** Not Started  
**Priority:** P2 - MEDIUM  
**Target Completion:** 3-4 weeks after Phase 2

### Task 3.1: Board Rendering
- [ ] Square grid component (8x8)
- [ ] Square grid component (19x19)
- [ ] Hexagonal grid component
- [ ] Cell/space components
- [ ] Coordinate overlay
- [ ] Responsive sizing

### Task 3.2: Game Piece Visualization
- [ ] Ring stack component
- [ ] Marker display component
- [ ] Collapsed space display
- [ ] Player color system
- [ ] Stack height indicators

### Task 3.3: User Interaction
- [ ] Ring placement controls
- [ ] Move selection (drag or click)
- [ ] Valid move highlighting
- [ ] Move confirmation dialog
- [ ] Undo/redo buttons

### Task 3.4: Game State Display
- [ ] Current player indicator
- [ ] Ring count displays
- [ ] Territory statistics panel
- [ ] Move history list
- [ ] Timer/clock display

---

## 🤖 PHASE 4: Advanced Features

**Status:** Not Started  
**Priority:** P2 - MEDIUM  
**Target Completion:** 4-6 weeks after Phase 3

### Task 4.1: AI Implementation
- [ ] Random move AI (levels 1-3)
- [ ] Heuristic AI (levels 4-7)
- [ ] MCTS AI (levels 8-10)
- [ ] Position evaluation function
- [ ] Opening book

### Task 4.2: WebSocket Completion
- [ ] Move broadcasting
- [ ] Game state synchronization
- [ ] Spectator events
- [ ] Reconnection logic
- [ ] Game lobby system

### Task 4.3: Database Integration
- [ ] Game persistence
- [ ] User data storage
- [ ] Move history recording
- [ ] Statistics tracking
- [ ] Replay storage

### Task 4.4: Additional Features
- [ ] Spectator mode
- [ ] Replay viewer
- [ ] Game analysis tools
- [ ] Rating system
- [ ] Tournament support

---

## 📊 Progress Tracking

### Phase 1 Progress: 0/9 tasks (0%)
- [ ] 1.1 Fix BoardState (0%)
- [ ] 1.2 Marker System (0%)
- [ ] 1.3 Movement Validation (0%)
- [ ] 1.4 Phase Transitions (0%)
- [ ] 1.5 Capture System (0%)
- [ ] 1.6 Line Formation (0%)
- [ ] 1.7 Territory Disconnection (0%)
- [ ] 1.8 Forced Elimination (0%)
- [ ] 1.9 Player State (0%)

### Phase 2 Progress: 0/3 tasks (0%)
### Phase 3 Progress: 0/4 tasks (0%)
### Phase 4 Progress: 0/4 tasks (0%)

### Overall Progress: 0/20 major tasks (0%)

---

## 🎯 Current Sprint

**Sprint Goal:** Complete Phase 1 Core Logic  
**Sprint Duration:** 3-5 weeks  
**Current Task:** None - Awaiting start

**This Week:**
- [ ] Task not yet assigned

**Next Week:**
- [ ] Task not yet assigned

---

## 📝 Notes

### Implementation Order Rationale

1. **BoardState first** - All other tasks depend on correct data structure
2. **Marker system early** - Fundamental to movement and territory mechanics
3. **Movement validation** - Required before captures work correctly
4. **Phase transitions** - Needed for proper game flow
5. **Capture system** - Builds on movement validation
6. **Line formation** - Requires markers and phases
7. **Territory disconnection** - Most complex, requires all previous
8. **Forced elimination** - Uses infrastructure from line formation
9. **Player state** - Final synchronization of all mechanics

### Testing Strategy

- Write tests alongside implementation (TDD approach)
- Run tests frequently during development
- Achieve 80%+ code coverage before Phase 2
- Use scenario tests from rules document to validate correctness

### Risk Mitigation

- Complex territory logic may take longer than estimated
- Marker system integration may uncover edge cases
- Plan buffer time for unexpected issues

---

**Document Version:** 1.0  
**Last Updated:** November 13, 2025  
**Maintained By:** Development Team
