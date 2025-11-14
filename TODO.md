# RingRift Development TODO

**Last Updated:** November 13, 2025  
**Current Phase:** Phase 1 - Core Game Logic Implementation  
**Overall Progress:** 3/9 Phase 1 tasks completed (33%)

This document is the **single source of truth** for RingRift development tracking.

---

## 🚀 PHASE 1: Core Game Logic Implementation

**Status:** IN PROGRESS (75% complete, critical gaps remain)  
**Priority:** P0 - CRITICAL  
**Target Completion:** 2-3 weeks remaining for completion

**VERIFIED ACTUAL STATUS (November 13, 2025):**
- ✅ Basic mechanics working (marker system, movement, basic captures, lines, territory)
- ⚠️ Chain captures NOT enforced (mandatory continuation missing)
- ⚠️ Player choice system NOT implemented (all choices default)
- ⚠️ Strategic gameplay reduced due to missing player agency

---

### Task 1.1: Fix BoardState Data Structure ✅
**Priority:** P0 - CRITICAL | **Estimated:** 1-2 days | **Status:** COMPLETED

**Files:**
- `src/shared/types/game.ts` ✅
- `src/server/game/BoardManager.ts` ✅

**Subtasks:**
- [x] Update `BoardState` interface to separate stacks, markers, collapsed spaces
- [x] Add `MarkerType` type definition ('regular' | 'collapsed')
- [x] Add `collapsedSpaces: Map<string, number>` field
- [x] Keep `MarkerInfo` interface with type field (better than plain number)
- [x] Update all BoardManager methods to work with new structure
- [x] Update `createBoard()` to initialize collapsedSpaces

**Acceptance Criteria:**
- [x] BoardState clearly separates rings, markers, and collapsed spaces
- [x] All existing code compiles with new structure
- [x] MarkerInfo includes type field for extensibility

**Rule Reference:** N/A - Internal structure improvement

**Completed:** November 13, 2025

---

### Task 1.2: Implement Marker System ✅
**Priority:** P0 - CRITICAL | **Estimated:** 2-3 days | **Status:** COMPLETED

**Dependencies:** Task 1.1 (BoardState fix) ✅

**Files:**
- `src/server/game/BoardManager.ts` ✅
- `src/server/game/GameEngine.ts` ✅
- `src/server/game/RuleEngine.ts` ✅

**Subtasks:**
- [x] Add `setMarker(position, player, board)` to BoardManager
- [x] Add `getMarker(position, board)` to BoardManager
- [x] Add `removeMarker(position, board)` to BoardManager
- [x] Add `flipMarker(position, newPlayer, board)` to BoardManager
- [x] Add `collapseMarker(position, player, board)` to BoardManager
- [x] Add `getCollapsedSpace(position, board)` to BoardManager
- [x] Add `setCollapsedSpace(position, player, board)` to BoardManager
- [x] Add `isCollapsedSpace(position, board)` to BoardManager
- [x] Implement marker placement in `applyMove()` when ring moves
- [x] Implement marker flipping when moving over opponent markers
- [x] Implement marker collapsing when moving over own markers
- [x] Implement marker removal when landing on same-color marker
- [x] Update path validation to respect collapsed spaces
- [ ] Write unit tests for all marker operations (deferred to Phase 2)
- [ ] Write integration tests for marker flow during movement (deferred to Phase 2)

**Acceptance Criteria:**
- [x] Marker methods available in BoardManager
- [x] Markers placed at starting position when rings move
- [x] Opponent markers flip to your color when jumped over
- [x] Own markers collapse to territory when jumped over
- [x] Landing on own marker removes it
- [x] Collapsed spaces block movement
- [ ] All tests pass (deferred to Phase 2)

**Rule Reference:** Section 8.3 - Marker Interaction, Section 4.2.1

**Progress:** 13/15 subtasks completed (87%) - Core functionality complete

**Completed:** November 13, 2025

---

### Task 1.3: Fix Movement Validation ✅
**Priority:** P0 - CRITICAL | **Estimated:** 2-3 days | **Status:** COMPLETED

**Dependencies:** Task 1.2 (Marker system) ✅

**Files:**
- `src/server/game/RuleEngine.ts` ✅

**Subtasks:**
- [x] Implement minimum distance validation - enforce distance ≥ stack height
- [x] Fix `validateStackMovement()` to check path for collapsed spaces
- [x] Implement landing on any valid space beyond markers (not just first)
- [x] Add validation for marker interactions during path
- [x] Ensure cannot pass through other rings/stacks
- [x] Add validation for landing on same-color marker
- [x] Add `isPathClear()` helper method for path validation
- [x] Add `getPathPositions()` helper for path calculation
- [x] Fix TypeScript strict optional property handling
- [ ] Write unit tests for distance validation (deferred to Phase 2)
- [ ] Write unit tests for path validation (deferred to Phase 2)
- [ ] Write unit tests for landing rules (deferred to Phase 2)
- [ ] Write tests for all edge cases from FAQ Q2 (deferred to Phase 2)

**Acceptance Criteria:**
- [x] Movement requires distance ≥ stack height
- [x] Can land on any valid space beyond markers meeting distance
- [x] Path validation blocks movement through collapsed spaces
- [x] Path validation blocks movement through other rings
- [x] Landing on same-color marker is valid
- [x] Landing on opponent marker is blocked (returns false)
- [x] Markers can be passed through (get flipped/collapsed)
- [ ] All tests pass (deferred to Phase 2)

**Rule Reference:** Section 8.2, FAQ Q2

**Progress:** 9/13 subtasks completed (69%) - Core functionality complete

**Completed:** November 13, 2025

---

### Task 1.4: Fix Game Phase Transitions ✅
**Priority:** P1 - HIGH | **Estimated:** 1-2 days | **Status:** COMPLETED

**Dependencies:** Task 1.3 (Movement validation) ✅

**Files:**
- `src/shared/types/game.ts` ✅
- `src/server/game/GameEngine.ts` ✅

**Subtasks:**
- [x] Verify GamePhase type (was already correct - no 'main_game', has 'line_processing')
- [x] Rewrite `advanceGame()` to follow correct phase flow
- [x] Implement ring_placement → movement transition
- [x] Implement movement → capture transition (with capture availability check)
- [x] Implement capture → line_processing transition
- [x] Implement line_processing → territory_processing transition
- [x] Implement territory_processing → next player transition
- [x] Add `hasValidCaptures()` helper method
- [x] Add `getAdjacentPositions()` helper method
- [x] Add phase transition logic based on player state (rings in hand, stacks on board)
- [ ] Write unit tests for each phase transition (deferred to Phase 2)
- [ ] Write integration test for complete turn flow (deferred to Phase 2)

**Acceptance Criteria:**
- [x] Phases match rules: ring_placement → movement → capture → line_processing → territory_processing
- [x] Phase transitions enforce correct game flow
- [x] Movement phase checks for valid captures before transitioning
- [x] Territory processing correctly transitions to next player
- [x] Next player's starting phase determined by ring/stack state
- [ ] All tests pass (deferred to Phase 2)

**Rule Reference:** Section 4, Section 15.2

**Implemented Phase Flow:**
```
1. Ring Placement → Movement (always)
2. Movement → Capture (if valid captures exist) OR Line Processing (if no captures)
3. Capture → Line Processing (always)
4. Line Processing → Territory Processing (always)
5. Territory Processing → Next Player's Turn
   - Start at Ring Placement if: no stacks on board OR has rings in hand
   - Start at Movement if: no rings in hand
```

**Progress:** 10/12 subtasks completed (83%) - Core functionality complete

**Completed:** November 13, 2025

---

### Task 1.5: Complete Capture System ⏳
**Priority:** P0 - CRITICAL | **Estimated:** 3-4 days | **Status:** INCOMPLETE

**Dependencies:** Task 1.3 (Movement validation) ✅

**Files:**
- `src/server/game/GameEngine.ts` ⚠️ (basic capture only)
- `src/server/game/RuleEngine.ts` ⚠️ (validation incomplete)
- `src/shared/types/game.ts` ✅ (updated)

**Subtasks:**
- [x] Distinguish overtaking (rings stay in play) vs elimination (rings removed)
- [x] Implement cap height comparison (≥) in capture validation
- [x] Implement flexible landing during captures (any valid space beyond target)
- [x] Add captured ring to bottom of capturing stack (overtaking)
- [x] Implement proper stack merging logic for captures
- [x] Add `captureTarget` field to Move type
- [x] Implement `calculateCapHeight()` method
- [x] Implement `validateCapture()` with full validation
- [x] Implement `isValidCaptureDirection()` for path checking
- [x] Implement `isValidCaptureLanding()` for landing validation
- [x] Implement `isBeyondTarget()` helper method
- [x] Update `applyMove()` to handle overtaking_capture move type
- [x] Implement capture detection in `hasValidCaptures()`
- [ ] **CRITICAL: Implement mandatory chain capture continuation in GameEngine**
- [ ] **CRITICAL: Handle player choice when multiple capture directions available**
- [ ] Write unit tests for cap height comparison (deferred to Phase 2)
- [ ] Write unit tests for chain capture sequences (deferred to Phase 2)
- [ ] Write tests for 180° reversal pattern (FAQ Q15.3.1) (deferred to Phase 2)
- [ ] Write tests for cyclic pattern (FAQ Q15.3.2) (deferred to Phase 2)

**Acceptance Criteria:**
- [x] Overtaking adds captured ring to bottom of stack
- [ ] **Chain captures are mandatory once started** ❌ NOT IMPLEMENTED
- [x] Cap height correctly determines capture eligibility
- [x] Can land on any valid space beyond target
- [ ] **Multiple capture sequences work correctly** ❌ NOT IMPLEMENTED
- [ ] All tests pass including FAQ examples (deferred to Phase 2)

**Rule Reference:** Sections 9-10, Section 15.3, FAQ Q14

**Key Rules:**
- Overtaking: Top ring of target stack added to bottom of capturing stack ✅
- Elimination: Rings permanently removed (line formations, territory disconnections)
- Chain captures: Mandatory once started, player chooses direction at each step ❌

**Progress:** 13/19 subtasks completed (40% functional) - Single captures work, chain captures NOT enforced

**Status:** INCOMPLETE - Chain capture logic is critical missing feature

---

### Task 1.6: Implement Line Formation ⏳
**Priority:** P1 - HIGH | **Estimated:** 3-4 days | **Status:** INCOMPLETE

**Dependencies:** Task 1.2 (Marker system) ✅, Task 1.4 (Phase transitions) ✅

**Files:**
- `src/server/game/GameEngine.ts` ⚠️ (defaulting to Option 2)
- `src/server/game/BoardManager.ts` ✅
- `src/server/game/RuleEngine.ts` ⚠️

**Subtasks:**
- [x] Fix `findAllLines()` to detect 4+ consecutive markers for 8x8 (CRITICAL: markers not stacks)
- [x] Fix `findAllLines()` to detect 5+ consecutive markers for 19x19/hex
- [x] Implement Option 1: Collapse all + eliminate ring (for exact or longer lines)
- [x] Implement Option 2: Collapse only required + no elimination (for longer lines only)
- [ ] **CRITICAL: Add player choice mechanism for longer lines** (currently defaults to Option 2)
- [x] Implement `eliminatePlayerRingOrCap()` method
- [ ] **CRITICAL: Handle multiple line processing in player-chosen order** (uses first found)
- [x] Check for new lines after each collapse
- [x] Update player's `eliminatedRings` counter
- [x] Update player's `territorySpaces` counter
- [ ] Write unit tests for line detection (deferred to Phase 2)
- [ ] Write unit tests for graduated rewards (deferred to Phase 2)
- [ ] Write integration tests for multiple lines (deferred to Phase 2)

**Acceptance Criteria:**
- [x] Lines of 4+ (8x8) or 5+ (19x19/hex) detected correctly
- [x] Exactly minimum length: must use Option 1
- [ ] **Longer than minimum: player CHOOSES Option 1 or 2** ❌ Defaults to Option 2
- [x] Ring elimination tracked correctly
- [x] Territory spaces tracked correctly
- [ ] **Multiple lines processed in PLAYER-CHOSEN order** ❌ Uses first found
- [ ] All tests pass (deferred to Phase 2)

**Rule Reference:** Section 11, Section 11.2

**Graduated Rewards:**
- **8x8**: Exactly 4 markers → must collapse all + eliminate ring. 5+ → player chooses Option 1 or 2
- **19x19/Hex**: Exactly 5 markers → must collapse all + eliminate ring. 6+ → player chooses Option 1 or 2
- **Option 1**: Collapse entire line + eliminate one ring/cap
- **Option 2**: Collapse only required number (4 or 5) + no ring elimination

**Progress:** 10/13 subtasks completed (70% functional) - Detection works, player choices missing

**Status:** INCOMPLETE - Player choice mechanism needed for strategic gameplay

---

### Task 1.7: Implement Territory Disconnection ✅
**Priority:** P1 - HIGH | **Estimated:** 4-5 days | **Status:** COMPLETED

**Dependencies:** Task 1.2 (Marker system) ✅, Task 1.4 (Phase transitions) ✅

**Files:**
- `src/server/game/GameEngine.ts` ✅
- `src/server/game/BoardManager.ts` ✅

**Subtasks:**
- [x] Implement `findDisconnectedRegions()` in BoardManager
- [x] Use Von Neumann adjacency for square boards (4-direction orthogonal)
- [x] Use hexagonal adjacency for hex board (6-direction) - ALL territory operations
- [x] Detect regions surrounded by collapsed spaces/edges/single-player markers
- [x] Implement representation check (region lacks active player's stacks)
- [x] Implement self-elimination prerequisite check (canProcessDisconnectedRegion)
- [x] Collapse region to moving player's color
- [x] Collapse border markers to moving player's color
- [x] Eliminate all rings within region
- [x] Mandatory self-elimination after region elimination
- [x] Check for chain reactions (new disconnections)
- [x] Update player elimination counts
- [ ] Update player territory counts (deferred to Task 1.9)
- [ ] Write unit tests for disconnection detection (all board types) (deferred to Phase 2)
- [ ] Write unit tests for representation checking (all board types) (deferred to Phase 2)
- [ ] Write unit tests for prerequisite validation (all board types) (deferred to Phase 2)
- [ ] Write tests for chain reactions (all board types) (deferred to Phase 2)
- [ ] Write test for example from Section 16.8.6 (deferred to Phase 2)
- [ ] Write hexagonal-specific territory tests (deferred to Phase 2)

**Acceptance Criteria:**
- [x] Correctly detects disconnected regions
- [x] Uses correct adjacency type (Von Neumann for square, hex for hex)
- [x] Representation check works correctly
- [x] Self-elimination prerequisite prevents illegal disconnections
- [x] All rings eliminated and counted correctly
- [x] Border markers collapsed correctly
- [x] Chain reactions processed correctly
- [ ] All tests pass including rules example (deferred to Phase 2)

**Rule Reference:** Section 12, Section 12.2, FAQ Q15

**Key Concepts:**
- **Physical Disconnection**: Region cut off by collapsed spaces, board edges, or continuous single-player marker border
- **Representation**: Region must lack at least one active player's ring stacks
- **Self-Elimination Prerequisite**: Must have at least one ring/cap outside region before processing
- **Processing**: Collapse region + border, eliminate all internal rings, then mandatory self-elimination

**Hexagonal Board Note:** Territory connectivity uses 6-direction hexagonal adjacency (unlike square boards which use 4-direction Von Neumann). This is critical for correct disconnection detection.

**Implementation Details:**
- `findDisconnectedRegions()` - Main detection method using flood fill with representation checks
- `exploreRegion()` - Flood fill helper to find connected regions
- `analyzeRegionBorder()` - Checks if border meets disconnection criteria
- `getRepresentedPlayers()` - Determines which players have stacks in region
- `getBorderMarkerPositions()` - Identifies border markers for collapse
- `processDisconnectedRegions()` - Main processing loop with chain reaction support
- `canProcessDisconnectedRegion()` - Self-elimination prerequisite validation
- `processOneDisconnectedRegion()` - Processes single region with all steps

**Progress:** 13/20 subtasks completed (65%) - Core functionality complete

**Completed:** November 13, 2025

---

### Task 1.8: Add Forced Elimination ✅
**Priority:** P1 - HIGH | **Estimated:** 1 day | **Status:** COMPLETED

**Dependencies:** Task 1.6 (Line formation - for eliminatePlayerRingOrCap method) ✅

**Files:**
- `src/server/game/GameEngine.ts` ✅

**Subtasks:**
- [x] Add turn start check: player has no valid moves but controls stacks
- [x] Implement forced cap elimination
- [x] Update player's `eliminatedRings` counter (reuses eliminatePlayerRingOrCap)
- [ ] Add player choice for which stack to eliminate from (TODO: currently uses first stack)
- [x] Implement `hasValidPlacements()` helper method
- [x] Implement `hasValidMovements()` helper method
- [x] Implement `getAllDirections()` helper method
- [x] Implement `hasValidActions()` combined check method
- [x] Implement `processForcedElimination()` method
- [x] Integrate forced elimination check in game flow (after territory_processing)
- [ ] Write unit tests for forced elimination detection (deferred to Phase 2)
- [ ] Write unit tests for cap elimination (deferred to Phase 2)
- [ ] Write integration test for forced elimination scenario (deferred to Phase 2)

**Acceptance Criteria:**
- [x] Detects when player has no valid moves but controls stacks
- [x] Forces cap elimination at turn start (before next player begins)
- [ ] Player can choose which stack to eliminate from (defaults to first for now)
- [x] Elimination counted correctly (reuses existing logic)
- [x] Victory check after forced elimination
- [ ] All tests pass (deferred to Phase 2)

**Rule Reference:** Section 4.4, FAQ Q24

**Implementation Details:**
- Check occurs in `advanceGame()` after `territory_processing` phase completes
- Before next player starts their turn, checks if they have no valid actions but control stacks
- If true, calls `processForcedElimination()` to eliminate a cap
- After elimination, checks victory conditions (game may end)
- If game continues, advances to actual next player and sets their starting phase

**Progress:** 10/13 subtasks completed (77%) - Core functionality complete

**Completed:** November 13, 2025

---

### Task 1.9: Fix Player State Updates ✅
**Priority:** P1 - HIGH | **Estimated:** 1 day | **Status:** COMPLETED

**Dependencies:** Tasks 1.1-1.8 (all previous tasks) ✅

**Files:**
- `src/server/game/GameEngine.ts` ✅

**Subtasks:**
- [x] Decrement `ringsInHand` when ring is placed
- [x] Increment `eliminatedRings` when rings are eliminated
- [x] Update `territorySpaces` when territory is collapsed
- [x] Ensure all player state fields remain synchronized
- [ ] Add validation to prevent negative values (deferred to Phase 2)
- [ ] Write unit tests for state updates (deferred to Phase 2)
- [ ] Write integration tests for state synchronization (deferred to Phase 2)

**Acceptance Criteria:**
- [x] `ringsInHand` decreases on placement
- [x] `eliminatedRings` increases on elimination
- [x] `territorySpaces` increases on collapse
- [x] State always matches actual game board
- [ ] No negative values possible (deferred to Phase 2)
- [ ] All tests pass (deferred to Phase 2)

**Rule Reference:** N/A - Correct state tracking

**Implementation Details:**
- Added `updatePlayerEliminatedRings()` helper method
- Added `updatePlayerTerritorySpaces()` helper method
- Updated `applyMove()` to decrement `ringsInHand` on ring placement
- Updated `eliminatePlayerRingOrCap()` to call `updatePlayerEliminatedRings()`
- Updated `collapseLineMarkers()` to call `updatePlayerTerritorySpaces()`
- Updated `processOneDisconnectedRegion()` to call both helpers for complete tracking

**Progress:** 4/7 subtasks completed (57%) - Core state synchronization complete

**Completed:** November 13, 2025

---

### Task 1.11: Player Choice System ❌
**Priority:** P0 - CRITICAL | **Estimated:** 1-2 weeks | **Status:** NOT STARTED

**Dependencies:** Tasks 1.5, 1.6, 1.7 (features requiring choices)

**Rationale:** Currently ALL player decisions default to first option, eliminating strategic gameplay. This is architectural gap affecting multiple features.

**Files to Create/Modify:**
- `src/shared/types/game.ts` (add PlayerChoice types)
- `src/server/game/PlayerInteractionManager.ts` (NEW - handles choices)
- `src/server/game/GameEngine.ts` (integrate choice system)
- `src/client/components/PlayerChoiceDialog.tsx` (NEW - UI for choices)

**Subtasks:**
- [ ] Design PlayerChoice interface and types
- [ ] Create async choice request/response system
- [ ] Implement timeout handling for choices
- [ ] Add choice validation logic
- [ ] Integrate with GameEngine for:
  - [ ] Line processing order selection (multiple lines)
  - [ ] Graduated line reward choice (Option 1 vs Option 2)
  - [ ] Ring/cap elimination selection
  - [ ] Disconnected region processing order
  - [ ] Capture direction selection (multiple valid captures)
- [ ] Create UI components for each choice type
- [ ] Add AI decision logic for each choice type
- [ ] Write tests for choice system

**Acceptance Criteria:**
- [ ] Human players prompted for all strategic choices
- [ ] AI players make automatic decisions
- [ ] Choices have configurable timeouts
- [ ] Invalid choices rejected with clear errors
- [ ] All game mechanics use choice system (no defaults)

**Rule References:**
- Section 11.2 - Graduated line rewards require player choice
- Section 10.3 - Chain captures require direction choice
- Section 12.2 - Region processing requires order choice

**Impact:** CRITICAL - Without this, game lacks strategic depth

---

### Task 1.10: Hexagonal Board Support Validation ✅
**Priority:** P1 - HIGH | **Estimated:** 2-3 days | **Status:** COMPLETED

**Dependencies:** Tasks 1.1-1.9 (all core mechanics) ✅

**Files:**
- `src/server/game/BoardManager.ts` ✅
- `src/server/game/test-hexagonal-validation.ts` ✅

**Subtasks:**
- [x] Verify hexagonal position generation (331 spaces for size=11)
- [x] Validate cube coordinate system (x + y + z = 0)
- [x] Test 6-direction hexagonal adjacency in all contexts
- [x] Verify line detection uses 3 axes (not 4 like square boards)
- [x] Confirm movement validation handles hexagonal distances correctly
- [x] Validate territory disconnection uses hexagonal (6-direction) adjacency
- [x] Fix position generation algorithm (size-1 for radius)
- [x] Fix edge detection methods
- [x] Create comprehensive validation test suite
- [ ] Write integration tests for hexagonal board gameplay (deferred to Phase 2)
- [ ] Document hexagonal coordinate system for developers (deferred)

**Acceptance Criteria:**
- [x] Hexagonal board generates correct 331 positions
- [x] All adjacency calculations use 6-direction hexagonal
- [x] Line formation correctly identifies 3 axes (not 4)
- [x] Territory disconnection uses hexagonal adjacency
- [x] All game mechanics work correctly on hexagonal board
- [x] Tests pass for hexagonal-specific scenarios
- [ ] Documentation clearly explains cube coordinate system (deferred)

**Rule Reference:** Section 16.9 - Comparing the Three Editions, Hexagonal Version specs

**Hexagonal-Specific Rules:**
- **Board Size**: 11 spaces per side, 331 total spaces ✅
- **Adjacency**: 6-direction for movement, lines, AND territory (unified) ✅
- **Line Axes**: 3 main axes (not 4 like square boards) ✅
- **Coordinates**: Cube coordinates (x, y, z) where x + y + z = 0 ✅
- **Line Length**: 5+ markers (same as 19x19)
- **Rings**: 36 per player (same as 19x19)

**Critical Differences from Square Boards:**
1. Territory uses hexagonal (6-dir) not Von Neumann (4-dir) ✅
2. Only 3 line axes instead of 4 ✅
3. Cube coordinate system instead of Cartesian ✅
4. Different edge detection logic ✅

**Implementation Details:**
- Fixed position generation using `radius = size - 1` (10 for size=11)
- Formula: 3r² + 3r + 1 = 3(100) + 30 + 1 = 331 positions
- Edge positions correctly identified at distance=10 from center
- All 8 validation tests passing

**Progress:** 9/11 subtasks completed (82%) - Core validation complete

**Completed:** November 13, 2025

---

## 🧪 PHASE 0: Testing Foundation (CRITICAL - NEW)

**Status:** Not Started  
**Priority:** P0 - CRITICAL  
**Target Completion:** 1-2 weeks (PARALLEL to Phase 1)

**Rationale:** Testing infrastructure must be in place BEFORE completing core logic to enable test-driven development and prevent regressions.

### Task 0.1: Testing Framework Setup
- [ ] Install and configure Jest/Vitest
- [ ] Set up test coverage reporting (target 80%+)
- [ ] Create test utilities and fixtures
- [ ] Configure TypeScript for tests
- [ ] Add test scripts to package.json

### Task 0.2: CI/CD Pipeline
- [ ] Create GitHub Actions workflow
- [ ] Add linting step (ESLint)
- [ ] Add type checking step (tsc)
- [ ] Add unit test step
- [ ] Add coverage reporting
- [ ] Set up pre-commit hooks

### Task 0.3: Initial Test Coverage
- [ ] Write tests for BoardManager utilities
- [ ] Write tests for existing GameEngine methods
- [ ] Write tests for RuleEngine validation
- [ ] Write tests for shared type utilities
- [ ] Document testing patterns

---

## 🧪 PHASE 2: Testing & Validation

**Status:** Not Started  
**Priority:** P1 - HIGH  
**Target Completion:** 2-3 weeks after Phase 1

**Note:** Phase 0 must be complete before this phase. Phase 2 focuses on comprehensive test coverage for all implemented features.

### Task 2.1: Unit Tests
- [ ] BoardManager position utilities
- [ ] BoardManager adjacency calculations (Moore, Von Neumann, Hexagonal)
- [ ] BoardManager line detection
- [ ] BoardManager territory disconnection detection
- [ ] RuleEngine movement validation
- [ ] RuleEngine capture validation
- [ ] RuleEngine line formation
- [ ] RuleEngine territory disconnection
- [ ] GameEngine state transitions
- [ ] GameEngine marker mechanics

### Task 2.2: Integration Tests
- [ ] Complete turn sequence
- [ ] Ring placement → movement → capture flow
- [ ] Line formation → ring elimination
- [ ] Territory disconnection → ring elimination
- [ ] Chain capture sequences
- [ ] Forced elimination scenarios
- [ ] Multiple player scenarios
- [ ] Victory condition triggers

### Task 2.3: Scenario Tests (From Rules Document)
- [ ] FAQ Q15.3.1: 180° reversal capture pattern (square boards)
- [ ] FAQ Q15.3.2: Cyclic capture pattern (square boards)
- [ ] Section 16.8.6: Territory disconnection example (19x19, Von Neumann)
- [ ] Section 11.2: Graduated line rewards scenarios (all board types)
- [ ] Section 16.8.7: Victory through territory control (all board types)
- [ ] Section 16.8.8: Chain reaction example (all board types)
- [ ] All FAQ scenarios Q1-Q24
- [ ] Victory through ring elimination (all board types)
- [ ] Hexagonal-specific capture patterns (6-direction)
- [ ] Hexagonal line formation (3 axes instead of 4)
- [ ] Hexagonal territory disconnection (6-direction adjacency)

### Task 2.4: Edge Case Tests
- [ ] FAQ Q11: Stalemate with rings in hand
- [ ] FAQ Q8, Q24: No valid moves forcing elimination
- [ ] FAQ Q12: Chain capture eliminating all player rings
- [ ] FAQ Q23: Self-elimination prerequisite failing
- [ ] Multiple disconnected regions simultaneously
- [ ] Simultaneous line and territory events
- [ ] Board edge cases
- [ ] Maximum stack heights

---

## 🤖 PHASE 1.5: AI Engine Implementation (NEW)

**Status:** Not Started  
**Priority:** P1 - HIGH  
**Target Completion:** 2-3 weeks after Phase 1

**Rationale:** AI opponents needed for single-player mode. Start with TypeScript implementation (no microservice needed for MVP).

### Task 1.5a: AI Infrastructure
- [ ] Design AI player interface
- [ ] Create AIEngine base class
- [ ] Implement difficulty scaling system (1-10)
- [ ] Add AI move timing controls
- [ ] Create position evaluation framework

### Task 1.5b: Basic AI Implementation
- [ ] Implement RandomAI (difficulty 1-2)
  - [ ] Random valid move selection
  - [ ] Basic move filtering
- [ ] Implement HeuristicAI (difficulty 3-5)
  - [ ] Material evaluation
  - [ ] Territory evaluation
  - [ ] Mobility evaluation
  - [ ] Simple position scoring

### Task 1.5c: Advanced AI (Optional for MVP)
- [ ] Implement MinimaxAI (difficulty 6-8)
  - [ ] Alpha-beta pruning
  - [ ] Position evaluation
  - [ ] Move ordering
- [ ] Create opening book system
- [ ] Add endgame heuristics

### Task 1.5d: AI Testing
- [ ] Write unit tests for each AI level
- [ ] Test AI move legality
- [ ] Verify difficulty scaling
- [ ] Performance benchmarks

**Files:**
- `src/server/game/ai/AIEngine.ts`
- `src/server/game/ai/AIPlayer.ts`
- `src/server/game/ai/RandomAI.ts`
- `src/server/game/ai/HeuristicAI.ts`
- `src/server/game/ai/evaluators/`

---

## 🎨 PHASE 3: Frontend Implementation

**Status:** Not Started  
**Priority:** P1 - HIGH (increased from P2)  
**Target Completion:** 3-4 weeks after Phase 1.5

### Task 3.1: Board Rendering
- [ ] Square grid component (8x8)
- [ ] Square grid component (19x19)
- [ ] Hexagonal grid component (11 spaces per side, 331 total)
  - [ ] Implement cube coordinate system (x, y, z)
  - [ ] Hexagonal cell rendering
  - [ ] 6-direction adjacency visualization
  - [ ] Proper hexagonal grid layout algorithm
- [ ] Cell/space components (unified for square and hex)
- [ ] Coordinate overlay (support both square and cube coordinates)
- [ ] Responsive sizing (handle different board shapes)
- [ ] Visual polish and animations
- [ ] Board type selector UI

### Task 3.2: Game Piece Visualization
- [ ] Ring stack component (show individual rings)
- [ ] Marker display component (player-colored markers)
- [ ] Collapsed space display (claimed territory)
- [ ] Player color system
- [ ] Stack height indicators (visual cue for cap height vs total height)
- [ ] Hover effects and highlights

### Task 3.3: User Interaction
- [ ] Ring placement controls
- [ ] Move selection (drag or click)
- [ ] Valid move highlighting
- [ ] Move confirmation dialog
- [ ] Undo/redo buttons
- [ ] Graduated line reward choice UI (Option 1 vs Option 2)
- [ ] Disconnected region processing order UI
- [ ] Forced elimination ring/cap selection
- [ ] Multiple capture direction choice

### Task 3.4: Game State Display
- [ ] Current player indicator
- [ ] Ring count displays (in hand, on board, eliminated)
- [ ] Territory statistics panel (collapsed spaces per player)
- [ ] Move history list
- [ ] Timer/clock display
- [ ] Victory progress indicators (ring elimination %, territory control %)

---

## 📊 PHASE 2.5: Monitoring & DevOps (NEW)

**Status:** Not Started  
**Priority:** P2 - MEDIUM  
**Target Completion:** 1-2 weeks (PARALLEL to Phase 3)

**Rationale:** Production readiness requires observability and deployment automation.

### Task 2.5a: Monitoring Infrastructure
- [ ] Set up Prometheus metrics collection
- [ ] Configure Grafana dashboards
- [ ] Implement structured logging (Winston)
- [ ] Add error tracking (Sentry integration)
- [ ] Create health check endpoints

### Task 2.5b: DevOps & Deployment
- [ ] Document deployment procedures
- [ ] Create rollback procedures
- [ ] Set up staging environment
- [ ] Configure production environment
- [ ] Implement blue-green deployment

### Task 2.5c: Alerting & Monitoring
- [ ] Set up alerting for critical errors
- [ ] Configure performance monitoring
- [ ] Add uptime monitoring
- [ ] Create runbooks for common issues

---

## 🤖 PHASE 4: Advanced Features

**Status:** Not Started  
**Priority:** P2 - MEDIUM  
**Target Completion:** 4-6 weeks after Phase 3

**Note:** Basic AI moved to Phase 1.5. This phase focuses on advanced features.

### Task 4.1: Advanced AI (Optional)
- [ ] MCTS AI (levels 8-10)
  - [ ] Monte Carlo Tree Search implementation
  - [ ] Advanced position evaluation
  - [ ] Multi-player dynamics modeling
- [ ] Neural network AI (levels 9-10)
  - [ ] Consider Python microservice for ML
  - [ ] Model training infrastructure
  - [ ] GPU acceleration setup
- [ ] Advanced opening book
- [ ] Endgame database

### Task 4.2: WebSocket Completion
- [ ] Move broadcasting between players
- [ ] Game state synchronization
- [ ] Spectator events
- [ ] Reconnection logic
- [ ] Game lobby system
- [ ] Player matching

### Task 4.3: Database Integration
- [ ] Game persistence (save/load)
- [ ] User data storage
- [ ] Move history recording
- [ ] Statistics tracking
- [ ] Replay storage
- [ ] Leaderboards

### Task 4.4: Additional Features
- [ ] Spectator mode (read-only view)
- [ ] Replay viewer with step controls
- [ ] Game analysis tools
- [ ] Rating system (ELO or similar)
- [ ] Tournament support
- [ ] Game variants/modes

---

## 📊 Progress Tracking

### Phase 0 Progress: 0/3 tasks (0%) - NEW CRITICAL PHASE
- [ ] 0.1 Testing Framework Setup
- [ ] 0.2 CI/CD Pipeline
- [ ] 0.3 Initial Test Coverage

### Phase 1 Progress: 10/11 tasks (75% COMPLETE - Critical gaps remain)
- [x] 1.1 Fix BoardState (100%) ✅
- [x] 1.2 Marker System (90%) ✅
- [x] 1.3 Movement Validation (75%) ✅
- [x] 1.4 Phase Transitions (85%) ✅
- [⚠️] 1.5 Capture System (40%) ⚠️ Chain captures NOT enforced
- [⚠️] 1.6 Line Formation (70%) ⚠️ Player choices default
- [x] 1.7 Territory Disconnection (70%) ✅
- [x] 1.8 Forced Elimination (80%) ✅
- [x] 1.9 Player State (90%) ✅
- [x] 1.10 Hexagonal Board Validation (85%) ✅
- [ ] 1.11 Player Choice System (0%) ❌ NEW CRITICAL TASK

### Phase 1.5 Progress: 0/4 tasks (0%) - NEW AI PHASE
- [ ] 1.5a AI Infrastructure
- [ ] 1.5b Basic AI Implementation
- [ ] 1.5c Advanced AI (Optional)
- [ ] 1.5d AI Testing

### Phase 2 Progress: 0/4 task groups (0%)
### Phase 2.5 Progress: 0/3 tasks (0%) - NEW MONITORING PHASE
### Phase 3 Progress: 0/4 task groups (0%)
### Phase 4 Progress: 0/4 task groups (0%)

### Overall Progress: 6/30 major tasks (20%)

---

## 🎯 Current Sprint

**Sprint Goal:** Complete Phase 1 Core Logic + Establish Testing Foundation  
**Sprint Duration:** 3-5 weeks  
**Current Task:** Task 1.6 - Line Formation

**This Week (Completed):**
- [x] Task 1.1 - Fix BoardState ✅
- [x] Task 1.2 - Implement Marker System ✅
- [x] Task 1.3 - Fix Movement Validation ✅
- [x] Task 1.4 - Fix Game Phase Transitions ✅
- [x] Task 1.5 - Complete Capture System (core implementation) ✅
- [x] Task 1.6 - Implement Line Formation (core implementation) ✅

**Next Week (Priority):**
- [ ] Task 0.1 - Set up Testing Framework (PARALLEL)
- [ ] Task 1.7 - Territory Disconnection (complex logic)
- [ ] Task 1.8 - Forced Elimination (when blocked)

---

## 📝 Implementation Notes

### Implementation Order Rationale

**Phase 0 (Parallel to Phase 1):**
1. **Testing infrastructure first** - Enables TDD and prevents regressions
2. **CI/CD early** - Automates quality checks

**Phase 1 (Core Logic):**
1. **BoardState first** - All other tasks depend on correct data structure ✅
2. **Marker system early** - Fundamental to movement and territory mechanics ✅
3. **Movement validation** - Required before captures work correctly ✅
4. **Phase transitions** - Needed for proper game flow ✅
5. **Capture system** - Builds on movement validation ✅
6. **Line formation** - Requires markers and phases
7. **Territory disconnection** - Most complex, requires all previous
8. **Forced elimination** - Uses infrastructure from line formation
9. **Player state** - Final synchronization of all mechanics

**Phase 1.5 (AI Engine):**
10. **AI after core logic** - Needs complete rule engine to function
11. **Simple AI first** - Random and heuristic sufficient for MVP
12. **Advanced AI optional** - Minimax/MCTS can wait for post-MVP

**Phase 2-3 (Testing & UI):**
13. **Comprehensive testing** - Before UI to ensure stable backend
14. **Frontend after backend** - UI built on stable game engine

**Phase 2.5 (DevOps):**
15. **Monitoring in parallel** - Set up during frontend development

**Phase 4 (Advanced Features):**
16. **Advanced AI last** - Neural networks and ML after MVP proven

### Testing Strategy

- Write tests alongside implementation (TDD approach when feasible)
- Run tests frequently during development
- Achieve 80%+ code coverage before Phase 3
- Use scenario tests from rules document to validate correctness
- Test edge cases from FAQ thoroughly

### Key Documentation References

**Game Rules:**
- **Complete Rules:** `ringrift_complete_rules.md` - Authoritative source
- **Section 4:** Turn Sequence - Phase flow and turn structure
- **Section 8:** Movement Rules - Distance, landing, marker interaction
- **Section 9-10:** Capture Mechanics - Overtaking vs elimination, chains
- **Section 11:** Line Formation - Graduated rewards system
- **Section 12:** Territory Disconnection - Von Neumann adjacency, representation
- **FAQ Q1-Q24:** Edge cases and clarifications

**Architecture:**
- **Architecture Plan:** `ringrift_architecture_plan.md` - Original design
- **Architecture Assessment:** `ARCHITECTURE_ASSESSMENT.md` - Current state analysis ✨ NEW
- **Technical Analysis:** `TECHNICAL_ARCHITECTURE_ANALYSIS.md` - Detailed technical review
- **Known Issues:** `KNOWN_ISSUES.md` - Bug tracking

### Risk Mitigation

- Complex territory logic may take longer than estimated
- Marker system integration uncovered some edge cases (addressed in 1.2-1.3)
- Phase transitions require careful testing to avoid state bugs
- Plan buffer time for unexpected issues
- Keep documentation updated as implementation progresses

---

## 🔧 Development Guidelines

### Code Quality Standards
- All code must be TypeScript with strict type checking
- Follow existing code style and conventions
- Add JSDoc comments for all public methods
- Include rule references in comments where applicable
- Keep functions focused and single-purpose

### Testing Requirements
- Unit tests for all new methods
- Integration tests for feature workflows
- Scenario tests for rule examples
- Edge case coverage for FAQ questions
- Minimum 80% code coverage target

### Documentation Requirements
- Update this TODO.md as tasks complete
- Add code comments with rule section references
- Document complex algorithms and logic
- Keep KNOWN_ISSUES.md updated with bugs
- Update README.md with new features

---

**Document Version:** 2.0  
**Last Updated:** November 13, 2025  
**Maintained By:** Development Team
