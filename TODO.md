# RingRift Development TODO

**Last Updated:** November 14, 2025  
**Current Phase:** Phase 1 - Core Game Logic Implementation  
**Overall Progress:** 7/30 major tasks completed (~23%)

This document is the **single source of truth** for RingRift development tracking.

---

## 🚀 PHASE 1: Core Game Logic Implementation

**Status:** IN PROGRESS (75% complete, critical gaps remain)  
**Priority:** P0 - CRITICAL  
**Target Completion:** 2-3 weeks remaining for completion

**VERIFIED ACTUAL STATUS (November 14, 2025):**
- ✅ Basic mechanics working (marker system, movement, basic captures, lines, territory)
- ✅ Chain captures enforced engine-side (mandatory continuation + capture-direction choice implemented in GameEngine)
- ✅ Player choice system wired at the engine level (via PlayerInteractionManager + PlayerChoice types) and now partially wired via WebSockets + UI
- ⚠️ Strategic gameplay still reduced in practice because full choice/chain flows are not yet exercised by AI or covered by comprehensive tests

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
```text
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
**Priority:** P0 - CRITICAL | **Estimated:** 3-4 days | **Status:** IN PROGRESS (engine-level logic implemented; tests & UI integration pending)

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
- [x] **CRITICAL: Implement mandatory chain capture continuation in GameEngine (engine-level)**
- [x] **CRITICAL: Handle player choice when multiple capture directions available (engine-level via PlayerInteractionManager)**
- [ ] Write unit tests for cap height comparison (deferred to Phase 2)
- [ ] Write unit tests for chain capture sequences (deferred to Phase 2)
- [ ] Write tests for 180° reversal pattern (FAQ Q15.3.1) (deferred to Phase 2)
- [ ] Write tests for cyclic pattern (FAQ Q15.3.2) (deferred to Phase 2)

**Acceptance Criteria:**
- [x] Overtaking adds captured ring to bottom of stack
- [x] **Chain captures are mandatory once started** (implemented in GameEngine via `chainCaptureState`)
- [x] Cap height correctly determines capture eligibility
- [x] Can land on any valid space beyond target
- [ ] **Multiple capture sequences work correctly** ❌ NOT FULLY COVERED BY TESTS
- [ ] All tests pass including FAQ examples (deferred to Phase 2)

**Rule Reference:** Sections 9-10, Section 15.3, FAQ Q14

**Key Rules:**
- Overtaking: Top ring of target stack added to bottom of capturing stack ✅
- Elimination: Rings permanently removed (line formations, territory disconnections)
- Chain captures: Mandatory once started, player chooses direction at each step ✅ (engine-level, tests pending)

**Progress:** 15/19 subtasks completed (~70% functional) - Single and chained captures enforced engine-side; tests & UI/AI wiring still pending

**Status:** INCOMPLETE - Chain capture logic implemented but needs more tests and UI/AI integration

---

### Task 1.6: Implement Line Formation ⏳
**Priority:** P1 - HIGH | **Estimated:** 3-4 days | **Status:** INCOMPLETE

**Dependencies:** Task 1.2 (Marker system) ✅, Task 1.4 (Phase transitions) ✅

**Files:**
- `src/server/game/GameEngine.ts` ⚠️ (defaulting to Option 2 when no interaction manager)
- `src/server/game/BoardManager.ts` ✅
- `src/server/game/RuleEngine.ts` ⚠️

**Subtasks:**
- [x] Fix `findAllLines()` to detect 4+ consecutive markers for 8x8 (CRITICAL: markers not stacks)
- [x] Fix `findAllLines()` to detect 5+ consecutive markers for 19x19/hex
- [x] Implement Option 1: Collapse all + eliminate ring (for exact or longer lines)
- [x] Implement Option 2: Collapse only required + no elimination (for longer lines only)
- [x] Implement `eliminatePlayerRingOrCap()` method
- [x] Check for new lines after each collapse
- [x] Update player's `eliminatedRings` counter
- [x] Update player's `territorySpaces` counter
- [ ] **CRITICAL: Add player choice mechanism for longer lines** (currently defaults to Option 2 when no interaction manager)
- [ ] **CRITICAL: Handle multiple line processing in player-chosen order** (currently uses first found)
- [ ] Write unit tests for line detection (deferred to Phase 2)
- [ ] Write unit tests for graduated rewards (deferred to Phase 2)
- [ ] Write integration tests for multiple lines (deferred to Phase 2)

**Acceptance Criteria:**
- [x] Lines of 4+ (8x8) or 5+ (19x19/hex) detected correctly
- [x] Exactly minimum length: must use Option 1
- [ ] **Longer than minimum: player CHOOSES Option 1 or 2** ❌ Defaults to Option 2 when no interaction manager
- [x] Ring elimination tracked correctly
- [x] Territory spaces tracked correctly
- [ ] **Multiple lines processed in PLAYER-CHOSEN order** ❌ Uses first found
- [ ] All tests pass (deferred to Phase 2)

**Rule Reference:** Section 11, Section 11.2

**Progress:** 10/13 subtasks completed (70% functional) - Detection works, choice integration and tests remain

**Status:** INCOMPLETE - Player choice mechanism exists at engine level but needs full coverage and tests

---

### Task 1.7: Implement Territory Disconnection ✅
**Priority:** P1 - HIGH | **Estimated:** 4-5 days | **Status:** COMPLETED

(unchanged from previous version; see original document for details)

---

### Task 1.8: Add Forced Elimination ✅
**Priority:** P1 - HIGH | **Estimated:** 1 day | **Status:** COMPLETED

(unchanged from previous version; see original document for details)

---

### Task 1.9: Fix Player State Updates ✅
**Priority:** P1 - HIGH | **Estimated:** 1 day | **Status:** COMPLETED

(unchanged from previous version; see original document for details)

---

### Task 1.10: Hexagonal Board Support Validation ✅
**Priority:** P1 - HIGH | **Estimated:** 2-3 days | **Status:** COMPLETED

(unchanged from previous version; see original document for details)

---

### Task 1.11: Player Choice System ⏳ (UPDATED)
**Priority:** P0 - CRITICAL | **Estimated:** 1-2 weeks | **Status:** IN PROGRESS (~75%)

**Dependencies:** Tasks 1.5, 1.6, 1.7

**Files:**
- `src/shared/types/game.ts` (PlayerChoice, PlayerChoiceResponse, PlayerChoiceResponseFor) ✅
- `src/server/game/PlayerInteractionManager.ts` (type-safe facade) ✅
- `src/server/game/WebSocketInteractionHandler.ts` (WebSocket transport for choices) ✅
- `src/server/websocket/server.ts` (wiring PlayerInteractionManager into GameEngine) ✅
- `src/server/game/GameEngine.ts` (integrate choice system) ✅
- `src/client/contexts/GameContext.tsx` (expose pendingChoice/respondToChoice) ✅
- `src/client/components/ChoiceDialog.tsx` (UI for choices) ✅
- `src/client/pages/GamePage.tsx` (wiring ChoiceDialog into backend game flow) ✅
- `tests/unit/PlayerInteractionManager.test.ts` (manager tests) ✅
- `tests/unit/WebSocketInteractionHandler.test.ts` (new transport tests) ✅

**Subtasks:**
- [x] Design PlayerChoice interface and types
- [x] Create async choice request/response wrapper (`PlayerInteractionManager.requestChoice`)
- [x] Integrate type-safe `PlayerChoiceResponseFor<TChoice>` for engine-facing responses
- [x] Implement timeout handling and server-side validation in WebSocketInteractionHandler
- [x] Integrate with GameEngine for:
  - [x] Line processing order selection (multiple lines)
  - [x] Graduated line reward choice (Option 1 vs Option 2)
  - [x] Ring/cap elimination selection
  - [x] Disconnected region processing order
  - [x] Capture direction selection (chain captures)
- [x] Create UI scaffold for each choice type via `ChoiceDialog`
- [x] Wire `ChoiceDialog` to server-driven choices over WebSockets via GameContext
- [ ] Add AI decision logic for each choice type (via AIServiceClient or local heuristics)
- [x] Write basic tests for PlayerInteractionManager
- [x] Write tests for WebSocketInteractionHandler (happy path, invalid option, wrong player)
- [ ] Add integration tests covering at least one full choice flow through GameEngine

**Acceptance Criteria:**
- [ ] Human players prompted for all strategic choices via UI (using shared PlayerChoice types) **(partially true; UI wired, but end-to-end scenarios and UX polish pending)**
- [ ] AI players make automatic decisions using AIServiceClient or local heuristics ❌
- [x] Choices have configurable timeouts with safe defaults (implemented in WebSocketInteractionHandler)
- [x] Invalid choices rejected with clear errors on the server (WebSocketInteractionHandler validation + tests)
- [x] All game mechanics that require decisions are wired to the choice system at the engine level

**Next Steps for Task 1.11:**
- [ ] Propagate `choiceType` end-to-end to support discriminated-union handling:
  - [ ] Have client emit `choiceType: choice.type` in `player_choice_response`.
  - [ ] Optionally assert on `choiceType` in WebSocketInteractionHandler for extra safety.
  - [ ] Use `PlayerChoiceResponseFor<TChoice>` more broadly within GameEngine for clearer branching.
- [ ] Add at least one integration test that:
  - [ ] Drives a line reward choice from GameEngine → PlayerInteractionManager → WebSocketInteractionHandler → mocked client → back.
- [ ] Add AI choice logic (for early levels) using simple TypeScript heuristics until AIService integration is in place.

---

## 🧪 PHASE 0: Testing Foundation (CRITICAL - UPDATED)

**Status:** PARTIALLY IN PROGRESS  
**Priority:** P0 - CRITICAL

Updates based on recent work:
- Jest is installed and configured; ts-jest warnings indicate future config cleanup needed.
- Unit tests are in place for:
  - PlayerInteractionManager
  - WebSocketInteractionHandler

**New Subtasks:**
- [x] Add focused unit tests for PlayerInteractionManager
- [x] Add focused unit tests for WebSocketInteractionHandler
- [ ] Address ts-jest deprecation warnings (move config to `transform` and `tsconfig.json` `isolatedModules`)

(Existing Phase 0 tasks remain; see original section.)

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

## 🧪 PHASE 2: Testing & Validation (UNCHANGED, WITH NEW DEPENDENCIES)

Add explicit dependency:
- [ ] Depends on Task 1.11 (Player Choice System) to be functionally complete before writing full end-to-end tests.

Add new bullets under Scenario/Integration tests:
- [ ] Scenario: multi-step chain capture with capture-direction choices across multiple responses (WebSocket + UI).
- [ ] Scenario: conflicting/invalid `player_choice_response` (wrong player, stale choiceId) is safely rejected and does not hang.

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

## 🤖 PHASE 1.5: AI Engine Implementation (NEW REFINEMENTS)

Add a new subtask under AI Testing / Integration:
- [ ] Integrate AI choice responses with PlayerInteractionManager:
  - [ ] Implement an AI-side handler that satisfies `PlayerInteractionHandler`.
  - [ ] Ensure AI responds to PlayerChoice prompts with valid `selectedOption` values.
  - [ ] Mirror WebSocketInteractionHandler validation semantics to avoid divergence between human and AI flows.

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

---

## 🎨 PHASE 3: Frontend Implementation (UPDATED WITH CHOICE/WEBSOCKET NOTES)

Under **Task 3.3: User Interaction**, update bullets:
- [x] Choice dialog scaffold for:
  - [x] Line order
  - [x] Line reward options
  - [x] Ring/cap elimination
  - [x] Region order
  - [x] Capture direction
- [ ] Integrate ChoiceDialog with GameContext pendingChoice/choiceDeadline for backend games.
- [ ] Add visual indication of active choice timeout based on `choiceDeadline`.
- [ ] Add optimistic UI handling for choice submissions (disable buttons after selection until response ack or error).

Under **Task 3.4: Game State Display**, add:
- [ ] Display current pending choice (type and prompt) in a status panel for debugging and UX clarity.

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
- [x] Define and implement choice-related events:
  - [x] `player_choice_required` with payload `PlayerChoice`
  - [x] `player_choice_response` with payload `PlayerChoiceResponse`
- [x] Implement a concrete `PlayerInteractionHandler` (e.g. `WebSocketInteractionHandler`) that:
  - [x] Emits `player_choice_required` to the relevant client(s)
  - [x] Tracks pending choices keyed by `choice.id` and `playerNumber`
  - [x] Resolves/rejects Promises based on incoming `player_choice_response` or timeouts
- [x] Add server-side validation that `PlayerChoiceResponse.selectedOption` is one of the allowed options for the original choice
- [x] Ensure GameEngine instances are constructed with a `PlayerInteractionManager` that wraps this handler
- [x] Standardize WebSocket message schema using shared, discriminated-union payload types (inspired by Triphlex `WebSocketMessage` design) and introduce `PlayerChoiceResponseFor<TChoice>` to support discriminated-union style typing
- [ ] Add a client-side WebSocket request/response helper (on top of Socket.IO) patterned after Triphlex's `WebSocketManager.request()` for RPC-style calls with timeouts

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

## 🤖 PHASE 4: Advanced Features – WebSocket Completion (4.2) (UPDATED)

Mark the following WebSocket choice subtasks as completed:

### Task 4.2: WebSocket Completion (Choice-related items)

- [x] Define and implement choice-related events:
  - [x] `player_choice_required` with payload `PlayerChoice`
  - [x] `player_choice_response` with payload `PlayerChoiceResponse`
- [x] Implement a concrete `PlayerInteractionHandler` (`WebSocketInteractionHandler`) that:
  - [x] Emits `player_choice_required` to the relevant client(s)
  - [x] Tracks pending choices keyed by `choice.id` and `playerNumber`
  - [x] Resolves/rejects Promises based on incoming `player_choice_response` or timeouts
- [x] Add server-side validation that `PlayerChoiceResponse.selectedOption` is one of the allowed options for the original choice (with tests)
- [x] Ensure GameEngine instances are constructed with a `PlayerInteractionManager` that wraps this handler
- [x] Introduce `PlayerChoiceResponseFor<TChoice>` to standardize message schema and support discriminated-union style typing
- [ ] Add a client-side WebSocket request/response helper (beyond Socket.IO’s basic emit/on) for more general RPC-style calls

Keep non-choice WebSocket tasks (move broadcasting, reconnection, etc.) as TODO.

---

## 🔜 Near-Term Implementation Focus (UPDATED)

**NEW/REFINED near-term cross-cutting items:**

1. **Tighten choice typing end-to-end**
   - [ ] Use `PlayerChoiceResponseFor<TChoice>` more broadly in `GameEngine` where choices are consumed.
   - [x] Update `GameContext.respondToChoice` to include `choiceType: choice.type` in `player_choice_response` payloads.
   - [x] Optionally, add a non-fatal assertion in `WebSocketInteractionHandler.handleChoiceResponse` to log mismatch between `response.choiceType` and `choice.type`.

2. **Add integration tests around choice + WebSocket + UI**
   - [ ] Write a GameEngine integration test that uses a fake `PlayerInteractionHandler` to simulate at least one line reward scenario and one capture-direction scenario.
   - [ ] Add a test harness (or Cypress/Playwright later) for a minimal browser flow that exercises `GamePage` + `GameContext` + `ChoiceDialog` for a simple backend-driven choice.

3. **Refine move transport schema (future step)**
   - [ ] Plan the migration of `handlePlayerMove` from `{ moveNumber, moveType, position }` to a shared `Move`-based schema.
   - [ ] Add a transitional adapter on the server side that accepts the old schema but immediately converts it into a `Move` for GameEngine consumption.
   - [ ] Only after that, adjust `GameContext.submitMove` to emit the richer `Move` shape directly.

4. **Progressive lint/type cleanup (longer-term)**
   - [ ] Create targeted tasks for high-impact lint cleanup:
     - [ ] Remove `any` from shared type boundaries (especially WebSocket and PlayerChoice flows).
     - [ ] Reduce `no-non-null-assertion` usage in core engine modules by introducing safer helpers.
   - [ ] Defer broad `no-console` cleanup in test scaffolds until after core behaviour is stable.

---

## 📊 Progress Summary

### Phase 0 Progress: 1/3 tasks (33%)
- [x] 0.1 Testing Framework Setup (basic Jest + ts-jest wired, though config needs modernization)
- [ ] 0.2 CI/CD Pipeline
- [ ] 0.3 Initial Test Coverage

### Phase 1 Progress: 10/11 tasks (~80%)
- [x] 1.1 Fix BoardState
- [x] 1.2 Marker System
- [x] 1.3 Movement Validation
- [x] 1.4 Phase Transitions
- [⚠️] 1.5 Capture System (engine done; tests/UI/AI pending)
- [⚠️] 1.6 Line Formation (engine done; choices/tests pending)
- [x] 1.7 Territory Disconnection
- [x] 1.8 Forced Elimination
- [x] 1.9 Player State
- [x] 1.10 Hexagonal Board Validation
- [⚠️] 1.11 Player Choice System (engine+transport+UI wired; AI, integration tests pending)

### Phase 4 (WebSocket Completion – choice-related subset):
- [x] Core choice events, handler, and validation complete
- [ ] Remaining WebSocket features (move broadcasting, reconnection, lobby, etc.)

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
- **Architecture Assessment:** `ARCHITECTURE_ASSESSMENT.md` - Current state analysis
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
- Keep `KNOWN_ISSUES.md` updated with bugs
- Update `README.md` with new features
- Leverage historical Triphlex documentation (e.g. diagrams and worked examples) to strengthen `ringrift_complete_rules.md` and related docs, while keeping RingRift's rules as the single source of truth

---

**Document Version:** 2.1  
**Last Updated:** November 14, 2025  
**Maintained By:** Development Team
