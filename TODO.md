# RingRift Development TODO

**Last Updated:** November 16, 2025  
**Current Phase:** Phase 1 - Core Game Logic Implementation  
**Overall Progress:** 9/30 major tasks completed (~30%)

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

**Progress:** 15/19 subtasks completed (~70% functional) - Single and chained captures enforced engine-side; a shared capture-segment validator now lives in `src/shared/engine/core.ts` and is used by `RuleEngine.validateCaptureSegment`; tests & UI/AI wiring still pending

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
- [x] Add AI decision logic for each choice type using local TypeScript heuristics (AIInteractionHandler + DelegatingInteractionHandler wired through PlayerInteractionManager)
- [ ] Integrate Python AIServiceClient-based decision-making for choices (service-backed PlayerChoices), building on the existing move loop:
  - [ ] Start with a single concrete choice type (e.g. `line_reward_option`) so we can exercise the full AI → service → GameEngine path end-to-end.
  - [ ] Ensure AI choice responses share the same validation semantics as human/WebSocket choices by funnelling everything through `PlayerInteractionManager` + `DelegatingInteractionHandler`.
  - [ ] Add targeted tests that exercise at least one service-backed choice end-to-end (AI profile → AIServiceClient → GameEngine), including failure modes and fallbacks to local heuristics.
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
- [x] Add AI choice logic (for early levels) using simple TypeScript heuristics until AIService integration is in place (implemented via AIInteractionHandler + DelegatingInteractionHandler with unit tests).
- [ ] Extend AI-service-backed PlayerChoices beyond `line_reward_option`:
  - [ ] Finalize the `/ai/choice/ring_elimination` endpoint to use `RingEliminationChoiceRequest/Response` and a real heuristic (e.g. smallest `capHeight`, tie-breaking on `totalHeight`) with full `GameState` available.
  - [ ] Add `getRingEliminationChoice` to `AIEngine` that delegates to `AIServiceClient.getRingEliminationChoice` and returns the selected option.
  - [ ] Update `AIInteractionHandler.selectRingEliminationOption` to call `globalAIEngine.getRingEliminationChoice` first, with robust fallback to the existing local heuristic on error or missing AI config.
  - [ ] Add focused unit tests for `AIEngine` and `AIInteractionHandler` covering the service-backed ring elimination path and its fallback behaviour.
  - [ ] Plan similar service-backed integration for `region_order` once the ring elimination path is stable.

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
- [x] Install and configure Jest/Vitest
- [x] Set up test coverage reporting (target 80%+)
- [x] Create test utilities and fixtures
- [x] Configure TypeScript for tests
- [x] Add test scripts to package.json

### Task 0.2: CI/CD Pipeline
- [x] Create GitHub Actions workflow
- [x] Add linting step (ESLint)
- [x] Add type checking step (tsc)
- [x] Add unit test step
- [x] Add coverage reporting
- [x] Set up pre-commit hooks

### Task 0.3: Initial Test Coverage
- [x] Write tests for BoardManager utilities
- [x] Write tests for existing GameEngine methods
- [x] Write tests for RuleEngine validation
- [ ] Write tests for shared type utilities
- [x] Document testing patterns

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

Add AI Testing / Integration subtasks:
- [x] Integrate AI choice responses with PlayerInteractionManager:
  - [x] Implement an AI-side handler that satisfies `PlayerInteractionHandler` (see `AIInteractionHandler`).
  - [x] Ensure AI responds to PlayerChoice prompts with valid `selectedOption` values for all choice variants.
  - [x] Mirror WebSocketInteractionHandler validation semantics by funnelling all engine choices through `PlayerInteractionManager` + `DelegatingInteractionHandler` so humans and AI share the same interface.
- [x] Introduce shared AI configuration via `AIProfile` and centralise AI setup:
  - [x] Add `AIProfile` and supporting types (`AIControlMode`, `AITacticType`) to `src/shared/types/game.ts`.
  - [x] Extend `Player` with an optional `aiProfile` field to surface AI config in GameState and to clients.
  - [x] Extend `AIEngine` with `createAIFromProfile` so all server-side AI configuration flows through a single façade.
  - [x] Update `WebSocketServer.getOrCreateGameEngine` to construct `AIProfile` instances for AI opponents and configure `globalAIEngine` via `createAIFromProfile`, keeping GameEngine/PlayerInteractionManager the single source of truth for player types.
- [ ] Plan and implement AIServiceClient-based move and choice integration:
  - [ ] Treat `globalAIEngine` as a façade that can delegate to either local TypeScript heuristics or the Python `AIServiceClient` on a per-player/per-aiType basis.
  - [ ] Add configuration on players/AI profiles to select AI type (local heuristic vs service-backed) and difficulty, and plumb this into `globalAIEngine`.
  - [ ] Implement service-backed move selection that calls the Python AI service via `AIServiceClient.getAIMove` and applies the result through `GameEngine` (reusing the existing `maybePerformAITurn` broadcast path).
  - [ ] Design and, if needed, implement AI-service endpoints for answering `PlayerChoice` prompts directly, or define a fallback strategy where local heuristics (`AIInteractionHandler`) are used when the service cannot answer.
  - [ ] Add tests that cover service success/failure cases and ensure the system degrades gracefully to local heuristics when the AI service is unavailable.

**Status:** IN PROGRESS (local AI heuristics integrated; service-backed AI pending)  
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
- [x] Integrate ChoiceDialog with GameContext pendingChoice/choiceDeadline for backend games.
- [x] Add visual indication of active choice timeout based on `choiceDeadline`.
- [x] Add optimistic UI handling for choice submissions (disable buttons after selection until response ack or error).

Under **Task 3.4: Game State Display**, add:
- [ ] Display current pending choice (type and prompt) in a status panel for debugging and UX clarity.

**Status:** Not Started  
**Priority:** P1 - HIGH (increased from P2)  
**Target Completion:** 3-4 weeks after Phase 1.5

### Task 3.1: Board Rendering
- [x] Square grid component (8x8)
- [x] Square grid component (19x19)
- [x] Hexagonal grid component (11 spaces per side, 331 total)
  - [x] Implement cube coordinate system (x, y, z)
  - [x] Hexagonal cell rendering
  - [x] 6-direction adjacency visualization
  - [x] Proper hexagonal grid layout algorithm
- [ ] Cell/space components (unified for square and hex)
- [x] Coordinate overlay (support both square and cube coordinates)
- [x] Responsive sizing (handle different board shapes)
- [ ] Visual polish and animations
- [x] Board type selector UI

### Task 3.2: Game Piece Visualization
- [ ] Ring stack component (show individual rings)
- [ ] Marker display component (player-colored markers)
- [ ] Collapsed space display (claimed territory)
- [ ] Player color system
- [ ] Stack height indicators (visual cue for cap height vs total height)
- [ ] Hover effects and highlights

- [ ] **3.5.2 Piece & territory visualization**
  - [ ] Extend `BoardView` to render **ring stacks** visually rather than as raw numbers:
    - [ ] Introduce a per-cell stack widget that renders **thin oval “ring” shapes** stacked vertically, as if viewing a column of coins or donut-shaped rings from a slightly elevated angle:
      - [ ] Draw the top ring as a wider ellipse with a visible top surface; render lower rings as thinner side views partially hidden beneath the top ring.
      - [ ] Color each ring by owner using the fixed player palette.
    - [ ] Represent the **cap** as “all top rings of the same color” in the stack:
      - [ ] Do **not** make cap rings thicker; instead, outline or highlight all cap rings (e.g. subtle border, inner glow, or ring-offset effect) to distinguish them from rings lower in the stack.
    - [ ] Keep `H# C#` notation to indicate total height and cap height, either as a small overlay badge or as text immediately below the stack widget.
  - [ ] Render **markers** from `board.markers` as distinct symbols:
    - [ ] Outlined circle overlay in the cell using the owner’s color.
    - [ ] Ensure markers remain visible on both empty and stacked cells (e.g. corner badge).
  - [ ] Render **collapsed spaces** from `board.collapsedSpaces`:
    - [ ] Tint the cell background by owner color, darkened, with partial opacity.
    - [ ] Add a subtle pattern or border to distinguish collapsed territory from regular occupied cells.

### Task 3.3: User Interaction
- [ ] Ring placement controls
  - [ ] In the local sandbox mode on `GamePage` (when no `gameId` is present), allow clicking an empty cell to **place a ring on the board** for a chosen/current player, updating the `BoardState.stacks` map and reusing the same ring stack widget used in backend games.
  - [ ] Once the sandbox is wired to the real GameEngine, align sandbox ring placement with the rules (respecting per-player ring limits and phase flow), and clearly label any simpler "free placement" mode as experimental.
- [x] Move selection (drag or click)
- [x] Valid move highlighting
- [ ] Move confirmation dialog
- [ ] Undo/redo buttons
- [x] Graduated line reward choice UI (Option 1 vs Option 2)
- [x] Disconnected region processing order UI
- [x] Forced elimination ring/cap selection
- [x] Multiple capture direction choice

### Task 3.4: Game State Display
- [x] Current player indicator
  - [x] Highlight the active player consistently in both backend GamePage view and local sandbox using `GameState.currentPlayer` and `GameState.currentPhase`.
  - [x] Use a unified color/typography system so the current player stands out clearly while remaining readable in dark mode.
- [x] Ring count displays (in hand, on board, eliminated)
  - [x] Derive counts from `GameState.players[*].ringsInHand`, `GameState.board.stacks`, and `GameState.players[*].eliminatedRings` rather than maintaining separate local counters.
  - [x] Ensure displays stay correct for 2–4 players and mixed human/AI configurations.
- [x] Territory statistics panel (collapsed spaces per player)
  - [x] Compute territory statistics from `GameState.board.collapsedSpaces` and any BoardManager helpers, keeping TS as the source of truth rather than re-implementing rules client-side.
  - [ ] Show both absolute counts and simple percentages of total board spaces for each player.
- [ ] Move history list
  - [ ] Render a scrollable history based on `GameState.moveHistory`, with a simple, human-readable notation for placements, movements, captures, line collapses, and territory events.
  - [ ] Prepare the history panel to later include PlayerChoice events (line options, region order, eliminations, capture directions) as a separate, filterable layer.
- [x] Timer/clock display
  - [x] Use `GameState.timeControl` and `players[*].timeRemaining` to render per-player clocks in minutes:seconds, without enforcing time rules in the client.
  - [ ] Integrate clock highlighting with the current player indicator so the active player’s clock is visually dominant and paused clocks appear clearly inactive.
- [ ] Victory progress indicators (ring elimination %, territory control %)
  - [ ] Drive victory progress from `GameState.totalRingsInPlay`, `GameState.totalRingsEliminated`, `GameState.victoryThreshold`, and `GameState.territoryVictoryThreshold`.
  - [ ] Present progress for each player as simple bar/percentage indicators that remain readable with the chosen dark-theme palette.

### Task 3.5: Concrete GUI/HUD Work Items (Board, HUD, History)
These items break down requested GUI improvements into implementable
steps. Initial focus should be on the **GamePage backend view** (real
GameEngine + WebSocket) with patterns reusable in the local sandbox.

- [ ] **3.5.1 Board contrast & accessibility**
  - [x] Improve default board and cell colors for square boards in `BoardView`:
    - [x] Add explicit background colors for empty cells (e.g. alternating dark/light squares on square boards).
    - [x] Increase border contrast and hover/selection states using Tailwind utilities (e.g. `border-slate-600`, `bg-slate-800`, `ring-offset-2`).
    - [x] Ensure text/icons in cells maintain accessible contrast against backgrounds.
    - [x] Brighten the overall board palette so the board is not nearly black against a dark page background, while preserving clear separation between the board surface and surrounding HUD panels.
    - [x] For 8x8 and 19x19 square boards, make the square backgrounds **significantly lighter** (roughly “50% closer to white” compared to the current palette), using either lighter Tailwind slate shades or added transparency so the board surface reads clearly as the primary play area.
  - [x] Adjust square cell sizes to improve readability and visual room for stacks:
    - [x] For 8x8 boards, increase the side length of each square to approximately **2× the previous size**.
    - [x] For 19x19 boards, increase the side length of each square by roughly **30%** (within the constraints of Tailwind’s size scale) so stacks and markers remain legible without overwhelming the viewport.
    - [x] Verify that these size changes still allow the full board to fit comfortably on common laptop screens without horizontal scrolling in the default layout.
  - [x] Apply consistent dark-theme backgrounds for key containers (LoginPage, GamePage panels, **including the local sandbox setup dialog in GamePage**) using Tailwind `bg-slate-*`/`text-slate-*` classes:
    - [x] Fix the local sandbox initialization dialog so that **select inputs and their default/placeholder text have high contrast** against their backgrounds (e.g. light text on dark background, or lighter input backgrounds), and the dialog container is not the same shade as its contents.
    - [x] Ensure side-panel text on GamePage (Selection, Status, Players/AI Profiles) uses high-contrast text colors (e.g. `text-slate-100`) rather than mid-gray that is too close to the panel background.
  - [ ] Add a simple visual legend component explaining colors used for players, markers, and collapsed spaces.
  - [x] Add **faint movement grid overlays** for readability and rules intuition:
    - [x] On square boards, render a faint horizontal/vertical/diagonal (triangular) grid of thin lines connecting the centers of each square, indicating lines of possible travel.
    - [x] On hex boards, render a faint equilateral triangular grid of thin lines connecting the centers of each hex/circle, indicating lines of possible travel along all 6 directions.
    - [x] At each grid intersection, render a small, translucent dot/node so movement lines and junctions are visually discoverable without overpowering the board.
    - [x] Implement a reusable `computeBoardMovementGrid(board: BoardState)` helper that returns normalized centers + adjacency edges for all supported board types, so future overlays (valid-move highlighting, AI visualizations, sandbox harness, and history playback) can re-use the same board-local geometry independent of the DOM.
    - [ ] Extend documentation in `CURRENT_STATE_ASSESSMENT.md` and `RINGRIFT_IMPROVEMENT_PLAN.md` to mention the movement grid overlay and normalized center geometry as part of the frontend rules/UX foundation.

- [ ] **3.5.2 Piece & territory visualization**
  - [ ] Extend `BoardView` to render **ring stacks** visually rather than as raw numbers:
    - [ ] Introduce a small per-cell stack widget (vertical chips,visually in a vertical stack, with thin layers for each chip and a small margin visible for chips below the top chip with each chip in the stack colored by ring owner, and, with notation H# C# indicating total height and cap height displayed on the top chip.)
    - [ ] Map `Player`/ring numbers → a fixed Tailwind color palette (e.g. P1 = emerald, P2 = sky, P3 = amber, P4 = fuchsia) using explicit class names to keep Tailwind happy.
    - [ ] Show stack height and cap height via badge/overlay (e.g. text label).
    - [ ] Ensure that for both square and hex boards, **stacks up to height 10** can be rendered (including the H#/C# label) without extending significantly beyond the visual bounds of a cell; this may require tuning ring oval heights, spacing, and font sizes separately for 8x8 vs 19x19/hex boards.
  - [ ] Render **markers** from `board.markers` as distinct symbols:
    - [ ] Outlined circle overlay in the cell using the owner’s color.
    - [ ] Ensure markers remain visible on both empty and stacked cells (e.g. corner badge).
  - [ ] Render **collapsed spaces** from `board.collapsedSpaces`:
    - [ ] Tint the cell background by owner color, darkened, with partial opacity.
    - [ ] Add a subtle pattern or border to distinguish collapsed territory from regular occupied cells.
  - [ ] For the **hex board**, refine the layout and movement cues:
    - [ ] Adjust the per-row offsets and spacing so that the circular cells are arranged in a **proper hexagonal close packing**, with each circle touching (tangential to) all of its neighbors where appropriate, and without visual overlap.
    - [ ] Ensure the hex board’s faint triangular movement grid (see 3.5.1) aligns with circle centers and directions of legal movement.

- [ ] **3.5.3 Turn HUD, player status, and clocks**
  - [ ] Add a **per-player HUD panel** in `GamePage` (backend mode) showing:
    - [ ] Player name / AI profile.
    - [ ] Whether the player is human or AI.
    - [ ] Ring counts (in hand, on board, eliminated) derived from `GameState`.
    - [ ] Territory spaces controlled from `GameState.board.eliminatedRings` / BoardManager helpers.
  - [ ] Implement a **current player indicator** integrated with GameEngine’s `currentPlayer` and `currentPhase`:
    - [ ] Highlight the active player’s HUD row with a strong accent color.
    - [ ] Show current phase (“Ring placement”, “Movement”, “Capture”, “Lines”, “Territory”).
  - [ ] Add **per-player clocks** wired to `GameState.timeControl` / `players[*].timeRemaining`:
    - [ ] Display time remaining for each player in minutes:seconds.
    - [ ] Highlight the active player’s clock;  dim and pause others.
    - [ ] Initially treat clocks as **display-only** (no enforcement changes) and defer rule-level timing edge cases to a later server pass.

- [ ] **3.5.4 Game history & notation**
  - [ ] Add a **move/choice history panel** to `GamePage` (backend mode):
    - [ ] Use `GameState.moveHistory` plus local event log entries (e.g. line formation, territory disconnection, PlayerChoices).
    - [ ] Derive a simple, human-readable notation for moves and choices, both for square and hexagonal boards, grounded in `ringrift_complete_rules.md` (e.g. `P1: place (c3)`, `P2: move (d4→d8)`, `P1: capture (e5×e7→e9)`, `choice: line_reward_option=option_1`).  Visually label the game boards with corresponding coordinates.
    - [ ] Support basic filtering (all events vs placements vs moves vs captures vs line formation, territory disconnection, ring elimination, etc. choices, by player, etc).
  - [ ] In the sandbox (`/sandbox`), maintain a simpler local history log based on user interactions until full GameEngine integration exists.

- [ ] **3.5.5 Turn-stage controls & richer interactions (longer-term)**
  - [ ] For the backend game view, refine interaction flows to make **turn stages** explicit:
    - [ ] Ring placement: affordances for “place”, “skip”, or “forced placement” according to rules.
    - [ ] Movement: highlight all legal moves from a selected stack (using backend-provided `validMoves`) and valid overtaking capture segments out to at least the first segment when multi segment overtaking captures are allowed.
    - [ ] Capture: visually distinguish capture moves from non-captures; clearly show capture targets and possible landings.  If depth of multi segment possible landing squares is >1, distinguish first segment from subsequent segments.
    - [ ] Line processing and territory processing: surface system-driven events in the history panel and HUD rather than as “silent” transitions.
  - [ ] For the local sandbox (`/sandbox`), plan a **lightweight local-engine harness**:
    - [ ] Instantiate a browser-only GameEngine-like state (or call the existing TS GameEngine directly) for a single local game.
    - [ ] Allow users to step through full turns (placement, movement, capture, line/territory processing) without a backend connection.
    - [ ] Keep this behind a clear “experimental sandbox” flag until parity with backend behaviour is validated by tests.

---

## 🧪 Sandbox Harness – Backend-wrapped now, client-local later

**Status:** Stage 1 planned/partially implemented · **Priority:** P1 – HIGH (supports UX, testing, and AI debugging)

To keep the rules source-of-truth unified while still moving toward a rich local sandbox, the sandbox harness is split into two stages:

- [ ] **Stage 1 – Thin client wrapper around the existing backend GameEngine (current focus)**
  - [x] Allow the `/sandbox` route (`GamePage` without a `:gameId`) to act as a frontend-only entry point that can be swapped to a backend-backed game without changing board/choice UI.
  - [x] In `GamePage`, wire the "Start Local Game" action to first attempt a `CreateGameRequest` via the same `/games` API used by `LobbyPage`, then navigate to `/game/:gameId` on success so the sandbox uses the real server GameEngine, WebSocket, PlayerChoice layer, and AI turns.
  - [ ] Surface any backend-creation failures in the sandbox setup UI with a clear message, then fall back to the existing local-only board for quick visual experiments (already partially implemented via `backendSandboxError`, but not yet rendered in the UI).
  - [ ] Once this flow is stable, document it in `CURRENT_STATE_ASSESSMENT.md` and `RINGRIFT_IMPROVEMENT_PLAN.md` as the canonical Stage 1 sandbox harness.
- [ ] **Stage 2 – True client-local GameEngine harness (future)**
  - [x] Implement an initial browser-safe `LocalSandboxState` controller under `src/client/sandbox/localSandboxController.ts`, wiring the `ring_placement` and a minimal `movement` phase into the `/sandbox` branch of `GamePage` using the shared `BoardView` + movement grid and shared helpers from `src/shared/engine/core.ts` (experimental, local-only for now).
  - [ ] Refactor the shared TS GameEngine (or a thin wrapper) so it can be imported into the client bundle in a browser-safe way (no Node-only deps, no direct DB/Redis usage).
  - [ ] Replace the Stage 1 backend-backed sandbox with a client-local harness that can step through full turns (placement → movement → capture → lines → territory) using the same BoardView, movement grid (`computeBoardMovementGrid`), and PlayerChoice UI, with the server remaining the authority for ranked/networked games.
  - [ ] Keep this client-local harness behind an "experimental sandbox" flag until its behaviour is validated against backend engines and Jest/Rust tests.

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
   - [x] Use `PlayerChoiceResponseFor<TChoice>` more broadly in `GameEngine` where choices are consumed.
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

5. **Leverage Rust engine as reference implementation**
   - [x] Treat `RingRift Rust/ringrift` as an authoritative reference for tricky rule areas (chain capture, line/territory processing, PlayerChoice model) rather than as a runtime dependency.
   - [x] Port at least one core Rust `test_chain_capture` scenario into Jest (see `tests/unit/GameEngine.chainCapture.test.ts`).
   - [ ] Systematically identify additional high-value Rust tests (especially chain capture, stalemate, and territory edge cases) to mirror in Jest.
   - [ ] Add short notes in relevant Jest test files pointing back to the original Rust tests they were ported from, to preserve design rationale and ease future cross-checking.

6. **Backend AI configuration & diagnostics (immediate next step)**
   - [x] Extend the backend `CreateGame` pipeline to carry `aiOpponents.mode` and `aiOpponents.aiType` through `CreateGameSchema` → `CreateGameRequest` → Prisma `gameState.aiOpponents`.
   - [x] Update `WebSocketServer.getOrCreateGameEngine` to map `aiOpponents` into per-player `AIProfile` instances and pass them into `globalAIEngine.createAIFromProfile`.
   - [x] Extend `LobbyPage` to expose AI configuration controls (mode, type, count, difficulty) and submit a fully typed `CreateGameRequest`.
   - [x] Add a diagnostic panel in `GamePage` that lists each player and, for AI players, surfaces their `aiProfile` (difficulty, mode, aiType) for debugging and tuning.
   - [x] Add targeted BoardManager territory-disconnection tests that mirror the core Section 12 examples from `ringrift_complete_rules.md` for regression safety.
   - [x] Add at least one GameEngine-level territory-disconnection scenario test that covers a full move → disconnection → elimination flow, using the new BoardManager tests as fixtures.

7. **Playable game lifecycle quick wins (aligned with PLAYABLE_GAME_IMPLEMENTATION_PLAN.md)**
   - **Victory signalling and persistence:**
     - [x] Extend `GameEngine`/`RuleEngine` usage so that after each move (including engine-driven chains and automatic consequences) the server has an explicit victory check (`checkGameEnd`), and propagate the resulting `gameResult` from `GameEngine.makeMove` into the WebSocket layer.
     - [x] In `WebSocketServer.handlePlayerMove` and `maybePerformAITurn`, when the engine reports a game end, persist `status: COMPLETED`, `winnerId`, and `endedAt` via Prisma and emit a `game_over` event with a structured payload (`{ gameId, gameState, gameResult }`).
   - **Minimal victory UI:**
     - [x] In `GameContext`, listen for `game_over` and store a small `victoryState` (`GameResult`) while updating the local `gameState` snapshot.
     - [x] In `GamePage` (backend mode), render a minimal `VictoryModal` overlay that shows the winner/reason and offers an action to return to the lobby.
   - **Auto-start / lifecycle smoothing:**
     - [ ] In `WebSocketServer.getOrCreateGameEngine`, add a simple auto-start rule for backend games: if the hydrated game has enough players and all AI profiles are configured, call `gameEngine.startGame()` and set the DB status to `ACTIVE` if still `waiting`.
   - **HUD alignment:**
     - [ ] Ensure that any new HUD elements (current player, phase, ring counts, territory stats) added to `GamePage`/`GameHUD` derive their data directly from `GameState` fields (`players`, `board.stacks`, `board.collapsedSpaces`, `board.eliminatedRings`, `totalRingsInPlay`, `totalRingsEliminated`, `victoryThreshold`, `territoryVictoryThreshold`) so that the HUD and rules always stay in sync.

---

======= SEARCH
## 🧪 Capture Sequence Enumeration Parity Harness (NEW)

**File:** `tests/unit/captureSequenceEnumeration.test.ts`  
**Status:** IMPLEMENTED (expensive but deterministic diagnostic)

This Jest suite exhaustively compares **sandbox vs backend** capture-sequence enumeration over a large set of seeded-random positions, using the *real* marker and collapsed-space rules from both engines.

**Core behaviour:**
- Uses shared capture segment helpers and real marker/territory logic:
  - Sandbox path delegates to `applyMarkerEffectsAlongPathOnBoard` + sandbox helpers.
  - Backend path delegates to `applyMarkerEffectsAlongPathOnBoard` with `BoardManager` APIs.
- For each maximal capture chain, keeps an evolving `BoardState` and stores the `finalBoard` for statistics.
- Normalizes sequences into string keys for strict parity comparison per position.

**Parameters / generation:**
- `MAX_SEQUENCES = 100_000` per position (hard cap to avoid explosion).
- ~50 seeded-random positions per board type, with constraints:
  - **square8:** 2–6 targets.
  - **square19:** 2–4 targets.
  - **hexagonal:** 2–4 targets.
- 0–2 initial collapsed spaces per board, placed away from a guaranteed primary capture ray so at least one legal capture always exists.

**Diagnostics printed per board type:**
- Case with **max number of distinct capture sequences** (with longest-chain example).
- Case with **max capture chain length** (example chain).
- Case with **max markers** on the final board after a valid capture sequence (board + example sequence).
- Case with **max collapsed spaces** on the final board after a valid capture sequence (board + example sequence).

These all use the same summary format (`logCaseSummary`, `logOutcomeSummary`) so console output is easy to scan and compare across board types.

**Runtime caveats:**
- This suite is intentionally heavy and currently takes several minutes to run (square19 + hexagonal are the slowest).
- For CI, consider either:
  - Reducing cases and/or `MAX_SEQUENCES`, or
  - Running this suite in a separate, slower “diagnostic” job rather than the main per-PR test job.

## 🧪 Trace Parity & GameTrace Infrastructure (NEW)

**Status (November 18, 2025):** Core trace/parity infrastructure is implemented and exercised by Jest suites. Remaining work focuses on **semantic parity** (move semantics and phase alignment) plus hardening of replay helpers.

**Implemented:**
- [x] Promote `GameHistoryEntry` and `GameTrace` in `src/shared/types/game.ts` as the canonical event log for both engines (initial state + per-move snapshots, including S-invariant, optional state hashes, and board summaries).
- [x] Add trace utilities in `tests/utils/traces.ts`:
  - [x] `runSandboxAITrace(boardType, numPlayers, seed, maxSteps)` – run seeded AI-vs-AI games in `ClientSandboxEngine` and emit `GameTrace`.
  - [x] `replayTraceOnBackend(trace)` – rebuild a backend `GameEngine` from `trace.initialState` and replay canonical moves via `findMatchingBackendMove`.
  - [x] `replayTraceOnSandbox(trace)` – rebuild a fresh sandbox engine and re-apply the same canonical moves via `applyCanonicalMove`.
- [x] Introduce a canonical sandbox mutation path:
  - [x] `applyCanonicalMoveInternal(move, opts)` in `ClientSandboxEngine` applies canonical moves (place_ring, skip_placement, move_stack/move_ring, overtaking_capture) using shared sandbox helpers and returns a `stateChanged` boolean.
  - [x] `applyCanonicalMove` wraps the internal helper, computing S/hashes and calling `appendHistoryEntry` only when the state hash changes, with an option to bypass no-dead-placement checks during canonical replays.
- [x] Add `tests/unit/ClientSandboxEngine.traceStructure.test.ts` to enforce structural invariants on sandbox-emitted traces (contiguous `moveNumber` from 1, `actor === action.player`, non-negative S-invariants, explicit `skip_placement` entries when no legal placements exist).
- [x] Extend README.md and tests/README.md with a "Trace parity & GameTrace" section documenting these helpers and how to run the parity/debug suites.
- [x] Wire trace diagnostics controlled by env vars:
  - [x] `RINGRIFT_TRACE_DEBUG` – logs sandbox trace opening sequences and backend mismatch snapshots (including valid move lists) to `logs/ai/trace-parity.log`.
  - [x] `RINGRIFT_AI_DEBUG` – mirrors AI/trace diagnostics to the console and enables extra sandbox AI debug logging.

**Next steps (tracked in Phase 1E / sandbox modularization and trace parity work):**
- [ ] Unify sandbox AI turns (`maybeRunAITurn`) so that all AI actions are expressed as canonical `Move` objects and routed through `applyCanonicalMoveInternal` + history recording, eliminating bespoke mutation paths.
- [ ] Solidify backend replay helpers by introducing a reusable `replayMovesOnBackend(initialConfig, moves: Move[]): GameTrace` that wraps `GameEngine` + `findMatchingBackendMove` and returns a backend `GameTrace` for analysis.
- [ ] Keep the backend↔sandbox parity suites (`Backend_vs_Sandbox.traceParity.test.ts`, `Sandbox_vs_Backend.seed5.traceDebug.test.ts`, `Backend_vs_Sandbox.aiParallelDebug.test.ts`) green as semantic gaps are closed, using the new logging to diagnose remaining discrepancies (e.g. overtaking_capture vs non-enumerated backend moves, capture vs placement phase mismatches for specific seeds).

---

### Key Documentation References

**Game Rules:**
- **Complete Rules:** `ringrift_complete_rules.md` - Authoritative source
- **Section 4:** Turn Sequence - Phase flow and turn structure
- **Section 8:** Movement Rules - Distance, landing, marker interaction
- **Section 9-10:** Capture Mechanics - Overtaking vs elimination, chains
- **Section 11:** Line Formation - Graduated rewards system
- **Section 12:** Territory Disconnection - Von Neumann adjacency, representation
- **FAQ Q1-Q24:** Edge cases and clarifications

**Architecture & Status (current, code-verified):**
- **Architecture Assessment:** `ARCHITECTURE_ASSESSMENT.md` - Current architecture and refactoring axes
- **Codebase Evaluation (historical):** `deprecated/CODEBASE_EVALUATION.md` - Earlier code-level evaluation, preserved for context only
- **Current State:** `CURRENT_STATE_ASSESSMENT.md` - Factual, code-verified implementation status
- **Known Issues:** `KNOWN_ISSUES.md` - Bug/issue tracking (P0/P1 gaps)

**Roadmap & Playable Experience:**
- **Strategic Roadmap:** `STRATEGIC_ROADMAP.md` - Phased strategic plan (testing → core logic → UI → AI → multiplayer)
- **Playable Game Implementation:** `PLAYABLE_GAME_IMPLEMENTATION_PLAN.md` - End-to-end playable lifecycle plan (lobby → game → victory) and new-task context summary

**Historical Design Docs (superseded, in `deprecated/`):**
- `deprecated/ringrift_architecture_plan.md` – Original architecture plan
- `deprecated/TECHNICAL_ARCHITECTURE_ANALYSIS.md` – Detailed technical analysis
- `deprecated/RINGRIFT_IMPROVEMENT_PLAN.md` – Earlier improvement plan (replaced by the documents above)

## 📊 Progress Summary

### Phase 0 Progress: 2/3 tasks (67%)
- [x] 0.1 Testing Framework Setup (basic Jest + ts-jest wired, though config needs modernization)
- [x] 0.2 CI/CD Pipeline
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



## 📌 1–2 Week Implementation Checklist (Concrete Next Steps)

1. **Chain capture & PlayerChoice tests (engine-level)**
   - [x] Add additional chain capture tests derived from Rust scenarios (multi-step, 180° reversal, diagonal and orthogonal rays).
   - [x] Add integration tests for CaptureDirectionChoice using both a direct handler and WebSocketIntegrationHandler.
   - [x] Add a WebSocket-backed integration test for LineRewardChoice and RingEliminationChoice.
   - [ ] Mirror at least one more complex chain capture scenario from `RingRift Rust/ringrift/tests/chain_capture_tests.rs` into Jest (including a full move → chain → processAutomaticConsequences path).

2. **Short, prioritized TODO/roadmap slice for the next 1–2 weeks**
   - [ ] Keep this checklist and the "Near-Term Implementation Focus" section in TODO.md in sync with CURRENT_STATE_ASSESSMENT and STRATEGIC_ROADMAP.
   - [ ] Revisit this list after each significant engine/AI/UX change and prune/extend items accordingly.

3. **GamePage / ChoiceDialog UX polish**
   - [ ] Add a choice timeout indicator in GamePage using `choiceDeadline` (countdown in seconds for `pendingChoice`).
   - [ ] Add a small status panel in GamePage to show the current `pendingChoice` (type, player, remaining time).
   - [ ] Update ChoiceDialog to disable buttons once a selection is made until the choice resolves or errors.
   - [ ] Add a minimal event log (recent moves/choices) in GamePage to aid debugging.

4. **Targeted testing after UX changes**
   - [ ] Re-run the choice/chain/territory/AI test suites after UX changes to ensure no regressions:
     - `tests/unit/GameEngine.chainCapture.test.ts`
     - `tests/unit/GameEngine.chainCaptureChoiceIntegration.test.ts`
     - `tests/unit/GameEngine.captureDirectionChoiceWebSocketIntegration.test.ts`
     - `tests/unit/GameEngine.lineRewardChoiceWebSocketIntegration.test.ts`
     - `tests/unit/GameEngine.territoryDisconnection.test.ts`
     - `tests/unit/AIEngine.serviceClient.test.ts`

### Sandbox & BoardView Visual/State-Update Fixes (NEW)
- [ ] Fix `/sandbox` chain capture re-rendering by coupling `ClientSandboxEngine` capture resolution steps to a React state bump (for example, `setSandboxTurn(t => t + 1)`), and clearing `selected` / `validTargets` / capture-related state after each human click that triggers a board mutation.
- [ ] Ensure any human click that changes the sandbox board (including capture-direction choices and continuation landings) always triggers a fresh `gameState` read for `BoardView`, so ring stacks and markers never display stale heights after the final capture segment.
- [ ] Align the movement grid overlay with square boards (8×8 and 19×19) by adjusting `computeBoardMovementGrid` and the SVG viewBox/transform in `BoardView` so grid node dots and movement lines coincide with cell centers for all board types.
- [ ] Finalize valid-target highlight and marker styling across square and hex boards so that:
  - [ ] Valid destinations use a thin, bright-green ring inset inside the cell border plus a subtle green background tint, with no highlight extending outside the cell.
  - [ ] Hex and square markers share the three-layer pattern (outer colored ring, smaller dark middle disc, tiny inner core) with correct relative sizes and transparent outer backgrounds so the outer ring color remains vivid.
- [ ] Add a brief manual test script (and/or a lightweight Jest/React test) that exercises a multi-segment capture in `/sandbox` on square8 and hex, confirming that stack heights and markers are visually correct after each segment and especially after the final capture segment.

## PHASE 3S: Sandbox Stage 2 – Fully Local Playable Game (NEW)

**Current Status (November 15, 2025):** Core client-local sandbox functionality is implemented. `ClientSandboxEngine` plus `sandboxMovement.ts`, `sandboxCaptures.ts`, `sandboxElimination.ts`, `sandboxLines.ts`, `sandboxLinesEngine.ts`, `sandboxTerritory.ts`, `sandboxTerritoryEngine.ts`, and `sandboxVictory.ts` now drive a full rules-complete game loop in `/sandbox` (movement, overtaking and mandatory chain captures, line processing, territory disconnection on square + hex boards, and ring/territory victories), with dedicated Jest suites under `tests/unit/ClientSandboxEngine.*.test.ts`. The remaining tasks below focus on richer HUD/UX integration, stronger sandbox AI heuristics, and additional lifecycle polish rather than basic rules coverage.

**Goal:** Make `http://localhost:3000/sandbox` capable of running a complete 2–4 player RingRift game entirely in the browser, with rules enforcement and AI players that make random valid choices for all stages and possibilities of their turns, until a valid victory/conclusion is reached.

**Scope:** Client-only harness that reuses the existing rules engine and PlayerChoice architecture as much as possible, without duplicating rules logic or diverging from the backend GameEngine semantics.

### 3S.1: Design Client-Local Engine Harness
- [ ] Define a browser-safe engine façade for the sandbox (e.g. `ClientSandboxEngine`) under `src/client/sandbox/` that:
  - [ ] Maintains a `GameState` instance in memory.
  - [ ] Drives turn/phase progression using the same phase model as `GameEngine` (`ring_placement → movement → capture → line_processing → territory_processing → next player`).
  - [ ] Uses the existing shared types from `src/shared/types/game.ts` so that sandbox GameState is structurally identical to the backend GameState.
- [ ] Decide, and document in `ARCHITECTURE_ASSESSMENT.md`, whether the sandbox harness:
  - [ ] Imports a refactored, browser-safe subset of `GameEngine`/`RuleEngine`/`BoardManager` (preferred long-term), or
  - [ ] Wraps calls to shared engine helpers in `src/shared/engine/core.ts` plus a thin sandbox-specific orchestration layer.

### 3S.2: Player Interaction in Sandbox (Human + AI)
- [ ] Introduce a sandbox-specific `SandboxInteractionHandler` that satisfies the same `PlayerInteractionHandler` contract used by `PlayerInteractionManager` on the server, but implemented entirely on the client:
  - [ ] For **human** players: surface `PlayerChoice` objects into React state (similar to `GameContext.pendingChoice`), and expose a callback that resolves the internal `requestChoice` Promise when the user selects an option via the existing `ChoiceDialog` UI.
  - [ ] For **AI** players: choose randomly among the provided `options` for any `PlayerChoice` (line order, line reward, ring elimination, region order, capture direction) to guarantee forward progress while keeping the logic simple.
- [ ] Ensure the sandbox interaction layer reuses the same choice types (`PlayerChoice`, `PlayerChoiceResponseFor<TChoice>`) and semantics as the backend, so behaviour remains consistent and future test scenarios can be shared.

### 3S.3: Sandbox Turn Loop & Phases
- [ ] Implement a deterministic “sandbox turn loop” that:
  - [ ] Alternates between players 1–N based on `GameState.currentPlayer` and `nextPlayer` logic matching `GameEngine`.
  - [ ] For each turn, walks through the same phase sequence as the backend engine:
    - [ ] Ring placement phase (honouring `ringsInHand` and placement rules).
    - [ ] Movement phase (valid moves only).
    - [ ] Capture phase, including mandatory chain continuation.
    - [ ] Line processing (graduated rewards and line/region ordering via PlayerChoices).
    - [ ] Territory processing (disconnections, chain reactions, self-elimination prerequisite).
  - [ ] Uses the sandbox interaction handler for all PlayerChoices during the turn.
- [ ] Define clear, sandbox-local victory detection mirroring `RuleEngine.checkGameEnd`/`GameEngine.endGame` semantics and surface the resulting `GameResult` to the UI.

### 3S.4: `/sandbox` UI Integration
- [ ] Extend the existing `/sandbox` branch of `GamePage` to:
  - [ ] Replace the current “local-only board with manual clicks” model with a `ClientSandboxEngine`-driven game that uses the same `BoardView`, movement-grid overlay, `ChoiceDialog`, and (where appropriate) `GameHUD` components as backend games.
  - [ ] Support configuration of 2–4 players with arbitrary human/AI mixes (using the existing local sandbox setup form as the entry point).
  - [ ] For human players, wire cell clicks into sandbox move submission (source/target selection) using a pattern similar to backend click-to-move, but targeting the sandbox engine instead of the WebSocket server.
  - [ ] For AI players, have the sandbox loop automatically call into the random-choice AI interaction handler when it is their turn.
- [ ] Ensure that a complete game can be started and played in the sandbox from ring placement through to a valid victory or conclusion (including chain captures, lines, territory, and forced elimination), with the UI clearly indicating when the game is over.

### 3S.5: Testing & Documentation for Sandbox
- [ ] Add at least one sandbox-focused test (unit or integration) that:
  - [ ] Constructs a small sandbox game state, drives a few turns through the `ClientSandboxEngine` (or equivalent), and asserts that invariant properties hold (e.g. no illegal moves are accepted, phases progress as expected, victory is recognized when configured conditions are met).
- [ ] Update `TODO.md`, `PLAYABLE_GAME_IMPLEMENTATION_PLAN.md`, and `CURRENT_STATE_ASSESSMENT.md` to:
  - [ ] Mark Sandbox Stage 1 (backend-wrapped sandbox entry) as complete.
  - [ ] Clearly describe Sandbox Stage 2 as a fully local GameEngine harness that mirrors backend behaviour but lives entirely in the browser for experimentation and teaching.
  - [ ] Document that the sandbox AI is intentionally simple (random among valid options) to prioritise correctness and completeness of rules over strength.

### 3S.6: Sandbox Stage 2 – Next Steps to Fully Playable Local Sandbox (Updated Nov 16, 2025)
- [ ] Wire sandbox AI turn driving into `GamePage`:
  - [ ] After each human action in sandbox mode, call `sandboxEngine.maybeRunAITurn()` (potentially in a loop) while `currentPlayer` is an AI and `gameStatus === 'active'`, so AI turns proceed automatically without manual clicks.
  - [ ] Extend `ClientSandboxEngine.maybeRunAITurn` to handle both `ring_placement` and `movement` phases. For ring placement, choose among legal placement positions computed via `hasAnyLegalMoveOrCaptureFrom`; for movement, consider both simple moves from `enumerateSimpleMovementLandings` and overtaking captures from `enumerateCaptureSegmentsFrom` (via `sandboxCaptures` / `sandboxCaptureSearch`).
- [ ] Surface sandbox-valid targets in the UI:
  - [ ] Implement a sandbox-specific valid-move enumeration that mirrors backend `validMoves` but uses `ClientSandboxEngine` and the `sandboxMovement` / `sandboxCaptures` helpers instead of WebSocket data.
  - [ ] When a stack is selected in sandbox mode, highlight all legal landing positions (including capture landings) in `BoardView` by setting `validTargets`, and ignore clicks on illegal cells.
- [ ] Unify HUD between backend and sandbox games:
  - [ ] Reuse `GameHUD` in local sandbox mode, driven by the sandbox `GameState` from `ClientSandboxEngine.getGameState()`, so ring counts, territory stats, current player, and phase are presented consistently across both surfaces.
  - [ ] Ensure sandbox HUD reflects `GameState.totalRingsEliminated`, `victoryThreshold`, and `territoryVictoryThreshold` so local games expose progress toward victory just like backend games.
- [ ] Harden sandbox AI choice behaviour:
  - [ ] Keep the current “random among options” policy for all `PlayerChoice` types, but ensure the interaction handler always returns a valid option and never hangs if the choice is cleared mid-turn.
  - [ ] Add a minimal timeout/guard around sandbox choices so miswired UI cannot stall the turn loop.
- [ ] Add end-to-end sandbox flow tests:
  - [ ] Add a focused Jest suite (e.g. `ClientSandboxEngine.playableFlow.test.ts`) that simulates a short 2-player sandbox game with one human and one AI, driving the engine via `handleHumanCellClick` and `maybeRunAITurn` and asserting that:
    - [ ] All moves and captures applied by the AI are legal.
    - [ ] Phases progress correctly for both players.
    - [ ] The game reaches a valid victory state (`GameResult`) without entering an infinite loop.
  - [ ] Add a second scenario that exercises at least one line-formation + territory-disconnection sequence in the sandbox, to confirm local rules stay aligned with backend behaviour.

## 🧱 Architecture & Refactoring – Rules + Engines + AI (NEW)

**Status (November 16, 2025):** INITIAL REFACTOR COMPLETE · **Priority:** P1 – HIGH (supports rules consistency, testability, and future sandbox/backend unification)

### A. Completed Refactor (shared core helpers)
- [x] Consolidate path/distance helpers between RuleEngine and shared core:
  - [x] Updated `src/server/game/RuleEngine.ts` to use `getPathPositions` from `src/shared/engine/core.ts` instead of a private implementation.
  - [x] Updated `RuleEngine.isPathClear` and `RuleEngine.isPathClearForHypothetical` to operate on `getPathPositions(from, to).slice(1, -1)` so that both backend and sandbox engines share identical path geometry for movement, capture, and no-dead-placement checks.
  - [x] Re-ran the Jest suite to confirm that movement/capture tests (`RuleEngine.movementCapture.test.ts`, `RuleEngine.placementMultiRing.test.ts`, and the various `GameEngine`/`ClientSandboxEngine` parity tests) still pass, ensuring no behavioural regressions.

### B. Planned Refactors (rules/engines/AI alignment)
- [ ] Introduce a shared "movement/capture board view" helper in `src/shared/engine/core.ts` to centralize "can this stack move or capture?" logic:
  - [ ] Define a minimal `MovementBoardView` interface (parallel to `CaptureSegmentBoardView`) with methods like `isValidPosition`, `isCollapsedSpace`, `getStackAt`, and `getMarkerOwner`, implemented via:
    - [ ] A `BoardManager`-backed adapter on the backend (for `RuleEngine`/`GameEngine`).
    - [ ] A `BoardState`-backed adapter on the client (for `ClientSandboxEngine`).
  - [ ] Implement a shared `hasAnyLegalMoveOrCaptureFromOnBoard(boardType, from, player, boardView)` helper in `src/shared/engine/core.ts` that encapsulates:
    - [ ] Non-capture movement enumeration (respecting stack height, collapsed spaces, and markers).
    - [ ] Capture ray-walk and landing validation (delegating to `validateCaptureSegmentOnBoard` for segment legality).
  - [ ] Refactor both:
    - [ ] `RuleEngine.hasAnyLegalMoveOrCaptureFrom` to delegate to the shared helper via a `BoardManager` adapter.
    - [ ] `ClientSandboxEngine.hasAnyLegalMoveOrCaptureFrom` to delegate via a sandbox `BoardState` adapter.
  - [ ] Add focused unit tests around the new shared helper, using simple square and hex boards, to assert that backend and sandbox observers compute the same set of "at least one legal action" outcomes for representative positions.
  - [ ] Once parity is confirmed, update `TODO.md`, `CURRENT_STATE_ASSESSMENT.md`, and `ARCHITECTURE_ASSESSMENT.md` to note that movement/capture reachability is now defined in a single shared location.

### C. Future Opportunities (beyond first shared helper)
- [ ] Evaluate consolidating capture move generation (ray-walk logic) between `RuleEngine.getValidCaptures`, `GameEngine.getCaptureOptionsFromPosition`, and sandbox capture helpers, using `validateCaptureSegmentOnBoard` as the single legality oracle and shared direction sets from `getMovementDirectionsForBoardType`.
- [ ] Explore moving additional small geometric helpers (e.g. adjacency checks, simple distance helpers) fully into `src/shared/engine/core.ts` and making `BoardManager` a thin adapter over those helpers for both backend and potential future client-local engines.

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

## 🧱 PHASE 1E: Engine Modularization – Backend & Sandbox (NEW)

**Status:** Not Started  
**Priority:** P1 - HIGH  
**Rationale:** GameEngine/RuleEngine and ClientSandboxEngine/sandbox helpers have grown large and complex. We need to modularize engine-related code while preserving movement/capture parity, move-type unification, and existing AI/parity harness behaviour.

### 1E.1: Baseline & Constraints (Movement Parity Safe)
- [ ] Reconfirm current movement/capture/move-type invariants before refactor:
  - [ ] `move_stack` is canonical non-capture movement; `move_ring` remains a legacy alias accepted by backend/sandbox and tests.
  - [ ] Backend movement semantics for non-capture moves are constrained to legal rays via `RuleEngine.isStraightLineMovement`.
  - [ ] Path rules: distance ≥ stack height; stacks/collapsed spaces block; markers do not block; legal landing: empty/own marker/stack; illegal: opponent marker.
- [ ] Re-run and document as the guardrail suite for this refactor:
  - [ ] `tests/unit/movementReachabilityParity.test.ts`
  - [ ] `tests/unit/reachabilityParity.RuleEngine_vs_Sandbox.test.ts`
  - [ ] `tests/unit/ClientSandboxEngine.moveParity.test.ts`
  - [ ] `tests/unit/GameEngine.landingOnOwnMarker.test.ts`
  - [ ] `tests/unit/captureSequenceEnumeration.test.ts`
  - [ ] `tests/unit/Backend_vs_Sandbox.aiParallelDebug.test.ts`
- [ ] Note in `CURRENT_STATE_ASSESSMENT.md` that engine modularization must keep these suites green throughout.

### 1E.2: Backend Engine Modularization (RuleEngine, GameEngine, AI/Interaction Glue)

**Files in scope:**
- `src/server/game/RuleEngine.ts`
- `src/server/game/GameEngine.ts`
- `src/server/game/BoardManager.ts` (as needed for helpers/adapters)
- `src/server/game/ai/AIEngine.ts`, `AIInteractionHandler.ts`, `DelegatingInteractionHandler.ts`
- `src/server/game/PlayerInteractionManager.ts`
- `src/server/game/WebSocketInteractionHandler.ts`

**Goals:** Keep public entrypoints and types stable while splitting concerns into smaller, cohesive modules (target: each file well under ~700 lines).

- [ ] Design target backend engine module layout (document in `ARCHITECTURE_ASSESSMENT.md`):
  - [ ] Turn/phase orchestration (e.g. `GameEngine` core turn loop, `advanceGame`, game end handling).
  - [ ] Rules/validation and enumerators (e.g. movement, captures, lines, territory, placements), ideally under `src/server/game/rules/` alongside `placementHelpers.ts`.
  - [ ] Board/geometry helpers (adapters around `BoardManager` + `src/shared/engine/core.ts`).
  - [ ] AI/choice integration glue (AIEngine + interaction handlers) separate from pure rules.
- [ ] Extract RuleEngine responsibilities into dedicated modules without changing semantics:
  - [ ] Movement & simple non-capture enumeration/validation.
  - [ ] Capture validation and capture-sequence helpers.
  - [ ] Line formation helpers / enumerators.
  - [ ] Territory disconnection helpers.
  - [ ] Shared reachability helpers (wrapping `hasAnyLegalMoveOrCaptureFromOnBoard`).
- [ ] Extract GameEngine responsibilities into dedicated modules:
  - [ ] Turn/phase state machine and per-turn state (`hasPlacedThisTurn`, `mustMoveFromStackKey`, `chainCaptureState`).
  - [ ] Move application and game end handling (`applyMove`, `makeMove`, `getValidMoves`, `checkGameEnd`).
  - [ ] Forced elimination and ring/territory victory handling.
  - [ ] PlayerChoice plumbing (line reward choice, capture direction choice, region/ring elimination ordering) kept in a clearly separated interaction layer.
- [ ] Introduce narrow internal interfaces where useful (e.g. `RulesEngineFacade`, `TurnEngine`, `BoardViewAdapter`) while keeping external API stable:
  - [ ] Preserve existing public methods used by tests, WebSocket handlers, and CLI tools.
  - [ ] Avoid breaking call sites in `test-complete-backend.ts` and WebSocket server.
- [ ] After each extraction step, re-run:
  - [ ] Core backend engine tests (all `tests/unit/GameEngine.*.test.ts`, `tests/unit/RuleEngine.*.test.ts`).
  - [ ] Player interaction/choice tests (`tests/unit/PlayerInteractionManager.test.ts`, `tests/unit/WebSocketInteractionHandler.test.ts`, WebSocket/AI integration tests).
  - [ ] Movement/capture parity suites listed in 1E.1.

### 1E.3: Sandbox Engine Modularization (ClientSandboxEngine + Helpers)

**Files in scope:**
- `src/client/sandbox/ClientSandboxEngine.ts`
- `src/client/sandbox/sandboxMovement.ts`
- `src/client/sandbox/sandboxCaptures.ts`
- `src/client/sandbox/sandboxCaptureSearch.ts`
- `src/client/sandbox/sandboxLines.ts`, `sandboxLinesEngine.ts`
- `src/client/sandbox/sandboxTerritory.ts`, `sandboxTerritoryEngine.ts`
- `src/client/sandbox/sandboxElimination.ts`
- `src/client/sandbox/sandboxVictory.ts`
- `src/client/sandbox/localSandboxController.ts`

**Goals:** Keep sandbox rules in parity with backend while making the client-local engine easier to understand, test, and extend.

- [ ] Design target sandbox engine module layout (document in `CURRENT_STATE_ASSESSMENT.md` and reference from `PLAYABLE_GAME_IMPLEMENTATION_PLAN.md`):
  - [ ] Sandbox turn/phase orchestration layer (client-local analogue of GameEngine turn loop).
  - [ ] Movement and capture helpers (building on `sandboxMovement`/`sandboxCaptures` and shared `core.ts`).
  - [ ] Line and territory helpers (`sandboxLines*`, `sandboxTerritory*`).
  - [ ] Victory/termination helpers (`sandboxVictory`, forced elimination).
  - [ ] Sandbox AI loop (`maybeRunAITurn`) and PlayerChoice glue, kept distinct from pure rules.
- [ ] Clarify and, where helpful, re-split responsibilities so that:
  - [ ] `ClientSandboxEngine` focuses on state orchestration, turn loop, and choice/AI integration.
  - [ ] Pure rules/geometry logic lives in small helper modules that mirror backend structure and shared `core.ts` helpers.
- [ ] Ensure any new sandbox modules:
  - [ ] Use shared types from `src/shared/types/game.ts` and helpers from `src/shared/engine/core.ts` rather than duplicating logic.
  - [ ] Keep movement and capture semantics identical to the backend.
- [ ] After refactors, re-run sandbox-focused suites:
  - [ ] `tests/unit/ClientSandboxEngine.*.test.ts` (all).
  - [ ] Sandbox AI tests (`tests/unit/ClientSandboxEngine.aiSimulation.test.ts`, `tests/unit/ClientSandboxEngine.aiMovementCaptures.test.ts`).

### 1E.4: Parity & Behavioural Regression Guardrail
- [ ] Add a short comment block in new backend and sandbox engine modules noting that movement/capture semantics are verified by the parity harnesses listed in 1E.1.
- [ ] Ensure that the backend↔sandbox harness tests still pass after modularization:
  - [ ] `tests/unit/Sandbox_vs_Backend.aiHeuristicCoverage.test.ts` (allowing for known AI-depth issues documented in `KNOWN_ISSUES.md`).
  - [ ] `tests/unit/Backend_vs_Sandbox.aiParallelDebug.test.ts`.
- [ ] Keep the move-type unification behaviour stable:
  - [ ] Backend `validateMove` continues to treat `move_ring` as a legacy alias for `move_stack`.
  - [ ] Sandbox `applyCanonicalMove` continues to accept both while emitting `move_stack` for AI moves.

### 1E.5: Documentation & Follow-up Work Hooks
- [ ] Add a concise "Engine Module Layout" section to:
  - [ ] `ARCHITECTURE_ASSESSMENT.md` (backend engine layout).
  - [ ] `CURRENT_STATE_ASSESSMENT.md` (sandbox/local engine layout and parity notes).
- [ ] Update `PLAYABLE_GAME_IMPLEMENTATION_PLAN.md` to reference the new modular engine structure as the place to plug in:
  - [ ] Future AI termination fixes.
  - [ ] Game phase/state debugging.
  - [ ] Additional parity harnesses and Rust-test ports.
- [ ] Add a brief note in `TODO.md` under "Near-Term Implementation Focus" pointing back to Phase 1E once the initial layout is agreed.

---

**Document Version:** 2.1  
**Last Updated:** November 16, 2025  
**Maintained By:** Development Team
****
