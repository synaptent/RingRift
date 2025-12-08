# RingRift Architecture Refactoring Opportunities Analysis

> **Doc Status (2025-12-08): Active**
>
> Comprehensive analysis of refactoring opportunities across the RingRift codebase, prioritized by impact on maintainability, debuggability, and separation of concerns.
>
> **Related Documents:**
> - `RULES_ENGINE_ARCHITECTURE.md` - Current engine architecture
> - `docs/architecture/STATE_MACHINES.md` - Existing state machine documentation
> - `docs/architecture/DOMAIN_AGGREGATE_DESIGN.md` - Aggregate patterns
> - `AI_ARCHITECTURE.md` - AI service architecture

---

## Executive Summary

This analysis identifies **critical refactoring opportunities** across five major areas of the codebase:

| Area | LOC Analyzed | Key Issues | Estimated Savings |
|------|--------------|------------|-------------------|
| **TS Shared Engine** | ~18,000 | Validator/mutator duplication, orchestration spread across 4 files | ~1,600 LOC |
| **Python AI Service** | ~15,700 | HeuristicAI god class (2,278 LOC), inheritance coupling | ~3,000 LOC |
| **Client React** | ~12,000+ | Mega-components (2.9K LOC), scattered UI state | ~2,500 LOC |
| **Server/Sandbox** | ~7,000+ | Decision phase duplication, AI fallback duplication | ~700 LOC |
| **Test Infrastructure** | ~200K+ | Parity test redundancy, fixture duplication | ~8,500 LOC |

**Total Estimated Code Reduction: ~16,300 LOC**

---

## 1. TypeScript Shared Engine Refactoring

### 1.1 Critical Issues

#### Validator/Mutator Duplication (HIGH IMPACT)
**Problem:** Validators and mutators exist in two places, not synchronized:
- `/validators/*.ts` (5 files, ~400 LOC)
- `/aggregates/*.ts` (7 files) - duplicate implementations

**Files Affected:**
- `validators/PlacementValidator.ts` duplicates `aggregates/PlacementAggregate.ts`
- `validators/MovementValidator.ts` duplicates `aggregates/MovementAggregate.ts`
- Same pattern for Capture, Line, Territory validators

**Recommendation:** Delete `/validators/` directory; re-export from aggregates:
```typescript
// validators/index.ts (NEW)
export { validatePlacementOnBoard } from '../aggregates/PlacementAggregate';
export { validateMovement } from '../aggregates/MovementAggregate';
// ... etc
```

#### Orchestration Logic Spread (HIGHEST IMPACT)
**Problem:** Turn processing logic duplicated across 4 files:
1. `turnOrchestrator.ts` (1,450 LOC) - main entry point
2. `turnLogic.ts` (313 LOC) - phase advancement
3. `phaseStateMachine.ts` (400 LOC) - phase transitions
4. `GameEngine.ts` (280 LOC) - inline transitions

**Example Duplication:**
```typescript
// turnOrchestrator.ts:200-212
if (newCaptures.length > 0) {
  this.state = mutatePhaseChange(this.state, 'chain_capture');
}

// GameEngine.ts:224-272 (SAME LOGIC)
const nextCaptures = enumerateCaptureMoves(...);
if (nextCaptures.length > 0) {
  this.state = mutatePhaseChange(this.state, 'chain_capture');
}
```

**Recommendation:**
1. Keep `turnLogic.ts` as golden source for `advanceTurnAndPhase()`
2. Use `TurnStateMachine` (already defined in `/fsm/TurnStateMachine.ts`) as single source of truth
3. Delete `phaseStateMachine.ts` wrapper
4. Remove inline transitions from `GameEngine.ts`

#### State Machine Clarity (MEDIUM-HIGH IMPACT)
**Problem:** Three coexisting state management approaches:
1. `TurnStateMachine` (explicit FSM with type-safe transitions)
2. `PhaseStateMachine` (procedural wrapper)
3. `GameEngine.checkStateTransitions()` (inline conditionals)

**Recommendation:** Adopt `TurnStateMachine` as the single source of truth:
```typescript
// orchestration/turnProcessing.ts (NEW)
export function processTurn(gameState: GameState, move: Move): ProcessTurnResult {
  const fsmState = deriveStateFromGame(gameState);
  const event = moveToEvent(move);
  const result = transition(fsmState, event, context);
  // Apply FSM-prescribed actions
}
```

### 1.2 Priority Actions

| Priority | Action | Lines Saved | Effort |
|----------|--------|-------------|--------|
| P1 | Delete `/validators/` directory | ~400 | Small |
| P1 | Delete `/mutators/` directory | ~400 | Small |
| P2 | Consolidate orchestration to TurnStateMachine | ~800 | Large |
| P2 | Consolidate helpers into aggregates | ~3,500 | Medium |
| P3 | Simplify public API in `index.ts` | ~200 | Small |

---

## 2. Python AI Service Refactoring

### 2.1 Critical Issues

#### HeuristicAI God Class (HIGH IMPACT)
**File:** `ai-service/app/ai/heuristic_ai.py` (2,278 LOC)
- Contains 46 private methods
- Handles: move selection, 30+ evaluation features, optimization paths, swap evaluation, weight profiles

**Recommendation:** Extract into focused classes:
```
app/ai/heuristic/
├── evaluator.py        # Core feature evaluation
├── evaluators/
│   ├── control.py      # Stack/territory control
│   ├── threat.py       # Opponent threats
│   ├── structure.py    # Line potential/connectivity
│   └── recovery.py     # Recovery features
├── move_selection.py   # select_move orchestration
├── swap_evaluation.py  # Swap bonus + classifier (300 lines)
└── heuristic_ai.py     # Router only (200 lines)
```

#### Inheritance Coupling (HIGH IMPACT)
**Problem:** MinimaxAI and MCTSAI inherit from HeuristicAI just to reuse `evaluate_position()`:
```python
class MinimaxAI(HeuristicAI):  # Inherits 46 unused private methods
    """Only needs evaluate_position from parent..."""
```

**Recommendation:** Composition over inheritance:
```python
class EvaluationProvider:
    """Provides evaluate_position() without tree-search baggage."""
    def evaluate_position(self, game_state: GameState) -> float: ...

class MinimaxAI(BaseAI):
    def __init__(self, player_number, config):
        self.evaluator = EvaluationProvider()

    def evaluate_position(self, game_state) -> float:
        return self.evaluator.evaluate_position(game_state)
```

#### Move Ordering Duplication (MEDIUM IMPACT)
**Problem:** Move type priorities duplicated in 3 places:
- `minimax_ai.py:331-360` - `_score_and_sort_moves()`
- `minimax_ai.py:930-948` - `_score_noisy_moves()`
- `minimax_ai.py:873-906` - `_order_moves_with_killers()`

**Recommendation:** Extract to shared module:
```python
# app/ai/move_ordering.py
class MoveOrderingHeuristic:
    PRIORITY_MAP = {
        "territory_claim": 5,
        "line_formation": 4,
        "recovery_slide": 3,
        "chain_capture": 2,
        "overtaking_capture": 1,
    }

    @classmethod
    def get_priority(cls, move: Move) -> int:
        return cls.PRIORITY_MAP.get(move.type, 0)
```

#### Implicit Evaluation Strategy (MEDIUM IMPACT)
**Problem:** `select_move()` has 150+ lines of branching logic for different evaluation paths:
```python
if should_pick_random: -> random path
elif USE_BATCH_EVAL: -> batch path
elif USE_MAKE_UNMAKE: -> fast path
elif USE_PARALLEL_EVAL: -> parallel path
else: -> legacy path
```

**Recommendation:** Explicit Strategy pattern:
```python
class EvaluationStrategy(Enum):
    LEGACY = "legacy"
    INCREMENTAL = "incremental"
    BATCH = "batch"
    PARALLEL = "parallel"

class MoveEvaluator(ABC):
    @abstractmethod
    def evaluate_moves(self, game_state, moves) -> List[Tuple[Move, float]]: ...

class LegacyEvaluator(MoveEvaluator): ...
class IncrementalEvaluator(MoveEvaluator): ...
class BatchEvaluator(MoveEvaluator): ...
```

### 2.2 Priority Actions

| Priority | Action | Lines Saved | Effort |
|----------|--------|-------------|--------|
| P1 | Extract swap evaluation (300 lines) | 250 | 3 days |
| P1 | Create EvaluationProvider interface | 800 | 4 days |
| P1 | Extract move ordering heuristics | 100 | 2 days |
| P2 | Implement Strategy pattern for evaluation | 500 | 5 days |
| P2 | Centralize env var config | 200 | 2 days |
| P3 | Split HeuristicAI evaluators | 1,000 | 10 days |

---

## 3. Client React Architecture Refactoring

### 3.1 Critical Issues

#### Mega-Component Architecture (HIGHEST IMPACT)
**Problem:** Two host components are massive:
- `SandboxGameHost.tsx`: 2,992 LOC, 25+ useState calls
- `BackendGameHost.tsx`: 1,953 LOC

**Recommendation:** Decompose into feature containers:
```
SandboxGameHost (150 LOC - orchestrator only)
├── SandboxGameBoard (400 LOC)
├── SandboxGameHUD (300 LOC)
├── SandboxChoiceHandler (200 LOC)
├── SandboxReplayIntegration (300 LOC)
└── SandboxDiagnosticsPanel (150 LOC)
```

#### Scattered UI State (HIGH IMPACT)
**Problem:** Game state management across multiple contexts with impossible states possible:
```typescript
// Current GameContext model allows impossible states:
pendingChoice: PlayerChoice | null;     // Phase: decision-active
victoryState: GameResult | null;        // Phase: game-over
// Both can be non-null simultaneously (stale choice during game-over)
```

**Recommendation:** Use compound state machines:
```typescript
type GamePhase =
  | { type: 'connecting'; percentage: number }
  | { type: 'live'; currentPlayer: Player }
  | { type: 'decision'; choice: PlayerChoice; deadline: number | null }
  | { type: 'game-over'; result: GameResult }
  | { type: 'rematch-negotiation'; request: RematchRequestPayload }
  | { type: 'error'; message: string };

// Illegal states become unrepresentable
```

#### Backend/Sandbox Duplication (HIGH IMPACT)
**Problem:** Both `BackendGameHost` and `SandboxGameHost` redundantly implement:
- Board rendering (~100 LOC each)
- HUD rendering (~100 LOC each)
- Choice dialog handling (~50 LOC each)
- Victory modal flow (~80 LOC each)
- Move animation logic (~150 LOC each)

**Recommendation:** Create unified `GameFacade` abstraction:
```typescript
interface GameFacade {
  gameState: GameState;
  validMoves: Move[];
  submitMove(move: PartialMove): void;
  respondToChoice(choice: PlayerChoice, option: unknown): void;
  onGameStateChange(listener: (state: GameState) => void): () => void;
}

class BackendGameFacade implements GameFacade { ... }
class SandboxGameFacade implements GameFacade { ... }

// Single UnifiedGameHost works for both
function UnifiedGameHost({ facade }: { facade: GameFacade }) { ... }
```

#### Prop Drilling (MEDIUM-HIGH IMPACT)
**Problem:** `BoardView.tsx` has 24+ props covering board state, interactions, overlays, animations

**Recommendation:** Extract selection/highlighting as hooks:
```typescript
function useBoardSelection(validMoves?: Move[]) {
  const [selected, setSelected] = useState<Position>();
  const validTargets = useMemo(() =>
    findValidTargets(selected, validMoves), [selected, validMoves]);
  return { selected, setSelected, validTargets };
}
```

### 3.2 Priority Actions

| Priority | Action | Lines Saved | Effort |
|----------|--------|-------------|--------|
| P1 | Extract GameFacade abstraction | 600 | 1 week |
| P1 | Implement compound state machines | 400 | 1 week |
| P2 | Decompose SandboxGameHost | 1,500 | 2 weeks |
| P2 | Extract BoardView props into hooks | 300 | 3 days |
| P3 | Consolidate countdown/timer logic | 200 | 2 days |

---

## 4. Server/Sandbox Architecture Refactoring

### 4.1 Critical Issues

#### Decision Phase Duplication (HIGH IMPACT)
**Problem:** Decision timeout state duplicated:
- Server: `GameSession` lines 167-173 (6 private fields)
- Sandbox: `ClientSandboxEngine` embedded + `SandboxOrchestratorAdapter`

**Recommendation:** Extract shared DecisionPhaseState machine:
```typescript
// src/shared/stateMachines/decisionPhase.ts
type DecisionPhaseState =
  | { kind: 'idle' }
  | { kind: 'pending'; phase: GamePhase; player: number; startedAt: number }
  | { kind: 'warning_issued'; phase: GamePhase; remainingMs: number }
  | { kind: 'expired'; phase: GamePhase; expiredAt: number }
  | { kind: 'timeout_auto_resolved'; resolvedMoveId: string }
  | { kind: 'cancelled'; reason: string };

// Functions:
function initializeDecisionPhase(phase, player, startedAt): DecisionPhaseState;
function markDecisionWarningIssued(state, remainingMs): DecisionPhaseState;
function markDecisionExpired(state): DecisionPhaseState;
```

#### AI Fallback Logic Duplication (HIGH IMPACT)
**Problem:** AI fallback duplicated in:
- Server: `GameSession.handleNoMoveFromService` (34 lines)
- Sandbox: `maybeRunAITurnSandbox` in `sandboxAI.ts`

**Recommendation:** Extract to shared `AIFallbackHandler`:
```typescript
// src/shared/ai/AIFallbackHandler.ts
function selectFallbackMove(
  playerNumber: number,
  state: GameState,
  validMoves: Move[],
  rng: LocalAIRng
): Move | null;
```

#### Error Handling Inconsistency (MEDIUM IMPACT)
**Problem:** Different error patterns across layers:
- HTTP routes: `createError(message, statusCode, code)`
- WebSocket: Direct error callbacks, no client notification
- GameSession: Throws `new Error(message)` with no error codes

**Recommendation:** Create GameDomainErrors:
```typescript
// src/server/errors/GameDomainErrors.ts
class GameError extends Error {
  code: GameErrorCode;
  context: Record<string, unknown>;
  isFatal: boolean;
}

enum GameErrorCode {
  INVALID_MOVE = 'GAME_INVALID_MOVE',
  GAME_NOT_ACTIVE = 'GAME_NOT_ACTIVE',
  AI_SERVICE_UNAVAILABLE = 'GAME_AI_SERVICE_UNAVAILABLE',
  CHOICE_TIMEOUT = 'GAME_CHOICE_TIMEOUT',
}
```

### 4.2 Priority Actions

| Priority | Action | Lines Saved | Effort |
|----------|--------|-------------|--------|
| P1 | Extract DecisionPhaseState machine | 100 | Small |
| P1 | Extract AIFallbackHandler | 100 | Small |
| P1 | Create GameDomainErrors | 50 | Small |
| P2 | Extract DecisionPhaseManager | 200 | Medium |
| P2 | Extract AITurnCoordinator | 150 | Large |
| P2 | Create MoveApplicationPipeline | 100 | Large |

---

## 5. Test Infrastructure Refactoring

### 5.1 Critical Issues

#### Parity Test Redundancy (HIGH IMPACT)
**Problem:** 31 `*Parity*.test.ts` files (7,571 LOC) largely duplicate contract vector coverage

**Examples:**
- `TraceParity.seed5.firstDivergence.test.ts` - duplicates territory vectors
- `Backend_vs_Sandbox.CaptureAndTerritoryParity.test.ts` (1,625 LOC) - duplicates territory.vectors.json

**Recommendation:** Move parity tests to diagnostic category:
```
tests/
├── unit/core/          # Shared engine rules tests
├── unit/adapters/      # Backend/sandbox behavior tests
├── contracts/          # Contract vector runner
├── parity/
│   └── diagnostic/     # Move 31 *Parity*.test.ts files here
└── e2e/
```

#### Fixture Duplication (MEDIUM IMPACT)
**Problem:**
- TypeScript: 15+ files contain inline `createTestBoard()` instead of importing
- Python: 122 test files with no `conftest.py`, each defining own fixtures

**Recommendation:**
1. Create Python `conftest.py`:
```python
# ai-service/tests/conftest.py
@pytest.fixture
def mock_game_state():
    return GameState(boardType=BoardType.SQUARE8, ...)

@pytest.fixture
def move_factory():
    return lambda p, x, y: Move(...)
```

2. Create TypeScript mock factories:
```typescript
// tests/utils/mockFactories.ts
export class MockFactories {
  static createPythonRulesClient(overrides = {}) { ... }
  static createGameSession(overrides = {}) { ... }
  static createSocketIOServer(overrides = {}) { ... }
}
```

#### Test Organization Issues (MEDIUM IMPACT)
**Problem:**
- `tests/unit/` contains 389 files mixing unit, integration, parity, scenarios
- Naming inconsistency: `*.shared.test.ts`, `*Parity*.test.ts`, `*.rules.test.ts`

**Recommendation:** Restructure directories:
```
tests/
├── unit/
│   ├── core/           # Shared engine (*.shared.test.ts)
│   ├── adapters/       # Host implementations
│   └── mocks/          # Mock utilities
├── integration/        # Service-to-service (real I/O)
│   ├── websocket/
│   ├── game-flow/
│   └── ai-service/
├── contracts/          # Contract vector runner
├── scenarios/          # Consolidate all scenario tests
├── parity/diagnostic/  # Trace/seed-specific tests
└── e2e/
```

### 5.2 Priority Actions

| Priority | Action | Lines Saved | Effort |
|----------|--------|-------------|--------|
| P1 | Create Python conftest.py | 400 | 2 days |
| P1 | Create TS mockFactories.ts | 400 | 2 days |
| P2 | Move parity tests to diagnostic | 6,000 | 1 week |
| P2 | Consolidate fixture utilities | 800 | 3 days |
| P3 | Restructure test directories | 0 | 1 week |
| P3 | Expand contract vector coverage | +1,000 | 2 weeks |

---

## 6. State Machine Consolidation Opportunities

### 6.1 Current State Machines (Well-Defined)

| Machine | Location | Status |
|---------|----------|--------|
| `GameSessionStatus` | `shared/stateMachines/gameSession.ts` | Good |
| `AIRequestState` | `shared/stateMachines/aiRequest.ts` | Good |
| `ChoiceStatus` | `shared/stateMachines/choice.ts` | Good |
| `PlayerConnectionState` | `shared/stateMachines/connection.ts` | Good |
| `TurnStateMachine` | `shared/engine/fsm/TurnStateMachine.ts` | Defined but underutilized |

### 6.2 Missing State Machines (Should Be Extracted)

| Machine | Current Location | Recommended |
|---------|------------------|-------------|
| `DecisionPhaseState` | Scattered in GameSession | `shared/stateMachines/decisionPhase.ts` |
| `GamePhase` (UI) | Scattered in GameContext | Compound state machine in context |
| `EvaluationStrategy` | Implicit in HeuristicAI | `ai-service/app/ai/strategies.py` |

### 6.3 Key State Machine Recommendation

**Make TurnStateMachine the single orchestration authority:**

```typescript
// Current: 3 parallel approaches
// 1. TurnStateMachine (FSM) - defined but not used
// 2. PhaseStateMachine (procedural) - wrapper
// 3. GameEngine.checkStateTransitions() - inline

// Proposed: Single source of truth
function processTurn(state: GameState, move: Move): ProcessTurnResult {
  const fsmState = deriveStateFromGame(state);
  const event = moveToEvent(move);
  const result = TurnStateMachine.transition(fsmState, event, context);

  if (!result.ok) return { success: false, error: result.error };

  let nextState = state;
  for (const action of result.actions) {
    nextState = applyAction(nextState, action);
  }

  return { success: true, nextState, pendingDecision: ... };
}
```

---

## 7. Implementation Roadmap

### Prerequisites Discovered During Implementation

**GameState Type Unification (BLOCKER for validator/mutator consolidation):**

Two separate `GameState` types exist in the codebase:
1. `src/shared/engine/types.ts` - Internal engine GameState (leaner)
2. `src/shared/types/game.ts` - Shared GameState (fuller, with `boardType`, `history`, `spectators`)

The aggregates use the shared GameState, while GameEngine.ts uses the internal engine GameState. This type mismatch prevents direct re-export of aggregate functions from validators/mutators.

**Status:**
- [x] Moved `isValidPosition` from `validators/utils.ts` to `core.ts`
- [x] Updated imports in aggregates and helpers to use `core.ts`
- [ ] **BLOCKED:** Validator/mutator consolidation requires GameState type unification

**Completed Quick Win:**
```typescript
// core.ts now exports isValidPosition
export function isValidPosition(pos: Position, boardType: BoardType, boardSize: number): boolean;
```

### Phase 1: Foundation (Weeks 1-2)

**TS Engine:**
- [ ] ~~Delete `/validators/` directory, re-export from aggregates~~ (BLOCKED - needs type unification)
- [ ] ~~Delete `/mutators/` directory, re-export from aggregates~~ (BLOCKED - needs type unification)
- [ ] **NEW:** Unify `GameState` types (engine/types.ts → types/game.ts)

**Python AI:**
- [x] Extract swap evaluation (~300 lines) → `ai-service/app/ai/swap_evaluation.py`
  - `SwapWeights` dataclass for configurable weights
  - `SwapEvaluator` class with position classification and swap bonus logic
  - HeuristicAI delegates to SwapEvaluator (maintains backward compatibility)
- [x] Create EvaluationProvider interface → `ai-service/app/ai/evaluation_provider.py`
  - `EvaluationProvider` Protocol for position evaluation
  - `HeuristicEvaluator` class with all evaluation features (Tier 0/1/2)
  - `EvaluatorConfig` dataclass for evaluator configuration
  - Enables composition over inheritance for MinimaxAI/MCTSAI
- [x] Extract move ordering heuristics → `ai-service/app/ai/move_ordering.py`
  - `MoveTypePriority` enum with standard priority values
  - `MovePriorityScorer` class for move scoring
  - `KillerMoveTable` for killer move heuristic
  - `order_moves()`, `filter_noisy_moves()`, `score_noisy_moves()` utilities

**Tests:**
- [x] Create Python `conftest.py` with shared fixtures → `ai-service/tests/conftest.py`
  - Factory fixtures: `player_factory`, `move_factory`, `ring_stack_factory`, `board_state_factory`, `game_state_factory`
  - Common fixtures: `empty_game_state`, `game_state_with_stacks`, `game_state_mid_game`
  - AI config fixtures: `ai_config_easy`, `ai_config_medium`, `ai_config_hard`
  - Autouse cache-clearing fixture for test isolation
- [x] Create TS mock factories → Extended `tests/utils/fixtures.ts`
  - `createMockPythonRulesClient()` - Mock Python rules client
  - `createMockSocket()` - Mock Socket.IO socket
  - `createMockSocketIOServer()` - Mock Socket.IO server
  - `createMockUserSockets()` - User-to-socket mapping
  - `createTestMove()` - Test move factory

### Phase 2: State Consolidation (Weeks 3-4)

**Server/Sandbox:**
- [x] Extract DecisionPhaseState machine to shared → `src/shared/decisions/DecisionPhaseState.ts`
  - Discriminated union state machine: idle, pending, warning, expired, resolved, cancelled
  - State transitions: `initializeDecision()`, `issueWarning()`, `expireDecision()`, `resolveDecision()`
  - Query helpers: `isDecisionActive()`, `getRemainingTime()`, `getDecisionMetadata()`
  - Timeout configuration with per-choice-type overrides
- [x] Extract AIFallbackHandler to shared → `src/shared/ai/AIFallbackHandler.ts`
  - `FallbackContext` and `FallbackResult` types for consistent fallback handling
  - `selectFallbackMove()` wrapper with diagnostics
  - RNG utilities: `createLocalAIRng()`, `deriveFallbackSeed()`
  - Cumulative diagnostics tracking for sessions
- [x] Create GameDomainErrors → `src/shared/errors/GameDomainErrors.ts`
  - `GameErrorCode` enum with categorized error codes (GAME_, MOVE_, AI_, DECISION_, PLAYER_)
  - `GameError` base class with code, context, HTTP status mapping
  - Specific error classes: `InvalidMoveError`, `NotYourTurnError`, `AIServiceTimeoutError`, etc.
  - Utilities: `isGameError()`, `wrapError()`, `getHttpStatus()`

**Client:**
- [x] Extract GameFacade abstraction → `src/client/facades/`
  - `GameFacade.ts` - Unified interface for backend and sandbox game hosts
  - `FacadeConnectionStatus`, `GameFacadeMode`, `FacadeDecisionState`, `FacadePlayerInfo` types
  - Utilities: `extractChainCapturePath()`, `deriveMustMoveFrom()`, `canSubmitMove()`, `canInteract()`
- [x] Create useGamePlayViewModels hook → `src/client/facades/useGamePlayViewModels.ts`
  - Derives all view models from facade: `boardViewModel`, `hudViewModel`, `victoryViewModel`, `eventLogViewModel`
  - Handles optional capture highlighting and decision highlights
  - `useInstructionText()` helper for phase-specific instructions
- [x] Create useCellInteractions hook → `src/client/facades/useCellInteractions.ts`
  - Shared cell click/double-click/context-menu handling
  - Selection state management
  - Invalid move detection with reason analysis
- [ ] Implement compound state machines for GameContext

### Phase 3: Component Decomposition (Weeks 5-6)

**Client:**
- [ ] Decompose SandboxGameHost into feature containers
- [ ] Extract BoardView props into hooks
- [ ] Consolidate countdown/timer logic

**TS Engine:**
- [ ] Consolidate orchestration logic to TurnStateMachine
- [ ] Consolidate helpers into aggregates

### Phase 4: Test Cleanup (Weeks 7-8)

- [ ] Move 31 parity tests to diagnostic category
- [ ] Restructure test directories
- [ ] Expand contract vector coverage for gaps

### Phase 5: AI Service Refinement (Weeks 9-10)

- [ ] Implement Strategy pattern for evaluation paths
- [ ] Split HeuristicAI into focused evaluator classes
- [ ] Centralize environment variable config

---

## 8. Metrics and Success Criteria

### Code Reduction Targets

| Area | Current LOC | Target LOC | Reduction |
|------|-------------|------------|-----------|
| TS Validators/Mutators | 800 | 0 | 100% |
| Orchestration duplication | 1,200 | 400 | 67% |
| HeuristicAI | 2,278 | 1,400 | 38% |
| SandboxGameHost | 2,992 | 1,200 | 60% |
| Parity tests in CI | 7,571 | 1,500 | 80% |

### Maintainability Targets

| Metric | Current | Target |
|--------|---------|--------|
| Largest component | 2,992 LOC | <800 LOC |
| Average prop count (React) | 12-24 | <8 |
| useState calls per page | 25+ | <15 |
| Files with >500 LOC | 15+ | <5 |
| State machine coverage | 40% | 80% |

### Test Performance Targets

| Metric | Current | Target |
|--------|---------|--------|
| Core test suite time | ~5 min | <3 min |
| Parity tests in CI | Always | Optional |
| Test clarity (purpose clear) | 60% | 90% |

---

## 9. Risk Assessment

### High-Risk Refactorings

| Refactoring | Risk | Mitigation |
|-------------|------|------------|
| TurnStateMachine adoption | Breaking game logic | Extensive contract vector validation |
| SandboxGameHost decomposition | UI regression | Visual regression testing |
| Python EvaluationProvider | AI strength regression | Benchmark suite before/after |
| Parity test removal | Missing regressions | Ensure contract vectors cover scenarios |

### Low-Risk Quick Wins

| Refactoring | Risk | Impact | Status |
|-------------|------|--------|--------|
| Delete validators/ directory | Minimal | ~400 LOC saved | BLOCKED (type unification needed) |
| Python conftest.py | None | Better test organization | ✅ DONE |
| Extract swap_evaluation.py | Minimal | Testability improvement | ✅ DONE |
| mockFactories.ts | None | Better test maintainability | ✅ DONE (extended fixtures.ts) |
| Move isValidPosition to core.ts | None | Better organization | ✅ DONE |

---

## 10. Conclusion

The RingRift codebase has **solid architectural foundations** but suffers from:

1. **Code duplication** across validators/aggregates, orchestration logic, and test fixtures
2. **God classes** (HeuristicAI, SandboxGameHost) that violate single responsibility
3. **Underutilized state machines** (TurnStateMachine defined but not used as authority)
4. **Scattered UI state** allowing impossible states
5. **Test redundancy** with 7,500+ LOC of parity tests duplicating contract vectors

**Key Recommendations:**
1. **Adopt TurnStateMachine** as single orchestration authority (highest impact)
2. **Extract GameFacade** to unify backend/sandbox flows
3. **Use composition over inheritance** in Python AI (decouple HeuristicAI)
4. **Implement compound state machines** in React for game phases
5. **Move parity tests to diagnostic** category (biggest test cleanup)

**Estimated Total Effort:** 10-12 weeks for full implementation
**Estimated LOC Reduction:** ~16,300 lines (15% of total codebase)
**Maintainability Improvement:** 40-50% reduction in time-to-understand for new contributors
