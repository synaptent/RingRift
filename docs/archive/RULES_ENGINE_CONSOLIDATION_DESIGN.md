> **Doc Status (2025-11-30): Historical / Superseded**
> **Role:** Historical consolidation design for the RingRift rules engine. **This work was completed in November 2025** – the orchestrator and adapters are now at 100% rollout.
>
> **For current documentation, see:**
>
> - `docs/architecture/ORCHESTRATOR_ROLLOUT_PLAN.md` – Production rollout status (Phase 4 complete)
> - `docs/rules/RULES_SSOT_MAP.md` – Current host integration and SSOT boundaries
> - `RULES_ENGINE_ARCHITECTURE.md` – Current architecture overview
> - [`historical/CURRENT_STATE_ASSESSMENT.md`](historical/CURRENT_STATE_ASSESSMENT.md) – Implementation status
>
> **SSoT alignment:** This document is a derived architectural design over the **Rules/invariants semantics SSoT** and the **Canonical TS rules surface**:
>
> - Narrative rules docs: `RULES_CANONICAL_SPEC.md`, `ringrift_complete_rules.md`, `ringrift_compact_rules.md`.
> - Shared TypeScript rules engine helpers + aggregates under `src/shared/engine/**` and the turn orchestrator under `src/shared/engine/orchestration/**`.
> - Contracts and v2 contract vectors under `src/shared/engine/contracts/**` and `tests/fixtures/contract-vectors/v2/**`, plus the TS + Python contract runners (`tests/contracts/contractVectorRunner.test.ts`, `ai-service/tests/contracts/test_contract_vectors.py`).
> - Derived rules/architecture docs: `RULES_ENGINE_ARCHITECTURE.md`, `RULES_IMPLEMENTATION_MAPPING.md`, `docs/rules/RULES_ENGINE_SURFACE_AUDIT.md`, `docs/MODULE_RESPONSIBILITIES.md`, `docs/SHARED_ENGINE_CONSOLIDATION_PLAN.md`.
>
> **Precedence:** This file is **not** a semantics SSoT. If anything here disagrees with the shared TS engine/orchestrator, contracts/vectors, or their tests (including Python contract/parity suites), **code + tests + canonical docs win** and this design must be updated.

# Rules Engine Consolidation Design

**Task:** Priority #1 - Rules Engine Architecture Consolidation
**Date:** 2025-11-26
**Status:** ✅ COMPLETED (November 2025) – Now historical reference only
**Author:** Architect Mode

---

## Executive Summary

This document presents the architectural design for consolidating RingRift's fragmented rules engine implementation into a single canonical source of truth. Currently, THREE rule engine surfaces must be kept in lockstep:

1. **Shared TypeScript Engine** (`src/shared/engine/*`)
2. **Backend Game Engine** (`src/server/game/*`)
3. **Python AI Rules Engine** (`ai-service/app/rules/*`)

Additionally, the client sandbox (`src/client/sandbox/**`) maintains its own orchestration mirroring server logic.

### Key Design Decisions

| Decision               | Recommendation                          | Rationale                                                                       |
| ---------------------- | --------------------------------------- | ------------------------------------------------------------------------------- |
| Single Source of Truth | `src/shared/engine/`                    | Already contains aggregate pattern, pure functions, comprehensive test coverage |
| Backend Pattern        | Thin adapter over shared engine         | Reduce duplicate logic to WebSocket/interaction concerns only                   |
| Sandbox Pattern        | Thin adapter over shared engine         | Consolidate with backend where possible                                         |
| Python Integration     | Contract tests + JSON schema generation | Balance parity guarantees with performance requirements                         |
| Migration Approach     | Phased, backward-compatible             | Preserve existing test infrastructure during transition                         |

---

## 0. Reconciliation Notes (2025-11-28)

This draft predates the latest consolidation work and should be read as an **aspirational design**, not a literal description of current files/APIs. The authoritative view of the rules surfaces and consolidation status now lives in:

- `RULES_ENGINE_ARCHITECTURE.md`
- `RULES_IMPLEMENTATION_MAPPING.md`
- `docs/rules/RULES_ENGINE_SURFACE_AUDIT.md`
- `docs/MODULE_RESPONSIBILITIES.md`
- `docs/SHARED_ENGINE_CONSOLIDATION_PLAN.md`

Key clarifications when reading the sections below:

1. **Engine entry points & module layout**
   - The _actual_ exported surfaces today are the functions and types in `src/shared/engine/index.ts` plus the orchestrator exports in `src/shared/engine/orchestration/index.ts` (e.g. `processTurn`, `processTurnAsync`, `validateMove`, `getValidMoves`, `evaluateVictory`).
   - Names like `resolveTerritory`, `detectLines`, `computeVictoryState`, and modules such as `decisionRouter.ts`, `gameStateSchema.ts`, `moveSchema.ts`, `resultSchema.ts`, or an `internal/` directory are **conceptual design handles**. Their responsibilities are currently implemented across:
     - `src/shared/engine/territoryProcessing.ts`, `src/shared/engine/territoryDecisionHelpers.ts`, `src/shared/engine/lineDetection.ts`, `src/shared/engine/lineDecisionHelpers.ts`.
     - `src/shared/engine/orchestration/turnOrchestrator.ts`, `src/shared/engine/orchestration/phaseStateMachine.ts`, and `src/shared/validation/schemas.ts` / `src/shared/engine/contracts/schemas.ts`.
   - When this document shows alternative filenames, treat them as **target refactors**, not as promises that those files exist today.

2. **Backend and sandbox adapters**
   - There is **no `src/server/game/adapters/EngineAdapter.ts`** or `src/client/sandbox/adapters/SandboxEngineAdapter.ts` in the current codebase. The active adapter/host surfaces are:
     - Backend: `src/server/game/GameEngine.ts`, `src/server/game/RuleEngine.ts`, `src/server/game/turn/TurnEngine.ts`, `src/server/game/turn/TurnEngineAdapter.ts`, `src/server/game/RulesBackendFacade.ts`, and WebSocket/server routes.
     - Sandbox: `src/client/sandbox/ClientSandboxEngine.ts`, `src/client/sandbox/SandboxOrchestratorAdapter.ts`, and the `sandbox*.ts` helpers (`sandboxPlacement.ts`, `sandboxMovement.ts`, `sandboxLines.ts`, `sandboxTerritory.ts`, `sandboxVictory.ts`, `sandboxElimination.ts`, `sandboxGameEnd.ts`, `sandboxCaptures.ts`, `sandboxCaptureSearch.ts`).
   - Sections below that mention `ServerEngineAdapter`/`SandboxEngineAdapter` are **future-facing patterns**. They should be interpreted as “what we want GameEngine/ClientSandboxEngine to look like” rather than as additional shared SSoT modules.

3. **Legacy module migration**
   - Backend helpers such as `src/server/game/rules/lineProcessing.ts` and `src/server/game/rules/territoryProcessing.ts` have already been removed and are treated as **historical** in current docs.
   - Consolidated sandbox engines like `sandboxLinesEngine.ts` / `sandboxTerritoryEngine.ts` no longer exist. The remaining `sandbox*.ts` files listed above are thin host helpers over the shared engine/orchestrator, not alternative rules engines.
   - Migration tables and “Files to Remove” sections should therefore be read as a **historical plan**; the authoritative list of completed vs open items is in `docs/SHARED_ENGINE_CONSOLIDATION_PLAN.md`.

4. **Contract vectors and Python parity**
   - The core of the “contract-based parity” design is already implemented using:
     - TS-side generator and serialization under `src/shared/engine/contracts/testVectorGenerator.ts` and `src/shared/engine/contracts/serialization.ts`.
     - Canonical v2 JSON fixtures under `tests/fixtures/contract-vectors/v2/*.vectors.json`.
     - TS runner: `tests/contracts/contractVectorRunner.test.ts`.
     - Python runner: `ai-service/tests/contracts/test_contract_vectors.py`.
   - Snippets below that refer to `generate-vectors.ts`, `ts_vectors/*.json`, or `test_contract_parity.py` are earlier naming sketches; the **paths above are the SSoT** for contract layout and parity tests.

5. **Migration phases and timelines**
   - Phases P1–P5 in this draft describe a **multi-week migration plan**. Large parts of P1–P3 are already complete and captured as “done” in `docs/SHARED_ENGINE_CONSOLIDATION_PLAN.md` and the reconciled rules docs; remaining items should be treated as **aspirational** and re-validated against current code/tests before implementation.

When in doubt, consult the SSoT docs and the live TS engine/orchestrator/contracts first, then treat the remainder of this file as a design sketch for future refactors.

---

## 1. Current State Analysis

This document presents the architectural design for consolidating RingRift's fragmented rules engine implementation into a single canonical source of truth. Currently, THREE rule engine surfaces must be kept in lockstep:

1. **Shared TypeScript Engine** (`src/shared/engine/*`)
2. **Backend Game Engine** (`src/server/game/*`)
3. **Python AI Rules Engine** (`ai-service/app/rules/*`)

Additionally, the client sandbox (`src/client/sandbox/**`) maintains its own orchestration mirroring server logic.

### Key Design Decisions

| Decision               | Recommendation                          | Rationale                                                                       |
| ---------------------- | --------------------------------------- | ------------------------------------------------------------------------------- |
| Single Source of Truth | `src/shared/engine/`                    | Already contains aggregate pattern, pure functions, comprehensive test coverage |
| Backend Pattern        | Thin adapter over shared engine         | Reduce duplicate logic to WebSocket/interaction concerns only                   |
| Sandbox Pattern        | Thin adapter over shared engine         | Consolidate with backend where possible                                         |
| Python Integration     | Contract tests + JSON schema generation | Balance parity guarantees with performance requirements                         |
| Migration Approach     | Phased, backward-compatible             | Preserve existing test infrastructure during transition                         |

---

## 1. Current State Analysis

### 1.1 Surface Inventory

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CURRENT ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────────────────────────────────┐                              │
│   │     Shared Engine (src/shared/engine)     │  ◄── Canonical Source       │
│   │  ~4,500 LOC | 26 files                   │                              │
│   │  6 Aggregates + Core                      │                              │
│   └────────────────┬─────────────────────────┘                              │
│                    │                                                         │
│       ┌────────────┼────────────┬────────────────────┐                      │
│       │            │            │                    │                      │
│       ▼            ▼            ▼                    ▼                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────────────────┐              │
│  │ Backend │  │ Sandbox │  │  Tests  │  │    Python AI      │              │
│  │ ~6K LOC │  │ ~5.2K   │  │ Parity  │  │  ~2K LOC          │              │
│  │ Adapter │  │ Adapter │  │ Suites  │  │  DUPLICATES LOGIC │ ◄── Risk!   │
│  └─────────┘  └─────────┘  └─────────┘  └───────────────────┘              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Aggregate Structure (Already Exists)

The shared engine already implements a domain aggregate pattern:

| Aggregate              | Location                           | Responsibilities                                 |
| ---------------------- | ---------------------------------- | ------------------------------------------------ |
| **PlacementAggregate** | `aggregates/PlacementAggregate.ts` | Ring placement validation, mutation, enumeration |
| **MovementAggregate**  | `aggregates/MovementAggregate.ts`  | Non-capturing movement logic                     |
| **CaptureAggregate**   | `aggregates/CaptureAggregate.ts`   | Overtaking captures, chain capture state         |
| **LineAggregate**      | `aggregates/LineAggregate.ts`      | Line detection, collapse decisions               |
| **TerritoryAggregate** | `aggregates/TerritoryAggregate.ts` | Region detection, Q23, elimination               |
| **VictoryAggregate**   | `aggregates/VictoryAggregate.ts`   | Victory condition evaluation                     |

### 1.3 Current Pain Points

1. **Python Duplication**: `ai-service/app/rules/` reimplements ~2000 LOC of rules logic
2. **Shadow Contract Overhead**: Runtime parity validation adds latency
3. **Adapter Inconsistency**: Backend and sandbox have different patterns for same logic
4. **Test Fragmentation**: Multiple parity test suites must stay synchronized

---

## 2. Canonical Rules Engine API Design

### 2.1 Entry Points

The consolidated engine exposes four primary entry points:

```typescript
// ═══════════════════════════════════════════════════════════════════════════
// PRIMARY ENGINE ENTRY POINTS
// Location: src/shared/engine/index.ts
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Process a turn from beginning to end, handling all phases.
 *
 * This is the top-level orchestration function that adapters call.
 * It processes placement → movement/capture → lines → territory → victory.
 *
 * @param state Current game state (immutable)
 * @param move The move to apply
 * @param delegates Host-specific callbacks for async decisions
 * @returns ProcessTurnResult with next state and pending decisions
 */
export function processTurn(
  state: GameState,
  move: Move,
  delegates: TurnProcessingDelegates
): ProcessTurnResult;

/**
 * Resolve all territory regions for the current player.
 *
 * Called during territory_processing phase. Handles Q23 prerequisites,
 * region collapse, border markers, and self-elimination bookkeeping.
 *
 * @param state Current game state
 * @param options Processing options
 * @returns TerritoryResolutionResult with updated state and pending eliminations
 */
export function resolveTerritory(
  state: GameState,
  options?: TerritoryResolutionOptions
): TerritoryResolutionResult;

/**
 * Detect all marker lines on the board and compute collapse options.
 *
 * Returns lines that meet the minimum length threshold for the board type.
 *
 * @param state Current game state
 * @param player Player to detect lines for
 * @returns Array of detected lines with collapse options
 */
export function detectLines(state: GameState, player: number): LineDetectionResult;

/**
 * Compute the victory state for the current game.
 *
 * Evaluates all victory conditions: ring elimination, territory control,
 * last player standing, and stalemate resolution.
 *
 * @param state Current game state
 * @returns VictoryState with winner, reason, and detailed scoring
 */
export function computeVictoryState(state: GameState): VictoryState;
```

### 2.2 Module Boundaries

```
src/shared/engine/
├── index.ts                    # PUBLIC: Entry points and re-exports
│
├── core.ts                     # PUBLIC: Geometry, stack calculations, hashing
├── types.ts                    # PUBLIC: ValidationResult, Action types
│
├── aggregates/                 # PUBLIC: Domain aggregates
│   ├── PlacementAggregate.ts   #   Placement validation, mutation, enumeration
│   ├── MovementAggregate.ts    #   Movement validation, mutation, enumeration
│   ├── CaptureAggregate.ts     #   Capture validation, mutation, chain state
│   ├── LineAggregate.ts        #   Line detection, collapse decisions
│   ├── TerritoryAggregate.ts   #   Region detection, processing, Q23
│   └── VictoryAggregate.ts     #   Victory evaluation, tie-breaking
│
├── orchestration/              # PUBLIC: Turn flow coordination
│   ├── turnOrchestrator.ts     #   processTurn implementation
│   ├── phaseStateMachine.ts    #   Phase transitions
│   └── decisionRouter.ts       #   Route player decisions to aggregates
│
├── contracts/                  # PUBLIC: Schema definitions for Python
│   ├── gameStateSchema.ts      #   JSON Schema for GameState
│   ├── moveSchema.ts           #   JSON Schema for Move types
│   └── resultSchema.ts         #   JSON Schema for results
│
└── internal/                   # PRIVATE: Implementation details
    ├── geometry.ts             #   Board geometry calculations
    ├── adjacencyCache.ts       #   Cached adjacency graphs
    └── stateCloning.ts         #   Immutable state utilities
```

### 2.3 Public vs Internal Boundary

| Category                | Export Strategy                 | Stability                                                 |
| ----------------------- | ------------------------------- | --------------------------------------------------------- |
| **Entry Points**        | Direct export from `index.ts`   | Stable: Breaking changes require major version            |
| **Aggregate Functions** | Named exports via `index.ts`    | Stable: Signature changes are breaking                    |
| **Types**               | Re-export from `types.ts`       | Stable: Field additions allowed, removals breaking        |
| **Core Utilities**      | Selective export from `core.ts` | Semi-stable: New functions OK, signature changes breaking |
| **Internal Helpers**    | Not exported                    | Unstable: Can change freely                               |

### 2.4 Type Specifications

```typescript
// ═══════════════════════════════════════════════════════════════════════════
// PROCESS TURN TYPES
// ═══════════════════════════════════════════════════════════════════════════

export interface ProcessTurnResult {
  /** Next game state after applying the move */
  nextState: GameState;

  /** Whether the turn completed or requires more decisions */
  status: 'complete' | 'awaiting_decision';

  /** Pending decision if status is 'awaiting_decision' */
  pendingDecision?: PendingDecision;

  /** Victory result if game ended */
  victoryResult?: VictoryState;

  /** Processing metadata for debugging/logging */
  metadata: ProcessingMetadata;
}

export interface PendingDecision {
  /** Type of decision required */
  type: 'line_order' | 'line_reward' | 'region_order' | 'elimination_target' | 'capture_direction';

  /** Player who must make the decision */
  player: number;

  /** Available options */
  options: Move[];

  /** Timeout in milliseconds (adapter concern) */
  timeoutMs?: number;

  /** Context for UI rendering */
  context: DecisionContext;
}

export interface TurnProcessingDelegates {
  /** Resolve a player decision asynchronously */
  resolveDecision(decision: PendingDecision): Promise<Move>;

  /** Log processing events */
  onProcessingEvent?(event: ProcessingEvent): void;

  /** Override default auto-selection for AI */
  autoSelectStrategy?: AutoSelectStrategy;
}

// ═══════════════════════════════════════════════════════════════════════════
// TERRITORY RESOLUTION TYPES
// ═══════════════════════════════════════════════════════════════════════════

export interface TerritoryResolutionResult {
  /** Next game state after territory processing */
  nextState: GameState;

  /** Regions that were processed */
  processedRegions: ProcessedRegion[];

  /** Pending self-eliminations */
  pendingEliminations: EliminationDecision[];

  /** Whether more regions need processing */
  hasMoreRegions: boolean;
}

export interface ProcessedRegion {
  /** Identifier for this region */
  id: string;

  /** Spaces that were collapsed */
  collapsedSpaces: Position[];

  /** Border markers that were collapsed */
  borderMarkers: Position[];

  /** Rings eliminated from internal stacks */
  eliminatedRings: { player: number; count: number }[];

  /** Territory gained */
  territoryGained: number;
}

// ═══════════════════════════════════════════════════════════════════════════
// LINE DETECTION TYPES
// ═══════════════════════════════════════════════════════════════════════════

export interface LineDetectionResult {
  /** Detected lines meeting minimum length */
  lines: DetectedLine[];

  /** Board type used for detection */
  boardType: BoardType;

  /** Minimum line length for this board */
  minimumLength: number;
}

export interface DetectedLine {
  /** Line positions in order */
  positions: Position[];

  /** Player who owns this line */
  player: number;

  /** Line length */
  length: number;

  /** Direction vector */
  direction: Position;

  /** Available collapse options */
  collapseOptions: LineCollapseOption[];
}

export interface LineCollapseOption {
  /** Option type */
  type: 'collapse_all' | 'minimum_collapse';

  /** Positions that would be collapsed */
  positions: Position[];

  /** Whether this grants elimination reward */
  grantsReward: boolean;

  /** Territory gained */
  territoryGained: number;
}

// ═══════════════════════════════════════════════════════════════════════════
// VICTORY STATE TYPES
// ═══════════════════════════════════════════════════════════════════════════

export interface VictoryState {
  /** Whether the game has ended */
  isGameOver: boolean;

  /** Winner player number (undefined if draw or ongoing) */
  winner?: number;

  /** Victory reason */
  reason?: VictoryReason;

  /** Detailed scores for all players */
  scores: PlayerScore[];

  /** Tie-breaking information */
  tieBreaker?: TieBreaker;
}

export type VictoryReason =
  | 'ring_elimination'
  | 'territory_control'
  | 'last_player_standing'
  | 'stalemate_resolution'
  | 'resignation';

export interface PlayerScore {
  player: number;
  eliminatedRings: number;
  territorySpaces: number;
  ringsOnBoard: number;
  ringsInHand: number;
  isEliminated: boolean;
}
```

---

## 3. Backend/Sandbox Adapter Pattern

### 3.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ADAPTER ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Canonical Engine (Pure)                           │    │
│  │  ┌──────────┬──────────┬──────────┬──────────┬──────────┬────────┐  │    │
│  │  │Placement │Movement  │Capture   │Line      │Territory │Victory │  │    │
│  │  │Aggregate │Aggregate │Aggregate │Aggregate │Aggregate │Aggregate│ │    │
│  │  └──────────┴──────────┴──────────┴──────────┴──────────┴────────┘  │    │
│  │                                                                      │    │
│  │  ┌────────────────────────────────────────────────────────────────┐  │    │
│  │  │              Turn Orchestrator (processTurn)                   │  │    │
│  │  └────────────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│              ┌─────────────────────┼─────────────────────┐                  │
│              │                     │                     │                  │
│              ▼                     ▼                     ▼                  │
│  ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────────┐      │
│  │  Server Adapter   │ │  Sandbox Adapter  │ │  Test Harness Adapter │      │
│  │                   │ │                   │ │                       │      │
│  │ - WebSocket I/O   │ │ - Local state     │ │ - Trace recording     │      │
│  │ - Persistence     │ │ - UI callbacks    │ │ - Snapshot comparison │      │
│  │ - Player sessions │ │ - AI integration  │ │ - Parity validation   │      │
│  │ - Timeout mgmt    │ │ - Debug logging   │ │                       │      │
│  └───────────────────┘ └───────────────────┘ └───────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Server Adapter Specification

**Location:** `src/server/game/adapters/EngineAdapter.ts`

```typescript
/**
 * Server adapter for the canonical rules engine.
 *
 * Responsibilities:
 * - Receive WebSocket/HTTP move requests
 * - Convert to canonical Move format
 * - Call processTurn() on canonical engine
 * - Handle pending decisions via player interaction
 * - Persist state changes to database
 * - Emit WebSocket events for state updates
 *
 * NOT responsible for:
 * - Rules logic (delegated to canonical engine)
 * - Victory detection (delegated to canonical engine)
 * - Line/territory detection (delegated to canonical engine)
 */
export class ServerEngineAdapter {
  constructor(
    private readonly session: GameSession,
    private readonly interactionHandler: InteractionHandler,
    private readonly persistence: StatePersistence,
    private readonly eventEmitter: GameEventEmitter
  ) {}

  /**
   * Process an incoming move from a player.
   */
  async processMove(move: Move): Promise<MoveResult> {
    // 1. Validate move format (not rules - engine does that)
    this.validateMoveFormat(move);

    // 2. Create delegates that route decisions to real players
    const delegates: TurnProcessingDelegates = {
      resolveDecision: (decision) => this.resolvePlayerDecision(decision),
      onProcessingEvent: (event) => this.logEvent(event),
    };

    // 3. Delegate to canonical engine
    const result = await processTurn(this.session.gameState, move, delegates);

    // 4. Handle results
    if (result.status === 'complete') {
      await this.persistence.saveState(result.nextState);
      this.eventEmitter.emit('turn_complete', result);
    }

    if (result.victoryResult?.isGameOver) {
      await this.handleGameEnd(result.victoryResult);
    }

    return { success: true, result };
  }

  /**
   * Route a pending decision to the appropriate player.
   */
  private async resolvePlayerDecision(decision: PendingDecision): Promise<Move> {
    const player = this.session.getPlayer(decision.player);

    if (player.type === 'ai') {
      return this.resolveAIDecision(decision);
    }

    // Request decision via WebSocket
    return this.interactionHandler.requestPlayerChoice(
      decision.player,
      decision.type,
      decision.options,
      decision.timeoutMs
    );
  }
}
```

### 3.3 Sandbox Adapter Specification

**Location:** `src/client/sandbox/adapters/SandboxEngineAdapter.ts`

```typescript
/**
 * Sandbox adapter for local game simulation.
 *
 * Responsibilities:
 * - Manage local game state
 * - Route AI decisions through local AI engine
 * - Provide hooks for UI rendering
 * - Support debug/trace modes
 *
 * NOT responsible for:
 * - Rules logic (delegated to canonical engine)
 * - Network communication
 * - Persistence
 */
export class SandboxEngineAdapter {
  private gameState: GameState;
  private readonly aiEngine: LocalAIEngine;
  private readonly uiCallbacks: SandboxUICallbacks;

  constructor(config: SandboxConfig) {
    this.gameState = createInitialState(config.boardType, config.players);
    this.aiEngine = new LocalAIEngine(config.aiProfile);
    this.uiCallbacks = config.callbacks;
  }

  /**
   * Execute a turn (player or AI).
   */
  async executeTurn(move: Move): Promise<TurnResult> {
    const delegates: TurnProcessingDelegates = {
      resolveDecision: (decision) => this.resolveLocalDecision(decision),
      onProcessingEvent: (event) => this.uiCallbacks.onEvent?.(event),
    };

    const result = await processTurn(this.gameState, move, delegates);

    this.gameState = result.nextState;
    this.uiCallbacks.onStateChange(this.gameState);

    return result;
  }

  /**
   * Let AI make a move for current player.
   */
  async executeAITurn(): Promise<TurnResult> {
    const aiMove = await this.aiEngine.selectMove(this.gameState);
    return this.executeTurn(aiMove);
  }

  /**
   * Resolve decisions locally (AI or auto-select).
   */
  private async resolveLocalDecision(decision: PendingDecision): Promise<Move> {
    const player = this.gameState.players.find((p) => p.playerNumber === decision.player);

    if (player?.type === 'ai') {
      return this.aiEngine.selectDecision(decision);
    }

    // For human players in sandbox, use UI callback
    return this.uiCallbacks.requestDecision(decision);
  }
}
```

### 3.4 Legacy Module Migration

| Current Module                                 | Migration Path                       | Phase   |
| ---------------------------------------------- | ------------------------------------ | ------- |
| `src/server/game/GameEngine.ts`                | Thin wrapper calling `processTurn()` | Phase 2 |
| `src/server/game/RuleEngine.ts`                | Replace with aggregate validation    | Phase 2 |
| `src/server/game/BoardManager.ts`              | Replace with core utilities          | Phase 3 |
| `src/server/game/rules/lineProcessing.ts`      | Remove (use `LineAggregate`)         | Phase 2 |
| `src/server/game/rules/territoryProcessing.ts` | Remove (use `TerritoryAggregate`)    | Phase 2 |
| `src/client/sandbox/ClientSandboxEngine.ts`    | Thin wrapper calling `processTurn()` | Phase 2 |
| `src/client/sandbox/sandboxLines*.ts`          | Remove (use `LineAggregate`)         | Phase 2 |
| `src/client/sandbox/sandboxTerritory*.ts`      | Remove (use `TerritoryAggregate`)    | Phase 2 |

---

## 4. Python Integration Strategy

### 4.1 Option Analysis

| Option                  | Pros                        | Cons                                   | Latency Impact |
| ----------------------- | --------------------------- | -------------------------------------- | -------------- |
| **A: HTTP Bridge**      | Simple, current approach    | High latency (~50ms), network overhead | High           |
| **B: WASM Compilation** | Perfect parity, fast        | Complex tooling, memory constraints    | Low            |
| **C: Code Generation**  | Good parity, maintainable   | Requires transpiler maintenance        | None           |
| **D: Contract Tests**   | Catches drift, low overhead | Doesn't prevent bugs                   | None           |

### 4.2 Recommended Approach: Contract Tests + JSON Schema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PYTHON INTEGRATION STRATEGY                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TypeScript Canonical Engine                                                 │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │  src/shared/engine/                                                 │     │
│  │  ├── contracts/                                                     │     │
│  │  │   ├── gameStateSchema.ts  ────► GameState JSON Schema           │     │
│  │  │   ├── moveSchema.ts       ────► Move JSON Schema                │     │
│  │  │   └── resultSchema.ts     ────► Result JSON Schema              │     │
│  │  └── test-vectors/                                                  │     │
│  │      └── generate-vectors.ts ────► Test vector generation          │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                       │                                      │
│                                       │ JSON Schemas + Test Vectors          │
│                                       ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │  Contract Validation Layer                                          │     │
│  │  tests/parity/                                                      │     │
│  │  ├── ts_vectors/*.json       ◄─── Generated from TS engine         │     │
│  │  ├── schema/*.json           ◄─── JSON Schema definitions          │     │
│  │  └── test_contract_parity.py ◄─── Validates Python against vectors │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                       │                                      │
│                                       │ Validated Contract                   │
│                                       ▼                                      │
│  Python AI Service                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │  ai-service/app/rules/                                              │     │
│  │  ├── contract_types.py      ◄─── Generated from JSON Schema        │     │
│  │  ├── game_engine.py         ◄─── AI-optimized implementation       │     │
│  │  └── validators/*.py        ◄─── Simplified validators             │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Contract Test Specification

```typescript
// ═══════════════════════════════════════════════════════════════════════════
// TEST VECTOR GENERATION (TypeScript)
// Location: src/shared/engine/contracts/generate-vectors.ts
// ═══════════════════════════════════════════════════════════════════════════

interface TestVector {
  /** Unique identifier for this test case */
  id: string;

  /** Human-readable description */
  description: string;

  /** Input state before the move */
  inputState: GameState;

  /** Move to apply */
  move: Move;

  /** Expected output state after the move */
  expectedState: GameState;

  /** Expected intermediate decisions (if any) */
  expectedDecisions?: PendingDecision[];

  /** State hash for quick comparison */
  stateHash: string;

  /** S-invariant value for progress verification */
  sInvariant: number;

  /** Test category for organization */
  category: TestCategory;
}

type TestCategory =
  | 'placement'
  | 'movement'
  | 'capture'
  | 'chain_capture'
  | 'line_exact'
  | 'line_overlength'
  | 'territory_simple'
  | 'territory_q23'
  | 'victory_elimination'
  | 'victory_territory'
  | 'forced_elimination';

/**
 * Generate comprehensive test vectors from the canonical engine.
 */
function generateTestVectors(): TestVector[] {
  const vectors: TestVector[] = [];

  // Generate vectors for each category
  vectors.push(...generatePlacementVectors());
  vectors.push(...generateMovementVectors());
  vectors.push(...generateCaptureVectors());
  vectors.push(...generateLineSenarios());
  vectors.push(...generateTerritoryScenarios());
  vectors.push(...generateVictoryScenarios());

  // Add seed-based comprehensive traces
  for (const seed of CANONICAL_TEST_SEEDS) {
    vectors.push(...generateTraceFromSeed(seed));
  }

  return vectors;
}
```

```python
# ═══════════════════════════════════════════════════════════════════════════
# CONTRACT PARITY TEST (Python)
# Location: ai-service/tests/parity/test_contract_parity.py
# ═══════════════════════════════════════════════════════════════════════════

import pytest
import json
from pathlib import Path
from app.game_engine import GameEngine
from app.rules.core import hash_game_state, compute_s_invariant

VECTORS_DIR = Path(__file__).parent / "ts_vectors"

def load_test_vectors():
    """Load all test vectors from JSON files."""
    vectors = []
    for vector_file in VECTORS_DIR.glob("*.json"):
        with open(vector_file) as f:
            vectors.extend(json.load(f))
    return vectors

@pytest.mark.parametrize("vector", load_test_vectors(), ids=lambda v: v["id"])
def test_move_parity(vector):
    """Verify Python engine matches TypeScript for this test vector."""
    # Parse input state
    input_state = GameState.from_dict(vector["inputState"])
    move = Move.from_dict(vector["move"])

    # Apply move with Python engine
    result_state = GameEngine.apply_move(input_state, move)

    # Verify state hash matches
    result_hash = hash_game_state(result_state)
    assert result_hash == vector["stateHash"], (
        f"Hash mismatch for {vector['id']}: "
        f"expected {vector['stateHash']}, got {result_hash}"
    )

    # Verify S-invariant
    s_value = compute_s_invariant(result_state)
    assert s_value == vector["sInvariant"], (
        f"S-invariant mismatch for {vector['id']}: "
        f"expected {vector['sInvariant']}, got {s_value}"
    )

@pytest.mark.parametrize("vector", load_test_vectors(), ids=lambda v: v["id"])
def test_decision_enumeration_parity(vector):
    """Verify Python enumerates same decisions as TypeScript."""
    if "expectedDecisions" not in vector:
        pytest.skip("No decision expectations for this vector")

    input_state = GameState.from_dict(vector["inputState"])
    move = Move.from_dict(vector["move"])

    # Get decisions from Python engine
    py_decisions = GameEngine.enumerate_decisions(input_state, move)

    # Compare decision sets (order-independent)
    expected_ids = {d["id"] for d in vector["expectedDecisions"]}
    actual_ids = {d.id for d in py_decisions}

    assert expected_ids == actual_ids, (
        f"Decision mismatch for {vector['id']}: "
        f"expected {expected_ids}, got {actual_ids}"
    )
```

### 4.4 JSON Schema Generation

```typescript
// ═══════════════════════════════════════════════════════════════════════════
// JSON SCHEMA DEFINITIONS
// Location: src/shared/engine/contracts/schemas/
// ═══════════════════════════════════════════════════════════════════════════

// gameState.schema.json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://ringrift.game/schemas/game-state.json",
  "title": "GameState",
  "type": "object",
  "required": ["board", "players", "currentPlayer", "currentPhase", "gameStatus"],
  "properties": {
    "board": { "$ref": "#/definitions/BoardState" },
    "players": {
      "type": "array",
      "items": { "$ref": "#/definitions/Player" }
    },
    "currentPlayer": { "type": "integer", "minimum": 1 },
    "currentPhase": { "$ref": "#/definitions/GamePhase" },
    "gameStatus": { "$ref": "#/definitions/GameStatus" }
  },
  "definitions": {
    "BoardState": {
      "type": "object",
      "required": ["type", "size", "stacks", "markers", "collapsedSpaces"],
      "properties": {
        "type": { "enum": ["square8", "square19", "hexagonal"] },
        "size": { "type": "integer" },
        "stacks": { "$ref": "#/definitions/StackMap" },
        "markers": { "$ref": "#/definitions/MarkerMap" },
        "collapsedSpaces": { "$ref": "#/definitions/CollapsedMap" }
      }
    },
    // ... additional definitions
  }
}
```

---

## 5. Migration Strategy

### 5.1 Phase Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MIGRATION TIMELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 1: Foundation (Week 1-2)                                             │
│  ├─ Implement processTurn orchestrator                                      │
│  ├─ Generate JSON schemas from types                                        │
│  ├─ Create test vector generation tooling                                   │
│  └─ Establish baseline parity metrics                                       │
│                                                                              │
│  Phase 2: Backend Consolidation (Week 3-4)                                  │
│  ├─ Create ServerEngineAdapter                                              │
│  ├─ Migrate GameEngine to use processTurn                                   │
│  ├─ Remove duplicate line/territory processing                              │
│  └─ Update WebSocket handlers                                               │
│                                                                              │
│  Phase 3: Sandbox Consolidation (Week 5-6)                                  │
│  ├─ Create SandboxEngineAdapter                                             │
│  ├─ Migrate ClientSandboxEngine                                             │
│  ├─ Remove legacy sandbox*Engine files                                      │
│  └─ Update AI integration                                                   │
│                                                                              │
│  Phase 4: Python Contract Migration (Week 7-8)                              │
│  ├─ Generate comprehensive test vectors                                     │
│  ├─ Implement contract parity tests                                         │
│  ├─ Simplify Python validators to thin schema validation                    │
│  └─ Remove shadow contract runtime overhead                                 │
│                                                                              │
│  Phase 5: Cleanup & Documentation (Week 9-10)                               │
│  ├─ Remove deprecated modules                                               │
│  ├─ Update architecture documentation                                       │
│  ├─ Convert existing parity tests to contract tests                         │
│  └─ Performance optimization pass                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Phase 1: Foundation (2 weeks)

**Goal:** Create orchestration layer and contract infrastructure

**Tasks:**

| Task                                 | File(s)                                                | Effort |
| ------------------------------------ | ------------------------------------------------------ | ------ |
| Implement `processTurn` orchestrator | `src/shared/engine/orchestration/turnOrchestrator.ts`  | 3 days |
| Create phase state machine           | `src/shared/engine/orchestration/phaseStateMachine.ts` | 2 days |
| Implement decision router            | `src/shared/engine/orchestration/decisionRouter.ts`    | 2 days |
| Generate JSON schemas                | `src/shared/engine/contracts/schemas/*.json`           | 1 day  |
| Create test vector generator         | `src/shared/engine/contracts/generate-vectors.ts`      | 2 days |

**Success Criteria:**

- [ ] `processTurn` handles complete turn flow for all phases
- [ ] JSON schemas validate existing test fixtures
- [ ] 100+ test vectors generated covering all move types

**Tests Affected:**

- New: `tests/unit/processTurn.orchestration.test.ts`
- New: `tests/contracts/schema-validation.test.ts`

### 5.3 Phase 2: Backend Consolidation (2 weeks)

**Goal:** Server GameEngine becomes thin adapter over canonical engine

**Tasks:**

| Task                             | File(s)                                        | Effort |
| -------------------------------- | ---------------------------------------------- | ------ |
| Create `ServerEngineAdapter`     | `src/server/game/adapters/EngineAdapter.ts`    | 2 days |
| Migrate `GameEngine.processMove` | `src/server/game/GameEngine.ts`                | 3 days |
| Update `RulesBackendFacade`      | `src/server/game/RulesBackendFacade.ts`        | 1 day  |
| Remove `lineProcessing.ts`       | `src/server/game/rules/lineProcessing.ts`      | 1 day  |
| Remove `territoryProcessing.ts`  | `src/server/game/rules/territoryProcessing.ts` | 1 day  |
| Update WebSocket handlers        | `src/server/websocket/server.ts`               | 2 days |

**Success Criteria:**

- [ ] All existing backend tests pass without modification
- [ ] Backend GameEngine LOC reduced by 40%+
- [ ] Parity tests continue to pass

**Tests Affected:**

- Existing: `tests/unit/GameEngine.*.test.ts` (must pass unchanged)
- Modified: Backend parity tests use new adapter

### 5.4 Phase 3: Sandbox Consolidation (2 weeks)

**Goal:** ClientSandboxEngine becomes thin adapter over canonical engine

**Tasks:**

| Task                               | File(s)                                               | Effort  |
| ---------------------------------- | ----------------------------------------------------- | ------- |
| Create `SandboxEngineAdapter`      | `src/client/sandbox/adapters/SandboxEngineAdapter.ts` | 2 days  |
| Migrate `ClientSandboxEngine`      | `src/client/sandbox/ClientSandboxEngine.ts`           | 3 days  |
| Remove `sandboxLinesEngine.ts`     | `src/client/sandbox/sandboxLinesEngine.ts`            | 0.5 day |
| Remove `sandboxTerritoryEngine.ts` | `src/client/sandbox/sandboxTerritoryEngine.ts`        | 0.5 day |
| Remove `sandboxLines.ts`           | `src/client/sandbox/sandboxLines.ts`                  | 0.5 day |
| Remove `sandboxTerritory.ts`       | `src/client/sandbox/sandboxTerritory.ts`              | 0.5 day |
| Update AI integration              | `src/client/sandbox/sandboxAI.ts`                     | 2 days  |

**Success Criteria:**

- [ ] All sandbox parity tests pass
- [ ] AI simulation tests terminate correctly
- [ ] Sandbox LOC reduced by 50%+

**Tests Affected:**

- Existing: `tests/unit/ClientSandboxEngine.*.test.ts` (must pass unchanged)
- Existing: `tests/scenarios/RulesMatrix.*.test.ts` (must pass unchanged)

### 5.5 Phase 4: Python Contract Migration (2 weeks)

**Goal:** Python AI service validated by contracts, runtime shadow removed

**Tasks:**

| Task                              | File(s)                                           | Effort |
| --------------------------------- | ------------------------------------------------- | ------ |
| Generate comprehensive vectors    | `tests/parity/ts_vectors/*.json`                  | 2 days |
| Implement contract parity tests   | `ai-service/tests/parity/test_contract_parity.py` | 2 days |
| Generate Python types from schema | `ai-service/app/rules/contract_types.py`          | 1 day  |
| Simplify Python validators        | `ai-service/app/rules/validators/*.py`            | 2 days |
| Remove shadow contract code       | `ai-service/app/rules/default_engine.py`          | 1 day  |
| Update Python game_engine         | `ai-service/app/game_engine.py`                   | 2 days |

**Success Criteria:**

- [ ] 100% of test vectors pass in Python
- [ ] Shadow contract runtime removed
- [ ] Python test execution time reduced by 30%+

**Tests Affected:**

- New: `ai-service/tests/parity/test_contract_parity.py`
- Deprecated: `ai-service/tests/parity/test_rules_parity.py` (converted to contracts)

### 5.6 Phase 5: Cleanup & Documentation (2 weeks)

**Goal:** Remove deprecated code, finalize documentation

**Tasks:**

<<<<<<< Updated upstream
| Task                                   | File(s)                        | Effort |
| -------------------------------------- | ------------------------------ | ------ |
| Remove deprecated backend modules      | Various                        | 1 day  |
| Remove deprecated sandbox modules      | Various                        | 1 day  |
| Update `RULES_ENGINE_ARCHITECTURE.md`  | `RULES_ENGINE_ARCHITECTURE.md` | 2 days |
| Update `CANONICAL_ENGINE_API.md`       | `docs/architecture/CANONICAL_ENGINE_API.md` | 1 day  |
| Convert parity tests to contract tests | `tests/unit/*Parity*.test.ts`  | 2 days |
| Performance optimization               | Various                        | 3 days |
=======
| Task                                   | File(s)                                     | Effort |
| -------------------------------------- | ------------------------------------------- | ------ |
| Remove deprecated backend modules      | Various                                     | 1 day  |
| Remove deprecated sandbox modules      | Various                                     | 1 day  |
| Update `RULES_ENGINE_ARCHITECTURE.md`  | `RULES_ENGINE_ARCHITECTURE.md`              | 2 days |
| Update `CANONICAL_ENGINE_API.md`       | `docs/architecture/CANONICAL_ENGINE_API.md` | 1 day  |
| Convert parity tests to contract tests | `tests/unit/*Parity*.test.ts`               | 2 days |
| Performance optimization               | Various                                     | 3 days |
>>>>>>> Stashed changes

**Files to Remove:**

- `src/server/game/rules/lineProcessing.ts`
- `src/server/game/rules/territoryProcessing.ts`
- `src/client/sandbox/sandboxLines.ts`
- `src/client/sandbox/sandboxLinesEngine.ts`
- `src/client/sandbox/sandboxTerritory.ts`
- `src/client/sandbox/sandboxTerritoryEngine.ts`
- `src/client/sandbox/sandboxCaptures.ts` (merge into CaptureAggregate if needed)
- Deprecated parity test files

---

## 6. Risk Assessment

### 6.1 Risk Matrix

| Risk                                    | Probability | Impact | Mitigation                                                       |
| --------------------------------------- | ----------- | ------ | ---------------------------------------------------------------- |
| **Parity regression during migration**  | Medium      | High   | Keep existing tests green throughout, no-flag migration approach |
| **Performance degradation**             | Low         | Medium | Benchmark before/after each phase, maintain latency SLOs         |
| **Python contract drift**               | Medium      | Medium | Automated vector regeneration in CI, nightly contract runs       |
| **Backend/sandbox behavior divergence** | Low         | High   | Single adapter base class, comprehensive cross-adapter tests     |
| **Test infrastructure breakage**        | Medium      | Medium | Phase migrations behind feature flags, rollback capability       |

### 6.2 Rollback Strategy

Each phase includes rollback capability:

1. **Phase 1 Rollback:** New orchestrator can be bypassed; old direct aggregate calls remain
2. **Phase 2 Rollback:** Feature flag `RINGRIFT_USE_NEW_ADAPTER=0` falls back to legacy
3. **Phase 3 Rollback:** Sandbox adapter flag independent of backend
4. **Phase 4 Rollback:** Shadow contracts can be re-enabled if contracts insufficient
5. **Phase 5 Rollback:** N/A (cleanup only after all phases stable)

### 6.3 Success Metrics

| Metric                 | Current           | Target                        | Measurement                   |
| ---------------------- | ----------------- | ----------------------------- | ----------------------------- |
| Total rules engine LOC | ~18,000           | ~12,000                       | `cloc` on rules-related files |
| Duplicate logic count  | 3 implementations | 1 implementation + 2 adapters | Manual audit                  |
| Test execution time    | ~180s             | ~120s                         | CI timing                     |
| Parity test count      | ~150              | ~50 (contract tests)          | Test file count               |
| Python shadow latency  | ~50ms/move        | 0 (compile-time)              | Runtime metrics               |

---

## 7. Invariants to Preserve

### 7.1 Board Cell Exclusivity

Each board cell is exclusive: empty OR stack OR marker OR collapsed. Never combinations.

```typescript
// Invariant check - must hold after every mutation
function assertBoardInvariants(board: BoardState): void {
  for (const key of board.stacks.keys()) {
    if (board.collapsedSpaces.has(key)) {
      throw new Error(`Stack on collapsed space at ${key}`);
    }
    if (board.markers.has(key)) {
      throw new Error(`Stack and marker coexist at ${key}`);
    }
  }
  for (const key of board.markers.keys()) {
    if (board.collapsedSpaces.has(key)) {
      throw new Error(`Marker on collapsed space at ${key}`);
    }
  }
}
```

### 7.2 S-Invariant Non-Decrease

```
S = markers + collapsedSpaces + eliminatedRings
```

S must be non-decreasing over the course of a game.

### 7.3 Q23: Self-Elimination Prerequisite

A player may only process a territory region they control if they have at least one stack/cap outside that region.

### 7.4 Victory Condition Order

1. Ring elimination (priority 1, any player)
2. Last player standing (priority 2)
3. Territory control (priority 3, on game completion)

---

## 8. Questions Requiring Clarification

1. **AI Latency SLO:** What is the acceptable latency for AI move computation? This affects whether Python can call back to TS engine or must have local rules.

2. **Shadow Contract Deprecation:** Can we fully remove runtime shadow validation, or should we keep it for production debugging?

3. **Test Vector Scope:** Should test vectors cover all edge cases discovered in production, or just canonical scenarios from the rules spec?

4. **Feature Flag Strategy:** Should the new adapter be opt-in (flag enables new code) or opt-out (flag disables new code)?

5. **Documentation Audience:** Should `CANONICAL_ENGINE_API.md` target internal developers only, or also third-party integrators?

---

## 9. Appendix: File Change Summary

### New Files

| File                                                   | Purpose                      |
| ------------------------------------------------------ | ---------------------------- |
| `src/shared/engine/orchestration/turnOrchestrator.ts`  | `processTurn` implementation |
| `src/shared/engine/orchestration/phaseStateMachine.ts` | Phase transition logic       |
| `src/shared/engine/orchestration/decisionRouter.ts`    | Decision routing             |
| `src/shared/engine/contracts/schemas/*.json`           | JSON Schema definitions      |
| `src/shared/engine/contracts/generate-vectors.ts`      | Test vector generator        |
| `src/server/game/adapters/EngineAdapter.ts`            | Server adapter               |
| `src/client/sandbox/adapters/SandboxEngineAdapter.ts`  | Sandbox adapter              |
| `ai-service/tests/parity/test_contract_parity.py`      | Contract tests               |

### Modified Files

| File                                        | Change Type         |
| ------------------------------------------- | ------------------- |
| `src/shared/engine/index.ts`                | Add new exports     |
| `src/server/game/GameEngine.ts`             | Delegate to adapter |
| `src/server/game/RulesBackendFacade.ts`     | Simplify modes      |
| `src/client/sandbox/ClientSandboxEngine.ts` | Delegate to adapter |
| `ai-service/app/game_engine.py`             | Remove shadow code  |

### Removed Files (Phase 5)

| File                                           | Reason                         |
| ---------------------------------------------- | ------------------------------ |
| `src/server/game/rules/lineProcessing.ts`      | Replaced by LineAggregate      |
| `src/server/game/rules/territoryProcessing.ts` | Replaced by TerritoryAggregate |
| `src/client/sandbox/sandboxLines.ts`           | Replaced by LineAggregate      |
| `src/client/sandbox/sandboxLinesEngine.ts`     | Replaced by LineAggregate      |
| `src/client/sandbox/sandboxTerritory.ts`       | Replaced by TerritoryAggregate |
| `src/client/sandbox/sandboxTerritoryEngine.ts` | Replaced by TerritoryAggregate |

---

## 10. Conclusion

This design consolidates RingRift's three rules engine surfaces into a single canonical implementation while preserving behavioral parity and test coverage. The key innovations are:

1. **Thin Adapter Pattern:** Backend and sandbox become orchestration-only layers
2. **Contract-Based Python Parity:** Replace runtime shadow validation with compile-time contracts
3. **Phased Migration:** Each phase is independently deployable and rollback-capable
4. **Comprehensive Test Vectors:** Machine-generated vectors ensure parity across all implementations

**Recommended Implementation Order:**

1. Phase 1 (Foundation) - Highest priority, unblocks all other phases
2. Phase 2 (Backend) - Second priority, reduces maintenance burden
3. Phase 4 (Python) - Can run in parallel with Phase 2
4. Phase 3 (Sandbox) - Can run in parallel with Phase 4
5. Phase 5 (Cleanup) - Final polish

**Total Estimated Effort:** 8-10 weeks for full migration

---

_Document generated for Priority #1: Rules Engine Consolidation_
_Review date: 2025-11-26_
