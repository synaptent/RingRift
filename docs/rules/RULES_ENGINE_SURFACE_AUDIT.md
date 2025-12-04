# Rules Engine Surface Audit

> **SSoT alignment:** This document is an audit/diagnostic view over the rules engine surfaces. It defers to:
>
> - **Rules semantics SSoT:** `RULES_CANONICAL_SPEC.md`, `ringrift_complete_rules.md` / `ringrift_compact_rules.md`, and the shared TypeScript rules engine under `src/shared/engine/**` (helpers → aggregates → turn orchestrator → contracts plus v2 contract vectors in `tests/fixtures/contract-vectors/v2/**`). This is the **Rules/invariants semantics SSoT** for RingRift.
> - **Lifecycle/API SSoT:** `docs/CANONICAL_ENGINE_API.md` and the shared TS/WebSocket types (`src/shared/types/game.ts`, `src/shared/engine/orchestration/types.ts`, `src/shared/types/websocket.ts`, `src/shared/validation/websocketSchemas.ts`) for the executable Move + orchestrator + WebSocket lifecycle.
> - **Precedence:** Backend (`GameEngine`, `RuleEngine`, `BoardManager`, `TurnEngineAdapter`), client sandbox (`ClientSandboxEngine`, `SandboxOrchestratorAdapter`), and Python rules engine (`ai-service/app/game_engine.py`, `ai-service/app/rules/*`) are **hosts/adapters** over those SSoTs. If this audit ever conflicts with the shared TS engine, orchestrator/contracts, WebSocket schemas, or tests, **code + tests win** and this document must be updated to match.
>
> This file inventories and critiques surfaces around the canonical TS rules engine; it is not itself a semantics SSoT.

**Doc Status (2025-11-26): Active (with historical/diagnostic analysis)**

- Canonical rules semantics SSoT is the **shared TypeScript engine** under `src/shared/engine/`, specifically: helpers → domain aggregates → turn orchestrator → contracts (`schemas.ts`, `serialization.ts`, `testVectorGenerator.ts` + v2 vectors under `tests/fixtures/contract-vectors/v2/`).
- Move/decision/WebSocket lifecycle semantics are documented in `docs/CANONICAL_ENGINE_API.md` and the shared TS/WebSocket types (`src/shared/types/game.ts`, `src/shared/engine/orchestration/types.ts`, `src/shared/types/websocket.ts`, `src/shared/validation/websocketSchemas.ts`).
- Backend (`GameEngine`, `RuleEngine`, `BoardManager`, `TurnEngineAdapter`), client sandbox (`ClientSandboxEngine`, `SandboxOrchestratorAdapter`), and Python rules engine (`ai-service/app/game_engine.py`, `ai-service/app/rules/*`) are **hosts/adapters** over this SSoT. This audit treats them as consumers of the shared engine, not as independent rules engines.
- Sections that describe a fully-populated TS `validators/*` / `mutators/*` tree should be read as **semantic boundary diagrams** and partially historical; the implemented canonical surface is helpers + aggregates + orchestrator + contracts.

**Task:** T1-W1-A  
**Date:** 2025-11-26  
**Status:** Complete

## Executive Summary

This audit examines the four rules engine surfaces identified in `ARCHITECTURE_REMEDIATION_PLAN.md` Weakness 1:

1. **Shared Engine** (`src/shared/engine/`) - Canonical rules logic
2. **Server Game** (`src/server/game/`) - Backend orchestration with player interaction
3. **Python AI Service** (`ai-service/app/rules/`) - Python port for AI training/evaluation
4. **Client Sandbox** (`src/client/sandbox/`) - Client-side game orchestration

**Key Finding:** The codebase is actively transitioning toward a single canonical engine in `src/shared/engine/`. Both Server Game and Client Sandbox surfaces now extensively delegate to shared helpers, with explicit documentation stating they are "orchestration layers" only. The Python AI Service duplicates core logic but maintains parity through shadow contract validation.

---

## 0. Rules Entry Surfaces (SSoT Checklist)

This section names the **only modules that are allowed to encode rules semantics** versus modules that must treat the shared engine as an external contract. New rules logic should not be added outside the **Allowed to encode rules semantics** column without first updating this table and the canonical specs.

### 0.1 Module‑level checklist

| Area               | Module / Directory                                                                  | Role                                      | Allowed to encode rules semantics? | Must call / depend on                          | Must **not** do                                                               |
| ------------------ | ----------------------------------------------------------------------------------- | ----------------------------------------- | ---------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------- |
| **Shared Engine**  | `src/shared/engine/validators/*.ts`                                                 | Low‑level move validation                 | ✅ Yes (canonical)                 | Shared helpers (geometry, core.ts)             | Depend on server/client/Python hosts, or on transport/websocket details       |
| **Shared Engine**  | `src/shared/engine/aggregates/**`                                                   | Capture/territory/line/victory aggregates | ✅ Yes (canonical)                 | Validators, `core.ts`, `types/game.ts`         | Reach into server/client state directly; emit WebSocket payloads              |
| **Shared Engine**  | `src/shared/engine/orchestration/**` (turn orchestrator, phase state machines)      | Turn/phase orchestration                  | ✅ Yes (canonical)                 | Aggregates, validators, shared types           | Talk to sockets/HTTP directly; apply host‑specific timeouts or retries        |
| **Shared Types**   | `src/shared/types/game.ts` (`Move`, `GameState`, `GameHistoryEntry`, guards)        | Canonical Move / GameState / history      | ✅ Yes (canonical model)           | N/A (pure type/shape SSoT)                     | Encode host‑specific concerns (HTTP, DB, AI)                                  |
| **Server Game**    | `src/server/game/RuleEngine.ts`                                                     | Host adapter around shared engine         | ⚠️ Only glue logic                 | `src/shared/engine/**`, `BoardManager`         | Introduce new rules branches not present in shared engine                     |
| **Server Game**    | `src/server/game/GameEngine.ts`                                                     | Backend orchestrator over shared engine   | ⚠️ Only glue + host decisions      | RuleEngine, TurnEngineAdapter, shared types    | Re‑implement capture/territory/line/victory logic instead of using aggregates |
| **Server Game**    | `src/server/game/turn/TurnEngine.ts` / `TurnEngineAdapter.ts`                       | Backend turn engine façade                | ⚠️ Only integration logic          | Shared turn orchestrator + state machines      | Diverge phase/turn semantics from shared `turnLogic`                          |
| **Client Sandbox** | `src/client/sandbox/ClientSandboxEngine.ts`                                         | Client host over shared engine            | ⚠️ Only glue + UI orchestration    | Shared engine (`src/shared/engine/**`)         | Encode new rules; change legal moves independent of shared engine             |
| **Client Sandbox** | `src/client/sandbox/SandboxOrchestratorAdapter.ts`                                  | Sandbox turn orchestration adapter        | ⚠️ Only integration logic          | Shared orchestrator + board helpers            | Drift from backend turn/phase semantics                                       |
| **Python Host**    | `ai-service/app/game_engine.py`                                                     | Python host adapter for rules             | ⚠️ Transitional                    | Python rules mirror of shared engine types     | Become a second “canonical” rules engine; diverge from shared TS semantics    |
| **Python Rules**   | `ai-service/app/rules/**`                                                           | Python port for training/evaluation       | ⚠️ Only as documented mirror       | Canonical TS specs; contract vectors           | Be used as the source of truth for live game semantics                        |
| **Orchestrators**  | `src/server/game/GameSession.ts`, `src/server/game/ai/**`, state machines           | Session/AI orchestration                  | ❌ No                              | Shared engine + contracts + validation schemas | Decide legal moves or phases directly                                         |
| **UI/Frontend**    | `src/client/components/**`, `src/client/adapters/**`, `src/client/sandbox/**`       | Presentation, view models, local sandbox  | ❌ No                              | Shared types/VMs; sandbox/host façades         | Change rules; filter or mutate legal moves outside orchestrator/validators    |
| **API/Transport**  | `src/server/routes/**`, `src/shared/validation/**`, `src/shared/types/websocket.ts` | HTTP/WebSocket contracts                  | ❌ No                              | Shared types; schemas; backend orchestrator    | Implement rules; modify GameState invariants directly                         |

### 0.2 Change checklist

Before introducing or changing rules behaviour:

1. **Locate the correct SSoT:**
   - If it is a rules semantic (what moves are legal, how phases advance, how victory is decided), change **only**:
     - Shared validators/aggregates/orchestrator, and
     - The canonical rules docs (`RULES_CANONICAL_SPEC.md`, `ringrift_complete_rules.md`, `CANONICAL_ENGINE_API.md`).
2. **Verify hosts are adapters only:**
   - For any change in the shared engine, audit:
     - `src/server/game/GameEngine.ts` / `RuleEngine.ts`
     - `src/client/sandbox/ClientSandboxEngine.ts` / `SandboxOrchestratorAdapter.ts`
     - `ai-service/app/game_engine.py` / `ai-service/app/rules/**`
   - Ensure they **call into** the updated shared functions instead of adding new branching rules locally.
3. **Update contract tests:**
   - Add or update contract vectors / parity tests using:
     - `Move`, `GameTrace`, `GameHistoryEntry` from `src/shared/types/game.ts`.
   - Keep these tests **host‑agnostic**: they should not rely on server/client/Python specifics beyond calling hosts via their façades.
4. **Document the surface:**
   - If a new helper or aggregate becomes part of the canonical engine surface, add it to the appropriate table in this document and, if relevant, to `SHARED_ENGINE_CONSOLIDATION_PLAN.md`.
5. **Avoid new “shadow” engines:**
   - Do not introduce new rules engines under `src/server/**`, `src/client/**`, or `ai-service/**` that re‑implement move legality or phase transitions. If you need a new adapter, express it as a thin wrapper over the shared engine.

> **Rule of thumb:** if a module needs to ask “is this move legal?” or “what phase comes next?”, it should go through a shared validator/aggregate/orchestrator function, not compute that answer itself.

---

## 1. Surface Inventory Table

| Surface               | Location                | File Count | Approx. Line Count | Key Entry Points                                                                                                                  | Primary Responsibility                                      |
| --------------------- | ----------------------- | ---------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **Shared Engine**     | `src/shared/engine/`    | 51 files   | ~8,000             | `turnLogic.ts`, `orchestration/` (turnOrchestrator.ts, phaseStateMachine.ts, types.ts), `aggregates/`, `validators/`, `mutators/` | Canonical rules definitions, pure functions                 |
| **Server Game**       | `src/server/game/`      | 12 files   | ~6,000             | `GameEngine.ts`, `RuleEngine.ts`, `RulesBackendFacade.ts`                                                                         | Backend orchestration, WebSocket interaction, parity bridge |
| **Python AI Service** | `ai-service/app/rules/` | 14 files   | ~2,000             | `default_engine.py`, validators/, mutators/                                                                                       | Python port for AI training, parity validation              |
| **Client Sandbox**    | `src/client/sandbox/`   | 18 files   | ~7,500             | `ClientSandboxEngine.ts`, `SandboxOrchestratorAdapter.ts` (legacy `sandbox*Engine.ts` now removed)                                | Client-side game hosting, local AI                          |

---

## 2. Surface Analysis

### 2.1 Surface 1: Shared Engine (`src/shared/engine/`)

#### Responsibilities

- **Game Phases:** Ring placement, movement, capture, chain capture, line processing, territory processing
- **State Transitions:** Turn advancement via `advanceTurnAndPhase()`, phase normalization
- **Invariants:** Board exclusivity (stack/marker/collapsed mutual exclusion), cap height computation, victory thresholds

#### Key Files and Functions

| File                                                                              | Key Exports                                                                                                                                                                                       | Purpose                                        |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| [`core.ts`](../src/shared/engine/core.ts)                                         | `calculateCapHeight`, `getPathPositions`, `validateCaptureSegmentOnBoard`, `hashGameState`                                                                                                        | Core utilities, geometry, state hashing        |
| [`turnLogic.ts`](../src/shared/engine/turnLogic.ts)                               | `advanceTurnAndPhase`, `PerTurnState`                                                                                                                                                             | Turn/phase state machine                       |
| [`orchestration/`](../src/shared/engine/orchestration/)                           | `processTurn` / `processTurnAsync`, `validateMove`, `getValidMoves`, `hasValidMoves`, `ProcessTurnResult`, `PendingDecision`, `VictoryState`                                                      | Canonical turn orchestrator & decision surface |
| [`aggregates/`](../src/shared/engine/aggregates/)                                 | `PlacementAggregate`, `MovementAggregate`, `CaptureAggregate`, `LineAggregate`, `TerritoryAggregate`, `VictoryAggregate`                                                                          | Domain aggregates composed by orchestrator     |
| [`movementLogic.ts`](../src/shared/engine/movementLogic.ts)                       | `enumerateSimpleMoveTargetsFromStack`                                                                                                                                                             | Movement reachability                          |
| [`captureLogic.ts`](../src/shared/engine/captureLogic.ts)                         | `enumerateCaptureMoves`, `CaptureBoardAdapters`                                                                                                                                                   | Capture enumeration                            |
| [`lineDetection.ts`](../src/shared/engine/lineDetection.ts)                       | `findLinesForPlayer`, `findAllLinesShared`                                                                                                                                                        | Line geometry detection                        |
| [`lineDecisionHelpers.ts`](../src/shared/engine/lineDecisionHelpers.ts)           | `enumerateProcessLineMoves`, `applyProcessLineDecision`                                                                                                                                           | Line decision moves                            |
| [`territoryDetection.ts`](../src/shared/engine/territoryDetection.ts)             | `findDisconnectedRegions`                                                                                                                                                                         | Territory region detection                     |
| [`territoryProcessing.ts`](../src/shared/engine/territoryProcessing.ts)           | `applyTerritoryRegion`, `filterProcessableTerritoryRegions`                                                                                                                                       | Territory collapse logic                       |
| [`territoryDecisionHelpers.ts`](../src/shared/engine/territoryDecisionHelpers.ts) | `enumerateProcessTerritoryRegionMoves`, `applyProcessTerritoryRegionDecision`                                                                                                                     | Territory decision moves                       |
| [`victoryLogic.ts`](../src/shared/engine/victoryLogic.ts)                         | `evaluateVictory`                                                                                                                                                                                 | Victory condition evaluation                   |
| `validators/`                                                                     | `PlacementValidator` (board-level + GameState-level placement & skip-placement validation); other move families validate via shared helpers and aggregates rather than separate validator classes | Move validation                                |
| `mutators/`                                                                       | `PlacementMutator`, `MovementMutator`, `CaptureMutator`, `LineMutator`, `TerritoryMutator`                                                                                                        | State mutation used by aggregates              |

#### Public API Pattern

```typescript
// Pure function pattern - takes state + inputs, returns new state
function applyProcessLineDecision(state: GameState, move: Move): LineDecisionApplicationOutcome;
function enumerateCaptureMoves(boardType, from, player, adapters, moveNumber): Move[];
function evaluateVictory(state: GameState): VictoryVerdict;
```

#### Dependencies

- Depends only on `src/shared/types/game.ts` and `src/shared/utils/`
- **Zero dependencies on Server or Client code**

---

### 2.2 Surface 2: Server Game (`src/server/game/`)

#### Responsibilities

- **Game Phases:** All phases (orchestrates shared helpers)
- **State Transitions:** `advanceGame()` wraps shared `advanceTurnAndPhase()`
- **Invariants:** Delegates to shared; adds WebSocket/interaction management

#### Key Files and Functions

| File                                                                             | Key Exports                                  | Purpose                                                                                                                                                  |
| -------------------------------------------------------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`GameEngine.ts`](../src/server/game/GameEngine.ts) (~3,329 lines)               | `GameEngine` class                           | Stateful orchestrator with player interaction                                                                                                            |
| [`RuleEngine.ts`](../src/server/game/RuleEngine.ts) (~1,564 lines)               | `RuleEngine` class                           | Stateless validation, move enumeration                                                                                                                   |
| [`RulesBackendFacade.ts`](../src/server/game/RulesBackendFacade.ts) (~364 lines) | `RulesBackendFacade` class                   | Parity bridge (TS/Python shadow modes)                                                                                                                   |
| [`BoardManager.ts`](../src/server/game/BoardManager.ts) (~1,283 lines)           | `BoardManager` class                         | Board state CRUD, adjacency                                                                                                                              |
| `rules/captureChainEngine.ts` (legacy; historical, file removed)                 | `updateChainCaptureStateAfterCapture`        | Historical backend chain-capture state tracking module; superseded by `GameEngine` + shared `captureChainHelpers` and kept here only as a naming anchor. |
| `rules/lineProcessing.ts` (legacy; historical, file removed)                     | `processLinesForCurrentPlayer`               | Historical backend line-processing orchestration; modern flows use shared `lineDecisionHelpers` + aggregates and backend adapters.                       |
| `rules/territoryProcessing.ts` (legacy; historical, file removed)                | `processDisconnectedRegionsForCurrentPlayer` | Historical backend territory-processing orchestration; modern flows use shared `territoryDetection`/`territoryProcessing`/`territoryDecisionHelpers`.    |
| [`turn/TurnEngine.ts`](../src/server/game/turn/TurnEngine.ts)                    | `advanceGameForCurrentPlayer`                | Turn orchestration                                                                                                                                       |

#### Delegation Pattern to Shared Engine

**Historical documentation from legacy `territoryProcessing.ts` (server `rules/territoryProcessing.ts`, since removed; retained here as a historical note):**

```typescript
/**
 * Legacy backend territory processing module.
 * **IMPORTANT:** This file exists primarily to provide the backend-specific
 * orchestration layer (player interaction, GameState updates) around the
 * canonical shared territory helpers. The actual territory semantics are defined in:
 * - shared/engine/territoryDetection.ts - Region detection
 * - shared/engine/territoryProcessing.ts - Collapse logic
 * - shared/engine/territoryBorders.ts - Border markers
 * - shared/engine/territoryDecisionHelpers.ts - Decision helpers
 * Do NOT add new territory processing logic here; extend the shared helpers instead.
 */
```

**Server GameEngine imports from shared:**

```typescript
import {
  filterProcessableTerritoryRegions,
  applyTerritoryRegion,
} from '../../../shared/engine/territoryProcessing';
import {
  enumerateProcessTerritoryRegionMoves,
  applyProcessTerritoryRegionDecision,
} from '../../../shared/engine/territoryDecisionHelpers';
import {
  enumerateProcessLineMoves,
  applyProcessLineDecision,
} from '../../../shared/engine/lineDecisionHelpers';
import { findLinesForPlayer } from '../../../shared/engine/lineDetection';
import { evaluateVictory } from '../../../shared/engine/victoryLogic';
import { enumerateCaptureMoves } from '../../../shared/engine/captureLogic';
import { enumerateSimpleMoveTargetsFromStack } from '../../../shared/engine/movementLogic';
import { validatePlacementOnBoard } from '../../../shared/engine/validators/PlacementValidator';
```

#### RulesBackendFacade Modes

| Mode           | Behavior                                |
| -------------- | --------------------------------------- |
| `ts` (default) | TypeScript GameEngine is authoritative  |
| `shadow`       | TS authoritative + Python parity checks |
| `python`       | Python validation with TS fallback      |

---

### 2.3 Surface 3: Python AI Service (`ai-service/app/rules/`)

#### Responsibilities

- **Game Phases:** All phases (Python port of TS logic)
- **State Transitions:** Mirrors TS `GameEngine.apply_move()`
- **Invariants:** Shadow contract validation against canonical engine

#### Key Files and Functions

| File                                                                          | Key Exports                                                                                               | Purpose                            |
| ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| [`core.py`](../ai-service/app/rules/core.py) (~245 lines)                     | `calculate_cap_height`, `get_path_positions`, `BoardView`                                                 | Python port of TS core utilities   |
| [`default_engine.py`](../ai-service/app/rules/default_engine.py) (~994 lines) | `DefaultRulesEngine` class                                                                                | Orchestrator with shadow contracts |
| [`factory.py`](../ai-service/app/rules/factory.py)                            | `get_rules_engine()`                                                                                      | Singleton factory                  |
| [`geometry.py`](../ai-service/app/rules/geometry.py) (~190 lines)             | `BoardGeometry` class                                                                                     | Board geometry helpers             |
| [`interfaces.py`](../ai-service/app/rules/interfaces.py)                      | `Validator`, `Mutator`, `RulesEngine` protocols                                                           | Type contracts                     |
| `validators/`                                                                 | `PlacementValidator`, `MovementValidator`, `CaptureValidator`, `LineValidator`, `TerritoryValidator`      | Move validation                    |
| `mutators/`                                                                   | `PlacementMutator`, `MovementMutator`, `CaptureMutator`, `LineMutator`, `TerritoryMutator`, `TurnMutator` | State mutation                     |

#### Shadow Contract Pattern

**From [`default_engine.py`](../ai-service/app/rules/default_engine.py:141-160):**

```python
def apply_move(self, state: GameState, move: Move) -> GameState:
    # Canonical result: always computed via GameEngine.apply_move
    next_via_engine = GameEngine.apply_move(state, move)

    # Per-move mutator shadow contracts (board + players only)
    if move.type == MoveType.PLACE_RING:
        mutator_state = state.model_copy(deep=True)
        PlacementMutator().apply(mutator_state, move)
        # Assert board/player parity...

    # Always return canonical GameEngine state
    return next_via_engine
```

**All mutators delegate to Python GameEngine static methods:**

```python
class PlacementMutator(Mutator):
    def apply(self, state: GameState, move: Move) -> None:
        from app.game_engine import GameEngine
        GameEngine._apply_place_ring(state, move)
```

#### Duplication Areas

| Python File       | Duplicated From                                 |
| ----------------- | ----------------------------------------------- |
| `core.py`         | `src/shared/engine/core.ts`                     |
| `geometry.py`     | `src/shared/engine/core.ts` (partial)           |
| `validators/*.py` | `src/shared/engine/validators/` (reimplemented) |
| `mutators/*.py`   | `src/shared/engine/mutators/` (thin wrappers)   |

---

### 2.4 Surface 4: Client Sandbox (`src/client/sandbox/`)

#### Responsibilities

- **Game Phases:** All phases (client-side hosting)
- **State Transitions:** Wraps shared turn/phase helpers
- **Invariants:** Mirrors backend via shared delegation

#### Key Files and Functions

| File                                                                                            | Key Exports                                  | Purpose                                                                                                                                          |
| ----------------------------------------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| [`ClientSandboxEngine.ts`](../src/client/sandbox/ClientSandboxEngine.ts) (~2,712 lines)         | `ClientSandboxEngine` class                  | Main orchestrator                                                                                                                                |
| [`SandboxOrchestratorAdapter.ts`](../src/client/sandbox/SandboxOrchestratorAdapter.ts)          | `SandboxOrchestratorAdapter` class           | Adapter wrapping turn orchestrator                                                                                                               |
| ~~[`sandboxTurnEngine.ts`](../src/client/sandbox/sandboxTurnEngine.ts)~~ (~380 lines)           | `advanceTurnAndPhaseForCurrentPlayerSandbox` | **Removed – responsibilities moved to shared `turnLogic.advanceTurnAndPhase` + `ClientSandboxEngine`/`SandboxOrchestratorAdapter` turn helpers** |
| [`sandboxPlacement.ts`](../src/client/sandbox/sandboxPlacement.ts) (~222 lines)                 | `enumerateLegalRingPlacements`               | Placement helpers                                                                                                                                |
| [`sandboxMovement.ts`](../src/client/sandbox/sandboxMovement.ts) (~70 lines)                    | `enumerateSimpleMovementLandings`            | Movement helpers                                                                                                                                 |
| ~~[`sandboxMovementEngine.ts`](../src/client/sandbox/sandboxMovementEngine.ts)~~                | (removed)                                    | (movement orchestration now in `ClientSandboxEngine`)                                                                                            |
| [`sandboxCaptures.ts`](../src/client/sandbox/sandboxCaptures.ts) (~173 lines)                   | `enumerateCaptureSegmentsFromBoard`          | Capture helpers                                                                                                                                  |
| [`sandboxCaptureSearch.ts`](../src/client/sandbox/sandboxCaptureSearch.ts)                      | Capture search utilities                     | Chain capture enumeration                                                                                                                        |
| [`sandboxLines.ts`](../src/client/sandbox/sandboxLines.ts) (~134 lines)                         | `findAllLinesOnBoard`                        | Line detection                                                                                                                                   |
| ~~[`sandboxLinesEngine.ts`](../src/client/sandbox/sandboxLinesEngine.ts) (~285 lines)~~         | (removed)                                    | (line decisions now via shared helpers + `ClientSandboxEngine`)                                                                                  |
| [`sandboxTerritory.ts`](../src/client/sandbox/sandboxTerritory.ts) (~599 lines)                 | `findDisconnectedRegionsOnBoard`             | Territory detection                                                                                                                              |
| ~~[`sandboxTerritoryEngine.ts`](../src/client/sandbox/sandboxTerritoryEngine.ts) (~373 lines)~~ | (removed)                                    | (territory decisions now via shared helpers + `ClientSandboxEngine`)                                                                             |
| [`sandboxVictory.ts`](../src/client/sandbox/sandboxVictory.ts) (~120 lines)                     | `checkSandboxVictory`                        | Victory checking                                                                                                                                 |
| [`sandboxElimination.ts`](../src/client/sandbox/sandboxElimination.ts)                          | Elimination helpers                          | Ring elimination processing                                                                                                                      |
| [`sandboxGameEnd.ts`](../src/client/sandbox/sandboxGameEnd.ts)                                  | Game end utilities                           | End-game state handling                                                                                                                          |
| [`sandboxAI.ts`](../src/client/sandbox/sandboxAI.ts)                                            | Local AI helpers                             | Local AI move selection                                                                                                                          |
| [`localSandboxController.ts`](../src/client/sandbox/localSandboxController.ts)                  | `LocalSandboxController`                     | Local game session controller                                                                                                                    |

#### Delegation Pattern to Shared Engine

**Explicit documentation in [`sandboxTerritory.ts`](../src/client/sandbox/sandboxTerritory.ts:1-23):**

```typescript
/**
 * Legacy sandbox territory helpers.
 *
 * **IMPORTANT:** This file exists primarily as a thin adapter layer between
 * the sandbox engine (ClientSandboxEngine) and the canonical shared territory
 * helpers. The actual territory semantics are defined in:
 * - src/shared/engine/territoryDetection.ts - Region detection
 * - src/shared/engine/territoryProcessing.ts - Collapse logic
 * - src/shared/engine/territoryBorders.ts - Border markers
 * - src/shared/engine/territoryDecisionHelpers.ts - Decision helpers
 *
 * Do NOT add new territory processing logic here; extend the shared helpers instead.
 */
```

**Sandbox delegates to shared (current posture):**

```typescript
// sandboxVictory.ts
import { evaluateVictory } from '../../shared/engine/victoryLogic';

// sandboxTerritory.ts
import { findDisconnectedRegions as findDisconnectedRegionsShared } from '../../shared/engine/territoryDetection';
import {
  applyTerritoryRegion,
  canProcessTerritoryRegion,
} from '../../shared/engine/territoryProcessing';

// sandboxLines.ts
import { findAllLines as findAllLinesShared } from '../../shared/engine/lineDetection';

// ClientSandboxEngine.ts (line / territory decision moves)
import {
  enumerateProcessLineMoves,
  applyProcessLineDecision,
  applyChooseLineRewardDecision,
  enumerateProcessTerritoryRegionMoves,
  applyProcessTerritoryRegionDecision,
  applyEliminateRingsFromStackDecision,
} from '../../shared/engine';

// ClientSandboxEngine.ts (turn/phase sequencer)
import { advanceTurnAndPhase } from '../../shared/engine/turnLogic';

// sandboxPlacement.ts
import { validatePlacementOnBoard } from '../../shared/engine/validators/PlacementValidator';

// sandboxMovement.ts
import { enumerateSimpleMoveTargetsFromStack } from '../../shared/engine/movementLogic';

// sandboxCaptures.ts
import { enumerateCaptureMoves } from '../../shared/engine/captureLogic';
```

---

## 3. Overlap Matrix

This matrix shows which rules responsibilities are implemented in each surface:

| Responsibility                | Shared Engine |      Server Game      |      Python AI      | Client Sandbox  |
| ----------------------------- | :-----------: | :-------------------: | :-----------------: | :-------------: |
| **Turn/Phase State Machine**  | ✅ Canonical  |     ➡️ Delegates      |    🔄 Duplicates    |  ➡️ Delegates   |
| **Ring Placement Validation** | ✅ Canonical  |     ➡️ Delegates      |    🔄 Duplicates    |  ➡️ Delegates   |
| **Movement Reachability**     | ✅ Canonical  |     ➡️ Delegates      |    🔄 Duplicates    |  ➡️ Delegates   |
| **Capture Enumeration**       | ✅ Canonical  |     ➡️ Delegates      |    🔄 Duplicates    |  ➡️ Delegates   |
| **Line Detection**            | ✅ Canonical  |     ➡️ Delegates      |    🔄 Duplicates    |  ➡️ Delegates   |
| **Line Decision Moves**       | ✅ Canonical  |     ➡️ Delegates      |    🔄 Duplicates    |  ➡️ Delegates   |
| **Territory Detection**       | ✅ Canonical  |     ➡️ Delegates      |    🔄 Duplicates    |  ➡️ Delegates   |
| **Territory Processing**      | ✅ Canonical  |     ➡️ Delegates      |    🔄 Duplicates    |  ➡️ Delegates   |
| **Territory Decision Moves**  | ✅ Canonical  |     ➡️ Delegates      |    🔄 Duplicates    |  ➡️ Delegates   |
| **Victory Evaluation**        | ✅ Canonical  |     ➡️ Delegates      |    🔄 Duplicates    |  ➡️ Delegates   |
| **Cap Height Computation**    | ✅ Canonical  |        ➡️ Uses        |    🔄 Duplicates    |     ➡️ Uses     |
| **Board Geometry**            | ✅ Canonical  |        ➡️ Uses        |    🔄 Duplicates    |     ➡️ Uses     |
| **State Hashing**             | ✅ Canonical  |        ➡️ Uses        |    🔄 Duplicates    |     ➡️ Uses     |
| **Player Interaction**        |    ❌ None    |        ✅ Owns        |       ❌ None       | ✅ Owns (local) |
| **WebSocket/Persistence**     |    ❌ None    |        ✅ Owns        |       ❌ None       |     ❌ None     |
| **AI Move Selection**         |    ❌ None    |     ❌ Delegates      |       ✅ Owns       | ✅ Owns (local) |
| **Parity Validation**         |    ❌ None    | ✅ RulesBackendFacade | ✅ Shadow contracts |     ❌ None     |

**Legend:**

- ✅ Canonical: Source of truth for this responsibility
- ➡️ Delegates: Calls shared engine helpers
- 🔄 Duplicates: Reimplements logic (parity risk)
- ✅ Owns: Unique to this surface

---

## 4. Dependency Graph

```mermaid
graph TB
    subgraph "Shared Engine (Canonical)"
        SE_Core[core.ts]
        SE_Types[types.ts]
        SE_Turn[turnLogic.ts]
        SE_Movement[movementLogic.ts]
        SE_Capture[captureLogic.ts]
        SE_Lines[lineDetection.ts + lineDecisionHelpers.ts]
        SE_Territory[territoryDetection.ts + territoryProcessing.ts + territoryDecisionHelpers.ts]
        SE_Victory[victoryLogic.ts]
        SE_Validators[validators/]
        SE_Mutators[mutators/]
        SE_Orchestrator[orchestration/turnOrchestrator.ts + aggregates/]
    end

    subgraph "Server Game (Backend Orchestrator)"
        SG_GameEngine[GameEngine.ts]
        SG_RuleEngine[RuleEngine.ts]
        SG_Facade[RulesBackendFacade.ts]
        SG_BoardManager[BoardManager.ts]
        SG_TurnEngine[TurnEngine.ts]
        SG_LineProcessing[lineProcessing.ts (legacy; removed)]
        SG_TerritoryProcessing[territoryProcessing.ts (legacy; removed)]
        SG_CaptureChain[captureChainEngine.ts (legacy; removed)]
    end

    subgraph "Python AI Service (Duplicated Port)"
        PY_Core[core.py]
        PY_Geometry[geometry.py]
        PY_Engine[default_engine.py]
        PY_GameEngine[game_engine.py]
        PY_Validators[validators/]
        PY_Mutators[mutators/]
    end

    subgraph "Client Sandbox (Client Orchestrator)"
        CS_Engine[ClientSandboxEngine.ts]
        CS_Placement[sandboxPlacement.ts]
        CS_Movement[sandboxMovement.ts]
        CS_Captures[sandboxCaptures.ts]
        CS_Lines[sandboxLines.ts]
        CS_Territory[sandboxTerritory.ts]
        CS_Victory[sandboxVictory.ts]
        %% Historical sandbox turn engine (now removed; responsibilities live in ClientSandboxEngine + shared turnLogic)
        %% CS_Turn[sandboxTurnEngine.ts (legacy; removed)]
    end

    %% Server Game delegates to Shared
    SG_GameEngine --> SE_Turn
    SG_GameEngine --> SE_Lines
    SG_GameEngine --> SE_Territory
    SG_GameEngine --> SE_Victory
    SG_GameEngine --> SE_Capture
    SG_GameEngine --> SE_Movement
    SG_RuleEngine --> SE_Validators
    SG_RuleEngine --> SE_Capture
    SG_RuleEngine --> SE_Movement
    SG_TurnEngine --> SE_Turn
    SG_LineProcessing --> SE_Lines
    SG_TerritoryProcessing --> SE_Territory
    SG_BoardManager --> SE_Core

    %% Client Sandbox delegates to Shared
    CS_Turn --> SE_Turn
    CS_Placement --> SE_Validators
    CS_Movement --> SE_Movement
    CS_Captures --> SE_Capture
    CS_Lines --> SE_Lines
    CS_Territory --> SE_Territory
    CS_Victory --> SE_Victory
    CS_Engine --> SE_Core

    %% Python duplicates Shared (parity risk)
    PY_Core -.-> SE_Core
    PY_Geometry -.-> SE_Core
    PY_Validators -.-> SE_Validators
    PY_Mutators -.-> SE_Mutators
    PY_Engine -.-> SE_Orchestrator

    %% Parity bridge
    SG_Facade --> PY_Engine

    style SE_Core fill:#90EE90
    style SE_Turn fill:#90EE90
    style SE_Lines fill:#90EE90
    style SE_Territory fill:#90EE90
    style SE_Victory fill:#90EE90
    style PY_Core fill:#FFB6C1
    style PY_Geometry fill:#FFB6C1
    style PY_Validators fill:#FFB6C1
    style PY_Mutators fill:#FFB6C1
```

**Legend:**

- 🟢 Green: Canonical source of truth
- 🔴 Pink: Duplicated code (parity risk)
- Solid arrows: Direct dependency/delegation
- Dashed arrows: Duplicated reimplementation

---

## 5. Parity Status

### 5.1 Parity Validation Infrastructure

| Surface            | Mechanism                             | Coverage                 |
| ------------------ | ------------------------------------- | ------------------------ |
| **Server Game**    | `RulesBackendFacade` shadow mode      | Per-move validation      |
| **Python AI**      | `DefaultRulesEngine` shadow contracts | Per-move-type validation |
| **Client Sandbox** | Parity test suite                     | Trace-level replay       |

### 5.2 Known Divergence Risks

| Area                   | Risk                           | Mitigation                                    |
| ---------------------- | ------------------------------ | --------------------------------------------- |
| **Python core.py**     | Manual sync with TS core.ts    | Shadow contracts + parity tests               |
| **Python validators**  | Reimplemented validation logic | Parity test fixtures                          |
| **Python geometry.py** | Partial duplication of core.ts | Some functions duplicated within Python       |
| **LPS tracking**       | Per-host implementation        | Eventually converging on shared state machine |

### 5.3 Authoritative Source

**The Shared Engine (`src/shared/engine/`) is the authoritative source for all rules logic.**

Both Server Game and Client Sandbox documentation explicitly reference shared modules as the canonical source. The Python AI Service uses shadow contracts to continuously validate against `GameEngine.apply_move()`.

---

## 6. Recommended Canonical Boundary

### 6.1 What Should Be in the Canonical Engine

**Core Rules Logic (keep in `src/shared/engine/`):**

| Module                        | Status  | Notes                        |
| ----------------------------- | ------- | ---------------------------- |
| `turnLogic.ts`                | ✅ Keep | Turn/phase state machine     |
| `movementLogic.ts`            | ✅ Keep | Movement reachability        |
| `captureLogic.ts`             | ✅ Keep | Capture enumeration          |
| `lineDetection.ts`            | ✅ Keep | Line geometry                |
| `lineDecisionHelpers.ts`      | ✅ Keep | Line decision moves          |
| `territoryDetection.ts`       | ✅ Keep | Territory region detection   |
| `territoryProcessing.ts`      | ✅ Keep | Territory collapse logic     |
| `territoryDecisionHelpers.ts` | ✅ Keep | Territory decision moves     |
| `victoryLogic.ts`             | ✅ Keep | Victory evaluation           |
| `core.ts`                     | ✅ Keep | Utilities, geometry, hashing |
| `validators/`                 | ✅ Keep | Move validation              |
| `mutators/`                   | ✅ Keep | State mutation               |

**Proposed Additions:**

- `lpsTracker.ts` - Extract LPS tracking from per-host implementations
- `gameStateMachine.ts` - Unified game lifecycle state machine

### 6.2 What Should Be in Adapter Layers

**Keep as orchestration-only:**

| Surface               | Responsibility                                     | Should NOT contain                   |
| --------------------- | -------------------------------------------------- | ------------------------------------ |
| **Server Game**       | WebSocket interaction, player choices, persistence | Rules logic                          |
| **Client Sandbox**    | Local game hosting, UI interaction                 | Rules logic                          |
| **Python AI Service** | AI training/evaluation interface                   | Rules logic (should be thin adapter) |

### 6.3 What Can Be Deprecated/Removed

| File/Directory                                                   | Action             | Rationale                                                                                                                                                       |
| ---------------------------------------------------------------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/client/sandbox/sandboxTerritory.ts` internal helpers        | Remove             | Marked as `@deprecated` DEAD CODE                                                                                                                               |
| `ai-service/app/rules/geometry.py`                               | Merge into core.py | Duplicates core.py functions                                                                                                                                    |
| `ai-service/app/rules/validators/*`                              | Thin adapter only  | Currently reimplements validation                                                                                                                               |
| `src/server/game/rules/lineProcessing.ts` (legacy; removed)      | Historical only    | Historical backend line-processing adapter; removed after consolidation onto shared `lineDecisionHelpers`/aggregates.                                           |
| `src/server/game/rules/territoryProcessing.ts` (legacy; removed) | Historical only    | Historical backend territory-processing adapter; removed after consolidation onto shared `territoryDetection`/`territoryProcessing`/`territoryDecisionHelpers`. |

---

## 7. Recommendations for Remediation

### 7.1 Short-Term (T1 Tasks)

1. **Extract LPS tracking** to shared engine (currently duplicated in Server and Sandbox)
2. **Document canonical boundary** in `RULES_ENGINE_ARCHITECTURE.md`
3. **Add eslint rule** to prevent rules logic in orchestration layers
4. **Clean up deprecated code** in sandboxTerritory.ts

### 7.2 Medium-Term (T2 Tasks)

1. **Python parity layer refactor:**
   - Replace reimplemented validators with thin adapters over TS-generated JSON schema
   - Consider code generation or WASM compilation for perfect parity

2. **Server Game simplification:**
   - Complete migration of remaining orchestration to shared helpers
   - BoardManager should become a pure state container

3. **Client Sandbox consolidation:**
   - Complete the adapter pattern already started
   - Remove all local rules implementations

### 7.3 Long-Term Vision

**Single Canonical Engine Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    Canonical Rules Engine                    │
│                   (src/shared/engine/)                       │
│                                                              │
│  • Pure functions: state → state                            │
│  • Zero I/O, zero side effects                              │
│  • Complete rules semantics                                  │
│  • Generated from single source (TypeScript)                │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   TS Adapter    │  │  Python Adapter │  │   Rust Adapter  │
│  (Server/Web)   │  │  (AI Training)  │  │   (Future)      │
│                 │  │                 │  │                 │
│ • Orchestration │  │ • Generated/    │  │ • WASM target   │
│ • Interaction   │  │   transpiled    │  │ • High-perf     │
│ • Persistence   │  │ • Parity tests  │  │   inference     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## 8. Audit Appendix

### 8.1 File Line Counts

| Surface     | File                                                                              | Lines  |
| ----------- | --------------------------------------------------------------------------------- | ------ |
| **Shared**  | core.ts                                                                           | ~650   |
| **Shared**  | turnLogic.ts                                                                      | ~250   |
| **Shared**  | movementLogic.ts                                                                  | ~200   |
| **Shared**  | captureLogic.ts                                                                   | ~350   |
| **Shared**  | lineDetection.ts                                                                  | ~200   |
| **Shared**  | lineDecisionHelpers.ts                                                            | ~400   |
| **Shared**  | territoryDetection.ts                                                             | ~250   |
| **Shared**  | territoryProcessing.ts                                                            | ~300   |
| **Shared**  | territoryDecisionHelpers.ts                                                       | ~400   |
| **Shared**  | victoryLogic.ts                                                                   | ~150   |
| **Server**  | GameEngine.ts                                                                     | ~3,329 |
| **Server**  | RuleEngine.ts                                                                     | ~1,564 |
| **Server**  | BoardManager.ts                                                                   | ~1,283 |
| **Python**  | default_engine.py                                                                 | ~994   |
| **Python**  | core.py                                                                           | ~245   |
| **Sandbox** | ClientSandboxEngine.ts                                                            | ~2,712 |
| **Sandbox** | sandboxTerritory.ts                                                               | ~599   |
| **Sandbox** | sandboxTurnEngine.ts (legacy; removed, historical line count from prior revision) | ~380   |

### 8.2 Import Analysis Summary

**Server GameEngine imports from shared:** 15+ shared modules  
**Client Sandbox imports from shared:** 10+ shared modules  
**Python imports from shared:** 0 (must reimplement)

### 8.3 Test Coverage for Parity

| Test Suite                     | Location                   | Purpose                   |
| ------------------------------ | -------------------------- | ------------------------- |
| `test_rules_parity.py`         | `ai-service/tests/parity/` | Python vs TS parity       |
| `RulesMatrix.*.test.ts`        | `tests/scenarios/`         | Cross-engine scenarios    |
| `Backend_vs_Sandbox.*.test.ts` | `tests/unit/`              | Backend vs Sandbox parity |
| `TraceParity.*.test.ts`        | `tests/unit/`              | Move-by-move trace parity |

---

## Conclusion

The codebase has made significant progress toward a single canonical rules engine in `src/shared/engine/`. Both Server Game and Client Sandbox surfaces now explicitly document themselves as "orchestration layers" that delegate to shared helpers.

**The primary remaining parity risk is the Python AI Service**, which must reimplement rules logic due to language boundary. The shadow contract system provides runtime parity validation, but a more sustainable long-term solution would be code generation or WASM compilation.

**Next Steps:**

1. Complete T1-W1-B: Define narrow stable API boundary in shared engine
2. Complete T1-W1-C: Create adapter layer specifications
3. Address Python parity layer in T2 phase
