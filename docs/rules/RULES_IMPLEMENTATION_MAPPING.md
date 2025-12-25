# RingRift Rules Implementation Mapping

> **SSoT alignment:** This document is a derived mapping between canonical rules and implementation. It defers to:
>
> - **Rules/invariants semantics SSoT:** `../../RULES_CANONICAL_SPEC.md` (RR‑CANON rules), COMPLETE_RULES.md` / COMPACT_RULES.md`, and the shared TypeScript rules engine under `src/shared/engine/**` (helpers → aggregates → orchestrator → contracts plus v2 contract vectors in `tests/fixtures/contract-vectors/v2/**`).
> - **Lifecycle/API SSoT:** `../architecture/CANONICAL_ENGINE_API.md` and the shared TS/WebSocket types (`src/shared/types/game.ts`, `src/shared/engine/orchestration/types.ts`, `src/shared/types/websocket.ts`, `src/shared/validation/websocketSchemas.ts`) for the executable Move + decision + WebSocket lifecycle.
> - **Precedence:** Backend (`GameEngine` + `TurnEngineAdapter` over the shared orchestrator, with `BoardManager`), client sandbox (`ClientSandboxEngine` + `SandboxOrchestratorAdapter`), and Python rules/AI engine (`ai-service/app/game_engine/__init__.py`, `ai-service/app/rules/*`) are **hosts/adapters** over those SSoTs. Legacy backend helpers in `RuleEngine.ts` are treated as **diagnostics-only** orchestration wrappers and must not be considered canonical execution paths. If this document ever contradicts the rules spec, shared TS engine, orchestrator/contracts, WebSocket schemas, or tests, **code + tests win** and this mapping must be updated to match.
>
> This file is for traceability (rules ↔ implementation/tests), not a standalone semantics SSoT.
>
> **Interpretation note:** Validator/Mutator names used here are semantic anchors, not necessarily literal TS filenames; the shared engine expresses many responsibilities via helpers/aggregates/orchestrator.

**Doc Status (2025-12-21): Active**

- Rules/invariants semantics SSoT lives in `../../RULES_CANONICAL_SPEC.md` (RR‑CANON rules) + the shared TypeScript rules engine under `src/shared/engine/` (helpers → aggregates → orchestrator → contracts).
- Move/decision/WebSocket lifecycle semantics SSoT lives in `../architecture/CANONICAL_ENGINE_API.md` and the shared TS/WebSocket types (`src/shared/types/game.ts`, `src/shared/engine/orchestration/types.ts`, `src/shared/types/websocket.ts`, `src/shared/validation/websocketSchemas.ts`).
- Backend (`GameEngine` hosting `TurnEngineAdapter` over the shared orchestrator, with `BoardManager`), client sandbox (`ClientSandboxEngine` + `SandboxOrchestratorAdapter`), and Python rules/AI engine (`ai-service/app/game_engine/__init__.py`, `ai-service/app/rules/*`) are **hosts/adapters** over this SSoT and are validated via shared tests, contract vectors, and parity suites. Legacy backend orchestration helpers in `RuleEngine.ts` are retained for diagnostics/parity only and are not part of the canonical production path.

This document maps the canonical RingRift rules in [`RULES_CANONICAL_SPEC.md`](../../RULES_CANONICAL_SPEC.md) to the current implementation and tests, and provides the inverse view from implementation components back to canonical rules.

Canonical rule IDs of the form `RR-CANON-RXXX` always refer to entries in [`RULES_CANONICAL_SPEC.md`](../../RULES_CANONICAL_SPEC.md). The original narrative rules in [`COMPLETE_RULES.md`]COMPLETE_RULES.md) and the compact spec in [`COMPACT_RULES.md`]COMPACT_RULES.md) are treated as commentary and traceability sources only.

### Quick map (top surfaces)

| Area                     | Primary TS surface                                                                                                     | Key tests                                                                                                                                               |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Turn/phase orchestration | `src/shared/engine/orchestration/turnOrchestrator.ts`, `phaseStateMachine.ts`                                          | `tests/contracts/contractVectorRunner.test.ts`, `tests/unit/TraceFixtures.sharedEngineParity.test.ts`                                                   |
| Lines & territory        | `src/shared/engine/lineDetection.ts`, `lineDecisionHelpers.ts`, `territoryDetection.ts`, `territoryDecisionHelpers.ts` | `tests/unit/GameEngine.lines.scenarios.test.ts`, `tests/unit/BoardManager.territoryDisconnection.test.ts`                                               |
| Forced elimination / ANM | `src/shared/engine/orchestration/turnOrchestrator.ts`, `src/shared/engine/globalActions.ts`                            | `tests/unit/GameEngine.gameEndExplanation.shared.test.ts`, `ai-service/tests/invariants/test_active_no_moves_movement_forced_elimination_regression.py` |

---

## 0. RR-CANON Traceability Index (P0)

This section exists to ensure the most cross-cutting RR-CANON rules have an
explicit “anchor” back to concrete engine surfaces and tests. It is not a full
catalogue.

- **RR-CANON-R073** (mandatory phase transitions): TS `src/shared/engine/orchestration/turnOrchestrator.ts` + `src/shared/engine/orchestration/phaseStateMachine.ts`; Python `ai-service/app/rules/phase_machine.py`.
- **RR-CANON-R074** (record all actions/skips): TS move recording + decision moves in `src/shared/types/game.ts` and orchestrator decisions; Python canonical contract `ai-service/app/rules/history_contract.py` (write-time enforcement via `ai-service/app/db/game_replay.py`).
- **RR-CANON-R075** (canonical replay semantics / no silent transitions): TS `src/shared/engine/phaseValidation.ts` + `tests/unit/PhaseRecording.invariant.test.ts`; Python `ai-service/app/rules/history_contract.py` + `ai-service/app/rules/history_validation.py`.
- **RR-CANON-R076** (core rules vs host layer boundaries): see `../architecture/CANONICAL_ENGINE_API.md` + `src/shared/engine/**` vs hosts (`src/server/game/**`, `src/client/sandbox/**`, `ai-service/app/game_engine/__init__.py`).
- **RR-CANON-R093** (post-movement capture eligibility from landing position only): TS capture enumeration in `src/shared/engine/aggregates/CaptureAggregate.ts`; Python capture enumeration in `ai-service/app/game_engine/__init__.py`.
- **RR-CANON-R110, RR-CANON-R111, RR-CANON-R112, RR-CANON-R113, RR-CANON-R114, RR-CANON-R115** (recovery eligibility, slide, success criteria, extraction, cascade + recording): TS `src/shared/engine/aggregates/RecoveryAggregate.ts` + `src/shared/engine/lpsTracking.ts`; Python `ai-service/app/rules/global_actions.py` + recovery logic in `ai-service/app/game_engine/__init__.py`.
- **RR-CANON-R130** (line reward semantics referenced by recovery): TS `src/shared/engine/aggregates/LineAggregate.ts`; contract vectors under `tests/fixtures/contract-vectors/v2/line_processing.vectors.json`.
- **RR-CANON-R175, RR-CANON-R176, RR-CANON-R177, RR-CANON-R178, RR-CANON-R179** (turn-material / elimination + ranking algorithm): TS elimination + ranking via `src/shared/engine/playerStateHelpers.ts` + victory evaluation; Python equivalents in `ai-service/app/rules/core.py` and `ai-service/app/rules/default_engine.py`.
- **RR-CANON-R208–R209** (multi-phase line→territory sequencing + chain-capture boundaries): TS orchestrator (`src/shared/engine/orchestration/turnOrchestrator.ts`) + FSM adapter (`src/shared/engine/fsm/FSMAdapter.ts`); Python `ai-service/app/rules/phase_machine.py` and FSM orchestration `ai-service/app/rules/fsm.py`.

## 1. Overview: Implementation Landscape

### 1.1 Languages and frameworks

- **TypeScript / Node**
  - Shared rules engine and core game logic in `src/shared/engine/**`.
  - Shared types and configs in [`TypeScript.types`](../../src/shared/types/game.ts:1).
  - Server-side orchestration and WebSocket game backend in `src/server/**`.
  - Client sandbox, UI, and local rules harness in `src/client/**`.
- **Python**
  - Alternative rules engine used for parity and AI in `ai-service/app/**`.
  - Training env and dataset generators in [`Python.env`](../../ai-service/app/training/env.py:1) and siblings.

Frameworks and libraries that materially affect rules behaviour:

- Node/Express backend and WebSocket server (hosting the TS rules engine).
- React front-end for sandbox & UI.
- Jest and custom test harnesses for rules and parity regression tests.
- Prometheus-style metrics for TS↔Python parity in [`TypeScript.rulesParityMetrics`](../../src/server/utils/rulesParityMetrics.ts:12).

**Source of truth for rules logic**

- **Primary** rules semantics live in the **shared TypeScript engine** under `src/shared/engine/**`, especially:
  - Board and marker/ring operations in [`TypeScript.core`](../../src/shared/engine/core.ts:1).
  - Movement and capture in [`TypeScript.movementLogic`](../../src/shared/engine/movementLogic.ts:1) and [`TypeScript.captureLogic`](../../src/shared/engine/captureLogic.ts:1).
  - Lines and rewards in [`TypeScript.lineDetection`](../../src/shared/engine/lineDetection.ts:1) and [`TypeScript.lineDecisionHelpers`](../../src/shared/engine/lineDecisionHelpers.ts:1).
  - Territory disconnection and processing in [`TypeScript.territoryDetection`](../../src/shared/engine/territoryDetection.ts:1), [`TypeScript.territoryProcessing`](../../src/shared/engine/territoryProcessing.ts:1), and [`TypeScript.territoryDecisionHelpers`](../../src/shared/engine/territoryDecisionHelpers.ts:1).
  - Turn/phase and forced elimination in [`TypeScript.turnLogic`](../../src/shared/engine/turnLogic.ts:1).
  - Victory and S‑invariant bookkeeping in [`TypeScript.VictoryAggregate`](../../src/shared/engine/aggregates/VictoryAggregate.ts:45) and [`TypeScript.computeProgressSnapshot()`](../../src/shared/engine/core.ts:531).
- The **server** packages these helpers into a networked game implementation:
  - Stateful game loop in [`TypeScript.GameEngine`](../../src/server/game/GameEngine.ts:92).
  - Stateless rules interface in [`TypeScript.RuleEngine`](../../src/server/game/RuleEngine.ts:46).
  - Board geometry and invariants in [`TypeScript.BoardManager`](../../src/server/game/BoardManager.ts:37).
  - Turn/phase progression and forced elimination in [`TypeScript.advanceGameForCurrentPlayer`](../../src/server/game/turn/TurnEngine.ts:91).
- The **client sandbox** exposes the same semantics locally:
  - Canonical sandbox harness in [`TypeScript.ClientSandboxEngine`](../../src/client/sandbox/ClientSandboxEngine.ts:137).
  - Pure helper engines for movement, capture, lines, territory, turn, and victory in `src/client/sandbox/sandbox*.ts`.
- The **Python rules service** is intended to be semantically equivalent to the TS shared engine, with systematic parity checks and S‑invariant comparison via:
  - [`Python.game_engine`](../../ai-service/app/game_engine/__init__.py:1) (high-level engine).
  - [`TypeScript.PythonRulesClient`](../../src/server/services/PythonRulesClient.ts:33) and [`TypeScript.RulesBackendFacade`](../../src/server/game/RulesBackendFacade.ts:54).

### 1.2 Major subsystems and entry points

- **Server entry & game lifecycle**
  - HTTP/WebSocket bootstrap in [`src/server/index.ts`](../../src/server/index.ts).
  - Game orchestration in [`TypeScript.GameEngine`](../../src/server/game/GameEngine.ts:92) and [`TypeScript.GameSession`](../../src/server/game/GameSession.ts:1).
  - Turn engine and forced elimination in [`TypeScript.advanceGameForCurrentPlayer`](../../src/server/game/turn/TurnEngine.ts:91).
- **Shared core rules / engine**
  - Core board ops and S‑invariant in [`TypeScript.core`](../../src/shared/engine/core.ts:1).
  - Movement, capture, lines, territory, victory, and turn state in the shared engine modules listed above.
- **Client sandbox & rules UI**
  - Sandbox entry at `/sandbox` is driven from [`TypeScript.GamePage`](../../src/client/pages/GamePage.tsx:1) using [`TypeScript.ClientSandboxEngine`](../../src/client/sandbox/ClientSandboxEngine.ts:137).
  - Board interactions and visualisation via [`TypeScript.BoardView`](../../src/client/components/BoardView.tsx:1) and related HUD components.
- **AI / training**
  - Self-play and environment for RL in [`Python.RingRiftEnv`](../../ai-service/app/training/env.py:1).
  - Dataset generators for moves and territory in [`Python.generate_data`](../../ai-service/app/training/generate_data.py:1) and [`Python.generate_territory_dataset`](../../ai-service/app/training/generate_territory_dataset.py:1).
- **Tests & parity harnesses**
  - Shared-engine unit tests (movement, capture, lines, territory, victory) under `tests/unit/**`.
  - Scenario/FAQ matrix tests under `tests/scenarios/**` (e.g. disconnected-region examples and FAQ Q15/Q23).
  - Backend-versus-sandbox parity suites such as [`tests/unit/sandboxLines.test.ts`](../../tests/unit/sandboxLines.test.ts) and [`tests/unit/Seed17Move16And33Parity.GameEngine_vs_Sandbox.test.ts`](../../tests/unit/Seed17Move16And33Parity.GameEngine_vs_Sandbox.test.ts).

### 1.3 Core data models

The core game models live in [`TypeScript.types`](../../src/shared/types/game.ts:1) and directly encode many entity and invariant rules:

- `BoardType`, `BOARD_CONFIGS`, `BoardState`, `GameState`, `GamePhase`, `Move`, `RingStack`, `MarkerInfo`, `Territory`, `LineInfo`.
- Per-player fields `ringsInHand`, `eliminatedRings`, `territorySpaces`.
- Global counters `totalRingsInPlay`, `totalRingsEliminated`, `victoryThreshold`, `territoryVictoryThreshold`.
- Board metadata `size`, `type`, and cached `formedLines`, `eliminatedRings`, `territories` for diagnostics/tests.

These types are used consistently by the shared TS engine, server backend, client sandbox, Python parity service, and AI training code, and thus serve as the canonical concrete schema for many of RR‑CANON rules (R001–R003, R010, R020–R023, R050–R052, R060–R062).

---

## 2. Rule Clusters

For mapping purposes, the RR‑CANON rules are grouped into the following clusters:

1. **Board & entities**
   - Geometry & configs: R001–R003 (including coordinate systems R002).
   - Rings, stacks, control, markers, regions: R020–R023, R030–R031, R040–R041 (explicitly including control/cap-height R022).
   - State validity & resource accounting: R050–R052, R060–R062 (with GameState fields R051).
2. **Turn, phases, and forced elimination**
   - Turn structure and deterministic phase ordering: R070–R072.
   - Forced elimination when blocked: R100 (entry condition).
3. **Placement & skip**
   - Placement mandatory/optional/forbidden and no‑dead‑placement on empty or stacked cells: R080–R082.
4. **Non‑capture movement & marker behaviour**
   - Availability, path length, legal landings, and marker effects: R090–R092, with marker invariants from R030–R031.
5. **Overtaking capture & chains**
   - Forced elimination (R100), single capture segment legality (R101), application semantics (R102), and chain behaviour (R103).
6. **Lines and graduated rewards**
   - Line definition, processing order, and rewards: R120–R122.
7. **Territory disconnection & region processing**
   - Region discovery, physical disconnection, representation, self‑elimination prerequisite, processing order and mechanics: R140–R145.
8. **Victory and termination**
   - Ring‑elimination, territory, last‑player‑standing, global stalemate ladder: R170–R173 (including elimination victory R170, territory victory R171, last-player-standing R172, stalemate and tiebreaks R173).
   - Randomness constraint and S‑invariant: R190–R191.
9. **Global legal actions and ANM invariants**
   - Global legal action enumeration and turn-material predicates: R200–R201.
   - Active-no-moves (ANM) state detection and avoidance: R202–R203.
   - Phase-local decision exits, forced-elimination taxonomy and target choice: R204–R206.
   - Real actions vs forced elimination for LPS victory and termination analysis: R207.

The sections below first map these clusters forward to implementation, then map major implementation components back to RR‑CANON rules.

---

## 3. Forward Mapping: Rules → Implementation

Status legend:

- **HC** = mapped with high confidence (direct algorithmic correspondence).
- **LC** = mapped with low confidence / inferred (behaviour scattered, or partially delegated).
- **NM** = no clear implementation found.

### 3.1 Board & entities (R001–R003, R020–R023, R030–R031, R040–R041, R050–R052, R060–R062)

**R001–R003 Board types, coordinates, adjacency (HC)**

(Explicitly covering R001 board-type configuration, R002 coordinate systems, and R003 adjacency relations.)

- **Primary implementation**
  - Board configs and geometry parameters in [`TypeScript.types`](../../src/shared/types/game.ts:1) (`BoardType`, `BOARD_CONFIGS`).
  - Adjacency and distance helpers in [`TypeScript.core`](../../src/shared/engine/core.ts:1) (`calculateDistance`, `getPathPositions`, movement/line/territory ray-walks).
  - BoardManager neighbour utilities in [`TypeScript.BoardManager`](../../src/server/game/BoardManager.ts:37) (`getHexagonalNeighbors`, `getMooreNeighbors`, `getVonNeumannNeighbors`, `getAdjacentPositions`).
  - Sandbox board-validity helpers in [`TypeScript.ClientSandboxEngine.isValidPosition`](../../src/client/sandbox/ClientSandboxEngine.ts:936) and [`TypeScript.sandboxTerritory.getTerritoryNeighbors`](../../src/client/sandbox/sandboxTerritory.ts:88).
- **Supporting / tests**
  - Movement tests in [`tests/unit/movement.shared.test.ts`](../../tests/unit/movement.shared.test.ts).
  - Line detection tests in [`tests/unit/lineDecisionHelpers.shared.test.ts`](../../tests/unit/lineDecisionHelpers.shared.test.ts) and [`tests/unit/sandboxLines.test.ts`](../../tests/unit/sandboxLines.test.ts).
  - Territory adjacency and region tests in [`tests/unit/territoryDecisionHelpers.shared.test.ts`](../../tests/unit/territoryDecisionHelpers.shared.test.ts) and [`tests/unit/BoardManager.territoryDisconnection.test.ts`](../../tests/unit/BoardManager.territoryDisconnection.test.ts).

**R020–R023 Rings, stacks, control, capHeight, allowed mutations (HC)**

- **Primary implementation**
  - Stack structure and metadata in [`TypeScript.types`](../../src/shared/types/game.ts:1) (`RingStack`, `stackHeight`, `capHeight`, `controllingPlayer`).
  - Cap calculations in [`TypeScript.calculateCapHeight()`](../../src/shared/engine/core.ts:1).
  - Placement and stack building in:
    - Backend placement paths inside [`TypeScript.GameEngine`](../../src/server/game/GameEngine.ts:92) near `applyMove()`.
    - Sandbox placement in [`TypeScript.ClientSandboxEngine.tryPlaceRings`](../../src/client/sandbox/ClientSandboxEngine.ts:1615).
  - Overtaking stack mutation in:
    - Shared capture helpers in [`TypeScript.enumerateCaptureMoves()`](../../src/shared/engine/captureLogic.ts:26).
    - Backend capture application in [`TypeScript.GameEngine`](../../src/server/game/GameEngine.ts:92) near `performOvertakingCapture()`.
    - Sandbox capture in [`TypeScript.sandboxCaptures.applyCaptureSegmentOnBoard`](../../src/client/sandbox/sandboxCaptures.ts:98) and [`TypeScript.ClientSandboxEngine.applyCaptureSegment`](../../src/client/sandbox/ClientSandboxEngine.ts:701).
  - Cap and ring removal operations (forced elimination, line rewards, territory, self‑elimination) in:
    - [`TypeScript.BoardManager`](../../src/server/game/BoardManager.ts:37) and backend helpers.
    - Sandbox elimination helper [`TypeScript.sandboxElimination.forceEliminateCapOnBoard`](../../src/client/sandbox/sandboxElimination.ts:1).
- **Supporting / tests**
  - Stack and capture semantics in [`tests/unit/captureLogic.shared.test.ts`](../../tests/unit/captureLogic.shared.test.ts) and chain-capture tests such as [`tests/unit/GameEngine.chainCapture.test.ts`](../../tests/unit/GameEngine.chainCapture.test.ts).
  - Mixed-color stack and control examples in scenario tests under `tests/scenarios/**`.

**R030–R031 Markers and collapsed spaces (HC)**

- **Primary implementation**
  - Board maps `markers` and `collapsedSpaces` in [`TypeScript.types`](../../src/shared/types/game.ts:1).
  - Marker/territory invariants and exclusivity (no stack+marker, no marker+collapsed) in [`TypeScript.BoardManager`](../../src/server/game/BoardManager.ts:37) near `assertBoardInvariants()`.
  - Marker path behaviour and collapse rules in [`TypeScript.applyMarkerEffectsAlongPathOnBoard()`](../../src/shared/engine/core.ts:619).
  - Sandbox equivalents in:
    - [`TypeScript.ClientSandboxEngine.setMarker`](../../src/client/sandbox/ClientSandboxEngine.ts:964), `flipMarker`, `collapseMarker`.
    - [`TypeScript.ClientSandboxEngine.applyMarkerEffectsAlongPath`](../../src/client/sandbox/ClientSandboxEngine.ts:1089), which in turn delegates to the shared helper; simple movement and capture chains in the sandbox now rely on the shared Movement/Capture aggregates rather than their own marker-path mutators.
- **Supporting / tests**
  - Landing-on-own-marker behaviour in [`tests/unit/ClientSandboxEngine.landingOnOwnMarker.test.ts`](../../tests/unit/ClientSandboxEngine.landingOnOwnMarker.test.ts).
  - Movement tests that exercise marker flips/collapses in [`tests/unit/movement.shared.test.ts`](../../tests/unit/movement.shared.test.ts).

**R040–R041 Regions and territory counts (HC)**

- **Primary implementation**
  - Region abstraction and `Territory` type in [`TypeScript.types`](../../src/shared/types/game.ts:1).
  - Region discovery and metadata in [`TypeScript.territoryDetection.findDisconnectedRegions`](../../src/shared/engine/territoryDetection.ts:36).
  - Territory counters per player (`territorySpaces`) and collapsed-space ownership in `BoardState.collapsedSpaces` and `Player.territorySpaces` in [`TypeScript.types`](../../src/shared/types/game.ts:1).
  - Territory gains in:
    - Line processing helpers (collapsed markers) in [`TypeScript.lineDecisionHelpers`](../../src/shared/engine/lineDecisionHelpers.ts:1).
    - Territory region processing in [`TypeScript.territoryProcessing.applyTerritoryRegion`](../../src/shared/engine/territoryProcessing.ts:172) and sandbox variants.
- **Supporting / tests**
  - Territory region tests in [`tests/unit/territoryDecisionHelpers.shared.test.ts`](../../tests/unit/territoryDecisionHelpers.shared.test.ts).
  - Scenario tests such as [`tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts`](../../tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts) and FAQ_Q15/Q23 territory examples.
  - v2 contract vectors for compact Q23-style mini-regions under `tests/fixtures/contract-vectors/v2/territory.vectors.json`, including single-step forced-elimination and two-step sequences:
    - Square8: `territory.forced_elimination.simple_square8`, `territory.square_region_then_elim.step1_region`, `territory.square_region_then_elim.step2_elim`.
    - Hexagonal: `territory.forced_elimination.simple_hexagonal`, `territory.hex_region_then_elim.step1_region`, `territory.hex_region_then_elim.step2_elim`.
  - TS↔Python parity for these vectors via `tests/contracts/contractVectorRunner.test.ts` and `ai-service/tests/contracts/test_contract_vectors.py`, plus backend↔sandbox parity in `tests/unit/TerritoryDecisions.SquareRegionThenElim.GameEngine_vs_Sandbox.test.ts` and `tests/unit/TerritoryDecisions.HexRegionThenElim.GameEngine_vs_Sandbox.test.ts` (skipped when the orchestrator adapter is enabled).
  - Region-order (Q20) two-region scenario `Rules_12_3_region_order_choice_two_regions_square8` is captured as a multi-step v2 sequence tagged `sequence:square_territory.two_regions_then_elim` (vectors `territory.square_two_regions_then_elim.step1_regionB`, `territory.square_two_regions_then_elim.step2_regionA`, `territory.square_two_regions_then_elim.step3_elim` in `tests/fixtures/contract-vectors/v2/territory.vectors.json`), treated as the TS↔Python SSOT for this scenario via the same contract runners, and mirrored at the host level by `tests/unit/TerritoryDecisions.GameEngine_vs_Sandbox.test.ts` and `tests/unit/TerritoryDecisions.SquareTwoRegionThenElim.GameEngine_vs_Sandbox.test.ts` (diagnostic parity suites skipped under the orchestrator adapter by default).
  - Hexagonal multi-region Q20/Q23-style behaviour is likewise captured as a multi-step v2 sequence tagged `sequence:hex_territory.two_regions_then_elim` (vectors `territory.hex_two_regions_then_elim.step1_regionB`, `territory.hex_two_regions_then_elim.step2_regionA`, `territory.hex_two_regions_then_elim.step3_elim` in `tests/fixtures/contract-vectors/v2/territory.vectors.json`), forming part of the TS↔Python SSOT for advanced hex territory flows via the shared contract runners, and mirrored at the host level by `tests/unit/TerritoryDecisions.HexTwoRegionThenElim.GameEngine_vs_Sandbox.test.ts` (diagnostic parity suite, skipped under the orchestrator adapter by default).

**R050–R052 BoardState/GameState fields and invariants (HC)**

- **Primary implementation**
  - Types in [`TypeScript.types`](../../src/shared/types/game.ts:1).
  - Invariant enforcement in:
    - [`TypeScript.BoardManager`](../../src/server/game/BoardManager.ts:37) via `assertBoardInvariants()`.
    - Sandbox invariant hook [`TypeScript.ClientSandboxEngine.assertBoardInvariants`](../../src/client/sandbox/ClientSandboxEngine.ts:2337).
  - Ring-conservation and S‑invariant checks in:
    - [`TypeScript.computeProgressSnapshot()`](../../src/shared/engine/core.ts:531).
    - History appenders [`TypeScript.GameEngine`](../../src/server/game/GameEngine.ts:92) near `appendHistoryEntry()` and [`TypeScript.ClientSandboxEngine.appendHistoryEntry`](../../src/client/sandbox/ClientSandboxEngine.ts:271).
  - Hash-based state equality in [`TypeScript.hashGameState()`](../../src/shared/engine/core.ts:1) for Python parity and history.
- **Supporting / tests**
  - Invariant tests (board sanity, S‑invariant monotonicity) under `tests/unit/**`, including dedicated invariant tests for territory and forced elimination in `ai-service/tests/invariants/**`.

**R060–R062 Ring-elimination accounting and thresholds (HC)**

- **Primary implementation**
  - `totalRingsInPlay`, `victoryThreshold`, `territoryVictoryThreshold` computed in [`TypeScript.types`](../../src/shared/types/game.ts:1) and initialised in shared initial-state helpers.
  - Elimination crediting in:
    - Line decisions via [`TypeScript.lineDecisionHelpers`](../../src/shared/engine/lineDecisionHelpers.ts:1) and backend/sandbox cap-elimination helpers.
    - Territory region processing via [`TypeScript.territoryProcessing.applyTerritoryRegion`](../../src/shared/engine/territoryProcessing.ts:172).
    - Forced elimination via [`TypeScript.turnLogic`](../../src/shared/engine/turnLogic.ts:135) delegates and [`TypeScript.turn.processForcedElimination`](../../src/server/game/turn/TurnEngine.ts:286).
    - Sandbox forced elimination in [`TypeScript.sandboxElimination.forceEliminateCapOnBoard`](../../src/client/sandbox/sandboxElimination.ts:1).
    - Stalemate conversion of rings in hand in [`TypeScript.VictoryAggregate.evaluateVictory`](../../src/shared/engine/aggregates/VictoryAggregate.ts:45) and Python equivalents.
- **Supporting / tests**
  - Victory and elimination accounting tests in [`tests/unit/GameEngine.victory.scenarios.test.ts`](../../tests/unit/GameEngine.victory.scenarios.test.ts).
  - Stalemate and tiebreaker behaviour in tests under `tests/scenarios/**` and `ai-service/tests/invariants/**`.

### 3.2 Turn, phases, and forced elimination (R070–R072, R100)

**R070–R071 Turn phases and deterministic progression (HC)**

- **Primary implementation**
  - Canonical phase machine in [`TypeScript.turnLogic.advanceTurnAndPhase`](../../src/shared/engine/turnLogic.ts:135).
  - Backend orchestration in [`TypeScript.advanceGameForCurrentPlayer`](../../src/server/game/turn/TurnEngine.ts:91), which wires:
    - Placement availability (`hasValidPlacements`).
    - Movement and capture availability (`hasValidMovements`, `hasValidCaptures`).
    - Forced elimination callback.
    - Next-player computation.
  - Backend move application and automatic consequences (lines, territory, victory, phase/turn advancement) in [`TypeScript.GameEngine`](../../src/server/game/GameEngine.ts:92) (`makeMove()`, `applyDecisionMove()`, `advanceGame()`).
  - Sandbox equivalent phase progression via [`TypeScript.ClientSandboxEngine.startTurnForCurrentPlayer`](../../src/client/sandbox/ClientSandboxEngine.ts:1513), [`TypeScript.ClientSandboxEngine.maybeProcessForcedEliminationForCurrentPlayer`](../../src/client/sandbox/ClientSandboxEngine.ts:1539), and [`TypeScript.ClientSandboxEngine.advanceTurnAndPhaseForCurrentPlayer`](../../src/client/sandbox/ClientSandboxEngine.ts:1603), which delegate to shared `advanceTurnAndPhase` with sandbox-specific delegates.
- **Supporting / tests**
  - Turn/phase sequencing tests under [`tests/unit/RefactoredEngine.test.ts`](../../tests/unit/RefactoredEngine.test.ts) and parity suites that replay canonical move sequences into both backend and sandbox engines.

**R072 Legal-action requirement and forced-elimination entry (HC)**

- **Primary implementation**
  - Turn-logic delegates used in [`TypeScript.advanceGameForCurrentPlayer`](../../src/server/game/turn/TurnEngine.ts:91) and shared [`TypeScript.turnLogic`](../../src/shared/engine/turnLogic.ts:135) to decide when a player has no placements, movements, or captures and must apply forced elimination.
  - Backend blocked-state resolution & stress tests in [`TypeScript.GameEngine.resolveBlockedStateForCurrentPlayerForTesting`](../../src/server/game/GameEngine.ts:2583).
  - Sandbox forced-elimination detection (no placements, no movement, at-or-above per-player cap) in [`TypeScript.ClientSandboxEngine.maybeProcessForcedEliminationForCurrentPlayer`](../../src/client/sandbox/ClientSandboxEngine.ts:1539).
- **Supporting / tests**
  - Regression tests in `ai-service/tests/invariants/test_active_no_moves_movement_forced_elimination_regression.py`.
  - Backend turn/forced-elimination scenarios under `tests/unit/**` and `tests/scenarios/**`.

**R100 Forced elimination semantics (HC, with implementation-specific tie-breaking)**

- **Primary implementation**
  - Shared semantics: eliminate an entire cap from a controlled stack when blocked, crediting all eliminated rings to the acting player. Implemented via:
    - [`TypeScript.turn.processForcedElimination`](../../src/server/game/turn/TurnEngine.ts:286) on the server.
    - Sandbox equivalent in [`TypeScript.ClientSandboxEngine.forceEliminateCap`](../../src/client/sandbox/ClientSandboxEngine.ts:888) and [`TypeScript.sandboxElimination.forceEliminateCapOnBoard`](../../src/client/sandbox/sandboxElimination.ts:1).
  - Implementation tie-breaking (not fixed by RR‑CANON):
    - `processForcedElimination()` chooses an eligible stack to eliminate from, preferring stacks with positive `capHeight` and heuristics about smallest caps. This is an implementation choice; RR‑CANON-R100 only requires that _some_ cap be eliminated.
- **Supporting / tests**
  - Forced-elimination invariants and regression tests under `ai-service/tests/invariants/**`.
  - Backend/sandbox parity tests that ensure both hosts resolve blocked states in the same way.

### 3.3 Placement & skip (R080–R082)

**R080–R082 Placement on empty/stack + no-dead-placement, skip semantics (HC for checks; LC for complete coverage of all textual edge cases, covering R080 mandatory/optional/forbidden placement, R081 empty-cell placement, R082 placement on existing stacks)**

- **Primary implementation**
  - Shared placement validation helper [`TypeScript.validatePlacementOnBoard()`](../../src/shared/engine/validators/PlacementValidator.ts:1) encodes empty/stack placement legality, capacity, and no‑dead‑placement preconditions given a `PlacementContext`.
  - Backend placement validation in [`TypeScript.RuleEngine.validateRingPlacement`](../../src/server/game/RuleEngine.ts:161), which now delegates to `PlacementAggregate.validatePlacement`, and enumeration in [`TypeScript.RuleEngine.getValidRingPlacements`](../../src/server/game/RuleEngine.ts:839).
  - Backend skip semantics in [`TypeScript.RuleEngine.validateSkipPlacement`](../../src/server/game/RuleEngine.ts:116) and placement-phase move generation in [`TypeScript.RuleEngine.getValidMoves`](../../src/server/game/RuleEngine.ts:839), both delegating to `PlacementAggregate.validateSkipPlacement` / `evaluateSkipPlacementEligibility` and producing explicit `skip_placement` moves when R080 conditions are met.
  - Shared no‑dead‑placement helper [`TypeScript.hasAnyLegalMoveOrCaptureFromOnBoard()`](../../src/shared/engine/core.ts:367).
  - Shared placement mutators in [`TypeScript.applyPlacementOnBoard()` / `mutatePlacement()` / `applyPlacementMove()`](../../src/shared/engine/aggregates/PlacementAggregate.ts:687) act as the single source of truth for placement side‑effects and are consumed by backend `RuleEngine.processRingPlacement`, backend `GameEngine.applyMove`, and sandbox `ClientSandboxEngine.tryPlaceRings()` / canonical move application.
  - Sandbox placement logic aligned with shared helpers:
    - Placement enumeration in [`TypeScript.sandboxPlacement.enumerateLegalRingPlacements`](../../src/client/sandbox/sandboxPlacement.ts:125).
    - Hypothetical stack creation and no‑dead‑placement checks in [`TypeScript.sandboxPlacement.createHypotheticalBoardWithPlacement`](../../src/client/sandbox/sandboxPlacement.ts:29) and [`TypeScript.sandboxPlacement.hasAnyLegalMoveOrCaptureFrom`](../../src/client/sandbox/sandboxPlacement.ts:85).
    - Human placement path in [`TypeScript.ClientSandboxEngine.handleHumanCellClick`](../../src/client/sandbox/ClientSandboxEngine.ts:457) and AI placement in [`TypeScript.ClientSandboxEngine.maybeRunAITurn`](../../src/client/sandbox/ClientSandboxEngine.ts:524), both funnelling through [`TypeScript.ClientSandboxEngine.applyCanonicalMoveInternal`](../../src/client/sandbox/ClientSandboxEngine.ts:1746) and `tryPlaceRings()`.
- **Supporting / tests**
  - Placement tests (including no-dead-placement) in shared movement/placement suites and scenario tests that create illegal high stacks.
  - Backend vs sandbox parity tests that verify placement legality matches between hosts.
- **Status**
  - R080–R082 are **HC** for core semantics. Edge-case text in the Complete Rules about mandatory placement when "no legal moves exist" is implemented via RuleEngine + TurnEngine interactions and is covered indirectly rather than as a single dedicated function.

### 3.4 Non‑capture movement and marker interaction (R090–R092)

**R090–R091 Movement availability and path/distance rules (HC)**

- **Primary implementation**
  - Shared movement target enumeration in [`TypeScript.enumerateSimpleMoveTargetsFromStack()`](../../src/shared/engine/movementLogic.ts:55) using `MovementBoardView`, consolidated in [`TypeScript.MovementAggregate`](../../src/shared/engine/aggregates/MovementAggregate.ts:1) (`validateMovement`, `enumerateMovementTargets`, `enumerateSimpleMovesForPlayer`, `applySimpleMovement`, `applyMovement`).
  - Shared legality checks in [`TypeScript.hasAnyLegalMoveOrCaptureFromOnBoard()`](../../src/shared/engine/core.ts:367) and `MovementAggregate.validateMovement`.
  - Backend movement validation in [`TypeScript.RuleEngine.validateStackMovement`](../../src/server/game/RuleEngine.ts:209) (delegating to shared movement helpers) and backend movement application in [`TypeScript.RuleEngine.processStackMovement`](../../src/server/game/RuleEngine.ts:404) / [`TypeScript.GameEngine.applyMove`](../../src/server/game/GameEngine.ts:92), both wired through `MovementAggregate.applySimpleMovement`.
  - Sandbox movement enumeration in [`TypeScript.sandboxMovement.enumerateSimpleMovementLandings`](../../src/client/sandbox/sandboxMovement.ts:23) and click-driven movement host [`TypeScript.ClientSandboxEngine.handleLegacyMovementClick`](../../src/client/sandbox/ClientSandboxEngine.ts:1897).
- **Supporting / tests**
  - Shared movement/aggregate tests in [`tests/unit/movement.shared.test.ts`](../../tests/unit/movement.shared.test.ts) and [`tests/unit/MovementAggregate.shared.test.ts`](../../tests/unit/MovementAggregate.shared.test.ts).
  - Backend aggregate integration in [`tests/unit/movement.shared.test.ts`](../../tests/unit/movement.shared.test.ts).
  - Sandbox movement/aggregate parity in [`tests/unit/ClientSandboxEngine.movementParity.shared.test.ts`](../../tests/unit/ClientSandboxEngine.movementParity.shared.test.ts) and broader backend–sandbox parity in `tests/unit/ClientSandboxEngine.*.test.ts` and `tests/unit/Seed17Move16And33Parity.GameEngine_vs_Sandbox.test.ts`.

**R092 Marker interaction during non-capture movement (HC)**

- **Primary implementation**
  - Shared path helper [`TypeScript.applyMarkerEffectsAlongPathOnBoard()`](../../src/shared/engine/core.ts:619), used by both movement and capture segments.
  - Backend usage in movement and capture application inside [`TypeScript.GameEngine`](../../src/server/game/GameEngine.ts:92) (`applyMove()`, `performOvertakingCapture()`).
  - Sandbox usage in [`TypeScript.ClientSandboxEngine.applyMarkerEffectsAlongPath`](../../src/client/sandbox/ClientSandboxEngine.ts:1089).
- **Landing-on-own-marker elimination**
  - Implemented in:
    - Shared progress accounting (marker removed + immediate top-ring elimination) within backend movement and capture application logic.
    - Sandbox movement in [`TypeScript.ClientSandboxEngine.handleLegacyMovementClick`](../../src/client/sandbox/ClientSandboxEngine.ts:1897) (checks `landedOnOwnMarker` and performs cap reduction and elimination via MovementAggregate).
    - Sandbox capture application in [`TypeScript.ClientSandboxEngine.applyCaptureSegment`](../../src/client/sandbox/ClientSandboxEngine.ts:702).
- **Supporting / tests**
  - Dedicated landing-on-own-marker invariants in [`tests/unit/ClientSandboxEngine.landingOnOwnMarker.test.ts`](../../tests/unit/ClientSandboxEngine.landingOnOwnMarker.test.ts).
  - Shared movement tests verifying marker flipping and collapsing behaviour.

### 3.5 Overtaking capture and chains (R100–R103)

**R101–R102 Single capture segment legality and application (HC)**

- **Primary implementation**
  - Shared segment validator [`TypeScript.validateCaptureSegmentOnBoard()`](../../src/shared/engine/core.ts:202).
  - Shared capture-move generator [`TypeScript.enumerateCaptureMoves()`](../../src/shared/engine/captureLogic.ts:26).
  - Backend validation in [`TypeScript.RuleEngine.validateCapture`](../../src/server/game/RuleEngine.ts:286) and [`TypeScript.RuleEngine.validateChainCaptureContinuation`](../../src/server/game/RuleEngine.ts:326).
  - Backend application in [`TypeScript.GameEngine`](../../src/server/game/GameEngine.ts:92) via `performOvertakingCapture()`.
  - Sandbox capture enumeration in [`TypeScript.sandboxCaptures.enumerateCaptureSegmentsFromBoard`](../../src/client/sandbox/sandboxCaptures.ts:37).
  - Sandbox application in [`TypeScript.sandboxCaptures.applyCaptureSegmentOnBoard`](../../src/client/sandbox/sandboxCaptures.ts:98) and movement host [`TypeScript.ClientSandboxEngine.applyCaptureSegment`](../../src/client/sandbox/ClientSandboxEngine.ts:702).
- **Supporting / tests**
  - Shared capture tests in [`tests/unit/captureLogic.shared.test.ts`](../../tests/unit/captureLogic.shared.test.ts).
  - Server chain-capture tests such as [`tests/unit/GameEngine.chainCapture.test.ts`](../../tests/unit/GameEngine.chainCapture.test.ts) and cyclic/hex-specific suites (e.g. `GameEngine.cyclicCapture.hex.height3.test.ts`).
  - Capture-search helpers for AI in [`TypeScript.sandboxCaptureSearch.findMaxCaptureChains`](../../src/client/sandbox/sandboxCaptureSearch.ts:80), aligning with Rust/Node parity scripts.

**R103 Chain overtaking rule (HC)**

- **Primary implementation**
  - Shared enumeration of follow-up segments via [`TypeScript.enumerateCaptureMoves()`](../../src/shared/engine/captureLogic.ts:26).
  - Backend chain state tracked in `chainCaptureState` within [`TypeScript.GameEngine`](../../src/server/game/GameEngine.ts:92) and [`TypeScript.CaptureAggregate`](../../src/shared/engine/aggregates/CaptureAggregate.ts:1).
  - Sandbox chain engine now lives in [`TypeScript.ClientSandboxEngine.performCaptureChainInternal`](../../src/client/sandbox/ClientSandboxEngine.ts:1960), which:
    - Applies an initial segment.
    - Transitions the game into `chain_capture` phase while continuations exist.
    - Forces continuation until no legal segments remain.
    - Calls host `onCaptureSegmentApplied` once per segment and `onMovementComplete` exactly once when the chain terminates, so history snapshots see post-chain automatic consequences.
- **Supporting / tests**
  - Mandatory chain and directional-flexibility examples in capture tests and scenarios (`180° reversal`, cyclic patterns).
  - Tests verifying that lines/territory are only processed after the entire chain completes.

### 3.6 Lines and graduated rewards (R120–R122)

**R120 Line definition and detection (HC)**

- **Primary implementation**
  - Shared line detection in [`TypeScript.lineDetection.findAllLines`](../../src/shared/engine/lineDetection.ts:21) and helper `findLinesForPlayer()`.
  - Backend exposure via [`TypeScript.BoardManager`](../../src/server/game/BoardManager.ts:37) (`findAllLines()`, `findLinesForPlayer()`).
  - Sandbox adapter [`TypeScript.sandboxLines.findAllLinesOnBoard`](../../src/client/sandbox/sandboxLines.ts:124), now a thin wrapper over the shared detector.
- **Supporting / tests**
  - Shared line decision tests in [`tests/unit/lineDecisionHelpers.shared.test.ts`](../../tests/unit/lineDecisionHelpers.shared.test.ts).
  - Sandbox line tests in [`tests/unit/sandboxLines.test.ts`](../../tests/unit/sandboxLines.test.ts).
  - GameEngine line scenario tests in [`tests/unit/GameEngine.lines.scenarios.test.ts`](../../tests/unit/GameEngine.lines.scenarios.test.ts).

**R121–R122 Line processing order and rewards (HC, with some legacy behaviour retained)**

- **Primary implementation**
  - Shared canonical line-decision helpers in [`TypeScript.lineDecisionHelpers`](../../src/shared/engine/lineDecisionHelpers.ts:1):
    - `enumerateProcessLineMoves()`, `enumerateChooseLineRewardMoves()`.
    - `applyProcessLineDecision()`, `applyChooseLineRewardDecision()`.
  - Backend decision-phase integration in [`TypeScript.GameEngine.getValidLineProcessingMoves`](../../src/server/game/GameEngine.ts:1199) and [`TypeScript.GameEngine.applyDecisionMove`](../../src/server/game/GameEngine.ts:1258), including `pendingLineRewardElimination` and explicit `eliminate_rings_from_stack` moves.
  - Sandbox canonical application path in:
    - [`TypeScript.ClientSandboxEngine.getValidLineProcessingMovesForCurrentPlayer`](../../src/client/sandbox/ClientSandboxEngine.ts:1492), which now calls the shared `enumerateProcessLineMoves()` / `enumerateChooseLineRewardMoves()` helpers and `findLinesForPlayer()` directly.
    - [`TypeScript.ClientSandboxEngine.processLinesForCurrentPlayer`](../../src/client/sandbox/ClientSandboxEngine.ts:1515), which applies `process_line` / `choose_line_option` Moves (legacy alias: `choose_line_reward`) via `applyProcessLineDecision()` / `applyChooseLineRewardDecision()` and then enforces sandbox-specific automatic cap elimination and history recording policy.
  - Legacy non-move-driven helper (historical only; module now removed):
    - Former server helper `lineProcessing.ts` exposed [`TypeScript.lineProcessing.processLinesForCurrentPlayer`], but its behaviour has since been consolidated into shared `lineDecisionHelpers` + backend `GameEngine`/`RuleEngine` flows. The file `src/server/game/rules/lineProcessing.ts` no longer exists in the current tree.
- **Supporting / tests**
  - Line reward and option-1/option-2 semantics in [`tests/unit/lineDecisionHelpers.shared.test.ts`](../../tests/unit/lineDecisionHelpers.shared.test.ts).
  - AI and WebSocket integration tests for line-reward choices in:
    - [`tests/unit/GameEngine.lines.scenarios.test.ts`](../../tests/unit/GameEngine.lines.scenarios.test.ts).
    - [`tests/unit/GameEngine.lines.scenarios.test.ts`](../../tests/unit/GameEngine.lines.scenarios.test.ts).

### 3.7 Territory disconnection and region processing (R140–R145)

**R140–R142 Region discovery, physical disconnection, representation (HC)**

- **Primary implementation**
  - Region enumeration and classification in [`TypeScript.territoryDetection.findDisconnectedRegions`](../../src/shared/engine/territoryDetection.ts:36).
  - Sandbox delegate [`TypeScript.sandboxTerritory.findDisconnectedRegionsOnBoard`](../../src/client/sandbox/sandboxTerritory.ts:431).
  - Border-marker extraction in [`TypeScript.territoryBorders.getBorderMarkerPositionsForRegion`](../../src/shared/engine/territoryBorders.ts:1) and sandbox adapter [`TypeScript.sandboxTerritory.getBorderMarkerPositionsForRegion`](../../src/client/sandbox/sandboxTerritory.ts:441).
  - Representation checks (`ActiveColors` vs `RegionColors`) encoded within shared detection/processing helpers.
- **Supporting / tests**
  - Territory disconnection scenarios in [`tests/unit/BoardManager.territoryDisconnection.test.ts`](../../tests/unit/BoardManager.territoryDisconnection.test.ts).
  - Rules-matrix and FAQ tests such as [`tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts`](../../tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts) and [`tests/scenarios/FAQ_Q15.test.ts`](../../tests/scenarios/FAQ_Q15.test.ts).
  - Python invariant tests such as `test_active_no_moves_territory_processing_regression.py`.

**R143 Self‑elimination prerequisite (HC)**

- **Primary implementation**
  - Shared gating helper [`TypeScript.territoryProcessing.canProcessTerritoryRegion`](../../src/shared/engine/territoryProcessing.ts:99), used by both TS backend and sandbox to enforce the hypothetical-outside-stack requirement.
  - Backend gating in:
    - [`TypeScript.RuleEngine.canProcessDisconnectedRegionForRules`](../../src/server/game/RuleEngine.ts:1).
    - Legacy server territory processor (historical only; module now removed). Earlier revisions exposed [`TypeScript.territoryProcessing.processDisconnectedRegionsForCurrentPlayer`], but that behaviour has been consolidated into shared `territoryProcessing` + backend `GameEngine`/`RuleEngine` flows. The file `src/server/game/rules/territoryProcessing.ts` no longer exists in the current tree.
  - Sandbox gating in:
    - [`TypeScript.ClientSandboxEngine.canProcessDisconnectedRegion`](../../src/client/sandbox/ClientSandboxEngine.ts:1315).
    - [`TypeScript.ClientSandboxEngine.getValidTerritoryProcessingMovesForCurrentPlayer`](../../src/client/sandbox/ClientSandboxEngine.ts:2156), which filters shared `enumerateProcessTerritoryRegionMoves()` by the same prerequisite.
- **Supporting / tests**
  - Territory decision helper tests in [`tests/unit/territoryDecisionHelpers.shared.test.ts`](../../tests/unit/territoryDecisionHelpers.shared.test.ts).
  - FAQ Q23 scenarios in [`tests/scenarios/FAQ_Q15.test.ts`](../../tests/scenarios/FAQ_Q15.test.ts) and matrix tests driving explicit self-elimination decisions.

**R144–R145 Region processing order and collapse/elimination (HC, with both legacy and move-driven implementations)**

- **Primary implementation**
  - Shared canonical region-application logic [`TypeScript.territoryProcessing.applyTerritoryRegion`](../../src/shared/engine/territoryProcessing.ts:172) and end-to-end decision helpers [`TypeScript.territoryDecisionHelpers.enumerateProcessTerritoryRegionMoves`](../../src/shared/engine/territoryDecisionHelpers.ts:123) (canonical `choose_territory_option` MoveType; legacy alias: `process_territory_region`) and `applyProcessTerritoryRegionDecision()`.
  - Backend integration:
    - RuleEngine decision move generation for `choose_territory_option` (legacy alias: `process_territory_region`) and `eliminate_rings_from_stack` (territory context) in [`TypeScript.RuleEngine.getValidMoves`](../../src/server/game/RuleEngine.ts:839).
    - Legacy automatic processor `TypeScript.territoryProcessing.processDisconnectedRegionsForCurrentPlayer` (module removed; replaced by move-driven phases in `GameEngine`).
  - Sandbox integration:
    - Region discovery and automatic processing in [`TypeScript.ClientSandboxEngine.processDisconnectedRegionsForCurrentPlayer`](../../src/client/sandbox/ClientSandboxEngine.ts:2057), which now drives `choose_territory_option` (legacy alias: `process_territory_region`) and in-loop self-elimination via shared `applyProcessTerritoryRegionDecision()` and `applyEliminateRingsFromStackDecision()`.
    - Canonical move application in [`TypeScript.ClientSandboxEngine.applyCanonicalMoveInternal`](../../src/client/sandbox/ClientSandboxEngine.ts:1746) for `choose_territory_option` (legacy alias: `process_territory_region`) and `eliminate_rings_from_stack`.
    - Flags `_pendingTerritorySelfElimination` and `_pendingLineRewardElimination` used to mirror backend phase behaviour in sandbox parity traces.
- **Supporting / tests**
  - Territory region and self-elimination tests under [`tests/unit/territoryDecisionHelpers.shared.test.ts`](../../tests/unit/territoryDecisionHelpers.shared.test.ts).
  - Scenario tests focused on multi-region and chain reactions in `tests/scenarios/**`.
  - Invariant/regression tests in `ai-service/tests/invariants/**`.

### 3.8 Victory, last-player-standing, stalemate, and S‑invariant (R170–R173, R190–R191)

**R170–R172 Elimination and territory victories and last-player-standing (HC)**

(Explicitly covering R170 ring-elimination victory, R171 territory-control victory, and R172 last-player-standing.)

- **Primary implementation**
  - Victory evaluation in [`TypeScript.VictoryAggregate.evaluateVictory`](../../src/shared/engine/aggregates/VictoryAggregate.ts:45), which:
    - Checks ring-elimination thresholds (`victoryThreshold`).
    - Checks `territorySpaces` vs `territoryVictoryThreshold`.
    - Encodes last-player-standing semantics via per-player legal-action availability at the next turn.
  - Backend usage:
    - `RuleEngine.checkGameEnd()` in [`TypeScript.RuleEngine.checkGameEnd`](../../src/server/game/RuleEngine.ts:728).
    - `GameEngine.makeMove()` and `applyDecisionMove()` call `checkGameEnd()` after each move/decision; terminal states normalise phases away from decision phases.
  - Sandbox victory wrapper [`TypeScript.sandboxVictory.checkSandboxVictory`](../../src/client/sandbox/sandboxVictory.ts:70) and game-end orchestration in [`TypeScript.sandboxGameEnd.checkAndApplyVictorySandbox`](../../src/client/sandbox/sandboxGameEnd.ts:1) and [`TypeScript.ClientSandboxEngine.checkAndApplyVictory`](../../src/client/sandbox/ClientSandboxEngine.ts:1389).
- **Supporting / tests**
  - Victory scenario tests in [`tests/unit/GameEngine.victory.scenarios.test.ts`](../../tests/unit/GameEngine.victory.scenarios.test.ts).
  - AI invariants around last-player-standing and stalemate ladders in `ai-service/tests/invariants/**`.

**R173 Global stalemate and tiebreakers (HC)**

- **Primary implementation**
  - Stalemate handling and tiebreakers in [`TypeScript.VictoryAggregate.evaluateVictory`](../../src/shared/engine/aggregates/VictoryAggregate.ts:45), including:
    - Conversion of rings in hand to eliminated rings.
    - Ranking by collapsed spaces, then eliminated rings, then markers, then last actor.
  - Backend invocation via `RuleEngine.checkGameEnd()` and `GameEngine` termination hooks.
  - Sandbox victory wrapper mirrors the same fields when building `GameResult` in [`TypeScript.sandboxVictory`](../../src/client/sandbox/sandboxVictory.ts:95).
- **Supporting / tests**
  - FAQ-style stalemate tests and invariants under `tests/scenarios/**` and `ai-service/tests/invariants/**`.

**R190–R191 No randomness, progress & termination invariant (HC for accounting; NM for strict enforcement of “no randomness” at engine boundary)**

- **Primary implementation**
  - S‑invariant metric and monotonicity in [`TypeScript.computeProgressSnapshot()`](../../src/shared/engine/core.ts:531) and comments in [`RULES_CANONICAL_SPEC.md`](../../RULES_CANONICAL_SPEC.md).
  - Backend history entries that record `progressBefore`/`progressAfter` and hashes in [`TypeScript.GameEngine`](../../src/server/game/GameEngine.ts:92) near `appendHistoryEntry()`.
  - Sandbox history mirroring in [`TypeScript.ClientSandboxEngine.appendHistoryEntry`](../../src/client/sandbox/ClientSandboxEngine.ts:271), including S‑invariant and board summaries.
  - TS↔Python parity checks comparing S‑invariant and state hashes in [`TypeScript.RulesBackendFacade.compareTsAndPython`](../../src/server/game/RulesBackendFacade.ts:309).
- **Randomness**
  - The core rules engine is deterministic; however, the _AI policies_ intentionally use pseudo‑randomness:
    - Local AI RNG in [`TypeScript.localAIMoveSelection`](../../src/shared/engine/localAIMoveSelection.ts:1) and sandbox AI in [`TypeScript.sandboxAI`](../../src/client/sandbox/sandboxAI.ts:1).
    - Python training env uses random seeds for self‑play and data generation.
  - This is consistent with RR‑CANON-R190 in the sense that randomness lives outside the pure transition rules; the underlying legal move generator and state transitions remain deterministic given a seed and move choices.
- **Supporting / tests**
  - Invariant tests and parity comparisons across TS and Python backends.
  - Tests that assert `S` monotonicity and hash equality across move application.

**Territory contract index (TS↔Python SSOT sequences)**

- **Primary implementation**
  - v2 contract vectors under `tests/fixtures/contract-vectors/v2/territory.vectors.json` and related bundles, grouped by `sequence:*_territory.*` tags and exercised by the shared contract runner (`tests/contracts/contractVectorRunner.test.ts`) plus the Python contract tests (`ai-service/tests/contracts/test_contract_vectors.py`).
- **Supporting / tests**
  - Combined index documented in `RULES_SCENARIO_MATRIX.md` under “Territory contract index”, currently including:
    - `sequence:square_territory.region_then_elim` → `territory.square_region_then_elim.step1_region`, `territory.square_region_then_elim.step2_elim` with host parity in `tests/unit/TerritoryDecisions.SquareRegionThenElim.GameEngine_vs_Sandbox.test.ts`.
    - `sequence:hex_territory.region_then_elim` → `territory.hex_region_then_elim.step1_region`, `territory.hex_region_then_elim.step2_elim` with host parity in `tests/unit/TerritoryDecisions.HexRegionThenElim.GameEngine_vs_Sandbox.test.ts`.
    - `sequence:square_territory.two_regions_then_elim` → `territory.square_two_regions_then_elim.step1_regionB`, `territory.square_two_regions_then_elim.step2_regionA`, `territory.square_two_regions_then_elim.step3_elim` with host parity in `tests/unit/TerritoryDecisions.GameEngine_vs_Sandbox.test.ts` and `tests/unit/TerritoryDecisions.SquareTwoRegionThenElim.GameEngine_vs_Sandbox.test.ts`.
    - `sequence:square19_territory.two_regions_then_elim` → `territory.square19_two_regions_then_elim.step1_regionB`, `territory.square19_two_regions_then_elim.step2_regionA`, `territory.square19_two_regions_then_elim.step3_elim` with host parity in `tests/unit/TerritoryDecisions.Square19TwoRegionThenElim.GameEngine_vs_Sandbox.test.ts`.
    - `sequence:hex_territory.two_regions_then_elim` → `territory.hex_two_regions_then_elim.step1_regionB`, `territory.hex_two_regions_then_elim.step2_regionA`, `territory.hex_two_regions_then_elim.step3_elim` with host parity in `tests/unit/TerritoryDecisions.HexTwoRegionThenElim.GameEngine_vs_Sandbox.test.ts`.
- **Notes**
  - These sequences collectively act as the SSOT for advanced territory and self-elimination flows (single-region and multi-region) across square8, square19, and hex; any new high-leverage territory scenarios should either extend this table or add new sequences in the same style, with matching backend↔sandbox parity tests.
  - For mixed line+territory flows (Q7/Q20 overlength line followed by a single-cell region) on all three board families, the corresponding SSOT multi-phase turn vectors are:
    - `sequence:turn.line_then_territory.square8`
    - `sequence:turn.line_then_territory.square19`
    - `sequence:turn.line_then_territory.hex`
      These live in `tests/fixtures/contract-vectors/v2/multi_phase_turn.vectors.json` and are tied to orchestrator-backed multi-phase tests plus Python line+territory parity/snapshot suites (see `RULES_SCENARIO_MATRIX.md` row `combined_line_then_territory_full_sequence`).

**Forced-elimination & territory-line endgame contracts (TS↔Python SSOT bundles)**

- **Primary implementation**
  - Forced-elimination chains and host gating behaviour are captured in the v2 `forced_elimination.vectors.json` bundle under IDs such as:
    - `forced_elimination.monotone_chain.step1.square8`
    - `forced_elimination.monotone_chain.step2.square8`
    - `forced_elimination.monotone_chain.final.square8`
    - `forced_elimination.rotation.skip_eliminated.square8`
    - `forced_elimination.territory_explicit.square8`
    - `forced_elimination.territory_no_host_fe.square8`
    - `forced_elimination.anm_guard.hexagonal`
  - Late-game line/territory endgame interactions (e.g. overlength lines feeding into near-victory territory states) are expressed in `territory_line_endgame.vectors.json` under IDs:
    - `territory_line.overlong_line.step1.square8`
    - `territory_line.single_point_swing.square19`
    - `territory_line.decision_auto_exit.square8`
- **Supporting / tests**
  - TS contract runner:
    - `tests/contracts/contractVectorRunner.test.ts` loads all v2 bundles (including `forced_elimination` and `territory_line_endgame`) and validates their assertions via the shared orchestrator.
  - Python contract/parity:
    - `ai-service/tests/contracts/test_contract_vectors.py` discovers all `*.vectors.json` bundles under `tests/fixtures/contract-vectors/v2/` (including `forced_elimination` and `territory_line_endgame`) and validates them against the Python rules engine.
    - Targeted forced-elimination parity and invariants live in:
      - `ai-service/tests/parity/test_forced_elimination_sequences_parity.py`
      - `ai-service/tests/invariants/test_anm_and_termination_invariants.py`
      - `ai-service/tests/invariants/test_forced_elimination_first_class_regression.py`
- **Notes**
  - Together, these bundles form the SSOT for forced-elimination chains, host-level forced-elimination gating, and late-game line/territory endgame behaviour; scenario suites such as `tests/scenarios/ForcedEliminationAndStalemate.test.ts` and victory/turn-sequence tests remain the narrative layer over these contracts.

**Multi-region territory & mixed line+territory index (TS↔Python SSOT sequences)**

- **Primary implementation**
  - Mixed multi-region line+territory turn sequences are expressed as v2 multi-phase vectors tagged:
    - `sequence:turn.line_then_territory.multi_region.square8`
    - `sequence:turn.line_then_territory.multi_region.square19`
    - `sequence:turn.line_then_territory.multi_region.hex`
  - These vectors live in `tests/fixtures/contract-vectors/v2/multi_phase_turn.vectors.json` under ids such as:
    - `multi_phase.line_then_multi_region_territory.square8.step1_line/step2_regionB/step3_regionA`,
    - `multi_phase.line_then_multi_region_territory.square19.*`,
    - `multi_phase.line_then_multi_region_territory.hex.*`,
      and are marked with `skip` so the generic multi-step runner treats them as metadata rather than executable sequences.
- **Supporting / tests**
  - Backend ↔ sandbox host parity for these mixed multi-region flows is enforced by:
    - `tests/parity/Backend_vs_Sandbox.CaptureAndTerritoryParity.test.ts`:
      - `combined line + multi-region territory parity (line then two regions, square8)`,
      - `combined line + multi-region territory parity (line then two regions, square19)`,
      - `combined line + multi-region territory parity (line then two regions, hexagonal)`.
  - Python parity hooks:
    - `ai-service/tests/parity/test_line_and_territory_scenario_parity.py::test_turn_line_then_territory_sequence_metadata` asserts that the above sequence ids exist and are tagged correctly in `multi_phase_turn.vectors.json`.
- **Notes**
  - Together, these mixed multi-region sequences and parity tests form the SSOT for Q20/Q23-style “line then multi-region territory” behaviour across square8, square19, and hex. New multi-region mixed scenarios should extend this index and add corresponding host parity + metadata tests rather than introducing ad-hoc test-only flows.

**Unmapped / partially mapped rules in this cluster**

- RR‑CANON-R190's "no randomness" constraint is not enforced by any explicit runtime check; instead it is an architectural contract. **Status: LC** for the enforcement aspect, **HC** for termination invariant R191 via S‑metric.

### 3.9 Global legal actions and ANM invariants (R200–R207)

**R200–R201 Global legal actions and turn-material predicates (HC)**

- **Primary implementation**
  - Global legal action enumeration via [`TypeScript.getValidMoves`](../../src/shared/engine/orchestration/turnOrchestrator.ts) and [`TypeScript.hasValidMoves`](../../src/shared/engine/orchestration/turnOrchestrator.ts), which aggregate placements, movements, captures, decisions, and forced elimination into a unified Move list.
  - Turn-material predicates computed in [`TypeScript.VictoryAggregate`](../../src/shared/engine/aggregates/VictoryAggregate.ts:45) when determining whether a player has controlled stacks or rings in hand.
  - Backend usage in [`TypeScript.GameEngine`](../../src/server/game/GameEngine.ts:92) via `getValidMovesForCurrentPlayer()` and orchestrator-backed `TurnEngineAdapter`.
  - Sandbox equivalent via [`TypeScript.ClientSandboxEngine.getValidMovesForCurrentPlayer`](../../src/client/sandbox/ClientSandboxEngine.ts).
- **Supporting / tests**
  - Orchestrator integration tests under `tests/unit/TurnEngineAdapter.integration.test.ts`.
  - ANM regression scenarios in `ai-service/tests/invariants/test_anm_and_termination_invariants.py`.

**R202–R203 Active-no-moves (ANM) state detection and avoidance (HC)**

- **Primary implementation**
  - ANM detection via `hasValidMoves(state)` in the orchestrator; if false for the current player when `gameStatus == ACTIVE`, the orchestrator applies forced elimination or terminates the game.
  - Invariant checks in [`TypeScript.turnLogic.advanceTurnAndPhase`](../../src/shared/engine/turnLogic.ts:135) delegates that ensure no player is left without legal actions.
  - Python ANM checks in `ai-service/app/game_engine/__init__.py` via `has_valid_moves()` and forced-elimination gating.
- **Supporting / tests**
  - `INV-ACTIVE-NO-MOVES` invariant enforced by orchestrator soaks (`scripts/run-orchestrator-soak.ts`).
  - ANM regression tests in `ai-service/tests/invariants/test_anm_and_termination_invariants.py`.

**R204 Phase-local decision exits (HC)**

- **Primary implementation**
  - Phase-exit logic in [`TypeScript.orchestration/phaseStateMachine`](../../src/shared/engine/orchestration/phaseStateMachine.ts) handles auto-exit from `line_processing` and `territory_processing` when no decisions remain.
  - Backend phase advancement in [`TypeScript.advanceGameForCurrentPlayer`](../../src/server/game/turn/TurnEngine.ts:91) and sandbox equivalents.
- **Supporting / tests**
  - Multi-phase turn tests under `tests/unit/TurnEngineAdapter.integration.test.ts`.
  - Decision-phase auto-exit covered by contract vectors in `tests/fixtures/contract-vectors/v2/multi_phase_turn.vectors.json`.

**R205–R206 Forced-elimination taxonomy and target choice (LC for interactive choice; HC for deterministic policies)**

- **Primary implementation**
  - Host-level forced elimination triggered by [`TypeScript.turnLogic`](../../src/shared/engine/turnLogic.ts:135) delegates when a player has stacks but no legal placements, movements, or captures.
  - Explicit elimination decisions during line and territory processing via `eliminate_rings_from_stack` moves enumerated by [`TypeScript.lineDecisionHelpers`](../../src/shared/engine/lineDecisionHelpers.ts) and [`TypeScript.territoryDecisionHelpers`](../../src/shared/engine/territoryDecisionHelpers.ts).
  - **Note:** Per `KNOWN_ISSUES.md` P0.1, current engines auto-select elimination targets in some forced-elimination situations rather than surfacing an interactive choice to human players. This is a known deviation from RR-CANON-R206.
- **Supporting / tests**
  - Forced-elimination chain vectors in `tests/fixtures/contract-vectors/v2/forced_elimination.vectors.json`.
  - Backend↔sandbox elimination parity tests.

**R207 Real actions vs forced elimination for LPS and termination (HC)**

- **Primary implementation**
  - Last-player-standing (LPS) victory logic in [`TypeScript.VictoryAggregate.evaluateVictory`](../../src/shared/engine/aggregates/VictoryAggregate.ts:45) distinguishes real actions (placements, movements, captures) from forced elimination.
  - S‑invariant monotonicity and termination guarantees enforced via [`TypeScript.computeProgressSnapshot()`](../../src/shared/engine/core.ts:531).
- **Supporting / tests**
  - LPS victory tests in `tests/unit/GameEngine.victory.scenarios.test.ts`.
  - Termination invariant suites in `ai-service/tests/invariants/test_anm_and_termination_invariants.py`.

---

## 4. Inverse Mapping: Implementation → Rules

This section lists the major rules-related components and the canonical rules they implement or orchestrate. Where behaviour appears without a corresponding rule in [`RULES_CANONICAL_SPEC.md`](../../RULES_CANONICAL_SPEC.md), it is flagged explicitly.

### 4.1 Shared engine modules

- [`TypeScript.core`](../../src/shared/engine/core.ts:1)
  - **Rules:** R001–R003 (geometry, distance), R020–R023 (stack metadata), R030–R031 (marker/collapsed semantics), R050–R052 (S‑invariant portions of state validity), R060–R062 (elimination accounting via helpers), R090–R092 (movement path + markers), R101–R102 (capture geometry), R190–R191 (progress metric).
  - **Notes:** This module is the lowest-level rules substrate and is heavily reused by movement, capture, lines, and territory helpers.

- [`TypeScript.movementLogic`](../../src/shared/engine/movementLogic.ts:1)
  - **Rules:** R090–R091 (non-capture path and distance checks), R072 (used indirectly via has-any-move queries).
  - **Notes:** Pure helper for generating simple non-capturing moves; does not apply state changes directly.

- [`TypeScript.captureLogic`](../../src/shared/engine/captureLogic.ts:1)
  - **Rules:** R101–R103 (capture segment enumeration, chain possibilities), R100 (used to detect capture availability for forced elimination preconditions).
  - **Notes:** Stateless enumeration; application is handled by GameEngine / sandbox engines.

- [`TypeScript.lineDetection`](../../src/shared/engine/lineDetection.ts:1)
  - **Rules:** R120 (line definition and eligibility).
  - **Notes:** Shared for backend, sandbox, and Python parity views.

- [`TypeScript.lineDecisionHelpers`](../../src/shared/engine/lineDecisionHelpers.ts:1)
  - **Rules:** R121–R122 (line processing order and graduated rewards), partially R060–R061 (elimination accounting), R191 (S progress via collapse).
  - **Notes:** Encodes the canonical graduated reward options; hosts that bypass these helpers risk drifting from RR‑CANON.

- [`TypeScript.territoryDetection`](../../src/shared/engine/territoryDetection.ts:36)
  - **Rules:** R040–R041, R140–R142 (region discovery, physical disconnection, representation, explicitly including R141 physical disconnection criterion).
  - **Notes:** Old bespoke territory search code in sandbox and server remains for reference but is explicitly marked deprecated in favour of this module.

- [`TypeScript.territoryProcessing`](../../src/shared/engine/territoryProcessing.ts:1)
  - **Rules:** R143–R145 (self‑elimination prerequisite and region collapse mechanics), R060–R062 (elimination crediting), R191 (progress).
  - **Notes:** Provides both pure geometric application (`applyTerritoryRegion()`) and gating helper (`canProcessTerritoryRegion()`).

- [`TypeScript.territoryDecisionHelpers`](../../src/shared/engine/territoryDecisionHelpers.ts:1)
  - **Rules:** R144–R145 (move-driven region processing and elimination decisions), R143 (via use of `canProcessTerritoryRegion()`).
  - **Notes:** Canonical move shapes for `choose_territory_option` (legacy alias: `process_territory_region`) and `eliminate_rings_from_stack` in territory context.

- [`TypeScript.turnLogic`](../../src/shared/engine/turnLogic.ts:135)
  - **Rules:** R070–R072 (turn phases and deterministic progression), R100 (forced elimination entry via delegates), R172–R173 (last-player-standing and stalemate readiness via legal-action checks).
  - **Notes:** Implementation chooses concrete parameters (like maximum loop counts) that are not specified in RR‑CANON but do not change semantics.

- [`TypeScript.VictoryAggregate`](../../src/shared/engine/aggregates/VictoryAggregate.ts:45)
  - **Rules:** R170–R173, R061–R062, R191 (termination justification).
  - **Notes:** This is the canonical place where all three victory conditions and stalemate ladders are encoded.

### 4.2 Server orchestration components

- [`TypeScript.BoardManager`](../../src/server/game/BoardManager.ts:37)
  - **Rules:** R001–R003 (board sizes and adjacency realised as concrete coordinates), R021–R023 (stack exclusivity and mutation guards), R030–R031 (marker/collapsed invariants), R050–R052 (board validity).
  - **Implementation-without-canonical-rule:**
    - Development-only repair behaviour (e.g., auto-removing markers when a stack is written to a key) is not explicitly described in RR‑CANON and is best understood as a defensive invariant enforcement, not a separate rule of play.

- [`TypeScript.RuleEngine`](../../src/server/game/RuleEngine.ts:46)
  - **Rules:** R070–R072 (phase-specific move generation), R080–R082 (placement and skip), R090–R092 (movement legality), R100–R103 (capture legality and chain continuation), R120–R122 (line decision move enumeration), R140–R145 (territory decision move enumeration), R170–R173 (end-of-turn victory check), R191 (via S‑invariant fields passed to Python backend).
  - **Notes:** This class is the canonical move generator/validator for the TS backend.

- [`TypeScript.GameEngine`](../../src/server/game/GameEngine.ts:92)
  - **Rules:** Orchestrates many clusters via shared helpers:
    - R070–R072: sequencing of movement, capture chains, lines, territory, and victory checks in `makeMove()`, `applyDecisionMove()`, and `advanceGame()`.
    - R090–R092, R101–R103: application of movement/capture segments and marker path effects in `applyMove()` and `performOvertakingCapture()`.
    - R120–R122: integration of line decision phases and automatic behaviours.
    - R140–R145: integration of territory decision phases and legacy automatic region processing.
    - R170–R173: calls to `RuleEngine.checkGameEnd()` and terminal phase normalisation.
    - R190–R191: history and S‑invariant tracking, plus TS↔Python parity hooks.
  - **Implementation-without-canonical-rule:**
    - Blocked-state resolver `resolveBlockedStateForCurrentPlayerForTesting()` performs global searches and repeated forced-elimination passes to recover from inconsistent or degenerate test positions. RR‑CANON does not specify this recovery algorithm; it is a test harness utility, not a rule of play.

- [`TypeScript.advanceGameForCurrentPlayer`](../../src/server/game/turn/TurnEngine.ts:91)
  - **Rules:** Encapsulates R070–R072 and R100 by binding shared `advanceTurnAndPhase()` to concrete notions of "has any placement/movement/capture" and "apply forced elimination".
  - **Notes:** Also approximates per-player ring-cap usage in terms of `BOARD_CONFIGS.ringsPerPlayer`, which is an implementation detail consistent with but not fully specified by RR‑CANON.

- [`TypeScript.RulesBackendFacade`](../../src/server/game/RulesBackendFacade.ts:54) and [`TypeScript.PythonRulesClient`](../../src/server/services/PythonRulesClient.ts:33)
  - **Rules:** No new game rules; these modules enforce **consistency** between TS and Python engines for all rule clusters, particularly R170–R173 and R190–R191.
  - **Implementation-without-canonical-rule:**
    - Modes such as `ts`, `shadow`, and `python` and metrics like `rules_valid_mismatch_total` are purely infrastructural and have no counterpart in RR‑CANON.

### 4.3 Client sandbox and AI components

- [`TypeScript.ClientSandboxEngine`](../../src/client/sandbox/ClientSandboxEngine.ts:137)
  - **Rules:** Mirrors backend semantics for virtually all clusters, with client-local history and debug tooling. In particular:
    - Placement and no‑dead‑placement: R080–R082.
    - Movement, capture, and chain capture: R090–R092, R101–R103.
    - Line and territory decision phases including explicit `eliminate_rings_from_stack`: R120–R122, R140–R145.
    - Turn and forced elimination via shared `turnLogic.advanceTurnAndPhase` plus ClientSandboxEngine turn helpers (start-of-turn/forced-elimination glue): R070–R072, R100.
    - Victory and S‑invariant tracking: R170–R173, R190–R191.
  - **Implementation-without-canonical-rule:**
    - `traceMode`, `_debugCheckpointHook`, and history-normalisation details are instrumentation for parity and testing rather than rules of play.

- Sandbox helper modules (`sandboxMovement.ts`, `sandboxCaptures.ts`, `sandboxLines.ts`, `sandboxTerritory.ts`, `sandboxVictory.ts`, `sandboxElimination.ts`)
  - **Rules:** These modules collectively re-express R090–R092, R101–R103, R120–R122, R140–R145, R170–R173 and R191 in pure, often functional form for browser-safe clients. Turn/phase sequencing and forced elimination (R070–R072, R100) are now canonical in shared `turnLogic` + host delegates rather than in a separate sandbox turn engine. `sandboxTurnEngine.ts`, `sandboxLinesEngine.ts`, and `sandboxTerritoryEngine.ts` have been removed; their responsibilities now live in shared turn/line/territory helpers and in `ClientSandboxEngine` orchestration.
  - **Notes:** Where they diverged historically from backend behaviour (e.g., marker handling on capture landing, collapsed-only region detection), recent changes have intentionally realigned them to shared helpers and RR‑CANON, with remaining legacy surfaces marked tests/tools-only in `LEGACY_CODE_ELIMINATION_PLAN.md`.

- AI and training env (`ai-service/app/training/env.py`, `generate_data.py`, `generate_territory_dataset.py`)
  - **Rules:** These components assume the correctness of the TS/Python engines and use them to generate trajectories and labels. They depend implicitly on all clusters but do not define new rules.

- Tests under `tests/unit/**` and `tests/scenarios/**`
  - **Rules:** Provide executable specifications for many individual RR‑CANON rules or combinations (e.g., FAQ Q15/Q23, line reward FAQs, cyclic capture examples).
  - **Notes:** Some older tests reflect pre-canonical interpretations (e.g., partially-deprecated territory search code); newer shared-engine tests and canonical scenario suites should be treated as normative where tests disagree.

---

## 5. Gaps and Suspicious Areas

### 5.1 Canonical rules with no clear or only partial implementation

- **R190 No randomness (strict reading)** – No explicit runtime guard prevents injecting randomness into the rules layer. Determinism is instead maintained by convention and testing.
  - **Impact:** Low for gameplay correctness (engines are written deterministically), but important for formal verification; Static Verification Agent should treat any random sampling inside core move-generation or transition functions as a defect.
  - **Evidence:** Randomness is confined to AI policy selection in [`TypeScript.localAIMoveSelection`](../../src/shared/engine/localAIMoveSelection.ts:1), [`TypeScript.sandboxAI`](../../src/client/sandbox/sandboxAI.ts:1), and Python training infrastructure.

- **Some obscure FAQ edge cases** (e.g., unreachable "cannot perform forced elimination because all caps are already eliminated" scenario from Complete Rules Q24) are intentionally treated as **non-realizable** in the engine and are not encoded.
  - **Impact:** None for legal play; the canonical spec (Section 11–12) already resolves these as unreachable.

Overall, for the main RR‑CANON clusters (geometry, placement, movement, capture, lines, territory, victory, S‑invariant), there is **no major rule with zero implementation**. The primary risk is subtle drift between host variants (backend vs sandbox vs Python) rather than missing logic.

### 5.2 Implementation elements without a clear canonical rule

- **Board invariant "repair" behaviour** in [`TypeScript.BoardManager`](../../src/server/game/BoardManager.ts:37) and sandbox `assertBoardInvariants`.
  - Removes or rewrites inconsistent cell contents (e.g., stack + marker overlaps) in test/dev builds.
  - **Hypothesis:** Debugging aid / defence-in-depth, not a rule of play.
  - **Impact:** Low, but Static Verification Agent should ensure these repairs are never triggered on legal game trajectories.

- **Forced-elimination stack-selection heuristic** in [`TypeScript.turn.processForcedElimination`](../../src/server/game/turn/TurnEngine.ts:286) and sandbox equivalents.
  - Chooses among eligible stacks/caps using a specific ordering (e.g., smallest positive caps first).
  - RR‑CANON-R100 allows any cap choice; tiebreak is unspecified.
  - **Hypothesis:** Implementation-specific heuristic intended to be benign.
  - **Impact:** Low for rules correctness; may affect fairness or AI evaluation symmetry.

- **Blocked-state resolver for synthetic test states** in [`TypeScript.GameEngine.resolveBlockedStateForCurrentPlayerForTesting`](../../src/server/game/GameEngine.ts:2583).
  - Performs repeated passes of forced elimination and special-case logic to un-stick contrived positions.
  - **Hypothesis:** Test harness only; not reachable in normal play from legal starting positions.
  - **Impact:** Static Verification Agent should treat this as out-of-scope for canonical-play proofs but may verify it respects invariants.

- **Legacy territory search helpers** in [`TypeScript.sandboxTerritory`](../../src/client/sandbox/sandboxTerritory.ts:199) and server `territoryProcessing.ts` are now marked deprecated in comments.
  - **Hypothesis:** Kept for comparison and debugging; shared `territoryDetection` and `territoryProcessing` are the canonical implementations.
  - **Impact:** Medium risk if callers accidentally use legacy helpers instead of the shared ones; Dynamic Verification Agent should confirm that runtime code paths use the canonical helpers.

- **Trace / parity tooling flags** like `traceMode`, `_debugCheckpointHook`, and environment-controlled debug printouts across sandbox and server code.
  - **Hypothesis:** Diagnostics only.
  - **Impact:** None for rules, but these paths should be considered when analysing parity test harnesses.

---

## 6. Guidance for Downstream Agents

### 6.1 Static Verification Agent

Focus areas:

- Prove or mechanically check that shared helpers in [`TypeScript.core`](../../src/shared/engine/core.ts:1), [`TypeScript.movementLogic`](../../src/shared/engine/movementLogic.ts:1), [`TypeScript.captureLogic`](../../src/shared/engine/captureLogic.ts:1), [`TypeScript.lineDecisionHelpers`](../../src/shared/engine/lineDecisionHelpers.ts:1), [`TypeScript.territoryDetection`](../../src/shared/engine/territoryDetection.ts:36), [`TypeScript.territoryProcessing`](../../src/shared/engine/territoryProcessing.ts:1), [`TypeScript.turnLogic`](../../src/shared/engine/turnLogic.ts:135), and [`TypeScript.VictoryAggregate`](../../src/shared/engine/aggregates/VictoryAggregate.ts:45) satisfy the corresponding RR‑CANON rules enumerated in Section 3.
- Check that S‑invariant accounting in [`TypeScript.computeProgressSnapshot()`](../../src/shared/engine/core.ts:531) is monotone and bounded for all rule-legal transitions (movement, capture, line, territory, forced elimination, stalemate).
- Confirm that legacy helpers marked deprecated (e.g., old territory region search in [`TypeScript.sandboxTerritory`](../../src/client/sandbox/sandboxTerritory.ts:199) and server `territoryProcessing.ts`) are not used in normal game flows.
- Ensure no randomness or non-determinism leaks into the core transition functions; randomness should be confined to AI selectors and not influence legality or state transitions.

### 6.2 Dynamic Verification Agent

Focus areas:

- Map scenario tests to RR‑CANON rules:
  - Movement and marker behaviour → [`tests/unit/movement.shared.test.ts`](../../tests/unit/movement.shared.test.ts), [`tests/unit/ClientSandboxEngine.landingOnOwnMarker.test.ts`](../../tests/unit/ClientSandboxEngine.landingOnOwnMarker.test.ts).
  - Capture chains and cyclic patterns → `GameEngine.cyclicCapture.*.test.ts` and [`tests/unit/GameEngine.chainCapture.test.ts`](../../tests/unit/GameEngine.chainCapture.test.ts).
  - Lines and graduated rewards → [`tests/unit/lineDecisionHelpers.shared.test.ts`](../../tests/unit/lineDecisionHelpers.shared.test.ts), [`tests/unit/GameEngine.lines.scenarios.test.ts`](../../tests/unit/GameEngine.lines.scenarios.test.ts), sandbox line tests.
  - Territory disconnection and Q15/Q23 → [`tests/unit/territoryDecisionHelpers.shared.test.ts`](../../tests/unit/territoryDecisionHelpers.shared.test.ts), [`tests/unit/BoardManager.territoryDisconnection.test.ts`](../../tests/unit/BoardManager.territoryDisconnection.test.ts), [`tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts`](../../tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts), [`tests/scenarios/FAQ_Q15.test.ts`](../../tests/scenarios/FAQ_Q15.test.ts).
  - Victory and stalemate ladders → [`tests/unit/GameEngine.victory.scenarios.test.ts`](../../tests/unit/GameEngine.victory.scenarios.test.ts) and invariants under `ai-service/tests/invariants/**`.
- Exercise parity suites:
  - Backend vs sandbox parity (`Seed17`, backend-vs-sandbox AI traces, etc.).
  - TS vs Python parity through `RulesBackendFacade` metrics and invariant tests.
- For each failing or brittle scenario, map back to specific RR‑CANON rules via this document to decide whether the engine or the test is out-of-spec.

### 6.3 Consistency & Edge-Case Agent

Focus areas:

- Cross-cutting behaviours:
  - Turn/phase transitions, especially around decision phases (`line_processing`, `territory_processing`, `chain_capture`) in [`TypeScript.GameEngine`](../../src/server/game/GameEngine.ts:92) and [`TypeScript.ClientSandboxEngine`](../../src/client/sandbox/ClientSandboxEngine.ts:137).
  - Forced elimination entry and stack selection heuristics in [`TypeScript.turn.processForcedElimination`](../../src/server/game/turn/TurnEngine.ts:286) and sandbox equivalents.
  - Landing-on-own-marker elimination interactions with line and territory processing (ensure orderings match RR‑CANON).
- Host alignment:
  - Confirm that backend, sandbox, and Python engines produce identical legal move sets and state transitions for all RR‑CANON-relevant scenarios, using the parity harnesses wired through [`TypeScript.RulesBackendFacade`](../../src/server/game/RulesBackendFacade.ts:54) and [`TypeScript.ClientSandboxEngine`](../../src/client/sandbox/ClientSandboxEngine.ts:137).
  - Pay special attention to long chains (captures, lines, and territory chain reactions) where historical differences have occurred.
- Edge cases flagged as suspicious in Section 5 (board repairs, legacy helpers, test-only recovery code) should be validated to never appear on trajectories starting from legal initial states and legal RR‑CANON moves.

This mapping is intended as a stable reference for future rules work: when rules or implementations evolve, updates should be expressed as changes to specific RR‑CANON IDs, with corresponding updates in the mapping sections above.
