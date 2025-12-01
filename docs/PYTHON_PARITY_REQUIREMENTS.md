# Python Parity Requirements

> **SSoT alignment:** This document is a derived parity/contract view over the canonical TS rules engine. It explains how the Python rules implementation in `ai-service/app/**` is validated against the **Canonical TS rules surface** (`src/shared/engine/**` helpers + aggregates + orchestrator + contracts and v2 contract vectors), and how parity fixtures/tests are maintained. It does not define new rules semantics.
>
> **Doc Status (2025-11-26): Active (with historical TypeScript implementation details)**
>
> - Canonical TS rules surface is the helpers + aggregates + orchestrator stack under `src/shared/engine/`: `core.ts`, movement/capture/line/territory/victory helpers, `aggregates/*Aggregate.ts`, and `orchestration/turnOrchestrator.ts` / `phaseStateMachine.ts`, plus contract vectors in `src/shared/engine/contracts/**` and `tests/fixtures/contract-vectors/v2/*.json`.
> - Function and class names under `validators/*.ts` and `mutators/*.ts` in the tables below should be read as **semantic anchors**, not necessarily 1:1 concrete TS files. As of this date, only `validators/PlacementValidator.ts` and a subset of `mutators/*Mutator.ts` exist on the TS side; the rest of the semantics are expressed via helpers, aggregates, and the orchestrator and are exercised by:
>   - TS contract tests in `tests/contracts/contractVectorRunner.test.ts`
>   - Shared-engine parity in `tests/unit/TraceFixtures.sharedEngineParity.test.ts`
>   - Backend↔sandbox parity in `tests/unit/Backend_vs_Sandbox.*.test.ts`
>   - Python contract tests in `ai-service/tests/contracts/test_contract_vectors.py`
> - When this document refers to e.g. `MovementValidator` or `LineMutator` on the TS side, it is describing the **domain boundary** that the Python `validators/*` and `mutators/*` modules implement and validate against, with the shared contract vectors as the canonical spec.

**Task:** T1-W1-F  
**Date:** 2025-11-26  
**Status:** Complete

## 1. Overview

This document specifies the parity requirements between the TypeScript canonical engine ([`src/shared/engine/`](../src/shared/engine/)) and the Python AI service rules implementation ([`ai-service/app/rules/`](../ai-service/app/rules/)).

In particular, **Wave 4 – Orchestrator Rollout & Invariant Hardening** introduces
an explicit SLO around orchestrator-driven contract vectors:

- The v2 contract vectors under `tests/fixtures/contract-vectors/v2/**` include
  families that exercise orchestrator‑first paths (turn/phase transitions,
  line→territory sequences, capture chains, and LPS/forced‑elimination tails).
- These vectors are treated as a **gating SLO** via:
  - the `orchestrator-parity` CI job in
    [`.github/workflows/ci.yml`](../.github/workflows/ci.yml:1), which runs:
    - `npm run test:orchestrator-parity:ts` (TS orchestrator suites), then
    - `./scripts/run-python-contract-tests.sh --verbose`
      (Python runner over the v2 contract vectors).
- Any failure in `ai-service/tests/contracts/test_contract_vectors.py` under this
  job is considered a direct breach of `SLO-CI-ORCH-PARITY` in
  [`docs/ORCHESTRATOR_ROLLOUT_PLAN.md`](./ORCHESTRATOR_ROLLOUT_PLAN.md#62-ci-slos)
  and must block rollout until resolved (either by fixing a bug or updating
  vectors/fixtures in line with the canonical TS engine).

### 1.1 Purpose of Python Rules Engine

The Python rules engine serves two primary purposes:

1. **AI Training**: Provides rules validation for reinforcement learning training loops
2. **AI Inference**: Enables move generation and state simulation during AI move selection

### 1.2 Relationship to TypeScript Canonical Engine

| Aspect           | TypeScript Engine                  | Python AI Service                       |
| ---------------- | ---------------------------------- | --------------------------------------- |
| **Role**         | Canonical source of truth          | Derived implementation                  |
| **Location**     | `src/shared/engine/`               | `ai-service/app/rules/`                 |
| **Exports**      | 90+ functions/types via `index.ts` | Thin adapter via `RulesEngine` protocol |
| **Validation**   | Pure functions                     | Shadow contract validation against TS   |
| **Code Sharing** | Used by Server + Client            | Language boundary prevents sharing      |

### 1.3 Canonical Move Lifecycle & SSoT References

- Move, `MoveType`, `GamePhase`, `GameStatus`, and player choice types are defined in the shared TS types under [`src/shared/types/game.ts`](../src/shared/types/game.ts).
- Orchestrator result/decision types (`ProcessTurnResult`, `PendingDecision`, `DecisionType`, `VictoryState`, `ProcessingMetadata`) are defined under [`src/shared/engine/orchestration/types.ts`](../src/shared/engine/orchestration/types.ts).
- WebSocket transport and `PlayerChoice*` decision surfaces are defined in [`src/shared/types/websocket.ts`](../src/shared/types/websocket.ts) and validated by [`src/shared/validation/websocketSchemas.ts`](../src/shared/validation/websocketSchemas.ts).
- The canonical description of the Move → PendingDecision → PlayerChoice → WebSocket → Move.id lifecycle lives in [`docs/CANONICAL_ENGINE_API.md`](./CANONICAL_ENGINE_API.md); this document defers all lifecycle semantics to that SSoT and focuses purely on TS↔Python rules parity.

### 1.4 Architecture Pattern

The Python AI service uses a **shadow contract validation** pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                     DefaultRulesEngine                          │
│                                                                 │
│  ┌──────────────────────┐    ┌───────────────────────────────┐ │
│  │  Python Mutators     │    │  GameEngine.apply_move        │ │
│  │  (PlacementMutator,  │◄───│  (Canonical Python GameEngine) │ │
│  │   MovementMutator,   │    │                               │ │
│  │   CaptureMutator,    │    │  Mirrors TS semantics         │ │
│  │   LineMutator,       │    │                               │ │
│  │   TerritoryMutator)  │    └───────────────────────────────┘ │
│  └──────────────────────┘                                       │
│           │                                                     │
│           ▼                                                     │
│  Shadow Contract Assertion: mutator_state == engine_state       │
└─────────────────────────────────────────────────────────────────┘
```

### 1.5 Orchestrator‑driven contract vector families

Wave 4 introduces explicit **orchestrator‑first contract vector bundles** under
`tests/fixtures/contract-vectors/v2/**`. These are treated as part of the
`SLO-CI-ORCH-PARITY` gate because they encode composite turn/phase behaviour
that must remain aligned across TS and Python:

- `chain_capture.vectors.json` (`category: chain_capture`) – multi‑segment
  orchestrator‑driven chain captures, including continued `continue_capture_segment`
  moves and sequence‑tagged bundles exercised by
  `tests/contracts/contractVectorRunner.test.ts`.
- `territory_processing.vectors.json` (`category: territory_processing`) – composite
  **territory region + self‑elimination** flows generated by
  `scripts/generate-orchestrator-contract-vectors.ts`, using the canonical
  `processTurn` path.

The Python runner in `ai-service/tests/contracts/test_contract_vectors.py` loads
all v2 bundles (including these orchestrator‑focused families) via
`VECTOR_CATEGORIES` and validates them against `GameEngine.apply_move()`. Any
regression in these vectors under the `orchestrator-parity` CI job is a direct
SLO breach and must be triaged as a rules/orchestrator parity issue, not an
AI‑only concern.

---

## 2. Function Parity Matrix

### 2.1 Core Utilities

| TS Function                                                             | TS File   | Python Function                                                                                  | Python File        | Parity Status | Notes                |
| ----------------------------------------------------------------------- | --------- | ------------------------------------------------------------------------------------------------ | ------------------ | ------------- | -------------------- |
| [`calculateCapHeight()`](../src/shared/engine/core.ts)                  | `core.ts` | [`calculate_cap_height()`](../ai-service/app/rules/core.py:58)                                   | `core.py`          | ✅ MATCHED    | Direct port          |
| [`calculateDistance()`](../src/shared/engine/core.ts)                   | `core.ts` | [`calculate_distance()`](../ai-service/app/rules/core.py:76)                                     | `core.py`          | ✅ MATCHED    | Direct port          |
| [`getPathPositions()`](../src/shared/engine/core.ts)                    | `core.ts` | [`get_path_positions()`](../ai-service/app/rules/core.py:94)                                     | `core.py`          | ✅ MATCHED    | Direct port          |
| [`hashGameState()`](../src/shared/engine/core.ts)                       | `core.ts` | [`hash_game_state()`](../ai-service/app/rules/core.py:208)                                       | `core.py`          | ✅ MATCHED    | Direct port          |
| [`summarizeBoard()`](../src/shared/engine/core.ts)                      | `core.ts` | [`summarize_board()`](../ai-service/app/rules/core.py:127)                                       | `core.py`          | ✅ MATCHED    | Direct port          |
| [`computeProgressSnapshot()`](../src/shared/engine/core.ts)             | `core.ts` | [`compute_progress_snapshot()`](../ai-service/app/rules/core.py:157)                             | `core.py`          | ✅ MATCHED    | Direct port          |
| [`countRingsInPlayForPlayer()`](../src/shared/engine/core.ts)           | `core.ts` | [`count_rings_in_play_for_player()`](../ai-service/app/rules/core.py:179)                        | `core.py`          | ✅ MATCHED    | Direct port          |
| [`countRingsOnBoardForPlayer()`](../src/shared/engine/core.ts)          | `core.ts` | -                                                                                                | -                  | ⚠️ MISSING    | Computed inline      |
| [`getMovementDirectionsForBoardType()`](../src/shared/engine/core.ts)   | `core.ts` | [`_get_all_directions()`](../ai-service/app/board_manager.py)                                    | `board_manager.py` | ✅ MATCHED    | Different location   |
| [`applyMarkerEffectsAlongPathOnBoard()`](../src/shared/engine/core.ts)  | `core.ts` | [`_process_markers_along_path()`](../ai-service/app/game_engine.py:971)                          | `game_engine.py`   | ✅ MATCHED    | Different name       |
| [`hasAnyLegalMoveOrCaptureFromOnBoard()`](../src/shared/engine/core.ts) | `core.ts` | [`_has_any_legal_move_or_capture_from_on_board()`](../ai-service/app/game_engine.py:1205)        | `game_engine.py`   | ✅ MATCHED    | Direct port          |
| [`validateCaptureSegmentOnBoard()`](../src/shared/engine/core.ts)       | `core.ts` | [`_validate_capture_segment_on_board_for_reachability()`](../ai-service/app/game_engine.py:1090) | `game_engine.py`   | ✅ MATCHED    | Different name       |
| `positionToString()`                                                    | `game.ts` | [`Position.to_key()`](../ai-service/app/models.py)                                               | `models.py`        | ✅ MATCHED    | Method style         |
| `stringToPosition()`                                                    | `game.ts` | -                                                                                                | -                  | ⚠️ MISSING    | Not needed in Python |
| `positionsEqual()`                                                      | `game.ts` | `==` operator                                                                                    | -                  | ✅ MATCHED    | Via Pydantic         |

### 2.2 Placement Domain

| TS Function                                                                           | TS File                 | Python Function                                                                      | Python File               | Parity Status | Notes           |
| ------------------------------------------------------------------------------------- | ----------------------- | ------------------------------------------------------------------------------------ | ------------------------- | ------------- | --------------- |
| [`validatePlacementOnBoard()`](../src/shared/engine/validators/PlacementValidator.ts) | `PlacementValidator.ts` | [`PlacementValidator.validate()`](../ai-service/app/rules/validators/placement.py:7) | `validators/placement.py` | ✅ MATCHED    | Reimplemented   |
| [`validatePlacement()`](../src/shared/engine/validators/PlacementValidator.ts)        | `PlacementValidator.ts` | [`PlacementValidator.validate()`](../ai-service/app/rules/validators/placement.py:7) | `validators/placement.py` | ✅ MATCHED    | Combined        |
| [`validateSkipPlacement()`](../src/shared/engine/validators/PlacementValidator.ts)    | `PlacementValidator.ts` | [`_get_skip_placement_moves()`](../ai-service/app/game_engine.py:1407)               | `game_engine.py`          | ✅ MATCHED    | Via enumeration |
| [`applyPlacementMove()`](../src/shared/engine/placementHelpers.ts)                    | `placementHelpers.ts`   | [`GameEngine._apply_place_ring()`](../ai-service/app/game_engine.py:2306)            | `game_engine.py`          | ✅ MATCHED    | Different name  |
| [`evaluateSkipPlacementEligibility()`](../src/shared/engine/placementHelpers.ts)      | `placementHelpers.ts`   | [`_get_skip_placement_moves()`](../ai-service/app/game_engine.py:1407)               | `game_engine.py`          | ✅ MATCHED    | Combined        |

### 2.3 Movement Domain

| TS Function                                                                      | TS File                | Python Function                                                                    | Python File              | Parity Status | Notes          |
| -------------------------------------------------------------------------------- | ---------------------- | ---------------------------------------------------------------------------------- | ------------------------ | ------------- | -------------- |
| [`enumerateSimpleMoveTargetsFromStack()`](../src/shared/engine/movementLogic.ts) | `movementLogic.ts`     | [`_get_movement_moves()`](../ai-service/app/game_engine.py:2183)                   | `game_engine.py`         | ✅ MATCHED    | Returns Move[] |
| [`validateMovement()`](../src/shared/engine/validators/MovementValidator.ts)     | `MovementValidator.ts` | [`MovementValidator.validate()`](../ai-service/app/rules/validators/movement.py:8) | `validators/movement.py` | ✅ MATCHED    | Reimplemented  |

### 2.4 Capture Domain

| TS Function                                                                | TS File               | Python Function                                                                  | Python File             | Parity Status | Notes         |
| -------------------------------------------------------------------------- | --------------------- | -------------------------------------------------------------------------------- | ----------------------- | ------------- | ------------- |
| [`enumerateCaptureMoves()`](../src/shared/engine/captureLogic.ts)          | `captureLogic.ts`     | [`_get_capture_moves()`](../ai-service/app/game_engine.py:1594)                  | `game_engine.py`        | ✅ MATCHED    | Direct port   |
| [`validateCapture()`](../src/shared/engine/validators/CaptureValidator.ts) | `CaptureValidator.ts` | [`CaptureValidator.validate()`](../ai-service/app/rules/validators/capture.py:8) | `validators/capture.py` | ✅ MATCHED    | Reimplemented |

### 2.5 Line Domain

| TS Function                                                                       | TS File                  | Python Function                                                               | Python File        | Parity Status | Notes              |
| --------------------------------------------------------------------------------- | ------------------------ | ----------------------------------------------------------------------------- | ------------------ | ------------- | ------------------ |
| [`findAllLines()`](../src/shared/engine/lineDetection.ts)                         | `lineDetection.ts`       | [`BoardManager.find_all_lines()`](../ai-service/app/board_manager.py)         | `board_manager.py` | ✅ MATCHED    | Different location |
| [`findLinesForPlayer()`](../src/shared/engine/lineDetection.ts)                   | `lineDetection.ts`       | Filtered from `find_all_lines()`                                              | `board_manager.py` | ✅ MATCHED    | Inline filter      |
| [`enumerateProcessLineMoves()`](../src/shared/engine/lineDecisionHelpers.ts)      | `lineDecisionHelpers.ts` | [`_get_line_processing_moves()`](../ai-service/app/game_engine.py:2022)       | `game_engine.py`   | ✅ MATCHED    | Different name     |
| [`enumerateChooseLineRewardMoves()`](../src/shared/engine/lineDecisionHelpers.ts) | `lineDecisionHelpers.ts` | [`_get_line_processing_moves()`](../ai-service/app/game_engine.py:2022)       | `game_engine.py`   | ⚠️ PARTIAL    | Combined           |
| [`applyProcessLineDecision()`](../src/shared/engine/lineDecisionHelpers.ts)       | `lineDecisionHelpers.ts` | [`GameEngine._apply_line_formation()`](../ai-service/app/game_engine.py:2788) | `game_engine.py`   | ✅ MATCHED    | Different name     |
| [`applyChooseLineRewardDecision()`](../src/shared/engine/lineDecisionHelpers.ts)  | `lineDecisionHelpers.ts` | [`GameEngine._apply_line_formation()`](../ai-service/app/game_engine.py:2788) | `game_engine.py`   | ✅ MATCHED    | Combined           |

### 2.6 Territory Domain

| TS Function                                                                                  | TS File                       | Python Function                                                                    | Python File        | Parity Status | Notes              |
| -------------------------------------------------------------------------------------------- | ----------------------------- | ---------------------------------------------------------------------------------- | ------------------ | ------------- | ------------------ |
| [`findDisconnectedRegions()`](../src/shared/engine/territoryDetection.ts)                    | `territoryDetection.ts`       | [`BoardManager.find_disconnected_regions()`](../ai-service/app/board_manager.py)   | `board_manager.py` | ✅ MATCHED    | Different location |
| [`canProcessTerritoryRegion()`](../src/shared/engine/territoryProcessing.ts)                 | `territoryProcessing.ts`      | [`_can_process_disconnected_region()`](../ai-service/app/game_engine.py:2160)      | `game_engine.py`   | ✅ MATCHED    | Different name     |
| [`filterProcessableTerritoryRegions()`](../src/shared/engine/territoryProcessing.ts)         | `territoryProcessing.ts`      | Inline in `_get_territory_processing_moves()`                                      | `game_engine.py`   | ✅ MATCHED    | Inline             |
| [`getProcessableTerritoryRegions()`](../src/shared/engine/territoryProcessing.ts)            | `territoryProcessing.ts`      | Inline in `_get_territory_processing_moves()`                                      | `game_engine.py`   | ✅ MATCHED    | Inline             |
| [`applyTerritoryRegion()`](../src/shared/engine/territoryProcessing.ts)                      | `territoryProcessing.ts`      | [`GameEngine._apply_territory_claim()`](../ai-service/app/game_engine.py:2894)     | `game_engine.py`   | ✅ MATCHED    | Different name     |
| [`enumerateProcessTerritoryRegionMoves()`](../src/shared/engine/territoryDecisionHelpers.ts) | `territoryDecisionHelpers.ts` | [`_get_territory_processing_moves()`](../ai-service/app/game_engine.py:2082)       | `game_engine.py`   | ✅ MATCHED    | Different name     |
| [`enumerateTerritoryEliminationMoves()`](../src/shared/engine/territoryDecisionHelpers.ts)   | `territoryDecisionHelpers.ts` | [`_get_territory_processing_moves()`](../ai-service/app/game_engine.py:2082)       | `game_engine.py`   | ✅ MATCHED    | Combined           |
| [`applyProcessTerritoryRegionDecision()`](../src/shared/engine/territoryDecisionHelpers.ts)  | `territoryDecisionHelpers.ts` | [`GameEngine._apply_territory_claim()`](../ai-service/app/game_engine.py:2894)     | `game_engine.py`   | ✅ MATCHED    | Different name     |
| [`applyEliminateRingsFromStackDecision()`](../src/shared/engine/territoryDecisionHelpers.ts) | `territoryDecisionHelpers.ts` | [`GameEngine._apply_forced_elimination()`](../ai-service/app/game_engine.py:3164)  | `game_engine.py`   | ✅ MATCHED    | Different name     |
| [`getBorderMarkerPositionsForRegion()`](../src/shared/engine/territoryBorders.ts)            | `territoryBorders.ts`         | [`BoardManager.get_border_marker_positions()`](../ai-service/app/board_manager.py) | `board_manager.py` | ✅ MATCHED    | Different location |

### 2.7 Victory Domain

| TS Function                                                 | TS File           | Python Function                                                       | Python File      | Parity Status | Notes          |
| ----------------------------------------------------------- | ----------------- | --------------------------------------------------------------------- | ---------------- | ------------- | -------------- |
| [`evaluateVictory()`](../src/shared/engine/victoryLogic.ts) | `victoryLogic.ts` | [`GameEngine._check_victory()`](../ai-service/app/game_engine.py:269) | `game_engine.py` | ✅ MATCHED    | Different name |
| [`getLastActor()`](../src/shared/engine/victoryLogic.ts)    | `victoryLogic.ts` | Inline in `_check_victory()`                                          | `game_engine.py` | ✅ MATCHED    | Inline         |

### 2.8 Turn Management

| TS Function                                                  | TS File        | Python Function                                                      | Python File      | Parity Status | Notes                       |
| ------------------------------------------------------------ | -------------- | -------------------------------------------------------------------- | ---------------- | ------------- | --------------------------- |
| [`advanceTurnAndPhase()`](../src/shared/engine/turnLogic.ts) | `turnLogic.ts` | [`GameEngine._update_phase()`](../ai-service/app/game_engine.py:440) | `game_engine.py` | ✅ MATCHED    | Combined with `_end_turn()` |

### 2.9 Mutators

| TS Mutator         | TS File                                                                                               | Python Mutator                                                        | Python File             | Parity Status | Notes                                                                                  |
| ------------------ | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ----------------------- | ------------- | -------------------------------------------------------------------------------------- |
| `PlacementMutator` | `mutators/PlacementMutator.ts`                                                                        | [`PlacementMutator`](../ai-service/app/rules/mutators/placement.py:6) | `mutators/placement.py` | ✅ MATCHED    | Delegates to GameEngine                                                                |
| `MovementMutator`  | `mutators/MovementMutator.ts`                                                                         | [`MovementMutator`](../ai-service/app/rules/mutators/movement.py:6)   | `mutators/movement.py`  | ✅ MATCHED    | Delegates to GameEngine                                                                |
| `CaptureMutator`   | `mutators/CaptureMutator.ts`                                                                          | [`CaptureMutator`](../ai-service/app/rules/mutators/capture.py:6)     | `mutators/capture.py`   | ✅ MATCHED    | Delegates to GameEngine                                                                |
| `LineMutator`      | semantic boundary over shared line helpers (`lineDecisionHelpers.ts`, `LineAggregate.ts`)             | [`LineMutator`](../ai-service/app/rules/mutators/line.py:6)           | `mutators/line.py`      | ✅ MATCHED    | Python mutator implements the same domain boundary validated via line contract vectors |
| `TerritoryMutator` | semantic boundary over shared territory helpers (`territoryProcessing.ts`, `TerritoryAggregate.ts`)   | [`TerritoryMutator`](../ai-service/app/rules/mutators/territory.py:6) | `mutators/territory.py` | ✅ MATCHED    | Python mutator implements the same domain boundary validated via territory contracts   |
| `TurnMutator`      | semantic boundary over shared turn orchestrator (`turnLogic.ts`, `orchestration/turnOrchestrator.ts`) | [`TurnMutator`](../ai-service/app/rules/mutators/turn.py:6)           | `mutators/turn.py`      | ✅ MATCHED    | Python mutator implements the same domain boundary validated via turn/phase contracts  |

### 2.10 Validators

| TS Validator         | TS File                                                                                             | Python Validator                                                          | Python File               | Parity Status | Notes                                                                                 |
| -------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------- | ------------- | ------------------------------------------------------------------------------------- |
| `PlacementValidator` | `validators/PlacementValidator.ts`                                                                  | [`PlacementValidator`](../ai-service/app/rules/validators/placement.py:6) | `validators/placement.py` | ✅ MATCHED    | Reimplemented                                                                         |
| `MovementValidator`  | semantic boundary over shared movement helpers (`movementLogic.ts`, `movementApplication.ts`)       | [`MovementValidator`](../ai-service/app/rules/validators/movement.py:7)   | `validators/movement.py`  | ✅ MATCHED    | Python validator enforces the same rules as TS movement helpers + RuleEngine binding  |
| `CaptureValidator`   | semantic boundary over shared capture helpers (`captureLogic.ts`, `captureChainHelpers.ts`)         | [`CaptureValidator`](../ai-service/app/rules/validators/capture.py:7)     | `validators/capture.py`   | ✅ MATCHED    | Python validator enforces the same rules as TS capture helpers + RuleEngine binding   |
| `LineValidator`      | semantic boundary over shared line helpers (`lineDetection.ts`, `lineDecisionHelpers.ts`)           | [`LineValidator`](../ai-service/app/rules/validators/line.py:15)          | `validators/line.py`      | ✅ MATCHED    | Python validator enforces the same rules as TS line helpers + RuleEngine binding      |
| `TerritoryValidator` | semantic boundary over shared territory helpers (`territoryDetection.ts`, `territoryProcessing.ts`) | [`TerritoryValidator`](../ai-service/app/rules/validators/territory.py:6) | `validators/territory.py` | ✅ MATCHED    | Python validator enforces the same rules as TS territory helpers + RuleEngine binding |

### 2.11 Functions NOT Implemented in Python

| TS Function                       | TS File                   | Status        | Reason                                      |
| --------------------------------- | ------------------------- | ------------- | ------------------------------------------- |
| `chooseLocalMoveFromCandidates()` | `localAIMoveSelection.ts` | 🚫 NOT NEEDED | Python has own AI selection                 |
| `SQUARE_MOORE_DIRECTIONS`         | `core.ts`                 | ⚠️ INLINE     | Defined directly in `_get_all_directions()` |
| `HEX_DIRECTIONS`                  | `core.ts`                 | ⚠️ INLINE     | Defined directly in `_get_all_directions()` |

---

## 3. Type Parity Matrix

### 3.1 Core Types

| TS Type            | TS File   | Python Type   | Python File | Parity Status |
| ------------------ | --------- | ------------- | ----------- | ------------- |
| `Position`         | `game.ts` | `Position`    | `models.py` | ✅ MATCHED    |
| `BoardType`        | `game.ts` | `BoardType`   | `models.py` | ✅ MATCHED    |
| `BoardState`       | `game.ts` | `BoardState`  | `models.py` | ✅ MATCHED    |
| `BoardConfig`      | `game.ts` | `BoardConfig` | `core.py`   | ✅ MATCHED    |
| `RingStack`        | `game.ts` | `RingStack`   | `models.py` | ✅ MATCHED    |
| `MarkerInfo`       | `game.ts` | `MarkerInfo`  | `models.py` | ✅ MATCHED    |
| `MarkerType`       | `game.ts` | `str` literal | -           | ⚠️ PARTIAL    |
| `GameState`        | `game.ts` | `GameState`   | `models.py` | ✅ MATCHED    |
| `GameStatus`       | `game.ts` | `GameStatus`  | `models.py` | ✅ MATCHED    |
| `GamePhase`        | `game.ts` | `GamePhase`   | `models.py` | ✅ MATCHED    |
| `Player`           | `game.ts` | `Player`      | `models.py` | ✅ MATCHED    |
| `Move`             | `game.ts` | `Move`        | `models.py` | ✅ MATCHED    |
| `MoveType`         | `game.ts` | `MoveType`    | `models.py` | ✅ MATCHED    |
| `LineInfo`         | `game.ts` | `LineInfo`    | `models.py` | ✅ MATCHED    |
| `Territory`        | `game.ts` | `Territory`   | `models.py` | ✅ MATCHED    |
| `GameResult`       | `game.ts` | -             | -           | ⚠️ MISSING    |
| `PlayerChoice`     | `game.ts` | -             | -           | 🚫 NOT NEEDED |
| `ProgressSnapshot` | `game.ts` | `dict`        | -           | ⚠️ PARTIAL    |

### 3.2 Engine Types

| TS Type            | TS File    | Python Type | Python File     | Parity Status |
| ------------------ | ---------- | ----------- | --------------- | ------------- |
| `ValidationResult` | `types.ts` | `bool`      | -               | ⚠️ SIMPLIFIED |
| `GameAction`       | `types.ts` | `Move`      | `models.py`     | ✅ MATCHED    |
| `ActionType`       | `types.ts` | `MoveType`  | `models.py`     | ✅ MATCHED    |
| `Validator`        | `types.ts` | `Validator` | `interfaces.py` | ✅ MATCHED    |
| `Mutator`          | `types.ts` | `Mutator`   | `interfaces.py` | ✅ MATCHED    |

### 3.3 Domain Result Types

| TS Type             | TS File                 | Python Type         | Python File | Parity Status |
| ------------------- | ----------------------- | ------------------- | ----------- | ------------- |
| `TurnAdvanceResult` | `turnLogic.ts`          | -                   | -           | ⚠️ INLINE     |
| `PerTurnState`      | `turnLogic.ts`          | GameState fields    | `models.py` | ✅ MATCHED    |
| `VictoryResult`     | `victoryLogic.ts`       | GameState fields    | `models.py` | ✅ MATCHED    |
| `VictoryReason`     | `victoryLogic.ts`       | -                   | -           | ⚠️ MISSING    |
| `PlacementContext`  | `PlacementValidator.ts` | -                   | -           | ⚠️ INLINE     |
| `SimpleMoveTarget`  | `movementLogic.ts`      | `Move`              | `models.py` | ✅ MATCHED    |
| `ChainCaptureState` | `game.ts`               | `ChainCaptureState` | `models.py` | ✅ MATCHED    |

---

## 4. Behavioral Requirements

### 4.1 Capture Enumeration Parity

**Critical Requirement:** [`enumerateCaptureMoves()`](../src/shared/engine/captureLogic.ts) and [`GameEngine._get_capture_moves()`](../ai-service/app/game_engine.py:1594) must produce identical sets of legal capture moves.

**Verification Points:**

- Cap height comparison: `attacker.cap_height >= target.cap_height`
- Self-capture allowed (capturing own stack)
- Ray-walking termination on collapsed spaces
- Landing marker ownership validation
- Segment distance >= stack height

**Test Strategy:**

```json
{
  "fixture_type": "capture_enumeration",
  "input": { "state": "<GameState>", "player": 1 },
  "expected_output": { "moves": ["<Move>", ...] }
}
```

### 4.2 Movement Validation Parity

**Critical Requirement:** [`validateMovement()`](../src/shared/engine/validators/MovementValidator.ts) and [`MovementValidator.validate()`](../ai-service/app/rules/validators/movement.py:8) must agree on all inputs.

**Verification Points:**

- Minimum distance = stack height
- Path clear of stacks and collapsed spaces
- Landing on opponent marker forbidden
- Must-move constraint after placement
- Straight-line direction only

**Test Strategy:**

```json
{
  "fixture_type": "movement_validation",
  "input": { "state": "<GameState>", "move": "<Move>" },
  "expected_output": { "valid": true/false }
}
```

### 4.3 Line Detection Parity

**Critical Requirement:** [`findAllLines()`](../src/shared/engine/lineDetection.ts) and [`BoardManager.find_all_lines()`](../ai-service/app/board_manager.py) must detect identical lines.

**Verification Points:**

- Line length thresholds per board type (3 for square8, 4 for others)
- Overlength line detection
- Player ownership from marker ownership
- Axis-aligned direction iteration

### 4.4 Territory Detection Parity

**Critical Requirement:** [`findDisconnectedRegions()`](../src/shared/engine/territoryDetection.ts) and [`BoardManager.find_disconnected_regions()`](../ai-service/app/board_manager.py) must identify identical regions.

**Verification Points:**

- Region connectivity via player-owned markers
- Border marker identification
- Region processability (stack outside region required)

### 4.5 Victory Evaluation Parity

**Critical Requirement:** [`evaluateVictory()`](../src/shared/engine/victoryLogic.ts) and [`GameEngine._check_victory()`](../ai-service/app/game_engine.py:269) must produce identical results.

**Verification Points:**

- Ring elimination threshold (victory_threshold)
- Territory control threshold (territory_victory_threshold)
- Last-player-standing (R172) tracking
- Global stalemate tie-breaker ladder
- Rings-in-hand counted as eliminated for tie-breaking

### 4.6 Heuristic Training & Evaluation Sanity

While heuristic evaluation is **not** part of the canonical rules surface, the Python training/evaluation harness must behave in a way that meaningfully exercises different heuristic weight profiles and avoids degenerate “all policies look the same” failures.

**Critical Requirements:**

- The shared fitness evaluator [`evaluate_fitness`](../ai-service/scripts/run_cmaes_optimization.py) must:
  - Distinguish a strong baseline (`heuristic_v1_balanced`) from a clearly bad profile (e.g. all-zero weights) under identical conditions.
  - Correctly apply candidate weights to the `HeuristicAI` instances (no accidental reuse of baseline weights).
  - Expose per-evaluation diagnostics (wins, draws, losses, and candidate–baseline L2 distance) via `debug_hook` so plateau investigations can see whether distinct policies are being explored.
- `HeuristicAI` position evaluation must _depend_ on the active weight vector:
  - On a nontrivial mid-game state, evaluations for baseline vs all-zero (or similarly extreme) profiles must differ for the same player.

**Test Strategy:**

- `ai-service/tests/test_heuristic_training_evaluation.py`:
  - `test_evaluate_fitness_zero_profile_is_strictly_worse_than_baseline`:
    - Asserts baseline vs baseline evaluates to ≈0.5 with symmetric W/L and zero weight distance.
    - Asserts an all-zero profile loses decisively vs the same baseline and has non-zero `weight_l2`, confirming that the fitness harness both applies and distinguishes different weights.
  - `test_heuristic_ai_position_evaluation_depends_on_weights`:
    - Builds a mid-game Square8 state via the shared training helper (`create_initial_state` + baseline self-play).
    - Verifies that `HeuristicAI.get_evaluation_breakdown(state)["total"]` differs between baseline and all-zero profiles on the same state.
- `ai-service/tests/test_multi_start_evaluation.py`:
  - Smoke-tests the `eval_mode="multi-start"` path in `evaluate_fitness` using a small, temporary Square8 state pool written in JSONL format.
  - Confirms that the wiring between eval pools (`app/training/eval_pools.py`), the multi-start loop, and CLI-facing parameters (`--eval-mode`, `--state-pool-id`) is functional and yields baseline-vs-baseline fitness values in `[0.0, 1.0]`.
- `ai-service/tests/test_eval_randomness_integration.py`:
  - Integration-tests the `eval_randomness` parameter of `evaluate_fitness` in `scripts/run_cmaes_optimization.py`.
  - Asserts that with a fixed `seed`, both `eval_randomness=0.0` (fully deterministic evaluation) and small non-zero values (controlled stochastic tie-breaking) produce deterministic, repeatable fitness values.
  - Guards against regressions where hidden RNG sources (e.g. numpy, Python `random`) or refactors to tie-breaking introduce non-determinism into CMA-ES/GA evaluation despite fixed seeds.

These tests are not rules-parity checks, but they are treated as **required sanity guards** for the heuristic training stack. Any change to `evaluate_fitness`, `HeuristicAI`, or the weight-application path that would cause these tests to pass trivially (e.g. equal fitness or identical evaluations for very different profiles) must be investigated as a potential wiring or plateau-diagnostics regression.

---

## 5. Test Parity Strategy

### 5.1 Shared Test Fixtures

Primary cross-language rules parity is enforced via **contract vectors (v2)** stored in [`tests/fixtures/contract-vectors/v2/`](../tests/fixtures/contract-vectors/v2/).

- TypeScript runner: [`tests/contracts/contractVectorRunner.test.ts`](../tests/contracts/contractVectorRunner.test.ts)
- Python runner: [`ai-service/tests/contracts/test_contract_vectors.py`](../ai-service/tests/contracts/test_contract_vectors.py)
- Vector schemas: [`src/shared/engine/contracts/schemas.ts`](../src/shared/engine/contracts/schemas.ts) and [`src/shared/engine/contracts/serialization.ts`](../src/shared/engine/contracts/serialization.ts)
- JSON examples: [`tests/fixtures/contract-vectors/v2/README.md`](../tests/fixtures/contract-vectors/v2/README.md)

Legacy **trace-based fixtures** under [`tests/fixtures/rules-parity/`](../tests/fixtures/rules-parity/) are retained for historical debugging and seed/trace investigations (see `docs/PARITY_SEED_TRIAGE.md` and `RULES_SCENARIO_MATRIX.md`). They should not be treated as the primary spec for TS↔Python parity.

### 5.2 Deterministic Seed-Based Scenarios

For complex multi-move scenarios, use seeded RNG:

```python
# Python
import random
random.seed(42)
state = GameEngine.apply_move(state, ai.select_move(state))
```

```typescript
// TypeScript
import { createSeededRng } from '@shared/utils/rng';
const rng = createSeededRng(42);
const state = applyMove(state, ai.selectMove(state, rng));
```

### 5.3 Property-Based Testing for Equivalence

> **Status:** As of 2025-11-26 this section is aspirational; property-based suites are not yet wired into CI and should be treated as future work.

Use hypothesis (Python) and fast-check (TypeScript) for:

1. **State hash equivalence:** `hash_game_state(state) === hashGameState(state)`
2. **Move enumeration equivalence:** `set(py_moves) === set(ts_moves)`
3. **State transition equivalence:** `apply_move(state, move) === applyMove(state, move)`

### 5.4 Existing Parity & Sanity Tests

| Test Suite                                   | Location                      | Coverage                                                                                    |
| -------------------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------- |
| `contractVectorRunner.test.ts`               | `tests/contracts/`            | Contract vectors (TS): shared engine semantics → v2 contract vectors                        |
| `test_contract_vectors.py`                   | `ai-service/tests/contracts/` | Contract vectors (Python): TS↔Python rules parity on v2 vectors                            |
| `test_rules_parity_fixtures.py`              | `ai-service/tests/parity/`    | TS→Python trace/plateau parity using generated fixtures                                     |
| `test_ts_seed_plateau_snapshot_parity.py`    | `ai-service/tests/parity/`    | Seed plateau snapshot parity against TS engine                                              |
| `test_ai_plateau_progress.py`                | `ai-service/tests/parity/`    | AI plateau progress parity vs TS plateau traces                                             |
| `test_line_and_territory_scenario_parity.py` | `ai-service/tests/parity/`    | Focused line+territory scenario parity using TS-generated fixtures                          |
| `test_default_engine_equivalence.py`         | `ai-service/tests/rules/`     | Move application parity (DefaultRulesEngine vs GameEngine)                                  |
| `test_default_engine_flags.py`               | `ai-service/tests/rules/`     | Mutator shadow contracts / safety flags                                                     |
| `test_heuristic_training_evaluation.py`      | `ai-service/tests/`           | Heuristic training/evaluation sanity: fitness harness and HeuristicAI weight sensitivity    |
| `TraceFixtures.sharedEngineParity.test.ts`   | `tests/unit/`                 | Legacy trace replay parity (diagnostic; see `docs/PARITY_SEED_TRIAGE.md` for usage context) |

---

## 6. Maintenance Protocol

### 6.1 Change Detection

When modifying the TypeScript canonical engine:

1. **Update `CANONICAL_ENGINE_API.md`** if API signature changes
2. **Run parity test suite** to detect Python divergence
3. **Update Python implementation** if behavior changed
4. **Add new fixtures** for new functionality

### 6.2 Sync Workflow

```mermaid
flowchart TD
    A[TS Engine Change] --> B{API Change?}
    B -->|Yes| C[Update CANONICAL_ENGINE_API.md]
    B -->|No| D[Run Parity Tests]
    C --> D
    D --> E{Tests Pass?}
    E -->|Yes| F[Done]
    E -->|No| G[Update Python Implementation]
    G --> H[Add/Update Fixtures]
    H --> D
```

### 6.3 CI Integration

The following CI checks enforce rules and parity guarantees (see [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)):

1. **`test` (Jest umbrella):** Runs the full TS Jest suite, including shared-engine unit tests and `tests/contracts/contractVectorRunner.test.ts`.
2. **`ts-rules-engine`:** Runs `npm run test:ts-rules-engine`, focusing on rules-level suites and shared-engine invariants.
3. **`python-core`:** Runs `python -m pytest` over `ai-service/tests` excluding `tests/parity/`, covering core Python rules/AI tests (including `test_heuristic_training_evaluation.py`).
4. **`python-rules-parity`:** Generates TS→Python rules-parity fixtures via `tests/scripts/generate_rules_parity_fixtures.ts`, then runs `ai-service/tests/parity/test_rules_parity_fixtures.py` under `pytest`.

### 6.4 Documentation Requirements

For each TypeScript engine change:

| Change Type       | Documentation Required                 |
| ----------------- | -------------------------------------- |
| New export        | Update this document + API doc         |
| Signature change  | Update this document + migration guide |
| Behavioral change | Update this document + add fixtures    |
| Bug fix           | Add regression fixture                 |

---

## 7. Parity Summary

### 7.1 Overall Statistics

| Category            | MATCHED | PARTIAL | MISSING | NOT NEEDED |
| ------------------- | ------- | ------- | ------- | ---------- |
| Core Utilities      | 13      | 0       | 2       | 0          |
| Placement           | 5       | 0       | 0       | 0          |
| Movement            | 2       | 0       | 0       | 0          |
| Capture             | 2       | 0       | 0       | 0          |
| Line                | 5       | 1       | 0       | 0          |
| Territory           | 8       | 0       | 0       | 0          |
| Victory             | 2       | 0       | 0       | 0          |
| Turn                | 1       | 0       | 0       | 0          |
| Mutators            | 6       | 0       | 0       | 0          |
| Validators          | 5       | 0       | 0       | 0          |
| **Total Functions** | **49**  | **1**   | **2**   | **3**      |

### 7.2 Type Parity Summary

| Category        | MATCHED | PARTIAL | MISSING | NOT NEEDED |
| --------------- | ------- | ------- | ------- | ---------- |
| Core Types      | 14      | 2       | 1       | 1          |
| Engine Types    | 4       | 1       | 0       | 0          |
| Domain Types    | 4       | 1       | 2       | 0          |
| **Total Types** | **22**  | **4**   | **3**   | **1**      |

### 7.3 Key Findings

1. **Strong Parity:** 49 of 52 tracked functions have MATCHED parity status
2. **Shadow Contract Pattern:** Python mutators delegate to `GameEngine` static methods, ensuring semantic alignment
3. **Primary Risk:** `BoardManager` and `GameEngine` static methods in Python are manually synced; changes require vigilance
4. **Missing Types:** `VictoryReason`, `GameResult` are not explicitly typed in Python (inline/simplified)

### 7.4 Recommendations

1. **Code Generation:** Consider generating Python from TypeScript AST for perfect sync
2. **WASM Compilation:** Long-term option to compile TS engine to WASM for Python consumption
3. **Fixture Expansion:** Add fixtures for edge cases discovered during AI training
4. **Automated Diffing:** Implement AST-based diffing to detect undocumented TS changes

---

## Appendix A: File Inventory

### TypeScript Canonical Engine (`src/shared/engine/`)

| File                          | Lines | Purpose                    |
| ----------------------------- | ----- | -------------------------- |
| `index.ts`                    | ~350  | Public API exports         |
| `core.ts`                     | ~650  | Core utilities             |
| `turnLogic.ts`                | ~250  | Turn/phase state machine   |
| `movementLogic.ts`            | ~200  | Movement reachability      |
| `captureLogic.ts`             | ~350  | Capture enumeration        |
| `lineDetection.ts`            | ~200  | Line geometry              |
| `lineDecisionHelpers.ts`      | ~400  | Line decision moves        |
| `territoryDetection.ts`       | ~250  | Territory region detection |
| `territoryProcessing.ts`      | ~300  | Territory collapse         |
| `territoryDecisionHelpers.ts` | ~400  | Territory decision moves   |
| `victoryLogic.ts`             | ~150  | Victory evaluation         |
| `validators/*.ts`             | ~600  | Move validation            |
| `mutators/*.ts`               | ~800  | State mutation             |

### Python AI Service (`ai-service/app/rules/`)

| File                | Lines | Purpose                        |
| ------------------- | ----- | ------------------------------ |
| `__init__.py`       | ~10   | Package exports                |
| `interfaces.py`     | ~55   | Protocol definitions           |
| `core.py`           | ~245  | Core utilities                 |
| `geometry.py`       | ~190  | Board geometry                 |
| `default_engine.py` | ~994  | Shadow contract engine         |
| `factory.py`        | ~20   | Engine factory                 |
| `validators/*.py`   | ~350  | Move validation                |
| `mutators/*.py`     | ~100  | State mutation (thin wrappers) |

### Python Game Engine (`ai-service/app/`)

| File               | Lines  | Purpose                |
| ------------------ | ------ | ---------------------- |
| `game_engine.py`   | ~3,186 | Canonical Python rules |
| `board_manager.py` | ~600   | Board state operations |

---

## Appendix B: Shadow Contract Verification

The [`DefaultRulesEngine.apply_move()`](../ai-service/app/rules/default_engine.py:285) method implements per-move shadow contracts:

```python
def apply_move(self, state: GameState, move: Move) -> GameState:
    # 1. Canonical result from GameEngine
    next_via_engine = GameEngine.apply_move(state, move)

    # 2. Shadow contract for each move type
    if move.type == MoveType.PLACE_RING:
        mutator_state = state.model_copy(deep=True)
        PlacementMutator().apply(mutator_state, move)
        # Assert board/player parity...

    # 3. Always return canonical state
    return next_via_engine
```

This ensures that:

- Mutators are validated against the canonical engine
- Any divergence raises `RuntimeError` with diagnostic details
- The canonical `GameEngine` result is always authoritative
