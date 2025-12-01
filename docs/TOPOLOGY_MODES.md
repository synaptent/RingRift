# Topology Modes: square8, square19, hexagonal

> **Doc Status (2025-11-28): Active (topology overview, non-semantics)**  
> **Role:** Describe the supported board topologies (square8, square19, hexagonal), how they are represented in code, and where topology-aware helpers live. This doc is a **derived architecture view** only – it does **not** define rules semantics or victory conditions.
>
> **Upstream SSoTs:**  
> • **Rules semantics SSoT:** shared TS engine under `src/shared/engine/**` + contracts/vectors (`tests/fixtures/contract-vectors/v2/**`, `tests/contracts/contractVectorRunner.test.ts`, `ai-service/tests/contracts/test_contract_vectors.py`), and written rules (`ringrift_complete_rules.md`, `ringrift_compact_rules.md`, `RULES_CANONICAL_SPEC.md`).  
> • **Lifecycle/API SSoT:** `docs/CANONICAL_ENGINE_API.md` and shared types/schemas (`src/shared/types/game.ts`, `src/shared/engine/orchestration/types.ts`, `src/shared/types/websocket.ts`, `src/shared/validation/websocketSchemas.ts`).  
> • **Geometry helpers:** `src/shared/engine/core.ts` (movement directions, distance, paths), `src/shared/engine/movementLogic.ts`, `src/shared/engine/captureLogic.ts`, and their Python analogues under `ai-service/app/**`.
> • **Environment / rollout presets:** For how topology choices are combined with orchestrator rollout flags across environments (CI, staging, production), see the canonical env/phase presets table in `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` §8.1.1.
>
> If this document ever disagrees with those SSoTs or the shared engine tests, **code + tests win** and this file must be updated.

**Related docs:**

- `RULES_ENGINE_ARCHITECTURE.md` (section 5: invariants, termination, and geometry)
- `SHARED_ENGINE_CONSOLIDATION_PLAN.md` (Phase 1 capture + movement consolidation notes)
- `RULES_IMPLEMENTATION_MAPPING.md`, `RULES_SCENARIO_MATRIX.md`
- Test meta-docs: `tests/README.md`, `tests/TEST_LAYERS.md`, `tests/TEST_SUITE_PARITY_PLAN.md`

---

## 1. Overview: What “Topology Mode” Means

RingRift currently supports three board topologies:

- `square8` – 8×8 square board, Chebyshev (king-move) geometry.
- `square19` – 19×19 square board, same geometry at a larger scale.
- `hexagonal` – hex board using cube/axial coordinates on a finite-radius hex.

At runtime, the topology is selected via:

- `BoardType` enum in TS (`src/shared/types/game.ts`) and Python (`ai-service/app/models.py`).
- `BOARD_CONFIGS` in TS (`src/shared/engine/index.ts`), which map `BoardType` → size and geometry parameters.
- Database/config fields such as `Game.boardType` and frontend sandbox configuration.

Topology affects:

- Movement and capture reachability (directions, distances, ray-walk paths).
- Territory region detection and borders.
- Scenario/test fixtures that assume particular coordinates.

The **rules semantics** (placement, movement, capture, territory, victory) are expressed in a topology-agnostic way and specialised via shared helpers (directions, distance, paths) per board type.

---

## 2. Square Boards (square8, square19)

### 2.1 Coordinate System

- Represented as `(x, y)` with `0 ≤ x < size`, `0 ≤ y < size`.
- `BOARD_CONFIGS.square8.size = 8`, `BOARD_CONFIGS.square19.size = 19`.
- String keys use `positionToString({ x, y })` (e.g. `"3,5"`).

### 2.2 Movement Directions

- Directions are derived from `getMovementDirectionsForBoardType('square8' | 'square19')` in `src/shared/engine/core.ts`.
- For both square boards, adjacency is the **8-direction Moore neighbourhood**:
  - Orthogonal: N, S, E, W.
  - Diagonal: NE, NW, SE, SW.
- TS implementation:
  - `getMovementDirectionsForBoardType` returns `(dx, dy)` pairs for the 8 directions.
  - `getPathPositions(from, to)` walks along a chosen direction, yielding intermediate positions.
- Python analogue:
  - `BoardManager._get_all_directions` in `ai-service/app/board_manager.py`.
  - `GameEngine._get_path_positions` for straight-line paths.

These direction helpers are used by:

- `movementLogic.ts` (non-capture movement reachability).
- `captureLogic.ts` (capture ray-walks).
- Territory detection (`territoryDetection.ts`) when tracing borders/regions.

### 2.3 Distance and Path Blocking

- Distance:
  - Square boards use **Chebyshev distance**:
    - `distance(from, to) = max(|dx|, |dy|)`.
  - Implemented in TS via `calculateDistance('square8' | 'square19', from, to)` in `core.ts`.
  - Python analogue: `BoardGeometry.calculate_distance` in `ai-service/app/rules/geometry.py`.
- Path blocking:
  - `getPathPositions` yields all intermediate cells between `from` and `to`.
  - Movement and capture helpers require that each intermediate cell:
    - Is on-board.
    - Is not a collapsed space.
    - Contains no stack.
  - Markers **do not** block movement or capture rays; they are processed separately for marker-path effects.

### 2.4 Typical Usage

- `square8`:
  - Primary board for AI training and self-play.
  - Used heavily in plateau/AI parity tests (`ai-service/tests/parity/**`).
  - Many scenario tests under `tests/scenarios/` assume 8×8 coordinates.
- `square19`:
  - Larger board used for cyclic capture and territory stress tests.
  - See `tests/unit/GameEngine.cyclicCapture.scenarios.test.ts` and territory-focused scenario suites.

---

## 3. Hexagonal Board (hexagonal)

### 3.1 Coordinate System

- Represented in **cube coordinates** `(x, y, z)` with the invariant `x + y + z = 0`.
- Board radius `r` is derived from `BOARD_CONFIGS.hexagonal.size`:
  - Typical pattern: size = `2r + 1`, so radius = `size - 1`.
  - Valid positions satisfy `max(|x|, |y|, |z|) ≤ r`.
- String keys use `positionToString({ x, y, z })` (e.g. `"2,-1,-1"`).
- TS:
  - Hex positions are defined in `src/shared/types/game.ts`.
  - `getMovementDirectionsForBoardType('hexagonal')` yields the 6 cube-axis directions.
- Python:
  - `BoardManager._get_all_directions(BoardType.HEXAGONAL)` returns the same 6 directions.

### 3.2 Movement Directions and Distance

- Directions:
  - 6 unit vectors in cube coordinates, e.g.:
    - `(1, -1, 0)`, `(-1, 1, 0)`, `(1, 0, -1)`, `(-1, 0, 1)`, `(0, 1, -1)`, `(0, -1, 1)`.
  - Used for both movement and capture rays.
- Distance:
  - Hex boards use cube distance:
    - `distance(from, to) = (|dx| + |dy| + |dz|) / 2`.
  - Implemented in TS via `calculateDistance('hexagonal', from, to)` in `core.ts`.
  - Python analogue: `BoardGeometry.calculate_distance` when `BoardType.HEXAGONAL`.

### 3.3 Paths and Regions

- Paths:
  - `getPathPositions` steps along a chosen cube direction, yielding intermediate `(x, y, z)` cells.
  - Movement and capture apply the same path-blocking rules as square boards (no stacks/collapsed spaces on intermediate cells).
- Regions and borders:
  - Territory detection (`territoryDetection.ts` and Python `territory.py`) operate over the same cube coordinate space.
  - Tests such as `TerritoryParity.GameEngine_vs_Sandbox.test.ts` include hex cases.

### 3.4 Example Uses

- Cyclic capture stress tests:
  - `tests/unit/GameEngine.cyclicCapture.hex.scenarios.test.ts`.
  - Diagnostic chain-search tests using `sandboxCaptureSearch.findMaxCaptureChains`.
- AI plateau and invariants:
  - Some parity/plateau tests in `ai-service/tests/parity/**` run on hex to ensure geometry-general behaviour.

---

## 4. Where Topology Lives in Code

### 4.1 Shared TS Engine

- `src/shared/engine/core.ts`:
  - `getMovementDirectionsForBoardType(boardType)`.
  - `calculateDistance(boardType, from, to)`.
  - `getPathPositions(from, to)`.
- `src/shared/engine/movementLogic.ts`:
  - Uses directions and paths to compute non-capture movement reachability.
- `src/shared/engine/captureLogic.ts`:
  - Uses directions and paths to enumerate overtaking capture segments.
- `src/shared/engine/territoryDetection.ts`, `territoryBorders.ts`:
  - Use topology to find disconnected regions and border markers.

### 4.2 Hosts/Adapters

- Backend (`src/server/game`):
  - `BoardManager.ts` wraps topology helpers for server-side boards.
  - `GameEngine.ts`, `RuleEngine.ts` call into shared movement/capture/territory helpers.
- Client sandbox (`src/client/sandbox`):
  - `ClientSandboxEngine.ts` and helpers (`sandboxMovement.ts`, `sandboxCaptures.ts`, `sandboxTerritory.ts`) build board views and delegates to shared engine functions.
- Python (`ai-service/app`):
  - `board_manager.py` exposes direction lists, path helpers, and on-board checks.
  - `game_engine.py` uses those helpers for movement, capture, placement, and territory logic.

> When changing topology-related behaviour (directions, distance, paths), update the shared TS helpers first, then ensure all hosts/adapters call those helpers or their Python equivalents, and finally update or add tests in the parity and scenario suites referenced above.

---

## 5. Tests That Anchor Topology Behaviour

Non-exhaustive list of tests that rely on or exercise topology semantics:

- Movement and capture:
  - `tests/unit/MovementLogic.shared.test.ts`
  - `tests/unit/CaptureAggregate.shared.test.ts`
  - `tests/unit/GameEngine.cyclicCapture.scenarios.test.ts`
  - `tests/unit/GameEngine.cyclicCapture.hex.scenarios.test.ts`
- Territory and victory:
  - `tests/unit/TerritoryParity.GameEngine_vs_Sandbox.test.ts`
  - `tests/unit/TerritoryCore.GameEngine_vs_Sandbox.test.ts`
  - `tests/unit/GameEngine.territoryDisconnection.test.ts`
- Contract vectors:
  - `tests/contracts/contractVectorRunner.test.ts`
  - `tests/fixtures/contract-vectors/v2/*.json` (movement, capture, line, territory vectors)
- Python parity:
  - `ai-service/tests/parity/test_line_and_territory_scenario_parity.py`
  - `ai-service/tests/parity/test_chain_capture_parity.py`

These suites provide the practical guard-rails for topology changes. Any change to directions, distance, or path generation should be validated by running at least:

- `npm test -- --testPathPattern="MovementLogic.shared"`
- `npm test -- --testPathPattern="CaptureAggregate.shared"`
- `npm test -- --testPathPattern="GameEngine.cyclicCapture"`
- `pytest ai-service/tests/parity`
