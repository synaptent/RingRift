# RingRift Current State Assessment

**Assessment Date:** November 24, 2025
**Assessor:** Code + Test Review
**Purpose:** Factual status of the codebase as it exists today

> This document is the **Single Source of Truth** for the project's implementation status.
> It supersedes `IMPLEMENTATION_STATUS.md` and should be read together with:
>
> - `KNOWN_ISSUES.md` – P0/P1 issues and gaps
> - `TODO.md` – phase/task tracker
> - `STRATEGIC_ROADMAP.md` – phased roadmap to MVP

The intent here is accuracy, not optimism. When in doubt, the **code and tests** win over any percentage or label.

---

## 📊 Executive Summary

**Overall:** Strong architectural foundation and rules implementation; **not yet production-ready**.

- **Core Rules:** Movement, markers, captures (including chains), lines, territory, forced elimination, and victory are implemented once in the shared TypeScript rules engine under
  [`src/shared/engine`](src/shared/engine/types.ts:1) and reused by backend and sandbox hosts. These helpers are exercised by focused Jest suites.
- **Backend & Sandbox Hosts:** The backend `RuleEngine` / `GameEngine` and the client `ClientSandboxEngine` now act as thin adapters over the shared helpers, wiring in IO (WebSockets/HTTP, persistence, AI) while delegating movement, capture, line, territory, placement, victory, and turn sequencing to shared validators/mutators and geometry helpers.
- **Backend Play:** WebSocket-backed games work end-to-end, including AI turns via the Python service / local fallback and server-driven PlayerChoices surfaced to the client.
- **Session Management:** `GameSessionManager` and `GameSession` provide robust, lock-protected game state access.
- **Frontend:** The React client has a usable lobby, backend GamePage (board + HUD + victory modal), and a rich local sandbox harness.
- **Testing:** Solid around shared helpers, host parity, and several interaction paths, with a **rules/FAQ scenario matrix** in place
  ([`RULES_SCENARIO_MATRIX.md`](RULES_SCENARIO_MATRIX.md:1)) including dedicated FAQ suites under
  [`tests/scenarios/FAQ_*.test.ts`](tests/scenarios/FAQ_Q09_Q14.test.ts:1). Several seeded trace-parity suites and multi-host/AI fuzz harnesses remain **diagnostic or partial** rather than hard CI gates.
- **Gaps:** Multiplayer UX (spectators, reconnection UX, matchmaking, chat UI) and advanced AI are still **clearly incomplete**.

A reasonable label for the current state is: **engine/AI-focused beta suitable for developers and playtesters**, not for a broad public audience.

---

## ✅ Verified Implementation Status

### 1. Core Game Logic & Engines

- **BoardManager**
  - Supports 8×8, 19×19, and hex boards.
  - Position generation, adjacency, distance.
  - Delegates line detection and (where applicable) territory region detection to shared helpers such as
    [`lineDetection.ts`](src/shared/engine/lineDetection.ts:21),
    [`territoryDetection.ts`](src/shared/engine/territoryDetection.ts:36), and
    [`territoryBorders.ts`](src/shared/engine/territoryBorders.ts:35), keeping geometry consistent across backend, sandbox, and shared tests.
- **RuleEngine**
  - Movement validation (distance ≥ stack height, blocking, landing rules) via shared validators like
    [`MovementValidator.ts`](src/shared/engine/validators/MovementValidator.ts:1).
  - Capture validation, including overtaking and mandatory chains, via
    [`CaptureValidator.ts`](src/shared/engine/validators/CaptureValidator.ts:1).
  - Placement and territory validation wired through
    [`PlacementValidator.ts`](src/shared/engine/validators/PlacementValidator.ts:1) and
    [`TerritoryValidator.ts`](src/shared/engine/validators/TerritoryValidator.ts:1) so backend validation semantics match the shared helpers and sandbox engine.
  - Enumerates valid moves per phase as an adapter over shared core helpers (movement reachability, capture enumeration, territory processing, victory).
- **Shared rules engine (`src/shared/engine/`)**
  - Canonical `GameState` / `GameAction` types, validators, and mutators for all core mechanics.
  - Movement & captures:
    [`movementLogic.ts`](src/shared/engine/movementLogic.ts:1),
    [`captureLogic.ts`](src/shared/engine/captureLogic.ts:1),
    [`MovementMutator.ts`](src/shared/engine/mutators/MovementMutator.ts:1),
    [`CaptureMutator.ts`](src/shared/engine/mutators/CaptureMutator.ts:1).
  - Lines:
    [`lineDetection.ts`](src/shared/engine/lineDetection.ts:21),
    [`LineMutator.ts`](src/shared/engine/mutators/LineMutator.ts:1).
  - Territory geometry and processing:
    [`territoryDetection.ts`](src/shared/engine/territoryDetection.ts:36),
    [`territoryBorders.ts`](src/shared/engine/territoryBorders.ts:35),
    [`territoryProcessing.ts`](src/shared/engine/territoryProcessing.ts:1),
    [`TerritoryMutator.ts`](src/shared/engine/mutators/TerritoryMutator.ts:1).
  - Line & territory decision helpers:
    - [`lineDecisionHelpers.ts`](src/shared/engine/lineDecisionHelpers.ts:1) – enumerates and applies canonical `process_line` and `choose_line_reward` `Move`s and wires `pendingLineRewardElimination` in a host‑agnostic way.
    - [`territoryDecisionHelpers.ts`](src/shared/engine/territoryDecisionHelpers.ts:1) – enumerates and applies canonical `process_territory_region` and `eliminate_rings_from_stack` `Move`s, enforcing Q23 gating and per‑player elimination bookkeeping for both hosts.
  - Placement & no‑dead‑placement:
    [`PlacementValidator.ts`](src/shared/engine/validators/PlacementValidator.ts:1),
    [`PlacementMutator.ts`](src/shared/engine/mutators/PlacementMutator.ts:1),
    plus the no‑dead‑placement helper
    `hasAnyLegalMoveOrCaptureFromOnBoard` in
    [`core.ts`](src/shared/engine/core.ts:1) used by both backend and sandbox.
  - Victory & stalemate ladder:
    [`victoryLogic.ts`](src/shared/engine/victoryLogic.ts:51) with bare‑board stalemate handling shared between hosts.
  - Turn sequencing:
    [`turnLogic.ts`](src/shared/engine/turnLogic.ts:132) and the shared
    [`TurnMutator.ts`](src/shared/engine/mutators/TurnMutator.ts:1) capture the canonical phase/turn progression, which backend and sandbox hosts mirror.
  - Host usage:
    - Backend [`GameEngine.ts`](src/server/game/GameEngine.ts:1) and [`RuleEngine.ts`](src/server/game/RuleEngine.ts:1) call these helpers (including the decision helpers) to enumerate decision `Move`s such as `process_line`, `choose_line_reward`, `process_territory_region`, and `eliminate_rings_from_stack`, and then apply them by `moveId`.
    - [`ClientSandboxEngine.ts`](src/client/sandbox/ClientSandboxEngine.ts:1) and sandbox engines such as [`sandboxLinesEngine.ts`](src/client/sandbox/sandboxLinesEngine.ts:1) and [`sandboxTerritoryEngine.ts`](src/client/sandbox/sandboxTerritoryEngine.ts:1) are thin adapters over the same helpers for line and territory phases.
- **GameEngine (backend host)**
  - Orchestrates turn/phase loop: `ring_placement → movement → capture → chain_capture → line_processing → territory_processing → next player`.
  - Uses shared validators/mutators and helpers for movement, capture, line detection/processing, territory disconnection/processing, placement, and victory while also handling persistence, WebSocket events, and AI delegation.
  - Integrates `PlayerChoice` flows for line order/rewards, ring/cap elimination, region order, and capture direction via `PlayerInteractionManager`, always resolving choices into canonical `Move` objects applied through the shared rules surface.
- **ClientSandboxEngine & Sandbox Canonicalization**
  - Client-local sandbox engine acts as a thin host over the same shared helpers used by the backend for movement, chain capture, lines, territory, placement, turn sequencing, and victory.
  - Emits **canonical `Move` history** for both AI and human flows, suitable for trace replay and RulesMatrix/FAQ scenario validation.
  - Sandbox **chain-capture semantics** for human flows are explicitly tested via FAQ 15.3.1 scenario parity.
  - Shared RNG hooks threaded through local AI selection for both sandbox and backend, with injectable RNG for parity tests.

### 2. Backend Infrastructure

- **HTTP API**
  - Auth endpoints (`/api/auth`): register/login/refresh/logout.
  - Game endpoints (`/api/games`): create/list/join/leave, lobby listing.
  - User endpoints (`/api/users`): profile and stats basics.
- **WebSocket Server** (`src/server/websocket/server.ts`)
  - Authenticated Socket.IO server.
  - `join_game` / `leave_game` / `player_move` / `chat_message` / `player_choice_response`.
  - Auto-start logic: marks game `ACTIVE` when all players are ready.
  - AI turns via `maybePerformAITurn` using `globalAIEngine` and `AIServiceClient`.
  - **Victory signalling**: updates DB and emits `game_over` with `GameResult`.
- **Session Management** (`src/server/game/GameSessionManager.ts`)
  - Manages `GameSession` instances with distributed locking (Redis-backed).
  - Ensures atomic move processing and state updates.

### 3. AI Integration

- **Python AI Service** (`ai-service/`)
  - FastAPI service with Random, Heuristic, Minimax, and MCTS AI implementations.
  - Endpoints for move selection and position evaluation.
  - Difficulty-based AI type mapping (1-2: Random, 3-5: Heuristic, 6-8: Minimax, 9-10: MCTS).
- **TypeScript Boundary**
  - `AIServiceClient` and `AIEngine`/`globalAIEngine`.
  - `RulesBackendFacade` mediates between TS engine and Python service, supporting `shadow` and `python` (authoritative) modes.
  - Service-backed move selection and several PlayerChoices (`line_reward_option`, `ring_elimination`, `region_order`) with tested fallbacks to local heuristics.
  - Full AIProfile support with difficulty (1-10), mode (service/local_heuristic), and aiType overrides.
- **Game Creation with AI**
  - Backend API supports creating games with AI opponents via `aiOpponents` configuration.
  - AI games auto-start immediately without waiting for additional human players.
  - AI games are unrated by default (configurable).
  - GameSession automatically initializes AI players and triggers AI turn loop.
- **Integration Tests**
  - `FullGameFlow` integration test acts as a regression harness for the unified chain-capture / Move model and the S‑invariant.
  - AI turn execution tested through GameSession and WebSocketServer integration tests.

### 4. Frontend Client

- **LobbyPage**
  - Lists available games via `/games/lobby/available`.
  - Allows creating games with board type, maxPlayers, rated/private flags, time control, and **AI configuration**.
  - AI opponent controls: count (0-3), difficulty (1-10), mode (service/local_heuristic), AI type override.
  - Clear UI feedback for AI difficulty levels (Beginner/Intermediate/Advanced/Expert).
- **GamePage (Backend Mode)**
  - Connects via GameContext to WebSocket, receives `game_state` and `game_over`.
  - Renders `BoardView` for 8×8, 19×19, and hex boards.
  - Uses backend-provided `validMoves` for click‑to‑move.
  - Uses `ChoiceDialog` to render server-driven PlayerChoices.
  - Uses `VictoryModal` to show `gameResult` on `game_over`.
  - **AI opponent display**: Shows AI indicator badges and difficulty labels in game header.
- **GameHUD**
  - Displays current player, phase, and per-player ring/territory statistics.
  - **AI thinking indicators**: Animated dots when AI is making moves.
  - **AI difficulty badges**: Color-coded difficulty and AI type labels for each AI player.
- **Local Sandbox (`/sandbox`)**
  - `ClientSandboxEngine` + sandbox modules implement a fully local rules‑complete engine.
  - Supports mixed human/AI games with unified "place then move" turn semantics.

### 5. Testing & Parity Infrastructure

- **Configuration:** [`jest.config.js`](jest.config.js:1), [`tests/README.md`](tests/README.md:1).
- **Suites:**
  - Unit tests for `BoardManager`, `RuleEngine`, and backend `GameEngine`.
  - ClientSandboxEngine tests for parity and local victory conditions.
  - PlayerInteractionManager, WebSocketInteractionHandler, AIInteractionHandler tests.
  - AIEngine/AIServiceClient tests for success/failure/fallback paths.
  - WebSocketServer integration tests.
- **Shared-helper rules suites (TypeScript engine):**
  - Movement & captures:
    - [`movement.shared.test.ts`](tests/unit/movement.shared.test.ts:1) – canonical non‑capturing movement reachability and integration with the no‑dead‑placement helper `hasAnyLegalMoveOrCaptureFromOnBoard`.
    - [`captureLogic.shared.test.ts`](tests/unit/captureLogic.shared.test.ts:1) – overtaking capture enumeration and reachability over [`captureLogic.ts`](src/shared/engine/captureLogic.ts:1), including coverage for the same no‑dead‑placement helper.
    - [`captureSequenceEnumeration.test.ts`](tests/unit/captureSequenceEnumeration.test.ts:1) – legacy capture‑sequence enumeration harness kept as a regression/diagnostic suite.
    - [`RuleEngine.movement.scenarios.test.ts`](tests/unit/RuleEngine.movement.scenarios.test.ts:1) and [`RuleEngine.movementCapture.test.ts`](tests/unit/RuleEngine.movementCapture.test.ts:1) – backend adapter alignment with shared movement/capture helpers.
  - Lines:
    - [`lineDetection.shared.test.ts`](tests/unit/lineDetection.shared.test.ts:1) – shared marker‑line geometry.
    - [`lineDecisionHelpers.shared.test.ts`](tests/unit/lineDecisionHelpers.shared.test.ts:1) – canonical enumeration and application of line‑decision `Move`s (`process_line`, `choose_line_reward`) and reward options over the shared helpers.
    - [`LineDetectionParity.rules.test.ts`](tests/unit/LineDetectionParity.rules.test.ts:1),
      [`Seed14Move35LineParity.test.ts`](tests/unit/Seed14Move35LineParity.test.ts:1) – rules‑level geometry/regression guards for historically tricky line‑detection states.
  - Territory:
    - [`territoryBorders.shared.test.ts`](tests/unit/territoryBorders.shared.test.ts:1) – shared border‑marker expansion.
    - [`territoryProcessing.shared.test.ts`](tests/unit/territoryProcessing.shared.test.ts:1) – shared region‑processing pipeline (collapse + internal elimination).
    - [`territoryDecisionHelpers.shared.test.ts`](tests/unit/territoryDecisionHelpers.shared.test.ts:1) – canonical `process_territory_region` / `eliminate_rings_from_stack` decision semantics, including Q23 gating and elimination bookkeeping.
    - [`territoryProcessing.rules.test.ts`](tests/unit/territoryProcessing.rules.test.ts:1),
      [`sandboxTerritory.rules.test.ts`](tests/unit/sandboxTerritory.rules.test.ts:1),
      [`sandboxTerritoryEngine.rules.test.ts`](tests/unit/sandboxTerritoryEngine.rules.test.ts:1) – rules‑level suites for Q23, region collapse, and internal vs self‑elimination.
  - Placement:
    [`placement.shared.test.ts`](tests/unit/placement.shared.test.ts:1),
    [`RuleEngine.placementMultiRing.test.ts`](tests/unit/RuleEngine.placementMultiRing.test.ts:1).
  - Victory & termination:
    [`victory.shared.test.ts`](tests/unit/victory.shared.test.ts:1),
    [`SInvariant.seed17FinalBoard.test.ts`](tests/unit/SInvariant.seed17FinalBoard.test.ts:1),
    [`SharedMutators.invariants.test.ts`](tests/unit/SharedMutators.invariants.test.ts:1),
    [`GameEngine.turnSequence.scenarios.test.ts`](tests/unit/GameEngine.turnSequence.scenarios.test.ts:1),
    [`SandboxAI.ringPlacementNoopRegression.test.ts`](tests/unit/SandboxAI.ringPlacementNoopRegression.test.ts:1),
    [`GameEngine.aiSimulation.test.ts`](tests/unit/GameEngine.aiSimulation.test.ts:1).
- **Host parity suites (backend ↔ sandbox ↔ shared):**
  - Movement / capture / placement:
    [`MovementCaptureParity.RuleEngine_vs_Sandbox.test.ts`](tests/unit/MovementCaptureParity.RuleEngine_vs_Sandbox.test.ts:1),
    [`PlacementParity.RuleEngine_vs_Sandbox.test.ts`](tests/unit/PlacementParity.RuleEngine_vs_Sandbox.test.ts:1),
    [`movementReachabilityParity.test.ts`](tests/unit/movementReachabilityParity.test.ts:1),
    [`reachabilityParity.RuleEngine_vs_Sandbox.test.ts`](tests/unit/reachabilityParity.RuleEngine_vs_Sandbox.test.ts:1),
    [`ClientSandboxEngine.moveParity.test.ts`](tests/unit/ClientSandboxEngine.moveParity.test.ts:1).
  - Territory & borders:
    [`BoardManager.territoryDisconnection.square8.test.ts`](tests/unit/BoardManager.territoryDisconnection.square8.test.ts:1),
    [`BoardManager.territoryDisconnection.test.ts`](tests/unit/BoardManager.territoryDisconnection.test.ts:1),
    [`BoardManager.territoryDisconnection.hex.test.ts`](tests/unit/BoardManager.territoryDisconnection.hex.test.ts:1) – board‑level region‑detection geometry suites for `square8`, `square19`, and `hex` (primarily diagnostic/regression harnesses over the shared territory‑detection helpers).
    [`GameEngine.territoryDisconnection.test.ts`](tests/unit/GameEngine.territoryDisconnection.test.ts:1),
    [`GameEngine.territoryDisconnection.hex.test.ts`](tests/unit/GameEngine.territoryDisconnection.hex.test.ts:1) – backend host territory‑processing scenarios wired through the shared detection and decision helpers.
    [`TerritoryBorders.Backend_vs_Sandbox.test.ts`](tests/unit/TerritoryBorders.Backend_vs_Sandbox.test.ts:1),
    [`TerritoryCore.GameEngine_vs_Sandbox.test.ts`](tests/unit/TerritoryCore.GameEngine_vs_Sandbox.test.ts:1),
    [`TerritoryPendingFlag.GameEngine_vs_Sandbox.test.ts`](tests/unit/TerritoryPendingFlag.GameEngine_vs_Sandbox.test.ts:1) – backend↔sandbox parity on region borders, core processing, and pending‑territory flags.
    [`TerritoryParity.GameEngine_vs_Sandbox.test.ts`](tests/unit/TerritoryParity.GameEngine_vs_Sandbox.test.ts:1),
    [`TerritoryDecision.seed5Move45.parity.test.ts`](tests/unit/TerritoryDecision.seed5Move45.parity.test.ts:1),
    [`TerritoryDetection.seed5Move45.parity.test.ts`](tests/unit/TerritoryDetection.seed5Move45.parity.test.ts:1),
    [`TerritoryDecisions.GameEngine_vs_Sandbox.test.ts`](tests/unit/TerritoryDecisions.GameEngine_vs_Sandbox.test.ts:1) – heavy 19×19 parity and seed‑based diagnostics (**diagnostic, may be `describe.skip`**; canonical territory decision semantics now live in [`territoryDecisionHelpers.shared.test.ts`](tests/unit/territoryDecisionHelpers.shared.test.ts:1) plus the RulesMatrix/FAQ Q23 suites).
  - Victory:
    [`VictoryParity.RuleEngine_vs_Sandbox.test.ts`](tests/unit/VictoryParity.RuleEngine_vs_Sandbox.test.ts:1),
    [`GameEngine.victory.scenarios.test.ts`](tests/unit/GameEngine.victory.scenarios.test.ts:1),
    [`ClientSandboxEngine.victory.test.ts`](tests/unit/ClientSandboxEngine.victory.test.ts:1).
  - Shared vs host orchestration and traces:
    [`RefactoredEngine.test.ts`](tests/unit/RefactoredEngine.test.ts:1),
    [`RefactoredEngineParity.test.ts`](tests/unit/RefactoredEngineParity.test.ts:1),
    [`TraceFixtures.sharedEngineParity.test.ts`](tests/unit/TraceFixtures.sharedEngineParity.test.ts:1),
    the `Backend_vs_Sandbox.*.test.ts` and `Sandbox_vs_Backend.*.test.ts` trace/AI parity suites, and
    `Seed17*.GameEngine_vs_Sandbox.test.ts` geometry/territory checkpoints (see [`tests/README.md`](tests/README.md:431) for taxonomy).
- **Python ↔ TypeScript parity (rules service and fixtures):**
  - [`Python_vs_TS.traceParity.test.ts`](tests/unit/Python_vs_TS.traceParity.test.ts:1) exercising Python vs shared engine traces.
  - [`RulesBackendFacade.fixtureParity.test.ts`](tests/unit/RulesBackendFacade.fixtureParity.test.ts:1) and `ai-service/tests/parity/*` validating cross-language fixtures.
- **Scenario / matrix suites:**
  - RulesMatrix scenario coverage under `tests/scenarios/RulesMatrix.*.test.ts`, for example
    [`RulesMatrix.Territory.MiniRegion.test.ts`](tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts:1).
  - FAQ suites mapping Q1–Q24 to concrete positions and expected behaviour:
    [`FAQ_Q01_Q06.test.ts`](tests/scenarios/FAQ_Q01_Q06.test.ts:1),
    [`FAQ_Q07_Q08.test.ts`](tests/scenarios/FAQ_Q07_Q08.test.ts:1),
    [`FAQ_Q09_Q14.test.ts`](tests/scenarios/FAQ_Q09_Q14.test.ts:1),
    [`FAQ_Q15.test.ts`](tests/scenarios/FAQ_Q15.test.ts:1),
    [`FAQ_Q16_Q18.test.ts`](tests/scenarios/FAQ_Q16_Q18.test.ts:1),
    [`FAQ_Q19_Q21_Q24.test.ts`](tests/scenarios/FAQ_Q19_Q21_Q24.test.ts:1),
    [`FAQ_Q22_Q23.test.ts`](tests/scenarios/FAQ_Q22_Q23.test.ts:1).
  - The full mapping from rules/FAQ sections to scenario suites lives in
    [`RULES_SCENARIO_MATRIX.md`](RULES_SCENARIO_MATRIX.md:1).
- **Trace Parity:**
  - Trace and parity harnesses using `GameTrace` and
    [`tests/utils/traces.ts`](tests/utils/traces.ts:1) compare backend, sandbox, and shared engines on seeded games.
  - Backend FAQ-style chain-capture scenario suites pass under the unified `chain_capture` + `continue_capture_segment` model.
- **RNG & Determinism:**
  - Shared `SeededRNG` utility
    ([`src/shared/utils/rng.ts`](src/shared/utils/rng.ts:1)) and per-game `rngSeed` field in `GameState` for deterministic replay.
  - Determinism-focused Jest suites
    ([`RNGDeterminism.test.ts`](tests/unit/RNGDeterminism.test.ts:1),
    [`Sandbox_vs_Backend.aiRngParity.test.ts`](tests/unit/Sandbox_vs_Backend.aiRngParity.test.ts:1),
    `ai-service/tests/test_determinism.py`) and seeded trace helpers
    ([`tests/utils/traces.ts`](tests/utils/traces.ts:1)) for reproducible backend↔sandbox/AI parity.

---

## ❌ Major Gaps & Risks

### P0 – Confidence in Exhaustive Rules Coverage

- The **rules/FAQ scenario matrix** ([`RULES_SCENARIO_MATRIX.md`](RULES_SCENARIO_MATRIX.md:1)) and dedicated FAQ suites
  ([`tests/scenarios/FAQ_*.test.ts`](tests/scenarios/FAQ_Q09_Q14.test.ts:1)) now cover all FAQ questions (Q1–Q24) and a large set of high-value examples. Line and territory **decision phases** in particular are anchored by the shared-helper suites
  [`lineDecisionHelpers.shared.test.ts`](tests/unit/lineDecisionHelpers.shared.test.ts:1) and
  [`territoryDecisionHelpers.shared.test.ts`](tests/unit/territoryDecisionHelpers.shared.test.ts:1), plus targeted RulesMatrix territory scenarios such as
  [`RulesMatrix.Territory.MiniRegion.test.ts`](tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts:1). The remaining gap is exhaustive coverage of every composite diagram and multi-turn example in
  [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1), especially for deeply nested capture + line + territory chains.
- Some complex capture + line + territory combinations are still not encoded as focused rules-level tests and rely on seeded trace/simulation harnesses for coverage. When those harnesses uncover new behaviour, the expected fix is to (1) adjust the shared helpers (including the decision helpers) and (2) add or extend RulesMatrix/FAQ scenarios, rather than making host-local one-off patches.
- Backend↔sandbox semantic parity for canonical rules flows (movement, captures, lines, territory, placement, victory, and turn sequencing) is validated by shared-helper suites and targeted host parity suites (for example
  [`MovementCaptureParity.RuleEngine_vs_Sandbox.test.ts`](tests/unit/MovementCaptureParity.RuleEngine_vs_Sandbox.test.ts:1),
  [`PlacementParity.RuleEngine_vs_Sandbox.test.ts`](tests/unit/PlacementParity.RuleEngine_vs_Sandbox.test.ts:1),
  [`VictoryParity.RuleEngine_vs_Sandbox.test.ts`](tests/unit/VictoryParity.RuleEngine_vs_Sandbox.test.ts:1),
  and [`TraceFixtures.sharedEngineParity.test.ts`](tests/unit/TraceFixtures.sharedEngineParity.test.ts:1)). Heavy 19×19 territory parity and long-seed RNG/AI harnesses (for example
  [`TerritoryParity.GameEngine_vs_Sandbox.test.ts`](tests/unit/TerritoryParity.GameEngine_vs_Sandbox.test.ts:1) and
  [`Sandbox_vs_Backend.aiRngFullParity.test.ts`](tests/unit/Sandbox_vs_Backend.aiRngFullParity.test.ts:1)) are treated as **diagnostic** suites (some may be `describe.skip`) rather than hard CI gates; when they reveal divergences, those are tracked as engine bugs and converted into focused RulesMatrix + shared-helper tests.
  **NOTE:** Maintainers should periodically review which heavy parity suites remain skipped vs enforced in CI and ensure that any territory or line semantics they exercise are first-class in the decision-helper shared tests plus RulesMatrix scenario coverage.

### P1 – Multiplayer UX & Lifecycle Polish

- **Spectator Mode:** Basic UI support added, but no dedicated spectator browser.
- **Chat:** Basic in-game chat UI implemented, but backend integration pending persistence.
- **Reconnection:** Improved UX with banner, but complex reconnect/resync situations limited.
- **Matchmaking:** Limited to manually refreshed lobby list; no automated matching.

### P1 – AI Depth and Observability

- AI strength is moderate: Random and Heuristic engines are stable and used across all difficulties, and Minimax/MCTS/Descent variants are wired through the canonical 1–10 difficulty ladder and covered by targeted tests. However, there is not yet a deployed ML-trained policy network or aggressive think-time budget tuning, so AI play is geared toward development/playtesting rather than strong competitive opponents.
- Observability is primarily log-based: AI calls record latency, thinking time, evaluation scores, AI type, difficulty, and per-player service-failure/local-fallback counters, but there is no dedicated metrics pipeline or dashboard (e.g. Prometheus/Grafana) aggregating these signals across games/environments.

---

## 🎯 Recommended Immediate Focus

1.  **Rules/FAQ Scenario Suites**
    - Build a Jest suite matrix keyed to `ringrift_complete_rules.md` + FAQ.
    - Focus on complex chain captures, multi-line + territory turns, and hex-board edge cases.

2.  **Backend ↔ Sandbox Parity Hardening**
    - Use existing trace/parity harnesses to close remaining semantic gaps.
    - Treat any sandbox‑vs‑backend divergence on legal moves/phase transitions as a P0 engine issue.

3.  **Frontend Lifecycle Polish**
    - Tighten HUD (current player, phase, timers, progress indicators).
    - Improve reconnection UX and spectator views.

These themes are elaborated and broken into concrete tasks in `TODO.md` and `STRATEGIC_ROADMAP.md`.
