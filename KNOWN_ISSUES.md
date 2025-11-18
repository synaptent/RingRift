# Known Issues & Bugs

**Last Updated:** November 15, 2025  
**Status:** Code-verified assessment based on actual implementation  
**Related Documents:** [CURRENT_STATE_ASSESSMENT.md](./CURRENT_STATE_ASSESSMENT.md) · [TODO.md](./TODO.md) · [STRATEGIC_ROADMAP.md](./STRATEGIC_ROADMAP.md) · [CODEBASE_EVALUATION.md](./CODEBASE_EVALUATION.md)

This document tracks **current, code-verified issues** in the RingRift codebase.

Earlier versions of this file (and some older architecture docs) described the
marker system, BoardState, movement validation, phase transitions, territory
disconnection, and the PlayerChoice/chain-capture system as "not implemented".
Those statements are now obsolete:

- BoardState, markers, collapsed spaces, and stack data structures are fully
  implemented and used throughout the engine.
- Movement, overtaking captures, line formation/collapse, territory
disconnection, forced elimination, and hex boards are all implemented and
  generally aligned with `ringrift_complete_rules.md`.
- The PlayerChoice layer (shared types + PlayerInteractionManager +
  WebSocketInteractionHandler + AIInteractionHandler + DelegatingInteractionHandler)
  is wired into GameEngine for all rule-driven decisions (line order,
  line reward, ring/cap elimination, region order, capture direction), and
  is exercised in both human and AI flows.
- AI turns in backend games are driven through `globalAIEngine` and the Python
  AI service via `AIServiceClient`.

The remaining issues are primarily about **coverage, UX, and integration depth**
rather than missing core mechanics.

---

## 🔴 P0 – Critical Issues (Correctness & Confidence)

### P0.1 – Incomplete Scenario Test Coverage for Rules & FAQ

**Component(s):** GameEngine, RuleEngine, BoardManager, tests  
**Severity:** Critical for long-term confidence  
**Status:** Core behaviours covered by focused tests; many rule/FAQ scenarios still untested

**What’s implemented and tested:**

- Board topology, adjacency, distance, and basic territory disconnection are
  covered by unit tests (including both square and hex boards).
- Movement and capture validation (distance ≥ stack height, path blocking,
  landing rules) have focused tests in `tests/unit/RuleEngine.movementCapture.test.ts`.
- Chain capture enforcement and capture-direction choices are exercised by
  `GameEngine.chainCapture.test.ts`,
  `GameEngine.chainCaptureChoiceIntegration.test.ts`, and
  `GameEngine.captureDirectionChoiceWebSocketIntegration.test.ts`.
- Territory disconnection and self-elimination flows are validated by
  `BoardManager.territoryDisconnection*.test.ts` and
  `GameEngine.territoryDisconnection*.test.ts`.
- PlayerInteractionManager, WebSocketInteractionHandler, AIInteractionHandler,
  AIEngine/AIServiceClient and various choice flows
  (`line_reward_option`, `ring_elimination`, `region_order`) have unit and
  integration tests (`AIEngine.serviceClient.test.ts`,
  `AIInteractionHandler.test.ts`, `GameEngine.lineRewardChoiceAIService.integration.test.ts`,
  `GameEngine.lineRewardChoiceWebSocketIntegration.test.ts`,
  `GameEngine.regionOrderChoiceIntegration.test.ts`, etc.).

**What’s still missing:**

- A **systematic scenario suite** derived from `ringrift_complete_rules.md` and
  the FAQ (Q1–Q24) that:
  - Encodes each emblematic capture/chain example (including 180° reversals and
    cyclic patterns) as a test fixture.
  - Covers complex line + territory interactions and late-game territory
    victories across all board types.
  - Exercises forced elimination, stalemates, and corner/edge cases.
- Broader coverage across all turn phases and choice sequences, especially
  when multiple choices occur in a single turn (e.g. chain → line → territory
  with several PlayerChoices in between).
- Clear, per-module coverage targets and CI-enforced minimums for the rules
  axis (BoardManager/RuleEngine/GameEngine).

**Impact:**

The engine behaves correctly in many targeted scenarios, and integration tests
confirm that the PlayerChoice and AI boundaries are wired, but we cannot yet
claim **exhaustive** rules/FAQ coverage. Refactors still carry risk,
especially in less-tested corners of the rules.

**Planned direction (see TODO.md / STRATEGIC_ROADMAP.md):**

- Build a rules/FAQ scenario test matrix keyed to sections and FAQ numbers.
- Group tests along the four axes (rules/state, AI boundary, WebSocket/game
  loop, UI integration) so targeted runs are easy.
- Raise coverage thresholds per axis once baseline suites are in place.

### P0.2 – Backend ↔ Sandbox Semantic Trace Parity Gaps

**Component(s):** GameEngine, ClientSandboxEngine, trace utilities, AI turn logic  
**Severity:** Critical for engine correctness/debuggability  
**Status:** Trace infrastructure in place; several semantic mismatches remain

**What’s implemented and working:**

- Canonical trace types (`GameHistoryEntry`, `GameTrace`) defined in
  `src/shared/types/game.ts` and used across backend and sandbox.
- Shared trace helpers in `tests/utils/traces.ts`:
  - `runSandboxAITrace` – generates sandbox AI-vs-AI traces from
    `ClientSandboxEngine`.
  - `replayTraceOnBackend` – rebuilds a backend `GameEngine` from
    `trace.initialState` and replays canonical moves using
    `findMatchingBackendMove`.
  - `replayTraceOnSandbox` – replays canonical moves through a fresh
    `ClientSandboxEngine`.
- Backend replay now calls `engine.stepAutomaticPhasesForTesting()` between
  moves, so internal `line_processing` / `territory_processing` phases no
  longer stall replay.
- Diagnostic env vars and logging:
  - `RINGRIFT_TRACE_DEBUG=1` – writes sandbox opening sequences and
    backend mismatch snapshots to `logs/ai/trace-parity.log`.
  - `RINGRIFT_AI_DEBUG=1` – mirrors AI/trace diagnostics to the console.
- Parity/debug suites exist and are wired into Jest:
  - `Backend_vs_Sandbox.traceParity.test.ts`
  - `Sandbox_vs_Backend.seed5.traceDebug.test.ts`
  - `Backend_vs_Sandbox.aiParallelDebug.test.ts`

**What’s going wrong:**

Even with phase-stepping and trace structure fixes, several **semantic
mismatches** remain between backend and sandbox traces:

- In some AI-generated positions (e.g. certain `square8` seeds such as seed 5
  in `Sandbox_vs_Backend.seed5.traceDebug.test.ts`), the sandbox emits
  `overtaking_capture` moves where the backend never enumerates a matching
  capture from the same position.
- In other positions, the sandbox AI attempts `place_ring` while the backend
  is already in a capture phase with only `overtaking_capture` moves legal.
  From the backend’s perspective, the attempted placement is illegal given
  the current phase and rules.
- These discrepancies appear **after** internal-phase replay issues were fixed,
  so they reflect genuine differences in how the sandbox AI and backend
  enumerate/apply moves and transitions between phases, not simply replay
  harness problems.

**Impact:**

- The parity suites above intermittently fail, even though the S-invariant,
  hashing, and trace logging are wired correctly.
- Debugging backend bugs via sandbox AI traces is harder than it should be
  because some failures are caused by sandbox semantics diverging from the
  backend rather than by true backend rule bugs.
- Until these semantic gaps are closed, trace-based diagnostics must always
  distinguish **infrastructure issues** (now mostly resolved) from
  **sandbox/backend semantic differences**.

**Planned direction (see TODO.md Phase 1E / Sandbox Stage 2):**

- Unify all sandbox canonical mutations through a single, well-structured
  path in `ClientSandboxEngine`:
  - Treat `applyCanonicalMoveInternal` as the single source of truth for
    applying canonical `Move` objects (place_ring, skip_placement,
    move_stack/move_ring, overtaking_capture).
  - Refactor `maybeRunAITurn` so that **every** AI action is expressed as a
    canonical `Move` and routed through the same mutation + history pipeline
    used by `applyCanonicalMove`, instead of using bespoke mutation logic.
- Harden backend replay helpers:
  - Introduce a generic `replayMovesOnBackend(initialConfig, moves: Move[]):
    GameTrace` helper that builds a backend `GameEngine`, finds matching
    backend moves via `findMatchingBackendMove`, and produces a backend
    `GameTrace` suitable for parity analysis.
  - Treat `replayTraceOnBackend` / `replayMovesOnBackend` as canonical
    backend replay APIs used by parity tests and future tooling.
- Use the existing parity suites plus the new logging to iteratively close
  semantic gaps:
  - When a parity failure occurs, confirm first that the backend has a
    genuinely matching legal move for the sandbox action (or vice versa).
  - Where semantics diverge (e.g. missing backend overtaking captures, phase
    transitions that disagree), treat that as a P0/P1 engine bug and fix the
    underlying rules/phase logic rather than patching tests.

---

## 🟠 P1 – High-Priority Issues (UX, AI, Multiplayer)

### P1.1 – Frontend UX & Sandbox Harness Still Early

**Component(s):** React client (BoardView, GamePage, GameContext, ChoiceDialog)  
**Severity:** High for player experience  
**Status:** Functional but minimal

**Current capabilities:**

- `BoardView` renders 8×8, 19×19, and hex boards with improved contrast and a
  simple stack widget.
- `computeBoardMovementGrid(BoardState)` plus an SVG movement-grid overlay draw
  faint movement lines and node dots for both square and hex boards; this
  provides a **canonical geometric foundation** for future visual features.
- `GamePage` supports:
  - Backend-driven games (`/game/:gameId`) using `GameContext` and WebSockets
    to receive `GameState`, surface `pendingChoice`, and submit moves and
    `PlayerChoiceResponse`s.
  - A sandbox entry route (`/sandbox`) that currently acts as a frontend entry
    point and, in Stage 1, can create a backend game and hand off to the
    backend-driven view (see TODO.md for details).
- `ChoiceDialog` renders all PlayerChoice variants and is wired to
  `GameContext.respondToChoice`, so humans can answer line-reward,
  ring-elimination, region-order, and capture-direction prompts in
  backend-driven games.

**Missing / rough areas:**

- HUD and status:
  - No fully fleshed-out per-player HUD with ring/territory counts,
    AI profile info, and timers.
  - Limited visibility into current phase, pending choice type, and choice
    deadlines.
  - No in-UI move/choice history log for debugging and teaching.
- Sandbox harness (Stage 2):
  - The backend-wrapped Stage 1 sandbox harness exists (create game then
    navigate to `/game/:gameId`), but a **true client-local GameEngine
    harness** has not yet been implemented.
  - Local sandbox interactions are still simple and not fully rules-aligned.
- End-of-game flows:
  - Victory/defeat summary and post-game analysis UI are not yet implemented.

**Impact:**

Developers and early testers can play backend-driven games and exercise
PlayerChoices, but the experience is still developer-centric and not yet ready
for a wider, non-technical audience.

**Planned direction:**

- Implement a richer HUD (current phase, current player, ring/territory
  statistics, AI profile, timers).
- Build Stage 2: a client-local GameEngine sandbox harness that reuses
  `BoardView`, the movement grid, and the PlayerChoice layer for
  rules-aligned local experimentation.
- Add move/choice history and better visual cues for chain captures,
  line/territory processing, and forced elimination.

---

### P1.2 – WebSocket Game Loop: Lobby/Reconnection/Spectators Incomplete

**Component(s):** `src/server/websocket/server.ts`, GameContext, client pages  
**Severity:** High for robust multiplayer  
**Status:** Core loop present; broader lifecycle incomplete

**Current capabilities:**

- Backend-driven games use Socket.IO to:
  - Receive and broadcast `GameState` updates.
  - Relay `player_choice_required` and `player_choice_response` events through
    `WebSocketInteractionHandler` and GameContext/ChoiceDialog.
  - Orchestrate AI turns via `WebSocketServer.maybePerformAITurn`, which calls
    `globalAIEngine.getAIMove` and feeds moves into `GameEngine.makeMove`.
- There are focused integration tests for
  WebSocket-backed choice flows and AI turns.

**Missing / incomplete:**

- Robust lobby/matchmaking flows (listing/joining games, matchmaking, private
  games) in both server and client.
- Reconnection and resynchronization semantics (what happens when a client
  disconnects and rejoins mid-game).
- Spectator support (joining games read-only, receiving updates, leaving).
- A single, authoritative doc that lays out the full WebSocket turn/choice
  lifecycle as observed by the client.

**Impact:**

Backend-driven single games work well enough for development and testing, but a
full multiplayer UX with lobbies, reconnection, and spectators is not yet
available.

**Planned direction:**

- Document and implement the canonical WebSocket event flow for a turn,
  including AI turns and PlayerChoices.
- Build lobby, reconnection, and spectator features atop the existing
  WebSocket + GameContext foundation.

---

### P1.3 – AI Boundary: Service-Backed Choices Limited, Advanced Tactics Not Yet Implemented

**Component(s):** `ai-service/app/main.py`, `AIServiceClient.ts`, `AIEngine.ts`, `AIInteractionHandler.ts`  
**Severity:** High for long-term AI strength  
**Status:** Moves and several PlayerChoices service-backed; others local-only; no deep search yet

**Current capabilities:**

- Python FastAPI AI service (`ai-service/`) exposes:
  - `/ai/move` – move selection.
  - `/ai/evaluate` – position evaluation.
  - `/ai/choice/line_reward_option` – selects between Option 1 and 2 using a
    simple but explicit heuristic.
  - `/ai/choice/ring_elimination` – selects an elimination target based on
    smallest capHeight and totalHeight.
  - `/ai/choice/region_order` – chooses a region based on size and local
    enemy stack context.
- TypeScript AI boundary:
  - `AIServiceClient` implements typed clients for all of the above
    endpoints.
  - `AIEngine` exposes `getAIMove`, `getLineRewardChoice`,
    `getRingEliminationChoice`, and `getRegionOrderChoice`, mapping shared
    `AIProfile`/`AITacticType` onto the service.
  - `AIInteractionHandler` delegates `line_reward_option`, `ring_elimination`,
    and `region_order` choices to `globalAIEngine` when configured, with
    robust fallbacks to local heuristics on error.
  - Integration tests (e.g.
    `GameEngine.lineRewardChoiceAIService.integration.test.ts`,
    `AIEngine.serviceClient.test.ts`, `AIInteractionHandler.test.ts`,
    `GameEngine.regionOrderChoiceIntegration.test.ts`) exercise these paths,
    including failure modes for `line_reward_option`.

**Still limited:**

- `line_order` and `capture_direction` choices are currently answered via
  local heuristics in `AIInteractionHandler` and do not yet consult the
  Python service.
- AI does not yet use deep search (minimax/MCTS) or long-term planning; the
  Python side is still based on random and heuristic engines.
- There is limited observability around AI choice latency and errors beyond
  logs.

**Impact:**

The AI boundary is healthy and exercised for moves and several PlayerChoices,
which is enough for meaningful single-player games and testing. However, AI
strength is still limited, and advanced tactics will require future Python-side
work (stronger heuristics, search/ML) plus potentially additional endpoints.

**Planned direction:**

- Treat the current service-backed choices as the baseline and consider
  extending service coverage to line ordering and capture direction where
  helpful.
- Add metrics around AI service calls (latency, error rates, fallback counts)
  to guide future improvements.
- Incrementally introduce deeper search or ML-based agents on the Python side
  behind the existing endpoints.

---

## 🟢 P2 – Medium-Priority Issues (Persistence, Ops, Polish)

### P2.1 – Database Integration for Games/Replays Incomplete

**Component(s):** Prisma schema, game routes/services  
**Status:** Schema present; many higher-level features not wired end-to-end

- Prisma schema defines users, games, moves, etc., but:
  - GameEngine/GameState are not yet fully persisted across restarts.
  - Move history, ratings, and replay views are not yet exposed in the UI.
- This limits long-term features like leaderboards, replays, and statistics.

**Planned:** Wire game lifecycle events into DB writes, then expose history and
leaderboards in the API/UI.

---

### P2.2 – Monitoring & Operational Observability Limited

**Component(s):** Backend services (Node + FastAPI), Docker/CI  
**Status:** Logging in place; metrics/traces minimal

- Winston logging exists on the Node side; FastAPI uses standard logging.
- There is no unified metrics/monitoring setup (Prometheus/Grafana/Sentry,
  etc.) configured in this repo.

**Planned:** Introduce basic metrics and error tracking once the core gameplay
loop and tests are further stabilised.

---

## 🕰️ Historical Issues (Resolved)

These issues have been addressed but are kept here for context:

- **Marker system & BoardState structure** – Now fully implemented with
  `stacks`, `markers`, and `collapsedSpaces`, and used consistently in rules
  and engine code.
- **Movement validation & unified landing rules** – Distance ≥ stack height,
  path blocking, marker interactions, and landing legality were fixed and are
  covered by focused RuleEngine tests.
- **Territory disconnection & self-elimination prerequisite** – Implemented in
  BoardManager + GameEngine, with dedicated tests for both square and hex
  boards.
- **Phase transitions & forced elimination** – GameEngine now follows the
  documented turn/phase sequence with forced elimination when a player is
  blocked with stacks but has no legal actions.
- **PlayerChoice system and chain capture enforcement** – Shared
  `PlayerChoice` types, PlayerInteractionManager, WebSocketInteractionHandler,
  AIInteractionHandler, DelegatingInteractionHandler, and GameEngine
  integration are in place; chain captures are enforced and capture-direction
  choices are driven through this layer for both humans and AI, with tests.
- **Rule Fix (Nov 15, 2025): Overtaking own stacks now allowed** – Players can
  now overtake their own stacks when cap height requirements are met. The
  same-player restriction was removed from `validateCaptureSegmentOnBoard` in
  `src/shared/engine/core.ts` and capture enumeration in
  `src/server/game/RuleEngine.ts`. Test coverage added in
  `tests/unit/RuleEngine.movementCapture.test.ts`.
- **Rule Fix (Nov 15, 2025): Placement validation enforces legal moves** –
  Ring placement now validates that the resulting position has at least one
  legal move or capture available. Implemented via
  `hasAnyLegalMoveOrCaptureFrom` helper in `src/server/game/RuleEngine.ts`
  with test coverage in `tests/unit/RuleEngine.movementCapture.test.ts`.

For a more narrative description of what works today vs what remains, see
[CURRENT_STATE_ASSESSMENT.md](./CURRENT_STATE_ASSESSMENT.md).
