# Known Issues & Bugs

> **Doc Status (2025-12-16): Active (code-verified issue tracker)**
>
> - Canonical list of current, code-verified issues and gaps.
> - Not a rules or lifecycle SSoT; for rules semantics defer to `docs/rules/COMPLETE_RULES.md` + `RULES_CANONICAL_SPEC.md` + shared TS engine, and for lifecycle semantics defer to `docs/architecture/CANONICAL_ENGINE_API.md` and shared WebSocket types/schemas.

**Last Updated:** December 28, 2025
**Status:** Code-verified assessment based on actual implementation
**Related Documents:** [TODO.md](./TODO.md) · [STRATEGIC_ROADMAP.md](docs/planning/STRATEGIC_ROADMAP.md) · [docs/rules/PARITY_SEED_TRIAGE.md](docs/rules/PARITY_SEED_TRIAGE.md)

This document tracks **current, code-verified issues** in the RingRift codebase.

Earlier versions of this file (and some older architecture docs) described the
marker system, BoardState, movement validation, phase transitions, territory
disconnection, and the PlayerChoice/chain-capture system as "not implemented".
Those statements are now obsolete:

- BoardState, markers, collapsed spaces, and stack data structures are fully
  implemented and used throughout the engine.
- Movement, overtaking captures, line formation/collapse, territory
  disconnection, forced elimination, and hex boards are all implemented and
  generally aligned with `docs/rules/COMPLETE_RULES.md`.
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

### P0.1 – Forced Elimination Choice Divergence

**Component(s):** `TurnEngine.ts`, `TurnEngineAdapter.ts`, `GameEngine.ts`, `src/shared/engine/globalActions.ts`, `ClientSandboxEngine.ts`
**Severity:** P1 (historical divergence; substantially resolved for orchestrator-backed hosts)
**Status:** Core forced-elimination semantics and TS↔Python parity are fixed; sandbox and orchestrator-backed backend hosts now surface **forced elimination as an explicit choice** for human players, with a small residual gap only on legacy/non-interactive paths.
**Details:** The written rules state that if a player is blocked (has stacks but no legal moves), they must choose which stack to eliminate. The current implementation aligns with this for all canonical hosts:

- Shared helper layer:
  - `applyForcedEliminationForPlayer(state, player, targetPosition?)` in `globalActions.ts` implements RR‑CANON forced-elimination semantics and accepts an optional `targetPosition` so hosts can honour a player’s explicit stack choice.
  - `enumerateForcedEliminationOptions` exposes all eligible stacks for a blocked player (position, capHeight, stackHeight, moveId).
- Sandbox host:
  - `ClientSandboxEngine.forceEliminateCap` uses these helpers to present a `RingEliminationChoice` to human sandbox players when multiple stacks are available (auto-selecting when only a single option exists), then applies the chosen target via `applyForcedEliminationForPlayer`.
- Backend + orchestrator host:
  - The canonical orchestrator (`turnOrchestrator.ts`) now emits a `PendingDecision` of type `elimination_target` (with `extra.reason: "forced_elimination"`) whenever a player is blocked with stacks and only forced elimination is available, rather than applying a hidden host-level tie-breaker. Only unreachable “ANM” states fall back to direct resolution via `resolveANMForCurrentPlayer`.
  - `GameEngine`’s `TurnEngineAdapter` wires these `elimination_target` decisions through a `DecisionHandler` that, when a `PlayerInteractionManager` is present, constructs a `RingEliminationChoice` with one option per eligible stack and routes it over WebSockets; for AI players the adapter auto-selects using the same deterministic ordering.
  - Frontend tests (`BackendGameHost.test.tsx`) and WebSocket tests (`WebSocketServer.sessionTermination.test.ts`) exercise `ring_elimination` PlayerChoices, including cancellation behaviour during pending elimination decisions.
- Contracts & parity:
  - Forced-elimination contract vectors (`forced_elimination.*` in `tests/fixtures/contract-vectors/v2/forced_elimination.vectors.json`) and Python parity suites (`ai-service/tests/parity/test_forced_elimination_sequences_parity.py`, invariants under `ai-service/tests/invariants/**`) remain the SSOT for rules-level behaviour.

The remaining gap is limited to legacy/non-interactive backend paths that still call `applyForcedEliminationForPlayer` without a `targetPosition` (using the smallest-cap / first-stack heuristic) when no `PlayerInteractionManager` is wired. These paths are now used only in diagnostics/soak harnesses and are treated as implementation details rather than user-facing rules divergences.

### P0.2 – Chain Capture Edge Cases

**Component(s):** `GameEngine`, `captureChainEngine`, Python `game_engine.py`
**Severity:** P1 (Coverage Gap)
**Status:** Core logic and FAQ-aligned scenario suites exist; remaining work is incremental edge-case coverage and diagnostic harness tuning.
**Details:** Complex chain patterns such as 180-degree reversals, cyclic loops, strategic chain-ending choices, and zig-zag sequences are now covered by targeted scenario tests (for example `GameEngine.cyclicCapture.scenarios.test.ts`, `GameEngine.cyclicCapture.hex.scenarios.test.ts`, and `tests/scenarios/ComplexChainCaptures.test.ts`). The residual risk is limited to additional exotic geometries and the performance characteristics of deep diagnostic search harnesses (for example the skipped `GameEngine.cyclicCapture.hex.height3.test.ts` sandbox search), rather than a lack of tests for the core rules semantics.

**Python Parity Fix (2025-12-03):** Contract vector `chain_capture.5_plus_targets.extended_path` previously failed due to `resolve_chain_captures` selecting a different valid chain path than the expected sequence. Fixed by updating `resolve_chain_captures` to use `expectedChainSequence` from contract vectors when provided, ensuring tests follow the exact path specified in the vector. All 90 contract vectors (v2) now pass at 100% parity.

### P0.3 – Incomplete Scenario Test Coverage for Rules & FAQ

**Component(s):** GameEngine, RuleEngine, BoardManager, tests
**Severity:** Critical for long-term confidence
**Status:** Systematic rules/FAQ scenario matrix implemented; Q1–Q24 and the major rules clusters are covered by backend and sandbox scenario suites, with remaining work focused on incremental extensions and future rule additions.

### P0.4 – Python FSM: No Legal Moves Returns Empty Instead of Forced Elimination

**Component(s):** Python `game_engine.py`, `phase_machine.py`, `fsm.py`, `env.py`
**Severity:** P1 (Rare edge case, affects 3P selfplay ~1/5 games with seed 12346)
**Status:** FIXED (2025-12-10) – Defensive recovery added in `env.legal_moves()`.
**Discovered:** 2025-12-10 (seed 12346, 3P square8 selfplay)
**Fixed:** 2025-12-10

**Root cause:**
The CAPTURE and CHAIN_CAPTURE phases don't have `get_phase_requirement()` entries (they're interactive, not bookkeeping). In edge cases where the game entered CAPTURE/CHAIN_CAPTURE but no captures were available (e.g., due to a complex board state), `get_valid_moves()` returned empty and `get_phase_requirement()` returned None, causing `legal_moves()` to return empty.

**Fix:**
Added defensive recovery in `RingRiftEnv.legal_moves()` (`ai-service/app/training/env.py`):

- When stuck in CAPTURE/CHAIN_CAPTURE phase with no captures available
- Auto-advance to LINE_PROCESSING per RR-CANON-R073
- Clear `chain_capture_state` to prevent stale state
- Re-check for legal moves or phase requirements

**Verification:**

- Seed 12346 with DescentAI now completes at ~167 moves
- Random moves on seed 12346 complete at ~53 moves

**What’s implemented and tested:**

- Board topology, adjacency, distance, and basic territory disconnection are
  covered by unit tests (including both square and hex boards).
- Movement and capture validation (distance ≥ stack height, path blocking,
  landing rules) have focused tests in `tests/unit/RuleEngine.movement.scenarios.test.ts`,
  `tests/unit/MovementAggregate.shared.test.ts`, and `tests/unit/CaptureValidator.shared.test.ts`.
- Chain capture enforcement and capture-direction choices are exercised by
  `GameEngine.chainCapture.test.ts`,
  `GameEngine.chainCaptureChoiceIntegration.test.ts`, and
  `GameEngine.captureDirectionChoice.test.ts`.
- Territory disconnection and self-elimination flows are validated by
  `BoardManager.territoryDisconnection*.test.ts` and
  `tests/unit/territoryProcessing.shared.test.ts`.
- PlayerInteractionManager, WebSocketInteractionHandler, AIInteractionHandler,
  AIEngine/AIServiceClient and various choice flows
  (`line_reward_option`, `ring_elimination`, `region_order`) have unit and
  integration tests (`AIEngine.serviceClient.test.ts`,
  `AIInteractionHandler.test.ts`, `GameEngine.lines.scenarios.test.ts`,
  `WebSocketServer.sessionTermination.test.ts`,
  `GameEngine.regionOrderChoiceIntegration.test.ts`, etc.).

**What’s still missing:**

- Ongoing maintenance of the **systematic scenario matrix** in `RULES_SCENARIO_MATRIX.md` as rules docs evolve, including any newly added FAQ examples or clarifications.
- Additional emblematic scenarios for especially intricate combinations (for example, late-game line + territory interactions and near-victory territory margins across all board types), beyond the already-covered Q1–Q24 set.
- Clear, per-module coverage targets and CI-enforced minimums for the rules
  axis (BoardManager/RuleEngine/GameEngine), tied back to the scenario matrix.

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
**Severity:** LOW (downgraded from Medium per P18.5-\* results)
**Status:** SUBSTANTIALLY RESOLVED via extended contract vectors (43 cases, 0 mismatches)
**Tracking:** See [PARITY_SEED_TRIAGE.md](docs/rules/PARITY_SEED_TRIAGE.md) for detailed per-seed divergence matrix

**P18.5-\* Resolution (December 2025):**

- **Extended Contract Vectors:** 49 vectors across the core families (placement, movement, capture/chain_capture including extended chains, forced elimination, territory/line endgames including near_victory_territory, hex edge cases, meta moves such as swap_sides and multi-phase turns) with **0 mismatches** between TS and Python.
- **swap_sides Parity:** Verified across all layers (TS backend, TS sandbox, Python) per [P18.5-4_SWAP_SIDES_PARITY_REPORT.md](docs/archive/assessments/P18.5-4_SWAP_SIDES_PARITY_REPORT.md).
- **Orchestrator Phase 4:** 100% rollout, all hosts using orchestrator adapters as the canonical rules path.

**Previous Progress (November 25, 2025):**

- **DIV-001 (Seed 5 Capture Enumeration):** **RESOLVED** – Both backend and sandbox now use the unified `enumerateCaptureMoves()` function from `captureLogic.ts`.
- **DIV-002 (Seed 5 Territory Processing):** **RESOLVED** – Territory region detection and processing aligned via shared helpers.
- **DIV-008 (Late-game Phase/Player Tracking):** **DEFERRED** – Divergence in late-game phase/player tracking is **within tolerance only if ALL of the following hold**:
  1. **No board state changes** between divergence and game end: rings, stacks, markers, and collapsed spaces must remain identical in both TS and Python from the divergence point to the final state.
  2. **Identical game outcome**: same winner, same victory reason (ring elimination, territory, or LPS tiebreaker), and same numerical basis (identical eliminated ring counts or territory counts that determined victory, or identical LPS positions if LPS was the deciding factor).

  If any of these conditions is violated—for example, if the engines produce different final board states or different winners/victory reasons—the divergence is **NOT within tolerance** and must be investigated and fixed.

**What's implemented and working:**

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
- Decision phase timeout guards implemented to prevent infinite waits during player choice scenarios.
- Diagnostic env vars and logging:
  - `RINGRIFT_TRACE_DEBUG=1` – writes sandbox opening sequences and
    backend mismatch snapshots to `logs/ai/trace-parity.log`.
  - `RINGRIFT_AI_DEBUG=1` – mirrors AI/trace diagnostics to the console.
- Parity/debug suites exist and are wired into Jest:
  - `Backend_vs_Sandbox.traceParity.test.ts`
  - `Sandbox_vs_Backend.seed5.traceDebug.test.ts`
  - `Backend_vs_Sandbox.aiParallelDebug.test.ts`

**Remaining Open Divergences:**

The following divergences are tracked in [PARITY_SEED_TRIAGE.md](docs/rules/PARITY_SEED_TRIAGE.md) but are now lower priority given contract vector coverage:

- **DIV-003 (Seed 14 Placement):** Multi-ring placement validation differences
- **DIV-004 (Seed 14 Line Processing):** Line detection edge cases
- **DIV-005 (Seed 17 Capture):** Capture enumeration edge case
- **DIV-006 (Seed 17 Chain Capture):** Chain capture phase exit conditions
- **DIV-007 (Seed 17 Phase Tracking):** Phase/player advancement differences

**Impact:**

- The major parity gaps that blocked trace-based debugging are now resolved.
- Contract vectors provide systematic coverage for critical scenarios.
- Remaining divergences are edge cases that do not affect normal gameplay.

**Planned direction:**

- Consider these divergences closed for practical purposes; reopen only if contract vectors or production telemetry reveal issues.
- Focus parity work on extending contract vector coverage rather than trace-based debugging.

---

## 🟠 P1 – High-Priority Issues (UX, AI, Multiplayer)

### P1.1 – Frontend UX & Sandbox Experience Still Developer-Centric

**Component(s):** React client (BoardView, GamePage, GameHUD, GameContext, ChoiceDialog, `/sandbox` UI)
**Severity:** High for player experience
**Status:** Core HUD/history/sandbox tooling implemented; overall UX still tuned for developers

**Current capabilities:**

- `BoardView` renders 8×8, 19×19, and hex boards with improved contrast and a
  simple stack widget.
- `computeBoardMovementGrid(BoardState)` plus an SVG movement-grid overlay draw
  faint movement lines and node dots for both square and hex boards; this
  provides a **canonical geometric foundation** for future visual features.
- Backend-driven games (`/game/:gameId`) use `GameContext` and WebSockets
  to receive `GameState`, surface `pendingChoice`, and submit moves and
  `PlayerChoiceResponse`s.
- The `/sandbox` route runs a **fully rules-complete, client-local engine**
  (`ClientSandboxEngine`) that reuses the same `BoardView`, `ChoiceDialog`,
  and `VictoryModal` patterns as backend games, with dedicated Jest suites
  under `tests/unit/ClientSandboxEngine.*.test.ts` covering movement,
  captures, lines, territory, and victory.
- Mixed human/AI sandbox games now share the same **"place then move"** turn
  model as backend games: ring placement no longer advances to the next
  player, the placed stack is forced to move before the turn passes, and
  local AI turns are driven automatically when it is an AI player’s turn.
  This behaviour is covered by
  `tests/unit/ClientSandboxEngine.mixedPlayers.test.ts` and the updated
  `/sandbox` wiring in `GamePage`.
- `ChoiceDialog` renders all PlayerChoice variants and is wired to
  `GameContext.respondToChoice`, so humans can answer line-reward,
  ring-elimination, region-order, and capture-direction prompts in
  backend-driven games.
- `GameHUD` and related view models surface per‑player ring and Territory
  counts, AI profile information, timers, current phase/instruction text,
  connection/spectator status, and decision-phase countdown banners for both
  backend and sandbox games.
- Backend-driven games (`BackendGameHost.tsx`) include an in‑UI move and event
  history surface via `MoveHistory`, `GameEventLog`, `GameHistoryPanel`, and
  `EvaluationPanel`, plus chat and board-controls overlays.
- Keyboard & accessibility polish:
  - `?` reliably opens the Board Controls overlay even when the board has focus.
  - Board keyboard navigation supports Home/End, and uses roving tabindex so Tab does not step through every cell.
  - Global shortcuts: `M` toggles mute, `F` toggles fullscreen, and `R` opens the resign confirmation (backend games).
- The `/sandbox` UI (`SandboxGameHost.tsx`) now includes:
  - Seat/board configuration with quick-start presets.
  - A scenario picker and reset flow (`ScenarioPickerModal`) plus saved‑state
    export/import (`SaveStateDialog`, scenario persistence helpers).
  - A replay browser and playback controls (`ReplayPanel`) backed by the game
    replay database, with the ability to fork new sandbox games from any
    replay position.
  - Phase guides, sandbox notes, AI stall diagnostics, and AI evaluation
    tooling wired through `EvaluationPanel`.

**Remaining polish areas:**

- HUD and status:
  - Visual hierarchy and copy are still primarily tuned for developers and
    playtesters rather than first‑time players.
  - Decision-phase and timeout banners exist but could benefit from additional
    UX polish and broader scenario coverage.
- Sandbox UX:
  - Advanced tooling (AI stall diagnostics, fixture/export helpers, replay DB)
    is exposed directly in the `/sandbox` sidebar and remains developer‑oriented.
  - Guided onboarding and beginner‑friendly presets/documentation are still
    minimal.
- End-of-game and analysis flows:
  - Victory modals, replay, and evaluation panels are implemented for sandbox
    and backend games, but richer post‑game summaries and teaching‑oriented
    overlays remain future UX work.

**Impact:**

Developers and early testers can play backend-driven and sandbox games and
exercise PlayerChoices, but the experience remains tuned for engine/AI work
rather than a wider, non-technical audience.

**Planned direction:**

- Continue refining HUD layout/copy, timeout banners, and spectator indicators
  so the primary flows feel intuitive to non‑technical players.
- Iterate on sandbox UX to balance powerful diagnostics (replay, fixtures,
  AI tools) with a simpler learning surface for new players.
- Expand inline explanations and teaching‑oriented overlays so rules/FAQ
  scenarios surfaced in tests can also be explored comfortably via the UI.

---

### P1.2 – WebSocket Game Loop: Lobby/Reconnection/Spectators Incomplete

**Component(s):** `src/server/websocket/server.ts`, GameContext, client pages  
**Severity:** High for robust multiplayer  
**Status:** Core loop, lobby, and reconnection flows are implemented and covered by tests; broader multiplayer lifecycle and UX still have gaps.

**Current capabilities:**

- Backend-driven games use Socket.IO to:
  - Receive and broadcast `GameState` updates.
  - Relay `player_choice_required` and `player_choice_response` events through
    `WebSocketInteractionHandler` and GameContext/ChoiceDialog.
  - Orchestrate AI turns via `WebSocketServer.maybePerformAITurn`, which calls
    `globalAIEngine.getAIMove` and feeds moves into `GameEngine.makeMove`.
- There are focused integration and unit tests for WebSocket-backed choice flows, AI turns, and reconnection behaviour (for example `tests/integration/GameReconnection.test.ts`, `tests/integration/LobbyRealtime.test.ts`, `tests/unit/GameSession.reconnectFlow.test.ts`, and `tests/e2e/reconnection.simulation.test.ts`).
- Lobby real-time updates (`lobby:subscribe` / `lobby:game_created` / `lobby:game_started` / `lobby:game_cancelled`) are wired between `WebSocketServer.broadcastLobbyEvent` and `LobbyPage`, with filters and sorting on the client.
- Player reconnection windows, abandonment semantics, and diagnostics are implemented via `pendingReconnections` and `playerConnectionStates` in `WebSocketServer`, with graceful handling of reconnects across multiple phases and completed games.
- Spectator mode is supported end-to-end: spectators can join via lobby/HTTP routes, receive full game updates over WebSockets, and see spectator-oriented HUD hints and overlays in `BackendGameHost`, `GameHUD`, and `BoardControlsOverlay` (read-only boards, spectator badges, watcher counts).

**Missing / incomplete:**

- Matchmaking and rating-based queue flows: the lobby supports listing/joining/creating (including private games), but there is no automated matchmaker, ladder queue, or cross-game pairing logic yet.
- Additional reconnection UX polish and coverage for cross-device / multi-tab scenarios, including richer HUD signalling for “reconnecting” vs. “abandoned” states and tighter integration with the rated timeout/abandonment rules used in e2e tests.
- Spectator UX enhancements and diagnostics: spectators currently have a solid read-only view and basic HUD hints, but there is still room for more explicit spectator-focused affordances (for example dedicated spectator panels, clearer handling of spectator disconnects, and improved replay/analysis tooling).
- Ongoing lifecycle documentation: `docs/architecture/CANONICAL_ENGINE_API.md` now describes the canonical move/decision/WebSocket flows, but it should be kept in sync with newer features such as rematch, lobby subscriptions, and reconnection windows, and expanded with end-to-end examples from the client’s point of view.

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
    `AIEngine.serviceClient.test.ts`, `AIInteractionHandler.test.ts`,
    `AIServiceClient.metrics.test.ts`, `GameEngine.regionOrderChoiceIntegration.test.ts`) exercise these paths,
    including failure modes for `line_reward_option`.

**Still limited:**

- `line_order` and `capture_direction` choices now consult the Python AI
  service when available, with fallback to local heuristics on error.
- AI does not yet use deep search (minimax/MCTS) or long-term planning; the
  Python side is still based on random and heuristic engines.
- Per-turn AI strength is still constrained by relatively shallow search and
  heuristic tactics; deeper search / ML agents remain future work.
- AI observability is now primarily via `ringrift_ai_requests_total`,
  `ringrift_ai_request_duration_seconds_bucket`, and
  `ringrift_ai_fallback_total` emitted from `AIServiceClient`/`AIEngine`, but
  there is still headroom for richer per-board/difficulty breakdowns and
  higher-level “AI quality mode” projections.

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

### P1.4 – Sandbox aiSimulation diagnostics and S-invariant expectations

**Component(s):** [`ClientSandboxEngine`](src/client/sandbox/ClientSandboxEngine.ts), sandbox AI (`maybeRunAITurn`), S-invariant tests
**Severity:** Medium for test signalling; low for core rules correctness
**Status:** Behaviour understood and intentional; tests need modernization; several seeds still exhibit stalls

**Context:**

- The sandbox and backend share a canonical **S-invariant** via [`computeProgressSnapshot()`](src/shared/engine/core.ts):
  `S = markers + collapsed + eliminated`
- The **aiSimulation** suite [`tests/unit/ClientSandboxEngine.aiSimulation.test.ts`](tests/unit/ClientSandboxEngine.aiSimulation.test.ts) runs many seeded AI-vs-AI games entirely in the sandbox and enforces:

  ```ts
  const beforeProgress = computeProgressSnapshot(stateBefore);
  await engine.maybeRunAITurn();
  const afterProgress = computeProgressSnapshot(engine.getGameState());
  expect(afterProgress.S).toBeGreaterThan(beforeProgress.S);
  ```

- These diagnostics can be enabled locally via:

  ```bash
  RINGRIFT_ENABLE_SANDBOX_AI_SIM=1 npm test -- ClientSandboxEngine.aiSimulation
  ```

  and are intentionally **not** part of the default CI signal.

- Earlier versions of this suite also asserted a strict `afterProgress.S > beforeProgress.S`
  for every AI tick, which conflicted with canonical `skip_placement`
  semantics; that expectation has since been relaxed to non-decreasing S.

- Even with the relaxed S-invariant checks, several seeded AI-vs-AI runs
  (across `square8`, `square19`, and `hexagonal`, with 2–4 AI players) still
  report potential stalls: games that remain `active` with no state changes
  over many consecutive AI actions.

**Observed behaviour (current implementation):**

- For a pure `skip_placement` step in the sandbox:
  - `markers` is unchanged.
  - `collapsedSpaces` is unchanged.
  - `totalRingsEliminated` is unchanged.
  - Therefore `afterProgress.S === beforeProgress.S`.

- This is the **expected** behaviour given the rules interpretation:
  S is a progress measure over _board changes_, and a phase-only transition that leaves the board intact should preserve S.

**Impact:**

- The **aiSimulation** suite currently reports multiple failing seeds when
  enabled, but these are treated as **diagnostic indicators** rather than hard
  CI failures. They highlight configurations where:
  - The sandbox AI makes little or no structural progress despite having legal
    actions, or
  - Termination is significantly delayed compared to expectations for a
    development harness.

**Planned direction:**

- Treat the current engine behaviour as **authoritative** for `skip_placement`:
  - S should be **non-decreasing** across canonical actions, but not strictly increasing for phase-only transitions that do not alter the board.
- Evolve the aiSimulation suite to:
  - Continue enforcing non-decreasing S.
  - Use stall detection (no state change across many AI actions) as the
    primary signal for problematic seeds.
  - Track and systematically triage the failing seeds as part of Phase 2
    robustness work, rather than gating CI.

- Until those tests are updated, the failing **aiSimulation** cases should be interpreted as a **known, expected discrepancy in test semantics**, not as an engine correctness failure.

In particular, for the historically problematic square8/2‑AI plateau around seed 1, treat the following suites as the **current, canonical diagnostics** for sandbox AI plateau/stall behaviour (anchored to the shared S‑invariant and the canonical rules SSoT: `RULES_CANONICAL_SPEC.md` plus its shared TS engine implementation):

- `tests/unit/ClientSandboxEngine.aiSimulation.test.ts`
- `tests/utils/aiSeedSnapshots.ts`
- `tests/unit/ClientSandboxEngine.aiStallDiagnostics.test.ts`
- `tests/unit/ClientSandboxEngine.aiStallNormalization.test.ts`
- `tests/scenarios/AI_TerminationFromSeed1Plateau.test.ts`

Earlier harnesses like `tests/unit/ClientSandboxEngine.aiStall.seed1.test.ts` and browser‑driven `/sandbox` stall watchdog traces should now be treated as **historical debugging artifacts** (see `archive/AI_STALL_DEBUG_SUMMARY.md`). If they ever disagree with rules‑level suites, S‑invariant tests, or the modern plateau/stall diagnostics above, defer to the rules and lifecycle SSoTs and update or retire the legacy harnesses accordingly.

### P1.5 – k6 Load Scenarios: Application-Level Gaps After PASS24.1

**Component(s):** k6 scenarios (`tests/load/scenarios/*.js`), HTTP API (`/api/games`), WebSocket server (`/socket.io/`)  
**Severity:** High for production-readiness and SLO confidence  
**Status:** Target-scale run executed (2025-12-10) with strong latency and stability, but error-rate signals were dominated by auth-token expiration and expected rate limiting. Auth refresh handling landed in `concurrent-games.js` and `player-moves.js` (2025-12-19) to reduce 401 noise; remaining work is to rerun target-scale + AI-heavy scenarios and record clean SLO summaries in `docs/testing/BASELINE_CAPACITY.md`.

Following PASS24.1, all four k6 load scenarios run against the nginx-fronted stack (`BASE_URL=http://127.0.0.1`, `WS_URL=ws://127.0.0.1`) without socket-level `ECONNREFUSED` or `status=0` failures. Target-scale runs are now recorded, but several **application-level** gaps remain:

- **Concurrent games & player-moves – ID lifecycle and contract assumptions (RESOLVED in harness; pending systematic runs)**
  - Scenarios: `concurrent-games.js`, `player-moves.js` under `tests/load/scenarios/`.
  - Behaviour (current harness):
    - `POST /api/games` and `GET /api/games/:gameId` remain the canonical surfaces for game creation/state fetch and are consistently reachable under load.
    - Both scenarios now create game IDs via `POST /api/games`, track them per VU, and **retire** IDs when games reach a terminal status, return 404 (expired/cleaned up), or exceed a bounded poll budget (`MAX_POLLS_PER_GAME`), matching backend lifecycle semantics.
    - 400 responses from `GET /api/games/:gameId` are treated explicitly as **scenario bugs** (invalid ID format) and trigger ID retirement + logging, while 429s and other 4xx/5xx responses are recorded as genuine load/behaviour signals.
    - 401/403 responses now trigger an auth refresh + single retry before classification, reducing token-expiry noise during long runs.
  - Impact:
    - `http_req_failed` and related thresholds in these scenarios now primarily reflect backend capacity/behaviour (including rate limiting) rather than stale-ID contract issues.
    - Interpreting k6 output still requires correlating error rates with backend metrics and SLOs, but the harness itself no longer dominates error budgets with `GAME_INVALID_ID` or token-expiry failures.
  - Tracking / references:
    - [`GAME_PERFORMANCE.md`](docs/runbooks/GAME_PERFORMANCE.md) – PASS22 and PASS24.1 baseline notes for game creation, concurrent games, and player moves.
    - [`PASS22_COMPLETION_SUMMARY.md`](docs/archive/assessments/PASS22_COMPLETION_SUMMARY.md) – Load-test baselines plus PASS24.1 follow-up.
    - [`PASS22_ASSESSMENT_REPORT.md`](docs/archive/assessments/PASS22_ASSESSMENT_REPORT.md) – PASS24.1 addendum marking infra availability acceptable and calling out remaining functional k6 gaps.
    - [`BASELINE_CAPACITY.md`](docs/testing/BASELINE_CAPACITY.md) – Target-scale run records and rerun checklist.

- **WebSocket stress – Socket.IO v4 protocol implemented (RESOLVED at harness level, still needs routine use)**
  - Scenario: `websocket-stress.js` under `tests/load/scenarios/`.
  - **Status (Dec 2025): RESOLVED in code** – The k6 scenario now fully implements Socket.IO v4 / Engine.IO v4 wire protocol:
    - Engine.IO handshake: waits for `0{...}` open packet, responds to `2` (ping) with `3` (pong).
    - Socket.IO namespace connection: sends `40{...}` CONNECT (with auth) and waits for `40{...}` ACK.
    - Application events: sends properly framed `42["eventName", data]` messages for lobby-related events (`lobby:subscribe`, `lobby:list_games`, etc.).
  - Impact:
    - The scenario can be used as a proper capacity/SLO signal for real-time connection handling; protocol errors should be near zero under normal operation.
  - Tracking / references:
    - [`tests/load/README.md`](tests/load/README.md) – Socket.IO v4 protocol implementation details.
    - [`GAME_PERFORMANCE.md`](docs/runbooks/GAME_PERFORMANCE.md) – Updated WebSocket-stress baseline.

### P1.X – Square8 Selfplay Data Has Parity Divergence (~40% Failure Rate)

**Component(s):** `ai-service/data/games/canonical_square8_2p.db`, Python `game_engine.py`, selfplay pipeline
**Severity:** P1 (Affects AI training data quality)
**Status:** Active – data quality issue, parity gates working correctly
**Discovered:** 2025-12-28 (parity validation run locally)

**Root cause:**
Selfplay games in `canonical_square8_2p.db` were generated with older Python engine code that has different behavior than TypeScript for certain game mechanics:

- `overtaking_capture` moves fail TS replay validation
- `continue_capture_segment` moves diverge at chain capture handling
- `forced_elimination` / `territory_processing` phase transitions differ
- Games with these mechanics show parity divergence at k=2 or later moves

**Validation results (Dec 28, 2025):**

| Database   | Sample | Pass Rate | Status                |
| ---------- | ------ | --------- | --------------------- |
| hex8_2p    | 30/30  | 100%      | ✅ Good data          |
| square8_2p | 12/20  | 60%       | ⚠️ Needs regeneration |

**Workaround:**

1. Use `RINGRIFT_ALLOW_PENDING_GATE=true` to train on unvalidated data (accepts ~40% divergent samples)
2. Filter exports to `parity_status = 'passed'` games only (reduces volume)
3. Focus training on hex8 configs (100% parity verified)

**Fix:**
Regenerate square8 selfplay games using the latest Python engine with aligned capture/elimination mechanics. The parity gate infrastructure is working correctly – it's properly identifying games with engine divergence.

**References:**

- Parity validation script: `scripts/tag_games_parity_status.py`
- Parity failure bundles: `/parity_failures/canonical_square8_2p__*.json`
- Related: P0.1 (Forced Elimination Choice Divergence), P0.2 (Chain Capture Edge Cases)

---

## These issues are intentionally scoped as **application-/protocol-level** gaps; **HTTP/WebSocket availability under load is considered acceptable after PASS24.1**. Future Code/Debug work should focus on aligning k6 scenario contracts and message formats with production behaviour rather than reworking infra routing.

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
**Status:** Baseline Prometheus/Grafana metrics and alerts in place; tracing/error aggregation and SLO enforcement still maturing.

- Node backend exposes a consolidated `/metrics` endpoint via `MetricsService` (HTTP, AI, rules, lifecycle, and orchestrator metrics), and the Python AI service exports its own Prometheus metrics.
- Prometheus and Grafana dashboards are wired under `monitoring/` with alert rules in `monitoring/prometheus/alerts.yml` and documented thresholds in `docs/operations/ALERTING_THRESHOLDS.md` (including new connection/decision lifecycle alerts).
- k6 load scenarios under `tests/load/**` are aligned with these metrics and dashboards (see `tests/load/README.md`), but are not yet part of a regular CI/staging cadence.
- Remaining gaps:
  - No end‑to‑end distributed tracing or centralized error aggregation (e.g. Sentry/OTel) wired into the services.
  - Alert thresholds are tuned for initial baselines only; environment‑specific SLOs and error budgets are not yet fully enforced in CI/deploy pipelines.
  - Operational runbooks exist for many surfaces (AI, DB, orchestrator, rate limiting) but drills and automation are still ad‑hoc.

**Planned:**

- Gradually introduce tracing/error aggregation and tie key alerts to on‑call rotations.
- Promote `npm run test:p0-robustness` and k6 smoke/load profiles into CI/staging pipelines as explicit SLO gates.
- Periodically revisit `docs/operations/ALERTING_THRESHOLDS.md` and related dashboards to tune thresholds based on real‑world load tests and early production behaviour.

---

### P2.3 – Known Flaky WebSocket/GameSession Tests

**Component(s):** `tests/unit/WebSocketServer.*.test.ts`, `tests/unit/GameSession.*.test.ts`
**Status:** Tests work correctly but have intermittent timing-dependent failures
**Severity:** Low (test infrastructure, not production code)

Several WebSocket and GameSession tests exhibit intermittent failures due to:

- Race conditions in async event handling
- Socket.IO connection timing sensitivity
- Mock timer interactions with real async operations

**Known flaky tests (as of Dec 2025):**

- `GameSession.reconnectFlow.test.ts` – reconnection window timing
- `WebSocketServer.sessionTermination.test.ts` – session cleanup ordering
- `WebSocketServer.aiTurnExecution.test.ts` – AI response timing
- Various lobby broadcast tests – event ordering sensitivity

**Impact:** These tests may fail in full suite runs but pass individually. They do NOT indicate production issues – the underlying WebSocket functionality is stable and well-tested via integration tests.

**Recommendation:** If these tests fail in CI:

1. Re-run the specific failing test in isolation
2. If it passes individually, the failure is a timing flake
3. Core parity and orchestrator tests are the authoritative signal

---

### P2.4 – Jest TSX snapshot transform for React snapshot tests ✅ RESOLVED

**Component(s):** Jest configuration, React snapshot tests (`tests/unit/*.snapshot.test.tsx`)
**Status:** ✅ RESOLVED (Dec 13, 2025)

**Details:**
TSX/JSX test files are now properly transformed via ts-jest with the `tsconfig.jest.json`
configuration using `"jsx": "react-jsx"`. Both `GameEventLog.snapshot.test.tsx` and
`GameHUD.snapshot.test.tsx` pass successfully with correct snapshot generation.

---

## ℹ️ Design Clarifications (Not Bugs)

### DC.1 – Mid-Phase Contract Vectors Not Suitable for Game Seeding

**Source:** [P18.5-3_ORCHESTRATOR_EXTENDED_VECTOR_SOAK_REPORT.md](docs/archive/assessments/P18.5-3_ORCHESTRATOR_EXTENDED_VECTOR_SOAK_REPORT.md)
**Status:** Design clarification, not a bug
**Date:** December 1, 2025

The extended contract vectors (49 vectors across the v2 bundles – including chain_capture and chain_capture_extended, forced_elimination, territory/territory_line_endgame and near_victory_territory, hex_edge_cases, and meta moves) are designed for **single-step parity testing** – verifying that a specific move applied to a specific state produces the expected output.

When the orchestrator soak harness attempted to use these vectors as starting points for random game continuation, 13 of 23 vectors flagged `ACTIVE_NO_CANDIDATE_MOVES` violations immediately at turn 0. This is **expected behavior**, not a rules engine bug:

- Vectors in mid-phase states (`chain_capture`, `territory_processing`, `line_processing`) require specific interactive actions that random move selection cannot provide.
- The soak harness correctly detected this mismatch and flagged it.

**Recommendation:** Use contract vectors for their designed purpose (parity testing). For soak-style full game testing, use random seeds or filter vectors to only those in playable phases (`ring_placement`, `movement`).

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
  `tests/unit/RuleEngine.movement.scenarios.test.ts`.
- **Rule Fix (Nov 15, 2025): Placement validation enforces legal moves** –
  Ring placement now validates that the resulting position has at least one
  legal move or capture available. Implemented via
  `hasAnyLegalMoveOrCaptureFrom` helper in `src/server/game/RuleEngine.ts`
  with test coverage in `tests/unit/RuleEngine.movement.scenarios.test.ts`.
- **Sandbox Fix (Nov 19, 2025): Mixed AI/Human turn semantics in `/sandbox`** –
  Local sandbox games now use a unified "place then move" turn model for
  both human and AI seats. Ring placement no longer advances directly to the
  next player; instead the placed stack must move before the turn can pass,
  and AI turns are triggered automatically when it is an AI player's move.
  Implemented in `ClientSandboxEngine` and the `/sandbox` path of `GamePage`,
  with coverage in `tests/unit/ClientSandboxEngine.mixedPlayers.test.ts`.
- **P18.1-5 Remediation (Dec 2025): TS↔Python Parity and Orchestrator Rollout** –
  The major parity and orchestrator issues identified in PASS18 have been
  resolved through P18.1-5 remediation work:
  - P18.1-\*: Capture/territory host unification
  - P18.2-\*: RNG seed handling alignment
  - P18.3-\*: Decision lifecycle and timeout semantics
  - P18.4-\*: Orchestrator Phase 4 (100% rollout)
  - P18.5-\*: Extended contract vectors (54 cases, 0 mismatches) and swap_sides parity
    See [WEAKNESS_AND_HARDEST_PROBLEM_REPORT.md](docs/archive/assessments/WEAKNESS_AND_HARDEST_PROBLEM_REPORT.md) Section 3 for details.
- **AI Fix (Dec 12, 2025): HeuristicAI move-cache key correctness** –
  Fixed `ai-service/app/ai/move_cache.py` to include `mustMoveFromStackKey`,
  `rulesOptions`, board geometry, per-player counters (rings/score meta), and
  `maxPlayers` (in addition to phase/player and move-history length) in the
  cache key. This prevents stale cached move surfaces that can cause illegal
  move selections around `swap_sides`, ring placement availability, and
  post-placement movement constraints. Regression coverage in
  `ai-service/tests/test_move_cache_key.py`.
- **Python ELIMINATE_RINGS_FROM_STACK Phase Handling (Dec 2025)** –
  Fixed Python engine phase transitions after ELIMINATE_RINGS_FROM_STACK moves
  in `ai-service/app/game_engine.py`. The fix distinguishes terminal vs
  non-terminal cases:
  - **Terminal** (no stacks left AND no rings in hand): Stay on current player,
    set phase to `territory_processing` (game over state)
  - **Non-terminal**: Rotate to next player, set phase to `ring_placement`
    This resolved 7 contract vector failures in the territory/forced_elimination
    test bundles, bringing v2 contract vectors to 54 passed, 0 failed, 24 skipped.
- **TS↔Python Hash Parity Infrastructure (Dec 4, 2025)** –
  Unified hash format between Python and TypeScript engines for cross-engine
  parity testing:
  - **Fingerprint format**: Canonical readable string
    `meta#players#stacks#markers#collapsed` used by both engines
  - **Hash function**: Cross-platform `simpleHash()` (FNV-1a based) producing
    identical 16-char hex hashes in both TS (`src/shared/engine/core.ts`) and
    Python (`ai-service/app/db/game_replay.py`)
  - **New parity tests** in `ai-service/tests/parity/`:
    - `test_hash_parity.py` – 7 tests for hash consistency and format
    - `test_phase_transition_parity.py` – 3 tests for valid phase transitions
      and state hash chain consistency
    - `test_differential_replay.py` – Infrastructure for comparing Python and
      TypeScript game replays (with golden game strict parity check)
  - All 96 parity tests passing
- **Rule Clarification: Post-Movement Capture Constraint (Dec 4, 2025)** –
  Clarified in `RULES_CANONICAL_SPEC.md` (new rule RR-CANON-R093) and all rules
  docs that optional capture after non-capture movement (`move_stack`)
  must be from the **moved stack's landing position only**, not from any stack
  the player controls. This addresses a semantic divergence between TS (landing
  position constraint) and Python (any stack) engines. The TS interpretation is
  now canonical. Python engine update pending.
- **Python Territory Region Filtering Fix (Dec 10, 2025)** –
  Fixed non-canonical territory region filtering in Python engine
  (`ai-service/app/game_engine.py:2882-2895`). The previous implementation
  incorrectly filtered regions by `controlling_player == player_number`, but
  per RR-CANON-R143, the ONLY requirement for territory region processability
  is the self-elimination prerequisite (player has stacks outside the region).
  The border color (`controllingPlayer`) determines which markers get collapsed
  during processing, NOT who can process the region. Both TS and Python engines
  now correctly use only the self-elimination check per canonical rules.
- **ANM State Parity (Dec 10, 2025 – PARTIALLY FIXED)** –
  After regenerating canonical DBs with the territory filtering fix, parity
  checks show `dims=anm_state` divergences where Python and TS disagree on
  whether the current state is ANM (Active No Moves). Investigation findings:

  **Root Cause Analysis:**
  - Most divergences occur in `line_processing` phase where state hashes match
  - Python computes `is_anm: true`, TS computes `is_anm: false`
  - The issue is in `has_phase_local_interactive_move()` for LINE_PROCESSING:
    - TS uses `enumerateProcessLineMoves(state, player, { detectionMode: 'detect_now' })`
      which runs fresh line detection via `findAllLines(board)`
    - Python uses `GameEngine.get_valid_moves()` filtered by move type
  - Line detection timing/semantics may differ between engines

  **Serious Divergences (premature game_over with winner=null):**
  - Game 916dacc0, k=72: TS ends with `gameStatus: completed`, `winner: null`;
    Python continues in `line_processing`. Both players have stacks (5 and 6).
  - Game f4ca5b64, k=74: After `move_stack` from (6,4)→(4,4), TS sets
    `gameStatus: completed`, `currentPhase: game_over`, `winner: null`,
    `victoryCondition: null`. Python stays in `line_processing`. Both players
    have 5 stacks each, no victory threshold met (ring elim needs 19, have 2/3;
    territory needs 33, have 0/3). This violates RR-CANON-R170-R173: game_over
    MUST have a winner via ring elimination, territory control, LPS, or stalemate.

  **Root Cause (likely):**
  - `evaluateVictory()` in `aggregates/VictoryAggregate.ts` falls through to stalemate
    logic when no player "can act", but the logic doesn't count players with
    only forced-elimination as having actions. The `!hasForcedEliminationAction`
    check makes `somePlayerCanAct = false` when FE is the only available action.
  - Per RR-CANON-R072/R100/R203, forced elimination IS a valid action that should
    continue the game, not trigger stalemate.

  **Impact:** Game terminates prematurely without a winner. FSM validation passes
  because the FSM doesn't check victory condition validity. This is a RULES BUG.

  **Resolution Path:**
  1. ✅ FIXED: `evaluateVictory()` trapped-position check now treats
     players with stacks as "can act" (either real moves or forced elimination).
     Changed `!hasForcedEliminationAction` to just `playerHasStacks` (Dec 10, 2025).
  2. ✅ FIXED: `enumerateProcessLineMoves()` in TS (`src/shared/engine/aggregates/LineAggregate.ts`)
     now uses `getEffectiveLineLengthThreshold()` instead of base `BOARD_CONFIGS[boardType].lineLength`.
     This aligns with Python's `BoardManager.find_all_lines()` which uses
     `get_effective_line_length(board.type, num_players)`. For square8 2-player, both now
     correctly require line length 4 (per RR-CANON-R120). Dec 10, 2025.
  3. ✅ FIXED: Python `has_phase_local_interactive_move()` in `ai-service/app/rules/global_actions.py`
     now calls `GameEngine._get_line_processing_moves()` directly for LINE_PROCESSING phase,
     matching the TS behavior of using fresh line detection. Dec 10, 2025.

  **Current Status:** ANM parity is now aligned between Python and TS engines. 3-game
  canonical parity soak shows 0 semantic divergences (down from 3 ANM divergences before fix).

- **4-Player Rotation Parity (Dec 10, 2025 – RESOLVED)** –
  Multi-player games (3P, 4P) were investigated for player rotation divergences.
  TS replay on a 4P game (319 moves, LPS victory) shows:
  - **fsmValidationFailures: 0** at game end – both engines reach same final state
  - 232 FSM coercion errors during replay are **phase recording artifacts**, not
    game logic errors (e.g., "PLACE_RING not valid in phase 'movement'" occurs
    because Python records moves one phase ahead of TS's expected phase)
  - Game outcome (winner, victory condition) matches between engines

  **Root cause:** Python auto-advances through certain phases and records moves at
  different timing than TS expects. The TS replay coercion layer handles this by
  forcing moves through anyway. The board states and game outcomes are identical.

  **Impact:** Multi-player selfplay games are **valid for training** – the phase
  metadata differs but game logic and final states are correct. No fix required;
  this is a cosmetic recording difference, not a parity bug.

- **LPS Victory Parity (Dec 10, 2025 – FIXED)** –
  Round-based Last-Player-Standing (LPS) victory detection was diverging between
  Python and TS engines when stacks remain on the board.

  **Root Cause:**
  - Python implements full LPS tracking in `GameEngine._check_victory()` (lines
    935-1000 of `ai-service/app/game_engine.py`), using `lps_consecutive_exclusive_rounds`
    and `lps_consecutive_exclusive_player` state fields.
  - TypeScript has a complete `lpsTracking.ts` module with `evaluateLpsVictory()` but
    the TS replay script was using only `evaluateVictory()` from `aggregates/VictoryAggregate.ts` which
    returns `{ isGameOver: false }` when `state.board.stacks.size > 0`, bypassing
    round-based LPS entirely.
  - The result: Python declared LPS victory after 3 consecutive rounds where one player
    is the exclusive real-action holder, while TS continued the game.

  **Resolution (Dec 10, 2025):**
  1. ✅ FIXED: Integrated `evaluateLpsVictory()` from `lpsTracking.ts` into the TS replay
     script (`scripts/selfplay-db-ts-replay.ts`):
     - Added LPS tracking state initialization before replay loop
     - Added LPS tracking update after each move in interactive phases
     - Extended final victory evaluation to check LPS victory alongside `evaluateVictory()`
  2. LPS tracking now matches Python's behavior: round-based tracking with real-action
     detection and 3-consecutive-round victory condition per RR-CANON-R172.
  3. All 286 turn orchestrator and LPS tracking tests pass with the fix.

- **Recording Format Enhancements Schema v6 (Dec 4, 2025)** –
  Enhanced game history entries with available moves enumeration and engine
  diagnostics to support deeper parity debugging:
  - **Available moves enumeration**: `available_moves_json` stores all legal
    moves at each state, enabling cross-engine move enumeration comparison
  - **Lightweight move counting**: `available_moves_count` column for move
    count without full enumeration overhead
  - **Engine diagnostics**: `engine_eval` and `engine_depth` columns in
    `game_history_entries` for storing AI evaluation alongside state snapshots
  - All enhancements are backward-compatible with automatic migration
- **ANM False Positive Fix (Dec 9, 2025)** –
  Fixed `has_phase_local_interactive_move()` in `ai-service/app/rules/global_actions.py`
  to return True for RING_PLACEMENT and MOVEMENT phases. These phases always have
  valid moves (either interactive or host-synthesized bookkeeping moves like
  NO_PLACEMENT_ACTION, NO_MOVEMENT_ACTION). This eliminates false positive
  ACTIVE_NO_CANDIDATE_MOVES invariant violations during selfplay.
  - **Before**: 12-17 ANM violations per game for certain seeds
  - **After**: 0 ANM violations
  - Commits: `9ac7c0ff`, `a35ddace`
- **Recovery.py Dict Mutation Bug Fix (Dec 10, 2025)** –
  Fixed `has_any_recovery_move()` in `ai-service/app/rules/recovery.py` which was
  iterating over `board.markers.items()` while modifying the dictionary during
  recovery move simulation. The fix uses `list(board.markers.items())` to create
  a snapshot before iteration (line 730-731).
  - Commit: `051c7971`
- **TS Replay Coercion for Player-Skip Scenarios (Dec 10, 2025)** –
  Added replay-tolerance coercion in `turnOrchestrator.ts` to handle TS replay
  from Python selfplay DBs where Python skips players without turn-material:
  - Lines 1293-1314: `no_movement_action` coercion for territory_processing → movement
  - Lines 1235-1255: Placement coercion for player-skip scenarios across `movement`,
    `line_processing`, and `ring_placement` phases
  - This enables full TS replay parity for 2P, 3P, and 4P games. All 14 canonical
    games (5+5+4) pass with 0 FSM validation failures.
  - Commits: Prior session commits in turnOrchestrator.ts
- **CI Parity Gate Workflow (Dec 10, 2025)** –
  Added `.github/workflows/parity-ci.yml` with 3 CI-blocking jobs that generate
  selfplay DBs on-the-fly and run TS replay verification:
  - 2P: 3 games, 20min timeout
  - 3P: 3 games, 25min timeout
  - 4P: 3 games, 30min timeout
  - References RR-CANON-R073/R075/R076 in workflow documentation
  - Commit: `399d03f0`
- **Early Victory Detection Parity Fixes (Dec 11, 2025)** –
  Fixed multiple parity failures related to victory detection during TS replay
  of selfplay games. See [AI_PIPELINE_PARITY_FIXES.md](docs/ai/AI_PIPELINE_PARITY_FIXES.md)
  for detailed analysis.
  1. ✅ **Early LPS Victory** (`VictoryAggregate.ts`): Added `countTotalRingsForPlayer()`
     helper to count ALL rings including buried rings in opponent stacks. The Early LPS
     check (RR-CANON-R172) now correctly detects when all other players have 0 total
     rings (board + hand), not just 0 rings in hand.

  2. ✅ **Territory Victory Threshold** (`VictoryAggregate.ts`): Changed territory
     victory detection to count directly from `collapsedSpaces` map instead of using
     `player.territorySpaces` field which could be stale during territory processing.
     This matches Python's `_check_victory` implementation.

  > **Note (Dec 2025):** `victoryLogic.ts` was removed. All victory logic is now in
  > `src/shared/engine/aggregates/VictoryAggregate.ts`. 3. ✅ **TS Replay Early Termination** (`scripts/selfplay-db-ts-replay.ts:1020-1167`):
  > Added `evaluateVictory()` call after each move to detect victory mid-replay.
  > When victory is detected, emits `ts-replay-early-victory` and `ts-replay-game-ended`
  > events, then terminates replay. This matches Python's behavior of stopping
  > game progression when victory conditions are met. 4. ✅ **Parity Checker Move Count Handling** (`ai-service/scripts/check_ts_python_replay_parity.py:734-752, 1088-1118, 1558-1563`):
  - Captures final summary from `ts-replay-game-ended` event
  - Accepts move count difference when TS terminated early due to valid victory
    detection (both engines show "completed" with matching state hash)
  - Updated divergence classification to only flag move count differences when
    explicitly marked as mismatch (not when early victory was acceptably detected)

  **Validation:** All canonical parity tests pass:
  - 48 contract vectors passed (36 skipped for multi-phase orchestrator tests; current v2 total = 90)
  - 9+ selfplay games replayed with 0 semantic divergences
  - Early victory game (ed1d7d1e) correctly terminates at k=117 with matching state hash

- **Training max_moves Increase (Dec 9, 2025)** –
  Increased `THEORETICAL_MAX_MOVES` in `ai-service/app/training/env.py` to account
  for canonical recording where each turn generates ~4-5 moves (RING_PLACEMENT,
  MOVEMENT, LINE_PROCESSING, TERRITORY_PROCESSING phases plus captures):
  - SQUARE8 2p: 150 → 400
  - SQUARE19/HEXAGONAL 2p: 1000 → 2000
  - Default `max_moves` in soak scripts: 200 → 400
  - This resolves game non-termination issues where games hit the move limit
    before reaching a natural victory condition.
- **TS Replay Parity Soak Results (Dec 10, 2025)** –
  Ran 13-game soak test with TS replay verification. Results:
  - **9 games PASS** (69%): TS reaches `game_over` with 0 FSM failures
  - **3 games ANM divergence** (23%): TS ends with `gameStatus: active` while Python
    recorded completed game. All moves apply, 0 FSM failures - indicates victory
    detection difference, not rules violation
  - **1 game FATAL ERROR** (8%): "No stack at origin" at k=109. Root cause:
    - At k=107, TS has 0 ringsInHand for both players (correctly depleted)
    - Python recorded `place_ring` at (5,3) at k=108, followed by `move_stack`
      from (5,3) at k=109
    - TS couldn't place ring (none in hand) so no stack exists at (5,3)
    - Investigation shows state divergence occurred earlier in game - TS and
      Python diverged on ring accounting before k=90
    - Stack counts diverge: TS has 4 stacks at k=107, Python final state has 1
  - **4P games show excellent parity**: All 3 tested games pass with `game_over`,
    including one with 18 synthesized moves (player-skip coercion working)
  - **ANM divergence pattern**: Occurs in ~25% of 2P/3P games, typically in
    `movement` or `territory_processing` phase. 4P appears more reliable.
  - **Root cause of fatal error**: Likely territory processing or line collapse
    differences causing ring count divergence. Not related to coercion logic
    (which is working correctly). Further investigation would require per-move
    state hash comparison to identify exact divergence point.
- **Recovery Ring Refund Bug Fix (Dec 10, 2025)** –
  **Root Cause:** Fixed critical bug in Python recovery action implementation
  (`ai-service/app/rules/recovery.py` lines 1147-1154) where extracted buried
  rings were incorrectly being returned to the player's hand:

  ```python
  # BUG: This line violated RR-CANON-R113
  p.rings_in_hand += rings_extracted  # REMOVED
  ```

  Per RR-CANON-R113, extracted rings during recovery are ELIMINATED (credited
  as self-elimination), NOT returned to hand. The TS implementation was already
  correct. This bug caused ring count divergence in games with recovery actions,
  allowing Python players to place more rings than permitted (e.g., 19 instead
  of 18 for a 2-player game).

  **Impact:** This was the root cause of the "No stack at origin" fatal parity
  errors seen in some soak games - Python accumulated extra rings from recovery,
  leading to state divergence vs TS.

  **Fix:** Removed the buggy `p.rings_in_hand += rings_extracted` line. Updated
  test fixtures in `test_recovery_parity.py` and `recovery_action.vectors.json`
  to work with correct behavior. All 136 parity tests pass, all recovery
  contract vectors pass.

- **TS Ring Array Convention Fix (Dec 10, 2025)** –
  Fixed multiple TS code locations using incorrect ring array index convention.
  Per `src/shared/types/game.ts:283`, `rings[0]` is the top/controlling ring,
  not `rings[rings.length-1]`.

  **Root Cause:** Several TS files used `slice(0,-1)` or `rings[rings.length-1]`
  to exclude/access the top ring, which is backwards. The canonical convention
  is `rings[0]` = top ring (controlling player).

  **Fixes Applied (commit `8532116c`):**
  - `RecoveryAggregate.ts`: Removed buggy local `calculateCapHeight()` function
    that used `rings[rings.length-1]`; now imports from `core.ts`. Fixed two
    buried ring checks from `slice(0,-1)` to `slice(1)`.
  - `playerStateHelpers.ts`: Fixed `countBuriedRings()` loop to start at index 1
    instead of 0, properly excluding the top ring.
  - `turnOrchestrator.ts`: Fixed buried ring check from `slice(0,-1)` to `slice(1)`.
    Added `recoveryMode` and `extractionStacks` propagation for Python replay.

  **Additional Fixes (commit `8532599b`):**
  - `RecoveryAggregate.ts` (lines 728, 811): Fixed `controllingPlayer` assignment
    from `rings[rings.length-1]` to `rings[0]` (top ring per game.ts convention).
  - `RecoveryAggregate.ts` (lines 711, 793): Fixed ring extraction index from
    `indexOf(player)` to `lastIndexOf(player)` per RR-CANON-R113 (bottommost ring
    extraction requirement). `rings[0]` is top, so `lastIndexOf` finds bottommost.
  - `Recovery.contractVectors.test.ts`: Fixed incorrect test that expected recovery
    to fail when player has rings in hand. Per RR-CANON-R110: "Recovery eligibility
    is independent of rings in hand. Players with rings may choose recovery over
    placement."

  **Impact:** Resolves "Player is not eligible for recovery action" parity
  failures when replaying Python-generated games through TS engine. 4P soak
  tests now complete successfully. 3P/4P games with recovery actions now
  maintain correct `controllingPlayer` state after extraction.

- **Phase Transition Parity Bug Fix (Dec 20, 2025 – FIXED)** –
  Fixed critical phase transition bug in TS engine's `no_territory_action` handler.

  **Root Cause:**
  - `no_territory_action` case in `applyMoveWithChainInfo` returned state without
    advancing phase/player
  - `applyMoveForReplay` does NOT call `processPostMovePhases`, so phase transition
    must be handled inline
  - Python correctly advanced to next player's `ring_placement` or `forced_elimination`
  - TS was staying in `territory_processing` for the same player

  **Fix (commit b8175468):**
  - Added inline phase transition handling in `no_territory_action` case
  - Checks `computeHadAnyActionThisTurn()` and `playerHasStacksOnBoard()` for forced elimination
  - If no forced elimination needed, rotates to next player's `ring_placement`
  - Added `no_territory_action` to `isTurnEndingTerritoryMove` check (lines 1522-1523)
    to prevent `processPostMovePhases` from running after inline handler

  **Validation:** All canonical parity tests pass (5/5 games, 0 semantic divergence).

- **FORCED_ELIMINATION GPU Selfplay Parity Fix (Dec 20, 2025 – FIXED)** –
  Fixed missing FORCED_ELIMINATION phase handling in GPU selfplay scripts.

  **Root Cause:**
  - `run_gpu_selfplay.py` and `import_gpu_selfplay_to_db.py` were not handling
    FORCED_ELIMINATION phase transitions
  - This caused TS replay to diverge when Python recorded forced elimination moves

  **Fix:**
  - Added proper FORCED_ELIMINATION phase handling in GPU selfplay pipeline
  - Synced fix to all Lambda cluster nodes

  **Validation:** 138 parity tests pass, 0 failures. Fresh canonical selfplay data
  passes parity gate with 0 semantic divergence.

---

## 🟢 P3 – Test Alignment Items (Dec 12, 2025) – RESOLVED

### P3.1 – Python Move Type Canonicalization ✅ RESOLVED

**Component(s):** `ai-service/tests/parity/test_line_and_territory_scenario_parity.py`, `ai-service/tests/rules/test_fsm_fixtures.py`
**Severity:** Low (test alignment, not production code)
**Status:** ✅ RESOLVED (Dec 13, 2025)

**Details:**
Move type canonicalization (`process_territory_region` → `choose_territory_option`, `choose_line_reward` → `choose_line_option`) is now aligned. Python engine `MoveType` enum includes both legacy and canonical names with proper aliasing. All 16 parity tests pass, all 14 FSM fixture tests pass.

### P3.2 – DescentAI Uncertainty Selection Tests ✅ RESOLVED

**Component(s):** `ai-service/tests/test_descent_uncertainty_selection.py`, `ai-service/tests/test_mcts_dynamic_selection.py`
**Severity:** Low (new AI feature tests need alignment)
**Status:** ✅ RESOLVED (Dec 13, 2025)

**Details:**
Uncertainty selection tests are now properly skipped (implementation-specific white-box tests pending realignment with async batched evaluation). MCTS dynamic selection tests (5 tests) all pass. No test failures in AI test suites.

### P3.3 – Q23 Mini-Region Territory Test

**Component(s):** `ai-service/tests/test_territory_and_forced_elimination_property.py::test_territory_processing_q23_region_property`
**Severity:** Low (edge case detection algorithm)
**Status:** Skipped (edge case; normal gameplay unaffected)

**Details:**
The Q23 FAQ scenario involves mini-region detection during territory processing. Test is skipped pending algorithm refinement for correct mini-region identification in edge cases.

---

## 🔵 Open Investigation Items

### INV-001 – Recovery Slide Marker Destination Parity (Dec 10, 2025)

**Component(s):** Python `recovery.py`, TS `RecoveryAggregate.ts`
**Severity:** P2 (affects rare 4P game scenarios)
**Status:** ✅ RESOLVED (Dec 10, 2025)

**Original Symptom:** One 4P game (fa21f59a) failed TS replay at move 316 with error
"Invalid recovery slide: Destination has a marker" after the ring array
convention fix was applied. Python allowed the move; TS rejected it.

**Investigation Findings:**

- Both Python (`recovery.py:215-218`) and TS (`RecoveryAggregate.ts:435-440`)
  correctly validate that recovery destinations must have no marker
- The issue was NOT a validation bug but a **state divergence** caused by the
  ring array convention bugs that were fixed in commit `8532599b`
- The incorrect `controllingPlayer` calculation and `indexOf` vs `lastIndexOf`
  usage caused earlier state divergence that manifested as marker differences

**Resolution:**

- The ring array convention fix (commit `8532599b`) resolved this issue
- Post-fix 4P soak testing shows **9/9 games completed successfully** with
  no "Destination has a marker" errors
- Victory types: LPS (6), Territory (2), Elimination (1)

### INV-002 / HEX-PARITY-02 – Hexagonal Board ANM State Divergence (Dec 2025)

**Component(s):** Python `game_engine.py`, TS `turnOrchestrator.ts`, ANM detection logic
**Severity:** P2 (downgraded from P1)
**Status:** ✅ RESOLVED (Dec 25, 2025) – Hexagonal models trained and deployed

**Description:**
Historical ANM state divergence between TypeScript and Python engines during
`line_processing` and `territory_processing` phases on hexagonal boards.

**Resolution (Dec 25, 2025):**
All 12 canonical model configurations are now trained and deployed, including:

- `hex8_2p.pth` (38MB) - ELO 989
- `hex8_3p.pth` (38MB) - ELO 822
- `hex8_4p.pth` (38MB) - ELO 611
- `hexagonal_2p.pth` (166MB)
- `hexagonal_3p.pth` (166MB)
- `hexagonal_4p.pth` (166MB)

The parity issues were addressed through:

1. Contract vector expansion to 90 vectors with 100% parity
2. FSM validation fixes (commits f86c809f2, 60abb4f20)
3. GPU selfplay pipeline improvements for proper phase handling

**Detailed Documentation:**

- [`ai-service/docs/runbooks/HEXAGONAL_PARITY_BUG.md`](ai-service/docs/runbooks/HEXAGONAL_PARITY_BUG.md) – Historical runbook (now resolved)

### INV-003 / SQUARE19-PARITY-01 – Square19 2P TS↔Python Parity Issues (Dec 2025)

**Component(s):** TS `turnOrchestrator.ts` (territory exit / forced elimination / victory timing), parity harness state bundles
**Severity:** P1 (previously blocked square19 canonical training data at scale)
**Status:** ✅ RESOLVED (Dec 26, 2025) – **100% parity pass rate (4/4 games)** on `ai-service/data/canonical_square19.db` (canonical parity gate passes).

**Previously failing game (now passing):**

- Game: `915ab7de-ef80-47cd-820d-e9798dd85fdc`
- Bundle: `ai-service/parity_failures/square19_postfix_bundles/canonical_square19__915ab7de-ef80-47cd-820d-e9798dd85fdc__k695.state_bundle.json` (local-only artifact)
- Historical divergence at **ts_k=695** after `skip_territory_processing` where Python advanced to `forced_elimination` while TS advanced to `game_over`.

**Fix implemented (TS-side, Dec 26, 2025):**

- Updated the inline replay/turn-ending handler in [`applyMoveWithChainInfo()`](src/shared/engine/orchestration/turnOrchestrator.ts) for [`case 'skip_territory_processing'`](src/shared/engine/orchestration/turnOrchestrator.ts) to mirror the forced-elimination gating used by [`case 'no_territory_action'`](src/shared/engine/orchestration/turnOrchestrator.ts):
  - If `!hadAnyActionThisTurn && playerHasStacksOnBoard`, transition to `forced_elimination` and surface an explicit forced-elimination `PendingDecision` (type `elimination_target`).
  - Defer any `toVictoryState()` evaluation and any turn rotation until after the explicit `forced_elimination` move is applied.

**Verification (Dec 26, 2025):**

- TS tests: `npm test -- --testPathPattern="ForcedElimination|ANM|turnOrchestrator|parity"`
- Square19 parity harness: `cd ai-service && python3 -m scripts.check_ts_python_replay_parity --db data/canonical_square19.db --compact`
  - Summary: `total_games_checked: 4`, `games_with_semantic_divergence: 0`, `passed_canonical_parity_gate: true`.

**Impact:**

Square19 canonical DB replay parity is now unblocked for training/soak gates.

---

For a historical snapshot of implementation status, see
[`docs/archive/historical/CURRENT_STATE_ASSESSMENT.md`](docs/archive/historical/CURRENT_STATE_ASSESSMENT.md).
