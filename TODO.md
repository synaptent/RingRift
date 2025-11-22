# RingRift TODO / Task Tracker

**Last Updated:** November 21, 2025

This file is the canonical high-level task tracker for the project.
When it disagrees with older planning docs (for example files under
`deprecated/`), this file and the status docs
[`CURRENT_STATE_ASSESSMENT.md`](CURRENT_STATE_ASSESSMENT.md)
win.

For **rules semantics themselves** (what the correct behaviour _should_ be),
the ultimate source of canonical truth is the rules documentation:
[`ringrift_complete_rules.md`](ringrift_complete_rules.md) (and, where
applicable, [`ringrift_compact_rules.md`](ringrift_compact_rules.md)).
When there is any ambiguity, parity mismatch, or question about an engine or
sandbox implementation, tests and code should be treated as converging toward
those documents, not the other way around.

Priorities:

- **P0** – Critical for rules correctness / engine parity.
- **P1** – High-value for playable, stable online games.
- **P2** – Important but can follow P0/P1.

## Phase 2 – Robustness & Testing (IN PROGRESS, P0)

### P0.1 – Rules/FAQ Scenario Matrix

- [x] Build and maintain a scenario matrix in
      [`RULES_SCENARIO_MATRIX.md`](RULES_SCENARIO_MATRIX.md) that maps
      examples from [`ringrift_complete_rules.md`](ringrift_complete_rules.md)
      and the FAQ to concrete Jest suites.
- [x] For each major rule cluster (movement, chain captures, lines,
      territory, victory), ensure there is at least one emblematic scenario
      that is tested in both:
  - Backend engine (`GameEngine` / `RuleEngine` in
    [`src/server/game`](src/server/game/))
  - Client sandbox engine (`ClientSandboxEngine` in
    [`src/client/sandbox/ClientSandboxEngine.ts`](src/client/sandbox/ClientSandboxEngine.ts))
- [x] Keep scenario IDs consistent across:
  - [`tests/scenarios/rulesMatrix.ts`](tests/scenarios/rulesMatrix.ts)
  - Scenario suites under `tests/scenarios/`
  - Any parity tests that reference specific rules/FAQ cases.

### P0.2 – Backend ↔ Sandbox Parity

- [ ] Regularly run and keep green the trace-parity and heuristic coverage
      suites:
  - [`Backend_vs_Sandbox.traceParity.test.ts`](tests/unit/Backend_vs_Sandbox.traceParity.test.ts)
  - [`Sandbox_vs_Backend.seed5.traceDebug.test.ts`](tests/unit/Sandbox_vs_Backend.seed5.traceDebug.test.ts)
  - [`Sandbox_vs_Backend.aiHeuristicCoverage.test.ts`](tests/unit/Sandbox_vs_Backend.aiHeuristicCoverage.test.ts)
  - [`RefactoredEngineParity.test.ts`](tests/unit/RefactoredEngineParity.test.ts) (minimal legacy-backend vs shared-engine parity harness; currently covers a basic placement + movement sequence and should be extended over time)
- [ ] When a parity failure appears:
  - [ ] Extract the first divergence index using the
        `TraceParity.*.firstDivergence` helpers.
  - [ ] Turn the failing position into a small, focused unit test under
        `tests/unit/` (e.g., movement, capture, or placement parity).
  - [ ] Fix the underlying engine/sandbox discrepancy.
  - [ ] Mark the corresponding row in
        [`RULES_SCENARIO_MATRIX.md`](RULES_SCENARIO_MATRIX.md) as covered.
- [ ] Treat any genuine rules mismatch between backend
      [`GameEngine`](src/server/game/GameEngine.ts) /
      [`RuleEngine`](src/server/game/RuleEngine.ts) and
      [`ClientSandboxEngine`](src/client/sandbox/ClientSandboxEngine.ts)
      as a **P0 bug**.

### P0.3 – S-Invariant & Termination

- [ ] Keep the S-invariant tests passing and expand them as new rules
      interactions are implemented:
  - [`ProgressSnapshot.core.test.ts`](tests/unit/ProgressSnapshot.core.test.ts)
  - Sandbox AI simulation diagnostics in
    [`ClientSandboxEngine.aiSimulation.test.ts`](tests/unit/ClientSandboxEngine.aiSimulation.test.ts)
  - Backend AI-style simulations in
    [`GameEngine.aiSimulation.test.ts`](tests/unit/GameEngine.aiSimulation.test.ts)
- [ ] For any new rule that can change markers, collapsed spaces, or
      eliminated rings, ensure it is reflected in
      [`computeProgressSnapshot`](src/shared/engine/core.ts) and covered by
      tests.

### P0.4 – Unified Move model for all player-chosen decisions (backend + sandbox)

Goal: every player-chosen decision permitted by the rules – including chain
capture direction, line order & rewards, territory region order, and ring
elimination targets – is represented as a `Move` and enumerated via
`RuleEngine.getValidMoves` / `GameEngine.getValidMoves`, not only via
`PlayerChoice` flows. This enables:

- A single canonical move space for backend AI, sandbox AI, and clients.
- Stronger parity guarantees between backend and sandbox behaviour.
- Easier replay, trace analysis, and tooling around complete turn histories.

Planned work:

- [x] Extend the shared `Move` model in
      [`src/shared/types/game.ts`](src/shared/types/game.ts) with additional
      decision types, for example:
  - [x] `continue_capture_segment` for capture chain continuation segments
        (`from`, `captureTarget`, `to`) in backend `GameEngine` / `RuleEngine`.
  - [x] `process_line` / `choose_line_reward` for line order & reward
        decisions (line id, marker positions, reward option).
  - [x] `process_territory_region` for disconnected region order
        (region id, representative position, size).
  - [x] `eliminate_rings_from_stack` for explicit elimination targets
        (stack position, elimination count / cap height).
- [x] Introduce or clarify interactive phases in `GameState.currentPhase`
      where these decisions occur, e.g.:
  - [x] `chain_capture` phase for mandatory capture continuations (backend `GameEngine` / `TurnEngine` wired).
  - [x] Explicit `line_processing` interactive steps where `getValidMoves`
        returns `process_line` / `choose_line_reward` moves (backend complete; sandbox parity pending).
  - [x] Explicit `territory_processing` interactive steps where
        `getValidMoves` returns `process_territory_region` moves (backend region enumeration complete; sandbox Move/phase parity and adoption of explicit elimination Moves in WebSocket/AI and sandbox flows remain future work).
- [x] Refactor `RuleEngine.getValidMoves` to:
  - [x] Preserve existing behaviour for `ring_placement`, `movement`, and
        `capture` (`place_ring`, `skip_placement`, `move_stack`,
        `overtaking_capture`).
  - Note: In `chain_capture` phase, enumeration of follow-up capture
    segments is handled by `GameEngine.getValidMoves`, which has access to
    internal `chainCaptureState`; `RuleEngine` remains responsible for
    segment-level validation of `overtaking_capture` /
    `continue_capture_segment`.
  - [x] In `line_processing` phase, enumerate all eligible lines and
        available reward options as `process_line` / `choose_line_reward`
        moves instead of (or in addition to) `LineOrderChoice` and
        `LineRewardChoice`.
  - [x] In `territory_processing` phase, enumerate eligible disconnected
        regions as `process_territory_region` moves instead of (or in addition
        to) `RegionOrderChoice` for ordering (explicit elimination moves remain
        future work).
- [ ] Update `GameEngine` to drive these phases via `makeMove` rather than
      direct `PlayerInteractionManager` calls, so that:
  - [ ] Human clients and AI both select from the same `getValidMoves`
        result set for all interactive decisions.
  - [ ] Internal post-move processors (`lineProcessing`,
        `territoryProcessing`, `captureChainEngine`) are refactored to apply
        selected decision-moves instead of bespoke `PlayerChoice` branches.
- [ ] Adjust `PlayerInteractionManager` and WebSocket handlers so that
      frontends treat these new decision-move types as first-class actions
      (selecting among moves) rather than distinct `PlayerChoice` payloads.
- [ ] Keep `ClientSandboxEngine` in sync by:
  - [ ] Importing the same extended `Move` types.
  - [ ] Mirroring the new interactive phases and decision-move handling in
        the sandbox (e.g., chain capture, line processing, territory
        processing, elimination) using the existing sandbox engines
        (`sandboxMovementEngine`, `sandboxLinesEngine`, `sandboxTerritoryEngine`,
        `sandboxElimination`).
  - [ ] Updating `sandboxAI` to select among these richer `Move` sets,
        staying in lockstep with backend `getValidMoves`.
- [ ] Align sandbox Move/phase handling with backend:
  - [ ] Route all sandbox actions (human and AI) through the canonical
        `applyCanonicalMoveInternal` path using the same `Move` types and
        `GamePhase` transitions as the backend, including `chain_capture`,
        `line_processing`, and `territory_processing`.
  - [ ] Ensure `ClientSandboxEngine` respects the same decision semantics
        for `continue_capture_segment`, `process_line`, and
        `process_territory_region`.
  - [ ] Update sandbox parity and RulesMatrix-backed tests to assert both
        the legal-move sets and resulting phases for these advanced phases.
- [ ] Extend and/or add parity tests to cover the new decision-move
      surface:
  - [ ] Backend vs sandbox parity for chain capture continuation decisions.
  - [ ] Backend vs sandbox parity for line order/reward and region
        order/elimination decisions.
  - [ ] Trace parity tests that confirm all such decisions are now
        represented as `Move`s and faithfully replayable.

**Current P0.4 status (as of 2025‑11‑19):**

- [x] Backend `GameEngine` / `RuleEngine` now model capture-chain continuation via a distinct `chain_capture` phase and `continue_capture_segment` moves.
- [x] Backend capture-sequence enumeration now uses `captureChainEngine.getCaptureOptionsFromPosition` plus shared `validateCaptureSegmentOnBoard`, keeping [`captureSequenceEnumeration.test.ts`](tests/unit/captureSequenceEnumeration.test.ts) green across square and hex boards.
- [x] Backend territory-processing now enumerates explicit `eliminate_rings_from_stack` Moves via `RuleEngine.getValidEliminationDecisionMoves`, and `RingEliminationChoice.options[].moveId` / `RegionOrderChoice.options[].moveId` are wired to canonical `Move.id` values for elimination and disconnected-region decisions (sandbox Move/phase parity and WebSocket/AI adoption of these Move ids remain future work).
- [ ] Several scenario/parity suites are temporarily red while the new chain-capture model is wired through all surfaces:
  - [ ] [`ComplexChainCaptures.test.ts`](tests/scenarios/ComplexChainCaptures.test.ts)
  - [ ] [`RulesMatrix.ChainCapture.GameEngine.test.ts`](tests/scenarios/RulesMatrix.ChainCapture.GameEngine.test.ts)
  - [ ] [`GameEngine.cyclicCapture.*.test.ts`](tests/unit/GameEngine.cyclicCapture.scenarios.test.ts)
  - [ ] [`Backend_vs_Sandbox.aiParallelDebug.test.ts`](tests/unit/Backend_vs_Sandbox.aiParallelDebug.test.ts)
  - [ ] [`Sandbox_vs_Backend.aiHeuristicCoverage.test.ts`](tests/unit/Sandbox_vs_Backend.aiHeuristicCoverage.test.ts)
  - [ ] [`TraceParity.seed*.firstDivergence.test.ts`](tests/unit/TraceParity.seed5.firstDivergence.test.ts)
- [x] Sandbox engine and `sandboxAI` now participate in the new `chain_capture` / `continue_capture_segment` Move model for AI turns and canonical traces; remaining divergences (e.g. seed 14 trace parity) are localized and tracked via `TraceParity.seed14.*` / `ParityDebug.seed14.*` helpers.

**Near-term P0.4 tasks inferred from current test failures:**

- [ ] Finalize backend chain-capture semantics for cyclic/triangle patterns so that FAQ scenarios in [`rulesMatrix.ts`](tests/scenarios/rulesMatrix.ts) and [`GameEngine.cyclicCapture.*.test.ts`](tests/unit/GameEngine.cyclicCapture.scenarios.test.ts) pass under the new model.
- [ ] Migrate `ComplexChainCaptures` and `RulesMatrix.ChainCapture` suites to drive chains via `chain_capture` + `continue_capture_segment` rather than internal while-loops.
- [ ] Mirror the chain-capture phase and `continue_capture_segment` moves into [`ClientSandboxEngine`](src/client/sandbox/ClientSandboxEngine.ts) and [`sandboxAI`](src/client/sandbox/sandboxAI.ts), then restore:
  - [ ] [`Sandbox_vs_Backend.aiHeuristicCoverage.test.ts`](tests/unit/Sandbox_vs_Backend.aiHeuristicCoverage.test.ts)
  - [ ] [`Backend_vs_Sandbox.*trace*.test.ts`](tests/unit/Backend_vs_Sandbox.aiParallelDebug.test.ts)
- [ ] Once chain capture is stable across backend and sandbox, extend the same Move/phase unification to line-processing and territory-processing (as described above).
- [ ] Keep `tests/integration/FullGameFlow.test.ts` green by ensuring AI-vs-AI backend games using local AI fallback always reach a terminal `gameStatus` (e.g. `completed`/`finished`) within the configured move budget. Treat regressions here as part of P0.4 since they exercise the unified Move model end-to-end.

### P0.5 – Python AI-service rules engine parity (P0)

- [x] Audit mismatches between Python [`GameEngine`](ai-service/app/game_engine.py) /
      [`BoardManager`](ai-service/app/board_manager.py) /
      [`models`](ai-service/app/models.py) and the canonical TS shared rules
      engine under `src/shared/engine/` (types, validators, mutators,
      `GameEngine`), plus backend [`GameEngine`](src/server/game/GameEngine.ts) /
      [`RuleEngine`](src/server/game/RuleEngine.ts) adapters, using the rules
      docs as the primary spec.
- [x] Align Python movement and overtaking captures with TS:
  - [x] Ray-based movement using board-type directions (Moore for square,
        6-dir for hex), enforcing minimum distance ≥ stack height and blocking on
        stacks/collapsed spaces while processing markers on the path.
  - [x] Cap-height-based overtaking captures that allow both own-stack and
        opponent-stack targets, with from→target→landing geometry validated
        analogously to
        [`validateCaptureSegmentOnBoard`](src/shared/engine/core.ts).
  - [x] Chain-capture application that leaves markers on departure, processes
        markers along both legs, transfers exactly one captured ring per segment
        to the bottom of the attacker, supports merging at the landing stack, and
        updates a Python [`ChainCaptureState`](ai-service/app/models.py)
        mirroring TS [`ChainCaptureState`](src/server/game/rules/captureChainEngine.ts).
- [x] Placement parity for AI-service engine:
  - [x] Support multi-ring placement on empty spaces (1–3 rings) and exactly
        1 ring when placing on existing stacks, mirroring
        [`RuleEngine.validateRingPlacement`](src/server/game/RuleEngine.ts)
        semantics.
  - [x] Enforce per-player ring caps derived from total rings in play /
        max players (Python analogue of `BOARD_CONFIGS[boardType].ringsPerPlayer`).
  - [x] Implement TS-style no-dead-placement via a hypothetical board helper
        (Python analogue of
        [`createHypotheticalBoardWithPlacement`](src/server/game/rules/placementHelpers.ts))
        plus a reachability helper equivalent to
        [`hasAnyLegalMoveOrCaptureFromOnBoard`](src/shared/engine/core.ts).
- [x] Lines parity for AI-service engine:
  - [x] Ensure Python line detection in
        [`BoardManager.find_all_lines`](ai-service/app/board_manager.py)
        matches TS [`BoardManager.findAllLines`](src/server/game/BoardManager.ts)
        in geometry and minimum length.
  - [x] Refactor Python line-processing move generation and
        [`_apply_line_formation`](ai-service/app/game_engine.py) to mirror
        [`lineProcessing`](src/server/game/rules/lineProcessing.ts) semantics
        (exact-length vs overlength options, collapsed markers, reward
        eliminations) while preserving the unified `Move` model.
- [x] Territory disconnection parity for AI-service engine:
  - [x] Extend Python
        [`BoardManager.find_disconnected_regions`](ai-service/app/board_manager.py)
        and territory-processing moves to respect the single-border-color marker
        logic, region representation rules, and cascades implemented in
        [`territoryProcessing`](src/server/game/rules/territoryProcessing.ts).
  - [x] Ensure mandatory self-elimination prerequisites and elimination
        accounting match TS behaviour.
- [x] Turn engine / forced elimination / victory semantics (AI-service):
  - [x] Align Python phase transitions and forced-elimination moves in
        [`GameEngine._update_phase`](ai-service/app/game_engine.py) and
        [`_end_turn`](ai-service/app/game_engine.py) with the TS
        [`TurnEngine`](src/server/game/turn/TurnEngine.ts) semantics
        (`skip_placement`, forced elimination only when no other actions are
        available, re-checking actions after forced elimination in the same turn).
  - [x] Bring [`GameEngine._check_victory`](ai-service/app/game_engine.py)
        into parity with
        [`RuleEngine.checkGameEnd`](src/server/game/RuleEngine.ts),
        including last-player-standing and stalemate (rings-in-hand conversion)
        tie-break ladders.
- [ ] Lint/typing hygiene for `ai-service` rules engine:
  - [ ] Gradually reduce flake8 long-line violations in
        [`game_engine.py`](ai-service/app/game_engine.py) and related files
        without sacrificing the diagnostic value of comments.
  - [ ] Optionally add targeted `# type: ignore[...]` annotations for known
        Pydantic alias patterns (e.g. `capture_target`, `collapsed_markers`) where
        Pylance cannot infer the dynamic `__init__` parameters, to keep the
        signal-to-noise ratio of editor diagnostics high.

## Phase 3 – Multiplayer Polish (P1)

### P1.1 – WebSocket Lifecycle & Reconnection

- [ ] Tighten and test WebSocket lifecycle around:
  - Reconnects and late joins in
    [`WebSocketServer`](src/server/websocket/server.ts) and
    [`WebSocketInteractionHandler`](src/server/game/WebSocketInteractionHandler.ts).
  - Consistent `game_over` handling and clearing of any pending choices.
  - Spectator join/leave flows, ensuring spectators are always read-only.
- [ ] Add focused Jest integration tests for lifecycle aspects under
      `tests/unit/`, and cross-link them from
      [`tests/README.md`](tests/README.md).

### P1.2 – Game HUD & GamePage UX

- [ ] Enhance [`GameHUD`](src/client/components/GameHUD.tsx) to show:
  - Current player and phase.
  - Per-player ring counts (in hand / on board / eliminated).
  - Territory spaces per player.
  - Basic timer readout derived from `timeControl` and `timeRemaining`.
- [ ] Ensure the same HUD behaviour is shared between backend games
      (`/game/:gameId`) and sandbox games (`/sandbox`).
- [ ] Add a minimal per-game event log in
      [`GamePage`](src/client/pages/GamePage.tsx) for moves, PlayerChoices,
      and `game_over` events.

### P1.3 – Lobby, Spectators, and Chat

- [ ] Improve lobby UX in
      [`LobbyPage`](src/client/pages/LobbyPage.tsx) with clearer status,
      filters, and navigation.
- [x] Implement a basic spectator UI (read-only board + HUD) that uses
      the same `GameContext` as players but disables input.
- [ ] Wire a simple in-game chat panel to the existing `chat_message`
      events in the WebSocket server.

## Phase 4 – Advanced AI (P2)

### P2.1 – Stronger Opponents

- [ ] Promote at least one stronger AI implementation
      (Minimax/MCTS/NeuralNet) from
      [`ai-service/app/ai`](ai-service/app/ai/__init__.py) into the primary
      `/ai/move` path, behind a non-default `AIProfile`.
- [ ] Add tests around `AIServiceClient` to cover latency, timeouts, and
      fallback usage in more detail.

### P2.2 – AI Telemetry

- [ ] Add lightweight logging/metrics around AI calls in
      [`AIServiceClient`](src/server/services/AIServiceClient.ts) and
      [`AIInteractionHandler`](src/server/game/ai/AIInteractionHandler.ts),
      capturing:
  - Request type
  - Duration
  - Success vs failure vs fallback
- [ ] Surface basic metrics in the existing Prometheus/Grafana stack
      defined in [`docker-compose.yml`](docker-compose.yml).

## Cross-Cutting – Documentation & CI

- [ ] Keep the following docs synchronized whenever behaviour changes:
  - [`README.md`](README.md)
  - [`QUICKSTART.md`](QUICKSTART.md)
  - [`CURRENT_STATE_ASSESSMENT.md`](CURRENT_STATE_ASSESSMENT.md)
  - [`STRATEGIC_ROADMAP.md`](STRATEGIC_ROADMAP.md)
- [ ] When adding new scenario suites or parity harnesses, update
      [`tests/README.md`](tests/README.md) and
      [`RULES_SCENARIO_MATRIX.md`](RULES_SCENARIO_MATRIX.md) so newcomers
      know where to look.
- [ ] Gradually tighten Jest coverage thresholds in
      [`jest.config.js`](jest.config.js) once the scenario matrix and
      parity suites are stable.

This file is intentionally concise; for deeper narrative context and
ongoing issue lists, refer to
[`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) and
[`AI_ARCHITECTURE.md`](AI_ARCHITECTURE.md).

## Consolidated Execution Tracks & Plan

The following tracks and steps summarize the recommended direction from the
latest project assessment. They should be kept in sync with
`CURRENT_STATE_ASSESSMENT.md` and `STRATEGIC_ROADMAP.md` as work progresses.

### Track 1 – Rules/FAQ Scenario Matrix & Parity Hardening (P0)

- [x] Expand and maintain `RULES_SCENARIO_MATRIX.md` to map
      `ringrift_complete_rules.md` + FAQ examples to concrete Jest suites.
- [x] For each major rule cluster (movement, chain captures, lines,
      territory, victory), ensure at least one emblematic scenario is tested in
      both:
  - [x] Backend engine (`GameEngine` / `RuleEngine` under `src/server/game/`).
  - [x] Client sandbox engine (`ClientSandboxEngine` under
        `src/client/sandbox/ClientSandboxEngine.ts`).
- [x] Keep scenario IDs consistent across:
  - [x] `RULES_SCENARIO_MATRIX.md`.
  - [x] `tests/scenarios/rulesMatrix.ts`.
  - [x] Suites under `tests/scenarios/`.
- [ ] Use parity harnesses as scenario generators:
  - [ ] Run `Backend_vs_Sandbox.traceParity.test.ts`,
        `Sandbox_vs_Backend.seed5.traceDebug.test.ts`, and
        `Sandbox_vs_Backend.aiHeuristicCoverage.test.ts` regularly.
  - [ ] When a parity failure occurs, extract the first divergence via the
        `TraceParity.*.firstDivergence` helpers.
  - [ ] Promote the failing position to a focused unit test under
        `tests/unit/` (e.g. movement/capture/placement parity).
  - [ ] Fix the underlying engine/sandbox discrepancy and mark the
        corresponding row in `RULES_SCENARIO_MATRIX.md` as covered.
- [ ] Define and document "must-cover" scenario sets per rules axis
      (movement, capture/chains, lines, territory, victory) and tie Jest
      coverage thresholds to these modules once the baseline is in place.
- [ ] For each axis ID (M*/C*/L*/T*/V\*) in `RULES_SCENARIO_MATRIX.md`,
      ensure there is at least one `rulesMatrix.ts` scenario and matching
      backend + sandbox test suite, and cross-link those suites from the
      matrix.

### Track 2 – Multiplayer Lifecycle & HUD/UX (P1)

- [ ] Tighten WebSocket lifecycle around reconnection and late joins:
  - [ ] In `WebSocketServer` + `WebSocketInteractionHandler`, ensure that
        reconnecting clients re-emit `join_game`, rehydrate state from the
        DB + `GameEngine`, and clear any stale choices.
  - [ ] Add focused Jest integration tests for reconnect flows and
        `game_over` handling in `tests/unit/` and cross-link from
        `tests/README.md`.
- [x] Clarify and enforce spectator semantics:
  - [x] Ensure spectators are always read-only at the server level.
  - [x] Provide a dedicated spectator view in the client (using
        `GameContext` but with input disabled).
  - [x] Add tests for spectator join/leave flows.
- [ ] Enhance `GameHUD` and GamePage UX:
  - [ ] Ensure `GameHUD` shows:
    - [ ] Current player and phase.
    - [ ] Per-player rings in hand / on board / eliminated.
    - [ ] Territory spaces per player.
    - [ ] (Optional) Timer readouts based on `timeControl` and
          `timeRemaining`.
  - [ ] Add a minimal per-game event log in `GamePage` for moves,
        PlayerChoices, and `game_over` events.
  - [ ] Improve phase-specific prompts and invalid-move feedback (toasts,
        subtle animations) in the frontend.

### Track 3 – Sandbox as a First-Class Rules Lab (P0–P1)

- [ ] Finish unifying sandbox canonical mutations in
      `ClientSandboxEngine`:
  - [ ] Route all AI and human actions through the canonical
        `applyCanonicalMoveInternal` path.
  - [ ] Remove bespoke mutation logic in sandbox AI so that sandbox phases
        and move types stay in lockstep with backend `GameEngine` /
        `RuleEngine`.
- [ ] Expose rules/FAQ scenarios directly in the sandbox UI:
  - [ ] Add a simple scenario selector (e.g. dropdown) in `/sandbox` backed
        by `rulesMatrix.ts` / `RULES_SCENARIO_MATRIX.md`.
  - [ ] Allow loading a named scenario into the sandbox for visual
        inspection and step-through play.
- [ ] Add visual debugging aids to the sandbox view:
  - [ ] Overlays for detected lines and their rewards.
  - [ ] Territory region highlighting and disconnection visualization.
  - [ ] Chain capture path visualization (e.g. arrows or highlighted
        segments).

### Track 4 – Incremental AI Improvements & Observability (P1–P2)

- [ ] Add lightweight metrics/logging around AI calls:
  - [ ] In `AIServiceClient` and `AIInteractionHandler`, log request type,
        latency, success/failure, and fallback usage.
  - [ ] Optionally expose these metrics via existing or new monitoring
        tooling (e.g. Prometheus if added later).
- [ ] Reflect AI mode in the HUD:
  - [ ] Display whether each AI player is using service-backed or local
        heuristic decisions.
  - [ ] Add a simple AI service health indication (if health endpoint is
        defined).
- [ ] Make targeted Heuristic AI improvements (within tight bounds):
  - [ ] Refine evaluation weights for early/mid/late game phases using
        available data (line potential, basic mobility, territory).
  - [ ] Avoid large structural changes until rules/parity work is fully
        stabilized.
- [ ] Prepare Python AI foundations (background work):
  - [ ] Implement robust `get_valid_moves(game_state)` and
        `apply_move(game_state, move)` in the AI service, per
        `AI_IMPROVEMENT_PLAN.md`.
  - [ ] Keep this evolution behind existing endpoints so the TS boundary
        remains stable.

### Track 5 – Persistence, Replays, and Stats (P2)

- [ ] Clarify and enforce game lifecycle transitions in the database:
  - [ ] Ensure consistent `status` transitions (`WAITING` → `ACTIVE` →
        `COMPLETED`), along with `startedAt` / `endedAt` timestamps.
  - [ ] Add targeted tests around lifecycle persistence.
- [ ] Surface simple history/replay in the UI:
  - [ ] Add a move list panel (based on `GameHistoryEntry`) in the
        GamePage HUD.
  - [ ] Plan a dedicated replay view powered by stored moves and backend
        `GameEngine` replays once the basics are stable.
- [ ] Build initial stats/leaderboards once game results are reliable:
  - [ ] Aggregate per-user stats (wins/losses, rating if enabled).
  - [ ] Expose a minimal leaderboard view in the client.

### Suggested 2–4 Week Execution Plan (Guidance)

These are not strict milestones but a suggested ordering that can be
reflected by checking off the above track items.

- [ ] **Week 1 – Lock in Rules Confidence (Tracks 1 & 3)**
  - [ ] Expand `RULES_SCENARIO_MATRIX.md` and scenario suites under
        `tests/scenarios/` for key movement, capture, line, territory, and
        victory examples.
  - [ ] Run parity suites regularly; for each failure, promote a minimal
        unit test and fix the underlying discrepancy.
  - [ ] Begin unifying sandbox canonical move handling in
        `ClientSandboxEngine`.
- [ ] **Week 2 – Multiplayer Lifecycle & HUD (Track 2 + more Track 1)**
  - [ ] Implement reconnection + spectator semantics end-to-end and add
        tests.
  - [ ] Flesh out `GameHUD` and integrate a basic event log into
        `GamePage`.
  - [ ] Continue adding rules scenarios and parity fixes.
- [ ] **Week 3 – Sandbox UX & AI Observability (Tracks 3 & 4)**
  - [ ] Add scenario picker into the sandbox UI and basic visual helpers
        for lines/territory/chain captures.
  - [ ] Add AI telemetry and, optionally, surface simple metrics.
  - [ ] Implement one or two bounded heuristic AI improvements that
        clearly improve play.
- [ ] **Week 4 – Persistence/Polish & Buffer (Track 5 + cleanup)**
  - [ ] Introduce a minimal move history/replay panel in GamePage or a
        dedicated route.
  - [ ] Update docs (`CURRENT_STATE_ASSESSMENT`, `IMPLEMENTATION_STATUS`,
        `TODO`, `STRATEGIC_ROADMAP`) to reflect new coverage and UX.
  - [ ] Use remaining time to close lingering P0 items and fix UX
        papercuts discovered via playtesting.

This section is intentionally high-level and should be pruned or merged into
earlier phases as items are completed or re-scoped.
