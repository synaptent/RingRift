# RingRift Strategic Roadmap

**Version:** 3.0
**Last Updated:** November 20, 2025
**Status:** Engine/Rules Beta (Playable, not yet fully stable)
**Philosophy:** Robustness, Parity, and Scale

---

## 🎯 Executive Summary

**Current State:** Engine/Rules Beta (Core loop implemented; unified `chain_capture`/`continue_capture_segment` model live on the backend; sandbox + AI parity and end-to-end termination still under active work—see [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md:1) and [`TODO.md`](TODO.md:73) for current P0 gaps).
**Goal:** Production-Ready Multiplayer Game
**Timeline:** 4-8 weeks to v1.0 (assuming P0 rules/parity/termination items are resolved first)
**Strategy:** Harden testing & parity → Polish UX → Expand Multiplayer Features

---

## 🚀 Strategic Phases

### **PHASE 1: Core Playability (COMPLETED)** ✅

- [x] Game Engine & Rules (Movement, Capture, Lines, Territory)
- [x] Board Management (8x8, 19x19, Hex)
- [x] Basic AI Integration (Service + Fallback)
- [x] Frontend Basics (Board, Lobby, HUD, Victory)
- [x] Infrastructure (Docker, DB, WebSocket)

### **PHASE 2: Robustness & Testing (IN PROGRESS)**

**Priority:** P0 - CRITICAL
**Goal:** Ensure 100% rule compliance and stability

#### 2.1 Comprehensive Scenario Testing

- [ ] Build test matrix for all FAQ edge cases
- [ ] Implement scenario tests for complex chain captures
- [ ] Verify all board types (especially Hexagonal edge cases)

#### 2.2 Sandbox Stage 2

- [x] Stabilize client-local sandbox with unified “place then move” turn semantics
      for both human and AI seats (including mixed games), and automatic local AI
      turns when it is an AI player’s move. Implemented in the browser-only
      sandbox via `ClientSandboxEngine` and the `/sandbox` path of `GamePage`,
      with coverage from `ClientSandboxEngine.mixedPlayers` tests.
- [ ] Ensure parity between backend and sandbox engines and improve AI-vs-AI
      termination behaviour using the sandbox AI simulation diagnostics
      (`ClientSandboxEngine.aiSimulation` with `RINGRIFT_ENABLE_SANDBOX_AI_SIM=1`),
      as tracked in P0.2 / P1.4 of `KNOWN_ISSUES.md`.

### **PHASE 3: Multiplayer Polish**

**Priority:** P1 - HIGH
**Goal:** Seamless online experience

#### 3.1 Spectator Mode

- [ ] UI for watching active games
- [ ] Real-time updates for spectators

#### 3.2 Social Features

- [ ] In-game chat
- [ ] User profiles and stats
- [ ] Leaderboards

#### 3.3 Matchmaking

- [ ] Automated matchmaking queue
- [ ] ELO-based matching

### **PHASE 4: Advanced AI**

**Priority:** P2 - MEDIUM
**Goal:** Challenging opponents for all skill levels

#### 4.1 Machine Learning

- [ ] Train neural network models
- [ ] Deploy advanced models to Python service

#### 4.2 Advanced Heuristics

- [ ] Implement MCTS/Minimax for intermediate levels

---

## 🔗 Alignment with TODO Tracks

The high-level phases above correspond to the more detailed execution tracks
and checklists in `TODO.md`:

- **Phase 2: Robustness & Testing** ↔ **Track 1** (Rules/FAQ Scenario Matrix &
  Parity Hardening) and parts of **Track 3** (Sandbox as a Rules Lab).
- **Phase 3: Multiplayer Polish** ↔ **Track 2** (Multiplayer Lifecycle &
  HUD/UX) and parts of **Track 5** (Persistence, Replays, and Stats).
- **Phase 4: Advanced AI** ↔ **Track 4** (Incremental AI Improvements &
  Observability) and **P2** items in `AI_IMPROVEMENT_PLAN.md`.

For day-to-day planning, treat `TODO.md` (including the
"Consolidated Execution Tracks & Plan" section) as the canonical, granular
list of tasks that roll up into these phases.

---

## 📊 Success Metrics for v1.0

1.  **Reliability:** >99.9% uptime, zero critical bugs.
2.  **Performance:** AI moves <1s, UI updates <16ms.
3.  **Engagement:** Users completing full games without errors.
4.  **Compliance:** 100% pass rate on rule scenario matrix.

---

## Recommended Next Steps (Prioritized)

These are distilled from the code, the failing tests you just ran, and the existing roadmap, but phrased as concrete next actions.

### Tier 1 – Rules Confidence & Engine Parity (P0)

1. __Finish P0.4 unified Move model rollout on the backend__

   - Update or replace the remaining __CaptureDirection__ tests to target the new `chain_capture` + `continue_capture_segment` API instead of `chooseCaptureDirectionFromState`:

     - `GameEngine.captureDirectionChoice.test.ts`
     - `GameEngine.captureDirectionChoiceWebSocketIntegration.test.ts`
     - `GameEngine.chainCaptureChoiceIntegration.test.ts`

   - Once tests express expectations in terms of `getValidMoves` and `Move` types only, you’ll have a single canonical chain-capture model.

2. __Close sandbox vs backend semantic gaps for AI moves__

   - Focus on the concrete mismatches surfaced in:

     - `Sandbox_vs_Backend.aiHeuristicCoverage.test.ts`
     - `Backend_vs_Sandbox.aiParallelDebug.test.ts`
     - `Python_vs_TS.traceParity.test.ts`

   - For each mismatch:

     - Use the logged `scenarioLabel`, `seed`, `step`, and `sandboxMove` to reduce to a small deterministic unit test (movement/capture/placement parity) under `tests/unit/`.
     - Fix whichever engine (backend or sandbox) is incorrect, with a strong bias toward aligning `ClientSandboxEngine` to backend `GameEngine`/`RuleEngine` semantics.

   - This work moves you toward the “one canonical Move space everywhere” goal and will make trace-based debugging trustworthy.

3. __Get `FullGameFlow.test.ts` green__

   - Investigate why `finalState.gameStatus` remains `'active'` under local AI fallback.

   - Ensure that:

     - AI fallback never gets stuck in a loop of legal-but-non-progressing moves.
     - Termination conditions (ring-elimination and territory-control) are reached under typical AI-vs-AI games.

   - Once this passes, you can credibly say “a full backend game with AI fallback completes without crashing or stalling.”

### Tier 2 – Scenario Matrix & Explicit Rules Coverage (P0/P1)

4. __Expand `RULES_SCENARIO_MATRIX.md` and scenario suites__

   - Systematically cover FAQ and rules examples in Jest:

     - Complex chain captures (180° reversals, cycles, zig‑zag) for `square8`, `square19`, and hex.
     - Mixed line + territory + forced-elimination turns.
     - Edge/corner territory scenarios on hex.

   - Ensure each scenario is:

     - Represented once in `RULES_SCENARIO_MATRIX.md` with an ID.
     - Covered by at least one backend `GameEngine` test and, ideally, a sandbox test.

5. __Align sandbox AI and backend Move semantics__

   - Refactor `ClientSandboxEngine` and `sandboxAI` so that *all* sandbox actions are:

     - Expressed as canonical `Move` objects (`place_ring`, `move_stack`, `overtaking_capture`, `continue_capture_segment`, etc.).
     - Applied through a single canonical `applyCanonicalMoveInternal` path.

   - This makes the sandbox a faithful “frontend shell” over the same decision space as the backend and makes both your parity and heuristic-coverage suites much easier to reason about.

### Tier 3 – Multiplayer Lifecycle & UX (P1)

6. __WebSocket lifecycle & reconnection polish__

   - Tighten and test:

     - Rejoin flows (client disconnects and reconnects mid-game).
     - Spectator joins/leaves and enforcing read-only semantics.
     - `game_over` handling (no lingering pending choices, sockets leave rooms cleanly).

   - Add or extend integration tests under `tests/unit/WebSocketServer.aiTurn.integration.test.ts` and related suites.

7. __HUD and GamePage UX__

   - Enhance `GameHUD` and `GamePage` to display, for both backend and sandbox:

     - Current player + phase.
     - Rings in hand/on board/eliminated per player.
     - Territory counts.
     - Simple timers (`timeControl` and `timeRemaining`) even if initially cosmetic.

   - Add a compact event log for moves and PlayerChoices to aid debugging and teaching.

### Tier 4 – AI Observability & Gradual Strengthening (P1–P2)

8. __Add AI telemetry__

   - In `AIServiceClient` and `AIInteractionHandler`, record and/or log:

     - Call type (move vs choice type).
     - Latency.
     - Success vs timeout vs error + fallback.

   - Optionally expose these via Prometheus/Grafana later; for now, even structured logs are a big win for debugging.

9. __Targeted heuristic AI improvements (no big bang)__

   - Once parity is solid, apply small, measurable adjustments to `heuristic_ai.py` based on observed weaknesses (e.g., tunnel‑vision on lines, poor territory awareness) using your AI simulation and trace harnesses.

### Tier 5 – Persistence, Replays, and Stats (P2)

10. __Lifecycle persistence and basic replays__

    - Ensure DB game lifecycle (`WAITING` → `ACTIVE` → `COMPLETED`), `startedAt`/`endedAt`, and `winnerId` are always correct.
    - Add a simple move-history panel in the client (based on `GameHistoryEntry`) and a server-side “replay on GameEngine” helper for offline analysis.

11. __Stats & leaderboards__

    - Once game results are reliable, wire up ELO/ratings and simple leaderboards using existing Prisma models.
