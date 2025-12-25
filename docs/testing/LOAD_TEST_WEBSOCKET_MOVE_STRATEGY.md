# WebSocket Move Load Test Strategy

> Status: Draft (MVP WebSocket gameplay load tests)  
> Date: 2025-12-04  
> Owner Modes: Architect + Code + Debug
>
> Related docs:
>
> - [`STRATEGIC_ROADMAP.md`](../planning/STRATEGIC_ROADMAP.md:1)
> - [`PLAYER_MOVE_TRANSPORT_DECISION.md`](../architecture/PLAYER_MOVE_TRANSPORT_DECISION.md:1)
> - [`LOAD_TEST_BASELINE_REPORT.md`](LOAD_TEST_BASELINE_REPORT.md:1)
> - [`API_REFERENCE.md`](../architecture/API_REFERENCE.md:1)
> - [`websocket-stress.js`](../../tests/load/scenarios/websocket-stress.js:1)
> - [`player-moves.js`](../../tests/load/scenarios/player-moves.js:1)
> - [`game-creation.js`](../../tests/load/scenarios/game-creation.js:1)
> - [`concurrent-games.js`](../../tests/load/scenarios/concurrent-games.js:1)
> - [`playwright.config.ts`](../../playwright.config.ts:1)

---

## 1. Overview

This document defines an implementation-ready strategy for validating the **WebSocket gameplay SLOs** (move latency, decision latency, stalls, and reliability) on the **canonical WebSocket path**, as defined in the WebSocket gameplay SLOs in [`STRATEGIC_ROADMAP.md` §2.2](../planning/STRATEGIC_ROADMAP.md:324).

### 1.1 Goal

- Exercise the **real gameplay protocol over Socket.IO/WebSockets** (join, moves, state updates, game-over and, later, decision phases).
- Produce **load-test metrics directly mapped** to the WebSocket gameplay SLOs and stall definitions in [`STRATEGIC_ROADMAP.md` §2.2](../planning/STRATEGIC_ROADMAP.md:324).
- Provide scenarios suitable for:
  - Fast CI/dev smoke checks.
  - P-01-style concurrency validation in staging/perf.
  - Medium-term decision-phase stress tests.

### 1.2 Relationship to HTTP harness and existing k6 scenarios

The canonical move transport decision is documented in [`PLAYER_MOVE_TRANSPORT_DECISION.md`](../architecture/PLAYER_MOVE_TRANSPORT_DECISION.md:1):

- **WebSocket is the canonical move transport** for interactive clients.
- Any HTTP move endpoint (for example, `POST /api/games/:gameId/moves`) is a **thin internal/test harness** only, exposed under “Internal / Test harness APIs” in [`API_REFERENCE.md`](../architecture/API_REFERENCE.md:65).

Existing k6 scenarios under `tests/load/scenarios` focus primarily on HTTP flows and low-level WebSocket transport:

- [`game-creation.js`](../../tests/load/scenarios/game-creation.js:1) and [`concurrent-games.js`](../../tests/load/scenarios/concurrent-games.js:1) validate **HTTP game lifecycle and polling** (auth, game creation, state reads).
- [`player-moves.js`](../../tests/load/scenarios/player-moves.js:1) drives the **HTTP move harness** (when enabled) and defines move-centric metrics tied to the HTTP path.
- [`websocket-stress.js`](../../tests/load/scenarios/websocket-stress.js:1) measures pure WebSocket transport characteristics using `diagnostic:ping` / `diagnostic:pong`, but **does not exercise gameplay semantics** (no real moves or game state).

### 1.3 This strategy

This strategy introduces **WebSocket gameplay–centric load tests** that:

- Use HTTP only for:
  - Health checks.
  - Authentication.
  - Game creation and, optionally, state readbacks for debugging.
- Use Socket.IO/WebSockets for gameplay events, including:
  - `join_game`
  - `player_move`
  - `player_move_by_id`
  - `game_state`
  - `game_over`
  - (Medium-term) decision-phase events such as `player_choice_required`, `player_choice_response`, and timeout notifications.
- Emit metrics that map **directly** to the gameplay SLOs in [`STRATEGIC_ROADMAP.md` §2.2–2.3](../planning/STRATEGIC_ROADMAP.md:324), including:
  - Human move round-trip latency and stall rate.
  - AI turn latency from the client’s perspective.
  - WebSocket connection success and stability.

---

## 2. Tooling choices

Three main tooling approaches are available for WebSocket gameplay load tests. This section compares them in the context of this repository, then gives a short-term and medium-term recommendation.

### 2.1 k6 WebSocket API

[`k6`](../../tests/load/scenarios/game-creation.js:1) already underpins the current load-testing framework and is used together with **manual Engine.IO/Socket.IO framing** in [`websocket-stress.js`](../../tests/load/scenarios/websocket-stress.js:1).

**Pros (in this repo’s context)**

- **Shared ecosystem with existing load tests**
  - Reuses the k6 configuration, helpers, and CI plumbing already used by:
    - [`game-creation.js`](../../tests/load/scenarios/game-creation.js:1)
    - [`concurrent-games.js`](../../tests/load/scenarios/concurrent-games.js:1)
    - [`player-moves.js`](../../tests/load/scenarios/player-moves.js:1)
    - [`websocket-stress.js`](../../tests/load/scenarios/websocket-stress.js:1)
- **Excellent for synthetic load and concurrency (P-01 scenarios)**
  - Built-in support for large VU counts, ramping, and steady-state phases.
  - Well suited to the P-01 target scales in [`STRATEGIC_ROADMAP.md` §2.2](../planning/STRATEGIC_ROADMAP.md:324).
- **Socket.IO framing code already exists**
  - [`websocket-stress.js`](../../tests/load/scenarios/websocket-stress.js:1) contains working Engine.IO/Socket.IO handshake and framing logic that can be adapted to gameplay events (`join_game`, `player_move_by_id`, etc.).

**Cons**

- **Manual protocol framing is brittle**
  - There is **no direct use of `socket.io-client`**; the k6 script constructs Engine.IO/Socket.IO frames manually.
  - Any future protocol change (Socket.IO version, auth framing, namespace changes) requires hand-updating these scripts.
- **Less ergonomic for complex, stateful, multi-phase games**
  - Managing multi-step decision flows, reconnection semantics, or spectator flows inside a single k6 script becomes verbose.
  - Shared state across VUs is limited to what k6 supports in its execution model.

### 2.2 Node/socket.io-client harness

A second option is a **TypeScript harness** under a directory such as `tests/load/harnesses/`, built using [`socket.io-client`](../../package.json:1) and the shared WebSocket/game types:

- [`websocket.ts`](../../src/shared/types/websocket.ts:1)
- [`game.ts`](../../src/shared/types/game.ts:1)
- [`websocketSchemas.ts`](../../src/shared/validation/websocketSchemas.ts:1)

This harness would act as a programmable client swarm that uses the **real Socket.IO client semantics** and the same schemas as the production React client.

**Pros**

- **Real Socket.IO semantics**
  - Uses `socket.io-client` for connection, reconnection, and namespaced events instead of manual Engine.IO framing.
  - Lower risk of protocol drift; the harness fails in similar ways to the browser client if the protocol changes.
- **Type-safe against shared types and schemas**
  - Can import the canonical types and schemas from:
    - [`websocket.ts`](../../src/shared/types/websocket.ts:1)
    - [`game.ts`](../../src/shared/types/game.ts:1)
    - [`websocketSchemas.ts`](../../src/shared/validation/websocketSchemas.ts:1)
  - Enables strong typing for move payloads, game state updates, and decision events.
- **Best suited for rich, scripted scenarios**
  - Particularly valuable for:
    - Decision-heavy phases (line and territory choices).
    - Complex reconnect/rematch workflows.
    - “AI as a service” experiments with custom traffic patterns.

**Cons**

- **New plumbing and commands for CI**
  - Requires Node-based runners and separate npm scripts (for example, `npm run load:websocket:harness`).
  - Needs explicit integration into the P-01 perf gate described in [`STRATEGIC_ROADMAP.md`](../planning/STRATEGIC_ROADMAP.md:413).
- **Less ergonomic for very high VU counts**
  - Cannot easily match the **pure VU scaling** of k6 for “100s of lightweight connections”.
  - Better suited for dozens to low-hundreds of concurrent simulated clients rather than thousands.

### 2.3 Playwright-based browser E2E

The third option is to extend the existing Playwright E2E infrastructure:

- [`playwright.config.ts`](../../playwright.config.ts:1)
- Existing specs under `tests/e2e` (for example, metrics, reconnection, and game-flow tests).

These tests would use the **real React SPA** and the [`GameContext`](../../src/client/contexts/GameContext.tsx:1) WebSocket client.

**Pros**

- **Highest realism**
  - Exercises the full stack: browser, React app, WebSocket client, backend, rules engine, and AI service.
  - Measures **user-perceived latency** from UI actions (clicks) to rendered board updates.
- **Natural SLO spot checks**
  - Easy to encode assertions like “p95 click-to-board-update latency must be below the staging SLOs in [`STRATEGIC_ROADMAP.md` §2.2](../planning/STRATEGIC_ROADMAP.md:335)”.
  - Ideal as a **perf smoke** to guard against regressions in the WebSocket client or rendering pipeline.

**Cons**

- **Heavy in CI**
  - Requires real browsers (headed or headless) and a running frontend.
  - More expensive in both wall-clock time and resources than k6 or Node-only harnesses.
- **Limited concurrency**
  - Not well suited for **capacity tests**; better used for targeted SLO checks at modest concurrency.
  - Scaling to dozens of parallel browsers is possible but complex.

### 2.4 Recommendation

**Short-term MVP (this subtask’s target):**

- **Primary:**
  - Implement a k6 WebSocket gameplay scenario [`websocket-gameplay.js`](../../tests/load/scenarios/websocket-gameplay.js:1) alongside [`websocket-stress.js`](../../tests/load/scenarios/websocket-stress.js:1), focused on move throughput and stall rates.
- **Supplementary:**
  - Add a small Playwright E2E spec [`websocket-move-latency.e2e.spec.ts`](../../tests/e2e/websocket-move-latency.e2e.spec.ts:1) that measures user-perceived move round-trip latency via the real browser client path.

**Medium-term ideal:**

- Keep k6 as the **canonical synthetic load harness** for WebSocket gameplay, used for P-01-scale concurrency tests and regressions.
- Introduce a strongly typed Node/socket.io-client harness under `tests/load/harnesses/` for:
  - Decision-phase stress tests.
  - Complex scripted flows (timeouts, auto-resolution, rematches, reconnects).
- Maintain a **small, focused set** of Playwright specs as realism and regression guards for WebSocket latency and UX-level correctness.

---

## 3. Scenario matrix

This section defines the initial WebSocket gameplay scenarios S1–S4. Each scenario specifies its purpose, tooling, target environments, concurrency/duration (high level), and key metrics/thresholds.

### S1: Single-game WebSocket gameplay smoke

**Purpose**

- Quick CI/dev smoke that validates the full happy path over WebSockets:
  - Health check → auth → game creation → WebSocket connect → `join_game` → a short sequence of moves → `game_over`.

**Tool / harness**

- k6 WebSocket gameplay scenario [`websocket-gameplay.js`](../../tests/load/scenarios/websocket-gameplay.js:1) in a **“smoke” configuration**.

**Target environments**

- Local dev.
- CI smoke (lightweight P-01 check).

**Concurrency and duration (approximate)**

- 1–3 virtual users (VUs).
- Each VU plays 1–2 games to completion.
- Total wall-clock time: a few minutes (including ramp-up).

**Key metrics and thresholds**

- `ws_move_rtt_ms` (Trend):
  - p95 and p99 should be **comfortably below** the staging-level WebSocket SLOs in [`STRATEGIC_ROADMAP.md` §2.2](../planning/STRATEGIC_ROADMAP.md:335) when run against staging/perf.
- `ws_moves_attempted_total` (Counter):
  - Must be **> 0**; scenario fails if no moves are attempted.
- `ws_move_success_rate` (Rate):
  - Target success rate ≥ **99%** for this small-scale smoke.
- `ws_move_stalled_total` (Counter):
  - Stalls (RTT > 2000 ms) should be **0** in dev/CI.
- WebSocket connection and handshake success:
  - Reuse/extend metrics from [`websocket-stress.js`](../../tests/load/scenarios/websocket-stress.js:1); success rate close to 100% on healthy environments.

### S2: Many small games concurrently (WebSocket move throughput / stall rate)

**Purpose**

- P-01-style concurrency validation focused on **WebSocket move throughput and stall rate** under many small, overlapping games.

**Tool / harness**

- [`websocket-gameplay.js`](../../tests/load/scenarios/websocket-gameplay.js:1) in a **“throughput” configuration** (distinct from S1’s smoke config).

**Target environments**

- Primary: staging and/or dedicated perf environment.
- Optional: scaled-down version in dev for quick experiments (reduced VU counts and durations).

**Concurrency and duration (approximate)**

- Ramp to approximately **20–40 VUs**.
- Sustain target concurrency for **10–15 minutes** of steady state.
- Each VU plays several short games, yielding a high total move count.

**Key metrics and thresholds**

- `ws_move_rtt_ms` (Trend):
  - p95 and p99 must meet or beat the WebSocket gameplay SLOs in [`STRATEGIC_ROADMAP.md` §2.2](../planning/STRATEGIC_ROADMAP.md:335).
- `ws_moves_attempted_total` (Counter):
  - High enough to give statistically meaningful distributions (for example, tens of thousands of moves).
- `ws_move_success_rate` (Rate):
  - Threshold aligned with P-01 error-budget assumptions (for example, ≥ **99.5%** successful moves over the steady-state window).
- `ws_move_stalled_total` (Counter) and stall rate:
  - Stall definition: RTT > 2000 ms, matching [`STRATEGIC_ROADMAP.md` §2.2](../planning/STRATEGIC_ROADMAP.md:346).
  - Stall rate targets:
    - ≤ **0.5%** of moves in staging.
    - ≤ **0.2%** in a dedicated perf environment.
- WebSocket connection/handshake metrics (from [`websocket-stress.js`](../../tests/load/scenarios/websocket-stress.js:1)):
  - Handshake success > **98%**.
  - Connection success > **95%**, with investigation required for any systematic failures.

### S3: AI games over WebSockets (human vs AI)

**Purpose**

- Validate **human-vs-AI gameplay over WebSockets**, including AI service calls and client-observed AI turn latency.
- Connect WebSocket move and AI-turn metrics to the AI SLOs in [`STRATEGIC_ROADMAP.md` §2.3](../planning/STRATEGIC_ROADMAP.md:353).

**Tool / harness**

- Variant of [`websocket-gameplay.js`](../../tests/load/scenarios/websocket-gameplay.js:1) that creates and drives **human-vs-AI** games.
  - AI seating and difficulty configuration should mirror the HTTP harness behaviour in [`player-moves.js`](../../tests/load/scenarios/player-moves.js:1).

**Target environments**

- Staging or perf environments where the AI service is running with **realistic capacity** and production-like configuration.

**Concurrency and duration (approximate)**

- Moderate concurrency (for example, 10–20 VUs) to generate:
  - A mix of human and AI turns.
  - Enough AI turns to produce stable latency histograms.
- Duration aligned with P-01 runs (for example, 10–15 minutes steady state).

**Key metrics and thresholds**

- Human move RTT: `ws_move_rtt_ms` (Trend)
  - Same SLO mapping as S2 for human moves.
- AI turn latency: `ws_ai_turn_latency_ms` (Trend)
  - Definition: time from the client observing “AI turn started” to receiving the authoritative `game_state` that reflects the AI move.
  - Targets aligned with AI SLOs in [`STRATEGIC_ROADMAP.md` §2.3](../planning/STRATEGIC_ROADMAP.md:353) for end-to-end AI turn latency.
- Error and fallback rates:
  - Counters for AI failures or fallbacks (for example, surfacing `ai_fallback_total` from backend metrics).
  - Should remain within the AI fallback SLOs in [`STRATEGIC_ROADMAP.md` §2.3](../planning/STRATEGIC_ROADMAP.md:390).

### S4: Decision-phase stress (line / territory choices under load) – medium-term

**Purpose**

- Stress **decision-heavy flows** over WebSockets, including:
  - `player_choice_required` / `player_choice_response` events for line and territory decisions.
  - Timeouts and auto-resolution events such as `decision_phase_timeout_warning` and `decision_phase_timed_out`.
- Validate that decision-phase latency, timeouts, and auto-resolution semantics behave correctly under load.

**Tool / harness (medium-term)**

- Node/socket.io-client harness under `tests/load/harnesses/` that uses shared types and scripted decision flows rather than k6:
  - Types and schemas from:
    - [`websocket.ts`](../../src/shared/types/websocket.ts:1)
    - [`game.ts`](../../src/shared/types/game.ts:1)
    - [`websocketSchemas.ts`](../../src/shared/validation/websocketSchemas.ts:1)
  - Scenarios that explicitly drive multi-step decision sequences for many concurrent games.

**Target environments**

- Initially: dev and CI for functional stress and correctness.
- Later: staging/perf for focused “decision-phase drills” (short, high-intensity tests).

**Key metrics and thresholds**

- Decision RTT: `ws_decision_rtt_ms` (Trend)
  - Time from emitting a decision (`player_choice_response`) to receiving the corresponding `game_state` and/or decision-resolution event.
- `ws_decision_stalled_total` (Counter):
  - Count of decisions whose RTT exceeds a stall threshold (for example, 2000 ms, mirroring the move stall definition in [`STRATEGIC_ROADMAP.md` §2.2](../planning/STRATEGIC_ROADMAP.md:346)).
- Auto-resolution and rejection counters:
  - Counts of auto-resolved decisions, rejected choices, and timeout events (for example, CHOICE_REJECTED codes and `decision_phase_timed_out` events).

---

## 4. Metrics and SLO mapping

This section defines the core metrics expected from the k6 WebSocket gameplay scenario, the Node/socket.io-client harness, and the Playwright E2E spec, and explains how they map to the SLOs in [`STRATEGIC_ROADMAP.md` §2.2–2.3](../planning/STRATEGIC_ROADMAP.md:324).

### 4.1 WebSocket move latency and success

- `ws_move_rtt_ms` (Trend)
  - **Definition:** Time from sending a WebSocket move (`player_move` or `player_move_by_id`) to receiving the next authoritative `game_state` that reflects that move.
  - **SLO mapping:**
    - Directly corresponds to “human move submission → authoritative broadcast” latency in [`STRATEGIC_ROADMAP.md` §2.2](../planning/STRATEGIC_ROADMAP.md:335).
    - S1/S2/S3 scenarios should compute p95/p99 and compare to the environment-specific SLOs (staging vs production/perf).

- `ws_moves_attempted_total` (Counter)
  - **Definition:** Total number of WebSocket move attempts across all VUs / clients.
  - Used to ensure tests actually exercise gameplay and to provide denominators for success and stall rates.

- `ws_move_success_rate` (Rate)
  - **Definition:**
    - Numerator: moves that result in the expected `game_state` transition without error codes.
    - Denominator: `ws_moves_attempted_total`.
  - **Success criteria:**
    - No `MOVE_REJECTED`, `ACCESS_DENIED`, or `INTERNAL_ERROR` codes associated with the move.
  - **SLO mapping:**
    - Ties into the WebSocket availability/error-budget targets in [`STRATEGIC_ROADMAP.md` §2.4](../planning/STRATEGIC_ROADMAP.md:397).

- `ws_move_stalled_total` (Counter)
  - **Definition:**
    - Count of moves where `ws_move_rtt_ms` exceeds **2000 ms**, matching the stall definition for human moves in [`STRATEGIC_ROADMAP.md` §2.2](../planning/STRATEGIC_ROADMAP.md:346).
  - **Derived metrics:**
    - Stall rate = `ws_move_stalled_total / ws_moves_attempted_total`.
  - **Targets:**
    - ≤ **0.5%** stall rate in staging.
    - ≤ **0.2%** in perf/production-style environments.

### 4.2 Error code counters

The WebSocket gameplay protocol defines structured error codes in [`websocket.ts`](../../src/shared/types/websocket.ts:52). Load tests should expose per-code counters, for example:

- `ws_error_move_rejected_total`
- `ws_error_access_denied_total`
- `ws_error_internal_error_total`
- Additional counters per relevant error category (for example, invalid state, invalid payload, rate-limited, etc.).

These counters should be tagged (for example, by `gameId`, environment, scenario name) so they can be correlated with backend metrics and logs during P-01 runs.

### 4.3 Connection-level metrics

The new WebSocket gameplay scenarios should **reuse and extend** the connection metrics already emitted by [`websocket-stress.js`](../../tests/load/scenarios/websocket-stress.js:1), including:

- Connection success rate (for example, `websocket_connection_success_rate`).
- Handshake success rate (for example, `websocket_handshake_success_rate`).
- Ping/pong or message-level latency (for example, `websocket_message_latency_ms`) for low-level protocol health.

These transport metrics complement the **gameplay-level** metrics (`ws_move_rtt_ms` etc.) and should be used together to attribute latency or failure spikes to either transport issues or game/AI logic.

### 4.4 Decision-phase metrics (for S4)

For decision-heavy flows, the Node/socket.io-client harness should emit:

- `ws_decision_rtt_ms` (Trend)
  - Time from emitting `player_choice_response` to receiving the corresponding `game_state` or terminal decision event.
  - Aligned with decision metadata and `GameStateUpdateMeta` semantics in [`websocketSchemas.ts`](../../src/shared/validation/websocketSchemas.ts:1).

- `ws_decision_stalled_total` (Counter)
  - Count of decisions whose RTT exceeds a configured stall threshold (for example, 2000 ms).
  - Similar stall-rate targets to move stalls (≤0.5% staging, ≤0.2% perf).

- Auto-resolution and error counters
  - For example, counts of:
    - Auto-resolved decisions (timeout-based).
    - CHOICE_REJECTED errors.
    - Decision-phase internal errors.

These metrics allow S4 to stress the **decision lifecycle** specifically, without being conflated with normal move latency.

### 4.5 Playwright E2E metrics

The Playwright spec [`websocket-move-latency.e2e.spec.ts`](../../tests/e2e/websocket-move-latency.e2e.spec.ts:1) should measure **user-perceived** latency:

- Click-to-board-update timings:
  - Time from a user action (for example, clicking a valid move on the board) to the UI reflecting the new `game_state`.
  - Aggregated as a Trend (for example, `browser_ws_move_rtt_ms`) with p95/p99 compared against staging-level SLOs in [`STRATEGIC_ROADMAP.md` §2.2](../planning/STRATEGIC_ROADMAP.md:335).
- Simple assertion-style thresholds:
  - For example: p95 ≤ **300 ms**, p99 ≤ **600 ms** under typical staging load.

---

## 5. Implementation plan for MVP

This section describes what Code mode should implement for the MVP WebSocket gameplay load testing. **No code is implemented here**; this is an execution-ready plan.

### 5.1 New k6 WebSocket gameplay scenario

**File:** [`websocket-gameplay.js`](../../tests/load/scenarios/websocket-gameplay.js:1)

**High-level behaviour**

- **`setup()` phase:**
  - Perform a health check via `GET /health`.
  - Authenticate using [`loginAndGetToken()`](../../tests/load/auth/helpers.js:1) to obtain a JWT suitable for WebSocket auth and HTTP calls.

- **Per-VU execution:**
  - Use `POST /api/games` to create AI-capable games (for example, human vs AI).
  - Derive the WebSocket URL (for example, `/socket.io`) from the base URL and reuse the Engine.IO/Socket.IO handshake logic from [`websocket-stress.js`](../../tests/load/scenarios/websocket-stress.js:1).
  - Connect over the Socket.IO protocol using the JWT for auth.
  - Emit `join_game` for each created game, waiting for an initial `game_state`.
  - On each `game_state`:
    - Determine whether it is this player’s turn.
    - Extract `validMoves` and choose a `moveId` (simple selection strategy is acceptable for load tests).
    - Emit `player_move_by_id` with the chosen `moveId`.
    - Measure `ws_move_rtt_ms` as the time from emitting the move to the next `game_state` that reflects it.
  - Continue until `game_over` is observed or a maximum move count is reached.

**Metrics**

- Emit the metrics defined in section 4, at minimum:
  - `ws_move_rtt_ms` (Trend).
  - `ws_moves_attempted_total`, `ws_move_success_rate`, `ws_move_stalled_total`.
  - Error code counters (for example, `ws_error_move_rejected_total`).
  - Connection/handshake metrics aligned with [`websocket-stress.js`](../../tests/load/scenarios/websocket-stress.js:1).

**Scenarios / configurations**

- **S1-like smoke config:**
  - 1–3 VUs, 1–2 games per VU, short duration.
  - Intended for dev and CI smoke jobs.
- **S2-like throughput config:**
  - 20–40 VUs, 10–15 minutes steady state.
  - Intended for staging/perf runs as part of the P-01 performance gate in [`STRATEGIC_ROADMAP.md`](../planning/STRATEGIC_ROADMAP.md:592).

### 5.2 Playwright WebSocket move latency spec

**File:** [`websocket-move-latency.e2e.spec.ts`](../../tests/e2e/websocket-move-latency.e2e.spec.ts:1)

**Behaviour**

- Use existing Playwright helpers and fixtures (see `tests/e2e` and [`playwright.config.ts`](../../playwright.config.ts:1)) to:
  - Log in as a test user via the normal UI or API.
  - Create a **human-vs-AI** game using the same API surfaces as the SPA.
  - Navigate to the game page and wait until the WebSocket connection is active (for example, via UI state or network inspection).
- Within the browser context:
  - Instrument moves by:
    - Recording the timestamp at user action (clicking a valid move).
    - Waiting for the UI/board to reflect the new `game_state`.
    - Recording the completion timestamp and computing a per-move RTT.
  - Collect per-move timings into an array and compute p95/p99.
  - Assert that these timings meet staging-level SLOs (for example, p95 ≤ 300 ms, p99 ≤ 600 ms), consistent with [`STRATEGIC_ROADMAP.md` §2.2](../planning/STRATEGIC_ROADMAP.md:335).

**Output**

- Test should:
  - Fail loudly if SLOs are violated, with clear logging of outlier moves.
  - Optionally emit structured timing data to a JSON artifact for offline analysis.

### 5.3 CI / staging wiring

**k6 WebSocket gameplay scenario**

- Add npm scripts in [`package.json`](../../package.json:1) such as:
  - `"load:websocket:gameplay:smoke": "k6 run tests/load/scenarios/websocket-gameplay.js --env MODE=smoke"`
  - `"load:websocket:gameplay:throughput": "k6 run tests/load/scenarios/websocket-gameplay.js --env MODE=throughput"`
- Wire the **smoke** variant into CI as a fast check (for example, part of a “perf smoke” workflow).
- Run the **throughput** variant in staging/perf as part of the P-01 performance gate.

**Playwright spec**

- Tag [`websocket-move-latency.e2e.spec.ts`](../../tests/e2e/websocket-move-latency.e2e.spec.ts:1) appropriately (for example, with a `@perf-smoke` annotation or a Playwright project name).
- Add a CI job that:
  - Brings up the full stack (backend + frontend + AI service) in a staging-like configuration.
  - Runs only the perf-smoke tagged specs, including the WebSocket move latency test.
  - Publishes timing summaries alongside k6 results for end-to-end visibility.

---

## 6. Risks and limitations

This section outlines key risks and trade-offs for WebSocket gameplay load testing, with suggested mitigations.

### 6.1 Dev environment rate limiting and capacity

- As observed in the HTTP harness runs in [`LOAD_TEST_BASELINE_REPORT.md`](LOAD_TEST_BASELINE_REPORT.md:1), the local/dev environment can hit adaptive rate limits (for example, on `POST /api/games`) and even transient connection failures under aggressive k6 patterns.
- Running S2-style throughput scenarios in dev risks producing misleading failures (rate limiting, connection refused) that are environment artefacts rather than true server-side bottlenecks.

**Mitigations**

- Use **lower VU counts** and shorter durations for dev/CI runs; reserve higher concurrency for staging or dedicated perf environments.
- Consider environment-specific rate-limit tuning or test-only overrides when running WebSocket gameplay scenarios in dev.

### 6.2 AI capacity effects on latency

- S3-style human-vs-AI scenarios couple WebSocket move timing with AI-service performance.
- High AI load, slow models, or AI fallbacks will directly impact:
  - `ws_ai_turn_latency_ms` (client-observed AI turns).
  - Per-move `ws_move_rtt_ms` for AI turns, even if WebSocket transport is healthy.

**Mitigations**

- Interpret S3 metrics **alongside AI service metrics** from [`ai-service/app/metrics.py`](../../ai-service/app/metrics.py:1) and Node-side metrics in [`rulesParityMetrics`](../../src/server/utils/rulesParityMetrics.ts:1).
- When AI is the dominant contributor to latency, focus on scaling AI capacity or reducing AI evaluation cost rather than attributing issues to WebSockets.

### 6.3 Complexity of decision-phase scripting

- Decision phases (line and territory choices, timeouts, auto-resolution) have complex state machines and timing semantics.
- Encoding these flows directly in k6 with manual Socket.IO framing would be error-prone and hard to maintain.

**Mitigations**

- Treat S4 as a **medium-term** goal and implement it using the Node/socket.io-client harness, where rich scripting and shared types/schemas are available.
- Keep k6-focused scenarios (S1–S3) centred on **move flows** and simpler gameplay patterns.

### 6.4 Protocol drift with manual Engine.IO/Socket.IO framing

- The k6 scenarios depend on hand-crafted Engine.IO/Socket.IO frames, which can drift from the server and browser client if:
  - Socket.IO is upgraded.
  - Auth or namespace conventions change.
  - New events or metadata are added.

**Mitigations**

- Centralise Socket.IO framing helpers in a single module (for example, a shared helper imported by both [`websocket-stress.js`](../../tests/load/scenarios/websocket-stress.js:1) and [`websocket-gameplay.js`](../../tests/load/scenarios/websocket-gameplay.js:1)).
- Document and pin the expected Socket.IO/Engine.IO versions in [`package.json`](../../package.json:1) and/or a short protocol note, and review WebSocket load scripts whenever these dependencies change.
- Consider gradually moving the most complex WebSocket flows (for example, S4 decision phases) into the Node/socket.io-client harness where protocol details are handled by `socket.io-client`.

### 6.5 Observability gaps between tools

- k6, Node harnesses, and Playwright may each emit slightly different metric names and tags, complicating cross-tool comparisons.

**Mitigations**

- Standardise metric names and tags as described in section 4, especially for:
  - Move RTT and stall metrics.
  - Decision RTT and stall metrics.
  - Connection and handshake health.
  - Error-code counters derived from [`websocket.ts`](../../src/shared/types/websocket.ts:52).
- Add short runbook snippets (for example, under `docs/runbooks/`) showing how to interpret and correlate k6, Node, Playwright, and backend/AI metrics during P-01 runs.

---

End of document.
