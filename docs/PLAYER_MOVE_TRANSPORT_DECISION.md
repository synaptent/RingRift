# Player Move Transport Architecture Decision (HTTP vs WebSocket)

> Status: Accepted
> Date: 2025-12-04
> Owner Modes: Architect + Code + Debug
>
> Related docs:
>
> - [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md:1)
> - [`RULES_ENGINE_ARCHITECTURE.md`](../RULES_ENGINE_ARCHITECTURE.md:1)
> - [`API_REFERENCE.md`](./API_REFERENCE.md:1)
> - [`player-moves.js`](../tests/load/scenarios/player-moves.js:1)
> - [`LOAD_TEST_BASELINE_REPORT.md`](./LOAD_TEST_BASELINE_REPORT.md:1)

---

## 1. Context

### 1.1 Current gameplay networking architecture

- **WebSocket gameplay protocol (canonical today):**
  - Socket.IO v4 server under [`WebSocketServer`](../src/server/websocket/server.ts:1).
  - Per-game interaction pipeline in [`WebSocketInteractionHandler.ts`](../src/server/game/WebSocketInteractionHandler.ts:1).
  - Session and state orchestration in [`GameSessionManager.ts`](../src/server/game/GameSessionManager.ts:1) and [`TurnEngine.ts`](../src/server/game/turn/TurnEngine.ts:1), backed by the canonical turn orchestrator in [`turnOrchestrator.ts`](../src/shared/engine/orchestration/turnOrchestrator.ts:1).
  - Real-time client protocol documented in [`CANONICAL_ENGINE_API.md`](./CANONICAL_ENGINE_API.md:1) and surfaced in the WebSocket section of [`API_REFERENCE.md`](./API_REFERENCE.md:1).
- **HTTP APIs (supporting surfaces):**
  - Auth, game creation, game state readbacks, and diagnostics provided via [`src/server/routes`](../src/server/routes/index.ts:1), including [`game.ts`](../src/server/routes/game.ts:1).
  - REST API contract documented in [`API_REFERENCE.md`](./API_REFERENCE.md:1).
- **Domain move semantics:**
  - All rules semantics and turn sequencing live in the shared engine and orchestrator stack under [`src/shared/engine`](../src/shared/engine/types.ts:1) and the backend host described in [`RULES_ENGINE_ARCHITECTURE.md`](../RULES_ENGINE_ARCHITECTURE.md:1).
  - Server-side gameplay hosts (including [`GameSessionManager.ts`](../src/server/game/GameSessionManager.ts:1)) ultimately delegate move application to the orchestrator-backed domain API (conceptually `GameSessionManager.applyMove()`), with WebSocket as the public transport.

### 1.2 Existing k6 load tests

- k6 scenarios live under [`tests/load/scenarios`](../tests/load/scenarios/game-creation.js:1):
  - [`game-creation.js`](../tests/load/scenarios/game-creation.js:1) – create-game and GET `/api/games/:gameId` performance.
  - [`concurrent-games.js`](../tests/load/scenarios/concurrent-games.js:1) – many simultaneous active games and state polling.
  - [`player-moves.js`](../tests/load/scenarios/player-moves.js:1) – intended to exercise move submission and turn processing.
  - `websocket-stress.js` – WebSocket connection and ping-pong stress (transport-level only).
- WebSocket gameplay SLOs and stall definitions are specified in [`STRATEGIC_ROADMAP.md` §2.2](../STRATEGIC_ROADMAP.md:324).
- The `player-moves` scenario defines custom metrics aligned to those SLOs:
  - `move_submission_latency_ms`
  - `move_submission_success_rate`
  - `stalled_moves_total` (moves taking >2s)
  - `turn_processing_latency_ms`

### 1.3 Current limitation: HTTP moves disabled

- Today, **all production gameplay moves are carried over WebSockets** via `player_move` / `player_move_by_id` events (see the WebSocket API section of [`API_REFERENCE.md`](./API_REFERENCE.md:489)).
- The HTTP-based move submission path in [`player-moves.js`](../tests/load/scenarios/player-moves.js:1) is intentionally guarded:
  - `const MOVE_HTTP_ENDPOINT_ENABLED = false;` in [`player-moves.js`](../tests/load/scenarios/player-moves.js:86) keeps the hypothetical `POST /api/games/:gameId/moves` path inactive.
  - No HTTP move endpoint is currently implemented in [`game.ts`](../src/server/routes/game.ts:1).
- As a result:
  - The `player-moves` scenario acts as **"game creation + polling under load"** rather than a true move-throughput test.
  - Move-related k6 metrics remain effectively zero and thresholds trivially pass, even though real WebSocket move paths are not being exercised end-to-end.

### 1.4 Goals and non-goals

- **Goals**
  - Make WebSocket the **single canonical move transport** for interactive game clients.
  - Provide a narrowly scoped **HTTP move harness** for:
    - k6 and other HTTP-centric load tools.
    - Internal operations tools and diagnostics.
  - Ensure both transports share a **single domain `applyMove` API** so there is exactly one place where move semantics live.
- **Non-goals**
  - Expose a second fully featured, public HTTP move API for general clients.
  - Change the canonical **rules/turn semantics SSoT**, which remains the shared engine + orchestrator stack described in [`RULES_ENGINE_ARCHITECTURE.md`](../RULES_ENGINE_ARCHITECTURE.md:1) and [`CANONICAL_ENGINE_API.md`](./CANONICAL_ENGINE_API.md:1).

---

## 2. Options considered

### Option A: WebSocket as the only move transport

**Description:** All move submission and game updates use the Socket.IO/WebSocket protocol. No HTTP move endpoint exists, including for internal tools.

**Pros**

- Single canonical protocol and code path for gameplay.
- Efficient, bidirectional, low-latency channel well suited to real-time games.
- Matches existing implementation and the WebSocket gameplay SLOs in [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md:324).
- Simplifies auth and authorization surface: only the WebSocket channel needs to be hardened for gameplay moves.

**Cons**

- Harder to drive via traditional HTTP-only load tools; requires:
  - k6 WebSocket primitives, or
  - a separate socket.io-client / Playwright-based harness.
- Observability and diagnostics for move submission must all flow through WebSocket tooling.
- Internal tools that expect simple HTTP endpoints must either:
  - embed WebSocket clients, or
  - call through a separate service that speaks WebSockets on their behalf.

### Option B: HTTP as a full, public alternative to WebSocket for moves

**Description:** Introduce a first-class `POST /api/games/:gameId/moves` HTTP endpoint intended for general client use, with semantics equivalent to `player_move` events, and treat HTTP and WebSocket as co-equal public transports.

**Pros**

- Straightforward to script and test with almost any tool (curl, k6, Postman, etc.).
- Stateless, per-request semantics are familiar to many infra/ops teams.
- Easier to expose through API gateways or HTTP-only infrastructure.

**Cons**

- **Two public APIs for the same domain operation** (HTTP and WebSocket) increases:
  - Drift risk between transports (edge-case behaviour, timing, error handling).
  - Long-term maintenance and documentation burden.
  - Surface area for auth, rate limiting, and abuse prevention.
- Still need a real-time push/update channel for board state and decisions:
  - WebSocket (or SSE/long polling) must exist anyway for game updates.
- Encourages clients to treat HTTP as the "primary" move path (easier to adopt) even though WebSocket is operationally better suited for real-time play.

### Option C (Chosen): WebSocket as canonical, HTTP as thin internal/test harness

**Description:** Keep WebSocket as the only **public** move transport for interactive clients, and introduce an HTTP move endpoint only as a **thin, clearly internal/test-only harness** layered over the same domain `applyMove` API.

**Pros**

- Single domain implementation for move semantics (shared engine + orchestrator).
- WebSocket remains the authoritative client protocol for gameplay, matching existing production behaviour and SLOs.
- HTTP harness serves load tests, CI, and internal tools **without** becoming a second product API.
- Clear story for parity: both WebSocket and HTTP harness call the same `GameSessionManager.applyMove()` path and emit the same rules events.

**Cons**

- Requires explicit scoping and guardrails (feature flags, environment checks, network controls) to prevent the harness from being treated as a public API.
- Slightly more configuration complexity:
  - Different behaviour by environment (local, CI, staging, production).
  - Additional metrics and alerts for harness misuse.
- Load tests that want to validate the exact WebSocket move path still need a WebSocket-capable tool in addition to the HTTP harness.

---

## 3. Decision

The following decisions are **accepted** and should be treated as the canonical source of truth for player move transport:

1. **Canonical transport**
   - WebSocket remains the **canonical, long-term move transport** for all interactive game clients.
   - Client applications (web, mobile, desktop) **must** submit moves over the WebSocket protocol (`player_move` / `player_move_by_id` events) as documented in [`CANONICAL_ENGINE_API.md`](./CANONICAL_ENGINE_API.md:1) and the WebSocket section of [`API_REFERENCE.md`](./API_REFERENCE.md:1).

2. **Single domain API for move semantics**
   - All move semantics are implemented and validated in a **single, shared domain API**, conceptually:
     - [`GameSessionManager.applyMove()`](../src/server/game/GameSessionManager.ts:1), which:
       - Accepts a validated, canonical `Move`/`PlayerChoice` payload.
       - Delegates to the orchestrator-backed rules engine stack described in [`RULES_ENGINE_ARCHITECTURE.md`](../RULES_ENGINE_ARCHITECTURE.md:1).
       - Publishes authoritative state updates and notifications to interested channels (WebSocket sessions, observers, metrics).
   - Both of the following call sites must use this same domain API:
     - WebSocket move handlers in [`WebSocketInteractionHandler.ts`](../src/server/game/WebSocketInteractionHandler.ts:1).
     - Any HTTP move harness endpoint implemented in [`game.ts`](../src/server/routes/game.ts:1).

3. **HTTP move harness semantics**
   - An HTTP endpoint of the form:
     - `POST /api/games/:gameId/moves`
   - **may be implemented** as a **thin adapter** over [`GameSessionManager.applyMove()`](../src/server/game/GameSessionManager.ts:1) with the following constraints:
     - The harness performs:
       - Authentication using the same JWT/session model as other game routes.
       - Authorization consistent with game-seat/spectator rules from [`CANONICAL_ENGINE_API.md`](./CANONICAL_ENGINE_API.md:1).
       - Request validation using the same schemas as the WebSocket payloads.
     - The harness does **not** introduce new move semantics, extra side effects, or divergent error codes; on success it returns the same effective result that the WebSocket handler would have produced.
   - The harness is **not** a general public API:
     - It is intended for **internal/test harness** use only (k6, CI, ops tools).
     - It should be clearly documented as such in [`API_REFERENCE.md`](./API_REFERENCE.md:1) under an "Internal / Test harness APIs" section.

4. **Environment and feature-flag controls**
   - Harness availability is controlled by an explicit feature flag, conceptually:
     - `ENABLE_HTTP_MOVE_HARNESS` (name subject to implementation details).
   - Recommended behaviour:
     - **Local / CI / dedicated loadtest environments:**
       - Harness **enabled by default** to simplify instrumentation and experimentation.
     - **Staging:**
       - Harness **enabled**, but restricted to:
         - Authenticated users with an explicit "loadtest" or "ops" role, and/or
         - IP ranges or VPN networks reserved for SRE/engineering.
     - **Production:**
       - Harness **disabled by default**.
       - If enabled for specific drills, access must be tightly scoped (auth scopes, network controls) and time-bounded.
   - Any change to harness exposure in production must be treated like an API surface change and reviewed under the security model in [`SECURITY_THREAT_MODEL.md`](./SECURITY_THREAT_MODEL.md:1).

5. **k6 player-moves scenario behaviour**
   - [`player-moves.js`](../tests/load/scenarios/player-moves.js:1) remains the canonical HTTP-oriented load test for move and turn performance.
   - Its behaviour is defined as:
     - When the harness is **enabled**:
       - Use `POST /api/games/:gameId/moves` to submit **real moves** over HTTP.
       - Record:
         - `move_submission_latency_ms` for end-to-end HTTP move latency.
         - `turn_processing_latency_ms` as a proxy for move-to-update latency.
         - `stalled_moves_total` for moves exceeding the stall threshold from [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md:344).
         - `move_submission_success_rate` and an explicit `moves_attempted_total` counter.
     - When the harness is **disabled**:
       - The scenario:
         - Continues to create games and poll state over HTTP.
         - Logs a clear message indicating that it is running in **"game creation + polling only"** mode.
       - Move-specific metrics may not be emitted or may be explicitly marked as "harness disabled" in their tags.

---

## 4. Consequences

### 4.1 Positive consequences

- **Single source of truth for move semantics**
  - All move logic flows through the shared orchestrator stack and a single backend domain API, regardless of transport.
  - Reduces the risk of subtle behaviour differences between WebSocket and HTTP paths.
- **Clear production SLO mapping**
  - WebSocket gameplay SLOs in [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md:324) now map unambiguously to the actual production move path.
  - Load-test SLOs for moves and turns can be expressed in terms of:
    - WebSocket round-trip latencies.
    - HTTP harness to `applyMove` latency when the harness is enabled.
- **Robust testing and tooling story**
  - k6 and other HTTP-only tools have a first-class way to exercise move throughput and latency without bypassing the canonical rules logic.
  - Internal diagnostics (for example, admin tools or replay harnesses) can call the HTTP harness rather than reimplementing move logic.

### 4.2 Negative consequences and trade-offs

- **Harness surface must be tightly controlled**
  - The HTTP harness expands the attack and abuse surface if exposed broadly.
  - Operators must ensure it does **not** become a de facto public API for untrusted clients.
  - Any misconfiguration that exposes the harness without appropriate auth/rate limiting could be used for scripted abuse (for example, move spamming).
- **Complexity in environment configuration**
  - Additional feature flags and environment-specific configuration must be maintained and documented in [`ENVIRONMENT_VARIABLES.md`](./ENVIRONMENT_VARIABLES.md:1) and deployment runbooks.
  - CI and staging pipelines need to set harness flags consistently so benchmarks are reproducible.
- **Dual-path testing requirements**
  - Even with an HTTP harness, WebSocket-based load tests remain necessary to validate:
    - Move latency as observed by real clients over WebSockets.
    - Connection churn, reconnection, and spectator behaviour at scale.

### 4.3 Testing implications

- **Load tests**
  - HTTP-based:
    - [`player-moves.js`](../tests/load/scenarios/player-moves.js:1) drives `POST /api/games/:gameId/moves` when the harness is enabled.
    - [`game-creation.js`](../tests/load/scenarios/game-creation.js:1) and [`concurrent-games.js`](../tests/load/scenarios/concurrent-games.js:1) continue to focus on game creation and state polling.
  - WebSocket-based:
    - Existing `websocket-stress.js` and future socket.io/Playwright scenarios validate the canonical WebSocket path for move latency and stall behaviour.
- **Contract and parity tests**
  - WebSocket and HTTP harness paths must:
    - Use the same input validation schemas for move payloads.
    - Produce identical results (state transitions, error codes) when given the same move.
  - Regression suites should assert that:
    - A move accepted over WebSockets would also be accepted via the HTTP harness for the same authenticated player.
    - Error categories (for example, `GAME_INVALID_MOVE`, `GAME_NOT_YOUR_TURN`) remain aligned across transports.

### 4.4 Security and access model

- The HTTP move harness is treated as an **internal** surface:
  - Only reachable from:
    - Trusted networks (VPN, internal VPCs), and/or
    - Authenticated principals with appropriate roles/scopes.
  - Not intended for untrusted public internet clients or third-party integrations.
- WebSockets remain the **only** supported public move channel for first-party clients.
- Security reviews for the harness should:
  - Reuse the threat categories from [`SECURITY_THREAT_MODEL.md`](./SECURITY_THREAT_MODEL.md:1).
  - Ensure rate limiting and abuse detection mirror or exceed protections on the WebSocket move handlers.

---

## 5. Implementation outline and follow-up tasks

The following tasks are **out of scope** for this document but are expected to be implemented by Code/Debug mode in follow-up subtasks. They are listed here to bind this decision to concrete next steps.

1. **Implement HTTP move harness endpoint**
   - Add `POST /api/games/:gameId/moves` to [`game.ts`](../src/server/routes/game.ts:1) as a thin adapter over [`GameSessionManager.applyMove()`](../src/server/game/GameSessionManager.ts:1).
   - Wire an `ENABLE_HTTP_MOVE_HARNESS` feature flag and environment-specific defaults into the route registration.
   - Ensure route-level auth and authorization logic matches the WebSocket move handlers.

2. **Update k6 `player-moves` scenario**
   - Replace the hard-coded `MOVE_HTTP_ENDPOINT_ENABLED = false` in [`player-moves.js`](../tests/load/scenarios/player-moves.js:86) with an environment-driven toggle (for example, `__ENV.MOVE_HTTP_ENDPOINT_ENABLED`).
   - Implement real move submission against `POST /api/games/:gameId/moves`, generating legal moves based on the current game state rather than the current placeholder payload.
   - Add a `moves_attempted_total` counter and tighten thresholds so the scenario fails if:
     - No moves are attempted, or
     - Move success rate or stall metrics violate the SLOs in [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md:324).

3. **Design WebSocket-based move load tests**
   - Extend existing WebSocket stress tooling or introduce a Playwright-based harness that:
     - Authenticates real clients.
     - Joins games and submits moves over WebSockets.
     - Measures end-to-end move latency and stall rates consistent with [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md:324).
   - Ensure these tests run (at least) in staging as part of the P-01 performance gate defined in [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md:590).

4. **Document harness configuration and runbooks**
   - Update [`ENVIRONMENT_VARIABLES.md`](./ENVIRONMENT_VARIABLES.md:1) with:
     - `ENABLE_HTTP_MOVE_HARNESS` and any related flags.
     - Recommended defaults per environment.
   - Add runbook entries (for example, under `docs/runbooks/`) describing:
     - How to safely enable the harness for a specific load test.
     - How to verify that it is disabled again afterwards.
     - How to interpret and react to harness-specific metrics and alerts.

---

## 6. Summary

- WebSockets are the **authoritative move transport** for RingRift gameplay.
- All move semantics are implemented once in the shared engine + orchestrator stack and exposed through a single backend domain API used by both WebSocket handlers and any HTTP harness endpoint.
- An HTTP `POST /api/games/:gameId/moves` endpoint may exist, but **only** as an internal/test harness with tight scoping and feature-flag control.
- Load-testing and operational tooling should rely on:
  - The HTTP move harness where it simplifies scripting and analysis.
  - WebSocket-native scenarios where precise end-to-end gameplay SLO validation is required.
