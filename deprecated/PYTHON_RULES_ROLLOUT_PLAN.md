# Python Rules Rollout Plan (P0.5)

This document defines **acceptance criteria**, **operational steps**, and the **design for python-authoritative mode** for the RingRift rules engine.

It builds on the existing implementation and tests:

- TypeScript rules engine (canonical spec) under:
  - [`src/shared/engine`](src/shared/engine/core.ts)
  - [`src/server/game/GameEngine.ts`](src/server/game/GameEngine.ts)
  - [`src/server/game/RuleEngine.ts`](src/server/game/RuleEngine.ts)
- Python rules engine and FastAPI service:
  - [`ai-service/app/game_engine.py`](ai-service/app/game_engine.py)
  - [`ai-service/app/board_manager.py`](ai-service/app/board_manager.py)
  - [`ai-service/app/main.py`](ai-service/app/main.py) (`/rules/evaluate_move`)
- Cross-engine adapter and parity instrumentation:
  - [`src/server/services/PythonRulesClient.ts`](src/server/services/PythonRulesClient.ts)
  - [`src/server/game/RulesBackendFacade.ts`](src/server/game/RulesBackendFacade.ts)
  - [`src/server/utils/rulesParityMetrics.ts`](src/server/utils/rulesParityMetrics.ts)
  - [`src/server/index.ts`](src/server/index.ts) (`/metrics` endpoint)
- Tests:
  - Python:
    - [`ai-service/tests/test_rules_evaluate_move.py`](ai-service/tests/test_rules_evaluate_move.py)
  - TypeScript:
    - [`tests/unit/PythonRulesClient.test.ts`](tests/unit/PythonRulesClient.test.ts)
    - [`tests/unit/RulesBackendFacade.test.ts`](tests/unit/RulesBackendFacade.test.ts)
    - [`tests/unit/RulesBackendFacade.fixtureParity.test.ts`](tests/unit/RulesBackendFacade.fixtureParity.test.ts)
    - [`tests/unit/WebSocketServer.rulesBackend.integration.test.ts`](tests/unit/WebSocketServer.rulesBackend.integration.test.ts)

The rollout is structured around the environment variable `RINGRIFT_RULES_MODE`:

- `ts` (default) – TS rules engine authoritative, Python unused.
- `shadow` – TS authoritative, Python called in shadow for parity.
- `python` – **future**: Python authoritative, TS in shadow as regression net.

---

## 1. Mode Semantics (current state)

### 1.1 `RINGRIFT_RULES_MODE=ts`

- All rules decisions use the TS backend:
  - `GameEngine.makeMove` / `makeMoveById` in [`src/server/game/GameEngine.ts`](src/server/game/GameEngine.ts).
- Python rules are **not** invoked.
- No parity metrics are emitted.

### 1.2 `RINGRIFT_RULES_MODE=shadow`

- TS remains authoritative:
  - All live state transitions go through `GameEngine` as in `ts` mode.
- For each applied move, [`RulesBackendFacade.runPythonShadow`](src/server/game/RulesBackendFacade.ts) is invoked:
  - Calls `PythonRulesClient.evaluateMove(tsBefore, move)` to hit `/rules/evaluate_move`.
  - Compares:
    - `valid` verdict (`tsResult.success` vs Python `valid`).
    - State hash (`hashGameState(tsAfter)` vs Python `stateHash`).
    - S-invariant (`computeProgressSnapshot(tsAfter).S` vs Python `sInvariant`).
    - `gameStatus`.
  - Increments Prometheus counters on mismatch:
    - `rules_parity_valid_mismatch_total`
    - `rules_parity_hash_mismatch_total`
    - `rules_parity_S_mismatch_total`
    - `rules_parity_gameStatus_mismatch_total`
  - Emits structured logs via `logRulesMismatch` with kind:
    - `'valid' | 'hash' | 'S' | 'gameStatus' | 'shadow_error'`.

- Shadow failures (Python exceptions, 5xx, timeouts):
  - Do **not** affect TS behaviour.
  - Are logged only as `'shadow_error'`.
  - Do not increment the semantic mismatch counters above.

### 1.3 `RINGRIFT_RULES_MODE=python` (validation-gated, not yet production-authoritative)

- **Current code behaviour (backend implementation):**
  - [`RulesBackendFacade.applyMove`](src/server/game/RulesBackendFacade.ts:45) and
    `applyMoveById` consult Python **first**:
    - Call `PythonRulesClient.evaluateMove(tsBefore, canonicalMove)` against
      `/rules/evaluate_move`.
    - If `py.valid === false`:
      - Return `success: false` with `error` derived from `validation_error`.
      - Do **not** mutate TS `GameEngine` state.
    - If `py.valid === true`:
      - Apply the move via TS `GameEngine` (`makeMove` / `makeMoveById`) to
        advance the authoritative backend `GameState`.
      - Compare TS vs Python using the same parity invariants as shadow mode:
        - `valid`, `state_hash`, `s_invariant`, `gameStatus`.
      - Increment `rules_parity_*` counters and emit `logRulesMismatch` entries
        on mismatch, but **do not** override Python’s verdict for that call.
    - On Python transport/runtime failure:
      - Log `'backend_fallback'` with context.
      - Fall back to TS `GameEngine` for that move.

  - In other words, Python is now the **validation/parity source** in
    `RINGRIFT_RULES_MODE=python`, while TS still mutates live state. TS runs in
    a "reverse shadow" role for metrics.

- **Operational status:**
  - This mode is **not yet enabled in staging or production** and remains
    **outside the strict P0.5 acceptance scope**.
  - P0.5 is considered successful once:
    - Behavioural parity between TS and Python is demonstrated via the
      fixture/trace tests (Phase 0).
    - `RINGRIFT_RULES_MODE=shadow` is stable in staging and production with
      clean parity metrics and logs (Phases 1 and 2).
  - Flipping production to `RINGRIFT_RULES_MODE=python` is a follow-on
    operational decision gated on:
    - Shadow-mode stability over time.
    - Python latency/availability.
    - Observed parity metrics under real load.

---

## 2. Observability and Metrics

### 2.1 Metric registration

- Metrics are defined in [`src/server/utils/rulesParityMetrics.ts`](src/server/utils/rulesParityMetrics.ts):
  - `rules_parity_valid_mismatch_total`
  - `rules_parity_hash_mismatch_total`
  - `rules_parity_S_mismatch_total`
  - `rules_parity_gameStatus_mismatch_total`

- The Node server registers default Prometheus metrics and exposes `/metrics`:
  - In [`src/server/index.ts`](src/server/index.ts):

    ```ts
    import client from 'prom-client';

    const app = express();
    const server = createServer(app);

    client.collectDefaultMetrics();

    app.get('/metrics', async (_req, res) => {
      try {
        res.set('Content-Type', client.register.contentType);
        const metrics = await client.register.metrics();
        res.send(metrics);
      } catch (err) {
        logger.error('Failed to generate /metrics payload', {
          error: err instanceof Error ? err.message : String(err),
        });
        res.status(500).send('metrics_unavailable');
      }
    });
    ```

### 2.2 Logs

- All parity-related logs use `logRulesMismatch` in [`rulesParityMetrics.ts`](src/server/utils/rulesParityMetrics.ts):
  - Kinds:
    - `'valid' | 'hash' | 'S' | 'gameStatus' | 'backend_fallback' | 'shadow_error'`.
  - Payload should include (where possible):
    - `gameId`
    - `moveNumber`
    - `playerNumber`
    - `boardType`
    - `phase`
    - `tsHash` / `pyHash`
    - `tsS` / `pyS`
    - `tsStatus` / `pyStatus`
    - `mode: getRulesMode()`

---

## 3. Rollout Phases and Acceptance Criteria

### Phase 0 – Local and CI (complete / ongoing)

**Goal:** Confidence that the Python rules engine and TS↔Python adapter are wired correctly.

Checklist:

- Python:
  - `/rules/evaluate_move` semantics tested in [`test_rules_evaluate_move.py`](ai-service/tests/test_rules_evaluate_move.py).
  - TS→Python fixture parity validated in [`test_rules_parity_fixtures.py`](ai-service/tests/parity/test_rules_parity_fixtures.py), including:
    - Engine-level parity via `DefaultRulesEngine` + [`GameEngine.apply_move`](ai-service/app/game_engine.py).
    - HTTP-level parity against `/rules/evaluate_move` using FastAPI `TestClient`.
- TS:
  - `PythonRulesClient` and `RulesBackendFacade` are unit-tested:
    - [`tests/unit/PythonRulesClient.test.ts`](tests/unit/PythonRulesClient.test.ts)
    - [`tests/unit/RulesBackendFacade.test.ts`](tests/unit/RulesBackendFacade.test.ts)
    - [`tests/unit/RulesBackendFacade.fixtureParity.test.ts`](tests/unit/RulesBackendFacade.fixtureParity.test.ts)
  - WebSocket integration confirmed to use `RulesBackendFacade`:
    - [`tests/unit/WebSocketServer.rulesBackend.integration.test.ts`](tests/unit/WebSocketServer.rulesBackend.integration.test.ts)
  - Cross-language trace parity harness:
    - Python→TS vectors generated by [`ai-service/tests/parity/generate_test_vectors.py`](ai-service/tests/parity/generate_test_vectors.py)
      and consumed by [`tests/unit/Python_vs_TS.traceParity.test.ts`](tests/unit/Python_vs_TS.traceParity.test.ts).
    - TS→Python rules-parity fixtures generated by [`tests/scripts/generate_rules_parity_fixtures.ts`](tests/scripts/generate_rules_parity_fixtures.ts)
      and loaded by [`test_rules_parity_fixtures.py`](ai-service/tests/parity/test_rules_parity_fixtures.py).

Acceptance criteria:

- All Jest and Python tests pass in CI.
- No parity metrics are incremented in unit-level fixture tests when Python mirrors TS.
- Shadow failures are logged only as `'shadow_error'` and do not break TS flows.

### Phase 1 – Staging with `RINGRIFT_RULES_MODE=shadow`

**Goal:** Exercise Python rules in a realistic environment without affecting gameplay.

Configuration:

- On staging backend:
  - `RINGRIFT_RULES_MODE=shadow`
  - `AI_SERVICE_URL` pointing to the staging AI-service instance.
- Verify `/metrics` is reachable and returns the parity metrics.

Acceptance criteria (staging):

1. **Functional**
   - All existing Jest suites (especially parity and scenario tests) pass when run against staging config.
   - No user-visible regressions in staging games (moves accepted/rejected as expected, game termination still correct).

2. **Parity**
   - Under representative load (e.g., running `FullGameFlow`, rules matrix tests, and some AI simulations):
     - `rules_parity_valid_mismatch_total` == 0.
     - `rules_parity_hash_mismatch_total` == 0.
     - `rules_parity_S_mismatch_total` == 0.
     - `rules_parity_gameStatus_mismatch_total` == 0.
   - Any non-zero increments must be:
     - Investigated.
     - Root-caused (TS bug, Python bug, or spec issue).
     - Fixed or documented as an accepted exception BEFORE moving to production shadow.

3. **Performance**
   - Measure `/rules/evaluate_move` latency via:
     - TS-side timing inside `PythonRulesClient` (optional future work).
     - AI-service logs or profiling.
   - Targets (staging, non-binding but indicative):
     - P50 < 5 ms
     - P95 < 25 ms
     - P99 < 50 ms

### Phase 2 – Production with `RINGRIFT_RULES_MODE=shadow`

**Goal:** Verify parity at production scale.

Configuration:

- On production backend:
  - `RINGRIFT_RULES_MODE=shadow`
  - `AI_SERVICE_URL` pointing to production AI-service.

Recommended rollout:

- Start with a **subset** of traffic if possible (e.g., a feature flag that only enables shadow for some games or scenarios).
- Gradually move to full production once confidence is high.

Acceptance criteria (production shadow):

1. **Parity metrics**
   - Over a rolling window (e.g., 24–72 hours):
     - No unexpected increases in `rules_parity_*_mismatch_total`.
   - Any observed mismatches:
     - Are traced to specific scenarios.
     - Are either:
       - Fixed.
       - Or documented as known/acceptable deviations (e.g., intentional spec change).

2. **Latency**
   - Python rules latency does not materially impact backend performance:
     - Because Python is only used in shadow, absolute latency is less critical, but:
       - P99 of `/rules/evaluate_move` should still be within acceptable bounds for future authoritative use.
   - No noticeable impact on Node latency or WebSocket responsiveness.

3. **Stability**
   - Python service restarts, transient failures, and timeouts:
     - Are observed only as `'shadow_error'` logs.
     - Do not cause TS gameplay failures or client-visible errors.

### Phase 3 – Planning for `RINGRIFT_RULES_MODE=python` (authoritative)

**Goal:** Define how Python becomes authoritative for rules, with TS in shadow as regression net.

This phase is **not yet implemented** in code and is **beyond the strict P0.5 acceptance scope**. P0.5 is considered successful once:

- Behavioural parity between TS and Python is demonstrated via the fixture/trace tests above.
- `RINGRIFT_RULES_MODE=shadow` is stable in staging and production with clean parity metrics and logs.

Authoritative `python` mode is a follow-on rollout decision once those conditions are met and operational readiness is confirmed.

#### 3.1 Authoritative flow (desired)

In [`RulesBackendFacade.applyMove`](src/server/game/RulesBackendFacade.ts) and `applyMoveById`:

- When `getRulesMode() === 'python'`:
  1. Compute Python result first:

     ```ts
     const py = await pythonClient.evaluateMove(tsBefore, canonicalMove);
     ```

  2. If `py.valid === false`:
     - Treat move as invalid and return `success: false` with `error` from `validationError`.
     - Optionally run TS in shadow to compare verdicts and increment parity metrics as before.

  3. If `py.valid === true` and `py.nextState` provided:
     - Use Python state as authoritative for gameplay:
       - Either:
         - Translate Python `GameState` JSON into TS `GameState` structure and inject into `GameEngine` (requires a controlled setter or rehydration path).
         - Or:
           - Treat Python as the sole state source for gameplay and adapt WebSocket/HTTP responses to read from it (larger refactor).
     - Run TS in **reverse shadow**:
       - Re-apply the move via TS `GameEngine` from a TS-internal mirror of the previous state.
       - Use `runPythonShadow`-style logic but invert roles (TS shadow vs Python authoritative) to preserve `rules_parity_*` metrics semantics.

  4. On Python failure (exceptions, 5xx, bad responses):
     - Log `'backend_fallback'` with full context.
     - Fall back to TS `GameEngine` as in `ts` mode for that move.
     - Optionally increment a separate counter (e.g. `rules_parity_backend_fallback_total`) to track how often this occurs.

#### 3.2 Acceptance criteria for flipping to python-authoritative

Before enabling `python` mode beyond testing:

1. **Shadow parity stability**
   - In production shadow (Phase 2), parity metrics remain zero or have only documented exceptions over a significant window (e.g., millions of moves or several weeks of normal gameplay).

2. **Python engine coverage**
   - Python rules tests cover:
     - Movement and capture parity vs TS (unit & fixture-level).
     - Line formation and territory processing semantics.
     - Turn and termination semantics (forced elimination, stalemate).
   - The Python S-invariant remains non-decreasing and matches TS in all tested scenarios.

3. **Operational readiness**
   - AI-service:
     - Is horizontally scalable enough to handle rules load.
     - Has appropriate monitoring (CPU, memory, request latency/error rates).
   - Node backend:
     - Has alerts for:
       - `rules_parity_*_mismatch_total` increases.
       - `backend_fallback` log spikes.
       - `/metrics` unavailability.

4. **Rollout strategy**
   - Start with:
     - A dedicated environment or a small slice of production traffic with `RINGRIFT_RULES_MODE=python`.
   - Keep TS shared engine and all parity test suites active in CI.
   - Maintain a quick rollback plan:
     - Flip back to `RINGRIFT_RULES_MODE=ts` or `shadow` if issues arise.

---

## 4. Recommended Next Engineering Steps (for P0.5)

From a code perspective, the following next steps will further solidify P0.5:

1. **Fixture-level TS↔Python parity tests through HTTP:**
   - Extend AI-service tests to replay TS-generated fixtures (e.g. from [`tests/scenarios`](tests/scenarios/LineAndTerritory.test.ts)) through `/rules/evaluate_move`, comparing hashes and S-invariants against TS expectations.

2. **Optional TS integration tests against a live Python service:**
   - In the Node repo, add an integration test that:
     - Spins up the AI-service (or assumes it’s running).
     - Uses a real `PythonRulesClient` and `RulesBackendFacade` with `RINGRIFT_RULES_MODE=shadow`.
     - Drives a small script of moves and asserts parity metrics remain zero.

3. **Further instrumentation for Python latency:**
   - Time `PythonRulesClient.evaluateMove` and expose:
     - `rules_parity_python_eval_latency_seconds` histogram.
   - This will make it easier to prove Python can handle authoritative load in Phase 3.

With this plan and the existing code/tests, the remaining work for P0.5 is primarily:

- Running in `shadow` mode in staging and production.
- Watching metrics/logs.
- Closing any parity gaps discovered before considering `python` as an authoritative rules mode.
