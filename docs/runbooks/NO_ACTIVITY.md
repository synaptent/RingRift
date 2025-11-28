# No Activity Runbook

> **Doc Status (2025-11-28): Active Runbook**  
> **Role:** Guide for interpreting and responding to `NoActiveGames` alerts.
>
> **SSoT alignment:** This is **derived operational guidance** over:
>
> - Prometheus alert rules in `monitoring/prometheus/alerts.yml` (e.g. `NoActiveGames`, `ringrift_games_active`).
> - Prometheus configuration in `monitoring/prometheus/prometheus.yml`.
> - Product expectations for game activity and the behaviour of the game session / matchmaking stack.
>
> **Precedence:** Alert thresholds, business expectations, and metric definitions are owned by monitoring configs, product requirements, and backend code. This runbook describes **how to investigate and react** when the alert fires; it is **not** a source of truth for thresholds or semantics.

---

## 1. When This Alert Fires

**Alert:** `NoActiveGames` (see `monitoring/prometheus/alerts.yml`)

- Expression: `ringrift_games_active == 0` for 30 minutes.
- Severity: `info` (product-facing signal).
- Intent: Highlight periods where **no games are active**, which may be normal during off-peak hours but suspicious during expected-traffic windows or after a rollout.

You should treat this alert as:

- **Likely benign** in environments or time windows where low traffic is expected (e.g. staging, local dev, off-peak production).
- **Potentially symptomatic** of a deeper issue if:
  - Other availability alerts are firing (`ServiceDegraded`, `ServiceOffline`, `NoHTTPTraffic`, `NoWebSocketConnections`).
  - Product analytics or external monitoring show users attempting to play but failing.

---

## 2. Triage

> Goal: Decide whether `NoActiveGames` reflects normal behaviour or a real problem with game creation, matchmaking, or connectivity.

1. **Check environment and expected traffic**
   - Identify the environment: `production`, `staging`, `dev`, etc.
   - Cross-check with known usage patterns (internal knowledge or external analytics):
     - Is this during a typical **quiet** period?
     - Was this environment intentionally drained (e.g. for maintenance)?
   - If this is staging or a sandpit environment with occasional use, the alert may purely be informational.

2. **Compare game activity vs traffic and connections**

   In Prometheus:

   ```promql
   # Active games
   ringrift_games_active

   # HTTP request volume (overall signal of traffic)
   sum(rate(http_requests_total[5m]))

   # WebSocket connection count
   ringrift_websocket_connections
   ```

   Interpret the combination:
   - `ringrift_games_active == 0`, **low** HTTP traffic, **low** WebSocket connections  
     → Likely normal inactivity (off-peak or unused environment).
   - `ringrift_games_active == 0`, **non-trivial** HTTP traffic, but **low** WebSocket connections  
     → Users may be hitting login/home but not establishing game sessions (check `NoWebSocketConnections`, `NoHTTPTraffic`, `ServiceDegraded`).
   - `ringrift_games_active == 0`, **non-trivial** HTTP traffic, **normal** WebSocket connections  
     → Suspicious: connections exist but games are not being created; investigate matchmaking / game creation.

3. **Check for correlated alerts**

   Look at other alert groups in `alerts.yml`:
   - **Availability:** `ServiceDegraded`, `ServiceMinimalMode`, `ServiceOffline`, `HighErrorRate`, `NoHTTPTraffic`.
   - **Business/WebSocket:** `NoWebSocketConnections`, `HighWebSocketConnections`.
   - **Resources / Latency:** `HighP99Latency`, `HighMemoryUsage`, `HighEventLoopLag`.

   If any of these are firing, treat `NoActiveGames` as an additional symptom and **pivot** to the relevant runbooks (`SERVICE_DEGRADATION.md`, `SERVICE_OFFLINE.md`, `NO_TRAFFIC.md`, `WEBSOCKET_ISSUES.md`, `HIGH_ERROR_RATE.md`, `HIGH_LATENCY.md`).

4. **Smoke-test game creation and session activity**

   From an operator or test account perspective:
   - Use the **UI**:
     - Log in via the normal frontend.
     - Navigate to the lobby.
     - Start a new game (e.g. vs AI or another account).
     - Play a few moves and confirm the game state updates.

   - Optionally, use a simple synthetic via HTTP/WebSocket tooling if available (or rely on existing Playwright e2e tests).

   While performing the smoke test, watch for:
   - HTTP 4xx/5xx errors from APIs related to lobbies or games.
   - WebSocket connection failures or disconnects.
   - Any `SERVICE_UNAVAILABLE` / degradation headers that would explain lack of active games.

5. **Inspect logs for game/session related errors**

   On the host running the stack (docker-compose by default):

   ```bash
   cd /path/to/ringrift

   docker compose ps

   # Tail recent app logs for game/session activity
   docker compose logs app --tail=300 | sed -n '1,200p'
   ```

   Look for:
   - Errors from game session, lobby, or matchmaking routes.
   - Exceptions thrown during game creation or early moves.
   - Repeated authentication failures that might prevent users from starting games.
   - Messages indicating degraded or offline mode from `ServiceStatusManager` / degradation middleware.

6. **Cross-check incident documentation (if impact is real)**

   If you determine that there _should_ be active games but there are none, treat this as an availability / product-impact incident:
   - Use `docs/incidents/AVAILABILITY.md` for broader guidance on documenting and handling availability incidents.
   - Use `docs/incidents/TRIAGE_GUIDE.md` for generic triage steps and communication patterns.
   - Plan to update incident records and, if the issue was material, a post-mortem using `docs/incidents/POST_MORTEM_TEMPLATE.md`.

---

## 3. Remediation (High Level)

### 3.1 If the alert is expected / benign

- Confirm that the environment/time window reasonably explains the lack of games (e.g. internal-only staging, off-peak hours).
- Optionally:
  - Document the rationale in your incident or ops log.
  - Revisit `docs/ALERTING_THRESHOLDS.md` and `alerts.yml` later with product/ops to tune thresholds or notification channels if `NoActiveGames` is too noisy for this environment.

### 3.2 If users **should** be playing but are not

1. **Check for underlying outages or degradation**
   - If `ServiceDegraded` / `ServiceOffline` / `NoHTTPTraffic` are firing, **follow those runbooks first** (`SERVICE_DEGRADATION.md`, `SERVICE_OFFLINE.md`, `NO_TRAFFIC.md`).
   - If `NoWebSocketConnections` is firing, follow `WEBSOCKET_ISSUES.md`.

2. **Fix issues preventing game creation**

   Once you have ruled out a general outage:
   - Investigate API responses for game creation / lobby routes (via the UI or API client):
     - Look for 4xx/5xx errors or validation failures returned when starting a game.
     - Correlate with `HIGH_ERROR_RATE.md` if 5xx rates are elevated.
   - Confirm database and Redis are healthy since both are involved in session and game persistence:
     - If database issues are suspected, follow `DATABASE_DOWN.md` / `DATABASE_PERFORMANCE.md`.
     - If Redis issues are suspected, follow `REDIS_DOWN.md` / `REDIS_PERFORMANCE.md`.

3. **Review recent changes impacting game creation or matchmaking**
   - Check for recent deployments, feature flags, or configuration changes affecting:
     - Lobby and matchmaking logic.
     - Game session lifecycle management.
   - If `NoActiveGames` began immediately after such a change:
     - Consider rolling back using `DEPLOYMENT_ROLLBACK.md`.
     - Re-test game creation after rollback.

4. **Coordinate with product and support**
   - If this is production and active players are expected:
     - Notify the on-call or product owner that there is effectively a **silent outage of gameplay**, even if the site appears up.
     - Determine if any user communication (status page, social, in-app banners) is needed.

---

## 4. Validation

Before treating the incident as resolved:

- **Metrics:**
  - [ ] `ringrift_games_active` rises above zero as new games start.
  - [ ] The `NoActiveGames` alert clears in Alertmanager and remains clear for at least one full evaluation window.
  - [ ] If other alerts were involved (e.g. `NoWebSocketConnections`, `NoHTTPTraffic`, `HighErrorRate`), they have cleared.

- **Functional checks:**
  - [ ] At least one test user can successfully:
    - Log in.
    - Create a game (vs AI or another test account).
    - Play several moves without unexpected errors or disconnects.
  - [ ] Any existing smoke tests or e2e flows for game start and basic play pass for the affected environment.

- **User impact assessment:**
  - [ ] Product / support stakeholders agree that observed activity levels have returned to normal expectations for the current time window.

---

## 5. TODO / Environment-Specific Notes

Fill these in **per environment** (staging, production, etc.) and keep them current:

- [ ] Define expected baseline ranges for `ringrift_games_active` by time of day / day of week and environment (even high-level, e.g. "usually > 0 during 18:00–23:00 in production").
- [ ] Document which dashboards (Grafana or other) show game activity, traffic, and WebSocket connections together.
- [ ] Note any synthetic monitoring or regular test flows (e.g. scheduled e2e games) that should keep `ringrift_games_active` occasionally non-zero even during quieter periods.
- [ ] Capture known scenarios where `NoActiveGames` is expected (maintenance windows, deployments, tournaments offline, etc.) so on-call can quickly dismiss benign alerts.
