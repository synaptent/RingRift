# WebSocket Issues Runbook

> **Doc Status (2025-11-28): Active Runbook**  
> **Role:** Guide for investigating `NoWebSocketConnections` and related real-time connectivity issues.
>
> **SSoT alignment:** This is **derived operational guidance** over:
>
> - Prometheus alert rules in `monitoring/prometheus/alerts.yml` (e.g. `NoWebSocketConnections`, `HighWebSocketConnections`, metric `ringrift_websocket_connections`).
> - Prometheus configuration in `monitoring/prometheus/prometheus.yml`.
> - WebSocket server implementation in the backend (e.g. `src/server/websocket/server.ts`, `WebSocketInteractionHandler` and related tests).
> - HTTP and degradation behaviour documented in other runbooks (`NO_TRAFFIC.md`, `SERVICE_DEGRADATION.md`, `SERVICE_OFFLINE.md`).
>
> **Precedence:** Metric definitions, alert thresholds, and WebSocket server code are authoritative. This runbook is advisory triage/remediation guidance and **must not** be treated as a source of truth for thresholds or semantics.

---

## 1. When This Alert Fires

**Primary alert:** `NoWebSocketConnections` (see `monitoring/prometheus/alerts.yml`)

- Expression: `ringrift_websocket_connections == 0` for 15 minutes.
- Severity: `warning`.
- Intent: Signal that **no active WebSocket connections** have been observed for a sustained period.

You should treat this as:

- **Potentially benign** in environments where no one is expected to be connected (staging, dev, or off-peak production).
- **Concerning** when:
  - HTTP traffic is non-zero (users navigating the site), but WebSocket connections remain at zero.
  - Other availability or latency alerts suggest users should be playing real-time games.

> Note: `HighWebSocketConnections` is covered by `WEBSOCKET_SCALING.md` and focuses on capacity/scaling rather than total absence of connections.

---

## 2. Triage

> Goal: Distinguish **"no one is here"** from **"users cannot establish WebSocket connections"**, and identify whether the issue is at the app, network, or configuration layer.

### 2.1 Check environment and expected usage

- Identify the environment: `production`, `staging`, `dev`, etc.
- Determine whether any sessions are expected right now:
  - Off-peak hours for production may legitimately have very few or no games.
  - Staging may only have connections during tests or demos.
- If this environment is often quiet, `NoWebSocketConnections` may be informational unless correlated with other alerts.

### 2.2 Compare WebSocket connections to HTTP traffic and game activity

In Prometheus:

```promql
# WebSocket connection count
ringrift_websocket_connections

# HTTP request volume (signal for user traffic)
sum(rate(http_requests_total[5m]))

# Active games (optional context)
ringrift_games_active
```

Interpret the combination:

- `ringrift_websocket_connections == 0`, **low** HTTP traffic, `ringrift_games_active == 0`  
  → Likely normal inactivity (no one is online).
- `ringrift_websocket_connections == 0`, **non-trivial** HTTP traffic  
  → Suspicious: users can reach HTTP endpoints but cannot establish WebSocket connections.  
  → Check `NO_TRAFFIC.md` if HTTP is also zero.
- `ringrift_websocket_connections` flapping between 0 and small values  
  → Intermittent connections/disconnections; investigate server stability or client errors.

### 2.3 Check for correlated alerts

Look at other alert groups in `alerts.yml`:

- **Availability / traffic:**
  - `NoHTTPTraffic` → see `NO_TRAFFIC.md`.
  - `ServiceDegraded`, `ServiceMinimalMode`, `ServiceOffline` → `SERVICE_DEGRADATION.md`, `SERVICE_OFFLINE.md`.
- **Latency / resources:**
  - `HighP99Latency`, `HighP95Latency`, `HighMedianLatency`, `HighGameMoveLatency` → `HIGH_LATENCY.md`.
  - `HighEventLoopLag`, `HighMemoryUsage`, `HighActiveHandles` → see respective resource runbooks.

If other availability alerts are firing, treat `NoWebSocketConnections` as a symptom and follow those runbooks first.

### 2.4 Smoke-test WebSocket connectivity

From a client perspective (for the affected environment):

1. **UI-based test:**
   - Open the frontend in a browser.
   - Log in and navigate to the lobby.
   - Start or join a game.
   - Observe whether the UI reports connection problems (e.g. unable to connect to real-time channel, stuck loading states).

2. **Low-level test (if you have a WebSocket client tool):**
   - Use a WebSocket client (browser devtools, `wscat`, etc.) to connect to the WebSocket endpoint:

     ```bash
     # Example with wscat or similar tool
     wscat -c WS_BASE_URL  # replace WS_BASE_URL with the real ws:// or wss:// endpoint
     ```

   - Confirm whether:
     - The handshake succeeds (HTTP 101 Switching Protocols).
     - The connection remains open for a reasonable period.
     - Messages can be sent/received in simple test flows if available.

### 2.5 Inspect server and proxy logs

On the host running the stack (docker-compose by default):

```bash
cd /path/to/ringrift

docker compose ps

# Tail recent app and WebSocket-related logs
# (WebSocket server usually runs inside the main app container)
docker compose logs app --tail=300 | sed -n '1,200p'
```

Look for:

- Errors during the WebSocket upgrade handshake (HTTP 400/401/403/500 on upgrade requests).
- Authentication or authorization failures linked to WebSocket connections.
- Repeated connection attempts followed by immediate disconnects.
- Messages from `WebSocketServer` / `WebSocketInteractionHandler` indicating failures establishing or maintaining sessions.

If you’re behind a load balancer or reverse proxy (nginx, ingress, etc.), inspect its logs for:

- Dropped or rejected upgrade requests.
- Misrouted traffic or TLS handshake issues.

### 2.6 Validate HTTP and readiness surfaces

Confirm that the application itself is healthy:

```bash
curl -sS APP_BASE/health | jq . || curl -sS APP_BASE/health
curl -sS APP_BASE/ready | jq . || curl -sS APP_BASE/ready
```

- If `/health` or `/ready` show general problems (DB/Redis/AI unhealthy), follow the corresponding availability/runbook paths first.
- If these are healthy but WebSockets still fail, focus on network/proxy configuration and WebSocket-specific code paths.

---

## 3. Remediation (High Level)

### 3.1 If this is expected / benign

- Confirm with product/ops that no active real-time sessions are expected for this environment/time window.
- Optionally:
  - Document the justification in an ops log or incident note.
  - Revisit `ALERTING_THRESHOLDS.md` and `alerts.yml` later to tune notification behaviour if `NoWebSocketConnections` is noisy for low-traffic environments.

### 3.2 If users should be connected but are not

1. **Rule out broader outages**
   - If `NoHTTPTraffic`, `ServiceOffline`, or `ServiceDegraded` are firing, follow `NO_TRAFFIC.md`, `SERVICE_OFFLINE.md`, `SERVICE_DEGRADATION.md` first.

2. **Fix WebSocket upgrade and routing issues**
   - Check that your load balancer / proxy configuration:
     - Allows WebSocket upgrades (supports `Upgrade: websocket`, `Connection: Upgrade`).
     - Correctly forwards headers and does not strip required connection information.
     - Routes WebSocket traffic to the correct backend service/port.
   - Verify TLS / certificate configuration if using `wss://`:
     - Ensure certificates are valid and not causing handshake failures.

3. **Address application-level handshake or auth failures**
   - From logs and tests, identify whether connections are failing due to:
     - Invalid/expired auth tokens used during connection.
     - Misconfigured origins or CORS-like restrictions applied at the WebSocket layer.
   - Fix underlying auth/config issues via normal code/config changes; do not patch around them in this runbook.

4. **Stabilise the WebSocket process**
   - If the app container that serves WebSockets is crash-looping or frequently restarting:
     - Investigate using application logs and relevant resource runbooks (`HIGH_MEMORY.md`, `EVENT_LOOP_LAG.md`, `RESOURCE_LEAK.md`).
     - Once stabilized, verify that `ringrift_websocket_connections` begins to rise again under normal traffic.

5. **Coordinate with other latency/availability work**
   - If WebSocket failures coincide with high latency or degradation, address underlying causes as per `HIGH_LATENCY.md` and the degradation runbooks.

---

## 4. Validation

Before treating the incident as resolved:

- **Metrics / alerts:**
  - [ ] `ringrift_websocket_connections` is non-zero when test clients are connected and behaves as expected under normal use.
  - [ ] `NoWebSocketConnections` has cleared in Alertmanager and remains clear for at least one full evaluation window.
  - [ ] Any correlated alerts (e.g. `NoHTTPTraffic`, `ServiceDegraded`, `HighP99Latency`) have cleared or stabilised at acceptable levels.

- **Functional checks:**
  - [ ] Clients can establish and maintain WebSocket connections from the UI.
  - [ ] Real-time game updates (moves, timers, lobby updates) function as expected in at least one test game.
  - [ ] Existing WebSocket-related tests (e.g. WebSocket resilience/connectivity scenarios) pass in the affected environment, if runnable.

---

## 5. TODO / Environment-Specific Notes

Fill in the following for each environment (staging, production, etc.):

- [ ] List the specific proxies, load balancers, or gateways involved in WebSocket routing (names, locations, relevant configuration docs).
- [ ] Document the canonical WebSocket endpoint URLs (including `ws://`/`wss://` schemes and ports) and how they are exposed externally.
- [ ] Link to dashboards that show `ringrift_websocket_connections` alongside HTTP traffic, latency, and error rates.
- [ ] Capture any environment-specific quirks (e.g. firewalls, IP whitelists, rate limits) that frequently affect WebSocket connectivity.
