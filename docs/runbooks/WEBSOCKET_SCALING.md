# WebSocket Scaling Runbook

> **Doc Status (2025-11-28): Active Runbook**  
> **Role:** Guide for responding to high WebSocket connection counts and scaling concerns, as signalled by `HighWebSocketConnections` alerts.
>
> **SSoT alignment:** This is **derived operational guidance** over:
>
> - Prometheus alert rules in `monitoring/prometheus/alerts.yml` (e.g. `HighWebSocketConnections`, metric `ringrift_websocket_connections`).
> - Prometheus configuration in `monitoring/prometheus/prometheus.yml`.
> - WebSocket server implementation and topology in the backend (e.g. `src/server/websocket/server.ts`, `WebSocketInteractionHandler`, deployment topology).
>
> **Precedence:** Connection limits, deployment topology, and alert thresholds are defined by **code + configuration + infrastructure**. This runbook provides investigation and mitigation steps but is **not** authoritative for limits or semantics.

---

## 1. When This Alert Fires

**Alert:** `HighWebSocketConnections` (see `monitoring/prometheus/alerts.yml`)

- Expression: `ringrift_websocket_connections > 1000` for 5 minutes.
- Severity: `warning`.
- Intent: Signal that the number of concurrent WebSocket connections may be approaching capacity, requiring either **scaling** or investigation for connection leaks.

Note:

- This alert is about **high connection counts**, not total absence (see `WEBSOCKET_ISSUES.md` for `NoWebSocketConnections`).
- The exact threshold and window are defined in the alert rule; treat those as the SSoT.

---

## 2. Triage

> Goal: Decide whether traffic is legitimately high (e.g. peak period) or whether we are seeing **connection leaks**, misbehaving clients, or insufficient capacity.

### 2.1 Confirm current and historical connection levels

In Prometheus:

```promql
# Current connection count
ringrift_websocket_connections

# Connection trends over the last hour
max_over_time(ringrift_websocket_connections[1h])
```

Answer:

- Is this a sudden spike or a gradual growth?
- Has the connection count remained elevated for a long period (indicative of leaks) or is it tied to known peak times/events?

### 2.2 Compare connections to HTTP traffic and game activity

```promql
# WebSocket connections
ringrift_websocket_connections

# HTTP request rate
sum(rate(http_requests_total[5m]))

# Active games
ringrift_games_active
```

Interpretation examples:

- High WebSocket connections **and** high HTTP traffic + active games  
  → Likely legitimate load; scaling may be appropriate.
- High WebSocket connections but **low** HTTP traffic and low `ringrift_games_active`  
  → Suspicious: connections may not be closing correctly (leaks or long-lived idle connections).

### 2.3 Check for related alerts and resource pressure

Look for alerts that may indicate we’re close to capacity or unstable under load:

- **Resources:**
  - `HighMemoryUsage`, `HighMemoryUsageCritical` → `HIGH_MEMORY.md`.
  - `HighEventLoopLag`, `HighEventLoopLagCritical` → `EVENT_LOOP_LAG.md`.
  - `HighActiveHandles` → `RESOURCE_LEAK.md`.
- **Latency / availability:**
  - `HighP99Latency`, `HighP95Latency`, `HighMedianLatency`, `HighGameMoveLatency` → `HIGH_LATENCY.md`.
  - `ServiceDegraded`, `ServiceMinimalMode`, `ServiceOffline` → `SERVICE_DEGRADATION.md`, `SERVICE_OFFLINE.md`.

If these are firing alongside `HighWebSocketConnections`, treat high connection count as a **contributing factor** to resource/latency issues.

### 2.4 Inspect server state and logs

On the host running the stack (docker-compose by default):

```bash
cd /path/to/ringrift

docker compose ps

# Tail app logs for WebSocket-related activity
docker compose logs app --tail=300 | sed -n '1,200p'
```

Look for:

- Evidence of many concurrent WebSocket sessions being established.
- Repeated connection attempts from the same clients/IPs (possible abuse or reconnect storms).
- Slow handling of WebSocket messages or backpressure warnings.
- Errors from `WebSocketServer` / `WebSocketInteractionHandler` suggesting overload.

If a proxy or load balancer sits in front of the app, also inspect its dashboards/logs for:

- Backend connection utilisation.
- Any configured per-node or per-IP WebSocket connection limits.

---

## 3. Remediation (High Level)

### 3.1 If high connections are expected (legitimate traffic)

1. **Confirm capacity vs limits**
   - Check documented connection limits per instance/node (in infra docs, configuration, or platform dashboards).
   - Ensure that the current connection count is not exceeding safe limits for:
     - File descriptors/sockets.
     - Process memory.
     - CPU/utilisation patterns.

2. **Scale out where appropriate**
   - Follow your deployment runbooks to add capacity:
     - Increase the number of app/WebSocket-serving instances.
     - Adjust load balancer configuration to distribute connections across instances.
   - Ensure the metric `ringrift_websocket_connections` reflects the total across instances (as configured in your metrics setup).

3. **Monitor after scaling**
   - Confirm that resource alerts (memory, event loop lag, handles) remain green.
   - Check that P95/P99 latencies stay within acceptable bounds.

### 3.2 If high connections are likely due to leaks or misbehaving clients

1. **Verify connection lifetime and idle behaviour**
   - Use logs/metrics to determine if WebSocket connections are:
     - Remaining open indefinitely with no activity.
     - Not closing even when clients appear to have left the site.

2. **Enforce sensible server-side policies (via code/config, not ad-hoc)**
   - Review WebSocket server code and configuration to ensure it:
     - Implements reasonable idle timeouts.
     - Handles heartbeat/ping-pong to detect dead clients.
     - Cleans up connections on application- or network-level errors.

3. **Investigate client behaviour**
   - Check whether certain client versions or integration points:
     - Open multiple concurrent connections per user.
     - Fail to close connections when navigating away or going offline.
   - Coordinate with client teams to adjust behaviour if necessary.

4. **Handle abuse or anomalous patterns**
   - If a small number of IPs or tokens are responsible for a disproportionate number of connections, coordinate with security/ops:
     - Apply rate limits or connection caps at the edge where appropriate.
     - Consider blocking abusive sources according to existing security policies.

### 3.3 Coordinate with related runbooks

If high WebSocket connections are coincident with:

- Latency or availability incidents → follow `HIGH_LATENCY.md`, `SERVICE_DEGRADATION.md`, `SERVICE_OFFLINE.md`.
- Resource saturation or leaks → follow `HIGH_MEMORY.md`, `EVENT_LOOP_LAG.md`, `RESOURCE_LEAK.md`.

Scaling and leak fixes should be implemented via normal change-management processes (code/infra changes, reviewed and deployed), not edited directly in this runbook.

---

## 4. Validation

Before treating the scaling incident as resolved:

- **Metrics / alerts:**
  - [ ] `ringrift_websocket_connections` is within expected, documented capacity for the current environment and time of day.
  - [ ] `HighWebSocketConnections` alert has cleared in Alertmanager and remains clear over at least one evaluation window.
  - [ ] Related resource/latency alerts (if any) have cleared or stabilised.

- **System behaviour:**
  - [ ] Users can connect and stay connected without frequent disconnects.
  - [ ] Real-time gameplay (moves, timers, lobby updates) is responsive and stable under typical or peak load.
  - [ ] No abnormal error rates or timeouts associated with WebSocket operations.

---

## 5. TODO / Environment-Specific Notes

Fill in for each environment (staging, production, etc.) and keep updated:

- [ ] Document per-instance and per-cluster WebSocket connection limits (from infra/platform docs).
- [ ] Record which components terminate WebSockets (app instances, load balancers, gateways) and any per-component limits or quotas.
- [ ] Provide links to dashboards that show `ringrift_websocket_connections`, resource utilisation, and latency side-by-side.
- [ ] Capture known peak periods (events, tournaments, promotional campaigns) where higher connection counts are expected and acceptable.
- [ ] Note any pre-approved scaling strategies (e.g. autoscaling rules, manual scale-up steps) for handling sustained high connection counts.
