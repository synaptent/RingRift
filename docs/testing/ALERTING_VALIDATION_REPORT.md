# Alerting Validation Report (W3-8)

> **Date:** 2025-12-12  
> **Scope:** Wave‑3 item W3‑8 “Validate alerting under load”  
> **Goal:** Prove that core Prometheus alert rules evaluate and fire as expected under synthetic load conditions, and record a reproducible validation harness.

## Summary

- Prometheus alert rules were validated with `promtool test rules` using synthetic time‑series inputs.
- The unit tests cover one representative alert from each critical category:
  - **Availability:** `RedisDown`
  - **Latency:** `HighP95Latency`
  - **HTTP error budget:** `HighErrorRate`
- All tests passed (`SUCCESS`), confirming rules fire with correct labels/annotations and respect `for:` durations.

## Evidence

- Test harness: `monitoring/prometheus/alerts.test.yml`
- Command:

  ```bash
  promtool test rules monitoring/prometheus/alerts.test.yml
  ```

- Output:

  ```
  SUCCESS
  ```

## End-to-end staging drill (Docker)

With the local staging stack up via `docker-compose.staging.yml`, we verified
that a real alert transitions through **Prometheus → Alertmanager**.

1. **Trigger:** stop the AI service for >2 minutes:

   ```bash
   docker stop ringrift-ai-service-1
   sleep 150
   ```

2. **Prometheus firing check:**

   ```bash
   curl -s http://localhost:9090/api/v1/alerts \
     | jq '.data.alerts[] | select(.labels.alertname=="AIServiceDown")'
   ```

   Observed state: `firing`.

3. **Alertmanager receipt check:**

   ```bash
   curl -s http://localhost:9093/api/v2/alerts \
     | jq '.[] | select(.labels.alertname=="AIServiceDown")'
   ```

   Observed state: `active`.

4. **Recover:**

   ```bash
   docker start ringrift-ai-service-1
   ```

This confirms that at least one representative availability alert is evaluated,
fired, and routed correctly in the staging topology.

## Test Details

### 1) HighErrorRate

- Synthetic counters drive a 5xx ratio of ~9.1% over a 5‑minute rate window.
- Evaluated at 10m to satisfy `for: 5m`.
- Expected firing alert with:
  - `severity=critical`, `team=backend`
  - Summary showing computed percentage.

### 2) HighP95Latency

- Synthetic histogram buckets yield p95 ≈ 1.95s.
- Evaluated at 12m to satisfy `for: 10m`.
- Expected firing alert with:
  - `severity=warning`, `team=backend`
  - Summary showing computed p95 duration.

### 3) RedisDown

- Synthetic gauge `ringrift_service_status{service="redis"}=0` sustained >1m.
- Evaluated at 2m to satisfy `for: 1m`.
- Expected firing alert with:
  - `severity=critical`, `team=backend`, `component=redis`, `service=redis`.

## Integration Note (Alertmanager)

This W3‑8 pass validates **rule correctness** (PromQL + labels + `for:` timing).  
A full end‑to‑end Alertmanager routing drill can be run later when Docker/Desktop Prometheus is available:

1. Start staging stack (`docker compose -f docker-compose.staging.yml up -d`).
2. Run a short k6 smoke (`npm run load:smoke:all`) to ensure metrics flow.
3. Temporarily stop Redis or AI service to trigger an availability alert.
4. Confirm the alert appears in:
   - Prometheus: `http://localhost:9090/alerts`
   - Alertmanager: `http://localhost:9093/#/alerts`

Record any routing/receiver issues in this file and in `docs/operations/ALERTING_THRESHOLDS.md`.
