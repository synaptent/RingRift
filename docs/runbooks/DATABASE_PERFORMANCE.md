# Database Performance Runbook

> **Doc Status (2025-11-28): Active Runbook**  
> **Role:** Guide for investigating high database response times.
>
> **SSoT alignment:** Derived over database schema, query patterns, and deployment configuration.
>
> **Precedence:** Database configuration, schema definitions, and migration history are authoritative. This runbook documents response steps.

---

## 1. When This Alert Fires

**Alert:** `DatabaseResponseTimeSlow`  
Triggered when high-percentile database response times exceed configured thresholds.

---

## 2. Triage

1. **Identify slow query patterns**
   - Use database tooling to find the most time-consuming queries during the alert window.
2. **Correlate with load and application changes**
   - Check whether traffic spikes or new features correspond to increased database load.

---

## 3. Remediation (High Level)

1. **Optimise queries and indexes**
   - Add or adjust indexes, or refactor particularly expensive queries where safe.
2. **Scale database resources if necessary**
   - Increase capacity or adjust connection pooling when query patterns are healthy but load is higher than planned.

---

## 4. Validation

- [ ] Database response time returns to normal baselines.
- [ ] Application latency improves for database-backed operations.
- [ ] Database performance alerts clear.

---

## 5. TODO / Environment-Specific Notes

- Document standard tooling for query analysis and any performance budgets for key operations.
