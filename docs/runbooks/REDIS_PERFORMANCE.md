# Redis Performance Runbook

> **Doc Status (2025-11-28): Active Runbook**  
> **Role:** Guide for investigating high Redis response times.
>
> **SSoT alignment:** Derived over cache usage patterns, Redis deployment configuration, and rate limiting design.
>
> **Precedence:** Redis configuration and backend code that uses Redis are authoritative; this runbook describes how to respond when alerts fire.

---

## 1. When This Alert Fires

**Alert:** `RedisResponseTimeSlow`  
Triggered when Redis response time exceeds configured thresholds.

---

## 2. Triage

1. **Check Redis resource utilisation**
   - Inspect CPU, memory, and network saturation on Redis instances.
2. **Review key usage patterns**
   - Look for very large keys, hot keys, or blocking operations that could cause latency.

---

## 3. Remediation (High Level)

1. **Tune or refactor cache usage**
   - Reduce payload sizes, avoid expensive operations in the critical path, and ensure key TTLs are appropriate.
2. **Scale Redis or shard if needed**
   - Increase capacity or introduce sharding where supported and appropriate.

---

## 4. Validation

- [ ] Redis latency metrics return to normal baselines.
- [ ] Application latency for cache-dependent operations improves.
- [ ] Redis performance alerts clear.

---

## 5. TODO / Environment-Specific Notes

- Document per-environment Redis topology, including clustering, replication, and failover behaviour.
