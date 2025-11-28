# Rate Limiting Runbook

> **Doc Status (2025-11-28): Active Runbook**  
> **Role:** Guide for investigating and responding to high rate-limiting activity in the RingRift backend.
>
> **SSoT alignment:** This is **derived operational guidance** over:
>
> - **Monitoring SSoT:** Prometheus alert rules in `monitoring/prometheus/alerts.yml` (group `rate-limiting`, alerts `HighRateLimitHits`, `SustainedRateLimiting`) and scrape configuration in `monitoring/prometheus/prometheus.yml`.
> - **Metrics & instrumentation:** The `ringrift_rate_limit_hits_total` counter defined in `src/server/services/MetricsService.ts`, and emitted by rate-limiting middleware in `src/server/middleware/rateLimiter.ts`.
> - **Rate limiting implementation & configuration:**
>   - Core rate limiting logic and middleware in `src/server/middleware/rateLimiter.ts`.
>   - Environment-driven configuration for per-endpoint and per-user quotas (`RATE_LIMIT_*` env vars), as documented in `docs/ENVIRONMENT_VARIABLES.md`.
>   - Redis-backed vs in-memory limiters, and the fallback limiter semantics.
> - **Monitoring & operations docs:** `monitoring/README.md`, `docs/ALERTING_THRESHOLDS.md`, `docs/incidents/RESOURCES.md`.
>
> **Precedence:**
>
> - The **backend implementation and configuration** (`rateLimiter.ts`, `MetricsService.ts`, `ENVIRONMENT_VARIABLES.md`, deployment manifests) are authoritative for rate limit behaviour and thresholds.
> - `monitoring/prometheus/alerts.yml` is authoritative for **alert names, thresholds, and PromQL expressions**.
> - This runbook explains **how to interpret and act on** those alerts. If it conflicts with code or configs, **code + configs + `alerts.yml` win**, and this document should be updated.

---

## 1. When These Alerts Fire

**Alerts (from `monitoring/prometheus/alerts.yml`, `rate-limiting` group):**

- `HighRateLimitHits` (warning)
- `SustainedRateLimiting` (warning)

**Conceptual behaviour (refer to `alerts.yml` for canonical details):**

- `ringrift_rate_limit_hits_total` is incremented whenever a request is rejected by the rate limiter middleware:
  - The metric is labeled by `endpoint` (normalized request path) and `limiter` (which logical limiter fired, e.g. `auth`, `authLogin`, `api`, `gameMoves`, `fallback`, `custom`).
  - The middleware logs a `Rate limit exceeded` / `Adaptive rate limit exceeded` / `User rate limit exceeded` / `Fallback rate limit exceeded` warning and increments the counter.
- `HighRateLimitHits` fires when **one endpoint / limiter** sees a burst of hits above a configured rate over a relatively short window (see `alerts.yml` for exact rate and window). This typically indicates:
  - A misbehaving or abusive client hammering a specific endpoint, or
  - A legitimate high-volume client pattern that is now hitting configured limits (e.g. heavy polling, aggressive retries, or a spike in gameplay actions).
- `SustainedRateLimiting` fires when **rate limiting is happening across endpoints** at a sustained level, using `ringrift_rate_limit_hits_total` aggregated across endpoints and limiters over a longer window. It usually means:
  - Rate limits are globally too tight for the current traffic mix, or
  - There is a broad abuse/scanning pattern.

**User-facing impact:**

- Users hitting the limit receive HTTP `429 Too Many Requests` responses with standard headers:
  - `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`, and `Retry-After` (seconds).
- Depending on which limiter fired:
  - **Auth / authLogin**: Users may be temporarily blocked from logging in or performing authentication actions.
  - **authRegister / authPasswordReset**: Users may be blocked from creating accounts or requesting password resets in bulk.
  - **api / apiAuthenticated / game / gameMoves**: Clients may see throttling on gameplay or general API operations.
  - **websocket**: Clients may be prevented from establishing excessive WebSocket connections.

---

## 2. Quick Triage (First 5–10 Minutes)

> Goal: Confirm which limiter(s) and endpoint(s) are driving the alerts, determine whether this is **abuse vs legitimate growth**, and understand the environment (prod vs staging vs CI).

### 2.1 Confirm which alert(s) are firing and where

In Alertmanager / your monitoring UI:

1. Identify which of the `rate-limiting` alerts are active:
   - `HighRateLimitHits`
   - `SustainedRateLimiting`
2. For each alert, capture:
   - **Environment / cluster** (e.g. staging vs production).
   - **Start time** and **duration**.
   - Any **labels** surfaced by the alert (sometimes includes `endpoint`).

### 2.2 Inspect rate limit hits in Prometheus

Use Prometheus to inspect the `ringrift_rate_limit_hits_total` counter. The exact windows and thresholds are defined in `alerts.yml`; use windows that roughly match those definitions.

Example queries (adjust ranges as needed):

```promql
# Top endpoints / limiters by recent rate-limit hits
sum by (endpoint, limiter)(rate(ringrift_rate_limit_hits_total[5m]))

# Overall rate-limit hit rate across all endpoints
sum(rate(ringrift_rate_limit_hits_total[15m]))
```

Look for:

- **Which endpoints** (`endpoint` label) are being limited most heavily.
- **Which limiter** (`limiter` label) is firing (e.g. `auth`, `authLogin`, `api`, `gameMoves`, `websocket`, `fallback`, `custom`).
- Whether hits are:
  - Concentrated on **auth/login** → likely brute-force or credential-stuffing attempts, or
  - Concentrated on **game/gameMoves/api** → heavy gameplay or a polling client, or
  - Spread across many endpoints → potential excessive retry logic or bot traffic.

### 2.3 Correlate with HTTP metrics, errors, and business activity

Check for correlation with:

- **HTTP error/latency runbooks:**
  - `HighErrorRate` / `ElevatedErrorRate` → see `HIGH_ERROR_RATE.md`.
  - `HighP99Latency` / `HighP95Latency` / `HighMedianLatency` → see `HIGH_LATENCY.md`.
- **Traffic level:**
  - Ensure total request volume (`sum(rate(http_requests_total[5m]))`) is in a plausible range for the environment; a sudden traffic spike plus rate-limiting may indicate a launch, marketing campaign, or attack.
- **Business/game metrics:**
  - `ringrift_games_active`, `ringrift_games_total` to see if overall gameplay volume is spiking.

If HTTP-level alerts are **not** firing and business metrics are stable, heavy rate limiting may be preventing overload effectively. If HTTP errors and latency are also elevated, the limiter may be too lenient or being overwhelmed.

### 2.4 Inspect logs for rate limiting events

The rate limiting middleware logs when limits are exceeded. For a docker-compose deployment, from the repo root:

```bash
# Inspect recent app logs for rate limit warnings
docker compose logs app --tail=500 2>&1 \
  | grep -E 'Rate limit exceeded|Adaptive rate limit exceeded|User rate limit exceeded|Fallback rate limit exceeded' \
  | tail -n 100
```

Look for patterns:

- Repeated entries with the same **IP**, **userId**, **limiter**, and **path**.
- Whether the warnings are dominated by:
  - `limiter: auth` / `authLogin` / `authRegister` / `authPasswordReset` (security-sensitive).
  - `limiter: api` / `apiAuthenticated` / `game` / `gameMoves` (API / gameplay).
  - `limiter: websocket` or `fallback` / `custom`.

Cross-check timestamp ranges against the alert firing times.

### 2.5 Determine environment posture and configuration

Confirm in the relevant environment:

- Which **backing store** is used for rate limiting:
  - Redis-backed limiters via `initializeRateLimiters(redis)` (preferred in production).
  - In-memory limiters via `initializeMemoryRateLimiters()` or the `fallbackRateLimiter` (typically for development/testing or when Redis is unavailable).
- The **effective limits** for the affected limiter(s), using:
  - Code: `getRateLimitConfigs()` / `getRateLimitConfig()` in `src/server/middleware/rateLimiter.ts`.
  - Environment variables: `RATE_LIMIT_*` keys (see `docs/ENVIRONMENT_VARIABLES.md` and `docs/DEPLOYMENT_REQUIREMENTS.md`).

This establishes whether limits are unexpectedly low for the current traffic, or if the behaviour is consistent with configuration.

---

## 3. Deep Diagnosis

> Goal: Classify the pattern as **abuse/misuse**, **client bug**, **legitimate growth**, or **configuration error**, and decide whether to adjust limits, block traffic, or change client behaviour.

### 3.1 Classify traffic: abuse vs legitimate usage

Use a combination of Prometheus, logs, and (if available) upstream load balancer / API gateway metrics.

**Questions to answer:**

1. **Is a small set of IPs or users responsible?**
   - Check logs for repeated `ip`/`userId` combinations.
   - If load balancer or API gateway metrics are available, inspect top sources.
2. **Are the endpoints consistent with normal gameplay or auth flows?**
   - High limits on `/api/game/*` during peaks may be legitimate.
   - Spikes on `/api/auth/login` or `/api/auth/password-reset` from many IPs may suggest credential stuffing.
3. **Is traffic volume abnormal for this environment?**
   - Compare to known baselines or previous days.
   - Use Prometheus: `sum(rate(http_requests_total[5m]))` and `ringrift_games_active`.
4. **Is there evidence of a client bug?**
   - Tight polling loops, unbounded retries, or missing `Retry-After` handling can all drive excessive calls.

Classify into one (or more) of:

- **Abuse / attack** (e.g. brute-force logins, scanning).
- **Misconfigured client** (e.g. polling every 100ms).
- **Legitimate usage growth** (e.g. more players, new integration).
- **Configuration too strict** (limits too low for realistic behaviour).

### 3.2 Understand which limiter(s) are binding

Using Prometheus and logs, identify which rate limiter keys are binding:

- **Auth / authLogin / authRegister / authPasswordReset**
  - Governed by `RATE_LIMIT_AUTH_*`, `RATE_LIMIT_AUTH_LOGIN_*`, `RATE_LIMIT_AUTH_REGISTER_*`, `RATE_LIMIT_AUTH_PWD_RESET_*`.
  - Security-sensitive; err on the side of **blocking suspicious patterns** and keeping these strict.
- **api / apiAuthenticated / game / gameMoves**
  - Governed by `RATE_LIMIT_API_*`, `RATE_LIMIT_API_AUTH_*`, `RATE_LIMIT_GAME_*`, `RATE_LIMIT_GAME_MOVES_*`.
  - Balance gameplay smoothness against server capacity.
- **websocket**
  - `RATE_LIMIT_WS_*`; controls per-IP connection churn.
- **fallback / custom**
  - `fallbackRateLimiter` (sliding window) or ad-hoc `customRateLimiter()` usage.

For each active limiter, confirm:

- The **current env var values** and whether they match expectations for this environment.
- Whether there have been recent changes to these values via deployment config or infrastructure changes.

### 3.3 Inspect client behaviour and retry logic

If rate limiting is affecting legitimate usage, investigate client behaviour:

- Check frontend / SDK code and documentation for:
  - Handling of `429` responses and `Retry-After` headers.
  - Backoff strategies for polling endpoints or high-frequency actions.
- For partner integrations or bots, confirm their documented request budget and compliance with API usage guidelines.

Where possible, coordinate with the client owner to:

- Reduce request frequency (poll less often, batch operations).
- Respect `Retry-After` to avoid hammering the service.

### 3.4 Consider Redis vs in-memory semantics

If the environment is unexpectedly using in-memory limiters or the `fallbackRateLimiter`:

- Check Redis health and `RedisDown` / `RedisResponseTimeSlow` alerts and their runbooks (`REDIS_DOWN.md`, `REDIS_PERFORMANCE.md`).
- In-memory limiters reset on process restart and don’t share state across instances; in multi-node environments this can:
  - Reduce the effectiveness of limits.
  - Lead to uneven per-node behaviour.

If Redis issues are present, treat them **first** as a dependency incident; rate limiting alerts may be a side effect.

---

## 4. Remediation

> Goal: Protect the system and users while avoiding unnecessary throttling of legitimate traffic. Prefer **architectural and configuration fixes** over ad-hoc hacks.

### 4.1 If this is clearly abuse / attack traffic

1. **Maintain or tighten limits on sensitive endpoints**
   - Do **not** relax `auth`/`authLogin`/`authPasswordReset` limits solely to clear alerts.
2. **Block abusive sources where possible**
   - Use infrastructure controls (WAF, load balancer, firewall rules) to:
     - Block known-bad IPs or IP ranges.
     - Rate-limit or CAPTCHA-challenge suspicious traffic upstream.
3. **Ensure observability for ongoing detection**
   - Confirm that logs capture sufficient fields (IP, userId, path, limiter, user agent) for future analysis.

### 4.2 If this is a client bug or misconfiguration

1. **Coordinate with the client owner**
   - Share examples from logs (path, approximate rate, timestamps).
   - Recommend concrete changes: reduced polling, better caching, proper handling of `Retry-After`.
2. **Consider temporary targeted exceptions (only if safe)**
   - For **known**, well-behaved partners, you may temporarily:
     - Add a dedicated limiter profile with higher limits (e.g. behind an authenticated key), or
     - Introduce feature flags gating the stricter limiter.
   - Any such exception should:
     - Be clearly documented in deployment configuration and environment docs.
     - Include a follow-up to remove or normalize once the client is fixed.

### 4.3 If this is legitimate growth and limits are too strict

1. **Validate capacity and SLOs**
   - Review `docs/DEPLOYMENT_REQUIREMENTS.md` and relevant performance runbooks (`HIGH_LATENCY.md`, `GAME_PERFORMANCE.md`).
   - Confirm that the underlying infrastructure (CPU, memory, DB/Redis) has capacity to handle higher throughput.
2. **Adjust limits deliberately via configuration**
   - Update the appropriate `RATE_LIMIT_*` environment variables for the affected limiter(s).
   - Keep a clear separation:
     - **Security-sensitive** limits (auth, password reset) should remain conservative.
     - **Gameplay / API** limits may be raised to match real-world usage when capacity allows.
3. **Deploy using normal change management**
   - Treat limit changes like any other config change: test in staging, roll out in production with monitoring.

### 4.4 If configuration or infrastructure is broken

- If Redis is down or misconfigured:
  - Follow `REDIS_DOWN.md` and `REDIS_PERFORMANCE.md` to restore a healthy Redis backing store.
  - Once Redis is stable, ensure `initializeRateLimiters(redis)` is used instead of `initializeMemoryRateLimiters()` or `fallbackRateLimiter` in production.
- If multiple app instances are running with inconsistent env vars:
  - Validate deployment manifests and config maps; ensure `RATE_LIMIT_*` settings are **consistent** across instances.

---

## 5. Validation

Before considering a rate-limiting incident resolved, verify:

### 5.1 Metrics and alerts

- [ ] `HighRateLimitHits` and/or `SustainedRateLimiting` have cleared and remained clear for at least one full evaluation window.
- [ ] `ringrift_rate_limit_hits_total` rates have:
  - Returned to expected baselines for the environment, **or**
  - Settled at a new, documented baseline that is justified by capacity and product expectations.
- [ ] Related alerts (e.g. `HighErrorRate`, `HighP99Latency`, `RedisDown`) are either green or tracked by their own incidents/runbooks.

### 5.2 Behavioural checks

- [ ] Representative clients can complete typical flows without frequent `429` responses under normal use.
- [ ] Known abusive patterns are blocked or throttled upstream where appropriate.
- [ ] For any changes to `RATE_LIMIT_*` env vars, a follow-up check confirms that the service remains within latency and error SLOs.

### 5.3 Documentation and configuration

- [ ] Any **permanent** changes to limits are reflected in:
  - `docs/ENVIRONMENT_VARIABLES.md` and, if applicable, `docs/DEPLOYMENT_REQUIREMENTS.md`.
  - Operations runbooks or playbooks relevant to partners/integrations.
- [ ] If exceptions were introduced for specific clients, they are documented with:
  - Owner, scope, rationale, and expiry/review date.

---

## 6. Related Documentation & Runbooks

- **Monitoring & thresholds:**
  - `monitoring/prometheus/alerts.yml`
  - `monitoring/prometheus/prometheus.yml`
  - `monitoring/README.md`
  - `docs/ALERTING_THRESHOLDS.md`

- **Rate limiting implementation & configuration:**
  - `src/server/middleware/rateLimiter.ts`
  - `src/server/services/MetricsService.ts` (see `ringrift_rate_limit_hits_total`)
  - `src/server/cache/redis.ts`
  - `docs/ENVIRONMENT_VARIABLES.md`

- **Related runbooks:**
  - `HIGH_ERROR_RATE.md` — when rate-limiting coincides with HTTP 5xxs.
  - `HIGH_LATENCY.md` and `GAME_PERFORMANCE.md` — to understand whether throttling is successfully preserving latency.
  - `NO_TRAFFIC.md` — in case over-aggressive rate limiting effectively blocks all traffic.

- **Incidents & resources:**
  - `docs/incidents/RESOURCES.md`
  - `docs/incidents/AVAILABILITY.md`

Use this runbook as a **playbook** for investigation and remediation. Always defer to the implementation, configuration, and `alerts.yml` for the exact semantics and thresholds of rate limiting in each environment.
