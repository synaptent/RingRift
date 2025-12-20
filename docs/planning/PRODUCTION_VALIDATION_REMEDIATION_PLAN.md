<!--
PRODUCTION VALIDATION REMEDIATION PLAN
Purpose: produce clean, repeatable SLO signal for baseline, target-scale, and AI-heavy runs.
-->

# Production Validation Remediation Plan

**Status:** Draft (2025-12-19)  
**Owner:** TBD  
**Scope:** Staging or perf stacks only (no production runs)

## 1. Goal

Produce clean, repeatable production-validation signals by rerunning:

- Baseline (20G/60P + WS companion)
- Target-scale (100G/300P + WS companion)
- AI-heavy (75G/300P + WS companion)

All runs must complete with minimal auth or rate-limit noise so SLO gating reflects real capacity.

## 2. Current Risks

- Auth refresh noise previously produced widespread 401s in target-scale runs.
- Rate limiting dominated the error budget, masking true system performance.
- AI-heavy saturation is still unverified at a clean signal level.

## 3. Clean-Signal Preflight (delta from standard load runs)

### 3.1 Required tooling

- `k6`, `node`, and `jq` must be installed.
- `BASE_URL/health` and `AI_SERVICE_URL/health` must return 200.

### 3.2 Seeded user pool and password alignment

- Seed load-test users (`npm run load:seed-users`).
- Align passwords:
  - Seeder uses `LOADTEST_USER_PASSWORD` (default `TestPassword123!`).
  - k6 pool uses `LOADTEST_USER_POOL_PASSWORD` (default `LoadTestK6Pass123`).
  - Set both to the same value.

### 3.3 User pool sizing

- Use a pool size at least equal to peak VUs.
- Recommended: `LOADTEST_USER_POOL_SIZE=400` for 300 VUs.

### 3.4 Optional staging-only rate limit bypass

To remove 429 noise during capacity validation, enable staging bypass for load-test users:

- `RATE_LIMIT_BYPASS_ENABLED=true`
- `RATE_LIMIT_BYPASS_USER_PATTERN='loadtest.*@loadtest\\.local'`

Disable immediately after the run.

### 3.5 Token TTL alignment

If the API does not return `expiresIn`, set:

- `LOADTEST_AUTH_TOKEN_TTL_S` to the real JWT TTL.
- `LOADTEST_AUTH_REFRESH_WINDOW_S` (default 60).

## 4. Ready-to-Run Environment Block

```bash
export STAGING_URL="https://staging.example.com"
export WS_URL="wss://staging.example.com"
export AI_SERVICE_URL="https://ai-staging.example.com"

export LOADTEST_EMAIL="loadtest_user_1@loadtest.local"
export LOADTEST_PASSWORD="TestPassword123!"

export LOADTEST_USER_POOL_SIZE=400
export LOADTEST_USER_POOL_PASSWORD="TestPassword123!"
export LOADTEST_USER_POOL_PREFIX="loadtest_user_"
export LOADTEST_USER_POOL_DOMAIN="loadtest.local"

export RATE_LIMIT_BYPASS_ENABLED=true
```

## 5. Execution Plan

### 5.1 Baseline (20G/60P)

```bash
SEED_LOADTEST_USERS=true tests/load/scripts/run-baseline.sh --staging
npm run slo:verify tests/load/results/BCAP_STAGING_BASELINE_20G_60P_staging_<timestamp>.json -- --env staging
```

### 5.2 Target-scale (100G/300P)

```bash
SEED_LOADTEST_USERS=true tests/load/scripts/run-target-scale.sh --staging
npm run slo:verify tests/load/results/BCAP_SQ8_3P_TARGET_100G_300P_staging_<timestamp>.json -- --env production
npm run slo:verify tests/load/results/websocket_BCAP_SQ8_3P_TARGET_100G_300P_staging_<timestamp>.json -- --env production
```

### 5.3 AI-heavy (75G/300P, 3 AI seats)

```bash
SEED_LOADTEST_USERS=true tests/load/scripts/run-ai-heavy.sh --staging
npm run slo:verify tests/load/results/BCAP_SQ8_4P_AI_HEAVY_75G_300P_staging_<timestamp>.json -- --env staging
npm run slo:verify tests/load/results/websocket_BCAP_SQ8_4P_AI_HEAVY_75G_300P_staging_<timestamp>.json -- --env staging
```

## 6. Artifacts to Capture

- Raw k6 JSON: `tests/load/results/<scenario>_staging_<timestamp>.json`
- Summary JSON: `tests/load/results/<scenario>_staging_<timestamp>_summary.json`
- SLO report: `tests/load/results/<scenario>_staging_<timestamp>_slo_report.json`
- WebSocket companion raw + report (if enabled)

Update the following after each run:

- `docs/testing/BASELINE_CAPACITY.md`
- `docs/production/PRODUCTION_READINESS_CHECKLIST.md` (if gating status changes)

Optional aggregation for a single go/no-go summary:

```bash
npx ts-node scripts/analyze-load-slos.ts
```

## 7. Acceptance Criteria (clean signal)

- Error rate under 1 percent; 401 and 429 near zero in raw outputs.
- `contract_failures_total == 0` and `id_lifecycle_mismatches_total == 0`.
- Concurrency targets met:
  - Baseline: >= 20 games / 60 players
  - Target-scale: >= 100 games / 300 players
  - AI-heavy: >= 75 games / 300 players
- Target-scale SLO verification passes at production thresholds.
- AI-heavy AI latency and fallback meet production targets per BCAP policy.

## 8. Triage If Results Are Noisy

- 401s: set `LOADTEST_AUTH_TOKEN_TTL_S` to actual TTL or ensure `expiresIn` is returned by auth.
- 429s: increase user pool size or enable staging bypass for load-test users.
- AI latency spikes: confirm AI service scaling, queue depth, and model selection; rerun AI-heavy only after remediation.
