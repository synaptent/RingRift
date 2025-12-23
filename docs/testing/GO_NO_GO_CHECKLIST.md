# RingRift v1.0 Production Go/No-Go Checklist

> **Doc Status (2025-12-05): Operational checklist (non‑SSoT)**
>
> - Single authoritative **go/no‑go checklist** for deciding whether a given build + environment is ready for a production‑like rollout.
> - This document is intentionally **procedural**, not a Single Source of Truth (SSoT) for rules, SLOs, or project direction.
> - **SSoT alignment:** when any thresholds, SLOs, or goals mentioned here appear to disagree with their canonical definitions, treat the following as authoritative and update this checklist to match:
>   - [`PROJECT_GOALS.md`](../PROJECT_GOALS.md:1) – v1.0 goals, SLOs, and environment & rollout success criteria.
>   - [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md:1) – phased execution plan and performance / scale SLO roadmap (including P‑01 load SLOs and pre‑launch performance gate).
>   - [`../archive/historical/CURRENT_STATE_ASSESSMENT.md`](../archive/historical/CURRENT_STATE_ASSESSMENT.md) – factual implementation and test status (test counts, completed waves, health labels).
>   - [`ALERTING_THRESHOLDS.md`](../operations/ALERTING_THRESHOLDS.md:1) – canonical Prometheus alert rules, thresholds, and how they map to P‑01 SLOs and k6 scenarios.
> - **SSoT tooling:**
>   - `npm run ssot-check` → [`scripts/ssot/ssot-check.ts`](../scripts/ssot/ssot-check.ts:1) must be green for the candidate build; this guards docs/config SSoTs (including the docs banner and link checks this checklist relies on).

---

## How to use this checklist

- Apply this checklist to a **specific candidate build + environment** (for example, staging pre‑prod, or first production rollout).
- For a **GO** decision:
  - All items below that are **not explicitly marked “Recommended”** must be checked.
  - Each checked item must have **concrete evidence** attached (CI runs, load reports, Grafana snapshots, drill notes, tickets).
- If any required gate fails, is unknown, or has stale / missing evidence, the decision is **NO‑GO** until:
  - The issue is fixed and evidence updated, or
  - A deliberate, documented waiver is approved by the relevant owner (Engineering, Ops, Product).

### Release metadata

Fill this section for each checklist run.

- **Candidate commit / build artifact:** `____________________________________`
- **Target environment (staging / pre‑prod / prod):** `__________________`
- **Checklist run date:** `__________________`
- **Operators (Engineering / Ops / Product):** `__________________`

---

## 1. Rules & engine correctness (orchestrator, parity, invariants)

These gates ensure the orchestrator, shared rules engine, and cross‑language parity are green for the candidate build. They operationalise P0 “Architecture Production Hardening” and rules‑parity work described in:

- [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md:184) (§“P0: Architecture Production Hardening”, “P0: Rules Fidelity & Parity”)
- [`../archive/historical/CURRENT_STATE_ASSESSMENT.md`](../archive/historical/CURRENT_STATE_ASSESSMENT.md) (§“P0 – Production Hardening” and “P0 – Engine Parity & Rules Coverage”)

### Required

- [ ] **Orchestrator gating suites and soaks are green for the candidate build**
  - **Automation / commands**
    - CI jobs for orchestrator parity/soaks are green for the candidate commit (see CI workflow in `.github/workflows/ci.yml`).
    - Local spot‑checks as needed:
      - `npm run orchestrator:gating` → [`scripts/run-orchestrator-gating.ts`](../scripts/run-orchestrator-gating.ts:1)  
        (aggregated orchestrator/rules gating for P0 robustness).
      - `npm run soak:orchestrator:smoke` / `npm run soak:orchestrator:short` → [`scripts/run-orchestrator-soak.ts`](../scripts/run-orchestrator-soak.ts:1).
  - **Evidence**
    - Links to the latest green CI runs for orchestrator gating / soaks.
    - Optional local soak JSON summary under `results/` for the candidate image.
  - **References**
    - Orchestrator rollout posture and invariants: [`ORCHESTRATOR_ROLLOUT_PLAN.md`](./ORCHESTRATOR_ROLLOUT_PLAN.md:1), [`INVARIANTS_AND_PARITY_FRAMEWORK.md`](./rules/INVARIANTS_AND_PARITY_FRAMEWORK.md:1).
    - Environment posture expectations: [`../archive/historical/CURRENT_STATE_ASSESSMENT.md`](../archive/historical/CURRENT_STATE_ASSESSMENT.md).

- [ ] **TS↔Python contract and parity tests are green**
  - **Automation / commands**
    - Cross‑language contract tests:
      - `./scripts/run-python-contract-tests.sh` → [`scripts/run-python-contract-tests.sh`](../scripts/run-python-contract-tests.sh:1)  
        (Python contract test runner; must report 0 mismatches).
    - TS↔Python parity metrics:
      - `TS_NODE_PROJECT=tsconfig.server.json npx ts-node ../scripts/run-ts-python-parity-metric.ts` (or corresponding CI job) → [`scripts/run-ts-python-parity-metric.ts`](../scripts/run-ts-python-parity-metric.ts:1).
    - If new contract vectors are added as part of this release:
      - `TS_NODE_PROJECT=tsconfig.server.json npx ts-node ../scripts/generate-orchestrator-contract-vectors.ts` → [`scripts/generate-orchestrator-contract-vectors.ts`](../scripts/generate-orchestrator-contract-vectors.ts:1), followed by re‑running both contract and parity suites.
  - **Evidence**
    - Latest CI or local logs showing **100% pass** on contract vectors and parity metrics for the candidate build.
  - **References**
    - [[`../archive/historical/CURRENT_STATE_ASSESSMENT.md`](../archive/historical/CURRENT_STATE_ASSESSMENT.md)](../CURRENT_STATE_ASS

ESSMENT.md:258) (§“P0 – Engine Parity & Rules Coverage”).

- Contract/vector test docs referenced there.

- [ ] **Golden replay suites (TS shared engine + Python parity) are green**
  - **Automation / commands**
    - TypeScript shared‑engine golden replays:
      - `npm test -- --runTestsByPath tests/golden/goldenReplay.test.ts`
    - Python golden replays (structural invariants over JSON fixtures):
      - `cd ai-service && python -m pytest tests/golden/test_golden_replay.py -v`
    - Python golden replay parity over promoted GameReplayDBs (when present):
      - `cd ai-service && python -m pytest tests/parity/test_golden_replay_parity.py -v`
        - This test automatically skips if there are no `*.db` fixtures under `ai-service/tests/fixtures/golden_games/`.
  - **Evidence**
    - CI or local runs for the candidate build showing:
      - `tests/golden/goldenReplay.test.ts` green.
      - `ai-service/tests/golden/test_golden_replay.py` green.
      - `ai-service/tests/parity/test_golden_replay_parity.py` green, or explicitly skipped due to lack of golden DB fixtures with a documented rationale in the release ticket.
  - **References**
    - Golden replay design and invariants: [`docs/testing/GOLDEN_REPLAYS.md`](./testing/GOLDEN_REPLAYS.md:1).
    - Replay DB schema and golden promotion pipeline: [`ENGINE_TOOLING_PARITY_RESEARCH_PLAN.md`](./ENGINE_TOOLING_PARITY_RESEARCH_PLAN.md:1), [`ai-service/docs/GAME_RECORD_SPEC.md`](../ai-service/docs/GAME_RECORD_SPEC.md:1).

- [ ] **Rules health report shows no red regressions for the candidate build**
  - **Automation / commands**
    - `./scripts/rules-health-report.sh` → [`scripts/rules-health-report.sh`](../scripts/rules-health-report.sh:1).
  - **Evidence**
    - Latest rules health report (markdown / text) attached to the release ticket, with no P0/P1 regressions compared to the previous green baseline.
  - **References**
    - Rules engine design and health surfaces in [`RULES_ENGINE_ARCHITECTURE.md`](../RULES_ENGINE_ARCHITECTURE.md:1) and related rules docs listed in [`CURRENT_RULES_STATE.md`](../CURRENT_RULES_STATE.md:1).

---

## 2. Multiplayer lifecycle (lobby, reconnection, spectators, rematch)

These gates ensure the multiplayer lifecycle (lobby, join/leave, reconnection, spectators, rematch) is covered by automated tests and remains green.

### Required

- [ ] **Multiplayer integration and E2E suites covering lobby & reconnection are green in CI**
  - **Automation / commands**
    - CI jobs running at least:
      - Backend WebSocket + reconnection tests (e.g. `tests/unit/GameSession.reconnectFlow.test.ts`, WebSocket server tests) and
      - E2E reconnection and timeout flows (e.g. `reconnection.simulation.test.ts`, `timeout-and-ratings.e2e.spec.ts`, `decision-phase-timeout.e2e.spec.ts`) as described in [`../archive/historical/CURRENT_STATE_ASSESSMENT.md`](../archive/historical/CURRENT_STATE_ASSESSMENT.md).
    - Local confirmation (optional for pre‑prod, recommended for first production launch):
      - `npm run test:integration`
      - `npm run test:e2e:smoke`
  - **Evidence**
    - Links to green CI runs for the candidate commit that include the multiplayer / reconnection / spectator suites listed in [`../archive/historical/CURRENT_STATE_ASSESSMENT.md`](../archive/historical/CURRENT_STATE_ASSESSMENT.md).
  - **References**
    - Multiplayer lifecycle and E2E coverage discussion in [`../archive/historical/CURRENT_STATE_ASSESSMENT.md`](../archive/historical/CURRENT_STATE_ASSESSMENT.md).
    - v1.0 environment & rollout success criteria in [`PROJECT_GOALS.md`](../PROJECT_GOALS.md:195-230).

---

## 3. AI quality & fallbacks

These gates ensure AI behaviour under normal conditions and under degradation matches the expectations in the AI architecture and P‑01 SLOs, and that fallbacks behave correctly.

### Required

- [ ] **AI integration, fallback, and concurrency tests are green**
  - **Automation / commands**
    - Core TS unit / integration suites for AI boundary & fallback behaviour (as summarised in [`../archive/historical/CURRENT_STATE_ASSESSMENT.md`](../archive/historical/CURRENT_STATE_ASSESSMENT.md)) must be green in CI for the candidate build. At minimum this includes tests under:
      - `tests/unit/AIEngine.fallback.test.ts`
      - `tests/unit/AIServiceClient.concurrency.test.ts`
      - Other AI boundary tests referenced by [`AI_ARCHITECTURE.md`](../AI_ARCHITECTURE.md:1).
    - Optionally run them locally:
      - `npm test -- --runTestsByPath tests/unit/AIEngine.fallback.test.ts tests/unit/AIServiceClient.concurrency.test.ts`
  - **Evidence**
    - CI run links showing these suites green at the candidate commit.
  - **References**
    - AI integration and SLOs: [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md:353) (§“2.3 AI turn SLOs”).
    - AI architecture and fallback design: [`AI_ARCHITECTURE.md`](../AI_ARCHITECTURE.md:1).
    - AI incident runbook: [`docs/incidents/AI_SERVICE.md`](./incidents/AI_SERVICE.md:1).

- [ ] **AI degradation drills and alerts are understood and ready**
  - **Automation / commands**
    - Staging **AI degradation drill** successfully completed in the last N days using:
      - [`docs/runbooks/AI_SERVICE_DEGRADATION_DRILL.md`](./runbooks/AI_SERVICE_DEGRADATION_DRILL.md:1)
      - Validates alerts `AIServiceDown`, `AIFallbackRateHigh`, `AIFallbackRateCritical`, `AIRequestHighLatency`, `AIErrorsIncreasing` defined in [`ALERTING_THRESHOLDS.md`](../operations/ALERTING_THRESHOLDS.md:417) and [`monitoring/prometheus/alerts.yml`](../monitoring/prometheus/alerts.yml:1).
      - Optionally capture structured drill reports via:
        - `./node_modules/.bin/ts-node scripts/run-ai-degradation-drill.ts --env staging --phase baseline`
        - `./node_modules/.bin/ts-node scripts/run-ai-degradation-drill.ts --env staging --phase degraded`
        - `./node_modules/.bin/ts-node scripts/run-ai-degradation-drill.ts --env staging --phase recovery`
        - and attach the resulting `results/ops/ai_degradation.staging.*.json` files as evidence.
  - **Evidence**
    - Drill notes (date, environment, findings) linked from the release ticket or ops log.
  - **References**
    - AI incidents guide: [`docs/incidents/AI_SERVICE.md`](./incidents/AI_SERVICE.md:1).
    - Alert behaviour for AI service: [`ALERTING_THRESHOLDS.md`](../operations/ALERTING_THRESHOLDS.md:417).

---

## 4. Tests & coverage gate (TS + Python)

This gate ties the overall automated test surface and coverage targets to a single decision point.

### Required

- [ ] **All CI‑gated TypeScript and Python test suites are green for the candidate build**
  - **Automation / commands**
    - CI must successfully run the standard TS and Python suites summarised in:
      - [`../archive/historical/CURRENT_STATE_ASSESSMENT.md`](../archive/historical/CURRENT_STATE_ASSESSMENT.md) (§“Test Coverage Status” and “Test Categories”).
    - Local summary check (optional in CI, required for manual pre‑prod validation):
      - `./scripts/test-summary.sh` → [`scripts/test-summary.sh`](../scripts/test-summary.sh:1)  
        (summarises Jest and pytest status).
  - **Evidence**
    - CI run(s) for the candidate commit showing:
      - All required Jest suites green (core, integration, contract, parity, E2E where wired).
      - Python test suite green for `ai-service`.
    - Output from a recent `./scripts/test-summary.sh` run (or CI equivalent) attached to the release ticket.
  - **References**
    - Test status and categories: [`../archive/historical/CURRENT_STATE_ASSESSMENT.md`](../archive/historical/CURRENT_STATE_ASSESSMENT.md).
    - v1.0 test coverage requirements: [`PROJECT_GOALS.md`](../PROJECT_GOALS.md:164-174).

- [ ] **Coverage meets or exceeds v1.0 targets**
  - **Automation / commands**
    - Generate coverage for the candidate build:
      - `npm run test:coverage`
    - Analyse high‑impact coverage gaps:
      - `node scripts/analyze-coverage.js` → [`scripts/analyze-coverage.js`](../scripts/analyze-coverage.js:1).
  - **Evidence**
    - Coverage reports (for example Codecov summary and local `coverage/coverage-summary.json`) showing that:
      - Overall coverage and key contexts meet or exceed the v1.0 targets defined in [`PROJECT_GOALS.md`](../PROJECT_GOALS.md:164-174).  
        (Do **not** change those targets here; this checklist only checks whether they are met.)
  - **References**
    - Coverage status and targets: [`../archive/historical/CURRENT_STATE_ASSESSMENT.md`](../archive/historical/CURRENT_STATE_ASSESSMENT.md), [`PROJECT_GOALS.md`](../PROJECT_GOALS.md:164-174).

---

## 5. Load, SLOs, and capacity (P‑01)

These gates connect the P‑01 k6 load scenarios, SLO documentation, and alerting thresholds into a single pre‑launch decision point, as described in:

- [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md:257) (§“Performance & Scalability (P‑01)”, including HTTP, WebSocket, AI, and availability SLOs, plus §5.1 “Pre‑launch performance gate”).
- [`ALERTING_THRESHOLDS.md`](../operations/ALERTING_THRESHOLDS.md:925) (§“Load Test SLO Mapping”).
- Baseline results: [`LOAD_TEST_BASELINE.md`](./LOAD_TEST_BASELINE.md:1), [`LOAD_TEST_BASELINE_REPORT.md`](./LOAD_TEST_BASELINE_REPORT.md:1).

### Required

- [ ] **All four k6 P‑01 scenarios are green at target scale for the target environment**
  - **Automation / commands** (run against the staging or perf environment that mirrors the target production topology):
    - HTTP + game creation:
      - `npx k6 run tests/load/scenarios/game-creation.js` → [`tests/load/scenarios/game-creation.js`](../tests/load/scenarios/game-creation.js:1)
      - `npx k6 run tests/load/scenarios/concurrent-games.js` → [`tests/load/scenarios/concurrent-games.js`](../tests/load/scenarios/concurrent-games.js:1)
    - WebSocket gameplay:
      - `npx k6 run tests/load/scenarios/player-moves.js` → [`tests/load/scenarios/player-moves.js`](../tests/load/scenarios/player-moves.js:36)
      - `npx k6 run tests/load/scenarios/websocket-stress.js` → [`tests/load/scenarios/websocket-stress.js`](../tests/load/scenarios/websocket-stress.js:20)
    - Use the shared summary helper:
      - [`tests/load/summary.js`](../tests/load/summary.js:1) to aggregate metrics and compare against SLOs.
  - **Evidence**
    - JSON summaries and/or Grafana snapshots for P1–P4 runs at the documented target scale:
      - HTTP, WebSocket, AI, and availability metrics meeting or exceeding SLOs from [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md:296) (§2.1–2.4).
    - Latest baseline report updated in:
      - [`LOAD_TEST_BASELINE_REPORT.md`](./LOAD_TEST_BASELINE_REPORT.md:1) and optionally referenced from release notes.
  - **References**
    - P‑01 scenarios and SLOs: [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md:413), [`ALERTING_THRESHOLDS.md`](../operations/ALERTING_THRESHOLDS.md:925).
    - Baseline ranges and interpretation: [`LOAD_TEST_BASELINE.md`](./LOAD_TEST_BASELINE.md:1), [`LOAD_TEST_BASELINE_REPORT.md`](./LOAD_TEST_BASELINE_REPORT.md:1).

- [ ] **Baseline, target-scale, and AI-heavy k6 runs executed with SLO verification and recorded**
  - **Automation / commands**
    - Baseline (20G/60P, WS companion optional): `cd tests/load && ./scripts/run-baseline.sh --staging` (or `--local` for smoke); WS companion SLOs verified separately if run.
    - Target scale (100G/300P + WS companion): `cd tests/load && ./scripts/run-target-scale.sh --staging --skip-confirm true`.
    - AI-heavy probe (4p/3 AI seats, ~75G/300P): `cd tests/load && ./scripts/run-ai-heavy.sh --staging` (use `SMOKE=1` for a short/local wiring check).
    - Each script auto-runs `tests/load/scripts/verify-slos.js` against the configured `THRESHOLD_ENV`; rerun the verifier manually for WS companion outputs if applicable.
  - **Evidence**
    - Result + summary JSON (and WS companion if run) stored under `tests/load/results/` for this candidate.
    - `docs/testing/BASELINE_CAPACITY.md` updated with date, scenario ID, env, notes, and result paths for the baseline, target-scale, and AI-heavy runs (including SLO pass/fail).
  - **References**
    - Run parameters and recording template: [`docs/testing/BASELINE_CAPACITY.md`](./BASELINE_CAPACITY.md:1).
    - SLO verifier: [`tests/load/scripts/verify-slos.js`](../tests/load/scripts/verify-slos.js:1).

- [ ] **No core SLO violations under P‑01 load**
  - **Automation / commands**
    - During or immediately after the load runs above, confirm that:
      - All relevant Prometheus alerts defined in [`ALERTING_THRESHOLDS.md`](../operations/ALERTING_THRESHOLDS.md:925) (HTTP error/latency, WebSocket stalls, AI SLOs, availability) remain **green**.
  - **Evidence**
    - Grafana dashboards (“Game Performance”, “System Health”, “Rules/Orchestrator”) show:
      - SLO metrics within thresholds and with sufficient headroom (per §2 of [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md:296)).
    - No persistent critical alerts during the steady‑state portion of the P‑01 runs.
  - **References**
    - SLO definitions and interpretation guidance in [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md:296-397) and [`ALERTING_THRESHOLDS.md`](../operations/ALERTING_THRESHOLDS.md:925-1077).

---

## 6. Monitoring, alerting, and deployment config

These gates ensure the monitoring / alerting surface and deployment configuration match the documented expectations before a production‑like rollout.

### Required

- [ ] **Deployment configuration for the target environment validates cleanly**
  - **Automation / commands**
    - From the project root:
      - `npm run validate:deployment` → [`scripts/validate-deployment-config.ts`](../scripts/validate-deployment-config.ts:1)
        - Validates `.env.example`, `docker-compose*.yml`, env schema (`src/server/config/env.ts`), `.env.staging`, and CI workflow alignment.
      - For a **prod-preview style deployment**, optionally run the production preview go/no-go harness:
        - `./node_modules/.bin/ts-node scripts/run-prod-preview-go-no-go.ts --env prod-preview --expectedTopology single`
        - and attach the resulting `results/ops/prod_preview_go_no_go.prod-preview.*.json` report as evidence.
  - **Evidence**
    - Latest `npm run validate:deployment` output attached to the release ticket, with **0 errors** for the target environment configuration.
    - (Optional) Latest `prod_preview_go_no_go` JSON report for the target environment, demonstrating a passing topology/config/auth/game/AI smoke.
  - **References**
    - Deployment / topology expectations: [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md:241), [`docs/DEPLOYMENT_REQUIREMENTS.md`](./DEPLOYMENT_REQUIREMENTS.md:1).

#### Operational helper: production‑preview go/no‑go harness (staging / pre‑prod)

- **Purpose**
  - Provides a single, lightweight **production‑preview go/no‑go smoke** for a prod‑like stack (for example, staging or pre‑prod) without starting containers.
  - Validates that:
    - The app topology matches the expected prod‑preview value (for example `RINGRIFT_APP_TOPOLOGY=single`).
    - Deployment config validation is clean (`npm run validate:deployment`).
    - Core auth flows work end‑to‑end.
    - Lobby / game creation, WebSocket session, basic reconnection, and at least one AI path succeed.
    - AI service readiness is healthy from the backend’s point of view.

- **How to run**

  ```bash
  TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/run-prod-preview-go-no-go.ts --env staging --operator your-handle
  ```

  - Optional flags:
    - `--baseUrl https://staging.example.com` to override the default `BASE_URL` / `APP_BASE` / `http://localhost:3000`.
    - `--expectedTopology single` to override the default expected topology (defaults to `'single'`).
  - Script entrypoint: [`scripts/run-prod-preview-go-no-go.ts`](../scripts/run-prod-preview-go-no-go.ts:1).

- **JSON report**
  - Location (by default):

    ```text
    results/ops/prod_preview_go_no_go.<env>.<timestamp>.json
    ```

  - Key fields:
    - `drillType: "prod_preview_go_no_go"` – identifies this as the production‑preview go/no‑go harness.
    - `environment` / `operator` – environment label and operator handle used when running the harness.
    - `topologySummary`:
      - `appTopology` – effective `RINGRIFT_APP_TOPOLOGY` value seen by the backend.
      - `expectedTopology` – the topology you told the harness to expect.
      - `configOk` – whether [`npm run validate:deployment`](../scripts/validate-deployment-config.ts:1) passed.
    - `checks` – per‑check results:
      - `topology_and_config` – topology matches expectations **and** deployment config validation is clean.
      - `auth_smoke_test` – `scripts/test-auth.sh` HTTP auth flow is working.
      - `game_session_smoke` – lobby / game creation / WebSocket / reconnection / AI smoke via the existing [`scripts/game-session-load-smoke.ts`](../scripts/game-session-load-smoke.ts:1) harness.
      - `ai_service_readiness` – AI readiness based on the same `HealthCheckService` surface used by the AI degradation drill.
    - `overallPass` – `true` only if **all** checks report `status: "pass"`.

- **Relationship to this checklist**
  - For a candidate build + environment where this harness is applicable (staging / pre‑prod with a prod‑like topology):
    - “Prod‑like topology configured” and “deployment config validates cleanly” evidence can be satisfied by:
      - `checks[].name === "topology_and_config"` with `status: "pass"` and a reasonable `topologySummary`.
    - “Auth OK” evidence can be satisfied by:
      - `checks[].name === "auth_smoke_test"` with `status: "pass"`.
    - “Game session OK (lobby, WebSocket, reconnection, AI path)” evidence can be satisfied by:
      - `checks[].name === "game_session_smoke"` with `status: "pass"`.
    - “AI wiring / readiness OK” evidence can be satisfied by:
      - `checks[].name === "ai_service_readiness"` with `status: "pass"`.
  - `overallPass: true` in the prod‑preview go/no‑go report is a **necessary precondition** for a GO decision in that environment, but **not sufficient on its own**:
    - The full checklist (including load, SLOs, drills, and documentation gates) must still be satisfied or explicitly waived.

- [ ] **Monitoring configuration and alerts validate cleanly**
  - **Automation / commands**
    - From the project root:
      - `npm run validate:monitoring` → [`scripts/validate-monitoring-configs.sh`](../scripts/validate-monitoring-configs.sh:1)
        - Validates Prometheus and Alertmanager configs (including `monitoring/prometheus/alerts.yml`).
  - **Evidence**
    - Successful `npm run validate:monitoring` output for the candidate configuration.
    - Confirmed presence of dashboards and alert rules referenced in [`ALERTING_THRESHOLDS.md`](../operations/ALERTING_THRESHOLDS.md:1).
  - **References**
    - Alerts & thresholds: [`ALERTING_THRESHOLDS.md`](../operations/ALERTING_THRESHOLDS.md:1).
    - Prometheus config: [`monitoring/prometheus/alerts.yml`](../monitoring/prometheus/alerts.yml:1), [`monitoring/prometheus/prometheus.yml`](../monitoring/prometheus/prometheus.yml:1).

- [ ] **Monitoring stack is live and wired to real notification channels for the target environment**
  - **Automation / commands**
    - Confirm via:
      - Running a test alert (for example, temporarily lowering a threshold or using an Alertmanager “test” integration) and verifying it reaches the configured Slack / email / PagerDuty channel.
  - **Evidence**
    - Documented notification endpoints for the target environment (Slack channel, email group, PagerDuty service, etc.).
    - A recent “test alert” entry in the alerting channel or ops log.
  - **References**
    - Alert routing and escalation: [`ALERTING_THRESHOLDS.md`](../operations/ALERTING_THRESHOLDS.md:1146).

---

## 7. Security & operations drills (secrets, backups, AI degradation)

These gates ensure the security‑critical and operational drills that underpin production readiness (S‑05 and ops) have been exercised recently in a staging or pre‑prod environment.

### Required

- [ ] **Secrets rotation drill completed successfully in the last N days**
  - **Automation / commands / runbooks**
    - Run the **staging secrets‑rotation drill** as described in:
      - [`docs/SECRETS_MANAGEMENT.md`](./SECRETS_MANAGEMENT.md:126) (§“Secrets Rotation Drill (staging)”).
      - [`docs/runbooks/SECRETS_ROTATION_DRILL.md`](./runbooks/SECRETS_ROTATION_DRILL.md:1).
    - Use `npm run validate:deployment`, `npm run validate:monitoring`, and `npm run ssot-check` as pre/post sanity checks where the runbook recommends.
  - **Evidence**
    - Drill record (date, environment, rotated secrets, issues found, rollback outcome) captured in the ops/security log and linked from this checklist.
  - **References**
    - Secrets SSoT and drills: [`SECRETS_MANAGEMENT.md`](./SECRETS_MANAGEMENT.md:1), [`DATA_LIFECYCLE_AND_PRIVACY.md`](./DATA_LIFECYCLE_AND_PRIVACY.md:113).

- [ ] **Database backup & restore drill completed successfully in the last N days**
  - **Automation / commands / runbooks**
    - Run the **staging database backup & restore drill** as described in:
      - [`docs/runbooks/DATABASE_BACKUP_AND_RESTORE_DRILL.md`](./runbooks/DATABASE_BACKUP_AND_RESTORE_DRILL.md:1).
    - Use `npm run validate:deployment`, `npm run validate:monitoring`, and `npm run ssot-check` around the drill as recommended by that runbook.
  - **Evidence**
    - Drill record including:
      - Backup command used, restore target, Prisma migrate status, and basic app smoke results against the restored DB.
  - **References**
    - Data lifecycle and backup expectations: [`DATA_LIFECYCLE_AND_PRIVACY.md`](./DATA_LIFECYCLE_AND_PRIVACY.md:113), [`OPERATIONS_DB.md`](./OPERATIONS_DB.md:1).

- [ ] **AI degradation drill completed successfully in the last N days**
  - **Automation / commands / runbooks**
    - Run the staging **AI service degradation drill** using:
      - [`docs/runbooks/AI_SERVICE_DEGRADATION_DRILL.md`](./runbooks/AI_SERVICE_DEGRADATION_DRILL.md:1).
    - Validate that:
      - Alerts `AIServiceDown`, `AIFallbackRateHigh`, `AIFallbackRateCritical`, and `ServiceDegraded` fire and clear as expected (see [`ALERTING_THRESHOLDS.md`](../operations/ALERTING_THRESHOLDS.md:417)).
      - Fallback behaviour matches [`docs/incidents/AI_SERVICE.md`](./incidents/AI_SERVICE.md:1) and AI fallbacks behave as in automated tests.
    - Optionally capture structured drill reports via:
      - `./node_modules/.bin/ts-node scripts/run-ai-degradation-drill.ts --env staging --phase baseline`
      - `./node_modules/.bin/ts-node scripts/run-ai-degradation-drill.ts --env staging --phase degraded`
      - `./node_modules/.bin/ts-node scripts/run-ai-degradation-drill.ts --env staging --phase recovery`
      - and attach the resulting `results/ops/ai_degradation.staging.*.json` files as evidence.
  - **Evidence**
    - Drill notes (timeline, metrics screenshots, alert states, user‑visible behaviour) captured in incident / ops logs.
  - **References**
    - AI incidents and fallbacks: [`docs/incidents/AI_SERVICE.md`](./incidents/AI_SERVICE.md:1).

---

## 8. Documentation & SSoTs

These gates ensure that all planning/SSoT documents that define “production‑ready” are up‑to‑date and that SSoT tooling is green for the candidate.

### Required

- [ ] **Goals, roadmap, and current‑state docs are consistent and up‑to‑date for this release**
  - **Checks**
    - [`PROJECT_GOALS.md`](../PROJECT_GOALS.md:1) (goals SSoT) accurately reflects the v1.0 objectives and success criteria being used for this launch.
    - [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md:1) (roadmap & SLOs) is updated where necessary to match any agreed changes in direction or SLOs (without modifying the canonical SLO values in this checklist).
    - [`../archive/historical/CURRENT_STATE_ASSESSMENT.md`](../archive/historical/CURRENT_STATE_ASSESSMENT.md) has been refreshed recently enough that its test counts, health labels, and wave completion statuses match the code that is actually being deployed.
  - **Evidence**
    - Review notes or PR references showing these docs were checked / updated as part of the release preparation.

- [ ] **SSoT checks are green for the candidate build**
  - **Automation / commands**
    - From the project root, on the candidate commit:
      - `npm run ssot-check` → [`scripts/ssot/ssot-check.ts`](../scripts/ssot/ssot-check.ts:1).
  - **Evidence**
    - Latest `npm run ssot-check` output attached to the release ticket, with:
      - `docs-banner-ssot`, `docs-link-ssot`, and related checks passing.
      - Any pre‑existing failures clearly understood and explicitly waived (for example, known future work in rules semantics), with a note that they are not regressions introduced by this release.
  - **Notes**
    - This checklist is **operational** and intentionally **not** a SSoT for rules, SLOs, or roadmap details. If `docs-banner-ssot` is ever extended to cover this file, its banner must continue to mark it as non‑SSoT and subordinate to the documents listed at the top.

---

## 9. Sign‑off

The items above are the **single go/no‑go checklist** for declaring a given build + environment “production‑ready” for v1.0. Once all required gates are green and evidence is attached, the following roles record their decisions.

| Role        | Name | Date | Decision (GO / NO‑GO) | Notes / Links to Evidence |
| ----------- | ---- | ---- | --------------------- | ------------------------- |
| Engineering |      |      |                       |                           |
| Operations  |      |      |                       |                           |
| Product     |      |      |                       |                           |
