# RingRift Dependency Upgrade Plan (Wave-Based)

> **SSoT alignment:** This document is a **derived plan** over the canonical sources of truth for rules, lifecycle, architecture, CI, and security:
>
> - Rules & lifecycle SSoTs: `RULES_CANONICAL_SPEC.md`, `ringrift_complete_rules.md`, `docs/CANONICAL_ENGINE_API.md`, shared TS engine under `src/shared/engine/**`, contracts under `src/shared/engine/contracts/**`, WebSocket types/schemas under `src/shared/types/websocket.ts` and `src/shared/validation/websocketSchemas.ts`.
> - Architecture & topology SSoTs: `ARCHITECTURE_ASSESSMENT.md`, `ARCHITECTURE_REMEDIATION_PLAN.md`, `RULES_ENGINE_ARCHITECTURE.md`, `docs/RULES_ENGINE_SURFACE_AUDIT.md`, `docs/OPERATIONS_DB.md`, and `docs/DEPLOYMENT_REQUIREMENTS.md`.
> - Test layering & parity meta-docs: `tests/TEST_LAYERS.md`, `tests/TEST_SUITE_PARITY_PLAN.md`, `docs/PYTHON_PARITY_REQUIREMENTS.md`, `AI_ARCHITECTURE.md`.
> - CI / supply-chain design: `docs/SUPPLY_CHAIN_AND_CI_SECURITY.md`, `.github/workflows/ci.yml`.
>
> **Precedence:** If this plan ever disagrees with the code, tests, or CI configuration, **code + tests + CI win**. This file must then be updated to match the implemented behaviour.
>
> **Doc Status (2025-11-29): Active (dependency upgrade plan, non-semantics)**
>
> This document describes the **wave-based dependency upgrade strategy** for the Node/TypeScript monolith and the Python AI service. It is intentionally conservative about semantics and defers to the rules and lifecycle SSoTs for "what is correct"; it only defines **how** we update dependencies while keeping semantics and behaviour stable.

---

## 1. Current Baselines & Scope

This section records the current dependency and tooling baselines **before** the aggressive upgrade waves.

### 1.1 Node / TypeScript Stack (Monorepo Root)

Source: `package.json`, `package-lock.json`.

- **Runtime libraries (selected):**
  - React: `react@^19.2.0`, `react-dom@^19.2.0`.
  - React Router DOM: `react-router-dom@^7.9.6`.
  - Express: `express@^5.1.0`.
  - Prisma: `prisma@^6.19.0`, `@prisma/client@^6.19.0`.
  - Vite: `vite@^7.2.4`.
  - Zod: `zod@^3.22.4` (effective version from lockfile currently `3.25.76`).
  - Rate limiter: `rate-limiter-flexible@^8.3.0`.
  - Redis client: `redis@^5.10.0`.
  - Metrics: `prom-client@^15.1.3`.

- **Tooling / dev dependencies (selected):**
  - TypeScript: `typescript@^5.3.3`.
  - Jest: `jest@^29.7.0` with `ts-jest@^29.1.1`.
  - Playwright: `@playwright/test@^1.56.1`.
  - ESLint: `eslint@^9.39.1` with `@typescript-eslint/*@^8.48.0`.
  - Vite React plugin: `@vitejs/plugin-react@^5.1.1`.

- **Type packages:**
  - `@types/node`: **bumped in this task** from `^20.10.5` → `^24.10.1`.
  - `@types/jest`: `^29.5.8` (npm reports 29.5.14 current/wanted vs 30.0.0 latest).
  - `@types/uuid`: `^11.0.0` (latest on npm is `10.0.0`; we **do not downgrade**).

- **Engines (from `package.json`):**
  - Node: `>=18.0.0` (tested primarily with Node 22.x locally and in CI).
  - npm: `>=9.0.0`.

- **Outdated (from `npm outdated`, `logs/npm-outdated.log`):**
  - `@types/node`: now updated to `^24.10.1`.
  - `@types/jest`: 29.5.14 current/wanted, 30.0.0 latest (major types bump; Jest runtime remains 29.x for now).
  - `@types/uuid`: latest reported as 10.0.0 (reverse change; stay on 11.x).
  - `rate-limiter-flexible`: 8.3.0 → 9.0.0 (major runtime bump).
  - `zod`: effective 3.25.76 (via lockfile) → 4.1.13 (major runtime bump).

### 1.2 Python AI Service Stack (`ai-service/`)

Source: `ai-service/requirements.txt`, `ai-service/DEPENDENCY_UPDATES.md`.

- **Web/API & infra (current validated pins, 2025-11-29):**
  - `fastapi==0.122.0`
  - `uvicorn[standard]==0.38.0`
  - `pydantic==2.10.5`
  - `httpx==0.28.1`
  - `redis==7.1.0`
  - `prometheus_client==0.23.1`
  - `aiohttp==3.12.14`
  - `python-dotenv==1.2.1`

- **Core ML/DS:**
  - `numpy==2.2.1`, `scipy==1.15.1`, `scikit-learn==1.6.1`.
  - `pandas==2.2.3`, `matplotlib==3.10.0`, `h5py==3.13.0`.

- **Deep learning:**
  - `torch==2.8.0`, `torchvision==0.21.0`.
  - `tensorboard==2.18.0`, `tensorboardX==2.6.2.2`.

- **Dev & test tooling (Wave 3‑A tooling stack):**
  - `pytest==9.0.1`, `pytest-asyncio==1.3.0`, `pytest-timeout==2.4.0`.
  - `black==25.11.0`, `flake8==7.3.0`.

- **Misc & monitoring:**
  - `gymnasium==1.0.0`, `cma>=3.3.0`, `lz4==4.4.5`, `msgpack==1.1.0`, `tqdm==4.67.1`, `psutil==6.1.1`, `pytz==2025.1`, `python-dateutil==2.9.0`.

- **Python versions:**
  - Local development / tests: Python **3.13.x** in a repo-root `.venv` (see `ai-service/DEPENDENCY_UPDATES.md`).
  - Production Docker image: Python **3.11-slim** (see `ai-service/Dockerfile`).

- **Installer baseline (this task):**
  - Within `ai-service/`, `python -m pip install --upgrade pip` was run and logged to `logs/pip-upgrade.log` **before** any Wave&nbsp;3 changes. At that baseline snapshot, no pinned packages in `requirements.txt` had been changed from the versions listed above; subsequent waves (for example the Wave&nbsp;3‑A tooling stack) may update those pins, with the current state tracked in `ai-service/DEPENDENCY_UPDATES.md` and `ai-service/requirements.txt`.

- **Outdated (from `python -m pip list --outdated`, `logs/pip-outdated.log`):**
  - The snapshot in `logs/pip-outdated.log` was taken at the **Wave 0 / baseline** stage and shows many outdated packages in the global environment, including some that were pinned at that time (`aiohttp`, `gymnasium`, `h5py`, `pandas`, `numpy`, `torch`, `torchvision`, `pytest`, `pytest-asyncio`, `pytest-timeout`, `python-dotenv`, `pydantic`, `psutil`, etc.).
  - At that baseline point, **only `pip` itself had been upgraded** to give a modern installer; subsequent waves (see `ai-service/DEPENDENCY_UPDATES.md` and `ai-service/requirements.txt`) have since updated and re‑validated several of those pins (for example the FastAPI/uvicorn/httpx/Redis stack and the pytest/black/flake8 tooling set).

### 1.3 Test Guardrails (Baseline)

We rely on existing layering and parity plans rather than inventing new categories. See:

- `tests/TEST_LAYERS.md` – how unit, contract/scenario, integration, and E2E tests are structured.
- `tests/TEST_SUITE_PARITY_PLAN.md` – how TS↔Python and backend↔sandbox parity are validated.
- `docs/PYTHON_PARITY_REQUIREMENTS.md` – Python rules/AI parity requirements.
- `AI_ARCHITECTURE.md` – AI, rules, and training architecture plus determinism expectations.

**Pre-upgrade E2E baseline (Chromium):**

From `logs/playwright/baseline.chromium.log` (and `baseline.chromium.tail.log`), running a focused Playwright subset:

- Suites included:
  - `tests/e2e/auth.e2e.spec.ts`
  - `tests/e2e/game-flow.e2e.spec.ts`
  - `tests/e2e/multiplayer.e2e.spec.ts`
  - `tests/e2e/ratings.e2e.spec.ts`
  - `tests/e2e/error-recovery.e2e.spec.ts`
- Observations:
  - Auth flows, ratings, error-recovery, and general lobby/WebSocket behaviour **pass**.
  - **Known red test (pre-upgrade):**
    - File: `tests/e2e/multiplayer.e2e.spec.ts`.
    - Test: `Multiplayer Game E2E › Real-Time WebSocket Updates › Game state syncs between both players after multiple moves`.
    - This test currently fails due to intensive reconnects / Redis unavailability patterns in dev; it is treated as a **known red sentinel**, not a new regression.

This known-red status must be preserved and explicitly called out when interpreting regression signals during dependency upgrades.

### 1.4 Log-Handling Policy (Shared Constraint)

Because Jest, Playwright, npm, and pip can produce high-volume output, we standardise on **log redirection + safe viewing**:

- Long-running or noisy commands **must redirect stdout/stderr** to files under `logs/` or `test-results/`:
  - Example: `npx playwright test --project=chromium > logs/playwright/baseline.chromium.log 2>&1`.
  - Example: `cd ai-service && python -m pytest -q > ../logs/pytest/ai-service.latest.log 2>&1`.
- To inspect logs in this repo (including via assistants/tools), we use `scripts/safe-view.js` to generate capped views:

  ```bash
  node scripts/safe-view.js logs/playwright/baseline.chromium.log \
    logs/playwright/baseline.chromium.view.txt --max-lines=400
  ```

  and/or produce small tail slices (for example `tail -n 200` into `*.tail.log`).

All upgrade waves in this plan assume and preserve this logging pattern.

---

## 2. Wave 0 – Documentation-First Baseline (This Task)

**Goal:** Record the current dependency baselines, test guardrails, and log-handling expectations **before** changing any additional package versions.

**Status:**

- ✅ Node/TS baselines and `npm outdated` snapshot captured (see section 1.1).
- ✅ Python baselines, `pip list --outdated`, and `pip` upgrade captured (see section 1.2 and `ai-service/DEPENDENCY_UPDATES.md`).
- ✅ E2E baseline (Chromium) established with a known-red multiplayer sync test (section 1.3).
- ✅ Logging policy documented (section 1.4).
- ✅ `@types/node` upgraded to `^24.10.1` in `package.json`.
- ✅ `pip` upgraded in the AI-service virtualenv.
- ⏳ All other dependency changes are **deferred** to Waves 1–3 below.

Wave 0 is complete when this plan, `docs/DEPLOYMENT_REQUIREMENTS.md`, `ai-service/DEPENDENCY_UPDATES.md`, and the documentation indexes (`docs/INDEX.md`, `DOCUMENTATION_INDEX.md`) all reference the wave structure and baselines.

---

## 3. Wave 1 – Tooling & Test Infrastructure (Low-Risk, Dev-Only)

**Scope:** Type packages and dev tooling where upgrades are low-risk and do **not** meaningfully change runtime semantics.

### 3.1 Candidate Packages

- **Already updated:**
  - `@types/node`: `^24.10.1` (to align with modern Node APIs while keeping runtime `engines.node >= 18`).

- **Potential updates (Wave 1 candidates):**
  - `@types/jest`:
    - Current: 29.5.x.
    - Latest: 30.x.
    - **Constraint:** Jest runtime remains on 29.7.x in this wave; we must verify that `@types/jest@30` is backwards-compatible enough for our Jest 29 config, or postpone this bump.
  - Other dev-only tools where:
    - The upgrade is a **patch or minor** version (no major semver bumps), and
    - Release notes indicate no breaking changes for TypeScript 5.3 / Jest 29 / ESLint 9.

**Non-goals in Wave 1:**

- No changes to runtime libraries that affect request/response semantics, schema validation, rate limiting, or rules logic.
- No Zod or `rate-limiter-flexible` major bumps (those are Wave 2).

### 3.2 Guardrail Tests for Wave 1

Wave 1 focuses on catching **typing / config regressions** and unexpected Jest/TS/ESLint interactions.

**Minimum guardrail set:**

1. **Jest unit + integration suites (Node/TS):**
   - Run at least the standard CI test command (or equivalent subset):

     ```bash
     npm run test:ci
     ```

     which exercises:
     - Shared engine unit suites (`*.shared.test.ts`).
     - Server routes & middleware (`tests/unit/auth.routes.test.ts`, `tests/unit/rateLimiter.test.ts`, `tests/unit/securityHeaders.test.ts`, `tests/unit/metricsMiddleware.test.ts`, etc.).
     - Client components (`tests/unit/components/**`).
     - Integration suites (`tests/integration/**`).

2. **Rules-focused Jest subset (optional but recommended):**
   - `npm run test:ts-rules-engine` to exercise rules-level suites and ensure TS tooling changes do not degrade engine tests.

3. **Targeted E2E smoke:**
   - At minimum, run the Playwright E2E smoke lane:

     ```bash
     npm run test:e2e:smoke
     ```

     which exercises:
     - Auth flows (`auth.e2e.spec.ts`).
     - Helper + POM wiring (`helpers.smoke.e2e.spec.ts`).
     - Sandbox host → backend game path (`sandbox.e2e.spec.ts`).

   - This validates that tooling changes have not broken the dev server, test environment, or basic routing/WebSocket/sandbox wiring.

4. **Orchestrator invariants & S-invariant regression harness (informational, non-gating in Wave 1):**
   - For upgrades that touch orchestrator-adjacent tooling (e.g. `ts-node`, Jest config, or shared-engine contracts) it is useful to run:

     ```bash
     # Single short backend game via shared orchestrator; fails hard on invariant violations
     npm run soak:orchestrator:smoke

     # Seeded backend regression harness promoted from soak violations
     npm run test:orchestrator:s-invariant
     ```

   - These commands exercise:
     - Backend `GameEngine` + `TurnEngineAdapter` under the shared TS orchestrator.
     - Core invariants: S-invariant monotonicity, `totalRingsEliminated` monotonicity, and basic board-structure checks.
     - Seeded reproductions of any `S_INVARIANT_DECREASED` events discovered by the soak harness (see `tests/unit/OrchestratorSInvariant.regression.test.ts` and `docs/STRICT_INVARIANT_SOAKS.md`).

   - **Status (2025-11-29):** The orchestrator S-invariant bug is still under active repair, so these checks are recorded as **informational** for dependency work rather than hard Wave 1 gates. If a future wave fixes the underlying bug and un-skips the regression tests, these commands can be promoted to required guardrails for orchestrator-affecting upgrades.

**Interpretation:**

- If Wave 1 fails only due to type errors or test harness issues (for example mismatched `@types/jest`), we either:
  - Adjust the config/tests to match the new types where it is obviously safe, or
  - Roll back the type upgrade and document the incompatibility here.
- Any functional change in runtime behaviour detected by these tests should halt Wave 1 and be investigated **as if** it were a Wave 2 regression (even though only tooling changed).

### 3.3 Wave 1 Implementation Status (2025-11-29)

- Applied dev-only upgrades:
  - `@types/node` pinned at `^24.10.1` in `package.json` and synced into `package-lock.json`.
  - `@types/jest` upgraded from `^29.5.8` to `^30.0.0` while keeping Jest runtime at `29.7.0`.
- Harness adjustments:
  - Fixed `tests/unit/SandboxGameHost.test.tsx` to read the mocked sandbox context after render (instead of at module load), resolving a harness-only failure.
  - Confirmed Jest is not responsible for executing Playwright `.e2e.spec.ts` suites; E2E remains the responsibility of the Playwright runner.
- Guardrails run:
  - `npm run test:ci` → **red**, but failures are in existing rules/heuristic/UI semantics tests (for example `tests/unit/components/BoardView.test.tsx`, `tests/unit/GameSession.orchestratorSelection.test.ts`, `tests/unit/heuristicEvaluation.test.ts`, `tests/unit/MovementAggregate.shared.test.ts`, `tests/unit/GameEngine.landingOnOwnMarker.test.ts`, `tests/unit/RuleEngine.placementMultiRing.test.ts`). These were already unstable and are **tracked outside** this dependency plan; no new type or harness regressions were observed.
  - `npm run test:ts-rules-engine` → **green**; core TS rules suites executed successfully under the new type packages.
  - `npx playwright test auth.e2e.spec.ts game-flow.e2e.spec.ts --project=chromium` → **green**; auth and game-flow E2E flows continue to work in a degraded environment where Redis is unavailable. Logs show expected degradation headers and "rate limiter not available" warnings but no new error modes.
- Environment notes:
  - Local runs used Node v23.x, which produced some `EBADENGINE`/`MODULE_TYPELESS_PACKAGE_JSON` style warnings from tooling packages. CI continues to target supported Node LTS versions (18/20/22), so these warnings are treated as local-only.
- Conclusion:
  - Wave 1 successfully upgraded the targeted dev-only type packages and validated them against Jest and rules-engine guardrails.
  - No runtime behavioural regressions attributable to dependency changes were observed; remaining red tests represent pre-existing semantics issues to be addressed separately from the dependency plan.

---

## 4. Wave 2 – Runtime / Framework Dependencies (Behaviour-Changing)

**Scope:** Node/TypeScript runtime dependencies where upgrades can change behaviour and API contracts.

### 4.1 Primary Targets

1. **Zod 3.x → 4.x**
   - Current:
     - `zod@^3.22.4` (effective ~3.25.76 via lockfile).
   - Target:
     - `zod@^4.x`.
   - Impact surface:
     - Shared validation schemas: `src/shared/validation/schemas.ts`.
     - WebSocket schemas: `src/shared/validation/websocketSchemas.ts`.
     - Engine contracts: `src/shared/engine/contracts/schemas.ts` and related helpers.
     - Any server-side request/response validation using Zod.
   - Expected work:
     - Update schemas to Zod 4 API changes (especially transforms, refinements, and error shapes).
     - Adjust tests that assert on error formats, message text, or `safeParse` results.

2. **`rate-limiter-flexible` 8.x → 9.x**
   - Current: `rate-limiter-flexible@^8.3.0`.
   - Target: `^9.x`.
   - Impact surface:
     - Server middleware: `src/server/middleware/rateLimiter.ts`.
     - Redis client configuration: `src/server/cache/redis.ts`.
     - Tests: `tests/unit/rateLimiter.test.ts`, `tests/unit/securityHeaders.test.ts`, `tests/unit/degradationHeaders.test.ts`, `docs/runbooks/RATE_LIMITING.md`.
   - Expected work:
     - Migrate configuration objects (e.g., `points`, `duration`, `keyPrefix`, storage options) to the new API.
     - Ensure Redis-backed and in-memory limiters behave as before under typical and burst loads.

3. **Other runtime libraries (opportunistic):**
   - Only if they are:
     - Clearly outdated with important security/performance fixes, **and**
     - Upgradeable without large refactors.
   - Any such changes must be added to this section with their guards before implementation.

### 4.2 Guardrail Tests for Wave 2 (Node/TS)

Wave 2 introduces real behavioural risk. Guardrails must span:

1. **Validation & contracts:**
   - `tests/unit/validation.schemas.test.ts`.
   - `tests/unit/WebSocketPayloadValidation.test.ts`.
   - `tests/unit/contracts.validation.test.ts`.
   - Contract vector runner:
     - `tests/contracts/contractVectorRunner.test.ts`.
   - Shared engine validation around contracts and serialization:
     - `src/shared/engine/contracts/serialization.ts`, `src/shared/engine/contracts/schemas.ts` (indirect via existing tests).

2. **Security & rate limiting:**
   - `tests/unit/rateLimiter.test.ts`.
   - `tests/unit/securityHeaders.test.ts`.
   - `tests/unit/degradationHeaders.test.ts`.
   - `tests/unit/metricsMiddleware.test.ts`.
   - Runbooks & metrics expectations: `docs/runbooks/RATE_LIMITING.md`, `docs/ALERTING_THRESHOLDS.md` (for operator behaviour; these should be manually reviewed for obvious mismatches after changes).

3. **Rules & engine invariants (regression net):**
   - Representative `.shared` suites:
     - `tests/unit/movement.shared.test.ts`.
     - `tests/unit/captureLogic.shared.test.ts`.
     - `tests/unit/territoryBorders.shared.test.ts`.
     - `tests/unit/territoryProcessing.shared.test.ts`.
     - `tests/unit/victory.shared.test.ts`.
   - Orchestrator & contract vectors:
     - `tests/unit/RefactoredEngine.test.ts`.
     - `tests/unit/MovementAggregate.shared.test.ts`.
     - `tests/fixtures/contract-vectors/v2/**` via `tests/contracts/contractVectorRunner.test.ts`.

4. **E2E subset (Chromium):**

   Run, with logging and safe-view, **at least**:

   ```bash
   npx playwright test \
     tests/e2e/auth.e2e.spec.ts \
     tests/e2e/game-flow.e2e.spec.ts \
     tests/e2e/multiplayer.e2e.spec.ts \
     tests/e2e/ratings.e2e.spec.ts \
     tests/e2e/error-recovery.e2e.spec.ts \
     --project=chromium \
     > logs/playwright/wave2.chromium.log 2>&1

   node scripts/safe-view.js \
     logs/playwright/wave2.chromium.log \
     logs/playwright/wave2.chromium.view.txt --max-lines=600
   ```

   - **Important:** The multiplayer suite includes the **known-red** test described in section 1.3. We must:
     - Confirm that it still fails with the same basic symptomatology (WebSocket/Redis behaviour) after runtime upgrades.
     - Treat any change in its failure mode (for example new error messages, timeouts, or status codes) as a **regression signal** even though the test is red.

5. **Server smoke tests:**
   - `tests/unit/server.health-and-routes.test.ts`.
   - `tests/integration/FullGameFlow.test.ts`.
   - `tests/integration/GameReconnection.test.ts`.
   - `tests/integration/LobbyRealtime.test.ts`.

These suites together give high confidence that schema validation, rate limiting, and game/session flows still behave correctly under the new runtimes.

### 4.3 Rollback & Interpretation

- Any regression in **validation semantics** (for example previously accepted payloads now rejected, or vice versa) must be triaged against the canonical rules and lifecycle docs before being accepted.
- Any **tightening** of validation that is obviously correct (for example rejecting previously-unchecked invalid payloads) can be accepted, but must be:
  - Reflected in tests and error expectations.
  - Documented in `docs/API_REFERENCE.md` / `docs/CANONICAL_ENGINE_API.md` as needed.
- If `rate-limiter-flexible` behaviour diverges (for example quota, windowing, or Redis failure semantics), we should:
  - Prefer to restore existing behaviour for now (configuration tweaks), and
  - Only adopt new semantics intentionally with matching updates to runbooks and alerting docs.

### 4.4 Wave 2 Implementation Status (2025-11-29)

This section records the **concrete outcomes** of the Wave 2 Node/TS runtime upgrades as of
2025‑11‑29. It is derived from the code, tests, and logs listed below; if any of those change,
this summary must be updated.

#### 4.4.1 Dependencies Upgraded

- **Zod**: lockfile now resolved to `zod@^4.1.13`.
- **rate-limiter-flexible**: `^9.0.0` (from `^8.3.0`).

Other runtime libraries described in §4.1 were **not** changed in this wave.

#### 4.4.2 Zod 4 Migration (Validation & Contracts)

**Primary surfaces touched:**

- Shared validation schemas:
  - `src/shared/validation/schemas.ts`
  - `src/shared/validation/websocketSchemas.ts`
- Engine contracts & validators:
  - `src/shared/engine/contracts/validators.ts`
  - `src/shared/engine/contracts/schemas.ts` (indirectly via tests)
  - Contract vector runner: `tests/contracts/contractVectorRunner.test.ts`
- Env/config parsing:
  - `src/server/config/env.ts`
  - `src/server/config/unified.ts`

**Key behavioural notes:**

- **Record schemas:** migrated to Zod 4‑compatible usage, e.g.:

  ```ts
  z.record(z.string(), valueSchema);
  ```

  instead of relying on `z.coerce.string()` for record keys, which is not supported in Zod 4.

- **Error shape:** code that previously inspected `ZodError.errors` now prefers the Zod 4
  `issues` array but remains backward‑compatible:

  ```ts
  const zodError = result.error as unknown as {
    issues?: Array<{ path: (string | number)[]; message: string }>;
    errors?: Array<{ path: (string | number)[]; message: string }>;
  };

  const issues = Array.isArray(zodError.issues)
    ? zodError.issues
    : Array.isArray((zodError as any).errors)
      ? (zodError as any).errors
      : [];
  ```

  This pattern is used both in env parsing and engine contract validation so that
  error-reporting helpers continue to work even if older Zod instances appear in tooling.

- **Env boolean semantics tightened:**

  In `src/server/config/env.ts` / `unified.ts`, several boolean‑ish environment variables now
  resolve to actual booleans instead of stringly‑typed flags. For example:

  ```ts
  ENABLE_METRICS: z
    .string()
    .optional()
    .transform((val) => (val === undefined ? true : val !== 'false' && val !== '0')),

  AI_FALLBACK_ENABLED: z
    .string()
    .optional()
    .transform((val) => (val === undefined ? true : val !== 'false' && val !== '0')),
  ```

  Semantics:
  - Unset → default `true`.
  - `'false'` / `'0'` → `false`.
  - Any other value → `true`.

  Existing env‑config tests already expected this truthiness; Wave 2 simply codifies it via
  Zod 4 transforms so that consumers of unified config see real booleans.

- **Test expectations de‑brittled:** where tests previously asserted Zod 3‑specific message
  text (for example requiring the word `"integer"`), they have been updated to match Zod 4’s
  wording while still asserting the **intent** of the validation (e.g. expecting `"expected int"`).

**Guardrail commands & logs (validation):**

```bash
# Zod‑backed validation suites
npx jest tests/unit/validation.schemas.test.ts \
  tests/unit/WebSocketPayloadValidation.test.ts \
  tests/unit/contracts.validation.test.ts \
  tests/contracts/contractVectorRunner.test.ts \
  --runInBand \
  > logs/jest/wave2.zod-validation.log 2>&1

# Env/config schemas and parseEnv() behaviour
npx jest tests/unit/env.config.test.ts --runInBand \
  > logs/jest/wave2.env-config.log 2>&1
```

- Both logs show all tests **passing** under Zod 4.
- No external API shape changes were introduced; error envelopes and response codes remain
  governed by `src/server/errors/errorCodes.ts` and documented in `docs/API_REFERENCE.md`.

#### 4.4.3 `rate-limiter-flexible` v9 Migration

**Code surface:**

- Middleware: `src/server/middleware/rateLimiter.ts`
- Redis wiring: `src/server/cache/redis.ts`

The existing configuration patterns for both Redis‑backed and in‑memory limiters remain
valid under `rate-limiter-flexible@^9.0.0`. No code changes were required beyond the
package‑version bump in `package.json` / `package-lock.json`.

The following suites act as the primary regression net for rate limiting and degradation
behaviour:

```bash
npx jest tests/unit/rateLimiter.test.ts \
  tests/unit/securityHeaders.test.ts \
  tests/unit/degradationHeaders.test.ts \
  tests/unit/metricsMiddleware.test.ts \
  --runInBand \
  > logs/jest/wave2.rate-limiter-unit.log 2>&1
```

- `logs/jest/wave2.rate-limiter-unit.log` shows **4/4 suites passing (95 tests)**.
- Console output confirms that degraded/no‑Redis semantics, CORS/security headers, and
  metrics integration still match the behaviour described in `docs/runbooks/RATE_LIMITING.md`
  and `docs/ALERTING_THRESHOLDS.md`.

#### 4.4.4 Wave 2 Playwright E2E Subset (Chromium)

Wave 2 re‑ran the pre‑existing Chromium E2E subset from §1.3 after the Zod and
rate‑limiter upgrades:

```bash
npx playwright test \
  tests/e2e/auth.e2e.spec.ts \
  tests/e2e/game-flow.e2e.spec.ts \
  tests/e2e/multiplayer.e2e.spec.ts \
  tests/e2e/ratings.e2e.spec.ts \
  tests/e2e/error-recovery.e2e.spec.ts \
  --project=chromium \
  > logs/playwright/wave2.chromium.log 2>&1

node scripts/safe-view.js \
  logs/playwright/wave2.chromium.log \
  logs/playwright/wave2.chromium.view.txt --max-lines=600
```

**Log artefacts:**

- Raw log: `logs/playwright/wave2.chromium.log` (large; treat as binary for tooling).
- Safe view: `logs/playwright/wave2.chromium.view.txt` (600‑line capped plain‑text summary).

**Per‑spec outcome summary (from the safe‑view file):**

- `tests/e2e/auth.e2e.spec.ts` – **all green** (4/4 tests passed):
  - Registration + login happy path.
  - Invalid credentials.
  - Navigation between login/register pages.

- `tests/e2e/game-flow.e2e.spec.ts` – **all listed tests red**:
  - Creates AI game from lobby and renders board + HUD.
  - Interactive ring‑placement cells.
  - Submits a ring‑placement move and logs it.
  - Resync after full page reload.
  - Navigate back to lobby from game page.
  - Displays correct game phase during ring placement.

  The failure summaries in `wave2.chromium.view.txt` point to assertions inside the spec
  itself (for example specific expectations on rendered HUD/phase text), not to Zod or
  rate‑limiter errors. These tests were already unstable before Wave 2 and remain
  tracked outside this dependency plan.

- `tests/e2e/multiplayer.e2e.spec.ts` – **mostly red**, including the known‑red sentinel:
  - Game creation + join by game ID.
  - Multiple players joining updates player count.
  - Turn‑based play (turn alternation and indicators).
  - Real‑time WebSocket updates (single‑move reflection and multi‑move sync).
  - Game completion, disconnection handling, chat/event log, and spectator scenarios.

  All failing multiplayer tests share a common failure point in the shared `setupPlayer`
  helper:

  ```ts
  // tests/e2e/multiplayer.e2e.spec.ts
  async function setupPlayer(page: Page, user: TestUser): Promise<void> {
    await registerUser(page, user.username, user.email, user.password);
    await expect(page.getByText(user.username)).toBeVisible({ timeout: 10_000 });
  }
  ```

  Representative error (including the **known‑red sentinel**):

  ```text
  Error: expect(locator).toBeVisible() failed

  Locator:  getByText('e2e-user-…')
  Expected: visible
  Received: hidden
  Timeout:  10000ms

    at setupPlayer (.../tests/e2e/multiplayer.e2e.spec.ts:66:49)
    at ...:235:9

  Error Context: test-results/multiplayer.e2e-Multiplaye-38dcc-between-players-after-moves-chromium/error-context.md
  ```

  - Sentinel test (pre‑existing known red):
    - File: `tests/e2e/multiplayer.e2e.spec.ts`.
    - Test: `Multiplayer Game E2E › Real-Time WebSocket Updates › Game state syncs between both players after multiple moves`.
    - Status under Wave 2: still **failing** due to the same `toBeVisible` timeout against the
      newly registered username; no new error categories (Zod errors, 5xx responses, TS
      compile failures) were introduced.

- `tests/e2e/ratings.e2e.spec.ts` – **red**, with many tests timing out or failing on
  leaderboard/profile expectations. Ratings/leaderboard flows remain unstable and are
  being tracked as their own workstream; Wave 2 did not attempt to change these semantics.

- `tests/e2e/error-recovery.e2e.spec.ts` – **mixed**:
  - A small subset (e.g. "handles temporary network disconnection gracefully",
    "handles API 400 bad request gracefully") remain **green**.
  - Many more detailed flows (WebSocket disconnection handling, API 500/404 handling,
    session expiry, rate‑limiting, form validation, page reload behaviour) are still **red**.

  Error locations in the safe‑view log point back into the E2E spec expectations rather
  than to runtime exceptions or schema errors.

**Compile/runtime guardrail:**

- The Wave 2 Playwright run starts the dev server, executes the full subset, and
  writes normal WebSocket and Redis‑availability logs; there are **no TypeScript or
  Zod compile failures** in `wave2.chromium.log`.
- This confirms that the prior Zod‑4‑related TS build issues for E2E tests have been
  resolved.

#### 4.4.5 External API / Engine Docs Impact

- `docs/API_REFERENCE.md` – error envelope, rate‑limit categories, and REST surface did
  **not** change in Wave 2. Zod 4 is now the underlying validation engine for many
  schemas, but observable HTTP request/response formats remain consistent with the
  documented contracts.
- `docs/CANONICAL_ENGINE_API.md` – canonical engine types and orchestrator APIs are
  unchanged. Contract vectors and orchestrator tests continue to pass under Zod 4.

Accordingly, no Wave 2 edits were required in those two documents; they already describe the
externally observable behaviour.

---

## 5. Wave 3 – Python AI/Rules Engine Dependencies

**Scope:** Targeted upgrades of pinned Python packages in `ai-service/requirements.txt`, with a focus on test/tooling stacks and select ML/infra libraries.

### 5.1 Candidate Packages

- **Test/tooling:**
  - `pytest`, `pytest-asyncio`, `pytest-timeout`.
  - `black`, `flake8`.
  - **Status (Wave 3‑A, 2025‑11‑29):** The AI service has already been exercised under an upgraded tooling stack (`pytest==9.0.1`, `pytest-asyncio==1.3.0`, `pytest-timeout==2.4.0`, `black==25.11.0`, `flake8==7.3.0`), with the validated versions pinned in `ai-service/requirements.txt` and the execution details (commands, guardrail subset, log paths, and pass/fail summary) recorded in `ai-service/DEPENDENCY_UPDATES.md`.
- **Core libs:**
  - `aiohttp`, `httpx`, `python-dotenv`, `redis`, `prometheus_client` (to align with the modern server stack described in `ai-service/DEPENDENCY_UPDATES.md`).
- **ML / DL (later waves or sub-waves):**
  - `numpy`, `scipy`, `scikit-learn`.
  - `torch`, `torchvision`.
  - `gymnasium`, `pandas`, `matplotlib`, `h5py`.

The intent is to **avoid a blanket upgrade** and instead:

1. Bring infra/test tooling up to date and stable.
2. Then upgrade ML/DL dependencies one at a time, starting with the least coupled.

### 5.2 Guardrail Tests for Wave 3 (Python)

All tests are under `ai-service/tests/**`. The following groups are required:

1. **Parity & contract-level suites:**
   - `ai-service/tests/contracts/test_contract_vectors.py`.
   - `ai-service/tests/parity/test_rules_parity_fixtures.py`.
   - `ai-service/tests/parity/test_ts_seed_plateau_snapshot_parity.py`.
   - `ai-service/tests/parity/test_line_and_territory_scenario_parity.py`.

2. **Determinism & invariants:**
   - `ai-service/tests/test_engine_determinism.py`.
   - `ai-service/tests/test_no_random_in_rules_core.py`.
   - `ai-service/tests/invariants/test_active_no_moves_*.py`.
   - `ai-service/tests/test_eval_randomness_integration.py`.

3. **Training & dataset pipelines:**
   - `ai-service/tests/integration/test_training_pipeline_e2e.py`.
   - `ai-service/tests/test_generate_territory_dataset_smoke.py`.
   - `ai-service/tests/test_training_lps_alignment.py`.
   - `ai-service/tests/test_hex_training.py`.
   - `ai-service/tests/test_model_versioning.py`.

4. **AI behaviour & evaluation:**
   - `ai-service/tests/test_heuristic_ai.py`.
   - `ai-service/tests/test_heuristic_parity.py`.
   - `ai-service/tests/test_mcts_ai.py`.
   - `ai-service/tests/test_descent_ai.py`.
   - `ai-service/tests/test_parallel_self_play.py`.
   - `ai-service/tests/test_multi_board_evaluation.py`.
   - `ai-service/tests/test_multi_start_evaluation.py`.
   - `ai-service/tests/test_cmaes_optimization.py`.

5. **Env & infra:**
   - `ai-service/tests/test_env_interface.py`.
   - `ai-service/tests/test_streaming_dataloader.py`.
   - `ai-service/tests/test_distributed_training.py`.

**Execution pattern:**

- For each candidate upgrade (or small batch of related packages):

  ```bash
  cd ai-service
  python -m pip install --upgrade <packages...> \
    > ../logs/pip-upgrade.wave3.<label>.log 2>&1

  python -m pytest -q \
    > ../logs/pytest/ai-service.wave3.<label>.log 2>&1

  cd ..
  node scripts/safe-view.js \
    logs/pytest/ai-service.wave3.<label>.log \
    logs/pytest/ai-service.wave3.<label>.view.txt --max-lines=600
  ```

### 5.3 Interpretation & Rollback

- Any change in rules semantics or determinism must be triaged against:
  - `docs/PYTHON_PARITY_REQUIREMENTS.md`.
  - `AI_ARCHITECTURE.md`.
  - TS-side contract tests and parity suites.
- For ML/DL stacks (`torch`, `numpy`, etc.), we **expect** some numerical noise but must:
  - Preserve determinism where the architecture already enforces it.
  - Ensure training and evaluation tests (especially parity and plateau tests) still pass.
- If a particular upgrade causes intractable instability, we:
  - Roll it back in `requirements.txt`.
  - Record the failure mode and decision in `ai-service/DEPENDENCY_UPDATES.md`.

---

## 6. CI & Documentation Integration

This plan is intended to work **with** the existing CI and documentation structure rather than replace it.

- CI jobs summarised in `docs/SUPPLY_CHAIN_AND_CI_SECURITY.md` (lint/typecheck, Jest coverage, TS rules engine focus, Docker build, Node & Python dependency audits, Playwright E2E) provide the **automation backbone** for these waves.
- For each wave, we should:
  - Reference this document from PR descriptions.
  - Attach or link to the relevant log view files (for example `logs/playwright/wave2.chromium.view.txt`, `logs/pytest/ai-service.wave3.<label>.view.txt`).
  - Update `ai-service/DEPENDENCY_UPDATES.md` and, where appropriate, `docs/DEPLOYMENT_REQUIREMENTS.md` to keep environment expectations in sync.

When all three waves are complete and stable, this plan should be updated with:

- The final set of upgraded versions (Node/TS and Python).
- Any intentionally-accepted behaviour changes (with links back to rules/API docs).
- A brief summary of remaining known issues (including any still-red tests and why they are acceptable).
