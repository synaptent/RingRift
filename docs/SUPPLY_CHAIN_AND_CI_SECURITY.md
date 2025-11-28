# RingRift Supply Chain & CI/CD Security (S-05.F)

> **SSoT alignment:** This document is a derived design/plan over the following canonical operational and semantics sources:
>
> - **Operational SSoT:** CI workflows under `.github/workflows/*.yml` (especially `ci.yml`), Dockerfiles (`Dockerfile`, `ai-service/Dockerfile`), docker-compose stacks (`docker-compose*.yml`), monitoring and alerting configs under `monitoring/**`, and runtime env/config validation in `src/server/config/**`, `src/shared/utils/envFlags.ts`, and `scripts/validate-deployment-config.ts`.
> - **Rules semantics & lifecycle SSoTs:** Shared TS rules engine under `src/shared/engine/**` plus v2 contract vectors, and lifecycle/API contracts described in `docs/CANONICAL_ENGINE_API.md` and implemented in `src/shared/types/**`, `src/shared/engine/orchestration/types.ts`, and `src/shared/validation/websocketSchemas.ts`. This doc is **downstream** of those semantics; it does not define rules or Move semantics.
> - **Precedence:** If this document ever conflicts with CI configs, Dockerfiles, monitoring/alerting configs, or validated env/config code, **code + tests + live configs win**, and this document must be updated to match them.
>
> **Doc Status (2025-11-27): Active (with historical/aspirational content)** \
> Design and plan for supply-chain and CI/CD hardening (S-05.F.x). Describes intended safeguards and tracks; some controls are already implemented in `ci.yml` and Dockerfiles, others remain backlog. This is not a rules or lifecycle SSoT; it complements `SECURITY_THREAT_MODEL.md`, `DATA_LIFECYCLE_AND_PRIVACY.md`, and `DOCUMENTATION_INDEX.md` for overall security posture, and it is downstream of the rules semantics SSoT (shared TS engine + contracts + vectors) and lifecycle/API SSoT (`CANONICAL_ENGINE_API.md` + shared types/schemas).

**Related docs:** [`docs/SECURITY_THREAT_MODEL.md`](./SECURITY_THREAT_MODEL.md:1), [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md:155), [`docs/DATA_LIFECYCLE_AND_PRIVACY.md`](./DATA_LIFECYCLE_AND_PRIVACY.md:1), [`docs/OPERATIONS_DB.md`](./OPERATIONS_DB.md:1), [`archive/FINAL_ARCHITECT_REPORT.md`](../archive/FINAL_ARCHITE_REPORT.md:1), [`tests/README.md`](../tests/README.md:1), [`tests/TEST_LAYERS.md`](../tests/TEST_LAYERS.md:1), [`tests/TEST_SUITE_PARITY_PLAN.md`](../tests/TEST_SUITE_PARITY_PLAN.md:1)

This document defines a focused plan for hardening RingRift’s supply chain and CI/CD pipeline. It elaborates on the S-05.F backlog item from [`docs/SECURITY_THREAT_MODEL.md`](./SECURITY_THREAT_MODEL.md:188) and is scoped to:

- Dependencies for the Node/TypeScript stack (see [`package.json`](../package.json:1) and [`package-lock.json`](../package-lock.json:1)) and the Python AI service (see [`ai-service/requirements.txt`](../ai-service/requirements.txt:1)).
- Build artefacts produced by the main [`Dockerfile`](../Dockerfile:1), the AI-service [`ai-service/Dockerfile`](../ai-service/Dockerfile:1), and local stacks defined in [`docker-compose.yml`](../docker-compose.yml:1) and [`docker-compose.staging.yml`](../docker-compose.staging.yml:1).
- CI workflows under [`.github/workflows`](../.github/workflows/ci.yml:1) that build, test, and scan the stack.
- Secret handling in local `.env` files (for example [`.env.example`](../.env.example:1), [`.env.staging`](../.env.staging:1)) and in CI (for example `GITHUB_TOKEN`, `SNYK_TOKEN`).

The goal is to define practical, low-friction safeguards that fit the existing architecture and can be implemented incrementally without introducing vendor lock-in or major structural changes.

## 1. Supply Chain & CI/CD Threat Model (S-05.F)

This section describes the assets and threat scenarios specific to RingRift’s build and delivery pipeline. It complements Section 2.6 of [`docs/SECURITY_THREAT_MODEL.md`](./SECURITY_THREAT_MODEL.md:188).

### 1.1 Assets in the build & delivery pipeline

| Asset                     | Description                                                                                                                                                                                                                                                                                                                                      | Notes / examples                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| Source code               | TypeScript/Node backend and frontend under `src/`, shared rules engine under [`src/shared/engine`](../src/shared/engine/core.ts:1), Python AI service under [`ai-service/`](../ai-service/README.md:1), Prisma schema and migrations under [`prisma/`](../prisma/schema.prisma:1), Jest/Playwright tests under [`tests/`](../tests/README.md:1). | Primary trust anchor; compromised code here propagates into all artefacts.      |
| Dependency graphs         | Node packages defined in [`package.json`](../package.json:1) and locked in [`package-lock.json`](../package-lock.json:1); Python packages pinned in [`ai-service/requirements.txt`](../ai-service/requirements.txt:1).                                                                                                                           | Third-party risk surface (npm, PyPI).                                           |
| Build artefacts           | Backend + frontend bundles under `dist/` built via `npm run build` and the multi-stage [`Dockerfile`](../Dockerfile:1); Python AI container built from [`ai-service/Dockerfile`](../ai-service/Dockerfile:1).                                                                                                                                    | Artefacts deployed to staging/production.                                       |
| CI workflows              | GitHub Actions workflow [`ci.yml`](../.github/workflows/ci.yml:1) running lint, tests, coverage, dependency audits, security scanning, and Docker build checks.                                                                                                                                                                                  | Automates build, test, and security checks for changes to `main` and `develop`. |
| Runtime images & services | Container images referenced in [`docker-compose.yml`](../docker-compose.yml:1) and [`docker-compose.staging.yml`](../docker-compose.staging.yml:1), including Postgres, Redis, Prometheus, Grafana, and Nginx.                                                                                                                                   | Base images and their tags determine much of the runtime attack surface.        |
| Secrets & configuration   | Application secrets and configuration loaded via `.env` files, validated by [`src/server/config.ts`](../src/server/config.ts:1) and [`src/shared/utils/envFlags.ts`](../src/shared/utils/envFlags.ts:1). CI secrets such as `GITHUB_TOKEN` and `SNYK_TOKEN` used by [`ci.yml`](../.github/workflows/ci.yml:1).                                   | Compromise or leakage can lead to full environment takeover.                    |

### 1.2 Threat scenarios

The following threats focus on realistic failure modes for RingRift’s current stack:

- **T1 – Dependency compromise (npm / PyPI)**
  - Malicious or compromised packages introduced via routine updates (for example `npm update`, `pip install -U`).
  - Typosquatting or dependency-confusion attacks leading to unexpected code execution.
  - Legitimate packages shipping backdoored versions which are transparently pulled into CI builds.

- **T2 – CI pipeline compromise**
  - Unreviewed changes to CI workflows under [`.github/workflows`](../.github/workflows/ci.yml:1) introducing data exfiltration or secret theft.
  - Attackers obtaining commit access and relying on insufficient CI checks (for example tests or audits not required for `main`).
  - Use of third-party GitHub Actions without pinned versions or digests, allowing supply-chain attacks via action updates.

- **T3 – Build artefact and image tampering**
  - Docker images built outside of CI and pushed directly to a registry or deployed manually.
  - Images rebuilt from moving `latest` tags in [`docker-compose.yml`](../docker-compose.yml:1) (for example `nginx:alpine`, `prom/prometheus:latest`, `grafana/grafana:latest`) without review.
  - Lack of provenance or SBOMs makes it difficult to answer “what exactly is running in production?” during an incident.

- **T4 – Secret exposure and misuse**
  - Accidental commit of `.env` files or local overrides (mitigated by [`.gitignore`](../.gitignore:1) but still a risk for new contributors).
  - Over-broad CI secrets (tokens with more privileges than needed) or reuse of the same credentials across environments.
  - Secrets echoed into logs or error messages from CI or runtime processes.

- **T5 – Unreviewed or weakly gated deployments**
  - Direct pushes to `main` or `develop` without review, followed by automatic CI builds.
  - Manual deployments from developer laptops using local Docker builds rather than CI artefacts.
  - Merges that bypass or ignore failing checks due to weak or undocumented branch-protection policies.

- **T6 – AI-service and training pipeline risks (medium term)**
  - Loading unverified model binaries into the AI service (for example ad-hoc `.pt` files built on developer machines).
  - Training data pipelines under [`ai-service/app/training`](../ai-service/app/training/generate_data.py:1) pulling from unvetted sources or unreviewed scripts.
  - Weak controls over where trained models are stored and who can update them, making targeted model backdoors harder to detect.

These threats are not independent; an attacker who compromises dependencies or CI may escalate to secrets, artefacts, and ultimately production data and game integrity.

## 2. Current Controls & Gaps

RingRift already implements several meaningful controls for S-05.F, especially around dependency audits and Docker-based builds. This section maps the threats above to current controls and the most important gaps.

### 2.1 CI job map (from `.github/workflows/ci.yml`)

For reference when working on S‑05.F.x tracks, this is the current CI job set as defined in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml:1), grouped by concern:

The CI workflow currently defines the following human‑readable job display names (as they appear under `name:` in `.github/workflows/ci.yml`):

- Lint and Type Check
- Run Tests
- TS Rules Engine (rules-level)
- Build Application
- Security Scan
- Docker Build Test
- Python Rules Parity (fixture-based)
- Python Dependency Audit
- Playwright E2E Tests

- **Lint & type safety**
  - `lint-and-typecheck` (**"Lint and Type Check" job**) – ESLint plus TypeScript compilation for root, server, and client (`npx tsc`, `npx tsc -p tsconfig.server.json`, `npx tsc -p tsconfig.client.json`).
- **General Jest tests & coverage**
  - `test` (**"Run Tests" job**) – `npm run test:coverage` over the Jest suite (see `tests/README.md`, `tests/TEST_LAYERS.md`, and `tests/TEST_SUITE_PARITY_PLAN.md` for the core vs diagnostics split and rules/trace/integration taxonomy). Uploads coverage to Codecov and posts an LCOV summary comment on PRs.
- **Rules-level Jest focus**
  - `ts-rules-engine` (**"TS Rules Engine (rules-level)" job**) – `npm run test:ts-rules-engine` targeting the shared‑engine / rules‑level suites (helpers → aggregates → orchestrator → contracts). This job is the primary TS rules semantics signal for CI.
- **Build & artefact packaging**
  - `build` (**"Build Application" job**) – `npm run build` for server and client, then archives `dist/` as an artefact.
- **Node dependency & supply‑chain scans**
  - `security-scan` (**"Security Scan" job**) – `npm audit --production --audit-level=high` plus a Snyk scan (`snyk/actions/node`) with `--severity-threshold=high` against the Node/TypeScript dependency graph.
- **Docker build sanity check**
  - `docker-build` (**"Docker Build Test" job**) – Docker Buildx build of the main `Dockerfile` (tagged `ringrift:test`, push disabled) to ensure the multi‑stage image still builds inside CI.
- **Python rules parity (TS→Python fixtures)**
  - `python-rules-parity` (**"Python Rules Parity (fixture-based)" job**) – generates TS→Python rules‑parity fixtures via `tests/scripts/generate_rules_parity_fixtures.ts`, then runs `python -m pytest ai-service/tests/parity/test_rules_parity_fixtures.py` under Python 3.11. This is the primary SSoT‑backed TS↔Python rules parity signal.
- **Python dependency audit**
  - `python-dependency-audit` (**"Python Dependency Audit" job**) – installs `ai-service/requirements.txt`, then runs `pip-audit -r requirements.txt --severity HIGH` to fail on known HIGH/CRITICAL dependency vulnerabilities.
- **E2E/browser‐level tests (currently non‑gating)**
  - `e2e-tests` (**"Playwright E2E Tests" job**) – stand‑up of Postgres+Redis via CI services, then Playwright E2E tests (`npm run test:e2e`) against a locally built app. Marked `continue-on-error: true` while the full infra stack is still being hardened; artefacts (Playwright reports) are uploaded for inspection.

When S‑05.F.1 is implemented, this job map should be used to define which jobs are **required gates** for `main`/`develop` merges (for example `lint-and-typecheck`, `test`, `ts-rules-engine`, `security-scan`, `docker-build`, `python-rules-parity`, `python-dependency-audit`) and which are optional/diagnostic (for example `e2e-tests`).

| Threat                                          | Current controls                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Gaps / improvement opportunities                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **T1 – Dependency compromise**                  | Node dependencies are locked via [`package-lock.json`](../package-lock.json:1). CI job `security-scan` in [`ci.yml`](../.github/workflows/ci.yml:169) runs `npm audit --production --audit-level=high` and a Snyk scan with `--severity-threshold=high`. Python dependencies for the AI service are pinned with exact versions in [`ai-service/requirements.txt`](../ai-service/requirements.txt:1), and the `python-dependency-audit` job runs `pip-audit -r requirements.txt --severity HIGH` (see [`ci.yml`](../.github/workflows/ci.yml:252)).                                                                             | No documented policy for when and how dependency updates are performed (for example monthly patch window vs ad-hoc). No consolidated SBOM or dependency inventory for the stack. Audit results are not yet summarized in docs or release notes.                                                                                                                                                                                                                                                                  |
| **T2 – CI pipeline compromise**                 | All CI configuration is centralized in [`ci.yml`](../.github/workflows/ci.yml:1). Lint, tests (`npm run test:coverage`, `npm run test:ts-rules-engine`), build, Docker build, security-scan, Python parity tests, and Python dependency audits all run for `push` and `pull_request` events on `main` and `develop`.                                                                                                                                                                                                                                                                                                           | There is no repo-level documentation of which jobs **must** be green before merging into `main` / `develop`. Branch-protection and review requirements live outside the repo and are not described in [`CONTRIBUTING.md`](../CONTRIBUTING.md:1). Third-party actions in CI are not consistently pinned to a specific major/minor version or digest.                                                                                                                                                              |
| **T3 – Build artefact and image tampering**     | Backend uses a multi-stage [`Dockerfile`](../Dockerfile:1) with a non-root runtime user (`ringrift`) and explicit copy of built artefacts. The AI service uses [`ai-service/Dockerfile`](../ai-service/Dockerfile:1) with a slim Python base and health checks. Local and staging stacks are defined via [`docker-compose.yml`](../docker-compose.yml:1) and [`docker-compose.staging.yml`](../docker-compose.staging.yml:1). CI has a `docker-build` job that performs a Docker Buildx build with caching but does not push images.                                                                                           | Base images in Dockerfiles and `docker-compose` use moving tags (for example `node:18-alpine`, `python:3.11-slim`, `prom/prometheus:latest`, `grafana/grafana:latest`, `nginx:alpine`). There is no documented policy that production deployments must use images built and tagged by CI, nor any guidance on recording image digests per release. No SBOM or provenance data is attached to built images. Image signing / attestation is not in scope today but could be a future enhancement.                  |
| **T4 – Secret exposure and misuse**             | Secrets are separated from code via `.env` files, with sample configuration in [`.env.example`](../.env.example:1) and staging guidance in [`.env.staging`](../.env.staging:1). Runtime configuration and topology are validated in [`src/server/config.ts`](../src/server/config.ts:1) and [`src/shared/utils/envFlags.ts`](../src/shared/utils/envFlags.ts:1), which prevent placeholder JWT secrets in `NODE_ENV=production`. CI uses GitHub Secrets for tokens such as `SNYK_TOKEN` and `GITHUB_TOKEN`. Logs use redaction helpers like `redactEmail` in [`src/server/utils/logger.ts`](../src/server/utils/logger.ts:63). | There is no repo-level secret-management guideline covering where secrets should live (for example CI secret store vs runtime secret manager), recommended rotation cadence, and who can access which secrets. `.env` handling for developers is described in [`README.md`](../README.md:198) but does not explicitly call out safe practices (for example “never commit real .env files”, “treat local .env as disposable”). Rotation playbooks (especially for JWT and DB credentials) are not yet documented. |
| **T5 – Unreviewed or weakly gated deployments** | CI already runs substantial checks (lint, tests, coverage, security scans, Docker build, parity tests, dependency audits) on `main` and `develop`. Database migration procedures and expectations for staging vs production are documented in [`docs/OPERATIONS_DB.md`](./OPERATIONS_DB.md:87) and summarized in [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md:155).                                                                                                                                                                                                                                                        | There is no concise description of the **minimum bar** for merging to `main` (required reviews, required status checks, no-fast-forward merges, etc.). There is no documented expectation that production deployments originate from tagged commits built by CI rather than ad-hoc local builds. Release procedures for when and how to run `prisma migrate deploy` in production are documented, but not tied to CI build artefacts or tags.                                                                    |
| **T6 – AI-service and training pipeline risks** | The AI service has its own Dockerfile [`ai-service/Dockerfile`](../ai-service/Dockerfile:1), pinned Python dependencies, and tests under [`ai-service/tests`](../ai-service/tests/test_env_interface.py:1). Training and dataset generation tooling lives under [`ai-service/app/training`](../ai-service/app/training/generate_data.py:1) and is described in [`AI_ARCHITECTURE.md`](../AI_ARCHITECTURE.md:1).                                                                                                                                                                                                                | There is no documented policy for how trained models are versioned, stored, and promoted to production. Training and evaluation jobs can currently be run from developer environments without a standardized container image or signed artefacts. Model integrity (for example checksums, signatures) and deployment provenance are not yet specified.                                                                                                                                                           |

The implementation tracks in Section 3 target the highest-value gaps above while staying compatible with the existing architecture and CI workflow.

## 3. Implementation Tracks (S-05.F.x)

This section defines concrete, narrow S-05.F.x tracks that can be implemented as future code-mode tasks. Tracks are ordered by impact and implementation cost; they are intentionally small and composable.

### S-05.F.1 – CI Required Checks & PR / Release Policy

- **Goal:** Make the expectations for reviews and required CI checks on `main` / `develop` explicit and discoverable for all contributors.
- **Risk reduction:** High (reduces the chance that compromised or untested changes reach production).
- **Complexity:** Low (primarily documentation plus light CI configuration).
- **Scope (likely files):** [`CONTRIBUTING.md`](../CONTRIBUTING.md:1), [`README.md`](../README.md:1), comments in [`ci.yml`](../.github/workflows/ci.yml:1); hosting-platform branch-protection settings (out of repo, but documented here).
- **Implementation outline:**
  - Enumerate which CI jobs must be green before merging into `main` (for example `lint-and-typecheck`, `test`, `ts-rules-engine`, `security-scan`, `python-dependency-audit`, `docker-build`).
  - Add a "CI & PR policy" section to [`CONTRIBUTING.md`](../CONTRIBUTING.md:1) describing:
    - Required reviews (for example at least one non-author review for `main`-bound PRs).
    - Required status checks and the expectation that they are enforced via branch protection.
    - Guidelines for when it is acceptable to re-run or temporarily disable specific jobs (for example flaky integration tests) and how to document exceptions.
  - Document the expectation that production deployments come from tagged commits on `main` that have passed all required checks.
- **Dependencies:** Existing CI jobs in [`ci.yml`](../.github/workflows/ci.yml:1) and the security posture described in [`docs/SECURITY_THREAT_MODEL.md`](./SECURITY_THREAT_MODEL.md:1).

### S-05.F.2 – Dependency Governance & SBOM Reporting

- **Goal:** Establish a lightweight dependency-governance process and produce machine-readable SBOMs for Node and Python stacks in CI.
- **Risk reduction:** High (supports faster response to upstream CVEs and improves incident response).
- **Complexity:** Medium (adds CI steps and a small amount of documentation).
- **Scope (likely files):** [`ci.yml`](../.github/workflows/ci.yml:1), [`package.json`](../package.json:1), [`ai-service/requirements.txt`](../ai-service/requirements.txt:1), release/ops notes in [`CURRENT_STATE_ASSESSMENT.md`](../CURRENT_STATE_ASSESSMENT.md:1) or [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md:155).
- **Implementation outline:**
  - Define a simple cadence for dependency updates (for example "apply non-breaking updates monthly; review major upgrades explicitly") and document it in [`CONTRIBUTING.md`](../CONTRIBUTING.md:1).
  - Extend the existing `security-scan` and `python-dependency-audit` jobs in [`ci.yml`](../.github/workflows/ci.yml:1) to:
    - **(Implemented, 2025-11-27 – Node)** Generate a CycloneDX SBOM for the Node/TypeScript dependency graph in the `security-scan` job using `npx @cyclonedx/cyclonedx-npm --output-format JSON --output-file sbom-node.json`, and upload it as a `sbom-node` artefact via `actions/upload-artifact` (current retention: 7 days).
    - **(Implemented, 2025-11-27 – Python)** Generate a CycloneDX SBOM for the AI-service Python environment in the `python-dependency-audit` job by installing `cyclonedx-bom` and running `python -m cyclonedx_py environment --output-format json --output-file sbom-python.json` from `ai-service/`, and upload it as a `sbom-python` artefact (current retention: 7 days).
    - **(Planned)** Add lightweight guidance for operators and incident responders on how to retrieve and inspect these SBOM artefacts when triaging CVEs or doing post-incident analysis.
  - Keep SBOM generation strictly **observational** (non-gating) so that it does not change build or test outcomes; treat SBOMs as supporting artefacts for incident response, audits, and future tooling.
  - Optionally, add a brief summary of dependency risks (for example high-severity findings from `npm audit` / `pip-audit`) to release notes or a changelog document.
- **Dependencies:** Builds on existing dependency-audit jobs already wired in [`ci.yml`](../.github/workflows/ci.yml:169) and Python tooling choices documented in [`AI_ARCHITECTURE.md`](../AI_ARCHITECTURE.md:1).

### S-05.F.3 – Docker Image Hardening, Tagging & Provenance

- **Goal:** Make Docker images more predictable and auditable by pinning base images, clarifying tagging conventions, and encouraging CI-built artefacts for deployment.
- **Risk reduction:** Medium–High (reduces exposure to surprise base-image changes and makes rollback safer).
- **Complexity:** Medium (changes to Dockerfiles, `docker-compose` files, and docs).
- **Scope (likely files):** [`Dockerfile`](../Dockerfile:1), [`ai-service/Dockerfile`](../ai-service/Dockerfile:1), [`docker-compose.yml`](../docker-compose.yml:1), [`docker-compose.staging.yml`](../docker-compose.staging.yml:1), deployment guidance in [`README.md`](../README.md:254) and [`docs/OPERATIONS_DB.md`](./OPERATIONS_DB.md:87).
- **Implementation outline:**
  - Replace moving base tags (for example `node:18-alpine`, `python:3.11-slim`, `prom/prometheus:latest`, `grafana/grafana:latest`, `nginx:alpine`) with pinned tags and, where practical, digests documented in comments.
  - Clarify in docs that production deployments should use images built by CI (for example tagged `ringrift-backend:<git-sha>` and `ringrift-ai-service:<git-sha>`), not arbitrary local builds.
  - Optionally add a CI job that builds and, in staging environments, pushes images to a registry with immutable tags; record those tags in release notes or deployment manifests.
  - Leave image signing and attestation (for example Sigstore/Cosign) as an explicitly **optional long-term enhancement**, not a requirement for initial S-05.F completion.
- **Dependencies:** Relies on the existing Docker build pipeline in [`ci.yml`](../.github/workflows/ci.yml:195) and the operations patterns described in [`docs/OPERATIONS_DB.md`](./OPERATIONS_DB.md:87).

### S-05.F.4 – Secret Management & Rotation Guidelines

- **Goal:** Document a provider-agnostic secret-management approach for local dev, CI, staging, and production, including rotation playbooks for critical secrets.
- **Risk reduction:** High (limits blast radius of leaked credentials and improves recovery from incidents).
- **Complexity:** Low–Medium (documentation plus small code/CI changes if needed).
- **Scope (likely files):** [`README.md`](../README.md:198), [`.env.example`](../.env.example:1), [`.env.staging`](../.env.staging:1), [`docs/OPERATIONS_DB.md`](./OPERATIONS_DB.md:127), [`docs/SECURITY_THREAT_MODEL.md`](./SECURITY_THREAT_MODEL.md:188), and possibly comments in [`ci.yml`](../.github/workflows/ci.yml:1).
- **Implementation outline:**
  - Define which secrets exist (JWT signing keys, DB credentials, Redis passwords, AI service tokens, third-party API keys) and which environments use each class of secret, without listing actual values.
  - Describe where secrets **should** live (for example CI secrets store, runtime secret manager, local `.env` for developers) and explicitly discourage committing real `.env` files or copying secrets between environments.
  - Add high-level rotation procedures for JWT keys and DB credentials (for example generate new key, deploy code that accepts both old and new, rotate tokens, retire old key), aligned with token-lifecycle guidance in S-05.A and data-lifecycle guidance in S-05.E.
  - Clarify how to safely use secrets in CI steps (for example avoid echoing them, prefer environment variables over command-line flags, restrict scopes of tokens like `SNYK_TOKEN`).
- **Dependencies:** Builds on existing env-validation logic in [`src/server/config.ts`](../src/server/config.ts:1) and [`src/shared/utils/envFlags.ts`](../src/shared/utils/envFlags.ts:1), as well as the data-lifecycle design in [`docs/DATA_LIFECYCLE_AND_PRIVACY.md`](./DATA_LIFECYCLE_AND_PRIVACY.md:1).

### S-05.F.5 – AI-Service Build & Training Pipeline Hardening

- **Goal:** Ensure that AI models and training pipelines are reproducible and verifiable, reducing the risk of model-level backdoors or regressions reaching production.
- **Risk reduction:** Medium (primarily affects AI quality and targeted compromise risk) but important as AI features grow.
- **Complexity:** Medium (touches training scripts, storage conventions, and AI-service startup logic).
- **Scope (likely files):** [`ai-service/Dockerfile`](../ai-service/Dockerfile:1), training scripts under [`ai-service/app/training`](../ai-service/app/training/generate_data.py:1), AI model loading in the service (see [`ai-service/app`](../ai-service/app/rules/default_engine.py:1)), and documentation in [`AI_ARCHITECTURE.md`](../AI_ARCHITECTURE.md:1).
- **Implementation outline:**
  - Standardize on a small set of Docker images for running training and evaluation jobs, ensuring they share the same base as the production AI-service image.
  - Define a versioning and storage scheme for trained models (for example `<model-family>/<version>/<board-type>` with checksums recorded in a manifest).
  - Update the AI service to load only models that appear in the manifest and to log the model identifier and checksum at startup for observability.
  - Optionally, record which model version is in use in game telemetry so that regressions and incidents can be correlated with model deployments.
- **Dependencies:** Builds on the existing AI-service architecture in [`AI_ARCHITECTURE.md`](../AI_ARCHITECTURE.md:1) and Python test harnesses under [`ai-service/tests`](../ai-service/tests/test_env_interface.py:1).

---

Treat this document as the **authoritative design** for S-05.F. Individual S-05.F.x tracks should be implemented incrementally and linked back here and to the S-05 backlog in [`docs/SECURITY_THREAT_MODEL.md`](./SECURITY_THREAT_MODEL.md:212) as they are completed.
