# SSoT Banner Guide

> **Doc Status (2025-11-27): Active**  
> **Role:** Canonical templates and conventions for Single Source of Truth (SSoT) banners across the RingRift documentation set. This file is the SSoT for _banner structure and wording only_; it does **not** define rules or lifecycle semantics itself.
>
> **SSoT alignment:** This guide is a derived meta‑doc over the existing SSoT layers:
>
> - **Rules/invariants semantics SSoT:** `RULES_CANONICAL_SPEC.md`, `ringrift_complete_rules.md`, `ringrift_compact_rules.md`, and the shared TS rules engine under `src/shared/engine/**` plus v2 contract vectors in `tests/fixtures/contract-vectors/v2/**`.
> - **Lifecycle/API SSoT:** `docs/CANONICAL_ENGINE_API.md` and the shared TS/WebSocket types under `src/shared/types/game.ts`, `src/shared/engine/orchestration/types.ts`, `src/shared/types/websocket.ts`, and `src/shared/validation/websocketSchemas.ts`.
> - **TS↔Python parity & determinism SSoT:** `docs/PYTHON_PARITY_REQUIREMENTS.md` plus the TS and Python parity/determinism test suites.
> - **AI/training SSoT:** `AI_ARCHITECTURE.md` and the executable training stack under `ai-service/app/training/**`.
> - **Operational SSoTs:** CI/config and infra artefacts + their drift guards (see `docs/SUPPLY_CHAIN_AND_CI_SECURITY.md`, `scripts/ssot/*.ts`, and `DOCUMENTATION_INDEX.md`).
>
> If this guide ever conflicts with those executable or canonical artefacts, **code, tests, and canonical rules/lifecycle docs win**. Update this guide and individual banners to match them.

---

## 1. Purpose

RingRift uses short, standardised **SSoT banners** at the top of key documents to:

- Make the **source of truth** for any given topic explicit.
- Clarify whether the doc is **canonical**, **derived** over one or more SSoTs, or **historical / non‑binding**.
- Provide a consistent hook for automated checks (`scripts/ssot/docs-banner-ssot-check.ts`) so that SSoT framing cannot silently drift or be removed.

This guide defines:

1. The **canonical banner skeleton** all SSoT‑aware docs should follow.
2. **Category‑specific templates** (rules semantics, lifecycle/API, operational, parity, AI/training, architecture, runbooks/incidents, historical docs).
3. Expectations for **content and precedence statements** within the banner.
4. The **minimum invariants** enforced (or expected to be enforced) by SSoT checks.

---

## 2. Canonical Banner Skeleton

Every SSoT‑aware doc should start with a block that contains, in order:

1. A **status line** (Active / Derived / Historical) and, optionally, a last‑updated date.
2. A **role line** describing what the doc is the source of truth for.
3. An **SSoT alignment paragraph** that:
   - Names the concrete SSoT layer(s) it depends on.
   - States clearly whether the doc is canonical or derived.
   - Includes the category‑specific snippet (e.g. `Rules/invariants semantics SSoT`, `Lifecycle/API SSoT`, `Operational SSoT`, `Canonical TS rules surface`).
4. A short **precedence statement**: what wins on conflict (usually “code + tests + canonical X doc win”).

A generic skeleton (to be adapted per category):

```md
> **Doc Status (YYYY-MM-DD): Active / Derived / Historical**  
> **Role:** Concise description of what this doc is for (e.g., canonical reference for X, derived architecture over Y, or historical post‑mortem for Z).
>
> **SSoT alignment:** This document is a [canonical / derived / historical] view over:
>
> - Primary SSoT(s): `...`
> - Related executable artefacts: `...`
> - Relevant tests or CI checks: `...`
>
> **Precedence:** If this document ever conflicts with the executable rules engine, contract vectors, shared types/schemas, or their tests, **code + tests win** and this document must be updated.
```

Automated checks currently require **at minimum** the substring `SSoT alignment` plus the category‑specific snippet listed below.

---

## 3. Category Templates

This section defines recommended templates and required snippets per doc category. These are intentionally concrete so they can be validated by simple substring / pattern checks.

### 3.1 Rules / Invariants Semantics SSoT Docs

**Examples:**

- `RULES_CANONICAL_SPEC.md` (RR‑CANON spec)
- `ringrift_complete_rules.md`, `ringrift_compact_rules.md`
- `RULES_ENGINE_ARCHITECTURE.md`
- `RULES_IMPLEMENTATION_MAPPING.md`
- `docs/RULES_ENGINE_SURFACE_AUDIT.md`

**Required snippet:** `Rules/invariants semantics SSoT`

**Banner template (example):**

```md
> **Doc Status (2025-11-27): Active**  
> **Role:** Developer‑facing description of the RingRift rules engine architecture and how hosts (backend, sandbox, Python) are wired around the canonical rules semantics.
>
> **SSoT alignment:** This document is a derived architectural view over the **Rules/invariants semantics SSoT**, which is:
>
> - `RULES_CANONICAL_SPEC.md` and the player‑facing rulebooks `ringrift_complete_rules.md` / `ringrift_compact_rules.md`.
> - The shared TypeScript rules engine at `src/shared/engine/**` plus v2 contract vectors under `tests/fixtures/contract-vectors/v2/**`.
> - The parity and determinism test suites in TS and Python that lock in behaviour.
>
> **Precedence:** Backend, sandbox, and Python hosts are **adapters** over this SSoT. If this file ever conflicts with the shared TS engine, contract vectors, or their tests, **the code + tests win** and this document must be updated.
```

Use this pattern for any doc that explains or catalogues rules behaviour but is not itself the canonical rules spec.

### 3.2 Lifecycle / API SSoT Docs

**Examples:**

- `docs/CANONICAL_ENGINE_API.md`
- `docs/STATE_MACHINES.md` (derived over the lifecycle SSoT)

**Required snippet for canonical doc:** `Lifecycle/API SSoT`

**Banner template for `CANONICAL_ENGINE_API.md`:**

```md
> **Doc Status (2025-11-27): Active**  
> **Role:** Canonical specification for the Move, decision, and WebSocket lifecycle exposed by the RingRift engine.
>
> **SSoT alignment:** This document, together with the shared TS/WebSocket types and schemas, forms the **Lifecycle/API SSoT**:
>
> - `src/shared/types/game.ts`
> - `src/shared/engine/orchestration/types.ts`
> - `src/shared/types/websocket.ts`
> - `src/shared/validation/websocketSchemas.ts`
>
> **Precedence:** If any other doc or host disagrees with these types/schemas and this lifecycle description, **the shared TS types + validation schemas win** and derived docs must be updated.
```

**Derived lifecycle/state‑machine docs** (e.g. `docs/STATE_MACHINES.md`) should explicitly call out that they are derived over the Lifecycle/API SSoT and defer to it on conflict, e.g.:

```md
> **SSoT alignment:** This document is a derived catalogue of shared state machines over the **Lifecycle/API SSoT** defined in `docs/CANONICAL_ENGINE_API.md` and the shared TS/WebSocket types. On conflict, the canonical API doc and shared types win.
```

### 3.3 TS↔Python Parity & Determinism Docs

**Examples:**

- `docs/PYTHON_PARITY_REQUIREMENTS.md`

**Required snippet:** `Canonical TS rules surface`

**Banner template:**

```md
> **Doc Status (2025-11-27): Active**  
> **Role:** Canonical specification for TS↔Python parity and determinism requirements over the shared rules engine.
>
> **SSoT alignment:** This document is a derived contract over the **Canonical TS rules surface**:
>
> - Shared TS rules engine modules under `src/shared/engine/**`.
> - Contract schemas and serialization in `src/shared/engine/contracts/**` plus v2 contract vectors in `tests/fixtures/contract-vectors/v2/**`.
> - Python adapter and parity suites under `ai-service/tests/parity/**` and `ai-service/tests/test_engine_determinism.py` / `test_no_random_in_rules_core.py`.
>
> **Precedence:** TS rules modules + contract vectors + parity tests are authoritative for semantics. This document describes how Python must match them; on conflict, **TS engine + parity tests win**.
```

### 3.4 AI / Training & Datasets Docs

**Examples:**

- `AI_ARCHITECTURE.md`
- `docs/AI_TRAINING_AND_DATASETS.md`
- `docs/AI_TRAINING_PREPARATION_GUIDE.md`
- `docs/AI_TRAINING_ASSESSMENT_FINAL.md`

**Required snippet for architecture doc:** `rules semantics SSoT` (to emphasise that AI hosts sit on top of the rules semantics SSoT).

**Banner template for `AI_ARCHITECTURE.md`:**

```md
> **Doc Status (2025-11-27): Active**  
> **Role:** High-level description of the AI service, model architectures, training pipelines, and how they integrate with the RingRift rules engine.
>
> **SSoT alignment:** This document is a derived architecture over:
>
> - The **rules semantics SSoT** (shared TS rules engine + contract vectors) for game behaviour.
> - The AI/training implementation SSoT under `ai-service/app/ai/**` and `ai-service/app/training/**`, plus their tests.
>
> **Precedence:** If this document ever conflicts with the shared TS rules engine, contract vectors, or the executable AI/training code and tests, **those executable artefacts win** and this document must be updated.
```

Other AI docs should make the same relationship explicit and clearly mark themselves as **derived**.

### 3.5 Operational / CI / Supply‑Chain Docs

**Examples:**

- `docs/SUPPLY_CHAIN_AND_CI_SECURITY.md`
- `docs/SECRETS_MANAGEMENT.md`
- `docs/DATA_LIFECYCLE_AND_PRIVACY.md`
- `docs/DEPLOYMENT_REQUIREMENTS.md`
- `docs/OPERATIONS_DB.md`
- `docs/ALERTING_THRESHOLDS.md`

**Required snippet (for the CI/security doc already wired into checks):** `Operational SSoT`

**Banner template for `docs/SUPPLY_CHAIN_AND_CI_SECURITY.md`:**

```md
> **Doc Status (2025-11-27): Active**  
> **Role:** Canonical reference for CI/CD, supply-chain, and dependency security controls for RingRift.
>
> **SSoT alignment:** This document is part of the **Operational SSoT** for CI and supply-chain security:
>
> - CI workflows under `.github/workflows/*.yml`.
> - Dependency manifests (`package.json`, `ai-service/requirements.txt`, Dockerfiles) and associated validation scripts.
> - SSoT drift-guard checks in `scripts/ssot/*.ts`.
>
> **Precedence:** YAML workflows, config validation scripts, and security scans are authoritative for behaviour; this document must describe **what actually runs in CI**. On conflict, **workflow code + scripts win**.
```

Other operational docs should:

- Reference the concrete config files (`docker-compose.yml`, `monitoring/prometheus/*.yml`, `src/server/config/**/*.ts`, etc.).
- State explicitly whether they are **canonical operational policy** (e.g. alert thresholds) or **derived explanation**.

### 3.6 Architecture / Design Docs (Derived)

**Examples:**

- `ARCHITECTURE_ASSESSMENT.md`
- `ARCHITECTURE_REMEDIATION_PLAN.md`
- `docs/MODULE_RESPONSIBILITIES.md`
- `docs/DOMAIN_AGGREGATE_DESIGN.md`
- `RULES_ENGINE_ARCHITECTURE.md` (rules‑focused, also covered under §3.1)

**Required snippet (for those already wired):** `SSoT alignment`

**Banner template (generic derived architecture doc):**

```md
> **Doc Status (2025-11-27): Active (derived)**  
> **Role:** Assessment / design document describing the current architecture and remediation plan for RingRift.
>
> **SSoT alignment:** This document is a derived architectural analysis over:
>
> - The rules semantics SSoT (`src/shared/engine/**` + contract vectors).
> - The lifecycle/API SSoT (`docs/CANONICAL_ENGINE_API.md` + shared types/schemas).
> - The operational SSoTs for CI, deployment, and monitoring.
>
> **Precedence:** This file is **never** the source of truth for behaviour. On any conflict with executable code, configs, or tests, **those executable artefacts win** and this document must be updated.
```

### 3.7 Runbooks & Incident Docs

**Examples:**

- `docs/runbooks/**`
- `docs/incidents/**`
- `docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md`

Runbooks and incident guides are typically **derived** over:

- Operational SSoTs (deployment requirements, env vars, monitoring configs).
- Security and data lifecycle SSoTs (for incidents involving data or abuse).

Recommended banner pattern:

```md
> **Doc Status (2025-11-27): Active Runbook**  
> **Role:** Step-by-step operational guide for $TOPIC (e.g., routine deployments, database migrations, rollback, incident triage).
>
> **SSoT alignment:** This runbook is a derived operational procedure over:
>
> - The Operational SSoT for deployment and infrastructure (`docs/DEPLOYMENT_REQUIREMENTS.md`, `docker-compose*.yml`, `src/server/config/**/*.ts`).
> - The Security & Data Lifecycle SSoTs (`docs/SECURITY_THREAT_MODEL.md`, `docs/DATA_LIFECYCLE_AND_PRIVACY.md`, `docs/SECRETS_MANAGEMENT.md`) where applicable.
>
> **Precedence:** If this runbook ever diverges from the actual deployment configs, monitoring rules, or security/data policies, **those configs/policies win** and the runbook must be updated.
```

Incident docs should also state whether they are **historical post‑mortems** or **active incident playbooks**, and which SSoT layers they interpret.

### 3.8 Historical / Archive Docs (Non‑SSoT)

**Examples:**

- `archive/**`
- Early phase reports and superseded plans.

Historical docs are **explicitly non‑SSoT**. Their banners should:

- Make clear that they are **snapshot / historical**.
- Point to the **current** SSoT location(s) for the same topic.

Recommended banner pattern:

```md
> **Doc Status (2025-11-27): Historical / Superseded**  
> **Role:** Archived report capturing the state of the system and plans at a specific point in time. Preserved for context and traceability.
>
> **SSoT alignment:** This document is **not** an SSoT. It has been superseded by:
>
> - Current architecture assessments: `ARCHITECTURE_ASSESSMENT.md`, `ARCHITECTURE_REMEDIATION_PLAN.md`.
> - Current rules semantics and lifecycle docs: `RULES_CANONICAL_SPEC.md`, `docs/CANONICAL_ENGINE_API.md`, and the shared TS rules engine under `src/shared/engine/**`.
>
> **Precedence:** On any conflict with current code, tests, or canonical docs, **this historical document loses**. Treat it as context only.
```

---

## 4. Automated Checks & Expectations

Current and planned SSoT drift guards for banners:

1. **Substring‑based checks (implemented):**
   - `scripts/ssot/docs-banner-ssot-check.ts` currently verifies that a curated list of core docs:
     - Contain the phrase `SSoT alignment`, and
     - Contain the configured category‑specific snippet (e.g. `Rules/invariants semantics SSoT`, `Lifecycle/API SSoT`, `Operational SSoT`, `rules semantics SSoT`, `Canonical TS rules surface`).
   - This ensures that the SSoT framing cannot be silently removed.

2. **Planned structural checks (recommended pattern):**
   - Future versions of `docs-banner-ssot-check.ts` **may**:
     - Require banners to appear within the first N lines of the file.
     - Look for headings or blockquotes that match simple patterns like `> **SSoT alignment:**`.
     - Expand coverage to additional docs (runbooks, incidents, security/ops docs).

3. **Doc categories vs SSoT types:**
   - When adding a new doc, explicitly decide whether it is:
     - A **canonical** SSoT doc for some surface.
     - A **derived** doc over one or more SSoTs.
     - A **historical / archive** doc.
   - Then:
     - Apply the appropriate banner template.
     - If it is core to correctness (rules, lifecycle, infra, security, data), consider wiring it into `docs-banner-ssot-check.ts` or a future specialised SSoT check.

---

## 5. How to Use This Guide

When creating or updating docs:

1. **Classify the doc** (rules semantics, lifecycle/API, parity, AI/training, operational, architecture, runbook/incident, historical).
2. **Copy the appropriate template** from §3 and adapt:
   - Status and date.
   - Role description.
   - Concrete paths to SSoT artefacts and tests.
3. **Ensure required snippets are preserved** (e.g. `Rules/invariants semantics SSoT`, `Lifecycle/API SSoT`, `Operational SSoT`, `Canonical TS rules surface`).
4. **Align precedence statements** with the actual SSoT layer (code/tests/config win over prose).
5. If the doc is critical, consider adding or extending automated checks so that future drift is caught by CI.

This guide should evolve alongside the SSoT landscape. When new SSoT layers or major doc categories are introduced, update this file first and then propagate the corresponding banners and checks.
