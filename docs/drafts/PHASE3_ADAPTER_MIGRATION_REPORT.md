# Phase 3 Adapter Migration Report (Historical)

> **⚠️ HISTORICAL / COMPLETED** – This migration was completed in November 2025. The orchestrator is now at 100% rollout.
> For current status, see:
> - `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` – Production rollout status (Phase 4 complete)
> - `CURRENT_STATE_ASSESSMENT.md` – Implementation status

> **Doc Status (2025-11-30): Historical (completed migration report)**
> **Role:** Historical report for the Phase 3 migration to the shared TS orchestrator adapter. This document captures intent, milestones, and operational constraints around the rollout of the canonical shared engine in both backend and sandbox hosts.
>
> **SSoT alignment:** This report is a **derived operational doc** over the existing SSoTs:
>
> - **Rules/invariants semantics SSoT:** `RULES_CANONICAL_SPEC.md`, `ringrift_complete_rules.md`, `ringrift_compact_rules.md`, and the shared TypeScript rules engine under `src/shared/engine/**` plus v2 vectors in `tests/fixtures/contract-vectors/v2/**`.
> - **Lifecycle/API SSoT:** `docs/CANONICAL_ENGINE_API.md` and shared types/schemas in `src/shared/types/game.ts`, `src/shared/engine/orchestration/types.ts`, `src/shared/types/websocket.ts`, and `src/shared/validation/websocketSchemas.ts`.
> - **Operational SSoTs:** Deployment and topology configs (`docker-compose*.yml`, `src/server/config/**/*.ts`), orchestrator rollout controls in `OrchestratorRolloutService` / `ShadowModeComparator`, and related design notes in `docs/SHARED_ENGINE_CONSOLIDATION_PLAN.md` and `docs/drafts/ORCHESTRATOR_ROLLOUT_FEATURE_FLAGS.md`.
>
> **Precedence:** This report is **not** the source of truth for behaviour. If it ever conflicts with the shared TS engine, contract vectors, orchestrator implementation, or deployment/topology configs, **code + tests + configs win**, and this document must be updated to reflect reality.

---

## 1. Overview

Phase 3 of the rules engine consolidation focuses on promoting the shared TS orchestrator under `src/shared/engine/orchestration/**` to the primary execution path for turn processing across:

- The backend host (GameSession, RuleEngine, TurnEngine, WebSocket handlers)
- The client sandbox host (ClientSandboxEngine and sandbox adapters)
- The Python AI integration surface (via contract vectors and parity tests)

This report tracks the migration from legacy per-host turn pipelines to the canonical orchestrator adapter, including rollout controls, feature flags, and safety nets.

**Goals:**

- Route all production gameplay through the shared orchestrator without regressing determinism, parity, or performance.
- Maintain a fast, well-instrumented rollback path to the legacy pipeline.
- Use real traffic (via shadow mode and gradual rollout) to harden the orchestrator before full cutover.

---

## 2. Scope and Dependencies

### 2.1 In Scope

- Backend adoption of the orchestrator adapter for turn processing.
- Client sandbox adoption of the shared orchestrator surface.
- Rollout controls and feature flags:
  - `ORCHESTRATOR_ADAPTER_ENABLED`
  - `ORCHESTRATOR_ROLLOUT_PERCENTAGE`
  - `ORCHESTRATOR_SHADOW_MODE_ENABLED`
  - `ORCHESTRATOR_ALLOWLIST_USERS` / `ORCHESTRATOR_DENYLIST_USERS`
  - `ORCHESTRATOR_CIRCUIT_BREAKER_ENABLED` and associated thresholds.
- Monitoring and rollback procedures tied to these flags.

### 2.2 Out of Scope

- Changes to the underlying rules semantics (those are governed by the rules semantics SSoT and its test matrix).
- Long-term decommissioning of legacy host pipelines (tracked in `docs/drafts/LEGACY_CODE_ELIMINATION_PLAN.md`).

### 2.3 Upstream SSoTs and Artefacts

- Shared orchestrator: `src/shared/engine/orchestration/**`
- Orchestrator adapters and rollout logic: `src/server/services/OrchestratorRolloutService.ts`, `src/server/services/ShadowModeComparator.ts`, `src/client/sandbox/SandboxOrchestratorAdapter.ts`
- Parity and determinism scaffolding: rules parity fixtures and tests under `tests/` and `ai-service/tests/parity/**`.

---

## 3. Migration Stages (High-Level)

1. **Shadow Mode Enablement**
   - Run orchestrator in shadow alongside legacy pipeline.
   - Compare outcomes via `ShadowModeComparator` and parity metrics.
   - Alert on divergences without impacting live games.

2. **Canary Rollout**
   - Enable orchestrator for a small percentage of traffic via `ORCHESTRATOR_ROLLOUT_PERCENTAGE`.
   - Use allowlists/denylists to target internal accounts first.
   - Watch latency, error rate, and divergence metrics closely.

3. **Progressive Increase**
   - Gradually raise `ORCHESTRATOR_ROLLOUT_PERCENTAGE` (e.g. 10% → 25% → 50% → 100%).
   - Keep `ORCHESTRATOR_CIRCUIT_BREAKER_ENABLED=true` in production.
   - Use SSoT tests and soak runs to validate invariants.

4. **Full Cutover and Legacy Freeze**
   - Set `ORCHESTRATOR_ROLLOUT_PERCENTAGE=100` in all production clusters.
   - Treat legacy pipeline as a fallback path only (no new features).
   - Track remaining technical debt in `docs/drafts/LEGACY_CODE_ELIMINATION_PLAN.md`.

---

## 4. Operational Checklists (Draft)

> These sections are placeholders; concrete steps and commands should be
> filled in as migrations are executed. Until then, this report is
> **non-canonical** and advisory only.

### 4.1 Pre-Rollout Checklist

- [ ] All orchestrator-related unit and integration tests pass locally.
- [ ] Parity tests for backend vs sandbox vs Python are green.
- [ ] Monitoring dashboards include:
  - Orchestrator vs legacy error rates
  - Turn latency distributions (P50/P95/P99)
  - Circuit breaker state and recent trips
- [ ] Runbooks for orchestrator regressions and rollbacks are linked from `docs/runbooks/INDEX.md`.

### 4.2 Rollout Execution Notes

- [ ] Initial shadow-mode deployment completed on staging.
- [ ] First canary percentage and user allowlist agreed with the team.
- [ ] Changes to flags recorded in change management tooling (or release notes).

### 4.3 Rollback Strategy

- [ ] Document quick rollback steps (e.g. set `ORCHESTRATOR_ADAPTER_ENABLED=false` or `ORCHESTRATOR_ROLLOUT_PERCENTAGE=0`, redeploy, confirm via health checks).
- [ ] Ensure alerts fire if rollback is triggered by the circuit breaker.

---

## 5. Status and Follow-Ups

This report is intentionally kept as a **draft** until the Phase 3 migration is fully complete and validated in production.

Planned follow-ups:

- Replace this draft with a finalized, dated migration report once Phase 3 is complete.
- Cross-link final status into `docs/PASSXX_*` assessment reports and `DOCUMENTATION_AUDIT_REPORT.md`.
- Ensure `docs/ENVIRONMENT_VARIABLES.md` and `docs/DEPLOYMENT_REQUIREMENTS.md` accurately reflect the final, stable set of orchestrator-related flags and expectations.
