# Pass 17 – Focused Assessment Report (Post-Orchestrator & AI Healthcheck Hardening)

> **⚠️ HISTORICAL DOCUMENT** – This is a point-in-time assessment from November 2025.
> For current project status, see:
>
> - [`../historical/CURRENT_STATE_ASSESSMENT.md`](../historical/CURRENT_STATE_ASSESSMENT.md) – Latest implementation status
> - `docs/PASS18A_ASSESSMENT_REPORT.md` – Most recent assessment pass

> **Assessment Date:** 2025-11-30
> **Assessment Pass:** 17 (post‑consolidation, rollout & invariants focus)
> **Assessor:** Architect mode – rules/orchestrator/AI/ops convergence

> This pass builds on PASS8–PASS16, the consolidated shared engine, and the
> current status/roadmap docs:
>
> - [`../historical/CURRENT_STATE_ASSESSMENT.md`](../historical/CURRENT_STATE_ASSESSMENT.md)
> - `STRATEGIC_ROADMAP.md`
> - `TODO.md`
> - `docs/ORCHESTRATOR_ROLLOUT_PLAN.md`
> - `docs/INVARIANTS_AND_PARITY_FRAMEWORK.md`
> - `docs/STRICT_INVARIANT_SOAKS.md`
> - `docs/SUPPLY_CHAIN_AND_CI_SECURITY.md`
>
> It focuses specifically on **environment rollout**, **rules parity &
> invariants**, and **AI healthchecks**, and assumes frontend UX and component
> testing have been substantially improved by the work tracked in PASS14–PASS16.

---

## 1. Updated Executive Summary (Pass 17)

- **Weakest area (Pass 17): Deep rules parity & invariants for territory / chain‑capture / endgames across TS↔Python and hosts.**  
  The shared engine and orchestrator are strong, and parity for core mechanics
  is excellent, but the hardest remaining correctness work concentrates in:
  - territory detection / processing and forced‑elimination sequences,
  - long multi‑phase traces (lines → territory → LPS / victory),
  - and remaining red / fragile scenario and parity suites called out in
    `STRATEGIC_ROADMAP.md` (P0 territory parity, property‑based tests, and
    dataset‑level validation) and `TODO.md` (P0.4 chain‑capture parity).

- **Hardest outstanding problem: Orchestrator‑first rollout and final legacy shutdown, now primarily an **environment + SLO enforcement** project.**  
  Implementation and instrumentation are in place (canonical orchestrator
  adapters, metrics, invariant and parity counters, Python AI healthchecks), but
  the remaining work is:
  - Driving environments through the rollout phases in
    `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` (§8) with real staging/production stacks.
  - Enforcing SLO‑driven gates on promotion and on `RINGRIFT_RULES_MODE` /
    `ORCHESTRATOR_*` phase changes.
  - Decommissioning or quarantining legacy backend/sandbox turn paths once
    rollout SLOs have held over multiple windows.

- **Key progress since PASS16 (orchestrator + AI invariants):**
  - **Orchestrator invariant metrics are live and fully wired:**
    - `MetricsService` now exports
      `ringrift_orchestrator_invariant_violations_total{type,invariant_id}` with
      stable `INV-*` IDs (e.g. `INV-S-MONOTONIC`, `INV-ACTIVE-NO-MOVES`,
      `INV-STATE-STRUCTURAL`, `INV-ORCH-VALIDATION`, `INV-TERMINATION`).
    - Prometheus alerts `OrchestratorInvariantViolationsStaging` and
      `OrchestratorInvariantViolationsProduction` in
      `monitoring/prometheus/alerts.yml` use this metric directly.
    - `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` no longer treats this metric as
      “planned”; staging and production SLOs reference the live label set.
  - **Python strict‑invariant metrics + alerts are fully integrated:**
    - `ai-service/scripts/run_self_play_soak.py` emits
      `ringrift_python_invariant_violations_total{invariant_id,type}` for strict
      anomalies (including `INV-S-MONOTONIC` and `INV-ACTIVE-NO-MOVES`).
    - `PythonInvariantViolations` in `monitoring/prometheus/alerts.yml` and
      `docs/operations/ALERTING_THRESHOLDS.md` promotes these to a warning‑level AI/Rules
      signal.
    - `docs/INVARIANTS_AND_PARITY_FRAMEWORK.md` and
      `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` now treat Python AI healthchecks as a
      first‑class **auxiliary CI/rollout signal** rather than a side channel.
  - **Rules parity metrics are bucketed by runtime suite:**
    - `RulesBackendFacade.compareTsAndPython` uses `recordRulesParityMismatch`
      to populate `ringrift_rules_parity_mismatches_total{mismatch_type,suite}`
      with suites `runtime_shadow`, `runtime_python_mode`, and `runtime_ts`.
    - New tests in `tests/unit/RulesBackendFacade.test.ts` assert that
      mismatches in python and shadow modes are recorded under the expected
      suites, tightening the contract between `RINGRIFT_RULES_MODE`, runtime
      behaviour, and parity alerts.
  - **Python AI healthchecks are formally part of the rollout story:**
    - CI job `python-ai-healthcheck` and the nightly workflow
      `ai-healthcheck-nightly.yml` are documented in
      `docs/SUPPLY_CHAIN_AND_CI_SECURITY.md` and
      `docs/INVARIANTS_AND_PARITY_FRAMEWORK.md` as exercising invariant IDs
      across `square8`, `square19`, and `hexagonal`.
    - `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` now explicitly treats green AI
      healthchecks and a quiet `PythonInvariantViolations` stream as
      **pre‑conditions** (P1/P2 signals) when promoting builds alongside the
      P0 orchestrator SLOs.

- **Frontend/host architecture and component testing have improved materially since PASS15–16.**  
  New host‑UX components (e.g. board controls overlay, sandbox touch controls),
  stronger adapter tests (`gameViewModels`, `choiceViewModels`), and additional
  E2E and unit tests under `tests/client/**` and `tests/unit/**` significantly
  reduce the previous host‑architecture and component‑testing weakness. These
  areas are no longer the primary bottleneck for v1.0 readiness.

---

## 2. Current Weakest Areas (Pass 17 View)

### 2.1 Territory / Forced-Elimination Parity & Invariants (TS↔Python, all boards)

- **Why it is weak:**
  - `STRATEGIC_ROADMAP.md` still lists P0 items for:
    - “Strengthen TS↔Python parity for territory detection and processing.”
    - “Parity for territory decision enumeration and forced‑elimination
      sequences.”
    - Property‑based tests and dataset‑level validation for territory /
      combined‑margin training data.
  - Python tests such as
    `ai-service/tests/parity/test_line_and_territory_scenario_parity.py` and
    the scaffolding for TS snapshot parity indicate that complex
    line+territory scenarios are **partially** covered, but not yet enforced via
    a fully symmetric TS snapshot harness.
  - Hex and high‑player‑count boards remain the highest‑risk geometrics for
    subtle territory and elimination bugs.
- **Impact:**
  - High – territory and elimination semantics are core to late‑game fairness,
    S‑invariant guarantees, and training correctness.
  - Disagreements in this area can produce divergent outcomes between TS and
    Python, or between hosts, and are hard to detect without targeted
    scenarios.
- **Hardness:**
  - High – requires coordinated fixtures, careful alignment with
    `RULES_CANONICAL_SPEC.md`, and likely property‑based exploration on both
    TS and Python sides.

### 2.2 Legacy / Sandbox Parity for Chain Capture & Advanced Phases

- **Why it is weak:**
  - `TODO.md` P0.4 still lists several chain‑capture and RulesMatrix
    scenario/parity suites as temporarily red or fragile while the new
    `chain_capture`/`continue_capture_segment` model is fully wired through
    backend and sandbox hosts.
  - Some archived trace/seed parity suites remain as diagnostics rather than
    green, canonical coverage, especially around historically problematic seeds
    (e.g. cyclic captures, ambiguous lines).
- **Impact:**
  - Medium‑High – issues here mostly affect edge seeds and diagnostic traces,
    but can surface as confusing replays or sandbox differences.
- **Hardness:**
  - Medium‑High – requires careful migration of scenario tests and tracing
    harnesses to the unified Move/phase model without regressing core rules.

### 2.3 Ops / Rollout Discipline (Phase Execution vs Plan)

- **Why it is weak:**
  - `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` now has precise env presets, SLOs, and
    alerts, but the plan is **aspirational** until:
    - A dedicated staging environment runs in “Phase 1 – orchestrator‑only”
      posture for sustained bake‑in windows.
    - Production environments progress through Phase 2 (shadow) and
      Phase 3–4 (incremental rollout → orchestrator‑only) with SLOs enforced.
  - No repo‑level doc yet summarises **actual current rollout posture per
    environment** (staging/prod), only the intended phases and env/flag
    presets.
- **Impact:**
  - High – production correctness depends as much on disciplined rollout and
    rollback as on the rules implementation itself.
- **Hardness:**
  - Medium – mostly operational / process work, but it requires coordination
    outside this repo and discipline in following the runbooks and SLOs.

---

## 3. Hardest Remaining Problems

1. **Operationalising orchestrator‑first rollout (Phases 1–4) with SLO gates.**
   - Move beyond “design complete” to **executed phases**:
     - Staging orchestrator‑only with `SLO-STAGE-ORCH-*` monitored and enforced.
     - Production shadow runs (`RINGRIFT_RULES_MODE=shadow`,
       `ORCHESTRATOR_SHADOW_MODE_ENABLED=true`, rollout percentage 0) with
       acceptable `ringrift_orchestrator_shadow_mismatch_rate` and no
       `RulesParityGameStatusMismatch` incidents.
     - Incremental production rollout with stable error and invariant budgets.
   - Coordinate orchestrator SLOs with Python AI healthchecks and rules parity
     alerts so that no single signal is interpreted in isolation.

2. **Deep parity and invariant coverage for territory / combined‑margin games.**
   - Extend contract vectors, scenario suites, and Python parity tests to cover
     more combinations of:
     - Board types (especially hex),
     - Player counts,
     - Sequences where territory decisions and forced elimination interact with
       S‑invariant and LPS (last‑player‑standing) logic.
   - Introduce property‑based tests (fast‑check on TS, Hypothesis on Python)
     for structural invariants on mid‑/late‑game snapshots.

3. **Data‑ and training‑level validation for AI pipelines.**
   - `STRATEGIC_ROADMAP.md` and AI training docs still identify missing
     dataset‑level checks (range/consistency/metadata validation for territory
     and combined margins), as well as stronger evaluation loops for heuristic
     and future NN models.
   - Without these, AI training may silently consume suboptimal or inconsistent
     data even if the underlying rules engines are correct.

---

## 4. Doc & Status Updates in This Pass

The following status and SSoT docs were updated as part of this pass to remove
drift and make the rollout/invariants story fully coherent:

- `docs/ORCHESTRATOR_ROLLOUT_PLAN.md`
  - Staging and production invariant SLOs now reference the **live**
    `ringrift_orchestrator_invariant_violations_total{type,invariant_id}` metric
    instead of a “planned” one.
  - Added an “Auxiliary CI signal – Python AI self‑play invariants” subsection
    tying the `python-ai-healthcheck` job and the nightly AI healthcheck
    workflow into the rollout SLO narrative.
- `docs/INVARIANTS_AND_PARITY_FRAMEWORK.md`
  - CI/alert mapping updated so that:
    - Python strict‑invariant soaks and AI healthchecks map to
      `ringrift_python_invariant_violations_total` and the
      `PythonInvariantViolations` alert.
    - TS orchestrator soaks map cleanly to
      `ringrift_orchestrator_invariant_violations_total{type,invariant_id}` and
      `OrchestratorInvariantViolations*`.
- `docs/operations/ALERTING_THRESHOLDS.md` and `monitoring/prometheus/alerts.yml`
  - Confirmed and documented alert wiring for orchestrator invariants,
    Python invariants, and rules parity, closing prior “no metrics/alerts yet”
    gaps.
- `tests/unit/RulesBackendFacade.test.ts`
  - New tests assert that runtime hash mismatches in python/shadow modes are
    recorded under `suite="runtime_python_mode"` and `suite="runtime_shadow"`,
    respectively, hardening the contract between `RINGRIFT_RULES_MODE`,
    runtime behaviour, and parity metrics.

These changes are **already applied in this repo** as part of Pass 17.

---

## 5. Recommended Next Steps (Inputs to Roadmap / TODO)

These items are intended to feed into `STRATEGIC_ROADMAP.md` and `TODO.md` as
concrete follow‑ups rather than replacing their existing P0/P1 tracks.

1. **P17.A – Territory & Endgame Parity Hardening (P0, Rules/Parity)**
   - Extend line+territory snapshot parity harnesses:
     - Generate canonical TS snapshots for the scenarios in
       `test_line_and_territory_scenario_parity.py`.
     - Add a TS snapshot‑based test that loads these initial/final states and
       asserts parity with Python.
   - Add a small property‑based test layer for territory invariants on both TS
     and Python, seeded from the structural invariants already defined in
     `docs/INVARIANTS_AND_PARITY_FRAMEWORK.md`.

2. **P17.B – Orchestrator Rollout Execution (P0, Env/Ops)**
   - Document the **current** rollout phase per environment (even if only in
     an internal ops note) and cross‑link it from
     [`../historical/CURRENT_STATE_ASSESSMENT.md`](../historical/CURRENT_STATE_ASSESSMENT.md) / `WEAKNESS_ASSESSMENT_REPORT.md`.
   - Run at least one full **Phase 1 – Staging orchestrator‑only** bake‑in
     window with SLOs monitored, and capture findings for future passes.
   - Plan and execute a controlled **Phase 2 – Production shadow** trial using
     the documented `RINGRIFT_RULES_MODE=shadow` + orchestrator shadow flags,
     with dashboards watching `ringrift_orchestrator_shadow_mismatch_rate` and
     `ringrift_rules_parity_mismatches_total{suite="runtime_shadow"}`.

3. **P17.C – AI Dataset & Training Validation (P1, AI)**
   - Implement dataset‑level validation passes for territory / combined‑margin
     datasets (schema, range, cross‑field invariants) and wire them into CI as
     non‑flaky gates.
   - Document evaluation loops for heuristic/NN models in
     `AI_TRAINING_AND_DATASETS.md`, making clear how they rely on the invariant
     and parity framework.

---

## 6. Notes on Git / Deployment Status

- The working tree currently contains a **substantial set of uncommitted
  changes**, including:
  - New docs and runbooks.
  - Orchestrator metrics/tests improvements.
  - Frontend host UX components and tests.
  - Python AI soak and heuristic‑mode test harnesses.
- These changes have **not** been committed or pushed in this assessment pass.
  To preserve reviewability and follow the repository’s contribution
  guidelines, commits and `git push` actions should be performed explicitly by
  a human operator, using appropriately scoped commit messages and PRs.

For future passes, treat this report as the **starting point** for planning
rules/territory parity hardening, orchestrator rollout execution, and AI
dataset validation work, building on the now‑solid shared engine,
orchestrator, and AI healthcheck foundations.
