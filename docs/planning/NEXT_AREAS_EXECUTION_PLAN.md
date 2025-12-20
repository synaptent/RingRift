# Next Areas Execution Plan

> **Created:** 2025-12-17
> **Updated:** 2025-12-19
> **Owner:** Tool-driven agent
> **Goal:** Execute the next slices from the Overall Assessment with tight scope per lane.

This plan assumes the working tree is unstable due to other agents. Each lane is scoped to the smallest file surface area needed to make safe, coherent progress and avoids touching unrelated files.

---

## Context Snapshot (Overall Assessment)

- **State:** Stable beta with consolidated orchestrator, strong TS-Python parity, and large test suites (`CURRENT_STATE_ASSESSMENT.md`).
- **Primary risk:** Production validation lacks a clean signal because auth/rate-limit noise dominated target runs (`WEAKNESS_AND_HARDEST_PROBLEM_REPORT.md`, `BASELINE_CAPACITY.md`).
- **Quality gap:** Line coverage ~69% vs >=80% target and scenario matrix expansion still pending (`CURRENT_STATE_ASSESSMENT.md`, `PROJECT_GOALS.md`, `KNOWN_ISSUES.md`).
- **UX/test polish:** Client coverage and weird-state UX/telemetry alignment are still P1 (`TODO.md`, `docs/UX_RULES_WEIRD_STATES_SPEC.md`).
- **Data readiness:** Canonical large-board datasets remain low volume (`ai-service/TRAINING_DATA_REGISTRY.md`).

---

## Priority Order

1. Parity gate documentation hygiene (doc-only, completed).
2. Clean-scale validation rerun (baseline/target/AI-heavy with auth refresh + WS companion).
3. Scenario matrix + endgame coverage.
4. Client UX/test hardening.
5. Canonical data pipeline scale-up.
6. WebSocket lifecycle polish.

---

## Lane 1: Parity Gate Documentation Hygiene

**Goal:** Remove static parity status claims and document the unified parity + history gate workflow.

**Scope:**

- `docs/runbooks/PARITY_VERIFICATION_RUNBOOK.md`
- `docs/ai/AI_TRAINING_AND_DATASETS.md`
- `docs/ai/AI_IMPROVEMENT_PROGRESS.md`
- `docs/ai/AI_PIPELINE_PARITY_FIXES.md`
- `DOCUMENTATION_INDEX.md`

**Plan:**

- Replace static parity status claims with guidance to check gate summaries.
- Add the combined parity + canonical history gate command to training docs.
- Mark parity-fix docs as historical and link to the parity runbook.
- Ensure the plan is indexed in the documentation index.

**Definition of Done:**

- Runbook and AI docs point to the unified gate and current summaries.
- No doc claims that parity is "passing" without pointing to gate outputs.

**Progress Log:**

- [x] Remove static parity status claims from parity runbook.
- [x] Add unified parity+history gate instructions to training docs.
- [x] Mark parity-fix notes as historical and link to the runbook.
- [x] Add the execution plan to the documentation index.

---

## Lane 2: Clean-Scale Validation Rerun

**Goal:** Make the next baseline/target/AI-heavy run produce a clean SLO signal with auth refresh + WS companion and updated capacity docs.

**Scope:**

- `tests/load/**`
- `docs/planning/PRODUCTION_VALIDATION_REMEDIATION_PLAN.md`
- `BASELINE_CAPACITY.md`

**Plan:**

- Align WS gameplay thresholds with `tests/load/config/thresholds.json` and emit reconnect metrics in summaries.
- Ensure load preflight captures auth TTL / pool credentials and the runners invoke it by default.
- Update remediation and capacity docs to point at the rerun steps and outputs.

**Definition of Done:**

- WS gameplay SLOs and summary output are aligned with thresholds.json.
- Baseline/target/AI-heavy runner scripts guard against auth noise.
- Capacity doc references the new validation run results (post-run).

**Progress Log:**

- [x] Add extended preflight checks (auth TTL / pool credential sanity) and document them.
- [x] Add WS reconnect simulation + metrics and document the knobs.
- [x] Align WS gameplay thresholds with `thresholds.json` true-error SLOs.
- [x] Include ws\_\* and reconnect metrics in load-test summaries.
- [x] Update capacity docs with clean-run results (PV-08 notes added).

---

## Lane 3: Scenario Matrix + Endgame Coverage

**Goal:** Expand multi-phase scenario coverage, especially GameEndExplanation paths across boards.

**Scope:**

- `tests/scenarios/**`
- `tests/unit/GameEndExplanation.*`
- `RULES_SCENARIO_MATRIX.md`
- `docs/UX_RULES_*`

**Plan:**

- Locate missing scenario rows in the matrix (ANM, FE, territory edge cases).
- Add one multi-phase scenario per missing category (square8 and square19/hex where applicable).
- Add or extend GameEndExplanation unit tests for FE/ANM/territory outcomes.

**Definition of Done:**

- Scenario matrix includes new entries for the identified gaps.
- At least one new multi-phase scenario and one GameEndExplanation test per gap.

**Progress Log:**

- [x] Locate and audit RULES_SCENARIO_MATRIX.md gaps.
- [x] Add scenarios for ANM/FE/territory outcomes.
- [x] Add/update GameEndExplanation tests for the new scenarios.

---

## Lane 4: Client UX/Test Hardening

**Goal:** Raise confidence in VictoryModal/GameHUD/ChoiceDialog flows with canonical endgame outcomes and telemetry.

**Scope:**

- `src/client/components/GameHUD.tsx`
- `src/client/components/ChoiceDialog.tsx`
- `src/client/components/VictoryModal.tsx`
- `tests/unit/**`, `tests/integration/**` as needed
- `docs/UX_RULES_*`

**Plan:**

- Audit endgame and decision-flow UX paths for ANM/FE/territory cases.
- Add targeted UI tests (or integration tests) for those flows.
- Align telemetry / copy with UX rules specs if drift is found.

**Definition of Done:**

- Critical endgame and decision flows have test coverage.
- UX copy and telemetry match canonical outcomes.

**Progress Log:**

- [ ] Audit client flows against UX specs.
- [ ] Add tests for VictoryModal/GameHUD/ChoiceDialog edge cases.
- [ ] Update UX copy/telemetry for any mismatches.

---

## Lane 5: Canonical Data Pipeline Scale-Up

**Goal:** Scale canonical square19/hex datasets with parity + history gates and document provenance.

**Scope:**

- `ai-service/scripts/generate_canonical_selfplay.py`
- `ai-service/scripts/check_ts_python_replay_parity.py`
- `ai-service/scripts/check_canonical_phase_history.py`
- `ai-service/TRAINING_DATA_REGISTRY.md`

**Plan:**

- Define target game counts for square19/hex.
- Run canonical selfplay gate and parity/history validation.
- Update registry and attach gate summaries.

**Definition of Done:**

- canonical_square19.db and canonical_hex.db meet volume targets and pass gates.
- Registry lists new datasets with gate summary references.

**Progress Log:**

- [ ] Define volume targets and update registry placeholders.
- [ ] Run generate_canonical_selfplay + parity/history gates (requires runtime).
- [ ] Update registry entries with gate summaries.

---

## Lane 6: WebSocket Lifecycle Polish

**Goal:** Harden reconnection windows, decision timeouts, and spectator flows under load.

**Scope:**

- `src/server/websocket/**`
- `src/client/hooks/useGameConnection.ts`
- `tests/load/scenarios/websocket-*.js`
- `tests/integration/**`

**Plan:**

- Audit reconnection and decision timeout logic (server + client).
- Add tests for reconnect windows and pending decisions.
- Extend load-test coverage for reconnect windows and spectator joins.

**Definition of Done:**

- Reconnection/timeout flows are test-covered and telemetry-backed.

**Progress Log:**

- [ ] Audit WS lifecycle handling (reconnect + timeout).
- [ ] Add tests around reconnect windows and pending decisions.
- [ ] Extend WS load tests for spectator/reconnect flows.

---

## Notes

- Ignore unrelated working tree changes unless they directly intersect the lane's scope.
- Prefer existing scripts (`tests/load/scripts/**`, `ai-service/scripts/**`) over ad-hoc tooling.
