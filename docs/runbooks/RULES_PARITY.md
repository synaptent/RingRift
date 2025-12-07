# Rules Parity Runbook

> **Doc Status (2025-11-28): Active Runbook**  
> **Role:** Guide for investigating and resolving rules parity alerts between the canonical TypeScript rules engine and the Python rules implementation.
>
> **SSoT alignment:** This is **derived operational guidance** over:
>
> - **Monitoring SSoT:** Prometheus alert rules in `monitoring/prometheus/alerts.yml` (group `rules-parity`, alerts `RulesParityValidationMismatch`, `RulesParityHashMismatch`, `RulesParityGameStatusMismatch`) and scrape configuration in `monitoring/prometheus/prometheus.yml`.
> - **Canonical rules semantics:** Shared TypeScript engine helpers, aggregates, and orchestrator under `src/shared/engine/**`, plus contracts and v2 contract vectors in `src/shared/engine/contracts/**` and `tests/fixtures/contract-vectors/v2/**`.
> - **Parity metrics & logging:** `rulesParityMetrics` and `logRulesMismatch` in `src/server/utils/rulesParityMetrics.ts`, and the `RulesBackendFacade` / `PythonRulesClient` parity harness in `src/server/game/RulesBackendFacade.ts` and `src/server/services/PythonRulesClient.ts`.
> - **Python rules implementation:** Canonical Python rules engine in `ai-service/app/game_engine.py` and the derived/parity-focused modules under `ai-service/app/rules/**` (`default_engine.py`, `validators/*.py`, `mutators/*.py`).
> - **Parity specs & plans:** `docs/PYTHON_PARITY_REQUIREMENTS.md`, `RULES_ENGINE_ARCHITECTURE.md`, `RULES_IMPLEMENTATION_MAPPING.md`, `RULES_ENGINE_SURFACE_AUDIT.md`, `docs/PARITY_SEED_TRIAGE.md`, `docs/STRICT_INVARIANT_SOAKS.md`, and `tests/TEST_SUITE_PARITY_PLAN.md`.
>
> **Precedence:**
>
> - The **shared TypeScript rules engine + orchestrator** is the single source of truth for rules semantics.
> - Contract vectors, tests, and monitoring configs are authoritative for **what we check and alert on**.
> - This runbook explains **how to investigate and remediate** divergence. If it conflicts with code/config/tests, **code + configs + tests win** and this document should be updated.
>
> For a high-level “rules vs AI vs infra” classification, see `AI_ARCHITECTURE.md` §0 (AI Incident Overview). For suite‑by‑suite details of which tests are canonical vs diagnostic, see `tests/README.md` and `tests/TEST_SUITE_PARITY_PLAN.md`.
>
> Runtime rules flags: `ORCHESTRATOR_ADAPTER_ENABLED` is hardcoded to `true`; `ORCHESTRATOR_SHADOW_MODE_ENABLED` and `RINGRIFT_RULES_MODE` control shadow/authoritative posture. The legacy `ORCHESTRATOR_ROLLOUT_PERCENTAGE` flag was removed; rollout is always 100%.

---

## 1. When These Alerts Fire

**Alerts (from `monitoring/prometheus/alerts.yml`, `rules-parity` group):**

- `RulesParityValidationMismatch` (warning)
- `RulesParityHashMismatch` (warning)
- `RulesParityGameStatusMismatch` (critical)

**Conceptual behaviour (do not treat this as canonical; see `alerts.yml`):**

- These alerts watch counters derived from the rules parity harness (metrics named like `ringrift_rules_parity_*_mismatch_total`) over a recent time window.
- They fire when **validation verdicts**, **post-move state hashes**, or **game status (win/loss/draw)** differ between:
  - The canonical TypeScript engine path, and
  - The Python rules engine (typically exercised in shadow/contract mode via `RulesBackendFacade` and/or the Python parity test suites).

**Impact (by alert):**

- `RulesParityValidationMismatch`
  - Indicates disagreement on **whether a move is legal**.
  - Risk: players on different hosts (TS vs Python) may be allowed/denied different moves.

- `RulesParityHashMismatch`
  - Indicates disagreement on the **resulting game state** after an apparently identical move (board contents, ring stacks, markers, etc.).
  - Risk: divergence in long-running games, subtle bugs in capture/territory/line processing, or desync between training and inference.

- `RulesParityGameStatusMismatch`
  - Indicates disagreement on **game outcome** (win/loss/draw) and is marked `critical`.
  - Risk: different engines may declare different winners; this is a P0 whenever it can affect user-visible games or training results that feed back into the product.

**Context:**

- In normal operation, parity mismatches should be extremely rare and generally confined to:
  - Known historical seeds/scenarios tracked in `docs/PARITY_SEED_TRIAGE.md`.
  - Experimental Python changes being evaluated in shadow or training-only contexts.
- When alerts fire outside of those expected pockets, you must assume **a new semantic divergence** has been introduced and treat it as a regression until proven otherwise.

---

## 2. Quick Triage (First 5–10 Minutes)

> Goal: Confirm that the alert is real, identify which **parity dimension** is failing (validation vs hash vs game status), and determine whether this is coming from CI/test harnesses, staging traffic, or production game paths.

### 2.1 Parity triage checklist (canonical vs diagnostic)

1. **Check canonical rules suites first**
   - Run or confirm the status of:
     - TS `.shared` and rules‑level suites for the affected domain (movement, capture, lines, territory, victory).
     - TS contract vector runner: `tests/contracts/contractVectorRunner.test.ts`.
     - Python contract tests: `ai-service/tests/contracts/test_contract_vectors.py`.
   - If these are green and agree with the written rules (`RULES_CANONICAL_SPEC.md`, `ringrift_complete_rules.md`), treat their behaviour as authoritative.
2. **Identify which parity harnesses are firing**
   - Use `tests/TEST_SUITE_PARITY_PLAN.md` and `tests/README.md` to see whether the failing harness is:
     - A **canonical** fixture‑driven parity check (contracts / structured fixtures), or
     - A **diagnostic** trace/seed parity suite (recorded games, historical seeds, archived tests).
   - Remember: trace/seed suites are derived; when they disagree with `.shared` + contracts, the traces are usually stale.
3. **Check orchestrator flags and host mode**
   - Verify that parity jobs are running in the expected rules mode:
     - `ORCHESTRATOR_ADAPTER_ENABLED`
     - `ORCHESTRATOR_ROLLOUT_PERCENTAGE`
     - `ORCHESTRATOR_SHADOW_MODE_ENABLED`
     - `RINGRIFT_RULES_MODE`
   - Ensure that canonical parity jobs use the orchestrator‑ON posture (`RINGRIFT_RULES_MODE=ts`) unless they are explicitly testing legacy/SHADOW behaviour.
4. **Decide where to focus investigation**
   - If canonical contracts + `.shared` tests disagree between TS and Python → focus on aligning Python to TS (or, rarely, updating TS + contracts when the spec says TS is wrong).
   - If only diagnostic trace/seed suites disagree while contracts and `.shared` tests are green → treat traces as stale; update or archive them per the plan in `tests/TEST_SUITE_PARITY_PLAN.md` and `docs/PARITY_SEED_TRIAGE.md`.
   - If mismatches appear only under legacy/SHADOW rules modes → confirm whether they reflect known historical behaviour or a genuine regression; do not bend canonical TS rules to satisfy legacy traces.

### 2.1 Confirm which parity alert(s) are active

In Alertmanager / your monitoring UI:

1. Verify which of the following are firing and in which environment:
   - `RulesParityValidationMismatch`
   - `RulesParityHashMismatch`
   - `RulesParityGameStatusMismatch`
2. Note:
   - **Start time and duration** of each alert.
   - Whether all three are firing together or a subset only (e.g. `GameStatus` only).
   - Any annotations attached to the alert.

In Prometheus, use high-level queries (adjust windows to match `alerts.yml` if needed):

```promql
# Validation mismatch volume in the last hour
increase(ringrift_rules_parity_valid_mismatch_total[1h])

# Hash mismatch volume in the last hour
increase(ringrift_rules_parity_hash_mismatch_total[1h])

# Game status mismatches in the last hour
increase(ringrift_rules_parity_game_status_mismatch_total[1h])
```

If additional labels (e.g. `scenario`, `board_type`, `kind`, `engine`) are present, also inspect per-label breakdowns to see whether mismatches are concentrated in specific:

- Board types (e.g. square vs hex).
- Phases (placement, movement, capture, line, territory, victory).
- Test suites or harnesses.

### 2.2 Check recent rules / orchestrator / Python changes

Quickly review whether any of the following changed in the relevant environment shortly before the alert started:

- Shared TS rules engine (`src/shared/engine/**`).
- Game engine / orchestrator integration (`src/server/game/GameEngine.ts`, `RuleEngine.ts`, `RulesBackendFacade.ts`, `turnOrchestrator.ts`, state machines in `src/shared/stateMachines/**`).
- Python rules engine (`ai-service/app/game_engine.py`, `ai-service/app/rules/**`).
- Orchestrator rollout and feature flags (`ORCHESTRATOR_ADAPTER_ENABLED`, `ORCHESTRATOR_ROLLOUT_PERCENTAGE`, `ORCHESTRATOR_SHADOW_MODE_ENABLED`, `RINGRIFT_RULES_MODE`) per `docs/ORCHESTRATOR_ROLLOUT_PLAN.md`.

If there were no changes in any of these areas, triage should include **environment drift** (older image deployed in some clusters, partial rollouts, stale parity fixtures) as a possible cause.

### 2.3 Look for parity mismatch logs

`src/server/utils/rulesParityMetrics.ts` provides a structured logging helper:

- `logRulesMismatch(kind, details)` emits log lines with message key `rules_parity_mismatch` and a `kind` field (`valid`, `hash`, `S`, `gameStatus`, `backend_fallback`, `shadow_error`).

On a docker-compose deployment, from the repo root:

```bash
cd /path/to/ringrift

# Tail recent app logs for parity mismatches
docker compose logs app --tail=500 2>&1 \
  | grep -E 'rules_parity_mismatch' \
  | tail -n 100
```

From these entries, extract:

- Which **kind** of mismatch is observed (`valid`, `hash`, `gameStatus`, etc.).
- Any attached **scenario identifiers**, seeds, or move indices (often included in `details`).
- Whether mismatches occur in **shadow mode only** (Python running as a shadow engine) or in actual user-facing paths.

### 2.4 Determine whether this is CI-only or live-traffic driven

- If alerts are firing only in a **CI-like environment** or around scheduled parity jobs, check:
  - `scripts/ssot/python-parity-ssot-check.ts`
  - `scripts/run-python-contract-tests.sh`
  - `tests/scripts/generate_rules_parity_fixtures.ts`
  - Python parity tests under `ai-service/tests/parity/**` and `ai-service/tests/contracts/test_contract_vectors.py`.
- If alerts are firing in **staging or production**, verify whether:
  - Parity harness is running in **shadow-only mode** (TS authoritative, Python shadow).
  - Any user-facing paths actually depend on the Python rules engine for real decisions (e.g. certain AI/training flows).

This distinction affects urgency and the allowed short-term mitigations (see **4.4**).

---

## 3. Deep Diagnosis

> Goal: Pinpoint the **exact semantic disagreement** and which side (TS vs Python) needs to be changed.

### 3.1 Identify the domain of the mismatch (placement / movement / capture / line / territory / victory)

Use a combination of:

- Log fields from `rules_parity_mismatch` events (often including move type, board type, phase, and scenario IDs).
- Contract vector / fixture identifiers from parity tests.
- Domain-specific test failures in the following suites:
  - TS: `tests/contracts/contractVectorRunner.test.ts`, `tests/unit/LineDetectionParity.rules.test.ts`, `tests/unit/TerritoryDetection.*.test.ts`, `tests/unit/RefactoredEngineParity.test.ts`, etc.
  - Python: `ai-service/tests/contracts/test_contract_vectors.py`, `ai-service/tests/parity/test_rules_parity_fixtures.py`, `ai-service/tests/parity/test_line_and_territory_scenario_parity.py`, `ai-service/tests/parity/test_ts_seed_plateau_snapshot_parity.py`, etc.

Cross-reference the failing area with the **Python parity spec** in `docs/PYTHON_PARITY_REQUIREMENTS.md` (see the parity matrices for each domain: core, placement, movement, capture, line, territory, victory, turn).

### 3.2 Reproduce the mismatch in a controlled harness

Whenever possible, reproduce the mismatch via:

1. **Contract vectors (preferred):**
   - Locate or add a v2 contract vector under `tests/fixtures/contract-vectors/v2/**` that covers the failing move/sequence.
   - Run:

     ```bash
     # TS side
     npm test -- tests/contracts/contractVectorRunner.test.ts

     # Python side (from ai-service directory / env)
     cd ai-service
     pytest tests/contracts/test_contract_vectors.py
     ```

   - Ensure both sides are reading the **same fixture**.

2. **Generated parity fixtures:**
   - Use `tests/scripts/generate_rules_parity_fixtures.ts` to capture problematic scenarios from TS into JSON fixtures under `tests/fixtures/rules-parity/**`.
   - Run the Python parity runners that consume those fixtures, such as `ai-service/tests/parity/test_rules_parity_fixtures.py`.

3. **Seed-based / plateau diagnostics:**
   - For complex long-sequence divergences (especially involving LPS, capture chains, or territory), follow the workflows in:
     - `docs/PARITY_SEED_TRIAGE.md`
     - `tests/TEST_SUITE_PARITY_PLAN.md`
     - `docs/STRICT_INVARIANT_SOAKS.md` (invariant soak tests).

Reproduction in a deterministic harness is a prerequisite before attempting any code changes.

### 3.3 Decide which side is wrong (TS vs Python) using SSoT rules

Use these principles:

- The **shared TS engine + orchestrator** is the _default_ semantic SSoT.
- However, if diagnosis shows that TS behaviour itself violates:
  - The canonical rules spec (`RULES_CANONICAL_SPEC.md`, `ringrift_complete_rules.md`, `ringrift_compact_rules.md`), or
  - Previously agreed behaviour captured by contract vectors and tests,

  then the TS engine may need to be updated and the **contract vectors and docs** refreshed accordingly.

Checklist:

- [ ] Compare both behaviours against the human-readable rules specs and examples.
- [ ] Compare against existing v2 contract vectors for nearby cases.
- [ ] Consult recent architecture / audit docs: `RULES_ENGINE_ARCHITECTURE.md`, `RULES_IMPLEMENTATION_MAPPING.md`, `RULES_ENGINE_SURFACE_AUDIT.md`.

If in doubt, convene a short rules review and document the decision in `docs/PARITY_SEED_TRIAGE.md` or a dedicated incident/decision record.

### 3.4 Inspect the relevant implementation on each side

Once you know the domain and suspect side:

- **TypeScript side:**
  - `src/shared/engine/core.ts`, `movementLogic.ts`, `captureLogic.ts`, `lineDetection.ts`, `territoryProcessing.ts`, `victoryLogic.ts`.
  - Aggregates and mutators under `src/shared/engine/aggregates/**` and `src/shared/engine/mutators/**`.
  - Orchestrator and state machines in `src/shared/engine/orchestration/**`, `src/shared/stateMachines/**`.

- **Python side:**
  - Rules engine core: `ai-service/app/game_engine.py`, `ai-service/app/board_manager.py`.
  - Parity-focused implementation: `ai-service/app/rules/core.py`, `default_engine.py`, `validators/*.py`, `mutators/*.py`.

Use the parity matrices in `docs/PYTHON_PARITY_REQUIREMENTS.md` to see exactly which TS functions are mirrored by which Python functions.

---

## 4. Remediation

> Goal: Fix the underlying semantic divergence, keep the canonical rules surface coherent, and restore green parity metrics/alerts.

### 4.1 Fixing Python parity when TS semantics are correct (most common)

Typical steps when TS behaviour is deemed correct and Python is at fault:

1. **Update Python rules implementation:**
   - Adjust the relevant function(s) in `ai-service/app/game_engine.py`, `ai-service/app/board_manager.py`, or `ai-service/app/rules/**` to match TS behaviour.
   - Keep changes local to the specific domain (placement, movement, capture, line, territory, victory) where possible.

2. **Extend/adjust tests and fixtures:**
   - Add or update contract vectors under `tests/fixtures/contract-vectors/v2/**` to cover the corrected behaviour.
   - Extend Python tests (`ai-service/tests/parity/**`, `ai-service/tests/rules/**`) with targeted regression cases.
   - Where appropriate, also add TS-side regression tests in `tests/unit/**`.

3. **Re-run parity harnesses:**
   - TS: `npm test -- tests/contracts/contractVectorRunner.test.ts` and focused unit/parity suites.
   - Python: `cd ai-service && pytest tests/contracts/test_contract_vectors.py tests/parity`.
   - Optional: run `scripts/run-python-contract-tests.sh` and `npm run ssot-check` locally.

4. **Update docs if semantics were clarified (but not changed):**
   - If the fix surfaced ambiguous or undocumented behaviour, update `RULES_CANONICAL_SPEC.md` and/or supplementary clarifications under `docs/supplementary/RULES_RULESET_CLARIFICATIONS.md`.

### 4.2 Fixing TS semantics when the canonical spec says TS is wrong

Rare but possible when TS behaviour drifted from the written rules:

1. **Align TS engine with the formal rules spec:**
   - Update the relevant helper/aggregate/orchestrator logic under `src/shared/engine/**`.
   - Add regression tests that clearly encode the intended semantics.

2. **Update contract vectors and parity docs:**
   - Adjust v2 contract vectors to match the corrected behaviour.
   - Update `docs/PYTHON_PARITY_REQUIREMENTS.md` and, if needed, `CANONICAL_ENGINE_API.md`.

3. **Update Python to match the new canonical semantics:**
   - Treat the TS change as a spec change and mirror it into Python (as in **4.1**).

4. **Communicate and record the change:**
   - Capture a short change note (or incident-style entry) linking the old vs new behaviour and why the canonical spec was updated.

### 4.3 Handling game-status mismatches in live environments (P0 behaviour)

When `RulesParityGameStatusMismatch` is firing in an environment that serves real games:

1. **Freeze risky rollout paths:**
   - Follow `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` to:
     - Halt further increases to `ORCHESTRATOR_ROLLOUT_PERCENTAGE`.
     - If necessary, reduce rollout back to a known-good level or disable experimental engines/shadow paths.

2. **Ensure a single canonical engine is used for outcomes:**
   - Confirm that user-visible win/loss/draw decisions are taken from a single, trusted engine (usually the TS path).
   - Avoid switching the authoritative engine mid-incident without a clear rollout/rollback plan.

3. **Capture representative examples:**
   - Preserve at least one full game trace for each distinct divergence pattern (board type, victory condition, etc.).
   - Feed these into the parity triage workflow (fixtures, seeds, and docs).

### 4.4 Short-term mitigations vs long-term fixes

Short-term mitigations that **may** be acceptable while working on a proper fix (document in incidents):

- Reducing or disabling usage of the Python rules engine in any path that could influence production outcomes (keeping it for offline training only).
- Temporarily forcing all production decisions through the TS canonical engine, even if this disables certain experimental flows.

Long-term fixes **must**:

- Restore clean semantic parity between TS and Python for the affected domains.
- Be captured in contract vectors, tests, and parity docs so that regressions are prevented.

---

## 5. Validation

Before considering a rules-parity incident resolved:

### 5.1 Metrics and alerts

- [ ] `RulesParityValidationMismatch`, `RulesParityHashMismatch`, and `RulesParityGameStatusMismatch` (if they were firing) have all cleared and remained clear for at least one full evaluation window.
- [ ] The associated mismatch counters (e.g. `ringrift_rules_parity_valid_mismatch_total`, `ringrift_rules_parity_hash_mismatch_total`, `ringrift_rules_parity_game_status_mismatch_total`) are flat or increasing only due to **known historical edge cases** documented in `docs/PARITY_SEED_TRIAGE.md`.

### 5.2 Test suites

- [ ] TS contract vectors and parity suites are green:
  - `tests/contracts/contractVectorRunner.test.ts`
  - Relevant `tests/unit/*Parity*.test.ts` and `tests/unit/*Territory*/**.test.ts`.
- [ ] Python parity and rules tests are green:
  - `ai-service/tests/contracts/test_contract_vectors.py`
  - `ai-service/tests/parity/**`
  - Relevant `ai-service/tests/rules/**`.
- [ ] `npm run ssot-check` passes, in particular the `python-parity-ssot` and `rules-semantics-ssot` checks.

### 5.3 Behavioural checks

- [ ] For at least one representative example per fixed divergence, you have:
  - Verified that TS and Python now agree on validation, resulting hash, and game status.
  - Captured the scenario in a durable fixture (v2 contract vector or parity fixture) with a regression test.
- [ ] If orchestrator rollout was adjusted, it has been **safely** returned to the intended level per `docs/ORCHESTRATOR_ROLLOUT_PLAN.md`.

### 5.4 Documentation

- [ ] Any clarified or newly agreed semantics are reflected in:
  - `RULES_CANONICAL_SPEC.md` / rules markdowns.
  - `docs/PYTHON_PARITY_REQUIREMENTS.md` (function/type parity tables).
  - `docs/PARITY_SEED_TRIAGE.md` (for seeds/traces that remain exceptional by design).

---

## 6. Related Documentation & Runbooks

- **Rules semantics & architecture:**
  - `RULES_CANONICAL_SPEC.md`
  - `ringrift_complete_rules.md`, `ringrift_compact_rules.md`
  - `RULES_ENGINE_ARCHITECTURE.md`
  - `RULES_IMPLEMENTATION_MAPPING.md`
  - `RULES_ENGINE_SURFACE_AUDIT.md`
  - `RULES_SCENARIO_MATRIX.md`
  - `docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md`

- **Parity & invariants:**
  - `docs/PYTHON_PARITY_REQUIREMENTS.md`
  - `docs/PARITY_SEED_TRIAGE.md`
  - `docs/STRICT_INVARIANT_SOAKS.md`
  - `tests/TEST_SUITE_PARITY_PLAN.md`

- **Orchestrator rollout & modes:**
  - `docs/ORCHESTRATOR_ROLLOUT_PLAN.md`
  - `docs/drafts/ORCHESTRATOR_ROLLOUT_FEATURE_FLAGS.md`

- **Monitoring SSoT & ops:**
  - `monitoring/prometheus/alerts.yml`
  - `monitoring/prometheus/prometheus.yml`
  - `monitoring/README.md`
  - `docs/ALERTING_THRESHOLDS.md`
  - `docs/OPERATIONS_DB.md`

- **AI / training context (where Python rules are heavily exercised):**
  - `ai-service/AI_ASSESSMENT_REPORT.md`
  - `ai-service/AI_IMPROVEMENT_PLAN.md`
  - `docs/AI_TRAINING_AND_DATASETS.md`
  - `docs/AI_TRAINING_PREPARATION_GUIDE.md`
  - `docs/AI_TRAINING_ASSESSMENT_FINAL.md`

Use this runbook as a **playbook for investigation and coordination**; always defer to the canonical rules SSoT (rules spec + shared TS engine), monitoring configs, and parity tests for the ground truth of what “correct” means.
