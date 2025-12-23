# Next Areas Execution Plan

> **Created:** 2025-12-17
> **Updated:** 2025-12-20
> **Owner:** Tool-driven agent
> **Goal:** Execute the next slices from the Overall Assessment with tight scope per lane.

This plan assumes the working tree is unstable due to other agents. Each lane is scoped to the smallest file surface area needed to make safe, coherent progress and avoids touching unrelated files.

---

## Context Snapshot (Overall Assessment)

- **State:** Stable beta with consolidated orchestrator, strong TS-Python parity, and large test suites (historical snapshot: [`docs/archive/historical/CURRENT_STATE_ASSESSMENT.md`](../archive/historical/CURRENT_STATE_ASSESSMENT.md)).
- **Primary risk:** Canonical large-board datasets (square19/hex) are still low-volume (3 + 1 games) because canonical selfplay soaks exit with returncode -15 (SIGTERM) before higher-volume runs complete.
- **Quality gap:** Line coverage ~69% vs >=80% target and scenario matrix expansion still pending (historical: [`docs/archive/historical/CURRENT_STATE_ASSESSMENT.md`](../archive/historical/CURRENT_STATE_ASSESSMENT.md), `PROJECT_GOALS.md`, `KNOWN_ISSUES.md`).
- **UX/test polish:** Client coverage and weird-state UX/telemetry alignment are still P1 (`TODO.md`, `docs/UX_RULES_WEIRD_STATES_SPEC.md`).
- **Data readiness:** Canonical square8 gate now passes; square19/hex still need successful gated runs (`ai-service/TRAINING_DATA_REGISTRY.md`).
- **Documentation drift:** Core docs still reference `ringrift_complete_rules.md` / `ringrift_compact_rules.md`, but those files are missing from the repo; update references or restore stubs.

---

## Priority Order

1. Canonical rules parity validation (TS↔Python recovery/capture alignment + parity bundles).
2. Rules documentation SSoT alignment (missing rulebooks, anchor hygiene).
3. Canonical data pipeline scale-up (square19/hex runs + gate summaries).
4. IG-GMO experimental tier wiring (AI factory + docs).
5. Clean-scale validation rerun (baseline/target/AI-heavy with auth refresh + WS companion).
6. Scenario matrix + endgame coverage.
7. Client UX/test hardening.
8. WebSocket lifecycle polish.

---

## Lane 0: Canonical Rules Parity + Recovery Alignment

**Goal:** Ensure TS/Python core rules align with the canonical spec, especially recovery-in-movement semantics, and resolve any parity divergences.

**Scope:**

- `RULES_CANONICAL_SPEC.md`
- `src/shared/types/game.ts`
- `src/shared/engine/**`
- `ai-service/app/models/core.py`
- `ai-service/app/rules/**`
- `ai-service/app/ai/gpu_*`
- `ai-service/scripts/check_ts_python_replay_parity.py`
- `ai-service/scripts/diff_state_bundle.py`

**Plan:**

- Run parity with state bundles on a fresh DB and diff the first divergence.
- Fix any phase/move-type mismatches (especially recovery recorded as movement).
- Align GPU/internal comments and exports with canonical phase definitions.
- Update or add targeted parity tests if needed.

**Definition of Done:**

- Parity gate passes on at least one recent DB without semantic/structural mismatches.
- No documentation/code path implies a standalone recovery phase.

**Progress Log:**

- [x] Captured the square8 structural issue and traced it to self-capture gating in older parity runs.
- [x] Documented self-capture explicitly in the canonical spec and restored legacy rulebook shims.
- [x] Verified parity gate passes on a square8 DB using the updated codebase (TS replay via NODE_PATH to existing node_modules).

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

- [x] Audit client flows against UX specs (existing tests already cover ANM/FE/territory endgame flows).
- [x] Confirmed existing VictoryModal/GameHUD/ChoiceDialog suites already cover edge cases (no new tests required).
- [x] No UX copy/telemetry mismatches found during audit.

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

- canonical_square19.db and canonical_hexagonal.db meet volume targets and pass gates with a `game_moves`-backed schema.
- Registry lists new datasets with gate summary references.

**Progress Log:**

- [x] Define volume targets and update registry placeholders.
- [x] Ran canonical parity + history checks on P2P square8 DBs (no semantic divergences; end-of-game-only current_player mismatches in 4/50 games).
- [x] Canonical square8_2p gate passed via Vast.ai (200 games, canonical_ok true; `data/games/canonical_square8_2p.db` + `data/games/db_health.canonical_square8_2p.json`).
- [x] Repro: square19/hex soaks terminate with returncode -15 and 0 games recorded using default heuristic evaluation path.
- [x] Mitigation: `RINGRIFT_USE_MAKE_UNMAKE=true` enables square19 canonical soak; 1-game run passed parity/history (light band).
- [x] Found crash in make/unmake path (`NoneType.to_key` for non-spatial moves); fixed in `ai-service/app/ai/heuristic_ai.py` to skip `move.to` for no-op/decision moves.
- [x] Generated low-volume canonical DBs via direct soak + gate-only runs: square19 (3 games) and hexagonal (1 game) now `canonical_ok=true` with `data/games/db_health.canonical_*.json`.
- [x] Default large-board parity gates to `RINGRIFT_USE_MAKE_UNMAKE=true` (unless explicitly overridden) in `run_canonical_selfplay_parity_gate.py` and parity-only runs in `generate_canonical_selfplay.py`.
- [ ] Scale square19/hex to volume targets and stabilize `generate_canonical_selfplay.py` for >1 game on large boards (make/unmake + on-the-fly parity settings).
- [ ] Update `ai-service/TRAINING_DATA_REGISTRY.md` with square19/hex once gates pass (square8 already updated).

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

- [x] Audit WS lifecycle handling (reconnect + timeout + spectator coverage is already in unit/E2E suites).
- [x] Confirmed reconnect window + pending decision coverage in `GameSession.reconnect*` unit suites.
- [x] Extend WS load tests with optional reconnect simulation + metrics.
- [x] Extend WS load tests for spectator joins and pending decision timeouts (optional env-driven).

---

## Lane 7: IG-GMO Experimental Tier Wiring

**Goal:** Wire IG-GMO into the AI factory and document it as an experimental tier without changing canonical gameplay rules.

**Scope:**

- `src/server/game/ai/AIEngine.ts` (or AI factory entrypoint)
- `ai-service/app/main.py` (difficulty ladder exposure)
- `docs/ai/AI_TRAINING_AND_DATASETS.md`
- `docs/ai/AI_IMPROVEMENT_PROGRESS.md`

**Plan:**

- Confirm the intended difficulty band name and mapping for IG-GMO.
- Add the tier to the server-side AI factory + AI service ladder.
- Mark it as experimental in AI docs and expose any guardrails.

**Definition of Done:**

- IG-GMO is selectable in the AI factory with clear “experimental” docs.
- No canonical rules or training data expectations are altered.

**Progress Log:**

- [ ] Confirm tier naming and wiring entrypoints.
- [ ] Implement AI factory + AI service ladder wiring.
- [ ] Update AI docs to flag IG-GMO as experimental.

---

## Notes

- Ignore unrelated working tree changes unless they directly intersect the lane's scope.
- Prefer existing scripts (`tests/load/scripts/**`, `ai-service/scripts/**`) over ad-hoc tooling.
