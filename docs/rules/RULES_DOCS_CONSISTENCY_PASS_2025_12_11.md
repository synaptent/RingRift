# Rules Docs Consistency Pass (2025-12-11)

> **Doc Status (2025-12-11): Active (worklog / tracker)**
>
> **Purpose:** Track a repo-wide consistency pass resolving known contradictions across rulebooks, canonical spec, UX/analysis docs, and TS engine comments/tests.

---

## Scope

This pass focuses on four high-impact canonical clarifications and their propagation:

1. LPS “real actions” vs recovery
2. Square8 `lineLength` (2p exception)
3. Recovery legality (“at least `lineLength`”, overlength allowed)
4. Territory processing optionality (subset vs mandatory exhaustion)

---

## Canonical Decisions (SSoT)

- **LPS real actions (RR-CANON-R172):** Real actions are ring placement, non-capture movement, and overtaking captures. **Recovery and forced elimination are not real actions** and do not reset LPS tracking.
- **Square8 `lineLength`:** `lineLength = 4` for **2-player** square8; `lineLength = 3` for **3–4 player** square8. `lineLength = 4` for square19 and hex.
- **Recovery legality (RR-CANON-R112):** A recovery slide completes a line of **at least** `lineLength`; overlength lines are allowed and use Option 1/2 semantics.
- **Territory processing (RR-CANON-R140–R145 + RR-CANON-R075):** Player may process **any subset** of eligible regions; “stop early” is recorded as `skip_territory_processing` (distinct from `no_territory_action` when no regions exist).

---

## Changes Applied (High Level)

- **Canonical spec:** `RULES_CANONICAL_SPEC.md`
- **Player-facing rules:** `COMPLETE_RULES.md`, `COMPACT_RULES.md`, `HUMAN_RULES.md`
- **TS engine alignment (comments + phase validation):**
  - `src/shared/engine/lpsTracking.ts`
  - `src/shared/engine/playerStateHelpers.ts`
  - `src/shared/engine/phaseValidation.ts`
  - `src/shared/engine/orchestration/turnOrchestrator.ts`
  - `src/shared/engine/aggregates/RecoveryAggregate.ts`
  - `src/shared/types/game.ts`
  - Host comments: `src/server/game/GameEngine.ts`, `src/client/sandbox/ClientSandboxEngine.ts`
- **TS tests updated:** `tests/unit/engine/phaseValidation.test.ts`
- **Derived / analysis / planning docs aligned:** examples include
  - `docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md`
  - `docs/testing/SKIPPED_TESTS_TRIAGE.md`
  - `docs/architecture/AI_ARCHITECTURE.md`
  - `docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md`
  - `docs/supplementary/rules_analysis/RECOVERY_ACTION_DESIGN_AND_IMPLEMENTATION.md`
  - `docs/supplementary/rules_analysis/rules_analysis_recovery_action_interactions.md`
  - `docs/archive/plans/RECOVERY_ACTION_IMPLEMENTATION_PLAN.md`
  - `TODO.md`

---

## Verification Commands

- **TS (focused):**
  - `npm test -- tests/unit/engine/phaseValidation.test.ts`
  - `npm test -- tests/unit/playerStateHelpers.test.ts`
- **Python (focused):**
  - `pytest ai-service/tests/rules/test_recovery.py`
  - `pytest ai-service/tests/parity/test_recovery_parity.py`

---

## Known Follow-ups (Not Addressed Here)

- **TS↔Python contract drift:** Align the canonical MoveType surface across TS and Python, including (a) `choose_line_option` vs legacy `choose_line_reward`, and (b) `choose_territory_option` vs legacy `process_territory_region`.
- **Legacy vs canonical replay support:** Some TS validation paths still mention backwards-compatibility allowances (e.g., legacy move types during forced elimination). If legacy support remains necessary, make it explicitly opt-in and keep canonical validators strict.

Notes:

- This tracker’s follow-up pass below focuses on eliminating those naming drifts while keeping legacy replay tolerant but clearly non-canonical.

---

## Follow-up Pass (2025-12-12): MoveType Contract Unification (P0)

> **Goal:** Eliminate TS↔Python drift around phase↔MoveType mapping and “alias” move types by (1) choosing canonical names, (2) enforcing canonical strictness on write paths, and (3) keeping legacy tolerance explicit and opt-in for replay tooling only.

### Canonical naming decisions (SSoT)

- **Line option decision move:** canonical `choose_line_option`
  - Legacy alias: `choose_line_reward` (accepted for replay only; non-canonical for new recordings).
- **Territory region decision move:** canonical `choose_territory_option`
  - Legacy alias: `process_territory_region` (accepted for replay only; non-canonical for new recordings).

### Work items (tracked)

1. **Canonical spec / docs naming sweep**
   - Update `RULES_CANONICAL_SPEC.md` to use `choose_line_option` and `choose_territory_option`, treating `choose_line_reward` and `process_territory_region` as legacy aliases.
   - Update API/architecture docs (e.g. `docs/architecture/CANONICAL_ENGINE_API.md`, `docs/architecture/RULES_ENGINE_ARCHITECTURE.md`) to match.

2. **Python engine parity alignment**
   - Update `ai-service/app/game_engine/__init__.py` to **emit** canonical choice moves:
     - `choose_line_option` (instead of legacy `choose_line_reward`)
     - `choose_territory_option` (instead of legacy `process_territory_region`)
   - Keep legacy aliases accepted for replay only.

3. **Canonical history contract strictness (Python)**
   - Update `ai-service/app/rules/history_contract.py` to accept only canonical names for canonical history (`choose_line_option`, `choose_territory_option`, etc.).
   - Ensure DB write-time checks (e.g. `ai-service/app/db/game_replay.py`) align with the updated contract.

4. **Square19/Hex ring-count sync ✅ DONE (2025-12-12)**
   - Canonical values: `square19` = **72 rings/player**; victory thresholds **72/96/120** for 2/3/4 players.
   - Canonical values: `hexagonal` = **96 rings/player**; victory thresholds **96/128/160** for 2/3/4 players.
   - Synced TS + Python configs (`src/shared/types/game.ts`, `ai-service/app/rules/core.py`) and updated docs/tests.
   - Validation: `npm run lint`; focused Jest and pytest runs (including hex training encoder normalization and GPU parity tests where available).

5. **Legacy replay tolerance as explicit opt-in (TS)**
   - Gate phase coercions / compatibility allowances in `src/shared/engine/orchestration/turnOrchestrator.ts` behind an explicit replay option.
   - Keep default behavior strict for canonical engines and canonical DB generation.

6. **Experimental ruleset overrides (TS + GPU tooling) ✅ DONE (2025-12-12)**
   - Added per-game overrides for `ringsPerPlayer` and `lpsRoundsRequired` via `rulesOptions` (plus `GameState.lpsRoundsRequired` for serialization parity).
   - Updated shared TS engine entry points to honour overrides (initial state, placement cap checks, LPS evaluation, FSM context).
   - Fixed GPU simulator step semantics so each `ParallelGameRunner._step_games()` call processes at most one phase per game (restoring RR-CANON-R172 timing expectations in `ai-service/tests/gpu/test_gpu_lps_victory.py`).

### Verification commands (recommended)

- **TS (focused):**
  - `npm test -- tests/scenarios/FAQ_Q16_Q18.test.ts`
  - `npm test -- tests/scenarios/MultiplayerRotation.test.ts`
- **Python (focused):**
  - `pytest ai-service/tests/rules/test_phase_machine.py`
  - `pytest ai-service/tests/parity/test_recovery_parity.py`

---

## Follow-up Pass (2025-12-12): SSoT Drift Guards + CI Alignment (P0)

> **Goal:** Make contract drift and CI/doc mismatches fail fast so rules/engine changes don’t silently desync.

### Changes applied

- **Phase↔MoveType drift guard (TS↔Python):**
  - Added `scripts/ssot/phase-move-contract-ssot-check.ts` and wired it into `npm run ssot-check`.
  - Made `src/shared/engine/phaseValidation.ts` canonical-only (legacy aliases live under `src/shared/engine/legacy/`) to match Python `ai-service/app/rules/history_contract.py`.
- **SSoT docs alignment:**
  - Updated `docs/rules/RULES_IMPLEMENTATION_MAPPING.md` to reference missing RR-CANON rule anchors required by `rules-semantics-ssot`.
  - Updated `docs/architecture/CANONICAL_ENGINE_API.md` MoveType literal listing to include all non-legacy MoveTypes required by `lifecycle-api-ssot`.
- **CI alignment:**
  - Reintroduced the CI job `TS Orchestrator Parity (adapter-ON)` in `.github/workflows/ci.yml` to match SSoT guard expectations and supply-chain documentation.
- **Python FSM parity correctness:**
  - Updated `ai-service/app/rules/fsm.py` to respect RR-CANON-R075 “no silent phase skipping” (line_processing always advances to territory_processing) and to handle `CHOOSE_TERRITORY_OPTION` / `SKIP_TERRITORY_PROCESSING`.
  - Updated `ai-service/tests/rules/test_fsm_parity.py` expectations accordingly.

### Verification commands

- `npm run ssot-check`
- `pytest ai-service/tests/rules/test_fsm_parity.py`

---

## Follow-up Pass (2025-12-12): Recovery + Turn-Rotation Semantics Sweep (Docs + Copy)

> **Goal:** Remove remaining drift around recovery reachability, turn rotation, and ANM/LPS terminology by aligning the canonical spec and all derived docs to RR‑CANON‑R110/R112/R114/R115 and RR‑CANON‑R201.

### Canonical spec updates

- Updated RR‑CANON‑R200 global legal actions to include recovery moves in `movement` when recovery-eligible.
- Updated RR‑CANON‑R172 LPS wording to clarify that “full rounds” include only non‑permanently‑eliminated seats (per RR‑CANON‑R201 turn rotation).
- Updated RR‑CANON‑R115 recovery recording semantics to:
  - Use `from`/`to`, `recoveryMode`, `recoveryOption`, `collapsePositions`, `extractionStacks` fields.
  - Specify recovery-context territory self‑elimination via `eliminationContext: 'recovery'` (paired with RR‑CANON‑R114).

### Derived / associated docs aligned

- `docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md` – replaced “fully eliminated for turn rotation” with canonical permanent elimination terminology and updated rotation expectations.
- `docs/rules/INVARIANTS_AND_PARITY_FRAMEWORK.md` – updated formal invariant statements to allow temporarily eliminated players to remain `currentPlayer`, and redefined `INV-ANM-TURN-MATERIAL-SKIP` in terms of permanent elimination skip.
- `docs/supplementary/rules_analysis/rules_analysis_recovery_action_interactions.md` – updated recovery semantics (global fallback, stack-strike, Option 1/2 costs) and marked projections/checklists as historical where appropriate.
- `docs/architecture/SHARED_ENGINE_CONSOLIDATION_PLAN.md` – clarified forbidden placement uses `no_placement_action`, not `skip_placement`.
- `docs/ux/UX_RULES_TEACHING_GAP_ANALYSIS.md`, `docs/ux/UX_RULES_TEACHING_SCENARIOS.md` – updated teaching terminology (“permanently eliminated”).
- `docs/planning/WAVE_2025_12.md` – clarified legacy replay notes about “skipping players with no turn-material”.
- `ai-service/docs/SELFPLAY_ANALYSIS_REPORT_2025_12_10.md` – added a prominent “resolved” banner so the historical recovery contradiction isn’t treated as current.

### Code comment alignment (no rules changes)

- Clarified recovery eligibility wording in:
  - `src/shared/engine/playerStateHelpers.ts`
  - `ai-service/app/rules/core.py`
  - `ai-service/tests/parity/test_recovery_parity.py`
- Clarified GPU recovery docstring to note current GPU implementation coverage:
  - `ai-service/app/ai/gpu_parallel_games.py`

### Follow-ups (implementation work)

- **Recovery→Territory payment (RR‑CANON‑R114):** Implement recovery-context territory self-elimination (buried extraction) for `territory_processing` when a recovery slide creates a disconnection; thread `eliminationContext: 'recovery'` through TS+Python move models + validators and add parity tests.
- **GPU recovery parity:** If GPU self-play is used for canonical data, implement stack‑strike recovery in the GPU batch engine (or explicitly gate GPU mode as non‑canonical).
