# Documentation Audit Report

**Date:** November 26, 2025  
**Scope:** Project-wide documentation review for accuracy, consistency, and organization

## Executive Summary

This audit reviews all documentation against the codebase to ensure accuracy, eliminate redundancy, and optimize organization. The audit covered documentation in root (`/`), `docs/`, `archive/`, `deprecated/`, and `ai-service/` directories.

**Key Findings:**

- Root directory contains **17 documentation files** (within target of ~20)
- `docs/` contains **20 active documents** plus organized subdirectories
- `deprecated/` contains **34 files** requiring cleanup
- `archive/` contains **11 historical reports** (appropriately archived)
- Multiple documents reference superseded architecture or outdated implementation status

## Current Documentation Inventory

### Root Directory Documentation (17 files - ✅ Within Target)

| File                               | Purpose                       | Status                   | Recommendation               |
| ---------------------------------- | ----------------------------- | ------------------------ | ---------------------------- |
| `README.md`                        | Main entry point              | ✅ Active, comprehensive | Keep, update dates           |
| `QUICKSTART.md`                    | Getting started guide         | ✅ Active                | Keep                         |
| `CONTRIBUTING.md`                  | Contribution guidelines       | ✅ Active                | Keep                         |
| `AI_ARCHITECTURE.md`               | AI Service architecture       | ✅ Active                | Keep                         |
| `ARCHITECTURE_ASSESSMENT.md`       | Comprehensive arch review     | ✅ Active                | Keep                         |
| `ARCHITECTURE_REMEDIATION_PLAN.md` | Architecture improvement plan | ⚠️ Partially complete    | Review implementation status |
| `CURRENT_RULES_STATE.md`           | Rules implementation status   | ✅ Active                | Keep                         |
| `CURRENT_STATE_ASSESSMENT.md`      | Code-verified status          | ✅ Active, canonical     | Keep                         |
| `KNOWN_ISSUES.md`                  | Bug/gap tracker               | ✅ Active                | Keep                         |
| `STRATEGIC_ROADMAP.md`             | Phased implementation plan    | ✅ Active                | Keep                         |
| `TODO.md`                          | Task tracker                  | ✅ Active                | Keep                         |
| `ringrift_complete_rules.md`       | Complete rulebook             | ✅ Active, canonical     | Keep                         |
| `ringrift_compact_rules.md`        | Compact rules                 | ✅ Active, canonical     | Keep                         |
| `RULES_CANONICAL_SPEC.md`          | Rules specification           | ✅ Active                | Keep                         |
| `RULES_ENGINE_ARCHITECTURE.md`     | Rules engine arch             | ✅ Active                | Keep                         |
| `RULES_IMPLEMENTATION_MAPPING.md`  | Rules → code mapping          | ✅ Active                | Keep                         |
| `RULES_SCENARIO_MATRIX.md`         | Test scenario mapping         | ✅ Active                | Keep                         |

**Analysis:** Root documentation is well-organized and within target count. All files serve clear purposes and are actively maintained.

### docs/ Directory (20 active + subdirectories)

#### Active Documents (20)

| File                                       | Purpose               | Status    |
| ------------------------------------------ | --------------------- | --------- |
| `INDEX.md`                                 | Documentation hub     | ✅ Active |
| `API_REFERENCE.md`                         | API documentation     | ✅ Active |
| `CANONICAL_ENGINE_API.md`                  | Engine API spec       | ✅ Active |
| `MODULE_RESPONSIBILITIES.md`               | Module structure      | ✅ Active |
| `DOMAIN_AGGREGATE_DESIGN.md`               | Aggregate patterns    | ✅ Active |
| `PYTHON_PARITY_REQUIREMENTS.md`            | TS/Python parity      | ✅ Active |
| `RULES_ENGINE_SURFACE_AUDIT.md`            | Engine surface review | ✅ Active |
| `PARITY_SEED_TRIAGE.md`                    | Parity debugging      | ✅ Active |
| `AI_TRAINING_AND_DATASETS.md`              | AI training guide     | ✅ Active |
| `AI_TRAINING_PREPARATION_GUIDE.md`         | AI training setup     | ✅ Active |
| `ENVIRONMENT_VARIABLES.md`                 | Env config reference  | ✅ Active |
| `SECRETS_MANAGEMENT.md`                    | Secrets handling      | ✅ Active |
| `SECURITY_THREAT_MODEL.md`                 | Security analysis     | ✅ Active |
| `SUPPLY_CHAIN_AND_CI_SECURITY.md`          | CI/CD security        | ✅ Active |
| `DATA_LIFECYCLE_AND_PRIVACY.md`            | Data handling         | ✅ Active |
| `OPERATIONS_DB.md`                         | Database operations   | ✅ Active |
| `DEPLOYMENT_REQUIREMENTS.md`               | Deployment guide      | ✅ Active |
| `ALERTING_THRESHOLDS.md`                   | Monitoring config     | ✅ Active |
| `STRICT_INVARIANT_SOAKS.md`                | Soak testing          | ✅ Active |
| `INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md` | Incident report       | ✅ Active |

#### Subdirectories

**docs/drafts/ (9 files)**

- Contains work-in-progress planning documents
- Should be reviewed for completion or archival

**docs/incidents/ (7 files)**

- Incident response playbooks
- Well-organized and appropriate

**docs/runbooks/ (6 files)**

- Operational runbooks
- Well-organized and appropriate

**docs/supplementary/ (5 files)**

- Supporting documentation
- Appropriate location

### archive/ Directory (11 files - ✅ Appropriate)

Historical reports and analyses that provide context but are superseded:

- `AI In Depth Improvement Analysis.md` - Historical AI analysis
- `DOCUMENTATION_REVIEW_REPORT.md` - Previous documentation review
- `FINAL_ARCHITECT_REPORT.md` - Historical architecture report
- `FINAL_RULES_AUDIT_REPORT.md` - Historical rules audit
- `HARDEST_PROBLEMS_REPORT.md` - Historical problem analysis
- `RULES_ANALYSIS_PHASE2.md` - Historical rules analysis
- `RULES_DYNAMIC_VERIFICATION.md` - Historical verification approach
- `RULES_STATIC_VERIFICATION.md` - Historical verification approach
- `SEED5_LPS_ALIGNMENT_PROGRESS.md` - Historical parity work
- `SEED5_TAIL_DIVERGENCE_DIAGNOSTIC_SUMMARY.md` - Historical diagnostics
- `TRACE_PARITY_CONTINUATION_TASK.md` - Historical task tracking

**Status:** ✅ Properly archived, provides historical context

### deprecated/ Directory (34 files - ⚠️ Requires Cleanup)

Contains superseded documents that should be reviewed:

**Candidates for Deletion** (truly obsolete with no historical value):

- `DOCUMENTATION_UPDATE_SUMMARY 14.15.34.md` (duplicate timestamp version)
- `DOCUMENTATION_UPDATE_SUMMARY.md` (superseded)
- `P0_TASK_18_STEP_2_SUMMARY 14.14.42.md` (duplicate timestamp version)
- `P0_TASK_18_STEP_2_SUMMARY.md` (superseded)
- `P0_TASK_18_STEP_3_SUMMARY 14.14.50.md` (duplicate timestamp version)
- `P0_TASK_18_STEP_3_SUMMARY.md` (superseded)
- `DOCKER_SETUP.md` (superseded by README/QUICKSTART)
- `VSCODE_DOCKER_GUIDE.md` (superseded by README/QUICKSTART)
- `IMPLEMENTATION_STATUS.md` (superseded by CURRENT_STATE_ASSESSMENT)
- `FAQ_TEST_IMPLEMENTATION_SUMMARY.md` (superseded by rules matrix)

**Candidates for Archive** (historical value):

- All `P0_TASK_*` documents → move to archive/
- All `RULES_ANALYSIS_*` documents → move to archive/ if not already there
- `AI_ASSESSMENT_REPORT.md`, `AI_CODE_REVIEW_REPORT.md`, `AI_IMPROVEMENT_PLAN.md` → move to archive/
- `CODEBASE_EVALUATION.md` → superseded by ARCHITECTURE_ASSESSMENT (move to archive/)
- `TECHNICAL_ARCHITECTURE_ANALYSIS.md` → superseded by ARCHITECTURE_ASSESSMENT (move to archive/)
- `REFACTORING_ARCHITECTURE_DESIGN.md` → superseded by ARCHITECTURE_REMEDIATION_PLAN (move to archive/)
- `ringrift_architecture_plan.md` → early planning doc (move to archive/)
- `RULES_GAP_ANALYSIS_REPORT.md` → already in archive/, remove from deprecated/

### ai-service/ Documentation (2 files - ✅ Appropriate)

- `ai-service/README.md` - Service-specific documentation
- `ai-service/AI_ASSESSMENT_REPORT.md` - AI service assessment

**Status:** ✅ Properly located within service directory

## Cross-Reference Analysis

### Documentation Consistency Issues

1. **Date Inconsistencies:**
   - `README.md` shows "Last Updated: November 22, 2025"
   - Should be updated to reflect current audit date (November 26, 2025)

2. **Architecture Documentation:**
   - `ARCHITECTURE_ASSESSMENT.md` references `deprecated/CODEBASE_EVALUATION.md`
   - `ARCHITECTURE_REMEDIATION_PLAN.md` implementation status needs verification
   - Multiple architecture documents should clearly indicate supersession relationships

3. **Rules Documentation:**
   - `ringrift_complete_rules.md` and `ringrift_compact_rules.md` are canonical (✅)
   - `RULES_CANONICAL_SPEC.md` serves different purpose (developer-focused spec vs player-focused rules)
   - No conflicts identified

4. **Status Documentation:**
   - `CURRENT_STATE_ASSESSMENT.md` is canonical for implementation status
   - `CURRENT_RULES_STATE.md` focuses specifically on rules implementation
   - `KNOWN_ISSUES.md` tracks active bugs/gaps
   - `TODO.md` tracks task-level work
   - Clear separation of concerns (✅)

### Codebase Alignment Issues

**Items Found in README.md Requiring Verification:**

1. **Deployment Topology Section:**
   - README describes `RINGRIFT_APP_TOPOLOGY` environment variable
   - ✅ Confirmed implemented in `src/server/config/topology.ts`
   - ✅ Docker Compose configs reference this correctly

2. **Sandbox AI Stall Diagnostics:**
   - README describes detailed stall detection features
   - Need to verify: `window.__RINGRIFT_SANDBOX_TRACE__` implementation
   - Need to verify: Environment flags `RINGRIFT_ENABLE_SANDBOX_AI_STALL_DIAGNOSTICS`

3. **Contract Test Vectors:**
   - README lists 14 TypeScript vectors, 15 Python tests
   - Should verify counts match actuals in `tests/fixtures/contract-vectors/v2/`

4. **API Endpoints:**
   - README documents comprehensive API surface
   - Should verify all endpoints exist and match descriptions

## Recommendations

### Immediate Actions (Priority 1)

1. **Clean up deprecated/ directory:**

   ```bash
   # Delete truly obsolete files
   rm deprecated/DOCUMENTATION_UPDATE_SUMMARY*.md
   rm deprecated/P0_TASK_18*.md
   rm deprecated/DOCKER_SETUP.md
   rm deprecated/VSCODE_DOCKER_GUIDE.md
   rm deprecated/IMPLEMENTATION_STATUS.md
   rm deprecated/FAQ_TEST_IMPLEMENTATION_SUMMARY.md

   # Move to archive/
   mv deprecated/AI_ASSESSMENT_REPORT.md archive/
   mv deprecated/AI_CODE_REVIEW_REPORT.md archive/
   mv deprecated/AI_IMPROVEMENT_PLAN.md archive/
   mv deprecated/AI_STALL_*.md archive/
   mv deprecated/AI_TOURNAMENT_RESULTS*.md archive/
   mv deprecated/CODEBASE_EVALUATION.md archive/
   mv deprecated/TECHNICAL_ARCHITECTURE_ANALYSIS.md archive/
   mv deprecated/REFACTORING_ARCHITECTURE_DESIGN.md archive/
   mv deprecated/ringrift_architecture_plan.md archive/
   mv deprecated/RINGRIFT_IMPROVEMENT_PLAN.md archive/
   mv deprecated/PLAYABLE_GAME_IMPLEMENTATION_PLAN.md archive/
   mv deprecated/PYTHON_RULES_*.md archive/
   mv deprecated/BOARD_TYPE_IMPLEMENTATION_PLAN.md archive/
   mv deprecated/P0_TASK_19*.md archive/
   mv deprecated/P0_TASK_20*.md archive/
   mv deprecated/P0_TASK_21*.md archive/
   mv deprecated/P1_AI_FALLBACK*.md archive/

   # Remove duplicate in deprecated
   rm deprecated/RULES_GAP_ANALYSIS_REPORT.md  # Already in archive/
   ```

2. **Update README.md:**
   - Update "Last Updated" date to November 26, 2025
   - Verify all code references are accurate

3. **Review docs/drafts/ for completion:**
   - Move completed drafts to main docs/
   - Archive obsolete drafts
   - Update draft status indicators

### Secondary Actions (Priority 2)

4. **Add status headers to all documentation:**

   ```markdown
   ---
   Status: Active | Draft | Archived | Deprecated
   Last Updated: YYYY-MM-DD
   Supersedes: [filename if applicable]
   Superseded By: [filename if applicable]
   ---
   ```

5. **Create DOCUMENTATION_INDEX.md** in root:
   - Consolidated index of all active documentation
   - Clear indication of canonical sources
   - Deprecation/supersession relationships

6. **Enhance docs/INDEX.md:**
   - Add links to archive/ and deprecated/ with explanations
   - Add "last updated" dates
   - Add brief descriptions of each document

### Long-term Actions (Priority 3)

7. **Establish documentation maintenance process:**
   - Regular quarterly reviews
   - Automated link checking
   - Codebase-documentation sync verification

8. **Set up documentation CI checks:**
   - Verify internal links
   - Check for broken cross-references
   - Validate code examples

9. **Create documentation templates:**
   - Standard headers for all doc types
   - Consistent formatting
   - Required metadata fields

## docs/drafts/ Analysis

Files in docs/drafts/ requiring review:

| File                                               | Status                | Recommendation                      |
| -------------------------------------------------- | --------------------- | ----------------------------------- |
| `LEGACY_CODE_ELIMINATION_PLAN.md`                  | Planning              | Review completion, possibly archive |
| `PHASE1_REMEDIATION_PLAN.md`                       | Historical            | Move to archive/                    |
| `PHASE3_ADAPTER_MIGRATION_AUDIT.md`                | Completed             | Move to docs/ or archive/           |
| `PHASE3_ADAPTER_MIGRATION_REPORT.md`               | Completed             | Move to docs/ or archive/           |
| `PHASE4_PYTHON_CONTRACT_TEST_REPORT.md`            | Completed             | Move to docs/ or archive/           |
| `PHASE7_AI_WEBSOCKET_RESILIENCE_AUDIT.md`          | Completed             | Move to docs/ or archive/           |
| `REMAINING_IMPLEMENTATION_TASKS.md`                | Superseded by TODO.md | Archive                             |
| `RULES_ENGINE_CONSOLIDATION_DESIGN.md`             | Reference             | Keep in drafts or move to docs/     |
| `RULES_ENGINE_R172_RINGCAP_IMPLEMENTATION_PLAN.md` | Historical            | Move to archive/                    |

## Conclusion

The documentation is generally well-organized with clear canonical sources. Main issues are:

1. **Excessive deprecated/ content** that should be archived or deleted
2. **Some date inconsistencies** requiring updates
3. **drafts/ directory** contains completed work that should be promoted or archived
4. **Root directory is optimal** at 17 files (target ~20)

Following the recommendations above will result in:

- Cleaner deprecated/ directory (down from 34 to ~0-5 files)
- Better organized historical content in archive/
- More accurate status tracking across all docs
- Improved discoverability and maintenance

## Verification Checklist

- [ ] Delete obsolete files from deprecated/
- [ ] Move historical files from deprecated/ to archive/
- [ ] Update README.md date
- [ ] Review and reorganize docs/drafts/
- [ ] Add status headers to key documents
- [ ] Verify code references in README.md
- [ ] Update docs/INDEX.md with archive/deprecated explanations
- [ ] Create DOCUMENTATION_INDEX.md in root
- [ ] Set up documentation review schedule
- [ ] Implement documentation CI checks

---

**Next Steps:** Execute Priority 1 actions, then proceed with Priority 2 and 3 as capacity allows.

## 2025-11-26 Alignment Updates

### Pass 1: Rules Engine Alignment

- Updated `CURRENT_RULES_STATE.md` to reflect the decomposed shared engine structure (helpers, aggregates, orchestration) instead of a monolithic `src/shared/engine/GameEngine.ts`, to point the change checklist link at `archive/RULES_CHANGE_CHECKLIST.md`, and to describe the actual TS/Python parity and determinism suites (RulesBackendFacade integration test, Backend_vs_Sandbox parity tests, and `ai-service/tests/parity/**`).
- Updated `RULES_SCENARIO_MATRIX.md` Movement axis rows (M1–M3) so their backend/sandbox test references match the current suite: shared movement helper tests in `tests/unit/movement.shared.test.ts`, backend movement/capture tests in `tests/unit/RuleEngine.movementCapture.test.ts`, and the RulesMatrix-driven scenarios in `tests/scenarios/RulesMatrix.Comprehensive.test.ts`, removing references to non-existent `RulesMatrix.Movement.*` and `MovementCaptureParity.*` files.

### Pass 2: Comprehensive Documentation-to-Codebase Alignment

**Total: 10 files modified across 6 verticals**

#### 1. Rules Engine Vertical

- **Status:** No changes needed - all documents already correctly aligned
- Verified: `CURRENT_RULES_STATE.md`, `RULES_IMPLEMENTATION_MAPPING.md`, `RULES_SCENARIO_MATRIX.md`, `RULES_CANONICAL_SPEC.md`, `docs/RULES_ENGINE_SURFACE_AUDIT.md`, `docs/PARITY_SEED_TRIAGE.md`, `docs/PYTHON_PARITY_REQUIREMENTS.md`

#### 2. Architecture & Engine Surface Vertical (2 files modified)

- **[`docs/MODULE_RESPONSIBILITIES.md`](docs/MODULE_RESPONSIBILITIES.md):**
  - Updated file counts from 44 to 51 files
  - Added `aggregates/` folder (6 files) to location list
- **[`docs/RULES_ENGINE_SURFACE_AUDIT.md`](docs/RULES_ENGINE_SURFACE_AUDIT.md):**
  - Updated Shared Engine from 26 to 51 files
  - Updated Client Sandbox from 12 to 18 files
  - Added 7 missing sandbox modules

#### 3. API Vertical (1 file modified)

- **[`docs/API_REFERENCE.md`](docs/API_REFERENCE.md):**
  - Added complete WebSocket API section (22 events: 8 client→server, 14 server→client)
  - Added Utility Endpoints section (GET /, POST /client-errors)
  - Added Rate Limiting error codes
  - Added AI Service error codes
  - Added session diagnostics endpoint
  - Updated Related Documentation section

#### 4. Ops & Infrastructure Vertical (1 file modified)

- **[`docs/ALERTING_THRESHOLDS.md`](docs/ALERTING_THRESHOLDS.md):**
  - Added 5 missing alerts from `monitoring/prometheus/alerts.yml`:
    - `NoHTTPTraffic`, `HighActiveHandles`, `LongRunningGames`
    - `DatabaseResponseTimeSlow`, `RedisResponseTimeSlow`
  - Added new "Service Response Time Alerts" section

#### 5. AI/ML Vertical (3 files modified)

- **[`ai-service/README.md`](ai-service/README.md):**
  - Corrected difficulty-to-AI-type mappings to match `app/main.py`
- **[`docs/AI_TRAINING_AND_DATASETS.md`](docs/AI_TRAINING_AND_DATASETS.md):**
  - Added documentation for `--gamma` CLI flag in territory dataset generator
- **[`docs/AI_TRAINING_PREPARATION_GUIDE.md`](docs/AI_TRAINING_PREPARATION_GUIDE.md):**
  - Corrected weight count from 18 to 17
  - Fixed hexagonal radius from 11 to 10

#### 6. Contributor & Testing Vertical (3 files modified)

- **[`QUICKSTART.md`](QUICKSTART.md):**
  - Fixed Grafana port from 3001 to 3002
  - Added monitoring profile requirement notes
- **[`tests/README.md`](tests/README.md):**
  - Updated directory structure with missing dirs and files
- **[`README.md`](README.md):**
  - Fixed broken `ARCHITECTURE_ASSESSMENT.md` filename
  - Corrected Docker Compose command descriptions

#### 7. Territory region semantics docs alignment

- Updated `RULES_ENGINE_ARCHITECTURE.md` Territory section so its board geometries, region definition, and disconnection criteria match the canonical RR‑CANON territory rules and the shared TS implementation (`territoryDetection.ts`, `territoryBorders.ts`, `territoryProcessing.ts`, `territoryDecisionHelpers.ts`). Backend `BoardManager.findDisconnectedRegions` and sandbox adapters are now explicitly documented as delegating to the shared helpers.

#### 8. Turn orchestrator and aggregates docs alignment

- Updated `RULES_ENGINE_ARCHITECTURE.md` shared-engine overview so its Movement, Lines, Territory, and Turn sequencing sections match the decomposed helper/aggregate/orchestrator architecture. The doc now:
  - Attributes movement/capture application and marker-path effects to `core.ts`, `movementApplication.ts`, and `captureChainHelpers.ts`, and points at `MovementAggregate` / `CaptureAggregate` as the domain aggregates.
  - Describes Lines via shared `lineDetection.ts`, canonical `lineDecisionHelpers.ts`, and the `LineAggregate` rather than legacy line mutators.
  - Notes `TerritoryAggregate` alongside `territoryDetection.ts`, `territoryBorders.ts`, `territoryProcessing.ts`, and `territoryDecisionHelpers.ts` as the canonical Territory stack.
  - Calls out the orchestration layer (`phaseStateMachine.ts`, `turnDelegateHelpers.ts`, and `turnOrchestrator.ts`) and the backend/client adapters (`TurnEngineAdapter`, `SandboxOrchestratorAdapter`) as the primary turn/phase entry points.

#### 9. Backend and sandbox adapter semantics alignment

- Clarified backend and sandbox host/adapters docs so they explicitly treat the shared orchestrator + aggregates as the single rules source of truth:
  - Updated `RULES_ENGINE_ARCHITECTURE.md` to describe server `GameEngine.ts` and `TurnEngineAdapter` as backend **hosts** over the shared orchestrator, and `ClientSandboxEngine.ts` plus `SandboxOrchestratorAdapter.ts` as sandbox **hosts** over the same surface.
  - Removed lingering references to a monolithic shared `src/shared/engine/GameEngine.ts` and to dead validator classes (`MovementValidator.ts`, `CaptureValidator.ts`, `LineValidator.ts`, `TerritoryValidator.ts`) in favour of the real helpers + aggregates + decision helpers.
  - Aligned parity/test references in architecture docs to the current suites (`Backend_vs_Sandbox.*`, `TerritoryParity.GameEngine_vs_Sandbox`, `TerritoryCore.GameEngine_vs_Sandbox`, `TraceFixtures.sharedEngineParity`, and the Python parity tests under `ai-service/tests/parity/`).
- Updated test-meta docs to match the current parity harnesses:
  - `tests/TEST_LAYERS.md` now lists the Backend_vs_Sandbox, Territory parity, and TraceFixtures shared-engine suites as the main parity layer, instead of the removed `MovementCaptureParity.*`, `PlacementParity.*`, and `VictoryParity.*` files.
  - `tests/TEST_SUITE_PARITY_PLAN.md` has been refreshed to treat `Backend_vs_Sandbox.*`, `TerritoryParity.GameEngine_vs_Sandbox`, `TerritoryCore.GameEngine_vs_Sandbox`, and `TraceFixtures.sharedEngineParity.test.ts` as the primary TS trace-level parity suites, anchored to shared-engine rules tests and contract vectors.

#### 10. Canonical engine API & decision surface tightening

@@

- Add a WebSocket-focused subsection that ties `game_state.validMoves`, `player_move` / `player_move_by_id`, and `player_choice_required` / `player_choice_response` together as thin transports over canonical `Move` options.

* +#### 11. Architecture & parity docs alignment for orchestrator + contracts
* +- Clarified remaining architecture and parity documentation so they no longer
* imply a monolithic shared `GameEngine.ts` or a fully-populated
* `validators/*` / `mutators/*` surface on the TS side, and instead treat the
* helpers + aggregates + orchestrator stack and the contract vectors as
* canonical:
* - **`AI_ARCHITECTURE.md`**: Updated the "Shared TypeScript Rules Engine" section
* to describe the orchestrator-centric model:
* - Helpers under `src/shared/engine/` (movement/capture/line/territory/victory).
* - Domain aggregates under `src/shared/engine/aggregates/`.
* - Turn orchestration via `orchestration/turnOrchestrator.ts` and
*      `phaseStateMachine.ts`.
* - Backend and sandbox host/adapters (`GameEngine.ts` + `TurnEngineAdapter`,
*      `ClientSandboxEngine.ts` + `SandboxOrchestratorAdapter`) called out
*      explicitly as thin hosts over the shared orchestrator.
* - Parity/tests section now points at shared-engine unit suites,
*      Backend_vs_Sandbox parity, territory parity, and
*      `TraceFixtures.sharedEngineParity.test.ts` rather than the removed
*      `MovementCaptureParity.*`/`VictoryParity.*` files.
* - **`docs/MODULE_RESPONSIBILITIES.md`**: Marked as **Partially historical** and
* added an upfront status note explaining that:
* - The file still lists the original design-time `validators/*` and
*      `mutators/*` modules (with a monolithic `GameEngine.ts`).
* - The implemented canonical surface is now helpers + aggregates +
*      orchestrator; only `validators/PlacementValidator.ts` and a subset of
*      mutators currently exist in TS.
* - Readers should consult `RULES_ENGINE_ARCHITECTURE.md` and this audit
*      section for the authoritative architecture.
* - **`docs/PYTHON_PARITY_REQUIREMENTS.md`**: Added a status block clarifying
* that TS `validators/*` and `mutators/*` names are **semantic anchors** for
* the Python implementation rather than a literal TS file list, and that the
* actual parity contract is expressed through:
* - Helpers + aggregates + orchestrator under `src/shared/engine/`.
* - Contract schemas and serialization under `src/shared/engine/contracts/**`.
* - TS contract tests in `tests/contracts/contractVectorRunner.test.ts` and
*      shared-engine trace tests in `tests/unit/TraceFixtures.sharedEngineParity.test.ts`.
* - Python contract tests in
*      `ai-service/tests/contracts/test_contract_vectors.py`.
* - **`RULES_SCENARIO_MATRIX.md`**: Updated the overtaking-vs-move parity row
* to reference the current TS suites:
* - `tests/unit/RuleEngine.movementCapture.test.ts` plus
*      `tests/unit/movement.shared.test.ts` and
*      `tests/unit/captureLogic.shared.test.ts` for core semantics.
* - Noted that overtaking vs move-stack behaviour is now also exercised
*      inside the multi-domain Backend_vs_Sandbox parity harnesses and
*      contract-vector–driven shared-engine tests
*      (`tests/unit/Backend_vs_Sandbox.traceParity.test.ts`,
*      `tests/unit/TraceFixtures.sharedEngineParity.test.ts`).
* +- These changes keep the rules/AI/parity docs in sync with the
* helpers + aggregates + orchestrator architecture and make it explicit where
* documents are partially historical vs authoritative.

- Tightened `docs/CANONICAL_ENGINE_API.md` around the orchestrator and decision surfaces so the docs now:
  - Describe `ProcessTurnResult`, `PendingDecision`, `DecisionType`, and `VictoryState` directly from `src/shared/engine/orchestration/types.ts`.
  - Treat `Move` (from `src/shared/types/game.ts`) as the single canonical action model, with all decision phases expressed as choosing one `Move` from orchestrator-provided options.
  - Document the explicit mapping between `DecisionType` and `PlayerChoiceType` (`line_order`, `line_reward`/`line_reward_option`, `region_order`, `elimination_target`/`ring_elimination`, `capture_direction`, `chain_capture`) and how `PlayerChoice` options carry stable `moveId` references back to canonical `Move.id`s where appropriate.
  - Add a WebSocket-focused subsection that ties `game_state.validMoves`, `player_move` / `player_move_by_id`, and `player_choice_required` / `player_choice_response` together as thin transports over canonical `Move` options.

#### Minor Issues Noted (Not Corrected)

- `docs/runbooks/INDEX.md` references `docker-compose.prod.yml` which doesn't exist (placeholder for future production config)

## 2025-11-26 AI/Python Parity & Training Docs Alignment

This vertical focused on aligning AI/Python parity and training documentation with the settled rules SSoT (shared TS engine + v2 contract vectors + CANONICAL_ENGINE_API) and with the current test/CI topology.

### Scope

Docs and files touched in this pass:

- `docs/PYTHON_PARITY_REQUIREMENTS.md`
- `AI_ARCHITECTURE.md`
- `docs/AI_TRAINING_AND_DATASETS.md`
- `docs/AI_TRAINING_PREPARATION_GUIDE.md`
- `ai-service/README.md`

### 1. Python parity requirements doc hardening (`docs/PYTHON_PARITY_REQUIREMENTS.md`)

Status before:

- Already contained a detailed TS↔Python function/type parity matrix, but:
  - Framed TS `validators/*` / `mutators/*` as if they all existed as concrete files (monolithic `GameEngine.ts` era).
  - Treated legacy trace fixtures under `tests/fixtures/rules-parity/` as the main parity vehicle.
  - CI section referenced non-existent jobs (`test-typescript-parity`, `test-python-parity`, `test-cross-language`).

Key changes:

- Added an explicit **rules SSoT + contracts** status preamble:
  - Canonical rules semantics = helpers → aggregates → orchestrator under `src/shared/engine/` + v2 contract vectors under `tests/fixtures/contract-vectors/v2/` and schemas/serialization under `src/shared/engine/contracts/**`.
  - TS `validators/*` and `mutators/*` names in the matrix are now described as **semantic anchors**, not a literal file inventory; only `validators/PlacementValidator.ts` and a subset of `mutators/*Mutator.ts` exist in TS.
- New subsection **1.3 Canonical Move Lifecycle & SSoT References**:
  - Defer all Move/decision/WebSocket lifecycle semantics to `docs/CANONICAL_ENGINE_API.md` and the shared TS type files:
    - `src/shared/types/game.ts`
    - `src/shared/engine/orchestration/types.ts`
    - `src/shared/types/websocket.ts` + `src/shared/validation/websocketSchemas.ts`.
- Test strategy section updated to make **v2 contract vectors** the primary parity mechanism:
  - TS runner: `tests/contracts/contractVectorRunner.test.ts`.
  - Python runner: `ai-service/tests/contracts/test_contract_vectors.py`.
  - Trace fixtures under `tests/fixtures/rules-parity/` are now explicitly documented as **legacy/diagnostic** inputs for seed/trace triage, tied to `docs/PARITY_SEED_TRIAGE.md` and `RULES_SCENARIO_MATRIX.md`.
- CI integration section now matches `.github/workflows/ci.yml`:
  - `test` (umbrella Jest) → includes shared-engine and contract-vector suites.
  - `ts-rules-engine` → targeted TS rules/engine layer.
  - `python-rules-parity` → fixture generation via `tests/scripts/generate_rules_parity_fixtures.ts` then `ai-service/tests/parity/test_rules_parity_fixtures.py` under pytest.
- Parity test table expanded to include:
  - `test_rules_parity_fixtures.py`, `test_ts_seed_plateau_snapshot_parity.py`, `test_ai_plateau_progress.py`, and `test_line_and_territory_scenario_parity.py` alongside `test_default_engine_equivalence.py` and `test_default_engine_flags.py`.
- Marked the property-based testing section as **aspirational** (not yet wired into CI) to avoid over-promising tooling that does not exist.

Net effect:

- The doc is now an accurate, SSoT-aligned reference for TS↔Python rules parity:
  - Rules semantics SSoT = TS helpers + aggregates + orchestrator + contracts.
  - Parity SSoT = v2 contract vectors + contract runners + parity/plateau/invariant suites.
  - Trace fixtures & seed tests are positioned correctly as secondary diagnostics.

### 2. AI architecture doc determinism & parity references (`AI_ARCHITECTURE.md`)

Status before:

- Already deferred to `docs/CANONICAL_ENGINE_API.md` for lifecycle semantics at a high level, but the RNG determinism section referenced an outdated Python determinism test (`test_determinism.py`) and did not call out the shared-engine no-randomness guards.

Key changes:

- RNG/determinism testing section updated to match current suites:
  - TS side:
    - `tests/unit/RNGDeterminism.test.ts` (historical; removed, exercised the raw `SeededRNG` implementation; coverage now subsumed by `EngineDeterminism.shared.test.ts` and `NoRandomInCoreRules.test.ts`).
    - `tests/unit/EngineDeterminism.shared.test.ts` → shared-engine determinism and turn replay.
    - `tests/unit/NoRandomInCoreRules.test.ts` → ensures no unseeded randomness in core rules/helpers/aggregates/orchestrator.
    - AI RNG parity: `Sandbox_vs_Backend.aiRngParity.test.ts`, `Sandbox_vs_Backend.aiRngFullParity.test.ts`, `GameSession.aiDeterminism.test.ts`.
  - Python side:
    - `ai-service/tests/test_engine_determinism.py` → Python rules engine determinism.
    - `ai-service/tests/test_no_random_in_rules_core.py` → no random in Python rules core.
    - Parity/plateau tests under `ai-service/tests/parity/` (e.g. `test_ts_seed_plateau_snapshot_parity.py`, `test_ai_plateau_progress.py`, `test_line_and_territory_scenario_parity.py`).
- Framed the RNG determinism contract explicitly as:
  - **same seed + same history ⇒ same sequence of AI moves** across TS backend, sandbox, and Python AI service, modulo NN/GPU nondeterminism.

Net effect:

- `AI_ARCHITECTURE.md` now correctly reflects the **determinism/no-randomness SSoT** and the actual guard suites on both the TS and Python sides.

### 3. Training & dataset docs SSoT alignment

#### 3.1 `docs/AI_TRAINING_AND_DATASETS.md`

Status before:

- Already described the territory/combined-margin dataset generator and training flows in detail, but did not:
  - Explicitly position the TS shared engine + contract vectors as the rules SSoT.
  - Tie the training code’s rules semantics back to the orchestrator + contracts parity backbone.

Key changes:

- Added a **Rules SSoT and parity safeguards** block right after the component list:
  - Canonical rules semantics = TS shared engine (`src/shared/engine/**`) + v2 contract vectors (`tests/fixtures/contract-vectors/v2/**`).
  - Python `GameEngine` + `BoardManager` + `DefaultRulesEngine` + `TerritoryMutator` are **host/adapter** implementations validated by:
    - Contract-vector tests (`tests/contracts/contractVectorRunner.test.ts`, `ai-service/tests/contracts/test_contract_vectors.py`).
    - Parity/plateau suites under `tests/unit/*Parity*` and `ai-service/tests/parity/`.
    - Mutator shadow contracts and divergence guards documented in `RULES_ENGINE_ARCHITECTURE.md` and `docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md`.
- Kept the existing description of `generate_territory_dataset.py`, `RingRiftEnv`, and the JSONL schema, but now framed as **derived from** and guarded by the above parity backbone.

Net effect:

- Training/docs clearly state that Python training flows **consume and respect** the TS rules SSoT and contract vectors instead of acting as a separate rules spec.

#### 3.2 `docs/AI_TRAINING_PREPARATION_GUIDE.md`

Status before:

- Comprehensive pre-flight checklist for training, but its status header did not explicitly call out the rules SSoT and parity doc dependencies.

Key changes:

- Replaced the ad-hoc status line with a standardized **Doc Status** block:
  - Marked as **Active (training infrastructure checklist)**.
  - Explicitly stated assumptions:
    - Rules SSoT = TS shared engine + v2 contract vectors.
    - Lifecycle semantics = `docs/CANONICAL_ENGINE_API.md`.
    - TS↔Python parity specifics = `docs/PYTHON_PARITY_REQUIREMENTS.md` + `RULES_ENGINE_ARCHITECTURE.md`.

Net effect:

- Training preparation guide is now explicitly layered on top of the canonical rules/parity SSoT instead of silently re-specifying assumptions.

### 4. AI service README alignment (`ai-service/README.md`)

Status before:

- Good description of the AI service API and difficulty ladder, but:
  - Lacked a Doc Status/SSoT preamble.
  - Did not explicitly state that the Python rules engine is a host/adapter over the TS SSoT + contract vectors.

Key changes:

- Added a standardized **Doc Status** header:
  - Marked as **Active (Python AI microservice)**.
  - Declared the SSoT layering:
    - Rules SSoT = TS shared engine (`src/shared/engine/**`) + v2 contract vectors (`tests/fixtures/contract-vectors/v2/**`).
    - Lifecycle semantics = `docs/CANONICAL_ENGINE_API.md`.
    - TS↔Python parity details = `docs/PYTHON_PARITY_REQUIREMENTS.md` + `RULES_ENGINE_ARCHITECTURE.md`.
- Left the remainder of the README intact (API endpoints, difficulty ladder, RNG seeding, integration) since it is already code-accurate and consistent with the SSoT.

Net effect:

- The AI service README is firmly anchored to the project-wide rules/parity SSoT and clearly communicates the host/adapter role of the Python rules engine.

### 5. Summary & outcomes

Across these changes, all AI/Python parity and training docs now:

- **Defer rules semantics** to the TS shared engine + v2 contract vectors, rather than re-specifying Move/decision/WebSocket flows or treating Python as a co-equal SSoT.
- **Defer lifecycle semantics** and transport types to `docs/CANONICAL_ENGINE_API.md` and the shared TS type files (`src/shared/types/game.ts`, `src/shared/engine/orchestration/types.ts`, `src/shared/types/websocket.ts`, `src/shared/validation/websocketSchemas.ts`).
- **Treat Python as a host/adapter** validated by contract vectors + parity suites + determinism/no-randomness tests:
  - TS: shared-engine tests, Backend_vs_Sandbox parity, territory parity, trace fixtures, determinism/no-random-in-core suites.
  - Python: contract runners, parity/plateau suites, determinism/no-random-in-core suites, territory/divergence guards.
- Clearly mark aspirational/diagnostic sections (property-based testing, some training infrastructure enhancements) as **future work** rather than implying completed implementation.

This closes the loop on the requested vertical: AI/Python parity and training docs are now consistently wired into the same SSoT spine as the rest of the rules/architecture documentation.

## 2025-11-26 Architecture & Engine-Topology Docs Alignment

This follow-on vertical focused on aligning the higher-level architecture/topology docs with the settled rules SSoT (shared TS engine helpers → aggregates → orchestrator → contracts) and the canonical Move/decision/WebSocket lifecycle SSoT (`docs/CANONICAL_ENGINE_API.md`).

### Scope

Docs touched in this pass:

- `RULES_ENGINE_ARCHITECTURE.md`
- `docs/RULES_ENGINE_SURFACE_AUDIT.md`
- `RULES_IMPLEMENTATION_MAPPING.md`

### 1. RULES_ENGINE_ARCHITECTURE.md – SSoT and Python host positioning

Status before:

- Already described the shared TS engine and orchestrator stack, but:
  - Lacked a standard Doc Status header tied explicitly to the helpers → aggregates → orchestrator → contracts SSoT and to `docs/CANONICAL_ENGINE_API.md`.
  - Framed the Phase 2 rollout as “Python engine becomes single source of truth for validation” without explicitly preserving the TS shared engine as rules SSoT.

Key changes:

- Added a **Doc Status (2025-11-26): Active (with historical/aspirational content)** block at the top that:
  - Declares the rules semantics SSoT as the shared TS engine under `src/shared/engine/` (helpers → aggregates → orchestrator → contracts + v2 contract vectors).
  - Declares the lifecycle SSoT as `docs/CANONICAL_ENGINE_API.md` + the shared TS/WebSocket types.
  - States explicitly that backend (`GameEngine` + `TurnEngineAdapter`), client sandbox (`ClientSandboxEngine` + `SandboxOrchestratorAdapter`), and Python rules engine (`ai-service/app/game_engine.py`, `ai-service/app/rules/*`) are **hosts/adapters** over this SSoT, not independent rules engines.
  - Marks the Python mutator-first refactor and rollout phases as **aspirational design** layered on top of the TS rules SSoT.
- Tightened the intro paragraph to say the Python architecture is a **parity-validated host over the canonical TS engine** in online validation flows, instead of an alternative SSoT.
- After the Contract Testing bullet list, added a clarifying paragraph:
  - The canonical Move/decision/WebSocket lifecycle and engine decision surfaces are documented in `docs/CANONICAL_ENGINE_API.md`.
  - This architecture doc is explicitly scoped to: how the shared TS rules engine is hosted by backend/sandbox/Python and how Python is rolled out as a parity-validated validation host.
- Adjusted the Phase 2 rollout goal from:
  - “Make the Python engine the single source of truth for validation …” to
  - “Make the Python engine the **primary online validation host over the canonical TS orchestrator + contracts**, while keeping the TS shared engine as the rules SSoT.”

Net effect:

- `RULES_ENGINE_ARCHITECTURE.md` now:
  - Is explicitly anchored to the helpers → aggregates → orchestrator → contracts rules SSoT and to `docs/CANONICAL_ENGINE_API.md` for lifecycle semantics.
  - Clearly positions Python as a parity-validated **host**, even in `RINGRIFT_RULES_MODE=python`, rather than redefining the SSoT.

### 2. RULES_ENGINE_SURFACE_AUDIT.md – Doc Status + SSoT framing

Status before:

- Already accurately described the four “surfaces” (shared engine, server game, Python AI, client sandbox) and their dependency graph.
- Did not use the standard Doc Status taxonomy, and some language still mirrored the pre-aggregate/monolithic-`GameEngine` era TS validators/mutators layout.

Key changes:

- Added a **Doc Status (2025-11-26): Active (with historical/diagnostic analysis)** header that:
  - Declares the rules semantics SSoT as the shared TS engine under `src/shared/engine/` (helpers → aggregates → orchestrator → contracts).
  - Declares the lifecycle semantics SSoT as `docs/CANONICAL_ENGINE_API.md` + shared TS/WebSocket types.
  - Treats backend, sandbox, and Python as **hosts/adapters** over the shared engine, emphasising that this audit views them as consumers, not SSoTs.
  - Flags the fully-populated TS `validators/*` / `mutators/*` tree in some diagrams as a **semantic boundary diagram** and partially historical – the implemented canonical surface is helpers + aggregates + orchestrator + contracts.
- Left the rest of the audit intact, since it already:
  - Correctly identifies the shared engine as the authoritative rules surface.
  - Accurately describes how server/sandbox delegate to the shared helpers.
  - Treats Python as a duplicated port with shadow contracts and parity risk.

Net effect:

- `docs/RULES_ENGINE_SURFACE_AUDIT.md` is now explicitly wired into the same SSoT framing as `RULES_CANONICAL_SPEC.md`, `docs/CANONICAL_ENGINE_API.md`, and `ARCHITECTURE_ASSESSMENT.md`, while preserving its diagnostic value and historical context.

### 3. RULES_IMPLEMENTATION_MAPPING.md – SSoT header & host/adapters clarification

Status before:

- Already provided a detailed RR‑CANON rules → implementation mapping and the inverse view.
- Implicitly treated the shared TS engine as canonical, but the intro didn’t explicitly state the SSoT hierarchy or host/adapter relationships.

Key changes:

- Added a **Doc Status (2025-11-26): Active** header stating that:
  - Rules/invariants semantics SSoT = `RULES_CANONICAL_SPEC.md` (RR‑CANON) + shared TS engine under `src/shared/engine/` (helpers → aggregates → orchestrator → contracts).
  - Lifecycle semantics SSoT = `docs/CANONICAL_ENGINE_API.md` + `src/shared/types/game.ts`, `src/shared/engine/orchestration/types.ts`, `src/shared/types/websocket.ts`, `src/shared/validation/websocketSchemas.ts`.
  - Backend, sandbox, and Python rules/AI engines are **hosts/adapters** over this SSoT and are validated via shared tests, contract vectors, and parity suites.
- Left the body of the mapping unchanged, since it already:
  - Correctly maps RR‑CANON rule clusters to the shared TS helpers/aggregates/orchestrator.
  - Accurately treats server, sandbox, and Python components as orchestrators and hosts that depend on the shared core.

Net effect:

- `RULES_IMPLEMENTATION_MAPPING.md` is now explicitly and visibly aligned with the project-wide SSoT framing and can be safely used as the canonical rules↔implementation index for future verification and refactoring work.

### Outcome

With these edits, the architecture/topology docs are now consistent with the SSoT decisions established earlier in this audit:

- Rules semantics SSoT = shared TS engine helpers → aggregates → orchestrator → contracts (+ v2 contract vectors).
- Lifecycle semantics SSoT = `docs/CANONICAL_ENGINE_API.md` + shared TS/WebSocket types.
- Backend, sandbox, and Python engines = **hosts/adapters** that must remain parity-validated but are not new sources of truth.

This closes the loop on the architecture/topology documentation vertical and makes it much harder for new docs to accidentally re-introduce a monolithic or multi-SSoT framing of the rules engine.

## 2025-11-27 SSoT Drift Guards & Documentation Banners

@@

- `[PASS] docs-banner-ssot`
- -This establishes a fully green SSoT drift-guard baseline. Future documentation or CI changes that:
  +This establishes a fully green SSoT drift-guard baseline. Future documentation or CI changes that:
  - Add/remove RR‑CANON rule IDs,
  - Introduce new Move/Decision/WebSocket types,
  - Change TS↔Python parity/contract wiring,
  - Add/remove CI jobs or core infra configs,
  - Or remove/alter SSoT banners on key docs,
    will cause CI to fail until the corresponding mappings/docs/tests are updated, ensuring the documentation set remains coupled to executable sources of truth.

* +## 2025-11-28 Pass 16: AI & Training Docs, Monitoring Runbooks, Link SSoT
* +This pass focused on three areas:
* +1. **Hardening docs-link-ssot and monitoring runbook coverage** so that all Prometheus `runbook_url` entries resolve to real, SSoT-aligned markdown files under `docs/runbooks/`.
  +2. **Aligning AI training documentation** (`docs/AI_TRAINING_PREPARATION_GUIDE.md` and `docs/AI_TRAINING_ASSESSMENT_FINAL.md`) with the implemented Python AI/training stack (MemoryConfig, bounded transposition table, self-play data generation, hex augmentation, training pipeline), while keeping rules/AI code and tests as the SSoT.
  +3. **Recording this work in the audit trail** and confirming that `docs-link-ssot` is enforced alongside the other SSoT checks.
* +### 1. docs-link-ssot and monitoring runbook coverage
* +**Files involved:**
  +- `scripts/ssot/docs-link-ssot-check.ts`
  +- `monitoring/prometheus/alerts.yml`
  +- `docs/runbooks/*.md`
* +**What `docs-link-ssot` enforces:**
  +- For a curated set of markdown docs (core architecture/rules/AI/training/ops docs) and selected config files:
* - All file-relative markdown links resolve to real files inside the repo.
* - External URLs and `#anchor`-only links are ignored.
* - From `docs/*`, relative links are resolved as:
* - `./FOO.md` → `docs/FOO.md`
* - `../BAR.md` → `BAR.md` at repo root.
* - Paths under `src/...` and `ai-service/...` in docs are treated as **repo-root-relative** and must land on real modules.
    +- For Prometheus alerting configuration:
* - Each `runbook_url` pointing at a GitHub doc under `docs/runbooks/*.md` must map to a real local file under `docs/runbooks/`.
* +**Runbook work in this pass:**
  +- Verified existing runbooks and added **minimal, non-canonical runbook stubs** for all `runbook_url` paths in `monitoring/prometheus/alerts.yml` that previously pointed to missing files, including (non-exhaustive list):
* - `DATABASE_DOWN.md`, `REDIS_DOWN.md`
* - `AI_SERVICE_DOWN.md`, `AI_ERRORS.md`, `AI_PERFORMANCE.md`, `AI_FALLBACK.md`
* - `SERVICE_DEGRADATION.md`, `SERVICE_OFFLINE.md`, `HIGH_ERROR_RATE.md`, `HIGH_LATENCY.md`, `NO_TRAFFIC.md`, `NO_ACTIVITY.md`
* - `GAME_PERFORMANCE.md`, `GAME_HEALTH.md`, `EVENT_LOOP_LAG.md`, `RESOURCE_LEAK.md`, `HIGH_MEMORY.md`
* - `WEBSOCKET_ISSUES.md`, `WEBSOCKET_SCALING.md`, `RATE_LIMITING.md`, `RULES_PARITY.md`, `DATABASE_PERFORMANCE.md`, `REDIS_PERFORMANCE.md`.
    +- Each stub is intentionally **non-canonical** and carries an explicit SSoT-aligned banner/role block that:
* - Marks the file as an operational guide and **derived** from code/config/monitoring SSoTs.
* - States that on conflict, code + configs + tests win.
* - Provides a short "When This Alert Fires", "Triage", "Remediation (High Level)", and "Validation" scaffold, plus a TODO section for environment-specific details.
    +- For docs-link-ssot purposes, links inside runbooks are kept conservative (prose and backticks preferred over code links) to avoid creating fragile new link targets.
* +**Outcome:**
  +- Every `runbook_url` in `monitoring/prometheus/alerts.yml` now maps to an existing `docs/runbooks/*.md` file.
  +- `docs-link-ssot` treats the mapping from alertmanager URLs → local markdown as canonical, so any future additions or renames of runbooks must update both the config and the corresponding file, or CI will fail.
* +### 2. AI training docs alignment with implemented stack
* +**Docs involved:**
  +- `docs/AI_TRAINING_PREPARATION_GUIDE.md`
  +- `docs/AI_TRAINING_ASSESSMENT_FINAL.md`
* +**Code/tests treated as SSoT for this vertical:**
  +- Memory limiting and search:
* - `ai-service/app/utils/memory_config.py` + `ai-service/tests/test_memory_config.py`
* - `ai-service/app/ai/bounded_transposition_table.py` + `ai-service/tests/test_bounded_transposition_table.py`
    +- Data generation and hex augmentation:
* - `ai-service/app/training/generate_data.py`
* - `ai-service/app/training/generate_territory_dataset.py` + `ai-service/tests/test_generate_territory_dataset_smoke.py`
* - `ai-service/app/training/hex_augmentation.py` + `ai-service/tests/test_hex_augmentation.py`, `ai-service/tests/test_hex_training.py`
    +- Training pipeline and scheduling:
* - `ai-service/app/training/train.py`, `train_loop.py`, `distributed.py`, `data_loader.py`, `model_versioning.py`, `heuristic_features.py`
* - `ai-service/tests/test_train_improvements.py`, `ai-service/tests/test_lr_schedulers.py`, `ai-service/tests/test_streaming_dataloader.py`, `ai-service/tests/test_distributed_training.py`, `ai-service/tests/integration/test_training_pipeline_e2e.py`
* +**Key doc alignment adjustments:**
  +- `docs/AI_TRAINING_PREPARATION_GUIDE.md`:
* - Already contained a detailed pre-flight checklist; this pass verified its references against the current code and tests and adjusted wording where needed so that:
* - **MemoryConfig** is described as implemented and wired via `MemoryConfig.from_env()` and used to bound training/inference allocations.
* - **BoundedTranspositionTable** is marked as implemented and integrated into `MinimaxAI`/`DescentAI` with explicit size limits derived from `MemoryConfig`.
* - **Self-play data generation** uses the `generate_data.py` CLI (`--num-games`, `--board-type`, `--seed`, `--max-moves`, `--batch-size`) exactly as implemented.
* - **Hex augmentation** is documented as using full D6 symmetry via `HexSymmetryTransform` / `augment_hex_sample` in `hex_augmentation.py` plus `augment_hex_data` in `generate_data.py`, with tests in `test_hex_augmentation.py` and `test_hex_training.py` as the canonical semantics.
* - The doc already had an SSoT alignment banner tying it to the rules SSoT and AI/training SSoT; this pass confirmed that banner still matches `docs/SSOT_BANNER_GUIDE.md` and doesn’t introduce new broken links.
    +- `docs/AI_TRAINING_ASSESSMENT_FINAL.md`:
* - Added an explicit **Doc Status / SSoT alignment** block at the top, marking it as a **derived assessment** over:
* - Rules semantics SSoT (shared TS engine + v2 contract vectors + `docs/CANONICAL_ENGINE_API.md`).
* - AI/training SSoT (Python AI and training modules + their tests).
* - Ensured that references to MemoryConfig, bounded transposition tables, hex augmentation, self-play generation, and the training pipeline match the actual code locations and test coverage. All code links are now repo-root-correct and pass `docs-link-ssot`.
* - Clarified that training recommendations and CMA-ES/NN results in this report are **interpretive** and must defer to code/tests on any conflict.
* +**Link hygiene work:**
  +- Removed or avoided any residual pseudo-links (`[1](player=1)`-style patterns) and line-suffix paths (e.g. `foo.py:212-222`) that would be mis-parsed as links by `docs-link-ssot`.
  +- Normalized all intra-repo code references to either:
* - Relative paths from `docs/` that resolve to real files, or
* - Inline code blocks (backticks) when stability of the path is not guaranteed.
* +**Outcome:**
  +- AI training docs now:
* - Defer rules semantics and lifecycle to the established SSoTs.
* - Accurately describe the implemented MemoryConfig, bounded TT, generate-data CLI, hex augmentation, and training pipeline features.
* - Carry explicit SSoT banners so future maintenance knows they are **derived** docs, not new SSoTs.
* +### 3. Documentation audit trail & remaining gaps
* +This section (Pass 16) was added to record the link/runbook/AI-training-docs vertical and its coupling to the SSoT harness.
* +**Current SSoT check status after this pass:**
  +- `[PASS] rules-semantics-ssot`
  +- `[PASS] lifecycle-api-ssot`
  +- `[PASS] python-parity-ssot`
  +- `[PASS] ci-config-ssot`
  +- `[PASS] docs-banner-ssot`
  +- `[PASS] docs-link-ssot`
* +**Known remaining gaps / future work (docs + tests):**
  +- AI training docs:
* - The existing training docs lean heavily toward infrastructure and pre-flight checklists; deeper narrative documentation of the **end-to-end AlphaZero-style loop**, **self-play curriculum**, and **profiling/observability** for long training runs is still mostly aspirational.
* - CI integration of long-running self-play/soak tests and large training/evaluation batches remains intentionally conservative to avoid overloading shared runners; future work should document and gate any expansion carefully.
    +- Monitoring runbooks:
* - Many of the new runbook stubs are intentionally high-level and contain TODO sections for environment-specific steps (e.g., where to find dashboards, playbooks for scaling out AI pods, or how to interpret AI plateau alerts). As these playbooks are exercised in real incidents, the runbooks should be incrementally enriched while preserving their SSoT alignment banners.
    +- SSoT harness coverage:
* - `docs-link-ssot` currently covers a curated subset of docs and configs. If new canonical or near-canonical docs are added in future (e.g. more detailed AI training pipeline docs, new incident retrospectives), they should be considered for inclusion in the harness to keep link hygiene enforced.
* +Overall, this pass brings **link correctness**, **monitoring runbook coverage**, and **AI training documentation** up to the same SSoT standard as rules/architecture/parity docs, while keeping executable code, schemas, and tests as the canonical sources of truth.

This pass focused on **closing remaining automated SSoT gaps** and wiring the documentation set more tightly to executable SSoTs via CI.

### 1. CI/config vs docs SSoT alignment (`ci-config-ssot`)

**Files involved:**

- `.github/workflows/ci.yml`
- `docs/SUPPLY_CHAIN_AND_CI_SECURITY.md`
- `scripts/ssot/ci-config-ssot-check.ts`

**What the check enforces:**

- Existence of core operational artefacts:
  - `docker-compose.yml`, `docker-compose.staging.yml`, `Dockerfile`
  - `monitoring/prometheus/alerts.yml`, `monitoring/prometheus/prometheus.yml`, `monitoring/alertmanager/alertmanager.yml`
- Presence of the CI workflow (`.github/workflows/ci.yml`) and the CI/security doc (`docs/SUPPLY_CHAIN_AND_CI_SECURITY.md`).
- Exact matching between **CI job display names** (under `name:` in `ci.yml`) and human-readable labels in the CI/security doc.

The enforced job names are:

- `Lint and Type Check`
- `Run Tests`
- `TS Rules Engine (rules-level)`
- `Build Application`
- `Security Scan`
- `Docker Build Test`
- `Python Rules Parity (fixture-based)`
- `Python Dependency Audit`
- `Playwright E2E Tests`

**Changes made:**

- Verified `.github/workflows/ci.yml` defines each of the expected job names exactly under `name:`.
- Updated `docs/SUPPLY_CHAIN_AND_CI_SECURITY.md` to:
  - Include an explicit **"CI job map"** subsection that lists each job by its display name (exact string) and maps it to:
    - The underlying workflow job ID (e.g. `lint-and-typecheck`, `test`, `ts-rules-engine`, `build`, `security-scan`, `docker-build`, `python-rules-parity`, `python-dependency-audit`, `e2e-tests`).
    - The associated commands (e.g. `npm run test:coverage`, `npm run test:ts-rules-engine`, `npm audit`, `pip-audit`, Docker Buildx invocation, Playwright E2E run).
  - Clarify which jobs are intended as **required gates** for `main`/`develop` (lint/typecheck, umbrella tests, TS rules-engine, Docker build, Node & Python dependency audits, Python rules parity) vs. diagnostic/observability jobs (Playwright E2E in CI, SBOM generation).
- Left `scripts/ssot/ci-config-ssot-check.ts` unchanged, treating its expectations as canonical.

**Outcome:**

- `ci-config-ssot` now passes under `npm run ssot-check`.
- Any future addition/removal/rename of CI jobs under `name:` in `ci.yml` will require a corresponding update in `docs/SUPPLY_CHAIN_AND_CI_SECURITY.md`, or CI will fail.

### 2. Documentation SSoT banners (`docs-banner-ssot`)

**Files involved:**

- `RULES_ENGINE_ARCHITECTURE.md`
- `RULES_IMPLEMENTATION_MAPPING.md`
- `docs/RULES_ENGINE_SURFACE_AUDIT.md`
- `docs/CANONICAL_ENGINE_API.md`
- `docs/SUPPLY_CHAIN_AND_CI_SECURITY.md`
- `AI_ARCHITECTURE.md`
- `docs/PYTHON_PARITY_REQUIREMENTS.md`
- `ARCHITECTURE_ASSESSMENT.md`
- `ARCHITECTURE_REMEDIATION_PLAN.md`
- `docs/MODULE_RESPONSIBILITIES.md`
- `scripts/ssot/docs-banner-ssot-check.ts`

**What the check enforces:**

- For each targeted doc, the presence of:
  - A generic **"SSoT alignment"** banner fragment, and
  - A **document-specific anchor snippet** that encodes its SSoT role, for example:
    - `"Rules/invariants semantics SSoT"` (rules semantics derived docs).
    - `"Lifecycle/API SSoT"` (canonical lifecycle doc).
    - `"Operational SSoT"` (CI/config/infra docs).
    - `"rules semantics SSoT"` (AI architecture host over rules semantics).
    - `"Canonical TS rules surface"` (Python parity doc anchored to TS surface).

The check is intentionally conservative: if either the generic banner or the required snippet is removed/edited, `npm run ssot-check` will fail, forcing an explicit decision about SSoT framing before merging.

**Changes made in this pass (building on 2025‑11‑26 work):**

- **`RULES_ENGINE_ARCHITECTURE.md`**
  - Confirmed presence of an **SSoT alignment banner** referencing the **Rules/invariants semantics SSoT** (shared TS engine under `src/shared/engine/**` + contract vectors under `tests/fixtures/contract-vectors/v2/**`) and the lifecycle/API SSoT (`docs/CANONICAL_ENGINE_API.md` + shared types/schemas).
  - Banner now explicitly states that backend, sandbox, and Python engines are **hosts/adapters** over the shared TS rules SSoT, not independent SSoTs.
- **`RULES_IMPLEMENTATION_MAPPING.md`**
  - Verified and retained an SSoT alignment banner that:
    - Anchors the mapping to the **Rules/invariants semantics SSoT** (RR‑CANON spec + shared TS engine).
    - States that any discrepancies must be resolved in favour of `RULES_CANONICAL_SPEC.md` + shared TS engine + contract vectors.
- **`docs/RULES_ENGINE_SURFACE_AUDIT.md`**
  - Ensured the doc carries an "SSoT alignment" banner referring to the Rules/invariants semantics SSoT and explicitly positioning this file as a **derived surface audit**.
  - Banner clarifies that diagrams showing a fully-populated `validators/*` / `mutators/*` tree are **partially historical**; the canonical implementation is helpers → aggregates → orchestrator → contracts.
- **`docs/CANONICAL_ENGINE_API.md`**
  - Confirmed Lifecycle/API SSoT banner is present, deferring Move/decision/WebSocket semantics to executable types (`src/shared/types/game.ts`, `src/shared/engine/orchestration/types.ts`, `src/shared/types/websocket.ts`, `src/shared/validation/websocketSchemas.ts`) and tests.
- **`docs/SUPPLY_CHAIN_AND_CI_SECURITY.md`**
  - Ensured the **Operational SSoT** banner references:
    - CI workflows (`.github/workflows/*.yml`),
    - Dockerfiles & compose stacks,
    - Monitoring configs,
    - Env/config validation code.
  - Banner states explicitly that if documentation conflicts with these artefacts, **code/config/tests win**.
- **`AI_ARCHITECTURE.md`**
  - Verified the banner includes a **"rules semantics SSoT"** snippet that:
    - Anchors the AI service and Python rules engine as **hosts over the rules semantics SSoT** (shared TS engine + contracts + contract vectors).
    - Defers lifecycle semantics to `docs/CANONICAL_ENGINE_API.md`.
- **`docs/PYTHON_PARITY_REQUIREMENTS.md`**
  - Confirmed the presence of a banner that:
    - References the **Canonical TS rules surface** and the shared TS helpers/aggregates/orchestrator/contract stack.
    - Positions the Python rules engine as a parity-validated host over that surface, with parity enforced via contract vectors and determinism/no-randomness suites.
- **`ARCHITECTURE_ASSESSMENT.md`, `ARCHITECTURE_REMEDIATION_PLAN.md`, `docs/MODULE_RESPONSIBILITIES.md`**
  - Verified each carries an "SSoT alignment" banner marking them as **derived architecture docs**:
    - They analyse or plan around architecture and module layout.
    - They explicitly defer semantics and lifecycle to the rules and API SSoTs.
    - They warn readers to prefer executable SSoTs on conflict.

**Outcome:**

- `docs-banner-ssot` now passes under `npm run ssot-check`.
- Any future removal or mutation of SSoT banners in these key docs will cause CI to fail, forcing explicit discussion rather than silent drift.

### 3. Documentation index & audit: making SSoT/banners discoverable

**Files involved:**

- `DOCUMENTATION_INDEX.md`
- `DOCUMENTATION_AUDIT_REPORT.md`

**Changes:**

- **`DOCUMENTATION_INDEX.md`**
  - Added an explicit explanation of how docs are grouped and bannered:
    - **Section 1 – Canonical Sources of Truth (SSoT):** executable / canonical artefacts with SSoT banners.
    - **Section 2 – Active Operational & Contributor Docs:** derived docs with "SSoT alignment" banners that defer to code/config/tests.
    - **Section 3 – Historical & Archived Documentation:** explicitly non-SSoT reports marked as potentially stale.
  - Documented that `npm run ssot-check` (specifically `docs-banner-ssot`) enforces the presence of these banners on key docs, preventing accidental removal of SSoT framing.
- **`DOCUMENTATION_AUDIT_REPORT.md`** (this file)
  - Added this section to record:
    - The introduction of CI-enforced SSoT banners.
    - The CI job name ↔ doc alignment for supply-chain/CI security.
    - The fact that **all SSoT checks now pass** under `npm run ssot-check`.

**Outcome:**

- The documentation map and audit report now describe **both**:
  - The semantic SSoTs (rules, lifecycle, parity, AI/training, CI/infra).
  - The **mechanical drift guards** (`rules-semantics-ssot`, `lifecycle-api-ssot`, `python-parity-ssot`, `ci-config-ssot`, `docs-banner-ssot`) that keep derived docs in sync.

### 4. Current SSoT/CI status

After these changes, `npm run ssot-check` reports:

- `[PASS] rules-semantics-ssot`
- `[PASS] lifecycle-api-ssot`
- `[PASS] python-parity-ssot`
- `[PASS] ci-config-ssot`
- `[PASS] docs-banner-ssot`

This establishes a fully green SSoT drift-guard baseline. Future documentation or CI changes that:

- Add/remove RR‑CANON rule IDs,
- Introduce new Move/Decision/WebSocket types,
- Change TS↔Python parity/contract wiring,
- Add/remove CI jobs or core infra configs,
- Or remove/alter SSoT banners on key docs,

will cause CI to fail until the corresponding mappings/docs/tests are updated, ensuring the documentation set remains coupled to executable sources of truth.

## 2025-12-01 PASS20-21 Documentation Updates

### PASS20 Completion (Orchestrator Phase 3)

**Scope:** Orchestrator migration Phase 3 completion and test suite stabilization

**Key Achievements:**

- ✅ ~1,176 lines legacy code removed
- ✅ Feature flags hardcoded/removed
- ✅ All 2,987 TypeScript tests passing
- ✅ TEST_CATEGORIES.md documentation created
- ✅ ORCHESTRATOR_MIGRATION_COMPLETION_PLAN.md created

**Documents Updated:**

- `CURRENT_STATE_ASSESSMENT.md` - Updated with PASS20 completion status
- `docs/PASS20_COMPLETION_SUMMARY.md` - Created comprehensive summary
- `docs/PASS20_ASSESSMENT.md` - Created assessment report
- `docs/TEST_CATEGORIES.md` - Created test categorization guide

**Code Changes:**

- Removed `RuleEngine` deprecated methods (~120 lines)
- Removed feature flag infrastructure (~19 lines)
- Removed `ClientSandboxEngine` legacy methods (786 lines)
- Deleted obsolete test files (193 lines)
- **Total legacy code removed:** ~1,176 lines

### PASS21 Observability Implementation

**Scope:** Observability infrastructure and load testing framework

**Key Achievements:**

- ✅ 3 Grafana dashboards created (22 panels total)
  - `monitoring/grafana/dashboards/game-performance.json` - Game metrics, AI latency, terminations
  - `monitoring/grafana/dashboards/rules-correctness.json` - Parity and correctness metrics
  - `monitoring/grafana/dashboards/system-health.json` - HTTP, WebSocket, infrastructure health
- ✅ k6 load testing framework implemented
  - Scenario P1: Mixed human vs AI ladder (40-60 players, 20-30 moves)
  - Scenario P2: AI-heavy concurrent games (60-100 players, 10-20 AI games)
  - Scenario P3: Reconnects and spectators (40-60 players + 20-40 spectators)
  - Scenario P4: Long-running AI games (10-20 games, 60+ moves)
- ✅ Monitoring stack moved from optional to default
- ✅ DOCUMENTATION_INDEX.md created
- ✅ Observability score improved: 2.5/5 → 4.5/5

**Documents Updated:**

- `CURRENT_STATE_ASSESSMENT.md` - Updated observability score, added PASS21 summary, updated component scores
- `STRATEGIC_ROADMAP.md` - Marked monitoring/load testing items complete, updated load scenarios
- `TODO.md` - Marked Wave 5 complete, added Wave 6 (observability) and Wave 7 (validation)
- `PROJECT_GOALS.md` - Updated production readiness criteria, marked environment rollout complete
- `ARCHITECTURE_ASSESSMENT.md` - Added observability to strengths, updated completion status
- `DOCUMENTATION_AUDIT_REPORT.md` - Recorded PASS20-21 changes (this section)
- `docs/PASS21_ASSESSMENT_REPORT.md` - Created comprehensive assessment

**Infrastructure Added:**

- `monitoring/grafana/dashboards/game-performance.json` (dashboard definition)
- `monitoring/grafana/dashboards/rules-correctness.json` (dashboard definition)
- `monitoring/grafana/dashboards/system-health.json` (dashboard definition)
- `monitoring/grafana/provisioning/dashboards.yml` (provisioning config)
- `monitoring/grafana/provisioning/datasources.yml` (datasource config)
- k6 load testing scenarios (4 production-scale tests)

**Test Coverage Improvements:**

- GameContext.tsx: 0% → 89.52%
- SandboxContext.tsx: 0% → 84.21%
- Overall coverage: 65.55% → ~69%

### Cross-Document Consistency Verification

**Metrics Alignment Check (PASS20-21):**

All documents now use consistent metrics:

- ✅ TypeScript tests: 2,987 passing, ~130 skipped
- ✅ Python tests: 836 passing
- ✅ Contract vectors: 49/49 (0 mismatches)
- ✅ Legacy code removed: ~1,176 lines (PASS20)
- ✅ Observability score: 4.5/5 (improved from 2.5/5)
- ✅ Overall coverage: ~69% lines (improved from 65.55%)
- ✅ Orchestrator status: Phase 3 complete, 100% rollout

**Cross-Reference Validation:**

Documents correctly reference each other:

- ✅ CURRENT_STATE_ASSESSMENT.md ← PASS20_COMPLETION_SUMMARY.md
- ✅ CURRENT_STATE_ASSESSMENT.md ← PASS21_ASSESSMENT_REPORT.md
- ✅ STRATEGIC_ROADMAP.md → PROJECT_GOALS.md (scope/success criteria)
- ✅ TODO.md → STRATEGIC_ROADMAP.md (tactical → strategic)
- ✅ PROJECT_GOALS.md ← CURRENT_STATE_ASSESSMENT.md (goals ← status)
- ✅ ARCHITECTURE_ASSESSMENT.md → CURRENT_STATE_ASSESSMENT.md (architecture → implementation)

**Key Documentation Relationships:**

```
PROJECT_GOALS.md (What we want to achieve)
     ↓
STRATEGIC_ROADMAP.md (How we plan to get there - SLOs, phases)
     ↓
TODO.md (Tactical execution - waves, tasks)
     ↓
CURRENT_STATE_ASSESSMENT.md (Where we are now - factual status)
     ↑
PASS20_COMPLETION_SUMMARY.md & PASS21_ASSESSMENT_REPORT.md (Historical achievements)
```

### Verification Checklist Updates

**PASS20-21 Items:**

- [x] Updated CURRENT_STATE_ASSESSMENT.md with PASS20-21 progress
- [x] Updated STRATEGIC_ROADMAP.md with completed items
- [x] Updated TODO.md with Wave 5 complete, Wave 6/7 added
- [x] Updated PROJECT_GOALS.md with production readiness progress
- [x] Updated ARCHITECTURE_ASSESSMENT.md with observability
- [x] Added PASS20-21 section to DOCUMENTATION_AUDIT_REPORT.md
- [x] Verified metric consistency across all documents
- [x] Validated cross-references between documents
- [x] Created PASS20-21_DOCUMENTATION_UPDATE_PLAN.md for reference

### Summary of Changes

**6 core documents updated** with PASS20-21 achievements:

1. **CURRENT_STATE_ASSESSMENT.md** - Assessment date updated to Post-PASS20-21, added PASS21 summary block, updated test coverage metrics (+GameContext/SandboxContext), updated component scores (Observability 2.5→4.5), updated current focus
2. **STRATEGIC_ROADMAP.md** - Marked observability/monitoring complete, updated legacy code count (1,147→1,176), added dashboard/load testing completion markers, updated load scenario status
3. **TODO.md** - Marked Wave 5 complete with ✅, marked all Wave 5 subtasks complete, added Wave 6 (Observability) complete, added Wave 7 (Production Validation) as next
4. **PROJECT_GOALS.md** - Updated highest-risk area analysis, updated test coverage table, marked orchestrator criteria complete, added observability criteria, updated environment rollout status
5. **ARCHITECTURE_ASSESSMENT.md** - Updated last updated date, added observability to remediation complete note, added observability to strengths, updated next steps
6. **DOCUMENTATION_AUDIT_REPORT.md** - Added this PASS20-21 section documenting all changes

**Documentation produced:**

- `PASS20-21_DOCUMENTATION_UPDATE_PLAN.md` - Comprehensive update plan with all changes detailed

**Consistency validated:**

- All documents reference same test counts (2,987 TS, 836 Python, 49 contract vectors)
- All documents show ~1,176 lines legacy code removed
- All documents show observability 4.5/5
- All documents show Phase 3 orchestrator complete
- All documents reference PASS21 observability achievements
- Cross-references validated and accurate
