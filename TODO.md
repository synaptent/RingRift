# RingRift Task Tracker

**Last Updated:** 2025-12-06
**Project Health:** GREEN
**Purpose:** Canonical task tracker for near- and mid-term work

---

## Document Role

This is the canonical high-level task/backlog tracker. For rules semantics, defer to `RULES_CANONICAL_SPEC.md`. For implementation status, see `CURRENT_STATE_ASSESSMENT.md`.

**Priority Levels:**

- **P0** – Critical for rules correctness / engine parity
- **P1** – High-value for playable, stable online games
- **P2** – Important but can follow P0/P1

---

## Executive Summary

### Completed Waves (All ✅)

| Wave      | Name                                 | Completion |
| --------- | ------------------------------------ | ---------- |
| Phase 1.5 | Architecture Remediation             | Nov 2025   |
| Wave 5    | Orchestrator Production Rollout      | Dec 2025   |
| Wave 6    | Observability & Production Readiness | Dec 2025   |
| Wave 7    | Production Validation & Scaling      | Dec 2025   |
| Wave 8    | Player Experience & UX Polish        | Dec 2025   |
| Wave 9    | AI Strength & Optimization           | Dec 2025   |
| Wave 10   | Game Records & Training Data         | Dec 2025   |
| Wave 11   | Test Hardening & Golden Replays      | Dec 2025   |
| Wave 12   | Matchmaking & Ratings                | Dec 2025   |
| Wave 13   | Multi-Player (3-4 Players)           | Dec 2025   |
| Wave 14   | Accessibility & Code Quality         | Dec 2025   |

### Current Architecture

- **TypeScript:** 6 domain aggregates, canonical orchestrator at 100%
- **Python:** 3-layer design, 836 tests passing
- **Cross-language:** 54 contract vectors with 0 mismatches
- **Phases:** 8 canonical phases including `game_over` terminal phase

---

## 1. Completed Work Summary

## 2. Documentation Audit Plan (Dec 2025)

**Slice 1 – Landing/entry docs**

- [x] README.md – date/status labels, current status section matches scripts/env; flag removal reflected.
- [x] DOCUMENTATION_INDEX.md – update last-updated date and any stale links.
- [x] docs/INDEX.md – ensure quick index matches current canonical docs and dates.
- [x] QUICKSTART.md – verify setup commands/env vars vs `package.json` scripts and `src/server/config/env.ts`.

**Slice 2 – Architecture/APIs**

- [x] docs/architecture/CANONICAL_ENGINE_API.md – orchestrator hardcoded-on posture, endpoints current.
- [x] RULES_ENGINE_ARCHITECTURE.md – remove/mark legacy rollout flags; align entrypoints with `turnOrchestrator`.
- [x] AI_ARCHITECTURE.md – ensure orchestrator posture/env flags reflect hardcoding; note current AI endpoints.

**Slice 3 – Testing/parity**

- [x] docs/testing/TEST_CATEGORIES.md – CI vs diagnostic commands match `package.json`.
- [x] docs/testing/GOLDEN_REPLAYS.md – instructions match current replay/parity harnesses.
- [x] docs/rules parity/runbook docs – update any legacy flag references; align to current contract/parity scripts.

**Slice 4 – AI training stack**

- [x] docs/ai/AI_TRAINING_AND_DATASETS.md – current generators/flags; canonical datasets.
- [x] AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md – runnable vs WIP callouts (NN demo scripts/tests).
- [x] Scripts/tests alignment – `run_nn_training_baseline.py`, `test_nn_training_baseline_demo.py`, `test_neural_net_ai_demo.py`: reflected in docs.

**Slice 5 – Ops/runbooks**

- [x] Sweep remaining ops/runbooks for legacy rollout percentage; mark historical where needed.
- [x] Ensure shadow/circuit-breaker guidance is current and consistent.

**Slice 6 – Rules canon cross-check**

- [x] Spot-check `RULES_CANONICAL_SPEC.md` vs `src/shared/engine/**` for forced elimination + explicit no-op moves; note gaps (no speculative edits).

### 1.1 Architecture Remediation (Phase 1.5) ✅

**Completed November 2025**

Created the canonical turn orchestrator architecture:

- Canonical orchestrator in `src/shared/engine/orchestration/`
- 6 domain aggregates (Placement, Movement, Capture, Line, Territory, Victory)
- Contract schemas in `src/shared/engine/contracts/`
- Backend adapter: `TurnEngineAdapter.ts` (326 lines)
- Sandbox adapter: `SandboxOrchestratorAdapter.ts` (476 lines)
- Python contract test runner with 100% parity on 54 vectors

**Documentation:**

- `src/shared/engine/orchestration/README.md`
- `archive/PHASE1_REMEDIATION_PLAN.md`
- `archive/PHASE4_PYTHON_CONTRACT_TEST_REPORT.md`

### 1.2 Orchestrator Production Rollout (Wave 5) ✅

**Completed December 2025 (Phase 3 Complete)**

- Orchestrator at 100% in all environments
- ~1,176 lines legacy code removed
- Feature flags hardcoded/removed
- Legacy paths deprecated and removed
- Circuit breaker configured (5% error threshold, 300s window)
- All validation suites green

### 1.3 Observability Infrastructure (Wave 6) ✅

**Completed December 2025**

- 3 Grafana dashboards (game-performance, rules-correctness, system-health)
- 22 panels across performance, correctness, and health metrics
- k6 load testing framework with 4 production-scale scenarios
- Monitoring stack runs by default
- `DOCUMENTATION_INDEX.md` created

### 1.4 Production Validation (Wave 7) ✅

**Completed December 2025**

Load test results:

- Game creation p95: 15ms (target <800ms) – 53x headroom
- GET /api/games/:id p95: 10.79ms (target <400ms) – 37x headroom
- WebSocket message latency p95: 2ms (target <200ms) – 100x headroom

Operational drills completed:

- Secrets rotation drill (~30s recovery)
- Backup/restore drill (11MB backup, 40K games)
- AI service outage simulation (<75s detection)

### 1.5 Player Experience (Wave 8) ✅

**Completed December 2025**

- `OnboardingModal.tsx` for first-time players
- `useFirstTimePlayer` hook for tracking onboarding state
- Scenario picker with "Learn the Basics" preset highlighting
- Decision-phase countdown with visual urgency
- Dark theme for turn number and player card panels
- `SpectatorHUD`, `EvaluationGraph`, `MoveAnalysisPanel` components
- `TeachingOverlay` component for rules/FAQ scenarios

### 1.6 AI Integration (Wave 9) ✅

**Completed December 2025**

- Service-backed PlayerChoices: line_reward_option, ring_elimination, region_order
- Difficulty ladder (1-10) with engine selection
- AI telemetry and fallback monitoring
- Training data registry and canonical parity gate
- Model versioning infrastructure

### 1.7 Game Records & Training Data (Wave 10) ✅

**Substantially Complete December 2025**

- Python and TypeScript `GameRecord` types
- `GameRecordRepository` for CRUD operations
- Self-play recording with rich metadata
- Canonical parity gate (`run_canonical_selfplay_parity_gate.py`)
- Training data registry (`TRAINING_DATA_REGISTRY.md`)
- Training dataset exporter with rank-aware values
- Self-play browser UI in sandbox

### 1.8 Test Hardening (Wave 11) ✅

**Substantially Complete December 2025**

- 29 golden game candidates across all board types
- Golden replay framework in `tests/golden/`
- Hash parity infrastructure between Python and TypeScript
- Schema v6 with `available_moves_json`, `engine_eval`, `engine_depth` columns

### 1.9 Matchmaking & Ratings (Wave 12) ✅

**Completed December 2025**

- `RatingService.ts` with ELO implementation
- Leaderboard page
- Rating integration with game completion

### 1.10 Multi-Player Support (Wave 13) ✅

**Completed December 2025**

- 3-4 player games supported
- Multiplayer evaluation pools
- Territory/line processing for >2 players

### 1.11 Accessibility (Wave 14) ✅

**Completed December 2025**

- `AccessibilityContext` for user preferences
- `useKeyboardNavigation` hook (arrow keys, Enter, Escape, Tab)
- `ScreenReaderAnnouncer` component
- `AccessibilitySettingsPanel` with colorblind modes
- High-contrast mode and reduced motion support
- Comprehensive `docs/ACCESSIBILITY.md` guide

### 1.12 Rules Engine Parity (P0.5) ✅

**Completed December 2025**

- Unified hash format between Python and TypeScript
- All 96 parity tests passing
- Phase transitions, movement, capture, line, territory all aligned
- Forced elimination parity verified
- Contract vectors expanded to 54 with swap_sides coverage

---

## 2. Remaining Work

### 2.1 Active P0 Tasks

#### Backend ↔ Sandbox Parity Maintenance

- [ ] Keep trace-parity and heuristic coverage suites green
- [ ] When parity failures appear, extract divergence and fix
- [ ] Treat rules mismatches as P0 bugs

#### WebSocket Lifecycle Polish

- [ ] Adjust `PlayerInteractionManager` for new decision-move types
- [ ] Enhanced reconnection UX messaging
- [ ] Multiplayer UX documentation improvements

#### Canonical Move/Phase Fidelity & Replay Gate

- [x] Enforce phase ↔ move whitelist in both TS and Python (`validate_canonical_move`, `phase_move_contract`) when adding phases/move types
- [ ] Run parity + canonical history gates on any new/modified DBs:
  - [x] Identify target DB(s) and location(s)
  - [x] Run `ai-service/scripts/run_parity_and_history_gate.py --db <path> [--emit-state-bundles-dir DIR]`
    - Pass: coverage*selfplay, distributed_soak_20251207/selfplay_square8*{2p,3p,4p}_local, selfplay_{square8,square19,hexagonal}_{2p,3p,4p}, distributed_soak_fetch/selfplay_square19_{2p,3p}_mbp-64gb, distributed_soak_fetch/selfplay_square8\_\_ (mbp/mac-studio), parity_test/_, selfplay_hex_mps_smoke, selfplay.db, canonical_square8.db
    - Removed (history fail): fixtures/golden_games/golden_square8_2p.db, legacy_noncanonical/canonical_square19.bad.db
  - [ ] Archive parity/history results alongside DB (or update registry entry)
- [x] Add and keep a small golden replay pack in CI that replays in TS + Python and fails on any semantic drift
  - [x] Exported new canonical golden from `canonical_square8_2p.db` -> `tests/fixtures/golden-games/golden_square8_2p_d033.json`
- [ ] Audit orchestrator/Python `game_engine.py` changes for silent transitions or forced elimination without recorded moves
- [ ] Audit checklist:
  - [ ] Confirm `turnOrchestrator` never advances phase without move (explicit `no_*` moves present)
  - [ ] Confirm Python `game_engine.py` mirrors forced_elimination and no-op phases
  - [ ] Spot-check replay logs for silent forced_elimination or skipped phases

#### Orchestrator & Territory Branch Coverage Hardening

- [ ] Add branch-coverage cases in `tests/unit/turnOrchestrator.core.branchCoverage.test.ts`:
  - [x] Forced-elimination pending decisions when blocked with stacks
  - [x] No-op phase transitions (`no_line_action_required`, `no_territory_action_required`)
  - [x] Chain → line → territory sequence coverage
- [ ] Add branch-coverage cases in `tests/unit/TerritoryAggregate.advanced.branchCoverage.test.ts`:
  - [x] Mini-region elimination / interior ring handling
  - [x] Disconnected region ordering with multiple regions
  - [x] Multi-player tie-break ladders
- [ ] Mirror high-risk TS cases in Python parity suites to keep contract expectations aligned

### 2.2 Active P1 Tasks

#### Frontend UX Polish

- [ ] Responsive board rendering for mobile devices
- [ ] Touch-optimized controls for mobile
- [ ] Simplified mobile HUD layout
- [ ] Additional contextual tooltips for game mechanics
- [ ] Key moments replay

#### Rules UX Iteration 0002 (Hotspot-driven)

- [x] Draft `docs/ux/rules_iterations/UX_RULES_IMPROVEMENT_ITERATION_0002.md` with telemetry baselines/targets for ANM/FE loops, structural stalemate, and mini-regions
- [x] Implement HUD/VictoryModal copy + prompts aligned to the iteration targets (FE/ANM messaging, stalemate clarity)
- [x] Update sandbox/TeachingOverlay flows for the selected contexts and add focused unit tests per surface

#### Structured End-Explanation Model

- [x] Author `docs/UX_RULES_EXPLANATION_MODEL_SPEC.md` defining the payload (outcome type, reason code, rules citations, teaching tags, telemetry)
- [x] Implement mapping from engine outcomes + `weirdStateReasons` to the new payload in the shared layer with unit tests
- [x] Wire `GameHUD` and `VictoryModal` to render the structured explanation and emit matching telemetry

#### Rules Concepts Index & Teaching Coverage

- [x] Create `docs/UX_RULES_CONCEPTS_INDEX.md` linking rules sections ↔ UX surfaces ↔ telemetry labels for ANM/FE, structural stalemate, mini-regions, capture chains, and LPS
- [x] Cross-link the index from `docs/UX_RULES_TEACHING_SCENARIOS.md` and `docs/UX_RULES_WEIRD_STATES_SPEC.md`
- [x] Audit high-risk teaching scenarios and add/flag gaps in `src/shared/teaching/teachingScenarios.ts` and `src/client/components/TeachingOverlay.tsx`

#### Spectator/Replay Polish

- [ ] Enhance spectator HUD (reconnection state, watcher counts, phase/choice banners) and validate via integration tests
- [ ] Ensure move history lists all phase moves, including `no_*` and `forced_elimination`, and supports export/share
- [ ] Add fast-forward/rewind affordances and keyboard bindings for replay; prefetch upcoming turns for smooth playback

#### E2E Test Coverage

- [ ] Full game flow replay tests
- [ ] Complex multiplayer scenarios
- [ ] Visual regression baseline establishment

#### AI Production Wiring

- [x] Wire MinimaxAI for medium-high difficulty levels
- [x] Expose MCTS in production behind AIProfile
- [x] Complete service-backing for line_order and capture_direction choices
- [x] Fix RNG determinism (seeded RNG per game)

#### Client Component Tests

- [x] `LobbyPage.tsx` unit tests
- [x] Remaining HUD/log component tests (GameHUD, VictoryModal, ReplayPanel, GameEventLog, MoveHistory, BoardView)

### 2.3 Active P2 Tasks

#### AI Improvement

- [ ] Weight sensitivity analysis on all board types
- [ ] CMA-ES optimization on pruned weight set
- [ ] Board-type specific profiles
- [ ] Opening book generation
- [ ] Transposition table for position caching

#### Game Records

- [x] JSONL export format for training data
- [x] Algebraic notation (RRN) generator/parser
- [ ] State reconstruction checkpoint caching
- [ ] Position search/filter functionality

#### Replay System

- [x] Checkpoint caching for efficient backward navigation (via `useReplayPlayback` prefetch)
- [x] Forward/backward navigation controls (`ReplayPanel/PlaybackControls.tsx` with keyboard shortcuts)
- [x] Game loading from database/file (`SelfPlayBrowser` + `GameRecordRepository`)

#### Persistence & Stats

- [ ] Game lifecycle transitions in database
- [ ] Move history/replay panel in GamePage

#### AI Ladder Calibration & Eval

- [ ] Run tiered training/promotion pipeline (`ai-service/scripts/run_tier_training_pipeline.py`, `ai-service/scripts/run_full_tier_gating.py`) and record outcomes in `ai-service/config/tier_candidate_registry.square8_2p.json`
- [ ] Refresh eval pools (`ai-service/app/training/eval_pools.py`) and rerun difficulty calibration (`ai-service/scripts/analyze_difficulty_calibration.py`) against `docs/ai/AI_TIER_PERF_BUDGETS.md`
- [ ] Keep large-board search parity tests (`ai-service/tests/test_search_board_parity.py`) and `ai-service/scripts/benchmark_search_board_large_board.py` perf budgets aligned; document deviations in results logs

### 2.4 Deferred Work

#### Phase 4 (Post-MVP)

- [ ] Tier 2 sandbox cleanup (~1,200 lines remaining)
- [ ] Add SSOT banners to sandbox modules
- [ ] Archive diagnostic-only modules

#### Security & Operations

- [ ] Security hardening review
- [ ] Scale testing at production volumes
- [ ] Backup/recovery drill institutionalization

---

## 3. Pending Technical Tasks

### 3.1 Engine Work

| Task                                 | Priority | Status      |
| ------------------------------------ | -------- | ----------- |
| Keep S-invariant tests passing       | P0       | Ongoing     |
| Unified Move model for all decisions | P0       | ✅ Complete |
| Python AI service parity             | P0       | ✅ Complete |
| Orchestrator production hardening    | P0       | ✅ Complete |

### 3.2 Testing Work

| Task                      | Priority | Status      |
| ------------------------- | -------- | ----------- |
| Rules/FAQ scenario matrix | P0       | ✅ Complete |
| Contract vector expansion | P1       | 54 vectors  |
| Golden replay framework   | P1       | ✅ Complete |
| E2E multiplayer coverage  | P1       | In Progress |

### 3.3 Frontend Work

| Task                  | Priority | Status      |
| --------------------- | -------- | ----------- |
| HUD and GameHost UX   | P1       | ✅ Complete |
| Spectator experience  | P1       | ✅ Complete |
| Mobile responsiveness | P2       | Pending     |
| Analysis mode UI      | P2       | ✅ Complete |

### 3.4 AI Work

| Task                          | Priority | Status      |
| ----------------------------- | -------- | ----------- |
| AI telemetry                  | P1       | ✅ Complete |
| Difficulty ladder             | P1       | ✅ Complete |
| Heuristic weight optimization | P2       | In Progress |
| MinimaxAI production wiring   | P2       | ✅ Complete |

---

## 4. Documentation Tasks

### 4.1 Keep Updated

- [ ] `CURRENT_STATE_ASSESSMENT.md` – Implementation status
- [ ] `STRATEGIC_ROADMAP.md` – Phased roadmap
- [ ] `KNOWN_ISSUES.md` – P0/P1 issues
- [ ] `tests/README.md` – Test categories
- [ ] `docs/UX_RULES_EXPLANATION_MODEL_SPEC.md` – Structured end-explanation model
- [ ] `docs/UX_RULES_CONCEPTS_INDEX.md` – Rules concepts index and cross-links
- [ ] `docs/ux/rules_iterations/UX_RULES_IMPROVEMENT_ITERATION_0002.md` – Telemetry-driven rules UX iteration spec
- [ ] `ai-service/TRAINING_DATA_REGISTRY.md` – Canonical vs legacy DB status after new gates

### 4.2 CI/CD

- [ ] Tighten Jest coverage thresholds once stable
- [ ] Promote parity and integration lanes to required checks
- [ ] Document security/dependency audit status

---

## 5. Reference: Historical Wave Details

Detailed historical information for completed waves is preserved in:

- `docs/archive/assessments/PASS18_ASSESSMENT.md` – Host parity, RNG, decision lifecycle
- `docs/archive/assessments/PASS19_ASSESSMENT.md` – E2E infrastructure, fixtures
- `docs/archive/assessments/PASS20_COMPLETION_SUMMARY.md` – Orchestrator Phase 3
- `docs/archive/assessments/PASS21_ASSESSMENT_REPORT.md` – Observability infrastructure
- `docs/testing/LOAD_TEST_BASELINE_REPORT.md` – Load test results
- `docs/testing/GO_NO_GO_CHECKLIST.md` – Production readiness checklist

---

## Related Documentation

| Document                      | Purpose                        |
| ----------------------------- | ------------------------------ |
| `CURRENT_STATE_ASSESSMENT.md` | Implementation status snapshot |
| `STRATEGIC_ROADMAP.md`        | Phased roadmap to MVP          |
| `KNOWN_ISSUES.md`             | P0/P1 issues and gaps          |
| `RULES_CANONICAL_SPEC.md`     | Rules semantics SSoT           |
| `docs/TEST_CATEGORIES.md`     | Test organization guide        |
| `DOCUMENTATION_INDEX.md`      | Full docs map                  |
