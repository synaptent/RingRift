# RingRift Comprehensive Improvement Plan

**Created:** December 3, 2025
**Last Updated:** December 5, 2025 (Engine Architecture Review)
**Based on:** Full project review including TODO.md, KNOWN_ISSUES.md, STRATEGIC_ROADMAP.md, CURRENT_STATE_ASSESSMENT.md, PROJECT_GOALS.md

> **Dec 5, 2025 Engine Review Summary:** Comprehensive architecture review confirms both TypeScript (A-) and Python (A) engines demonstrate excellent separation of concerns, strong canonical rules adherence, and mature parity testing. 54 contract vectors with 0 mismatches. Orchestrator at 100% rollout. See [`CURRENT_STATE_ASSESSMENT.md`](CURRENT_STATE_ASSESSMENT.md#-engine-architecture--refactoring-status-dec-2025-review) for detailed findings.

---

## Executive Summary

RingRift is a sophisticated turn-based multiplayer strategy game currently in **stable beta** with a consolidated architecture. The project demonstrates strong engineering fundamentals:

- **Orchestrator at 100% rollout** across all environments
- **2,987 TypeScript tests** and **836 Python tests** passing
- **54 contract vectors** ensuring cross-language parity (0 mismatches)
- **3 Grafana dashboards** and **k6 load testing framework** implemented
- Comprehensive documentation ecosystem

The project is approaching production readiness but has key gaps that need addressing.

### Wave Completion Summary

| Wave    | Name                                 | Status                                                             |
| ------- | ------------------------------------ | ------------------------------------------------------------------ |
| Wave 5  | Orchestrator Production Rollout      | ✅ COMPLETE                                                        |
| Wave 6  | Observability & Production Readiness | ✅ COMPLETE                                                        |
| Wave 7  | Production Validation & Scaling      | ✅ COMPLETE                                                        |
| Wave 8  | Player Experience & UX Polish        | ✅ COMPLETE                                                        |
| Wave 9  | AI Strength & Optimization           | ✅ COMPLETE                                                        |
| Wave 10 | Game Records & Training Data         | ✅ COMPLETE                                                        |
| Wave 11 | Test Hardening & Golden Replays      | ✅ SUBSTANTIALLY COMPLETE (29 game candidates)                     |
| Wave 12 | Matchmaking & Ratings                | ✅ COMPLETE                                                        |
| Wave 13 | Multi-Player (3-4 Players)           | ✅ COMPLETE                                                        |
| Wave 14 | Accessibility & Code Quality         | ✅ COMPLETE (accessibility ✅, settings modal ✅, queue wiring ✅) |

---

## Current Status Summary

### Strengths

| Area          | Status            | Evidence                                                     |
| ------------- | ----------------- | ------------------------------------------------------------ |
| Rules Engine  | Excellent (4.7/5) | Shared TS orchestrator, 6 domain aggregates, 100% rollout    |
| Test Coverage | Good (4.0/5)      | 2,987 TS + 836 Python tests, 54 contract vectors             |
| Observability | Good (4.5/5)      | Grafana dashboards, k6 framework, Prometheus metrics         |
| Architecture  | Excellent         | Clean separation: shared engine → host adapters → transports |
| Documentation | Good (4.0/5)      | Comprehensive docs, DOCUMENTATION_INDEX.md                   |

### Gaps

| Area                  | Status         | Impact                                                  |
| --------------------- | -------------- | ------------------------------------------------------- |
| Frontend UX           | Medium (3.5/5) | Developer-oriented, needs player polish                 |
| Production Validation | ✅ Complete    | All load tests passing, go/no-go approved               |
| Advanced AI           | Limited        | Primarily heuristic-based, no deep search in production |
| Social Features       | Basic          | No matchmaking, limited spectator UX                    |

---

## Wave 7: Production Validation & Scaling ✅ COMPLETE (Dec 3, 2025)

**Goal:** Validate system performance at production scale and establish operational baselines.

**Why First:** The k6 load testing framework is implemented but hasn't been run at scale. This blocks confident production deployment.

**Status:** ✅ ALL TASKS COMPLETE

**Key Deliverables:**

- [`docs/LOAD_TEST_BASELINE_REPORT.md`](docs/LOAD_TEST_BASELINE_REPORT.md) - Comprehensive load test results and lessons learned
- [`docs/GO_NO_GO_CHECKLIST.md`](docs/GO_NO_GO_CHECKLIST.md) - Production readiness checklist
- [`monitoring/alertmanager/alertmanager.local.yml`](monitoring/alertmanager/alertmanager.local.yml) - Local development alertmanager config

### 7.1 Load Test Execution ✅ COMPLETE (Dec 3, 2025)

| Task            | Description                       | SLO Target         | Result       | Status  |
| --------------- | --------------------------------- | ------------------ | ------------ | ------- |
| Run Scenario P1 | Player moves polling              | p95 latency ≤300ms | p95=11.2ms   | ✅ PASS |
| Run Scenario P2 | Concurrent games (10 VUs)         | p95 ≤400ms         | p95=10.8ms   | ✅ PASS |
| Run Scenario P3 | WebSocket stress (50 connections) | 95%+ success       | 100% success | ✅ PASS |
| Run Scenario P4 | Game creation                     | p95 ≤800ms         | p95=15ms     | ✅ PASS |

**Issue Fixed:** Rebuilt Docker container with updated `GameIdParamSchema` that accepts both UUID and CUID formats.

**Report:** [`docs/LOAD_TEST_BASELINE_REPORT.md`](docs/LOAD_TEST_BASELINE_REPORT.md)

### 7.2 Baseline Metrics Establishment ✅ COMPLETE (Dec 3, 2025)

- [x] Capture "healthy system" metric ranges from local Docker runs
- [x] Document p50/p95/p99 latencies for all critical paths
- [x] Establish capacity model (games per instance, concurrent players)
- [x] Tune alert thresholds based on observed behavior

**Key Baselines:**
| Metric | Baseline | SLO | Headroom |
|--------|----------|-----|----------|
| Game creation p95 | 15ms | 800ms | 53x |
| GET /api/games/:id p95 | 10.8ms | 400ms | 37x |
| WebSocket latency p95 | 2ms | 200ms | 100x |
| Error rate | 0% | <1% | N/A |

### 7.3 k6 Scenario Protocol Alignment ✅ COMPLETE

| Issue              | Problem                                                        | Fix                                                   | Status  |
| ------------------ | -------------------------------------------------------------- | ----------------------------------------------------- | ------- |
| Game ID contracts  | k6 assumes game IDs remain valid longer than backend lifecycle | Aligned k6 scripts with actual expiry semantics       | ✅ DONE |
| WebSocket protocol | Message format doesn't match Socket.IO client                  | Implemented Socket.IO v4 / Engine.IO v4 wire protocol | ✅ DONE |

### 7.4 Operational Drills ✅ COMPLETE (Dec 3, 2025)

- [x] Execute secrets rotation drill – Token invalidation verified, ~30s recovery
- [x] Execute backup/restore drill – 11MB backup, full integrity verified (40K games)
- [x] Simulate incident response scenarios – AI service outage, detection <75s
- [x] Document lessons learned → Added to [`docs/LOAD_TEST_BASELINE_REPORT.md`](docs/LOAD_TEST_BASELINE_REPORT.md)

### 7.5 Go/No-Go Validation ✅ COMPLETE (Dec 3, 2025)

- [x] Validate monitoring infrastructure (Prometheus, Grafana, Alertmanager)
- [x] Execute go/no-go checklist → [`docs/GO_NO_GO_CHECKLIST.md`](docs/GO_NO_GO_CHECKLIST.md)
- [x] **Verdict: ✅ GO (with caveats)** - System ready for soft launch

**Lessons Learned:**

- Docker Compose doesn't auto-reload .env changes (must export vars)
- Nginx restart needed after app container recreation
- Prometheus scrape interval (15s) determines detection speed
- Alertmanager needs production notification channels before full launch

---

## Priority 2: Frontend UX Polish (Phase 3)

**Goal:** Transform developer-oriented UI into player-friendly experience.

### 2.1 HUD & Game Host UX

- [ ] Ensure same HUD behavior between backend and sandbox games
- [ ] Add visual hierarchy tuned for first-time players (not just developers)
- [x] Improve phase-specific prompts and invalid-move feedback (Dec 4, 2025)
  - BackendGameHost: movement/capture clicks now use `useInvalidMoveFeedback` + `analyzeInvalidMove` to provide cell-level shake animations and contextual toasts for illegal sources/targets (including chain_capture continuation), instead of silently ignoring them.
  - SandboxGameHost/useSandboxInteractions: mirrored the same invalid-move behaviour for local games using `ClientSandboxEngine.getValidMoves`, so sandbox and backend now share consistent phase-specific prompts and invalid-move UX.
- [x] Enhanced decision-phase and timeout banners (Dec 3, 2025)
  - ChoiceDialog has severity-based colors (emerald→amber→red), pulsing animation on critical
  - GameHUD DecisionPhaseBanner has full urgency styling with data-severity attributes
  - Server-capped countdown styling and "Server deadline" label implemented
  - Tests cover normal (>10s), warning (3-10s), critical (≤3s) thresholds

### 2.2 Sandbox Experience

- [x] Guided onboarding and beginner-friendly presets (Dec 3, 2025)
  - Added "Learn the Basics" preset with "Recommended" badge
  - Added tooltips with learning-oriented descriptions for all presets
- [x] Visual debugging aids (Dec 3, 2025):
  - [x] Overlays for detected lines and rewards (toggleable)
  - [x] Territory region highlighting (toggleable)
  - [x] Chain capture path visualization (arrows/highlighted segments) (Dec 3, 2025)
    - BoardView prop `chainCapturePath` renders SVG overlay with arrows and pulsing current position
    - SandboxGameHost extracts path from moveHistory during chain_capture phase
- [ ] Simplify advanced tooling exposure (balance diagnostics vs learning surface)

### 2.3 Spectator & Replay

- [ ] Enhanced spectator-focused affordances
- [ ] Post-game summaries and teaching overlays
- [ ] Integration with evaluation panels for analysis mode

### 2.4 Client Component Tests (P1.4 from TODO)

- [x] LobbyPage.tsx – lobby filters, game list, navigation wiring (Dec 3, 2025)
  - 18 tests already exist in tests/unit/client/LobbyPage.test.tsx covering filters, navigation, create game
- [ ] Additional targeted unit tests for key pages

---

## Priority 3: Multiplayer Polish (Phase 3)

**Goal:** Seamless online multiplayer experience.

### 3.1 WebSocket Lifecycle & Reconnection

- [x] Tighten reconnection flows in WebSocketServer + WebSocketInteractionHandler (Dec 3, 2025)
  - WebSocketServer: `handleJoinGame` always sends a fresh `game_state` snapshot (including `validMoves`) on reconnect for both players and spectators.
  - GameContext: treats each `game_state` payload as authoritative and clears stale `pendingChoice` / `choiceDeadline` / timeout warnings so HUD decision banners never survive a reconnect.
  - GameConnection: Socket.IO reconnect handlers (`reconnect_attempt`, `reconnect`, `request_reconnect`) now drive a clean `reconnecting → connected` cycle with explicit join re-emits.
- [x] Add focused Jest integration tests for lifecycle aspects (Dec 3, 2025)
  - `tests/unit/GameSession.reconnectFlow.test.ts` – asserts reconnection window behaviour and that reconnects call `getGameState` and emit `game_state` snapshots.
  - `tests/unit/contexts/GameContext.test.tsx` – verifies fresh `game_state` snapshots clear stale pending choices and deadlines (simulated reconnect), matching HUD expectations.
  - `tests/unit/hooks/useGameConnection.test.tsx` – covers `reconnecting` labels/colouring and connection health derivation used by BackendGameHost/GameHUD.
- [x] Richer HUD signaling for "reconnecting" vs "abandoned" states (Dec 3, 2025)
  - GameContext: `disconnectedOpponents` state, `gameEndedByAbandonment` derived value
  - GameConnection: Socket.IO listeners for `player_disconnected` and `player_reconnected`
  - BackendGameHost: Orange "opponent disconnected" banner, red "abandonment" banner
  - useGameConnection hook exposes disconnection state

### 3.2 Lobby & Matchmaking

- [ ] Improve lobby UX with clearer status, filters, navigation
- [ ] Automated matchmaking queue (future)
- [ ] ELO-based matching (future)

### 3.3 Social Features

- [ ] In-game chat persistence
- [ ] User profiles and stats
- [x] Leaderboards (see Wave 12.3 - complete Dec 4, 2025)

---

## Priority 4: AI Improvements (Phase 4)

**Goal:** Challenging opponents for all skill levels.

### 4.1 Production AI Ladder Wiring

- [ ] Wire up MinimaxAI for mid-high difficulties (currently falling back to HeuristicAI)
- [ ] Expose MCTS/advanced AI implementations in production ladder
- [ ] Service-back remaining choices (line_order, capture_direction)

### 4.2 RNG Determinism Fix

- [ ] Replace global `random` usage in Python AI with per-game seeded RNG
- [ ] Update `ZobristHash` to use stable, seeded RNG
- [ ] Pass RNG seeds from TS backend to Python service

### 4.3 Weight Optimization (Track 7 from TODO)

- [ ] Complete weight sensitivity analysis on all board types
- [ ] Classify weights by signal strength
- [ ] Run CMA-ES optimization on pruned weight set
- [ ] Validate optimized weights via tournament

### 4.4 Analysis Mode (P2.4 from TODO)

- [ ] Store evaluation snapshots per move
- [ ] Extend gameViewModels with evaluationHistory
- [ ] Gate analysis panel behind spectator-only toggle

---

## Priority 5: Test Coverage & Quality

**Goal:** Reach 80% coverage target and solidify parity.

### 5.1 Coverage Improvements

| Current           | Target          | Action                                    |
| ----------------- | --------------- | ----------------------------------------- |
| ~69% lines        | 80% lines       | Focus on GameContext, SandboxContext gaps |
| 170 skipped tests | Review & reduce | Audit intentionally skipped tests         |

### 5.2 Parity Hardening

- [x] Run parity harnesses regularly
  - TS↔Python contract vectors and trace fixtures are integrated via
    `tests/parity/test_rules_parity_fixtures.py` (state+action and
    trace-based parity).
  - Replay-level parity is exercised via
    `tests/parity/test_differential_replay.py`, which compares Python
    `game_history_entries` against the TS `selfplay-db-ts-replay.ts`
    harness.
- [x] For each failure, extract first divergence and promote to focused unit test
  - Chain-capture and advanced-phase issues have dedicated tests:
    - `tests/parity/test_chain_capture_parity.py`
    - `tests/parity/test_chain_capture_phase_fix.py`
    - `tests/rules/test_default_engine_equivalence.py` (env-based
      continuation cases).
  - Territory/line-processing and plateau scenarios are covered by:
    - `tests/parity/test_line_and_territory_scenario_parity.py`
    - `tests/parity/test_ts_seed_plateau_snapshot_parity.py`.
- [x] Define "must-cover" scenario sets per rules axis
  - Documented in `docs/ENGINE_TOOLING_PARITY_RESEARCH_PLAN.md` and
    encoded as:
    - Contract vectors (placement, movement, capture/chain_capture,
      forced_elimination, territory, swap_sides).
    - TS-generated parity fixtures and trace files.
    - Self-play soaks with invariant sampling (`run_self_play_soak.py`)
      and DB-backed replay parity.

### 5.3 Contract Vector Expansion

- [ ] Add vectors as edge cases discovered in production play
- [ ] Cover any remaining hexagonal geometry edge cases

---

## Priority 6: Game Record System (Phase 5)

**Goal:** Comprehensive game storage, notation, and replay.

### 6.1 Core Implementation

- [x] Create Python `GameRecord` types (`ai-service/app/models/game_record.py`)
- [x] Create TypeScript `GameRecord` types (`src/shared/types/gameRecord.ts`)
- [ ] Implement JSONL export format for training data
- [ ] Implement algebraic notation generator/parser

### 6.2 Replay System

- [x] Implement `reconstructStateAtMove(gameRecord, moveIndex)` (`src/shared/engine/replayHelpers.ts`)
- [ ] Add checkpoint caching for efficient navigation
- [x] Create `ReplayControls` UI component (`src/client/components/ReplayPanel.tsx`)
- [x] Integrate replay into sandbox page (via `SandboxGameHost.tsx`)

### 6.3 Self-Play Game Recording (Track 11)

- [ ] Add default-enabled game recording to CMA-ES optimization
- [x] Create state pool export utility (`ai-service/scripts/export_state_pool.py`)
- [ ] Wire into iterative pipeline

---

## Priority 7: Security & Operational Readiness

**Goal:** Production-grade security and operational procedures.

### 7.1 Security Hardening

- [ ] Execute secrets rotation drill
- [ ] Execute backup/restore drill
- [ ] Simulate incident response scenarios
- [ ] Document lessons learned

### 7.2 CI & Dependency Maintenance

- [ ] Perform broader Python dependency modernization
- [ ] Tighten Jest coverage thresholds
- [ ] Promote TS Parity and TS Integration lanes to required checks

---

## Wave 8: Player Experience & UX Polish ✅ COMPLETE (Dec 4, 2025)

**Goal:** Transform the developer-oriented UI into a player-friendly experience suitable for public release.

**Rationale:** Current UX is optimized for developers and playtesters. First-time players need clearer visual hierarchy, better onboarding, and more intuitive controls.

**Status:** ✅ ALL TASKS COMPLETE

### 8.1 First-Time Player Experience ✅ COMPLETE (Dec 4, 2025)

- [x] Create onboarding modal for first-time players
  - Multi-step introduction (Welcome, Phases, Victory Conditions, Ready to Play)
  - Keyboard navigation support (arrow keys, Enter, Escape)
  - Shows on first visit to sandbox
- [x] Create `useFirstTimePlayer` hook for tracking onboarding state
  - Persists state to localStorage
  - Tracks: welcome seen, first game completed, games played
- [x] Enhance "Learn the Basics" preset visibility for first-time players
  - Pulsing animation and ring highlight
  - "Start Here" badge replaces "Recommended" for new players
  - "👇 Start here" indicator in Quick Start section header
- [x] Simplify sandbox presets – hide advanced options behind "Show Advanced" toggle
  - Advanced sections (Scenarios, Self-Play Games, Manual Config) collapsed by default for first-time players
  - Toggle expands/collapses with rotation animation
  - Returning players see expanded by default
- [x] Add contextual tooltips explaining game mechanics during play
  - HUD: Turn vs Move tooltip in Game Progress panel clarifying turn cycles vs individual moves.
  - HUD: Phase indicator tooltip summarising the current phase description plus role-specific guidance (spectator vs active player).
  - HUD: VictoryConditionsPanel tooltips for Elimination, Territory, and Last Player Standing, aligned with rules docs and victory logic.
- [x] Redesign HUD visual hierarchy for clarity (Dec 4, 2025)
  - Adapter-based GameHUD: primary HUD band (phase, whose turn, time-pressure chip, victory tooltips) now lives at the top of the sidebar in BackendGameHost, visually colocated with the board so first-time players see core state without scrolling.
  - SandboxGameHost: retains its local header card and HUD band directly above the board; both backend and sandbox hosts now follow the same general pattern of "board on the left, HUD band and supporting panels on the right" for desktop, stacking vertically on small screens via existing responsive classes.

### 8.2 Game Flow Polish

- [x] Improve phase-specific prompts with clearer action buttons (Dec 3, 2025)
  - Added `actionHint` and `spectatorHint` to PhaseViewModel interface
  - PHASE_INFO now includes role-specific contextual hints for all 6 game phases
  - PhaseIndicator displays action hints when it's the player's turn, spectator hints for observers
  - GameHUDFromViewModel derives `isMyTurn` from players array and passes to PhaseIndicator
- [x] Enhanced invalid-move feedback (subtle animations, explanatory toasts) (Dec 4, 2025)
  - `useInvalidMoveFeedback` hook centralises invalid-move analysis, toasts, and cell-shake animations.
  - BackendGameHost and SandboxGameHost both pass `shakingCellKey` through to `BoardView`, so invalid clicks on empty cells, opponent stacks, out-of-range targets, or illegal chain-capture landings produce consistent visual and copy feedback across backend and sandbox games.
  - Covered by `tests/unit/hooks/useInvalidMoveFeedback.test.tsx`, which verifies shake timing, toast wiring, and reason analysis for key scenarios.
- [x] Decision-phase countdown with visual urgency (color changes, pulsing) ✅ Already implemented in 2.1
- [x] HUD time-pressure cues and per-player clock severity styling (Dec 4, 2025)
  - GameHUDFromViewModel surfaces a compact ⏱ time-pressure chip tied to the active decision phase, with severity derived from getCountdownSeverity and role-aware copy ("Your decision timer" vs "Time left for Alice's decision").
  - PlayerCardFromVM uses the same severity thresholds to emphasise low time on active player clocks (amber for warning, pulsing red for critical) while keeping the legacy HUD timers unchanged.
- [x] Victory/defeat screens with game summary (Dec 3, 2025)
  - VictoryModal has trophy animations, confetti, final stats table
  - GameSummary shows board type, total turns, player count, rated status
  - Per-player stats: rings on board, captured, territory, total moves
  - Win/lose/draw states with color-coded messaging
  - Note: "Key moments" feature deferred for future enhancement

### 8.3 UI/UX Theme Polish ✅ COMPLETE (Dec 4, 2025)

- [x] Dark theme for turn number panel (`GameProgress` component)
- [x] Dark theme for player card panels (`PlayerCardFromVM`, `RingStatsFromVM`)
- [x] Extract `VictoryConditionsPanel` component for flexible placement
- [x] Move Victory panel below game info panel (sandbox layout)
- [x] Fix MoveHistory scroll to stay within container (prevent page scroll)

### 8.4 Spectator & Analysis Experience ✅ COMPLETE (Dec 4, 2025)

- [x] `SpectatorHUD` component (`src/client/components/SpectatorHUD.tsx`)
  - Dedicated spectator layout with player standings
  - Current player indicator with phase hints
  - Collapsible analysis section with recent moves
- [x] `EvaluationGraph` component (`src/client/components/EvaluationGraph.tsx`)
  - SVG timeline showing per-player evaluation over moves
  - Click-to-jump for move navigation
  - Current move indicator line
- [x] `MoveAnalysisPanel` component (`src/client/components/MoveAnalysisPanel.tsx`)
  - Move quality badges (excellent/good/neutral/inaccuracy/mistake/blunder)
  - Evaluation delta calculation and display
  - Think time and engine depth metrics
- [x] `TeachingOverlay` component (`src/client/components/TeachingOverlay.tsx`)
  - Contextual help for all 9 game concepts
  - `useTeachingOverlay` hook for state management
  - `TeachingTopicButtons` quick-access component
  - `getTeachingTopicForMove` helper function

### 8.5 Backend vs Sandbox HUD Equivalence ✅ COMPLETE (Dec 4, 2025)

Both `BackendGameHost` and `SandboxGameHost` use the shared `toHUDViewModel` adapter from `src/client/adapters/gameViewModels.ts`. This ensures that core HUD semantics (phase labels, decision copy, countdown severity, status chips) are identical regardless of game host, while allowing for intentional differences in transport and connection semantics.

**Test Coverage:** `tests/unit/HUD.backend-vs-sandbox.equivalence.test.tsx` (11 tests)

- Line reward decision equivalence (player and spectator views)
- Movement phase equivalence (no decision)
- Chain capture / capture_direction decision equivalence
- Territory processing / region_order decision equivalence
- Ring elimination decision equivalence
- Time-pressure chip severity equivalence
- Intentional differences documentation (heartbeat staleness, spectators)

**Intentional Differences:**
| Field | Backend | Sandbox | Reason |
|-------|---------|---------|--------|
| `connectionStatus` | Real WebSocket state | Always `'connected'` | Sandbox has no network layer |
| `lastHeartbeatAt` | Tracked timestamp | Always `null` | No heartbeats without WebSocket |
| `isConnectionStale` | Derived from heartbeat age | Always `false` | No staleness without heartbeats |
| `isSpectator` | True for non‑players | Always `false` | No spectator concept in local sandbox |

### 8.6 Mobile & Responsive ✅ COMPLETE (Dec 4, 2025)

- [x] Mobile detection hooks (`src/client/hooks/useIsMobile.ts`)
  - `useIsMobile()` – viewport width ≤768px detection with reactive updates
  - `useIsTouchDevice()` – touch capability detection via `ontouchstart` / `navigator.maxTouchPoints`
  - `useMobileState()` – combined hook returning `{ isMobile, isTouch }` for layout decisions
  - SSR-safe with `typeof window === 'undefined'` guards
  - Tests in `tests/unit/hooks/useIsMobile.test.tsx`
- [x] Touch-optimized controls for stack selection and movement
  - Already implemented via `SandboxTouchControlsPanel.tsx` (drag-to-move, tap-to-select)
  - E2E coverage in `tests/e2e/sandbox.touch.e2e.spec.ts`
- [x] Simplified mobile HUD layout (`src/client/components/MobileGameHUD.tsx`)
  - Compact single-row phase bar with expandable player cards
  - Touch-friendly tap interactions for player details
  - Decision timer with same severity thresholds as desktop HUD
  - Uses shared `HUDViewModel` adapter for consistency with `GameHUD`

### 8.7 Accessibility (Wave 14.5) ✅ COMPLETE (Dec 5, 2025)

> **Note:** Accessibility feature work landed primarily under Wave 14.5; this section summarizes the Wave‑8 intent and points to the canonical accessibility docs.

- [x] Keyboard navigation for all game actions – implemented via `useKeyboardNavigation` and global game shortcuts, with per-phase handling for board, dialogs, and core actions.
- [x] Screen reader support for game state announcements – implemented through ARIA roles/labels on HUD and BoardView plus `ScreenReaderAnnouncer` and related hooks. See [`docs/ACCESSIBILITY.md`](docs/ACCESSIBILITY.md).
- [x] High-contrast mode option – provided by `AccessibilityContext` + `AccessibilitySettingsPanel` and `accessibility.css` high‑contrast theme.
- [x] Colorblind-friendly player color palette – deuteranopia/protanopia/tritanopia modes provided via `AccessibilityContext` and applied across HUD/BoardView.

---

## Wave 9: AI Strength & Optimization ✅ COMPLETE (Dec 4, 2025)

**Goal:** Provide challenging AI opponents across all skill levels, from beginner to expert.

**Rationale:** Heuristic-based AI with CMA-ES optimized weights across all major board configurations.

**Status:** ✅ ALL CORE TASKS COMPLETE (9.4 Search Enhancements deferred to future)

### 9.1 Production AI Ladder ✅ COMPLETE (Dec 4, 2025)

The canonical difficulty ladder in `ai-service/app/main.py` already maps all 10 difficulty levels to the appropriate AI engines:

| Difficulty | AI Type   | Randomness | Think Time |
| ---------- | --------- | ---------- | ---------- |
| 1          | RANDOM    | 0.50       | 150ms      |
| 2          | HEURISTIC | 0.30       | 200ms      |
| 3          | MINIMAX   | 0.20       | 1,250ms    |
| 4          | MINIMAX   | 0.10       | 2,100ms    |
| 5          | MINIMAX   | 0.05       | 3,500ms    |
| 6          | MINIMAX   | 0.02       | 4,800ms    |
| 7          | MCTS      | 0.00       | 7,000ms    |
| 8          | MCTS      | 0.00       | 9,600ms    |
| 9          | DESCENT   | 0.00       | 12,600ms   |
| 10         | DESCENT   | 0.00       | 16,000ms   |

All 5 PlayerChoice endpoints are fully service-backed:

- [x] line_reward_option ✅ (`AIInteractionHandler` → `AIEngine` → `AIServiceClient` → Python)
- [x] ring_elimination ✅
- [x] region_order ✅
- [x] line_order ✅
- [x] capture_direction ✅

### 9.2 Heuristic Weight Optimization ✅ SUBSTANTIALLY COMPLETE (Dec 4, 2025)

All major board configurations have been optimized via CMA-ES:

| Configuration | Status      | Fitness | Generations |
| ------------- | ----------- | ------- | ----------- |
| Square8 2p    | ✅ Complete | 95.8%   | 25          |
| Square19 2p   | ✅ Complete | 83.3%   | 15          |
| 3-player      | ✅ Complete | 65%     | 20          |
| 4-player      | ✅ Complete | 75%     | 20          |
| Hexagonal     | ⚠️ Deferred | -       | -           |

**Completed Tasks:**

- [x] Weight sensitivity analysis on square8, square19 (see `analyze_weight_sensitivity.py`)
- [x] Classify weights by signal strength - visible in code comments:
  - Strong positive: STACK_CONTROL (+9.74), ELIMINATED_RINGS (+12.45), VICTORY_PROXIMITY (+21.28)
  - Inverted: GAP_POTENTIAL (-0.43), MARKER_COUNT on 19x19 (-1.90)
  - Pruned: STACK_DIVERSITY_BONUS (0.03), SWAP_EXPLORATION_TEMPERATURE (~0)
- [x] CMA-ES optimization with classified weights → `BASE_V1_BALANCED_WEIGHTS`
- [x] Tournament validation against baseline (documented in weight profile comments)
- [x] Board-specific profiles: `HEURISTIC_V1_SQUARE19_2P`, player-count profiles

**Key Optimization Insights:**

- Center control: Overrated on small boards (-55%), critical on large boards (+145% on 19x19)
- Marker count: Positive on small boards (+3.38), negative on large boards (-1.90)
- Stack diversity: Near-zero on 2p (0.03), significant on multiplayer (+1.99 on 4p)
- Gap potential: Inverted from positive to negative (-0.43) - gaps hurt

**Deferred:**

- [ ] Hexagonal board-specific optimization (eval pool `hex_4p/pool_v1.jsonl` ready)

### 9.3 RNG Determinism ✅ COMPLETE (Dec 4, 2025)

- [x] Replace global `random` with per-game seeded RNG in Python AI
  - Fixed `heuristic_ai._sample_moves_for_training()` to use `self.rng.sample()` instead of global `random.sample()`
  - All AI classes inherit from `BaseAI` which creates `self.rng = random.Random(self.rng_seed)` per-instance
- [x] ZobristHash uses stable, seeded RNG (already correct)
  - Uses `self._rng = random.Random(42)` with fixed seed for consistent hash keys across all games
  - Zobrist keys must be identical for identical positions, so fixed seed is correct behavior
- [x] RNG seeds passed from TS backend to Python service in /ai/move requests
  - `AIServiceClient.getAIMove()` passes `seed` derived from `gameState.rngSeed`
  - `main.py` `/ai/move` endpoint computes `effective_seed` via `_select_effective_seed()` and creates `AIConfig(rngSeed=effective_seed)`
  - `BaseAI.__init__()` uses `config.rng_seed` to seed the per-instance RNG

### 9.4 Search Enhancements (Future)

- [ ] Move ordering heuristics for better alpha-beta pruning
- [ ] Transposition table for position caching
- [ ] Iterative deepening with time limits
- [ ] Opening book from strong AI self-play

### 9.5 AI Observability ✅ COMPLETE (Dec 4, 2025)

- [x] Per-difficulty latency tracking in Grafana
  - Added "AI Latency by Difficulty (P95)" panel with all 10 difficulty levels
  - Added "AI Latency by Engine Type (P95)" panel with color-coded AI types (RANDOM, HEURISTIC, MINIMAX, MCTS, DESCENT)
  - Documented expected latency ranges: Difficulty 1-2: <500ms, 3-6: 1-5s, 7-8: 7-10s, 9-10: 12-16s
- [x] Fallback rate monitoring by endpoint
  - Added dedicated "AI Fallback Rate by Reason" panel with thresholds (>1/min yellow, >5/min red)
  - Existing `ringrift_ai_fallback_total{reason}` metric already labels by reason (timeout, service_unavailable, etc.)
- [x] AI request metrics infrastructure verified complete
  - `aiRequestDuration` histogram with `ai_type` and `difficulty` labels in MetricsService.ts
  - `AI_MOVE_LATENCY` histogram with matching labels in Python ai-service/app/metrics.py
  - `aiFallbackTotal` counter with `reason` label in MetricsService.ts

**Note:** Win rate vs random and move consistency metrics deferred to Wave 10 (Game Records) as they require game record storage infrastructure.

---

## Wave 10: Game Records & Training Data (IN PROGRESS)

**Goal:** Comprehensive game storage, notation, and replay system for analysis, training, and competitive features.

**Rationale:** Self-play games are valuable training data. Players want replays. Analysis tools need game history.

### 10.1 Game Record Types ✅ COMPLETE (Dec 4, 2025)

- [x] Python `GameRecord` types in `ai-service/app/models/game_record.py`
  - `GameOutcome`, `RecordSource` enums
  - `PlayerRecordInfo`, `MoveRecord`, `GameRecordMetadata`, `FinalScore`, `GameRecord` Pydantic models
  - RRN (RingRift Notation) system: `RRNCoordinate`, `RRNMove` types
  - Functions: `_generate_rrn`, `parse_rrn_move`, `game_record_to_rrn`, `rrn_to_moves`
- [x] TypeScript `GameRecord` types in `src/shared/types/gameRecord.ts`
  - Mirrors Python types with TypeScript equivalents
  - RRN functions: `positionToRRN`, `rrnToPosition`, `moveRecordToRRN`, `parseRRNMove`, `gameRecordToRRN`, `rrnToMoves`
  - `CoordinateUtils` object with utilities for all board types (square8, square19, hexagonal)
  - Handles `exactOptionalPropertyTypes` via conditional spreading
- [x] JSONL export format for training data pipelines
  - `gameRecordToJsonlLine()` and `jsonlLineToGameRecord()` functions
- [x] Algebraic notation (RRN) generator and parser
  - Square boards: `a1`-`h8` (8×8) or `a1`-`s19` (19×19)
  - Hex boards: `(x,y)` or `(x,y,z)` axial coordinates
  - Move notation: `Pa1` (placement), `e4-e6` (movement), `d4xd5-d6` (capture), `d4xd5-d6+` (chain capture)
- [x] Coordinate conversion utilities for all board types
  - `CoordinateUtils.getAllPositions()`, `isValid()`, `distance()`, `getAdjacent()`

### 10.2 Database Integration ✅ SUBSTANTIALLY COMPLETE (Dec 4, 2025)

- [x] Enhanced existing Prisma `Game` model with GameRecord fields (Dec 4, 2025)
  - Added `recordMetadata Json?` (source, tags, generation, candidateId, recordVersion)
  - Added `finalScore Json?` (ringsEliminated, territorySpaces, ringsRemaining per player)
  - Added `outcome String?` (ring_elimination, territory_control, etc.)
  - Created migration `20251204204323_add_game_record_fields`
- [x] Created `GameRecordRepository` service (`src/server/services/GameRecordRepository.ts`)
  - `saveGameRecord(gameId, finalState, outcome, finalScore, metadata)` - saves completed games
  - `getGameRecord(gameId)` - loads full GameRecord by ID
  - `listGameRecords(filter)` - query with board type, outcome, player, date filters
  - `exportAsJsonl(filter)` - async generator for training data pipelines
  - `countGameRecords(filter)` and `deleteOldRecords(beforeDate)` for data management
- [x] Wired game storage into online game completion (Dec 4, 2025)
  - `GameSession.finishGameWithResult()` calls `gameRecordRepository.saveGameRecord()`
  - Added `mapGameResultToOutcome()` helper to convert GameResult.reason to GameOutcome
  - Added `computeFinalScore()` helper to extract per-player statistics from GameState
  - Non-fatal failure handling - GameRecord storage errors don't affect game completion
- [ ] Wire game storage into self-play scripts (CMA-ES, soak tests) - Future work

### 10.3 Self-Play Recording (Track 11) ✅ SUBSTANTIALLY COMPLETE (Dec 4, 2025)

- [x] Default-enabled game recording in `run_cmaes_optimization.py`
  - Per-run DB at `{run_dir}/games.db` (for example `logs/cmaes/runs/<run_id>/games.db`), created via `get_or_create_db(...)`.
  - Recording is enabled by default (`record_games=True`) and can be disabled via `--no-record` or `RINGRIFT_RECORD_SELFPLAY_GAMES=false`.
  - Rich metadata stored in `games.metadata_json`: `source="cmaes"`, board type, num_players, generation, candidate index, run_id, and optional recording-context tags.
- [x] State pool export utility (`ai-service/scripts/export_state_pool.py`)
  - Extracts mid-game states from one or more `GameReplayDB` SQLite databases using `GameReplayDB.get_state_at_move(...)`.
  - Writes JSONL evaluation pools under `data/eval_pools/**` that are consumable by `app.training.eval_pools.load_state_pool(...)` and the CMA-ES / GA fitness harnesses.
- [x] Database merge utility (`ai-service/scripts/merge_game_dbs.py`)
  - Merges multiple per-run `games.db` files into a consolidated `GameReplayDB`, preserving metadata and optionally renaming conflicting `game_id`s.
  - Intended for long-running experiments and for preparing unified corpora before exporting evaluation pools or running replay parity/health checks.
- [x] Environment variables: `RINGRIFT_RECORD_SELFPLAY_GAMES`, `RINGRIFT_SELFPLAY_DB_PATH`
  - Implemented in `ai-service/app/db/recording.py` and honoured by `run_self_play_soak.py`, `run_cmaes_optimization.py`, and other self-play / training scripts via `should_record_games(...)` and `get_or_create_db(...)`.

### 10.4 Replay System ✅ SUBSTANTIALLY COMPLETE (Dec 4, 2025)

- [x] `reconstructStateAtMove(gameRecord, moveIndex)` in shared engine (Dec 4, 2025)
  - Implemented in `src/shared/engine/replayHelpers.ts` and exported via the canonical engine surface in `src/shared/engine/index.ts`.
  - Uses `createInitialGameState(...)` plus the orchestrator's `processTurn(...)` to reconstruct a `GameState` from a `GameRecord` at any move index (0 = initial), suitable for TS-only replay tooling and TS↔Python cross-checks.
- [ ] Golden replay suite (`golden_games`) for end-to-end rules coverage (deferred to Wave 11)
  - Curated set of recorded games (line+territory, chain capture, pie rule, LPS/multi-player, structural invariants) stored under `ai-service/tests/fixtures/golden_games/` (GameReplayDB) and `tests/fixtures/golden-games/` (GameRecord JSONL).
  - Backed by strict TS and Python tests that replay these traces end-to-end and assert full parity and invariant preservation. Design is captured in `docs/ENGINE_TOOLING_PARITY_RESEARCH_PLAN.md` (Golden Replay Suite section) for implementation in Wave 10+.
- [x] Checkpoint caching for efficient backward navigation (Dec 4, 2025)
  - Implemented via `useReplayPlayback` hook's `prefetchAdjacent` function
  - Prefetches adjacent states (2 ahead, 1 behind) plus key positions (0%, 25%, 50%, 75%, 100%) for long games
  - Uses React Query for caching with `getCachedState` lookup before fetch
  - Adaptive prefetch depth based on playback state and speed
- [x] `ReplayControls` UI component with play/pause/step/seek (Dec 4, 2025)
  - Full `PlaybackControls` component in `src/client/components/ReplayPanel/PlaybackControls.tsx`
  - Play/pause, step forward/backward, jump to start/end, speed selector (0.5x/1x/2x/4x)
  - Scrubber slider for random access to any move
  - Keyboard shortcuts: arrows/h/l step, space play/pause, Home/End jump, [ ] speed
- [x] `MoveList` component with move annotations (Dec 4, 2025)
  - `MoveHistory` component in `src/client/components/MoveHistory.tsx`
  - Compact chess-like notation with player color indicators
  - Click-to-navigate to any move via `onMoveClick` callback
  - Auto-scroll to current move with container-only scrolling
- [x] Sandbox integration for replay viewing via the `/sandbox` page (Self-Play Browser → `LoadableScenario` Option B path) and the ReplayPanel bridge to the Python `GameReplayDB` (Option A); full GameRecord-based replay system remains planned.

### 10.5 Self-Play Browser UI

- [x] API endpoints in `src/server/routes/selfplay.ts`
- [x] `SelfPlayGameService` for database access (read-only SQLite with 0-game databases filtered out of the sandbox dropdown).
- [x] `SelfPlayBrowser` component for game discovery (`src/client/components/SelfPlayBrowser.tsx`), wired into `SandboxGameHost` via the "Browse Self-Play Games" panel.
- [x] Filter by board type, player count, outcome, source (backed by `/api/selfplay/games` query params and client-side dropdown filters).
- [x] Fork games from replay position into sandbox by loading a selected self-play game as a `LoadableScenario` and replaying its moves through `ClientSandboxEngine` so the history slider can scrub the full game locally.

---

## Wave 11: Test Hardening & Golden Replays ✅ COMPLETE (Dec 4, 2025)

**Goal:** Build a curated suite of "golden" recorded games that exercise all major rules axes, providing end-to-end replay parity tests between TypeScript and Python engines.

**Rationale:** Golden replay tests provide the highest confidence in cross-language parity. A curated set of games covering line+territory, chain capture, pie rule, LPS/multi-player, and structural invariants ensures that any engine change is immediately validated against known-correct behavior.

**Status:** ✅ TEST INFRASTRUCTURE COMPLETE

**Key Deliverables:**

- [`tests/golden/goldenReplayHelpers.ts`](tests/golden/goldenReplayHelpers.ts) - TypeScript invariant checkers and replay utilities
- [`tests/golden/goldenReplay.test.ts`](tests/golden/goldenReplay.test.ts) - Jest test runner with dynamic test generation
- [`ai-service/tests/golden/test_golden_replay.py`](ai-service/tests/golden/test_golden_replay.py) - Python pytest runner
- [`docs/testing/GOLDEN_REPLAYS.md`](docs/testing/GOLDEN_REPLAYS.md) - Documentation for golden replay system

### 11.1 Golden Game Fixtures ✅ SUBSTANTIALLY COMPLETE (Dec 5, 2025)

- [x] Create `tests/fixtures/golden-games/` directory for JSONL GameRecord fixtures
- [x] Create `ai-service/tests/fixtures/golden_games/` directory for Python fixtures
- [x] Generate self-play games covering board types and player counts (Dec 5, 2025):
  - 29 total candidates available across all board types/player counts
  - Square8/19 2-player: 3 games from canonical DBs
  - Hexagonal 2-player: 10 games from `golden_hexagonal.db`
  - Square8 3-player: 8 games (2 with winners) from `golden_3player.db`
  - Square8 4-player: 8 games from `golden_4player.db`
- [ ] Curate golden game set covering (some fixtures still needed):
  - [ ] Line detection + reward scenarios (single/multiple lines, reward choices)
  - [ ] Territory formation + region ordering
  - [ ] Chain capture (simple, multi-hop, cyclic)
  - [ ] Swap/pie rule activation
  - [x] Multi-player games (3-4 player)
  - [x] Hexagonal geometry specifics

### 11.2 TypeScript Golden Replay Tests ✅ COMPLETE (Dec 4, 2025)

- [x] Create `tests/golden/goldenReplay.test.ts` test suite
- [x] Implement `replayAndAssertInvariants(gameRecord)` helper in `goldenReplayHelpers.ts`
- [x] Assert structural invariants at each move:
  - INV-BOARD-CONSISTENCY: valid positions, stack heights, no duplicate rings
  - INV-TURN-SEQUENCE: monotonic move numbers, valid phase transitions
  - INV-PLAYER-RINGS: ring counts within bounds
  - INV-PHASE-VALID: current phase is valid, phase-specific state consistent
  - INV-ACTIVE-PLAYER: active player in range, not eliminated (unless game over)
  - INV-GAME-STATUS: valid status, winner only when completed
- [x] Assert final state matches recorded outcome via `assertFinalStateMatchesOutcome`

### 11.3 Python Golden Replay Tests ✅ COMPLETE (Dec 4, 2025)

- [x] Create `ai-service/tests/golden/test_golden_replay.py` test suite
- [x] Implement `replay_and_assert_invariants(game_info)` helper
- [x] Mirror TS invariant assertions in Python (6 invariant checkers)
- [x] Graceful skip when no fixtures exist (tests pass without fixtures)

### 11.4 Cross-Language Parity Validation ✅ COMPLETE (Dec 4, 2025)

- [x] Document golden game curation process in `docs/testing/GOLDEN_REPLAYS.md`
- [x] Test infrastructure supports category/board type/player count coverage tracking
- [ ] Create script to generate golden fixtures from existing self-play DBs (future)
- [ ] Add CI job running golden replay tests on both TS and Python (future, tests already CI-compatible)

### 11.5 Coverage Gap Analysis (Future)

- [ ] Run coverage analysis on golden replay tests
- [ ] Identify rules axes not yet covered by golden games
- [ ] Expand golden game set to fill coverage gaps
- [ ] Target: every major rules branch exercised by at least one golden game

---

## Wave 13: Multi-Player (3-4 Players) ✅ SUBSTANTIALLY COMPLETE (Dec 4, 2025)

**Goal:** Full support for 3 and 4 player games.

**Rationale:** Multi-player support enables richer strategic gameplay and social play.

**Status:** ✅ CORE INFRASTRUCTURE COMPLETE - Rules engine, AI, and UI all support 3-4 players

### 13.1 Rules Verification ✅ COMPLETE (Dec 4, 2025)

The shared rules engine has supported 3-4 players since initial implementation:

- [x] Rules engine supports variable `numPlayers` (2-4) throughout all aggregates
  - `TurnMutator.ts`: Wraps `currentPlayer` using `numPlayers` for turn cycling
  - `LineAggregate.ts`: Uses `getEffectiveLineLengthThreshold(boardType, numPlayers)` for multi-player line requirements
  - `VictoryAggregate.ts`: Last Player Standing (LPS) logic handles 3-4 player elimination correctly
  - `heuristicEvaluation.ts`: Position evaluation accounts for 3+ player threat assessment
- [x] Python AI engine matches TypeScript multi-player semantics
  - `ai-service/app/game_engine.py`: Full 3-4 player support
  - `ai-service/app/rules/core.py`: Turn cycling and phase transitions for variable player counts
- [x] Swap/pie rule correctly disabled for 3-4 player games
  - `swapSidesHelpers.ts`: `isSwapSidesAvailable()` returns false if `state.players.length !== 2`
- [x] Evaluation pools exist for multiplayer scenarios
  - `ai-service/data/eval_pools/square19_3p/pool_v1.jsonl` - 3-player Square19 pool
  - `ai-service/data/eval_pools/hex_4p/pool_v1.jsonl` - 4-player Hexagonal pool
- [x] Python tests verify multi-player pool loading
  - `ai-service/tests/test_eval_pools_multiplayer.py` - Tests 3p Square19 and 4p Hex pool loading

### 13.2 UI Adaptations ✅ COMPLETE (Dec 4, 2025)

The UI already supports 3-4 player games:

- [x] Sandbox configuration panel supports 2-4 player selection
  - `SandboxGameHost.tsx`: Player count buttons (2, 3, 4) in "Players" section
  - Per-seat player type configuration (human/AI) for all active seats
- [x] Lobby shows player count and capacity
  - `LobbyPage.tsx`: Displays `{playerCount}/{game.maxPlayers} players`
  - Filter by player count in lobby filters
- [x] GameHUD adapts to variable player counts
  - `GameHUD.tsx`: Iterates over `players` array, works for 1-4 players
  - `PlayerCardFromVM`: Renders for each player in the game
- [x] VictoryModal shows multi-player game summary
  - `VictoryModal.tsx`: Displays `{summary.playerCount}` and per-player stats
- [x] ReplayPanel shows player count in game metadata
  - `ReplayPanel.tsx`: Displays `{game.numPlayers}P` badges
- [x] ScenarioPickerModal shows player count for scenarios
  - `ScenarioPickerModal.tsx`: Displays `{scenario.playerCount}P` badges
- [x] Self-play browser filters by player count
  - `SelfPlayBrowser.tsx`: `playerCountFilter` state with 2/3/4/all options

### 13.3 AI for Multi-Player ✅ COMPLETE (Dec 4, 2025)

AI heuristics have been optimized for multi-player:

- [x] CMA-ES weight optimization completed for 3-player and 4-player configurations
  - 3-player optimization: 65% fitness, 20 generations
  - 4-player optimization: 75% fitness, 20 generations
  - Documented in `IMPROVEMENT_PLAN.md` Wave 9.2 and `AI_TRAINING_ASSESSMENT_FINAL.md`
- [x] Heuristic weights include multiplayer-specific adjustments
  - `heuristic_weights.py`: `HEURISTIC_V1_SQUARE19_3P_WEIGHTS` profile
  - Stack diversity bonus: near-zero on 2p (0.03), significant on 4p (+1.99)
- [x] Heuristic evaluation handles 3+ player threat assessment
  - `heuristicEvaluation.ts`: Comments document "In 3+ player games, reward being one of the few players with real actions left"
  - Position evaluation penalizes being the only player without actions
  - Penalizes positions where a single opponent is much closer to victory

### 13.4 Future Enhancements (Optional)

- [ ] Add 3-4 player contract vectors to `tests/fixtures/contract-vectors/v2/`
- [ ] Create dedicated E2E tests for 3-4 player online games (`tests/e2e/multiplayer-3-4p.e2e.spec.ts`)
- [ ] Hex-specific multi-player optimization (eval pool ready at `hex_4p/pool_v1.jsonl`)

---

## Wave 14: Accessibility & Code Quality ✅ COMPLETE (Dec 5, 2025)

**Goal:** Make RingRift accessible to all players and improve overall code quality.

**Rationale:** Accessibility features enable players with disabilities to enjoy the game. Code quality improvements ensure long-term maintainability and reduce technical debt.

**Status:** ✅ COMPLETE

### 14.1 Keyboard Navigation (P1) ✅ COMPLETE (Dec 4, 2025)

- [x] Implement keyboard focus management for game board
  - Arrow keys for cell navigation
  - Enter/Space for selection/confirmation
  - Escape for cancel/deselect
  - Home/End for jumping to first/last cell
- [x] Add visible focus indicators for all interactive elements
  - `PLAYER_FOCUS_RING_CLASSES` with player-specific colors (emerald/sky/amber/fuchsia)
  - `getPlayerFocusRingClass()` helper for consistent styling
- [x] Implement keyboard shortcuts for common actions
  - `KeyboardShortcutsHelp.tsx` now shows Board Navigation, Game Actions, Dialog Navigation, and General shortcuts
  - Game Actions: ? (help), R (resign), M (mute), F (fullscreen), Ctrl+Z/Ctrl+Shift+Z (undo/redo sandbox)
- [x] Create `useKeyboardNavigation` hook for board interactions
  - Grid-based focus management for square8, square19, and hexagonal boards
  - `moveFocus()`, `handleKeyDown()`, `registerCellRef()`, `isFocused()` methods
  - Spectator mode detection (blocks selection when `isSpectator`)
- [x] Create `useGlobalGameShortcuts` hook for global shortcuts
  - Document-level event listener for R, M, F, ?, Ctrl+Z, Ctrl+Shift+Z
  - Ignores shortcuts when focus is in input/textarea
- [x] Unit tests for keyboard navigation (50 tests passing)
  - `tests/unit/hooks/useKeyboardNavigation.test.tsx`

**Files Created:**

- `src/client/hooks/useKeyboardNavigation.ts` - Core hooks and helpers
- `tests/unit/hooks/useKeyboardNavigation.test.tsx` - Comprehensive test coverage

**Files Modified:**

- `src/client/hooks/index.ts` - Exports for new hooks
- `src/client/components/KeyboardShortcutsHelp.tsx` - Added Game Actions section with Home/End

### 14.2 Screen Reader Support (P1) ✅ SUBSTANTIALLY COMPLETE (Dec 5, 2025)

- [x] Add ARIA labels to GameHUD elements (Dec 4, 2025)
  - Phase indicator with `role="status"` and comprehensive `aria-label` (phase name, description, action hint)
  - Player cards with `aria-label` including name, stats, turn status, ring counts
  - Score summary with `role="region"` and accessible labels
  - Decision phase banner with `role="status"`, severity-appropriate `aria-live` (polite/assertive)
  - Spectator mode banner with `role="status"` and `aria-label`
  - Spectator count chip with accessible label
  - Victory condition tooltips with `aria-label` attributes
- [x] Add ARIA labels to BoardView/board cells (Dec 5, 2025)
  - Board container with `role="grid"` and comprehensive `aria-label` including navigation instructions
  - Each cell with `role="gridcell"`, detailed `aria-label` (position, stack info, valid move target status)
  - `aria-selected` attribute for selected cells
  - Keyboard-focused vs selected distinction with appropriate focus rings
  - Screen reader live region in BoardView for selection announcements
- [x] Implement live regions for game state changes (Dec 5, 2025)
  - Phase indicator uses `aria-live="polite"` for phase transitions
  - Decision phase banner uses `aria-live="assertive"` for critical timeouts
  - Spectator banner uses `aria-live="polite"`
  - BoardView has internal `aria-live="polite"` region for selection and valid move announcements
- [x] Extend `ScreenReaderAnnouncer.tsx` component (Dec 5, 2025)
  - Priority queue system via `useGameAnnouncements` hook
  - Polite vs assertive modes based on announcement category
  - `GameAnnouncements` helper object with generators for all event types:
    - Turn changes, phase transitions, moves, placements, captures
    - Chain captures, line formations, territory claims
    - Victory/defeat, player elimination, timer warnings
    - Cell selection, valid moves, decisions required, ring stats
  - `useGameStateAnnouncements` hook for automatic announcements on state changes
  - Category-based configuration: priority, politeness, debounce timing
  - Alternating live regions to ensure duplicate messages are announced
- [x] Screen reader announcer wired to game hosts (Dec 5, 2025)
  - BackendGameHost and SandboxGameHost both include `<ScreenReaderAnnouncer>`
  - **Priority queue mode fully integrated** via `useGameAnnouncements` and `useGameStateAnnouncements` hooks
  - Automatic announcements for turn changes, phase transitions, victories, timer warnings
  - Queue-based processing with priority sorting (high → medium → low)
  - Category-based debouncing to prevent announcement spam
- [x] Global settings modal with accessibility panel (Dec 5, 2025)
  - `SettingsModal.tsx` accessible from navbar gear icon
  - `AccessibilitySettingsPanel.tsx` for high contrast, colorblind modes, reduced motion, large text
  - Keyboard shortcut: `Ctrl+,` / `Cmd+,` to open settings
  - Focus trap and escape-to-close for modal accessibility
  - Color preview for player colors in each colorblind mode

**Future Enhancements (Deferred):**

- [ ] Create dedicated screen reader-friendly game log component (text descriptions of all moves)
- [ ] Add move-by-move announcements for captures, line formations, and territory claims

### 14.3 Visual Accessibility (P2) ✅ SUBSTANTIALLY COMPLETE (Dec 5, 2025)

- [x] Create `AccessibilityContext` (`src/client/contexts/AccessibilityContext.tsx`)
  - Manages user preferences: `highContrastMode`, `colorVisionMode`, `reducedMotion`, `largeText`
  - Persists to localStorage with key `ringrift-accessibility-preferences`
  - Detects system `prefers-reduced-motion` preference
  - Applies CSS classes to document root: `.high-contrast`, `.reduce-motion`, `.large-text`, `data-color-vision`
- [x] Create `AccessibilitySettingsPanel` component (`src/client/components/AccessibilitySettingsPanel.tsx`)
  - Toggle switches for high contrast, reduced motion, large text
  - Dropdown for color vision mode (normal, deuteranopia, protanopia, tritanopia)
  - Color preview showing player color palette
  - Reset to defaults button
- [x] Wire `AccessibilityProvider` into `App.tsx`
- [x] Create dedicated accessibility stylesheet (`src/client/styles/accessibility.css`)
  - High-contrast mode: stronger borders, brighter focus indicators
  - Reduced motion: disables animations and transitions
  - Large text mode: scales up font sizes
  - Color vision patterns for player indicators
  - Touch target sizing for accessibility (44px minimum)
  - Screen reader utilities (`.sr-only`, skip links, live region styling)
- [x] Colorblind-friendly player color palettes
  - Normal: emerald/sky/amber/fuchsia
  - Deuteranopia/Protanopia: blue/orange/cyan/violet (avoids red-green confusion)
  - Tritanopia: pink/cyan/lime/orange (avoids blue-yellow confusion)
  - Helper functions: `getPlayerColorClass()`, `getPlayerColor()` in context

**Files Created:**

- `src/client/contexts/AccessibilityContext.tsx` - Core context and hooks
- `src/client/components/AccessibilitySettingsPanel.tsx` - Settings UI
- `src/client/styles/accessibility.css` - Accessibility CSS rules

**Future Enhancements (Deferred):**

- [ ] Add AccessibilitySettingsPanel to a settings page or modal in Layout
- [ ] Board-level pinch-to-zoom implementation
- [ ] Additional player indicator shape differentiation (beyond patterns)

### 14.4 Test Coverage Improvements (P1) ✅ SUBSTANTIALLY COMPLETE (Dec 5, 2025)

- [x] Audit and reduce skipped tests (Dec 5, 2025)
  - Current count: 102 skipped tests (significantly improved from ~170)
  - **Audit findings:**
    - 16 in `statePersistence.branchCoverage.test.ts` - INTENTIONAL browser-only (DOM APIs: `File.text()`, `URL.createObjectURL`, `createElement`)
    - 5 in `GameSession.branchCoverage.test.ts` - Require AI/WebSocket infrastructure mocking
    - 4 in `envFlags.test.ts` - Production JWT secrets validation (env-specific)
    - ~20 in parity/trace tests - Require fixture data or Python service
    - Remaining - Various edge cases needing complex setup
  - **Recommendation:** Accept ~50 intentional skips as documented platform limitations
- [x] Context coverage analysis (Dec 5, 2025)
  - **GameContext.test.tsx**: 43+ tests covering connection lifecycle, game state updates, choices, chat, victory, reconnection
  - **SandboxContext.test.tsx**: 33+ tests covering provider lifecycle, stall watchdog, diagnostics mode, AI integration
  - **GameContext.branchCoverage.test.tsx**: Additional 14 tests for edge cases (warnings, rematch functions, errors)
  - **Combined coverage for `src/client/contexts/`:**
    - Lines: 77.06% (326/423) - target 80%
    - Branches: 55.55% (85/153) - socket event handlers are hard to mock fully
    - Functions: 70.78% (63/89)
    - Statements: 74.49% (333/447)
  - **Assessment:** Remaining branch gaps are in deep socket event simulation paths that require complex mock Socket.IO wiring; existing tests provide excellent functional coverage of all public APIs
- [x] Add accessibility-focused tests (Dec 5, 2025)
  - `tests/unit/hooks/useKeyboardNavigation.test.tsx` - 50 tests for keyboard navigation
  - `tests/unit/components/ScreenReaderAnnouncer.test.tsx` - Screen reader announcements
  - ARIA attribute verification in existing component tests
- [ ] Expand contract vector coverage (future)
  - Add hexagonal geometry edge cases
  - Add 3-4 player contract vectors
  - Document vector coverage per rules axis

### 14.5 Documentation Updates (P2) ✅ SUBSTANTIALLY COMPLETE (Dec 5, 2025)

- [x] Create ACCESSIBILITY.md guide (`docs/ACCESSIBILITY.md`)
  - Keyboard shortcuts reference (board navigation, game actions, dialog navigation, general)
  - Screen reader usage guide (game board, HUD, live announcements)
  - Color contrast information (color palettes for all vision modes, pattern differentiation)
  - Visual accessibility settings (high contrast, reduced motion, large text)
  - Quick reference section with shortcuts summary and ARIA roles
- [x] Update CONTRIBUTING.md with accessibility guidelines
  - ARIA requirements for new components (roles, labels, live regions, state attributes)
  - Focus management patterns (tab order, focus trapping, focus restoration code example)
  - Keyboard navigation requirements (hooks to use, code examples)
  - Color and visual accessibility (AccessibilityContext usage)
  - Testing accessibility changes (unit test examples, manual testing checklist)
  - Accessibility files reference table
- [ ] Add accessibility testing to CI (Future)
  - axe-core integration for automated checks
  - Keyboard-only E2E test suite
  - Color contrast validation
  - Note: Infrastructure in place via existing Jest tests; CI integration deferred to future wave

---

## Wave 12: Matchmaking & Ratings ✅ COMPLETE (Dec 4, 2025)

**Goal:** Implement player matchmaking and rating system for competitive play.

**Rationale:** Players need a way to find opponents of similar skill and track their progress.

**Status:** ✅ ALL CORE TASKS COMPLETE

### 12.1 Rating System ✅ COMPLETE (Dec 4, 2025)

- [x] Elo rating calculation (`src/server/services/RatingService.ts`)
  - `calculateExpectedScore()` - probability of winning based on rating difference
  - `calculateNewRating()` - standard Elo formula with K-factor
  - `calculateMultiplayerRatings()` - pairwise calculation for 3-4 player games
  - `getKFactor()` - higher K for provisional players (<20 games)
- [x] Rating persistence in database
  - User table includes `rating`, `gamesPlayed`, `gamesWon` fields
  - `processGameResult()` updates ratings after game completion
  - Integration with `GameSession.finishGameWithResult()` for online games
- [x] Rating constants in `src/shared/types/user.ts`
  - `DEFAULT_RATING: 1500` - starting rating for new players
  - `K_FACTOR: 32` - rating adjustment sensitivity
  - `PROVISIONAL_GAMES: 20` - games before rating stabilizes
  - `MIN_RATING: 100`, `MAX_RATING: 3000` - bounds
- [x] Test coverage
  - `tests/unit/RatingService.elo.test.ts` - Elo calculation tests
  - `tests/unit/RatingService.integration.test.ts` - database integration
  - `tests/e2e/ratings.e2e.spec.ts` - end-to-end rating flow

**Future Enhancement:** Rating history tracking (RatingHistory table) for profile graphs.

### 12.2 Matchmaking Queue ✅ COMPLETE (Dec 4, 2025)

- [x] Queue system implementation (`src/server/services/MatchmakingService.ts`)
  - In-memory queue with `QueueEntry` structure (userId, socketId, preferences, rating, joinedAt)
  - FCFS processing with rating-based matching
  - Bidirectional rating compatibility check
- [x] Rating-based matching
  - Rating range expansion over time (50 points per 5 seconds)
  - Maximum wait time of 60 seconds before accepting any rating
  - Board type compatibility check
- [x] Timeout handling
  - `MAX_WAIT_TIME_MS: 60000` - 1 minute max queue time
  - Estimated wait time calculation in status updates
- [x] WebSocket notifications
  - `matchmaking-status` - queue position, estimated wait time
  - `match-found` - game ID when match created
  - Error handling with re-queue or notification

### 12.3 Leaderboards ✅ COMPLETE (Dec 4, 2025)

- [x] API endpoints (`src/server/routes/user.ts`)
  - `GET /users/leaderboard` - paginated leaderboard (limit, offset)
  - `GET /users/:userId/rating` - individual player rating and rank
  - Sorted by rating descending, active users with games played
- [x] UI component (`src/client/pages/LeaderboardPage.tsx`)
  - Table view with rank, player name, rating, win rate, games played
  - Loading and error states
  - Dark theme styling consistent with rest of app
- [x] Navigation integration
  - Link in `Layout.tsx` navigation bar
  - Link from `HomePage.tsx` "View Leaderboard" card
  - Route at `/leaderboard` in `App.tsx`
- [x] API client (`src/client/services/api.ts`)
  - `userApi.getLeaderboard({ limit, page })` method

**Future Enhancement:** Filtering by time period (weekly/monthly) and board type.

---

## Recommended Execution Order

### Phase A: Production Readiness (Wave 7 Completion)

**Focus:** Validate infrastructure at scale before player-facing changes.

| Task                                       | Wave | Priority |
| ------------------------------------------ | ---- | -------- |
| Run load test Scenarios P1-P4              | 7.1  | HIGH     |
| Document baseline metrics & capacity model | 7.2  | HIGH     |
| Execute operational drills                 | 7.4  | MEDIUM   |
| CI/dependency maintenance                  | 7.2  | LOW      |

### Phase B: Player Experience (Wave 8)

**Focus:** Transform developer UI into player-friendly experience.

| Task                                | Wave | Priority |
| ----------------------------------- | ---- | -------- |
| HUD visual hierarchy redesign       | 8.1  | HIGH     |
| Phase-specific prompts & feedback   | 8.2  | HIGH     |
| Victory/defeat screens with summary | 8.2  | MEDIUM   |
| Interactive tutorial mode           | 8.1  | MEDIUM   |
| Spectator HUD with annotations      | 8.3  | LOW      |

### Phase C: AI Strength (Wave 9)

**Focus:** Provide challenging opponents for competitive play.

| Task                                     | Wave | Priority |
| ---------------------------------------- | ---- | -------- |
| Wire MinimaxAI for mid-high difficulties | 9.1  | HIGH     |
| Complete weight sensitivity analysis     | 9.2  | HIGH     |
| Run CMA-ES optimization                  | 9.2  | MEDIUM   |
| RNG determinism fix                      | 9.3  | MEDIUM   |
| Service-back remaining choices           | 9.1  | LOW      |

### Phase D: Game Records & Replay (Wave 10)

**Focus:** Enable analysis, training data, and competitive features.

| Task                                  | Wave | Priority |
| ------------------------------------- | ---- | -------- |
| Self-play recording in CMA-ES         | 10.3 | HIGH     |
| State pool export utility             | 10.3 | HIGH     |
| Replay system with controls           | 10.4 | MEDIUM   |
| Self-play browser UI                  | 10.5 | MEDIUM   |
| Database integration for online games | 10.2 | LOW      |

### Longer-term (Post-MVP)

| Task                   | Notes                                      |
| ---------------------- | ------------------------------------------ |
| Matchmaking system     | Automated queue, ELO-based matching        |
| Mobile & responsive UI | Touch controls, mobile HUD                 |
| Accessibility features | Keyboard nav, screen reader, high-contrast |
| Advanced AI search     | Opening book, transposition tables         |

---

## Success Metrics for v1.0

These metrics are an execution-focused view of the project-level success criteria defined in [`PROJECT_GOALS.md`](PROJECT_GOALS.md:1); if they diverge, treat [`PROJECT_GOALS.md`](PROJECT_GOALS.md:1) as canonical and adjust this table.

| Metric          | Target              | Current           |
| --------------- | ------------------- | ----------------- |
| Reliability     | >99.9% uptime       | Not measured      |
| AI Move Latency | p95 <1s             | ~1.5s (staging)   |
| UI Updates      | <16ms               | Needs validation  |
| Rule Compliance | 100% pass           | 100% (54 vectors) |
| Test Coverage   | 80%                 | ~69%              |
| Game Completion | >95% without errors | Needs validation  |

---

## Key Files & References

| Document                            | Purpose                |
| ----------------------------------- | ---------------------- |
| `TODO.md`                           | Active task tracker    |
| `KNOWN_ISSUES.md`                   | Current bugs (P0-P2)   |
| `STRATEGIC_ROADMAP.md`              | Phased roadmap & SLOs  |
| `CURRENT_STATE_ASSESSMENT.md`       | Implementation status  |
| `docs/TEST_CATEGORIES.md`           | CI vs diagnostic tests |
| `docs/PASS21_ASSESSMENT_REPORT.md`  | Latest assessment      |
| `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` | Rollout phases         |

---

## Conclusion

RingRift has achieved a strong architectural foundation with excellent test coverage and documentation. The project is positioned for production deployment pending:

1. **Load test validation** – Confirm SLO compliance at scale
2. **UX polish** – Transform developer UI to player-friendly experience
3. **AI improvements** – Wire up advanced tactics for competitive play

The recommended approach is to complete production validation first (establishing confidence in the infrastructure), then focus on UX polish and AI improvements in parallel. This ensures the platform is stable while user-facing improvements are made.

**Estimated time to production-ready:** 4-6 weeks of focused development.
