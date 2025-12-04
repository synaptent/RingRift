# RingRift Comprehensive Improvement Plan

**Created:** December 3, 2025
**Last Updated:** December 3, 2025
**Based on:** Full project review including TODO.md, KNOWN_ISSUES.md, STRATEGIC_ROADMAP.md, CURRENT_STATE_ASSESSMENT.md, PROJECT_GOALS.md

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

| Wave    | Name                                 | Status         |
| ------- | ------------------------------------ | -------------- |
| Wave 5  | Orchestrator Production Rollout      | ✅ COMPLETE    |
| Wave 6  | Observability & Production Readiness | ✅ COMPLETE    |
| Wave 7  | Production Validation & Scaling      | ✅ COMPLETE    |
| Wave 8  | Player Experience & UX Polish        | 🔄 IN PROGRESS |
| Wave 9  | AI Strength & Optimization           | 📋 PLANNED     |
| Wave 10 | Game Records & Training Data         | 📋 PLANNED     |

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
- [ ] Leaderboards

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

- [ ] Run parity harnesses regularly
- [ ] For each failure, extract first divergence and promote to focused unit test
- [ ] Define "must-cover" scenario sets per rules axis

### 5.3 Contract Vector Expansion

- [ ] Add vectors as edge cases discovered in production play
- [ ] Cover any remaining hexagonal geometry edge cases

---

## Priority 6: Game Record System (Phase 5)

**Goal:** Comprehensive game storage, notation, and replay.

### 6.1 Core Implementation

- [ ] Create Python `GameRecord` types
- [ ] Create TypeScript `GameRecord` types
- [ ] Implement JSONL export format for training data
- [ ] Implement algebraic notation generator/parser

### 6.2 Replay System

- [ ] Implement `reconstructStateAtMove(gameRecord, moveIndex)`
- [ ] Add checkpoint caching for efficient navigation
- [ ] Create `ReplayControls` UI component
- [ ] Integrate replay into sandbox page

### 6.3 Self-Play Game Recording (Track 11)

- [ ] Add default-enabled game recording to CMA-ES optimization
- [ ] Create state pool export utility
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

## Wave 8: Player Experience & UX Polish (IN PROGRESS)

**Goal:** Transform the developer-oriented UI into a player-friendly experience suitable for public release.

**Rationale:** Current UX is optimized for developers and playtesters. First-time players need clearer visual hierarchy, better onboarding, and more intuitive controls.

**Status:** Waves 8.1, 8.3, 8.4 complete; 8.2 mostly complete.

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
- [ ] Redesign HUD visual hierarchy for clarity (existing HUD is functional, minor polish deferred)

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

### 8.5 Mobile & Responsive

- [ ] Responsive board rendering for mobile devices
- [ ] Touch-optimized controls for stack selection and movement
- [ ] Simplified mobile HUD layout

### 8.6 Accessibility

- [ ] Keyboard navigation for all game actions
- [ ] Screen reader support for game state announcements
- [ ] High-contrast mode option
- [ ] Colorblind-friendly player color palette

---

## Wave 9: AI Strength & Optimization (PLANNED)

**Goal:** Provide challenging AI opponents across all skill levels, from beginner to expert.

**Rationale:** Current AI is heuristic-based with limited lookahead. Players need stronger opponents for competitive play and skill development.

### 9.1 Production AI Ladder

- [ ] Wire MinimaxAI for medium-high difficulty levels
- [ ] Expose MCTS implementation in production behind AIProfile
- [ ] Complete service-backing for all PlayerChoices:
  - [x] line_reward_option ✅
  - [x] ring_elimination ✅
  - [x] region_order ✅
  - [ ] line_order
  - [ ] capture_direction

### 9.2 Heuristic Weight Optimization

- [ ] Complete weight sensitivity analysis on square8, square19, hex
- [ ] Classify weights by signal strength:
  - Strong positive (>55% win rate) → Keep
  - Strong negative (<45% win rate) → Invert sign
  - Noise band (45-55%) → Prune or zero-initialize
- [ ] Run CMA-ES optimization on pruned weight set
- [ ] Validate via tournament against baseline
- [ ] Create board-type specific profiles if needed

### 9.3 RNG Determinism

- [ ] Replace global `random` with per-game seeded RNG in Python AI
- [ ] Update ZobristHash to use stable, seeded RNG
- [ ] Pass RNG seeds from TS backend to Python service in /ai/move requests

### 9.4 Search Enhancements (Future)

- [ ] Move ordering heuristics for better alpha-beta pruning
- [ ] Transposition table for position caching
- [ ] Iterative deepening with time limits
- [ ] Opening book from strong AI self-play

### 9.5 AI Observability

- [ ] Per-difficulty latency tracking in Grafana
- [ ] AI quality metrics (win rate vs random, move consistency)
- [ ] Fallback rate monitoring by endpoint

---

## Wave 10: Game Records & Training Data (IN PROGRESS)

**Goal:** Comprehensive game storage, notation, and replay system for analysis, training, and competitive features.

**Rationale:** Self-play games are valuable training data. Players want replays. Analysis tools need game history.

### 10.1 Game Record Types

- [ ] Python `GameRecord` types in `ai-service/app/models/game_record.py`
- [ ] TypeScript `GameRecord` types in `src/shared/types/gameRecord.ts`
- [ ] JSONL export format for training data pipelines
- [ ] Algebraic notation (RRN) generator and parser
- [ ] Coordinate conversion utilities for all board types

### 10.2 Database Integration

- [ ] Add `games` and `moves` tables to Prisma schema
- [ ] Create `GameRecordRepository` for CRUD operations
- [ ] Wire game storage into online game completion
- [ ] Wire game storage into self-play scripts (CMA-ES, soak tests)

### 10.3 Self-Play Recording (Track 11)

- [ ] Default-enabled game recording in `run_cmaes_optimization.py`
  - Per-run DB at `{output_dir}/games.db`
  - Rich metadata: source, generation, candidate, board_type, num_players
- [ ] State pool export utility (`scripts/export_state_pool.py`)
- [ ] Database merge utility (`scripts/merge_game_dbs.py`)
- [ ] Environment variables: `RINGRIFT_RECORD_SELFPLAY_GAMES`, `RINGRIFT_SELFPLAY_DB_PATH`

### 10.4 Replay System

- [ ] `reconstructStateAtMove(gameRecord, moveIndex)` in shared engine
- [ ] Checkpoint caching for efficient backward navigation
- [ ] `ReplayControls` UI component with play/pause/step/seek
- [ ] `MoveList` component with move annotations
- [x] Sandbox integration for replay viewing via the `/sandbox` page (Self-Play Browser → `LoadableScenario` Option B path) and the ReplayPanel bridge to the Python `GameReplayDB` (Option A); full GameRecord-based replay system remains planned.

### 10.5 Self-Play Browser UI

- [x] API endpoints in `src/server/routes/selfplay.ts`
- [x] `SelfPlayGameService` for database access (read-only SQLite with 0-game databases filtered out of the sandbox dropdown).
- [x] `SelfPlayBrowser` component for game discovery (`src/client/components/SelfPlayBrowser.tsx`), wired into `SandboxGameHost` via the "Browse Self-Play Games" panel.
- [x] Filter by board type, player count, outcome, source (backed by `/api/selfplay/games` query params and client-side dropdown filters).
- [x] Fork games from replay position into sandbox by loading a selected self-play game as a `LoadableScenario` and replaying its moves through `ClientSandboxEngine` so the history slider can scrub the full game locally.

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
