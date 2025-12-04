# RingRift Development Wave Plan

> **Created:** December 3, 2025
> **Status:** Active Planning Document
> **Purpose:** Define prioritized development waves for production readiness

## Current State Summary

| Area          | Score                   | Status                                          |
| ------------- | ----------------------- | ----------------------------------------------- |
| Rules Engine  | 4.7/5                   | Excellent - Orchestrator at 100% rollout        |
| Documentation | 5.0/5                   | Excellent - 110+ documents catalogued           |
| Observability | 4.5/5                   | Good - 3 dashboards, k6 framework               |
| Backend       | 4.2/5                   | Good - Full API, WebSocket, AI integration      |
| Frontend      | 3.5/5                   | Medium - Functional but developer-centric       |
| AI Service    | 4.0/5                   | Good - Multiple implementations, service-backed |
| Test Coverage | 69% lines, 53% branches | Good - Target 70%+                              |

**Test Status:**

- TypeScript: 2,987 passing, 0 failing
- Python: 836 passing
- Contract Vectors: 54 cases, 0 mismatches

---

## Wave 7 - Production Validation ✅ COMPLETE (2025-12-03)

**Goal:** Validate system readiness for production deployment

### 7.1 - Load Test Execution ✅ COMPLETE

- [x] Run game-creation scenario (p95=13ms, p99=19ms)
- [x] Run concurrent-games scenario (latency passed, gauge threshold needs adjustment)
- [x] Run player-moves scenario (all thresholds passed)
- [x] Run websocket-stress scenario (500 connections, 15-min test, all passed)

### 7.2 - Baseline Metrics ✅ COMPLETE

- [x] Establish latency baselines (10-25ms p95 across scenarios)
- [x] Document capacity model (500+ WebSocket connections confirmed)
- [x] Create LOAD_TEST_BASELINE.md

### 7.3 - Operational Drills ✅ COMPLETE (2025-12-03)

- [x] Secrets rotation drill (JWT, database credentials) - VERIFIED
- [x] Backup/restore procedure verification - VERIFIED (11MB backup, row counts match)
- [x] Incident response simulation - VERIFIED (AI service outage, graceful degradation)
- [x] Redis failover testing - VERIFIED (finding: app restart required after Redis recovery)

**Results documented in:** `docs/runbooks/OPERATIONAL_DRILLS_RESULTS_2025_12_03.md`

### 7.4 - Production Preview ✅ COMPLETE (2025-12-03)

- [x] Deploy staging configuration validated (requires production build)
- [x] Run smoke tests - ALL PASSED (health, auth, AI, Prometheus, Grafana)
- [x] Load test verification - 100% success rate, p95=17ms
- [x] Document deployment runbook - CREATED

**Deployment runbook:** `docs/runbooks/PRODUCTION_DEPLOYMENT_RUNBOOK.md`

**Wave 7 Summary:** Production validation complete. System ready for staging/production deployment.

---

## Wave 8 - Branch Coverage & Test Quality

**Goal:** Increase test coverage from 53% to 70%+ branches

### 8.1 - Coverage Analysis

- [ ] Generate detailed coverage report by module
- [ ] Identify top 20 files with lowest branch coverage
- [ ] Map uncovered branches to specific test cases needed

### 8.2 - Rules Engine Coverage

- [ ] Add tests for remaining uncovered branches in `/src/shared/engine/`
- [ ] Focus on edge cases in territory processing
- [ ] Cover all paths in chain capture logic
- [ ] Add hex board edge case coverage

### 8.3 - Backend Coverage

- [ ] WebSocket error handling paths
- [ ] Authentication edge cases
- [ ] Rate limiting boundary conditions
- [ ] Session management edge cases

### 8.4 - Frontend Coverage

- [ ] Component error states
- [ ] Loading state transitions
- [ ] Context provider edge cases
- [ ] Hook cleanup and unmount scenarios

**Target Metrics:**

- Branch coverage: 53% → 70%
- Line coverage: 69% → 80%
- Add ~1,780 new branch paths covered

**Estimated Effort:** 3-5 days

---

## Wave 9 - AI Ladder Wiring & Optimization

**Goal:** Enable full AI difficulty spectrum from Random to MCTS

### 9.1 - Minimax/MCTS Production Enablement

- [ ] Wire MinimaxAI for difficulties 6-7
- [ ] Wire MCTS for difficulties 8-9
- [ ] Add difficulty-specific configuration profiles
- [ ] Benchmark AI response times per difficulty

### 9.2 - Service-Backed Choices Completion

- [ ] Complete `line_order` choice service backing
- [ ] Complete `capture_direction` choice service backing
- [ ] Add timeout handling for AI choice requests
- [ ] Implement graceful fallback on service errors

### 9.3 - Heuristic Weight Optimization

- [ ] Run CMA-ES optimization with current game pool
- [ ] Export optimized weight profiles
- [ ] A/B test optimized vs baseline heuristics
- [ ] Document weight tuning process

### 9.4 - AI Observability

- [ ] Add per-difficulty latency metrics
- [ ] Add AI quality metrics (move evaluation scores)
- [ ] Create AI performance dashboard
- [ ] Add AI error rate tracking

**Estimated Effort:** 5-7 days

---

## Wave 10 - Player Experience & UX Polish

**Goal:** Transform developer-oriented UI into player-friendly experience

### 10.1 - First-Time Player Experience

- [ ] Redesign HUD visual hierarchy for clarity
- [ ] Add contextual tooltips for game elements
- [ ] Create interactive tutorial mode
- [ ] Add phase-specific help overlays

### 10.2 - Game Flow Polish

- [ ] Add phase transition animations
- [ ] Improve invalid-move feedback (visual + audio)
- [ ] Add move confirmation for irreversible actions
- [ ] Enhance victory/defeat screens

### 10.3 - Spectator & Analysis

- [ ] Add move annotation system
- [ ] Create post-game analysis view
- [ ] Add teaching overlays for key moments
- [ ] Implement game timeline scrubbing

### 10.4 - Accessibility

- [ ] Full keyboard navigation
- [ ] Screen reader support (ARIA labels)
- [ ] High-contrast mode
- [ ] Reduced motion mode

### 10.5 - Mobile & Responsive

- [ ] Touch-friendly board interaction
- [ ] Responsive layout for tablets
- [ ] Mobile-optimized game controls
- [ ] Portrait mode support

**Estimated Effort:** 10-14 days

---

## Wave 11 - Game Records & Replay System

**Goal:** Comprehensive game storage and replay functionality

### 11.1 - Storage Infrastructure

- [ ] Finalize game record types (TS + Python)
- [ ] Implement game serialization to database
- [ ] Add move history persistence
- [ ] Create game query API endpoints

### 11.2 - Algebraic Notation (RRN)

- [ ] Design RingRift Notation (RRN) format
- [ ] Implement RRN generator
- [ ] Implement RRN parser
- [ ] Add notation export/import UI

### 11.3 - Replay System

- [ ] Build replay player component
- [ ] Add playback controls (play/pause/step/speed)
- [ ] Implement position seeking
- [ ] Add annotation overlay during replay

### 11.4 - Training Data Export

- [ ] Design JSONL training data format
- [ ] Add self-play game recording
- [ ] Implement batch export utility
- [ ] Create data validation tools

**Estimated Effort:** 7-10 days

---

## Wave 12 - Matchmaking & Ratings

**Goal:** Implement player matchmaking and rating system

### 12.1 - Rating System

- [ ] Implement Elo/Glicko-2 rating calculation
- [ ] Add rating persistence to database
- [ ] Create rating update on game completion
- [ ] Build rating history tracking

### 12.2 - Matchmaking Queue

- [ ] Design matchmaking queue system
- [ ] Implement rating-based matching
- [ ] Add queue timeout handling
- [ ] Create queue status UI

### 12.3 - Leaderboards

- [ ] Build leaderboard API endpoints
- [ ] Create leaderboard UI component
- [ ] Add filtering (by time period, board type)
- [ ] Implement pagination

**Estimated Effort:** 5-7 days

---

## Wave 13 - Multi-Player (3-4 Players)

**Goal:** Full support for 3 and 4 player games

### 13.1 - Rules Verification

- [ ] Verify all rules for 3-4 player games
- [ ] Add contract vectors for multi-player scenarios
- [ ] Test victory conditions with multiple players
- [ ] Verify territory calculations

### 13.2 - UI Adaptations

- [ ] Update board rendering for additional players
- [ ] Adapt HUD for multiple opponents
- [ ] Add player order visualization
- [ ] Update victory/elimination screens

### 13.3 - AI for Multi-Player

- [ ] Extend AI evaluation for multi-player
- [ ] Add coalition/threat assessment
- [ ] Tune heuristics for multi-player dynamics
- [ ] Test AI performance in 3-4 player games

**Estimated Effort:** 5-7 days

---

## Priority Order & Dependencies

```
Wave 7.3-7.4 (Production Validation)
    │
    ▼
Wave 8 (Branch Coverage)
    │
    ├──────────────────┐
    ▼                  ▼
Wave 9 (AI)        Wave 10 (UX)
    │                  │
    └──────┬───────────┘
           ▼
    Wave 11 (Replay)
           │
           ▼
    Wave 12 (Matchmaking)
           │
           ▼
    Wave 13 (Multi-Player)
```

## Quick Start Options

Choose based on your priorities:

1. **Production-First:** Wave 7 → Wave 8 → Deploy
2. **AI-Focused:** Wave 7 → Wave 9 → Wave 11
3. **User-Focused:** Wave 7 → Wave 10 → Wave 12
4. **Quality-First:** Wave 7 → Wave 8 → Wave 10

## Next Session Recommendations

For immediate continuation, select one of:

| Option | Wave      | Focus                | Effort   |
| ------ | --------- | -------------------- | -------- |
| 1      | Wave 7.3  | Operational drills   | 1-2 days |
| 2      | Wave 8.1  | Coverage analysis    | 1 day    |
| 3      | Wave 9.1  | AI ladder wiring     | 2-3 days |
| 4      | Wave 10.1 | First-time player UX | 3-4 days |

---

**Document Maintainer:** Claude Code
**Last Updated:** December 3, 2025
