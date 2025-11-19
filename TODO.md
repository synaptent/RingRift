# RingRift Development TODO

**Last Updated:** November 18, 2025
**Current Phase:** Phase 2 - Robustness & Testing
**Overall Progress:** Phase 1 Complete, Phase 2 Started

---

## 🚀 PHASE 1: Core Game Logic Implementation (COMPLETED) ✅

**Status:** COMPLETED
**Achievements:**

- ✅ Board Management (8x8, 19x19, Hex)
- ✅ Game Engine (Turns, Phases, Victory)
- ✅ Rules Engine (Movement, Capture, Lines, Territory)
- ✅ AI Integration (Service + Fallback)
- ✅ Frontend Basics (Lobby, Board, HUD, Victory)

---

## 🧪 PHASE 2: Robustness & Testing (IN PROGRESS)

**Priority:** P0 - CRITICAL
**Goal:** Ensure 100% rule compliance and stability

### Task 2.1: Comprehensive Scenario Testing

- [ ] Build test matrix for all FAQ edge cases
- [ ] Implement scenario tests for complex chain captures
- [ ] Verify all board types (especially Hexagonal edge cases)

### Task 2.2: Sandbox Stage 2

- [x] Stabilize client-local sandbox with unified “place then move” turn semantics
      for both human and AI seats (including mixed games), and automatic local AI
      turns when it is an AI player’s move. Covered by `ClientSandboxEngine`
      and `/sandbox` wiring plus `ClientSandboxEngine.mixedPlayers` tests.
- [ ] Harden backend ↔ sandbox parity and AI-vs-AI termination behaviour
      (see P0.2 / P1.4 in KNOWN_ISSUES and the sandbox AI simulation diagnostics
      in `ClientSandboxEngine.aiSimulation`).

---

## 🎨 PHASE 3: Multiplayer Polish

**Priority:** P1 - HIGH
**Goal:** Seamless online experience

### Task 3.1: Spectator Mode

- [ ] UI for watching active games
- [ ] Real-time updates for spectators

### Task 3.2: Social Features

- [ ] In-game chat
- [ ] User profiles and stats
- [ ] Leaderboards

### Task 3.3: Matchmaking

- [ ] Automated matchmaking queue
- [ ] ELO-based matching

---

## 🤖 PHASE 4: Advanced AI

**Priority:** P2 - MEDIUM
**Goal:** Challenging opponents for all skill levels

### Task 4.1: Machine Learning

- [ ] Train neural network models
- [ ] Deploy advanced models to Python service

### Task 4.2: Advanced Heuristics

- [ ] Implement MCTS/Minimax for intermediate levels
