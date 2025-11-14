# Known Issues & Bugs

**Last Updated:** November 13, 2025  
**Status:** Code-verified assessment based on actual implementation  
**Related Documents:** [CURRENT_STATE_ASSESSMENT.md](./CURRENT_STATE_ASSESSMENT.md) · [TODO.md](./TODO.md) · [STRATEGIC_ROADMAP.md](./STRATEGIC_ROADMAP.md) · [CODEBASE_EVALUATION.md](./CODEBASE_EVALUATION.md)

This document tracks **current, code-verified issues** in the RingRift codebase. Older issues that described the marker system, BoardState, movement validation, phase transitions, and territory disconnection as “not implemented” have been superseded: those systems are now implemented and generally aligned with the rules. The remaining P0/P1 issues are about **player agency, chain enforcement, integration, and tests**, not basic mechanics.

**Verification Basis:**
- `src/server/game/{BoardManager,RuleEngine,GameEngine}.ts`
- `src/shared/types/game.ts`
- `ai-service/` (Python AI) + `src/server/services/AIServiceClient.ts`
- `CURRENT_STATE_ASSESSMENT.md`, `TODO.md`, `STRATEGIC_ROADMAP.md`, `CODEBASE_EVALUATION.md`

---

## 🔴 Critical Issues (Prevent Core-Correct Gameplay)

### Issue #0: Player Choice System Not Fully Integrated End-to-End
**Priority:** P0 – CRITICAL  
**Component:** GameEngine.ts, PlayerInteractionManager.ts, WebSocket/UI/AI integration  
**Status:** Engine-level integration COMPLETE; transport/UI/AI integration NOT STARTED  
**Severity:** BLOCKS FULL STRATEGIC GAMEPLAY

**Description**
The engine now exposes a generic mechanism for **player choices** at key decision points, using shared `PlayerChoice` types plus `PlayerInteractionManager`. GameEngine already calls this abstraction for line order, line reward, ring/cap elimination, region order, and capture direction during chains. However, there is still no concrete WebSocket/UI/AI plumbing, so choices cannot yet be surfaced to humans or AI; until that is implemented, the system effectively behaves as if choices were defaulted.

**Places where choices are required by rules:**
1. **Line processing order:** When multiple lines form, the moving player must choose which line to process first (Section 11.3).  
2. **Graduated line rewards:** For long lines, the player must choose Option 1 (collapse all + eliminate ring/cap) vs Option 2 (collapse only required segment, no elimination) (Section 11.2).  
3. **Ring/cap elimination target:** When required to eliminate one of your rings or a cap, you must choose which stack/ring (Sections 11.2, 12.2, 12.3).  
4. **Region processing order:** When multiple disconnected regions exist, the moving player chooses which region to process first (Sections 12.2–12.3).  
5. **Capture direction:** When multiple overtaking capture directions are available during a chain, the moving player chooses which legal segment to perform next (Section 10.3, FAQ Q14).

**Current behaviour in code (simplified):**
- First line found is processed.
- Long lines always use Option 2 (preserve rings, collapse only required segment).
- Eliminations always target the first eligible stack.
- Disconnected regions processed in first-found order.
- Capture continuation and direction choice are not surfaced through any interaction system.

**Impact:**
- Strategic agency is effectively removed for both humans and AI.
- Engine cannot be considered a faithful implementation of the rules until this is fixed.

**Required solution (high level):**
- Introduce a **PlayerInteractionManager** (server-side abstraction) and shared `PlayerChoice<T>` types that:
  - Emit choice requests (with IDs, prompts, options) without knowing about transport (HTTP/WebSocket/UI).  
  - Await responses from either a human (via UI) or AI (via AIServiceClient).  
  - Provide an async API (`await getPlayerChoice(...)`) that GameEngine can use at all choice points.
- Add corresponding client components (e.g. `PlayerChoiceDialog`) and AI hooks for answering choices.

**Rule references:** Sections 4.5, 10.3, 11.2, 11.3, 12.2–12.3; FAQ Q7, Q14, Q15.

---

### Issue #1: Chain Captures Not Fully Enforced
**Priority:** P0 – CRITICAL  
**Component:** GameEngine.ts, RuleEngine.ts  
**Status:** Engine-level continuation ENFORCED (~70%); tests and UI/AI integration missing

**Description**
Single overtaking captures are validated and applied correctly (including cap height comparisons, stack updates, and landing rules). However, the engine does **not fully enforce mandatory chain capture continuation** while legal captures remain, and it doesn’t surface **choice of capture direction** to the player.

**Expected behaviour (per rules):**
- Once a player begins an overtaking capture sequence:
  - If *any* legal capture segment exists from the current landing space, the player **must** perform one of them (mandatory continuation).  
  - When multiple choices exist, the player chooses *which* capture segment to perform (strategic chain ending is allowed by choosing a path that leads to no further captures).  
- 180° reversal and cyclic patterns must be legal when consistent with movement rules; chains must end only when no legal captures remain.

**Current behaviour:**
- Capturing moves are validated in isolation and can be applied.
- There is no robust chain state in GameEngine that:
  - Forces additional captures when available.  
  - Invokes the (still-missing) PlayerInteractionManager to pick among multiple capture directions.  
- As a result, a player may be able to stop early, even if further legal captures are available.

**Impact:**
- Violates the mandatory chain capture rule (Section 10.3, FAQ Q14).
- Tactical sequences and examples from the rules/FAQ (e.g. 180° and cyclic patterns) are not fully enforceable.

**Required solution:**
- Add explicit chain state to GameEngine and integrate with RuleEngine’s capture validation:
  - After each capture, recompute available captures for that stack.  
  - If any exist, require a further capture (unless no legal segments exist).  
  - Use PlayerInteractionManager to choose capture direction when multiple options exist.  
- Add scenario tests for 180° reversal and cyclic patterns (FAQ 15.3.x).

**Rule references:** Section 10.3; FAQ Q14, Q15.3.1, Q15.3.2.

---

## 🟡 High Priority Issues (Major Gaps / Integration)

### Issue #2: Limited Test Coverage for Engine & Rules
**Priority:** P1 – HIGH  
**Component:** Tests (server + shared)  
**Status:** Incomplete

**Description:**
- Jest is configured and CI runs tests with coverage, and there are some unit tests (e.g. BoardManager position & adjacency).  
- However, **core engine modules (GameEngine, RuleEngine) lack comprehensive tests**, and there are no scenario tests based on the rules & FAQ.

**Impact:**
- Refactoring or extending engine behaviour is risky.  
- There is no automated proof that all rules (especially complex edge cases) are correctly implemented.

**Required solution:**
- Add unit tests for:
  - Movement and path validation, including unified landing rules and marker interactions.  
  - Capture validation (non-capture vs overtaking, distance rules, landing rules).  
  - Line detection and collapse, including graduated rewards and interactions between multiple lines.  
  - Territory disconnection, including representation and the self-elimination prerequisite.  
  - Phase transitions and forced elimination.
- Add scenario tests directly derived from `ringrift_complete_rules.md` and FAQs (Q1–Q24), especially for chain captures and territory disconnection.

---

### Issue #3: AI Integration into the Game Loop Missing
**Priority:** P1 – HIGH  
**Component:** GameEngine.ts, routes, AIServiceClient.ts, ai-service  
**Status:** AI service & client exist, not wired into turn flow

**Description:**
- The Python AI microservice (`ai-service/`) and the TypeScript client (`AIServiceClient`) are implemented and tested in isolation.  
- There is no end-to-end integration that:
  - Marks a player as `type: 'ai'` with difficulty/AI type config in GameState.  
  - Detects an AI player’s turn and calls `AIServiceClient.getAIMove(...)`.  
  - Validates and applies the AI’s move via RuleEngine/GameEngine.  
  - Integrates AI decisions into the eventual PlayerInteractionManager for choices.

**Impact:**
- Single-player and AI-vs-AI modes are not yet available, despite infrastructure being present.

**Required solution:**
- Extend player types to include AI config.  
- Integrate AIServiceClient into the game loop so that when it’s an AI player's turn, the server requests a move from the AI service, validates it, and applies it.  
- Define how AI will answer PlayerChoice prompts (either via additional AI endpoints or simple heuristics for early levels).

---

### Issue #4: No Playable Board/UI Yet
**Priority:** P1 – HIGH  
**Component:** Client (React), WebSocket flows  
**Status:** App shell present, no game UI

**Description:**
- The React client has routing, layout, auth, and some pages, but:
  - No board rendering (8×8, 19×19, or hex).  
  - No ring/stack/marker/collapsed-space visualization.  
  - No click-to-move, no choice dialogs, no game state panel.

**Impact:**
- Humans cannot actually play or visually inspect game states.  
- All validation currently relies on manual testing or future automated tests.

**Required solution:**
- Implement minimal board UI and interaction as described in `STRATEGIC_ROADMAP.md` (Phase 2): board components, move selection, and PlayerChoice dialogs.

---

## 🟢 Medium Priority Issues (Integration / UX / Infrastructure)

### Issue #5: WebSocket Game Events Incomplete
**Priority:** P2 – MEDIUM  
**Component:** `src/server/websocket/server.ts`, client  
**Status:** Basic server wiring only

**Description:**
- Socket.IO is set up, but game-specific events (join game, broadcast moves, game state updates, spectator events) are only partially implemented or missing.

**Impact:**
- Real-time multiplayer and spectators cannot function as designed.

**Required solution:**
- Define and implement core events:
  - `join_game`, `leave_game`, `player_move`, `game_state_update`, `game_ended`, `spectator_joined`, `spectator_left`.  
- Ensure events are consistent with shared WebSocket types in `src/shared/types/websocket.ts`.

---

### Issue #6: Database Integration Incomplete
**Priority:** P2 – MEDIUM  
**Component:** Prisma models, services  
**Status:** Schema defined, services incomplete

**Description:**
- `prisma/schema.prisma` defines models for users, games, moves, etc.  
- GameEngine is not yet wired to persist game states and move histories to the database.

**Impact:**
- No persistent game history, leaderboards, or replay data.

**Required solution:**
- Implement services that:
  - Create/update game records as games start and progress.  
  - Persist move history.  
  - Expose APIs for listing games, loading replays, and leaderboard data.

---

## ✅ Resolved or No Longer Current Issues (for historical context)

These issues were present in earlier versions and are now resolved or substantially completed. They remain here for historical context but should not guide future work.

### (Historical) Marker System Not Implemented
- **Status:** ✅ RESOLVED  
- Marker CRUD (`setMarker`, `flipMarker`, `collapseMarker`, etc.), path processing, and landing behaviour are implemented in `BoardManager` and `GameEngine` and match the rules.

### (Historical) BoardState Missing Collapsed Spaces
- **Status:** ✅ RESOLVED  
- BoardState now has explicit `stacks`, `markers`, and `collapsedSpaces` maps, and all current engine code uses this structure.

### (Historical) Movement Validation Incomplete
- **Status:** ✅ RESOLVED (Implementation), tests still needed  
- Movement validation now enforces minimum distance, path blocking, and unified landing rules; remaining work is adding thorough tests.

### (Historical) Territory Disconnection Not Implemented
- **Status:** ✅ RESOLVED (Implementation), tests still needed  
- Territory detection, disconnection analysis, and processing (including self-elimination prerequisite) are implemented in `BoardManager` and `GameEngine` and align with Section 12 and FAQ Q15.

### (Historical) Phase Transitions Incorrect
- **Status:** ✅ RESOLVED  
- Game phases now follow the documented sequence: `ring_placement → movement → capture → line_processing → territory_processing → next player`, including forced elimination when blocked.

### (Historical) Forced Elimination Not Implemented
- **Status:** ✅ RESOLVED  
- GameEngine checks at the beginning of a turn if a player has no legal placements/moves/captures but controls stacks, and forces elimination of a cap accordingly (Section 4.4).

### (Historical) Player State Not Updated Correctly
- **Status:** ✅ RESOLVED  
- `ringsInHand`, `eliminatedRings`, and `territorySpaces` are now updated consistently in GameEngine.

---

## 📋 Summary & Fix Order (Updated)

**Recommended order going forward:**

1. **P0 – Core correctness & agency**  
   1. Implement PlayerInteractionManager + PlayerChoice types (Issue #0).  
   2. Enforce full chain capture behaviour with choice of direction (Issue #1).  

2. **P1 – Confidence & AI/UX**  
   3. Expand test coverage for GameEngine/RuleEngine/BoardManager; add scenario tests (Issue #2).  
   4. Integrate Python AI service into the game loop and choice system (Issue #3).  
   5. Implement minimal game UI (Issue #4).

3. **P2 – Multiplayer & persistence**  
   6. Flesh out WebSocket events (Issue #5).  
   7. Wire GameEngine/GameState to database models (Issue #6).

Once Issues #0–#4 are addressed, the project will have:
- A rules-faithful engine including player choices and mandatory chains.
- A solid test harness validating rules against the official spec.
- A working UI and AI integration for real human vs AI play.
