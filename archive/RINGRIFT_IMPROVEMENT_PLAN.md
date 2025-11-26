# ⚠️ DEPRECATED: RingRift Improvement & Implementation Plan

> **⚠️ HISTORICAL DOCUMENT. For current status and plans, see `TODO.md` and `STRATEGIC_ROADMAP.md`.**
>
> **This is a historical document preserved for context.**
>
> **For current status and plans, see:**
>
> - [`CURRENT_STATE_ASSESSMENT.md`](../CURRENT_STATE_ASSESSMENT.md)
> - [`IMPLEMENTATION_STATUS.md`](../IMPLEMENTATION_STATUS.md)
> - [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md)
> - [`TODO.md`](../TODO.md)

**Document Version:** 3.0 (CODE-VERIFIED & ALIGNED)

**Created:** November 13, 2025  
**Updated:** November 13, 2025  
**Author:** Codebase Analysis & Planning System  
**Status:** Aligned with current code and documentation

---

## 0. How to Use This Plan

This document is the **high-level improvement plan** for RingRift. For detailed status and task tracking, see:

- **Current factual status:** `CURRENT_STATE_ASSESSMENT.md`
- **Strategic phases and milestones:** `STRATEGIC_ROADMAP.md`
- **Granular tasks and progress:** `TODO.md`
- **Issue list and priorities:** `KNOWN_ISSUES.md`
- **Codebase evaluation:** `CODEBASE_EVALUATION.md`

Those files collectively represent the single source of truth. This plan summarizes **what to do next** to turn the current engine into a fully playable, AI-capable game, focusing on **sustainable architecture** and **rule fidelity**.

> **Note:** Earlier versions of this document described the marker system, BoardState, movement validation, phase transitions, and territory disconnection as “not implemented.” Those systems are now implemented and generally aligned with `ringrift_complete_rules.md`. The critical gaps are now about **player choice, chain enforcement, testing, AI integration, and UI**.

---

## 1. Executive Summary (Updated)

The RingRift project has:

- ✅ **Strong foundations** – clean TypeScript architecture, shared types, Python AI microservice, Docker/CI, and excellent rules documentation.
- ✅ **Core mechanics implemented** – marker system, unified movement rules, overtaking captures, line detection/collapse, territory disconnection, phase transitions, forced elimination, hex board.
- ✅ **Refactoring Complete** – `src/shared/engine/` refactor is done; `ai-service/app/rules/` shell is established.
- ⚠️ **Critical gaps remaining**:
  - **Python Validator/Mutator Implementation** (P0) – Implementing concrete Python logic to match TypeScript.
  - **Player choice system** for all rules-mandated decisions (P0).
  - **Chain capture enforcement** (mandatory continuation + direction choice) (P0).
  - **Comprehensive tests** for engine & rules (P1).
  - **Python AI integration** into the turn loop and choice system (P1).
  - **Minimal playable UI** (board + interaction) (P1).

**Goal:** Reach a **single-player, rules-faithful, test-backed MVP** in ~8–12 weeks of focused work, while preserving the Python AI microservice for future ML/self-play.

---

## 2. Updated Key Findings

### 2.1 Strengths

- **Architecture:**
  - TypeScript-first monolith with clear separation: BoardManager, RuleEngine, GameEngine.
  - Shared types in `src/shared/types` prevent drift between client/server.
  - Hexagonal and square board support are unified under `BOARD_CONFIGS`.

- **Rules implementation:**
  - Unified movement & landing rules (non-capture & capture) implemented and now reflected in the rules doc.
  - Marker mechanics, line formation/collapse, territory disconnection, forced elimination, and victory conditions are present and mostly correct.

- **Infrastructure:**
  - PostgreSQL + Prisma schema, Redis cache client, Socket.IO server, Python FastAPI AI service, and `AIServiceClient` are all in place.
  - CI runs linting, type-checking, tests with coverage, and Docker builds.

### 2.2 Critical Gaps (What’s missing now)

1. **Player choice system (P0)** – all multiple-choice decisions are auto-resolved instead of being chosen by the player or AI.
2. **Chain capture enforcement (P0)** – chain Overtaking is not fully enforced; direction choices are not surfaced.
3. **Testing (P1)** – BoardManager has tests; RuleEngine/GameEngine have minimal coverage; scenario tests from rules/FAQ are missing.
4. **AI integration (P1)** – Python AI is not invoked as part of any actual game turn.
5. **UI (P1)** – no board render or move/choice UI, so humans cannot play.

---

## 3. Phased Improvement Plan (Aligned with STRATEGIC_ROADMAP)

This plan mirrors `STRATEGIC_ROADMAP.md`, but emphasizes the **current state** and the **most important improvements**.

### Phase 0 – Testing & Quality Foundation (1–2 weeks)

**Goals:** Make it safe to change the engine and refactor.

**Key actions:**

- Confirm Jest setup (`jest.config.js`, `tests/*`) and add convenience scripts (`test:watch`, `test:coverage`) if missing.
- Expand existing tests for `BoardManager` to cover:
  - Markers (set/flip/collapse/remove).
  - Line detection across all board types.
  - Territory disconnection detection for basic patterns.
- Add initial RuleEngine tests for:
  - Valid/invalid moves (movement and captures).
  - Unified landing rules and marker interactions.

**Success criteria:**

- Tests run reliably in CI.
- First set of rule-driven tests exist and pass.

---

### Phase 1 – Player Choice System & Chain Captures (2–3 weeks)

**Goals:** Fully encode player agency and mandatory chains per the rules.

#### 1.1 PlayerInteractionManager & Choice Types (P0)

**Design:**

- Add shared types in `src/shared/types/game.ts`, e.g.:

```ts
export interface PlayerChoice<T> {
  id: string;
  type:
    | 'line_order'
    | 'line_reward_option'
    | 'ring_elimination'
    | 'region_order'
    | 'capture_direction';
  player: number;
  prompt: string;
  options: T[];
  timeoutMs?: number;
  defaultOption?: T;
}

export interface PlayerChoiceResponse<T> {
  choiceId: string;
  selectedOption: T;
}
```

- Implement a **PlayerInteractionManager** (server-side) that:
  - Exposes `async requestChoice<T>(choice: PlayerChoice<T>): Promise<T>`.
  - Internally, emits events (via WebSocket or callback) without coupling GameEngine to transport/UI.
  - Handles timeouts/defaults.

**Integrate into GameEngine at all choice points:**

- Line processing order & Option 1 vs Option 2 (Section 11.2–11.3).
- Which ring/stack cap to eliminate when required.
- Which disconnected region to process first (Section 12.2–12.3).
- Which capture segment to follow when multiple captures are legal (Section 10.3).

#### 1.2 Chain Capture Enforcement (P0)

**Work:**

- Track chain capture state in GameEngine (e.g. current capturing stack + mustCapture flag).
- After each capture:
  - Query RuleEngine for possible follow-up captures from the new landing spot.
  - If any exist, require another capture.
  - Use PlayerInteractionManager to pick the segment when multiple choices exist.

**Tests:**

- Encode FAQ chain examples (Q14, 15.3.1, 15.3.2) and verify behaviour.

**Success criteria (Phase 1):**

- All rules-mandated choices are surfaced and resolved through a single PlayerInteraction system.
- Chain captures continue until no legal captures remain, with direction choices validated by tests.

---

### Phase 2 – Minimal Playable UI (2–3 weeks)

**Goals:** Humans can play a full game locally against another human, using the engine as the source of truth.

**Frontend geometry & movement grid foundation:** The React board layer now includes a reusable movement-grid overlay in `BoardView` (toggled via the `showMovementGrid` prop) and the shared `computeBoardMovementGrid(board: BoardState)` helper in `src/client/utils/boardMovementGrid.ts`. Future UI features—such as valid-move highlighting, interaction paths in the sandbox harness, AI path visualizations, and history playback overlays—should reuse this normalized-center geometry rather than re-deriving per-component coordinates.

**Shared engine core for rules helpers:** A small but growing set of pure, browser-safe engine helpers now lives in `src/shared/engine/core.ts` (for example, `calculateCapHeight`, `getPathPositions`, and `getMovementDirectionsForBoardType`). Both the Node.js GameEngine and any client-side/local harnesses (such as the `/sandbox` mode) should prefer these shared helpers rather than re-implementing geometric or cap/stack logic in isolation.

**Key UI components:**

- Board components:
  - `SquareBoard` (8×8, 19×19) and `HexBoard` (hex), built on the existing `BoardView` and
    the shared `computeBoardMovementGrid(board: BoardState)` geometry helper.
  - `Cell` / `HexCell` with coordinates and click handlers.
- Piece visualization:
  - Ring stacks (including cap vs total height cues).
  - Markers and collapsed spaces.
- Interaction & HUD:
  - Click-to-select source and destination.
  - Display of valid moves (ideally using RuleEngine.getValidMoves), rendered as overlays
    that reuse the same normalized centers used by the faint movement grid.
  - Dialogs for PlayerChoice prompts (line options, region order, elimination, capture direction).
  - Current player, ring counts, territory counts, victory progress.

**Success criteria:**

- A local 2-player game can be played end-to-end on at least one board type (e.g. 8×8) using the UI.
- All choices required by rules can be made via the UI.

---

### Phase 3 – Python AI Integration (2–3 weeks)

**Goals:** Single-player vs AI mode using the existing Python AI microservice.

**Key tasks:**

- **Implement Python Validators/Mutators:** Fill in the `ai-service/app/rules/` shell with concrete logic matching `src/shared/engine/`.
- Extend `Player` type in `src/shared/types/game.ts` to include:
  - `type: 'human' | 'ai'`
  - AI config (difficulty level, AIType).
- In the server’s turn loop:
  - When it is an AI player’s turn, call `AIServiceClient.getAIMove(gameState, playerNumber, difficulty, aiType)`.
  - Validate the returned move with RuleEngine.validateMove.
  - Apply via GameEngine.makeMove().
- Define how AI participates in PlayerChoice decisions:
  - Either by adding choice endpoints to the Python service, or by implementing simple heuristics in TypeScript for early AI levels.

**Success criteria:**

- A human can play a complete game vs AI.
- AI moves are always legal, and the game engine, not the AI service, is the authority.
- Python service can independently validate moves and simulate states for advanced search.

---

### Phase 4 – Testing & Scenario Validation (1–2 weeks)

**Goals:** High confidence that the implementation matches the rules.

**Key actions:**

- Add unit tests for all engine subsystems (BoardManager, RuleEngine, GameEngine).
- Implement scenario tests directly from `ringrift_complete_rules.md` and FAQs Q1–Q24.
- Add performance/regression tests for AI integration.

**Success criteria:**

- 80–90% coverage on game logic modules.
- All documented rule and FAQ scenarios pass.

---

### Phase 5 – Multiplayer, Persistence, and Extras (future)

**Goals:** Bring online play and persistence to parity with the core game.

**Focus areas:**

- WebSocket events and synchronization.
- Database persistence for games/moves/users.
- Spectator mode, replays, rating system.

These are already laid out in more detail in `STRATEGIC_ROADMAP.md` and should follow once Phases 0–3 are stable.

---

## 4. Alignment with Current Code & Docs

This plan assumes and depends on the following being **already true** (and they are, per the latest code and assessments):

- BoardState uses `stacks`, `markers`, and `collapsedSpaces` correctly.
- Marker system, movement, captures, line formation/collapse, territory disconnection, phase transitions, and forced elimination are implemented and aligned with the updated `ringrift_complete_rules.md` (including unified landing rules and the refined Last Player Standing wording).
- `KNOWN_ISSUES.md` now:
  - Treats player choice system and chain capture enforcement as the main P0 issues.
  - Treats AI integration, tests, and UI as P1.
- `CODEBASE_EVALUATION.md` describes the real state of the code (≈75% core logic complete, infra excellent, AI ready but not wired).

This document is therefore focused on **closing the remaining gap between a strong foundation and a fully playable, AI-capable, rule-faithful game**.

---

## 5. Recommended Immediate Next Steps

If you want a concrete “next week” plan:

1. **Start PlayerInteractionManager design** (Phase 1.1):
   - Define shared `PlayerChoice` types.
   - Sketch the async request/response contract for server ↔ client/AI.

2. **Implement chain capture enforcement hooks** (Phase 1.2):
   - Add chain state to GameEngine.
   - Use existing RuleEngine capture validation to detect follow-ups.

3. **Add 2–3 critical scenario tests**:
   - A simple chain capture with multiple directions.
   - A long line giving Option 1 vs Option 2.
   - A simple disconnection example from the rules.

Once those are in place, the next logical chunk is the minimal board UI (Phase 2) plus AI integration (Phase 3), as outlined above.

---

**Summary:** The improvement plan is now aligned with the real codebase and the updated documentation. The work ahead is clear: **Player choices → Chain enforcement → Tests → UI → AI integration**, with multiplayer and persistence following once the core experience is solid.
