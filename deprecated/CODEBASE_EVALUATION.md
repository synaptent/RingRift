# RingRift Codebase Evaluation & Development Recommendations

**Evaluation Date:** November 14, 2025  
**Evaluator:** Development Analysis System (code-verified)  
**Repository:** https://github.com/an0mium/RingRift

---

## 📊 Executive Summary

RingRift is a **sophisticated multiplayer strategy game** with:

- **Excellent architecture and documentation**
- **Core game engine ~75% implemented and aligned with the rules**
- **Python AI microservice and TypeScript client in place but not yet integrated into gameplay**
- **Minimal UI and limited tests**, which currently block actual play and confident refactoring

### High-Level Assessment

| Category              | Rating       | Status Overview |
|-----------------------|-------------|-----------------|
| **Documentation**     | ⭐⭐⭐⭐⭐ (A+) | Exceptional game rules & design docs |
| **Architecture**      | ⭐⭐⭐⭐⭐ (A)  | Clean, modular, TypeScript-first |
| **Core Game Logic**   | ⭐⭐⭐½☆ (B+) | ~75% complete; player choice & chain captures missing |
| **Frontend/UI**       | ⭐⭐½☆☆ (C-) | Minimal board/choice UI; backend play works but UX is rough |
| **AI Implementation** | ⭐⭐⭐⭐☆ (B)  | Python service + TS client + AIEngine wired into backend AI turns (moves only) |
| **Testing**           | ⭐⭐☆☆☆ (C-) | Jest + CI configured; focused tests for BoardManager, movement/capture, AI turns, choices, and territory |
| **DevOps/CI**         | ⭐⭐⭐⭐☆ (A-) | GitHub Actions, Docker, env setup in good shape |
| **Overall Readiness** | 🔶 **~60%** | Strong foundation, incomplete execution |

**Key Reality:**
- You can now play backend-driven games through a minimal React UI (BoardView + GamePage + ChoiceDialog) against human and AI opponents, but you **cannot yet be fully confident** all rules are implemented correctly due to low overall test coverage and incomplete scenario coverage for player choice/chain capture mechanics.

---

## 🧱 Architecture & Technology Stack

### Backend (TypeScript / Node.js)

- **Runtime:** Node.js 18+
- **Language:** TypeScript (strict configuration)
- **Framework:** Express.js
- **Game Engine:** `src/server/game/`
  - `GameEngine.ts` – Orchestrates phases, applies moves, manages timers & game state
  - `RuleEngine.ts` – Validates moves, computes valid moves, checks game end conditions
  - `BoardManager.ts` – Board topology, positions, markers, stacks, line & territory analysis
- **Game AI Integration:**
  - `src/server/services/AIServiceClient.ts` – HTTP client for **Python FastAPI AI service** (ai-service)
- **Persistence / Infra:**
  - PostgreSQL via Prisma (`prisma/schema.prisma`)
  - Redis cache (`src/server/cache/redis.ts`)
  - WebSocket server via Socket.IO (`src/server/websocket/server.ts`)
  - Middleware for auth, rate limiting, error handling

### Frontend (React + TypeScript)

- **Build Tool:** Vite
- **Framework:** React 18 with hooks
- **Routing / Pages:** `src/client/pages/*`
- **State:** React Contexts (`AuthContext`, `GameContext`)
- **Styling:** Tailwind CSS
- **HTTP:** Axios (`src/client/services/api.ts`)
- **Current Status:** Shell + layout + basic auth flows; **no game board UI yet**.

### AI Service (Python FastAPI)

- **Location:** `ai-service/`
  - `app/main.py` – FastAPI app
  - `app/ai/random_ai.py` – Random move AI
  - `app/ai/heuristic_ai.py` – Heuristic-based AI
- **Integration Surface:**
  - TypeScript `AIServiceClient` calls `/ai/move`, `/ai/evaluate`, `/health`
- **Status:** Deployable service, but **GameEngine does not yet call AIServiceClient in any game loop or route**.

### DevOps & Tooling

- **CI:** `.github/workflows/ci.yml` – Lint, type-check, tests, build, security scans, Docker build
- **Testing:** `jest.config.js`, `tests/` directory with setup & a couple of unit tests
- **Formatting & Linting:** ESLint, Prettier, Husky pre-commit hook
- **Docker:** `Dockerfile`, `docker-compose.yml` for app + PostgreSQL + Redis + AI service

Overall: **The stack is modern, robust, and production-ready from an infrastructure standpoint.**

---

## ✅ Verified Strengths

### 1. Documentation Quality (A+)

- `ringrift_complete_rules.md` – Detailed, authoritative rules spec
- `RULES_ANALYSIS_PHASE1.md`, `RULES_ANALYSIS_PHASE2.md` – Deep rule breakdowns
- `CURRENT_STATE_ASSESSMENT.md` – Code-verified status snapshot
- `STRATEGIC_ROADMAP.md` – Phased implementation roadmap (already updated to keep Python AI)
- `RINGRIFT_IMPROVEMENT_PLAN.md`, `TODO.md` – Fine-grained tasks and status
- `TECHNICAL_ARCHITECTURE_ANALYSIS.md`, `ARCHITECTURE_ASSESSMENT.md` – Architecture decisions & evaluations

**Impact:** You have a clearer spec and plan than most production projects. Implementation can follow the docs rather than guesswork.

### 2. Core Game Engine Implementation (B+)

Based on `GameEngine.ts`, `RuleEngine.ts`, `BoardManager.ts`, and tests:

**Implemented & Working (per CURRENT_STATE_ASSESSMENT):**

- **Board Manager (~90%)**
  - Position generation: 8x8, 19x19, hexagonal (331 hex cells) ✅
  - Adjacency types: Moore, Von Neumann, Hexagonal ✅
  - Marker model: `markers` map with MarkerInfo, `collapsedSpaces` tracking ✅
  - Stack operations: get/set/remove stacks, player stack queries ✅
  - Line detection: based on markers, not stacks, respecting required line lengths ✅
  - Territory discovery: connected components and per-player territories, for all board types ✅
  - Disconnection detection: regions & borders using collapsed spaces + marker borders ✅

- **Game Engine (~75%)**
  - Move application: ring placement, movement, overtaking capture, stack building ✅
  - Marker behaviour along paths: leave marker on departure, flip/collapse markers, handle landing on own marker ✅
  - Line processing: detection + collapsing + elimination logic **with defaults** (no player choice yet) ✅⚠️
  - Territory disconnection processing: disconnection detection, border collapse, ring elimination, mandatory self-elimination ✅
  - Phase flow: `ring_placement → movement → capture → line_processing → territory_processing → next player`, including forced elimination when blocked ✅
  - Hex board: specialized logic and validations (distances, adjacency, edge detection) ✅

- **Rule Engine (~60%)**
  - Move validation: ring placement, stack movement, overtaking captures ✅
  - Distance rules: stack height minimum for movement, hex distance for hex boards ✅
  - Capture validation: cap height comparisons, straight-line & landing rules, path blocking ✅
  - Valid move generation: basic `getValidRingPlacements`, `getValidStackMovements`, `getValidCaptures` ✅⚠️
  - Game end detection: ring elimination and territory control thresholds ✅

**Major Incomplete Elements:**

- **Player choice system** – All decisions default to first option or hard-coded behaviours:
  - Which line to process when multiple exist
  - Option 1 vs Option 2 for graduated line rewards
  - Which stack/cap to eliminate when required
  - Which disconnected region to process first
  - Which capture direction to follow in chain captures

- **Chain capture enforcement** – Single captures are valid, but:
  - Mandatory continuation of chain captures is not fully enforced in GameEngine
  - Complex patterns (180° reversals, cycles) are not exhaustively tested

### 3. Python AI Microservice + TypeScript Client (C+ but strategic)

- Python service exists (`ai-service/app/`), with:
  - `RandomAI` and `HeuristicAI` classes
  - FastAPI endpoints for move generation and evaluation

- TypeScript AI client (`AIServiceClient.ts`):
  - Configurable base URL (`AI_SERVICE_URL`)
  - `getAIMove`, `evaluatePosition`, `healthCheck`, and cache control methods
  - Logging & error handling via `logger`

**Current Gap:** The actual game loop (GameEngine and routes) does not yet:

- Decide when a player is AI-controlled
- Call `AIServiceClient.getAIMove()` when it is the AI’s turn
- Await AI decisions and inject resulting `Move` into `makeMove`

**Strategic Decision (per user preference):**
- Keep the **Python AI microservice** as the primary AI path, to support future ML and self-play.
- For robustness, we may still implement a simple TypeScript fallback AI (e.g. random/heuristic) for offline/local use or when the AI service is unavailable.

### 4. Tooling & CI (A-)

- `jest.config.js` – TypeScript Jest config, coverage thresholds set (80%)
- `tests/` – Environment setup + initial unit tests for BoardManager and board position utilities
- `.github/workflows/ci.yml` – Lint, type-check, tests with coverage + Codecov, build & Docker test
- `.husky/pre-commit` – Git hooks for lint/format enforcement

The pipeline is strong; the **missing piece is test volume and coverage**, not infrastructure.

---

## 🔴 Key Gaps & Risks

### 1. Missing Player Choice System (CRITICAL)

Architectural gap: **no generic player interaction mechanism**. GameEngine currently hard-codes choices:

- Processes the **first** line found instead of letting player choose
- Always uses **Option 2** for long lines (no elimination)
- Eliminates from the **first stack** instead of letting the player choose
- Processes the **first disconnected region** rather than player-defined order
- Has no interactive mechanism to choose capture directions when multiple chain options exist

**Consequence:**
- Rules are structurally implemented but **strategic agency is missing**.
- You can’t claim full rules compliance.
- AI cannot be meaningfully strategic without a unified way to decide choices.

**Needed:**

- A `PlayerInteractionManager` or similar abstraction that:
  - Emits choice requests (with IDs, prompts, options)
  - Collects responses from either the UI (human) or AIServiceClient (AI)
  - Integrates smoothly into turn/phase flow without making GameEngine depend on network/UI directly

### 2. Chain Captures Not Fully Enforced (CRITICAL)

- RuleEngine has logic for validating capture moves and hints of chain reaction processing.
- GameEngine’s phase/turn logic **does not fully require** a player to continue capturing when possible.
- Complex patterns mentioned in the rules/FAQ (e.g., 180° reversals, cycles) are not clearly encoded as scenario tests.

**Consequence:**
- Games played through the engine would diverge from actual RingRift rules.
- AI evaluation and training would be based on incorrect dynamics.

### 3. UI is Minimal (BLOCKS HUMAN PLAY)

Front-end currently provides:

- Layout and routing
- Basic auth and placeholder pages (Home, Lobby, Game, etc.)

But **no game board UI**:

- No board grid for any board type
- No ring/marker/collapsed-space rendering
- No click-to-move or choice dialogs
- No visual representation of chains, lines, territory, or forced elimination

**Consequence:**
- Human users cannot play or even inspect game states visually.
- Testing is limited to programmatic tests and logs.

### 4. Testing Coverage is Very Low

- Jest is configured with ambitious thresholds, but:
  - Only a small handful of tests exist (BoardManager position & adjacency tests)
  - No scenario tests built from the rules documentation
  - No integration tests for full turns or games

**Consequence:**
- Refactoring core logic is risky.
- Many edge cases from `ringrift_complete_rules.md` and FAQs are unverified.

### 5. AI Not Yet in the Game Loop

- Python AI service and TS client are ready to be used.
- No code path from "it’s an AI player’s turn" → "ask AI for a move" → `GameEngine.makeMove()`.

**Consequence:**
- No single-player experience, despite the infrastructure being mostly ready.

---

## 🧭 Recommended Strategic Plan (Codebase-Focused)

This plan is consistent with, and refines, the existing `STRATEGIC_ROADMAP.md`, with emphasis on:

- **MVP-first:** a fully playable local game
- **Testing-first:** tests validating rules + scenarios
- **AI-ready:** Python microservice as the primary AI engine

### Phase 0 – Testing & Quality Foundation (1–2 weeks)

**Goals:** Make it safe to change the engine.

1. **Solidify Jest setup (mostly done)**
   - Confirm `tests/setup.ts` + `tests/test-environment.js` work across server tests.
   - Add `npm run test:watch` and `npm run test:coverage` scripts if missing.

2. **Initial unit tests:**
   - Expand BoardManager tests to cover:
     - Marker CRUD and collapsing
     - Line detection edge cases for each board type
     - Disconnected region detection for simple patterns
   - Add RuleEngine tests for:
     - Simple valid/invalid moves (movement + capture)

3. **CI gate:**
   - Enforce `npm test` on PRs (already present) but temporarily relax coverage thresholds **until tests catch up**, then tighten again.

### Phase 1 – Finish Core Rules (2–3 weeks)

**Goals:** Fully rule-compliant engine, **before** heavy UI work.

1. **Player Choice Infrastructure**

   - Add shared types in `src/shared/types/game.ts`:

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

   - Implement a `PlayerInteractionManager` on the server that:
     - Emits choice events (to WebSocket or callback) without knowing about UI/transport
     - Awaits responses (with timeout)
     - Provides a synchronous-appearing API to GameEngine (e.g. `await getPlayerChoice(...)`)
   - Integrate at all choice points in `GameEngine.ts`:
     - Line ordering and Option 1 vs 2
     - Elimination stack selection
     - Region processing order
     - Capture direction selection in chains

2. **Chain Capture Enforcement**

   - Extend GameEngine to:
     - Mark when a chain capture is in progress
     - After applying a capture, compute available follow-up captures for that stack
     - Force the player (via UI/AI) to continue selecting capture moves until none remain
   - Add explicit rule-based tests for:
     - Mandatory continuation
     - 180° reversal patterns
     - Cyclic capture sequences

3. **Rule Scenario Tests**

   - Derive tests directly from `ringrift_complete_rules.md` and FAQs Q1–Q24.
   - Encode a handful of emblematic scenarios first (e.g., simple captures, line formation, basic disconnections), then expand.

### Phase 2 – Minimal Playable UI (2–3 weeks)

**Goals:** Human-usable local 2-player game.

1. **Board Rendering Components**
   - `SquareBoard` and `HexBoard` React components that consume a normalized board-state view from the server (or a client mirror of `GameState`).
   - `Cell` / `HexCell` components with appropriate coordinates.
   - Visual layers for stacks, markers, and collapsed spaces.

2. **Interaction & Choices**
   - Click-to-select stack and destination; highlight valid moves.
   - Show choices via modal or side panel when PlayerInteractionManager requests input.
   - Display current phase, active player, ring/territory counts.

3. **Local 2-Player Mode**
   - Initially, skip multiplayer; just host a single game on the backend, with the client connected as both players.

### Phase 3 – AI Integration (2–3 weeks)

**Goals:** Single-player mode powered by Python AI.

1. **Define AI Player in GameState**
   - Extend player type to include `type: 'human' | 'ai'` and AI config (difficulty, AI type).

2. **Wire AIServiceClient into Game Loop**
   - In the server, when it’s an AI player’s turn:
     - Use `AIServiceClient.healthCheck()` to confirm availability.
     - Call `getAIMove(currentGameState, playerNumber, difficulty, aiType)`.
     - Validate the returned move through RuleEngine to avoid trust issues.
     - Apply via `GameEngine.makeMove()`.

3. **AI + Choice Integration**
   - When a choice is needed for an AI player, either:
     - Delegate to AI service via a dedicated `/ai/choice` endpoint, or
     - Use simple heuristics locally in TypeScript for now.

4. **Fallback Strategy**
   - If AI service is down, provide:
     - A simple TypeScript random/heuristic AI
     - Or degrade gracefully with an error instead of hanging.

### Phase 4 – Validation & Polish (1–2 weeks)

- Heavy scenario-driven tests across board types.
- Performance tuning for AI latency (< 2 seconds typical).
- UX polish: animations, loading states, friendly error messages.

### Phase 5 – Multiplayer, Persistence, and Extras (future)

- Use existing WebSocket skeleton to sync moves across clients.
- Use Prisma models to persist game and move history.
- Implement spectator mode, replays, rating system, etc.

---

## 🧪 Testing Strategy (Code-Centric)

1. **Unit Tests (80–90% coverage on game modules)**
   - BoardManager: positions, adjacency, markers, lines, disconnected regions.
   - RuleEngine: validateMove for all move types, getValidMoves.
   - GameEngine: phase transitions, state updates, forced elimination, line and territory post-processing.

2. **Integration Tests**
   - End-to-end turn flows: place → move → capture → line → territory.
   - Forced elimination scenarios.
   - Hex vs square board differences.

3. **Scenario Tests from Rules/FAQ**
   - Encoded as structured setups + expected outcomes.
   - Validate complex interactions beyond unit-level guarantees.

4. **AI Integration Tests**
   - Mock AIServiceClient (or use a test instance) to:
     - Ensure the game waits for AI moves.
     - Ensure invalid moves from AI are rejected.

---

## 🔍 Files & Areas Worth Examining (for Documentation & Planning)

From the current project tree:

- **Core Engine & Rules**
  - `src/server/game/BoardManager.ts`
  - `src/server/game/GameEngine.ts`
  - `src/server/game/RuleEngine.ts`
  - `src/server/game/ai/AIEngine.ts`, `AIPlayer.ts` (TS-side AI scaffolding)

- **AI Integration**
  - `src/server/services/AIServiceClient.ts`
  - `ai-service/app/main.py`, `ai-service/app/ai/*.py`

- **Shared Types & Validation**
  - `src/shared/types/game.ts`
  - `src/shared/types/websocket.ts`
  - `src/shared/validation/schemas.ts`

- **Frontend Shell**
  - `src/client/App.tsx`, `src/client/components/Layout.tsx`
  - `src/client/pages/GamePage.tsx`, `LobbyPage.tsx`, `HomePage.tsx`, etc.

- **Docs & Plans**
  - `CURRENT_STATE_ASSESSMENT.md`
  - `STRATEGIC_ROADMAP.md`
  - `RINGRIFT_IMPROVEMENT_PLAN.md`
  - `TODO.md`
  - `ARCHITECTURE_ASSESSMENT.md`
  - `TECHNICAL_ARCHITECTURE_ANALYSIS.md`
  - `BOARD_TYPE_IMPLEMENTATION_PLAN.md`

These documents now mostly reflect the current state; this evaluation aligns with them and clarifies where earlier assessments (that assumed marker/territory systems were missing) are superseded by the current code.

---

## ✅ Summary

- The **architecture, documentation, and infrastructure are excellent**.
- The **core engine is substantially implemented and code-verified** against the rules, especially markers, lines, territory, phases, and hex boards.
- The **critical remaining engine gaps** are player choice and full chain capture enforcement.
- The **biggest blockers to actual use** are the missing UI and low test coverage.
- The **Python AI microservice and TypeScript AI client exist and should be kept**, with the next step being to actually integrate them into the game loop and choice system.

If you follow the phased plan above (which dovetails with `STRATEGIC_ROADMAP.md` but emphasises Python AI integration and the true state of the core engine), RingRift can realistically reach a **playable, single-player MVP in ~8–10 weeks** of focused work.

---

## Refactoring Axes – Codebase View (TS/React/Python AI)

This section mirrors the four architectural refactoring axes from `ARCHITECTURE_ASSESSMENT.md` and reframes them in terms of the current codebase reality and next concrete improvements.

### Axis 1 – Game Rules & State Architecture

From the code’s perspective:
- `BoardManager`, `RuleEngine`, and `GameEngine` together already encode most of the rules in `ringrift_complete_rules.md`: movement, markers, overtaking captures, chain continuation, line formation/collapse, territory disconnection, forced elimination, and victory checks across 8×8, 19×19, and hex boards.  
- The PlayerInteraction* layer (PlayerInteractionManager + WebSocketInteractionHandler + AIInteractionHandler + DelegatingInteractionHandler) provides the abstraction GameEngine uses to request PlayerChoices (line order, rewards, eliminations, region order, capture direction) from humans and AI.

Code‑level technical debt:
- There is no **single, authoritative mapping** from rules sections/FAQ items to specific engine methods and tests; knowledge is spread between docs and test names.  
- Some invariants are encoded implicitly (e.g., the self‑elimination prerequisite is enforced inside `processDisconnectedRegions` and helper methods, but only partially documented in comments).  
- The boundaries between BoardManager, RuleEngine, and GameEngine are mostly clean but not described as explicit contracts, which makes it harder for new contributors to know where to add logic or tests.

Code‑centric recommendations:
- Add a compact rules–code–tests matrix (likely in `CURRENT_STATE_ASSESSMENT.md` or a new `RULES_TO_CODE_MAPPING.md`) that, for each major rules section, lists:
  - Primary engine entrypoints and helpers.  
  - The Jest test files/cases that verify them.  
- Treat that matrix as the checklist for future refactors: when we change rules or code, we update both the matrix and the tests.  
- Where logic is “emergent” (e.g., forced elimination, chain termination rules, or specific FAQ edge cases), lift the underlying invariants into named helpers and comments so that tests can reference them directly.

### Axis 2 – AI Boundary & Integration

From the code’s perspective:
- The AI surface is split across `ai-service/app/main.py`, Pydantic models, `src/server/services/AIServiceClient.ts`, and `src/server/game/ai/AIEngine.ts`.  
- AI turns and some PlayerChoices (notably `line_reward_option`, with ring/region choices following) are already routed through `globalAIEngine` and `AIInteractionHandler`.

Code‑level technical debt:
- The TypeScript and Python sides agree informally on request/response shape (via shared naming and comments), but there is no small, central **“AI Contract”** document or type that both strictly follow.  
- Policy decisions (service‑backed vs local heuristic, error handling, difficulty mapping) are scattered across `AIEngine`, `AIInteractionHandler`, WebSocket server, and tests.

Code‑centric recommendations:
- Define a minimal, versioned AI contract: TypeScript types and Python models that are clearly treated as external interfaces, plus a short doc section linking them.  
- Centralise AI profile policy in `AIEngine`/`globalAIEngine` so that GameEngine and interaction handlers only depend on “ask AI for move/choice” semantics, not on how or where those decisions are computed.  
- Extend tests around AI to cover both service success and failure paths systematically for moves and for at least the most important PlayerChoice types, using the real `AIEngine` façade rather than mocking deep internals.

### Axis 3 – WebSocket/Game Loop Reliability

From the code’s perspective:
- `src/server/websocket/server.ts`, `WebSocketInteractionHandler`, and `GameContext`/`ChoiceDialog` implement the current game loop and choice flows over Socket.IO.  
- Integration tests like `WebSocketServer.aiTurn.integration.test.ts` exercise parts of this loop, but not all failure/reconnection scenarios.

Code‑level technical debt:
- Some event flows (join/leave, reconnection, spectator handling) are stubs or only lightly tested.  
- There is no single document that states, “Here is the authoritative sequence of WebSocket events for a turn / AI turn / PlayerChoice.”  
- Message schemas live in `src/shared/types/websocket.ts` but are not treated as versioned contracts, so it’s easy to break them ad hoc.

Code‑centric recommendations:
- Write down the **canonical WebSocket turn flow** (including AI) and align server+client code and tests to that flow.  
- Harden `WebSocketInteractionHandler` around duplicate/stale choice responses, timeouts, and unexpected disconnects, and encode those behaviours in tests.  
- Treat `src/shared/types/websocket.ts` as the primary API surface: changes here should be deliberate and reflected in both server and client code, plus tests.

### Axis 4 – Testing & Quality Gates

From the code’s perspective:
- Tests already exist for many critical areas (BoardManager adjacency/territory, RuleEngine movement/capture, GameEngine chain captures and territory disconnection, PlayerInteractionManager, WebSocketInteractionHandler, AIEngine/AIServiceClient, AIInteractionHandler, GameEngine choice integrations), but overall coverage is still low relative to rule complexity.  
- CI is wired to run tests and collect coverage, but thresholds and grouping do not yet reflect the four axes.

Code‑level technical debt:
- Some older documents and comments still refer to “no tests” or “basic engine not implemented”, which no longer match reality and can mislead planning.  
- There is no clear way to say, “Run all tests relevant to rules/state” vs “Run all tests relevant to AI boundary”.

Code‑centric recommendations:
- Group tests logically (by directory or naming convention) along the four axes, and add npm scripts to run those groupings conveniently.  
- Incrementally raise coverage thresholds once each axis has a baseline suite.  
- Keep `CODEBASE_EVALUATION.md`, `CURRENT_STATE_ASSESSMENT.md`, `STRATEGIC_ROADMAP.md`, and `TODO.md` in sync so that high‑level status, evaluation, roadmap, and concrete tasks all tell the same story.

These axis‑specific views are intended to make it easier to choose the next deep refactor: pick a rule cluster or boundary in Axis 1–3, then drive tests and changes under Axis 4 to lock in the improvements.
