# RingRift Architecture Assessment

**Assessment Date:** November 14, 2025  
**Status:** Comprehensive architecture review and optimization recommendations

**Status Update (November 14, 2025):** Core game logic, WebSocket-backed backend play, AI turns, and basic board/choice UI are now in place. For the most up-to-date implementation status and component completion percentages, see `CURRENT_STATE_ASSESSMENT.md` and `TODO.md`. The architectural recommendations in this document remain valid and are intentionally more long-term than the day-to-day status documents.

---

## Executive Summary

RingRift follows a **TypeScript-first architecture** with Node.js backend and React frontend. This assessment evaluates the current implementation distribution, identifies architectural strengths and gaps, and provides recommendations for optimal feature distribution.

This document, together with `deprecated/CODEBASE_EVALUATION.md`, supersedes earlier high-level design plans such as `deprecated/ringrift_architecture_plan.md` and the more aspirational parts of `deprecated/TECHNICAL_ARCHITECTURE_ANALYSIS.md`. Those older documents are now treated as historical context only; any architectural guidance still considered valid has been merged into the current assessment, roadmap, and improvement plan.

**Overall Architecture Grade: B+**
- ✅ Excellent: Architecture planning and documentation
- ✅ Good: Technology stack choices and type safety
- ⚠️ Needs Work: Implementation completeness
- ⚠️ Needs Work: Testing infrastructure
- ❌ Missing: Advanced AI engine and production monitoring (CI/CD and Jest-based unit/integration suites now exist but still need broader coverage, stronger quality gates, and production-grade monitoring/alerting. For current per-component completion percentages, see `CURRENT_STATE_ASSESSMENT.md`.)

---

## Current Architecture Distribution

### What's Implemented in TypeScript

#### ✅ **Backend Core (Node.js + TypeScript)**

**Game Logic Layer** (`src/server/game/`)
- `GameEngine.ts` - Core game orchestration (captures, lines, territory, phases; chain enforcement still incomplete)
- `RuleEngine.ts` - Move validation and rule enforcement (movement/capture rules implemented, edge cases pending)
- `BoardManager.ts` - Board state management (positions, markers, stacks, lines, territories)
- **Status**: ~70% complete – core rules implemented, player choice integration and chain captures still missing

**API Layer** (`src/server/routes/`)
- `auth.ts` - Authentication endpoints
- `game.ts` - Game management endpoints
- `user.ts` - User management endpoints
- **Status**: Basic structure in place, minimal implementation

**Infrastructure** (`src/server/`)
- `index.ts` - Server entry point
- `middleware/` - Auth, error handling, rate limiting
- `cache/redis.ts` - Redis integration
- `database/connection.ts` - Prisma/PostgreSQL
- `websocket/server.ts` - Socket.io setup
- **Status**: Well-structured, good foundation

#### ✅ **Frontend Core (React + TypeScript)**

**Client Layer** (`src/client/`)
- `App.tsx` - Main application component
- `components/` - Reusable UI components (including `BoardView` and `ChoiceDialog`)
- `pages/GamePage.tsx` - Local sandbox and backend game views with board rendering and setup UI
- `contexts/` - React context providers (`AuthContext`, `GameContext` for WebSocket game state)
- `services/api.ts` - API client
- **Status**: Basic shell plus minimal game UI (board for 8×8, 19×19, hex and pre-game setup); move wiring, HUD, and choice wiring still needed

#### ✅ **Shared Types** (`src/shared/`)

**Type Definitions**
- `types/game.ts` - Game state, moves, board types ✅ WELL DESIGNED
- `types/user.ts` - User and authentication types
- `types/websocket.ts` - WebSocket event types
- `validation/schemas.ts` - Zod validation schemas
- **Status**: Excellent type coverage, comprehensive

### What's External to TypeScript

#### 🔧 **Infrastructure Services** (Separate Containers)

1. **PostgreSQL Database**
   - User accounts, game history, ratings
   - Managed via Prisma ORM (TypeScript)
   - Status: Schema defined via `prisma/schema.prisma` ✅

2. **Redis Cache**
   - Active game state, session management
   - Accessed via TypeScript client
   - Status: Integration code exists ✅

3. **Docker Infrastructure**
   - Container orchestration
   - Environment management
   - Status: Configuration complete ✅

#### ❌ **Missing Components** (Not Yet Implemented)

1. **AI Integration** - PARTIAL / BOUNDARY GAP
   - Python FastAPI AI microservice exists in `ai-service/` with Random/Heuristic AIs and is integrated into backend games via `AIEngine` / `AIServiceClient` and `WebSocketServer.maybePerformAITurn`, so AI players can select and apply moves.
   - Several PlayerChoices (for example `line_reward_option`, `ring_elimination`, and `region_order`) are already service-backed behind `globalAIEngine` / `AIInteractionHandler`, with remaining choices currently answered via local heuristics.
   - Recommendation: Harden the AI boundary (clear request/response contracts, failure/timeout behaviour, metrics) and progressively move more choice logic behind the service where it makes sense, keeping simple heuristics available as a low-latency fallback.

2. **Frontend UI** - MAJOR GAP (PARTIALLY ADDRESSED)
   - Game board rendering now implemented via `BoardView` for 8×8, 19×19, and hex boards.
   - `GamePage` provides a local sandbox setup (players, human/AI flags, board type) and a read-only backend game view.
   - Missing: move input wiring, valid-move highlighting, full HUD, and real-time PlayerChoice dialogs.
   - Recommendation: Treat the current UI as a scaffold and focus on wiring it to backend moves and the PlayerInteractionManager.

3. **Testing Infrastructure** - PARTIAL
   - Jest configuration and basic tests exist (e.g., BoardManager, PlayerInteractionManager).
   - No comprehensive integration or scenario tests yet; coverage remains low (<10%).
   - Recommendation: Expand tests alongside new engine/interaction work and enforce coverage via CI.

---

## Optimal Architecture Recommendations

### Principle: **Monolith First, Microservices When Needed**

#### Current Monolith (TypeScript) ✅ **KEEP**

```
┌─────────────────────────────────────────────┐
│         TypeScript Monolith                 │
│                                             │
│  ┌──────────────┐      ┌──────────────┐   │
│  │   Frontend   │      │   Backend    │   │
│  │  React + TS  │◄────►│  Node.js+TS  │   │
│  └──────────────┘      └──────────────┘   │
│         │                      │            │
│  ┌──────────────────────────────────────┐  │
│  │      Shared Types (TypeScript)       │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
         │                      │
    ┌─────────┐          ┌─────────┐
    │PostgreSQL│          │  Redis  │
    └─────────┘          └─────────┘
```

**Advantages:**
- ✅ Single codebase, easier to maintain
- ✅ Type safety across entire stack
- ✅ Faster development iteration
- ✅ Shared validation logic
- ✅ Simpler deployment

**When to Keep in Monolith:**
- Game logic and rules ✅
- Move validation ✅
- State management ✅
- API endpoints ✅
- WebSocket handlers ✅
- Simple AI (random, basic heuristics) ✅

#### Future Microservice: AI Engine ⚠️ **SEPARATE LATER**

```
┌─────────────────┐         ┌──────────────────┐
│  TypeScript     │         │   AI Service     │
│  Monolith       │◄───────►│  Python/Rust     │
│                 │  gRPC/  │                  │
│  • Game Logic   │  REST   │  • MCTS Engine   │
│  • Simple AI    │         │  • Neural Nets   │
│  • API/WebSocket│         │  • Training      │
└─────────────────┘         └──────────────────┘
```

**When to Separate:**
- AI becomes performance bottleneck
- Need for ML model training
- GPU acceleration required
- Team has Python/ML expertise

**For MVP:** Keep simple AI in TypeScript ✅

**Recommendation for AI Architecture:**
```typescript
// Phase 1: TypeScript AI (MVP)
src/server/game/ai/
├── AIEngine.ts           // Main AI orchestrator
├── RandomAI.ts          // Difficulty 1-2
├── HeuristicAI.ts       // Difficulty 3-5
└── MinimaxAI.ts         // Difficulty 6-8

// Phase 2: Optional Python microservice (if needed)
ai-service/
├── main.py              // FastAPI server
├── mcts_engine.py       // Monte Carlo Tree Search
├── neural_network.py    // ML-based evaluation
└── training/            // Model training scripts
```

---

## Implementation Priority Matrix

### What Belongs Where

| Feature | Location | Language | Priority | Status |
|---------|----------|----------|----------|--------|
| **Core Game Logic** | Monolith | TypeScript | P0 | 40% ⚠️ |
| Game state management | Monolith | TypeScript | P0 | 60% ⚠️ |
| Move validation | Monolith | TypeScript | P0 | 50% ⚠️ |
| Rule enforcement | Monolith | TypeScript | P0 | 30% ⚠️ |
| **API Endpoints** | Monolith | TypeScript | P0 | 70% ⚠️ |
| Authentication | Monolith | TypeScript | P0 | 80% ✅ |
| Game CRUD | Monolith | TypeScript | P0 | 40% ⚠️ |
| **WebSocket Events** | Monolith | TypeScript | P0 | 50% ⚠️ |
| Real-time moves | Monolith | TypeScript | P0 | 30% ⚠️ |
| Game broadcasts | Monolith | TypeScript | P0 | 40% ⚠️ |
| **Frontend UI** | Monolith | TypeScript | P1 | 30% ⚠️ |
| Board rendering | Monolith | TypeScript | P1 | 60% ⚠️ |
| Move interface | Monolith | TypeScript | P1 | 10% ❌ |
| **Simple AI** | Monolith | TypeScript | P1 | 20% ⚠️ |
| Random moves | Monolith | TypeScript | P1 | 20% ⚠️ |
| Basic heuristics | Monolith | TypeScript | P1 | 20% ⚠️ |
| **Testing** | Monolith | TypeScript | P0 | 10% ⚠️ |
| Unit tests | Monolith | TypeScript | P0 | 10% ⚠️ |
| Integration tests | Monolith | TypeScript | P1 | 5% ⚠️ |
| **Advanced AI** | Microservice | Python | P2 | 0% ❌ |
| MCTS engine | Separate | Python | P2 | 0% ❌ |
| Neural networks | Separate | Python | P3 | 0% ❌ |

---

## Architectural Decisions & Rationale

### ✅ **Decision 1: TypeScript Monolith for Core**

**Reasoning:**
1. **Type Safety**: Shared types prevent runtime errors
2. **Developer Velocity**: Single language reduces context switching
3. **Code Reuse**: Validation logic shared between client/server
4. **Maintenance**: Easier to refactor and maintain
5. **Team Size**: Small teams benefit from unified codebase

**Trade-offs:**
- ❌ Slightly slower than compiled languages (negligible for game logic)
- ❌ Node.js not ideal for CPU-intensive AI (mitigated by eventual microservice)

### ✅ **Decision 2: PostgreSQL for Persistence**

**Reasoning:**
1. **ACID compliance** for game integrity
2. **Complex queries** for leaderboards and statistics
3. **JSON support** for flexible game state storage
4. **Mature ecosystem** with Prisma ORM

**Trade-offs:**
- ❌ Not as fast as specialized databases for read-heavy workloads

### ✅ **Decision 3: Redis for Game State Cache**

**Reasoning:**
1. **In-memory speed** for active games
2. **Pub/Sub** for real-time updates
3. **Session management** for WebSocket connections
4. **TTL support** for automatic cleanup

### ⚠️ **Decision 4: Defer AI Microservice**

**Reasoning:**
1. **MVP doesn't need advanced AI**: Simple heuristics sufficient initially
2. **Premature optimization**: Don't separate until proven necessary
3. **Overhead**: Microservices add deployment and communication complexity

**When to Revisit:**
- AI calculations exceed 1 second response time
- Want to add ML-based opponents
- Need GPU acceleration for neural networks

---

## Missing Components - Action Items

### 🔴 **Critical: Add to TODO.md**

#### 1. Testing Infrastructure (P0)

```typescript
// New directory structure needed
src/server/__tests__/
├── unit/
│   ├── GameEngine.test.ts
│   ├── RuleEngine.test.ts
│   ├── BoardManager.test.ts
│   └── ...
├── integration/
│   ├── api.test.ts
│   ├── websocket.test.ts
│   └── database.test.ts
└── e2e/
    ├── gameplay.test.ts
    └── multiplayer.test.ts

src/client/__tests__/
├── components/
│   └── GameBoard.test.tsx
└── integration/
    └── gameplay.test.tsx
```

**Tools Required:**
- Jest/Vitest for unit tests
- Supertest for API testing
- Playwright for E2E tests
- React Testing Library for components

#### 2. AI Engine Implementation (P1)

```typescript
// MVP: TypeScript implementation
src/server/game/ai/
├── AIEngine.ts              // Main interface
├── AIPlayer.ts              // Base class
├── RandomAI.ts              // Levels 1-2
├── HeuristicAI.ts           // Levels 3-5
├── MinimaxAI.ts             // Levels 6-8 (optional for MVP)
└── evaluators/
    ├── MaterialEvaluator.ts
    ├── TerritoryEvaluator.ts
    └── MobilityEvaluator.ts
```

#### 3. Frontend Game UI (P1)

```typescript
// Critical components needed
src/client/components/game/
├── GameBoard/
│   ├── GameBoard.tsx           // Main board container
│   ├── SquareBoard.tsx         // 8x8 and 19x19 rendering
│   ├── HexagonalBoard.tsx      // Hex board rendering
│   ├── BoardCell.tsx           // Individual cells
│   └── BoardOverlay.tsx        // Valid moves, highlights
├── GamePieces/
│   ├── RingStack.tsx           // Ring stack visualization
│   ├── Marker.tsx              // Marker display
│   └── CollapsedSpace.tsx      // Territory display
├── GameControls/
│   ├── MoveControls.tsx        // Move input
│   ├── GameInfo.tsx            // Score, time, status
│   └── PlayerPanel.tsx         // Player information
└── GameInterface.tsx           // Complete game UI
```

#### 4. CI/CD Pipeline (P1)

```yaml
# .github/workflows/ci.yml needed
name: CI/CD Pipeline
on: [push, pull_request]
jobs:
  test:
    - Run linting (ESLint)
    - Run type checking (tsc)
    - Run unit tests (Jest/Vitest)
    - Run integration tests
    - Generate coverage report
  
  build:
    - Build frontend
    - Build backend
    - Build Docker images
  
  deploy:
    - Deploy to staging (on merge to main)
    - Deploy to production (on release tag)
```

#### 5. Monitoring & Observability (P2)

```typescript
// Monitoring infrastructure needed
src/server/monitoring/
├── metrics.ts               // Prometheus metrics
├── logging.ts              // Structured logging
├── tracing.ts              // Distributed tracing
└── healthcheck.ts          // Health endpoints
```

---

## Recommended Architecture Additions to TODO

### Phase 0: Testing Foundation (NEW - P0)

**Duration:** 1-2 weeks  
**Parallel to Phase 1 core logic fixes**

- [ ] Set up Jest/Vitest testing framework
- [ ] Configure test coverage reporting (aim for 80%+)
- [ ] Create test utilities and fixtures
- [ ] Write tests for existing code BEFORE refactoring
- [ ] Set up CI pipeline to run tests automatically
- [ ] Add pre-commit hooks for linting and testing

### Phase 1.5: AI Engine (NEW - P1)

**Duration:** 2-3 weeks  
**After core game logic complete**

- [ ] Design AI interface and difficulty system
- [ ] Implement RandomAI (difficulty 1-2)
- [ ] Implement HeuristicAI (difficulty 3-5)
- [ ] Create position evaluation functions
- [ ] Add AI timing controls (difficulty via think time)
- [ ] Implement AI move generation
- [ ] Write AI unit tests
- [ ] Add AI integration tests

### Phase 2.5: Monitoring & DevOps (NEW - P2)

**Duration:** 1-2 weeks  
**During Phase 3 frontend work**

- [ ] Set up Prometheus metrics
- [ ] Configure Grafana dashboards
- [ ] Implement structured logging (Winston)
- [ ] Add error tracking (Sentry)
- [ ] Create health check endpoints
- [ ] Set up alerting for critical errors
- [ ] Document deployment procedures
- [ ] Create rollback procedures

---

## Architecture Anti-Patterns to Avoid

### ❌ **DON'T: Premature Microservices**

**Problem:** Splitting services too early adds complexity without benefit

**Current Status:** ✅ GOOD - Using monolith appropriately

**When to Split:**
- Service has different scaling needs
- Team grows to 8+ developers
- Component causes performance bottleneck
- Different technology truly beneficial

### ❌ **DON'T: Bypass Type Safety**

**Problem:** Using `any` type defeats TypeScript benefits

**Current Issues:**
- Some `any` types in GameResult.reason field
- Some implicit any in older code

**Fix:** Strict TypeScript configuration enforced ✅

### ❌ **DON'T: Logic in Multiple Layers**

**Problem:** Game rules duplicated in client and server

**Current Status:** ⚠️ RISK - No client validation yet

**Solution:**
```typescript
// Shared validation in src/shared/validation/
export const validateMove = (move: Move, state: GameState): boolean => {
  // Shared logic used by both client and server
};

// Client uses for UI hints
// Server uses for authoritative validation
```

### ❌ **DON'T: Tight Coupling**

**Problem:** GameEngine directly accessing database

**Current Status:** ✅ GOOD - Proper layering exists

```typescript
// GOOD: Separation of concerns
Controller → Service → GameEngine → RuleEngine
     ↓
 Database
```

---

## Performance Considerations

### Current Architecture Performance Profile

| Component | Expected Load | Bottleneck Risk | Mitigation |
|-----------|---------------|-----------------|------------|
| Game Logic | Med-High | Low | Pure functions, efficient |
| WebSocket | High | Medium | Redis pub/sub scaling |
| Database | Medium | Low | Proper indexing + caching |
| AI Engine | Variable | High | Async processing needed |
| Frontend | Low | Low | React optimization |

### Optimization Strategy

**Phase 1 (MVP):**
- ✅ Redis caching for active games
- ✅ Database indexing on foreign keys
- ✅ WebSocket message batching
- ⚠️ Need: AI move caching

**Phase 2 (Growth):**
- Horizontal scaling via load balancer
- Database read replicas
- CDN for static assets
- Background job processing for AI

**Phase 3 (Scale):**
- Microservice extraction if needed
- Advanced caching strategies
- Database sharding if needed
- AI GPU acceleration

---

## Conclusion & Action Plan

### Architecture Strengths ✅

1. **Type Safety**: Comprehensive TypeScript coverage
2. **Modern Stack**: React, Node.js, PostgreSQL, Redis
3. **Documentation**: Excellent architecture planning
4. **Infrastructure**: Docker, proper separation of concerns
5. **Scalability**: Designed for growth

### Critical Gaps ❌

1. **Testing**: Jest-based unit and integration suites exist, but coverage is still low relative to the rules’ complexity and does not yet encode all rule/FAQ scenarios (HIGH PRIORITY).
2. **Core Logic**: The core loop (movement, captures, lines, territory, choices) is implemented and used in backend and sandbox games, but some edge-case rules and invariants still lack exhaustive tests and refinement (HIGH PRIORITY).
3. **Frontend UI**: Game HUD, timers, post-game flows, and lobby/matchmaking remain minimal; UX is suitable for developers and playtesters but not yet for production audiences (BLOCKING for polished play).
4. **AI Engine & Boundary**: The Python AI service plus TypeScript `AIEngine`/`AIServiceClient` are integrated for moves and several PlayerChoices, but stronger AI behaviour, explicit service contracts, and resilience under failure/latency are still needed (HIGH PRIORITY).
5. **Monitoring/Observability**: Basic logging and CI exist, but there is no end-to-end monitoring/alerting pipeline or AI/service-level telemetry; Prometheus/Grafana in `docker-compose.yml` are not yet fully wired into the application (MEDIUM PRIORITY).

### Immediate Action Items

**Week 1-2: Testing Foundation**
1. Add testing framework
2. Write tests for existing code
3. Set up CI pipeline

**Week 3-5: Core Logic (Parallel to Testing)**
4. Complete game rules implementation
5. Test each rule as implemented
6. Validate against game rules document

**Week 6-8: AI & Frontend**
7. Implement basic AI (TypeScript)
8. Build game board UI
9. Create interactive game interface

**Week 9-10: Integration & Polish**
10. End-to-end testing
11. Performance optimization
12. Production deployment prep

### Architecture Decision: APPROVED ✅

**The current TypeScript monolith architecture is OPTIMAL for RingRift.**

**Do NOT over-engineer** by adding microservices prematurely. Focus on:
1. Completing core implementation
2. Adding comprehensive tests
3. Building the UI
4. Deploying MVP

**Future-proof**: Architecture supports scaling when needed.

---

**Next Review:** After Phase 1 completion  
**Document Version:** 1.0  
**Maintained By:** Development Team

---

## Refactoring Axes & Deep Improvement Plan (TS/React/Python AI)

This section summarizes the four main refactoring axes we will use to guide deep, root‑cause improvements to the TypeScript/Node backend, React client, and Python AI service. It is intentionally high‑level and architecture‑oriented; concrete tasks and status live in `TODO.md`, `CURRENT_STATE_ASSESSMENT.md`, and `STRATEGIC_ROADMAP.md`.

### 1. Game Rules & State Architecture

**Scope**  
GameEngine, BoardManager, RuleEngine, shared types (`src/shared/types/game.ts`), and the PlayerInteraction* layer (PlayerInteractionManager, WebSocketInteractionHandler, AIInteractionHandler, DelegatingInteractionHandler) as they relate to `ringrift_complete_rules.md`.

**Current Position (TS side)**
- Core mechanics (movement, markers, overtaking captures, line formation/collapse, territory disconnection, forced elimination, hex + square boards) are implemented and generally aligned with the rules doc.  
- PlayerChoice flows and mandatory chain captures are present and exercised by several unit/integration tests.  
- The Rust engine provides a partially independent reference implementation but is not the runtime source of truth; the rules document is.

**Key Risks / Technical Debt**
- Rules fidelity currently depends on a combination of code reading, scattered tests, and cross‑referencing the rules doc; there is no single “rules ↔ code” mapping artifact.  
- Some invariants (e.g., self‑elimination prerequisite for territory, chain capture edge cases, ordering of lines and regions) are enforced by emergent behaviour rather than explicitly codified contracts.  
- GameEngine, BoardManager, and RuleEngine are relatively well factorized, but their responsibility boundaries are implicit rather than documented as formal interfaces and invariants.

**Recommended Direction**
- **Rules–Code Conformance Matrix:** Create and maintain a concise matrix (or section in `CURRENT_STATE_ASSESSMENT.md`) that maps each major rules section (4.x–13.x, FAQ 1–24) to specific code entrypoints and tests. This becomes the single authority for “what implements what” in TS.  
- **Explicit Module Contracts:** Document and, where useful, enforce clearer contracts:
  - BoardManager: topology + pure board operations + territory/line detection (no phase or choice knowledge).  
  - RuleEngine: pure validation and move generation (no state mutation, no transport, no timers).  
  - GameEngine: the only mutator of GameState and the only place that owns phase transitions, chain capture orchestration, and calls into PlayerInteractionManager.  
  - PlayerInteractionManager*: pure “question/answer” abstraction with no game rules, used by GameEngine but not vice‑versa.  
- **Invariants as First‑Class Citizens:** For each of the following rule clusters, define and test explicit invariants:
  - Turn sequence & forced elimination (Sections 4.x, FAQ 15.2, 24).  
  - Non‑capture movement & marker interaction (Section 8.x).  
  - Overtaking/chain capture semantics (Sections 9–10, FAQ 14, 15.3.x).  
  - Line formation & graduated rewards (Section 11.2–11.3, FAQ 7, 22).  
  - Territory disconnection & self‑elimination prerequisite (Section 12.2–12.3, FAQ 15, 23).  
- **Deep Refactor Style:** When discrepancies are found, prefer shaping GameEngine/BoardManager/RuleEngine around the rules document and these invariants rather than layering special‑case fixes.

### 2. AI Boundary & Integration

**Scope**  
Python AI service (`ai-service/`), AIServiceClient, AIEngine/globalAIEngine, AIInteractionHandler, and their integration with GameEngine + PlayerInteractionManager.

**Current Position**
- Python FastAPI service provides move selection and some choice endpoints, designed to accept full GameState + options where useful.  
- TypeScript has a clear façade (`AIEngine` + `AIServiceClient`) and a global configurator (`globalAIEngine`) that mediates AI profiles and service calls.  
- AI moves in backend games are service‑backed; `line_reward_option` is fully service‑backed end‑to‑end with fallbacks; ring elimination and region order are wired with service endpoints and TS façades.

**Key Risks / Technical Debt**
- The AI boundary is present but not yet treated as a strict contract:  
  - Request/response shapes and error semantics are implicit, not documented as versioned contracts.  
  - Choice decisions for AI are partly service‑backed and partly local heuristics, with behaviour controlled by code rather than configuration.  
- Observability around AI (latency, errors, fallbacks, profile usage) is limited, making tuning and debugging harder.  
- The AI integration tests cover representative paths but not a systematic matrix of “AI type × difficulty × failure mode × choice type”.

**Recommended Direction**
- **Treat AI as a First‑Class Boundary:** Document a small, stable contract (even if implemented over HTTP) covering:
  - Move requests (inputs/outputs, required GameState fields, error semantics).  
  - Choice requests (per PlayerChoice type, including how options are encoded).  
  - Timeouts, retry strategies, and fallback rules.  
- **Configuration‑Driven Behaviour:** Move AI policy decisions out of scattered code into a central configuration model:
  - Per‑profile toggles for “service‑backed vs local heuristic” per choice type.  
  - Difficulty‑dependent behaviour (e.g., which rules/heuristics to apply on the Python side).  
- **Unified AI Path:** Ensure all AI decisions (moves and choices) flow through the same high‑level path: GameEngine → PlayerInteractionManager → DelegatingInteractionHandler → AIInteractionHandler → AIEngine → AIServiceClient. Document this path and use it consistently in tests and diagrams.  
- **Observability:** Add structured logging and basic metrics around AI calls (latency, error rates, fallback counts) to support future stronger AI and service tuning.

### 3. WebSocket/Game Loop Reliability

**Scope**  
WebSocket server (`src/server/websocket/server.ts`), WebSocketInteractionHandler, GameContext/ChoiceDialog on the client, and the orchestration of human/AI turns and PlayerChoices over Socket.IO.

**Current Position**
- WebSocket server and event types exist; backend games can be driven via WebSockets with both human and AI players.  
- PlayerChoice flows (for humans) and AI turns are integrated into the game loop.  
- There are focused integration tests for AI turns and some choice flows.

**Key Risks / Technical Debt**
- WebSocket responsibilities (join/leave, reconnection, lobby/matchmaking, spectator support, robust error handling) are only partially implemented.  
- The game loop’s behaviour under adverse conditions (disconnects, late or duplicate choice responses, AI timeouts) is not comprehensively specified or tested.  
- Message schemas are mostly centralized but not yet treated as a versioned contract for the client.

**Recommended Direction**
- **Clarify Game Loop Semantics:** Document the authoritative turn/phase/choice lifecycle as it appears over WebSockets, including:
  - How and when GameState snapshots are emitted.  
  - How PlayerChoices are requested, acknowledged, and resolved (including timeouts and server‑side defaults).  
  - How AI turns are sequenced relative to human turns and UI updates.
- **Solidify Event Contracts:** Treat the WebSocket message types in `src/shared/types/websocket.ts` as a public API and:
  - Ensure every server event has an explicit, typed payload and corresponding client handler.  
  - Add simple versioning and compatibility notes to avoid accidental breaking changes.  
- **Resilience Patterns:** Define and implement patterns for:
  - Reconnection and resynchronization (how a client re‑joins and re‑hydrates state).  
  - Handling duplicate/stale `player_choice_response` messages safely.  
  - Graceful degradation when AI or backend operations fail mid‑turn.

### 4. Testing & Quality Gates

**Scope**  
Jest test suites (unit + integration), test utilities, CI configuration, and quality thresholds (coverage, linting, type‑checking).

**Current Position**
- Jest, ts‑jest, CI, and basic thresholds are configured.  
- There is a growing set of targeted tests for BoardManager, RuleEngine movement/capture, GameEngine chain captures and territory disconnection, PlayerInteractionManager, WebSocketInteractionHandler, AIEngine/AIServiceClient, AIInteractionHandler, and some GameEngine‑level integrations.

**Key Risks / Technical Debt**
- Overall coverage remains low relative to the rules’ complexity; many rule/FAQ scenarios are still not represented in tests.  
- Some older assessments/documents assume either “no tests” or “almost no engine implementation”, which is now outdated and can mislead future contributors.  
- CI thresholds and test structure are not yet aligned with the four axes; tests are not clearly partitioned by domain (rules/state, AI boundary, WebSocket/game loop, UI integration).

**Recommended Direction**
- **Rule‑Driven Scenario Suites:** Build explicit scenario suites keyed to `ringrift_complete_rules.md` and FAQ sections, organized by rule cluster rather than by file.  
- **Axis‑Oriented Test Groups:** Make it easy to run and reason about tests per refactoring axis (e.g., `rules-state`, `ai-boundary`, `websocket-loop`, `ui-integration`). This can be as simple as directory naming and npm scripts.  
- **Realistic Coverage Targets:** Use staged coverage goals (per axis and per module) instead of a single global number; raise them as the suite matures.  
- **Quality Gates in CI:** Once test suites per axis are solid, treat failures as blockers and avoid merging changes that dilute coverage or break key invariants without updating tests and docs.

These four axes should be treated as the backbone for ongoing architectural work. `STRATEGIC_ROADMAP.md`, `RINGRIFT_IMPROVEMENT_PLAN.md`, `CURRENT_STATE_ASSESSMENT.md`, `CODEBASE_EVALUATION.md`, and `TODO.md` each reference them from different angles (timeline, improvement focus, factual status, evaluation, and tasks respectively) and should remain synchronized with this assessment.

**Next Review:** After Phase 1 completion  
**Document Version:** 1.1  
**Maintained By:** Development Team
