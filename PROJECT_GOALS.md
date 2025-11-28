# RingRift Project Goals

> **Doc Status (2025-11-27): Active (project direction SSoT)**
>
> - Single authoritative source for product/technical goals and non-goals.
> - Not a rules or lifecycle SSoT; for rules semantics defer to `ringrift_complete_rules.md` + `RULES_CANONICAL_SPEC.md` + shared TS engine, and for lifecycle semantics defer to `docs/CANONICAL_ENGINE_API.md` and shared WebSocket types/schemas.

**Version:** 1.0  
**Created:** November 26, 2025  
**Status:** Authoritative Source for Project Direction

> This document is the **single authoritative source** for understanding what RingRift is, what it aims to achieve, and how success will be measured. When other documents conflict with this one regarding project direction, this document takes precedence.

---

## 1. Vision Statement

RingRift is a web-based multiplayer abstract strategy game where 2-4 players compete for territory and ring elimination through strategic stack building, marker placement, and tactical captures. The game features no chance elements and perfect information, with a focus on multi-player dynamics that create shifting alliances and emergent strategic depth. Our goal is to deliver a polished, production-ready online implementation that faithfully captures the richness of the tabletop experience while leveraging digital capabilities for AI opponents, real-time multiplayer, and accessible cross-platform play.

---

## 2. Target Users

### Primary Users

- **Strategy Game Enthusiasts**: Players who enjoy abstract strategy games like Go, Chess, YINSH, TZAAR, and DVONN. They value deep tactical gameplay, perfect information, and games where skill determines outcomes.

- **Online Board Game Players**: Users seeking asynchronous or real-time multiplayer board game experiences accessible from web browsers without downloads or installations.

- **AI Training Researchers**: Developers and researchers interested in game AI, using RingRift as a testbed for heuristic evaluation, MCTS, minimax, and neural network approaches in a complex multi-player domain.

### User Characteristics

- Comfortable with learning moderately complex rule systems
- Appreciate games that reward strategic thinking over luck
- Range from casual players (8×8 simplified version) to serious competitors (19×19 full version)
- May play with 2, 3, or 4 human/AI players in any combination
- Expect smooth, responsive web-based gameplay

---

## 3. Core Objectives

### 3.1 Product Objectives (Gameplay Features)

| Objective                          | Description                                                                                                                                                                                              | Rationale                                                      |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **Complete rules implementation**  | Faithfully implement all game mechanics from [`ringrift_complete_rules.md`](ringrift_complete_rules.md) including movement, captures, chains, lines, territory disconnection, and all victory conditions | Core game value depends on correct rules                       |
| **Multiple board types**           | Support 8×8 square (64 spaces), 19×19 square (361 spaces), and hexagonal (331 spaces) boards                                                                                                             | Provides accessibility gradient and geometric variety          |
| **Flexible player configurations** | Support 2-4 players with any combination of human and AI participants                                                                                                                                    | Enables solo practice, competitive play, and social gaming     |
| **AI opponents**                   | Provide AI opponents at multiple difficulty levels (1-10) with responsive move selection                                                                                                                 | Allows single-player experience and fills seats in multiplayer |
| **Real-time multiplayer**          | WebSocket-based live gameplay with synchronized state and spectator support                                                                                                                              | Core online multiplayer experience                             |
| **Victory tracking**               | Correct implementation of all three victory paths: Ring Elimination (>50% rings), Territory Control (>50% board), Last Player Standing                                                                   | Game cannot be complete without proper resolution              |

### 3.2 Technical Objectives (Architecture & Performance)

| Objective                     | Description                                                                      | Rationale                                       |
| ----------------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------- |
| **Canonical rules engine**    | Single shared rules engine in TypeScript used by both backend and client sandbox | Eliminates divergence, simplifies maintenance   |
| **Cross-language parity**     | Python AI service rules engine matches TypeScript implementation behavior        | Enables ML training on same rules as production |
| **Consolidated architecture** | Turn orchestrator pattern with domain aggregates and host adapters               | Clean separation of concerns, testability       |
| **Performance targets**       | AI moves <1s, UI updates <16ms, game state sync <200ms                           | Responsive user experience                      |
| **Scalable infrastructure**   | Docker-based deployment with PostgreSQL, Redis, and separate AI service          | Production-ready operations                     |

### 3.3 Quality Objectives (Testing & Reliability)

| Objective                       | Description                                                                                              | Rationale                     |
| ------------------------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------- |
| **Comprehensive test coverage** | 1000+ tests across TypeScript and Python codebases                                                       | Confidence in correctness     |
| **Rules/FAQ scenario matrix**   | Test cases derived directly from [`ringrift_complete_rules.md`](ringrift_complete_rules.md) FAQ examples | Ensures rules fidelity        |
| **Contract testing**            | Cross-language parity validated by shared test vectors                                                   | Guarantees engine consistency |
| **Parity harnesses**            | Backend ↔ sandbox ↔ Python engine behavior validation                                                  | Catches divergence early      |
| **CI/CD pipeline**              | Automated testing, linting, security scanning on every change                                            | Maintains quality bar         |

---

## 4. Success Criteria (v1.0 Readiness)

### 4.1 Performance SLOs

From [`STRATEGIC_ROADMAP.md`](STRATEGIC_ROADMAP.md:144-149):

| Metric              | Target          | Measurement                               |
| ------------------- | --------------- | ----------------------------------------- |
| **System uptime**   | >99.9%          | Core gameplay surfaces availability       |
| **AI move latency** | <1 second (p95) | Time from AI turn start to move broadcast |
| **UI frame rate**   | <16ms updates   | Smooth 60fps rendering during gameplay    |
| **Move validation** | <200ms (p95)    | Human move submission to broadcast        |
| **HTTP API**        | <500ms (p95)    | Login, game creation, state fetch         |

### 4.2 Test Coverage Requirements

| Category                  | Requirement              | Current Status                        |
| ------------------------- | ------------------------ | ------------------------------------- |
| **TypeScript tests**      | All passing              | 1195+ tests passing                   |
| **Python tests**          | All passing              | 245 tests passing                     |
| **Contract vectors**      | 100% parity              | 12 vectors, 100% pass rate            |
| **Rules scenario matrix** | All FAQ examples covered | Coverage in progress                  |
| **Integration tests**     | Core workflows passing   | AI resilience, reconnection, sessions |

### 4.3 Feature Completion Criteria

- [ ] All 24 FAQ scenarios from rules document have corresponding tests
- [ ] All three board types (8×8, 19×19, hex) fully playable
- [ ] All victory conditions (Ring Elimination, Territory Control, Last Player Standing) correctly implemented
- [ ] AI difficulty ladder (1-10) functional with fallback handling
- [ ] Backend-driven games playable end-to-end via web client
- [ ] Sandbox mode provides rules-complete local play
- [ ] Spectator mode allows read-only game viewing

---

## 5. MVP Scope (v1.0)

### 5.1 Required Features

**Game Mechanics (Must Have)**

- Ring placement (1-3 rings on empty, 1 ring on stacks)
- Stack movement with minimum distance rules
- Overtaking captures with mandatory chain continuation
- Line detection and processing (3+ for 8×8, 4+ for 19×19/hex)
- Territory disconnection detection and processing
- Graduated line rewards (Option 1 vs Option 2 for long lines)
- Forced elimination when blocked
- All three victory conditions with proper resolution
- Player choice system for all decision points

**Board Types (Must Have)**

- 8×8 square board (18 rings/player, 64 spaces)
- 19×19 square board (36 rings/player, 361 spaces)
- Hexagonal board (36 rings/player, 331 spaces, 11 per side)

**Game Modes (Must Have)**

- Backend multiplayer games (2-4 players via WebSocket)
- AI opponents (difficulty 1-10)
- Local sandbox mode (browser-only, full rules)
- Mixed human/AI player configurations

**Infrastructure (Must Have)**

- Docker-based deployment
- PostgreSQL database with Prisma ORM
- Redis for session caching and locking
- WebSocket server for real-time communication
- Python AI service with FastAPI
- JWT authentication

### 5.2 Required User Flows

1. **Account creation and login** - Register, authenticate, maintain session
2. **Game creation** - Choose board type, player count, AI configuration
3. **Game joining** - Browse lobby, join waiting games
4. **Gameplay** - Move selection, choice dialogs, state synchronization
5. **Game completion** - Victory display, return to lobby

---

## 6. Future Scope (Post-v1.0)

### 6.1 Planned for Later Phases

| Feature                   | Description                                                              | Phase     |
| ------------------------- | ------------------------------------------------------------------------ | --------- |
| **Advanced AI**           | Neural network-based AI, MCTS/minimax and descent at higher difficulties | Phase 4   |
| **Rating system**         | ELO-based player rankings with matchmaking                               | Phase 3   |
| **Automated matchmaking** | Queue-based matching by rating and preferences                           | Phase 3   |
| **Leaderboards**          | Global and periodic rankings                                             | Phase 3   |
| **In-game chat**          | Real-time messaging during games                                         | Phase 3   |
| **Game replays**          | Move-by-move replay viewer                                               | Phase 5   |
| **Tournament support**    | Structured competition format                                            | Future    |
| **Mobile optimization**   | Touch-friendly UI, responsive design polish                              | Post-v1.0 |
| **Game timers**           | Configurable time controls with UI display                               | Post-v1.0 |

### 6.2 Nice-to-Haves (Deferred)

- Interactive tutorial system
- Video demonstrations of rules
- Strategy guides within the application
- Achievement/badge system
- Social features (friends, invitations)
- Game analysis tools
- Multi-language support

---

## 7. Non-Goals (Explicitly Out of Scope)

### 7.1 What RingRift Will NOT Do

| Non-Goal                       | Rationale                                                                                                         |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| **Native mobile apps**         | Web-based approach ensures universal accessibility; native apps add maintenance burden without proportional value |
| **Random/chance elements**     | Core game design principle - RingRift is a perfect information game                                               |
| **Real-money wagering**        | Regulatory complexity; focus is on gameplay quality                                                               |
| **Social media integration**   | Privacy concerns; not core to gameplay experience                                                                 |
| **Blockchain/NFT features**    | Adds complexity without gameplay value                                                                            |
| **Embedded video/voice chat**  | Existing external solutions are adequate                                                                          |
| **Offline-first architecture** | Real-time multiplayer is core; local sandbox covers offline needs                                                 |

### 7.2 Boundaries and Constraints

- **Single-region deployment** for v1.0 (multi-region is post-v1.0)
- **Single app instance topology** is the supported production configuration
- **Browser-only client** (no desktop/mobile native apps planned)
- **English-only** for initial release
- **3 board types only** (8×8, 19×19, hex) - no custom board sizes
- **4 players maximum** per game

---

## 8. Document References

### 8.1 Implementation & Execution

| Document                                                     | Purpose                                   |
| ------------------------------------------------------------ | ----------------------------------------- |
| [`STRATEGIC_ROADMAP.md`](STRATEGIC_ROADMAP.md)               | Phased execution plan, SLOs, milestones   |
| [`TODO.md`](TODO.md)                                         | Task-level tracking, priority assignments |
| [`CURRENT_STATE_ASSESSMENT.md`](CURRENT_STATE_ASSESSMENT.md) | Code-verified implementation status       |
| [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md)                         | Active bugs and gaps with priorities      |

### 8.2 Rules & Design Authority

| Document                                                   | Purpose                                           |
| ---------------------------------------------------------- | ------------------------------------------------- |
| [`ringrift_complete_rules.md`](ringrift_complete_rules.md) | **Authoritative rulebook** - canonical game rules |
| [`ringrift_compact_rules.md`](ringrift_compact_rules.md)   | Implementation-oriented rules summary             |
| [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md)       | Technical specification of rule semantics         |
| [`RULES_SCENARIO_MATRIX.md`](RULES_SCENARIO_MATRIX.md)     | Test scenario mapping to rules/FAQ                |

### 8.3 Architecture & Technical

| Document                                                                                 | Purpose                         |
| ---------------------------------------------------------------------------------------- | ------------------------------- |
| [`ARCHITECTURE_ASSESSMENT.md`](ARCHITECTURE_ASSESSMENT.md)                               | System architecture review      |
| [`RULES_ENGINE_ARCHITECTURE.md`](RULES_ENGINE_ARCHITECTURE.md)                           | Rules engine design             |
| [`AI_ARCHITECTURE.md`](AI_ARCHITECTURE.md)                                               | AI service architecture         |
| [`src/shared/engine/orchestration/README.md`](src/shared/engine/orchestration/README.md) | Turn orchestrator documentation |

### 8.4 Operations & Development

| Document                             | Purpose                    |
| ------------------------------------ | -------------------------- |
| [`README.md`](README.md)             | Project overview and setup |
| [`QUICKSTART.md`](QUICKSTART.md)     | Getting started guide      |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Contribution guidelines    |
| [`docs/INDEX.md`](docs/INDEX.md)     | Documentation index        |

---

## Appendix A: Game Design Principles

_Derived from [`ringrift_complete_rules.md`](ringrift_complete_rules.md:153-177)_

RingRift's design is guided by these core principles:

1. **Perfect Information**: No hidden information, no random elements. All game state is visible to all players at all times.

2. **Deterministic Resolution**: Given the same inputs, the same outcomes always result. This enables reproducibility, AI training, and replay verification.

3. **Emergent Complexity**: Simple rules create complex interactions. Stack building, marker flipping, line formation, and territory disconnection interweave to produce deep strategic possibilities.

4. **Multi-Player Dynamics**: Designed for 3 players (extensible to 2-4), creating natural alliance formation and leader-balancing behavior that pure 2-player games lack.

5. **Dual Victory Paths**: Ring elimination (tactical) and territory control (strategic) provide multiple routes to victory, rewarding different play styles.

6. **Incremental Learning**: The 8×8 simplified version provides an accessible entry point; 19×19 and hexagonal versions offer increased depth for experienced players.

---

## Appendix B: Version History

| Version | Date       | Changes                                                                                                                                         |
| ------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.0     | 2025-11-26 | Initial creation consolidating goals from README.md, STRATEGIC_ROADMAP.md, CURRENT_STATE_ASSESSMENT.md, TODO.md, and ringrift_complete_rules.md |

---

_This document should be reviewed and updated whenever project direction changes significantly. It is intended to be read by all project contributors and stakeholders to ensure alignment on objectives and priorities._
