# RingRift Project Goals

> **Doc Status (2025-11-27): Active (project direction SSoT)**
>
> - Single authoritative source for product/technical goals and non-goals.
> - Not a rules or lifecycle SSoT; for rules semantics defer to `ringrift_complete_rules.md` + `RULES_CANONICAL_SPEC.md` + shared TS engine, and for lifecycle semantics defer to `docs/CANONICAL_ENGINE_API.md` and shared WebSocket types/schemas.

**Version:** 1.0  
**Created:** November 26, 2025  
**Status:** Authoritative Source for Project Direction

> This document is the **single authoritative source** for RingRift’s product and technical goals. When other documents conflict with this one regarding project direction or success criteria, this document takes precedence.
>
> It is intentionally goal-focused: it describes **what we are trying to achieve**, not the live implementation status or detailed rules semantics.

## Source of truth and related documents

- **Goals SSoT (this file):** Canonical statement of product experience goals and high-level technical objectives (engine SSOT, parity, SLOs). Use this document as the primary reference for _what success looks like_.
- **Implementation status & metrics SSoT:** For current implementation status, test counts, and coverage metrics, defer to [`CURRENT_STATE_ASSESSMENT.md`](CURRENT_STATE_ASSESSMENT.md:1). That file is the single source of truth for live numbers and status labels.
- **Rules-specific status:** For rules-focused status and navigation across rules docs and verification/audit reports, see [`CURRENT_RULES_STATE.md`](CURRENT_RULES_STATE.md:1).
- **Specialised goal surfaces (subordinate docs):**
  - [`AI_ARCHITECTURE.md`](AI_ARCHITECTURE.md:1) – AI-specific goals, architecture, and training/improvement plans, all subordinate to the product/technical goals defined here.
  - [`STRATEGIC_ROADMAP.md`](STRATEGIC_ROADMAP.md:1) – phased execution plan and SLO roadmap that operationalises the goals in this file; it should be read as an implementation roadmap, not as a separate goals SSoT.

---

## 1. Purpose & Vision

RingRift exists to deliver a polished, production-ready online implementation of the RingRift tabletop strategy game as a high-quality environment for human and AI players to explore deep, perfect-information multiplayer gameplay.

RingRift is a web-based multiplayer abstract strategy game where 2-4 players compete for territory and ring elimination through strategic stack building, marker placement, and tactical captures. The game features no chance elements and perfect information, with a focus on multi-player dynamics that create shifting alliances and emergent strategic depth. Our goal is to deliver a polished, production-ready online implementation that faithfully captures the richness of the tabletop experience while leveraging digital capabilities for AI opponents, real-time multiplayer, and accessible cross-platform play.

### 1.1 Ruleset Design Goals & Rationale

The RingRift ruleset is intentionally designed around a **modest number of simple, universal mechanics**—stack building and movement, overtaking vs. elimination captures, line formation and collapse, territory disconnection, and last-player-standing turn semantics. These mechanics are combined to achieve the following design goals:

- **High emergent complexity from simple rules.**
  The core rules are meant to be teachable and mechanically consistent across board types and player counts, while still producing a rich decision space with deep tactical and strategic play. Complexity arises from the interaction of stacking, movement constraints, capture chains, line rewards, and territory disconnection, not from rule bloat. This is the primary reason RingRift exists as a distinct abstract strategy game rather than reusing an existing design.

- **Exciting, tense, and strategically non-trivial games.**
  Games should remain **live and contested** for a long time: temporary leads in ring count or territory are deliberately not a straightforward proxy for eventual victory. Multiple victory paths (ring elimination, territory control, last-player-standing) plus multi-player dynamics and chain reactions are designed so that:
  - Seemingly “won” positions can still collapse through territory cascades or alliance shifts.
  - Short-term sacrifices and self-elimination can be correct play toward long-term advantage.
  - The board’s geometry (lines and region disconnections) constantly reshapes incentives and risk.

- **Human–computer competitive balance in a perfect-information, zero-sum game.**
  Unlike many classic perfect-information strategy games where engines quickly outclass humans, RingRift is explicitly tuned so that strong human players can **compete with and sometimes outplay strong AIs**:
  - The default 3-player configuration and multi-player variants create social and political dynamics (temporary alliances, leader-punishing behaviour) that are difficult for purely algorithmic agents to model.
  - The extremely high branching factor (up to millions of choices per turn), long tactical chains (especially in captures and territory disconnections), and subtle tradeoffs between rings, markers, and territory make exhaustive search prohibitively expensive even for well-optimized engines.
  - The rules support a spectrum of AI strengths while intentionally preserving room for human creativity, intuition, long term strategic play, coalition forming, and non-myopic tactical planning to matter at all skill levels.

Together, these goals define **how the game should feel**: simple to describe at the rules level, but with deep, emergent strategy; high tension and comeback potential; and a long-term target where humans and AIs can meaningfully co-exist as competitive opponents. Detailed rules semantics, examples, and strategy notes are defined in the authoritative rulebook [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1) and the canonical specification [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:1); this section records the *purpose* those rules are serving.

---

## 2. Target Users

### Primary Users

- **Strategy Game Enthusiasts**: Players who enjoy abstract strategy games like Go, Chess, YINSH, TZAAR, and DVONN. They value deep tactical gameplay, perfect information, and games where skill determines outcomes.

- **Online Board Game Players**: Users seeking asynchronous or real-time multiplayer board game experiences accessible from web browsers without downloads or installations.

- **AI Training Researchers**: Developers and researchers interested in game AI, using RingRift as a testbed for heuristic evaluation, MCTS, minimax, and neural network approaches in a complex multi-player domain.

### User Characteristics

- Comfortable with learning moderately complex rule systems
- Appreciate games that reward strategic thinking over luck
- Range from casual players (8×8 simplified version) to serious competitors (19×19 and hex full versions)
- May play with 2, 3, or 4 human/AI players in any combination
- Expect smooth, responsive web-based gameplay

---

## 3. Current Phase Objectives

> For the current v1.0 phase, these objectives describe what the project must deliver in terms of gameplay features, architecture, and quality. Read them together with [`STRATEGIC_ROADMAP.md`](STRATEGIC_ROADMAP.md:1) for the phased execution plan and [`CURRENT_STATE_ASSESSMENT.md`](CURRENT_STATE_ASSESSMENT.md:1) for factual implementation status.

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

| Objective                       | Description                                                                                                                                                        | Rationale                     |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------- |
| **Comprehensive test coverage** | Extensive automated test suites across TypeScript and Python (unit, integration, scenario, parity, and E2E); see [`CURRENT_STATE_ASSESSMENT.md`](CURRENT_STATE_ASSESSMENT.md:1) for live counts and coverage details. | Confidence in correctness     |
| **Rules/FAQ scenario matrix**   | Test cases derived directly from [`ringrift_complete_rules.md`](ringrift_complete_rules.md) FAQ examples                                                           | Ensures rules fidelity        |
| **Contract testing**            | Cross-language parity validated by shared contract test vectors                                                                                                    | Guarantees engine consistency |
| **Parity harnesses**            | Backend ↔ sandbox ↔ Python engine behavior validation                                                    | Catches divergence early      |
| **CI/CD pipeline**              | Automated testing, linting, security scanning on every change                                            | Maintains quality bar         |

### 3.4 Current Highest-Risk Rules Semantics Area (Host Integration & Deep Parity)

> **Status (2025-11-30): Highest-risk semantic area and hardest outstanding problem**

Following the successful remediation of Active-No-Moves (ANM) semantics, the highest risk has shifted to **deep multi-engine parity and host integration** for advanced phases.

- **Weakest aspect (host integration & parity):**
  Ensuring that the **backend GameEngine** and **client SandboxEngine** correctly integrate the shared rules helpers for complex multi-phase turns involving:
  - **Capture chains:** Correctly enumerating and enforcing mandatory continuation across all board types.
  - **Territory disconnection:** Correctly processing simultaneous line formation and territory collapse without violating Q23 or S-invariants.
  - **RNG parity:** Ensuring sandbox AI simulation reliably predicts backend AI behaviour for the same seed.

- **Hardest outstanding problem (operational execution):**
  **Orchestrator-first rollout execution & deep parity verification.**
  - Moving from "design complete" to "production reality" by executing the phased rollout in staging/prod with strict SLO enforcement.
  - Achieving 100% parity for the "long tail" of complex interactions (e.g., hex board territory disconnection with simultaneous line formation) where engines are most likely to diverge.

This area is governed by the canonical rules in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:1) and the parity framework in [`docs/INVARIANTS_AND_PARITY_FRAMEWORK.md`](docs/INVARIANTS_AND_PARITY_FRAMEWORK.md:1).

High-level risk framing and historical assessment for this area are summarised in [`WEAKNESS_ASSESSMENT_REPORT.md`](WEAKNESS_ASSESSMENT_REPORT.md:1).

---

## 4. Success Criteria / Metrics (v1.0 Readiness)

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
 
| Category                  | Requirement              | Current Status                                                                                             |
| ------------------------- | ------------------------ | ---------------------------------------------------------------------------------------------------------- |
| **TypeScript tests**      | All passing              | All TypeScript test suites passing; see `CURRENT_STATE_ASSESSMENT.md` for live counts and coverage details |
| **Python tests**          | All passing              | All Python test suites passing; see `CURRENT_STATE_ASSESSMENT.md` for live counts and coverage details     |
| **Contract vectors**      | 100% parity              | Contract-based TS↔Python parity suites passing                                                             |
| **Rules scenario matrix** | All FAQ examples covered | Coverage in progress                                                                                       |
| **Integration tests**     | Core workflows passing   | AI resilience, reconnection, sessions                                                                      |
 
> **Note:** Live test counts and coverage breakdowns are maintained in [`CURRENT_STATE_ASSESSMENT.md`](CURRENT_STATE_ASSESSMENT.md:236). This document is not the single source of truth for those numbers; it records only the high-level requirements.
 
#### Metrics & Test Suites (qualitative overview)
 
To avoid duplicating live metrics, this section describes the **shape** of the test and CI surface; concrete counts and coverage percentages remain in [`CURRENT_STATE_ASSESSMENT.md`](CURRENT_STATE_ASSESSMENT.md:1).
 
- **TypeScript suites:** Jest unit and integration tests for the shared engine, backend hosts, frontend client, and sandbox; Playwright E2E tests for core auth and gameplay flows.
- **Python suites:** pytest suites for the AI service covering rules engine behaviour, parity tests, training/heuristic harnesses, and invariants.
- **Cross-language & CI gates:** Contract-vector tests (TS and Python) plus orchestrator and host-parity suites that must be green in CI before promotion to staging/production.
 
### 4.3 Feature Completion Criteria

- [ ] All 24 FAQ scenarios from rules document have corresponding tests
- [ ] All three board types (8×8, 19×19, hex) fully playable
- [ ] All victory conditions (Ring Elimination, Territory Control, Last Player Standing) correctly implemented
- [ ] AI difficulty ladder (1-10) functional with fallback handling
- [ ] Backend-driven games playable end-to-end via web client
- [ ] Sandbox mode provides rules-complete local play
- [ ] Spectator mode allows read-only game viewing

### 4.4 Environment & Rollout Success Criteria

Environment posture and rollout discipline are first-class parts of v1.0 readiness, not an afterthought. At a high level, v1.0 is considered **environment‑ready** when:

- **Canonical orchestrator is authoritative in production**  
  - Production gameplay traffic (HTTP + WebSocket) flows through the shared turn orchestrator via the backend adapter; legacy/alternate turn paths are removed or quarantined behind explicit diagnostics flags.  
  - The effective production profile matches the **Phase 3–4 orchestrator‑ON presets** in `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` (TS rules authoritative, `ORCHESTRATOR_ADAPTER_ENABLED=true`, `ORCHESTRATOR_ROLLOUT_PERCENTAGE` at or near 100, shadow mode disabled for normal play).

- **Rollout phases are executed with SLO gates, not best‑effort**  
  - Staging runs in a sustained **Phase 1 – orchestrator‑only** posture, with SLOs from `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` §8 (error rate, invariant violations, parity alerts) monitored and enforced.  
  - Production progresses through **Phase 2 – legacy authoritative + shadow**, then **Phase 3–4 – incremental orchestrator rollout → orchestrator‑only**, only when SLOs and error budgets are respected over the defined windows.  
  - Rollback paths and circuit‑breaker behaviour for orchestrator regressions are documented and exercised in staging before use in production.

- **Invariants, parity, and AI healthchecks are part of promotion criteria**  
  - Orchestrator invariant metrics (`ringrift_orchestrator_invariant_violations_total{type,invariant_id}`) and Python strict‑invariant metrics (`ringrift_python_invariant_violations_total{invariant_id,type}`) have dashboards and alerts wired as in `docs/ALERTING_THRESHOLDS.md` and `docs/INVARIANTS_AND_PARITY_FRAMEWORK.md`.  
  - Cross‑language parity suites (contract vectors, plateau snapshots, line+territory snapshots) and the Python AI self‑play healthcheck profile (nightly) are expected to be **stable and green** before promoting builds to staging/production.

These criteria are intentionally high‑level and goal‑oriented; detailed rollout tables, environment presets, and alert thresholds live in `STRATEGIC_ROADMAP.md`, `CURRENT_STATE_ASSESSMENT.md`, `docs/ORCHESTRATOR_ROLLOUT_PLAN.md`, and `docs/ALERTING_THRESHOLDS.md`. When changing rollout strategy or SLOs, update those documents for implementation detail, and this section for the overarching success definition.

---

## 5. Scope & Non-Goals (v1.0 and beyond)

Sections 5–7 collectively define what is in scope for the current v1.0 phase (required features and user flows), what is intentionally deferred to later phases, and what is explicitly out of scope or constrained for this project.

### 5.1 Required Features

**Game Mechanics (Must Have)**

- Ring placement (1-3 rings on empty, 1 ring on stacks)
- Stack movement with minimum distance rules
- Overtaking captures with mandatory chain continuation
- Line detection and processing (4+ for 8×8, 4+ for 19×19/hex)
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

## 8. Relationship to Other Docs

This goals document sits at the top of the planning stack for **project direction**:

- [`PROJECT_GOALS.md`](PROJECT_GOALS.md:1) (this file) defines the high-level product and technical goals, success criteria, and scope boundaries for the current phase.
- [`STRATEGIC_ROADMAP.md`](STRATEGIC_ROADMAP.md:1) translates those goals into a phased implementation and SLO roadmap; when direction or success criteria change, update this file first and then adjust the roadmap.
- [`CURRENT_STATE_ASSESSMENT.md`](CURRENT_STATE_ASSESSMENT.md:1) reports factual, code-verified implementation status relative to these goals and the roadmap; it does not define new goals.
- [`archive/FINAL_ARCHITECT_REPORT.md`](archive/FINAL_ARCHITECT_REPORT.md:1) and other archived reports provide historical context; where they disagree with this document or the current roadmap/state assessment on direction, treat them as superseded.

The tables below group key related documents by role so readers can quickly jump between **goals**, **plan**, and **current reality**.

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
