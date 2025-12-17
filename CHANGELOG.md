# Changelog

All notable changes to RingRift will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-beta] - 2025-12-17

### Added

**Game Engine**

- Complete implementation of all 8 canonical game phases
- Support for 3 board types: 8×8 square, 19×19 square, and hexagonal (469 spaces)
- Support for 2-4 players with any mix of humans and AI
- Ring stacking mechanics with capture chains
- Line formation and territory processing
- Three victory conditions: Ring Elimination, Territory Control, Last Player Standing
- Recovery actions for players who lose all stacks
- Forced elimination for blocked players

**AI System**

- 10 AI difficulty levels from random to neural network-guided
- NNUE-style neural network for value and policy prediction
- MCTS integration with policy guidance
- Minimax with alpha-beta pruning
- GPU-accelerated parallel game generation for self-play
- Curriculum learning and difficulty mixing

**Multiplayer**

- Real-time WebSocket-based gameplay
- Spectator mode for watching games
- Chat system
- Game lobby with matchmaking
- Elo-based rating system and leaderboards

**Infrastructure**

- TypeScript shared engine (single source of truth)
- Python AI service with rules parity (81 contract vectors, 100% match)
- PostgreSQL database with Prisma ORM
- Redis for session management and caching
- Docker Compose for local development and production
- Prometheus/Grafana monitoring stack

**Documentation**

- Complete rulebook (`ringrift_complete_rules.md`)
- Canonical specification (`RULES_CANONICAL_SPEC.md`, v1.0)
- Compact implementation spec (`ringrift_compact_rules.md`)
- Human-friendly rules summary (`ringrift_simple_human_rules.md`)
- API documentation with Swagger
- Contributing guidelines

### Technical Notes

- 10,177 TypeScript tests across 595 suites
- 1,824 Python tests
- ~69% line coverage
- Canonical rules IDs (RR-CANON-RXXX) for formal verification

### Known Limitations

- No hosted demo yet (local setup required)
- Mobile UI not optimized
- AI response time can be slow at highest difficulty levels
- No tournament mode

---

## Release Types

- **Major versions** (X.0.0): Breaking changes to game rules or API
- **Minor versions** (0.X.0): New features, backwards compatible
- **Patch versions** (0.0.X): Bug fixes and minor improvements
- **Pre-release** (-alpha, -beta, -rc): Testing releases before stable
