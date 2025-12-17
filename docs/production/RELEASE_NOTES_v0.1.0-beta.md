# RingRift v0.1.0-beta Release Notes

> **Release Date:** TBD
> **Status:** Draft - First public beta release

## Overview

RingRift v0.1.0-beta is the first public release of RingRift, an abstract strategy game with zero randomness featuring ring stacking, tactical captures, and territory control.

## Highlights

### Complete Rules Engine

- **TypeScript + Python parity**: Identical game logic in both languages
- **81 contract vectors**: Ensuring cross-language correctness
- **Formal specification**: RR-CANON-RXXX rules with parameterized behavior
- **8 canonical aggregates**: Domain-driven design for complex game state

### 10-Level AI Difficulty Ladder

- **Level 0-2**: Random, Pure Heuristic, MCTS-100
- **Level 3-5**: MCTS-500, MCTS-1000, Neural-Guided MCTS
- **Level 6-8**: AlphaZero-style descent with increasing search depth
- **Level 9**: Tournament-strength with full neural network evaluation

### Multiple Board Configurations

- **8×8 Square**: Quick games, 18 rings/player
- **19×19 Square**: Strategic depth, 72 rings/player
- **Hexagonal (469 cells)**: Maximum complexity, 96 rings/player
- **2-4 players**: All configurations support multiplayer

### Real-Time Multiplayer

- WebSocket-based live game state synchronization
- Spectator mode for watching ongoing games
- Replay system for completed games
- Rating system with Elo-based rankings

## Technical Features

- **10,177 TypeScript tests** across 595 suites
- **1,824 Python tests** for AI service
- **~69% line coverage**
- **CI/CD pipeline** with parity validation gate

## Known Limitations

- **No hosted demo**: Local setup required (see Quick Start)
- **Developer-focused UX**: Not optimized for casual onboarding
- **Single-instance deployment**: No horizontal scaling yet
- **AI training ongoing**: Neural network models continue to improve

## Installation

```bash
# Clone and install
git clone https://github.com/an0mium/RingRift.git
cd ringrift
npm install

# Configure environment
cp .env.example .env

# Start with Docker (recommended)
docker-compose up -d

# Or start manually
npm run dev
```

**URLs after startup:**

- Frontend: http://localhost:5173
- Backend API: http://localhost:3000
- AI Service: http://localhost:8001 (optional)

## Documentation

- [Complete Rules](ringrift_complete_rules.md) - Full rulebook with examples
- [Canonical Specification](RULES_CANONICAL_SPEC.md) - Formal rules spec
- [Quick Start Guide](QUICKSTART.md) - Detailed setup instructions
- [AI Service](ai-service/README.md) - Neural network training documentation

## What's Next

- Hosted demo deployment
- Improved onboarding/tutorial experience
- Mobile-responsive UI
- Tournament system enhancements

## Contributors

Thanks to everyone who contributed to this release!

---

**Full Changelog**: https://github.com/an0mium/RingRift/commits/v0.1.0-beta
