# RingRift

> _A fractured arena where shifting rings of influence determine power — where moves create fault lines, and mastery comes from navigating both space and its ultimate collapse._

A web-based multiplayer strategy game featuring ring stacking, tactical captures, and territory control across multiple board configurations.

![RingRift CI/CD](https://github.com/an0mium/RingRift/actions/workflows/ci.yml/badge.svg)
![Parity CI Gate](https://github.com/an0mium/RingRift/actions/workflows/parity-ci.yml/badge.svg)

---

## The Game

RingRift is an abstract strategy game where you build stacks of rings, claim territory, and outmaneuver your opponents. Every move leaves a mark on the board, and the landscape is constantly shifting.

The game rewards both careful planning and bold plays. You'll form temporary alliances, execute dramatic chain captures, and watch as entire regions of the board collapse in your favor — or against you.

**Design Philosophy:** High emergent complexity from simple rules. Games remain live and contested — seemingly "won" positions can collapse through territory cascades, and short-term sacrifices can be correct play toward long-term advantage.

### Core Mechanics

- **Ring Stacking**: Build and move stacks of rings; control is determined by the top ring
- **Movement**: Move in straight lines, leaving markers behind as you go
- **Overtaking Captures**: Jump over enemy stacks to claim their top ring, adding it to your stack
- **Chain Captures**: Once you start capturing, continue until no captures remain
- **Line Formation**: Create lines of 4+ markers to collapse them into permanent territory
- **Territory Disconnection**: Cut off regions of the board to claim them entirely

### Victory Conditions

Win by achieving **any one** of three conditions:

1. **Ring Elimination**: Remove enough rings to reach the victory threshold
2. **Territory Control**: Control more than half of all board spaces
3. **Last Player Standing**: Be the only player who can still take meaningful actions

---

## Quick Start

### Prerequisites

- Node.js 18+ and npm 9+
- Docker and Docker Compose
- PostgreSQL 14+ and Redis 6+ (or use Docker)

### Setup

```bash
# Clone and install
git clone <repository-url>
cd ringrift
npm install

# Configure environment
cp .env.example .env

# Start services
docker-compose up -d postgres redis
npm run db:migrate
npm run db:generate

# Run development servers
npm run dev              # Both frontend and backend
# Or individually:
npm run dev:server       # Backend → http://localhost:3000
npm run dev:client       # Frontend → http://localhost:5173
```

### AI Service (Optional)

```bash
cd ai-service
source ../.venv/bin/activate
uvicorn app.main:app --port 8001 --reload
```

---

## Features

| Feature             | Description                                          |
| ------------------- | ---------------------------------------------------- |
| **Multiple Boards** | 8×8 square, 19×19 square, and hexagonal (469 spaces) |
| **2-4 Players**     | Human/AI combinations with flexible matchmaking      |
| **AI Opponents**    | 10-level difficulty ladder (Random → MCTS → Descent) |
| **Real-time Play**  | WebSocket-based live gameplay with spectator mode    |
| **Rating System**   | ELO-based rankings and leaderboards                  |
| **Replay System**   | Watch and analyze completed games                    |

---

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Client  │◄──►│   Express API   │◄──►│   PostgreSQL    │
│   (Vite/TS)     │    │   + WebSocket   │    │   + Redis       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                       ┌──────┴──────┐
                       │ AI Service  │
                       │  (Python)   │
                       └─────────────┘
```

### Technology Stack

| Layer      | Technologies                                          |
| ---------- | ----------------------------------------------------- |
| Frontend   | React 18, TypeScript, Vite, Tailwind CSS, React Query |
| Backend    | Node.js, Express, Socket.IO, Prisma ORM, Zod          |
| Database   | PostgreSQL, Redis                                     |
| AI Service | Python, FastAPI, PyTorch (MCTS, neural networks)      |
| Monitoring | Prometheus, Grafana, CloudWatch                       |

### Rules Engine

The game logic is implemented in a canonical TypeScript engine with Python parity:

- **Single Source of Truth**: `src/shared/engine/` with 8 canonical phases
- **Cross-language Parity**: 81 contract vectors, 100% TS↔Python match
- **Domain Aggregates**: Placement, Movement, Capture, Line, Territory, Victory

---

## Testing

```bash
npm test                          # All Jest tests
npm run test:coverage             # Coverage report
npm run test:orchestrator-parity  # Rules parity tests

cd ai-service && pytest           # Python tests
```

| Metric           | Value               |
| ---------------- | ------------------- |
| TypeScript tests | 10,177 (595 suites) |
| Python tests     | 1,727               |
| Contract vectors | 81 (100% parity)    |
| Line coverage    | ~69%                |

---

## Documentation

| Document                                                   | Purpose                     |
| ---------------------------------------------------------- | --------------------------- |
| [QUICKSTART.md](QUICKSTART.md)                             | Detailed setup guide        |
| [ringrift_complete_rules.md](ringrift_complete_rules.md)   | Full rulebook with examples |
| [RULES_CANONICAL_SPEC.md](RULES_CANONICAL_SPEC.md)         | Formal rules specification  |
| [CONTRIBUTING.md](CONTRIBUTING.md)                         | Contribution guidelines     |
| [CURRENT_STATE_ASSESSMENT.md](CURRENT_STATE_ASSESSMENT.md) | Project status              |
| [docs/INDEX.md](docs/INDEX.md)                             | Full documentation index    |

---

## API Overview

### REST Endpoints

```
POST /api/auth/register        # User registration
POST /api/auth/login           # Authentication
GET  /api/games                # List games
POST /api/games                # Create game
GET  /api/games/:id            # Game details
POST /api/games/:id/join       # Join game
GET  /api/users/leaderboard    # Rankings
```

### WebSocket Events

| Client → Server          | Server → Client          |
| ------------------------ | ------------------------ |
| `join_game`              | `game_state`             |
| `player_move`            | `game_over`              |
| `player_choice_response` | `player_choice_required` |
| `chat_message`           | `error`                  |

---

## Production Deployment

```bash
npm run build
docker-compose up -d
```

The Docker stack includes: app, nginx, postgres, redis, ai-service, prometheus, grafana.

> **Note**: The backend is designed for single-instance deployment. See [QUICKSTART.md](QUICKSTART.md) for multi-instance considerations.

---

## Project Status

**Stable Beta** – Engine complete, production validation in progress.

- 14 development waves complete
- All core mechanics implemented and tested
- Active focus: scaling tests, security hardening, UX polish

For detailed status, see [CURRENT_STATE_ASSESSMENT.md](CURRENT_STATE_ASSESSMENT.md) and [TODO.md](TODO.md).

---

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License – see LICENSE file for details.

---

_Built by the RingRift Team_
