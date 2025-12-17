# RingRift

> _A fractured arena where shifting rings of influence determine power — where moves create fault lines, and mastery comes from navigating both space and its ultimate collapse._

A web-based multiplayer strategy game featuring ring stacking, tactical captures, and territory control. Play against friends or challenge AI opponents across multiple board sizes.

![RingRift CI/CD](https://github.com/an0mium/RingRift/actions/workflows/ci.yml/badge.svg)
![Parity CI Gate](https://github.com/an0mium/RingRift/actions/workflows/parity-ci.yml/badge.svg)

---

## The Game

RingRift is an abstract strategy game with zero randomness — every outcome is determined by player decisions. Build stacks of rings, claim territory through line formation, and outmaneuver your opponents as the board transforms beneath you.

**What makes it special:**

- High emergent complexity from simple rules
- Multiple victory paths keep games dynamic
- "Won" positions can collapse through cascading reactions
- Strong humans can compete with strong AI despite deep decision trees

### Core Mechanics

| Mechanic                    | Description                                                 |
| --------------------------- | ----------------------------------------------------------- |
| **Ring Stacking**           | Build and move stacks; the top ring determines control      |
| **Movement**                | Move in straight lines, leaving markers behind              |
| **Overtaking Captures**     | Jump over enemy stacks to claim their top ring              |
| **Chain Captures**          | Once you start capturing, continue until no captures remain |
| **Line Formation**          | Create lines of 4+ markers to collapse them into territory  |
| **Territory Disconnection** | Cut off regions of the board to claim them entirely         |

### Victory Conditions

Win by achieving **any one** of:

1. **Ring Elimination** — Remove enough opponent rings to reach the threshold
2. **Territory Control** — Control your fair share of spaces AND more than all opponents combined
3. **Last Player Standing** — Be the only player who can still take meaningful actions

---

## Quick Start

### Prerequisites

- Node.js 18+ and npm 9+
- Docker and Docker Compose (optional, for containerized setup)
- PostgreSQL 14+ and Redis 6+ (or use Docker)

### Setup

```bash
# Clone and install
git clone https://github.com/an0mium/RingRift.git
cd ringrift
npm install

# Configure environment
cp .env.example .env

# Start services (choose one):

# Option 1: Docker (recommended)
docker-compose up -d

# Option 2: Manual
docker-compose up -d postgres redis  # Just databases
npm run db:migrate
npm run db:generate
npm run dev  # Starts both frontend and backend
```

**URLs:**

- Frontend: http://localhost:5173
- Backend API: http://localhost:3000
- AI Service: http://localhost:8001 (if running)

### AI Service (Optional)

For AI opponents beyond the client-side sandbox:

```bash
cd ai-service
pip install -r requirements.txt
uvicorn app.main:app --port 8001 --reload
```

---

## Features

| Feature             | Description                                          |
| ------------------- | ---------------------------------------------------- |
| **Multiple Boards** | 8×8 square, 19×19 square, and hexagonal (469 spaces) |
| **2-4 Players**     | Any combination of humans and AI                     |
| **10 AI Levels**    | From random moves to neural network-guided search    |
| **Real-time Play**  | WebSocket-based with live state sync                 |
| **Rating System**   | Elo-based rankings and leaderboards                  |
| **Replay System**   | Watch and analyze completed games                    |
| **Spectator Mode**  | Watch ongoing games                                  |

---

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Client  │◄──►│   Express API   │◄──►│   PostgreSQL    │
│   (Vite + TS)   │    │   + WebSocket   │    │   + Redis       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                       ┌──────┴──────┐
                       │ AI Service  │
                       │  (Python)   │
                       └─────────────┘
```

### Tech Stack

| Layer      | Technologies                             |
| ---------- | ---------------------------------------- |
| Frontend   | React 18, TypeScript, Vite, Tailwind CSS |
| Backend    | Node.js, Express, Socket.IO, Prisma ORM  |
| Database   | PostgreSQL, Redis                        |
| AI Service | Python, FastAPI, PyTorch                 |
| Monitoring | Prometheus, Grafana                      |

### Rules Engine

The game logic lives in a canonical TypeScript engine with Python parity:

- **Single Source of Truth**: `src/shared/engine/` — 69 files, 8 canonical phases
- **Cross-language Parity**: 81 contract vectors ensure TS↔Python match exactly
- **Domain Aggregates**: Placement, Movement, Capture, Line, Territory, Victory

---

## Testing

```bash
# TypeScript tests
npm test
npm run test:coverage

# Python tests
cd ai-service && pytest

# Parity tests (TS↔Python rules)
npm run test:orchestrator-parity
```

| Metric           | Value               |
| ---------------- | ------------------- |
| TypeScript tests | 10,177 (595 suites) |
| Python tests     | 1,824               |
| Contract vectors | 81 (100% parity)    |
| Line coverage    | ~69%                |

---

## API Overview

### REST Endpoints

```
POST /api/auth/register     # Create account
POST /api/auth/login        # Authenticate
GET  /api/games             # List games
POST /api/games             # Create game
GET  /api/games/:id         # Game details
POST /api/games/:id/join    # Join game
GET  /api/users/leaderboard # Rankings
```

### WebSocket Events

| Client → Server          | Server → Client          |
| ------------------------ | ------------------------ |
| `join_game`              | `game_state`             |
| `player_move`            | `game_over`              |
| `player_choice_response` | `player_choice_required` |
| `chat_message`           | `error`                  |

---

## Documentation

| Document                                                 | Purpose                     |
| -------------------------------------------------------- | --------------------------- |
| [QUICKSTART.md](QUICKSTART.md)                           | Detailed setup guide        |
| [ringrift_complete_rules.md](ringrift_complete_rules.md) | Full rulebook with examples |
| [RULES_CANONICAL_SPEC.md](RULES_CANONICAL_SPEC.md)       | Formal rules specification  |
| [CONTRIBUTING.md](CONTRIBUTING.md)                       | Contribution guidelines     |
| [ai-service/README.md](ai-service/README.md)             | AI service documentation    |

---

## Production Deployment

```bash
npm run build
docker-compose -f docker-compose.prod.yml up -d
```

The Docker stack includes: app, nginx, postgres, redis, ai-service, prometheus, grafana.

See [QUICKSTART.md](QUICKSTART.md) for detailed deployment options.

---

## Project Status

**Stable Beta** — Engine complete, production validation in progress.

- 14 development waves complete
- All core mechanics implemented and tested
- Active focus: scaling tests, security hardening, UX polish

---

## Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and run tests
4. Commit (`git commit -m 'Add amazing feature'`)
5. Push (`git push origin feature/amazing-feature`)
6. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

_Built with passion for strategy games_
