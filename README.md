# RingRift - Multiplayer Strategy Game

⚠️ **PROJECT STATUS: PLAYABLE CORE LOGIC 75% COMPLETE - UI AND TESTING NEEDED** ⚠️

> **Important:** Core game mechanics are largely implemented (~75%) but the project **lacks a playable UI and comprehensive testing**. The game cannot be played visually yet. Critical features like player choice system and chain capture enforcement are missing. See [CURRENT_STATE_ASSESSMENT.md](./CURRENT_STATE_ASSESSMENT.md) for verified status.

A web-based multiplayer implementation of the RingRift strategy game supporting 2-4 players with flexible human/AI combinations across multiple board configurations.

## 📋 Current Status

**Last Updated:** November 13, 2025  
**Verification:** Code-verified assessment  
**Overall Progress:** 58% Complete (strong foundation, critical gaps remain)

### ✅ What's Working (75% of Core Logic)
- ✅ Project infrastructure (Docker, database, Redis, WebSocket)
- ✅ TypeScript type system and architecture (100%)
- ✅ Comprehensive game rules documentation
- ✅ Server and client scaffolding
- ✅ **Marker system** - Placement, flipping, collapsing fully functional
- ✅ **Movement validation** - Distance rules, path checking working
- ✅ **Basic captures** - Single captures work correctly
- ✅ **Line detection** - All board types (8x8, 19x19, hexagonal)
- ✅ **Territory disconnection** - Detection and processing implemented
- ✅ **Phase transitions** - Correct game flow through all phases
- ✅ **Player state tracking** - Ring counts, eliminations, territory
- ✅ **Hexagonal board support** - Full 331-space board validated

### ⚠️ Critical Gaps (Blocks Playability)
- ❌ **Player choice system NOT implemented** - All decisions default (NO player agency)
- ❌ **Chain captures NOT enforced** - Mandatory continuation missing
- ❌ **No playable UI** - Board rendering not implemented (cannot see or play game)
- ❌ **Limited testing** - Cannot verify rule compliance (<10% coverage)
- ❌ **AI service not integrated** - Python service exists but disconnected

### 🎯 What This Means
**Can Do:**
- Create games programmatically via TypeScript
- Execute basic moves (ring placement, movement, single captures)
- Process lines and territory disconnection
- Track game state through all phases

**Cannot Do:**
- Play the game visually (no UI)
- Make strategic choices (all default to first option)
- Execute chain captures (not mandatory)
- Play against AI (not connected)
- Verify rules work (no comprehensive tests)
- Play multiplayer (infrastructure exists but not functional)

### 📊 Component Status
| Component | Status | Completion |
|-----------|--------|-----------|
| Type System | ✅ Complete | 100% |
| Board Manager | ✅ Complete | 90% |
| Game Engine | ⚠️ Partial | 75% |
| Rule Engine | ⚠️ Partial | 60% |
| Frontend UI | ❌ Skeleton | 10% |
| AI Integration | ❌ Not Connected | 40% |
| Testing | ❌ Minimal | 5% |
| Multiplayer | ❌ Infrastructure Only | 30% |

**For complete assessment, see [CURRENT_STATE_ASSESSMENT.md](./CURRENT_STATE_ASSESSMENT.md)**  
**For detailed issues, see [KNOWN_ISSUES.md](./KNOWN_ISSUES.md)**  
**For roadmap, see [TODO.md](./TODO.md)**

---

## 🎯 Overview

RingRift is a sophisticated turn-based strategy game featuring:
- **Multiple Board Types**: 8x8 square, 19x19 square, and hexagonal layouts
- **Flexible Player Support**: 2-4 players with human/AI combinations
- **Real-time Multiplayer**: WebSocket-based live gameplay
- **Spectator Mode**: Watch games in progress
- **Rating System**: ELO-based player rankings
- **Time Controls**: Configurable game timing
- **Cross-platform**: Web-based for universal accessibility

## 🏗️ Architecture

### Technology Stack

#### Backend
- **Runtime**: Node.js with TypeScript
- **Framework**: Express.js with comprehensive middleware
- **Database**: PostgreSQL with Prisma ORM
- **Real-time**: Socket.IO for WebSocket communication
- **Caching**: Redis for session management and game state
- **Authentication**: JWT-based with bcrypt password hashing
- **Validation**: Zod schemas for type-safe data validation
- **Logging**: Winston for structured logging
- **Security**: Helmet, CORS, rate limiting

#### Frontend
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite for fast development and optimized builds
- **Routing**: React Router for SPA navigation
- **State Management**: React Query for server state, Context API for client state
- **Styling**: Tailwind CSS for utility-first styling
- **WebSocket**: Socket.IO client for real-time communication
- **HTTP Client**: Axios with interceptors for API communication

#### Infrastructure
- **Containerization**: Docker with multi-stage builds
- **Orchestration**: Docker Compose for development
- **Database**: PostgreSQL with connection pooling
- **Caching**: Redis for high-performance data access
- **Environment**: Environment-based configuration

### System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Client  │    │   Express API   │    │   PostgreSQL    │
│                 │    │                 │    │                 │
│ • Game UI       │◄──►│ • REST Routes   │◄──►│ • Game Data     │
│ • Real-time     │    │ • WebSocket     │    │ • User Profiles │
│ • State Mgmt    │    │ • Game Engine   │    │ • Match History │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │      Redis      │              │
         └──────────────►│                 │◄─────────────┘
                        │ • Sessions      │
                        │ • Game Cache    │
                        │ • Rate Limiting │
                        └─────────────────┘
```

## ⚠️ Development Notice

**This application is not yet functional for gameplay.** The current codebase provides:
- Infrastructure setup and configuration
- Type definitions and data structures
- Game rules documentation
- Basic class skeletons

**To contribute or continue development, please review:**
1. [IMPLEMENTATION_STATUS.md](./IMPLEMENTATION_STATUS.md) - Detailed analysis of current state
2. [KNOWN_ISSUES.md](./KNOWN_ISSUES.md) - Specific bugs and missing features
3. [CONTRIBUTING.md](./CONTRIBUTING.md) - Development priorities and guidelines

---

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm 9+
- Docker and Docker Compose
- PostgreSQL 14+ (or use Docker)
- Redis 6+ (or use Docker)

### Development Setup

1. **Clone and Install**
```bash
git clone <repository-url>
cd ringrift
npm install
```

2. **Environment Configuration**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Database Setup**
```bash
# Start services with Docker
docker-compose up -d postgres redis

# Setup database
npm run db:migrate
npm run db:generate
```

4. **Start Development**
```bash
# Start both frontend and backend
npm run dev

# Or start individually
npm run dev:server  # Backend on :5000
npm run dev:client  # Frontend on :3000
```

### Production Deployment

```bash
# Build application
npm run build

# Start with Docker
docker-compose up -d

# Or manual deployment
npm start
```

## 🎮 Game Features

### Core Gameplay
- **Ring Placement**: Strategic positioning of rings on the board
- **Movement Phase**: Tactical ring repositioning
- **Row Formation**: Create rows of markers to remove opponent rings
- **Victory Conditions**: Remove required number of opponent rings

### Board Configurations
- **8x8 Square**: Compact tactical gameplay
- **19x19 Square**: Extended strategic depth
- **Hexagonal**: Unique geometric challenges

### Multiplayer Features
- **Real-time Synchronization**: Instant move updates
- **Spectator Mode**: Watch games with live commentary
- **Chat System**: In-game communication
- **Reconnection**: Seamless game resumption
- **Time Controls**: Blitz, rapid, and classical formats

### AI Integration
- **Difficulty Levels**: 1-10 skill ratings
- **Smart Opponents**: Strategic decision making
- **Mixed Games**: Human-AI combinations
- **Learning Algorithms**: Adaptive gameplay

## 🔧 API Documentation

### Authentication Endpoints
```
POST /api/auth/register    # User registration
POST /api/auth/login       # User authentication
GET  /api/auth/profile     # Get user profile
PUT  /api/auth/profile     # Update user profile
```

### Game Management
```
GET    /api/games          # List games
POST   /api/games          # Create new game
GET    /api/games/:id      # Get game details
POST   /api/games/:id/join # Join game
POST   /api/games/:id/leave # Leave game
POST   /api/games/:id/moves # Make move
```

### User Operations
```
GET /api/users             # List users
GET /api/users/:id         # Get user details
GET /api/users/leaderboard # Rating leaderboard
```

### WebSocket Events
```
join_game      # Join game room
leave_game     # Leave game room
player_move    # Send move
chat_message   # Send chat
game_update    # Receive game state
player_joined  # Player joined notification
player_left    # Player left notification
```

## 🛡️ Security Features

### Authentication & Authorization
- JWT token-based authentication
- Secure password hashing with bcrypt
- Role-based access control
- Session management with Redis

### API Security
- Rate limiting per endpoint
- CORS configuration
- Helmet security headers
- Input validation with Zod schemas
- SQL injection prevention with Prisma

### Game Security
- Move validation on server
- Anti-cheat mechanisms
- Secure WebSocket connections
- Game state integrity checks

## 📊 Performance Optimizations

### Backend Optimizations
- Database connection pooling
- Redis caching for frequent queries
- Efficient game state serialization
- Optimized database indexes
- Background job processing

### Frontend Optimizations
- Code splitting with React.lazy
- Memoization for expensive calculations
- Virtual scrolling for large lists
- Optimistic UI updates
- Service worker for offline capability

### Real-time Performance
- WebSocket connection pooling
- Efficient event broadcasting
- Delta updates for game state
- Compression for large payloads
- Heartbeat monitoring

## 🧪 Testing Strategy

### Backend Testing
```bash
npm test                   # Run all tests
npm run test:watch        # Watch mode
npm run test:coverage     # Coverage report
```

### Frontend Testing
```bash
npm run test:client       # Client tests
npm run test:e2e          # End-to-end tests
```

### Test Coverage
- Unit tests for game logic
- Integration tests for API endpoints
- WebSocket connection testing
- UI component testing
- End-to-end gameplay scenarios

## 📈 Monitoring & Analytics

### Application Monitoring
- Structured logging with Winston
- Error tracking and alerting
- Performance metrics collection
- Database query monitoring
- WebSocket connection analytics

### Game Analytics
- Player behavior tracking
- Game duration statistics
- Move pattern analysis
- Rating system metrics
- User engagement data

## 🔄 Development Workflow

### Code Quality
- TypeScript for type safety
- ESLint for code standards
- Prettier for formatting
- Husky for git hooks
- Conventional commits

### CI/CD Pipeline
- Automated testing on PR
- Code quality checks
- Security vulnerability scanning
- Automated deployment
- Database migration handling

## 📚 Additional Resources

### Game Rules
- Complete rule documentation in `ringrift_complete_rules.md`
- Interactive tutorial system
- Strategy guides and tips
- Video demonstrations

### Development Guides
- Architecture decision records
- API integration examples
- WebSocket implementation guide
- Database schema documentation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- Documentation: [Wiki](link-to-wiki)
- Issues: [GitHub Issues](link-to-issues)
- Discussions: [GitHub Discussions](link-to-discussions)
- Email: support@ringrift.com

---

Built with ❤️ by the RingRift Team
