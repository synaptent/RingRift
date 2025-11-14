# RingRift Strategic Roadmap

**Version:** 2.0  
**Created:** November 13, 2025  
**Last Updated:** November 13, 2025  
**Status:** Code-Verified Plan  
**Philosophy:** MVP-First, Testing-First, Playability-First

---

## 🎯 Executive Summary

**Current State:** 58% Complete (Strong foundation, critical gaps)  
**Goal:** Playable single-player game with AI opponents  
**Timeline:** 8-12 weeks to MVP  
**Strategy:** Complete core logic → Build minimal UI → Integrate AI → Comprehensive testing

**Key Insight:** The project has excellent architecture but needs focused execution on:
1. Player choice system (enables strategic gameplay)
2. Chain capture enforcement (critical rule)
3. Minimal playable UI (enables testing and use)
4. Python AI service integration (single-player mode)
5. Comprehensive testing (verification and confidence)

---

## 📊 Current State (Verified November 13, 2025)

### What Works ✅
- Type system and data structures (100%)
- Board management (90%): Position system, adjacency, markers
- Basic game mechanics (75%): Ring placement, movement, single captures
- Line detection and collapse (70%)
- Territory disconnection (70%)
- Phase transitions (85%)
- Infrastructure: Docker, PostgreSQL, Redis, WebSocket (95%)

### Critical Gaps ❌
- **Player choice system (0%)** - All decisions default
- **Chain captures (40%)** - Not enforced as mandatory
- **Playable UI (10%)** - Cannot see or interact with game
- **Testing (5%)** - Cannot verify correctness
- **AI integration (40%)** - Python service exists but disconnected

### Architecture Decision: **Keep Python AI Service** ✅
**Rationale:**
- Enables future machine learning capabilities
- GPU acceleration for neural networks
- scikit-learn, TensorFlow, PyTorch ecosystem
- Separation of concerns (game logic vs AI)
- Can add simple TypeScript AI for basic levels while keeping Python for advanced

---

## 🚀 Strategic Phases

### **PHASE 0: Testing Foundation** (1-2 weeks) - PARALLEL WITH PHASE 1
**Priority:** CRITICAL  
**Goal:** Enable confident development

#### 0.1 Jest/Vitest Setup (2-3 days)
- [x] Jest already configured
- [ ] Add test coverage reporting (target 80%+)
- [ ] Create test utilities and fixtures
- [ ] Add watch mode and coverage scripts
- [ ] Document testing patterns

#### 0.2 CI/CD Pipeline (2-3 days)
- [ ] Create GitHub Actions workflow
- [ ] Add linting step (ESLint --fix)
- [ ] Add type checking (tsc --noEmit)
- [ ] Add test execution
- [ ] Add coverage threshold enforcement
- [ ] Set up pre-commit hooks (Husky)

#### 0.3 Initial Test Coverage (3-5 days)
- [ ] Write tests for BoardManager core methods
- [ ] Write tests for RuleEngine validation
- [ ] Write tests for GameEngine state transitions
- [ ] Establish testing patterns for team

**Deliverable:** Automated testing pipeline with initial coverage

---

### **PHASE 1: Complete Core Game Logic** (2-3 weeks) - CRITICAL
**Priority:** P0  
**Goal:** Fully functional rule engine

#### 1.1 Player Choice System (1 week)
**NEW CRITICAL TASK**

**Design:**
```typescript
// src/shared/types/game.ts
interface PlayerChoice<T> {
  id: string;
  type: 'line_option' | 'ring_elimination' | 'region_order' | 'capture_direction';
  player: number;
  prompt: string;
  options: T[];
  timeout?: number;
  defaultOption?: T;
}

interface PlayerChoiceResponse<T> {
  choiceId: string;
  selectedOption: T;
}
```

**Implementation:**
- [ ] Create `PlayerInteractionManager.ts` class
- [ ] Add async choice request/response mechanism
- [ ] Integrate with GameEngine at all choice points:
  - [ ] Line processing order (multiple lines)
  - [ ] Graduated line rewards (Option 1 vs 2)
  - [ ] Ring/cap elimination selection
  - [ ] Region processing order
  - [ ] Capture direction (multiple valid)
- [ ] Add timeout handling (default to first option)
- [ ] Create choice validation logic
- [ ] Add AI decision hooks (for automated choices)

**Testing:**
- [ ] Unit tests for each choice type
- [ ] Integration tests for choice flow
- [ ] Timeout behavior tests

**Deliverable:** Strategic decisions are player-controlled, not defaults

#### 1.2 Chain Capture Enforcement (3-4 days)
**CRITICAL FIX**

- [ ] Add `chainCaptureInProgress` flag to GameState
- [ ] Modify phase transitions to enforce chain continuation
- [ ] Prevent other actions when chain active
- [ ] Implement `getAvailableChainCaptures()` method
- [ ] Add validation to require chain continuation
- [ ] Test 180° reversal patterns (FAQ Q15.3.1)
- [ ] Test cyclic patterns (FAQ Q15.3.2)

**Deliverable:** Chain captures work as specified in rules

#### 1.3 Comprehensive Rule Testing (1 week)
- [ ] Write scenario tests for all FAQ examples (Q1-Q24)
- [ ] Test all board types (8x8, 19x19, hexagonal)
- [ ] Test victory conditions
- [ ] Test edge cases
- [ ] Achieve 80%+ coverage on game logic

**Deliverable:** Verified rule compliance with comprehensive tests

**Phase 1 Success Criteria:**
- ✅ Can play complete game programmatically via tests
- ✅ All FAQ scenarios pass
- ✅ Player can make all strategic choices
- ✅ Chain captures enforced correctly
- ✅ 80%+ test coverage on game logic
- ✅ Zero critical TODOs in game flow

---

### **PHASE 2: Minimal Playable UI** (2-3 weeks) - HIGH PRIORITY
**Priority:** P1  
**Goal:** Visual game interface for 2-player local play

#### 2.1 Board Rendering (1 week)
- [ ] Create BoardGrid component (unified for square/hex)
- [ ] Implement square board rendering (8x8, 19x19)
- [ ] Implement hexagonal board rendering (331 spaces)
- [ ] Create Cell component with click handlers
- [ ] Add coordinate display overlay
- [ ] Responsive sizing for different screens
- [ ] Visual polish and animations

#### 2.2 Game Pieces & State (3-4 days)
- [ ] RingStack component (show individual rings)
- [ ] Marker component (player colors)
- [ ] CollapsedSpace component (claimed territory)
- [ ] CurrentPlayer indicator
- [ ] Ring count displays (in hand, on board, eliminated)
- [ ] Territory count display
- [ ] Move history list
- [ ] Victory progress indicators

#### 2.3 User Interaction (3-4 days)
- [ ] Click to select ring/stack
- [ ] Click to select destination
- [ ] Valid move highlighting
- [ ] Move confirmation dialog
- [ ] Undo/redo buttons (if time permits)

#### 2.4 Player Choice Dialogs (2-3 days)
- [ ] LineRewardChoice component (Option 1 vs 2)
- [ ] RingEliminationChoice component
- [ ] RegionOrderChoice component
- [ ] CaptureDirectionChoice component
- [ ] Generic ChoiceDialog component

**Phase 2 Success Criteria:**
- ✅ Can play 2-player local game with visual board
- ✅ All moves executable via mouse/touch
- ✅ All choices selectable via UI
- ✅ Game completes to victory condition
- ✅ Playtesters can complete games without bugs

---

### **PHASE 3: Python AI Integration** (2-3 weeks) - HIGH PRIORITY
**Priority:** P1  
**Goal:** Single-player mode with AI opponents

#### 3.1 AI Service Connection (3-4 days)
- [ ] Complete `AIServiceClient.ts` implementation
- [ ] Test HTTP connectivity to Python service
- [ ] Add error handling and retries
- [ ] Implement move evaluation endpoint calls
- [ ] Add AI service health checks
- [ ] Test service startup with Docker Compose

#### 3.2 Basic AI Levels in Python (1 week)
**Using existing Python service in `ai-service/`**

- [ ] Enhance RandomAI (difficulty 1-2)
  - [ ] Better move filtering
  - [ ] Basic illegal move avoidance
- [ ] Enhance HeuristicAI (difficulty 3-5)
  - [ ] Material evaluation (ring count)
  - [ ] Territory evaluation (collapsed spaces)
  - [ ] Mobility evaluation (valid moves)
  - [ ] Position scoring function
- [ ] Add difficulty parameter to API endpoints
- [ ] Test each difficulty level

#### 3.3 AI Player Integration (3-4 days)
- [ ] Add AI player type to game setup
- [ ] Integrate AI move selection in GameEngine
- [ ] Add move delay for human readability
- [ ] Handle AI decision for player choices
- [ ] Test human vs AI games
- [ ] Test AI vs AI games

#### 3.4 Advanced AI (Optional - Future)
**Leverage Python for ML capabilities**

- [ ] Implement MCTS (Monte Carlo Tree Search)
- [ ] Add opening book system
- [ ] Neural network considerations
- [ ] Self-play training infrastructure

**Phase 3 Success Criteria:**
- ✅ Can play single-player vs AI
- ✅ AI makes legal moves every time
- ✅ Difficulty levels feel different
- ✅ AI response time acceptable (<2 seconds)
- ✅ Python service stable and reliable

---

### **PHASE 4: Testing & Polish** (1-2 weeks) - HIGH PRIORITY
**Priority:** P1  
**Goal:** Production-ready single-player game

#### 4.1 Comprehensive Testing
- [ ] Integration tests for complete games
- [ ] E2E tests with UI interaction
- [ ] AI integration tests
- [ ] Performance testing
- [ ] Bug fixes from testing

#### 4.2 UX Polish
- [ ] Smooth animations
- [ ] Loading states
- [ ] Error messages
- [ ] Tutorial/help system
- [ ] Game rules reference in-app
- [ ] Keyboard shortcuts

#### 4.3 Documentation
- [ ] Update all documentation
- [ ] API documentation
- [ ] Deployment guide
- [ ] Contributing guide
- [ ] User manual

**Phase 4 Success Criteria:**
- ✅ Zero critical bugs
- ✅ Playtesters complete games without frustration
- ✅ All features documented
- ✅ Ready for beta release

---

### **PHASE 5: Multiplayer** (3-4 weeks) - MEDIUM PRIORITY
**Priority:** P2  
**Goal:** Online play with matchmaking

#### 5.1 WebSocket Game Sync (1 week)
- [ ] Move broadcasting
- [ ] Game state synchronization
- [ ] Reconnection handling
- [ ] Spectator support

#### 5.2 Lobby & Matchmaking (1 week)
- [ ] Game creation UI
- [ ] Lobby browser
- [ ] Player matching
- [ ] Game invitations

#### 5.3 Database Integration (1 week)
- [ ] Game persistence
- [ ] Move history storage
- [ ] User profiles
- [ ] Statistics tracking

**Phase 5 Success Criteria:**
- ✅ Can play multiplayer online
- ✅ Games persist and can be resumed
- ✅ Spectators can watch
- ✅ Reliable connection management

---

### **PHASE 6: Advanced Features** (4+ weeks) - LOW PRIORITY
**Priority:** P3 (Post-MVP)

#### 6.1 Advanced AI (Python ML)
- [ ] Neural network AI (levels 9-10)
- [ ] Self-play training
- [ ] GPU acceleration
- [ ] Model versioning

#### 6.2 Competitive Features
- [ ] ELO rating system
- [ ] Leaderboards
- [ ] Tournament support
- [ ] Replay system with analysis
- [ ] Game analysis tools

#### 6.3 Mobile & Polish
- [ ] Mobile app (React Native?)
- [ ] Progressive Web App
- [ ] Push notifications
- [ ] Achievements system

---

## 🎯 Immediate Next Steps (Next 2 Weeks)

### Week 1: Testing Foundation + Player Choices
**Days 1-2:** Set up CI/CD pipeline
- Configure GitHub Actions
- Add pre-commit hooks
- Set up test coverage reporting

**Days 3-5:** Implement Player Choice System
- Design choice interfaces
- Create PlayerInteractionManager
- Integrate with GameEngine
- Write tests

**Days 6-7:** Chain Capture Enforcement
- Add mandatory continuation logic
- Test chain scenarios
- Fix related bugs

### Week 2: Complete Core Logic Testing
**Days 8-10:** Comprehensive Rule Tests
- Write all FAQ scenario tests
- Test all board types
- Edge case coverage

**Days 11-12:** Bug Fixes
- Fix issues found in testing
- Refactor as needed

**Days 13-14:** Documentation & Prep for Phase 2
- Update all docs
- Plan UI architecture
- Design component hierarchy

---

## 📈 Success Metrics & Milestones

### Milestone 1: Core Logic Complete (Week 3-4)
**Metrics:**
- [ ] 80%+ test coverage on game logic
- [ ] All FAQ scenarios pass
- [ ] Zero P0/P1 issues in KNOWN_ISSUES.md
- [ ] Can play complete game via API/tests

### Milestone 2: Playable UI (Week 6-7)
**Metrics:**
- [ ] Can complete 2-player game with UI
- [ ] All moves/choices work via mouse
- [ ] 5+ playtest sessions completed successfully
- [ ] <10 critical UI bugs

### Milestone 3: AI Integration (Week 9-10)
**Metrics:**
- [ ] AI makes valid moves 100% of time
- [ ] 3+ difficulty levels working
- [ ] Python service uptime >99%
- [ ] AI response time <2 seconds

### Milestone 4: MVP Release (Week 12)
**Metrics:**
- [ ] Zero critical bugs
- [ ] 10+ beta testers playing
- [ ] Positive playtester feedback
- [ ] Production deployment successful

---

## 🏗️ Technical Architecture Decisions

### Why Keep Python AI Service ✅

**Advantages:**
1. **Machine Learning Ready:** TensorFlow, PyTorch, scikit-learn ecosystem
2. **GPU Acceleration:** CUDA support for neural networks
3. **Separation of Concerns:** Game logic separate from AI
4. **Language Strengths:** Python excels at ML/AI
5. **Future Scalability:** Can add multiple AI services

**Implementation:**
- Keep existing FastAPI service in `ai-service/`
- Complete `AIServiceClient.ts` for TypeScript integration
- Use REST API for move evaluation
- Consider gRPC for lower latency (future)
- Docker Compose orchestrates both services

**Fallback:**
- If Python service unavailable, use simple TypeScript AI
- Graceful degradation to random moves
- Health checks ensure service availability

### Architecture Diagram

```
┌──────────────────────────────────────┐
│          React Frontend              │
│   ┌──────────────────────────────┐   │
│   │   Game Board Components      │   │
│   │   Player Choice Dialogs      │   │
│   └──────────────────────────────┘   │
└─────────────┬────────────────────────┘
              │ WebSocket + HTTP
┌─────────────▼────────────────────────┐
│      TypeScript Game Server          │
│   ┌──────────────────────────────┐   │
│   │     GameEngine               │   │
│   │     RuleEngine               │   │
│   │     BoardManager             │   │
│   │  PlayerInteractionManager    │   │
│   └──────────────────────────────┘   │
│            │                          │
│            ├──► PostgreSQL            │
│            ├──► Redis Cache           │
│            │                          │
│            └──► AIServiceClient  ─────┼──┐
└──────────────────────────────────────┘  │
                                          │ HTTP
┌─────────────────────────────────────────▼─┐
│       Python AI Microservice              │
│   ┌─────────────────────────────────────┐ │
│   │  FastAPI Server                     │ │
│   │    ├── RandomAI (Levels 1-2)       │ │
│   │    ├── HeuristicAI (Levels 3-5)    │ │
│   │    ├── MCTSAI (Levels 6-8)         │ │
│   │    └── NeuralAI (Levels 9-10)      │ │
│   └─────────────────────────────────────┘ │
│              GPU (Optional)                │
└────────────────────────────────────────────┘
```

---

## ⚠️ Risk Management

### Technical Risks

**Risk 1: Player Choice System Complexity**
- **Mitigation:** Start with simple implementation, iterate
- **Fallback:** Default choices with warning to user

**Risk 2: Python AI Service Latency**
- **Mitigation:** Caching, move pre-computation
- **Fallback:** TypeScript AI for low latencies

**Risk 3: UI Complexity (Hexagonal Grid)**
- **Mitigation:** Start with square boards, add hex later
- **Fallback:** Simplified rendering

**Risk 4: Testing Time**
- **Mitigation:** Parallel testing with development
- **Fallback:** Focus on critical paths first

### Project Risks

**Risk 1: Scope Creep**
- **Mitigation:** Strict MVP definition, defer nice-to-haves
- **Response:** Regular scope reviews

**Risk 2: Timeline Overruns**
- **Mitigation:** Buffer time in estimates, agile sprints
- **Response:** Cut Phase 5/6 features if needed

**Risk 3: Technical Debt**
- **Mitigation:** Code reviews, refactoring sprints
- **Response:** Allocate 20% time for debt reduction

---

## 🔄 Development Workflow

### Sprint Structure (2-week sprints)
1. **Planning:** Define sprint goals from roadmap
2. **Development:** Focus on phase objectives
3. **Testing:** Continuous testing throughout sprint
4. **Review:** Demo and retrospective
5. **Deploy:** To staging environment

### Code Quality Gates
- ✅ TypeScript compiles without errors
- ✅ ESLint passes (no errors, warnings OK)
- ✅ Tests pass with 80%+ coverage
- ✅ Code reviewed by peer
- ✅ Documentation updated

### Definition of Done
- [ ] Code complete and reviewed
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] No critical bugs
- [ ] Deployed to staging

---

## 📚 Documentation Standards

### Required Documentation
1. **Code Comments:** JSDoc for all public methods
2. **README Updates:** Keep accurate status
3. **API Documentation:** OpenAPI spec for all endpoints
4. **Architecture Decisions:** ADR for major choices
5. **User Guide:** How to play

### Documentation Locations
- Technical docs: `docs/` directory
- API docs: OpenAPI in `api-docs/`
- User docs: Wiki or in-app help
- Development docs: `CONTRIBUTING.md`, `TODO.md`

---

## 🎓 Learning & Best Practices

### Key Principles
1. **Test-Driven Development:** Write tests first when possible
2. **Continuous Integration:** Every commit tested
3. **User-Centric:** Playability before features
4. **Iterative:** Working increments, not big bang
5. **Documentation:** Code is read more than written

### Common Pitfalls to Avoid
❌ Building features without tests  
❌ Premature optimization  
❌ Scope creep (adding features before MVP)  
❌ Ignoring technical debt  
❌ Building UI before backend stable  

### Success Patterns
✅ Small, focused PRs  
✅ Comprehensive testing  
✅ Regular refactoring  
✅ Clear communication  
✅ Working software over documentation  

---

## 🎯 Vision & Long-Term Goals

### 6-Month Vision (Post-MVP)
- **Active player base:** 100+ weekly active users
- **AI Skill:** Neural network AI competing with strong players
- **Platform:** Web + mobile apps
- **Community:** Tournament system, rankings

### 12-Month Vision
- **Machine Learning:** Self-play trained AI
- **Analytics:** Game analysis tools
- **Monetization:** Premium features (ad-free, advanced AI)
- **Competitive Scene:** Regular tournaments

### Technology Evolution
- **Frontend:** Consider migrating to Next.js for SSR
- **Backend:** Consider microservices if scale requires
- **AI:** Distributed training for neural networks
- **Database:** Consider sharding if data grows

---

## 📊 Resource Allocation

### Development Time Estimates
- **Phase 0:** 40-60 hours (1-2 weeks)
- **Phase 1:** 80-120 hours (2-3 weeks)
- **Phase 2:** 80-120 hours (2-3 weeks)
- **Phase 3:** 80-120 hours (2-3 weeks)
- **Phase 4:** 40-80 hours (1-2 weeks)
- **Total to MVP:** 320-500 hours (8-12 weeks)

### Team Recommendations
- **Solo Developer:** 12-16 weeks to MVP
- **2 Developers:** 8-10 weeks to MVP
- **3+ Developers:** 6-8 weeks to MVP

### Focus Areas by Role
**Full-Stack Developer:**
- Phases 0, 1, 2, 3, 4 (all phases)

**Backend Specialist:**
- Phase 0, 1 (core logic, testing)

**Frontend Specialist:**
- Phase 2 (UI), Phase 4 (polish)

**ML/AI Specialist:**
- Phase 3 (Python AI), Phase 6.1 (advanced AI)

**QA/Testing:**
- Phase 0, 4 (test infrastructure, comprehensive testing)

---

## ✅ Success Criteria

### MVP Complete When:
1. ✅ Can play single-player game vs AI with visual UI
2. ✅ All core rules correctly implemented
3. ✅ 80%+ test coverage on game logic
4. ✅ AI makes valid moves 100% of time
5. ✅ Zero critical bugs
6. ✅ 10+ successful beta test sessions
7. ✅ Complete documentation
8. ✅ Deployed to production

### Quality Bar:
- **Functionality:** All core features work correctly
- **Reliability:** <1% error rate in production
- **Performance:** AI response <2 seconds, UI responsive
- **Usability:** Playtesters complete games without help
- **Maintainability:** Code reviewed, tested, documented

---

## 🚀 Call to Action

### For Contributors
1. **Review** this roadmap and provide feedback
2. **Choose** a phase/task to work on
3. **Clone** the repository and set up environment
4. **Start** with Phase 0 or Phase 1.1
5. **Test** everything you build
6. **Document** your changes
7. **Submit** PRs for review

### For Users
1. **Star** the repository to follow progress
2. **Join** beta testing program (coming soon)
3. **Report** bugs and suggestions
4. **Share** the project with strategy game enthusiasts

### For Stakeholders
1. **Review** progress against milestones
2. **Provide** resources as needed
3. **Connect** with potential users
4. **Support** the development team

---

## 📝 Changelog

### Version 2.0 (November 13, 2025)
- Complete rewrite based on code verification
- Added Player Choice System as critical Phase 1 task
- Committed to keeping Python AI service
- Revised timeline based on actual completion status
- Added detailed implementation plans for each phase
- Clarified MVP definition and success criteria

### Version 1.0 (Previous)
- Initial roadmap (overly optimistic on completion status)

---

**Document Maintained By:** Development Team  
**Next Review:** Weekly during development  
**Questions?** See [CONTRIBUTING.md](./CONTRIBUTING.md) or open an issue

---

**Let's build an amazing strategy game! 🎮🚀**
