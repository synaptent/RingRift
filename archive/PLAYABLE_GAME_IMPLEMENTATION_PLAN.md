# ⚠️ DEPRECATED: RingRift Playable Game Implementation Plan

> **This is a historical document preserved for context.**
>
> **For current status and plans, see:**
>
> - [[`../docs/archive/historical/CURRENT_STATE_ASSESSMENT.md`](../docs/archive/historical/CURRENT_STATE_ASSESSMENT.md)](../docs/archive/historical/CURRENT_STATE_ASSESSMENT.md)
> - [`IMPLEMENTATION_STATUS.md`](../IMPLEMENTATION_STATUS.md)
> - [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md)
> - [`TODO.md`](../TODO.md)

**Created: November 15, 2025 · Status Updated: November 16, 2025**

> **Historical Note:** This document captures a snapshot of the end-to-end playability plan as of mid-November 2025. Many items described here have since been implemented or superseded. For canonical, up-to-date status and tasks, see [`../docs/archive/historical/CURRENT_STATE_ASSESSMENT.md`](../docs/archive/historical/CURRENT_STATE_ASSESSMENT.md), `IMPLEMENTATION_STATUS.md`, `TODO.md`, `STRATEGIC_ROADMAP.md`, and `KNOWN_ISSUES.md`.

## Executive Summary

After analyzing the current codebase, RingRift is **remarkably close** to being fully playable. The core infrastructure is solid:

- ✅ **Backend game creation works** - LobbyPage → REST API → Database persistence
- ✅ **WebSocket lifecycle is complete** - Game loading, engine initialization, move handling
- ✅ **AI integration is functional** - Service-backed AI with automatic turn handling
- ✅ **Choice system is wired** - Player decisions flow through WebSocket with timeout handling
- ✅ **Frontend displays games** - GamePage connects to backend, renders board, shows choices

**The gap to playability is smaller than expected.** Most issues are polish/UX rather than missing architecture.

---

## Documentation & Status Sources

To keep this implementation plan aligned with the rest of the project, treat the following documents as the primary, code-verified sources of truth for current status and architecture:

- [`../docs/archive/historical/CURRENT_STATE_ASSESSMENT.md`](../docs/archive/historical/CURRENT_STATE_ASSESSMENT.md) – factual, code-verified snapshot of what works today
- `KNOWN_ISSUES.md` – current bug/issue list, including P0/P1 priorities
- `TODO.md` – phase- and task-oriented development plan with progress markers
- `STRATEGIC_ROADMAP.md` – higher-level strategic phases and milestones
- `ARCHITECTURE_ASSESSMENT.md` – architecture and code-health evaluation, organised around the four refactoring axes (rules/state, AI boundary, WebSocket/game loop, testing & quality)

This `PLAYABLE_GAME_IMPLEMENTATION_PLAN.md` is intentionally focused on the **end-to-end playable experience** (from lobby through victory) and should be kept in sync with those documents when implementation status changes.

---

## Strategic Questions - ANSWERED

### 1. Backend-driven vs client-local sandbox?

**RECOMMENDATION: Focus on backend-driven games first**

**Rationale:**

- Backend infrastructure (routes, WebSocket, GameEngine persistence) is already built
- LobbyPage → `gameApi.createGame()` → navigation to `/game/:gameId` works
- WebSocket server properly initializes GameEngine from database with move replay
- AI turns execute automatically via `maybePerformAITurn()`
- Local sandbox is useful for quick testing but shouldn't block playability goals

**Current State:**

- GamePage already attempts backend-first: tries `createGame()` before falling back to local
- WebSocket server has comprehensive game lifecycle management
- Only gaps: game status transitions and victory detection UI

### 2. AI fallback strategy: service-backed vs local?

**RECOMMENDATION: Default to service-backed with graceful degradation**

**Rationale:**

- `globalAIEngine` already abstracts local vs service-backed AI
- WebSocket server creates AI from `aiProfile` persisted in game state
- AIEngine attempts service call and logs errors without crashing
- Service-backed AI enables richer strategic play (heuristic, minimax, MCTS)

**Implementation:**

```typescript
// In AIEngine, current behavior is already reasonable:
// 1. Try service-backed AI first
// 2. If service unavailable, log error and use local heuristic
// 3. Never crash the game loop

// Only missing: explicit fallback in AIEngine.getAIMove()
// Currently returns null on service failure; should construct local move
```

**Action Items:**

- [ ] Add explicit fallback in `AIEngine.getAIMove()` - if service fails, use local heuristic
- [ ] Display AI mode (service/local) in GamePage UI so users know which is active
- [ ] Add health check endpoint for AI service connectivity

### 3. Minimal viable HUD?

**RECOMMENDATION: 5 essential elements**

1. **Current player indicator** - Visual highlight of whose turn it is
2. **Phase indicator** - Placement / Movement / Line Resolution / etc.
3. **Ring counts** - Rings in hand / eliminated for each player
4. **Timer display** - Time remaining per player (optional for first version)
5. **Victory notification** - Clear modal when game ends

**Current State:**

- GamePage shows phase/player/status in small text (top-right)
- ChoiceDialog handles strategic decisions
- Missing: prominent turn indicator, ring counts, victory modal

### 4. Support all player counts or 2-player first?

**RECOMMENDATION: Support 2-4 players from the start**

**Rationale:**

- GameEngine, WebSocket, and LobbyPage already handle 2-4 players
- No architectural changes needed - just testing
- Game rules don't change based on player count (just more players)
- Limiting to 2-player first provides no implementation savings

**Current State:**

- Database schema supports player1Id through player4Id
- LobbyPage dropdown allows 2-4 players
- GameEngine creates players array from database
- WebSocket broadcasts to all participants

---

## Core Gaps Analysis

### Sandbox Stage 2 (Client-Local) Status ✅

- `ClientSandboxEngine` now drives a fully local GameState-shaped sandbox engine under `src/client/sandbox/ClientSandboxEngine.ts`.
- Sandbox line detection and rewards are aligned with the backend TS/Rust engines and covered by `tests/unit/ClientSandboxEngine.lines.test.ts`.
- Sandbox movement, captures (including mandatory chain continuation), and forced elimination are factored into `sandboxMovement.ts`, `sandboxCaptures.ts`, and `sandboxElimination.ts`, keeping logic reusable and close to backend semantics.
- Next sandbox parity steps: territory disconnection + victory detection mirroring backend `GameEngine`.

### Gap 1: Game Status Transitions ⚠️

**Current State:**

- Games created with status `waiting`
- Join endpoint transitions to `active` when ≥2 players
- No automatic start for AI-only games
- No completion/victory detection

**What's Needed:**

```typescript
// In WebSocket server after game engine creation:
// 1. Check if all AI opponents filled → auto-start
// 2. After each move, check GameEngine for victory conditions
// 3. Emit victory event and update database status
```

**Files to Modify:**

- `src/server/websocket/server.ts` - Add auto-start logic in `getOrCreateGameEngine()`
- `src/server/websocket/server.ts` - Add victory check in `handlePlayerMove()` and `maybePerformAITurn()`

### Gap 2: Victory Signalling & UI 🎯

**Current State:**

- `RuleEngine` and `GameEngine` already cooperate to perform a game-end check after moves via `checkGameEnd`, and `GameEngine.endGame` updates `GameState` with `winner` and a `GameResult`.
- `WebSocketServer.handlePlayerMove` currently ignores the `gameResult` returned by `makeMove` and always continues by broadcasting `game_state` only.
- The database game record is not yet consistently updated to `COMPLETED`/`winnerId` when the engine ends a game.
- The frontend (`GameContext` / `GamePage`) has no dedicated `game_over` signal or victory UI; users only see the final board state.

**What's Needed:**

```typescript
// In WebSocketServer.handlePlayerMove (and maybePerformAITurn):
const result = await gameEngine.makeMove(engineMove);

if (!result.success) { /* ... existing error handling ... */ }

if (result.gameResult) {
  // Persist DB status and emit a dedicated game_over event
  await prisma.game.update({
    where: { id: gameId },
    data: {
      status: GameStatus.completed,
      winnerId: /* map gameResult.winner to userId or AI marker */,
      endedAt: new Date(),
    },
  });

  this.io.to(gameId).emit('game_over', {
    type: 'game_over',
    data: result.gameResult,
    timestamp: new Date().toISOString(),
  });

  return; // Do not emit a new game_state after completion unless you want a final snapshot
}

// Otherwise continue to emit the usual game_state update
```

On the frontend:

```typescript
// In GameContext.tsx (pseudo-code)
const [victoryState, setVictoryState] = useState<GameResult | null>(null);

socket.on('game_over', (payload) => {
  setGameState((prev) => ({ ...prev, gameStatus: 'completed', winner: payload.data.winner }));
  setVictoryState(payload.data);
});
```

**Files to Modify:**

- `src/server/websocket/server.ts` – wire `gameResult` from `GameEngine.makeMove()` / `maybePerformAITurn()` into DB updates and a `game_over` event.
- `src/client/contexts/GameContext.tsx` – add `victoryState` handling on `game_over`.
- `src/client/pages/GamePage.tsx` – render a minimal victory UI (modal/banner) driven by `victoryState`.
- `src/shared/types/game.ts` – ensure `GameResult` and any victory-reason enums are up to date and shared with the client.

### Gap 3: Enhanced HUD ✨

**Current State:**

- GamePage shows minimal status (phase, player, game ID)
- Player info buried in sidebar
- No visual emphasis on current turn

**What's Needed:**

```typescript
// Component hierarchy:
// <GameHUD>
//   <CurrentPlayerBanner player={currentPlayer} />
//   <PhaseIndicator phase={currentPhase} />
//   <PlayerStats players={players} currentPlayerNumber={currentPlayer} />
//   <MoveHistory moves={gameState.moveHistory} />
// </GameHUD>
```

**Files to Create:**

- `src/client/components/GameHUD.tsx`
- `src/client/components/CurrentPlayerBanner.tsx`
- `src/client/components/PlayerStats.tsx`

### Gap 4: Lobby Game List 📋

**Current State:**

- Backend route `/game/lobby/available` exists
- LobbyPage only has create form, no game list
- No way to view/join existing waiting games

**What's Needed:**

```typescript
// Add to LobbyPage:
// 1. Fetch available games on mount
// 2. Display game cards with: board type, player count, creator
// 3. Join button calls `/game/:gameId/join` endpoint
// 4. Auto-refresh list periodically
```

**Files to Modify:**

- `src/client/pages/LobbyPage.tsx` - Add game list section
- `src/client/services/api.ts` - Add `getAvailableGames()` method

### Gap 5: Ring Placement Phase UI 🔵

**Current State:**

- GameEngine handles ring placement validation
- GamePage supports move submission
- No visual affordance for where rings can be placed

**What's Needed:**

```typescript
// During placement phase:
// 1. Highlight all valid placement positions (edge positions typically)
// 2. Show "Place Ring {{ringNumber}}/{{totalRings}}" prompt
// 3. Disable move selection until placement complete
```

**Files to Modify:**

- `src/client/pages/GamePage.tsx` - Add placement mode handling
- `src/client/utils/boardMovementGrid.ts` - Add placement position calculation

---

## Detailed Implementation Roadmap

### Phase 1: Core Playability (1-2 days) 🎮

**Goal: Enable completing a full 2-player human vs AI game from lobby to victory**

#### Step 1.1: Fix Game Auto-Start

- [ ] In `WebSocketServer.getOrCreateGameEngine()`, after creating players array:
  ```typescript
  // If all players ready (humans connected OR AI), start game
  const allReady = players.every((p) => p.isReady);
  if (allReady && game.status === 'waiting') {
    await prisma.game.update({
      where: { id: gameId },
      data: { status: 'ACTIVE', startedAt: new Date() },
    });
  }
  ```

#### Step 1.2: Wire Victory Signalling End-to-End

Engine-side victory detection already exists via `RuleEngine.checkGameEnd` and `GameEngine.endGame`, but the WebSocket and UI layers do not yet surface it cleanly. This step aligns the behaviour across backend, DB, and frontend.

- [ ] In `GameEngine.makeMove`, ensure that when `checkGameEnd` reports a game over, `makeMove` returns a populated `gameResult` alongside `success: true`. (This is largely in place today; adjust types/fields only as needed.)
- [ ] In `WebSocketServer.handlePlayerMove` and `maybePerformAITurn`:
  - [ ] Check `result.gameResult` and, when present:
    - [ ] Persist `status: COMPLETED`, `winnerId`, and `endedAt` on the corresponding `game` record via Prisma.
    - [ ] Emit a `game_over` event with a structured payload to all sockets in the room.
    - [ ] Avoid emitting a follow-up `game_state` after the game has completed unless you explicitly want a final snapshot.
- [ ] In `GameContext`, listen for `game_over` and store a `victoryState` that includes winner, reason, and finalScore.
- [ ] In `GamePage`, render a minimal, clear victory UI (modal or banner) driven by `victoryState`, providing at least a "Return to Lobby" action and the ability to inspect the final board.

#### Step 1.3: Victory Modal UI

- [ ] Create `src/client/components/VictoryModal.tsx`:

  ```typescript
  interface VictoryModalProps {
    isOpen: boolean;
    winner: Player;
    reason: string;
    onClose: () => void;
  }

  export function VictoryModal({ isOpen, winner, reason, onClose }: VictoryModalProps) {
    // Full-screen overlay with confetti animation
    // Display winner name, avatar, reason
    // Buttons: View Game / Return to Lobby / Rematch
  }
  ```

- [ ] Add victory listener to GameContext:
  ```typescript
  socket.on('game_over', (data) => {
    setVictoryState(data.data);
  });
  ```

#### Step 1.4: Basic HUD

- [ ] Create `src/client/components/GameHUD.tsx` with:
  - Current player banner (large, colored)
  - Phase indicator (icon + text)
  - Ring count per player
  - Move history (last 5 moves)

- [ ] Integrate into GamePage above BoardView

### Phase 2: Enhanced UX (1 day) ✨

**Goal: Make the game feel polished and intuitive**

#### Step 2.1: Lobby Game List

- [ ] Add game list to LobbyPage above create form
- [ ] Show: Board type icon, player count (2/4), creator name, time created
- [ ] Join button with confirmation dialog
- [ ] Auto-refresh every 10 seconds

#### Step 2.2: Ring Placement UX

- [ ] When `currentPhase === 'placement'`:
  - Highlight all edge positions (valid placement spots)
  - Show prompt: "Place ring {{ringNumber}}/{{totalRings}}"
  - Disable "from" selection (only "to" matters in placement)

#### Step 2.3: Move Validation Feedback

- [ ] When invalid move attempted, show toast notification with reason
- [ ] Add visual shake animation on invalid cell click
- [ ] Display valid move count in HUD ("{{count}} moves available")

#### Step 2.4: Loading States

- [ ] Show skeleton UI while loading game state
- [ ] Display "Waiting for opponent" during AI thinking
- [ ] Add connection status indicator (online/offline/reconnecting)

### Phase 3: Robustness (1 day) 🛡️

**Goal: Handle edge cases and failure modes gracefully**

#### Step 3.1: AI Service Fallback

- [ ] Modify `AIEngine.getAIMove()`:

  ```typescript
  try {
    return await this.getServiceBackedMove(playerNumber, gameState);
  } catch (error) {
    logger.warn('AI service unavailable, falling back to local heuristic', { error });
    return this.getLocalHeuristicMove(playerNumber, gameState);
  }
  ```

- [ ] Add health check: `GET /api/ai-service/health`
- [ ] Display AI mode in player stats (🌐 service / 💻 local)

#### Step 3.2: Reconnection Handling

- [ ] In GameContext, detect socket disconnect
- [ ] Show reconnection banner
- [ ] On reconnect, re-emit `join_game` and sync state

#### Step 3.3: Error Boundaries

- [ ] Wrap GamePage in React error boundary
- [ ] On error, show recovery options: reload / return to lobby
- [ ] Log errors to backend for debugging

#### Step 3.4: Choice Timeout Handling

- [ ] Already implemented in WebSocketInteractionHandler ✅
- [ ] Add visual countdown in ChoiceDialog (already exists ✅)
- [ ] Show notification when choice times out with default selected

### Phase 4: Testing & Refinement (1 day) 🧪

**Goal: Validate complete game flow with various scenarios**

In parallel with the backend-driven game lifecycle, `TODO.md` now tracks a dedicated **Phase 3S – Sandbox Stage 2** that will make `http://localhost:3000/sandbox` capable of running a complete 2–4 player game entirely in the browser using a client-local engine harness, with human and AI players. The sandbox will:

- Reuse the shared `GameState` and PlayerChoice models so that sandbox logic stays aligned with the backend `GameEngine`/`RuleEngine`.
- Provide full turn/phase progression (ring placement → movement → capture → line processing → territory processing → next player) with rules enforcement.
- Accept human input for all turn stages via the existing `BoardView` + `ChoiceDialog` UI.
- Drive AI turns using a simple random-valid-choice handler that selects uniformly among legal moves and PlayerChoice options.
- Detect victory conditions locally and present a clear end-of-game summary in the `/sandbox` UI.

The sandbox Stage 2 work is intentionally scoped and tracked in `TODO.md` under **PHASE 3S: Sandbox Stage 2 – Fully Local Playable Game**, and should be treated as a complementary playability surface that:

- Shares the same rules semantics as backend `GameEngine`.
- Is ideal for experimentation, teaching, and rapid rule validation.
- Does **not** replace the backend-driven ranked/networked game flow, which remains the authoritative path for real games.

**Rules/FAQ scenario matrix:** As part of Phase 4 and Phase 2 in `TODO.md`, a systematic **rules/FAQ scenario test matrix** will be built under `tests/`, keyed to sections of `ringrift_complete_rules.md` and FAQ Q1–Q24. The initial focus will be on:

- Chain capture patterns (including 180° reversal and cyclic examples).
- Territory disconnection scenarios for 8×8, 19×19, and hex boards.
- Graduated line rewards and multi-line processing order.
- Forced elimination and stalemate/blocked positions.

This matrix will be kept in sync between backend-driven and sandbox-driven flows to ensure consistent rules behaviour across both surfaces.

#### Test Scenarios:

1. **Human vs AI (2-player)**
   - [ ] Create game from lobby
   - [ ] Place all rings
   - [ ] Move and capture
   - [ ] Form lines and make choices
   - [ ] Verify AI responds
   - [ ] Complete game to victory

2. **Human vs Human (2-player)**
   - [ ] Create private game
   - [ ] Join from second browser/account
   - [ ] Play full game
   - [ ] Test chat messages

3. **4-player Mixed**
   - [ ] 2 humans + 2 AI
   - [ ] Verify turn order
   - [ ] Test territory disconnection with 4 players

4. **Edge Cases**
   - [ ] Disconnect/reconnect mid-game
   - [ ] Multiple tabs same user
   - [ ] AI service down (fallback triggers)
   - [ ] Choice timeout
   - [ ] Invalid move submission

5. **Choice Types**
   - [ ] Line reward choice (graduated rewards)
   - [ ] Ring elimination choice
   - [ ] Region order choice (territory disconnection)
   - [ ] Capture direction choice

---

## Priority Files to Modify

### Critical (Phase 1):

1. `src/server/game/GameEngine.ts` - Add `checkVictoryCondition()`
2. `src/server/websocket/server.ts` - Add victory detection + auto-start
3. `src/shared/types/game.ts` - Add `VictoryCondition` type
4. `src/client/components/VictoryModal.tsx` - NEW FILE
5. `src/client/components/GameHUD.tsx` - NEW FILE
6. `src/client/contexts/GameContext.tsx` - Add victory state
7. `src/client/pages/GamePage.tsx` - Integrate HUD + victory modal

### Important (Phase 2):

8. `src/client/pages/LobbyPage.tsx` - Add game list
9. `src/client/services/api.ts` - Add `getAvailableGames()`
10. `src/client/utils/boardPlacementHelpers.ts` - NEW FILE

### Nice-to-Have (Phase 3):

11. `src/server/game/ai/AIEngine.ts` - Enhance fallback logic
12. `src/server/routes/health.ts` - NEW FILE (AI service health)
13. `src/client/components/ErrorBoundary.tsx` - NEW FILE

---

## Quick Wins (Can implement now)

### 1. Auto-Start AI Games (15 min)

```typescript
// In WebSocketServer.getOrCreateGameEngine(), after player creation:
if (players.length >= 2 && game.status !== 'ACTIVE') {
  await prisma.game.update({
    where: { id: gameId },
    data: { status: 'ACTIVE', startedAt: new Date() },
  });
}
```

### 2. Victory Check Stub (20 min)

```typescript
// In GameEngine.ts:
public checkVictoryCondition(): { isGameOver: boolean; winner?: number } {
  const playersWithRings = this.gameState.players.filter(p =>
    p.ringsInHand > 0 || p.eliminatedRings < BOARD_CONFIGS[this.gameState.boardType].ringsPerPlayer
  );

  if (playersWithRings.length === 1) {
    return { isGameOver: true, winner: playersWithRings[0].playerNumber };
  }

  return { isGameOver: false };
}
```

### 3. Victory Event Emission (10 min)

```typescript
// In WebSocketServer.handlePlayerMove() and maybePerformAITurn():
const victoryCheck = gameEngine.checkVictoryCondition();
if (victoryCheck.isGameOver) {
  this.io.to(gameId).emit('game_over', {
    winner: victoryCheck.winner,
    timestamp: new Date().toISOString(),
  });
}
```

### 4. Simple Victory Toast (15 min)

```typescript
// In GameContext.tsx:
socket.on('game_over', (data) => {
  const winner = gameState?.players.find((p) => p.playerNumber === data.winner);
  alert(`Game Over! ${winner?.username || 'Player ' + data.winner} wins!`);
});
```

**Total: ~60 minutes to basic end-to-end playability**

---

## Testing Strategy

### Manual Testing Checklist:

- [ ] Create 2-player game human vs AI from lobby
- [ ] Game auto-starts immediately
- [ ] Place all rings (both players)
- [ ] Make movement move
- [ ] Verify AI responds automatically
- [ ] Form a line, get choice dialog
- [ ] Select choice option
- [ ] Continue until victory condition
- [ ] See victory notification
- [ ] Return to lobby

### Automated Testing:

- Current 103 tests should continue passing ✅
- Add integration test: `tests/e2e/fullGame.test.ts`
- Test each victory condition scenario
- Test AI turn sequencing

---

## Architecture Strengths (Leverage these!)

1. **Unified Choice System** - PlayerInteractionManager handles all strategic decisions consistently
2. **AI Abstraction** - globalAIEngine seamlessly switches between local/service-backed
3. **Type Safety** - Discriminated unions on PlayerChoiceResponse prevent invalid states
4. **WebSocket Persistence** - Game state automatically synced across all clients
5. **Move Replay** - Historical moves reconstructed from database on reconnect

**Don't reinvent these - build on them!**

---

## Recommended Next Actions

### Immediate (Today):

1. ✅ Review this plan
2. Implement quick wins (1 hour) → immediate playability
3. Create VictoryModal component
4. Add victory check to GameEngine

### This Week:

1. Complete Phase 1 (Core Playability)
2. Test end-to-end human vs AI game
3. Begin Phase 2 (Enhanced UX)

### Next Week:

1. Complete Phases 2-3
2. Comprehensive testing
3. Bug fixes and polish
4. Deploy to staging environment

---

## Success Metrics

**Definition of "Fully Playable":**

- ✅ User creates game from lobby with AI opponent
- ✅ Game starts automatically
- ✅ User places all rings
- ✅ User makes movement/capture moves
- ✅ AI responds within 5 seconds
- ✅ Strategic choices appear and accept responses
- ✅ Game detects victory condition
- ✅ Victory screen displays
- ✅ User returns to lobby for new game
- ✅ No crashes or undefined states

**Target: All metrics passing within 3-4 days of development**

---

## Conclusion

RingRift's architecture is **excellent**. The core systems are sound:

- Game engine logic is solid (19 test suites, 103 tests passing)
- WebSocket lifecycle is complete
- AI integration works end-to-end
- Choice system is elegant and extensible

**The delta to full playability is surprisingly small** - mostly victory detection, UI polish, and edge case handling. The hardest problems (game rules, state management, choice architecture) are already solved.

**Recommended approach: Quick wins first** → Get basic victory detection working → Polish UX → Comprehensive testing. This delivers immediate value while building toward the complete vision.

---

## New Task Context Summary

Use the following as a concise context block when spinning up a new task or agent for RingRift:

- **Project:** RingRift – a strategic abstract board game with complex movement, capture, line, and territory rules, implemented as a full-stack TypeScript/Node/React app with a Python FastAPI AI service and a Rust reference engine.
- **Core engine status:** BoardManager, RuleEngine, and GameEngine together implement ~75–80% of the rules across 8×8, 19×19, and hex boards. Movement/markers, overtaking captures with mandatory chains, line formation with graduated rewards, territory disconnection with self-elimination, forced elimination, and phase flow are all implemented and exercised via focused Jest tests. The Rust engine provides additional reference scenarios but is not used at runtime.
- **Interaction/choice system:** A unified PlayerChoice architecture exists (shared types + PlayerInteractionManager + WebSocketInteractionHandler + AIInteractionHandler + DelegatingInteractionHandler). GameEngine uses this to request choices for line order/rewards, ring/cap elimination, region order, and capture direction, from both humans (via WebSockets + React ChoiceDialog) and AI (via local heuristics or the Python AI service).
- **AI integration:** The Python AI service is wired into the backend through AIServiceClient and AIEngine/globalAIEngine. Backend games can already use service-backed AI for move selection and some PlayerChoices (`line_reward_option`, `ring_elimination`, `region_order`), with robust fallbacks to local heuristics and tests covering success/failure paths.
- **Backend/WebSocket lifecycle:** Games are created and persisted via Express + Prisma routes; WebSocketServer hydrates GameEngine instances from the DB (including AI profiles), replays historical moves, and drives both human and AI turns. GameContext on the client maintains the live game state, valid moves (where available), and pending PlayerChoices.
- **Frontend status:** React + Vite + Tailwind client with:
  - BoardView rendering 8×8, 19×19, and hex boards, including improved contrast, larger squares, and a shared `computeBoardMovementGrid(board)` helper for faint movement-grid overlays.
  - GamePage with two modes: backend game view (`/game/:gameId`) wired to WebSocket/PlayerChoice/AI, and a local sandbox entry that can create backend games or fall back to a local-only board.
  - ChoiceDialog and HUD-like side panels that expose selection info, player/AI profiles, status, and a basic event log.
- **Docs & planning:**
  - [`../docs/archive/historical/CURRENT_STATE_ASSESSMENT.md`](../docs/archive/historical/CURRENT_STATE_ASSESSMENT.md) – factual description of what works now.
  - `KNOWN_ISSUES.md` – P0/P1 issues, now focused on test coverage, UX, and AI depth rather than missing mechanics.
  - `TODO.md` – phased development tasks, including PlayerChoice, chain captures, AI, WebSocket, sandbox, and HUD items.
  - `STRATEGIC_ROADMAP.md` – phased roadmap (testing/core logic → minimal UI → AI → polish → multiplayer).
  - `deprecated/RINGRIFT_IMPROVEMENT_PLAN.md` – historical improvement-focused plan around rules fidelity, playability, and AI.
  - `ARCHITECTURE_ASSESSMENT.md` & `deprecated/CODEBASE_EVALUATION.md` – architecture and code-quality views structured along four axes: rules/state, AI boundary, WebSocket/game loop, and testing/quality gates.

When creating a new task, anchor it to one or more of these axes (rules/state, AI boundary, WebSocket/game loop, testing/quality) and clearly state whether the goal is **rules fidelity**, **playability/UX**, **AI strength**, or **operational robustness**. This keeps changes aligned with the long-term architecture and minimises technical debt.
