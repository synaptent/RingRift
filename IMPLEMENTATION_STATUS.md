# Implementation Status

**Last Updated:** November 20, 2025

This file is a **short summary** of the current, code‑verified status. For full detail, see:

- `CURRENT_STATE_ASSESSMENT.md` – authoritative breakdown by component
- `KNOWN_ISSUES.md` – P0/P1 issues and gaps
- `TODO.md` – phase/task tracking with progress
- `STRATEGIC_ROADMAP.md` – phased roadmap toward MVP

---

## 🔎 High-Level Summary

**Overall Status:** Strong foundation, **not yet production-ready**

- Core rules and engine are largely implemented and exercised by focused tests.
- Backend WebSocket + AI loop supports real backend-driven games with AI turns and PlayerChoices.
- React client has a functional lobby, backend GamePage, HUD, and victory modal, plus a rich local sandbox harness.
- Testing is substantial for core mechanics and interaction flows but **not yet systematically scenario-driven** across all rules/FAQ examples.

This is best described as an **engine/AI and playtest-focused beta**, not a polished public release.

---

## ✅ What’s Verified Today

### Core Game Logic & Engines

- **BoardManager**
  - Supports 8×8, 19×19, and hex boards
  - Position generation, adjacency, distance
  - Line detection and territory region detection (square + hex)
- **RuleEngine**
  - Movement validation (distance ≥ stack height, blocking, landing rules)
  - Capture validation, including overtaking and mandatory chains
  - Line formation helpers and territory helpers (full logic shared with GameEngine)
- **GameEngine**
  - Turn/phase loop: `ring_placement → movement → capture → chain_capture → line_processing → territory_processing → next player` on the backend; sandbox migration to this explicit `chain_capture` phase is in progress.
  - Marker system, forced elimination, chain captures, graduated line rewards, territory disconnection.
  - Victory checks for ring-elimination and territory-control.
  - PlayerChoice integration for line order/rewards, ring/cap elimination, region order, and (in sandbox/legacy flows) capture direction via `PlayerInteractionManager`; on the backend, capture direction is now expressed as `continue_capture_segment` moves in `chain_capture` (see [`TODO.md`](TODO.md:73)).

### Backend Infrastructure

- **HTTP API**
  - Auth endpoints (`/api/auth`): register/login/refresh/logout, with placeholders for email/password flows
  - Game endpoints (`/api/games`): create/list/join/leave, lobby listing for available games
  - User endpoints (`/api/users`): profile and stats basics
- **WebSocket Server** (`src/server/websocket/server.ts`)
  - Authenticated Socket.IO server
  - `join_game` / `leave_game` / `player_move` / `chat_message` / `player_choice_response`
  - Auto-start logic: when all players (human + AI) are ready and DB status is `WAITING`, mark game `ACTIVE`
  - AI turns via `maybePerformAITurn` using `globalAIEngine` and `AIServiceClient`
  - **Victory signalling**: when GameEngine returns a `gameResult`, the server:
    - Updates the Prisma `game` row to `COMPLETED` with `winnerId` (if human) and `endedAt`
    - Emits a `game_over` event including `{ gameId, gameState, gameResult }`

### AI Integration

- Python FastAPI AI service under `ai-service/` with Random/Heuristic AIs
- TypeScript boundary:
  - `AIServiceClient` and `AIEngine`/`globalAIEngine`
  - Service-backed move selection and several PlayerChoices (`line_reward_option`, `ring_elimination`, `region_order`) with tested fallbacks to local heuristics
- **FullGameFlow integration test** (`tests/integration/FullGameFlow.test.ts`) exists and now acts as a regression harness for the unified chain-capture / Move model and the S‑invariant:
  - Runs a 2‑AI game fully in `GameEngine` with the AI service mocked as failing and local AI fallback selecting moves from `GameEngine.getValidMoves` in all interactive phases (including `chain_capture`).
  - Enforces that the global progress metric `S = markers + collapsed + eliminated` never decreases and fails fast on long S‑plateaus (potential stalls).
  - Currently **passing**; regressions here are tracked as part of P0.4 in [`TODO.md`](TODO.md:73) since this test exercises end-to-end termination and chain-capture semantics.

### Frontend Client

- **LobbyPage**
  - Lists available games via `/games/lobby/available`
  - Allows creating games with board type, maxPlayers, rated/private flags, time control, and AI configuration (`aiOpponents` → persisted `aiProfile`/`aiOpponents` in game state)
  - Join button calls the join endpoint and navigates to `/game/:gameId`
- **GamePage (backend mode)**
  - Connects via GameContext to WebSocket, receives `game_state` and `game_over`
  - Renders `BoardView` for 8×8, 19×19, and hex boards
  - Uses backend-provided `validMoves` for click‑to‑move (select source, then legal destination)
  - Uses `ChoiceDialog` to render server-driven PlayerChoices and send `player_choice_response`
  - Uses `VictoryModal` to show `gameResult` on `game_over` and allows returning to the lobby
  - Shows basic diagnostics (phase/current player/choice events)
- **GameHUD**
  - Displays current player, phase, and per-player ring/territory statistics based on `GameState`
- **Local Sandbox (`/sandbox`)**
  - `ClientSandboxEngine` + sandbox modules implement a fully local rules‑complete engine with:
    - Movement, chain captures, line processing, territory disconnection, and victory
    - Simple random-choice AI for all PlayerChoices
  - Shares `BoardView`, `ChoiceDialog`, `VictoryModal`, and HUD concepts with backend games

### Testing

- Jest + ts‑jest configured with a suite of **unit and integration tests** across:
  - BoardManager, RuleEngine, GameEngine
  - ClientSandboxEngine and sandbox helpers
  - PlayerInteractionManager, WebSocketInteractionHandler, AIInteractionHandler
  - AIEngine/AIServiceClient (including failure/fallback behaviour)
  - WebSocketServer AI-turn and choice integration
- Trace and parity harnesses compare backend vs sandbox behaviour via `GameTrace` utilities.
- Backend FAQ-style chain-capture scenario suites (triangle loops, 180° reversals, zig‑zag patterns, and hex cyclic captures) now pass under the unified `chain_capture` + `continue_capture_segment` model. However, several backend↔sandbox parity and AI-trace harnesses remain **red** until the sandbox engine and `sandboxAI` are migrated to the same model (see the updated P0.4 section of [`TODO.md`](TODO.md:73) for the current failing-suite list and remediation plan).

Overall, **core logic is well covered by carefully chosen tests**, though not yet by a complete rules/FAQ scenario matrix.

---

## 🚧 In Progress / Known Gaps (High Level)

The following items are **not complete**, even though some docs previously claimed higher percentages:

### Multiplayer & UX

- **Spectator Mode:** Basic UI support added (spectator badge, read-only view).
- **Chat:** Basic in-game chat UI implemented (local state wired, backend integration pending).
- **Reconnection:** Improved UX with reconnection banner and status indicators.
- Matchmaking is limited to a manually refreshed lobby list; there is **no automated matching**.

### AI Depth

- Current AI opponents in live games use RandomAI and HeuristicAI via the Python AI service and local fallbacks. Prototype Minimax, MCTS, and NeuralNet-based AIs exist under `ai-service/app/ai`, but they are **not** yet wired into the primary `/ai/move` path or used by default.
- AI service contracts are stable for the existing endpoints, but deeper search and learning models are still considered experimental and are not treated as production-strength.

### Testing & Rules Coverage

- **Scenario Matrix:** Initial scenario-driven tests (`tests/scenarios/`) are now in place for complex chain captures (180° reversals, cycles) and line/territory interactions, keyed to FAQ examples.
- Backend↔sandbox semantic parity is strong for basic movement/placement, but currently **broken** in several chain‑capture and AI-trace harnesses because the unified Move model (`chain_capture` + `continue_capture_segment` on the backend) has not yet been fully mirrored in the sandbox engine and `sandboxAI`; see P0.4 in [`TODO.md`](TODO.md:73).

For details and specific P0/P1 items, see `KNOWN_ISSUES.md`.

---

## 🎯 Recommended Focus (Next Step)

For advancing toward the documented MVP (playable single‑player games with confidence in rules and stability), the immediate focus should be:

1. **Scenario-driven tests** for complex rules/FAQ examples, especially chain capture edge cases, combined line+territory turns, and hex‑board territory patterns.
2. **Sandbox + backend parity hardening**, ensuring that sandbox traces and backend replays agree across a broad set of seeds.
3. **Frontend lifecycle polish** around reconnection, spectator views, and clearer HUD/timer displays for both backend and sandbox games.

These priorities are expanded and tracked in `TODO.md` and `STRATEGIC_ROADMAP.md`.
In particular, see the "Consolidated Execution Tracks & Plan" section in
`TODO.md`, which groups the work into:

- Track 1 – Rules/FAQ Scenario Matrix & Parity Hardening (P0)
- Track 2 – Multiplayer Lifecycle & HUD/UX (P1)
- Track 3 – Sandbox as a First-Class Rules Lab (P0–P1)
- Track 4 – Incremental AI Improvements & Observability (P1–P2)
- Track 5 – Persistence, Replays, and Stats (P2)
