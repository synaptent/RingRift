# Client Architecture

> **Doc Status (2025-11-27): Active (client layering & hosts/adapters)**
>
> - Describes the client-side layering (UI, view models, hooks, GameContext, sandbox hosts/adapters).
> - Not a rules or lifecycle SSoT. Rules semantics SSoT = `RULES_CANONICAL_SPEC.md` + shared TS rules engine under `src/shared/engine/` (helpers → domain aggregates → turn orchestrator → contracts + v2 contract vectors). Lifecycle semantics SSoT = `docs/CANONICAL_ENGINE_API.md` + shared TS/WebSocket types and schemas.
> - `ClientSandboxEngine` and `SandboxOrchestratorAdapter` are **hosts/adapters** over the shared TS engine, not independent rules engines.

This document describes the client-side architecture for RingRift, focusing on the separation between game domain logic and UI presentation.

## Overview

The client follows a **layered architecture** pattern:

```
┌─────────────────────────────────────────────────────────┐
│                     UI Components                        │
│    (GameHUD, GameEventLog, BoardView, VictoryModal)     │
├─────────────────────────────────────────────────────────┤
│                      View Models                         │
│  (HUDViewModel, BoardViewModel, EventLogViewModel, etc) │
├─────────────────────────────────────────────────────────┤
│                    Custom Hooks                          │
│   (useGameState, useGameActions, useGameConnection)     │
├─────────────────────────────────────────────────────────┤
│                    Game Context                          │
│         (GameContext - central state store)             │
├─────────────────────────────────────────────────────────┤
│                  Domain Services                         │
│  (SandboxOrchestratorAdapter, ClientSandboxEngine)      │
├─────────────────────────────────────────────────────────┤
│                   Shared Types                           │
│         (GameState, Player, BoardState, etc)            │
└─────────────────────────────────────────────────────────┘
```

## Layer Responsibilities

### 1. UI Components (`src/client/components/`)

Pure presentation components that:

- Receive data via props or view models
- Handle user interactions
- Emit events/callbacks for actions
- Have no direct knowledge of GameContext or domain services

**Key Components:**

- [`GameHUD.tsx`](components/GameHUD.tsx) - Game status, phase, players display
- [`GameEventLog.tsx`](components/GameEventLog.tsx) - Move history display
- [`BoardView.tsx`](components/BoardView.tsx) - Board rendering and interaction
- [`VictoryModal.tsx`](components/VictoryModal.tsx) - End game display

### 2. View Models (`src/client/adapters/gameViewModels.ts`)

Transform functions that convert raw domain state (`GameState`) into presentation-ready view models.

**Why View Models?**

- Decouple components from direct `GameState` knowledge
- Make components easier to test in isolation
- Enable reuse in different contexts (replay viewer, analysis mode)
- Centralize presentation logic (colors, labels, formatting)

**Available Adapters:**

```typescript
// Transform GameState to HUD display data
toHUDViewModel(gameState, options): HUDViewModel

// Transform game history to event log entries
toEventLogViewModel(gameState): EventLogViewModel

// Transform board state to renderable board
toBoardViewModel(boardState, options): BoardViewModel

// Transform game result to victory display
toVictoryViewModel(gameResult): VictoryViewModel | null
```

### 3. Custom Hooks (`src/client/hooks/`)

Provide focused access to specific aspects of game state:

#### `useGameState.ts`

Read-only access to game state with optional view model transformation:

```typescript
// Raw state access
const { gameState, currentPlayer } = useGameState();

// Pre-transformed view model
const hudViewModel = useHUDViewModel({ instruction, currentUserId });
const boardViewModel = useBoardViewModel({ showValidMoves: true });
const eventLogViewModel = useEventLogViewModel();
```

#### `useGameConnection.ts`

Connection-related state and reconnection logic:

```typescript
const {
  connectionStatus, // 'connected' | 'connecting' | 'reconnecting' | 'disconnected'
  isConnected,
  reconnectAttempts,
  lastHeartbeatAt,
} = useGameConnection();
```

#### `useGameActions.ts`

Action submission methods:

```typescript
const {
  submitMove, // Submit a game move
  submitChoice, // Submit a decision (line/territory)
  sendChatMessage, // Send chat message
  validMoves, // Available moves for current player
} = useGameActions();
```

### 4. Game Context (`src/client/contexts/GameContext.tsx`)

Central state management handling:

- WebSocket connection lifecycle
- State synchronization with server
- Reconnection logic
- Choice/decision handling
- Chat messages

**Note:** Direct use of GameContext is discouraged. Prefer using the focused hooks above.

### 5. Domain Services (`src/client/sandbox/`)

Game engine and orchestration:

- [`ClientSandboxEngine.ts`](sandbox/ClientSandboxEngine.ts) - Local game state management
- [`SandboxOrchestratorAdapter.ts`](sandbox/SandboxOrchestratorAdapter.ts) - Clean domain API

---

## Component Contracts

### GameHUD

**Legacy Props (backward compatible):**

```typescript
interface GameHUDLegacyProps {
  gameState: GameState;
  currentPlayer: Player | undefined;
  instruction?: string;
  connectionStatus?: ConnectionStatus;
  isSpectator?: boolean;
  lastHeartbeatAt?: number | null;
  currentUserId?: string;
}
```

**View Model Props (recommended):**

```typescript
interface GameHUDViewModelProps {
  viewModel: HUDViewModel;
  timeControl?: TimeControl;
}
```

### GameEventLog

**Legacy Props:**

```typescript
interface GameEventLogLegacyProps {
  history?: HistoryEntry[];
  currentMoveNumber?: number;
  onReplayMove?: (moveNumber: number) => void;
  isReplaying?: boolean;
}
```

**View Model Props:**

```typescript
interface GameEventLogViewModelProps {
  viewModel: EventLogViewModel;
  onReplayMove?: (moveNumber: number) => void;
}
```

### BoardView

Currently uses raw domain types. Future migration path:

```typescript
interface BoardViewProps {
  viewModel: BoardViewModel;
  onCellClick?: (position: Position) => void;
  // ... interaction callbacks
}
```

### VictoryModal

```typescript
interface VictoryModalProps {
  result: GameResult; // or VictoryViewModel in future
  playerNumber: number;
  onClose?: () => void;
}
```

---

## Adding New Game Modes or Views

### 1. Adding a New View (e.g., Replay Viewer)

1. **Create view-specific hooks** in `src/client/hooks/`:

   ```typescript
   // useReplayState.ts
   export function useReplayState() {
     const { gameState } = useGameState();
     const [replayIndex, setReplayIndex] = useState(0);
     // ... replay-specific logic
   }
   ```

2. **Extend view model adapters** if needed:

   ```typescript
   // In gameViewModels.ts
   export function toReplayBoardViewModel(
     boardState: BoardState,
     highlightedMove?: Move
   ): BoardViewModel {
     // Add replay-specific highlighting
   }
   ```

3. **Create the page component**:
   ```typescript
   // ReplayPage.tsx
   function ReplayPage() {
     const boardVM = useReplayBoardViewModel();
     return <BoardView viewModel={boardVM} />;
   }
   ```

### 2. Adding a New Game Mode (e.g., Analysis Mode)

1. **Define mode-specific state** in context or dedicated store

2. **Create mode-specific hooks**:

   ```typescript
   export function useAnalysisMode() {
     const { gameState } = useGameState();
     const [variations, setVariations] = useState<Variation[]>([]);
     // ... analysis logic
   }
   ```

3. **Extend or compose view models**:
   ```typescript
   export interface AnalysisBoardViewModel extends BoardViewModel {
     analysisMarkers: AnalysisMarker[];
     bestMove?: Position;
   }
   ```

### 3. Adding New UI Components

1. **Use view model as input** (avoid raw GameState):

   ```typescript
   interface MyComponentProps {
     viewModel: MyViewModel; // Prefer this
     // NOT: gameState: GameState
   }
   ```

2. **Define clear prop interface** in the component file

3. **Export types** for consumer reference

4. **Add to component index** if creating a reusable component

---

## Migration Guide

### Migrating a Component to View Models

1. **Keep backward compatibility**:

   ```typescript
   type Props = LegacyProps | ViewModelProps;

   function isViewModelProps(p: Props): p is ViewModelProps {
     return 'viewModel' in p;
   }

   function MyComponent(props: Props) {
     if (isViewModelProps(props)) {
       return <ViewModelImpl {...props} />;
     }
     return <LegacyImpl {...props} />;
   }
   ```

2. **Create a view model adapter** in `gameViewModels.ts`

3. **Update consumers gradually** to use view models

4. **Deprecate legacy interface** when fully migrated

### Using New Hooks

Replace direct context usage:

```typescript
// Before
const { gameState, connectionStatus, submitMove } = useGame();

// After
const { gameState } = useGameState();
const { connectionStatus } = useGameConnection();
const { submitMove } = useGameActions();
```

---

## Design Principles

1. **Single Responsibility**: Each hook/adapter handles one concern
2. **Immutability**: View models are read-only snapshots
3. **Type Safety**: Strong typing for all interfaces
4. **Backward Compatibility**: Support legacy patterns during migration
5. **Testability**: Pure functions for adapters, hooks mockable for components

---

## File Structure

```
src/client/
├── adapters/
│   └── gameViewModels.ts      # View model transformers
├── components/
│   ├── BoardView.tsx          # Board rendering
│   ├── GameEventLog.tsx       # Move history (supports VM)
│   ├── GameHUD.tsx            # Game status (supports VM)
│   ├── VictoryModal.tsx       # End game display
│   └── ui/                    # Generic UI primitives
├── contexts/
│   └── GameContext.tsx        # Central game state
├── hooks/
│   ├── index.ts               # Exports
│   ├── useGameActions.ts      # Action submission
│   ├── useGameConnection.ts   # Connection state
│   └── useGameState.ts        # Game state access
├── pages/
│   ├── GamePage.tsx           # Main game page
│   └── ...
├── sandbox/
│   ├── ClientSandboxEngine.ts # Local game engine
│   └── SandboxOrchestratorAdapter.ts
└── ARCHITECTURE.md            # This file
```

---

## Related Documentation

- [Shared Engine Architecture](../../shared/engine/orchestration/README.md)
- [Module Responsibilities](../../docs/MODULE_RESPONSIBILITIES.md)
- [Domain Aggregate Design](../../docs/DOMAIN_AGGREGATE_DESIGN.md)
