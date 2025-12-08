# Hook Integration Plan - SandboxGameHost Refactoring

> **Created:** December 2025
> **Status:** Planning
> **Goal:** Complete integration of deferred hooks to reduce SandboxGameHost from ~2893 to ~2600 lines

## Executive Summary

Two hooks remain to be integrated into SandboxGameHost:

1. **useSandboxEvaluation** - Lower complexity, ~65 line reduction
2. **useSandboxScenarios** - Higher complexity, ~130 line reduction

Total expected reduction: **~195 lines** (from 2893 to ~2700)

---

## Phase 1: useSandboxEvaluation Integration (Recommended First)

### Why Start Here

- Straightforward input/output model
- No complex state reset coordination
- Clear dependency boundaries
- Lower risk of breaking existing functionality

### Current State in SandboxGameHost

**State variables (lines 360-364):**

```typescript
const [sandboxEvaluationHistory, setSandboxEvaluationHistory] = useState<...>([]);
const [sandboxEvaluationError, setSandboxEvaluationError] = useState<string | null>(null);
const [isSandboxAnalysisRunning, setIsSandboxAnalysisRunning] = useState(false);
```

**Request handler (lines 495-548):** ~55 lines
**Auto-evaluation effect (lines 1770-1799):** ~30 lines

### Integration Steps

#### Step 1.1: Add hook import

```typescript
import { useSandboxEvaluation } from '../hooks/useSandboxEvaluation';
```

#### Step 1.2: Replace state declarations with hook call

```typescript
// BEFORE (lines 360-364):
const [sandboxEvaluationHistory, setSandboxEvaluationHistory] = useState<...>([]);
const [sandboxEvaluationError, setSandboxEvaluationError] = useState<string | null>(null);
const [isSandboxAnalysisRunning, setIsSandboxAnalysisRunning] = useState(false);

// AFTER:
const {
  evaluationHistory: sandboxEvaluationHistory,
  evaluationError: sandboxEvaluationError,
  isEvaluating: isSandboxAnalysisRunning,
  requestEvaluation: requestSandboxEvaluation,
  clearHistory: clearEvaluationHistory,
} = useSandboxEvaluation({
  engine: sandboxEngine,
  gameState: sandboxGameState,
  developerToolsEnabled,
  isInReplayMode,
  isViewingHistory,
  evaluationEndpoint: '/api/games/sandbox/evaluate',
});
```

#### Step 1.3: Remove redundant code

- Delete `requestSandboxEvaluation` callback (lines 495-548)
- Delete auto-evaluation useEffect (lines 1770-1799)
- Keep `lastEvaluatedMoveRef` reset in engine destruction effect (already done)

#### Step 1.4: Verify all usages work

Search for and verify:

- `sandboxEvaluationHistory` - Used in EvaluationPanel
- `sandboxEvaluationError` - Used in error display
- `isSandboxAnalysisRunning` - Used in loading states
- `requestSandboxEvaluation` - Used in manual evaluation trigger

### Hook Modifications Required

The `useSandboxEvaluation` hook needs these updates:

```typescript
// In useSandboxEvaluation.ts, update interface:
export interface SandboxEvaluationOptions {
  engine: ClientSandboxEngine | null;
  gameState: GameState | null;
  developerToolsEnabled: boolean;
  isInReplayMode?: boolean;
  isViewingHistory?: boolean;
  evaluationEndpoint?: string; // Default: '/api/games/sandbox/evaluate'
  stateVersion?: number; // To trigger re-evaluation
}
```

### Estimated Impact

- Lines removed: ~85
- Lines added: ~20 (hook call + imports)
- Net reduction: ~65 lines

---

## Phase 2: useSandboxScenarios Integration (Higher Complexity)

### Why More Complex

- Requires 4 wrapper callbacks to be created
- Must coordinate 11+ state variable resets atomically
- Bidirectional state synchronization needed
- Self-play move replay has async timing concerns

### Current State in SandboxGameHost

**State variables (lines 368-410):**

```typescript
const [isInReplayMode, setIsInReplayMode] = useState(false);
const [replayState, setReplayState] = useState<GameState | null>(null);
const [replayAnimation, setReplayAnimation] = useState<...>(null);
const [isViewingHistory, setIsViewingHistory] = useState(false);
const [historyViewIndex, setHistoryViewIndex] = useState(0);
const [hasHistorySnapshots, setHasHistorySnapshots] = useState(true);
const [showScenarioPicker, setShowScenarioPicker] = useState(false);
const [showSelfPlayBrowser, setShowSelfPlayBrowser] = useState(false);
const [lastLoadedScenario, setLastLoadedScenario] = useState<...>(null);
```

**Handlers:**

- `handleLoadScenario` (lines 764-912): ~150 lines
- `handleForkFromReplay` (lines 919-977): ~60 lines
- `handleResetScenario` (lines 979-1025): ~45 lines

### Integration Steps

#### Step 2.1: Create wrapper callbacks

```typescript
// Callback 1: Initialize sandbox with scenario
const initSandboxWithScenario = useCallback(
  (scenario: ScenarioData): ClientSandboxEngine | null => {
    // Determine player types from scenario or use defaults
    const playerTypes =
      scenario.source === 'selfplay'
        ? (['ai', 'ai'] as LocalPlayerType[])
        : (config.playerTypes.slice(0, config.numPlayers) as LocalPlayerType[]);

    // Create interaction handler
    const handler = createSandboxInteractionHandler(playerTypes);

    // Initialize engine
    const engine = initLocalSandboxEngine({
      boardType: scenario.gameState.boardType,
      numPlayers: scenario.gameState.players.length,
      playerTypes,
      interactionHandler: handler,
    });

    if (!engine) return null;

    // Load the scenario state
    const serialized = serializeGameState(scenario.gameState);
    engine.initFromSerializedState(serialized, { normalizeState: true });

    return engine;
  },
  [config, createSandboxInteractionHandler, initLocalSandboxEngine]
);

// Callback 2: Reset all UI state
const resetGameUIState = useCallback(() => {
  setSelected(undefined);
  setValidTargets([]);
  setSandboxPendingChoice(null);
  setIsSandboxVictoryModalDismissed(false);
  setBackendSandboxError(null);
  setSandboxCaptureChoice(null);
  setSandboxCaptureTargets([]);
  setSandboxStallWarning(null);
  setSandboxLastProgressAt(null);
}, []);

// Callback 3: Handle scenario load complete
const handleScenarioLoadComplete = useCallback(
  (scenario: LoadedScenario) => {
    void logSandboxScenarioLoaded(scenario);
    resetGameUIState();
  },
  [resetGameUIState]
);

// Callback 4: Bump state version
const handleStateVersionChange = useCallback(() => {
  setSandboxStateVersion((v) => v + 1);
}, []);
```

#### Step 2.2: Integrate the hook

```typescript
const {
  // Scenario state
  lastLoadedScenario,
  showScenarioPicker,
  setShowScenarioPicker,
  showSelfPlayBrowser,
  setShowSelfPlayBrowser,

  // Replay state
  isInReplayMode,
  setIsInReplayMode,
  replayState,
  replayAnimation,
  setReplayAnimation,

  // History playback
  isViewingHistory,
  setIsViewingHistory,
  historyViewIndex,
  setHistoryViewIndex,
  hasHistorySnapshots,
  setHasHistorySnapshots,

  // Handlers
  handleLoadScenario,
  handleForkFromReplay,
  handleResetScenario,
  clearScenarioContext,
} = useSandboxScenarios({
  initSandboxWithScenario,
  onScenarioLoaded: handleScenarioLoadComplete,
  onStateVersionChange: handleStateVersionChange,
});
```

#### Step 2.3: Remove redundant code

- Delete old state declarations (9 useState calls)
- Delete `handleLoadScenario` implementation (lines 764-912)
- Delete `handleForkFromReplay` implementation (lines 919-977)
- Delete `handleResetScenario` implementation (lines 979-1025)

#### Step 2.4: Verify all usages work

Search for and verify:

- Dialog visibility: `showScenarioPicker`, `showSelfPlayBrowser`
- Replay state: `isInReplayMode`, `replayState`, `replayAnimation`
- History state: `isViewingHistory`, `historyViewIndex`, `hasHistorySnapshots`
- Scenario context: `lastLoadedScenario`

### Hook Modifications Required

The `useSandboxScenarios` hook needs these updates:

```typescript
// Add onUIStateReset callback option
export interface SandboxScenariosOptions {
  initSandboxWithScenario: (scenario: ScenarioData) => ClientSandboxEngine | null;
  onScenarioLoaded?: (scenario: LoadedScenario) => void;
  onStateVersionChange?: () => void;
  onUIStateReset?: () => void; // NEW: Called to reset parent UI state
}

// Update handleLoadScenario to call onUIStateReset
const handleLoadScenario = useCallback(
  async (scenario: ScenarioData) => {
    // ... existing logic ...

    // After successful load, reset UI state
    onUIStateReset?.();

    // ... rest of logic ...
  },
  [initSandboxWithScenario, onScenarioLoaded, onStateVersionChange, onUIStateReset]
);
```

### Estimated Impact

- Lines removed: ~255 (state + handlers)
- Lines added: ~80 (wrapper callbacks)
- Net reduction: ~175 lines

---

## Implementation Order

```
┌─────────────────────────────────────────┐
│  Phase 1: useSandboxEvaluation          │
│  Estimated time: 2-3 hours              │
│  Risk: Low                              │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Validation & Testing                   │
│  - Manual evaluation trigger            │
│  - Auto-evaluation on move              │
│  - Error handling                       │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Phase 2: useSandboxScenarios           │
│  Estimated time: 4-5 hours              │
│  Risk: Medium-High                      │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Validation & Testing                   │
│  - Scenario picker loading              │
│  - Self-play browser loading            │
│  - Replay mode functionality            │
│  - Fork from replay                     │
│  - Reset scenario                       │
│  - History playback                     │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Final Integration Testing              │
│  - Full workflow tests                  │
│  - Edge cases                           │
│  - Performance validation               │
└─────────────────────────────────────────┘
```

---

## Risk Mitigation

### Phase 1 Risks (Low)

| Risk                  | Mitigation                      |
| --------------------- | ------------------------------- |
| API endpoint mismatch | Pass endpoint as option         |
| State version sync    | Include stateVersion in options |

### Phase 2 Risks (Medium-High)

| Risk                        | Mitigation                                        |
| --------------------------- | ------------------------------------------------- |
| Atomic state reset          | Create dedicated resetGameUIState callback        |
| Interaction handler closure | Recreate handler with correct player types        |
| Async move replay timing    | Keep existing sequential replay logic             |
| Bidirectional state sync    | Document which states are owned by hook vs parent |

### Rollback Strategy

1. Keep old code commented initially
2. Feature flag the new implementation if needed
3. Unit test each phase independently
4. Integration test full workflows before removing old code

---

## Success Criteria

- [x] TypeScript compiles without errors
- [x] All hook tests pass (443 total including 69 new sandbox hook tests)
- [ ] Scenario loading works from picker (needs manual testing)
- [ ] Self-play game loading works from browser (needs manual testing)
- [ ] Replay mode displays correctly (needs manual testing)
- [ ] Fork from replay creates playable game (needs manual testing)
- [ ] Reset scenario restores original state (needs manual testing)
- [ ] History playback navigates correctly (needs manual testing)
- [x] Auto-evaluation triggers on developer tools (hook integration complete)
- [x] Manual evaluation works (hook integration complete)
- [x] SandboxGameHost reduced to ~2708 lines (from ~2893, ~185 line reduction)
- [x] New hook tests added (Phase 1 of REMAINING_COMPLEX_TASKS_PLAN complete):
  - useSandboxEvaluation: 19 tests
  - useSandboxPersistence: 19 tests
  - useSandboxScenarios: 31 tests

---

## Files Affected

### Primary

- `src/client/pages/SandboxGameHost.tsx` - Main refactoring target
- `src/client/hooks/useSandboxEvaluation.ts` - Minor updates
- `src/client/hooks/useSandboxScenarios.ts` - Add onUIStateReset callback

### Secondary

- `src/client/hooks/index.ts` - Verify exports
- `TODO.md` - Update integration status

---

**Document Maintainer:** Claude Code
**Last Updated:** December 2025
