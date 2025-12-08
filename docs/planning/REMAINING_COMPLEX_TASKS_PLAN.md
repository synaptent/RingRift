# Remaining Complex Tasks Plan

> **Created:** December 2025
> **Status:** Planning
> **Goal:** Complete remaining complex tasks from architecture improvements

## Executive Summary

After completing hook integrations (useSandboxPersistence, useSandboxEvaluation, useSandboxScenarios, useBoardOverlays), the following complex tasks remain:

| Task                              | Complexity | Priority | Estimated Effort |
| --------------------------------- | ---------- | -------- | ---------------- |
| Additional Sandbox Hook Tests     | Medium     | P1       | Medium           |
| Tier 2 Sandbox Cleanup            | High       | P2       | High             |
| Manual Testing of Scenario/Replay | Low        | P1       | Low              |
| FSM Validation Promotion          | High       | P1       | Blocked          |

---

## Phase 1: Additional Sandbox Hook Tests (Recommended Next)

### Why This Task

- New hooks lack dedicated unit tests
- Ensures refactoring didn't introduce regressions
- Enables confident future refactoring

### Hooks Needing Tests

#### 1.1 useSandboxPersistence Tests

**File:** `tests/unit/hooks/useSandboxPersistence.test.tsx`

Test cases:

- [ ] Initial state is correct (autoSaveGames default, idle status)
- [ ] Engine change captures initial game state
- [ ] Victory detection triggers auto-save
- [ ] Server save success updates status to 'saved'
- [ ] Server save failure falls back to local storage
- [ ] Local storage fallback updates pendingLocalGames
- [ ] GameSyncService subscription works
- [ ] Engine destruction resets refs

#### 1.2 useSandboxEvaluation Tests

**File:** `tests/unit/hooks/useSandboxEvaluation.test.tsx`

Test cases:

- [ ] Initial state is correct (empty history, no error, not evaluating)
- [ ] requestEvaluation makes API call with serialized state
- [ ] Successful evaluation adds to history
- [ ] Failed evaluation sets error message
- [ ] 404 error shows appropriate message
- [ ] 503 error shows appropriate message
- [ ] Auto-evaluation triggers when developerToolsEnabled
- [ ] Auto-evaluation skips in replay mode
- [ ] Auto-evaluation skips in history viewing mode
- [ ] clearHistory resets state
- [ ] Engine change clears history

#### 1.3 useSandboxScenarios Tests

**File:** `tests/unit/hooks/useSandboxScenarios.test.tsx`

Test cases:

- [ ] Initial state is correct (no scenario, pickers closed)
- [ ] handleLoadScenario calls initSandboxWithScenario
- [ ] handleLoadScenario calls onUIStateReset
- [ ] handleLoadScenario calls onScenarioLoaded
- [ ] handleLoadScenario closes pickers
- [ ] handleLoadScenario resets history state
- [ ] handleForkFromReplay creates fork scenario
- [ ] handleForkFromReplay clears scenario context
- [ ] handleResetScenario reloads original
- [ ] clearScenarioContext resets all state
- [ ] setters update state correctly

### Implementation Approach

```typescript
// Example test structure for useSandboxEvaluation
describe('useSandboxEvaluation', () => {
  const mockEngine = {
    getGameState: jest.fn(),
    getSerializedState: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
    global.fetch = jest.fn();
  });

  it('should initialize with correct default state', () => {
    const { result } = renderHook(() =>
      useSandboxEvaluation({
        engine: null,
        developerToolsEnabled: false,
      })
    );

    expect(result.current.evaluationHistory).toEqual([]);
    expect(result.current.evaluationError).toBeNull();
    expect(result.current.isEvaluating).toBe(false);
  });

  // ... more tests
});
```

### Estimated Impact

- ~200-300 lines of test code per hook
- Total: ~600-900 lines of new tests
- Provides regression safety for future changes

---

## Phase 2: Manual Testing of Scenario/Replay Features

### Why This Task

- Hook integration changed handler flow
- Self-play replay logic was preserved but needs verification
- User-facing functionality must work correctly

### Test Scenarios

#### 2.1 Scenario Picker Loading

1. Open sandbox mode
2. Click "Load Scenario" button
3. Select a curated scenario
4. Verify: Board loads with correct state
5. Verify: Scenario name appears in UI
6. Verify: No console errors

#### 2.2 Self-Play Browser Loading

1. Open sandbox mode
2. Click "Browse Self-Play Games" button
3. Select a self-play game with recorded moves
4. Verify: Initial state loads
5. Verify: Move replay starts automatically
6. Verify: History slider shows correct move count
7. Verify: requestedReplayGameId is set (check ReplayPanel)

#### 2.3 Replay Mode Display

1. Load a self-play game
2. Verify: Replay panel appears
3. Verify: Can step through moves
4. Verify: Board state matches move history

#### 2.4 Fork from Replay

1. Load a replay
2. Navigate to mid-game position
3. Click "Fork" button
4. Verify: New playable game starts from that position
5. Verify: Can make moves
6. Verify: Scenario context is cleared

#### 2.5 Reset Scenario

1. Load a scenario
2. Make some moves
3. Click "Reset" button
4. Verify: Board returns to original scenario state
5. Verify: For self-play, move replay runs again

#### 2.6 History Playback

1. Load a scenario or play some moves
2. Use history slider
3. Verify: Can navigate to any move
4. Verify: Board state updates correctly
5. Verify: Cannot make moves while viewing history

---

## Phase 3: Tier 2 Sandbox Cleanup (Future)

### Current State

- SandboxGameHost: ~2,708 lines (down from ~2,893)
- Target: ~2,000 lines or less

### Potential Extraction Candidates

#### 3.1 Setup/Configuration UI (~200 lines)

Extract pre-game setup into a dedicated component:

```typescript
// useSandboxSetup.ts
export function useSandboxSetup() {
  // Board type selection
  // Player count selection
  // Player type configuration
  // Advanced options
}
```

#### 3.2 Game Controls/Actions (~150 lines)

Extract game action handlers:

```typescript
// useSandboxActions.ts
export function useSandboxActions() {
  // handleCopySandboxTrace
  // handleExportGameState
  // handleSaveToFile
  // handleNewGame
}
```

#### 3.3 Keyboard Shortcuts (~100 lines)

Extract keyboard handling:

```typescript
// useSandboxKeyboard.ts
export function useSandboxKeyboard() {
  // Keyboard shortcut handlers
  // Help overlay toggle
}
```

#### 3.4 AI Turn Management (~200 lines)

Extract AI turn logic:

```typescript
// useSandboxAI.ts
export function useSandboxAI() {
  // AI move scheduling
  // AI difficulty management
  // AI telemetry
}
```

### Estimated Impact

- Additional ~650 lines extractable
- Would bring SandboxGameHost to ~2,050 lines
- Requires careful dependency management

---

## Phase 4: FSM Validation Promotion (Blocked)

### Current Status

- FSM shadow validation infrastructure is in place
- Running in shadow mode (validates but doesn't enforce)
- Need confidence in shadow validation results before promotion

### Prerequisites

1. Collect shadow validation metrics over time
2. Analyze mismatch rate between FSM and legacy validation
3. Ensure 0% false positives in FSM validation
4. Document any edge cases found

### Promotion Steps (Once Ready)

1. Review shadow validation logs
2. Fix any FSM validation gaps
3. Gradually increase FSM authority
4. Eventually disable legacy path

---

## Implementation Order

```
┌─────────────────────────────────────────┐
│  Phase 1: Hook Tests                    │
│  Priority: P1                           │
│  Complexity: Medium                     │
│  Unblocks: Confident future refactoring │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Phase 2: Manual Testing                │
│  Priority: P1                           │
│  Complexity: Low                        │
│  Unblocks: Production confidence        │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Phase 3: Tier 2 Cleanup (Optional)     │
│  Priority: P2                           │
│  Complexity: High                       │
│  Unblocks: Long-term maintainability    │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Phase 4: FSM Promotion (Blocked)       │
│  Priority: P1                           │
│  Complexity: High                       │
│  Blocked by: Shadow validation metrics  │
└─────────────────────────────────────────┘
```

---

## Success Criteria

### Phase 1 (Hook Tests) ✅ COMPLETE

- [x] useSandboxPersistence has 19 test cases
- [x] useSandboxEvaluation has 19 test cases
- [x] useSandboxScenarios has 31 test cases (including extractChainCapturePath utility)
- [x] All tests pass (69 new tests total for the 3 hooks)

### Phase 2 (Manual Testing)

- [ ] All 6 scenario/replay features verified working
- [ ] No console errors during testing
- [ ] Document any issues found

### Phase 3 (Tier 2 Cleanup)

- [ ] SandboxGameHost reduced to ~2,000 lines
- [ ] All extracted hooks have tests
- [ ] No functionality regressions

### Phase 4 (FSM Promotion)

- [ ] Shadow validation shows 0% false positives
- [ ] FSM becomes primary validation path
- [ ] Legacy validation path removed

---

## Recommendations

**Immediate Focus:** Phase 1 (Hook Tests)

- Provides safety net for all previous refactoring
- Enables confident future changes
- Can be done incrementally

**After Hook Tests:** Phase 2 (Manual Testing)

- Quick verification pass
- Catches any integration issues
- Documents working state

**Defer:** Phase 3 and Phase 4

- Lower priority until core functionality verified
- Can be tackled in future sprints

---

**Document Maintainer:** Claude Code
**Last Updated:** December 2025
