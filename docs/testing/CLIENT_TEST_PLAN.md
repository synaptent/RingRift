# React Frontend Test Coverage Plan - RingRift

**Created**: December 27, 2025
**Goal**: Achieve 80%+ coverage of critical frontend components

## Executive Summary

- **Current State**: 41 component tests + 65 subdirectory tests + 19 hook tests exist
- **Test Framework**: Jest + React Testing Library (already configured)
- **Total Components**: 44 main components, 10 UI components, 43 hooks, 10 pages, 5 contexts, 5 services
- **Coverage Gap**: ~30 critical untested components + hooks
- **Estimated Effort**: 120-150 tests (150-200 hours)

---

## Priority 1: Critical Path Components (Must Test)

### 1.1 Core Game Rendering

| Component        | Lines | Tests Needed | Est. Hours |
| ---------------- | ----- | ------------ | ---------- |
| BoardView.tsx    | 2,477 | 12-15        | 20         |
| GameHUD.tsx      | 2,234 | 12-15        | 20         |
| VictoryModal.tsx | 756   | 5-8          | 8          |
| ChoiceDialog.tsx | 491   | 3-5          | 6          |

**What's Already Tested**:

- VictoryModal: render, event handlers (partial)
- GameHUD: phase indicator, countdown (partial)
- ChoiceDialog: keyboard navigation, countdown (good coverage)
- BoardView: accessibility, chain capture, movement grid (good coverage)

**What's Missing**:

- BoardView: touch gestures, animation state, cell interactions
- GameHUD: connection status changes, dynamic player updates
- VictoryModal: rematch flow, all game end reasons
- ChoiceDialog: all choice types (recovery line, territory region)

**Test Files to Create**:

```
BoardView.touch-gestures.test.tsx
BoardView.animation.test.tsx
GameHUD.connection-status.test.tsx
VictoryModal.rematch-flow.test.tsx
ChoiceDialog.all-types.test.tsx
```

### 1.2 State Management Hooks

| Hook                     | Tests Needed | Est. Hours |
| ------------------------ | ------------ | ---------- |
| useGameState.ts          | 4-5          | 8          |
| useGameActions.ts        | 5-6          | 10         |
| useGameConnection.ts     | 4-5          | 10         |
| useCountdown.ts          | 2-3          | 4          |
| useKeyboardNavigation.ts | 3-4          | 6          |

**Missing Tests**:

- useGameState: error recovery, optimistic updates, undo/redo
- useGameConnection: WebSocket reconnection with backoff
- useKeyboardNavigation: arrow key grid navigation, focus trapping

### 1.3 Page-Level Integration

| Page                | Tests Needed | Est. Hours |
| ------------------- | ------------ | ---------- |
| BackendGameHost.tsx | 3-4          | 12         |
| SandboxGameHost.tsx | 4-5          | 15         |
| LobbyPage.tsx       | 3-4          | 10         |
| GamePage.tsx        | 2-3          | 8          |

---

## Priority 2: Important Non-Blocking Components

### 2.1 UI Components

| Component        | Tests Needed | Est. Hours |
| ---------------- | ------------ | ---------- |
| Dialog.tsx       | 2-3          | 3          |
| StatusBanner.tsx | 1-2          | 2          |
| InlineAlert.tsx  | 1-2          | 2          |

### 2.2 Game Data & Replay

| Component/Hook   | Tests Needed | Est. Hours |
| ---------------- | ------------ | ---------- |
| ReplayService.ts | 4-5          | 10         |
| ReplayPanel.tsx  | 3-4          | 8          |
| MoveHistory.tsx  | 2-3          | 4          |

---

## Priority 3: Nice-to-Have

- AccessibilityContext, SoundContext, SettingsModal
- TeachingOverlay, OnboardingModal, ScenarioPickerModal
- EvaluationPanel, EvaluationGraph
- useSandbox* hooks (8), useBackend* hooks (8)

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2) - 20 Tests

- BoardView: touch gestures, animations
- GameHUD: connection status
- VictoryModal: rematch flow
- ChoiceDialog: all choice types

### Phase 2: State Management (Weeks 3-4) - 18 Tests

- useGameState, useGameConnection, useKeyboardNavigation, useCountdown

### Phase 3: Page Integration (Weeks 5-6) - 12 Tests

- BackendGameHost, SandboxGameHost, LobbyPage

### Phase 4: Services & Contexts (Weeks 7-8) - 16 Tests

- ReplayService, SoundContext, API services

### Phase 5: Hook & Component Gap Closure (Weeks 9-10) - 30 Tests

- Remaining hooks and components

### Phase 6: Coverage Refinement (Week 11) - 20 Tests

- Edge cases, accessibility, mobile flows

---

## Testing Patterns

### Component with View Model

```typescript
describe('GameHUD', () => {
  it('should render with view model', () => {
    const vm = toHUDViewModel(gameState, currentPlayer);
    render(<GameHUD viewModel={vm} />);
    expect(screen.getByText(vm.phaseLabel)).toBeInTheDocument();
  });
});
```

### Custom Hook Testing

```typescript
describe('useGameConnection', () => {
  it('should reconnect with exponential backoff', async () => {
    const { result } = renderHook(() => useGameConnection());
    act(() => result.current.disconnect());
    await waitFor(() => expect(result.current.isConnected).toBe(true));
  });
});
```

### Page Integration Test

```typescript
describe('BackendGameHost full flow', () => {
  it('should complete game from join to victory', async () => {
    render(
      <GameContextProvider>
        <BackendGameHost gameId="123" />
      </GameContextProvider>
    );
    act(() => mockWebSocket.emit('gameStarted', ...));
    expect(screen.getByText(/Board/)).toBeInTheDocument();
  });
});
```

---

## Coverage Targets

| Category     | Current | Target |
| ------------ | ------- | ------ |
| BoardView    | 60%     | 85%    |
| GameHUD      | 70%     | 85%    |
| ChoiceDialog | 85%     | 90%    |
| Hooks        | 25%     | 70%    |
| Services     | 5%      | 60%    |

---

## Summary

| Category               | Tests    | Priority | Hours    |
| ---------------------- | -------- | -------- | -------- |
| Game Rendering         | 20       | P1       | 50       |
| State Management       | 18       | P1       | 40       |
| Page Integration       | 12       | P1       | 35       |
| Services               | 12       | P2       | 25       |
| UI Components          | 8        | P2       | 10       |
| Hooks (Lower Priority) | 26       | P3       | 40       |
| Edge Cases             | 20       | P3       | 30       |
| **TOTAL**              | **~120** | -        | **~230** |

**Recommendation**: Start with Priority 1 tests (~50 tests, ~125 hours) to achieve critical path coverage. These directly impact game reliability and user experience.
