# RingRift Architecture Analysis

> Generated: 2025-12-01
> Current State: Post-refactoring session (boardMutationHelpers, hasAnyRealAction extraction)

## Executive Summary

The RingRift codebase has two primary "god classes" that concentrate too many responsibilities:

- **GameEngine.ts**: ~~4,182~~ → 2,626 lines (40% reduction after legacy path removal), 54+ private properties
- **ClientSandboxEngine.ts**: 3,027 lines, 17+ private properties

This analysis identifies remaining refactoring opportunities, prioritized by impact and effort.

---

## Critical Issues (P0-P1)

### 1. God Class Decomposition

#### GameEngine.ts (~~4,344~~ → 2,626 lines)

**Current Responsibilities (8+):**

1. Game state management
2. Move validation and application
3. Decision phase orchestration
4. Chain capture state machine
5. LPS (Last Player Standing) victory tracking
6. Swap sides (pie rule) handling
7. Timer/clock management
8. Territory processing coordination
9. Line detection and rewards

**Recommended Extraction:**

| Component                | Responsibility                     | Est. Lines | Effort |
| ------------------------ | ---------------------------------- | ---------- | ------ |
| DecisionPhaseManager     | Handle all decision phase logic    | ~400       | HIGH   |
| ChainCaptureOrchestrator | Chain capture state machine        | ~300       | MEDIUM |
| LpsVictoryManager        | LPS tracking and victory detection | ~200       | MEDIUM |
| SwapSidesOrchestrator    | Pie rule swap handling             | ~150       | LOW    |
| GameEngineCore           | Reduced core with delegation       | ~2,500     | -      |

**Priority**: P1 (High)
**Total Effort**: 200-400 hours

#### ClientSandboxEngine.ts (3,027 lines)

**Current Responsibilities (9+):**

1. Local game state simulation
2. Human player interaction
3. AI move coordination
4. Capture enumeration and application
5. Territory processing
6. Victory detection
7. Move validation
8. History/undo management
9. Decision UI coordination

**Recommended Extraction:**

| Component              | Responsibility                 | Est. Lines | Effort |
| ---------------------- | ------------------------------ | ---------- | ------ |
| SandboxCaptureManager  | Capture logic and chain state  | ~350       | MEDIUM |
| SandboxDecisionHandler | Decision phase UI coordination | ~300       | MEDIUM |
| SandboxVictoryManager  | Victory detection and end-game | ~200       | LOW    |
| SandboxEngineCore      | Reduced core with delegation   | ~1,800     | -      |

**Priority**: P1 (High)
**Total Effort**: 200-300 hours

---

### 2. Remaining Code Duplication

#### ~~Capture Enumeration~~ ✅ COMPLETED

- **Location**: GameEngine lines ~2100-2250 vs CaptureAggregate
- **Issue**: ~~Similar capture enumeration logic exists in both~~ **FIXED**: Dead code removed
- **Fix**: ~~Fully delegate to shared CaptureAggregate~~ **DONE**: Removed `performOvertakingCapture`, `eliminateTopRingAt`, `processMarkersAlongPath` (-162 lines)
- **Effort**: ~~LOW (2-4 hours)~~ Completed
- **Priority**: ~~P0~~ **DONE**

#### ~~Territory Detection~~ ✅ COMPLETED

- **Locations**:
  - `src/shared/engine/territoryDetection.ts` - authoritative implementation
  - ~~`src/server/game/rules/territoryProcessing.ts`~~ - never existed
  - ~~`src/client/sandbox/sandboxTerritoryEngine.ts`~~ - never existed
- **Issue**: ~~Three implementations~~ **FIXED**: TerritoryAggregate duplicated territoryDetection.ts
- **Fix**: ~~Consolidate~~ **DONE**: TerritoryAggregate now delegates to shared territoryDetection.ts (-359 lines)
- **Effort**: ~~MEDIUM (4-6 hours)~~ Completed
- **Priority**: ~~P1~~ **DONE**

#### ~~Victory Detection~~ ✅ ALREADY COMPLETE

- **Locations**:
  - `src/shared/engine/aggregates/VictoryAggregate.ts` - authoritative implementation
  - `src/client/sandbox/sandboxVictory.ts` - thin adapter using shared VictoryAggregate
- **Issue**: ~~Parallel implementations~~ **NOT AN ISSUE**: Already properly shared
- **Fix**: ~~Extract to shared VictoryDetector~~ **DONE**: VictoryAggregate already exists, sandboxVictory imports `evaluateVictory` from shared
- **Effort**: ~~MEDIUM (6-8 hours)~~ Already done
- **Priority**: ~~P2~~ **DONE**

---

### 3. Large Files Requiring Attention

| File                   | Lines           | Primary Issue           | Priority |
| ---------------------- | --------------- | ----------------------- | -------- |
| GameEngine.ts          | ~~4,344~~ 2,626 | God class (reduced 40%) | P1       |
| ClientSandboxEngine.ts | 3,027           | God class               | P1       |
| GameSession.ts         | ~1,900          | Mixed concerns          | P2       |
| game.ts (routes)       | ~1,950          | Logic in routes         | P2       |
| BoardView.tsx          | ~1,730          | UI + logic mixed        | P2       |
| SandboxGameHost.tsx    | ~1,680          | Complex state           | P2       |
| WebSocketServer.ts     | ~1,530          | Scattered handlers      | P2       |

---

### 4. Long Methods (>100 lines)

#### GameEngine.ts

| Method                       | Lines   | Location    | Issue                           |
| ---------------------------- | ------- | ----------- | ------------------------------- |
| ~~applyMove~~                | ~~190~~ | ~~Removed~~ | Deprecated - removed 2025-12-02 |
| processAutomaticConsequences | ~160    | Line 1730   | Complex phase transitions       |
| advanceGame                  | ~140    | Line 1900   | State machine logic             |
| handleChainCaptureDecision   | ~120    | Line 2050   | Chain capture complexity        |

> **Note (2025-12-02)**: The `applyMove` method and related legacy path code (~2,720 lines total)
> have been removed. All move processing now goes through the shared orchestrator.

#### ClientSandboxEngine.ts

| Method                     | Lines | Location  | Issue              |
| -------------------------- | ----- | --------- | ------------------ |
| applyCanonicalMoveInternal | ~140  | Line 1800 | Move type dispatch |

**Effort**: 4-8 hours per method to decompose
**Priority**: P2

---

## Structural Improvements (P2)

### 1. Orchestration Layer Consolidation

**Current State (Updated 2025-12-02):**

- Backend: TurnEngineAdapter (orchestrator at 100% - legacy path removed)
- Client: SandboxOrchestratorAdapter + turnOrchestrator
- ~2,720 lines of legacy code removed from GameEngine

**Status**: ✅ COMPLETED - Legacy path fully removed. The orchestrator is now the only
move processing path. Rollout infrastructure retained for circuit breaker monitoring.

**Remaining Work**:

- [ ] Remove orchestrator rollout service (provides circuit breaker monitoring)
- [ ] Remove feature flags: `ORCHESTRATOR_ADAPTER_ENABLED`, `ORCHESTRATOR_ROLLOUT_PERCENTAGE`

**Effort**: LOW (8-16 hours for remaining cleanup)
**Priority**: P3

### 2. Test Coverage Gaps

| Component           | Current | Target | Gap |
| ------------------- | ------- | ------ | --- |
| GameEngine          | ~70%    | 85%    | 15% |
| ClientSandboxEngine | ~65%    | 80%    | 15% |
| GameSession         | ~50%    | 75%    | 25% |
| WebSocket handlers  | ~40%    | 70%    | 30% |

**Priority Areas:**

1. Chain capture edge cases
2. Territory self-elimination rules
3. LPS victory conditions
4. Swap sides timing

**Effort**: 80-120 hours total
**Priority**: P1-P2

### 3. Type Safety Improvements

**Issues Found:**

- `any` types in 23 locations
- Missing strict null checks in move handlers
- Loose typing in WebSocket message handlers

**Effort**: LOW (8-12 hours)
**Priority**: P3

---

## Quick Wins (< 1 week)

| Task                                     | Effort | Impact          | Priority |
| ---------------------------------------- | ------ | --------------- | -------- |
| Remove capture enumeration duplication   | 2-4h   | Consistency     | P0       |
| Consolidate territory detection          | 4-6h   | Maintainability | P1       |
| Extract victory detection to shared      | 6-8h   | DRY             | P2       |
| Add missing JSDoc to public APIs         | 4-6h   | Documentation   | P3       |
| Remove unused exports from shared/engine | 2-3h   | Bundle size     | P3       |

---

## Implementation Phases

### Phase 1: Critical Fixes (1-2 weeks)

- [ ] Consolidate capture enumeration to CaptureAggregate
- [ ] Unify territory detection implementations
- [ ] Add critical test coverage for chain captures

### Phase 2: God Class Decomposition (2-4 months)

- [ ] Extract DecisionPhaseManager from GameEngine
- [ ] Extract ChainCaptureOrchestrator from GameEngine
- [ ] Extract SandboxCaptureManager from ClientSandboxEngine
- [ ] Extract SandboxDecisionHandler from ClientSandboxEngine

### Phase 3: Quality Improvements (1-2 months)

- [ ] Decompose long methods (>100 lines)
- [ ] Add comprehensive test coverage
- [ ] Consolidate orchestration layer
- [ ] Extract victory detection to shared

### Phase 4: Maintenance (Ongoing)

- [ ] Type safety improvements
- [ ] Documentation updates
- [ ] Remove deprecated code per P20.3-2 timeline

---

## Effort Estimation Summary

| Phase     | Hours       | Timeline (1 dev) |
| --------- | ----------- | ---------------- |
| Phase 1   | 15-25       | 1-2 weeks        |
| Phase 2   | 400-600     | 2-4 months       |
| Phase 3   | 150-200     | 1-2 months       |
| Phase 4   | Ongoing     | -                |
| **Total** | **565-825** | **4-8 months**   |

---

## Recent Refactoring Completed

### Session: 2025-12-01

| Task                                | Result                                                                      | Lines Saved                   |
| ----------------------------------- | --------------------------------------------------------------------------- | ----------------------------- |
| Created boardMutationHelpers.ts     | New shared module                                                           | -104 (GameEngine)             |
| Extracted hasAnyRealActionForPlayer | Shared delegate pattern                                                     | -32 total                     |
| Removed stringToPositionLocal       | Use shared helper                                                           | -19 (Sandbox)                 |
| Consolidated shouldOfferSwapSides   | Already shared                                                              | -                             |
| Reviewed ClockManager extraction    | Skipped (minimal benefit)                                                   | -                             |
| Verified EliminationHelpers         | Already in territoryDecisionHelpers.ts                                      | -                             |
| Reviewed deprecated methods         | Kept per DO NOT REMOVE notes                                                | -                             |
| **P0: Removed dead capture code**   | `performOvertakingCapture`, `eliminateTopRingAt`, `processMarkersAlongPath` | **-162 (GameEngine)**         |
| **P1: Unified territory detection** | TerritoryAggregate now delegates to shared territoryDetection.ts            | **-359 (TerritoryAggregate)** |
| **P2: Victory detection**           | Already shared via VictoryAggregate + sandboxVictory                        | -                             |

**Net Result:**

- GameEngine: 4,463 → 4,182 lines (**-281 total**)
- ClientSandboxEngine: 3,063 → 3,027 lines (-36)
- TerritoryAggregate: 1,553 → 1,194 lines (**-359**)
- **Combined: -676 lines of code removed**

---

## Files Reference

### Shared Engine Modules

```
src/shared/engine/
├── index.ts                    # Main exports
├── core.ts                     # Core utilities
├── boardMutationHelpers.ts     # NEW: Map/Array mutation helpers
├── playerStateHelpers.ts       # Player action availability
├── swapSidesHelpers.ts         # Pie rule logic
├── historyHelpers.ts           # Move history utilities
├── territoryDecisionHelpers.ts # Elimination helpers
├── territoryDetection.ts       # Territory geometry
├── territoryProcessing.ts      # Territory application
├── aggregates/
│   ├── CaptureAggregate.ts     # Capture enumeration
│   ├── LineAggregate.ts        # Line detection
│   └── TerritoryAggregate.ts   # Territory processing
└── orchestration/
    └── turnOrchestrator.ts     # Phase transitions
```

### Main Engine Files

```
src/server/game/GameEngine.ts        # 2,626 lines - Backend authority (reduced from 4,182)
src/client/sandbox/ClientSandboxEngine.ts  # 3,027 lines - Client simulation
```
