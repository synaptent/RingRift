# turnOrchestrator.ts Decomposition Plan

**Status:** PLANNING
**Risk Level:** EXTREME - This is the SOURCE OF TRUTH for game rules
**Current Size:** 3,963 lines, 117 commits

## Overview

The `turnOrchestrator.ts` file is the canonical game rules implementation that:

1. Orchestrates all turn phases (placement, movement, capture, territory, line processing)
2. Validates moves via FSM integration
3. Handles victory detection and game ending
4. Manages special rules (pie rule, LPS, forced eliminations)

## Current Exports

| Export                     | Line | Purpose                           |
| -------------------------- | ---- | --------------------------------- |
| `toVictoryState`           | 411  | Convert GameState to VictoryState |
| `ProcessTurnOptions`       | 1348 | Options interface for processTurn |
| `processTurn`              | 1418 | Main turn processing function     |
| `validateMove`             | 3387 | Validate a single move            |
| `getValidMoves`            | 3488 | Enumerate all valid moves         |
| `hasValidMoves`            | 3825 | Check if any valid moves exist    |
| `ApplyMoveForReplayResult` | 3929 | Result type for replay            |
| `applyMoveForReplay`       | 3951 | Apply move for replay validation  |

## Decomposition Target

```
src/shared/engine/orchestration/
├── turnOrchestrator.ts      # Main entry (~500 lines) - delegates to modules
├── phaseHandlers/           # Phase-specific logic
│   ├── index.ts
│   ├── placement.ts         # Placement phase handling (~400 lines)
│   ├── movement.ts          # Movement phase handling (~400 lines)
│   ├── capture.ts           # Capture phase handling (~300 lines)
│   ├── territory.ts         # Territory phase handling (~300 lines)
│   └── lineProcessing.ts    # Line processing phase (~400 lines)
├── victory.ts               # Victory detection (~600 lines)
├── specialRules.ts          # Pie rule, LPS, forced eliminations (~400 lines)
├── moveValidation.ts        # validateMove, getValidMoves, hasValidMoves (~500 lines)
├── replay.ts                # applyMoveForReplay (~200 lines)
└── types.ts                 # Already exists, keep as is
```

## Migration Strategy

### Phase 1: Extract Types and Helpers (Low Risk)

1. Move internal helper functions to separate files
2. Keep all exports at top level
3. Test parity after each extraction

### Phase 2: Extract Phase Handlers (Medium Risk)

1. Create `phaseHandlers/` directory
2. Extract one phase at a time
3. Run full parity test suite after each

### Phase 3: Extract Victory Logic (Medium Risk)

1. Move `toVictoryState` and victory detection
2. Move game ending logic
3. Verify parity

### Phase 4: Extract Special Rules (Medium Risk)

1. Move pie rule handling
2. Move LPS logic
3. Move forced eliminations
4. Verify parity

### Phase 5: Extract Move Validation (Low Risk)

1. Move `validateMove`, `getValidMoves`, `hasValidMoves`
2. These are leaf functions with clear interfaces

### Phase 6: Refactor Core Orchestrator

1. Make `processTurn` delegate to extracted modules
2. Keep main orchestrator thin (~500 lines)

## Critical Success Criteria

1. **ALL aggregate imports remain at top level** - no breaking existing code
2. **No change to public API** - same exports, same signatures
3. **100% parity verification** - run `check_ts_python_replay_parity.py` with 10K+ seeds
4. **FSM validation unchanged** - FSM layer must work exactly as before
5. **Game end explanation logic preserved** - critical for UX

## Testing Protocol

Before each PR merge:

```bash
# Run TypeScript tests
npm test -- --testPathPattern="turnOrchestrator|orchestration"

# Run parity tests (10K seeds minimum)
cd ai-service
python scripts/check_ts_python_replay_parity.py \
  --db data/games/canonical_hex8.db \
  --games 10000

# Run replay validation
python scripts/run_canonical_selfplay_parity_gate.py --board-type hex8
```

## Risk Mitigation

1. **Feature branch:** `feature/turn-orchestrator-decomposition`
2. **Small PRs:** One phase per PR (not one giant PR)
3. **Rollback ready:** Keep original file until fully validated
4. **Python parity:** Ensure Python mirror stays in sync

## Dependencies

The turnOrchestrator imports from:

- `../aggregates/*` - Domain aggregates (Placement, Movement, Capture, Territory, Victory)
- `../fsm` - FSM validation
- `../gameEndExplanation` - Game end UX
- `../globalActions` - Global action helpers
- `../legacy/*` - Legacy move handling

These dependencies should remain unchanged.

## Timeline

| Phase                    | Effort     | Risk            |
| ------------------------ | ---------- | --------------- |
| Phase 1: Types/Helpers   | 1 day      | Low             |
| Phase 2: Phase Handlers  | 3 days     | Medium          |
| Phase 3: Victory Logic   | 1 day      | Medium          |
| Phase 4: Special Rules   | 1 day      | Medium          |
| Phase 5: Move Validation | 1 day      | Low             |
| Phase 6: Core Refactor   | 1 day      | Medium          |
| **Total**                | **8 days** | **Medium-High** |

## Decision: Proceed or Defer?

Given the extreme risk (SOURCE OF TRUTH), recommend:

1. **Defer full decomposition** until Q1 2026
2. **Start with low-risk extractions** (move validation, replay)
3. **Keep phase handlers inline** for now
4. **Focus on Python infrastructure** improvements first

The p2p_orchestrator.py decomposition (28K lines) has higher ROI and lower risk.
