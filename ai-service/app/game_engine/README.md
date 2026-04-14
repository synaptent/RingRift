# Game Engine Module

Canonical Python game-engine package for RingRift replay and rules semantics.

## Overview

This package is the stable public import surface for the Python game engine mirror used by replay, parity, and canonical rules enforcement.

- `GameEngine` remains the supported import path: `from app.game_engine import GameEngine`
- `PhaseRequirement` and `PhaseRequirementType` live in `phase_requirements.py` and describe the bookkeeping moves hosts must emit
- Direct imports from `app._game_engine_legacy` are deprecated compatibility only

## Key Components

### GameEngine

```python
from app.game_engine import GameEngine

# Create engine instance
engine = GameEngine(board_type="hex8", num_players=2)

# Initialize game state
state = engine.create_initial_state()

# Get valid moves
moves = engine.get_valid_moves(state)

# Apply a move
new_state = engine.apply_move(state, move)

# Check game over
is_over, winner = engine.is_game_over(state)
```

### PhaseRequirement

```python
from app.game_engine import PhaseRequirement, PhaseRequirementType

# Describe the bookkeeping move a host must emit
req = PhaseRequirement(
    type=PhaseRequirementType.NO_TERRITORY_ACTION_REQUIRED,
    player=2,
    eligible_positions=[],
)
```

Hosts should treat `PhaseRequirement` as a structural signal. When the engine surfaces one of these requirements, the host is responsible for constructing the corresponding canonical move and applying it.

## Compatibility Notes

- Import `GameEngine` from `app.game_engine`, not `app._game_engine_legacy`
- Use `app.board_manager` only for lower-level board helpers; it is not the supported replacement public facade for the game engine package
- `PhaseRequirementType` is the canonical enum for no-action and forced-elimination bookkeeping requirements

## See Also

- `app.game_engine.phase_requirements` - canonical phase-requirement types
- `app.board_manager` - lower-level board helpers used by rules and diagnostics
- `src/shared/engine/` - TypeScript source of truth
