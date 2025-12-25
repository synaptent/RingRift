# Game Engine Module

Legacy game engine wrapper for RingRift AI service.

## Overview

This module provides the Python game engine implementation that mirrors the TypeScript source of truth. It's currently being migrated to use `app.board_manager` directly.

**Note**: This module is deprecated. New code should use `app.board_manager.BoardManager` directly.

**Target Removal Date**: Q2 2026

## Key Components

### GameEngine (Legacy)

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
from app.game_engine import PhaseRequirement

# Define requirements for game phases
req = PhaseRequirement(
    min_pieces=3,
    max_pieces=10,
    required_territory=0.5,
)
```

## Migration Path

Replace `GameEngine` usage with `BoardManager`:

```python
# OLD (deprecated)
from app.game_engine import GameEngine
engine = GameEngine(board_type="hex8", num_players=2)
state = engine.create_initial_state()

# NEW (recommended)
from app.board_manager import BoardManager
board = BoardManager(board_type="hex8", num_players=2)
state = board.create_initial_state()
```

## See Also

- `app.board_manager` - Current canonical game logic
- `src/shared/engine/` - TypeScript source of truth
