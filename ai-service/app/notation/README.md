# Notation Module

Algebraic notation conversion for RingRift game records.

## Scope

This module implements the algebraic notation defined in
`ai-service/docs/specs/GAME_NOTATION_SPEC.md`. It is intended for
human-readable exports and debugging, not as a canonical replay format.

Coverage notes:

- Encodes canonical move types: `place_ring`, `skip_placement`, `swap_sides`,
  `move_stack`, `overtaking_capture`, `continue_capture_segment`,
  `process_line`, `choose_line_option`, `choose_territory_option`,
  `eliminate_rings_from_stack`, `forced_elimination`.
- Legacy aliases accepted for replay compatibility: `move_ring`, `build_stack`,
  `choose_line_reward`, `process_territory_region`, `line_formation`,
  `territory_claim`, `chain_capture`.
- Not encoded: `no_*_action`, `skip_capture`, `recovery_slide`,
  `skip_recovery`, `skip_territory_processing`, `resign`, `timeout`.

## API

```python
from app.notation import (
    position_to_algebraic,
    algebraic_to_position,
    move_to_algebraic,
    algebraic_to_move,
    moves_to_notation_list,
    game_to_pgn,
    parse_pgn,
)
```

## Position Notation

```python
from app.models import BoardType, Position
from app.notation import position_to_algebraic, algebraic_to_position

pos = Position(x=3, y=4)
assert position_to_algebraic(pos, BoardType.SQUARE8) == "d5"

hex_pos = Position(x=3, y=-2, z=-1)
assert position_to_algebraic(hex_pos, BoardType.HEXAGONAL) == "3.-2"

assert algebraic_to_position("d5", BoardType.SQUARE8) == pos
assert algebraic_to_position("3.-2", BoardType.HEXAGONAL) == hex_pos
```

## Move Notation

```python
from datetime import datetime
from app.models import BoardType, Move, MoveType, Position
from app.notation import move_to_algebraic, algebraic_to_move

move = Move(
    id="m1",
    type=MoveType.OVERTAKING_CAPTURE,
    player=1,
    from_pos=Position(x=0, y=0),
    to=Position(x=2, y=0),
    capture_target=Position(x=1, y=0),
    timestamp=datetime.now(),
    think_time=0,
    move_number=12,
)

assert move_to_algebraic(move, BoardType.SQUARE8) == "C a1-c1 xb1"

# algebraic_to_move needs GameState context plus player and move_number.
state = ...  # GameState for context
parsed = algebraic_to_move("C a1-c1 xb1", state, player=1, move_number=12)
```

## Game Records (PGN-style)

```python
from app.notation import game_to_pgn, moves_to_notation_list, parse_pgn

notation_list = moves_to_notation_list([move], BoardType.SQUARE8)

pgn = game_to_pgn(
    moves=[move],
    metadata={
        "game_id": "game-1",
        "board": "square8",
        "date": "2025-12-30",
        "player1": "AI-1",
        "player2": "AI-2",
        "result": "1-0",
    },
    board_type=BoardType.SQUARE8,
)

metadata, moves = parse_pgn(pgn)
```

## Coordinate Systems

- Square boards: chess-style `a1` to `h8` (8x8) or `a1` to `s19` (19x19).
- Hex boards: cube coordinates encoded as `x.y` (for example `3.-2`).
