# Notation Module

Algebraic notation conversion for RingRift game records.

## Overview

This module provides notation utilities per GAME_NOTATION_SPEC.md:

- Position to/from algebraic notation (a1, b2, etc.)
- Move to/from algebraic notation
- PGN-style game record format
- Move type codes

## Key Components

### Position Notation

```python
from app.notation import position_to_algebraic, algebraic_to_position

# Convert position to algebraic (0-indexed to chess-style)
notation = position_to_algebraic(row=0, col=0)  # "a1"
notation = position_to_algebraic(row=7, col=7)  # "h8"

# Convert algebraic back to position
row, col = algebraic_to_position("e4")  # (3, 4)
```

### Move Notation

```python
from app.notation import move_to_algebraic, algebraic_to_move

# Convert move dict to algebraic
move = {"from": [2, 3], "to": [4, 5], "type": "slide"}
notation = move_to_algebraic(move)  # "Sd3-e5"

# Parse algebraic notation to move dict
move = algebraic_to_move("Pd4")  # Place at d4
move = algebraic_to_move("Sc2-d3")  # Slide from c2 to d3
```

### Move Type Codes

```python
from app.notation import MOVE_TYPE_TO_CODE, CODE_TO_MOVE_TYPE

# Move type to single-letter code
MOVE_TYPE_TO_CODE["place"]    # "P"
MOVE_TYPE_TO_CODE["slide"]    # "S"
MOVE_TYPE_TO_CODE["recover"]  # "R"
MOVE_TYPE_TO_CODE["pass"]     # "-"

# Code to move type
CODE_TO_MOVE_TYPE["P"]  # "place"
CODE_TO_MOVE_TYPE["S"]  # "slide"
```

### PGN Format

```python
from app.notation import game_to_pgn, parse_pgn

# Convert game to PGN string
pgn = game_to_pgn(
    moves=[move1, move2, ...],
    metadata={
        "Event": "Training Game",
        "Date": "2025.12.24",
        "White": "AI-v3",
        "Black": "AI-v2",
        "Result": "1-0",
    },
)
# Returns:
# [Event "Training Game"]
# [Date "2025.12.24"]
# [White "AI-v3"]
# [Black "AI-v2"]
# [Result "1-0"]
#
# 1. Pd4 Pd5 2. Sc2-d3 Sf7-e6 ...

# Parse PGN back to moves
metadata, moves = parse_pgn(pgn_string)
```

### Notation List

```python
from app.notation import moves_to_notation_list

# Convert list of moves to notation
moves = [
    {"type": "place", "to": [3, 3]},
    {"type": "slide", "from": [3, 3], "to": [4, 4]},
]
notation_list = moves_to_notation_list(moves)
# ["Pd4", "Sd4-e5"]
```

## Notation Format

### Position Format

- Columns: a-h (or more for larger boards)
- Rows: 1-8 (or more for larger boards)
- Example: "e4", "a1", "h8"

### Move Format

```
<TypeCode><From>[-<To>]
```

| Type Code | Move Type | Example |
| --------- | --------- | ------- |
| P         | Place     | Pd4     |
| S         | Slide     | Sc2-d3  |
| R         | Recover   | Rb5     |
| -         | Pass      | -       |

### Extended Notation

For moves with additional information:

```
Sd4-e5x     # Capture
Sd4-e5+     # Check/threat
Sd4-e5!     # Strong move
Sd4-e5?     # Weak move
```

## Board Coordinate Systems

| Board Type | Columns | Rows | Example |
| ---------- | ------- | ---- | ------- |
| square8    | a-h     | 1-8  | e4      |
| square19   | a-s     | 1-19 | k10     |
| hex8       | a-i     | 1-9  | e5      |
| hexagonal  | a-y     | 1-25 | m13     |
