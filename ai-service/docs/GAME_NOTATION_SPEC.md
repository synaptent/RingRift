# RingRift Game Notation Specification

## Overview

This document specifies the algebraic notation system for RingRift games, covering:

- Board coordinate systems for all board types
- Human-readable algebraic position notation
- Move encoding format for game records
- Turn action vs turn vs move taxonomy

## 1. Coordinate Systems

### 1.1 Square Boards (8×8 and 19×19)

**Internal Representation**: Cartesian coordinates `(x, y)` where:

- `x` = column (0-indexed from left)
- `y` = row (0-indexed from bottom)

**Algebraic Notation**: Chess-style `<column><row>`

- Columns: `a-h` (8×8) or `a-s` (19×19)
- Rows: `1-8` (8×8) or `1-19` (19×19)

```
Square 8×8 Board Layout:

    a   b   c   d   e   f   g   h
  +---+---+---+---+---+---+---+---+
8 |   |   |   |   |   |   |   |   | 8
  +---+---+---+---+---+---+---+---+
7 |   |   |   |   |   |   |   |   | 7
  +---+---+---+---+---+---+---+---+
6 |   |   |   |   |   |   |   |   | 6
  +---+---+---+---+---+---+---+---+
5 |   |   |   |   |   |   |   |   | 5
  +---+---+---+---+---+---+---+---+
4 |   |   |   |   |   |   |   |   | 4
  +---+---+---+---+---+---+---+---+
3 |   |   |   |   |   |   |   |   | 3
  +---+---+---+---+---+---+---+---+
2 |   |   |   |   |   |   |   |   | 2
  +---+---+---+---+---+---+---+---+
1 |   |   |   |   |   |   |   |   | 1
  +---+---+---+---+---+---+---+---+
    a   b   c   d   e   f   g   h
```

**Coordinate Mapping**:

```
Position(x=0, y=0) → "a1" (bottom-left)
Position(x=7, y=7) → "h8" (top-right, 8×8)
Position(x=3, y=4) → "d5"
```

**Conversion Functions**:

```python
def position_to_algebraic_square(pos: Position) -> str:
    col = chr(ord('a') + pos.x)
    row = str(pos.y + 1)
    return f"{col}{row}"

def algebraic_to_position_square(notation: str) -> Position:
    col = ord(notation[0]) - ord('a')
    row = int(notation[1:]) - 1
    return Position(x=col, y=row)
```

### 1.2 Hexagonal Board

**Internal Representation**: Cube coordinates `(x, y, z)` where:

- Constraint: `x + y + z = 0` (always)
- Board radius: 11 (center at origin)
- Valid positions: `|x| ≤ 11`, `|y| ≤ 11`, `|z| ≤ 11`
- Total spaces: 469

**Algebraic Notation Options**:

#### Option A: Compact Cube Notation (Primary)

Format: `<x>.<y>` (z is derived from constraint)

```
Position(x=0, y=0, z=0)   → "0.0"     (center)
Position(x=1, y=0, z=-1)  → "1.0"     (east of center)
Position(x=-3, y=5, z=-2) → "-3.5"
Position(x=11, y=-11, z=0) → "11.-11" (far edge)
```

**Conversion Functions**:

```python
def position_to_algebraic_hex(pos: Position) -> str:
    return f"{pos.x}.{pos.y}"

def algebraic_to_position_hex(notation: str) -> Position:
    parts = notation.split('.')
    x = int(parts[0])
    y = int(parts[1])
    z = -x - y  # Derived from cube constraint
    return Position(x=x, y=y, z=z)
```

#### Option B: Ring-Direction Notation (Human-Friendly Alternative)

Format: `<direction><ring>` for edge positions, `c` for center

```
Directions: n, ne, e, se, s, sw, w, nw (8 cardinal/ordinal)
Ring: 1-11 (distance from center)

Center:       "c"
Ring 1 East:  "e1"
Ring 5 NE:    "ne5"
Ring 11 South: "s11"
```

This notation is less precise for off-axis positions and is provided
as an optional human-readable format for common positions.

### 1.3 Coordinate Summary Table

| Board Type   | Internal Format | Algebraic Format | Example              |
| ------------ | --------------- | ---------------- | -------------------- |
| Square 8×8   | `(x, y)` 0-7    | `a1` to `h8`     | `d4` = (3, 3)        |
| Square 19×19 | `(x, y)` 0-18   | `a1` to `s19`    | `k10` = (10, 9)      |
| Hexagonal    | `(x, y, z)`     | `x.y`            | `3.-2` = (3, -2, -1) |

---

## 2. Turn Action Taxonomy

### 2.1 Definitions

**Turn Action** (atomic unit)
: A single discrete action taken by a player that modifies game state.
This is the fundamental unit of game progression.

**Turn** (player opportunity)
: A complete sequence of turn actions taken by one player before control
passes to another player. May contain multiple turn actions due to
chain captures, line processing, or territory claims.

**Move** (legacy term)
: Historically used interchangeably with "turn action". In this spec,
"move" specifically refers to spatial movement actions (moving rings/stacks).

**Game Record**
: The complete sequence of turn actions from game start to termination.

### 2.2 Turn Action Types

#### Phase: PLACEMENT

| Action Type      | Code | Description                                         |
| ---------------- | ---- | --------------------------------------------------- |
| `place_ring`     | `P`  | Place a ring from hand onto an empty space          |
| `skip_placement` | `SP` | Skip placement (pass)                               |
| `swap_sides`     | `SW` | Invoke pie rule (swap colors after first placement) |

#### Phase: MOVEMENT

| Action Type  | Code | Description                                       |
| ------------ | ---- | ------------------------------------------------- |
| `move_stack` | `M`  | Move entire stack to adjacent space               |
| `move_ring`  | `MR` | Move single ring from stack (if stack height > 1) |

#### Phase: CAPTURE (Chain)

| Action Type          | Code | Description                                        |
| -------------------- | ---- | -------------------------------------------------- |
| `overtaking_capture` | `C`  | Initial capture: jump over opponent to land beyond |
| `continue_capture`   | `CC` | Continue chain capture from landing position       |
| `end_capture_chain`  | `EC` | Explicitly end capture chain (when optional)       |

#### Phase: LINE_PROCESSING

| Action Type          | Code | Description                                |
| -------------------- | ---- | ------------------------------------------ |
| `process_line`       | `L`  | Select which line to process (if multiple) |
| `choose_line_reward` | `LR` | Choose reward option for formed line       |

#### Phase: TERRITORY_PROCESSING

| Action Type         | Code | Description                             |
| ------------------- | ---- | --------------------------------------- |
| `process_territory` | `T`  | Process a disconnected territory region |

#### Phase: ELIMINATION

| Action Type       | Code | Description                                     |
| ----------------- | ---- | ----------------------------------------------- |
| `eliminate_rings` | `E`  | Forced elimination when stack exceeds cap       |
| `build_stack`     | `B`  | Move rings within controlled territory (legacy) |

### 2.3 Turn Structure Examples

**Simple Turn** (single action):

```
Turn 1 (P1): P d4        # Player 1 places at d4
Turn 2 (P2): P e5        # Player 2 places at e5
```

**Complex Turn** (multiple actions):

```
Turn 15 (P1): M d4-e5    # Move stack, triggers line
             L 1         # Process line 1
             LR 2        # Choose line reward option 2
```

**Chain Capture Turn**:

```
Turn 20 (P2): C a1-c1 xb1      # Capture over b1
             CC c1-e1 xd1      # Continue over d1
             CC e1-g1 xf1      # Continue over f1
             EC                 # End chain (no more captures)
```

---

## 3. Move Notation Format

### 3.1 Syntax Grammar

```ebnf
turn_action   = action_code, [position], ["-", position], [metadata] ;
action_code   = "P" | "SP" | "SW" | "M" | "MR" | "C" | "CC" | "EC"
              | "L" | "LR" | "T" | "E" | "B" ;
position      = square_pos | hex_pos ;
square_pos    = column, row ;
column        = "a".."s" ;
row           = digit, [digit] ;
hex_pos       = integer, ".", integer ;
metadata      = {" ", meta_item} ;
meta_item     = capture_target | marker_placed | line_ref | option_ref ;
capture_target = "x", position ;
marker_placed  = "@", position ;
line_ref       = "#L", digit ;
option_ref     = "#", digit ;
```

### 3.2 Action Notation Examples

#### Placement Phase

```
P d4          # Place ring at d4
P 3.-2        # Place ring at hex position (3, -2, -1)
SP            # Skip placement
SW            # Swap sides (pie rule)
```

#### Movement Phase

```
M d4-e5       # Move stack from d4 to e5
M 0.0-1.0     # Move stack on hex board
MR d4-e5      # Move single ring from d4 to e5 (partial stack)
```

#### Capture Phase

```
C a1-c1 xb1   # Capture: jump from a1 over b1 to c1
CC c1-e1 xd1  # Chain capture continuation
EC            # End capture chain
```

#### Line Processing Phase

```
L 1           # Process line index 1 (if multiple lines formed)
LR 2          # Choose line reward option 2
```

#### Territory Processing Phase

```
T 3           # Process territory region 3
```

#### Elimination Phase

```
E d4          # Eliminate rings from stack at d4 (forced)
```

### 3.3 Metadata Annotations

| Annotation | Meaning                    | Example       |
| ---------- | -------------------------- | ------------- |
| `x<pos>`   | Capture target position    | `C a1-c1 xb1` |
| `@<pos>`   | Marker left at position    | `M d4-e5 @d4` |
| `#L<n>`    | Line reference             | `L 1 #L1`     |
| `#<n>`     | Option/choice index        | `LR 2 #2`     |
| `!`        | Check (opponent in danger) | `M d4-e5!`    |
| `!!`       | Brilliant move             | `C a1-c1!!`   |
| `?`        | Dubious move               | `P d4?`       |
| `+`        | Forms line                 | `M d4-e5+`    |
| `++`       | Territory claim            | `M d4-e5++`   |

---

## 4. Game Record Format

### 4.1 Header Format

```
[Game "RingRift"]
[Board "square8"]
[Date "2024-12-01"]
[Player1 "Alice"]
[Player2 "Bob"]
[Result "1-0"]
[Termination "ring_elimination"]
[RNGSeed "12345"]
```

### 4.2 Move List Format

**Turn-based notation** (primary):

```
1. P d4      P e5
2. P c3      P f6
3. M d4-e4   M e5-d5
4. C e4-e6 xe5  M d5-c5
...
```

**Action-sequence notation** (for complex turns):

```
1. P d4
2. P e5
3. P c3
4. P f6
5. M d4-e4
6. M e5-d5
7. C e4-e6 xe5
   L 1
   LR 2
8. M d5-c5
...
```

### 4.3 Complete Game Example (Square 8×8)

```
[Game "RingRift"]
[Board "square8"]
[Date "2024-12-01"]
[Player1 "HeuristicAI_v1"]
[Player2 "HeuristicAI_v2"]
[Result "1-0"]
[Termination "ring_elimination"]
[TotalMoves "47"]

1. P d4      P e5
2. P c4      P d5
3. P e4      P c5
4. P d3      P e6
5. M d4-d5   M e5-e4
6. M c4-d4   M d5-c4
7. M e4-d4+  L 1; LR 1
8. M c5-d5   C d4-b4 xc4
9. CC b4-b2 xb3  EC
...
47. M g7-h8  {P1 wins by ring elimination}
```

### 4.4 Hexagonal Board Example

```
[Game "RingRift"]
[Board "hexagonal"]
[Date "2024-12-01"]
[Player1 "Alice"]
[Player2 "Bob"]
[Result "0-1"]

1. P 0.0     P 1.0
2. P -1.1    P 2.-1
3. M 0.0-1.-1  M 1.0-0.1
4. C 1.-1-3.-1 x2.-1  EC
...
```

---

## 5. Implementation API

### 5.1 TypeScript Interface

```typescript
// src/shared/notation/algebraic.ts

interface NotationOptions {
  boardType: BoardType;
  compactForm: boolean; // Short vs verbose notation
  includeMetadata: boolean;
}

// Position conversion
function positionToAlgebraic(pos: Position, boardType: BoardType): string;
function algebraicToPosition(notation: string, boardType: BoardType): Position;

// Move conversion
function moveToAlgebraic(move: Move, options: NotationOptions): string;
function algebraicToMove(notation: string, state: GameState): Move | null;

// Game record
function gameToNotation(game: GameRecord): string;
function notationToGame(notation: string): GameRecord | null;

// Validation
function isValidNotation(notation: string, boardType: BoardType): boolean;
```

### 5.2 Python Interface

```python
# ai-service/app/notation/algebraic.py

from app.models import Position, Move, GameState, BoardType

def position_to_algebraic(pos: Position, board_type: BoardType) -> str:
    """Convert Position to algebraic notation."""
    ...

def algebraic_to_position(notation: str, board_type: BoardType) -> Position:
    """Parse algebraic notation to Position."""
    ...

def move_to_algebraic(move: Move, board_type: BoardType) -> str:
    """Convert Move to algebraic notation."""
    ...

def algebraic_to_move(notation: str, state: GameState) -> Move:
    """Parse algebraic notation to Move in context of game state."""
    ...

def game_to_pgn(moves: List[Move], metadata: dict) -> str:
    """Generate PGN-style game record."""
    ...
```

---

## 6. Relationship to Move Model

### 6.1 Move Type Mapping

| Notation Code | MoveType Enum                | Internal Identifier          |
| ------------- | ---------------------------- | ---------------------------- |
| `P`           | `place_ring`                 | `PLACE_RING`                 |
| `SP`          | `skip_placement`             | `SKIP_PLACEMENT`             |
| `SW`          | `swap_sides`                 | `SWAP_SIDES`                 |
| `M`           | `move_stack`                 | `MOVE_STACK`                 |
| `MR`          | `move_ring`                  | `MOVE_RING`                  |
| `C`           | `overtaking_capture`         | `OVERTAKING_CAPTURE`         |
| `CC`          | `continue_capture_segment`   | `CONTINUE_CAPTURE_SEGMENT`   |
| `EC`          | `end_capture_chain`          | `END_CAPTURE_CHAIN`          |
| `L`           | `process_line`               | `PROCESS_LINE`               |
| `LR`          | `choose_line_reward`         | `CHOOSE_LINE_REWARD`         |
| `T`           | `process_territory_region`   | `PROCESS_TERRITORY_REGION`   |
| `E`           | `eliminate_rings_from_stack` | `ELIMINATE_RINGS_FROM_STACK` |
| `B`           | `build_stack`                | `BUILD_STACK`                |

### 6.2 Canonical Move Structure

```typescript
interface Move {
  id: string;
  type: MoveType;
  player: number;
  from?: Position; // Origin (for movements/captures)
  to: Position; // Destination or target
  captureTarget?: Position; // For captures
  moveNumber: number;
  timestamp: Date;
  thinkTime?: number;

  // Notation-derived
  algebraicNotation?: string; // e.g., "M d4-e5"
}
```

---

## 7. Version History

| Version | Date       | Changes               |
| ------- | ---------- | --------------------- |
| 1.0     | 2024-12-01 | Initial specification |

---

## Appendix A: Quick Reference Card

### Square Board Positions

```
a1 = (0,0)  bottom-left
h8 = (7,7)  top-right (8×8)
s19 = (18,18)  top-right (19×19)
```

### Hex Board Positions

```
0.0 = center
1.0 = east of center
-1.1 = southwest
```

### Action Codes

```
P   = Place ring
M   = Move stack
C   = Capture (start)
CC  = Capture (continue)
L   = Line decision
LR  = Line reward choice
T   = Territory process
E   = Eliminate rings
```

### Move Format

```
<code> <from>-<to> [x<captured>] [+] [!]

Examples:
  P d4           Place at d4
  M d4-e5        Move from d4 to e5
  C a1-c1 xb1    Capture over b1
  L 1; LR 2      Process line 1, choose option 2
```
