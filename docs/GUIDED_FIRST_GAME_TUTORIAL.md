# Guided First Game Tutorial

This tutorial walks you through your first 5 moves in RingRift, teaching core mechanics step by step.

## Prerequisites

- Open the sandbox at [ringrift.ai](https://ringrift.ai) or run locally
- Select **8×8 board** with **2 players**
- Start with an empty board

---

## Move 1: Ring Placement

**Goal:** Place your first stack on the board

### What to Do

1. Click any empty cell
2. Place 2-3 rings of your color

### What You Learn

- Rings start in your "hand" (18 total on 8×8)
- Place 1-3 rings on empty cells, or 1 ring on existing stacks
- **Rule:** You can't place if it would leave the stack with no legal moves

```
Place near the center for flexibility:

      1   2   3   4   5   6   7   8
    ┌───┬───┬───┬───┬───┬───┬───┬───┐
  4 │   │   │   │ R │   │   │   │   │  ← Place 2 rings at d4
    │   │   │   │ R │   │   │   │   │
    └───┴───┴───┴───┴───┴───┴───┴───┘
```

---

## Move 2: Movement & Markers

**Goal:** Move your stack and observe the marker

### What to Do

1. Wait for opponent's placement (or play both sides in sandbox)
2. Click your stack, then click a destination
3. Move at least 2 cells (your stack height)

### What You Learn

- Movement distance must be ≥ stack height
- A **marker** of your color is left at the starting cell
- Markers persist and will form lines later

```
Move from d4 to d6 (2 cells, matches stack height of 2):

BEFORE                         AFTER
      3   4   5                      3   4   5
    ┌───┬───┬───┐                  ┌───┬───┬───┐
  4 │   │ R │   │       ──►      4 │   │ ● │   │  ← marker left behind
    │   │ R │   │                  │   │   │   │
    ├───┼───┼───┤                  ├───┼───┼───┤
  5 │   │   │   │                5 │   │   │   │
    ├───┼───┼───┤                  ├───┼───┼───┤
  6 │   │   │   │                6 │   │ R │   │  ← stack moved here
    │   │   │   │                  │   │ R │   │
    └───┴───┴───┘                  └───┴───┴───┘
```

---

## Move 3: Building Toward a Line

**Goal:** Continue moving to create a row of markers

### What to Do

1. Move your stack again in the same direction
2. Leave another marker, building toward 4 in a row

### What You Learn

- Strategic movement builds marker patterns
- Planning ahead for line completion
- Each move extends your marker trail

```
Move from d6 to d8:

      3   4   5
    ┌───┬───┬───┐
  4 │   │ ● │   │  ← first marker
    ├───┼───┼───┤
  5 │   │   │   │
    ├───┼───┼───┤
  6 │   │ ● │   │  ← second marker (new!)
    ├───┼───┼───┤
  7 │   │   │   │
    ├───┼───┼───┤
  8 │   │ R │   │  ← stack now here
    │   │ R │   │
    └───┴───┴───┘

You now have 2 markers in column 4!
```

---

## Move 4: First Capture

**Goal:** Capture an opponent's ring

### Setup

Position your stack adjacent to an opponent's stack with empty space beyond.

### What to Do

1. Click your stack
2. Click the cell **beyond** the opponent's stack
3. Watch your stack jump over and capture

### What You Learn

- Captures require jumping over adjacent enemy stacks
- Captured ring goes to **bottom** of your stack
- Your stack is now taller (and can move farther)

```
Your 2-stack at e5, opponent 1-stack at e6:

BEFORE                         AFTER
      4   5   6                      4   5   6
    ┌───┬───┬───┐                  ┌───┬───┬───┐
  5 │   │ R │   │       ──►      5 │   │ ● │   │  ← marker
    │   │ R │   │                  │   │   │   │
    ├───┼───┼───┤                  ├───┼───┼───┤
  6 │   │ B │   │ ← enemy        6 │   │   │   │  ← enemy stack shrunk
    ├───┼───┼───┤                  ├───┼───┼───┤
  7 │   │   │   │                7 │   │ R │   │  ← your stack landed
    │   │   │   │                  │   │ R │   │
    │   │   │   │                  │   │ B │   │  ← captured ring at bottom!
    └───┴───┴───┘                  └───┴───┴───┘
```

---

## Move 5: Line Completion

**Goal:** Complete a line of 4 markers and claim territory

### Setup

Maneuver to complete a row/column/diagonal of 4+ markers

### What to Do

1. Move your stack to leave the 4th marker in a row
2. **Line processing triggers automatically**
3. Choose elimination target (any stack you control)
4. Watch markers collapse to territory

### What You Learn

- 4+ markers in a row = line (on 8×8 2-player)
- Lines collapse to permanent territory
- **Cost:** You must eliminate 1 ring from any controlled stack

```
Complete the line in column 4:

BEFORE (3 markers)              AFTER (line collapses!)
      3   4   5                      3   4   5
    ┌───┬───┬───┐                  ┌───┬───┬───┐
  4 │   │ ● │   │                4 │   │ ▓ │   │  ← territory!
    ├───┼───┼───┤                  ├───┼───┼───┤
  5 │   │ ● │   │     ──►        5 │   │ ▓ │   │  ← territory!
    ├───┼───┼───┤                  ├───┼───┼───┤
  6 │   │ ● │   │                6 │   │ ▓ │   │  ← territory!
    ├───┼───┼───┤                  ├───┼───┼───┤
  7 │   │ R │   │ ← move to 4    7 │   │ ▓ │   │  ← territory!
    │   │ R │   │                  ├───┼───┼───┤
    ├───┼───┼───┤                8 │   │ R │   │  ← stack moved here
  8 │   │   │   │                  │   │   │   │    (lost 1 ring as cost)
    └───┴───┴───┘                  └───┴───┴───┘
```

---

## You've Learned the Core Game!

### Mechanics Covered

1. **Ring Placement** - Building your initial position
2. **Movement** - Moving stacks, leaving markers
3. **Markers** - Building patterns for lines
4. **Captures** - Overtaking opponent stacks
5. **Lines & Territory** - Converting markers to permanent control

### What's Next?

| Concept                | When It Happens                                 |
| ---------------------- | ----------------------------------------------- |
| **Chain Captures**     | When multiple captures are possible in sequence |
| **Territory Regions**  | When markers fully enclose an area              |
| **Forced Elimination** | When you have no legal moves                    |
| **Victory**            | Reach ring elimination or territory threshold   |

---

## Practice Scenarios

Load these from the sandbox scenario menu:

1. **"First Capture Tutorial"** - Practice basic captures
2. **"Chain Capture Tutorial"** - Learn mandatory chain mechanics
3. **"Line Completion Tutorial"** - Practice line formation
4. **"Territory Encirclement"** - Learn region claiming

---

## Quick Reference

| Term         | Meaning                             |
| ------------ | ----------------------------------- |
| Stack height | Number of rings in the stack        |
| Cap height   | Consecutive top rings of same color |
| Marker       | Dot left when stack moves           |
| Line         | 4+ markers in a row (8×8 2p)        |
| Territory    | Collapsed permanent space           |

**Ready for a real game?** [Play at ringrift.ai](https://ringrift.ai)
