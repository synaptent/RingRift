# Marker Interactions Visual Guide

This document provides visual explanations of how markers work in RingRift.

## 1. Marker Creation (Movement)

When a stack moves, it leaves a marker of the moving player's color behind.

```
BEFORE: Stack at A moves to B          AFTER: Marker left at A

     A       B       C                      A       B       C
   ┌───┐   ┌───┐   ┌───┐                ┌───┐   ┌───┐   ┌───┐
   │ R │   │   │   │   │     ──►        │ ● │   │ R │   │   │
   │ R │   │   │   │   │                │   │   │ R │   │   │
   │ B │   │   │   │   │                │   │   │ B │   │   │
   └───┘   └───┘   └───┘                └───┘   └───┘   └───┘

   R = Red ring, B = Blue ring, ● = Red marker
```

**Key Points:**

- The marker color matches the moving player (controller of the stack)
- Markers persist until they form lines or are displaced

---

## 2. Landing on a Marker

When a stack lands on a marker (any color), two things happen:

1. The marker is removed from the board
2. The top ring of the landing stack is eliminated (credited to the moving player)

```
BEFORE: Red stack moves to space       AFTER: Marker removed, top ring eliminated
        with Blue marker

     A       B                              A       B
   ┌───┐   ┌───┐                        ┌───┐   ┌───┐
   │ R │   │ ○ │     ──►                │ ● │   │ R │
   │ R │   │   │     (land)             │   │   │ B │
   │ B │   │   │                        │   │   │   │
   └───┘   └───┘                        └───┘   └───┘

   ○ = Blue marker, ● = Red marker
   Red eliminated one ring (their own top R), marker at B removed
```

**Important:** Landing on your OWN marker also eliminates your top ring!

---

## 3. Line Formation

Lines are formed from consecutive markers of the same color along a direction.

```
HORIZONTAL LINE (4+ markers = line on most boards)

     1     2     3     4     5
   ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐
   │ ● │─│ ● │─│ ● │─│ ● │ │   │  ◄── 4 Red markers = valid line!
   └───┘ └───┘ └───┘ └───┘ └───┘

DIAGONAL LINE

     1     2     3     4
   ┌───┐                         ● = marker
   │ ● │\
   └───┘ ┌───┐                   4 markers diagonally
         │ ● │\                  = valid line
         └───┘ ┌───┐
               │ ● │\
               └───┘ ┌───┐
                     │ ● │
                     └───┘
```

**Line Requirements by Board:**
| Board | 2-Player | 3-4 Player |
|-------|----------|------------|
| 8×8 | 4 | 3 |
| Hex8 | 4 | 3 |
| 19×19 | 4 | 4 |
| Hex | 4 | 4 |

---

## 4. Line Processing

When a line forms, it collapses into territory.

```
EXACT LENGTH LINE (4 markers on 8x8 2p)

BEFORE: Line detected                  AFTER: Collapsed to territory

   ┌───┐ ┌───┐ ┌───┐ ┌───┐            ┌───┐ ┌───┐ ┌───┐ ┌───┐
   │ ● │ │ ● │ │ ● │ │ ● │    ──►     │ ▓ │ │ ▓ │ │ ▓ │ │ ▓ │
   └───┘ └───┘ └───┘ └───┘            └───┘ └───┘ └───┘ └───┘

   ● = Red marker
   ▓ = Red collapsed territory

   COST: Player MUST eliminate 1 ring from any controlled stack (unless you control no stacks)
```

---

## 5. Overlength Lines (Options)

Lines longer than minimum give player a choice:

```
OVERLENGTH LINE (5 markers when 4 required)

   ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐
   │ ● │ │ ● │ │ ● │ │ ● │ │ ● │   5 markers!
   └───┘ └───┘ └───┘ └───┘ └───┘

OPTION 1: Collapse ALL (more territory, 1 ring cost)
   ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐
   │ ▓ │ │ ▓ │ │ ▓ │ │ ▓ │ │ ▓ │  + eliminate 1 ring
   └───┘ └───┘ └───┘ └───┘ └───┘

OPTION 2: Collapse MINIMUM (less territory, no cost)
   ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐
   │ ● │ │ ▓ │ │ ▓ │ │ ▓ │ │ ▓ │  + no ring cost
   └───┘ └───┘ └───┘ └───┘ └───┘
                     ↑ player chooses which 4 to collapse
```

---

## 6. Territory Formation from Markers

Markers act as walls that can disconnect regions:

```
ENCIRCLEMENT PATTERN

     1     2     3     4     5
   ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐
 1 │   │ │ ● │ │ ● │ │ ● │ │   │
   └───┘ └───┘ └───┘ └───┘ └───┘
   ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐
 2 │ ● │ │ B │ │   │ │ R │ │ ● │  ← B, R are stacks inside
   └───┘ └───┘ └───┘ └───┘ └───┘
   ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐
 3 │   │ │ ● │ │ ● │ │ ● │ │   │
   └───┘ └───┘ └───┘ └───┘ └───┘

When Red's markers (●) form complete border:
- If Blue (B) has no stacks INSIDE, Red can claim the region
- All stacks inside are eliminated
- Region collapses to Red territory
- Red must sacrifice one cap from a stack OUTSIDE
```

---

## 7. Marker vs Territory vs Stack

Quick reference for cell contents:

```
CELL STATES:

   ┌───┐     ┌───┐     ┌───┐     ┌───┐
   │   │     │ ● │     │ ▓ │     │ R │
   └───┘     └───┘     └───┘     │ B │
                                 └───┘
   Empty    Marker   Territory   Stack

   - Stacks can MOVE through empty cells and markers
   - Stacks CANNOT move through/onto territory or other stacks
   - Landing on marker: marker removed, top ring eliminated
   - Territory: permanent, blocks all movement
```

---

## 8. Marker Interaction Summary

| Action                       | Result                                                                                                            |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Move stack                   | Leave marker at origin                                                                                            |
| Land on marker (any color)   | Remove marker, eliminate top ring                                                                                 |
| Form line of markers         | Line processing (collapse options)                                                                                |
| Surround region with markers | Territory processing (if the region lacks at least one active color; in 2p, that means no opponent stacks inside) |
| Move through marker          | Markers don't block movement; landing on one removes it and eliminates your top ring.                             |

---

## 9. Complete Movement Example

```
BEFORE: Red to move

     A       B       C       D       E
   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐
   │ R │   │ ● │   │ ● │   │ ● │   │   │  row 1
   │ R │   │   │   │   │   │   │   │   │
   └───┘   └───┘   └───┘   └───┘   └───┘

Red (2-stack at A) wants to form a line...

OPTION A: Move to E (4 spaces, legal for height-2 stack)
Result: Marker at A, land on E (empty), now have 4-marker line!

     A       B       C       D       E
   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐
   │ ● │   │ ● │   │ ● │   │ ● │   │ R │  ← 4 markers = LINE!
   │   │   │   │   │   │   │   │   │ R │
   └───┘   └───┘   └───┘   └───┘   └───┘

OPTION B: Move to D (land on marker)
Result: Marker at A, marker at D removed, RED LOSES TOP RING

     A       B       C       D       E
   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐
   │ ● │   │ ● │   │ ● │   │ R │   │   │  ← only 3 markers (no line)
   │   │   │   │   │   │   │   │   │   │    top R eliminated!
   └───┘   └───┘   └───┘   └───┘   └───┘
```

---

## See Also

- [../rules/HUMAN_RULES.md](../rules/HUMAN_RULES.md) - Full human rules
- [RULES_CANONICAL_SPEC.md](../../RULES_CANONICAL_SPEC.md) - Formal specification
