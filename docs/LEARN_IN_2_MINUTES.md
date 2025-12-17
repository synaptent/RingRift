# Learn RingRift in 2 Minutes

> **TL;DR:** Place rings, move stacks, leave markers, form lines, claim territory. First to eliminate enough rings or control enough territory wins.

---

## The 3 Core Concepts

### 1. Stacks & Control

```
     You control       Enemy controls
     ┌───┐             ┌───┐
     │ R │ ← top       │ B │ ← top
     │ R │             │ B │
     │ B │             │ R │ ← buried (yours!)
     └───┘             └───┘
```

- **Stacks** are piles of rings on one cell
- **You control** stacks where YOUR color is on top
- Move only stacks you control; buried rings can resurface later

### 2. Movement & Markers

```
BEFORE                    AFTER
  A     B     C            A     B     C
┌───┐ ┌───┐ ┌───┐       ┌───┐ ┌───┐ ┌───┐
│ R │ │   │ │   │  ──►  │ ● │ │   │ │ R │
│ R │ │   │ │   │       │   │ │   │ │ R │
└───┘ └───┘ └───┘       └───┘ └───┘ └───┘
                          ↑
                       marker left behind
```

- Move in a straight line (min distance = stack height)
- You **always leave a marker** where you started
- Land on a marker? It's removed and your **top ring is eliminated**

### 3. Lines & Territory

```
4 markers in a row = LINE!     →    Collapses to TERRITORY
┌───┐ ┌───┐ ┌───┐ ┌───┐           ┌───┐ ┌───┐ ┌───┐ ┌───┐
│ ● │ │ ● │ │ ● │ │ ● │    ──►    │ ▓ │ │ ▓ │ │ ▓ │ │ ▓ │
└───┘ └───┘ └───┘ └───┘           └───┘ └───┘ └───┘ └───┘
                                   permanent, blocks movement
```

- **4+ markers in a row** = line (collapses to territory)
- Costs: you must **eliminate 1 ring** from any stack you control
- Territory is **permanent** and counts toward victory

---

## Victory (Pick One)

| Path                  | What You Need                               |
| --------------------- | ------------------------------------------- |
| **Ring Elimination**  | Eliminate enough rings (18 on 8×8 2-player) |
| **Territory Control** | Own more territory than opponents combined  |
| **Last Standing**     | Be the only one who can still move          |

---

## Captures (Bonus Mechanic)

```
Jump over enemy stack → take their top ring

BEFORE                      AFTER
  A     B     C               A     B     C
┌───┐ ┌───┐ ┌───┐          ┌───┐ ┌───┐ ┌───┐
│ R │ │ B │ │   │   ──►    │ ● │ │ B │ │ R │
│ R │ │ B │ │   │          │   │ │   │ │ R │
└───┘ └───┘ └───┘          └───┘ └───┘ │ B │ ← captured!
                                       └───┘
```

- **Overtake** = jump over adjacent stack, land beyond
- Captured ring goes to **bottom** of your stack (stays in play)
- **Chain captures** are mandatory once started

---

## Quick Reference Card

| Term          | Meaning                                     |
| ------------- | ------------------------------------------- |
| **Stack**     | Pile of rings; top ring = controller        |
| **Marker**    | Colored dot left when you move              |
| **Territory** | Collapsed space; permanent, blocks movement |
| **Cap**       | Consecutive top rings of same color         |
| **Overtake**  | Capture by jumping over opponent stack      |

---

## Ready to Play?

🎮 **[Play Now at ringrift.ai](https://ringrift.ai)**

📖 **[Full Rules](../ringrift_simple_human_rules.md)** for complete details

🎯 **Sandbox Mode** to practice without opponents
