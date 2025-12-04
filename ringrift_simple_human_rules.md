# RingRift – Simple Human Rules (Canonical Summary)

> This document is a human-oriented summary of the current RingRift rules.
> When there is any doubt, the **single sources of truth** are:
>
> - `ringrift_complete_rules.md` – the full rulebook.
> - `RULES_CANONICAL_SPEC.md` – the formal, parameterised spec (RR‑CANON).
>
> This file keeps language informal and explanatory while following those
> documents closely. When older prose elsewhere disagrees (for example, about
> exact line lengths on 8×8), treat **this document and RR‑CANON** as
> authoritative.

---

## 1. What Kind of Game Is RingRift?

RingRift is a **perfect‑information, no‑randomness** strategy game for **2–4
players**. Each player controls rings of their own colour on a shared board.
During the game you:

- Build and move **stacks** of rings.
- Leave **markers** behind as you move.
- Form **lines** of markers and collapse them into permanent **territory**.
- Cause large **regions** of the board to disconnect and collapse.
- Sometimes must **sacrifice your own rings** (forced elimination) to keep
  the game moving.

You win by achieving **any one** of three victory conditions:

1. Eliminating more than half of all rings in the game.
2. Controlling more than half of all board spaces as your territory.
3. Becoming the **Last Player Standing** in terms of “real actions”.

If nobody can move at all (including forced eliminations), the game ends in a
structured **stalemate** and tiebreakers determine the winner.

---

## 2. Boards, Pieces, and Basic Concepts

### 2.1 Board types

There are three standard boards:

- **8×8 square** (simplified version)
  - 64 spaces.
  - Each player has **18 rings**.
  - Territory victory: **>32** collapsed spaces.
  - Lines:
    - Geometrically, any 3+ markers in a row matter.
    - For **2‑player games**, the _threshold_ for line processing is
      **4 in a row** (see §5.1).
    - For **3–4 player games**, lines of **3+** markers are already enough to
      trigger line processing.

- **19×19 square** (full version)
  - 361 spaces.
  - **36 rings** per player.
  - Territory victory: **>180** spaces.
  - Lines: **4+** markers in a row.

- **Hexagonal** (radius 10; 11 cells per side)
  - 331 spaces.
  - **36 rings** per player.
  - Territory victory: **>165** spaces.
  - Lines: **4+** markers along hex directions.

These parameters (size, rings per player, line length, adjacency) come from
RR‑CANON‑R001 and are the base for all variants.

### 2.2 Stacks, caps, markers, and territory

- **Stack**
  - One or more rings in the same cell.
  - Rings are ordered from top to bottom.

- **Controlling player**
  - The colour of the **top ring** in a stack.
  - Only the controlling player may move or capture with that stack.

- **Cap**
  - The **cap height** is the number of consecutive rings, from the top
    downward, that match the controlling player’s colour.
  - Example: Blue, Blue, Blue, Red → cap height for Blue is 3.

- **Marker**
  - When you move a stack, you leave a **marker** of your colour on the
    starting space.
  - Markers help form **lines** and also act as part of “walls” for territory
    detection.

- **Collapsed space (territory)**
  - When a line or region is processed, some spaces become permanently
    collapsed territory for a player.
  - Collapsed spaces:
    - Never revert.
    - Block movement and capture.
    - Count toward the **territory victory** threshold.

---

## 3. Turn Structure – What Happens on Your Turn

On your turn, with `currentPlayer = P`, the rules engine walks through
phases in this fixed order:

1. **Ring Placement** (optional in most cases).
2. **Movement** (required if any legal move/capture exists).
3. **Capture / Chain Capture** (if a capture starts, you must continue
   while legal segments exist).
4. **Line Processing** (resolve completed lines).
5. **Territory Processing** (resolve disconnected regions).
6. **Victory / Termination Check**.

### 3.1 Ring placement

- If you have rings in hand, you may place:
  - **1–3 rings** on an **empty** space, or
  - **1 ring** onto an existing stack.
- Placement is limited by:
  - Your **own‑colour ring supply** (`ringsPerPlayer` cap); this is a cap on
    your own rings that can be in play (on board + in hand).
  - Rings you still hold in hand.
  - Local per‑spot limits (max 3 onto empties, 1 onto stacks).
- **No‑dead‑placement rule**:
  - You may not place a stack that will have **no legal movement or capture**
    immediately after placement.
  - This ensures newly placed stacks will always have something they could
    do on a future movement/capture phase.

### 3.2 Movement

After (or instead of) placing, you **move** a stack you control:

- Choose a stack where the top ring is your colour.
- Move it in a straight line:
  - Square: 8 directions (orthogonal + diagonal).
  - Hex: 6 hex directions.
- The **distance travelled**:
  - Must be at least the stack’s total height.
- You cannot:
  - Move through other stacks or collapsed spaces.
  - Land on collapsed spaces.
- **Landing on markers**:
  - You may land on any marker (yours or an opponent's).
  - When you do, the marker is removed and the top ring of your stack's
    cap is immediately eliminated (credited to you).
- As you leave a space, you always leave a **marker** of your colour
  behind.

### 3.3 Overtaking captures and capture chains

If your move jumps over another stack and lands beyond it on a legal space,
that is an **overtaking capture**:

- You take one ring from the top of the overtaken stack, (or the ring if it is a single ring).
- You add this overtaken ring to the **bottom** of your overtaking stack.
- It stays in play but now sits at the bottom of your overtaking stack.

**Capture chains**:

- Once you perform an overtaking capture, you enter a capture chain phase.
- While there is **any legal capture segment** from your current landing
  position, you are **required** to keep capturing.
- However, when there are multiple possible capture directions, you may
  choose which legal segment to take next.
- You may:
  - Change direction;
  - Jump 180° back over previously visited stacks;
  - Capture multiple times from the same stack (as it changes each time).

The chain ends only when **no further legal capture segment** is available.

### 3.4 Forced elimination when blocked

At the **start of your turn**, before placement or movement:

- If you:
  - Control at least one stack on the board, **and**
  - Have **no legal placements**, **no legal movements**, and **no legal
    overtaking captures** anywhere,
- Then you are **blocked** and must perform a **forced elimination**:
  - You must eliminate the **entire cap** of **one** stack you control.
  - All those rings are removed from play and added to your **eliminated
    rings** total.
  - This is treated as a legal action and counts toward your ring‑elimination
    progress.

Forced elimination ensures that as long as any stacks exist on the board,
**someone** always has a legal action: either movement/capture, placement, or
forced elimination.

---

## 4. Post‑Movement Processing: Lines and Territory

Once placement, movement, and any capture chains are finished, you then
handle the following **global effects** in this strict order:

1. Resolve **lines** (marker formations).
2. Resolve **territory / disconnected regions**.
3. Check victory / termination.

### 4.1 Lines – forming and collapsing lines of markers

A **line** is a straight contiguous run of your **markers** of at least
a certain length for the board:

- 8×8:
  - In **2‑player** games, you need to make 4 in a row.
  - In **3–4 player** games, 3‑in‑a‑row is enough.
- 19×19 and Hex: you need to make 4 in a row.

For each line that is eligible, you process it one at a time in an order
**you choose**:

1. **Lines of exactly required length**:
   - Replace all markers in that line with **collapsed spaces** of your
     colour.
   - You must **eliminate**:
     - Either a single ring or an entire cap from a stack you control.

2. **Overlength lines** :
   - If you form a line longer than the minimum required length, you choose between:
     - **Option 1 (full reward)**:
       - Collapse the **entire line** to your territory.
       - Eliminate a ring or cap you control (as above).
     - **Option 2 (minimum collapse)**:
       - Collapse **exactly lineLength** consecutive markers of your choice
         within the line to your territory.
       - **No rings are eliminated**.

After each line is processed, lines may still remain; you repeat until there are no more eligible lines.

### 4.2 Territory – disconnected regions

After lines, inspect the board for **disconnected regions**:

- Intuitively: areas cut off from other areas by markers of a specific color and collapsed spaces so
  that they cannot be reached by standard adjacency from outside.
- For square boards, the inside cannot be reached by moves from orthogonally adjacent spaces without crossing over markers of a particular color or collapsed spaces, because it is fully separated from other areas by a complete border of markers of a particular color plus collapsed spaces.
- For hex boards, the inside cannot be reached by moves from hex adjacent spaces without crossing over markers of a particular color or collapsed spaces, because it is fully separated from other areas by a complete border of markers of a particular color plus collapsed spaces.
- The border can be formed by markers of any particular color plus collapsed spaces, but all the markers in the border must be of the same color.
- There can be multiple disconnected regions that can exist simultaneously, and depending on which color marker is helping form the border, they can partially overlap.

You can only process a disconencted region if you can afford self elimination for it. When you process a disconnected region:

1. **Eliminate all stacks inside the region**:
   - Every ring in those stacks, for all colours, is removed from the board.
   - These eliminations are credited to the **acting player** (the one whose
     turn it is).

2. **Collapse the region**:
   - All spaces inside become **collapsed territory** for the acting player.

3. **Mandatory self‑elimination**:
   - You must still have at least one ring or stack **outside** the processed region under your control, and you must eliminate **one standalone ring**, or **one cap** from some stack you control outside the processed region.
   - These rings are also credited to you as eliminated rings counting toward victory.

4. **Repeat**:
   - New disconnected regions may appear after each collapse.
   - You keep processing them, one at a time, until none remain.

Territory processing is often where massive swings in eliminated‑ring totals
and territory counts come from.

---

## 5. Victory Conditions – How You Win

RingRift has three main victory conditions plus a stalemate resolution. In
normal play, exactly one of these will apply first.

### 5.1 Ring Elimination Victory

You win by eliminating **more than 50%** of all rings that were in play for
the chosen board and player count.

Thresholds (from the rulebook):

- 8×8:
  - 2 players: >18 rings eliminated.
  - 3 players: >27.
  - 4 players: >36.
- 19×19 / Hex:
  - 2 players: >36.
  - 3 players: >54.
  - 4 players: >72.

Your eliminated‑rings total includes:

- Rings you eliminate from:
  - Line rewards,
  - Territory self‑elimination,
  - Forced elimination when blocked,
  - Collapsing disconnected regions (all rings inside are credited to you).

It does **not** include:

- Rings still on the board (even if buried in your stacks),
- Rings captured by overtaking but not eliminated.

Because the threshold is strictly >50%, **no two players can meet this
condition simultaneously**.

### 5.2 Territory Victory

You win by controlling **more than 50%** of all board spaces as your own
collapsed territory.

Thresholds:

- 8×8: >32 collapsed spaces.
- 19×19: >180.
- Hex: >165.

Again, because each threshold is more than half the board, only one player
can satisfy this at a time.

### 5.3 Last Player Standing (LPS)

Last Player Standing is a third victory condition that depends on
**who can still really act** near the end of the game.

#### 5.3.1 Real actions (for LPS)

On your own turn, you have a **real action** if, at the start of your action,
you have at least one of:

- A legal ring **placement**.
- A legal non‑capture **movement**.
- A legal **overtaking capture**.

Having **only forced elimination** available **does not** count as having a
real action for LPS purposes.

#### 5.3.2 Formal LPS condition

Player P wins by Last Player Standing if all of the following hold:

1. Over **at least one complete round** of turns (each non‑eliminated player
   taking exactly one turn in order):
   - On each of P’s turns in that round, P has **at least one real action**
     available at the start of their turn.
   - On every other player’s turns that round, those players have **no real
     actions** at the start of their turn (they may have only forced
     eliminations or no actions at all).

2. Immediately after that round completes and **P’s next turn begins**:
   - P is **still the only player** who has any real action available.

If another player regains a real action (for example by gaining control of a
stack) before the condition is fully satisfied, the LPS “candidate” reset
and must be re‑established over a fresh round.

### 5.4 Global stalemate and tiebreakers

If no victory by rings, territory, or LPS has triggered, but the game reaches
a state where **no one can act at all**, it ends in a **global stalemate**
and tiebreakers pick the winner.

Global stalemate occurs when:

- For every player:
  - No legal placements,
  - No legal non‑capture moves,
  - No legal captures,
  - No legal forced elimination.
- In practice this means:
  - There are no stacks left on the board, and
  - There are no legal placements (for example because ring supplies are
    exhausted or placements would be illegal everywhere).

When this happens:

1. **Rings in hand** are converted into **eliminated rings** for their
   owners.

2. The **winner** is determined by the following ordered tiebreakers:
   1. Most collapsed spaces (territory).
   2. If tied: most eliminated rings (including hand‑to‑eliminated
      conversion).
   3. If still tied: most markers on the board.
   4. If still tied: last player to complete a valid turn action.

This ensures that even in a stalemate, there is a **single, well‑defined
winner**.

---

## 6. Progress and Guaranteed Termination (Intuition Only)

Internally, the rules are justified by a simple monotone progress measure:

- Let **M** = number of markers on the board.
- Let **C** = number of collapsed spaces.
- Let **E** = total eliminated rings.
- Define `S = M + C + E`.

Key facts:

- Moves and capture chains always either:
  - Increase markers, or
  - Increase eliminated rings, or both.
- Collapses and territory keep `M + C` non‑decreasing.
- Eliminations never decrease.
- Forced elimination always increases `E`.

The total possible amount of progress is bounded by:

`S ≤ 2·(boardSpaces) + (totalRings)`.

So there can only be **finitely many** real actions in any game. Eventually:

- Someone hits a ring‑elimination or territory threshold, or
- The LPS condition is satisfied, or
- No one can act at all and the stalemate tiebreakers apply.

In all cases, the game ends in **finite time** with a single winner.t
