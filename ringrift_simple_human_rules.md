# RingRift – Simple Human Rules (Canonical Summary)

> **Version:** 1.2 | **Last Updated:** 2025-12-17 | **Spec Alignment:** RR-CANON v1.0 (Territory eligibility per RR-CANON-R022/R145)
>
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

Quick canonical reminders:

- On **8×8**, required line length is **4 for 2-player** and **3 for 3–4 player** games.
- Movement and capture landings may be on empty spaces or markers of **any** colour; if you land on a marker, remove it and eliminate the top ring of the landing stack.
- In digital/engine play, forced steps are still explicit recorded moves (no silent “automatic” transitions).

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

1. Eliminating enough rings to reach the Ring Elimination victory threshold (depends on player count).
2. Controlling enough territory to dominate the board (at least floor(totalSpaces/numPlayers)+1 AND more than all opponents combined).
3. Becoming the **Last Player Standing** in terms of "real actions".

If nobody can move at all (including forced eliminations), the game ends in a
structured **stalemate** and tiebreakers determine the winner.

---

## 2. Boards, Pieces, and Basic Concepts

### 2.1 Board types

There are three standard boards:

- **8×8 square** (simplified version)
  - 64 spaces.
  - Each player has **18 rings**.
  - Territory victory: minimum **33** (2p), **22** (3p), **17** (4p) AND more than opponents combined.
  - Lines: **4+** markers (2-player) or **3+** markers (3–4 player).

- **19×19 square** (full version)
  - 361 spaces.
  - **72 rings** per player.
  - Territory victory: minimum **181** (2p), **121** (3p), **91** (4p) AND more than opponents combined.
  - Lines: **4+** markers in a row.

- **Hexagonal** (radius 12; 13 cells per side)
  - 469 spaces.
  - **96 rings** per player.
  - Territory victory: minimum **235** (2p), **157** (3p), **118** (4p) AND more than opponents combined.
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
    downward, that match the controlling player's colour.
  - Example: Blue, Blue, Blue, Red → cap height for Blue is 3.
  - **Elimination costs vary by context:**
    - **Line processing:** Eliminate **one ring** from the top of any stack you control (including standalone rings).
    - **Territory processing:** Eliminate your **entire cap** from any stack you control—**including standalone rings (height 1)**. All controlled stacks are eligible per RR-CANON-R022/R145.
    - **Forced elimination:** Eliminate your **entire cap** from **any** stack you control—**including standalone rings**.

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
3. **Capture / Chain Capture** (optional to start from landing position only;
   if a capture starts, you must continue while legal segments exist).
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

### 3.5 Recovery action

> **Key Mechanic:** Recovery actions let you stay in the game even after losing all your stacks! Keep some of your rings buried in opponents' stacks as "survival insurance."

If you control **no stacks** but have **markers on the board** and **buried rings**
(your rings at the bottom of opponents' stacks), you may perform a **recovery action**.
Recovery eligibility is independent of rings in hand; even if you have rings in hand,
you may skip placement and take recovery instead:

1. **Slide** one of your markers to an adjacent destination:
   - **Empty cell** (normal recovery slide), or
   - **Adjacent stack** (stack-strike, only in fallback-class recovery).
2. The slide is legal if **either**:
   - **(a) Line formation:** Completes a line of **at least** `lineLength` of your markers (4 for 8×8 2-player, 3 for 8×8 3–4 player, 4 for 19×19/Hex).
   - **(b) Fallback-class recovery:** If no line-forming slide exists, one of the following adjacent recovery actions is permitted:
     - **(b1) Fallback repositioning:** Slide to an adjacent empty cell (including slides that cause territory disconnection).
     - **(b2) Stack-strike:** Slide onto an adjacent stack; your marker is removed and the stack's top ring is eliminated and credited to you.
3. **Skip option:** You may skip recovery entirely to save your buried rings.
4. **Line recovery (a):** Overlength lines follow standard Option 1 / Option 2 semantics:
   - **Option 1:** Collapse all markers; pay one buried ring extraction.
   - **Option 2:** Collapse exactly `lineLength` markers of your choice; pay nothing.
5. **Fallback-class recovery (b):** Costs one buried ring extraction and does not trigger line processing. In stack-strike (b2), the marker is sacrificed and the attacked stack's top ring is also eliminated.
6. **Pay the cost** by extracting your bottommost ring from any stack containing your buried rings:
   - That ring is permanently eliminated (credited to you).
   - The stack shrinks by 1; its new top ring determines control.
7. **Territory disconnection:** If your recovery slide (line or fallback) results in
   claimable **disconnected regions**, you may claim them by extracting additional buried rings
   (one per region, from stacks outside each region).

Recovery actions allow temporarily eliminated players to strike back from
apparent defeat, turning buried rings into a strategic resource.

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

- 8×8: **4+** markers (2-player) or **3+** markers (3–4 player).
- 19×19 and Hex: **4+** markers in a row.

For each line that is eligible, you process it one at a time in an order
**you choose**:

1. **Lines of exactly required length**:
   - Replace all markers in that line with **collapsed spaces** of your
     colour.
   - You must **eliminate one ring** from the top of any stack you control (including height-1 standalone rings). Any controlled stack is an eligible target.
   - **Exception for recovery actions:** When a line is formed via recovery, you pay one buried ring extraction instead.

2. **Overlength lines** :
   - If you form a line longer than the minimum required length, you choose between:
     - **Option 1 (full reward)**:
       - Collapse the **entire line** to your territory.
       - Eliminate **one ring** from the top of any stack you control (same rules as exact-length lines).
       - **Exception for recovery actions:** When a line is formed via recovery, you pay one buried ring extraction instead.
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

You can only process a disconnected region if you can afford self elimination for it. A region is eligible for processing when it lacks representation from at least one active player (meaning at least one player with rings on the board has no stack in that region). **Empty regions** (containing no stacks at all, only empty cells and/or markers) automatically lack representation from all players and are thus fully eligible for processing.

When you process a disconnected region:

1. **Eliminate all stacks inside the region**:
   - Every ring in those stacks, for all colours, is removed from the board.
   - These eliminations are credited to the **acting player** (the one whose
     turn it is).
   - For empty regions (no stacks), this step eliminates zero rings, but processing is still valid.

2. **Collapse the region**:
   - All spaces inside become **collapsed territory** for the acting player.

3. **Mandatory self‑elimination**:
   - You must still have at least one stack cap **outside** the processed region under your control, and you must eliminate the **entire cap** (all consecutive top rings of your colour) from one of your controlled stacks outside the processed region.
   - **All controlled stacks outside the region are eligible** for cap elimination, including:
     - **(a) Mixed-colour stacks** with rings of other colours buried beneath your cap,
     - **(b) Single-colour stacks of height > 1** consisting entirely of your rings, AND
     - **(c) Height-1 standalone rings** (your single ring is the entire cap and is eliminated).
   - This is consistent with RR-CANON-R022/R145: any controlled stack is an eligible target.
   - **Exception for recovery actions:** When territory processing is triggered by a recovery action, the self-elimination cost is one buried ring extraction per region, not an entire cap.
   - These rings are also credited to you as eliminated rings counting toward victory.

4. **Repeat**:
   - New disconnected regions may appear after each collapse.
   - You may process **any subset** of eligible regions, one at a time, in any order you choose. You can stop early; any remaining eligible regions stay on the board.

Territory processing is often where massive swings in eliminated‑ring totals
and territory counts come from.

---

## 5. Victory Conditions – How You Win

RingRift has three main victory conditions plus a stalemate resolution. In
normal play, exactly one of these will apply first.

### 5.1 Ring Elimination Victory

You win by eliminating a number of rings equal to two thirds of your starting
rings in hand plus one third of your opponents' combined starting rings in
hand. In multi-player games, the threshold increases with player count.

Thresholds:

- 8×8 (18 rings/player): **18** (2p), **24** (3p), **30** (4p) rings eliminated.
- 19×19 (72 rings/player): **72** (2p), **96** (3p), **120** (4p) rings eliminated.
- Hex (96 rings/player): **96** (2p), **128** (3p), **160** (4p) rings eliminated.

Your eliminated‑rings total includes:

- Rings you eliminate from:
  - Line rewards,
  - Territory self‑elimination,
  - Forced elimination when blocked,
  - Collapsing disconnected regions (all rings inside are credited to you).

It does **not** include:

- Rings still on the board (even if buried in your stacks),
- Rings captured by overtaking but not eliminated.

Because only the acting player's eliminated total can increase on their turn
and the game ends immediately when the threshold is reached, **no two players
can meet this condition simultaneously**.

### 5.2 Territory Victory

You win by satisfying **both** conditions:

1. **Minimum threshold:** Your territory >= floor(totalSpaces / numPlayers) + 1
2. **Dominance:** Your territory > all opponents' territories combined

Minimum thresholds by board and player count:

- 8×8 (64 spaces): 33 (2p), 22 (3p), 17 (4p)
- 19×19 (361 spaces): 181 (2p), 121 (3p), 91 (4p)
- Hex (469 spaces): 235 (2p), 157 (3p), 118 (4p)

Because the dominance condition requires strictly more territory than all
opponents combined, only one player can satisfy this at a time.

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

Recovery actions also **do not** count as real actions for LPS purposes.

#### 5.3.2 Formal LPS condition

Player P wins by Last Player Standing if all of the following hold:

1. **Round 1:** Over one complete round of turns (each non‑eliminated
   player taking exactly one turn in order):
   - On P's turn in that round, P has **at least one real action** available
     and **takes at least one such action**.
   - On every other player's turns that round, those players have **no real
     actions** at the start of their turn (they may have only forced
     eliminations or no actions at all).

2. **Round 2:** After the first round completes (including all
   post-movement processing), on the following round:
   - P remains the **only player** who has taken any legal real action and
     takes at least one such action on their turn.

3. **Victory declared:** After the second round completes (including all
   post-movement processing), P is declared the winner by Last Player
   Standing. This applies regardless of relative territory or rings eliminated.

#### 5.3.3 Temporarily inactive players and recovery

Some players may still have rings on the board but be **temporarily inactive**
for LPS purposes:

- They control no stacks and have no legal placements (no rings in hand, or
  every placement would be illegal), or
- They do control stacks but have **no** legal placements, no legal non‑capture
  moves, and no legal overtaking captures, so their only possible turn action
  is forced elimination.

These players do **not** count as having real actions during their turns, but
their rings may still be buried in mixed‑colour stacks and can become active
again later.

A temporarily inactive player can return to full activity if they regain a real
action, for example:

- Gaining control of a multi‑colour stack when its top ring becomes their
  colour; or
- Reducing the height of a stack they control so that it can move again.

If any temporarily inactive player regains a real action before **both
rounds** in the LPS condition have been completed, the LPS condition is
not met and must be re‑established over fresh rounds.

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

### 5.5 Final player rankings

After the game ends, all players are ranked from 1st to Nth place.

#### 5.5.1 Temporary vs permanent elimination

During the game, players can lose the ability to act in two different ways:

- **Temporarily eliminated**: A player who has no stacks on the board, no
  rings in hand, but still has rings **buried** in mixed‑colour stacks
  controlled by others. These players may regain activity if their buried
  rings surface later.

- **Permanently eliminated**: A player with **zero rings anywhere** in the
  game—none controlled, none buried, none in hand. This is irreversible.

The distinction matters for ranking: **only permanently eliminated players**
receive an elimination rank at the turn they reach zero rings. Temporarily
eliminated players remain "in contention" until they either recover or the
game ends.

#### 5.5.2 Ranking algorithm

1. **Winner**: The player who triggered the victory condition (ring
   elimination, territory, or LPS) is ranked 1st. In stalemate, the
   tiebreaker winner is 1st.

2. **Remaining players**: All non‑winners are ranked by the following
   tiebreakers (in order):
   1. Later permanent elimination turn (or never eliminated) beats earlier.
   2. More collapsed spaces (territory).
   3. More eliminated rings credited.
   4. More markers on the board.
   5. Later last valid turn action.

Players with identical values across all tiebreakers share the same rank
(standard 1‑2‑2‑4 competition ranking).

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
