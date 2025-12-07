# RingRift Compact Rules (Engine / AI Implementation Spec)

**Purpose:** This document is a compact, implementation‑oriented specification of the RingRift rules. It is designed for engine/AI authors, not for teaching humans. It encodes the _minimum complete_ rule set needed to implement a correct engine for all supported versions.

- Full, narrative rules: `ringrift_complete_rules.md`
- This file: focuses on **state**, **version parameters**, and **transition rules**.
- Omitted here (see Complete Rules for these): flavour prose, extended strategy notes, FAQ-style walkthroughs, and long-form examples. When semantics and prose diverge, this file + `RULES_CANONICAL_SPEC.md` win.

---

## 1. Version Parameters & Board Model

### 1.1 Supported board types

`BoardType ∈ { square8, square19, hexagonal }`

For each board type, define a static configuration:

| BoardType | size | totalSpaces | ringsPerPlayer | lineLength | movementAdjacency | lineAdjacency | territoryAdjacency  | boardGeometry   |
| --------- | ---- | ----------- | -------------- | ---------- | ----------------- | ------------- | ------------------- | --------------- |
| square8   | 8    | 64          | 18             | 3          | Moore (8-dir)     | Moore         | Von Neumann (4-dir) | orthogonal grid |
| square19  | 19   | 361         | 36             | 4          | Moore             | Moore         | Von Neumann         | orthogonal grid |
| hexagonal | 13   | 469         | 48             | 4          | Hex (6-dir)       | Hex           | Hex                 | hex coordinates |

- **Ring supply semantics:** For each player P, `ringsPerPlayer` is the maximum number of rings of P's own colour that may ever be in play: all of P's rings currently on the board in any stack (regardless of which player controls those stacks) plus all of P's rings in hand must never exceed this value. Rings of other colours that P has captured and that are buried in stacks P controls do **not** count against P's `ringsPerPlayer` cap; they remain, by colour, part of the original owner's supply for conservation and victory accounting.
  - Quick supply check: `ringsInHand[P] + ringsOfColorOnBoard[P]` must always equal the starting supply for P's board type (18 on square8, 36 on square19, 48 on hex).

- **Coordinates**:
  - Square boards: integer `(x, y)` in `[0, size-1] × [0, size-1]`.
  - Hex board: cube coordinates `(x, y, z)` with `x + y + z = 0` and `max(|x|,|y|,|z|) ≤ radius`, where `radius = size - 1`.

### 1.2 Adjacency relations

You must implement three adjacency notions, driven by config:

- **Movement & capture directions**
  - `movementAdjacency`:
    - Moore for square8/square19: 8 directions (orthogonal + diagonal).
    - Hex for hexagonal: 6 cube directions.

- **Line adjacency**
  - Same as movement adjacency for each board type.

- **Territory adjacency**
  - square8/square19: Von Neumann (4 orthogonals).
  - hexagonal: same 6 hex directions as movement.

You must also support **straight-line rays** along these directions for movement, capture, and line-detection.

### 1.3 Quick mini-scenarios (implementation sanity checks)

- **Chain capture continuation:** After an initial capture spawns multiple follow-ups, enter `chain_capture` and require continuation until no capture remains. Legal options must be enumerated explicitly; there is no free skip.
- **Territory disconnection:** After line resolution, recompute regions using board-type adjacency; any region with zero controlled stacks collapses. Collapsed rings are credited to their owners (supply/victory) even if controlled by another player.
- **Forced elimination entry:** Only enter `forced_elimination` when the current player controls ≥1 stack and has zero placements/movements/captures. Record the FE move explicitly; never apply FE silently during territory exit.

### 1.4 Core state

At minimum, your engine must maintain:

- **BoardState**:
  - `stacks: Map<PosKey, RingStack>`
    - `RingStack = { position: Position; rings: PlayerId[]; stackHeight: number; capHeight: number; controllingPlayer: PlayerId }`
    - `rings` ordered bottom→top; `controllingPlayer = rings[top]`.
  - `markers: Map<PosKey, MarkerInfo>`
    - `MarkerInfo = { player: PlayerId; position: Position; type: 'regular' }`
  - `collapsedSpaces: Map<PosKey, PlayerId>`
  - `territories: Map<RegionId, Territory>` (optional cache; engine can recompute)
  - `formedLines: LineInfo[]` (optional cache)
  - `eliminatedRings: { [player: PlayerId]: number }`
  - `size: number`
  - `type: BoardType`

- **GameState**:
  - `boardType: BoardType`
  - `board: BoardState`
  - `players: Player[]` with per-player fields including:
    - `ringsInHand: number`
    - `eliminatedRings: number`
    - `territorySpaces: number`
  - `currentPhase: GamePhase ∈ { ring_placement, movement, capture, chain_capture, line_processing, territory_processing, forced_elimination, game_over }`
  - `currentPlayer: PlayerId`
  - `moveHistory: Move[]`
  - Victory metadata:
    - `totalRingsInPlay` (initial total over all players, board type)
    - `totalRingsEliminated`
    - `victoryThreshold` = floor(totalRingsInPlay/2) + 1
    - `territoryVictoryThreshold` = floor(totalSpaces/2) + 1

- **Stacks, cap height, control**
  - `stackHeight = rings.length`
  - `controllingPlayer = rings[top]`
  - `capHeight = number of consecutive rings from the top that equal controllingPlayer`.

You must maintain `capHeight` and `stackHeight` correctly after **placement**, **movement**, **overtaking**, and **elimination**.

---

## 2. Turn & Phase Structure

Each turn of `currentPlayer` is a deterministic state machine:

1. **Optional / conditional ring_placement**
2. **Mandatory movement** (if any legal movement or capture exists)
3. **Optional start of overtaking capture**, followed by **mandatory chain continuation** while legal
4. **Line processing** (zero or more lines)
5. **Territory disconnection processing** (zero or more regions)
6. **Victory/termination check**
7. Advance `currentPlayer` and compute that player’s starting phase (ring_placement or movement) for their next turn.

### 2.1 Ring placement legality

Let `P` be `currentPlayer`.

- Placement is **mandatory** if:
  - `P.ringsInHand > 0` AND `P` controls **no stacks on the board**.
  - OR: `P.ringsInHand > 0` AND `P` has stacks on the board but **no legal movement or capture** is available from any stack.
- Placement is **optional** (may be skipped) if: `P.ringsInHand > 0` and `P` controls at least one stack with legal movement/capture.
- Placement is **forbidden** if: `P.ringsInHand == 0`.

Placement options (if allowed):

1. **On empty space**:
   - Choose a non-collapsed, empty cell.
   - Place **1–3 rings** forming a new stack, subject to:
     - You cannot exceed `ringsInHand`, and
     - After placement, the total number of rings of your colour that are in play (on the board in any stack, regardless of controlling player, plus in your hand) must not exceed the `boardConfig.ringsPerPlayer` own-colour supply cap for this board type. Captured opponent-colour rings in stacks you control do **not** count against this limit.
   - **Legality constraint (no-dead-placement):** After placement, there must exist at least one legal non-capture or capture move for this exact stack according to movement/capture rules. If not, this placement is illegal.

2. **On existing stack (any owner)**:
   - Choose a non-collapsed cell with a stack.
   - Place **exactly one ring** of `P` on top of that stack (**never more than one ring per placement action**).
   - New controllingPlayer becomes `P`.
   - **Legality constraint (no-dead-placement):** After this placement, that stack must have at least one legal move/capture.

> **Note:** Multi-ring placement is only ever allowed on **empty** spaces. When placing onto an existing stack, you may place at most **one** ring per placement action.

If no legal placement exists when placement is mandatory, the player **skips placement** and proceeds to forced-elimination / movement logic as below.

> **Important (ringsInHand == 0 scenario):** When placement is **forbidden** because `P.ringsInHand == 0` but `P` controls stacks on the board, the engine must immediately proceed to the **movement phase** and enumerate movement/capture moves for P's controlled stacks. The `skip_placement` move type is only valid when `ringsInHand > 0`; when `ringsInHand == 0`, the placement phase is implicitly bypassed and movement moves are the only valid actions (unless P is blocked and must perform forced elimination).

### 2.2 Movement phase (required when possible)

After placement step, define the set of **controlled stacks** `S = { stacks where controllingPlayer = P }`.

- If `S` is empty:
  - If `P.ringsInHand > 0`, you must fall back to the ring‑placement rules in Section 2.1/6: your only way to act is by placing a new stack that satisfies the no‑dead‑placement rule.
  - If `P.ringsInHand == 0`, you have no material under your direct control and therefore no movement or capture action; your turn ends immediately and play passes to the next player. You are temporarily inactive or eliminated according to Section 7.3.
- Otherwise (`S` non‑empty), compute all legal movement/capture moves from stacks in `S` (Section 3 & 4).
  - If **no legal moves or captures** exist _and_ there is at least one legal placement (per the no‑dead‑placement rule), you must either place (if placement is mandatory) or you may choose to place (if placement is optional).
  - If **no legal moves/captures** and **no legal placements** exist, you are **blocked with stacks** and must go to forced elimination (Section 2.3).
  - If ≥1 legal move/capture exists:
    - Movement is **mandatory**.
    - If a placement occurred this turn, you **must move that specific stack**.
    - Otherwise, you may choose any controlled stack with a legal move/capture.

Movement may be **non-capturing** or an **overtaking capture** as its first segment.

### 2.3 Forced elimination when blocked

At the start of your action (after any mandatory placement checks) and before moving:

- If:
  - `P` controls at least one stack **on the board**, AND
  - There is **no legal placement, movement, or overtaking capture** for `P` (taking the no‑dead‑placement rule into account),
- Then `P` must **eliminate one entire cap** from any one stack they control:
  - Remove the top `capHeight` rings from that stack.
  - Update `board.stacks` and `eliminatedRings` counters.
  - If that stack becomes empty, remove it.

If after this elimination `P` still has no legal action, their turn ends.

**Control-flip edge case:** If `P`'s only control over a stack was a cap of height 1 (a single ring of `P`'s colour on top of opponent rings), forced elimination removes that cap and flips stack control to the opponent. If this causes `P` to have **zero controlled stacks** and **zero rings in hand**, `P` becomes "temporarily inactive" (per Section 7.3) immediately. Turn rotation should then skip `P` and proceed to the next player who has material; `P`'s turn effectively ends at the moment of forced elimination without any further action.

However, as long as any stacks remain on the board, it is never legal for the game to remain in an `active` state with the current player having no legal action. In any situation where **no player** has any legal placement, movement, or capture but at least one stack still exists on the board, the controlling player of some stack on their turn must satisfy the condition above and perform a forced elimination. Successive forced eliminations continue (possibly cycling through multiple players) until **no stacks remain**; only then can the game reach a structurally terminal state that is resolved by the stalemate rules in Section 7.4.

---

## 3. Non-Capture Movement

### 3.1 Path and distance

Given a stack at `from` with `height = H` (stackHeight):

- Possible directions: movement directions per `movementAdjacency` for the board.
- A path is a sequence of positions along a straight line in one of these directions:
  - `from = p0, p1, p2, ..., pk`.

**Distance** for movement is the number of steps `d = k`.

**Legal path constraints** for non-capture movement:

1. `d ≥ H` (must move at least stack height).
2. All intermediate cells `p1..pk-1` must:
   - Not be collapsed spaces.
   - Not contain any stack (`board.stacks[pos] == undefined`).
   - May contain markers (any color).
3. Landing cell `pk` must:
   - Not be a collapsed space.
   - Not contain any stack.
   - May contain any marker (own or opponent). Landing on any marker is legal but incurs a cap-elimination cost (see Section 3.2).

### 3.2 Marker interaction (movement)

When moving along the path:

- At departure `from`: leave a marker of `P` (`MarkerInfo` of type `'regular'`).
- For each intermediate cell `pi` with a marker:
  - If marker belongs to opponent `Q ≠ P`: **flip** it to `P`.
  - If marker belongs to `P`: **collapse** it to `collapsedSpaces[pos] = P` and remove marker.
- At landing cell `pk`:
  - If there is any marker (own or opponent), **remove** it (do not collapse); then place the stack and immediately eliminate the top ring of that stack's cap, crediting that eliminated ring to `P` for victory-condition purposes.

You are **not required** to stop at the first legal landing after markers; any landing `pk` satisfying distance and landing conditions is allowed.

---

## 4. Overtaking Capture Movement

### 4.1 Single capture segment

A single overtaking capture segment is defined by `(from, target, landing)`:

- `from`: current stack position with controllingPlayer `P`, height `H`, capHeight `CH`.
- `target`: position of a stack to be overtaken.
- `landing`: empty or same-color marker cell beyond `target`.

**Validation requirements**:

1. **Stacks/players**:
   - `from` contains a stack controlled by `P`.
   - `target` contains a stack `T` (any owner) with capHeight `CH_T`.
   - `CH ≥ CH_T`.
   - `T.controllingPlayer` may be `P` or opponent; capturing own stack is allowed.

2. **Geometry**:
   - `from`, `target`, `landing` lie on the same straight line direction allowed by `movementAdjacency`.
   - For hex, use cube-coordinate straight lines; for square, orthogonal or diagonal only.

3. **Path constraints**:
   - On `from → target` (excluding endpoints): no collapsed spaces, no stacks.
   - On `target → landing` (excluding endpoints): no collapsed spaces, no stacks.

4. **Distance constraints**:

- Let `segmentDistance = distance(from, landing)` (Chebyshev/king-move distance on square boards, cube distance on hex).
- `segmentDistance ≥ H`.
- `landing` must be strictly further from `from` than `target` is (i.e. beyond the target in same direction).

5. **Landing cell**:
   - Not a collapsed space.
   - Does not contain a stack.
   - May contain any marker (own or opponent). Landing on a marker incurs a cap-elimination cost (see Section 4.2).

6. **Markers on path**:
   - Process as in non-capture movement:
     - flip opponent markers to `P`;
     - collapse `P` markers on intermediate cells;
     - if landing on any marker, remove it and eliminate the top ring of the attacking stack's cap.

If all conditions hold, the segment is legal.

### 4.2 Applying a capture segment

When a legal `(from, target, landing)` is executed:

1. Process markers along path as in movement (flip opponent markers, collapse own markers on intermediate cells).
2. Move the attacking stack from `from` to `landing` (update stacks map; `from` becomes empty).
3. Pop the **top ring** from the `target` stack and append it to the **bottom** of the attacking stack's `rings` array.
4. Recompute `stackHeight` and `capHeight` for the updated stack.
5. If the `target` stack becomes empty, remove it.
6. If landing on any marker (own or opponent), remove the marker and eliminate the top ring of the attacking stack's cap.

Capturing from your own stack is allowed; the ring changes vertical position but remains in play.

### 4.3 Chain overtaking rule (mandatory continuation)

Once any overtaking segment is performed in a turn:

- **Chain capturing is mandatory** as long as there exists at least one legal capture segment starting from the current stack position.
- At each step:
  - Generate all legal capture segments from current position.
  - If none exist → chain ends.
  - If one or more exist → **you must choose one** and execute it.
- You may choose a segment that leads to a state with **no further captures**, thereby ending the chain, even if other choices would allow more captures.
- Chains may:
  - Change direction between segments.
  - 180° reverse over previously captured stacks.
  - Recapture from the same target multiple times (as long as it still has rings and path constraints remain satisfied).

---

## 5. Lines & Graduated Line Rewards

### 5.1 Line detection

A **line** is a maximal sequence of positions `[p0, p1, ..., pk]` with all:

- Containing **markers** (not stacks) of the **same player** `P`;
- Consecutive along a single line-adjacency axis (Moore for square, 3 axes for hex);
- Not interrupted by empty cells, stacks, or collapsed spaces.

A line is **eligible** if `length ≥ lineLength` for the board type.

### 5.2 Line processing order

After a turn’s movement/captures:

1. Enumerate all eligible lines.
2. While any eligible line remains:
   - The moving player chooses **one** line to process next.
   - Apply collapse/elimination for that line (below).
   - Recompute lines (since collapses may break others).

### 5.3 Collapse & elimination logic

Let `L` be a chosen line of `len = |L|`, belonging to player `P`, with `requiredLen = lineLength` (3 or 4, depending on board type).

Two cases:

1. **Exact length: `len == requiredLen`**
   - Collapse **all markers in L** to `collapsedSpaces[pos] = P` and remove markers.
   - `P` must **eliminate** either:
     - one of their standalone rings (from any stack they control), or
     - the **entire cap** of one of their controlled stacks.
   - All eliminated rings update `eliminatedRings` counters and victory totals.

2. **Overlength: `len > requiredLen`**
   - `P` chooses **Option 1** or **Option 2**:
     - **Option 1 (max territory):**
       - Collapse **all** `len` markers in `L` to `P`’s collapsed spaces.
       - Eliminate one ring / cap as above.
     - **Option 2 (ring preservation):**
       - Choose **any contiguous segment** of `requiredLen` positions within `L`.
       - Collapse exactly those `requiredLen` markers to `P`’s collapsed spaces.
       - **No rings are eliminated**.

Collapsed spaces are permanent: they cannot hold stacks or markers and act as obstacles for movement and capture.

---

## 6. Territory Disconnection & Region Collapse

### 6.1 Region & border definitions

For territory processing, use `territoryAdjacency` (Von Neumann on square, hex adjacency on hex).

- **Region**: A maximal set of cells `R` such that:
  - Each cell in `R` is **not** collapsed.
  - Cells in `R` are connected via `territoryAdjacency`.

- **Border** of `R`: Neighbor cells of `R` (via `territoryAdjacency`) that are **not** in `R`. These can be:
  - Board edge (off-board),
  - Collapsed spaces of any color,
  - Markers, stacks, or empty cells.

A region `R` is **physically disconnected** when **all paths** from any cell in `R` to the rest of the board’s non-collapsed cells must cross:

- Collapsed spaces, and/or
- Board edges, and/or
- Markers of exactly **one single player** `B` (border color);
- i.e. the non-collapsed marker portions of the border belong to one player only.

### 6.2 Representation criterion

Let `ActiveColors` = set of players that currently have at least one ring on the board (in any stack).

For region `R`, define `RegionColors` = set of players that control at least one **stack** in `R` (top ring ownership). Markers and empties do **not** count.

`R` is **color-disconnected** if `RegionColors` is a **proper subset** of `ActiveColors`:

- At least one active player has no stack in `R`.

**Important rule:** If `RegionColors == ActiveColors` (all active players have representation in `R`), `R` is **never** disconnected, regardless of border.

### 6.3 Self-elimination prerequisite (per region)

For each candidate region `R` and moving player `P`:

- Consider a hypothetical state in which all rings in `R` have been eliminated.
- If in that hypothetical state `P` would still control at least one ring or stack cap **outside R**, then `P` is allowed to process `R` and pay the mandatory self-elimination.
- Otherwise, `R` **cannot** be processed by `P` at this time.

Each ring/cap outside the region can pay for **one** processed region only; after actually eliminating it, re-evaluate for remaining regions.

### 6.4 Processing disconnected regions

After all line processing is complete:

1. Discover all regions `R` that are both:
   - physically disconnected, and
   - color-disconnected.

2. For each such region, in any order chosen by `P`:
   - Check the self-elimination prerequisite.
   - If it fails, skip this region (it remains unchanged).
   - If it passes, process the region:

**Collapse and elimination steps for region `R`:**

1. **Collapse interior:**
   - For every cell in `R`, set `collapsedSpaces[pos] = P` and clear stacks/markers from these cells.

2. **Collapse border markers of the single border color:**
   - For all markers of the border color `B` that lie on the border and participate in the disconnection (i.e., contiguous along the border path), collapse them as well to `P`’s color.

3. **Eliminate rings inside `R`:**
   - All rings from stacks originally inside the region are removed and counted as eliminated, contributing to `P`’s elimination total (they are attributed to `P` as the causing player).

4. **Mandatory self-elimination:**
   - `P` must eliminate one of their remaining rings or one entire stack cap **outside** this region.

5. Update all ring/territory counters and derived stats.

6. After each region is processed, recompute regions again; new regions may have become disconnected.

All eliminated rings (from inside regions and self-eliminations) count toward `P`’s ring-elimination victory total.

---

## 7. Victory Conditions & Game End

The game ends immediately if **any** of these is true after a full turn (including post-movement processing):

### 7.1 Ring-elimination victory

For player `P`:

- Let `P.eliminatedRingsTotal` be the total number of rings credited to `P` as eliminated (through lines, regions, forced elimination, stalemate conversion of rings in hand, etc.).
- If `P.eliminatedRingsTotal ≥ victoryThreshold` (strictly more than 50% of `totalRingsInPlay`), `P` wins.
- This cannot occur for multiple players simultaneously by construction.

### 7.2 Territory-control victory

For player `P`:

- Let `territorySpaces[P]` = number of cells where `collapsedSpaces[pos] == P`.
- If `territorySpaces[P] ≥ territoryVictoryThreshold` (>50% of board space), `P` wins.

### 7.3 Last-player-standing victory

Last-player-standing is a **third formal victory condition**, alongside ring-elimination (Section 7.1) and territory-control (Section 7.2).

For this rule, define a **real action** for a player `P` on their own turn as any legal:

- ring placement (Section 2.1),
- non-capture movement (Section 3), or
- overtaking capture segment or chain (Section 4),

available at the start of their action. Having only forced elimination available (Section 2.3) does **not** count as having a real action for last-player-standing purposes.

A **full round of turns** is one contiguous cycle of turns in player order in which each non-eliminated player takes exactly one turn.

A player `P` wins by last-player-standing if all of the following hold:

- There exists at least one full round of turns such that:
  - On each of `P`’s turns in that round, `P` has at least one legal real action available at the start of their action; and
  - On every other player’s turns in that same round, those players have **no** legal real action available at the start of their action (they may have only forced-elimination actions or no legal actions at all); and
- Immediately after that round completes (including all line and territory processing), at the start of `P`’s next turn `P` is still the only player who has any legal real action.

A player is **temporarily inactive** (has no real actions on their own turn, but remains in the game) when:

- They have stacks but no legal non-capture moves or overtaking captures and cannot place any ring, OR
- They control no stacks on the board and cannot place any ring, BUT
- They still have rings on the board in stacks controlled by others.

Such a player can potentially become active again if capture or elimination expose one of their buried rings as the new top ring of a stack, thereby giving them a controlled stack on a later turn and restoring at least one real action.

If any temporarily inactive player regains a real action **before** the full-round condition above has been satisfied, the last-player-standing condition effectively resets and must be re-satisfied from that point.

A player is **eliminated** (has no legal actions on their own turn, and cannot in future turns) when:

- They have no stacks or rings anywhere on the board AND
- They have `ringsInHand == 0`

### 7.4 Stalemate resolution

If **no** player has any legal placement, movement, capture, or forced elimination available (global stalemate):

- In practice, this can only occur once **no stacks remain** anywhere on the board. Structural terminality with stacks still present is ruled out by the forced-elimination rule in Section 2.3, because any such situation must be resolved by successive forced eliminations until all stacks have been removed.

1. Convert any rings in hand for each player into **eliminated rings** for that player.
2. Compute the following ranking, in order:
   1. **Most collapsed spaces** controlled.
   2. If tied, **most eliminated rings** (including rings converted from hand in step 1).
   3. If still tied, **most markers** on the board.
   4. If still tied, the **last player to complete a valid turn action**.
3. Highest rank wins.

---

## 8. AI/Engine Notes (Implementation-Oriented)

- Treat the rules as a pure **state-transition system**:
  - State: `GameState`.
  - Inputs: `Move` proposals from players/AI.
  - Transitions: apply placement → movement/capture → lines → regions → victory checks.

- For AI search, useful derived functions:
  - `generateLegalPlacements(GameState, player)`
  - `generateLegalNonCaptureMoves(GameState, player)`
  - `generateLegalCaptureSegments(GameState, player)` and chain expansion from a given position.
  - `computeLines(BoardState)` returning candidate lines.
  - `computeDisconnectedRegions(BoardState)` with metadata about border colors and representation.

- Carefully respect **version parameters** (lineLength; adjacency type; ringsPerPlayer; thresholds) so that the same engine logic can run square8, square19, and hex.

This compact spec plus the full narrative rules and tests should suffice to produce a complete, correct implementation of RingRift’s rules for both server engines and AI agents.

---

## 9. Progress & Termination Invariant (Implementation Note)

For engine and AI authors, it is useful to track a simple global progress measure over the course of a game:

- Let **M** = number of markers currently on the board.
- Let **C** = number of collapsed spaces currently on the board.
- Let **E** = total number of _eliminated_ rings credited to any player (including rings eliminated from lines, territory disconnections, forced eliminations, and stalemate conversion of rings in hand).
- Define the **progress metric**:

  ```text
  S = M + C + E
  ```

Under the rules above:

- Any legal **movement** (non-capture or overtaking) always places at least one new marker on the departure space, and never removes collapsed spaces or resurrects eliminated rings. Landing on one of your own markers removes that marker but immediately eliminates the top ring of the moving stack, keeping `M + E` unchanged at the landing cell. The departure marker therefore ensures **S strictly increases** on every move.
- **Collapsing markers to territory** (e.g., via lines or region processing) replaces markers with collapsed spaces one-for-one, so `M + C` remains unchanged by that operation alone.
- **Eliminations** (from lines, disconnected regions, forced elimination, or stalemate conversion of rings in hand) strictly increase `E` and never decrease markers or collapsed spaces.
- **Forced elimination when blocked** (Section 4.4 / 2.3) always eliminates at least one ring, so it strictly increases `E` even on turns where no movement is possible.
- No rule ever decreases the number of collapsed spaces or eliminated rings.

On any turn where a player performs a legal action (movement, chain capture segment, region processing, or forced elimination), **S strictly increases**. The only turns that may leave `S` unchanged are rare “pure forfeits” where a player is required to place but has no legal placement and no stacks, so they simply pass.

Because the board has a finite number of spaces and a finite total number of rings, there is a finite upper bound on `S`:

```text
S ≤ (#boardSpaces) + (#boardSpaces) + (totalRingsInPlay) = 2·N + R_total.
```

Since `S` is non-decreasing and bounded, there can only be **finitely many** turns that involve a real action (movement or forced elimination) in any game. Eventually, no player can have any legal placement, movement, or capture without exceeding this bound. At that point the game is in a global no-moves state and must be resolved by the rules in Section 7 (victory thresholds, last-player-standing, or stalemate tiebreakers).

This invariant is primarily an implementation aid, but it justifies the expectation that:

- Every correctly implemented RingRift engine must see any legal game terminate in finite time.
- Any long-running simulation that fails to terminate within a large move cap either:
  - violates the rules’ accounting for markers, collapsed spaces, or eliminated rings, or
  - is using a move cap that is simply too small relative to the worst-case legal game length.
