# RingRift Compact Rules (Engine / AI Implementation Spec)

**Purpose:** This document is a compact, implementation‑oriented specification of the RingRift rules. It is designed for engine/AI authors, not for teaching humans. It encodes the *minimum complete* rule set needed to implement a correct engine for all supported versions.

- Full, narrative rules: `ringrift_complete_rules.md`
- This file: focuses on **state**, **version parameters**, and **transition rules**.

---
## 1. Version Parameters & Board Model

### 1.1 Supported board types

`BoardType ∈ { square8, square19, hexagonal }`

For each board type, define a static configuration:

| BoardType   | size | totalSpaces | ringsPerPlayer | lineLength | movementAdjacency | lineAdjacency | territoryAdjacency | boardGeometry  |
|------------|------|-------------|----------------|------------|-------------------|---------------|--------------------|----------------|
| square8    | 8    | 64          | 18             | 4          | Moore (8-dir)     | Moore         | Von Neumann (4-dir)| orthogonal grid|
| square19   | 19   | 361         | 36             | 5          | Moore             | Moore         | Von Neumann        | orthogonal grid|
| hexagonal  | 11   | 331         | 36             | 5          | Hex (6-dir)       | Hex           | Hex                | hex coordinates|

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

### 1.3 Core state

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
  - `currentPhase: GamePhase ∈ { ring_placement, movement, capture, line_processing, territory_processing }`
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
   - Place one or more rings (any count ≤ `ringsInHand`) forming a new stack.
   - **Legality constraint:** After placement, there must exist at least one legal non-capture or capture move for this exact stack according to movement/capture rules. If not, this placement is illegal.

2. **On existing stack (any owner)**:
   - Choose a non-collapsed cell with a stack.
   - Place **exactly one ring** of `P` on top of that stack.
   - New controllingPlayer becomes `P`.
   - **Legality constraint:** After this placement, that stack must have at least one legal move/capture.

If no legal placement exists when placement is mandatory, the player **skips placement** and proceeds to forced-elimination / movement logic as below.

### 2.2 Movement phase (required when possible)

After placement step, define the set of **controlled stacks** `S = { stacks where controllingPlayer = P }`.

- If `S` is empty AND `P.ringsInHand == 0` → go to **forced elimination** (Section 2.3).
- Otherwise, compute all legal movement/capture moves from stacks in `S` (Section 3 & 4).
  - If **no legal moves or captures** exist *and* `P.ringsInHand > 0`:
    - If placement was not yet attempted but was possible, this situation should not occur (enforce via placement legality). If it occurs, treat as forced elimination.
  - If **no legal moves/captures** and `P.ringsInHand == 0` → forced elimination.
  - If ≥1 legal move/capture exists:
    - Movement is **mandatory**.
    - If a placement occurred this turn, you **must move that specific stack**.
    - Otherwise, you may choose any controlled stack with a legal move/capture.

Movement may be **non-capturing** or an **overtaking capture** as its first segment.

### 2.3 Forced elimination when blocked

At the start of your action (after any mandatory placement checks) and before moving:

- If:
  - `P` controls at least one stack **on the board**, AND
  - `P` has `ringsInHand == 0`, AND
  - There is **no legal placement, movement, or overtaking capture** for `P`,
- Then `P` must **eliminate one entire cap** from any one stack they control:
  - Remove the top `capHeight` rings from that stack.
  - Update `board.stacks` and `eliminatedRings` counters.
  - If that stack becomes empty, remove it.

If after this elimination `P` still has no legal action, their turn ends.

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
   - If it has a marker, it **must** be a marker of `P` (same player as controllingPlayer). Landing on opponent marker is illegal.

### 3.2 Marker interaction (movement)

When moving along the path:

- At departure `from`: leave a marker of `P` (`MarkerInfo` of type `'regular'`).
- For each intermediate cell `pi` with a marker:
  - If marker belongs to opponent `Q ≠ P`: **flip** it to `P`.
  - If marker belongs to `P`: **collapse** it to `collapsedSpaces[pos] = P` and remove marker.
- At landing cell `pk`:
  - If there is a `P` marker, **remove** it (no collapse on landing); then place the stack.

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
   - Let `segmentDistance = distance(from, landing)` (Manhattan on square, cube distance on hex).
   - `segmentDistance ≥ H`.
   - `landing` must be strictly further from `from` than `target` is (i.e. beyond the target in same direction).

5. **Landing cell**:
   - Not a collapsed space.
   - Does not contain a stack.
   - If contains marker, it must be a marker of `P`.

6. **Markers on path**:
   - Process as in non-capture movement:
     - flip opponent markers to `P`;
     - collapse `P` markers on intermediate cells;
     - remove a `P` marker if landing on it.

If all conditions hold, the segment is legal.

### 4.2 Applying a capture segment

When a legal `(from, target, landing)` is executed:

1. Process markers along path as in movement.
2. Move the attacking stack from `from` to `landing` (update stacks map; `from` becomes empty).
3. Pop the **top ring** from the `target` stack and append it to the **bottom** of the attacking stack’s `rings` array.
4. Recompute `stackHeight` and `capHeight` for the updated stack.
5. If the `target` stack becomes empty, remove it.

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

Let `L` be a chosen line of `len = |L|`, belonging to player `P`, with `requiredLen = lineLength` (4 or 5).

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

A player `P` wins by last-player-standing if, after a completed turn and post-processing:

- All **other** players have **no legal action** on their next turns:
  - no legal placement,
  - no legal movement,
  - no legal capture,
- And `P` still has at least one legal action available on their own next turn.

A player is effectively “dead” when:
- They control no stacks on the board, AND
- They have `ringsInHand == 0`, OR
- They have stacks but no legal moves/captures *and* cannot place any ring.

Control may later be regained if another player’s captures expose a top ring of their color on some stack.

### 7.4 Stalemate resolution

If **no** player has any legal placement, movement, or capture (global stalemate):

1. Convert any rings in hand for each player into **eliminated rings** for that player.
2. Compute the following ranking, in order:
   1. **Most collapsed spaces** controlled.
   2. If tied, **most eliminated rings**.
   3. If still tied, **most markers** on the board.
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
