# RingRift Canonical Rules Specification

> **Doc Status (2025-11-26): Active**
>
> - This document is the **rules-level canonical spec** for RingRift semantics. It normalizes the narrative sources into precise, implementation-ready constraints.
> - It defines **RR-CANON-RXXX** rule IDs for core invariants, resources, turn/phase structure, line/territory semantics, and victory conditions.
> - It intentionally **does not** re-specify the engine API or Move/decision/WebSocket lifecycle. For those, see:
>   - [`docs/CANONICAL_ENGINE_API.md` §3.9–3.10](docs/CANONICAL_ENGINE_API.md) for `Move`, `PendingDecision`, `PlayerChoice*`, WebSocket payloads, and the orchestrator-backed decision loop.
>   - `src/shared/types/game.ts` and `src/shared/engine/orchestration/types.ts` for the canonical type definitions.
> - Engine implementations in TS and Python must:
>   - Treat `src/shared/engine/` (**helpers → aggregates → orchestrator → contracts**) as the SSoT for rules behavior.
>   - Treat this file as the **rules invariant SSoT** whenever the prose sources diverge.
>
> **Purpose.** This document is a normalization of the RingRift rules for engine/AI implementation and verification. It reconciles [`ringrift_complete_rules.md`](ringrift_complete_rules.md) ("Complete Rules") and [`ringrift_compact_rules.md`](ringrift_compact_rules.md) ("Compact Spec") into a single canonical, implementation-ready ruleset.

The canonical rules here are binding whenever the two source documents diverge. For each rule we provide:

- A stable rule identifier `RR-CANON-RXXX`.
- A precise statement of the rule.
- Applicability (all versions vs version-specific).
- References into the Complete and Compact docs.

The Compact Spec is generally treated as primary for formal semantics, and the Complete Rules as primary for examples, motivation, and prose. Explicit exceptions and judgment calls are documented in Sections 12 and 13.

---

## 0. Conventions

- **Players.** 2–4 players; 3-player games are the default in all examples.
- **Board types.**
  - `square8`: 8×8 orthogonal grid.
  - `square19`: 19×19 orthogonal grid.
  - `hexagonal`: hex board with radius 10 (11 cells per side).
- **Notation.**
  - "Stack" = one or more rings in a single cell.
  - "Cap" = all consecutive rings from the top of a stack belonging to the controlling player.
  - "Eliminated ring" = ring permanently removed from the board and credited to a player.
  - "Collapsed space" = permanently claimed territory cell, impassable to movement and capture.
- **Version parameters.** Unless otherwise stated, all numeric length thresholds (line length, victory thresholds, etc.) come from the board-type configuration in RR-CANON-R001.

---

## 1. Entities and Components

### 1.1 Board types and geometry

- **[RR-CANON-R001] Board type configuration.**
  - For each `BoardType ∈ { square8, square19, hexagonal }`, define:
    - `size`, `totalSpaces`, `ringsPerPlayer`, `lineLength`, `movementAdjacency`, `lineAdjacency`, `territoryAdjacency`, `boardGeometry` exactly as in the table in the Compact Spec (§1.1).
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §1.1; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§1.2.1, 16.10.

- **[RR-CANON-R002] Coordinate systems.**
  - Square boards use integer coordinates `(x,y)` with `0 ≤ x,y < size`.
  - Hex board uses cube coordinates `(x,y,z)` with `x + y + z = 0` and `max(|x|,|y|,|z|) ≤ size-1`.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §1.1.

- **[RR-CANON-R003] Adjacency relations.**
  - Movement and line directions use `movementAdjacency` / `lineAdjacency` from RR-CANON-R001.
  - Territory connectivity uses `territoryAdjacency` from RR-CANON-R001.
  - Straight-line rays along these directions are used for movement, capture, and line detection.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §§1.2, 3, 4, 5; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§2.1, 3.1, 8, 11, 12, 16.9.4.1.

### 1.2 Players and identifiers

- **[RR-CANON-R010] Player identifiers.**
  - Each player has a unique `PlayerId`.
  - Exactly 2–4 players participate; 3 is the default.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §1.3, §7; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§1.1, 1.2, 13, 15.1, 19×19/8×8 setup sections.

### 1.3 Rings, stacks, and control

- **[RR-CANON-R020] Rings per player (own-colour supply cap).**
  - For each board type, each player P has a fixed personal supply of rings of P's **own colour**:
    - `square8`: 18 rings.
    - `square19`: 36 rings.
    - `hexagonal`: 36 rings.
  - At all times, the total number of rings of P's colour that are **in play** (on the board in any stack, regardless of which player currently controls those stacks, plus in P's hand) must be ≤ this `ringsPerPlayer` value for the chosen board type.
  - Rings of other colours that P has captured and that are buried in stacks P controls **do not** count against P's `ringsPerPlayer` cap; they continue to belong, by colour, to their original owners for conservation, elimination, and victory accounting.
  - Eliminated rings of P's colour are permanently out of play and do not refresh or expand P's supply beyond `ringsPerPlayer`; they only change how much of that fixed supply is currently eliminated versus still in play.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md:18) §1.1, §7.1; [`ringrift_complete_rules.md`](ringrift_complete_rules.md:340) §§1.2.1, 3.2.1, 16.3–16.4, 16.9.2.

- **[RR-CANON-R021] Stack definition.**
  - A stack is an ordered sequence of one or more rings on a single board cell.
  - Rings are ordered bottom→top.
  - A cell may contain **either** a stack, **or** a marker, **or** be empty, **or** be collapsed; never more than one of these simultaneously.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §1.3; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§5–7.

- **[RR-CANON-R022] Control and cap height.**
  - `controllingPlayer` of a stack is the color of its top ring.
  - `stackHeight = rings.length`.
  - `capHeight` is the number of consecutive rings from the top that belong to `controllingPlayer`.
  - Control changes whenever the top ring changes color (due to overtaking or elimination).
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §1.3; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§5.1–5.3, 7.2, 15.4 Q16.

- **[RR-CANON-R023] Stack mutation operations.**
  - Legal stack mutations are limited to:
    - Adding rings at the **top** via placement.
    - Adding rings at the **bottom** via overtaking capture.
    - Removing the **top ring** of a stack via overtaking capture.
    - Removing an entire cap (all consecutive top rings of the controlling color) via forced elimination, line processing, or region processing.
    - Removing all rings in a region during territory collapse.
  - Stacks may **never** be split or reordered in any other way.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §§2–4, 5, 6; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§5–7, 9–12, 15.4 Q1.

### 1.4 Markers and collapsed spaces

- **[RR-CANON-R030] Marker definition.**
  - A marker is a color token occupying a cell that is not currently a stack or collapsed space.
  - Each marker belongs to exactly one player.
  - Markers are created only as departure markers from movement/capture (RR-CANON-R082).
  - Markers may be flipped to another player or collapsed into Territory as movement/capture passes over them.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §§1.3, 3.2, 4.1; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§3.2.2, 8, 11, 12, 16.5–16.6.

- **[RR-CANON-R031] Collapsed spaces.**
  - A collapsed space is a cell permanently claimed as Territory by exactly one player.
  - Collapsed spaces may not contain stacks or markers, may not be moved through, and act as barriers for both movement and Territory connectivity.
  - Collapsed spaces are created only by:
    - Line processing (RR-CANON-R120–R122).
    - Territory region processing (RR-CANON-R140–R146).
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §§1.3, 3–6; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§2.4, 3.1, 11–12, 16.8.

### 1.5 Regions and Territories

- **[RR-CANON-R040] Region for Territory processing.**
  - For a given board state, a **region** is a maximal set of non-collapsed cells connected via `territoryAdjacency`.
  - Regions may contain empty cells, markers, and stacks.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §6.1; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§12.1–12.2, 15.4 Q15.

- **[RR-CANON-R041] Territory counts.**
  - Each player `P` has an integer `territorySpaces[P]` = number of collapsed spaces whose owner is `P`.
  - This is used for Territory-victory and stalemate tiebreaks.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §§1.3, 7.2, 7.4; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§11–13.2, 13.4.

---

## 2. Core Game State

- **[RR-CANON-R050] BoardState fields.**
  - Canonical `BoardState` must at minimum contain:
    - `stacks: Map<PosKey, RingStack>`.
    - `markers: Map<PosKey, MarkerInfo>`.
    - `collapsedSpaces: Map<PosKey, PlayerId>`.
    - Optionally cached: territories, formedLines, etc.
    - Board metadata: `size`, `type`.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §1.3; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§3–12.

- **[RR-CANON-R051] GameState fields.**
  - Canonical `GameState` must at minimum contain:
    - Board type and `BoardState`.
    - Per-player data:
      - `ringsInHand`.
      - `eliminatedRings` (credited to that player).
      - `territorySpaces`.
    - Turn/phase: `currentPlayer`, `currentPhase` ∈ { `ring_placement`, `movement`, `capture`, `chain_capture`, `line_processing`, `territory_processing` }.
    - Victory metadata: `totalRingsInPlay`, `totalRingsEliminated`, `victoryThreshold`, `territoryVictoryThreshold`.
    - History: `moveHistory` (implementation-defined structure).
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §1.3, §2, §7; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§4, 13, 15.2.

- **[RR-CANON-R052] State validity invariants.**
  - For any valid state:
    - Each cell is in exactly one of: empty, marker, stack, collapsed.
    - No ring appears in more than one place; ring counts across stacks, hands, and eliminated totals never exceed initial rings per player.
    - `stackHeight` and `capHeight` for every stack are consistent with the ring list and top ring.
    - `totalRingsInPlay` = initial sum of rings for all players.
    - For each player `P`,
      - `ringsInHand[P]` + (rings in stacks owned by `P` + rings belonging to `P` buried in others' stacks) + rings credited in `eliminatedRings` for all causing players = initial rings for `P`.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §§1, 2, 4, 6–7, 9; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§3–13, 15.4 Q6, Q10, Q16.

---

## 3. Resources and Scoring

- **[RR-CANON-R060] Ring-elimination accounting.**
  - When a ring is eliminated (via line, region, forced elimination, or stalemate conversion of rings in hand), it is:
    - Removed from the board or from `ringsInHand`.
    - Credited to a **causing player** `P` as part of `P.eliminatedRingsTotal`.
  - Eliminated rings from any color (including self-elimination) contribute to the causing player's elimination total.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §§1.3, 5–7, 9; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§9.2, 11–13, 15.4 Q6, Q11–Q12.

- **[RR-CANON-R061] Ring-elimination victory threshold.**
  - `victoryThreshold = floor(totalRingsInPlay / 2) + 1`.
  - A player wins by elimination when their credited eliminated ring total reaches or exceeds `victoryThreshold`.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §1.3, §7.1; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§1.3, 13.1, 16.3, 16.9.4.5.

- **[RR-CANON-R062] Territory-control victory threshold.**
  - `territoryVictoryThreshold = floor(totalSpaces / 2) + 1`.
  - A player wins by Territory when their `territorySpaces[P]` reaches or exceeds `territoryVictoryThreshold`.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §1.3, §7.2; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§13.2, 16.2.

---

## 4. Turn / Phase / Step Structure

- **[RR-CANON-R070] Turn phases.**
  - A full turn for `currentPlayer = P` consists of the following ordered phases:
    1. Optional/conditional **ring placement**.
    2. Mandatory **movement** if any legal movement or capture exists.
    3. Optional start of **overtaking capture**, then mandatory chain continuation while legal.
    4. **Line processing** (zero or more lines).
    5. **Territory disconnection processing** (zero or more regions).
    6. **Victory / termination check**.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §2; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§4, 11–13, 15.2.

- **[RR-CANON-R071] Phase progression is deterministic.**
  - Phases always execute in the order of RR-CANON-R070.
  - Within line and Territory phases, player choices (which line/region to process next, and which self-elimination to apply) may affect future availability of lines/regions, but never the phase order.
  - References: same as RR-CANON-R070.

- **[RR-CANON-R072] Legal-action requirement and forced elimination entry.**
  - At the start of P's action, after accounting for any mandatory placement:
    - If P has at least one controlled stack and **no legal placement, no legal non-capture movement, and no legal overtaking capture**, P must attempt a **forced elimination** per RR-CANON-R100.
    - This ensures that as long as any stacks exist on the board, some player always has a legal action (movement or forced elimination) on their turn.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §2.2–2.3; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§4.2–4.4, 13.4, 13.5, 15.4 Q24.

---

## 5. Action Types and Legality

### 5.1 Ring placement

- **[RR-CANON-R080] Placement mandatory/optional/forbidden.**
  - Let P be current player.
  - Placement is **forbidden** if `P.ringsInHand == 0`.
  - Placement is **mandatory** if:
    - `P.ringsInHand > 0` and P controls **no stacks** on the board; or
    - `P.ringsInHand > 0`, P controls at least one stack, but **no legal movement or capture** is available from any controlled stack.
  - Placement is **optional** (may be skipped) if `P.ringsInHand > 0` and P controls at least one stack with a legal movement or capture.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md:100) §2.1; [`ringrift_complete_rules.md`](ringrift_complete_rules.md:362) §§4.1, 6.1, 15.2.

- **[RR-CANON-R081] Placement on empty cell.**
  - If placement is allowed:
    - P may choose a non-collapsed, empty cell and place **1–3 rings** there, forming a new stack.
    - P may not exceed `ringsInHand`, and after placement the total number of rings of P's colour that are in play (on the board in any stack, regardless of controlling player, plus in P's hand) must not exceed P's `ringsPerPlayer` own-colour supply cap for the board type (RR-CANON-R020). Captured opponent-colour rings in stacks P controls do **not** affect this check.
    - **No-dead-placement rule:** after hypothetical placement, that new stack must have at least one legal non-capture move or overtaking capture under the standard rules. Otherwise the placement is illegal.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md:100) §2.1; [`ringrift_complete_rules.md`](ringrift_complete_rules.md:366) §§4.1, 6.2, 15.4 Q17.

- **[RR-CANON-R082] Placement on existing stack.**
  - P may choose a non-collapsed cell containing any stack (friendly or opponent).
  - P may place **exactly one** ring of their color on top of that stack.
  - The stack's controlling player becomes P; capHeight is recomputed.
  - No-dead-placement rule applies: after placement, the updated stack must have at least one legal move or capture; otherwise the placement is illegal.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md:100) §2.1; [`ringrift_complete_rules.md`](ringrift_complete_rules.md:574) §§4.1, 6.3, 7.2.

### 5.2 Non-capture movement

- **[RR-CANON-R090] Movement availability.**
  - After placement (if any), let S be the set of stacks with `controllingPlayer = P`.
  - If S is empty and placement was impossible or forbidden, P has no legal movement and either becomes temporarily inactive or proceeds to forced elimination per RR-CANON-R100 (if they control a stack via some edge case such as future control change).
  - If S is non-empty, any stack in S that satisfies RR-CANON-R091–R092 for at least one direction has at least one legal move.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md:166) §§2.2, 3; [`ringrift_complete_rules.md`](ringrift_complete_rules.md:385) §§4.2, 8, 16.5.1.

- **[RR-CANON-R091] Path and distance for non-capture movement.**
  - Let a controlled stack be at `from` with height `H ≥ 1`.
  - A candidate path is a straight-line sequence `from = p0, p1, ..., pk = landing` along one movement direction.
  - Distance is `d = k`.
  - Requirements:
    - `d ≥ H`.
    - For all intermediate cells `p1..pk-1`:
      - Not collapsed.
      - Contain no stack.
      - May contain markers.
    - Landing cell `pk`:
      - Not collapsed.
      - Contains no stack.
      - If contains a marker, it must belong to P; landing on opponent markers is illegal.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §3.1; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§8.1–8.2, 16.9.4.1, 15.4 Q2.

- **[RR-CANON-R092] Marker interaction during non-capture movement.**
  - At departure `from`, place a regular marker of P (replacing any existing marker or empty cell; movement from collapsed spaces is illegal).
  - For each intermediate cell with a marker:
    - If marker belongs to opponent Q ≠ P, flip it to P.
    - If marker belongs to P, remove it and set that cell to a collapsed space owned by P.
  - At landing cell:
    - If there is a marker of P, remove that marker (do **not** collapse it).
    - Then place the moving stack.
    - Immediately eliminate the top ring of that stack and credit it to P.
  - P is **not required** to stop at the first legal landing after markers; any landing that satisfies RR-CANON-R091 is legal.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §3.2; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§4.2.1, 8.2, 8.3, 15.4 Q2.

### 5.3 Overtaking capture

- **[RR-CANON-R100] Forced elimination when blocked.**
  - Condition (start of P's action after placement checks):
    - P controls at least one stack (some cell whose top ring belongs to P), and
    - There exists **no** legal placement, non-capture movement, or overtaking capture for P.
  - Action:
    - P must choose one controlled stack and eliminate its entire cap (all consecutive top rings of P on that stack).
    - Those rings are credited to P as eliminated.
    - If the stack becomes empty, remove it.
  - If after this elimination P still has no legal action, P's turn ends.
  - Note (canonical choice): text in the Complete Rules suggesting that caps might already be eliminated while stacks remain is treated as unreachable; the forced-elimination rule is always applicable whenever P controls any stack.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §2.3; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§4.4, 13.4, 13.5, 15.4 Q24.

- **[RR-CANON-R101] Single capture segment legality.**
  - A single overtaking capture segment is a triple (`from`, `target`, `landing`) where:
    - `from` contains a stack controlled by P with height H and capHeight CH.
    - `target` contains a stack T with capHeight CH_T (any owner).
    - `landing` is an empty cell or a cell with a P marker, strictly beyond `target` along one movement direction.
  - Requirements:
    - CH ≥ CH_T.
    - from, target, landing lie on one straight-line movement direction.
    - On from→target (excluding endpoints): no stacks, no collapsed spaces.
    - On target→landing (excluding endpoints): no stacks, no collapsed spaces.
    - Distance `distance(from, landing) ≥ H`.
    - Landing cell not collapsed; if it has a marker, it must belong to P.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §4.1; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§9–10, 15.3, 15.4 Q2–Q5.

- **[RR-CANON-R102] Applying a capture segment.**
  - When executing a legal capture segment:
    - Process markers along the path exactly as in non-capture movement (RR-CANON-R092).
    - Move the attacking stack from `from` to `landing`.
    - Pop the top ring from the target stack and append it to the **bottom** of the attacking stack's ring list.
    - Recompute stackHeight and capHeight for both stacks.
    - If the target stack becomes empty, remove it.
    - If landing on a P marker, remove the marker, then land and immediately eliminate the top ring of the attacking stack, crediting it to P (before line/territory processing).
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §4.2; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§9–10, 15.3, 15.4 Q4–Q6.

- **[RR-CANON-R103] Chain overtaking rule.**
  - Once P performs any legal overtaking capture segment in a turn, overtaking capture becomes a **chain**:
    - From the new position of the capturing stack, generate all legal capture segments per RR-CANON-R101.
    - If none exist, the chain ends.
    - If one or more exist, P **must** choose one and execute it.
    - P may choose a segment that leads to a position with no further legal captures, thereby ending the chain early, even if other available segments would allow more captures.
  - Chains may:
    - Change direction between segments.
    - Reverse 180° over previously targeted stacks.
    - Capture multiple rings from the same stack over multiple segments, as long as legality is preserved each time.
  - Line formation and Territory disconnection created mid-chain are **not processed** until the entire chain (and movement phase) ends.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §4.3; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§4.3, 10.3, 15.3, 15.4 Q5, Q9, Q12.

---

## 6. Lines and Graduated Line Rewards

- **[RR-CANON-R120] Line definition and detection.**
  - A line for player P is a maximal sequence of positions `[p0, ..., pk]` such that:
    - Each `pi` currently contains a **marker** of P (no stacks, no collapsed spaces).
    - Each `pi` is adjacent to `pi+1` along a single line axis (per `lineAdjacency`).
    - No empty cells, opponent markers, stacks, or collapsed spaces appear between positions in the sequence.
  - A line is **eligible** if `len = k+1 ≥ lineLength` for the board type.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §5.1; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§11.1, 15.3, 16.3, 16.9.4.3.

- **[RR-CANON-R121] Line processing order.**
  - After movement and any captures:
    - Compute all eligible lines in the current state.
    - While at least one eligible line remains:
      - Moving player P chooses one eligible line L to process.
      - Apply collapse/elimination per RR-CANON-R122.
      - Recompute eligible lines; continue until none remain or P cannot pay required eliminations.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §5.2; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§4.5, 11.2–11.3, 15.4 Q7.

- **[RR-CANON-R122] Line collapse and elimination.**
  - Let requiredLen = `lineLength` for this board type (3 or 4).
  - For a chosen eligible line of length len:
    - **Case 1: len == requiredLen.**
      - Collapse **all** markers in the line to collapsed spaces owned by P.
      - P **must** eliminate either:
        - a single standalone ring they control, or
        - the entire cap of a stack they control.
      - If P controls no eligible ring or cap (which cannot occur as long as P controls at least one stack), this state should be treated as unreachable in a correct engine; see Section 13.
    - **Case 2: len > requiredLen.**
      - P chooses Option 1 or Option 2:
        - **Option 1 (max Territory):** collapse **all len** markers in the line and then eliminate one ring or one cap as above.
        - **Option 2 (ring preservation):** choose any contiguous subsegment of length requiredLen within the line; collapse exactly those requiredLen markers; eliminate **no** rings.
  - After each processed line, update all counters and recompute lines.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §5.3; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§4.5, 11.2–11.3, 15.4 Q7, Q22.

---

## 7. Territory Disconnection and Region Processing

- **[RR-CANON-R140] Region discovery.**
  - Using `territoryAdjacency`, compute all maximal regions of non-collapsed cells as in RR-CANON-R040.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §6.1; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§12.1–12.2, 16.8, 16.9.6.

- **[RR-CANON-R141] Physical disconnection criterion.**
  - A region R is **physically disconnected** if every adjacency path from any cell in R to any non-collapsed cell outside R must cross only:
    - Collapsed spaces (any color), and/or
    - Board edge (off-board), and/or
    - Markers belonging to exactly **one** player B (the border color).
  - All non-collapsed marker cells that participate in blocking paths must belong to B; if markers of multiple players are required to block all paths, R is not physically disconnected.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §6.1; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§12.1–12.2, 15.4 Q15, Q20.

- **[RR-CANON-R142] Color-representation criterion.**
  - Let ActiveColors be the set of players that currently have at least one ring anywhere on the board (in any stack).
  - Let RegionColors be the set of players that control at least one stack (by top ring) whose cell lies in R.
  - R is **color-disconnected** if RegionColors is a **strict subset** of ActiveColors.
  - If RegionColors == ActiveColors, R is **never** treated as disconnected, regardless of physical barriers.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §6.2; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§12.1–12.2, 15.4 Q10, Q15.

- **[RR-CANON-R143] Self-elimination prerequisite.**
  - For each candidate region R and moving player P:
    - Consider the hypothetical state where all rings in R are eliminated.
    - If P would still control at least one ring or stack cap **outside R** in that hypothetical state, then P may process R (subject to paying a self-elimination cost).
    - Otherwise, P is **not allowed** to process R at this time; R remains unchanged.
  - Each self-elimination ring/cap outside R can be used to pay for **one** region only. After processing a region, recompute the prerequisite for remaining regions.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §6.3; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§12.2, 12.3, 15.4 Q15, Q23.

- **[RR-CANON-R144] Region processing order.**
  - After all line processing completes:
    - Compute all regions R that are both physically disconnected and color-disconnected.
    - In any order chosen by P:
      - For each R, check the self-elimination prerequisite.
      - If it fails, skip R.
      - If it passes, process R per RR-CANON-R145.
      - After processing each region, recompute candidate regions (new disconnections may appear).
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §6.4; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§12.2–12.3, 16.8–16.9.8.

- **[RR-CANON-R145] Region collapse and elimination.**
  - For a region R being processed for moving player P:
    1. **Collapse interior:**
       - For every cell in R, remove any stacks or markers and set the cell as a collapsed space owned by P.
    2. **Collapse border markers of the single border color B:**
       - Identify all markers of color B that lie on the border of R and are part of at least one blocking path used to establish physical disconnection.
       - Collapse those markers to P-owned collapsed spaces (they become part of P's Territory).
    3. **Eliminate internal rings:**
       - For every stack originally in R, remove all rings and credit them as eliminated to P.
    4. **Mandatory self-elimination:**
       - Eliminate one ring or entire cap from a P-controlled stack that lies outside R, as required by the self-elimination prerequisite.
    5. Update all counts and recompute regions.
  - All eliminated rings from steps 3 and 4 are credited to P.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §6.4; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§12.2–12.3, 16.8–16.9.8.

---

## 8. Victory and Game End

- **[RR-CANON-R170] Ring-elimination victory.**
  - After completing all phases of a player's turn (including line and Territory processing), if any player P has `P.eliminatedRingsTotal ≥ victoryThreshold`, that player wins immediately by elimination.
  - Multiple players cannot simultaneously satisfy this because total eliminated rings cannot exceed 100% of rings in play.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md:409) §7.1; [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1204) §§13.1, 16.9.4.5.

- **[RR-CANON-R171] Territory-control victory.**
  - After a full turn, if any player P has `territorySpaces[P] ≥ territoryVictoryThreshold`, that player wins immediately by Territory.
  - Multiple players cannot simultaneously satisfy this because thresholds are >50% of total spaces.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md:417) §7.2; [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1220) §§13.2, 16.2.

- **[RR-CANON-R172] Last-player-standing victory.**
  - Last-player-standing is a third formal victory condition, alongside ring-elimination (RR-CANON-R170) and Territory-control (RR-CANON-R171).
  - For the purposes of this rule, a **real action** for a player P on their own turn means any legal:
    - ring placement (RR-CANON-R080–R082),
    - non-capture movement (RR-CANON-R090–R092), or
    - overtaking capture segment or chain (RR-CANON-R100–R103),
      and **does not** include pure forced-elimination actions from RR-CANON-R100.
  - A **full round of turns** is one contiguous cycle of turns in player order in which each non-eliminated player receives exactly one turn.
  - A player P wins by last-player-standing if all of the following hold:
    - There exists at least one full round of turns such that:
      - On each of P's turns in that round, P has at least one legal real action available at the start of their action; and
      - On every other player's turns in that same round, those players have **no** legal real action available at the start of their action (they may have only forced-elimination actions, or no legal actions at all); and
    - Immediately after that round completes (including all line and Territory processing), at the start of P's next turn P is still the only player who has any legal real action.
  - Players who still have rings on the board (including rings buried inside mixed-colour stacks) but whose only legal actions on their turns are forced eliminations, or who have no legal actions at all, are **temporarily inactive** for last-player-standing purposes. They prevent an LPS victory until they have been continuously in this "no real actions" state on each of their turns throughout at least one qualifying full round as above. If any such player regains a real action (for example, by gaining control of a stack when a buried ring of theirs becomes the top ring) before the condition above has been met, the last-player-standing condition resets and must be re-satisfied from that point.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md:424) §7.3; [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1228) §§13.3, 16.6, 16.9.4.5.

- **[RR-CANON-R173] Global stalemate and tiebreaks.**
  - If **no** player has any legal placement, movement, capture, or forced elimination available (global stalemate):
    - This can occur only once no stacks remain on the board; otherwise forced elimination would be available to someone.
    - Convert all rings in hand for each player into eliminated rings credited to that player.
    - Rank players by, in order:
      1. Most collapsed spaces (Territory).
      2. If tied, most eliminated rings (including converted rings in hand).
      3. If still tied, most markers on the board.
      4. If still tied, last player to have completed a valid turn action.
    - Highest rank wins.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §7.4; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§13.4, 15.4 Q11.

---

## 9. Randomness and Determinism

- **[RR-CANON-R190] No randomness.**
  - RingRift is a perfect-information game with no chance elements. All state transitions are deterministic given:
    - The current `GameState`.
    - The player's choice of legal action and any required tie-breaking choices (e.g., which line/region to process next, which self-elimination to perform).
  - References: [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §1.1; [`ringrift_compact_rules.md`](ringrift_compact_rules.md) preamble, §8–9.

- **[RR-CANON-R191] Progress and termination invariant.**
  - Define global measure `S = M + C + E` where:
    - M = number of markers on the board.
    - C = number of collapsed spaces.
    - E = total eliminated rings credited to any player.
  - Under all legal moves and forced eliminations:
    - Any movement or capture increases S by at least 1.
    - Collapsing markers to Territory preserves M + C.
    - Any elimination strictly increases E.
    - Forced elimination strictly increases E.
    - No rule ever decreases C or E.
  - Because S is bounded above by `2 * totalSpaces + totalRingsInPlay`, only finitely many real actions are possible and every legal game terminates in finite time.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §9; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §13.5.

---

## 10. Optional / Variant Rules and Version Differences

The canonical ruleset is **parameterized** by board type. No separate optional rules are defined here; instead, the board-type configuration (RR-CANON-R001) encodes all standard variants.

Key version-dependent parameters:

- Line length (`lineLength`):
  - `square8`: 3.
  - `square19`, `hexagonal`: 4.
- Adjacency:
  - Movement and lines use Moore for square boards, hex adjacency for hex.
  - Territory uses Von Neumann for square boards, hex adjacency for hex.
- Rings per player, total rings, and derived victory thresholds follow RR-CANON-R001, R061–R062.

Any localized prose that diverges from these parameterized values in the Complete Rules (e.g., examples that use "4 or 5" markers as the required line length) is treated as **non-canonical** in favor of the table in the Compact Spec and RR-CANON-R001.

---

## 11. Relationship Between Complete and Compact Rules

### 11.1 General relationship

- The Compact Spec is designed as an implementation-oriented restatement of the Complete Rules.
- In almost all cases, it:
  - Preserves the semantics of the Complete Rules.
  - Removes narrative redundancies and clarifies edge cases.
  - Parameterizes version differences cleanly via board-type configuration.
- Canonical priority:
  - When the two sources are clearly consistent, this spec simply restates their common meaning.
  - When the Complete Rules and Compact Spec appear to diverge, this spec generally follows the **Compact Spec** where it is more precise or explicitly labeled as implementation guidance.
  - Explicit exceptions and judgment calls are listed below.

### 11.2 Notable differences

Below are the most important differences, categorized by type, with canonical interpretation.

1. **Movement landing flexibility (non-capture).**
   - Complete Rules: Older text in §10.2 claims that capture landing flexibility "differs from non-capture movement in 19×19/Hex", implying a first-valid-landing constraint for some versions.
   - Compact Spec: §3.2 unifies the rule: **all** versions allow landing on any valid space beyond markers meeting the distance requirement; you are not required to stop at the first such space.
   - Category: **Genuine contradiction** (outdated prose vs updated rule).
   - Canonical: Follow the unified rule from the Compact Spec (RR-CANON-R091–R092). The note in the Complete Rules is treated as obsolete.

2. **Line-length wording ("4 or 5" vs required length).**
   - Complete Rules: §4.5 and some examples sometimes speak of "lines of exactly the required length (4 or 5)" in a way that conflates 8×8 and 19×19 requirements.
   - Compact Spec: §5 consistently uses `lineLength` per board type (3 for square8, 4 for others), and treats any len > lineLength as "overlength".
   - Category: **Simplification / correction**.
   - Canonical: Use parameterized `lineLength` only; RR-CANON-R120–R122 control.

3. **Graduated line rewards threshold on 8×8.**
   - Complete Rules: Some prose suggests the "graduated rewards" choice on 8×8 begins at 4+ markers; elsewhere it mentions 5+ as the start of the more interesting tradeoff.
   - Compact Spec: Treats any len > lineLength (i.e., >3 on 8×8) as overlength with Options 1 and 2.
   - Category: **Simplification** (compact version generalizes rule).
   - Canonical: Follow the general rule: any len > lineLength is overlength with Options 1 and 2 (RR-CANON-R122).

4. **Forced elimination impossibility edge case.**
   - Complete Rules: FAQ Q24 mentions the possibility that a player "cannot perform" forced elimination because "all caps have already been eliminated", suggesting a skip.
   - Compact Spec: §2.3 assumes that if a player controls a stack, they always have at least one ring in the cap to eliminate.
   - Category: **Genuine contradiction**, but the Complete Rules scenario is structurally impossible under the stack model.
   - Canonical: Treat forced elimination as always applicable whenever a player controls any stack (RR-CANON-R100), and treat the Q24 skip example as unreachable commentary.

5. **Territory border marker collapse scope.**
   - Complete Rules: §12.2 states that only spaces occupied by markers of the single color that "actually forms the disconnecting border" are collapsed, but does not fully define "actually forms".
   - Compact Spec: §6.4 clarifies that border markers "that participate in the disconnection (i.e., contiguous along the border path)" are collapsed.
   - Category: **Alternative presentation / clarification**.
   - Canonical: Follow the Compact Spec interpretation (RR-CANON-R145): collapse only those border markers of the single border color that are part of at least one minimal blocking path around the region.

Overall, **[`ringrift_compact_rules.md`](ringrift_compact_rules.md)** is treated as the authoritative source when conflicts arise, except where noted otherwise in this section.

---

## 12. Ambiguities and Underspecified Behaviors

This section lists remaining areas where the prose rules allow multiple reasonable readings. For each, we state plausible interpretations and the canonical choice used in RR-CANON rules.

### 12.1 Forced elimination reachability

- **Sources.** Complete FAQ Q24; Complete §4.4; Compact §2.3.
- **Ambiguity.** FAQ Q24 suggests a player might "control stacks but all caps have already been eliminated", making forced elimination impossible. Under the stack model, any controlled stack always has at least one top ring of the controlling color, hence capHeight ≥ 1.
- **Interpretation A.** Allow a hypothetical state where a player "controls" a ringless stack or a stack whose top ring is not of their color, blocking forced elimination and forcing a pass.
- **Interpretation B (canonical).** Stack control is always defined by the actual top ring; thus any controlled stack always has at least one ring of the controlling player, and forced elimination is always applicable when a player controls any stack and has no other legal actions.
- **Canonical choice.** Interpretation B, encoded in RR-CANON-R100. FAQ Q24's skip case is treated as non-realizable in a valid engine state.

### 12.2 Border-marker collapse extent

- **Sources.** Complete §12.2, §16.8; Compact §6.1–6.4.
- **Ambiguity.** When collapsing a disconnected region, do **all** markers of the single border color collapse, or only those that are part of the minimal enclosing barrier?
- **Interpretation A.** Collapse all markers of the border color that are orthogonally adjacent to the region, even if some are not necessary for disconnection.
- **Interpretation B (canonical).** Collapse exactly those markers of the border color that participate in at least one blocked path from the region to the rest of the board (i.e., are on some minimal enclosing "border path").
- **Canonical choice.** Interpretation B, as formalized in RR-CANON-R145, consistent with Compact §6.4. This avoids surprising collapses of distant, non-enclosing markers.

### 12.3 Overlength line processing when elimination is impossible

- **Sources.** Complete §11.2–11.3, FAQ Q7; Compact §5.2–5.3.
- **Ambiguity.** If a player has overlength lines but no eliminable rings/caps (e.g., all rings are buried under opposing caps), can they still partially collapse lines using Option 2?
- **Interpretation A (strict).** Any overlength line processing still requires the ability to eliminate a ring, even if Option 2 is chosen, because the "line processing" concept is globally guarded by elimination capability.
- **Interpretation B (canonical).** For overlength lines, Option 2 explicitly allows collapsing exactly `requiredLen` markers with **no elimination**, regardless of the player's ability to eliminate rings otherwise.
- **Canonical choice.** Interpretation B, as encoded in RR-CANON-R122 and supported by FAQ Q7: players may always use Option 2 for overlength lines even when short on rings.

### 12.4 Simultaneous multi-region disconnections

- **Sources.** Complete §§12.2–12.3, 16.9.8; Compact §6.3–6.4.
- **Ambiguity.** When a single move simultaneously disconnects multiple regions, can the moving player choose any subset to process, or must they process all disconnectable regions if they can afford self-elimination?
- **Interpretation A.** Once a region qualifies and passes the self-elimination prerequisite, processing it is mandatory.
- **Interpretation B (canonical).** Disconnected regions are processed "in any order chosen by the moving player", and nothing requires processing all eligible regions; skipping a region is allowed.
- **Canonical choice.** Interpretation B. RR-CANON-R144 treats region processing as optional per-region, constrained only by the prerequisite. This matches the emphasis on player choice and strategic chain-reaction control in the Complete Rules examples.

### 12.5 Ownership of eliminated rings in chain reactions

- **Sources.** Complete §12.2–12.3, 16.9.7–16.9.8; Compact §6.4, §7.1.
- **Ambiguity.** When multiple regions collapse in a chain reaction across several turns, how are eliminated rings attributed if subsequent region collapses are "automatic" results of prior collapses?
- **Interpretation A.** Attribute each region's eliminated rings to the player whose move **directly** caused that region to become newly disconnected (may differ from the player who caused earlier collapses).
- **Interpretation B (canonical).** Attribute eliminated rings to the player whose turn is currently being processed and who chooses to process the region (i.e., who pays the self-elimination cost).
- **Canonical choice.** Interpretation B, encoded in RR-CANON-R145 and RR-CANON-R170, and consistent with Compact §6.4 / §7.1: the "moving player" who processes a region is always the credited cause.

---

## 13. Summary for Downstream Agents

- The **core, authoritative transition rules** are those encoded as RR-CANON rules in Sections 1–9.
- Board-type differences are fully captured by RR-CANON-R001 and the parameterized use of `lineLength`, adjacency, and ring counts.
- Victory, last-player-standing, and stalemate behavior follow RR-CANON-R170–R173; no other implicit termination conditions exist.
- When discrepancies between prose sources arise, treat [`ringrift_compact_rules.md`](ringrift_compact_rules.md) as primary, except for the explicitly documented judgment calls in Section 11–12.
- The progress invariant RR-CANON-R191 guarantees finite termination and is useful for static and dynamic verification.

This canonical spec is intended to be stable; future rule changes should be expressed as additions or replacements of specific RR-CANON rule IDs, preserving traceability back to the original narrative and compact documents.
