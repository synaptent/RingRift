# RingRift Canonical Rules Specification

> **Doc Status (2025-12-06): Active**
>
> - This document is the **rules-level canonical spec** for RingRift semantics and the single source of truth for rules behaviour. It normalizes the narrative sources into precise, implementation-ready constraints.
> - It defines **RR-CANON-RXXX** rule IDs for core invariants, resources, turn/phase structure, line/territory semantics, and victory conditions.
> - It intentionally **does not** re-specify the engine API or Move/decision/WebSocket lifecycle. For those, see:
>   - [`docs/CANONICAL_ENGINE_API.md` §3.9–3.10](docs/CANONICAL_ENGINE_API.md) for `Move`, `PendingDecision`, `PlayerChoice*`, WebSocket payloads, and the orchestrator-backed decision loop.
>   - `src/shared/types/game.ts` and `src/shared/engine/orchestration/types.ts` for the canonical type definitions.
> - **Single Source of Truth (SSoT):** The canonical rules defined in this document are the **ultimate authority** for RingRift game semantics. All implementations—TypeScript shared engine, Python AI service, client sandbox, replay systems, and any future engines—must derive from and faithfully implement these canonical rules.
> - **Implementation hierarchy:**
>   - **This file** (`RULES_CANONICAL_SPEC.md`) is the _normative_ rules SSoT: when behaviour is in doubt, the RR‑CANON‑RXXX rules here win.
>   - **TS shared engine** (`src/shared/engine/**`) is the _primary executable derivation_ of this spec. If the TS engine and this document ever disagree, that is a bug and the TS engine code must be updated to match the canonical rules described here.
>   - **Python AI service** (`ai-service/app/**`) is a _host adapter_ that must mirror the canonical rules. If Python disagrees with the canonical rules or the validated TS engine behaviour, Python must be updated—never the other way around.
>   - **Client sandbox, replay, and other hosts** similarly derive from the canonical rules via the shared engine; they must not introduce independent rules semantics.
>
> **How to use this doc.** Start with the quick map below, then jump to the relevant section; keep the Compact Spec open for concise semantics and use the Complete Rules for narrative/examples.
> **Change log anchor.** When rules change, record the delta in §14 (“Change log & traceability”) with pointers into commits/tests that enforced the change.
>
> **Purpose.** This document is a normalization of the RingRift rules for engine/AI implementation and verification. It reconciles [`ringrift_complete_rules.md`](ringrift_complete_rules.md) ("Complete Rules") and [`docs/rules/ringrift_compact_rules.md`](docs/rules/ringrift_compact_rules.md) ("Compact Spec") into a single canonical, implementation-ready ruleset.

The canonical rules here are binding whenever the two source documents diverge. For each rule we provide:

- A stable rule identifier `RR-CANON-RXXX`.
- A precise statement of the rule.
- Applicability (all versions vs version-specific).
- References into the Complete and Compact docs.

**Quick navigation**

- §1–2 Resources & setup
- §3 Turn/phase structure
- §4 Movement & captures
- §5 Lines
- §6 Territory
- §7 Victory & LPS
- §8–9 Hashing & invariants
- §10–12 Edge cases & judgment calls
- §14 Change log & traceability

The Compact Spec is generally treated as primary for formal semantics, and the Complete Rules as primary for examples, motivation, and prose. Explicit exceptions and judgment calls are documented in Sections 12 and 13.

---

## 0. Conventions

- **Players.** 2–4 players; 3-player games are the default in all examples.
- **Board types.**
  - `square8`: 8×8 orthogonal grid.
  - `square19`: 19×19 orthogonal grid.
- `hexagonal`: hex board with radius 12 (13 cells per side).
- **Notation.**
  - "Stack" = one or more rings in a single cell.
  - "Cap" = all consecutive rings from the top of a stack belonging to the controlling player.
  - "Eliminated ring" = ring permanently removed from the board and credited to a player.
  - "Collapsed space" = permanently claimed territory cell, impassable to movement and capture.
- **Version parameters.** Unless otherwise stated, all numeric length thresholds (line length, victory thresholds, etc.) come from the board-type configuration in RR-CANON-R001.
- **2-player balancing (pie rule).** In 2‑player games only, after Player 1’s first completed turn from the canonical empty starting position, Player 2 has a one-time option to **swap sides** (RR‑CANON‑R180): seats and colours for players 1 and 2 are exchanged with no change to board geometry, and it remains Player 2’s turn.

---

## 1. Entities and Components

### 1.1 Board types and geometry

- **[RR-CANON-R001] Board type configuration.**
  - For each `BoardType ∈ { square8, square19, hexagonal }`, define:
    - `size`, `totalSpaces`, `ringsPerPlayer`, `lineLength`, `movementAdjacency`, `lineAdjacency`, `territoryAdjacency`, `boardGeometry` exactly as in the table in the Compact Spec (§1.1).
  - References: [`docs/rules/ringrift_compact_rules.md`](docs/rules/ringrift_compact_rules.md) §1.1; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§1.2.1, 16.10.

- **[RR-CANON-R002] Coordinate systems.**
  - Square boards use integer coordinates `(x,y)` with `0 ≤ x,y < size`.
  - Hex board uses cube coordinates `(x,y,z)` with `x + y + z = 0` and `max(|x|,|y|,|z|) ≤ size-1`.
  - References: [`docs/rules/ringrift_compact_rules.md`](docs/rules/ringrift_compact_rules.md) §1.1.

- **[RR-CANON-R003] Adjacency relations.**
  - Movement and line directions use `movementAdjacency` / `lineAdjacency` from RR-CANON-R001.
  - Territory connectivity uses `territoryAdjacency` from RR-CANON-R001.
  - Straight-line rays along these directions are used for movement, capture, and line detection.
  - References: [`docs/rules/ringrift_compact_rules.md`](docs/rules/ringrift_compact_rules.md) §§1.2, 3, 4, 5; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§2.1, 3.1, 8, 11, 12, 16.9.4.1.

### 1.2 Players and identifiers

- **[RR-CANON-R010] Player identifiers.**
  - Each player has a unique `PlayerId`.
  - Exactly 2–4 players participate; 3 is the default.
  - References: [`docs/rules/ringrift_compact_rules.md`](docs/rules/ringrift_compact_rules.md) §1.3, §7; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§1.1, 1.2, 13, 15.1, 19×19/8×8 setup sections.

### 1.3 Rings, stacks, and control

- **[RR-CANON-R020] Rings per player (own-colour supply cap).**
  - For each board type, each player P has a fixed personal supply of rings of P's **own colour**:
    - `square8`: 18 rings.
    - `square19`: 36 rings.
  - `hexagonal`: 48 rings.
  - At all times, the total number of rings of P's colour that are **in play** (on the board in any stack, regardless of which player currently controls those stacks, plus in P's hand) must be ≤ this `ringsPerPlayer` value for the chosen board type.
  - Rings of other colours that P has captured and that are buried in stacks P controls **do not** count against P's `ringsPerPlayer` cap; they continue to belong, by colour, to their original owners for conservation, elimination, and victory accounting.
  - Eliminated rings of P's colour are permanently out of play and do not refresh or expand P's supply beyond `ringsPerPlayer`; they only change how much of that fixed supply is currently eliminated versus still in play.
  - References: [`docs/rules/ringrift_compact_rules.md`](docs/rules/ringrift_compact_rules.md:18) §1.1, §7.1; [`ringrift_complete_rules.md`](ringrift_complete_rules.md:340) §§1.2.1, 3.2.1, 16.3–16.4, 16.9.2.

- **[RR-CANON-R021] Stack definition.**
  - A stack is an ordered sequence of one or more rings on a single board cell.
  - Rings are ordered bottom→top.
  - A cell may contain **either** a stack, **or** a marker, **or** be empty, **or** be collapsed; never more than one of these simultaneously.
  - References: [`docs/rules/ringrift_compact_rules.md`](docs/rules/ringrift_compact_rules.md) §1.3; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§5–7.

- **[RR-CANON-R022] Control and cap height.**
  - `controllingPlayer` of a stack is the color of its top ring.
  - `stackHeight = rings.length`.
  - `capHeight` is the number of consecutive rings from the top that belong to `controllingPlayer`.
  - Control changes whenever the top ring changes color (due to overtaking or elimination).
  - References: [`docs/rules/ringrift_compact_rules.md`](docs/rules/ringrift_compact_rules.md) §1.3; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§5.1–5.3, 7.2, 15.4 Q16.

- **[RR-CANON-R023] Stack mutation operations.**
  - Legal stack mutations are limited to:
    - Adding rings at the **top** via placement.
    - Adding rings at the **bottom** via overtaking capture.
    - Removing the **top ring** of a stack via overtaking capture.
    - Removing an entire cap (all consecutive top rings of the controlling color) via forced elimination, line processing, or region processing.
    - Removing all rings in a region during territory collapse.
  - Stacks may **never** be split or reordered in any other way.
  - References: [`docs/rules/ringrift_compact_rules.md`](docs/rules/ringrift_compact_rules.md) §§2–4, 5, 6; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§5–7, 9–12, 15.4 Q1.

### 1.4 Markers and collapsed spaces

- **[RR-CANON-R030] Marker definition.**
  - A marker is a color token occupying a cell that is not currently a stack or collapsed space.
  - Each marker belongs to exactly one player.
  - Markers are created only as departure markers from movement/capture (RR-CANON-R082).
  - Markers may be flipped to another player or collapsed into Territory as movement/capture passes over them.
  - References: [`docs/rules/ringrift_compact_rules.md`](docs/rules/ringrift_compact_rules.md) §§1.3, 3.2, 4.1; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§3.2.2, 8, 11, 12, 16.5–16.6.

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
  - **Crediting rule:** Whenever any turn action causes a space to collapse (via marker path processing, line processing, or territory region processing), the acting player's `territorySpaces` must be incremented by the number of newly collapsed spaces.
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
    - Turn/phase: `currentPlayer`, `currentPhase` ∈ { `ring_placement`, `movement`, `capture`, `chain_capture`, `line_processing`, `territory_processing`, `forced_elimination` } during active play, and `game_over` once victory/stalemate is reached. `game_over` is terminal-only and MUST NOT be used for move recording or phase traversal.
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
    3. Optional start of **overtaking capture from the moved stack's landing position only** (see RR-CANON-R093), then mandatory chain continuation while legal.
    4. **Line processing** (zero or more lines).
    5. **Territory disconnection processing** (zero or more regions).
    6. **Forced elimination** (conditional): entered only if P had no actions available in all prior phases (placement, movement, capture, line processing, territory processing) but still controls at least one stack. P must eliminate the entire cap of one controlled stack per RR-CANON-R100.
    7. **Victory / termination check**.
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

- **[RR-CANON-R073] Mandatory phase transitions for currentPhase.**
  - The `currentPhase` field in `GameState` must change at specific points during turn processing:
    - **ring_placement → movement:** After a `place_ring` or `skip_placement` move, `currentPhase` MUST change to `movement` for the same player.
    - **movement → capture:** After a non-capture movement (`move_stack` or `move_ring`), if legal capture segments exist from the landing position per RR-CANON-R093, `currentPhase` MUST change to `capture`. If no such captures exist, proceed to `line_processing`.
    - **capture → chain_capture:** After executing an `overtaking_capture` segment, if additional legal capture segments exist from the new landing position per RR-CANON-R103, `currentPhase` MUST change to `chain_capture` and the chain must continue.
    - **chain_capture → line_processing:** After executing a `continue_capture_segment`, if no additional legal captures exist from the new position, `currentPhase` MUST change to `line_processing`.
    - **capture → line_processing:** After executing an `overtaking_capture` segment where no chain continuation is required (no further captures from landing), `currentPhase` MUST change to `line_processing`.
    - **movement → line_processing:** After a non-capture movement where no capture segments exist from the landing position, `currentPhase` MUST change to `line_processing`.
    - **line_processing → territory_processing:** After all lines for the current player are processed (or none existed), `currentPhase` MUST change to `territory_processing`.
    - **territory_processing → forced_elimination:** After all territory regions are processed (or none existed), if P had no actions available in any prior phase (no placement, no movement, no capture, no line processing, no territory processing) but P still controls at least one stack, `currentPhase` MUST change to `forced_elimination`.
    - **territory_processing → ring_placement/movement (next player):** After all territory regions are processed (or none existed), if P performed at least one action in any prior phase OR P has no controlled stacks, the turn ends and the next player's `currentPhase` is set to `ring_placement` (if `ringsInHand > 0`) or `movement` (if `ringsInHand == 0` but stacks exist).
    - **forced_elimination → ring_placement/movement (next player):** After P executes the `forced_elimination` move, victory checks are performed and the turn ends.
  - These transitions are not optional; any engine implementation that does not perform them violates the canonical rules.
  - Note: The `skip_capture` move type exists for cases where a player declines an optional capture opportunity; it explicitly transitions from `capture` to `line_processing` without executing a capture.
  - References: RR-CANON-R070, RR-CANON-R093, RR-CANON-R100, RR-CANON-R103; TS `turnOrchestrator.ts` `processPostMovePhases` and `phaseStateMachine.ts`.

- **[RR-CANON-R074] Explicit recording of all turn actions and voluntary skips.**
  - At the rules level, a **turn action** for player P is any state change that:
    - places one or more rings for P,
    - moves a stack or ring controlled by P (capture or non-capture),
    - processes a line owned by P (RR-CANON-R120–R122),
    - processes a Territory region or applies required self-elimination for P (RR-CANON-R140–R145), or
    - performs a forced elimination or explicit elimination decision (RR-CANON-R100, RR-CANON-R205).
  - A **voluntary forgo decision** is any choice by P to decline an otherwise legal action that the rules allow them to take at that point in the turn (for example, `skip_placement` when placements are legal, or `skip_capture` when a capture is optional).
  - Canonical constraint:
    - Whenever P performs a turn action **or** voluntarily forgoes an available action, that decision MUST be represented as an explicit, player-visible move or choice in the game record and engine APIs (e.g., `place_ring`, `move_stack`, `overtaking_capture`, `process_line`, `choose_line_reward`, `process_territory_region`, `eliminate_rings_from_stack`, `skip_placement`, `skip_capture`, `skip_territory_processing`).
    - Engines and hosts MUST NOT treat such actions or voluntary skips as implicit side effects of other moves or silently "assume" them without recording a corresponding decision.
    - Internal helper steps that are logically equivalent to a rules-level action (for example, auto-collapsing a single exact-length line, auto-processing a single Territory region, or auto-selecting a self-elimination target) are permitted **only** as UX conveniences when they also surface and persist a concrete move/choice in the canonical history.
  - This applies uniformly to:
    - Analogue play (each step is announced and agreed upon by players), and
    - Digital implementations, including AI self-play and parity/training harnesses, which must record every rules-level action and voluntary skip as an explicit move/choice for reproducibility and analysis.

- **[RR-CANON-R075] Canonical replay and between-move semantics.**
  - Canonical recordings (for example, GameReplayDBs used for AI training and TS↔Python parity) are sequences of explicit moves and decisions. For such recordings:
    - **Every phase must be visited and every phase transition must produce a recorded action.** When a turn advances through any phase (`ring_placement`, `movement`, `capture`, `line_processing`, `territory_processing`, `forced_elimination`), the canonical move history must contain an explicit move for that phase—even if no actions are possible. Phase transitions MUST NOT occur silently without a corresponding move in the canonical history.
    - **All players must visit all phases:** Even players who are eliminated, have no material, or otherwise have no chance of making a turn action must still traverse each phase and record the appropriate "no action" move. This ensures replay consistency and enables deterministic state reconstruction from move history alone.
    - **Recording requirements—distinguishing "no action possible" from "voluntary skip":**
      - **No actions possible (forced no-op):** When a player enters a phase with no possible actions (e.g., no regions eligible for processing in `territory_processing`, no lines formed in `line_processing`, no legal moves in `movement`), the canonical history must record a distinct "no action" move type such as `no_territory_action`, `no_line_action`, `no_movement_action`, etc. This semantically indicates the player had no choice.
      - **Voluntary skip when choices exist:** When a player has one or more available actions but chooses to skip (e.g., `skip_placement` when ring placement is legal, `skip_capture` when an optional capture exists, `skip_territory_processing` when regions could be processed), the voluntary skip must be recorded as a distinct "skip" move type. This semantically indicates the player made an active choice to forgo available actions.
      - **Single choice available:** When a player has exactly one legal action in a phase (e.g., one region to process, one line to collapse, one legal movement), that action must still be recorded as an explicit move in the canonical history. The presence of only one choice does not permit omitting the move record. UX conveniences that auto-select the only option are permitted **only if** they also emit the corresponding canonical move.
      - **No implicit phase advancement:** Silent advancement through phases without recording a move violates canonical replay semantics and will cause parity divergence between engine implementations.
    - **No additional line-processing, Territory-processing, or ring elimination may occur "between moves".** Every collapse, elimination, or region/line resolution step must be attributable to a specific, explicit move in the record:
      - `place_ring`, `skip_placement`,
      - `move_stack` / `move_ring`,
      - `overtaking_capture`, `continue_capture_segment`,
      - `process_line`, `choose_line_reward`,
      - `process_territory_region`, `choose_territory_option`,
      - `eliminate_rings_from_stack`, or
      - a host-level `forced_elimination` action modelled as an explicit move.
    - Replay engines MUST NOT inject extra collapses, region resolutions, or forced eliminations solely as a consequence of "advancing phases" or "resolving ANM" when consuming a canonical recording; if such work is required by the rules, it must appear as explicit moves/choices in the recording itself.
  - Hosts are free to implement richer UX flows (for example, auto-processing a single exact-length line for a human player) **so long as** those flows still emit the corresponding canonical moves into history. Canonical replay and parity tooling must treat the move history as the sole source of truth for state changes.

- **[RR-CANON-R076] Implementation architecture: core rules layer vs host layer.**
  - All implementations of the rules engine MUST separate concerns into two layers:
    - **Core Rules Layer** (shared engine, Python GameEngine, turn orchestrator):
      - MUST enforce strict no-skipping semantics for all phases per RR-CANON-R075.
      - MUST NOT auto-generate moves of any kind, including no-action moves.
      - When a phase has no available actions, MUST return a "pending decision" requiring an explicit no-action move (e.g., `no_line_action`, `no_territory_action`, `no_movement_action`).
      - Phase transitions occur ONLY in response to explicit moves applied via the public API.
      - This layer is the single source of truth for rules correctness and cross-engine parity.
    - **Host/Adapter Layer** (ClientSandboxEngine, backend game host, AI agents):
      - For **live play UX**: MAY auto-fill required no-action moves when the core layer returns `no_X_action_required` pending decisions. This is a UX convenience that streamlines gameplay by not requiring player input when no choices exist.
      - For **replay/trace mode**: MUST apply explicit moves from the recorded sequence without auto-filling. The recorded history is the authoritative move sequence.
      - Auto-generated moves MUST be added to the canonical history via the standard move-apply API so they are recorded and replayable.
  - This separation ensures:
    - **Parity**: Core rules implementations in TS and Python produce identical state transitions for identical move sequences.
    - **Testability**: Core rules can be tested in isolation without UX concerns.
    - **Correctness**: Silent phase skipping bugs cannot occur in the core layer.
  - Implementation references:
    - TS shared orchestrator: `src/shared/engine/orchestration/turnOrchestrator.ts` returns `PendingDecision` values, including `DecisionType` variants such as `'no_line_action_required'` / `'no_territory_action_required'` for decision phases. For these decision types, `PendingDecision.options` is intentionally empty and hosts must synthesize and apply the corresponding `no_*_action` bookkeeping moves via the public API. For interactive phases (`ring_placement` and `movement`), the orchestrator’s `getValidMoves` now returns **only interactive moves** (e.g., `place_ring`, `skip_placement`, `move_stack`, captures); when no such moves exist, it returns an empty list and hosts are responsible for constructing and applying explicit `no_placement_action` / `no_movement_action` moves. The core TS engine no longer fabricates these moves itself, fully satisfying the “no auto‑generation in core” requirement.
    - Python GameEngine: `ai-service/app/game_engine.py` exposes `get_valid_moves(state, player)` for **interactive-only** legal moves in the current phase, and a separate `get_phase_requirement(state, player)` / `synthesize_bookkeeping_move(requirement, state)` pair for required `NO_*_ACTION` / `FORCED_ELIMINATION` bookkeeping moves. This keeps the core rules layer free of move fabrication while still making canonical bookkeeping moves host-accessible.
    - Host layer: `src/client/sandbox/ClientSandboxEngine.ts`, `src/server/game/turn/TurnEngineAdapter.ts`, and `ai-service/app/training/env.py` consume these pending-decision / phase-requirement surfaces, auto-filling bookkeeping moves for live play and self-play while replay/trace paths apply only the recorded explicit moves.

---

## 4.5 Active-No-Moves & Forced Elimination Semantics (R2xx)

- **[RR-CANON-R200] Global legal actions for the current player.**
  - Let `state` be a valid `GameState` with `gameStatus == ACTIVE` and `currentPlayer = P`.
  - A **global legal action** for P in `state` is any of:
    - A legal **ring placement** move for P, as defined by RR-CANON-R080–R082, on the current board type. This is evaluated using the hypothetical effect of `place_ring` on `state` (respecting `ringsInHand[P]`, `ringsPerPlayer`, and the no-dead-placement rule), and does **not** depend on `currentPhase`; placements are considered globally available whenever they would be legal when P next enters `ring_placement`.
    - A legal **interactive move** for P in the current phase:
      - In `ring_placement`: any legal `place_ring` or explicit `skip_placement` move.
      - In `movement`, `capture`, or `chain_capture`: any legal non-capture movement or overtaking capture segment/chain under RR-CANON-R090–R103.
      - In `line_processing`: any legal `process_line` or `choose_line_reward` decision for P under RR-CANON-R120–R122.
      - In `territory_processing`: any legal `process_territory_region`, `choose_territory_option`, or `eliminate_rings_from_stack` decision for P under RR-CANON-R140–R145.
    - A legal **forced-elimination** action for P under RR-CANON-R100 (see RR-CANON-R205).
  - Global legal actions are defined uniformly for all supported board types (`square8`, `square19`, `hexagonal`) and for all supported player counts (2–4 players).

- **[RR-CANON-R201] Turn-material and temporarily eliminated players.**
  - A player P has **turn-material** in a state if and only if either:
    - P controls at least one stack (there exists a stack whose top ring has P's colour); or
    - `ringsInHand[P] > 0`.
  - A player P is **temporarily eliminated for turn rotation** in a state if P has no turn-material (no controlled stacks and `ringsInHand[P] == 0`), regardless of whether P still owns buried rings of their colour inside other players' stacks.
  - In any state with `gameStatus == ACTIVE`, `currentPlayer` must always be chosen from players who have turn-material. Temporarily eliminated players must not be skipped when rotating the turn order, and should pass through the evaluation of victory or stalemate conditions that immediately terminate the game (RR-CANON-R170–R173).

- **[RR-CANON-R202] Active-no-moves (ANM) state.**
  - Given a valid `GameState` with `gameStatus == ACTIVE` and `currentPlayer = P`, define the predicate `ANM(state, P)` ("active-no-moves for P") as:
    - P has turn-material in `state` (RR-CANON-R201); and
    - P has **no** global legal action in `state` (RR-CANON-R200).
  - Intuitively, an ANM state is one in which the current player still has material but the rules would offer them no way to act, even allowing for placements and forced elimination.

- **[RR-CANON-R203] ANM avoidance and immediate resolution.**
  - From the standard initial setup and under the legal moves defined by RR-CANON-R080–R103, RR-CANON-R120–R122, and RR-CANON-R140–R145, the game may **never remain** in an ANM state for any player:
    - For every reachable state with `gameStatus == ACTIVE` and `currentPlayer = P`, either:
      - `ANM(state, P)` is false (P has at least one global legal action); or
      - The rules-mandated transition sequence from `state` (consisting of automatic phase exits, forced elimination under RR-CANON-R072/RR-CANON-R100, and victory/stalemate checks under RR-CANON-R170–R173) must immediately:
        - either terminate the game, or
        - rotate to a new ACTIVE state whose `currentPlayer` Q satisfies `ANM(nextState, Q) == false`.
  - In particular:
    - No non-terminal **movement**, **capture**, or **chain_capture** phase may leave the game in `gameStatus == ACTIVE` with `ANM(state, P) == true`; if P is blocked with stacks, RR-CANON-R072 and RR-CANON-R100 must apply forced elimination.
    - No non-terminal **line_processing** or **territory_processing** phase may leave the game in `gameStatus == ACTIVE` with `ANM(state, P) == true`; see RR-CANON-R204 for phase-exit requirements.
    - The only legal way for all players simultaneously to have no global legal actions is the global-stalemate shape of RR-CANON-R173, which is resolved immediately into a terminal state.

- **[RR-CANON-R204] Phase-local behaviour and decision-phase exits.**
  - In addition to RR-CANON-R070–R072, the following phase-specific rules must hold for the current player P in any `GameState` with `gameStatus == ACTIVE`:
    - **Line-processing exit.**
      - If `currentPhase == line_processing` and P has no legal line decisions (no `process_line` or `choose_line_reward` moves), the engine must immediately advance out of `line_processing`:
        - to `territory_processing` if any Territory decisions exist for P; otherwise
        - to victory evaluation and turn rotation per RR-CANON-R170–R173.
      - It is illegal to leave the game in `gameStatus == ACTIVE` and `currentPhase == line_processing` with `ANM(state, P) == true`.
    - **Territory-processing exit.**
      - If `currentPhase == territory_processing` and P has no legal Territory decisions (no `process_territory_region`, `choose_territory_option`, or `eliminate_rings_from_stack` moves), the engine must:
        - transition to `forced_elimination` phase if P had no actions in all prior phases (placement, movement, capture, line, territory) but still controls at least one stack; or
        - call end-of-turn, rotate `currentPlayer` to the next player with turn-material (RR-CANON-R201), and evaluate victory per RR-CANON-R170–R173.
      - It is illegal to leave the game in `gameStatus == ACTIVE` and `currentPhase == territory_processing` with `ANM(state, P) == true`.
    - **Forced-elimination phase.**
      - Entered only when P had no actions in all prior phases but still controls stacks.
      - P must execute a `forced_elimination` move that eliminates the entire cap of one controlled stack per RR-CANON-R100.
      - After the `forced_elimination` move, the engine evaluates victory and rotates to the next player.
    - **Other phases.**
      - For `ring_placement`, `movement`, `capture`, and `chain_capture`, ANM avoidance is already enforced by RR-CANON-R070–R072, RR-CANON-R080–R082, and RR-CANON-R090–R103:
        - If P has rings in hand but no legal placements, P must effectively skip placement and proceed to movement.
        - **If P has no rings in hand (`ringsInHand == 0`) but controls stacks, placement is forbidden (RR-CANON-R080) and the engine must immediately transition to `movement` phase and enumerate movement/capture moves for P's controlled stacks.** This scenario commonly arises mid-game when a player has placed all their rings but still has board presence. The `getValidMoves` enumeration must return movement moves, not `skip_placement` (which requires `ringsInHand > 0`).
        - If P controls stacks but has no legal placement, movement, or capture, forced elimination must be applied (RR-CANON-R072/RR-CANON-R100).
        - If P has no turn-material at all, they must not remain `currentPlayer` in an ACTIVE state (RR-CANON-R201).

- **[RR-CANON-R205] Forced-elimination action taxonomy.**
  - Forced elimination appears in two distinct but related forms:
    - **Phase-level forced elimination (`forced_elimination` phase).**
      - Triggered when player P traverses all turn phases (placement, movement, capture, line, territory) with no available actions but still controls at least one stack.
      - The engine transitions to the `forced_elimination` phase and P must execute a `forced_elimination` move that eliminates the entire cap of some P-controlled stack per RR-CANON-R100.
      - Phase-level forced elimination is treated as a **global legal action** for ANM purposes (RR-CANON-R200–R203) but is **not** a "real action" for Last-Player-Standing under RR-CANON-R172.
      - The `forced_elimination` phase is the canonical location for recording this action, ensuring clear phase semantics and replay consistency.
    - **Explicit elimination decisions (during other phases).**
      - During line processing (RR-CANON-R120–R122), elimination of a ring or cap as a line reward is represented as an explicit decision (`eliminate_rings_from_stack`) tied to the processed line.
      - During Territory processing (RR-CANON-R140–R145), mandatory self-elimination from a stack outside the processed region is likewise represented as an explicit `eliminate_rings_from_stack` decision.
      - These explicit elimination moves are phase-local **interactive actions** for P and therefore count as global legal actions under RR-CANON-R200.
  - In all cases, any forced elimination or explicit elimination must remove at least one ring belonging to the acting player and must increase the global eliminated-ring count in RR-CANON-R060–R061 and RR-CANON-R191.

- **[RR-CANON-R206] Forced-elimination target choice (interactive choice + deterministic policies).**
  - When RR-CANON-R100 requires P to perform a forced elimination, the **rules-level requirement** is that P may choose **any** eligible elimination target they control, where an eligible target is either:
    - a standalone ring (a height‑1 stack whose top ring has colour P); or
    - the entire cap (all consecutive top rings of colour P) of a stack they control.
  - RR-CANON-R100 deliberately does **not** constrain which eligible target must be chosen; any such choice yields a rule-legal successor state. This applies uniformly to:
    - Analogue play, in which P explicitly chooses a stack or standalone ring; and
    - Digital implementations, in which the acting agent (human or AI) must be able to choose which eligible stack/ring to eliminate.
  - Canonical constraint for digital hosts:
    - Hosts must expose forced elimination as an **interactive decision** for the acting agent (for example, via an explicit "choose stack to eliminate from" choice surface) whenever more than one eligible target exists.
    - Hosts **must not** unilaterally apply a hidden deterministic tie-breaker (such as "smallest cap first" or "first in scan order") to select the target on behalf of a human player.
    - Deterministic policies are permitted only as part of an **agent’s own** decision-making (for example, an AI, bot, or invariant harness that always chooses the smallest cap), not as an invisible host-level rule that bypasses player choice.
    - When deterministic behaviour is desired for reproducibility (e.g., strict-invariant soaks or AI self-play), agents and harnesses should use **seeded RNG or stable policies** to select among eligible targets; the host’s role is to expose the full choice set and apply the selected `eliminate_rings_from_stack` move, not to override that selection.
  - Current TS and Python engines still auto-select an eligible stack to eliminate in some forced-elimination situations (see [`KNOWN_ISSUES.md` P0.1](KNOWN_ISSUES.md:39)). This behaviour is treated as a **known deviation** from RR-CANON-R206 and must be corrected over time; until then, engines must at least remain aligned with each other and clearly document the heuristic in use.

- **[RR-CANON-R207] Real actions, ANM, and progress.**
  - For Last-Player-Standing victory (RR-CANON-R172), a player's **real actions** are exactly those listed in RR-CANON-R172:
    - Ring placements (RR-CANON-R080–R082).
    - Non-capture movements (RR-CANON-R090–R092).
    - Overtaking capture segments and chains (RR-CANON-R100–R103).
    - Pure forced-elimination actions from RR-CANON-R100 do **not** count as real actions for the purposes of RR-CANON-R172.
  - For ANM invariants and termination analysis:
    - Forced-elimination actions **do** count as global legal actions (RR-CANON-R200–R203); sequences in which forced elimination is the only available action are legal but must strictly increase the eliminated-ring component `E` of the progress metric `S = M + C + E` in RR-CANON-R191.
    - Because each forced elimination removes at least one ring from the acting player's cap and total rings are finite, any segment of play in which some player is repeatedly forced to eliminate caps must terminate in finitely many steps.
  - The ANM semantics in RR-CANON-R200–R203, together with the progress invariant in RR-CANON-R191, therefore justify the `INV-ACTIVE-NO-MOVES`, `INV-PHASE-CONSISTENCY`, and `INV-TERMINATION` invariants described in [`docs/INVARIANTS_AND_PARITY_FRAMEWORK.md`](docs/INVARIANTS_AND_PARITY_FRAMEWORK.md:119).

- **[RR-CANON-R208] Multi-phase turn sequence for line→Territory turns.**
  - For any turn in which an interactive action by the current player P (placement, movement, capture, or chain-capture segment) creates at least one new line owned by P and/or disconnects a Territory region they control, the canonical sequence of phases is:
    1. **Interactive phase:** `ring_placement`, `movement`, or `capture` in which the triggering action occurs. A chain-capture may immediately follow per RR-CANON-R209.
    2. **Chain capture (if any):** zero or more `chain_capture` segments while additional overtaking segments remain from the current chain origin (RR-CANON-R090–R103, RR-CANON-R209).
    3. **Line processing:** a single `line_processing` phase in which P may apply `process_line` / `choose_line_reward` decisions for any eligible lines they own (RR-CANON-R120–R122).
    4. **Territory processing:** upon exiting `line_processing`, if any disconnected regions for P satisfy the Q23 prerequisite (RR-CANON-R140–R145 and §7.3), a `territory_processing` phase is entered and Territory decisions are applied; otherwise, the engine skips directly to victory evaluation and turn rotation (RR-CANON-R170–R173).
  - Within a single turn, no other interactive phase may interleave between `line_processing` and `territory_processing`, and the engine must not re-enter `movement` or `capture` before the line+Territory consequences of the triggering action have been fully resolved. RR-CANON-R204’s phase-exit rules must be applied so that ANM states are never left pending between these phases.
  - References:
    - TS contract vectors `tests/fixtures/contract-vectors/v2/multi_phase_turn.vectors.json` (e.g. `multi_phase.placement_capture_line`, `multi_phase.full_sequence_with_territory` tagged `sequence:turn.line_then_territory.*`).
    - TS snapshot exporters and plateau/multi-region parity tests for combined line+Territory scenarios.
    - Python parity suite `ai-service/tests/parity/test_line_and_territory_scenario_parity.py` (line+Territory scenario and TS snapshot parity).

- **[RR-CANON-R209] Chain-capture phase boundaries.**
  - The dedicated `chain_capture` phase is an **interactive** phase that may only occur between the triggering movement/capture and the subsequent decision phases of RR-CANON-R208:
    - **Entry.**
      - After a legal `overtaking_capture` or `continue_capture_segment` by P, if the canonical capture aggregate reports that further capture segments are available from the landing position (`mustContinue == true` under the shared helpers), the engine must:
        - set `currentPhase` to `chain_capture`; and
        - record a chain state whose origin and owner define which `continue_capture_segment` moves are legal.
    - **During `chain_capture`.**
      - The only legal interactive moves for P are `continue_capture_segment` moves whose `from` matches the current chain origin and whose geometry is validated by the capture segment rules (RR-CANON-R100–R103).
      - It is illegal to accept any placement, non-capture movement, Territory decision, or line-processing decision for P while `currentPhase == chain_capture`.
    - **Exit.**
      - When the canonical capture aggregate reports that no further segments are available (`mustContinue == false`), the engine must immediately:
        - clear the internal chain-capture state; and
        - transition either:
          - to `line_processing` (if any lines for P exist, per RR-CANON-R208 and RR-CANON-R120–R122); or
          - directly to the next applicable phase per RR-CANON-R204 when no lines exist.
      - It is illegal to leave the game in `gameStatus == ACTIVE` and `currentPhase == chain_capture` if P has no legal `continue_capture_segment` moves.
  - References:
    - Shared capture chain helpers and aggregates under `src/shared/engine/**`.
    - TS turn-orchestrator multi-phase vectors (`multi_phase_turn.vectors.json`) with `chain_capture` in their `expectedPhaseSequence`.
    - Python chain-capture handling in `ai-service/app/game_engine.py` and the multi-phase parity tests listed above.

> **Cross-references (non-normative but recommended):**
>
> - Scenario-level ANM and forced-elimination behaviour, including concrete examples for RR-CANON-R200–R207, is catalogued in [`docs/ACTIVE_NO_MOVES_BEHAVIOUR.md`](docs/ACTIVE_NO_MOVES_BEHAVIOUR.md:1).
> - Invariant and parity expectations for these rules are described in [`docs/INVARIANTS_AND_PARITY_FRAMEWORK.md`](docs/INVARIANTS_AND_PARITY_FRAMEWORK.md:119) under `INV-ACTIVE-NO-MOVES`, `INV-PHASE-CONSISTENCY`, `INV-TERMINATION`, and `PARITY-TS-PY-ACTIVE-NO-MOVES`.

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
      - May contain any marker (own or opponent); landing on markers is always legal but incurs a cap-elimination cost per RR-CANON-R092.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §3.1; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§8.1–8.2, 16.9.4.1, 15.4 Q2.

- **[RR-CANON-R092] Marker interaction during non-capture movement.**
  - At departure `from`, place a regular marker of P (replacing any existing marker or empty cell; movement from collapsed spaces is illegal).
  - For each intermediate cell with a marker:
    - If marker belongs to opponent Q ≠ P, flip it to P.
    - If marker belongs to P, remove it and set that cell to a collapsed space owned by P.
  - At landing cell:
    - If there is any marker (own or opponent), remove that marker (do **not** collapse it).
    - Then place the moving stack.
    - If a marker was present (regardless of owner), immediately eliminate the top ring of the moving stack's cap and credit it to P.
  - P is **not required** to stop at the first legal landing after markers; any landing that satisfies RR-CANON-R091 is legal.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §3.2; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§4.2.1, 8.2, 8.3, 15.4 Q2.

- **[RR-CANON-R093] Post-movement capture eligibility (landing position constraint).**
  - After a non-capture movement (`move_stack` or `move_ring`) by P, the **optional capture opportunity** described in RR-CANON-R070 phase 3 is evaluated **only** from the stack that just moved, at its landing position.
  - Specifically:
    - Let `landing` be the cell where the moving stack landed after the non-capture movement.
    - Enumerate all legal overtaking capture segments (per RR-CANON-R101) whose `from` equals `landing`.
    - If at least one such capture segment exists, P may optionally begin an overtaking capture from `landing`; once started, chain continuation rules (RR-CANON-R103) apply.
    - If no such capture segment exists from `landing`, the turn proceeds directly to line processing (RR-CANON-R070 phase 4).
  - Importantly: captures from **other** stacks controlled by P (stacks that were not the subject of the movement) are **not** available as the optional post-movement capture; they may only be used on a subsequent turn when those stacks are moved.
  - This constraint ensures that the "optional capture after movement" decision is tied to the stack that was moved, making the post-movement phase deterministic based on the landing position.
  - References: TS `turnOrchestrator.ts` `processPostMovePhases` (capture detection from landing position); Python `game_engine.py` `_update_phase` (alignment required).

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
  - Control-flip edge case: If P's only control over a stack was a cap of height 1 (a single ring of P on top of opponent rings), forced elimination removes that cap and flips stack control to the opponent. If this causes P to have **zero controlled stacks** and **zero rings in hand**, P becomes "temporarily inactive" (per RR-CANON-R170) immediately. Turn rotation should skip P and proceed to the next player who has material.
  - Note (canonical choice): text in the Complete Rules suggesting that caps might already be eliminated while stacks remain is treated as unreachable; the forced-elimination rule is always applicable whenever P controls any stack.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §2.3; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§4.4, 13.4, 13.5, 15.4 Q24.

- **[RR-CANON-R101] Single capture segment legality.**
  - A single overtaking capture segment is a triple (`from`, `target`, `landing`) where:
    - `from` contains a stack controlled by P with height H and capHeight CH.
    - `target` contains a stack T with capHeight CH_T (any owner).
    - `landing` is an empty cell or a cell with any marker (own or opponent), strictly beyond `target` along one movement direction.
  - Requirements:
    - CH ≥ CH_T.
    - from, target, landing lie on one straight-line movement direction.
    - On from→target (excluding endpoints): no stacks, no collapsed spaces.
    - On target→landing (excluding endpoints): no stacks, no collapsed spaces.
    - Distance `distance(from, landing) ≥ H`.
    - Landing cell not collapsed; may contain any marker (landing on markers incurs cap-elimination cost per RR-CANON-R102).
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §4.1; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§9–10, 15.3, 15.4 Q2–Q5.

- **[RR-CANON-R102] Applying a capture segment.**
  - When executing a legal capture segment:
    - Process markers along the path exactly as in non-capture movement (RR-CANON-R092).
    - Move the attacking stack from `from` to `landing`.
    - Pop the top ring from the target stack and append it to the **bottom** of the attacking stack's ring list.
    - Recompute stackHeight and capHeight for both stacks.
    - If the target stack becomes empty, remove it.
    - If landing on any marker (own or opponent), remove the marker (do **not** collapse it), then land and immediately eliminate the top ring of the attacking stack's cap, crediting it to P (before line/territory processing).
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
  - A line is **eligible** if `len = k+1 ≥ requiredLength`, where `requiredLength` is:
    - 4 for `square8` in 2-player games.
    - 3 for `square8` in 3-4 player games.
    - 4 for `square19` and `hexagonal` (all player counts).
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §5.1; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§11.1, 15.3, 16.3, 16.9.4.3.

- **[RR-CANON-R121] Line processing order.**
  - After movement and any captures:
    - Compute all eligible lines in the current state.
    - While at least one eligible line remains:
      - Moving player P chooses one eligible line L to process. This choice is
        always represented as an explicit, player-visible line-processing
        decision in the `line_processing` phase (for example:
        - a `process_line` decision for every exact-length line, including
          the case where L is the **only** eligible line; and
        - a `process_line` decision followed by a `choose_line_reward`
          decision for overlength lines when a reward choice is available).
      - Apply collapse/elimination for L per RR-CANON-R122 as a consequence
        of that explicit decision.
      - Recompute eligible lines; continue until none remain or P cannot pay required eliminations.
  - Engines and hosts MUST NOT auto-collapse lines as a silent side effect of
    movement or capture moves, even when there is only a single eligible line
    for P. Every processed line (exact-length or overlength) corresponds to
    at least one explicit decision move recorded in the game history.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §5.2; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§4.5, 11.2–11.3, 15.4 Q7.

- **[RR-CANON-R122] Line collapse and elimination.**
  - Let requiredLen be the effective threshold defined in RR-CANON-R120 (3 or 4).
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
      - This choice between Option 1 and Option 2 is always an explicit,
        player-visible decision (typically via `choose_line_reward`); engines
        must not silently default to either option for overlength lines.
  - After each processed line, update all counters and recompute lines.
  - References: [`ringrift_compact_rules.md`](ringrift_compact_rules.md) §5.3; [`ringrift_complete_rules.md`](ringrift_complete_rules.md) §§4.5, 11.2–11.3, 15.4 Q7, Q22.

> **Example E1 – Exact-length line with no follow-up decisions (8×8).**
>
> - Board: `square8`, 2-player.
> - Situation: Late game, Player 1 (P1) has just completed a non-capture movement that forms a single, exact-length line of their own markers and does **not** disconnect any Territory regions.
> - Canonical sequence:
>   1. P1 plays their interactive move (e.g., a `move_stack`).
>   2. The engine detects one eligible line for P1. Even though it is the only line, P1 must process it via an explicit `process_line` decision:
>      - One `process_line` move is recorded for that line.
>      - Collapse and any required line-reward elimination are applied as consequences of that move (and, when applicable, a follow-up `choose_line_reward` decision for overlength lines).
>   3. No other lines remain; the engine does **not** auto-collapse additional markers or perform extra eliminations between moves.
>   4. Because no Territory regions were disconnected, there is no `territory_processing` phase for this turn; the engine proceeds directly to victory checks and turn rotation.
> - Replay behaviour:
>   - A canonical GameReplayDB for this turn records:
>     - the original `move_stack` (or `overtaking_capture`), and
>     - the explicit `process_line` (and `choose_line_reward`, if needed).
>   - Canonical replay must reach the post-line state **only** by applying those explicit moves in order; it may not inject additional collapses or eliminations between them.

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
  - A **full round of turns** is one contiguous cycle of turns in player order in which each player receives exactly one turn.
  - A player P wins by last-player-standing if all of the following hold:
    - **First round:** There exists at least one full round of turns such that:
      - On P's turn in that round, P has at least one legal real action available at the start of their action **and takes at least one such action**; and
      - On every other player's turns in that same round, those players have **no** legal real action available at the start of their action (they may have only forced-elimination actions, or no legal actions at all).
    - **Second round:** After the first round completes (including all line and Territory processing), on the following round P remains the only player who has taken any legal real action.
    - **Victory declared:** After the second round completes (including all post-movement processing), and after every other seat (including empty seats with no stacks and no rings) has recorded its required no-action or forced-elimination moves for that round, P is declared the winner by last-player-standing. This happens before the first seat begins any action in a subsequent round.
  - Players who still have rings on the board (including rings buried inside mixed-colour stacks) but whose only legal actions on their turns are forced eliminations, or who have no legal actions at all, are **temporarily inactive** for last-player-standing purposes. A player is temporarily inactive on their own turns when either:
    - they control no stacks on the board and have no legal placements (because they have no rings in hand or all placements would be illegal); or
    - they do control stacks but have no legal placements, no legal moves or overtaking captures, and no other legal turn actions at all, so their only possible turn action is forced elimination (RR-CANON-R100).
  - **Empty/temporarily inactive seats still take turns:** All seats, including those with no stacks and no rings in hand, must still traverse every phase of their turn and record the canonical no-action/FE moves required by RR-CANON-R075. There is no skipping of empty seats for LPS purposes; their turns are needed to satisfy the two full-round condition.
  - Temporarily inactive players prevent an LPS victory until they have been continuously in this "no real actions" state on each of their turns throughout both qualifying rounds above. A temporarily inactive player can return to full activity if they regain a real action, most commonly by gaining control of a multicolour stack whose top ring becomes their colour or by reducing the height of a stack they control so that it can move again. If any such player regains a real action before both rounds have been completed, the last-player-standing condition is not met and must be re-established from that point.
  - References: [`ringrift_simple_human_rules.md`](ringrift_simple_human_rules.md:321) §5.3; [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1376) §13.3.

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
   - Compact Spec: §5 uses `lineLength` per board type (3 for square8, 4 for square19/hexagonal), and treats any len > lineLength as "overlength".
   - Category: **Simplification / correction**.
   - Canonical: Use parameterized `lineLength` only; RR-CANON-R120–R122 control.

3. **Graduated line rewards threshold on 8×8.**
   - Complete Rules: Some prose suggests the "graduated rewards" choice on 8×8 begins at 4+ markers; elsewhere it mentions 5+ as the start of the more interesting tradeoff.
   - Compact Spec: Treats any len > lineLength (i.e., >3 on 8×8) as overlength with Options 1 and 2.
   - Category: **Simplification** (compact version generalizes rule).
   - Canonical: Follow the general rule: any len > lineLength is overlength with Options 1 and 2 (RR-CANON-R122).

4. **Explicit line decisions vs auto-processing.**
   - Some early engine implementations and prose describe auto-collapsing a
     single exact-length line immediately after a movement/capture, without
     surfacing a separate line-processing decision.
   - RR-CANON-R121–R122 clarify that **all** line processing—both exact-length
     and overlength—must be expressed as explicit, player-visible decisions in
     the `line_processing` phase (`process_line` / `choose_line_reward` +
     any required self-elimination decisions). Engines must not silently
     process lines as a side effect of other moves, even when there is only
     one eligible line.
   - Canonical: Treat line processing as purely decision-driven; any
     auto-processing behaviour is a non-canonical UX shortcut and must not
     affect the recorded game history or training data.

5. **Forced elimination impossibility edge case.**
   - Complete Rules: FAQ Q24 mentions the possibility that a player "cannot perform" forced elimination because "all caps have already been eliminated", suggesting a skip.
   - Compact Spec: §2.3 assumes that if a player controls a stack, they always have at least one ring in the cap to eliminate.
   - Category: **Genuine contradiction**, but the Complete Rules scenario is structurally impossible under the stack model.
   - Canonical: Treat forced elimination as always applicable whenever a player controls any stack (RR-CANON-R100), and treat the Q24 skip example as unreachable commentary.

6. **Territory border marker collapse scope.**
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
