# RingRift Rules Analysis - Phase 2: Consistency & Strategic Assessment

**Date:** November 18, 2025
**Analyst:** Architect Mode

> **Purpose:** Supporting analysis of how [`ringrift_complete_rules.md`](ringrift_complete_rules.md) and [`ringrift_compact_rules.md`](ringrift_compact_rules.md) align, plus strategic commentary for rules and engine maintainers. This file is not a canonical rules source.
> **Audience:** Rules designers, engine implementers, and AI authors.
> **Relationship to other docs:** The canonical rules specifications are [`ringrift_complete_rules.md`](ringrift_complete_rules.md) and [`ringrift_compact_rules.md`](ringrift_compact_rules.md). Earlier "Phase 1" analysis has been removed from the repository; this Phase 2 document is preserved as a supplement to the main rules docs and the scenario matrix in [`RULES_SCENARIO_MATRIX.md`](RULES_SCENARIO_MATRIX.md).

## 1. Consistency Check

**Documents Analyzed:** `ringrift_complete_rules.md` vs. `ringrift_compact_rules.md`

**Verdict:** ✅ **Mutually Consistent**

The two documents are highly consistent in their definition of the game's mechanics. The "Compact" rules serve as an excellent formal specification of the narrative "Complete" rules.

- **Core Mechanics:** Both correctly define the three board types, the dual adjacency systems (Moore/Von Neumann for square boards), and the unique Hexagonal adjacency.
- **Movement & Capture:** The "Unified Landing Rule" (landing on any valid space beyond markers/target) is present in both. The "No-dead-placement" rule is explicitly defined in both (Section 4.1 in Complete, Section 2.1 in Compact).
- **Victory Conditions:** The thresholds are mathematically identical (e.g., `>50%` vs `floor(total/2) + 1`).
- **Terminology:** Terms like "Stack Height", "Cap Height", "Overtaking", and "Elimination" are used consistently.

**Minor Note:** The Compact rules are more rigorous in their definition of the "Self-Elimination Prerequisite" for territory disconnection (Section 6.3), which clarifies the "hypothetical" check mentioned in the Complete rules. This is a feature of the compact spec, not a discrepancy.

---

## 2. Goal Assessment

**Stated Goals:** Exciting, Complex, Strategic, Tactical.

### Assessment

- **Exciting:** ✅ **High.** The mechanics of **Chain Reactions** (territory disconnection) and **Chain Captures** create high-volatility turns where the board state can swing dramatically. The "Forced Elimination" rule ensures the game always progresses towards a climax, preventing boring stalls.
- **Complex:** ✅ **Very High.** The game features a high branching factor due to the combination of movement freedom (any distance >= stack height), placement options, and the dual-layer logic of "Overtaking" (stack building) vs "Elimination" (scoring). The distinction between Moore (movement) and Von Neumann (territory) adjacency on square boards adds a subtle but deep layer of complexity.
- **Strategic:** ✅ **High.** Players must balance three competing resources: **Board Position** (territory), **Material** (rings in hand/on board), and **Tempo** (turn initiative). The "Graduated Line Rewards" (Option 1 vs Option 2) force players to choose between immediate territory gain and long-term material preservation.
- **Tactical:** ✅ **High.** The **Mandatory Chain Capture** rule creates forced sequences that players must calculate in advance. The geometry of "180° Reversal" and "Cyclic" capture patterns rewards precise calculation.

**Conclusion:** The rules as written successfully meet all stated design goals.

---

## 3. Proposed Rule Changes & Improvements

### A. Assessment of "Last Player to Move" Stalemate Resolution

**Proposal:** Change the stalemate tiebreaker (when no player can move) to simply declare the **"Last player to make a valid turn action"** as the winner.

**Analysis:**
Currently, RingRift is a **Territory and Material** game. Victory comes from controlling space or eliminating enemies. The current tiebreakers (Territory > Rings > Markers) reinforce this identity.

Changing the stalemate condition to "Last Player to Move" would:

1.  **Shift the Genre:** It would turn the endgame into a **Blocking/Tempo** game (like Nim or Amazons) rather than a Territory game.
2.  **Undermine Strategy:** A player with 45% territory could lose to a player with 5% territory simply because the latter made the last legal move before the board locked up. This contradicts the "Territory Victory" goal.
3.  **Reduce Complexity:** It removes the nuance of managing territory/rings in the endgame, replacing it with a simpler "don't run out of moves" heuristic.

**Verdict:** ❌ **Not Recommended.**
While "Last Player Standing" (Section 13.3) is already a valid victory condition (winning because opponents _cannot_ move while you _can_), applying this logic to a **Global Stalemate** (where _no one_ can move) is counter-productive. It would decouple the victory condition from the primary mechanics (Territory/Elimination). The current tiebreakers correctly reward the player who "won" the board, even if the game ended in a deadlock.

### B. Candidate for Improvement: "Unified Adjacency" (Rejected)

- _Idea:_ Use Moore (8-way) or Von Neumann (4-way) for _everything_ on square boards to simplify the rules.
- _Assessment:_
  - Using 8-way for territory makes disconnection nearly impossible (too hard to surround).
  - Using 4-way for movement makes the game too static and grid-like.
- _Conclusion:_ The current hybrid approach (8-way move, 4-way territory) is necessary for the specific "Complex" and "Exciting" goals of RingRift.

### C. Strong Candidate: "Explicit Pass" (Minor Tweak)

- _Observation:_ Currently, if a player has rings in hand but no legal placement (due to the no-dead-placement rule) and no stacks, they effectively "skip" their turn.
- _Proposal:_ Formalize this as a "Pass" action.
- _Benefit:_ Cleans up the state machine and makes "Last Player Standing" checks explicit (if all other players Pass, and you Move, you win).
- _Status:_ This is already effectively how the engine works, but making it an explicit rule term helps clarity.

### D. Strong Candidate: "Mercy" Threshold (UI/Implementation Level)

- _Observation:_ A player with 0 rings on board and 0 in hand is technically still in the game (waiting for a stack control change), but effectively eliminated.
- _Proposal:_ In the UI/Engine, explicitly flag these players as "Dormant". If all opponents agree, or if the game state mathematically precludes their return (e.g., all their rings are buried too deep to ever surface), they could be eliminated.
- _Benefit:_ Speeds up multiplayer games.

---

## 4. Final Recommendation

**The rules are solid.** They are internally consistent and meet the ambitious design goals.

**Do not change the stalemate resolution.** The current hierarchy (Territory > Rings > Markers) ensures that even in a deadlock, the player who played the "better" game according to the primary metrics is rewarded.

**Focus on Implementation:** The complexity of RingRift lies in its _state_, not just its rules. The best way to "improve" the game now is not to change the rules, but to provide **better UI tools** for players to manage that complexity:

1.  **Visualizing Disconnection:** Highlight borders that are 1 move away from disconnecting a region.
2.  **Move Preview:** Show the result of "Option 1 vs Option 2" line collapses before committing.
3.  **Chain Planning:** Visual indicators for mandatory chain capture paths.

These UI enhancements will make the "Complex" and "Tactical" goals accessible without simplifying the rules themselves.

---

## 5. Backend ↔ Sandbox Decision-Move Model Parity

This section documents how the unified Move/GamePhase model for **advanced decision phases** (chain capture, line processing, territory processing, explicit elimination) is implemented consistently across:

- The backend engine (GameEngine + RuleEngine).
- The client sandbox (ClientSandboxEngine + sandbox helpers).
- AI and WebSocket flows.

The goal is that any _interactive_ decision the rules require can be expressed as a canonical `Move`, enumerated via `getValidMoves`, and applied by both engines in a structurally identical way.

### 5.1 Chain capture (GamePhase = `chain_capture`)

**Canonical Move type:** `continue_capture_segment`

- **Backend enumeration**
  - Phase entry:
    - After an `overtaking_capture` or `continue_capture_segment`, GameEngine updates an internal `chainCaptureState`.
    - If further capture segments exist from the new landing position, GameEngine:
      - Sets `currentPhase = 'chain_capture'`.
      - Leaves the same player active.
  - Move enumeration:
    - `GameEngine.getValidMoves` in `chain_capture` calls a shared capture-chain helper to enumerate all legal segments from `chainCaptureState.currentPosition`.
    - Each candidate is relabelled as a `continue_capture_segment` Move with:
      - `from` (current stack position).
      - `captureTarget` (the overtaken stack).
      - `to` (landing position).
      - A stable `id` derived from the geometry (e.g. `continue-x1,y1-x3,y3-x5,y5`).

- **Backend application**
  - `GameEngine.makeMove` treats `continue_capture_segment` as:
    - A normal capture segment that:
      - Applies marker effects and overtaking semantics.
      - Updates `chainCaptureState`.
    - If further segments remain:
      - Remains in `chain_capture` and returns early (no line/territory processing yet).
    - If the chain is exhausted:
      - Clears `chainCaptureState`.
      - Falls through to automatic or move-driven line/territory processing.

- **Sandbox enumeration**
  - `ClientSandboxEngine.enumerateCaptureSegmentsFrom` uses `enumerateCaptureSegmentsFromSandbox` and the same geometric rules to list legal segments from a given stack.
  - This is used by:
    - Human move highlighting (`getValidLandingPositionsForCurrentPlayer`).
    - Sandbox AI (`maybeRunAITurn`).

- **Sandbox application**
  - Human/AI chains:
    - `handleMovementClick` + `performCaptureChain`:
      - Apply segments one-by-one using `applyCaptureSegmentOnBoard`.
      - Use a local `CaptureDirectionChoice` when multiple directions exist.
  - Canonical replay:
    - `applyCanonicalCaptureSegment`:
      - Applies exactly one segment via `applyCaptureSegment`.
      - If further segments exist:
        - Sets `currentPhase = 'chain_capture'` and defers consequences.
      - If no segments remain:
        - Calls `advanceAfterMovement` to run line/territory processing and turn advancement.

**Parity guarantee:** Both engines:

- Use the same geometric capture validator (shared core and helpers).
- Represent each segment as a canonical Move with the same fields.
- Defer post-capture consequences until the chain terminates.

### 5.2 Line processing (GamePhase = `line_processing`)

**Canonical Move types:**

- `process_line` – choose which detected line to process.
- `choose_line_reward` – for overlength lines, choose Option 1 vs Option 2 style rewards.

**Backend side**

- **Enumeration**
  - `RuleEngine.getValidMoves` when `currentPhase === 'line_processing'`:
    - Delegates to `getValidLineProcessingDecisionMoves`:
      - One `process_line` Move per detected line for `currentPlayer`, carrying:
        - `formedLines[0] = LineInfo` (positions, length, direction, player).
        - A stable `id` of the form `process-line-{index}-{positionsKey}`.
      - One `choose_line_reward` Move for each overlength line, with:
        - `formedLines[0]` as above.
        - `id = choose-line-reward-{index}-{positionsKey}`.
  - `GameEngine.getValidLineProcessingMoves` mirrors this logic for the Move-driven decision-phase flag, using BoardManager directly.

- **Application**
  - `GameEngine.applyDecisionMove`:
    - Resolves `formedLines[0]` back to a current `LineInfo` by matching marker positions.
    - For both `process_line` and `choose_line_reward`:
      - Delegates to `processOneLine`, which:
        - Exact-length (`length == requiredLength`):
          - Collapses all markers to territory.
          - Calls `eliminatePlayerRingOrCapWithChoice` (thenable RingEliminationChoice).
        - Overlength:
          - With no interaction manager: Option 2 semantics (minimum collapse, no elimination).
          - With interaction manager: issues a `LineRewardChoice` and interprets the response.
    - After processing:
      - If more lines exist for the same player:
        - Remains in `line_processing`.
      - Otherwise:
        - Transitions to `territory_processing`.

**Sandbox side**

- **Enumeration**
  - `ClientSandboxEngine.getValidLineProcessingMovesForCurrentPlayer`:
    - Finds all lines via `findAllLinesOnBoard`.
    - Builds `process_line` and `choose_line_reward` Moves with:
      - `formedLines[0]` mirroring backend.
      - `id` schemes identical to backend (`process-line-*`, `choose-line-reward-*`).

- **Application**
  - Automatic resolution:
    - `processLinesForCurrentPlayer`:
      - Loops while lines exist for `currentPlayer`.
      - For each iteration:
        - Collapses markers using `collapseLineMarkers`.
        - For exact-length lines: calls `forceEliminateCapOnBoard`.
        - For overlength lines: collapses the minimum number of markers (no elimination).
  - Canonical replay:
    - `applyCanonicalProcessLine`:
      - Uses `formedLines[0]` when present, or recomputes lines.
      - Applies the same exact-length vs overlength semantics as the automatic path, but for a single line and without issuing any PlayerChoices.
    - `applyCanonicalChooseLineReward`:
      - For exact-length lines: delegates back to `applyCanonicalProcessLine`.
      - For overlength lines: applies Option 1 semantics:
        - Collapses the _entire_ line.
        - Calls `forceEliminateCap` for the moving player.

**Parity guarantee:**

- The same marker sequences and cap eliminations are applied for:
  - Automatic sandbox line resolution.
  - Canonical sandbox replay of backend `process_line` / `choose_line_reward` Moves.
  - Backend automatic/interactive line processing.
- Stable `Move.id` formats (`process-line-*`, `choose-line-reward-*`) allow transports and tests to map PlayerChoices to canonical Moves in both engines.

### 5.3 Territory processing and explicit self-elimination (GamePhase = `territory_processing`)

**Canonical Move types:**

- `process_territory_region` – choose which disconnected region to process.
- `eliminate_rings_from_stack` – choose a self-elimination stack/cap when required.

**Backend side**

- **Region enumeration**
  - `RuleEngine.getValidMoves` with `currentPhase === 'territory_processing'`:
    - Calls `getValidTerritoryProcessingDecisionMoves`:
      - Finds disconnected regions for `currentPlayer`.
      - Filters via a self-elimination prerequisite:
        - Player must control at least one stack outside the region.
      - For each eligible region:
        - Produces a `process_territory_region` Move with:
          - `disconnectedRegions[0] = Territory` (spaces).
          - `id = process-region-{index}-{representativeKey}`.
  - `GameEngine.getValidTerritoryProcessingMoves` mirrors this pattern using its own helper and `canProcessDisconnectedRegion`.

- **Elimination enumeration**
  - `RuleEngine.getValidEliminationDecisionMoves`:
    - Only active when _no_ eligible disconnected regions remain.
    - Enumerates controlled stacks for `currentPlayer` and, for each:
      - Computes `capHeight`.
      - Builds an `eliminate_rings_from_stack` Move with:
        - `to = stack.position`.
        - `eliminatedRings` diagnostics.
        - `eliminationFromStack` (position, capHeight, totalHeight).
        - `id = eliminate-{positionKey}`.

- **Application**
  - Region processing:
    - `GameEngine.applyDecisionMove` for `process_territory_region`:
      - Re-identifies the region from `disconnectedRegions[0]`.
      - Re-applies the self-elimination prerequisite defensively.
      - Delegates to `processOneDisconnectedRegion`, which:
        - Eliminates all rings _inside_ the region.
        - Collapses region + border markers to territory.
        - Credits all internal eliminations to the moving player.
        - Performs mandatory self-elimination via `eliminatePlayerRingOrCapWithChoice`.
      - Re-checks for further eligible regions:
        - If any remain: stays in `territory_processing`.
        - If none: calls `advanceGame` and `stepAutomaticPhasesForTesting`.
  - Explicit elimination:
    - `GameEngine.applyDecisionMove` for `eliminate_rings_from_stack`:
      - Validates target stack and controller.
      - Uses the same `eliminateFromStack` helper as line rewards.
      - Treats this as the final step in `territory_processing`, then advances the turn.

**Sandbox side**

- **Region enumeration**
  - Core engine:
    - `processDisconnectedRegionsForCurrentPlayer` and `processDisconnectedRegionsForCurrentPlayerEngine`:
      - Use `findDisconnectedRegionsOnBoard`.
      - Filter regions using a callback `canProcessDisconnectedRegion(region.spaces, player, state)`, identical to the backend prerequisite.
      - When multiple regions exist and an interaction handler is present:
        - Emit a `RegionOrderChoice` with options carrying `regionId`, size, and a representative position.
  - Canonical Move surface:
    - `ClientSandboxEngine.getValidTerritoryProcessingMovesForCurrentPlayer`:
      - Uses `findDisconnectedRegionsOnBoard`.
      - Applies the same self-elimination prerequisite.
      - Builds `process_territory_region` Moves with:
        - `disconnectedRegions[0] = region`.
        - `id = process-region-{index}-{representativeKey}`.

- **Elimination enumeration**
  - Forced elimination:
    - `forceEliminateCapOnBoard` + `maybeProcessForcedEliminationForCurrentPlayer`:
      - Handle _forced_ eliminations when a player is globally blocked.
      - These do not expose explicit Moves; they are automatic stalemate resolution steps.
  - Explicit self-elimination parity:
    - `ClientSandboxEngine.getValidEliminationDecisionMovesForCurrentPlayer`:
      - Mirrors backend `getValidEliminationDecisionMoves`:
        - Defers until no eligible regions remain.
        - Enumerates controlled stacks for `currentPlayer`.
        - For each stack:
          - Computes cap/height.
          - Produces `eliminate_rings_from_stack` with:
            - `to = stack.position`.
            - `eliminatedRings` and `eliminationFromStack` diagnostics.
            - `id = eliminate-{positionKey}`.

- **Application**
  - Region processing:
    - Automatic:
      - `processDisconnectedRegionsForCurrentPlayer` and `processDisconnectedRegionsForCurrentPlayerEngine` call `processDisconnectedRegionOnBoard`:
        - Eliminates internal rings.
        - Collapses region and borders.
        - Applies mandatory self-elimination via `forceEliminateCapOnBoard`.
        - Asserts S-invariant monotonicity.
    - Canonical Move:
      - `applyCanonicalProcessTerritoryRegion`:
        - Accepts a `process_territory_region` Move.
        - Re-evaluates the self-elimination prerequisite against the current board.
        - Calls `processDisconnectedRegionOnBoard` once for the chosen region.
  - Explicit elimination:
    - `applyCanonicalEliminationMove`:
      - Accepts an `eliminate_rings_from_stack` Move.
      - Identifies the target stack by `to`.
      - Eliminates the entire cap, credits eliminatedRings and totalRingsEliminated.
      - Updates stack or removes it when empty.

**Parity guarantee:**

- When backend territory-processing decisions are expressed as canonical Moves (`process_territory_region`, `eliminate_rings_from_stack`), the sandbox can:
  - Enumerate the same decision surface from its own state.
  - Apply the same structural effects using `applyCanonicalProcessTerritoryRegion` and `applyCanonicalEliminationMove`.
- Self-elimination ordering matches the rules:
  - Always process eligible regions first.
  - Only surface explicit elimination decisions after all eligible regions have been processed.
