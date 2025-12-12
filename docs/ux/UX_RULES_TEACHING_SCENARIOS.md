# UX Rules Teaching Scenarios Spec

> **Doc Status (2025-12-05): Draft – W‑UX‑3**
>
> **Role:** Define a scenario‑driven teaching blueprint for complex RingRift mechanics (ANM/FE, structural stalemate, mini‑regions, chain captures, line vs territory ordering, landing on own markers), to be used by TeachingOverlay, Sandbox presets, and any tutorial carousel.
>
> **Inputs:**
>
> - Rules canon and examples in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:193), [`ringrift_complete_rules.md`](ringrift_complete_rules.md:260).
> - Edge‑cases and consistency analysis in [`docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md`](docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md:361).
> - ANM/FE behaviour catalogue in [`docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md`](docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md:39).
> - Territory mini‑region scenarios in [`tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts`](tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts:19).
> - Shared helpers tests in [`tests/unit/territoryDecisionHelpers.shared.test.ts`](tests/unit/territoryDecisionHelpers.shared.test.ts:76) and [`tests/unit/lineDecisionHelpers.shared.test.ts`](tests/unit/lineDecisionHelpers.shared.test.ts:23).
> - Copy and weird‑state mappings in [`docs/UX_RULES_COPY_SPEC.md`](docs/UX_RULES_COPY_SPEC.md:1) and [`docs/UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:1).
> - Telemetry schema in [`docs/UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:1).
> - Navigation map for high‑risk concepts in [`docs/UX_RULES_CONCEPTS_INDEX.md`](docs/UX_RULES_CONCEPTS_INDEX.md:1) to keep teaching flows aligned with rules docs, UX surfaces, and telemetry labels.

This spec does **not** implement any code. It defines:

- A prioritized list of **rules concepts** needing scenario‑driven teaching.
- A standard **scenario metadata shape** for teaching flows.
- **Scenario families / flows** with 2–5 steps each.
- A mapping of flows to client **surfaces** (TeachingOverlay, sandbox presets, tutorial carousel).
- Links back to weird‑state reason codes and `rules_context` for telemetry.

Code‑mode tasks will:

- Create concrete board positions and scenario JSON / fixtures.
- Hook them into the existing scenario loader and TeachingOverlay.
- Wire telemetry using [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:172).

## Teaching topics referenced from the game-end explanation model are defined in this document; see [`UX_RULES_EXPLANATION_MODEL_SPEC.md`](docs/UX_RULES_EXPLANATION_MODEL_SPEC.md:1) for how they are used.

## 0. Iteration Log & Improvement History

This spec is part of the broader rules‑UX improvement loop described in [`UX_RULES_IMPROVEMENT_LOOP.md`](docs/UX_RULES_IMPROVEMENT_LOOP.md:24). Concrete changes to teaching flows and their surrounding UX are tracked in numbered iteration records under `docs/ux/rules_iterations/`:

- [`UX_RULES_IMPROVEMENT_ITERATION_0001.md`](docs/ux/rules_iterations/UX_RULES_IMPROVEMENT_ITERATION_0001.md:1) – Initial hotspot‑oriented plan for ANM/FE loops, structural stalemate, and mini‑regions (pre‑implementation design sketch).
- [`UX_RULES_IMPROVEMENT_ITERATION_0002.md`](docs/ux/rules_iterations/UX_RULES_IMPROVEMENT_ITERATION_0002.md:1) – Backfilled record for W1–W5 work aligning game‑end explanations (HUD + VictoryModal + TeachingOverlay) with ANM/FE, structural stalemate, and territory mini‑regions, and wiring them into telemetry.
- [`UX_RULES_IMPROVEMENT_ITERATION_0003.md`](docs/ux/rules_iterations/UX_RULES_IMPROVEMENT_ITERATION_0003.md:1) – Backfilled record for W1–W5 work on territory mini‑regions (Q23) and capture‑chain teaching flows, including curated scenarios and concordance updates.

Future runs of the rules‑UX loop SHOULD:

- Continue this numbering (0004, 0005, …) using the same template.
- Store iteration notes under `docs/ux/rules_iterations/`.
- Reference the relevant teaching flows and `rulesConcept` ids from this spec when scoping new work.

## 1. Prioritized Rules Concepts

Based on audits in [`docs/supplementary/RULES_DOCS_UX_AUDIT.md`](docs/supplementary/RULES_DOCS_UX_AUDIT.md:23), edge‑case report [`RULES_CONSISTENCY_EDGE_CASES.md`](docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md:361), and the mini‑region / line / territory helpers tests, the following concepts are highest priority:

1. **Forced Elimination loops & ANM shapes**
   - Where players “have no moves” and caps disappear unexpectedly.
   - ANM‑SCEN‑01/02/03 in [`ACTIVE_NO_MOVES_BEHAVIOUR.md`](docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md:39).
   - Reason codes `ANM_MOVEMENT_FE_BLOCKED`, `FE_SEQUENCE_CURRENT_PLAYER` in [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:121).

2. **Territory mini‑regions and self‑elimination (Q23 archetype)**
   - 2×2 region archetype `Rules_12_2_Q23_mini_region_square8_numeric_invariant` in [`tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts`](tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts:19).
   - Interaction between interior eliminations and mandatory self‑elimination.

3. **Multi‑region territory with limited self‑elimination budget**
   - Cases where multiple regions are disconnected but only one can be processed this turn (SCEN‑TERRITORY‑002 in [`archive/RULES_DYNAMIC_VERIFICATION.md`](archive/RULES_DYNAMIC_VERIFICATION.md:435)).

4. **Lines vs Territory ordering in multi‑phase turns**
   - Movement / capture → chain capture → lines → territory → victory (`RR‑CANON‑R070–R071` in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:195)).
   - Combined line+territory scenarios in [`tests/unit/GameEngine.territoryDisconnection.test.ts`](tests/unit/GameEngine.territoryDisconnection.test.ts:459).

5. **Capture chains and mandatory continuation**
   - Optional start, mandatory continuation, freedom of direction (FAQ Q15, RR‑CANON‑R103).
   - 180° reversal and cyclic capture scenarios in FAQ tests (not listed here but referenced in [`archive/RULES_DYNAMIC_VERIFICATION.md`](archive/RULES_DYNAMIC_VERIFICATION.md:302)).

6. **Landing on own marker and self‑elimination**
   - Movement / capture that lands on a marker of your colour and immediately eliminates your top ring (`RR‑CANON‑R092`](RULES_CANONICAL_SPEC.md:418)).

7. **Structural stalemate and tiebreak ladder**
   - Global stalemate, tie‑break by territory → eliminated rings (including rings in hand) → markers → last actor (`RR‑CANON‑R173`](RULES_CANONICAL_SPEC.md:619)).

8. **Last‑Player‑Standing (LPS) vs “just keep playing”**
   - Clarifying that LPS is a distinct rule; implementation currently treats it as a compromise (CCE‑006 in [`RULES_CONSISTENCY_EDGE_CASES.md`](docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md:443)).
   - ANM‑SCEN‑07 in [`ACTIVE_NO_MOVES_BEHAVIOUR.md`](docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md:160).

Each scenario flow defined below targets at least one of these concepts and, where appropriate, is linked to a **weird‑state reason code** from [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:121).

---

## 2. Scenario Metadata Shape

This section defines an abstract metadata shape; concrete implementations in TS/JSON can vary as long as the semantics are preserved.

### 2.1 Type Sketch

```ts
type RulesConcept =
  | 'anm_forced_elimination'
  | 'territory_mini_region'
  | 'territory_multi_region_budget'
  | 'line_vs_territory_multi_phase'
  | 'capture_chain_mandatory'
  | 'landing_on_own_marker'
  | 'structural_stalemate'
  | 'last_player_standing';

type TeachingStepKind = 'guided' | 'interactive';

type TeachingScenarioMetadata = {
  scenarioId: string; // Unique, stable scenario id (string).
  rulesConcept: RulesConcept; // Primary rules concept.
  flowId: string; // Family id (e.g. 'fe_loop_intro').
  stepIndex: number; // 0‑based or 1‑based index within flow (be consistent).
  stepKind: TeachingStepKind; // Guided explanation or interactive exercise.

  // UX & telemetry linkage
  rulesDocAnchor?: string; // e.g. 'ringrift_complete_rules.md#134-end-of-game-stalemate-resolution'.
  uxWeirdStateReasonCode?: string; // Optional reason code from UX_RULES_WEIRD_STATES_SPEC (e.g. 'ANM_MOVEMENT_FE_BLOCKED').
  telemetryRulesContext?: string; // Optional override for rules_context in telemetry.

  // Display / recommended usage
  recommendedBoardType: 'square8' | 'square19' | 'hexagonal';
  recommendedNumPlayers: 2 | 3 | 4;
  showInTeachingOverlay: boolean;
  showInSandboxPresets: boolean;
  showInTutorialCarousel: boolean;

  // Optional hints for UI
  learningObjectiveShort: string; // One‑sentence objective for UI list.
  difficultyTag?: 'intro' | 'intermediate' | 'advanced';
};
```

Notes:

- `uxWeirdStateReasonCode` SHOULD be filled for scenarios that are directly about weird states:
  - e.g. FE/ANM flows reference `ANM_MOVEMENT_FE_BLOCKED`.
  - Structural stalemate flows reference `STRUCTURAL_STALEMATE_TIEBREAK`.
- `telemetryRulesContext` SHOULD match the concept‑level `rules_context` names used in [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:145).

### 2.2 Conventions

- **Scenario id naming:**
  - Use dotted families: `teaching.fe_loop.step_1`, `teaching.mini_region.q23.step_2`.
  - Where directly linked to a rules doc FAQ, embed that id: `teaching.territory.q23.mini_region.step_1`.
- **Flow id naming:**
  - Short, stable ids, e.g. `fe_loop_intro`, `mini_region_intro`, `multi_phase_line_then_territory`.
- **Board type:**
  - Prefer `square8` for introductory flows, except where geometry is critical (e.g. hex movement).
- **Num players:**
  - Prefer 2‑player for tightly focused mechanics unless multi‑player dynamics are essential.

---

## 3. Scenario Families and Flows

Each subsection describes one scenario family (flow), its id, and its steps.

### 3.1 Flow: Forced Elimination Loop & ANM – `fe_loop_intro`

- **Flow id:** `fe_loop_intro`
- **Primary concept:** `anm_forced_elimination`
- **Weird‑state reason code:** `ANM_MOVEMENT_FE_BLOCKED` (and in the extended step, `FE_SEQUENCE_CURRENT_PLAYER`).
- **Recommended board:** `square8`, `numPlayers = 2`.
- **Target surfaces:** TeachingOverlay, sandbox presets, tutorial carousel.

#### Steps

1. **Step id:** `teaching.fe_loop.step_1`
   - **Step index:** 1
   - **Kind:** `guided`
   - **Learning objective:**  
     `"Recognise when you have no legal placements, movements, or captures, and that forced elimination will apply."`
   - **Board description:**  
     A small position where the local player controls one mid‑sized stack on the edge, surrounded by collapsed territory / blocking stacks such that:
     - No legal non‑capture moves or captures exist.
     - No legal placements (no‑dead‑placement + cap).
     - A forced elimination is available from that stack.
   - **Expected outcome:**  
     The player reads an annotated overlay highlighting:
     - “Real moves” vs forced elimination.
     - Why no placements/moves exist.
     - That FE will remove the cap as per R100.
   - **Metadata:**
     - `rulesDocAnchor` → [`ringrift_complete_rules.md`](ringrift_complete_rules.md:499) §4.4.
     - `uxWeirdStateReasonCode` → `ANM_MOVEMENT_FE_BLOCKED`.

2. **Step id:** `teaching.fe_loop.step_2`
   - **Step index:** 2
   - **Kind:** `interactive`
   - **Learning objective:**  
     `"Execute a forced elimination when no real move exists."`
   - **Board description:**  
     Same starting position as step 1; the player is required to perform the FE action (via a dedicated FE UI) to proceed.
   - **Expected outcome:**  
     After FE:
     - The cap is removed.
     - Eliminated rings are credited to the acting player.
     - The UI emphasises that the turn then ends or transitions to the next phase.
   - **Metadata:**  
     `uxWeirdStateReasonCode` → `ANM_MOVEMENT_FE_BLOCKED`.

3. **Step id:** `teaching.fe_loop.step_3`
   - **Step index:** 3
   - **Kind:** `interactive`
   - **Learning objective:**  
     `"See how repeated FE over multiple turns can shrink your stacks and move the game toward plateau or LPS."`
   - **Board description:**  
     2‑player position where:
     - P1 has several blocked stacks that will be repeatedly forced to eliminate caps on successive P1 turns.
     - P2 retains active stacks with moves.
   - **Expected outcome:**  
     Over several guided turns, P1 experiences multiple FE events until:
     - P1 loses all stacks, becoming inactive.
     - The scenario either:
       - Ends with P2 approaching elimination / territory victory, or
       - Sets up an LPS teaching segue.
   - **Metadata:**  
     `uxWeirdStateReasonCode` → `FE_SEQUENCE_CURRENT_PLAYER`.

---

### 3.2 Flow: Territory Mini‑Region (Q23) – `mini_region_intro`

- **Flow id:** `mini_region_intro`
- **Primary concept:** `territory_mini_region`
- **Weird‑state reason code:** none directly (regular rules; confusion comes from interaction of eliminations and self‑elimination).
- **Existing scenario anchor:**  
  `Rules_12_2_Q23_mini_region_square8_numeric_invariant` via [`tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts`](tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts:19).
- **Recommended board:** `square8`, `numPlayers = 2` or `3`.

#### Steps

1. **Step id:** `teaching.mini_region.q23.step_1`
   - **Index:** 1
   - **Kind:** `guided`
   - **Learning objective:**  
     `"Understand the shape of a disconnected mini‑region and why it is eligible for territory processing."`
   - **Board description:**  
     A 2×2 region of victim stacks surrounded by another player’s markers and collapsed territory, as in the Q23 numeric invariant scenario.
   - **Expected outcome:**  
     Overlay explains:
     - Physical disconnection.
     - Colour representation criterion.
     - Which player is allowed to process the region.
   - **Metadata:**
     - `rulesDocAnchor` → [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1815) FAQ Q23; [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:549) R141–R143.

2. **Step id:** `teaching.mini_region.q23.step_2`
   - **Index:** 2
   - **Kind:** `interactive`
   - **Learning objective:**  
     `"Trigger region processing and observe how interior rings and border markers are handled."`
   - **Board description:**  
     Same mini‑region; moving player must choose to process the region.
   - **Expected outcome:**  
     After processing:
     - All interior rings are eliminated and credited to the acting player.
     - Region spaces and relevant border markers become collapsed territory.
   - **Metadata:**
     - `rulesDocAnchor` → [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:575) R145.

3. **Step id:** `teaching.mini_region.q23.step_3`
   - **Index:** 3
   - **Kind:** `interactive`
   - **Learning objective:**  
     `"Apply the mandatory self‑elimination cost correctly from stacks outside the region."`
   - **Board description:**  
     Extend step 2 by including one outside stack for the acting player; after region processing, the player must choose a ring/cap to eliminate from that stack.
   - **Expected outcome:**  
     Player sees:
     - That at least one ring/cap outside the region must exist for processing.
     - How that self‑elimination contributes to elimination totals.

---

### 3.3 Flow: Multi‑Region Territory with Limited Self‑Elimination – `multi_region_budget`

- **Flow id:** `multi_region_budget`
- **Primary concept:** `territory_multi_region_budget`
- **Weird‑state reason code:** none directly (but connects to confusion about “why didn’t that other region collapse?”).
- **Recommended board:** `square19`, `numPlayers = 3`.

#### Steps

1. **Step id:** `teaching.territory.multi_region.step_1`
   - **Index:** 1
   - **Kind:** `guided`
   - **Learning objective:**  
     `"See an example with two disconnected regions but only one available outside stack to pay the cost."`
   - **Board description:**  
     Two disconnected regions R1 and R2, both candidate regions for the moving player, but only one outside stack available.
   - **Expected outcome:**  
     Explanation emphasises:
     - Self‑elimination prerequisite.
     - One outside stack can only pay for one region.

2. **Step id:** `teaching.territory.multi_region.step_2`
   - **Index:** 2
   - **Kind:** `interactive`
   - **Learning objective:**  
     `"Choose which region to process when you cannot afford both."`
   - **Board description:**  
     Same as step 1; player must choose either R1 or R2.
   - **Expected outcome:**  
     Only the chosen region is processed; the other remains for later turns.
   - **Rules anchor:**  
     [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:574) R144; SCEN‑TERRITORY‑002 in [`archive/RULES_DYNAMIC_VERIFICATION.md`](archive/RULES_DYNAMIC_VERIFICATION.md:435).

---

### 3.4 Flow: Lines vs Territory Multi‑Phase Turn – `line_vs_territory_blocking`

- **Flow id:** `line_vs_territory_blocking`
- **Primary concept:** `line_vs_territory_multi_phase`
- **Weird‑state reason code:** none directly (standard multi‑phase semantics).
- **Recommended board:** `square8`, `numPlayers = 3`.

#### Steps

1. **Step id:** `teaching.multi_phase.step_1`
   - **Index:** 1
   - **Kind:** `guided`
   - **Learning objective:**  
     `"Understand the fixed order: movement → capture/chain → lines → territory → victory."`
   - **Board description:**  
     Static diagram showing markers and stacks where a single move will both:
     - Complete a line.
     - Disconnect a region.
   - **Expected outcome:**  
     Overlay narrates the multi‑phase sequence and what happens in each phase.

2. **Step id:** `teaching.multi_phase.step_2`
   - **Index:** 2
   - **Kind:** `interactive`
   - **Learning objective:**  
     `"Play a move that creates both a line and a disconnected region, and then resolve them in the correct order."`
   - **Board description:**  
     Start from a pre‑built midgame; the player is prompted to execute a particular movement / capture.
   - **Expected outcome:**  
     After the move:
     - Chain capture (if available).
     - Line processing (collapsed markers, elimination).
     - Territory processing (disconnected regions, interior + self‑elimination).
     - Victory check (if thresholds are crossed).
   - **Rules anchor:**  
     [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:327) R120–R122; [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:574) R144; SCEN‑TERRITORY‑003 in [`archive/RULES_DYNAMIC_VERIFICATION.md`](archive/RULES_DYNAMIC_VERIFICATION.md:459).

3. **Step id:** `teaching.multi_phase.step_3`
   - **Index:** 3
   - **Kind:** `guided`
   - **Learning objective:**  
     `"See how different line‑processing choices (Option 1 vs Option 2) affect later territory and victory."`
   - **Board description:**  
     Overlength line scenario where choosing Option 1 vs Option 2 changes whether a region becomes disconnected or whether victory thresholds are crossed.
   - **Expected outcome:**  
     The overlay compares “Option 1 branch” vs “Option 2 branch” outcomes.

---

### 3.5 Flow: Capture Chains & Mandatory Continuation – `capture_chain_mandatory`

- **Flow id:** `capture_chain_mandatory`
- **Primary concept:** `capture_chain_mandatory`
- **Weird‑state reason code:** none (but strongly connected to DOCUX‑P1 in [`RULES_DOCS_UX_AUDIT.md`](docs/supplementary/RULES_DOCS_UX_AUDIT.md:29) – HUD misleading chain capture text).
- **Recommended board:** `square8` and `square19` variants.

#### Steps

1. **Step id:** `teaching.capture_chain.step_1`
   - **Index:** 1
   - **Kind:** `guided`
   - **Learning objective:**  
     `"See that starting a capture is optional, but once started, continuation is mandatory while any capture exists."`
   - **Board description:**  
     Simple 8×8 example where a capture is available but not forced.
   - **Expected outcome:**  
     Text emphasises optional start vs mandatory continuation.

2. **Step id:** `teaching.capture_chain.step_2`
   - **Index:** 2
   - **Kind:** `interactive`
   - **Learning objective:**  
     `"Perform a short chain capture and choose among multiple continuation directions."`
   - **Board description:**  
     3–4 legal continuation segments from a central stack (a simplified version of FAQ Q15).
   - **Expected outcome:**  
     Player must:
     - Start a capture.
     - Choose one continuation direction.
     - See that chain must continue until no captures remain.

3. **Step id:** `teaching.capture_chain.step_3`
   - **Index:** 3
   - **Kind:** `interactive`
   - **Learning objective:**  
     `"End a chain early by choosing a continuation that leads to no further captures, even when other longer chains exist."`
   - **Board description:**  
     Position where:
     - From the current landing square, you have:
       - A continuation that leads to a longer chain.
       - A continuation that ends the chain immediately.
   - **Expected outcome:**  
     Player chooses the branch that ends the chain, internalising the “mandatory continuation but choice of path” rule.

---

### 3.6 Flow: Landing on Own Marker – `landing_on_own_marker_intro`

- **Flow id:** `landing_on_own_marker_intro`
- **Primary concept:** `landing_on_own_marker`
- **Recommended board:** `square8`.
- **Rules anchor:** [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:418) R092.

#### Steps

1. **Step id:** `teaching.landing_marker.step_1`
   - **Index:** 1
   - **Kind:** `guided`
   - **Learning objective:**  
     `"Understand that landing on any marker (including your own) eliminates the top ring of your stack."`
   - **Board description:**  
     Simple movement path across a mix of empty and marker spaces, emphasising the landing cell.
   - **Expected outcome:**  
     Text explains:
     - Path vs landing.
     - Flip/collapse vs elimination on landing.

2. **Step id:** `teaching.landing_marker.step_2`
   - **Index:** 2
   - **Kind:** `interactive`
   - **Learning objective:**  
     `"Choose between a safe landing and a risky landing that sacrifices your top ring."`
   - **Board description:**  
     Player has two candidate landing cells:
     - One empty (no elimination).
     - One with own marker (elimination).
   - **Expected outcome:**  
     Player sees the elimination occur and how it counts toward elimination totals.

---

### 3.7 Flow: Structural Stalemate & Tiebreak – `structural_stalemate_intro`

- **Flow id:** `structural_stalemate_intro`
- **Primary concept:** `structural_stalemate`
- **Weird‑state reason code:** `STRUCTURAL_STALEMATE_TIEBREAK`.
- **Recommended board:** `square8`, `numPlayers = 3`.

#### Steps

1. **Step id:** `teaching.structural_stalemate.step_1`
   - **Index:** 1
   - **Kind:** `guided`
   - **Learning objective:**  
     `"Recognise when no moves or forced eliminations remain for any player."`
   - **Board description:**  
     An end position where:
     - No stacks remain.
     - Some markers and collapsed territory exist.
     - No placements satisfy no‑dead‑placement and caps.
   - **Expected outcome:**  
     Explanation of:
     - Why no moves are available.
     - How this differs from ANM for a single player.

2. **Step id:** `teaching.structural_stalemate.step_2`
   - **Index:** 2
   - **Kind:** `guided`
   - **Learning objective:**  
     `"Understand the tie‑break ladder: territory → eliminated rings → markers → last actor."`
   - **Board description:**  
     Same or simplified board; overlay points out each count and shows who wins.
   - **Rules anchor:**  
     [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1381) §13.4; [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:619) R173.

---

### 3.8 Flow: Last Player Standing – `last_player_standing_intro`

- **Flow id:** `last_player_standing_intro`
- **Primary concept:** `last_player_standing`
- **Weird‑state reason code:** `LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS`.
- **Recommended board:** `square8`, `numPlayers = 3`.

#### Steps

1. **Step id:** `teaching.lps.step_1`
   - **Index:** 1
   - **Kind:** `guided`
   - **Learning objective:**  
     `"Distinguish real actions (placements, moves, captures) from forced elimination and automatic processing."`
   - **Board description:**  
     Example where some players have only FE or no moves, while one player can still move.
   - **Expected outcome:**  
     Overlay labels each player’s status:
     - “Has real moves”.
     - “Only forced elimination”.
     - “Inactive / fully eliminated”.

2. **Step id:** `teaching.lps.step_2`
   - **Index:** 2
   - **Kind:** `guided`
   - **Learning objective:**  
     `"See how a full round with only one player having real moves yields LPS victory."`
   - **Board description:**  
     Narrative / pseudo‑turn walkthrough; no actual moves from the user.
   - **Expected outcome:**  
     Text traces the condition in [`RR‑CANON‑R172`](RULES_CANONICAL_SPEC.md:603).

3. **Step id:** `teaching.lps.step_3`
   - **Index:** 3
   - **Kind:** `interactive` (optional; can be deferred until engines explicitly support early LPS).
   - **Learning objective:**  
     `"Play from a position where you are about to become Last Player Standing."`
   - **Board description:**  
     Setup where one move by the player can leave others without real actions.
   - **Expected outcome:**  
     Once implemented, the scenario should end by LPS.

---

## 4. Surfaces Mapping

For each flow, this table indicates where it should appear.

| Flow id                       | TeachingOverlay | Sandbox presets | Tutorial carousel | Tied weird‑state reason                                 |
| :---------------------------- | :-------------: | :-------------: | :---------------: | :------------------------------------------------------ |
| `fe_loop_intro`               |       ✅        |       ✅        |        ✅         | `ANM_MOVEMENT_FE_BLOCKED`, `FE_SEQUENCE_CURRENT_PLAYER` |
| `mini_region_intro`           |       ✅        |       ✅        |        ◻️         | —                                                       |
| `multi_region_budget`         |       ✅        |       ✅        |        ◻️         | —                                                       |
| `line_vs_territory_blocking`  |       ✅        |       ✅        |        ✅         | —                                                       |
| `capture_chain_mandatory`     |       ✅        |       ✅        |        ✅         | —                                                       |
| `landing_on_own_marker_intro` |       ✅        |       ✅        |        ◻️         | —                                                       |
| `structural_stalemate_intro`  |       ✅        |       ✅        |        ✅         | `STRUCTURAL_STALEMATE_TIEBREAK`                         |
| `last_player_standing_intro`  |       ✅        |       ◻️        |        ✅         | `LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS`           |

Suggested runtime behaviour:

- **TeachingOverlay:** exposes flows grouped by `rulesConcept` and `difficultyTag`. When a weird‑state overlay is triggered (e.g. FE or structural stalemate), show a “Related teaching flows” list using the appropriate flows.
- **Sandbox presets:** present curated teaching scenarios under a “Rules Clinic” tab, sorted by difficulty.
- **Tutorial carousel:** for new players, present a minimal sequence:
  - Capture & movement basics (existing).
  - `capture_chain_mandatory`.
  - `line_vs_territory_blocking`.
  - `mini_region_intro` step 1/2.
  - `fe_loop_intro` step 1/2.

---

## 5. Telemetry Integration

Flows SHOULD cooperate with the telemetry schema in [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:218):

- Each `TeachingScenarioMetadata` entry MUST define:
  - `rulesConcept` → mapped to `rules_context`.
  - `flowId` and `stepIndex`.
  - Optional `uxWeirdStateReasonCode` for association with `weird_state_overlay_shown` and `resign_after_weird_state` events.
- The following events SHOULD be emitted for teaching flows (see W‑UX‑1/W‑UX‑4):
  - `sandbox_scenario_loaded` / `sandbox_scenario_completed`.
  - `teaching_step_started` / `teaching_step_completed`.

This enables W‑UX‑4 to answer questions such as:

- “Do players who went through `fe_loop_intro` resign less often after ANM/FE weird states?”
- “Which teaching steps have a low completion rate, suggesting copy/board complexity issues?”

---

## 6. Implementation Notes for Code‑Mode

When a Code‑mode task implements these flows, it should:

1. **Define scenario metadata** in a central place (e.g. `teachingScenarios.json` or an extended section of `curated.json`), matching the shape in §2.
2. **Reuse or extend existing scenario infrastructure**:
   - Scenario loader [`src/client/sandbox/scenarioLoader.ts`](src/client/sandbox/scenarioLoader.ts:112).
   - Curated scenario config [`src/client/public/scenarios/curated.json`](src/client/public/scenarios/curated.json:175).
3. **Keep rules semantics authoritative**:
   - Any scenario must obey the shared TS engine semantics (no “special rules”).
   - Where ambiguity exists, prefer canonical semantics from [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:1).
4. **Add tests**:
   - For each new scenario family, add at least one test that:
     - Loads the scenario.
     - Verifies basic invariants (board type, number of players, concept tags).
     - Asserts that completing the scenario produces the expected rule outcome (correct territory / elimination / FE behaviour).

This spec, together with [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:1) and [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:1), provides the blueprint Code‑mode can follow to implement scenario‑driven teaching for the most confusing parts of the RingRift ruleset.
