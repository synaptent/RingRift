# Rules UX Improvement Iteration 0002 – Hotspot‑Driven Game‑End Explanations

- **Iteration id:** 0002
- **Axis:** `rules-ux`
- **Mode:** Architect spec → Code implementation
- **Status:** **Backfilled iteration record for work delivered in waves W1–W5** (routing, teaching flows, game‑end explanations, copy alignment). Metrics and time windows described below are **illustrative / synthetic** rather than live telemetry.

## 1. Introduction and framing

This iteration builds directly on the weakest‑aspect assessment in [`WEAKNESS_AND_HARDEST_PROBLEM_REPORT.md`](WEAKNESS_AND_HARDEST_PROBLEM_REPORT.md:210), which identifies **rules UX and onboarding** as RingRift’s weakest area, and on the rules‑UX system described in [`UX_RULES_IMPROVEMENT_LOOP.md`](docs/UX_RULES_IMPROVEMENT_LOOP.md:24). It also assumes the concepts vocabulary in [`UX_RULES_CONCEPTS_INDEX.md`](docs/UX_RULES_CONCEPTS_INDEX.md:17) and the structured game‑end payload model [`GameEndExplanation`](docs/UX_RULES_EXPLANATION_MODEL_SPEC.md:129) are the canonical reference points for rules concepts and explanations.

Iteration 0002 is explicitly **hotspot‑driven**. It assumes the existence of rules‑UX telemetry aggregates conforming to [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:13) and pre‑aggregated hotspot snapshots for Square‑8 2‑player games produced by the analyzer [`scripts/analyze_rules_ux_telemetry.ts`](scripts/analyze_rules_ux_telemetry.ts:1). The goal is to turn those hotspots into concrete changes to game‑end explanations (HUD + VictoryModal), teaching flows, and (optionally) sandbox/debug UX, without redefining underlying rules semantics.

Compared to Iteration 0001 ([`UX_RULES_IMPROVEMENT_ITERATION_0001.md`](docs/ux/rules_iterations/UX_RULES_IMPROVEMENT_ITERATION_0001.md:1)), which primarily focused on **copy routing and teaching flows**, Iteration 0002 treats [`GameEndExplanation`](docs/UX_RULES_EXPLANATION_MODEL_SPEC.md:129) as the central contract for _“Why did the game end?”_ and scopes changes around using that model consistently for a small set of high‑confusion concepts.

## 2. Inputs and required artifacts

Even when live data is not embedded in this document, Iteration 0002 assumes the following logical inputs are available for each run:

### 2.1 Telemetry snapshot and hotspot report

- A **pre‑aggregated telemetry JSON snapshot** for Square‑8 2‑player games over a clearly defined window (e.g. the last 28 days), matching the [`RulesUxAggregatesRoot`](src/shared/telemetry/rulesUxHotspotTypes.ts:1) shape described in the telemetry tooling docs and consumed by [`scripts/analyze_rules_ux_telemetry.ts`](scripts/analyze_rules_ux_telemetry.ts:1).
- A **hotspot summary JSON** and short Markdown report emitted by [`scripts/analyze_rules_ux_telemetry.ts`](scripts/analyze_rules_ux_telemetry.ts:425) for the same window. These must, at minimum, expose per‑`rulesContext` metrics for:
  - help opens per 100 games;
  - help reopen rate;
  - resign‑after‑weird‑state rate;
  - per‑source breakdowns (HUD, VictoryModal, TeachingOverlay, sandbox) where available.
- The snapshot must be filterable on the following dimensions (even if not all are used in this iteration):
  - `rulesContext` / `rules_concept` (see [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:93));
  - `source` / `RulesUxSource` (HUD, VictoryModal, TeachingOverlay, sandbox) as defined in [`rulesUxEvents.ts`](src/shared/telemetry/rulesUxEvents.ts:78);
  - board type `square8` and `num_players = 2`.

For Iteration 0002, the hotspot report is expected to confirm that contexts aligned with the concepts `anm_fe_core`, `structural_stalemate`, and `territory_mini_regions` remain in the **top band of confusion** (high help opens per 100 games and/or elevated resign‑after‑weird‑state ratios). If live telemetry is not yet available, these contexts are treated as **pre‑telemetry defaults** based on historical audits.

### 2.2 Concepts and teaching catalogue

- Canonical high‑risk concepts and linkages from [`UX_RULES_CONCEPTS_INDEX.md`](docs/UX_RULES_CONCEPTS_INDEX.md:17), especially:
  - `anm_fe_core` – Active‑No‑Moves & Forced Elimination;
  - `structural_stalemate` – plateau endings and tiebreak ladder;
  - `territory_mini_regions` – mini‑regions / Q23‑style self‑elimination.
- Teaching flows and scenarios from [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:141), particularly:
  - flow `fe_loop_intro` for ANM/FE loops;
  - flow `structural_stalemate_intro` for structural stalemate endings;
  - flow `mini_region_intro` for mini‑regions and self‑elimination cost.
- Existing TeachingOverlay topics and routing described in [`TeachingOverlay.tsx`](src/client/components/TeachingOverlay.tsx:84) and implemented metadata in [`teachingScenarios.ts`](src/shared/teaching/teachingScenarios.ts:51).

### 2.3 Explanation model and weird‑state mapping

- The structured game‑end payload model [`GameEndExplanation`](docs/UX_RULES_EXPLANATION_MODEL_SPEC.md:129), including:
  - `primaryConceptId`, `rulesContextTags`, and `weirdStateContext` fields;
  - `uxCopy.shortSummaryKey` and `uxCopy.detailedSummaryKey` keys for HUD and VictoryModal;
  - `teaching.teachingTopics` and `teaching.recommendedFlows` linkage to teaching content;
  - `telemetry.rulesContext` and `telemetry.weirdStateReasonCode` for low‑cardinality logging.
- Weird‑state reason codes and mappings from [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:71), especially:
  - `ANM_MOVEMENT_FE_BLOCKED` and `FE_SEQUENCE_CURRENT_PLAYER` for ANM/FE;
  - `STRUCTURAL_STALEMATE_TIEBREAK` for structural stalemate;
  - `ANM_TERRITORY_NO_ACTIONS` for territory‑phase mini‑region exits.

## 3. Selected target concepts for Iteration 0002

Iteration 0002 focuses on **three** concepts from [`UX_RULES_CONCEPTS_INDEX.md`](docs/UX_RULES_CONCEPTS_INDEX.md:21). In each case, the intent is to standardise how [`GameEndExplanation`](docs/UX_RULES_EXPLANATION_MODEL_SPEC.md:129) represents the concept and how HUD, VictoryModal, TeachingOverlay, sandbox, and telemetry consume that representation.

### 3.1 `anm_fe_core` – Active‑No‑Moves & Forced Elimination

- **Concept id:** `anm_fe_core`.
- **Canonical telemetry contexts:**
  - `rulesContext = 'anm_forced_elimination'` (see [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:93));
  - `weirdStateType ∈ { 'active-no-moves-movement', 'forced-elimination' }` per [`RulesUxWeirdStateType`](src/shared/telemetry/rulesUxEvents.ts:46).
- **Rationale for selection:**
  - Historically top source of "I have pieces but no moves and my caps disappeared" complaints (see [`WEAKNESS_AND_HARDEST_PROBLEM_REPORT.md`](WEAKNESS_AND_HARDEST_PROBLEM_REPORT.md:104) and [`ACTIVE_NO_MOVES_BEHAVIOUR.md`](docs/ACTIVE_NO_MOVES_BEHAVIOUR.md:1)).
  - Already a primary focus of Iteration 0001, but still expected to be a major hotspot in telemetry (`help_open` and `resign_after_weird_state` events for `anm_forced_elimination`).
  - Directly underpins many endings that are _perceived_ as unfair ring‑elimination or hidden Last‑Player‑Standing logic.
- **Explanation‑model alignment:**
  - For games where ANM/FE had a material impact on the outcome (e.g. multiple FE sequences on the eventual loser), [`GameEndExplanation.primaryConceptId`](docs/UX_RULES_EXPLANATION_MODEL_SPEC.md:139) SHOULD often be set to `'anm_fe_core'` even if the formal `outcomeType` is `ring_elimination` or `last_player_standing`.
  - `weirdStateContext.reasonCodes` SHOULD include `ANM_MOVEMENT_FE_BLOCKED` and/or `FE_SEQUENCE_CURRENT_PLAYER` when those reason codes fired in the endgame turns, with `weirdStateContext.primaryReasonCode` pointing to whichever sequence best explains the player’s perspective.
  - `rulesContextTags` and `telemetry.rulesContext` SHOULD include `anm_forced_elimination` whenever ANM/FE is treated as the primary concept for the ending.

### 3.2 `structural_stalemate` – Structural stalemates and tiebreak ladder

- **Concept id:** `structural_stalemate`.
- **Canonical telemetry contexts:**
  - `rulesContext = 'structural_stalemate'` (see [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:96));
  - `weirdStateType = 'structural-stalemate'`.
- **Rationale for selection:**
  - Plateau endings resolved via territory → eliminated rings → markers → last actor are intrinsically non‑obvious and have historically been mis‑perceived as draws or bugs (see [`WEAKNESS_AND_HARDEST_PROBLEM_REPORT.md`](WEAKNESS_AND_HARDEST_PROBLEM_REPORT.md:106)).
  - Iteration 0001 defined copy and teaching flows but did not yet standardise the structured tiebreak representation in [`GameEndExplanation.tiebreakSteps`](docs/UX_RULES_EXPLANATION_MODEL_SPEC.md:141).
  - Telemetry for stalemate endings is critical to understanding whether explanations meaningfully reduce resigns and help reopens after these outcomes.
- **Explanation‑model alignment:**
  - For any game that terminates via the canonical global stalemate ladder ([`RR‑CANON R173`](RULES_CANONICAL_SPEC.md:619)), [`GameEndExplanation.outcomeType`](docs/UX_RULES_EXPLANATION_MODEL_SPEC.md:137) SHOULD be `structural_stalemate`, `victoryReasonCode` SHOULD be `victory_structural_stalemate_tiebreak`, and `primaryConceptId` SHOULD be `'structural_stalemate'`.
  - `tiebreakSteps` MUST be populated to reflect each ladder step actually consulted (territory spaces, eliminated rings including rings in hand, markers, last real action).
  - `weirdStateContext.reasonCodes` SHOULD contain `STRUCTURAL_STALEMATE_TIEBREAK` with matching `rulesContextTags = ['structural_stalemate']`.

### 3.3 `territory_mini_regions` – Mini‑regions / Q23‑style self‑elimination

- **Concept id:** `territory_mini_regions`.
- **Canonical telemetry contexts:**
  - `rulesContext = 'territory_mini_region'` (see [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:98));
  - `weirdStateType = 'active-no-moves-territory'` when represented via `ANM_TERRITORY_NO_ACTIONS`.
- **Rationale for selection:**
  - Mini‑region scenarios (FAQ Q23) are a concentrated source of “why did I lose my own ring?” confusion, despite solid engine semantics and tests ([`RulesMatrix.Territory.MiniRegion.test.ts`](tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts:19)).
  - Iteration 0001 targeted teaching and discoverability; Iteration 0002 focuses on **game‑end explanations** whenever a mini‑region self‑elimination materially affects the terminal score.
  - Telemetry for `territory_mini_region` contexts is important to validate whether these explanations and teaching flows are actually reducing repeated help opens.
- **Explanation‑model alignment:**
  - For endings where a mini‑region decision directly shifts the territory or elimination balance in the final scoring window, [`GameEndExplanation.primaryConceptId`](docs/UX_RULES_EXPLANATION_MODEL_SPEC.md:139) SHOULD often be `'territory_mini_regions'`, even if the `outcomeType` is standard `territory_control` or `ring_elimination`.
  - `weirdStateContext.reasonCodes` SHOULD include `ANM_TERRITORY_NO_ACTIONS` when the engine auto‑exits from a territory phase after mini‑region processing with no further actions available.
  - `rulesContextTags` and `telemetry.rulesContext` SHOULD include `territory_mini_region` whenever a mini‑region archetype is the primary explanatory lens for the ending.

## 4. UX changes – high‑level design

This section specifies **what needs to change** at a UX contract level for Iteration 0002. It is intentionally implementation‑agnostic: Code‑mode tasks are responsible for introducing concrete builder functions, wiring, and tests that satisfy these requirements.

### 4.1 Game‑end explanation surfaces (HUD + VictoryModal)

For each targeted concept, [`GameEndExplanation`](docs/UX_RULES_EXPLANATION_MODEL_SPEC.md:129) SHOULD be treated as the **single source of truth** for why the game ended and why a particular player won. HUD and VictoryModal MUST consume this payload rather than re‑deriving explanations from raw `GameResult` or ad‑hoc flags.

#### 4.1.1 Common requirements across concepts

- Introduce or standardise a shared builder such as [`buildGameEndExplanation()`](docs/UX_RULES_EXPLANATION_MODEL_SPEC.md:425) that produces a complete [`GameEndExplanation`](docs/UX_RULES_EXPLANATION_MODEL_SPEC.md:129) instance for every completed game (backend host and sandbox host).
- Ensure the builder:
  - Sets `primaryConceptId` to one of `anm_fe_core`, `structural_stalemate`, or `territory_mini_regions` when the corresponding concept is the _primary explanatory lens_ for the ending;
  - Populates `rulesContextTags` from the same vocabulary as [`RulesUxContext`](src/shared/telemetry/rulesUxEvents.ts:60);
  - Fills `weirdStateContext.reasonCodes` with the relevant weird‑state reason codes from [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:82) when present;
  - Sets `telemetry.rulesContext` to the single best `RulesUxContext` for the ending (typically matching the selected concept’s main `rulesContext`).
- HUD end‑of‑game banners and compact status text MUST derive their labels and tooltips from `uxCopy.shortSummaryKey` and `primaryConceptId`, not from scattered `if`/`switch` logic.
- VictoryModal MUST use:
  - `uxCopy.detailedSummaryKey` for the main explanatory paragraph(s);
  - `scoreBreakdown` and, when present, `tiebreakSteps` to drive any structured score tables for stalemate / territory outcomes;
  - `rulesReferences` to surface deep‑link affordances into formal rules / FAQ docs.

#### 4.1.2 Concept‑specific expectations

**A. `anm_fe_core` (ANM/FE)**

- When ANM/FE activity is judged to be _primary_ to the perceived ending (multiple FE sequences, or final elimination reached via FE rather than captures or territory):
  - `primaryConceptId` SHOULD be `'anm_fe_core'`.
  - `weirdStateContext.reasonCodes` SHOULD list `ANM_MOVEMENT_FE_BLOCKED` and/or `FE_SEQUENCE_CURRENT_PLAYER` in temporal order, with `primaryReasonCode` set to whichever reason best captures the final decisive FE sequence.
  - `rulesContextTags` SHOULD include `anm_forced_elimination` and optionally companion tags such as `last_player_standing` when relevant.
- Expected copy‑key contracts (no literal copy specified):
  - HUD / compact banner keys under `uxCopy.shortSummaryKey` SHOULD follow a pattern such as `game_end.ring_elimination.with_anm_fe.short` or `game_end.lps.with_anm_fe.short` when FE heavily shaped the outcome.
  - VictoryModal keys under `uxCopy.detailedSummaryKey` SHOULD follow patterns like `game_end.ring_elimination.with_anm_fe.long` or reuse the FE‑focused explanation keys described in [`UX_RULES_EXPLANATION_MODEL_SPEC.md`](docs/UX_RULES_EXPLANATION_MODEL_SPEC.md:261) and [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:211).
- `debug` SHOULD include small, QA‑oriented fields such as:
  - counts of FE events per player in the last N turns;
  - whether any ANM/FE invariants from [`ACTIVE_NO_MOVES_BEHAVIOUR.md`](docs/ACTIVE_NO_MOVES_BEHAVIOUR.md:39) were involved.

**B. `structural_stalemate`**

- For endings resolved via the canonical stalemate ladder:
  - `outcomeType` MUST be `structural_stalemate`;
  - `victoryReasonCode` MUST be `victory_structural_stalemate_tiebreak`;
  - `primaryConceptId` MUST be `'structural_stalemate'`.
- `tiebreakSteps` MUST include at least one step with `kind = 'territory_spaces'` and SHOULD include additional steps when ties propagate to eliminated rings, markers, or last real action.
- `weirdStateContext.reasonCodes` MUST contain `STRUCTURAL_STALEMATE_TIEBREAK` with `rulesContextTags = ['structural_stalemate']`.
- Expected copy‑key contracts:
  - HUD summary key under `uxCopy.shortSummaryKey` SHOULD be a stable key such as `game_end.structural_stalemate.short`.
  - VictoryModal explanation under `uxCopy.detailedSummaryKey` SHOULD use a key such as `game_end.structural_stalemate.tiebreak.detailed`, matching the narrative structure outlined in [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:221).
- `debug` SHOULD expose the computed ladder inputs (territory counts, eliminated ring totals including rings in hand, marker counts, id of last real actor) for QA to compare against expectations from [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:619).

**C. `territory_mini_regions`**

- For endings where mini‑region processing and its self‑elimination cost materially affect the final score:
  - `primaryConceptId` SHOULD be `'territory_mini_regions'`.
  - `rulesContextTags` SHOULD include `territory_mini_region`.
  - `weirdStateContext.reasonCodes` SHOULD contain `ANM_TERRITORY_NO_ACTIONS` when the engine auto‑exits territory processing immediately after a mini‑region has been fully processed and no further region actions are possible.
- Expected copy‑key contracts:
  - HUD summary key under `uxCopy.shortSummaryKey` SHOULD follow the pattern `game_end.territory.mini_region.short`.
  - VictoryModal explanation key under `uxCopy.detailedSummaryKey` SHOULD follow the pattern `game_end.territory.mini_region.detailed`, aligning with the worked example in [`UX_RULES_EXPLANATION_MODEL_SPEC.md`](docs/UX_RULES_EXPLANATION_MODEL_SPEC.md:393).
- `rulesReferences.rulesDocsLinks` SHOULD always include anchors to FAQ Q23 and the relevant canonical rules sections referenced in [`UX_RULES_CONCEPTS_INDEX.md`](docs/UX_RULES_CONCEPTS_INDEX.md:25).

### 4.2 Teaching surfaces (TeachingOverlay and flows)

When a game ends and a [`GameEndExplanation`](docs/UX_RULES_EXPLANATION_MODEL_SPEC.md:129) is present, TeachingOverlay SHOULD use the explanation payload rather than ad‑hoc heuristics to decide which topics and flows to promote. For Iteration 0002:

- For `primaryConceptId = 'anm_fe_core'`:
  - `teaching.teachingTopics` SHOULD include an ANM/FE‑aligned topic id such as `anm_forced_elimination` (or equivalent, as defined in [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:141)).
  - `teaching.recommendedFlows` SHOULD prioritise `fe_loop_intro` and any extended FE loop flows added after Iteration 0001.
  - VictoryModal and HUD “What happened?” entrypoints SHOULD, when present, open TeachingOverlay pre‑filtered to the `fe_loop_intro` flow for the relevant player perspective.
- For `primaryConceptId = 'structural_stalemate'`:
  - `teaching.teachingTopics` SHOULD include `structural_stalemate`.
  - `teaching.recommendedFlows` SHOULD include `structural_stalemate_intro` (see [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:426)).
  - VictoryModal SHOULD surface a single, clear teaching entrypoint (button or link) that opens TeachingOverlay on `structural_stalemate_intro`, using the same `rulesContext` used for telemetry.
- For `primaryConceptId = 'territory_mini_regions'`:
  - `teaching.teachingTopics` SHOULD include `territory_mini_region`.
  - `teaching.recommendedFlows` SHOULD include `mini_region_intro`.
  - When a player opens help from a territory banner or end‑of‑game screen tagged with `rulesContext = 'territory_mini_region'`, TeachingOverlay SHOULD default to showing `mini_region_intro` in the recommended list.

In all three cases, TeachingOverlay SHOULD emit `teaching_step_started` and `teaching_step_completed` telemetry tagged with the same `rulesContext` that appears in `GameEndExplanation.telemetry.rulesContext`, so that W‑UX hotspot analyses can relate post‑game teaching engagement to specific endings.

### 4.3 Sandbox and debug UX (optional for this iteration)

Sandbox and internal debug tooling are **out of scope** for end‑user UX changes in Iteration 0002, but a small amount of additional instrumentation is recommended for QA and developer workflows:

- In sandbox or replay views where a completed game is shown, expose a developer‑only panel or toggle that renders the final [`GameEndExplanation`](docs/UX_RULES_EXPLANATION_MODEL_SPEC.md:129) payload alongside the board, especially for curated scenarios covering:
  - ANM/FE loop archetypes for `anm_fe_core`;
  - plateau stalemate test cases for `structural_stalemate`;
  - mini‑region invariants from [`RulesMatrix.Territory.MiniRegion.test.ts`](tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts:19) for `territory_mini_regions`.
- For each of these scenario families, document (in code comments or test names) which concept and `rulesContext` they are expected to exercise, so that QA can quickly validate that the produced `GameEndExplanation` matches this spec.

## 5. Acceptance criteria for Iteration 0002

From an architect/UX perspective, Iteration 0002 is complete when the following behaviours are true in live builds (for Square‑8 2‑player games) and verifiable via tests and telemetry. These criteria are **concept‑centred** rather than implementation‑centred.

### 5.1 Concept‑aligned explanations

For each targeted concept (`anm_fe_core`, `structural_stalemate`, `territory_mini_regions`):

- Whenever the corresponding concept is the **primary driver of the ending**, the final [`GameEndExplanation`](docs/UX_RULES_EXPLANATION_MODEL_SPEC.md:129) for that game:
  - sets `primaryConceptId` to the concept id;
  - includes at least one appropriate `rulesContextTags` entry (`anm_forced_elimination`, `structural_stalemate`, or `territory_mini_region`);
  - includes any relevant weird‑state reason codes in `weirdStateContext.reasonCodes` (e.g. `ANM_MOVEMENT_FE_BLOCKED`, `FE_SEQUENCE_CURRENT_PLAYER`, `STRUCTURAL_STALEMATE_TIEBREAK`, `ANM_TERRITORY_NO_ACTIONS`);
  - sets `telemetry.rulesContext` consistently with these tags.
- HUD and VictoryModal always render game‑end summaries using `uxCopy.shortSummaryKey` and `uxCopy.detailedSummaryKey` that are **consistent with the selected concept** (e.g. stalemate keys for `structural_stalemate`, territory mini‑region keys for `territory_mini_regions`).

### 5.2 Teaching entrypoints

- For each targeted concept, at least one **teaching entrypoint** is present on relevant end‑game screens:
  - ANM/FE endings: a help/teaching entrypoint from VictoryModal or HUD routes into the `fe_loop_intro` flow with `rulesContext = 'anm_forced_elimination'`.
  - Structural stalemate endings: an entrypoint routes into `structural_stalemate_intro` with `rulesContext = 'structural_stalemate'`.
  - Territory mini‑region‑influenced endings: an entrypoint routes into `mini_region_intro` with `rulesContext = 'territory_mini_region'`.
- These entrypoints are backed by telemetry (`help_open`, `weird_state_details_open`, `teaching_step_*`) that consistently label the underlying `rulesContext`, allowing hotspot reports to distinguish improvements or regressions by concept.

### 5.3 Telemetry alignment

- For Square‑8 2‑player games, hotspot snapshots produced by [`scripts/analyze_rules_ux_telemetry.ts`](scripts/analyze_rules_ux_telemetry.ts:425) include **non‑zero** entries for each targeted `rulesContext`.
- For representative post‑iteration windows, events associated with the targeted concepts (help opens, help reopens, resign‑after‑weird‑state) correctly reference the `rulesContext` values and, where applicable, weird‑state reason codes in line with the mappings in [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:332) and [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:336).
- It is possible to compute for each targeted concept:
  - help opens per 100 games;
  - help reopen fraction;
  - resign‑after‑weird‑state ratio
    using only `rulesContext`‑level aggregates and the rules‑UX hotspot analyzer.

### 5.4 QA scenarios and tests

- For each concept, at least one **named test scenario** or existing test is explicitly documented as the canonical QA hook for validating the new UX behaviour, for example:
  - ANM/FE: invariants in [`test_active_no_moves_movement_forced_elimination_regression.py`](ai-service/tests/invariants/test_active_no_moves_movement_forced_elimination_regression.py:1) and related ANM suites; any new game‑end UX tests that assert the produced [`GameEndExplanation`](docs/UX_RULES_EXPLANATION_MODEL_SPEC.md:129) matches expectations in FE‑heavy endings.
  - Structural stalemate: stalemate/plateau scenarios used in engine tests and parity fixtures as listed in [`UX_RULES_CONCEPTS_INDEX.md`](docs/UX_RULES_CONCEPTS_INDEX.md:24); end‑to‑end tests that assert `tiebreakSteps` and HUD/VictoryModal explanations align.
  - Territory mini‑regions: [`RulesMatrix.Territory.MiniRegion.test.ts`](tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts:19) and helper tests in [`territoryDecisionHelpers.shared.test.ts`](tests/unit/territoryDecisionHelpers.shared.test.ts:76) extended or wrapped to assert correct `primaryConceptId` and `rulesContextTags` in `GameEndExplanation` for Q23‑style endings.
- These scenarios are enumerated in Code‑mode test files and can be used by QA to confirm that behaviour matches this spec across HUD, VictoryModal, teaching entrypoints, and telemetry.

## 6. Implementation plan and hand‑off to Code mode

The following are **proposed implementation tasks** to be tracked as separate Code‑mode subtasks. They are intentionally high‑level; this document is the **scope authority** for what Iteration 0002 should achieve.

1. **Implement and wire [`buildGameEndExplanation()`](docs/UX_RULES_EXPLANATION_MODEL_SPEC.md:425) for targeted concepts**
   - Introduce or finalise a shared builder that produces [`GameEndExplanation`](docs/UX_RULES_EXPLANATION_MODEL_SPEC.md:129) instances for completed games, with explicit handling for `anm_fe_core`, `structural_stalemate`, and `territory_mini_regions` as defined in §3–§4.
   - Wire the builder into the backend game host, sandbox host, and any replay/export paths so that HUD, VictoryModal, and sandbox/debug surfaces can consume the explanation payload.

2. **Update HUD and VictoryModal to consume [`GameEndExplanation`](docs/UX_RULES_EXPLANATION_MODEL_SPEC.md:129) for the three concepts**
   - Refactor end‑of‑game UI in [`GameHUD.tsx`](src/client/components/GameHUD.tsx:73) and [`VictoryModal.tsx`](src/client/components/VictoryModal.tsx:73) so that:
     - concept‑aligned explanations use `primaryConceptId`, `uxCopy` keys, and `tiebreakSteps` from `GameEndExplanation`;
     - weird‑state banners and “What happened?” affordances use `weirdStateContext` and `rulesReferences` instead of ad‑hoc conditions.

3. **Extend teaching flows and wiring for `anm_fe_core`, `structural_stalemate`, and `territory_mini_regions`**
   - Ensure TeachingOverlay topic selection and recommended flows are driven by `GameEndExplanation.teaching` for these concepts, reusing existing flows (`fe_loop_intro`, `structural_stalemate_intro`, `mini_region_intro`) as defined in [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:141).
   - Add or adjust TeachingOverlay routing so that end‑game entrypoints with the corresponding `primaryConceptId` or `rulesContext` open the correct teaching flows.

4. **Align rules‑UX telemetry with `GameEndExplanation` for targeted endings**
   - Ensure that emission of rules‑UX events (`help_open`, `weird_state_banner_impression`, `weird_state_details_open`, `resign_after_weird_state`, `teaching_step_*`) uses `rulesContext` and, where appropriate, `reasonCode` values that are consistent with `GameEndExplanation.telemetry` and `weirdStateContext` for the same game.
   - Verify that snapshots consumed by [`scripts/analyze_rules_ux_telemetry.ts`](scripts/analyze_rules_ux_telemetry.ts:425) can distinguish the three targeted contexts via `rulesContext` alone.

5. **Add or extend regression tests for representative scenarios**

## 7. Outcome summary (backfilled for W1–W5)

Because this document is being written after substantial W1–W5 work has already landed, this section records how those waves map onto the Iteration 0002 plan. Future iterations (0004, 0005, …) should treat this as the **pre‑iteration baseline** for game‑end explanations.

- **ANM/FE (`anm_fe_core` / `anm_forced_elimination`):**
  - Weird‑state reason codes and copy for ANM/FE were defined in [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:82) (RWS‑001, RWS‑004) and wired into HUD banners and TeachingOverlay topics (`teaching.active_no_moves`, `teaching.forced_elimination`) as part of W1/W2.
  - Rules‑UX copy baselines for ANM/FE, including the distinction between **real moves** and forced elimination, were consolidated in [`UX_RULES_COPY_SPEC.md` §2–3, §10](docs/UX_RULES_COPY_SPEC.md:23).
  - ANM/FE teaching flows (`fe_loop_intro`) were specified in [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:141) and integrated into TeachingOverlay and sandbox presets in W2/W3.

- **Structural stalemate (`structural_stalemate`):**
  - Canonical stalemate semantics and tiebreak ladder were clarified in [`ringrift_complete_rules.md` §13.4](ringrift_complete_rules.md:1410) and [`RULES_CANONICAL_SPEC.md` R173](RULES_CANONICAL_SPEC.md:619).
  - UX copy and weird‑state reason codes for structural stalemate (`STRUCTURAL_STALEMATE_TIEBREAK`) were defined in [`UX_RULES_WEIRD_STATES_SPEC.md` §3.1–3.2](docs/UX_RULES_WEIRD_STATES_SPEC.md:185) and wired into HUD/Victory surfaces in W2/W4.
  - Teaching flows for stalemate (`structural_stalemate_intro`) were added to [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:423), with TeachingOverlay routing updated in W3/W4.

- **Territory mini‑regions (`territory_mini_regions` / `territory_mini_region`):**
  - Q23 mini‑region semantics were tightened and cross‑referenced across the rulebook, canonical spec, and edge‑case docs (`RULES_CONSISTENCY_EDGE_CASES`, `RULES_DYNAMIC_VERIFICATION`).
  - The `territory_mini_region` concept row in the concordance table in [`RULES_DOCS_UX_AUDIT.md` §4](docs/supplementary/RULES_DOCS_UX_AUDIT.md:142) now ties together rules docs, HUD copy, TeachingOverlay topics, and curated scenarios.
  - A dedicated `mini_region_intro` teaching flow was authored in [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:204) and surfaced via TeachingOverlay and sandbox presets during W2/W3.

- **Game‑end explanations and structured model:**
  - The `GameEndExplanation` model and builder pipeline were designed and implemented in W4, as documented in [`UX_RULES_EXPLANATION_MODEL_SPEC.md`](docs/UX_RULES_EXPLANATION_MODEL_SPEC.md:129) and validated via tests (`GameEndExplanation.*.test.ts` in `tests/unit`).
  - HUD and VictoryModal now consume structured explanations for complex endings (ANM/FE‑heavy elimination, LPS, structural stalemate, mini‑region territory outcomes), rather than ad‑hoc string construction, bringing behaviour in line with the intent of §4 in this iteration.

- **Telemetry and hotspot analysis:**
  - The rules‑UX telemetry envelope and metrics were formalised in [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:1).
  - The hotspot analyzer CLI [`scripts/analyze_rules_ux_telemetry.ts`](scripts/analyze_rules_ux_telemetry.ts:33) and its tests (`analyze_rules_ux_telemetry.test.ts`) were implemented as part of W3, providing the concrete mechanism that future runs of Iteration 0002 will use to confirm whether ANM/FE, structural stalemate, and territory mini‑regions remain top hotspots.

In other words, while Iteration 0002 is framed as a hotspot‑driven plan, the core structural changes it describes (weird‑state reason codes, teaching flows, game‑end explanations, and telemetry wiring) have already been implemented in W1–W5. Future telemetry‑backed executions of this iteration should treat:

- the current `GameEndExplanation` builder and UX wiring as the starting point;
- this document’s acceptance criteria and metrics (§5) as the checklist for validating that behaviour; and
- the concordance table in [`RULES_DOCS_UX_AUDIT.md`](docs/supplementary/RULES_DOCS_UX_AUDIT.md:124) as the single reference for ensuring any further copy tweaks remain aligned with canonical rules.
  - For each concept, add tests or extend existing ones so that they assert:
    - the produced [`GameEndExplanation`](docs/UX_RULES_EXPLANATION_MODEL_SPEC.md:129) matches this spec for the scenario (concept id, rules contexts, weird‑state reason codes, score/tiebreak structure);
    - HUD and VictoryModal render the expected concept‑aligned explanation keys for those payloads;
    - TeachingOverlay entrypoints and telemetry are correctly wired.

6. **Document iteration outcomes in future iteration notes**
   - After implementation and a stabilisation period, capture before/after hotspot metrics for the three targeted concepts (help opens per 100 games, help reopen rates, resign‑after‑weird‑state ratios) and summarise them in a follow‑up iteration note, referencing this document as the scope authority.

All of the tasks above SHOULD be tracked as separate Code‑mode subtasks. This iteration spec [`UX_RULES_IMPROVEMENT_ITERATION_0002.md`](docs/ux/rules_iterations/UX_RULES_IMPROVEMENT_ITERATION_0002.md:1) is the **single source of truth** for what “Rules UX Iteration 0002 – Hotspot‑Driven Game‑End Explanations” is meant to achieve; implementation details, scheduling, and test harness choices belong in Code and Orchestrator modes rather than here.
