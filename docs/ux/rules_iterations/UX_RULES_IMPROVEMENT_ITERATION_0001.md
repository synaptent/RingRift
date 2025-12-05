# Rules UX Improvement Iteration 0001 – FE loops, structural stalemate, mini-regions

- **Iteration id:** 0001
- **Dates:** Planned 28-day window following first full production release with rules-UX telemetry and weird-state overlays live (hypothetical for this design).
- **Release(s):** First telemetry-complete release for Square-8 2p; subsequent minor patches within the same 28-day window.
- **Participants:**
  - Rules UX owner (product / design)
  - Telemetry & data owner (backend / data)
  - Implementation owners (frontend, server, teaching / sandbox)
  - QA & rules documentation

## 1. Focus and hypotheses

### 1.1 Targeted rules contexts

This first iteration focuses on three `rulesContext` values that combine high conceptual complexity with newly wired weird-state and teaching infrastructure:

- `anm_forced_elimination` – Active-No-Moves movement states and Forced Elimination (ANM/FE loops).
- `structural_stalemate` – global plateau endings resolved by the territory / rings / markers / last-actor ladder.
- `territory_mini_region` – disconnected mini-region territory processing (Q23 archetype) and self-elimination cost.

### 1.2 Targeted surfaces

- In-game HUD:
  - Weird-state banners and phase hints related to ANM/FE and structural stalemate (see [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:1)).
  - Territory phase banners and hints for mini-regions (Q23).
- VictoryModal:
  - Explanations for structural stalemate and games where FE heavily influenced the outcome.
- TeachingOverlay:
  - Flows `fe_loop_intro`, `mini_region_intro`, and `structural_stalemate_intro` as defined in [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:1).
- Sandbox:
  - Rules clinic scenarios for FE loops and mini-regions.

### 1.3 Hypotheses

- **H1 (ANM/FE clarity):** Clarifying ANM/FE banners and routing players into `fe_loop_intro` will reduce rapid help reopen and `resign_after_weird_state` rates for `anm_forced_elimination`.
- **H2 (Structural stalemate transparency):** Expanding structural stalemate explanations in VictoryModal and TeachingOverlay will reduce “bug” reports and resigns after `structural_stalemate` endings.
- **H3 (Mini-region understanding):** Improving Q23-style mini-region teaching will reduce help opens and reopens for `territory_mini_region` without increasing confusion elsewhere.

## 2. Telemetry focus and hotspots (planned)

Telemetry is defined in [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:1) via the `RulesUxEvent` envelope and the core counter:

```text
ringrift_rules_ux_events_total{
  type, board_type, num_players, ai_difficulty, topic, rules_concept, weird_state_type
}
```

We will treat `rules_concept` as the aggregate proxy for `rulesContext` where needed and use game counters:

```text
ringrift_games_completed_total{
  board_type, num_players, difficulty, is_ranked, terminal_reason
}
```

### 2.1 ANM/FE – `anm_forced_elimination`

Planned cuts:

- **Help opens per 100 games** (Square-8 2p, all difficulties, grouped by `rules_concept` ≈ `anm_forced_elimination`):
  - Numerator:
    - `ringrift_rules_ux_events_total{type="rules_help_open", rules_concept="anm_forced_elimination"}`.
  - Denominator:
    - `ringrift_games_completed_total{board_type="square8", num_players=2}`.
- **Rapid help reopen rate** for ANM/FE help sessions (derived metric as in §6.2 of [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:400)):
  - Fraction of help sessions tagged with `rules_context="anm_forced_elimination"` that see a `help_reopen` within 30 seconds.
- **Resign-after-weird-state ratio** for FE-related weird states (see §6.3 of [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:456)):
  - Numerator: `ringrift_rules_ux_events_total{type="rules_weird_state_resign", weird_state_type="forced-elimination"}`.
  - Denominator: `ringrift_rules_ux_events_total{type="rules_weird_state_help", weird_state_type="forced-elimination"}` or, if available, `weird_state_banner_impression` counts keyed to `anm_forced_elimination`.

Hotspot definition (for pre-change window):

- ANM/FE is considered a hotspot if, over 28 days:
  - It is in the top 3 contexts by help opens per 100 games, **and/or**
  - Its rapid reopen fraction is ≥ 0.3, **and/or**
  - Its resign-after-weird-state ratio is ≥ 0.15 with at least 50 impressions.

### 2.2 Structural stalemate – `structural_stalemate`

Planned cuts:

- **Endings by structural stalemate** (denominator):
  - `ringrift_games_completed_total{board_type="square8", num_players=2, terminal_reason="stalemate"}`.
- **Help opens and weird-state help**:
  - `ringrift_rules_ux_events_total{type="rules_help_open", rules_concept="structural_stalemate"}`.
  - `ringrift_rules_ux_events_total{type="rules_weird_state_help", weird_state_type="structural-stalemate"}`.
- **Resigns after stalemate explanation**:
  - Where available, count `resign_after_weird_state` events with `rules_context="structural_stalemate"` relative to structural-stalemate impressions.

Hotspot definition:

- Structural stalemate is considered a hotspot if:
  - ≥ 5% of Square-8 2p games end by stalemate **and** players frequently open or reopen help for this context, **or**
  - The ratio of “bug-like” reports referencing stalemate (from qualitative inputs) is high relative to its share of endings.

### 2.3 Territory mini-region – `territory_mini_region`

Planned cuts:

- **Help opens by territory mini-region concept**:
  - `ringrift_rules_ux_events_total{type="rules_help_open", rules_concept="territory_mini_region"}`.
- **Teaching flows usage** for `mini_region_intro`:
  - `sandbox_scenario_loaded` / `sandbox_scenario_completed` and `teaching_step_started` / `teaching_step_completed` for flows with `rulesConcept="territory_mini_region"` (per [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:205)).

Hotspot definition:

- Territory mini-regions are a hotspot if they sit in the top 5 contexts by help opens per 100 games and have low completion rates for associated teaching flows (`mini_region_intro`) or repeated reopen patterns.

## 3. Qualitative inputs

### 3.1 Prior audits and dynamic verification

Use the following as qualitative baselines:

- [`RULES_DOCS_UX_AUDIT.md`](docs/supplementary/RULES_DOCS_UX_AUDIT.md:1):
  - DOCUX-P1–P4 already addressed initial copy mismatches but highlight enduring confusion around chain captures, elimination thresholds, and territory self-elimination.
- [`RULES_CONSISTENCY_EDGE_CASES.md`](RULES_CONSISTENCY_EDGE_CASES.md:1):
  - Structural stalemate compromises (CCE-006) and FE selection (CCE-007) emphasise the need to explain implementation compromises clearly.
- [`RULES_DYNAMIC_VERIFICATION.md`](RULES_DYNAMIC_VERIFICATION.md:1):
  - Territory and FE scenarios (SCEN-TERRITORY-00x, ANM-SCEN-0x) provide canonical board shapes for structural stalemate and mini-region behaviour.

### 3.2 Pain points for this iteration

Planned focus areas:

1. **ANM/FE loops:**
   - Players report “I had no moves and my pieces just disappeared” with limited explanation of forced elimination semantics.
   - ANM/FE weird-state banners exist (see [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:82)) but need clearer emphasis on “no real moves” versus “no moves at all”.

2. **Structural stalemate endings:**
   - Plateau endings with tiebreak ladders (territory → eliminated rings → markers → last actor) are non-obvious; players often perceive draws or bugs instead of legitimate wins/losses.
   - VictoryModal copy exists but can under-specify the tiebreak order and how rings in hand are treated (see [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:185)).

3. **Territory mini-regions (Q23):**
   - Q23-style examples are subtle: players struggle to understand why an apparently “enclosed” region collapses and why a self-elimination cost must be paid from an outside stack.
   - [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:205) defines `mini_region_intro` but the flow is not yet strongly integrated with live weird-state or phase hints.

## 4. Proposed UX adjustments (plan only)

### 4.1 ANM/FE – `anm_forced_elimination`

**Change 4.1A – Clarify ANM/FE HUD banner language**

- **Target spec:**
  - [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:141) (RWS-001 and RWS-004 HUD copy).
  - [`UX_RULES_COPY_SPEC.md`](docs/UX_RULES_COPY_SPEC.md:1) ANM/FE HUD section.
- **Intent:**
  - Emphasise that the player has **no real moves** (placements, moves, captures) rather than “no moves” overall; explicitly connect FE cap removals to Ring Elimination totals.
- **Target code surfaces (implementation later):**
  - [`GameHUD.tsx`](src/client/components/GameHUD.tsx:1) weird-state banner rendering.
  - [`WeirdStateUxMapping`](src/client/shared/WeirdStateUxMapping.ts:1) or equivalent mapping module.
- **Tests to touch (planned):**
  - Add / update HUD copy regression tests in a `GameHudWeirdStateUxRegression.test.tsx` file under `tests/unit/`.

**Change 4.1B – Strengthen FE loop teaching flow**

- **Target spec:**
  - [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:141) `fe_loop_intro` flow.
- **Intent:**
  - Emphasise multi-turn FE loops where a player repeatedly hits ANM and loses caps across several turns; connect to structural stalemate and Last Player Standing concepts.
- **Planned adjustments:**
  - Add one additional interactive step (`teaching.fe_loop.step_4`) showing a short multi-turn FE loop that ends in either a structural stalemate or a near-LPS state.
  - Ensure this step carries `uxWeirdStateReasonCode="FE_SEQUENCE_CURRENT_PLAYER"` for telemetry linkage.
- **Target code surfaces:**
  - Scenario metadata in [`teachingScenarios.ts`](src/shared/teaching/teachingScenarios.ts:1).
  - TeachingOverlay surfaces in [`TeachingOverlay.tsx`](src/client/components/TeachingOverlay.tsx:1).
- **Tests to touch (planned):**
  - Extend a `TeachingOverlayUxRegression.test.tsx` suite to assert that `fe_loop_intro` includes the new step and emits `teaching_step_*` events with the correct `rules_context`.

### 4.2 Structural stalemate – `structural_stalemate`

**Change 4.2A – Expand VictoryModal explanation**

- **Target spec:**
  - [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:185) RWS-005 VictoryModal explanation.
- **Intent:**
  - Make the tiebreak ladder fully explicit in player-facing copy:
    - Territory spaces
    - Eliminated rings (including rings in hand)
    - Markers
    - Last real action
- **Target code surfaces:**
  - [`VictoryModal.tsx`](src/client/components/VictoryModal.tsx:1) copy for structural stalemate outcomes.
- **Tests to touch (planned):**
  - `GameEndUxRegression.test.tsx` to snapshot the updated structural stalemate explanation and ensure it is selected when termination reason == structural stalemate.

**Change 4.2B – Dedicated structural stalemate teaching flow entry point**

- **Target spec:**
  - [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:424) `structural_stalemate_intro` flow.
- **Intent:**
  - Ensure that when a game ends via structural stalemate, VictoryModal and HUD weird-state banners offer a clear “What happened?” link directly into the `structural_stalemate_intro` TeachingOverlay content.
- **Target code surfaces:**
  - VictoryModal “See details” routing into TeachingOverlay.
  - HUD weird-state banners for `STRUCTURAL_STALEMATE_TIEBREAK`.
- **Tests to touch (planned):**
  - Extend `TeachingOverlayUxRegression.test.tsx` to assert that opening stalemate help from VictoryModal routes to the correct teaching flow and rules concept.

### 4.3 Territory mini-region – `territory_mini_region`

**Change 4.3A – Clarify self-elimination cost in mini-region teaching**

- **Target spec:**
  - [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:214) mini-region steps, especially `teaching.mini_region.q23.step_3`.
- **Intent:**
  - Make the outside-stack self-elimination cost explicit and visually emphasised in both text and diagram; clarify that at least one outside ring must exist to process the region.
- **Target code surfaces:**
  - TeachingOverlay step copy and diagrams for `mini_region_intro`.
- **Tests to touch (planned):**
  - Add a `ScenarioConceptsUxRegression.test.ts` suite ensuring that the mini-region teaching scenario metadata includes the correct `rulesConcept` and that sandbox flows emit the appropriate telemetry.

**Change 4.3B – Improve discoverability of mini-region scenarios from territory hints**

- **Target spec:**
  - [`UX_RULES_IMPROVEMENT_LOOP.md`](docs/UX_RULES_IMPROVEMENT_LOOP.md:1) §2.3 (Surface routing & discoverability).
- **Intent:**
  - When help is opened from a territory decision banner tied to `territory_mini_region`, the TeachingOverlay should prioritise `mini_region_intro` flows in its recommended list.
- **Target code surfaces:**
  - HUD phase hints and territory decision banners.
  - TeachingOverlay topic routing logic.
- **Tests to touch (planned):**
  - Extend `TeachingOverlayUxRegression.test.tsx` to verify that territory help entrypoints launch the correct teaching flows based on `rules_context`.

## 5. Validation plan

### 5.1 Quantitative validation (before vs after)

After changes ship and a stabilisation period (e.g. 28 days) has passed, re-run the hotspot queries from §2 using two windows:

- **Pre-change window:** last full 28 days before the release.
- **Post-change window:** days 7–35 after the release (to skip the initial rollout week).

Key comparisons:

- ANM/FE (`anm_forced_elimination`):
  - Help opens per 100 games (Square-8 2p) should **decrease** or hold steady while rapid reopen fraction decreases by ~10–20% relative.
  - Resign-after-weird-state ratio for FE weird states should trend downward, especially for games where TeachingOverlay FE flows were shown.
- Structural stalemate (`structural_stalemate`):
  - Fraction of stalemate endings with help opens that are quickly followed by resigns should decrease.
  - Where derived metrics exist, the share of stalemate-related support tickets should decrease.
- Territory mini-region (`territory_mini_region`):
  - Help opens per 100 games should decrease or remain stable while completion rates for `mini_region_intro` flows increase.

### 5.2 Qualitative validation

- Run 3–5 internal play sessions focused on:
  - Deliberately creating ANM/FE loops and observing HUD + TeachingOverlay behaviour.
  - Playing to structural stalemate endings and reading the VictoryModal explanation.
  - Walking through mini-region scenarios in TeachingOverlay and sandbox.
- Collect short-form feedback from each participant:
  - “Did the FE explanation make it obvious why caps were disappearing?”
  - “Did the stalemate explanation make it clear why the winner was chosen?”
  - “Did the mini-region tutorial make you confident you could spot similar patterns in live games?”

Success criteria are **directional**, not statistically strict; the main goal is to avoid regressions and clear negative signals.

## 6. Linkage to future Code-mode subtasks

This iteration intentionally stops at the **design and planning** layer. The following Code-mode subtasks should be created to implement it:

- **UX-CODE-7A – Refine ANM/FE HUD and overlay copy**
  - **Scope:** Implement Change 4.1A.
  - **Targets:** [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:141), [`UX_RULES_COPY_SPEC.md`](docs/UX_RULES_COPY_SPEC.md:1), [`GameHUD.tsx`](src/client/components/GameHUD.tsx:1), weird-state mapping utilities.
  - **Tests:** HUD weird-state regression tests (copy snapshots and telemetry labels).

- **UX-CODE-7B – Extend FE loop teaching flow**
  - **Scope:** Implement Change 4.1B.
  - **Targets:** [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:141), [`teachingScenarios.ts`](src/shared/teaching/teachingScenarios.ts:1), [`TeachingOverlay.tsx`](src/client/components/TeachingOverlay.tsx:1).
  - **Tests:** TeachingOverlay regression tests ensuring telemetry and routing for `fe_loop_intro`.

- **UX-CODE-7C – Improve structural stalemate explanations and routing**
  - **Scope:** Implement Changes 4.2A and 4.2B.
  - **Targets:** [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:185), [`UX_RULES_COPY_SPEC.md`](docs/UX_RULES_COPY_SPEC.md:1), [`VictoryModal.tsx`](src/client/components/VictoryModal.tsx:1), TeachingOverlay routing.
  - **Tests:** Game-end UX regression tests for stalemate copy and TeachingOverlay entrypoints.

- **UX-CODE-7D – Strengthen mini-region teaching and discoverability**
  - **Scope:** Implement Changes 4.3A and 4.3B.
  - **Targets:** [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:205), TeachingOverlay territory topics, sandbox scenario metadata.
  - **Tests:** Scenario metadata and telemetry tests confirming correct `rulesConcept` and routing from territory help to `mini_region_intro`.

The outcomes of these subtasks should be fed back into the next iteration document (e.g. `UX_RULES_IMPROVEMENT_ITERATION_0002.md`) along with updated telemetry comparisons from §5.
