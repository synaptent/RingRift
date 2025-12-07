# UX Rules Improvement Loop

> **Doc Status (2025-12-05): Partially implemented – W‑UX‑4 process is defined; core telemetry, weird-state mapping, and teaching scenario specs are now wired.**
>
> **Role:** Define a lightweight, repeatable process that uses rules‑UX telemetry and targeted documentation/UX specs to continuously improve the weakest aspect of RingRift: **player‑facing rules UX and onboarding**, especially around weird states (ANM, forced elimination, structural stalemate) and complex mechanics (mini‑regions, chains, lines vs territory).
>
> **Inputs:**
>
> - Telemetry schema in [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:1).
> - Weird‑state mapping in [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:1).
> - Teaching flows in [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:1).
> - Rules copy baseline in [`UX_RULES_COPY_SPEC.md`](docs/UX_RULES_COPY_SPEC.md:1).
> - Prior audits: [`docs/supplementary/RULES_DOCS_UX_AUDIT.md`](docs/supplementary/RULES_DOCS_UX_AUDIT.md:23), [`archive/FINAL_RULES_AUDIT_REPORT.md`](archive/FINAL_RULES_AUDIT_REPORT.md:90), [`archive/RULES_DYNAMIC_VERIFICATION.md`](archive/RULES_DYNAMIC_VERIFICATION.md:665).

This document does **not** prescribe any runtime behaviour. It describes:

- The **cadence and inputs** for rules‑UX iterations.
- A minimal **iteration workflow & summary template**.
- **Roles and responsibilities**.
- A simple repository structure for iteration notes.

---

## 1. Cadence and Inputs

### 1.1 Cadence

Default cadence:

- **Every 4 weeks** in active development, or
- **Once per release** if releases are less frequent.

Strong recommendation:

- Run at least one rules‑UX iteration for every **major UX change** (e.g., major HUD / VictoryModal rewrite, new TeachingOverlay release).
- Run a **lightweight post‑mortem** after any externally‑reported incident clearly tied to rules confusion (e.g. support tickets, forum threads about “buggy” forced elimination or stalemates).

### 1.2 Core Inputs

Each iteration uses four classes of inputs:

1. **Quantitative telemetry** (W‑UX‑1)

   From [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:1) and the implemented metrics:
   - `ringrift_rules_ux_events_total{type, board_type, num_players, ai_difficulty, topic, rules_concept, weird_state_type}`.
   - `ringrift_games_started_total{board_type, num_players, difficulty, is_ranked, is_calibration_game}`.
   - `ringrift_games_completed_total{board_type, num_players, difficulty, is_ranked, terminal_reason}`.
   - Derived aggregates such as:
     - Help and weird‑state help events per `rules_concept` / `weird_state_type`.
     - Help sessions and reopen rates per rules context (via downstream jobs keyed by `rules_context`).
     - `rules_weird_state_resign` / `resign_after_weird_state` rates by `weird_state_type` and reason code (see §6.3 in [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:262)).

2. **Qualitative reports and audits**
   - Existing findings in [`docs/supplementary/RULES_DOCS_UX_AUDIT.md`](docs/supplementary/RULES_DOCS_UX_AUDIT.md:23) (DOCUX‑P1…P7 etc.).
   - New bug reports, support tickets, and community feedback tagged as:
     - “Rules confusion”.
     - “Weird end of game”.
     - “Forced elimination / no moves seems wrong”.
     - “Stalemate / plateau confusion”.
   - Internal developer notes from rules changes, AI calibration, or sandbox refactors.

3. **Engine / rules correctness signals**
   - Rules invariants and parity signals from:
     - [`docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md`](docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md:39).
     - [`docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md`](docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md:361).
     - [`archive/RULES_DYNAMIC_VERIFICATION.md`](archive/RULES_DYNAMIC_VERIFICATION.md:665).
   - These do **not** directly drive UX changes, but ensure that UX proposals do not contradict canonical behaviour.

4. **Rules concepts index**
   - Consolidated high‑risk concepts in [`UX_RULES_CONCEPTS_INDEX.md`](docs/UX_RULES_CONCEPTS_INDEX.md:1), aligning rules semantics, UX surfaces, teaching flows, and telemetry/tests.
   - When selecting targets for each iteration, consult this index to focus work on the most fragile rules concepts and to reuse existing mappings instead of rediscovering them ad‑hoc.
   - Use the rules‑UX concordance table in [`RULES_DOCS_UX_AUDIT.md` §4](docs/supplementary/RULES_DOCS_UX_AUDIT.md:124) as the primary map from concept ids to rules docs, in‑app copy, and player‑facing docs when deciding where to change or validate wording.

---

## 2. Iteration Workflow

Each iteration follows the same minimal workflow:

1. **Prepare & snapshot**
2. **Extract hotspots**
3. **Propose UX adjustments**
4. **Plan & implement (separate Code‑mode tasks)**
5. **Validate & document**

### 2.1 Step 1 – Prepare & Snapshot

**Goal:** Capture a reproducible baseline for this iteration.

Activities:

- Freeze a **time window**:
  - Usually the last **28 days** or “since last UX iteration”.
- Pull high‑level metrics (segmented by `board_type`, `num_players`, `difficulty` where useful):
  - Total games started and completed.
  - Distribution of terminal reasons (`elimination`, `territory`, `structural_stalemate`, `last_player_standing`, `resign`, `timeout`).
  - Aggregate counts of:
    - `help_open`, `help_topic_view`, `help_reopen`.
    - `weird_state_banner_impression`.
    - `weird_state_details_open`.
    - `resign_after_weird_state`.
    - `sandbox_scenario_loaded` / `sandbox_scenario_completed`.
- Save the snapshot (charts or raw numbers) into the iteration notes (see §3).

### 2.2 Step 2 – Extract Hotspots

**Goal:** Identify which rules contexts and surfaces are currently most confusing or fragile.

Using PromQL/pipeline queries derived from [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:262), or by running the hotspot analyzer CLI [`scripts/analyze_rules_ux_telemetry.ts`](scripts/analyze_rules_ux_telemetry.ts:1) against a pre‑aggregated `RulesUxAggregatesRoot` JSON snapshot (see [`analyze_rules_ux_telemetry.test.ts`](tests/unit/analyze_rules_ux_telemetry.test.ts:27) for the expected heuristics and severity classification):

1. **Top help‑opens per 100 games by `rules_context`**
   - Focus on contexts that exceed a minimum volume threshold (e.g. > 100 `help_open` events in the period).
   - Rank contexts by `help_open` per 100 games:
     - `anm_forced_elimination`
     - `structural_stalemate`
     - `territory_mini_region`
     - `capture_chain_mandatory`
     - etc.

2. **High “help reopen within 30 seconds” contexts**
   - For each `rules_context`, compute the fraction of help sessions that triggered `help_reopen` within 30 seconds (see derived metric notes in [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:229)).
   - Treat high reopen ratios as a strong signal of **unsatisfying explanations**.

3. **Resigns after weird states**
   - For each `rules_context`, compute the ratio:
     - `resign_after_weird_state` / `weird_state_banner_impression`.
   - High ratios at low sample sizes may be noisy; focus on contexts with both:
     - Enough impressions (e.g. at least 50).
     - Substantial resign‑per‑impression rates.

4. **Correlate with qualitative feedback**
   - Cross‑check contexts identified above with:
     - Open items in [`RULES_DOCS_UX_AUDIT.md`](docs/supplementary/RULES_DOCS_UX_AUDIT.md:69).
     - Recent support tickets and user reports.
   - Highlight any **new** contexts not covered in prior audits.

### 2.3 Step 3 – Propose UX Adjustments

**Goal:** Design a **small, concrete set of UX changes** to address the top hotspots, referring only to existing specs.

Categories of adjustments:

1. **Copy tweaks (text‑only)**
   - HUD / VictoryModal / TeachingOverlay copy, consistent with:
     - [`UX_RULES_COPY_SPEC.md`](docs/UX_RULES_COPY_SPEC.md:1).
     - Weird‑state mapping in [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:1).
   - Examples:
     - Clarify that chain captures are **mandatory** once started (DOCUX‑P1).
     - Replace “eliminate all opponent rings” with “more than half of all rings” for elimination victory (DOCUX‑P2).
     - Add explicit self‑elimination explanation to territory HUD/tooltip (DOCUX‑P4).

2. **Weird‑state UX refinement**
   - Map emergent patterns to reason codes defined in [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:121).
   - Example proposals:
     - Add or refine the `RWS‑001 ANM_MOVEMENT_FE_BLOCKED` banner when FE is triggered.
     - Improve copy for structural stalemate explanation (`RWS‑005`).
     - Introduce an explicit LPS explanation (`RWS‑006`) once engines support early LPS.

3. **Teaching scenarios & flows**
   - Add or refine flows in [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:1).
   - Example:
     - If `anm_forced_elimination` consistently tops help‑opens and resigns, prioritise implementing `fe_loop_intro` and ensure it is rotated prominently in TeachingOverlay and sandbox presets.
     - If `territory_mini_region` is a hotspot, polish and surface `mini_region_intro` more aggressively.

4. **Surface routing & discoverability**
   - Ensure that:
     - Weird‑state banners link to the correct TeachingOverlay topics (per [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:438)).
     - Teaching flows are discoverable from the relevant surfaces (e.g. offering `fe_loop_intro` after an FE‑related resignation).

Each iteration should select a **small batch** (e.g. 2–5 changes) with clear linkage to the metrics and specs above. Implementation details are deferred to Code‑mode tasks.

### 2.4 Step 4 – Plan & Implement (Code‑Mode Work)

**Goal:** Translate proposals into concrete implementation tasks.

- For each proposed change, create one or more Code‑mode tasks referencing:
  - The relevant spec(s): `UX_RULES_COPY_SPEC`, `UX_RULES_WEIRD_STATES_SPEC`, `UX_RULES_TEACHING_SCENARIOS`, `UX_RULES_TELEMETRY_SPEC`.
  - The rules docs (e.g. `RULES_CANONICAL_SPEC`, `ringrift_complete_rules`) if semantics are potentially impacted.
- Example task categories:
  - “Update chain capture HUD copy to remove ‘end your turn’ and reference mandatory continuation.”
  - “Wire reason code `STRUCTURAL_STALEMATE_TIEBREAK` to structural stalemate result and banner.”
  - “Implement teaching flow `fe_loop_intro` with three scenarios and telemetry instrumentation.”

Implementation and testing themselves are out of scope for this document but MUST be tracked in the iteration summary (see §3).

### 2.5 Step 5 – Validate & Document

**Goal:** After changes ship, run a **follow‑up pass** to assess whether they helped.

- Wait for a stabilisation period (e.g. 2–4 weeks of live traffic post‑release).
- Recalculate the same metrics as in Step 1/2, focusing on:
  - Hotspot `rules_context` previously targeted.
  - Help‑open and help‑reopen rates.
  - `resign_after_weird_state` ratios.
  - Completion rates of relevant teaching flows (e.g. `sandbox_scenario_completed` and `teaching_step_completed` for the flows touched).
- Perform a **directional comparison**:
  - Did help opens or reopens per 100 games **decrease** for the targeted context?
  - Did resigns after weird states **decrease**, especially for the targeted reason codes?
  - Did teaching flows for targeted concepts see **higher completion rates**?

Scientific rigour (A/B tests, statistical significance) is ideal but **not required**. The primary goal is to detect clear regressions and obvious improvements.

Document key observations in the iteration summary and mark open threads for the next iteration.

---

## 3. Iteration Summary Template

All iteration notes should live under:

- `docs/ux/rules_iterations/ITERATION_YYYYMMDD_short_name.md` for dated, ad-hoc iteration summaries; and
- `docs/ux/rules_iterations/UX_RULES_IMPROVEMENT_ITERATION_####.md` for numbered, process-driven iterations (e.g. W-UX-7).

For example:

- `docs/ux/rules_iterations/ITERATION_20251205_fe_and_stalemate.md`
- `docs/ux/rules_iterations/UX_RULES_IMPROVEMENT_ITERATION_0001.md` (first full telemetry-driven iteration focusing on ANM/FE, structural stalemate, and mini-regions)
- Iteration 0002 (hotspot-driven, GameEndExplanation-based) is specified in [`UX_RULES_IMPROVEMENT_ITERATION_0002.md`](docs/ux/rules_iterations/UX_RULES_IMPROVEMENT_ITERATION_0002.md:1).

### 3.1 Suggested File Structure

Each iteration summary SHOULD follow this template:

```markdown
# Rules UX Iteration – <Short Name>

- **Iteration id:** ITERATION_YYYYMMDD_short_name
- **Dates:** <start date> → <end date>
- **Release(s):** <tags / versions / branches>
- **Participants:**
  - Rules UX owner: <role or handle>
  - Telemetry owner: <role or handle>
  - Implementation owner(s): <roles/teams>

## 1. Focus and hypotheses

- Targeted rules contexts (rules_context):
  - `anm_forced_elimination`
  - `structural_stalemate`
  - `territory_mini_region`
- Targeted surfaces:
  - HUD weird-state banners
  - VictoryModal explanations
  - TeachingOverlay flows (`fe_loop_intro`, `structural_stalemate_intro`)
- Brief hypotheses:
  - H1: Clarifying FE banners will reduce rapid help reopen and resign rates after FE.
  - H2: Adding a structural stalemate teaching flow will reduce “bug” reports on plateau endings.

## 2. Baseline snapshot

### 2.1 Telemetry (pre-change window)

- Time window: <start→end> (e.g. 2025-11-01 → 2025-11-28)
- Key metrics (averaged or summed over window):
  - Help opens per 100 games by `rules_context` (top 5).
  - Help reopen ratio within 30s by `rules_context` (top 5).
  - Resigns after weird states per impression by `rules_context`.
  - Teaching flows usage/completion (if applicable).

### 2.2 Qualitative inputs

- Summary of newly observed player feedback.
- Relevant items from RULES_DOCS_UX_AUDIT, bug tracker, community channels.

## 3. Changes made

### 3.1 Copy and weird-state UX

- Change 1: <short description>
  - Surfaces: <HUD / VictoryModal / TeachingOverlay>
  - Specs: <links into UX_RULES_COPY_SPEC, UX_RULES_WEIRD_STATES_SPEC>
  - Code tasks: <issue ids / PR links>

### 3.2 Teaching scenarios

- Change 2: Implemented flow `fe_loop_intro` steps 1–3.
  - Specs: [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:97)
  - Linked weird-state reasons: `ANM_MOVEMENT_FE_BLOCKED`, `FE_SEQUENCE_CURRENT_PLAYER`.

### 3.3 Instrumentation

- Confirmed telemetry events are present for:
  - `weird_state_banner_impression` (FE and stalemate).
  - `resign_after_weird_state`.
  - `sandbox_scenario_loaded` / `sandbox_scenario_completed` for new flows.

## 4. Follow-up metrics (post-change window)

- Time window: <start→end> (e.g. 2025-12-01 → 2025-12-28)
- Same metrics as §2.1, focusing on targeted contexts.

## 5. Observations and conclusions

- Did the metrics move in the expected direction?
- Any unexpected regressions?
- Qualitative player feedback after the change.

## 6. Next iteration candidates

- Proposed contexts or surfaces to focus on in the next iteration.
- Any deeper structural work (e.g., LPS implementation, new teaching flows) that should be queued.
```

The summary SHOULD remain short (2–3 pages) and action‑oriented.

---

## 4. Roles and Responsibilities

This section describes roles by **responsibility**, not by person.

### 4.1 Rules UX Owner

- Typically a product / design / documentation role with good understanding of RR‑CANON and player sentiment.
- Responsibilities:
  - Own this improvement loop doc and the iteration schedule.
  - Decide iteration scope (which `rules_context` to prioritise).
  - Draft UX proposals referencing:
    - [`UX_RULES_COPY_SPEC.md`](docs/UX_RULES_COPY_SPEC.md:1),
    - [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:1),
    - [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:1).
  - Co‑author iteration summaries under `docs/ux/rules_iterations/`.

### 4.2 Telemetry & Data Owner

- Typically an engineer / analyst with access to metrics infrastructure.
- Responsibilities:
  - Ensure [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:1) is implemented and up to date.
  - Produce the quantitative snapshots (Steps 1–2) each iteration.
  - Maintain the Prometheus / warehouse queries used as standard hotspots.
  - Flag metric schema changes that would break historical comparisons.

### 4.3 Implementation Owner(s)

- Frontend and backend engineers, possibly with AI/engine support.
- Responsibilities:
  - Implement copy changes, reason‑code mappings, and teaching flows as per specs.
  - Add or update tests to keep rules semantics aligned with `RULES_CANONICAL_SPEC`.
  - Coordinate release timing and feature flags for UX experiments.

### 4.4 QA / Validation

- Responsibilities:
  - Verify that UX changes align with canonical rules and engine behaviour.
  - Test weird‑state scenarios (FE, structural stalemate, mini‑regions, etc.) across:
    - Human vs AI,
    - Sandbox,
    - Different board types and player counts.
  - Confirm telemetry events fire as expected in representative flows.

### 4.5 Documentation & Support

- Responsibilities:
  - Keep public documentation (`ringrift_complete_rules`, simplified rules, site docs) aligned with any UX changes.
  - Look for patterns in support tickets that tie back to rules confusion, and feed them into §1.2 inputs for the next iteration.

---

## 5. Storage and Organisation of Iteration Notes

- All iteration notes MUST reside under:
  - `docs/ux/rules_iterations/`

- Iteration file names MAY follow either:
  - `ITERATION_YYYYMMDD_short_name.md` for organic, date-keyed notes (see §3.1 example), or
  - `UX_RULES_IMPROVEMENT_ITERATION_####.md` for numbered, explicitly tracked iterations (e.g. [`UX_RULES_IMPROVEMENT_ITERATION_0001.md`](docs/ux/rules_iterations/UX_RULES_IMPROVEMENT_ITERATION_0001.md:1)) introduced by W-UX-7.

- `DOCUMENTATION_INDEX.md` SHOULD include an index entry pointing to this directory and the latest iteration file (see W‑UX‑2/3/4 cross‑reference tasks).

- `RULES_DOCS_UX_AUDIT.md` SHOULD cross‑link to this improvement loop to show how audit findings are fed into ongoing work.

---

## 6. Relationship to W‑UX‑1‒4

- **W‑UX‑1 (Telemetry instrumentation & hotspot identification)**
  - Implemented via the events and metrics defined in [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:1).
  - Step 1–2 of this loop rely directly on those metrics.

- **W‑UX‑2 (Context‑aware weird‑state UX for ANM/FE and structural stalemates)**
  - Implemented via reason codes and copy in [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:1).
  - Iteration proposals in Step 3 often target which reason codes to surface or refine next.

- **W‑UX‑3 (Scenario‑driven teaching flows)**
  - Implemented via `rulesConcept` flows defined in [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:1).
  - Step 3 may propose new flows or changes to existing ones; Step 5 checks their impact.

- **W‑UX‑4 (This process)**
  - Provides the **glue**: a small, documented loop that ties telemetry, weird‑state UX, and teaching into an evolving rules‑UX system, instead of one‑off fixes.

When future work changes any of the three specs above, this improvement loop SHOULD be updated to reference new contexts, metrics, or flows as necessary.
