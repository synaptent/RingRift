# Rules UX Improvement Iteration 0003 – Territory mini‑regions & capture chains

- **Iteration id:** 0003
- **Axis:** `rules-ux`
- **Record type:** Backfilled / dry‑run iteration record for work already landed in waves W1–W5 (Dec 2025) – metrics and windows below are **illustrative only**, not derived from live telemetry.
- **Board focus:** `square8` (primary), `square19` (secondary)
- **Player counts:** 2–3 players (territory mini‑regions most visible in 2–3p games)
- **Environments:** Local dev, CI, internal playtests, sandbox
- **Data sources:**
  - Synthetic `RulesUxAggregatesRoot`‑style aggregates as exercised in [`analyze_rules_ux_telemetry.test.ts`](tests/unit/analyze_rules_ux_telemetry.test.ts:27)
  - Qualitative audits and scenario tests (Q23 mini‑regions, capture‑chain patterns) from:
    - [`RULES_DOCS_UX_AUDIT.md`](docs/supplementary/RULES_DOCS_UX_AUDIT.md:47)
    - [`RULES_DYNAMIC_VERIFICATION.md`](RULES_DYNAMIC_VERIFICATION.md:435)
    - [`RulesMatrix.Territory.MiniRegion.test.ts`](tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts:19)
    - [`territoryDecisionHelpers.shared.test.ts`](tests/unit/territoryDecisionHelpers.shared.test.ts:76)
    - [`ScenarioConceptsUxRegression.test.ts`](tests/unit/ScenarioConceptsUxRegression.test.ts:1)

This iteration is documented **after** the underlying changes were implemented (W1–W5). It serves as a concrete example of how to record a rules‑UX iteration using the improvement loop defined in [`UX_RULES_IMPROVEMENT_LOOP.md`](docs/UX_RULES_IMPROVEMENT_LOOP.md:24), and as a template for future telemetry‑driven runs.

---

## 1. Focus and hypotheses

### 1.1 Targeted concepts

From the high‑risk concepts index in [`UX_RULES_CONCEPTS_INDEX.md`](docs/UX_RULES_CONCEPTS_INDEX.md:21):

- **`territory_mini_regions`**
  - Mini‑regions / Q23‑style disconnected regions with mandatory outside self‑elimination.
  - Canonical semantics: FAQ Q23 in [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1931) and territory rules [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:549) R141–R145.
  - Concordance entry in [`RULES_DOCS_UX_AUDIT.md` §4](docs/supplementary/RULES_DOCS_UX_AUDIT.md:142).

- **`capture_chains` / `capture_chain_mandatory`**
  - Capture chains with optional start but mandatory continuation while any capture exists, including 180° reversals and cyclic patterns.
  - Canonical captures and chain rules in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:480) R101–R103 and FAQ capture patterns in [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1626).
  - Concept row in [`UX_RULES_CONCEPTS_INDEX.md`](docs/UX_RULES_CONCEPTS_INDEX.md:26).

### 1.2 Hypotheses

- **H1 (Mini‑regions – Q23 clarity):**  
  If we provide a dedicated teaching flow for mini‑regions, align HUD/Victory copy with the self‑elimination prerequisite, and cross‑link rulebook and in‑app explanations via the concordance table, then:
  - help opens and rapid help reopens with `rulesContext = 'territory_mini_region'` should decrease over time, and
  - players should report fewer “why did I lose my own ring?” surprises in Q23‑style positions.

- **H2 (Capture chains – mandatory continuation):**  
  If all player‑facing copy and teaching flows for capture chains consistently emphasise:
  - optional **start**,
  - mandatory **continuation** while any capture exists, and
  - player **choice of path** (including deliberately ending the chain),  
    then:
  - we should see fewer confusion reports around “ending your turn” mid‑chain, and
  - help opens tagged to `rulesContext = 'capture_chain_mandatory'` should stabilise or fall relative to total games, especially after teaching flows are surfaced more prominently.

---

## 2. Starting conditions (example snapshot – synthetic)

Because Architect mode cannot run the telemetry stack, this section uses **illustrative** numbers derived from the hotspot analyzer test [`analyze_rules_ux_telemetry.test.ts`](tests/unit/analyze_rules_ux_telemetry.test.ts:27). They demonstrate the intended metrics shape, not real production data.

### 2.1 Synthetic telemetry slice

From the analyzer unit test:

```ts
const aggregates: RulesUxAggregatesRoot = {
  board: 'square8',
  num_players: 2,
  window: { start: '2025-11-01T00:00:00Z', end: '2025-11-30T23:59:59Z' },
  games: { started: 1200, completed: 800 },
  contexts: [
    makeContextAgg('anm_forced_elimination', {
      /* ... */
    }),
    makeContextAgg('structural_stalemate', {
      /* ... */
    }),
    makeContextAgg('territory_mini_region', {
      help_open: 4,
      help_reopen: 0,
      weird_state_banner_impression: 10,
      weird_state_details_open: 3,
      resign_after_weird_state: 0,
    }),
  ],
};
```

For `rulesContext = 'territory_mini_region'` this implies, **purely as an example**:

- `gamesCompleted = 800`
- `help_open = 4` → `helpOpensPer100Games ≈ 0.5`
- `help_reopen = 0` → `helpReopenRate = 0 / 4 = 0`
- `weird_state_banner_impression = 10`
- `resign_after_weird_state = 0` → `resignAfterWeirdRate = 0 / 10 = 0`

These values are **not** live telemetry; they simply mirror the kind of per‑context metrics the analyzer [`scripts/analyze_rules_ux_telemetry.ts`](scripts/analyze_rules_ux_telemetry.ts:33) computes:

- `helpOpensPer100Games`
- `helpReopenRate`
- `resignAfterWeirdRate`

For `capture_chain_mandatory` no synthetic dataset is wired into the analyzer test yet; the starting condition for that context comes from qualitative audits and scenario tests rather than counts.

### 2.2 Qualitative baselines

Key qualitative inputs for this iteration:

- **Mini‑regions (Q23) confusion:**
  - FAQ Q23 in [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1931) and its discussion of mandatory self‑elimination for disconnected regions.
  - Edge‑case and numeric‑invariant analysis in [`RULES_CONSISTENCY_EDGE_CASES.md`](RULES_CONSISTENCY_EDGE_CASES.md:361) and [`RULES_DYNAMIC_VERIFICATION.md`](RULES_DYNAMIC_VERIFICATION.md:435) (SCEN‑TERRITORY‑002 and related).
  - Territory helpers and scenario tests:
    - [`RulesMatrix.Territory.MiniRegion.test.ts`](tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts:19)
    - [`territoryDecisionHelpers.shared.test.ts`](tests/unit/territoryDecisionHelpers.shared.test.ts:76)

- **Capture chain confusion (mandatory vs optional):**
  - HUD and copy issues DOCUX‑P1/P3 in [`RULES_DOCS_UX_AUDIT.md`](docs/supplementary/RULES_DOCS_UX_AUDIT.md:33) – legacy text suggesting players can “end your turn” mid‑chain and that all lines offer the same reward choices.
  - Chain‑capture patterns and edge cases in [`RULES_CONSISTENCY_EDGE_CASES.md`](RULES_CONSISTENCY_EDGE_CASES.md:184) and capture pattern examples in [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1629).
  - Planned teaching flows for capture chains in [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:342).

---

## 3. Selected targets and contexts

### 3.1 `territory_mini_regions`

- **Concept id:** `territory_mini_regions` (index row in [`UX_RULES_CONCEPTS_INDEX.md`](docs/UX_RULES_CONCEPTS_INDEX.md:25)).
- **Canonical `rulesContext`:** `territory_mini_region` (see [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:98)).
- **Weird‑state linkage:** When territory decisions auto‑exhaust, `ANM_TERRITORY_NO_ACTIONS` from [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:100) is the canonical reason code.

This concept covers all Q23‑style “mini‑region + outside self‑elimination” situations where the acting player:

- collapses a small disconnected region into territory,
- eliminates all interior rings (all colours), and
- must also eliminate one of their own rings or caps **outside** the region, provided the self‑elimination prerequisite holds.

### 3.2 `capture_chains` / `capture_chain_mandatory`

- **Concept id:** `capture_chains` (index row in [`UX_RULES_CONCEPTS_INDEX.md`](docs/UX_RULES_CONCEPTS_INDEX.md:26)).
- **Canonical `rulesContext`:** `capture_chain_mandatory` (see [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:102)).

This concept captures mandatory‑continuation capture behaviour:

- Starting a capture is optional.
- Once any capture segment is taken, continuation is **mandatory** while any capture exists.
- When multiple segments are available, the player chooses the path and may deliberately end the chain early by choosing a branch that leads to no further captures.

---

## 4. Planned and shipped changes (W1–W5 backfill)

This section summarises what Iteration 0003 was **intended** to do and what has already been shipped in waves W1–W5. Future iterations (0004+) should treat this as the “baseline” state.

### 4.1 Territory mini‑regions (Q23) – `territory_mini_regions`

**Planned interventions:**

1. **Dedicated teaching flow for Q23 mini‑regions**
   - Flow `mini_region_intro` defined in [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:204) with guided and interactive steps:
     - Shape of a disconnected mini‑region and eligibility for processing.
     - How interior rings and border markers are handled.
     - How and why the outside self‑elimination cost is applied.

2. **Telemetry and rules‑context wiring**
   - `rulesContext = 'territory_mini_region'` defined in [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:98) for help and teaching events.
   - `ANM_TERRITORY_NO_ACTIONS` reason code in [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:100) used when territory decisions auto‑exhaust with no remaining actions.

3. **Concordance and copy alignment**
   - Concordance row for `territory_mini_region` in [`RULES_DOCS_UX_AUDIT.md` §4](docs/supplementary/RULES_DOCS_UX_AUDIT.md:142) mapping:
     - Canonical rules (FAQ Q23, RR‑CANON R141–R145).
     - In‑app copy (TeachingOverlay territory topic, sandbox territory tooltips).
     - Out‑of‑app docs and scenario matrix entries.

**Shipped outcomes (W2 + W5):**

- `mini_region_intro` flow now exists in shared teaching metadata (`teachingScenarios` in [`src/shared/teaching/scenarioTelemetry.ts`](src/shared/teaching/scenarioTelemetry.ts:1) and related teaching fixtures) and is referenced from TeachingOverlay and sandbox presets.
- Territory teaching copy in [`UX_RULES_COPY_SPEC.md` §7](docs/UX_RULES_COPY_SPEC.md:207) explicitly emphasises:
  - that disconnected regions eliminate interior rings, and
  - that an additional self‑elimination cost is often required from an outside stack.
- The concordance table row for `territory_mini_region` in [`RULES_DOCS_UX_AUDIT.md`](docs/supplementary/RULES_DOCS_UX_AUDIT.md:142) now provides a single “where is this explained” map for Q23 across rulebook, HUD, TeachingOverlay, sandbox scenarios, and docs.

### 4.2 Capture chains – `capture_chains` / `capture_chain_mandatory`

**Planned interventions:**

1. **Corrected HUD and TeachingOverlay copy**
   - Remove “end your turn” language for chain captures and ensure all surfaces stress:
     - Optional **start**;
     - Mandatory **continuation** while legal captures exist;
     - Player **choice of direction/path**.
   - Copy baselines in [`UX_RULES_COPY_SPEC.md` §5](docs/UX_RULES_COPY_SPEC.md:145) (capture and chain‑capture semantics) and chain‑capture teaching text referenced from [`TeachingOverlay.tsx`](src/client/components/TeachingOverlay.tsx:60).

2. **Scenario‑driven teaching for capture chains**
   - Flow `capture_chain_mandatory` in [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:342) describing:
     - Optional vs mandatory portions of the chain.
     - Multiple‑direction choices.
     - Ending chains early by deliberate path choice.

3. **Telemetry linkage**
   - `rulesContext = 'capture_chain_mandatory'` in [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:102) so that help opens and teaching flow events for chain‑capture scenarios can be isolated.

**Shipped outcomes (W1 + W2 + W5):**

- HUD and sandbox phase copy for capture and chain capture now follow the canonical wording in [`UX_RULES_COPY_SPEC.md` §5](docs/UX_RULES_COPY_SPEC.md:145), removing the mis‑leading “end your turn” phrasing flagged as DOCUX‑P1 in [`RULES_DOCS_UX_AUDIT.md`](docs/supplementary/RULES_DOCS_UX_AUDIT.md:33).
- TeachingOverlay exposes dedicated chain‑capture explanations and tips, with `capture_chain_mandatory` as the underlying rules concept; capture flows are exercised in `ScenarioConceptsUxRegression` tests in [`tests/unit/ScenarioConceptsUxRegression.test.ts`](tests/unit/ScenarioConceptsUxRegression.test.ts:1).
- Capture‑chain scenarios in curated sandbox configuration (`curated.json` under `learn.capture.chain`) are aligned with the semantics in [`UX_RULES_COPY_SPEC.md`](docs/UX_RULES_COPY_SPEC.md:202).

---

## 5. Validation plan (for future telemetry‑backed runs)

Although this backfilled record does **not** include real telemetry, future operators can treat it as a runbook for validating the impact of W1–W5 and any subsequent tweaks.

### 5.1 Quantitative checks

Using the rules‑UX hotspot analyzer [`scripts/analyze_rules_ux_telemetry.ts`](scripts/analyze_rules_ux_telemetry.ts:33) over a well‑defined window (e.g. 28 days, Square‑8 2p):

1. **Territory mini‑regions (`rulesContext = 'territory_mini_region'`):**
   - Compute:
     - `helpOpensPer100Games`
     - `helpReopenRate`
     - `resignAfterWeirdRate`
   - Compare pre‑ vs post‑W1–W5 (or pre‑ vs post‑subsequent iterations):
     - Expect `helpOpensPer100Games` and `helpReopenRate` to decrease or remain stable while overall comprehension improves (as seen in qualitative feedback).
     - Ensure `resignAfterWeirdRate` remains low; spikes may indicate persistent confusion about self‑elimination or scoring.

2. **Capture chains (`rulesContext = 'capture_chain_mandatory'`):**
   - Once events for chain‑capture teaching flows and help opens are wired:
     - Track `helpOpensPer100Games` and `helpReopenRate`.
     - Treat sustained high reopen rates as a signal that copy or visualisation may still be unclear.

### 5.2 Qualitative checks

For both concepts:

- Use sandbox and curated scenarios (Q23 mini‑regions, multi‑segment capture‑chain examples) to run guided internal play sessions.
- For each participant, capture short answers to:
  - “Can you explain why you had to eliminate an outside ring in this mini‑region scenario?”
  - “What determines when a capture chain must continue vs when it can stop?”
- Treat persistent misunderstandings as candidates for future iteration targets, even if telemetry does not yet show high severity.

---

## 6. Outcome summary (backfilled for W1–W5)

From the perspective of the improvement loop in [`UX_RULES_IMPROVEMENT_LOOP.md`](docs/UX_RULES_IMPROVEMENT_LOOP.md:75), Iteration 0003 documents a **completed** round of work whose implementation was spread across W1–W5:

- **Territory mini‑regions (Q23):**
  - Rules semantics were clarified and centralised in rulebook, canonical spec, and edge‑case docs.
  - A dedicated `mini_region_intro` teaching flow and curated scenarios now exist for Q23‑style positions.
  - Player‑facing copy (HUD, TeachingOverlay, sandbox) is aligned with the self‑elimination prerequisite, and the concordance table row in [`RULES_DOCS_UX_AUDIT.md`](docs/supplementary/RULES_DOCS_UX_AUDIT.md:142) ensures future copy work stays in sync.

- **Capture chains:**
  - Legacy HUD copy suggesting you can “end your turn” during capture chains has been replaced with canonical wording emphasising mandatory continuation and choice of path.
  - Teaching flows and curated scenarios now explicitly demonstrate capture chains, including deliberate chain‑ending choices.
  - Concept ids and telemetry `rulesContext` values (`capture_chain_mandatory`) are in place so that future telemetry‑driven iterations can measure confusion and improvement.

Because live rules‑UX telemetry is not yet available, this record treats all metrics as **examples or placeholders**. Its primary purpose is to:

- capture how W1–W5 already addressed territory mini‑regions and capture chains,
- define the metrics and scenarios future operators should use, and
- provide a concrete template for iteration records 0004, 0005, and beyond once real data is flowing.
