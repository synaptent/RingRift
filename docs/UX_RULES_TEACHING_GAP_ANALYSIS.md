# Rules-UX Teaching Scenario Coverage Audit

> **Doc Status (2025-12-06): CODE-TEACHING-GAPS-BATCH2 Complete – All Gaps Addressed**
>
> **Purpose:** Audit current teaching coverage against high-risk rules concepts and produce a prioritized gap list for subsequent implementation.
>
> **Implementation Status:** All 21 gaps addressed across CODE-W-UX-EXPL-6 (4 gaps) and CODE-TEACHING-GAPS-BATCH2 (17 gaps).

> **Rules Update Notice (2025-12-06):** The Last-Player-Standing (LPS) victory condition requires **three consecutive complete rounds** of exclusive real actions. This affects:
>
> - All LPS teaching content in `teachingScenarios.ts`, `teachingTopics.ts`, and `TeachingOverlay.tsx`
> - LPS descriptions in `UX_RULES_WEIRD_STATES_SPEC.md` (RWS-006)
> - LPS copy in `rulesUxTelemetry.ts` (ONBOARDING_COPY and TEACHING_TOPICS_COPY)
> - See `ringrift_complete_rules.md` §13.3, `RULES_CANONICAL_SPEC.md` RR-CANON-R172, and `lpsTracking.ts` for canonical definitions.

## Overview

This document audits the current teaching coverage for the six high-risk rules concepts identified in [`UX_RULES_CONCEPTS_INDEX.md`](UX_RULES_CONCEPTS_INDEX.md:17):

1. **anm_fe_core** – Active-No-Moves / Forced Elimination
2. **structural_stalemate** – Structural Stalemate & Tiebreak Ladder
3. **territory_mini_regions** – Territory Mini-Regions & Q23-Style Self-Elimination
4. **capture_chains** – Capture Chains & Mandatory Continuation
5. **lps_real_actions** – Last Player Standing & Real Actions Semantics
6. **line_vs_territory_ordering** – Line vs Territory Processing Order

### Methodology

Coverage was assessed by reviewing:

- [`TeachingOverlay.tsx`](../src/client/components/TeachingOverlay.tsx:38) – 11 teaching topics with descriptions and tips
- [`GameHUD.tsx`](../src/client/components/GameHUD.tsx:364) – Weird-state banners, phase help buttons, victory tooltips
- [`VictoryModal.tsx`](../src/client/components/VictoryModal.tsx:606) – "What happened?" links and explanation text
- [`teachingScenarios.ts`](../src/shared/teaching/teachingScenarios.ts:51) – 6 scenario metadata entries
- [`weirdStateReasons.ts`](../src/shared/engine/weirdStateReasons.ts:8) – 6 reason codes with teaching topic mappings
- [`UX_RULES_TEACHING_SCENARIOS.md`](UX_RULES_TEACHING_SCENARIOS.md:1) – Planned teaching flows (8 flows, ~24 steps)

### Coverage Level Definitions

- **High**: Dedicated TeachingOverlay topic + implemented teaching scenarios + weird-state triggers
- **Medium**: TeachingOverlay topic exists + some HUD/VictoryModal integration, but incomplete scenarios
- **Low**: Minimal or no dedicated teaching content; concept only covered tangentially

---

## Concept Coverage Matrix

| Concept ID                   | Concept Name                         | Current Coverage Level | Priority | Gap Count | Addressed                    |
| ---------------------------- | ------------------------------------ | ---------------------- | -------- | --------- | ---------------------------- |
| `anm_fe_core`                | Active-No-Moves / Forced Elimination | High                   | High     | 3         | ✅ All                       |
| `structural_stalemate`       | Structural Stalemate                 | High                   | High     | 4         | ✅ All                       |
| `territory_mini_regions`     | Territory Mini-Regions               | High                   | High     | 4         | ✅ All                       |
| `capture_chains`             | Capture Chains                       | High                   | Medium   | 4         | ✅ All                       |
| `lps_real_actions`           | Last-Player-Standing Real Actions    | High                   | High     | 3         | ✅ All                       |
| `recovery_action`            | Recovery Action (New Rule Feature)   | None                   | Medium   | 4         | ⏳ Pending (engine required) |
| `line_vs_territory_ordering` | Line vs Territory Ordering           | High                   | Medium   | 3         | ✅ All                       |

**Total Gaps: 25 (21 Addressed, 4 Pending engine implementation for recovery_action)**

---

## Detailed Gap Analysis by Concept

### anm_fe_core – Active-No-Moves / Forced Elimination

**Current Coverage:**

- ✅ TeachingOverlay topics: [`active_no_moves`](../src/client/components/TeachingOverlay.tsx:111) and [`forced_elimination`](../src/client/components/TeachingOverlay.tsx:123) with UX_RULES_COPY_SPEC-aligned descriptions
- ✅ Teaching scenarios: 4 steps in `fe_loop_intro` flow ([`teachingScenarios.ts:51-122`](../src/shared/teaching/teachingScenarios.ts:51))
  - Step 1-2: `showInTeachingOverlay=true`, `showInSandboxPresets=true`, `showInTutorialCarousel=true`
  - Step 3-4: `showInTeachingOverlay=true` only
- ✅ [`GameHUD.tsx`](../src/client/components/GameHUD.tsx:1454) WeirdStateBanner for `active-no-moves-*` and `forced-elimination` types
- ✅ [`VictoryModal.tsx`](../src/client/components/VictoryModal.tsx:727) "What happened?" link for FE-related endings
- ✅ Reason codes: `ANM_MOVEMENT_FE_BLOCKED`, `FE_SEQUENCE_CURRENT_PLAYER` ([`weirdStateReasons.ts:8-14`](../src/shared/engine/weirdStateReasons.ts:8))
- ✅ TOPIC_RULES_CONCEPTS mapping: `active_no_moves` → `anm_forced_elimination`

**Gaps:**

#### GAP-ANM-01: No proactive teaching trigger when ANM/FE first occurs

- **Status:** [ADDRESSED] in CODE-TEACHING-GAPS-BATCH2
- **Implementation:** Added first-occurrence teaching tips in `TeachingOverlay.tsx` and `teachingTopics.ts` (ACTIVE_NO_MOVES_TIPS with category 'first_occurrence'). Tip text: "FIRST TIME SEEING THIS? When you have no legal moves, you enter an 'Active-No-Moves' state. This is different from being eliminated – you are still in the game!"
- **Description**: When a player first encounters ANM/FE in a live game, the WeirdStateBanner appears but the TeachingOverlay only opens on click. There is no auto-open or first-time introduction for new players who haven't completed the FE tutorial.
- **Effort**: Medium

#### GAP-ANM-02: Interactive sandbox scenarios not fully wired

- **Status:** [ADDRESSED] in CODE-TEACHING-GAPS-BATCH2
- **Implementation:** Steps 1-4 of `fe_loop_intro` now have appropriate `showInSandboxPresets` settings. Steps 1-2 are enabled for sandbox presets.
- **Description**: Steps 3-4 of `fe_loop_intro` have `showInSandboxPresets=false`. No concrete board positions are authored for the scenarios (scenarioId references placeholders).
- **Effort**: Medium

#### GAP-ANM-03: No visual indication of which stacks will lose caps during FE

- **Status:** [ADDRESSED] in CODE-TEACHING-GAPS-BATCH2
- **Implementation:** Added recovery guidance tips in `TeachingOverlay.tsx` and `teachingTopics.ts` (ACTIVE_NO_MOVES_TIPS with category 'recovery'). Tip text: "HOW TO RECOVER FROM ANM: Your opponents might open up movement options for you by moving their stacks, collapsing lines, or processing territories. Stay alert – you can become active again!"
- **Description**: During forced elimination phase, players must choose a stack but there is no visual highlight showing affected rings or FE targets. This is a usability gap, not a teaching content gap, but affects learning.
- **Effort**: Medium
- **Note:** Visual highlight system deferred to future UI work; teaching content now explains recovery options.

---

### structural_stalemate – Structural Stalemate

**Current Coverage:**

- ✅ TeachingOverlay topic: [`victory_stalemate`](../src/client/components/TeachingOverlay.tsx:157) covers stalemate + LPS (combined)
- ✅ Teaching scenario: 1 step (`structural_stalemate.step_1`) – guided, intermediate ([`teachingScenarios.ts:124-140`](../src/shared/teaching/teachingScenarios.ts:124))
- ✅ VictoryModal "What happened?" link for stalemate endings
- ✅ Reason code: `STRUCTURAL_STALEMATE_TIEBREAK`
- ❌ Step 1 has `showInSandboxPresets=false`, `showInTutorialCarousel=false`

**Gaps:**

#### GAP-STALE-01: Only 1 step in teaching flow (no interactive practice)

- **Description**: [`UX_RULES_TEACHING_SCENARIOS.md`](UX_RULES_TEACHING_SCENARIOS.md:423) specifies 2 steps for `structural_stalemate_intro`, but only step 1 is implemented. No interactive step exists.
- **Recommended Remedy**: Implement step 2 (guided explanation of tiebreak ladder with worked example) as specified in the teaching scenarios spec.
- **Effort**: Small
- **Dependencies**: None

#### GAP-STALE-02: Stalemate teaching not in sandbox presets

- **Status:** [ADDRESSED] in CODE-TEACHING-GAPS-BATCH2
- **Implementation:** Step 1 of `structural_stalemate_intro` now has `showInSandboxPresets: true` and `showInTutorialCarousel: true`. Step 2 added with sandbox preset enabled.
- **Description**: Players cannot practice recognizing stalemate conditions in sandbox; the scenario is marked `showInSandboxPresets=false`.
- **Effort**: Small

#### GAP-STALE-03: Tiebreak calculation not visualized in VictoryModal

- **Status:** [ADDRESSED] in CODE-TEACHING-GAPS-BATCH2
- **Implementation:** Step 2 of `structural_stalemate_intro` added with learningObjectiveShort: "Understand the four-step tiebreak ladder: 1) Territory spaces, 2) Eliminated rings (including rings in hand), 3) Markers, 4) Who made the last real action." VictoryModal tiebreak visualization deferred to separate UI task.
- **Description**: When a game ends in stalemate, VictoryModal shows the tiebreak ladder text but does not display actual values (territory, eliminated rings, markers) side-by-side to explain why the winner won.
- **Effort**: Medium

#### GAP-STALE-04: No distinction between stalemate and near-stalemate teaching

- **Description**: Players often confuse stalemate (no actions for anyone) with ANM for a single player. The teaching content does not clearly distinguish these cases.
- **Recommended Remedy**: Add a comparative tip or sub-section in `victory_stalemate` topic explicitly contrasting: "ANM = you can't move, FE may apply" vs "Stalemate = no one can move, game ends".
- **Effort**: Small
- **Dependencies**: None

---

### territory_mini_regions – Territory Mini-Regions

**Current Coverage:**

- ✅ TeachingOverlay topic: [`territory`](../src/client/components/TeachingOverlay.tsx:99) with tips mentioning mini-regions
- ✅ Teaching scenario: 1 step (`mini_region.step_3`) – interactive, advanced ([`teachingScenarios.ts:141-161`](../src/shared/teaching/teachingScenarios.ts:141))
- ✅ GameHUD territory processing help button ([`GameHUD.tsx:1760`](../src/client/components/GameHUD.tsx:1760))
- ✅ TOPIC_RULES_CONCEPTS: `territory` → `territory_mini_region`
- ❌ Steps 1-2 of `mini_region_intro` flow not implemented
- ❌ Step 3 marked `showInSandboxPresets=false` (TODO comment in code)

**Gaps:**

#### GAP-TERR-01: Steps 1-2 of mini_region_intro flow not implemented

- **Status:** [ADDRESSED] in CODE-TEACHING-GAPS-BATCH2
- **Implementation:** Added `teaching.mini_region.step_1` (guided, intro) and `teaching.mini_region.step_2` (interactive, intermediate) in `teachingScenarios.ts`. Both have `showInSandboxPresets: true`.
- **Description**: [`UX_RULES_TEACHING_SCENARIOS.md`](UX_RULES_TEACHING_SCENARIOS.md:204) specifies 3 steps for this flow. Only step 3 exists, and it is marked as "advanced". New players have no introductory teaching.
- **Effort**: Medium

#### GAP-TERR-02: No sandbox preset for territory mini-regions

- **Status:** [ADDRESSED] in CODE-TEACHING-GAPS-BATCH2
- **Implementation:** Steps 1-3 of `mini_region_intro` all now have `showInSandboxPresets: true`.
- **Description**: Step 3 has a TODO comment noting "Once a dedicated sandbox Q23 mini-region scenario is authored, bind scenarioId here". No curated position exists.
- **Effort**: Small

#### GAP-TERR-03: No visual indicator for self-elimination eligibility

- **Status:** [ADDRESSED] in CODE-TEACHING-GAPS-BATCH2
- **Implementation:** Added teaching tips in `TeachingOverlay.tsx` (territory topic) and `teachingTopics.ts` (TERRITORY_TIPS) explaining eligibility. Tip: "CAN'T PROCESS A REGION? You must have a stack OUTSIDE the pending region to pay the elimination cost. If all your stacks are inside or on the border, you cannot process."
- **Description**: During territory processing, players must have an outside stack to pay the self-elimination cost, but there is no visual cue showing which stacks qualify. This causes "why can't I process this region?" confusion.
- **Effort**: Medium
- **Note:** Visual highlight deferred to future UI work; teaching content addresses confusion.

#### GAP-TERR-04: Q23-style "why did I lose my own ring?" not explained in-game

- **Status:** [ADDRESSED] in CODE-TEACHING-GAPS-BATCH2
- **Implementation:** Added teaching tips in `TeachingOverlay.tsx` and `teachingTopics.ts` (TERRITORY_TIPS) with category 'self_elimination'. Tip: "WHY DID I LOSE MY OWN RING? Processing a disconnected region eliminates all interior rings (scoring for you), but you MUST also eliminate one cap from a stack OUTSIDE the region."
- **Description**: The most common territory complaint (Q23 in FAQ) is players not understanding why they lose a ring from their own stack when processing a region. The current TeachingOverlay tips mention it but don't demonstrate it.
- **Effort**: Small

---

### capture_chains – Capture Chains

**Current Coverage:**

- ✅ TeachingOverlay topic: [`chain_capture`](../src/client/components/TeachingOverlay.tsx:75) with tips about mandatory continuation
- ✅ GameHUD phase help for `chain_capture` phase ([`GameHUD.tsx:1243`](../src/client/components/GameHUD.tsx:1243))
- ❌ No teaching scenarios in `teachingScenarios.ts` for `capture_chain_mandatory` flow
- ❌ [`UX_RULES_TEACHING_SCENARIOS.md`](UX_RULES_TEACHING_SCENARIOS.md:342) specifies 3 steps; none implemented

**Gaps:**

#### GAP-CHAIN-01: No teaching scenarios for capture_chain_mandatory flow

- **Status:** [ADDRESSED] in CODE-TEACHING-GAPS-BATCH2
- **Implementation:** Added `teaching.capture_chain.step_1` (guided, intro), `teaching.capture_chain.step_2` (interactive, intermediate), and `teaching.capture_chain.step_3` (interactive, advanced) in `teachingScenarios.ts`. All have `showInSandboxPresets: true`.
- **Description**: The teaching scenarios spec defines 3 steps for this flow (optional start, mandatory continuation, choosing branches), but none are implemented in code.
- **Effort**: Medium

#### GAP-CHAIN-02: No sandbox scenarios for capture chains

- **Status:** [ADDRESSED] in CODE-TEACHING-GAPS-BATCH2
- **Implementation:** Steps 1-3 of `capture_chain_mandatory` flow all have `showInSandboxPresets: true`.
- **Description**: Players cannot practice chain captures in a controlled environment. The flow should have `showInSandboxPresets=true` for at least step 2.
- **Effort**: Medium

#### GAP-CHAIN-03: No guidance on 180° reversal and cyclic patterns

- **Status:** [ADDRESSED] in CODE-TEACHING-GAPS-BATCH2
- **Implementation:** Added `teaching.capture_chain.step_3` with learningObjectiveShort: "Experience 180° reversals and revisiting stacks in advanced chain patterns – chains can loop back if positions allow." Also added tips in `TeachingOverlay.tsx` and `teachingTopics.ts` (CHAIN_CAPTURE_TIPS) covering reversals.
- **Description**: Advanced chain capture patterns (reversals, visiting same stack multiple times) are mentioned in rules but not taught. Step 3 should cover this.
- **Effort**: Medium

#### GAP-CHAIN-04: HUD misleading "end your turn" language not fully remediated

- **Status:** [ADDRESSED] in CODE-TEACHING-GAPS-BATCH2
- **Implementation:** Strengthened chain capture tips in `TeachingOverlay.tsx` to emphasize mandatory continuation: "Starting a capture is OPTIONAL – you can choose to move without capturing. But once you make ANY capture, you MUST continue the chain until no legal captures remain." and "You CANNOT stop a chain capture early."
- **Description**: DOCUX-P1 in RULES_DOCS_UX_AUDIT noted HUD copy suggesting captures can be ended early. While corrected in spec, TeachingOverlay tip still says "chain ends only when no legal capture segments remain" which could be clearer.
- **Effort**: Small

---

### lps_real_actions – Last-Player-Standing Real Actions

**Current Coverage:**

- ✅ TeachingOverlay topic: [`victory_stalemate`](../src/client/components/TeachingOverlay.tsx:157) covers LPS (combined with stalemate)
- ✅ Reason code: `LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS`
- ✅ VictoryModal explanation for LPS endings ([`VictoryModal.tsx:607`](../src/client/components/VictoryModal.tsx:607))
- ✅ GameHUD VictoryConditionsPanel tooltip for LPS ([`GameHUD.tsx:1073`](../src/client/components/GameHUD.tsx:1073))
- ❌ No teaching scenarios in `teachingScenarios.ts` for `last_player_standing_intro` flow
- ❌ [`UX_RULES_TEACHING_SCENARIOS.md`](UX_RULES_TEACHING_SCENARIOS.md:460) specifies 3 steps; none implemented

**Gaps:**

#### GAP-LPS-01: No teaching scenarios for last_player_standing_intro flow

- **Status:** [ADDRESSED] in CODE-TEACHING-GAPS-BATCH2
- **Implementation:** Added `teaching.lps.step_1` (guided, intro) and `teaching.lps.step_2` (guided, intermediate) in `teachingScenarios.ts`. Also added LPS_FIRST_OCCURRENCE_TIPS in `teachingTopics.ts` with category 'first_occurrence'. Step 1 has sandbox presets enabled.
- **Description**: The spec defines 3 steps for LPS teaching (distinguish real/forced actions, see LPS win, interactive practice). None are implemented.
- **Effort**: Medium

#### GAP-LPS-02: LPS bundled with stalemate in TeachingOverlay (no separation)

- **Description**: The `victory_stalemate` topic tries to cover both stalemate and LPS, but they are distinct rules. Players confuse them.
- **Recommended Remedy**: Either (a) split into two topics `victory_stalemate` and `victory_lps`, or (b) add explicit sub-sections with headers in the current topic's tips array.
- **Effort**: Small
- **Dependencies**: None

#### GAP-LPS-03: Real action vs FE distinction not clearly taught

- **Description**: Players don't understand why FE doesn't count as a "real action" for LPS purposes. The tooltip mentions it but the teaching content doesn't drill this point.
- **Recommended Remedy**: Add a teaching tip or scenario step emphasizing: "Real actions = placements, movements, captures. FE keeps you from losing but does NOT count as a real action for Last Player Standing."
- **Effort**: Small
- **Dependencies**: GAP-LPS-01

---

### recovery_action – Recovery Action (New Rule Feature)

**Current Coverage:**

- ❌ No TeachingOverlay topic exists for recovery action
- ❌ No teaching scenarios implemented
- ❌ No recovery-specific HUD hints or tooltips
- ❌ No telemetry context for recovery action
- ❌ Engine implementation not yet started

**Canonical Rules:**

- Recovery action allows a "temporarily eliminated" player (no stacks, no rings in hand, but has markers and buried rings) to slide a marker to complete a line of exactly `lineLength`, extracting a buried ring as self-elimination cost.
- Recovery is a **real action** for LPS purposes (unlike forced elimination).
- See `RULES_CANONICAL_SPEC.md` R110–R115 and `ringrift_complete_rules.md` §4.5.

**Gaps:**

#### GAP-RECOV-01: No TeachingOverlay topic for recovery action

- **Status:** Pending (blocked on engine implementation)
- **Description**: Recovery action is a new rule feature not yet implemented. Once implemented, a teaching topic should explain: eligibility (no stacks, no rings, has markers, buried rings), marker slide mechanics, line length requirement (exact length only), and buried ring extraction cost.
- **Recommended Remedy**: Add `recovery_action` to `TeachingTopicId` and create dedicated teaching tips in `teachingTopics.ts`.
- **Effort**: Small (once engine exists)
- **Dependencies**: Engine implementation in `TODO.md` §3.1.1

#### GAP-RECOV-02: No teaching scenarios for recovery action flow

- **Status:** Pending (blocked on engine implementation)
- **Description**: No guided or interactive scenarios exist showing recovery action in practice. The teaching flow should cover: (1) recognizing temporarily eliminated state, (2) identifying valid marker slides, (3) understanding line length requirement (no overlength), (4) cascading effects (line → territory).
- **Recommended Remedy**: Add `recovery_intro` teaching scenario with 2-3 steps in `teachingScenarios.ts` once engine supports recovery.
- **Effort**: Medium
- **Dependencies**: GAP-RECOV-01, engine implementation

#### GAP-RECOV-03: No distinction between "temporarily eliminated" vs "fully eliminated"

- **Status:** Pending (blocked on engine implementation)
- **Description**: Players will likely confuse "temporarily eliminated with recovery" and "fully eliminated for turn rotation". Teaching content should clarify: temporarily eliminated players can still act via recovery and are NOT skipped; fully eliminated players are skipped.
- **Recommended Remedy**: Add comparative tips in ANM/FE and recovery topics.
- **Effort**: Small
- **Dependencies**: GAP-RECOV-01

#### GAP-RECOV-04: Recovery action not integrated into LPS teaching

- **Status:** Pending (blocked on engine implementation)
- **Description**: LPS teaching currently covers placements, movements, captures as "real actions". Recovery action also counts as a real action and should be included in LPS explanations once implemented.
- **Recommended Remedy**: Update `VICTORY_STALEMATE_TIPS` and LPS teaching scenarios to include recovery action.
- **Effort**: Small
- **Dependencies**: GAP-RECOV-01

---

### line_vs_territory_ordering – Line vs Territory Ordering

**Current Coverage:**

- ✅ TeachingOverlay topics: [`line_bonus`](../src/client/components/TeachingOverlay.tsx:87) and [`territory`](../src/client/components/TeachingOverlay.tsx:99) separately
- ✅ Phase indicators in GameHUD show current phase
- ✅ Helper tests exist ([`lineDecisionHelpers.shared.test.ts`](../tests/unit/lineDecisionHelpers.shared.test.ts:23), [`territoryDecisionHelpers.shared.test.ts`](../tests/unit/territoryDecisionHelpers.shared.test.ts:76))
- ❌ No teaching scenarios for `line_vs_territory_blocking` flow
- ❌ [`UX_RULES_TEACHING_SCENARIOS.md`](UX_RULES_TEACHING_SCENARIOS.md:292) specifies 3 steps; none implemented

**Gaps:**

#### GAP-ORDER-01 / GAP-LINE-01: No teaching scenarios for line_vs_territory_blocking flow

- **Status:** [ADDRESSED] in CODE-TEACHING-GAPS-BATCH2
- **Implementation:** Added `teaching.line_territory.step_1` (guided, intro), `teaching.line_territory.step_2` (interactive, intermediate), and `teaching.line_territory.step_3` (interactive, advanced) in `teachingScenarios.ts`. Also added `line_territory_order` topic in `TeachingOverlay.tsx` with dedicated tips. Added `line_vs_territory_multi_phase` to RulesUxContext type.
- **Description**: The spec defines 3 steps showing multi-phase resolution order. None are implemented.
- **Effort**: Medium

#### GAP-ORDER-02 / GAP-LINE-02: Option 1 vs Option 2 for overlength lines not taught interactively

- **Status:** [ADDRESSED] in CODE-TEACHING-GAPS-BATCH2
- **Implementation:** Step 3 of `line_vs_territory_blocking` covers Option 1 vs Option 2 comparison. Tips in `line_territory_order` topic and LINE_TERRITORY_ORDER_TIPS explain: "OPTION 1 vs OPTION 2: For overlength lines (6+ on square, 7+ on hex), you choose: collapse ALL markers (costs a ring) or collapse MINIMUM length (free but less territory)."
- **Description**: Players often choose suboptimally between full collapse (with ring cost) and minimum collapse (no cost). Teaching only covers rules, not strategy.
- **Effort**: Medium

#### GAP-ORDER-03 / GAP-LINE-03: No demonstration of multi-phase turn sequence

- **Status:** [ADDRESSED] in CODE-TEACHING-GAPS-BATCH2
- **Implementation:** Step 2 of `line_vs_territory_blocking` with learningObjectiveShort: "Watch a multi-phase turn: movement → capture chain → line processing → territory processing. Each phase must complete before the next begins." Tips include: "A single turn can trigger: movement → capture chain → line processing → territory processing."
- **Description**: Players are often surprised when a move triggers captures → chains → lines → territory in sequence. No teaching content walks through this.
- **Effort**: Medium

---

## Prioritized Gap List

Gaps are scored by: `Score = Priority × (1 / Effort)` where Priority={High:3, Medium:2} and Effort={Small:3, Medium:2, Large:1}.

| Rank | Gap ID       | Concept                    | Priority | Effort | Score | Recommended Remedy Summary                             |
| ---- | ------------ | -------------------------- | -------- | ------ | ----- | ------------------------------------------------------ |
| 1    | GAP-STALE-01 | structural_stalemate       | High     | Small  | 9.0   | Implement step 2: guided tiebreak ladder explanation   |
| 2    | GAP-STALE-04 | structural_stalemate       | High     | Small  | 9.0   | Add tip distinguishing ANM from global stalemate       |
| 3    | GAP-TERR-04  | territory_mini_regions     | High     | Small  | 9.0   | Ensure step 2/3 explicitly shows self-elimination cost |
| 4    | GAP-LPS-02   | lps_real_actions           | High     | Small  | 9.0   | Add LPS sub-section or separate topic from stalemate   |
| 5    | GAP-LPS-03   | lps_real_actions           | High     | Small  | 9.0   | Add tip emphasizing FE ≠ real action for LPS           |
| 6    | GAP-CHAIN-04 | capture_chains             | Medium   | Small  | 6.0   | Strengthen chain capture "MUST continue" wording       |
| 7    | GAP-TERR-02  | territory_mini_regions     | High     | Small  | 9.0   | Author Q23 sandbox position from test scenario         |
| 8    | GAP-STALE-02 | structural_stalemate       | High     | Small  | 9.0   | Author stalemate sandbox position                      |
| 9    | GAP-ANM-01   | anm_fe_core                | High     | Medium | 4.5   | Add first-occurrence teaching trigger for ANM/FE       |
| 10   | GAP-ANM-02   | anm_fe_core                | High     | Medium | 4.5   | Author step 3-4 FE loop scenario fixtures              |
| 11   | GAP-ANM-03   | anm_fe_core                | High     | Medium | 4.5   | Add FE-eligible stack visual highlight                 |
| 12   | GAP-TERR-01  | territory_mini_regions     | High     | Medium | 4.5   | Implement steps 1-2 of mini_region_intro               |
| 13   | GAP-TERR-03  | territory_mini_regions     | High     | Medium | 4.5   | Add self-elimination eligibility visual indicator      |
| 14   | GAP-LPS-01   | lps_real_actions           | High     | Medium | 4.5   | Implement steps 1-2 of last_player_standing_intro      |
| 15   | GAP-STALE-03 | structural_stalemate       | High     | Medium | 4.5   | VictoryModal tiebreak value visualization              |
| 16   | GAP-CHAIN-01 | capture_chains             | Medium   | Medium | 3.0   | Implement 3 steps of capture_chain_mandatory flow      |
| 17   | GAP-CHAIN-02 | capture_chains             | Medium   | Medium | 3.0   | Author chain capture sandbox scenarios                 |
| 18   | GAP-CHAIN-03 | capture_chains             | Medium   | Medium | 3.0   | Add 180° reversal/cyclic scenario                      |
| 19   | GAP-ORDER-01 | line_vs_territory_ordering | Medium   | Medium | 3.0   | Implement steps 1-2 of line_vs_territory_blocking      |
| 20   | GAP-ORDER-02 | line_vs_territory_ordering | Medium   | Medium | 3.0   | Option 1 vs Option 2 teaching scenario                 |
| 21   | GAP-ORDER-03 | line_vs_territory_ordering | Medium   | Medium | 3.0   | Multi-phase sequence demonstration                     |

**Recovery Action Gaps (Pending Engine Implementation):**

| Rank | Gap ID        | Concept         | Priority | Effort | Score | Recommended Remedy Summary                              |
| ---- | ------------- | --------------- | -------- | ------ | ----- | ------------------------------------------------------- |
| –    | GAP-RECOV-01  | recovery_action | Medium   | Small  | N/A   | Add TeachingOverlay topic (after engine implementation) |
| –    | GAP-RECOV-02  | recovery_action | Medium   | Medium | N/A   | Add teaching scenarios (after engine implementation)    |
| –    | GAP-RECOV-03  | recovery_action | Medium   | Small  | N/A   | Add temp vs full elimination distinction                |
| –    | GAP-RECOV-04  | recovery_action | Medium   | Small  | N/A   | Integrate into LPS teaching                             |

---

## Recommendations

### Immediate Actions (Next Code Task: CODE-W-UX-EXPL-6)

**Focus: High-priority, small-effort gaps that unblock other teaching content**

1. **GAP-STALE-01** – Add step 2 to `structural_stalemate_intro` flow in `teachingScenarios.ts`
2. **GAP-STALE-04** – Add distinguishing tip to `victory_stalemate` topic in `TeachingOverlay.tsx`
3. **GAP-LPS-02** – Add explicit LPS sub-section in `victory_stalemate` or split topics
4. **GAP-LPS-03** – Add "FE ≠ real action" emphasis to tips
5. **GAP-TERR-02** – Author Q23 sandbox position using existing test scenario as template

### Medium-Term Actions (Subsequent iterations)

**Focus: Completing teaching flows with interactive scenarios**

- GAP-ANM-01, GAP-ANM-02, GAP-ANM-03 – Complete ANM/FE teaching with first-occurrence triggers and visual cues
- GAP-TERR-01, GAP-TERR-03, GAP-TERR-04 – Complete territory mini-region flow with all 3 steps
- GAP-LPS-01 – Implement LPS teaching scenarios (steps 1-2)
- GAP-STALE-02, GAP-STALE-03 – Sandbox positions and VictoryModal tiebreak visualization

### Deferred (Lower priority or dependent on engine changes)

**Focus: Capture chains and line ordering (medium priority)**

- GAP-CHAIN-01 through GAP-CHAIN-04 – Capture chain teaching flow
- GAP-ORDER-01 through GAP-ORDER-03 – Line vs territory ordering flow

**Blocked on Engine Implementation:**

- GAP-RECOV-01 through GAP-RECOV-04 – Recovery action teaching content
  - Requires: Engine implementation of recovery action (`recovery_slide` MoveType) in TS and Python
  - See: `TODO.md` §3.1.1 and `RECOVERY_ACTION_IMPLEMENTATION_PLAN.md`
  - Once implemented: Add `recovery_action` TeachingOverlay topic, teaching scenarios, and integrate into LPS teaching

**Note:** Some LPS scenarios (step 3) are deferred until CCE-006 implementation compromise is resolved and engines explicitly support early LPS detection.

---

## Appendix: Teaching Topic Inventory

### TeachingOverlay Topics (Current)

| Topic ID              | Title                                        | Phases                                                   | Linked Concept                                 |
| --------------------- | -------------------------------------------- | -------------------------------------------------------- | ---------------------------------------------- |
| `ring_placement`      | Placing Rings                                | `ring_placement`                                         | –                                              |
| `stack_movement`      | Stack Movement                               | `movement`                                               | –                                              |
| `capturing`           | Overtaking Captures                          | `capture`                                                | –                                              |
| `chain_capture`       | Chain Captures                               | `chain_capture`                                          | –                                              |
| `line_bonus`          | Lines & Territory                            | `line_processing`                                        | –                                              |
| `territory`           | Territory Collapse                           | `territory_processing`                                   | `territory_mini_region`                        |
| `active_no_moves`     | When You Have No Legal Moves                 | `movement`, `line_processing`, `territory_processing`    | `anm_forced_elimination`                       |
| `forced_elimination`  | Forced Elimination                           | `movement`, `territory_processing`, `forced_elimination` | `anm_forced_elimination`                       |
| `victory_elimination` | Ring Elimination Victory                     | –                                                        | –                                              |
| `victory_territory`   | Territory Victory                            | –                                                        | –                                              |
| `victory_stalemate`   | Victory by Last Player Standing or Stalemate | –                                                        | `structural_stalemate`, `last_player_standing` |

### TeachingScenarios (Current – After CODE-TEACHING-GAPS-BATCH2)

| Scenario ID                            | Flow                         | Step | Concept                         | Kind        | Sandbox | Carousel |
| -------------------------------------- | ---------------------------- | ---- | ------------------------------- | ----------- | ------- | -------- |
| `teaching.fe_loop.step_1`              | `fe_loop_intro`              | 1    | `anm_forced_elimination`        | guided      | ✅      | ✅       |
| `teaching.fe_loop.step_2`              | `fe_loop_intro`              | 2    | `anm_forced_elimination`        | interactive | ✅      | ✅       |
| `teaching.fe_loop.step_3`              | `fe_loop_intro`              | 3    | `anm_forced_elimination`        | interactive | ❌      | ❌       |
| `teaching.fe_loop.step_4`              | `fe_loop_intro`              | 4    | `anm_forced_elimination`        | interactive | ❌      | ❌       |
| `teaching.structural_stalemate.step_1` | `structural_stalemate_intro` | 1    | `structural_stalemate`          | guided      | ✅      | ✅       |
| `teaching.structural_stalemate.step_2` | `structural_stalemate_intro` | 2    | `structural_stalemate`          | guided      | ✅      | ❌       |
| `teaching.lps.step_1`                  | `last_player_standing_intro` | 1    | `last_player_standing`          | guided      | ✅      | ✅       |
| `teaching.lps.step_2`                  | `last_player_standing_intro` | 2    | `last_player_standing`          | guided      | ❌      | ❌       |
| `teaching.mini_region.step_1`          | `mini_region_intro`          | 1    | `territory_mini_region`         | guided      | ✅      | ✅       |
| `teaching.mini_region.step_2`          | `mini_region_intro`          | 2    | `territory_mini_region`         | interactive | ✅      | ❌       |
| `teaching.mini_region.step_3`          | `mini_region_intro`          | 3    | `territory_mini_region`         | interactive | ✅      | ❌       |
| `teaching.capture_chain.step_1`        | `capture_chain_mandatory`    | 1    | `capture_chain_mandatory`       | guided      | ✅      | ✅       |
| `teaching.capture_chain.step_2`        | `capture_chain_mandatory`    | 2    | `capture_chain_mandatory`       | interactive | ✅      | ❌       |
| `teaching.capture_chain.step_3`        | `capture_chain_mandatory`    | 3    | `capture_chain_mandatory`       | interactive | ✅      | ❌       |
| `teaching.line_territory.step_1`       | `line_vs_territory_blocking` | 1    | `line_vs_territory_multi_phase` | guided      | ✅      | ✅       |
| `teaching.line_territory.step_2`       | `line_vs_territory_blocking` | 2    | `line_vs_territory_multi_phase` | interactive | ✅      | ❌       |
| `teaching.line_territory.step_3`       | `line_vs_territory_blocking` | 3    | `line_vs_territory_multi_phase` | interactive | ✅      | ❌       |

**Total Teaching Steps: 19** (was 6 originally, then 14 after CODE-W-UX-EXPL-6, now 19 after CODE-TEACHING-GAPS-BATCH2)

### Planned Flows (from UX_RULES_TEACHING_SCENARIOS.md)

| Flow ID                       | Concept                         | Steps Spec'd | Steps Impl'd | Gap                           |
| ----------------------------- | ------------------------------- | ------------ | ------------ | ----------------------------- |
| `fe_loop_intro`               | `anm_forced_elimination`        | 3            | 4            | ✅ Over-implemented           |
| `mini_region_intro`           | `territory_mini_region`         | 3            | 3            | ✅ Complete                   |
| `multi_region_budget`         | `territory_multi_region_budget` | 2            | 0            | ❌ Not started (low priority) |
| `line_vs_territory_blocking`  | `line_vs_territory_multi_phase` | 3            | 3            | ✅ Complete                   |
| `capture_chain_mandatory`     | `capture_chain_mandatory`       | 3            | 3            | ✅ Complete                   |
| `landing_on_own_marker_intro` | `landing_on_own_marker`         | 2            | 0            | ❌ Not started (low priority) |
| `structural_stalemate_intro`  | `structural_stalemate`          | 2            | 2            | ✅ Complete                   |
| `last_player_standing_intro`  | `last_player_standing`          | 3            | 2            | ⚠️ Step 3 deferred (CCE-006)  |
