# UX Rules Game-End Explanation Model Spec

> **Doc Status (2025-12-05): Draft – W‑UX‑7 (Structured "Why Did the Game End?" Explanation Model)**
>
> **Role:** Define a single structured payload, `GameEndExplanation`, that explains **why** a RingRift game ended and can be consumed consistently by HUD, VictoryModal, sandbox, and teaching overlays.
>
> **Inputs:**
>
> - Weakest-aspect assessment in [`WEAKNESS_AND_HARDEST_PROBLEM_REPORT.md`](WEAKNESS_AND_HARDEST_PROBLEM_REPORT.md:92) – players do not understand ANM/FE, structural stalemates, and mini-region territory endings.
> - Remediation plan W‑UX‑7 in [`NEXT_WAVE_REMEDIATION_PLAN.md`](NEXT_WAVE_REMEDIATION_PLAN.md:1).
> - Rules concepts in [`docs/UX_RULES_CONCEPTS_INDEX.md`](docs/UX_RULES_CONCEPTS_INDEX.md:1).
> - Weird-state reasons in [`docs/UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:71) and [`src/shared/engine/weirdStateReasons.ts`](src/shared/engine/weirdStateReasons.ts:1).
> - Teaching flows in [`docs/UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:33) and [`src/shared/teaching/teachingScenarios.ts`](src/shared/teaching/teachingScenarios.ts:1).
> - Rules canon in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:193) and [`ringrift_complete_rules.md`](ringrift_complete_rules.md:260).
> - Telemetry taxonomy in [`docs/UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:145) and [`rulesUxEvents.ts`](src/shared/telemetry/rulesUxEvents.ts:1).

---

## 1. Introduction & goals

This spec addresses the weakest-UX problem identified in [`WEAKNESS_AND_HARDEST_PROBLEM_REPORT.md`](WEAKNESS_AND_HARDEST_PROBLEM_REPORT.md:92): players often do not understand *why* the game ended, especially in Active‑No‑Moves / Forced Elimination (ANM/FE), structural stalemate, Last Player Standing (LPS), and territory mini‑region scenarios.
Instead of each surface re-deriving a bespoke explanation, the engine or adapter layer produces a **single structured `GameEndExplanation` payload** that all end-of-game UX consumers share.

Goals:

- Deliver **consistent explanations** across HUD, VictoryModal, sandbox, replay, and teaching overlays.
- Bridge **raw engine outcomes** with **weird-state reasons**, **rules references**, **concept ids**, and **teaching flows**.
- Keep the model **extensible and backwards-compatible** so that new reasons, rules references, or teaching flows can be added without breaking existing consumers.
- Stay **semantics-first**: explain *why* the game ended and what rules applied, not how UI chooses to present it.

## 2. Scope and non-goals

In scope:

- **Square-8 2-player core** use case is primary, but the model must not preclude other boards (Square-19, hexagonal) or player counts.
- Explanations for standard outcomes:
  - Ring-elimination and territory victories.
  - Last Player Standing (LPS) once implemented per [`RR‑CANON‑R172`](RULES_CANONICAL_SPEC.md:603).
  - Resignation, timeout, and abandonment (for completeness, even if trivially explainable).
- Explanations for advanced or confusing outcomes:
  - LPS outcomes where ANM/FE contributed heavily.
  - Structural stalemates resolved by the **four-step tiebreak ladder** (territory → eliminated rings → markers → last real action) per [`RR‑CANON‑R173`](RULES_CANONICAL_SPEC.md:619).
  - Territory mini-region **self-elimination** and Q23-style cases where interior eliminations and mandatory external self-elimination decide the result.

Out of scope:

- Exact UX copy strings; those live in [`docs/UX_RULES_COPY_SPEC.md`](docs/UX_RULES_COPY_SPEC.md:331) and localisation resources.
- Detailed UI layout or component behaviour of HUD, VictoryModal, or overlays.
- Any change to rules semantics, scoring, or tiebreak ordering defined in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:193).
- Low-level engine flags or full game-record schema (already covered by existing engine specs).

## 3. Data model – `GameEndExplanation`

The engine or server adapter produces a single structured object, `GameEndExplanation`, at the moment the game ends. All UX surfaces (HUD, VictoryModal, sandbox, TeachingOverlay) consume this object **read-only**.

The notation below is pseudo-TypeScript for clarity; it does not prescribe exact implementation details.

```ts
type PlayerId = string;

type GameEndOutcomeType =
  | 'ring_majority'
  | 'territory_control'
  | 'last_player_standing'
  | 'structural_stalemate'
  | 'resignation'
  | 'timeout'
  | 'abandoned';

type GameEndVictoryReasonCode =
  | 'victory_ring_majority_standard'
  | 'victory_ring_majority_time'
  | 'victory_territory_control'
  | 'victory_structural_stalemate_tiebreak'
  | 'victory_last_player_standing'
  | 'victory_resignation'
  | 'victory_timeout'
  | 'victory_abandoned';

// Aligns with RulesContext / rules_context taxonomy in
// UX_RULES_TELEMETRY_SPEC and rulesUxEvents.ts.
type RulesContextTag =
  | 'victory_elimination'
  | 'victory_territory'
  | 'structural_stalemate'
  | 'last_player_standing'
  | 'anm_forced_elimination'
  | 'anm_line_no_actions'
  | 'anm_territory_no_actions'
  | 'territory_mini_region'
  | 'territory_multi_region'
  | 'other';

// Aligns with concept ids in UX_RULES_CONCEPTS_INDEX.md.
type RulesConceptId =
  | 'anm_forced_elimination'
  | 'structural_stalemate'
  | 'last_player_standing'
  | 'territory_mini_regions'
  | 'territory_multi_region_budget'
  | 'victory_elimination'
  | 'victory_territory'
  | 'line_vs_territory_multi_phase'
  | 'capture_chain_mandatory'
  | string;

// Canonical weird-state reason codes; actual enum lives in
// src/shared/engine/weirdStateReasons.ts.
type RulesWeirdStateReasonCode =
  | 'ANM_MOVEMENT_FE_BLOCKED'
  | 'ANM_LINE_NO_ACTIONS'
  | 'ANM_TERRITORY_NO_ACTIONS'
  | 'FE_SEQUENCE_CURRENT_PLAYER'
  | 'STRUCTURAL_STALEMATE_TIEBREAK'
  | 'LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS'
  | string;

type GameEndScoreBreakdown = {
  ringsEliminated: number;
  ringsEliminatedBySelf: number;
  territorySpaces: number;
  markers: number;
  bonusPoints: number;
  penaltyPoints: number;
  totalScore: number;
};

type GameEndTiebreakCriterion =
  | 'territory'
  | 'eliminated_rings'
  | 'markers'
  | 'last_real_action';

type GameEndTiebreakStep = {
  index: number; // 1-based step in the ladder
  criterion: GameEndTiebreakCriterion;
  perPlayer: Record<PlayerId, number>;
  winningPlayerIds: PlayerId[]; // can be multiple until final step
  resolved: boolean; // true when this step yields a unique winner
};

type GameEndWeirdStateContext = {
  reasonCodes: RulesWeirdStateReasonCode[];
  primaryReasonCode?: RulesWeirdStateReasonCode;
  rulesContextTags: RulesContextTag[];
  primaryConceptId?: RulesConceptId;
};

type GameEndRulesDocId =
  | 'RULES_CANON'
  | 'RINGRIFT_COMPLETE_RULES'
  | 'RINGRIFT_FAQ'
  | 'ACTIVE_NO_MOVES_BEHAVIOUR'
  | 'RULES_DYNAMIC_VERIFICATION'
  | string;

type GameEndRulesReference = {
  id: string; // stable id, e.g. 'RR-CANON-R173' or 'FAQ-Q23'
  doc: GameEndRulesDocId;
  anchor: string; // document-appropriate anchor or section id
  label: string; // short label for "Learn more" UI
};

type TeachingTopicId = string; // from UX_RULES_TEACHING_SCENARIOS / TeachingOverlay
type TeachingFlowId = string; // e.g. 'fe_loop_intro', 'mini_region_intro'

type GameEndTeachingLink = {
  primaryConceptId?: RulesConceptId;
  teachingTopicIds: TeachingTopicId[];
  recommendedFlowIds?: TeachingFlowId[];
  priority: 'primary' | 'secondary';
};

type GameEndUxCopy = {
  shortSummaryKey: string; // HUD / compact banner
  detailedSummaryKey: string; // VictoryModal / overlays
  badgeKey?: string; // optional badge / chip label key
  iconKey?: string; // optional icon id for UI
  followupCtaKeys?: string[]; // optional CTA keys ("View teaching flow", etc.)
};

type GameEndNextActionType =
  | 'open_teaching_flow'
  | 'open_rules_doc'
  | 'open_sandbox_replay'
  | 'share_game_record';

type GameEndNextAction = {
  type: GameEndNextActionType;
  labelKey: string;
  teachingFlowId?: TeachingFlowId;
  rulesReferenceId?: string;
};

type GameEndDebugInfo = {
  rawOutcomeReason?: string; // engine result enum / reason
  rawWeirdStateFlags?: string[];
  terminationMoveIndex?: number;
  parityTraceId?: string; // optional link into parity / AI trace tooling
};

type GameEndExplanation = {
  schemaVersion: 1;

  // Top-level game metadata
  gameId: string;
  rulesetId: string; // e.g. 'square8_2p_core'
  boardType: 'square8' | 'square19' | 'hexagonal' | string;
  numPlayers: number;

  // Outcome basics
  winningPlayerId: PlayerId | null; // null for pure draws
  outcomeType: GameEndOutcomeType;
  victoryReasonCode: GameEndVictoryReasonCode;
  isDraw: boolean;

  // Scoring / tiebreak detail
  scoreBreakdown?: Record<PlayerId, GameEndScoreBreakdown>;
  tiebreakSteps?: GameEndTiebreakStep[]; // populated for structural stalemate or complex ties

  // Weird-state / rules context
  weirdStateContext?: GameEndWeirdStateContext;
  rulesContextTags: RulesContextTag[]; // overall tags for telemetry & filtering

  // Rules docs and teaching
  rulesReferences?: GameEndRulesReference[];
  teachingLinks?: GameEndTeachingLink[];

  // UX copy and suggested next actions
  uxCopy: GameEndUxCopy;
  nextActions?: GameEndNextAction[];

  // Optional debug block for developer tooling only
  debug?: GameEndDebugInfo;
};
```

### 3.1 Design notes

- **Single source of truth:** All consumers read `GameEndExplanation`; none re-derive weird-state flags, LPS detection, or tiebreak details.
- **Alignment to existing taxonomies:**
  - `RulesContextTag` corresponds to `rules_context` in [`docs/UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:145) and the `RulesContext` type in [`rulesUxEvents.ts`](src/shared/telemetry/rulesUxEvents.ts:1).
  - `RulesWeirdStateReasonCode` is the same enum defined in [`src/shared/engine/weirdStateReasons.ts`](src/shared/engine/weirdStateReasons.ts:1) and documented in [`docs/UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:71).
  - `RulesConceptId` values come from [`docs/UX_RULES_CONCEPTS_INDEX.md`](docs/UX_RULES_CONCEPTS_INDEX.md:1).
  - `TeachingTopicId` / `TeachingFlowId` align with topics and flows in [`docs/UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:136) and [`src/shared/teaching/teachingScenarios.ts`](src/shared/teaching/teachingScenarios.ts:1).
- **Versioned schema:** `schemaVersion` allows additive evolution (new optional fields, new enum members) without breaking existing consumers.
- **Optional detail:** `scoreBreakdown`, `tiebreakSteps`, and `debug` can be omitted in trivial endings (simple resignation) but MUST be populated for structural stalemate and other parity-sensitive endings.

### 3.2 Top-level metadata

Top-level fields (`gameId`, `rulesetId`, `boardType`, `numPlayers`) allow:

- HUD and VictoryModal to show succinct metadata ("Square-8, 2 players").
- TeachingOverlay and sandbox to filter or jump to **similar games** (e.g. other Square-8 structural stalemates).
- Telemetry and offline analysis to group explanations by ruleset and board type.

### 3.3 Victory detail

- `outcomeType` and `victoryReasonCode` summarise *how* the game ended at a high level (ring majority, territory, LPS, stalemate, resignation, timeout).
- `scoreBreakdown` provides per-player scores in a standardised shape that matches elimination + territory semantics in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:170) and [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1334).
- `tiebreakSteps` encodes the **stalemate ladder** from [`RR‑CANON‑R173`](RULES_CANONICAL_SPEC.md:619):
  - Step 1: compare territory spaces.
  - Step 2: compare eliminated rings (including rings in hand).
  - Step 3: compare markers.
  - Step 4: compare "last real action" actor.

For non-stalemate endings, `tiebreakSteps` MAY be omitted or MAY contain a single resolved step for parity debugging (implementation choice).

### 3.4 Weird-state and rules context

`weirdStateContext` exists so that ANM/FE, structural stalemate, and LPS information is present **even when it is not the nominal victory condition**. Examples:

- FE sequences that shrank a player's stacks leading up to a standard ring-majority win.
- A structural stalemate state that almost occurred before one player converted territory into elimination.

Key design points:

- `reasonCodes` is a list of `RulesWeirdStateReasonCode` values from [`src/shared/engine/weirdStateReasons.ts`](src/shared/engine/weirdStateReasons.ts:1).
- `primaryReasonCode` highlights the single most salient weird-state reason (if any) for HUD / VictoryModal emphasis.
- `rulesContextTags` is a small set of tags for telemetry / filtering, drawn from the same taxonomy as `rules_context` in [`docs/UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:145).
- `primaryConceptId` links directly into the concepts index, e.g. `'anm_forced_elimination'`, `'structural_stalemate'`, `'territory_mini_regions'`.

### 3.5 Rules references

`rulesReferences` is an optional list of identifiers suitable for "Learn more" affordances. Each entry ties the explanation back to canonical rules text or FAQs, for example:

- `{ id: 'RR-CANON-R173', doc: 'RULES_CANON', anchor: 'R173', label: 'RR‑CANON R173 – Structural stalemate' }`.
- `{ id: 'FAQ-Q23', doc: 'RINGRIFT_FAQ', anchor: 'Q23', label: 'FAQ Q23 – Mini-regions and self-elimination' }`.

Surfaces can choose how many links to display, but SHOULD preserve ids so that telemetry and teaching flows can associate with specific rules references.

### 3.6 Teaching linkage

`teachingLinks` connects a game-end explanation to **teaching topics** and **scenario flows**:

- `primaryConceptId` points at a single high-level concept (e.g. `'structural_stalemate'` or `'territory_mini_regions'`) from [`docs/UX_RULES_CONCEPTS_INDEX.md`](docs/UX_RULES_CONCEPTS_INDEX.md:1).
- `teachingTopicIds` reference TeachingOverlay topics (e.g. `'teaching.active_no_moves'`, `'teaching.victory_stalemate'`) as defined in [`src/client/components/TeachingOverlay.tsx`](src/client/components/TeachingOverlay.tsx:72).
- `recommendedFlowIds` reference scenario flows like `'fe_loop_intro'`, `'mini_region_intro'`, `'structural_stalemate_intro'`, `'last_player_standing_intro'` from [`docs/UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:136) and [`src/shared/teaching/teachingScenarios.ts`](src/shared/teaching/teachingScenarios.ts:1).

This allows VictoryModal or TeachingOverlay to say "Replay a similar situation" or "Walk through a structural stalemate example" without hard-coding rules logic.

### 3.7 UX-oriented fields

- `uxCopy.shortSummaryKey` – compact, stable key for HUD or small banners (e.g. `'game_end.ring_majority.standard.short'`).
- `uxCopy.detailedSummaryKey` – key for full explanations in VictoryModal or overlays (e.g. `'game_end.structural_stalemate.detail'`).
- `uxCopy.badgeKey` / `iconKey` – optional keys to select a small badge or icon (e.g. "Structural stalemate", "Last Player Standing").
- `uxCopy.followupCtaKeys` – optional CTA keys that, combined with `nextActions`, tell clients which buttons to show (e.g. "Open teaching flow", "View rules section").
- `nextActions` – structured hints for follow-up actions (open teaching flow, open rules doc, open sandbox replay, share game record).

Clients MAY ignore some of these fields on constrained surfaces (e.g. HUD) but SHOULD support at least `shortSummaryKey` and `detailedSummaryKey`.

### 3.8 Debugging fields

`debug` carries optional information intended only for developer tooling, logs, or parity checks:

- `rawOutcomeReason` – engine-native result reason enum.
- `rawWeirdStateFlags` – raw engine flags that fed into `RulesWeirdStateReasonCode`.
- `terminationMoveIndex` – move index where the game ended.
- `parityTraceId` – identifier linking to AI parity traces or recorded games in the analysis pipeline.

Debug fields MUST NOT be relied on by user-facing UX copy and MAY be omitted in privacy-sensitive contexts.

## 4. Mapping from existing sources

This section describes, at a conceptual level, where each part of `GameEndExplanation` is populated. Implementation will be done in Code mode (e.g. in a helper such as `buildGameEndExplanation`).

### 4.1 Engine / rules layer

Source: game result / termination logic in the shared engine and server adapter (GameResult, victory evaluation, scoring).

Populates:

- `winningPlayerId`, `outcomeType`, `victoryReasonCode`, `isDraw`.
- `scoreBreakdown` (per-player elimination counts, territory, markers, total score).
- `tiebreakSteps` for structural stalemate or other tiebreak-driven endings, following [`RR‑CANON‑R173`](RULES_CANONICAL_SPEC.md:619).
- `gameId`, `rulesetId`, `boardType`, `numPlayers`.

### 4.2 Weird-state mapping

Source: weird-state detection and mapping in [`src/shared/engine/weirdStateReasons.ts`](src/shared/engine/weirdStateReasons.ts:1) and the UX mapping defined in [`docs/UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:71).

Populates:

- `weirdStateContext.reasonCodes` and `weirdStateContext.primaryReasonCode`.
- `weirdStateContext.rulesContextTags` by mapping reason codes to `rules_context`.
- `weirdStateContext.primaryConceptId` by mapping reason codes to concept ids in [`docs/UX_RULES_CONCEPTS_INDEX.md`](docs/UX_RULES_CONCEPTS_INDEX.md:1).

### 4.3 Rules docs and concepts index

Source: canonical rules and concept indexing in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:193), [`ringrift_complete_rules.md`](ringrift_complete_rules.md:260), and [`docs/UX_RULES_CONCEPTS_INDEX.md`](docs/UX_RULES_CONCEPTS_INDEX.md:1).

Populates:

- `rulesReferences` – mapping outcomes and weird-state reasons to rules sections and FAQs (e.g. R172 LPS, R173 stalemate, FAQ Q23 mini-regions).
- `rulesContextTags` – high-level contexts such as `'victory_elimination'`, `'structural_stalemate'`, `'territory_mini_region'`.
- `teachingLinks.primaryConceptId` – concept ids for ANM/FE, structural stalemate, mini-regions, LPS, etc.

### 4.4 Teaching metadata

Source: teaching topics and flows in [`docs/UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:136) and [`src/shared/teaching/teachingScenarios.ts`](src/shared/teaching/teachingScenarios.ts:1), plus TeachingOverlay topic ids in [`src/client/components/TeachingOverlay.tsx`](src/client/components/TeachingOverlay.tsx:72).

Populates:

- `teachingLinks.teachingTopicIds` – e.g. `'teaching.active_no_moves'`, `'teaching.victory_stalemate'`, `'teaching.territory'`.
- `teachingLinks.recommendedFlowIds` – e.g. `'fe_loop_intro'`, `'structural_stalemate_intro'`, `'last_player_standing_intro'`, `'mini_region_intro'`.

### 4.5 UX copy spec

Source: copy and key naming in [`docs/UX_RULES_COPY_SPEC.md`](docs/UX_RULES_COPY_SPEC.md:331).

Populates:

- `uxCopy.shortSummaryKey` and `uxCopy.detailedSummaryKey` (with new keys added in the copy spec as needed).
- `uxCopy.badgeKey`, `uxCopy.iconKey`, and `uxCopy.followupCtaKeys` (optional).

### 4.6 Telemetry hints

Source: `rules_context` taxonomy and events in [`docs/UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:218) and [`rulesUxEvents.ts`](src/shared/telemetry/rulesUxEvents.ts:1).

Populates:

- `rulesContextTags` – ensuring that the same contexts used for telemetry are also present in explanations.
- Optional inspiration for `nextActions` (e.g. using telemetry to measure whether users follow recommended flows).

## 5. Worked examples

The following examples illustrate how `GameEndExplanation` should be instantiated for representative endings. They are **illustrative only**; exact ids and numeric values are not normative.

### 5.1 Example A – Standard ring-majority win (no weird states)

A straightforward 2-player Square-8 game where Player 1 wins by ring elimination; no ANM/FE or structural stalemate occurred.

```ts
const exampleRingMajority: GameEndExplanation = {
  schemaVersion: 1,
  gameId: 'game_0001',
  rulesetId: 'square8_2p_core',
  boardType: 'square8',
  numPlayers: 2,

  winningPlayerId: 'P1',
  outcomeType: 'ring_majority',
  victoryReasonCode: 'victory_ring_majority_standard',
  isDraw: false,

  scoreBreakdown: {
    P1: {
      ringsEliminated: 5,
      ringsEliminatedBySelf: 0,
      territorySpaces: 8,
      markers: 6,
      bonusPoints: 0,
      penaltyPoints: 0,
      totalScore: 5,
    },
    P2: {
      ringsEliminated: 0,
      ringsEliminatedBySelf: 0,
      territorySpaces: 4,
      markers: 3,
      bonusPoints: 0,
      penaltyPoints: 0,
      totalScore: 0,
    },
  },

  tiebreakSteps: [],

  weirdStateContext: undefined,
  rulesContextTags: ['victory_elimination'],

  rulesReferences: [
    {
      id: 'RR-CANON-R170',
      doc: 'RULES_CANON',
      anchor: 'R170',
      label: 'RR‑CANON R170 – Ring elimination victory',
    },
  ],

  teachingLinks: [
    {
      primaryConceptId: 'victory_elimination',
      teachingTopicIds: ['teaching.victory_elimination'],
      recommendedFlowIds: ['capture_chain_mandatory'],
      priority: 'secondary',
    },
  ],

  uxCopy: {
    shortSummaryKey: 'game_end.ring_majority.standard.short',
    detailedSummaryKey: 'game_end.ring_majority.standard.detail',
    badgeKey: 'badge.victory.ring_majority',
  },

  nextActions: [
    {
      type: 'open_sandbox_replay',
      labelKey: 'game_end.cta.view_replay',
    },
  ],
};
```

### 5.2 Example B – Last Player Standing with ANM/FE involvement

A 3-player Square-8 game ends by LPS: only Player 2 has real actions for a full round, while Players 1 and 3 are stuck in ANM/FE loops.

```ts
const exampleLpsWithAnmFe: GameEndExplanation = {
  schemaVersion: 1,
  gameId: 'game_0102',
  rulesetId: 'square8_3p_core',
  boardType: 'square8',
  numPlayers: 3,

  winningPlayerId: 'P2',
  outcomeType: 'last_player_standing',
  victoryReasonCode: 'victory_last_player_standing',
  isDraw: false,

  scoreBreakdown: undefined, // LPS can be presented without full score details
  tiebreakSteps: [],

  weirdStateContext: {
    reasonCodes: [
      'ANM_MOVEMENT_FE_BLOCKED',
      'FE_SEQUENCE_CURRENT_PLAYER',
      'LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS',
    ],
    primaryReasonCode: 'LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS',
    rulesContextTags: ['last_player_standing', 'anm_forced_elimination'],
    primaryConceptId: 'last_player_standing',
  },
  rulesContextTags: ['last_player_standing'],

  rulesReferences: [
    {
      id: 'RR-CANON-R172',
      doc: 'RULES_CANON',
      anchor: 'R172',
      label: 'RR‑CANON R172 – Last Player Standing',
    },
    {
      id: 'RR-CANON-R100',
      doc: 'RULES_CANON',
      anchor: 'R100',
      label: 'RR‑CANON R100 – Forced elimination when blocked',
    },
  ],

  teachingLinks: [
    {
      primaryConceptId: 'last_player_standing',
      teachingTopicIds: ['teaching.victory_stalemate'],
      recommendedFlowIds: ['last_player_standing_intro'],
      priority: 'primary',
    },
    {
      primaryConceptId: 'anm_forced_elimination',
      teachingTopicIds: ['teaching.active_no_moves', 'teaching.forced_elimination'],
      recommendedFlowIds: ['fe_loop_intro'],
      priority: 'secondary',
    },
  ],

  uxCopy: {
    shortSummaryKey: 'game_end.lps.short',
    detailedSummaryKey: 'game_end.lps.with_anm_fe.detail',
    badgeKey: 'badge.victory.last_player_standing',
  },

  nextActions: [
    {
      type: 'open_teaching_flow',
      labelKey: 'game_end.cta.learn_about_lps',
      teachingFlowId: 'last_player_standing_intro',
    },
    {
      type: 'open_teaching_flow',
      labelKey: 'game_end.cta.learn_about_anm_fe',
      teachingFlowId: 'fe_loop_intro',
    },
  ],
};
```

### 5.3 Example C – Structural stalemate resolved by tiebreak ladder

A 3-player structural stalemate on Square-8 is resolved via the four-step tiebreak ladder. Player 3 wins on the **territory** step.

```ts
const exampleStructuralStalemate: GameEndExplanation = {
  schemaVersion: 1,
  gameId: 'game_0203',
  rulesetId: 'square8_3p_core',
  boardType: 'square8',
  numPlayers: 3,

  winningPlayerId: 'P3',
  outcomeType: 'structural_stalemate',
  victoryReasonCode: 'victory_structural_stalemate_tiebreak',
  isDraw: false,

  scoreBreakdown: {
    P1: {
      ringsEliminated: 2,
      ringsEliminatedBySelf: 0,
      territorySpaces: 10,
      markers: 5,
      bonusPoints: 0,
      penaltyPoints: 0,
      totalScore: 10,
    },
    P2: {
      ringsEliminated: 3,
      ringsEliminatedBySelf: 1,
      territorySpaces: 9,
      markers: 4,
      bonusPoints: 0,
      penaltyPoints: 0,
      totalScore: 9,
    },
    P3: {
      ringsEliminated: 1,
      ringsEliminatedBySelf: 0,
      territorySpaces: 12,
      markers: 3,
      bonusPoints: 0,
      penaltyPoints: 0,
      totalScore: 12,
    },
  },

  tiebreakSteps: [
    {
      index: 1,
      criterion: 'territory',
      perPlayer: { P1: 10, P2: 9, P3: 12 },
      winningPlayerIds: ['P3'],
      resolved: true,
    },
  ],

  weirdStateContext: {
    reasonCodes: ['STRUCTURAL_STALEMATE_TIEBREAK'],
    primaryReasonCode: 'STRUCTURAL_STALEMATE_TIEBREAK',
    rulesContextTags: ['structural_stalemate'],
    primaryConceptId: 'structural_stalemate',
  },
  rulesContextTags: ['structural_stalemate'],

  rulesReferences: [
    {
      id: 'RR-CANON-R173',
      doc: 'RULES_CANON',
      anchor: 'R173',
      label: 'RR‑CANON R173 – Structural stalemate tiebreak ladder',
    },
    {
      id: 'COMPLETE-13.4',
      doc: 'RINGRIFT_COMPLETE_RULES',
      anchor: 'section-13.4',
      label: 'Complete rules §13.4 – End of game stalemate resolution',
    },
  ],

  teachingLinks: [
    {
      primaryConceptId: 'structural_stalemate',
      teachingTopicIds: ['teaching.victory_stalemate'],
      recommendedFlowIds: ['structural_stalemate_intro'],
      priority: 'primary',
    },
  ],

  uxCopy: {
    shortSummaryKey: 'game_end.structural_stalemate.short',
    detailedSummaryKey: 'game_end.structural_stalemate.detail',
    badgeKey: 'badge.victory.structural_stalemate',
  },

  nextActions: [
    {
      type: 'open_teaching_flow',
      labelKey: 'game_end.cta.learn_about_stalemate',
      teachingFlowId: 'structural_stalemate_intro',
    },
  ],
};
```

### 5.4 Example D – Territory mini-region self-elimination (Q23-style)

A Square-8 2-player game ends after a mini-region (Q23 archetype) is processed, including required self-elimination from an external stack, yielding a decisive territory victory.

```ts
const exampleMiniRegionTerritory: GameEndExplanation = {
  schemaVersion: 1,
  gameId: 'game_0304',
  rulesetId: 'square8_2p_core',
  boardType: 'square8',
  numPlayers: 2,

  winningPlayerId: 'P1',
  outcomeType: 'territory_control',
  victoryReasonCode: 'victory_territory_control',
  isDraw: false,

  scoreBreakdown: {
    P1: {
      ringsEliminated: 4,
      ringsEliminatedBySelf: 1, // mandatory self-elimination outside the region
      territorySpaces: 9,
      markers: 4,
      bonusPoints: 0,
      penaltyPoints: 0,
      totalScore: 9,
    },
    P2: {
      ringsEliminated: 1,
      ringsEliminatedBySelf: 0,
      territorySpaces: 3,
      markers: 2,
      bonusPoints: 0,
      penaltyPoints: 0,
      totalScore: 3,
    },
  },

  tiebreakSteps: [],

  weirdStateContext: {
    reasonCodes: ['ANM_TERRITORY_NO_ACTIONS'],
    primaryReasonCode: 'ANM_TERRITORY_NO_ACTIONS',
    rulesContextTags: ['territory_mini_region'],
    primaryConceptId: 'territory_mini_regions',
  },
  rulesContextTags: ['victory_territory', 'territory_mini_region'],

  rulesReferences: [
    {
      id: 'FAQ-Q23',
      doc: 'RINGRIFT_FAQ',
      anchor: 'Q23',
      label: 'FAQ Q23 – Territory mini-regions and self-elimination',
    },
    {
      id: 'RR-CANON-R141-145',
      doc: 'RULES_CANON',
      anchor: 'R141-145',
      label: 'RR‑CANON R141–R145 – Territory region processing',
    },
  ],

  teachingLinks: [
    {
      primaryConceptId: 'territory_mini_regions',
      teachingTopicIds: ['teaching.territory'],
      recommendedFlowIds: ['mini_region_intro'],
      priority: 'primary',
    },
  ],

  uxCopy: {
    shortSummaryKey: 'game_end.territory.mini_region.short',
    detailedSummaryKey: 'game_end.territory.mini_region.detail',
    badgeKey: 'badge.victory.territory',
  },

  nextActions: [
    {
      type: 'open_teaching_flow',
      labelKey: 'game_end.cta.learn_about_mini_regions',
      teachingFlowId: 'mini_region_intro',
    },
    {
      type: 'open_rules_doc',
      labelKey: 'game_end.cta.read_faq_q23',
      rulesReferenceId: 'FAQ-Q23',
    },
  ],
};
```

## 6. Integration notes & follow-up tasks

This section identifies the likely producer and consumers of `GameEndExplanation` and outlines concrete follow-up tasks for Code mode.

### 6.1 Producer

- Implement a shared helper such as `buildGameEndExplanation(gameResult, context)` in a shared module accessible to both server and client (or at least to the primary game server).
- The helper should:
  - Take the final `GameResult` and any additional termination context (e.g. last move index, raw weird-state flags).
  - Call into existing victory / scoring logic to populate `outcomeType`, `victoryReasonCode`, `scoreBreakdown`, and `tiebreakSteps`.
  - Call into the weird-state mapping from [`src/shared/engine/weirdStateReasons.ts`](src/shared/engine/weirdStateReasons.ts:1) and [`docs/UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:71) to populate `weirdStateContext`.
  - Map outcomes and weird states to concept ids in [`docs/UX_RULES_CONCEPTS_INDEX.md`](docs/UX_RULES_CONCEPTS_INDEX.md:1) and to teaching flows in [`docs/UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:136).
  - Attach `rulesReferences`, `rulesContextTags`, and `uxCopy` keys using small mapping tables aligned with [`docs/UX_RULES_COPY_SPEC.md`](docs/UX_RULES_COPY_SPEC.md:331).

### 6.2 Consumers

Primary consumers:

- **HUD** (e.g. [`src/client/components/GameHUD.tsx`](src/client/components/GameHUD.tsx:1)) – can show compact "Why the game ended" banners using `uxCopy.shortSummaryKey`, `uxCopy.badgeKey`, and `rulesContextTags`.
- **VictoryModal** (e.g. [`src/client/components/VictoryModal.tsx`](src/client/components/VictoryModal.tsx:1)) – can render full explanations using `uxCopy.detailedSummaryKey`, tiebreak visualisations from `tiebreakSteps`, and links from `rulesReferences` and `teachingLinks`.
- **TeachingOverlay** (e.g. [`src/client/components/TeachingOverlay.tsx`](src/client/components/TeachingOverlay.tsx:72)) – can use `teachingLinks` and `nextActions` to recommend scenario flows.
- **Sandbox / replay tools** – can surface `GameEndExplanation` alongside controls, and use `debug` to show technical details when needed.

Secondary consumers:

- **Telemetry** – can log `rulesContextTags`, `weirdStateContext.reasonCodes`, and `rulesReferences` ids alongside user interactions with post-game surfaces.
- **AI / parity analysis** – can associate parity traces or training examples with specific end-of-game explanations via `debug.parityTraceId`.

### 6.3 Follow-up Code-mode tasks

The following tasks are out of scope for this document but should be created and executed in Code mode:

1. **Implement the producer:**
   - Implement `buildGameEndExplanation` (or equivalent) in a shared module.
   - Ensure it is called for all game endings (PvP, vs AI, sandbox).

2. **Wire into game-end handling:**
   - Attach `GameEndExplanation` to server-side game-end payloads sent to clients.
   - Ensure sandbox engine tests and parity fixtures can assert on explanation contents.

3. **Update HUD and VictoryModal:**
   - Refactor [`src/client/components/GameHUD.tsx`](src/client/components/GameHUD.tsx:1) and [`src/client/components/VictoryModal.tsx`](src/client/components/VictoryModal.tsx:1) to consume `GameEndExplanation` instead of re-deriving weird-state explanations.
   - Add specific UI affordances for LPS, structural stalemate, and mini-region endings (using `tiebreakSteps`, `weirdStateContext`, and `teachingLinks`).

4. **Extend teaching flows:**
   - Ensure flows in [`docs/UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:136) and [`src/shared/teaching/teachingScenarios.ts`](src/shared/teaching/teachingScenarios.ts:1) cover ANM/FE, LPS, structural stalemate, and mini-region endings referenced by `teachingLinks`.
   - Wire TeachingOverlay entrypoints from game-end surfaces based on `nextActions`.

5. **Add tests for hard endings:**
   - Add unit / integration tests that construct representative board states for:
     - Standard ring-majority victory without weird states.
     - LPS with ANM/FE involvement.
     - Structural stalemate resolved by the tiebreak ladder.
     - Q23-style mini-region self-elimination territory wins.
   - Assert that `buildGameEndExplanation` produces outputs equivalent to the examples in §5 (up to ids and labels).

6. **Telemetry validation:**
   - Ensure `rulesContextTags`, `weirdStateContext.reasonCodes`, and teaching flows are logged consistently with [`docs/UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:218).
   - Add dashboards or queries that can answer "Which endings most often confuse players?" based on explanation model fields.

When these tasks are implemented, W‑UX‑7 will provide a unified, extensible, and testable backbone for explaining **why** RingRift games end, especially in the structurally complex cases identified as the project's weakest UX area.