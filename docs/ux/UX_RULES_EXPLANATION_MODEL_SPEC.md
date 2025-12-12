# UX Rules Game-End Explanation Model Spec

> **Doc status (2025-12-05): Draft – W‑UX‑7.**
>
> **Role:** Define a single structured game-end explanation payload that can be produced once by the engine/adapter layer and consumed consistently by HUD, VictoryModal, TeachingOverlay, and sandbox tooling.

## 1. Introduction and goals

The weakest-aspect assessment in [`WEAKNESS_AND_HARDEST_PROBLEM_REPORT.md`](WEAKNESS_AND_HARDEST_PROBLEM_REPORT.md:92) identifies **rules UX and onboarding** as RingRift's primary fragility, especially when games end through Active-No-Moves / Forced Elimination (ANM/FE), structural stalemate plateaus, and territory mini-region scenarios.
Players often see caps disappear, turns auto-advance, or the game end abruptly and cannot easily answer the basic question: _“Why did the game end, and why did this player win?”_

Existing surfaces already expose parts of the answer—raw `GameResult` data, weird-state reason codes, rules-UX telemetry contexts, and teaching flows—but they are stitched together ad hoc in [`GameHUD.tsx`](src/client/components/GameHUD.tsx:1), [`VictoryModal.tsx`](src/client/components/VictoryModal.tsx:1), and [`TeachingOverlay.tsx`](src/client/components/TeachingOverlay.tsx:1).
This spec defines a single structured payload, `GameEndExplanation`, that unifies those fragments.

Goals for this model:

- **Consistency across surfaces.** HUD, VictoryModal, sandbox overlays, and future teaching flows should all answer _“why did this game end?”_ from the same structured explanation instead of re-deriving it locally.
- **Traceability to rules semantics.** Every explanation ties back to canonical rules text ([`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:654), [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1350)) and to high-risk concepts catalogued in [`UX_RULES_CONCEPTS_INDEX.md`](docs/UX_RULES_CONCEPTS_INDEX.md:17).
- **Bridge to teaching and telemetry.** Weird-state reason codes, `rules_context` tags, and teaching topics from [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:71), [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:33), and [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:93) are embedded directly, so that explanations, scenarios, and hotspot analysis point at the same concepts.
- **Engine-friendly and extensible.** The model can be produced in one pass from existing engine and adapter data, and extended with new fields without breaking existing consumers.

## 2. Scope and non-goals

### 2.1 In scope

- **Square-8 two-player core use-case**, with fields and enums chosen so they naturally generalise to other board types and player counts.
- Explanations for **all normal and weird endings**, including:
  - Standard ring-elimination and territory-control victories.
  - Last-Player-Standing outcomes, including those reached via long ANM/FE sequences.
  - Structural stalemates resolved by the tiebreak ladder.
  - Territory mini-region self-elimination and related territory-chain cases.
- Representation of both **primary outcome** (who won and by which rule) and **supporting context** (weird-state reasons, tiebreak details, contributing scores).

### 2.2 Out of scope

- Exact user-facing copy strings; those remain in [`UX_RULES_COPY_SPEC.md`](docs/UX_RULES_COPY_SPEC.md:1) and localisation bundles.
- Concrete UI layout or animation for HUD, VictoryModal, or TeachingOverlay; those are component-level design concerns.
- Changes to underlying rules semantics; this spec only organises information derived from canonical rules.
- Multi-game or match-level summaries; `GameEndExplanation` is strictly per-game.

## 3. Data model: `GameEndExplanation`

This section defines the logical shape of the explanation model in pseudo-TypeScript.
Implementations in TypeScript and Python should mirror these shapes but may add host-specific fields as needed.

### 3.1 Top-level enums and helper types

```ts
type GameEndOutcomeType =
  | 'ring_elimination'
  | 'territory_control'
  | 'last_player_standing'
  | 'structural_stalemate'
  | 'resignation'
  | 'timeout'
  | 'abandonment';

type GameEndVictoryReasonCode =
  // High-level, UX-facing reasons aligned with UX_RULES_CONCEPTS_INDEX concept ids.
  | 'victory_ring_majority'
  | 'victory_territory_majority'
  | 'victory_last_player_standing'
  | 'victory_structural_stalemate_tiebreak'
  | 'victory_resignation'
  | 'victory_timeout'
  | 'victory_abandonment';

type GameEndTiebreakKind = 'territory_spaces' | 'eliminated_rings' | 'markers' | 'last_real_action';

interface GameEndPlayerScoreBreakdown {
  playerId: string;
  eliminatedRings: number;
  territorySpaces: number;
  markers: number;
  extra?: Record<string, number>; // board- or mode-specific fields
}

interface GameEndTiebreakStep {
  kind: GameEndTiebreakKind;
  winnerPlayerId: string | null; // null if still tied at this step
  valuesByPlayer: Record<string, number>;
}

// These aliased string types must line up with existing shared enums:
// - RulesWeirdStateReasonCode in src/shared/engine/weirdStateReasons.ts
// - RulesUxContext in src/shared/telemetry/rulesUxEvents.ts
type GameEndRulesWeirdStateReasonCode = string;
type GameEndRulesContext = string;

interface GameEndWeirdStateContext {
  reasonCodes: GameEndRulesWeirdStateReasonCode[];
  primaryReasonCode?: GameEndRulesWeirdStateReasonCode;
  rulesContexts: GameEndRulesContext[];
  teachingTopicIds: string[]; // e.g. ['teaching.active_no_moves']
}

interface GameEndRulesReference {
  rulesSpecAnchor?: string; // e.g. 'RR-CANON R170'
  rulesDocsLinks?: {
    path: string; // e.g. 'RULES_CANONICAL_SPEC.md'
    anchor?: string; // e.g. '#170-ring-elimination-victory'
  }[];
}

interface GameEndTeachingLink {
  teachingTopics: string[]; // e.g. ['anm_fe_core', 'structural_stalemate']
  recommendedFlows?: string[]; // e.g. ['fe_loop_intro', 'structural_stalemate_intro']
}

interface GameEndUxCopyKeys {
  shortSummaryKey: string; // HUD / compact text lookup key
  detailedSummaryKey: string; // VictoryModal / overlay lookup key
}

interface GameEndDebugInfo {
  rawGameResultReason?: string;
  rawEngineOutcome?: unknown;
  notes?: string;
}
```

### 3.2 `GameEndExplanation` shape

```ts
interface GameEndExplanation {
  // 1) Top-level metadata
  gameId?: string;
  boardType: 'square8' | 'square19' | 'hexagonal';
  numPlayers: number;
  isRanked?: boolean;

  winnerPlayerId: string | null; // null for true draws (currently unreachable)
  outcomeType: GameEndOutcomeType;
  victoryReasonCode: GameEndVictoryReasonCode;
  primaryConceptId?: string; // e.g. 'anm_fe_core', 'structural_stalemate'

  // 2) Victory and scoring detail
  scoreBreakdown?: GameEndPlayerScoreBreakdown[];
  tiebreakSteps?: GameEndTiebreakStep[]; // populated for structural stalemate and other ladders

  // 3) Weird-state and rules context
  weirdStateContext?: GameEndWeirdStateContext;
  rulesContextTags?: GameEndRulesContext[];

  // 4) Rules references
  rulesReferences?: GameEndRulesReference;

  // 5) Teaching linkage
  teaching?: GameEndTeachingLink;

  // 6) UX-oriented copy keys
  uxCopy: GameEndUxCopyKeys;

  // 7) Telemetry hints (optional, but recommended)
  telemetry?: {
    rulesContext: GameEndRulesContext;
    weirdStateReasonCode?: GameEndRulesWeirdStateReasonCode;
    additionalRulesContexts?: GameEndRulesContext[];
  };

  // 8) Developer-only debugging information
  debug?: GameEndDebugInfo;
}
```

At runtime, the engine/adapter is responsible for ensuring that:

- `outcomeType` and `victoryReasonCode` are coherent with the underlying `GameResult`.
- `rulesContextTags` and `telemetry.rulesContext` are drawn from the same vocabulary as [`RulesContext`](docs/UX_RULES_TELEMETRY_SPEC.md:93).
- `weirdStateContext.reasonCodes` are valid [`RulesWeirdStateReasonCode`](src/shared/engine/weirdStateReasons.ts:8) values.
- `teaching.teachingTopics` and `teaching.recommendedFlows` match ids defined in [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:135) and [`teachingScenarios.ts`](src/shared/teaching/teachingScenarios.ts:51).

## 4. Field origins and mapping from existing sources

This section sketches where each part of `GameEndExplanation` should be populated from, without prescribing concrete implementation code.

### 4.1 Engine / rules layer

- **Source:** shared TypeScript engine under [`src/shared/engine`](src/shared/engine/index.ts:1), mirrored in Python.
- **Fields populated:**
  - `boardType`, `numPlayers`, `isRanked` (where available).
  - `winnerPlayerId`, `outcomeType`, and raw `GameResult.reason` (e.g. `ring_elimination`, `territory_control`, `game_completed`, `last_player_standing`, `resignation`, `timeout`, `abandonment`).
  - `scoreBreakdown` per player (rings eliminated, territory spaces, markers).
  - `tiebreakSteps` for structural stalemate (territory → eliminated rings → markers → last actor) as defined in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:682).
  - Any parity or termination notes used in `debug`.

### 4.2 Weird-state mapping

- **Source:** [`weirdStateReasons.ts`](src/shared/engine/weirdStateReasons.ts:8) and [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:71).
- **Fields populated:**
  - `weirdStateContext.reasonCodes` and `weirdStateContext.primaryReasonCode` from helpers such as `getWeirdStateReasonForType` / `getWeirdStateReasonForGameResult`.
  - `weirdStateContext.rulesContexts` by mapping each reason code through `getRulesContextForReason`.
  - `rulesContextTags` unioned from these contexts.
  - `telemetry.weirdStateReasonCode` and `telemetry.rulesContext` for low-cardinality logging.
  - `weirdStateContext.teachingTopicIds` using `getTeachingTopicForReason` and the TeachingOverlay mapping in [`TeachingOverlay.tsx`](src/client/components/TeachingOverlay.tsx:171).

### 4.3 Rules docs and concepts index

- **Source:** [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:654), [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1350), and the concept mapping in [`UX_RULES_CONCEPTS_INDEX.md`](docs/UX_RULES_CONCEPTS_INDEX.md:17).
- **Fields populated:**
  - `rulesReferences.rulesSpecAnchor` for the primary rule governing the ending (e.g. `RR-CANON R170`, `RR-CANON R171`, `RR-CANON R172`, `RR-CANON R173`).
  - `rulesReferences.rulesDocsLinks` pointing into canonical docs and FAQ anchors (e.g. FAQ Q23 for mini-regions).
  - `primaryConceptId` set to the most important concept id from the index (e.g. `anm_fe_core`, `structural_stalemate`, `territory_mini_regions`, `lps_real_actions`).

### 4.4 Teaching metadata

- **Source:** [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:135) and [`teachingScenarios.ts`](src/shared/teaching/teachingScenarios.ts:51).
- **Fields populated:**
  - `teaching.teachingTopics` with relevant high-level topics (e.g. `anm_forced_elimination`, `structural_stalemate`, `territory_mini_region`).
  - `teaching.recommendedFlows` with flow ids such as `fe_loop_intro`, `structural_stalemate_intro`, `mini_region_intro`, `last_player_standing_intro`.

### 4.5 UX copy and telemetry

- **Source:** [`UX_RULES_COPY_SPEC.md`](docs/UX_RULES_COPY_SPEC.md:54), [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:93).
- **Fields populated:**
  - `uxCopy.shortSummaryKey` for HUD and compact banners (e.g. `game_end.ring_elimination.short`, `game_end.structural_stalemate.short`).
  - `uxCopy.detailedSummaryKey` for VictoryModal and TeachingOverlay (e.g. `game_end.lps.detailed`).
  - `telemetry.rulesContext` as the single best `RulesContext` for this ending (usually matching `primaryConceptId`).

## 5. Worked examples

This section gives non-normative examples of `GameEndExplanation` instances in pseudo-JSON.

### 5.1 Standard ring-majority win (no weird states)

```json
{
  "gameId": "g_square8_2p_001",
  "boardType": "square8",
  "numPlayers": 2,
  "isRanked": true,
  "winnerPlayerId": "P1",
  "outcomeType": "ring_elimination",
  "victoryReasonCode": "victory_ring_majority",
  "primaryConceptId": "victory_ring_elimination",
  "scoreBreakdown": [
    { "playerId": "P1", "eliminatedRings": 19, "territorySpaces": 4, "markers": 12 },
    { "playerId": "P2", "eliminatedRings": 6, "territorySpaces": 2, "markers": 10 }
  ],
  "uxCopy": {
    "shortSummaryKey": "game_end.ring_elimination.short",
    "detailedSummaryKey": "game_end.ring_elimination.long"
  },
  "rulesReferences": {
    "rulesSpecAnchor": "RR-CANON R170",
    "rulesDocsLinks": [
      { "path": "RULES_CANONICAL_SPEC.md", "anchor": "#170-ring-elimination-victory" },
      {
        "path": "ringrift_complete_rules.md",
        "anchor": "#131-ring-elimination-victory-primary-victory-path"
      }
    ]
  },
  "telemetry": {
    "rulesContext": "victory_ring_elimination"
  }
}
```

### 5.2 Last Player Standing with ANM/FE involvement

```json
{
  "gameId": "g_square8_3p_042",
  "boardType": "square8",
  "numPlayers": 3,
  "winnerPlayerId": "P2",
  "outcomeType": "last_player_standing",
  "victoryReasonCode": "victory_last_player_standing",
  "primaryConceptId": "lps_real_actions",
  "scoreBreakdown": [
    { "playerId": "P1", "eliminatedRings": 8, "territorySpaces": 0, "markers": 5 },
    { "playerId": "P2", "eliminatedRings": 10, "territorySpaces": 3, "markers": 7 },
    { "playerId": "P3", "eliminatedRings": 4, "territorySpaces": 0, "markers": 2 }
  ],
  "weirdStateContext": {
    "reasonCodes": [
      "ANM_MOVEMENT_FE_BLOCKED",
      "FE_SEQUENCE_CURRENT_PLAYER",
      "LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS"
    ],
    "primaryReasonCode": "LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS",
    "rulesContexts": ["anm_forced_elimination", "last_player_standing"],
    "teachingTopicIds": [
      "teaching.active_no_moves",
      "teaching.forced_elimination",
      "teaching.victory_stalemate"
    ]
  },
  "teaching": {
    "teachingTopics": ["anm_forced_elimination", "last_player_standing"],
    "recommendedFlows": ["fe_loop_intro", "last_player_standing_intro"]
  },
  "uxCopy": {
    "shortSummaryKey": "game_end.lps.short",
    "detailedSummaryKey": "game_end.lps.with_anm_fe.detailed"
  },
  "rulesReferences": {
    "rulesSpecAnchor": "RR-CANON R172",
    "rulesDocsLinks": [
      { "path": "RULES_CANONICAL_SPEC.md", "anchor": "#172-last-player-standing-victory" },
      { "path": "ringrift_complete_rules.md", "anchor": "#133-last-player-standing" }
    ]
  },
  "telemetry": {
    "rulesContext": "last_player_standing",
    "weirdStateReasonCode": "LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS",
    "additionalRulesContexts": ["anm_forced_elimination"]
  }
}
```

### 5.3 Structural stalemate resolved by the tiebreak ladder

```json
{
  "gameId": "g_square19_3p_317",
  "boardType": "square19",
  "numPlayers": 3,
  "winnerPlayerId": "P3",
  "outcomeType": "structural_stalemate",
  "victoryReasonCode": "victory_structural_stalemate_tiebreak",
  "primaryConceptId": "structural_stalemate",
  "scoreBreakdown": [
    { "playerId": "P1", "eliminatedRings": 30, "territorySpaces": 70, "markers": 5 },
    { "playerId": "P2", "eliminatedRings": 28, "territorySpaces": 65, "markers": 4 },
    { "playerId": "P3", "eliminatedRings": 29, "territorySpaces": 72, "markers": 6 }
  ],
  "tiebreakSteps": [
    {
      "kind": "territory_spaces",
      "winnerPlayerId": "P3",
      "valuesByPlayer": { "P1": 70, "P2": 65, "P3": 72 }
    }
  ],
  "weirdStateContext": {
    "reasonCodes": ["STRUCTURAL_STALEMATE_TIEBREAK"],
    "primaryReasonCode": "STRUCTURAL_STALEMATE_TIEBREAK",
    "rulesContexts": ["structural_stalemate"],
    "teachingTopicIds": ["teaching.victory_stalemate"]
  },
  "teaching": {
    "teachingTopics": ["structural_stalemate"],
    "recommendedFlows": ["structural_stalemate_intro"]
  },
  "uxCopy": {
    "shortSummaryKey": "game_end.structural_stalemate.short",
    "detailedSummaryKey": "game_end.structural_stalemate.tiebreak.detailed"
  },
  "rulesReferences": {
    "rulesSpecAnchor": "RR-CANON R173",
    "rulesDocsLinks": [
      { "path": "RULES_CANONICAL_SPEC.md", "anchor": "#173-global-stalemate-and-tiebreaks" },
      { "path": "ringrift_complete_rules.md", "anchor": "#134-end-of-game-stalemate-resolution" }
    ]
  },
  "telemetry": {
    "rulesContext": "structural_stalemate",
    "weirdStateReasonCode": "STRUCTURAL_STALEMATE_TIEBREAK"
  }
}
```

### 5.4 Territory mini-region self-elimination (Q23-style case)

```json
{
  "gameId": "g_square8_2p_mr_q23",
  "boardType": "square8",
  "numPlayers": 2,
  "winnerPlayerId": "P1",
  "outcomeType": "territory_control",
  "victoryReasonCode": "victory_territory_majority",
  "primaryConceptId": "territory_mini_regions",
  "scoreBreakdown": [
    { "playerId": "P1", "eliminatedRings": 14, "territorySpaces": 35, "markers": 3 },
    { "playerId": "P2", "eliminatedRings": 9, "territorySpaces": 10, "markers": 1 }
  ],
  "weirdStateContext": {
    "reasonCodes": ["ANM_TERRITORY_NO_ACTIONS"],
    "primaryReasonCode": "ANM_TERRITORY_NO_ACTIONS",
    "rulesContexts": ["territory_mini_region"],
    "teachingTopicIds": ["teaching.territory"]
  },
  "teaching": {
    "teachingTopics": ["territory_mini_region"],
    "recommendedFlows": ["mini_region_intro"]
  },
  "uxCopy": {
    "shortSummaryKey": "game_end.territory.mini_region.short",
    "detailedSummaryKey": "game_end.territory.mini_region.detailed"
  },
  "rulesReferences": {
    "rulesSpecAnchor": "RR-CANON R141–R145",
    "rulesDocsLinks": [
      { "path": "RULES_CANONICAL_SPEC.md", "anchor": "#141-physical-disconnection-criterion" },
      {
        "path": "ringrift_complete_rules.md",
        "anchor": "#q23-what-happens-if-i-cannot-eliminate-any-rings-when-processing-a-disconnected-region"
      }
    ]
  },
  "telemetry": {
    "rulesContext": "territory_mini_region",
    "weirdStateReasonCode": "ANM_TERRITORY_NO_ACTIONS"
  }
}
```

## 6. Integration notes and follow-up tasks

### 6.1 Likely producer

- Introduce a shared builder function, e.g. `buildGameEndExplanation(gameResult, gameState, options)`, in a shared module under `src/shared/engine` or a thin adapter layer.
- The builder should be invoked exactly once per completed game (backend host, sandbox host, and AI self-play harnesses as needed) and persisted or attached to the final game summary payload.

### 6.2 Known consumers

- **HUD**: use `GameEndExplanation` to drive end-of-game banners and compact score summaries instead of reinterpreting `GameResult`.
- **VictoryModal**: use `scoreBreakdown`, `tiebreakSteps`, and `uxCopy` for the header and explanation text, and `rulesReferences` / `teaching` for the “What happened?” affordance.
- **TeachingOverlay**: when opened from game-end contexts, prefer the `teaching` and `weirdStateContext` fields to choose topics and flows.
- **Sandbox / replay tools**: surface `GameEndExplanation` alongside board snapshots to make parity debugging and teaching easier.

### 6.3 Follow-up implementation tasks (Code mode)

- Implement `buildGameEndExplanation()` in a shared module, wired to existing `GameResult`, weird-state helpers, and telemetry types.
- Update backend game-end handling to attach `GameEndExplanation` to the server payload consumed by clients and replay tools.
- Update [`GameHUD.tsx`](src/client/components/GameHUD.tsx:980) and [`VictoryModal.tsx`](src/client/components/VictoryModal.tsx:379) to render explanations, scores, and teaching links from `GameEndExplanation` rather than bespoke logic.
- Extend tests covering ANM/FE, LPS, structural stalemate, and mini-region cases to assert that the produced `GameEndExplanation` matches scenarios described in this spec.
- Optionally, add dev tooling (e.g. debug overlays or logs) that dump `GameEndExplanation.debug` for tricky endings.

This spec is the authoritative contract for the `GameEndExplanation` model; any future changes to fields or semantics should be reflected here before Code-mode implementation is updated.
