import type { BoardType } from '../types/game';
import type { RulesWeirdStateReasonCode } from '../engine/weirdStateReasons';
import type { RulesUxContext } from '../telemetry/rulesUxEvents';

/**
 * Shared metadata types for rules teaching scenarios.
 *
 * This mirrors the abstract shape defined in docs/UX_RULES_TEACHING_SCENARIOS.md §2
 * and is consumed by both client hosts (TeachingOverlay, sandbox) and tests.
 */
export type RulesConcept =
  | 'anm_forced_elimination'
  | 'territory_mini_region'
  | 'territory_multi_region_budget'
  | 'line_vs_territory_multi_phase'
  | 'capture_chain_mandatory'
  | 'landing_on_own_marker'
  | 'structural_stalemate'
  | 'last_player_standing';

export type TeachingStepKind = 'guided' | 'interactive';

export interface TeachingScenarioMetadata {
  scenarioId: string;
  rulesConcept: RulesConcept;
  flowId: string;
  stepIndex: number;
  stepKind: TeachingStepKind;

  rulesDocAnchor?: string;
  uxWeirdStateReasonCode?: RulesWeirdStateReasonCode;
  telemetryRulesContext?: RulesUxContext;

  recommendedBoardType: BoardType;
  recommendedNumPlayers: 2 | 3 | 4;
  showInTeachingOverlay: boolean;
  showInSandboxPresets: boolean;
  showInTutorialCarousel: boolean;

  learningObjectiveShort: string;
  difficultyTag?: 'intro' | 'intermediate' | 'advanced';
}

/**
 * Minimal initial flow: Forced Elimination loop & Active–No–Moves (fe_loop_intro).
 *
 * Board positions for these scenarios are defined separately in curated scenario
 * JSON or fixtures. For now the scenarioId fields serve as stable ids that can be
 * wired to concrete boards in a later Code-mode pass.
 */
export const TEACHING_SCENARIOS: readonly TeachingScenarioMetadata[] = [
  // ============================================================
  // Flow: fe_loop_intro – Active-No-Moves / Forced Elimination
  // ============================================================
  {
    scenarioId: 'teaching.fe_loop.step_1',
    rulesConcept: 'anm_forced_elimination',
    flowId: 'fe_loop_intro',
    stepIndex: 1,
    stepKind: 'guided',
    rulesDocAnchor: 'ringrift_complete_rules.md#forced-elimination-when-blocked',
    uxWeirdStateReasonCode: 'ANM_MOVEMENT_FE_BLOCKED',
    telemetryRulesContext: 'anm_forced_elimination',
    recommendedBoardType: 'square8',
    recommendedNumPlayers: 2,
    showInTeachingOverlay: true,
    showInSandboxPresets: true,
    showInTutorialCarousel: true,
    learningObjectiveShort:
      'Recognise when you have no legal placements, movements, or captures and forced elimination will apply.',
    difficultyTag: 'intro',
  },
  {
    scenarioId: 'teaching.fe_loop.step_2',
    rulesConcept: 'anm_forced_elimination',
    flowId: 'fe_loop_intro',
    stepIndex: 2,
    stepKind: 'interactive',
    rulesDocAnchor: 'ringrift_complete_rules.md#forced-elimination-when-blocked',
    uxWeirdStateReasonCode: 'ANM_MOVEMENT_FE_BLOCKED',
    telemetryRulesContext: 'anm_forced_elimination',
    recommendedBoardType: 'square8',
    recommendedNumPlayers: 2,
    showInTeachingOverlay: true,
    showInSandboxPresets: true,
    showInTutorialCarousel: true,
    learningObjectiveShort: 'Execute a forced elimination when no real move exists.',
    difficultyTag: 'intro',
  },
  {
    scenarioId: 'teaching.fe_loop.step_3',
    rulesConcept: 'anm_forced_elimination',
    flowId: 'fe_loop_intro',
    stepIndex: 3,
    stepKind: 'interactive',
    rulesDocAnchor: 'ringrift_complete_rules.md#forced-elimination-when-blocked',
    uxWeirdStateReasonCode: 'FE_SEQUENCE_CURRENT_PLAYER',
    telemetryRulesContext: 'anm_forced_elimination',
    recommendedBoardType: 'square8',
    recommendedNumPlayers: 2,
    showInTeachingOverlay: true,
    showInSandboxPresets: false,
    showInTutorialCarousel: false,
    learningObjectiveShort:
      'See how repeated forced elimination over multiple turns can shrink your stacks toward plateau or Last Player Standing.',
    difficultyTag: 'intermediate',
  },
  {
    scenarioId: 'teaching.fe_loop.step_4',
    rulesConcept: 'anm_forced_elimination',
    flowId: 'fe_loop_intro',
    stepIndex: 4,
    stepKind: 'interactive',
    rulesDocAnchor: 'ringrift_complete_rules.md#forced-elimination-when-blocked',
    uxWeirdStateReasonCode: 'FE_SEQUENCE_CURRENT_PLAYER',
    telemetryRulesContext: 'anm_forced_elimination',
    recommendedBoardType: 'square8',
    recommendedNumPlayers: 2,
    showInTeachingOverlay: true,
    showInSandboxPresets: false,
    showInTutorialCarousel: false,
    learningObjectiveShort:
      'Follow a multi-turn forced-elimination loop and see how repeated cap removals can lead into structural stalemate or Last Player Standing.',
    difficultyTag: 'intermediate',
  },

  // ============================================================
  // Flow: structural_stalemate_intro – Stalemate & Tiebreak Ladder
  // GAP-STALE-01: Added step 2 for tiebreak ladder explanation
  // ============================================================
  {
    scenarioId: 'teaching.structural_stalemate.step_1',
    rulesConcept: 'structural_stalemate',
    flowId: 'structural_stalemate_intro',
    stepIndex: 1,
    stepKind: 'guided',
    rulesDocAnchor: 'ringrift_complete_rules.md#end-of-game-stalemate-resolution',
    uxWeirdStateReasonCode: 'STRUCTURAL_STALEMATE_TIEBREAK',
    telemetryRulesContext: 'structural_stalemate',
    recommendedBoardType: 'square8',
    recommendedNumPlayers: 2,
    showInTeachingOverlay: true,
    showInSandboxPresets: true,
    showInTutorialCarousel: true,
    learningObjectiveShort:
      'Recognise when no moves or forced eliminations remain for any player – this is structural stalemate, different from one player having no moves.',
    difficultyTag: 'intro',
  },
  {
    scenarioId: 'teaching.structural_stalemate.step_2',
    rulesConcept: 'structural_stalemate',
    flowId: 'structural_stalemate_intro',
    stepIndex: 2,
    stepKind: 'guided',
    rulesDocAnchor: 'ringrift_complete_rules.md#end-of-game-stalemate-resolution',
    uxWeirdStateReasonCode: 'STRUCTURAL_STALEMATE_TIEBREAK',
    telemetryRulesContext: 'structural_stalemate',
    recommendedBoardType: 'square8',
    recommendedNumPlayers: 2,
    showInTeachingOverlay: true,
    showInSandboxPresets: true,
    showInTutorialCarousel: false,
    learningObjectiveShort:
      'Understand the four-step tiebreak ladder: 1) Territory spaces, 2) Eliminated rings (including rings in hand), 3) Markers, 4) Who made the last real action.',
    difficultyTag: 'intermediate',
  },

  // ============================================================
  // Flow: last_player_standing_intro – LPS Real Actions
  // GAP-LPS-02: Explicit LPS teaching separate from stalemate
  // ============================================================
  {
    scenarioId: 'teaching.lps.step_1',
    rulesConcept: 'last_player_standing',
    flowId: 'last_player_standing_intro',
    stepIndex: 1,
    stepKind: 'guided',
    rulesDocAnchor: 'ringrift_complete_rules.md#last-player-standing',
    uxWeirdStateReasonCode: 'LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS',
    telemetryRulesContext: 'last_player_standing',
    recommendedBoardType: 'square8',
    recommendedNumPlayers: 2,
    showInTeachingOverlay: true,
    showInSandboxPresets: true,
    showInTutorialCarousel: true,
    learningObjectiveShort:
      'Distinguish real actions (placements, movements, captures) from forced elimination and automatic processing – only real actions count for LPS.',
    difficultyTag: 'intro',
  },
  {
    scenarioId: 'teaching.lps.step_2',
    rulesConcept: 'last_player_standing',
    flowId: 'last_player_standing_intro',
    stepIndex: 2,
    stepKind: 'guided',
    rulesDocAnchor: 'ringrift_complete_rules.md#last-player-standing',
    uxWeirdStateReasonCode: 'LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS',
    telemetryRulesContext: 'last_player_standing',
    recommendedBoardType: 'square8',
    recommendedNumPlayers: 3,
    showInTeachingOverlay: true,
    showInSandboxPresets: false,
    showInTutorialCarousel: false,
    learningObjectiveShort:
      'Understand LPS requires THREE consecutive rounds: first round you have and take real actions while others have none, second and third rounds confirm you remain the only active player.',
    difficultyTag: 'intermediate',
  },

  // ============================================================
  // Flow: mini_region_intro – Territory Mini-Regions (Q23)
  // ============================================================
  {
    scenarioId: 'teaching.mini_region.step_1',
    rulesConcept: 'territory_mini_region',
    flowId: 'mini_region_intro',
    stepIndex: 1,
    stepKind: 'guided',
    rulesDocAnchor:
      'ringrift_complete_rules.md#q23-what-happens-if-i-cannot-eliminate-any-rings-when-processing-a-disconnected-region',
    uxWeirdStateReasonCode: 'ANM_TERRITORY_NO_ACTIONS',
    telemetryRulesContext: 'territory_mini_region',
    recommendedBoardType: 'square8',
    recommendedNumPlayers: 2,
    showInTeachingOverlay: true,
    showInSandboxPresets: true,
    showInTutorialCarousel: true,
    learningObjectiveShort:
      'Understand the shape of a disconnected mini-region and why it is eligible for territory processing.',
    difficultyTag: 'intro',
  },
  {
    scenarioId: 'teaching.mini_region.step_2',
    rulesConcept: 'territory_mini_region',
    flowId: 'mini_region_intro',
    stepIndex: 2,
    stepKind: 'interactive',
    rulesDocAnchor:
      'ringrift_complete_rules.md#q23-what-happens-if-i-cannot-eliminate-any-rings-when-processing-a-disconnected-region',
    uxWeirdStateReasonCode: 'ANM_TERRITORY_NO_ACTIONS',
    telemetryRulesContext: 'territory_mini_region',
    recommendedBoardType: 'square8',
    recommendedNumPlayers: 2,
    showInTeachingOverlay: true,
    showInSandboxPresets: true,
    showInTutorialCarousel: false,
    learningObjectiveShort:
      'Trigger region processing and observe how interior rings are eliminated and credited to you, while border markers become territory.',
    difficultyTag: 'intermediate',
  },
  {
    scenarioId: 'teaching.mini_region.step_3',
    rulesConcept: 'territory_mini_region',
    flowId: 'mini_region_intro',
    stepIndex: 3,
    stepKind: 'interactive',
    rulesDocAnchor:
      'ringrift_complete_rules.md#q23-what-happens-if-i-cannot-eliminate-any-rings-when-processing-a-disconnected-region',
    uxWeirdStateReasonCode: 'ANM_TERRITORY_NO_ACTIONS',
    telemetryRulesContext: 'territory_mini_region',
    recommendedBoardType: 'square8',
    recommendedNumPlayers: 2,
    showInTeachingOverlay: true,
    showInSandboxPresets: true,
    showInTutorialCarousel: false,
    learningObjectiveShort:
      'Apply the mandatory self-elimination cost correctly – you must have a stack OUTSIDE the region to pay the elimination cost.',
    difficultyTag: 'advanced',
  },

  // ============================================================
  // Flow: capture_chain_mandatory – Chain Captures
  // GAP-CHAIN-01: Full flow implementation (steps 1-3)
  // GAP-CHAIN-03: Step 3 covers 180° reversals and cyclic patterns
  // ============================================================
  {
    scenarioId: 'teaching.capture_chain.step_1',
    rulesConcept: 'capture_chain_mandatory',
    flowId: 'capture_chain_mandatory',
    stepIndex: 1,
    stepKind: 'guided',
    rulesDocAnchor: 'ringrift_complete_rules.md#chain-captures',
    telemetryRulesContext: 'capture_chain_mandatory',
    recommendedBoardType: 'square8',
    recommendedNumPlayers: 2,
    showInTeachingOverlay: true,
    showInSandboxPresets: true,
    showInTutorialCarousel: true,
    learningObjectiveShort:
      'Starting a capture is optional, but once started, you MUST continue capturing while any legal capture exists.',
    difficultyTag: 'intro',
  },
  {
    scenarioId: 'teaching.capture_chain.step_2',
    rulesConcept: 'capture_chain_mandatory',
    flowId: 'capture_chain_mandatory',
    stepIndex: 2,
    stepKind: 'interactive',
    rulesDocAnchor: 'ringrift_complete_rules.md#chain-captures',
    telemetryRulesContext: 'capture_chain_mandatory',
    recommendedBoardType: 'square8',
    recommendedNumPlayers: 2,
    showInTeachingOverlay: true,
    showInSandboxPresets: true,
    showInTutorialCarousel: false,
    learningObjectiveShort:
      'Perform a short chain capture and choose among multiple continuation directions – you pick the path, but cannot stop early.',
    difficultyTag: 'intermediate',
  },
  {
    scenarioId: 'teaching.capture_chain.step_3',
    rulesConcept: 'capture_chain_mandatory',
    flowId: 'capture_chain_mandatory',
    stepIndex: 3,
    stepKind: 'interactive',
    rulesDocAnchor: 'ringrift_complete_rules.md#chain-captures',
    telemetryRulesContext: 'capture_chain_mandatory',
    recommendedBoardType: 'square8',
    recommendedNumPlayers: 2,
    showInTeachingOverlay: true,
    showInSandboxPresets: true,
    showInTutorialCarousel: false,
    learningObjectiveShort:
      'Experience 180° reversals and revisiting stacks in advanced chain patterns – chains can loop back if positions allow.',
    difficultyTag: 'advanced',
  },

  // ============================================================
  // Flow: line_vs_territory_blocking – Line vs Territory Processing
  // GAP-LINE-01: Line formation triggers
  // GAP-LINE-02: Line vs territory processing order
  // GAP-LINE-03: Multi-line scenarios
  // ============================================================
  {
    scenarioId: 'teaching.line_territory.step_1',
    rulesConcept: 'line_vs_territory_multi_phase',
    flowId: 'line_vs_territory_blocking',
    stepIndex: 1,
    stepKind: 'guided',
    rulesDocAnchor: 'ringrift_complete_rules.md#line-formation-and-processing',
    telemetryRulesContext: 'line_vs_territory_multi_phase',
    recommendedBoardType: 'square8',
    recommendedNumPlayers: 2,
    showInTeachingOverlay: true,
    showInSandboxPresets: true,
    showInTutorialCarousel: true,
    learningObjectiveShort:
      'Understand when lines form: 5+ same-colored markers aligned orthogonally (6+ on hex). Lines are processed BEFORE territories.',
    difficultyTag: 'intro',
  },
  {
    scenarioId: 'teaching.line_territory.step_2',
    rulesConcept: 'line_vs_territory_multi_phase',
    flowId: 'line_vs_territory_blocking',
    stepIndex: 2,
    stepKind: 'interactive',
    rulesDocAnchor: 'ringrift_complete_rules.md#line-formation-and-processing',
    telemetryRulesContext: 'line_vs_territory_multi_phase',
    recommendedBoardType: 'square8',
    recommendedNumPlayers: 2,
    showInTeachingOverlay: true,
    showInSandboxPresets: true,
    showInTutorialCarousel: false,
    learningObjectiveShort:
      'Watch a multi-phase turn: movement → capture → chain → line processing → territory processing. Each phase must complete before the next begins.',
    difficultyTag: 'intermediate',
  },
  {
    scenarioId: 'teaching.line_territory.step_3',
    rulesConcept: 'line_vs_territory_multi_phase',
    flowId: 'line_vs_territory_blocking',
    stepIndex: 3,
    stepKind: 'interactive',
    rulesDocAnchor: 'ringrift_complete_rules.md#line-formation-and-processing',
    telemetryRulesContext: 'line_vs_territory_multi_phase',
    recommendedBoardType: 'square8',
    recommendedNumPlayers: 2,
    showInTeachingOverlay: true,
    showInSandboxPresets: true,
    showInTutorialCarousel: false,
    learningObjectiveShort:
      'Compare Option 1 (full collapse with ring cost) vs Option 2 (minimum collapse, no cost) for overlength lines – understand the strategic tradeoff.',
    difficultyTag: 'advanced',
  },
];
