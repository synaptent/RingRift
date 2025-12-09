/**
 * Teaching topic type definitions and content metadata.
 *
 * This module provides typed definitions for teaching topics, teaching tips,
 * and the relationship between topics and rules concepts for the RingRift
 * teaching infrastructure.
 *
 * @see docs/UX_RULES_TEACHING_SCENARIOS.md for the authoritative teaching flow spec
 * @see docs/UX_RULES_TEACHING_GAP_ANALYSIS.md for gaps addressed by these definitions
 */

import type { RulesConcept } from './teachingScenarios';

/**
 * Canonical teaching topic identifiers used by TeachingOverlay and related
 * surfaces. These map to the topics defined in TeachingOverlay.tsx.
 */
export type TeachingTopicId =
  // Core mechanics
  | 'ring_placement'
  | 'stack_movement'
  | 'capturing'
  | 'chain_capture'
  | 'line_bonus'
  | 'line_territory_order'
  | 'territory'
  // ANM / FE topics
  | 'active_no_moves'
  | 'forced_elimination'
  // Victory condition topics
  | 'victory_elimination'
  | 'victory_territory'
  | 'victory_stalemate';

/**
 * Teaching tip with optional metadata for categorization and display.
 */
export interface TeachingTip {
  /** The tip text content */
  text: string;
  /** Optional category for grouping tips (e.g., 'lps', 'stalemate', 'tiebreak') */
  category?: string;
  /** Optional emphasis level for UI rendering */
  emphasis?: 'normal' | 'important' | 'critical';
  /** Optional gap ID this tip addresses (for traceability) */
  addressesGap?: string;
}

/**
 * Enhanced teaching topic content with structured tips.
 */
export interface EnhancedTeachingContent {
  id: TeachingTopicId;
  title: string;
  icon: string;
  description: string;
  tips: TeachingTip[];
  relatedConcepts?: RulesConcept[];
}

/**
 * Teaching tips specifically for the victory_stalemate topic, addressing:
 * - GAP-LPS-02: Explicit LPS sub-section
 * - GAP-LPS-03: "FE ≠ real action" emphasis
 * - GAP-STALE-04: ANM vs global stalemate distinction
 * - GAP-STALE-01: Tiebreak ladder explanation
 */
export const VICTORY_STALEMATE_TIPS: TeachingTip[] = [
  // === Last Player Standing (LPS) ===
  {
    text: 'LAST PLAYER STANDING: You win if you are the only player who can make real moves (placements, movements, or captures) for THREE consecutive complete rounds.',
    category: 'lps',
    emphasis: 'important',
    addressesGap: 'GAP-LPS-02',
  },
  // Three-round requirement emphasis
  {
    text: 'LPS requires THREE rounds: First round, you must have and take at least one real action while all others have none. Second and third rounds, you remain the only player with real actions. Victory is declared after the third round completes.',
    category: 'lps',
    emphasis: 'critical',
    addressesGap: 'GAP-LPS-02',
  },
  // GAP-LPS-03: Emphasize FE ≠ real action
  {
    text: 'Forced elimination is NOT a real action: even if you are forced to eliminate caps, that does not count as a move for LPS purposes. If your opponent has real moves and you only have forced eliminations, they have not lost yet.',
    category: 'lps',
    emphasis: 'critical',
    addressesGap: 'GAP-LPS-03',
  },
  // === Structural Stalemate ===
  {
    text: 'STRUCTURAL STALEMATE: This happens when NO player has ANY real moves or forced eliminations available – the game is truly stuck.',
    category: 'stalemate',
    emphasis: 'important',
    addressesGap: 'GAP-STALE-04',
  },
  // GAP-STALE-04: Distinction between single-player ANM and global stalemate
  {
    text: 'ANM vs Stalemate: When only YOU have no moves, the game continues – other players can still play. A structural stalemate only occurs when NOBODY can move at all.',
    category: 'stalemate',
    emphasis: 'important',
    addressesGap: 'GAP-STALE-04',
  },
  // GAP-STALE-01: Tiebreak ladder
  {
    text: 'TIEBREAK LADDER: In a stalemate, the winner is determined by: 1) Territory spaces, 2) Eliminated rings (including rings in hand), 3) Markers on board, 4) Who made the last real action.',
    category: 'tiebreak',
    emphasis: 'normal',
    addressesGap: 'GAP-STALE-01',
  },
];

/**
 * Teaching tips for forced elimination topic, addressing GAP-LPS-03.
 */
export const FORCED_ELIMINATION_TIPS: TeachingTip[] = [
  {
    text: 'Rings removed by forced elimination are permanently eliminated and count toward global Ring Elimination victory, just like eliminations from movement onto markers, line rewards, or territory processing.',
    category: 'mechanics',
    emphasis: 'normal',
  },
  {
    text: 'Forced elimination does NOT count as a "real move" for Last Player Standing, even though each step is recorded as a forced_elimination move in its own phase.',
    category: 'lps',
    emphasis: 'critical',
    addressesGap: 'GAP-LPS-03',
  },
  {
    text: 'You cannot skip forced elimination when its conditions are met; the rules may let you choose the stack, but some legal forced_elimination move must be recorded.',
    category: 'mechanics',
    emphasis: 'normal',
  },
];

/**
 * Teaching tips for active_no_moves topic.
 * GAP-ANM-01: First-occurrence teaching trigger for ANM/FE
 * GAP-ANM-03: Recovery guidance for ANM situations
 */
export const ACTIVE_NO_MOVES_TIPS: TeachingTip[] = [
  {
    text: 'Active–No–Moves only looks at real moves: placements, movements, and captures. Forced elimination and automatic line/territory processing do not count as real moves for Last Player Standing.',
    category: 'anm',
    emphasis: 'important',
  },
  {
    text: 'If you still control stacks but have no placements or movements, the game applies forced elimination caps until a real move becomes available or your stacks are exhausted.',
    category: 'anm',
    emphasis: 'normal',
  },
  {
    text: 'On some boards a full plateau can occur where no player has real moves or forced eliminations; in that case the game ends and the final score comes from territory and eliminated rings, not further play.',
    category: 'plateau',
    emphasis: 'normal',
  },
  // GAP-ANM-01: First-occurrence context
  {
    text: 'FIRST TIME SEEING THIS? When you have no legal moves, you enter an "Active-No-Moves" state. This is different from being eliminated – you are still in the game!',
    category: 'first_occurrence',
    emphasis: 'important',
    addressesGap: 'GAP-ANM-01',
  },
  // GAP-ANM-03: Recovery guidance
  {
    text: 'HOW TO RECOVER FROM ANM: Your opponents might open up movement options for you by moving their stacks, collapsing lines, or processing territories. Stay alert – you can become active again!',
    category: 'recovery',
    emphasis: 'important',
    addressesGap: 'GAP-ANM-03',
  },
  {
    text: 'While blocked, forced elimination gradually removes your ring caps. If moves open up before you lose all caps, you can resume normal play.',
    category: 'recovery',
    emphasis: 'normal',
    addressesGap: 'GAP-ANM-03',
  },
];

/**
 * Teaching tips for chain_capture topic.
 * GAP-CHAIN-04: Strengthened "MUST continue" wording
 */
export const CHAIN_CAPTURE_TIPS: TeachingTip[] = [
  {
    text: 'Starting a capture is OPTIONAL – you can choose to move without capturing. But once you make ANY capture, you MUST continue the chain until no legal captures remain.',
    category: 'mandatory',
    emphasis: 'critical',
    addressesGap: 'GAP-CHAIN-04',
  },
  {
    text: 'You CANNOT stop a chain capture early. Plan your first capture carefully – if it leads to an unfavorable chain, you must follow through.',
    category: 'mandatory',
    emphasis: 'critical',
    addressesGap: 'GAP-CHAIN-04',
  },
  {
    text: 'When multiple capture targets are available, YOU choose which direction to continue. The mandatory rule is about continuation, not direction.',
    category: 'branch',
    emphasis: 'normal',
  },
  {
    text: 'Chain captures can include 180° reversals – you may jump back the way you came if the position allows it.',
    category: 'advanced',
    emphasis: 'normal',
  },
  {
    text: 'Chains can visit the same stack multiple times in different directions, creating complex capture patterns. Each segment must be a legal capture.',
    category: 'advanced',
    emphasis: 'normal',
  },
];

/**
 * Teaching tips for line_territory_order topic.
 * GAP-LINE-01: Line formation triggers
 * GAP-LINE-02: Line vs territory processing order
 * GAP-LINE-03: Multi-line scenarios
 */
export const LINE_TERRITORY_ORDER_TIPS: TeachingTip[] = [
  // GAP-LINE-01: When lines form
  {
    text: 'WHEN LINES FORM: A line forms when 5+ same-colored markers align orthogonally (horizontally, vertically, or diagonally). On hex boards, you need 6+ markers.',
    category: 'formation',
    emphasis: 'important',
    addressesGap: 'GAP-LINE-01',
  },
  // GAP-LINE-02: Processing order
  {
    text: 'PROCESSING ORDER: Lines are ALWAYS processed BEFORE territories. When a move creates both, lines collapse first, then territory regions are evaluated.',
    category: 'order',
    emphasis: 'critical',
    addressesGap: 'GAP-LINE-02',
  },
  {
    text: 'A single turn can trigger: movement → capture chain → line processing → territory processing. Each phase must complete before the next begins.',
    category: 'order',
    emphasis: 'normal',
    addressesGap: 'GAP-LINE-02',
  },
  // GAP-LINE-03: Multi-line scenarios
  {
    text: 'MULTIPLE LINES: If multiple lines form, each is processed separately. You may need to make choices for each overlength line.',
    category: 'multi_line',
    emphasis: 'normal',
    addressesGap: 'GAP-LINE-03',
  },
  {
    text: 'OPTION 1 vs OPTION 2: For overlength lines (6+ on square, 7+ on hex), you choose: collapse ALL markers (costs a ring) or collapse MINIMUM length (free but less territory).',
    category: 'options',
    emphasis: 'important',
  },
  {
    text: 'Line collapse affects what territory regions form. Choosing Option 1 (full collapse) may create more territory but costs a ring; Option 2 (minimum) is safely free.',
    category: 'strategy',
    emphasis: 'normal',
  },
];

/**
 * Teaching tips for territory topic.
 * GAP-TERR-03: Self-elimination eligibility indicator
 * GAP-TERR-04: Q23 "why did I lose my own ring?" explanation
 */
export const TERRITORY_TIPS: TeachingTip[] = [
  {
    text: 'Territory processing happens AFTER line processing. Lines collapse first, then disconnected regions are evaluated.',
    category: 'order',
    emphasis: 'normal',
  },
  // GAP-TERR-04: Self-elimination cost explanation
  {
    text: 'WHY DID I LOSE MY OWN RING? Processing a disconnected region eliminates all interior rings (scoring for you), but you MUST also eliminate one cap from a stack OUTSIDE the region.',
    category: 'self_elimination',
    emphasis: 'critical',
    addressesGap: 'GAP-TERR-04',
  },
  {
    text: 'The self-elimination cost ensures territory is not free. You need at least one stack outside the region with caps to pay this cost.',
    category: 'self_elimination',
    emphasis: 'normal',
    addressesGap: 'GAP-TERR-04',
  },
  // GAP-TERR-03: Eligibility indicator
  {
    text: "CAN'T PROCESS A REGION? You must have a stack OUTSIDE the pending region to pay the elimination cost. If all your stacks are inside or on the border, you cannot process.",
    category: 'eligibility',
    emphasis: 'important',
    addressesGap: 'GAP-TERR-03',
  },
  {
    text: 'Border stacks (on region boundary) become markers when the region is processed. They cannot pay the self-elimination cost.',
    category: 'mechanics',
    emphasis: 'normal',
  },
];

/**
 * Teaching tips for LPS first-occurrence.
 * GAP-LPS-01: First-occurrence trigger when LPS condition starts forming
 */
export const LPS_FIRST_OCCURRENCE_TIPS: TeachingTip[] = [
  {
    text: 'LAST PLAYER STANDING ALERT: When you are the only player with real moves for a full round, the LPS countdown begins. Two more rounds of exclusive actions wins the game!',
    category: 'first_occurrence',
    emphasis: 'critical',
    addressesGap: 'GAP-LPS-01',
  },
  {
    text: 'Watch the LPS indicator in the HUD – it shows when the three-round countdown is active and how close you are to victory (or defeat).',
    category: 'indicator',
    emphasis: 'important',
    addressesGap: 'GAP-LPS-01',
  },
  {
    text: 'If you regain real moves before the third round completes, the LPS countdown resets. Fight back by opening up movement options!',
    category: 'recovery',
    emphasis: 'normal',
    addressesGap: 'GAP-LPS-01',
  },
];

/**
 * Map from teaching topic ID to the rules concepts it relates to.
 * Used to surface related teaching scenarios when a topic is opened.
 */
export const TOPIC_TO_CONCEPTS: Partial<Record<TeachingTopicId, RulesConcept[]>> = {
  active_no_moves: ['anm_forced_elimination'],
  forced_elimination: ['anm_forced_elimination'],
  territory: ['territory_mini_region'],
  chain_capture: ['capture_chain_mandatory'],
  line_bonus: ['line_vs_territory_multi_phase'],
  line_territory_order: ['line_vs_territory_multi_phase'],
  victory_stalemate: ['structural_stalemate', 'last_player_standing'],
};

/**
 * Get the rules concepts related to a teaching topic.
 */
export function getConceptsForTopic(topicId: TeachingTopicId): RulesConcept[] {
  return TOPIC_TO_CONCEPTS[topicId] ?? [];
}
