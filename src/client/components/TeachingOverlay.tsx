import { useState, useEffect, useMemo, useRef } from 'react';
import type { BoardType, GamePhase, Move } from '../../shared/types/game';
import type { RulesUxContext, RulesUxWeirdStateType } from '../../shared/telemetry/rulesUxEvents';
import type { RulesWeirdStateReasonCode } from '../../shared/engine/weirdStateReasons';
import {
  TEACHING_SCENARIOS,
  type RulesConcept,
  type TeachingScenarioMetadata,
} from '../../shared/teaching/teachingScenarios';
import { getRulesUxContextForTeachingScenario } from '../../shared/teaching/scenarioTelemetry';
import {
  logRulesUxEvent,
  newTeachingFlowId,
  TEACHING_TOPICS_COPY,
} from '../utils/rulesUxTelemetry';
import type { FSMDecisionSurfaceViewModel } from '../adapters/gameViewModels';
import { Dialog } from './ui/Dialog';

export type TeachingTopic =
  | 'ring_placement'
  | 'stack_movement'
  | 'capturing'
  | 'chain_capture'
  | 'line_bonus'
  | 'line_territory_order'
  | 'territory'
  | 'active_no_moves'
  | 'forced_elimination'
  | 'recovery_action'
  | 'victory_elimination'
  | 'victory_territory'
  | 'victory_stalemate';

export interface TeachingContent {
  title: string;
  icon: string;
  description: string;
  tips: string[];
  relatedPhases?: GamePhase[];
}

const TEACHING_CONTENT: Record<TeachingTopic, TeachingContent> = {
  ring_placement: {
    title: TEACHING_TOPICS_COPY.ring_placement.heading,
    icon: '◯',
    // Canonical description from UX_RULES_COPY_SPEC.md §2 / placement overview
    description: TEACHING_TOPICS_COPY.ring_placement.body,
    tips: [
      'Placing adjacent to your existing rings helps build territory',
      'Stacking on opponents can set up future captures',
      'The placement phase continues until all players have placed all rings',
    ],
    relatedPhases: ['ring_placement'],
  },
  stack_movement: {
    title: TEACHING_TOPICS_COPY.stack_movement.heading,
    icon: '→',
    // UX_RULES_COPY_SPEC.md §4 – Stack Movement description
    description: TEACHING_TOPICS_COPY.stack_movement.body,
    tips: [
      'Taller stacks can threaten long moves and future captures',
      'You cannot move through other stacks or territory spaces',
      'Landing on a marker can eliminate the top ring of your stack',
    ],
    relatedPhases: ['movement'],
  },
  capturing: {
    title: TEACHING_TOPICS_COPY.capturing.heading,
    icon: '×',
    // UX_RULES_COPY_SPEC.md §5 – Capturing description
    description: TEACHING_TOPICS_COPY.capturing.body,
    tips: [
      'Captures move rings into your stacks; only later eliminations remove rings from the game',
      'Your stack’s cap height must be at least as high as the stack you jump over',
      'Capturing can build powerful stacks but also makes them tempting targets',
    ],
    relatedPhases: ['capture'],
  },
  chain_capture: {
    title: TEACHING_TOPICS_COPY.chain_capture.heading,
    icon: '⇉',
    // UX_RULES_COPY_SPEC.md §5 – Chain Capture description
    description: TEACHING_TOPICS_COPY.chain_capture.body,
    tips: [
      // GAP-CHAIN-04: Strengthened mandatory continuation wording
      'Starting a capture is OPTIONAL – you can choose to move without capturing. But once you make ANY capture, you MUST continue the chain until no legal captures remain.',
      'You CANNOT stop a chain capture early. Plan your first capture carefully – if it leads to an unfavorable chain, you must follow through.',
      'When multiple capture targets are available, YOU choose which direction to continue. The mandatory rule is about continuation, not direction.',
      'Chain captures can include 180° reversals – you may jump back the way you came if the position allows it.',
    ],
    relatedPhases: ['chain_capture'],
  },
  line_bonus: {
    title: TEACHING_TOPICS_COPY.line_bonus.heading,
    icon: '═',
    // UX_RULES_COPY_SPEC.md §6 – Lines description
    description: TEACHING_TOPICS_COPY.line_bonus.body,
    tips: [
      // GAP-LINE-01: When lines form
      'WHEN LINES FORM: A line forms when 5+ same-colored markers align orthogonally (horizontally, vertically, or diagonally). On hex boards, you need 6+ markers.',
      'Lines are formed from markers, not rings – horizontal, vertical, and diagonal lines all count.',
      'LINE ELIMINATION COST: You must eliminate ONE ring from the top of any stack you control (including standalone rings). Any controlled stack is an eligible target.',
      'Exact-length lines always collapse fully into Territory and require you to pay the line elimination cost (one ring).',
      'Overlength lines can trade safety for value: you may collapse a shorter scoring segment with no elimination, or collapse the full line and pay the ring cost.',
    ],
    relatedPhases: ['line_processing'],
  },
  // GAP-LINE-02, GAP-LINE-03: Line vs Territory processing order teaching
  line_territory_order: {
    title: 'Line & Territory Order',
    icon: '📊',
    description:
      'Lines and territories are processed in a specific order after movement. Understanding this sequence is crucial for strategic planning.',
    tips: [
      // GAP-LINE-02: Processing order
      'PROCESSING ORDER: Lines are ALWAYS processed BEFORE territories. When a move creates both, lines collapse first, then territory regions are evaluated.',
      'A single turn can trigger: movement → capture chain → line processing → territory processing. Each phase must complete before the next begins.',
      // GAP-LINE-03: Multi-line and options
      'MULTIPLE LINES: If multiple lines form, each is processed separately. You may need to make choices for each overlength line.',
      'OPTION 1 vs OPTION 2: For overlength lines (6+ on square, 7+ on hex), you choose: collapse ALL markers (costs a ring) or collapse MINIMUM length (free but less territory).',
      'Line collapse affects what territory regions form. Choosing Option 1 (full collapse) may create more territory but costs a ring; Option 2 (minimum) is safely free.',
    ],
    relatedPhases: ['line_processing', 'territory_processing'],
  },
  territory: {
    title: TEACHING_TOPICS_COPY.territory.heading,
    icon: '▣',
    // UX_RULES_COPY_SPEC.md §7 – Territory description
    description: TEACHING_TOPICS_COPY.territory.body,
    tips: [
      'Territory comes from collapsing marker lines and resolving disconnected regions.',
      'Once a space becomes Territory it cannot be captured back or undone.',
      'Crossing the territory threshold ends the game immediately, even if other wins were possible.',
      // GAP-TERR-04: Self-elimination cost explanation (FAQ Q23)
      'TERRITORY ELIMINATION COST: Processing a disconnected region eliminates all interior rings (scoring for you), but you MUST also eliminate your ENTIRE CAP from an eligible stack OUTSIDE the region.',
      // RR-CANON-R082: Eligible cap targets for territory processing
      'ELIGIBLE STACKS FOR TERRITORY: Only multicolor stacks you control (with buried rings of other colors) OR single-color stacks of height > 1 are eligible. Standalone rings (height 1) are NOT eligible for territory processing.',
      // GAP-TERR-03: Eligibility indicator
      "CAN'T PROCESS A REGION? You must have an ELIGIBLE stack OUTSIDE the pending region. If all your outside stacks are standalone rings (height 1), you cannot process the territory.",
      // Explicit rules/FAQ reference for mini-regions
      'For the canonical mini-region pattern and numeric example, see FAQ Q23 "What happens if I cannot eliminate any rings when processing a disconnected region?" in ringrift_complete_rules §12.2.',
    ],
    relatedPhases: ['territory_processing'],
  },
  active_no_moves: {
    title: TEACHING_TOPICS_COPY.active_no_moves.heading,
    icon: '⛔',
    // UX_RULES_COPY_SPEC.md §10.4 – teaching.active_no_moves description
    description: TEACHING_TOPICS_COPY.active_no_moves.body,
    tips: [
      'Active–No–Moves only looks at real moves: placements, movements, and captures. Forced elimination and automatic line/territory processing do not count as real moves for Last Player Standing.',
      'If you still control stacks but have no placements or movements, the game applies forced elimination caps until a real move becomes available or your stacks are exhausted.',
      'On some boards a full plateau can occur where no player has real moves or forced eliminations; in that case the game ends and the final score comes from territory and eliminated rings, not further play.',
      // GAP-ANM-01: First-occurrence context
      'FIRST TIME SEEING THIS? When you have no legal moves, you enter an "Active-No-Moves" state. This is different from being eliminated – you are still in the game!',
      // GAP-ANM-03: Recovery guidance
      'HOW TO RECOVER FROM ANM: Your opponents might open up movement options for you by moving their stacks, collapsing lines, or processing territories. Stay alert – you can become active again!',
      // Explicit rules/FAQ reference for ANM/FE behaviour
      'For full timing diagrams and examples, see "Active–No–Moves & Forced Elimination" in ringrift_complete_rules §10.4 together with ACTIVE_NO_MOVES_BEHAVIOUR.md (ANM-SCEN-01).',
    ],
    relatedPhases: ['movement', 'line_processing', 'territory_processing'],
  },
  forced_elimination: {
    title: TEACHING_TOPICS_COPY.forced_elimination.heading,
    icon: '💥',
    // UX_RULES_COPY_SPEC.md §10.4 – teaching.forced_elimination description
    description: TEACHING_TOPICS_COPY.forced_elimination.body,
    tips: [
      'FORCED ELIMINATION COST: Your ENTIRE CAP is removed from a controlled stack. Unlike territory processing, ANY controlled stack is eligible—including standalone rings (height 1).',
      'Rings removed by forced elimination are permanently eliminated and count toward global Ring Elimination victory, just like eliminations from movement onto markers, line rewards, or territory processing.',
      'Forced elimination does not count as a "real move" for Last Player Standing, even though each step is recorded as a forced_elimination move in its own phase.',
      'You cannot skip forced elimination when its conditions are met; the rules may let you choose the stack, but some legal forced_elimination move must be recorded.',
      // Explicit rules reference for FE triggers
      'Formal FE triggers and guarantees are defined in ringrift_complete_rules §4.4 "Forced elimination when blocked" and in RULES_RULESET_CLARIFICATIONS under the FE invariants.',
    ],
    relatedPhases: ['movement', 'territory_processing', 'forced_elimination'],
  },
  // GAP-RECOV-01: TeachingOverlay topic for recovery action
  recovery_action: {
    title: 'Recovery Action',
    icon: '🔄',
    // Recovery action description based on RR-CANON-R110–R115
    description:
      'When you have no stacks but still own markers and have buried rings inside opponent stacks, you can use a Recovery Action to stay in the game by sliding markers to form lines.',
    tips: [
      // GAP-RECOV-01: Eligibility explanation
      'WHEN CAN YOU RECOVER? You are eligible for recovery when: (1) you control NO stacks, (2) you own at least one marker on the board, and (3) you have at least one buried ring (your color inside any stack, not as top ring).',
      // GAP-RECOV-01: Rings in hand clarification
      'RINGS IN HAND: Having rings in hand does NOT prevent recovery eligibility. If you have rings but no stacks to place them on, you can still use recovery action if you have markers and buried rings.',
      // GAP-RECOV-02: Recovery action mechanics
      'HOW TO RECOVER: Slide one of your markers to an adjacent empty cell. If the slide completes a line of markers (at least the required line length), the line collapses into territory.',
      // GAP-RECOV-02: Fallback recovery
      'FALLBACK RECOVERY: If no slide can form a line anywhere on the board, you may slide a marker to any adjacent empty cell, or perform a "stack-strike" onto an adjacent stack (sacrificing the marker to eliminate the top ring).',
      // GAP-RECOV-03: Cost and payment
      'RECOVERY COST: Exact-length lines cost 1 buried ring. Overlength lines offer a choice: Option 1 (collapse all) costs 1 buried ring; Option 2 (collapse minimum) is FREE. Fallback recovery costs 1 buried ring.',
      // GAP-RECOV-03: Temporary vs permanent elimination distinction
      'TEMPORARILY ELIMINATED vs PERMANENTLY ELIMINATED: You are "temporarily eliminated" (can still recover) if you have markers + buried rings. You are "permanently eliminated" only when you have NO rings anywhere (no stacks, no hand, no buried rings).',
      // GAP-RECOV-04: Recovery vs real actions for LPS
      'RECOVERY AND LPS: Recovery actions do NOT count as "real actions" for Last Player Standing. Like forced elimination, recovery keeps you in the game but does not prevent an opponent from winning via LPS.',
      // Skip option
      'SKIP RECOVERY: You may choose to skip recovery entirely, preserving your buried rings for a future turn.',
      // Rules reference
      'For detailed recovery mechanics, see ringrift_complete_rules §4.5 and RULES_CANONICAL_SPEC R110–R115.',
    ],
    relatedPhases: ['movement'],
  },
  victory_elimination: {
    title: TEACHING_TOPICS_COPY.victory_elimination.heading,
    icon: '💎',
    // UX_RULES_COPY_SPEC.md §3.1 – TeachingOverlay victory topic – elimination
    description: TEACHING_TOPICS_COPY.victory_elimination.body,
    tips: [
      'Track eliminated rings globally, across all players',
      'Eliminations come from movement onto markers, line rewards, territory processing, and forced eliminations',
      'A large line or territory resolution can suddenly push you over the elimination threshold',
    ],
  },
  victory_territory: {
    title: TEACHING_TOPICS_COPY.victory_territory.heading,
    icon: '🏰',
    // UX_RULES_COPY_SPEC.md §3.2 – TeachingOverlay victory topic – territory
    description: TEACHING_TOPICS_COPY.victory_territory.body,
    tips: [
      'Territory is permanent: once claimed, those spaces never return to neutral.',
      'Territory victory requires BOTH: at least your fair share of the board (1/N in N-player) AND more territory than all opponents combined.',
      'On an 8×8 2-player board, you need at least 33 Territory spaces. In 3-player, you need 22+ AND more than both opponents combined.',
      'Territory victories often come from big line/region resolutions rather than single moves.',
    ],
  },
  victory_stalemate: {
    title: TEACHING_TOPICS_COPY.victory_stalemate.heading,
    icon: '👑',
    // UX_RULES_COPY_SPEC.md §3.3 – TeachingOverlay victory topic – stalemate / LPS
    description: TEACHING_TOPICS_COPY.victory_stalemate.body,
    tips: [
      // === Last Player Standing (LPS) ===
      'LAST PLAYER STANDING: You win if you are the only player who can make real moves (placements, movements, or captures) for THREE consecutive complete rounds.',
      // Three-round requirement emphasis
      'LPS requires THREE rounds: Across all three rounds, you must have and take at least one real action while all others have none. Victory is declared after the third round completes.',
      // GAP-LPS-03: Emphasize FE ≠ real action
      'Forced elimination is NOT a real action: even if you are forced to eliminate caps, that does not count as a move for LPS purposes. If your opponent has real moves and you only have forced eliminations, they have not lost yet.',
      // GAP-RECOV-04: Recovery also ≠ real action for LPS
      "Recovery actions are NOT real actions: like forced elimination, recovery slides (marker slides by temporarily eliminated players) do not count toward breaking an opponent's LPS progress.",
      // Real actions summary for clarity
      'REAL ACTIONS FOR LPS: Only placements, movements, and captures count as real actions. Forced elimination and recovery do NOT count for LPS.',
      // === Structural Stalemate ===
      'STRUCTURAL STALEMATE: This happens when NO player has ANY real moves or forced eliminations available – the game is truly stuck.',
      // GAP-STALE-04: Distinction between single-player ANM and global stalemate
      'ANM vs Stalemate: When only YOU have no moves, the game continues – other players can still play. A structural stalemate only occurs when NOBODY can move at all.',
      // Tiebreak ladder
      'TIEBREAK LADDER: In a stalemate, the winner is determined by: 1) Territory spaces, 2) Eliminated rings (including rings in hand), 3) Markers on board, 4) Who made the last real action.',
      // Explicit rules reference for LPS + structural stalemate ladder
      'See ringrift_complete_rules §13.3 "Last Player Standing" and §13.4 "Structural stalemate and tiebreak ladder" for the full formal definitions and worked examples.',
    ],
  },
};

/**
 * Mapping from TeachingOverlay topics to the primary rulesConcept families
 * they relate to. Used to surface related scenario-driven teaching flows
 * defined in shared teaching metadata.
 */
const TOPIC_RULES_CONCEPTS: Partial<Record<TeachingTopic, RulesConcept[]>> = {
  active_no_moves: ['anm_forced_elimination'],
  forced_elimination: ['anm_forced_elimination'],
  territory: ['territory_mini_region'],
  chain_capture: ['capture_chain_mandatory'],
  line_bonus: ['line_vs_territory_multi_phase'],
  line_territory_order: ['line_vs_territory_multi_phase'],
  victory_stalemate: ['structural_stalemate', 'last_player_standing'],
};

export interface WeirdStateOverlayContext {
  reasonCode: RulesWeirdStateReasonCode;
  rulesContext: RulesUxContext;
  weirdStateType?: RulesUxWeirdStateType;
  boardType: BoardType;
  numPlayers: number;
  isRanked?: boolean;
  isSandbox?: boolean;
  overlaySessionId: string;
}

export interface TeachingOverlayProps {
  /** The topic to display */
  topic: TeachingTopic;
  /** Whether the overlay is visible */
  isOpen: boolean;
  /** Callback when overlay is closed */
  onClose: () => void;
  /** Position hint for the overlay */
  position?: 'center' | 'bottom-right';
  /** Additional CSS classes */
  className?: string;
  /** Optional weird-state context when opened from a weird-state help surface. */
  weirdStateOverlayContext?: WeirdStateOverlayContext | null;
  /**
   * Optional FSM decision surface for contextual explanations.
   * When provided, the overlay will include dynamic tips based on the
   * current FSM decision state (pending lines, regions, chain captures, etc.).
   */
  fsmContext?: FSMDecisionSurfaceViewModel | null;
}

/**
 * Overlay component that displays teaching content for game concepts.
 * Used during analysis, replay, or when players need help understanding a mechanic.
 */
export function TeachingOverlay({
  topic,
  isOpen,
  onClose,
  position = 'center',
  className = '',
  weirdStateOverlayContext,
  fsmContext,
}: TeachingOverlayProps) {
  const content = TEACHING_CONTENT[topic];

  // Generate FSM-aware dynamic tips based on current decision surface
  const fsmAwareTips = useMemo((): string[] => {
    if (!fsmContext?.isActive) return [];

    const tips: string[] = [];
    const {
      decisionType,
      summary,
      actionHint,
      pendingLineCount,
      pendingRegionCount,
      chainContinuationCount,
      forcedEliminationCount: _forcedEliminationCount,
    } = fsmContext;

    // Add contextual tip based on FSM decision type
    if (summary) {
      tips.push(`CURRENT STATE: ${summary}`);
    }
    if (actionHint) {
      tips.push(`WHAT TO DO: ${actionHint}`);
    }

    // Add phase-specific FSM guidance
    switch (decisionType) {
      case 'chain_capture':
        if (chainContinuationCount > 1) {
          tips.push(
            `You have ${chainContinuationCount} possible capture targets. Choose wisely - once you start, you must continue until no captures remain.`
          );
        }
        break;
      case 'line_order_required':
        if (pendingLineCount > 1) {
          tips.push(
            `Processing order matters: which line you resolve first may affect what territory regions form afterward.`
          );
        }
        break;
      case 'region_order_required':
        if (pendingRegionCount > 1) {
          tips.push(
            `Multiple disconnected regions detected. Each requires eliminating the entire cap from an outside stack.`
          );
        }
        break;
      case 'forced_elimination':
        tips.push(
          `Forced elimination is required because you have more rings than allowed. This is not optional - you must select a stack.`
        );
        break;
    }

    return tips;
  }, [fsmContext]);

  const relatedScenarios: TeachingScenarioMetadata[] = useMemo(() => {
    const concepts = TOPIC_RULES_CONCEPTS[topic];
    if (!concepts || concepts.length === 0) {
      return [];
    }
    return TEACHING_SCENARIOS.filter(
      (scenario) =>
        concepts.includes(scenario.rulesConcept) && scenario.showInTeachingOverlay === true
    );
  }, [topic]);

  const [currentFlowId, setCurrentFlowId] = useState<string | null>(null);
  const [currentStepIndex, setCurrentStepIndex] = useState<number | null>(null);
  const [teachingFlowId, setTeachingFlowId] = useState<string | null>(null);

  const lastWeirdStateContextRef = useRef<WeirdStateOverlayContext | null>(null);
  const hasShownForSessionRef = useRef<string | null>(null);
  const hasDismissedForSessionRef = useRef<string | null>(null);
  const prevIsOpenRef = useRef(false);
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);

  const selectedScenario: TeachingScenarioMetadata | undefined = useMemo(() => {
    if (!currentFlowId || currentStepIndex == null) {
      return undefined;
    }
    return relatedScenarios.find(
      (scenario) => scenario.flowId === currentFlowId && scenario.stepIndex === currentStepIndex
    );
  }, [currentFlowId, currentStepIndex, relatedScenarios]);

  // Reset local teaching-flow state whenever the overlay closes or the topic changes.
  useEffect(() => {
    if (!isOpen) {
      setCurrentFlowId(null);
      setCurrentStepIndex(null);
      setTeachingFlowId(null);
    }
  }, [isOpen, topic]);

  // Track the last non-null weird-state context so that dismiss events can still
  // be emitted even if the parent clears the prop before closing the overlay.
  useEffect(() => {
    if (weirdStateOverlayContext) {
      lastWeirdStateContextRef.current = weirdStateOverlayContext;
    }
  }, [weirdStateOverlayContext]);

  // Emit weird_state_overlay_shown / weird_state_overlay_dismiss lifecycle events
  // when the TeachingOverlay is used in a weird-state help context.
  useEffect(() => {
    const ctx = weirdStateOverlayContext ?? lastWeirdStateContextRef.current;
    const wasOpen = prevIsOpenRef.current;
    const nowOpen = isOpen;
    prevIsOpenRef.current = nowOpen;

    if (!ctx) {
      return;
    }

    if (nowOpen && ctx.overlaySessionId !== hasShownForSessionRef.current) {
      hasShownForSessionRef.current = ctx.overlaySessionId;
      void logRulesUxEvent({
        type: 'weird_state_overlay_shown',
        boardType: ctx.boardType,
        numPlayers: ctx.numPlayers,
        rulesContext: ctx.rulesContext,
        source: 'teaching_overlay',
        ...(ctx.weirdStateType !== undefined ? { weirdStateType: ctx.weirdStateType } : {}),
        reasonCode: ctx.reasonCode,
        ...(ctx.isRanked !== undefined ? { isRanked: ctx.isRanked } : {}),
        ...(ctx.isSandbox !== undefined ? { isSandbox: ctx.isSandbox } : {}),
        overlaySessionId: ctx.overlaySessionId,
      });
    }

    if (wasOpen && !nowOpen && ctx.overlaySessionId !== hasDismissedForSessionRef.current) {
      hasDismissedForSessionRef.current = ctx.overlaySessionId;
      void logRulesUxEvent({
        type: 'weird_state_overlay_dismiss',
        boardType: ctx.boardType,
        numPlayers: ctx.numPlayers,
        rulesContext: ctx.rulesContext,
        source: 'teaching_overlay',
        ...(ctx.weirdStateType !== undefined ? { weirdStateType: ctx.weirdStateType } : {}),
        reasonCode: ctx.reasonCode,
        ...(ctx.isRanked !== undefined ? { isRanked: ctx.isRanked } : {}),
        ...(ctx.isSandbox !== undefined ? { isSandbox: ctx.isSandbox } : {}),
        overlaySessionId: ctx.overlaySessionId,
      });
    }
  }, [isOpen, weirdStateOverlayContext]);

  // When opened from a weird-state surface, auto-select the first related
  // teaching step, preferring:
  //   1) An exact uxWeirdStateReasonCode match, then
  //   2) A rules-concept match for the provided rulesContext, then
  //   3) The first related scenario as a generic fallback.
  // In all cases we emit teaching_step_started for the chosen step.
  useEffect(() => {
    if (!isOpen) return;
    if (!weirdStateOverlayContext) return;
    if (currentFlowId || currentStepIndex != null) return;
    if (relatedScenarios.length === 0) return;

    const ctx = weirdStateOverlayContext;

    // 1) Prefer an exact reason-code match when available.
    let preferred: TeachingScenarioMetadata | undefined =
      relatedScenarios.find((scenario) => scenario.uxWeirdStateReasonCode === ctx.reasonCode) ??
      undefined;

    // 2) Otherwise fall back to a rules-context match derived from the
    //    scenario's rulesConcept (e.g. structural_stalemate, territory_mini_region).
    if (!preferred && ctx.rulesContext) {
      preferred = relatedScenarios.find(
        (scenario) => getRulesUxContextForTeachingScenario(scenario) === ctx.rulesContext
      );
    }

    // 3) Generic fallback: first related scenario for the topic.
    const fallback = preferred ?? relatedScenarios[0];
    if (!fallback) return;

    handleScenarioSelect(fallback, { isAuto: true });
  }, [isOpen, weirdStateOverlayContext, relatedScenarios, currentFlowId, currentStepIndex]);

  function handleScenarioSelect(
    scenario: TeachingScenarioMetadata,
    options: { isAuto?: boolean } = {}
  ) {
    const nextFlowId = scenario.flowId;
    const nextStepIndex = scenario.stepIndex;
    setCurrentFlowId(nextFlowId);
    setCurrentStepIndex(nextStepIndex);

    let flowId = teachingFlowId;
    if (!flowId || currentFlowId !== nextFlowId) {
      flowId = newTeachingFlowId();
      setTeachingFlowId(flowId);
    }

    const rulesContext = getRulesUxContextForTeachingScenario(scenario);
    const ctxForTeaching = weirdStateOverlayContext ?? lastWeirdStateContextRef.current;

    void logRulesUxEvent({
      type: 'teaching_step_started',
      source: 'teaching_overlay',
      boardType: scenario.recommendedBoardType,
      numPlayers: scenario.recommendedNumPlayers,
      ...(rulesContext ? { rulesContext } : {}),
      rulesConcept: scenario.rulesConcept,
      scenarioId: scenario.scenarioId,
      teachingFlowId: flowId,
      ...(ctxForTeaching?.overlaySessionId
        ? { overlaySessionId: ctxForTeaching.overlaySessionId }
        : {}),
      payload: {
        flowId: scenario.flowId,
        stepIndex: scenario.stepIndex,
        topic,
        startedAutomatically: options.isAuto === true,
      },
    });
  }

  function handleMarkStepUnderstood() {
    if (!selectedScenario) {
      return;
    }

    let flowId = teachingFlowId;
    if (!flowId) {
      flowId = newTeachingFlowId();
      setTeachingFlowId(flowId);
    }

    const rulesContext = getRulesUxContextForTeachingScenario(selectedScenario);
    const ctxForTeaching = weirdStateOverlayContext ?? lastWeirdStateContextRef.current;

    void logRulesUxEvent({
      type: 'teaching_step_completed',
      source: 'teaching_overlay',
      boardType: selectedScenario.recommendedBoardType,
      numPlayers: selectedScenario.recommendedNumPlayers,
      ...(rulesContext ? { rulesContext } : {}),
      rulesConcept: selectedScenario.rulesConcept,
      scenarioId: selectedScenario.scenarioId,
      teachingFlowId: flowId,
      ...(ctxForTeaching?.overlaySessionId
        ? { overlaySessionId: ctxForTeaching.overlaySessionId }
        : {}),
      payload: {
        flowId: selectedScenario.flowId,
        stepIndex: selectedScenario.stepIndex,
        topic,
        completionAction: 'mark_understood',
      },
    });
  }

  // Close on Escape key
  useEffect(() => {
    if (!isOpen || position === 'center') return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose, position]);

  if (!isOpen || !content) return null;

  const cardClasses = `relative bg-slate-900 border border-slate-600 rounded-xl shadow-2xl max-w-md w-full mx-4 overflow-hidden ${className}`;

  const cardBody = (
    <>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-slate-800/80 border-b border-slate-700">
        <div className="flex items-center gap-3">
          <span className="text-2xl">{content.icon}</span>
          <h2 id="teaching-title" className="text-lg font-bold text-slate-100">
            {content.title}
          </h2>
        </div>
        <button
          ref={closeButtonRef}
          type="button"
          onClick={onClose}
          className="p-1 rounded hover:bg-slate-700 text-slate-400 hover:text-slate-200 transition-colors"
          aria-label="Close"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </div>

      {/* Description */}
      <div className="px-4 py-3">
        <p className="text-sm text-slate-300 leading-relaxed">{content.description}</p>
      </div>

      {/* FSM-Aware Current Situation (when active) */}
      {fsmAwareTips.length > 0 && (
        <div className="px-4 pb-4">
          <h3 className="text-xs font-semibold text-amber-400 uppercase tracking-wide mb-2">
            Current Situation (FSM)
          </h3>
          <ul className="space-y-2 bg-amber-900/20 rounded-lg p-3 border border-amber-700/30">
            {fsmAwareTips.map((tip, idx) => (
              <li key={`fsm-${idx}`} className="flex items-start gap-2 text-sm text-amber-200">
                <span className="text-amber-400 mt-0.5">→</span>
                <span>{tip}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Tips */}
      <div className="px-4 pb-4">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Tips</h3>
        <ul className="space-y-2">
          {content.tips.map((tip, idx) => (
            <li key={idx} className="flex items-start gap-2 text-sm text-slate-400">
              <span className="text-emerald-400 mt-0.5">•</span>
              <span>{tip}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Related teaching scenarios for this topic (flows & steps) */}
      {relatedScenarios.length > 0 && (
        <div className="px-4 pb-4">
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
            Related teaching steps
          </h3>
          <ul className="space-y-1.5">
            {relatedScenarios.map((scenario) => {
              const isActive =
                currentFlowId === scenario.flowId && currentStepIndex === scenario.stepIndex;

              return (
                <li key={scenario.scenarioId}>
                  <button
                    type="button"
                    onClick={() => handleScenarioSelect(scenario)}
                    className={`w-full text-left rounded-lg border px-3 py-2 text-xs transition ${
                      isActive
                        ? 'border-emerald-500 bg-emerald-900/40 text-emerald-50'
                        : 'border-slate-700 bg-slate-900/60 text-slate-200 hover:border-slate-400'
                    }`}
                    data-testid="teaching-related-step"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="font-semibold text-slate-100">
                        {scenario.flowId} · Step {scenario.stepIndex}
                      </div>
                      {isActive && (
                        <span className="text-[10px] uppercase tracking-wide text-emerald-300">
                          Selected
                        </span>
                      )}
                    </div>
                    <div className="text-slate-400">{scenario.learningObjectiveShort}</div>
                  </button>
                </li>
              );
            })}
          </ul>

          {selectedScenario && (
            <div
              className="mt-3 rounded-lg border border-slate-700 bg-slate-900/80 px-3 py-2 text-xs text-slate-200"
              data-testid="teaching-step-details"
            >
              <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-400">
                Step details
              </div>
              <div className="mt-1 font-semibold text-slate-100">
                {selectedScenario.flowId} · Step {selectedScenario.stepIndex}
              </div>
              <p className="mt-1 text-slate-300">{selectedScenario.learningObjectiveShort}</p>
              <p className="mt-1 text-slate-400">
                Play this scenario from the sandbox presets to practice this situation with a real
                board.
              </p>
              <button
                type="button"
                onClick={handleMarkStepUnderstood}
                className="mt-2 inline-flex items-center justify-center rounded-md border border-emerald-500 bg-emerald-700/30 px-3 py-1 text-[11px] font-semibold text-emerald-100 hover:bg-emerald-600/40"
              >
                Mark as understood
              </button>
            </div>
          )}
        </div>
      )}

      {/* Related Phases Badge */}
      {content.relatedPhases && content.relatedPhases.length > 0 && (
        <div className="px-4 pb-3 flex items-center gap-2">
          <span className="text-[10px] text-slate-500 uppercase">Applies to:</span>
          {content.relatedPhases.map((phase) => (
            <span
              key={phase}
              className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700"
            >
              {phase.replace('_', ' ')}
            </span>
          ))}
        </div>
      )}

      {/* Footer hint */}
      <div className="px-4 py-2 bg-slate-800/50 border-t border-slate-700/50 text-center">
        <span className="text-[10px] text-slate-500">
          Press <kbd className="px-1 py-0.5 bg-slate-700 rounded text-[9px]">Esc</kbd> or click
          outside to close
        </span>
      </div>
    </>
  );

  if (position === 'center') {
    return (
      <Dialog
        isOpen={isOpen}
        onClose={onClose}
        labelledBy="teaching-title"
        initialFocusRef={closeButtonRef}
        overlayClassName="z-[60] items-center justify-center"
        backdropClassName="bg-black/50 backdrop-blur-sm"
        className={cardClasses}
      >
        {cardBody}
      </Dialog>
    );
  }

  return (
    <div className="fixed bottom-4 right-4" style={{ zIndex: 60 }}>
      <div className={cardClasses} role="dialog" aria-labelledby="teaching-title">
        {cardBody}
      </div>
    </div>
  );
}

/**
 * Hook to manage teaching overlay state
 */
export function useTeachingOverlay() {
  const [currentTopic, setCurrentTopic] = useState<TeachingTopic | null>(null);

  const showTopic = (topic: TeachingTopic) => setCurrentTopic(topic);
  const hideTopic = () => setCurrentTopic(null);

  return {
    currentTopic,
    isOpen: currentTopic !== null,
    showTopic,
    hideTopic,
  };
}

/**
 * Determine which teaching topic is relevant for a given move
 */
export function getTeachingTopicForMove(move: Move): TeachingTopic | null {
  switch (move.type) {
    case 'place_ring':
    case 'skip_placement':
      return 'ring_placement';
    case 'move_stack':
    case 'move_ring':
    case 'build_stack':
      return 'stack_movement';
    case 'overtaking_capture':
      return 'capturing';
    case 'continue_capture_segment':
      return 'chain_capture';
    case 'process_line':
    case 'choose_line_option':
    case 'choose_line_reward':
      return 'line_bonus';
    case 'process_territory_region':
    case 'choose_territory_option':
    case 'eliminate_rings_from_stack':
    case 'skip_territory_processing':
      return 'territory';
    case 'forced_elimination':
      return 'forced_elimination';
    case 'recovery_slide':
    case 'skip_recovery':
      return 'recovery_action';
    default:
      return null;
  }
}

/**
 * Quick-access buttons for all teaching topics
 */
export function TeachingTopicButtons({
  onSelectTopic,
  className = '',
}: {
  onSelectTopic: (topic: TeachingTopic) => void;
  className?: string;
}) {
  const topics: { topic: TeachingTopic; label: string; icon: string }[] = [
    { topic: 'ring_placement', label: 'Placement', icon: '◯' },
    { topic: 'stack_movement', label: 'Movement', icon: '→' },
    { topic: 'capturing', label: 'Capture', icon: '×' },
    { topic: 'chain_capture', label: 'Chain', icon: '⇉' },
    { topic: 'line_bonus', label: 'Lines', icon: '═' },
    { topic: 'line_territory_order', label: 'Phase Order', icon: '📊' },
    { topic: 'territory', label: 'Territory', icon: '▣' },
    { topic: 'recovery_action', label: 'Recovery', icon: '🔄' },
  ];

  const victoryTopics: { topic: TeachingTopic; label: string; icon: string }[] = [
    { topic: 'victory_elimination', label: 'Elimination', icon: '💎' },
    { topic: 'victory_territory', label: 'Territory', icon: '🏰' },
    { topic: 'victory_stalemate', label: 'Stalemate', icon: '👑' },
  ];

  return (
    <div className={`space-y-2 ${className}`}>
      <div className="text-[10px] text-slate-500 uppercase tracking-wide">Game Mechanics</div>
      <div className="flex flex-wrap gap-1">
        {topics.map(({ topic, label, icon }) => (
          <button
            key={topic}
            type="button"
            onClick={() => onSelectTopic(topic)}
            className="px-2 py-1 rounded text-[11px] bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 transition-colors"
          >
            {icon} {label}
          </button>
        ))}
      </div>

      <div className="text-[10px] text-slate-500 uppercase tracking-wide mt-3">
        Victory Conditions
      </div>
      <div className="flex flex-wrap gap-1">
        {victoryTopics.map(({ topic, label, icon }) => (
          <button
            key={topic}
            type="button"
            onClick={() => onSelectTopic(topic)}
            className="px-2 py-1 rounded text-[11px] bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 transition-colors"
          >
            {icon} {label}
          </button>
        ))}
      </div>
    </div>
  );
}

export const TEACHING_TOPICS = Object.keys(TEACHING_CONTENT) as TeachingTopic[];

const TEACHING_TOPIC_SET = new Set<string>(TEACHING_TOPICS);

export function parseTeachingTopic(value: string | null | undefined): TeachingTopic | null {
  if (!value) return null;
  if (!TEACHING_TOPIC_SET.has(value)) return null;
  return value as TeachingTopic;
}
