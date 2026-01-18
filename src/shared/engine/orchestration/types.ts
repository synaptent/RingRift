/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Turn Orchestration Types
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Types for the canonical processTurn orchestrator and related functions.
 */

import type { GameState, Position, Move, BoardType, Territory } from '../../types/game';
import type { GameEndExplanation } from '../gameEndExplanation';

// ═══════════════════════════════════════════════════════════════════════════
// PROCESS TURN TYPES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * FSM-derived decision surface.
 * Contains the raw data for pending decisions as determined by FSM orchestration.
 * This is the authoritative source for decision timing per FSM integration.
 */
export interface FSMDecisionSurface {
  /** Type of pending decision if any */
  pendingDecisionType?:
    | 'chain_capture'
    | 'line_order_required'
    | 'line_elimination_required'
    | 'no_line_action_required'
    | 'region_order_required'
    | 'no_territory_action_required'
    | 'forced_elimination';

  /** Detected lines for the current player (line_processing phase) */
  pendingLines?: Array<{ positions: Position[]; player?: number }>;

  /** Territory regions for the current player (territory_processing phase) */
  pendingRegions?: Array<{ positions: Position[]; eliminationsRequired?: number }>;

  /** Chain capture continuation targets (chain_capture phase) */
  chainContinuations?: Array<{ target: Position }>;

  /** Number of rings that must be eliminated (forced_elimination phase) */
  forcedEliminationCount?: number;
}

/**
 * Result of processing a turn via the canonical orchestrator.
 */
export interface ProcessTurnResult {
  /** Next game state after applying the move */
  nextState: GameState;

  /** Whether the turn completed or requires more decisions */
  status: 'complete' | 'awaiting_decision';

  /** Pending decision if status is 'awaiting_decision' (undefined if complete) */
  pendingDecision: PendingDecision | undefined;

  /** Victory result if game ended (undefined if ongoing) */
  victoryResult: VictoryState | undefined;

  /** Processing metadata for debugging/logging */
  metadata: ProcessingMetadata;

  /**
   * FSM-derived decision surface (Phase 2 of FSM integration).
   * Contains the raw FSM orchestration data for decisions.
   * Hosts can use this for advanced decision handling or debugging.
   * The `pendingDecision` field is derived from this surface.
   */
  fsmDecisionSurface?: FSMDecisionSurface;
}

/**
 * Types of decisions that may be pending.
 *
 * Per RR-CANON-R075, when a phase has no actions, the core rules layer returns
 * a pending decision requiring an explicit no-action move. The host layer can
 * auto-fill these for live play UX or wait for explicit moves during replay.
 */
export type DecisionType =
  | 'line_order'
  | 'line_reward'
  | 'line_elimination_required'
  | 'region_order'
  | 'elimination_target'
  | 'capture_direction'
  | 'chain_capture'
  // RR-CANON-R075: Required no-action moves when a phase has no available actions
  | 'no_line_action_required'
  | 'no_territory_action_required'
  | 'no_movement_action_required'
  | 'no_placement_action_required';

/**
 * Context information for rendering a decision UI.
 */
export interface DecisionContext {
  /** Human-readable description of what's being decided */
  description: string;

  /** Relevant positions on the board (for highlighting) */
  relevantPositions?: Position[];

  /** Additional metadata */
  extra?: Record<string, unknown>;
}

// ═══════════════════════════════════════════════════════════════════════════
// DISCRIMINATED UNION DECISION TYPES
// ═══════════════════════════════════════════════════════════════════════════
//
// These provide type-safe access to decision-specific context. Use type guards
// like `isLineOrderDecision(decision)` to narrow the type in switch statements.

/**
 * Base properties shared by all pending decisions.
 */
interface PendingDecisionBase {
  /** Player who must make the decision */
  player: number;

  /** Available options */
  options: Move[];

  /** Timeout in milliseconds (adapter concern) */
  timeoutMs?: number;

  /** Context for UI rendering */
  context: DecisionContext;

  /**
   * Intermediate game state at the point when this decision was surfaced.
   * This allows decision handlers to broadcast the state with the triggering
   * move already applied, so the UI shows the board update while the player
   * is deciding (e.g., territory regions are visible after the move).
   */
  intermediateState?: GameState;
}

/**
 * Decision for choosing which line to process first when multiple lines exist.
 * RR-CANON-R033: When multiple lines exist, player chooses processing order.
 */
export interface LineOrderDecision extends PendingDecisionBase {
  type: 'line_order';
  /** Detected lines available for processing */
  lines?: DetectedLineInfo[];
}

/**
 * Decision for choosing line collapse reward (elimination bonus).
 * RR-CANON-R031: After collapsing a line, player may eliminate opponent rings.
 */
export interface LineRewardDecision extends PendingDecisionBase {
  type: 'line_reward';
  /** Line that was collapsed (for context) */
  collapsedLine?: DetectedLineInfo;
}

/**
 * Decision for choosing which territory region to process first.
 * RR-CANON-R052: Disconnected territory regions processed in player-chosen order.
 */
export interface RegionOrderDecision extends PendingDecisionBase {
  type: 'region_order';
  /** Available regions for processing */
  regions?: Territory[];
}

/**
 * Decision for choosing which stack to eliminate from during forced elimination.
 * RR-CANON-R072/R100/R206: Forced elimination when no legal moves available.
 */
export interface EliminationTargetDecision extends PendingDecisionBase {
  type: 'elimination_target';
  /** Reason for elimination (forced_elimination vs line/territory) */
  eliminationReason?: 'forced_elimination' | 'line_reward' | 'territory_disconnection';
}

/**
 * Decision for choosing capture direction when multiple directions possible.
 */
export interface CaptureDirectionDecision extends PendingDecisionBase {
  type: 'capture_direction';
}

/**
 * Decision for continuing a chain capture.
 * RR-CANON-R025: After overtaking capture, must continue if possible.
 */
export interface ChainCaptureDecision extends PendingDecisionBase {
  type: 'chain_capture';
  /** Current chain capture position */
  chainPosition?: Position;
}

/**
 * Decision requiring eliminate_rings_from_stack after choose_line_option with 'eliminate'.
 * RR-CANON-R123: Line elimination is a separate move from the line option choice.
 */
export interface LineEliminationDecision extends PendingDecisionBase {
  type: 'line_elimination_required';
  /** Position of the line where elimination is pending */
  linePosition?: Position;
}

/**
 * Bookkeeping decision when no line actions available.
 * RR-CANON-R075: Explicit no-action move required for replay determinism.
 */
export interface NoLineActionDecision extends PendingDecisionBase {
  type: 'no_line_action_required';
}

/**
 * Bookkeeping decision when no territory actions available.
 * RR-CANON-R075: Explicit no-action move required for replay determinism.
 */
export interface NoTerritoryActionDecision extends PendingDecisionBase {
  type: 'no_territory_action_required';
}

/**
 * Bookkeeping decision when no movement actions available.
 * RR-CANON-R075: Explicit no-action move required for replay determinism.
 */
export interface NoMovementActionDecision extends PendingDecisionBase {
  type: 'no_movement_action_required';
}

/**
 * Bookkeeping decision when no placement actions available.
 * RR-CANON-R075: Explicit no-action move required for replay determinism.
 */
export interface NoPlacementActionDecision extends PendingDecisionBase {
  type: 'no_placement_action_required';
}

/**
 * A pending decision that requires player input.
 *
 * This is a discriminated union - use the `type` field to narrow to specific
 * decision types with their associated context.
 *
 * @example
 * ```typescript
 * function handleDecision(decision: PendingDecision) {
 *   switch (decision.type) {
 *     case 'line_order':
 *       // TypeScript knows decision.lines is available
 *       console.log(`${decision.lines?.length} lines to choose from`);
 *       break;
 *     case 'chain_capture':
 *       // TypeScript knows decision.chainPosition is available
 *       console.log(`Continue from ${decision.chainPosition}`);
 *       break;
 *   }
 * }
 * ```
 */
export type PendingDecision =
  | LineOrderDecision
  | LineRewardDecision
  | LineEliminationDecision
  | RegionOrderDecision
  | EliminationTargetDecision
  | CaptureDirectionDecision
  | ChainCaptureDecision
  | NoLineActionDecision
  | NoTerritoryActionDecision
  | NoMovementActionDecision
  | NoPlacementActionDecision;

// ═══════════════════════════════════════════════════════════════════════════
// TYPE GUARDS
// ═══════════════════════════════════════════════════════════════════════════

/** Type guard for line order decisions */
export function isLineOrderDecision(d: PendingDecision): d is LineOrderDecision {
  return d.type === 'line_order';
}

/** Type guard for line reward decisions */
export function isLineRewardDecision(d: PendingDecision): d is LineRewardDecision {
  return d.type === 'line_reward';
}

/** Type guard for region order decisions */
export function isRegionOrderDecision(d: PendingDecision): d is RegionOrderDecision {
  return d.type === 'region_order';
}

/** Type guard for elimination target decisions */
export function isEliminationTargetDecision(d: PendingDecision): d is EliminationTargetDecision {
  return d.type === 'elimination_target';
}

/** Type guard for chain capture decisions */
export function isChainCaptureDecision(d: PendingDecision): d is ChainCaptureDecision {
  return d.type === 'chain_capture';
}

/** Type guard for no-action decisions (any type) */
export function isNoActionDecision(d: PendingDecision): boolean {
  return (
    d.type === 'no_line_action_required' ||
    d.type === 'no_territory_action_required' ||
    d.type === 'no_movement_action_required' ||
    d.type === 'no_placement_action_required'
  );
}

/**
 * Delegates that hosts provide for async decision resolution.
 */
export interface TurnProcessingDelegates {
  /**
   * Resolve a player decision asynchronously.
   * Called when the engine needs player input for line reward, region order, etc.
   */
  resolveDecision(decision: PendingDecision): Promise<Move>;

  /**
   * Log processing events for debugging/tracing.
   */
  onProcessingEvent?(event: ProcessingEvent): void;

  /**
   * Override default auto-selection for AI decisions.
   * When provided, AI players use this strategy instead of default random.
   */
  autoSelectStrategy?: AutoSelectStrategy;
}

/**
 * Strategy for auto-selecting decisions (used by AI players).
 */
export interface AutoSelectStrategy {
  /**
   * Select a decision option for an AI player.
   * @param decision The pending decision
   * @param state Current game state
   * @returns The selected move
   */
  select(decision: PendingDecision, state: GameState): Move;
}

/**
 * Processing events emitted during turn execution.
 */
export interface ProcessingEvent {
  /** Event type */
  type: ProcessingEventType;
  /** Timestamp */
  timestamp: Date;
  /** Event-specific payload */
  payload: Record<string, unknown>;
}

export type ProcessingEventType =
  | 'turn_started'
  | 'phase_changed'
  | 'move_applied'
  | 'decision_required'
  | 'decision_resolved'
  | 'lines_detected'
  | 'line_collapsed'
  | 'territory_processed'
  | 'victory_detected'
  | 'turn_completed';

/**
 * Metadata about turn processing for debugging.
 */
export interface ProcessingMetadata {
  /** Move that was processed */
  processedMove: Move;

  /** Phases traversed during processing */
  phasesTraversed: string[];

  /** Lines detected (if any) */
  linesDetected: number;

  /** Regions processed (if any) */
  regionsProcessed: number;

  /** Processing duration in ms */
  durationMs: number;

  /** S-invariant before and after */
  sInvariantBefore: number;
  sInvariantAfter: number;
}

// ═══════════════════════════════════════════════════════════════════════════
// TERRITORY RESOLUTION TYPES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Options for territory resolution.
 */
export interface TerritoryResolutionOptions {
  /** Whether to process all regions automatically or one at a time */
  mode?: 'all' | 'single';

  /** Override regions to process (for testing) */
  testOverrideRegions?: Territory[];
}

/**
 * Result of resolving territory regions.
 */
export interface TerritoryResolutionResult {
  /** Next game state after territory processing */
  nextState: GameState;

  /** Regions that were processed */
  processedRegions: ProcessedRegion[];

  /** Pending self-eliminations */
  pendingEliminations: EliminationDecision[];

  /** Whether more regions need processing */
  hasMoreRegions: boolean;
}

/**
 * Information about a processed region.
 */
export interface ProcessedRegion {
  /** Identifier for this region */
  id: string;

  /** Spaces that were collapsed */
  collapsedSpaces: Position[];

  /** Border markers that were collapsed */
  borderMarkers: Position[];

  /** Rings eliminated from internal stacks */
  eliminatedRings: { player: number; count: number }[];

  /** Territory gained */
  territoryGained: number;
}

/**
 * A pending elimination decision.
 */
export interface EliminationDecision {
  /** Player who must eliminate */
  player: number;

  /** Stack to eliminate from */
  stackPosition: Position;

  /** Number of rings to eliminate (cap height) */
  count: number;
}

// ═══════════════════════════════════════════════════════════════════════════
// LINE DETECTION TYPES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Result of detecting lines.
 */
export interface LineDetectionResult {
  /** Detected lines meeting minimum length */
  lines: DetectedLineInfo[];

  /** Board type used for detection */
  boardType: BoardType;

  /** Minimum line length for this board */
  minimumLength: number;
}

/**
 * Information about a detected line.
 */
export interface DetectedLineInfo {
  /** Line positions in order */
  positions: Position[];

  /** Player who owns this line */
  player: number;

  /** Line length */
  length: number;

  /** Direction vector */
  direction: Position;

  /** Available collapse options */
  collapseOptions: LineCollapseOption[];
}

/**
 * A collapse option for a line.
 */
export interface LineCollapseOption {
  /** Option type */
  type: 'collapse_all' | 'minimum_collapse';

  /** Positions that would be collapsed */
  positions: Position[];

  /** Whether this grants elimination reward */
  grantsReward: boolean;

  /** Territory gained */
  territoryGained: number;
}

// ═══════════════════════════════════════════════════════════════════════════
// VICTORY STATE TYPES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Complete victory state information.
 */
export interface VictoryState {
  /** Whether the game has ended */
  isGameOver: boolean;

  /** Winner player number (undefined if draw or ongoing) */
  winner: number | undefined;

  /** Victory reason (undefined if game not over) */
  reason: VictoryReason | undefined;

  /** Detailed scores for all players */
  scores: PlayerScore[];

  /** Tie-breaking information (undefined if no tie) */
  tieBreaker: TieBreaker | undefined;

  /**
   * Optional structured explanation of how and why the game ended.
   *
   * This is a shared-engine-only projection built from GameState at the
   * point victory is detected. Hosts may choose to surface this in UI
   * or telemetry without affecting core rules semantics.
   */
  gameEndExplanation?: GameEndExplanation;
}

export type VictoryReason =
  | 'ring_elimination'
  | 'territory_control'
  | 'last_player_standing'
  | 'stalemate_resolution'
  | 'resignation';

/**
 * Score breakdown for a player.
 */
export interface PlayerScore {
  player: number;
  eliminatedRings: number;
  territorySpaces: number;
  ringsOnBoard: number;
  ringsInHand: number;
  markerCount: number;
  isEliminated: boolean;
}

/**
 * Tie-breaking information.
 */
export interface TieBreaker {
  /** Which criterion broke the tie */
  criterion: 'territory' | 'eliminated_rings' | 'markers' | 'last_actor';
  /** Description */
  description: string;
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE STATE MACHINE TYPES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Internal state maintained during turn processing.
 */
export interface TurnProcessingState {
  /** Current game state (may be mutated during processing) */
  gameState: GameState;

  /** Original move being processed */
  originalMove: Move;

  /** Per-turn flags */
  perTurnFlags: PerTurnFlags;

  /** Lines pending processing */
  pendingLines: DetectedLineInfo[];

  /** Regions pending processing */
  pendingRegions: Territory[];

  /** Whether chain capture is in progress */
  chainCaptureInProgress: boolean;

  /** Chain capture continuation position (undefined if not in chain capture) */
  chainCapturePosition: Position | undefined;

  /** Processing events accumulated */
  events: ProcessingEvent[];

  /** Phases traversed */
  phasesTraversed: string[];

  /** Start time for duration tracking */
  startTime: number;
}

/**
 * Per-turn flags maintained by the orchestrator.
 */
export interface PerTurnFlags {
  /** Whether player has placed this turn */
  hasPlacedThisTurn: boolean;

  /** Stack key that must move after placement (undefined if no forced move) */
  mustMoveFromStackKey: string | undefined;

  /** Whether elimination reward is pending */
  eliminationRewardPending: boolean;

  /** Number of elimination rewards pending */
  eliminationRewardCount: number;

  /**
   * Whether the player performed any meaningful action this turn.
   * This tracks actions across all phases (placement, movement, capture,
   * line processing, territory processing). Used to determine if the
   * forced_elimination phase should be entered per RR-CANON-R070/R204.
   */
  hadActionThisTurn: boolean;
}
