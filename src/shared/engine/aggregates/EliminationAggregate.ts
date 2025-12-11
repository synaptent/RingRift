/**
 * EliminationAggregate - Canonical elimination logic for RingRift.
 *
 * This is the SINGLE SOURCE OF TRUTH for all elimination semantics.
 * All other code should delegate to this module.
 *
 * Canonical Rules (from RULES_CANONICAL_SPEC.md):
 *
 * | Context              | Cost                    | Eligible Stacks                              | Reference      |
 * |----------------------|-------------------------|----------------------------------------------|----------------|
 * | Line Processing      | 1 ring from top         | Any controlled stack (including height-1)    | RR-CANON-R122  |
 * | Territory Processing | Entire cap              | Multicolor OR single-color height > 1        | RR-CANON-R145  |
 * | Forced Elimination   | Entire cap              | Any controlled stack (including height-1)    | RR-CANON-R100  |
 * | Recovery Action      | 1 buried ring extraction| Any stack with player's buried ring          | RR-CANON-R113  |
 *
 * Height-1 standalone rings are NOT eligible for territory processing but ARE
 * eligible for line, forced elimination, and recovery (as extraction source).
 *
 * @module EliminationAggregate
 */

import type { BoardState, Position, RingStack } from '../../types/game';
import { positionToString } from '../../types/game';

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Context in which elimination occurs. Determines cost and eligibility.
 *
 * - 'line': Line processing reward (RR-CANON-R122) - 1 ring, any stack
 * - 'territory': Territory self-elimination (RR-CANON-R145) - entire cap, eligible stacks only
 * - 'forced': Forced elimination when no moves (RR-CANON-R100) - entire cap, any stack
 * - 'recovery': Recovery buried ring extraction (RR-CANON-R113) - 1 buried ring
 */
export type EliminationContext = 'line' | 'territory' | 'forced' | 'recovery';

/**
 * Reason for elimination - used for audit trail and debugging.
 */
export type EliminationReason =
  | 'line_reward_option1' // Line collapse Option 1 (collapse all, pay 1 ring)
  | 'line_reward_exact' // Exact-length line (always pays 1 ring)
  | 'territory_self_elimination' // Territory processing self-elimination cost
  | 'forced_elimination_anm' // Forced elimination due to Active-No-Moves
  | 'recovery_buried_extraction' // Recovery action buried ring extraction
  | 'capture_overtake'; // Capture that results in ring elimination (not directly used here)

/**
 * Parameters for an elimination operation.
 */
export interface EliminationParams {
  /** Context determines cost and eligibility rules */
  context: EliminationContext;
  /** Player performing the elimination */
  player: number;
  /** Position of the stack to eliminate from */
  stackPosition: Position;
  /** Current board state */
  board: BoardState;
  /** Optional reason for audit trail */
  reason?: EliminationReason;
  /** For recovery: index of buried ring to extract (default: bottommost) */
  buriedRingIndex?: number;
}

/**
 * Result of an elimination operation.
 */
export interface EliminationResult {
  /** Whether elimination succeeded */
  success: boolean;
  /** Number of rings eliminated */
  ringsEliminated: number;
  /** Updated board state (new object, original not mutated) */
  updatedBoard: BoardState;
  /** Updated stack after elimination (null if stack removed) */
  updatedStack: RingStack | null;
  /** Error message if success is false */
  error?: string;
  /** Audit event for debugging */
  auditEvent?: EliminationAuditEvent;
}

/**
 * Audit event for tracking elimination operations.
 */
export interface EliminationAuditEvent {
  timestamp: Date;
  context: EliminationContext;
  reason?: EliminationReason;
  player: number;
  stackPosition: Position;
  ringsEliminated: number;
  stackHeightBefore: number;
  stackHeightAfter: number;
  capHeightBefore: number;
  controllingPlayerBefore: number;
  controllingPlayerAfter: number | null;
}

/**
 * Stack eligibility result with explanation.
 */
export interface StackEligibility {
  eligible: boolean;
  reason: string;
}

// ═══════════════════════════════════════════════════════════════════════════
// CORE FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Calculate cap height - number of consecutive rings from top belonging to controlling player.
 *
 * @param rings - Array of player numbers representing the stack (index 0 = top)
 * @returns Cap height (0 if stack is empty)
 */
export function calculateCapHeight(rings: number[]): number {
  if (rings.length === 0) return 0;
  const topPlayer = rings[0];
  let capHeight = 0;
  for (const ring of rings) {
    if (ring === topPlayer) {
      capHeight++;
    } else {
      break;
    }
  }
  return capHeight;
}

/**
 * Check if a stack is eligible for elimination in a given context.
 *
 * Eligibility rules per canonical spec:
 * - Line (RR-CANON-R122): Any controlled stack, including height-1
 * - Territory (RR-CANON-R145): Multicolor OR single-color height > 1; NOT height-1
 * - Forced (RR-CANON-R100): Any controlled stack, including height-1
 * - Recovery (RR-CANON-R113): Any stack containing player's buried ring
 *
 * @param stack - The stack to check
 * @param context - Elimination context
 * @param player - Player performing elimination
 * @returns Eligibility result with explanation
 */
export function isStackEligibleForElimination(
  stack: RingStack,
  context: EliminationContext,
  player: number
): StackEligibility {
  // Recovery has special rules - checks for buried rings, not control
  if (context === 'recovery') {
    // Find any buried ring belonging to player
    const hasBuriedRing = stack.rings.slice(1).some((ring: number) => ring === player);
    if (!hasBuriedRing) {
      return { eligible: false, reason: 'No buried rings of player in stack' };
    }
    return { eligible: true, reason: 'Stack contains buried ring of player' };
  }

  // For line/territory/forced: must control the stack
  if (stack.controllingPlayer !== player) {
    return { eligible: false, reason: 'Player does not control stack' };
  }

  const capHeight = calculateCapHeight(stack.rings);
  if (capHeight <= 0) {
    return { eligible: false, reason: 'No cap to eliminate' };
  }

  // Line and Forced: any controlled stack is eligible (including height-1)
  if (context === 'line' || context === 'forced') {
    return { eligible: true, reason: `${context} allows any controlled stack` };
  }

  // Territory: must be multicolor OR single-color height > 1
  // Height-1 standalone rings are NOT eligible
  const isMulticolor = stack.stackHeight > capHeight;
  const isSingleColorTall = stack.stackHeight === capHeight && stack.stackHeight > 1;

  if (isMulticolor) {
    return { eligible: true, reason: 'Multicolor stack (buried rings of other colors)' };
  }
  if (isSingleColorTall) {
    return { eligible: true, reason: 'Single-color stack with height > 1' };
  }

  // Height-1 standalone ring - not eligible for territory
  return {
    eligible: false,
    reason: 'Height-1 standalone ring not eligible for territory elimination',
  };
}

/**
 * Calculate how many rings to eliminate based on context.
 *
 * @param stack - The stack being eliminated from
 * @param context - Elimination context
 * @returns Number of rings to eliminate
 */
export function getRingsToEliminate(stack: RingStack, context: EliminationContext): number {
  const capHeight = calculateCapHeight(stack.rings);

  switch (context) {
    case 'line':
      // Line processing: always 1 ring (RR-CANON-R122)
      return 1;

    case 'territory':
    case 'forced':
      // Territory and Forced: entire cap (RR-CANON-R145, RR-CANON-R100)
      return capHeight;

    case 'recovery':
      // Recovery: 1 buried ring extraction (RR-CANON-R113)
      return 1;

    default:
      return capHeight;
  }
}

/**
 * Perform elimination from a stack.
 *
 * This is the CANONICAL elimination function. All elimination operations
 * should go through this function to ensure consistent semantics.
 *
 * @param params - Elimination parameters
 * @returns Elimination result with updated board
 */
export function eliminateFromStack(params: EliminationParams): EliminationResult {
  const { context, player, stackPosition, board, reason, buriedRingIndex } = params;
  const posKey = positionToString(stackPosition);
  const stack = board.stacks.get(posKey);

  // Validate stack exists
  if (!stack) {
    return {
      success: false,
      ringsEliminated: 0,
      updatedBoard: board,
      updatedStack: null,
      error: `No stack at position ${posKey}`,
    };
  }

  // Check eligibility
  const eligibility = isStackEligibleForElimination(stack, context, player);
  if (!eligibility.eligible) {
    return {
      success: false,
      ringsEliminated: 0,
      updatedBoard: board,
      updatedStack: stack,
      error: eligibility.reason,
    };
  }

  // Determine rings to eliminate
  let ringsToEliminate: number;
  let remainingRings: number[];

  if (context === 'recovery') {
    // Recovery: extract bottommost buried ring of player
    ringsToEliminate = 1;
    const extractIndex =
      buriedRingIndex ??
      stack.rings
        .slice(1)
        .reverse()
        .findIndex((ring) => ring === player);
    const actualIndex = stack.rings.length - 1 - extractIndex;

    if (actualIndex < 1 || stack.rings[actualIndex] !== player) {
      return {
        success: false,
        ringsEliminated: 0,
        updatedBoard: board,
        updatedStack: stack,
        error: 'No valid buried ring to extract',
      };
    }

    // Remove the buried ring, keeping everything else
    remainingRings = [...stack.rings.slice(0, actualIndex), ...stack.rings.slice(actualIndex + 1)];
  } else {
    // Line/Territory/Forced: remove from top
    ringsToEliminate = getRingsToEliminate(stack, context);
    remainingRings = stack.rings.slice(ringsToEliminate);
  }

  // Create updated board (immutable)
  const updatedStacks = new Map(board.stacks);
  const updatedEliminatedRings = { ...board.eliminatedRings };
  updatedEliminatedRings[player] = (updatedEliminatedRings[player] || 0) + ringsToEliminate;

  let updatedStack: RingStack | null = null;

  if (remainingRings.length > 0) {
    // Stack still has rings
    updatedStack = {
      ...stack,
      rings: remainingRings,
      stackHeight: remainingRings.length,
      capHeight: calculateCapHeight(remainingRings),
      controllingPlayer: remainingRings[0],
    };
    updatedStacks.set(posKey, updatedStack);
  } else {
    // Stack is now empty - remove it
    updatedStacks.delete(posKey);
  }

  const updatedBoard: BoardState = {
    ...board,
    stacks: updatedStacks,
    markers: new Map(board.markers),
    collapsedSpaces: new Map(board.collapsedSpaces),
    territories: new Map(board.territories),
    formedLines: [...board.formedLines],
    eliminatedRings: updatedEliminatedRings,
  };

  // Create audit event
  const auditEvent: EliminationAuditEvent = {
    timestamp: new Date(),
    context,
    ...(reason !== undefined && { reason }),
    player,
    stackPosition,
    ringsEliminated: ringsToEliminate,
    stackHeightBefore: stack.stackHeight,
    stackHeightAfter: remainingRings.length,
    capHeightBefore: calculateCapHeight(stack.rings),
    controllingPlayerBefore: stack.controllingPlayer,
    controllingPlayerAfter: updatedStack?.controllingPlayer ?? null,
  };

  // Emit audit event if enabled
  emitEliminationAuditEvent(auditEvent);

  return {
    success: true,
    ringsEliminated: ringsToEliminate,
    updatedBoard,
    updatedStack,
    auditEvent,
  };
}

/**
 * Enumerate all eligible stacks for elimination in a given context.
 *
 * @param board - Current board state
 * @param player - Player performing elimination
 * @param context - Elimination context
 * @param excludePositions - Positions to exclude (e.g., region being processed)
 * @returns Array of eligible stacks
 */
export function enumerateEligibleStacks(
  board: BoardState,
  player: number,
  context: EliminationContext,
  excludePositions?: Set<string>
): RingStack[] {
  const eligible: RingStack[] = [];

  for (const [posKey, stack] of board.stacks.entries()) {
    // Skip excluded positions (e.g., stacks inside territory region)
    if (excludePositions?.has(posKey)) {
      continue;
    }

    const eligibility = isStackEligibleForElimination(stack, context, player);
    if (eligibility.eligible) {
      eligible.push(stack);
    }
  }

  return eligible;
}

/**
 * Check if player has any eligible elimination targets for a given context.
 *
 * @param board - Current board state
 * @param player - Player to check
 * @param context - Elimination context
 * @param excludePositions - Positions to exclude
 * @returns True if player has at least one eligible target
 */
export function hasEligibleEliminationTarget(
  board: BoardState,
  player: number,
  context: EliminationContext,
  excludePositions?: Set<string>
): boolean {
  for (const [posKey, stack] of board.stacks.entries()) {
    if (excludePositions?.has(posKey)) {
      continue;
    }

    const eligibility = isStackEligibleForElimination(stack, context, player);
    if (eligibility.eligible) {
      return true;
    }
  }
  return false;
}

// ═══════════════════════════════════════════════════════════════════════════
// AUDIT LOGGING
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Emit an elimination audit event.
 *
 * Events are only emitted when RINGRIFT_ELIMINATION_AUDIT env var is set.
 */
function emitEliminationAuditEvent(event: EliminationAuditEvent): void {
  if (typeof process !== 'undefined' && process.env && process.env.RINGRIFT_ELIMINATION_AUDIT) {
    // eslint-disable-next-line no-console
    console.log('[ELIMINATION_AUDIT]', JSON.stringify(event));
  }
}
