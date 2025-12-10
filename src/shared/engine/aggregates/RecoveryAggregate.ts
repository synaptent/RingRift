/**
 * ===============================================================================
 * RecoveryAggregate - Recovery Action Domain
 * ===============================================================================
 *
 * This aggregate handles recovery action validation, enumeration, and mutation
 * for temporarily eliminated players who can slide a marker to complete a line.
 *
 * Rule Reference: RR-CANON-R110–R115 (Recovery Action)
 *
 * Key Rules:
 * - RR-CANON-R110: Recovery eligibility (no stacks, no rings in hand, has markers, has buried rings)
 * - RR-CANON-R111: Marker slide adjacency (Moore for square, hex-adjacency for hex)
 * - RR-CANON-R112: Line requirement (at least lineLength markers), overlength allowed with Option 1/2
 * - RR-CANON-R113: Buried ring extraction cost:
 *   - Option 1 (collapse all): 1 buried ring extraction
 *   - Option 2 (collapse lineLength, overlength only): 0 (free)
 * - RR-CANON-R114: Cascade processing (territory regions after line collapse)
 * - RR-CANON-R115: Recording semantics (recovery_slide move type)
 *
 * Cost Model (Option 1 / Option 2):
 * - Exact-length lines: Always collapse all markers, cost = 1 buried ring
 * - Overlength lines: Player chooses:
 *   - Option 1: Collapse all markers, cost = 1 buried ring
 *   - Option 2: Collapse exactly lineLength consecutive markers, cost = 0 (free)
 *
 * Design principles:
 * - Pure functions: No side effects, return new state
 * - Type safety: Full TypeScript typing
 * - Consistent with other aggregates
 */

import type { GameState, BoardState, Position, Move, LineInfo, RingStack } from '../../types/game';
import { BOARD_CONFIGS, positionToString, stringToPosition } from '../../types/game';

import { getEffectiveLineLengthThreshold } from '../rulesConfig';
import { isEligibleForRecovery, countBuriedRings } from '../playerStateHelpers';
import { calculateCapHeight } from '../core';

// ===============================================================================
// Types
// ===============================================================================

/**
 * Recovery option type.
 * - Option 1: Collapse all markers in the line, pay 1 buried ring
 * - Option 2: Collapse exactly lineLength consecutive markers, pay 0 (free, overlength only)
 */
export type RecoveryOption = 1 | 2;

/**
 * Recovery mode type - which success criterion was met (RR-CANON-R112).
 * - "line": Condition (a) - completes a line of at least lineLength markers
 * - "fallback": Condition (b) - no line available, any adjacent slide that doesn't disconnect territory
 * Note: Territory disconnection is NOT a valid recovery criterion.
 */
export type RecoveryMode = 'line' | 'fallback';

/**
 * A valid recovery slide move.
 */
export interface RecoverySlideMove extends Move {
  type: 'recovery_slide';
  player: number;
  /** Source marker position */
  from: Position;
  /** Adjacent destination (empty cell) */
  to: Position;
  /**
   * Which recovery criterion was satisfied (RR-CANON-R112).
   * - "line": Completed a line of lineLength markers
   * - "fallback": No line available, repositioning without territory disconnection
   */
  recoveryMode?: RecoveryMode;
  /**
   * Which option to use for overlength lines (line mode only).
   * - Option 1: Collapse all markers, pay 1 buried ring
   * - Option 2: Collapse exactly lineLength markers, pay 0 (free)
   * Required for overlength lines. For exact-length lines, this is ignored (always Option 1).
   */
  option?: RecoveryOption;
  /**
   * For Option 2: which consecutive markers to collapse (exactly lineLength positions).
   * Must include the destination position. Required when option = 2.
   */
  collapsePositions?: Position[];
  /**
   * Stacks from which to extract buried rings for self-elimination cost.
   * - For line mode Option 1: Length must be 1
   * - For line mode Option 2: Length must be 0 (empty array)
   * - For fallback mode: Length must be 1
   * Each string is a position key (e.g., "3,4").
   */
  extractionStacks: string[];
}

/**
 * A potential recovery slide target (before option and extraction stacks are chosen).
 */
export interface RecoverySlideTarget {
  /** Source marker position */
  from: Position;
  /** Adjacent destination */
  to: Position;
  /** Length of the line that would be formed */
  formedLineLength: number;
  /** Whether this is an overlength line (> lineLength) */
  isOverlength: boolean;
  /** Cost for Option 1 (collapse all): always 1 */
  option1Cost: 1;
  /** Whether Option 2 is available (only for overlength lines) */
  option2Available: boolean;
  /** Cost for Option 2 (collapse lineLength): always 0 */
  option2Cost: 0;
  /** Positions of all markers in the formed line */
  linePositions: Position[];
  /**
   * @deprecated Use option1Cost or option2Cost instead.
   * Kept for backwards compatibility during transition.
   */
  cost: number;
}

/**
 * Result of applying a recovery slide.
 */
export interface RecoveryApplicationOutcome {
  /** Updated game state after recovery */
  nextState: GameState;
  /** The line that was formed (undefined for fallback mode) */
  formedLine: LineInfo | undefined;
  /** Which markers were actually collapsed (all for Option 1, lineLength for Option 2, empty for fallback) */
  collapsedPositions: Position[];
  /** Which option was used (1 or 2, undefined for fallback mode) */
  optionUsed: RecoveryOption | undefined;
  /** Number of buried rings extracted (1 for Option 1/fallback, 0 for Option 2) */
  extractionCount: number;
  /** Territory spaces gained */
  territoryGained: number;
}

/**
 * Validation result for recovery moves.
 */
export interface RecoveryValidationResult {
  valid: boolean;
  reason?: string;
  code?: string;
}

// ===============================================================================
// Enumeration
// ===============================================================================

/**
 * Enumerate all valid recovery slide targets for a player.
 *
 * Returns targets without extraction stack selection - the caller must
 * choose which option and extraction stacks when constructing the full move.
 *
 * Cost Model (Option 1 / Option 2):
 * - Exact-length lines: Only Option 1 available (cost = 1 buried ring)
 * - Overlength lines: Player chooses:
 *   - Option 1: Collapse all markers, cost = 1 buried ring
 *   - Option 2: Collapse exactly lineLength markers, cost = 0 (free)
 *
 * @param state - Current game state
 * @param playerNumber - Player to enumerate recovery moves for
 * @returns Array of valid recovery slide targets
 */
export function enumerateRecoverySlideTargets(
  state: GameState,
  playerNumber: number
): RecoverySlideTarget[] {
  // Check eligibility first
  if (!isEligibleForRecovery(state, playerNumber)) {
    return [];
  }

  const lineLength = getEffectiveLineLengthThreshold(state.board.type, state.players.length);
  const buriedRingCount = countBuriedRings(state.board, playerNumber);
  const targets: RecoverySlideTarget[] = [];

  // Get adjacency directions based on board type
  const directions = getAdjacencyDirections(state.board.type);

  // For each marker owned by the player
  for (const [posKey, marker] of state.board.markers) {
    if (marker.player !== playerNumber) continue;

    const fromPos = stringToPosition(posKey);

    // Check each adjacent direction
    for (const dir of directions) {
      const toPos = addPositions(fromPos, dir);

      // Must be valid position
      if (!isValidPosition(toPos, state.board)) continue;

      // Must be empty (no stack, no marker, not collapsed)
      if (getStack(toPos, state.board)) continue;
      if (getMarker(toPos, state.board) !== undefined) continue;
      if (isCollapsedSpace(toPos, state.board)) continue;

      // Would this slide complete a line of at least lineLength?
      const lineInfo = getFormedLineInfo(state.board, fromPos, toPos, playerNumber);
      const formedLineLength = lineInfo.length;

      if (formedLineLength >= lineLength) {
        const isOverlength = formedLineLength > lineLength;

        // For exact-length: need 1 buried ring (Option 1 only)
        // For overlength: Option 2 is free, so always legal
        const canUseOption1 = buriedRingCount >= 1;
        const canUseOption2 = isOverlength; // Option 2 is free but only available for overlength

        // At least one option must be available
        if (canUseOption1 || canUseOption2) {
          targets.push({
            from: fromPos,
            to: toPos,
            formedLineLength,
            isOverlength,
            option1Cost: 1,
            option2Available: isOverlength,
            option2Cost: 0,
            linePositions: lineInfo.positions,
            // Deprecated: keep for backwards compatibility
            cost: 1, // Minimum cost is always 1 for Option 1
          });
        }
      }
    }
  }

  return targets;
}

/**
 * Check if a player has any valid recovery moves.
 *
 * This is a quick check for LPS purposes - returns true if at least one
 * recovery slide is available.
 *
 * Uses the expanded recovery criteria (RR-CANON-R112):
 * (a) Line formation - completes a line of lineLength markers
 * (b) Fallback repositioning - if no line available, any slide that doesn't
 *     cause territory disconnection
 *
 * Note: Territory disconnection is NOT a valid recovery criterion.
 *
 * @param state - Current game state
 * @param playerNumber - Player to check
 * @returns True if at least one recovery move is available
 */
export function hasAnyRecoveryMove(state: GameState, playerNumber: number): boolean {
  // Quick eligibility check first
  if (!isEligibleForRecovery(state, playerNumber)) {
    return false;
  }

  const buriedRingCount = countBuriedRings(state.board, playerNumber);
  if (buriedRingCount < 1) {
    return false;
  }

  // Use early-exit enumeration
  const lineLength = getEffectiveLineLengthThreshold(state.board.type, state.players.length);
  const directions = getAdjacencyDirections(state.board.type);
  let validFallbackExists = false;

  for (const [posKey, marker] of state.board.markers) {
    if (marker.player !== playerNumber) continue;

    const fromPos = stringToPosition(posKey);

    for (const dir of directions) {
      const toPos = addPositions(fromPos, dir);

      if (!isValidPosition(toPos, state.board)) continue;
      if (getStack(toPos, state.board)) continue;
      if (getMarker(toPos, state.board) !== undefined) continue;
      if (isCollapsedSpace(toPos, state.board)) continue;

      const lineInfo = getFormedLineInfo(state.board, fromPos, toPos, playerNumber);
      const formedLineLength = lineInfo.length;

      if (formedLineLength >= lineLength) {
        // Line recovery found
        return true;
      }

      // Check if this could be a valid fallback (no territory disconnect)
      // TODO: Add territory disconnection check when implemented
      // For now, we assume all non-line slides are valid fallbacks
      if (!validFallbackExists) {
        validFallbackExists = true;
      }
    }
  }

  // If no line recovery found, but valid fallback exists
  return validFallbackExists;
}

/**
 * Information about an eligible extraction stack.
 */
export interface EligibleExtractionStack {
  /** Position key of the stack (e.g., "3,4") */
  positionKey: string;
  /** Position of the stack */
  position: Position;
  /** Index of the bottommost buried ring in the stack.rings array */
  bottomRingIndex: number;
  /** Total stack height */
  stackHeight: number;
  /** Current controlling player (top ring owner) */
  controllingPlayer: number;
}

/**
 * Enumerate all stacks from which a player can extract a buried ring.
 *
 * Per RR-CANON-R113: The player chooses which stack to extract from if
 * multiple stacks contain their buried rings. The bottommost ring from
 * the chosen stack is extracted.
 *
 * A stack is eligible if:
 * 1. It contains at least one of the player's rings
 * 2. At least one of those rings is buried (not the top ring)
 *
 * @param board - Current board state
 * @param playerNumber - Player seeking extraction
 * @returns Array of eligible extraction stacks with metadata
 */
export function enumerateEligibleExtractionStacks(
  board: BoardState,
  playerNumber: number
): EligibleExtractionStack[] {
  const eligibleStacks: EligibleExtractionStack[] = [];

  for (const [posKey, stack] of board.stacks) {
    // Find the bottommost ring of this player
    // Per game.ts:283, rings[0] = top, rings[length-1] = bottom
    // Use lastIndexOf to find the bottommost occurrence (highest index)
    const bottomRingIndex = stack.rings.lastIndexOf(playerNumber);

    // Player has no ring in this stack
    if (bottomRingIndex === -1) continue;

    // Check if it's buried (not the top ring)
    // Per game.ts:283, rings[0] is the top ring
    const isTopRing = bottomRingIndex === 0;
    if (isTopRing) continue; // Not buried, cannot extract

    eligibleStacks.push({
      positionKey: posKey,
      position: stringToPosition(posKey),
      bottomRingIndex,
      stackHeight: stack.stackHeight,
      controllingPlayer: stack.controllingPlayer,
    });
  }

  return eligibleStacks;
}

/**
 * Calculate the cost of a recovery slide for a given option.
 *
 * Cost Model (Option 1 / Option 2):
 * - Option 1 (collapse all): 1 buried ring extraction
 * - Option 2 (collapse lineLength, overlength only): 0 (free)
 *
 * @param option - Which option (1 or 2)
 * @returns Number of buried rings required
 */
export function calculateRecoveryCost(option: RecoveryOption): number {
  return option === 1 ? 1 : 0;
}

// ===============================================================================
// Validation
// ===============================================================================

/**
 * Validate a recovery slide move.
 *
 * @param state - Current game state
 * @param move - Move to validate
 * @returns Validation result
 */
export function validateRecoverySlide(
  state: GameState,
  move: RecoverySlideMove
): RecoveryValidationResult {
  const { player, from, to, option, collapsePositions, extractionStacks } = move;
  const buriedRingCount = countBuriedRings(state.board, player);

  // Check eligibility
  if (!isEligibleForRecovery(state, player)) {
    return {
      valid: false,
      reason: 'Player is not eligible for recovery action',
      code: 'RECOVERY_NOT_ELIGIBLE',
    };
  }

  // Check from position has player's marker
  const fromKey = positionToString(from);
  const marker = state.board.markers.get(fromKey);
  if (!marker || marker.player !== player) {
    return {
      valid: false,
      reason: 'No marker at source position',
      code: 'RECOVERY_NO_MARKER_AT_SOURCE',
    };
  }

  // Check to position is adjacent
  if (!isAdjacent(from, to, state.board.type)) {
    return {
      valid: false,
      reason: 'Destination is not adjacent to source',
      code: 'RECOVERY_NOT_ADJACENT',
    };
  }

  // Check to position is empty
  if (getStack(to, state.board)) {
    return {
      valid: false,
      reason: 'Destination has a stack',
      code: 'RECOVERY_DEST_HAS_STACK',
    };
  }
  if (getMarker(to, state.board) !== undefined) {
    return {
      valid: false,
      reason: 'Destination has a marker',
      code: 'RECOVERY_DEST_HAS_MARKER',
    };
  }
  if (isCollapsedSpace(to, state.board)) {
    return {
      valid: false,
      reason: 'Destination is collapsed space',
      code: 'RECOVERY_DEST_COLLAPSED',
    };
  }

  // Check for fallback mode (RR-CANON-R112(b))
  // Fallback slides bypass line formation check but still cost 1 buried ring
  const { recoveryMode } = move;
  if (recoveryMode === 'fallback') {
    // Fallback mode: no line required, but need buried rings for cost
    // Per RR-CANON-R112(b): fallback slides must NOT cause territory disconnection
    // The move was generated with this check, so we trust it was valid when created
    // Validate extraction stacks for fallback cost (1 buried ring)
    if (extractionStacks.length !== 1) {
      return {
        valid: false,
        reason: `Fallback recovery requires exactly 1 extraction stack, got ${extractionStacks.length}`,
        code: 'RECOVERY_WRONG_EXTRACTION_COUNT',
      };
    }
    if (buriedRingCount < 1) {
      return {
        valid: false,
        reason: 'Fallback recovery requires at least 1 buried ring',
        code: 'RECOVERY_INSUFFICIENT_BURIED_RINGS',
      };
    }
    // Validate the extraction stack
    const stackKey = extractionStacks[0];
    const stack = state.board.stacks.get(stackKey);
    if (!stack) {
      return {
        valid: false,
        reason: `No stack at extraction position ${stackKey}`,
        code: 'RECOVERY_INVALID_EXTRACTION_STACK',
      };
    }
    const hasBuriedRing = stack.rings
      .slice(1) // All except top (rings[0] is top)
      .some((ringPlayer) => ringPlayer === player);
    if (!hasBuriedRing) {
      return {
        valid: false,
        reason: `No buried ring for player ${player} at ${stackKey}`,
        code: 'RECOVERY_NO_BURIED_RING_IN_STACK',
      };
    }
    return { valid: true };
  }

  // Line mode: Check line formation (RR-CANON-R112(a))
  const lineLength = getEffectiveLineLengthThreshold(state.board.type, state.players.length);
  const lineInfo = getFormedLineInfo(state.board, from, to, player);
  const formedLineLength = lineInfo.length;

  if (formedLineLength < lineLength) {
    return {
      valid: false,
      reason: `Slide does not form a line of at least ${lineLength} markers`,
      code: 'RECOVERY_INSUFFICIENT_LINE',
    };
  }

  const isOverlength = formedLineLength > lineLength;

  // Determine effective option
  // For exact-length lines, option is ignored (always Option 1)
  // For overlength lines, option must be specified
  let effectiveOption: RecoveryOption = 1;
  if (isOverlength) {
    if (option === undefined) {
      return {
        valid: false,
        reason: 'Overlength line requires option (1 or 2) to be specified',
        code: 'RECOVERY_OPTION_REQUIRED',
      };
    }
    effectiveOption = option;
  }

  // Validate Option 2 specific requirements
  if (effectiveOption === 2) {
    if (!collapsePositions) {
      return {
        valid: false,
        reason: 'Option 2 requires collapsePositions to be specified',
        code: 'RECOVERY_COLLAPSE_POSITIONS_REQUIRED',
      };
    }

    if (collapsePositions.length !== lineLength) {
      return {
        valid: false,
        reason: `Option 2 requires exactly ${lineLength} collapse positions, got ${collapsePositions.length}`,
        code: 'RECOVERY_WRONG_COLLAPSE_COUNT',
      };
    }

    // Verify collapsePositions are consecutive and part of the formed line
    const validationResult = validateCollapsePositions(
      collapsePositions,
      lineInfo.positions,
      to,
      state.board.type
    );
    if (!validationResult.valid) {
      return validationResult;
    }
  }

  // Check extraction cost matches selected option
  const expectedCost = calculateRecoveryCost(effectiveOption);
  if (expectedCost > 0 && buriedRingCount < expectedCost) {
    return {
      valid: false,
      reason: 'Not enough buried rings to pay recovery cost',
      code: 'RECOVERY_INSUFFICIENT_BURIED_RINGS',
    };
  }
  if (extractionStacks.length !== expectedCost) {
    return {
      valid: false,
      reason: `Option ${effectiveOption} requires ${expectedCost} extractions, got ${extractionStacks.length}`,
      code: 'RECOVERY_WRONG_EXTRACTION_COUNT',
    };
  }

  // Validate each extraction stack (for Option 1)
  for (const stackKey of extractionStacks) {
    const stack = state.board.stacks.get(stackKey);
    if (!stack) {
      return {
        valid: false,
        reason: `No stack at extraction position ${stackKey}`,
        code: 'RECOVERY_INVALID_EXTRACTION_STACK',
      };
    }

    // Check stack has player's buried ring
    const hasBuriedRing = stack.rings
      .slice(1) // All except top (rings[0] is top)
      .includes(player);
    if (!hasBuriedRing) {
      return {
        valid: false,
        reason: `Stack at ${stackKey} has no buried ring of player ${player}`,
        code: 'RECOVERY_NO_BURIED_RING_IN_STACK',
      };
    }
  }

  return { valid: true };
}

/**
 * Validate that collapse positions are consecutive and part of the formed line.
 */
function validateCollapsePositions(
  collapsePositions: Position[],
  linePositions: Position[],
  destinationPos: Position,
  boardType: string
): RecoveryValidationResult {
  // All collapse positions must be in the formed line
  const lineKeys = new Set(linePositions.map(positionToString));
  for (const pos of collapsePositions) {
    const key = positionToString(pos);
    if (!lineKeys.has(key)) {
      return {
        valid: false,
        reason: `Collapse position ${key} is not part of the formed line`,
        code: 'RECOVERY_INVALID_COLLAPSE_POSITION',
      };
    }
  }

  // Destination must be included in collapse positions
  const destKey = positionToString(destinationPos);
  if (!collapsePositions.some((p) => positionToString(p) === destKey)) {
    return {
      valid: false,
      reason: 'Collapse positions must include the destination position',
      code: 'RECOVERY_DEST_NOT_IN_COLLAPSE',
    };
  }

  // Collapse positions must be consecutive
  // Sort by position along the line direction
  const sortedCollapse = [...collapsePositions].sort((a, b) => {
    const lineStart = linePositions[0];
    const distA = Math.abs(a.x - lineStart.x) + Math.abs(a.y - lineStart.y);
    const distB = Math.abs(b.x - lineStart.x) + Math.abs(b.y - lineStart.y);
    return distA - distB;
  });

  // Verify adjacency between consecutive positions
  for (let i = 0; i < sortedCollapse.length - 1; i++) {
    if (!isAdjacent(sortedCollapse[i], sortedCollapse[i + 1], boardType)) {
      return {
        valid: false,
        reason: 'Collapse positions must be consecutive',
        code: 'RECOVERY_COLLAPSE_NOT_CONSECUTIVE',
      };
    }
  }

  return { valid: true };
}

// ===============================================================================
// Application
// ===============================================================================

/**
 * Apply a recovery slide move to the game state.
 *
 * This function:
 * 1. Moves the marker from -> to
 * 2. Detects the formed line
 * 3. Collapses markers based on selected option:
 *    - Option 1: Collapse all markers in the line
 *    - Option 2: Collapse only the specified lineLength consecutive markers
 * 4. Extracts buried rings as self-elimination cost (Option 1 only)
 * 5. Updates territory and eliminated ring counts
 *
 * Note: Territory cascade (disconnected regions) is NOT handled here.
 * That should be handled by the turn orchestrator after this function returns.
 *
 * @param state - Current game state
 * @param move - Recovery slide move to apply
 * @returns Application outcome with new state
 */
export function applyRecoverySlide(
  state: GameState,
  move: RecoverySlideMove
): RecoveryApplicationOutcome {
  const { player, from, to, option, collapsePositions, extractionStacks, recoveryMode } = move;

  // Clone state for mutation
  const nextState = cloneGameState(state);
  const board = nextState.board;

  // 1. Move the marker from -> to
  const fromKey = positionToString(from);
  const toKey = positionToString(to);

  board.markers.delete(fromKey);
  board.markers.set(toKey, {
    player,
    position: to,
    type: 'regular',
  });

  // Get player state for updates
  const playerState = nextState.players.find((p) => p.playerNumber === player);

  // Check for fallback mode (RR-CANON-R112(b))
  // Fallback mode: no line collapse, just marker move + ring extraction
  if (recoveryMode === 'fallback') {
    // Extract 1 buried ring (fallback cost)
    let extractionCount = 0;
    for (const stackKey of extractionStacks) {
      const stack = board.stacks.get(stackKey);
      if (!stack) continue;

      // Find and remove player's bottommost ring per RR-CANON-R113
      // rings[0] is top, so lastIndexOf finds the bottommost occurrence
      const ringIndex = stack.rings.lastIndexOf(player);
      if (ringIndex === -1) continue;

      // Remove the ring
      stack.rings.splice(ringIndex, 1);
      stack.stackHeight--;
      extractionCount++;

      // Update player's eliminated rings
      if (playerState) {
        playerState.eliminatedRings++;
      }

      // Update stack control if needed
      if (stack.rings.length === 0) {
        board.stacks.delete(stackKey);
      } else {
        // rings[0] is the top ring per game.ts convention
        stack.controllingPlayer = stack.rings[0];
        stack.capHeight = calculateCapHeight(stack.rings);
      }
    }

    return {
      nextState,
      formedLine: undefined,
      collapsedPositions: [],
      optionUsed: undefined,
      extractionCount,
      territoryGained: 0,
    };
  }

  // Line mode: detect and collapse the formed line
  const lineLength = getEffectiveLineLengthThreshold(board.type, nextState.players.length);
  const formedLine = detectFormedLine(board, to, player);
  if (!formedLine || formedLine.length < lineLength) {
    throw new Error('Recovery slide did not form a valid line');
  }

  const isOverlength = formedLine.length > lineLength;

  // Determine effective option and collapse positions
  let effectiveOption: RecoveryOption = 1;
  let positionsToCollapse: Position[];

  if (isOverlength && option === 2) {
    // Option 2: Collapse only the specified positions
    effectiveOption = 2;
    if (!collapsePositions || collapsePositions.length !== lineLength) {
      throw new Error(
        `Option 2 requires exactly ${lineLength} collapse positions, got ${collapsePositions?.length ?? 0}`
      );
    }
    positionsToCollapse = collapsePositions;
  } else {
    // Option 1 (or exact-length): Collapse all markers in the line
    effectiveOption = 1;
    positionsToCollapse = formedLine.positions;
  }

  // 3. Collapse markers - selected markers become collapsed spaces (territory)
  for (const pos of positionsToCollapse) {
    const posKey = positionToString(pos);
    board.markers.delete(posKey);
    board.collapsedSpaces.set(posKey, player);
  }

  // Update player's territory count
  if (playerState) {
    playerState.territorySpaces += positionsToCollapse.length;
  }

  // 4. Extract buried rings (self-elimination cost) - Option 1 only
  //    Per RR-CANON-R113: Extract the bottommost ring from the chosen stack
  let extractionCount = 0;
  if (effectiveOption === 1) {
    for (const stackKey of extractionStacks) {
      const stack = board.stacks.get(stackKey);
      if (!stack) continue;

      // Find and remove player's bottommost ring per RR-CANON-R113
      // rings[0] is top, so lastIndexOf finds the bottommost occurrence
      const ringIndex = stack.rings.lastIndexOf(player);
      if (ringIndex === -1) continue;

      // Remove the ring
      stack.rings.splice(ringIndex, 1);
      stack.stackHeight--;
      extractionCount++;

      // Update player's eliminated rings
      if (playerState) {
        playerState.eliminatedRings++;
      }

      // Update stack control if needed
      if (stack.rings.length === 0) {
        // Stack is now empty, remove it
        board.stacks.delete(stackKey);
      } else {
        // Update controlling player - rings[0] is the top ring per game.ts convention
        stack.controllingPlayer = stack.rings[0];
        // Recalculate cap height
        stack.capHeight = calculateCapHeight(stack.rings);
      }
    }
  }

  return {
    nextState,
    formedLine,
    collapsedPositions: positionsToCollapse,
    optionUsed: effectiveOption,
    extractionCount,
    territoryGained: positionsToCollapse.length,
  };
}

// ===============================================================================
// Helpers
// ===============================================================================

/**
 * Get adjacency directions based on board type.
 * Square boards use Moore neighborhood (8 directions).
 * Hex boards use 6 hex-adjacent directions.
 */
function getAdjacencyDirections(boardType: string): Position[] {
  if (boardType === 'hexagonal') {
    // 6 hex-adjacent directions
    return [
      { x: 1, y: 0, z: -1 },
      { x: 1, y: -1, z: 0 },
      { x: 0, y: -1, z: 1 },
      { x: -1, y: 0, z: 1 },
      { x: -1, y: 1, z: 0 },
      { x: 0, y: 1, z: -1 },
    ];
  } else {
    // Moore neighborhood (8 directions) for square boards
    return [
      { x: 1, y: 0 },
      { x: 1, y: 1 },
      { x: 0, y: 1 },
      { x: -1, y: 1 },
      { x: -1, y: 0 },
      { x: -1, y: -1 },
      { x: 0, y: -1 },
      { x: 1, y: -1 },
    ];
  }
}

/**
 * Check if two positions are adjacent.
 */
function isAdjacent(from: Position, to: Position, boardType: string): boolean {
  const directions = getAdjacencyDirections(boardType);
  return directions.some(
    (d) =>
      from.x + d.x === to.x &&
      from.y + d.y === to.y &&
      (boardType !== 'hexagonal' || (from.z || 0) + (d.z || 0) === (to.z || 0))
  );
}

/**
 * Add two positions.
 */
function addPositions(a: Position, b: Position): Position {
  const result: Position = {
    x: a.x + b.x,
    y: a.y + b.y,
  };
  // Only include z if both positions have it defined
  if (a.z !== undefined && b.z !== undefined) {
    result.z = a.z + b.z;
  }
  return result;
}

/**
 * Check if a position is valid on the board.
 */
function isValidPosition(position: Position, board: BoardState): boolean {
  const size = board.size;
  if (board.type === 'hexagonal') {
    const radius = size - 1;
    const q = position.x;
    const r = position.y;
    const s = position.z || -q - r;
    return (
      Math.abs(q) <= radius && Math.abs(r) <= radius && Math.abs(s) <= radius && q + r + s === 0
    );
  } else {
    return position.x >= 0 && position.x < size && position.y >= 0 && position.y < size;
  }
}

/**
 * Get marker at position.
 */
function getMarker(position: Position, board: BoardState): number | undefined {
  const posKey = positionToString(position);
  return board.markers.get(posKey)?.player;
}

/**
 * Get stack at position.
 */
function getStack(position: Position, board: BoardState): RingStack | undefined {
  const posKey = positionToString(position);
  return board.stacks.get(posKey);
}

/**
 * Check if position is a collapsed space.
 */
function isCollapsedSpace(position: Position, board: BoardState): boolean {
  const posKey = positionToString(position);
  return board.collapsedSpaces.has(posKey);
}

/**
 * Result of line formation detection.
 */
interface FormedLineInfo {
  /** Length of the longest line formed */
  length: number;
  /** Positions of all markers in the longest line */
  positions: Position[];
  /** Direction of the line */
  direction: Position;
}

/**
 * Get information about the line that would be formed by sliding
 * a marker from `from` to `to`.
 *
 * Simulates the marker move and detects lines containing the new position.
 * Returns the longest line found with its positions.
 */
function getFormedLineInfo(
  board: BoardState,
  from: Position,
  to: Position,
  player: number
): FormedLineInfo {
  // Simulate the marker move on a temporary copy
  const tempMarkers = new Map(board.markers);
  const fromKey = positionToString(from);
  const toKey = positionToString(to);

  tempMarkers.delete(fromKey);
  tempMarkers.set(toKey, {
    player,
    position: to,
    type: 'regular' as const,
  });

  // Create temporary board view
  const tempBoard: BoardState = {
    ...board,
    markers: tempMarkers,
  };

  // Find lines containing the new position
  const directions = getLineDirections(board.type);
  let bestLine: FormedLineInfo = {
    length: 0,
    positions: [],
    direction: { x: 0, y: 0 },
  };

  for (const direction of directions) {
    const positions = collectLinePositionsInDirection(to, direction, player, tempBoard);
    if (positions.length > bestLine.length) {
      bestLine = {
        length: positions.length,
        positions,
        direction,
      };
    }
  }

  return bestLine;
}

/**
 * Collect all positions in a line from a starting point in both directions.
 */
function collectLinePositionsInDirection(
  start: Position,
  direction: Position,
  player: number,
  board: BoardState
): Position[] {
  const positions: Position[] = [start];

  // Check forward
  let current = start;
  while (true) {
    const next = addPositions(current, direction);
    if (!isValidPosition(next, board)) break;
    const markerPlayer = getMarker(next, board);
    if (markerPlayer !== player) break;
    if (isCollapsedSpace(next, board) || getStack(next, board)) break;
    positions.push(next);
    current = next;
  }

  // Check backward
  const reverseDir: Position = {
    x: -direction.x,
    y: -direction.y,
  };
  if (direction.z !== undefined) {
    reverseDir.z = -direction.z;
  }
  current = start;
  while (true) {
    const prev = addPositions(current, reverseDir);
    if (!isValidPosition(prev, board)) break;
    const markerPlayer = getMarker(prev, board);
    if (markerPlayer !== player) break;
    if (isCollapsedSpace(prev, board) || getStack(prev, board)) break;
    positions.unshift(prev);
    current = prev;
  }

  return positions;
}

/**
 * Get line directions for line detection.
 * Different from adjacency - only need 4 directions (or 3 for hex)
 * since we check both directions from each point.
 */
function getLineDirections(boardType: string): Position[] {
  if (boardType === 'hexagonal') {
    return [
      { x: 1, y: 0, z: -1 },
      { x: 1, y: -1, z: 0 },
      { x: 0, y: -1, z: 1 },
    ];
  } else {
    return [
      { x: 1, y: 0 }, // Horizontal
      { x: 0, y: 1 }, // Vertical
      { x: 1, y: 1 }, // Diagonal SE
      { x: 1, y: -1 }, // Diagonal NE
    ];
  }
}

/**
 * Detect the formed line at a position after marker placement.
 */
function detectFormedLine(board: BoardState, position: Position, player: number): LineInfo | null {
  const directions = getLineDirections(board.type);
  const config = BOARD_CONFIGS[board.type];

  for (const direction of directions) {
    const positions = collectLinePositions(position, direction, player, board);
    if (positions.length >= config.lineLength) {
      return {
        positions,
        player,
        length: positions.length,
        direction,
      };
    }
  }

  return null;
}

/**
 * Collect all positions in a line from a starting point.
 */
function collectLinePositions(
  start: Position,
  direction: Position,
  player: number,
  board: BoardState
): Position[] {
  const positions: Position[] = [start];

  // Forward
  let current = start;
  while (true) {
    const next = addPositions(current, direction);
    if (!isValidPosition(next, board)) break;
    if (getMarker(next, board) !== player) break;
    if (isCollapsedSpace(next, board) || getStack(next, board)) break;
    positions.push(next);
    current = next;
  }

  // Backward
  const reverseDir: Position = {
    x: -direction.x,
    y: -direction.y,
  };
  if (direction.z !== undefined) {
    reverseDir.z = -direction.z;
  }
  current = start;
  while (true) {
    const prev = addPositions(current, reverseDir);
    if (!isValidPosition(prev, board)) break;
    if (getMarker(prev, board) !== player) break;
    if (isCollapsedSpace(prev, board) || getStack(prev, board)) break;
    positions.unshift(prev);
    current = prev;
  }

  return positions;
}

/**
 * Deep clone a game state for mutation.
 */
function cloneGameState(state: GameState): GameState {
  return {
    ...state,
    board: {
      ...state.board,
      stacks: new Map([...state.board.stacks].map(([k, v]) => [k, { ...v, rings: [...v.rings] }])),
      markers: new Map([...state.board.markers].map(([k, v]) => [k, { ...v }])),
      collapsedSpaces: new Map(state.board.collapsedSpaces),
    },
    players: state.players.map((p) => ({ ...p })),
  };
}
