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
 * - RR-CANON-R112: Line requirement (at least lineLength markers)
 * - RR-CANON-R113: Buried ring extraction cost (1 base + 1 per marker beyond lineLength)
 * - RR-CANON-R114: Cascade processing (territory regions after line collapse)
 * - RR-CANON-R115: Recording semantics (recovery_slide move type)
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

// ===============================================================================
// Types
// ===============================================================================

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
   * Stacks from which to extract buried rings for self-elimination cost.
   * Length must equal the total cost (1 + overlength markers).
   * Each string is a position key (e.g., "3,4").
   */
  extractionStacks: string[];
}

/**
 * A potential recovery slide target (before extraction stacks are chosen).
 */
export interface RecoverySlideTarget {
  /** Source marker position */
  from: Position;
  /** Adjacent destination */
  to: Position;
  /** Length of the line that would be formed */
  formedLineLength: number;
  /** Number of buried rings required (1 + overlength) */
  cost: number;
}

/**
 * Result of applying a recovery slide.
 */
export interface RecoveryApplicationOutcome {
  /** Updated game state after recovery */
  nextState: GameState;
  /** The line that was formed and collapsed */
  formedLine: LineInfo;
  /** Number of buried rings extracted */
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
 * choose which stacks to extract from when constructing the full move.
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
      const formedLineLength = getFormedLineLength(state.board, fromPos, toPos, playerNumber);

      if (formedLineLength >= lineLength) {
        // Calculate cost: 1 base + overlength
        const cost = 1 + Math.max(0, formedLineLength - lineLength);

        // Only legal if player has enough buried rings
        if (buriedRingCount >= cost) {
          targets.push({
            from: fromPos,
            to: toPos,
            formedLineLength,
            cost,
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
 * @param state - Current game state
 * @param playerNumber - Player to check
 * @returns True if at least one recovery move is available
 */
export function hasAnyRecoveryMove(state: GameState, playerNumber: number): boolean {
  // Quick eligibility check first
  if (!isEligibleForRecovery(state, playerNumber)) {
    return false;
  }

  // Use early-exit enumeration
  const lineLength = getEffectiveLineLengthThreshold(state.board.type, state.players.length);
  const buriedRingCount = countBuriedRings(state.board, playerNumber);
  const directions = getAdjacencyDirections(state.board.type);

  for (const [posKey, marker] of state.board.markers) {
    if (marker.player !== playerNumber) continue;

    const fromPos = stringToPosition(posKey);

    for (const dir of directions) {
      const toPos = addPositions(fromPos, dir);

      if (!isValidPosition(toPos, state.board)) continue;
      if (getStack(toPos, state.board)) continue;
      if (getMarker(toPos, state.board) !== undefined) continue;
      if (isCollapsedSpace(toPos, state.board)) continue;

      const formedLineLength = getFormedLineLength(state.board, fromPos, toPos, playerNumber);

      if (formedLineLength >= lineLength) {
        const cost = 1 + Math.max(0, formedLineLength - lineLength);
        if (buriedRingCount >= cost) {
          return true; // Early exit
        }
      }
    }
  }

  return false;
}

/**
 * Calculate the cost of a recovery slide.
 *
 * Cost = 1 (base) + max(0, actualLineLength - lineLength)
 *
 * @param lineLength - Required minimum line length for the board/player-count
 * @param actualLineLength - Actual length of the formed line
 * @returns Number of buried rings required
 */
export function calculateRecoveryCost(lineLength: number, actualLineLength: number): number {
  return 1 + Math.max(0, actualLineLength - lineLength);
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
  const { player, from, to, extractionStacks } = move;

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

  // Check line formation
  const lineLength = getEffectiveLineLengthThreshold(state.board.type, state.players.length);
  const formedLineLength = getFormedLineLength(state.board, from, to, player);

  if (formedLineLength < lineLength) {
    return {
      valid: false,
      reason: `Slide does not form a line of at least ${lineLength} markers`,
      code: 'RECOVERY_INSUFFICIENT_LINE',
    };
  }

  // Check extraction cost
  const cost = calculateRecoveryCost(lineLength, formedLineLength);
  if (extractionStacks.length !== cost) {
    return {
      valid: false,
      reason: `Recovery requires ${cost} extractions, got ${extractionStacks.length}`,
      code: 'RECOVERY_WRONG_EXTRACTION_COUNT',
    };
  }

  // Validate each extraction stack
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
      .slice(0, -1) // All except top
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

// ===============================================================================
// Application
// ===============================================================================

/**
 * Apply a recovery slide move to the game state.
 *
 * This function:
 * 1. Moves the marker from -> to
 * 2. Detects and collapses the formed line
 * 3. Extracts buried rings as self-elimination cost
 * 4. Updates territory and eliminated ring counts
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
  const { player, from, to, extractionStacks } = move;

  // Clone state for mutation
  const nextState = cloneGameState(state);
  const board = nextState.board;

  const lineLength = getEffectiveLineLengthThreshold(board.type, nextState.players.length);

  // 1. Move the marker from -> to
  const fromKey = positionToString(from);
  const toKey = positionToString(to);

  board.markers.delete(fromKey);
  board.markers.set(toKey, {
    player,
    position: to,
    type: 'regular',
  });

  // 2. Detect the formed line
  const formedLine = detectFormedLine(board, to, player);
  if (!formedLine || formedLine.length < lineLength) {
    throw new Error('Recovery slide did not form a valid line');
  }

  // 3. Collapse the line - all markers become collapsed spaces (territory)
  for (const pos of formedLine.positions) {
    const posKey = positionToString(pos);
    board.markers.delete(posKey);
    board.collapsedSpaces.set(posKey, player);
  }

  // Update player's territory count
  const playerState = nextState.players.find((p) => p.playerNumber === player);
  if (playerState) {
    playerState.territorySpaces += formedLine.length;
  }

  // 4. Extract buried rings (self-elimination cost)
  let extractionCount = 0;
  for (const stackKey of extractionStacks) {
    const stack = board.stacks.get(stackKey);
    if (!stack) continue;

    // Find and remove player's bottommost ring
    const ringIndex = stack.rings.findIndex((r) => r === player);
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
      // Update controlling player (top ring)
      stack.controllingPlayer = stack.rings[stack.rings.length - 1];
      // Recalculate cap height
      stack.capHeight = calculateCapHeight(stack.rings);
    }
  }

  return {
    nextState,
    formedLine,
    extractionCount,
    territoryGained: formedLine.length,
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
 * Calculate the length of the line that would be formed by sliding
 * a marker from `from` to `to`.
 *
 * Simulates the marker move and detects lines containing the new position.
 */
function getFormedLineLength(
  board: BoardState,
  from: Position,
  to: Position,
  player: number
): number {
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
  let maxLineLength = 0;

  for (const direction of directions) {
    const lineLength = countLineInDirection(to, direction, player, tempBoard);
    maxLineLength = Math.max(maxLineLength, lineLength);
  }

  return maxLineLength;
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
 * Count the length of a line in a direction (both directions from start).
 */
function countLineInDirection(
  start: Position,
  direction: Position,
  player: number,
  board: BoardState
): number {
  let count = 1; // Start position counts

  // Check forward
  let current = start;
  while (true) {
    const next = addPositions(current, direction);
    if (!isValidPosition(next, board)) break;
    const markerPlayer = getMarker(next, board);
    if (markerPlayer !== player) break;
    if (isCollapsedSpace(next, board) || getStack(next, board)) break;
    count++;
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
    count++;
    current = prev;
  }

  return count;
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
 * Calculate cap height for a ring array.
 */
function calculateCapHeight(rings: number[]): number {
  if (rings.length === 0) return 0;
  const topPlayer = rings[rings.length - 1];
  let capHeight = 1;
  for (let i = rings.length - 2; i >= 0; i--) {
    if (rings[i] === topPlayer) {
      capHeight++;
    } else {
      break;
    }
  }
  return capHeight;
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
