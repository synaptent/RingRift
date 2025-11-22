import {
  BoardType,
  Position,
  BoardState,
  GameState,
  ProgressSnapshot,
  BoardSummary,
  positionToString,
} from '../types/game';

/**
 * Shared, browser-safe core helpers for RingRift engine logic.
 *
 * These functions are intentionally pure and depend only on shared
 * types so they can be used by both the Node.js GameEngine wrapper
 * and any future client-side/local-engine harnesses.
 */

/**
 * A simple direction vector in board-local coordinates. For hex boards,
 * directions use cube coordinates (x,y,z) with x + y + z = 0.
 */
export interface Direction {
  x: number;
  y: number;
  z?: number;
}

/**
 * Canonical 8-direction Moore neighborhood for square boards.
 */
export const SQUARE_MOORE_DIRECTIONS: Direction[] = [
  { x: 1, y: 0 }, // E
  { x: 1, y: 1 }, // SE
  { x: 0, y: 1 }, // S
  { x: -1, y: 1 }, // SW
  { x: -1, y: 0 }, // W
  { x: -1, y: -1 }, // NW
  { x: 0, y: -1 }, // N
  { x: 1, y: -1 }, // NE
];

/**
 * Canonical 6-direction set for hexagonal boards in cube coordinates.
 */
export const HEX_DIRECTIONS: Direction[] = [
  { x: 1, y: 0, z: -1 }, // East
  { x: 0, y: 1, z: -1 }, // Southeast
  { x: -1, y: 1, z: 0 }, // Southwest
  { x: -1, y: 0, z: 1 }, // West
  { x: 0, y: -1, z: 1 }, // Northwest
  { x: 1, y: -1, z: 0 }, // Northeast
];

/**
 * Get canonical movement/capture directions for a given board type.
 *
 * Square boards use 8-direction Moore adjacency; hex boards use the
 * 6 standard cube-coordinate directions.
 */
export function getMovementDirectionsForBoardType(boardType: BoardType): Direction[] {
  if (boardType === 'hexagonal') {
    return HEX_DIRECTIONS;
  }
  return SQUARE_MOORE_DIRECTIONS;
}

/**
 * Calculate cap height for a ring stack.
 *
 * Rule Reference: Section 5.2 - Cap height is consecutive rings of
 * the same color from the top of the stack.
 */
export function calculateCapHeight(rings: number[]): number {
  if (rings.length === 0) return 0;

  const topColor = rings[0];
  let capHeight = 1;

  for (let i = 1; i < rings.length; i++) {
    if (rings[i] === topColor) {
      capHeight++;
    } else {
      break;
    }
  }

  return capHeight;
}

/**
 * Get all positions along a straight-line path between two positions
 * in board-local coordinates. This is used for marker processing and
 * movement path validation.
 */
export function getPathPositions(from: Position, to: Position): Position[] {
  const path: Position[] = [from];

  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const dz = (to.z || 0) - (from.z || 0);

  const steps = Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz));
  const stepX = steps > 0 ? dx / steps : 0;
  const stepY = steps > 0 ? dy / steps : 0;
  const stepZ = steps > 0 ? dz / steps : 0;

  for (let i = 1; i <= steps; i++) {
    const pos: Position = {
      x: Math.round(from.x + stepX * i),
      y: Math.round(from.y + stepY * i),
    };
    if (to.z !== undefined) {
      pos.z = Math.round((from.z || 0) + stepZ * i);
    }
    path.push(pos);
  }

  return path;
}

/**
 * Calculate distance between two positions based on board type.
 * - Square boards: Chebyshev (king-move) distance
 * - Hex boards: cube-coordinate distance
 */
export function calculateDistance(boardType: BoardType, from: Position, to: Position): number {
  if (boardType === 'hexagonal') {
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const dz = (to.z || 0) - (from.z || 0);
    return (Math.abs(dx) + Math.abs(dy) + Math.abs(dz)) / 2;
  }

  const dx = Math.abs(to.x - from.x);
  const dy = Math.abs(to.y - from.y);
  // Chebyshev distance aligns with 8-direction movement: one step per king move.
  const dist = Math.max(dx, dy);
  return dist;
}

/**
 * Minimal, board-agnostic view used for validating capture segments.
 * Implemented by adapters in the server RuleEngine/GameEngine and any
 * future client-side harnesses.
 */
export interface CaptureSegmentBoardView {
  /** True if the position is on the board and addressable. */
  isValidPosition(pos: Position): boolean;
  /** True if this space is a collapsed territory space (cannot move/capture through or land on). */
  isCollapsedSpace(pos: Position): boolean;
  /**
   * Lightweight stack view at a position. Only the controlling player,
   * cap height, and total stack height are required for capture rules.
   */
  getStackAt(pos: Position):
    | {
        controllingPlayer: number;
        capHeight: number;
        stackHeight: number;
      }
    | undefined;
  /** Optional marker lookup for landing-on-own-marker checks. */
  getMarkerOwner?(pos: Position): number | undefined;
}

/**
 * Minimal, board-agnostic view used for movement + capture reachability
 * checks ("does this stack have any legal move or capture?"). This is
 * deliberately similar to CaptureSegmentBoardView but separated to keep
 * responsibilities clear.
 */
export interface MovementBoardView {
  /** True if the position is on the board and addressable. */
  isValidPosition(pos: Position): boolean;
  /** True if this space is a collapsed territory space (cannot move/capture through or land on). */
  isCollapsedSpace(pos: Position): boolean;
  /** Lightweight stack view (controlling player, cap height, total height). */
  getStackAt(pos: Position):
    | {
        controllingPlayer: number;
        capHeight: number;
        stackHeight: number;
      }
    | undefined;
  /** Optional marker lookup used for landing-on-own-marker checks. */
  getMarkerOwner?(pos: Position): number | undefined;
}

/**
 * Shared, rules-aligned validator for a single overtaking capture
 * segment from `from` over `target` to `landing` for `player`.
 *
 * This function is intentionally pure and depends only on a minimal
 * board view so it can be used by:
 * - The server RuleEngine (via a BoardManager adapter)
 * - The server GameEngine
 * - Any client/local harnesses that want to reason about captures
 *
 * Rule References: Sections 10.1, 10.2, FAQ Q3.
 */
export function validateCaptureSegmentOnBoard(
  boardType: BoardType,
  from: Position,
  target: Position,
  landing: Position,
  player: number,
  board: CaptureSegmentBoardView
): boolean {
  const debug = typeof process !== 'undefined' && !!process.env.RINGRIFT_DEBUG_CAPTURE;

  if (
    !board.isValidPosition(from) ||
    !board.isValidPosition(target) ||
    !board.isValidPosition(landing)
  ) {
    if (debug) console.log('Invalid position(s)');
    return false;
  }

  const attacker = board.getStackAt(from);
  if (!attacker || attacker.controllingPlayer !== player) {
    if (debug) console.log('Invalid attacker', attacker, player);
    return false;
  }

  const targetStack = board.getStackAt(target);
  if (!targetStack) {
    if (debug) console.log('No target stack');
    return false;
  }

  // Cap height must be >= target's cap height (Section 10.1)
  if (attacker.capHeight < targetStack.capHeight) {
    if (debug) console.log('Insufficient cap height', attacker.capHeight, targetStack.capHeight);
    return false;
  }

  // Rule fix: Players can overtake their own stacks
  // (No same-player restriction per rules clarification)

  // Direction: must be straight line along a valid axis.
  const dx = target.x - from.x;
  const dy = target.y - from.y;
  const dz = (target.z || 0) - (from.z || 0);

  if (boardType === 'hexagonal') {
    // In cube coordinates, moving along an axis means exactly two
    // coordinates change (the third is implied by x + y + z = 0).
    const coordChanges = [dx !== 0, dy !== 0, dz !== 0].filter(Boolean).length;
    if (coordChanges !== 2) {
      if (debug) console.log('Invalid hex direction');
      return false;
    }
  } else {
    // Square boards: orthogonal or diagonal only.
    if (dx === 0 && dy === 0) {
      if (debug) console.log('Zero movement');
      return false;
    }
    if (dx !== 0 && dy !== 0 && Math.abs(dx) !== Math.abs(dy)) {
      if (debug) console.log('Invalid square direction');
      return false;
    }
  }

  // Path from attacker to target (exclusive) must be clear of stacks
  // and collapsed spaces. Markers are allowed.
  const pathToTarget = getPathPositions(from, target).slice(1, -1);
  for (const pos of pathToTarget) {
    if (!board.isValidPosition(pos)) {
      if (debug) console.log('Invalid path pos', pos);
      return false;
    }
    if (board.isCollapsedSpace(pos)) {
      if (debug) console.log('Path blocked by collapsed', pos);
      return false;
    }
    const stack = board.getStackAt(pos);
    if (stack) {
      if (debug) console.log('Path blocked by stack', pos);
      return false;
    }
  }

  // Landing must be beyond target in the same direction from `from`.
  const dx2 = landing.x - from.x;
  const dy2 = landing.y - from.y;
  const dz2 = (landing.z || 0) - (from.z || 0);

  if (dx !== 0 && Math.sign(dx) !== Math.sign(dx2)) {
    if (debug) console.log('Direction mismatch X');
    return false;
  }
  if (dy !== 0 && Math.sign(dy) !== Math.sign(dy2)) {
    if (debug) console.log('Direction mismatch Y');
    return false;
  }
  if (dz !== 0 && Math.sign(dz) !== Math.sign(dz2)) {
    if (debug) console.log('Direction mismatch Z');
    return false;
  }

  const distToTarget = Math.abs(dx) + Math.abs(dy) + Math.abs(dz);
  const distToLanding = Math.abs(dx2) + Math.abs(dy2) + Math.abs(dz2);
  if (distToLanding <= distToTarget) {
    if (debug) console.log('Landing not beyond target');
    return false;
  }

  // Total distance must be at least stack height (Section 10.2).
  // "The capturing stack must move a distance equal to or greater than its height (H)."
  // This allows extended landings beyond the target as long as the path is clear.
  const segmentDistance = calculateDistance(boardType, from, landing);
  if (segmentDistance < attacker.stackHeight) {
    if (debug)
      console.log(
        'Distance < stackHeight',
        segmentDistance,
        typeof segmentDistance,
        attacker.stackHeight,
        typeof attacker.stackHeight
      );
    return false;
  }

  // Path from target to landing (exclusive) must also be clear.
  const pathFromTarget = getPathPositions(target, landing).slice(1, -1);
  for (const pos of pathFromTarget) {
    if (!board.isValidPosition(pos)) {
      if (debug) console.log('Invalid landing path pos', pos);
      return false;
    }
    if (board.isCollapsedSpace(pos)) {
      if (debug) console.log('Landing path blocked by collapsed', pos);
      return false;
    }
    const stack = board.getStackAt(pos);
    if (stack) {
      if (debug) console.log('Landing path blocked by stack', pos);
      return false;
    }
  }

  // Landing space must be empty (no stack) and not collapsed.
  if (board.isCollapsedSpace(landing)) {
    if (debug) console.log('Landing is collapsed');
    return false;
  }
  const landingStack = board.getStackAt(landing);
  if (landingStack) {
    if (debug) console.log('Landing occupied by stack');
    return false;
  }

  return true;
}

/**
 * Shared helper to answer the question: "Does this stack have at least
 * one legal non-capture move or overtaking capture?" for a given board
 * type and minimal board view. This is used by both the backend
 * RuleEngine and the client sandbox to enforce no-dead-placement and
 * forced-elimination semantics while keeping the core logic in one
 * place.
 */
export function hasAnyLegalMoveOrCaptureFromOnBoard(
  boardType: BoardType,
  from: Position,
  player: number,
  board: MovementBoardView,
  options?: {
    /** Optional cap on how far to search for non-capture moves. */
    maxNonCaptureDistance?: number;
    /** Optional cap on how far beyond the target to search for capture landings. */
    maxCaptureLandingDistance?: number;
  }
): boolean {
  const stack = board.getStackAt(from);
  if (!stack || stack.controllingPlayer !== player) {
    return false;
  }

  const directions = getMovementDirectionsForBoardType(boardType);

  const defaultMaxNonCapture = options?.maxNonCaptureDistance ?? stack.stackHeight + 5;
  const defaultMaxCaptureLanding = options?.maxCaptureLandingDistance ?? stack.stackHeight + 5;

  // === Non-capture movement ===
  for (const dir of directions) {
    for (let distance = stack.stackHeight; distance <= defaultMaxNonCapture; distance++) {
      const target: Position = {
        x: from.x + dir.x * distance,
        y: from.y + dir.y * distance,
      };
      if (dir.z !== undefined) {
        target.z = (from.z || 0) + dir.z * distance;
      }

      if (!board.isValidPosition(target)) {
        break; // Off board in this direction
      }

      if (board.isCollapsedSpace(target)) {
        break; // Cannot move into collapsed space
      }

      const path = getPathPositions(from, target).slice(1, -1);
      let blocked = false;
      for (const pos of path) {
        if (!board.isValidPosition(pos)) {
          blocked = true;
          break;
        }
        if (board.isCollapsedSpace(pos)) {
          blocked = true;
          break;
        }
        const pathStack = board.getStackAt(pos);
        if (pathStack && pathStack.stackHeight > 0) {
          blocked = true;
          break;
        }
      }
      if (blocked) {
        break; // Further distances along this ray are blocked
      }

      const landingStack = board.getStackAt(target);
      const markerOwner = board.getMarkerOwner?.(target);

      if (!landingStack || landingStack.stackHeight === 0) {
        // Empty space or marker
        if (markerOwner === undefined || markerOwner === player) {
          return true;
        }
      } else {
        // Landing on a stack (for merging) is also a legal move
        return true;
      }
    }
  }

  // === Capture reachability ===
  for (const dir of directions) {
    let step = 1;
    let targetPos: Position | undefined;

    // Find first stack along this ray that could be a capture target
    while (true) {
      const pos: Position = {
        x: from.x + dir.x * step,
        y: from.y + dir.y * step,
      };
      if (dir.z !== undefined) {
        pos.z = (from.z || 0) + dir.z * step;
      }

      if (!board.isValidPosition(pos)) {
        break;
      }

      if (board.isCollapsedSpace(pos)) {
        break;
      }

      const stackAtPos = board.getStackAt(pos);
      if (stackAtPos && stackAtPos.stackHeight > 0) {
        // Rule fix: can overtake own stacks; only capHeight comparison matters.
        if (stack.capHeight >= stackAtPos.capHeight) {
          targetPos = pos;
        }
        break;
      }

      step++;
    }

    if (!targetPos) continue;

    // From the target, search for valid landing positions beyond it.
    for (let landingStep = 1; landingStep <= defaultMaxCaptureLanding; landingStep++) {
      const landing: Position = {
        x: targetPos.x + dir.x * landingStep,
        y: targetPos.y + dir.y * landingStep,
      };
      if (dir.z !== undefined) {
        landing.z = (targetPos.z || 0) + dir.z * landingStep;
      }

      if (!board.isValidPosition(landing)) {
        break;
      }

      if (board.isCollapsedSpace(landing)) {
        break;
      }

      const landingStack = board.getStackAt(landing);
      if (landingStack && landingStack.stackHeight > 0) {
        break;
      }

      // Use the shared capture-segment validator to ensure full
      // consistency with all other capture checks.
      const view: CaptureSegmentBoardView = {
        isValidPosition: (pos: Position) => board.isValidPosition(pos),
        isCollapsedSpace: (pos: Position) => board.isCollapsedSpace(pos),
        getStackAt: (pos: Position) => board.getStackAt(pos),
        getMarkerOwner: (pos: Position) => board.getMarkerOwner?.(pos),
      };

      if (validateCaptureSegmentOnBoard(boardType, from, targetPos, landing, player, view)) {
        return true;
      }
    }
  }

  return false;
}

/**
 * Compute the canonical S-invariant snapshot for a given GameState.
 *
 * S = M + C + E
 *   M = markers.size
 *   C = collapsedSpaces.size
 *   E = totalRingsEliminated (falling back to the sum of
 *       board.eliminatedRings when needed).
 */
export function computeProgressSnapshot(state: GameState): ProgressSnapshot {
  const markers = state.board.markers.size;
  const collapsed = state.board.collapsedSpaces.size;

  const eliminatedFromBoard = Object.values(state.board.eliminatedRings ?? {}).reduce(
    (sum, value) => sum + value,
    0
  );

  const eliminated =
    (state as GameState & { totalRingsEliminated?: number }).totalRingsEliminated ??
    eliminatedFromBoard;

  const S = markers + collapsed + eliminated;
  return { markers, collapsed, eliminated, S };
}

/**
 * Build a lightweight, order-independent summary of a BoardState. This is
 * primarily used for parity debugging and log output and is kept stable
 * across engines so that backend and sandbox traces can be compared.
 */
export function summarizeBoard(board: BoardState): BoardSummary {
  const stacks: string[] = [];
  for (const [key, stack] of board.stacks.entries()) {
    stacks.push(`${key}:${stack.controllingPlayer}:${stack.stackHeight}:${stack.capHeight}`);
  }
  stacks.sort();

  const markers: string[] = [];
  for (const [key, marker] of board.markers.entries()) {
    markers.push(`${key}:${marker.player}`);
  }
  markers.sort();

  const collapsedSpaces: string[] = [];
  for (const [key, owner] of board.collapsedSpaces.entries()) {
    collapsedSpaces.push(`${key}:${owner}`);
  }
  collapsedSpaces.sort();

  return { stacks, markers, collapsedSpaces };
}

/**
 * Canonical hash of a GameState used by tests and diagnostic tooling to
 * detect state changes and compare backend/sandbox traces. The exact
 * string format is opaque to callers; only equality is relied upon.
 */
export function hashGameState(state: GameState): string {
  const boardSummary = summarizeBoard(state.board);

  const playersMeta = state.players
    .map((p) => `${p.playerNumber}:${p.ringsInHand}:${p.eliminatedRings}:${p.territorySpaces}`)
    .sort()
    .join('|');

  const meta = `${state.currentPlayer}:${state.currentPhase}:${state.gameStatus}`;

  return [
    meta,
    playersMeta,
    boardSummary.stacks.join('|'),
    boardSummary.markers.join('|'),
    boardSummary.collapsedSpaces.join('|'),
  ].join('#');
}

export interface MarkerPathHelpers {
  setMarker(position: Position, playerNumber: number, board: BoardState): void;
  collapseMarker(position: Position, playerNumber: number, board: BoardState): void;
  flipMarker(position: Position, playerNumber: number, board: BoardState): void;
}

/**
 * Apply marker effects for a move or capture segment from `from` to `to` on
 * the given board, using the provided helper callbacks.
 *
 * By default this mirrors the backend movement behaviour:
 *   - Leave a marker on the true departure space.
 *   - Process intermediate markers (collapse/flip).
 *   - Remove a same-colour marker on the landing space.
 *
 * Callers that need finer-grained control (e.g. capture segments that want
 * to avoid placing a departure marker on an intermediate stack such as the
 * capture target) can pass options to disable the departure marker while
 * still reusing the intermediate/landing semantics.
 */
export function applyMarkerEffectsAlongPathOnBoard(
  board: BoardState,
  from: Position,
  to: Position,
  playerNumber: number,
  helpers: MarkerPathHelpers,
  options?: { leaveDepartureMarker?: boolean }
): void {
  const path = getPathPositions(from, to);
  if (path.length === 0) return;

  const leaveDepartureMarker = options?.leaveDepartureMarker !== false;

  const fromKey = positionToString(from);
  // Leave a marker on the departure space if it isn't already collapsed.
  if (leaveDepartureMarker && !board.collapsedSpaces.has(fromKey)) {
    const existing = board.markers.get(fromKey);
    if (!existing) {
      helpers.setMarker(from, playerNumber, board);
    }
  }

  // Process intermediate positions (excluding endpoints)
  const intermediate = path.slice(1, -1);
  for (const pos of intermediate) {
    const key = positionToString(pos);
    if (board.collapsedSpaces.has(key)) {
      continue;
    }
    const marker = board.markers.get(key);
    if (!marker) {
      continue;
    }
    if (marker.player === playerNumber) {
      // Own marker collapses to territory
      helpers.collapseMarker(pos, playerNumber, board);
    } else {
      // Opponent marker flips to mover's color
      helpers.flipMarker(pos, playerNumber, board);
    }
  }

  // Landing: remove own marker if present
  const landingKey = positionToString(to);
  const landingMarker = board.markers.get(landingKey);
  if (landingMarker && landingMarker.player === playerNumber) {
    board.markers.delete(landingKey);
  }
}
