import { BoardType, Position } from '../types/game';

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
  { x: 1, y: 0 },   // E
  { x: 1, y: 1 },   // SE
  { x: 0, y: 1 },   // S
  { x: -1, y: 1 },  // SW
  { x: -1, y: 0 },  // W
  { x: -1, y: -1 }, // NW
  { x: 0, y: -1 },  // N
  { x: 1, y: -1 }   // NE
];

/**
 * Canonical 6-direction set for hexagonal boards in cube coordinates.
 */
export const HEX_DIRECTIONS: Direction[] = [
  { x: 1, y: 0, z: -1 },  // East
  { x: 0, y: 1, z: -1 },  // Southeast
  { x: -1, y: 1, z: 0 },  // Southwest
  { x: -1, y: 0, z: 1 },  // West
  { x: 0, y: -1, z: 1 },  // Northwest
  { x: 1, y: -1, z: 0 }   // Northeast
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
      y: Math.round(from.y + stepY * i)
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
 * - Square boards: Manhattan distance
 * - Hex boards: cube-coordinate distance
 */
export function calculateDistance(boardType: BoardType, from: Position, to: Position): number {
  if (boardType === 'hexagonal') {
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const dz = (to.z || 0) - (from.z || 0);
    return (Math.abs(dx) + Math.abs(dy) + Math.abs(dz)) / 2;
  }

  return Math.abs(to.x - from.x) + Math.abs(to.y - from.y);
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
  if (!board.isValidPosition(from) || !board.isValidPosition(target) || !board.isValidPosition(landing)) {
    return false;
  }

  const attacker = board.getStackAt(from);
  if (!attacker || attacker.controllingPlayer !== player) {
    return false;
  }

  const targetStack = board.getStackAt(target);
  if (!targetStack) {
    return false;
  }

  // Cap height must be >= target's cap height (Section 10.1)
  if (attacker.capHeight < targetStack.capHeight) {
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
      return false;
    }
  } else {
    // Square boards: orthogonal or diagonal only.
    if (dx === 0 && dy === 0) {
      return false;
    }
    if (dx !== 0 && dy !== 0 && Math.abs(dx) !== Math.abs(dy)) {
      return false;
    }
  }

  // Path from attacker to target (exclusive) must be clear of stacks
  // and collapsed spaces. Markers are allowed.
  const pathToTarget = getPathPositions(from, target).slice(1, -1);
  for (const pos of pathToTarget) {
    if (!board.isValidPosition(pos)) {
      return false;
    }
    if (board.isCollapsedSpace(pos)) {
      return false;
    }
    const stack = board.getStackAt(pos);
    if (stack) {
      return false;
    }
  }

  // Landing must be beyond target in the same direction from `from`.
  const dx2 = landing.x - from.x;
  const dy2 = landing.y - from.y;
  const dz2 = (landing.z || 0) - (from.z || 0);

  if (dx !== 0 && Math.sign(dx) !== Math.sign(dx2)) return false;
  if (dy !== 0 && Math.sign(dy) !== Math.sign(dy2)) return false;
  if (dz !== 0 && Math.sign(dz) !== Math.sign(dz2)) return false;

  const distToTarget = Math.abs(dx) + Math.abs(dy) + Math.abs(dz);
  const distToLanding = Math.abs(dx2) + Math.abs(dy2) + Math.abs(dz2);
  if (distToLanding <= distToTarget) {
    return false;
  }

  // Total distance must be at least stack height (Section 10.2).
  const segmentDistance = calculateDistance(boardType, from, landing);
  if (segmentDistance < attacker.stackHeight) {
    return false;
  }

  // Path from target to landing (exclusive) must also be clear.
  const pathFromTarget = getPathPositions(target, landing).slice(1, -1);
  for (const pos of pathFromTarget) {
    if (!board.isValidPosition(pos)) {
      return false;
    }
    if (board.isCollapsedSpace(pos)) {
      return false;
    }
    const stack = board.getStackAt(pos);
    if (stack) {
      return false;
    }
  }

  // Landing space must be empty (no stack) and not collapsed.
  if (board.isCollapsedSpace(landing)) {
    return false;
  }
  const landingStack = board.getStackAt(landing);
  if (landingStack) {
    return false;
  }

  // If there's a marker at landing, it must belong to the attacker.
  const markerOwner = board.getMarkerOwner?.(landing);
  if (markerOwner !== undefined && markerOwner !== attacker.controllingPlayer) {
    return false;
  }

  return true;
}
