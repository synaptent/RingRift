import {
  BoardType,
  Position,
  BoardState,
  GameState,
  ProgressSnapshot,
  BoardSummary,
  positionToString,
  type GameStatus,
  type GamePhase,
} from '../types/game';
import { debugLog, flagEnabled } from '../utils/envFlags';

/**
 * Shared, browser-safe core helpers for RingRift engine logic.
 *
 * **SSoT (Single Source of Truth):** This file is part of the TypeScript
 * shared engine (`src/shared/engine/**`), which is the *primary executable
 * derivation* of the canonical rules defined in `RULES_CANONICAL_SPEC.md`.
 *
 * All implementations—Python AI service, client sandbox, backend hosts, and
 * replay systems—must derive from and faithfully mirror the behaviour of this
 * shared engine. If any implementation disagrees with this code, that
 * implementation must be updated to match—not the other way around.
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
 * Count the number of rings of a given player's colour that are currently
 * on the board in any stack, regardless of which player controls those
 * stacks. This is used for own-colour supply cap checks (ringsPerPlayer).
 *
 * Rule reference:
 * - RR-CANON-R020 / Compact §1.1 – own-colour supply cap.
 */
export function countRingsOnBoardForPlayer(board: BoardState, playerNumber: number): number {
  let count = 0;

  for (const stack of board.stacks.values()) {
    // RingStack.rings is ordered top→bottom, but for counting by colour
    // only the owner ID matters, not position within the stack.
    for (const owner of stack.rings) {
      if (owner === playerNumber) {
        count += 1;
      }
    }
  }

  return count;
}

/**
 * Count the total number of rings of a given player's colour that are
 * currently in play: all rings of that colour on the board in any stack
 * (regardless of controlling player) plus that player's rings in hand.
 *
 * This mirrors the formal definition of the ringsPerPlayer own-colour
 * supply cap in the compact/canonical rules.
 */
export function countRingsInPlayForPlayer(state: GameState, playerNumber: number): number {
  const countOnBoard = countRingsOnBoardForPlayer(state.board, playerNumber);

  const player = state.players.find((p) => p.playerNumber === playerNumber);
  const ringsInHand = player ? player.ringsInHand : 0;

  return countOnBoard + ringsInHand;
}

/**
 * Calculate cap height for a ring stack.
 *
 * Rule Reference: RR-CANON-R145 - Cap height is consecutive rings of
 * the same color from the top of the stack.
 *
 * Territory self-elimination costs the entire cap. Eligible stacks:
 * - (i) Mixed-colour stack: P controls with other players' rings buried beneath.
 *       Cap = top consecutive P rings. Eliminating exposes buried rings.
 * - (ii) Single-colour stack (height > 1): Entire stack is the cap.
 *        Eliminating removes the stack from the board.
 * - (iii) Height-1 stack: Cap height = 1 (single ring).
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
    // Only add z to intermediate positions when z is an actual number.
    // This fixes a parity bug where z: null from replay DBs would incorrectly
    // create intermediate positions with z: 0, causing marker key mismatches
    // (e.g., "4,2" vs "4,2,0") during path-based marker processing.
    if (typeof to.z === 'number') {
      pos.z = Math.round((from.z ?? 0) + stepZ * i);
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
  const debug = flagEnabled('RINGRIFT_DEBUG_CAPTURE');

  if (
    !board.isValidPosition(from) ||
    !board.isValidPosition(target) ||
    !board.isValidPosition(landing)
  ) {
    debugLog(debug, 'Invalid position(s)');
    return false;
  }

  const attacker = board.getStackAt(from);
  if (!attacker || attacker.controllingPlayer !== player) {
    debugLog(debug, 'Invalid attacker', attacker, player);
    return false;
  }

  const targetStack = board.getStackAt(target);
  if (!targetStack) {
    debugLog(debug, 'No target stack');
    return false;
  }

  // Cap height must be >= target's cap height (Section 10.1)
  if (attacker.capHeight < targetStack.capHeight) {
    debugLog(debug, 'Insufficient cap height', attacker.capHeight, targetStack.capHeight);
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
      debugLog(debug, 'Invalid hex direction');
      return false;
    }
  } else {
    // Square boards: orthogonal or diagonal only.
    if (dx === 0 && dy === 0) {
      debugLog(debug, 'Zero movement');
      return false;
    }
    if (dx !== 0 && dy !== 0 && Math.abs(dx) !== Math.abs(dy)) {
      debugLog(debug, 'Invalid square direction');
      return false;
    }
  }

  // Path from attacker to target (exclusive) must be clear of stacks
  // and collapsed spaces. Markers are allowed.
  const pathToTarget = getPathPositions(from, target).slice(1, -1);
  for (const pos of pathToTarget) {
    if (!board.isValidPosition(pos)) {
      debugLog(debug, 'Invalid path pos', pos);
      return false;
    }
    if (board.isCollapsedSpace(pos)) {
      debugLog(debug, 'Path blocked by collapsed', pos);
      return false;
    }
    const stack = board.getStackAt(pos);
    if (stack) {
      debugLog(debug, 'Path blocked by stack', pos);
      return false;
    }
  }

  // Landing must be beyond target in the same direction from `from`.
  const dx2 = landing.x - from.x;
  const dy2 = landing.y - from.y;
  const dz2 = (landing.z || 0) - (from.z || 0);

  if (dx !== 0 && Math.sign(dx) !== Math.sign(dx2)) {
    debugLog(debug, 'Direction mismatch X');
    return false;
  }
  if (dy !== 0 && Math.sign(dy) !== Math.sign(dy2)) {
    debugLog(debug, 'Direction mismatch Y');
    return false;
  }
  if (dz !== 0 && Math.sign(dz) !== Math.sign(dz2)) {
    debugLog(debug, 'Direction mismatch Z');
    return false;
  }

  const distToTarget = Math.abs(dx) + Math.abs(dy) + Math.abs(dz);
  const distToLanding = Math.abs(dx2) + Math.abs(dy2) + Math.abs(dz2);
  if (distToLanding <= distToTarget) {
    debugLog(debug, 'Landing not beyond target');
    return false;
  }

  // Total distance must be at least stack height (Section 10.2).
  // "The capturing stack must move a distance equal to or greater than its height (H)."
  // This allows extended landings beyond the target as long as the path is clear.
  const segmentDistance = calculateDistance(boardType, from, landing);
  if (segmentDistance < attacker.stackHeight) {
    debugLog(
      debug,
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
      debugLog(debug, 'Invalid landing path pos', pos);
      return false;
    }
    if (board.isCollapsedSpace(pos)) {
      debugLog(debug, 'Landing path blocked by collapsed', pos);
      return false;
    }
    const stack = board.getStackAt(pos);
    if (stack) {
      debugLog(debug, 'Landing path blocked by stack', pos);
      return false;
    }
  }

  // Landing space must be empty (no stack) and not collapsed.
  if (board.isCollapsedSpace(landing)) {
    debugLog(debug, 'Landing is collapsed');
    return false;
  }
  const landingStack = board.getStackAt(landing);
  if (landingStack) {
    debugLog(debug, 'Landing occupied by stack');
    return false;
  }

  // RR-CANON-R091/R101: Landing on any marker (own or opponent) is legal.
  // The marker is removed and the top cap ring is eliminated.
  // Validation allows this; mutation handles the elimination cost.

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
    /** Enable detailed debug logging for this check */
    debug?: boolean;
  }
): boolean {
  const debug = options?.debug || process.env.RINGRIFT_DEBUG_MOVEMENT === 'true';
  const stack = board.getStackAt(from);
  if (!stack || stack.controllingPlayer !== player) {
    if (debug) {
      debugLog(debug, '[hasAnyLegalMove] No stack or wrong player at', from, 'player:', player);
    }
    return false;
  }

  const directions = getMovementDirectionsForBoardType(boardType);

  const defaultMaxNonCapture = options?.maxNonCaptureDistance ?? stack.stackHeight + 5;
  const defaultMaxCaptureLanding = options?.maxCaptureLandingDistance ?? stack.stackHeight + 5;

  if (debug) {
    debugLog(
      debug,
      '[hasAnyLegalMove] Checking stack at',
      positionToString(from),
      'height:',
      stack.stackHeight,
      'boardType:',
      boardType,
      'directions:',
      directions.length,
      'maxDist:',
      defaultMaxNonCapture
    );
  }

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

      if (!landingStack || landingStack.stackHeight === 0) {
        // Empty space or marker - landing on any marker (own or opponent) is legal
        // per RR-CANON-R091/R092 (uniform marker landing rule).
        if (debug) {
          debugLog(
            debug,
            '[hasAnyLegalMove] FOUND non-capture move from',
            positionToString(from),
            'to',
            positionToString(target),
            'distance:',
            distance
          );
        }
        return true;
      } else {
        // Landing on a stack is NOT allowed - stacks block the ray.
        // Rule 8.1: "Cannot pass through other rings or stacks"
        // Rule 8.2: "Landing on ... empty or occupied by a single marker"
        break;
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
        if (debug) {
          debugLog(
            debug,
            '[hasAnyLegalMove] FOUND capture from',
            positionToString(from),
            'over',
            positionToString(targetPos),
            'to',
            positionToString(landing)
          );
        }
        return true;
      }
    }
  }

  if (debug) {
    debugLog(debug, '[hasAnyLegalMove] NO legal moves found from', positionToString(from));
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
 * Canonical fingerprint of a GameState - a deterministic, human-readable string
 * representation useful for debugging parity issues between engines.
 *
 * Format: meta#players#stacks#markers#collapsed
 * - meta: currentPlayer:currentPhase:gameStatus
 * - players: sorted pipe-separated player stats (playerNumber:ringsInHand:eliminatedRings:territorySpaces)
 * - stacks: sorted pipe-separated stack info (posKey:controller:height:capHeight)
 * - markers: sorted pipe-separated marker info (posKey:player)
 * - collapsed: sorted pipe-separated collapsed space info (posKey:owner)
 *
 * This format is designed to be identical between TypeScript and Python implementations.
 */
export function fingerprintGameState(state: GameState): string {
  const boardSummary = summarizeBoard(state.board);

  const playersMeta = state.players
    .map((p) => `${p.playerNumber}:${p.ringsInHand}:${p.eliminatedRings}:${p.territorySpaces}`)
    .sort()
    .join('|');

  // Canonicalise terminal status strings so that legacy 'finished'
  // and newer 'completed' values fingerprint identically. This keeps
  // the cross-engine hash stable even when hosts use different but
  // equivalent terminal status labels.
  const canonicalStatus: GameStatus =
    state.gameStatus === 'finished' ? ('completed' as GameStatus) : state.gameStatus;

  // For terminal states, currentPlayer/currentPhase are host-local
  // metadata and not semantically meaningful. Canonicalise them so
  // that engines which differ only in their choice of terminal
  // phase/player still produce identical fingerprints.
  // Note: 'game_over' is the canonical terminal phase as of RR-PARITY-FIX-2024-12.
  const isTerminal = canonicalStatus === 'completed' || canonicalStatus === 'abandoned';
  const metaPlayer = isTerminal ? 0 : state.currentPlayer;
  const metaPhase: GamePhase = isTerminal ? 'game_over' : state.currentPhase;

  const meta = `${metaPlayer}:${metaPhase}:${canonicalStatus}`;

  return [
    meta,
    playersMeta,
    boardSummary.stacks.join('|'),
    boardSummary.markers.join('|'),
    boardSummary.collapsedSpaces.join('|'),
  ].join('#');
}

/**
 * Compute a SHA-256 hash of the game state fingerprint, truncated to 16 hex chars.
 * This matches the Python _compute_state_hash format for cross-engine parity testing.
 *
 * @param state - The game state to hash
 * @returns 16 character hex string (first 64 bits of SHA-256)
 */
export function hashGameStateSHA256(state: GameState): string {
  const fingerprint = fingerprintGameState(state);
  // Compact hash used for cross-engine parity with Python's _compute_state_hash.
  return simpleHash(fingerprint);
}

/**
 * Simple string hash function that produces consistent results across
 * environments and matches Python's _simple_hash implementation in
 * app/db/game_replay.py.
 *
 * Note: This is NOT cryptographically secure - only for parity comparison.
 */
function simpleHash(str: string): string {
  let h1 = 0xdeadbeef | 0;
  let h2 = 0x41c6ce57 | 0;

  for (let i = 0; i < str.length; i += 1) {
    const ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761) | 0;
    h2 = Math.imul(h2 ^ ch, 1597334677) | 0;
  }

  h1 = (Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909)) | 0;
  h2 = (Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909)) | 0;

  // Combine into 16 hex chars without relying on 53-bit double precision.
  const hi = h2 >>> 0;
  const lo = h1 >>> 0;
  const hiHex = hi.toString(16).padStart(8, '0');
  const loHex = lo.toString(16).padStart(8, '0');
  return (hiHex + loHex).slice(0, 16);
}

/**
 * Legacy hash function - returns the readable fingerprint string.
 * @deprecated Use fingerprintGameState for readable output or hashGameStateSHA256 for compact comparison.
 */
export function hashGameState(state: GameState): string {
  return fingerprintGameState(state);
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
