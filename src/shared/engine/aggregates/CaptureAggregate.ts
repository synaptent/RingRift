/**
 * ═══════════════════════════════════════════════════════════════════════════
 * CaptureAggregate - Consolidated Overtaking Capture Domain
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * This aggregate consolidates all overtaking capture validation, mutation,
 * enumeration, and chain capture logic from:
 *
 * - captureLogic.ts → capture enumeration, validation
 * - validators/CaptureValidator.ts → validation
 * - mutators/CaptureMutator.ts → mutation
 *
 * Rule Reference: Section 10 - Overtaking Capture
 *
 * Key Rules:
 * - RR-CANON-R070: Overtaking capture (attacker cap height >= target cap height)
 * - RR-CANON-R071: Capture execution (remove target, move attacker stack, leave marker)
 * - RR-CANON-R072: Cap height comparison uses current game state
 * - RR-CANON-R084: Chain captures (consecutive captures from same stack)
 * - RR-CANON-R085: Chain capture must extend from previous capture position
 * - Self-capture is legal (can overtake own stacks)
 *
 * Design principles:
 * - Pure functions: No side effects, return new state
 * - Type safety: Full TypeScript typing
 * - Backward compatibility: Source files continue to export their functions
 */

import type { GameState, BoardState, Position, Move, BoardType } from '../../types/game';
import { positionToString } from '../../types/game';

import type { ValidationResult, OvertakingCaptureAction, ContinueChainAction } from '../types';

import {
  getMovementDirectionsForBoardType,
  getPathPositions,
  calculateCapHeight,
  validateCaptureSegmentOnBoard,
  CaptureSegmentBoardView,
} from '../core';

import { isValidPosition } from '../validators/utils';

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Individual segment in a capture chain, tracking the from, target, landing
 * positions and the cap height of the captured stack.
 *
 * This is used by host engines to track the full history of a multi-segment
 * chain capture as it progresses.
 */
export interface ChainCaptureSegment {
  from: Position;
  target: Position;
  landing: Position;
  capturedCapHeight: number;
}

/**
 * Full chain capture state used by host engines (GameEngine, etc.) to track
 * mandatory chain captures across multiple segments.
 *
 * This differs from ChainCaptureStateSnapshot in that it maintains the full
 * segment history, visited positions for cycle detection, and cached
 * available moves for the current position.
 */
export interface ChainCaptureState {
  playerNumber: number;
  startPosition: Position;
  currentPosition: Position;
  segments: ChainCaptureSegment[];
  /** Full capture moves (from=currentPosition) that the player may choose from */
  availableMoves: Move[];
  /** Positions visited by the capturing stack to help avoid pathological cycles */
  visitedPositions: Set<string>;
}

/**
 * Adapter interface for board queries used by capture enumeration.
 * Callers construct this from a GameState/BoardState using lightweight
 * stack/marker projections.
 */
export interface CaptureBoardAdapters {
  /** True if the position is on the board and addressable. */
  isValidPosition(pos: Position): boolean;
  /** True if this space is a collapsed territory space. */
  isCollapsedSpace(pos: Position): boolean;
  /**
   * Lightweight stack view at a position.
   * Returns undefined if no stack exists at the position.
   */
  getStackAt(pos: Position):
    | {
        controllingPlayer: number;
        capHeight: number;
        stackHeight: number;
      }
    | undefined;
  /** Optional marker lookup for landing-on-own-marker checks. */
  getMarkerOwner(pos: Position): number | undefined;
}

/**
 * Snapshot of chain capture state used by enumeration and continuation checks.
 *
 * Hosts maintain this state across capture segments to track:
 * - Current position of the capturing stack
 * - Which targets have already been captured in this chain
 * - Whether the chain must continue
 */
export interface ChainCaptureStateSnapshot {
  /**
   * Player performing the capture chain.
   * Must match GameState.currentPlayer in normal engine flows.
   */
  player: number;

  /**
   * Position of the capturing stack at the start of the next segment.
   * For the first segment this is the origin of the initial overtaking
   * capture; for later segments it is the landing position of the most
   * recent segment.
   */
  currentPosition: Position;

  /**
   * Target positions (stringified via positionToString) that have already
   * been captured in this chain. Used to prevent immediate backtracking
   * when combined with disallowRevisitedTargets option.
   */
  capturedThisChain: Position[];
}

/**
 * Configuration for capture-chain segment enumeration.
 */
export interface ChainCaptureEnumerationOptions {
  /**
   * When true, filter out any candidate segment whose captureTarget is
   * present in ChainCaptureStateSnapshot.capturedThisChain.
   *
   * Default: false.
   */
  disallowRevisitedTargets?: boolean;

  /**
   * Move number to embed in the generated Move instances.
   * When not provided, callers are expected to patch moveNumber themselves.
   */
  moveNumber?: number;

  /**
   * How generated moves should be typed:
   * - 'initial'      – type: 'overtaking_capture'
   * - 'continuation' – type: 'continue_capture_segment'
   *
   * Default: 'continuation'.
   */
  kind?: 'initial' | 'continuation';
}

/**
 * Result of asking the engine whether a capture chain may or must
 * continue from the current position.
 */
export interface ChainCaptureContinuationInfo {
  /**
   * True when at least one legal capture segment exists from the current
   * position. Under the standard rules this implies that the chain MUST
   * continue.
   */
  mustContinue: boolean;

  /**
   * Concrete Move instances describing each legal next segment, suitable
   * for inclusion in getValidMoves during the chain_capture phase.
   */
  availableContinuations: Move[];
}

/**
 * Parameters for applying a single capture segment.
 */
export interface CaptureSegmentParams {
  /** Origin of the capturing stack. */
  from: Position;
  /** Position of the stack being captured. */
  target: Position;
  /** Landing position after the capture. */
  landing: Position;
  /** Player performing the capture. */
  player: number;
}

/**
 * Result of applying a capture segment mutation.
 */
export interface CaptureApplicationOutcome {
  /**
   * Next GameState after applying the capture, including all stack,
   * marker, collapsed-space, and elimination bookkeeping.
   */
  nextState: GameState;

  /**
   * Number of rings transferred from target stack to attacker stack.
   * This is typically 1 (the top ring of the target).
   */
  ringsTransferred: number;

  /**
   * True if further captures are available from the landing position,
   * indicating the chain must continue.
   */
  chainContinuationRequired: boolean;
}

/**
 * Result of validating a capture action.
 */
export type CaptureValidationResult = ValidationResult;

/**
 * Result of applying a capture mutation.
 */
export type CaptureMutationResult =
  | { success: true; newState: GameState; chainCaptures: Position[] }
  | { success: false; reason: string };

// ═══════════════════════════════════════════════════════════════════════════
// Re-exports from SharedCore
// ═══════════════════════════════════════════════════════════════════════════

// Re-export the segment validation function from core for convenience
export { validateCaptureSegmentOnBoard } from '../core';
export type { CaptureSegmentBoardView } from '../core';

// ═══════════════════════════════════════════════════════════════════════════
// Internal Helpers
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Create a CaptureBoardAdapters adapter from a BoardState.
 */
function createBoardAdapters(
  board: BoardState,
  boardType: BoardType,
  size: number
): CaptureBoardAdapters {
  return {
    isValidPosition: (pos: Position) => isValidPosition(pos, boardType, size),
    isCollapsedSpace: (pos: Position) => board.collapsedSpaces.has(positionToString(pos)),
    getStackAt: (pos: Position) => {
      const key = positionToString(pos);
      const stack = board.stacks.get(key);
      if (!stack) return undefined;
      return {
        controllingPlayer: stack.controllingPlayer,
        capHeight: stack.capHeight,
        stackHeight: stack.stackHeight,
      };
    },
    getMarkerOwner: (pos: Position) => {
      const marker = board.markers.get(positionToString(pos));
      return marker?.player;
    },
  };
}

/**
 * Create a CaptureSegmentBoardView from a BoardState (used by validateCaptureSegmentOnBoard).
 */
function createBoardView(
  board: BoardState,
  boardType: BoardType,
  size: number
): CaptureSegmentBoardView {
  return {
    isValidPosition: (pos: Position) => isValidPosition(pos, boardType, size),
    isCollapsedSpace: (pos: Position) => board.collapsedSpaces.has(positionToString(pos)),
    getStackAt: (pos: Position) => {
      const key = positionToString(pos);
      const stack = board.stacks.get(key);
      if (!stack) return undefined;
      return {
        controllingPlayer: stack.controllingPlayer,
        capHeight: stack.capHeight,
        stackHeight: stack.stackHeight,
      };
    },
    getMarkerOwner: (pos: Position) => {
      const marker = board.markers.get(positionToString(pos));
      return marker?.player;
    },
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Validation Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Validate an OVERTAKING_CAPTURE action against full GameState.
 *
 * Rule Reference: Section 10 - Overtaking Capture
 *
 * Checks:
 * - Phase must be 'movement', 'capture', or 'chain_capture'
 * - Must be the player's turn
 * - All positions must be valid
 * - Delegates to validateCaptureSegmentOnBoard for capture-specific validation
 */
export function validateCapture(
  state: GameState,
  action: OvertakingCaptureAction
): CaptureValidationResult {
  // 1. Phase Check
  // Capture can happen in 'movement' (initial capture) or 'capture'/'chain_capture' phases
  if (
    state.currentPhase !== 'movement' &&
    state.currentPhase !== 'capture' &&
    state.currentPhase !== 'chain_capture'
  ) {
    return { valid: false, reason: 'Not in a phase allowing capture', code: 'INVALID_PHASE' };
  }

  // 2. Turn Check
  if (action.playerId !== state.currentPlayer) {
    return { valid: false, reason: 'Not your turn', code: 'NOT_YOUR_TURN' };
  }

  // 3. Position Validity
  if (
    !isValidPosition(action.from, state.board.type, state.board.size) ||
    !isValidPosition(action.to, state.board.type, state.board.size) ||
    !isValidPosition(action.captureTarget, state.board.type, state.board.size)
  ) {
    return { valid: false, reason: 'Position off board', code: 'INVALID_POSITION' };
  }

  // 4. Use Shared Core Validator
  const boardView = createBoardView(state.board, state.board.type, state.board.size);

  const isValid = validateCaptureSegmentOnBoard(
    state.board.type,
    action.from,
    action.captureTarget,
    action.to,
    action.playerId,
    boardView
  );

  if (!isValid) {
    return {
      valid: false,
      reason: 'Invalid capture move according to core rules',
      code: 'INVALID_CAPTURE',
    };
  }

  return { valid: true };
}

// ═══════════════════════════════════════════════════════════════════════════
// Enumeration Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Enumerate all legal overtaking capture segments for the given player from
 * the specified stack position, using the same ray-walk semantics as the
 * backend RuleEngine / GameEngine.
 *
 * Rule Reference: Section 10 - Overtaking Capture
 *
 * This function walks each direction from the origin, finds potential targets,
 * then enumerates all valid landing positions beyond each target.
 */
export function enumerateCaptureMoves(
  boardType: BoardType,
  from: Position,
  playerNumber: number,
  adapters: CaptureBoardAdapters,
  moveNumber: number
): Move[] {
  const results: Move[] = [];
  const directions = getMovementDirectionsForBoardType(boardType);

  // We need the attacker stack to check cap height
  const attacker = adapters.getStackAt(from);
  if (!attacker || attacker.controllingPlayer !== playerNumber) {
    return results;
  }

  for (const dir of directions) {
    let step = 1;
    let targetPos: Position | undefined;

    // 1. Find target - walk outward until we find a stack
    while (true) {
      const pos: Position = {
        x: from.x + dir.x * step,
        y: from.y + dir.y * step,
        ...(dir.z !== undefined && { z: (from.z || 0) + dir.z * step }),
      };

      if (!adapters.isValidPosition(pos)) {
        break;
      }

      if (adapters.isCollapsedSpace(pos)) {
        break;
      }

      const stack = adapters.getStackAt(pos);
      if (stack && stack.stackHeight > 0) {
        // Found a potential target - check cap height
        if (attacker.capHeight >= stack.capHeight) {
          targetPos = pos;
        }
        break;
      }

      step++;
    }

    if (!targetPos) continue;

    // 2. Find landing positions beyond target
    let landingStep = 1;
    while (true) {
      const landingPos: Position = {
        x: targetPos.x + dir.x * landingStep,
        y: targetPos.y + dir.y * landingStep,
        ...(dir.z !== undefined && { z: (targetPos.z || 0) + dir.z * landingStep }),
      };

      if (!adapters.isValidPosition(landingPos)) {
        break;
      }

      if (adapters.isCollapsedSpace(landingPos)) {
        break;
      }

      const landingStack = adapters.getStackAt(landingPos);
      if (landingStack && landingStack.stackHeight > 0) {
        break;
      }

      // Validate the full segment using the shared validator
      const view: CaptureSegmentBoardView = {
        isValidPosition: adapters.isValidPosition,
        isCollapsedSpace: adapters.isCollapsedSpace,
        getStackAt: adapters.getStackAt,
        getMarkerOwner: adapters.getMarkerOwner,
      };

      const ok = validateCaptureSegmentOnBoard(
        boardType,
        from,
        targetPos,
        landingPos,
        playerNumber,
        view
      );

      if (ok) {
        results.push({
          id: `capture-${positionToString(from)}-${positionToString(targetPos)}-${positionToString(landingPos)}-${moveNumber}`,
          type: 'overtaking_capture',
          player: playerNumber,
          from,
          captureTarget: targetPos,
          to: landingPos,
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber,
        });
      }

      landingStep++;
    }
  }

  return results;
}

/**
 * Enumerate all capture moves for a player from all their stacks.
 *
 * Iterates over all stacks controlled by the player and returns Move
 * objects for each valid capture.
 */
export function enumerateAllCaptureMoves(state: GameState, player: number): Move[] {
  const moves: Move[] = [];
  const adapters = createBoardAdapters(state.board, state.board.type, state.board.size);
  const moveNumber = state.moveHistory.length + 1;

  for (const stack of state.board.stacks.values()) {
    if (stack.controllingPlayer !== player || stack.stackHeight <= 0) {
      continue;
    }

    const captures = enumerateCaptureMoves(
      state.board.type,
      stack.position,
      player,
      adapters,
      moveNumber
    );

    moves.push(...captures);
  }

  return moves;
}

/**
 * Enumerate all legal capture-chain segments for the specified player and
 * starting position on the given GameState.
 *
 * Responsibilities:
 *
 * - Delegate the geometric legality checks to enumerateCaptureMoves.
 * - Optionally filter out segments that revisit an already-captured target
 *   according to ChainCaptureEnumerationOptions.disallowRevisitedTargets.
 * - Normalize the Move.type field to either:
 *   - 'overtaking_capture'   (kind === 'initial'), or
 *   - 'continue_capture_segment' (kind === 'continuation').
 */
export function enumerateChainCaptureSegments(
  state: GameState,
  snapshot: ChainCaptureStateSnapshot,
  options?: ChainCaptureEnumerationOptions
): Move[] {
  const adapters = createBoardAdapters(state.board, state.board.type, state.board.size);
  const moveNumber = options?.moveNumber ?? state.moveHistory.length + 1;
  const kind = options?.kind ?? 'continuation';

  // Get all capture moves from the current position
  let moves = enumerateCaptureMoves(
    state.board.type,
    snapshot.currentPosition,
    snapshot.player,
    adapters,
    moveNumber
  );

  // Filter out revisited targets if requested
  if (options?.disallowRevisitedTargets && snapshot.capturedThisChain.length > 0) {
    const visitedSet = new Set(snapshot.capturedThisChain.map(positionToString));
    moves = moves.filter((m) => {
      if (!m.captureTarget) return true;
      return !visitedSet.has(positionToString(m.captureTarget));
    });
  }

  // Normalize move types
  const targetType = kind === 'initial' ? 'overtaking_capture' : 'continue_capture_segment';
  return moves.map((m) => ({
    ...m,
    type: targetType as Move['type'],
    id: `${targetType}-${m.from ? positionToString(m.from) : 'unknown'}-${m.captureTarget ? positionToString(m.captureTarget) : 'unknown'}-${positionToString(m.to)}-${moveNumber}`,
  }));
}

/**
 * Convenience wrapper that answers "must this chain continue?" in a single
 * call, returning both the boolean and the concrete segment list.
 *
 * Hosts should call this helper after applying each capture segment, and
 * transition to the `chain_capture` phase when mustContinue is true, or
 * proceed to line processing otherwise.
 */
export function getChainCaptureContinuationInfo(
  state: GameState,
  player: number,
  currentPosition: Position
): ChainCaptureContinuationInfo {
  const snapshot: ChainCaptureStateSnapshot = {
    player,
    currentPosition,
    capturedThisChain: [],
  };

  const segments = enumerateChainCaptureSegments(state, snapshot, {
    kind: 'continuation',
  });

  return {
    mustContinue: segments.length > 0,
    availableContinuations: segments,
  };
}

/**
 * Enumerate chain capture continuation positions.
 *
 * Returns an array of positions that represent valid landing positions
 * for chain captures from the given position.
 */
export function enumerateChainCaptures(
  state: GameState,
  fromPosition: Position,
  player: number
): Position[] {
  const info = getChainCaptureContinuationInfo(state, player, fromPosition);
  return info.availableContinuations.map((m) => m.to);
}

// ═══════════════════════════════════════════════════════════════════════════
// Mutation Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Apply a capture mutation to the state.
 *
 * This is the core mutation function that handles:
 * - Removing the attacker stack from origin
 * - Placing a marker at origin
 * - Processing markers along the path (flip opponent's, collapse own)
 * - Removing the target stack and transferring rings
 * - Moving the combined stack to landing position
 * - Handling landing on own marker (eliminate bottom ring)
 *
 * Preconditions: The move has been validated by validateCapture.
 */
export function mutateCapture(
  state: GameState,
  action: OvertakingCaptureAction | ContinueChainAction
): GameState {
  // Deep copy state for immutability
  const newState = {
    ...state,
    board: {
      ...state.board,
      stacks: new Map(state.board.stacks),
      markers: new Map(state.board.markers),
      collapsedSpaces: new Map(state.board.collapsedSpaces),
      eliminatedRings: { ...state.board.eliminatedRings },
    },
    players: state.players.map((p) => ({ ...p })),
    moveHistory: [...state.moveHistory],
  } as GameState & {
    totalRingsEliminated: number;
    lastMoveAt: Date;
  };

  const fromKey = positionToString(action.from);
  const targetKey = positionToString(action.captureTarget);
  const toKey = positionToString(action.to);

  const attacker = newState.board.stacks.get(fromKey);
  const target = newState.board.stacks.get(targetKey);

  if (!attacker || !target) {
    throw new Error('CaptureMutator: Missing attacker or target stack');
  }

  // 1. Remove attacker from origin
  newState.board.stacks.delete(fromKey);

  // 2. Place marker at origin
  newState.board.markers.set(fromKey, {
    player: action.playerId,
    position: action.from,
    type: 'regular',
  });

  // 2.5 Process markers along the path (flip opponent's, collapse own)
  // Path includes from -> target -> landing
  const path1 = getPathPositions(action.from, action.captureTarget);
  const path2 = getPathPositions(action.captureTarget, action.to);

  // Combine paths, getting intermediate cells only
  // path1: [from, ..., target]
  // path2: [target, ..., to]
  // We want intermediate cells: path1.slice(1, -1) and path2.slice(1, -1)
  const intermediatePositions = [...path1.slice(1, -1), ...path2.slice(1, -1)];

  for (const pos of intermediatePositions) {
    const key = positionToString(pos);
    const marker = newState.board.markers.get(key);

    if (marker) {
      if (marker.player !== action.playerId) {
        // Flip opponent marker
        newState.board.markers.set(key, {
          ...marker,
          player: action.playerId,
        });
      } else {
        // Collapse own marker
        newState.board.markers.delete(key);
        newState.board.collapsedSpaces.set(key, action.playerId);
      }
    }
  }

  // 3. Remove target stack
  newState.board.stacks.delete(targetKey);

  // 4. Create new stack at landing
  // Rule 4.2.3: Pop the top ring from the target stack and append it to
  // the bottom of the attacking stack's rings array.
  // rings array is [top, ..., bottom] (consistent with calculateCapHeight).
  const capturedRing = target.rings[0];
  const newRings = [...attacker.rings, capturedRing];

  newState.board.stacks.set(toKey, {
    position: action.to,
    rings: newRings,
    stackHeight: newRings.length,
    capHeight: calculateCapHeight(newRings),
    controllingPlayer: attacker.controllingPlayer, // Attacker remains on top
  });

  // 4.5 Update target stack (if rings remain)
  if (target.rings.length > 1) {
    const remainingRings = target.rings.slice(1);
    newState.board.stacks.set(targetKey, {
      ...target,
      rings: remainingRings,
      stackHeight: remainingRings.length,
      capHeight: calculateCapHeight(remainingRings),
      controllingPlayer: remainingRings[0], // New top ring
    });
  }

  // 5. Handle landing on marker (if applicable)
  // Per RR-CANON-R101/R102: landing on any marker (own or opponent) removes the marker
  // and eliminates the top ring of the attacking stack's cap.
  const landingMarker = newState.board.markers.get(toKey);
  if (landingMarker) {
    // Remove the marker (do not collapse it)
    newState.board.markers.delete(toKey);

    const currentStack = newState.board.stacks.get(toKey);
    if (!currentStack) {
      throw new Error(`Expected stack at landing position ${toKey}`);
    }

    // Eliminate the TOP ring of the attacking stack's cap
    // TOP ring is rings[0] per actual codebase convention (consistent with calculateCapHeight)
    const topRingOwner = currentStack.rings[0];
    const reducedRings = currentStack.rings.slice(1); // Remove first element (the top)

    // Update elimination counts
    newState.totalRingsEliminated = (newState.totalRingsEliminated || 0) + 1;
    newState.board.eliminatedRings[topRingOwner] =
      (newState.board.eliminatedRings[topRingOwner] || 0) + 1;

    const player = newState.players.find((p) => p.playerNumber === topRingOwner);
    if (player) {
      player.eliminatedRings++;
    }

    // Update the stack on board with the reduced rings
    if (reducedRings.length > 0) {
      newState.board.stacks.set(toKey, {
        ...currentStack,
        rings: reducedRings,
        stackHeight: reducedRings.length,
        capHeight: calculateCapHeight(reducedRings),
        controllingPlayer: reducedRings[0], // New top ring is the controller
      });
    } else {
      // Stack eliminated completely
      newState.board.stacks.delete(toKey);
    }
  }

  // 6. Update timestamps
  newState.lastMoveAt = new Date();

  return newState;
}

/**
 * Update or initialize the internal chain capture state after an
 * overtaking capture has been successfully applied to the board.
 *
 * This is used by host engines (GameEngine) to track mandatory chain
 * captures. It creates a new state on the first capture segment and
 * updates the existing state for subsequent segments.
 *
 * @param state - Current chain capture state (undefined on first capture)
 * @param move - The capture move that was just applied
 * @param capturedCapHeight - The cap height of the stack that was captured
 * @returns Updated chain capture state, or undefined if move is invalid
 */
export function updateChainCaptureStateAfterCapture(
  state: ChainCaptureState | undefined,
  move: Move,
  capturedCapHeight: number
): ChainCaptureState | undefined {
  if (!move.from || !move.captureTarget || !move.to) {
    return state;
  }

  const segment: ChainCaptureSegment = {
    from: move.from,
    target: move.captureTarget,
    landing: move.to,
    capturedCapHeight,
  };

  if (!state) {
    return {
      playerNumber: move.player,
      startPosition: move.from,
      currentPosition: move.to,
      segments: [segment],
      availableMoves: [],
      visitedPositions: new Set<string>([positionToString(move.from)]),
    };
  }

  // Continuing an existing chain
  state.currentPosition = move.to;
  state.segments.push(segment);
  state.visitedPositions.add(positionToString(move.from));
  return state;
}

/**
 * Apply a capture segment with full marker effects and return detailed outcome.
 *
 * This is a higher-level mutation function that:
 * - Delegates to mutateCapture for the core mutation
 * - Checks for chain capture continuation
 * - Returns detailed outcome information
 *
 * Preconditions: The move has been validated.
 */
export function applyCaptureSegment(
  state: GameState,
  params: CaptureSegmentParams
): CaptureApplicationOutcome {
  const action: OvertakingCaptureAction = {
    type: 'OVERTAKING_CAPTURE',
    playerId: params.player,
    from: params.from,
    captureTarget: params.target,
    to: params.landing,
  };

  const nextState = mutateCapture(state, action);

  // Check if chain continuation is required
  const continuationInfo = getChainCaptureContinuationInfo(
    nextState,
    params.player,
    params.landing
  );

  return {
    nextState,
    ringsTransferred: 1, // Standard capture transfers 1 ring
    chainContinuationRequired: continuationInfo.mustContinue,
  };
}

/**
 * Apply a capture and return a result type for easier error handling.
 *
 * This wrapper catches errors and returns a discriminated union result.
 */
export function applyCapture(state: GameState, move: Move): CaptureMutationResult {
  if (move.type !== 'overtaking_capture' && move.type !== 'continue_capture_segment') {
    return {
      success: false,
      reason: `Expected 'overtaking_capture' or 'continue_capture_segment' move, got '${move.type}'`,
    };
  }

  if (!move.from || !move.captureTarget) {
    return {
      success: false,
      reason: 'Move.from and Move.captureTarget are required for capture moves',
    };
  }

  try {
    const outcome = applyCaptureSegment(state, {
      from: move.from,
      target: move.captureTarget,
      landing: move.to,
      player: move.player,
    });

    // Collect chain capture positions if continuation is required
    const chainCaptures: Position[] = [];
    if (outcome.chainContinuationRequired) {
      const info = getChainCaptureContinuationInfo(outcome.nextState, move.player, move.to);
      for (const continuation of info.availableContinuations) {
        if (continuation.to) {
          chainCaptures.push(continuation.to);
        }
      }
    }

    return {
      success: true,
      newState: outcome.nextState,
      chainCaptures,
    };
  } catch (error) {
    return {
      success: false,
      reason: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}
