import {
  BoardState,
  GameState,
  Position,
  RingStack,
  positionToString,
} from '../../shared/types/game';
import {
  calculateDistance,
  calculateCapHeight,
  getPathPositions,
  validateCaptureSegmentOnBoard,
  CaptureSegmentBoardView,
} from '../../shared/engine/core';
import {
  applyCaptureSegmentOnBoard,
  enumerateCaptureSegmentsFromBoard,
  CaptureBoardAdapters,
  CaptureApplyAdapters,
} from './sandboxCaptures';
import type { MarkerPathHelpers } from './sandboxMovement';
import { applyMarkerEffectsAlongPathOnBoard } from './sandboxMovement';
import { isSandboxCaptureDebugEnabled } from '../../shared/utils/envFlags';

const isTestEnv =
  typeof process !== 'undefined' &&
  !!(process as any).env &&
  (process as any).env.NODE_ENV === 'test';

const SANDBOX_CAPTURE_DEBUG_ENABLED = isSandboxCaptureDebugEnabled();

/**
 * Hook-based adapter that allows movement and capture application logic to be
 * shared between different sandbox engine hosts (e.g. ClientSandboxEngine),
 * while keeping this module free of direct stateful dependencies.
 */
export interface SandboxMovementEngineHooks {
  getGameState(): GameState;
  setGameState(state: GameState): void;

  // Board/position helpers
  isValidPosition(pos: Position): boolean;
  isCollapsedSpace(pos: Position, board?: BoardState): boolean;
  getMarkerOwner(pos: Position, board?: BoardState): number | undefined;

  // Stack helpers
  getPlayerStacks(playerNumber: number, board: BoardState): RingStack[];

  // Marker helpers used by marker-path processing.
  setMarker(position: Position, playerNumber: number, board: BoardState): void;
  collapseMarker(position: Position, playerNumber: number, board: BoardState): void;
  flipMarker(position: Position, playerNumber: number, board: BoardState): void;

  // Optional hook for disambiguating capture directions when multiple
  // overtaking segments are available.
  chooseCaptureSegment?(
    options: Array<{ from: Position; target: Position; landing: Position }>
  ): Promise<{ from: Position; target: Position; landing: Position } | undefined>;

  // Optional callback invoked for each capture segment in a chain so hosts
  // can record canonical moves/history in a backend-like way.
  onCaptureSegmentApplied?(info: {
    before: GameState;
    after: GameState;
    from: Position;
    target: Position;
    landing: Position;
    playerNumber: number;
    segmentIndex: number;
    isFinal: boolean;
  }): Promise<void> | void;

  // Optional callback invoked after a successful simple (non-capturing)
  // movement so hosts can record canonical move/history entries.
  onSimpleMoveApplied?(info: {
    before: GameState;
    after: GameState;
    from: Position;
    landing: Position;
    playerNumber: number;
  }): Promise<void> | void;

  // Called after any successful movement or capture chain so the host can
  // perform post-movement processing (lines, territory, victory, turn
  // advancement, etc.).
  onMovementComplete(): Promise<void>;
}

/**
 * Internal helper: apply marker effects along the path using the host's
 * marker callbacks so that BoardState invariants remain consistent.
 *
 * The optional options parameter is forwarded to the underlying
 * applyMarkerEffectsAlongPathOnBoard helper so callers can opt out of
 * placing a departure marker when needed (e.g. the second leg of a
 * capture path starting at the intermediate capture stack).
 */
function applyMarkerEffectsAlongPathWithHooks(
  hooks: SandboxMovementEngineHooks,
  from: Position,
  to: Position,
  playerNumber: number,
  board: BoardState,
  options?: { leaveDepartureMarker?: boolean }
): void {
  const helpers: MarkerPathHelpers = {
    setMarker: (pos, player, b) => hooks.setMarker(pos, player, b),
    collapseMarker: (pos, player, b) => hooks.collapseMarker(pos, player, b),
    flipMarker: (pos, player, b) => hooks.flipMarker(pos, player, b),
  };

  applyMarkerEffectsAlongPathOnBoard(board, from, to, playerNumber, helpers, options);
}

/**
 * Handle a human or AI movement click in sandbox mode. This mirrors the
 * movement semantics previously implemented directly on ClientSandboxEngine
 * but operates via hook-based access to the host state.
 */
export async function handleMovementClickSandbox(
  hooks: SandboxMovementEngineHooks,
  selectedFromKey: string | undefined,
  position: Position
): Promise<{
  nextSelectedFromKey: string | undefined;
}> {
  const state = hooks.getGameState();
  const board = state.board;
  const key = positionToString(position);
  const stackAtPos = board.stacks.get(key);

  // Invalid destination clears any selection.
  if (!hooks.isValidPosition(position)) {
    if (selectedFromKey === '2,7')
      console.log('[Sandbox Movement Engine Debug] Invalid position', position);
    return { nextSelectedFromKey: undefined };
  }

  // No source selected yet: select a stack belonging to the current player.
  if (!selectedFromKey) {
    if (stackAtPos && stackAtPos.controllingPlayer === state.currentPlayer) {
      return { nextSelectedFromKey: key };
    }
    return { nextSelectedFromKey: undefined };
  }

  // Clicking the same cell clears selection.
  if (key === selectedFromKey) {
    return { nextSelectedFromKey: undefined };
  }

  const fromKey = selectedFromKey;
  const movingStack = board.stacks.get(fromKey);
  if (!movingStack || movingStack.controllingPlayer !== state.currentPlayer) {
    if (selectedFromKey === '2,7')
      console.log('[Sandbox Movement Engine Debug] Invalid moving stack', {
        fromKey,
        movingStack,
        currentPlayer: state.currentPlayer,
      });
    return { nextSelectedFromKey: undefined };
  }

  const fromPos = movingStack.position;

  // Disallow landing on collapsed spaces.
  if (hooks.isCollapsedSpace(position, board)) {
    if (selectedFromKey === '2,7')
      console.log('[Sandbox Movement Engine Debug] Collapsed destination', position);
    return { nextSelectedFromKey: undefined };
  }

  // Determine whether this click represents a capture or a simple move.
  const fullPath = getPathPositions(fromPos, position);
  if (fullPath.length <= 1) {
    if (selectedFromKey === '2,7')
      console.log('[Sandbox Movement Engine Debug] Path too short', fullPath);
    return { nextSelectedFromKey: undefined };
  }

  const intermediate = fullPath.slice(1, -1);
  let targetPos: Position | undefined;
  for (const pos of intermediate) {
    const posKey = positionToString(pos);
    const stack = board.stacks.get(posKey);
    if (stack) {
      targetPos = pos;
      break;
    }
  }

  // If there is an intermediate stack, attempt an overtaking capture.
  if (targetPos) {
    const view: CaptureSegmentBoardView = {
      isValidPosition: (pos: Position) => hooks.isValidPosition(pos),
      isCollapsedSpace: (pos: Position) => hooks.isCollapsedSpace(pos, board),
      getStackAt: (pos: Position) => {
        const sKey = positionToString(pos);
        const stack = board.stacks.get(sKey);
        if (!stack) return undefined;
        return {
          controllingPlayer: stack.controllingPlayer,
          capHeight: stack.capHeight,
          stackHeight: stack.stackHeight,
        };
      },
      getMarkerOwner: (pos: Position) => hooks.getMarkerOwner(pos, board),
    };

    const isValid = validateCaptureSegmentOnBoard(
      state.boardType,
      fromPos,
      targetPos,
      position,
      movingStack.controllingPlayer,
      view
    );

    if (!isValid) {
      return { nextSelectedFromKey: undefined };
    }

    // Start a mandatory capture chain beginning with this segment.
    await performCaptureChainSandbox(
      hooks,
      fromPos,
      targetPos,
      position,
      movingStack.controllingPlayer
    );

    return { nextSelectedFromKey: undefined };
  }

  // No intermediate stack: treat as a simple non-capturing move.
  for (const pos of intermediate) {
    const pathKey = positionToString(pos);
    if (hooks.isCollapsedSpace(pos, board) || board.stacks.has(pathKey)) {
      if (selectedFromKey === '2,7')
        console.log('[Sandbox Movement Engine Debug] Path blocked', { pos, pathKey });
      return { nextSelectedFromKey: undefined };
    }
  }

  const distance = calculateDistance(state.boardType, fromPos, position);
  if (distance < movingStack.stackHeight) {
    if (selectedFromKey === '2,7')
      console.log('[Sandbox Movement Engine Debug] Distance too short', {
        distance,
        stackHeight: movingStack.stackHeight,
      });
    return { nextSelectedFromKey: undefined };
  }

  // Marker landing rules must mirror the shared movement helpers used by
  // hasAnyLegalMoveOrCaptureFromOnBoard and enumerateSimpleMovementLandings:
  //   - You may land on empty spaces or your own markers.
  //   - You may NOT land on an opponent marker; such a cell is only a
  //     transit point along the ray.
  const destinationStack = board.stacks.get(key);
  const landingMarkerOwner = hooks.getMarkerOwner(position, board);
  if (!destinationStack || destinationStack.stackHeight === 0) {
    if (landingMarkerOwner !== undefined && landingMarkerOwner !== movingStack.controllingPlayer) {
      if (selectedFromKey === '2,7')
        console.log('[Sandbox Movement Engine Debug] Opponent marker at landing', {
          position,
          landingMarkerOwner,
          currentPlayer: state.currentPlayer,
        });
      return { nextSelectedFromKey: undefined };
    }
  }

  const nextStacks = new Map(board.stacks);
  nextStacks.delete(fromKey);

  const existingDest = nextStacks.get(key);
  if (existingDest && existingDest.rings.length > 0) {
    // Merge stacks when landing on an existing stack. The resulting
    // cap height is recalculated from the combined rings so that
    // same-color stacks (e.g. H2 C2) are represented correctly in
    // both sandbox logic and BoardView.
    const mergedRings = [...existingDest.rings, ...movingStack.rings];
    const mergedStack: RingStack = {
      position,
      rings: mergedRings,
      stackHeight: mergedRings.length,
      capHeight: calculateCapHeight(mergedRings),
      controllingPlayer: mergedRings[0],
    };
    nextStacks.set(key, mergedStack);
  } else {
    // Move to an empty position (with no stack). Any same-colour marker
    // on the landing cell will be handled by marker-path processing and
    // subsequent landing-on-own-marker elimination.
    nextStacks.set(key, {
      ...movingStack,
      position,
    });
  }

  // Remember whether this movement lands on an existing same-color marker
  // before marker processing removes it. This mirrors the backend
  // GameEngine behaviour where landing on your own marker immediately
  // eliminates your top ring.
  const landedOnOwnMarker =
    landingMarkerOwner !== undefined && landingMarkerOwner === movingStack.controllingPlayer;

  applyMarkerEffectsAlongPathWithHooks(
    hooks,
    fromPos,
    position,
    movingStack.controllingPlayer,
    board
  );

  let stacksAfterMove: Map<string, RingStack> = nextStacks;
  let eliminatedRingsMap = state.board.eliminatedRings;
  let playersAfterMove = state.players;
  let totalRingsEliminatedDelta = 0;

  if (landedOnOwnMarker) {
    const stackAtLanding = stacksAfterMove.get(key);
    if (stackAtLanding && stackAtLanding.stackHeight > 0) {
      const [, ...remainingRings] = stackAtLanding.rings;

      stacksAfterMove = new Map(stacksAfterMove);

      if (remainingRings.length > 0) {
        const newStack: RingStack = {
          ...stackAtLanding,
          rings: remainingRings,
          stackHeight: remainingRings.length,
          capHeight: calculateCapHeight(remainingRings),
          controllingPlayer: remainingRings[0],
        };
        stacksAfterMove.set(key, newStack);
      } else {
        stacksAfterMove.delete(key);
      }

      const creditedPlayer = movingStack.controllingPlayer;
      eliminatedRingsMap = {
        ...eliminatedRingsMap,
        [creditedPlayer]: (eliminatedRingsMap[creditedPlayer] || 0) + 1,
      };
      playersAfterMove = playersAfterMove.map((p) =>
        p.playerNumber === creditedPlayer ? { ...p, eliminatedRings: p.eliminatedRings + 1 } : p
      );
      totalRingsEliminatedDelta = 1;
    }
  }

  const beforeState: GameState = state;

  const nextState: GameState = {
    ...state,
    board: {
      ...board,
      stacks: stacksAfterMove,
      eliminatedRings: eliminatedRingsMap,
    },
    players: playersAfterMove,
    totalRingsEliminated: state.totalRingsEliminated + totalRingsEliminatedDelta,
  };

  hooks.setGameState(nextState);

  // After any successful movement, notify the host so it can perform
  // post-movement processing (lines, territory, victory, turn
  // advancement, etc.).
  await hooks.onMovementComplete();

  // Allow hosts that care about canonical history to observe this
  // simple movement as a single logical action with pre/post snapshots
  // that include automatic consequences.
  if (hooks.onSimpleMoveApplied) {
    const afterState = hooks.getGameState();
    await hooks.onSimpleMoveApplied({
      before: beforeState,
      after: afterState,
      from: fromPos,
      landing: position,
      playerNumber: state.currentPlayer,
    });
  }

  return { nextSelectedFromKey: undefined };
}

/**
 * Perform an overtaking capture chain starting from an initial segment.
 * Subsequent segments are mandatory: if further captures are available,
 * the engine will either auto-continue (single option) or request a
 * CaptureDirectionChoice via the host's interaction handler.
 *
 * NOTE: This helper only applies capture segments on the board and then
 * delegates post-movement consequences to advanceAfterMovementSandbox.
 */
export async function performCaptureChainSandbox(
  hooks: SandboxMovementEngineHooks,
  initialFrom: Position,
  initialTarget: Position,
  initialLanding: Position,
  playerNumber: number
): Promise<void> {
  let state = hooks.getGameState();
  let currentPosition = initialLanding;
  let from = initialFrom;
  let target = initialTarget;
  let landing = initialLanding;
  let segmentIndex = 0;

  // Apply the initial capture segment and stage it for potential
  // onCaptureSegmentApplied notification once we know whether the
  // chain will continue.
  let before = hooks.getGameState();
  applyCaptureSegmentWithHooks(hooks, from, target, landing, playerNumber, state.board);

  // After the initial segment, mirror backend GameEngine semantics by
  // entering a dedicated 'chain_capture' phase while further
  // continuations remain. This keeps sandbox history/state hashes
  // aligned with backend traces without changing the external
  // turn-sequencing API (the chain is still resolved internally).
  state = hooks.getGameState();
  if (state.currentPhase !== 'chain_capture') {
    hooks.setGameState({
      ...state,
      currentPhase: 'chain_capture',
    });
  }

  let after = hooks.getGameState();

  let pendingSegment: {
    before: GameState;
    after: GameState;
    from: Position;
    target: Position;
    landing: Position;
    playerNumber: number;
    segmentIndex: number;
  } | null = {
    before,
    after,
    from,
    target,
    landing,
    playerNumber,
    segmentIndex,
  };

  // eslint-disable-next-line no-constant-condition
  while (true) {
    state = hooks.getGameState();
    const options = enumerateCaptureSegmentsFromSandbox(
      hooks,
      currentPosition,
      playerNumber,
      state.board
    );

    if (SANDBOX_CAPTURE_DEBUG_ENABLED) {
      // eslint-disable-next-line no-console
      console.log(
        'Sandbox capture chain debug:',
        'currentPosition=',
        currentPosition,
        'options=',
        options.map((opt) => ({
          from: opt.from,
          target: opt.target,
          landing: opt.landing,
        }))
      );
    }

    if (options.length === 0) {
      // No further continuations. The currently pending segment is the
      // final segment in the chain. Perform post-movement processing
      // first so that hosts see the same "after" state a backend
      // GameEngine history entry would record (after automatic
      // consequences and turn advancement), then notify the optional
      // onCaptureSegmentApplied hook exactly once.
      await hooks.onMovementComplete();

      if (hooks.onCaptureSegmentApplied && pendingSegment) {
        const finalAfter = hooks.getGameState();
        await hooks.onCaptureSegmentApplied({
          ...pendingSegment,
          after: finalAfter,
          isFinal: true,
        });
      }

      return;
    }

    // At least one continuation exists, so the pending segment is an
    // intermediate link in the chain. Notify hosts about it now using
    // the board state immediately after the segment was applied.
    if (hooks.onCaptureSegmentApplied && pendingSegment) {
      await hooks.onCaptureSegmentApplied({
        ...pendingSegment,
        isFinal: false,
      });
    }

    let nextSegment: { from: Position; target: Position; landing: Position } | undefined;

    if (hooks.chooseCaptureSegment) {
      nextSegment = await hooks.chooseCaptureSegment(options);
    } else {
      nextSegment = options[0];
    }

    if (!nextSegment) {
      // Host declined to choose a continuation; treat the chain as
      // terminated without further consequences.
      return;
    }

    from = currentPosition;
    target = nextSegment.target;
    landing = nextSegment.landing;

    // Apply the chosen continuation and stage it as the next pending
    // segment. Post-movement consequences (lines/territory/turn
    // advancement) are still deferred until the chain is exhausted.
    before = hooks.getGameState();
    applyCaptureSegmentWithHooks(hooks, from, target, landing, playerNumber, state.board);
    after = hooks.getGameState();

    segmentIndex += 1;
    pendingSegment = {
      before,
      after,
      from,
      target,
      landing,
      playerNumber,
      segmentIndex,
    };

    currentPosition = landing;
  }
}

/**
 * Enumerate all legal overtaking capture segments for the given player from
 * the specified stack position, using the same adapters as the main engine.
 */
export function enumerateCaptureSegmentsFromSandbox(
  hooks: SandboxMovementEngineHooks,
  from: Position,
  playerNumber: number,
  board: BoardState
): Array<{ from: Position; target: Position; landing: Position }> {
  const adapters: CaptureBoardAdapters = {
    isValidPosition: (pos: Position) => hooks.isValidPosition(pos),
    isCollapsedSpace: (pos: Position, b: BoardState) => hooks.isCollapsedSpace(pos, b),
    getMarkerOwner: (pos: Position, b: BoardState) => hooks.getMarkerOwner(pos, b),
  };

  return enumerateCaptureSegmentsFromBoard(
    hooks.getGameState().boardType,
    board,
    from,
    playerNumber,
    adapters
  );
}

function applyCaptureSegmentWithHooks(
  hooks: SandboxMovementEngineHooks,
  from: Position,
  target: Position,
  landing: Position,
  playerNumber: number,
  board: BoardState
): void {
  const landingKey = positionToString(landing);

  if (SANDBOX_CAPTURE_DEBUG_ENABLED) {
    const attackerBefore = board.stacks.get(positionToString(from));
    const targetBefore = board.stacks.get(positionToString(target));
    // eslint-disable-next-line no-console
    console.log('Sandbox capture segment before:', {
      from,
      target,
      landing,
      attackerBefore,
      targetBefore,
    });
  }

  // Detect whether we are about to land on an existing same-color marker
  // before marker processing removes it.
  const landingMarkerOwner = hooks.getMarkerOwner(landing, board);
  const landedOnOwnMarker = landingMarkerOwner === playerNumber;

  const adapters: CaptureApplyAdapters = {
    applyMarkerEffectsAlongPath: (f, t, player, options) =>
      applyMarkerEffectsAlongPathWithHooks(hooks, f, t, player, board, options),
  };

  applyCaptureSegmentOnBoard(board, from, target, landing, playerNumber, adapters);

  if (SANDBOX_CAPTURE_DEBUG_ENABLED) {
    const attackerAfter = board.stacks.get(positionToString(landing));
    const targetAfter = board.stacks.get(positionToString(target));
    // eslint-disable-next-line no-console
    console.log('Sandbox capture segment after:', {
      from,
      target,
      landing,
      attackerAfter,
      targetAfter,
    });
  }

  const stacksAfterCapture: Map<string, RingStack> = new Map(board.stacks);
  let eliminatedRingsMap = board.eliminatedRings;
  const state = hooks.getGameState();
  let playersAfterCapture = state.players;
  let totalRingsEliminatedDelta = 0;

  if (landedOnOwnMarker) {
    const stackAtLanding = stacksAfterCapture.get(landingKey);
    if (stackAtLanding && stackAtLanding.stackHeight > 0) {
      const [, ...remainingRings] = stackAtLanding.rings;

      if (remainingRings.length > 0) {
        const newStack: RingStack = {
          ...stackAtLanding,
          rings: remainingRings,
          stackHeight: remainingRings.length,
          capHeight: calculateCapHeight(remainingRings),
          controllingPlayer: remainingRings[0],
        };
        stacksAfterCapture.set(landingKey, newStack);
      } else {
        stacksAfterCapture.delete(landingKey);
      }

      const creditedPlayer = playerNumber;
      eliminatedRingsMap = {
        ...eliminatedRingsMap,
        [creditedPlayer]: (eliminatedRingsMap[creditedPlayer] || 0) + 1,
      };
      playersAfterCapture = playersAfterCapture.map((p) =>
        p.playerNumber === creditedPlayer ? { ...p, eliminatedRings: p.eliminatedRings + 1 } : p
      );
      totalRingsEliminatedDelta = 1;
    }
  }

  const nextState: GameState = {
    ...state,
    board: {
      ...board,
      stacks: stacksAfterCapture,
      markers: new Map(board.markers),
      collapsedSpaces: new Map(board.collapsedSpaces),
      eliminatedRings: eliminatedRingsMap,
    },
    players: playersAfterCapture,
    totalRingsEliminated: state.totalRingsEliminated + totalRingsEliminatedDelta,
  };

  hooks.setGameState(nextState);
}

// Post-movement advancement (lines, territory, victory, next-player turn
// setup) is intentionally delegated to the host via hooks.onMovementComplete.
// This keeps turn-sequencing semantics centralized in the host engine
// (e.g. ClientSandboxEngine) and avoids duplicating stalemate/victory rules
// here.
