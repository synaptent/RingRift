import { useState, useCallback, useRef, useEffect } from 'react';
import type { Move, Position, GameState } from '../../shared/types/game';
import { stringToPosition } from '../../shared/types/game';
import { debugLog, isSandboxAnimationDebugEnabled } from '../../shared/utils/envFlags';
import type { MoveAnimationData } from '../components/BoardView';

/**
 * Hook to manage move animations.
 * Generates animation data from moves and manages the animation lifecycle.
 */
export function useMoveAnimation() {
  const [pendingAnimation, setPendingAnimation] = useState<MoveAnimationData | null>(null);
  const animationIdRef = useRef(0);

  /**
   * Generate animation data from a move.
   * For chain captures, provide the full capture path in intermediatePositions.
   */
  const triggerAnimation = useCallback(
    (
      move: Move,
      playerNumber: number,
      options?: {
        stackHeight?: number;
        capHeight?: number;
        chainCapturePath?: Position[] | undefined;
      }
    ) => {
      const id = `anim-${++animationIdRef.current}`;

      // Determine animation type
      let type: MoveAnimationData['type'] = 'move';
      if (move.type === 'place_ring') {
        type = 'place';
      } else if (move.type === 'overtaking_capture') {
        type =
          options?.chainCapturePath && options.chainCapturePath.length > 0
            ? 'chain_capture'
            : 'capture';
      }

      const animation: MoveAnimationData = {
        type,
        ...(move.from ? { from: move.from } : {}),
        to: move.to,
        ...(options?.chainCapturePath ? { intermediatePositions: options.chainCapturePath } : {}),
        playerNumber,
        ...(options?.stackHeight !== undefined ? { stackHeight: options.stackHeight } : {}),
        ...(options?.capHeight !== undefined ? { capHeight: options.capHeight } : {}),
        id,
      };

      setPendingAnimation(animation);
    },
    []
  );

  /**
   * Clear the current animation (called when animation completes).
   */
  const clearAnimation = useCallback(() => {
    setPendingAnimation(null);
  }, []);

  return {
    pendingAnimation,
    triggerAnimation,
    clearAnimation,
  };
}

/**
 * Hook that automatically detects moves from game state changes
 * and triggers animations.
 */
export function useAutoMoveAnimation(gameState: GameState | null) {
  const { pendingAnimation, triggerAnimation, clearAnimation } = useMoveAnimation();
  const prevMoveCountRef = useRef<number>(0);
  const prevGameIdRef = useRef<string | null>(null);
  const prevBoardRef = useRef<GameState['board'] | null>(null);

  useEffect(() => {
    if (!gameState) {
      // When there is no active game, ensure any leftover animation
      // from a previous game is cleared so the board renders cleanly.
      clearAnimation();
      prevMoveCountRef.current = 0;
      prevGameIdRef.current = null;
      prevBoardRef.current = null;
      return;
    }

    // Reset on game change
    if (prevGameIdRef.current !== gameState.id) {
      prevGameIdRef.current = gameState.id;
      prevMoveCountRef.current = gameState.moveHistory.length;
      // Clear any pending animation from the previous game so a freshly
      // started sandbox match does not show a stale destination pulse.
      clearAnimation();
      prevBoardRef.current = gameState.board;
      return;
    }

    const currentMoveCount = gameState.moveHistory.length;
    const prevMoveCount = prevMoveCountRef.current;
    const prevBoard = prevBoardRef.current;

    // Check if a new move was made
    if (currentMoveCount > prevMoveCount && currentMoveCount > 0) {
      const lastMove = gameState.moveHistory[currentMoveCount - 1];

      if (lastMove && lastMove.to) {
        // Determine from/to positions as robustly as possible. Prefer the
        // explicit move fields, but fall back to a simple board diff when
        // from is missing (or when engines emit minimal moves).
        let from: Position | undefined = lastMove.from;
        let to: Position | undefined = lastMove.to;

        if ((!from || !to) && prevBoard) {
          const fromTo = deriveMovePositionsFromBoards(prevBoard, gameState.board, lastMove.player);
          from = from ?? fromTo.from;
          to = to ?? fromTo.to;
        }

        if (to) {
          const playerNumber = lastMove.player ?? gameState.currentPlayer;

          // Look for chain capture path in recent move history
          let chainCapturePath: Position[] | undefined;
          if (lastMove.type === 'overtaking_capture') {
            const recentMoves = gameState.moveHistory.slice(-10);
            const chainMoves = recentMoves.filter(
              (m) =>
                m.type === 'overtaking_capture' &&
                m.player === playerNumber &&
                m.moveNumber === lastMove.moveNumber
            );

            if (chainMoves.length > 1) {
              // Build the chain path from all captures in this turn
              chainCapturePath = chainMoves.slice(0, -1).map((m) => m.to);
            }
          }

          // Get stack info from the destination
          const destKey = `${to.x},${to.y}${(to as Position & { z?: number }).z !== undefined ? `,${(to as Position & { z?: number }).z}` : ''}`;
          const destStack = gameState.board.stacks.get(destKey);

          const moveForAnimation: Move = {
            ...lastMove,
            ...(from ? { from } : {}),
            to,
          };

          debugLog(
            isSandboxAnimationDebugEnabled(),
            '[SandboxAnimationDebug] useAutoMoveAnimation: triggering animation',
            {
              gameId: gameState.id,
              moveNumber: lastMove.moveNumber,
              moveType: lastMove.type,
              explicitFrom: lastMove.from,
              explicitTo: lastMove.to,
              derivedFrom: from,
              derivedTo: to,
              destKey,
              destStackHeight: destStack?.stackHeight ?? null,
              destCapHeight: destStack?.capHeight ?? null,
            }
          );

          triggerAnimation(moveForAnimation, playerNumber, {
            stackHeight: destStack?.stackHeight ?? 1,
            capHeight: destStack?.capHeight ?? 1,
            chainCapturePath,
          });
        }
      }
    }

    prevMoveCountRef.current = currentMoveCount;
    prevBoardRef.current = gameState.board;
  }, [gameState, triggerAnimation, clearAnimation]);

  return {
    pendingAnimation,
    clearAnimation,
  };
}

/**
 * Heuristic board-diff helper to infer movement origins and destinations
 * when Move.from is missing or incomplete. This keeps animations robust
 * even when some engines emit minimal move payloads.
 */
function deriveMovePositionsFromBoards(
  prevBoard: GameState['board'],
  nextBoard: GameState['board'],
  _playerNumber: number
): { from?: Position; to?: Position } {
  const removedKeys: string[] = [];
  const addedKeys: string[] = [];

  // Detect stacks that disappeared or appeared between states.
  for (const key of prevBoard.stacks.keys()) {
    if (!nextBoard.stacks.has(key)) {
      removedKeys.push(key);
    }
  }

  for (const key of nextBoard.stacks.keys()) {
    if (!prevBoard.stacks.has(key)) {
      addedKeys.push(key);
    }
  }

  const result: { from?: Position; to?: Position } = {};

  if (removedKeys.length === 1) {
    result.from = stringToPosition(removedKeys[0]);
  }

  if (addedKeys.length === 1) {
    result.to = stringToPosition(addedKeys[0]);
  }

  return result;
}
