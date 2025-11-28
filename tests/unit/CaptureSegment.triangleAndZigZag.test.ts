import { BoardManager } from '../../src/server/game/BoardManager';
import {
  validateCaptureSegmentOnBoard,
  CaptureSegmentBoardView,
} from '../../src/shared/engine/core';
import { BoardType, BoardState, GameState, Position, RingStack } from '../../src/shared/types/game';
import { getChainCaptureContinuationInfo } from '../../src/shared/engine/aggregates/CaptureAggregate';

describe('Capture segment core + enumerator: triangle and zig-zag scenarios', () => {
  const boardType: BoardType = 'square8';

  function createEmptyBoard(): { boardManager: BoardManager; board: BoardState } {
    const boardManager = new BoardManager(boardType);
    const board = boardManager.createBoard();
    return { boardManager, board };
  }

  function setStack(
    boardManager: BoardManager,
    board: BoardState,
    position: Position,
    player: number,
    height: number
  ): void {
    const rings = Array(height).fill(player);
    const stack: RingStack = {
      position,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer: player,
    };
    boardManager.setStack(position, stack, board);
  }

  function makeCaptureView(boardManager: BoardManager, board: BoardState): CaptureSegmentBoardView {
    return {
      isValidPosition: (pos: Position) => boardManager.isValidPosition(pos),
      isCollapsedSpace: (pos: Position) => boardManager.isCollapsedSpace(pos, board),
      getStackAt: (pos: Position) => {
        const stack = boardManager.getStack(pos, board);
        if (!stack) return undefined;
        return {
          controllingPlayer: stack.controllingPlayer,
          capHeight: stack.capHeight,
          stackHeight: stack.stackHeight,
        };
      },
      getMarkerOwner: (pos: Position) => boardManager.getMarker(pos, board),
    };
  }

  it('validateCaptureSegmentOnBoard accepts triangle second and third segments (FAQ 15.3.2)', () => {
    const { boardManager, board } = createEmptyBoard();

    // Board state after first triangle segment:
    // P1 at (3,5) H2; P2 at (4,4) H1; P2 at (4,3) H1.
    setStack(boardManager, board, { x: 3, y: 5 }, 1, 2);
    setStack(boardManager, board, { x: 4, y: 4 }, 2, 1);
    setStack(boardManager, board, { x: 4, y: 3 }, 2, 1);

    const view = makeCaptureView(boardManager, board);

    // Second segment: (3,5) over (4,4) to (5,3).
    const from2: Position = { x: 3, y: 5 };
    const target2: Position = { x: 4, y: 4 };
    const landing2: Position = { x: 5, y: 3 };

    const okSecond = validateCaptureSegmentOnBoard(boardType, from2, target2, landing2, 1, view);
    expect(okSecond).toBe(true);

    // Board state after second segment:
    // For validator purposes we only need the attacker and next target.
    const { boardManager: bm3, board: board3 } = createEmptyBoard();
    setStack(bm3, board3, { x: 5, y: 3 }, 1, 3); // attacker after second capture
    setStack(bm3, board3, { x: 4, y: 3 }, 2, 1); // remaining triangle vertex

    const view3 = makeCaptureView(bm3, board3);

    // Third segment: (5,3) over (4,3) to (2,3).
    const from3: Position = { x: 5, y: 3 };
    const target3: Position = { x: 4, y: 3 };
    const landing3: Position = { x: 2, y: 3 };

    const okThird = validateCaptureSegmentOnBoard(boardType, from3, target3, landing3, 1, view3);
    expect(okThird).toBe(true);
  });

  it('getCaptureOptionsFromPosition enumerates triangle second and third segments', () => {
    // Second segment enumeration
    {
      const { boardManager, board } = createEmptyBoard();
      const from: Position = { x: 3, y: 5 };
      const target: Position = { x: 4, y: 4 };
      const landing: Position = { x: 5, y: 3 };

      setStack(boardManager, board, from, 1, 2);
      setStack(boardManager, board, target, 2, 1);
      setStack(boardManager, board, { x: 4, y: 3 }, 2, 1);

      const gameState = {
        id: 'triangle-second',
        boardType,
        board: { ...board, type: boardType, size: 8 },
        moveHistory: [] as any[],
        currentPlayer: 1,
        currentPhase: 'capture',
      } as unknown as GameState;

      const { availableContinuations: moves } = getChainCaptureContinuationInfo(gameState, 1, from);

      const hasExpected = moves.some(
        (m) =>
          m.player === 1 &&
          m.from &&
          m.captureTarget &&
          m.to &&
          m.from.x === from.x &&
          m.from.y === from.y &&
          m.captureTarget.x === target.x &&
          m.captureTarget.y === target.y &&
          m.to.x === landing.x &&
          m.to.y === landing.y
      );
      expect(hasExpected).toBe(true);
    }

    // Third segment enumeration
    {
      const { boardManager, board } = createEmptyBoard();
      const from: Position = { x: 5, y: 3 };
      const target: Position = { x: 4, y: 3 };
      const landing: Position = { x: 2, y: 3 };

      setStack(boardManager, board, from, 1, 3);
      setStack(boardManager, board, target, 2, 1);

      const gameState = {
        id: 'triangle-third',
        boardType,
        board: { ...board, type: boardType, size: 8 },
        moveHistory: [] as any[],
        currentPlayer: 1,
        currentPhase: 'capture',
      } as unknown as GameState;

      const { availableContinuations: moves } = getChainCaptureContinuationInfo(gameState, 1, from);

      const hasExpected = moves.some(
        (m) =>
          m.player === 1 &&
          m.from &&
          m.captureTarget &&
          m.to &&
          m.from.x === from.x &&
          m.from.y === from.y &&
          m.captureTarget.x === target.x &&
          m.captureTarget.y === target.y &&
          m.to.x === landing.x &&
          m.to.y === landing.y
      );
      expect(hasExpected).toBe(true);
    }
  });

  it('validateCaptureSegmentOnBoard accepts zig-zag second and third segments', () => {
    const { boardManager, board } = createEmptyBoard();

    // Board for second zig-zag segment:
    // P1 at (2,2) H2; P2 at (3,2) H1; P2 at (4,3) H1.
    setStack(boardManager, board, { x: 2, y: 2 }, 1, 2);
    setStack(boardManager, board, { x: 3, y: 2 }, 2, 1);
    setStack(boardManager, board, { x: 4, y: 3 }, 2, 1);

    const view = makeCaptureView(boardManager, board);

    // Second segment: (2,2) over (3,2) to (4,2).
    const from2: Position = { x: 2, y: 2 };
    const target2: Position = { x: 3, y: 2 };
    const landing2: Position = { x: 4, y: 2 };

    const okSecond = validateCaptureSegmentOnBoard(boardType, from2, target2, landing2, 1, view);
    expect(okSecond).toBe(true);

    // Board for third zig-zag segment:
    const { boardManager: bm3, board: board3 } = createEmptyBoard();
    setStack(bm3, board3, { x: 4, y: 2 }, 1, 3);
    setStack(bm3, board3, { x: 4, y: 3 }, 2, 1);

    const view3 = makeCaptureView(bm3, board3);

    // Third segment: (4,2) over (4,3) to (4,5).
    const from3: Position = { x: 4, y: 2 };
    const target3: Position = { x: 4, y: 3 };
    const landing3: Position = { x: 4, y: 5 };

    const okThird = validateCaptureSegmentOnBoard(boardType, from3, target3, landing3, 1, view3);
    expect(okThird).toBe(true);
  });

  it('getCaptureOptionsFromPosition enumerates zig-zag second and third segments', () => {
    // Second segment
    {
      const { boardManager, board } = createEmptyBoard();
      const from: Position = { x: 2, y: 2 };
      const target: Position = { x: 3, y: 2 };
      const landing: Position = { x: 4, y: 2 };

      setStack(boardManager, board, from, 1, 2);
      setStack(boardManager, board, target, 2, 1);
      setStack(boardManager, board, { x: 4, y: 3 }, 2, 1);

      const gameState = {
        id: 'zigzag-second',
        boardType,
        board: { ...board, type: boardType, size: 8 },
        moveHistory: [] as any[],
        currentPlayer: 1,
        currentPhase: 'capture',
      } as unknown as GameState;

      const { availableContinuations: moves } = getChainCaptureContinuationInfo(gameState, 1, from);

      const hasExpected = moves.some(
        (m) =>
          m.player === 1 &&
          m.from &&
          m.captureTarget &&
          m.to &&
          m.from.x === from.x &&
          m.from.y === from.y &&
          m.captureTarget.x === target.x &&
          m.captureTarget.y === target.y &&
          m.to.x === landing.x &&
          m.to.y === landing.y
      );
      expect(hasExpected).toBe(true);
    }

    // Third segment
    {
      const { boardManager, board } = createEmptyBoard();
      const from: Position = { x: 4, y: 2 };
      const target: Position = { x: 4, y: 3 };
      const landing: Position = { x: 4, y: 5 };

      setStack(boardManager, board, from, 1, 3);
      setStack(boardManager, board, target, 2, 1);

      const gameState = {
        id: 'zigzag-third',
        boardType,
        board: { ...board, type: boardType, size: 8 },
        moveHistory: [] as any[],
        currentPlayer: 1,
        currentPhase: 'capture',
      } as unknown as GameState;

      const { availableContinuations: moves } = getChainCaptureContinuationInfo(gameState, 1, from);

      const hasExpected = moves.some(
        (m) =>
          m.player === 1 &&
          m.from &&
          m.captureTarget &&
          m.to &&
          m.from.x === from.x &&
          m.from.y === from.y &&
          m.captureTarget.x === target.x &&
          m.captureTarget.y === target.y &&
          m.to.x === landing.x &&
          m.to.y === landing.y
      );
      expect(hasExpected).toBe(true);
    }
  });
});
