import { GameEngine } from '../../src/server/game/GameEngine';
import { Move, Position, BoardState, positionToString } from '../../src/shared/types/game';
import {
  enumerateCaptureSegmentsFromBoard,
  CaptureBoardAdapters,
} from '../../src/client/sandbox/sandboxCaptures';
import { BOARD_CONFIGS } from '../../src/shared/types/game';

describe('Parity Debug: Seed 14', () => {
  test('Minimal Repro: Vertical Overtaking Capture (Range 3)', async () => {
    // Correct GameEngine constructor usage
    const engine = new GameEngine(
      'test-game',
      'square8',
      [
        {
          id: 'p1',
          username: 'Player 1',
          type: 'human',
          playerNumber: 1,
          isReady: true,
          timeRemaining: 600,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'p2',
          username: 'Player 2',
          type: 'human',
          playerNumber: 2,
          isReady: true,
          timeRemaining: 600,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ],
      { type: 'rapid', initialTime: 600, increment: 0 },
      false
    );

    const gameEngine = engine as any;

    // Force state to Capture phase (since RuleEngine only checks captures in this phase)
    gameEngine.gameState.currentPhase = 'capture';
    gameEngine.gameState.currentPlayer = 2; // P2's turn

    // Clear board by creating a new one
    gameEngine.gameState.board = gameEngine.boardManager.createBoard();

    // Place P2 stack at (1,3) with height 3
    gameEngine.boardManager.setStack(
      { x: 1, y: 3 },
      {
        position: { x: 1, y: 3 },
        controllingPlayer: 2,
        stackHeight: 3,
        capHeight: 3,
        isCap: false,
        rings: [2, 2, 2], // Mock rings
      },
      gameEngine.gameState.board
    );

    // Place P1 stack at (1,4) with height 1 (target)
    gameEngine.boardManager.setStack(
      { x: 1, y: 4 },
      {
        position: { x: 1, y: 4 },
        controllingPlayer: 1,
        stackHeight: 1,
        capHeight: 1,
        isCap: false,
        rings: [1], // Mock rings
      },
      gameEngine.gameState.board
    );

    // Ensure (1,6) is empty

    // 1. Check Backend Moves
    const backendMoves = gameEngine.getValidMovesDebug(2);

    const targetMove = backendMoves.find(
      (m: Move) =>
        m.type === 'overtaking_capture' &&
        m.from &&
        m.from.x === 1 &&
        m.from.y === 3 &&
        m.to.x === 1 &&
        m.to.y === 6
    );

    console.log('Backend allows capture:', !!targetMove);

    // 2. Check Sandbox Moves (via direct capture enumeration)
    const boardState = gameEngine.gameState.board;
    const fromPos: Position = { x: 1, y: 3 };

    // Simple adapter implementation for the test
    const adapters: CaptureBoardAdapters = {
      isValidPosition: (pos: Position) => {
        const config = BOARD_CONFIGS['square8'];
        // Square boards are size x size
        return pos.x >= 0 && pos.x < config.size && pos.y >= 0 && pos.y < config.size;
      },
      isCollapsedSpace: (pos: Position, board: BoardState) => {
        return false;
      },
      getMarkerOwner: (pos: Position, board: BoardState) => {
        return undefined;
      },
    };

    const sandboxSegments = enumerateCaptureSegmentsFromBoard(
      'square8',
      boardState,
      fromPos,
      2, // playerNumber
      adapters
    );

    const sandboxTargetMove = sandboxSegments.find(
      (s) => s.landing.x === 1 && s.landing.y === 6 && s.target.x === 1 && s.target.y === 4
    );

    console.log('Sandbox allows capture:', !!sandboxTargetMove);

    if (sandboxTargetMove && !targetMove) {
      console.log('MISMATCH REPRODUCED: Sandbox allows capture, Backend does not.');
    } else if (sandboxTargetMove && targetMove) {
      console.log('BOTH ALLOW: Mismatch NOT reproduced with this simple state.');
    } else {
      console.log('NEITHER ALLOW: Mismatch NOT reproduced.');
    }

    // Assert that they should match
    expect(!!targetMove).toBe(!!sandboxTargetMove);
  });
});
