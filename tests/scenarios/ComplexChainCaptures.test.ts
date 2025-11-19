import { GameEngine } from '../../src/server/game/GameEngine';
import { Position, Player, BoardType, TimeControl, RingStack } from '../../src/shared/types/game';

/**
 * Scenario Tests: Complex Chain Captures
 *
 * Covers:
 * - 180° Reversal Pattern (FAQ 15.3.1)
 * - Cyclic Patterns (FAQ 15.3.2)
 * - Multi-step chains with direction changes
 */

describe('Scenario: Complex Chain Captures', () => {
  beforeAll(() => {
    jest.useFakeTimers();
  });

  afterAll(() => {
    jest.useRealTimers();
  });

  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  const basePlayers: Player[] = [
    {
      id: 'p1',
      username: 'Player1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player2',
      type: 'human',
      playerNumber: 2,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  function createEngine(boardType: BoardType = 'square8'): GameEngine {
    return new GameEngine('scenario-chain', boardType, basePlayers, timeControl, false);
  }

  // Helper to set up board state
  function setupBoard(
    engine: GameEngine,
    stacks: { pos: Position; player: number; height: number }[]
  ) {
    const engineAny: any = engine;
    const boardManager = engineAny.boardManager;
    const gameState = engineAny.gameState;

    // Clear existing stacks
    gameState.board.stacks.clear();

    for (const s of stacks) {
      const rings = Array(s.height).fill(s.player);
      const stack: RingStack = {
        position: s.pos,
        rings,
        stackHeight: rings.length,
        capHeight: rings.length,
        controllingPlayer: s.player,
      };
      boardManager.setStack(s.pos, stack, gameState.board);
    }

    // Force capture phase
    gameState.currentPhase = 'capture';
    gameState.currentPlayer = 1;
  }

  test('Cyclic Pattern: Triangle Loop', async () => {
    // Setup a triangle of opponent stacks that allows a cyclic capture
    // P1 at (2,2) H1
    // P2 at (2,3) H1
    // P2 at (3,4) H1
    // P2 at (1,4) H1
    //
    // Path:
    // (2,2) -> jump (2,3) -> land (2,4) [H2]
    // (2,4) -> jump (3,4) -> land (4,4) [H3] - wait, this drifts away.
    //
    // Let's try a tighter loop on 19x19 to allow more space if needed, but 8x8 is fine.
    //
    // Better Cyclic Setup:
    // P1 at (3,3) H1
    // P2 at (3,4) H1
    // P2 at (4,4) H1
    // P2 at (4,3) H1
    //
    // 1. (3,3) jumps (3,4) -> lands (3,5). Stack H2.
    // 2. (3,5) jumps (4,4) [diagonal back-left] -> lands (5,3). Stack H3.
    // 3. (5,3) jumps (4,3) [left] -> lands (3,3). Stack H4.
    // Back at start!

    const engine = createEngine('square8');
    const startPos = { x: 3, y: 3 };
    const target1 = { x: 3, y: 4 };
    const target2 = { x: 4, y: 4 };
    const target3 = { x: 4, y: 3 };

    setupBoard(engine, [
      { pos: startPos, player: 1, height: 1 },
      { pos: target1, player: 2, height: 1 },
      { pos: target2, player: 2, height: 1 },
      { pos: target3, player: 2, height: 1 },
    ]);

    // Step 1: Jump (3,3) over (3,4) to (3,5)
    // This might auto-chain if subsequent moves are forced.
    const step1 = await engine.makeMove({
      player: 1,
      type: 'overtaking_capture',
      from: startPos,
      captureTarget: target1,
      to: { x: 3, y: 5 },
    } as any);
    expect(step1.success).toBe(true);

    const engineAny: any = engine;
    const board = engineAny.gameState.board;

    // Check if chain completed automatically
    const finalStack = board.stacks.get('2,3');

    if (finalStack) {
      // Auto-completed
      expect(finalStack.stackHeight).toBe(4);
      expect(finalStack.controllingPlayer).toBe(1);
      expect(board.stacks.get('3,4')).toBeUndefined();
      expect(board.stacks.get('4,4')).toBeUndefined();
      expect(board.stacks.get('4,3')).toBeUndefined();
    } else {
      // Manual steps needed

      // Step 2: Jump (3,5) over (4,4) to (5,3)
      const step2 = await engine.makeMove({
        player: 1,
        type: 'overtaking_capture',
        from: { x: 3, y: 5 },
        captureTarget: target2,
        to: { x: 5, y: 3 },
      } as any);
      expect(step2.success).toBe(true);

      // Step 3: Jump (5,3) over (4,3) to (2,3)
      const step3 = await engine.makeMove({
        player: 1,
        type: 'overtaking_capture',
        from: { x: 5, y: 3 },
        captureTarget: target3,
        to: { x: 2, y: 3 },
      } as any);
      expect(step3.success).toBe(true);

      const finalStackManual = board.stacks.get('2,3');
      expect(finalStackManual).toBeDefined();
      expect(finalStackManual.stackHeight).toBe(4);
    }
  });
});
