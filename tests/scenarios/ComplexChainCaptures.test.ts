import { GameEngine } from '../../src/server/game/GameEngine';
import { Position, Player, BoardType, TimeControl, RingStack, GameState } from '../../src/shared/types/game';
import { getCaptureOptionsFromPosition as getCaptureOptionsFromPositionShared } from '../../src/server/game/rules/captureChainEngine';
import { BoardManager } from '../../src/server/game/BoardManager';
import { RuleEngine } from '../../src/server/game/RuleEngine';

/**
 * Scenario Tests: Complex Chain Captures
 *
 * Covers:
 * - 180° Reversal Pattern (FAQ 15.3.1)
 * - Cyclic Patterns (FAQ 15.3.2)
 * - Multi-step chains with direction changes
 */

describe('Scenario: Complex Chain Captures (FAQ 15.3.1, 15.3.2)', () => {
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

  /**
   * Resolve any active capture chain for the current player by repeatedly
   * applying continue_capture_segment moves from GameEngine.getValidMoves
   * while the game remains in the 'chain_capture' phase.
   *
   * This mirrors the unified Move-based chain-capture model and is shared
   * across the FAQ-style scenarios in this file.
   */
  async function resolveChainIfPresent(engine: GameEngine): Promise<void> {
    const engineAny: any = engine;

    const MAX_STEPS = 16;
    let steps = 0;

    // No-op when no chain is currently active.
    if ((engineAny.gameState as GameState).currentPhase !== 'chain_capture') {
      return;
    }

    while ((engineAny.gameState as GameState).currentPhase === 'chain_capture') {
      steps++;
      if (steps > MAX_STEPS) {
        throw new Error('resolveChainIfPresent: exceeded maximum chain-capture steps');
      }

      const state = engineAny.gameState as GameState;
      const currentPlayer = state.currentPlayer;
      const moves = engine.getValidMoves(currentPlayer);

      // eslint-disable-next-line no-console
      console.log('resolveChainIfPresent (Complex) debug', {
        phase: state.currentPhase,
        currentPlayer,
        moveCount: moves.length,
        moveTypes: moves.map((m: any) => m.type),
      });

      const chainMoves = moves.filter((m: any) => m.type === 'continue_capture_segment');

      expect(chainMoves.length).toBeGreaterThan(0);

      const next = chainMoves[0];

      const result = await engine.makeMove({
        player: next.player,
        type: 'continue_capture_segment',
        from: next.from,
        captureTarget: next.captureTarget,
        to: next.to,
      } as any);

      expect(result.success).toBe(true);
    }
  }

  test('FAQ_15_3_2_CyclicPattern_TriangleLoop', async () => {
    // Cyclic triangle pattern (FAQ 15.3.2).
    //
    // Setup:
    // P1 at (3,3) H1
    // P2 at (3,4) H1
    // P2 at (4,4) H1
    // P2 at (4,3) H1
    //
    // The rules allow a closed-loop chain where P1 overtakes all three P2 stacks
    // and returns to the original file. Under the unified chain_capture model,
    // we:
    //   - start the chain with a single overtaking_capture, then
    //   - let the engine enumerate and apply any mandatory continuation segments
    //     via continue_capture_segment moves in the 'chain_capture' phase.
    // We assert only aggregate outcomes (final heights / control), not the
    // exact landing coordinate, to remain tolerant of different but still-legal
    // paths chosen by the engine.
 
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
 
    // Start the chain: (3,3) jumps (3,4) to land at (3,5).
    const step1 = await engine.makeMove({
      player: 1,
      type: 'overtaking_capture',
      from: startPos,
      captureTarget: target1,
      to: { x: 3, y: 5 },
    } as any);
    expect(step1.success).toBe(true);

    // Debug: directly probe the shared chain enumerator from the chain
    // position, mirroring GameEngine.chainCapture.triangleAndZigZagState.test.ts.
    const engineAnyDebug: any = engine;
    const stateAfterDebug: GameState = engineAnyDebug.gameState as GameState;
    const boardManagerDebug: BoardManager = engineAnyDebug.boardManager as BoardManager;
    const ruleEngineDebug: RuleEngine = engineAnyDebug.ruleEngine as RuleEngine;
    const chainStateDebug = engineAnyDebug.chainCaptureState as
      | { currentPosition: Position }
      | undefined;

    if (chainStateDebug) {
      const followUpsDebug = getCaptureOptionsFromPositionShared(
        chainStateDebug.currentPosition,
        1,
        stateAfterDebug,
        {
          boardManager: boardManagerDebug,
          ruleEngine: ruleEngineDebug,
        }
      );

      // eslint-disable-next-line no-console
      console.log('FAQ triangle followUps', followUpsDebug);

      expect(followUpsDebug.length).toBeGreaterThan(0);

      const hasExpectedDebug = followUpsDebug.some(
        (m) =>
          m.player === 1 &&
          m.from &&
          m.captureTarget &&
          m.to &&
          m.from.x === 3 &&
          m.from.y === 5 &&
          m.captureTarget.x === 4 &&
          m.captureTarget.y === 4 &&
          m.to.x === 5 &&
          m.to.y === 3
      );
      expect(hasExpectedDebug).toBe(true);
    }
 
    // Resolve any mandatory capture continuations via the chain_capture phase.
    await resolveChainIfPresent(engine);
 
    const engineAny: any = engine;
    const board = engineAny.gameState.board;
    const stacks = board.stacks as Map<string, RingStack>;
    const allStacks: RingStack[] = Array.from(stacks.values());
 
    const blueStacks = allStacks.filter((s) => s.controllingPlayer === 1);
    const redStacks = allStacks.filter((s) => s.controllingPlayer === 2);
 
    // One Blue-controlled stack of height 4, no remaining Red stacks.
    expect(blueStacks.length).toBe(1);
    expect(blueStacks[0].stackHeight).toBe(4);
    expect(blueStacks[0].controllingPlayer).toBe(1);
    expect(redStacks.length).toBe(0);
  });

  test('FAQ_15_3_1_180_degree_reversal_basic', async () => {
    // Rules reference:
    // - Section 10.3 (Chain Overtaking)
    // - FAQ 15.3.1 (180° Reversal Pattern)
    //
    // Conceptual mapping to the rules example:
    // - Blue: stack height 4 at A
    // - Red: stack height 3 at B on the same line
    // - Plenty of empty spaces on both sides of the line
    //
    // This scenario focuses on the *effect* of a legal 180° reversal sequence
    // rather than prescribing a specific landing coordinate. Starting from
    // a simple A–B–(empty...) line, we:
    // - Perform an initial overtaking capture from A over B.
    // - Let the engine drive any mandatory follow-up chain captures.
    // - Assert that the final board state matches the FAQ’s cumulative
    //   effect: Blue has overtaken twice from the same stack at B.

    const engine = createEngine('square19');

    // Use a simple horizontal line on rank y = 4
    const A: Position = { x: 4, y: 4 }; // Blue start
    const B: Position = { x: 6, y: 4 }; // Red target stack
    const C: Position = { x: 8, y: 4 }; // First landing point (one legal capture segment)

    // Blue height 4 at A; Red height 3 at B.
    setupBoard(engine, [
      { pos: A, player: 1, height: 4 },
      { pos: B, player: 2, height: 3 },
    ]);

    const engineAnyLocal: any = engine;
 
    const step1Local = await engine.makeMove({
      player: 1,
      type: 'overtaking_capture',
      from: A,
      captureTarget: B,
      to: C,
    } as any);
 
    expect(step1Local.success).toBe(true);
 
    // Drive any mandatory follow-up segments via the unified chain_capture
    // phase so that the final board state reflects the full 180° reversal.
    await resolveChainIfPresent(engine);
 
    const boardLocal = engineAnyLocal.gameState.board;
    const stacks = boardLocal.stacks as Map<string, RingStack>;
    const allStacks: RingStack[] = Array.from(stacks.values());

    const blueStacks: RingStack[] = allStacks.filter((s) => s.controllingPlayer === 1);
    const redStacksAtB = stacks.get('6,4');

    // There should be exactly one Blue-controlled stack on the board after the
    // chain completes (the overtaker). Its exact coordinate depends on which
    // landing the engine chose for the second segment, so we only check count
    // and heights, not position.
    expect(blueStacks.length).toBe(1);

    const finalBlue = blueStacks[0];

    // Starting from 4 Blue rings and a 3-ring Red stack at B, a 180° reversal
    // pattern that overtakes twice from B should leave:
    // - Blue with 6 rings in the overtaker stack (4 original + 2 captured).
    // - Red’s original stack at B reduced to a single ring (height 1).
    // We assert these aggregate effects without assuming the exact landing
    // coordinate of the final segment.
    expect(finalBlue.stackHeight).toBe(6);
    expect(finalBlue.controllingPlayer).toBe(1);

    expect(redStacksAtB).toBeDefined();
    expect(redStacksAtB!.stackHeight).toBe(1);
  });
  test('Strategic_Chain_Ending_Choice', async () => {
    // Rules reference: Section 10.3 (Strategic Chain-Ending)
    // "You can deliberately choose a capture that leads to a position with NO further legal captures,
    // thus ending the mandatory chain—even if other available capture choices... would have allowed it to continue longer."
    //
    // Setup:
    // P1 at (3,3) H1
    // P2 at (3,4) H1 (Option A: leads to dead end)
    // P2 at (4,3) H1 (Option B: leads to more captures)
    // P2 at (6,3) H1 (Target for Option B continuation)
    //
    // Option A: (3,3) jumps (3,4) -> lands (3,5). No further captures. Chain ends.
    // Option B: (3,3) jumps (4,3) -> lands (5,3). From (5,3), can jump (6,3) -> (7,3).
    //
    // We verify that P1 can choose Option A and stop.

    const engine = createEngine('square8');
    const startPos = { x: 3, y: 3 };
    const targetA = { x: 3, y: 4 }; // Dead end path
    const targetB = { x: 4, y: 3 }; // Continuation path
    const targetB2 = { x: 6, y: 3 };

    setupBoard(engine, [
      { pos: startPos, player: 1, height: 1 },
      { pos: targetA, player: 2, height: 1 },
      { pos: targetB, player: 2, height: 1 },
      { pos: targetB2, player: 2, height: 1 },
    ]);

    // Choose Option A (Dead End)
    const step1 = await engine.makeMove({
      player: 1,
      type: 'overtaking_capture',
      from: startPos,
      captureTarget: targetA,
      to: { x: 3, y: 5 },
    } as any);

    expect(step1.success).toBe(true);

    const engineAny: any = engine;
    const board = engineAny.gameState.board;

    // Verify chain ended
    const finalStack = board.stacks.get('3,5');
    expect(finalStack).toBeDefined();
    expect(finalStack.stackHeight).toBe(2);
    expect(finalStack.controllingPlayer).toBe(1);

    // Verify other targets remain
    expect(board.stacks.get('4,3')).toBeDefined();
    expect(board.stacks.get('6,3')).toBeDefined();
  });

  test('Multi_Directional_ZigZag_Chain', async () => {
    // Setup a zig-zag chain:
    // P1 at (0,0) H1
    // P2 at (1,1) H1
    // P2 at (3,1) H1
    // P2 at (3,3) H1
    //
    // Path:
    // 1. (0,0) -> (1,1) -> (2,2) [SE]
    // 2. (2,2) -> (3,1) -> (4,0) [NE]
    // 3. (4,0) -> (3,3) -> (2,6) [Wait, (4,0) to (3,3) is not straight line?]
    // (4,0) is x=4, y=0. (3,3) is x=3, y=3. dx=-1, dy=3. Not straight.
    //
    // Let's fix coordinates for a valid zig-zag.
    // 1. (0,0) -> (1,1) -> (2,2) [SE]
    // 2. (2,2) -> (3,2) -> (4,2) [E]
    // 3. (4,2) -> (4,3) -> (4,4) [S]
    //
    // P1 at (0,0) H1
    // P2 at (1,1) H1
    // P2 at (3,2) H1
    // P2 at (4,3) H1

    const engine = createEngine('square8');
    const startPos = { x: 0, y: 0 };
    const target1 = { x: 1, y: 1 };
    const target2 = { x: 3, y: 2 };
    const target3 = { x: 4, y: 3 };
 
    setupBoard(engine, [
      { pos: startPos, player: 1, height: 1 },
      { pos: target1, player: 2, height: 1 },
      { pos: target2, player: 2, height: 1 },
      { pos: target3, player: 2, height: 1 },
    ]);
 
    // Start the zig-zag chain with a single overtaking_capture:
    // (0,0) -> (1,1) -> (2,2).
    const step1 = await engine.makeMove({
      player: 1,
      type: 'overtaking_capture',
      from: startPos,
      captureTarget: target1,
      to: { x: 2, y: 2 },
    } as any);
    expect(step1.success).toBe(true);

    // Debug: directly probe the shared chain enumerator from the chain
    // position for the zig-zag scenario.
    const engineAnyZig: any = engine;
    const stateAfterZig: GameState = engineAnyZig.gameState as GameState;
    const boardManagerZig: BoardManager = engineAnyZig.boardManager as BoardManager;
    const ruleEngineZig: RuleEngine = engineAnyZig.ruleEngine as RuleEngine;
    const chainStateZig = engineAnyZig.chainCaptureState as
      | { currentPosition: Position }
      | undefined;

    if (chainStateZig) {
      const followUpsZig = getCaptureOptionsFromPositionShared(
        chainStateZig.currentPosition,
        1,
        stateAfterZig,
        {
          boardManager: boardManagerZig,
          ruleEngine: ruleEngineZig,
        }
      );

      // eslint-disable-next-line no-console
      console.log('FAQ zigzag followUps', followUpsZig);

      expect(followUpsZig.length).toBeGreaterThan(0);

      const hasExpectedZig = followUpsZig.some(
        (m) =>
          m.player === 1 &&
          m.from &&
          m.captureTarget &&
          m.to &&
          m.from.x === 2 &&
          m.from.y === 2 &&
          m.captureTarget.x === 3 &&
          m.captureTarget.y === 2 &&
          m.to.x === 4 &&
          m.to.y === 2
      );
      expect(hasExpectedZig).toBe(true);
    }
 
    // Resolve any mandatory continuation segments via the chain_capture phase,
    // allowing the engine to choose any legal zig-zag continuation.
    await resolveChainIfPresent(engine);
 
    const engineAny: any = engine;
    const board = engineAny.gameState.board;
    const stacks = board.stacks as Map<string, RingStack>;
    const allStacks: RingStack[] = Array.from(stacks.values());
 
    const blueStacks = allStacks.filter((s) => s.controllingPlayer === 1);
    const redStacks = allStacks.filter((s) => s.controllingPlayer === 2);
 
    // One Blue-controlled stack of height 4, no remaining Red stacks, regardless
    // of the exact intermediate landings chosen by the engine.
    expect(blueStacks.length).toBe(1);
    expect(blueStacks[0].stackHeight).toBe(4);
    expect(blueStacks[0].controllingPlayer).toBe(1);
    expect(redStacks.length).toBe(0);
  });
});
