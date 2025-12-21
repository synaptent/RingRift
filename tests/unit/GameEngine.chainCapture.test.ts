import { GameEngine } from '../../src/server/game/GameEngine';
import { Position, Player, BoardType, TimeControl, RingStack } from '../../src/shared/types/game';

/**
 * Basic behavioural tests for the chain-capture enforcement layer in GameEngine.
 *
 * These tests do NOT attempt to validate full board legality or RuleEngine
 * integration. Instead, they focus narrowly on the new TsChainCaptureState
 * gate that prevents players from:
 *   - moving a different player's piece while a chain is in progress
 *   - playing any non-overtaking or wrong-origin move during an active chain
 *
 * Note: Some tests in this suite manipulate internal engine state
 * (chainCaptureState, gameState.currentPhase) directly, which is incompatible
 * with the orchestrator adapter. These tests are skipped when
 * ORCHESTRATOR_ADAPTER_ENABLED=true.
 */

const orchestratorEnabled = process.env.ORCHESTRATOR_ADAPTER_ENABLED === 'true';

describe('GameEngine chain capture enforcement (TsChainCaptureState)', () => {
  const boardType: BoardType = 'square8';
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

  function createEngine(): GameEngine {
    // GameEngine constructor reassigns playerNumber and timeRemaining, so
    // passing these base players is sufficient for our purposes here.
    return new GameEngine('test-game-chain', boardType, basePlayers, timeControl, false);
  }

  /**
   * Drive any active chain_capture phase to completion by repeatedly applying
   * continue_capture_segment moves from GameEngine.getValidMoves.
   *
   * NOTE: We must re-fetch engineAny.gameState on each iteration because
   * GameEngine.appendHistoryEntry() reassigns this.gameState to a new object.
   * Caching the reference once at the start would cause us to read stale phase
   * values and incorrectly skip chain continuation.
   */
  async function resolveChainIfPresent(engine: GameEngine): Promise<void> {
    const engineAny: any = engine;

    // Re-fetch gameState reference - it may be reassigned by appendHistoryEntry
    const getGameState = () => engineAny.gameState as any;

    if (getGameState().currentPhase !== 'chain_capture') {
      return;
    }

    const MAX_STEPS = 16;
    let steps = 0;

    while (getGameState().currentPhase === 'chain_capture') {
      steps++;
      if (steps > MAX_STEPS) {
        throw new Error('resolveChainIfPresent: exceeded maximum chain-capture steps');
      }

      const currentPlayer = getGameState().currentPlayer;
      const moves = engine.getValidMoves(currentPlayer);

      // Debug: log the actual moves returned
      console.log('[resolveChainIfPresent] moves returned:', {
        count: moves.length,
        types: moves.map((m: any) => m.type),
        phase: getGameState().currentPhase,
        currentPlayer,
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

  // This test manipulates internal chainCaptureState directly, which the orchestrator ignores
  (orchestratorEnabled ? test.skip : test)(
    'rejects moves from a different player while a chain capture is in progress',
    async () => {
      const engine = createEngine();

      const chainStart: Position = { x: 3, y: 3 };
      const chainCurrent: Position = { x: 5, y: 5 };

      // Force an internal chain state as if player 1 had started a capture.
      (engine as any).chainCaptureState = {
        playerNumber: 1,
        startPosition: chainStart,
        currentPosition: chainCurrent,
        segments: [],
        availableMoves: [],
        visitedPositions: new Set<string>(['3,3']),
      };

      const result = await engine.makeMove({
        // Wrong player attempts to move while chain is active
        player: 2,
        type: 'move_stack',
        from: chainCurrent,
        to: { x: 6, y: 6 },
      } as any);

      expect(result.success).toBe(false);
      expect(result.error).toBe('Chain capture in progress: only the capturing player may move');
    }
  );

  // This test manipulates internal chainCaptureState directly, which the orchestrator ignores
  (orchestratorEnabled ? test.skip : test)(
    'rejects non-overtaking or wrong-origin moves from the capturing player during an active chain',
    async () => {
      const engine = createEngine();

      const chainStart: Position = { x: 3, y: 3 };
      const chainCurrent: Position = { x: 5, y: 5 };

      (engine as any).chainCaptureState = {
        playerNumber: 1,
        startPosition: chainStart,
        currentPosition: chainCurrent,
        segments: [],
        availableMoves: [],
        visitedPositions: new Set<string>(['3,3', '5,5']),
      };

      // Case 1: correct player but wrong move type
      const wrongType = await engine.makeMove({
        player: 1,
        type: 'move_stack',
        from: chainCurrent,
        to: { x: 6, y: 6 },
      } as any);

      expect(wrongType.success).toBe(false);
      expect(wrongType.error).toBe(
        'Chain capture in progress: must continue capturing with the same stack'
      );

      // Case 2: correct player and type, but from a different origin than currentPosition
      const wrongOrigin = await engine.makeMove({
        player: 1,
        type: 'overtaking_capture',
        from: { x: 4, y: 4 },
        captureTarget: { x: 6, y: 6 },
        to: { x: 7, y: 7 },
      } as any);

      expect(wrongOrigin.success).toBe(false);
      expect(wrongOrigin.error).toBe(
        'Chain capture in progress: must continue capturing with the same stack'
      );
    }
  );

  test('performs a full two-step chain capture end-to-end (ported from Rust)', async () => {
    // This scenario mirrors the Rust test_chain_capture setup on an 8x8 board:
    // Red stack at (2,2) height 2, Blue at (2,3) height 1, Green at (2,5) height 1.
    // Red jumps over Blue to land at (2,4), then is forced to continue and
    // jumps over Green to land at (2,7), capturing both Blue and Green.
    const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

    const players: Player[] = [
      {
        id: 'red',
        username: 'Red',
        type: 'human',
        playerNumber: 1,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      {
        id: 'blue',
        username: 'Blue',
        type: 'human',
        playerNumber: 2,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      {
        id: 'green',
        username: 'Green',
        type: 'human',
        playerNumber: 3,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ];

    const engine = new GameEngine('chain-e2e', 'square8', players, timeControl, false);
    const engineAny: any = engine;
    const boardManager = engineAny.boardManager as any;
    const gameState = engineAny.gameState as any;

    // Set current phase/player so that capture validation passes
    gameState.currentPhase = 'capture';
    gameState.currentPlayer = 1;

    // Helper to build a stack for a given player and height
    const makeStack = (playerNumber: number, height: number, position: Position) => {
      const rings = Array(height).fill(playerNumber);
      const stack: RingStack = {
        position,
        rings,
        stackHeight: rings.length,
        capHeight: rings.length,
        controllingPlayer: playerNumber,
      };
      boardManager.setStack(position, stack, gameState.board);
    };

    const redPos: Position = { x: 2, y: 2 };
    const bluePos: Position = { x: 2, y: 3 };
    const greenPos: Position = { x: 2, y: 5 };

    makeStack(1, 2, redPos); // Red height 2 at (2,2)
    makeStack(2, 1, bluePos); // Blue height 1 at (2,3)
    makeStack(3, 1, greenPos); // Green height 1 at (2,5)

    const result = await engine.makeMove({
      player: 1,
      type: 'overtaking_capture',
      from: redPos,
      captureTarget: bluePos,
      to: { x: 2, y: 4 },
    } as any);

    expect(result.success).toBe(true);

    // Resolve the mandatory continuation segment(s) via the explicit
    // chain_capture phase so that the full two-step chain is applied.
    await resolveChainIfPresent(engine);

    // After the chain resolves, the original red stack and both targets
    // should be gone, and the capturing stack should be at (2,7) with height 4.
    const board = gameState.board;
    const stackAtRed = board.stacks.get('2,2');
    const stackAtBlue = board.stacks.get('2,3');
    const stackAtGreen = board.stacks.get('2,5');
    const stackAtFinal = board.stacks.get('2,7');

    expect(stackAtRed).toBeUndefined();
    expect(stackAtBlue).toBeUndefined();
    expect(stackAtGreen).toBeUndefined();
    expect(stackAtFinal).toBeDefined();
    expect(stackAtFinal!.stackHeight).toBe(4);
    expect(stackAtFinal!.controllingPlayer).toBe(1);

    // Internal chain state should be cleared once no further captures exist.
    expect(engineAny.chainCaptureState).toBeUndefined();
  });

  test('Q15_3_1_180_degree_reversal_pattern_backend', async () => {
    // This scenario mirrors the Rust integration test
    // `RingRift Rust/ringrift/tests/chain_capture_tests.rs::test_chain_capture_180_reversal`.
    // Setup (SquareSmall / square8):
    // - Red at (2,2) height 2
    // - Blue at (2,3) height 2
    // - Empty at (2,1) and (2,4)
    // Expected behaviour:
    // 1. Red captures Blue by jumping from (2,2) over (2,3) to land at (2,4).
    //    Attacker becomes height 3 (R,R,B).
    // 2. From (2,4) the only valid follow-up capture is a 180° reversal back
    //    over (2,3), landing at (2,1). Final stack at (2,1) has height 4
    //    (R,R,B,B), original positions are empty, and the chain is complete.

    const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

    const players: Player[] = [
      {
        id: 'red',
        username: 'Red',
        type: 'human',
        playerNumber: 1,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      {
        id: 'blue',
        username: 'Blue',
        type: 'human',
        playerNumber: 2,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ];

    const engine = new GameEngine('chain-180', 'square8', players, timeControl, false);
    const engineAny: any = engine;
    const boardManager = engineAny.boardManager as any;
    const gameState = engineAny.gameState as any;

    // Ensure we are in capture phase and it is Red's turn so that
    // RuleEngine.validateMove accepts an overtaking_capture.
    gameState.currentPhase = 'capture';
    gameState.currentPlayer = 1;

    const makeStack = (playerNumber: number, height: number, position: Position) => {
      const rings = Array(height).fill(playerNumber);
      const stack: RingStack = {
        position,
        rings,
        stackHeight: rings.length,
        capHeight: rings.length,
        controllingPlayer: playerNumber,
      };
      boardManager.setStack(position, stack, gameState.board);
    };

    const redPos: Position = { x: 2, y: 2 };
    const bluePos: Position = { x: 2, y: 3 };

    // Red height 2 at (2,2); Blue height 2 at (2,3).
    makeStack(1, 2, redPos);
    makeStack(2, 2, bluePos);

    const result = await engine.makeMove({
      player: 1,
      type: 'overtaking_capture',
      from: redPos,
      captureTarget: bluePos,
      to: { x: 2, y: 4 },
    } as any);

    expect(result.success).toBe(true);

    // Resolve the forced 180° reversal continuation via chain_capture so that
    // the full sequence (to 2,1) is applied.
    await resolveChainIfPresent(engine);

    const board = gameState.board;
    const stackAtInitial = board.stacks.get('2,2');
    const stackAtBlue = board.stacks.get('2,3');
    const stackAtIntermediate = board.stacks.get('2,4');
    const stackAtFinal = board.stacks.get('2,1');

    // Original positions and intermediate landing should be empty after the
    // chain completes.
    expect(stackAtInitial).toBeUndefined();
    expect(stackAtBlue).toBeUndefined();
    expect(stackAtIntermediate).toBeUndefined();

    // Final stack at (2,1) should contain all rings from the sequence: R2 + B2.
    expect(stackAtFinal).toBeDefined();
    expect(stackAtFinal!.stackHeight).toBe(4);
    expect(stackAtFinal!.controllingPlayer).toBe(1);

    // Chain state should be cleared once no further captures exist.
    expect(engineAny.chainCaptureState).toBeUndefined();
  });

  test('respects marker interaction and landing rules during chain capture (mirrors Rust test_chain_capture_landing_rule_with_markers)', async () => {
    // Mirrors `RingRift Rust/ringrift/tests/chain_capture_tests.rs::test_chain_capture_landing_rule_with_markers`.
    // Setup (square8):
    // - Red at (1,1) h1
    // - Blue at (1,2) h1 (capture target)
    // - Green marker at (1,3)
    // - Empty at (1,4)
    // Expected behaviour:
    // Red jumps over Blue at (1,2) and lands on the first empty space after
    // the marker at (1,3), i.e. at (1,4). The marker is processed along the
    // path (flipped/collapsed per marker rules), but does not block landing.

    const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

    const players: Player[] = [
      {
        id: 'red',
        username: 'Red',
        type: 'human',
        playerNumber: 1,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      {
        id: 'blue',
        username: 'Blue',
        type: 'human',
        playerNumber: 2,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      // Optional third player index to mirror Rust's use of a distinct marker color
      {
        id: 'green',
        username: 'Green',
        type: 'human',
        playerNumber: 3,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ];

    const engine = new GameEngine('chain-markers', 'square8', players, timeControl, false);
    const engineAny: any = engine;
    const boardManager = engineAny.boardManager as any;
    const gameState = engineAny.gameState as any;

    // Ensure capture phase & correct player so RuleEngine allows the capture
    gameState.currentPhase = 'capture';
    gameState.currentPlayer = 1;

    const makeStack = (playerNumber: number, height: number, position: Position) => {
      const rings = Array(height).fill(playerNumber);
      const stack: RingStack = {
        position,
        rings,
        stackHeight: rings.length,
        capHeight: rings.length,
        controllingPlayer: playerNumber,
      };
      boardManager.setStack(position, stack, gameState.board);
    };

    const redPos: Position = { x: 1, y: 1 };
    const bluePos: Position = { x: 1, y: 2 };
    const markerPos: Position = { x: 1, y: 3 };
    const landingPos: Position = { x: 1, y: 4 };

    // Red H1 at (1,1), Blue H1 at (1,2)
    makeStack(1, 1, redPos);
    makeStack(2, 1, bluePos);

    // Place a Green marker at (1,3) to test marker interaction along the path
    boardManager.setMarker(markerPos, 3, gameState.board);

    const result = await engine.makeMove({
      player: 1,
      type: 'overtaking_capture',
      from: redPos,
      captureTarget: bluePos,
      to: landingPos,
    } as any);

    expect(result.success).toBe(true);

    const board = gameState.board;
    const stackAtStart = board.stacks.get('1,1');
    const stackAtTarget = board.stacks.get('1,2');
    const stackAtLanding = board.stacks.get('1,4');

    // Original positions should be empty; attacker ends at (1,4)
    expect(stackAtStart).toBeUndefined();
    expect(stackAtTarget).toBeUndefined();
    expect(stackAtLanding).toBeDefined();
    expect(stackAtLanding!.stackHeight).toBe(2); // Red + Blue
    expect(stackAtLanding!.controllingPlayer).toBe(1);

    // Marker at (1,3) should have been processed along the path.
    // According to TS engine rules, jumping over an opponent marker flips
    // it to the mover's color (regular marker) rather than blocking landing.
    const markerKey = '1,3';
    const marker = board.markers.get(markerKey);
    expect(marker).toBeDefined();
    expect(marker!.player).toBe(1);

    // Chain should be complete after this single capture with no further options.
    expect(engineAny.chainCaptureState).toBeUndefined();
  });

  test('terminates chain when landing for next target is blocked (mirrors Rust test_chain_capture_termination_blocked_landing)', async () => {
    // Mirrors `RingRift Rust/ringrift/tests/chain_capture_tests.rs::test_chain_capture_termination_blocked_landing`.
    // Setup (square8):
    // - Red(2,2) h1 (attacker)
    // - Blue(2,3) h1 (first target)
    // - Green(2,5) h1 (would-be second target)
    // - Red(2,6) h1 (blocker on landing for Green)
    // Expected behaviour:
    // Red captures Blue and lands at (2,4). From there, Green at (2,5) is a
    // geometric target but its landing space (2,6) is occupied by a stack, so
    // no valid follow-up captures exist and the chain terminates automatically.

    const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

    const players: Player[] = [
      {
        id: 'red',
        username: 'Red',
        type: 'human',
        playerNumber: 1,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      {
        id: 'blue',
        username: 'Blue',
        type: 'human',
        playerNumber: 2,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      {
        id: 'green',
        username: 'Green',
        type: 'human',
        playerNumber: 3,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ];

    const engine = new GameEngine(
      'chain-termination-blocked',
      'square8',
      players,
      timeControl,
      false
    );
    const engineAny: any = engine;
    const boardManager = engineAny.boardManager as any;
    const gameState = engineAny.gameState as any;

    // Ensure capture phase & correct player so RuleEngine allows the capture
    gameState.currentPhase = 'capture';
    gameState.currentPlayer = 1;

    const makeStack = (playerNumber: number, height: number, position: Position) => {
      const rings = Array(height).fill(playerNumber);
      const stack: RingStack = {
        position,
        rings,
        stackHeight: rings.length,
        capHeight: rings.length,
        controllingPlayer: playerNumber,
      };
      boardManager.setStack(position, stack, gameState.board);
    };

    const redPos: Position = { x: 2, y: 2 };
    const bluePos: Position = { x: 2, y: 3 };
    const greenPos: Position = { x: 2, y: 5 };
    const blockerPos: Position = { x: 2, y: 6 };

    // Initial stacks: attacker, first target, potential second target, blocker
    makeStack(1, 1, redPos);
    makeStack(2, 1, bluePos);
    makeStack(3, 1, greenPos);
    makeStack(1, 1, blockerPos);

    const result = await engine.makeMove({
      player: 1,
      type: 'overtaking_capture',
      from: redPos,
      captureTarget: bluePos,
      to: { x: 2, y: 4 },
    } as any);

    expect(result.success).toBe(true);

    const board = gameState.board;
    const stackAtRed = board.stacks.get('2,2');
    const stackAtBlue = board.stacks.get('2,3');
    const stackAtLanding = board.stacks.get('2,4');
    const stackAtGreen = board.stacks.get('2,5');
    const stackAtBlocker = board.stacks.get('2,6');

    // Red and Blue should be gone after the first capture
    expect(stackAtRed).toBeUndefined();
    expect(stackAtBlue).toBeUndefined();

    // Attacker should now be at (2,4) with both rings (Red + Blue)
    expect(stackAtLanding).toBeDefined();
    expect(stackAtLanding!.stackHeight).toBe(2);
    expect(stackAtLanding!.controllingPlayer).toBe(1);

    // Green and the blocking Red stack should remain unchanged
    expect(stackAtGreen).toBeDefined();
    expect(stackAtGreen!.stackHeight).toBe(1);
    expect(stackAtGreen!.controllingPlayer).toBe(3);

    expect(stackAtBlocker).toBeDefined();
    expect(stackAtBlocker!.stackHeight).toBe(1);
    expect(stackAtBlocker!.controllingPlayer).toBe(1);

    // Because the only geometric target (Green at (2,5)) has no legal landing
    // beyond it, the chain must terminate automatically.
    expect(engineAny.chainCaptureState).toBeUndefined();
  });

  test('terminates chain when all potential targets have higher cap height (mirrors Rust test_chain_capture_termination_no_valid_targets)', async () => {
    // Mirrors `RingRift Rust/ringrift/tests/chain_capture_tests.rs::test_chain_capture_termination_no_valid_targets`.
    // Setup (square8):
    // - Red(2,2) h1 (attacker)
    // - Blue(2,3) h1 (first target)
    // - Green(3,4) h2 (potential target with higher cap)
    // Expected behaviour:
    // Red captures Blue and lands at (2,4). From there, Green at (3,4) is a
    // geometric target, but its cap height (2) is greater than the attacker's
    // cap (1), so no valid follow-up captures exist and the chain terminates.

    const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

    const players: Player[] = [
      {
        id: 'red',
        username: 'Red',
        type: 'human',
        playerNumber: 1,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      {
        id: 'blue',
        username: 'Blue',
        type: 'human',
        playerNumber: 2,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      {
        id: 'green',
        username: 'Green',
        type: 'human',
        playerNumber: 3,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ];

    const engine = new GameEngine('chain-termination-cap', 'square8', players, timeControl, false);
    const engineAny: any = engine;
    const boardManager = engineAny.boardManager as any;
    const gameState = engineAny.gameState as any;

    // Ensure capture phase & correct player so RuleEngine allows the capture
    gameState.currentPhase = 'capture';
    gameState.currentPlayer = 1;

    const makeStack = (playerNumber: number, height: number, position: Position) => {
      const rings = Array(height).fill(playerNumber);
      const stack: RingStack = {
        position,
        rings,
        stackHeight: rings.length,
        capHeight: rings.length,
        controllingPlayer: playerNumber,
      };
      boardManager.setStack(position, stack, gameState.board);
    };

    const redPos: Position = { x: 2, y: 2 };
    const bluePos: Position = { x: 2, y: 3 };
    const greenPos: Position = { x: 3, y: 4 };

    // Initial stacks: attacker (H1), first target (H1), potential second target (H2)
    makeStack(1, 1, redPos);
    makeStack(2, 1, bluePos);
    makeStack(3, 2, greenPos);

    const result = await engine.makeMove({
      player: 1,
      type: 'overtaking_capture',
      from: redPos,
      captureTarget: bluePos,
      to: { x: 2, y: 4 },
    } as any);

    expect(result.success).toBe(true);

    const board = gameState.board;
    const stackAtRed = board.stacks.get('2,2');
    const stackAtBlue = board.stacks.get('2,3');
    const stackAtLanding = board.stacks.get('2,4');
    const stackAtGreen = board.stacks.get('3,4');

    // Red and Blue should be gone after the first capture
    expect(stackAtRed).toBeUndefined();
    expect(stackAtBlue).toBeUndefined();

    // Attacker should now be at (2,4) with both rings (Red + Blue)
    expect(stackAtLanding).toBeDefined();
    expect(stackAtLanding!.stackHeight).toBe(2);
    expect(stackAtLanding!.controllingPlayer).toBe(1);

    // Green stack with higher cap should remain unchanged
    expect(stackAtGreen).toBeDefined();
    expect(stackAtGreen!.stackHeight).toBe(2);
    expect(stackAtGreen!.controllingPlayer).toBe(3);

    // Because all geometric targets have higher cap height than the attacker,
    // no valid follow-up captures exist, and the chain must terminate.
    expect(engineAny.chainCaptureState).toBeUndefined();
  });

  // This test spies on processAutomaticConsequences which isn't used in orchestrator mode
  (orchestratorEnabled ? test.skip : test)(
    'runs post-movement processing once after an engine-driven chain capture',
    async () => {
      // Use the same two-step chain scenario as the "full two-step" test, but
      // focus on verifying that processAutomaticConsequences is invoked once
      // after the chain capture has fully resolved. This ensures the
      // capture → processAutomaticConsequences wiring remains intact.

      const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

      const players: Player[] = [
        {
          id: 'red',
          username: 'Red',
          type: 'human',
          playerNumber: 1,
          isReady: true,
          timeRemaining: timeControl.initialTime * 1000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'blue',
          username: 'Blue',
          type: 'human',
          playerNumber: 2,
          isReady: true,
          timeRemaining: timeControl.initialTime * 1000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'green',
          username: 'Green',
          type: 'human',
          playerNumber: 3,
          isReady: true,
          timeRemaining: timeControl.initialTime * 1000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ];

      const engine = new GameEngine(
        'chain-post-processing',
        'square8',
        players,
        timeControl,
        false
      );
      const engineAny: any = engine;
      const boardManager = engineAny.boardManager as any;
      const gameState = engineAny.gameState as any;

      // Ensure capture phase & correct player so RuleEngine allows capture.
      gameState.currentPhase = 'capture';
      gameState.currentPlayer = 1;

      const makeStack = (playerNumber: number, height: number, position: Position) => {
        const rings = Array(height).fill(playerNumber);
        const stack: RingStack = {
          position,
          rings,
          stackHeight: rings.length,
          capHeight: rings.length,
          controllingPlayer: playerNumber,
        };
        boardManager.setStack(position, stack, gameState.board);
      };

      const redPos: Position = { x: 2, y: 2 };
      const bluePos: Position = { x: 2, y: 3 };
      const greenPos: Position = { x: 2, y: 5 };

      // Same configuration as the "full two-step chain" test.
      makeStack(1, 2, redPos);
      makeStack(2, 1, bluePos);
      makeStack(3, 1, greenPos);

      // Spy on processAutomaticConsequences to ensure it is invoked exactly once
      // after the engine-driven chain is complete.
      const processSpy = jest
        .spyOn(engineAny, 'processAutomaticConsequences')
        .mockResolvedValue(undefined);

      const result = await engine.makeMove({
        player: 1,
        type: 'overtaking_capture',
        from: redPos,
        captureTarget: bluePos,
        to: { x: 2, y: 4 },
      } as any);

      expect(result.success).toBe(true);

      // Resolve the remaining chain-capture segment(s). The engine should only
      // invoke processAutomaticConsequences once, after the final segment for
      // this turn has been applied.
      await resolveChainIfPresent(engine);

      // processAutomaticConsequences should have been called exactly once for
      // this turn, after chain-capture resolution concluded.
      expect(processSpy).toHaveBeenCalledTimes(1);

      // For sanity, the final chain state and board geometry should mirror the
      // expectations from the earlier full two-step chain test.
      const board = gameState.board;
      const stackAtRed = board.stacks.get('2,2');
      const stackAtBlue = board.stacks.get('2,3');
      const stackAtGreen = board.stacks.get('2,5');
      const stackAtFinal = board.stacks.get('2,7');

      expect(stackAtRed).toBeUndefined();
      expect(stackAtBlue).toBeUndefined();
      expect(stackAtGreen).toBeUndefined();
      expect(stackAtFinal).toBeDefined();
      expect(stackAtFinal!.stackHeight).toBe(4);
      expect(stackAtFinal!.controllingPlayer).toBe(1);

      expect(engineAny.chainCaptureState).toBeUndefined();
    }
  );
});
