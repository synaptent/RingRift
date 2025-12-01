/**
 * Chain Capture Extended Scenarios - P19B.2-2
 *
 * Tests for extended chain capture scenarios with 4+ targets.
 * These test the sequential decision-making during chain captures
 * and verify all targets are captured correctly.
 *
 * Rule Reference: Section 10.3 (Chain Overtaking)
 * - RR-CANON-R084: Chain captures (consecutive captures from same stack)
 * - RR-CANON-R085: Chain capture must extend from previous capture position
 */

import { GameEngine } from '../../src/server/game/GameEngine';
import {
  Position,
  Player,
  TimeControl,
  RingStack,
  GameState,
  positionToString,
} from '../../src/shared/types/game';
import { getChainCaptureContinuationInfo } from '../../src/shared/engine/aggregates/CaptureAggregate';
import {
  createChainCapture3Fixture,
  createChainCapture4Fixture,
  createChainCapture5PlusFixture,
  createChainCaptureZigzagFixture,
  createChainCaptureEdgeTerminationFixture,
  ChainCaptureExtendedFixture,
} from '../fixtures/chainCaptureExtendedFixture';

describe('Scenario: Extended chain capture with 4+ targets (P19B.2-2)', () => {
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

  function createEngine(): GameEngine {
    return new GameEngine('scenario-chain-extended', 'square8', basePlayers, timeControl, false);
  }

  // Helper to set up board state from fixture
  function setupBoardFromFixture(engine: GameEngine, fixture: ChainCaptureExtendedFixture) {
    const engineAny: any = engine;
    const boardManager = engineAny.boardManager;
    const gameState = engineAny.gameState;

    // Clear existing stacks
    gameState.board.stacks.clear();

    // Set up stacks from fixture
    for (const [, stack] of fixture.gameState.board.stacks) {
      boardManager.setStack(stack.position, stack, gameState.board);
    }

    // Force movement phase
    gameState.currentPhase = 'movement';
    gameState.currentPlayer = 1;
  }

  /**
   * Resolve any active capture chain for the current player by repeatedly
   * applying continue_capture_segment moves from GameEngine.getValidMoves
   * while the game remains in the 'chain_capture' phase.
   */
  async function resolveChainIfPresent(engine: GameEngine): Promise<number> {
    const engineAny: any = engine;

    const MAX_STEPS = 16;
    let steps = 0;
    let captureCount = 0;

    // No-op when no chain is currently active.
    if ((engineAny.gameState as GameState).currentPhase !== 'chain_capture') {
      return captureCount;
    }

    while ((engineAny.gameState as GameState).currentPhase === 'chain_capture') {
      steps++;
      if (steps > MAX_STEPS) {
        throw new Error('resolveChainIfPresent: exceeded maximum chain-capture steps');
      }

      const state = engineAny.gameState as GameState;
      const currentPlayer = state.currentPlayer;
      const moves = engine.getValidMoves(currentPlayer);

      const chainMoves = moves.filter((m: any) => m.type === 'continue_capture_segment');

      if (chainMoves.length === 0) {
        // No more chain moves available, but still in chain_capture phase?
        // This shouldn't happen, but let's break to avoid infinite loop
        break;
      }

      const next = chainMoves[0];

      const result = await engine.makeMove({
        player: next.player,
        type: 'continue_capture_segment',
        from: next.from,
        captureTarget: next.captureTarget,
        to: next.to,
      } as any);

      expect(result.success).toBe(true);
      captureCount++;
    }

    return captureCount;
  }

  test('should complete 4-target chain capture', async () => {
    const fixture = createChainCapture4Fixture();
    const engine = createEngine();

    setupBoardFromFixture(engine, fixture);

    const engineAny: any = engine;

    // Start the chain with the initial overtaking_capture
    const initialMove = fixture.initialMove;
    const step1 = await engine.makeMove({
      player: initialMove.player,
      type: 'overtaking_capture',
      from: initialMove.from,
      captureTarget: initialMove.captureTarget,
      to: initialMove.to,
    } as any);

    expect(step1.success).toBe(true);

    // Resolve all mandatory chain continuations
    const additionalCaptures = await resolveChainIfPresent(engine);

    // Total captures should be 4 (1 initial + 3 continuations)
    const totalCaptures = 1 + additionalCaptures;
    expect(totalCaptures).toBe(fixture.expectedCaptureCount);

    // Verify final board state
    const board = engineAny.gameState.board;
    const stacks = board.stacks as Map<string, RingStack>;
    const allStacks: RingStack[] = Array.from(stacks.values());

    const blueStacks = allStacks.filter((s) => s.controllingPlayer === 1);
    const redStacks = allStacks.filter((s) => s.controllingPlayer === 2);

    // One Blue-controlled stack of height 5, no remaining Red stacks
    expect(blueStacks.length).toBe(1);
    expect(blueStacks[0].stackHeight).toBe(fixture.expectedFinalHeight);
    expect(blueStacks[0].controllingPlayer).toBe(1);
    expect(redStacks.length).toBe(0);

    // Verify final position
    const finalStack = stacks.get(positionToString(fixture.expectedFinalPosition));
    expect(finalStack).toBeDefined();
    expect(finalStack!.controllingPlayer).toBe(1);
  });

  test('should complete 5-target chain capture', async () => {
    const fixture = createChainCapture5PlusFixture();
    const engine = createEngine();

    setupBoardFromFixture(engine, fixture);

    const engineAny: any = engine;

    // Start the chain with the initial overtaking_capture
    const initialMove = fixture.initialMove;
    const step1 = await engine.makeMove({
      player: initialMove.player,
      type: 'overtaking_capture',
      from: initialMove.from,
      captureTarget: initialMove.captureTarget,
      to: initialMove.to,
    } as any);

    expect(step1.success).toBe(true);

    // Resolve all mandatory chain continuations
    const additionalCaptures = await resolveChainIfPresent(engine);

    // Total captures should be 5 (1 initial + 4 continuations)
    const totalCaptures = 1 + additionalCaptures;
    expect(totalCaptures).toBe(fixture.expectedCaptureCount);

    // Verify final board state
    const board = engineAny.gameState.board;
    const stacks = board.stacks as Map<string, RingStack>;
    const allStacks: RingStack[] = Array.from(stacks.values());

    const blueStacks = allStacks.filter((s) => s.controllingPlayer === 1);
    const redStacks = allStacks.filter((s) => s.controllingPlayer === 2);

    // One Blue-controlled stack of height 6, no remaining Red stacks
    expect(blueStacks.length).toBe(1);
    expect(blueStacks[0].stackHeight).toBe(fixture.expectedFinalHeight);
    expect(blueStacks[0].controllingPlayer).toBe(1);
    expect(redStacks.length).toBe(0);

    // Verify final position
    const finalStack = stacks.get(positionToString(fixture.expectedFinalPosition));
    expect(finalStack).toBeDefined();
    expect(finalStack!.controllingPlayer).toBe(1);
  });

  test('should handle chain capture decisions correctly at each step', async () => {
    const fixture = createChainCapture4Fixture();
    const engine = createEngine();

    setupBoardFromFixture(engine, fixture);

    const engineAny: any = engine;

    // Start the chain
    const initialMove = fixture.initialMove;
    await engine.makeMove({
      player: initialMove.player,
      type: 'overtaking_capture',
      from: initialMove.from,
      captureTarget: initialMove.captureTarget,
      to: initialMove.to,
    } as any);

    // Track each step of the chain
    let stepNumber = 1;
    const chainPositions: Position[] = [initialMove.to as Position];

    while ((engineAny.gameState as GameState).currentPhase === 'chain_capture') {
      stepNumber++;
      const state = engineAny.gameState as GameState;
      const currentPlayer = state.currentPlayer;
      const moves = engine.getValidMoves(currentPlayer);

      // Verify we get chain continuation moves
      const chainMoves = moves.filter((m: any) => m.type === 'continue_capture_segment');
      expect(chainMoves.length).toBeGreaterThan(0);

      // All chain moves should originate from the current chain position
      const lastPosition = chainPositions[chainPositions.length - 1];
      for (const move of chainMoves) {
        expect(move.from).toEqual(lastPosition);
      }

      // Execute the first available continuation
      const next = chainMoves[0];
      await engine.makeMove({
        player: next.player,
        type: 'continue_capture_segment',
        from: next.from,
        captureTarget: next.captureTarget,
        to: next.to,
      } as any);

      chainPositions.push(next.to as Position);

      if (stepNumber > 10) break; // Safety limit
    }

    // Verify the chain had 4 captures total
    expect(chainPositions.length).toBe(fixture.expectedCaptureCount);
  });

  test('should stop chain when no more targets available', async () => {
    const fixture = createChainCapture4Fixture();
    const engine = createEngine();

    setupBoardFromFixture(engine, fixture);

    const engineAny: any = engine;

    // Start the chain
    const initialMove = fixture.initialMove;
    await engine.makeMove({
      player: initialMove.player,
      type: 'overtaking_capture',
      from: initialMove.from,
      captureTarget: initialMove.captureTarget,
      to: initialMove.to,
    } as any);

    // Resolve all chain captures
    await resolveChainIfPresent(engine);

    // After chain completes, phase should NOT be chain_capture
    const finalState = engineAny.gameState as GameState;
    expect(finalState.currentPhase).not.toBe('chain_capture');

    // Verify the chain enumerator shows no more valid captures
    const board = finalState.board;
    const stacks = board.stacks as Map<string, RingStack>;
    const blueStacks = Array.from(stacks.values()).filter((s) => s.controllingPlayer === 1);

    expect(blueStacks.length).toBe(1);
    const finalStack = blueStacks[0];

    const continuationInfo = getChainCaptureContinuationInfo(finalState, 1, finalStack.position);

    expect(continuationInfo.mustContinue).toBe(false);
    expect(continuationInfo.availableContinuations.length).toBe(0);
  });

  test('should verify chain sequence matches expected order', async () => {
    const fixture = createChainCapture4Fixture();
    const engine = createEngine();

    setupBoardFromFixture(engine, fixture);

    const engineAny: any = engine;
    const capturedTargets: string[] = [];

    // Start the chain
    const initialMove = fixture.initialMove;
    await engine.makeMove({
      player: initialMove.player,
      type: 'overtaking_capture',
      from: initialMove.from,
      captureTarget: initialMove.captureTarget,
      to: initialMove.to,
    } as any);

    capturedTargets.push(positionToString(initialMove.captureTarget as Position));

    // Continue the chain and track captured targets
    while ((engineAny.gameState as GameState).currentPhase === 'chain_capture') {
      const state = engineAny.gameState as GameState;
      const currentPlayer = state.currentPlayer;
      const moves = engine.getValidMoves(currentPlayer);
      const chainMoves = moves.filter((m: any) => m.type === 'continue_capture_segment');

      if (chainMoves.length === 0) break;

      const next = chainMoves[0];
      capturedTargets.push(positionToString(next.captureTarget as Position));

      await engine.makeMove({
        player: next.player,
        type: 'continue_capture_segment',
        from: next.from,
        captureTarget: next.captureTarget,
        to: next.to,
      } as any);
    }

    // Verify we captured the expected targets
    const expectedTargetStrings = fixture.expectedTargets.map(positionToString);
    expect(capturedTargets.length).toBe(expectedTargetStrings.length);

    // All expected targets should have been captured
    for (const target of expectedTargetStrings) {
      expect(capturedTargets).toContain(target);
    }
  });

  test('should create correct markers during chain capture', async () => {
    const fixture = createChainCapture4Fixture();
    const engine = createEngine();

    setupBoardFromFixture(engine, fixture);

    const engineAny: any = engine;

    // Start the chain
    const initialMove = fixture.initialMove;
    await engine.makeMove({
      player: initialMove.player,
      type: 'overtaking_capture',
      from: initialMove.from,
      captureTarget: initialMove.captureTarget,
      to: initialMove.to,
    } as any);

    // Resolve all chain captures
    await resolveChainIfPresent(engine);

    // Verify markers were created at departure positions
    const board = engineAny.gameState.board;
    const markers = board.markers;

    // There should be markers at each departure position (4 captures = 4 markers)
    // Note: The 5th marker was created at target 4's position by the 4th capture
    expect(markers.size).toBeGreaterThanOrEqual(4);

    // The original attacker position should have a marker
    expect(markers.has(positionToString({ x: 0, y: 0 }))).toBe(true);
    expect(markers.get(positionToString({ x: 0, y: 0 })).player).toBe(1);
  });

  test('should complete 3-target chain capture (simpler case)', async () => {
    const fixture = createChainCapture3Fixture();
    const engine = createEngine();

    setupBoardFromFixture(engine, fixture);

    const engineAny: any = engine;

    // Start the chain with the initial overtaking_capture
    const initialMove = fixture.initialMove;
    const step1 = await engine.makeMove({
      player: initialMove.player,
      type: 'overtaking_capture',
      from: initialMove.from,
      captureTarget: initialMove.captureTarget,
      to: initialMove.to,
    } as any);

    expect(step1.success).toBe(true);

    // Resolve all mandatory chain continuations
    const additionalCaptures = await resolveChainIfPresent(engine);

    // Total captures should be 3 (1 initial + 2 continuations)
    const totalCaptures = 1 + additionalCaptures;
    expect(totalCaptures).toBe(fixture.expectedCaptureCount);

    // Verify final board state
    const board = engineAny.gameState.board;
    const stacks = board.stacks as Map<string, RingStack>;
    const allStacks: RingStack[] = Array.from(stacks.values());

    const blueStacks = allStacks.filter((s) => s.controllingPlayer === 1);
    const redStacks = allStacks.filter((s) => s.controllingPlayer === 2);

    // One Blue-controlled stack of height 4, no remaining Red stacks
    expect(blueStacks.length).toBe(1);
    expect(blueStacks[0].stackHeight).toBe(fixture.expectedFinalHeight);
    expect(redStacks.length).toBe(0);

    // Verify final position
    const finalStack = stacks.get(positionToString(fixture.expectedFinalPosition));
    expect(finalStack).toBeDefined();
    expect(finalStack!.controllingPlayer).toBe(1);
  });

  test('should complete zigzag chain capture with direction changes', async () => {
    const fixture = createChainCaptureZigzagFixture();
    const engine = createEngine();

    setupBoardFromFixture(engine, fixture);

    const engineAny: any = engine;

    // Start the chain with the initial overtaking_capture
    const initialMove = fixture.initialMove;
    const step1 = await engine.makeMove({
      player: initialMove.player,
      type: 'overtaking_capture',
      from: initialMove.from,
      captureTarget: initialMove.captureTarget,
      to: initialMove.to,
    } as any);

    expect(step1.success).toBe(true);

    // Track the directions used during the chain
    const directionsUsed: string[] = [];
    let prevPos = initialMove.to as Position;

    while ((engineAny.gameState as GameState).currentPhase === 'chain_capture') {
      const state = engineAny.gameState as GameState;
      const currentPlayer = state.currentPlayer;
      const moves = engine.getValidMoves(currentPlayer);

      const chainMoves = moves.filter((m: any) => m.type === 'continue_capture_segment');
      if (chainMoves.length === 0) break;

      const next = chainMoves[0];

      // Determine direction of this capture
      const dx = (next.to as Position).x - prevPos.x;
      const dy = (next.to as Position).y - prevPos.y;
      let dir = '';
      if (dx > 0 && dy === 0) dir = 'E';
      else if (dx < 0 && dy === 0) dir = 'W';
      else if (dx === 0 && dy > 0) dir = 'S';
      else if (dx === 0 && dy < 0) dir = 'N';
      else if (dx > 0 && dy > 0) dir = 'SE';
      else if (dx < 0 && dy < 0) dir = 'NW';
      else if (dx > 0 && dy < 0) dir = 'NE';
      else if (dx < 0 && dy > 0) dir = 'SW';
      directionsUsed.push(dir);

      await engine.makeMove({
        player: next.player,
        type: 'continue_capture_segment',
        from: next.from,
        captureTarget: next.captureTarget,
        to: next.to,
      } as any);

      prevPos = next.to as Position;
    }

    // Verify we had direction changes (not all same direction)
    // This tests that chain captures CAN change direction per the rules
    const uniqueDirections = new Set(directionsUsed);
    expect(uniqueDirections.size).toBeGreaterThan(1);

    // Verify all targets were captured (no red stacks remain)
    const board = engineAny.gameState.board;
    const stacks = board.stacks as Map<string, RingStack>;
    const allStacks: RingStack[] = Array.from(stacks.values());
    const redStacks = allStacks.filter((s) => s.controllingPlayer === 2);
    expect(redStacks.length).toBe(0);

    // Verify the chain captured all expected targets
    const blueStacks = allStacks.filter((s) => s.controllingPlayer === 1);
    expect(blueStacks.length).toBe(1);
    expect(blueStacks[0].stackHeight).toBe(fixture.expectedFinalHeight);
  });

  test('should terminate chain at board edge when no valid landing exists', async () => {
    const fixture = createChainCaptureEdgeTerminationFixture();
    const engine = createEngine();

    setupBoardFromFixture(engine, fixture);

    const engineAny: any = engine;

    // Start the chain with the initial overtaking_capture
    const initialMove = fixture.initialMove;
    const step1 = await engine.makeMove({
      player: initialMove.player,
      type: 'overtaking_capture',
      from: initialMove.from,
      captureTarget: initialMove.captureTarget,
      to: initialMove.to,
    } as any);

    expect(step1.success).toBe(true);

    // After the single capture, there should be no chain continuation
    // because from (7,7) there's no valid landing position (board edge)
    const additionalCaptures = await resolveChainIfPresent(engine);

    // Should be exactly 1 capture (the initial one, no continuations)
    const totalCaptures = 1 + additionalCaptures;
    expect(totalCaptures).toBe(fixture.expectedCaptureCount);
    expect(totalCaptures).toBe(1);

    // Verify final board state
    const board = engineAny.gameState.board;
    const stacks = board.stacks as Map<string, RingStack>;

    // Final stack at corner (7,7)
    const finalStack = stacks.get(positionToString(fixture.expectedFinalPosition));
    expect(finalStack).toBeDefined();
    expect(finalStack!.stackHeight).toBe(2);
    expect(finalStack!.controllingPlayer).toBe(1);

    // Phase should have moved past chain_capture
    expect(engineAny.gameState.currentPhase).not.toBe('chain_capture');
  });

  /**
   * NOTE: For generating chain capture contract vectors, use the existing generator:
   *   npx ts-node scripts/generate-extended-contract-vectors.ts
   *
   * This generates:
   * - Family A: Chain capture long-tail vectors (depth-3 linear on square8/19/hex)
   * - Family B: Forced elimination vectors
   * - Family C: Territory + line endgame vectors
   * - Family D: Hex edge case vectors
   * - Family E: Meta-moves (swap_sides)
   *
   * Output goes to: tests/fixtures/contract-vectors/v2/
   */

  test('should verify all markers are placed at departure positions', async () => {
    const fixture = createChainCapture3Fixture();
    const engine = createEngine();

    setupBoardFromFixture(engine, fixture);

    const engineAny: any = engine;
    const departurePositions: string[] = [];

    // Track all departure positions
    departurePositions.push(positionToString(fixture.initialMove.from as Position));

    // Start the chain
    const initialMove = fixture.initialMove;
    await engine.makeMove({
      player: initialMove.player,
      type: 'overtaking_capture',
      from: initialMove.from,
      captureTarget: initialMove.captureTarget,
      to: initialMove.to,
    } as any);

    // Continue and track each departure
    while ((engineAny.gameState as GameState).currentPhase === 'chain_capture') {
      const state = engineAny.gameState as GameState;
      const moves = engine.getValidMoves(state.currentPlayer);
      const chainMoves = moves.filter((m: any) => m.type === 'continue_capture_segment');

      if (chainMoves.length === 0) break;

      const next = chainMoves[0];
      departurePositions.push(positionToString(next.from as Position));

      await engine.makeMove({
        player: next.player,
        type: 'continue_capture_segment',
        from: next.from,
        captureTarget: next.captureTarget,
        to: next.to,
      } as any);
    }

    // Verify all departure positions have markers
    const board = engineAny.gameState.board;
    const markers = board.markers;

    for (const pos of departurePositions) {
      expect(markers.has(pos)).toBe(true);
      expect(markers.get(pos).player).toBe(1);
    }

    // Should have exactly 3 markers (one per capture segment)
    expect(departurePositions.length).toBe(3);
  });
});
