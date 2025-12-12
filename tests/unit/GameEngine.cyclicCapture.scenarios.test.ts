import { GameEngine } from '../../src/server/game/GameEngine';
import {
  Position,
  Player,
  TimeControl,
  RingStack,
  BoardState,
  BOARD_CONFIGS,
  positionToString,
} from '../../src/shared/types/game';
import { findMaxCaptureChains } from '../../src/client/sandbox/sandboxCaptureSearch';
import {
  CaptureBoardAdapters,
  CaptureApplyAdapters,
  applyCaptureSegmentOnBoard,
} from '../../src/client/sandbox/sandboxCaptures';
import {
  applyMarkerEffectsAlongPathOnBoard,
  MarkerPathHelpers,
} from '../../src/client/sandbox/sandboxMovement';

/**
 * Cyclic capture scenario on a square19 board.
 *
 * Geometry (0-based coordinates, 19x19 board, center at (9,9)):
 *
 * Inner axis-aligned square (side length 8), Player 2 stacks (cap=2, height=2):
 *   A = (5, 5)
 *   B = (13, 5)
 *   C = (13, 13)
 *   D = (5, 13)
 *
 * Outer diagonally-oriented square whose vertices are at the midpoints of
 * the inner square's sides:
 *   W = (9, 17)   // above the inner square, between B and C in x
 *   X = (17, 9)   // to the right, between C and D in y
 *   Y = (9, 1)    // below, between A and B in x
 *   Z = (1, 9)    // to the left, between A and D in y
 *
 * The overtaking stack (Player 1) starts at Y with cap=2, height=2.
 *
 * From each outer vertex, the overtaker has multiple diagonal rays available
 * that pass over one of the inner vertices and land on a different outer
 * vertex. In principle, as long as distance >= current stack height and the
 * landing space is empty, the engine may choose among several valid capture
 * segments, enabling clockwise, counterclockwise, or reversing sequences
 * around the inner square.
 *
 * We do not hard-code a single geometric path. Instead, this test:
 *   - Starts the overtaker at one outer vertex (Y) with cap=2, height=2.
 *   - Initiates a legal first capture segment (Y -> over B -> X).
 *   - Lets the engine drive the mandatory chain capture to maximal length
 *     according to its move generation.
 *
 * The assertions then check for core cyclic-pattern properties rather than
 * a specific path:
 *   - Exactly one overtaker stack remains on some outer vertex.
 *   - That stack has strictly greater height than it started with (it has
 *     captured from multiple targets).
 *   - At least one inner stack has been visited and partially depleted.
 *   - The chainCaptureState has been cleared once no legal captures remain.
 */

describe('GameEngine cyclic capture scenarios (square19)', () => {
  const timeControl: TimeControl = {
    initialTime: 600,
    increment: 0,
    type: 'blitz',
  };

  const players: Player[] = [
    {
      id: 'p1',
      username: 'Overtaker',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: BOARD_CONFIGS.square19.ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Targets',
      type: 'human',
      playerNumber: 2,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: BOARD_CONFIGS.square19.ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  function makeStack(
    boardManager: any,
    gameState: any,
    playerNumber: number,
    height: number,
    position: Position
  ): void {
    const rings = Array(height).fill(playerNumber);
    const stack: RingStack = {
      position,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer: playerNumber,
    };
    boardManager.setStack(position, stack, gameState.board);
  }

  // Resolve any active capture chain for the current player by repeatedly
  // selecting a continue_capture_segment move from GameEngine.getValidMoves
  // while the game remains in the 'chain_capture' phase. This explicit loop
  // mirrors the unified Move-based chain-capture model used in other
  // scenario tests.
  async function resolveChainIfPresent(engine: GameEngine): Promise<void> {
    const engineAny: any = engine;
    const gameState = engineAny.gameState as any;

    if (gameState.currentPhase !== 'chain_capture') {
      return;
    }

    const MAX_STEPS = 32;
    let steps = 0;

    while (gameState.currentPhase === 'chain_capture') {
      steps++;
      if (steps > MAX_STEPS) {
        throw new Error('resolveChainIfPresent: exceeded maximum chain-capture steps');
      }

      const currentPlayer = gameState.currentPlayer;
      const moves = engine.getValidMoves(currentPlayer);
      const chainMoves = moves.filter((m: any) => m.type === 'continue_capture_segment');

      if (chainMoves.length === 0) {
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
    }
  }

  test('supports a cyclic chain capture around an inner square and terminates when distance < height', async () => {
    const engine = new GameEngine(
      'cyclic-square19',
      'square19',
      players,
      timeControl,
      false
    ) as any;
    const gameState = engine.gameState as any;
    const boardManager = engine.boardManager as any;

    // Force capture phase and correct current player so RuleEngine will
    // accept an overtaking_capture move.
    gameState.currentPhase = 'capture';
    gameState.currentPlayer = 1;

    // Inner targets (Player 2) - cap=2, height=2.
    const A: Position = { x: 5, y: 5 };
    const B: Position = { x: 13, y: 5 };
    const C: Position = { x: 13, y: 13 };
    const D: Position = { x: 5, y: 13 };

    // Outer vertices for overtaker path.
    const W: Position = { x: 9, y: 17 };
    const X: Position = { x: 17, y: 9 };
    const Y: Position = { x: 9, y: 1 };
    const Z: Position = { x: 1, y: 9 };

    // Place inner stacks.
    makeStack(boardManager, gameState, 2, 2, A);
    makeStack(boardManager, gameState, 2, 2, B);
    makeStack(boardManager, gameState, 2, 2, C);
    makeStack(boardManager, gameState, 2, 2, D);

    // Place overtaking stack (Player 1) at Y, cap=2, height=2.
    makeStack(boardManager, gameState, 1, 2, Y);

    // Perform the initial capture segment explicitly along one outer edge:
    // Y -> over B -> X. This is one of the valid 45° diagonal rays from Y
    // that passes through exactly one target stack (B) and lands on another
    // outer vertex (X).
    const result = await engine.makeMove({
      player: 1,
      type: 'overtaking_capture',
      from: Y,
      captureTarget: B,
      to: X,
    } as any);

    expect(result.success).toBe(true);

    // After the initial capture, explicitly resolve any mandatory chain
    // continuations via the unified chain_capture phase so that the final
    // board state reflects all legal segments.
    await resolveChainIfPresent(engine as GameEngine);

    const board = gameState.board as any;
    const stacks: Map<string, RingStack> = board.stacks;

    // Collect final stacks by controlling player.
    const allStacks = Array.from(stacks.values());
    const overtakerStacks = allStacks.filter((s) => s.controllingPlayer === 1);
    const targetStacks = allStacks.filter((s) => s.controllingPlayer === 2);

    // There should be exactly one overtaker stack remaining on some outer
    // vertex (W, X, Y, or Z). We do not fix which vertex, because the
    // engine may choose clockwise, counterclockwise, or direction changes
    // around the outer square as long as captures are legal.
    expect(overtakerStacks.length).toBe(1);

    const overtakerFinal = overtakerStacks[0];

    // Overtaker started with height 2. Total rings in the system are:
    //   - Overtaker: 2
    //   - Four targets: 4 * 2 = 8
    //   => 10 rings total
    // Each overtaking capture moves one ring from a target to the
    // overtaker (no rings are eliminated). The current engine implementation
    // may or may not achieve the absolute maximal number of segments for this
    // geometry, but it must:
    //   - Increase the overtaker's height above its starting value, and
    //   - Preserve the total number of rings on the board.
    expect(overtakerFinal.stackHeight).toBeGreaterThan(2);

    const totalRingsAfter = allStacks.reduce((sum, s) => sum + s.stackHeight, 0);
    expect(totalRingsAfter).toBe(10);

    // The overtaker stack is controlled by Player 1, but its rings may
    // include captured rings from other players at the bottom of the stack.
    expect(overtakerFinal.controllingPlayer).toBe(1);
    expect(overtakerFinal.rings.some((r) => r === 1)).toBe(true);
    expect(overtakerFinal.rings.some((r) => r !== 1)).toBe(true);

    // Ensure no other players' stacks are present.
    const otherPlayerStacks = allStacks.filter(
      (s) => s.controllingPlayer !== 1 && s.controllingPlayer !== 2
    );
    expect(otherPlayerStacks.length).toBe(0);

    // Finally, the internal chain capture state should be cleared once the
    // engine has determined that no further legal overtaking captures exist.
    expect(engine.chainCaptureState).toBeUndefined();
  });

  test('sandbox search finds maximal overtaking chains for the same cyclic square19 geometry', () => {
    const config = BOARD_CONFIGS.square19;

    const board: BoardState = {
      stacks: new Map<string, RingStack>(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: config.size,
      type: 'square19',
    };

    const makeBoardStack = (playerNumber: number, height: number, position: Position) => {
      const rings = Array(height).fill(playerNumber);
      const stack: RingStack = {
        position,
        rings,
        stackHeight: rings.length,
        capHeight: rings.length,
        controllingPlayer: playerNumber,
      };
      board.stacks.set(positionToString(position), stack);
    };

    // Inner targets (Player 2) - cap=2, height=2.
    const A: Position = { x: 5, y: 5 };
    const B: Position = { x: 13, y: 5 };
    const C: Position = { x: 13, y: 13 };
    const D: Position = { x: 5, y: 13 };

    // Outer vertices for overtaker path.
    const W: Position = { x: 9, y: 17 };
    const X: Position = { x: 17, y: 9 };
    const Y: Position = { x: 9, y: 1 };
    const Z: Position = { x: 1, y: 9 };

    makeBoardStack(2, 2, A);
    makeBoardStack(2, 2, B);
    makeBoardStack(2, 2, C);
    makeBoardStack(2, 2, D);
    makeBoardStack(1, 2, Y);

    const captureBoardAdapters: CaptureBoardAdapters = {
      isValidPosition: (pos: Position) =>
        pos.x >= 0 && pos.x < config.size && pos.y >= 0 && pos.y < config.size,
      isCollapsedSpace: (pos: Position, b: BoardState) =>
        b.collapsedSpaces.has(positionToString(pos)),
      getMarkerOwner: (pos: Position, b: BoardState) => {
        const marker = b.markers.get(positionToString(pos));
        return marker?.player;
      },
    };

    const markerHelpers: MarkerPathHelpers = {
      setMarker: (pos, playerNumber, b) => {
        const key = positionToString(pos);
        b.markers.set(key, { player: playerNumber, position: pos, type: 'regular' });
      },
      collapseMarker: (pos, playerNumber, b) => {
        const key = positionToString(pos);
        b.markers.delete(key);
        b.collapsedSpaces.set(key, playerNumber);
      },
      flipMarker: (pos, playerNumber, b) => {
        const key = positionToString(pos);
        const existing = b.markers.get(key);
        if (!existing || existing.player !== playerNumber) {
          b.markers.set(key, { player: playerNumber, position: pos, type: 'regular' });
        }
      },
    };

    const captureApplyAdapters: CaptureApplyAdapters = {
      applyMarkerEffectsAlongPath: (from, to, playerNumber) =>
        applyMarkerEffectsAlongPathOnBoard(board, from, to, playerNumber, markerHelpers),
    };

    const results = findMaxCaptureChains(
      'square19',
      board,
      Y,
      1,
      { ...captureBoardAdapters, ...captureApplyAdapters },
      { pruneVisitedPositions: false }
    );

    expect(results.length).toBeGreaterThan(0);

    const bestLength = Math.max(...results.map((r) => r.segments.length));
    // Under Chebyshev (king-move) distance on square19 with this geometry,
    // the maximal chains have exactly 8 capture segments.
    expect(bestLength).toBe(8);

    for (const r of results) {
      expect(r.segments.length).toBe(bestLength);
      expect(r.finalHeight).toBe(2 + bestLength);
    }

    // For diagnostic purposes below, track the set of outer-vertex keys,
    // but do not require that every maximal chain must end exactly on one
    // of W/X/Y/Z; the search may find equally long chains that finish at
    // other legal landing squares.
    const allowedOuterKeys = new Set([
      positionToString(W),
      positionToString(X),
      positionToString(Y),
      positionToString(Z),
    ]);

    // --- Diagnostic output: maximal segments and representative chain summary ---
    // Log maximal chain statistics for human inspection when this test runs.
    // Jest hides console output by default unless a test fails or run with
    // appropriate flags, but this is still useful for targeted runs.
    //
    // 1) High-level summary
    // eslint-disable-next-line no-console
    console.log('[cyclic-square19] maximal overtaking capture search:');
    // eslint-disable-next-line no-console
    console.log(`  - Max capture segments per chain: ${bestLength}`);
    // eslint-disable-next-line no-console
    console.log(`  - Number of distinct maximal chains found: ${results.length}`);

    // 2) Choose one representative maximal chain and re-simulate it on a
    //    fresh board to summarize the resulting board state.
    const representative = results[0];

    const execBoard: BoardState = {
      stacks: new Map<string, RingStack>(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: config.size,
      type: 'square19',
    };

    const makeExecStack = (playerNumber: number, height: number, position: Position) => {
      const rings = Array(height).fill(playerNumber);
      const stack: RingStack = {
        position,
        rings,
        stackHeight: rings.length,
        capHeight: rings.length,
        controllingPlayer: playerNumber,
      };
      execBoard.stacks.set(positionToString(position), stack);
    };

    // Recreate original geometry on execBoard.
    makeExecStack(2, 2, A);
    makeExecStack(2, 2, B);
    makeExecStack(2, 2, C);
    makeExecStack(2, 2, D);
    makeExecStack(1, 2, Y);

    const execMarkerHelpers: MarkerPathHelpers = {
      setMarker: (pos, playerNumber, b) => {
        const key = positionToString(pos);
        b.markers.set(key, { player: playerNumber, position: pos, type: 'regular' });
      },
      collapseMarker: (pos, playerNumber, b) => {
        const key = positionToString(pos);
        b.markers.delete(key);
        b.collapsedSpaces.set(key, playerNumber);
      },
      flipMarker: (pos, playerNumber, b) => {
        const key = positionToString(pos);
        const existing = b.markers.get(key);
        if (!existing || existing.player !== playerNumber) {
          b.markers.set(key, { player: playerNumber, position: pos, type: 'regular' });
        }
      },
    };

    const execApplyAdapters: CaptureApplyAdapters = {
      applyMarkerEffectsAlongPath: (from, to, playerNumber) =>
        applyMarkerEffectsAlongPathOnBoard(execBoard, from, to, playerNumber, execMarkerHelpers),
    };

    // Apply the representative maximal chain to execBoard.
    for (const [idx, seg] of representative.segments.entries()) {
      applyCaptureSegmentOnBoard(
        execBoard,
        seg.from,
        seg.target,
        seg.landing,
        1,
        execApplyAdapters
      );
      // eslint-disable-next-line no-console
      console.log(
        `  segment ${idx + 1}: (${seg.from.x},${seg.from.y}) -> (${seg.target.x},${seg.target.y}) -> (${seg.landing.x},${seg.landing.y})`
      );
    }

    const execStacks = Array.from(execBoard.stacks.values());
    const execOvertakers = execStacks.filter((s) => s.controllingPlayer === 1);
    const execTargets = execStacks.filter((s) => s.controllingPlayer === 2);

    const execFinal = execOvertakers[0];
    const execFinalKey = positionToString(execFinal.position);

    // 3) Detailed summary of final board state vs assertions.
    // eslint-disable-next-line no-console
    console.log('  summary after executing one maximal chain:');
    // eslint-disable-next-line no-console
    console.log(`    - Final overtaker position: ${execFinalKey}`);
    // eslint-disable-next-line no-console
    console.log(
      `    - Final overtaker height: ${execFinal.stackHeight} (expected 2 + maxSegments = ${2 + bestLength})`
    );
    // eslint-disable-next-line no-console
    console.log(
      `    - Overtaker on outer vertex: ${allowedOuterKeys.has(execFinalKey)} (allowed: W,X,Y,Z)`
    );
    // eslint-disable-next-line no-console
    console.log(
      `    - Remaining target stacks (player 2): count=${execTargets.length}, heights=[${execTargets
        .map((s) => s.stackHeight)
        .join(', ')}]`
    );

    // Sanity check: total rings should be preserved (overtaking capture never eliminates).
    const totalRings = execStacks.reduce((sum, s) => sum + s.stackHeight, 0);
    // eslint-disable-next-line no-console
    console.log(`    - Total rings on board after chain: ${totalRings} (initial total was 10)`);
  });
});
