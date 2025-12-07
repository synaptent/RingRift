import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardState,
  BOARD_CONFIGS,
  Position,
  RingStack,
  Player,
  TimeControl,
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
 * Cyclic capture scenario on a hexagonal board.
 *
 * We construct an equilateral triangle in cube coordinates using three
 * hex directions that sum to zero:
 *   d1 = ( 1, -1,  0)
 *   d2 = ( 0,  1, -1)
 *   d3 = (-1,  0,  1)
 * with d1 + d2 + d3 = (0,0,0).
 *
 * Let r = 4. The outer triangle vertices (o1, o2, o3) are:
 *   o1 = (-r,  r,  0)
 *   o2 = ( r, -r,  0)
 *   o3 = ( r,  r, -2r)
 *
 * Each side has hex distance 2r, and the three sides follow distinct
 * cube-coordinate axes (constant z, constant x, constant y).
 *
 * The midpoints of these sides are:
 *   m12 = (0,  0,  0)          // midpoint between o1 and o2 (z = 0 axis)
 *   m23 = ( r,  0, -r)         // midpoint between o2 and o3 (x = r axis)
 *   m31 = (0,  r, -r)         // midpoint between o3 and o1 (y = r axis)
 *
 * We place three target stacks for Player 2 at these midpoints (A, B, C),
 * forming a smaller equilateral triangle of side length 4 in hex distance
 * centred around the origin. We place the overtaking stack for Player 1 at
 * one outer vertex (o1). From o1 there is a legal overtaking segment over A
 * (the origin) to o2; similarly, from each outer vertex there are capture
 * rays over the midpoints to the next vertex, enabling cyclic patterns
 * around the triangle.
 *
 * As with the square19 cyclic scenario, we:
 *   - Use a sandbox search to enumerate all legal overtaking chains for
 *     this fixed geometry on the hex board.
 *   - Identify maximal chains by segment count.
 *   - Apply one representative maximal chain to a fresh board and log a
 *     summary of the resulting state.
 *
 * A parallel GameEngine-based scenario test can be added later if desired;
 * for now this file focuses on the geometry and maximal-chain properties
 * via the sandbox helpers.
 */

describe('GameEngine cyclic capture scenarios (hexagonal; FAQ 15.3.x)', () => {
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
      ringsInHand: 36,
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
      ringsInHand: 36,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  /**
   * Resolve any active capture chain for the current player by repeatedly
   * selecting a continue_capture_segment move from GameEngine.getValidMoves
   * while the game remains in the 'chain_capture' phase. This mirrors the
   * unified Move-based chain-capture model used in other scenario suites.
   */
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

  test('FAQ_15_3_x_hex_cyclic_chain_capture_around_inner_triangle', async () => {
    const engine = new GameEngine('cyclic-hex', 'hexagonal', players, timeControl, false) as any;
    const gameState = engine.gameState as any;
    const boardManager = engine.boardManager as any;

    // Force capture phase and correct current player so RuleEngine will
    // accept an overtaking_capture move.
    gameState.currentPhase = 'capture';
    gameState.currentPlayer = 1;

    const r = 4;

    // Outer vertices (Player 1 overtaker path).
    const O1: Position = { x: -r, y: r, z: 0 };
    const O2: Position = { x: r, y: -r, z: 0 };
    const O3: Position = { x: r, y: r, z: -2 * r };

    // Midpoints (Player 2 targets) on each side.
    const A: Position = { x: 0, y: 0, z: 0 };
    const B: Position = { x: r, y: 0, z: -r };
    const C: Position = { x: 0, y: r, z: -r };

    const makeStackEngine = (playerNumber: number, height: number, position: Position): void => {
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

    // Place inner target stacks (Player 2), cap=2, height=2.
    makeStackEngine(2, 2, A);
    makeStackEngine(2, 2, B);
    makeStackEngine(2, 2, C);

    // Place overtaking stack (Player 1) at O1, cap=2, height=2.
    makeStackEngine(1, 2, O1);

    // Perform the initial capture segment explicitly along one outer edge:
    // O1 -> over A -> O2.
    const result = await engine.makeMove({
      player: 1,
      type: 'overtaking_capture',
      from: O1,
      captureTarget: A,
      to: O2,
    } as any);

    expect(result.success).toBe(true);

    // After the initial capture, explicitly resolve any mandatory chain
    // continuations via the unified chain_capture phase so that the final
    // board state reflects all legal segments before we make assertions
    // about stacks and internal chain state.
    await resolveChainIfPresent(engine as GameEngine);

    const board = gameState.board as any;
    const stacks: Map<string, RingStack> = board.stacks;

    const allStacks = Array.from(stacks.values());
    const overtakerStacks = allStacks.filter((s) => s.controllingPlayer === 1);
    const targetStacks = allStacks.filter((s) => s.controllingPlayer === 2);

    // There should be exactly one overtaker stack remaining somewhere on the board.
    expect(overtakerStacks.length).toBe(1);

    const overtakerFinal = overtakerStacks[0];

    // Overtaker started with height 2. Total rings in the system are:
    //   - Overtaker: 2
    //   - Three targets: 3 * 2 = 6
    //   => 8 rings total
    // Each overtaking capture moves one ring from a target to the
    // overtaker (no rings are eliminated). The engine may or may not
    // achieve the absolute maximal number of segments for this geometry,
    // but it must:
    //   - Increase the overtaker's height above its starting value, and
    //   - Preserve the total number of rings on the board.
    expect(overtakerFinal.stackHeight).toBeGreaterThan(2);

    const totalRingsAfter = allStacks.reduce((sum, s) => sum + s.stackHeight, 0);
    expect(totalRingsAfter).toBe(8);

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

    // At least one target stack must have been partially or fully depleted.
    const totalTargetRings = targetStacks.reduce((sum, s) => sum + s.stackHeight, 0);
    expect(totalTargetRings).toBeLessThan(6);
  });

  /**
   * Sandbox-based maximal chain search for the fixed hex triangle geometry.
   */
  test('FAQ_15_3_x_hex_cyclic_chain_maximal_search_sandbox', () => {
    const config = BOARD_CONFIGS.hexagonal;
    const radius = config.size - 1; // cube radius (size 13 -> radius 12)

    const board: BoardState = {
      stacks: new Map<string, RingStack>(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: config.size,
      type: 'hexagonal',
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

    const r = 4; // outer triangle scale

    // Outer vertices (Player 1 overtaker path).
    const O1: Position = { x: -r, y: r, z: 0 };
    const O2: Position = { x: r, y: -r, z: 0 };
    const O3: Position = { x: r, y: r, z: -2 * r };

    // Midpoints (Player 2 targets) on each side.
    const A: Position = { x: 0, y: 0, z: 0 }; // midpoint between O1 and O2
    const B: Position = { x: r, y: 0, z: -r }; // midpoint between O2 and O3
    const C: Position = { x: 0, y: r, z: -r }; // midpoint between O3 and O1

    // Place inner target stacks (Player 2), cap=2, height=2.
    makeBoardStack(2, 2, A);
    makeBoardStack(2, 2, B);
    makeBoardStack(2, 2, C);

    // Place overtaking stack (Player 1) at O1, cap=2, height=2.
    makeBoardStack(1, 2, O1);

    const captureBoardAdapters: CaptureBoardAdapters = {
      isValidPosition: (pos: Position) => {
        if (pos.z === undefined) return false;
        const sum = pos.x + pos.y + pos.z;
        if (sum !== 0) return false;
        const maxAbs = Math.max(Math.abs(pos.x), Math.abs(pos.y), Math.abs(pos.z));
        return maxAbs <= radius;
      },
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
      'hexagonal',
      board,
      O1,
      1,
      { ...captureBoardAdapters, ...captureApplyAdapters },
      { pruneVisitedPositions: false }
    );

    expect(results.length).toBeGreaterThan(0);

    const bestLength = Math.max(...results.map((r) => r.segments.length));
    // For this hex triangular geometry we expect at least 6 segments in a
    // maximal chain (three targets of height 2 plus possible recaptures).
    expect(bestLength).toBeGreaterThanOrEqual(6);

    for (const r of results) {
      expect(r.segments.length).toBe(bestLength);
      expect(r.finalHeight).toBe(2 + bestLength);
    }

    // --- Diagnostic output: maximal segments and representative chain summary ---
    // eslint-disable-next-line no-console
    console.log('[hex-triangle] maximal overtaking capture search:');
    // eslint-disable-next-line no-console
    console.log(`  - Max capture segments per chain: ${bestLength}`);
    // eslint-disable-next-line no-console
    console.log(`  - Number of distinct maximal chains found: ${results.length}`);

    const representative = results[0];

    const execBoard: BoardState = {
      stacks: new Map<string, RingStack>(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: config.size,
      type: 'hexagonal',
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
    makeExecStack(1, 2, O1);

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
        `  segment ${idx + 1}: (${seg.from.x},${seg.from.y},${seg.from.z}) -> (${seg.target.x},${seg.target.y},${seg.target.z}) -> (${seg.landing.x},${seg.landing.y},${seg.landing.z})`
      );
    }

    const execStacks = Array.from(execBoard.stacks.values());
    const execOvertakers = execStacks.filter((s) => s.controllingPlayer === 1);
    const execTargets = execStacks.filter((s) => s.controllingPlayer === 2);

    const execFinal = execOvertakers[0];

    // 3) Detailed summary of final board state vs assertions.
    // eslint-disable-next-line no-console
    console.log('  summary after executing one maximal hex chain:');
    // eslint-disable-next-line no-console
    console.log(`    - Final overtaker position: ${positionToString(execFinal.position)}`);
    // eslint-disable-next-line no-console
    console.log(
      `    - Final overtaker height: ${execFinal.stackHeight} (expected 2 + maxSegments = ${
        2 + bestLength
      })`
    );
    // eslint-disable-next-line no-console
    console.log(
      `    - Remaining target stacks (player 2): count=${execTargets.length}, heights=[${execTargets
        .map((s) => s.stackHeight)
        .join(', ')}]`
    );

    const totalRings = execStacks.reduce((sum, s) => sum + s.stackHeight, 0);
    // eslint-disable-next-line no-console
    console.log(`    - Total rings on board after chain: ${totalRings}`);
  });
});
