import {
  BoardState,
  BOARD_CONFIGS,
  Position,
  RingStack,
  TimeControl,
  positionToString,
} from '../../src/shared/types/game';
import { findMaxCaptureChains } from '../../src/client/sandbox/sandboxCaptureSearch';
import {
  CaptureBoardAdapters,
  CaptureApplyAdapters,
  applyCaptureSegmentOnBoard,
} from '../../src/client/sandbox/sandboxCaptures';
import { applyMarkerEffectsAlongPathOnBoard, MarkerPathHelpers } from '../../src/client/sandbox/sandboxMovement';

/**
 * Diagnostic sandbox test: same hex triangular geometry as the main
 * cyclicCapture hex scenario, but with target stacks of height 3 and
 * an overtaker of height 3.
 *
 * Geometry is the r=4 outer triangle with inner midpoints:
 *   O1 = (-4,  4,  0)
 *   O2 = ( 4, -4,  0)
 *   O3 = ( 4,  4, -8)
 *   A  = ( 0,  0,  0)
 *   B  = ( 4,  0, -4)
 *   C  = ( 0,  4, -4)
 *
 * This is directly parallel to the height-2 sandbox scenario in
 * GameEngine.cyclicCapture.hex.scenarios.test.ts, differing only in
 * stack heights/capHeights. This lets us check whether increasing the
 * heights to 3 still yields non-zero maximal overtaking chains under
 * the compact rules implementation.
 */

describe('Hex cyclic capture (height 3 targets, r=4 triangle) sandbox diagnostics', () => {
  const timeControl: TimeControl = {
    initialTime: 600,
    increment: 0,
    type: 'blitz',
  };

  test('sandbox search with height-3 targets on r=4 triangle reports maximal chain length', () => {
    const config = BOARD_CONFIGS.hexagonal;
    const radius = config.size - 1; // cube radius (size 11 -> radius 10)

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

    // Outer vertices (Player 1 overtaker path), identical to height-2 scenario.
    const O1: Position = { x: -r, y: r, z: 0 };
    const O2: Position = { x: r, y: -r, z: 0 };
    const O3: Position = { x: r, y: r, z: -2 * r };

    // Midpoints (Player 2 targets) on each side.
    const A: Position = { x: 0, y: 0, z: 0 }; // midpoint between O1 and O2
    const B: Position = { x: r, y: 0, z: -r }; // midpoint between O2 and O3
    const C: Position = { x: 0, y: r, z: -r }; // midpoint between O3 and O1

    // Place inner target stacks (Player 2), cap=3, height=3.
    makeBoardStack(2, 3, A);
    makeBoardStack(2, 3, B);
    makeBoardStack(2, 3, C);

    // Place overtaking stack (Player 1) at O1, cap=3, height=3.
    makeBoardStack(1, 3, O1);

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
      {
        pruneVisitedPositions: true,
        maxDepth: 16,
      },
    );

    // At minimum, raising target/overtaker heights from 2 to 3 on the same
    // geometry should still admit at least one legal overtaking chain.
    expect(results.length).toBeGreaterThan(0);

    // All results are maximal-length chains by construction; take the
    // segment count from the first result instead of spreading a very
    // large array into Math.max (which can overflow the JS call stack).
    const bestLength = results[0].segments.length;

    // eslint-disable-next-line no-console
    console.log('[hex-triangle r=4, height=3] maximal overtaking capture search:');
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
    makeExecStack(2, 3, A);
    makeExecStack(2, 3, B);
    makeExecStack(2, 3, C);
    makeExecStack(1, 3, O1);

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

    for (const [idx, seg] of representative.segments.entries()) {
      applyCaptureSegmentOnBoard(
        execBoard,
        seg.from,
        seg.target,
        seg.landing,
        1,
        execApplyAdapters,
      );
      // eslint-disable-next-line no-console
      console.log(
        `  segment ${idx + 1}: (${seg.from.x},${seg.from.y},${seg.from.z}) -> (${seg.target.x},${seg.target.y},${seg.target.z}) -> (${seg.landing.x},${seg.landing.y},${seg.landing.z})`,
      );
    }

    const execStacks = Array.from(execBoard.stacks.values());
    const execOvertakers = execStacks.filter((s) => s.controllingPlayer === 1);
    const execTargets = execStacks.filter((s) => s.controllingPlayer === 2);

    const execFinal = execOvertakers[0];

    // eslint-disable-next-line no-console
    console.log('  summary after executing one maximal hex chain (height=3, r=4):');
    // eslint-disable-next-line no-console
    console.log(
      `    - Final overtaker position: ${positionToString(execFinal.position)}`,
    );
    // eslint-disable-next-line no-console
    console.log(
      `    - Final overtaker height: ${execFinal.stackHeight} (expected 3 + maxSegments = ${
        3 + bestLength
      })`,
    );
    // eslint-disable-next-line no-console
    console.log(
      `    - Remaining target stacks (player 2): count=${execTargets.length}, heights=[${execTargets
        .map((s) => s.stackHeight)
        .join(', ')}]`,
    );

    const totalRings = execStacks.reduce((sum, s) => sum + s.stackHeight, 0);
    // eslint-disable-next-line no-console
    console.log(`    - Total rings on board after chain: ${totalRings}`);
  });
});
