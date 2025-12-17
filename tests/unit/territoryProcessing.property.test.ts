import fc from 'fast-check';

import {
  Territory,
  positionToString,
  type GameState,
  type Player,
  type Position,
} from '../../src/shared/types/game';
import {
  getProcessableTerritoryRegions,
  applyTerritoryRegion,
  type TerritoryProcessingContext,
} from '../../src/shared/engine/territoryProcessing';
import { getBorderMarkerPositionsForRegion } from '../../src/shared/engine/territoryBorders';
import {
  createTestBoard,
  createTestGameState,
  createTestPlayer,
  addStack,
  addMarker,
  pos,
} from '../utils/fixtures';
import {
  enumerateProcessTerritoryRegionMoves,
  applyProcessTerritoryRegionDecision,
  enumerateTerritoryEliminationMoves,
  applyEliminateRingsFromStackDecision,
} from '../../src/shared/engine/territoryDecisionHelpers';
import { calculateCapHeight } from '../../src/shared/engine/core';

/**
 * Property-based harness for territory-processing invariants over simple 2x2
 * disconnected regions on square8.
 *
 * This test generalises the concrete Q23-style scenario in
 * `territoryProcessing.shared.test.ts` by varying the 2x2 region's position
 * within the interior of the board and asserting that the core invariants
 * (internal eliminations, border collapse, and territory credit) continue to
 * hold.
 *
 * It is intentionally lightweight (small board, bounded runs) so it can run
 * as part of the normal Jest suite without requiring a separate diagnostics
 * profile.
 */
describe('territoryProcessing.property - 2x2 region invariants on square8', () => {
  const movingPlayer = 1;
  const victimPlayer = 2;

  it('preserves collapse/elimination invariants for any interior 2x2 region', () => {
    fc.assert(
      fc.property(
        // Choose the top-left corner of a 2x2 region such that the 1-ring
        // border around it stays within the 0..7 board bounds.
        fc.integer({ min: 2, max: 4 }),
        fc.integer({ min: 2, max: 4 }),
        (x0, y0) => {
          const board = createTestBoard('square8');

          const regionSpaces = [pos(x0, y0), pos(x0 + 1, y0), pos(x0, y0 + 1), pos(x0 + 1, y0 + 1)];

          const internalStackHeight = 2;
          for (const p of regionSpaces) {
            addStack(board, p, victimPlayer, internalStackHeight);
          }
          const expectedInternalRings = regionSpaces.length * internalStackHeight;

          // Place a rectangular ring of markers one cell around the region,
          // mirroring the concrete scenario test but with a variable origin.
          const borderCoords: Array<[number, number]> = [];
          for (let x = x0 - 1; x <= x0 + 2; x++) {
            borderCoords.push([x, y0 - 1]);
            borderCoords.push([x, y0 + 2]);
          }
          for (let y = y0; y <= y0 + 1; y++) {
            borderCoords.push([x0 - 1, y]);
            borderCoords.push([x0 + 2, y]);
          }
          borderCoords.forEach(([x, y]) => {
            if (x >= 0 && x < board.size && y >= 0 && y < board.size) {
              addMarker(board, pos(x, y), movingPlayer);
            }
          });

          // Outside stack for the moving player to satisfy self-elimination
          // prerequisite; exact location is irrelevant as long as it is
          // outside the region.
          addStack(board, pos(0, 0), movingPlayer, 3);

          const expectedBorder = getBorderMarkerPositionsForRegion(board, regionSpaces, {
            mode: 'rust_aligned',
          });
          expect(expectedBorder.length).toBeGreaterThan(0);

          const ctx: TerritoryProcessingContext = { player: movingPlayer };
          const regions: Territory[] = getProcessableTerritoryRegions(board, ctx);
          expect(regions).toHaveLength(1);

          const region = regions[0];

          const outcome = applyTerritoryRegion(board, region, ctx);

          // Territory gain equals region + border size and all such spaces
          // are collapsed to the moving player with no remaining stacks.
          const territoryGain = outcome.territoryGainedByPlayer[movingPlayer] ?? 0;
          expect(territoryGain).toBe(regionSpaces.length + expectedBorder.length);

          const regionKeys = new Set(regionSpaces.map((p) => positionToString(p)));
          const borderKeys = new Set(expectedBorder.map((p) => positionToString(p)));

          for (const key of regionKeys) {
            expect(outcome.board.stacks.has(key)).toBe(false);
            expect(outcome.board.collapsedSpaces.get(key)).toBe(movingPlayer);
          }
          for (const key of borderKeys) {
            expect(outcome.board.stacks.has(key)).toBe(false);
            expect(outcome.board.collapsedSpaces.get(key)).toBe(movingPlayer);
          }

          const internalElims = outcome.eliminatedRingsByPlayer[movingPlayer] ?? 0;
          expect(internalElims).toBe(expectedInternalRings);
          expect(outcome.board.eliminatedRings[movingPlayer]).toBe(expectedInternalRings);
        }
      ),
      {
        numRuns: 32,
      }
    );
  });
});

/**
 * Additional property-based checks for the shared territory decision helpers:
 *
 * - enumerateProcessTerritoryRegionMoves must surface exactly the processable
 *   regions discovered by getProcessableTerritoryRegions for random Q23-style
 *   2x2 regions on square8.
 * - Applying the surfaced move via applyProcessTerritoryRegionDecision must
 *   match the lower-level applyTerritoryRegion outcome and project its deltas
 *   into GameState-level aggregates.
 * - When a single region has been processed and at least one eligible stack
 *   remains outside the region, enumerateTerritoryEliminationMoves must surface
 *   exactly those stacks and applyEliminateRingsFromStackDecision must update
 *   elimination counters monotonically.
 */
describe('territoryDecisionHelpers.property – decision enumeration and self-elimination (square8)', () => {
  const boardType = 'square8';
  const movingPlayer: Player['playerNumber'] = 1;
  const victimPlayer: Player['playerNumber'] = 2;

  function buildRandomQ23LikeState(
    x0: number,
    y0: number,
    internalHeights: [number, number, number, number],
    outsideHeight: number
  ): {
    state: GameState;
    regionSpaces: Position[];
    expectedInternalRings: number;
    expectedBorder: Position[];
  } {
    const board = createTestBoard(boardType);

    const regionSpaces: Position[] = [
      pos(x0, y0),
      pos(x0 + 1, y0),
      pos(x0, y0 + 1),
      pos(x0 + 1, y0 + 1),
    ];

    let expectedInternalRings = 0;
    regionSpaces.forEach((p, idx) => {
      const h = internalHeights[idx];
      addStack(board, p, victimPlayer, h);
      expectedInternalRings += h;
    });

    const borderCoords: Array<[number, number]> = [];
    for (let x = x0 - 1; x <= x0 + 2; x++) {
      borderCoords.push([x, y0 - 1]);
      borderCoords.push([x, y0 + 2]);
    }
    for (let y = y0; y <= y0 + 1; y++) {
      borderCoords.push([x0 - 1, y]);
      borderCoords.push([x0 + 2, y]);
    }
    borderCoords.forEach(([x, y]) => {
      if (x >= 0 && x < board.size && y >= 0 && y < board.size) {
        addMarker(board, pos(x, y), movingPlayer);
      }
    });

    // Outside stack for the moving player to satisfy the self-elimination
    // prerequisite; we randomise its height to vary the self-elim cost.
    addStack(board, pos(0, 0), movingPlayer, outsideHeight);

    const expectedBorder = getBorderMarkerPositionsForRegion(board, regionSpaces, {
      mode: 'rust_aligned',
    });

    const state = createTestGameState({
      boardType,
      board,
      currentPlayer: movingPlayer,
      currentPhase: 'territory_processing',
      players: [
        createTestPlayer(1, { ringsInHand: 0, eliminatedRings: 0, territorySpaces: 0 }),
        createTestPlayer(2, { ringsInHand: 0, eliminatedRings: 0, territorySpaces: 0 }),
      ],
    }) as unknown as GameState;

    return { state, regionSpaces, expectedInternalRings, expectedBorder };
  }

  it('enumerateProcessTerritoryRegionMoves matches getProcessableTerritoryRegions and applyProcessTerritoryRegionDecision preserves invariants', () => {
    fc.assert(
      fc.property(
        // Same region placement constraints as the board-level property: keep a
        // one-cell margin to ensure the marker border stays within 0..7.
        fc.integer({ min: 2, max: 4 }),
        fc.integer({ min: 2, max: 4 }),
        fc.tuple(
          fc.integer({ min: 1, max: 3 }),
          fc.integer({ min: 1, max: 3 }),
          fc.integer({ min: 1, max: 3 }),
          fc.integer({ min: 1, max: 3 })
        ),
        // Any controlled stack is eligible per RR-CANON-R022/R145 (including height-1).
        // Using min: 2 here to ensure a multi-ring cap for property testing variety.
        fc.integer({ min: 2, max: 4 }),
        (x0, y0, heights, outsideHeight) => {
          const { state, regionSpaces, expectedInternalRings, expectedBorder } =
            buildRandomQ23LikeState(
              x0,
              y0,
              heights as [number, number, number, number],
              outsideHeight
            );

          const boardBefore = state.board;
          const ctx: TerritoryProcessingContext = { player: movingPlayer };

          const processable = getProcessableTerritoryRegions(boardBefore, ctx);
          expect(processable.length).toBe(1);

          const processableKey = new Set(processable[0].spaces.map((p) => positionToString(p)));

          const beforeCollapsed = boardBefore.collapsedSpaces.size;
          const beforeBoardElims = { ...boardBefore.eliminatedRings };
          const p1Before = state.players.find((p) => p.playerNumber === movingPlayer)!;
          const totalBefore = state.totalRingsEliminated;

          const moves = enumerateProcessTerritoryRegionMoves(state, movingPlayer);
          expect(moves.length).toBe(1);

          const move = moves[0];
          expect(move.type).toBe('choose_territory_option');
          expect(move.player).toBe(movingPlayer);
          expect(move.disconnectedRegions && move.disconnectedRegions.length).toBeGreaterThan(0);

          const fromMove = move.disconnectedRegions![0];
          const fromMoveKey = new Set(fromMove.spaces.map((p) => positionToString(p)));
          expect(fromMoveKey).toEqual(processableKey);

          // Sanity: the self-elimination prerequisite must hold – there is at
          // least one stack for movingPlayer outside the region.
          const regionKeySet = new Set(regionSpaces.map((p) => positionToString(p)));
          const outsideStacks = Array.from(boardBefore.stacks.entries()).filter(
            ([key, stack]) => stack.controllingPlayer === movingPlayer && !regionKeySet.has(key)
          );
          expect(outsideStacks.length).toBeGreaterThan(0);

          const outcome = applyProcessTerritoryRegionDecision(state, move);
          expect(outcome.pendingSelfElimination).toBe(true);

          const next = outcome.nextState;
          const boardAfter = next.board;
          const p1After = next.players.find((p) => p.playerNumber === movingPlayer)!;

          const territoryGain = p1After.territorySpaces - p1Before.territorySpaces;
          expect(territoryGain).toBe(regionSpaces.length + expectedBorder.length);

          const boardElimsAfter = boardAfter.eliminatedRings[movingPlayer] ?? 0;
          const boardElimsBefore = beforeBoardElims[movingPlayer] ?? 0;
          const playerElimsDelta = p1After.eliminatedRings - p1Before.eliminatedRings;
          const totalDelta = next.totalRingsEliminated - totalBefore;

          expect(boardElimsAfter - boardElimsBefore).toBe(expectedInternalRings);
          expect(playerElimsDelta).toBe(expectedInternalRings);
          expect(totalDelta).toBe(expectedInternalRings);

          const regionKeySetAfter = new Set(regionSpaces.map((p) => positionToString(p)));
          const borderKeySetAfter = new Set(expectedBorder.map((p) => positionToString(p)));

          for (const key of regionKeySetAfter) {
            expect(boardAfter.stacks.has(key)).toBe(false);
            expect(boardAfter.collapsedSpaces.get(key)).toBe(movingPlayer);
          }
          for (const key of borderKeySetAfter) {
            expect(boardAfter.stacks.has(key)).toBe(false);
            expect(boardAfter.collapsedSpaces.get(key)).toBe(movingPlayer);
          }

          expect(boardAfter.collapsedSpaces.size).toBeGreaterThanOrEqual(beforeCollapsed);
        }
      ),
      { numRuns: 32 }
    );
  });

  it('after processing a region, enumerateTerritoryEliminationMoves surfaces exactly the eligible cap targets and applyEliminateRingsFromStackDecision updates elimination counters', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 2, max: 4 }),
        fc.integer({ min: 2, max: 4 }),
        fc.tuple(
          fc.integer({ min: 1, max: 3 }),
          fc.integer({ min: 1, max: 3 }),
          fc.integer({ min: 1, max: 3 }),
          fc.integer({ min: 1, max: 3 })
        ),
        // Randomise the outside-stack height to vary self-elimination cost.
        // Any controlled stack is eligible per RR-CANON-R022/R145 (including height-1).
        // Using min: 2 for property testing variety.
        fc.integer({ min: 2, max: 4 }),
        (x0, y0, heights, outsideHeight) => {
          const { state } = buildRandomQ23LikeState(
            x0,
            y0,
            heights as [number, number, number, number],
            outsideHeight
          );

          const regionMoves = enumerateProcessTerritoryRegionMoves(state, movingPlayer);
          expect(regionMoves.length).toBe(1);

          const regionMove = regionMoves[0];
          const { nextState } = applyProcessTerritoryRegionDecision(state, regionMove);
          const nextStateWithHistory: GameState = {
            ...nextState,
            moveHistory: [...nextState.moveHistory, regionMove],
          };

          const boardAfterRegion = nextStateWithHistory.board;

          // RR-CANON-R082: Only eligible cap targets are surfaced:
          // (1) Multicolor stacks (stackHeight > capHeight), OR
          // (2) Single-color stacks with height > 1
          const stacksForPlayer: Array<{ key: string; cap: number }> = [];
          for (const [key, stack] of boardAfterRegion.stacks.entries()) {
            if (stack.controllingPlayer !== movingPlayer) continue;
            const capHeight = calculateCapHeight(stack.rings);
            if (capHeight <= 0) continue;
            const isMulticolor = stack.stackHeight > capHeight;
            const isSingleColorTall = stack.stackHeight === capHeight && stack.stackHeight > 1;
            if (isMulticolor || isSingleColorTall) {
              stacksForPlayer.push({ key, cap: capHeight });
            }
          }

          const elimMoves = enumerateTerritoryEliminationMoves(nextStateWithHistory, movingPlayer);
          const moveTargets = new Set(
            elimMoves
              .map((m) => (m.to ? positionToString(m.to) : undefined))
              .filter((k): k is string => !!k)
          );
          const expectedTargets = new Set(stacksForPlayer.map((s) => s.key));

          expect(moveTargets).toEqual(expectedTargets);

          if (elimMoves.length === 0) {
            return;
          }

          const beforeBoardElims = { ...boardAfterRegion.eliminatedRings };
          const p1Before = nextState.players.find((p) => p.playerNumber === movingPlayer)!;
          const totalBefore = nextState.totalRingsEliminated;

          const chosen = elimMoves[0];
          const { nextState: afterElim } = applyEliminateRingsFromStackDecision(
            nextStateWithHistory,
            chosen
          );

          const boardAfterElim = afterElim.board;
          const p1After = afterElim.players.find((p) => p.playerNumber === movingPlayer)!;

          const boardDelta =
            (boardAfterElim.eliminatedRings[movingPlayer] ?? 0) -
            (beforeBoardElims[movingPlayer] ?? 0);
          const playerDelta = p1After.eliminatedRings - p1Before.eliminatedRings;
          const totalDelta = afterElim.totalRingsEliminated - totalBefore;

          expect(boardDelta).toBeGreaterThan(0);
          expect(playerDelta).toBe(boardDelta);
          expect(totalDelta).toBe(boardDelta);
        }
      ),
      { numRuns: 32 }
    );
  });
});
