import fc from 'fast-check';

import { Territory, positionToString } from '../../src/shared/types/game';
import {
  getProcessableTerritoryRegions,
  applyTerritoryRegion,
  type TerritoryProcessingContext,
} from '../../src/shared/engine/territoryProcessing';
import { getBorderMarkerPositionsForRegion } from '../../src/shared/engine/territoryBorders';
import { createTestBoard, addStack, addMarker, pos } from '../utils/fixtures';

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

