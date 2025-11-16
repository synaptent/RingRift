import { BoardManager } from '../../src/server/game/BoardManager';
import { createTestBoard } from '../utils/fixtures';

/**
 * Hex territory disconnection tests (scaffold).
 *
 * TODO:
 *   - Mirror a concrete hex territory scenario from the Rust engine once
 *     hex BoardManager territory detection is finalized to Rust parity.
 *   - Use the Rust reference implementation and tests as the source of truth:
 *       • RingRift Rust/ringrift/src/rules/territory.rs
 *       • Any hex-specific territory scenarios added under that module
 *       • Supporting design docs such as:
 *           - RingRift Rust/hex_fix_and_gui_plan.md
 *           - RingRift Rust/ringrift-territory-fix-plan.md
 *
 * Sketch of the eventual test:
 *   - const manager = new BoardManager('hexagonal');
 *   - const board = createTestBoard('hexagonal');
 *   - Populate stacks/markers to match the chosen Rust hex scenario.
 *   - Call manager.findDisconnectedRegions(board, movingPlayer).
 *   - Assert on:
 *       • number of regions
 *       • region sizes
 *       • controlling/boundary colors
 *       • which regions are fully disconnected/eligible
 */

describe('BoardManager territory disconnection (hexagonal)', () => {
  it.todo(
    'mirrors a concrete hex territory scenario from the Rust engine (see RingRift Rust/ringrift/src/rules/territory.rs)'
  );
});
