import { computeProgressSnapshot, summarizeBoard } from '../../src/shared/engine/core';
import { BoardType, GameState, Player, Position } from '../../src/shared/types/game';
import {
  createTestBoard,
  createTestGameState,
  createTestPlayer,
  addStack,
  addMarker,
  addCollapsedSpace,
} from '../utils/fixtures';

/**
 * Regression harness for the known seed-17 AI-vs-AI game on square8.
 *
 * This test reconstructs the **final** board position for move 52 from the
 * logged Sandbox_vs_Backend.seed17.traceDebug context and asserts that the
 * shared S-invariant helper (computeProgressSnapshot) is consistent with the
 * geometric board summary (summarizeBoard) and the per-player elimination
 * bookkeeping encoded in the state hash.
 *
 * Logged context (from logs/seed17_trace_debug2.log):
 *   - boardType: square8, 2 players
 *   - Final entry (moveNumber 52, actor 2, type move_stack 0,0 → 0,7)
 *   - original/sandbox progressAfter:
 *       markers = 2, collapsed = 45, eliminated = 27, S = 74
 *   - stateHashAfter players meta:
 *       "1:2:2:0|2:0:25:38"
 *       → P1: ringsInHand=2, eliminatedRings=2, territorySpaces=0
 *       → P2: ringsInHand=0, eliminatedRings=25, territorySpaces=38
 *   - boardAfterSummary:
 *       stacks:
 *         "5,3:1:2:2",
 *         "5,6:1:1:1",
 *         "6,4:1:4:3"
 *       markers:
 *         "7,4:1",
 *         "7,6:2"
 *       collapsedSpaces: 45 entries (see below).
 */

describe('SInvariant – seed17 final board regression (square8 / 2p)', () => {
  const boardType: BoardType = 'square8';

  function parsePosition(key: string): Position {
    const [xStr, yStr] = key.split(',');
    return { x: parseInt(xStr, 10), y: parseInt(yStr, 10) };
  }

  it('seed17_final_board_progress_snapshot_matches_geometry', () => {
    const board = createTestBoard(boardType);

    // --- Reconstruct stacks from boardAfterSummary.stacks ---
    const stackSpecs = ['5,3:1:2:2', '5,6:1:1:1', '6,4:1:4:3'];

    for (const spec of stackSpecs) {
      const [posKey, playerStr, heightStr] = spec.split(':');
      const pos = parsePosition(posKey);
      const player = parseInt(playerStr, 10);
      const height = parseInt(heightStr, 10);

      // NOTE: addStack derives capHeight from total height, so we do not
      // attempt to exactly reproduce the logged capHeight here. S-invariant
      // depends only on markers, collapsed spaces, and eliminated rings, so
      // stack cap details are irrelevant for this regression.
      addStack(board, pos, player, height);
    }

    // --- Reconstruct markers from boardAfterSummary.markers ---
    const markerSpecs = ['7,4:1', '7,6:2'];
    for (const spec of markerSpecs) {
      const [posKey, playerStr] = spec.split(':');
      const pos = parsePosition(posKey);
      const player = parseInt(playerStr, 10);
      addMarker(board, pos, player);
    }

    // --- Reconstruct collapsed spaces from boardAfterSummary.collapsedSpaces ---
    const collapsedSpecs = [
      '0,0:2',
      '0,1:2',
      '0,2:2',
      '0,3:2',
      '0,4:2',
      '0,5:2',
      '0,6:2',
      '0,7:2',
      '1,0:2',
      '1,1:2',
      '1,2:2',
      '1,3:2',
      '1,4:2',
      '1,5:2',
      '1,6:2',
      '1,7:2',
      '2,0:2',
      '2,1:2',
      '2,2:2',
      '2,3:2',
      '2,4:2',
      '2,5:2',
      '2,6:2',
      '2,7:2',
      '3,0:2',
      '3,1:2',
      '3,2:2',
      '3,3:2',
      '3,4:1',
      '3,5:1',
      '3,6:2',
      '3,7:2',
      '4,0:2',
      '4,1:2',
      '4,2:2',
      '4,3:2',
      '4,6:2',
      '4,7:2',
      '5,0:2',
      '5,1:2',
      '5,2:2',
      '5,5:2',
      '6,0:2',
      '7,0:2',
      '7,1:2',
    ];

    for (const spec of collapsedSpecs) {
      const [posKey, ownerStr] = spec.split(':');
      const pos = parsePosition(posKey);
      const owner = parseInt(ownerStr, 10);
      addCollapsedSpace(board, pos, owner);
    }

    // Sanity: geometry matches the logged counts.
    const summary = summarizeBoard(board);
    expect(summary.markers.length).toBe(2);
    expect(summary.collapsedSpaces.length).toBe(45);

    // --- Reconstruct players and global elimination metadata from stateHashAfter ---
    const players: Player[] = [
      createTestPlayer(1, {
        ringsInHand: 2,
        eliminatedRings: 2,
        territorySpaces: 0,
      }),
      createTestPlayer(2, {
        ringsInHand: 0,
        eliminatedRings: 25,
        territorySpaces: 38,
      }),
    ];

    const state: GameState = createTestGameState({
      boardType,
      board,
      players,
      totalRingsEliminated: 27,
    });

    // Also mirror the board-level eliminatedRings summary implied by the
    // per-player eliminatedRings so computeProgressSnapshot's fallback path
    // would agree if totalRingsEliminated were absent.
    state.board.eliminatedRings = { 1: 2, 2: 25 };

    const snapshot = computeProgressSnapshot(state);

    // Core S-invariant expectations from the canonical final board as
    // logged in the seed17 trace:
    //   markers = 2
    //   collapsed = 45
    //   eliminated = 27
    //   S = 2 + 45 + 27 = 74
    expect(snapshot.markers).toBe(summary.markers.length);
    expect(snapshot.collapsed).toBe(summary.collapsedSpaces.length);
    expect(snapshot.eliminated).toBe(27);
    expect(snapshot.S).toBe(2 + 45 + 27);
  });
});
